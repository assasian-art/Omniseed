# JEV INTEGRATION — "Two Heads, One Brain"

**Status:** implemented, offline-verified, **untrained by default** (see §6).
**Scope:** C++ only. No new Python dependency is introduced into the decision
path, and nothing in this document is a runtime requirement.

---

## 1. The problem this solves

A language model produces *text*. A program needs *decisions*. Bridging the two
by generating text and parsing it back is the standard approach and it has four
known costs:

| Cost | Why it hurts here |
|---|---|
| Latency scales with `max_new_tokens` | one full RWKV forward per token |
| Output can be malformed | we already ship a `GrammarDecoder` to fight this |
| Cost scales with wording | more explanation = more tokens = more time |
| Failure mode is silent | a fluent sentence that parses is not a correct decision |

The alternative, and the one this document implements, is to read the decision
**directly off the model's internal state** without ever producing a token.

---

## 2. What the reference architecture actually does

TypeSafe's **Jev** (public early access 2026-09-15) is the clearest published
instance of this pattern. It is worth being precise about what it is, because
the marketing language and the mechanism are different things:

* **Framing.** Jev is marketed as the first *System One* model — fast, intuitive
  judgement, in Kahneman's sense — as opposed to the *System Two* deliberate
  reasoning of an autoregressive chat model.
* **Mechanism.** A **non-autoregressive parallel sampler**. The state is encoded
  **once**, and every question is projected into a decision space **in parallel**.
  There is no token loop, so adding questions barely moves the response time.
* **Primitives.** Three typed question shapes, which is the part worth copying:
  * **Choice** — pick one of a list (≤ 255 options) → `choice`,
    `probabilities`, `confidence`
  * **Score** — rate against a rubric (2–10 levels) → `score`,
    `probabilities`, `confidence`
  * **Noul** — is this statement true? → `P(yes) ∈ [0, 1]`
* **Calibration.** Training uses **RLCD** (Reinforcement Learning for Calibrated
  Decisions), which penalises *overconfident errors* asymmetrically, so an
  output tagged 80% is right about 80% of the time. This is the single most
  important property and the one hardest to reproduce.
* **Reported figures** (TypeSafe's own workflow benchmark): 76.0% task accuracy,
  ~0.4 s end-to-end (70–500 ms), $0.0001/call, 0.00% format-break rate.

The open-source reproductions converge on the same trick, and it is the trick
used here. `OpenJev`, for instance, takes frozen Qwen weights, **intercepts the
first-token logits, softmaxes over just the candidate option token IDs, and
skips the autoregressive loop entirely**. `mini-jev` does the same over a
restricted token-ID set. `Outlines`/`Guidance` reach the same place from the
other direction — constrain the sampler with an FSM so only the option set can
ever be emitted.

### Known limits of the pattern (stated up front, because they apply here too)

* **"Zero hallucination" means schema conformance, not correctness.** Jev's 0%
  is *zero format errors*. A model can still confidently pick the wrong option.
  The same is true of the head in this repository.
* **No chain of thought.** The output is a probability vector, not an argument.
  Anything that requires audit or explanation cannot live in System-1.
* **Long-horizon reasoning degrades.** These models are gut-checks, not
  planners. Multi-step or spatial reasoning is out of scope by construction.
* **The option set is a hard boundary.** 255 is the stated cap; beyond that you
  must decompose hierarchically.
* **Network RTT can dominate.** A 100 ms model behind a 300 ms round trip is a
  400 ms feature. This is precisely why OmniSeed does it *in-process*.

---

## 3. The design here: two heads over one hidden state

```
                  prompt tokens
                        |
                        v
        +-------------------------------------+
        |  RWKV-7 backbone — the one brain    |
        |  one forward pass -> hidden h[E]    |
        +-------------------------------------+
                 |                    |
                 v                    v
   +-----------------------+   +-----------------------+
   | System-1 decision head|   | System-2 text head    |
   | 1 matvec [A,E] x [E]  |   | autoregressive loop   |
   | softmax over A actions|   | max_new_tokens fwd's  |
   +-----------------------+   +-----------------------+
                 |                    ^
                 v                    |
        +-----------------+  no       |
        | self-routing    |-----------+
        | confidence>=thr |  escalate
        +-----------------+
                 | yes
                 v
        JSON decision (System-2 never entered)
```

Both heads read **the same vector**. That is the whole point: there is no second
model, no separate encoder, and no Python process. `RwkvModel::forward` gained
one optional out-parameter, and the decision head is a projection over it.

### 3.1 The tap

`src/core/rwkv.cpp`, head section:

```cpp
Tensor xl3("xl3", {E}, DType::F32);
{ ... layer_norm(xin3, &ln_out_w_, &ln_out_b_, xl3); }

// System-1 tap ("two heads, one brain")
if (hidden_out != nullptr) {
    if (hidden_out->dtype() != DType::F32 || hidden_out->numel() != E)
        *hidden_out = Tensor("hidden", {static_cast<int64_t>(E)}, DType::F32);
    std::memcpy(hidden_out->f32(), xl3.f32(),
                static_cast<size_t>(E) * sizeof(float));
}
```

`xl3` is the post-`ln_out` residual — the model's complete opinion about the
token it just read. Three properties matter:

1. **It is taken before the vocab projection**, so the head sees the same
   representation the text head is about to use, not a derived copy.
2. **It is never fed back into the recurrence.** Reading it cannot change any
   generated token. Passing `nullptr` is bit-identical to the old behaviour.
3. **The tensor is reused across steps** — reallocated only on a shape/dtype
   mismatch, so a steady-state decode does zero extra allocation.

### 3.2 The head

`include/omniseed/decision_head.h`, `src/decision_head.cpp`. This is Jev's
**Choice** primitive, specialised to a fixed action space:

```
logits[a] = bias[a] + sum_i proj[a][i] * h[i]      // [A,E] x [E] — one matvec
p         = softmax(logits)                        // over A, max-subtracted
action    = argmax(p);  confidence = p[action];  margin = p1 - p2
```

The action space is deliberately small (7). Each extra action costs `E` more
floats of projection and dilutes the softmax:

| Action | Meaning |
|---|---|
| `ABSTAIN` | no opinion — say nothing |
| `HOLD` | stay put, no new exposure |
| `BUY` | open / add long |
| `SELL` | open / add short |
| `CLOSE` | flatten an existing position |
| `HEDGE` | reduce net exposure without flattening |
| `EXPLAIN` | "I have a view, but expressing it needs language" |

`ABSTAIN` and `EXPLAIN` are the two escapes that make the head **fail-closed**:
an uncertain head escalates instead of guessing, and a head that is certain it
has nothing to say still says nothing.

### 3.3 Routing — `src/agent/agent_loop.cpp`

```
ABSTAIN                                  -> routing "abstain",  no fast path
EXPLAIN                                  -> routing "system2"
confidence >= threshold && margin >= floor -> routing "self",     fast path
otherwise                                -> routing "system2"
```

`DecisionMode` controls how far the head is allowed to go:

| Mode | Behaviour |
|---|---|
| `Off` (default) | System-2 only. The head is never allocated — zero cost. |
| `Hybrid` | Consult the head; if it self-routes, answer from it; else fall through to the unchanged text path. |
| `DecisionOnly` | Never enter the token loop. If the head will not self-route, return its verdict (normally `ABSTAIN`) rather than quietly generating text. |

`DecisionOnly` refusing to fall through is deliberate. A mode named
"decision-only" that silently generated text on low confidence would be lying
about what it does.

One honest edge case: on the prefix-cache fast path the prompt is not re-fed, so
there is no *fresh* hidden state. Rather than advance the recurrent state purely
to manufacture one — which would change what System-2 produces — the loop
escalates and says so.

---

## 4. Measured performance

`tests/test_decision_head.cpp` measures both sides with the same clock and the
same scalar fp32 inner loop, so this is like-for-like rather than a model of a
model. The System-2 baseline is deliberately **charitable**: it grants System-2
a bare output-head matvec `[V,E]` per token and *nothing else* — no attention,
no FFN, no channel mix. Real System-2 does all of that too, so the measured
ratio is a **lower bound**.

| Scenario | System-1 `decide()` | 1 System-2 token step | Ratio over 160 tokens |
|---|---|---|---|
| tiny (E=64, V=256) | ~1.1–1.7 µs | ~36 µs | **≈ 3,400–5,000×** |
| mid (E=256, V=8192) | ~3.5–4.8 µs | ~5.4–7.5 ms | **≈ 250,000×** |

The mandated bar is 100×. The test asserts `ratio >= 100`, `decide_ns < 1 ms`,
and `decide_ns < step_ns`, and prints the measured values on every run so a
regression is visible rather than silent.

Two caveats stated plainly:

* `DecisionResult::ms` / `last_ns()` include the two clock reads and will be
  dominated by them at this scale. Benchmarks should time a loop externally, as
  the test does.
* These numbers are the *head* only. End-to-end latency still includes the
  prompt prefill, which the decision head does not remove.

---

## 5. Using it

```bash
# System-1 acts when it is confident, System-2 otherwise
omniseed ask --model models/rwkv7-0.1B.gguf --mode hybrid \
             --prompt "should I add to the AAPL position?" \
             --decision-head models/decision-head.bin

# Never generate text; return the decision or ABSTAIN
omniseed chat --model models/rwkv7-0.1B.gguf --mode decision-only

# Retune the bar
omniseed ask --mode hybrid --decision-threshold 0.90 --prompt "..."
```

`chat` prints `[decision] {json}` per turn and marks the turns System-1 answered
by itself. `ask` prints the same line once.

Server (`-DOMNISEED_BUILD_SERVER=ON`):

```bash
omniseed_server --decision-mode hybrid --decision-threshold 0.85 \
                --decision-head models/decision-head.bin
# env fallbacks: OMNISEED_DECISION_MODE, OMNISEED_DECISION_HEAD
```

`/health` reports `"decision_mode"` and `"decision_head"` (`"fitted"`,
`"untrained"`, or `false`). `/ask` adds `"decision": {...}` and
`"fast_path": bool` to its response when the head was consulted. Reply strings
are now JSON-escaped; previously a reply containing a quote produced a malformed
body.

---

## 6. Honest limitations — read this before trusting an output

> ⚠️ **Two claims below are now out of date for the trading head**, which §32
> fitted and calibrated. They are kept as written and corrected in place with
> **Update (§32)** markers, because "the shipped head is untrained" was the
> correct description of this tree for most of its life and is still correct for
> every head except the ones listed in `docs/CALIBRATION.md`.

**The shipped head is untrained.** `DecisionHead::init()` seeds the projection
deterministically (SplitMix64, scaled by `1/sqrt(E)`) so tests and CI are
reproducible. That projection has **no learned meaning**. A seeded head will
happily report `BUY` at 0.97 confidence because that is what the arithmetic
says, not because it is right.

Three guards make this hard to mistake:

1. `DecisionMode::Off` is the default in the CLI, the server, and
   `AgentLoop::Config`. An untrained head cannot silently change behaviour.
2. `trained()` is true only when **every** action row was supplied via
   `set_action()` or `load()` — a half-populated head is still a placeholder.
   `provenance()` says which.
3. Both the CLI and the server log a `WARN` at startup when the mode is on and
   the head is not fitted.

**Confidence is not calibrated.** Jev's central contribution is RLCD, which
trains the output probabilities to match long-run frequencies. The softmax here
is a raw, uncalibrated score. A threshold of 0.85 does not mean "85% likely to
be correct"; it is a tuned operating point. Do not present it as a probability
of success.

**Update (§32):** the *trading* `DecisionHead` now loads from
`models/heads/trading_head.bin` with `trained() == true` and a fitted
temperature `T = 13.325`, and its confidence **is** calibrated — measured ECE
0.622 → 0.056 on 369 held-out bars. Read that carefully, though: calibration
fixes the *number*, not the *judgement*. The same head is only **0.244 accurate
over 7 actions** (near chance), so its maximum calibrated confidence is
**0.4966**, below the 0.85 gate, and it **self-routes 0/369**. A well-calibrated
bad head is a bad head that knows it. The remaining heads are still uncalibrated
and the paragraph above still applies to them verbatim. See `docs/CALIBRATION.md`.

**`target_asset` is empty unless mapped.** Call `set_action_asset()` (or fit a
projection that carries asset identity) before reading that field.

**What fitting would require.** The projection is a linear map
`R^E -> R^A`, so it can be fitted offline from `(hidden_state, correct_action)`
pairs collected by running the backbone over labelled episodes, with a
cross-entropy loss — and, to get calibration, an asymmetric penalty on
overconfident errors in the RLCD spirit. **This is now implemented** as
`tools/train_heads.py` (§32) for the language and trading heads, via plain
cross-entropy plus post-hoc temperature scaling rather than an asymmetric RLCD
penalty. `set_action()` and `save()`/`load()` remain the runtime interface.

**`to_json()` being valid JSON does not mean the decision is correct.** Same
distinction TypeSafe draws between schema conformance and semantic accuracy.

---

## 7. Test coverage

`tests/test_decision_head.cpp` — ctest `omniseed_decision_head`, **149 checks,
0 failures**, fully offline (no model, no GGUF, no network):

* action space and labels, including the `COUNT` sentinel
* geometry, config-driven construction, degenerate-width rejection
* the one-matvec contract; uniform weights → uniform, fail-closed distribution
* softmax/argmax against exactly computable fixtures, and sign sensitivity
* fail-closed routing: `ABSTAIN` and `EXPLAIN` never self-route, however confident
* the confidence gate and the margin gate (a near-tie escalates)
* determinism (same seed → bit-identical) and seed sensitivity
* `decide()` does not mutate its input
* the `Tensor` overload agrees with the raw-pointer path; wrong width/dtype/null
  are refused rather than read out of bounds
* `to_json` well-formedness and contract fields
* provenance: `trained()` only once every row is fitted; re-setting does not
  double-count
* persistence round-trip (fitted **and** untrained), plus rejection of missing
  files, bad magic, and truncation
* the ≥100× speed claim, with the measured numbers printed

`omniseed_lora` (5.18 s), `omniseed_trading` (2129 checks),
`omniseed_trading_edge` (115), `omniseed_market_perception`,
`omniseed_paper_session` (47), `omniseed_platform`, `omniseed_real_weights`,
`omniseed_sides`, `omniseed_qat_ternary` — all unchanged and green. Full board:
**10/10 passed**.

---

## 8. Sources

* TypeSafe AI — *Introduction*, https://docs.typesafe.ai/introduction
  (primitives, confidence, parallel evaluation, System One framing)
* *TypeSafe Jev 技术拆解：非自回归决策原语、RLCD 与本地开源实现*, 2026-09-19,
  https://www.kevnu.com/zh/posts/... (parallel sampler diagram, RLCD, benchmark
  table, limits, OpenJev / mini-jev / jevlike / vLLM PR #57250 / Outlines /
  Guidance)
* Community discussion of the pattern's limits (no CoT, black-box
  probabilities, long-horizon degradation), 2026-09.

Facts about Jev's internals that TypeSafe has **not** published (parameter
count, layer count, hidden dimension) are not guessed at anywhere in this
document or in the code.
