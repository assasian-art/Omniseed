# OMNISEED — JEV Decision Capabilities: what is built, what is not

The mandate lists ten core decision capabilities and five domain head families.
This document records the **status of each**, with the evidence, and states the
gaps plainly. A feature list that only contains ticks is not a status report.

> ⚠️ **This is a dated snapshot.** It was written before §32 (head training and
> calibration), §33 (the multimodal joint), §34 (module coverage), §35 (the
> uncertainty split) and §36 (batched head evaluation). Where a later milestone
> changed the answer, the original text is kept and marked **Update (§NN)** below
> it, so the record of what was believed — and why — survives alongside the
> correction. Sections carrying an update: **2, 6, 9, 10** and Part 2
> (vision/audio).

Legend: **DONE** · **PARTIAL** · **NOT STARTED**

---

## Part 1 — The ten core capabilities

### 1. Sub-millisecond decisions — **DONE**

Measured, not estimated. `tests/test_heads.cpp` E1/E2, Release build, E = 768,
2000 iterations per row, on the development box:

| Head | Actions / labels | Cost |
|---|---|---|
| `decide(general)` | 4 actions | **0.498 us** |
| `decide(trading)` | 7 actions | 0.756 us |
| `decide(vision)` | 4 actions | 0.940 us |
| `decide(audio)` | 4 actions | 0.941 us |
| `decide(language)` | 6 actions | **1.150 us** (worst) |
| `classify(language.intent)` | 7 labels | 2.139 us |
| `score()` | 3 rows | 0.505 us |
| `HeadRouter::plan()` | 9 matvecs | 15.307 us |

Against the mandate's **1 ms** budget the worst case is **~870x inside**. Against
the mandate's **897 ns** figure: the 4-action domains beat it (general is 498 ns)
and the 6-action language domain is ~1.3x over it. The figure scales with the
action count, so it is a per-action-space number rather than a single constant —
which is why the assertion in the test is the 1 ms budget and the per-row numbers
are printed.

The router is the one figure worth explaining: 15.3 us for 6,912 MACs is *not*
arithmetic-bound. The returned `ActivationPlan` owns a `heads` vector and a
`reason` string, and those two allocations cost more than the multiply-accumulates.
`plan()` reserves both, which took it from 19.4 us to 15.3 us. Anything further
means changing the return type.

### 2. Calibrated confidence — **DONE for 5 heads, NOT STARTED for the rest**

> **Superseded in part by §32 (Milestone 4).** This section was written when
> *nothing* in the tree was fitted. It is kept for the record, and the current
> status is stated below it. Full detail: **`docs/CALIBRATION.md`** and
> `PROJECT_STATE.md` §32.

**What was true then, and is still true of the unfitted heads:** every head's
projection starts as a **seeded placeholder**. `softmax` over an untrained
projection produces a number in [0,1] that looks like a probability and is not
one. Every head exposes `trained()` / `provenance()`, and
`UnifiedPipeline::provenance()` reports `NOT FULLY FITTED (placeholder)`.

**What changed in §32:** temperature scaling **is now implemented and measured**,
for the heads that have labelled data. `tools/train_heads.py` fits a projection
offline on a held-out split, fits one temperature **per label set** by 5-fold CV,
and measures ECE before and after. The results are real and modest:

| head | K | n | accuracy | T | ECE → calibrated |
|---|---|---|---|---|---|
| `language.language` | 4 | 71 | 0.845 | 5.161 | 0.139 → 0.089 |
| `trading.regime` | 4 | 369 | 0.705 | 3.250 | 0.210 → 0.034 |
| `language.sentiment` | 3 | 71 | 0.676 | 3.583 | 0.248 → 0.087 |
| `language.intent` | 7 | 71 | 0.634 | 23.441 | 0.360 → 0.198 |
| `DecisionAction` | 7 | 369 | **0.244** | 13.325 | 0.622 → 0.056 |

Two findings worth carrying forward, both measured rather than assumed:

* **A temperature is per-set, not per-head.** A single pooled T made
  `language.language` *worse* (ECE 0.172 → 0.194) while fixing
  `language.intent` (0.451 → 0.115), because the two sets want T ≈ 5 and T ≈ 23.
* **Calibration does not create accuracy.** `DecisionAction` is calibrated to
  ECE 0.056 and is still 0.244 accurate over 7 classes — near chance. A
  well-calibrated bad head is a bad head that knows it. Its maximum calibrated
  confidence is 0.4966, below the 0.85 gate, so it **self-routes 0/369**.

**Still NOT STARTED:** the vision and audio heads, and the `DecisionAction` head's
*accuracy* (as opposed to its calibration). The language heads' `confidence`
remains a documented **heuristic strength** — how many independent cues agreed —
and is explicitly not the calibrated `DecisionHead::confidence_score`.

The machinery a further calibration pass would need is in place: per-head fitting
hooks (`set_action_row`, `set_label_row`, `set_row`, `set_head_gate`,
`set_domain_probe`), `fitted_rows()` counters, `trained()` gates, and a `margin`
on every distribution so a caller can see that a call was a coin flip.

**Do not ship a decision on an untrained head's confidence.** `trained()` is how
you tell.

### 3. Multi-class classification — **DONE**

`ClassificationHead` (`include/omniseed/classification_head.h`). One
`[total_labels, E]` projection; each registered set owns a contiguous slice, so a
new domain costs its labels' rows and nothing else. 13 sets registered by
default across all five domains. Returns `top_k` **and** `margin`
(`p(top) - p(second)`), because a 0.51/0.49 split has a winner and no
information.

Max-subtract-before-`exp` softmax: without it a large logit overflows to `+inf`,
every `exp()` becomes `inf`, `inf/inf` is NaN, and `argmax` turns a NaN
distribution into **label 0** — a real label.

### 4. Type-safe JSON — **PARTIAL**

What is genuinely enforced:

* **A field is present iff its head ran.** `UnifiedOutput::normalise()` drops
  anything the activation plan does not name and records it in `warnings`. A
  stale decision from a previous turn cannot leak into this turn's answer.
* **No bare `nan` / `inf` ever reaches the wire.** `heads_detail::append_float`
  substitutes `0.0`. A document containing `nan` is invalid, and a downstream
  parse failure is a far worse outcome than a visible default.
* **Failure has an unambiguous channel.** A classification that fails carries the
  reason in `domain` as `<error: ...>` and leaves `top_k` **empty**. No registered
  set name can begin with `<`, so empty is unambiguous — a fabricated uniform
  distribution would be indistinguishable from a real, useless one.
* **Escaping** for the two characters that would actually break a document, plus
  control characters. Tests assert the documents are structurally valid
  (balanced, no bare NaN tokens).

What is **not** there: no JSON schema, no compile-time field checking, and no
reflection. C++ has no reflection, so the documents are assembled by hand. This
is "type-safe" in the sense that field *presence* is structurally enforced, not
in the sense a schema validator would mean.

### 5. Fail-closed safety — **DONE**

| Situation | Routing / result |
|---|---|
| Router not ready | plan = the safe **superset** `{decision, token}`, reason recorded |
| No head gate fired | same superset, reason `no-gate-fired` |
| Decision confidence < threshold | `system2` — escalate, do not guess |
| Decision margin < floor | `system2` |
| argmax is the abstain action | `abstain` |
| argmax is the explain action | `system2` |
| Domain has no escape hatches | still escalates on low confidence |
| Unknown classification set | `top_k` empty, error in `domain` |
| Unregistered domain | routing `error`, `matvecs` 0 |
| Not-ready / width-mismatch / null | zeroed result with `matvecs == 0` |

The invariant tested directly: with the threshold set above the maximum
attainable confidence, **no domain can self-route**
(`tests/test_heads.cpp` D9); and with a margin floor of 1.0, likewise (D11).

`0.85` is shared by `DecisionHead`, the router's `fast_path_confidence`, and the
sniper's regime gate, so no two layers can disagree about what "confident" means.

### 6. Batch processing — **DONE at the head level**

> **Update (§36).** The text below was written when no head had a batch API. One
> now does. Kept for the record; the correction follows it.

Every head takes a single `h[E]`. There is no batched forward pass and no
`[B, E]` input path. The reason is not oversight: the edge runtime is a
single-threaded scalar RWKV-7 at ~1.4 tok/s, where a batch would buy latency, not
throughput. A batch API belongs with the serving path, not with the heads.

**Update (§36):** the *heads* now take a `[B, E]` block —
`classify_batch` / `decide_batch` / `score_batch`, built on
`batch_gemm` (`[B, E] × [L, E]ᵀ + bias[L]`). Measured on this box, E = 768,
Release:

| | per-row | batched | |
|---|---|---|---|
| `classify_batch`, 369 real rows | 9.5 µs/row | **1.7 µs/row** | 5.6× |
| `decide_batch`, 369 real rows | 9.9 µs/row | **2.0 µs/row** | 5.0× |
| whole stack, 1000 signals | 25.27 ms | **4.50 ms** | **5.62×** |

The reasoning above is still correct about the **backbone**: this tree's RWKV-7 is
the scalar recurrent form, so 1000 signals still need 1000 sequential forwards.
Batching applies to the readout, which is a GEMM — and on the *backtest* path the
forwards are already paid for and cached to `hidden.f32`, so the readout was the
remaining cost. The claim "one forward pass for 1000 signals" would be false and
is not made.

The scalar kernel is **bit-identical** to the per-row path; the AVX2 kernel is
not (eight lanes vs one accumulator) and is held to "the answer does not change"
instead — 0 top-1, action and routing mismatches over 369 real + 200 synthetic
rows. Full detail: **`docs/BATCH.md`**.

### 7. Streaming — **NOT STARTED** (at the head level)

The **token** head streams tokens — that is what the existing generator does. The
decision, classification and scoring heads are single-shot by construction: one
matvec, one answer. There is no incremental/streaming decision API, and no
partial-result protocol.

### 8. Hierarchical routing — **PARTIAL (one level)**

One level is real: `HeadRouter` reads `h[E]` and decides *which* heads run and in
*what mode*, and its domain readout is the first branch of a hierarchy
(domain -> that domain's action space). What is missing is a **second level** — a
sub-router that would, say, pick a specialist action space *within* trading. The
structure supports it (`DomainDecisionHead` already holds N independent action
spaces), but nothing drives it.

### 9. Ensemble voting — **PARTIAL**

Real, but **only inside trading**, and it predates this work: `src/trading/router.cpp`
has `EnsembleVerdict` / `evaluate()` / `should_veto_long()`, with a veto weight
floor of 0.50 so a lone weak signal cannot veto a trade
(`tests/test_strategy_zoo.cpp` B4 pins it). The ensemble **blocks, never
promotes**, and that invariant is tested.

There is **no general ensemble across heads** — nothing combines the decision
head's action with the classification head's label and the scoring head's
priority into a vote. That would be a natural next step and is not built.

**Update (§35):** the *operator* a cross-head ensemble needs now exists.
`UncertaintyDecomposition::from_ensemble()` implements the rigorous
Depeweg 2018 decomposition (`total = H(mean p)`, `aleatoric = mean H(p)`,
`epistemic = total − aleatoric ≥ 0 by Jensen`) and is tested. **Nothing can feed
it** — no head ensemble has been fitted, so there is no `q(θ)`. As elsewhere in
this tree, the gap is the *joint*, not the capability. See `docs/UNCERTAINTY.md`.

### 10. Feedback hooks — **PARTIAL**

No mechanism records an outcome and feeds it back **into a head**. What exists is
the *plumbing* a feedback loop would write through — the per-head fitting hooks
and their `fitted_rows()` accounting — but nothing drives them, and there is no
journal of (prediction, outcome) pairs. Note that the trading path does have a
frozen entry journal (`tools/decision_bridge.py` re-reads it), which is the
closest thing in the tree to the substrate this would need.

**Update (§34):** there *is* a feedback substrate, and it was broken.
`SelfImprovement` (`src/agent/self_improvement.*`) records a task-keyed trace with
a success/fail count, an average duration and a last-used stamp, and replays the
best known step list for a matching task. That is a real (prediction, outcome)
store. But its `save()` wrote a `uint32` magic `0x54524953` while `load()`
compared the **string** `"SRIT"` — little-endian those bytes are `'S','I','R','T'`
— so `load()` rejected every file `save()` wrote and **the trace cache never
persisted**. Silent, because an unreadable cache is indistinguishable from an
empty one. Fixed in §34 with one shared `kMagic`, and pinned by test C9.

So the honest status is: the store exists and now persists, but nothing yet
closes the loop from a *head's* prediction to a *recorded market outcome*.

---

## Part 2 — The five domain head families

| Domain | Decision action space | Classification label sets | Domain-specific logic |
|---|---|---|---|
| **Trading** | `abstain, hold, buy, sell, close, hedge, explain` | `trading.regime`, `trading.action` | **Yes** — `src/trading/*`: regime engine, strategy zoo, sniper, ensemble router, risk engine |
| **Language** | `abstain, answer, clarify, search, execute, explain` | `language.intent`, `language.language`, `language.sentiment`, `language.task` | **Yes** — `src/language/*`: deterministic lexical intent / script / sentiment |
| **Vision** | `abstain, describe, flag_anomaly, explain` | `vision.scene`, `vision.anomaly` | **No** — label space only |
| **Audio** | `abstain, transcribe, respond, explain` | `audio.wake`, `audio.emotion`, `audio.speaker` | **No** — label space only |
| **General** | `abstain, answer, route, explain` | `general.routing`, `general.priority` | **Yes** — `ScoringHead` (priority/urgency/confidence) |

The vision and audio rows are the honest ones: their action spaces and label sets
are **registered**, so a fitted projection would light them up immediately — but
no projection has been fitted for either, because neither has labelled data.

**Update (§33):** the *pipe* now exists even though the water does not.
`MultimodalBridge` (`include/omniseed/multimodal.h`) turns a
`VisionEncoder` output and `FocalCodec` codes into the discrete token space
`RwkvModel::forward()` accepts, by nearest-neighbour quantisation against the
model's own token-embedding matrix (no training, no extra RAM). So an image now
becomes an `h[E]` the heads can read. What is still missing is the fitted
vision/audio **projection**: the heads run and print a label, but with
`trained() == false` that label is meaningless. See `docs/MULTIMODAL.md`.

The trading action space mirrors `DecisionHead`'s exactly (lower-cased), so the
two heads cannot drift into disagreeing about what actions exist. That is asserted
in `tests/test_heads.cpp` D6.

---

## Part 3 — Known limitations, stated once

1. **Nothing is trained *except* the heads listed in §32.** Every remaining
   projection is a deterministic placeholder — the vision and audio heads in
   particular. `trained()` / `provenance()` is how you tell them apart; see
   capability 2.
2. **Romanised Bengali reads as English.** Language detection is by **script**.
   `"ami bhalo achi"` is pure ASCII and is therefore indistinguishable from
   English without a transliteration model this tree does not have.
   `tests/test_language_heads.cpp` B4 pins the actual behaviour so a future fix
   is a deliberate, visible change rather than a surprise.
3. **Bengali "না" is ambiguous.** It is both the negator and the sentence-final
   question particle (`তুমি যাবে না?` = "won't you go?"). A lexical scorer cannot
   separate those without syntax, so a trailing "না" will flip a nearby sentiment
   word. Documented in `src/language/sentiment.cpp`.
4. **Sentiment requires a lexicon hit.** A bare `"no"` with no sentiment word
   beside it reads as Neutral. That is the deliberate price of keeping negators
   out of both lexicons, which is what makes `"no problem"` come out positive.
5. **The router's plan is not allocation-free.** Its projection is; the returned
   struct owns a vector and a string. Measured and documented in
   `include/omniseed/router.h`.
6. **The CLI does not use the router on its default path.** `--mode` still drives
   the existing `DecisionMode` path, which has a live consumer contract
   (`tools/decision_bridge.py`). Rewiring it needs its own parity gate.
   `--mode text-only` is accepted as an alias for `off`.
