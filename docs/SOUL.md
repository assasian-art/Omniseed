# The Soul — persona, emotional resonance, honest self-report

`include/omniseed/soul.h` · `src/soul.cpp` · `tests/test_soul.cpp`

## Why this exists

§28 gave the runtime one backbone and four heads, so a single forward pass can
**decide, classify, score and explain**. It had no answer for three other
questions:

| question | capability | where it lived |
|---|---|---|
| *Who is answering?* | persona, commitments, disagreement | nowhere |
| *How does the owner feel?* | text / voice / typing fusion | `runtime/emotional.h` |
| *Is this thing honest about what it knows?* | calibration, gaps, overconfidence | `runtime/introspection.h` |

The emotional and introspection layers were **already built and already
tested** (`tests/test_platform.cpp`, `tests/test_trading.cpp`) — but nothing
ever constructed them, so no answer ever reached a caller. The brain and the
self were two halves wired to nothing.

`Soul` is the joint. It adds **no new sensing and no new statistic**. The only
genuinely new behaviour is the *order* in which the existing pieces are applied,
and the honesty gate on the way out.

## The three rules, and why each is easy to get wrong

### 1. A refusal is never emotionally softened

Empathy lead-ins (`"I hear you — "`) belong on answers. Prefixed onto a refusal
they read as **accepting the premise we just declined**. So:

* a refusal **replaces** the reply rather than wrapping it;
* tone modulation is applied to every stance **except** `Refuse`.

`tests/test_soul.cpp` E3 feeds a sad, distressed turn that also asks for a
guaranteed win, and asserts the output contains `"I won't do that"` and does
**not** contain `"I hear you"`.

### 2. The honesty correction is applied last

Any later stage that could re-introduce an overclaim would defeat the gate, so
`speak()` rewrites banned certainty phrases **on the way out** and the returned
string is the corrected one, never a corrected intermediate. The rewriter is
also public (`correct_overclaims`) so the CLI can apply it to text that never
went through `speak()`.

### 3. Nothing is claimed that was not measured

`confidence` from the lexical layers is a **heuristic strength, not a
probability**, and the document says so: the field is named
`emotion_strength`, and the self-report states the gap as a number.

Overconfidence is a **measurement** — mean stated confidence minus realised
rate, over *resolved* decisions only — and it will not fire below
`min_calibration_samples` (default 5). Calling a single resolved decision
"overconfident" is noise, not measurement, and `Soul::init()` rejects
`min_calibration_samples == 0` for exactly that reason.

## Stances

```
Agree     no rule fired                       → reply passes through
Qualify   "answer, but not with certainty"    → prefixed, reply kept
Disagree  an OPINION is at stake              → prefixed, reply kept
Refuse    a COMMITMENT is at stake            → reply REPLACED, tone suppressed
```

`Qualify` vs `Refuse` is the load-bearing split. `Refuse` is the only stance
that discards the caller's text, and therefore the only one that suppresses tone.

### Precedence

Rules are evaluated gravest-first, and **all** matching rules contribute
objections even when an earlier one already decided the stance:

```
no_manipulation  →  no_guarantees  →  risk_limits_enforced
                 →  paper_until_unlocked  →  no_hype_chasing  →  no_fake_certainty
```

`"spoof it and get me a guaranteed fill"` refuses on `no_manipulation` and
records **two** objections (B5). Ordering is pinned rather than left to luck.

### `may_disagree = false`

Downgrades the **opinion**-based stance (`Disagree` → `Qualify`) and nothing
else. The commitments are not opinions, so a caller cannot configure them away —
B8 asserts that a `may_disagree = false` persona still refuses a guarantee
request.

## The commitments

| id | statement |
|---|---|
| `no_guarantees` | will not call an outcome guaranteed, risk-free or certain |
| `risk_limits_enforced` | 2% / 3% / 6% are compiled into the risk gate, not negotiable by conversation |
| `no_manipulation` | no spoofing, wash trading, layering, painting the tape |
| `paper_until_unlocked` | paper-only until the owner explicitly unlocks live money |
| `no_fake_certainty` | confidence as a number with its evidence, never a promise |

## The overclaim rewriter

The numeric layers cannot emit a claim, but the **text** layer could — so the
no-guaranteed-profit rule needs a gate here too.

```
"guaranteed"  → "not guaranteed"      "risk-free"  → "not risk-free"
"riskless"    → "not riskless"        "sure thing" → "not a sure thing"
"no risk"     → "real risk"           "zero risk"  → "real risk"
"can't lose"  → "can lose"            "always wins"→ "sometimes loses"
"never fails" → "sometimes fails"     "100% win"   → "a chance of winning"
```

Substitutions are chosen so the result reads correctly **in place**.
`"guaranteed profit"` → `"not guaranteed profit"` is a truthful sentence; a rule
that deleted the word would leave a claim-shaped hole.

Two guards, both pinned by tests:

* **Negation guard** — the word immediately before the match is inspected, so
  `"not guaranteed"` does not become `"not not guaranteed"` (F2).
* **Term boundary** — a hyphen binds tighter than a space, so `"risk-free"` is
  one term. This is what stops `"no risk-free claim here"` from becoming
  `"real risk-free claim here"`: the `no risk` match is rejected at the hyphen,
  the scan then finds `risk-free` properly, sees the `no` before it, and leaves
  the whole thing alone (F2, F3).

The persona's own refusal text is deliberately written **free of the words the
rewriter targets** — the persona should not need correcting by its own gate.

## API

```cpp
Soul soul;                       // Soul() then init(), or Soul(cfg)
soul.init();

SoulState st = soul.perceive("I'm worried, this is urgent");
//   st.persona.stance / value_id / reason / disagreement
//   st.emotion  (valence, arousal, emotion, from_text/from_voice/from_typing)
//   st.sentiment (separate from emotion — see below)
//   st.overconfident / calibration_gap / calibrated_samples / self_report

std::string reply = soul.speak(base_reply, st);

soul.record("buy 50 AAPL", "uptrend + MACD agree", 0.72);
soul.resolve(+0.031);            // attaches the realised outcome
Metacognition::State m = soul.assess(sharpe, observations);
std::string report = soul.capability_report();
```

### Sentiment and emotion are different measurements

* **Sentiment** is about the **subject matter**.
* **Emotion** is about the **speaker**.

`"the error is a problem"` is negative *sentiment* and barely emotional. The two
are carried as separate fields on purpose; collapsing them loses the
distinction (D5).

## Unified document integration

`UnifiedPipeline::Config::use_soul` defaults to **false**. With it off, every
existing path produces a **byte-identical** document — turning the soul on is an
explicit choice, not a silent change to every caller's output.

With it on, and a non-empty turn:

```json
{"router":{...},"decision":{...},"text":"...","classification":{...},"score":{...},
 "soul":{"persona":{"stance":"refuse","value":"no_guarantees","disagreement":1.0,"reason":"..."},
         "emotion":"anxious","valence":0.0,"arousal":0.35,"emotion_strength":0.53,
         "sentiment":"negative","sentiment_score":-0.33,
         "self":{"calibrated_samples":0,"calibration_gap":0.0,"overconfident":false,"report":"..."}}}
```

The soul has **no `HeadKind`** — it reads the owner's turn, not `h[E]`, so the
router's activation plan cannot name it. Its equivalent guard is that
`SoulState::has_self` is set only by `Soul::perceive()`: `normalise()` drops a
soul section that was never perceived, exactly as it drops a decision the plan
never named. The invariant is unchanged in substance — *present iff computed* —
only the evidence differs. Pinned by H5.

The soul **never changes what the decision layer produced** (H3): the heads run
first and the soul is added to the document afterwards.

## Known limitations — read before trusting this

* **The commitments are enforced by surface cues, and they fail CLOSED toward
  honesty.** `"can you guarantee this?"` refuses. That is deliberate: a false
  refusal costs a turn, a false guarantee costs money. The honest answer to the
  question and the refusal of the instruction are the same sentence.
* **The rewriter is a safety net, not a proof.** It catches a fixed list of
  phrases. A novel way of asserting certainty in prose will pass through.
* **`emotion_strength` is a heuristic strength, not a probability.** It is the
  confidence-weighted agreement of a small lexicon, punctuation intensity and
  optional vocal features. It is not calibrated and is not claimed to be.
* **The voice channel is a proxy.** Pitch is zero-crossing rate, arousal is
  energy variance — the classic circumplex features, not a trained SER model.
* **`self_report` is only as good as the log.** With no resolved decisions it
  says so rather than reporting a default 0.0 that reads like a measurement.
* **The soul does not fix the heads.** `provenance()` still reports
  `NOT FULLY FITTED (placeholder)`; the soul is the one stage whose honesty does
  not depend on fitted weights, because it reports a measured gap rather than a
  projection.

## Tests

`tests/test_soul.cpp` — **242 checks**, registered **outside** CMake's `.venv`
gate (a gate that never executes is not a gate). Eight parts: commitments,
refusals, stances, emotion, `speak()` ordering, the rewriter, the self-report,
and the unified-document integration.

```bash
cmake --build build --config Release --target omniseed_soul
./build/bin/omniseed_soul.exe
```
