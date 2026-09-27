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

## Memory (§31) — what the soul remembers

A soul that forgets every exchange the moment it ends is a lookup table with
opinions. §31 wires the existing `MemoryCrystals` store into the Soul so it
remembers what the owner asked and what it answered.

```bash
./build/bin/omniseed.exe demo-soul      # the whole mechanism, with measurements
```

### `perceive()` is still const, and still writes nothing

This is the load-bearing decision. `UnifiedPipeline::run()` is a `const` method
that calls `Soul::perceive()`, and **a const method that silently appends to
long-term memory is a trap**: the caller cannot see the write in the signature,
and calling `run()` twice stops being idempotent in a way no type can express.

So remembering lives in named non-const entry points:

| call | what it does |
|---|---|
| `perceive_and_recall(turn)` | `perceive()`, then recall related memories, then store the owner's question |
| `remember(question, reply)` | store the soul's reply, reinforce what the turn recalled |
| `converse(turn, base_reply)` | the whole turn: recall → `speak()` → store. **This is the path a chat loop should call** |
| `recall(query, k)` | ranked recall, no state change |
| `top_match(query)` | the best match **ignoring** the floor — for measuring, not for claiming |
| `decay_memories()` | the Ebbinghaus pass, at the soul's clock |
| `advance_memory_clock(tokens)` | simulate elapsed time (tests, dream) |

`UnifiedPipeline::run_memorable()` is the non-const sibling of `run()`: same
document, plus recall, plus the question stored. I10 asserts that the const
`run()` really does **not** store.

### The same question twice

```
turn 1  owner: what is my position size limit
        soul : Two percent per trade, three per day, six per week.
turn 2  owner: how did the last aapl trade go
        soul : It closed up on light volume.                  ← no false memory
turn 3  owner: what is my position size limit
        soul : I said this before: "Two percent per trade, three per day,
               six per week.". Two percent per trade, three per day, six per week.
```

Two things had to be built for turn 3 to be honest rather than merely
impressive.

**1. A relevance floor.** With no floor, `retrieve()` returns the *k nearest*
crystals whatever they say, and "nearest" among a handful of unrelated memories
still looks like an answer — the soul announces that it remembers things it has
never seen. The first version of this demo did exactly that: an unrelated
question scored **0.775** and produced a confident `"I said this before: ..."`.

`min_relevance` (default **0.92**) is **measured, not guessed**. On the minimal
byte-level tokenizer:

| probe | score | verdict |
|---|---|---|
| a question the store holds (verbatim) | 0.988 | claimed |
| the same question, one character off | 0.990 | claimed |
| a paraphrase of it | 0.973 | claimed |
| a question from an earlier turn | 0.989 | claimed |
| a question the store has **never** seen (short) | 0.828 | no claim |
| a question the store has **never** seen (long) | 0.876 | no claim |

`demo-soul` prints this table every run and labels each row `ok` or `MISMATCH`,
so the threshold can be re-derived rather than trusted.

**2. Answers are paired with their questions.** A reply's words rarely resemble
the question's, so a reply crystal does not clear the floor against its own
question — the soul would only ever quote *what the owner said*. A recalled
question therefore brings its answer with it: the answer is relevant **by
construction**, not by its own cosine. That is what makes turn 3 quote the
soul's own prior sentence.

### Decay, and the bug that made it backwards

`effective = importance × exp(−age_days / τ) × (1 + 0.1·min(hits, 10))`

The previous `decay(double now_unix_seconds)` **ignored its argument** and
derived age as `last_access_token / 100000.0` — reading a stream position as if
it were already an elapsed age. Combined with `retrieve()` writing
`last_access_token = created_at_token + query_tokens.size()`, the model was
inverted in two ways at once:

* a crystal created late (large stream position) decayed **immediately**, while
  one created at position 0 could **never** decay;
* **recall made a memory older**, not younger — so using a memory could not save
  it from being forgotten.

Fixed:

* `crystallize()` sets `last_access_token = stream_pos` at birth;
* `decay(uint64_t now_token)` honours its argument and computes
  `age_days = (now_token − last_access_token) / tokens_per_day`, clamped at zero
  so a clock behind the stamp cannot produce a negative age and an importance
  above 1.0;
* `retrieve(..., now_token)` refreshes recency — **recall is reinforcement**;
* `reinforce(id, now_token, boost)` raises importance and resets the age.

Nothing called `decay()` before this change, so the fix cost no compatibility.
J1–J5 pin it, including that the same crystal survives at its own clock and dies
four days later at a later one.

### A repeated exchange is not filed twice

Reinforcement alone is not enough: a store that appends a crystal per turn is a
log file. Two dedupes:

* a question scoring ≥ `duplicate_recall_score` (0.98) against one already held
  is **reinforced, not re-filed**;
* if that turn's answer is also unchanged, the answer is not re-filed either.

`demo-soul` shows this directly — turn 3 repeats turn 1 and the store stays at
**4 crystals**.

### What is stored, and what is not

* The question and the reply are stored as **separate crystals**, each wrapped in
  its chat role markers so the crystal's own token stream records who spoke.
* **The memory gloss is never stored.** It is an annotation on the turn, not
  something the soul said. Storing it made the soul quote itself recursively:
  `"I said this before: \"I said this before: ...\""`.
* A turn shorter than `crystallize()`'s 8-token floor is **counted** in
  `memory_skipped()`, not padded to fit.
* Crystal summaries are capped at `max_len_tokens` (**96** for the soul; the
  memory layer's own default of 48 truncates an ordinary sentence mid-word).

### Known limitations of memory

* **The margin is thin, and it narrows as the store grows.** Byte-level bags are
  close to character histograms, so unrelated English still scores 0.83–0.88.
  The 0.92 floor sits in a ~0.10-wide gap. With the trained 8k–16k vocabulary the
  tokens are words rather than bytes and discrimination is far better — **but
  `min_relevance` and `duplicate_recall_score` must be re-measured against it**
  (`demo-soul` prints the numbers).
* **Recall is near-verbatim.** A paraphrase at 0.973 clears the floor; a looser
  one would not. The soul under-claims rather than over-claims, deliberately.
* **Speaker roles and question/reply links are session-local.** They are not
  persisted, so crystals restored from disk report role `unknown` and are not
  paired. `MemoryCrystals` persists crystals, not the Soul's index over them.
* **The gloss is not emotionally modulated.** It leads the reply, outside
  `speak()`, because it is an annotation rather than an utterance. Rule 2 still
  holds: the overclaim rewriter runs last, on the composed string.
* **A refusal is never decorated with a memory.** A refusal is the whole reply;
  attaching context would read as negotiating the thing just declined (I7).
* **`converse()` is the only stateful path.** A caller using
  `perceive_and_recall()` + `speak()` directly gets no gloss — and no storage of
  the reply, because `speak()` cannot know it will be remembered.

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

`tests/test_soul.cpp` — **378 checks**, registered **outside** CMake's `.venv`
gate (a gate that never executes is not a gate). Ten parts: commitments,
refusals, stances, emotion, `speak()` ordering, the rewriter, the self-report,
the unified-document integration, **memory & recall (§31)**, and **decay over
simulated time (§31)**.

```bash
cmake --build build --config Release --target omniseed_soul
./build/bin/omniseed_soul.exe
./build/bin/omniseed.exe demo-soul
```
