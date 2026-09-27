# OMNISEED — Domains

Five domains share one backbone. Each owns an **action space** (for the decision
head), a set of **label sets** (for the classification head), and — where it
exists — **domain-specific logic**.

All domain names come from `omniseed::Domain` (`include/omniseed/heads.h`):

```cpp
enum class Domain : int32_t { General = 0, Trading, Language, Vision, Audio, COUNT };
```

`General` is **0** on purpose: an unclassified input lands there, and a zero
default means "unknown" rather than a silently chosen specialist domain.
`domain_from_name()` returns `General` for anything unrecognised — a typo must not
be able to route a request to the trading head.

---

## Trading

The most complete domain, and the only one with a live consumer contract.

| | |
|---|---|
| Action space | `abstain, hold, buy, sell, close, hedge, explain` |
| Abstain / explain | index 0 / index 6 |
| Label sets | `trading.regime` = `{trend_up, trend_down, range, high_vol}`, `trading.action` = `{hold, buy, sell, close, hedge, abstain}` |
| Logic | `src/trading/regime_engine.cpp`, `strategy_zoo.cpp`, `router.cpp`, `sniper.cpp`; `src/decision_head.cpp` |

The action space **mirrors `DecisionHead`'s exactly** (lower-cased), so the two
heads cannot drift into disagreeing about what actions exist
(`tests/test_heads.cpp` D6 asserts it). `from_decision_result()` is the lossless
adapter and does the lower-casing.

`trading.regime` uses the four labels the regime engine already produces, so the
classifier and the engine cannot disagree about the vocabulary.

Risk limits are **C++-enforced**: 2% per trade / 3% daily. Python mirrors them for
display only. A change to those numbers is a change to the risk header, not to a
config file.

**The oracle is still in the tree.** `tools/monster/*.py` remains as the reference
implementation and `tests/test_regime_parity.py` /
`tests/test_strategy_parity.py` diff the C++ against it over identical bars and
fail on any divergence beyond `1e-9` relative. That gate is what makes retiring
the Python from the runtime path safe — a hand-port of ~1,400 lines of numerics
will drift otherwise, and a silently-diverging second implementation is worse
than no port at all.

---

## Language

Deterministic, offline, **no model and no weights**.

| | |
|---|---|
| Action space | `abstain, answer, clarify, search, execute, explain` |
| Label sets | `language.intent` (7), `language.language` (4), `language.sentiment` (3), `language.task` (4) |
| Logic | `src/language/intent.cpp`, `src/language/sentiment.cpp` |

### Why this is lexical and not a neural head

Intent, language and sentiment for a chat surface are overwhelmingly decided by
**surface cues**: a trailing `?`, an opening interrogative, a greeting phrase, a
Unicode script range. A deterministic lexical classifier answers those in
microseconds, is debuggable, cannot hallucinate, and — the property that matters
most here — **works with no trained weights**. The neural `ClassificationHead` is
the upgrade path for the cases this cannot reach; it is not a prerequisite for the
common ones.

### Bilingual by construction, not by translation

Bengali is detected from its **own Unicode block** (U+0980–U+09FF) rather than
from a romanisation, and the lexicons carry Bengali entries alongside English
ones, so a Bengali greeting is recognised as a greeting *directly* instead of via
a lossy English hop.

Two Bengali-specific rules that an English-shaped classifier would get wrong:

* **Question words sit mid-sentence**, not at the front (`কেমন আছো?`).
* **The imperative is a suffix**, not an opener (`ফাইলটা দেখাও`) — Bengali is
  verb-final.

### Sentiment: negation is the whole problem

The rule is a **scoping** rule, not a window rule, because a plain "look back 3
tokens" window gets the most common English construction exactly backwards:

```
"not bad, it's good"  ->  a naive window sees "not" within 3 tokens of "good"
                          and flips it, scoring this NEGATIVE.
```

So a match looks back (and, for Bengali, forward) for a negator but **stops at the
nearest other sentiment word**. `"good"` finds `"bad"` first and stops, so it is
not negated; `"bad"` finds `"not"` and is. The forward pass exists because Bengali
negates post-positionally — `ভালো না` is "good not", i.e. not good — where English
negates pre-verbally.

**Negators are in neither lexicon.** Keeping `"no"` out of the negative list is
what makes `"no problem"` come out positive: `"problem"` is negative and `"no"`
flips it. `tests/test_language_heads.cpp` D7 asserts this **structurally**, because
if `"no"` ever appears in the negative list that behaviour silently reverses.

### Honesty: `confidence` here is not a probability

The language heads report a **heuristic strength** in [0,1] — how many independent
cues agreed. It is **not** a calibrated probability, and calling it "92% confident"
would be a lie of exactly the kind the calibration requirement exists to prevent.
Calibrating a lexical classifier requires labelled data and a fit; until that
exists, `confidence` is documented as a prior. See `docs/JEV_FEATURES.md`.

---

## Vision

| | |
|---|---|
| Action space | `abstain, describe, flag_anomaly, explain` |
| Label sets | `vision.scene` = `{indoor, outdoor, nature, urban, unknown}`, `vision.anomaly` = `{normal, unusual}` |
| Logic | **none** |

Registered, so a fitted projection would light these up immediately — but nothing
produces a vision feature vector into `h[E]` yet. Registering a label space is not
the same as having the capability.

---

## Audio

| | |
|---|---|
| Action space | `abstain, transcribe, respond, explain` |
| Label sets | `audio.wake` = `{yes, no}`, `audio.emotion` = `{happy, sad, angry, neutral}`, `audio.speaker` = `{known, unknown}` |
| Logic | **none** at the head level |

The audio front-end (`src/audio/*`, mel filters in `models/mel_filters.npz`) exists
from earlier work, but no audio feature vector is fed into the shared hidden state
for these heads to read.

---

## General

The fallback domain, and the one with real logic.

| | |
|---|---|
| Action space | `abstain, answer, route, explain` |
| Label sets | `general.routing` = `{trading, language, vision, audio, general}`, `general.priority` = `{urgent, normal, low}` |
| Logic | `ScoringHead` — `src/scoring_head.cpp` |

`general.routing` is the label set a *fitted* classification head would use to send
an input to a specialist domain; `HeadRouter`'s domain readout is the cheap
alternative that runs today.

`ScoringHead` returns `priority`, `urgency` and `confidence` as **three independent
sigmoids**, deliberately not a softmax: forcing them to sum to 1 would make a
genuinely urgent item necessarily unimportant. `tests/test_heads.cpp` C4 asserts
priority > 0.99 *and* urgency < 0.01 from the same hidden state.

---

## Adding a domain

The structure is designed so a new domain costs its rows and nothing else:

```cpp
UnifiedPipeline p;
p.init(768);

// 1. an action space for the decision head
p.decisions().add_domain(Domain::Vision, {"abstain", "describe", "explain"},
                         "abstain", "explain");

// 2. one or more label sets for the classification head
p.classifier().add_label_set("vision.scene", {"indoor", "outdoor", "unknown"});

// 3. fit, then act
p.decisions().set_action_row(Domain::Vision, 1, row, bias);
```

Duplicate names are **refused**, not overwritten: two callers registering
`"language.intent"` with different labels is a bug that should surface at startup,
not as a silently different distribution at runtime.

Until every row is fitted, `trained()` returns false and `provenance()` says
`UNTRAINED`. That is not a formality — it is the difference between a head that
means something and one that does not.
