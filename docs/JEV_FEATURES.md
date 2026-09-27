# OMNISEED — JEV Decision Capabilities: what is built, what is not

The mandate lists ten core decision capabilities and five domain head families.
This document records the **status of each**, with the evidence, and states the
gaps plainly. A feature list that only contains ticks is not a status report.

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

### 2. Calibrated confidence — **NOT STARTED** (the honest gap)

**Nothing in this tree is calibrated, and this document will not pretend
otherwise.** Concretely:

* Every head's projection is a **seeded placeholder**. `softmax` over an untrained
  projection produces a number in [0,1] that looks like a probability and is
  not one. Every head exposes `trained()` / `provenance()`, and
  `UnifiedPipeline::provenance()` reports `NOT FULLY FITTED (placeholder)`.
* The language heads' `confidence` is documented — in the header and in the code
  — as a **heuristic strength**: how many independent cues agreed. It is
  explicitly *not* a calibrated probability, and the header says that calling it
  "92% confident" would be a lie of exactly the kind the calibration requirement
  exists to prevent.
* Temperature scaling is **not implemented**, because temperature scaling needs
  (a) labelled data, (b) a fitted projection, and (c) a held-out split. None of
  the three exists yet. Implementing the scalar without the data would produce a
  calibrated-looking number with no justification — worse than the current honest
  placeholder.

What *is* in place is the machinery a calibration pass would need: per-head
fitting hooks (`set_action_row`, `set_label_row`, `set_row`, `set_head_gate`,
`set_domain_probe`), `fitted_rows()` counters, `trained()` gates, and a
`margin` on every distribution so a caller can at least see that a call was a
coin flip.

**Do not ship a decision on an untrained head's confidence.**

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

### 6. Batch processing — **NOT STARTED**

Every head takes a single `h[E]`. There is no batched forward pass and no
`[B, E]` input path. The reason is not oversight: the edge runtime is a
single-threaded scalar RWKV-7 at ~1.4 tok/s, where a batch would buy latency, not
throughput. A batch API belongs with the serving path, not with the heads.

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

### 10. Feedback hooks — **NOT STARTED**

No mechanism records an outcome and feeds it back. What exists is the *plumbing* a
feedback loop would write through — the per-head fitting hooks and their
`fitted_rows()` accounting — but nothing drives them, and there is no journal of
(prediction, outcome) pairs. Note that the trading path does have a frozen entry
journal (`tools/decision_bridge.py` re-reads it), which is the closest thing in
the tree to the substrate this would need.

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
nothing produces a vision or audio feature vector into `h[E]` yet. Registering
the label space is not the same as having the capability, and the table says so.

The trading action space mirrors `DecisionHead`'s exactly (lower-cased), so the
two heads cannot drift into disagreeing about what actions exist. That is asserted
in `tests/test_heads.cpp` D6.

---

## Part 3 — Known limitations, stated once

1. **Nothing is trained.** Every projection is a deterministic placeholder. See
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
