# OMNISEED — Unified Architecture

**One brain. Multiple heads. A smart router.**

Token generation and decision-making are not two systems. They are two *outputs*
of the same RWKV-7 forward pass. This document describes how that is wired, what
it costs, and — in the last section — what is still missing.

---

## 1. The shape of it

```
                                  input
                                    |
                     +--------------v---------------+
                     |   RWKV-7 backbone (0.1B)     |
                     |   12 layers, E = 768         |
                     |   ONE forward pass           |
                     +--------------+---------------+
                                    |
                              h[E]  (the hidden state)
                                    |
        +---------------------------+---------------------------+
        |                           |                           |
        v                           v                           v
   +---------+              +---------------+           +---------------+
   | ROUTER  |              | DECISION HEAD |           |  TOKEN HEAD   |
   | 9 matvec|              | [A, E] softmax|           |  the existing |
   | ~15 us  |              | ~1 matvec     |           |  generator    |
   +----+----+              +-------+-------+           +-------+-------+
        |                           |                           ^
        | ActivationPlan            | DomainDecision            |
        | {heads, mode, domain}     | {action, confidence}      |
        |                           |                           |
        +---------------------------+---------------------------+
                                    |
                    "decide first, then let the text EXPLAIN"
                                    |
                    +---------------v----------------+
                    |   CLASSIFY HEAD  [L, E] softmax |
                    |   SCORE HEAD     3 sigmoids     |
                    +---------------+-----------------+
                                    |
                                    v
                     UnifiedOutput -> ONE JSON document
```

Every head reads the **same** `h[E]`. A head is a projection
`[K, E] x [E] -> [K]` plus a squashing function, so the marginal cost of an extra
head is `K * E` multiply-accumulates — not a second forward pass. That is the
whole design.

---

## 2. Why a router, and not four always-on heads

Generating text is the only **expensive** head: it is a token loop whose cost
scales with `max_new_tokens`. Every other head is a single matvec. So a
structured decision that needs no explanation must never pay for a decode.

The router makes that choice. It costs one small projection to do it.

### The mandate's rule, in `HeadRouter::refine()`

| decision confidence | task | result |
|---|---|---|
| `>= 0.85` | simple | **decision_only** — the token head is dropped entirely |
| `>= 0.85` | not simple | decision+text — a task that also wanted a classification still gets an explanation |
| `< 0.85` | any | decision+text — the decision becomes **prefix context** for the text |

`refine()` can only ever **downgrade**. It removes the token head and never adds
it back. Both failure directions therefore fail safe:

* an uncertain router cannot talk itself into spending a decode;
* a confident one cannot suppress an explanation it was asked for.

`0.85` is shared by `DecisionHead::Config::threshold`, `HeadRouter::Config::
fast_path_confidence`, and the sniper's regime gate, so **no two layers can
disagree about what "confident" means**.

---

## 3. The key insight

> The Token Head **explains** decisions. Decisions **guide** tokens.

The decision head runs first and costs a matvec. If it is confident and the task
is simple, the pipeline returns the decision and stops. Otherwise the decision is
handed to the text producer as **prefix context**, so the generated explanation
is *about the decision that was actually made* rather than a second, independent
guess.

The producer is supplied as a callback (`UnifiedPipeline::run(hidden, text_fn)`)
because the generator already exists in `AgentLoop`/the CLI. This file must not
become a second copy of it.

---

## 4. The heads

| Head | File | Shape | Cost at E = 768 | Output |
|---|---|---|---|---|
| Router | `src/router.cpp` | `[4, E]` + `[5, E]` | 6,912 MACs, **15.3 us measured** | `ActivationPlan` |
| Decision (domain) | `src/domain_decision.cpp` | `[A, E]` | `A * E` | `DomainDecision` |
| Decision (trading) | `src/decision_head.cpp` | `[7, E]` | 5,376 MACs | `DecisionResult` |
| Classify | `src/classification_head.cpp` | `[L, E]` | `L * E` | `ClassificationResult` |
| Score | `src/scoring_head.cpp` | `[3, E]` | 2,304 MACs | `ScoreResult` |
| Token | `src/agent/*` | the existing generator | `max_new_tokens` forwards | text |
| Language | `src/language/*.cpp` | **no weights** — lexical | ~1 us | intent / lang / sentiment |

The router's 15.3 us is a measurement, not an estimate: `tests/test_router.cpp`
D1 runs 4000 calls and asserts the mandate's <100 us budget. It is worth knowing
*where* that time goes — the 6,912 MACs are not the dominant cost. The returned
`ActivationPlan` owns a `heads` vector and a `reason` string, and those two small
allocations cost more than the arithmetic. `plan()` therefore reserves both.

The classification head is **one** projection `[total_labels, E]` with each
registered label set owning a contiguous slice, so a new domain costs its labels'
rows and nothing else.

The scoring head is deliberately **three sigmoids, not a softmax**. Priority,
urgency and confidence are separate questions; forcing them to sum to 1 would
make a genuinely urgent item necessarily unimportant.
`tests/test_heads.cpp` C4 asserts a priority > 0.99 *and* an urgency < 0.01 from
the same hidden state — a test a softmax could not pass.

---

## 5. The unified output

`UnifiedOutput::to_json()` emits **one** document. Key order is the mandate's:

```json
{
  "router":         {"activated_heads":["decision","token"],"mode":"decision+text",
                     "domain":"trading","domain_confidence":0.41,
                     "confidence":0.63,"simple":true,"reason":"gates fired: decision token ",
                     "matvecs":9},
  "decision":       {"domain":"trading","action":"hold","confidence":0.42,
                     "margin":0.11,"routing":"system2","abstain":false,"fast_path":false},
  "text":           "I'm holding because the regime reads range and ...",
  "classification": {"domain":"general.routing","top_k":[{"label":"general","probability":0.31}],
                     "margin":0.04,"matvecs":5},
  "score":          {"priority":0.22,"urgency":0.18,"confidence":0.50,"matvecs":3}
}
```

### Rule 1 — present iff its head ran

A field is present **if and only if** its head is named in the activation plan.
`normalise()` drops anything the plan does not name and records the drop in
`warnings`. A stale decision from a previous turn leaking silently into this
turn's answer is the failure being prevented, and it is silent by nature.

Consequences:

* `"decision": null` never appears. Absent means "the head did not run"; it is
  not the same as "the head abstained", and a consumer that conflates them will
  eventually act on a field that was never computed.
* A head that ran but **refused** to produce anything is also not claimed. An
  unknown label set leaves `top_k` empty, and the field is omitted.

### Rule 2 — the fast path is asserted by invocation count

When the plan is `decision_only`, the text producer is **never called** — not
called-and-discarded. `token_head_invoked` records whether it ran, because a
timing measurement cannot distinguish "skipped" from "fast".
`tests/test_unified_output.cpp` C1 asserts the counter is exactly **0**.

---

## 6. Fail-closed, everywhere

| Situation | What happens | Why |
|---|---|---|
| Router not ready | plan = `{decision, token}`, mode `decision+text` | the safe **superset**, never a shortcut |
| No head gate fired | same superset, reason `no-gate-fired` | "no opinion" must not become "guess" |
| Decision confidence < threshold | routing `system2` (escalate) | do not act on a coin flip |
| Decision margin < floor | routing `system2` | 0.90/0.88 has a winner and no information |
| argmax is the abstain action | routing `abstain` | say nothing |
| Unknown classification set | `domain` = `<error: ...>`, `top_k` **empty** | a fabricated uniform distribution is indistinguishable from a real, useless one |
| Unregistered domain | routing `error` | "I have no action space for this" is information |
| Softmax overflows | max-subtract before `exp` | without it `inf/inf = NaN`, and `argmax` turns a NaN distribution into **action 0**, which is a real action |

---

## 7. Provenance: everything here is UNTRAINED

A freshly constructed head has **no trained projection**. Every head seeds its
weights deterministically (`splitmix64`, `1/sqrt(E)` scaling) so tests and CI are
reproducible, and every head exposes `trained()` / `provenance()` so a caller can
tell a fitted head from a placeholder.

> A seeded head emits **well-formed but meaningless** values.

`UnifiedPipeline::provenance()` reports `NOT FULLY FITTED (placeholder)` until
every head is fitted. Never present an untrained head's output as a judgement.
`docs/JEV_FEATURES.md` records what this means for the calibration requirement.

---

## 8. What is NOT done

Stated plainly, because a document that only lists successes is not a design
document:

* **No head is fitted.** The projections are placeholders. Wiring is complete;
  training is not.
* **Vision and audio have label spaces and no logic.** `vision.scene`,
  `vision.anomaly`, `audio.wake`, `audio.emotion`, `audio.speaker` are registered
  in the classification head, so a fitted projection would light them up — but
  nothing produces a vision or audio feature vector into `h[E]` yet.
* **The CLI does not use the router on its default path.** `omniseed chat --mode`
  still drives the existing `DecisionMode` path, because that path has a live
  consumer contract (`tools/decision_bridge.py` reads its JSON). Rewiring it is a
  behaviour change that needs its own parity gate, not a quiet refactor.
  `--mode text-only` is accepted as an alias for `off`.
* **The HTTP server does not expose `mode` yet**, and the dashboard does not read
  the unified document. Both are I/O-layer work.
* **No batch, streaming, or feedback-hook API.** See `docs/JEV_FEATURES.md`.
