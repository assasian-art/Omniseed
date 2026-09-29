# EDGE RESEARCH — can the learned heads beat chance?

Owner directive **D2**. The trading action head is at chance on held-out data,
which is *consistent with* the efficient-market hypothesis on daily bars. Rather
than keep tuning a head that has no signal to find, this document:

1. **Repositions the learned heads** to the roles the evidence actually supports.
2. Names the **rule-based engines** as the edge source.
3. Pre-registers a **research program** with a fixed adoption bar, and records
   **every rejection** — including the ones that have not run yet, so a future
   reader cannot mistake "untried" for "worked".

---

## 1. The adoption bar (pre-registered)

A candidate becomes a shipped head **only** if it clears, on a held-out split:

| criterion | value | why |
| --- | --- | --- |
| accuracy | `>= 2 x chance` | the §41 / DECISION 1 bar, reused, not reinvented |
| n_holdout | `>= 30` | below this a ratio is noise |
| the holdout | genuinely held out | verified, not assumed — see `docs/HOLDOUT_DEFECT.md` (the §43 "in-sample" claim was retracted; the fixture holds only holdout rows) |

`chance = 1/K`, with `K` the number of labels in the set.

**A rejection is a result and is recorded as one.** A program that only reports
its successes is a program that cannot be audited.

---

## 2. The reposition (D2, adopted now)

### 2.1 What the audit says

From `tools/signal_audit.py` (D1). These accuracies **are held out** — the fixture
contains only the holdout rows, verified four ways in `docs/HOLDOUT_DEFECT.md`
(the §43 "in-sample" claim was a false alarm and is retracted there):

| set | held-out accuracy | chance | ratio | verdict |
| --- | ---: | ---: | ---: | --- |
| `language.intent` | 0.6338 | 0.1429 | 4.44x | SIGNAL |
| `language.language` | 0.8451 | 0.2500 | 3.38x | SIGNAL |
| `trading.regime` | 0.7046 | 0.3333 | 2.11x | SIGNAL |
| `language.sentiment` | 0.6761 | 0.3333 | 2.03x | SIGNAL |
| `DecisionAction` | 0.2439 | 0.1429 | 1.71x | **NO-SIGNAL** |

The action head is the **only** fitted set that fails. Everything else clears the
bar on genuine held-out numbers.

### 2.2 The revised role of the learned heads

| head | OLD implied role | NEW role, effective now | basis |
| --- | --- | --- | --- |
| `trading.action` | directional prediction | **VETO / ABSTAIN filter only.** Never originates a position. | held-out accuracy 1.71x chance |
| `trading.regime` | a feature | **regime classifier, load-bearing.** May gate strategies. | 2.11x chance, n=369 |
| `language.*` | a feature | unchanged — genuine signal | 2.0–4.4x chance |
| `vision.*`, `audio.*` | — | **unfitted**; no weights exist | §40, §43 || `general.priority` | — | **stays unfitted permanently** | no objective label exists |

**The action head may only ever REMOVE exposure, never add it.** Concretely: given
a proposal from a rule engine, the action head can return ABSTAIN and the
proposal dies; it can never return BUY and create a proposal. This is enforceable
by construction — the head's output is a filter input, not a signal source — and
it is the honest reading of a 1.008x-chance classifier.

**This is not a retreat.** It is a head that is *useful* because it is
*correctly labelled*: a fail-closed ABSTAIN filter with a measured, published
false-positive rate is a real component. A directional predictor that is right
14% of the time is not.

### 2.3 The edge source: the rule-based engines

The interpretable, parity-tested engines remain the origin of every position:

| engine | test | status |
| --- | --- | --- |
| regime engine | `test_regime_engine.cpp`, `test_regime_parity.py` | parity vs Python oracle |
| strategy zoo | `test_strategy_zoo.cpp`, `test_strategy_parity.py` | parity + pinned defects |
| sniper | `test_strategy_zoo.cpp` D2 | **gate effectively closed** — max S 0.8317 < 0.85, 0 proposals |
| risk engine | `test_paper_session.cpp` | 2%/3%/6% C++-enforced |

Their advantage over a learned head is not accuracy — it is **auditability**.
Every rule can be read, argued with, and pinned by a test. That is why they own
the edge, and a chance-level head does not.

---

## 3. The research program (pre-registered, nothing adopted yet)

Each track below must clear §1's bar on a genuinely held-out split **before** it
is adopted. Tracks are ordered by expected information per unit of work.

### 3.1 Track A — longer horizons

**Hypothesis.** Daily-bar direction is dominated by noise; 5/10/21-day forward
returns may carry more structure than 1-day.

**Why it might fail.** Horizon smoothing raises the base rate problem, not
necessarily the signal. A 21-day label has ~21x the autocorrelation, so an
apparently higher accuracy can be pure label persistence — the number of
*independent* observations is far below `n`.

**Bar.** `>= 2x chance` AND `n_holdout >= 30` AND an **effective sample size**
accounting for overlap, reported alongside the raw `n`.

**Status: NOT RUN.** Requires re-dumping with a new teacher horizon.

### 3.2 Track B — monster features as head inputs

**Hypothesis.** `trend_score`, `ER`, `Hurst`, `OFI`, `funding` are computed by
the monster modules and are already interpretable; as extra input dimensions they
may give the head something the raw `h[E]` lacks.

**Why it might fail.** The head reads `h[E]`, which is the backbone's summary of
*text described bars*. The monster features are derived from the *numbers* the
text was rendered from. Appending them re-injects information the text renderer
deliberately discarded (`bar_to_text` keeps percent change, range, volume ratio).
That could help — or it could just make the head learn the renderer's loss.

**Bar.** §1, plus: the feature-only baseline (no `h[E]` at all) must be beaten,
or the head is learning the features and the backbone is dead weight.

**Status: NOT RUN.**

### 3.3 Track C — multi-asset pooled training

**Hypothesis.** 369 rows from one instrument is a very small fit; pooling many
instruments multiplies `n`.

**Why it might fail.** Asset-specific label distributions mean pooling can
*degrade* a per-asset head. And the regime engine's own thresholds are
instrument-independent by design, so the teacher may already impose the pooling
that is being proposed — in which case pooling adds `n` without adding
information.

**Bar.** §1, reported **per asset** as well as pooled. A pooled win that hides a
per-asset loss is not a win.

**Status: NOT RUN.** `models/market/*.csv` availability is limited; check first.

### 3.4 Track D — rank / quantile targets

**Hypothesis.** Class targets on `|r| > 1.5%` thresholds are brittle; predicting
a *rank* or *quantile* within a rolling window is a smoother target.

**Why it might fail.** A rank target is a different question, not an easier one.
The head would be scored on rank agreement, which is not what any consumer
(strategy zoo, sniper, risk engine) currently asks for. A high rank accuracy that
does not convert into a usable action is a metric, not an edge.

**Bar.** §1, plus a **conversion test**: the improved rank score must move a
downstream, already-tested engine's behaviour in a pre-registered direction.

**Status: NOT RUN.**

---

## 4. Rejection log

Every evaluated track is recorded here whether it passed or failed. **No track has
been evaluated yet** — the program is pre-registered, not partially executed.
This section exists so that the first rejection has an obvious home.

| # | track | variant | n_holdout | accuracy | chance | ratio | verdict | date |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| — | — | *(none evaluated yet)* | — | — | — | — | — | — |

### 4.1 Baseline, for reference (already rejected as an edge)

| candidate | n | accuracy | chance | ratio | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| `DecisionAction`, shipped blob, its own holdout (§32) | 369 | 0.2439 | 0.1429 | **1.707x** | NO-SIGNAL |
| `DecisionAction`, §41 refit inside the holdout | 111 | 0.1441 | 0.1429 | 1.008x | NO-SIGNAL (contaminated: trained on 258 holdout rows — see `docs/HOLDOUT_DEFECT.md`) |

**The shipped blob's own held-out row (0.2439, n=369) is the baseline every track
must beat.** It is the larger, cleaner number — §41's 0.1441 comes from a head
refitted on 258 rows that are themselves the holdout, so it is both smaller *and*
less trustworthy, and it must not be quoted as the head's accuracy.

---

## 5. What would make this document wrong

Stated so the reasoning can be attacked directly:

- **If a track clears 2x chance on a genuinely held-out split**, the action head
  is repositioned back toward a signal source — with the same bar applied to
  itself, in a later milestone, not this one. It is currently at **1.71x** on its
  own holdout, so the bar is not far off; the four tracks exist to see if anything
  closes it honestly.
- **If §41's threshold is ever re-derived from a clean fit** (see
  `docs/HOLDOUT_DEFECT.md` §"What was owed"), the 0.50 value may move. **RESOLVED
  (§44): it was re-derived from the shipped blob and the real dump, and the answer
  did not move — `min_confidence = 0.50`, the reached fallback.** The tracked
  numbers are now the blob's own (`T=13.325159`, `acc=0.243900`); the contaminated
  refit is history. The baseline row below is unchanged, only now verified by the
  tool as well as by the fixture.
- **If the rule-based engines are shown to be unprofitable in paper**, the
  "edge source" claim in §2.3 is wrong regardless of what the heads do. Their
  parity tests prove they *compute what they claim*; they do not prove the claim
  is profitable. That proof does not exist yet and is not claimed here.
