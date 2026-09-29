# EDGE RESEARCH — can the learned heads beat chance?

Owner directive **D2**. The trading action head is at chance on held-out data,
which is *consistent with* the efficient-market hypothesis on daily bars. Rather
than keep tuning a head that has no signal to find, this document:

1. **Repositions the learned heads** to the roles the evidence actually supports.
2. Names the **rule-based engines** as the edge source.
3. Pre-registers a **research program** with a fixed adoption bar, and records
   **every rejection**.

> **§45 update.** All four tracks have now been **RUN** (§3, §4) — every one is a
> rejection — and running them exposed that **the adoption bar itself was wrong**
> (§1.1): `chance = 1/K` is the no-skill rate only for *uniform* labels, and a
> constant predictor cleared it. The corrected floor is the **majority-class
> rate**, and under it **no fitted head in this document beats a constant**. The
> original `1/K` ratios are kept visible beside their corrections.

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

### 1.1 Correction (§45): `chance = 1/K` is wrong for a skewed label set

The bar above was **pre-registered and is now known to be defective.** Running
Track B (§3.2) produced a control — a constant "always BUY" predictor — that
scored **0.3686 = 2.580x chance** on the `trading.action` holdout. A predictor
that learns *nothing* cleared a bar meant to admit only real signal.

The cause: `1/K` is the no-skill rate **only when the labels are uniform**. The
7-action teacher's labels are not. `BUY` is 34% of the training rows; the
holdout's per-class base rates are:

| class | holdout n | base rate |
| --- | ---: | ---: |
| BUY | 136 | 0.369 |
| SELL | 78 | 0.211 |
| HOLD | 68 | 0.184 |
| EXPLAIN | 39 | 0.106 |
| ABSTAIN | 35 | 0.095 |
| HEDGE | 8 | 0.022 |
| CLOSE | 5 | 0.014 |

So the **no-skill floor for this label set is `max_class_rate = 0.3686`**, not
`1/7 = 0.1429`. The corrected bar is:

| criterion | was | corrected |
| --- | --- | --- |
| accuracy | `>= 2 x (1/K)` | `>= majority-class rate` (a no-skill predictor must score *exactly* 1.00x) |
| — | — | a candidate must also beat a **constant** predictor by a stated margin to be called SIGNAL at all |
| macro-F1 | not required | **reported, because it is the only metric here a constant cannot win** |

This is deliberately **not** retro-applied to silently change past results. The
original `1/K` ratios stay in the document, each now accompanied by its
majority-class equivalent, so a reader can see exactly which numbers moved.

**What the correction does to the record:**

| candidate | acc | old ratio (`1/7`) | new ratio (majority) |
| --- | ---: | ---: | ---: |
| constant "always BUY" | 0.3686 | 2.580x | 1.000x |
| features-only (Track B) | 0.3415 | 2.390x | 0.926x |
| `h[E]` + features (Track B) | 0.2900 | 2.030x | 0.787x |
| shipped `trading.action` | 0.2439 | 1.707x | **0.662x** |

**No candidate in this document clears the corrected bar.** The shipped head is
*below* the constant. The §2.2 reposition (VETO-only) is therefore not merely
conservative — it is the only role the evidence supports, and the "1.71x" that
motivated calling it "not far off the bar" was a floor artifact.

The other fitted sets are re-checked under the corrected floor in §4.2.

---

## 2. The reposition (D2, adopted now)

### 2.1 What the audit says

From `tools/signal_audit.py` (D1). These accuracies **are held out** — the fixture
contains only the holdout rows, verified four ways in `docs/HOLDOUT_DEFECT.md`
(the §43 "in-sample" claim was a false alarm and is retracted there):

| set | held-out accuracy | chance `1/K` | ratio (old) | majority rate | ratio (corrected) | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `language.intent` | 0.6338 | 0.1429 | 4.44x | 0.3239 | **1.957x** | SIGNAL (weak) |
| `language.language` | 0.8451 | 0.2500 | 3.38x | 0.4789 | **1.765x** | SIGNAL (weak) |
| `language.sentiment` | 0.6761 | 0.3333 | 2.03x | 0.5352 | **1.263x** | weak |
| `trading.regime` | 0.7046 | 0.3333 | 2.11x | **0.8699** | **0.810x** | **NO-SIGNAL** |
| `audio.emotion` | 0.3103 | 0.2500 | 1.241x | 0.2857 | **1.086x** | NO-SIGNAL |
| `DecisionAction` | 0.2439 | 0.1429 | 1.71x | 0.3686 | **0.662x** | **NO-SIGNAL** |

**The "ratio (old)" column is what this document published before §45 and it was
measured against the wrong floor** — see §1.1. The corrected floor is the
majority-class rate of each set's own holdout, computed by
`tools/edge_tracks.py`/§4.2. Read the corrected column.

The action head is **not** the only fitted set that fails. **`trading.regime` —
which §2.2 called "load-bearing, may gate strategies" — also fails**, and falls
*below* a constant predictor (0.810x): its 0.7046 accuracy is less than the
86.99% base rate of the single class `range`. Its holdout has **no `trend_down`
rows at all**, so 1 of its 3 labels is unrepresented and the metric is computed
over a degenerate label mix.

What survives the correction is the **language** heads (1.26x–1.96x) — real but
modest, and well short of the 2x the old table appeared to show.

### 2.2 The revised role of the learned heads

**REVISED AGAIN IN §45 (§1.1): the basis column below quoted the old `1/K` ratio.
Under the corrected majority-class floor, `trading.regime` and `trading.action`
both fall BELOW a constant predictor, and only the language heads clear 1.0x.**

| head | OLD implied role | NEW role, effective now | basis (corrected §45) |
| --- | --- | --- | --- |
| `trading.action` | directional prediction | **VETO / ABSTAIN filter only.** Never originates a position. | 0.662x majority — *below a constant* |
| `trading.regime` | a feature | **NOT load-bearing.** Demoted to a diagnostic feature, not a gate. | 0.810x majority — *below a constant* |
| `language.*` | a feature | unchanged — genuine signal, but modest | 1.26–1.96x majority |
| `vision.*`, `audio.*` | — | `audio.emotion` **fitted but weak**; `vision.*` unfitted | §44b / §40 |
| `general.priority` | — | **stays unfitted permanently** | no objective label exists |

**The action head may only ever REMOVE exposure, never add it.** Concretely: given
a proposal from a rule engine, the action head can return ABSTAIN and the
proposal dies; it can never return BUY and create a proposal. This is enforceable
by construction — the head's output is a filter input, not a signal source — and
it is the honest reading of a classifier that scores *below* the base rate.

**This is not a retreat.** It is a head that is *useful* because it is
*correctly labelled*: a fail-closed ABSTAIN filter with a measured, published
false-positive rate is a real component. A directional predictor that is right
less often than "always BUY" is not.

**The `trading.regime` demotion is new in §45.** §2.2 originally kept it
"load-bearing" on a 2.11x ratio. That ratio used `1/K`; against the majority
class its 0.7046 accuracy is 0.810x — the head does **worse than always
predicting `range`**, which is 86.99% of its holdout. It may still be *read* as a
feature (the regime engine's rule-based output is a separate, parity-tested
thing), but the *learned* regime head must not gate anything.

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

## 3. The research program (pre-registered, all four RUN, none adopted)

Each track below must clear §1's bar on a genuinely held-out split **before** it
is adopted. All four have now been **RUN** (§45); every one is a rejection, and
running them found the bar itself defective (§1.1).

### 3.1 Track A — longer horizons

**Hypothesis.** Daily-bar direction is dominated by noise; 5/10/21-day forward
returns may carry more structure than 1-day.

**Why it might fail.** Horizon smoothing raises the base rate problem, not
necessarily the signal. A 21-day label has ~21x the autocorrelation, so an
apparently higher accuracy can be pure label persistence — the number of
*independent* observations is far below `n`.

**Bar.** `>= 2x chance` AND `n_holdout >= 30` AND an **effective sample size**
accounting for overlap, reported alongside the raw `n`.

**Status: RUN — REJECTED.** No horizon clears the bar (`tools/edge_tracks.py`,
Track A). `h[E]` does **not** depend on the teacher horizon, so this was a pure
re-label of the 1,231 dumped rows — **no re-dump was needed**, contrary to the
pre-registration's guess.

| horizon | n_holdout | n_eff (`n//h`) | accuracy | vs `1/K` | majority rate | vs majority | verdict |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 369 | 369 | 0.2385 | 1.669x | 0.3604 | 0.662x | NO-SIGNAL |
| 5 | 369 | 73 | 0.2439 | 1.707x | 0.3686 | 0.662x | NO-SIGNAL |
| 10 | 369 | 36 | 0.2547 | 1.783x | 0.4146 | 0.614x | NO-SIGNAL |
| 21 | 369 | 17 | 0.2087 | 1.461x | **0.5366** | 0.389x | NO-SIGNAL |

Note the **majority rate rises with the horizon** (0.3604 → 0.5366): the
longer the forward window, the more the teacher's labels pile into one class, so
the no-skill floor rises while the head's accuracy does not. By `h=21` a constant
predictor would score **53.66%** against the head's 20.87%.

Two things worth recording, because both were *predicted wrong*:

1. **`h=5` reproduces the shipped blob bit-for-bit** — `acc=0.24390`,
   `T=13.325159204929149`, `ECE=0.0555`. This is the harness's own control: if
   Track A did not regenerate the shipped head exactly, it would be measuring a
   different pipeline. It does, so the other three rows are trustworthy.
2. **A longer horizon does NOT help, and the longest horizon clearly hurts.**
   The best row is `h=10` at 1.783x — still 11% short of the bar, and its
   over-lap-adjusted `n_eff` is only 36, barely above the 30 floor. At `h=21`
   accuracy *falls* to 0.2087. This is exactly the failure mode §3.1 named:
   smoothing raises the base rate, not the signal, and by `h=21` the label is
   mostly persistence of the prior move.

**Adopted: nothing.**

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

**Status: RUN — REJECTED, and it exposed a defect in §1 itself.** The track
failed its own pre-registered condition *and* the control that checks the
condition then failed, which turned a "the features won" result into "the bar
cannot tell a win from a constant".

| variant | n_holdout | accuracy | vs `1/7` | vs majority | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| `h[E]` only (shipped control) | 369 | 0.2439 | 1.707x | **0.662x** | NO-SIGNAL |
| features only (no `h[E]`) | 369 | 0.3415 | **2.390x** | 0.926x | REJECT (base-rate artifact) |
| `h[E]` + features | 369 | 0.2900 | 2.030x | 0.787x | REJECT (did not beat features-only) |

The features-only row **appears** to clear the 2x bar. It does not. Three
controls, all computed *inside* `tools/edge_tracks.py` and stored in the JSON,
each destroy it:

| control | accuracy | what it proves |
| --- | ---: | --- |
| features, **labels shuffled** | 0.3144 | a model that cannot see the labels scores 2.20x |
| features, **columns permuted across rows** | 0.3333 | a model whose features are scrambled scores 2.33x |
| features, real | 0.3415 | the "signal" |
| **constant "always BUY"** | **0.3686** | the majority class alone scores **2.580x** |

**A predictor that learns nothing scores 2.580x chance.** That is the finding.
It means `chance = 1/K` in §1 is **the wrong floor for a skewed label set**: the
7-action teacher's `BUY` is 34% of the train rows and 37% of the holdout, so the
true no-skill rate is the majority-class rate `0.3686`, not `1/7 = 0.1429`. Every
ratio in this document was measured against a floor ~2.6x too low.

**Why the features-only head scored 0.3415 and not 0.3686:** it is a real (if
weak) fit that pulls a little accuracy *away* from the constant and trades it for
macro-F1. That reading is confirmed below.

**Adopted: nothing.** And the deeper consequence — the baseline itself is
suspect — is recorded in §4.2. The tool now refuses to adopt a row that passes
`2x1/K` but not the majority floor: the features-only verdict in the JSON is
`REJECT (passes 2x1/K but not the majority-class floor 0.3686 — base-rate
artifact)`, asserted by the gate.

### A note the skew forces: accuracy and macro-F1 disagree, and only one is honest

| predictor | accuracy | vs majority | macro-F1 |
| --- | ---: | ---: | ---: |
| constant "always BUY" | 0.3686 | 1.000x | 0.0769 |
| shipped `trading.action` head | 0.2439 | 0.662x | **0.1333** |

On accuracy the shipped head **loses to a constant**. On macro-F1 it **beats**
it (0.1333 vs 0.0769) — i.e. it genuinely emits more than one class and is not a
degenerate constant. Both statements are true and neither alone is the whole
truth. The honest summary is: **the head is a weak, real discriminator whose
`argmax` accuracy is dragged below the base rate by a skewed teacher.** That is
consistent with §2.2's reposition to a fail-closed VETO filter, and it is *not*
consistent with the 1.707x "ratio" §2.2 quoted, which measured the head against
the wrong floor.

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

**Status: RUN — REJECTED (but pooling is real).** `models/market/*.csv` held
**two** instruments, not one, so the track ran: both were dumped to `h[E]` and
pooled. Pooling **raised both** single-asset heads (AAPL 0.2439→0.3062, DEMO
0.2749→0.3127), so the hypothesis survived — but the pooled absolute accuracy
(0.3095) is still below a constant (0.3689). Result and tables in §4.3.

### 3.4 Track D — rank / quantile targets

**Hypothesis.** Class targets on `|r| > 1.5%` thresholds are brittle; predicting
a *rank* or *quantile* within a rolling window is a smoother target.

**Why it might fail.** A rank target is a different question, not an easier one.
The head would be scored on rank agreement, which is not what any consumer
(strategy zoo, sniper, risk engine) currently asks for. A high rank accuracy that
does not convert into a usable action is a metric, not an edge.

**Bar.** §1, plus a **conversion test**: the improved rank score must move a
downstream, already-tested engine's behaviour in a pre-registered direction.

**Status: RUN — REJECTED, twice over.** The tertile label is **balanced by
construction** (410 / 391 / 419 rows), so unlike the action set its floor
`chance = 1/3` is honest — and the head does not reach it.

| quantity | value | meaning |
| --- | ---: | --- |
| n_holdout | 366 | (3 rows lost to <30 history) |
| accuracy | 0.2951 | **0.885x chance — below chance** |
| label mix | 410 / 391 / 419 | balanced, so `1/3` is a valid floor |
| conversion separation | **−0.081** | predicted-top rows have LOWER realised return than predicted-bottom → **inverted** |

Both halves of the pre-registered test fail: the accuracy is below chance, and
the conversion direction is **reversed** (NOT ordered). This is the §3.4 failure
mode in its most direct form: *"a rank target is a different question, not an
easier one"* — here it is a question the head answers worse than a coin.

**Adopted: nothing.**

---

## 4. Rejection log

Every evaluated track is recorded here whether it passed or failed. **All four
tracks have now been evaluated (§45, `tools/edge_tracks.py`); every one is a
rejection**, and running them exposed a defect in the bar itself (§1.1).

| # | track | variant | n_holdout | accuracy | vs `1/K` | majority | vs majority | verdict |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | A | `n_forward=1` | 369 | 0.2385 | 1.669x | 0.3604 | 0.662x | REJECT |
| 2 | A | `n_forward=5` (shipped) | 369 | 0.2439 | 1.707x | 0.3686 | 0.662x | REJECT |
| 3 | A | `n_forward=10` | 369 | 0.2547 | 1.783x | 0.4146 | 0.614x | REJECT (best; still < floor) |
| 4 | A | `n_forward=21` | 369 | 0.2087 | 1.461x | 0.5366 | 0.389x | REJECT (worse) |
| 5 | B | `h[E]` only (control) | 369 | 0.2439 | 1.707x | 0.3686 | 0.662x | REJECT |
| 6 | B | features only | 369 | 0.3415 | 2.390x | 0.3686 | 0.926x | REJECT (base-rate artifact) |
| 7 | B | `h[E]` + features | 369 | 0.2900 | 2.030x | 0.3686 | 0.787x | REJECT (did not beat features-only) |
| 8 | C | pooled AAPL+DEMO | 740 | 0.3095 | 2.166x | 0.3689 | 0.839x | REJECT (base-rate artifact; but pooling helps BOTH assets) |
| 9 | D | forward-return tertile (5d) | 366 | 0.2951 | — | — | 0.885x (`1/3`) | REJECT (below chance + conversion inverted) |

Adoption: **nothing adopted.** All four tracks are complete. Track D's floor is
`1/3` because its label is balanced by construction, so the majority column is not
meaningful for it — the row shows the `1/3` ratio instead.

### 4.1 Baseline, for reference (already rejected as an edge)

| candidate | n | accuracy | vs `1/K` | vs majority | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| `DecisionAction`, shipped blob, its own holdout (§32) | 369 | 0.2439 | 1.707x | **0.662x** | NO-SIGNAL |
| `DecisionAction`, §41 refit inside the holdout | 111 | 0.1441 | 1.008x | 0.391x | NO-SIGNAL (contaminated: trained on 258 holdout rows — see `docs/HOLDOUT_DEFECT.md`) |
| **constant "always BUY"** (no model) | 369 | **0.3686** | 2.580x | **1.000x** | the true floor |

**The number every track must beat is not 0.2439 — it is the constant 0.3686.**
The shipped blob's own row (0.2439) is *below* it. This is the §1.1 correction
applied to the baseline itself.

### 4.2 The bar was defective — the corrected re-check of every fitted set

The single most important result of §45 is **not** a track outcome; it is that
the control for Track B (`a constant predictor`) scored **2.580x `1/K`**. Applied
to all six fitted sets, using each set's own holdout majority-class rate:

| set | acc | old ratio | majority rate | corrected ratio | who it was |
| --- | ---: | ---: | ---: | ---: | --- |
| `language.intent` | 0.6338 | 4.44x | 0.3239 | **1.957x** | SIGNAL, modest |
| `language.language` | 0.8451 | 3.38x | 0.4789 | **1.765x** | SIGNAL, modest |
| `language.sentiment` | 0.6761 | 2.03x | 0.5352 | **1.263x** | weak |
| `audio.emotion` | 0.3103 | 1.241x | 0.2857 | **1.086x** | NO-SIGNAL |
| `trading.regime` | 0.7046 | 2.11x | **0.8699** | **0.810x** | **below a constant** |
| `trading.action` | 0.2439 | 1.707x | 0.3686 | **0.662x** | below a constant |

**Consequences, stated plainly:**

1. **No fitted set clears 2x anything.** The best (`language.intent`) is 1.96x —
   and that is against a floor, not against a constant predictor, which is an
   even harder control not yet computed here.
2. **`trading.regime` is demoted** (§2.2): 0.810x a constant, on a holdout that
   is 86.99% one class and contains **no `trend_down` rows**.
3. **The §2.2 "not far off the bar" language was wrong.** It relied on 1.707x;
   corrected, the head is at 0.662x — further from a *useful* head, not closer.
4. **`audio.emotion` (§44b) is confirmed weak but honest** — 1.086x, and unlike
   the trading sets its label mix is near-balanced (2:2:2:1), so its floor is
   real. It is *not* an artifact; it is simply small.

**The mechanism, named:** `accuracy = argmax` on a skewed label set is dominated
by the base rate. A model that emits the majority class always wins the accuracy
metric and learns nothing. The **macro-F1**, which the trainer already reports,
is the metric a constant *cannot* win — and it is the one §2's verdicts should
have led with. Both metrics are now reported together everywhere.

### 4.3 Track C — multi-asset pooled training

Run on the two instruments present (`AAPL_1d.csv`, `DEMO_1d.csv`), each dumped to
`h[E]` (1,231 and 1,236 rows), split chronologically **per source** and pooled.

**Status: RUN — REJECTED, but the most interesting of the four.** The pooled head
*does* lift both single-asset heads — a genuine relative gain — yet it still does
not beat a constant, so the corrected bar refuses it.

| head | n_holdout | accuracy | vs `1/K` | majority | vs majority | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| pooled (AAPL+DEMO) | 740 | 0.3095 | **2.166x** | 0.3689 | **0.839x** | REJECT (base-rate artifact) |
| pooled head on AAPL | 369 | 0.3062 | 2.144x | 0.3686 | 0.831x | below constant |
| pooled head on DEMO | 371 | 0.3127 | 2.189x | 0.3693 | 0.847x | below constant |

**The per-asset comparison is the part that matters** (§3.3: "a pooled win that
hides a per-asset loss is not a win" — here there is no hidden loss):

| instrument | single-asset head | pooled head | pooling helps? |
| --- | ---: | ---: | --- |
| AAPL | 0.2439 | **0.3062** | **yes (+0.062)** |
| DEMO | 0.2749 | **0.3127** | **yes (+0.038)** |

So **pooling is real**: it raises both instruments' held-out accuracy, and the
lift is larger for the asset with less signal (AAPL). This is the one track whose
*hypothesis* survived contact — multi-asset data genuinely adds information that
one instrument does not carry. But the *adoption bar* is absolute, and the pooled
head (0.3095) is still **below a constant predictor (0.3689)**: pooling moves the
head toward the floor without reaching it. **Adopted: nothing** — but this is the
first track a *future* run should revisit, with a target that is not the skewed
7-action teacher (see §5).

The pre-registration's second warning also held: the regime teacher is
instrument-independent, so part of the pooling gain is the teacher, not the
instruments — which is why the absolute number lands near the base rate rather
than above it.

### 4.4 Track D — rank/quantile targets

Tertile label, **balanced by construction** (410 / 391 / 419), so `chance=1/3` is
an honest floor. `acc=0.2951` = **0.885x chance — below chance** — and the
pre-registered conversion test **inverts**: predicted-top rows carry a *lower*
mean realised rank than predicted-bottom (separation **−0.081**). Both halves of
the §3.4 test fail. Recorded in the §4 log as row 9.

---

## 5. What would make this document wrong

Stated so the reasoning can be attacked directly:

- **The 2x-`1/K` bar was wrong and is retracted (§1.1).** A constant predictor
  scored 2.580x it. Any surviving claim in this document that quotes a `1/K`
  ratio must be re-read against the majority-class floor; §2.1 and §2.2 have been
  corrected in place, with the old numbers kept visible.
- **Under the corrected floor, nothing passes — including the baseline.** If a
  future track clears the *majority-class* bar (or, harder, beats a constant
  predictor on macro-F1) on a genuinely held-out split, the action head is
  repositioned back toward a signal source — in a later milestone, not this one.
  All four tracks were run (§4) and none did.
- **Track C is the one to revisit, and it should be re-run on a different
  teacher.** Pooling AAPL+DEMO raised *both* assets' held-out accuracy (§4.3) —
  the only track whose hypothesis survived. The reason it still failed the bar is
  most likely the label: the 7-action teacher is a skewed, instrument-independent
  rule, so a pooled head converges on the base rate. Re-running pooling against a
  target that is **not** the skewed action teacher (e.g. a balanced one) is the
  obvious next experiment, and it is *not* claimed to work here.
- **If §41's threshold is ever re-derived from a clean fit** (see
  `docs/HOLDOUT_DEFECT.md` §"What was owed"), the 0.50 value may move. **RESOLVED
  (§44): it was re-derived from the shipped blob and the real dump, and the answer
  did not move — `min_confidence = 0.50`, the reached fallback.** The tracked
  numbers are now the blob's own (`T=13.325159`, `acc=0.243900`); the contaminated
  refit is history.
- **The `trading.regime` demotion (§2.2) is the one that most changes behaviour.**
  If its 0.810x-a-constant result is wrong, a gate that this document just closed
  should reopen. The number is computed on the 369-row holdout, whose mix is
  321 `range` / 38 `high_vol` / 10 `trend_up` (no `trend_down`); the reader can
  check it from `tests/fixtures/head_calibration/trading/labels.tsv`.
- **If the rule-based engines are shown to be unprofitable in paper**, the
  "edge source" claim in §2.3 is wrong regardless of what the heads do. Their
  parity tests prove they *compute what they claim*; they do not prove the claim
  is profitable. That proof does not exist yet and is not claimed here.
