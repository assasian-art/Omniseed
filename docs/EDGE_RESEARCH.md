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
>
> **§46 update.** Two follow-ups, both in §6. (i) The bar is corrected a second
> time — `acc > majority` is cleared by *noise*, so the operative bar is
> **`acc >= 2 x majority`** (§1.2). (ii) The surviving hypothesis (rebuild the
> teacher balanced) was **tested and failed**: a balanced teacher lowered the
> floor 0.3686 → 0.3198 and moved the head not at all, and a **perfectly
> balanced 3-class directional control sits below `1/3`** — so the failure is the
> absence of directional signal in `h[E]`, not the skew.

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

### 1.2 Correction (§46): the floor is right, but the *comparison* was still too weak

§1.1 fixed the floor (`1/K` → majority-class rate) but left the comparison as a
bare **`acc > majority`**. That is not a bar, it is a coin flip with a tiebreak:
`>` is cleared by **noise**. The balanced-teacher rebuild (§6) produced exactly
that counterexample —

> Track B's features-only head, under the balanced teacher, scores **0.3225**
> against a constant of **0.3198**. It clears `acc > majority`. Its margin is
> **0.0027, or z = 0.11** — one tenth of a standard error. A bare `>` bar would
> have adopted it.

So the bar is corrected a second time: the **`2x` multiplier applies to the
majority floor**, not just to `1/K`.

| criterion | §1.1 | §1.2 (operative) |
| --- | --- | --- |
| accuracy | `> majority-class rate` | **`>= 2 x majority-class rate`** |
| n_holdout | `>= 30` | `>= 30` (unchanged) |
| reported alongside | `ratio` vs `1/K` | `ratio` vs `1/K` **and** `ratio_vs_floor`, `macro_f1`, and `constant_z` |

For a **balanced** label set `majority = 1/K`, so §1.2 reduces to the original
pre-registration — the change only bites on skewed sets, which is where it
matters. `constant_z = (acc - majority) / sqrt(majority*(1-majority)/n)` is
reported so a reader can see whether a "win" over the constant is real or noise;
it is **diagnostic only** and never decides adoption.

The tool applies this bar in **one place** (`edge_tracks.py::_verdict`), and the
gate asserts that *every* row's `adopted` flag agrees with it (§A7 of
`tests/test_edge_tracks_gate.py`). Under §1.2 **no candidate in this document is
adopted** — including every balanced-teacher row (§6).

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

**ENFORCED IN CODE (§46).** "Must not gate" was, until §46, a sentence in a
document. It is now a property of the type system —
`include/omniseed/trading/regime_authority.h`:

- A strategy gate accepts a `GatingRegime`, never a bare label string. A
  `GatingRegime` carries its **provenance**, and `can_gate()` is true for exactly
  two sources: `RuleEngine` (the multi-axis `RegimeEngine`/`RegimeTracker`) and
  `LegacyRule` (the deterministic SMA-slope rule).
- `GatingRegime::from_learned(label)` — how a fitted head's output would enter —
  produces a token whose `can_gate()` is **false**, and `refusal_reason()` names
  the demotion. `from_external()` is false too.
- The single veto decision, `regime_veto()`, begins with `if (!gate.authorize())
  return "";`. That one line is the whole enforcement: a label without authority
  blocks nothing, whatever it says.
- `from_engine()` refuses a hand-built `RegimeState` (it checks
  `RegimeState::engine_minted`, set only by `RegimeEngine::detect()`), so a
  learned label cannot be laundered by assembling a struct around it.
- In `SniperVerdict`, an unauthorised label is additionally **neutralised** —
  `regime` is reset to `"range"` and `regime_score` to `0.5` — so it cannot move
  the score either. `regime_gated` / `regime_source` record which happened.

`tests/test_regime_authority.cpp` (786 checks, `omniseed_regime_authority`) pins
this in both directions and is non-vacuous: it runs the **real fitted
`models/heads/trading_regime_head.bin` over the committed 369-row holdout**, finds
**22 rows** where the head emits a label (`trend_down`/`high_vol`) that *would*
veto, asserts all 22 are refused — and asserts the identical labels through the
rule factory *do* veto. Removing the authority check turns 27 checks red.

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
  **RESOLVED (§46, §6.6): it was re-run against the balanced teacher. Pooling
  still lifts both instruments under an accuracy-shaped loss, but the lift is
  *not* robust to a balanced (inverse-frequency) loss, and the pooled head is
  still below the floor. "Survived contact" is downgraded accordingly.**
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

---

## 6. The balanced-teacher rebuild (§46) — the surviving hypothesis, tested

§4.3 named the one experiment worth running: *"re-run pooling against a target
that is **not** the skewed action teacher"*. §5 repeated it. This section is that
run, and its answer is **no** — the skew was not what was hiding the signal.

### 6.1 The hypothesis

The §45 correction left one honest escape hatch: maybe the head failed only
because the teacher's labels are skewed (37% `BUY`), so a constant predictor
occupied the bar and a real-but-weak signal was buried under the base rate.
If so, **rebuilding the teacher balanced should let the head through.**

### 6.2 The balanced teacher (v2)

Same 7-action vocabulary (`K` unchanged, so the head is the same shape), but
every threshold is a **trailing-window quantile of past forward returns** instead
of a fixed constant (`tools/train_heads.py::build_action_labels_balanced`, doc
string `BALANCED_TEACHER_DOC`). The window ends at `i-1`, so there is no
look-ahead. The directional branch (BUY/SELL/HOLD) is bucketed by the 1/3 and 2/3
quantiles, which equalises it **by construction**.

| teacher | holdout h=5 mix | majority floor |
| --- | --- | ---: |
| v1 (shipped, fixed ±1.5%) | BUY 136 / SELL 78 / HOLD 68 / EXPLAIN 39 / ABSTAIN 35 / HEDGE 8 / CLOSE 5 | **0.3686** |
| v2 (balanced, quantile) | (over all 1,231 rows: BUY 370 / SELL 335 / HOLD 240 / EXPLAIN 116 / ABSTAIN 75 / HEDGE 51 / CLOSE 44) | **0.3198** |

The floor **fell** (0.3686 → 0.3198) — the rebalance is real. It did **not** reach
`1/K = 0.1429`, and it cannot: the four regime-conditional actions (CLOSE, HEDGE,
ABSTAIN, EXPLAIN) inherit the regime engine's distribution, and `range` is 85% of
the tape. That residual skew is **inherent to the 7-action structure** and is
reported, not hidden — it is also why the bar (§1.2) must be the majority floor,
not `1/K`.

### 6.3 Result: the head does not move

`tools/edge_tracks_balanced.json` (teacher=balanced, unweighted — the
apples-to-apples comparison):

| track | v1 acc (ratio_vs_floor) | v2 acc (ratio_vs_floor) | adopted |
| --- | ---: | ---: | --- |
| A h=1 | 0.2385 (0.662x) | 0.1680 (0.544x) | no |
| A h=5 | 0.2439 (0.662x) | 0.2385 (0.746x) | no |
| A h=10 | 0.2547 (0.614x) | 0.2276 (0.689x) | no |
| A h=21 | 0.2087 (0.389x) | 0.2060 (0.547x) | no |
| B `h[E]` only | 0.2439 (0.662x) | 0.2385 (0.746x) | no |
| B features only | 0.3415 (0.926x) | 0.3225 (1.008x) | no |
| C pooled | 0.3095 (0.839x) | 0.3095 (0.902x) | no |

**Balancing the teacher moved the head by nothing.** The v2 `h[E]`-only accuracy
(0.2385) is *the same head* on a *fairer* target, and it is still **0.75x a
constant**. The hypothesis is dead: the failure is not the skew.

**It also produced the counterexample that forced §1.2.** Track B's features-only
head now scores 0.3225 against a constant of 0.3198 — it *clears* `acc > majority`
by **z = 0.11**. A bare `>` bar would have adopted it. That is why the bar is
`>= 2 x majority` (§1.2), and why the tool now records `constant_z`.

### 6.4 The re-weighted arm (the "(or explicitly re-weighted)" alternative)

`tools/edge_tracks_balanced_reweighted.json` re-fits with **inverse-frequency
class weights** (`--class-weight inverse`), i.e. the loss is macro-F1-shaped
rather than accuracy-shaped. Every verdict is unchanged (all rejected); the
weights do what they claim (the controls now sit *below* chance — labels-shuffled
0.1165, features-permuted 0.1247 — because a weighted fit cannot win by guessing
the majority), and macro-F1 rises (Track E 0.2667 → 0.2730) without accuracy
reaching the floor. Re-weighting is not a substitute for signal.

### 6.5 The decisive control — Track E

The sharpest test is not the 7-action teacher at all. **Track E** labels each bar
by its forward-return tertile against a trailing window, so the label mix is
~1/3 **by construction** and the by-construction floor is **exactly `1/3`** —
there is no majority-class ambiguity left to correct for.

> **Track E: acc = 0.2951, macro-F1 = 0.2667, against a 1/3 floor.**
> **Below chance** — and below the chronological holdout's own majority (0.3716).

A head that cannot beat `1/3` on a perfectly balanced directional target is not
suffering a skew artifact. **`h[E]` does not linearly encode the forward
direction of the bar.** That is the result, and it is consistent with the
efficient-market reading in the preamble.

### 6.6 Track C revisited (the §4.3 hypothesis)

Pooling AAPL+DEMO against the balanced teacher, with the §1.2 bar:

| instrument | single-asset head | pooled head | pooling helps? |
| --- | ---: | ---: | --- |
| AAPL | 0.2385 | 0.2737 | yes (+0.035) |
| DEMO | 0.2399 | 0.3450 | yes (+0.055) |
| pooled | — | 0.3095 (0.902x floor) | **not adopted** |

Pooling still lifts **both** instruments, and the pooled head is still below the
floor. Under the re-weighted loss the AAPL lift **disappears** (0.2466 pooled vs
0.2493 single) — so the pooling gain is **not robust to the loss**, and §4.3's
"survived contact" is downgraded to *"survives an accuracy-shaped loss, not a
balanced one."* Recorded as a correction, not buried.

### 6.7 Outcome

| question | answer |
| --- | --- |
| Was the §45 failure a label-skew artifact? | **No** (§6.3, §6.5) |
| Did the balanced teacher let anything through? | **No** — 0 of 14 rows adopted |
| Does re-weighting rescue it? | **No** (§6.4) |
| Does `h[E]` encode forward direction at all? | **No** — below `1/3` on a balanced target (§6.5) |
| Newly fitted blob to re-audit? | **None.** No blob was fitted or shipped; the balanced teacher is an experiment (`--teacher balanced`), not a new artifact. D1 (`tools/signal_audit.py`) and DECISION 1 (`derive_threshold.py`) therefore need **no** re-run — their inputs are unchanged. |

Artifacts: `tools/edge_tracks.json` (v1), `tools/edge_tracks_balanced.json` (v2),
`tools/edge_tracks_balanced_reweighted.json` (v2 + inverse weights). All three are
regenerated byte-for-byte by the gate (`tests/test_edge_tracks_gate.py`).

### 6.8 The intent head as a tie-breaker — pre-registered, REJECTED (§46.9)

Under the §1.2 floor `language.intent` sits at 0.6338 / 0.3239 = **1.957x** — below
the `2x` bar, above a constant. That is the one band where a head is neither
adoptable nor obviously worthless, so the remaining role worth testing is a
**tie-breaker**: consult the head only where the deterministic lexical classifier
is already unsure.

**Pre-registered before the run** (`tests/test_intent_tiebreak.cpp`, offline, real
blob + the committed 71-row held-out fixture):

```
m   = lexical top-1 - lexical top-2            TAU = 0.10   (one value, not tuned)
m >= TAU : lexical stands
m <  TAU : lexical stands UNLESS head_top1 > lexical_top1, then take the head
ADOPT iff (combined > lexical) AND (gain >= 2 rows)   <-- the noise guard
```

Both clauses are part of the pre-registration. The TAU curve is **reported, never
selected from** (selecting from it would be §41's defect).

| arm | accuracy | correct |
| --- | ---: | ---: |
| lexical only | **0.7465** | 53/71 |
| fitted head only | 0.6338 | 45/71 |
| combined (tie-break) | 0.7606 | 54/71 |

Consulted on **3** rows, overrode **1** (helped 1, hurt 0) ⇒ **gain = 1 row**,
discordant = 1, exact two-sided sign test **p = 1.0000**.

**Verdict: clause 1 FIRED, clause 2 did NOT ⇒ REJECTED.** The point estimate
improves by a single row out of 71; adopting on that is the §45 defect (a bar
cleared by noise). The rejection is recorded, not resolved by relaxing the guard.

Two free by-products: the head-only arm **re-derives** the D1 table's
`language.intent` holdout accuracy (0.6338 = 45/71) from the fixture rather than
copying it; and the lexical classifier **beats** the fitted head (0.7465 vs
0.6338), which is what the design predicts — the language domain is lexical on
purpose.

Curve (reported, not selected from): TAU 0.05/0.10/0.20/0.30 → 0.7606 (consulted
3/3/4/4); TAU 0.50 → 0.7465 (27); TAU 1.01 → 0.7465 (71).

**Non-vacuity, both directions:** corrupting `kMinGain` 2 → 1 flips the verdict to
ADOPT and turns **2** checks red; the first run, carrying a *placeholder* lexical
pin (0.6479 instead of the measured 0.7465), failed **4** checks. **No blob was
fitted or changed** — nothing in §4.2's corrected table needs re-auditing.


### 6.9 Track F — the LAST direction probe: a tiny NON-LINEAR probe on `h[E]`

§6.5 killed the **linear** probe. One escape hatch remained open: maybe `h[E]`
*does* encode the forward direction, but not linearly. Track F pre-registers a
single non-linear probe, runs it once, and the program closes either way.

#### The instrument, and why it took three attempts

Reported in full, because a probe that fails its own controls cannot be used to
pronounce on anything.

| attempt | probe | leak control | non-linear control | usable? |
| --- | --- | ---: | --- | --- |
| v0 | plain SGD, 400 epochs, 768 raw dims | **0.61** ✗ | not recoverable | **NO** — instrument invalid, no verdict taken |
| v1 | Adam, 2000 epochs, 768 raw dims | 0.81 ✓ | not recoverable | **NO** — too little power at N≈850 |
| v2 | Adam, 2000 epochs, **PCA-32 (train)** | **1.0000** ✓ | **0.6967** vs linear **0.5519** ✓ | **YES** |

v0's leak control injected the label *into* `h[E][:,0]` and the probe still
reached only **0.61** on the holdout. That is a probe failure, not evidence about
`h[E]`, so v0's candidate numbers (0.2787 / 0.2493) were **discarded as such** —
they are recorded here and in the tool's history, not used. Adam (the optimiser
`fit_softmax` already uses for every shipped head) and a train-split PCA
projection fixed the conditioning.

The projection does **not** weaken the comparison: a linear head on those
components *is* a linear head on `h[E]` restricted to that subspace, so the
non-linear probe remains a **strict superset** of the linear one it is
contrasted with. The basis is fit on the TRAIN split only, and its signs are
fixed so the seeded init — and therefore the whole fit — is reproducible.

#### The pre-registration (`tools/edge_tracks.py`, constants frozen before the run)

```
input  : h[E] -> standardise(train) -> PCA(32, train, sign-fixed)
hidden : exactly 1 layer, 64 units, tanh
output : K-way softmax, inverse-frequency class weights (from TRAIN)
fit    : full-batch Adam, 2000 epochs, lr 0.05, l2 1e-4, seeded init
         the holdout is never read during the fit
bar    : accuracy >= 2 x (majority-class rate of the holdout) AND n_holdout >= 30
```

Five rows, all pre-registered. **Control rows are not candidates**: they carry
`control_floor` / `cleared_bar` instead of `floor` / `adopted`, so nothing can
mistake a control for a head that cleared the bar.

#### Result (`tools/edge_tracks.json`, `direction_program`)

| row | role | target | n | acc | floor | ÷floor | required | got |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| **F1** | candidate | directional tertile — Track E's exact target, mask, split, holdout | 366 | **0.3197** | 0.3716 | **0.860x** | clear the bar | **NO-SIGNAL** |
| **F2** | candidate | balanced 7-action teacher (§45) | 369 | **0.2114** | 0.3198 | **0.661x** | clear the bar | **NO-SIGNAL** |
| Fleak | control | direction label injected into PC0 | 366 | **1.0000** | 0.3716 | 2.691x | must clear | ✓ |
| Fcap | control | in-split tertiles of pc0·pc1 — **mlp** | 366 | **0.6967** | 0.3333 | 2.090x | must clear | ✓ |
| Fcap | control | in-split tertiles of pc0·pc1 — **linear** | 366 | 0.5519 | 0.3333 | 1.656x | must NOT clear | ✓ |
| Fctl | control | F1 with the TRAIN labels shuffled | 366 | 0.3716 | 0.3716 | 1.000x | must NOT clear | ✓ |

`controls_valid = true`. Read the four controls as one sentence: the probe
**recovers a signal that is present** (1.0000), it **fits and generalises a
genuinely non-linear function that a linear probe on the same target cannot**
(0.6967 vs 0.5519), and it **cannot clear the bar on shuffled labels** (exactly
1.000x — it collapses onto the majority class).

Both candidates sit **below their own floors**. The calibration did not
manufacture that: F1 read **0.279 / 0.295 / 0.353 / 0.306 / 0.320** across the
five probe budgets, never once approaching the 0.372 floor.

F2 covers the other reading of "the balanced teacher" (the §45 balanced **action**
teacher rather than the direction teacher); it fails harder, and it fails the way
a skewed target should — 0.661x, with the classifier falling onto the majority
class.

---

### 6.10 The direction-prediction program is CLOSED

Every probe family has now been tried on the same frozen `h[E]`, the same
holdouts and the same corrected bar.

| track | probe family | target | best ÷floor | adopted |
| --- | --- | --- | ---: | --- |
| A | linear (7-action) | shipped + balanced teacher, 4 horizons | 0.926x | no |
| B | linear, + 12 hand features | shipped + balanced teacher | 1.008x (z=0.11 — noise) | no |
| C | linear, multi-asset pooled | shipped + balanced teacher | 0.902x | no |
| D | linear, rank/quantile | forward-return rank | 0.789x | no |
| **E** | **linear** | **perfectly balanced 3-class direction** | **0.885x** (below 1/3) | no |
| **F** | **non-linear (64-unit, 1 hidden layer)** | **balanced 3-class direction** | **0.860x** | no |
| F | non-linear | balanced 7-action teacher | 0.661x | no |

**The direction-prediction program is closed.** `h[E]` does not carry the forward
direction of the bar in a form any probe tried here can recover — not linearly
(Track E, below exact chance on a balanced target), and not through a 64-unit
non-linear head (Track F, 0.860x, with the instrument validated in both
directions).

**What that means for the product, stated plainly.** The system's trading edge
does **not** come from predicting direction. It lives in exactly three places,
all of which are already built and tested:

1. **Rule-based engines** — the deterministic regime/strategy/signal layers, which
   are the *only* components authorised to gate a trade (§2.2, §6.6).
2. **Risk management** — position sizing, the C++-enforced limits, and the
   regime veto that a learned label is now structurally unable to move.
3. **Honest abstention** — the calibrated heads' measured inability to reach the
   confidence bar, which makes them self-route to ABSTAIN rather than guess
   (0 of 369 rows committed).

No further direction experiment will be opened. Any future work on the heads
must be about *calibration and abstention*, not about finding signal that six
pre-registered tracks say is not there.
