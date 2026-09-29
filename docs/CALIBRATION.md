# CALIBRATION — head training, temperature scaling, and what the numbers say

Milestone 4. Before this milestone every head in the stack was a **seeded
placeholder**: structurally valid, arithmetically correct, and meaningless. Its
`provenance()` said so, and `trained()` was false — but a placeholder still
*emits a number*, and §28's router gates the fast path on that number at `0.85`.
An uncalibrated head that is right 24% of the time can still print `0.92`.

This document records what was trained, on what data, how the confidences were
calibrated, and — per the mandate's hard rule — **what the accuracy actually is,
including where it is bad.**

> **The one-sentence result.** A confidence number from a calibrated head is now
> backed by a measurement on data the head never saw. The trading action head's
> confidence is honestly low (max 0.4966), so it self-routes **0 of 369** rows
> and escalates everything to System-2. That is the fail-closed behaviour the
> router was designed for, and it only became visible *because* of calibration.

---

## 1. What was trained

Five readouts, all on the **same** `h[E]` from one RWKV-7 forward pass
(`E = 768`, post-`ln_out` residual). No head gets its own backbone.

| head | kind | labels (K) | blob |
|---|---|---|---|
| `language.intent` | `ClassificationHead` set | 7 | `models/heads/language_head.bin` |
| `language.language` | `ClassificationHead` set | 4 | `models/heads/language_head.bin` |
| `language.sentiment` | `ClassificationHead` set | 3 | `models/heads/language_head.bin` |
| `trading.regime` | `ClassificationHead` set | 4 | `models/heads/trading_regime_head.bin` |
| `DecisionAction` | `DecisionHead` | 7 | `models/heads/trading_head.bin` |

The label vocabulary is **not** duplicated in the trainer: `tools/train_heads.py`
parses `src/classification_head.cpp` (`default_domain_label_sets()`) and
`include/omniseed/decision_head.h` (`DecisionAction`) so there is exactly one
source of truth. If the C++ vocabulary changes, the trainer follows it or fails.

---

## 2. Data provenance

### 2.1 Language — `tools/data/head_language.tsv`

238 rows, hand-written, header
`text <TAB> language.intent <TAB> language.language <TAB> language.sentiment`.

| column | distribution |
|---|---|
| `language.intent` | command 34, farewell 19, greeting 25, question 42, statement 77, thanks 20, unknown 21 |
| `language.language` | bn 71, en 112, mixed 37, unknown 18 |
| `language.sentiment` | negative 39, neutral 129, positive 70 |

**Bengali is written in Bengali script, never romanised** — a romanised Bengali
string is a different token sequence and would be labelled `en` by the tokenizer
anyway. The generator (`tools/make_head_data.py`) refuses duplicate texts and
refuses embedded tabs/newlines, so the TSV cannot silently mis-align.

> The mandate asked for "50+ labelled examples, expand to 200+". The file has
> **238**. The mandate also referenced `tests/test_language_heads.cpp` as the
> source; that file's examples are assertions, not a labelled corpus, so the
> corpus was authored directly and the *label vocabulary* was taken from the
> test's sets instead.

### 2.2 Trading — `models/market/AAPL_1d.csv`

1,231 real daily AAPL bars.

> **Path discrepancy, reported not hidden.** The mandate named
> `models/market/paper/AAPL_1d.csv`; that file does not exist. The real file is
> `models/market/AAPL_1d.csv`.

> **No `regime` column.** The mandate assumed ground-truth regime labels were in
> the CSV. They are not. The regime label is therefore computed **in-script by
> the real `RegimeEngine`** (`RegimeEngine::scan_raw`), so the label the head is
> trained to imitate is exactly the label the runtime will produce.

The **action** label has no ground truth at all — the system has never traded.
It is produced by a **rule-based teacher** (`build_action_labels`), which is a
*definition*, not realised P&L:

```
first match wins, N = 5 bars forward, threshold = 1.5 %
  CLOSE   if trend_up/trend_down and the forward return opposes the trend by > threshold
  HEDGE   if high_vol and |r| <= threshold
  ABSTAIN if high_vol and |r| >  threshold
  EXPLAIN if range    and |r| <= 0.5 %
  BUY     if r > +threshold
  SELL    if r < -threshold
  HOLD    otherwise
```

**A high action accuracy would mean the head imitates the rule well. It would
say nothing about profitability.** The head is not being asked to be profitable;
it is being asked to reproduce a documented policy from `h[E]`.

### 2.3 How `h[E]` was collected — `tools/dump_hidden.cpp`

RWKV-7 is recurrent and constant-memory. Feeding all 1,231 bars through **one
evolving `RwkvState`** (streaming mode) makes the final hidden state a
*reachable* state of the model. Integer-epoch timestamps fed as independent
windows would be out-of-distribution. Measured ~55 ms/token; a window-mode run
would have been ~460k tokens ≈ 7 hours.

The collector has an **honesty contract**: if the backbone is absent, or a text
encodes to zero tokens, it writes nothing and exits non-zero. It never
synthesises a hidden state to fill a row.

---

## 3. The protocol

### 3.1 Split

| data | split | why |
|---|---|---|
| language | **stratified** 70/30 | rows are exchangeable |
| trading | **chronological** 70/30 | adjacent bars are NOT exchangeable; a random split would leak |

| head | train | holdout |
|---|---|---|
| each language set | 167 | 71 |
| `trading.regime`, `DecisionAction` | 862 | 369 |

### 3.2 Fit

`fit_softmax()` — full-batch Adam on the L2-regularised multinomial
log-likelihood, **initialised at the class log-priors** (so an uninformative
`h` starts at the base rate, not at uniform). It refuses to return non-finite
weights. Input `h` is standardised for the fit, and the standardisation is then
**folded back into raw-`h` form**, so the runtime does one matvec on the
unmodified `h[E]` — no preprocessing on the hot path.

### 3.3 Temperature — and why it is fitted by cross-validation

Temperature scaling is `softmax(logits / T)`. `T == 1.0` is an exact no-op.

Fitting `T` on a 15% slice of the holdout (~36 rows) gave `T` **systematically
1.5–2× too small** (measured against the oracle):

| set | T from a 15% slice | T oracle |
|---|---|---|
| `language.intent` | 19.4 | 28.7 |
| `language.language` | 3.5 | 6.5 |
| `language.sentiment` | 2.5 | 4.7 |

Every set came out **under-softened**. The fix is `fit_temperature_cv()`:
**5-fold inside the holdout**, average `log T` (geometric mean), and report the
**out-of-fold ECE** — the unbiased number, which is the one stored.

`shuffle=False` for the trading data, because a shuffled fold of a time series
is not a held-out fold.

---

## 4. Results (all measured, all held out)

### 4.1 Accuracy

These are **held-out** numbers: the fixture each row is scored on contains exactly
the holdout rows and nothing else, so scoring every fixture row scores only the
30% the fit never saw (verified in §4.5; the earlier in-sample claim was retracted
in §43).

| head | K | holdout n | accuracy | macro-F1 | T |
|---|---|---|---|---|---|
| `language.language` | 4 | 71 | **0.845** | — | 5.161 |
| `trading.regime` | 4 | 369 | **0.705** | 0.389 | 3.250 |
| `language.sentiment` | 3 | 71 | **0.676** | — | 3.583 |
| `language.intent` | 7 | 71 | **0.634** | — | 23.441 |
| `DecisionAction` | 7 | 369 | **0.244** | 0.133 | 13.325 |

⚠️ **`T` here is the shipped blob's temperature.** Do not confuse it with §41's
`T = 20.0427`, which belongs to a *different* head §41 refitted inside the holdout
— see §4.6.

### 4.2 Calibration (ECE, lower is better)

| head | ECE raw | ECE calibrated | top bucket `[0.9,1.0]` gap |
|---|---|---|---|
| `trading.regime` | 0.2099 | **0.0343** | 0.0323 — **misses** the ±3% target |
| `DecisionAction` | 0.6219 | **0.0555** | 0 rows in bucket |
| `language.language` | 0.1387 | **0.0892** | 0.0094 — within target |
| `language.sentiment` | 0.2476 | **0.0867** | 6 rows (too few to assert) |
| `language.intent` | 0.3604 | **0.1978** | 6 rows (too few to assert) |

Every set improved. `language.intent` improved the least and remains the worst
calibrated set (0.198) — it is the hardest problem (7 classes, 71 holdout rows)
and it stays honest about it.

### 4.3 The fail-closed result — the reason this milestone matters

```
DecisionAction  max calibrated confidence 0.4966  <  threshold 0.85
                => 0/369 held-out rows self-route (fail closed)
```

The raw softmax on this head claims high confidence routinely. The head is right
**24%** of the time. After calibration its *maximum* confidence across all 369
held-out rows is **0.4966** — it never once claims to know. So the router
escalates every row to System-2, which is exactly what a 24%-accurate head
should do. **Calibration is what makes the fail-closed gate actually fail
closed.**

---

## 4.4 `min_confidence` is DERIVED, not chosen — DECISION 1 (2026-09-29)

The standing question after §39 was: `0.4966` sits 0.0034 below the filter's
`0.50` bar, and that margin was a coincidence of two independently chosen
numbers. §39 left it alone and printed the finding, which was the right call but
not an answer. **DECISION 1 replaces the coincidence with a rule.**

### The rule

```
chance = 1 / K                        K = 7 DecisionActions -> 0.142857
accept a threshold iff               cumulative pool (conf >= T) has
                                     accuracy >= 2 * chance  AND  n >= 30
min_confidence = the LOWEST such threshold
                  else  keep 0.50 fail-closed
```

Two clarifications that a naive reading gets wrong, both recorded because they
change the answer:

1. **The pool is cumulative, not per-bucket.** Rows below a threshold are not
   discarded, they are *abstained* — and an abstain is a non-decision. What must
   clear 2× chance is the pool the filter actually acts on: every row at or above
   the threshold. The per-bucket table is printed too, but it is evidence, not
   the criterion.
2. **The scan refuses thresholds below `chance`.** A bar at or under 1/7 would
   commit on rows the head admits it knows nothing about — the opposite of a
   confidence gate. This floor was not in the first version of the scan, and the
   DECISION-1 control test immediately caught it choosing `T = 0.0` for a
   confidently-correct synthetic pool. A rule that can only ever return its own
   fallback is not a rule, so the control exists to prove it *can* say yes.

### The measured curve (held out, calibrated)

⚠️ **Whose head is this?** The curve is computed by `derive_threshold.py`, which
refits a head *inside the fixture* (258 of the 369 **holdout** rows) before
scanning. So `T = 20.0427` and `acc 0.1441` describe **that refit**, not the
shipped `trading_head.bin` (`T = 13.325159`, holdout acc **0.2439**, §4.1). See
§4.6 — the provenance is contaminated in the conservative direction and the
conclusion below is unaffected, but the two heads must not be conflated.

```
fixture  tests/fixtures/head_calibration/trading   369 rows (train 258 / holdout 111)
T = 20.0427      holdout acc 0.1441      ECE 0.0687      chance 0.1429
accept bar = 2/7 = 0.285714

confidence bucket      n     emp.acc   mean.conf
[0.15, 0.20)          41     0.1463     0.1861
[0.20, 0.25)          59     0.1525     0.2186
[0.25, 0.30)           8     0.1250     0.2704
[0.30, 0.35)           3     0.0000     0.3117
```

**The entire holdout confidence range is 0.15 – 0.3117.** The two buckets with
`n >= 30` reach 0.1463 and 0.1525 against a 0.2857 bar. The best cumulative pool
of `n >= 30` (43 rows) reaches **0.1923**. Nothing comes within 0.09 of the bar.

### The derived value

```
DERIVED min_confidence = 0.500000
because: NO threshold satisfies both conditions (>= 2x chance on the cumulative
         pool AND n >= 30). Keeping the fail-closed default 0.50 — that is a
         valid outcome, not a failure.
```

**`min_confidence` stays 0.50, fail-closed, and it is now a derived conclusion
rather than a retained default.** The stronger finding: the trading action head
has **no usable confidence signal at all** — its accuracy (0.1441) is
statistically indistinguishable from chance (0.1429), so *no* threshold could have
been principled. §39's "reported, not tuned" was the honest interim answer;
DECISION 1 is the proof that there was nothing to tune to.

### Paper only (L0)

This threshold gates **paper commits**. It does not gate, loosen or inform any
live-money path — that gate is separate and C++-enforced (mandate rule 4).

### Reproducing and gating

```bash
.venv/Scripts/python.exe tools/derive_threshold.py      # writes tools/threshold_curve.json
./build/bin/omniseed_threshold_derivation.exe           # re-derives from the BLOB
```

⚠️ **The two paths do NOT describe the same fit** (§4.6): the Python tool refits
*inside the fixture* (258 of the holdout's own rows) while the C++ test loads the
**shipped blob**. They agree on the **decision** (`0.500000`, the fallback) but
their `T`/accuracy belong to different heads, so they must not be compared
number-for-number.

The Python tool reproduces a fit from the fixture
(`chronological_split` + `fit_one(shuffle=False)`, deterministic — two runs give
byte-identical JSON). The C++ test re-derives the curve from the **committed
blob** through `DecisionHead::decide()`, applying the same rule, and asserts the
result equals this documented value: **23 checks, 0 fail, 0 skip.** It also
asserts the answer is the *fallback reached because nothing passed* — not a
threshold that happened to equal 0.50 — and it carries three controls proving the
rule can accept a good pool, reject a confident-wrong one, and reject a
below-chance one.

---

## 4.5 Per-head signal audit — D1 (2026-10-01)

The standing question after the action head landed at ≈ chance: **where does real
signal actually live?** Assuming is not an answer, so the table publishes it.
Regenerate with:

```bash
.venv/Scripts/python.exe tools/signal_audit.py --markdown   # add --json for machines
```

The rule is the §41 shape reused verbatim: `chance = 1/K`; a set is **SIGNAL**
when `accuracy >= 2*chance` on `n >= 30`; **INSUFFICIENT-DATA** below `n = 30`;
**NO-SIGNAL** otherwise.

```
set                    n_holdout  accuracy    chance   ratio  verdict
language.intent               71    0.6338    0.1429   4.437  SIGNAL
language.language             71    0.8451    0.2500   3.380  SIGNAL
trading.regime               369    0.7046    0.3333   2.114  SIGNAL
language.sentiment            71    0.6761    0.3333   2.028  SIGNAL
DecisionAction               369    0.2439    0.1429   1.707  NO-SIGNAL
SUMMARY: 5 fitted (4 SIGNAL, 1 NO-SIGNAL, 0 INSUFFICIENT-N), 8 unfitted
```

⚠️ **These accuracies ARE held out — verified, not assumed (§43 retraction).** An
earlier draft of this milestone claimed `holdout_acc` was in-sample because
`score_decision()` loops `f.n()` rows. That was wrong: `f.n()` is the **fixture**
size, and `tools/train_heads.py::emit_fixture()` writes **only the holdout rows**
into the fixture (`idx` = `hold_used` = `hold`). So scoring every fixture row *is*
scoring the holdout. Verified from the tree: each fixture's `hidden.f32` row count
equals `labels.tsv`'s and equals `metrics.tsv`'s `n` (71/71/71/369), and the
shipped `trading_head.bin` records `cals = 369` — it was calibrated on those 369
rows, which are the 30% the fit never saw. The full trace is in
`docs/HOLDOUT_DEFECT.md`.

The one *real* defect this milestone found is adjacent and narrower — see §4.6.

**The eight unfitted sets** (the tool hardcodes them with reasons so the table can
never imply coverage it does not have): `language.task` (no consumer — see §6),
`vision.scene` / `vision.anomaly` (no image corpus), `audio.wake` /
`audio.speaker` (owner voice not yet recorded), `audio.emotion` (no audio→E
adapter), `general.routing` / `general.priority` (no labelled rows anywhere).

**What this changes.** `trading.regime` at **2.114× chance on 369 held-out rows**
is the only trading-side set with real signal. That is the fact that makes D2's
repositioning defensible.

**Gating.** `tests/fixtures/head_calibration/*/metrics.tsv` is the source.
`tests/test_signal_audit.py` asserts the table above regenerates from the
committed fixtures **both ways** — its own independent re-derivation must equal
the published literals, and `signal_audit.py --json` must equal that
re-derivation. Both halves were shown to fail when corrupted and to pass when
restored (**54 checks, 0 fail**). Registered as `omniseed_signal_audit`,
stdlib-only so it runs in CI. A re-fit that moves a ratio now fails loudly instead
of landing silently.

---

## 4.6 The §41 threshold was derived from a head fitted on holdout rows

This is the genuine defect that the §43 investigation ended up finding, and it is
**not** about `metrics.tsv`. It is about `tools/derive_threshold.py`.

`threshold_curve.json` records `n_total = 369, n_train = 258, n_holdout = 111`.
That reads as "369 rows, split 258/111". But 369 is the **fixture** — the §32
**holdout** — and the script did precisely this:

```python
meta, H, ids, colnames, labels = th.load_dump(args.fixture)   # the FIXTURE, 369 rows
tr, hold = th.chronological_split(len(ya))                    # 258 / 111
```

So it re-fitted a head on **258 rows that are themselves in the §32 holdout**, then
scored the remaining 111. Consequences:

- Its `temperature = 20.042676` and `holdout_acc = 0.1441` describe **that** head,
  not the shipped `trading_head.bin` (whose T is 13.325159). The two are different
  fits; the doc previously implied one head had two numbers.
- `0.1441` is **optimistically biased** (258 of its 369 test-eligible rows were in
  its training set) — yet it *still* found no threshold clearing 2× chance.
- `n_total` is a **mislabel**: it is the fixture size, and calling it the total
  makes the contamination invisible.

**The conclusion is unaffected and in fact conservative.** A head trained partly on
holdout rows beat chance by only 1.008×; a head that never saw them (the shipped
one) still reads 1.707× as recorded, and *nothing* clears the 2× bar either way —
so `min_confidence` stays 0.50 for the right reason. §41's DECISION 1 stands.

**What is owed.** `derive_threshold.py` should read the **real dump**
(`state/trading`, 1,231 rows from `train_heads.py`), which fits a clean model and
then derives the threshold from *its own* holdout — one fit, one split, one
provenance. Recorded here as a follow-up with its own milestone, **not** patched
mid-flight: it changes a committed number and the C++ re-derivation test that
pins it. Both are named in `docs/HOLDOUT_DEFECT.md` §"What is owed".

**The C++ half is already clean.** `omniseed_threshold_derivation` does *not* use
the Python path — it loads the **shipped blob** and re-derives:
`head: loaded fitted projection + calibrated (n=369, ece=5.55%)`,
`derived=0.500000`, `max_conf=0.4966`, **23 checks, 0 fail**. So the *conclusion*
(`min_confidence = 0.50`) is confirmed against the real head by a gated test; only
the Python tool's `T`/`0.1441` provenance is contaminated.

⚠️ **Do not read `threshold_curve.json`'s `holdout_acc` as the shipped head's
accuracy.** The shipped head's held-out accuracy is **0.2439** (§4.1, correct).

---

## 5. Findings worth keeping

1. **One temperature per label set, not one per head.** A single pooled `T`
   made `language.language` *worse* (ECE 0.172 → 0.194) while fixing
   `language.intent` (0.451 → 0.115), because `intent` wants `T ≈ 23` and
   `language` wants `T ≈ 5`. Sets whose logit scales differ by an order of
   magnitude cannot share a scalar. `ClassificationHead` therefore stores a
   temperature **per set**.

2. **A temperature is a global soften, not a local patch.** Softening the
   `[0.9,1.0]` bucket softens every bucket. A scalar `T` therefore **cannot**
   fix bucket-*local* miscalibration — which is why `trading.regime`'s top
   bucket still misses the ±3% target even though its pooled ECE is 0.0343.

3. **`trained()` is all-or-nothing and it is honest.** `DecisionHead::trained()`
   is true only when all 7 action rows are fitted; all 7 had examples, so it is
   true — and `provenance()` composes the base string with
   `" + calibrated (n=…, ece=…%)"`. A loaded-but-untrained blob reports
   `loaded seeded placeholder (file was untrained)`, never `fitted`.

---

## 6. Honest gaps (do not overclaim)

- **The trading action head is LOW: holdout accuracy 0.2439 (1.71× chance).** It is
  a *held-out* number (§4.5). `CLOSE` and `HEDGE` have **zero** holdout recall.
  Per-class recall:
  `{ABSTAIN 0.20, HOLD 0.015, BUY 0.324, SELL 0.474, CLOSE 0.0, HEDGE 0.0, EXPLAIN 0.026}`.
  It is well-calibrated *and* weak — the two are independent. Calibration makes
  a weak head **safe**, not good. Its role is a **VETO/ABSTAIN filter, never a
  directional predictor** — see `docs/EDGE_RESEARCH.md`.
- **`trading.regime` is the load-bearing learned head (2.114× chance, n=369).**
- **The regime head never predicts `trend_down`** — there were only 8 such bars
  in 1,231 (label counts `{trend_up 58, trend_down 8, range 1050, high_vol 115}`).
  Accuracy 0.705 is dominated by the majority class; the macro-F1 of 0.389 is the
  number to quote.
- **`trading.regime`'s top bucket misses the mandate's ±3% target at 3.23%.**
  The test asserts a 5% quality bar, **always prints the exact gap**, and prints
  `MISSES the mandate's 3% target` when it is exceeded. It is not hidden.
- **The action labels are a rule, not P&L.** Accuracy here measures imitation of
  a definition. Nothing in this milestone claims the policy is profitable.
- **`assets` and `invalidation` are written empty / zero** in the blobs. Treat as
  "unknown", **never** as "no stop".
- **Nothing else is trained.** Vision and audio still have label spaces and no
  logic. There is no learned token head. The modality **dump paths now exist**
  (vision and audio both go through `MultimodalBridge` — §43), but vision has no
  corpus and audio has no `[T/2,384] → E=768` adapter, so neither can be fitted:
  see `docs/VISION_AUDIO_DATA.md`. `audio.wake` / `audio.speaker` additionally
  need the owner's recorded voice (`omniseed enroll-audio`, RUNBOOK §5).
- **The mandate's `n_samples=1255`** was an assumption; the real numbers are
  167/71 (language) and 862/369 (trading).

---

## 7. Reproducing

Training is **offline**. The runtime never trains — it loads a `.bin`.

```bash
# 1. Collect h[E]  (needs the backbone; ~1-2 min for the language set)
./build/bin/omniseed_dump_hidden.exe language \
    models/rwkv7-0.1B-ternary.gguf tools/data/head_language.tsv build/dump_language

./build/bin/omniseed_dump_hidden.exe market \
    models/rwkv7-0.1B-ternary.gguf models/market/AAPL_1d.csv build/dump_market --stream

# 2. Fit + calibrate + write the blobs and the test fixtures
python tools/train_heads.py            # or: .venv/Scripts/python.exe tools/train_heads.py

# 3. Verify
./build/bin/omniseed_calibration.exe   # 180 checks
```

`--dry-run` on the collector prints the plan and writes nothing.

---

## 8. Blob formats

### `DecisionHead` — version 3 (backward compatible to v2)

```
magic 'O','M','N','I','S','D','H','1'
int32 version              # 3
int32 E, int32 A
int32 fitted_flag
A × str name
A × str asset
A × f32 invalidation
A*E × f32 proj
A × f32 bias
--- v3 trailer (written last) ---
f32  temperature
i32  calibration_samples
f32  calibration_error
```

- **v2 still loads** (temperature 1.0, calibration unmeasured). v1 is rejected —
  a v1 blob carried no invalidation data.
- A non-finite or `<= 0` temperature is normalised to `1.0` rather than stored.
- The trailer is written **last**, so a v2 reader stops cleanly at the bias block.

### `ClassificationHead` — version 1 (new persistence in this milestone)

```
magic, int32 version, int32 E, int32 n_sets
per set: str name, int32 n_labels, n_labels × str label
f32 default_temperature
n_sets × f32 temperature
n_sets × i32 calibration_samples
n_sets × f32 calibration_error
total_labels * E × f32 proj
```

`load()` **rebuilds the sets from the file in file order**, then overwrites with
the stored projection. Any failure path sets `ready_ = false` — **fail closed**.

---

## 9. Loading a calibrated head at runtime

```cpp
#include "omniseed/classification_head.h"
#include "omniseed/decision_head.h"
using namespace omniseed;

// Classification head (regime, language sets)
ClassificationHead ch;
ch.init(768);
if (!ch.load("models/heads/trading_regime_head.bin")) {
    // fail closed: do not silently fall back to a seeded placeholder
}
// load() REBUILDS the sets from the file, so look the index up AFTER loading.
int32_t ri = ch.find_label_set("trading.regime");   // -1 if absent
// ch.trained()               == true
// ch.temperature(ri)         == 3.250    per set
// ch.temperature()           == 1.0      the DEFAULT, not the set's value
// ch.calibration_error(ri)   == 0.0343   that set's ECE
// ch.calibration_error()     == pooled, n-weighted over measured sets

// Decision head
DecisionHead dh;
dh.init(768);
dh.load("models/heads/trading_head.bin");
// dh.temperature() == 13.325, dh.calibration_error() == 0.0555,
// dh.calibrated() == true
```

> **`temperature()` and `calibration_error()` with no argument are NOT "the
> head's".** For `ClassificationHead`, `temperature()` is the *default* applied
> to future sets and `calibration_error()` is the *pooled* n-weighted value.
> Always pass the set index when you mean a specific set.

`calibration_error()` returns **`< 0` when unmeasured** — never `0.0`, which
would read as "perfectly calibrated".
