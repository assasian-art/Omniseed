# §43 — RETRACTED: `holdout_acc` is a holdout number. The real defect is in §41.

**Status: the headline finding of this milestone was WRONG. Corrected here.**
**Severity of the retraction: high** — a false accusation was published against a
committed metric before it was verified. The verification is below.

**Follow-up (§44): the genuine §41 defect found *by* this investigation is now FIXED.**
`tools/derive_threshold.py` reads the shipped blob and the real dump instead of
refitting inside the holdout; see §"What was owed — PAID in §44" below.

This file originally read *"`holdout_acc` is not a holdout number — every published
accuracy for a fitted head is an in-sample figure wearing a holdout label."*
**That is false.** It is kept, rather than deleted, because the way it was reached
is the more useful lesson.

## What was claimed (wrong)

`tests/test_calibration.cpp::score_decision()` loops:

```cpp
for (int64_t i = 0; i < f.n(); ++i) {      // f.n() == 369
```

The claim was: `f.n()` is the whole dataset (1,231 trading rows), so scoring all
`f.n()` rows scores the 70% the head was fitted on, and the column named
`holdout_acc` is therefore in-sample.

## What is actually true

**`f.n()` is the FIXTURE size, and the fixture contains ONLY the holdout rows.**
`tools/train_heads.py::emit_fixture()` is called with `idx = rr["_test_idx"]`, and
`_test_idx` is set to `hold_used`, which is `hold` — the holdout half of the split:

```python
hold_used = hold if len(hold) else tr          # train_heads.py:462
...
return { ..., "_test_idx": hold_used }         # :495
...
emit_fixture(args.fixture_dir, "trading", H, rr["_test_idx"], ...)   # :859
```

and inside `emit_fixture`:

```python
idx = np.asarray(idx, dtype=int)
H[idx].astype("<f4").tofile(os.path.join(d, "hidden.f32"))   # only those rows
```

So the fixture is a **view of the holdout**, not of the dataset. Scoring every
fixture row scores **only** rows the fit never saw. `metrics.tsv`'s
`holdout_acc = 0.2439` is a genuine held-out accuracy.

### Verified from the tree, four ways

| check | result |
|---|---|
| fixture `hidden.f32` row count vs `labels.tsv` rows vs `metrics.tsv` `n` | **all equal** (71/71/71 for language; 369/369/369 for trading) |
| the row count matches a 70/30 split of the full dump | 1,231 × 0.30 ≈ **369** ✔ |
| `emit_fixture` writes `H[idx]` with `idx = _test_idx = hold_used` | source, above |
| the shipped blob's own header | `calib_samples = 369` — it was calibrated on *those* 369 rows |
| **`docs/UNCERTAINTY.md` (§35), written months earlier** | *"The `h[E]` fixtures are the §32 held-out split"* — the tree already said so |

Five independent readings agree. The original claim was that `score_decision`'s
`f.n()` was "ALL rows" of the dataset; it is all rows **of the holdout**. The last
row is the sharpest: the correct fact was **already documented in this repository**
before the accusation was written, and the accusation did not consult it.

### And the "second discrepancy" was not one either

The original file also claimed the fixture "cannot reproduce the blob" (T
13.325159 shipped vs 20.042676 refit; max bias difference 46.089) and called it a
finding. It is the **expected** result of refitting on a different row set: the
blob was fitted on the dump's 862 training rows, while a refit "from the fixture"
fits on 369 *holdout* rows. Two different training sets cannot yield one head.
There is nothing wrong with the blob, and no evidence that `hidden.f32` was
re-dumped out of step. **That paragraph was a non-sequitur and is withdrawn.**

## The real defect, which the investigation did find — in §41

While chasing the phantom, a genuine provenance contamination surfaced in
**`tools/derive_threshold.py`** (DECISION 1, §41), not in `metrics.tsv`.

`threshold_curve.json` records `n_total = 369, n_train = 258, n_holdout = 111`.
But the script loads **the fixture** and splits *that*:

```python
meta, H, ids, colnames, labels = th.load_dump(args.fixture)   # the FIXTURE (369)
tr, hold = th.chronological_split(len(ya))                    # 258 / 111
```

So DECISION 1 refits a head on **258 rows that are themselves the §32 holdout**,
then scans on the remaining 111. Therefore:

- `threshold_curve.json`'s `temperature = 20.042676` and `holdout_acc = 0.1441`
  describe **that refit**, not the shipped `trading_head.bin` (T = 13.325159,
  held-out acc **0.2439**). §41's doc line "T = 20.0427 holdout acc 0.1441" was
  read as the shipped head's numbers. It is not.
- `0.1441` is **optimistically biased** — 258 of its 369 eligible rows were in its
  training set.
- `n_total` is a **mislabel**: it names the fixture size as if it were the total,
  which is what hid the contamination.
- **The conclusion is untouched, and conservative.** A head *helped* by seeing 258
  holdout rows reached only 1.008× chance; a clean head reads 1.707× (0.2439).
  Neither clears 2×, so `min_confidence = 0.50` stands for the right reason.
  DECISION 1 is **not** invalidated.

### What was owed (its own milestone) — **PAID in §44**

`derive_threshold.py` had to read the **real dump** (`build/head_data/trading`,
1,231 rows from `train_heads.py`) and the **shipped blob**, score the blob on its own
369-row holdout — one fit, one split, one provenance — and regenerate
`tools/threshold_curve.json`. That also meant the C++ `omniseed_threshold_derivation`
test's pinned values.

**§44 landed the fix.** The tool now:

- reads `models/heads/trading_head.bin` (via `uncertainty_audit.load_decision`) —
  **no model is fitted in the tool at all**, so there is no second head to conflate;
- loads the real dump (1,231 rows), rebuilds the action teacher with the same
  `train_heads.py::build_action_labels`, splits 862/369, and scores the blob on the
  369-row holdout;
- relabels `n_total` → `n_dataset` (1,231) and adds `n_train`/`n_holdout`/
  `fixture_n_holdout`/`holdout_matches_fixture`/`blob_calib_samples`/`blob_calib_ece`;
- cross-checks its holdout size against the committed fixture and prints `OK` /
  `** MISMATCH **`.

`tools/threshold_curve.json` now reads `T = 13.325159`, `holdout acc 0.2439` — the
shipped head's own numbers — and the answer is still `0.500000`, the **reached**
fallback. `tests/test_threshold_derivation_gate.py` (`omniseed_threshold_gate`, §44)
ties the tool's output to `metrics.tsv`'s `DecisionAction` row to the digit and
asserts the old defect's signature (`T=20.042676`, `0.1441`) is **absent**; both
directions were proven to fail when corrupted. C++ `omniseed_threshold_derivation`
still passes **23 checks, 0 fail** (it always loaded the blob).

**A follow-up worth noting:** the §44 fix did *not* change the §4.4 answer (`0.50`),
but it changed the *reason* favourably — the shipped head is **1.71×** chance
(0.2439), not the contaminated refit's 1.01×. Quoting `threshold_curve.json`'s
`holdout_acc` is now safe: it **is** the shipped head's accuracy.

## What changes, restated honestly

1. **§32's 0.2439 is correct and held out.** No correction to `metrics.tsv` is
   owed; no correction ever was. `docs/CALIBRATION.md` §4.1 keeps the number and
   §4.5 now states that the held-out status was **verified, not assumed**.
2. **§41's T / 0.1441 provenance** was marked in `docs/CALIBRATION.md` §4.4 and
   §4.6 as belonging to a refit inside the holdout, and is now **fixed (§44)** — the
   tool reads the shipped blob, so `threshold_curve.json` carries `T = 13.325159` /
   `acc 0.2439` and the tool's output is gated against `metrics.tsv`. The value 0.50
   is unchanged.
3. **`tools/signal_audit.py`** does **not** carry an in-sample warning — it would
   have been false. It reports `accuracy_is_held_out: True`, with the four-way
   verification above cited.

## The lesson (why this file is kept)

The §38 F3 / §43 shape was stated as *"a test and the artifact it checks must not
share a definition of the thing being checked."* That is still true — but this
milestone adds the sharper, more embarrassing half:

> **Before accusing a committed artifact of being wrong, establish what the
> artifact's inputs actually are.** I read `f.n()` as the dataset size without
> checking what `f` was loaded from, then built a whole document, a C++ test and a
> published table on that unchecked reading — while `docs/UNCERTAINTY.md` had
> stated the correct fact since §35. **Read the repo's own prior findings before
> declaring a new one.**

A test that *can* fail is necessary; here the failure was in the auditor, not the
system. The correction cost is why the four-way verification in the table above
exists — it is what should have been done *before* the accusation.

## The gate — `C4`, corrected

`tests/test_calibration.cpp`'s `C4` originally asserted the (false) defect: that
the all-rows score differs from a "holdout" score by > 0.005, with a comment
calling the fixture a superset of the training data. **It is rewritten** to assert
the truth instead:

- the fixture's row count **equals** `metrics.tsv`'s `n` — i.e. the fixture is the
  holdout, not a superset of it (this is the assertion that would have caught the
  original error),
- the blob's `calib_samples` **equals** the fixture row count — the blob was
  calibrated on exactly these rows,
- and the shipped all-rows accuracy **equals** the recorded `holdout_acc` within
  tolerance — a consistency check that is now *meaningful*, because the row set it
  scores is known to be held out.

A regression that put training rows back into the fixture would now change the
first two counts and fail loudly.
