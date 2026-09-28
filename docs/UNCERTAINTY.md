# Uncertainty, split: aleatoric vs epistemic

**Milestone 7 (§35).** `include/omniseed/core/uncertainty_split.h`,
`src/core/uncertainty_split.cpp`, `tests/test_uncertainty_split.cpp`,
`tools/uncertainty_audit.py`.

`core/uncertainty.h` answers *how* unsure a distribution is, with one number.
This answers *why*, and the two answers call for opposite responses:

| cause | meaning | right response |
|---|---|---|
| **aleatoric** | the classes genuinely overlap on this input; a perfect model would still be unsure | stay out — there is no edge, and more data will not create one |
| **epistemic** | the input lies outside the region the head was fitted on; the head is extrapolating | go and learn — more data *would* help |

## 1. What is measured, and where

Both halves are measured against the **real held-out heads** shipped in
`models/heads/`, using the committed `h[E]` fixtures in
`tests/fixtures/head_calibration/`. Reproduce with:

```
python tools/uncertainty_audit.py
```

* **Aleatoric** = `H(p) / ln(K)` where `p` is the head's **calibrated** softmax
  output — temperature already applied. Applying the fitted `T` from §32 is
  what turns the entropy from a raw number into a claim; without it the
  comparison is meaningless.
* **Epistemic** = the empirical CDF of the diagonal-Mahalanobis distance of
  `h[E]` from the fitted reference:
  `d(h) = sqrt( mean_e( ((h_e − μ_e)/σ_e)² ) )`, then
  `epistemic(h) = P_ref(d_ref ≤ d(h))`. In-distribution input sits near 0.5;
  out-of-distribution input goes to 1.0.

## 2. Aleatoric: monotone in the error rate for 4 of 5 heads

Error rate per quartile of calibrated normalised entropy, on the real holdout:

| head | K | n | acc | err by entropy quartile | verdict |
|---|---|---|---|---|---|
| `trading.regime` | 4 | 369 | 0.705 | 0.120 → 0.196 → 0.413 → 0.452 | **monotone** |
| `language.sentiment` | 3 | 71 | 0.676 | 0.059 → 0.353 → 0.353 → 0.500 | **monotone** |
| `language.language` | 4 | 71 | 0.845 | 0.000 → 0.000 → 0.235 → 0.350 | **monotone** |
| `language.intent` | 7 | 71 | 0.634 | 0.235 → 0.235 → 0.353 → 0.600 | **monotone** |
| `DecisionAction` | 7 | 369 | **0.244** | 0.772 → 0.804 → 0.685 → 0.763 | **no signal** |

Four of five are monotone, with the top quartile 2.6×–8.5× the error rate of
the bottom one. That is the aleatoric signal doing its job: a high-entropy
input really is one the head is more likely to get wrong, *after* calibration.

The fifth is honest and unsurprising. `DecisionAction` is at **0.244 accuracy
over 7 classes** — near chance (0.143). When a head is close to guessing
everywhere, nothing predicts its errors, and the audit prints `no signal`
rather than inventing a trend. See §32 for why that head is weak: its teacher
is a rule definition, not realised P&L.

## 3. Epistemic: no within-domain signal — and that is correct

The same quartile table, ordered by `h[E]` distance instead of entropy:

| head | err by distance quartile | verdict |
|---|---|---|
| `trading.regime` | 0.261 → 0.304 → 0.272 → 0.344 | no signal |
| `language.intent` | 0.529 → 0.235 → 0.412 → 0.300 | no signal (U-shaped) |
| `language.language` | 0.294 → 0.118 → 0.235 → 0.000 | no signal (**decreasing**) |
| `language.sentiment` | 0.412 → 0.294 → 0.294 → 0.300 | no signal |
| `DecisionAction` | 0.761 → 0.761 → 0.750 → 0.753 | no signal |

**This is the expected result, not a defect.** Within a single domain every
input is by construction in-distribution, so there is no epistemic uncertainty
to find; the top decile of distances is noise. A module that *did* find a
within-domain error signal here would be one that was firing on noise — which
is exactly why `tests/test_uncertainty_split.cpp` Part E asserts the in-domain
negative control (≤ 20% flagged) **together with** the cross-domain positive
(100% flagged). A detector that flags everything passes one and fails the
other; a detector that flags nothing does the reverse.

## 4. Epistemic: complete separation across domains

Reference fitted on one domain, scored on another:

| reference | scored | in-domain d (mean / p90 / max) | out-of-domain d (mean / min) | below in-domain p90 |
|---|---|---|---|---|
| trading | language.intent | 0.990 / 1.169 / 1.614 | 6.699 / **5.129** | **0 / 71** |
| language.intent | trading | 0.981 / 1.198 / 1.683 | 1.719 / 1.482 | **0 / 369** |

The first direction is a clean 5× jump: the *minimum* out-of-domain distance
(5.13) is more than three times the *maximum* in-domain distance (1.61). The
reverse direction is a weaker separation — fitting on only 71 vectors gives a
coarse CDF, and some trading vectors land inside the language CDF's upper
range — but every one of the 369 is still past the language reference's 90th
percentile, and 249/369 saturate at 1.0.

Why the separation is so total: the two domains differ in **both** statistics.
Trading `h[E]` has a larger row norm (262 vs 152) but a *smaller* per-dimension
spread (σ ≈ 0.93 vs 3.05) — a gap of 3.3× in the statistic the distance
actually uses. Note the direction is not the intuitive one, which is why the
test asserts the magnitude of the gap and not its sign.

## 5. The rigorous operator exists; the ensemble to feed it does not

The textbook decomposition (Depeweg et al. 2018) needs a distribution over
model **parameters** `q(θ)`:

```
total     = H( E_θ[ p(y|x,θ) ] )
aleatoric = E_θ[ H( p(y|x,θ) ) ]
epistemic = total − aleatoric = I(y; θ | x)     ≥ 0 by Jensen
```

`UncertaintyDecomposition::from_ensemble()` implements **exactly that**, in
nats, and it is tested — identical members give `epistemic == 0`, two disjoint
point masses give `ln2 / 0 / ln2`, and disagreeing members give `epistemic > 0`.

**Nothing in this tree can feed it.** No ensemble of heads has been fitted, so
there is no `q(θ)`. As in §30, §31 and §33, the gap is the joint, not the
capability: the mathematics is present and verified, the producer is missing.
Until a head ensemble exists, `split()` uses the two *measured* proxies above
and this document says so rather than implying a Bayesian decomposition it
cannot perform.

## 6. Fail-closed policy

Every failure path resolves to **maximum doubt**, never to confidence:

| condition | `distance()` | `epistemic()` | `aleatoric()` |
|---|---|---|---|
| no reference fitted | `−1.0` (not measured) | `1.0` | — |
| non-finite `h` | `−1.0` | `1.0` | — |
| `h` width ≠ reference | `−1.0` | `1.0` | — |
| `probs` not a distribution | — | — | `1.0` |
| `K ≤ 1` | — | — | `0.0` (no distribution to be unsure about) |

`load()` clears the object **before** reading, so a caller that ignores the
return value cannot keep an old reference alive and believe it describes the
file it just failed to read. `save()` refuses an unfitted reference.

## 7. Blob format

`UncertaintyDecomposition` v1, little-endian, self-describing:

```
magic[8] = 'O','M','N','I','S','U','S','1'
i32 version = 1
i32 E
i32 n_fit
f32 aleatoric_thresh
f32 epistemic_thresh
E     x f32  mu
E     x f32  sigma      (floored, strictly > 0)
n_fit x f32  reference distances, ascending
```

A non-positive sigma, unsorted distances, or any non-finite value is refused.
The stored thresholds are informational — `set_config()` overrides them, so a
reference fitted at one operating point can be scored at another.

## 8. Honest gaps

* **The epistemic proxy is a covariate-shift detector, not a posterior.** It
  says "this input is unlike the fitted data", which is a *necessary* but not
  sufficient condition for the head being wrong. §3 shows it does not predict
  within-domain error, and §4 shows it does detect cross-domain shift. Nothing
  here supports the stronger claim.
* **The reference is fitted on the holdout, not on the training split.** The
  `h[E]` fixtures are the §32 held-out split, so the in-domain statistics in
  §2–§4 are in-sample. This inflates nothing that is asserted (the cross-domain
  test compares two different distributions, and the aleatoric table depends on
  the head's predictions, not on the reference), but a production reference
  should be fitted on the training split. `fit()` takes any `[n, E]` block, so
  this is a data-plumbing change, not a code change.
* **`DecisionAction` cannot be evaluated.** At 0.244 accuracy over 7 classes
  there is no signal to find in either half. That is a §32 finding restated,
  not a new one.
* **Nothing consumes the split yet.** No head, router or risk gate calls
  `split()`; the module and its evidence exist, the wiring does not. The
  natural consumer is the router's fast path (§28): a `Both` verdict is a
  strictly better reason to escalate to System-2 than a bare confidence floor.
* **The aleatoric threshold (0.5) and epistemic threshold (0.90) are
  uncalibrated defaults.** They were chosen to be interpretable, not fitted.
  §2 gives the data to fit them; nobody has.
