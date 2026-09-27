# MONSTER TRADING MODEL — Design Document

**Status:** design + reference implementation (paper only). **No guaranteed-profit
claim.** Every number below is a *filter*, not a forecast.

## 0. Where the code lives (and one deliberate deviation)

The mandate asked for Python modules under `omniseed/trading/`. In this repo
`include/omniseed/trading/` and `src/trading/` are **C++17** — putting Python
there would mix languages in one directory. The Python orchestration layer in
this project lives in `tools/`, so the Monster modules live in:

```
tools/monster/features.py           pure, no-look-ahead primitives
tools/monster/sniper_engine.py      the 4-layer weighted confidence score
tools/monster/news_hunter.py        event detection + anomaly cross-reference
tools/monster/correlation_matrix.py cross-asset confirm / invalidate
tools/monster/sizing.py             adaptive ATR+confidence sizing (<=2% cap)
tools/monster/state_vector.py       the distilled vector handed to the engine
tools/monster_scan.py               CLI: CSVs -> state/monster/*.csv
```

The C++ engine gains **one small seam** (`PaperSession` monster gate) so the
Python features actually gate paper entries. Heavy logic stays in Python; the
engine receives only `(ts, confidence, veto, regime, detail)`.

---

## 1. SNIPER ENTRY ENGINE

A single **Confidence Score** `S ∈ [0,1]` is a weighted sum of four layer
scores, each itself in `[0,1]`:

```
S = 0.25·M  +  0.35·T  +  0.25·R  +  0.15·C
```

An entry is proposed only when **all hard vetoes pass** *and* `S >= 0.85`.
Weights are configurable; the defaults above reflect that *technical confluence*
is the most reliable bar-level evidence and *cross-asset* the noisiest.

### 1.1 M — Microstructure (0.25)

**Volume anomaly (always available).** Rolling z-score of volume over `w=20`
trailing bars:

```
z_v = (v_i - mu_v) / sigma_v          (sigma_v = 0 when the window is flat)
M_vol = 1 / (1 + exp(-z_v / 2))        logistic -> [0,1], 0.5 at z=0
```

**Order-book imbalance (when the provider publishes depth).** Free OHLCV feeds
almost never do, so this is optional and auto-detected:

```
OBI   = (V_bid - V_ask) / (V_bid + V_ask)          in [-1, 1]
M_obi = 0.5 + 0.5·OBI                              in [ 0, 1]
M     = max(M_obi, M_vol)          (depth present)
M     = M_vol                      (depth absent)
```

Taking the **max** (not a blend) is deliberate: either a strongly one-sided book
*or* an anomalous volume print is evidence of aggressive participation, so the
stronger of the two carries the layer. Blending would let a quiet book dilute a
genuine volume spike, which is backwards.

**Impact context (used for cost, not for the score).** Two standard
microstructure measures are computed and journalled so slippage assumptions are
honest rather than constant:

* **Kyle's lambda** (price impact per unit signed flow). With bar data we use
  the Lee–Ready tick rule `sign(r_i)` as a proxy for the sign of net flow, giving
  the per-unit-impact estimate `lambda_hat = |dP_i| / v_i`.
* **Amihud illiquidity** `ILLIQ_i = |r_i| / (close_i · v_i)` — the classic
  Amihud (2002) |return|/dollar-volume ratio.

High `lambda_hat` / `ILLIQ` ⇒ thin book ⇒ the caller *widens* the slippage
assumption. Neither can make a trade look better, so neither enters `S`.

### 1.2 T — Technical confluence (0.35)

Six **independent** bullish factors, each a 0/1 vote:

| # | factor | bullish when |
|---|---|---|
| 1 | VWAP reclaim | close crosses from below to above the rolling VWAP |
| 2 | RSI divergence | price prints a lower low while RSI prints a higher low |
| 3 | Fibonacci support | close within `tol` of the 0.382 / 0.5 / 0.618 retracement of the last swing |
| 4 | MACD thrust | histogram `> 0` and rising |
| 5 | Mean reversion | close `<` lower Bollinger band (oversold) |
| 6 | Trend alignment | `SMA_fast > SMA_slow` |

```
T = min(1.00, 0.90 + 0.10 · (votes - 3))   when votes >= 3
T = 0.50 · votes / 3                       when votes <  3   (veto)
HARD GATE: votes >= 3   (the "at least 3 independent factors" rule)
```

The mapping is deliberate: *meeting* the mandated three-factor confluence earns
0.90 of the layer, and each further agreeing factor adds 0.10 up to a cap of
1.00. Scoring `votes/6` instead would cap the reachable total at ≈0.81 and make
the 0.85 bar mathematically unreachable — a gate that can never open is a bug,
not discipline. With this mapping a bar needs **3+ agreeing factors, a
permissive regime, and a real volume anomaly** to clear 0.85, which is exactly
the "sniper" bar the mandate describes.

Divergence is only counted when the two swing lows are genuinely separated
(`k` bars apart), which is what stops a monotone ramp from manufacturing a vote.

### 1.3 R — Regime filter (0.25)

Higher-timeframe regime from the slope of `SMA200` plus ATR% (volatility):

```
slope   = (SMA200_i - SMA200_{i-n}) / SMA200_{i-n}       n = 20
atr_pct = ATR14_i / close_i

trend_up    : slope >  +s  and close > SMA200
trend_down  : slope <  -s  and close < SMA200
high_vol    : atr_pct > atr_hi
range       : otherwise
```

Scoring + veto:

```
R = 1.0   regime == trend_up
R = 0.5   regime == range        (neutral: allowed, but no bonus)
R = 0.25  regime == high_vol     (allowed only with a mean-reversion setup)
R = 0.0   regime == trend_down

VETO unless:  regime == trend_up
           or regime == range
           or (regime == high_vol  and  close < lower_BB and RSI < 30)
           or (regime == trend_down and  close < lower_BB and RSI < 30
                                       and bullish divergence present)
```

The last clause is the mandate's "no long sniper entries in a macro downtrend
**unless it is a verified mean-reversion setup**" — verified = oversold *and*
divergence, i.e. two independent confirmations, not just "it fell a lot".

### 1.4 C — Cross-asset confirmation (0.15)

Driven by the correlation matrix (§3). For every configured lead–lag relation
`(driver, responder, expected_sign)`:

```
if |ret_driver| >= move_threshold:
    confirming   if sign(ret_responder) == expected_sign
    invalidating otherwise
C = clamp(0.5 + 0.5·(n_confirm - n_invalidate) / max(1, n_active), 0, 1)

VETO when n_invalidate >= invalidate_veto   (default 2)
```

The veto is the mandate's "*Long Oil setup detected, but the broader commodity
complex is crashing ⇒ invalidate*". With no active relations `C = 0.5` (neutral,
never a penalty for missing data).

---

## 2. NEWS HUNTER (event-driven alpha)

Asynchronous; injected clock and fetcher so it is fully offline-testable.

**Keyword weights** `w_k ∈ (0,1]` — CPI 0.90, Fed/FOMC 0.90, rate cut/hike 0.85,
SEC approval 1.00, hack/exploit 1.00, bankruptcy 0.95, earnings beat/miss 0.80,
war/sanction 0.80, whale alert 0.60.

**Recency decay** (exponential, `tau = 300 s`):

```
d(dt) = exp(-dt / tau)
```

**Sentiment alignment** `a`: 1.0 sign-matched, 0.5 neutral, 0.0 opposing —
using the same lexicon family as the C++ `NewsFeed` (Tetlock 2007, *Giving
Content to Investor Sentiment*).

**Event score:**

```
E = w_k · d(dt) · a
```

**Anomaly detection** on 1-minute bars:

```
vol_mult = v / mu_v(60)        z_v = (v - mu_v) / sigma_v
anomaly  = (vol_mult >= 5.0) or (z_v >= 3.0)
```

**Match** = `anomaly AND a fresh event with E >= E_min` inside `match_window`
(default 300 s). A match sets `event_driven = True`, which **relaxes the
technical-lag requirement** (the `votes >= 3` gate drops to `votes >= 1`) so a
news spike is not blocked by indicators that have not caught up yet. It does
**not** relax any risk gate, and the `S >= 0.85` bar still applies.

This is the mechanised version of post-earnings-announcement drift (PEAD): the
market under-reacts to a shock, so the first bars after the shock carry the edge.

---

## 3. CROSS-ASSET CORRELATION MATRIX

Rolling Pearson correlation over a trailing window (`w = 60`), computed by the
same formula the C++ `RiskManager::correlation` uses (0 when variance is 0).

Default lead–lag expectations (the mandate's examples, made explicit):

| driver | responder | expected sign | rationale |
|---|---|---|---|
| GOLD | USDJPY | −1 | gold up ⇒ yen-funded carry unwinds |
| OIL | CADUSD | +1 | petro-currency |
| OIL | AIRLINES | −1 | input cost |
| BTC | QQQ | +1 | risk-on beta |
| USD | GOLD | −1 | dollar numeraire |

The matrix serves as **confirmation or invalidation** only. It never generates a
signal by itself, and the empirical correlation is journalled alongside the
assumed sign so a stale assumption is visible rather than silent.

---

## 4. ADAPTIVE POSITION SIZING (Kelly + volatility)

```
f_conf = 0.75 + 0.50 · clamp((S - 0.85) / 0.15, 0, 1)      -> 0.75 .. 1.25
g_vol  = clamp(atr_ref_pct / atr_pct, 0.5, 1.5)            inverse-vol target
risk_pct = clamp(base_risk · f_conf · g_vol, 0.01, 0.02)   <- HARD 2% CEILING
qty    = equity · risk_pct / (price · stop_pct)
```

The `[1%, 2%]` band is deliberately the **same** band the C++ engine's
`RiskManager::size_by_risk` enforces (`kMinRiskPerTradePct .. kMaxRiskPerTradePct`),
so the Python plan and the engine can never disagree about how much risk is
allowed. The formula is the same risk-budget identity the engine already uses;
the Monster only supplies a *smarter `risk_pct`*. The engine remains the
authority — even a bug in the Monster layer cannot exceed 2%. Kelly enters only
as the existing quarter-Kelly fraction in `RiskLimits`; we never bet full Kelly.

---

## 5. REGIME ENGINE — trend vs mean-reversion, measured not asserted

The first cut of the R layer used **one** number: the slope of SMA200. That is a
low-volatility filter wearing a trend filter's coat. On equities low vol and
uptrends coincide, so a vol proxy *looks* like a trend detector — and then stops
working the moment you move to FX, where the two are decoupled. §1.3 keeps that
rule as `regime_mode="legacy"` for A/B, but the default is now an ensemble.

`tools/monster/regime.py`. Two axes, deliberately separated.

### 5.1 Directional axis — eight independent views, blended

| statistic | trending when | mean-reverting when | source |
|---|---|---|---|
| variance ratio `VR(q)` | `VR > 1` | `VR < 1` | Lo & MacKinlay (1988) |
| Hurst exponent `E[R/S]_n ∝ n^H` | `H > 0.5` | `H < 0.5` | Hurst (1951) |
| efficiency ratio `ER = |P_t−P_{t−n}| / Σ|ΔP|` | high | low | Kaufman (1995) |
| ADX | `> 25` | `< 20` | Wilder (1978) |
| choppiness `100·log10(ΣTR/range)/log10(n)` | `< 38.2` | `> 61.8` | — |
| linreg `R²` of close on time | high | low | — |
| lag-1 autocorrelation `ρ(1)` | `> 0` | `< 0` | — |
| MA-ribbon consistency | `→ 1` or `→ 0` | `≈ 0.5` | — |

```
VR(q) = Var(r^(q)) / (q · Var(r^(1)))          overlapping, heteroskedasticity-robust
trend_score = Σ w_k · norm_k(stat_k) / Σ w_k    over whichever stats are finite
```

Each statistic is normalized so **0.5 means "random walk, no opinion"**. Weights
are renormalized over the finite ones, so a missing statistic cannot silently
drag the blend toward neutral.

**Three findings from building this, each of which changed the code:**

1. **Hurst must run on the INCREMENTS.** Feeding it the price level measures the
   self-affinity of a non-stationary path and returns `H > 1` — not a number
   Hurst can be. On returns the calibration is the textbook one (0.5 = random
   walk). The test suite guards this directly.

2. **ER is lookback-sensitive, so it is swept.** ER(20) on a market that pulled
   back over the last 20 bars inside a 200-bar uptrend reads ~0.04, which flipped
   the whole label to "range". ER is now computed at 10/20/40 and the *normalized*
   readings are averaged. A short pause no longer erases a long trend.

3. **`VR` and `ρ(1)` are structurally NEUTRAL on a drift trend.** A constant
   drift leaves returns iid, so both correctly report "no autocorrelation" — and
   a market that grinds up 49% over the window still only scores ~0.6. That is
   not a defect in those statistics, it is what they measure. So a statistic that
   *does* separate drift trends was added: **MA-ribbon consistency** (the
   fraction of the window where SMA20 sits above SMA50; ~0.5 on a random walk,
   ~1.0 on a sustained trend). Adding it moved the trend-series minimum from
   0.542 to 0.602 and dropped the random-walk median from 0.566 to 0.525.

### 5.2 Threshold calibration (measured, not guessed)

Fraction of BARS whose raw `trend_score` clears each bar, 6 seeds × 400 bars per
series type:

| threshold | trending series | OU series | random walk |
|---|---|---|---|
| 0.60 | 0.896 | 0.008 | 0.211 |
| 0.62 | 0.884 | 0.005 | 0.182 |
| **0.65** | **0.860** | **0.003** | **0.126** |
| 0.70 | 0.799 | 0.001 | 0.069 |

`trend_hi = 0.65`: 86% of true trend bars, 12.6% of random-walk bars, and
hysteresis (enter 0.66 / exit 0.55) cuts the churn further. The overlap is
**real and irreducible** — the directional axis is the noisy one. This is exactly
why the router consumes the *continuous* `trend_score` and the label is only a
coarse summary.

### 5.3 Volatility axis

Yang-Zhang (2000), gap-aware and ~14× more efficient than close-to-close:

```
σ² = σ²_o + k·σ²_c + (1−k)·σ²_rs      k = 0.34 / (1.34 + (n+1)/(n−1))
```

Three failure modes were found and fixed — the third is the one that mattered
most, and it was only visible once the axis was wired into the sniper:

* **YZ is a variance, so it collapses.** On a series with constant returns it is
  exactly 0 → NaN. But a market can have a 12% intraday range on an unchanged
  close, which is unambiguously stressed. So `ATR/close` (a *level*, which cannot
  collapse) is combined in with **max**, mirroring the M layer: either read being
  stressed is enough.
* **A flat history makes the percentile degenerate.** Ranking a value against a
  history of identical values ties everywhere, `<=` holds for all of them, and
  the percentile pins to **1.0** — reporting maximum stress for a dead-flat
  series. A flat history carries no information, so it now reads **0.5**.
* **The axis measured a BAR, not a REGIME.** Ranking the raw latest YZ against
  its own history fires on *every* volatility-expansion bar — and an entry bar is
  a volatility-expansion bar by nature. That is not a coincidence; it is the
  definition of a breakout. The arithmetic is fatal:

  ```
  max S while labelled high_vol = .25·1 + .35·1 + .25·0.25 + .15·1 = 0.8125 < 0.85
  ```

  So `high_vol` vetoes exactly the bars the sniper exists to find, and the gate
  can never open. Two changes fix it:

  1. **The percentile is taken on the smoothed estimate** — the mean of the last
     `vol_smooth = 5` YZ readings. Volatility clusters, so a regime is a
     *persistent* elevation. One spike moves the smoothed value by 1/5; a genuine
     regime shift moves all five. That is the difference between "this bar was
     big" and "this market is stressed".
  2. **`high_vol` additionally requires absolute stress** (`atr_stress > 0`, i.e.
     `ATR/close` above half the mandate's `atr_hi`). A pure percentile threshold
     at 0.90 marks ~10–15% of *all* bars high_vol by construction — it is a
     relative rank, so a dead-calm market still has a stressed-looking decile.

  The label now needs **both** reads, and `RegimeState.stressed` carries the
  conjunction so the hysteresis tracker latches on the same condition the label
  uses.

### 5.4 Tradeability — is mean reversion actually harvestable?

```
ΔX_t = a + λ·X_{t−1} + ε        half_life = −ln(2)/λ        (λ < 0 required)
```

Recovered within 3× of theory on a simulated OU process (2.36 bars measured vs
2.31 theoretical at θ = 0.30). But a half-life **alone is not evidence**: a pure
random walk routinely yields a spuriously short one (~7 bars on synthetic GBM).
So `mean_reversion_tradeable` requires a short half-life **AND** at least one
independent statistic to agree this is not a random walk (`VR < 0.95` or
`H < 0.45`). Without that corroboration the mean-reversion strategy would fade
random walks all day.

---

### 5.5 The 0.85 ceiling is a REGIME question

With `M = T = C = 1.0` — a maximal volume anomaly, all six factors agreeing, every
peer confirming — the reachable maximum of `S` depends only on the R layer:

| regime | R | ceiling on S | can host S ≥ 0.85? |
|---|---|---|---|
| `trend_up` | 1.00 | 1.0000 | **yes** |
| `range` | 0.50 | 0.8750 | yes, barely |
| `high_vol` | 0.25 | **0.8125** | **no — mathematically impossible** |
| `trend_down` | 0.00 | 0.7500 | no (and vetoed) |

So `S ≥ 0.85` is a statement about the *regime*, not about the score. A "0.85 is
unreachable" bug therefore presents as a regime problem long before it presents
as a score problem, and `test_monster_sniper.test_reachability` asserts this table
directly rather than hoping a fixture happens to trip the bar.

The measured maximum on a real constructed setup (pullback-and-reclaim inside an
uptrend, on a volume burst) is **S = 0.890**, with `vwap-reclaim + macd-thrust +
trend-align`, `R = 1.0`, `M = 1.00`. Getting there required the §5.3 fix.

---

## 6. STRATEGY ENSEMBLE + REGIME ROUTER

Momentum and mean reversion need **opposite** conditions, so any single fixed
strategy is wrong in half of all regimes. `tools/monster/strategies.py` holds the
zoo; `tools/monster/router.py` decides who gets listened to.

| strategy | regime fit | signal |
|---|---|---|
| `momentum` | trend | Donchian breakout of the **prior** n-bar range + MA ribbon |
| `mean_reversion` | range | z-score fade, gated on OU half-life |
| `breakout` | both | Bollinger-inside-Keltner squeeze → expansion |
| `ofi` | both | order flow imbalance → expected move (§6.2) |
| `vol_target` | — | inverse-variance exposure scaler (sizing overlay, not a direction) |

### 6.1 Continuous allocation, never a binary switch

```
w_trend = sigmoid(k·(trend_score − 0.5))            k = 6
weight_i = regime_weight(fit_i, w_trend) · confidence_i
conviction = Σ weight_i·direction_i / Σ weight_i      ∈ [−1, 1]
agreement  = Σ_{agreeing} weight_i / Σ weight_i       ∈ [ 0, 1]
```

The obvious implementation — "if trending use momentum else use mean reversion" —
is the expensive one: every flip pays the full spread out and back, and regime
transitions are exactly when the classifier is least sure. Near `trend_score=0.5`
the router holds both engines at ~half size and degrades through the ambiguity.

### 6.2 Order flow imbalance (Cont, Kukanov & Stoikov 2014)

```
OFI = Σ  [ 1{Pb_n ≥ Pb_n−1}·qb_n − 1{Pb_n ≤ Pb_n−1}·qb_n−1 ]
       − [ 1{Pa_n ≤ Pa_n−1}·qa_n − 1{Pa_n ≥ Pa_n−1}·qa_n−1 ]
ΔP_k = β·OFI_k + ε_k                    β = c / depth^λ     (λ = 1 stylized)
```

The paper reports a linear OFI→price-change relation with **R² of 35–79% across
50 US stocks** (most names 65–70%) at 10-second intervals. Two caveats stated up
front because they matter:

1. That R² is **contemporaneous, not predictive**. OFI is not a crystal ball. The
   tradable part is that OFI is autocorrelated and price impact is only partly
   permanent, so the next interval's flow is partly knowable.
2. It needs **depth**. Free OHLCV feeds do not publish the book. Without it we
   fall back to a signed-volume proxy — a *different and weaker* object, since it
   cannot see limit orders or cancellations, which is most of what OFI measures.
   The signal's reason string says `NOT-cks` and its confidence is discounted 30%
   rather than quietly pretending.

### 6.3 The ensemble can only BLOCK, never promote

`S` stays exactly the mandate's weighted sum. The ensemble adds one gate:

```
veto "ensemble-opposed"  ⇔  agreement ≥ 0.60  AND  conviction ≤ −0.35
                            AND  total active weight ≥ 0.50
```

The weight floor is not decoration. `agreement` is a **share**, so a single
active strategy always scores 1.00 — one lonely, low-confidence OFI proxy would
otherwise veto every long in the book. (It did, in a live smoke test. The
regression test is now in the suite.) An ensemble that can talk you *into* a
trade is an ensemble that can talk you into a bad one, so this layer is
structurally incapable of raising `S`.

### 6.4 A structural finding about the six technical factors

Measured over a 340-bar fixture, the factor counts were:

```
trend-align 140 | macd-thrust 43 | vwap-reclaim 6 | bb-oversold 1 | rsi-divergence 0
```

**The six "independent" factors are not independent.** They cluster into a
*trend* group (`trend-align`, `macd-thrust`) that fires constantly in a trend, and
a *pullback* group (`vwap-reclaim`, `bb-oversold`, `rsi-divergence`, `fib-support`)
that is rare inside one. The consequence is that requiring `votes ≥ 3` effectively
means **a pullback that reclaims inside an uptrend** — which is a coherent and
demanding setup (arguably exactly what a sniper entry is), but it is much rarer
than "three of six boxes ticked" suggests. On a *steady* uptrend the maximum
reachable is 2 votes. This is why the M layer (a genuine volume anomaly) carries
its own weight rather than being a tiebreaker.

---

## 7. FUNDING-RATE CARRY — the one non-directional edge

`tools/monster/funding.py`. Everything else here tries to be right about where
price goes. This does not care.

Perpetuals have no expiry, so exchanges invented the funding rate to tether them
to spot: when longs dominate, longs pay shorts. Hold spot long and perp short in
equal notional and the price exposure cancels, leaving the funding payment as the
only P&L. It is a **carry harvest, not a forecast**.

```
APR(gross) = rate_per_period × periods_per_year      (8h → 1095)
net = gross − roundtrip_fees·(365/holding) − spread·(365/holding) − cost_of_capital
```

**The mistake this module exists to prevent:** ranking on the gross number. A
worked example from the tests —

| | gross APR | basis | spot spread | verdict |
|---|---|---|---|---|
| BTCUSDT 0.012%/8h | 13.14% | 2 bps | 1 bp | clean, but net only 2.22% at a 30-day hold |
| MEME 0.020%/8h | **21.90%** | **500 bps** | **150 bps** | **blocked** — basis-shock + illiquid-spot |

The memecoin's gross is 67% higher and it is uninvestable. It ranks *below* the
clean trade.

Risk flags, all blocking: `funding-unstable` (history mostly negative),
`funding-flips` (sign churn), `rate-near-zero` (about to cross),
`basis-shock` (> 300 bps), `illiquid-spot` (> 50 bps spread), `below-hurdle`
(< 10% net).

**Delta-neutral is not risk-free.** The funding rate can flip sign and make you
the payer; the perp leg can be liquidated on a violent basis move and leave you
naked long spot; the spot venue can freeze withdrawals; a stablecoin can de-peg
and your "yield" is denominated in a depreciating asset. All four are flagged,
none are assumed away.

---

## 8. What this does **not** claim

* No guaranteed profit, no "win rate", no target return.
* The confidence score is a *confluence filter*, not a probability of profit.
  `S = 0.90` means "nine-tenths of the evidence I look at agrees", nothing more.
* The 3% daily / 6% weekly kill-switch and the 2% per-trade ceiling are enforced
  in the C++ risk engine and are not adjustable by the Monster layer.
