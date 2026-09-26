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

## 5. What this does **not** claim

* No guaranteed profit, no "win rate", no target return.
* The confidence score is a *confluence filter*, not a probability of profit.
  `S = 0.90` means "nine-tenths of the evidence I look at agrees", nothing more.
* The 3% daily / 6% weekly kill-switch and the 2% per-trade ceiling are enforced
  in the C++ risk engine and are not adjustable by the Monster layer.
