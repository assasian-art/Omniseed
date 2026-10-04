# =============================================================================
#  OmniSeed — tools/monster/features.py
#  Pure, deterministic feature primitives for the Monster layer.
#
#  NO LOOK-AHEAD CONTRACT (enforced by tests/test_monster_features.py):
#    every function computes the value at index `i` using ONLY inputs at
#    indices <= i. Anything that structurally needs future bars (swing
#    detection) returns *all* swings for offline analysis but is consumed
#    through `confirmed_*` helpers that only expose swings whose confirmation
#    bar (idx + right) is already in the past.
#
#  Undefined values are float('nan') (not 0.0) so a caller can never mistake
#  "not enough history" for a real reading. Use math.isfinite().
# =============================================================================
"""No-look-ahead indicator/statistics primitives."""

import math

NAN = float("nan")


def _nan_list(n):
    return [NAN] * n


def closes(bars):
    """bars: sequence of (time, open, high, low, close, volume) or dicts."""
    out = []
    for b in bars:
        out.append(float(b[4]) if not isinstance(b, dict) else float(b["close"]))
    return out


def highs(bars):
    return [float(b[2]) if not isinstance(b, dict) else float(b["high"])
            for b in bars]


def lows(bars):
    return [float(b[3]) if not isinstance(b, dict) else float(b["low"])
            for b in bars]


def volumes(bars):
    return [float(b[5]) if not isinstance(b, dict) else float(b["volume"])
            for b in bars]


def opens(bars):
    return [float(b[1]) if not isinstance(b, dict) else float(b["open"])
            for b in bars]


# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------
def sma(vals, period):
    """Simple moving average; NaN until `period` samples exist."""
    n = len(vals)
    out = _nan_list(n)
    if period <= 0:
        return out
    run = 0.0
    for i, v in enumerate(vals):
        run += v
        if i >= period:
            run -= vals[i - period]
        if i >= period - 1:
            out[i] = run / period
    return out


def ema(vals, period):
    """Exponential MA, seeded with the SMA of the first `period` samples."""
    n = len(vals)
    out = _nan_list(n)
    if period <= 0 or n < period:
        return out
    seed = sum(vals[:period]) / period
    out[period - 1] = seed
    k = 2.0 / (period + 1.0)
    prev = seed
    for i in range(period, n):
        prev = vals[i] * k + prev * (1.0 - k)
        out[i] = prev
    return out


# ---------------------------------------------------------------------------
# Oscillators
# ---------------------------------------------------------------------------
def rsi(vals, period=14):
    """Wilder's RSI. NaN until `period` changes exist."""
    n = len(vals)
    out = _nan_list(n)
    if n < period + 1 or period <= 0:
        return out
    gain = loss = 0.0
    for i in range(1, period + 1):
        d = vals[i] - vals[i - 1]
        gain += max(d, 0.0)
        loss += max(-d, 0.0)
    avg_g, avg_l = gain / period, loss / period
    out[period] = 100.0 if avg_l == 0.0 else 100.0 - 100.0 / (1.0 + avg_g / avg_l)
    for i in range(period + 1, n):
        d = vals[i] - vals[i - 1]
        avg_g = (avg_g * (period - 1) + max(d, 0.0)) / period
        avg_l = (avg_l * (period - 1) + max(-d, 0.0)) / period
        out[i] = 100.0 if avg_l == 0.0 else 100.0 - 100.0 / (1.0 + avg_g / avg_l)
    return out


def macd(vals, fast=12, slow=26, signal=9):
    """-> (macd_line, signal_line, histogram). All NaN-aligned to `vals`."""
    ef, es = ema(vals, fast), ema(vals, slow)
    line = [NAN] * len(vals)
    for i in range(len(vals)):
        if math.isfinite(ef[i]) and math.isfinite(es[i]):
            line[i] = ef[i] - es[i]
    valid = [v for v in line if math.isfinite(v)]
    sig_valid = ema(valid, signal) if valid else []
    sig = [NAN] * len(vals)
    j = 0
    for i in range(len(vals)):
        if math.isfinite(line[i]):
            if j < len(sig_valid):
                sig[i] = sig_valid[j]
            j += 1
    hist = [NAN] * len(vals)
    for i in range(len(vals)):
        if math.isfinite(line[i]) and math.isfinite(sig[i]):
            hist[i] = line[i] - sig[i]
    return line, sig, hist


def true_range(bars):
    h, l, c = highs(bars), lows(bars), closes(bars)
    n = len(bars)
    out = _nan_list(n)
    for i in range(n):
        if i == 0:
            out[i] = h[i] - l[i]
        else:
            out[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    return out


def atr(bars, period=14):
    """Wilder-smoothed Average True Range."""
    tr = true_range(bars)
    n = len(bars)
    out = _nan_list(n)
    if n < period or period <= 0:
        return out
    prev = sum(tr[:period]) / period
    out[period - 1] = prev
    for i in range(period, n):
        prev = (prev * (period - 1) + tr[i]) / period
        out[i] = prev
    return out


def atr_pct(bars, period=14):
    """ATR as a fraction of close — the volatility the sizer targets."""
    a, c = atr(bars, period), closes(bars)
    return [a[i] / c[i] if (math.isfinite(a[i]) and c[i]) else NAN
            for i in range(len(bars))]


# ---------------------------------------------------------------------------
# Bands / VWAP
# ---------------------------------------------------------------------------
def bollinger(vals, period=20, sigma=2.0):
    """-> (mid, upper, lower). Population std, trailing window."""
    n = len(vals)
    mid, up, lo = _nan_list(n), _nan_list(n), _nan_list(n)
    for i in range(period - 1, n):
        w = vals[i - period + 1:i + 1]
        m = sum(w) / period
        var = sum((x - m) ** 2 for x in w) / period
        sd = math.sqrt(var)
        mid[i], up[i], lo[i] = m, m + sigma * sd, m - sigma * sd
    return mid, up, lo


def rolling_vwap(bars, period=20):
    """Trailing volume-weighted average of the typical price."""
    n = len(bars)
    out = _nan_list(n)
    h, l, c, v = highs(bars), lows(bars), closes(bars), volumes(bars)
    for i in range(period - 1, n):
        num = den = 0.0
        for j in range(i - period + 1, i + 1):
            tp = (h[j] + l[j] + c[j]) / 3.0
            num += tp * v[j]
            den += v[j]
        out[i] = num / den if den > 0 else NAN
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def rolling_zscore(vals, period=20):
    """Trailing z-score of the current sample vs the previous `period` samples."""
    n = len(vals)
    out = _nan_list(n)
    for i in range(period, n):
        w = vals[i - period:i]              # strictly the PAST window
        m = sum(w) / period
        var = sum((x - m) ** 2 for x in w) / period
        sd = math.sqrt(var)
        out[i] = (vals[i] - m) / sd if sd > 0 else 0.0
    return out


def rolling_corr(a, b, period=60):
    """Trailing Pearson correlation. NaN until `period` pairs exist."""
    n = min(len(a), len(b))
    out = _nan_list(n)
    for i in range(period - 1, n):
        wa = a[i - period + 1:i + 1]
        wb = b[i - period + 1:i + 1]
        ma, mb = sum(wa) / period, sum(wb) / period
        num = sum((wa[j] - ma) * (wb[j] - mb) for j in range(period))
        va = sum((x - ma) ** 2 for x in wa)
        vb = sum((x - mb) ** 2 for x in wb)
        out[i] = num / math.sqrt(va * vb) if va > 0 and vb > 0 else 0.0
    return out


def log_returns(vals):
    """r_i = ln(v_i / v_{i-1}); index 0 is NaN."""
    out = _nan_list(len(vals))
    for i in range(1, len(vals)):
        if vals[i] > 0 and vals[i - 1] > 0:
            out[i] = math.log(vals[i] / vals[i - 1])
    return out


def amihud(bars):
    """Amihud (2002) illiquidity: |return| / dollar volume. Higher = thinner."""
    c, v = closes(bars), volumes(bars)
    r = log_returns(c)
    out = _nan_list(len(bars))
    for i in range(len(bars)):
        dv = c[i] * v[i]
        if math.isfinite(r[i]) and dv > 0:
            out[i] = abs(r[i]) / dv
    return out


def kyle_lambda(bars):
    """Per-unit price-impact proxy: |dP| / volume.

    Kyle's lambda is dP per unit of SIGNED order flow; with bar data we use the
    Lee-Ready tick rule (sign of the bar return) as the flow-sign proxy, which
    cancels to |dP| / volume. Reported for honest slippage, never for scoring.
    """
    c, v = closes(bars), volumes(bars)
    out = _nan_list(len(bars))
    for i in range(1, len(bars)):
        if v[i] > 0:
            out[i] = abs(c[i] - c[i - 1]) / v[i]
    return out


# ---------------------------------------------------------------------------
# Swings + Fibonacci (look-ahead aware)
# ---------------------------------------------------------------------------
def find_swings(vals, left=2, right=2, kind="low"):
    """All swing points in the series as (index, value).

    A swing low at i requires vals[i] strictly below the `left` bars before AND
    the `right` bars after it. Because it reads `right` future bars, the point
    is only *knowable* at index i+right — consumers must go through
    `confirmed_swings()`. Returned here for offline analysis/tests.
    """
    n = len(vals)
    out = []
    for i in range(left, n - right):
        ok = True
        for k in range(1, left + 1):
            if not (vals[i] < vals[i - k]):
                ok = False
                break
        if ok:
            for k in range(1, right + 1):
                if not (vals[i] < vals[i + k]):
                    ok = False
                    break
        if ok:
            out.append((i, vals[i]))
    return out


def confirmed_swings(swings, i, right=2):
    """Swings already knowable at index `i` (confirmation bar <= i)."""
    return [(idx, v) for (idx, v) in swings if idx + right <= i]


def fib_levels(swing_low, swing_high, ratios=(0.382, 0.5, 0.618)):
    """Retracement levels of an up-swing (buy-the-dip supports)."""
    span = swing_high - swing_low
    return [swing_high - span * f for f in ratios]


def last_swing_pair(swing_lows, swing_highs, i, right=2):
    """Newest confirmed (low, high) with low_idx < high_idx, else None."""
    lows_c = confirmed_swings(swing_lows, i, right)
    highs_c = confirmed_swings(swing_highs, i, right)
    best = None
    for li, lv in lows_c:
        for hi, hv in highs_c:
            if li < hi and hv > lv:
                if best is None or hi > best[2]:
                    best = (li, lv, hi, hv)
    return best


def bullish_divergence(rsi_vals, swing_lows, i, right=2, max_gap=60):
    """Price lower-low + RSI higher-low across two confirmed swing lows."""
    lows_c = confirmed_swings(swing_lows, i, right)
    if len(lows_c) < 2:
        return False
    (i1, p1), (i2, p2) = lows_c[-2], lows_c[-1]
    if not (0 < i2 - i1 <= max_gap):
        return False
    r1, r2 = rsi_vals[i1], rsi_vals[i2]
    if not (math.isfinite(r1) and math.isfinite(r2)):
        return False
    return p2 < p1 and r2 > r1


# ---------------------------------------------------------------------------
# Option-chain primitives (M4 — volatility harvesting)
# ---------------------------------------------------------------------------

def option_chain_fields(bars):
    """Return the set of option-chain keys available in the first bar.

    Expected keys (dict bars): iv_30d, rv_30d, call_iv_90pct, put_iv_90pct,
    skew_25d_delta, term_structure, gex. Missing keys are silently ignored.
    """
    if not bars:
        return set()
    b = bars[0]
    if isinstance(b, dict):
        return set(b.keys())
    return set()


def read_option_chain(bar):
    """Extract option-chain metrics from a single bar.

    Returns a dict with keys: iv_30d, rv_30d, call_iv_90pct, put_iv_90pct,
    skew_25d_delta, term_structure, gex. Missing values are NAN.

    DICT BARS ONLY, deliberately. The OHLCV tuple schema used everywhere else
    in the kernel is (time, open, high, low, close, volume), so index 5 is
    VOLUME. Reading an option chain off a tuple would silently promote volume
    to iv_30d -- a 1e6 share bar would look like 1e6 implied vol and every
    vol strategy would fire. An option chain arrives either as dict bars or as
    its own CSV stream (see the M4 brief), never packed into an OHLCV tuple,
    so we FAIL CLOSED on tuples instead of guessing.
    """
    out = {"iv_30d": NAN, "rv_30d": NAN, "call_iv_90pct": NAN,
           "put_iv_90pct": NAN, "skew_25d_delta": NAN,
           "term_structure": 0.0, "gex": 0.0}
    if not isinstance(bar, dict):
        return out
    for k in out:
        if k in bar:
            v = bar[k]
            out[k] = float(v) if (v is not None and math.isfinite(float(v))) else NAN
    return out


def iv_rv_ratio(bars, i):
    """IV_30d / RV_30d ratio at bar i. NAN if either is missing."""
    if i < 0 or not bars:
        return NAN
    oc = read_option_chain(bars[i])
    iv, rv = oc["iv_30d"], oc["rv_30d"]
    if not math.isfinite(iv) or not math.isfinite(rv) or rv <= 0:
        return NAN
    return iv / rv


def realized_variance(vals, n):
    """Trailing realized variance of log-returns over `n` bars."""
    if len(vals) < 2 or n < 2:
        return NAN
    r = []
    for i in range(1, len(vals)):
        if vals[i] > 0 and vals[i - 1] > 0:
            r.append(math.log(vals[i] / vals[i - 1]))
    if len(r) < n:
        return NAN
    w = r[-n:]
    m = sum(w) / len(w)
    return sum((x - m) ** 2 for x in w) / len(w)
