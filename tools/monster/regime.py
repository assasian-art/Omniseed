# =============================================================================
#  OmniSeed — tools/monster/regime.py
#  Multi-axis, CAUSAL market-regime engine.
#
#  The mandate's R layer used one number (the slope of SMA200). That is a
#  low-vol filter wearing a trend filter's coat: on equities low volatility and
#  uptrends coincide, so a vol proxy looks like a trend detector and then
#  quietly fails on FX. This module replaces it with an ENSEMBLE of independent
#  statistics that disagree in useful ways.
#
#  DIRECTIONAL AXIS — "is this market trending or mean-reverting?"
#    variance ratio  VR(q) = Var(r^q) / (q·Var(r))     Lo & MacKinlay (1988)
#    Hurst exponent  E[R/S]_n ∝ n^H                    Hurst (1951), R/S
#    efficiency ratio ER = |P_t - P_{t-n}| / Σ|ΔP|      Kaufman (1995)
#    ADX / DMI       trend STRENGTH, not direction      Wilder (1978)
#    choppiness      100·log10(ΣATR/(maxH-minL))/log10(n)
#    linreg R²       how line-like the window is
#    lag-1 autocorr  ρ(1) of log returns
#  Each is normalized to [0,1] (0.5 = random walk) and blended. The blend is
#  the trend_score. VR(q) > 1 / H > 0.5 / ρ(1) > 0 all say TREND; < 1 / < 0.5 /
#  < 0 all say MEAN REVERSION; the null hypothesis is always the random walk.
#
#  VOLATILITY AXIS — the most reliably detectable regime (vol clusters).
#    Yang-Zhang (2000) estimator: gap-aware, and ~14x more efficient than
#    close-to-close at the same sample size. Ranked against its own trailing
#    history so "high vol" means high FOR THIS ASSET, not high vs the S&P.
#
#  TRADEABILITY — an OU half-life from an OLS of ΔX on X_{t-1}:
#      half_life = -ln(2)/λ        (λ < 0 required)
#  A 3-bar half-life is scalpable; a 200-bar half-life is not a mean-reversion
#  trade, it is a slow bleed. Mean reversion is gated on this, not on a vibe.
#
#  NO LOOK-AHEAD: every statistic is computed on a window ENDING at index i.
#  `detect()` is pure. `RegimeTracker` adds hysteresis using only its own past
#  state. tests/test_monster_regime.py proves prefix invariance for each one.
#
#  HONEST SCOPE: a regime label is a description of the recent past, not a
#  forecast. Nothing here claims an edge. See docs/MONSTER_DESIGN.md §6.
# =============================================================================
"""Causal multi-axis regime detection: trend/reversion, volatility, tradeability."""

import math

from .features import NAN, atr, closes, highs, lows, opens, sma

# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


def _ramp(x, lo, hi):
    """Linear 0..1 ramp: lo -> 0, hi -> 1. NaN -> NaN."""
    if not math.isfinite(x):
        return NAN
    if hi == lo:
        return 0.5
    return _clamp((x - lo) / (hi - lo), 0.0, 1.0)


def _ols(xs, ys):
    """-> (slope, intercept, r2). r2 is 0 when the fit has no variance."""
    n = len(xs)
    if n < 2:
        return NAN, NAN, NAN
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    if sxx <= 0:
        return NAN, NAN, NAN
    slope = sxy / sxx
    inter = my - slope * mx
    syy = sum((y - my) ** 2 for y in ys)
    if syy <= 0:
        return slope, inter, 0.0
    r2 = (sxy * sxy) / (sxx * syy)
    return slope, inter, _clamp(r2, 0.0, 1.0)


def _mean(v):
    return sum(v) / len(v) if v else NAN


def _var(v):
    """Population variance."""
    n = len(v)
    if n < 2:
        return NAN
    m = sum(v) / n
    return sum((x - m) ** 2 for x in v) / n


# ---------------------------------------------------------------------------
# DIRECTIONAL AXIS
# ---------------------------------------------------------------------------
def variance_ratio(returns, q=2):
    """Lo-MacKinlay (1988) variance ratio -> (VR(q), robust z-statistic).

    VR > 1  positive autocorrelation -> TREND
    VR = 1  random walk
    VR < 1  negative autocorrelation -> MEAN REVERSION

    Uses the overlapping-return estimator with the HETEROSKEDASTICITY-ROBUST
    standard error (the homoskedastic one over-rejects badly on fat-tailed
    returns, which is all financial returns).
    """
    r = [x for x in returns if math.isfinite(x)]
    n = len(r)
    if q < 2 or n < q + 1:
        return NAN, NAN
    mu = sum(r) / n
    var1 = sum((x - mu) ** 2 for x in r) / (n - 1)
    if var1 <= 0:
        return NAN, NAN
    # unbiased numerator: m = q(n-q+1)(1 - q/n)
    m = q * (n - q + 1) * (1.0 - float(q) / n)
    if m <= 0:
        return NAN, NAN
    acc = 0.0
    for t in range(q - 1, n):
        s = 0.0
        for k in range(q):
            s += r[t - k]
        acc += (s - q * mu) ** 2
    vr = (acc / m) / var1

    denom = sum((x - mu) ** 2 for x in r) ** 2
    if denom <= 0:
        return vr, NAN
    theta = 0.0
    for j in range(1, q):
        num = 0.0
        for t in range(j, n):
            num += (r[t] - mu) ** 2 * (r[t - j] - mu) ** 2
        theta += (2.0 * (q - j) / q) ** 2 * (num / denom)
    if theta <= 0:
        return vr, NAN
    return vr, (vr - 1.0) / math.sqrt(theta)


def hurst_rs(vals, min_n=8, num_sizes=8):
    """Hurst exponent via rescaled range: E[R/S]_n ∝ n^H.

    H < 0.5 anti-persistent (MEAN REVERTING)
    H = 0.5 random walk
    H > 0.5 persistent (TRENDING)

    Deliberately a SOFT indicator: it is slow and noisy on short windows, so it
    carries modest weight in the blend and is never a hard switch on its own.
    """
    x = [v for v in vals if math.isfinite(v)]
    N = len(x)
    if N < min_n * 3:
        return NAN
    max_n = N // 2
    if max_n <= min_n:
        return NAN
    sizes = sorted({int(round(min_n + (max_n - min_n) * k / max(1, num_sizes - 1)))
                    for k in range(num_sizes)})
    sizes = [n for n in sizes if min_n <= n <= max_n and n >= 4]
    xs, ys = [], []
    for n in sizes:
        chunks = N // n
        if chunks < 1:
            continue
        ratios = []
        for c in range(chunks):
            seg = x[c * n:(c + 1) * n]
            m = sum(seg) / n
            z = 0.0
            zs = []
            for v in seg:
                z += v - m
                zs.append(z)
            rng = max(zs) - min(zs)
            s = math.sqrt(sum((v - m) ** 2 for v in seg) / n)
            if s > 0 and rng > 0:
                ratios.append(rng / s)
        if ratios:
            xs.append(math.log(n))
            ys.append(math.log(sum(ratios) / len(ratios)))
    if len(xs) < 3:
        return NAN
    slope, _, _ = _ols(xs, ys)
    return slope


def efficiency_ratio(vals, n=20):
    """Kaufman efficiency ratio: net travel / total path length.

    ER -> 1 straight line (trending); ER -> 0 wandered (ranging).
    """
    if n < 2 or len(vals) < n + 1:
        return NAN
    seg = vals[-(n + 1):]
    if any(not math.isfinite(v) for v in seg):
        return NAN
    net = abs(seg[-1] - seg[0])
    path = sum(abs(seg[i] - seg[i - 1]) for i in range(1, len(seg)))
    return net / path if path > 0 else 0.0


def adx(bars, period=14):
    """Wilder's ADX — trend STRENGTH (direction-blind). >25 trend, <20 range."""
    n = len(bars)
    if n < 2 * period + 1:
        return NAN
    h, l, c = highs(bars), lows(bars), closes(bars)
    trs, pdm, ndm = [], [], []
    for i in range(1, n):
        trs.append(max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])))
        up = h[i] - h[i - 1]
        dn = l[i - 1] - l[i]
        pdm.append(up if (up > dn and up > 0) else 0.0)
        ndm.append(dn if (dn > up and dn > 0) else 0.0)

    def _wilder(v):
        out = []
        if len(v) < period:
            return out
        run = sum(v[:period])
        out.append(run)
        for i in range(period, len(v)):
            run = run - run / period + v[i]
            out.append(run)
        return out

    str_, sp, sn = _wilder(trs), _wilder(pdm), _wilder(ndm)
    if len(str_) < period + 1:
        return NAN
    dxs = []
    for k in range(len(str_)):
        if str_[k] <= 0:
            continue
        pdi = 100.0 * sp[k] / str_[k]
        ndi = 100.0 * sn[k] / str_[k]
        tot = pdi + ndi
        dxs.append(100.0 * abs(pdi - ndi) / tot if tot > 0 else 0.0)
    if len(dxs) < period:
        return NAN
    a = sum(dxs[:period]) / period
    for i in range(period, len(dxs)):
        a = (a * (period - 1) + dxs[i]) / period
    return a


def choppiness(bars, n=14):
    """Choppiness index. >61.8 choppy/range, <38.2 trending."""
    if len(bars) < n + 1 or n < 2:
        return NAN
    h, l, c = highs(bars), lows(bars), closes(bars)
    seg = bars[-(n + 1):]
    hh, ll = highs(seg), lows(seg)
    rng = max(hh) - min(ll)
    if rng <= 0:
        return NAN
    tr_sum = 0.0
    off = len(bars) - (n + 1)
    for i in range(off + 1, len(bars)):
        tr_sum += max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    if tr_sum <= 0:
        return NAN
    return 100.0 * math.log10(tr_sum / rng) / math.log10(n)


def linreg_trend(vals, n=None):
    """-> (slope_per_bar, r2) of a linear fit to the window."""
    x = [v for v in vals if math.isfinite(v)]
    if n is not None:
        x = x[-n:]
    if len(x) < 3:
        return NAN, NAN
    idx = [float(i) for i in range(len(x))]
    slope, _, r2 = _ols(idx, x)
    return slope, r2


def autocorr1(returns):
    """Lag-1 autocorrelation of returns. >0 trend, <0 mean reversion."""
    r = [x for x in returns if math.isfinite(x)]
    n = len(r)
    if n < 10:
        return NAN
    m = sum(r) / n
    num = sum((r[i] - m) * (r[i - 1] - m) for i in range(1, n))
    den = sum((x - m) ** 2 for x in r)
    return num / den if den > 0 else NAN


def trend_consistency(vals, fast=20, slow=50):
    """Fraction of the window where the fast MA sits above the slow MA.

    Why this earns a seat: the variance ratio and rho(1) are structurally
    NEUTRAL on a drift trend, because a constant drift leaves returns iid and
    both statistics correctly report "no autocorrelation". So a market that
    grinds up 49% over the window can still only score ~0.6 on those two. This
    statistic measures the thing a trend-follower actually cares about — does
    the medium-term average keep sitting on the right side — and it separates
    cleanly: ~0.5 on a random walk, ~1.0 on a sustained trend.
    """
    n = len(vals)
    if n < slow + 5:
        return NAN
    sf, ss = sma(vals, fast), sma(vals, slow)
    agree = total = 0
    for i in range(n):
        if math.isfinite(sf[i]) and math.isfinite(ss[i]):
            total += 1
            if sf[i] > ss[i]:
                agree += 1
    if total < 10:
        return NAN
    return agree / float(total)


# ---------------------------------------------------------------------------
# VOLATILITY AXIS
# ---------------------------------------------------------------------------
def yang_zhang_vol(bars, n=20):
    """Yang-Zhang (2000) volatility estimator (per-bar, gap-aware).

        sigma^2 = sigma_o^2 + k*sigma_c^2 + (1-k)*sigma_rs^2
        k = 0.34 / (1.34 + (n+1)/(n-1))

    sigma_o  overnight (prev close -> open) variance
    sigma_c  open -> close variance
    sigma_rs Rogers-Satchell, which stays unbiased when the price drifts
    Combining all three makes it ~14x more efficient than close-to-close.
    """
    if len(bars) < n + 1 or n < 2:
        return NAN
    seg = bars[-(n + 1):]
    o, h, l, c = opens(seg), highs(seg), lows(seg), closes(seg)
    so, sc, srs = [], [], []
    for i in range(1, len(seg)):
        if o[i] <= 0 or c[i] <= 0 or o[i - 1] <= 0 or h[i] <= 0 or l[i] <= 0:
            continue
        so.append(math.log(o[i] / c[i - 1]))
        sc.append(math.log(c[i] / o[i]))
        srs.append(math.log(h[i] / c[i]) * math.log(h[i] / o[i])
                   + math.log(l[i] / c[i]) * math.log(l[i] / o[i]))
    if len(so) < 3:
        return NAN
    k = 0.34 / (1.34 + (n + 1.0) / (n - 1.0))
    v = _var(so) + k * _var(sc) + (1.0 - k) * _var(srs)
    return math.sqrt(v) if v > 0 else NAN


def percentile_rank(history, value):
    """Fraction of `history` <= value, in [0,1]. NaN-safe."""
    h = [x for x in history if math.isfinite(x)]
    if not h or not math.isfinite(value):
        return NAN
    return sum(1 for x in h if x <= value) / float(len(h))


# ---------------------------------------------------------------------------
# TRADEABILITY (is mean reversion actually harvestable?)
# ---------------------------------------------------------------------------
def ou_half_life(vals, n=60):
    """Half-life of mean reversion from an OLS of ΔX on X_{t-1}.

        ΔX_t = a + λ·X_{t-1} + ε        half_life = -ln(2)/λ

    Returns NaN unless λ < 0 (i.e. the series actually reverts). A 3-bar
    half-life is scalpable; a 200-bar half-life is not a mean-reversion trade.
    """
    x = [v for v in vals if math.isfinite(v)]
    if len(x) < 20:
        return NAN
    x = x[-(n + 1):]
    if len(x) < 20:
        return NAN
    lv = x[:-1]
    dv = [x[i + 1] - x[i] for i in range(len(x) - 1)]
    slope, _, _ = _ols(lv, dv)
    if not math.isfinite(slope) or slope >= 0:
        return NAN
    hl = -math.log(2.0) / slope
    return hl if hl > 0 else NAN


# ---------------------------------------------------------------------------
# Config / state
# ---------------------------------------------------------------------------
class RegimeConfig:
    """Thresholds. Deliberately module-level constants, not magic numbers."""

    window: int = 200           # lookback for the directional ensemble
    vr_q: tuple = (2, 4, 8)     # horizons swept for the variance ratio
    er_n: tuple = (10, 20, 40)  # ER is famously lookback-sensitive -> sweep it
    adx_period: int = 14
    chop_n: int = 14
    hurst_min_n: int = 8
    rho_window: int = 60
    yz_n: int = 20
    vol_smooth: int = 5         # a REGIME is sustained vol, not one big bar
    vol_hist: int = 120         # trailing history for the vol percentile
    atr_period: int = 14
    atr_hi: float = 0.05        # ATR/close treated as stressed (the mandate's)
    ou_n: int = 60
    sma_long: int = 200

    # blend weights (sum ~1.0; renormalized over whichever are finite)
    w_adx: float = 0.14
    w_er: float = 0.18
    w_chop: float = 0.13
    w_vr: float = 0.17
    w_hurst: float = 0.13
    w_r2: float = 0.09
    w_rho: float = 0.04
    w_consist: float = 0.12
    ma_fast: int = 20
    ma_slow: int = 50

    # label thresholds
    #
    # CALIBRATED, not guessed. Measured fraction of BARS whose raw trend_score
    # clears each bar (6 seeds x 400 bars per series type):
    #     thr     0.60    0.62    0.65    0.70
    #     trend  0.896   0.884   0.860   0.799
    #     ou     0.008   0.005   0.003   0.001
    #     random 0.211   0.182   0.126   0.069
    # 0.65 keeps 86% of true trend bars while cutting random-walk false
    # positives to 12.6%, and hysteresis cuts that further. The overlap is real
    # and unavoidable — the directional axis is the noisy one (volatility is the
    # reliable axis) — which is exactly why the router consumes the CONTINUOUS
    # trend_score and this label is only a coarse summary.
    trend_hi: float = 0.65      # trend_score at/above which we call it trending
    vol_stress: float = 0.90    # vol percentile at/above which we call high_vol
    # hysteresis: enter high, exit low, so the label does not flap on noise
    trend_enter: float = 0.66
    trend_exit: float = 0.55
    vol_enter: float = 0.92
    vol_exit: float = 0.80

    def __init__(self, **kw):
        for k, v in kw.items():
            if not hasattr(self, k):
                raise TypeError("unknown RegimeConfig field: %s" % k)
            setattr(self, k, v)


class RegimeState:
    """One bar's regime reading. Continuous scores + a discretised label."""

    __slots__ = ("ts", "trend_score", "vol_score", "label", "direction",
                 "adx", "er", "chop", "vr", "vr_z", "hurst", "r2", "rho1",
                 "yz_vol", "half_life", "components", "stressed")

    # numeric slots default to NaN, NEVER None: a None here propagates into
    # math.isfinite() downstream and raises TypeError far from the cause.
    _NUMERIC = ("trend_score", "vol_score", "adx", "er", "chop", "vr", "vr_z",
                "hurst", "r2", "rho1", "yz_vol", "half_life")

    def __init__(self, **kw):
        for s in self.__slots__:
            setattr(self, s, kw.get(s))
        for s in self._NUMERIC:
            if getattr(self, s) is None:
                setattr(self, s, NAN)
        if self.ts is None:
            self.ts = 0
        if self.direction is None:
            self.direction = ""
        if self.label is None:
            self.label = "range"
        if self.components is None:
            self.components = {}
        self.stressed = bool(self.stressed)

    # -- convenience -------------------------------------------------------
    @property
    def is_trend(self):
        return self.label in ("trend_up", "trend_down")

    @property
    def mean_reversion_tradeable(self):
        """Reverting, fast enough, AND corroborated by the ensemble.

        An OLS half-life is not on its own evidence of reversion: regressing
        ΔX on X_{t-1} on a pure random walk routinely produces a spurious short
        half-life (we measured ~7 bars on a synthetic GBM). So require the
        half-life to be short AND at least one independent statistic to agree
        that this is not a random walk.
        """
        hl = self.half_life
        if hl is None or not math.isfinite(hl) or hl > 30.0:
            return False
        vr_ok = self.vr is not None and math.isfinite(self.vr) and self.vr < 0.95
        h_ok = (self.hurst is not None and math.isfinite(self.hurst)
                and self.hurst < 0.45)
        return bool(vr_ok or h_ok)

    def detail(self):
        parts = ["regime=%s" % self.label, "trend=%.3f" % _f(self.trend_score),
                 "vol=%.2f" % _f(self.vol_score)]
        if self.direction:
            parts.append("dir=%s" % self.direction)
        if math.isfinite(_f(self.half_life)):
            parts.append("hl=%.0f" % self.half_life)
        return " ".join(parts)


def _f(x):
    return x if (x is not None and math.isfinite(x)) else NAN


# ---------------------------------------------------------------------------
# The detector
# ---------------------------------------------------------------------------
def _directional(win_bars, cfg):
    """-> (trend_score, components dict). Window must END at the bar of interest."""
    cl = closes(win_bars)
    comp = {}

    # variance ratio: average the normalized reading across horizons. Sweeping
    # q and looking at the SHAPE locates the horizon where structure lives.
    vrs, zs = [], []
    rets = [math.log(cl[i] / cl[i - 1]) for i in range(1, len(cl))
            if cl[i] > 0 and cl[i - 1] > 0]
    for q in cfg.vr_q:
        vr, z = variance_ratio(rets, q)
        if math.isfinite(vr):
            vrs.append(vr)
        if math.isfinite(z):
            zs.append(z)
    vr_mean = _mean(vrs) if vrs else NAN
    comp["vr"] = vr_mean
    comp["vr_z"] = _mean(zs) if zs else NAN

    a = adx(win_bars, cfg.adx_period)
    comp["adx"] = a
    # ER is lookback-sensitive by construction: a 20-bar pullback inside a
    # 200-bar uptrend drives ER(20) to ~0 and would flip the whole label. Sweep
    # several horizons and average the NORMALIZED readings, so a short-term
    # pause cannot erase a long-term trend.
    ers = [efficiency_ratio(cl, n) for n in cfg.er_n]
    ers = [x for x in ers if math.isfinite(x)]
    comp["er"] = _mean(ers) if ers else NAN
    ch = choppiness(win_bars, cfg.chop_n)
    comp["chop"] = ch
    # Hurst MUST run on the INCREMENTS. R/S analysis measures long memory in a
    # series; feeding it the price LEVEL measures the self-affinity of a
    # non-stationary path and returns H > 1, which is not a number Hurst can be.
    # On returns the calibration is the textbook one: 0.5 = random walk.
    h = hurst_rs(rets, cfg.hurst_min_n)
    comp["hurst"] = h
    slope, r2 = linreg_trend(cl, min(len(cl), 60))
    comp["r2"] = r2
    comp["slope"] = slope
    rho = autocorr1(rets[-cfg.rho_window:] if len(rets) > cfg.rho_window else rets)
    comp["rho1"] = rho
    consist = trend_consistency(cl, cfg.ma_fast, cfg.ma_slow)
    comp["consist"] = consist

    # Each view normalized so 0.5 == "random walk, no opinion".
    s_adx = _ramp(a, 20.0, 40.0)                                  # 25 trend, 20 range
    # average of the per-horizon normalized readings, not the ramp of the mean
    _er_norms = [_ramp(x, 0.20, 0.60) for x in ers]               # >0.3-0.4 trend
    s_er = (_mean(_er_norms) if _er_norms else NAN)
    s_chop = (1.0 - _ramp(ch, 38.2, 61.8)) if math.isfinite(ch) else NAN
    s_vr = (0.5 + _clamp(vr_mean - 1.0, -0.5, 0.5)) if math.isfinite(vr_mean) else NAN
    s_hurst = (0.5 + _clamp((h - 0.5) * 2.0, -0.5, 0.5)) if math.isfinite(h) else NAN
    s_r2 = _ramp(r2, 0.20, 0.70)
    s_rho = (0.5 + _clamp(rho * 5.0, -0.5, 0.5)) if math.isfinite(rho) else NAN
    # strength of the MA-ribbon agreement: 0.5 -> no opinion, 0 or 1 -> strong
    s_consist = (_clamp(abs(consist - 0.5) * 2.0, 0.0, 1.0)
                 if math.isfinite(consist) else NAN)

    pairs = [(s_adx, cfg.w_adx), (s_er, cfg.w_er), (s_chop, cfg.w_chop),
             (s_vr, cfg.w_vr), (s_hurst, cfg.w_hurst), (s_r2, cfg.w_r2),
             (s_rho, cfg.w_rho), (s_consist, cfg.w_consist)]
    tot_w = sum(w for s, w in pairs if math.isfinite(s))
    if tot_w <= 0:
        return 0.5, comp
    # Renormalize over whichever components are finite: a missing statistic
    # must not silently drag the blend toward "neutral".
    score = sum(s * w for s, w in pairs if math.isfinite(s)) / tot_w
    return _clamp(score, 0.0, 1.0), comp


def _volatility(win_bars, cfg):
    """-> (vol_score, yz_vol). Stress = the STRONGER of two independent reads.

    MEASURE A REGIME, NOT A BAR. Volatility clusters, so a *regime* is a
    PERSISTENT elevation, not one big print. Ranking the raw latest estimate
    against its own history made this axis fire on every vol-expansion bar —
    and entry bars ARE vol-expansion bars, so the sniper could never fire:

        max S in high_vol = .25*1 + .35*1 + .25*0.25 + .15*1 = 0.8125 < 0.85

    So the percentile is taken on the SMOOTHED estimate (mean of the last
    `vol_smooth` readings). A single spike moves it by 1/vol_smooth; a genuine
    regime shift moves all of them. That is the difference between "this bar
    was big" and "this market is stressed".

    Two further guards, both found by measurement:
      * YZ is a VARIANCE of log-price changes, so on constant returns it
        collapses to 0 and we would report NaN — yet a market can have a 12%
        intraday range on an unchanged close. `ATR/close` (a LEVEL, which
        cannot collapse) is combined in with max, mirroring the M layer.
      * Ranking against a FLAT history makes the percentile degenerate: every
        sample ties, `<=` holds for all of them, and it pins to 1.0. A flat
        history carries no information, so it reads 0.5.
    """
    n = len(win_bars)
    if n < cfg.yz_n + 3:
        return NAN, NAN, NAN
    # the whole YZ series for this window, each point causal
    ys = [NAN] * n
    for end in range(cfg.yz_n + 2, n + 1):
        ys[end - 1] = yang_zhang_vol(win_bars[:end], cfg.yz_n)
    sm = [NAN] * n
    need = max(2, cfg.vol_smooth // 2)
    for i in range(n):
        w = [x for x in ys[max(0, i - cfg.vol_smooth + 1):i + 1]
             if math.isfinite(x)]
        if len(w) >= need:
            sm[i] = sum(w) / len(w)

    yz_now = ys[-1]
    smooth_now = sm[-1]

    cl = closes(win_bars)
    a = atr(win_bars, cfg.atr_period)
    atr_pct_now = (a[-1] / cl[-1]) if (math.isfinite(a[-1]) and cl[-1] > 0) else NAN
    # 0 at half the stress threshold, 1 at the threshold (the mandate's atr_hi)
    atr_stress = _ramp(atr_pct_now, cfg.atr_hi * 0.5, cfg.atr_hi)

    vol_pct = NAN
    if math.isfinite(smooth_now):
        hist = [sm[j] for j in range(max(0, n - 1 - cfg.vol_hist), n - 1)
                if math.isfinite(sm[j])]
        if len(hist) >= 5:
            spread = max(hist) - min(hist)
            scale = max(1.0, abs(max(hist)))
            vol_pct = (0.5 if spread <= 1e-12 * scale
                       else percentile_rank(hist, smooth_now))

    cands = [x for x in (vol_pct, atr_stress) if math.isfinite(x)]
    score = max(cands) if cands else NAN
    # `atr_stress > 0` is the ABSOLUTE confirmation: the mandate's own atr_hi
    # defines "high volatility", and a purely relative percentile is not enough.
    # Ranking alone marks ~15% of ALL bars high_vol by construction (it is a
    # 90th-percentile threshold), including entry bars — which are vol-expansion
    # bars by nature. Since max S in high_vol is 0.8125 < 0.85, that made the
    # sniper bar unreachable in practice. A market must be stressed in absolute
    # terms too, not merely "the most volatile it has been lately".
    return score, yz_now, atr_stress


def detect(bars, i, cfg=None, window=None):
    """Regime at bar `i`, using ONLY bars[max(0, i-window+1) .. i].

    Pure function: no state, no look-ahead. Returns a RegimeState.
    """
    cfg = cfg or RegimeConfig()
    w = window or cfg.window
    lo = max(0, i - w + 1)
    win = bars[lo:i + 1]
    if len(win) < 20:
        return RegimeState(ts=_ts(bars[i]), trend_score=0.5, vol_score=NAN,
                           label="range", direction="", components={})

    trend, comp = _directional(win, cfg)
    vol_score, yz, atr_stress = _volatility(win, cfg)
    cl = closes(win)
    hl = ou_half_life(cl, cfg.ou_n)
    sma_long = sma(cl, cfg.sma_long)
    ref = sma_long[-1] if (len(cl) >= cfg.sma_long
                           and math.isfinite(sma_long[-1])) else NAN

    slope = comp.get("slope", NAN)
    direction = ""
    if math.isfinite(slope):
        if slope > 0:
            direction = "up"
        elif slope < 0:
            direction = "down"

    # --- label ------------------------------------------------------------
    # high_vol needs BOTH reads: relatively stressed (percentile) AND
    # absolutely stressed (the mandate's atr_hi). See _volatility().
    stressed = (math.isfinite(vol_score) and vol_score >= cfg.vol_stress
                and math.isfinite(atr_stress) and atr_stress > 0.0)
    if stressed:
        label = "high_vol"
    elif trend >= cfg.trend_hi and direction in ("up", "down"):
        # a trend claim also has to agree with the long-horizon reference
        if math.isfinite(ref):
            if direction == "up" and cl[-1] > ref:
                label = "trend_up"
            elif direction == "down" and cl[-1] < ref:
                label = "trend_down"
            else:
                label = "range"
        else:
            label = "trend_up" if direction == "up" else "trend_down"
    else:
        label = "range"

    return RegimeState(
        ts=_ts(bars[i]), trend_score=trend,         vol_score=vol_score, label=label,
        direction=direction, adx=comp.get("adx"), er=comp.get("er"),
        chop=comp.get("chop"), vr=comp.get("vr"), vr_z=comp.get("vr_z"),
        hurst=comp.get("hurst"), r2=comp.get("r2"), rho1=comp.get("rho1"),
        yz_vol=yz, half_life=hl, components=comp, stressed=stressed)


def _ts(bar):
    return int(bar[0]) if not isinstance(bar, dict) else int(bar["time"])


# ---------------------------------------------------------------------------
# Stateful tracker with hysteresis
# ---------------------------------------------------------------------------
class RegimeTracker:
    """Walks a bar series forward, applying hysteresis to the raw label.

    Raw per-bar labels flap: a two-bar pullback flips `trend_up` to `range` and
    back, and every flip costs real spread. Hysteresis (enter high, exit low) is
    the standard fix. This class only ever reads bar i and its OWN past state,
    so it is still causal — and the tests prove it by replaying a prefix.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or RegimeConfig()
        self._trend = False
        self._vol = False
        self._dir = ""

    def step(self, bars, i):
        st = detect(bars, i, self.cfg)
        cfg = self.cfg
        # volatility latch: the most reliable axis, so it latches hardest.
        # Entry requires BOTH the relative and the absolute stress reads.
        if st.stressed and _f(st.vol_score) >= cfg.vol_enter:
            self._vol = True
        elif not st.stressed or _f(st.vol_score) < cfg.vol_exit:
            self._vol = False
        # directional latch
        if st.trend_score >= cfg.trend_enter and st.direction in ("up", "down"):
            self._trend = True
            self._dir = st.direction
        elif st.trend_score < cfg.trend_exit or st.direction not in ("up", "down"):
            self._trend = False
            self._dir = st.direction

        if self._vol:
            st.label = "high_vol"
        elif self._trend:
            st.label = "trend_up" if self._dir == "up" else "trend_down"
        else:
            st.label = "range"
        return st

    def run(self, bars):
        return [self.step(bars, i) for i in range(len(bars))]


def scan(bars, cfg=None, tracked=True):
    """Regime for every bar. `tracked` applies hysteresis (recommended)."""
    cfg = cfg or RegimeConfig()
    if tracked:
        return RegimeTracker(cfg).run(bars)
    return [detect(bars, i, cfg) for i in range(len(bars))]
