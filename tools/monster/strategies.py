# =============================================================================
#  OmniSeed — tools/monster/strategies.py
#  A zoo of INDEPENDENT strategies, each returning a signed conviction in
#  [-1,1] plus its own confidence in [0,1].
#
#  Why a zoo and not one clever model: momentum and mean reversion need
#  OPPOSITE conditions, so a single fixed strategy is always wrong in half of
#  all regimes. The router (router.py) decides which to listen to.
#
#    momentum        Donchian breakout + MA ribbon          (trend-following)
#    mean_reversion  z-score fade, gated on OU half-life    (range)
#    breakout        Bollinger/Keltner squeeze -> expansion (transition)
#    ofi             order flow imbalance (Cont-Kukanov-Stoikov 2014)
#    vol_target      inverse-variance exposure scaling (Moreira-Muir 2017)
#                    -> a SIZING overlay, not a direction
#
#  THE OFI PATH IS THE INTERESTING ONE. CKS show price changes over short
#  intervals are driven by order flow imbalance, with a LINEAR relation
#      dP_k = beta * OFI_k + eps_k,        beta = c / depth^lambda
#  and an R^2 of 35-79% across 50 US stocks (most names 65-70%) at 10-second
#  intervals. Two honest caveats, stated up front because they matter:
#    1. That R^2 is CONTEMPORANEOUS, not predictive. OFI is not a crystal ball.
#       The tradable part is that OFI is autocorrelated and price impact is only
#       partly permanent, so the next interval's flow is partly knowable.
#    2. It needs DEPTH. Free OHLCV feeds do not publish the book. Without it we
#       fall back to a signed-volume proxy, which is a *different and weaker*
#       object, and we say so in the signal reason rather than quietly pretending.
#
#  NO LOOK-AHEAD: every strategy reads indices <= i. Donchian channels are
#  built from bars strictly BEFORE i (a breakout is "above the prior range",
#  never "above a range that includes itself").
# =============================================================================
"""Independent strategies: momentum, mean reversion, breakout, OFI, vol-target."""

import math

from .features import NAN, closes, highs, lows, volumes, atr, ema, sma

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


def _ramp(x, lo, hi):
    if x is None or not math.isfinite(x):
        return NAN
    if hi == lo:
        return 0.5
    return _clamp((x - lo) / (hi - lo), 0.0, 1.0)


def _safe(x, d=0.0):
    return x if (x is not None and math.isfinite(x)) else d


def rolling_std(vals, period):
    n = len(vals)
    out = [NAN] * n
    for i in range(period - 1, n):
        w = vals[i - period + 1:i + 1]
        if any(not math.isfinite(v) for v in w):
            continue
        m = sum(w) / period
        out[i] = math.sqrt(sum((x - m) ** 2 for x in w) / period)
    return out


def donchian(bars, n):
    """-> (upper, lower) where upper[i] = max high over the n bars BEFORE i.

    Strictly prior bars, so "close > upper" is a genuine breakout and not a
    self-fulfilling comparison against a window containing the current bar.
    """
    h, l = highs(bars), lows(bars)
    m = len(bars)
    up, lo = [NAN] * m, [NAN] * m
    for i in range(n, m):
        up[i] = max(h[i - n:i])
        lo[i] = min(l[i - n:i])
    return up, lo


def keltner(bars, period=20, mult=1.5):
    """Keltner channels: EMA(close) +/- mult*ATR. The squeeze benchmark."""
    c = closes(bars)
    mid = ema(c, period)
    a = atr(bars, period)
    up = [mid[i] + mult * a[i] if (math.isfinite(mid[i]) and math.isfinite(a[i]))
          else NAN for i in range(len(bars))]
    lo = [mid[i] - mult * a[i] if (math.isfinite(mid[i]) and math.isfinite(a[i]))
          else NAN for i in range(len(bars))]
    return mid, up, lo


# ---------------------------------------------------------------------------
# ORDER FLOW IMBALANCE (Cont, Kukanov & Stoikov 2014)
# ---------------------------------------------------------------------------
def ofi_from_snapshots(snaps):
    """Exact CKS order flow imbalance from a sequence of best-quote snapshots.

    snaps: [(bid_px, bid_sz, ask_px, ask_sz), ...]

    Canonical form (equivalent to the paper's OFI = Lb - Cb - Ms - Ls + Cs + Mb):

      OFI = SUM over events of
            1{Pb_n >= Pb_n-1}*qb_n - 1{Pb_n <= Pb_n-1}*qb_n-1
          - 1{Pa_n <= Pa_n-1}*qa_n + 1{Pa_n >= Pa_n-1}*qa_n-1

    i.e. bid-side events enter with +, ask-side with -.
    """
    total = 0.0
    for k in range(1, len(snaps)):
        pb0, qb0, pa0, qa0 = snaps[k - 1]
        pb1, qb1, pa1, qa1 = snaps[k]
        e_b = (qb1 if pb1 >= pb0 else 0.0) - (qb0 if pb1 <= pb0 else 0.0)
        e_a = (qa1 if pa1 <= pa0 else 0.0) - (qa0 if pa1 >= pa0 else 0.0)
        total += e_b - e_a
    return total


def impact_beta(depth, c=1.0, lam=1.0):
    """CKS price impact coefficient: beta = c / depth^lambda.

    The stylized model corresponds to lambda = 1 (beta = 1/(2D)).
    """
    if depth is None or not math.isfinite(depth) or depth <= 0:
        return NAN
    return c / (depth ** lam)


def ofi_proxy(bars, n=20):
    """Signed-volume proxy for OFI, for feeds with NO depth.

    Uses the Lee-Ready tick rule (sign of the bar return) as the flow sign:

        sv_i  = sign(close_i - close_{i-1}) * volume_i
        OFI_i = sum(sv, n) / sum(volume, n)          in [-1, 1]

    This is a DIFFERENT, WEAKER object than the CKS OFI: it cannot see limit
    orders or cancellations, which is most of what OFI actually measures. It is
    used only when no book is available, and callers label it as a proxy.
    """
    c, v = closes(bars), volumes(bars)
    m = len(bars)
    sv = [0.0] * m
    for i in range(1, m):
        if c[i] > c[i - 1]:
            sv[i] = v[i]
        elif c[i] < c[i - 1]:
            sv[i] = -v[i]
    out = [NAN] * m
    for i in range(n, m):
        num = sum(sv[i - n + 1:i + 1])
        den = sum(v[i - n + 1:i + 1])
        out[i] = num / den if den > 0 else 0.0
    return out


def zscore_series(vals, period=60):
    """Trailing z-score, strictly against the PAST window."""
    n = len(vals)
    out = [NAN] * n
    for i in range(period, n):
        w = [x for x in vals[i - period:i] if math.isfinite(x)]
        if len(w) < max(5, period // 3):
            continue
        mu = sum(w) / len(w)
        var = sum((x - mu) ** 2 for x in w) / len(w)
        sd = math.sqrt(var)
        out[i] = (vals[i] - mu) / sd if sd > 0 else 0.0
    return out


# ---------------------------------------------------------------------------
# Signal / series containers
# ---------------------------------------------------------------------------
class StrategySignal:
    __slots__ = ("name", "direction", "confidence", "regime_fit", "reason")

    def __init__(self, name, direction, confidence, regime_fit="both", reason=""):
        self.name = name
        self.direction = _clamp(_safe(direction), -1.0, 1.0)
        self.confidence = _clamp(_safe(confidence), 0.0, 1.0)
        self.regime_fit = regime_fit
        self.reason = reason

    @property
    def active(self):
        return self.confidence > 0.0 and self.direction != 0.0

    def __repr__(self):
        return ("%s(dir=%+.2f conf=%.2f fit=%s)"
                % (self.name, self.direction, self.confidence, self.regime_fit))


class StrategyConfig:
    donchian_n: int = 20
    sma_fast: int = 20
    sma_slow: int = 50
    z_window: int = 20
    z_entry: float = 2.0
    z_max: float = 3.5
    bb_period: int = 20
    bb_sigma: float = 2.0
    kc_period: int = 20
    kc_mult: float = 1.5
    squeeze_lookback: int = 6
    ofi_window: int = 20
    ofi_z_window: int = 60
    ofi_z_full: float = 3.0
    vol_target_power: float = 2.0
    vol_target_ref: int = 60
    vol_target_max_lev: float = 2.0
    vol_target_min_scale: float = 0.25

    def __init__(self, **kw):
        for k, v in kw.items():
            if not hasattr(self, k):
                raise TypeError("unknown StrategyConfig field: %s" % k)
            setattr(self, k, v)


class StrategySeries:
    """All precomputed series for one instrument. Index i is bar i."""

    def __init__(self, bars, cfg=None):
        self.cfg = cfg or StrategyConfig()
        self.bars = bars
        self.n = len(bars)
        self.ts = [int(b[0]) if not isinstance(b, dict) else int(b["time"])
                   for b in bars]
        self.close = closes(bars)
        self.vol = volumes(bars)
        self.sma_fast = sma(self.close, self.cfg.sma_fast)
        self.sma_slow = sma(self.close, self.cfg.sma_slow)
        self.sma_z = sma(self.close, self.cfg.z_window)
        self.std_z = rolling_std(self.close, self.cfg.z_window)
        self.don_up, self.don_lo = donchian(bars, self.cfg.donchian_n)
        self.bb_mid = sma(self.close, self.cfg.bb_period)
        self.bb_std = rolling_std(self.close, self.cfg.bb_period)
        self.bb_up = [self.bb_mid[i] + self.cfg.bb_sigma * self.bb_std[i]
                      if (math.isfinite(self.bb_mid[i])
                          and math.isfinite(self.bb_std[i])) else NAN
                      for i in range(self.n)]
        self.bb_lo = [self.bb_mid[i] - self.cfg.bb_sigma * self.bb_std[i]
                      if (math.isfinite(self.bb_mid[i])
                          and math.isfinite(self.bb_std[i])) else NAN
                      for i in range(self.n)]
        _, self.kc_up, self.kc_lo = keltner(bars, self.cfg.kc_period,
                                            self.cfg.kc_mult)
        self.ofi = ofi_proxy(bars, self.cfg.ofi_window)
        self.ofi_z = zscore_series(self.ofi, self.cfg.ofi_z_window)
        self.atr = atr(bars, 14)


def prepare(bars, cfg=None):
    return StrategySeries(bars, cfg)


# ---------------------------------------------------------------------------
# The strategies
# ---------------------------------------------------------------------------
def momentum(s, i, regime=None, cfg=None):
    """Donchian breakout confirmed by the MA ribbon.

    A breakout of the prior n-bar range is the oldest trend-following rule there
    is (Turtle traders, Donchian 1960s). The ribbon filter is what stops it from
    firing on every other bar in a chop.
    """
    cfg = cfg or s.cfg
    if i < 1 or not math.isfinite(s.don_up[i]) or not math.isfinite(s.don_lo[i]):
        return StrategySignal("momentum", 0.0, 0.0, "trend", "no-channel")
    c = s.close[i]
    up, lo = s.don_up[i], s.don_lo[i]
    rng = up - lo
    if rng <= 0:
        return StrategySignal("momentum", 0.0, 0.0, "trend", "flat-channel")

    ribbon = 0
    if math.isfinite(s.sma_fast[i]) and math.isfinite(s.sma_slow[i]):
        ribbon = 1 if s.sma_fast[i] > s.sma_slow[i] else -1

    if c > up:
        d = 1.0
        margin = (c - up) / rng
    elif c < lo:
        d = -1.0
        margin = (lo - c) / rng
    else:
        return StrategySignal("momentum", 0.0, 0.0, "trend", "inside-range")

    # agreeing ribbon raises confidence; opposing ribbon halves it
    agree = 1.0 if ribbon == d else (0.5 if ribbon == 0 else 0.25)
    conf = _clamp(_ramp(margin, 0.0, 0.25), 0.0, 1.0) * agree
    if regime is not None and getattr(regime, "trend_score", None) is not None:
        ts = _safe(regime.trend_score, 0.5)
        conf *= _clamp(0.5 + ts, 0.5, 1.5) / 1.5
    return StrategySignal("momentum", d, conf, "trend",
                          "donchian-breakout(margin=%.3f ribbon=%+d)" % (margin, ribbon))


def mean_reversion(s, i, regime=None, cfg=None):
    """Fade an extreme z-score, but ONLY where reversion is real and fast.

    Gated on the regime engine's OU half-life, which is the difference between
    "this is a mean-reverting market" and "this has fallen a long way and I hope".
    """
    cfg = cfg or s.cfg
    if i < 1 or not math.isfinite(s.sma_z[i]) or not math.isfinite(s.std_z[i]):
        return StrategySignal("mean_reversion", 0.0, 0.0, "range", "no-window")
    sd = s.std_z[i]
    if sd <= 0:
        return StrategySignal("mean_reversion", 0.0, 0.0, "range", "zero-sd")
    z = (s.close[i] - s.sma_z[i]) / sd

    tradeable = True
    hl = NAN
    if regime is not None:
        tradeable = bool(getattr(regime, "mean_reversion_tradeable", True))
        hl = getattr(regime, "half_life", NAN)
    if abs(z) < cfg.z_entry:
        return StrategySignal("mean_reversion", 0.0, 0.0, "range",
                              "z=%.2f below-entry" % z)
    if not tradeable:
        return StrategySignal("mean_reversion", 0.0, 0.0, "range",
                              "z=%.2f not-reverting" % z)

    d = -1.0 if z > 0 else 1.0
    conf = _clamp(abs(z) / cfg.z_max, 0.0, 1.0)
    # a fast half-life is better evidence than a slow one
    if math.isfinite(hl) and hl > 0:
        conf *= _clamp(1.25 - hl / 40.0, 0.4, 1.0)
    return StrategySignal("mean_reversion", d, conf, "range",
                          "fade(z=%.2f hl=%.0f)" % (z, _safe(hl, -1)))


def breakout(s, i, regime=None, cfg=None):
    """Volatility squeeze (Bollinger inside Keltner) then expansion.

    The squeeze says the market is coiling; the expansion says it has chosen a
    direction. Direction is taken from the band the close exits, and confidence
    is much higher when a squeeze actually preceded it — a band break out of
    nowhere is just noise.
    """
    cfg = cfg or s.cfg
    if i < 1:
        return StrategySignal("breakout", 0.0, 0.0, "both", "no-history")
    if not (math.isfinite(s.bb_up[i]) and math.isfinite(s.kc_up[i])):
        return StrategySignal("breakout", 0.0, 0.0, "both", "no-bands")

    c = s.close[i]
    squeeze = False
    for k in range(1, cfg.squeeze_lookback + 1):
        j = i - k
        if j < 0:
            break
        if (math.isfinite(s.bb_up[j]) and math.isfinite(s.kc_up[j])
                and s.bb_up[j] < s.kc_up[j] and s.bb_lo[j] > s.kc_lo[j]):
            squeeze = True
            break

    if c > s.bb_up[i]:
        d, edge = 1.0, (c - s.bb_up[i])
    elif c < s.bb_lo[i]:
        d, edge = -1.0, (s.bb_lo[i] - c)
    else:
        return StrategySignal("breakout", 0.0, 0.0, "both", "inside-bands")

    width = max(s.bb_up[i] - s.bb_lo[i], 1e-12)
    conf = _clamp(_ramp(edge / width, 0.0, 0.30), 0.0, 1.0)
    if squeeze:
        conf = _clamp(conf * 1.6, 0.0, 1.0)
    else:
        conf *= 0.4
    return StrategySignal("breakout", d, conf, "both",
                          "band-break(squeeze=%d)" % (1 if squeeze else 0))


def ofi(s, i, regime=None, cfg=None, snapshots=None):
    """Order flow imbalance -> expected price move via the CKS linear model.

    If `snapshots` (a quote book) is supplied we use the EXACT CKS OFI and
    beta = c/depth^lambda. Otherwise we fall back to the signed-volume proxy and
    say so in the reason string. Either way the direction is the sign of the
    expected move, never a claim about magnitude.
    """
    cfg = cfg or s.cfg
    if snapshots:
        raw = ofi_from_snapshots(snapshots)
        # depth = mean best-level size across the snapshots
        depths = [(sn[1] + sn[3]) / 2.0 for sn in snapshots]
        depth = sum(depths) / len(depths) if depths else NAN
        beta = impact_beta(depth, lam=1.0)
        if not math.isfinite(beta) or beta <= 0:
            return StrategySignal("ofi", 0.0, 0.0, "both", "no-depth")
        exp_move = beta * raw
        d = 1.0 if exp_move > 0 else (-1.0 if exp_move < 0 else 0.0)
        conf = _clamp(_ramp(abs(raw) / max(depth, 1e-12), 0.0, 1.0), 0.0, 1.0)
        return StrategySignal("ofi", d, conf, "both",
                              "cks-ofi(raw=%.1f beta=%.2e)" % (raw, beta))

    if i < 1 or not math.isfinite(s.ofi[i]) or not math.isfinite(s.ofi_z[i]):
        return StrategySignal("ofi", 0.0, 0.0, "both", "no-flow")
    z = s.ofi_z[i]
    if abs(z) < 1.0:
        return StrategySignal("ofi", 0.0, 0.0, "both", "flow-flat(z=%.2f)" % z)
    d = 1.0 if z > 0 else -1.0
    conf = _clamp(abs(z) / cfg.ofi_z_full, 0.0, 1.0) * 0.7  # proxy discount
    return StrategySignal("ofi", d, conf, "both",
                          "ofi-proxy(z=%.2f NOT-cks)" % z)


ALL = ("momentum", "mean_reversion", "breakout", "ofi")


def all_signals(s, i, regime=None, cfg=None, snapshots=None):
    """Every strategy's view of bar `i` (including the inactive ones)."""
    return [
        momentum(s, i, regime, cfg),
        mean_reversion(s, i, regime, cfg),
        breakout(s, i, regime, cfg),
        ofi(s, i, regime, cfg, snapshots),
    ]


# ---------------------------------------------------------------------------
# VOLATILITY HARVESTING (Moreira & Muir 2017)
# ---------------------------------------------------------------------------
def realized_variance(vals, n):
    """Trailing realized variance of returns over `n` bars."""
    r = []
    for i in range(1, len(vals)):
        if vals[i] > 0 and vals[i - 1] > 0:
            r.append(math.log(vals[i] / vals[i - 1]))
    if len(r) < n:
        return NAN
    w = r[-n:]
    m = sum(w) / len(w)
    return sum((x - m) ** 2 for x in w) / len(w)


def vol_target_scale(s, i, cfg=None):
    """Moreira-Muir inverse-variance exposure scaler -> a multiplier.

        f_sigma = (c / sigma_hat^2) * f        with c set so mean exposure ~ 1

    The published result: scaling a factor by the inverse of its realized
    variance earns a large alpha (4.86% for the market factor) that survives
    trading costs up to 56bps. It works because volatility is persistent and
    does NOT proportionally predict returns, so lower vol buys a better
    risk-return trade-off. Note the cost: turnover rises (mean |dw| = 0.73).

    We express it as scale = (rv_ref / rv)^power, clipped. rv == rv_ref -> 1.0.
    `power=2` is the paper's variance version; `power=1` is inverse-vol.
    """
    cfg = cfg or s.cfg
    if i < 1:
        return 1.0, NAN, NAN
    rv_now = realized_variance(s.close[:i + 1], cfg.vol_target_ref)
    rv_ref = realized_variance(s.close[:i + 1], max(cfg.vol_target_ref * 4,
                                                    cfg.vol_target_ref + 1))
    if not math.isfinite(rv_now) or not math.isfinite(rv_ref) or rv_now <= 0:
        return 1.0, rv_now, rv_ref
    scale = (rv_ref / rv_now) ** cfg.vol_target_power
    scale = _clamp(scale, cfg.vol_target_min_scale, cfg.vol_target_max_lev)
    return scale, rv_now, rv_ref
