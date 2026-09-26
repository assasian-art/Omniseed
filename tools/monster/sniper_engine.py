# =============================================================================
#  OmniSeed — tools/monster/sniper_engine.py
#  The Sniper Entry Engine: a weighted, multi-layer confidence score.
#
#      S = w_m*M + w_t*T + w_r*R + w_c*C          (all layers in [0,1])
#
#    M  microstructure  — volume-anomaly z-score (+ order-book imbalance when
#                         the provider publishes depth)
#    T  technical       — up to 6 INDEPENDENT bullish factors; >=3 required
#    R  regime          — higher-timeframe regime; counter-trend is vetoed
#                         unless a verified mean-reversion setup is present
#    C  cross-asset     — lead-lag confirmation / invalidation
#
#  An entry is PROPOSED only when no veto fires AND S >= min_confidence.
#
#  Honest scope: the score measures how much of the evidence agrees. It is not
#  a probability of profit. See docs/MONSTER_DESIGN.md.
# =============================================================================
"""Sniper entry engine — weighted multi-layer confirmation."""

import math
from dataclasses import dataclass, field

from . import features as F


@dataclass
class SniperConfig:
    # --- layer weights (must sum to ~1.0) ---------------------------------
    w_micro: float = 0.25
    w_tech: float = 0.35
    w_regime: float = 0.25
    w_cross: float = 0.15
    # --- decision bar -----------------------------------------------------
    min_confidence: float = 0.85
    min_factors: int = 3            # independent technical votes required
    min_factors_event: int = 1      # relaxed when an event is matched
    # --- microstructure ---------------------------------------------------
    vol_window: int = 20
    # --- technical --------------------------------------------------------
    sma_fast: int = 20
    sma_slow: int = 50
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    bb_period: int = 20
    bb_sigma: float = 2.0
    vwap_period: int = 20
    swing_left: int = 2
    swing_right: int = 2
    fib_ratios: tuple = (0.382, 0.5, 0.618)
    fib_tol: float = 0.015          # +/- 1.5% of price around a level
    div_max_gap: int = 60
    # --- regime -----------------------------------------------------------
    sma_regime: int = 200
    regime_slope_n: int = 20
    regime_slope_s: float = 0.01    # 1% SMA200 move over n bars
    atr_period: int = 14
    atr_hi: float = 0.05            # 5% ATR/close = high-volatility regime
    # --- cross-asset ------------------------------------------------------
    invalidate_veto: int = 2


@dataclass
class EvalContext:
    """Everything the bar series cannot know: depth, peers, news."""
    obi: float = None               # order-book imbalance in [-1,1]
    cross_confirm: int = 0
    cross_invalidate: int = 0
    cross_active: int = 0
    event_driven: bool = False


@dataclass
class SniperVerdict:
    ts: int = 0
    score: float = 0.0
    micro: float = 0.0
    tech: float = 0.0
    regime_score: float = 0.0
    cross: float = 0.0
    votes: int = 0
    regime: str = "range"
    veto: bool = False
    veto_reason: str = ""
    factors: list = field(default_factory=list)

    def propose(self, cfg):
        return (not self.veto) and self.score >= cfg.min_confidence

    def detail(self):
        """Compact, CSV-safe rationale (no commas)."""
        parts = [
            "S=%.3f" % self.score,
            "M=%.2f" % self.micro,
            "T=%.2f(%d)" % (self.tech, self.votes),
            "R=%.2f" % self.regime_score,
            "C=%.2f" % self.cross,
            "regime=%s" % self.regime,
        ]
        if self.veto:
            parts.append("VETO=%s" % (self.veto_reason or "gate"))
        if self.factors:
            parts.append("factors=%s" % "+".join(self.factors))
        return " ".join(parts)


class Prepared:
    """All feature series for one bar history, computed once."""

    def __init__(self, bars, cfg):
        self.cfg = cfg
        self.bars = bars
        self.n = len(bars)
        self.ts = [int(b[0]) if not isinstance(b, dict) else int(b["time"])
                   for b in bars]
        self.close = F.closes(bars)
        self.high = F.highs(bars)
        self.low = F.lows(bars)
        self.vol = F.volumes(bars)
        self.sma_fast = F.sma(self.close, cfg.sma_fast)
        self.sma_slow = F.sma(self.close, cfg.sma_slow)
        self.sma_regime = F.sma(self.close, cfg.sma_regime)
        self.rsi = F.rsi(self.close, cfg.rsi_period)
        _, _, self.macd_hist = F.macd(self.close)
        self.bb_mid, self.bb_up, self.bb_lo = F.bollinger(
            self.close, cfg.bb_period, cfg.bb_sigma)
        self.atr = F.atr(bars, cfg.atr_period)
        self.atr_pct = F.atr_pct(bars, cfg.atr_period)
        self.vwap = F.rolling_vwap(bars, cfg.vwap_period)
        self.vol_z = F.rolling_zscore(self.vol, cfg.vol_window)
        self.swings_low = F.find_swings(self.low, cfg.swing_left,
                                        cfg.swing_right, "low")
        self.swings_high = F.find_swings(self.high, cfg.swing_left,
                                         cfg.swing_right, "high")


def prepare(bars, cfg=None):
    return Prepared(bars, cfg or SniperConfig())


# ---------------------------------------------------------------------------
# Layer computations
# ---------------------------------------------------------------------------
def _micro(p, i, ctx, cfg):
    z = p.vol_z[i]
    z = z if math.isfinite(z) else 0.0
    m_vol = 1.0 / (1.0 + math.exp(-z / 2.0))
    if ctx is not None and ctx.obi is not None:
        obi = max(-1.0, min(1.0, ctx.obi))
        m_obi = 0.5 + 0.5 * obi
        # Either a strongly one-sided book OR an anomalous volume print is
        # evidence of aggressive participation, so take the STRONGER of the
        # two. Blending would let a quiet book dilute a real volume spike
        # (and vice versa), which is exactly backwards.
        return max(m_obi, m_vol)
    return m_vol


def _technical(p, i, cfg):
    """-> (score, votes, factor names). Six independent bullish factors."""
    names = []
    c = p.close[i]
    if i < 1:
        return 0.0, 0, names

    # 1. VWAP reclaim: closes back above a VWAP it was below.
    if (math.isfinite(p.vwap[i]) and math.isfinite(p.vwap[i - 1])
            and c > p.vwap[i] and p.close[i - 1] <= p.vwap[i - 1]):
        names.append("vwap-reclaim")

    # 2. RSI bullish divergence (price lower-low, RSI higher-low).
    if F.bullish_divergence(p.rsi, p.swings_low, i, cfg.swing_right,
                            cfg.div_max_gap):
        names.append("rsi-divergence")

    # 3. Fibonacci support: close within tol of a retracement level.
    pair = F.last_swing_pair(p.swings_low, p.swings_high, i, cfg.swing_right)
    if pair:
        for lvl in F.fib_levels(pair[1], pair[3], cfg.fib_ratios):
            if lvl > 0 and abs(c - lvl) / lvl <= cfg.fib_tol:
                names.append("fib-support")
                break

    # 4. MACD thrust: histogram positive and rising.
    if (i >= 1 and math.isfinite(p.macd_hist[i])
            and math.isfinite(p.macd_hist[i - 1])
            and p.macd_hist[i] > 0 and p.macd_hist[i] > p.macd_hist[i - 1]):
        names.append("macd-thrust")

    # 5. Mean reversion: close below the lower Bollinger band.
    if math.isfinite(p.bb_lo[i]) and c < p.bb_lo[i]:
        names.append("bb-oversold")

    # 6. Trend alignment: fast SMA above slow SMA.
    if (math.isfinite(p.sma_fast[i]) and math.isfinite(p.sma_slow[i])
            and p.sma_fast[i] > p.sma_slow[i]):
        names.append("trend-align")

    votes = len(names)
    # Meeting the mandated minimum confluence earns 0.90 of the layer; each
    # EXTRA agreeing factor adds 0.10 up to a cap of 1.00. Scoring votes/6 would
    # cap the reachable total at ~0.81 and make the 0.85 bar mathematically
    # unreachable — a gate that can never open is a bug, not discipline.
    if votes >= cfg.min_factors:
        t = min(1.0, 0.90 + 0.10 * (votes - cfg.min_factors))
    else:
        t = 0.5 * votes / max(1, cfg.min_factors)
    return t, votes, names


def _regime(p, i, cfg):
    """-> (name, score). Names: trend_up / trend_down / high_vol / range."""
    if i < cfg.regime_slope_n or not math.isfinite(p.sma_regime[i]):
        return "range", 0.5
    prev = p.sma_regime[i - cfg.regime_slope_n]
    if not math.isfinite(prev) or prev <= 0:
        return "range", 0.5
    slope = (p.sma_regime[i] - prev) / prev
    c = p.close[i]
    s = cfg.regime_slope_s
    if slope > s and c > p.sma_regime[i]:
        return "trend_up", 1.0
    if slope < -s and c < p.sma_regime[i]:
        return "trend_down", 0.0
    ap = p.atr_pct[i]
    if math.isfinite(ap) and ap > cfg.atr_hi:
        return "high_vol", 0.25
    return "range", 0.5


def _cross(ctx, cfg):
    if ctx is None or ctx.cross_active <= 0:
        return 0.5
    raw = 0.5 + 0.5 * (ctx.cross_confirm - ctx.cross_invalidate) / max(
        1, ctx.cross_active)
    return max(0.0, min(1.0, raw))


def _mean_reversion_confirmed(p, i, cfg):
    """Oversold AND below the lower band — two independent confirmations."""
    c = p.close[i]
    below_band = math.isfinite(p.bb_lo[i]) and c < p.bb_lo[i]
    oversold = math.isfinite(p.rsi[i]) and p.rsi[i] < cfg.rsi_oversold
    return below_band and oversold


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------
def evaluate(p, i, ctx=None, cfg=None):
    """Score bar `i` using only data at indices <= i."""
    cfg = cfg or p.cfg
    v = SniperVerdict(ts=p.ts[i])

    v.micro = _micro(p, i, ctx, cfg)
    v.tech, v.votes, v.factors = _technical(p, i, cfg)
    v.regime, v.regime_score = _regime(p, i, cfg)
    v.cross = _cross(ctx, cfg)

    v.score = (cfg.w_micro * v.micro + cfg.w_tech * v.tech
               + cfg.w_regime * v.regime_score + cfg.w_cross * v.cross)

    # --- hard vetoes (before the score is even considered) ---------------
    if v.regime == "trend_down":
        if not _mean_reversion_confirmed(p, i, cfg):
            v.veto, v.veto_reason = True, "counter-regime"
    elif v.regime == "high_vol":
        if not _mean_reversion_confirmed(p, i, cfg):
            v.veto, v.veto_reason = True, "high-vol-needs-mean-reversion"

    if not v.veto and ctx is not None and ctx.cross_invalidate >= cfg.invalidate_veto:
        v.veto, v.veto_reason = True, "cross-asset-invalidation"

    need = cfg.min_factors_event if (ctx and ctx.event_driven) else cfg.min_factors
    if not v.veto and v.votes < need:
        v.veto, v.veto_reason = True, "insufficient-confluence"

    return v


def scan(bars, ctx_by_ts=None, cfg=None):
    """Evaluate every bar -> list[SniperVerdict] (same length as bars)."""
    cfg = cfg or SniperConfig()
    p = prepare(bars, cfg)
    ctx_by_ts = ctx_by_ts or {}
    return [evaluate(p, i, ctx_by_ts.get(p.ts[i]), cfg) for i in range(p.n)]
