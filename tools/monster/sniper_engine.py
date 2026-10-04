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

from . import factory as FA
from . import features as F
from . import regime as RG
from . import router as RT
from . import strategies as ST


@dataclass
class SniperConfig:
    # --- layer weights (must sum to ~1.0) --------------------------------
    w_micro: float = 0.25
    w_tech: float = 0.35
    w_regime: float = 0.25
    w_cross: float = 0.15
    # --- optional vol-harvesting layer (M4) -----------------------------
    # Opt-in: default 0.0 keeps M3 bit-identical. When w_vol > 0 the
    # six vol strategies (vol_arb … var_swap) contribute a fifth
    # signal blend. The total is renormalised so w_micro+w_tech+
    # w_regime+w_cross+w_vol ≈ 1.0 when active.
    w_vol: float = 0.0
    vol_signals: tuple = ("vol_arb", "butterfly_arb", "skew_trend",
                          "calendar_spread", "gex_regime", "var_swap")
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
    #
    # regime_mode:
    #   "advanced" (default) — the multi-axis ensemble in regime.py: variance
    #       ratio, Hurst, efficiency ratio, ADX, choppiness, R², rho(1) and
    #       MA-ribbon consistency, with Yang-Zhang volatility and an OU
    #       half-life, plus hysteresis.
    #   "legacy" — the original single-statistic SMA200-slope rule, kept so the
    #       behaviour can be reproduced and A/B compared.
    regime_mode: str = "advanced"
    # --- strategy factory (opt-in; ensemble=True alone is untouched) --------
    # ensemble_factory routes the ensemble through factory.py's causal track
    # record (promotion/retirement = eligibility filter, BLOCK only). It is an
    # opt-in flag so that plain `ensemble=True` stays bit-identical with the
    # C++ omniseed_strategy_dump parity gate.
    ensemble_factory: bool = False
    sma_regime: int = 200
    regime_slope_n: int = 20
    regime_slope_s: float = 0.01    # 1% SMA200 move over n bars
    atr_period: int = 14
    atr_hi: float = 0.05            # 5% ATR/close = high-volatility regime
    # --- strategy ensemble (opt-in; the advanced regime engine is required) ---
    #
    # The ensemble can only ever BLOCK, never promote. It never raises S: an
    # ensemble that can talk you into a trade can talk you into a bad one.
    ensemble: bool = False
    ensemble_veto: float = 0.35
    ensemble_agreement: float = 0.60
    ensemble_min_weight: float = 0.50
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


def _fin(x):
    """True only for a real, finite number. None-safe on purpose."""
    return isinstance(x, (int, float)) and math.isfinite(x)


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
    # --- ensemble / regime detail (never part of S) -----------------------
    trend_score: float = float("nan")
    half_life: float = float("nan")
    conviction: float = 0.0
    agreement: float = 0.0

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
        if _fin(self.trend_score):
            parts.append("trend=%.3f" % self.trend_score)
        if _fin(self.half_life):
            parts.append("hl=%.0f" % self.half_life)
        if self.conviction or self.agreement:
            parts.append("ens=%+.2f/%.2f" % (self.conviction, self.agreement))
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
        # --- advanced regime + strategy ensemble (optional, causal) ---------
        self.regimes = None
        self.ensemble = None
        self.factory = None
        self.vol_series = None
        if cfg.regime_mode == "advanced":
            self.regimes = RG.scan(bars)
        if cfg.ensemble and self.regimes is not None:
            self.series = ST.prepare(bars)
            self.ensemble = RT.scan(
                self.series, self.regimes,
                RT.RouterConfig(ensemble_veto=cfg.ensemble_veto,
                                agreement_veto=cfg.ensemble_agreement,
                                veto_min_weight=cfg.ensemble_min_weight))
        if cfg.ensemble_factory and self.regimes is not None:
            if not hasattr(self, "series"):
                self.series = ST.prepare(bars)
            self.factory = FA.Factory(bars)
        if cfg.w_vol > 0.0:
            self.vol_series = ST.prepare(bars)


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


# The label -> layer-score map is the mandate's: a with-trend entry gets the
# full 1.0, a counter-trend one gets 0.0 and is vetoed below. Only the SOURCE of
# the label changed, never the mapping.
REGIME_SCORE = {"trend_up": 1.0, "range": 0.5, "high_vol": 0.25, "trend_down": 0.0}


def _regime_legacy(p, i, cfg):
    """The original single-statistic rule: SMA200 slope + ATR%. Kept for A/B."""
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


def _regime(p, i, cfg):
    """-> (name, score). Names: trend_up / trend_down / high_vol / range.

    Default path is the advanced multi-axis engine. It is strictly better
    informed than a single SMA200 slope: a slope rule is a low-volatility
    filter in disguise on equities (where low vol and uptrends coincide) and
    stops working the moment you move to FX.
    """
    if cfg.regime_mode == "legacy" or p.regimes is None:
        return _regime_legacy(p, i, cfg)
    st = p.regimes[i]
    label = st.label if st.label in REGIME_SCORE else "range"
    return label, REGIME_SCORE[label]


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
# M4 — VOL LAYER
# ---------------------------------------------------------------------------
def _vol_layer(p, i, cfg):
    """Average confidence from active vol strategies at bar i."""
    if not hasattr(p, "vol_series") or p.vol_series is None:
        return 0.0
    active = [sig for sig in ST.all_signals(p.vol_series, i, None, None)
              if sig.name in cfg.vol_signals and sig.active]
    if not active:
        return 0.0
    return sum(sig.confidence for sig in active) / len(active)


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

    # --- M4 vol-harvesting layer (opt-in, w_vol > 0) -------------
    v_vol = 0.0
    if cfg.w_vol > 0.0:
        v_vol = _vol_layer(p, i, cfg)

    v.score = (cfg.w_micro * v.micro + cfg.w_tech * v.tech
               + cfg.w_regime * v.regime_score + cfg.w_cross * v.cross
               + cfg.w_vol * v_vol)

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

    # --- strategy ensemble: BLOCK only, never promote --------------------
    #
    # This is deliberately the last gate and it can only ever say no. An
    # ensemble that can talk you into a trade can talk you into a bad one, and
    # S must stay exactly the mandate's weighted sum.
    if p.factory is not None:
        st = p.regimes[i]
        sigs = ST.all_signals(p.series, i, st, None, None)
        fv = p.factory.evaluate(
            sigs, i, st,
            RT.RouterConfig(ensemble_veto=cfg.ensemble_veto,
                            agreement_veto=cfg.ensemble_agreement,
                            veto_min_weight=cfg.ensemble_min_weight),
            ts=p.ts[i])
        v.conviction, v.agreement = fv.conviction, fv.agreement
        if not v.veto and fv.veto:
            v.veto, v.veto_reason = True, "ensemble-opposed"
    elif p.ensemble is not None:
        ev = p.ensemble[i]
        v.conviction, v.agreement = ev.conviction, ev.agreement
        if not v.veto and RT.should_veto_long(
                ev, RT.RouterConfig(ensemble_veto=cfg.ensemble_veto,
                                    agreement_veto=cfg.ensemble_agreement,
                                    veto_min_weight=cfg.ensemble_min_weight)):
            v.veto, v.veto_reason = True, "ensemble-opposed"

    if p.regimes is not None:
        st = p.regimes[i]
        v.trend_score = st.trend_score
        v.half_life = st.half_life

    return v


def scan(bars, ctx_by_ts=None, cfg=None):
    """Evaluate every bar -> list[SniperVerdict] (same length as bars)."""
    cfg = cfg or SniperConfig()
    p = prepare(bars, cfg)
    ctx_by_ts = ctx_by_ts or {}
    return [evaluate(p, i, ctx_by_ts.get(p.ts[i]), cfg) for i in range(p.n)]
