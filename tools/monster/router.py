# =============================================================================
#  OmniSeed — tools/monster/router.py
#  Regime-adaptive allocation across the strategy zoo.
#
#  THE ONE DESIGN DECISION THAT MATTERS: continuous sizing, not a binary switch.
#
#  The obvious implementation is "if trending use momentum else use mean
#  reversion". That is also the expensive one: every flip pays the full spread
#  on the way out and back, and regime transitions are exactly when a binary
#  classifier is least sure. So instead:
#
#      w_trend = sigmoid(k * (trend_score - 0.5))
#      w_range = 1 - w_trend
#
#  and each strategy is weighted by its regime fit times its own confidence.
#  Near trend_score = 0.5 the router holds both engines at ~half size and
#  degrades gracefully through the ambiguity instead of whipsawing.
#
#  OUTPUT
#      conviction  net directional view in [-1,1]  (signed, weighted)
#      agreement   share of active weight that agrees with the sign of
#                  conviction, in [0,1]. Low agreement = the zoo is split.
#
#  The router can only ever BLOCK, never promote: see sniper_engine, where a
#  high-agreement opposing ensemble vetoes the entry. It never raises S. That
#  asymmetry is deliberate — an ensemble that can talk you INTO a trade is an
#  ensemble that can talk you into a bad one.
# =============================================================================
"""Regime-adaptive strategy routing: continuous blend -> conviction + agreement."""

import math

from . import strategies as ST

# how much a regime-agnostic ("both") strategy is worth. Always eligible, never
# the dominant voice.
FIT_BOTH_WEIGHT = 1.0
FIT_MATCH_BONUS = 1.0


class RouterConfig:
    k: float = 6.0                  # sigmoid steepness on trend_score
    ensemble_veto: float = 0.35     # conviction at/below which a long is opposed
    agreement_veto: float = 0.60    # agreement required for that veto to bind
    veto_min_weight: float = 0.50   # total active weight required for that veto
    min_confidence: float = 0.05    # ignore signals quieter than this

    def __init__(self, **kw):
        for kk, v in kw.items():
            if not hasattr(self, kk):
                raise TypeError("unknown RouterConfig field: %s" % kk)
            setattr(self, kk, v)


class EnsembleVerdict:
    __slots__ = ("ts", "conviction", "agreement", "weight", "w_trend", "active",
                 "signals", "reasons")

    def __init__(self, ts=0, conviction=0.0, agreement=0.0, weight=0.0,
                 w_trend=0.5, active=0, signals=None, reasons=None):
        self.ts = ts
        self.conviction = conviction
        self.agreement = agreement
        self.weight = weight
        self.w_trend = w_trend
        self.active = active
        self.signals = signals or []
        self.reasons = reasons or []

    @property
    def opposed_long(self):
        return self.conviction <= 0.0

    @property
    def size_factor(self):
        """Confidence-scaled exposure multiplier in [0.75, 1.25].

        Only ever applied when the ensemble AGREES (conviction > 0); an
        opposed ensemble does not get to shrink the size, it gets to veto.
        """
        if self.conviction <= 0.0:
            return 1.0
        return 0.75 + 0.50 * max(0.0, min(1.0, self.agreement))

    def detail(self):
        return ("conv=%+.3f agree=%.2f wt=%.2f n=%d"
                % (self.conviction, self.agreement, self.weight, self.active))

    def __repr__(self):
        return "Ensemble(%s)" % self.detail()


def _sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def regime_weight(fit, w_trend):
    """How much a strategy of this regime fit is worth right now."""
    if fit == "trend":
        return w_trend
    if fit == "range":
        return 1.0 - w_trend
    return FIT_BOTH_WEIGHT


def route(signals, regime=None, cfg=None, ts=0):
    """Blend strategy signals into a single ensemble verdict."""
    cfg = cfg or RouterConfig()
    ts_score = 0.5
    if regime is not None and getattr(regime, "trend_score", None) is not None:
        v = regime.trend_score
        if v is not None and math.isfinite(v):
            ts_score = v
    w_trend = _sigmoid(cfg.k * (ts_score - 0.5))

    total = 0.0
    signed = 0.0
    active = 0
    kept = []
    for sig in signals:
        if sig is None or sig.confidence < cfg.min_confidence:
            continue
        if sig.direction == 0.0:
            continue
        w = regime_weight(sig.regime_fit, w_trend) * sig.confidence
        if w <= 0:
            continue
        total += w
        signed += w * sig.direction
        active += 1
        kept.append((sig, w))

    if total <= 0:
        return EnsembleVerdict(ts=ts, conviction=0.0, agreement=0.0, weight=0.0,
                               w_trend=w_trend, active=0, signals=[])

    conviction = max(-1.0, min(1.0, signed / total))
    if abs(conviction) < 1e-12:
        agreement = 0.0
    else:
        sgn = 1.0 if conviction > 0 else -1.0
        agree_w = sum(w for sig, w in kept if sig.direction * sgn > 0)
        agreement = agree_w / total

    reasons = ["%s:%s" % (sig.name, sig.reason) for sig, _ in kept]
    return EnsembleVerdict(ts=ts, conviction=conviction, agreement=agreement,
                           weight=total, w_trend=w_trend, active=active,
                           signals=kept, reasons=reasons)


def should_veto_long(verdict, cfg=None):
    """True when a WEIGHTY, high-agreement ensemble actively opposes a long.

    The weight floor is not decoration. `agreement` is a SHARE, so a single
    active strategy always scores 1.00 — one lonely, low-confidence OFI proxy
    would otherwise veto every long in the book. Requiring real weight behind
    the disagreement is the difference between a consensus and a lone voice.
    """
    cfg = cfg or RouterConfig()
    return (verdict.active > 0
            and verdict.agreement >= cfg.agreement_veto
            and verdict.conviction <= -cfg.ensemble_veto
            and verdict.weight >= cfg.veto_min_weight)


def evaluate(s, i, regime=None, cfg=None, snapshots=None, scfg=None):
    """Convenience: run the whole zoo at bar `i` and route it."""
    sigs = ST.all_signals(s, i, regime, scfg, snapshots)
    return route(sigs, regime, cfg, ts=s.ts[i])


def scan(s, regimes, cfg=None, scfg=None, snapshots_by_ts=None):
    """Ensemble verdict for every bar. `regimes` must align with s.bars."""
    snapshots_by_ts = snapshots_by_ts or {}
    out = []
    for i in range(s.n):
        reg = regimes[i] if i < len(regimes) else None
        snaps = snapshots_by_ts.get(s.ts[i])
        out.append(evaluate(s, i, reg, cfg, snaps, scfg))
    return out
