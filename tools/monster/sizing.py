# =============================================================================
#  OmniSeed — tools/monster/sizing.py
#  Adaptive position sizing: confidence scaling x inverse-volatility targeting,
#  under a HARD 2% ceiling.
#
#      f_conf = 0.75 + 0.50 * clamp((S - min_conf) / (1 - min_conf), 0, 1)
#      g_vol  = clamp(atr_ref_pct / atr_pct, 0.5, 1.5)
#      risk_pct = clamp(base_risk * f_conf * g_vol, 0.01, 0.02)
#      qty    = equity * risk_pct / (price * stop_pct)
#
#  The band [1%, 2%] is the SAME band the C++ RiskManager::size_by_risk
#  enforces (kMinRiskPerTradePct..kMaxRiskPerTradePct), so the Python plan and
#  the engine can never disagree about how much risk is allowed. The engine
#  remains the authority: even a bug here cannot exceed 2%.
#
#  The identity `qty = equity*risk / (price*stop)` is the risk-budget formula:
#  a stop-out at `stop_pct` loses exactly `risk_pct` of equity.
# =============================================================================
"""Adaptive ATR + confidence position sizing (<=2% hard cap)."""

from dataclasses import dataclass

MIN_RISK_PCT = 0.01     # mirrors C++ kMinRiskPerTradePct
MAX_RISK_PCT = 0.02     # mirrors C++ kMaxRiskPerTradePct


@dataclass
class SizingConfig:
    base_risk: float = 0.015
    min_confidence: float = 0.85
    atr_ref_pct: float = 0.02       # 2% ATR/close is the "neutral" volatility
    conf_lo: float = 0.75           # multiplier at the confidence threshold
    conf_hi: float = 1.25           # multiplier at S = 1.0
    vol_lo: float = 0.5
    vol_hi: float = 1.5


@dataclass
class Sizing:
    qty: float = 0.0
    risk_pct: float = 0.0
    f_conf: float = 1.0
    g_vol: float = 1.0
    allowed: bool = True
    reason: str = ""


def clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


def confidence_factor(confidence, cfg):
    if cfg.min_confidence >= 1.0:
        return cfg.conf_hi
    t = clamp((confidence - cfg.min_confidence) /
              (1.0 - cfg.min_confidence), 0.0, 1.0)
    return cfg.conf_lo + (cfg.conf_hi - cfg.conf_lo) * t


def volatility_factor(atr_pct, cfg):
    if atr_pct is None or not (atr_pct > 0):
        return 1.0                      # unknown vol -> neutral, never a bonus
    return clamp(cfg.atr_ref_pct / atr_pct, cfg.vol_lo, cfg.vol_hi)


def adaptive_risk_pct(confidence, atr_pct, cfg=None):
    """The Monster's smarter `risk_per_trade_pct`, clamped to [1%, 2%]."""
    cfg = cfg or SizingConfig()
    f = confidence_factor(confidence, cfg)
    g = volatility_factor(atr_pct, cfg)
    raw = cfg.base_risk * f * g
    return clamp(raw, MIN_RISK_PCT, MAX_RISK_PCT), f, g


def size(equity, price, stop_pct, confidence, atr_pct, cfg=None):
    """Risk-budget qty with the adaptive risk fraction. Refuses on bad input."""
    cfg = cfg or SizingConfig()
    if equity <= 0 or price <= 0 or stop_pct <= 0:
        return Sizing(allowed=False, reason="bad-inputs")
    risk, f, g = adaptive_risk_pct(confidence, atr_pct, cfg)
    qty = equity * risk / (price * stop_pct)
    s = Sizing(qty=qty, risk_pct=risk, f_conf=f, g_vol=g)
    if qty <= 0:
        return Sizing(allowed=False, reason="zero-qty", f_conf=f, g_vol=g,
                      risk_pct=risk)
    s.reason = ("risk=%.4f (base=%.4f x conf=%.2f x vol=%.2f)"
                % (risk, cfg.base_risk, f, g))
    return s
