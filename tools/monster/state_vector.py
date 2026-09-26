# =============================================================================
#  OmniSeed — tools/monster/state_vector.py
#  The DISTILLED output of the Monster layer.
#
#  This is the only thing that crosses into the C++ engine. Keeping it tiny is
#  the whole point: the heavy lifting (news, correlation, indicators) stays in
#  Python, and the edge brain receives a handful of floats plus a short
#  rationale string.
#
#  Engine-facing CSV schema (one row per bar, ascending ts):
#      ts,confidence,veto,regime,detail
#    ts          unix seconds (UTC) — matches the bar it was computed from
#    confidence  the sniper score S in [0,1]
#    veto        1 when a hard gate blocks the entry, else 0
#    regime      trend_up | trend_down | high_vol | range
#    detail      compact CSV-safe rationale (no commas)
# =============================================================================
"""Assemble and serialise the Monster state vector."""

from dataclasses import dataclass, field

from . import sizing as SZ

FEATURE_HEADER = "ts,confidence,veto,regime,detail"


@dataclass
class StateVector:
    ts: int = 0
    symbol: str = ""
    confidence: float = 0.0
    veto: bool = False
    regime: str = "range"
    micro: float = 0.0
    tech: float = 0.0
    regime_score: float = 0.0
    cross: float = 0.0
    votes: int = 0
    risk_pct: float = 0.0
    event_driven: bool = False
    detail: str = ""
    factors: list = field(default_factory=list)
    # --- regime / ensemble context (carried in `detail`, not in S) --------
    trend_score: float = float("nan")
    half_life: float = float("nan")
    conviction: float = 0.0
    agreement: float = 0.0

    def propose(self, min_confidence=0.85):
        return (not self.veto) and self.confidence >= min_confidence


def csv_safe(s):
    return s.replace(",", ";").replace("\n", " ").replace("\r", " ")


def build(symbol, verdict, atr_pct=None, sz_cfg=None):
    """Verdict (+ volatility) -> the distilled StateVector."""
    risk, f, g = SZ.adaptive_risk_pct(verdict.score, atr_pct, sz_cfg)
    detail = verdict.detail()
    if f != 1.0 or g != 1.0:
        detail += " size=%.4f" % risk
    return StateVector(
        ts=verdict.ts, symbol=symbol, confidence=verdict.score,
        veto=verdict.veto, regime=verdict.regime, micro=verdict.micro,
        tech=verdict.tech, regime_score=verdict.regime_score,
        cross=verdict.cross, votes=verdict.votes, risk_pct=risk,
        detail=detail, factors=list(verdict.factors),
        trend_score=getattr(verdict, "trend_score", float("nan")),
        half_life=getattr(verdict, "half_life", float("nan")),
        conviction=getattr(verdict, "conviction", 0.0),
        agreement=getattr(verdict, "agreement", 0.0))


def to_row(v):
    return "%d,%.4f,%d,%s,%s" % (v.ts, v.confidence, 1 if v.veto else 0,
                                 v.regime, csv_safe(v.detail))


def write_feature_csv(path, vectors):
    """Write the engine-facing feature CSV (header + one row per bar)."""
    import os
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(FEATURE_HEADER + "\n")
        for v in vectors:
            f.write(to_row(v) + "\n")
    return len(vectors)


def read_feature_csv(path):
    """-> {ts: (confidence, veto, regime, detail)}. Mirrors the C++ reader."""
    out = {}
    try:
        f = open(path, "r", encoding="utf-8")
    except OSError:
        return out
    with f:
        for line in f:
            line = line.strip()
            if not line or line[0].isdigit() is False:
                continue
            parts = line.split(",", 4)
            if len(parts) < 4:
                continue
            try:
                ts = int(parts[0])
                conf = float(parts[1])
            except ValueError:
                continue
            out[ts] = (conf, parts[2].strip() == "1", parts[3],
                       parts[4] if len(parts) > 4 else "")
    return out
