# =============================================================================
#  OmniSeed — tools/monster/__init__.py
#  The "Monster" trading intelligence layer (paper only).
#
#  Heavy logic lives here (Python); the C++ engine only ever receives the
#  distilled state vector (ts, confidence, veto, regime, detail). See
#  docs/MONSTER_DESIGN.md for the math and the honest-scope notes.
# =============================================================================
"""Monster trading intelligence layer — sniper scoring, news, correlation."""

__all__ = [
    "features",
    "sniper_engine",
    "news_hunter",
    "correlation_matrix",
    "sizing",
    "state_vector",
    "signals",
]
