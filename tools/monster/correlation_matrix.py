# =============================================================================
#  OmniSeed — tools/monster/correlation_matrix.py
#  Cross-asset confirmation / invalidation.
#
#  Markets do not move in isolation: a long-oil setup is invalidated when the
#  broader commodity complex is crashing. This module keeps a rolling Pearson
#  matrix over a trailing window and evaluates a small table of LEAD-LAG
#  relations against the intended trade direction.
#
#  It only ever CONFIRMS or INVALIDATES — it never originates a signal.
#  The empirical correlation is reported next to the assumed sign so a stale
#  assumption is visible rather than silent.
# =============================================================================
"""Rolling cross-asset correlation + lead-lag confirmation."""

from dataclasses import dataclass, field

from . import features as F

# (driver, responder, expected_sign_of_responder_vs_driver, rationale)
DEFAULT_PAIRS = [
    ("GOLD", "USDJPY", -1, "gold up => yen-funded carry unwinds"),
    ("OIL", "CADUSD", +1, "petro-currency"),
    ("OIL", "AIRLINES", -1, "input cost"),
    ("BTC", "QQQ", +1, "risk-on beta"),
    ("USD", "GOLD", -1, "dollar numeraire"),
    ("DXY", "GOLD", -1, "dollar numeraire"),
    ("OIL", "COMMODITY_INDEX", +1, "broad complex"),
]


@dataclass
class PairResult:
    driver: str
    responder: str
    expected: int
    actual_corr: float = 0.0
    active: bool = False
    confirming: bool = False
    invalidating: bool = False
    note: str = ""


@dataclass
class CrossResult:
    confirm: int = 0
    invalidate: int = 0
    active: int = 0
    pairs: list = field(default_factory=list)

    def detail(self):
        bits = []
        for p in self.pairs:
            if not p.active:
                continue
            tag = "confirm" if p.confirming else (
                "invalidate" if p.invalidating else "flat")
            bits.append("%s->%s:%s" % (p.driver, p.responder, tag))
        return "+".join(bits) if bits else "none"


class CorrelationMatrix:
    def __init__(self, window=60, pairs=None, move_threshold=0.005):
        self.window = int(window)
        self.pairs = list(pairs if pairs is not None else DEFAULT_PAIRS)
        self.move_threshold = float(move_threshold)

    # -- correlation -------------------------------------------------------
    def matrix(self, series_by_symbol, i):
        """Rolling Pearson for every pair that has data. -> {(a,b): r}."""
        out = {}
        syms = [s for s, v in series_by_symbol.items() if len(v) > i >= 0]
        for a in range(len(syms)):
            for b in range(a + 1, len(syms)):
                sa, sb = syms[a], syms[b]
                va, vb = series_by_symbol[sa], series_by_symbol[sb]
                r = self._rolling_corr(va, vb, i)
                if r is not None:
                    out[(sa, sb)] = r
        return out

    def _rolling_corr(self, va, vb, i):
        w = self.window
        if i + 1 < w or len(va) < i + 1 or len(vb) < i + 1:
            return None
        wa = va[i - w + 1:i + 1]
        wb = vb[i - w + 1:i + 1]
        ma, mb = sum(wa) / w, sum(wb) / w
        num = sum((wa[j] - ma) * (wb[j] - mb) for j in range(w))
        da = sum((x - ma) ** 2 for x in wa)
        db = sum((x - mb) ** 2 for x in wb)
        if da <= 0 or db <= 0:
            return 0.0
        return num / (da * db) ** 0.5

    # -- lead-lag evaluation ----------------------------------------------
    def evaluate(self, target, direction, rets_by_symbol, i):
        """Evaluate every relation touching `target` at index `i`.

        `direction` is +1 for a long, -1 for a short. A relation is ACTIVE when
        the leader moved more than `move_threshold`. It then has to pass TWO
        independent checks to CONFIRM:

          1. COHERENCE  — the follower moved the way the assumed sign predicts
                          (follower == sign x leader). An incoherent complex is
                          always an invalidation.
          2. ALIGNMENT  — the target itself is moving in OUR direction. A
                          perfectly coherent complex that is dragging the target
                          against our side confirms the OPPOSITE trade, so it
                          must invalidate ours.
        """
        res = CrossResult()
        d = 1 if direction >= 0 else -1

        def sgn(x):
            return 1 if x > 0 else (-1 if x < 0 else 0)

        for (driver, responder, expected, _why) in self.pairs:
            if target == driver:
                leader, follower, sign = driver, responder, expected
            elif target == responder:
                leader, follower, sign = driver, responder, expected
            else:
                continue
            if leader not in rets_by_symbol or follower not in rets_by_symbol:
                continue
            rl = rets_by_symbol[leader]
            rf = rets_by_symbol[follower]
            if i >= len(rl) or i >= len(rf):
                continue
            r_leader, r_follower = rl[i], rf[i]
            if not (r_leader == r_leader and r_follower == r_follower):
                continue                       # NaN guard
            pr = PairResult(driver=driver, responder=responder, expected=expected)
            pr.actual_corr = self._rolling_corr(rl, rf, i) or 0.0
            if abs(r_leader) < self.move_threshold:
                pr.note = "leader flat"
                res.pairs.append(pr)
                continue
            got_follower = sgn(r_follower)
            if got_follower == 0:
                # A quiet follower carries no information. It is NOT a
                # contradiction, so it must never count as an invalidation.
                pr.note = "follower flat"
                res.pairs.append(pr)
                continue
            pr.active = True
            res.active += 1

            want_follower = sgn(sign * r_leader)
            coherent = (got_follower == want_follower)
            aligned = (sgn(r_leader) == d if target == leader
                       else got_follower == d)

            if coherent and aligned:
                pr.confirming = True
                res.confirm += 1
            else:
                pr.invalidating = True
                res.invalidate += 1
                pr.note = ("complex coherent but target against us"
                           if coherent else "incoherent complex")
            res.pairs.append(pr)
        return res
