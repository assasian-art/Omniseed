# =============================================================================
#  OmniSeed — tools/monster/factory.py
#  The strategy factory: a causal track record per zoo strategy, with automatic
#  promotion and retirement.
#
#  The ensemble in router.py weights every strategy by *regime fit*. That is a
#  statement about the MARKET. It says nothing about whether the strategy has
#  actually been RIGHT lately. A mean-reversion rule can fit the current regime
#  perfectly and still be losing money hand over fist.
#
#  The factory closes that gap the only honest way: it keeps score. Every time
#  a strategy fires, the factory remembers the direction and the bar. `horizon`
#  bars later the outcome is settled against the realized forward return. Wins
#  and losses accumulate into a z-score against the coin-flip null:
#
#      p_hat = wins / scored
#      z     = (p_hat - 0.5) / sqrt(0.25 / scored)
#
#  That z is the whole decision. Enough settled outcomes and z >= 0 earns
#  PROMOTED. Deeply negative z forces RETIRED for `retry_window` bars, then the
#  strategy is readmitted on PROBATION at shrunken weight and must re-earn its
#  place. A strategy with no track record is NEW / PROBATION and speaks quietly.
#
#  THREE INVARIANTS (the M3 contract, mirrors router.py's header):
#      1) BLOCK ONLY  — promotion is an eligibility FILTER. A retired signal is
#         removed from route(); nothing is ever boosted. Removing weight can
#         only shrink |conviction|, never grow it, and never un-veto.
#      2) NO LOOK-AHEAD — at bar i only outcomes at indices <= i are read. A
#         firing settles only once the bar it is judged against is in the past.
#      3) FAIL CLOSED, NAMED REASONS — unknown strategy, no track record, or
#         retired: the factory says exactly that in the reason string, it never
#         silently passes a signal through.
# =============================================================================
"""Causal per-strategy track record with automatic promotion and retirement."""

import math

from . import features as F
from . import router as RT
from . import strategies as ST


def forward_return(close, i, horizon):
    """close[i+horizon]/close[i] - 1.0. Only ever called for settled (past) i."""
    return close[i + horizon] / close[i] - 1.0


class FactoryConfig:
    warmup: int = 5                 # settled outcomes before promotion is allowed
    horizon: int = 5                # bars held after a firing before it is scored
    min_confidence: float = 0.20    # a firing must clear this to be recorded
    retire_z: float = -2.0          # z-score of skill that forces retirement
    retry_window: int = 250         # bars a retired strategy waits for retry
    max_simultaneous: int = 0       # 0 = unlimited concurrent promoted
    shrinkage: float = 0.5          # weight multiplier on probationary signals

    def __init__(self, **kw):
        for kk, v in kw.items():
            if not hasattr(self, kk):
                raise TypeError("unknown FactoryConfig field: %s" % kk)
            setattr(self, kk, v)


class Record:
    """Running score for one strategy. State transitions live in Factory."""
    __slots__ = ("name", "firings", "scored", "wins", "losses", "sum_r",
                 "sum_r2", "last_outcome_i", "state", "retired_at_i",
                 "next_retry_i", "longest_promoted_run", "_promoted_run",
                 "retire_reason")

    def __init__(self, name):
        self.name = name
        self.firings = 0
        self.scored = 0
        self.wins = 0
        self.losses = 0
        self.sum_r = 0.0
        self.sum_r2 = 0.0
        self.last_outcome_i = -1
        self.state = "new"
        self.retired_at_i = -1
        self.next_retry_i = -1
        self.longest_promoted_run = 0
        self._promoted_run = 0
        self.retire_reason = ""

    def z(self):
        if self.scored <= 0:
            return 0.0
        p_hat = self.wins / self.scored
        return (p_hat - 0.5) / math.sqrt(0.25 / self.scored)

    def win_rate(self):
        return (self.wins / self.scored) if self.scored > 0 else 0.0

    def expectancy(self):
        return (self.sum_r / self.scored) if self.scored > 0 else 0.0

    def settle(self, r, direction, i, cfg):
        """Score one matured outcome: r = forward return over the holding."""
        self.scored += 1
        self.sum_r += r
        self.sum_r2 += r * r
        self.last_outcome_i = i
        signed = direction * r
        if signed > 0:
            self.wins += 1
        elif signed < 0:
            self.losses += 1
        if self.scored > 0:
            z = self.z()
            if z < cfg.retire_z and self.state != "retired":
                self.state = "retired"
                self.retired_at_i = i
                self.next_retry_i = i + cfg.retry_window
                self.retire_reason = "z=%.2f" % z


class Factory:
    """Owns the zoo's track records and decides who is allowed to speak."""

    def __init__(self, bars, cfg=None, names=None):
        self.cfg = cfg or FactoryConfig()
        self.bars = bars
        self.n = len(bars)
        self.close = F.closes(bars)
        self.ts = [int(b[0]) if not isinstance(b, dict) else int(b["time"])
                   for b in bars]
        self.names = list(names) if names is not None else list(ST.ALL)
        self.records = {nm: Record(nm) for nm in self.names}
        self.pending = {nm: [] for nm in self.names}

    def status_string(self, rec, i):
        """Recompute state for bar i and return its named reason. Fail-closed."""
        cfg = self.cfg
        z = rec.z()
        if i < cfg.warmup:
            if rec.state != "promoted":
                rec._promoted_run = 0
            rec.state = "new"
            return "warmup(%d)" % cfg.warmup
        if rec.state == "retired":
            if i >= rec.next_retry_i:
                rec.state = "probation"
            else:
                return ("retired(%s@bar%d,retry@bar%d)"
                        % (rec.retire_reason or "z", rec.retired_at_i,
                           rec.next_retry_i))
        if rec.state != "retired":
            if rec.scored >= cfg.warmup and z < cfg.retire_z:
                rec.state = "retired"
                rec.retired_at_i = i
                rec.next_retry_i = i + cfg.retry_window
                rec.retire_reason = "z=%.2f" % z
                rec._promoted_run = 0
                return ("retired(%s@bar%d,retry@bar%d)"
                        % (rec.retire_reason, rec.retired_at_i,
                           rec.next_retry_i))
            if rec.scored >= cfg.warmup and z >= 0.0:
                if rec.state != "promoted":
                    rec._promoted_run = 0
                rec.state = "promoted"
                rec._promoted_run += 1
                if rec._promoted_run > rec.longest_promoted_run:
                    rec.longest_promoted_run = rec._promoted_run
                return "promoted(z=%.2f,n=%d)" % (z, rec.scored)
            rec.state = "probation"
            rec._promoted_run = 0
        if rec.state == "probation":
            if rec.firings == 0:
                return "probation(awaiting-outcome)"
            if rec.scored < cfg.warmup:
                return "probation(%d/%d)" % (rec.scored, cfg.warmup)
            return "probation(z=%.2f)" % z
        return "probation(awaiting-outcome)"

    def _settle(self, i):
        """Score every firing whose horizon has elapsed by bar i. Causal."""
        horizon = self.cfg.horizon
        for nm in self.names:
            rec = self.records[nm]
            q = self.pending[nm]
            k = 0
            while k < len(q) and q[k][0] + horizon <= i:
                j, direction = q[k]
                r = forward_return(self.close, j, horizon)
                rec.settle(r, direction, j, self.cfg)
                k += 1
            if k:
                del q[:k]

    def evaluate(self, sigs, i, regime=None, cfg=None, ts=0):
        cfg = cfg or RT.RouterConfig()
        fcfg = self.cfg
        raw = list(sigs)
        known = set(self.names)
        base = RT.route(raw, regime, cfg, ts=ts)
        base_conviction = base.conviction
        base_agreement = base.agreement
        known_base = RT.route([s for s in raw if s is not None and s.name in known],
                              regime, cfg, ts=ts)

        self._settle(i)

        for sig in raw:
            if sig is None:
                continue
            if sig.name not in known:
                continue
            if sig.confidence >= fcfg.min_confidence and sig.direction != 0.0:
                rec = self.records[sig.name]
                rec.firings += 1
                self.pending[sig.name].append((i, sig.direction))

        states = {}
        status = []
        for sig in raw:
            if sig is None:
                continue
            if sig.name not in known:
                states[sig.name] = "unknown"
                status.append("%s:unknown(not-in-zoo)" % sig.name)
                continue
            rec = self.records[sig.name]
            reason = self.status_string(rec, i)
            states[sig.name] = rec.state
            status.append("%s:%s(%s)" % (sig.name, rec.state, reason))

        kept = []
        for sig in raw:
            if sig is None or sig.name not in known:
                continue
            st = states.get(sig.name, "new")
            if st == "promoted":
                kept.append(sig)
            elif st == "probation":
                kept.append(ST.StrategySignal(sig.name, sig.direction,
                                              sig.confidence * fcfg.shrinkage,
                                              sig.regime_fit, sig.reason))

        kept_verdict = RT.route(kept, regime, cfg, ts=ts)
        base_veto = RT.should_veto_long(known_base, cfg)
        kept_veto = RT.should_veto_long(kept_verdict, cfg)
        veto = base_veto or kept_veto
        if base_veto and kept_veto:
            which = "both"
        elif kept_veto:
            which = "kept"
        else:
            which = "base"

        if kept_verdict.active > 0:
            conviction, agreement = kept_verdict.conviction, kept_verdict.agreement
            weight = kept_verdict.weight
            active = kept_verdict.active
            kc, bc = kept_verdict.conviction, base_conviction
            if (kc > 0) != (bc > 0) and bc != 0.0:
                conviction, agreement = base_conviction, base_agreement
            elif abs(kc) > abs(bc) + 1e-9:
                conviction, agreement = base_conviction, base_agreement
        else:
            conviction, agreement = base_conviction, base_agreement
            weight = base.weight
            active = base.active

        reasons = []
        if base.reasons:
            reasons.append("base[" + " ".join(base.reasons) + "]")
        reasons.extend(status)
        if veto:
            reasons.append("VETO=%s" % which)
        veto_reason = ("ensemble-opposed(%s)" % which) if veto else ""

        return FactoryVerdict(ts=ts, conviction=conviction, agreement=agreement,
                              weight=weight, active=active,
                              base_conviction=base_conviction,
                              base_agreement=base_agreement, states=states,
                              reasons=reasons, veto=veto, veto_reason=veto_reason)

    def scan(self, regimes=None, router_cfg=None, factory_cfg=None,
             snapshots_by_ts=None):
        """One FactoryVerdict per bar. `regimes` aligns with self.bars."""
        if factory_cfg is not None:
            self.cfg = factory_cfg
        snapshots_by_ts = snapshots_by_ts or {}
        series = ST.prepare(self.bars)
        out = []
        for i in range(self.n):
            reg = regimes[i] if (regimes is not None and i < len(regimes)) else None
            snaps = snapshots_by_ts.get(self.ts[i])
            sigs = ST.all_signals(series, i, reg, series.cfg, snaps)
            out.append(self.evaluate(sigs, i, reg, router_cfg, ts=self.ts[i]))
        return out

    def stats(self):
        out = {}
        for nm in self.names:
            rec = self.records[nm]
            out[nm] = {
                "state": rec.state,
                "firings": rec.firings,
                "scored": rec.scored,
                "wins": rec.wins,
                "losses": rec.losses,
                "win_rate": rec.win_rate(),
                "expectancy": rec.expectancy(),
                "z": rec.z(),
                "longest_promoted_run": rec.longest_promoted_run,
            }
        return out


class FactoryVerdict:
    __slots__ = ("ts", "conviction", "agreement", "weight", "active",
                 "base_conviction", "base_agreement", "states", "reasons",
                 "veto", "veto_reason")

    def __init__(self, ts=0, conviction=0.0, agreement=0.0, weight=0.0,
                 active=0, base_conviction=0.0, base_agreement=0.0,
                 states=None, reasons=None, veto=False, veto_reason=""):
        self.ts = ts
        self.conviction = conviction
        self.agreement = agreement
        self.weight = weight
        self.active = active
        self.base_conviction = base_conviction
        self.base_agreement = base_agreement
        self.states = states or {}
        self.reasons = reasons or []
        self.veto = veto
        self.veto_reason = veto_reason

    def detail(self):
        parts = ["ens=%+.3f/%.2f(base=%+.3f) n=%d"
                 % (self.conviction, self.agreement, self.base_conviction,
                    self.active)]
        if self.veto:
            parts.append("VETO")
        for chunk in self.reasons:
            if chunk.startswith("base["):
                continue
            parts.append(chunk)
        return " ".join(parts)

    def __repr__(self):
        return "FactoryVerdict(%s)" % self.detail()
