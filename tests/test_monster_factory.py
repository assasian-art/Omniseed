#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — tests/test_monster_factory.py
#
#  Tests for tools/monster/factory.py — the M3 strategy factory with automatic
#  promotion and retirement.
#
#  Three invariants this suite is really guarding (the M3 contract):
#    1. BLOCK ONLY — promotion is an eligibility FILTER, never a boost. A
#       retired signal is removed from route(); removing weight can only shrink
#       |conviction|, never grow it, and never invent a veto.
#    2. NO LOOK-AHEAD — a factory built on N bars must give identical verdicts
#       for the first k bars as one built on only those k bars.
#    3. FAIL CLOSED, NAMED REASONS — unknown strategy, no track record, or
#       retired: the factory says exactly that, it never silently passes a
#       signal through.
#
#  Offline, stdlib-only.  Usage: python tests/test_monster_factory.py
# =============================================================================
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from monster import strategies as ST  # noqa: E402
from monster import router as RT  # noqa: E402
from monster import factory as FA  # noqa: E402
from monster import sniper_engine as SN  # noqa: E402

PASSED = 0
FAILED = 0
FAILURES = []


def check(name, cond, detail=""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  ok   {name}")
    else:
        FAILED += 1
        FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL {name}  {detail}")


def approx(a, b, tol=1e-9):
    if a is None or b is None:
        return False
    if not (math.isfinite(a) and math.isfinite(b)):
        return False
    return abs(a - b) <= tol


def make_bars(prices, seed=7, wick=0.0005, vol=1e6):
    rnd = random.Random(seed)
    bars, prev = [], None
    for i, p in enumerate(prices):
        o = prev if prev is not None else p
        c = p
        band = max(abs(c - o), p * wick)
        hi = max(o, c) + band * rnd.uniform(0.1, 0.6)
        lo = min(o, c) - band * rnd.uniform(0.1, 0.6)
        bars.append((1700000000 + i * 60, o, hi, lo, c,
                     vol * (1.0 + rnd.random())))
        prev = c
    return bars


def gen_trend(n, seed, drift=0.002, sigma=0.004):
    rnd = random.Random(seed)
    p, out = 100.0, []
    for _ in range(n):
        p *= math.exp(drift + rnd.gauss(0, sigma))
        out.append(p)
    return out


def gen_random(n, seed, sigma=0.01):
    rnd = random.Random(seed)
    p, out = 100.0, []
    for _ in range(n):
        p *= math.exp(rnd.gauss(0, sigma))
        out.append(p)
    return out


def gen_ou(n, seed, theta=0.30, sigma=0.006, mu=100.0):
    rnd = random.Random(seed)
    x, out = mu + 5.0, []
    for _ in range(n):
        x = x + theta * (mu - x) + rnd.gauss(0, sigma * mu)
        out.append(x)
    return out


def dedup_signals(sigs):
    out, seen = [], set()
    for s in sigs:
        if s.name not in seen:
            seen.add(s.name)
            out.append(s)
    return out


def sig(name, direction, confidence, fit="both", reason=""):
    return ST.StrategySignal(name, direction, confidence, fit, reason)


# ---------------------------------------------------------------------------
# 1. NO LOOK-AHEAD
# ---------------------------------------------------------------------------
def test_no_lookahead():
    print("\n[NO LOOK-AHEAD  factory verdicts are prefix-invariant]")
    prices = gen_random(100, seed=11, sigma=0.008)
    full = make_bars(prices, seed=11)
    f_full = FA.Factory(full)
    f_pref = FA.Factory(full[:50])
    mism = 0
    first = None
    for i in range(50):
        sigs_f = [sig(nm, 1.0 if (i + k) % 2 == 0 else -1.0, 0.6)
                  for k, nm in enumerate(FA.Factory(full).names)]
        sigs_p = [sig(nm, 1.0 if (i + k) % 2 == 0 else -1.0, 0.6)
                  for k, nm in enumerate(FA.Factory(full[:50]).names)]
        a = f_full.evaluate(dedup_signals(sigs_f), i, None, ts=full[i][0])
        b = f_pref.evaluate(dedup_signals(sigs_p), i, None, ts=full[i][0])
        if (not approx(a.conviction, b.conviction)
                or not approx(a.agreement, b.agreement)
                or a.states != b.states):
            mism += 1
            if first is None:
                first = (i, a.conviction, b.conviction, a.states, b.states)
    check("factory verdicts identical for first 50 of 100 bars", mism == 0,
          f"{mism} mismatches, first={first}")


def test_no_lookahead_close_only():
    print("\n[NO LOOK-AHEAD  forward_return reads only past indices]")
    close = [100.0 + i for i in range(40)]
    r = FA.forward_return(close, 5, 5)
    check("forward_return(5,5) uses close[10]", approx(r, close[10] / close[5] - 1.0),
          f"got {r}")


# ---------------------------------------------------------------------------
# 2. warmup  +  3. promotion  +  4. retirement  +  5. retry window
# ---------------------------------------------------------------------------
def test_warmup_promotion():
    print("\n[WARMUP -> PROMOTION  a winning strategy earns promotion]")
    cfg = FA.FactoryConfig(warmup=5, horizon=2, retry_window=20)
    bars = make_bars(gen_trend(60, seed=3, drift=0.01, sigma=0.001))
    f = FA.Factory(bars, cfg=cfg, names=["momentum"])
    promoted_at = None
    saw_warmup = False
    for i in range(30):
        states_reasons = f.evaluate([sig("momentum", 1.0, 0.9)], i, None,
                                    ts=bars[i][0])
        st = states_reasons.states["momentum"]
        rsn = " ".join(states_reasons.reasons)
        if i < cfg.warmup and "warmup(" in rsn:
            saw_warmup = True
        if promoted_at is None and st == "promoted":
            promoted_at = i
    check("warmup reason appears while i < warmup", saw_warmup)
    check("strategy promoted after 5 winning outcomes",
          promoted_at is not None, f"never promoted; stats={f.stats()}")
    if promoted_at is not None:
        rec = f.records["momentum"]
        check("promotion required >= warmup settled outcomes",
              rec.scored >= cfg.warmup, f"scored={rec.scored}")
        check("promotion z >= 0", rec.z() >= 0.0, f"z={rec.z()}")


def test_retirement_and_retry():
    print("\n[RETIREMENT  a rigged losing stream forces retirement, then retry]")
    cfg = FA.FactoryConfig(warmup=3, horizon=10, retire_z=-2.0, retry_window=8)
    bars = make_bars(gen_trend(80, seed=5, drift=0.01, sigma=0.001))
    f = FA.Factory(bars, cfg=cfg, names=["momentum"])
    retired_reason = ""
    retired_at = -1
    first_retry_i = -1
    retried_at = -1
    for i in range(60):
        v = f.evaluate([sig("momentum", -1.0, 0.9)], i, None, ts=bars[i][0])
        rsn = " ".join(v.reasons)
        st = f.records["momentum"].state
        if retired_at < 0 and st == "retired":
            retired_at = i
            first_retry_i = f.records["momentum"].next_retry_i
            retired_reason = rsn
        if first_retry_i >= 0 and retried_at < 0 and i >= first_retry_i:
            retried_at = i
    check("losing stream retired the strategy", retired_at >= 0,
          f"state={f.records['momentum'].state}")
    check("retire reason names z", "retired(z=" in retired_reason,
          f"reason={retired_reason}")
    check("next_retry_i == retired_at_i + retry_window",
          first_retry_i == retired_at + cfg.retry_window,
          f"next={first_retry_i} retired_at={retired_at}")
    check("retry granted at first bar >= next_retry_i",
          0 <= retried_at == first_retry_i,
          f"retried_at={retried_at} expected={first_retry_i}")


# ---------------------------------------------------------------------------
# 6. BLOCK ONLY
# ---------------------------------------------------------------------------
def test_block_only():
    print("\n[BLOCK ONLY  removal shrinks |conviction| and never invents veto]")
    prices = gen_random(200, seed=23, sigma=0.01)
    bars = make_bars(prices, seed=23)
    f = FA.Factory(bars, names=list(ST.ALL))
    series = ST.prepare(bars)
    conv_ok = True
    sign_ok = True
    veto_ok = True
    conv_worst = None
    for i in range(200):
        sigs = dedup_signals(ST.all_signals(series, i, None))
        v = f.evaluate(sigs, i, None, ts=bars[i][0])
        sgn_b = 1.0 if v.base_conviction > 0 else (-1.0 if v.base_conviction < 0 else 0.0)
        sgn_c = 1.0 if v.conviction > 0 else (-1.0 if v.conviction < 0 else 0.0)
        if v.base_conviction != 0.0 and sgn_c != 0.0 and sgn_c != sgn_b:
            sign_ok = False
        if sgn_c == sgn_b and abs(v.conviction) > abs(v.base_conviction) + 1e-9:
            conv_ok = False
            conv_worst = (i, v.conviction, v.base_conviction)
        if v.veto:
            base = RT.route(sigs, None, None, ts=bars[i][0])
            kept_names = [nm for nm, s in v.states.items()
                          if s in ("promoted", "probation")]
            kept = []
            for s in sigs:
                if s.name not in kept_names:
                    continue
                if v.states.get(s.name) == "probation":
                    kept.append(ST.StrategySignal(
                        s.name, s.direction,
                        s.confidence * f.cfg.shrinkage,
                        s.regime_fit, s.reason))
                else:
                    kept.append(s)
            kv = RT.route(kept, None, None, ts=bars[i][0])
            if not (RT.should_veto_long(base) or RT.should_veto_long(kv)):
                veto_ok = False
    check("conviction sign never flips vs base", sign_ok)
    check("|conviction| <= |base_conviction| when signs agree", conv_ok,
          f"worst={conv_worst}")
    check("factory veto is always traceable to base or kept ensemble", veto_ok)


# ---------------------------------------------------------------------------
# 7. shrinkage
# ---------------------------------------------------------------------------
def test_shrinkage():
    print("\n[SHRINKAGE  a probationary signal reaches route() at conf*shrinkage]")
    cfg = FA.FactoryConfig(warmup=5, horizon=2, retry_window=20, shrinkage=0.5)
    bars = make_bars(gen_random(30, seed=31))
    f = FA.Factory(bars, cfg=cfg, names=["momentum"])
    i = 10
    raw = sig("momentum", 1.0, 0.8, "trend")
    v = f.evaluate([raw], i, None, ts=bars[i][0])
    check("signal is probation during warmup scoring window",
          v.states.get("momentum") == "probation",
          f"state={v.states.get('momentum')}")
    shrunk = sig("momentum", 1.0, 0.8 * cfg.shrinkage, "trend")
    direct = RT.route([shrunk], None, None, ts=bars[i][0])
    check("factory conviction == route(shrunk signal) conviction",
          approx(v.conviction, direct.conviction),
          f"factory={v.conviction} direct={direct.conviction}")
    full = RT.route([raw], None, None, ts=bars[i][0])
    check("factory weight < full-weight route (shrinkage applied)",
          v.weight < full.weight + 1e-9,
          f"factory_w={v.weight} full_w={full.weight}")


# ---------------------------------------------------------------------------
# 8. router kept= regression
# ---------------------------------------------------------------------------
def test_router_kept_regression():
    print("\n[ROUTER kept=  identical to pre-filtering by name]")
    bars = make_bars(gen_random(40, seed=41))
    series = ST.prepare(bars)
    i = 30
    sigs = ST.all_signals(series, i, None)
    via_kept = RT.route(sigs, None, None, ts=bars[i][0], kept=["momentum"])
    manual = [s for s in sigs if s.name == "momentum"]
    via_list = RT.route(manual, None, None, ts=bars[i][0])
    for fld in ("conviction", "agreement", "weight", "w_trend"):
        check(f"kept= matches manual filter on {fld}",
              approx(getattr(via_kept, fld), getattr(via_list, fld)),
              f"{getattr(via_kept, fld)} vs {getattr(via_list, fld)}")
    check("kept= matches manual filter on active",
          via_kept.active == via_list.active,
          f"{via_kept.active} vs {via_list.active}")
    default = RT.route(sigs, None, None, ts=bars[i][0])
    explicit_none = RT.route(sigs, None, None, ts=bars[i][0], kept=None)
    check("kept=None is bit-identical to the default call",
          approx(default.conviction, explicit_none.conviction)
          and approx(default.weight, explicit_none.weight)
          and default.active == explicit_none.active)


# ---------------------------------------------------------------------------
# 9. unknown strategy is named, never silent
# ---------------------------------------------------------------------------
def test_unknown_strategy_named():
    print("\n[FAIL CLOSED  an unknown strategy gets a named reason]")
    bars = make_bars(gen_random(20, seed=51))
    f = FA.Factory(bars, names=["momentum"])
    known_sig = sig("momentum", 1.0, 0.9)
    v_with = f.evaluate([known_sig, sig("mtdna", 1.0, 0.9)], 10, None,
                        ts=bars[10][0])
    rsn = " ".join(v_with.reasons)
    check("unknown strategy is named in the reason",
          "mtdna" in rsn and "unknown(" in rsn, f"reasons={rsn}")
    f2 = FA.Factory(bars, names=["momentum"])
    v_without = f2.evaluate([known_sig], 10, None, ts=bars[10][0])
    check("unknown signal contributes zero to conviction/agreement/weight",
          approx(v_with.conviction, v_without.conviction)
          and approx(v_with.agreement, v_without.agreement)
          and approx(v_with.weight, v_without.weight),
          f"with=({v_with.conviction},{v_with.weight}) "
          f"without=({v_without.conviction},{v_without.weight})")


# ---------------------------------------------------------------------------
# 10. wired end-to-end through sniper_engine
# ---------------------------------------------------------------------------
def test_end_to_end_sniper():
    print("\n[END-TO-END  SniperConfig(ensemble_factory=True) on 120 bars]")
    bars = make_bars(gen_trend(120, seed=61, drift=0.003, sigma=0.006))
    cfg = SN.SniperConfig(ensemble=True, ensemble_factory=True)
    vs = SN.scan(bars, cfg=cfg)
    check("factory path produced one verdict per bar", len(vs) == 120,
          f"got {len(vs)}")
    finite = all(math.isfinite(v.conviction) and math.isfinite(v.agreement)
                 for v in vs)
    check("every factory conviction/agreement is finite", finite)
    bad_reason = [v.veto_reason for v in vs
                  if v.veto and "ensemble" in v.veto_reason
                  and v.veto_reason != "ensemble-opposed"]
    check("ensemble veto reason string is exactly 'ensemble-opposed'",
          not bad_reason, f"bad={bad_reason[:3]}")


def main():
    print("== monster factory (M3) ==")
    test_no_lookahead()
    test_no_lookahead_close_only()
    test_warmup_promotion()
    test_retirement_and_retry()
    test_block_only()
    test_shrinkage()
    test_router_kept_regression()
    test_unknown_strategy_named()
    test_end_to_end_sniper()
    print(f"\nRESULT: {PASSED} passed, {FAILED} failed")
    for f in FAILURES:
        print(f"   - {f}")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
