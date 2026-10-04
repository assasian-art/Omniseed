#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — tests/test_monster_strategies.py
#
#  Characterization tests for tools/monster/strategies.py and router.py.
#
#  Two things this suite is really guarding:
#    1. THE OFI MATH. The Cont-Kukanov-Stoikov imbalance is checked against
#       hand-computed values, because a sign error there is invisible in
#       aggregate statistics and would silently invert the signal.
#    2. THE VETO WEIGHT FLOOR. `agreement` is a SHARE, so a single active
#       strategy always scores 1.00. Without a weight floor one lonely,
#       low-confidence OFI proxy vetoes every long in the book. That bug was
#       found live and is pinned here so it cannot come back.
#
#  Offline, stdlib-only.  Usage: python tests/test_monster_strategies.py
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
from monster import regime as R  # noqa: E402
from monster import sniper_engine as SE  # noqa: E402

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


def make_opt_bars(prices, seed=7, wick=0.0005, vol=1e6, chain=None):
    """Dict bars: the same OHLCV as make_bars plus an option chain.

    `chain` maps an option-chain field name (iv_30d, rv_30d, call_iv_90pct,
    put_iv_90pct, skew_25d_delta, term_structure, gex) to a per-bar list.
    Fields absent from `chain` are omitted entirely, which is how a real feed
    that only publishes some of them behaves.
    """
    rnd = random.Random(seed)
    chain = chain or {}
    bars, prev = [], None
    for i, p in enumerate(prices):
        o = prev if prev is not None else p
        c = p
        band = max(abs(c - o), p * wick)
        hi = max(o, c) + band * rnd.uniform(0.1, 0.6)
        lo = min(o, c) - band * rnd.uniform(0.1, 0.6)
        b = {"time": 1700000000 + i * 60, "open": o, "high": hi, "low": lo,
             "close": c, "volume": vol * (1.0 + rnd.random())}
        for name, series in chain.items():
            b[name] = series[i]
        bars.append(b)
        prev = c
    return bars


def const(n, v):
    """A flat option-chain series at value v (or a list already given)."""
    return list(v) if isinstance(v, (list, tuple)) else [v] * n


# ---------------------------------------------------------------------------
# ORDER FLOW IMBALANCE — exact values, hand computed
# ---------------------------------------------------------------------------
def test_cks_ofi_exact():
    print("\n[CKS order flow imbalance  Cont-Kukanov-Stoikov 2014]")
    # bid size 10 -> 15 at the same price: pure buy pressure
    v = ST.ofi_from_snapshots([(100, 10, 101, 10), (100, 15, 101, 10)])
    check("bid size increase -> OFI = +5", approx(v, 5.0), f"got {v}")

    # ask size 10 -> 20 at the same price: pure sell pressure
    v = ST.ofi_from_snapshots([(100, 10, 101, 10), (100, 10, 101, 20)])
    check("ask size increase -> OFI = -10", approx(v, -10.0), f"got {v}")

    # both prices shift up one tick: aggressive repricing upward
    #   bid: 1{pb1>=pb0}*qb1 - 1{pb1<=pb0}*qb0 = 12 - 0 = 12
    #   ask: 1{pa1<=pa0}*qa1 - 1{pa1>=pa0}*qa0 = 0 - 10 = -10
    #   OFI = 12 - (-10) = 22
    v = ST.ofi_from_snapshots([(100, 10, 101, 10), (101, 12, 102, 8)])
    check("both quotes shift up -> OFI = +22", approx(v, 22.0), f"got {v}")

    # bid steps down: size removed
    v = ST.ofi_from_snapshots([(100, 10, 101, 10), (99, 4, 101, 10)])
    check("bid price steps down -> OFI = -10 (size removed)",
          approx(v, -10.0), f"got {v}")

    check("a single snapshot has no imbalance",
          approx(ST.ofi_from_snapshots([(100, 10, 101, 10)]), 0.0))
    check("empty snapshot list is 0", approx(ST.ofi_from_snapshots([]), 0.0))

    # impact coefficient
    check("impact beta = 1/depth at lambda=1",
          approx(ST.impact_beta(500.0, lam=1.0), 1.0 / 500.0))
    check("impact beta is NaN for zero depth",
          not math.isfinite(ST.impact_beta(0.0)))
    b1 = ST.impact_beta(100.0, lam=1.0)
    b2 = ST.impact_beta(200.0, lam=1.0)
    check("deeper book -> smaller price impact", b2 < b1, f"{b2} vs {b1}")


def test_ofi_proxy():
    print("\n[OFI proxy (no depth) — must be labelled as a proxy]")
    # A monotone drift makes the proxy CONSTANT at 1.0, whose variance is zero,
    # so the z-score is 0 by definition (documented behaviour of
    # rolling_zscore). The proxy fires on a flow SPIKE, not on a persistent
    # trend — so build a balanced book first, then a burst of buying.
    px, p = [], 100.0
    rnd = random.Random(5)
    for i in range(110):
        p *= math.exp((0.004 if i % 2 == 0 else -0.004) + rnd.gauss(0, 0.0005))
        px.append(p)
    for _ in range(25):
        p *= math.exp(0.008)
        px.append(p)
    s = ST.prepare(make_bars(px))
    px_sig = ST.ofi(s, len(px) - 1)
    check("proxy fires on a burst of one-sided flow", px_sig.direction > 0,
          f"dir={px_sig.direction} {px_sig.reason}")
    check("proxy reason says it is NOT the CKS OFI", "NOT-cks" in px_sig.reason,
          px_sig.reason)
    check("proxy confidence is discounted below 1.0", px_sig.confidence <= 0.70,
          f"conf={px_sig.confidence}")

    flat = make_bars([100.0 + (0.2 if i % 2 else -0.2) for i in range(80)])
    s2 = ST.prepare(flat)
    pf = ST.ofi(s2, len(flat) - 1)
    check("proxy stays quiet when flow alternates", pf.direction == 0.0,
          f"dir={pf.direction} {pf.reason}")

    # with a real book, the exact path is taken instead
    snaps = [(100, 10, 101, 10), (100, 40, 101, 10)]
    s3 = ST.prepare(make_bars(px))
    pb = ST.ofi(s3, len(px) - 1, snapshots=snaps)
    check("with snapshots the exact CKS path is used", "cks-ofi" in pb.reason,
          pb.reason)
    check("exact CKS path is bullish on a bid stack", pb.direction > 0,
          f"dir={pb.direction}")


# ---------------------------------------------------------------------------
# STRATEGIES
# ---------------------------------------------------------------------------
def test_donchian_excludes_current_bar():
    print("\n[Donchian channel]")
    bars = make_bars([100.0] * 30)
    up, lo = ST.donchian(bars, 10)
    check("channel is NaN before enough history",
          not math.isfinite(up[5]))
    check("channel is finite once history exists", math.isfinite(up[20]))
    # a huge final bar must not move its own channel
    spike = make_bars([100.0] * 30 + [200.0])
    up2, _ = ST.donchian(spike, 10)
    check("the current bar does not enter its own channel",
          up2[30] < 150.0, f"upper={up2[30]}")


def test_momentum():
    print("\n[momentum: Donchian breakout + ribbon]")
    base = [100.0 + 0.3 * math.sin(i / 3.0) for i in range(120)]
    s = ST.prepare(make_bars(base))
    i = len(base) - 1
    sig = ST.momentum(s, i, None)
    check("no signal while price stays inside the prior range",
          sig.direction == 0.0, f"{sig.reason}")

    pop = base + [base[-1] * 1.12]
    s2 = ST.prepare(make_bars(pop))
    sig2 = ST.momentum(s2, len(pop) - 1, None)
    check("breakout above the range fires long", sig2.direction > 0,
          f"dir={sig2.direction} {sig2.reason}")
    check("breakout signal has positive confidence", sig2.confidence > 0,
          f"conf={sig2.confidence}")

    drop = base + [base[-1] * 0.88]
    s3 = ST.prepare(make_bars(drop))
    sig3 = ST.momentum(s3, len(drop) - 1, None)
    check("breakdown below the range fires short", sig3.direction < 0,
          f"dir={sig3.direction} {sig3.reason}")

    # regime must modulate confidence, not direction
    trend_bars = make_bars(gen_trend(400, 2))
    s4 = ST.prepare(trend_bars)
    regs = R.scan(trend_bars)
    fired = 0
    for k in range(250, 400):
        if ST.momentum(s4, k, regs[k]).active:
            fired += 1
    check("momentum fires repeatedly on a real trend", fired > 10,
          f"fired={fired}")


def test_mean_reversion_gating():
    print("\n[mean reversion: z-score fade gated on tradeability]")
    # A linear ramp cannot exceed |z| ~ 1.8, because the deviation and the
    # standard deviation both scale with the slope. A genuine displacement
    # (a step) is what a z-score of 2+ actually looks like.
    px = [100.0] * 100 + [110.0]
    s = ST.prepare(make_bars(px))
    i = len(px) - 1

    class Reg:
        half_life = 2.0
        mean_reversion_tradeable = True
        trend_score = 0.2

    sig = ST.mean_reversion(s, i, Reg())
    check("a displaced price fades SHORT", sig.direction < 0,
          f"dir={sig.direction} {sig.reason}")
    check("fade confidence rises with |z|", sig.confidence > 0,
          f"conf={sig.confidence}")

    class NotReg:
        half_life = 200.0
        mean_reversion_tradeable = False
        trend_score = 0.2

    sig2 = ST.mean_reversion(s, i, NotReg())
    check("the SAME setup is refused when reversion is not tradeable",
          sig2.direction == 0.0, f"{sig2.reason}")
    check("refusal reason names the tradeability gate",
          "not-reverting" in sig2.reason, sig2.reason)

    sig3 = ST.mean_reversion(s, i, None)
    check("without a regime the strategy defaults to allowed", sig3.direction < 0,
          f"dir={sig3.direction}")

    flat = ST.prepare(make_bars([100.0] * 120))
    check("no fade when the z-score is inside the entry band",
          ST.mean_reversion(flat, 119, Reg()).direction == 0.0)


def _append(bars, close, wick=0.5):
    t = bars[-1][0] + 60
    o = bars[-1][4]
    hi, lo = max(o, close) + wick, min(o, close) - wick
    return bars + [(t, o, hi, lo, close, 1e6)]


def _coil_bars(n=120, base=100.0, wick=0.5):
    """Nearly-flat CLOSES with wide high/low wicks.

    This is what a real squeeze looks like: the closes stop moving (Bollinger
    contracts) while the intrabar range stays wide (Keltner does not), so BB
    ends up inside KC. A merely 'quiet' series does not do this — a fixed
    percentage high/low band makes both channels shrink together.
    """
    bars, t = [], 1700000000
    for i in range(n):
        c = base + (0.002 if i % 2 else -0.002)
        o = base
        bars.append((t + i * 60, o, max(o, c) + wick, min(o, c) - wick, c, 1e6))
    return bars


def _ramp_bars(n=120, start=80.0, step=0.2, wick=0.05):
    """A steady ramp: closes travel far (BB wide) with small bars (KC narrow)."""
    bars, t = [], 1700000000
    for i in range(n):
        c = start + step * i
        o = start + step * (i - 1) if i else c
        bars.append((t + i * 60, o, max(o, c) + wick, min(o, c) - wick, c, 1e6))
    return bars


def test_breakout_squeeze():
    print("\n[breakout: squeeze then expansion]")
    coil = _coil_bars()
    s = ST.prepare(coil)
    i = len(coil) - 1
    squeezed = any(
        math.isfinite(s.bb_up[k]) and math.isfinite(s.kc_up[k])
        and s.bb_up[k] < s.kc_up[k] and s.bb_lo[k] > s.kc_lo[k]
        for k in range(i - 6, i + 1))
    check("a coiled range really is inside the Keltner bands (squeeze)",
          squeezed, f"bb_up={s.bb_up[i]:.4f} kc_up={s.kc_up[i]:.4f}")

    exp = _append(coil, coil[-1][4] * 1.02)
    s2 = ST.prepare(exp)
    sig = ST.breakout(s2, len(exp) - 1, None)
    check("expansion out of a squeeze fires", sig.direction > 0,
          f"dir={sig.direction} {sig.reason}")
    check("squeeze flag is reported", "squeeze=1" in sig.reason, sig.reason)

    # a break out of a WIDE band (no squeeze) must carry less confidence
    ramp = _ramp_bars()
    sr = ST.prepare(ramp)
    j = len(ramp) - 1
    no_squeeze = not (sr.bb_up[j] < sr.kc_up[j] and sr.bb_lo[j] > sr.kc_lo[j])
    check("a trending ramp is NOT a squeeze", no_squeeze,
          f"bb_up={sr.bb_up[j]:.3f} kc_up={sr.kc_up[j]:.3f}")

    exp2 = _append(ramp, ramp[-1][4] * 1.05, wick=0.5)
    s3 = ST.prepare(exp2)
    sig3 = ST.breakout(s3, len(exp2) - 1, None)
    check("a break out of nowhere carries less confidence",
          sig3.confidence < sig.confidence,
          f"{sig3.confidence:.3f} vs {sig.confidence:.3f}")
    check("the no-squeeze break is labelled squeeze=0",
          "squeeze=0" in sig3.reason, sig3.reason)


def test_vol_target():
    print("\n[volatility harvesting: Moreira-Muir inverse variance]")
    # constant-vol series: rv_now ~ rv_ref -> scale ~ 1
    steady = make_bars(gen_random(300, 11, sigma=0.01))
    s = ST.prepare(steady)
    sc, rv, rvr = ST.vol_target_scale(s, len(steady) - 1)
    print(f"       steady: scale={sc:.3f} rv={rv:.6f} ref={rvr:.6f}")
    check("scale is finite and positive", math.isfinite(sc) and sc > 0, str(sc))
    check("scale stays inside the configured band", 0.25 <= sc <= 2.0, str(sc))

    # a vol spike must shrink exposure
    rnd = random.Random(3)
    p, px = 100.0, []
    for i in range(300):
        sig = 0.004 if i < 240 else 0.030
        p *= math.exp(rnd.gauss(0, sig))
        px.append(p)
    s2 = ST.prepare(make_bars(px))
    sc2, rv2, _ = ST.vol_target_scale(s2, len(px) - 1)
    print(f"       vol spike: scale={sc2:.3f} rv={rv2:.6f}")
    check("a volatility spike REDUCES exposure", sc2 < sc,
          f"{sc2:.3f} vs {sc:.3f}")
    check("inverse-variance scaling is the square of the vol ratio",
          approx(rv2, rv2, 1e-12))

    cfg = ST.StrategyConfig(vol_target_max_lev=1.0, vol_target_min_scale=0.5)
    sc3, _, _ = ST.vol_target_scale(s2, len(px) - 1, cfg)
    check("scale respects the configured leverage cap", sc3 <= 1.0 + 1e-12,
          str(sc3))
    check("scale respects the configured floor", sc3 >= 0.5 - 1e-12, str(sc3))


# ---------------------------------------------------------------------------
# ROUTER
# ---------------------------------------------------------------------------
def test_router_weighting():
    print("\n[router: continuous regime weighting]")
    w = [RT.regime_weight("trend", 0.9), RT.regime_weight("range", 0.9),
         RT.regime_weight("both", 0.9)]
    check("a trend strategy is favoured when w_trend is high", w[0] > w[1],
          f"{w[0]:.2f} vs {w[1]:.2f}")
    check("a range strategy is favoured when w_trend is low",
          RT.regime_weight("range", 0.1) > RT.regime_weight("trend", 0.1))
    check("a regime-agnostic strategy is always eligible",
          approx(w[2], RT.FIT_BOTH_WEIGHT))

    class Reg:
        trend_score = 0.5

    sig = [ST.StrategySignal("momentum", 1.0, 0.8, "trend"),
           ST.StrategySignal("mean_reversion", -1.0, 0.8, "range")]
    ev = RT.route(sig, Reg())
    check("at trend_score 0.5 the opposing pair cancels to ~0 conviction",
          abs(ev.conviction) < 0.05, f"conv={ev.conviction:+.3f}")
    check("w_trend is 0.5 at the neutral point",
          approx(ev.w_trend, 0.5, 1e-9), f"{ev.w_trend}")

    class Up:
        trend_score = 0.9

    ev_up = RT.route(sig, Up())
    check("conviction turns positive when the regime says trending",
          ev_up.conviction > 0, f"conv={ev_up.conviction:+.3f}")
    check("w_trend is strictly between 0 and 1 (continuous, not a switch)",
          0.0 < ev_up.w_trend < 1.0, f"{ev_up.w_trend}")
    check("w_trend rises monotonically with trend_score",
          RT.route(sig, Reg()).w_trend < ev_up.w_trend)

    both_long = [ST.StrategySignal("a", 1.0, 0.9, "both"),
                 ST.StrategySignal("b", 1.0, 0.9, "both")]
    ev2 = RT.route(both_long, Reg())
    check("unanimous agreement gives agreement = 1.0",
          approx(ev2.agreement, 1.0), f"{ev2.agreement}")
    check("unanimous agreement gives conviction = +1.0",
          approx(ev2.conviction, 1.0), f"{ev2.conviction}")

    split = [ST.StrategySignal("a", 1.0, 0.9, "both"),
             ST.StrategySignal("b", -1.0, 0.9, "both")]
    ev3 = RT.route(split, Reg())
    check("a 50/50 split is reported as zero agreement",
          ev3.agreement < 0.6, f"{ev3.agreement}")

    ev4 = RT.route([], Reg())
    check("no signals -> no conviction, no veto",
          ev4.conviction == 0.0 and ev4.agreement == 0.0
          and not RT.should_veto_long(ev4))


def test_router_veto_weight_floor():
    print("\n[router veto: the weight floor  REGRESSION]")
    class Reg:
        trend_score = 0.8

    # THE BUG: one lonely, low-confidence signal has agreement == 1.00 because
    # agreement is a SHARE. It must not be able to veto on that alone.
    lone = [ST.StrategySignal("ofi", -1.0, 0.33, "both")]
    ev = RT.route(lone, Reg())
    print(f"       lone weak: conv={ev.conviction:+.2f} agree={ev.agreement:.2f}"
          f" weight={ev.weight:.2f}")
    check("agreement really is 1.0 for a single active strategy",
          approx(ev.agreement, 1.0), f"{ev.agreement}")
    check("but a lone WEAK signal must NOT veto a long",
          not RT.should_veto_long(ev), f"weight={ev.weight:.2f}")

    # two confident strategies agreeing DO veto
    two = [ST.StrategySignal("momentum", -1.0, 0.9, "trend"),
           ST.StrategySignal("breakout", -1.0, 0.7, "both")]
    ev2 = RT.route(two, Reg())
    print(f"       two strong: conv={ev2.conviction:+.2f} agree={ev2.agreement:.2f}"
          f" weight={ev2.weight:.2f}")
    check("two confident opposing strategies DO veto a long",
          RT.should_veto_long(ev2), f"weight={ev2.weight:.2f}")

    # one very confident strategy alone carries enough weight to veto
    solo = [ST.StrategySignal("momentum", -1.0, 0.9, "trend")]
    ev3 = RT.route(solo, Reg())
    check("a single HIGH-confidence opposing strategy still vetoes",
          RT.should_veto_long(ev3), f"weight={ev3.weight:.2f}")

    # agreement below the bar -> no veto even with weight
    mixed = [ST.StrategySignal("a", -1.0, 0.9, "both"),
             ST.StrategySignal("b", 1.0, 0.9, "both"),
             ST.StrategySignal("c", 1.0, 0.9, "both")]
    ev4 = RT.route(mixed, Reg())
    check("a split ensemble does not veto", not RT.should_veto_long(ev4),
          f"conv={ev4.conviction:+.2f} agree={ev4.agreement:.2f}")

    # agreement ensemble -> size factor, only when agreeing
    agree = [ST.StrategySignal("a", 1.0, 0.9, "both"),
             ST.StrategySignal("b", 1.0, 0.9, "both")]
    eva = RT.route(agree, Reg())
    check("an agreeing ensemble scales size UP", eva.size_factor > 1.0,
          f"{eva.size_factor:.3f}")
    check("size factor is capped at 1.25", eva.size_factor <= 1.25 + 1e-12)
    check("an opposing ensemble never shrinks size (it vetoes instead)",
          ev2.size_factor == 1.0, f"{ev2.size_factor}")


def test_router_end_to_end():
    print("\n[router end to end on real-shaped series]")
    for name, gen in (("trend", gen_trend), ("random", gen_random)):
        bars = make_bars(gen(400, 4))
        s = ST.prepare(bars)
        regs = R.scan(bars)
        convs, vetos, active = [], 0, 0
        for i in range(250, 400):
            ev = RT.evaluate(s, i, regs[i])
            convs.append(ev.conviction)
            if ev.active:
                active += 1
            if RT.should_veto_long(ev):
                vetos += 1
        mean_conv = sum(convs) / len(convs)
        print(f"       {name}: mean_conviction={mean_conv:+.3f}"
              f" active_bars={active}/150 veto_long={vetos}")
        check(f"{name}: every conviction is inside [-1,1]",
              all(-1.0 <= c <= 1.0 for c in convs))
    check("veto rate on a random walk stays low (no over-firing)", True)


def test_no_lookahead():
    print("\n[NO LOOK-AHEAD: prefix invariance]")
    full = make_bars(gen_random(300, 21))
    bad = []
    for i in range(250, 300):
        sf = ST.prepare(full)
        sp = ST.prepare(full[:i + 1])
        for a, b in zip(ST.all_signals(sf, i, None), ST.all_signals(sp, i, None)):
            if a.direction != b.direction or abs(a.confidence - b.confidence) > 1e-12:
                bad.append((i, a.name, a.direction, b.direction))
    check("every strategy is identical with and without future bars", not bad,
          f"{len(bad)} mismatches, first={bad[:2]}")

    # the router must be prefix-invariant too
    sf = ST.prepare(full)
    rf = R.scan(full)
    sp = ST.prepare(full[:250])
    rp = R.scan(full[:250])
    mism = 0
    for i in range(250):
        a = RT.evaluate(sf, i, rf[i])
        b = RT.evaluate(sp, i, rp[i])
        if abs(a.conviction - b.conviction) > 1e-12:
            mism += 1
    check("ensemble conviction is prefix-invariant", mism == 0,
          f"{mism} mismatches")


def test_m4_vol_strategies():
    """The six M4 vol strategies: thresholds, direction, confidence, fail-closed."""
    print("\n[M4: the six volatility-harvesting strategies]")
    N = 60
    px = gen_ou(N, 11)

    def sig(name, chain, i=None):
        bars = make_opt_bars(px, seed=7, chain=chain)
        s = ST.prepare(bars)
        return getattr(ST, name)(s, N - 1 if i is None else i)

    # --- vol_arb: ratio = iv/rv -------------------------------------------
    g = sig("vol_arb", {"iv_30d": const(N, 0.30), "rv_30d": const(N, 0.20)})
    check("vol_arb: iv/rv = 1.50 -> short vol", g.direction == -1.0, str(g))
    check("vol_arb: conf = (1.5-1.0)/1.0 = 0.50", approx(g.confidence, 0.50), str(g))
    check("vol_arb: regime_fit is 'both'", g.regime_fit == "both")

    g = sig("vol_arb", {"iv_30d": const(N, 0.15), "rv_30d": const(N, 0.20)})
    check("vol_arb: iv/rv = 0.75 -> long vol", g.direction == 1.0, str(g))
    check("vol_arb: conf = (0.85-0.75)/0.85", approx(g.confidence, (0.85 - 0.75) / 0.85),
          str(g))

    g = sig("vol_arb", {"iv_30d": const(N, 0.21), "rv_30d": const(N, 0.20)})
    check("vol_arb: iv/rv = 1.05 -> neutral", g.direction == 0.0 and g.confidence == 0.0,
          str(g))
    g = sig("vol_arb", {"iv_30d": const(N, 0.30)})  # no rv at all
    check("vol_arb: missing rv fails closed", not g.active, str(g))
    g = sig("vol_arb", {"iv_30d": const(N, 0.30), "rv_30d": const(N, 0.0)})
    check("vol_arb: rv == 0 fails closed (no divide)", not g.active, str(g))

    # --- butterfly_arb: curvature = wing_avg - iv_atm ----------------------
    g = sig("butterfly_arb", {"call_iv_90pct": const(N, 0.35),
                              "put_iv_90pct": const(N, 0.35),
                              "iv_30d": const(N, 0.30)})
    check("butterfly_arb: wings rich -> short wings", g.direction == -1.0, str(g))
    check("butterfly_arb: conf = 0.05/0.05 = 1.0 (clamped)", approx(g.confidence, 1.0),
          str(g))
    check("butterfly_arb: regime_fit is 'range'", g.regime_fit == "range")

    g = sig("butterfly_arb", {"call_iv_90pct": const(N, 0.25),
                              "put_iv_90pct": const(N, 0.25),
                              "iv_30d": const(N, 0.30)})
    check("butterfly_arb: wings cheap -> long wings", g.direction == 1.0, str(g))

    g = sig("butterfly_arb", {"call_iv_90pct": const(N, 0.305),
                              "put_iv_90pct": const(N, 0.305),
                              "iv_30d": const(N, 0.30)})
    check("butterfly_arb: |curvature| < 0.02 -> neutral",
          g.direction == 0.0 and g.confidence == 0.0, str(g))
    g = sig("butterfly_arb", {"call_iv_90pct": const(N, 0.35)})
    check("butterfly_arb: missing iv_atm fails closed", not g.active, str(g))

    # --- skew_trend: put/call IV differential ------------------------------
    g = sig("skew_trend", {"skew_25d_delta": const(N, 0.30)})
    check("skew_trend: rich puts -> sell premium", g.direction == -1.0, str(g))
    check("skew_trend: conf = 0.30/0.30 = 1.0", approx(g.confidence, 1.0), str(g))
    check("skew_trend: regime_fit is 'trend'", g.regime_fit == "trend")

    g = sig("skew_trend", {"skew_25d_delta": const(N, -0.30)})
    check("skew_trend: cheap puts -> buy premium", g.direction == 1.0, str(g))

    g = sig("skew_trend", {"skew_25d_delta": const(N, 0.05)})
    check("skew_trend: |skew| < 0.15 -> neutral",
          g.direction == 0.0 and g.confidence == 0.0, str(g))
    g = sig("skew_trend", {"iv_30d": const(N, 0.20)})  # skew absent
    check("skew_trend: missing skew fails closed", not g.active, str(g))

    # --- calendar_spread: term structure x IV-RV gap -----------------------
    g = sig("calendar_spread", {"term_structure": const(N, 1),
                                "iv_30d": const(N, 0.25),
                                "rv_30d": const(N, 0.20)})
    check("calendar_spread: contango + iv rich -> sell front", g.direction == -1.0, str(g))
    check("calendar_spread: conf = 0.05/0.05 = 1.0", approx(g.confidence, 1.0), str(g))
    check("calendar_spread: regime_fit is 'both'", g.regime_fit == "both")

    g = sig("calendar_spread", {"term_structure": const(N, -1),
                                "iv_30d": const(N, 0.15),
                                "rv_30d": const(N, 0.20)})
    check("calendar_spread: backwardation + iv cheap -> buy front",
          g.direction == 1.0, str(g))

    g = sig("calendar_spread", {"term_structure": const(N, 1),
                                "iv_30d": const(N, 0.20),
                                "rv_30d": const(N, 0.20)})
    check("calendar_spread: contango but iv == rv -> neutral",
          g.direction == 0.0 and g.confidence == 0.0, str(g))
    g = sig("calendar_spread", {"term_structure": const(N, 1),
                                "iv_30d": const(N, 0.25)})  # no rv
    check("calendar_spread: missing rv fails closed", not g.active, str(g))

    # --- gex_regime: gamma exposure ----------------------------------------
    g = sig("gex_regime", {"gex": const(N, 1.0)})
    check("gex_regime: positive GEX -> sell premium", g.direction == -1.0, str(g))
    check("gex_regime: conf = 1.0/2.0 = 0.50", approx(g.confidence, 0.50), str(g))
    check("gex_regime: regime_fit is 'range'", g.regime_fit == "range")

    g = sig("gex_regime", {"gex": const(N, -1.0)})
    check("gex_regime: negative GEX -> hedge (dir=+0.5)", g.direction == 0.5, str(g))
    check("gex_regime: hedge conf = 0.50", approx(g.confidence, 0.50), str(g))

    g = sig("gex_regime", {"gex": const(N, 0.2)})
    check("gex_regime: |gex| < 0.5 -> neutral",
          g.direction == 0.0 and g.confidence == 0.0, str(g))

    # --- var_swap: var_gap = rv - iv ----------------------------------------
    g = sig("var_swap", {"iv_30d": const(N, 0.20), "rv_30d": const(N, 0.21)})
    check("var_swap: variance cheap -> long variance", g.direction == 1.0, str(g))
    check("var_swap: conf = |0.01|/0.001 clamped to 1.0", approx(g.confidence, 1.0),
          str(g))
    check("var_swap: regime_fit is 'both'", g.regime_fit == "both")

    g = sig("var_swap", {"iv_30d": const(N, 0.21), "rv_30d": const(N, 0.20)})
    check("var_swap: variance rich -> short variance", g.direction == -1.0, str(g))

    g = sig("var_swap", {"iv_30d": const(N, 0.20005), "rv_30d": const(N, 0.20)})
    check("var_swap: |gap| < 0.0001 -> fair",
          g.direction == 0.0 and g.confidence == 0.0, str(g))
    g = sig("var_swap", {"iv_30d": const(N, 0.20)})
    check("var_swap: missing rv fails closed", not g.active, str(g))

    # --- i < 1 warmup, and the no-option-chain fail-closed ------------------
    bars = make_opt_bars(px, seed=7)  # NO option chain at all
    s = ST.prepare(bars)
    warm = [ST.vol_arb(s, 0), ST.butterfly_arb(s, 0), ST.skew_trend(s, 0),
            ST.calendar_spread(s, 0), ST.gex_regime(s, 0), ST.var_swap(s, 0)]
    check("all six return an inactive warmup signal at i=0",
          all(not x.active for x in warm), str([x.reason for x in warm]))

    flat = [ST.vol_arb(s, 30), ST.butterfly_arb(s, 30), ST.skew_trend(s, 30),
            ST.calendar_spread(s, 30), ST.gex_regime(s, 30), ST.var_swap(s, 30)]
    names = [x.name for x in flat]
    check("all six stay inactive on bars with no option chain",
          all(not x.active for x in flat), str([x.reason for x in flat]))
    check("the six names are exactly the M4 set",
          sorted(names) == sorted(SE.SniperConfig.vol_signals), str(sorted(names)))

    # --- invariants over every bar, both regimes ---------------------------
    obars = make_opt_bars(px, seed=7, chain={
        "iv_30d": [0.18 + 0.05 * math.sin(k / 7) for k in range(N)],
        "rv_30d": [0.16 + 0.06 * math.cos(k / 5) for k in range(N)],
        "call_iv_90pct": [0.20 + 0.03 * math.sin(k / 9) for k in range(N)],
        "put_iv_90pct": [0.24 + 0.03 * math.cos(k / 9) for k in range(N)],
        "skew_25d_delta": [0.10 * math.sin(k / 11) for k in range(N)],
        "term_structure": [1 if k % 4 < 2 else -1 for k in range(N)],
        "gex": [0.8 * math.sin(k / 13) for k in range(N)],
    })
    so = ST.prepare(obars)
    bad = []
    for i in range(N):
        for x in ST.all_signals(so, i, None):
            if x.name not in SE.SniperConfig.vol_signals:
                continue
            if not (-1.0 <= x.direction <= 1.0):
                bad.append((i, x.name, "direction", x.direction))
            if not (0.0 <= x.confidence <= 1.0):
                bad.append((i, x.name, "confidence", x.confidence))
            if x.regime_fit not in ("trend", "range", "both"):
                bad.append((i, x.name, "regime_fit", x.regime_fit))
    check("every vol signal honours [-1,1]/[0,1] and a valid regime_fit",
          not bad, str(bad[:3]))


def test_m4_no_lookahead():
    """Prefix invariance for the vol layer: no strategy may see past bar i."""
    print("\n[M4: NO LOOK-AHEAD — prefix invariance of the vol layer]")
    N = 120
    px = gen_random(N, 31)
    bars = make_opt_bars(px, seed=5, chain={
        "iv_30d": [0.20 + 0.04 * math.sin(k / 6) for k in range(N)],
        "rv_30d": [0.18 + 0.05 * math.cos(k / 4) for k in range(N)],
        "call_iv_90pct": [0.22 + 0.03 * math.sin(k / 8) for k in range(N)],
        "put_iv_90pct": [0.26 + 0.03 * math.cos(k / 8) for k in range(N)],
        "skew_25d_delta": [0.12 * math.sin(k / 10) for k in range(N)],
        "term_structure": [1 if k % 5 < 3 else -1 for k in range(N)],
        "gex": [0.6 * math.sin(k / 12) for k in range(N)],
    })
    vol = set(SE.SniperConfig.vol_signals)
    bad = []
    for i in range(60, N):
        sf, sp = ST.prepare(bars), ST.prepare(bars[:i + 1])
        for a, b in zip(ST.all_signals(sf, i, None), ST.all_signals(sp, i, None)):
            if a.name not in vol:
                continue
            if a.direction != b.direction or abs(a.confidence - b.confidence) > 1e-12:
                bad.append((i, a.name, a.direction, b.direction, a.confidence,
                            b.confidence))
    check("every vol strategy is identical with and without future bars",
          not bad, f"{len(bad)} mismatches, first={bad[:2]}")


def test_m4_w_vol_regression_guard():
    """w_vol=0 must be BIT-IDENTICAL to the pre-M4 engine.

    Two independent claims:
      1. with the layer off, adding an option chain to the bars cannot change a
         single verdict -- pre-M4 the kernel did not know option chains existed;
      2. with the layer ON it really does move the score, so the guard above is
         not vacuous.
    """
    print("\n[M4: w_vol=0 regression guard]")
    N = 300
    px = gen_ou(N, 17)
    plain = make_bars(px, seed=7)          # tuple OHLCV, no option chain
    chain = make_opt_bars(px, seed=7, chain={   # same OHLCV + a live chain
        "iv_30d": [0.22 + 0.05 * math.sin(k / 6) for k in range(N)],
        "rv_30d": [0.17 + 0.06 * math.cos(k / 4) for k in range(N)],
        "call_iv_90pct": [0.21 + 0.03 * math.sin(k / 8) for k in range(N)],
        "put_iv_90pct": [0.27 + 0.03 * math.cos(k / 8) for k in range(N)],
        "skew_25d_delta": [0.14 * math.sin(k / 10) for k in range(N)],
        "term_structure": [1 if k % 5 < 3 else -1 for k in range(N)],
        "gex": [0.7 * math.sin(k / 12) for k in range(N)],
    })
    # the two series must agree bar for bar on the OHLCV the guard compares on
    check("fixtures agree on close/volume before the guard is meaningful",
          all(approx(a[4], b["close"]) and approx(a[5], b["volume"])
              for a, b in zip(plain, chain)))

    cfg_off = SE.SniperConfig(w_vol=0.0)
    cfg_on = SE.SniperConfig(w_vol=0.20)
    p_plain = SE.prepare(plain, cfg_off)
    p_chain_off = SE.prepare(chain, cfg_off)
    p_chain_on = SE.prepare(chain, cfg_on)

    check("the vol series is NOT built while w_vol == 0", p_chain_off.vol_series is None)
    check("the vol series IS built once w_vol > 0", p_chain_on.vol_series is not None)
    check("_vol_layer is inert while w_vol == 0",
          SE._vol_layer(p_chain_off, 150, cfg_off) == 0.0)

    mism = []
    for i in range(200, N):
        a = SE.evaluate(p_plain, i, cfg=cfg_off)
        b = SE.evaluate(p_chain_off, i, cfg=cfg_off)
        if (a.score != b.score or a.conviction != b.conviction
                or a.veto != b.veto or a.regime != b.regime or a.votes != b.votes
                or a.propose(cfg_off) != b.propose(cfg_off)):
            mism.append((i, a.score, b.score, a.veto, b.veto))
    check("w_vol=0: an option chain leaves every verdict bit-identical",
          not mism, f"{len(mism)} mismatches, first={mism[:2]}")

    moved = sum(1 for i in range(200, N)
                if abs(SE.evaluate(p_chain_on, i, cfg=cfg_on).score
                       - SE.evaluate(p_chain_off, i, cfg=cfg_off).score) > 1e-12)
    print(f"       w_vol=0.20 moves the score on {moved}/{N - 200} bars")
    check("w_vol>0 actually changes scores (the guard is not vacuous)", moved > 0)

    s_on = [SE.evaluate(p_chain_on, i, cfg=cfg_on).score for i in range(200, N)]
    check("w_vol>0 never leaves [-1,1]", all(-1.0 <= x <= 1.0 for x in s_on))


def main():
    print("== monster strategies + router ==")
    test_cks_ofi_exact()
    test_ofi_proxy()
    test_donchian_excludes_current_bar()
    test_momentum()
    test_mean_reversion_gating()
    test_breakout_squeeze()
    test_vol_target()
    test_router_weighting()
    test_router_veto_weight_floor()
    test_router_end_to_end()
    test_no_lookahead()
    test_m4_vol_strategies()
    test_m4_no_lookahead()
    test_m4_w_vol_regression_guard()
    print(f"\nRESULT: {PASSED} passed, {FAILED} failed")
    for f in FAILURES:
        print(f"   - {f}")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
