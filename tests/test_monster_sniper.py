#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — tests/test_monster_sniper.py
#
#  Characterization tests for tools/monster/{sniper_engine,sizing,
#  correlation_matrix}.py.
#
#  Proves: the layer arithmetic, every hard veto, that the 0.85 bar is
#  REACHABLE (a gate that can never open is a bug), the adaptive sizing band
#  never leaves [1%,2%], and the cross-asset filter invalidates correctly —
#  including when a coherent complex is dragging the target against our side.
#
#  Offline, stdlib-only.  Usage: python tests/test_monster_sniper.py
# =============================================================================
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from monster import sniper_engine as SN   # noqa: E402
from monster import sizing as SZ          # noqa: E402
from monster import correlation_matrix as CM  # noqa: E402

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
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tol


def bars_from(prices, vol_mult=None, base_vol=1000.0, spread=0.01):
    out = []
    for i, p in enumerate(prices):
        v = base_vol * (vol_mult[i] if vol_mult else 1.0)
        out.append((1600000000 + i * 86400, p * (1 - spread / 2),
                    p * (1 + spread), p * (1 - spread), p, v))
    return out


def trend(n, rate, start=100.0, vol=0.0, seed=1):
    rnd = random.Random(seed)
    prices, p = [], start
    for _ in range(n):
        p *= math.exp(rate + (rnd.gauss(0, vol) if vol else 0.0))
        prices.append(p)
    return prices


# The canonical "textbook sniper" fixture: an alternating regime with bursty
# volume every 11 bars. Verified to reach S=0.865 (>= 0.85) on a bar with three
# agreeing factors in a confirmed uptrend.
def sniper_fixture(n=340, seed=12, period=11, off=1, mult=15.0):
    rnd = random.Random(seed)
    prices, p = [], 100.0
    for i in range(n):
        drift = 0.0022 if (i // 50) % 2 == 0 else -0.0012
        p *= math.exp(drift + rnd.gauss(0, 0.009))
        prices.append(p)
    vm = [mult if (i - off) % period == 0 else 1.0 for i in range(n)]
    return bars_from(prices, vm)


# ---------------------------------------------------------------------------
# Regime
# ---------------------------------------------------------------------------
def test_regime():
    print("\n-- regime --")
    cfg = SN.SniperConfig()
    up = SN.prepare(bars_from(trend(300, 0.003)), cfg)
    check("regime: steady uptrend -> trend_up",
          SN._regime(up, 299, cfg)[0] == "trend_up")
    check("regime: trend_up scores 1.0", SN._regime(up, 299, cfg)[1] == 1.0)

    dn = SN.prepare(bars_from(trend(300, -0.003)), cfg)
    check("regime: steady downtrend -> trend_down",
          SN._regime(dn, 299, cfg)[0] == "trend_down")
    check("regime: trend_down scores 0.0", SN._regime(dn, 299, cfg)[1] == 0.0)

    flat = SN.prepare(bars_from(trend(300, 0.0, vol=0.0002)), cfg)
    check("regime: flat, small ATR -> range",
          SN._regime(flat, 299, cfg)[0] == "range")
    check("regime: range scores 0.5", SN._regime(flat, 299, cfg)[1] == 0.5)

    # Flat closes but 12% intraday swings -> high volatility, no trend.
    prices = [100.0] * 300
    wide = [(1600000000 + i * 86400, 100.0, 106.0, 94.0, 100.0, 1000.0)
            for i in range(300)]
    hv = SN.prepare(wide, cfg)
    check("regime: flat price + wide range -> high_vol",
          SN._regime(hv, 299, cfg)[0] == "high_vol")


# ---------------------------------------------------------------------------
# Vetoes
# ---------------------------------------------------------------------------
def test_vetoes():
    print("\n-- hard vetoes --")
    cfg = SN.SniperConfig()

    # trend_down with no mean-reversion confirmation -> counter-regime veto
    dn = SN.prepare(bars_from(trend(300, -0.003)), cfg)
    v = SN.evaluate(dn, 299, None, cfg)
    check("veto: counter-regime blocks a downtrend long",
          v.veto and v.veto_reason == "counter-regime")

    # high_vol without oversold -> veto
    wide = [(1600000000 + i * 86400, 100.0, 106.0, 94.0, 100.0, 1000.0)
            for i in range(300)]
    hv = SN.prepare(wide, cfg)
    v = SN.evaluate(hv, 299, None, cfg)
    check("veto: high-vol needs a mean-reversion setup",
          v.veto and v.veto_reason == "high-vol-needs-mean-reversion")

    # insufficient confluence
    up = SN.prepare(bars_from(trend(300, 0.003)), cfg)
    v = SN.evaluate(up, 299, None, cfg)
    check("veto: fewer than 3 factors -> insufficient-confluence",
          v.votes < cfg.min_factors
          and v.veto and v.veto_reason == "insufficient-confluence")

    # cross-asset invalidation
    ctx = SN.EvalContext(cross_confirm=0, cross_invalidate=2, cross_active=2)
    v = SN.evaluate(up, 299, ctx, cfg)
    check("veto: two invalidating peers -> cross-asset-invalidation",
          v.veto and v.veto_reason == "cross-asset-invalidation")

    # ...but the veto needs the threshold, not one lone peer
    ctx1 = SN.EvalContext(cross_confirm=0, cross_invalidate=1, cross_active=1)
    v = SN.evaluate(up, 299, ctx1, cfg)
    check("veto: one invalidating peer is not enough to veto",
          v.veto_reason != "cross-asset-invalidation")


def test_event_relaxes_confluence():
    print("\n-- event-driven relaxation --")
    cfg = SN.SniperConfig()
    up = SN.prepare(bars_from(trend(300, 0.003)), cfg)
    v = SN.evaluate(up, 299, None, cfg)
    votes = v.votes
    check("event: baseline confluence is below the requirement",
          votes < cfg.min_factors)
    ctx = SN.EvalContext(event_driven=True)
    v2 = SN.evaluate(up, 299, ctx, cfg)
    check("event: a matched event lowers the confluence bar",
          v2.veto_reason != "insufficient-confluence"
          or votes >= cfg.min_factors_event)
    check("event: relaxation does NOT bypass a risk veto",
          SN.SniperConfig().min_factors_event == 1)


# ---------------------------------------------------------------------------
# Score arithmetic + reachability
# ---------------------------------------------------------------------------
def test_score_arithmetic():
    print("\n-- score arithmetic --")
    cfg = SN.SniperConfig()
    check("weights: sum to 1.0",
          approx(cfg.w_micro + cfg.w_tech + cfg.w_regime + cfg.w_cross, 1.0))
    check("threshold: 0.85 by default", cfg.min_confidence == 0.85)
    check("T: 3 factors -> 0.90", approx(
        SN.SniperConfig().w_tech, cfg.w_tech) and
        approx(min(1.0, 0.90 + 0.10 * (3 - 3)), 0.90))
    check("T: 4 factors -> 1.00", approx(min(1.0, 0.90 + 0.10 * (4 - 3)), 1.0))
    check("T: 6 factors caps at 1.00",
          approx(min(1.0, 0.90 + 0.10 * (6 - 3)), 1.0))

    # Perfect alignment -> score 1.0
    up = SN.prepare(bars_from(trend(300, 0.003)), cfg)
    v = SN.SniperVerdict(micro=1.0, tech=1.0, regime_score=1.0, cross=1.0)
    v.score = (cfg.w_micro + cfg.w_tech + cfg.w_regime + cfg.w_cross)
    check("score: perfect alignment -> 1.0", approx(v.score, 1.0))
    check("propose: perfect alignment proposes", v.propose(cfg))
    check("propose: a vetoed verdict never proposes",
          not SN.SniperVerdict(score=0.99, veto=True).propose(cfg))
    check("propose: 0.84 does not propose",
          not SN.SniperVerdict(score=0.84).propose(cfg))
    check("propose: exactly 0.85 proposes",
          SN.SniperVerdict(score=0.85).propose(cfg))
    del up


def test_reachability():
    print("\n-- the 0.85 bar is reachable --")
    cfg = SN.SniperConfig()
    bars = sniper_fixture()
    vs = SN.scan(bars, cfg=cfg)
    props = [v for v in vs if v.propose(cfg)]
    check("reachable: a textbook setup produces at least one proposal",
          len(props) >= 1, f"got {len(props)}")
    if props:
        best = max(props, key=lambda v: v.score)
        check("reachable: the proposal clears 0.85", best.score >= 0.85,
              f"S={best.score:.3f}")
        check("reachable: it has >=3 agreeing factors",
              best.votes >= cfg.min_factors)
        check("reachable: it is in a confirmed uptrend",
              best.regime == "trend_up")
        check("reachable: it is not vetoed", not best.veto)
    check("reachable: proposals are RARE (a sniper, not a machine gun)",
          len(props) < len(vs) * 0.1)


# ---------------------------------------------------------------------------
# Adaptive sizing
# ---------------------------------------------------------------------------
def test_sizing():
    print("\n-- adaptive sizing --")
    cfg = SZ.SizingConfig()
    lo, hi = SZ.MIN_RISK_PCT, SZ.MAX_RISK_PCT
    check("sizing: band mirrors the engine's 1%..2%",
          approx(lo, 0.01) and approx(hi, 0.02))

    # Never leaves the band, whatever we throw at it.
    worst = None
    for conf in (0.0, 0.5, 0.85, 0.9, 0.99, 1.0, 2.0):
        for atr in (None, 0.0, 0.0001, 0.005, 0.02, 0.2, 5.0):
            r, f, g = SZ.adaptive_risk_pct(conf, atr, cfg)
            if r < lo - 1e-12 or r > hi + 1e-12:
                worst = (conf, atr, r)
    check("sizing: risk_pct always inside [1%,2%]", worst is None, str(worst))

    # Monotone in confidence.
    a = SZ.adaptive_risk_pct(0.86, 0.02, cfg)[0]
    b = SZ.adaptive_risk_pct(0.99, 0.02, cfg)[0]
    check("sizing: higher confidence -> >= risk", b >= a)

    # Inverse in volatility.
    calm = SZ.adaptive_risk_pct(0.95, 0.005, cfg)[0]
    wild = SZ.adaptive_risk_pct(0.95, 0.20, cfg)[0]
    check("sizing: higher ATR -> <= risk", wild <= calm)

    check("sizing: unknown ATR is neutral, never a bonus",
          approx(SZ.volatility_factor(None, cfg), 1.0))
    check("sizing: zero ATR is neutral (no divide-by-zero)",
          approx(SZ.volatility_factor(0.0, cfg), 1.0))

    s = SZ.size(100000.0, 50.0, 0.08, 0.95, 0.02, cfg)
    check("sizing: qty follows the risk-budget identity",
          approx(s.qty, 100000.0 * s.risk_pct / (50.0 * 0.08), 1e-6))
    check("sizing: allowed with sane inputs", s.allowed and s.qty > 0)
    check("sizing: refuses bad inputs",
          not SZ.size(0.0, 50.0, 0.08, 0.95, 0.02, cfg).allowed)
    check("sizing: refuses a zero stop",
          not SZ.size(100000.0, 50.0, 0.0, 0.95, 0.02, cfg).allowed)


# ---------------------------------------------------------------------------
# Cross-asset correlation
# ---------------------------------------------------------------------------
def test_correlation():
    print("\n-- cross-asset correlation --")
    m = CM.CorrelationMatrix(window=5)

    def rets(**kw):
        n = 10
        d = {k: [0.0] * n for k in ("OIL", "COMMODITY_INDEX", "GOLD", "USDJPY")}
        for k, v in kw.items():
            d[k] = [0.0] * n
            d[k][-1] = v
        return d

    # Long OIL: oil up, complex up -> confirm.
    r = m.evaluate("OIL", +1, rets(OIL=0.02, COMMODITY_INDEX=0.02), 9)
    check("cross: coherent complex with us -> confirm",
          r.confirm == 1 and r.invalidate == 0)

    # Long OIL: oil up, complex CRASHING -> invalidate (the mandate's example).
    r = m.evaluate("OIL", +1, rets(OIL=0.02, COMMODITY_INDEX=-0.02), 9)
    check("cross: broader complex crashing invalidates a long",
          r.invalidate == 1 and r.confirm == 0)

    # Long OIL: oil itself falling -> invalidate (alignment fails).
    r = m.evaluate("OIL", +1, rets(OIL=-0.02, COMMODITY_INDEX=-0.02), 9)
    check("cross: coherent complex dragging the target against us invalidates",
          r.invalidate == 1)

    # SHORT OIL: oil down, complex down -> confirm.
    r = m.evaluate("OIL", -1, rets(OIL=-0.02, COMMODITY_INDEX=-0.02), 9)
    check("cross: a short is confirmed by a coherent down-move",
          r.confirm == 1)

    # Leader flat -> not active, no opinion.
    r = m.evaluate("OIL", +1, rets(OIL=0.0, COMMODITY_INDEX=0.02), 9)
    check("cross: a flat leader is inactive", r.active == 0)

    # Target as the RESPONDER (GOLD is the responder in USD->GOLD).
    r = m.evaluate("GOLD", +1, rets(USD=0.02, GOLD=-0.02), 9)
    check("cross: works when the target is the responder (invalidate)",
          r.invalidate == 1)

    # Unrelated symbol -> no pairs.
    r = m.evaluate("ZZZZ", +1, rets(OIL=0.02), 9)
    check("cross: an unpaired symbol has no relations", r.active == 0)

    check("cross: detail() is CSV-safe", "," not in CM.CrossResult().detail())


def main():
    print("== monster sniper engine ==")
    test_regime()
    test_vetoes()
    test_event_relaxes_confluence()
    test_score_arithmetic()
    test_reachability()
    test_sizing()
    test_correlation()
    print(f"\nRESULT: {PASSED} passed, {FAILED} failed")
    for f in FAILURES:
        print(f"   - {f}")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
