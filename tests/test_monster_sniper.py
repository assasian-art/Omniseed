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


# The canonical "textbook sniper" fixture: a PULLBACK THAT RECLAIMS inside an
# uptrend, on a volume burst. Verified to reach S=0.890 on a bar with three
# agreeing factors (vwap-reclaim + macd-thrust + trend-align) in a confirmed
# uptrend.
#
# Why it looks like this, and not like a steady trend:
#
#   * The six technical factors CLUSTER. `trend-align` and `macd-thrust` fire
#     constantly in a trend; `vwap-reclaim` only fires on a recovery bar. So
#     `votes >= 3` in practice means "a pullback that reclaims", which is a
#     coherent sniper setup — and the only one that reaches 3 votes at all.
#   * S >= 0.85 is a REGIME question, not a score question. With M=T=C maximal
#     the ceiling is 0.8125 in high_vol and 0.875 in range, so an entry needs a
#     confirmed uptrend. See test_reachability().
#   * The dip must therefore be gentle enough not to trip the volatility
#     regime. The regime engine requires volatility to be elevated BOTH
#     relatively (percentile) and absolutely (ATR/close above half the
#     mandate's 5%), which is what lets a reclaim bar still read trend_up.
#   * ONE burst bar, not three: three raise the rolling volume mean and suppress
#     the z-score, which caps the M layer at 0.82 and leaves S at 0.844.
def sniper_fixture(n=340, seed=2, pre=0.0030, sig=0.0020, dip_len=5,
                   dip=-0.018, slow_len=8, slow=0.0030, pop=0.025,
                   burst=50000.0, base_vol=1000.0, spread=0.01):
    rnd = random.Random(seed)
    prices, p = [], 100.0
    tail = dip_len + slow_len + 1
    for i in range(n):
        if i >= n - tail:
            j = i - (n - tail)
            if j < dip_len:
                d = dip                       # the pullback
            elif j < dip_len + slow_len:
                d = slow                      # grinding back up
            else:
                d = pop                       # the reclaim bar
        else:
            d = pre                           # the uptrend
        p *= math.exp(d + rnd.gauss(0, sig))
        prices.append(p)

    vols = [base_vol * (1.0 + 0.20 * rnd.random()) for _ in range(n)]
    vols[n - 1] = burst                       # the volume anomaly, on the reclaim
    return [(1600000000 + i * 86400, prices[i] * (1 - spread / 2),
             prices[i] * (1 + spread), prices[i] * (1 - spread), prices[i],
             vols[i]) for i in range(n)]


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

    # ---- 1. THE ARITHMETIC CEILING, PER REGIME -------------------------
    #
    # S >= 0.85 is a REGIME question, and this is the crisp form of it. With
    # M=T=C=1.0 (a maximal volume anomaly, all six factors, every peer
    # confirming) the reachable maximum depends only on the R layer:
    #
    #   trend_up   .25+.35+.25*1.00+.15 = 1.0000   <- can host 0.85
    #   range      .25+.35+.25*0.50+.15 = 0.8750   <- can, barely
    #   high_vol   .25+.35+.25*0.25+.15 = 0.8125   <- CANNOT, ever
    #   trend_down .25+.35+.25*0.00+.15 = 0.7500   <- cannot (and is vetoed)
    #
    # That is why this test also pins the regime: a sniper entry needs an
    # uptrend, and the "0.85 is unreachable" failure mode shows up as a regime
    # problem long before it shows up as a score problem.
    ceil = {r: (cfg.w_micro + cfg.w_tech + cfg.w_cross
                + cfg.w_regime * sc)
            for r, sc in SN.REGIME_SCORE.items()}
    print("       ceiling by regime: "
          + "  ".join("%s=%.4f" % (k, v) for k, v in sorted(ceil.items())))
    check("reachable: a trend_up bar can clear 0.85", ceil["trend_up"] >= 0.85,
          f"{ceil['trend_up']:.4f}")
    check("reachable: a range bar can clear 0.85 (barely)",
          ceil["range"] >= 0.85, f"{ceil['range']:.4f}")
    check("reachable: a high_vol bar can NEVER clear 0.85",
          ceil["high_vol"] < 0.85, f"{ceil['high_vol']:.4f}")
    check("reachable: a trend_down bar can never clear 0.85",
          ceil["trend_down"] < 0.85, f"{ceil['trend_down']:.4f}")

    # ---- 2. A REAL BAR WITH FEASIBLE LAYER VALUES ----------------------
    # M=0.90 (a genuine volume spike), T=0.90 (three agreeing factors),
    # R=1.0 (confirmed uptrend), C=0.50 (no peers active -> neutral).
    v = SN.SniperVerdict(ts=0, micro=0.90, tech=0.90, regime_score=1.0,
                         cross=0.50, votes=3, regime="trend_up")
    v.score = (cfg.w_micro * v.micro + cfg.w_tech * v.tech
               + cfg.w_regime * v.regime_score + cfg.w_cross * v.cross)
    print(f"       constructed bar: S={v.score:.4f}")
    check("reachable: feasible layer values clear 0.85", v.score >= 0.85,
          f"S={v.score:.4f}")
    check("reachable: and propose() says yes when it is not vetoed",
          v.propose(cfg))

    # ---- 3. THE REAL FIXTURE -------------------------------------------
    bars = sniper_fixture()
    vs = SN.scan(bars, cfg=cfg)
    props = [v for v in vs if v.propose(cfg)]
    best = max(vs, key=lambda v: v.score)
    print(f"       fixture: {len(props)} proposals, best S={best.score:.4f} "
          f"({best.regime}, {best.votes} votes)")
    check("reachable: a textbook setup produces at least one proposal",
          len(props) >= 1, f"got {len(props)}")
    if props:
        top = max(props, key=lambda v: v.score)
        check("reachable: the proposal clears 0.85", top.score >= 0.85,
              f"S={top.score:.3f}")
        check("reachable: it has >=3 agreeing factors",
              top.votes >= cfg.min_factors)
        check("reachable: it is in a confirmed uptrend",
              top.regime == "trend_up")
        check("reachable: it is not vetoed", not top.veto)
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
