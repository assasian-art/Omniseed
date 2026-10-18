#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — tests/test_monster_exotics.py  (Phase 7)
#
#  Characterization tests for tools/monster/exotics.py.
#
#  Three properties carry the weight, per house rules:
#    CALIBRATION AGAINST KNOWN GROUND TRUTH (DFA on synthetic
#    persistent/anti-persistent/random series; OU half-life recovery; the
#    F-distribution p-value against published table values),
#    FAIL-CLOSED visibility (insufficient data says so), and
#    NO LOOK-AHEAD (prefix invariance everywhere; causal market_state).
#
#  Offline, stdlib-only.  Usage: python tests/test_monster_exotics.py
# =============================================================================
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from monster import exotics as EX  # noqa: E402

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


def lcg_series(n, seed=42, smooth=0):
    """Deterministic pseudo-noise; smooth>0 = low-pass (persistence-ish)."""
    rng = EX.Rng(seed)
    out = []
    prev = 0.0
    for _ in range(n):
        v = rng.gauss()
        if smooth:
            prev = (smooth * prev + v) / (smooth + 1.0)
            v = prev
        out.append(v)
    return out


def test_rng():
    print("\n[determinism]")
    a = [EX.Rng(7).next() for _ in range(1)]
    r1, r2 = EX.Rng(99), EX.Rng(99)
    same = all(r1.next() == r2.next() for _ in range(50))
    check("same seed -> bit-identical stream", same)


def test_f_distribution():
    print("\n[F distribution vs published tables]")
    # T093: the median reference 0.986 was WRONG (it is F_{10,1}, not
    # F_{1,10}). Since F_{1,v} = t_v^2, the median of F_{1,10} is
    # (t_{10,0.75})^2 = 0.7002^2 = 0.4903. betai is correct — the
    # 5% critical check below (4.965 -> 0.05) pins it to the table.
    p_med = EX.f_cdf_p(0.4903, 1, 10)
    check("P(F_{1,10}>0.4903) ~ 0.5 (median)", abs(p_med - 0.5) < 0.02,
          "%.4f" % p_med)
    p_crit = EX.f_cdf_p(4.965, 1, 10)
    check("P(F_{1,10}>4.965) ~ 0.05 (critical)", abs(p_crit - 0.05) < 0.01,
          "%.4f" % p_crit)
    p_big = EX.f_cdf_p(100.0, 2, 20)
    check("a huge F has a tiny p", p_big < 0.001, "%.6f" % p_big)
    check("I_x edges are exact", EX.betai(2, 3, 0.0) == 0.0
          and EX.betai(2, 3, 1.0) == 1.0)


def test_granger():
    print("\n[Granger causality]")
    n = 300
    rng = EX.Rng(5)
    x = [rng.gauss() for _ in range(n)]
    # y depends on its own past AND x's past -> x Granger-causes y
    y1 = []
    for t in range(n):
        v = 0.4 * (y1[-1] if y1 else 0.0) + 0.8 * (x[t - 1] if t else 0.0)
        v += 0.1 * rng.gauss()
        y1.append(v)
    g = EX.granger_causality(x, y1, n - 1, lags=2)
    check("x -> y (constructed) is detected", g["verdict"] == "granger-causes",
          "%s f=%.2f p=%.4f" % (g["verdict"], g["f"], g["p"]))
    # independent y: x must NOT be declared a cause (fail-closed to no)
    y2 = []
    for t in range(n):
        v = 0.4 * (y2[-1] if y2 else 0.0) + 0.1 * rng.gauss()
        y2.append(v)
    g2 = EX.granger_causality(x, y2, n - 1, lags=2)
    check("independent x -/-> y (no-causality)",
          g2["verdict"] == "no-causality",
          "%s p=%.4f" % (g2["verdict"], g2["p"]))
    g3 = EX.granger_causality(x, y1, 10, lags=2)
    check("short history says insufficient", g3["verdict"] == "insufficient")
    check("the note never claims interventional causality",
          "predictive" in g["note"])


def test_fractal():
    print("\n[fractal DFA calibration]")
    n = 512
    white = lcg_series(n, seed=11)
    f = EX.fractal_dfa(white, n - 1)
    check("white noise lands random-walk-ish (0.4<H<0.6)",
          0.4 < f["h"] < 0.6, "H=%.3f %s" % (f["h"], f["classification"]))
    # persistent: heavy smoothing -> long-range correlations
    smooth = lcg_series(n, seed=12, smooth=20)
    fp = EX.fractal_dfa(smooth, n - 1)
    check("smoothed (persistent) series H > 0.55", fp["h"] > 0.55,
          "H=%.3f" % fp["h"])
    # anti-persistent: first differences of noise
    anti = [white[j] - white[j - 1] for j in range(1, n)]
    fa = EX.fractal_dfa(anti, len(anti) - 1)
    check("differenced (anti-persistent) series H < 0.45", fa["h"] < 0.45,
          "H=%.3f" % fa["h"])
    fshort = EX.fractal_dfa(white, 10)
    check("short history says insufficient", fshort["classification"]
          == "insufficient" and fshort["note"] == "insufficient-history")


def test_chaos():
    print("\n[chaos metrics]")
    # AR(1) phi=0.9: half-life = -ln2/ln0.9 ~ 6.58
    rng = EX.Rng(21)
    x = [0.0]
    for _ in range(400):
        x.append(0.9 * x[-1] + 0.1 * rng.gauss())
    oh = EX.ou_half_life(x, len(x) - 1)
    check("AR(1) phi=0.9 half-life recovered (~6.58)",
          abs(oh["half_life"] - 6.579) < 0.6, "%.3f" % oh["half_life"])
    # random walk: phi ~ 1 -> no half-life, says so
    rw = [0.0]
    for _ in range(400):
        rw.append(rw[-1] + rng.gauss())
    ow = EX.ou_half_life(rw, len(rw) - 1)
    check("random walk yields no half-life (fail-closed)",
          math.isnan(ow["half_life"]) and ow["note"] != "",
          ow["note"])
    # T093: divergence-rate calibration against DETERMINISTIC ground truth
    # (the estimator's honest domain — see the docstring: on stochastic
    # series the NN noise floor masquerades as weak positive divergence).
    def logi(r, n=600, x0=0.21):
        s, x = [], x0
        for _ in range(n):
            x = r * x * (1.0 - x)
            s.append(x)
        return s
    l_chaos = EX.divergence_rate(logi(4.0), 599)
    check("chaotic logistic map (r=4.0) diverges (lyap > 0)",
          l_chaos["lyap"] > 0, "%.4f" % l_chaos["lyap"])
    l_2cycle = EX.divergence_rate(logi(3.2), 599)
    check("r=3.2 (2-cycle) converges (lyap < -0.5)", l_2cycle["lyap"] < -0.5,
          "%.4f" % l_2cycle["lyap"])
    l_fixed = EX.divergence_rate(logi(2.8), 599)
    check("r=2.8 (fixed point) converges (lyap < 0)", l_fixed["lyap"] < 0,
          "%.4f" % l_fixed["lyap"])
    # stochastic series: small-positive is the DOCUMENTED noise-floor
    # behaviour — assert the magnitude stays small, not the sign
    ld = EX.divergence_rate(x, len(x) - 1)
    lr = EX.divergence_rate(rw, len(rw) - 1)
    check("stochastic series read small (|lyap| < 0.06, noise floor)",
          abs(ld["lyap"]) < 0.06 and abs(lr["lyap"]) < 0.06,
          "ar1=%.4f rw=%.4f" % (ld["lyap"], lr["lyap"]))
    fn = EX.false_neighbors(x, len(x) - 1)
    check("FNN fractions are fractions", all(0.0 <= v <= 1.0
          for v in fn["fnn"].values()) or fn["note"] != "", str(fn))
    fns = EX.false_neighbors([1.0, 2.0, 3.0], 2)
    check("FNN on a stub says insufficient",
          fns["note"] == "insufficient-history")


def test_whatif():
    print("\n[what-if replay]")
    # 6 bars: hand-computable. Entry = bar[1] open=100; horizon 5 -> exit at
    # bar 5 close=110. gross = 10%; cost = 2*10bps = 0.2%.
    # T093: the 6th bar exists because T091 made the strict horizon REFUSE
    # when i+horizon exceeds the array (clipping to len(bars)-1 was
    # look-ahead). 5 bars + horizon 5 is now an honest refusal — asserted
    # below as "horizon-unavailable".
    bars = [(100, 101, 99, 100.5), (100, 104, 98, 103), (103, 106, 102, 105),
            (105, 108, 104, 107), (107, 111, 106, 110), (108, 111, 107, 110)]
    w = EX.whatif(bars, 0, +1, horizon=5, cost_bps=10.0)
    check("long pnl is hand-exact", abs(w["pnl"] - (0.10 - 0.002)) < 1e-9,
          "%.6f" % w["pnl"])
    check("MAE uses the worst low vs entry", abs(w["mae"] - (98 - 100) / 100.0)
          < 1e-9, "%.6f" % w["mae"])
    check("the flat baseline is named", w["flat_pnl"] == 0.0)
    wshort = EX.whatif(bars, 0, +1, horizon=1)
    check("a 1-bar horizon exits at bar 1 close",
          abs((wshort["pnl"] + 0.002) - (103 - 100) / 100.0) < 1e-9,
          "%.6f" % wshort["pnl"])
    wend = EX.whatif(bars, 4, +1)
    check("no next bar -> insufficient", wend["verdict"] == "insufficient")
    # no look-ahead: appending bars must not change the verdict at bar 0
    w2 = EX.whatif(bars + [(500, 600, 400, 550)], 0, +1, horizon=5)
    check("prefix invariance: later bars cannot change bar 0",
          w2["pnl"] == w["pnl"] and w2["mae"] == w["mae"])
    # strict horizon: 5 bars cannot honour horizon 5 at i=0 -> refusal
    wshort5 = EX.whatif(bars[:5], 0, +1, horizon=5)
    check("unreachable horizon is refused (no clipping to last bar)",
          wshort5["verdict"] == "insufficient"
          and wshort5["note"] == "horizon-unavailable")


def test_games():
    print("\n[game theory]")
    # matching pennies: zero-sum, no pure NE, unique mixed (0.5, 0.5)
    mp = EX.solve_2x2(1, -1, -1, 1, zero_sum=True)
    check("matching pennies has NO pure NE", mp["pure_ne"] == [])
    check("...and the mixed NE is (0.5, 0.5)",
          mp["mixed"] is not None and abs(mp["mixed"][0] - 0.5) < 1e-9
          and abs(mp["mixed"][1] - 0.5) < 1e-9, str(mp["mixed"]))
    # prisoner's dilemma (T=5 R=3 P=1 S=0): unique pure NE = (bottom, left)
    pd = EX.solve_2x2(3, 0, 5, 1)
    check("PD unique pure NE is bottom-left", pd["pure_ne"] == [(1, 0)],
          str(pd["pure_ne"]))
    check("PD has no interior mixed NE", pd["mixed"] is None)
    # weak-dominance game naming
    dom = EX.solve_2x2(2, 1, 1, 0)
    check("a dominance-solvable game names the dominated strategy",
          "row-bottom" in dom["dominated"] or "row-top" in dom["dominated"],
          dom["dominated"])
    # opponent model
    om = EX.opponent_model([-0.1] * 30, 29)
    check("a consistently losing rival is 'fade'", om["class"] == "fade",
          str(om))
    om2 = EX.opponent_model([0.1] * 30, 29)
    check("a consistently winning rival is 'trend'", om2["class"] == "trend")
    om3 = EX.opponent_model([0.1] * 3, 2)
    check("too few trades -> neutral fail-closed", om3["class"] == "neutral"
          and om3["note"] == "insufficient-history")


def test_genetic():
    print("\n[genetic search]")
    bounds = [(-5.0, 5.0)] * 3
    sphere = lambda v: -(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    r = EX.genetic_search(bounds, sphere, pop=40, gens=200, seed=3)
    check("sphere in 3D converges within 1e-2 of the optimum",
          r["best_fitness"] > -1e-2, "%.6f" % r["best_fitness"])
    check("the genome never leaves its box",
          all(-5.0 <= v <= 5.0 for v in r["best"]), str(r["best"]))
    r2 = EX.genetic_search(bounds, sphere, pop=40, gens=200, seed=3)
    check("same seed -> bit-identical search",
          r["best"] == r2["best"] and r["best_fitness"] == r2["best_fitness"])
    bad = EX.genetic_search([], sphere)
    check("empty bounds -> bad-config, not a crash", bad["note"] == "bad-config")


def _regime_bars(n=240, seed=77):
    """Piecewise momentum regimes: 30-bar trends then 30-bar reversals."""
    rng = EX.Rng(seed)
    bars = []
    price = 100.0
    for t in range(n):
        phase = (t // 30) % 2
        drift = 0.25 if phase == 0 else -0.25
        o = price
        c = price + drift + 0.15 * rng.gauss()
        h = max(o, c) + 0.1
        l = min(o, c) - 0.1
        bars.append((o, h, l, c))
        price = c
    return bars


def test_q_learning():
    print("\n[tabular Q-learning]")
    bars = _regime_bars()
    r = EX.q_learning(bars, episodes=60, seed=5)
    pnl_learned = EX.q_learning_pnl(bars, r["policy"])
    closes = [b[3] for b in bars]
    pnl_long = sum((closes[t + 1] - closes[t]) / closes[t]
                   for t in range(21, len(closes) - 1))
    check("the learned policy beats always-long on the SAME series",
          pnl_learned > pnl_long, "%.4f vs %.4f" % (pnl_learned, pnl_long))
    check("the learned policy beats always-flat (>= 0 after costs)",
          pnl_learned > 0.0, "%.4f" % pnl_learned)
    r2 = EX.q_learning(bars, episodes=60, seed=5)
    check("Q-learning is deterministic for a given seed",
          r["policy"] == r2["policy"])
    # reward arithmetic, hand-checked: long from t, one bar, cost 5bps
    c0, c1 = closes[30], closes[31]
    w = EX.whatif(bars, 30, +1, horizon=1, cost_bps=5.0)
    check("reward arithmetic matches the what-if replay",
          abs(w["pnl"] - (c1 - c0) / c0) < 0.01, "%.6f" % w["pnl"])
    # no look-ahead: state at t ignores bars after t
    bars2 = list(bars) + [(9999, 9999, 9999, 9999)] * 10
    st1 = EX.market_state(bars, 100)
    st2 = EX.market_state(bars2, 100)
    check("market_state is causal (prefix-invariant)", st1 == st2,
          "%s vs %s" % (st1, st2))
    check("market_state refuses short history",
          EX.market_state(bars, 10) is None)


def main():
    print("== monster exotics ==")
    test_rng()
    test_f_distribution()
    test_granger()
    test_fractal()
    test_chaos()
    test_whatif()
    test_games()
    test_genetic()
    test_q_learning()
    print(f"\nRESULT: {PASSED} passed, {FAILED} failed")
    for f in FAILURES:
        print(f"   - {f}")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
