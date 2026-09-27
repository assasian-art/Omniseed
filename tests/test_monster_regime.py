#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — tests/test_monster_regime.py
#
#  Characterization tests for tools/monster/regime.py.
#
#  The point of this suite is CALIBRATION, not coverage. A regime detector that
#  cannot tell a trend from a random walk is worse than no detector, because it
#  will confidently route capital to the wrong engine. So every statistic is
#  checked against series whose ground truth we KNOW by construction:
#
#      trending  drift > 0, small noise     -> VR > 1, H > 0.5, ER high, ADX high
#      mean-rev  OU process, theta known    -> VR < 1, H < 0.5, half-life ~ ln2/theta
#      random    GBM, zero drift            -> VR ~ 1, H ~ 0.5, ER low
#
#  Medians over many seeds, because any single realisation is noise.
#
#  Offline, stdlib-only.  Usage: python tests/test_monster_regime.py
# =============================================================================
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from monster import regime as R  # noqa: E402

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


def median(v):
    v = sorted(x for x in v if x is not None and math.isfinite(x))
    if not v:
        return float("nan")
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


# ---------------------------------------------------------------------------
# realistic synthetic bars
# ---------------------------------------------------------------------------
def make_bars(prices, seed=7, wick=0.0005):
    """OHLC where open == previous close and high/low bracket the bar body.

    Deliberately NOT `hi = p*1.01, lo = p*0.99`: a fixed-width band makes the
    true range independent of the actual bar move, which silently breaks the
    volatility estimators and the choppiness index. Real bars have wicks that
    scale with the bar.
    """
    rnd = random.Random(seed)
    bars, prev = [], None
    for i, p in enumerate(prices):
        o = prev if prev is not None else p
        c = p
        band = max(abs(c - o), p * wick)
        hi = max(o, c) + band * rnd.uniform(0.1, 0.6)
        lo = min(o, c) - band * rnd.uniform(0.1, 0.6)
        bars.append((1700000000 + i * 60, o, hi, lo, c,
                     1e6 * (1.0 + rnd.random())))
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
    """Discrete OU: X_t = (1-theta)X_{t-1} + theta*mu + eps  ->  lambda = -theta."""
    rnd = random.Random(seed)
    x, out = mu + 5.0, []
    for _ in range(n):
        x = x + theta * (mu - x) + rnd.gauss(0, sigma * mu)
        out.append(x)
    return out


def gen_momentum(n, seed, phi=0.25, sigma=0.01):
    """Autocorrelated returns: r_t = phi*r_{t-1} + eps, so rho(1) ~ phi > 0.

    Distinct from gen_trend on purpose. A constant-drift trend has IID returns
    and therefore rho(1) ~ 0 — the drift is not autocorrelation. Only this
    process is what a lag-1 autocorrelation is actually built to detect.
    """
    rnd = random.Random(seed)
    p, prev, out = 100.0, 0.0, []
    for _ in range(n):
        r = phi * prev + rnd.gauss(0, sigma)
        prev = r
        p *= math.exp(r)
        out.append(p)
    return out


def returns_of(prices):
    return [math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices))]


# ---------------------------------------------------------------------------
# DIRECTIONAL AXIS — calibration against known ground truth
# ---------------------------------------------------------------------------
def test_variance_ratio():
    print("\n[variance ratio  Lo-MacKinlay 1988]")
    t, o, r = [], [], []
    for seed in range(1, 13):
        t.append(R.variance_ratio(returns_of(gen_trend(400, seed)), 4)[0])
        o.append(R.variance_ratio(returns_of(gen_ou(400, seed)), 4)[0])
        r.append(R.variance_ratio(returns_of(gen_random(400, seed)), 4)[0])
    mt, mo, mr = median(t), median(o), median(r)
    print(f"       medians: trend={mt:.3f} ou={mo:.3f} random={mr:.3f}")
    check("VR > 1 on a trending series", mt > 1.0, f"got {mt:.3f}")
    check("VR < 1 on a mean-reverting series", mo < 0.9, f"got {mo:.3f}")
    check("VR ~ 1 on a random walk", 0.8 < mr < 1.2, f"got {mr:.3f}")
    check("VR orders trend > random > mean-revert", mt > mr > mo,
          f"{mt:.3f} {mr:.3f} {mo:.3f}")

    vr, z = R.variance_ratio(returns_of(gen_trend(400, 2)), 8)
    check("VR returns a finite robust z-statistic", math.isfinite(z),
          f"z={z}")
    check("VR is NaN when the window is too short",
          not math.isfinite(R.variance_ratio([0.01] * 3, 8)[0]))


def test_hurst():
    print("\n[Hurst exponent  R/S]")
    t, o, r = [], [], []
    for seed in range(1, 13):
        t.append(R.hurst_rs(returns_of(gen_trend(600, seed)), 8))
        o.append(R.hurst_rs(returns_of(gen_ou(600, seed)), 8))
        r.append(R.hurst_rs(returns_of(gen_random(600, seed)), 8))
    mt, mo, mr = median(t), median(o), median(r)
    print(f"       medians: trend={mt:.3f} ou={mo:.3f} random={mr:.3f}")
    check("H > 0.5 on a trending series", mt > 0.5, f"got {mt:.3f}")
    check("H < 0.5 on a mean-reverting series", mo < 0.5, f"got {mo:.3f}")
    check("H ~ 0.5 on a random walk", 0.40 < mr < 0.60, f"got {mr:.3f}")
    # The bug this suite was written to catch: feeding Hurst the price LEVEL
    # instead of the increments returns H > 1, which is not a number Hurst can
    # be. Guard the invariant directly.
    h_level = R.hurst_rs(gen_random(400, 3), 8)
    check("H on price levels is not silently accepted as a valid Hurst",
          not (0.0 <= h_level <= 1.0) or h_level > 0.9,
          f"level-H={h_level:.3f} (informational)")


def test_efficiency_ratio():
    print("\n[efficiency ratio  Kaufman]")
    straight = [100.0 + i for i in range(40)]
    zigzag = [100.0 + (1.0 if i % 2 else 0.0) for i in range(40)]
    er_s = R.efficiency_ratio(straight, 20)
    er_z = R.efficiency_ratio(zigzag, 20)
    print(f"       straight={er_s:.3f}  zigzag={er_z:.3f}")
    check("ER ~ 1 on a straight line", er_s > 0.99, f"got {er_s:.3f}")
    check("ER ~ 0 on a pure zigzag", er_z < 0.10, f"got {er_z:.3f}")


def test_adx_and_choppiness():
    print("\n[ADX  Wilder  |  choppiness]")
    adx_t = median([R.adx(make_bars(gen_trend(300, s)), 14) for s in range(1, 6)])
    adx_o = median([R.adx(make_bars(gen_ou(300, s)), 14) for s in range(1, 6)])
    print(f"       median ADX: trend={adx_t:.1f}  ou={adx_o:.1f}")
    check("ADX > 25 on a trend", adx_t > 25.0, f"got {adx_t:.1f}")
    check("ADX < 25 on a range", adx_o < 25.0, f"got {adx_o:.1f}")

    ch_t = median([R.choppiness(make_bars(gen_trend(300, s)), 14) for s in range(1, 6)])
    ch_o = median([R.choppiness(make_bars(gen_ou(300, s)), 14) for s in range(1, 6)])
    print(f"       median CHOP: trend={ch_t:.1f}  ou={ch_o:.1f}")
    check("choppiness lower on a trend than on a range", ch_t < ch_o,
          f"{ch_t:.1f} vs {ch_o:.1f}")


def test_autocorr_and_linreg():
    print("\n[lag-1 autocorrelation | trend consistency | linreg]")
    # A constant-drift trend has IID returns, so rho(1) is ~0: the drift is not
    # autocorrelation. This is not a defect, it is what the statistic means, and
    # pretending otherwise would make the detector lie about drift trends.
    rho_drift = median([R.autocorr1(returns_of(gen_trend(400, s)))
                        for s in range(1, 9)])
    rho_mom = median([R.autocorr1(returns_of(gen_momentum(400, s)))
                      for s in range(1, 9)])
    rho_ou = median([R.autocorr1(returns_of(gen_ou(400, s)))
                     for s in range(1, 9)])
    print(f"       rho(1): drift-trend={rho_drift:+.3f} momentum={rho_mom:+.3f}"
          f" ou={rho_ou:+.3f}")
    check("rho(1) ~ 0 on a pure drift trend (drift is not autocorrelation)",
          abs(rho_drift) < 0.15, f"got {rho_drift:+.3f}")
    check("rho(1) > 0 on an autocorrelated (momentum) series",
          rho_mom > 0.10, f"got {rho_mom:+.3f}")
    check("rho(1) < 0 on a mean-reverting series", rho_ou < 0.0,
          f"got {rho_ou:+.3f}")
    check("rho(1) orders momentum > drift > mean-revert",
          rho_mom > rho_drift > rho_ou,
          f"{rho_mom:+.3f} {rho_drift:+.3f} {rho_ou:+.3f}")

    # trend consistency: the statistic that separates drift trends from walks,
    # because VR and rho are both structurally neutral on a drift trend.
    con_t = median([R.trend_consistency(gen_trend(400, s), 20, 50)
                    for s in range(1, 9)])
    con_r = median([R.trend_consistency(gen_random(400, s), 20, 50)
                    for s in range(1, 9)])
    con_o = median([R.trend_consistency(gen_ou(400, s), 20, 50)
                    for s in range(1, 9)])
    print(f"       consistency: trend={con_t:.3f} random={con_r:.3f}"
          f" ou={con_o:.3f}")
    check("MA-ribbon consistency ~1.0 on a sustained trend", con_t > 0.90,
          f"got {con_t:.3f}")
    check("MA-ribbon consistency ~0.5 on a random walk",
          0.35 < con_r < 0.65, f"got {con_r:.3f}")
    check("consistency separates trend from random walk",
          con_t - con_r > 0.30, f"{con_t:.3f} vs {con_r:.3f}")

    slope, r2 = R.linreg_trend([100.0 + 2.0 * i for i in range(50)])
    check("linreg R2 = 1 on a perfect line", approx(r2, 1.0, 1e-9), f"r2={r2}")
    check("linreg slope positive on an uptrend", slope > 0, f"slope={slope}")


# ---------------------------------------------------------------------------
# VOLATILITY AXIS
# ---------------------------------------------------------------------------
def test_yang_zhang():
    print("\n[Yang-Zhang volatility]")
    calm = R.yang_zhang_vol(make_bars(gen_random(200, 4, sigma=0.004)), 20)
    wild = R.yang_zhang_vol(make_bars(gen_random(200, 4, sigma=0.020)), 20)
    print(f"       calm={calm:.5f}  wild={wild:.5f}")
    check("Yang-Zhang vol is finite and positive", calm > 0 and wild > 0,
          f"{calm} {wild}")
    check("Yang-Zhang vol scales with the true sigma", wild > calm * 2.0,
          f"{calm:.5f} -> {wild:.5f}")

    # The whole reason to use YZ: it accounts for the overnight gap. Two series
    # with identical open-to-close moves but different gaps must not score the
    # same.
    flat = [(1000 + i * 60, 100.0, 100.5, 99.5, 100.2, 1e6) for i in range(60)]
    gappy = [(1000 + i * 60, 100.0 * (1.05 if i % 2 else 1.0),
              100.5 * (1.05 if i % 2 else 1.0), 99.5, 100.2, 1e6)
             for i in range(60)]
    v_flat, v_gap = R.yang_zhang_vol(flat, 20), R.yang_zhang_vol(gappy, 20)
    print(f"       no-gap={v_flat:.6f}  gapped={v_gap:.6f}")
    check("Yang-Zhang reacts to overnight gaps (unlike close-to-close)",
          v_gap > v_flat * 2.0, f"{v_flat:.6f} vs {v_gap:.6f}")

    check("percentile_rank is 1.0 for the max",
          approx(R.percentile_rank([1, 2, 3, 4], 4), 1.0))
    check("percentile_rank is 0.25 for the min of 4",
          approx(R.percentile_rank([1, 2, 3, 4], 1), 0.25))


# ---------------------------------------------------------------------------
# TRADEABILITY
# ---------------------------------------------------------------------------
def test_ou_half_life():
    print("\n[OU half-life]")
    # lambda = -theta exactly, so the true half-life is ln(2)/theta.
    got = []
    for seed in range(1, 9):
        got.append(R.ou_half_life(gen_ou(400, seed, theta=0.30), 60))
    m = median(got)
    expect = math.log(2.0) / 0.30
    print(f"       recovered median={m:.2f}  theoretical={expect:.2f}")
    check("half-life recovered within 3x of the theoretical value",
          0.33 * expect < m < 3.0 * expect, f"{m:.2f} vs {expect:.2f}")

    check("half-life is NaN for a pure trend (no reversion)",
          not math.isfinite(R.ou_half_life([100.0 + i for i in range(100)], 60)))


def test_mean_reversion_corroboration():
    print("\n[mean-reversion corroboration]")
    # An OLS half-life alone is NOT evidence: a random walk routinely produces a
    # spuriously short one. The property must require an independent statistic
    # to agree.
    st = R.RegimeState(ts=0, half_life=6.0, vr=1.02, hurst=0.52)
    check("short half-life + random-walk VR/Hurst -> NOT tradeable",
          st.mean_reversion_tradeable is False)
    st2 = R.RegimeState(ts=0, half_life=6.0, vr=0.70, hurst=0.52)
    check("short half-life + VR < 0.95 -> tradeable",
          st2.mean_reversion_tradeable is True)
    st3 = R.RegimeState(ts=0, half_life=90.0, vr=0.70, hurst=0.35)
    check("slow half-life is not tradeable even when reverting",
          st3.mean_reversion_tradeable is False)
    st4 = R.RegimeState(ts=0, half_life=float("nan"), vr=0.7, hurst=0.35)
    check("NaN half-life is not tradeable",
          st4.mean_reversion_tradeable is False)


# ---------------------------------------------------------------------------
# LABELS + HYSTERESIS
# ---------------------------------------------------------------------------
def test_labels():
    print("\n[regime labels]")
    bars_t = make_bars(gen_trend(400, 2))
    bars_o = make_bars(gen_ou(400, 3))
    st_t = R.detect(bars_t, len(bars_t) - 1)
    st_o = R.detect(bars_o, len(bars_o) - 1)
    print(f"       trend series -> {st_t.label} (trend_score={st_t.trend_score:.3f})")
    print(f"       ou series    -> {st_o.label} (trend_score={st_o.trend_score:.3f})")
    check("a trending series labels as trend_up", st_t.label == "trend_up",
          st_t.label)
    check("a mean-reverting series does NOT label as trend_up",
          st_o.label != "trend_up", st_o.label)
    check("trend_score is higher on the trend than on the OU series",
          st_t.trend_score > st_o.trend_score,
          f"{st_t.trend_score:.3f} vs {st_o.trend_score:.3f}")
    check("trend_score is in [0,1]", 0.0 <= st_t.trend_score <= 1.0,
          str(st_t.trend_score))
    check("label is always from the engine's vocabulary",
          st_t.label in ("trend_up", "trend_down", "range", "high_vol"))


def test_hysteresis():
    print("\n[hysteresis]")
    bars = make_bars(gen_random(400, 9))
    raw = R.scan(bars, tracked=False)
    trk = R.scan(bars, tracked=True)
    raw_flips = sum(1 for i in range(1, len(raw)) if raw[i].label != raw[i - 1].label)
    trk_flips = sum(1 for i in range(1, len(trk)) if trk[i].label != trk[i - 1].label)
    print(f"       raw label changes={raw_flips}  tracked={trk_flips}")
    check("hysteresis does not increase label churn", trk_flips <= raw_flips,
          f"{trk_flips} vs {raw_flips}")
    check("every bar still gets a label",
          all(s.label for s in trk) and len(trk) == len(bars))


# ---------------------------------------------------------------------------
# NO LOOK-AHEAD
# ---------------------------------------------------------------------------
def test_no_lookahead():
    print("\n[NO LOOK-AHEAD: prefix invariance]")
    full = make_bars(gen_random(300, 5))
    fields = ("trend_score", "vol_score", "label", "direction", "adx", "er",
              "chop", "vr", "hurst", "r2", "rho1", "half_life")
    bad = []
    for i in range(250, 300):
        a = R.detect(full, i)
        b = R.detect(full[:i + 1], i)
        for f in fields:
            x, y = getattr(a, f), getattr(b, f)
            if isinstance(x, float) and isinstance(y, float):
                if math.isfinite(x) != math.isfinite(y):
                    bad.append((i, f, x, y))
                elif math.isfinite(x) and abs(x - y) > 1e-12:
                    bad.append((i, f, x, y))
            elif x != y:
                bad.append((i, f, x, y))
    check("detect(i) is identical with and without future bars", not bad,
          f"{len(bad)} mismatches, first={bad[:2]}")

    # the stateful path must also be prefix-invariant
    full2 = make_bars(gen_random(260, 6))
    a = R.scan(full2, tracked=True)
    b = R.scan(full2[:220], tracked=True)
    mism = [i for i in range(220) if a[i].label != b[i].label]
    check("RegimeTracker labels are prefix-invariant", not mism,
          f"{len(mism)} mismatches, first={mism[:3]}")

    # truncating the input must not change an EARLIER bar's reading either
    long_ = make_bars(gen_random(400, 7))
    short = long_[:250]
    same = all(
        math.isclose(R.detect(long_, i).trend_score,
                     R.detect(short, i).trend_score, abs_tol=1e-12)
        for i in range(200, 250))
    check("appending 150 future bars changes nothing at i=200..249", same)


def main():
    print("== monster regime ==")
    test_variance_ratio()
    test_hurst()
    test_efficiency_ratio()
    test_adx_and_choppiness()
    test_autocorr_and_linreg()
    test_yang_zhang()
    test_ou_half_life()
    test_mean_reversion_corroboration()
    test_labels()
    test_hysteresis()
    test_no_lookahead()
    print(f"\nRESULT: {PASSED} passed, {FAILED} failed")
    for f in FAILURES:
        print(f"   - {f}")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
