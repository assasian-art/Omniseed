#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — tests/test_monster_features.py
#
#  Characterization tests for tools/monster/features.py.
#
#  The important one is the NO-LOOK-AHEAD proof: every primitive must produce
#  identical values at index i whether or not later bars exist. Anything that
#  reads the future (swing confirmation) is checked to expose only what is
#  knowable at i.
#
#  Offline, stdlib-only.  Usage: python tests/test_monster_features.py
# =============================================================================
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from monster import features as F  # noqa: E402

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
    if not (math.isfinite(a) and math.isfinite(b)):
        return False
    return abs(a - b) <= tol


def make_bars(n, seed=11, start=100.0, vol=0.01):
    rnd = random.Random(seed)
    bars, p = [], start
    for i in range(n):
        p *= math.exp(rnd.gauss(0.0005, vol))
        hi, lo = p * 1.01, p * 0.99
        bars.append((1600000000 + i * 86400, p * 0.999, hi, lo, p,
                     1000.0 + rnd.random() * 100.0))
    return bars


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------
def test_sma_ema():
    print("\n-- sma / ema --")
    v = [1.0, 2.0, 3.0, 4.0, 5.0]
    s = F.sma(v, 3)
    check("sma: NaN before the window", math.isnan(s[0]) and math.isnan(s[1]))
    check("sma: exact mean at the window edge", approx(s[2], 2.0))
    check("sma: rolls forward", approx(s[4], 4.0))

    e = F.ema(v, 3)
    check("ema: NaN before the window", math.isnan(e[1]))
    check("ema: seeded with the SMA", approx(e[2], 2.0))
    # k = 2/(3+1) = 0.5 -> e3 = 4*0.5 + 2*0.5 = 3.0
    check("ema: first recursive step", approx(e[3], 3.0))

    check("sma: period<=0 is all-NaN", all(math.isnan(x) for x in F.sma(v, 0)))


def test_rsi():
    print("\n-- rsi --")
    up = [float(i) for i in range(1, 30)]
    r = F.rsi(up, 14)
    check("rsi: all-gains saturates at 100", approx(r[-1], 100.0))
    down = [float(30 - i) for i in range(30)]
    check("rsi: all-losses floors at 0", approx(F.rsi(down, 14)[-1], 0.0))
    check("rsi: NaN before period changes", math.isnan(r[13]))
    check("rsi: defined at index period", math.isfinite(r[14]))


def test_macd_bollinger_atr():
    print("\n-- macd / bollinger / atr --")
    bars = make_bars(120)
    line, sig, hist = F.macd(F.closes(bars))
    check("macd: histogram = line - signal where both exist",
          all(approx(hist[i], line[i] - sig[i], 1e-9)
              for i in range(len(bars))
              if math.isfinite(hist[i]) and math.isfinite(line[i])
              and math.isfinite(sig[i])))
    check("macd: NaN before the slow EMA", math.isnan(line[10]))

    mid, up, lo = F.bollinger(F.closes(bars), 20, 2.0)
    check("bollinger: lower < mid < upper", lo[-1] < mid[-1] < up[-1])
    check("bollinger: mid equals the SMA",
          approx(mid[-1], F.sma(F.closes(bars), 20)[-1]))

    a = F.atr(bars, 14)
    check("atr: positive", a[-1] > 0)
    check("atr: NaN before the window", math.isnan(a[5]))
    ap = F.atr_pct(bars, 14)
    check("atr_pct: positive fraction", 0 < ap[-1] < 1)


def test_vwap_and_stats():
    print("\n-- vwap / zscore / corr --")
    bars = make_bars(80)
    vw = F.rolling_vwap(bars, 20)
    check("vwap: inside the bar range",
          min(F.lows(bars)) <= vw[-1] <= max(F.highs(bars)))

    hist = [1.0, 1.1, 0.9, 1.0, 1.05, 0.95, 1.0, 1.1, 0.9, 1.0]
    z = F.rolling_zscore(hist + [50.0], 10)
    check("zscore: past window only -> large positive on a spike", z[-1] > 2.0)
    check("zscore: NaN before a full window", math.isnan(z[5]))
    flat = F.rolling_zscore([5.0] * 12, 10)
    check("zscore: zero variance -> 0 (not NaN/inf)", flat[-1] == 0.0)

    a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    check("corr: identical series -> +1",
          approx(F.rolling_corr(a, a, 5)[-1], 1.0, 1e-9))
    neg = [-x for x in a]
    check("corr: negated series -> -1",
          approx(F.rolling_corr(a, neg, 5)[-1], -1.0, 1e-9))
    check("corr: zero variance -> 0",
          F.rolling_corr([1.0] * 6, a, 5)[-1] == 0.0)


def test_microstructure_proxies():
    print("\n-- amihud / kyle --")
    bars = make_bars(60)
    am = F.amihud(bars)
    kl = F.kyle_lambda(bars)
    check("amihud: non-negative where defined",
          all(x >= 0 for x in am if math.isfinite(x)))
    check("amihud: NaN at index 0 (no return yet)", math.isnan(am[0]))
    check("kyle: NaN at index 0", math.isnan(kl[0]))
    check("kyle: non-negative where defined",
          all(x >= 0 for x in kl if math.isfinite(x)))


def test_swings_and_fib():
    print("\n-- swings / fibonacci --")
    vals = [5.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    sw = F.find_swings(vals, 2, 2, "low")
    check("swings: finds the V bottom", sw and sw[0][0] == 4)

    # A swing at idx=4 with right=2 is only knowable at index 6.
    check("swings: not confirmed before idx+right",
          F.confirmed_swings(sw, 5, 2) == [])
    check("swings: confirmed at idx+right",
          len(F.confirmed_swings(sw, 6, 2)) == 1)

    lv = F.fib_levels(100.0, 200.0)
    check("fib: 0.5 level of 100..200 is 150", approx(lv[1], 150.0))
    check("fib: 0.618 level", approx(lv[2], 200.0 - 100.0 * 0.618))
    check("fib: levels descend as ratios grow", lv[0] > lv[1] > lv[2])


def test_divergence():
    print("\n-- rsi divergence --")
    # price lower low, RSI higher low
    lows = [(2, 50.0), (10, 45.0)]
    rsi_vals = [float("nan")] * 20
    rsi_vals[2], rsi_vals[10] = 30.0, 38.0
    check("divergence: price LL + RSI HL -> True",
          F.bullish_divergence(rsi_vals, lows, 15, 2, 60))
    rsi_vals[10] = 20.0
    check("divergence: RSI also lower -> False",
          not F.bullish_divergence(rsi_vals, lows, 15, 2, 60))
    check("divergence: only one confirmed swing -> False",
          not F.bullish_divergence(rsi_vals, lows, 11, 2, 60))


# ---------------------------------------------------------------------------
# The important one: no look-ahead
# ---------------------------------------------------------------------------
def test_no_lookahead():
    print("\n-- NO LOOK-AHEAD --")
    full = make_bars(160, seed=21)
    cut = 120
    part = full[:cut]

    pairs = [
        ("sma", lambda b: F.sma(F.closes(b), 20)),
        ("ema", lambda b: F.ema(F.closes(b), 20)),
        ("rsi", lambda b: F.rsi(F.closes(b), 14)),
        ("macd_hist", lambda b: F.macd(F.closes(b))[2]),
        ("atr", lambda b: F.atr(b, 14)),
        ("atr_pct", lambda b: F.atr_pct(b, 14)),
        ("bollinger_lo", lambda b: F.bollinger(F.closes(b), 20)[2]),
        ("vwap", lambda b: F.rolling_vwap(b, 20)),
        ("vol_z", lambda b: F.rolling_zscore(F.volumes(b), 20)),
        ("amihud", lambda b: F.amihud(b)),
        ("kyle", lambda b: F.kyle_lambda(b)),
    ]
    for name, fn in pairs:
        a, b = fn(part), fn(full)
        ok = True
        for i in range(cut):
            x, y = a[i], b[i]
            if math.isnan(x) and math.isnan(y):
                continue
            if not approx(x, y, 1e-12):
                ok = False
                break
        check(f"no-lookahead: {name} unchanged by future bars", ok,
              f"diverged at index {i}")

    # rolling_corr: prefix must not depend on the future either.
    c1 = F.log_returns(F.closes(part))
    c2 = F.log_returns(F.closes(full))
    r1 = F.rolling_corr(c1, c1, 30)
    r2 = F.rolling_corr(c2, c2, 30)
    check("no-lookahead: rolling_corr prefix stable",
          all(approx(r1[i], r2[i], 1e-12) for i in range(cut)
              if math.isfinite(r1[i])))

    # swings: a swing confirmed only later must be invisible earlier.
    sw_full = F.find_swings(F.lows(full), 2, 2, "low")
    last = sw_full[-1][0]
    check("no-lookahead: final swing is not confirmed at its own bar",
          all(idx != last for idx, _ in F.confirmed_swings(sw_full, last, 2)))


def main():
    print("== monster features ==")
    test_sma_ema()
    test_rsi()
    test_macd_bollinger_atr()
    test_vwap_and_stats()
    test_microstructure_proxies()
    test_swings_and_fib()
    test_divergence()
    test_no_lookahead()
    print(f"\nRESULT: {PASSED} passed, {FAILED} failed")
    for f in FAILURES:
        print(f"   - {f}")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
