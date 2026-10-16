#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — tests/test_monster_chart.py
#
#  Characterization tests for tools/monster/chart_reader.py (Phase 3.1).
#
#  Candlestick patterns are CONTESTED evidence; the module exists to read
#  them as one low-weight zoo voice behind the same vetoes, so the tests pin
#  the three properties that make that honest: exact pattern geometry
#  (hand-constructed bars, exact reason strings), NO LOOK-AHEAD (prefix
#  invariance), and FAIL-CLOSED visibility (every call returns all six
#  pattern verdicts, inactive ones carrying a written reason).
#
#  Offline, stdlib-only.  Usage: python tests/test_monster_chart.py
# =============================================================================
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from monster import chart_reader as CR  # noqa: E402

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


def bar(o, h, l, c, v=100.0):
    return {"time": "t", "open": o, "high": h, "low": l, "close": c,
            "volume": v}


def sigs_at(bars, i):
    return {s.name: s for s in CR.chart_signals(bars, i)}


# ---------------------------------------------------------------------------
# PATTERN GEOMETRY
# ---------------------------------------------------------------------------
def test_engulfing():
    print("\n[engulfing]")
    # prev: bearish 100->90 (body 10); cur: bullish 88->103 (body 15),
    # open below prev close, close above prev open.
    bars = [bar(100, 101, 89, 90), bar(88, 104, 87, 103)]
    s = sigs_at(bars, 1)["chart.engulfing"]
    check("bullish engulfing fires +", s.direction == 1.0, str(s.direction))
    check("reason states both bodies",
          "bullish-engulfing" in s.reason and "15.00>10.00" in s.reason,
          s.reason)
    # mirror
    bars2 = [bar(90, 101, 89, 100), bar(102, 104, 86, 87)]
    s2 = sigs_at(bars2, 1)["chart.engulfing"]
    check("bearish engulfing fires -", s2.direction == -1.0, str(s2.direction))
    # body NOT larger -> no pattern
    bars3 = [bar(100, 101, 89, 90), bar(91, 104, 89, 102)]
    check("a smaller body does not engulf",
          sigs_at(bars3, 1)["chart.engulfing"].confidence == 0.0)
    # open == prev open edge (o < pc required strict)
    bars4 = [bar(100, 101, 89, 90), bar(90, 104, 89, 103)]
    check("open exactly at prev close is NOT an engulfing (strict <)",
          sigs_at(bars4, 1)["chart.engulfing"].confidence == 0.0)


def test_hammer_and_star():
    print("\n[hammer / shooting star]")
    # hammer: small body at the TOP, long lower wick (l=90, o=99, c=100,
    # h=100.5) -> lower=9, abody=1, upper=0.5
    bars = [bar(99, 100.5, 90, 100)]
    s = sigs_at(bars, 0)["chart.hammer"]
    check("hammer fires +", s.direction == 1.0, str(s.direction))
    check("reason names the wick ratio", "hammer lower-wick 9.00x" in s.reason,
          s.reason)
    # shooting star: mirror at a high
    bars2 = [bar(101, 110, 99.5, 100)]
    s2 = sigs_at(bars2, 0)["chart.hammer"]
    check("shooting star fires -", s2.direction == -1.0, str(s2.direction))
    # long wick but body too big -> no pattern
    bars3 = [bar(95, 101, 90, 100)]
    check("a large body is not a hammer",
          sigs_at(bars3, 0)["chart.hammer"].confidence == 0.0)


def test_marubozu_doji():
    print("\n[marubozu / doji]")
    bars = [bar(100, 110, 100, 110)]
    s = sigs_at(bars, 0)["chart.marubozu"]
    check("full-range bull body is a marubozu", s.direction == 1.0)
    check("reason quotes body share", "100% of range" in s.reason, s.reason)
    bars2 = [bar(110, 110, 100, 100)]
    check("full-range bear body is a bearish marubozu",
          sigs_at(bars2, 0)["chart.marubozu"].direction == -1.0)
    # doji: body 0.5 in a range of 10
    bars3 = [bar(100, 105, 95, 100.5)]
    d = sigs_at(bars3, 0)["chart.doji"]
    check("a 5%-range body is a doji", d.confidence == 0.0
          and d.direction == 0.0)
    check("the doji verdict says indecision", "indecision" in d.reason,
          d.reason)


def test_inside_bar():
    print("\n[inside-bar compression]")
    # 10 wide bars (range 2), then a COMPRESSED bar inside the previous range
    bars = [bar(100 + i, 101 + i, 99 + i, 100.5 + i) for i in range(10)]
    prev = bars[-1]
    bars.append(bar(prev["open"], prev["high"] - 0.5, prev["low"] + 0.5,
                    prev["open"] + 0.05))
    s = sigs_at(bars, 10)["chart.inside_bar"]
    check("an inside bar at compressed range is reported",
          "compression" in s.reason and s.confidence == 0.0, s.reason)
    check("it demands resolution rather than firing a direction",
          s.direction == 0.0)
    # inside but NOT compressed vs history -> no pattern
    wide = [bar(100 + 2 * i, 103 + 2 * i, 97 + 2 * i, 101 + 2 * i)
            for i in range(10)]
    wide.append(bar(101, 118, 99, 117))
    check("an inside bar at NORMAL range is not compression",
          sigs_at(wide, 10)["chart.inside_bar"].confidence == 0.0)


def test_breakout_no_lookahead():
    print("\n[breakout + no look-ahead]")
    bars = [bar(100 + i, 101 + i, 99 + i, 100.5 + i) for i in range(10)]
    # channel from bars 0..9: hh=110 (bar9 high), ll=99
    bars.append(bar(110, 112, 109.5, 111.5))     # close 111.5 > 110
    s = sigs_at(bars, 10)["chart.breakout"]
    check("a close above the prior channel fires +", s.direction == 1.0)
    check("the reason names the channel with the bars ACTUALLY scanned",
          "10dbar-high 110.00" in s.reason, s.reason)
    # the CURRENT bar's high must NOT be the channel (self-inclusion)
    bars2 = [bar(100 + i, 101 + i, 99 + i, 100.5 + i) for i in range(10)]
    bars2.append(bar(110, 130, 109, 105))        # spikes to 130 but CLOSES low
    check("a spike high that closes back inside is NOT a breakout",
          sigs_at(bars2, 10)["chart.breakout"].confidence == 0.0)
    # downward channel break
    bars3 = [bar(100 - i, 101 - i, 99 - i, 100.5 - i) for i in range(10)]
    bars3.append(bar(90, 90.5, 87, 87.5))        # channel ll = 90 (bar0 low)
    check("a close below the prior channel fires -",
          sigs_at(bars3, 10)["chart.breakout"].direction == -1.0)

    # NO LOOK-AHEAD: everything after bar i must be invisible
    full = bars + [bar(200, 250, 190, 240)]
    a = sigs_at(bars, 10)["chart.breakout"]
    b = sigs_at(full, 10)["chart.breakout"]
    check("prefix invariance: the verdict at bar i ignores later bars",
          a.reason == b.reason and a.direction == b.direction
          and a.confidence == b.confidence, "%r vs %r" % (a.reason, b.reason))


def test_fail_closed():
    print("\n[fail-closed visibility]")
    # zero-range bar
    z = sigs_at([bar(100, 100, 100, 100)], 0)
    check("every pattern reports on a zero-range bar",
          len(z) == 6 and all(s.confidence == 0.0 for s in z.values()))
    check("...with the zero-range reason",
          all("zero-range" in s.reason for s in z.values()),
          str([s.reason for s in z.values()]))
    # missing prev bar: two-bar patterns say so, single-bar ones still run
    single = sigs_at([bar(100, 105, 95, 104.5)], 0)
    check("six verdicts come back on a lone bar", len(single) == 6)
    check("the two-bar patterns say no-prev",
          "no-engulfing" in single["chart.engulfing"].reason
          and "no-inside_bar" in single["chart.inside_bar"].reason,
          single["chart.engulfing"].reason)
    check("single-bar patterns still EVALUATE without a prev bar",
          single["chart.marubozu"].reason == "no-marubozu"
          and single["chart.doji"].reason == "no-doji",
          single["chart.marubozu"].reason)
    # out-of-range index
    check("an out-of-range index is bad-bar, not a crash",
          all("bad-bar" in s.reason
              for s in CR.chart_signals([bar(100, 105, 95, 100)], 99)))
    # flat bar in history must not divide by zero in the channel
    flat = [bar(100, 101, 99, 100.5)] * 5 + [bar(100.5, 101.5, 99.5, 101)]
    try:
        sigs_at(flat, 5)
        check("flat history bars do not crash the channel walk", True)
    except ZeroDivisionError:
        check("flat history bars do not crash the channel walk", False)


def test_order_and_csv():
    print("\n[order + CSV round-trip]")
    names = [s.name for s in CR.chart_signals([bar(100, 105, 95, 100)], 0)]
    check("signals come back in the pinned pattern order",
          names == ["chart." + p for p in CR.PATTERN_ORDER], str(names))
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "bars.csv")
        with open(p, "w", encoding="utf-8") as f:
            f.write("time,open,high,low,close,volume,source\n")
            f.write("1,100,105,95,104,1000,binance\n")
            f.write("2,bad,row,,,,\n")           # skipped, not guessed
            f.write("3,100,110,100,110,1000,binance\n")
        bars = CR.read_chart_csv(p)
        check("the CSV reader keeps good rows", len(bars) == 2, str(len(bars)))
        s = sigs_at(bars, 1)["chart.marubozu"]
        check("bars read from CSV feed the same geometry",
              s.direction == 1.0 and "100%" in s.reason, s.reason)


def main():
    print("== monster chart reader ==")
    test_engulfing()
    test_hammer_and_star()
    test_marubozu_doji()
    test_inside_bar()
    test_breakout_no_lookahead()
    test_fail_closed()
    test_order_and_csv()
    print(f"\nRESULT: {PASSED} passed, {FAILED} failed")
    for f in FAILURES:
        print(f"   - {f}")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
