#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — tests/test_monster_funding.py
#
#  Characterization tests for tools/monster/funding.py.
#
#  Funding carry is the one strategy in the Monster layer that is NOT a
#  forecast, so the tests are pure arithmetic plus risk flagging. The single
#  most important assertion is the last one in test_memecoin_trap: a trade with
#  a spectacular GROSS apr and a wide book must come back NOT takeable. Ranking
#  on gross apr is the mistake this module exists to prevent.
#
#  Offline, stdlib-only.  Usage: python tests/test_monster_funding.py
# =============================================================================
import math
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from monster import funding as FD  # noqa: E402

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


# ---------------------------------------------------------------------------
# ARITHMETIC
# ---------------------------------------------------------------------------
def test_periods_and_annualization():
    print("\n[funding period arithmetic]")
    check("8h funding settles 1095 times a year",
          approx(FD.periods_per_year(8.0), 1095.0), str(FD.periods_per_year(8.0)))
    check("4h funding settles 2190 times a year",
          approx(FD.periods_per_year(4.0), 2190.0))
    check("1h funding settles 8760 times a year",
          approx(FD.periods_per_year(1.0), 8760.0))

    # the published worked example: 0.012% per 8h -> 13.14% APR
    apr = FD.annualize(0.00012, 8.0)
    check("0.012% per 8h annualizes to 13.14%", approx(apr, 0.1314, 1e-9),
          f"got {apr}")
    check("0.01% per 8h annualizes to 10.95%",
          approx(FD.annualize(0.0001, 8.0), 0.1095, 1e-9))
    check("halving the interval doubles the period count and the APR",
          approx(FD.annualize(0.00012, 4.0), 2 * apr, 1e-9))

    try:
        FD.periods_per_year(0)
        check("zero interval raises", False, "no exception")
    except ValueError:
        check("zero interval raises", True)


def test_net_apr():
    print("\n[net APR after costs]")
    cfg = FD.FundingConfig()
    # round trip = 2*(4+5) fees + 2*(1+2) spread = 18 + 6 = 24 bps
    # amortized over 30 days: 0.0024 * 365/30 = 0.0292
    # plus 8% cost of capital -> total drag 0.1092
    gross = FD.annualize(0.00012, 8.0)
    net = FD.net_apr(gross, cfg)
    expect = gross - 0.0024 * (365.0 / 30.0) - 0.08
    print(f"       gross={gross:.4f} net={net:.4f} expected={expect:.4f}")
    check("net APR matches the hand-computed cost stack",
          approx(net, expect, 1e-9), f"{net} vs {expect}")
    check("net is always below gross", net < gross, f"{net} vs {gross}")

    # a longer hold amortizes the one-off costs and improves the net
    long_cfg = FD.FundingConfig(holding_days=180.0)
    check("a longer hold improves the net APR",
          FD.net_apr(gross, long_cfg) > net,
          f"{FD.net_apr(gross, long_cfg):.4f} vs {net:.4f}")

    # an intraday rotation gets eaten alive by fees
    fast = FD.FundingConfig(holding_days=1.0)
    check("an intraday rotation is destroyed by fees",
          FD.net_apr(gross, fast) < 0, f"{FD.net_apr(gross, fast):.4f}")


# ---------------------------------------------------------------------------
# DIRECTION + RISK FLAGS
# ---------------------------------------------------------------------------
def test_direction():
    print("\n[direction: which leg pays]")
    pos = FD.evaluate(FD.FundingQuote("BTCUSDT", 0.00012, 8.0, "Bybit"))
    check("positive funding -> short the perp", pos.direction == "short-perp",
          pos.direction)
    neg = FD.evaluate(FD.FundingQuote("BTCUSDT", -0.00012, 8.0, "Bybit"))
    check("negative funding -> long the perp", neg.direction == "long-perp",
          neg.direction)
    check("gross APR is the magnitude either way",
          approx(pos.gross_apr, neg.gross_apr, 1e-12),
          f"{pos.gross_apr} vs {neg.gross_apr}")


def test_risk_flags():
    print("\n[risk flags]")
    clean = FD.evaluate(FD.FundingQuote("BTCUSDT", 0.0006, 8.0, "Bybit",
                                        65000, 65010, 1.0, 2.0))
    check("a clean, rich carry passes", not clean.blocked, str(clean.risks))
    check("a clean carry is 'take'", clean.verdict == "take", clean.verdict)
    print(f"       clean: {clean.detail()}")

    # basis shock
    shock = FD.evaluate(FD.FundingQuote("BTCUSDT", 0.0006, 8.0, "Bybit",
                                        65000, 67500, 1.0, 2.0))
    check("a 385bps basis is flagged as a convergence shock",
          "basis-shock" in shock.risks, str(shock.risks))
    check("basis shock blocks the trade", shock.blocked)

    # illiquid spot
    illq = FD.evaluate(FD.FundingQuote("ALT", 0.0006, 8.0, "X", 1.0, 1.0,
                                       150.0, 100.0))
    check("a 150bps spot spread is flagged illiquid",
          "illiquid-spot" in illq.risks, str(illq.risks))
    check("illiquidity blocks the trade", illq.blocked)

    # rate near zero = about to cross
    near = FD.evaluate(FD.FundingQuote("BTCUSDT", 0.00001, 8.0, "Bybit"))
    check("a rate of 0.1bps is flagged as about to cross",
          "rate-near-zero" in near.risks, str(near.risks))
    check("near-zero rate blocks the trade", near.blocked)

    # below hurdle
    low = FD.evaluate(FD.FundingQuote("BTCUSDT", 0.00012, 8.0, "Bybit",
                                      65000, 65010, 1.0, 2.0))
    check("a 2.2% net carry is below the 10% hurdle",
          "below-hurdle" in low.risks, str(low.risks))

    # unstable funding history
    hist = [0.0006, -0.0004, 0.0007, -0.0003, 0.0005, -0.0002, 0.0006]
    unstable = FD.evaluate(FD.FundingQuote("BTCUSDT", 0.0006, 8.0, "Bybit",
                                           65000, 65010, 1.0, 2.0,
                                           history=hist))
    check("a sign-flipping funding history is flagged",
          "funding-unstable" in unstable.risks, str(unstable.risks))
    check("unstable funding blocks the trade", unstable.blocked)


def test_history_summary():
    print("\n[funding history summary]")
    s = FD.summarize_history([0.0001] * 10, 8.0)
    check("all-positive history reports pct_positive = 1.0",
          approx(s["pct_positive"], 1.0), str(s["pct_positive"]))
    check("all-positive history reports zero flips", s["flips"] == 0,
          str(s["flips"]))
    check("current streak counts the whole run", s["current_streak"] == 10,
          str(s["current_streak"]))
    check("mean APR is annualized", approx(s["mean_apr"], 0.0001 * 1095.0),
          str(s["mean_apr"]))

    s2 = FD.summarize_history([0.0001, -0.0001, 0.0001, -0.0001], 8.0)
    check("alternating signs report 3 flips", s2["flips"] == 3, str(s2["flips"]))
    check("alternating signs report pct_positive = 0.5",
          approx(s2["pct_positive"], 0.5))

    check("empty history is handled without raising",
          FD.summarize_history([], 8.0)["n"] == 0)
    check("volatility of a constant series is 0",
          approx(FD.summarize_history([0.0001] * 20, 8.0)["vol_apr"], 0.0))


# ---------------------------------------------------------------------------
# SCANNER
# ---------------------------------------------------------------------------
def test_memecoin_trap():
    print("\n[the memecoin trap: gross APR is not the ranking key]")
    # 0.02% per 8h = 21.9% gross, which looks great next to BTC's 13.14%.
    # But a 150bps spot spread and a 500bps basis make it uninvestable.
    meme = FD.evaluate(FD.FundingQuote("MEME", 0.0002, 8.0, "Gate",
                                       1.0, 1.05, 150.0, 100.0))
    btc = FD.evaluate(FD.FundingQuote("BTCUSDT", 0.00012, 8.0, "Bybit",
                                      65000, 65010, 1.0, 2.0))
    print(f"       meme: {meme.detail()}")
    print(f"       btc : {btc.detail()}")
    check("the memecoin's GROSS apr really is higher",
          meme.gross_apr > btc.gross_apr,
          f"{meme.gross_apr:.4f} vs {btc.gross_apr:.4f}")
    check("but the memecoin is NOT takeable", meme.blocked, str(meme.risks))
    check("the memecoin is ranked BELOW the clean trade by the scanner",
          FD.scan([FD.FundingQuote("MEME", 0.0002, 8.0, "Gate", 1.0, 1.05,
                                   150.0, 100.0),
                   FD.FundingQuote("BTCUSDT", 0.0006, 8.0, "Bybit",
                                   65000, 65010, 1.0, 2.0)])[0].symbol
          == "BTCUSDT")


def test_scan_sorting_and_reader():
    print("\n[scanner + CSV reader]")
    quotes = [
        FD.FundingQuote("AAA", 0.0002, 8.0, "X", 10.0, 10.0, 1.0, 2.0),
        FD.FundingQuote("BBB", 0.0008, 8.0, "X", 10.0, 10.0, 1.0, 2.0),
        FD.FundingQuote("CCC", 0.0005, 8.0, "X", 10.0, 10.0, 1.0, 2.0),
    ]
    trades = FD.scan(quotes)
    take = FD.takeable(trades)
    nets = [t.net_apr for t in take]
    check("takeable trades are sorted by net APR descending",
          nets == sorted(nets, reverse=True), str([round(n, 4) for n in nets]))
    check("the richest carry is first", trades[0].symbol == "BBB",
          trades[0].symbol)
    check("blocked trades sink to the end",
          all(not t.blocked for t in trades[:len(take)]))

    rows = ("symbol,rate,exchange,spot_price,perp_price,spot_spread_bps,"
            "perp_spread_bps\n"
            "BTCUSDT,0.0006,Bybit,65000,65010,1,2\n"
            "BADROW,notanumber,,,,,\n"
            ",0.0005,,,,,\n"
            "ETHUSDT,0.0004,OKX,3000,3002,2,3\n")
    path = os.path.join(tempfile.gettempdir(), "omniseed_funding_test.csv")
    with open(path, "w", encoding="utf-8") as f:
        f.write(rows)
    got = FD.load_quotes_csv(path)
    check("reader skips malformed rows instead of guessing",
          len(got) == 2, f"got {len(got)}")
    check("reader keeps the good symbols",
          [q.symbol for q in got] == ["BTCUSDT", "ETHUSDT"],
          str([q.symbol for q in got]))
    check("reader parses prices", approx(got[0].spot_price, 65000.0))
    check("basis is computed in bps",
          abs(got[0].basis_bps - (65010 / 65000 - 1) * 10000) < 1e-6,
          str(got[0].basis_bps))
    check("a missing basis is NaN, not zero",
          not math.isfinite(FD.FundingQuote("X", 0.001).basis_bps))
    os.remove(path)


def test_no_guarantee_language():
    print("\n[honesty checks]")
    doc = FD.__doc__ or ""
    check("the module docstring states it is not risk-free",
          "not risk-free" in doc.lower() or "NOT RISK-FREE" in doc,
          "docstring must not imply safety")
    check("the module docstring refuses to promise profit",
          "no guaranteed profit" in doc.lower(), "missing disclaimer")
    clean = FD.evaluate(FD.FundingQuote("BTCUSDT", 0.0006, 8.0, "Bybit",
                                        65000, 65010, 1.0, 2.0))
    check("no field on a trade claims a guarantee",
          not any("guarantee" in str(getattr(clean, a)).lower()
                  for a in clean.__slots__),
          "found guarantee language on a BasisTrade")


def main():
    print("== monster funding ==")
    test_periods_and_annualization()
    test_net_apr()
    test_direction()
    test_risk_flags()
    test_history_summary()
    test_memecoin_trap()
    test_scan_sorting_and_reader()
    test_no_guarantee_language()
    print(f"\nRESULT: {PASSED} passed, {FAILED} failed")
    for f in FAILURES:
        print(f"   - {f}")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
