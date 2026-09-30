#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — tests/test_monster_signals.py
#
#  Characterization tests for tools/monster/signals.py — the external-signal
#  layer (on-chain flows, social volume, the event calendar, the commodity
#  complex, and the order-book depth path).
#
#  Proves, with hand-computed values where a value is claimed:
#    * the depth path uses the EXACT CKS OFI when a book is present, and the
#      signed-volume PROXY — LABELLED `proxy` AND DISCOUNTED — when it is not;
#    * on-chain signs are the standard reading (exchange inflow bearish, whale
#      accumulation bullish, rising stablecoin supply bullish);
#    * social attention is a TRAILING z-score (no look-ahead) and bias needs
#      BOTH attention and sentiment;
#    * a HIGH-impact calendar event opens a blackout window and a LOW-impact
#      one does not;
#    * the negative controls hold: a STALE feed is not a vote, an absent feed
#      is not a vote, and a coverage count of 0 means "no feed", not "neutral".
#
#  Offline, stdlib-only.  Usage: python tests/test_monster_signals.py
# =============================================================================
import importlib.util
import math
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOOLS = os.path.join(ROOT, "tools")
sys.path.insert(0, TOOLS)

from monster import signals as SG            # noqa: E402
from monster import strategies as ST         # noqa: E402
from monster import correlation_matrix as CM  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "monster_scan", os.path.join(TOOLS, "monster_scan.py"))
ms = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ms)

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


def mk_bars(n, rising=True):
    """Simple deterministic bars: (ts, o, h, l, c, v)."""
    bars = []
    for i in range(n):
        p = 100.0 + (i if rising else -i)
        bars.append((1600000000 + i * 86400, p - 0.5, p + 1.0, p - 1.0, p, 100.0))
    return bars


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------
def test_readers():
    print("\n-- readers --")
    d = tempfile.mkdtemp(prefix="osig_")
    try:
        p = os.path.join(d, "onchain.csv")
        with open(p, "w", encoding="utf-8") as f:
            f.write("ts,exchange_netflow,whale_net,stablecoin_delta\n")
            f.write("200,500,-200,100\n")
            f.write("100,0,0,0\n")               # out of order -> sorted
            f.write("150,1,2\n")                 # too few columns -> skipped
            f.write("not,a,row\n")               # non-numeric ts -> skipped
            f.write("300,x,1,2\n")               # unparsable float -> skipped
        rows = SG.read_onchain_csv(p)
        check("reader: malformed rows skipped", len(rows) == 2, str(rows))
        check("reader: rows are ts-ascending",
              [r[0] for r in rows] == [100, 200])
        check("reader: numeric payload parsed",
              approx(rows[1][1], 500.0) and approx(rows[1][2], -200.0))

        check("reader: a missing file yields [] (not an exception)",
              SG.read_onchain_csv(os.path.join(d, "nope.csv")) == [])

        # Calendar keeps TEXT columns; a bad impact skips the row.
        cp = os.path.join(d, "calendar.csv")
        with open(cp, "w", encoding="utf-8") as f:
            f.write("ts,kind,name,impact\n")
            f.write("500,macro,CPI,0.9\n")
            f.write("600,earnings,AAPL,bad\n")
        ev = SG.read_calendar_csv(cp)
        check("reader: calendar keeps text and skips bad impact",
              len(ev) == 1 and ev[0][1] == "macro" and ev[0][2] == "CPI")
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# 1. Order book / depth
# ---------------------------------------------------------------------------
def test_depth():
    print("\n-- depth (order book) --")
    # Hand-computed CKS OFI.
    #   s0=(100,5,101,4)  s1=(101,7,102,3): bid up +7, ask up -> -4  => 11
    book = SG.DepthBook({"norm": 100.0, "c": 1.0, "lam": 1.0})
    book.ingest([[1000, 100, 5, 101, 4], [1001, 101, 7, 102, 3]])
    s = book.signal(1001)
    check("depth: exact CKS OFI (bid up, ask up) == 11", approx(s.ofi, 11.0),
          str(s.ofi))
    check("depth: source is `cks` when a book is present", s.source == "cks")
    check("depth: not discounted on the CKS path", approx(s.discount, 1.0))
    check("depth: imbalance == ofi/norm", approx(s.imbalance, 0.11))
    check("depth: depth == last bid+ask size", approx(s.depth, 10.0))
    check("depth: beta == c / depth^lam", approx(s.beta, 0.1))

    # Bid down, ask down => -13 (the mirror image must be negative).
    b2 = SG.DepthBook({"norm": 100.0})
    b2.ingest([[1000, 100, 5, 101, 4], [1001, 99, 6, 100, 8]])
    s2 = b2.signal(1001)
    check("depth: bid falling is a negative OFI", approx(s2.ofi, -13.0),
          str(s2.ofi))

    # Book snapshots are time-respecting: nothing after `ts` is used.
    s_early = book.signal(1000)
    check("depth: only snapshots at/before ts are used",
          approx(s_early.ofi, 0.0))

    # No book -> the PROXY, labelled and discounted.
    bars = mk_bars(25, rising=True)
    s3 = SG.DepthBook({"norm": 100.0}).signal(10 ** 9, bars, 24)
    check("depth: falls back to `proxy` with no book", s3.source == "proxy")
    check("depth: proxy is discounted (0.70 default)",
          approx(s3.discount, 0.70) and approx(s3.imbalance, 0.70),
          str(s3.imbalance))
    check("depth: proxy of a monotone-up tape is positive", s3.imbalance > 0)

    # Neither book nor bars -> nothing, named.
    s4 = SG.DepthBook().signal(123)
    check("depth: no feed -> source `none` and a zero imbalance",
          s4.source == "none" and approx(s4.imbalance, 0.0))


# ---------------------------------------------------------------------------
# 2. On-chain
# ---------------------------------------------------------------------------
def test_onchain():
    print("\n-- on-chain --")
    oc = SG.OnChainFlow({"scale": 1000.0, "max_age": 86400.0})
    oc.ingest([[1000, 1000, 0, 0],        # exchange INFLOW  -> bearish
               [2000, 0, 1000, 0],        # whale accumulation -> bullish
               [3000, 0, 0, 1000]])       # stablecoin supply  -> bullish

    s_in = oc.signal(1000)
    check("on-chain: exchange inflow is BEARISH", s_in.score < 0,
          str(s_in.score))
    check("on-chain: inflow magnitude == w_exchange*tanh(1)",
          approx(s_in.score, 0.5 * (-math.tanh(1.0))))

    s_wh = oc.signal(2000)
    check("on-chain: whale accumulation is BULLISH", s_wh.score > 0)
    check("on-chain: whale magnitude == w_whale*tanh(1)",
          approx(s_wh.score, 0.3 * math.tanh(1.0)))

    s_st = oc.signal(3000)
    check("on-chain: rising stablecoin supply is BULLISH",
          approx(s_st.score, 0.2 * math.tanh(1.0)))

    check("on-chain: a fresh reading is not stale", not s_st.stale
          and s_st.source == "onchain")

    # The negative control: a reading older than max_age is NOT a vote.
    s_old = oc.signal(3000 + 86401)
    check("on-chain: a STALE reading is not a vote (score 0, source none)",
          approx(s_old.score, 0.0) and s_old.stale and s_old.source == "none")
    check("on-chain: staleness is reported, not hidden", s_old.age > 86400)

    # No feed at all.
    s_none = SG.OnChainFlow().signal(1000)
    check("on-chain: absent feed -> source none, score 0",
          s_none.source == "none" and approx(s_none.score, 0.0))


# ---------------------------------------------------------------------------
# 3. Social
# ---------------------------------------------------------------------------
def test_social():
    print("\n-- social --")
    sv = SG.SocialVolume({"window": 30, "z_full": 3.0, "max_age": 86400.0})
    base = [100, 102, 98, 101, 99, 100, 103, 97, 101]
    rows = [[1000 + i * 100, base[i], 50, 50] for i in range(len(base))]
    rows.append([1000 + 9 * 100, 400, 300, 100])          # the spike
    sv.ingest(rows)

    s = sv.signal(1000 + 9 * 100)
    check("social: mention spike saturates attention to 1.0",
          approx(s.attention, 1.0), str(s.attention))
    check("social: sentiment == (pos-neg)/(pos+neg)",
          approx(s.sentiment, 0.5), str(s.sentiment))
    check("social: bias == attention * sentiment",
          approx(s.bias, 0.5), str(s.bias))
    check("social: z is a large positive number", s.z > 100.0, str(s.z))
    check("social: fresh feed is named", s.source == "social" and not s.stale)

    # Attention needs a real trailing window (>= 5 rows) -> no look-ahead, no
    # invented variance.
    thin = SG.SocialVolume({"window": 30, "z_full": 3.0})
    thin.ingest([[10, 100, 1, 1], [20, 100, 1, 1], [30, 400, 3, 1]])
    st = thin.signal(30)
    check("social: fewer than 5 history rows -> attention 0 (no invented z)",
          approx(st.attention, 0.0) and approx(st.bias, 0.0))

    # Bias needs BOTH halves: loud but directionless -> ~0.
    flat = SG.SocialVolume({"window": 30, "z_full": 3.0})
    frows = [[1000 + i * 100, base[i], 50, 50] for i in range(len(base))]
    frows.append([1000 + 9 * 100, 400, 50, 50])           # pos == neg
    flat.ingest(frows)
    sf = flat.signal(1000 + 9 * 100)
    check("social: loud but directionless -> bias 0",
          approx(sf.attention, 1.0) and approx(sf.bias, 0.0))

    # Stale.
    so = sv.signal(1000 + 9 * 100 + 86401)
    check("social: stale feed -> no bias and source none",
          so.stale and so.source == "none" and approx(so.bias, 0.0))


# ---------------------------------------------------------------------------
# 4. Event calendar
# ---------------------------------------------------------------------------
def test_calendar():
    print("\n-- event calendar --")
    cal = SG.EventCalendar({"pre": 3600.0, "post": 1800.0,
                            "impact_min": 0.7, "tau": 3600.0})
    cal.ingest([(100000, "macro", "CPI", 0.9),
                (200000, "misc", "holiday", 0.3)])

    check("calendar: inside a high-impact window -> blackout",
          cal.signal(100000).blackout)
    check("calendar: inside the PRE window -> blackout",
          cal.signal(100000 - 3600).blackout)
    check("calendar: inside the POST window -> blackout",
          cal.signal(100000 + 1800).blackout)
    check("calendar: one second past the POST window -> no blackout",
          not cal.signal(100000 + 1801).blackout)
    check("calendar: before the PRE window -> no blackout",
          not cal.signal(100000 - 3601).blackout)

    lo = cal.signal(200000)
    check("calendar: a LOW-impact event never blackouts", not lo.blackout)
    check("calendar: a low-impact event still decays (context, not a gate)",
          lo.decay > 0.0)

    check("calendar: the blackout names its cause",
          "CPI" in cal.signal(100000).reason, cal.signal(100000).reason)

    # decay = impact * exp(-dt/tau)
    d0 = cal.signal(100000).decay
    d1 = cal.signal(100000 + 3600).decay
    check("calendar: decay == impact at the event",
          approx(d0, 0.9, 1e-9), str(d0))
    check("calendar: decay == impact*exp(-1) one tau later",
          approx(d1, 0.9 * math.exp(-1.0), 1e-9), str(d1))

    check("calendar: an empty calendar never blackouts",
          not SG.EventCalendar().signal(100000).blackout)


# ---------------------------------------------------------------------------
# 5. Commodities + pack CSV + the aggregator
# ---------------------------------------------------------------------------
def test_commodities_and_pack():
    print("\n-- commodities / pack / SignalSet --")
    series = {"OIL": [(1, 100.0), (2, 110.0)], "GOLD": [(1, 50.0), (2, 55.0)]}
    cr = SG.commodity_returns(series)
    check("commodities: log returns computed for present legs",
          set(cr) == {"OIL", "GOLD"})
    check("commodities: r1 == ln(110/100)",
          approx(cr["OIL"][1], math.log(1.1), 1e-12))
    check("commodities: an absent leg is skipped, not invented",
          "COPPER" not in cr)

    # Pack CSV round-trip.
    d = tempfile.mkdtemp(prefix="osig_")
    try:
        p = os.path.join(d, "signals.csv")
        packs = [
            SG.SignalPack(ts=1000, onchain=0.25, social=-0.5, depth=0.11,
                          depth_source="cks", coverage=3),
            SG.SignalPack(ts=2000, blackout=True, blackout_reason="macro:CPI",
                          coverage=0),
        ]
        n = SG.write_csv(p, packs)
        back = SG.read_csv(p)
        check("pack: write_csv returns the row count", n == 2)
        check("pack: round-trip preserves every field",
              len(back) == 2 and approx(back[1000].depth, 0.11)
              and back[1000].depth_source == "cks" and back[1000].coverage == 3)
        check("pack: blackout + reason survive the round-trip",
              back[2000].blackout and back[2000].blackout_reason == "macro:CPI")
        check("pack: detail() is CSV-safe",
              "," not in packs[0].detail() and "," not in packs[1].detail())

        # The aggregator: an empty set reports NO coverage, not a neutral vote.
        empty = SG.SignalSet()
        check("SignalSet: an empty set has no feeds", empty.empty())
        pe = empty.pack(1000)
        check("SignalSet: no feed -> coverage 0 and every field neutral",
              pe.coverage == 0 and approx(pe.onchain, 0.0)
              and approx(pe.social, 0.0) and pe.depth_source == "none")

        # from_dir wires all four readers and the pack counts live feeds.
        os.makedirs(os.path.join(d, "sig"), exist_ok=True)
        with open(os.path.join(d, "sig", "onchain.csv"), "w", encoding="utf-8") as f:
            f.write("ts,exchange_netflow,whale_net,stablecoin_delta\n1000,0,1000,0\n")
        with open(os.path.join(d, "sig", "social.csv"), "w", encoding="utf-8") as f:
            f.write("ts,mentions,positive,negative\n")
            for i, m in enumerate([100, 102, 98, 101, 99, 100, 103, 97, 101, 400]):
                f.write("%d,%d,300,100\n" % (1000 + i * 100, m))
        with open(os.path.join(d, "sig", "calendar.csv"), "w", encoding="utf-8") as f:
            f.write("ts,kind,name,impact\n1000,macro,CPI,0.9\n")
        with open(os.path.join(d, "sig", "depth.csv"), "w", encoding="utf-8") as f:
            f.write("ts,bid_px,bid_sz,ask_px,ask_sz\n1000,100,5,101,4\n")
        ss = SG.SignalSet.from_dir(os.path.join(d, "sig"))
        check("SignalSet: from_dir reports all four feeds",
              ss.feeds() == {"depth": 1, "onchain": 1, "social": 10,
                             "calendar": 1}, str(ss.feeds()))
        sp = ss.pack(1000)
        check("SignalSet: a blackout survives into the pack", sp.blackout)
        check("SignalSet: coverage counts the live families",
              sp.coverage == 3, str(sp.coverage))
        check("SignalSet: the pack names the depth source",
              sp.depth_source == "cks")
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# 6. Cross-asset time alignment (the §48 fix) + scanner integration
# ---------------------------------------------------------------------------
def test_cross_alignment():
    print("\n-- cross-asset time alignment --")
    m = CM.CorrelationMatrix(window=5)
    oil = [0.0] * 10
    ci = [0.0] * 15
    oil[9] = 0.02          # the target's CURRENT bar: OIL up
    ci[9] = -0.02          # the WRONG moment (same array position)
    ci[14] = 0.02          # the RIGHT moment (same timestamp)
    rets = {"OIL": oil, "COMMODITY_INDEX": ci}

    # Legacy: one shared index compares bar 9 of one series with bar 9 of the
    # other.  On series with different histories that is not the same instant.
    r_old = m.evaluate("OIL", +1, rets, 9)
    check("align: the positional read sees the WRONG moment (invalidate)",
          r_old.invalidate == 1 and r_old.confirm == 0)

    # Aligned: every peer is read at the index of the SAME timestamp.
    r_new = m.evaluate("OIL", +1, rets, 9,
                       {"OIL": 9, "COMMODITY_INDEX": 14})
    check("align: the timestamp-aligned read confirms",
          r_new.confirm == 1 and r_new.invalidate == 0)

    # A peer with no reading at this instant is SKIPPED, never mis-compared.
    r_skip = m.evaluate("OIL", +1, rets, 9, {"OIL": 9})
    check("align: a peer with no reading at this instant is skipped",
          r_skip.active == 0 and r_skip.confirm == 0 and r_skip.invalidate == 0)

    # Backward compatibility: no map -> exactly the old single-index behaviour.
    check("align: omitting idx_by_symbol preserves the legacy path",
          m.evaluate("OIL", +1, rets, 9).invalidate == 1)


def test_scan_integration():
    print("\n-- scanner: external signals --")
    d = tempfile.mkdtemp(prefix="osig_")
    try:
        csvdir = os.path.join(d, "market")
        outdir = os.path.join(d, "out")
        sigdir = os.path.join(d, "sig")
        for p in (csvdir, outdir, sigdir):
            os.makedirs(p, exist_ok=True)

        bars = mk_bars(40)
        with open(os.path.join(csvdir, "AAA_1d.csv"), "w", encoding="utf-8") as f:
            f.write("ts,open,high,low,close,volume\n")
            for (ts, o, h, lo, c, v) in bars:
                f.write("%d,%.2f,%.2f,%.2f,%.2f,%.2f\n" % (ts, o, h, lo, c, v))

        # A high-impact event exactly on bar 20's timestamp.
        ev_ts = bars[20][0]
        with open(os.path.join(sigdir, "calendar.csv"), "w", encoding="utf-8") as f:
            f.write("ts,kind,name,impact\n%d,macro,CPI,0.9\n" % ev_ts)
        # A live on-chain reading so the layer is not "empty".
        with open(os.path.join(sigdir, "onchain.csv"), "w", encoding="utf-8") as f:
            f.write("ts,exchange_netflow,whale_net,stablecoin_delta\n")
            f.write("%d,0,5000,0\n" % bars[0][0])

        ss = SG.SignalSet.from_dir(sigdir)
        sig_out = os.path.join(outdir, "signals.csv")
        summary = ms.run_scan(
            [{"asset": "equity", "symbol": "AAA"}],
            csv_dir=csvdir, out_dir=outdir,
            signals=ss, signals_out=sig_out)

        check("scan: the calendar blackout is reported in the summary",
              summary.get("signal_blackouts", 0) == 1,
              str(summary.get("signal_blackouts")))
        check("scan: signals.csv is written and readable",
              os.path.exists(sig_out) and len(SG.read_csv(sig_out)) == 40)
        check("scan: feeds are echoed in the summary",
              summary["signals"].get("calendar") == 1
              and summary["signals"].get("onchain") == 1)

        v = summary["symbols"]["AAA"]
        check("scan: exactly one bar is vetoed by the blackout",
              v.get("blackout_vetoed", 0) == 1, str(v.get("blackout_vetoed")))
        check("scan: the engine-facing CSV is still written",
              os.path.exists(v["csv"]))

        # The veto reaches the engine-facing CSV's `veto` column, so the C++
        # gate honours it without any C++ change.
        with open(v["csv"], "r", encoding="utf-8") as f:
            body = f.read()
        bar_line = ""
        for ln in body.splitlines():
            if ln.startswith("%d," % ev_ts):
                bar_line = ln
        check("scan: the feature CSV marks the blackout bar vetoed",
              bar_line.split(",")[2] == "1", bar_line)
        check("scan: detail names the veto and the signal layer",
              "event-blackout" in body and "sig[" in body)
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


def main():
    print("== monster external signals ==")
    test_readers()
    test_depth()
    test_onchain()
    test_social()
    test_calendar()
    test_commodities_and_pack()
    test_cross_alignment()
    test_scan_integration()
    print(f"\nRESULT: {PASSED} passed, {FAILED} failed")
    for f in FAILURES:
        print(f"   - {f}")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
