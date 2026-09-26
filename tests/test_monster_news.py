#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — tests/test_monster_news.py
#
#  Characterization tests for tools/monster/{news_hunter,state_vector}.py and
#  the tools/monster_scan.py CLI.
#
#  Proves: keyword weighting, exponential recency decay, sentiment alignment,
#  volume-anomaly detection, that an event-driven match requires BOTH halves,
#  the engine-facing CSV contract round-trips, and the scanner wires the whole
#  pipeline together offline.
#
#  Offline, stdlib-only.  Usage: python tests/test_monster_news.py
# =============================================================================
import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOOLS = os.path.join(ROOT, "tools")
sys.path.insert(0, TOOLS)

from monster import news_hunter as NH    # noqa: E402
from monster import state_vector as SV   # noqa: E402
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
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tol


spec = importlib.util.spec_from_file_location(
    "monster_scan", os.path.join(TOOLS, "monster_scan.py"))
ms = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ms)


def mk_bars(n, spike_at=None, spike=15.0, seed=12):
    import random
    rnd = random.Random(seed)
    bars, p = [], 100.0
    for i in range(n):
        drift = 0.0022 if (i // 50) % 2 == 0 else -0.0012
        p *= math.exp(drift + rnd.gauss(0, 0.009))
        # A little volume noise so the z-score is defined (a perfectly constant
        # series has zero variance, and the z-score is then 0 by definition).
        v = 1000.0 * (1.0 + rnd.random() * 0.2)
        if spike_at is not None and i == spike_at:
            v *= spike
        bars.append((1600000000 + i * 86400, p * 0.998, p * 1.012, p * 0.99,
                     p, v))
    return bars


# ---------------------------------------------------------------------------
# Keyword + sentiment
# ---------------------------------------------------------------------------
def test_keywords():
    print("\n-- keywords --")
    w, hits = NH.match_keywords("SEC approval for the spot ETF")
    check("keywords: 'sec approval' matched", "sec approval" in hits)
    check("keywords: highest weight wins", approx(w, 1.00))

    w2, _ = NH.match_keywords("Exchange HACK drains funds")
    check("keywords: hack is top weight", approx(w2, 1.00))

    w3, _ = NH.match_keywords("Whale Alert: large transfer")
    check("keywords: whale alert is a weaker signal", approx(w3, 0.60))

    w4, hits4 = NH.match_keywords("a quiet afternoon with nothing to report")
    check("keywords: no match -> weight 0 and no hits", w4 == 0.0 and not hits4)


def test_sentiment():
    print("\n-- sentiment --")
    check("sentiment: all positive -> +1",
          approx(NH.score_sentiment("beat surge rally record"), 1.0))
    check("sentiment: all negative -> -1",
          approx(NH.score_sentiment("miss plunge crash fraud"), -1.0))
    check("sentiment: neutral text -> 0",
          NH.score_sentiment("the market opened at nine") == 0.0)
    mixed = NH.score_sentiment("beat but miss")
    check("sentiment: mixed -> within (-1,1)", -1.0 < mixed < 1.0)


def test_ingest_and_decay():
    print("\n-- ingest / decay --")
    h = NH.NewsHunter({"tau": 300.0, "match_window": 300.0})
    n = h.ingest([
        {"ts": 1000, "title": "Nvidia earnings beat sparks a rally",
         "source": "x"},
        {"ts": 1000, "title": "weather report", "source": "x"},
    ])
    check("ingest: only market-moving headlines are kept", n == 1)
    check("ingest: event stored with keywords",
          h.events and h.events[0].keywords)

    ev = h.events[0]
    fresh = h.event_score(ev, 1000, +1)
    stale = h.event_score(ev, 1000 + 600, +1)
    check("decay: a fresh event scores higher than a stale one", fresh > stale)
    check("decay: one tau halves-ish (exp(-1))",
          approx(h.event_score(ev, 1000 + 300, +1), fresh * math.exp(-1.0),
                 1e-9))
    check("decay: beyond the match window -> 0 from best_event",
          h.best_event(1000 + 10_000, +1)[1] == 0.0)


def test_alignment():
    print("\n-- sentiment alignment --")
    h = NH.NewsHunter({"tau": 300.0})
    pos = NH.NewsEvent(ts=1000, weight=0.9, sentiment=+0.8)
    neg = NH.NewsEvent(ts=1000, weight=0.9, sentiment=-0.8)
    check("alignment: bullish news helps a long",
          h.event_score(pos, 1000, +1) > h.event_score(pos, 1000, -1))
    check("alignment: bullish news does NOT help a short",
          approx(h.event_score(pos, 1000, -1), 0.0))
    check("alignment: bearish news helps a short",
          h.event_score(neg, 1000, -1) > 0.0)
    neutral = NH.NewsEvent(ts=1000, weight=0.9, sentiment=0.0)
    check("alignment: neutral news is half credit",
          approx(h.event_score(neutral, 1000, +1), 0.9 * 0.5, 1e-9))


# ---------------------------------------------------------------------------
# Anomaly + the cross-reference
# ---------------------------------------------------------------------------
def test_anomaly():
    print("\n-- volume anomaly --")
    h = NH.NewsHunter({"vol_mult_min": 5.0, "vol_z_min": 3.0})
    bars = mk_bars(80, spike_at=70, spike=15.0)
    hit, mult, z = h.anomaly(bars, 70)
    check("anomaly: a 15x print is flagged", hit)
    check("anomaly: multiple is reported", mult > 10)
    check("anomaly: z is reported", z > 3.0)

    hit0, _, _ = h.anomaly(mk_bars(80), 70)
    check("anomaly: a normal bar is not flagged", not hit0)
    check("anomaly: index 0 has no history -> False",
          h.anomaly(bars, 0)[0] is False)


def test_match_requires_both():
    print("\n-- event match requires BOTH halves --")
    h = NH.NewsHunter({"tau": 300.0, "match_window": 300.0})
    bars = mk_bars(80, spike_at=70, spike=15.0)
    ts70 = bars[70][0]

    # anomaly only (no news)
    m = h.match(bars, 70, ts70, +1)
    check("match: anomaly without a headline is not event-driven",
          m.anomaly and not m.event_driven)

    # news only (no anomaly)
    h2 = NH.NewsHunter({"tau": 300.0, "match_window": 300.0})
    h2.ingest([{"ts": bars[70][0], "title": "SEC approval granted",
                "source": "x"}])
    m2 = h2.match(mk_bars(80), 70, bars[70][0], +1)
    check("match: a headline without an anomaly is not event-driven",
          not m2.anomaly and not m2.event_driven)

    # both
    h3 = NH.NewsHunter({"tau": 300.0, "match_window": 300.0})
    h3.ingest([{"ts": ts70, "title": "SEC approval granted", "source": "x"}])
    m3 = h3.match(bars, 70, ts70, +1)
    check("match: anomaly + fresh headline -> event-driven",
          m3.event_driven and m3.event_score > 0)
    check("match: the reason names the mechanism",
          "event-driven" in m3.reason)
    check("match: a stale headline does not match",
          not h3.match(bars, 70, ts70 + 100_000, +1).event_driven)


# ---------------------------------------------------------------------------
# State vector + CSV contract
# ---------------------------------------------------------------------------
def test_state_vector_csv():
    print("\n-- state vector / CSV contract --")
    cfg = SN.SniperConfig()
    bars = mk_bars(340, spike_at=1, spike=15.0)
    verdicts = SN.scan(bars, cfg=cfg)
    svs = [SV.build("AAPL", v, atr_pct=0.02) for v in verdicts]
    sv = SV.build("AAPL", verdicts[200], atr_pct=0.02)
    check("state: carries the symbol and ts",
          sv.symbol == "AAPL" and sv.ts == verdicts[200].ts)
    check("state: confidence matches the verdict",
          approx(sv.confidence, verdicts[200].score))
    check("state: risk_pct inside [1%,2%]",
          0.01 - 1e-9 <= sv.risk_pct <= 0.02 + 1e-9)
    check("state: detail is CSV-safe", "," not in SV.csv_safe(sv.detail))

    row = SV.to_row(sv)
    check("csv: row has 5 columns", len(row.split(",")) == 5)
    check("csv: veto encoded as 0/1", row.split(",")[2] in ("0", "1"))

    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "AAPL.csv")
        n = SV.write_feature_csv(path, svs)
        check("csv: wrote one row per bar", n == len(svs))
        back = SV.read_feature_csv(path)
        check("csv: round-trips every ts", len(back) == len(svs))
        first = back[svs[0].ts]
        check("csv: confidence round-trips", approx(first[0], svs[0].confidence,
                                                    1e-4))
        check("csv: veto round-trips", first[1] == svs[0].veto)
        check("csv: regime round-trips", first[2] == svs[0].regime)
        check("csv: missing file -> empty map",
              SV.read_feature_csv(os.path.join(tmp, "nope.csv")) == {})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# The scanner CLI
# ---------------------------------------------------------------------------
def _write_csv(path, bars):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("time,open,high,low,close,volume,source\n")
        for b in bars:
            f.write("%d,%.6f,%.6f,%.6f,%.6f,%.2f,fake\n" % b)


def test_scan_cli():
    print("\n-- monster_scan --")
    check("scan: parse_watch", ms.parse_watch("equity:AAPL,crypto:BTCUSDT")
          == [{"asset": "equity", "symbol": "AAPL"},
              {"asset": "crypto", "symbol": "BTCUSDT"}])
    try:
        ms.parse_watch("AAPL")
        check("scan: rejects a malformed watch item", False)
    except ValueError:
        check("scan: rejects a malformed watch item", True)

    tmp = tempfile.mkdtemp()
    try:
        csv_dir = os.path.join(tmp, "csv")
        out_dir = os.path.join(tmp, "out")
        # Two correlated-ish symbols so the cross-asset layer has data.
        _write_csv(os.path.join(csv_dir, "OIL_1d.csv"), mk_bars(340, seed=12))
        _write_csv(os.path.join(csv_dir, "COMMODITY_INDEX_1d.csv"),
                   mk_bars(340, seed=12))

        watch = [{"asset": "equity", "symbol": "OIL"},
                 {"asset": "equity", "symbol": "COMMODITY_INDEX"},
                 {"asset": "equity", "symbol": "MISSING"}]
        summary = ms.run_scan(watch, csv_dir=csv_dir, out_dir=out_dir,
                              timeframe="1d")
        check("scan: reports every watched symbol", len(summary["symbols"]) == 3)
        check("scan: a symbol with no CSV is reported, not crashed",
              summary["symbols"]["MISSING"].get("error") == "no-data")
        check("scan: wrote a feature CSV per data symbol",
              os.path.exists(os.path.join(out_dir, "OIL.csv")))
        check("scan: bar count matches the input",
              summary["symbols"]["OIL"]["bars"] == 340)
        check("scan: summary counts add up",
              summary["symbols"]["OIL"]["proposed"]
              + summary["symbols"]["OIL"]["vetoed"] <= 340)
        back = SV.read_feature_csv(os.path.join(out_dir, "OIL.csv"))
        check("scan: the emitted CSV is engine-readable", len(back) == 340)

        # News path: a spike + a matching headline must show up as event-driven.
        bars = mk_bars(340, spike_at=1, spike=15.0)
        _write_csv(os.path.join(csv_dir, "OIL_1d.csv"), bars)
        news = [{"ts": bars[1][0], "title": "Fed signals rate hike",
                 "source": "x"}]
        s2 = ms.run_scan([{"asset": "equity", "symbol": "OIL"}],
                         csv_dir=csv_dir, out_dir=out_dir, timeframe="1d",
                         news_items=news)
        check("scan: news items are ingested", s2["news_events"] == 1)
        check("scan: an event-driven bar is counted",
              s2["symbols"]["OIL"]["event_driven"] >= 1)
        check("scan: the engine-facing file is still written",
              os.path.exists(os.path.join(out_dir, "OIL.csv")))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    print("== monster news hunter ==")
    test_keywords()
    test_sentiment()
    test_ingest_and_decay()
    test_alignment()
    test_anomaly()
    test_match_requires_both()
    test_state_vector_csv()
    test_scan_cli()
    print(f"\nRESULT: {PASSED} passed, {FAILED} failed")
    for f in FAILURES:
        print(f"   - {f}")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
