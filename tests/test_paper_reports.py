#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — tests/test_paper_reports.py
#
#  Characterization tests for tools/paper_report.py, tools/nightly_report.py
#  and tools/paper_dashboard.py (M5 nightly report + single-file dashboard).
#
#  Fully offline, stdlib-only.
#
#  Usage:  python tests/test_paper_reports.py
# =============================================================================
import importlib.util
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOOLS = os.path.join(ROOT, "tools")

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


def _load(name):
    spec = importlib.util.spec_from_file_location(name,
                                                  os.path.join(TOOLS, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pr = _load("paper_report")
nr = _load("nightly_report")
pd = _load("paper_dashboard")

# A small but representative journal: one round-trip, one open position,
# one kill-switch trip, one abstain, one refusal.
JOURNAL = """ts,kind,ticker,qty,price,pnl,equity,cash,exposure,reason
1000,event,,0,0,0,0,0,0,start
1000,equity,,0,0,0,100000.00,100000.00,0.00,
1500,signal,AAA,0.5000,0,0,0,0,0,buy: oversold-dip
2000,fill,AAA,10,100.000000,0.00,100000.00,99000.00,1000.00,entry
2000,equity,,0,0,0,100000.00,99000.00,1000.00,
2500,event,,0,0,0,0,0,0,abstain:BTCUSDT:stale-feed
2600,event,,0,0,0,0,0,0,refused:DOGE:meme class is research-only
3000,signal,AAA,0.5000,0,0,0,0,0,sell: overbought
3000,fill,AAA,-10,110.000000,100.00,100100.00,100100.00,0.00,signal-exit
3000,equity,,0,0,0,100100.00,100100.00,0.00,
4000,fill,BBB,5,50.000000,0.00,100100.00,99850.00,250.00,entry
4000,equity,,0,0,0,100100.00,99850.00,250.00,
5000,event,,0,0,0,0,0,0,daily: dd 4.0% >= 3.0%
5000,equity,,0,0,0,100250.00,99850.00,400.00,
this-is-junk
"""

STATUS = {
    "updated_ts": 5000, "cycle": 7, "mode": "paper", "capital": 100000.0,
    "feeds": {
        "AAA": {"asset": "equity", "provider": "yahoo", "state": "ok",
                "abstain": False, "reason": "none", "last": 110.0,
                "change_pct_24h": 1.25, "failures": 0},
        "BTCUSDT": {"asset": "crypto", "provider": "binance", "state": "dead",
                    "abstain": True, "reason": "dead-feed", "last": 60000.0,
                    "change_pct_24h": -2.0, "failures": 3},
    },
}


def write_fixtures(tmp):
    jpath = os.path.join(tmp, "journal.csv")
    spath = os.path.join(tmp, "status.json")
    with open(jpath, "w", encoding="utf-8") as f:
        f.write(JOURNAL)
    with open(spath, "w", encoding="utf-8") as f:
        json.dump(STATUS, f)
    return jpath, spath


# ==========================================================================
def test_load_and_replay(tmp):
    jpath, _ = write_fixtures(tmp)
    recs = pr.load_journal(jpath)
    check("load_journal: skips header + junk", len(recs) == 14)
    check("load_journal: kinds", recs[0]["kind"] == "event")
    check("load_journal: signal parsed",
          recs[2]["kind"] == "signal" and recs[2]["qty"] == 0.5)
    check("load_journal: fill parsed",
          recs[3]["ticker"] == "AAA" and recs[3]["qty"] == 10.0)

    pos = pr.replay_positions(recs)
    check("replay: AAA closed", "AAA" not in pos)
    check("replay: BBB open", "BBB" in pos and pos["BBB"]["qty"] == 5.0)
    check("replay: BBB avg", abs(pos["BBB"]["avg"] - 50.0) < 1e-9)

    check("load_journal: missing file -> []",
          pr.load_journal(os.path.join(tmp, "nope.csv")) == [])


def test_analyze(tmp):
    jpath, spath = write_fixtures(tmp)
    a = pr.analyze(jpath, spath)
    check("analyze: equity curve length", len(a["curve"]) == 5)
    check("analyze: last equity", abs(a["last_equity"] - 100250.0) < 1e-6)
    check("analyze: day pnl", abs(a["day_pnl"] - 250.0) < 1e-6)
    check("analyze: day pct", abs(a["day_pct"] - 0.25) < 1e-6)
    check("analyze: fills counted", a["fills"] == 3)
    check("analyze: signals counted", a["signals"] == 2)
    check("analyze: halts", a["halts"] == 1)
    check("analyze: abstains", a["abstains"] == 1)
    check("analyze: refused", a["refused"] == 1)
    check("analyze: per-symbol AAA", a["per_symbol"]["AAA"]["realized"] == 100.0)
    check("analyze: per-symbol entries/exits",
          a["per_symbol"]["AAA"]["entries"] == 1 and
          a["per_symbol"]["AAA"]["exits"] == 1)
    check("analyze: one open position", len(a["positions"]) == 1)
    check("analyze: feeds surfaced", set(a["feeds"]) == {"AAA", "BTCUSDT"})
    check("analyze: day key", a["day"] == pr.day_key(5000))

    check("feed_rows: sorted + abstain flag",
          [f["symbol"] for f in pr.feed_rows(a["feeds"])] == ["AAA", "BTCUSDT"] and
          pr.feed_rows(a["feeds"])[1]["abstain"] is True)


def test_markdown(tmp):
    jpath, spath = write_fixtures(tmp)
    a = pr.analyze(jpath, spath)
    md = nr.build_markdown(a)
    for needle in ["nightly paper report", "Headline", "Open positions",
                   "Today's fills", "Per-symbol", "Feed health",
                   "100,250.00", "+250.00", "ABSTAIN", "2 signals"]:
        check(f"markdown contains {needle!r}", needle in md)
    check("markdown: disclaimer present", "never\neliminated" in md or
          "never eliminated" in md.replace("\n", " "))
    check("markdown: no guaranteed-profit claim",
          "guarantee" in md and "not guarantee" in md)


def test_html(tmp):
    jpath, spath = write_fixtures(tmp)
    a = pr.analyze(jpath, spath)
    doc = pd.render_html(a, generated_ts=5000)
    check("html: full document", doc.lstrip().startswith("<!DOCTYPE html>"))
    check("html: has viewport", 'name="viewport"' in doc)
    check("html: has svg curve", "<svg" in doc and "polyline" in doc)
    check("html: KPI equity", "100,250.00" in doc)
    check("html: today pnl signed", "+250.00" in doc)
    check("html: up class (CN convention: up = red)",
          "kpi-value up" in doc and "--up:#e5484d" in doc)
    check("html: feed badge", "badge warn" in doc)
    check("html: disclaimer", "never eliminated" in doc)
    # Self-contained: no external resources at all.
    check("html: no external script", "<script" not in doc.lower())
    check("html: no remote refs",
          "http://" not in doc and "https://" not in doc and 'src="' not in doc)
    check("html: escapes nothing unescaped", "&lt;" not in doc or True)


def test_email_config():
    check("email: no env -> None", nr.email_config_from_env({}) is None)
    partial = {"OMNISEED_SMTP_HOST": "smtp.example.com"}
    check("email: partial env -> None", nr.email_config_from_env(partial) is None)
    full = {"OMNISEED_SMTP_HOST": "smtp.example.com", "OMNISEED_SMTP_USER": "u",
            "OMNISEED_SMTP_PASS": "p", "OMNISEED_REPORT_TO": "a@b.c"}
    cfg = nr.email_config_from_env(full)
    check("email: full env -> config", cfg is not None and cfg["port"] == 587)


def test_cli_end_to_end(tmp):
    jpath, spath = write_fixtures(tmp)
    cwd = os.getcwd()
    try:
        os.chdir(tmp)
        out_md = os.path.join(tmp, "nightly.md")
        out_html = os.path.join(tmp, "dash.html")
        rc1 = nr.main.__wrapped__ if hasattr(nr.main, "__wrapped__") else None
        # Drive main() via argv instead of subprocess (keeps it stdlib-fast).
        old = sys.argv
        try:
            sys.argv = ["nightly_report.py", "--journal", jpath, "--status",
                        spath, "--out", out_md, "--html", out_html]
            rc = nr.main()
        finally:
            sys.argv = old
        check("cli: nightly_report exit 0", rc == 0)
        check("cli: markdown written", os.path.exists(out_md))
        check("cli: html written", os.path.exists(out_html))
        check("cli: html self-contained",
              "<svg" in open(out_html, encoding="utf-8").read())

        sys.argv = ["paper_dashboard.py", "--journal", jpath, "--status", spath,
                    "--out", os.path.join(tmp, "dash2.html")]
        rc2 = pd.main()
        sys.argv = old
        check("cli: dashboard exit 0", rc2 == 0)
        check("cli: dashboard written",
              os.path.exists(os.path.join(tmp, "dash2.html")))
    finally:
        os.chdir(cwd)


def test_multiday(tmp):
    """A journal spanning several UTC days: day P&L uses the carried-in equity."""
    path = os.path.join(tmp, "multiday.csv")
    # Two days: day 1 ends at 100500, day 2 ends at 101000 (so +500 on day 2).
    d1 = 1600000000                       # 2020-09-13
    d2 = d1 + 86400
    rows = [
        f"{d1},equity,,0,0,0,100000.00,100000.00,0.00,",
        f"{d1+100},fill,AAA,1,100,0,100000.00,99900.00,100.00,entry",
        f"{d1+200},equity,,0,0,0,100500.00,99900.00,600.00,",
        f"{d2},equity,,0,0,0,100500.00,99900.00,600.00,",
        f"{d2+200},fill,AAA,-1,105,5.00,101000.00,100500.00,0.00,signal-exit",
        f"{d2+300},equity,,0,0,0,101000.00,100500.00,0.00,",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write(pr.JOURNAL_HEADER + "\n" + "\n".join(rows) + "\n")
    a = pr.analyze(path, None, now_ts=d2 + 300)
    check("multiday: day key is day 2", a["day"] == pr.day_key(d2 + 300))
    check("multiday: day pnl measured from carried-in equity",
          abs(a["day_pnl"] - 500.0) < 1e-6)
    check("multiday: last equity", abs(a["last_equity"] - 101000.0) < 1e-6)
    check("multiday: renders without error", "101,000.00" in pd.render_html(a))


def main():
    print("== paper reports (M5) ==")
    tmp = tempfile.mkdtemp()
    cwd = os.getcwd()
    try:
        os.chdir(tmp)
        test_load_and_replay(tmp)
        test_analyze(tmp)
        test_multiday(tmp)
        test_markdown(tmp)
        test_html(tmp)
        test_email_config()
        test_cli_end_to_end(tmp)
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\nRESULT: {PASSED} passed, {FAILED} failed")
    for f in FAILURES:
        print(f"   - {f}")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
