#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — tests/test_paper_loop.py
#
#  Characterization tests for tools/paper_loop.py (M5 paper loop).
#
#  Fully offline and stdlib-only: the probe/fetch functions and the engine
#  runner are injected, so no network and no binary are needed.
#
#  Usage:  python tests/test_paper_loop.py
# =============================================================================
import importlib.util
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOOL = os.path.join(ROOT, "tools", "paper_loop.py")

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


spec = importlib.util.spec_from_file_location("paper_loop", TOOL)
pl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pl)


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------
NOW = [1700000000]


def fake_now():
    return NOW[0]


def ok_probe(asset, symbol, http, provider=None):
    tk = {"symbol": symbol, "provider": "fake", "last": 100.0,
          "change_pct_24h": 1.5, "volume_24h": 1e7, "ts": int(fake_now())}
    return tk, "ok", False, "none"


def stale_probe(asset, symbol, http, provider=None):
    tk = {"symbol": symbol, "provider": "fake", "last": 100.0,
          "change_pct_24h": 0.0, "volume_24h": 1e7,
          "ts": int(fake_now()) - pl.DEAD_AFTER_SECS - 10}
    return tk, "dead", True, "dead-feed"


def boom_probe(asset, symbol, http, provider=None):
    raise RuntimeError("provider down")


def fake_fetch(asset, symbol, tf, years, http, provider=None):
    rows = [(1700000000 + i * 86400, 100.0 + i, 101.0 + i, 99.0 + i,
             100.5 + i, 1e6) for i in range(5)]
    return rows, "fake", symbol


class Recorder:
    def __init__(self, result=None):
        self.calls = []
        self.result = result or {"ok": True, "timestamps": 5, "entries": 1,
                                 "exits": 1, "halts": 0, "abstains": 0,
                                 "refused": 0, "equity": 100010.0}

    def __call__(self, streams, skip):
        self.calls.append((streams, skip))
        return self.result


def make_loop(tmp, **kw):
    watch = kw.pop("watch", [{"asset": "equity", "symbol": "AAPL"},
                             {"asset": "crypto", "symbol": "BTCUSDT"}])
    return pl.PaperLoop(
        watch,
        journal=os.path.join(tmp, "journal.csv"),
        status=os.path.join(tmp, "status.json"),
        csv_dir=os.path.join(tmp, "csv"),
        now_fn=fake_now, **kw)


# ==========================================================================
# 1. watch parsing
# ==========================================================================
def test_parse_watch():
    w = pl.parse_watch("equity:AAPL,crypto:BTCUSDT,fx:EURUSD")
    check("parse_watch: three items", len(w) == 3)
    check("parse_watch: assets", [x["asset"] for x in w] ==
          ["equity", "crypto", "fx"])
    check("parse_watch: symbol", w[1]["symbol"] == "BTCUSDT")

    for bad in ["AAPL", "bogus:AAPL", ""]:
        try:
            pl.parse_watch(bad)
            check(f"parse_watch rejects {bad!r}", False)
        except ValueError:
            check(f"parse_watch rejects {bad!r}", True)


# ==========================================================================
# 2. CSV merge
# ==========================================================================
def test_csv_merge():
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "X_1d.csv")
        added, total = pl.merge_and_write(
            path, [(100, 1, 2, 0.5, 1.5, 10), (200, 2, 3, 1.5, 2.5, 20)],
            "fake")
        check("merge: two rows added", added == 2 and total == 2)
        # Re-merge the same ts plus one new -> only the new one counts.
        added2, total2 = pl.merge_and_write(
            path, [(200, 2, 3, 1.5, 2.5, 20), (300, 3, 4, 2.5, 3.5, 30)],
            "fake")
        check("merge: dedup by ts", added2 == 1 and total2 == 3)

        with open(path) as f:
            head = f.readline().strip()
            lines = f.read().strip().splitlines()
        check("merge: provenance header", head == "time,open,high,low,close,volume,source")
        check("merge: rows sorted", [int(l.split(",")[0]) for l in lines] ==
              [100, 200, 300])
        check("merge: source tagged", all(l.endswith(",fake") for l in lines))

        rows = pl.read_csv_rows(path)
        check("read_csv_rows: skips header", len(rows) == 3)
        check("read_csv_rows: keyed by ts", 200 in rows)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ==========================================================================
# 3. one cycle — healthy
# ==========================================================================
def test_cycle_healthy():
    tmp = tempfile.mkdtemp()
    cwd = os.getcwd()
    try:
        os.chdir(tmp)
        rec = Recorder()
        loop = make_loop(tmp, probe_fn=ok_probe, fetch_fn=fake_fetch,
                         runner=rec)
        st = loop.cycle(http=None)

        check("cycle: status written", os.path.exists(loop.status_path))
        check("cycle: two feeds", len(st["feeds"]) == 2)
        check("cycle: AAPL ok", st["feeds"]["AAPL"]["state"] == "ok")
        check("cycle: AAPL not abstaining",
              st["feeds"]["AAPL"]["abstain"] is False)
        check("cycle: last recorded", st["feeds"]["AAPL"]["last"] == 100.0)
        check("cycle: bars added", st["feeds"]["AAPL"]["bars_added"] == 5)

        check("cycle: engine called once", len(rec.calls) == 1)
        streams, skip = rec.calls[0]
        check("cycle: streams arg has AAPL", "AAPL,equity," in streams)
        check("cycle: streams arg has BTCUSDT", "BTCUSDT,crypto," in streams)
        check("cycle: nothing skipped", skip == "")
        check("cycle: last_session ok", st["last_session"]["ok"] is True)

        csv = os.path.join(tmp, "csv", "AAPL_1d.csv")
        check("cycle: csv written", os.path.exists(csv))
        check("cycle: csv has 5 bars", len(pl.read_csv_rows(csv)) == 5)

        # Second cycle with the same bars -> no new rows, still runs.
        st2 = loop.cycle(http=None)
        check("cycle: no new bars on refresh",
              st2["feeds"]["AAPL"]["bars_added"] == 0)
        check("cycle: cycle counter", st2["cycle"] == 2)

        on_disk = json.load(open(loop.status_path))
        check("status json: parseable + feeds", "feeds" in on_disk)
        check("status json: has last_session", "last_session" in on_disk)
        check("status json: updated_ts", on_disk["updated_ts"] == int(fake_now()))
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)


# ==========================================================================
# 4. failures -> dead -> skip
# ==========================================================================
def test_cycle_failure():
    tmp = tempfile.mkdtemp()
    cwd = os.getcwd()
    try:
        os.chdir(tmp)
        rec = Recorder()
        loop = make_loop(tmp, watch=[{"asset": "equity", "symbol": "AAPL"},
                                     {"asset": "crypto", "symbol": "BTCUSDT"}],
                         probe_fn=lambda a, s, h: (boom_probe(a, s, h)
                                                   if s == "AAPL"
                                                   else ok_probe(a, s, h)),
                         fetch_fn=fake_fetch, runner=rec)
        st = loop.cycle(http=None)
        check("failure: AAPL abstains", st["feeds"]["AAPL"]["abstain"] is True)
        check("failure: error captured", st["feeds"]["AAPL"]["error"] != "")
        for _ in range(2):
            st = loop.cycle(http=None)
        # Never succeeded -> "missing" wins over the failure count (the same
        # precedence the C++ FeedHealthMonitor uses), but it still abstains.
        check("failure: never-succeeded feed is missing",
              st["feeds"]["AAPL"]["state"] == "missing")
        check("failure: failures counted", st["feeds"]["AAPL"]["failures"] == 3)
        check("failure: missing-data reason",
              st["feeds"]["AAPL"]["reason"] == "missing-data")
        streams, skip = rec.calls[-1]
        check("failure: AAPL skipped", "AAPL" in skip)
        check("failure: AAPL not a stream", "AAPL,equity," not in streams)
        check("failure: no CSV for the dead feed",
              not os.path.exists(os.path.join(tmp, "csv", "AAPL_1d.csv")))
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)


def test_cycle_recovery_then_dead():
    """Succeed once (feed healthy), then fail repeatedly -> dead + skipped."""
    tmp = tempfile.mkdtemp()
    cwd = os.getcwd()
    try:
        os.chdir(tmp)
        state = {"fail": False}

        def flaky(asset, symbol, http, provider=None):
            if state["fail"]:
                raise RuntimeError("provider down")
            return ok_probe(asset, symbol, http)

        rec = Recorder()
        loop = make_loop(tmp, watch=[{"asset": "equity", "symbol": "AAPL"}],
                         probe_fn=flaky, fetch_fn=fake_fetch, runner=rec)
        st = loop.cycle(http=None)
        check("recovery: healthy first", st["feeds"]["AAPL"]["state"] == "ok")
        check("recovery: csv written",
              os.path.exists(os.path.join(tmp, "csv", "AAPL_1d.csv")))

        state["fail"] = True
        for _ in range(3):
            st = loop.cycle(http=None)
        check("recovery: 3 failures -> dead", st["feeds"]["AAPL"]["state"] == "dead")
        check("recovery: dead reason", st["feeds"]["AAPL"]["reason"] == "dead-feed")
        check("recovery: abstains after death",
              st["feeds"]["AAPL"]["abstain"] is True)
        # The symbol keeps its CSV, so it STAYS in the session (its open
        # position must still be marked and exitable) but is passed to --skip
        # so it opens no new risk.
        streams, skip = rec.calls[-1]
        check("recovery: still a stream (position stays managed)",
              "AAPL,equity," in streams)
        check("recovery: but skipped for new entries", "AAPL" in skip)
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)


# ==========================================================================
# 5. stale feed -> abstain
# ==========================================================================
def test_cycle_stale():
    tmp = tempfile.mkdtemp()
    cwd = os.getcwd()
    try:
        os.chdir(tmp)
        rec = Recorder()
        loop = make_loop(tmp, watch=[{"asset": "equity", "symbol": "AAPL"}],
                         probe_fn=stale_probe, fetch_fn=fake_fetch, runner=rec)
        st = loop.cycle(http=None)
        check("stale: abstains", st["feeds"]["AAPL"]["abstain"] is True)
        check("stale: reason dead-feed",
              st["feeds"]["AAPL"]["reason"] == "dead-feed")
        check("stale: session skipped (no fresh stream)",
              st["last_session"].get("skipped") is True)
        check("stale: engine not called", len(rec.calls) == 0)
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)


# ==========================================================================
# 6. dry-run + driver
# ==========================================================================
def test_dry_run_and_driver():
    tmp = tempfile.mkdtemp()
    cwd = os.getcwd()
    try:
        os.chdir(tmp)
        rec = Recorder()
        loop = make_loop(tmp, probe_fn=ok_probe, fetch_fn=fake_fetch,
                         runner=rec, dry_run=True)
        st = loop.cycle(http=None)
        check("dry-run: engine not called", len(rec.calls) == 0)
        check("dry-run: session flagged", st["last_session"].get("dry_run") is True)

        # Driver: cycles=3 -> sleep called twice, cycle counter reaches 3.
        loop2 = make_loop(tmp, probe_fn=ok_probe, fetch_fn=fake_fetch,
                          runner=rec, dry_run=True)
        sleeps = []
        loop2.run(cycles=3, interval=1, sleep_fn=lambda s: sleeps.append(s))
        check("driver: three cycles", loop2.cycle_no == 3)
        check("driver: sleeps between cycles", sleeps == [1, 1])
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    print("== paper loop (M5) ==")
    test_parse_watch()
    test_csv_merge()
    test_cycle_healthy()
    test_cycle_failure()
    test_cycle_recovery_then_dead()
    test_cycle_stale()
    test_dry_run_and_driver()
    print(f"\nRESULT: {PASSED} passed, {FAILED} failed")
    for f in FAILURES:
        print(f"   - {f}")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
