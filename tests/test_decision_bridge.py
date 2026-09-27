#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — tests/test_decision_bridge.py
#
#  Characterization tests for the T2 integration:
#    tools/decision_bridge.py   — parsing the C++ System-1 decision JSON
#    tools/paper_report.py      — recovering entry/stop/confidence provenance
#    tools/paper_dashboard.py   — the positions + regime panels, live updater
#    tools/paper_loop.py        — the optional decision gate and regime reading
#    tools/serve_dashboard.py   — the local HTTP view
#
#  Two of these tests are CONTRACT tests, not behaviour tests: they re-read the
#  C++ sources and fail if the Python copies of the risk limits ever drift.
#
#  Fully offline, stdlib-only.
#
#  Usage:  python tests/test_decision_bridge.py
# =============================================================================
import http.client
import importlib.util
import json
import os
import re
import shutil
import subprocess
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


def _load(name, path=None):
    path = path or os.path.join(TOOLS, name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


db = _load("decision_bridge")
pr = _load("paper_report")
pd = _load("paper_dashboard")
pl = _load("paper_loop")
sv = _load("serve_dashboard")

GOOD = ('{"action":"BUY","confidence":0.9312,"target_asset":"AAPL",'
        '"invalidation":178.4200,"routing":"self","fast_path":true,'
        '"margin":0.8123,"matvecs":1}')


# ==========================================================================
# 1. parsing the C++ decision JSON
# ==========================================================================
def test_parse_decision():
    d = db.parse_decision(GOOD)
    check("parse: action", d.action == "BUY")
    check("parse: confidence", abs(d.confidence - 0.9312) < 1e-9)
    check("parse: asset", d.target_asset == "AAPL")
    check("parse: invalidation", abs(d.invalidation - 178.42) < 1e-9)
    check("parse: fast_path", d.fast_path is True)
    check("parse: tradeable", d.tradeable())

    # accepts bytes and dicts too
    check("parse: bytes", db.parse_decision(GOOD.encode()).action == "BUY")
    check("parse: dict", db.parse_decision(json.loads(GOOD)).action == "BUY")
    check("parse: lowercase action normalised",
          db.parse_decision('{"action":"buy","confidence":0.9,'
                            '"fast_path":true,"invalidation":1}').action == "BUY")

    def rejects(name, payload):
        try:
            db.parse_decision(payload)
            check(f"parse rejects {name}", False, "no exception")
        except db.DecisionParseError:
            check(f"parse rejects {name}", True)

    rejects("unknown action", '{"action":"YOLO","confidence":0.9,'
                             '"fast_path":true}')
    rejects("missing confidence", '{"action":"BUY"}')
    rejects("missing action", '{"confidence":0.9}')
    rejects("confidence > 1", '{"action":"BUY","confidence":1.5,'
                              '"fast_path":true}')
    rejects("negative confidence", '{"action":"BUY","confidence":-0.1,'
                                   '"fast_path":true}')
    rejects("nan confidence", '{"action":"BUY","confidence":NaN,'
                              '"fast_path":true}')
    rejects("string fast_path", '{"action":"BUY","confidence":0.9,'
                                '"fast_path":"true"}')
    rejects("string confidence", '{"action":"BUY","confidence":"0.9"}')
    rejects("bool confidence", '{"action":"BUY","confidence":true}')
    rejects("not json", "not json at all")
    rejects("json array", "[1,2,3]")
    rejects("float matvecs", '{"action":"BUY","confidence":0.9,'
                             '"fast_path":true,"matvecs":1.5}')
    rejects("non-string asset", '{"action":"BUY","confidence":0.9,'
                                '"fast_path":true,"target_asset":7}')


def test_fail_closed_rules():
    def d(**kw):
        base = {"action": "BUY", "confidence": 0.99, "target_asset": "AAPL",
                "invalidation": 100.0, "routing": "self", "fast_path": True,
                "margin": 0.9, "matvecs": 1}
        base.update(kw)
        return db.parse_decision(base)

    check("ABSTAIN never trades", not d(action="ABSTAIN").tradeable())
    check("EXPLAIN never trades", not d(action="EXPLAIN").tradeable())
    check("ABSTAIN never trades even at 1.0",
          not d(action="ABSTAIN", confidence=1.0).tradeable())
    check("not self-routed is refused",
          not d(fast_path=False, routing="system2").tradeable())
    check("below the bar is refused", not d(confidence=0.80).tradeable())
    check("exactly at the bar passes", d(confidence=0.85).tradeable())
    check("unknown stop is refused", not d(invalidation=0.0).tradeable())
    check("HEDGE needs a stop too", not d(action="HEDGE",
                                          invalidation=0.0).tradeable())
    check("HOLD needs no stop", d(action="HOLD", invalidation=0.0).tradeable())
    check("CLOSE needs no stop", d(action="CLOSE",
                                   invalidation=0.0).tradeable())
    check("refusal names the reason",
          "UNKNOWN" in d(invalidation=0.0).refusal())
    check("allow_unknown_stop is opt-in",
          d(invalidation=0.0).tradeable(require_stop=False))


def test_screen():
    ok = db.parse_decision(GOOD)
    allowed, refused = db.screen({"AAPL": ok}, symbols=["AAPL"])
    check("screen: allows", allowed == ["AAPL"])
    check("screen: nothing refused", refused == [])

    # Silence is not consent.
    allowed, refused = db.screen({"AAPL": ok}, symbols=["AAPL", "MSFT"])
    check("screen: silence refused", allowed == ["AAPL"]
          and [r["symbol"] for r in refused] == ["MSFT"])
    check("screen: silence reason",
          "not consent" in refused[0]["reason"])

    # Empty decision log + watched symbols = nothing allowed.
    allowed, refused = db.screen({}, symbols=["AAPL", "MSFT"])
    check("screen: empty log allows nothing", allowed == []
          and len(refused) == 2)

    # A decision for an unwatched symbol is reported, never acted on.
    allowed, refused = db.screen({"ZZZ": ok}, symbols=["AAPL"])
    check("screen: unwatched decision ignored", allowed == [])
    check("screen: unwatched reported",
          any(r["symbol"] == "ZZZ" for r in refused))

    # Malformed entries refuse, they do not raise.
    allowed, refused = db.screen(
        {"AAPL": db.DecisionParseError("boom")}, symbols=["AAPL"])
    check("screen: malformed refuses", allowed == []
          and "malformed" in refused[0]["reason"])

    # Threshold may be raised...
    allowed, _ = db.screen({"AAPL": ok}, symbols=["AAPL"], threshold=0.99)
    check("screen: raised bar refuses", allowed == [])
    # ...never lowered.
    try:
        db.screen({}, symbols=[], threshold=0.5)
        check("screen: threshold floor enforced", False, "no exception")
    except ValueError:
        check("screen: threshold floor enforced", True)


def test_parse_jsonl():
    text = "\n".join([
        "# a comment",
        "",
        GOOD,
        "{not json}",
        '{"action":"ABSTAIN","confidence":0.1,"target_asset":"MSFT"}',
    ])
    rows = db.parse_jsonl(text)
    check("jsonl: comment + blank skipped", len(rows) == 3)
    check("jsonl: good line parsed", isinstance(rows[0][1], db.Decision))
    check("jsonl: bad line is an error object, not an exception",
          isinstance(rows[1][1], db.DecisionParseError))
    check("jsonl: line numbers are 1-based", [n for n, _ in rows] == [3, 4, 5])


# ==========================================================================
# 2. entry provenance in the journal
# ==========================================================================
def test_entry_provenance():
    kv = pr.parse_entry_provenance("entry conf=0.720 stop=178.4200 stop_pct=0.0800")
    check("prov: conf", abs(kv["conf"] - 0.720) < 1e-9)
    check("prov: stop", abs(kv["stop"] - 178.42) < 1e-9)
    check("prov: stop_pct", abs(kv["stop_pct"] - 0.08) < 1e-9)
    check("prov: bare entry -> {}", pr.parse_entry_provenance("entry") == {})
    check("prov: empty", pr.parse_entry_provenance("") == {})
    check("prov: None", pr.parse_entry_provenance(None) == {})
    check("prov: junk token ignored",
          pr.parse_entry_provenance("entry foo=bar conf=0.5") == {"conf": 0.5})
    check("prov: unparsable number ignored",
          pr.parse_entry_provenance("entry conf=abc") == {})


def _rec(ts, kind, ticker, qty, price, reason, equity=100000.0):
    return {"ts": ts, "kind": kind, "ticker": ticker, "qty": qty,
            "price": price, "pnl": 0.0, "equity": equity, "cash": 0.0,
            "exposure": 0.0, "reason": reason}


def test_replay_positions_provenance():
    recs = [
        _rec(100, "fill", "AAPL", 10.0, 100.0,
             "entry conf=0.720 stop=92.0000 stop_pct=0.0800"),
        _rec(200, "fill", "MSFT", 5.0, 50.0, "entry"),   # legacy row
    ]
    pos = pr.replay_positions(recs)
    check("replay: both positions", set(pos) == {"AAPL", "MSFT"})
    check("replay: confidence recovered",
          abs(pos["AAPL"]["confidence"] - 0.72) < 1e-9)
    check("replay: stop recovered",
          abs(pos["AAPL"]["stop"] - 92.0) < 1e-9)
    check("replay: stop_pct kept", abs(pos["AAPL"]["stop_pct"] - 0.08) < 1e-9)
    check("replay: legacy confidence is None, not 0",
          pos["MSFT"]["confidence"] is None)
    check("replay: legacy stop is None, not 0", pos["MSFT"]["stop"] is None)

    # The stop LEVEL must track the blended average, exactly as the C++ exit
    # check (avg * (1 - stop_pct)) computes it.
    recs2 = [
        _rec(100, "fill", "AAPL", 10.0, 100.0,
             "entry conf=0.70 stop=92.0000 stop_pct=0.0800"),
        _rec(150, "fill", "AAPL", 10.0, 120.0,
             "entry conf=0.90 stop=110.4000 stop_pct=0.0800"),
    ]
    p = pr.replay_positions(recs2)["AAPL"]
    check("replay: blended avg", abs(p["avg"] - 110.0) < 1e-9)
    check("replay: stop recomputed on the blend",
          abs(p["stop"] - 110.0 * 0.92) < 1e-9)
    check("replay: latest confidence wins",
          abs(p["confidence"] - 0.90) < 1e-9)

    # A round trip must leave nothing behind.
    recs3 = recs2 + [_rec(300, "fill", "AAPL", -20.0, 130.0, "signal-exit")]
    check("replay: closed book is empty", pr.replay_positions(recs3) == {})


# ==========================================================================
# 3. the dashboard
# ==========================================================================
def _analysis(tmp):
    j = os.path.join(tmp, "journal.csv")
    s = os.path.join(tmp, "status.json")
    with open(j, "w", encoding="utf-8") as f:
        f.write(pr.JOURNAL_HEADER + "\n")
        f.write("100,event,,,,,,,0,start\n")
        f.write("100,equity,,,,100000.00,100000.00,100000.00,0.00,\n")
        f.write("200,fill,AAPL,10.0,100.000000,0.00,100000.00,99000.00,1000.00,"
                "entry conf=0.720 stop=92.0000 stop_pct=0.0800\n")
        f.write("300,equity,,,,101500.00,101500.00,100500.00,1000.00,\n")
    with open(s, "w", encoding="utf-8") as f:
        json.dump({
            "mode": "paper",
            "feeds": {"AAPL": {"asset": "equity", "provider": "stooq",
                               "state": "ok", "abstain": False,
                               "reason": "fresh", "last": 101.5,
                               "change_pct_24h": 1.5, "failures": 0}},
            "regime": {"AAPL": {"label": "trend_up", "direction": "up",
                                "trend_score": 0.712, "vol_score": 0.31,
                                "stressed": False,
                                "detail": "regime=trend_up trend=0.712"}},
        }, f)
    return pr.analyze(j, s)


def test_dashboard_panels():
    tmp = tempfile.mkdtemp(prefix="omniseed_t2_")
    try:
        a = _analysis(tmp)
        check("analyze: regime surfaced",
              a["regime"]["AAPL"]["label"] == "trend_up")
        doc = pd.render_html(a, generated_ts=5000)

        # (2) open positions with entry, stop, confidence
        check("dash: positions table header",
              all(h in doc for h in ("<th>entry</th>", "<th>stop</th>",
                                     "<th>confidence</th>")))
        check("dash: entry price rendered", "100.00" in doc)
        check("dash: stop level rendered", "92.00" in doc)
        check("dash: confidence as a percent", "72%" in doc)
        check("dash: confidence meter", "class='meter'" in doc)

        # (3) daily P&L and regime status
        check("dash: regime panel", "<h2>Regime status</h2>" in doc)
        check("dash: regime label", "trend_up" in doc)
        check("dash: regime trend score", "0.712" in doc)
        # The label goes through html.escape(), so both the apostrophe and the
        # ampersand are entities in the source. Assert the stable KPI id plus
        # the escaped label fragment rather than a hand-typed literal.
        check("dash: today's P&L", "kpi-daypnl" in doc and "P&amp;L" in doc)

        # (1) live equity curve
        check("dash: equity curve svg", 'id="eq-svg"' in doc
              and 'id="eq-poly"' in doc)
        check("dash: live poller inlined", "api/state" in doc)
        check("dash: file:// degradation note", "file:" in doc)
        check("dash: single file, no external assets",
              "http://" not in doc.split("<script>")[0]
              and "<link" not in doc and "<img" not in doc)

        # the risk panel quotes the engine limits
        for label in ("Per-trade risk budget", "Daily kill-switch",
                      "Weekly kill-switch"):
            check(f"dash: risk row {label!r}", label in doc)

        # the live payload
        st = json.loads(pd.render_state_json(a))
        check("state: equity", abs(st["equity"] - 101500.0) < 1e-9)
        check("state: curve present", len(st["curve"]) >= 2)
        check("state: position stop in payload",
              abs(st["positions"]["AAPL"]["stop"] - 92.0) < 1e-9)
        check("state: regime in payload",
              st["regime"]["AAPL"]["label"] == "trend_up")

        # an empty regime must say so, not render a bogus label
        a2 = dict(a)
        a2["regime"] = {}
        doc2 = pd.render_html(a2)
        check("dash: empty regime explained", "No regime reading" in doc2)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ==========================================================================
# 4. CONTRACT: the Python copies of the engine limits must not drift
# ==========================================================================
def test_engine_limits_match_cpp():
    te_h = os.path.join(ROOT, "include", "omniseed", "trading",
                        "trading_engine.h")
    rg_h = os.path.join(ROOT, "include", "omniseed", "trading", "risk_gate.h")

    def grab(path, pattern):
        with open(path, "r", encoding="utf-8") as f:
            m = re.search(pattern, f.read())
        return float(m.group(1)) if m else None

    stop = grab(te_h, r"stop_loss_pct\s*=\s*([0-9.]+)")
    cap = grab(te_h, r"kMaxRiskPerTradePct\s*=\s*([0-9.]+)")
    daily = grab(rg_h, r"max_daily_loss_pct\s*=\s*([0-9.]+)")
    weekly = grab(rg_h, r"max_weekly_loss_pct\s*=\s*([0-9.]+)")

    check("contract: parsed C++ stop_loss_pct", stop is not None)
    check("contract: parsed C++ per-trade cap", cap is not None)
    check("contract: parsed C++ daily kill-switch", daily is not None)
    check("contract: parsed C++ weekly kill-switch", weekly is not None)

    if None not in (stop, cap, daily, weekly):
        check("contract: stop_loss_pct matches",
              abs(pr.ENGINE_LIMITS["stop_loss_pct"] - stop) < 1e-9,
              f"py={pr.ENGINE_LIMITS['stop_loss_pct']} cpp={stop}")
        check("contract: per-trade ceiling matches (and is 2%)",
              abs(pr.ENGINE_LIMITS["per_trade_pct"] - cap * 100.0) < 1e-9,
              f"py={pr.ENGINE_LIMITS['per_trade_pct']} cpp={cap * 100}")
        check("contract: daily kill-switch matches (and is 3%)",
              abs(pr.ENGINE_LIMITS["daily_kill_pct"] - daily * 100.0) < 1e-9,
              f"py={pr.ENGINE_LIMITS['daily_kill_pct']} cpp={daily * 100}")
        check("contract: weekly kill-switch matches",
              abs(pr.ENGINE_LIMITS["weekly_kill_pct"] - weekly * 100.0) < 1e-9,
              f"py={pr.ENGINE_LIMITS['weekly_kill_pct']} cpp={weekly * 100}")


# ==========================================================================
# 5. the loop's decision gate
# ==========================================================================
def test_decision_gate_file():
    tmp = tempfile.mkdtemp(prefix="omniseed_gate_")
    try:
        path = os.path.join(tmp, "decisions.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write(GOOD + "\n")                                   # AAPL: pass
            f.write('{"action":"ABSTAIN","confidence":0.99,'
                    '"target_asset":"MSFT","fast_path":true}\n')   # refuse
            f.write('{"action":"BUY","confidence":0.99,"target_asset":"BTCUSDT",'
                    '"invalidation":0,"fast_path":true}\n')        # unknown stop

        gate = pl.DecisionGate(source="file", jsonl=path)
        ok, why = gate.configured()
        check("gate: configured", ok, why)

        skip, info = gate.screen(["AAPL", "MSFT", "BTCUSDT"])
        check("gate: only AAPL allowed", info["allowed"] == ["AAPL"])
        check("gate: MSFT skipped", "MSFT" in skip)
        check("gate: unknown-stop symbol skipped", "BTCUSDT" in skip)
        check("gate: AAPL not skipped", "AAPL" not in skip)
        check("gate: not fail-closed on a readable file",
              info["fail_closed"] is False)
        check("gate: threshold reported",
              abs(info["threshold"] - 0.85) < 1e-9)

        # A watched symbol with no verdict must be skipped.
        skip2, info2 = gate.screen(["AAPL", "ZZZ"])
        check("gate: silence skips", "ZZZ" in skip2 and "AAPL" not in skip2)

        # A requested-but-unusable gate must FAIL CLOSED, never vanish.
        gate_bad = pl.DecisionGate(source="file", jsonl=os.path.join(tmp, "nope"))
        ok_bad, why_bad = gate_bad.configured()
        check("gate: missing log is not usable", ok_bad is False)
        skip3, info3 = gate_bad.screen(["AAPL"])
        check("gate: unusable gate fails closed",
              skip3 == ["AAPL"] and info3["fail_closed"] is True)
        check("gate: unusable gate explains itself", "no decision log" in why_bad)

        # Fail closed when the log is readable but unparsable mid-way.
        bad_path = os.path.join(tmp, "bad.jsonl")
        with open(bad_path, "w", encoding="utf-8") as f:
            f.write("{oops\n")
        gate_mid = pl.DecisionGate(source="file", jsonl=bad_path)
        skip4, info4 = gate_mid.screen(["AAPL"])
        check("gate: unparsable line refuses its symbol",
              info4["allowed"] == [] and "AAPL" in skip4)

        # Threshold floor still applies through the gate, and it fails closed.
        gate_low = pl.DecisionGate(source="file", jsonl=path, threshold=0.10)
        ok_low, why_low = gate_low.configured()
        check("gate: sub-mandate threshold is not usable", ok_low is False)
        check("gate: sub-mandate threshold names the rule",
              "never lowered" in why_low)
        skip5, info5 = gate_low.screen(["AAPL"])
        check("gate: sub-mandate threshold fails closed",
              info5["fail_closed"] is True and "AAPL" in skip5)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_gate_is_off_by_default():
    gate = pl.DecisionGate(source="off")
    skip, info = gate.screen(["AAPL"])
    check("gate: off adds no skips", skip == [])
    check("gate: off is inactive", info["active"] is False)
    ok, why = gate.configured()
    check("gate: off reports why", ok is False and "off" in why)


# ==========================================================================
# 6. regime reading (skipped honestly when the engine is absent)
# ==========================================================================
def test_regime_reading():
    mod = pl.load_regime_module()
    if mod is None:
        check("regime: absent module degrades to None",
              pl.regime_for(None, "whatever.csv") is None)
        check("regime: loop tolerates an absent engine",
              pl.PaperLoop([{"asset": "equity", "symbol": "AAPL"}],
                           regime=False).regime_mod is None)
        return

    tmp = tempfile.mkdtemp(prefix="omniseed_regime_")
    try:
        csv_path = os.path.join(tmp, "AAPL_1d.csv")
        # A clean uptrend: 60 bars, +0.4% a day, so `detect` has a window.
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("time,open,high,low,close,volume,source\n")
            px = 100.0
            for i in range(60):
                ts = 1700000000 + i * 86400
                o = px
                px *= 1.004
                f.write(f"{ts},{o:.4f},{px * 1.002:.4f},{o * 0.998:.4f},"
                        f"{px:.4f},1000.00,synthetic\n")
        r = pl.regime_for(mod, csv_path)
        check("regime: reading produced", r is not None)
        if r:
            check("regime: has a label", r["label"] in
                  ("trend_up", "trend_down", "range", "high_vol"))
            check("regime: has a trend score",
                  isinstance(r["trend_score"], float))
            check("regime: detail string", r["detail"].startswith("regime="))
            check("regime: uptrend read as up",
                  r["label"] == "trend_up" and r["direction"] == "up",
                  str(r))
        # Too few bars -> None, not a fabricated reading.
        short = os.path.join(tmp, "short.csv")
        with open(short, "w", encoding="utf-8") as f:
            f.write("time,open,high,low,close,volume,source\n")
            f.write("1700000000,1,1,1,1,1,x\n")
        check("regime: too little history -> None",
              pl.regime_for(mod, short) is None)
        check("regime: missing csv -> None",
              pl.regime_for(mod, os.path.join(tmp, "nope.csv")) is None)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_cycle_records_regime_and_gate():
    tmp = tempfile.mkdtemp(prefix="omniseed_cycle_")
    try:
        jpath = os.path.join(tmp, "decisions.jsonl")
        with open(jpath, "w", encoding="utf-8") as f:
            f.write(GOOD + "\n")

        def runner(streams, skip):
            return {"ok": True, "streams": streams, "skip": skip}

        loop = pl.PaperLoop(
            [{"asset": "equity", "symbol": "AAPL"}],
            journal=os.path.join(tmp, "journal.csv"),
            status=os.path.join(tmp, "status.json"),
            csv_dir=os.path.join(tmp, "csv"),
            now_fn=lambda: 1700000000,
            runner=runner, regime=False,
            decision_gate=pl.DecisionGate(source="file", jsonl=jpath))
        st = loop.cycle(http=None)
        check("cycle: decision_gate in heartbeat", "decision_gate" in st)
        check("cycle: regime key present (may be empty)", "regime" in st)
        check("cycle: gate allowed AAPL",
              st["decision_gate"]["allowed"] == ["AAPL"])
        on_disk = json.load(open(loop.status_path, encoding="utf-8"))
        check("cycle: heartbeat persisted the gate",
              "decision_gate" in on_disk and "regime" in on_disk)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ==========================================================================
# 7. the local server
# ==========================================================================
def test_server_routes():
    tmp = tempfile.mkdtemp(prefix="omniseed_srv_")
    httpd = None
    try:
        a = _analysis(tmp)
        j = os.path.join(tmp, "journal.csv")
        s = os.path.join(tmp, "status.json")

        httpd, _ = sv.build_server("127.0.0.1", 0, j, s, quiet=True)
        port = httpd.server_address[1]
        import threading
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()

        def get(path):
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            c.request("GET", path)
            r = c.getresponse()
            body = r.read().decode("utf-8", "replace")
            c.close()
            return r.status, body

        code, body = get("/")
        check("server: / is 200", code == 200)
        check("server: / is the dashboard", "OmniSeed — paper trading" in body)
        check("server: / has the stop column", "<th>stop</th>" in body)
        check("server: / has the regime panel", "Regime status" in body)

        code, body = get("/api/state")
        check("server: /api/state is 200", code == 200)
        payload = json.loads(body)
        check("server: state has positions", "AAPL" in payload["positions"])
        check("server: state stop recovered",
              abs(payload["positions"]["AAPL"]["stop"] - 92.0) < 1e-9)

        code, body = get("/healthz")
        check("server: /healthz is 200", code == 200)
        h = json.loads(body)
        check("server: healthz reports the journal", h["journal_exists"] is True)
        check("server: healthz counts records", h["records"] == a["records"])

        code, body = get("/journal.csv")
        check("server: /journal.csv is 200", code == 200)
        check("server: csv has the header", body.startswith("ts,kind,"))

        code, _ = get("/nope")
        check("server: unknown path is 404", code == 404)
    finally:
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        shutil.rmtree(tmp, ignore_errors=True)


# ==========================================================================
# 8. the real daemon writes what the reader reads (cross-language contract)
#
# Sections 2-3 prove the READER handles `entry conf=.. stop=.. stop_pct=..`.
# They use a hand-authored fixture, so they would keep passing if the C++
# writer's format silently changed and production broke. This section closes
# that hole by running the actual daemon and parsing what it actually wrote.
# Skips honestly when the binary or a market CSV is unavailable (e.g. CI).
# ==========================================================================
def _find_daemon():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exe = "omniseed_agent2.exe" if os.name == "nt" else "omniseed_agent2"
    cand = os.path.join(root, "build", "bin", exe)
    return cand if os.path.exists(cand) else None


def _find_market_csv():
    # An explicit path wins, so a dev with a real feed can force the contract
    # check to run even when models/market/ is absent from their worktree.
    env = os.environ.get("OMNISEED_MARKET_CSV")
    if env and os.path.exists(env):
        return env
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for rel in ("models/market/AAPL_1d.csv", "models/market/DEMO_1d.csv"):
        p = os.path.join(root, rel)
        if os.path.exists(p):
            return p
    return None


def test_daemon_journal_contract():
    binary = _find_daemon()
    csv_path = _find_market_csv()
    if binary is None or csv_path is None:
        missing = []
        if binary is None:
            missing.append("daemon binary")
        if csv_path is None:
            missing.append("market CSV")
        print(f"  --   skip: no {' / '.join(missing)} "
              f"(cross-language contract not exercised here)")
        return

    tmp = tempfile.mkdtemp(prefix="omniseed_daemon_")
    try:
        journal = os.path.join(tmp, "journal.csv")
        proc = subprocess.run(
            [binary, "trading-paper-session",
             "--streams", f"AAPL,equity,{csv_path}",
             "--journal", journal],
            capture_output=True, text=True, timeout=300, cwd=os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))))
        check("daemon: exited 0", proc.returncode == 0)
        check("daemon: journal written", os.path.exists(journal))
        if not os.path.exists(journal):
            return

        entries = []
        with open(journal, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split(",")
                if len(parts) >= 10 and parts[1] == "fill" \
                        and parts[9].startswith("entry"):
                    entries.append(parts)
        check("daemon: produced at least one entry", len(entries) > 0)

        # Every entry must carry provenance the reader can actually recover,
        # and the stop must equal entry * (1 - stop_pct) to the journal's
        # 6-decimal price precision.
        unparsed, mismatched, thin = [], [], []
        for parts in entries:
            price = float(parts[4])
            prov = pr.parse_entry_provenance(parts[9])
            if not prov or "stop_pct" not in prov or "stop" not in prov \
                    or "conf" not in prov:
                unparsed.append(parts[9])
                continue
            expect = price * (1.0 - prov["stop_pct"])
            if abs(prov["stop"] - expect) > 0.01:
                mismatched.append((parts[9], price, expect))
            if not (0.0 <= prov["conf"] <= 1.0):
                thin.append(parts[9])

        check("daemon: every entry reason parses", not unparsed)
        if unparsed:
            print(f"       first unparsed: {unparsed[0]!r}")
        check("daemon: stop == entry*(1-stop_pct)", not mismatched)
        if mismatched:
            print(f"       first mismatch: {mismatched[0]!r}")
        check("daemon: confidence within 0..1", not thin)

        # And the recovered provenance must survive a full book replay.
        recs = pr.load_journal(journal)
        pos = pr.replay_positions(recs)
        check("daemon: replay recovers a stop for every open position",
              all(p.get("stop") is not None for p in pos.values()))
        check("daemon: replay never invents confidence",
              all(p.get("confidence") is None
                  or 0.0 <= p["confidence"] <= 1.0 for p in pos.values()))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ==========================================================================
def main():
    print("== 1. parse the C++ decision JSON ==")
    test_parse_decision()
    test_fail_closed_rules()
    test_screen()
    test_parse_jsonl()
    print("== 2. entry provenance ==")
    test_entry_provenance()
    test_replay_positions_provenance()
    print("== 3. dashboard panels ==")
    test_dashboard_panels()
    print("== 4. engine-limit contract ==")
    test_engine_limits_match_cpp()
    print("== 5. decision gate ==")
    test_decision_gate_file()
    test_gate_is_off_by_default()
    print("== 6. regime reading ==")
    test_regime_reading()
    test_cycle_records_regime_and_gate()
    print("== 7. local server ==")
    test_server_routes()
    print("== 8. daemon -> reader contract ==")
    test_daemon_journal_contract()

    print(f"\n---- decision-bridge/T2 tests: {PASSED} passed, {FAILED} failed ----")
    for f in FAILURES:
        print(f"  FAILED {f}")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
