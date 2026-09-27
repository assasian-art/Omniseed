#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — tools/paper_report.py
#  M5 — shared analysis over the append-only paper journal + status heartbeat.
#
#  Read-only: parses state/paper_journal.csv (and optionally
#  state/paper_status.json) into the numbers a human actually wants —
#  equity curve, today's P&L, open positions, per-symbol contribution, fills,
#  kill-switch trips, and feed health. Used by both the nightly report and the
#  single-file dashboard, so the numbers can never disagree between them.
#
#  Honest scope: this reports what the paper engine did. It does not predict,
#  and it never claims a profit is guaranteed.
# =============================================================================
import datetime as dt
import json
import os

JOURNAL_HEADER = ("ts,kind,ticker,qty,price,pnl,equity,cash,exposure,reason")

# ---------------------------------------------------------------------------
# The engine's risk contract, restated for display.
#
# These are NOT knobs and they are NOT enforced here — they are copies of
# constants that live in C++, quoted so the dashboard can label its numbers
# without inventing them. The authoritative definitions are:
#
#   per_trade_pct   src/trading/trading_engine.cpp  (the engine CLAMPS the
#                   configured budget into [1%, 2%], so 2% is a ceiling)
#   stop_loss_pct   include/omniseed/trading/trading_engine.h  (RiskLimits)
#   daily_kill_pct  include/omniseed/trading/risk_gate.h
#   weekly_kill_pct include/omniseed/trading/risk_gate.h
#
# tests/test_decision_bridge.py::test_engine_limits_match_cpp re-reads those
# sources and fails if any value here drifts from them.
# ---------------------------------------------------------------------------
ENGINE_LIMITS = {
    "per_trade_pct": 2.0,      # ceiling on risk per trade, % of equity
    "stop_loss_pct": 0.08,     # per-position stop, fraction of entry
    "daily_kill_pct": 3.0,     # daily loss that halts NEW entries, % of equity
    "weekly_kill_pct": 6.0,    # weekly loss that halts NEW entries, %
}

# The C++ engine stamps each entry fill's free-text `reason` with its own
# decision context, because the 10-field schema and the Position struct have
# nowhere else to put it:
#     entry conf=<0..1> stop=<price> stop_pct=<frac>
# These are the keys we know how to read back. Anything else in the reason is
# the rule trace and is left alone.
_ENTRY_KV_FLOAT = ("conf", "stop", "stop_pct")


def day_key(ts):
    """UTC date string for a unix timestamp."""
    return dt.datetime.fromtimestamp(int(ts), dt.timezone.utc).strftime("%Y-%m-%d")


def load_journal(path):
    """-> list of record dicts (header + malformed rows skipped)."""
    out = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("ts,"):
                continue
            parts = line.split(",", 9)
            if len(parts) < 10:
                continue
            try:
                rec = {
                    "ts": int(parts[0]),
                    "kind": parts[1],
                    "ticker": parts[2],
                    "qty": float(parts[3] or 0.0),
                    "price": float(parts[4] or 0.0),
                    "pnl": float(parts[5] or 0.0),
                    "equity": float(parts[6] or 0.0),
                    "cash": float(parts[7] or 0.0),
                    "exposure": float(parts[8] or 0.0),
                    "reason": parts[9],
                }
            except ValueError:
                continue
            out.append(rec)
    return out


def parse_entry_provenance(reason):
    """'entry conf=0.72 stop=178.42 stop_pct=0.08' -> {'conf':0.72, ...}.

    Tolerant by design: an old journal row that is a bare 'entry' yields {},
    and an unparsable token is skipped rather than raising. A missing key means
    'the engine did not record it', which the dashboard must render as an
    unknown — never as a zero.
    """
    out = {}
    if not reason:
        return out
    for tok in reason.replace(";", " ").split():
        if "=" not in tok:
            continue
        k, _, v = tok.partition("=")
        k = k.strip()
        if k not in _ENTRY_KV_FLOAT:
            continue
        try:
            out[k] = float(v)
        except ValueError:
            continue
    return out


def replay_positions(records):
    """Rebuild the open book from the recorded fills (qty>0 buy, qty<0 sell).

    Also recovers each position's entry provenance (stop distance and the
    signal confidence) from the entry fills' `reason` column. The stop LEVEL is
    recomputed as ``avg * (1 - stop_pct)`` *after* the blend, which is the same
    arithmetic RiskManager::check_exit() uses — so the number shown on the
    dashboard is the number the engine would actually act on, even when a name
    was built from several entries at different prices.
    """
    pos = {}
    for r in records:
        if r["kind"] != "fill" or not r["ticker"]:
            continue
        t, q, px = r["ticker"], r["qty"], r["price"]
        if px <= 0:
            continue
        if q > 0:
            p = pos.setdefault(t, {"qty": 0.0, "avg": 0.0})
            total = p["qty"] + q
            p["avg"] = (p["qty"] * p["avg"] + q * px) / total
            p["qty"] = total
            kv = parse_entry_provenance(r["reason"])
            if "conf" in kv:
                p["confidence"] = kv["conf"]
            if "stop_pct" in kv:
                p["stop_pct"] = kv["stop_pct"]
            elif "stop" in kv and "stop_pct" not in p:
                # Legacy row: only the absolute level was recorded.
                p["_stop_level"] = kv["stop"]
            p["opened_ts"] = p.get("opened_ts", r["ts"])
            p["entries"] = p.get("entries", 0) + 1
        elif t in pos:
            pos[t]["qty"] += q
            if pos[t]["qty"] <= 1e-9:
                del pos[t]

    for p in pos.values():
        if "stop_pct" in p:
            p["stop"] = p["avg"] * (1.0 - p["stop_pct"])
        else:
            p["stop"] = p.pop("_stop_level", None)
        p.setdefault("confidence", None)
    return pos


def load_status(path):
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (ValueError, OSError):
        return {}


def analyze(journal_path, status_path=None, now_ts=None):
    records = load_journal(journal_path)
    status = load_status(status_path)
    if now_ts is None:
        now_ts = records[-1]["ts"] if records else 0

    equity_rows = [r for r in records if r["kind"] == "equity"]
    if not equity_rows:
        equity_rows = [r for r in records if r["kind"] == "fill"]
    curve = [(r["ts"], r["equity"]) for r in equity_rows]

    last_equity = curve[-1][1] if curve else 0.0
    first_equity = curve[0][1] if curve else 0.0

    today = day_key(now_ts)
    today_rows = [r for r in equity_rows if day_key(r["ts"]) == today]
    # Day P&L = last equity of today minus the equity carried into today.
    before_today = [r for r in equity_rows if day_key(r["ts"]) < today]
    base = before_today[-1]["equity"] if before_today else (
        today_rows[0]["equity"] if today_rows else first_equity)
    day_pnl = (today_rows[-1]["equity"] - base) if today_rows else 0.0
    day_pct = (day_pnl / base * 100.0) if base else 0.0

    fills = [r for r in records if r["kind"] == "fill"]
    signals = [r for r in records if r["kind"] == "signal"]
    today_fills = [r for r in fills if day_key(r["ts"]) == today]
    halts = [r for r in records
             if r["kind"] == "event" and
             ("halt" in r["reason"] or r["reason"].startswith(("daily:", "weekly:")))]
    abstains = [r for r in records
                if r["kind"] == "event" and r["reason"].startswith("abstain:")]
    refused = [r for r in records
               if r["kind"] == "event" and r["reason"].startswith("refused:")]

    per_symbol = {}
    for r in fills:
        t = r["ticker"]
        d = per_symbol.setdefault(t, {"entries": 0, "exits": 0,
                                      "realized": 0.0, "volume": 0.0})
        if r["qty"] > 0:
            d["entries"] += 1
        else:
            d["exits"] += 1
        d["realized"] += r["pnl"]
        d["volume"] += abs(r["qty"]) * r["price"]

    positions = replay_positions(records)
    for t, p in positions.items():
        last_px = None
        for r in reversed(fills):
            if r["ticker"] == t and r["price"] > 0:
                last_px = r["price"]
                break
        p["last"] = last_px or p["avg"]
        p["market_value"] = p["qty"] * p["last"]
        p["unrealized"] = (p["last"] - p["avg"]) * p["qty"]

    return {
        "journal": journal_path,
        "status_path": status_path or "",
        "records": len(records),
        "fills": len(fills),
        "signals": len(signals),
        "today_fills": today_fills,
        "curve": curve,
        "last_equity": last_equity,
        "first_equity": first_equity,
        "day": today,
        "day_pnl": day_pnl,
        "day_pct": day_pct,
        "halts": len(halts),
        "abstains": len(abstains),
        "refused": len(refused),
        "per_symbol": per_symbol,
        "positions": positions,
        "feeds": status.get("feeds", {}),
        "regime": status.get("regime", {}),
        "status": status,
        "last_ts": records[-1]["ts"] if records else 0,
    }


# ---------------------------------------------------------------------------
# Formatting helpers shared by the Markdown report and the HTML dashboard
# ---------------------------------------------------------------------------
def fmt_money(x):
    return f"{x:,.2f}"


def fmt_signed(x):
    return f"{x:+,.2f}"


def fmt_pct(x):
    return f"{x:+.2f}%"


def pnl_class(x):
    """CSS class for a P&L number. Chinese-market convention: up = red."""
    if x > 0:
        return "up"
    if x < 0:
        return "down"
    return "flat"


def feed_rows(feeds):
    rows = []
    for sym in sorted(feeds):
        f = feeds[sym]
        rows.append({
            "symbol": sym,
            "asset": f.get("asset", ""),
            "provider": f.get("provider", ""),
            "state": f.get("state", "missing"),
            "abstain": bool(f.get("abstain", True)),
            "reason": f.get("reason", ""),
            "last": float(f.get("last") or 0.0),
            "change_pct_24h": float(f.get("change_pct_24h") or 0.0),
            "failures": int(f.get("failures") or 0),
        })
    return rows
