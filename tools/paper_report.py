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


def replay_positions(records):
    """Rebuild the open book from the recorded fills (qty>0 buy, qty<0 sell)."""
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
        elif t in pos:
            pos[t]["qty"] += q
            if pos[t]["qty"] <= 1e-9:
                del pos[t]
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
