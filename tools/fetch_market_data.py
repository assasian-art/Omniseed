#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — tools/fetch_market_data.py
#
#  Downloads historical OHLCV bars into models/market/<TICKER>_<TF>.csv for
#  the C++ trading framework (omniseed trading-analyze / trading-simulate).
#
#  Sources (in order, no keys required):
#    1. Stooq        — https://stooq.com/q/d/l/?s=<sym>&i=<tf>   (daily/weekly;
#                      ticker mapping: AAPL.US, ^SIX3 etc. Free CSV export.)
#    2. Yahoo Finance chart API — https://query1.finance.yahoo.com/v8/finance/chart/<sym>
#                      (1d/1h; works without a key; may throttle — retry logic
#                      included). Respects an optional User-Agent override.
#    3. Synthetic    — deterministic geometric-brownian-motion sample so the
#                      whole pipeline is testable offline (marked SYNTH).
#
#  Usage:
#    python tools/fetch_market_data.py --ticker AAPL --timeframe 1d --years 5
#    python tools/fetch_market_data.py --demo-synth          # offline sample
#
#  Output CSV schema (exactly what load_bars_csv() reads):
#    time,open,high,low,close,volume        (time = unix seconds UTC)
# =============================================================================
import argparse
import datetime as dt
import math
import os
import random
import sys
import time
import urllib.request

OUT_DIR = os.path.join("models", "market")
UA = {"User-Agent": "Mozilla/5.0 (omniseed-fetch/1.0)"}

# --------------------------------------------------------------------------
# Stooq daily CSV: Date,Open,High,Low,Close,Volume
# --------------------------------------------------------------------------
def fetch_stooq(sym: str, tf: str) -> str:
    interval = {"1d": "d", "1wk": "w", "1mo": "m"}.get(tf)
    if interval is None:
        raise RuntimeError(f"stooq supports 1d/1wk/1mo, got {tf!r}")
    url = f"https://stooq.com/q/d/l/?s={sym}&i={interval}"
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        text = r.read().decode("utf-8", "replace")
    if "No data" in text[:200] or len(text) < 60:
        raise RuntimeError("stooq returned no data (check the symbol, e.g. AAPL.US)")
    return text

# --------------------------------------------------------------------------
# Yahoo chart API: JSON -> bars (1d or 1h)
# --------------------------------------------------------------------------
def fetch_yahoo(sym: str, tf: str, years: float) -> str:
    import json
    interval = {"1d": "1d", "1h": "1h", "1wk": "1wk"}.get(tf)
    if interval is None:
        raise RuntimeError(f"yahoo supports 1d/1h/1wk, got {tf!r}")
    span = int(years * 365.25 * 86400)
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
           f"?interval={interval}&range={max(1, int(years))}y"
           f"&_={span}")
    req = urllib.request.Request(url, headers=UA)
    last_err = None
    for attempt in range(3):                      # yahoo throttles: retry
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read().decode("utf-8", "replace"))
            break
        except Exception as e:                    # noqa: BLE001
            last_err = e
            time.sleep(2.0 * (attempt + 1))
    else:
        raise RuntimeError(f"yahoo failed after retries: {last_err}")

    res = data["chart"]["result"][0]
    ts = res.get("timestamp") or []
    q = res["indicators"]["quote"][0]
    rows = ["time,open,high,low,close,volume"]
    for i, t in enumerate(ts):
        o = q.get("open", [None] * len(ts))[i]
        h = q.get("high", [None] * len(ts))[i]
        l = q.get("low", [None] * len(ts))[i]
        c = q.get("close", [None] * len(ts))[i]
        v = q.get("volume", [None] * len(ts))[i] or 0
        if None in (o, h, l, c):
            continue
        rows.append(f"{int(t)},{o:.6f},{h:.6f},{l:.6f},{c:.6f},{int(v)}")
    return "\n".join(rows) + "\n"

# --------------------------------------------------------------------------
# Deterministic synthetic GBM (offline demo / tests)
# --------------------------------------------------------------------------
def synth_bars(days: int, seed: int = 7, p0: float = 100.0) -> str:
    rng = random.Random(seed)
    rows = ["time,open,high,low,close,volume"]
    t0 = int(dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc).timestamp())
    price = p0
    for d in range(days):
        drift = 0.0004
        vol = 0.015
        ret = drift + vol * rng.gauss(0.0, 1.0)
        o = price
        c = max(1.0, o * math.exp(ret))
        hi = max(o, c) * (1.0 + abs(rng.gauss(0.0, 0.006)))
        lo = min(o, c) * (1.0 - abs(rng.gauss(0.0, 0.006)))
        v = int(1e6 * (1.0 + 0.5 * abs(rng.gauss(0.0, 1.0))))
        ts = t0 + d * 86400
        rows.append(f"{ts},{o:.6f},{hi:.6f},{lo:.6f},{c:.6f},{v}")
        price = c
    return "\n".join(rows) + "\n"

# --------------------------------------------------------------------------
# Stooq date-column -> unix seconds
# --------------------------------------------------------------------------
def normalize(text: str, tf: str) -> str:
    out = []
    lines = text.strip().splitlines()
    if not lines:
        raise RuntimeError("empty response")
    header = lines[0].lower()
    if not header.startswith("date"):
        return text                            # already normalized (yahoo)
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) < 5:
            continue
        d = parts[0]
        try:
            t = int(dt.datetime.strptime(d, "%Y-%m-%d")
                    .replace(tzinfo=dt.timezone.utc).timestamp())
        except ValueError:
            continue
        o, h, l, c = parts[1:5]
        v = parts[5] if len(parts) > 5 else "0"
        out.append(f"{t},{o},{h},{l},{c},{v}")
    if not out:
        raise RuntimeError("no parsable rows")
    return "time,open,high,low,close,volume\n" + "\n".join(out) + "\n"

def main() -> int:
    ap = argparse.ArgumentParser(description="OmniSeed market data fetcher")
    ap.add_argument("--ticker", default="AAPL", help="AAPL | AAPL.US | ^SPX ...")
    ap.add_argument("--timeframe", default="1d", choices=["1d", "1h", "1wk", "1mo"])
    ap.add_argument("--years", type=float, default=5.0)
    ap.add_argument("--source", default="auto", choices=["auto", "stooq", "yahoo", "synth"])
    ap.add_argument("--demo-synth", action="store_true",
                    help="write a deterministic synthetic series and exit")
    ap.add_argument("--out", default="", help="override output path")
    a = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = a.out or os.path.join(OUT_DIR, f"{a.ticker.replace('/', '_').replace('^', '_')}_{a.timeframe}.csv")

    if a.demo_synth:
        text, src = synth_bars(int(a.years * 252)), "SYNTH"
    else:
        sym = a.ticker if "." in a.ticker or a.ticker.startswith("^") else f"{a.ticker}.US"
        text, src, err = None, None, None
        order = [a.source] if a.source != "auto" else ["stooq", "yahoo"]
        for s in order:
            try:
                text = fetch_stooq(sym, a.timeframe) if s == "stooq" else fetch_yahoo(a.ticker, a.timeframe, a.years)
                src = s.upper()
                break
            except Exception as e:            # noqa: BLE001
                err = e
                print(f"[fetch] {s} failed: {e}", file=sys.stderr)
        if text is None:
            print("[fetch] all sources failed; falling back to SYNTH "
                  "(marked clearly so nobody trades on it)", file=sys.stderr)
            text, src = synth_bars(int(a.years * 252)), "SYNTH"
        if src == "STOOQ":
            text = normalize(text, a.timeframe)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    n = len(text.strip().splitlines()) - 1
    print(f"[fetch] {src} -> {out_path} ({n} bars, {a.ticker} {a.timeframe})")
    if src == "SYNTH":
        print("[fetch] WARNING: SYNTHETIC data — do not treat as market truth")
    return 0

if __name__ == "__main__":
    sys.exit(main())
