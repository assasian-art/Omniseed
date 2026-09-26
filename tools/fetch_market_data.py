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
#                      whole pipeline is testable offline.
#
#  INTEGRITY RULES
#    * A provider failure is NEVER silently replaced with synthetic data.
#      An explicit --source that fails exits non-zero (code 2); the `auto`
#      fallback to synthetic requires --allow-synth. Synthetic output is
#      always labelled and written to DEMO_<tf>.csv by default so it cannot
#      clobber a real ticker's file.
#    * Every row carries a `source` provenance column (stooq|yahoo|synth).
#      load_bars_csv() reads fields 0..5 and ignores the extra column, so the
#      C++ contract is unchanged.
#    * --years bounds EVERY source. Stooq returns its full history, so without
#      the trim a `--years 5` request silently delivered decades of bars.
#
#  Usage:
#    python tools/fetch_market_data.py --ticker AAPL --timeframe 1d --years 5
#    python tools/fetch_market_data.py --demo-synth          # offline sample
#
#  Output CSV schema:
#    time,open,high,low,close,volume,source   (time = unix seconds UTC)
# =============================================================================
import argparse
import datetime as dt
import json
import math
import os
import random
import sys
import time
import urllib.request

OUT_DIR = os.path.join("models", "market")
UA = {"User-Agent": "Mozilla/5.0 (omniseed-fetch/1.0)"}

# Columns the C++ loader reads, in order. Anything after these is ignored by
# load_bars_csv() but kept for humans/audits.
BASE_COLUMNS = ["time", "open", "high", "low", "close", "volume"]
PROVENANCE_COLUMN = "source"

YEAR_SECONDS = 365.25 * 86400.0
SYNTH_SEED = 7
SYNTH_DAYS_PER_YEAR = 252          # trading days


class ProviderError(RuntimeError):
    """A real data source could not deliver usable bars."""


def _f(x, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


# --------------------------------------------------------------------------
# Emission: trim to --years, attach provenance, serialize.
# --------------------------------------------------------------------------
def emit_csv(rows, source, years=None) -> str:
    """rows: iterable of (time, open, high, low, close, volume) tuples.

    Sorts ascending, drops rows outside the `years` window measured back from
    the newest bar, and appends the provenance column.
    """
    clean = [r for r in rows if int(r[0]) > 0]
    clean.sort(key=lambda r: r[0])
    if years is not None and clean and years > 0:
        cutoff = clean[-1][0] - years * YEAR_SECONDS
        clean = [r for r in clean if r[0] >= cutoff]
    out = [",".join(BASE_COLUMNS + [PROVENANCE_COLUMN])]
    for t, o, h, l, c, v in clean:
        out.append(f"{int(t)},{_f(o):.6f},{_f(h):.6f},{_f(l):.6f},"
                   f"{_f(c):.6f},{int(_f(v))},{source}")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------
# Stooq daily CSV: Date,Open,High,Low,Close,Volume
# --------------------------------------------------------------------------
def fetch_stooq(sym: str, tf: str) -> str:
    interval = {"1d": "d", "1wk": "w", "1mo": "m"}.get(tf)
    if interval is None:
        raise ProviderError(f"stooq supports 1d/1wk/1mo, got {tf!r}")
    url = f"https://stooq.com/q/d/l/?s={sym}&i={interval}"
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        text = r.read().decode("utf-8", "replace")
    if "No data" in text[:200] or len(text) < 60:
        raise ProviderError(
            f"stooq returned no data for {sym!r} "
            "(check the symbol, e.g. AAPL.US)")
    return text


def parse_stooq(text: str) -> list:
    """Stooq `Date,Open,High,Low,Close,Volume` -> row tuples."""
    lines = text.strip().splitlines()
    if not lines:
        raise ProviderError("stooq: empty response")
    if not lines[0].lower().startswith("date"):
        raise ProviderError(
            f"stooq: unexpected header {lines[0][:60]!r} (rate limited?)")
    rows = []
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) < 5:
            continue
        try:
            t = int(dt.datetime.strptime(parts[0].strip(), "%Y-%m-%d")
                    .replace(tzinfo=dt.timezone.utc).timestamp())
        except ValueError:
            continue
        v = parts[5] if len(parts) > 5 else "0"
        rows.append((t, parts[1], parts[2], parts[3], parts[4], v))
    if not rows:
        raise ProviderError("stooq: no parsable rows")
    return rows


# --------------------------------------------------------------------------
# Yahoo chart API: JSON -> bars (1d or 1h)
# --------------------------------------------------------------------------
def fetch_yahoo(sym: str, tf: str, years: float) -> list:
    interval = {"1d": "1d", "1h": "1h", "1wk": "1wk"}.get(tf)
    if interval is None:
        raise ProviderError(f"yahoo supports 1d/1h/1wk, got {tf!r}")
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
           f"?interval={interval}&range={max(1, int(math.ceil(years)))}y")
    req = urllib.request.Request(url, headers=UA)
    last_err = None
    data = None
    for attempt in range(3):                      # yahoo throttles: retry
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read().decode("utf-8", "replace"))
            break
        except Exception as e:                    # noqa: BLE001
            last_err = e
            time.sleep(2.0 * (attempt + 1))
    if data is None:
        raise ProviderError(f"yahoo failed after retries: {last_err}")

    try:
        res = data["chart"]["result"][0]
        ts = res.get("timestamp") or []
        q = res["indicators"]["quote"][0]
    except (KeyError, IndexError, TypeError) as e:
        raise ProviderError(f"yahoo: unexpected payload ({e})") from e

    rows = []
    for i, t in enumerate(ts):
        o = (q.get("open") or [None] * len(ts))[i]
        h = (q.get("high") or [None] * len(ts))[i]
        l = (q.get("low") or [None] * len(ts))[i]
        c = (q.get("close") or [None] * len(ts))[i]
        v = (q.get("volume") or [0] * len(ts))[i] or 0
        if None in (o, h, l, c):
            continue
        rows.append((int(t), o, h, l, c, v))
    if not rows:
        raise ProviderError(f"yahoo: no usable bars for {sym!r}")
    return rows


# --------------------------------------------------------------------------
# Deterministic synthetic GBM (offline demo / tests)
# --------------------------------------------------------------------------
def synth_rows(days: int, seed: int = SYNTH_SEED, p0: float = 100.0) -> list:
    rng = random.Random(seed)
    t0 = int(dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc).timestamp())
    rows = []
    price = p0
    for d in range(max(1, days)):
        ret = 0.0004 + 0.015 * rng.gauss(0.0, 1.0)
        o = price
        c = max(1.0, o * math.exp(ret))
        hi = max(o, c) * (1.0 + abs(rng.gauss(0.0, 0.006)))
        lo = min(o, c) * (1.0 - abs(rng.gauss(0.0, 0.006)))
        v = int(1e6 * (1.0 + 0.5 * abs(rng.gauss(0.0, 1.0))))
        rows.append((t0 + d * 86400, o, hi, lo, c, v))
        price = c
    return rows


# --------------------------------------------------------------------------
# Source dispatch
# --------------------------------------------------------------------------
def fetch_source(name: str, ticker: str, sym: str, tf: str, years: float):
    """Fetch one named provider. Raises ProviderError on any failure."""
    if name == "stooq":
        return parse_stooq(fetch_stooq(sym, tf))
    if name == "yahoo":
        return fetch_yahoo(ticker, tf, years)
    raise ProviderError(f"unknown source {name!r}")


def default_out(ticker: str, tf: str, synthetic: bool) -> str:
    if synthetic:
        # Never write synthetic bars over a real ticker's file.
        return os.path.join(OUT_DIR, f"DEMO_{tf}.csv")
    safe = ticker.replace("/", "_").replace("^", "_")
    return os.path.join(OUT_DIR, f"{safe}_{tf}.csv")


def main() -> int:
    ap = argparse.ArgumentParser(description="OmniSeed market data fetcher")
    ap.add_argument("--ticker", default="AAPL", help="AAPL | AAPL.US | ^SPX ...")
    ap.add_argument("--timeframe", default="1d", choices=["1d", "1h", "1wk", "1mo"])
    ap.add_argument("--years", type=float, default=5.0,
                    help="history window applied to EVERY source (incl. stooq)")
    ap.add_argument("--source", default="auto", choices=["auto", "stooq", "yahoo", "synth"])
    ap.add_argument("--demo-synth", action="store_true",
                    help="write a deterministic synthetic series and exit")
    ap.add_argument("--allow-synth", action="store_true",
                    help="permit the auto fallback to synthetic data when every "
                         "real provider fails (never implicit)")
    ap.add_argument("--seed", type=int, default=SYNTH_SEED,
                    help="synthetic RNG seed (deterministic output)")
    ap.add_argument("--out", default="", help="override output path")
    a = ap.parse_args()

    if a.years <= 0:
        print("[fetch] --years must be > 0", file=sys.stderr)
        return 2

    synthetic = a.demo_synth or a.source == "synth"
    rows, source = None, None

    if synthetic:
        rows = synth_rows(int(a.years * SYNTH_DAYS_PER_YEAR), seed=a.seed)
        source = "synth"
    else:
        sym = a.ticker if "." in a.ticker or a.ticker.startswith("^") \
            else f"{a.ticker}.US"
        order = [a.source] if a.source != "auto" else ["stooq", "yahoo"]
        failures = []
        for name in order:
            try:
                rows = fetch_source(name, a.ticker, sym, a.timeframe, a.years)
                source = name
                break
            except Exception as e:            # noqa: BLE001
                failures.append((name, e))
                print(f"[fetch] {name} failed: {e}", file=sys.stderr)
        if rows is None:
            detail = "; ".join(f"{n}: {e}" for n, e in failures)
            if not a.allow_synth:
                print(f"[fetch] ERROR: every provider failed ({detail}). "
                      "Refusing to substitute synthetic data. Re-run with "
                      "--allow-synth if a synthetic sample is what you want, "
                      "or --source synth to ask for it explicitly.",
                      file=sys.stderr)
                return 2
            print(f"[fetch] every provider failed ({detail}); "
                  "--allow-synth given -> falling back to SYNTHETIC data",
                  file=sys.stderr)
            rows = synth_rows(int(a.years * SYNTH_DAYS_PER_YEAR), seed=a.seed)
            source = "synth"

    text = emit_csv(rows, source, a.years)
    n = len(text.strip().splitlines()) - 1
    if n == 0:
        print("[fetch] ERROR: 0 bars after trimming to "
              f"{a.years}y — nothing written", file=sys.stderr)
        return 2

    out_path = a.out or default_out(a.ticker, a.timeframe, source == "synth")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write(text)

    print(f"[fetch] {source} -> {out_path} ({n} bars, {a.ticker} "
          f"{a.timeframe}, {a.years}y window)")
    if source == "synth":
        print("[fetch] WARNING: SYNTHETIC data — not market truth, "
              "never use it to justify a trade", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
