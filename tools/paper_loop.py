#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — tools/paper_loop.py
#  M5 — the 24/7 multi-asset PAPER loop.
#
#  One cycle:
#    1. PROBE every watch item (stocks / crypto / forex / meme) for a live
#       ticker and derive its feed state (ok / degraded / dead / missing).
#    2. Refresh that symbol's provenance CSV from the bars adapter (dedup by
#       timestamp; the schema is the T6 contract) — but only while the feed is
#       OK. A feed that is not OK is ABSTAINed: no new data, no new risk.
#    3. Invoke the C++ engine (`omniseed_agent2 trading-paper-session`) which
#       replays the merged timeline, sizes per asset class, enforces the
#       daily+weekly kill-switch, and appends to the append-only journal.
#    4. Write state/paper_status.json (heartbeat + feed health + last session).
#
#  Honest scope: paper money only. No real orders, no guaranteed returns.
#  Network access lives here (Python); the engine itself never opens a socket.
#
#  Usage:
#    python tools/paper_loop.py --watch equity:AAPL,crypto:BTCUSDT,fx:EURUSD
#    python tools/paper_loop.py --once          # a single cycle (cron-friendly)
#    python tools/paper_loop.py --interval 300  # poll every 5 minutes, forever
# =============================================================================
import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import market_feeds as mf  # noqa: E402

DEFAULT_WATCH = "equity:AAPL,crypto:BTCUSDT,fx:EURUSD"
DEFAULT_JOURNAL = "state/paper_journal.csv"
DEFAULT_STATUS = "state/paper_status.json"
DEFAULT_CSV_DIR = "models/market/paper"

# Reuse the perception policy constants (mirrors market_perception.h).
DEGRADED_AFTER_SECS = mf.DEGRADED_AFTER_SECS
DEAD_AFTER_SECS = mf.DEAD_AFTER_SECS


def parse_watch(spec):
    """'equity:AAPL,crypto:BTCUSDT' -> [{'asset','symbol'}, ...]."""
    items = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"watch item must be ASSET:SYMBOL, got {part!r}")
        asset, symbol = part.split(":", 1)
        asset = asset.strip().lower()
        if asset not in mf.ASSETS:
            raise ValueError(f"unknown asset class {asset!r}")
        items.append({"asset": asset, "symbol": symbol.strip()})
    if not items:
        raise ValueError("empty watch list")
    return items


def default_binary():
    exe = "omniseed_agent2.exe" if os.name == "nt" else "omniseed_agent2"
    return os.path.join("build", "bin", exe)


# ---------------------------------------------------------------------------
# CSV merge (provenance-preserving)
# ---------------------------------------------------------------------------
def read_csv_rows(path):
    """-> {ts: (ts,o,h,l,c,v,source)} for an existing provenance CSV."""
    rows = {}
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line[0].isdigit() is False:
                continue                      # header / junk
            parts = line.split(",")
            if len(parts) < 6:
                continue
            try:
                ts = int(parts[0])
                o, h, l, c, v = (float(parts[i]) for i in range(1, 6))
            except ValueError:
                continue
            src = parts[6] if len(parts) > 6 else ""
            rows[ts] = (ts, o, h, l, c, v, src)
    return rows


def merge_and_write(path, new_rows, source):
    """Merge `new_rows` (t,o,h,l,c,v tuples) into `path`, dedup by ts."""
    merged = read_csv_rows(path)
    added = 0
    for r in new_rows:
        ts = int(r[0])
        if ts not in merged:
            added += 1
        merged[ts] = (ts, float(r[1]), float(r[2]), float(r[3]), float(r[4]),
                      float(r[5]), source)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("time,open,high,low,close,volume,source\n")
        for ts in sorted(merged):
            r = merged[ts]
            f.write(f"{r[0]},{r[1]:.6f},{r[2]:.6f},{r[3]:.6f},{r[4]:.6f},"
                    f"{r[5]:.2f},{r[6]}\n")
    return added, len(merged)


# ---------------------------------------------------------------------------
# Feed health (mirrors the C++ FeedHealthMonitor semantics)
# ---------------------------------------------------------------------------
def derive_state(staleness, failures):
    return mf.feed_state(staleness, failures)


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------
class PaperLoop:
    def __init__(self, watch, journal=DEFAULT_JOURNAL, status=DEFAULT_STATUS,
                 csv_dir=DEFAULT_CSV_DIR, capital=100000.0, years=1.0,
                 timeframe="1d", strategy="balanced", binary=None,
                 probe_fn=None, fetch_fn=None, runner=None, now_fn=None,
                 dry_run=False, min_volume=0.0):
        self.watch = watch
        self.journal = journal
        self.status_path = status
        self.csv_dir = csv_dir
        self.capital = capital
        self.years = years
        self.timeframe = timeframe
        self.strategy = strategy
        self.binary = binary or default_binary()
        self.probe_fn = probe_fn
        self.fetch_fn = fetch_fn
        self.runner = runner or self._run_engine
        self.now_fn = now_fn or time.time
        self.dry_run = dry_run
        self.min_volume = min_volume
        self.cycle_no = 0
        # per-symbol health, persisted across cycles
        self.health = {}

    # -- helpers -----------------------------------------------------------
    def csv_path(self, item):
        sym = item["symbol"].replace("/", "").replace(":", "_")
        return os.path.join(self.csv_dir, f"{sym}_{self.timeframe}.csv")

    def _probe(self, item, http):
        if self.probe_fn is not None:
            return self.probe_fn(item["asset"], item["symbol"], http)
        return mf.probe_ticker(item["asset"], item["symbol"], http)

    def _fetch(self, item, http):
        if self.fetch_fn is not None:
            return self.fetch_fn(item["asset"], item["symbol"],
                                 self.timeframe, self.years, http)
        return mf.fetch_bars(item["asset"], item["symbol"], self.timeframe,
                             self.years, http)

    # -- one cycle ---------------------------------------------------------
    def cycle(self, http):
        self.cycle_no += 1
        now = int(self.now_fn())
        feeds = {}

        for item in self.watch:
            key = item["symbol"]
            h = self.health.setdefault(key, {"failures": 0, "last_ok_ts": 0})
            rec = {"asset": item["asset"], "symbol": key, "provider": "",
                   "state": "missing", "abstain": True, "reason": "missing-data",
                   "failures": h["failures"], "last_ok_ts": h["last_ok_ts"],
                   "last": 0.0, "change_pct_24h": 0.0, "volume_24h": 0.0,
                   "error": "", "csv": self.csv_path(item)}
            try:
                tk, state, abstain, reason = self._probe(item, http)
                h["failures"] = 0
                h["last_ok_ts"] = now
                rec.update(provider=tk.get("provider", ""),
                           last=float(tk.get("last") or 0.0),
                           change_pct_24h=float(tk.get("change_pct_24h") or 0.0),
                           volume_24h=float(tk.get("volume_24h") or 0.0))
                if self.min_volume > 0 and rec["volume_24h"] < self.min_volume:
                    abstain, reason = True, "low-liquidity"
                rec.update(state=state, abstain=abstain, reason=reason,
                           failures=0, last_ok_ts=now)
            except Exception as e:                      # noqa: BLE001
                h["failures"] += 1
                stale = (now - h["last_ok_ts"]) if h["last_ok_ts"] else None
                state = derive_state(stale, h["failures"])
                abstain, reason = mf.abstain_decision(state, 0.0, 0.0)
                rec.update(state=state, abstain=abstain, reason=reason,
                           failures=h["failures"], last_ok_ts=h["last_ok_ts"],
                           error=str(e))
            feeds[key] = rec

            # Refresh bars only while the feed is OK (ABSTAIN otherwise).
            if not rec["abstain"]:
                try:
                    rows, source, _sym = self._fetch(item, http)
                    added, total = merge_and_write(self.csv_path(item), rows,
                                                   source)
                    rec["bars_added"] = added
                    rec["bars_total"] = total
                except Exception as e:                  # noqa: BLE001
                    rec["error"] = str(e)
                    rec["abstain"] = True
                    rec["reason"] = "fetch-failed"

        session = self._run_session(feeds)
        status = {
            "updated_ts": now,
            "updated_iso": dt.datetime.fromtimestamp(
                now, dt.timezone.utc).isoformat(),
            "cycle": self.cycle_no,
            "mode": "paper",
            "capital": self.capital,
            "timeframe": self.timeframe,
            "journal": self.journal,
            "feeds": feeds,
            "last_session": session,
        }
        self._write_status(status)
        return status

    def _streams_arg(self):
        """Every symbol that already has a CSV.

        ABSTAINed symbols are still included: the engine must keep marking
        their open positions (and be able to exit them). `--skip` gates new
        entries only — that is what makes ABSTAIN "no new risk", not "forget
        the position".
        """
        parts = []
        for item in self.watch:
            path = self.csv_path(item)
            if os.path.exists(path):
                parts.append(f"{item['symbol']},{item['asset']},{path}")
        return ";".join(parts)

    def _run_session(self, feeds):
        streams = self._streams_arg()
        skip = ",".join(k for k, v in feeds.items() if v.get("abstain"))
        if not streams:
            return {"ok": False, "skipped": True,
                    "reason": "no symbol has any data yet (all feeds ABSTAIN)"}
        if self.dry_run:
            return {"ok": True, "dry_run": True, "streams": streams,
                    "skip": skip}
        try:
            return self.runner(streams, skip)
        except Exception as e:                          # noqa: BLE001
            return {"ok": False, "error": str(e)}

    def _run_engine(self, streams, skip):
        cmd = [self.binary, "trading-paper-session", "--streams", streams,
               "--journal", self.journal, "--capital", str(self.capital),
               "--strategy", self.strategy]
        if skip:
            cmd += ["--skip", skip]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        out = proc.stdout or ""
        res = {"ok": proc.returncode == 0, "returncode": proc.returncode,
               "tail": out.strip().splitlines()[-1] if out.strip() else ""}
        m = re.search(r"(\d+) timestamps, (\d+) stream-bars", out)
        if m:
            res["timestamps"] = int(m.group(1))
            res["stream_bars"] = int(m.group(2))
        m = re.search(r"(\d+) entries, (\d+) exits, (\d+) halts, "
                      r"(\d+) abstains, (\d+) refused", out)
        if m:
            res.update(entries=int(m.group(1)), exits=int(m.group(2)),
                       halts=int(m.group(3)), abstains=int(m.group(4)),
                       refused=int(m.group(5)))
        m = re.search(r"equity\s*:\s*([-\d.]+)", out)
        if m:
            res["equity"] = float(m.group(1))
        if proc.returncode != 0 and proc.stderr:
            res["error"] = proc.stderr.strip().splitlines()[-1]
        return res

    def _write_status(self, status):
        os.makedirs(os.path.dirname(self.status_path) or ".", exist_ok=True)
        tmp = self.status_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(status, f, indent=2, sort_keys=True)
        os.replace(tmp, self.status_path)

    # -- driver ------------------------------------------------------------
    def run(self, cycles, interval, sleep_fn=time.sleep):
        cache = mf.Cache(os.path.join("state", "feed_cache"), 900.0)
        http = mf.HttpClient(cache=cache, limiter=mf.RateLimiter(20.0))
        n = 0
        while cycles == 0 or n < cycles:
            st = self.cycle(http)
            ok = sum(1 for f in st["feeds"].values() if not f["abstain"])
            print(f"[loop] cycle {st['cycle']}: {ok}/{len(st['feeds'])} feeds ok "
                  f"| session {st['last_session']}")
            n += 1
            if cycles == 0 or n < cycles:
                sleep_fn(max(1, interval))
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="OmniSeed 24/7 paper loop")
    ap.add_argument("--watch", default=DEFAULT_WATCH,
                    help="ASSET:SYMBOL list, e.g. equity:AAPL,crypto:BTCUSDT")
    ap.add_argument("--journal", default=DEFAULT_JOURNAL)
    ap.add_argument("--status", default=DEFAULT_STATUS)
    ap.add_argument("--csv-dir", default=DEFAULT_CSV_DIR)
    ap.add_argument("--capital", type=float, default=100000.0)
    ap.add_argument("--years", type=float, default=1.0)
    ap.add_argument("--timeframe", default="1d")
    ap.add_argument("--strategy", default="balanced")
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--cycles", type=int, default=0, help="0 = run forever")
    ap.add_argument("--once", action="store_true", help="exactly one cycle")
    ap.add_argument("--binary", default=None)
    ap.add_argument("--min-volume", type=float, default=0.0)
    ap.add_argument("--dry-run", action="store_true",
                    help="poll + refresh CSVs but do not run the engine")
    a = ap.parse_args()

    try:
        watch = parse_watch(a.watch)
    except ValueError as e:
        print(f"[loop] {e}", file=sys.stderr)
        return 2

    loop = PaperLoop(watch, journal=a.journal, status=a.status,
                     csv_dir=a.csv_dir, capital=a.capital, years=a.years,
                     timeframe=a.timeframe, strategy=a.strategy,
                     binary=a.binary, dry_run=a.dry_run,
                     min_volume=a.min_volume)
    cycles = 1 if a.once else a.cycles
    try:
        return loop.run(cycles, a.interval)
    except KeyboardInterrupt:
        print("[loop] stopped by user")
        return 0


if __name__ == "__main__":
    sys.exit(main())
