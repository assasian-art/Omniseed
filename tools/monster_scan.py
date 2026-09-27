#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — tools/monster_scan.py
#  CLI: provenance CSVs -> per-symbol Monster state-vector CSVs.
#
#  Pipeline (all offline; no network):
#    1. Load each watched symbol's provenance CSV (T6 schema).
#    2. Compute log returns for the whole basket (needed by the cross-asset
#       correlation matrix).
#    3. For every bar, assemble an EvalContext (cross-asset confirm/invalidate
#       counts + event-driven flag) and score it with the Sniper Engine.
#    4. Write  state/monster/<SYM>.csv   (ts,confidence,veto,regime,detail)
#       and    state/monster/summary.json (per-symbol counts + top scores).
#
#  The C++ engine consumes exactly those CSVs (--monster-features). Heavy logic
#  stays here. Honest scope: a filter, not a profit guarantee.
#
#  Usage:
#    python tools/monster_scan.py --watch equity:AAPL,crypto:BTCUSDT,fx:EURUSD
#    python tools/monster_scan.py --csv-dir models/market/paper --news news.json
# =============================================================================
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from monster import features as F                      # noqa: E402
from monster import state_vector as SV                  # noqa: E402
from monster import sniper_engine as SN                 # noqa: E402
from monster import news_hunter as NH                   # noqa: E402
from monster import correlation_matrix as CM            # noqa: E402

DEFAULT_CSV_DIR = "models/market/paper"
DEFAULT_OUT_DIR = "state/monster"
DEFAULT_WATCH = "equity:AAPL,crypto:BTCUSDT,fx:EURUSD"
DEFAULT_TIMEFRAME = "1d"


def csv_name(symbol, timeframe):
    return "%s_%s.csv" % (symbol.replace("/", "").replace(":", "_"), timeframe)


def read_bars(path):
    """Provenance CSV -> [(ts, o, h, l, c, v)] ascending. Malformed rows skipped."""
    bars = []
    try:
        f = open(path, "r", encoding="utf-8")
    except OSError:
        return bars
    with f:
        for line in f:
            line = line.strip()
            if not line or not line[0].isdigit():
                continue
            p = line.split(",")
            if len(p) < 6:
                continue
            try:
                bars.append((int(p[0]), float(p[1]), float(p[2]), float(p[3]),
                             float(p[4]), float(p[5])))
            except ValueError:
                continue
    bars.sort(key=lambda b: b[0])
    return bars


def parse_watch(spec):
    """'equity:AAPL,crypto:BTCUSDT' -> [{'asset','symbol'}, ...]."""
    items = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError("watch item must be ASSET:SYMBOL, got %r" % part)
        asset, sym = part.split(":", 1)
        items.append({"asset": asset.strip().lower(), "symbol": sym.strip()})
    if not items:
        raise ValueError("empty watch list")
    return items


def scan_symbol(symbol, bars, rets_by_symbol, cfg, sz_cfg=None,
                hunter=None, matrix=None, ticker_for_news=None):
    """-> [StateVector] for every bar of `symbol`."""
    prep = SN.prepare(bars, cfg)
    vectors = []
    for i in range(prep.n):
        ts = prep.ts[i]
        ctx = SN.EvalContext()
        if matrix is not None:
            cr = matrix.evaluate(symbol, +1, rets_by_symbol, i)
            ctx.cross_confirm = cr.confirm
            ctx.cross_invalidate = cr.invalidate
            ctx.cross_active = cr.active
        if hunter is not None:
            m = hunter.match(bars, i, ts, direction=+1,
                             ticker=ticker_for_news or symbol)
            ctx.event_driven = m.event_driven
        v = SN.evaluate(prep, i, ctx, cfg)
        ap = prep.atr_pct[i]
        sv = SV.build(symbol, v,
                      atr_pct=ap if ap == ap else None, sz_cfg=sz_cfg)
        sv.event_driven = ctx.event_driven
        vectors.append(sv)
    return vectors


def run_scan(watch, csv_dir=DEFAULT_CSV_DIR, out_dir=DEFAULT_OUT_DIR,
             timeframe=DEFAULT_TIMEFRAME, cfg=None, sz_cfg=None,
             news_items=None, news_cfg=None, matrix=None):
    """-> summary dict (also writes the feature CSVs)."""
    cfg = cfg or SN.SniperConfig()
    os.makedirs(out_dir, exist_ok=True)

    bars_by_sym, rets_by_symbol = {}, {}
    for item in watch:
        sym = item["symbol"]
        bars = read_bars(os.path.join(csv_dir, csv_name(sym, timeframe)))
        if not bars:
            continue
        bars_by_sym[sym] = bars
        rets_by_symbol[sym] = F.log_returns(F.closes(bars))

    hunter = None
    if news_items:
        hunter = NH.NewsHunter(news_cfg)
        hunter.ingest(news_items)

    matrix = matrix or CM.CorrelationMatrix()

    summary = {"symbols": {}, "min_confidence": cfg.min_confidence,
               "regime_mode": cfg.regime_mode, "ensemble": bool(cfg.ensemble),
               "news_events": len(hunter.events) if hunter else 0}
    for item in watch:
        sym = item["symbol"]
        if sym not in bars_by_sym:
            summary["symbols"][sym] = {"bars": 0, "error": "no-data"}
            continue
        vectors = scan_symbol(sym, bars_by_sym[sym], rets_by_symbol, cfg,
                              sz_cfg, hunter, matrix)
        path = os.path.join(out_dir, "%s.csv" % sym)
        SV.write_feature_csv(path, vectors)
        props = [v for v in vectors if v.propose(cfg.min_confidence)]
        veteos = sum(1 for v in vectors if v.veto)
        best = max(vectors, key=lambda v: v.confidence) if vectors else None
        regimes = {}
        for v in vectors:
            regimes[v.regime] = regimes.get(v.regime, 0) + 1
        ens_vetoes = sum(1 for v in vectors if v.detail.find("VETO=ensemble") >= 0)
        summary["symbols"][sym] = {
            "bars": len(vectors),
            "proposed": len(props),
            "vetoed": veteos,
            "ensemble_vetoed": ens_vetoes,
            "event_driven": sum(1 for v in vectors if v.event_driven),
            "regimes": regimes,
            "mean_conviction": round(
                sum(v.conviction for v in vectors) / len(vectors), 4)
            if vectors else 0.0,
            "best_confidence": round(best.confidence, 4) if best else 0.0,
            "best_regime": best.regime if best else "",
            "csv": path,
        }
    return summary


def load_news(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("items", [])


def main():
    ap = argparse.ArgumentParser(description="OmniSeed Monster scanner")
    ap.add_argument("--watch", default=DEFAULT_WATCH)
    ap.add_argument("--csv-dir", default=DEFAULT_CSV_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--timeframe", default=DEFAULT_TIMEFRAME)
    ap.add_argument("--news", default="", help="JSON list of news items")
    ap.add_argument("--min-confidence", type=float, default=0.85)
    ap.add_argument("--regime-mode", default="advanced",
                    choices=("advanced", "legacy"),
                    help="advanced = multi-axis regime ensemble (default)")
    ap.add_argument("--ensemble", action="store_true",
                    help="route the strategy zoo into a fail-closed veto")
    ap.add_argument("--summary", default="", help="summary json path override")
    a = ap.parse_args()

    try:
        watch = parse_watch(a.watch)
    except ValueError as e:
        print("[monster] %s" % e, file=sys.stderr)
        return 2

    cfg = SN.SniperConfig(min_confidence=a.min_confidence,
                          regime_mode=a.regime_mode,
                          ensemble=a.ensemble)
    summary = run_scan(watch, csv_dir=a.csv_dir, out_dir=a.out_dir,
                       timeframe=a.timeframe, cfg=cfg,
                       news_items=load_news(a.news))

    out = a.summary or os.path.join(a.out_dir, "summary.json")
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    os.replace(tmp, out)

    print("[monster] regime=%s ensemble=%s"
          % (summary["regime_mode"], summary["ensemble"]))
    for sym, s in summary["symbols"].items():
        if s.get("error"):
            print("[monster] %-10s %s" % (sym, s["error"]))
        else:
            print("[monster] %-10s %4d bars | %3d proposed | %3d vetoed "
                  "(%d ensemble) | best S=%.3f (%s)"
                  % (sym, s["bars"], s["proposed"], s["vetoed"],
                     s.get("ensemble_vetoed", 0), s["best_confidence"],
                     s["best_regime"]))
            print("[monster] %-10s regimes %s  mean_conviction=%+.3f"
                  % ("", s.get("regimes", {}), s.get("mean_conviction", 0.0)))
    print("[monster] wrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
