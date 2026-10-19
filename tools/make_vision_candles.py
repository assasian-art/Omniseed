#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — tools/make_vision_candles.py   (T113b, P12 CHART READING)
#
#  Route (a) from state/task_T113_brief.md: render windows of the LOCAL
#  models/market/AAPL_1d.csv bars as deterministic 32x32 candlestick PPMs.
#  The label is the NEXT bar after the rendered window (close up/down; a
#  flat tie bar -> window skipped, like anomaly's bad-image skip).
#
#  HONEST CAVEAT (fixed in the brief, written verbatim into the manifest):
#  the label IS the generator's next-bar direction — a head trained on these
#  images measures THIS generator (render style, palette, window/stride,
#  symbol), i.e. generator recognition, NOT "chart literacy" and NOT
#  forecasting skill on unseen symbols or renderers.  Per
#  docs/VISION_AUDIO_DATA.md §3.5: the generator params ARE the docs.
#
#  DETERMINISM: NO RNG at all — integer-only rendering, fixed palette,
#  windows sliced by fixed stride from a fixed CSV.  Same CSV + same params
#  => byte-identical PPMs (asserted by tests/test_make_vision_candles.py).
#
#  FAIL-CLOSED (exit 2, "REFUSED:"):
#    - CSV missing / bad header / non-numeric OHLC / too short for the
#      requested window+N.
#    - output dir has the .candlestick.complete sentinel: refuse without
#      --force.  The sentinel records the param hash; if the CURRENT params
#      hash differently the sentinel is STALE -> always refuse (never
#      silently mix render regimes — delete the dir to re-render).
#
#  usage:
#    python tools/make_vision_candles.py                       # N=500 -> models/modality/vision.candlestick
#    python tools/make_vision_candles.py --n 4 --out-dir /tmp/x
# =============================================================================
"""Synthesise vision.candlestick chart images from local AAPL bars (stdlib only)."""

import argparse
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV = os.path.join(ROOT, "models", "market", "AAPL_1d.csv")
DEFAULT_OUT = os.path.join(ROOT, "models", "modality", "vision.candlestick")

# --- render constants (THE documentation; changing any -> new param hash) ---
RENDER_VERSION = "candles-v1"       # bump on any visual change
W = H = 32                          # VisionEncoder-compatible geometry (P6 RGB)
CH = 3
HEADER_LEN = len(b"P6\n32 32\n255\n")
PALETTE = {                         # fixed, integer, documented
    "bg": (20, 24, 32),
    "up": (0, 200, 80),             # close >= open
    "down": (220, 60, 60),          # close <  open
}
DEFAULT_WINDOW = 16                 # candles per chart (<= 32 so each >= 1px)
DEFAULT_STRIDE = 1                  # window start step through the CSV
DEFAULT_N = 500                     # brief: scene-sized first dump
SENTINEL = ".candlestick.complete"

CAVEAT = (
    "The label IS the source bars. A head trained on synthetic chart images\n"
    "# learns THIS generator (render style, palette, window/stride, symbol) — it\n"
    "# measures generator recognition, not \"chart literacy\" and not forecasting\n"
    "# skill on unseen symbols or renderers. Any accuracy is a property of the\n"
    "# generator/label pair, per docs/VISION_AUDIO_DATA.md §3.5."
)


def fail(msg):
    print("REFUSED: %s" % msg)
    return 2


def read_bars(csv_path):
    """Parse the bar CSV -> list of (o,h,l,c) floats; fail-closed on any bad row."""
    if not os.path.exists(csv_path):
        return None, fail("CSV missing: %s" % csv_path)
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    if not lines:
        return None, fail("CSV empty: %s" % csv_path)
    head = lines[0].split(",")
    if head[:5] != ["time", "open", "high", "low", "close"]:
        return None, fail("bad CSV header %r (want time,open,high,low,close,...)"
                          % lines[0])
    bars = []
    for ln in lines[1:]:
        cols = ln.split(",")
        if len(cols) < 5:
            return None, fail("short row %r" % ln)
        try:
            o, h, l, c = (float(cols[1]), float(cols[2]),
                          float(cols[3]), float(cols[4]))
        except ValueError:
            return None, fail("non-numeric OHLC in row %r" % ln)
        bars.append((o, h, l, c))
    if len(bars) < 2:
        return None, fail("CSV too short: %d bars" % len(bars))
    return bars, 0


def param_hash(csv_path, window, stride, n):
    """Hash of EVERYTHING that changes the render or the row set."""
    with open(csv_path, "rb") as f:
        csv_sha = hashlib.sha256(f.read()).hexdigest()
    blob = json.dumps({"csv_sha256": csv_sha, "window": window,
                       "stride": stride, "n": n,
                       "render": RENDER_VERSION}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def price_to_y(p, lo, hi):
    """Map price -> row (0 = top). Integer, deterministic."""
    if hi <= lo:
        return H // 2
    return (H - 1) - int(round((p - lo) / (hi - lo) * (H - 1)))


def render(window_bars):
    """Render one window of (o,h,l,c) -> 32x32 RGB bytes. No RNG."""
    pix = bytearray()
    lo = min(b[2] for b in window_bars)
    hi = max(b[1] for b in window_bars)
    bg = PALETTE["bg"]
    for y in range(H):
        for x in range(W):
            pix.extend(bg)
    nb = len(window_bars)
    for i, (o, h, l, c) in enumerate(window_bars):
        x0 = i * W // nb
        x1 = ((i + 1) * W // nb) - 1
        if x1 < x0:
            x1 = x0
        col = PALETTE["up"] if c >= o else PALETTE["down"]
        yh, yl = price_to_y(h, lo, hi), price_to_y(l, lo, hi)
        yo, yc = price_to_y(o, lo, hi), price_to_y(c, lo, hi)
        body_top, body_bot = min(yo, yc), max(yo, yc)   # >= 1px (doji -> 1px)
        xm = (x0 + x1) // 2
        for y in range(yh, yl + 1):                     # wick, center column
            off = (y * W + xm) * CH
            pix[off:off + 3] = bytes(col)
        for y in range(body_top, body_bot + 1):         # body, full slot
            for x in range(x0, x1 + 1):
                off = (y * W + x) * CH
                pix[off:off + 3] = bytes(col)
    return bytes(pix)


def write_ppm(path, pix):
    with open(path, "wb") as f:
        f.write(b"P6\n32 32\n255\n")
        f.write(pix)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Synthesise deterministic candlestick chart images "
                    "from local bar CSVs (T113b)")
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--n", type=int, default=DEFAULT_N,
                    help="number of charts (default %d)" % DEFAULT_N)
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                    help="candles per chart (default %d)" % DEFAULT_WINDOW)
    ap.add_argument("--stride", type=int, default=DEFAULT_STRIDE,
                    help="window start step (default %d)" % DEFAULT_STRIDE)
    ap.add_argument("--force", action="store_true",
                    help="re-render even if the completion sentinel is present "
                         "(same param hash only; stale hash always refuses)")
    args = ap.parse_args(argv)

    if args.window < 2 or args.window > W:
        return fail("--window must be in [2, %d]" % W)
    if args.stride < 1:
        return fail("--stride must be >= 1")
    if args.n < 1:
        return fail("--n must be >= 1")

    bars, rc = read_bars(args.csv)
    if rc:
        return rc
    max_windows = (len(bars) - args.window - 1) // args.stride + 1
    if max_windows < args.n:
        return fail("CSV too short for --n %d --window %d --stride %d "
                    "(max %d windows from %d bars)"
                    % (args.n, args.window, args.stride, max_windows, len(bars)))

    phash = param_hash(args.csv, args.window, args.stride, args.n)
    sentinel = os.path.join(args.out_dir, SENTINEL)
    if os.path.exists(sentinel):
        prev = ""
        with open(sentinel, "r", encoding="utf-8") as f:
            prev = f.read()
        if ("param_hash=%s" % phash) not in prev:
            return fail("STALE completion sentinel in %s (recorded %s, current "
                        "params hash %s) — never mix render regimes; delete the "
                        "dir to re-render" %
                        (args.out_dir,
                         prev.split("param_hash=")[-1].splitlines()[0] if "param_hash=" in prev else "?",
                         phash))
        if not args.force:
            return fail("%s already complete (sentinel present); pass --force "
                        "to re-render the same params" % args.out_dir)

    # Collect windows; flat (tie) next bar -> skip, like anomaly bad-image skip.
    rows = []
    hist = {"up": 0, "down": 0}
    for s in range(0, max_windows * args.stride, args.stride):
        if len(rows) >= args.n:
            break
        nb = bars[s:s + args.window]
        nxt = bars[s + args.window]
        if nxt[3] > nb[-1][3]:
            label = "up"
        elif nxt[3] < nb[-1][3]:
            label = "down"
        else:
            continue                              # tie bar: window skipped
        rows.append((s, render(nb), label))
        hist[label] += 1
    if len(rows) < args.n:
        return fail("only %d non-tie windows available (asked --n %d)"
                    % (len(rows), args.n))

    os.makedirs(args.out_dir, exist_ok=True)
    # Incomplete leftovers without a sentinel: clear old charts first so a
    # smaller re-run cannot leave a mixed row set in the manifest's dir.
    for name in os.listdir(args.out_dir):
        if name.startswith("chart_") and name.endswith(".ppm"):
            os.remove(os.path.join(args.out_dir, name))

    manifest = os.path.join(args.out_dir, "images.tsv")
    with open(manifest, "w", encoding="utf-8", newline="\n") as f:
        f.write("# vision.candlestick — synthetic candlestick charts from the "
                "local AAPL_1d.csv (T113b, route (a)).\n")
        f.write("# label = next-bar direction after the rendered window "
                "(close vs last window close; flat tie bar -> window skipped).\n")
        f.write("# generator: tools/make_vision_candles.py render=%s "
                "window=%d stride=%d n=%d canvas=%dx%d PPM P6 RGB "
                "(VisionEncoder geometry)\n"
                % (RENDER_VERSION, args.window, args.stride, len(rows), W, H))
        f.write("# palette: bg=%s up(close>=open)=%s down=%s; body=full slot "
                "open..close, wick=high..low center column; price scaled to "
                "window min-low/max-high, integer rounding; NO RNG — same CSV "
                "+ params => byte-identical PPMs\n"
                % (PALETTE["bg"], PALETTE["up"], PALETTE["down"]))
        f.write("# source: %s\n" % os.path.relpath(args.csv, ROOT).replace("\\", "/"))
        f.write("# %s\n" % CAVEAT)
        f.write("# param_hash=%s\n" % phash)
        f.write("# histogram: up=%d down=%d (NOT force-balanced)\n"
                % (hist["up"], hist["down"]))
        f.write("path\tvision.candlestick\tdirection\n")
        for s, pix, label in rows:
            name = "chart_%04d.ppm" % s
            write_ppm(os.path.join(args.out_dir, name), pix)
            # col2 = LABEL (dump_hidden::read_modality_manifest reads col2 as
            # the label; header col2 = set-name is correct). col3 = marker
            # word, mirroring make_vision_anomaly's <label><TAB><kind> shape.
            f.write("%s\t%s\tdirection\n" % (name, label))

    with open(os.path.join(args.out_dir, "LICENCE.txt"), "w",
              encoding="utf-8", newline="\n") as f:
        f.write("Synthesised from the LOCAL models/market/AAPL_1d.csv "
                "(Yahoo provenance, no download).\n"
                "The generator params in images.tsv ARE the documentation; "
                "see the caveat there.\n")
    with open(sentinel, "w", encoding="utf-8", newline="\n") as f:
        f.write("param_hash=%s\nrender=%s\nwindow=%d\nstride=%d\nn=%d\n"
                % (phash, RENDER_VERSION, args.window, args.stride, len(rows)))

    print("vision.candlestick: %d charts (up=%d down=%d); params %s"
          % (len(rows), hist["up"], hist["down"], phash[:12]))
    print("wrote %s" % manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
