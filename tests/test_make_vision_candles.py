#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — tests/test_make_vision_candles.py  (T113b)
#
#  Pins the vision.candlestick generator contract:
#    A. determinism      — two runs, byte-identical PPMs + manifest
#    B. label correctness — next-bar direction recomputed INDEPENDENTLY from
#                           the CSV matches the manifest on every row
#    C. fail-closed      — missing CSV -> 2; bad header -> 2; sentinel
#                           present -> 2 without --force; param change after
#                           a sentinel -> STALE refusal -> 2 (even --force)
#    D. manifest shape   — '#' comment header, path<TAB>label<TAB>direction rows
#                           (col2=label: dump_hidden reads col2 as label),
#                           PPMs decode as P6 32x32 with 3072 pixel bytes
#  Also REPORTS the up/down histogram (never force-balanced).
#  Plain script: exits nonzero on first failure.  Stdlib only.
# =============================================================================
"""Unit tests for tools/make_vision_candles.py (T113b)."""

import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.path.join(ROOT, "tools", "make_vision_candles.py")
CSV = os.path.join(ROOT, "models", "market", "AAPL_1d.csv")
PPM_HDR = b"P6\n32 32\n255\n"

FAILURES = []


def check(ok, msg):
    print("%s %s" % ("PASS" if ok else "FAIL", msg))
    if not ok:
        FAILURES.append(msg)


def run_gen(extra):
    return subprocess.run([sys.executable, GEN] + extra,
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace")


def read_csv_closes():
    closes = []
    with open(CSV, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    assert lines[0].split(",")[:5] == ["time", "open", "high", "low", "close"]
    for ln in lines[1:]:
        closes.append(float(ln.split(",")[4]))
    return closes


def dir_files(d):
    out = {}
    for name in sorted(os.listdir(d)):
        with open(os.path.join(d, name), "rb") as f:
            out[name] = f.read()
    return out


def main():
    closes = read_csv_closes()
    with tempfile.TemporaryDirectory() as td:
        d1 = os.path.join(td, "run1")
        d2 = os.path.join(td, "run2")
        n, window, stride = 20, 8, 3

        # --- A. determinism ------------------------------------------------
        r = run_gen(["--n", str(n), "--window", str(window),
                     "--stride", str(stride), "--out-dir", d1])
        check(r.returncode == 0, "gen run1 exit 0 (got %d: %s)"
              % (r.returncode, (r.stdout + r.stderr)[:200]))
        r = run_gen(["--n", str(n), "--window", str(window),
                     "--stride", str(stride), "--out-dir", d2])
        check(r.returncode == 0, "gen run2 exit 0")
        f1, f2 = dir_files(d1), dir_files(d2)
        check(set(f1) == set(f2) and f1, "run1/run2 same non-empty file set")
        diffs = [k for k in f1 if f1[k] != f2.get(k)]
        check(not diffs, "byte-identical rerun (%d files, diffs=%s)"
              % (len(f1), diffs[:3]))

        # --- D. manifest + PPM shape ---------------------------------------
        mpath = os.path.join(d1, "images.tsv")
        with open(mpath, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        hdr_i = [i for i, ln in enumerate(lines) if not ln.startswith("#")]
        check(bool(hdr_i) and lines[hdr_i[0]] == "path\tvision.candlestick\tdirection",
              "manifest header path\\tvision.candlestick\\tdirection")
        check(all(ln.startswith("#") for ln in lines[:hdr_i[0]]),
              "comment header lines all '#' (dump_hidden vision contract)")
        rows = [ln.split("\t") for ln in lines[hdr_i[0] + 1:] if ln]
        check(len(rows) == n, "manifest has n=%d rows (got %d)" % (n, len(rows)))
        check(all(len(rw) == 3 and rw[1] in ("up", "down") and rw[2] == "direction"
                  for rw in rows),
              "rows are path<TAB>{up,down}<TAB>direction — col2 is the LABEL "
              "(dump_hidden reads col2 as label) (3 cols)")
        ppm_bad = []
        for rw in rows:                       # bare filenames, PPM decodes
            p = os.path.join(d1, rw[0])
            with open(p, "rb") as f:
                data = f.read()
            if data[:len(PPM_HDR)] != PPM_HDR or len(data) != len(PPM_HDR) + 32 * 32 * 3:
                ppm_bad.append(rw[0])
        check(not ppm_bad, "all PPMs decode P6 32x32 3072px (%s)" % ppm_bad[:3])

        # --- B. label correctness (independent recompute, every row) --------
        bad = []
        for rw in rows:
            s = int(rw[0][len("chart_"):-len(".ppm")])
            want = "up" if closes[s + window] > closes[s + window - 1] else (
                "down" if closes[s + window] < closes[s + window - 1] else "tie")
            if rw[1] != want:
                bad.append((rw[0], rw[1], want))
        check(not bad, "labels = independent next-bar recompute (%s)" % bad[:3])

        # --- C. fail-closed ------------------------------------------------
        r = run_gen(["--csv", os.path.join(td, "nope.csv"),
                     "--out-dir", os.path.join(td, "x")])
        check(r.returncode == 2, "missing CSV -> exit 2 (got %d)" % r.returncode)
        badcsv = os.path.join(td, "bad.csv")
        with open(badcsv, "w", encoding="utf-8", newline="\n") as f:
            f.write("oops,open,high,low,close\n1,2,3,4,5\n")
        r = run_gen(["--csv", badcsv, "--out-dir", os.path.join(td, "x")])
        check(r.returncode == 2, "bad CSV header -> exit 2 (got %d)" % r.returncode)

        d3 = os.path.join(td, "run3")
        r = run_gen(["--n", "5", "--window", "8", "--out-dir", d3])
        check(r.returncode == 0, "sentinel baseline run exit 0")
        r = run_gen(["--n", "5", "--window", "8", "--out-dir", d3])
        check(r.returncode == 2, "sentinel present, no --force -> exit 2")
        r = run_gen(["--n", "5", "--window", "8", "--out-dir", d3, "--force"])
        check(r.returncode == 0, "--force same params -> re-render ok")
        r = run_gen(["--n", "5", "--window", "16", "--out-dir", d3, "--force"])
        check(r.returncode == 2 and "STALE" in (r.stdout + r.stderr),
              "param change after sentinel -> STALE refusal exit 2 (got %d)"
              % r.returncode)

        # --- histogram report ----------------------------------------------
        hist = {"up": 0, "down": 0}
        for rw in rows:
            hist[rw[1]] += 1
        print("HISTOGRAM (n=%d, window=%d, stride=%d): up=%d down=%d  "
              "(NOT force-balanced)"
              % (n, window, stride, hist["up"], hist["down"]))

    print("%d checks failed" % len(FAILURES))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
