#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — tests/test_regime_parity.py
#
#  The C++ regime engine (src/trading/regime_engine.cpp) must produce the SAME
#  NUMBERS as the Python oracle (tools/monster/regime.py) it replaces.
#
#  This test exists because a 700-line numeric hand-port WILL drift from its
#  original. Without it the port is a second implementation that silently
#  disagrees with the first — which is strictly worse than not porting at all,
#  because the disagreement is invisible.
#
#  It is the gate on retiring the Python from the runtime path: as long as this
#  is green, the two are interchangeable and the C++ can be trusted alone.
#
#  Stdlib only. Skips honestly (and says why) when the dump binary is absent.
# =============================================================================
import json
import math
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

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
        print(f"  FAIL {name}: {detail}")


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def make_bars(n, mode="trend"):
    """Deterministic OHLCV that BOTH parsers will accept unchanged.

    load_bars_csv() drops a row unless time > 0, close > 0, high >= low,
    high >= close and low <= close. Anything we emit must satisfy all of that,
    otherwise the two sides would be reading different bar counts.
    """
    bars = []
    state = 12345

    def rnd():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) % (2 ** 64)
        return ((state >> 11) & 0x1FFFFFFFFFFFFF) / float(0x1FFFFFFFFFFFFF)

    px = 100.0
    for i in range(n):
        if mode == "trend":
            drift, amp = 0.0025, 0.004
        elif mode == "chop":
            px = 100.0 + 2.0 * math.sin(i * 0.7)
            drift, amp = 0.0, 0.004
        elif mode == "vol":
            drift, amp = 0.0, 0.03
        else:
            drift, amp = 0.0, 0.002
        r = drift + amp * (rnd() - 0.5)
        o = px
        c = px * (1.0 + r)
        hi = max(o, c) * (1.0 + amp * 0.5)
        lo = min(o, c) * (1.0 - amp * 0.5)
        # keep the row valid for load_bars_csv's filter
        hi = max(hi, o, c)
        lo = min(lo, o, c)
        bars.append((1700000000 + i * 86400, o, hi, lo, c, 1000.0 + i))
        px = c
    return bars


def write_csv(path, bars):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("time,open,high,low,close,volume\n")
        for t, o, h, l, c, v in bars:
            f.write(f"{t},{o:.10f},{h:.10f},{l:.10f},{c:.10f},{v:.4f}\n")


def read_csv_cpp_equivalent(path):
    """Read the CSV applying the SAME acceptance rules as load_bars_csv()."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            s = line.strip()
            if not s:
                continue
            if idx == 0 or (not s[0].isdigit()):
                continue                      # header
            p = s.split(",")
            if len(p) < 5:
                continue
            try:
                t = int(float(p[0]))
                o, h, l, c = (float(p[1]), float(p[2]), float(p[3]),
                              float(p[4]))
            except ValueError:
                continue
            if t > 0 and c > 0.0 and h >= l and h >= c and l <= c:
                out.append((t, o, h, l, c))
    out.sort(key=lambda b: b[0])
    return out


# ---------------------------------------------------------------------------
def find_dump():
    exe = "omniseed_regime_dump.exe" if os.name == "nt" else "omniseed_regime_dump"
    p = os.path.join(ROOT, "build", "bin", exe)
    return p if os.path.exists(p) else None


def run_dump(dump, csv, tracked, window=None):
    cmd = [dump, csv]
    if window:
        cmd += ["--window", str(window)]
    cmd.append("--tracked" if tracked else "--raw")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"regime_dump failed rc={proc.returncode}: "
                           f"{proc.stderr.strip()[:300]}")
    return json.loads(proc.stdout)


NUM_FIELDS = ("trend_score", "vol_score", "adx", "er", "chop", "vr", "vr_z",
              "hurst", "r2", "rho1", "yz_vol", "half_life")


def close_enough(a, b):
    """Both NaN -> equal. Otherwise a tight relative comparison.

    Float summation order differs between the two languages (Python's sum() vs
    a plain C++ loop), so exact bit equality is not achievable; 1e-9 relative
    is far tighter than any behavioural difference we care about and still
    catches a genuine divergence.
    """
    a_nan = a is None or (isinstance(a, float) and math.isnan(a))
    b_nan = b is None or (isinstance(b, float) and math.isnan(b))
    if a_nan or b_nan:
        return a_nan and b_nan
    return abs(a - b) <= 1e-9 + 1e-9 * max(abs(a), abs(b))


def compare(label, py_states, cpp, tol_label=True):
    rows = cpp["rows"]
    if len(rows) != len(py_states):
        check(f"parity/{label}: bar count", False,
              f"py {len(py_states)} vs cpp {len(rows)}")
        return
    check(f"parity/{label}: bar count", True)

    worst = {}
    bad = []
    for i, (ps, cs) in enumerate(zip(py_states, rows)):
        for fld in NUM_FIELDS:
            pv = getattr(ps, fld)
            cv = cs.get(fld)
            if not close_enough(pv, cv):
                d = abs((pv or 0.0) - (cv or 0.0))
                worst[fld] = max(worst.get(fld, 0.0), d)
                if len(bad) < 5:
                    bad.append(f"bar {i} {fld}: py={pv!r} cpp={cv!r}")
        if tol_label:
            if ps.label != cs.get("label"):
                bad.append(f"bar {i} label: py={ps.label} cpp={cs.get('label')}")
            if (ps.direction or "") != (cs.get("direction") or ""):
                bad.append(f"bar {i} direction: py={ps.direction!r} "
                           f"cpp={cs.get('direction')!r}")
            if bool(ps.stressed) != bool(cs.get("stressed")):
                bad.append(f"bar {i} stressed: py={ps.stressed} "
                           f"cpp={cs.get('stressed')}")

    check(f"parity/{label}: every numeric field matches", not bad,
          "; ".join(bad[:5]))
    if worst:
        print("       worst abs deltas: " +
              ", ".join(f"{k}={v:.3e}" for k, v in sorted(worst.items())))
    return worst


# ---------------------------------------------------------------------------
def main():
    print("== regime parity: C++ vs the Python oracle ==")
    dump = find_dump()
    if dump is None:
        print("  --   skip: build/bin/omniseed_regime_dump not built "
              "(parity not exercised here)")
        print(f"\n---- regime parity: {PASSED} passed, {FAILED} failed ----")
        return 0 if FAILED == 0 else 1

    sys.path.insert(0, os.path.join(ROOT, "tools"))
    try:
        from monster import regime as R
    except Exception as e:                       # noqa: BLE001
        print(f"  --   skip: cannot import the Python oracle ({e})")
        print(f"\n---- regime parity: {PASSED} passed, {FAILED} failed ----")
        return 0 if FAILED == 0 else 1

    tmp = tempfile.mkdtemp(prefix="omniseed_regime_parity_")
    try:
        cases = [("trend", 320), ("chop", 320), ("vol", 320)]
        # A real feed, when one is reachable, is the strongest case of all.
        real = os.environ.get("OMNISEED_MARKET_CSV")
        if not real:
            for rel in ("models/market/AAPL_1d.csv", "models/market/DEMO_1d.csv"):
                p = os.path.join(ROOT, rel)
                if os.path.exists(p):
                    real = p
                    break

        for mode, n in cases:
            bars = make_bars(n, mode)
            csv = os.path.join(tmp, f"{mode}.csv")
            write_csv(csv, bars)
            read_bars = read_csv_cpp_equivalent(csv)

            py_raw = R.scan(read_bars, tracked=False)
            cpp_raw = run_dump(dump, csv, tracked=False)
            print(f"-- {mode} ({len(read_bars)} bars, raw)")
            compare(mode, py_raw, cpp_raw)

            py_trk = R.scan(read_bars, tracked=True)
            cpp_trk = run_dump(dump, csv, tracked=True)
            print(f"-- {mode} (tracked/hysteresis)")
            compare(f"{mode}/tracked", py_trk, cpp_trk)

        if real:
            print(f"-- real feed: {real}")
            read_bars = read_csv_cpp_equivalent(real)
            if read_bars:
                py_raw = R.scan(read_bars, tracked=False)
                cpp_raw = run_dump(dump, real, tracked=False)
                compare("real/raw", py_raw, cpp_raw)
                py_trk = R.scan(read_bars, tracked=True)
                cpp_trk = run_dump(dump, real, tracked=True)
                compare("real/tracked", py_trk, cpp_trk)

        # A non-default window must move both sides identically.
        bars = make_bars(320, "trend")
        csv = os.path.join(tmp, "win.csv")
        write_csv(csv, bars)
        read_bars = read_csv_cpp_equivalent(csv)
        cfg = R.RegimeConfig(window=120)
        py_w = [R.detect(read_bars, i, cfg) for i in range(len(read_bars))]
        cpp_w = run_dump(dump, csv, tracked=False, window=120)
        print("-- window=120")
        compare("window120", py_w, cpp_w)
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n---- regime parity: {PASSED} passed, {FAILED} failed ----")
    for f in FAILURES:
        print(f"  FAILED {f}")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
