#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — tests/test_strategy_parity.py
#
#  The C++ strategy zoo, router and sniper (src/trading/strategy_zoo.cpp,
#  router.cpp, sniper.cpp) must produce the SAME NUMBERS as the Python oracles
#  they replace (tools/monster/strategies.py, router.py, sniper_engine.py).
#
#  This test exists because a ~1,400-line numeric hand-port WILL drift from its
#  original. Without it the port is a second implementation that silently
#  disagrees with the first — strictly worse than not porting at all, because
#  the disagreement is invisible and the C++ is the one making decisions.
#
#  It is the gate on retiring the Python from the runtime path: while this is
#  green the two are interchangeable, so the C++ can be trusted alone.
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
        print(f"  FAIL {name}: {detail}")


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def make_bars(n, mode="trend"):
    """Deterministic OHLCV that BOTH parsers will accept unchanged.

    load_bars_csv() drops a row unless time > 0, close > 0, high >= low,
    high >= close and low <= close. Anything we emit must satisfy all of that,
    otherwise the two sides would be reading different bar counts — and a
    parity test comparing different inputs proves nothing.
    """
    bars = []
    state = 987654321

    def rnd():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) % (2 ** 64)
        return ((state >> 11) & 0x1FFFFFFFFFFFFF) / float(0x1FFFFFFFFFFFFF)

    px = 100.0
    for i in range(n):
        if mode == "trend":
            drift, amp = 0.0022, 0.006
        elif mode == "chop":
            px = 100.0 + 2.5 * math.sin(i * 0.55)
            drift, amp = 0.0, 0.005
        elif mode == "vol":
            drift, amp = 0.0, 0.035
        elif mode == "squeeze":
            # A coiling range: shrinking amplitude, so Bollinger moves inside
            # Keltner and the breakout strategy's squeeze flag has something to
            # find. Without this the squeeze branch is never exercised.
            amp = max(0.0015, 0.03 * (1.0 - i / float(n)))
            drift = 0.0
        else:
            drift, amp = 0.0, 0.003
        r = drift + amp * (rnd() - 0.5)
        o = px
        c = px * (1.0 + r)
        hi = max(o, c) * (1.0 + amp * 0.5)
        lo = min(o, c) * (1.0 - amp * 0.5)
        hi = max(hi, o, c)
        lo = min(lo, o, c)
        bars.append((1700000000 + i * 86400, o, hi, lo, c, 1000.0 + i * 3.0))
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
            v = float(p[5]) if len(p) >= 6 else 0.0
            if t > 0 and c > 0.0 and h >= l and h >= c and l <= c:
                out.append((t, o, h, l, c, v))
    out.sort(key=lambda b: b[0])
    return out


# ---------------------------------------------------------------------------
def find_dump():
    exe = "omniseed_strategy_dump.exe" if os.name == "nt" else "omniseed_strategy_dump"
    p = os.path.join(ROOT, "build", "bin", exe)
    return p if os.path.exists(p) else None


def run_dump(dump, csv, extra):
    cmd = [dump, csv] + extra
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        raise RuntimeError(f"strategy_dump failed rc={proc.returncode}: "
                           f"{proc.stderr.strip()[:300]}")
    return json.loads(proc.stdout)


def close_enough(a, b):
    """Both NaN -> equal. Otherwise a tight relative comparison.

    Float summation order differs between the languages (Python's sum() over a
    list vs a plain C++ loop), so exact bit equality is not achievable. 1e-9
    relative is far tighter than any behavioural difference we care about and
    still catches a genuine divergence.
    """
    a_nan = a is None or (isinstance(a, float) and math.isnan(a))
    b_nan = b is None or (isinstance(b, float) and math.isnan(b))
    if a_nan or b_nan:
        return a_nan and b_nan
    return abs(a - b) <= 1e-9 + 1e-9 * max(abs(a), abs(b))


def load_oracles():
    """Import the Python oracles. Returns None (with a reason) when unavailable."""
    if TOOLS not in sys.path:
        sys.path.insert(0, TOOLS)
    try:
        import monster.regime as RG          # noqa: F401
        import monster.strategies as ST
        import monster.router as RT
        import monster.sniper_engine as SN
        return RG, ST, RT, SN
    except Exception as exc:                 # pragma: no cover - env dependent
        return ("unavailable", repr(exc))


# ---------------------------------------------------------------------------
# strategies + router parity
# ---------------------------------------------------------------------------
ROUTER_FIELDS = ("conviction", "agreement", "weight", "w_trend", "size_factor")


def compare_router(label, bars, cpp, ST, RT, RG, tracked=True):
    series = ST.prepare(bars)
    regimes = RG.scan(bars, tracked=tracked)
    rows = cpp["rows"]
    if len(rows) != len(bars):
        check(f"parity/{label}: bar count", False,
              f"py {len(bars)} vs cpp {len(rows)}")
        return
    check(f"parity/{label}: bar count", True)

    bad = []
    worst = {}
    n_reason_checked = 0
    for i, row in enumerate(rows):
        reg = regimes[i]
        sigs = ST.all_signals(series, i, reg)
        v = RT.route(sigs, reg, RT.RouterConfig(), ts=series.ts[i])

        cpp_sigs = row["signals"]
        if len(cpp_sigs) != len(sigs):
            bad.append(f"bar {i}: signal count py={len(sigs)} cpp={len(cpp_sigs)}")
            continue
        for ps, cs in zip(sigs, cpp_sigs):
            if ps.name != cs["name"]:
                bad.append(f"bar {i}: name py={ps.name} cpp={cs['name']}")
            for fld, pv, cv in (("direction", ps.direction, cs["direction"]),
                                ("confidence", ps.confidence, cs["confidence"])):
                if not close_enough(pv, cv):
                    d = abs((pv or 0.0) - (cv or 0.0))
                    worst[f"{ps.name}.{fld}"] = max(worst.get(f"{ps.name}.{fld}", 0.0), d)
                    bad.append(f"bar {i} {ps.name}.{fld}: py={pv!r} cpp={cv!r}")
            if ps.regime_fit != cs["fit"]:
                bad.append(f"bar {i} {ps.name}.fit: py={ps.regime_fit} cpp={cs['fit']}")
            # Reasons are formatted strings; comparing them catches a diverged
            # branch (e.g. "not-reverting" vs "below-entry") that the numbers
            # alone might not, since both can be direction=0/conf=0.
            n_reason_checked += 1
            if ps.reason != cs["reason"]:
                bad.append(f"bar {i} {ps.name}.reason: py={ps.reason!r} "
                           f"cpp={cs['reason']!r}")

        for fld in ROUTER_FIELDS:
            pv, cv = getattr(v, fld), row[fld]
            if not close_enough(pv, cv):
                d = abs((pv or 0.0) - (cv or 0.0))
                worst[fld] = max(worst.get(fld, 0.0), d)
                bad.append(f"bar {i} router.{fld}: py={pv!r} cpp={cv!r}")
        if int(v.active) != int(row["active"]):
            bad.append(f"bar {i} router.active: py={v.active} cpp={row['active']}")
        if bool(v.opposed_long) != bool(row["opposed_long"]):
            bad.append(f"bar {i} router.opposed_long: py={v.opposed_long} "
                       f"cpp={row['opposed_long']}")
        py_veto = RT.should_veto_long(v, RT.RouterConfig())
        if bool(py_veto) != bool(row["veto_long"]):
            bad.append(f"bar {i} router.veto_long: py={py_veto} cpp={row['veto_long']}")

    check(f"parity/{label}: signals + router match on every bar", not bad,
          "; ".join(bad[:5]))
    if worst:
        print("       worst abs deltas: " +
              ", ".join(f"{k}={v:.3e}" for k, v in sorted(worst.items())))
    return {"bad": bad, "reason_checks": n_reason_checked}


# ---------------------------------------------------------------------------
# sniper parity
# ---------------------------------------------------------------------------
SNIPER_FIELDS = ("score", "micro", "tech", "regime_score", "cross",
                 "trend_score", "half_life", "conviction", "agreement")


def compare_sniper(label, bars, cpp, SN, ST, RT, RG, **snkw):
    cfg = SN.SniperConfig(**snkw)
    prep = SN.prepare(bars, cfg)
    rows = cpp["rows"]
    if len(rows) != len(bars):
        check(f"parity/{label}: bar count", False,
              f"py {len(bars)} vs cpp {len(rows)}")
        return
    check(f"parity/{label}: bar count", True)

    bad = []
    worst = {}
    n_proposed = 0
    n_vetoed = 0
    for i, row in enumerate(rows):
        pv = SN.evaluate(prep, i, None, cfg)
        cv = row["sniper"]
        for fld in SNIPER_FIELDS:
            a, b = getattr(pv, fld), cv[fld]
            if not close_enough(a, b):
                d = abs((a or 0.0) - (b or 0.0))
                worst[fld] = max(worst.get(fld, 0.0), d)
                bad.append(f"bar {i} {fld}: py={a!r} cpp={b!r}")
        if pv.votes != cv["votes"]:
            bad.append(f"bar {i} votes: py={pv.votes} cpp={cv['votes']}")
        if pv.regime != cv["regime"]:
            bad.append(f"bar {i} regime: py={pv.regime} cpp={cv['regime']}")
        if bool(pv.veto) != bool(cv["veto"]):
            bad.append(f"bar {i} veto: py={pv.veto} cpp={cv['veto']}")
        if pv.veto_reason != cv["veto_reason"]:
            bad.append(f"bar {i} veto_reason: py={pv.veto_reason!r} "
                       f"cpp={cv['veto_reason']!r}")
        if list(pv.factors) != list(cv["factors"]):
            bad.append(f"bar {i} factors: py={pv.factors} cpp={cv['factors']}")
        if bool(pv.propose(cfg)) != bool(cv["propose"]):
            bad.append(f"bar {i} propose: py={pv.propose(cfg)} cpp={cv['propose']}")
        n_proposed += 1 if cv["propose"] else 0
        n_vetoed += 1 if cv["veto"] else 0

    check(f"parity/{label}: sniper matches on every bar", not bad,
          "; ".join(bad[:5]))
    if worst:
        print("       worst abs deltas: " +
              ", ".join(f"{k}={v:.3e}" for k, v in sorted(worst.items())))
    print(f"       ({n_proposed} proposed, {n_vetoed} vetoed of {len(rows)} bars)")
    return {"proposed": n_proposed, "vetoed": n_vetoed}


# ---------------------------------------------------------------------------
def main():
    print("== strategy parity: C++ zoo/router/sniper vs the Python oracles ==")
    dump = find_dump()
    if dump is None:
        print("  --   skip: build/bin/omniseed_strategy_dump not built "
              "(parity not exercised here)")
        print(f"\n---- strategy parity: {PASSED} passed, {FAILED} failed ----")
        return FAILED == 0

    oracles = load_oracles()
    if isinstance(oracles, tuple) and oracles and oracles[0] == "unavailable":
        print(f"  --   skip: Python oracle import failed: {oracles[1]}")
        print(f"\n---- strategy parity: {PASSED} passed, {FAILED} failed ----")
        return FAILED == 0
    RG, ST, RT, SN = oracles

    tmp = tempfile.mkdtemp(prefix="omniseed_strategy_parity_")
    try:
        cases = [
            ("trend", make_bars(600, "trend")),
            ("chop", make_bars(600, "chop")),
            ("squeeze", make_bars(400, "squeeze")),
            ("vol", make_bars(400, "vol")),
        ]
        for name, bars in cases:
            csv = os.path.join(tmp, f"{name}.csv")
            write_csv(csv, bars)
            parsed = read_csv_cpp_equivalent(csv)
            cpp = run_dump(dump, csv, [])
            compare_router(f"{name}/tracked", parsed, cpp, ST, RT, RG, tracked=True)
            cpp_raw = run_dump(dump, csv, ["--raw"])
            compare_router(f"{name}/raw", parsed, cpp_raw, ST, RT, RG, tracked=False)

        # ---- sniper: default (advanced regime, no ensemble) ----------------
        #
        # Kept to 250 bars: the oracle's `last_swing_pair` is O(lows x highs)
        # PER BAR, so a full 1255-bar feed is quadratic-to-cubic in Python and
        # would take minutes. 250 bars is far more than the 200-bar regime
        # window needs to warm up, so every branch is still exercised.
        for name in ("trend", "chop", "vol"):
            bars = make_bars(250, name)
            csv = os.path.join(tmp, f"sn_{name}.csv")
            write_csv(csv, bars)
            parsed = read_csv_cpp_equivalent(csv)
            cpp = run_dump(dump, csv, ["--sniper"])
            compare_sniper(f"sniper/{name}/advanced", parsed, cpp, SN, ST, RT, RG)
            cpp_leg = run_dump(dump, csv, ["--sniper", "--legacy"])
            compare_sniper(f"sniper/{name}/legacy", parsed, cpp_leg, SN, ST, RT, RG,
                           regime_mode="legacy")
            cpp_ens = run_dump(dump, csv, ["--sniper", "--ensemble"])
            compare_sniper(f"sniper/{name}/ensemble", parsed, cpp_ens, SN, ST, RT,
                           RG, ensemble=True)

        # ---- real market data, when this worktree happens to have a feed ----
        #
        # The session worktree has no .venv and no market CSVs; the owner's
        # checkout does. Probe both honestly rather than silently skipping a
        # case the user believes ran.
        real = None
        for cand in (os.path.join(ROOT, "models", "market", "AAPL_1d.csv"),
                     os.path.expanduser(
                         "~/OneDrive/Desktop/omniseed/models/market/AAPL_1d.csv"),
                     os.path.expanduser(
                         "~/Desktop/omniseed/models/market/AAPL_1d.csv")):
            if os.path.exists(cand):
                real = cand
                break
        if real:
            parsed = read_csv_cpp_equivalent(real)
            cpp = run_dump(dump, real, [])
            compare_router("real-feed/tracked", parsed, cpp, ST, RT, RG,
                           tracked=True)
            head = parsed[:250]
            csv = os.path.join(tmp, "real_head.csv")
            write_csv(csv, head)
            parsed_head = read_csv_cpp_equivalent(csv)
            cpp_s = run_dump(dump, csv, ["--sniper", "--ensemble"])
            compare_sniper("real-feed/sniper-ensemble", parsed_head, cpp_s, SN, ST,
                           RT, RG, ensemble=True)
        else:
            print("  --   no real feed found; real-data parity not exercised")
    finally:
        for fn in os.listdir(tmp):
            try:
                os.remove(os.path.join(tmp, fn))
            except OSError:
                pass
        try:
            os.rmdir(tmp)
        except OSError:
            pass

    print(f"\n---- strategy parity: {PASSED} passed, {FAILED} failed ----")
    if FAILURES:
        print("failures:")
        for f in FAILURES[:10]:
            print(f"  - {f}")
    return FAILED == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
