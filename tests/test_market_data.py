#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — tests/test_market_data.py
#
#  Characterization tests for tools/fetch_market_data.py.
#
#  Fully offline and stdlib-only: every network call is intercepted, so this
#  runs on any machine (and in CI) without touching stooq/yahoo.
#
#  What it pins down (the "market-data integrity" contract):
#    1. --source synth NEVER touches the network (routing bug fix).
#    2. A provider failure is explicit: non-zero exit, and NO file written.
#       Synthetic data is never substituted implicitly; --allow-synth opts in.
#    3. Every CSV carries a `source` provenance column.
#    4. --years bounds EVERY source, including stooq (which returns its full
#       history).
#    5. Synthetic output defaults to DEMO_<tf>.csv so it cannot clobber a real
#       ticker's data.
#    6. The emitted CSV still satisfies the C++ load_bars_csv() contract
#       (fields 0..5 = time,open,high,low,close,volume; extras ignored).
#
#  Usage:  python tests/test_market_data.py
# =============================================================================
import contextlib
import datetime as dt
import importlib.util
import io
import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOOL = os.path.join(ROOT, "tools", "fetch_market_data.py")

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
        print(f"  FAIL {name}  {detail}")


# --------------------------------------------------------------------------
# Load the tool as a module (it lives in tools/, not on sys.path).
# --------------------------------------------------------------------------
spec = importlib.util.spec_from_file_location("fetch_market_data", TOOL)
fmd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fmd)


# --------------------------------------------------------------------------
# Network interception
# --------------------------------------------------------------------------
class _Resp:
    def __init__(self, body):
        self._body = body.encode("utf-8") if isinstance(body, str) else body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _install_network(routes):
    """routes: {substring: body-or-Exception}. Returns the list of URLs hit."""
    import urllib.request
    hits = []

    def fake_urlopen(req, timeout=30):        # noqa: ARG001
        url = getattr(req, "full_url", str(req))
        hits.append(url)
        for key, val in routes.items():
            if key in url:
                if isinstance(val, Exception):
                    raise val
                return _Resp(val)
        raise AssertionError(f"unexpected network call: {url}")

    urllib.request.urlopen = fake_urlopen
    return hits


class _NoNetwork:
    """Context manager: any network access is a test failure."""

    def __enter__(self):
        self.hits = _install_network({})       # every URL -> AssertionError
        return self

    def __exit__(self, *exc):
        return False


@contextlib.contextmanager
def patched_env(argv, routes=None):
    """Patch sys.argv + urlopen + time.sleep; capture stdout/stderr."""
    old_argv, old_cwd = sys.argv, os.getcwd()
    old_sleep = time.sleep
    time.sleep = lambda *_: None               # no retry backoff in tests
    sys.argv = ["fetch_market_data.py"] + argv
    hits = _install_network(routes or {})
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = fmd.main()
        yield rc, out.getvalue(), err.getvalue(), hits
    finally:
        sys.argv = old_argv
        time.sleep = old_sleep
        os.chdir(old_cwd)


def stooq_csv(start_year=2000, years=20, step_days=7, base=100.0):
    """A valid Stooq-format daily/weekly CSV spanning `years`."""
    d0 = dt.datetime(start_year, 1, 3, tzinfo=dt.timezone.utc)
    n = int(years * 365.25 / step_days)
    lines = ["Date,Open,High,Low,Close,Volume"]
    for i in range(n):
        d = d0 + dt.timedelta(days=i * step_days)
        px = base + i * 0.01
        lines.append(f"{d:%Y-%m-%d},{px:.4f},{px + 1:.4f},"
                     f"{px - 1:.4f},{px + 0.5:.4f},{1000000 + i}")
    return "\n".join(lines) + "\n"


def read_rows(path):
    with open(path, encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]
    return lines[0], lines[1:]


print("== fetch_market_data: integrity characterization ==")

# --------------------------------------------------------------------------
# 1. synth is deterministic and seeded
# --------------------------------------------------------------------------
a = fmd.synth_rows(30, seed=7)
b = fmd.synth_rows(30, seed=7)
c = fmd.synth_rows(30, seed=8)
check("synth is deterministic for a fixed seed", a == b)
check("synth changes with the seed", a != c)
check("synth row shape is (t,o,h,l,c,v)", len(a[0]) == 6)
check("synth respects high>=max(o,c) and low<=min(o,c)",
      all(r[2] >= max(r[1], r[4]) and r[3] <= min(r[1], r[4]) for r in a))

# --------------------------------------------------------------------------
# 2. --source synth must never touch the network (routing fix)
# --------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    out = os.path.join(tmp, "s.csv")
    with patched_env(["--ticker", "ZZZ", "--source", "synth",
                      "--years", "1", "--out", out]) as (rc, so, se, hits):
        check("--source synth exits 0", rc == 0, f"rc={rc} err={se.strip()}")
        check("--source synth made ZERO network calls", hits == [], f"hits={hits}")
        check("--source synth warns it is synthetic", "SYNTHETIC" in se)
        check("--source synth wrote a file", os.path.exists(out))
        header, rows = read_rows(out)
        check("synth CSV header carries provenance",
              header == "time,open,high,low,close,volume,source", header)
        check("synth rows are tagged source=synth",
              rows and all(r.endswith(",synth") for r in rows))

# --------------------------------------------------------------------------
# 3. an explicit provider failure is explicit: exit 2, nothing written
# --------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    out = os.path.join(tmp, "s.csv")
    routes = {"stooq.com": RuntimeError("boom")}
    with patched_env(["--ticker", "ZZZ", "--source", "stooq",
                      "--out", out], routes) as (rc, so, se, hits):
        check("explicit --source stooq failure exits non-zero", rc == 2, f"rc={rc}")
        check("failed fetch writes NO file", not os.path.exists(out))
        check("failure message names the provider", "stooq" in se, se.strip())
        check("failure refuses to substitute synth", "Refusing" in se, se.strip())

# --------------------------------------------------------------------------
# 4. `auto` never falls back to synth implicitly
# --------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    out = os.path.join(tmp, "s.csv")
    routes = {"stooq.com": RuntimeError("x"), "yahoo.com": RuntimeError("y")}
    with patched_env(["--ticker", "ZZZ", "--out", out], routes) as (rc, so, se, hits):
        check("auto with all providers down exits non-zero", rc == 2, f"rc={rc}")
        check("auto failure writes NO file", not os.path.exists(out))
        check("auto tried both providers", len(hits) >= 2, f"hits={hits}")

    # ...and --allow-synth is the explicit opt-in
    with patched_env(["--ticker", "ZZZ", "--out", out, "--allow-synth"],
                     routes) as (rc, so, se, hits):
        check("auto + --allow-synth succeeds", rc == 0, f"rc={rc} err={se.strip()}")
        check("auto + --allow-synth warns loudly", "SYNTHETIC" in se)
        check("auto + --allow-synth wrote a file", os.path.exists(out))
        check("auto + --allow-synth tags rows source=synth",
              all(r.endswith(",synth") for r in read_rows(out)[1]))

# --------------------------------------------------------------------------
# 5. stooq --years trim (the headline bug: full history was never trimmed)
# --------------------------------------------------------------------------
body = stooq_csv(start_year=2000, years=20, step_days=7)
total_rows = len(body.strip().splitlines()) - 1
with tempfile.TemporaryDirectory() as tmp:
    full = os.path.join(tmp, "full.csv")
    with patched_env(["--ticker", "ZZZ", "--source", "stooq", "--years", "20",
                      "--timeframe", "1wk", "--out", full],
                     {"stooq.com": body}) as (rc, so, se, hits):
        check("stooq fetch exits 0", rc == 0, f"rc={rc} err={se.strip()}")
        header, rows = read_rows(full)
        check("stooq CSV carries provenance",
              header == "time,open,high,low,close,volume,source", header)
        check("stooq rows are tagged source=stooq",
              all(r.endswith(",stooq") for r in rows))
        check("stooq --years 20 keeps the whole history",
              len(rows) == total_rows, f"{len(rows)} vs {total_rows}")

    trim = os.path.join(tmp, "trim.csv")
    with patched_env(["--ticker", "ZZZ", "--source", "stooq", "--years", "2",
                      "--timeframe", "1wk", "--out", trim],
                     {"stooq.com": body}) as (rc, so, se, hits):
        header, rows = read_rows(trim)
        newest = int(rows[-1].split(",")[0])
        oldest = int(rows[0].split(",")[0])
        span_days = (newest - oldest) / 86400.0
        check("stooq --years 2 actually trims", len(rows) < total_rows,
              f"{len(rows)} vs {total_rows}")
        check("stooq --years 2 window is ~2 years (<= 731d)",
              span_days <= 731.0, f"{span_days:.1f}d")
        check("stooq --years 2 keeps a sane amount (~104 weekly bars)",
              90 <= len(rows) <= 120, f"{len(rows)}")
        check("trimmed rows are ascending in time",
              all(int(rows[i].split(",")[0]) <= int(rows[i + 1].split(",")[0])
                  for i in range(len(rows) - 1)))
        check("trim keeps the newest bar",
              newest == int(dt.datetime(2000, 1, 3, tzinfo=dt.timezone.utc)
                            .timestamp()) + (total_rows - 1) * 7 * 86400)

# --------------------------------------------------------------------------
# 6. a provider returning junk/empty data is a failure, not an empty CSV
# --------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    out = os.path.join(tmp, "s.csv")
    with patched_env(["--ticker", "ZZZ", "--source", "stooq", "--out", out],
                     {"stooq.com": "No data\n"}) as (rc, so, se, hits):
        check("stooq 'No data' is an explicit failure", rc == 2, f"rc={rc}")
        check("junk provider body writes no file", not os.path.exists(out))

# --------------------------------------------------------------------------
# 7. synthetic output cannot clobber a real ticker's file
# --------------------------------------------------------------------------
# NB: we chdir into the temp dir to exercise the *default* output path. We
# MUST restore the original cwd before leaving the `with` block: on Windows a
# process cannot remove a directory that is its own current working directory
# (WinError 32), which would make TemporaryDirectory cleanup blow up.
with tempfile.TemporaryDirectory() as tmp:
    _prev_cwd = os.getcwd()
    try:
        os.chdir(tmp)
        with patched_env(["--ticker", "AAPL", "--source", "synth",
                          "--years", "1"]) as (rc, so, se, hits):
            demo = os.path.join(tmp, "models", "market", "DEMO_1d.csv")
            real = os.path.join(tmp, "models", "market", "AAPL_1d.csv")
            check("synth writes DEMO_<tf>.csv", os.path.exists(demo))
            check("synth does NOT write AAPL_1d.csv", not os.path.exists(real))
    finally:
        os.chdir(_prev_cwd)

# --------------------------------------------------------------------------
# 8. the C++ load_bars_csv() contract is preserved
# --------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    out = os.path.join(tmp, "s.csv")
    with patched_env(["--ticker", "ZZZ", "--source", "synth", "--years", "1",
                      "--out", out]) as (rc, so, se, hits):
        header, rows = read_rows(out)
        hdr = header.split(",")
        check("header starts with the 6 columns the loader reads",
              hdr[:6] == ["time", "open", "high", "low", "close", "volume"], header)
        check("header does not start with a digit (loader skips it)",
              not header[0].isdigit())
        ok = True
        for r in rows:
            f = r.split(",")
            if len(f) < 6:
                ok = False
                break
            try:
                t, o, h, l, c, v = (float(f[0]), float(f[1]), float(f[2]),
                                    float(f[3]), float(f[4]), float(f[5]))
            except ValueError:
                ok = False
                break
            # mirrors the loader's acceptance test
            if not (t > 0 and c > 0 and h >= l and h >= c and l <= c):
                ok = False
                break
        check("every row satisfies the loader's validity rules", ok)
        check("provenance column contains no comma (cannot break split_csv)",
              all("," not in r.split(",")[6] for r in rows))

# --------------------------------------------------------------------------
# 9. --years must be positive
# --------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    out = os.path.join(tmp, "s.csv")
    with patched_env(["--ticker", "ZZZ", "--source", "synth", "--years", "0",
                      "--out", out]) as (rc, so, se, hits):
        check("--years 0 is rejected", rc == 2, f"rc={rc}")

print()
print(f"RESULT: {PASSED} passed, {FAILED} failed")
if FAILURES:
    for f in FAILURES:
        print("   -", f)
sys.exit(1 if FAILED else 0)
