#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — tests/test_market_feeds.py
#
#  Characterization tests for tools/market_feeds.py (M1 market perception).
#
#  Fully offline and stdlib-only: every network call is intercepted, so this
#  runs on any machine (and in CI) without touching binance/yahoo/etc.
#
#  What it pins down:
#    1. Symbol normalization per asset class (and refusal of bad input).
#    2. The rate limiter actually backs off, per key.
#    3. The TTL cache: miss -> put -> hit -> expiry, with counters.
#    4. HttpClient: cache short-circuits the network, retries then raises a
#       ProviderError naming the host, and non-JSON bodies are failures.
#    5. Every bar adapter parses its provider's real payload shape.
#    6. CSV output REUSES fetch_market_data.emit_csv, so the T6 provenance
#       column is single-sourced.
#    7. A provider failure is explicit: exit 2 and NO file written.
#    8. Meme is ticker/research-only (no OHLCV path).
#    9. The feed-health/abstain policy mirrors the C++ FeedPolicy.
#
#  Usage:  python tests/test_market_feeds.py
# =============================================================================
import contextlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOOL = os.path.join(ROOT, "tools", "market_feeds.py")

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
spec = importlib.util.spec_from_file_location("market_feeds", TOOL)
mf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mf)

# No real retry backoff in tests.
mf.time.sleep = lambda *_: None


# --------------------------------------------------------------------------
# Helpers
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


def install_network(routes):
    """routes: {substring: body-or-Exception}. Returns the list of URLs hit."""
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


@contextlib.contextmanager
def tmpdir():
    d = tempfile.mkdtemp(prefix="omf_")
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


@contextlib.contextmanager
def run_main(argv):
    old = sys.argv
    sys.argv = ["market_feeds.py"] + argv
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = mf.main()
        yield rc, out.getvalue(), err.getvalue()
    finally:
        sys.argv = old


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f if l.strip()]


def quiet_http(cache=None, retries=0):
    return mf.HttpClient(cache=cache,
                         limiter=mf.RateLimiter(per_minute=100000,
                                                sleep=lambda *_: None),
                         retries=retries)


BINANCE_KLINES = json.dumps([
    [1700000000000, "100.0", "110.0", "90.0", "105.0", "1234.5",
     1700086399999, "x", 10, "1", "2", "3"],
    [1700086400000, "105.0", "115.0", "95.0", "100.0", "2000.0",
     1700172799999, "x", 10, "1", "2", "3"],
])

BINANCE_TICKER = json.dumps({
    "lastPrice": "50000.0", "priceChangePercent": "2.5",
    "quoteVolume": "1000000.0", "closeTime": 1700000000000,
})

print("== market_feeds: perception integrity characterization ==")

# --------------------------------------------------------------------------
# 1. Symbol normalization
# --------------------------------------------------------------------------
check("normalize: equity is uppercased",
      mf.normalize("equity", "aapl")[0] == "AAPL")
check("normalize: crypto defaults to a USDT quote",
      mf.normalize("crypto", "btc")[0] == "BTCUSDT")
check("normalize: crypto keeps an explicit quote",
      mf.normalize("crypto", "ETHBTC")[0] == "ETHBTC")
check("normalize: fx splits base/quote",
      mf.normalize("fx", "eurusd")[1] == ("EUR", "USD"))
check("normalize: fx strips separators",
      mf.normalize("fx", "eur/usd")[0] == "EURUSD")
check("normalize: meme passes through", mf.normalize("meme", "0xAbC")[0] == "0xAbC")
try:
    mf.normalize("fx", "EUR")
    _bad_fx = False
except mf.ProviderError:
    _bad_fx = True
check("normalize: a short fx pair is refused", _bad_fx)

# --------------------------------------------------------------------------
# 2. Rate limiter
# --------------------------------------------------------------------------
slept = []
rl = mf.RateLimiter(per_minute=60.0, sleep=slept.append)   # 1.0s interval
rl.wait("host")
check("rate limiter: the first call does not sleep", slept == [])
rl.wait("host")
check("rate limiter: the second call backs off ~min_interval",
      len(slept) == 1 and abs(slept[0] - 1.0) < 0.05, f"slept={slept}")
rl.wait("other-host")
check("rate limiter: keys are isolated", len(slept) == 1)

# --------------------------------------------------------------------------
# 3. Cache
# --------------------------------------------------------------------------
with tmpdir() as d:
    c = mf.Cache(d, ttl_secs=100)
    check("cache: miss before put", c.get("k", now=1000) is None)
    c.put("k", "v", now=1000)
    check("cache: hit within the ttl", c.get("k", now=1050) == "v")
    check("cache: miss after the ttl", c.get("k", now=1200) is None)
    check("cache: counters track hits and misses",
          c.hits == 1 and c.misses == 2, f"hits={c.hits} misses={c.misses}")

# --------------------------------------------------------------------------
# 4. HttpClient
# --------------------------------------------------------------------------
with tmpdir() as d:
    http = quiet_http(cache=mf.Cache(d, ttl_secs=1000))
    hits = install_network({"example.com": '{"a": 1}'})
    j1 = http.get_json("https://example.com/x", key="k")
    j2 = http.get_json("https://example.com/x", key="k")
    check("http: parses JSON", j1 == {"a": 1})
    check("http: the second call is served from cache", len(hits) == 1, f"hits={hits}")
    check("http: counts a cache hit", http.cache_hits == 1)

http2 = quiet_http(retries=1)
install_network({"boom.com": RuntimeError("down")})
try:
    http2.get_text("https://boom.com/x")
    _raised = False
except mf.ProviderError as e:
    _raised = "boom.com" in str(e)
check("http: retries then raises a ProviderError naming the host", _raised)

http3 = quiet_http()
install_network({"bad.com": "definitely not json"})
try:
    http3.get_json("https://bad.com/x")
    _raised = False
except mf.ProviderError:
    _raised = True
check("http: a non-JSON body is an explicit failure", _raised)

# --------------------------------------------------------------------------
# 5. Bar adapters
# --------------------------------------------------------------------------
install_network({"api.binance.com": BINANCE_KLINES})
rows = mf.bars_binance("BTCUSDT", "1d", 1, quiet_http())
check("binance: parses klines into 6-field rows",
      len(rows) == 2 and all(len(r) == 6 for r in rows))
check("binance: ms timestamps become seconds", rows[0][0] == 1700000000)
check("binance: OHLC is preserved",
      tuple(rows[0][1:5]) == ("100.0", "110.0", "90.0", "105.0"))

YAHOO_BODY = json.dumps({"chart": {"result": [{
    "timestamp": [1700000000, 1700086400, 1700172800],
    "indicators": {"quote": [{
        "open":  [1.0, 2.0, None],
        "high":  [3.0, 4.0, None],
        "low":   [0.5, 1.5, None],
        "close": [2.5, 3.5, None],
        "volume": [10, 20, 30],
    }]},
}]}})
install_network({"query1.finance.yahoo.com": YAHOO_BODY})
yrows = mf.bars_yahoo("AAPL", "1d", 1, quiet_http())
check("yahoo: parses chart bars", len(yrows) == 2, f"got {len(yrows)}")
check("yahoo: bars with null OHLC are dropped", len(yrows) == 2)

FRANK_BODY = json.dumps({"base": "EUR", "rates": {
    "2023-01-02": {"USD": 1.05}, "2023-01-03": {"USD": 1.06}}})
install_network({"api.frankfurter.app": FRANK_BODY})
frows = mf.bars_frankfurter("EURUSD", "1d", 1, quiet_http())
check("frankfurter: parses ECB daily rates", len(frows) == 2)
check("frankfurter: a single rate becomes flat OHLC",
      frows[0][1] == frows[0][2] == frows[0][3] == frows[0][4])

CG_BODY = json.dumps({
    "prices": [[1700000000000, 100.0], [1700086400000, 101.0]],
    "total_volumes": [[1700000000000, 5000.0], [1700086400000, 6000.0]]})
install_network({"api.coingecko.com": CG_BODY})
crows = mf.bars_coingecko("BTCUSDT", "1d", 1, quiet_http())
check("coingecko: spot points become flat OHLC",
      len(crows) == 2 and crows[0][1] == crows[0][4])
try:
    mf.bars_coingecko("WEIRDCOIN", "1d", 1, quiet_http())
    _raised = False
except mf.ProviderError:
    _raised = True
check("coingecko: an unmapped coin is an explicit failure", _raised)

# --------------------------------------------------------------------------
# 6. CSV output reuses the T6 emitter (single-sourced provenance)
# --------------------------------------------------------------------------
text = mf.fmd.emit_csv([(1700000000, 100, 110, 90, 105, 1000)], "binance", 1)
lines = text.strip().splitlines()
check("csv: header is the T6 schema",
      lines[0] == "time,open,high,low,close,volume,source", lines[0])
check("csv: rows carry the provider as provenance",
      lines[1].endswith(",binance"), lines[1])

# --------------------------------------------------------------------------
# 7. fetch_bars: explicit failure when every provider is down
# --------------------------------------------------------------------------
install_network({"api.binance.com": RuntimeError("down"),
                 "api.coingecko.com": RuntimeError("down")})
try:
    mf.fetch_bars("crypto", "BTCUSDT", "1d", 1, quiet_http())
    _raised = False
except mf.ProviderError as e:
    _raised = "every provider failed" in str(e)
check("fetch_bars: all providers down -> ProviderError", _raised)

# --------------------------------------------------------------------------
# 8. probe_ticker + feed-health policy
# --------------------------------------------------------------------------
install_network({"api.binance.com": BINANCE_TICKER})
_old_now = mf._now
try:
    mf._now = lambda: 1700000000            # pretend "now" == the data ts
    tk, state, abstain, reason = mf.probe_ticker("crypto", "BTCUSDT", quiet_http())
    check("probe: returns a ticker snapshot", tk["last"] == 50000.0)
    check("probe: records the provider", tk["provider"] == "binance")
    check("probe: a fresh feed is ok and does not abstain",
          state == "ok" and abstain is False and reason == "none",
          f"state={state} abstain={abstain} reason={reason}")

    mf._now = lambda: 1700000000 + 1000     # 1000s later -> dead
    _, state2, abstain2, reason2 = mf.probe_ticker("crypto", "BTCUSDT", quiet_http())
    check("probe: a stale feed abstains with dead-feed",
          state2 == "dead" and abstain2 is True and reason2 == "dead-feed",
          f"state={state2} reason={reason2}")
finally:
    mf._now = _old_now

check("policy: never-seen is missing", mf.feed_state(None, 0) == "missing")
check("policy: 10s is ok", mf.feed_state(10, 0) == "ok")
check("policy: 300s is degraded", mf.feed_state(300, 0) == "degraded")
check("policy: 900s is dead", mf.feed_state(900, 0) == "dead")
check("policy: 3 consecutive failures is dead", mf.feed_state(10, 3) == "dead")
check("policy: abstain mapping for degraded",
      mf.abstain_decision("degraded", 0, 0) == (True, "stale-feed"))
check("policy: abstain mapping for ok",
      mf.abstain_decision("ok", 0, 0) == (False, "none"))
check("policy: the liquidity floor abstains",
      mf.abstain_decision("ok", 10, 100) == (True, "low-liquidity"))

# --------------------------------------------------------------------------
# 9. main(): end-to-end bar fetch writes a provenance-tagged CSV
# --------------------------------------------------------------------------
with tmpdir() as d:
    out = os.path.join(d, "BTCUSDT_1d.csv")
    install_network({"api.binance.com": BINANCE_KLINES})
    with run_main(["--asset", "crypto", "--symbol", "BTCUSDT", "--years", "1",
                   "--out", out, "--cache-dir", os.path.join(d, "cache")]) as (rc, so, se,):
        check("main: bars mode exits 0", rc == 0, f"rc={rc} err={se.strip()}")
    rows_out = read_lines(out)
    check("main: wrote a provenance header",
          rows_out[0] == "time,open,high,low,close,volume,source", rows_out[0])
    check("main: rows tagged with the provider",
          all(r.endswith(",binance") for r in rows_out[1:]))

# --------------------------------------------------------------------------
# 10. main(): provider failure is explicit (exit 2, no file)
# --------------------------------------------------------------------------
with tmpdir() as d:
    out2 = os.path.join(d, "BTCUSDT_1d.csv")
    install_network({"api.binance.com": RuntimeError("x"),
                     "api.coingecko.com": RuntimeError("y")})
    with run_main(["--asset", "crypto", "--symbol", "BTCUSDT", "--years", "1",
                   "--out", out2, "--cache-dir", os.path.join(d, "cache")]) as (rc, so, se):
        check("main: every provider down -> exit 2", rc == 2, f"rc={rc}")
        check("main: refuses to substitute synthetic data", "Refusing" in se or
              "refusing" in se, se.strip())
    check("main: nothing written on failure", not os.path.exists(out2))

# --------------------------------------------------------------------------
# 11. main(): meme is research-only (no OHLCV path)
# --------------------------------------------------------------------------
with tmpdir() as d:
    out3 = os.path.join(d, "meme_1d.csv")
    with run_main(["--asset", "meme", "--symbol", "0xabc", "--out", out3]) as (rc, so, se):
        check("main: meme bars mode is refused", rc == 2, f"rc={rc}")
        check("main: the refusal explains why", "research-only" in se, se.strip())
    check("main: meme wrote no file", not os.path.exists(out3))

# --------------------------------------------------------------------------
# 12. main(): argument validation + probe mode
# --------------------------------------------------------------------------
with run_main(["--years", "0"]) as (rc, so, se):
    check("main: --years 0 is rejected", rc == 2, f"rc={rc}")

install_network({"api.binance.com": BINANCE_TICKER})
with run_main(["--probe", "--asset", "crypto", "--symbol", "BTCUSDT"]) as (rc, so, se):
    check("main: probe exits 0", rc == 0, f"rc={rc} err={se.strip()}")
    check("main: probe prints the feed state", "feed state" in so, so)
    check("main: probe prints the ABSTAIN verdict", "ABSTAIN" in so, so)

print()
print(f"RESULT: {PASSED} passed, {FAILED} failed")
if FAILURES:
    for f in FAILURES:
        print("   -", f)
sys.exit(1 if FAILED else 0)
