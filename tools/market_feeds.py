#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — tools/market_feeds.py
#  M1 Market Perception: multi-asset adapters + cache + rate limits + health.
#
#  Providers (all keyless):
#    equity  yahoo        query1.finance.yahoo.com  (chart + quote)
#    crypto  binance      api.binance.com           (klines + 24h ticker)
#    crypto  coingecko    api.coingecko.com         (market_chart fallback)
#    fx      frankfurter  api.frankfurter.app       (ECB daily reference)
#    meme    dexscreener  api.dexscreener.com       (TICKER ONLY — research)
#
#  Design rules
#    * One HTTP path (HttpClient) so every request is cached, rate-limited and
#      retried in exactly one place.
#    * CSV output reuses tools/fetch_market_data.py's emit_csv(), so the T6
#      provenance contract (time,open,high,low,close,volume,source) has a
#      single definition.
#    * A provider failure is EXPLICIT: non-zero exit, no file written. Nothing
#      is silently substituted.
#    * Meme assets have no OHLCV here on purpose: DexScreener's free API
#      publishes pairs, not candles. Meme is therefore ticker/research-only,
#      matching RiskLimits::meme_research_only.
#
#  Usage:
#    python tools/market_feeds.py --asset crypto --symbol BTCUSDT --years 2
#    python tools/market_feeds.py --asset fx --symbol EURUSD --years 1 \
#        --out models/market/EURUSD_1d.csv
#    python tools/market_feeds.py --probe --asset equity --symbol AAPL
# =============================================================================
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# Reuse the T6 emitter + error type so the schema has ONE definition.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fmd = _load_module("fetch_market_data", os.path.join(_HERE, "fetch_market_data.py"))

ProviderError = fmd.ProviderError

ASSETS = ("equity", "crypto", "fx", "meme")

# Provider order per asset class (first success wins).
ASSET_PROVIDERS = {
    "equity": ("yahoo",),
    "crypto": ("binance", "coingecko"),
    "fx": ("frankfurter",),
    "meme": ("dexscreener",),
}

# CoinGecko needs its own slug, not the exchange symbol.
COINGECKO_IDS = {
    "BTCUSDT": "bitcoin", "BTCUSD": "bitcoin",
    "ETHUSDT": "ethereum", "ETHUSD": "ethereum",
    "SOLUSDT": "solana", "SOLUSD": "solana",
    "DOGEUSDT": "dogecoin", "DOGEUSD": "dogecoin",
}

UA = {"User-Agent": "Mozilla/5.0 (omniseed-market-feeds/1.0)"}
CACHE_DIR = os.path.join("models", "cache", "feeds")


def _now() -> float:
    return time.time()


# ---------------------------------------------------------------------------
# Rate limiter (token bucket by key)
# ---------------------------------------------------------------------------
class RateLimiter:
    """Enforces a minimum interval between calls that share a key (a host)."""

    def __init__(self, per_minute: float = 20.0, sleep=None):
        self.min_interval = 60.0 / max(1.0, float(per_minute))
        self._last = {}
        # Resolved lazily so tests can inject a recorder (and so patching
        # time.sleep after import still takes effect).
        self._sleep = sleep

    def _sleep_for(self, secs: float) -> None:
        (self._sleep or time.sleep)(secs)

    def wait(self, key: str) -> float:
        """Blocks if needed; returns the seconds actually slept."""
        now = time.monotonic()
        last = self._last.get(key)
        slept = 0.0
        if last is not None:
            delta = now - last
            if delta < self.min_interval:
                slept = self.min_interval - delta
                self._sleep_for(slept)
        self._last[key] = time.monotonic()
        return slept


# ---------------------------------------------------------------------------
# On-disk TTL cache
# ---------------------------------------------------------------------------
class Cache:
    """Content-addressed JSON cache. Key -> sha1 -> <root>/<hash>.json."""

    def __init__(self, root: str = CACHE_DIR, ttl_secs: float = 900.0):
        self.root = root
        self.ttl = float(ttl_secs)
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> str:
        h = hashlib.sha1(key.encode("utf-8")).hexdigest()
        return os.path.join(self.root, h + ".json")

    def get(self, key: str, now=None):
        now = _now() if now is None else now
        path = self._path(key)
        try:
            with open(path, encoding="utf-8") as f:
                blob = json.load(f)
        except (OSError, ValueError):
            self.misses += 1
            return None
        if now - float(blob.get("stored_at", 0.0)) > self.ttl:
            self.misses += 1
            return None
        self.hits += 1
        return blob.get("text")

    def put(self, key: str, text: str, now=None) -> None:
        now = _now() if now is None else now
        path = self._path(key)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"stored_at": now, "key": key, "text": text}, f)
        os.replace(tmp, path)


# ---------------------------------------------------------------------------
# HttpClient — the single place every request goes through
# ---------------------------------------------------------------------------
class HttpClient:
    def __init__(self, cache=None, limiter=None, user_agent=None, retries=2):
        self.cache = cache
        self.limiter = limiter or RateLimiter()
        self.user_agent = user_agent or UA
        self.retries = int(retries)
        self.requests = 0
        self.cache_hits = 0

    @staticmethod
    def _host(url: str) -> str:
        try:
            return url.split("//", 1)[1].split("/", 1)[0]
        except IndexError:
            return url

    def get_text(self, url: str, key=None, use_cache=True) -> str:
        key = key or url
        if use_cache and self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                self.cache_hits += 1
                return hit

        last_err = None
        for attempt in range(self.retries + 1):
            self.limiter.wait(self._host(url))
            req = urllib.request.Request(url, headers=self.user_agent)
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    text = r.read().decode("utf-8", "replace")
                self.requests += 1
                if self.cache is not None:
                    self.cache.put(key, text)
                return text
            except Exception as e:                      # noqa: BLE001
                last_err = e
                self.requests += 1
                if attempt < self.retries:
                    time.sleep(1.5 * (attempt + 1))
        raise ProviderError(f"{self._host(url)}: {last_err}")

    def get_json(self, url: str, key=None, use_cache=True) -> dict:
        text = self.get_text(url, key=key, use_cache=use_cache)
        try:
            return json.loads(text)
        except ValueError as e:
            raise ProviderError(f"{self._host(url)}: bad JSON ({e})") from e


# ---------------------------------------------------------------------------
# Symbol normalisation
# ---------------------------------------------------------------------------
def normalize(asset: str, symbol: str):
    """-> (canonical_symbol, extra) where extra is provider-specific."""
    s = symbol.strip()
    if asset == "equity":
        return s.upper(), None
    if asset == "crypto":
        up = s.upper()
        # A quote suffix only counts when something precedes it, so a bare base
        # symbol ("BTC", "ETH") still gets the default USDT quote, while an
        # explicit pair ("ETHBTC") is left untouched.
        has_quote = any(len(up) > len(q) and up.endswith(q)
                        for q in ("USDT", "USD", "BTC", "ETH"))
        if not has_quote:
            up = up + "USDT"
        return up, None
    if asset == "fx":
        up = s.upper().replace("/", "").replace("-", "")
        if len(up) != 6:
            raise ProviderError(
                f"fx symbol must be a 6-letter pair (e.g. EURUSD), got {symbol!r}")
        return up, (up[:3], up[3:])
    if asset == "meme":
        return s, None
    raise ProviderError(f"unknown asset class {asset!r}")


# ---------------------------------------------------------------------------
# Bar adapters -> (t, o, h, l, c, v) tuples
# ---------------------------------------------------------------------------
def bars_yahoo(sym, tf, years, http):
    interval = {"1d": "1d", "1h": "1h", "1wk": "1wk"}.get(tf)
    if interval is None:
        raise ProviderError(f"yahoo supports 1d/1h/1wk, got {tf!r}")
    url = ("https://query1.finance.yahoo.com/v8/finance/chart/" + sym +
           f"?interval={interval}&range={max(1, int(math.ceil(years)))}y")
    data = http.get_json(url, key=f"yahoo:bars:{sym}:{tf}:{years}")
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


def bars_binance(sym, tf, years, http):
    interval = {"1d": "1d", "1h": "1h", "1wk": "1w", "1mo": "1M"}.get(tf)
    if interval is None:
        raise ProviderError(f"binance supports 1d/1h/1wk/1mo, got {tf!r}")
    per_year = {"1d": 365, "1h": 8760, "1wk": 52, "1mo": 12}[tf]
    limit = int(min(1000, max(10, math.ceil(years * per_year) + 5)))
    url = ("https://api.binance.com/api/v3/klines?symbol=" + sym +
           f"&interval={interval}&limit={limit}")
    data = http.get_json(url, key=f"binance:bars:{sym}:{tf}:{years}")
    if not isinstance(data, list) or not data:
        raise ProviderError(f"binance: no klines for {sym!r}")
    rows = []
    for k in data:
        try:
            rows.append((int(k[0]) // 1000, k[1], k[2], k[3], k[4], k[5]))
        except (IndexError, TypeError, ValueError):
            continue
    if not rows:
        raise ProviderError(f"binance: unparsable klines for {sym!r}")
    return rows


def bars_coingecko(sym, tf, years, http):
    coin = COINGECKO_IDS.get(sym)
    if coin is None:
        raise ProviderError(
            f"coingecko: no id mapping for {sym!r} (add it to COINGECKO_IDS)")
    days = int(min(365, max(1, math.ceil(years * 365))))
    url = (f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
           f"?vs_currency=usd&days={days}")
    data = http.get_json(url, key=f"coingecko:bars:{coin}:{days}")
    prices = data.get("prices") or []
    vols = data.get("total_volumes") or []
    if not prices:
        raise ProviderError(f"coingecko: no prices for {coin!r}")
    rows = []
    for i, p in enumerate(prices):
        t = int(p[0]) // 1000
        px = float(p[1])
        v = float(vols[i][1]) if i < len(vols) else 0.0
        # CoinGecko publishes spot points, not candles -> flat OHLC.
        rows.append((t, px, px, px, px, v))
    return rows


def bars_frankfurter(sym, tf, years, http):
    if tf not in ("1d", "1wk"):
        raise ProviderError(f"frankfurter supports 1d/1wk, got {tf!r}")
    base, quote = sym[:3], sym[3:]
    start = (dt.date.today() - dt.timedelta(days=int(years * 365) + 5)).isoformat()
    url = f"https://api.frankfurter.app/{start}..?from={base}&to={quote}"
    data = http.get_json(url, key=f"frankfurter:bars:{base}{quote}:{start}")
    rates = data.get("rates") or {}
    if not rates:
        raise ProviderError(f"frankfurter: no rates for {sym!r}")
    rows = []
    for day, table in sorted(rates.items()):
        px = table.get(quote)
        if px is None:
            continue
        t = int(dt.datetime.strptime(day, "%Y-%m-%d")
                .replace(tzinfo=dt.timezone.utc).timestamp())
        px = float(px)
        rows.append((t, px, px, px, px, 0))
    if not rows:
        raise ProviderError(f"frankfurter: unparsable rates for {sym!r}")
    return rows


BAR_ADAPTERS = {
    "yahoo": bars_yahoo,
    "binance": bars_binance,
    "coingecko": bars_coingecko,
    "frankfurter": bars_frankfurter,
}


# ---------------------------------------------------------------------------
# Ticker adapters -> dict (the perception snapshot)
# ---------------------------------------------------------------------------
def ticker_yahoo(sym, http):
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
           "?interval=1d&range=5d")
    data = http.get_json(url, key=f"yahoo:ticker:{sym}", use_cache=False)
    try:
        res = data["chart"]["result"][0]
        meta = res["meta"]
        last = float(meta.get("regularMarketPrice") or 0.0)
        prev = float(meta.get("chartPreviousClose") or 0.0)
        ts = int(meta.get("regularMarketTime") or 0)
        vol = float(meta.get("regularMarketVolume") or 0.0)
    except (KeyError, IndexError, TypeError) as e:
        raise ProviderError(f"yahoo: unexpected quote payload ({e})") from e
    if last <= 0.0:
        raise ProviderError(f"yahoo: no price for {sym!r}")
    return {"symbol": sym, "provider": "yahoo", "last": last,
            "change_pct_24h": ((last / prev - 1.0) * 100.0) if prev > 0 else 0.0,
            "volume_24h": vol, "ts": ts or int(_now())}


def ticker_binance(sym, http):
    url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={sym}"
    data = http.get_json(url, key=f"binance:ticker:{sym}", use_cache=False)
    try:
        last = float(data["lastPrice"])
        change = float(data["priceChangePercent"])
        vol = float(data["quoteVolume"])
        ts = int(data.get("closeTime") or 0) // 1000
    except (KeyError, TypeError, ValueError) as e:
        raise ProviderError(f"binance: unexpected ticker payload ({e})") from e
    if last <= 0.0:
        raise ProviderError(f"binance: no price for {sym!r}")
    return {"symbol": sym, "provider": "binance", "last": last,
            "change_pct_24h": change, "volume_24h": vol,
            "ts": ts or int(_now())}


def ticker_frankfurter(sym, http):
    base, quote = sym[:3], sym[3:]
    url = f"https://api.frankfurter.app/latest?from={base}&to={quote}"
    data = http.get_json(url, key=f"frankfurter:ticker:{base}{quote}",
                         use_cache=False)
    px = (data.get("rates") or {}).get(quote)
    if px is None:
        raise ProviderError(f"frankfurter: no rate for {sym!r}")
    day = data.get("date")
    ts = int(dt.datetime.strptime(day, "%Y-%m-%d")
             .replace(tzinfo=dt.timezone.utc).timestamp()) if day else int(_now())
    return {"symbol": sym, "provider": "frankfurter", "last": float(px),
            "change_pct_24h": 0.0, "volume_24h": 0.0, "ts": ts}


def ticker_dexscreener(address, http):
    url = f"https://api.dexscreener.com/latest/dex/tokens/{address}"
    data = http.get_json(url, key=f"dexscreener:ticker:{address}",
                         use_cache=False)
    pairs = data.get("pairs") or []
    if not pairs:
        raise ProviderError(f"dexscreener: no pairs for {address!r}")
    def liq(p):
        return float((p.get("liquidity") or {}).get("usd") or 0.0)
    best = max(pairs, key=liq)
    sym = (best.get("baseToken") or {}).get("symbol") or address
    last = float(best.get("priceUsd") or 0.0)
    if last <= 0.0:
        raise ProviderError(f"dexscreener: no price for {address!r}")
    return {"symbol": sym, "provider": "dexscreener", "last": last,
            "change_pct_24h": float((best.get("priceChange") or {}).get("h24") or 0.0),
            "volume_24h": float((best.get("volume") or {}).get("h24") or 0.0),
            "liquidity_usd": liq(best),
            "ts": int(_now())}


TICKER_ADAPTERS = {
    "yahoo": ticker_yahoo,
    "binance": ticker_binance,
    "frankfurter": ticker_frankfurter,
    "dexscreener": ticker_dexscreener,
}


# ---------------------------------------------------------------------------
# Feed-health policy (mirrors the C++ FeedPolicy in market_perception.h)
# ---------------------------------------------------------------------------
DEGRADED_AFTER_SECS = 300
DEAD_AFTER_SECS = 900


def feed_state(staleness_secs, failures) -> str:
    if staleness_secs is None or staleness_secs < 0:
        return "missing"
    if failures >= 3:
        return "dead"
    if staleness_secs >= DEAD_AFTER_SECS:
        return "dead"
    if staleness_secs >= DEGRADED_AFTER_SECS:
        return "degraded"
    return "ok"


def abstain_decision(state, volume_24h, min_volume) -> tuple:
    """Mirrors PerceptionGate::check -> (abstain, reason)."""
    if state == "missing":
        return True, "missing-data"
    if state == "dead":
        return True, "dead-feed"
    if state == "degraded":
        return True, "stale-feed"
    if min_volume > 0 and volume_24h < min_volume:
        return True, "low-liquidity"
    return False, "none"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def fetch_bars(asset, symbol, tf, years, http, provider=None):
    """Try each provider for the asset class. Raises ProviderError if all fail."""
    sym, _ = normalize(asset, symbol)
    order = (provider,) if provider else ASSET_PROVIDERS[asset]
    failures = []
    for name in order:
        adapter = BAR_ADAPTERS.get(name)
        if adapter is None:
            failures.append((name, f"no OHLCV adapter for {asset}"))
            continue
        try:
            rows = adapter(sym, tf, years, http)
            return rows, name, sym
        except Exception as e:                      # noqa: BLE001
            failures.append((name, str(e)))
    detail = "; ".join(f"{n}: {e}" for n, e in failures)
    raise ProviderError(f"every provider failed for {sym} ({detail})")


def probe_ticker(asset, symbol, http, provider=None):
    """-> (ticker_dict, state, abstain, reason)."""
    sym, _ = normalize(asset, symbol)
    order = (provider,) if provider else ASSET_PROVIDERS[asset]
    failures = []
    for name in order:
        adapter = TICKER_ADAPTERS.get(name)
        if adapter is None:
            failures.append((name, "no ticker adapter"))
            continue
        try:
            tk = adapter(sym, http)
            stale = max(0, int(_now()) - int(tk.get("ts") or 0))
            state = feed_state(stale, 0)
            abstain, reason = abstain_decision(
                state, float(tk.get("volume_24h") or 0.0), 0.0)
            return tk, state, abstain, reason
        except Exception as e:                      # noqa: BLE001
            failures.append((name, str(e)))
    detail = "; ".join(f"{n}: {e}" for n, e in failures)
    raise ProviderError(f"every provider failed for {sym} ({detail})")


def main() -> int:
    ap = argparse.ArgumentParser(description="OmniSeed multi-asset market feeds")
    ap.add_argument("--asset", default="crypto", choices=list(ASSETS))
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--timeframe", default="1d",
                    choices=["1d", "1h", "1wk", "1mo"])
    ap.add_argument("--years", type=float, default=2.0)
    ap.add_argument("--provider", default=None,
                    help="force a provider (yahoo|binance|coingecko|"
                         "frankfurter|dexscreener)")
    ap.add_argument("--probe", action="store_true",
                    help="fetch the ticker + print the feed-health/abstain verdict")
    ap.add_argument("--out", default="", help="CSV path (bars mode)")
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    ap.add_argument("--cache-ttl", type=float, default=900.0)
    ap.add_argument("--rate-per-min", type=float, default=20.0)
    a = ap.parse_args()

    if a.years <= 0:
        print("[feeds] --years must be > 0", file=sys.stderr)
        return 2

    cache = Cache(a.cache_dir, a.cache_ttl)
    http = HttpClient(cache=cache, limiter=RateLimiter(a.rate_per_min))

    # Meme has no OHLCV by design (DexScreener publishes pairs, not candles).
    if a.asset == "meme" and not a.probe:
        print("[feeds] meme assets are ticker/research-only: DexScreener's free "
              "API has no OHLCV. Use --probe, or another asset class.",
              file=sys.stderr)
        return 2

    if a.probe:
        try:
            tk, state, abstain, reason = probe_ticker(
                a.asset, a.symbol, http, a.provider)
        except ProviderError as e:
            print(f"[feeds] ERROR: {e}", file=sys.stderr)
            return 2
        print(f"[probe] {tk['symbol']} ({a.asset}) via {tk['provider']}")
        print(f"  last        : {tk['last']}")
        print(f"  chg 24h     : {tk.get('change_pct_24h', 0.0):+.2f}%")
        print(f"  vol 24h     : {tk.get('volume_24h', 0.0):.0f}")
        if "liquidity_usd" in tk:
            print(f"  liquidity   : {tk['liquidity_usd']:.0f}")
        print(f"  feed state  : {state}")
        print(f"  ABSTAIN     : {'YES' if abstain else 'no'} ({reason})")
        if a.asset == "meme":
            print("  note        : meme is research-only "
                  "(RiskLimits::meme_research_only)")
        return 0

    try:
        rows, source, sym = fetch_bars(
            a.asset, a.symbol, a.timeframe, a.years, http, a.provider)
    except ProviderError as e:
        print(f"[feeds] ERROR: {e}", file=sys.stderr)
        print("[feeds] refusing to substitute synthetic data.", file=sys.stderr)
        return 2

    text = fmd.emit_csv(rows, source, a.years)
    n = len(text.strip().splitlines()) - 1
    if n == 0:
        print(f"[feeds] ERROR: 0 bars after trimming to {a.years}y",
              file=sys.stderr)
        return 2

    out = a.out or os.path.join(
        "models", "market", f"{sym}_{a.timeframe}.csv")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        f.write(text)

    print(f"[feeds] {a.asset}/{sym} via {source} -> {out} "
          f"({n} bars, {a.timeframe}, {a.years}y)")
    print(f"[feeds] http={http.requests} cache_hits={http.cache_hits} "
          f"cache_misses={cache.misses}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
