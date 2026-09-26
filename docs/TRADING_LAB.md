# OmniSeed — Trading Lab

_The evidence base for the multi-asset perception layer, the risk engine, and
the autonomy ladder. Everything here is **measured or enforced in code**, not
promised in prose._

> **Honest scope.** OmniSeed does not predict the market and does not claim to
> make money. Its only claim is that it **refuses to act on data it cannot
> trust**, sizes every position to a fixed risk budget, and cannot trade real
> money until a human writes an explicit unlock line. A positive out-of-sample
> result is *evidence*, never a guarantee. Edge decays; a 0.1B local model plus
> lexicon sentiment is **not** an institutional research desk.

---

## 1. Data sources (M1) — what we actually read

All providers are **keyless** and reached over plain HTTPS from the Python
tooling (`tools/market_feeds.py`). The C++ runtime never opens a socket for
market data — it consumes CSVs written by the tooling, keeping the zero-
dependency core intact.

| Asset class | Provider    | Endpoint (host)                  | Gives                    | Bars | Ticker |
|-------------|-------------|----------------------------------|--------------------------|:----:|:------:|
| equity      | Yahoo       | `query1.finance.yahoo.com`       | chart (1d/1h/1wk) + quote|  ✓   |   ✓    |
| crypto      | Binance     | `api.binance.com`                | klines + 24h ticker      |  ✓   |   ✓    |
| crypto      | CoinGecko   | `api.coingecko.com`              | market_chart (fallback)  |  ✓   |   —    |
| fx          | Frankfurter | `api.frankfurter.app`            | ECB daily reference      |  ✓   |   ✓    |
| meme        | DexScreener | `api.dexscreener.com`            | pair snapshot (TICKER)   |  ✗   |   ✓    |

**Provider order** is first-success-wins (`ASSET_PROVIDERS`): crypto tries
Binance then CoinGecko; everything else has a single source.

**Meme is deliberately bars-less.** DexScreener's free API publishes *pairs*,
not candles, so there is no honest OHLCV to build. Meme is therefore
**ticker/research-only** — the tool refuses `--asset meme` in bars mode (exit 2)
and the risk engine mirrors that with `RiskLimits::meme_research_only`.

### Provenance

Every bar CSV carries the T6 contract, emitted by the **single** definition
`tools/fetch_market_data.py::emit_csv`:

```
time,open,high,low,close,volume,source
```

`source` is the provider name (e.g. `binance`). `market_feeds.py` imports that
emitter via `importlib` rather than re-implementing it, so the schema cannot
drift between the two tools. A provider failure is **explicit**: non-zero exit,
**no file written**, nothing silently substituted.

### Rate limits & cache

* **Token-bucket limiter** — one shared bucket, refilled at `--rate-per-min`
  (default conservative per provider).
* **Content-addressed TTL cache** — responses are keyed by a sha1 of
  (provider, url, params) and stored as JSON under `--cache-dir` with a
  `--cache-ttl`. Re-runs inside the TTL never hit the network.
* **Retries** — transient transport errors retry with backoff; a hard provider
  error raises `ProviderError` naming the host.

---

## 2. Feed health & the ABSTAIN gate (M1)

Perception is where bad data becomes bad trades, so the module is built around
**refusing to act**. (`include/omniseed/trading/market_perception.h`.)

`FeedHealthMonitor` derives a per-symbol state from every observation/failure:

| State      | Trigger                                                    |
|------------|------------------------------------------------------------|
| `Ok`       | fresh update, no failure streak                            |
| `Degraded` | no fresh update for ≥ `degraded_after_secs` (default 300 s)|
| `Dead`     | no fresh update for ≥ `dead_after_secs` (900 s) **or**     |
|            | ≥ `dead_after_failures` consecutive provider errors (3)    |
| `Missing`  | never observed — **never** an implicit `Ok`                |

`PerceptionGate::check()` returns an `AbstainDecision` with a reason
(`missing-data`, `invalid-ticker`, `stale-feed`, `dead-feed`, `low-liquidity`).
A default-constructed `FeedHealth` (no symbol) means "the caller keeps no
monitor", so the ticker's own timestamp is the fallback; a health that *names*
the symbol but carries no data is `missing-data`.

**No look-ahead, no network, no threads** — the whole layer is deterministic
and unit-tested (`tests/test_market_perception.cpp`).

---

## 3. Risk engine (M4) — enforced, not aspirational

All limits live in `RiskLimits` / `RiskManager` and are applied to **every**
proposed order. The engine can refuse; nothing bypasses it.

| Control                        | Value / rule                                          |
|--------------------------------|-------------------------------------------------------|
| Risk per trade                 | clamped to **1–2 %** of equity (`kMin/kMaxRiskPerTradePct`) |
| Daily drawdown kill-switch     | default **3 %** from the day's high-water mark        |
| Weekly drawdown kill-switch    | default **6 %** from the week's high-water mark       |
| Meme per-trade risk            | **≤ 0.25 %**, or **0** while `meme_research_only`     |
| Pairwise correlation cap       | **0.80** — a redundant holding is refused             |
| Stale/dead feed                | **ABSTAIN** (see §2)                                   |

The daily and weekly windows are two independent `DailyRiskGovernor`s composed
into a `RiskGovernorSet`: entries are refused while **either** is halted, and
each re-arms on its own boundary. A bad day cannot trip the weekly window; a
bad week can.

**Sizing is asset-class aware.** `size_by_risk(equity, stop, AssetClass, …)`
applies the 1–2 % band to equities/crypto/fx, but for memes it uses
`min(risk_per_trade, meme_max_risk_pct)` **without** the 1 % floor — so a meme
position is never accidentally clamped *up* into a meaningful bet.

---

## 4. Autonomy ladder (M6) — code-enforced, human-unlocked

Live money is gated in **code**, and the only key is a line the owner writes by
hand in `PROJECT_STATE.md`. (`include/omniseed/trading/risk_gate.h`.)

| Level | Name            | Unlock line (exact, own line)  | Caps                                                      |
|:-----:|-----------------|--------------------------------|-----------------------------------------------------------|
| L0    | paper-only      | _(none — the default)_         | live orders refused outright                              |
| L1    | micro-live      | `UNLOCK L1 MICRO-LIVE`         | total ≤ **$100**, ≤ **1 %/trade**, ≥ **30** paper days    |
| L2    | scaled          | `UNLOCK L2 SCALE-UP`           | ≥ **90** live days within the drawdown caps               |

* `AutonomyGate::resolve()` scans the state text **line-by-line**, exact match
  after trimming whitespace. A near-miss — a `TODO:`, a quoted example,
  different case — unlocks **nothing**. The owner has to mean it.
* `AutonomyGate::allows_live()` re-checks the level, the duration minimum, and
  the per-order caps on **every** order, and always fills `why_not`.
* The kill-switch is always available, at every level, regardless of unlock.

---

## 5. The 24/7 paper loop (M5)

A long-running, multi-asset **paper** desk. Three pieces, split along the
"network lives in Python, the engine never opens a socket" line:

### `tools/paper_loop.py` — the poller

One cycle: **probe** every watch item (`ASSET:SYMBOL`) for a live ticker →
derive the feed state → **refresh** that symbol's provenance CSV (dedup by
timestamp) *only while the feed is OK* → invoke the C++ engine → write
`state/paper_status.json` (heartbeat + feed health + last session).

* A feed that is not OK is **ABSTAINed**: no new data, no new risk. Its open
  positions are still marked and can still exit.
* Health is tracked across cycles (consecutive failures, last success), exactly
  mirroring `FeedHealthMonitor` — including the precedence that a feed which has
  **never** succeeded reports `missing` even after repeated failures.
* Rate limiting + TTL caching are inherited from `market_feeds.HttpClient`, so
  a fast poll interval does not hammer the providers.

```bash
python tools/paper_loop.py --once                       # one cycle (cron)
python tools/paper_loop.py --interval 300               # poll forever
python tools/paper_loop.py --watch equity:AAPL,crypto:BTCUSDT,fx:EURUSD
```

### `omniseed_agent2 trading-paper-session` — the engine

One cash account, many streams. Bars from every stream are merged into a single
chronological timeline; at each timestamp the whole book is marked, the
composite daily+weekly kill-switch is consulted **once**, and then each stream
with a bar at that timestamp may exit or enter. Entries are sized per asset
class (meme stays research-only) and use the stream's *next* bar open (no
look-ahead).

* **Resume is real.** The append-only journal is the source of truth: a restart
  rebuilds cash and open positions from it (`PaperJournal::scan_existing` →
  `PaperBroker::seed`), so equity does not jump back to the starting cash. A
  re-run appends only genuinely new timestamps.
* Every bar mark, **every non-Hold signal**, every fill, kill-switch trip,
  ABSTAIN and risk refusal is journalled with the same schema as the
  single-asset daemon
  (`ts,kind,ticker,qty,price,pnl,equity,cash,exposure,reason`,
  `kind ∈ {start, equity, signal, fill, halt, event}`). The `signal` rows are
  the decision log — a non-Hold signal is recorded with its strength and rule
  trace whether or not it becomes a fill (`--no-signal-journal` opts out).

### `tools/nightly_report.py` + `tools/paper_dashboard.py` — the outputs

* **Nightly report** — Markdown summary (equity, today's P&L, open positions,
  today's fills, per-symbol contribution, kill-switch trips, feed health).
  **Email is opt-in and never faked**: it sends only when every
  `OMNISEED_SMTP_*` variable is present, otherwise it says so plainly and still
  writes the file.
* **Single-file dashboard** — one self-contained HTML document (inline CSS,
  inline SVG equity curve, plain tables; **no CDN, no external assets, no
  script**) showing the equity curve, open positions, today's P&L, feed health,
  recent fills and risk counters. Both tools share `tools/paper_report.py`, so
  the two can never disagree.

```bash
python tools/nightly_report.py --out state/nightly_report.md \
    --html state/nightly_report.html
python tools/paper_dashboard.py --out state/dashboard.html
```

### What a live run looks like on a Saturday

Running the loop against real providers on a weekend is instructive: the
daily equity (Yahoo) and FX (ECB reference) feeds are **stale**, so they
ABSTAIN — no new risk — while 24/7 crypto stays `ok`. And a BTC position that
would cost more than the 20 % per-name cap floors to zero shares, so the engine
**refuses** it rather than over-sizing. Both are the system working as intended,
not failures.

---

## 6. What is *not* here yet

Tracked honestly so it is not mistaken for done:

* Order-book imbalance, funding-rate scans, on-chain flows, event-calendar and
  social-volume signals (M7) — planned, not built.
* News/sentiment rationale per signal and a regime detector (M2) — planned.
* The strategy factory with automatic promotion/retirement (M3) — planned.

---

## 7. Test evidence

| Suite                        | What it pins down                                             |
|------------------------------|--------------------------------------------------------------|
| `omniseed_market_perception` | feed health, ABSTAIN reasons, governor set, asset-class risk, correlation cap, autonomy ladder |
| `omniseed_market_feeds`      | adapters, cache, limiter, normalisation, provenance reuse, meme refusal, explicit failure |
| `omniseed_market_data`       | T6 fetcher integrity (provenance, exit-2, no silent synth)    |
| `omniseed_trading_edge`      | walk-forward no-look-ahead, risk clamp, daily kill-switch, journal append/resume |
| `omniseed_paper_session`     | multi-asset merged timeline, per-asset gates, ABSTAIN, composite kill-switch, journal-based resume, determinism |
| `omniseed_paper_loop`        | watch parsing, CSV merge/dedup, feed health → ABSTAIN, engine invocation, status heartbeat |
| `omniseed_paper_reports`     | journal parsing, book replay, day P&L, Markdown/HTML rendering, opt-in SMTP config |

Run the board with `ctest --test-dir build -C Release`.

