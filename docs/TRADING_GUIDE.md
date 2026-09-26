# OmniSeed — Trading Guide

> **Read this first.** OmniSeed's trading stack minimizes losses through
> discipline — it does **not** eliminate them, and it does **not** guarantee
> profit. Every command in this guide prints that disclaimer. Backtests prove
> the past; they say nothing certain about the future.

## 1. What exists (honest map)

| Capability | State | Where |
|---|---|---|
| Technical indicators (SMA, EMA, RSI-Wilder, MACD, Bollinger, ATR) | real | `src/trading/trading_engine.cpp` |
| Rule-based signals + weighted multi-agent consensus | real | `src/agent/sub_agents.cpp` |
| Risk math: fractional Kelly, stops, drawdown halt, exposure caps | real | `src/trading/trading_engine.cpp` |
| Per-trade risk budget (1–2%) + daily drawdown kill-switch | real | `src/trading/trading_engine.cpp` |
| Backtester (next-bar-open fills, slippage + fees, no look-ahead) | real | `src/trading/trading_engine.cpp` |
| Walk-forward validation (select on train, replay out-of-sample) | real | `src/trading/walkforward.cpp` |
| Paper broker (slippage + fee fills, live marking) | real | `src/trading/simulate.cpp` |
| Paper daemon + append-only, resumable journal | real | `src/trading/paper_daemon.cpp` |
| News RSS ingest + lexicon sentiment + breaking alerts | real | `src/trading/news_feed.cpp` |
| Market data fetcher (Yahoo/Stooq/synthetic) | real | `tools/fetch_market_data.py` |
| News fetcher (RSS + offline sample) | real | `tools/fetch_news.py` |
| Self-model, goals, decision rationale, calibration | real | `src/runtime/introspection.cpp` |
| Cloud reasoning bridge (optional key, no TLS — see §6) | optional | `src/runtime/cloud_bridge.cpp` |
| Alpaca broker (paper endpoint default) | optional | `src/trading/broker_alpaca.cpp` |
| Live market data websocket feed | **absent** | — |
| Full vision backbone / FocalCodec codebooks | **pending** | PROJECT_STATE §9 |

## 2. Setup

### Market data (required)
```bash
# Real data (Stooq first, then Yahoo chart API — no key). Verify the row count.
.venv/Scripts/python.exe tools/fetch_market_data.py --ticker AAPL --timeframe 1d --years 5

# Offline demo (deterministic synthetic GBM — clearly marked SYNTH):
.venv/Scripts/python.exe tools/fetch_market_data.py --demo-synth
```
CSVs land in `models/market/<TICKER>_<TF>.csv`. **Always check bar timestamps
before acting on downloaded data — staleness is invisible to the signal math.**

#### Integrity rules (why a fetch can now *fail* instead of lying)

| Behaviour | Rule |
|---|---|
| Provenance | Every CSV carries a trailing `source` column (`stooq`/`yahoo`/`synth`). `load_bars_csv()` reads fields 0..5 and ignores it, so the C++ contract is unchanged. |
| No silent substitution | A provider failure is **never** replaced with synthetic bars. `--source stooq` (or `yahoo`) that fails prints the provider error and exits **2**, writing **no file**. |
| Explicit opt-in | `auto` (default) tries Stooq then Yahoo; if both fail it exits **2** unless `--allow-synth` is given, which then falls back to clearly-marked synthetic data. |
| No clobbering | Synthetic output defaults to `models/market/DEMO_<TF>.csv`, so it can never overwrite a real ticker's file. |
| `--years` is honoured everywhere | The window is applied to **every** source. Stooq returns its full history, so without the trim a `--years 5` request used to silently deliver decades of bars. |
| Empty result is a failure | `--years <= 0`, an unparsable/"No data" provider body, or 0 bars after trimming all exit **2**. |

```bash
# Ask for synthetic explicitly (never touches the network):
.venv/Scripts/python.exe tools/fetch_market_data.py --source synth --years 1

# Permit the auto fallback to synthetic when every real provider is down:
.venv/Scripts/python.exe tools/fetch_market_data.py --ticker AAPL --allow-synth
```

Exit codes: **0** = bars written, **2** = refused/failed (nothing written).
Characterization tests for all of the above: `tests/test_market_data.py`
(offline, stdlib-only).

### News (optional)
```bash
.venv/Scripts/python.exe tools/fetch_news.py --out models/news/headlines.xml
```

### Broker keys (optional, paper recommended)
```bash
export ALPACA_API_KEY_ID=...          # paper keys from Alpaca's dashboard
export ALPACA_API_SECRET_KEY=...
```
Keys live in the environment only — the runtime never persists them.

## 3. Usage

```bash
# One-shot analysis + 5-year backtest + Kelly sizing for a fresh entry
build/bin/omniseed_agent2.exe trading-analyze --ticker AAPL --timeframe 1d

# The multi-agent polling loop (analyst + news + risk + paper execution)
build/bin/omniseed_agent2.exe trading-watch --tickers AAPL,MSFT,TSLA --interval 60

# Paper simulation of a strategy over the full history
build/bin/omniseed_agent2.exe trading-simulate --ticker AAPL --strategy momentum

# Out-of-sample validation (select on train, replay untouched on test)
build/bin/omniseed_agent2.exe trading-walkforward --ticker AAPL --timeframe 1d

# Paper daemon: replay bars and append every action to a journal
build/bin/omniseed_agent2.exe trading-paper --ticker AAPL --timeframe 1d
```
Strategies: `balanced` (default), `momentum` (tighter stops), `mean-reversion`
(wider stops + take-profit).

### Self-awareness commands
```bash
build/bin/omniseed_agent2.exe introspect      # who am I / what am I doing / why
build/bin/omniseed_agent2.exe metacognition   # calibration + knowledge gaps
```

## 4. Risk management (the actual point)

* **Position sizing** — quarter-Kelly from realized win statistics. Before
  enough trades exist, sizing falls back to a fixed **2% probe position**.
  A negative Kelly edge refuses to trade at all.
* **Per-trade risk budget** (`RiskLimits::risk_per_trade_pct`) — the edge lab
  sizes each entry so that a stop-out loses a fixed fraction of equity:
  `qty = equity × risk / (price × stop_loss_pct)`. The value is **clamped
  into [1%, 2%]** — the engine will never risk more than 2% of equity on one
  trade, whatever is configured, and the sizing trace says `(clamped)` when
  it fires. The notional is then capped by the 20% concentration limit.
* **Daily kill-switch** (`RiskLimits::max_daily_loss_pct`, default 3%) —
  once the drawdown from a UTC day's **high-water mark** reaches the limit,
  new entries are refused for the rest of that day. The switch re-arms at
  the next UTC day boundary. Exits and stops are **never** blocked — the
  switch only gates *new* risk. (On a once-per-day bar series every bar is
  its own day, so the switch only has room to act on intraday bars.)
* **Stop-loss** — 8% per position by default (`RiskLimits::stop_loss_pct`).
  Sell signals also exit positions (reason `signal` in trade logs).
* **Drawdown halt** — new entries stop at a 20% portfolio drawdown from peak.
* **Concentration cap** — 20% of equity per name; the risk agent alerts
  (and can veto) above 30%.
* **Costs are real** — 5 bps fees + 2 bps slippage are modeled on every fill.

Rule of thumb: if a strategy's backtest win rate is below ~50% with profit
factor < 1.2, it is not ready for paper trading, let alone anything else.

## 5. Backtesting — what the numbers mean

* **Sharpe / Sortino** — annualized from per-bar returns; a constant series
  (zero variance) reports 0, not infinity.
* **Max drawdown** — peak-to-trough on the equity curve.
* **No look-ahead** — signals at bar *i* execute at bar *i+1*'s open;
  truncating the future never changes past equity (tested).
* **Win rate** — share of closed trades with positive P&L; profit factor is
  gross profit over gross loss.

## 5b. Edge lab — walk-forward + paper daemon

A single backtest over the whole history is an *in-sample* number: tune long
enough and it will always look good. The edge lab exists to separate the
strategy from the tuning.

### Walk-forward (`trading-walkforward`)

The series is cut into consecutive **train / test** folds. Parameters are
chosen on the train window only, then replayed **untouched** on the following
test window. The stitched test windows form a genuine out-of-sample (OOS)
curve.

```bash
build/bin/omniseed_agent2.exe trading-walkforward --ticker AAPL --timeframe 1d \
    --train 252 --test 63 --step 63        # [--anchored]
```

* **Hard no-look-ahead** — `test_begin == train_end` for every fold, and
  selection never sees a test bar. (Enforced by test.)
* **`degradation_pct`** = mean in-sample return − out-of-sample return. A
  large positive gap is the classic signature of overfitting; the CLI flags
  it above +5%.
* **`oos_hit_rate`** — fraction of folds that were profitable out of sample.
* `--anchored` grows the train window from bar 0 instead of rolling it.
* Honest scope: a positive OOS result is **evidence, not a promise**. Edges
  decay.

### Paper daemon (`trading-paper`)

A deterministic bar-replay loop that drives signals → risk engine →
`PaperBroker`, appending everything to an **append-only journal**.

```bash
build/bin/omniseed_agent2.exe trading-paper --ticker AAPL --timeframe 1d \
    --journal state/paper_journal.csv
```

Journal schema (CSV, one record per line):

```
ts,kind,ticker,qty,price,pnl,equity,cash,exposure,reason
kind ∈ {start, equity, signal, fill, halt, event}
```

* **`signal` rows are the decision log** — every non-Hold signal is recorded
  with its strength (in the `qty` column) and rule trace, whether or not it
  becomes a fill. Disable with `--no-signal-journal`.
* `event` rows carry kill-switch trips (`daily:`/`weekly:`), feed ABSTAINs
  (`abstain:<sym>:<reason>`) and risk-gate refusals (`refused:<sym>:<reason>`).

* **Crash-safe** — every record is flushed as it is written; the file is only
  ever appended.
* **Resumable** — a re-run reads the journal's last timestamp and skips bars
  already recorded, so you can point it at a growing CSV and it picks up
  where it left off. It never rewrites or duplicates history.
* **Kill-switch events are journaled** as `halt` records with the reason.
* No threads, no network, no real orders — paper money only.

## 5c. Multi-asset perception, risk gates & the autonomy ladder

The full evidence table lives in **[`TRADING_LAB.md`](TRADING_LAB.md)**; this is
the operational summary.

### Market feeds (`tools/market_feeds.py`)

Keyless, multi-asset bars/tickers with one HTTP path (cache + rate limit +
retry):

```bash
# Crypto daily bars -> provenance CSV (reuses the T6 schema):
python tools/market_feeds.py --asset crypto --symbol BTCUSDT --years 2

# FX:
python tools/market_feeds.py --asset fx --symbol EURUSD --years 1 \
    --out models/market/EURUSD_1d.csv

# Health probe — prints the feed state and the ABSTAIN verdict:
python tools/market_feeds.py --probe --asset equity --symbol AAPL
```

| Asset  | Provider(s)            | Bars | Ticker |
|--------|------------------------|:----:|:------:|
| equity | Yahoo                  |  ✓   |   ✓    |
| crypto | Binance → CoinGecko    |  ✓   |   ✓    |
| fx     | Frankfurter            |  ✓   |   ✓    |
| meme   | DexScreener            |  ✗   |   ✓    |

Meme is **research-only** (no free OHLCV exists) — bars mode is refused with
exit 2, mirroring `RiskLimits::meme_research_only`. A provider failure is
explicit: non-zero exit, no file, nothing silently substituted.

### The ABSTAIN gate (M1)

`FeedHealthMonitor` classifies every symbol `Ok → Degraded (300 s) → Dead
(900 s)` (or dead after 3 consecutive provider errors); an unobserved symbol is
`Missing`, **never** an implicit `Ok`. `PerceptionGate::check()` then returns an
`AbstainDecision` — stale, dead, missing or invalid input **refuses to produce
a trade**. Deterministic, no network, unit-tested.

### Risk gates (M4)

| Control                    | Value                                          |
|----------------------------|------------------------------------------------|
| Risk per trade             | clamped to **1–2 %** of equity                 |
| Daily kill-switch          | **3 %** drawdown from the day's high-water     |
| Weekly kill-switch         | **6 %** drawdown from the week's high-water    |
| Meme per-trade risk        | **≤ 0.25 %** (or 0 while research-only)        |
| Pairwise correlation cap   | **0.80** — a redundant holding is refused      |

Entries are refused while **either** drawdown window is halted; each re-arms on
its own boundary. Sizing is asset-class aware: the 1–2 % band applies to
equity/crypto/fx, but memes use `min(risk_per_trade, 0.25 %)` with **no 1 %
floor**, so a meme is never clamped *up*.

### Autonomy ladder (M6) — the only thing you must do by hand

Live money is gated in **code**. The unlock is a line you write, alone, in
`PROJECT_STATE.md`:

```
UNLOCK L1 MICRO-LIVE      # total <= $100, <= 1%/trade, >= 30 paper days
UNLOCK L2 SCALE-UP        # >= 90 live days within the drawdown caps
```

L0 (paper-only) is the default and refuses live orders outright. Matching is
line-exact after trimming — a `TODO:`, a quoted example, or different case
unlocks **nothing**. Every order re-checks the level, the duration minimum and
the caps, and logs *why* it refused.

## 5d. The 24/7 paper loop (M5)

Three pieces, split so the network lives in Python and the engine never opens a
socket. Details and the evidence table are in
**[`TRADING_LAB.md`](TRADING_LAB.md)** §5.

### The poller — `tools/paper_loop.py`

Probes every watch item, tracks feed health, refreshes each symbol's provenance
CSV **only while the feed is OK**, runs the engine, and writes
`state/paper_status.json` (heartbeat + feed health + last session).

```bash
python tools/paper_loop.py --once                 # one cycle (cron-friendly)
python tools/paper_loop.py --interval 300         # poll every 5 min, forever
python tools/paper_loop.py --watch equity:AAPL,crypto:BTCUSDT,fx:EURUSD
python tools/paper_loop.py --dry-run --once       # poll only, skip the engine
```

A feed that is not OK is **ABSTAINed** — no new data, no new risk (open
positions are still marked and can still exit).

### The engine — `trading-paper-session`

One cash account, many streams merged onto a single timeline, one composite
daily+weekly kill-switch, per-asset-class sizing, and a **real resume**: the
append-only journal is the source of truth, so a restart rebuilds cash and
positions instead of resetting to the starting cash.

```bash
build/bin/omniseed_agent2.exe trading-paper-session \
    --streams "AAPL,equity,models/market/AAPL_1d.csv;\
BTCUSDT,crypto,models/market/BTCUSDT_1d.csv" \
    --skip BTCUSDT --journal state/paper_journal.csv
```

`--skip SYM,...` marks a symbol's feed as not-OK (the poller passes this when a
feed is stale/dead). Journal schema is unchanged from §5b.

### The outputs

```bash
python tools/nightly_report.py --out state/nightly_report.md \
    --html state/nightly_report.html      # email only if OMNISEED_SMTP_* set
python tools/paper_dashboard.py --out state/dashboard.html
```

* The **nightly report** is Markdown: equity, today's P&L, open positions,
  today's fills, per-symbol contribution, kill-switch trips, feed health.
  Email is **opt-in and never faked** — without `OMNISEED_SMTP_HOST/USER/PASS`
  and `OMNISEED_REPORT_TO` it says so and still writes the file.
* The **dashboard** is one self-contained HTML file — inline CSS, an inline SVG
  equity curve, and plain tables. No CDN, no external assets, no JavaScript.

## 6. Cloud reasoning (optional) — the TLS story, plainly

The zero-dependency runtime has **no TLS stack**. Consequences:

* `OMNISEED_CLOUD_KEY` enables OpenAI-compatible chat APIs **only through a
  local HTTPS relay**: set `OMNISEED_CLOUD_PROXY=http://127.0.0.1:8080` and
  run a TLS-terminating proxy (e.g. `mitmproxy --mode reverse` or any
  localhost forwarder). Plain `http://` endpoints work directly.
* Without a relay, https providers are **refused loudly** — the runtime never
  silently downgrades or fakes a cloud answer. Local AgentLoop remains fully
  functional (that is the fallback by design).

## 7. Alpaca adapter

* **Default endpoint is PAPER** (`paper-api.alpaca.markets`) — real order
  lifecycle, fake money. Get paper keys from Alpaca's dashboard.
* **LIVE trading is a two-key gauntlet**: `--live-trading` **and** typing
  exactly `I ACCEPT REAL LOSS` at the prompt. There is no env/flag shortcut.
* All orders validate locally (symbol format, positive qty, side) **before**
  any network call; garbage never reaches the broker.
* HTTPS to Alpaca requires the same local relay as §6
  (`OMNISEED_CLOUD_PROXY`).

Safety checklist before enabling live mode:
1. 100+ paper trades with positive expectancy on the target strategy.
2. Max drawdown on paper within your real tolerance.
3. Position sizes you can afford to lose entirely.
4. You understand that a 0.1B-parameter local model + lexicon sentiment is
   **not** an institutional research desk.

## 8. Testing

`omniseed_trading` (ctest) covers: indicator math, signal determinism +
no-look-ahead, CSV round-trips, position accounting, Kelly/stops/drawdown
math, metrics, backtest invariants, RSS/sentiment/entity parsing, agent
consensus, broker accounting, introspection persistence + calibration,
finance (NPV/IRR/put-call parity), sandbox refusals, mock Alpaca lifecycle.

The T7+/M-series suites (also ctest):

* `omniseed_trading_edge` — walk-forward no-look-ahead + OOS determinism,
  risk-budget sizing clamp, daily kill-switch, paper journal append/resume.
* `omniseed_market_perception` — feed-health staleness/failures, the ABSTAIN
  gate, daily+weekly governor set, asset-class risk budgets, correlation cap,
  autonomy ladder.
* `omniseed_market_feeds` — feed adapters, cache, rate limiter, symbol
  normalisation, provenance reuse, meme research-only refusal, explicit
  failure. (Python, venv-gated.)
* `omniseed_market_data` — T6 fetcher integrity: provenance column, exit-2 on
  provider failure, `--years` trim, no implicit synthetic substitution.
  (Python, venv-gated.)
* `omniseed_paper_session` — multi-asset merged timeline, per-asset gates,
  feed ABSTAIN, composite kill-switch, journal-based resume, determinism.
* `omniseed_paper_loop` — watch parsing, provenance-CSV merge/dedup, feed
  health → ABSTAIN, engine invocation, status heartbeat. (Python, venv-gated.)
* `omniseed_paper_reports` — journal parsing, book replay, day P&L,
  Markdown/HTML rendering, opt-in SMTP config. (Python, venv-gated.)
