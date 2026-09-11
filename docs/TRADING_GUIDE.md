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
| Backtester (next-bar-open fills, slippage + fees, no look-ahead) | real | `src/trading/trading_engine.cpp` |
| Paper broker (slippage + fee fills, live marking) | real | `src/trading/simulate.cpp` |
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
# Real data (Yahoo chart API, no key). Verify the row count before trusting.
.venv/Scripts/python.exe tools/fetch_market_data.py --ticker AAPL --timeframe 1d --years 5

# Offline demo (deterministic synthetic GBM — clearly marked SYNTH):
.venv/Scripts/python.exe tools/fetch_market_data.py --demo-synth
```
CSVs land in `models/market/<TICKER>_<TF>.csv`. **Always check bar timestamps
before acting on downloaded data — staleness is invisible to the signal math.**

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
