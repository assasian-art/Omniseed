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

## 6. The Monster layer (sniper entry + news hunter + cross-asset)

Full math and honest-scope notes: **`docs/MONSTER_DESIGN.md`**. This section is
the operational summary.

### Where it runs

Heavy logic is Python (`tools/monster/`); the C++ engine gains exactly one
seam. The engine never computes an indicator for the Monster — it reads a
**distilled** row per bar:

```
ts,confidence,veto,regime,detail
```

| module | role |
|---|---|
| `tools/monster/features.py` | no-look-ahead primitives (SMA/EMA/RSI/MACD/ATR, rolling VWAP, z-score, Pearson, Amihud illiquidity, Kyle-lambda proxy, swings, Fibonacci) |
| `tools/monster/regime.py` | multi-axis causal regime engine (variance ratio, Hurst, ER, ADX, choppiness, R², ρ(1), MA consistency; Yang-Zhang vol; OU half-life; hysteresis) |
| `tools/monster/strategies.py` | the zoo: momentum, mean-reversion, breakout, order-flow imbalance, vol-target |
| `tools/monster/router.py` | regime-adaptive continuous allocation → conviction + agreement |
| `tools/monster/funding.py` | perpetual funding-rate carry scanner (net-of-cost, risk-flagged) |
| `tools/monster/sniper_engine.py` | the weighted 4-layer confidence score + hard vetoes |
| `tools/monster/news_hunter.py` | keyword-weighted events x recency decay x volume anomaly |
| `tools/monster/correlation_matrix.py` | lead-lag confirmation / invalidation |
| `tools/monster/sizing.py` | ATR- and confidence-scaled risk, clamped to `[1%, 2%]` |
| `tools/monster/state_vector.py` | the distilled vector + the engine-facing CSV |
| `tools/monster_scan.py` | CLI: provenance CSVs → `state/monster/*.csv` |

### The score

```
S = 0.25·M + 0.35·T + 0.25·R + 0.15·C            (all layers in [0,1])
```

`S` is **unchanged** by the regime/ensemble work. The R layer's *source* changed
(multi-axis ensemble by default, the old SMA200-slope rule behind
`--regime-mode legacy`); its mapping did not. The strategy ensemble is a separate,
strictly fail-closed gate — see below.

* **M** microstructure — volume-anomaly z-score (logistic), or order-book
  imbalance when the provider publishes depth (the **stronger** of the two).
* **T** technical — six independent bullish factors (VWAP reclaim, RSI
  divergence, Fibonacci support, MACD thrust, Bollinger oversold, trend
  alignment). **At least 3 must agree.**
* **R** regime — `trend_up` / `trend_down` / `high_vol` / `range`. A long in a
  macro downtrend is **vetoed** unless a verified mean-reversion setup is
  present (oversold *and* below the lower band *and* bullish divergence).
* **C** cross-asset — lead-lag confirmation / invalidation from the rolling
  correlation matrix.

An entry needs **no veto AND `S >= 0.85`**. The engine's own signal generator
must *also* say Buy, so the two layers have to line up.

> **A gate that can never open is a bug, not discipline.** The first cut scored
> `T = votes/6`, which caps the reachable total at ≈0.81 — the 0.85 bar was
> mathematically unreachable. `T` now pays 0.90 for meeting the mandated
> three-factor confluence and 0.10 per extra factor. A textbook setup reaches
> **S = 0.865**; on 420 bars of real daily data nothing clears 0.85, which is
> the intended behaviour for a sniper.

### The gate in the engine

```bash
omniseed_agent2 trading-paper-session \
    --streams "AAPL,equity,models/market/paper/AAPL_1d.csv" \
    --monster-features state/monster --monster-min-conf 0.85
```

**It fails closed.** No row for that bar, `veto=1`, or `confidence <` the bar
all block the entry. A missing feature file is not an error — it is a refusal.
The gate only ever blocks *entries*: stops and exits are never touched.

`tools/paper_loop.py --monster-features state/monster` refreshes the feature
CSVs before every session, so the whole loop is wired end to end.

### What a live run looks like

Against the real 420-bar demo basket (AAPL / BTCUSDT / EURUSD):

```
  actions     : 0 entries, 0 exits, 0 halts, 0 abstains, 0 refused
  monster     : 166 blocked (gate: state/monster, S>=0.85)
```

and with the bar lowered to 0.70, to prove the gate is actually gating rather
than simply off:

```
  actions     : 1 entries, 1 exits, 0 halts, 0 abstains, 0 refused
  monster     : 160 blocked (gate: state/monster, S>=0.70)
```

The journal records `monster-block:<sym>:<veto|no-row|S=..<..>` events, and every
`signal` row carries the distilled verdict, e.g. `monster[S=0.640 range]`.

#### Legacy vs advanced regime, on the same real basket

Same 420-bar AAPL / BTCUSDT / EURUSD data, `--min-confidence 0.85`:

| mode | AAPL best S | BTCUSDT | EURUSD | AAPL regime mix |
|---|---|---|---|---|
| `--regime-mode legacy` | 0.765 (`trend_up`) | 0.675 (`range`) | 0.765 (`trend_up`) | 265 range / **136 trend_up** |
| `--regime-mode advanced` | 0.640 (`range`) | 0.675 (`range`) | 0.640 (`range`) | 306 range / **77 trend_up** |
| `+ --ensemble` | 0.640, **3** ensemble vetoes | 0.675 | 0.640 | unchanged |

Three things to read out of that:

* **The advanced engine is more conservative, and that is the point.** The
  legacy SMA200-slope rule called 136/420 AAPL bars `trend_up` and **143/420**
  EURUSD bars `trend_up`; the multi-axis ensemble calls 77 and **10**. FX majors
  really are near a random walk, and the slope rule was not noticing.
* **Zero proposals in every mode.** The gate is fail-closed and the demo basket
  contains no sniper setup at 0.85. That is the correct outcome, not a failure.
* **The ensemble barely fires** — 3 vetoes out of 420 AAPL bars, and mean
  conviction ≈ 0 on all three symbols. The weight floor keeps a lone signal from
  vetoing, and the zoo is balanced rather than systematically biased.

### Hard constraints

The 2 % per-trade ceiling and the 3 % daily / 6 % weekly kill-switch are
**unchanged and enforced in C++**. The Monster's `sizing.py` clamps to the same
`[1%, 2%]` band the engine enforces, so a bug in the Python layer still cannot
exceed 2 %. Nothing here claims an edge.

### The regime engine and the strategy ensemble

`--regime-mode advanced` (default) replaces the single SMA200-slope rule with
eight independent statistics blended into a continuous `trend_score`, plus a
Yang-Zhang volatility axis and an OU half-life. Thresholds are **calibrated
against measured distributions**, not picked by eye; the table is in
`docs/MONSTER_DESIGN.md` §5.2. `--regime-mode legacy` reproduces the old rule.

The volatility axis needed one more fix than the other statistics, and it is
worth knowing why: it originally ranked the *raw latest* Yang-Zhang estimate
against its own history, which fires on every volatility-expansion bar — and an
entry bar is a volatility-expansion bar by construction. Since a `high_vol` label
caps `S` at `0.25·1 + 0.35·1 + 0.25·0.25 + 0.15·1 = 0.8125 < 0.85`, that made the
sniper gate **unopenable**. It now ranks the **smoothed** estimate (mean of the
last `vol_smooth = 5` readings) and additionally demands *absolute* stress
(`ATR/close` above half the mandate's `atr_hi`), so `high_vol` means "this market
is stressed", not "this bar was big". See `docs/MONSTER_DESIGN.md` §5.3.

`--ensemble` routes the strategy zoo (momentum / mean-reversion / breakout /
order-flow imbalance) by regime and adds one extra gate:

```
veto "ensemble-opposed"  ⇔  agreement ≥ 0.60 AND conviction ≤ −0.35
                            AND total active weight ≥ 0.50
```

It **cannot** raise `S` — only block. The weight floor exists because
`agreement` is a share, so one lonely low-confidence signal would otherwise
score 1.00 and veto every long (observed live, now a regression test).

`--monster-features` also accepts `--monster-min-conf`, and the ensemble summary
appears in `state/monster/summary.json`.

### Funding-rate carry

`tools/monster/funding.py` is the one **non-directional** piece: delta-neutral
long-spot / short-perp capturing the perpetual funding payment. It ranks on
**net** APR after round-trip fees, spread and cost of capital — never gross.
The test suite's worked example: a memecoin showing 21.90% gross is blocked
(500 bps basis, 150 bps spot spread) and ranks *below* a clean 13.14% gross trade
that nets 2.22%. Delta-neutral is not risk-free; the flags are listed in the
design doc §7.

---

## 7. What is *not* here yet

Tracked honestly so it is not mistaken for done:

* ~~On-chain flows (whale / exchange in-out), event-calendar and social-volume
  signals (M7 extras) — planned, not built.~~ **BUILT in §48** —
  `tools/monster/signals.py`. See §7b below.
* ~~A **live depth provider** for the order-book path.~~ **BUILT in §48.** The
  layer now CONSUMES a depth feed when one is supplied (exact CKS OFI +
  `beta = c/depth^λ`); when none is, it still falls back to the signed-volume
  proxy, labelled `proxy` and discounted 30%. What is still absent is a *free
  venue that publishes the book* — the reader exists, the feed is the caller's
  job, exactly like news and funding.
* ~~The strategy factory with automatic promotion/retirement (M3) — planned.~~
  **BUILT in §50** — `tools/monster/factory.py`, opt-in via
  `SniperConfig(ensemble_factory=True)`. Tracks each strategy causally
  (warmup → probation → promoted → retired → retry), filters the ensemble by
  eligibility with probation shrinkage, and never amplifies conviction or
  invents a veto (clamped to the unfiltered base). Parity-safe: the C++ and
  default Python paths are bit-identical.
* Options-based volatility harvesting (selling implied vs realized). The
  volatility work here is the *vol-managed exposure* form (Moreira-Muir), which
  is evidence-backed and needs no options venue.
* News is fetched by the caller (`tools/fetch_news.py`) and passed in; the
  Monster does not yet run its own background poller inside the loop.
* Funding quotes are supplied by the caller (`funding.load_quotes_csv`); there is
  no live exchange poller wired.

---

## 7b. The external-signal layer (§48)

`tools/monster/signals.py` is the "hacker tools" layer: the four signal families
§7 used to list as planned, plus the order-book path. It is deliberately **one
module with one contract**, because the recurring shape of this project is that
the pieces exist and the *joint* does not.

Three rules hold it together:

1. **Pure and offline.** Every reader takes a path; nothing opens a socket.
   Fetching stays the caller's job (`tools/fetch_news.py` and friends).
2. **Provenance is named.** A depth-derived imbalance says `cks`; the
   signed-volume fallback says `proxy` **and is discounted** — the proxy cannot
   see limit orders or cancellations, which is most of what the CKS OFI
   measures. A feed that is stale or absent says so and contributes **nothing**;
   it is never silently zero-filled into a vote.
3. **Fail-closed, never promoting.** The event calendar is a **blocker**. No
   external signal can raise the sniper score `S` — `S` stays the mandate's
   weighted sum of M/T/R/C.

| family | input (`<name>.csv`) | reading |
|---|---|---|
| order book | `depth.csv` — `ts,bid_px,bid_sz,ask_px,ask_sz` | exact CKS OFI + `beta = c/depth^λ`; `source=cks`. No book → the proxy, `source=proxy`, discounted |
| on-chain | `onchain.csv` — `ts,exchange_netflow,whale_net,stablecoin_delta` | exchange **inflow** = bearish, whale **accumulation** = bullish, rising **stablecoin supply** = bullish; each squashed with `tanh(raw/scale)` |
| social | `social.csv` — `ts,mentions,positive,negative` | attention = **trailing** mention-volume z (no look-ahead), bias = attention × sentiment |
| calendar | `calendar.csv` — `ts,kind,name,impact` | a **high-impact** event opens the blackout window `[ts-pre, ts+post]`; a low-impact one never does |
| commodities | `commodities.csv` — `ts,symbol,close` | legs (`OIL`/`GOLD`/`COPPER`) merged into the cross-asset matrix |

**Signs are written down so they can be argued with.** An exchange *inflow* is
coins moving *to* venues, i.e. sell pressure; a positive *whale net* is
accumulation. Both are one line of code and one test, and both are the part a
reader should check first.

**The blackout is a veto, and it is named even when it changes nothing.** If
another gate already blocked the bar, the reason becomes
`<first-reason>+event-blackout` rather than being swallowed — a blackout that
disappeared behind an earlier veto would let the log claim the calendar was
silent when it was not.

`--signals DIR` (and `--commodities CSV`) wire it into `tools/monster_scan.py`.
The per-bar readings are mirrored to `state/monster/signals.csv`
(`ts,onchain,social,depth,depth_source,blackout,coverage`), and `coverage`
counts the live families — so **"no signal" and "no feed" are never confused**,
which is the whole point of the layer.

### The cross-asset time-alignment fix (§48)

`CorrelationMatrix.evaluate` used **one index for every series**. That is only
correct when the symbols share a timeline; with different histories it compares
bar 100 of one series against bar 100 of another, i.e. two different moments.
The scanner now builds a `ts → index` map per symbol and passes `idx_by_symbol`,
so every peer is read at the index of the **same timestamp**, and a peer with no
reading at that instant is **skipped** rather than mis-compared. Omitting the
map reproduces the old behaviour exactly, so nothing that worked changed.

`tests/test_monster_signals.py` pins it both ways: the positional read sees the
wrong moment, the aligned read sees the right one.

---

## 8. Test evidence

| Suite                        | What it pins down                                             |
|------------------------------|--------------------------------------------------------------|
| `omniseed_market_perception` | feed health, ABSTAIN reasons, governor set, asset-class risk, correlation cap, autonomy ladder |
| `omniseed_market_feeds`      | adapters, cache, limiter, normalisation, provenance reuse, meme refusal, explicit failure |
| `omniseed_market_data`       | T6 fetcher integrity (provenance, exit-2, no silent synth)    |
| `omniseed_trading_edge`      | walk-forward no-look-ahead, risk clamp, daily kill-switch, journal append/resume |
| `omniseed_paper_session`     | multi-asset merged timeline, per-asset gates, ABSTAIN, composite kill-switch, resume, determinism, **Monster gate (fail-closed, veto, threshold)** |
| `omniseed_paper_loop`        | watch parsing, CSV merge/dedup, feed health → ABSTAIN, engine invocation, status heartbeat |
| `omniseed_paper_reports`     | journal parsing, book replay, day P&L, Markdown/HTML rendering, opt-in SMTP config |
| `omniseed_monster_features`  | indicator correctness + the **no-look-ahead proof** (prefix invariance) |
| `omniseed_monster_regime`    | **calibration against known ground truth** (trend / OU / random walk): VR, Hurst, ER, ADX, choppiness, ρ(1), MA consistency, Yang-Zhang, OU half-life recovery, the degenerate-vol guards, hysteresis, prefix invariance |
| `omniseed_monster_strategies`| **exact CKS order-flow imbalance vs hand-computed values**, the OFI proxy labelled as a proxy, Donchian excludes the current bar, squeeze vs no-squeeze confidence, vol-target band, router continuity, **the veto weight floor (regression)**, prefix invariance |
| `omniseed_monster_funding`   | funding-period arithmetic, the net-of-cost stack, direction on sign, every risk flag, history summary, **the memecoin trap**, CSV reader skipping malformed rows |
| `omniseed_monster_sniper`    | layer arithmetic, every hard veto, **0.85 reachability**, `[1%,2%]` sizing band, cross-asset invalidation |
| `omniseed_monster_news`      | keyword weights, recency decay, sentiment alignment, anomaly detection, event match requires BOTH halves, CSV round-trip, scan CLI |
| `omniseed_monster_signals`   | the exact CKS OFI vs the labelled+discounted proxy, on-chain signs and staleness, social attention/bias, the high- vs low-impact blackout, pack CSV round-trip, the timestamp-aligned cross-asset matrix, and the scanner wiring the blackout into the engine-facing veto |

Run the board with `ctest --test-dir build -C Release`.

