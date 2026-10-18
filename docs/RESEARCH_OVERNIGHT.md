# RESEARCH_OVERNIGHT.md — OmniSeed M4 Options Volatility Harvesting

## Hour 1 Deep Research — 2026-10-04

---

## 1. Variance Risk Premium (VRP) — Core Theory

- IV systematically exceeds RV ~85% of months for S&P 500 (Bakshi & Kapadia; Carr & Wu; Bollerslev, Tauchen & Zhou)
- Typical VRP: 2-4 vol points on SPX 30-day options (FlashAlpha, HL Hunt)
- VRP is compensation for jump risk, correlation spikes, vega convexity — NOT a mispricing
- Crypto VRP is larger: BTC 5-15 vol points, ETH 5-20 vol points (Hypercall course)
- VRP can go negative — worst case inversions reach -33 vol points (Jan 2009 SPX)

**Key identity for M4 signals:**
P&L of hedged option ≈ ½ × Γ × S² × (σ²_realized − σ²_implied) per unit time

---

## 2. IV-RV Spread Strategies

**Threshold-based signal (Demonjoy / industry standard):**
| IV/HV Ratio | Action |
|---|---|
| > 1.5 | Sell vol (straddle/strangle) |
| 1.2-1.5 | Light sell |
| 1.0-1.2 | Observe |
| 0.8-1.0 | Light buy |
| < 0.8 | Buy vol |

**QuantConnect dual-VIX signal (Zarattini, Mele, Aziz 2025):**
- eVRP = VIX − eRV_30 (expected vol from 10-day returns × √252)
- Term structure signal: VIX vs VIX3M
- Position sizing: VIX/100 weight
- Sharpe 0.729, outperformed SPY buy-and-hold over Jan 2016–Jul 2026

**Miao & Sorokina (arXiv 2609.01183) short-put VRP strategy:**
- Cash-secured put writing on nuclear/energy equities
- Unconditional: 18.7% annual, Sharpe 7.8, max drawdown ~0%
- GARCH-filtered (IV/RV ≥ 1.10): Sharpe 2.6, max drawdown 7.7%
- Positive returns all 25 years (2000-2024)

**Wysocki (arXiv 2508.16598) Kelly-VIX sizing:**
- Ultra-short-dated (0-5 DTE), far OTM options best risk-adjusted
- Hybrid Kelly+VIX-rank sizing balances return vs drawdown
- SPXW options

---

## 3. Volatility Surface Models

**SVI (Gatheral 2004):**
w(k) = a + b(ρ(k−m) + √((k−m)² + σ²))
- k = ln(K/F), w = σ²T
- 5 parameters: a, b, ρ, m, σ
- Fits total variance across strikes for each expiry
- FlashAlpha SDK implements SVI calibration

**SABR (Hagan 2002):**
- α (vol level), β (CEV exponent), ρ (correlation), ν (vol of vol)
- SVI generally preferred for fitting; SABR for dynamics/simulation
- tensorquantlib has both SABR and SVI implementations

**Butterfly arbitrage violations:**
- Violations in SVI surface = VRP harvesting opportunity
- Convexity check: d²w/dk² ≥ 0 for no-arbitrage

---

## 4. Gamma Exposure (GEX) — Regime Filter

**Key levels:**
- Gamma Flip: where net dealer γ crosses zero (structural pivot)
- Call Wall: max positive γ strike (resistance in pos γ regime)
- Put Wall: max negative γ strike (support in neg γ regime)

**Regime signal:**
- Positive GEX → mean-reverting, range-bound → sell premium
- Negative GEX → trending, expanding vol → buy premium or avoid

**Dealer hedging mechanics:**
- Long γ: dealers buy dips, sell rallies → stabilizes price
- Short γ: dealers sell dips, buy rallies → amplifies moves

**0DTE dominance:** 66.2% of SPX options volume Jul 2026 → extreme gamma sensitivity

---

## 5. Data Sources (Crypto-Forward — fits OmniSeed's Deribit/Binance focus)

**Deribit:**
- REST API v2: `/api/v2/public/options_chain` — strikes, IV, greeks, OI per expiry
- WebSocket: real-time quotes, trades, instruments
- Free tier (rate-limited), no API key needed for public data
- Largest BTC/ETH options exchange

**Binance Options:**
- `/eapi/v1/mark` — mark price + greeks
- `/eapi/v1/depth` — order book
- `/eapi/v1/exchangeInfo` — symbol metadata
- Unified with futures API

**Laevitas (derivatives data aggregator):**
- REST API v2: OHLCVT, funding, OI, liquidations, option Greeks
- 5 exchanges: Binance, Deribit, OKX, Bybit, Hyperliquid
- 5+ years historical data
- WebSocket + MCP server (20+ tools for AI agents)
- Pay-per-request via x402 protocol

**CryptoDataDownload:**
- Options analytics: volatility surfaces, higher-order Greeks, IV-vs-RV spreads
- S3-backed historical endpoints
- Flat $79.99/mo

---

## 6. GitHub Repos / Libraries

- **tensorquantlib** (raswanthmalai19): SABR, SVI, Black-Scholes, Heston, autodiff Greeks
- **flashalpha-python** (FlashAlpha-lab): GEX dashboards, IV scanners, vol surface monitors, 0DTE bots
- **pySABR**: SABR calibration
- **QuantConnect research**: VRP harvesting with dual-VIX signal
- **Volatility surface repos**: multiple on GitHub searching "iv_skew" / "vol_surface"

---

## 7. Key Signals for M4 Design

1. **IV-RV Spread Signal** — sell vol when IV ≫ RV, buy when IV ≪ RV
2. **Vol Surface Curvature** — butterfly arb violations → short overpriced wings
3. **Skew Tilt Signal** — put/call IV differential → directional vol bet
4. **Term Structure Signal** — contango vs backwardation → calendar spread
5. **GEX Regime Filter** — pos γ = range (sell premium), neg γ = trend (avoid or hedge)
6. **Variance Swap Proxy** — synthetic replication via 1/K² weighted options

---

## 8. M4 Integration Points (per M3 Architecture)

- Vol strategies must output `StrategySignal(name, direction∈[-1,1], confidence∈[0,1], regime_fit, reason)`
- Must plug into factory promotion/retirement (z-score vs coin-flip)
- Must satisfy BLOCK ONLY, NO LOOK-AHEAD, FAIL CLOSED invariants
- Option chain data flows via market_feeds (same as depth/onchain/social in §48)
- C++ engine consumes option-chain CSVs from Python tooling
- Regime engine (regime.py) provides vol regime context
- Sniper engine opt-in via SniperConfig layer weights
