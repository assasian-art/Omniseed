// =============================================================================
//  OmniSeed — trading/trading_engine.h
//  Trading Framework Core (Phase-16 "Omega Pass"):
//
//    * OHLCV / Bar            — the market-data currency (CSV load/save)
//    * MarketDataFeed         — pluggable price feeds (CSV replay now; live
//                               HTTP/WS feeds are user-provided adapters)
//    * SignalGenerator        — technical indicators (SMA, EMA, RSI, MACD,
//                               Bollinger bands, ATR) + rule-based signals
//    * PositionManager        — open positions, realized/unrealized P&L,
//                               gross/net exposure
//    * RiskManager            — fractional Kelly sizing, stop-loss, take
//                               profit, max drawdown + exposure limits
//    * Backtester             — bar-replay simulation of a SignalGenerator
//                               through the RiskManager/PositionManager with
//                               slippage + fees; equity curve + metrics
//                                 (Sharpe, Sortino, max DD, win rate, CAGR)
//
//  Honest-scope note (docs/TRADING_GUIDE.md): these are educational,
//  deterministic, pure-C++17 analytics. "Loss minimization, not elimination."
//  No module here guarantees profit; real trading requires a real broker +
//  licensed market data.
// =============================================================================
#pragma once

#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace omniseed {
namespace trading {

// ===========================================================================
// OHLCV bar + CSV persistence
// ===========================================================================
struct Bar {
    int64_t  time = 0;          // unix seconds (UTC)
    double   open = 0.0, high = 0.0, low = 0.0, close = 0.0, volume = 0.0;
};

// CSV with a header line: time,open,high,low,close,volume (unix seconds or
// "YYYY-MM-DD[ HH:MM]"). Sorted ascending on load; malformed rows skipped.
bool load_bars_csv(const std::string& path, std::vector<Bar>& out,
                   std::string& err);
bool save_bars_csv(const std::string& path, const std::vector<Bar>& bars,
                   std::string& err);

// ===========================================================================
// MarketDataFeed — pull prices from a user-selected source
// ===========================================================================
class MarketDataFeed {
public:
    enum class Kind { CsvReplay, Http };   // Http = stub unless wired by user

    struct Config {
        Kind kind = Kind::CsvReplay;
        std::string csv_path;              // CsvReplay source
        std::string ticker;
    };

    MarketDataFeed() : MarketDataFeed(Config{}) {}
    explicit MarketDataFeed(const Config& cfg) : cfg_(cfg) {}

    // Load the full history (CSV path) — bars are then replayed by index.
    bool load(std::string& err);
    bool loaded() const { return loaded_; }
    const std::vector<Bar>& bars() const { return bars_; }
    const std::string& ticker() const { return cfg_.ticker; }
    const std::string& error() const { return err_; }

    // Latest close (last bar); false when no data.
    bool latest_close(double& out) const;

private:
    Config cfg_;
    std::vector<Bar> bars_;
    bool loaded_ = false;
    std::string err_;
};

// ===========================================================================
// SignalGenerator — indicators + rule-based signals (all deterministic)
// ===========================================================================
struct Indicators {
    std::vector<double> sma20, sma50, sma200, ema12, ema26;
    std::vector<double> rsi14;             // 0..100, first 13 entries = 50
    std::vector<double> macd, macd_signal, macd_hist;
    std::vector<double> bb_mid, bb_upper, bb_lower;
    std::vector<double> atr14;
};

enum class Action : int8_t { Hold = 0, Buy, Sell };

struct Signal {
    Action  action = Action::Hold;
    double  strength = 0.0;    // 0..1 (sum of |voting rules|, capped)
    std::string reason;        // human-readable rule trace, e.g. "RSI<30; MAx"
};

class SignalGenerator {
public:
    struct Config {
        int32_t sma_fast = 20, sma_slow = 50;
        int32_t rsi_period = 14;
        double  rsi_buy_below  = 40.0;     // oversold entry (tuned on real
        double  rsi_sell_above = 70.0;     //   daily bars: 35/65 never fires)
        int32_t macd_fast = 12, macd_slow = 26, macd_signal_p = 9;
        int32_t bb_period = 20;
        double  bb_sigma = 2.0;
        int32_t atr_period = 14;
    };

    // Computes every indicator series over `bars` (size == bars.size()).
    static Indicators compute(const std::vector<Bar>& bars) {
        return compute(bars, Config{});
    }
    static Indicators compute(const std::vector<Bar>& bars, const Config& cfg);

    // Rules at index i (uses only data <= i; no look-ahead):
    //   +1 mean-reversion : close < lower BB and RSI < buy threshold
    //   +1 trend          : sma_fast > sma_slow and MACD hist > 0
    //   -1 exit           : close > upper BB, or RSI > sell threshold,
    //                       or sma_fast < sma_slow
    static Signal evaluate(const Indicators& ind, const std::vector<Bar>& bars, size_t i) {
        return evaluate(ind, bars, i, Config{});
    }
    static Signal evaluate(const Indicators& ind, const std::vector<Bar>& bars,
                           size_t i, const Config& cfg);
};

// ===========================================================================
// Position / PositionManager
// ===========================================================================
enum class Side : int8_t { Long = 0 };   // spot-style long-only (honest scope)

struct Position {
    std::string ticker;
    Side    side = Side::Long;
    double  qty = 0.0;
    double  avg_price = 0.0;
    int64_t opened_at = 0;
};

struct PortfolioState {
    double  cash = 0.0;
    double  equity = 0.0;                 // cash + market value of positions
    double  gross_exposure = 0.0;         // sum |market value|
    double  unrealized_pnl = 0.0;
    double  realized_pnl = 0.0;           // lifetime, session-scoped
    std::map<std::string, Position> positions;
};

class PositionManager {
public:
    explicit PositionManager(double starting_cash = 100000.0)
        : cash_(starting_cash), realized_(0.0) {}

    // Market fills at `price` with optional slippage applied by the caller
    // (Backtester applies bps slippage before calling these).
    bool buy (const std::string& ticker, double qty, double price,
              int64_t ts, std::string& err);
    bool sell(const std::string& ticker, double qty, double price,
              int64_t ts, std::string& err);

    // Marks positions to `prices` and recomputes equity/exposure.
    PortfolioState mark(const std::map<std::string, double>& prices) const;

    // Restore a persisted book (resume, M5). Positions with qty <= 0 are
    // dropped. The realized-P&L counter is left untouched (history lives in
    // the append-only journal, not here).
    void restore(double cash, const std::map<std::string, Position>& positions);

    double cash() const { return cash_; }
    double realized_pnl() const { return realized_; }
    const std::map<std::string, Position>& positions() const { return pos_; }

private:
    double cash_;
    double realized_;
    std::map<std::string, Position> pos_;
};

// ===========================================================================
// Asset class — perception/risk classification (M1/M4)
// ===========================================================================
enum class AssetClass : int8_t { Equity = 0, Crypto, Fx, Meme, Unknown };

inline const char* to_string(AssetClass ac) {
    switch (ac) {
        case AssetClass::Equity:  return "equity";
        case AssetClass::Crypto:  return "crypto";
        case AssetClass::Fx:      return "fx";
        case AssetClass::Meme:    return "meme";
        default:                  return "unknown";
    }
}

// ===========================================================================
// RiskManager — Kelly sizing, stops, drawdown + exposure limits
// ===========================================================================
struct RiskLimits {
    double  kelly_fraction   = 0.25;   // trade 25% of full Kelly (quarter-Kelly)
    double  max_position_pct = 0.20;   // <= 20% of equity in one name
    double  stop_loss_pct    = 0.08;   // 8% per-position stop
    double  take_profit_pct  = 0.0;    // 0 = disabled
    double  max_drawdown_pct = 0.20;   // halt new entries beyond 20% peak DD
    double  max_gross_exposure_pct = 1.0;  // <= 100% of equity
    double  fee_bps          = 5.0;    // round-trip cost model (per side)
    double  slippage_bps     = 2.0;
    // --- Edge-lab additions (T7.1) ---------------------------------------
    // Per-trade risk budget: a stop-out should lose this fraction of equity.
    // The engine CLAMPS this into [1%, 2%] — it will never risk more than 2%
    // of equity on a single trade, whatever the caller configures.
    double  risk_per_trade_pct = 0.015;
    // Daily kill-switch: once the day's loss reaches this fraction of the
    // day's start equity, NEW entries are refused until the next UTC day.
    // Exits/stops are never blocked (the switch only gates new risk).
    double  max_daily_loss_pct = 0.03;
    // --- M4 multi-asset risk engine --------------------------------------
    // Weekly kill-switch, measured from the week's high-water mark.
    double  max_weekly_loss_pct = 0.06;
    // Meme assets are capped far harder until they have a proven record, and
    // are RESEARCH-ONLY by default (no paper/live entries at all).
    double  meme_max_risk_pct  = 0.0025;   // 0.25% of equity
    bool    meme_research_only = true;
    // Pairwise correlation cap across book holdings (|Pearson r|).
    double  max_correlation    = 0.80;
};

// The per-trade risk budget is clamped into this band (see
// RiskManager::size_by_risk). Exposed so callers/tests can reason about it.
constexpr double kMinRiskPerTradePct = 0.01;
constexpr double kMaxRiskPerTradePct = 0.02;

class RiskManager {
public:
    struct Sizing {
        double qty = 0.0;
        std::string reason;            // sizing trace
        bool allowed = true;
    };

    // Fractional Kelly on win-rate/avg-win/avg-loss estimates; then capped by
    // max_position_pct of equity and available cash. win_prob in (0,1).
    static Sizing size_position(double equity, double price,
                                double win_prob, double avg_win,
                                double avg_loss, const RiskLimits& lim);

    // Risk-budget sizing (edge lab): qty such that a stop-out at
    // lim.stop_loss_pct loses exactly `risk_per_trade_pct` of equity —
    //   qty = equity * risk_pct / (price * stop_pct)
    // risk_per_trade_pct is CLAMPED into [kMinRiskPerTradePct,
    // kMaxRiskPerTradePct] (1%..2%), and the notional is then capped by
    // max_position_pct of equity. Refuses (allowed=false) on bad inputs or
    // when the resulting qty floors to 0. The reason trace names the clamp
    // whenever it fires, so an over-large config is visible, not silent.
    static Sizing size_by_risk(double equity, double price,
                               const RiskLimits& lim);

    // Per-trade risk budget for an asset class. Meme is capped by
    // meme_max_risk_pct; returns 0.0 when the class may not be traded at all
    // (meme_research_only). Non-meme classes use risk_per_trade_pct.
    static double effective_risk_pct(AssetClass ac, const RiskLimits& lim);

    // size_by_risk with the asset-class override applied. Refuses
    // (allowed=false, qty=0) for a research-only class.
    static Sizing size_by_risk(double equity, double price, AssetClass ac,
                               const RiskLimits& lim);

    // Pearson correlation of two equal-length series. 0 when undefined
    // (length < 2, mismatched lengths, or zero variance in either series).
    static double correlation(const std::vector<double>& a,
                              const std::vector<double>& b);

    // True when `candidate` keeps |correlation| below lim.max_correlation
    // against every series in `book`. `worst` reports the largest |r| seen.
    static bool correlation_ok(const std::vector<std::vector<double>>& book,
                               const std::vector<double>& candidate,
                               const RiskLimits& lim, double& worst,
                               std::string& why_not);

    // Stop/take-profit check for an open position.
    static Action check_exit(const Position& pos, double price,
                             const RiskLimits& lim);

    // Drawdown state from an equity curve (peak-trough / peak).
    static double drawdown(const std::vector<double>& equity);

    // Portfolio-level gate: true when new entries are permitted.
    static bool entries_allowed(const PortfolioState& ps, double equity_peak,
                                const RiskLimits& lim, std::string& why_not);
};

// ===========================================================================
// DailyRiskGovernor — per-UTC-day loss kill-switch (edge lab, T7.1)
// ===========================================================================
// Feed it once per bar, in chronological order, with the current mark-to-
// market equity. It tracks each UTC day's high-water mark and, once the
// drawdown from that peak reaches `max_daily_loss_pct`, refuses new entries
// for the remainder of the day. The switch re-arms automatically at the next
// day boundary. Exits and stops are never gated — only new risk is.
//
// Deterministic and allocation-free; the caller owns all state.
class DailyRiskGovernor {
public:
    struct Config {
        double  max_daily_loss_pct = 0.03;   // 3% of the day's opening equity
        int64_t day_seconds        = 86400;  // UTC day bucket width
    };

    DailyRiskGovernor() : DailyRiskGovernor(Config{}) {}
    explicit DailyRiskGovernor(const Config& cfg) : cfg_(cfg) {}

    // Returns true when new entries are permitted for `ts`. `why_not` is set
    // on the transition to halted. Must be called with non-decreasing ts.
    bool allow(int64_t ts, double equity, std::string& why_not);

    bool    halted() const { return halted_; }
    int64_t day_index() const { return day_; }
    double  day_start_equity() const { return day_start_; }
    double  day_peak_equity() const { return day_peak_; }
    // Number of times the switch has tripped (one per halted day).
    int64_t trips() const { return trips_; }

    void reset();

private:
    Config  cfg_;
    bool    have_day_ = false;
    int64_t day_ = 0;
    double  day_start_ = 0.0;
    double  day_peak_ = 0.0;
    bool    halted_ = false;
    int64_t trips_ = 0;
};

// ===========================================================================
// Backtester + performance metrics
// ===========================================================================
struct TradeRecord {
    std::string ticker;
    int64_t entry_ts = 0, exit_ts = 0;
    double  entry_price = 0.0, exit_price = 0.0, qty = 0.0;
    double  pnl = 0.0;                 // net of fees
    std::string exit_reason;           // "signal", "stop-loss", "take-profit"
};

struct BacktestReport {
    std::vector<double> equity_curve;  // per bar
    std::vector<TradeRecord> trades;
    // Metrics:
    double final_equity = 0.0, total_return_pct = 0.0, cagr_pct = 0.0;
    double sharpe = 0.0, sortino = 0.0, max_drawdown_pct = 0.0;
    double win_rate = 0.0, profit_factor = 0.0;
    int64_t bars = 0;
    std::string error;
};

class Backtester {
public:
    struct Config {
        RiskLimits risk;
        SignalGenerator::Config signals;
        double starting_cash = 100000.0;
        std::string ticker = "TEST";
    };

    // Replays `bars` once, bar by bar, honoring no look-ahead: signals at bar
    // i execute at bar i+1's open (next-bar-open convention).
    static BacktestReport run(const std::vector<Bar>& bars, const Config& cfg);
};

// Performance metrics over an equity curve sampled once per period
// (periods_per_year: 252 for daily, 252*6.5*60 for 1-minute bars, ...).
struct PerfMetrics {
    double sharpe = 0.0, sortino = 0.0, max_drawdown_pct = 0.0;
    double total_return_pct = 0.0, cagr_pct = 0.0;
};
PerfMetrics compute_metrics(const std::vector<double>& equity,
                            double periods_per_year);

} // namespace trading
} // namespace omniseed
