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

    explicit MarketDataFeed(const Config& cfg = {}) : cfg_(cfg) {}

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
    static Indicators compute(const std::vector<Bar>& bars,
                              const Config& cfg = {});

    // Rules at index i (uses only data <= i; no look-ahead):
    //   +1 mean-reversion : close < lower BB and RSI < buy threshold
    //   +1 trend          : sma_fast > sma_slow and MACD hist > 0
    //   -1 exit           : close > upper BB, or RSI > sell threshold,
    //                       or sma_fast < sma_slow
    static Signal evaluate(const Indicators& ind, const std::vector<Bar>& bars,
                           size_t i, const Config& cfg = {});
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

    double cash() const { return cash_; }
    double realized_pnl() const { return realized_; }
    const std::map<std::string, Position>& positions() const { return pos_; }

private:
    double cash_;
    double realized_;
    std::map<std::string, Position> pos_;
};

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
};

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
