// =============================================================================
//  OmniSeed — trading/simulate.h
//  Paper Trading Simulator (Phase-16 "Omega Pass"):
//
//    * PaperBroker        — simulated order execution with realistic
//                           slippage + fees, cash/position accounting
//    * PortfolioTracker   — equity snapshots over time (curve + peak)
//    * PerformanceMetrics — reuse of compute_metrics for live portfolios
//
//  The ExecutionAgent (agent/sub_agents.h) routes consensus verdicts here.
//  Real broker integration (Alpaca) is documented in docs/TRADING_GUIDE.md as
//  an opt-in adapter; this module never places real orders.
// =============================================================================
#pragma once

#include "omniseed/trading/trading_engine.h"

#include <map>
#include <string>
#include <vector>

namespace omniseed {
namespace trading {

// ===========================================================================
// PaperBrokerConfig
// ===========================================================================
struct PaperBrokerConfig {
    double starting_cash   = 100000.0;
    double fee_bps         = 5.0;      // per side
    double slippage_bps    = 2.0;      // applied against the caller's price
    double default_order_pct = 0.05;   // default order = 5% of cash
};

// ===========================================================================
// PaperBroker — deterministic market-order simulator (long-only)
// ===========================================================================
class PaperBroker {
public:
    explicit PaperBroker(const PaperBrokerConfig& cfg = {})
        : cfg_(cfg), pm_(cfg.starting_cash) {
        state_.cash = cfg.starting_cash;   // live before the first mark()
        peak_ = cfg.starting_cash;
    }

    // Market buy: pays `price * (1 + slippage)` plus the fee. Refuses when
    // cash is insufficient or the order is degenerate.
    bool market_buy(const std::string& ticker, double qty, double price,
                    int64_t ts, std::string& err);

    // Market sell: receives `price * (1 - slippage)` minus the fee;
    // realizes P&L. Refuses when the position is short.
    bool market_sell(const std::string& ticker, double qty, double price,
                     int64_t ts, std::string& err);

    // Convenience: buy `pct_of_cash` fraction of cash worth ( conviction-
    // scaled by the ExecutionAgent). Returns qty actually bought (0 = none).
    double buy_fraction(const std::string& ticker, double price, int64_t ts,
                        double pct_of_cash, std::string& err);

    // Marks to market and records an equity snapshot (call once per cycle).
    void mark(const std::map<std::string, double>& prices, int64_t ts);

    const PortfolioState& state() const { return state_; }
    double cash() const { return state_.cash; }
    const std::vector<double>& equity_curve() const { return curve_; }
    double equity_peak() const { return peak_; }

    void reset();

    const PaperBrokerConfig& config() const { return cfg_; }

private:
    PaperBrokerConfig cfg_;
    PositionManager pm_;                       // reuse core accounting
    PortfolioState state_;
    std::map<std::string, double> last_prices_;
    std::vector<double> curve_;
    double peak_ = 0.0;
    std::vector<TradeRecord> fills_;

    // Re-marks at the last known prices so cash/positions are live after
    // every fill (not only at explicit mark() cycles).
    void refresh() { state_ = pm_.mark(last_prices_); }
};

// ===========================================================================
// PortfolioTracker — thin wrapper: broker + prices per cycle -> metrics
// ===========================================================================
struct SimulatorReport {
    std::vector<double> equity_curve;
    std::vector<TradeRecord> trades;           // broker fills
    PerfMetrics metrics;
    double final_equity = 0.0;
    int64_t cycles = 0;
};

class PortfolioTracker {
public:
    explicit PortfolioTracker(PaperBroker& broker) : broker_(broker) {}

    // One simulation cycle: mark prices, record equity.
    void step(const std::map<std::string, double>& prices, int64_t ts);

    // Summarize: metrics over the recorded curve (periods_per_year depends
    // on the bar cadence: 252 for daily, 252*252 for minute bars, ...).
    SimulatorReport report(double periods_per_year) const;

private:
    PaperBroker& broker_;
};

} // namespace trading
} // namespace omniseed
