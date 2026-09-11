// =============================================================================
//  OmniSeed — trading/simulate.cpp
//  Paper trading: deterministic fills with slippage + fees on top of the
//  core PositionManager accounting.
// =============================================================================
#include "omniseed/trading/simulate.h"

#include <algorithm>
#include <cmath>
#include <map>

namespace omniseed {
namespace trading {

// ===========================================================================
// PaperBroker
// ===========================================================================
bool PaperBroker::market_buy(const std::string& ticker, double qty,
                             double price, int64_t ts, std::string& err) {
    if (qty <= 0.0 || price <= 0.0) { err = "bad qty/price"; return false; }
    const double fill = price * (1.0 + cfg_.slippage_bps * 1e-4);
    const double fee = qty * fill * cfg_.fee_bps * 1e-4;
    // Reserve check including the fee.
    if (qty * fill + fee > pm_.cash() + 1e-9) {
        err = "insufficient cash (incl. fee)";
        return false;
    }
    if (!pm_.buy(ticker, qty, fill, ts, err)) return false;
    refresh();                                  // cash/positions live now
    TradeRecord t;
    t.ticker = ticker; t.qty = qty; t.entry_price = fill;
    t.entry_ts = ts; t.exit_reason = "paper-buy";
    fills_.push_back(t);
    return true;
}

bool PaperBroker::market_sell(const std::string& ticker, double qty,
                              double price, int64_t ts, std::string& err) {
    if (qty <= 0.0 || price <= 0.0) { err = "bad qty/price"; return false; }
    const double fill = price * (1.0 - cfg_.slippage_bps * 1e-4);
    if (!pm_.sell(ticker, qty, fill, ts, err)) return false;
    refresh();                                  // realized P&L live now
    TradeRecord t;
    t.ticker = ticker; t.qty = qty; t.exit_price = fill;
    t.exit_ts = ts; t.exit_reason = "paper-sell";
    fills_.push_back(t);
    return true;
}

double PaperBroker::buy_fraction(const std::string& ticker, double price,
                                 int64_t ts, double pct_of_cash,
                                 std::string& err) {
    if (price <= 0.0) { err = "bad price"; return 0.0; }
    const double budget = pm_.cash() * std::min(1.0, std::max(0.0, pct_of_cash));
    const double qty = std::floor(budget / price);
    if (qty <= 0.0) { err = "budget too small"; return 0.0; }
    return market_buy(ticker, qty, price, ts, err) ? qty : 0.0;
}

void PaperBroker::mark(const std::map<std::string, double>& prices, int64_t) {
    last_prices_ = prices;
    refresh();
    curve_.push_back(state_.equity);
    peak_ = std::max(peak_, state_.equity);
}

void PaperBroker::reset() {
    pm_ = PositionManager(cfg_.starting_cash);
    state_ = PortfolioState{};
    last_prices_.clear();
    curve_.clear();
    fills_.clear();
    peak_ = cfg_.starting_cash;
}

// ===========================================================================
// PortfolioTracker
// ===========================================================================
void PortfolioTracker::step(const std::map<std::string, double>& prices,
                            int64_t ts) {
    broker_.mark(prices, ts);
}

SimulatorReport PortfolioTracker::report(double periods_per_year) const {
    SimulatorReport r;
    r.equity_curve = broker_.equity_curve();
    r.final_equity = r.equity_curve.empty() ? broker_.cash()
                                            : r.equity_curve.back();
    r.metrics = compute_metrics(r.equity_curve, periods_per_year);
    r.cycles = static_cast<int64_t>(r.equity_curve.size());
    return r;
}

} // namespace trading
} // namespace omniseed
