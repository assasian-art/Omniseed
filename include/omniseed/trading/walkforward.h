// =============================================================================
//  OmniSeed — trading/walkforward.h
//  Trading Edge Lab (T7.1): walk-forward validation.
//
//  A single backtest over the whole history is an in-sample number — it can
//  always be tuned to look good. Walk-forward splits the series into
//  consecutive train/test folds, picks the signal parameters on the TRAIN
//  window only, then replays them, untouched, on the following TEST window.
//  The stitched test windows form a genuine out-of-sample (OOS) equity curve.
//
//  Hard no-look-ahead: every test window begins exactly where its train
//  window ends (`test_begin == train_end`), and selection never sees a test
//  bar. Backtester::run is itself next-bar-open, so fills use only data that
//  was already observable.
//
//  Honest scope: an OOS curve that survives walk-forward is *evidence*, not a
//  promise. Past edges decay. See docs/TRADING_GUIDE.md.
// =============================================================================
#pragma once

#include "omniseed/trading/trading_engine.h"

#include <cstdint>
#include <string>
#include <vector>

namespace omniseed {
namespace trading {

// ===========================================================================
// One train/test fold
// ===========================================================================
struct WalkForwardFold {
    size_t train_begin = 0, train_end = 0;   // [begin, end)
    size_t test_begin  = 0, test_end  = 0;   // [begin, end); == train_end
    int    chosen_index = -1;                // index into the candidate grid
    SignalGenerator::Config chosen;          // parameters selected on TRAIN
    double train_return_pct = 0.0;           // in-sample result for `chosen`
    double train_sharpe = 0.0;
    BacktestReport test;                     // out-of-sample replay
};

// ===========================================================================
// Aggregate walk-forward result
// ===========================================================================
struct WalkForwardReport {
    std::vector<WalkForwardFold> folds;
    std::vector<double> oos_equity;          // stitched out-of-sample curve
    PerfMetrics oos_metrics;
    double oos_total_return_pct = 0.0;
    double is_mean_return_pct   = 0.0;       // mean in-sample return
    // In-sample minus out-of-sample mean return: a large positive gap is the
    // classic signature of overfitting.
    double degradation_pct = 0.0;
    // Fraction of folds whose OOS return was positive.
    double oos_hit_rate = 0.0;
    int64_t bars = 0;
    int64_t folds_run = 0;
    bool profitable_oos = false;
    std::string error;
};

// ===========================================================================
// WalkForward
// ===========================================================================
class WalkForward {
public:
    struct Config {
        size_t train_bars = 252;             // >= 60 (Backtester minimum)
        size_t test_bars  = 63;              // one quarter of a year
        size_t step_bars  = 63;              // == test_bars => non-overlapping
        bool   anchored   = false;           // false: rolling; true: growing
        RiskLimits risk;
        double starting_cash = 100000.0;
        std::string ticker = "TEST";
        // Parameter grid evaluated on each train window. Empty => the built-in
        // default grid (see default_candidates()).
        std::vector<SignalGenerator::Config> candidates;
    };

    // A small deterministic grid over the two knobs that matter most for this
    // rule set: the fast MA and the oversold entry threshold.
    static std::vector<SignalGenerator::Config> default_candidates();

    static WalkForwardReport run(const std::vector<Bar>& bars, const Config& cfg);
};

} // namespace trading
} // namespace omniseed
