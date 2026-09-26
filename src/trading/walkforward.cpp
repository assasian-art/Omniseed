// =============================================================================
//  OmniSeed — trading/walkforward.cpp
//  Walk-forward validation: select on train, replay untouched on test.
// =============================================================================
#include "omniseed/trading/walkforward.h"

#include <algorithm>
#include <cmath>

namespace omniseed {
namespace trading {

// ===========================================================================
// Default candidate grid
// ===========================================================================
std::vector<SignalGenerator::Config> WalkForward::default_candidates() {
    std::vector<SignalGenerator::Config> v;
    for (const int32_t fast : {10, 20, 30}) {
        for (const double buy : {35.0, 40.0, 45.0}) {
            SignalGenerator::Config c;
            c.sma_fast = fast;
            c.rsi_buy_below = buy;
            v.push_back(c);
        }
    }
    return v;   // 9 deterministic configs
}

// ===========================================================================
// run
// ===========================================================================
WalkForwardReport WalkForward::run(const std::vector<Bar>& bars,
                                   const Config& cfg) {
    WalkForwardReport rep;
    const size_t n = bars.size();
    rep.bars = static_cast<int64_t>(n);

    if (cfg.train_bars < 60 || cfg.test_bars < 60) {
        rep.error = "train_bars and test_bars must each be >= 60 "
                    "(Backtester's minimum replay length)";
        return rep;
    }
    if (cfg.step_bars == 0) {
        rep.error = "step_bars must be >= 1";
        return rep;
    }
    if (n < cfg.train_bars + cfg.test_bars) {
        rep.error = "history too short for one train/test fold";
        return rep;
    }

    std::vector<SignalGenerator::Config> cands =
        cfg.candidates.empty() ? default_candidates() : cfg.candidates;
    if (cands.empty()) {
        rep.error = "no candidate parameter sets";
        return rep;
    }

    Backtester::Config bcfg;
    bcfg.risk = cfg.risk;
    bcfg.ticker = cfg.ticker;

    double running_equity = cfg.starting_cash;
    int64_t profitable_folds = 0;

    for (size_t s = 0; s + cfg.train_bars + cfg.test_bars <= n;
         s += cfg.step_bars) {
        const size_t train_begin = cfg.anchored ? 0 : s;
        const size_t train_end   = s + cfg.train_bars;   // == test_begin
        const size_t test_begin  = train_end;
        const size_t test_end    = train_end + cfg.test_bars;

        const std::vector<Bar> train_slice(bars.begin() + static_cast<long>(train_begin),
                                           bars.begin() + static_cast<long>(train_end));
        const std::vector<Bar> test_slice(bars.begin() + static_cast<long>(test_begin),
                                          bars.begin() + static_cast<long>(test_end));

        // --- select parameters on the TRAIN window only -------------------
        int    best = -1;
        double best_sharpe = -1e300, best_ret = -1e300;
        BacktestReport best_train;
        for (size_t ci = 0; ci < cands.size(); ++ci) {
            bcfg.signals = cands[ci];
            bcfg.starting_cash = cfg.starting_cash;
            const BacktestReport r = Backtester::run(train_slice, bcfg);
            if (!r.error.empty()) continue;
            const bool better =
                r.sharpe > best_sharpe + 1e-9 ||
                (std::fabs(r.sharpe - best_sharpe) <= 1e-9 && r.total_return_pct > best_ret);
            if (better) {
                best = static_cast<int>(ci);
                best_sharpe = r.sharpe;
                best_ret = r.total_return_pct;
                best_train = r;
            }
        }
        if (best < 0) {
            rep.error = "no candidate produced a usable train backtest";
            return rep;
        }

        // --- replay the chosen params on the TEST window (out of sample) ---
        WalkForwardFold fold;
        fold.train_begin = train_begin;
        fold.train_end   = train_end;
        fold.test_begin  = test_begin;
        fold.test_end    = test_end;
        fold.chosen_index = best;
        fold.chosen = cands[static_cast<size_t>(best)];
        fold.train_return_pct = best_train.total_return_pct;
        fold.train_sharpe = best_train.sharpe;

        bcfg.signals = fold.chosen;
        bcfg.starting_cash = running_equity;      // chain equity across folds
        fold.test = Backtester::run(test_slice, bcfg);
        if (!fold.test.error.empty()) {
            rep.error = "out-of-sample replay failed: " + fold.test.error;
            return rep;
        }

        // Stitch the OOS curve; skip the first point after the first fold,
        // since it duplicates the previous fold's final equity.
        const std::vector<double>& eq = fold.test.equity_curve;
        const size_t from = rep.oos_equity.empty() ? 0 : 1;
        for (size_t i = from; i < eq.size(); ++i) rep.oos_equity.push_back(eq[i]);

        running_equity = fold.test.final_equity;
        if (fold.test.total_return_pct > 0.0) ++profitable_folds;
        rep.folds.push_back(fold);
    }

    if (rep.folds.empty()) {
        rep.error = "no fold fit in the history";
        return rep;
    }
    rep.folds_run = static_cast<int64_t>(rep.folds.size());

    // --- aggregate metrics ------------------------------------------------
    const double span_s = n >= 2
        ? static_cast<double>(bars.back().time - bars.front().time) : 0.0;
    double periods = 252.0;
    if (n >= 2 && span_s > 0.0)
        periods = static_cast<double>(n - 1) / (span_s / (252.0 * 86400.0));

    rep.oos_metrics = compute_metrics(rep.oos_equity, std::max(1.0, periods));
    const double first = rep.oos_equity.empty() ? cfg.starting_cash
                                                : rep.oos_equity.front();
    rep.oos_total_return_pct = first > 0.0
        ? (running_equity / first - 1.0) * 100.0 : 0.0;

    double is_sum = 0.0;
    for (const WalkForwardFold& f : rep.folds) is_sum += f.train_return_pct;
    rep.is_mean_return_pct = is_sum / static_cast<double>(rep.folds.size());
    rep.degradation_pct = rep.is_mean_return_pct - rep.oos_total_return_pct;
    rep.oos_hit_rate = static_cast<double>(profitable_folds) /
                       static_cast<double>(rep.folds.size());
    rep.profitable_oos = rep.oos_total_return_pct > 0.0;
    return rep;
}

} // namespace trading
} // namespace omniseed
