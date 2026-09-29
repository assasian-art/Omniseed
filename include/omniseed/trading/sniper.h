// =============================================================================
//  OmniSeed — trading/sniper.h
//  C++ port of tools/monster/sniper_engine.py
//  The Sniper Entry Engine: a weighted, multi-layer confidence score.
//
//      S = w_m*M + w_t*T + w_r*R + w_c*C          (all layers in [0,1])
//
//    M  microstructure  — volume-anomaly z-score (+ order-book imbalance when
//                         the provider publishes depth)
//    T  technical       — up to 6 INDEPENDENT bullish factors; >=3 required
//    R  regime          — higher-timeframe regime; counter-trend is vetoed
//                         unless a verified mean-reversion setup is present
//    C  cross-asset     — lead-lag confirmation / invalidation
//
//  An entry is PROPOSED only when no veto fires AND S >= min_confidence.
//
//  THE 0.85 BAR IS REACHABLE, AND THAT IS A FIX, NOT A LENIENCY. Scoring the
//  technical layer as votes/6 caps the reachable total at ~0.81 and makes a 0.85
//  gate mathematically unreachable — a gate that can never open is a bug, not
//  discipline. Meeting the mandated minimum confluence earns 0.90 of the layer;
//  each EXTRA agreeing factor adds 0.10 up to a cap of 1.00.
//
//  HONEST SCOPE: the score measures how much of the evidence agrees. It is not a
//  probability of profit. See docs/MONSTER_DESIGN.md.
// =============================================================================
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "omniseed/trading/regime_authority.h"
#include "omniseed/trading/regime_engine.h"
#include "omniseed/trading/router.h"
#include "omniseed/trading/strategy_zoo.h"
#include "omniseed/trading/trading_engine.h"

namespace omniseed {
namespace trading {

// The label -> layer-score map is the mandate's: a with-trend entry gets the
// full 1.0, a counter-trend one gets 0.0 and is vetoed below. Only the SOURCE of
// the label changed (multi-axis engine instead of SMA200 slope), never the map.
double regime_score_for_label(const std::string& label);
bool   regime_label_known(const std::string& label);

// Mirrors SniperConfig in tools/monster/sniper_engine.py field-for-field.
struct SniperConfig {
    // --- layer weights (sum to ~1.0) --------------------------------------
    double w_micro = 0.25;
    double w_tech = 0.35;
    double w_regime = 0.25;
    double w_cross = 0.15;
    // --- decision bar -----------------------------------------------------
    double  min_confidence = 0.85;
    int32_t min_factors = 3;        // independent technical votes required
    int32_t min_factors_event = 1;  // relaxed when an event is matched
    // --- microstructure ---------------------------------------------------
    int32_t vol_window = 20;
    // --- technical --------------------------------------------------------
    int32_t sma_fast = 20;
    int32_t sma_slow = 50;
    int32_t rsi_period = 14;
    double  rsi_oversold = 30.0;
    int32_t bb_period = 20;
    double  bb_sigma = 2.0;
    int32_t vwap_period = 20;
    int32_t swing_left = 2;
    int32_t swing_right = 2;
    std::vector<double> fib_ratios = {0.382, 0.5, 0.618};
    double  fib_tol = 0.015;        // +/- 1.5% of price around a level
    int32_t div_max_gap = 60;
    // --- regime -----------------------------------------------------------
    //
    // "advanced" (default) — the multi-axis ensemble in regime_engine.h:
    //     variance ratio, Hurst, efficiency ratio, ADX, choppiness, R^2,
    //     rho(1) and MA-ribbon consistency, with Yang-Zhang volatility and an
    //     OU half-life, plus hysteresis.
    // "legacy" — the original single-statistic SMA200-slope rule, kept so the
    //     behaviour can be reproduced and A/B compared.
    std::string regime_mode = "advanced";
    int32_t sma_regime = 200;
    int32_t regime_slope_n = 20;
    double  regime_slope_s = 0.01;  // 1% SMA200 move over n bars
    int32_t atr_period = 14;
    double  atr_hi = 0.05;          // 5% ATR/close = high-volatility regime
    // --- strategy ensemble (opt-in; the advanced regime engine is required) --
    //
    // The ensemble can only ever BLOCK, never promote. It never raises S: an
    // ensemble that can talk you into a trade can talk you into a bad one.
    bool    ensemble = false;
    double  ensemble_veto = 0.35;
    double  ensemble_agreement = 0.60;
    double  ensemble_min_weight = 0.50;
    // --- cross-asset ------------------------------------------------------
    int32_t invalidate_veto = 2;
};

// Everything the bar series cannot know: depth, peers, news.
struct EvalContext {
    bool   has_obi = false;         // order-book imbalance present?
    double obi = 0.0;               // in [-1, 1]
    int32_t cross_confirm = 0;
    int32_t cross_invalidate = 0;
    int32_t cross_active = 0;
    bool    event_driven = false;
};

struct SniperVerdict {
    int64_t ts = 0;
    double  score = 0.0;
    double  micro = 0.0;
    double  tech = 0.0;
    double  regime_score = 0.0;
    double  cross = 0.0;
    int32_t votes = 0;
    std::string regime = "range";
    bool    veto = false;
    std::string veto_reason;
    std::vector<std::string> factors;
    // --- regime authority (§46) -------------------------------------------
    // `regime_gated` is true iff the regime label was AUTHORISED to gate (i.e.
    // it came from the rule engine). It is false when the label was refused, in
    // which case `regime` was neutralised to "range" and no regime veto could
    // fire. A learned `trading.regime` label can never set this true.
    bool        regime_gated = false;
    std::string regime_source = "unset";
    // --- ensemble / regime detail (never part of S) -----------------------
    double trend_score = kRegimeNaN;
    double half_life = kRegimeNaN;
    double conviction = 0.0;
    double agreement = 0.0;

    bool propose(const SniperConfig& cfg) const {
        return !veto && score >= cfg.min_confidence;
    }
    // Compact, CSV-safe rationale (no commas).
    std::string detail() const;
};

// All feature series for one bar history, computed once.
class Prepared {
public:
    Prepared(const std::vector<Bar>& bars, const SniperConfig& cfg);
    explicit Prepared(const std::vector<Bar>& bars);

    const SniperConfig&              cfg() const { return cfg_; }
    const std::vector<Bar>&          bars() const { return bars_; }
    size_t                           n() const { return n_; }
    const std::vector<int64_t>&      ts() const { return ts_; }
    const std::vector<double>&       close() const { return close_; }
    const std::vector<double>&       high() const { return high_; }
    const std::vector<double>&       low() const { return low_; }
    const std::vector<double>&       vol() const { return vol_; }
    const std::vector<double>&       sma_fast() const { return sma_fast_; }
    const std::vector<double>&       sma_slow() const { return sma_slow_; }
    const std::vector<double>&       sma_regime() const { return sma_regime_; }
    const std::vector<double>&       rsi() const { return rsi_; }
    const std::vector<double>&       macd_hist() const { return macd_hist_; }
    const std::vector<double>&       bb_mid() const { return bb_mid_; }
    const std::vector<double>&       bb_up() const { return bb_up_; }
    const std::vector<double>&       bb_lo() const { return bb_lo_; }
    const std::vector<double>&       atr() const { return atr_; }
    const std::vector<double>&       atr_pct() const { return atr_pct_; }
    const std::vector<double>&       vwap() const { return vwap_; }
    const std::vector<double>&       vol_z() const { return vol_z_; }
    const std::vector<features::Swing>& swings_low() const { return swings_low_; }
    const std::vector<features::Swing>& swings_high() const { return swings_high_; }

    // Empty unless regime_mode == "advanced".
    bool has_regimes() const { return !regimes_.empty(); }
    const std::vector<RegimeState>& regimes() const { return regimes_; }
    // Null unless the ensemble was enabled (which requires the advanced engine).
    const StrategySeries* series() const { return series_ ? series_.get() : nullptr; }
    bool has_ensemble() const { return !ensemble_.empty(); }
    const std::vector<EnsembleVerdict>& ensemble() const { return ensemble_; }

private:
    SniperConfig         cfg_;
    std::vector<Bar>     bars_;
    size_t               n_ = 0;
    std::vector<int64_t> ts_;
    std::vector<double>  close_, high_, low_, vol_;
    std::vector<double>  sma_fast_, sma_slow_, sma_regime_;
    std::vector<double>  rsi_, macd_hist_;
    std::vector<double>  bb_mid_, bb_up_, bb_lo_;
    std::vector<double>  atr_, atr_pct_, vwap_, vol_z_;
    std::vector<features::Swing> swings_low_, swings_high_;
    std::vector<RegimeState>     regimes_;
    std::unique_ptr<StrategySeries> series_;
    std::vector<EnsembleVerdict> ensemble_;
};

// Layer computations, exposed so each can be pinned independently.
namespace sniper_detail {

// -> (score, votes, factor names)
void technical(const Prepared& p, size_t i, const SniperConfig& cfg, double& score,
               int32_t& votes, std::vector<std::string>& names);
double micro(const Prepared& p, size_t i, const EvalContext* ctx,
             const SniperConfig& cfg);
// -> (label, layer score)
void regime_legacy(const Prepared& p, size_t i, const SniperConfig& cfg,
                   std::string& label, double& score);
void regime(const Prepared& p, size_t i, const SniperConfig& cfg,
            std::string& label, double& score);
double cross(const EvalContext* ctx, const SniperConfig& cfg);
bool mean_reversion_confirmed(const Prepared& p, size_t i, const SniperConfig& cfg);

} // namespace sniper_detail

// Score bar `i` using only data at indices <= i.
SniperVerdict evaluate(const Prepared& p, size_t i, const EvalContext* ctx,
                       const SniperConfig& cfg);

// Evaluate every bar -> one verdict per bar (same length as bars).
std::vector<SniperVerdict> scan(const std::vector<Bar>& bars,
                                const std::map<int64_t, EvalContext>& ctx_by_ts,
                                const SniperConfig& cfg);

} // namespace trading
} // namespace omniseed
