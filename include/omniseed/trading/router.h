// =============================================================================
//  OmniSeed — trading/router.h
//  C++ port of tools/monster/router.py
//  Regime-adaptive allocation across the strategy zoo.
//
//  THE ONE DESIGN DECISION THAT MATTERS: continuous sizing, not a binary switch.
//
//  The obvious implementation is "if trending use momentum else use mean
//  reversion". That is also the expensive one: every flip pays the full spread
//  on the way out and back, and regime transitions are exactly when a binary
//  classifier is least sure. So instead
//
//      w_trend = sigmoid(k * (trend_score - 0.5))
//      w_range = 1 - w_trend
//
//  and each strategy is weighted by its regime fit times its own confidence.
//  Near trend_score = 0.5 the router holds both engines at ~half size and
//  degrades gracefully through the ambiguity instead of whipsawing.
//
//  OUTPUT
//      conviction  net directional view in [-1, 1] (signed, weighted)
//      agreement   share of active weight that agrees with the sign of
//                  conviction, in [0, 1]. Low agreement = the zoo is split.
//
//  THE ROUTER CAN ONLY EVER BLOCK, NEVER PROMOTE. See should_veto_long(): a
//  weighty, high-agreement opposing ensemble vetoes the entry; it never raises
//  the entry score. That asymmetry is deliberate — an ensemble that can talk you
//  INTO a trade is an ensemble that can talk you into a bad one.
// =============================================================================
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "omniseed/trading/regime_engine.h"
#include "omniseed/trading/strategy_zoo.h"

namespace omniseed {
namespace trading {

// How much a regime-agnostic ("both") strategy is worth. Always eligible, never
// the dominant voice.
constexpr double kFitBothWeight = 1.0;

// Mirrors RouterConfig in tools/monster/router.py field-for-field.
struct RouterConfig {
    double k = 6.0;                 // sigmoid steepness on trend_score
    double ensemble_veto = 0.35;    // conviction at/below which a long is opposed
    double agreement_veto = 0.60;   // agreement required for that veto to bind
    double veto_min_weight = 0.50;  // total active weight required for that veto
    double min_confidence = 0.05;   // ignore signals quieter than this
};

struct EnsembleVerdict {
    int64_t ts = 0;
    double  conviction = 0.0;
    double  agreement = 0.0;
    double  weight = 0.0;
    double  w_trend = 0.5;
    int32_t active = 0;
    // The signals that actually made it into the blend, with their weights.
    std::vector<std::pair<StrategySignal, double>> signals;
    std::vector<std::string> reasons;

    bool opposed_long() const { return conviction <= 0.0; }

    // Confidence-scaled exposure multiplier in [0.75, 1.25].
    //
    // Only ever applied when the ensemble AGREES (conviction > 0); an opposed
    // ensemble does not get to shrink the size, it gets to veto.
    double size_factor() const {
        if (conviction <= 0.0) return 1.0;
        return 0.75 + 0.50 * features::clampd(agreement, 0.0, 1.0);
    }

    std::string detail() const;
};

double sigmoid(double x);

// How much a strategy of this regime fit is worth right now.
double regime_weight(RegimeFit fit, double w_trend);

// Blend strategy signals into a single ensemble verdict.
EnsembleVerdict route(const std::vector<StrategySignal>& signals,
                      const RegimeState* regime, const RouterConfig& cfg,
                      int64_t ts);

// True when a WEIGHTY, high-agreement ensemble actively opposes a long.
//
// The weight floor is not decoration. `agreement` is a SHARE, so a single active
// strategy always scores 1.00 — one lonely, low-confidence OFI proxy would
// otherwise veto every long in the book. Requiring real weight behind the
// disagreement is the difference between a consensus and a lone voice.
bool should_veto_long(const EnsembleVerdict& v, const RouterConfig& cfg);

// Convenience: run the whole zoo at bar `i` and route it.
EnsembleVerdict evaluate(const StrategySeries& s, size_t i,
                         const RegimeState* regime, const RouterConfig& cfg,
                         const std::vector<Snapshot>* snapshots,
                         const StrategyConfig& scfg);

// Ensemble verdict for every bar. `regimes` must align with the bar series;
// a shorter (or empty) vector is tolerated and treated as "no regime".
std::vector<EnsembleVerdict> scan(const StrategySeries& s,
                                  const std::vector<RegimeState>& regimes,
                                  const RouterConfig& cfg,
                                  const StrategyConfig& scfg);

} // namespace trading
} // namespace omniseed
