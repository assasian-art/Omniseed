// =============================================================================
//  OmniSeed — trading/router.cpp
//  C++ implementation of the regime-adaptive ensemble router.
//
//  Faithful port of tools/monster/router.py. The two behaviours that must not
//  drift, because they are the whole point of the module:
//
//    1. CONTINUOUS regime weighting. `sigmoid(k*(trend-0.5))` means the blend
//       slides between the trend engine and the range engine instead of
//       switching. A binary switch pays the spread twice on every regime flip,
//       and flips cluster exactly where the classifier is least certain.
//
//    2. THE VETO IS WEIGHT-GATED. `agreement` is a SHARE, so one lone active
//       signal scores agreement = 1.00 by construction. Without the
//       `veto_min_weight` floor, a single low-confidence OFI proxy would veto
//       every long in the book. That is the difference between a consensus and
//       a lone voice, and tests/test_router.cpp pins it.
// =============================================================================
#include "omniseed/trading/router.h"

#include <cmath>
#include <cstdarg>
#include <cstdio>

namespace omniseed {
namespace trading {

namespace {

std::string fmt(const char* f, ...) {
    char buf[256];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof(buf), f, ap);
    va_end(ap);
    return std::string(buf);
}

} // namespace

std::string EnsembleVerdict::detail() const {
    return fmt("conv=%+.3f agree=%.2f wt=%.2f n=%d", conviction, agreement, weight,
               static_cast<int>(active));
}

double sigmoid(double x) {
    // Written the oracle's way, in two branches, so that a large |x| cannot
    // overflow exp() and hand back inf/inf = NaN.
    if (x >= 0.0) {
        const double z = std::exp(-x);
        return 1.0 / (1.0 + z);
    }
    const double z = std::exp(x);
    return z / (1.0 + z);
}

double regime_weight(RegimeFit fit, double w_trend) {
    switch (fit) {
        case RegimeFit::Trend: return w_trend;
        case RegimeFit::Range: return 1.0 - w_trend;
        case RegimeFit::Both:  return kFitBothWeight;
    }
    return kFitBothWeight;
}

EnsembleVerdict route(const std::vector<StrategySignal>& signals,
                      const RegimeState* regime, const RouterConfig& cfg,
                      int64_t ts) {
    EnsembleVerdict v;
    v.ts = ts;

    double ts_score = 0.5;
    if (regime != nullptr && regime_finite(regime->trend_score)) ts_score = regime->trend_score;
    v.w_trend = sigmoid(cfg.k * (ts_score - 0.5));

    double total = 0.0;
    double signed_ = 0.0;
    for (const StrategySignal& sig : signals) {
        if (sig.confidence < cfg.min_confidence) continue;
        if (sig.direction == 0.0) continue;
        const double w = regime_weight(sig.regime_fit, v.w_trend) * sig.confidence;
        if (w <= 0.0) continue;
        total += w;
        signed_ += w * sig.direction;
        v.active += 1;
        v.signals.emplace_back(sig, w);
    }

    if (total <= 0.0) {
        v.conviction = 0.0;
        v.agreement = 0.0;
        v.weight = 0.0;
        return v;
    }

    v.weight = total;
    v.conviction = std::max(-1.0, std::min(1.0, signed_ / total));

    if (std::fabs(v.conviction) < 1e-12) {
        // No net direction means "agreement" is undefined, and 0.0 is the
        // conservative answer: it can never satisfy a >= veto threshold.
        v.agreement = 0.0;
    } else {
        const double sgn = v.conviction > 0.0 ? 1.0 : -1.0;
        double agree_w = 0.0;
        for (const auto& kv : v.signals)
            if (kv.first.direction * sgn > 0.0) agree_w += kv.second;
        v.agreement = agree_w / total;
    }

    v.reasons.reserve(v.signals.size());
    for (const auto& kv : v.signals)
        v.reasons.push_back(kv.first.name + ":" + kv.first.reason);
    return v;
}

bool should_veto_long(const EnsembleVerdict& v, const RouterConfig& cfg) {
    return v.active > 0 && v.agreement >= cfg.agreement_veto &&
           v.conviction <= -cfg.ensemble_veto && v.weight >= cfg.veto_min_weight;
}

EnsembleVerdict evaluate(const StrategySeries& s, size_t i,
                         const RegimeState* regime, const RouterConfig& cfg,
                         const std::vector<Snapshot>* snapshots,
                         const StrategyConfig& scfg) {
    const std::vector<StrategySignal> sigs =
        all_signals(s, i, regime, scfg, snapshots);
    return route(sigs, regime, cfg, s.ts()[i]);
}

std::vector<EnsembleVerdict> scan(const StrategySeries& s,
                                  const std::vector<RegimeState>& regimes,
                                  const RouterConfig& cfg,
                                  const StrategyConfig& scfg) {
    std::vector<EnsembleVerdict> out;
    out.reserve(s.n());
    for (size_t i = 0; i < s.n(); ++i) {
        const RegimeState* reg = i < regimes.size() ? &regimes[i] : nullptr;
        out.push_back(evaluate(s, i, reg, cfg, nullptr, scfg));
    }
    return out;
}

} // namespace trading
} // namespace omniseed
