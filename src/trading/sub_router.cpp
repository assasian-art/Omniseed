//  OmniSeed — trading/sub_router.cpp  (T072b). Design: state/task_T072_brief.md
#include "omniseed/trading/sub_router.h"

namespace omniseed {
namespace trading {

const char* slot_name(SpecialistSlot s) {
    switch (s) {
        case SpecialistSlot::Momentum:      return "momentum";
        case SpecialistSlot::MeanReversion: return "mean_reversion";
        case SpecialistSlot::Abstain:       break;
    }
    return "abstain";
}

SpecialistSlot route(const std::string& regime_label,
                     double conviction,
                     bool mean_reversion_verified,
                     const SubRouterConfig& cfg) {
    // Disabled => the sub-router makes no claim at all (level-1 path only).
    if (!cfg.enable) return SpecialistSlot::Abstain;

    // `!(x >= bar)` is deliberate: a NaN conviction can never clear the bar.
    if (!(conviction >= cfg.min_conviction)) return SpecialistSlot::Abstain;

    if (regime_label == "trend_up") return SpecialistSlot::Momentum;

    // Range/chop: the one family whose edge is the absence of trend.
    if (regime_label == "range") return SpecialistSlot::MeanReversion;

    // High volatility: mean-reversion ONLY with the sniper's verified setup;
    // an unverified high-vol bar is the classic fakeout, so it abstains.
    if (regime_label == "high_vol")
        return mean_reversion_verified ? SpecialistSlot::MeanReversion
                                       : SpecialistSlot::Abstain;

    // trend_down and every unknown label: no specialist pick. Counter-trend
    // entries are the regime veto's business (sniper.h), not a slot here.
    return SpecialistSlot::Abstain;
}

}  // namespace trading
}  // namespace omniseed
