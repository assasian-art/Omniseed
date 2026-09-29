// =============================================================================
//  OmniSeed — trading/regime_authority.cpp
//  See the header for the WHY. This file is the two mints and the one predicate.
// =============================================================================
#include "omniseed/trading/regime_authority.h"

#include <string>

namespace omniseed {
namespace trading {

const char* regime_source_name(RegimeSource s) {
    switch (s) {
        case RegimeSource::RuleEngine:  return "rule-engine";
        case RegimeSource::LegacyRule:  return "legacy-rule";
        case RegimeSource::LearnedHead: return "learned-head";
        case RegimeSource::External:    return "external";
        case RegimeSource::Unset:       break;
    }
    return "unset";
}

GatingRegime GatingRegime::mint(RegimeSource src, const std::string& label) {
    GatingRegime g;
    g.source_ = src;
    g.label_ = label;
    return g;
}

GatingRegime GatingRegime::from_engine(const RegimeState& s) {
    // The engine_minted flag is the whole reason this factory is trustworthy: a
    // RegimeState assembled by hand never carries it, so it is downgraded to
    // External (non-gating) rather than accepted.
    if (!s.engine_minted) return mint(RegimeSource::External, s.label);
    return mint(RegimeSource::RuleEngine, s.label);
}

GatingRegime GatingRegime::from_legacy_rule(const std::string& label) {
    return mint(RegimeSource::LegacyRule, label);
}

GatingRegime GatingRegime::from_learned(const std::string& label) {
    return mint(RegimeSource::LearnedHead, label);
}

GatingRegime GatingRegime::from_external(const std::string& label) {
    return mint(RegimeSource::External, label);
}

const char* GatingRegime::refusal_reason() const {
    switch (source_) {
        case RegimeSource::RuleEngine:
        case RegimeSource::LegacyRule:
            return "";
        case RegimeSource::LearnedHead:
            return "learned trading.regime head is DEMOTED (0.810x a constant, "
                   "EDGE_RESEARCH 45) and may not gate";
        case RegimeSource::External:
            return "label is not rule-engine output (external, or an unminted "
                   "RegimeState) and may not gate";
        case RegimeSource::Unset:
            break;
    }
    return "no regime was minted";
}

std::string GatingRegime::detail() const {
    std::string out = "regime=";
    out += label_;
    out += " source=";
    out += regime_source_name(source_);
    out += can_gate() ? " gate=yes" : " gate=no";
    return out;
}

std::string regime_veto(const GatingRegime& gate, bool mean_reversion_confirmed) {
    // THE AUTHORITY CHECK. A token that may not gate returns no veto — it cannot
    // block a trade, no matter what its label says. This is the single line that
    // makes "the learned head cannot gate" structural rather than documentary.
    if (!gate.authorize()) return "";

    // A confirmed mean-reversion setup is allowed to trade against a hostile
    // regime: oversold below the lower band is the two-independent-signals case.
    if (mean_reversion_confirmed) return "";

    const std::string& l = gate.label();
    if (l == "trend_down") return "counter-regime";
    if (l == "high_vol") return "high-vol-needs-mean-reversion";
    return "";
}

} // namespace trading
} // namespace omniseed
