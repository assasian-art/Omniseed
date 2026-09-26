// =============================================================================
//  OmniSeed — trading/risk_gate.cpp
//  Daily+weekly kill-switches and the code-enforced autonomy ladder.
// =============================================================================
#include "omniseed/trading/risk_gate.h"

#include <cstdio>

namespace omniseed {
namespace trading {

// ===========================================================================
// RiskGovernorSet
// ===========================================================================
RiskGovernorSet::RiskGovernorSet(const Config& cfg)
    : cfg_(cfg),
      daily_(DailyRiskGovernor::Config{
          cfg.max_daily_loss_pct,
          cfg.day_seconds > 0 ? cfg.day_seconds : 86400}),
      weekly_(DailyRiskGovernor::Config{
          cfg.max_weekly_loss_pct,
          7 * (cfg.day_seconds > 0 ? cfg.day_seconds : 86400)}) {}

bool RiskGovernorSet::allow(int64_t ts, double equity, std::string& why_not) {
    std::string why_d, why_w;
    const bool ok_d = daily_.allow(ts, equity, why_d);
    const bool ok_w = weekly_.allow(ts, equity, why_w);
    if (!ok_d) { why_not = "daily: " + why_d; return false; }
    if (!ok_w) { why_not = "weekly: " + why_w; return false; }
    return true;
}

// ===========================================================================
// AutonomyGate
// ===========================================================================
AutonomyLevel AutonomyGate::resolve(const std::string& state_text,
                                    std::string& why) {
    // Line-by-line, exact match after trimming surrounding whitespace. A
    // near-miss (a TODO, a quoted example, different case) must NOT unlock
    // anything — the owner has to write the line, alone on its own line.
    bool l1 = false, l2 = false;
    size_t i = 0;
    for (;;) {
        size_t j = state_text.find('\n', i);
        const bool last = (j == std::string::npos);
        if (last) j = state_text.size();
        std::string line = state_text.substr(i, j - i);
        const size_t b = line.find_first_not_of(" \t\r");
        const size_t e = line.find_last_not_of(" \t\r");
        line = (b == std::string::npos) ? std::string()
                                        : line.substr(b, e - b + 1);
        if (line == "UNLOCK L1 MICRO-LIVE") l1 = true;
        if (line == "UNLOCK L2 SCALE-UP")   l2 = true;
        if (last) break;
        i = j + 1;
    }
    if (l2) {
        why = "owner unlock line 'UNLOCK L2 SCALE-UP' present";
        return AutonomyLevel::Scaled;
    }
    if (l1) {
        why = "owner unlock line 'UNLOCK L1 MICRO-LIVE' present";
        return AutonomyLevel::MicroLive;
    }
    why = "no owner unlock line -> paper-only";
    return AutonomyLevel::PaperOnly;
}

bool AutonomyGate::allows_live(AutonomyLevel level, double notional_usd,
                               double risk_pct, int64_t days_at_level,
                               const AutonomyConfig& cfg, std::string& why_not) {
    char buf[192];
    if (level == AutonomyLevel::PaperOnly) {
        why_not = "L0 paper-only: live orders are refused";
        return false;
    }
    if (notional_usd <= 0.0 || risk_pct <= 0.0) {
        why_not = "degenerate order: notional and risk must be > 0";
        return false;
    }

    if (level == AutonomyLevel::MicroLive) {
        if (days_at_level < cfg.l1_min_paper_days) {
            std::snprintf(buf, sizeof(buf),
                          "L1 needs %lld paper days, have %lld",
                          static_cast<long long>(cfg.l1_min_paper_days),
                          static_cast<long long>(days_at_level));
            why_not = buf;
            return false;
        }
        if (notional_usd > cfg.l1_max_total_usd) {
            std::snprintf(buf, sizeof(buf), "L1 notional %.2f > cap %.2f",
                          notional_usd, cfg.l1_max_total_usd);
            why_not = buf;
            return false;
        }
        if (risk_pct > cfg.l1_max_risk_pct) {
            std::snprintf(buf, sizeof(buf), "L1 risk %.4f > cap %.4f",
                          risk_pct, cfg.l1_max_risk_pct);
            why_not = buf;
            return false;
        }
        why_not = "L1 micro-live: within caps";
        return true;
    }

    // Scaled (L2)
    if (days_at_level < cfg.l2_min_live_days) {
        std::snprintf(buf, sizeof(buf), "L2 needs %lld live days, have %lld",
                      static_cast<long long>(cfg.l2_min_live_days),
                      static_cast<long long>(days_at_level));
        why_not = buf;
        return false;
    }
    why_not = "L2 scaled: within caps";
    return true;
}

} // namespace trading
} // namespace omniseed
