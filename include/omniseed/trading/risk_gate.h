// =============================================================================
//  OmniSeed — trading/risk_gate.h
//  M4/M6 — the composite risk gate and the autonomy ladder.
//
//    * RiskGovernorSet — the daily AND weekly drawdown kill-switches,
//                        composed over the tested DailyRiskGovernor.
//    * AutonomyGate    — the live-money ladder (M6). L0 paper-only is the
//                        default; L1/L2 require an EXPLICIT owner unlock line
//                        in PROJECT_STATE.md *and* pass per-order caps and
//                        duration minimums. Every check is code, not docs.
// =============================================================================
#pragma once

#include "omniseed/trading/trading_engine.h"

#include <cstdint>
#include <string>

namespace omniseed {
namespace trading {

// ===========================================================================
// RiskGovernorSet — daily + weekly drawdown kill-switch
// ===========================================================================
// Two independent windows over the same equity stream. Entries are refused
// while EITHER window is halted; each re-arms on its own boundary. The weekly
// window is a 7-day bucket, so a bad day cannot trip it but a bad week can.
class RiskGovernorSet {
public:
    struct Config {
        double  max_daily_loss_pct  = 0.03;
        double  max_weekly_loss_pct = 0.06;
        int64_t day_seconds         = 86400;
    };

    RiskGovernorSet() : RiskGovernorSet(Config{}) {}
    explicit RiskGovernorSet(const Config& cfg);

    // Feed once per bar, in chronological order. False when either window is
    // halted; `why_not` names which one.
    bool allow(int64_t ts, double equity, std::string& why_not);

    bool    halted() const { return daily_.halted() || weekly_.halted(); }
    int64_t daily_trips() const { return daily_.trips(); }
    int64_t weekly_trips() const { return weekly_.trips(); }
    int64_t total_trips() const { return daily_.trips() + weekly_.trips(); }
    void    reset() { daily_.reset(); weekly_.reset(); }

private:
    Config cfg_;
    DailyRiskGovernor daily_;
    DailyRiskGovernor weekly_;
};

// ===========================================================================
// Autonomy ladder (M6)
// ===========================================================================
enum class AutonomyLevel : int8_t { PaperOnly = 0, MicroLive, Scaled };

inline const char* to_string(AutonomyLevel lvl) {
    switch (lvl) {
        case AutonomyLevel::MicroLive: return "L1-micro-live";
        case AutonomyLevel::Scaled:    return "L2-scaled";
        default:                       return "L0-paper-only";
    }
}

struct AutonomyConfig {
    double  l1_max_total_usd   = 100.0;   // total live exposure ceiling
    double  l1_max_risk_pct    = 0.01;    // per-trade risk allowed at L1
    int64_t l1_min_paper_days  = 30;      // paper track record required
    int64_t l2_min_live_days   = 90;      // live track record required
};

class AutonomyGate {
public:
    // Scans `state_text` (the contents of PROJECT_STATE.md) for an explicit
    // owner unlock line. Recognised, exact, case-sensitive lines:
    //     UNLOCK L1 MICRO-LIVE
    //     UNLOCK L2 SCALE-UP
    // Returns the highest level unlocked; no line -> PaperOnly. `why` always
    // explains the decision.
    static AutonomyLevel resolve(const std::string& state_text, std::string& why);

    // True when a live order is permitted. Enforces: the level (PaperOnly
    // never trades live), the duration minimum for that level, and the L1
    // per-order caps. `days_at_level` is how long the account has been at it.
    static bool allows_live(AutonomyLevel level, double notional_usd,
                            double risk_pct, int64_t days_at_level,
                            const AutonomyConfig& cfg, std::string& why_not);
};

} // namespace trading
} // namespace omniseed
