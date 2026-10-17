// =============================================================================
//  OmniSeed — trading/sub_router.h   (T072b: LEVEL-2 routing)
//
//  The level-1 HeadRouter picks the DOMAIN (include/omniseed/router.h).
//  This picks the SPECIALIST ACTION SPACE inside the trading domain:
//
//      momentum | mean_reversion | abstain
//
//  Pure, deterministic, no look-ahead: it reads ONE bar's regime verdict, a
//  conviction scalar, and the verified mean-reversion flag, and returns a
//  slot. It NEVER widens confidence, never bypasses the risk governor, and
//  `Abstain` means "no specialist claim" — the base path continues unchanged.
//
//  DEFAULT OFF (`enable=false`): pre-T072 behaviour is bit-identical, the
//  same pattern as the M4 `w_vol` guard. On-arm tests flip it explicitly.
//
//  HONEST SCOPE: a slot narrows which strategy families are eligible; it is
//  not a forecast and not a probability. See state/task_T072_brief.md.
// =============================================================================
#pragma once

#include <string>

namespace omniseed {
namespace trading {

struct SubRouterConfig {
    bool   enable         = false;  // default OFF => parity with pre-T072
    double min_conviction = 0.60;   // below this the regime itself is noise
};

enum class SpecialistSlot { Momentum, MeanReversion, Abstain };

const char* slot_name(SpecialistSlot s);  // "momentum" | "mean_reversion" | "abstain"

// Fail-closed on every branch: unknown label, conviction below the bar (NaN
// included — the compare is written so NaN cannot pass), or `high_vol`
// without a VERIFIED mean-reversion setup all return Abstain.
SpecialistSlot route(const std::string& regime_label,
                     double conviction,
                     bool mean_reversion_verified,
                     const SubRouterConfig& cfg);

}  // namespace trading
}  // namespace omniseed
