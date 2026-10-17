// =============================================================================
//  OmniSeed — agent/ensemble_vote.h   (T073: CROSS-HEAD ENSEMBLE VOTING)
//
//  The decision / classify / score heads each emit a number in their own
//  shape; nothing let them vote together. This is the shared-ballot combiner:
//  every head is reduced to ONE ballot over a common action space, weights
//  are confidences, and every failure mode lands on `abstain`.
//
//  DEFAULT OFF (`enable=false`) — pre-T073 behaviour untouched, the same
//  pattern as M4 `w_vol` and T072's sub-router.
//
//  HONEST SCOPE: `confidence` is winner_share x mean(conf) — a mixture score,
//  NOT a probability of being right. Unfitted heads must cast `valid=false`;
//  a fake vote is worse than a missing one.
// =============================================================================
#pragma once

#include <string>
#include <vector>

namespace omniseed {
namespace agent {

struct Vote {
    std::string action;  // "" or "abstain" => no ballot
    double      confidence = 0.0;  // [0,1]
    bool        valid = true;      // false => head casts nothing (unfitted/stale)
};

struct EnsembleConfig {
    bool   enable       = false;  // default OFF => parity with pre-T073
    double min_vote_conf = 0.50;  // ballots below this are not cast
    double min_share     = 0.50;  // winner must EXCEED this share of weight
    double tie_eps       = 0.10;  // top-2 within eps*total => disagreement
};

struct EnsembleVerdict {
    std::string action = "abstain";
    double      confidence = 0.0;
    bool        disagreement = false;
    int         voters = 0;        // ballots actually cast
    std::string reason = "no-votes";  // no-votes|below-bar|disagreement|
                                      // majority-fail|majority|single
    std::string to_json() const;
};

// Pure + deterministic. Fail-closed at every step (see task_T073_brief.md).
EnsembleVerdict vote(const std::vector<Vote>& ballots, const EnsembleConfig& cfg);

}  // namespace agent
}  // namespace omniseed
