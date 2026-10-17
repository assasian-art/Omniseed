//  OmniSeed — agent/ensemble_vote.cpp  (T073b). Design: state/task_T073_brief.md
#include "omniseed/agent/ensemble_vote.h"

#include <algorithm>
#include <cstdio>
#include <map>
#include <vector>

namespace omniseed {
namespace agent {

namespace {

struct Tally {
    double weight = 0.0;
    int    count  = 0;
};

}  // namespace

EnsembleVerdict vote(const std::vector<Vote>& ballots, const EnsembleConfig& cfg) {
    EnsembleVerdict v;
    if (!cfg.enable) return v;  // default-off: pre-T073 path, no claim

    // --- 1. cast the surviving ballots -----------------------------------
    std::map<std::string, Tally> per_action;
    std::vector<double> cast_conf;
    double total = 0.0;
    for (const Vote& b : ballots) {
        if (!b.valid) continue;
        if (b.action.empty() || b.action == "abstain") continue;
        if (!(b.confidence >= cfg.min_vote_conf)) continue;  // NaN fails too
        if (b.confidence > 1.0) continue;                   // out-of-range: not a ballot
        Tally& t = per_action[b.action];
        t.weight += b.confidence;
        t.count += 1;
        cast_conf.push_back(b.confidence);
        total += b.confidence;
    }
    v.voters = static_cast<int>(cast_conf.size());

    if (v.voters == 0 || !(total > 0.0)) {  // no-votes / all below bar
        v.reason = v.voters == 0 ? "no-votes" : "below-bar";
        return v;
    }

    // --- 2. rank by weight (ties broken alphabetically for determinism) ---
    std::vector<std::pair<std::string, Tally>> ranked(per_action.begin(),
                                                      per_action.end());
    std::sort(ranked.begin(), ranked.end(),
              [](const std::pair<std::string, Tally>& a,
                 const std::pair<std::string, Tally>& b) {
                  if (a.second.weight != b.second.weight)
                      return a.second.weight > b.second.weight;
                  return a.first < b.first;
              });

    // --- 3. disagreement: top-2 too close to call -------------------------
    const double second_w = ranked.size() > 1 ? ranked[1].second.weight : 0.0;
    if (ranked.size() > 1 &&
        !(ranked[0].second.weight - second_w > cfg.tie_eps * total)) {
        v.disagreement = true;
        v.reason = "disagreement";
        return v;  // action stays "abstain"
    }

    // --- 4. majority share -------------------------------------------------
    const double share = ranked[0].second.weight / total;
    if (!(share > cfg.min_share)) {
        v.reason = "majority-fail";
        return v;
    }

    double mean_conf = 0.0;
    for (double c : cast_conf) mean_conf += c;
    mean_conf /= static_cast<double>(cast_conf.size());

    v.action = ranked[0].first;
    v.confidence = share * mean_conf;  // mixture score, NOT a probability
    if (v.confidence > 0.99) v.confidence = 0.99;
    if (!(v.confidence > 0.0)) v.confidence = 0.0;
    v.reason = v.voters >= 2 ? "majority" : "single";
    return v;
}

std::string EnsembleVerdict::to_json() const {
    char buf[192];
    std::snprintf(buf, sizeof(buf),
                  "{\"action\":\"%s\",\"confidence\":%.6f,\"disagreement\":%s,"
                  "\"voters\":%d,\"reason\":\"%s\"}",
                  action.c_str(), confidence, disagreement ? "true" : "false",
                  voters, reason.c_str());
    return std::string(buf);
}

}  // namespace agent
}  // namespace omniseed
