// =============================================================================
//  OmniSeed — src/domain_decision.cpp
//
//  The multi-domain decision head. See domain_decision.h for the routing ladder
//  and the honesty note about untrained (seeded) projections.
//
//  The softmax subtracts the max before exp() for the same reason the
//  classification head does: without it a large logit overflows, every exp()
//  becomes inf, and inf/inf is NaN — a distribution of NaNs that argmax() would
//  cheerfully turn into "action 0", which is a real action.
// =============================================================================
#include "omniseed/domain_decision.h"
#include "omniseed/core/platform.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <string>

namespace omniseed {

// ---------------------------------------------------------------------------
// DomainDecision
// ---------------------------------------------------------------------------
std::string DomainDecision::to_json() const {
    std::string out = "{\"domain\":\"";
    out += heads_detail::json_escape(domain);
    out += "\",\"action\":\"";
    out += heads_detail::json_escape(action);
    out += "\",\"confidence\":";
    heads_detail::append_float(out, confidence);
    out += ",\"margin\":";
    heads_detail::append_float(out, margin);
    out += ",\"abstain\":";
    out += abstain ? "true" : "false";
    out += ",\"routing\":\"";
    out += heads_detail::json_escape(routing);
    out += "\",\"fast_path\":";
    out += fast_path ? "true" : "false";
    out += ",\"matvecs\":";
    out += std::to_string(matvecs);
    out += '}';
    return out;
}

DomainDecision from_decision_result(const DecisionResult& d) {
    DomainDecision o;
    o.domain = "trading";
    // DecisionHead's names are UPPERCASE for the trading JSON contract; every
    // label set in this tree is lower-case. Normalise here so a consumer never
    // has to know which head produced the decision.
    std::string a = decision_action_name(d.action_type);
    for (char& c : a) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    o.action     = a;
    o.confidence = d.confidence_score;
    o.margin     = d.margin;
    o.routing    = d.routing;
    o.fast_path  = d.fast_path;
    o.matvecs    = d.matvecs;
    o.us         = d.ms * 1000.0;   // DecisionResult reports ms
    o.abstain    = (o.routing == "abstain" || o.routing == "error");
    return o;
}

// ---------------------------------------------------------------------------
// Default domains
// ---------------------------------------------------------------------------
int32_t add_default_domains(DomainDecisionHead& head) {
    int32_t added = 0;
    // Trading mirrors DecisionHead's action space exactly (lower-cased), so the
    // two cannot drift into disagreeing about what actions exist.
    if (head.add_domain(Domain::Trading,
                        {"abstain", "hold", "buy", "sell", "close", "hedge", "explain"},
                        "abstain", "explain") >= 0) ++added;
    if (head.add_domain(Domain::Language,
                        {"abstain", "answer", "clarify", "search", "execute", "explain"},
                        "abstain", "explain") >= 0) ++added;
    if (head.add_domain(Domain::Vision,
                        {"abstain", "describe", "flag_anomaly", "explain"},
                        "abstain", "explain") >= 0) ++added;
    if (head.add_domain(Domain::Audio,
                        {"abstain", "transcribe", "respond", "explain"},
                        "abstain", "explain") >= 0) ++added;
    if (head.add_domain(Domain::General,
                        {"abstain", "answer", "route", "explain"},
                        "abstain", "explain") >= 0) ++added;
    return added;
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------
DomainDecisionHead::DomainDecisionHead(const RwkvConfig& cfg, uint32_t seed) {
    init(cfg, seed);
}

bool DomainDecisionHead::init(const RwkvConfig& cfg, uint32_t seed) {
    return init(cfg.n_embd, seed);
}

bool DomainDecisionHead::init(int32_t n_embd, uint32_t seed) {
    ready_ = false;
    error_.clear();
    slots_.clear();
    proj_.clear();
    bias_.clear();
    fitted_.clear();
    total_actions_ = 0;
    fitted_rows_   = 0;
    scratch_.clear();

    if (n_embd <= 0) {
        error_ = "DomainDecisionHead::init: n_embd must be > 0";
        provenance_ = "error: " + error_;
        return false;
    }
    E_ = n_embd;
    seed_ = seed;
    ready_ = true;
    provenance_ = "seeded placeholder (UNTRAINED) — per-row splitmix64 seed " +
                  std::to_string(seed) + ", E=" + std::to_string(E_) +
                  ", 0 domains";
    return true;
}

void DomainDecisionHead::seed_weights(uint32_t seed) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(E_));
    for (const DomainSlot& s : slots_) {
        for (size_t ai = 0; ai < s.actions.size(); ++ai) {
            uint64_t h = 0xCBF29CE484222325ull ^ static_cast<uint64_t>(seed);
            const std::string dn = domain_name(s.domain);
            for (const char c : dn) { h ^= static_cast<unsigned char>(c); h *= 0x100000001B3ull; }
            h ^= 0x2F; h *= 0x100000001B3ull;
            for (const char c : s.actions[ai]) { h ^= static_cast<unsigned char>(c); h *= 0x100000001B3ull; }
            const size_t row = static_cast<size_t>(s.offset) + ai;
            for (int32_t e = 0; e < E_; ++e)
                proj_[row * static_cast<size_t>(E_) + static_cast<size_t>(e)] =
                    heads_detail::next_unit(h) * scale;
            bias_[row] = heads_detail::next_unit(h) * 0.1f;
        }
    }
}

// ---------------------------------------------------------------------------
// Domains
// ---------------------------------------------------------------------------
int32_t DomainDecisionHead::add_domain(Domain d, const std::vector<std::string>& actions,
                                       const std::string& abstain_action,
                                       const std::string& explain_action) {
    if (!ready_) return -1;
    if (actions.empty()) return -1;
    if (find_domain(d) >= 0) return -1;

    DomainSlot s;
    s.domain = d;
    s.actions = actions;
    s.offset = total_actions_;

    auto find_action = [&actions](const std::string& want) -> int32_t {
        if (want.empty()) return -1;
        for (size_t i = 0; i < actions.size(); ++i)
            if (actions[i] == want) return static_cast<int32_t>(i);
        return -1;
    };
    // An explicitly named escape hatch that is not in the list is a caller bug,
    // but refusing the whole domain would be worse than ignoring it: the domain
    // still works, it just escalates instead of abstaining. Fall back to the
    // conventional name.
    s.abstain_index = find_action(abstain_action);
    if (s.abstain_index < 0) s.abstain_index = find_action("abstain");
    s.explain_index = find_action(explain_action);
    if (s.explain_index < 0) s.explain_index = find_action("explain");

    slots_.push_back(s);

    const int32_t A = static_cast<int32_t>(actions.size());
    total_actions_ += A;
    proj_.resize(static_cast<size_t>(total_actions_) * static_cast<size_t>(E_), 0.0f);
    bias_.resize(static_cast<size_t>(total_actions_), 0.0f);
    fitted_.resize(static_cast<size_t>(total_actions_), 0);
    if (scratch_.size() < static_cast<size_t>(A)) scratch_.resize(static_cast<size_t>(A));

    seed_weights(seed_);
    provenance_ = "seeded placeholder (UNTRAINED) — per-row splitmix64 seed " +
                  std::to_string(seed_) + ", E=" + std::to_string(E_) + ", " +
                  std::to_string(slots_.size()) + " domains / " +
                  std::to_string(total_actions_) + " actions (" +
                  std::to_string(fitted_rows_) + " fitted)";
    return static_cast<int32_t>(slots_.size()) - 1;
}

int32_t DomainDecisionHead::find_domain(Domain d) const {
    for (size_t i = 0; i < slots_.size(); ++i)
        if (slots_[i].domain == d) return static_cast<int32_t>(i);
    return -1;
}

void DomainDecisionHead::set_action_row(Domain d, int32_t action_index,
                                        const float* row, float bias) {
    if (!ready_ || row == nullptr) return;
    const int32_t si = find_domain(d);
    if (si < 0) return;
    const DomainSlot& s = slots_[static_cast<size_t>(si)];
    if (action_index < 0 || action_index >= static_cast<int32_t>(s.actions.size())) return;
    const size_t r = static_cast<size_t>(s.offset) + static_cast<size_t>(action_index);
    std::memcpy(&proj_[r * static_cast<size_t>(E_)], row,
                sizeof(float) * static_cast<size_t>(E_));
    bias_[r] = bias;
    if (!fitted_[r]) { fitted_[r] = 1; ++fitted_rows_; }
}

bool DomainDecisionHead::trained() const {
    return ready_ && total_actions_ > 0 && fitted_rows_ == total_actions_;
}

// ---------------------------------------------------------------------------
// The hot path
// ---------------------------------------------------------------------------
DomainDecision DomainDecisionHead::failure(Domain d, const char* why) const {
    DomainDecision o;
    o.domain   = domain_name(d);
    o.action   = "<error: " + std::string(why) + ">";
    o.routing  = "error";
    o.abstain  = true;
    o.fast_path = false;
    o.matvecs  = 0;
    return o;
}

DomainDecision DomainDecisionHead::decide(const float* hidden, Domain d) const {
    if (!ready_ || hidden == nullptr) return failure(d, "not-ready");
    const int32_t si = find_domain(d);
    if (si < 0) return failure(d, "domain not registered");

    const DomainSlot& s = slots_[static_cast<size_t>(si)];
    const int32_t A = static_cast<int32_t>(s.actions.size());
    if (A <= 0) return failure(d, "empty action space");
    if (scratch_.size() < static_cast<size_t>(A)) scratch_.resize(static_cast<size_t>(A));

    const double t0 = platform::now_ms();

    float max_logit = -3.4e38f;
    for (int32_t ai = 0; ai < A; ++ai) {
        const size_t row = static_cast<size_t>(s.offset) + static_cast<size_t>(ai);
        const float* w = &proj_[row * static_cast<size_t>(E_)];
        float acc = bias_[row];
        for (int32_t e = 0; e < E_; ++e) acc += w[e] * hidden[e];
        scratch_[static_cast<size_t>(ai)] = acc;
        if (acc > max_logit) max_logit = acc;
    }
    float sum = 0.0f;
    for (int32_t ai = 0; ai < A; ++ai) {
        const float v = std::exp(scratch_[static_cast<size_t>(ai)] - max_logit);
        scratch_[static_cast<size_t>(ai)] = v;
        sum += v;
    }
    if (!(sum > 0.0f)) {
        for (int32_t ai = 0; ai < A; ++ai) scratch_[static_cast<size_t>(ai)] = 1.0f / A;
    } else {
        for (int32_t ai = 0; ai < A; ++ai) scratch_[static_cast<size_t>(ai)] /= sum;
    }

    int32_t best = 0;
    for (int32_t ai = 1; ai < A; ++ai)
        if (scratch_[static_cast<size_t>(ai)] > scratch_[static_cast<size_t>(best)]) best = ai;
    float second = 0.0f;
    for (int32_t ai = 0; ai < A; ++ai) {
        if (ai == best) continue;
        if (scratch_[static_cast<size_t>(ai)] > second) second = scratch_[static_cast<size_t>(ai)];
    }

    DomainDecision o;
    o.domain     = domain_name(d);
    o.action     = s.actions[static_cast<size_t>(best)];
    o.confidence = scratch_[static_cast<size_t>(best)];
    o.margin     = o.confidence - second;
    o.matvecs    = A;

    // --- the fail-closed ladder ---------------------------------------------
    if (best == s.abstain_index) {
        o.routing = "abstain";
        o.abstain = true;
    } else if (best == s.explain_index) {
        o.routing = "system2";
        o.abstain = false;   // there IS a view; it just needs words
    } else if (o.confidence < cfg_.threshold) {
        o.routing = "system2";
        o.abstain = false;
    } else if (cfg_.margin_floor > 0.0f && o.margin < cfg_.margin_floor) {
        o.routing = "system2";
        o.abstain = false;
    } else {
        o.routing = "self";
        o.abstain = false;
    }
    o.fast_path = (o.routing == "self");

    o.us = (platform::now_ms() - t0) * 1000.0;
    last_us_ = o.us;
    return o;
}

DomainDecision DomainDecisionHead::decide(const Tensor& hidden, Domain d) const {
    if (hidden.dtype() != DType::F32) return failure(d, "hidden state is not f32");
    if (hidden.numel() != static_cast<int64_t>(E_)) return failure(d, "hidden width mismatch");
    return decide(hidden.f32(), d);
}

} // namespace omniseed
