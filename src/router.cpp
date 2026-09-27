// =============================================================================
//  OmniSeed — src/router.cpp
//
//  The HeadRouter. See router.h for the contract, the cost budget, and the
//  honesty note about untrained (seeded) probes.
//
//  Hot path (plan) is allocation-free after init() and touches nothing but its
//  own scratch. It never mutates the hidden state, so routing cannot perturb a
//  conversation.
// =============================================================================
#include "omniseed/router.h"
#include "omniseed/core/platform.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace omniseed {

// ---------------------------------------------------------------------------
// Names
// ---------------------------------------------------------------------------
const char* head_kind_name(HeadKind k) {
    switch (k) {
        case HeadKind::Decision: return "decision";
        case HeadKind::Token:    return "token";
        case HeadKind::Classify: return "classify";
        case HeadKind::Score:    return "score";
        case HeadKind::COUNT:    break;
    }
    return "unknown";
}

bool head_kind_from_name(const std::string& name, HeadKind& out) {
    for (int32_t i = 0; i < kHeadKindCount; ++i) {
        const HeadKind k = static_cast<HeadKind>(i);
        if (name == head_kind_name(k)) { out = k; return true; }
    }
    return false;
}

const char* router_mode_name(RouterMode m) {
    switch (m) {
        case RouterMode::DecisionOnly:    return "decision_only";
        case RouterMode::DecisionAndText: return "decision+text";
        case RouterMode::TextOnly:        return "text_only";
        case RouterMode::COUNT:           break;
    }
    return "decision+text";
}

bool router_mode_from_name(const std::string& name, RouterMode& out) {
    // Every spelling a human is likely to type. The canonical wire forms are
    // first; the aliases exist so `--mode decision-only` and `--mode textonly`
    // are not errors.
    if (name == "decision_only" || name == "decision-only" || name == "decisiononly")
        { out = RouterMode::DecisionOnly; return true; }
    if (name == "decision+text" || name == "decision_text" || name == "decision-text" ||
        name == "hybrid" || name == "both")
        { out = RouterMode::DecisionAndText; return true; }
    if (name == "text_only" || name == "text-only" || name == "textonly" ||
        name == "off" || name == "text")
        { out = RouterMode::TextOnly; return true; }
    return false;
}

// ---------------------------------------------------------------------------
// ActivationPlan
// ---------------------------------------------------------------------------
bool ActivationPlan::has(HeadKind k) const {
    return std::find(heads.begin(), heads.end(), k) != heads.end();
}

std::string ActivationPlan::heads_json() const {
    std::string out = "[";
    for (size_t i = 0; i < heads.size(); ++i) {
        if (i) out += ',';
        out += '"';
        out += head_kind_name(heads[i]);
        out += '"';
    }
    out += ']';
    return out;
}

std::string ActivationPlan::to_json() const {
    std::string out = "{\"activated_heads\":";
    out += heads_json();
    out += ",\"mode\":\"";
    out += mode_name();
    out += "\",\"domain\":\"";
    out += heads_detail::json_escape(domain_name(domain));
    out += "\",\"domain_confidence\":";
    heads_detail::append_float(out, domain_confidence);
    out += ",\"confidence\":";
    heads_detail::append_float(out, confidence);
    out += ",\"simple\":";
    out += simple ? "true" : "false";
    out += ",\"reason\":\"";
    out += heads_detail::json_escape(reason);
    out += "\",\"matvecs\":";
    out += std::to_string(matvecs);
    out += '}';
    return out;
}

// ---------------------------------------------------------------------------
// Construction / geometry
// ---------------------------------------------------------------------------
HeadRouter::HeadRouter(const RwkvConfig& cfg, uint32_t seed) { init(cfg, seed); }

bool HeadRouter::init(const RwkvConfig& cfg, uint32_t seed) {
    return init(cfg.n_embd, seed);
}

bool HeadRouter::init(int32_t n_embd, uint32_t seed) {
    ready_       = false;
    fitted_rows_ = 0;
    error_.clear();

    if (n_embd <= 0) {
        error_ = "HeadRouter::init: n_embd must be > 0";
        provenance_ = "error: " + error_;
        return false;
    }

    E_ = n_embd;
    const int32_t R = kHeadKindCount + kDomainCount;
    gate_.assign(static_cast<size_t>(kHeadKindCount) * E_, 0.0f);
    gate_bias_.assign(kHeadKindCount, 0.0f);
    dom_.assign(static_cast<size_t>(kDomainCount) * E_, 0.0f);
    dom_bias_.assign(kDomainCount, 0.0f);
    fitted_.assign(static_cast<size_t>(R), 0);
    gates_.assign(kHeadKindCount, 0.0f);
    dom_logits_.assign(kDomainCount, 0.0f);

    seed_weights(seed);
    ready_ = true;
    provenance_ = "seeded placeholder (UNTRAINED) — splitmix64 seed " +
                  std::to_string(seed) + ", E=" + std::to_string(E_) +
                  ", rows=" + std::to_string(R) + " (0 fitted)";
    return true;
}

void HeadRouter::seed_weights(uint32_t seed) {
    // Multiply-add, NOT `seed | constant`. OR-ing a constant into the seed is
    // NOT injective: 0x9E3779B97F4A7C15 already has bits 0x4F10 set, so seeds
    // 20240 and 20241 OR to the same state and yield byte-identical probes.
    // Bumping a seed by one must actually change the router. The multiply-add
    // form is a bijection mod 2^64 (the multiplier is odd), and it matches
    // DecisionHead::seed_weights, which had it right all along.
    // tests/test_router.cpp B12 pins this.
    uint64_t s = static_cast<uint64_t>(seed) * 0x9E3779B97F4A7C15ull +
                 0xD1B54A32D192ED03ull;
    // Scaled so the seeded gates land near 0.5 rather than saturating: a probe
    // whose dot product is huge would make sigmoid a step function and every
    // plan would be decided by a single coordinate's sign. 1/sqrt(E) keeps the
    // dot product O(1) for unit-ish inputs.
    const float scale = 1.0f / std::sqrt(static_cast<float>(E_));
    for (float& w : gate_) w = heads_detail::next_unit(s) * scale;
    for (float& b : gate_bias_) b = heads_detail::next_unit(s) * 0.1f;
    for (float& w : dom_) w = heads_detail::next_unit(s) * scale;
    for (float& b : dom_bias_) b = heads_detail::next_unit(s) * 0.1f;
}

// ---------------------------------------------------------------------------
// Fitting hooks
// ---------------------------------------------------------------------------
void HeadRouter::set_head_gate(HeadKind k, const float* row, float bias) {
    if (!ready_ || row == nullptr) return;
    const int32_t idx = static_cast<int32_t>(k);
    if (idx < 0 || idx >= kHeadKindCount) return;
    std::memcpy(&gate_[static_cast<size_t>(idx) * E_], row,
                sizeof(float) * static_cast<size_t>(E_));
    gate_bias_[static_cast<size_t>(idx)] = bias;
    if (!fitted_[static_cast<size_t>(idx)]) {
        fitted_[static_cast<size_t>(idx)] = 1;
        ++fitted_rows_;
    }
}

void HeadRouter::set_domain_probe(Domain d, const float* row, float bias) {
    if (!ready_ || row == nullptr) return;
    const int32_t idx = static_cast<int32_t>(d);
    if (idx < 0 || idx >= kDomainCount) return;
    std::memcpy(&dom_[static_cast<size_t>(idx) * E_], row,
                sizeof(float) * static_cast<size_t>(E_));
    dom_bias_[static_cast<size_t>(idx)] = bias;
    const size_t slot = static_cast<size_t>(kHeadKindCount + idx);
    if (!fitted_[slot]) {
        fitted_[slot] = 1;
        ++fitted_rows_;
    }
}

bool HeadRouter::trained() const {
    if (!ready_) return false;
    const int32_t R = kHeadKindCount + kDomainCount;
    return fitted_rows_ == R && R > 0;
}

// ---------------------------------------------------------------------------
// The hot path
// ---------------------------------------------------------------------------
namespace {

inline float sigmoidf(float x) {
    if (x >= 0.0f) {
        const float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    const float z = std::exp(x);
    return z / (1.0f + z);
}

// A plan that runs the SAFE SUPERSET and says why. Used whenever the router
// cannot or should not take a shortcut.
ActivationPlan safe_plan(const char* reason) {
    ActivationPlan p;
    p.heads = {HeadKind::Decision, HeadKind::Token};
    p.mode = RouterMode::DecisionAndText;
    p.domain = Domain::General;
    p.simple = false;
    p.confidence = 0.0f;
    p.reason = reason;
    return p;
}

} // namespace

ActivationPlan HeadRouter::plan(const float* hidden) const {
    if (!ready_ || hidden == nullptr) return safe_plan("not-ready: no projection loaded");

    const double t0 = platform::now_ms();

    // --- head gates: one matvec per head, then a sigmoid --------------------
    for (int32_t i = 0; i < kHeadKindCount; ++i) {
        const float* row = &gate_[static_cast<size_t>(i) * E_];
        float acc = gate_bias_[static_cast<size_t>(i)];
        for (int32_t e = 0; e < E_; ++e) acc += row[e] * hidden[e];
        gates_[static_cast<size_t>(i)] = sigmoidf(acc);
    }
    // --- domain: one matvec per domain, then a softmax ---------------------
    float max_logit = -3.4e38f;
    for (int32_t d = 0; d < kDomainCount; ++d) {
        const float* row = &dom_[static_cast<size_t>(d) * E_];
        float acc = dom_bias_[static_cast<size_t>(d)];
        for (int32_t e = 0; e < E_; ++e) acc += row[e] * hidden[e];
        dom_logits_[static_cast<size_t>(d)] = acc;
        if (acc > max_logit) max_logit = acc;
    }
    float sum = 0.0f;
    for (int32_t d = 0; d < kDomainCount; ++d) {
        const float v = std::exp(dom_logits_[static_cast<size_t>(d)] - max_logit);
        dom_logits_[static_cast<size_t>(d)] = v;
        sum += v;
    }
    int32_t best_d = 0;
    if (sum > 0.0f) {
        for (int32_t d = 1; d < kDomainCount; ++d)
            if (dom_logits_[static_cast<size_t>(d)] > dom_logits_[static_cast<size_t>(best_d)])
                best_d = d;
        for (float& v : dom_logits_) v /= sum;
    } else {
        // Degenerate softmax (all logits -inf). Report a uniform belief rather
        // than a fabricated winner.
        for (float& v : dom_logits_) v = 1.0f / static_cast<float>(kDomainCount);
    }

    ActivationPlan p;
    // The returned plan owns a vector and a string, so a plan cannot be fully
    // allocation-free — but it can be TWO allocations instead of five. The
    // measurement in tests/test_router.cpp D1 shows the allocations, not the
    // 6,912 MACs, dominate the cost, so the reserve is worth having.
    p.heads.reserve(static_cast<size_t>(kHeadKindCount));
    p.domain = static_cast<Domain>(best_d);
    p.domain_confidence = dom_logits_[static_cast<size_t>(best_d)];
    p.matvecs = kHeadKindCount + kDomainCount;

    const bool want_decision = gates_[static_cast<size_t>(HeadKind::Decision)] >= cfg_.decision_gate;
    const bool want_token    = gates_[static_cast<size_t>(HeadKind::Token)] >= cfg_.token_gate;
    const bool want_classify = gates_[static_cast<size_t>(HeadKind::Classify)] >= cfg_.classify_gate;
    const bool want_score    = gates_[static_cast<size_t>(HeadKind::Score)] >= cfg_.score_gate;

    p.confidence = std::max(std::max(gates_[static_cast<size_t>(HeadKind::Decision)],
                                     gates_[static_cast<size_t>(HeadKind::Token)]),
                            std::max(gates_[static_cast<size_t>(HeadKind::Classify)],
                                     gates_[static_cast<size_t>(HeadKind::Score)]));

    // Fail-closed: if NO gate fired the router has no opinion at all, and the
    // safe reading of "no opinion" is to run the superset rather than to guess
    // a shortcut that would silently suppress a decision or an explanation.
    if (!want_decision && !want_token && !want_classify && !want_score) {
        ActivationPlan q = safe_plan("no-gate-fired (fail-closed: run the superset)");
        q.domain = p.domain;
        q.domain_confidence = p.domain_confidence;
        q.confidence = p.confidence;
        q.matvecs = p.matvecs;
        q.us = (platform::now_ms() - t0) * 1000.0;
        last_us_ = q.us;
        return q;
    }

    if (want_decision) p.heads.push_back(HeadKind::Decision);
    if (want_token)    p.heads.push_back(HeadKind::Token);
    if (want_classify) p.heads.push_back(HeadKind::Classify);
    if (want_score)    p.heads.push_back(HeadKind::Score);

    // The mode follows the two heads that define it; classify and score are
    // orthogonal and can ride along with either.
    if (want_decision && want_token)      p.mode = RouterMode::DecisionAndText;
    else if (want_decision)               p.mode = RouterMode::DecisionOnly;
    else                                  p.mode = RouterMode::TextOnly;

    // "Simple" = nothing beyond decision(+token) is needed. That is what
    // licenses the fast path in refine(); a task that also wants a
    // classification or a score is by definition not simple.
    p.simple = !want_classify && !want_score;

    p.reason.reserve(64);
    p.reason = "gates fired: ";
    p.reason += (want_decision ? "decision " : "");
    p.reason += (want_token ? "token " : "");
    p.reason += (want_classify ? "classify " : "");
    p.reason += (want_score ? "score " : "");

    p.us = (platform::now_ms() - t0) * 1000.0;
    last_us_ = p.us;
    return p;
}

ActivationPlan HeadRouter::plan(const Tensor& hidden) const {
    if (hidden.dtype() != DType::F32) return safe_plan("not-ready: hidden state is not f32");
    if (hidden.numel() != static_cast<int64_t>(E_)) return safe_plan("not-ready: hidden width mismatch");
    return plan(hidden.f32());
}

ActivationPlan HeadRouter::plan(const float* hidden, RouterMode forced) const {
    // The forced mode replaces the router's MODE decision only. The domain
    // readout still runs — a caller forcing "decision+text" has said nothing
    // about which domain this is, and inventing one would be worse than
    // reporting the router's actual (possibly uniform) belief.
    ActivationPlan p = plan(hidden);
    p.mode = forced;
    p.simple = (forced == RouterMode::DecisionOnly);
    p.reason = std::string("forced: ") + router_mode_name(forced);
    switch (forced) {
        case RouterMode::DecisionOnly:
            p.heads = {HeadKind::Decision};
            break;
        case RouterMode::TextOnly:
            p.heads = {HeadKind::Token};
            break;
        case RouterMode::DecisionAndText:
            p.heads = {HeadKind::Decision, HeadKind::Token};
            break;
        case RouterMode::COUNT:
            break;
    }
    return p;
}

ActivationPlan HeadRouter::refine(const ActivationPlan& p, float decision_confidence,
                                  const Config& cfg) {
    // The mandate's rule. Both guards are deliberate:
    //   * only DecisionAndText can be downgraded — TextOnly was asked for, and
    //     DecisionOnly is already minimal;
    //   * `simple` must be true — a confident answer to a task that also wanted
    //     a classification is not a reason to skip the explanation.
    if (p.mode != RouterMode::DecisionAndText) return p;
    if (!p.simple) return p;
    if (!(decision_confidence >= cfg.fast_path_confidence)) return p;

    ActivationPlan q = p;
    q.mode = RouterMode::DecisionOnly;
    q.heads.clear();
    for (const HeadKind k : p.heads)
        if (k != HeadKind::Token) q.heads.push_back(k);
    if (q.heads.empty()) q.heads.push_back(HeadKind::Decision);
    q.reason = "fast-path: decision confidence " +
               std::to_string(decision_confidence).substr(0, 5) +
               " >= threshold and the task is simple -> text head dropped";
    return q;
}

} // namespace omniseed
