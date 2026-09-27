// =============================================================================
//  OmniSeed — domain_decision.h
//
//  The DecisionHead pattern, generalised over ANY domain.
//
//  DecisionHead (decision_head.h) is the TRADING instance: a fixed seven-action
//  space, persisted weights, and a JSON contract that the paper daemon and
//  tools/decision_bridge.py already depend on. It is deliberately left exactly
//  as it was — rewriting a class that a live contract reads, in order to make
//  it "general", is how a working system acquires a silent regression.
//
//  This file is the generalisation alongside it:
//
//      DecisionHead       fixed trading actions, save/load, trading JSON
//      DomainDecisionHead N domains, N action spaces, one wire shape
//
//  Both are the same computation — [A, E] x [E] -> softmax -> argmax — so the
//  marginal cost of a second domain is its actions' rows, nothing more.
//
//  FAIL-CLOSED ROUTING, identical in spirit to DecisionHead's:
//      argmax is the ABSTAIN action      -> "abstain"  (say nothing)
//      argmax is the EXPLAIN action      -> "system2"  (needs language)
//      confidence < threshold            -> "system2"  (escalate, do not guess)
//      margin    < margin_floor          -> "system2"  (too close to call)
//      otherwise                         -> "self"     (act on it)
//  A domain may name its own escape actions; if it names none, an uncertain
//  decision still escalates, because "no escape hatch" must never mean
//  "therefore guess".
//
//  COST. One [A, E] matvec + softmax. For six actions at E = 768 that is 4,608
//  MACs — a few microseconds, three orders of magnitude inside the 1 ms budget.
//  Allocation-free after init(); h is never mutated.
//
//  HONESTY NOTE. A freshly constructed head has NO trained projection: init()
//  seeds it deterministically so tests are reproducible, and `trained()` /
//  `provenance()` report that. A seeded head emits well-formed but MEANINGLESS
//  actions. Never present one as a judgement.
// =============================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "omniseed/core/rwkv.h"
#include "omniseed/core/tensor.h"
#include "omniseed/decision_head.h"
#include "omniseed/heads.h"

namespace omniseed {

// ---------------------------------------------------------------------------
// DomainDecision — one structured decision, in a shape every domain shares.
//
// `action` is a lower-case label from the domain's own action list. Lower-case
// is the wire convention across every label set in this tree (see
// classification_head.h), and from_decision_result() lower-cases the trading
// head's UPPERCASE names to match, so a consumer never has to care which head
// produced a decision.
// ---------------------------------------------------------------------------
struct DomainDecision {
    std::string domain = "general";
    std::string action = "abstain";
    float       confidence = 0.0f;
    float       margin     = 0.0f;   // p(top) - p(second)
    bool        abstain    = false;  // true <=> routing is "abstain" or "error"
    std::string routing    = "abstain";  // self | system2 | abstain | error
    bool        fast_path  = false;  // true <=> routing == "self"
    int32_t     matvecs    = 0;
    double      us         = 0.0;

    std::string to_json() const;
};

// Lossless adapter: lift the trading head's DecisionResult into the shared
// shape. Action names are lower-cased; `target_asset` and `invalidation` have
// no equivalent here and stay on the trading result, where the risk engine and
// the paper daemon read them.
DomainDecision from_decision_result(const DecisionResult& d);

// ---------------------------------------------------------------------------
// DomainDecisionHead
// ---------------------------------------------------------------------------
class DomainDecisionHead {
public:
    struct Config {
        // Escalate below this. 0.85 matches DecisionHead::Config::threshold,
        // the router's fast_path_confidence, and the sniper's regime gate, so
        // no two layers can disagree about what "confident" means.
        float threshold = 0.85f;
        // Optional floor on p(top) - p(second). A 0.90/0.88 split clears the
        // confidence bar but is nearly a coin flip; set > 0 to demand a clear
        // winner. 0 disables the check.
        float margin_floor = 0.0f;
    };

    struct DomainSlot {
        Domain                   domain = Domain::General;
        std::vector<std::string> actions;
        int32_t                  offset = 0;        // first row in proj_
        int32_t                  abstain_index = -1;
        int32_t                  explain_index = -1;
    };

    DomainDecisionHead() = default;
    explicit DomainDecisionHead(const RwkvConfig& cfg, uint32_t seed = 4242u);

    bool init(const RwkvConfig& cfg, uint32_t seed = 4242u);
    bool init(int32_t n_embd, uint32_t seed = 4242u);

    bool ready() const { return ready_; }
    const std::string& error() const { return error_; }
    int32_t hidden_size() const { return E_; }

    const Config& config() const { return cfg_; }
    void set_config(const Config& c) { cfg_ = c; }
    void set_threshold(float t) { cfg_.threshold = t; }
    void set_margin_floor(float m) { cfg_.margin_floor = m; }

    // ---- domains ------------------------------------------------------------
    // Registers a domain and its action space. Returns the slot index, or -1 on
    // a duplicate domain, an empty action list, or a head that is not ready.
    //
    // `abstain_action` / `explain_action` name the escape hatches within
    // `actions`; pass "" for either to have none. If `abstain_action` is "" but
    // the list contains "abstain", it is used automatically — the vocabulary is
    // fixed across this tree, so the common case needs no ceremony.
    int32_t add_domain(Domain d, const std::vector<std::string>& actions,
                       const std::string& abstain_action = "",
                       const std::string& explain_action = "");

    int32_t domain_count() const { return static_cast<int32_t>(slots_.size()); }
    int32_t find_domain(Domain d) const;
    const DomainSlot& slot(int32_t i) const { return slots_[static_cast<size_t>(i)]; }
    int32_t total_actions() const { return total_actions_; }

    // ---- the hot path -------------------------------------------------------
    // One matvec + softmax + argmax, then the fail-closed routing ladder.
    // Allocation-free after init(); not thread-safe against itself.
    // An unregistered domain returns routing "error" — never a fabricated
    // action, because "I have no action space for this" is information.
    DomainDecision decide(const float* hidden, Domain d) const;
    DomainDecision decide(const Tensor& hidden, Domain d) const;

    // ---- offline fitting hooks ---------------------------------------------
    void set_action_row(Domain d, int32_t action_index, const float* row, float bias);

    // ---- provenance ---------------------------------------------------------
    bool trained() const;
    int32_t fitted_rows() const { return fitted_rows_; }
    const std::string& provenance() const { return provenance_; }
    double last_us() const { return last_us_; }

private:
    void seed_weights(uint32_t seed);
    DomainDecision failure(Domain d, const char* why) const;

    Config  cfg_;
    int32_t E_ = 0;
    uint32_t seed_ = 4242u;

    std::vector<DomainSlot> slots_;
    std::vector<float>      proj_;   // [total_actions, E_], row-major
    std::vector<float>      bias_;   // [total_actions]
    std::vector<uint8_t>    fitted_; // [total_actions]
    int32_t                 total_actions_ = 0;
    int32_t                 fitted_rows_   = 0;

    bool        ready_      = false;
    std::string provenance_ = "uninitialised";
    std::string error_;

    mutable std::vector<float> scratch_;
    mutable double             last_us_ = 0.0;
};

// ---------------------------------------------------------------------------
// The action spaces the mandate names, in one place. Registering them is
// idempotent per domain (a duplicate is refused).
// ---------------------------------------------------------------------------
int32_t add_default_domains(DomainDecisionHead& head);

} // namespace omniseed
