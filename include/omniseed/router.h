// =============================================================================
//  OmniSeed — router.h
//
//  The ROUTER: one lightweight readout of h[E] that decides WHICH heads run and
//  in WHAT MODE. It is the third cheap consumer of the same hidden state, not a
//  second model.
//
//      input -> RWKV-7 forward pass -> h[E]
//                                        |
//                     +------------------+------------------+
//                     |                                     |
//                HeadRouter  -> ActivationPlan          (other heads)
//                     |
//        {heads: [decision, token, classify, score],
//         mode:  decision_only | decision+text | text_only,
//         domain: trading | language | vision | audio | general}
//
//  WHY A ROUTER AND NOT FOUR ALWAYS-ON HEADS. Generating text is the only
//  expensive head: it is a token loop whose cost scales with max_new_tokens,
//  while every other head is a single matvec. A structured decision that needs
//  no explanation should therefore never pay for a decode. The router is what
//  makes that choice, and it costs one small projection to do it.
//
//  THE KEY RULE, from the mandate, implemented in refine():
//      confidence >= 0.85 AND the task is simple
//          -> DecisionOnly  (drop the token head entirely)
//      otherwise
//          -> DecisionAndText (the decision becomes PREFIX CONTEXT for the text)
//  refine() can only ever DOWNGRADE — it removes the token head and never adds
//  it back. An uncertain router must not be able to talk itself into spending
//  a decode, and a confident one must not be able to suppress an explanation it
//  was asked for. Both directions fail safe.
//
//  COST, MEASURED — not estimated. The projection is
//  (kHeads + kDomainCount) * E = 9 * 768 = 6,912 multiply-accumulates, plus a
//  4-way sigmoid and a 5-way softmax. tests/test_router.cpp D1 runs 4000 calls
//  at E = 768 and asserts the mandate's <100 us budget; run it on your own box
//  for your own number. Note WHERE the time goes: the arithmetic is not the
//  dominant cost. The returned ActivationPlan owns a `heads` vector and a
//  `reason` string, and those two small allocations cost more than the 6,912
//  MACs — which is why plan() reserves both. The projection itself allocates
//  nothing and h is never mutated.
//
//  HONESTY NOTE. Like every head in this tree, a freshly constructed router has
//  NO trained projection: init() seeds it deterministically so tests are
//  reproducible, and `trained()` / `provenance()` report that. A seeded router
//  produces well-formed but MEANINGLESS plans. That is why:
//    * `ready() == false` returns the SAFE SUPERSET (decision + text) and says
//      so in `reason`, rather than guessing a shortcut;
//    * an explicit CLI `--mode` always wins over the router;
//    * the router is only consulted when a caller asks for it — it never
//      silently changes existing behaviour.
// =============================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "omniseed/core/rwkv.h"
#include "omniseed/core/tensor.h"
#include "omniseed/heads.h"

namespace omniseed {

// ---------------------------------------------------------------------------
// Which head. Order is canonical and used for the JSON `heads` array, so it is
// part of the wire contract: do not reorder.
// ---------------------------------------------------------------------------
enum class HeadKind : int32_t {
    Decision = 0,
    Token,
    Classify,
    Score,
    COUNT
};

constexpr int32_t kHeadKindCount = static_cast<int32_t>(HeadKind::COUNT);

const char* head_kind_name(HeadKind k);
bool        head_kind_from_name(const std::string& name, HeadKind& out);

// ---------------------------------------------------------------------------
// What the router asks the pipeline to do.
//
//   DecisionOnly      a structured answer is enough; generate nothing
//   DecisionAndText   decide first, then let the text head EXPLAIN that decision
//   TextOnly          legacy behaviour: generate, do not decide
// ---------------------------------------------------------------------------
enum class RouterMode : int32_t {
    DecisionOnly = 0,
    DecisionAndText,
    TextOnly,
    COUNT
};

const char* router_mode_name(RouterMode m);
// Wire form is "decision_only" / "decision+text" / "text_only". Accepts the
// hyphen and camel variants too, because CLI users type all three.
bool router_mode_from_name(const std::string& name, RouterMode& out);

// ---------------------------------------------------------------------------
// ActivationPlan — the router's whole output.
// ---------------------------------------------------------------------------
struct ActivationPlan {
    // Heads to run, in canonical HeadKind order. Empty is not a valid plan;
    // a plan always names at least one head.
    std::vector<HeadKind> heads;
    RouterMode mode = RouterMode::DecisionAndText;

    Domain domain = Domain::General;
    float  domain_confidence = 0.0f;

    // The router's own confidence in this plan (the max head gate). This is
    // NOT the decision head's confidence — refine() takes that separately.
    float confidence = 0.0f;

    // True when no head beyond decision(+token) is needed. Combined with a
    // confident decision this is what licenses the decision-only fast path.
    bool simple = false;

    // Human-readable justification. Always non-empty, so a log line never has
    // to guess why a mode was chosen.
    std::string reason;

    int32_t matvecs = 0;   // kHeads + kDomainCount for a real plan
    double  us      = 0.0; // wall time of the projection + softmax

    bool has(HeadKind k) const;
    int32_t head_count() const { return static_cast<int32_t>(heads.size()); }
    const char* mode_name() const { return router_mode_name(mode); }
    std::string heads_json() const;
    std::string to_json() const;
};

// ---------------------------------------------------------------------------
// HeadRouter
// ---------------------------------------------------------------------------
class HeadRouter {
public:
    struct Config {
        // Per-head sigmoid gates. A head runs when its gate clears its
        // threshold. These are the fitted knobs; the defaults are neutral
        // (0.50 = "an even coin says run it").
        float decision_gate = 0.50f;
        float token_gate    = 0.50f;
        float classify_gate = 0.50f;
        float score_gate    = 0.50f;
        // The mandate's fast path: at or above this, a SIMPLE task skips text.
        // 0.85 matches DecisionHead::Config::threshold and the sniper's regime
        // gate, so the three layers cannot disagree about what "confident" is.
        float fast_path_confidence = 0.85f;
    };

    HeadRouter() = default;
    explicit HeadRouter(const RwkvConfig& cfg, uint32_t seed = 20240u);

    bool init(const RwkvConfig& cfg, uint32_t seed = 20240u);
    bool init(int32_t n_embd, uint32_t seed = 20240u);

    bool ready() const { return ready_; }
    const std::string& error() const { return error_; }
    int32_t hidden_size() const { return E_; }

    const Config& config() const { return cfg_; }
    void set_config(const Config& c) { cfg_ = c; }

    // ---- the hot path -------------------------------------------------------
    // One projection + sigmoid/softmax. Allocation-free after init(), never
    // mutates `hidden`. Not thread-safe against itself (the scratch is reused);
    // same documented contract as DecisionHead and Uncertainty.
    ActivationPlan plan(const float* hidden) const;
    ActivationPlan plan(const Tensor& hidden) const;
    // A forced mode (the CLI's --mode) bypasses the gates for the MODE only;
    // the domain readout still runs, because a caller forcing "decision+text"
    // has said nothing about which domain this is.
    ActivationPlan plan(const float* hidden, RouterMode forced) const;

    // The mandate's rule, applied after the decision head has run. Can only
    // ever remove the token head; never adds it back.
    static ActivationPlan refine(const ActivationPlan& p, float decision_confidence,
                                 const Config& cfg);

    // ---- offline fitting hooks ---------------------------------------------
    // Overwrite one head's gate row (E floats) + bias, or one domain's probe
    // row. Counts as fitted; trained() requires EVERY row to be fitted, so a
    // half-populated router cannot masquerade as a real one.
    void set_head_gate(HeadKind k, const float* row, float bias);
    void set_domain_probe(Domain d, const float* row, float bias);

    // ---- provenance ---------------------------------------------------------
    bool trained() const;
    int32_t fitted_rows() const { return fitted_rows_; }
    int32_t total_rows() const { return kHeadKindCount + kDomainCount; }
    const std::string& provenance() const { return provenance_; }
    // Wall time (us) of the most recent plan(); for benchmarks and dashboards.
    double last_us() const { return last_us_; }

private:
    void seed_weights(uint32_t seed);

    Config  cfg_;
    int32_t E_ = 0;

    // [kHeadKindCount, E_] gate rows + [kHeadKindCount] bias
    std::vector<float> gate_;      // row-major
    std::vector<float> gate_bias_;
    // [kDomainCount, E_] domain probes + [kDomainCount] bias
    std::vector<float> dom_;       // row-major
    std::vector<float> dom_bias_;

    std::vector<uint8_t> fitted_;  // [total_rows] "this row came from fitting"

    bool        ready_       = false;
    int32_t     fitted_rows_ = 0;
    std::string provenance_  = "uninitialised";
    std::string error_;

    // Reused scratch — mutable so plan() can stay const and allocation-free.
    mutable std::vector<float> gates_;
    mutable std::vector<float> dom_logits_;
    mutable double             last_us_ = 0.0;
};

} // namespace omniseed
