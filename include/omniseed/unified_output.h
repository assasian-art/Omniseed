// =============================================================================
//  OmniSeed — unified_output.h
//
//  ONE JSON DOCUMENT, EVERY HEAD'S OPINION.
//
//      {
//        "router":         {activated_heads, mode, domain, ...},
//        "decision":       {domain, action, confidence, margin, routing, ...},
//        "text":           "I'm recommending buy at 92% because ...",
//        "classification": {domain, top_k: [...], margin, ...},
//        "score":          {priority, urgency, confidence, ...}
//      }
//
//  OPTIONAL FIELDS ARE OMITTED, NOT NULLED. A field is present if and only if
//  its head actually ran. That matters more than it looks: a `"decision": null`
//  is indistinguishable from a decision head that abstained, and a consumer
//  that treats "absent" and "abstained" as the same thing will eventually act
//  on a field that was never computed.
//
//  THE INVARIANT THIS FILE ENFORCES: a field is emitted only when the thing
//  that produces it actually ran. For the four HEADS that means "named in the
//  router's activation plan"; if a caller fills in a decision the plan never
//  asked for, to_json() DROPS it and records the fact in `warnings`. Dropping
//  is the safe direction — emitting a head's output that the router did not
//  activate would let a stale value from a previous turn leak into the current
//  answer.
//
//  THE `soul` SECTION IS GATED THE SAME WAY, BY A DIFFERENT TEST. The soul is
//  not a projection of h[E] — it reads the owner's turn, not the hidden state —
//  so it has no HeadKind and the plan cannot name it. Its equivalent guard is
//  that SoulState carries `has_self`, which only `Soul::perceive()` ever sets:
//  normalise() drops a soul section that was never perceived, exactly as it
//  drops a decision the plan never named. The invariant is unchanged in
//  substance ("present iff computed"); only the evidence differs.
//
//  THE PIPELINE, and the one behaviour worth reading the code for:
//  UnifiedPipeline::run() consults the router, then runs only the heads the
//  plan names. When the plan is DecisionOnly the text producer is NEVER CALLED
//  — that is the entire point of the fast path, and it is asserted directly in
//  tests/test_unified_output.cpp rather than inferred from a timing number.
// =============================================================================
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "omniseed/classification_head.h"
#include "omniseed/domain_decision.h"
#include "omniseed/heads.h"
#include "omniseed/router.h"
#include "omniseed/scoring_head.h"
#include "omniseed/soul.h"

namespace omniseed {

// ---------------------------------------------------------------------------
// UnifiedOutput — the assembled document.
// ---------------------------------------------------------------------------
struct UnifiedOutput {
    // Always present: the router runs first by definition, and even a
    // "not-ready" router produces a plan (the safe superset, with a reason).
    ActivationPlan router;

    bool           has_decision = false;
    DomainDecision decision;

    bool        has_text = false;
    std::string text;

    bool                 has_classification = false;
    ClassificationResult classification;

    bool        has_score = false;
    ScoreResult score;

    // The soul's read on this turn: who is answering, how the owner sounds, and
    // what the decision log honestly supports. Present iff a soul stage was
    // configured AND it perceived the turn (see the note at the top).
    bool      has_soul = false;
    SoulState soul;

    // Diagnostics, never part of the head outputs.
    bool token_head_invoked = false;  // did the text producer actually run?
    std::vector<std::string> warnings;

    // Drops any field whose head is not in `router.heads`, recording each drop
    // in `warnings`. Idempotent. Returns how many fields were dropped.
    int32_t normalise();
    // True when no field is present that the plan does not name.
    bool consistent() const;

    std::string to_json() const;
};

// ---------------------------------------------------------------------------
// UnifiedPipeline — the router plus every head, over one backbone.
//
// Owns nothing the caller's model owns: it takes h[E] and returns a document.
// The text head is supplied as a callback because the generator already exists
// (AgentLoop / the CLI) and this file must not become a second copy of it.
// ---------------------------------------------------------------------------
class UnifiedPipeline {
public:
    struct Config {
        HeadRouter::Config        router;
        DomainDecisionHead::Config decision;
        // A classification set to run when the plan activates the classify head.
        // Empty disables it (the head simply never reports).
        std::string classification_set = "general.routing";
        int32_t     classification_top_k = 3;
        // Force a mode, bypassing the router's gates (the CLI's --mode).
        // std::nullopt-equivalent: `force_mode` false means "let the router decide".
        bool        force_mode = false;
        RouterMode  mode = RouterMode::DecisionAndText;
        // Apply the mandate's fast-path rule after the decision head runs.
        bool        use_fast_path = true;

        // The soul stage. OFF BY DEFAULT, deliberately: with it off the
        // decision-only and text-only paths produce byte-identical documents to
        // before this existed, so turning it on is an explicit choice rather
        // than a silent change to every caller's output.
        bool        use_soul = false;
        Soul::Config soul;
    };

    UnifiedPipeline() = default;

    // Registers the default label sets and action spaces. Idempotent per name,
    // so calling it twice is harmless.
    bool init(int32_t n_embd, uint32_t seed = 20240u);

    bool ready() const { return ready_; }
    const std::string& error() const { return error_; }

    const Config& config() const { return cfg_; }
    void set_config(const Config& c) { cfg_ = c; }

    HeadRouter&        router()        { return router_; }
    const HeadRouter&  router()  const { return router_; }
    DomainDecisionHead&       decisions()       { return decisions_; }
    const DomainDecisionHead& decisions() const { return decisions_; }
    ClassificationHead&       classifier()       { return classifier_; }
    const ClassificationHead& classifier() const { return classifier_; }
    ScoringHead&              scorer()       { return scorer_; }
    const ScoringHead&        scorer() const { return scorer_; }
    Soul&                     soul()         { return soul_; }
    const Soul&               soul()   const { return soul_; }
    // True when a soul stage was configured and initialised successfully.
    bool has_soul_stage() const { return soul_ready_; }

    // Run the pipeline on one hidden state. `text_fn` is invoked ONLY when the
    // plan names the token head; it receives the decision (which may be an
    // abstain/error result) so the text can explain what was decided.
    // Pass an empty function to run with no text head at all.
    UnifiedOutput run(const float* hidden,
                      const std::function<std::string(const DomainDecision&)>& text_fn) const;
    UnifiedOutput run(const float* hidden) const;

    // The same run, plus the soul's read on the owner's turn. The soul stage
    // runs only when Config::use_soul is set; otherwise this is identical to
    // run(hidden, text_fn) and the document carries no `soul` key at all.
    UnifiedOutput run(const float* hidden, const std::string& user_turn,
                      const std::function<std::string(const DomainDecision&)>& text_fn) const;
    UnifiedOutput run(const float* hidden, const std::string& user_turn) const;

    // ---- provenance ---------------------------------------------------------
    // True only when EVERY head is fitted. A pipeline of seeded heads emits
    // well-formed but meaningless documents.
    bool trained() const;
    std::string provenance() const;

private:
    bool ready_ = false;
    bool soul_ready_ = false;
    std::string error_;

    Config              cfg_;
    HeadRouter          router_;
    DomainDecisionHead  decisions_;
    ClassificationHead  classifier_;
    ScoringHead         scorer_;
    Soul                soul_;
};

} // namespace omniseed
