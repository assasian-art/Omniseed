// =============================================================================
//  OmniSeed — src/unified_output.cpp
//
//  Assembles the one document, and runs the pipeline that produces it.
//
//  The two rules this file exists to enforce:
//
//   1. A FIELD IS EMITTED ONLY IF ITS HEAD RAN. normalise() drops anything the
//      activation plan does not name, and records the drop in `warnings`. A
//      stale decision from a previous turn leaking into this turn's answer is
//      the failure mode being prevented, and it is silent by nature.
//
//   2. THE FAST PATH REALLY IS FAST. When the plan says DecisionOnly, the text
//      producer is never called — not called-and-discarded. That is asserted in
//      the tests by counting invocations, because a timing measurement alone
//      cannot tell "skipped" from "fast".
// =============================================================================
#include "omniseed/unified_output.h"
#include "omniseed/core/platform.h"

#include <string>

namespace omniseed {

// ---------------------------------------------------------------------------
// UnifiedOutput
// ---------------------------------------------------------------------------
int32_t UnifiedOutput::normalise() {
    int32_t dropped = 0;
    if (has_decision && !router.has(HeadKind::Decision)) {
        has_decision = false;
        decision = DomainDecision();
        warnings.push_back("dropped decision: not named in the activation plan");
        ++dropped;
    }
    if (has_text && !router.has(HeadKind::Token)) {
        has_text = false;
        text.clear();
        warnings.push_back("dropped text: not named in the activation plan");
        ++dropped;
    }
    if (has_classification && !router.has(HeadKind::Classify)) {
        has_classification = false;
        classification = ClassificationResult();
        warnings.push_back("dropped classification: not named in the activation plan");
        ++dropped;
    }
    if (has_score && !router.has(HeadKind::Score)) {
        has_score = false;
        score = ScoreResult();
        warnings.push_back("dropped score: not named in the activation plan");
        ++dropped;
    }
    // The soul has no HeadKind — it reads the turn, not h[E], so the plan
    // cannot name it. Its evidence of having run is `has_self`, which only
    // Soul::perceive() sets. Same rule, different witness.
    if (has_soul && !soul.has_self) {
        has_soul = false;
        soul = SoulState();
        warnings.push_back("dropped soul: never perceived a turn");
        ++dropped;
    }
    return dropped;
}

bool UnifiedOutput::consistent() const {
    if (has_decision && !router.has(HeadKind::Decision)) return false;
    if (has_text && !router.has(HeadKind::Token)) return false;
    if (has_classification && !router.has(HeadKind::Classify)) return false;
    if (has_score && !router.has(HeadKind::Score)) return false;
    if (has_soul && !soul.has_self) return false;
    return true;
}

std::string UnifiedOutput::to_json() const {
    std::string out = "{\"router\":";
    out += router.to_json();

    // Order is the mandate's: router, decision, text, classification, score,
    // then soul. A reader scanning the document top-down sees the plan before
    // its results, and the machine-readable heads before the human one.
    if (has_decision) {
        out += ",\"decision\":";
        out += decision.to_json();
    }
    if (has_text) {
        out += ",\"text\":\"";
        out += heads_detail::json_escape(text);
        out += '"';
    }
    if (has_classification) {
        out += ",\"classification\":";
        out += classification.to_json();
    }
    if (has_score) {
        out += ",\"score\":";
        out += score.to_json();
    }
    if (has_soul) {
        out += ",\"soul\":";
        out += soul.to_json();
    }
    if (!warnings.empty()) {
        out += ",\"warnings\":[";
        for (size_t i = 0; i < warnings.size(); ++i) {
            if (i) out += ',';
            out += '"';
            out += heads_detail::json_escape(warnings[i]);
            out += '"';
        }
        out += ']';
    }
    out += '}';
    return out;
}

// ---------------------------------------------------------------------------
// UnifiedPipeline
// ---------------------------------------------------------------------------
bool UnifiedPipeline::init(int32_t n_embd, uint32_t seed) {
    ready_ = false;
    error_.clear();

    if (n_embd <= 0) {
        error_ = "UnifiedPipeline::init: n_embd must be > 0";
        return false;
    }
    if (!router_.init(n_embd, seed))            { error_ = router_.error();     return false; }
    if (!decisions_.init(n_embd, seed + 1u))    { error_ = decisions_.error();  return false; }
    if (!classifier_.init(n_embd, seed + 2u))   { error_ = classifier_.error(); return false; }
    if (!scorer_.init(n_embd, seed + 3u))       { error_ = scorer_.error();     return false; }

    router_.set_config(cfg_.router);
    decisions_.set_config(cfg_.decision);
    // Idempotent per name: a caller that already registered its own sets keeps
    // them, and a second init() cannot double-allocate rows.
    add_default_label_sets(classifier_);
    add_default_domains(decisions_);

    // The soul stage is initialised only when it is configured to run, so a
    // caller that never asks for it cannot be broken by a bad soul config.
    soul_ready_ = false;
    if (cfg_.use_soul) {
        if (!soul_.init(cfg_.soul)) { error_ = soul_.error(); return false; }
        soul_ready_ = true;
    }

    ready_ = true;
    return true;
}

bool UnifiedPipeline::trained() const {
    return ready_ && router_.trained() && decisions_.trained() &&
           classifier_.trained() && scorer_.trained();
}

std::string UnifiedPipeline::provenance() const {
    std::string out = "UnifiedPipeline:";
    out += " router=";   out += router_.provenance();
    out += " | decisions="; out += decisions_.provenance();
    out += " | classifier="; out += classifier_.provenance();
    out += " | scorer="; out += scorer_.provenance();
    // The soul is the one stage whose honesty does not depend on fitted
    // weights: it reports a measured calibration gap, not a projection.
    out += " | soul=";
    out += soul_ready_ ? "active (rule-based, self-reporting)"
                       : "not configured";
    out += trained() ? " | ALL FITTED" : " | NOT FULLY FITTED (placeholder)";
    return out;
}

UnifiedOutput UnifiedPipeline::run(
    const float* hidden,
    const std::function<std::string(const DomainDecision&)>& text_fn) const {
    UnifiedOutput out;

    if (!ready_ || hidden == nullptr) {
        // Not-ready is itself a plan: the safe superset, with a reason. The
        // document is still well-formed, so a consumer never has to special-case
        // a missing router object.
        out.router = router_.plan(static_cast<const float*>(nullptr));
        out.normalise();
        return out;
    }

    // --- 1. the router decides what runs ------------------------------------
    out.router = cfg_.force_mode ? router_.plan(hidden, cfg_.mode) : router_.plan(hidden);

    // --- 2. the decision head, if the plan names it -------------------------
    if (out.router.has(HeadKind::Decision)) {
        out.decision = decisions_.decide(hidden, out.router.domain);
        out.has_decision = true;
    }

    // --- 3. the mandate's fast path -----------------------------------------
    // Applied AFTER the decision head, because it is the decision's confidence
    // that licenses dropping the text head. Only ever downgrades.
    if (cfg_.use_fast_path && out.has_decision && out.router.mode == RouterMode::DecisionAndText) {
        out.router = HeadRouter::refine(out.router, out.decision.confidence, cfg_.router);
    }

    // --- 4. the text head, if the (possibly refined) plan still names it -----
    //
    // The call is INSIDE the branch on purpose. A plan that dropped the token
    // head must not pay for a decode, and `token_head_invoked` records that the
    // producer never ran — which is the observable difference between "skipped"
    // and "fast".
    if (out.router.has(HeadKind::Token) && static_cast<bool>(text_fn)) {
        out.text = text_fn(out.has_decision ? out.decision : DomainDecision());
        out.has_text = true;
        out.token_head_invoked = true;
    }

    // --- 5. the cheap heads, if the plan names them --------------------------
    if (out.router.has(HeadKind::Classify) && !cfg_.classification_set.empty()) {
        out.classification = classifier_.classify(hidden, cfg_.classification_set,
                                                  cfg_.classification_top_k);
        // An empty top_k means the head refused (unknown set, width mismatch).
        // Reporting a head that produced nothing would violate the
        // "present iff it ran" rule, so the field is not claimed.
        out.has_classification = !out.classification.top_k.empty();
    }
    if (out.router.has(HeadKind::Score)) {
        out.score = scorer_.score(hidden);
        out.has_score = (out.score.matvecs > 0);
    }

    out.normalise();
    return out;
}

UnifiedOutput UnifiedPipeline::run(const float* hidden) const {
    return run(hidden, std::function<std::string(const DomainDecision&)>());
}

UnifiedOutput UnifiedPipeline::run(
    const float* hidden, const std::string& user_turn,
    const std::function<std::string(const DomainDecision&)>& text_fn) const {
    // The heads first, exactly as before — the soul must never change what the
    // decision layer produced, only add to the document.
    UnifiedOutput out = run(hidden, text_fn);

    if (!soul_ready_ || user_turn.empty()) return out;

    out.soul = soul_.perceive(user_turn);
    out.has_soul = out.soul.has_self;
    // Re-normalise so the soul obeys the same "present iff computed" rule the
    // heads do, even if a future caller sets has_soul by hand.
    out.normalise();
    return out;
}

UnifiedOutput UnifiedPipeline::run(const float* hidden,
                                   const std::string& user_turn) const {
    return run(hidden, user_turn,
               std::function<std::string(const DomainDecision&)>());
}

} // namespace omniseed
