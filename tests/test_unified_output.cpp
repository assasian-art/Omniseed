// =============================================================================
//  OmniSeed — tests/test_unified_output.cpp
//
//  ONE JSON DOCUMENT, EVERY HEAD'S OPINION — and the pipeline that produces it.
//
//  Part A — the document rules:
//    * a field is present IFF its head ran (present-iff-it-ran);
//    * normalise() DROPS anything the activation plan does not name, and says
//      so in `warnings` — the failure being prevented is a stale value from a
//      previous turn leaking silently into this turn's answer;
//    * the JSON key order is the mandate's (router, decision, text,
//      classification, score) and the text is escaped.
//
//  Part B — the pipeline: the router picks the heads, the decision head runs
//    first, and the mandate's fast path applies afterwards.
//
//  Part C — THE FAST PATH, asserted the only way it can be: by INVOCATION
//    COUNT. A timing measurement cannot distinguish "the text head was
//    skipped" from "the text head happened to be fast", so the text producer
//    here is a counter. If the plan is decision_only the counter must stay at
//    zero — not one-call-discarded, ZERO.
//
//  Part D — the cheap heads riding along, and the "present iff it ran" rule
//    when a head is named but refuses to produce anything.
//
//  Fully offline: no model, no GGUF, no network.
// =============================================================================
#include "omniseed/core/platform.h"
#include "omniseed/unified_output.h"

#include <functional>
#include <string>
#include <vector>

using namespace omniseed;

static int g_passed = 0;
static int g_failed = 0;
static std::string g_current;

#define TEST(name) g_current = (name); platform::log_info("TEST  %s", name)
#define CHECK(cond)                                                        \
    do {                                                                   \
        if (cond) { ++g_passed; }                                          \
        else {                                                             \
            ++g_failed;                                                    \
            platform::log_error("FAIL  %s  (line %d): %s",                 \
                                g_current.c_str(), __LINE__, #cond);       \
        }                                                                  \
    } while (0)

namespace {

constexpr int32_t kE = 256;

std::vector<float> ramp(int32_t n, float phase = 0.0f) {
    std::vector<float> v(static_cast<size_t>(n), 0.0f);
    for (int32_t i = 0; i < n; ++i)
        v[static_cast<size_t>(i)] =
            std::sin(0.05f * static_cast<float>(i) + phase) * 0.8f +
            std::cos(0.13f * static_cast<float>(i)) * 0.2f;
    return v;
}

bool has_substr(const std::string& hay, const std::string& needle) {
    return hay.find(needle) != std::string::npos;
}

// Every default domain registered, every default label set registered.
void configure_all_gates_on(UnifiedPipeline& p, float v) {
    UnifiedPipeline::Config c = p.config();
    c.router.decision_gate = v;
    c.router.token_gate    = v;
    c.router.classify_gate = v;
    c.router.score_gate    = v;
    p.router().set_config(c.router);
    p.set_config(c);
}

// decision+token only, so `simple` is true and the fast path is eligible.
void configure_simple_task(UnifiedPipeline& p, float fast_path_confidence) {
    UnifiedPipeline::Config c = p.config();
    c.router.decision_gate = 0.0f;    // decision always runs
    c.router.token_gate    = 0.0f;    // token always runs
    c.router.classify_gate = 1.01f;   // classify never runs
    c.router.score_gate    = 1.01f;   // score never runs
    c.router.fast_path_confidence = fast_path_confidence;
    c.use_fast_path = true;
    p.router().set_config(c.router);
    p.set_config(c);
}

} // namespace

// =============================================================================
//  Part A — the document rules
// =============================================================================
static void part_a_document() {
    TEST("A1: a fresh document claims nothing and is consistent");
    {
        UnifiedOutput o;
        CHECK(!o.has_decision);
        CHECK(!o.has_text);
        CHECK(!o.has_classification);
        CHECK(!o.has_score);
        CHECK(!o.token_head_invoked);
        CHECK(o.warnings.empty());
        CHECK(o.consistent());
        CHECK(o.normalise() == 0);   // nothing present, nothing to drop
    }

    TEST("A2: normalise() drops a field the plan does not name, and says so");
    {
        UnifiedOutput o;
        o.router.heads = {HeadKind::Decision};
        o.router.mode = RouterMode::DecisionOnly;

        o.has_decision = true;
        o.decision.action = "buy";
        o.has_text = true;
        o.text = "stale from a previous turn";
        o.has_score = true;
        o.score.priority = 0.9f;
        o.has_classification = true;
        o.classification.domain = "language.intent";

        CHECK(!o.consistent());
        CHECK(o.normalise() == 3);       // text, score, classification
        CHECK(o.has_decision);           // named in the plan => kept
        CHECK(o.decision.action == "buy");
        CHECK(!o.has_text);
        CHECK(o.text.empty());
        CHECK(!o.has_score);
        CHECK(o.score.priority == 0.0f);
        CHECK(!o.has_classification);
        CHECK(o.classification.domain.empty());
        CHECK(o.warnings.size() == 3);
        CHECK(o.consistent());

        // Idempotent: a second pass finds nothing left to drop.
        CHECK(o.normalise() == 0);
        CHECK(o.warnings.size() == 3);
    }

    TEST("A3: a dropped field is absent from the JSON, not null");
    {
        UnifiedOutput o;
        o.router.heads = {HeadKind::Decision};
        o.router.mode = RouterMode::DecisionOnly;
        o.has_decision = true;
        o.has_text = true;
        o.text = "stale";
        o.normalise();
        const std::string j = o.to_json();
        CHECK(j.front() == '{' && j.back() == '}');
        CHECK(!has_substr(j, "\"text\":"));
        CHECK(has_substr(j, ",\"decision\":"));
        // The drop is recorded rather than hidden.
        CHECK(has_substr(j, "\"warnings\":["));
        CHECK(has_substr(j, "dropped text"));
    }

    TEST("A4: key order is the mandate's — router, decision, text, classification, score");
    {
        UnifiedOutput o;
        o.router.heads = {HeadKind::Decision, HeadKind::Token,
                          HeadKind::Classify, HeadKind::Score};
        o.router.mode = RouterMode::DecisionAndText;
        o.has_decision = true;
        o.has_text = true;
        o.text = "because";
        o.has_classification = true;
        o.classification.domain = "language.intent";
        o.has_score = true;

        const std::string j = o.to_json();
        // Search for the TOP-LEVEL keys: a leading comma distinguishes them
        // from the head names inside the router's own "activated_heads" array.
        const size_t pr = j.find("{\"router\":");
        const size_t pd = j.find(",\"decision\":");
        const size_t pt = j.find(",\"text\":");
        const size_t pc = j.find(",\"classification\":");
        const size_t ps = j.find(",\"score\":");
        CHECK(pr == 0);
        CHECK(pd != std::string::npos);
        CHECK(pt != std::string::npos);
        CHECK(pc != std::string::npos);
        CHECK(ps != std::string::npos);
        CHECK(pr < pd);
        CHECK(pd < pt);
        CHECK(pt < pc);
        CHECK(pc < ps);
    }

    TEST("A5: the text is JSON-escaped, so a quote cannot break the document");
    {
        UnifiedOutput o;
        o.router.heads = {HeadKind::Token};
        o.router.mode = RouterMode::TextOnly;
        o.has_text = true;
        o.text = "he said \"hi\"\n";
        const std::string j = o.to_json();
        CHECK(j.front() == '{' && j.back() == '}');
        CHECK(has_substr(j, "he said \\\"hi\\\"\\n"));
        CHECK(j.find('\n') == std::string::npos);   // no raw newline
    }

    TEST("A6: token_head_invoked is separate from has_text");
    {
        // The flag is the observable difference between "the producer never
        // ran" and "the producer ran and returned nothing".
        UnifiedOutput o;
        CHECK(!o.token_head_invoked);
        o.has_text = true;
        o.text = "";
        CHECK(o.has_text);
        CHECK(!o.token_head_invoked);
    }
}

// =============================================================================
//  Part B — the pipeline
// =============================================================================
static void part_b_pipeline() {
    TEST("B1: a fresh pipeline is not ready and refuses a bad width");
    {
        UnifiedPipeline p;
        CHECK(!p.ready());
        CHECK(!p.init(0));
        CHECK(!p.ready());
        CHECK(!p.error().empty());
        CHECK(!p.trained());
    }

    TEST("B2: init registers every default domain and label set");
    {
        UnifiedPipeline p;
        CHECK(p.init(kE));
        CHECK(p.ready());
        CHECK(p.router().ready());
        CHECK(p.decisions().ready());
        CHECK(p.classifier().ready());
        CHECK(p.scorer().ready());

        CHECK(p.decisions().domain_count() == 5);
        CHECK(p.classifier().label_set_count() >= 13);
        CHECK(p.classifier().find_label_set("general.routing") >= 0);
        CHECK(p.classifier().find_label_set("language.intent") >= 0);

        // Seeded, so it must admit it.
        CHECK(!p.trained());
        CHECK(has_substr(p.provenance(), "NOT FULLY FITTED"));
        CHECK(has_substr(p.provenance(), "router="));
        CHECK(has_substr(p.provenance(), "decisions="));
        CHECK(has_substr(p.provenance(), "classifier="));
        CHECK(has_substr(p.provenance(), "scorer="));
    }

    TEST("B3: a not-ready pipeline still returns a well-formed document");
    {
        UnifiedPipeline p;
        const UnifiedOutput o = p.run(static_cast<const float*>(nullptr));
        CHECK(o.router.head_count() == 2);       // the safe superset
        CHECK(has_substr(o.router.reason, "not-ready"));
        CHECK(!o.has_decision);
        CHECK(!o.has_text);
        CHECK(!o.has_classification);
        CHECK(!o.has_score);
        CHECK(o.consistent());
        const std::string j = o.to_json();
        CHECK(j.front() == '{' && j.back() == '}');
        CHECK(has_substr(j, "\"router\":"));
    }

    TEST("B4: the decision head runs first and its domain comes from the router");
    {
        UnifiedPipeline p;
        p.init(kE);
        configure_all_gates_on(p, 0.0f);
        const std::vector<float> h = ramp(kE, 0.9f);
        const UnifiedOutput o = p.run(h.data());
        CHECK(o.has_decision);
        CHECK(o.decision.domain == domain_name(o.router.domain));
        CHECK(o.router.has(HeadKind::Decision));
        CHECK(o.consistent());
    }

    TEST("B5: the document is always consistent, for every gate setting");
    {
        UnifiedPipeline p;
        p.init(kE);
        const float gates[] = {0.0f, 0.3f, 0.5f, 0.7f, 1.01f};
        for (const float g : gates) {
            configure_all_gates_on(p, g);
            for (int32_t i = 0; i < 4; ++i) {
                const std::vector<float> h = ramp(kE, 0.7f * static_cast<float>(i));
                const UnifiedOutput o = p.run(h.data());
                CHECK(o.consistent());
                // A field is never present unless its head is named.
                CHECK(!o.has_decision       || o.router.has(HeadKind::Decision));
                CHECK(!o.has_text           || o.router.has(HeadKind::Token));
                CHECK(!o.has_classification || o.router.has(HeadKind::Classify));
                CHECK(!o.has_score          || o.router.has(HeadKind::Score));
            }
        }
    }

    TEST("B6: the text producer receives the decision as prefix context");
    {
        UnifiedPipeline p;
        p.init(kE);
        configure_simple_task(p, 2.0f);   // fast path disabled => token head runs
        const std::vector<float> h = ramp(kE, 1.4f);

        DomainDecision seen;
        bool called = false;
        const std::function<std::string(const DomainDecision&)> fn =
            [&](const DomainDecision& d) -> std::string {
                seen = d;
                called = true;
                return "explaining " + d.action;
            };
        const UnifiedOutput o = p.run(h.data(), fn);
        CHECK(called);
        CHECK(o.token_head_invoked);
        CHECK(o.has_text);
        CHECK(seen.domain == o.decision.domain);
        CHECK(seen.action == o.decision.action);
        CHECK(seen.confidence == o.decision.confidence);
        CHECK(has_substr(o.text, o.decision.action));
    }

    TEST("B7: an empty text function means no text head, even if the plan names it");
    {
        UnifiedPipeline p;
        p.init(kE);
        configure_simple_task(p, 2.0f);
        const std::vector<float> h = ramp(kE, 1.4f);
        const UnifiedOutput o = p.run(h.data());
        CHECK(o.router.has(HeadKind::Token));
        CHECK(!o.has_text);
        CHECK(!o.token_head_invoked);
        CHECK(o.consistent());

        const UnifiedOutput o2 = p.run(h.data(),
                                       std::function<std::string(const DomainDecision&)>());
        CHECK(!o2.has_text);
        CHECK(!o2.token_head_invoked);
    }
}

// =============================================================================
//  Part C — the fast path, asserted by invocation count
// =============================================================================
static void part_c_fast_path() {
    TEST("C1: decision_only NEVER calls the text producer — not once, ZERO");
    {
        UnifiedPipeline p;
        p.init(kE);
        configure_simple_task(p, 0.0f);   // any confidence licenses the fast path
        const std::vector<float> h = ramp(kE, 2.3f);

        int32_t calls = 0;
        const std::function<std::string(const DomainDecision&)> fn =
            [&](const DomainDecision&) -> std::string { ++calls; return "should not happen"; };

        const UnifiedOutput o = p.run(h.data(), fn);
        CHECK(calls == 0);                       // the whole point
        CHECK(!o.token_head_invoked);
        CHECK(!o.has_text);
        CHECK(o.router.mode == RouterMode::DecisionOnly);
        CHECK(!o.router.has(HeadKind::Token));
        CHECK(o.has_decision);
        CHECK(o.consistent());
        CHECK(!has_substr(o.to_json(), "\"text\":"));
    }

    TEST("C2: a fast path that cannot fire still runs the text head exactly once");
    {
        UnifiedPipeline p;
        p.init(kE);
        configure_simple_task(p, 2.0f);   // unreachable threshold
        const std::vector<float> h = ramp(kE, 2.3f);

        int32_t calls = 0;
        const std::function<std::string(const DomainDecision&)> fn =
            [&](const DomainDecision&) -> std::string { ++calls; return "explained"; };
        const UnifiedOutput o = p.run(h.data(), fn);
        CHECK(calls == 1);
        CHECK(o.token_head_invoked);
        CHECK(o.has_text);
        CHECK(o.router.mode == RouterMode::DecisionAndText);
        CHECK(has_substr(o.router.reason, "gates fired"));
    }

    TEST("C3: use_fast_path = false disables the shortcut entirely");
    {
        UnifiedPipeline p;
        p.init(kE);
        configure_simple_task(p, 0.0f);
        UnifiedPipeline::Config c = p.config();
        c.use_fast_path = false;
        p.set_config(c);

        int32_t calls = 0;
        const std::function<std::string(const DomainDecision&)> fn =
            [&](const DomainDecision&) -> std::string { ++calls; return "explained"; };
        const UnifiedOutput o = p.run(ramp(kE, 2.3f).data(), fn);
        CHECK(calls == 1);
        CHECK(o.token_head_invoked);
        CHECK(o.router.mode == RouterMode::DecisionAndText);
    }

    TEST("C4: the fast path needs a confident decision AND a simple task");
    {
        // Confidence alone is not enough when the task also wanted a
        // classification — the plan is not "simple" and the explanation stays.
        UnifiedPipeline p;
        p.init(kE);
        UnifiedPipeline::Config c = p.config();
        c.router.decision_gate = 0.0f;
        c.router.token_gate    = 0.0f;
        c.router.classify_gate = 0.0f;    // classify rides along => NOT simple
        c.router.score_gate    = 1.01f;
        c.router.fast_path_confidence = 0.0f;
        c.use_fast_path = true;
        p.router().set_config(c.router);
        p.set_config(c);

        int32_t calls = 0;
        const std::function<std::string(const DomainDecision&)> fn =
            [&](const DomainDecision&) -> std::string { ++calls; return "explained"; };
        const UnifiedOutput o = p.run(ramp(kE, 1.9f).data(), fn);
        CHECK(!o.router.simple);
        CHECK(calls == 1);
        CHECK(o.router.mode == RouterMode::DecisionAndText);
        CHECK(o.has_classification);
    }

    TEST("C5: forcing a mode bypasses the fast path (a forced plan is not simple)");
    {
        UnifiedPipeline p;
        p.init(kE);
        configure_simple_task(p, 0.0f);
        UnifiedPipeline::Config c = p.config();
        c.force_mode = true;
        c.mode = RouterMode::DecisionAndText;
        p.set_config(c);

        int32_t calls = 0;
        const std::function<std::string(const DomainDecision&)> fn =
            [&](const DomainDecision&) -> std::string { ++calls; return "explained"; };
        const UnifiedOutput o = p.run(ramp(kE, 1.1f).data(), fn);
        CHECK(calls == 1);                       // an explicit --mode is honoured
        CHECK(o.router.mode == RouterMode::DecisionAndText);
        CHECK(has_substr(o.router.reason, "forced"));
    }

    TEST("C6: forcing decision_only drops the text head");
    {
        UnifiedPipeline p;
        p.init(kE);
        UnifiedPipeline::Config c = p.config();
        c.force_mode = true;
        c.mode = RouterMode::DecisionOnly;
        p.set_config(c);

        int32_t calls = 0;
        const std::function<std::string(const DomainDecision&)> fn =
            [&](const DomainDecision&) -> std::string { ++calls; return "no"; };
        const UnifiedOutput o = p.run(ramp(kE, 1.1f).data(), fn);
        CHECK(calls == 0);
        CHECK(o.router.mode == RouterMode::DecisionOnly);
        CHECK(o.has_decision);
        CHECK(!o.has_text);
    }

    TEST("C7: forcing text_only produces text and no decision");
    {
        UnifiedPipeline p;
        p.init(kE);
        UnifiedPipeline::Config c = p.config();
        c.force_mode = true;
        c.mode = RouterMode::TextOnly;
        p.set_config(c);

        int32_t calls = 0;
        const std::function<std::string(const DomainDecision&)> fn =
            [&](const DomainDecision&) -> std::string { ++calls; return "legacy behaviour"; };
        const UnifiedOutput o = p.run(ramp(kE, 1.1f).data(), fn);
        CHECK(calls == 1);
        CHECK(o.has_text);
        CHECK(!o.has_decision);
        CHECK(o.text == "legacy behaviour");
        CHECK(o.router.mode == RouterMode::TextOnly);
        CHECK(o.consistent());
        CHECK(!has_substr(o.to_json(), "\"decision\":"));
    }
}

// =============================================================================
//  Part D — the cheap heads, and "present iff it ran"
// =============================================================================
static void part_d_cheap_heads() {
    TEST("D1: classify and score report when the plan names them");
    {
        UnifiedPipeline p;
        p.init(kE);
        UnifiedPipeline::Config c = p.config();
        c.router.decision_gate = 0.0f;
        c.router.token_gate    = 1.01f;
        c.router.classify_gate = 0.0f;
        c.router.score_gate    = 0.0f;
        c.classification_set   = "general.routing";
        c.use_fast_path        = false;
        p.router().set_config(c.router);
        p.set_config(c);

        const UnifiedOutput o = p.run(ramp(kE, 0.6f).data());
        CHECK(o.has_decision);
        CHECK(!o.has_text);
        CHECK(o.has_classification);
        CHECK(o.has_score);
        CHECK(o.classification.domain == "general.routing");
        CHECK(!o.classification.top_k.empty());
        CHECK(o.score.matvecs == 3);
        CHECK(o.consistent());

        const std::string j = o.to_json();
        CHECK(has_substr(j, "\"classification\":"));
        CHECK(has_substr(j, "\"score\":"));
        CHECK(!has_substr(j, "\"text\":"));
    }

    TEST("D2: an empty classification_set means the head never reports");
    {
        UnifiedPipeline p;
        p.init(kE);
        UnifiedPipeline::Config c = p.config();
        c.router.decision_gate = 0.0f;
        c.router.token_gate    = 1.01f;
        c.router.classify_gate = 0.0f;
        c.router.score_gate    = 0.0f;
        c.classification_set   = "";
        p.router().set_config(c.router);
        p.set_config(c);

        const UnifiedOutput o = p.run(ramp(kE, 0.6f).data());
        CHECK(o.router.has(HeadKind::Classify));
        CHECK(!o.has_classification);
        CHECK(o.has_score);
        CHECK(o.consistent());
    }

    TEST("D3: a head that REFUSES to produce anything is not claimed as present");
    {
        // An unknown label set is a failure, and a failure leaves top_k empty.
        // Reporting it as "present" would be the same lie as reporting a
        // fabricated uniform distribution.
        UnifiedPipeline p;
        p.init(kE);
        UnifiedPipeline::Config c = p.config();
        c.router.decision_gate = 0.0f;
        c.router.token_gate    = 1.01f;
        c.router.classify_gate = 0.0f;
        c.router.score_gate    = 1.01f;
        c.classification_set   = "no.such.set";
        p.router().set_config(c.router);
        p.set_config(c);

        const UnifiedOutput o = p.run(ramp(kE, 0.6f).data());
        CHECK(o.router.has(HeadKind::Classify));
        CHECK(!o.has_classification);
        CHECK(o.consistent());
        CHECK(!has_substr(o.to_json(), "\"classification\":"));
    }

    TEST("D4: the classification set is selectable by name");
    {
        UnifiedPipeline p;
        p.init(kE);
        UnifiedPipeline::Config c = p.config();
        c.router.decision_gate = 0.0f;
        c.router.token_gate    = 1.01f;
        c.router.classify_gate = 0.0f;
        c.router.score_gate    = 1.01f;
        c.use_fast_path        = false;
        p.router().set_config(c.router);

        const char* sets[] = {"trading.regime", "language.intent", "audio.emotion",
                              "vision.scene", "general.priority"};
        for (const char* s : sets) {
            c.classification_set = s;
            p.set_config(c);
            const UnifiedOutput o = p.run(ramp(kE, 1.8f).data());
            CHECK(o.has_classification);
            CHECK(o.classification.domain == s);
            CHECK(!o.classification.top_k.empty());
        }
    }

    TEST("D5: top_k is configurable and clamped");
    {
        UnifiedPipeline p;
        p.init(kE);
        UnifiedPipeline::Config c = p.config();
        c.router.decision_gate = 0.0f;
        c.router.token_gate    = 1.01f;
        c.router.classify_gate = 0.0f;
        c.router.score_gate    = 1.01f;
        c.classification_set   = "language.intent";   // 7 labels
        c.use_fast_path        = false;
        p.router().set_config(c.router);

        c.classification_top_k = 1;
        p.set_config(c);
        CHECK(p.run(ramp(kE, 0.4f).data()).classification.top_k.size() == 1);

        c.classification_top_k = 3;
        p.set_config(c);
        CHECK(p.run(ramp(kE, 0.4f).data()).classification.top_k.size() == 3);

        c.classification_top_k = 99;
        p.set_config(c);
        CHECK(p.run(ramp(kE, 0.4f).data()).classification.top_k.size() == 7);
    }

    TEST("D6: the pipeline is deterministic and never mutates the hidden state");
    {
        UnifiedPipeline p;
        p.init(kE);
        configure_all_gates_on(p, 0.0f);
        const std::vector<float> h = ramp(kE, 3.1f);
        const std::vector<float> before = h;
        const UnifiedOutput a = p.run(h.data());
        const UnifiedOutput b = p.run(h.data());
        CHECK(a.router.mode == b.router.mode);
        CHECK(a.router.domain == b.router.domain);
        CHECK(a.decision.action == b.decision.action);
        CHECK(a.decision.confidence == b.decision.confidence);
        CHECK(a.classification.domain == b.classification.domain);
        CHECK(a.score.priority == b.score.priority);
        CHECK(h == before);
    }
}

int main() {
    platform::log_info("=== omniseed unified output + pipeline ===");
    part_a_document();
    part_b_pipeline();
    part_c_fast_path();
    part_d_cheap_heads();
    platform::log_info("=== %d passed, %d failed ===", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
