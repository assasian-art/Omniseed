// =============================================================================
//  OmniSeed — tests/test_router.cpp
//
//  The HeadRouter: one cheap readout of h[E] that decides WHICH heads run and
//  in WHAT MODE.
//
//  Part A — vocabulary and geometry: head/mode names, every CLI spelling, the
//    canonical head ORDER (it is part of the wire contract).
//  Part B — the plan: gates, domain softmax, fail-closed on "no gate fired",
//    forced modes, determinism, input immutability.
//  Part C — refine(): the mandate's fast path. The two invariants that matter
//    are that it can only ever DOWNGRADE (never adds a head back) and that it
//    needs BOTH a confident decision AND a simple task — either alone is not
//    enough.
//  Part D — the BUDGET. "Must run in <100 microseconds" is a claim, so it is
//    measured here rather than asserted in a comment. The measurement is
//    deliberately loose (a 20x margin over the observed cost) because a test
//    that fails on a noisy CI box teaches nothing.
//  Part E — provenance: an unfitted router must say so.
//
//  Fully offline: no model, no GGUF, no network.
// =============================================================================
#include "omniseed/core/platform.h"
#include "omniseed/core/tensor.h"
#include "omniseed/router.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
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

constexpr int32_t kE = 768;   // the real edge width, so the budget is honest

std::vector<float> ramp(int32_t n, float phase = 0.0f) {
    std::vector<float> v(static_cast<size_t>(n), 0.0f);
    for (int32_t i = 0; i < n; ++i)
        v[static_cast<size_t>(i)] =
            std::sin(0.017f * static_cast<float>(i) + phase) * 0.7f +
            std::cos(0.041f * static_cast<float>(i)) * 0.3f;
    return v;
}

std::vector<float> filled(int32_t n, float v) {
    return std::vector<float>(static_cast<size_t>(n), v);
}

bool has_substr(const std::string& hay, const std::string& needle) {
    return hay.find(needle) != std::string::npos;
}

// Every head in `sub` must be present in `super`.
bool is_subset_of(const std::vector<HeadKind>& sub, const ActivationPlan& super) {
    for (const HeadKind k : sub)
        if (!super.has(k)) return false;
    return true;
}

// Canonical HeadKind order is part of the wire contract.
bool in_canonical_order(const std::vector<HeadKind>& heads) {
    for (size_t i = 1; i < heads.size(); ++i)
        if (static_cast<int32_t>(heads[i - 1]) >= static_cast<int32_t>(heads[i])) return false;
    return true;
}

HeadRouter::Config all_gates(float v) {
    HeadRouter::Config c;
    c.decision_gate = v;
    c.token_gate    = v;
    c.classify_gate = v;
    c.score_gate    = v;
    return c;
}

} // namespace

// =============================================================================
//  Part A — vocabulary and geometry
// =============================================================================
static void part_a_vocabulary() {
    TEST("A1: head names round-trip and reject nonsense");
    CHECK(kHeadKindCount == 4);
    CHECK(std::string(head_kind_name(HeadKind::Decision)) == "decision");
    CHECK(std::string(head_kind_name(HeadKind::Token))    == "token");
    CHECK(std::string(head_kind_name(HeadKind::Classify)) == "classify");
    CHECK(std::string(head_kind_name(HeadKind::Score))    == "score");
    for (int32_t i = 0; i < kHeadKindCount; ++i) {
        HeadKind out = HeadKind::COUNT;
        CHECK(head_kind_from_name(head_kind_name(static_cast<HeadKind>(i)), out));
        CHECK(out == static_cast<HeadKind>(i));
    }
    HeadKind bad = HeadKind::Decision;
    CHECK(!head_kind_from_name("nope", bad));
    CHECK(!head_kind_from_name("unknown", bad));
    CHECK(!head_kind_from_name("", bad));

    TEST("A2: canonical mode names");
    CHECK(std::string(router_mode_name(RouterMode::DecisionOnly))    == "decision_only");
    CHECK(std::string(router_mode_name(RouterMode::DecisionAndText)) == "decision+text");
    CHECK(std::string(router_mode_name(RouterMode::TextOnly))        == "text_only");

    TEST("A3: every spelling a CLI user is likely to type is accepted");
    {
        const char* only[] = {"decision_only", "decision-only", "decisiononly"};
        const char* both[] = {"decision+text", "decision_text", "decision-text",
                              "hybrid", "both"};
        const char* text[] = {"text_only", "text-only", "textonly", "off", "text"};
        RouterMode m = RouterMode::COUNT;
        for (const char* s : only) { CHECK(router_mode_from_name(s, m)); CHECK(m == RouterMode::DecisionOnly); }
        for (const char* s : both) { CHECK(router_mode_from_name(s, m)); CHECK(m == RouterMode::DecisionAndText); }
        for (const char* s : text) { CHECK(router_mode_from_name(s, m)); CHECK(m == RouterMode::TextOnly); }
        CHECK(!router_mode_from_name("nonsense", m));
        CHECK(!router_mode_from_name("", m));
        CHECK(!router_mode_from_name("DecisionOnly", m));   // exact, case-sensitive
    }

    TEST("A4: ActivationPlan::has and heads_json");
    {
        ActivationPlan p;
        p.heads = {HeadKind::Decision, HeadKind::Token};
        CHECK(p.has(HeadKind::Decision));
        CHECK(p.has(HeadKind::Token));
        CHECK(!p.has(HeadKind::Classify));
        CHECK(!p.has(HeadKind::Score));
        CHECK(p.head_count() == 2);
        CHECK(p.heads_json() == "[\"decision\",\"token\"]");
        CHECK(p.mode_name() != nullptr);
    }

    TEST("A5: ActivationPlan JSON is well-formed and complete");
    {
        ActivationPlan p;
        p.heads = {HeadKind::Decision, HeadKind::Classify};
        p.mode = RouterMode::DecisionOnly;
        p.domain = Domain::Trading;
        p.domain_confidence = 0.6f;
        p.confidence = 0.7f;
        p.simple = true;
        p.reason = "gates fired: decision classify";
        p.matvecs = 9;
        const std::string j = p.to_json();
        CHECK(j.front() == '{' && j.back() == '}');
        CHECK(has_substr(j, "\"activated_heads\":[\"decision\",\"classify\"]"));
        CHECK(has_substr(j, "\"mode\":\"decision_only\""));
        CHECK(has_substr(j, "\"domain\":\"trading\""));
        CHECK(has_substr(j, "\"domain_confidence\":0.6000"));
        CHECK(has_substr(j, "\"confidence\":0.7000"));
        CHECK(has_substr(j, "\"simple\":true"));
        CHECK(has_substr(j, "\"reason\":\"gates fired: decision classify\""));
        CHECK(has_substr(j, "\"matvecs\":9"));
    }

    TEST("A6: geometry — rows, width, and a clean refusal on a bad width");
    {
        HeadRouter r;
        CHECK(!r.ready());
        CHECK(!r.init(0));
        CHECK(!r.ready());
        CHECK(!r.error().empty());
        CHECK(r.init(kE));
        CHECK(r.ready());
        CHECK(r.hidden_size() == kE);
        CHECK(r.total_rows() == kHeadKindCount + kDomainCount);
        CHECK(r.total_rows() == 9);
        CHECK(r.fitted_rows() == 0);
        CHECK(!r.trained());
    }
}

// =============================================================================
//  Part B — the plan
// =============================================================================
static void part_b_plan() {
    TEST("B1: a not-ready router returns the SAFE SUPERSET and says why");
    {
        HeadRouter r;
        const ActivationPlan p = r.plan(static_cast<const float*>(nullptr));
        CHECK(p.head_count() == 2);
        CHECK(p.has(HeadKind::Decision));
        CHECK(p.has(HeadKind::Token));
        CHECK(p.mode == RouterMode::DecisionAndText);
        CHECK(p.mode_name() == std::string("decision+text"));
        CHECK(!p.simple);
        CHECK(!p.reason.empty());
        CHECK(has_substr(p.reason, "not-ready"));
        CHECK(p.matvecs == 0);

        // The same holds for a null hidden state on a ready router.
        HeadRouter q;
        q.init(kE);
        const ActivationPlan qp = q.plan(static_cast<const float*>(nullptr));
        CHECK(qp.has(HeadKind::Decision));
        CHECK(qp.has(HeadKind::Token));
    }

    TEST("B2: every gate firing activates all four heads, in canonical order");
    {
        HeadRouter r;
        r.init(kE);
        r.set_config(all_gates(0.0f));
        const std::vector<float> h = ramp(kE);
        const ActivationPlan p = r.plan(h.data());
        CHECK(p.head_count() == 4);
        CHECK(p.has(HeadKind::Decision));
        CHECK(p.has(HeadKind::Token));
        CHECK(p.has(HeadKind::Classify));
        CHECK(p.has(HeadKind::Score));
        CHECK(in_canonical_order(p.heads));
        CHECK(p.mode == RouterMode::DecisionAndText);
        CHECK(!p.simple);                 // classify/score ride along => not simple
        CHECK(p.matvecs == kHeadKindCount + kDomainCount);
        CHECK(!p.reason.empty());
    }

    TEST("B3: no gate firing fails CLOSED to the superset, never to a shortcut");
    {
        // The dangerous failure would be guessing "decision only" and silently
        // suppressing an explanation the caller wanted.
        HeadRouter r;
        r.init(kE);
        r.set_config(all_gates(1.01f));   // sigmoid can never reach this
        const std::vector<float> h = ramp(kE);
        const ActivationPlan p = r.plan(h.data());
        CHECK(p.head_count() == 2);
        CHECK(p.has(HeadKind::Decision));
        CHECK(p.has(HeadKind::Token));
        CHECK(p.mode == RouterMode::DecisionAndText);
        CHECK(has_substr(p.reason, "no-gate-fired"));
        CHECK(!p.simple);
        // The domain readout still ran, so the plan is not useless.
        CHECK(p.matvecs == kHeadKindCount + kDomainCount);
    }

    TEST("B4: a decision-only gate produces decision_only, and calls it simple");
    {
        HeadRouter r;
        r.init(kE);
        HeadRouter::Config c = all_gates(1.01f);
        c.decision_gate = 0.0f;
        r.set_config(c);
        const std::vector<float> h = ramp(kE, 0.5f);
        const ActivationPlan p = r.plan(h.data());
        CHECK(p.head_count() == 1);
        CHECK(p.has(HeadKind::Decision));
        CHECK(!p.has(HeadKind::Token));
        CHECK(p.mode == RouterMode::DecisionOnly);
        CHECK(p.simple);
    }

    TEST("B5: a token-only gate produces text_only");
    {
        HeadRouter r;
        r.init(kE);
        HeadRouter::Config c = all_gates(1.01f);
        c.token_gate = 0.0f;
        r.set_config(c);
        const std::vector<float> h = ramp(kE, 0.5f);
        const ActivationPlan p = r.plan(h.data());
        CHECK(p.head_count() == 1);
        CHECK(p.has(HeadKind::Token));
        CHECK(p.mode == RouterMode::TextOnly);
        CHECK(p.simple);
    }

    TEST("B6: classify and score can ride along with decision+text");
    {
        HeadRouter r;
        r.init(kE);
        r.set_config(all_gates(0.0f));
        const std::vector<float> h = ramp(kE, 1.3f);
        const ActivationPlan p = r.plan(h.data());
        CHECK(p.mode == RouterMode::DecisionAndText);   // mode follows decision+token
        CHECK(p.has(HeadKind::Classify));
        CHECK(p.has(HeadKind::Score));
        CHECK(!p.simple);                              // ...but it is not simple
    }

    TEST("B7: the domain readout is a proper distribution, always in range");
    {
        HeadRouter r;
        r.init(kE);
        for (int32_t i = 0; i < 5; ++i) {
            const std::vector<float> hi = ramp(kE, 0.4f * static_cast<float>(i));
            const ActivationPlan p = r.plan(hi.data());
            CHECK(static_cast<int32_t>(p.domain) >= 0);
            CHECK(static_cast<int32_t>(p.domain) < kDomainCount);
            CHECK(p.domain_confidence >= 0.0f);
            CHECK(p.domain_confidence <= 1.0f);
            CHECK(p.confidence >= 0.0f);
            CHECK(p.confidence <= 1.0f);
        }
    }

    TEST("B8: the domain is decided by the probe, not by a coordinate's sign");
    {
        // A hand-set probe with a huge bias must dominate, which proves the
        // domain really is the argmax of the probe and nothing else.
        HeadRouter r;
        r.init(kE);
        const std::vector<float> zero = filled(kE, 0.0f);
        const std::vector<float> flat = filled(kE, 0.0f);
        for (int32_t d = 0; d < kDomainCount; ++d)
            r.set_domain_probe(static_cast<Domain>(d), flat.data(), -10.0f);
        r.set_domain_probe(Domain::Language, flat.data(), 10.0f);
        const ActivationPlan p = r.plan(zero.data());
        CHECK(p.domain == Domain::Language);
        CHECK(p.domain_confidence > 0.99f);
    }

    TEST("B9: a forced mode replaces the MODE only, and still reports a domain");
    {
        HeadRouter r;
        r.init(kE);
        r.set_config(all_gates(0.0f));   // unforced would say DecisionAndText
        const std::vector<float> zero = filled(kE, 0.0f);
        const std::vector<float> flat = filled(kE, 0.0f);
        r.set_domain_probe(Domain::Vision, flat.data(), 10.0f);

        const ActivationPlan only = r.plan(zero.data(), RouterMode::DecisionOnly);
        CHECK(only.head_count() == 1);
        CHECK(only.heads[0] == HeadKind::Decision);
        CHECK(only.mode == RouterMode::DecisionOnly);
        CHECK(only.simple);
        CHECK(only.domain == Domain::Vision);            // domain still computed
        CHECK(only.domain_confidence > 0.99f);
        CHECK(has_substr(only.reason, "forced"));

        const ActivationPlan text = r.plan(zero.data(), RouterMode::TextOnly);
        CHECK(text.head_count() == 1);
        CHECK(text.heads[0] == HeadKind::Token);
        CHECK(text.mode == RouterMode::TextOnly);
        CHECK(text.domain == Domain::Vision);

        const ActivationPlan both = r.plan(zero.data(), RouterMode::DecisionAndText);
        CHECK(both.head_count() == 2);
        CHECK(both.has(HeadKind::Decision));
        CHECK(both.has(HeadKind::Token));
        CHECK(!both.simple);
    }

    TEST("B10: the Tensor overload checks width and dtype");
    {
        HeadRouter r;
        r.init(kE);
        std::vector<float> h = ramp(kE);
        Tensor good("h", {static_cast<int64_t>(kE)}, DType::F32, h.data());
        const ActivationPlan gp = r.plan(good);
        CHECK(gp.matvecs == kHeadKindCount + kDomainCount);

        Tensor narrow("h", {64}, DType::F32);
        CHECK(r.plan(narrow).matvecs == 0);
        CHECK(has_substr(r.plan(narrow).reason, "not-ready"));

        Tensor half("h", {static_cast<int64_t>(kE)}, DType::F16);
        CHECK(r.plan(half).matvecs == 0);
        CHECK(has_substr(r.plan(half).reason, "not-ready"));
    }

    TEST("B11: planning is deterministic and never mutates the hidden state");
    {
        HeadRouter r;
        r.init(kE);
        const std::vector<float> h = ramp(kE, 3.7f);
        const std::vector<float> before = h;
        for (int32_t i = 0; i < 8; ++i) {
            const ActivationPlan a = r.plan(h.data());
            const ActivationPlan b = r.plan(h.data());
            CHECK(a.mode == b.mode);
            CHECK(a.domain == b.domain);
            CHECK(a.heads == b.heads);
            CHECK(a.confidence == b.confidence);
            CHECK(a.domain_confidence == b.domain_confidence);
        }
        CHECK(h == before);
    }

    TEST("B12: same seed agrees, different seed differs");
    {
        HeadRouter a;
        HeadRouter b;
        HeadRouter c;
        a.init(kE, 20240u);
        b.init(kE, 20240u);
        c.init(kE, 20241u);
        const std::vector<float> h = ramp(kE, 0.8f);
        const ActivationPlan pa = a.plan(h.data());
        const ActivationPlan pb = b.plan(h.data());
        const ActivationPlan pc = c.plan(h.data());
        CHECK(pa.confidence == pb.confidence);
        CHECK(pa.domain_confidence == pb.domain_confidence);
        CHECK(pa.domain == pb.domain);
        CHECK(pa.domain_confidence != pc.domain_confidence);
    }
}

// =============================================================================
//  Part C — refine(): the mandate's fast path
// =============================================================================
static void part_c_refine() {
    const HeadRouter::Config cfg = HeadRouter::Config();

    TEST("C1: the shared threshold really is 0.85");
    CHECK(cfg.fast_path_confidence == 0.85f);

    TEST("C2: a confident decision on a SIMPLE task drops the text head");
    {
        ActivationPlan p;
        p.heads = {HeadKind::Decision, HeadKind::Token};
        p.mode = RouterMode::DecisionAndText;
        p.simple = true;
        const ActivationPlan q = HeadRouter::refine(p, 0.90f, cfg);
        CHECK(q.mode == RouterMode::DecisionOnly);
        CHECK(q.has(HeadKind::Decision));
        CHECK(!q.has(HeadKind::Token));
        CHECK(has_substr(q.reason, "fast-path"));
    }

    TEST("C3: the boundary is >= 0.85, and 0.8499 does NOT qualify");
    {
        ActivationPlan p;
        p.heads = {HeadKind::Decision, HeadKind::Token};
        p.mode = RouterMode::DecisionAndText;
        p.simple = true;
        CHECK(HeadRouter::refine(p, 0.85f, cfg).mode == RouterMode::DecisionOnly);
        CHECK(HeadRouter::refine(p, 0.8499f, cfg).mode == RouterMode::DecisionAndText);
        CHECK(HeadRouter::refine(p, 0.0f, cfg).mode == RouterMode::DecisionAndText);
        CHECK(HeadRouter::refine(p, -1.0f, cfg).mode == RouterMode::DecisionAndText);
    }

    TEST("C4: a NON-simple task keeps its explanation no matter how sure");
    {
        // A task that also asked for a classification is not a task you can
        // answer with a bare action.
        ActivationPlan p;
        p.heads = {HeadKind::Decision, HeadKind::Token, HeadKind::Classify};
        p.mode = RouterMode::DecisionAndText;
        p.simple = false;
        const ActivationPlan q = HeadRouter::refine(p, 0.99f, cfg);
        CHECK(q.mode == RouterMode::DecisionAndText);
        CHECK(q.has(HeadKind::Token));
    }

    TEST("C5: TextOnly was ASKED for, so it cannot be silently downgraded");
    {
        ActivationPlan p;
        p.heads = {HeadKind::Token};
        p.mode = RouterMode::TextOnly;
        p.simple = true;
        const ActivationPlan q = HeadRouter::refine(p, 0.99f, cfg);
        CHECK(q.mode == RouterMode::TextOnly);
        CHECK(q.has(HeadKind::Token));
    }

    TEST("C6: DecisionOnly is already minimal and is left alone");
    {
        ActivationPlan p;
        p.heads = {HeadKind::Decision};
        p.mode = RouterMode::DecisionOnly;
        p.simple = true;
        const ActivationPlan q = HeadRouter::refine(p, 0.99f, cfg);
        CHECK(q.mode == RouterMode::DecisionOnly);
        CHECK(q.has(HeadKind::Decision));
    }

    TEST("C7: refine NEVER adds a head — it can only remove");
    {
        // This is the invariant that makes the fast path safe in both
        // directions: an uncertain router cannot talk itself into spending a
        // decode, and a confident one cannot suppress what was asked for.
        std::vector<ActivationPlan> inputs;
        {
            ActivationPlan a; a.heads = {HeadKind::Decision, HeadKind::Token};
            a.mode = RouterMode::DecisionAndText; a.simple = true;  inputs.push_back(a);
        }
        {
            ActivationPlan a; a.heads = {HeadKind::Decision};
            a.mode = RouterMode::DecisionOnly; a.simple = true;    inputs.push_back(a);
        }
        {
            ActivationPlan a; a.heads = {HeadKind::Token};
            a.mode = RouterMode::TextOnly; a.simple = false;       inputs.push_back(a);
        }
        {
            ActivationPlan a;
            a.heads = {HeadKind::Decision, HeadKind::Token, HeadKind::Classify, HeadKind::Score};
            a.mode = RouterMode::DecisionAndText; a.simple = false; inputs.push_back(a);
        }
        {
            ActivationPlan a; a.heads = {HeadKind::Decision, HeadKind::Token};
            a.mode = RouterMode::DecisionAndText; a.simple = true;  inputs.push_back(a);
        }
        const float confs[] = {-1.0f, 0.0f, 0.5f, 0.8499f, 0.85f, 0.99f, 1.0f, 2.0f};
        for (const ActivationPlan& in : inputs) {
            for (const float c : confs) {
                const ActivationPlan out = HeadRouter::refine(in, c, cfg);
                CHECK(is_subset_of(out.heads, in));
                CHECK(!out.heads.empty());       // a plan always names a head
            }
        }
    }

    TEST("C8: the threshold is configurable, not hard-coded");
    {
        HeadRouter::Config loose = cfg;
        loose.fast_path_confidence = 0.50f;
        ActivationPlan p;
        p.heads = {HeadKind::Decision, HeadKind::Token};
        p.mode = RouterMode::DecisionAndText;
        p.simple = true;
        CHECK(HeadRouter::refine(p, 0.60f, loose).mode == RouterMode::DecisionOnly);
        CHECK(HeadRouter::refine(p, 0.40f, loose).mode == RouterMode::DecisionAndText);

        HeadRouter::Config strict = cfg;
        strict.fast_path_confidence = 1.01f;
        CHECK(HeadRouter::refine(p, 1.0f, strict).mode == RouterMode::DecisionAndText);
    }

    TEST("C9: refine preserves the domain and the head order it was given");
    {
        ActivationPlan p;
        p.heads = {HeadKind::Decision, HeadKind::Token, HeadKind::Classify};
        p.mode = RouterMode::DecisionAndText;
        p.simple = true;
        p.domain = Domain::Audio;
        p.domain_confidence = 0.42f;
        p.confidence = 0.77f;
        const ActivationPlan q = HeadRouter::refine(p, 0.95f, cfg);
        CHECK(q.domain == Domain::Audio);
        CHECK(q.domain_confidence == 0.42f);
        CHECK(q.confidence == 0.77f);
        CHECK(in_canonical_order(q.heads));
    }
}

// =============================================================================
//  Part D — the budget
// =============================================================================
static void part_d_budget() {
    TEST("D1: the router plans in well under 100 microseconds");
    {
        HeadRouter r;
        r.init(kE);
        const std::vector<float> h = ramp(kE, 0.25f);

        // Warm up so the first-call page faults are not measured.
        double warm = 0.0;
        for (int32_t i = 0; i < 64; ++i) {
            const ActivationPlan p = r.plan(h.data());
            warm += p.confidence + p.matvecs;
        }
        CHECK(warm != 0.0);

        constexpr int32_t kIters = 4000;
        const double t0 = platform::now_ms();
        double sink = 0.0;
        for (int32_t i = 0; i < kIters; ++i) {
            const ActivationPlan p = r.plan(h.data());
            sink += p.confidence + p.domain_confidence + p.matvecs;
        }
        const double elapsed_us = (platform::now_ms() - t0) * 1000.0;
        const double per_call_us = elapsed_us / static_cast<double>(kIters);

        platform::log_info("  router: %.3f us/call over %d calls (sink=%.3f)",
                           per_call_us, kIters, sink);
        CHECK(sink != 0.0);                 // the loop cannot be dead-stripped
        CHECK(per_call_us < 100.0);         // the mandate's budget
        CHECK(r.last_us() >= 0.0);
        CHECK(r.last_us() < 100.0);
    }

    TEST("D2: one plan costs one matvec per head plus one per domain");
    {
        HeadRouter r;
        r.init(kE);
        const std::vector<float> h = ramp(kE, 1.5f);
        const ActivationPlan p = r.plan(h.data());
        CHECK(p.matvecs == kHeadKindCount + kDomainCount);
        CHECK(p.matvecs == 9);
    }
}

// =============================================================================
//  Part E — provenance
// =============================================================================
static void part_e_provenance() {
    TEST("E1: an unfitted router says so, loudly");
    {
        HeadRouter r;
        r.init(kE);
        CHECK(!r.trained());
        CHECK(r.fitted_rows() == 0);
        CHECK(has_substr(r.provenance(), "UNTRAINED"));
        CHECK(has_substr(r.provenance(), "0 fitted"));
    }

    TEST("E2: trained() requires EVERY row, so a half-fitted router cannot lie");
    {
        HeadRouter r;
        r.init(kE);
        const std::vector<float> row = filled(kE, 0.01f);
        for (int32_t i = 0; i < kHeadKindCount; ++i) {
            r.set_head_gate(static_cast<HeadKind>(i), row.data(), 0.0f);
            CHECK(r.fitted_rows() == i + 1);
            CHECK(!r.trained());            // still short of the domain probes
        }
        for (int32_t d = 0; d < kDomainCount; ++d) {
            r.set_domain_probe(static_cast<Domain>(d), row.data(), 0.0f);
            const bool last = (d == kDomainCount - 1);
            CHECK(r.trained() == last);
        }
        CHECK(r.fitted_rows() == r.total_rows());
    }

    TEST("E3: fitting hooks reject out-of-range and null targets");
    {
        HeadRouter r;
        r.init(kE);
        const std::vector<float> row = filled(kE, 1.0f);
        r.set_head_gate(HeadKind::COUNT, row.data(), 0.0f);   // sentinel, not a head
        r.set_head_gate(HeadKind::Decision, nullptr, 0.0f);
        r.set_domain_probe(Domain::COUNT, row.data(), 0.0f);
        r.set_domain_probe(Domain::General, nullptr, 0.0f);
        CHECK(r.fitted_rows() == 0);

        // And on a router that was never initialised.
        HeadRouter n;
        n.set_head_gate(HeadKind::Decision, row.data(), 0.0f);
        CHECK(n.fitted_rows() == 0);
    }

    TEST("E4: a fitted gate actually steers the plan");
    {
        // Proves the fitting hook writes where the hot path reads.
        HeadRouter r;
        r.init(kE);
        r.set_config(all_gates(0.5f));
        const std::vector<float> zero = filled(kE, 0.0f);
        const std::vector<float> flat = filled(kE, 0.0f);

        // Drive decision and token to ~0 and classify/score to ~1.
        r.set_head_gate(HeadKind::Decision, flat.data(), -20.0f);
        r.set_head_gate(HeadKind::Token,    flat.data(), -20.0f);
        r.set_head_gate(HeadKind::Classify, flat.data(),  20.0f);
        r.set_head_gate(HeadKind::Score,    flat.data(),  20.0f);
        const ActivationPlan p = r.plan(zero.data());
        CHECK(!p.has(HeadKind::Decision));
        CHECK(!p.has(HeadKind::Token));
        CHECK(p.has(HeadKind::Classify));
        CHECK(p.has(HeadKind::Score));
        CHECK(p.mode == RouterMode::TextOnly);
        CHECK(!p.simple);
    }
}

int main() {
    platform::log_info("=== omniseed HeadRouter: plan, refine, budget ===");
    part_a_vocabulary();
    part_b_plan();
    part_c_refine();
    part_d_budget();
    part_e_provenance();
    platform::log_info("=== %d passed, %d failed ===", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
