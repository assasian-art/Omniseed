// =============================================================================
//  OmniSeed — tests/test_heads.cpp
//
//  "One backbone, multiple heads": the shared vocabulary, the classification
//  head, the scoring head, and the domain-general decision head.
//
//  Part A — the vocabulary (heads.h): domain names, round-trips, the JSON
//    forms, and the two formatting rules that exist to keep the documents
//    parseable (escape the two characters that would break JSON; never emit a
//    bare `nan`/`inf`).
//
//  Part B — ClassificationHead: geometry, many label sets over ONE projection,
//    duplicate/empty refusal, the softmax summing to 1 over the whole set (not
//    just the top-k), the margin, and every failure path leaving top_k EMPTY.
//
//  Part C — ScoringHead: three INDEPENDENT sigmoids. The test that matters is
//    that priority and urgency can be high and low simultaneously — that is
//    precisely what a softmax could not express, so it is asserted directly
//    with hand-set rows rather than inferred.
//
//  Part D — DomainDecisionHead: N domains over one projection, the fail-closed
//    routing ladder, the 0.85 threshold shared with DecisionHead, the margin
//    floor, and the lower-casing adapter.
//
//  Fully offline: no model, no GGUF, no network, no weights on disk.
//  Same tiny harness style as test_decision_head.cpp / test_regime_engine.cpp.
// =============================================================================
#include "omniseed/classification_head.h"
#include "omniseed/core/platform.h"
#include "omniseed/core/tensor.h"
#include "omniseed/domain_decision.h"
#include "omniseed/heads.h"
#include "omniseed/scoring_head.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

using namespace omniseed;

// A domain's action-space slot is a NESTED type; alias it so the tests read
// cleanly instead of spelling DomainDecisionHead::DomainSlot everywhere.
using DomainSlot = DomainDecisionHead::DomainSlot;

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
#define CHECK_NEAR(a, b, tol)                                              \
    do {                                                                   \
        const double va_ = static_cast<double>(a);                         \
        const double vb_ = static_cast<double>(b);                         \
        if (std::fabs(va_ - vb_) <= (tol)) { ++g_passed; }                 \
        else {                                                             \
            ++g_failed;                                                    \
            platform::log_error("FAIL  %s  (line %d): %.9g != %.9g",       \
                                g_current.c_str(), __LINE__, va_, vb_);    \
        }                                                                  \
    } while (0)

namespace {

constexpr int32_t kE = 64;

// Deterministic, non-degenerate hidden state.
std::vector<float> ramp(int32_t n, float phase = 0.0f) {
    std::vector<float> v(static_cast<size_t>(n), 0.0f);
    for (int32_t i = 0; i < n; ++i)
        v[static_cast<size_t>(i)] =
            std::sin(0.37f * static_cast<float>(i) + phase) * 0.8f +
            std::cos(0.11f * static_cast<float>(i)) * 0.2f;
    return v;
}

std::vector<float> filled(int32_t n, float v) {
    return std::vector<float>(static_cast<size_t>(n), v);
}

bool has_substr(const std::string& hay, const std::string& needle) {
    return hay.find(needle) != std::string::npos;
}

// A very small structural JSON sanity check: balanced braces/brackets outside
// of strings, and no bare NaN/Inf tokens. Catches the formatting failures that
// actually break a parser.
bool json_looks_valid(const std::string& s) {
    if (s.empty() || s.front() != '{' || s.back() != '}') return false;
    int depth_brace = 0;
    int depth_brack = 0;
    bool in_str = false;
    for (size_t i = 0; i < s.size(); ++i) {
        const char c = s[i];
        if (in_str) {
            if (c == '\\') { ++i; continue; }
            if (c == '"') in_str = false;
            continue;
        }
        if (c == '"') { in_str = true; continue; }
        if (c == '{') ++depth_brace;
        else if (c == '}') { if (--depth_brace < 0) return false; }
        else if (c == '[') ++depth_brack;
        else if (c == ']') { if (--depth_brack < 0) return false; }
    }
    if (in_str || depth_brace != 0 || depth_brack != 0) return false;
    if (has_substr(s, ":nan") || has_substr(s, ":inf") ||
        has_substr(s, ":-inf") || has_substr(s, ":NaN")) return false;
    return true;
}

} // namespace

// =============================================================================
//  Part A — the shared vocabulary
// =============================================================================
static void part_a_vocabulary() {
    TEST("A1: Domain has the five names the mandate requires, General == 0");
    CHECK(kDomainCount == 5);
    CHECK(static_cast<int32_t>(Domain::General) == 0);
    CHECK(std::string(domain_name(Domain::General))  == "general");
    CHECK(std::string(domain_name(Domain::Trading))  == "trading");
    CHECK(std::string(domain_name(Domain::Language)) == "language");
    CHECK(std::string(domain_name(Domain::Vision))   == "vision");
    CHECK(std::string(domain_name(Domain::Audio))    == "audio");

    TEST("A2: domain_from_name round-trips every name");
    for (int32_t i = 0; i < kDomainCount; ++i) {
        const Domain d = static_cast<Domain>(i);
        CHECK(domain_from_name(domain_name(d)) == d);
        CHECK(domain_known(domain_name(d)));
    }

    TEST("A3: an unrecognised domain name is general, never a specialist");
    // A typo must not be able to route a request to the trading head.
    CHECK(domain_from_name("tradng") == Domain::General);
    CHECK(domain_from_name("") == Domain::General);
    CHECK(domain_from_name("TRADING") == Domain::General);   // names are exact
    CHECK(!domain_known("tradng"));
    CHECK(!domain_known(""));

    TEST("A4: ClassificationResult JSON carries every field");
    {
        ClassificationResult r;
        r.domain = "language.intent";
        r.top_k.push_back(LabelProb{"question", 0.7f});
        r.top_k.push_back(LabelProb{"statement", 0.2f});
        r.margin = 0.5f;
        r.matvecs = 7;
        const std::string j = r.to_json();
        CHECK(json_looks_valid(j));
        CHECK(has_substr(j, "\"domain\":\"language.intent\""));
        CHECK(has_substr(j, "\"top_k\":["));
        CHECK(has_substr(j, "{\"label\":\"question\",\"probability\":0.7000}"));
        CHECK(has_substr(j, "{\"label\":\"statement\",\"probability\":0.2000}"));
        CHECK(has_substr(j, "\"margin\":0.5000"));
        CHECK(has_substr(j, "\"matvecs\":7"));
    }

    TEST("A5: an empty top_k is a legal, well-formed document");
    {
        ClassificationResult r;
        r.domain = "<error: unknown label set>";
        const std::string j = r.to_json();
        CHECK(json_looks_valid(j));
        CHECK(has_substr(j, "\"top_k\":[]"));
        CHECK(has_substr(j, "<error: unknown label set>"));
    }

    TEST("A6: ScoreResult JSON carries all three axes + matvecs");
    {
        ScoreResult s;
        s.priority = 0.9f;
        s.urgency = 0.1f;
        s.confidence = 0.5f;
        s.matvecs = 3;
        const std::string j = s.to_json();
        CHECK(json_looks_valid(j));
        CHECK(has_substr(j, "\"priority\":0.9000"));
        CHECK(has_substr(j, "\"urgency\":0.1000"));
        CHECK(has_substr(j, "\"confidence\":0.5000"));
        CHECK(has_substr(j, "\"matvecs\":3"));
    }

    TEST("A7: a NaN never reaches the wire");
    // A bare `nan` makes the document invalid, and a downstream parse failure
    // is a far worse outcome than a 0.0 the caller can see is a default.
    {
        const float nan_v = std::nanf("");
        const float inf_v = std::numeric_limits<float>::infinity();
        ClassificationResult r;
        r.top_k.push_back(LabelProb{"a", nan_v});
        r.top_k.push_back(LabelProb{"b", inf_v});
        r.margin = -inf_v;
        CHECK(json_looks_valid(r.to_json()));
        CHECK(has_substr(r.to_json(), "\"probability\":0.0"));

        ScoreResult s;
        s.priority = nan_v;
        s.urgency = inf_v;
        s.confidence = -inf_v;
        CHECK(json_looks_valid(s.to_json()));
        CHECK(has_substr(s.to_json(), "\"priority\":0.0"));
    }

    TEST("A8: JSON escaping handles quotes, backslashes and control chars");
    {
        ClassificationResult r;
        r.domain = std::string("a\"b\\c");
        r.top_k.push_back(LabelProb{std::string("x\ny\tz"), 0.5f});
        const std::string j = r.to_json();
        CHECK(json_looks_valid(j));
        CHECK(has_substr(j, "a\\\"b\\\\c"));
        CHECK(has_substr(j, "x\\ny\\tz"));
        CHECK(!has_substr(j, "\"x\ny"));   // no raw newline inside the string
    }
}

// =============================================================================
//  Part B — ClassificationHead
// =============================================================================
static void part_b_classification() {
    TEST("B1: a fresh head is not ready and invents no distribution");
    {
        ClassificationHead h;
        CHECK(!h.ready());
        CHECK(h.hidden_size() == 0);
        const std::vector<float> hid = ramp(kE);
        const ClassificationResult r = h.classify(hid.data(), "anything");
        CHECK(r.top_k.empty());                       // EMPTY, not uniform
        CHECK(r.matvecs == 0);
        CHECK(has_substr(r.domain, "<error: "));
    }

    TEST("B2: init refuses a non-positive width and reports why");
    {
        ClassificationHead h;
        CHECK(!h.init(0));
        CHECK(!h.ready());
        CHECK(!h.error().empty());
        CHECK(h.init(-4) == false);
        CHECK(h.init(kE));
        CHECK(h.ready());
        CHECK(h.hidden_size() == kE);
    }

    TEST("B3: label sets register, and duplicates are REFUSED not overwritten");
    {
        ClassificationHead h;
        CHECK(h.init(kE));
        CHECK(h.total_labels() == 0);
        CHECK(h.label_set_count() == 0);
        CHECK(!h.trained());   // no rows at all yet

        const int32_t a = h.add_label_set("t.one", {"alpha", "beta", "gamma"});
        CHECK(a == 0);
        CHECK(h.label_set_count() == 1);
        CHECK(h.total_labels() == 3);
        CHECK(h.label_set(0).name == "t.one");
        CHECK(h.label_set(0).labels.size() == 3);
        CHECK(h.label_set(0).labels[0] == "alpha");
        CHECK(h.label_set(0).offset == 0);

        // Duplicate name: refused, and the FIRST registration survives intact.
        CHECK(h.add_label_set("t.one", {"x", "y"}) == -1);
        CHECK(h.label_set_count() == 1);
        CHECK(h.total_labels() == 3);
        CHECK(h.label_set(0).labels[0] == "alpha");

        // Empty label list: refused.
        CHECK(h.add_label_set("t.empty", {}) == -1);
        CHECK(h.label_set_count() == 1);

        // A second set takes the NEXT slice of the same projection.
        const int32_t b = h.add_label_set("t.two", {"p", "q"});
        CHECK(b == 1);
        CHECK(h.label_set(1).offset == 3);
        CHECK(h.total_labels() == 5);
    }

    TEST("B4: add_label_set before init is refused (no projection to size)");
    {
        ClassificationHead h;
        CHECK(h.add_label_set("t", {"a", "b"}) == -1);
        CHECK(h.total_labels() == 0);
    }

    TEST("B5: find_label_set is exact");
    {
        ClassificationHead h;
        h.init(kE);
        h.add_label_set("language.intent", {"a", "b"});
        CHECK(h.find_label_set("language.intent") == 0);
        CHECK(h.find_label_set("language.intents") == -1);
        CHECK(h.find_label_set("") == -1);
        CHECK(h.find_label_set("Language.Intent") == -1);
    }

    TEST("B6: a classification is a PROPER distribution over the whole set");
    {
        ClassificationHead h;
        h.init(kE);
        h.add_label_set("t.five", {"a", "b", "c", "d", "e"});
        const std::vector<float> hid = ramp(kE);

        // top_k >= L returns every label, so the full distribution is visible.
        const ClassificationResult full = h.classify(hid.data(), "t.five", 99);
        CHECK(full.top_k.size() == 5);
        CHECK(full.matvecs == 5);
        float sum = 0.0f;
        for (const LabelProb& lp : full.top_k) {
            CHECK(lp.probability > 0.0f);
            CHECK(lp.probability <= 1.0f);
            sum += lp.probability;
        }
        CHECK_NEAR(sum, 1.0, 1e-4);

        // Sorted descending, so top_k[0] really is the argmax.
        for (size_t i = 1; i < full.top_k.size(); ++i)
            CHECK(full.top_k[i - 1].probability >= full.top_k[i].probability);

        // margin == p(top) - p(second).
        CHECK_NEAR(full.margin, full.top_k[0].probability - full.top_k[1].probability, 1e-5);
    }

    TEST("B7: top_k clamps to [1, L] and never yields an empty success");
    {
        ClassificationHead h;
        h.init(kE);
        h.add_label_set("t.three", {"a", "b", "c"});
        const std::vector<float> hid = ramp(kE, 0.3f);
        CHECK(h.classify(hid.data(), "t.three", 1).top_k.size() == 1);
        CHECK(h.classify(hid.data(), "t.three", 2).top_k.size() == 2);
        CHECK(h.classify(hid.data(), "t.three", 3).top_k.size() == 3);
        CHECK(h.classify(hid.data(), "t.three", 99).top_k.size() == 3);   // > L clamps
        CHECK(h.classify(hid.data(), "t.three", 0).top_k.size() == 1);    // < 1 clamps
        CHECK(h.classify(hid.data(), "t.three", -5).top_k.size() == 1);
    }

    TEST("B8: the two overloads agree, and the prefix of top-k is stable");
    {
        ClassificationHead h;
        h.init(kE);
        h.add_label_set("t.three", {"a", "b", "c"});
        const std::vector<float> hid = ramp(kE, 1.1f);
        const ClassificationResult by_name = h.classify(hid.data(), "t.three", 3);
        const ClassificationResult by_idx  = h.classify(hid.data(), 0, 3);
        CHECK(by_name.top_k.size() == by_idx.top_k.size());
        for (size_t i = 0; i < by_name.top_k.size(); ++i) {
            CHECK(by_name.top_k[i].label == by_idx.top_k[i].label);
            CHECK_NEAR(by_name.top_k[i].probability, by_idx.top_k[i].probability, 1e-7);
        }
        // Asking for fewer must not change the ranking.
        const ClassificationResult one = h.classify(hid.data(), "t.three", 1);
        CHECK(one.top_k[0].label == by_name.top_k[0].label);
        CHECK_NEAR(one.margin, by_name.margin, 1e-7);
    }

    TEST("B9: every failure leaves top_k EMPTY rather than fabricating a guess");
    {
        ClassificationHead h;
        h.init(kE);
        h.add_label_set("t.two", {"a", "b"});
        const std::vector<float> hid = ramp(kE);

        const ClassificationResult unknown = h.classify(hid.data(), "no.such.set");
        CHECK(unknown.top_k.empty());
        CHECK(unknown.matvecs == 0);
        CHECK(has_substr(unknown.domain, "<error: "));

        CHECK(h.classify(hid.data(), 7, 3).top_k.empty());     // bad index
        CHECK(h.classify(hid.data(), -1, 3).top_k.empty());
        CHECK(h.classify(nullptr, "t.two", 3).top_k.empty());  // null hidden

        // A registered set name can never begin with '<', so an empty top_k is
        // an unambiguous failure channel.
        for (int32_t i = 0; i < h.label_set_count(); ++i)
            CHECK(h.label_set(i).name.empty() || h.label_set(i).name[0] != '<');
    }

    TEST("B10: the Tensor overload checks width and dtype");
    {
        ClassificationHead h;
        h.init(kE);
        h.add_label_set("t.two", {"a", "b"});
        std::vector<float> hid = ramp(kE);

        Tensor good("h", {static_cast<int64_t>(kE)}, DType::F32, hid.data());
        const ClassificationResult ok = h.classify(good, "t.two", 2);
        CHECK(ok.top_k.size() == 2);
        CHECK(ok.matvecs == 2);

        Tensor narrow("h", {32}, DType::F32);
        const ClassificationResult bad_w = h.classify(narrow, "t.two", 2);
        CHECK(bad_w.top_k.empty());
        CHECK(has_substr(bad_w.domain, "width"));

        Tensor half("h", {static_cast<int64_t>(kE)}, DType::F16);
        const ClassificationResult bad_d = h.classify(half, "t.two", 2);
        CHECK(bad_d.top_k.empty());
        CHECK(has_substr(bad_d.domain, "f32"));
    }

    TEST("B11: classify is deterministic and never mutates the hidden state");
    {
        ClassificationHead h;
        h.init(kE);
        h.add_label_set("t.four", {"a", "b", "c", "d"});
        const std::vector<float> hid = ramp(kE, 2.2f);
        const std::vector<float> before = hid;

        const ClassificationResult r1 = h.classify(hid.data(), "t.four", 4);
        const ClassificationResult r2 = h.classify(hid.data(), "t.four", 4);
        for (size_t i = 0; i < r1.top_k.size(); ++i) {
            CHECK(r1.top_k[i].label == r2.top_k[i].label);
            CHECK(r1.top_k[i].probability == r2.top_k[i].probability);
        }
        CHECK(r1.margin == r2.margin);
        CHECK(hid == before);
    }

    TEST("B12: two heads with the same seed agree; a different seed does not");
    {
        ClassificationHead a;
        ClassificationHead b;
        ClassificationHead c;
        a.init(kE, 777u);
        b.init(kE, 777u);
        c.init(kE, 778u);
        a.add_label_set("t.four", {"w", "x", "y", "z"});
        b.add_label_set("t.four", {"w", "x", "y", "z"});
        c.add_label_set("t.four", {"w", "x", "y", "z"});
        const std::vector<float> hid = ramp(kE, 0.7f);

        const ClassificationResult ra = a.classify(hid.data(), "t.four", 4);
        const ClassificationResult rb = b.classify(hid.data(), "t.four", 4);
        const ClassificationResult rc = c.classify(hid.data(), "t.four", 4);
        for (size_t i = 0; i < ra.top_k.size(); ++i) {
            CHECK(ra.top_k[i].probability == rb.top_k[i].probability);
            CHECK(ra.top_k[i].label == rb.top_k[i].label);
        }
        bool differs = false;
        for (size_t i = 0; i < ra.top_k.size(); ++i)
            if (ra.top_k[i].probability != rc.top_k[i].probability) differs = true;
        CHECK(differs);
    }

    TEST("B13: seeded weights are per-row, so registration ORDER is irrelevant");
    {
        // Row weights are keyed on (seed, set name, label name), not on the
        // row index, so adding a set later cannot change an earlier set's rows.
        ClassificationHead a;
        ClassificationHead b;
        a.init(kE);
        b.init(kE);
        a.add_label_set("first", {"m", "n"});
        a.add_label_set("second", {"o", "p"});
        b.add_label_set("second", {"o", "p"});   // registered FIRST here
        b.add_label_set("first", {"m", "n"});

        const std::vector<float> hid = ramp(kE, 0.9f);
        const ClassificationResult a_first  = a.classify(hid.data(), "first", 2);
        const ClassificationResult b_first  = b.classify(hid.data(), "first", 2);
        for (size_t i = 0; i < a_first.top_k.size(); ++i) {
            CHECK(a_first.top_k[i].label == b_first.top_k[i].label);
            CHECK(a_first.top_k[i].probability == b_first.top_k[i].probability);
        }
    }

    TEST("B14: provenance reports UNTRAINED, and fitting flips trained()");
    {
        ClassificationHead h;
        h.init(kE);
        h.add_label_set("t.two", {"a", "b"});
        CHECK(!h.trained());
        CHECK(h.fitted_rows() == 0);
        CHECK(has_substr(h.provenance(), "UNTRAINED"));

        const std::vector<float> row = filled(kE, 0.01f);
        h.set_label_row(0, 0, row.data(), 0.0f);
        CHECK(h.fitted_rows() == 1);
        CHECK(!h.trained());                       // 1 of 2 is not trained
        h.set_label_row(0, 1, row.data(), 0.0f);
        CHECK(h.fitted_rows() == 2);
        CHECK(h.trained());

        // Re-fitting the same row must not double-count.
        h.set_label_row(0, 1, row.data(), 0.0f);
        CHECK(h.fitted_rows() == 2);
    }

    TEST("B15: fitting hooks reject out-of-range targets instead of writing OOB");
    {
        ClassificationHead h;
        h.init(kE);
        h.add_label_set("t.two", {"a", "b"});
        const std::vector<float> row = filled(kE, 1.0f);
        h.set_label_row(9, 0, row.data(), 0.0f);     // bad set
        h.set_label_row(0, 9, row.data(), 0.0f);     // bad label
        h.set_label_row(-1, 0, row.data(), 0.0f);
        h.set_label_row(0, 0, nullptr, 0.0f);        // null row
        CHECK(h.fitted_rows() == 0);
    }

    TEST("B16: a hand-set row makes the head report exactly what was fitted");
    {
        ClassificationHead h;
        h.init(kE);
        h.add_label_set("t.two", {"yes", "no"});
        const std::vector<float> hid = filled(kE, 1.0f);
        // With a zero hidden state only the bias matters, which makes the
        // expected distribution exact rather than merely plausible.
        const std::vector<float> zero = filled(kE, 0.0f);
        h.set_label_row(0, 0, filled(kE, 0.0f).data(), 4.0f);
        h.set_label_row(0, 1, filled(kE, 0.0f).data(), -4.0f);
        const ClassificationResult r = h.classify(zero.data(), "t.two", 2);
        CHECK(r.top_k[0].label == "yes");
        CHECK(r.top_k[0].probability > 0.99f);
        CHECK(r.top_k[1].label == "no");
        CHECK(r.margin > 0.99f);
        (void)hid;
    }

    TEST("B17: the default label sets cover every domain the mandate names");
    {
        ClassificationHead h;
        h.init(kE);
        const int32_t added = add_default_label_sets(h);
        CHECK(added == static_cast<int32_t>(default_domain_label_sets().size()));
        CHECK(added >= 13);

        const char* required[] = {
            "language.intent", "language.language", "language.sentiment",
            "language.task", "vision.scene", "vision.anomaly", "audio.wake",
            "audio.emotion", "audio.speaker", "trading.regime",
            "trading.action", "general.routing", "general.priority"};
        for (const char* name : required) CHECK(h.find_label_set(name) >= 0);

        // Idempotent: a second call adds nothing and duplicates nothing.
        CHECK(add_default_label_sets(h) == 0);
        CHECK(h.label_set_count() == added);
    }

    TEST("B18: the trading.regime label set matches the engine's vocabulary");
    {
        // The classifier and the regime engine must not be able to disagree
        // about what the four regime labels are.
        ClassificationHead h;
        h.init(kE);
        add_default_label_sets(h);
        const int32_t i = h.find_label_set("trading.regime");
        CHECK(i >= 0);
        const std::vector<std::string> want = {"trend_up", "trend_down", "range", "high_vol"};
        CHECK(h.label_set(i).labels == want);
    }

    TEST("B19: every default label set is usable and classifies to its own labels");
    {
        ClassificationHead h;
        h.init(kE);
        add_default_label_sets(h);
        const std::vector<float> hid = ramp(kE, 1.7f);
        for (int32_t s = 0; s < h.label_set_count(); ++s) {
            const ClassificationResult r = h.classify(hid.data(), s, 3);
            CHECK(!r.top_k.empty());
            CHECK(r.matvecs == static_cast<int32_t>(h.label_set(s).labels.size()));
            CHECK(r.domain == h.label_set(s).name);
            for (const LabelProb& lp : r.top_k)
                CHECK(std::find(h.label_set(s).labels.begin(), h.label_set(s).labels.end(),
                                lp.label) != h.label_set(s).labels.end());
        }
    }
}

// =============================================================================
//  Part C — ScoringHead
// =============================================================================
static void part_c_scoring() {
    TEST("C1: a fresh head scores nothing rather than inventing a score");
    {
        ScoringHead s;
        CHECK(!s.ready());
        const std::vector<float> hid = ramp(kE);
        const ScoreResult r = s.score(hid.data());
        CHECK(r.matvecs == 0);
        CHECK(r.priority == 0.0f);
        CHECK(r.urgency == 0.0f);
        CHECK(r.confidence == 0.0f);
        CHECK(!s.trained());
    }

    TEST("C2: init refuses a non-positive width");
    {
        ScoringHead s;
        CHECK(!s.init(0));
        CHECK(!s.ready());
        CHECK(!s.error().empty());
        CHECK(s.init(kE));
        CHECK(s.ready());
        CHECK(s.hidden_size() == kE);
        CHECK(ScoringHead::ROW_COUNT == 3);
    }

    TEST("C3: three sigmoids, all in [0,1], for three matvecs");
    {
        ScoringHead s;
        s.init(kE);
        const std::vector<float> hid = ramp(kE, 0.4f);
        const ScoreResult r = s.score(hid.data());
        CHECK(r.matvecs == 3);
        CHECK(r.priority   >= 0.0f && r.priority   <= 1.0f);
        CHECK(r.urgency    >= 0.0f && r.urgency    <= 1.0f);
        CHECK(r.confidence >= 0.0f && r.confidence <= 1.0f);
    }

    TEST("C4: the axes are INDEPENDENT — high priority with low urgency");
    {
        // This is the test a softmax could not pass. With hand-set rows the
        // three values are exact, so there is nothing to infer.
        ScoringHead s;
        s.init(kE);
        const std::vector<float> zero = filled(kE, 0.0f);
        const std::vector<float> flat = filled(kE, 0.0f);
        s.set_row(ScoringHead::Priority,   flat.data(),  6.0f);
        s.set_row(ScoringHead::Urgency,    flat.data(), -6.0f);
        s.set_row(ScoringHead::Confidence, flat.data(),  0.0f);

        const ScoreResult r = s.score(zero.data());
        CHECK(r.priority > 0.99f);   // urgent AND unimportant at the same time
        CHECK(r.urgency  < 0.01f);
        CHECK_NEAR(r.confidence, 0.5, 1e-4);
        CHECK(r.priority + r.urgency + r.confidence > 1.4f);   // proves NOT a softmax
    }

    TEST("C5: trained() requires all three rows, and fitting is idempotent");
    {
        ScoringHead s;
        s.init(kE);
        CHECK(!s.trained());
        CHECK(s.fitted_rows() == 0);
        CHECK(has_substr(s.provenance(), "UNTRAINED"));

        const std::vector<float> row = filled(kE, 0.02f);
        s.set_row(ScoringHead::Priority, row.data(), 0.0f);
        CHECK(s.fitted_rows() == 1);
        CHECK(!s.trained());
        s.set_row(ScoringHead::Urgency, row.data(), 0.0f);
        CHECK(!s.trained());
        s.set_row(ScoringHead::Confidence, row.data(), 0.0f);
        CHECK(s.fitted_rows() == 3);
        CHECK(s.trained());

        s.set_row(ScoringHead::Confidence, row.data(), 0.0f);
        CHECK(s.fitted_rows() == 3);   // no double count
    }

    TEST("C6: the Tensor overload checks width and dtype");
    {
        ScoringHead s;
        s.init(kE);
        std::vector<float> hid = ramp(kE);
        Tensor good("h", {static_cast<int64_t>(kE)}, DType::F32, hid.data());
        CHECK(s.score(good).matvecs == 3);
        Tensor narrow("h", {8}, DType::F32);
        CHECK(s.score(narrow).matvecs == 0);
        Tensor half("h", {static_cast<int64_t>(kE)}, DType::F16);
        CHECK(s.score(half).matvecs == 0);
    }

    TEST("C7: scoring is deterministic and never mutates the hidden state");
    {
        ScoringHead s;
        s.init(kE);
        const std::vector<float> hid = ramp(kE, 3.3f);
        const std::vector<float> before = hid;
        const ScoreResult a = s.score(hid.data());
        const ScoreResult b = s.score(hid.data());
        CHECK(a.priority == b.priority);
        CHECK(a.urgency == b.urgency);
        CHECK(a.confidence == b.confidence);
        CHECK(hid == before);
    }

    TEST("C8: same seed agrees, different seed differs");
    {
        ScoringHead a;
        ScoringHead b;
        ScoringHead c;
        a.init(kE, 555u);
        b.init(kE, 555u);
        c.init(kE, 556u);
        const std::vector<float> hid = ramp(kE, 0.2f);
        const ScoreResult ra = a.score(hid.data());
        const ScoreResult rb = b.score(hid.data());
        const ScoreResult rc = c.score(hid.data());
        CHECK(ra.priority == rb.priority && ra.urgency == rb.urgency &&
              ra.confidence == rb.confidence);
        CHECK(ra.priority != rc.priority || ra.urgency != rc.urgency ||
              ra.confidence != rc.confidence);
    }

    TEST("C9: null hidden is rejected");
    {
        ScoringHead s;
        s.init(kE);
        CHECK(s.score(static_cast<const float*>(nullptr)).matvecs == 0);
    }
}

// =============================================================================
//  Part D — DomainDecisionHead
// =============================================================================
static void part_d_domain_decision() {
    TEST("D1: a fresh head reports an ERROR, never a fabricated action");
    {
        DomainDecisionHead h;
        CHECK(!h.ready());
        const std::vector<float> hid = ramp(kE);
        const DomainDecision d = h.decide(hid.data(), Domain::Trading);
        CHECK(d.routing == "error");
        CHECK(d.abstain);
        CHECK(!d.fast_path);
        CHECK(d.matvecs == 0);
        CHECK(has_substr(d.action, "<error: "));
    }

    TEST("D2: init refuses a non-positive width");
    {
        DomainDecisionHead h;
        CHECK(!h.init(0));
        CHECK(!h.ready());
        CHECK(!h.error().empty());
        CHECK(h.init(kE));
        CHECK(h.ready());
        CHECK(h.hidden_size() == kE);
    }

    TEST("D3: domains register with their action spaces, duplicates refused");
    {
        DomainDecisionHead h;
        h.init(kE);
        const int32_t t = h.add_domain(Domain::Trading, {"abstain", "hold", "buy"});
        CHECK(t == 0);
        CHECK(h.domain_count() == 1);
        CHECK(h.total_actions() == 3);
        CHECK(h.slot(0).domain == Domain::Trading);
        CHECK(h.slot(0).abstain_index == 0);   // picked up from the vocabulary
        CHECK(h.slot(0).explain_index == -1);  // none named, none inferred

        CHECK(h.add_domain(Domain::Trading, {"a", "b"}) == -1);   // duplicate
        CHECK(h.domain_count() == 1);
        CHECK(h.total_actions() == 3);
        CHECK(h.add_domain(Domain::Language, {}) == -1);          // empty
        CHECK(h.domain_count() == 1);

        const int32_t l = h.add_domain(Domain::Language, {"abstain", "answer", "explain"});
        CHECK(l == 1);
        CHECK(h.slot(1).offset == 3);
        CHECK(h.total_actions() == 6);
        CHECK(h.slot(1).abstain_index == 0);
        CHECK(h.slot(1).explain_index == 2);
    }

    TEST("D4: explicit escape hatches are honoured, including 'none'");
    {
        DomainDecisionHead h;
        h.init(kE);
        h.add_domain(Domain::Vision, {"describe", "flag", "bail", "escalate"}, "bail", "escalate");
        CHECK(h.slot(0).abstain_index == 2);
        CHECK(h.slot(0).explain_index == 3);

        // No escape hatches named and none inferable: the slot records that,
        // and the ladder below still escalates on low confidence.
        h.add_domain(Domain::Audio, {"listen", "speak"});
        CHECK(h.slot(1).abstain_index == -1);
        CHECK(h.slot(1).explain_index == -1);
    }

    TEST("D5: find_domain is exact");
    {
        DomainDecisionHead h;
        h.init(kE);
        add_default_domains(h);
        CHECK(h.find_domain(Domain::Trading) == 0);
        CHECK(h.find_domain(Domain::Language) == 1);
        CHECK(h.find_domain(Domain::Vision) == 2);
        CHECK(h.find_domain(Domain::Audio) == 3);
        CHECK(h.find_domain(Domain::General) == 4);
    }

    TEST("D6: the default action spaces cover all five domains");
    {
        DomainDecisionHead h;
        h.init(kE);
        CHECK(add_default_domains(h) == 5);
        CHECK(add_default_domains(h) == 0);   // idempotent
        CHECK(h.domain_count() == 5);

        // Trading mirrors DecisionHead's action space exactly (lower-cased), so
        // the two heads cannot drift into disagreeing about what actions exist.
        const std::vector<std::string> want = {"abstain", "hold", "buy", "sell",
                                               "close", "hedge", "explain"};
        CHECK(h.slot(h.find_domain(Domain::Trading)).actions == want);
        CHECK(h.slot(h.find_domain(Domain::Trading)).abstain_index == 0);
        CHECK(h.slot(h.find_domain(Domain::Trading)).explain_index == 6);
    }

    TEST("D7: a decision is a proper distribution's argmax, with a valid routing");
    {
        DomainDecisionHead h;
        h.init(kE);
        add_default_domains(h);
        const std::vector<float> hid = ramp(kE, 0.6f);
        for (int32_t i = 0; i < kDomainCount; ++i) {
            const Domain d = static_cast<Domain>(i);
            const DomainDecision r = h.decide(hid.data(), d);
            const DomainSlot& slot = h.slot(h.find_domain(d));
            CHECK(r.domain == domain_name(d));
            CHECK(r.action == "<error: unregistered domain>" || r.action[0] != '<');
            if (r.routing != "error") {
                CHECK(std::find(slot.actions.begin(), slot.actions.end(), r.action) !=
                      slot.actions.end());
                CHECK(r.confidence >= 0.0f && r.confidence <= 1.0f);
                CHECK(r.margin >= 0.0f && r.margin <= 1.0f);
                CHECK(r.routing == "self" || r.routing == "system2" ||
                      r.routing == "abstain" || r.routing == "error");
                CHECK(r.matvecs == static_cast<int32_t>(slot.actions.size()));
                CHECK(r.fast_path == (r.routing == "self"));
                CHECK(r.abstain == (r.routing == "abstain" || r.routing == "error"));
            }
        }
    }

    TEST("D8: an unregistered domain is an ERROR, not a guess");
    {
        DomainDecisionHead h;
        h.init(kE);
        h.add_domain(Domain::Trading, {"abstain", "buy", "sell"});
        const std::vector<float> hid = ramp(kE);
        const DomainDecision r = h.decide(hid.data(), Domain::Vision);
        CHECK(r.routing == "error");
        CHECK(r.abstain);
        CHECK(r.matvecs == 0);
        CHECK(has_substr(r.action, "<error: "));
    }

    TEST("D9: the confidence threshold is fail-closed at exactly 1.0");
    {
        // A threshold above the maximum attainable confidence makes "self"
        // unreachable — the ladder must then never self-route.
        DomainDecisionHead h;
        h.init(kE);
        add_default_domains(h);
        h.set_threshold(1.01f);
        const std::vector<float> hid = ramp(kE, 1.9f);
        for (int32_t i = 0; i < kDomainCount; ++i) {
            const DomainDecision r = h.decide(hid.data(), static_cast<Domain>(i));
            CHECK(r.routing != "self");
            CHECK(!r.fast_path);
        }
    }

    TEST("D10: a zero threshold lets a confident argmax self-route");
    {
        DomainDecisionHead h;
        h.init(kE);
        add_default_domains(h);
        h.set_threshold(0.0f);
        const std::vector<float> hid = ramp(kE, 1.9f);
        int32_t self_routed = 0;
        for (int32_t i = 0; i < kDomainCount; ++i) {
            const DomainDecision r = h.decide(hid.data(), static_cast<Domain>(i));
            // The escape actions still short-circuit: abstain stays abstain.
            const DomainSlot& slot = h.slot(h.find_domain(static_cast<Domain>(i)));
            const int32_t ai = slot.abstain_index;
            const int32_t ei = slot.explain_index;
            if (r.action == (ai >= 0 ? slot.actions[static_cast<size_t>(ai)] : "") ||
                r.action == (ei >= 0 ? slot.actions[static_cast<size_t>(ei)] : "")) {
                CHECK(r.routing != "self");
            } else {
                CHECK(r.routing == "self");
                ++self_routed;
            }
        }
        CHECK(self_routed > 0);   // the ladder is reachable, not vacuous
    }

    TEST("D11: the margin floor blocks a near coin flip");
    {
        DomainDecisionHead h;
        h.init(kE);
        add_default_domains(h);
        h.set_threshold(0.0f);
        h.set_margin_floor(1.0f);   // margin can never reach 1.0
        const std::vector<float> hid = ramp(kE, 2.6f);
        for (int32_t i = 0; i < kDomainCount; ++i) {
            const DomainDecision r = h.decide(hid.data(), static_cast<Domain>(i));
            CHECK(r.routing != "self");
        }
    }

    TEST("D12: a hand-set action row decides exactly what was fitted");
    {
        DomainDecisionHead h;
        h.init(kE);
        h.add_domain(Domain::General, {"abstain", "yes", "no"});
        h.set_threshold(0.0f);
        const std::vector<float> zero = filled(kE, 0.0f);
        const std::vector<float> flat = filled(kE, 0.0f);
        h.set_action_row(Domain::General, 0, flat.data(), -8.0f);   // abstain
        h.set_action_row(Domain::General, 1, flat.data(),  8.0f);   // yes
        h.set_action_row(Domain::General, 2, flat.data(), -8.0f);   // no
        const DomainDecision r = h.decide(zero.data(), Domain::General);
        CHECK(r.action == "yes");
        CHECK(r.confidence > 0.99f);
        CHECK(r.margin > 0.99f);
        CHECK(r.routing == "self");
        CHECK(r.fast_path);
        CHECK(h.trained());
    }

    TEST("D13: trained() requires every action row, and fitting is idempotent");
    {
        DomainDecisionHead h;
        h.init(kE);
        add_default_domains(h);
        CHECK(!h.trained());
        CHECK(h.fitted_rows() == 0);
        CHECK(has_substr(h.provenance(), "UNTRAINED"));

        const std::vector<float> row = filled(kE, 0.01f);
        const int32_t total = h.total_actions();
        for (int32_t s = 0; s < h.domain_count(); ++s) {
            const DomainSlot& slot = h.slot(s);
            for (int32_t k = 0; k < static_cast<int32_t>(slot.actions.size()); ++k)
                h.set_action_row(slot.domain, k, row.data(), 0.0f);
        }
        CHECK(h.fitted_rows() == total);
        CHECK(h.trained());
        h.set_action_row(Domain::Trading, 0, row.data(), 0.0f);
        CHECK(h.fitted_rows() == total);
    }

    TEST("D14: fitting hooks reject out-of-range targets");
    {
        DomainDecisionHead h;
        h.init(kE);
        h.add_domain(Domain::General, {"a", "b"});
        const std::vector<float> row = filled(kE, 1.0f);
        h.set_action_row(Domain::Vision, 0, row.data(), 0.0f);   // unregistered domain
        h.set_action_row(Domain::General, 9, row.data(), 0.0f);  // bad index
        h.set_action_row(Domain::General, -1, row.data(), 0.0f);
        h.set_action_row(Domain::General, 0, nullptr, 0.0f);
        CHECK(h.fitted_rows() == 0);
    }

    TEST("D15: the adapter lower-cases the trading head's action names");
    {
        DecisionResult d;
        d.action_type      = DecisionAction::BUY;
        d.confidence_score = 0.91f;
        d.margin           = 0.4f;
        d.routing          = "self";
        d.fast_path        = true;
        const DomainDecision r = from_decision_result(d);
        CHECK(r.action == "buy");          // not "BUY"
        CHECK(r.domain == "trading");
        CHECK_NEAR(r.confidence, 0.91f, 1e-6);
        CHECK_NEAR(r.margin, 0.4f, 1e-6);
        CHECK(r.routing == "self");
        CHECK(r.fast_path);
        CHECK(!r.abstain);

        DecisionResult a;
        a.action_type = DecisionAction::ABSTAIN;
        a.routing     = "abstain";
        const DomainDecision ra = from_decision_result(a);
        CHECK(ra.action == "abstain");
        CHECK(ra.abstain);
        CHECK(!ra.fast_path);

        DecisionResult e;
        e.action_type = DecisionAction::SELL;
        e.routing     = "system2";
        const DomainDecision re = from_decision_result(e);
        CHECK(re.action == "sell");
        CHECK(!re.fast_path);
        CHECK(!re.abstain);
    }

    TEST("D16: deciding is deterministic and never mutates the hidden state");
    {
        DomainDecisionHead h;
        h.init(kE);
        add_default_domains(h);
        const std::vector<float> hid = ramp(kE, 4.1f);
        const std::vector<float> before = hid;
        for (int32_t i = 0; i < kDomainCount; ++i) {
            const Domain d = static_cast<Domain>(i);
            const DomainDecision a = h.decide(hid.data(), d);
            const DomainDecision b = h.decide(hid.data(), d);
            CHECK(a.action == b.action);
            CHECK(a.confidence == b.confidence);
            CHECK(a.margin == b.margin);
            CHECK(a.routing == b.routing);
        }
        CHECK(hid == before);
    }

    TEST("D17: same seed agrees, different seed differs");
    {
        DomainDecisionHead a;
        DomainDecisionHead b;
        DomainDecisionHead c;
        a.init(kE, 4242u);
        b.init(kE, 4242u);
        c.init(kE, 4243u);
        add_default_domains(a);
        add_default_domains(b);
        add_default_domains(c);
        const std::vector<float> hid = ramp(kE, 0.55f);
        const DomainDecision ra = a.decide(hid.data(), Domain::Trading);
        const DomainDecision rb = b.decide(hid.data(), Domain::Trading);
        const DomainDecision rc = c.decide(hid.data(), Domain::Trading);
        CHECK(ra.action == rb.action);
        CHECK(ra.confidence == rb.confidence);
        CHECK(ra.confidence != rc.confidence);
    }

    TEST("D18: the Tensor overload checks width and dtype, and null is rejected");
    {
        DomainDecisionHead h;
        h.init(kE);
        add_default_domains(h);
        std::vector<float> hid = ramp(kE);
        Tensor good("h", {static_cast<int64_t>(kE)}, DType::F32, hid.data());
        CHECK(h.decide(good, Domain::General).routing != "error");
        Tensor narrow("h", {16}, DType::F32);
        CHECK(h.decide(narrow, Domain::General).routing == "error");
        Tensor half("h", {static_cast<int64_t>(kE)}, DType::F16);
        CHECK(h.decide(half, Domain::General).routing == "error");
        CHECK(h.decide(static_cast<const float*>(nullptr), Domain::General).routing == "error");
    }

    TEST("D19: DomainDecision JSON is complete and well-formed");
    {
        DomainDecisionHead h;
        h.init(kE);
        add_default_domains(h);
        const std::vector<float> hid = ramp(kE, 0.8f);
        const std::string j = h.decide(hid.data(), Domain::Trading).to_json();
        CHECK(json_looks_valid(j));
        CHECK(has_substr(j, "\"domain\":\"trading\""));
        CHECK(has_substr(j, "\"action\":"));
        CHECK(has_substr(j, "\"confidence\":"));
        CHECK(has_substr(j, "\"margin\":"));
        CHECK(has_substr(j, "\"routing\":"));
        CHECK(has_substr(j, "\"abstain\":"));
        CHECK(has_substr(j, "\"fast_path\":"));
    }
}

// =============================================================================
//  Part E — the latency budget
//
//  The mandate's Phase 7 requires "Decision Head <1ms for ALL domains". That is
//  a claim, so it is measured per domain rather than asserted in a comment. The
//  budget is generous (a whole millisecond) precisely because the point is to
//  catch an accidental second forward pass, not to police a few microseconds.
// =============================================================================
static void part_e_latency() {
    TEST("E1: a decision is <1 ms for ALL five domains");
    {
        DomainDecisionHead h;
        h.init(kE);
        add_default_domains(h);
        const std::vector<float> hid = ramp(kE, 0.35f);

        constexpr int32_t kIters = 2000;
        double warm = 0.0;
        for (int32_t i = 0; i < 64; ++i) {
            const DomainDecision w = h.decide(hid.data(), Domain::General);
            warm += w.confidence;
        }
        CHECK(warm != 0.0);

        double sink    = 0.0;
        double worst   = 0.0;
        const char* worst_name = "";
        for (int32_t i = 0; i < kDomainCount; ++i) {
            const Domain d = static_cast<Domain>(i);
            const double t0 = platform::now_ms();
            for (int32_t k = 0; k < kIters; ++k) {
                const DomainDecision r = h.decide(hid.data(), d);
                sink += r.confidence + r.margin;
            }
            const double per_us = (platform::now_ms() - t0) * 1000.0 /
                                  static_cast<double>(kIters);
            platform::log_info("  decide(%-8s): %7.3f us/call", domain_name(d), per_us);
            if (per_us > worst) { worst = per_us; worst_name = domain_name(d); }
        }
        platform::log_info("  worst: %s at %.3f us (budget 1000 us)", worst_name, worst);
        CHECK(sink != 0.0);
        CHECK(worst < 1000.0);
    }

    TEST("E2: classification and scoring are also well inside 1 ms");
    {
        ClassificationHead c;
        c.init(kE);
        add_default_label_sets(c);
        ScoringHead s;
        s.init(kE);
        const std::vector<float> hid = ramp(kE, 0.75f);

        constexpr int32_t kIters = 2000;
        double sink = 0.0;
        for (int32_t i = 0; i < 64; ++i) {
            const ClassificationResult w = c.classify(hid.data(), "language.intent", 3);
            sink += w.margin;
        }
        CHECK(sink != 0.0);

        const double t0 = platform::now_ms();
        for (int32_t k = 0; k < kIters; ++k) {
            const ClassificationResult r = c.classify(hid.data(), "language.intent", 3);
            sink += r.margin;
        }
        const double classify_us = (platform::now_ms() - t0) * 1000.0 / kIters;

        const double t1 = platform::now_ms();
        for (int32_t k = 0; k < kIters; ++k) {
            const ScoreResult r = s.score(hid.data());
            sink += r.priority;
        }
        const double score_us = (platform::now_ms() - t1) * 1000.0 / kIters;

        platform::log_info("  classify(language.intent, 7 labels): %.3f us/call", classify_us);
        platform::log_info("  score(3 rows):                        %.3f us/call", score_us);
        CHECK(sink != 0.0);
        CHECK(classify_us < 1000.0);
        CHECK(score_us < 1000.0);
    }
}

int main() {
    platform::log_info("=== omniseed head stack: vocabulary, classification, "
                       "scoring, domain decision ===");
    part_a_vocabulary();
    part_b_classification();
    part_c_scoring();
    part_d_domain_decision();
    part_e_latency();
    platform::log_info("=== %d passed, %d failed ===", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
