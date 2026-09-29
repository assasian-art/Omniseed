// =============================================================================
//  OmniSeed — tests/test_intent_tiebreak.cpp
//
//  §46 Part 3 — a PRE-REGISTERED utility evaluation of the fitted
//  `language.intent` head as a TIE-BREAKER alongside the deterministic lexical
//  classifier (src/language/intent.cpp).
//
//  WHY THIS QUESTION. §45 corrected the adoption floor to the majority-class
//  rate; under it `language.intent` sits at 0.6338 / 0.3239 = 1.957x — BELOW the
//  2x adoption bar, but well ABOVE a constant predictor. That is an awkward
//  place to be: not good enough to replace the lexical classifier, too good to
//  throw away. The one role that fits is a tie-breaker: use the head only where
//  the lexical rule is already unsure.
//
//  THE RULE, FIXED BEFORE THE RUN (this is the pre-registration; it is not
//  adjusted afterwards, and no threshold is swept):
//
//      m      = lexical top-1 probability - lexical top-2 probability
//               ( = top-1 alone when the lexical classifier reports one
//                 candidate, which is its most confident case )
//      TAU    = 0.10                       <-- one value, chosen in advance
//      if m >= TAU:  the LEXICAL answer stands
//      else:         the lexical answer stands UNLESS the fitted head's
//                    calibrated top-1 probability is STRICTLY GREATER than the
//                    lexical top-1 probability, in which case the head's label
//                    is taken
//
//      ADOPT iff  (combined_accuracy > lexical_only_accuracy)  AND
//                 (the gain is at least kMinGain = 2 rows)   on the holdout.
//
//  The second clause is the NOISE GUARD, and it is part of the pre-registration,
//  not a post-hoc caveat: a one-row flip on 71 rows is a coin toss (a two-sided
//  sign test over a single discordant pair gives p = 1.0). Adopting on it would
//  be §41's defect — a bar taken from the very data it judges — so both clauses
//  are stated up front and neither is relaxed afterwards.
//
//  TAU = 0.10 is not tuned. It is the round value below which the lexical
//  classifier's own header calls its answer "a coin flip among six intents"
//  (confidence is a cue-strength heuristic, not a probability). Sweeping TAU and
//  keeping the best value would be §41's defect — a bar picked from the data it
//  judges — so the diagnostic curve is REPORTED but never SELECTED from.
//
//  OBSERVED (recorded after the run; the rule above was NOT changed):
//      lexical 53/71 = 0.7465    head-only 45/71 = 0.6338   combined 54/71 = 0.7606
//      consulted 3 rows, overrode 1 (helped 1, hurt 0)  =>  gain = 1 row
//      clause 1 FIRED, clause 2 did NOT  =>  REJECT.
//  The head-only arm reproduces the D1 audit table's `language.intent` holdout
//  accuracy (0.6338), which is what makes the fixture-alignment claim testable.
//
//  Fully offline: no model, no GGUF, no network.
// =============================================================================
#include "omniseed/classification_head.h"
#include "omniseed/core/platform.h"
#include "omniseed/language/language_heads.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace omniseed;
using namespace omniseed::language;

static int g_passed = 0;
static int g_failed = 0;
static int g_skipped = 0;
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
        const double _a = (a), _b = (b), _t = (tol);                       \
        if (std::fabs(_a - _b) <= _t) { ++g_passed; }                      \
        else {                                                             \
            ++g_failed;                                                    \
            platform::log_error("FAIL  %s  (line %d): |%g - %g| > %g",     \
                                g_current.c_str(), __LINE__, _a, _b, _t);  \
        }                                                                  \
    } while (0)

// ---------------------------------------------------------------------------
// THE PRE-REGISTRATION. Frozen.
// ---------------------------------------------------------------------------
static constexpr float kTau = 0.10f;        // lexical-margin threshold
static constexpr int   kMinHoldout = 30;    // a ratio below this is noise
static constexpr int   kMinGain = 2;        // rows; a 1-row flip is a coin toss

namespace {

bool file_exists(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return static_cast<bool>(f);
}

std::vector<float> load_f32(const std::string& path, int32_t E, int32_t& rows) {
    rows = 0;
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    const std::streamoff sz = f.tellg();
    if (sz <= 0 || E <= 0) return {};
    const int64_t n = static_cast<int64_t>(sz) / (4 * E);
    if (n <= 0) return {};
    std::vector<float> out(static_cast<size_t>(n) * static_cast<size_t>(E));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(sz));
    if (!f) return {};
    rows = static_cast<int32_t>(n);
    return out;
}

// One column of a fixture TSV, indexed by the `id` column (0..n-1).
std::vector<std::string> load_tsv_column(const std::string& path,
                                         const std::string& col) {
    std::vector<std::string> out;
    std::ifstream f(path);
    if (!f) return out;
    std::string header;
    if (!std::getline(f, header)) return out;
    if (!header.empty() && header.back() == '\r') header.pop_back();
    std::vector<std::string> cols;
    {
        std::stringstream ss(header);
        std::string c;
        while (std::getline(ss, c, '\t')) cols.push_back(c);
    }
    int idx = -1;
    for (size_t i = 0; i < cols.size(); ++i)
        if (cols[i] == col) idx = static_cast<int>(i);
    if (idx < 0) return out;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::vector<std::string> fields;
        std::string c;
        while (std::getline(ss, c, '\t')) fields.push_back(c);
        if (static_cast<int>(fields.size()) > idx) out.push_back(fields[idx]);
    }
    return out;
}

// The lexical margin, defined exactly as the pre-registration states.
float lexical_margin(const IntentResult& r) {
    if (r.top_k.empty()) return 0.0f;
    const float p0 = r.top_k[0].probability;
    const float p1 = r.top_k.size() >= 2 ? r.top_k[1].probability : 0.0f;
    return p0 - p1;
}

// Exact two-sided sign test over the discordant pairs (binomial, p = 0.5).
// This is the honest way to ask "is the override doing anything?": with one
// pair helped and none hurt the statistic is a coin toss, p = 1.0.
double sign_test_two_sided(int helped, int hurt) {
    const int n = helped + hurt;
    if (n <= 0) return 1.0;
    const int k = helped > hurt ? helped : hurt;
    double tail = 0.0;
    for (int i = k; i <= n; ++i) {
        double c = 1.0;                                   // C(n, i)
        for (int j = 0; j < i; ++j)
            c *= static_cast<double>(n - j) / static_cast<double>(j + 1);
        tail += c;
    }
    const double p = 2.0 * tail / std::pow(2.0, static_cast<double>(n));
    return p > 1.0 ? 1.0 : p;
}

struct Row {
    std::string text;
    std::string truth;
    std::string lex_label;
    float       lex_p1 = 0.0f;
    float       lex_margin = 0.0f;
    std::string head_label;
    float       head_p1 = 0.0f;
    std::string combined;
};

}  // namespace

// =============================================================================
int main() {
    platform::log_info("=== omniseed intent tie-breaker evaluation (§46) ===");
    TEST("pre-registered: head as a tie-breaker under a thin lexical margin");

    const std::string bin = "models/heads/language_head.bin";
    const std::string dir = "tests/fixtures/head_calibration/language_intent";
    if (!file_exists(bin) || !file_exists(dir + "/hidden.f32") ||
        !file_exists(dir + "/labels.tsv") || !file_exists(dir + "/text.tsv")) {
        ++g_skipped;
        platform::log_info("SKIP  blob or fixture (with text.tsv) absent");
        platform::log_info("---- intent tie-breaker: %d passed, %d failed, "
                           "%d skipped ----", g_passed, g_failed, g_skipped);
        return 0;
    }

    ClassificationHead head;
    CHECK(head.load(bin));
    CHECK(head.ready());
    CHECK(head.trained());
    CHECK(head.find_label_set("language.intent") >= 0);

    int32_t n = 0;
    const std::vector<float> H = load_f32(dir + "/hidden.f32", head.hidden_size(), n);
    const std::vector<std::string> truth =
        load_tsv_column(dir + "/labels.tsv", "language.intent");
    const std::vector<std::string> texts =
        load_tsv_column(dir + "/text.tsv", "text");
    CHECK(n > 0);
    CHECK(static_cast<int32_t>(truth.size()) == n);
    CHECK(static_cast<int32_t>(texts.size()) == n);
    if (n <= 0 || static_cast<int32_t>(truth.size()) != n ||
        static_cast<int32_t>(texts.size()) != n) {
        platform::log_info("---- intent tie-breaker: fixture unreadable ----");
        return 1;
    }
    CHECK(n >= kMinHoldout);

    // ---- score every row -------------------------------------------------
    IntentClassifier lex;
    std::vector<Row> rows;
    rows.reserve(static_cast<size_t>(n));
    int lex_ok = 0, head_ok = 0, comb_ok = 0;
    int overridden = 0;              // rows where the head replaced the lexical answer
    int override_helped = 0, override_hurt = 0;
    int consulted = 0;               // rows where m < TAU (the head was consulted)

    for (int32_t i = 0; i < n; ++i) {
        Row r;
        r.text = texts[static_cast<size_t>(i)];
        r.truth = truth[static_cast<size_t>(i)];

        const IntentResult lr = lex.classify(r.text);
        r.lex_label = intent_name(lr.intent);
        r.lex_p1 = lr.top_k.empty() ? 0.0f : lr.top_k[0].probability;
        r.lex_margin = lexical_margin(lr);

        const ClassificationResult hr =
            head.classify(H.data() + static_cast<size_t>(i) * head.hidden_size(),
                          "language.intent", 1);
        r.head_label = hr.top_k.empty() ? std::string("<none>") : hr.top_k[0].label;
        r.head_p1 = hr.top_k.empty() ? 0.0f : hr.top_k[0].probability;

        // ---- THE PRE-REGISTERED RULE -------------------------------------
        r.combined = r.lex_label;
        if (r.lex_margin < kTau) {
            ++consulted;
            if (r.head_p1 > r.lex_p1) {
                r.combined = r.head_label;
                if (r.combined != r.lex_label) {
                    ++overridden;
                    if (r.combined == r.truth) ++override_helped;
                    else if (r.lex_label == r.truth) ++override_hurt;
                }
            }
        }

        if (r.lex_label == r.truth) ++lex_ok;
        if (r.head_label == r.truth) ++head_ok;
        if (r.combined == r.truth) ++comb_ok;
        rows.push_back(r);
    }

    const double lex_acc = static_cast<double>(lex_ok) / n;
    const double head_acc = static_cast<double>(head_ok) / n;
    const double comb_acc = static_cast<double>(comb_ok) / n;

    platform::log_info("      n=%d  lexical=%.4f (%d)  head-only=%.4f (%d)  "
                       "combined=%.4f (%d)", static_cast<int>(n), lex_acc, lex_ok,
                       head_acc, head_ok, comb_acc, comb_ok);
    platform::log_info("      TAU=%.2f: head consulted on %d rows, overrode %d "
                       "(helped %d, hurt %d)", kTau, consulted, overridden,
                       override_helped, override_hurt);

    // ---- the diagnostic margin curve (REPORTED, never selected from) -----
    for (float t : {0.05f, 0.10f, 0.20f, 0.30f, 0.50f, 1.01f}) {
        int ok = 0, cons = 0;
        for (const Row& r : rows) {
            std::string a = r.lex_label;
            if (r.lex_margin < t) {
                ++cons;
                if (r.head_p1 > r.lex_p1) a = r.head_label;
            }
            if (a == r.truth) ++ok;
        }
        platform::log_info("      curve: TAU=%.2f -> acc=%.4f (consulted %d)",
                           t, static_cast<double>(ok) / n, cons);
    }

    // ---- THE ADOPTION DECISION (pre-registered: two clauses) -------------
    const bool beats = comb_acc > lex_acc;
    const int  gain  = comb_ok - lex_ok;
    const bool adopt = beats && gain >= kMinGain;
    const int  discordant = override_helped + override_hurt;
    const double p_two_sided = sign_test_two_sided(override_helped, override_hurt);

    platform::log_info("      DECISION: clause 1 (combined > lexical) %s; "
                       "clause 2 (gain >= %d rows) %s  =>  %s",
                       beats ? "FIRED" : "did not fire", kMinGain,
                       gain >= kMinGain ? "FIRED" : "did not fire",
                       adopt ? "ADOPT" : "REJECT");
    platform::log_info("      overrides: %d (helped %d, hurt %d)  gain=%d row(s)  "
                       "discordant=%d  sign-test p=%.4f",
                       overridden, override_helped, override_hurt, gain,
                       discordant, p_two_sided);

    // ---- PINNED. Every number below is a fixed property of the committed
    // blob + fixture + the frozen TAU. A change in any of the three must show
    // up here rather than silently shifting a published claim. ---------------
    CHECK(n == 71);
    CHECK(lex_ok == 53);
    CHECK(head_ok == 45);
    CHECK(comb_ok == 54);
    CHECK(consulted == 3);
    CHECK(overridden == 1);
    CHECK(override_helped == 1);
    CHECK(override_hurt == 0);
    CHECK(discordant == 1);
    CHECK_NEAR(p_two_sided, 1.0, 1e-12);
    // The head-only arm reproduces the D1 audit table's `language.intent`
    // holdout accuracy — which is what makes the fixture alignment testable
    // rather than assumed.
    CHECK_NEAR(head_acc, 0.6338, 1e-4);

    // ---- THE PRE-REGISTERED OUTCOME --------------------------------------
    CHECK(beats);      // clause 1 fired: the point estimate did improve ...
    CHECK(!adopt);     // ... but clause 2 refused a one-row gain. REJECTED.
    CHECK(gain < kMinGain);

    // ---- NON-VACUITY. A rule that never consults the head proves nothing
    // about the head; the tie-breaker must actually have been exercised. ----
    CHECK(consulted >= 1);
    CHECK(overridden >= 1);

    platform::log_info("---- intent tie-breaker: %d passed, %d failed, "
                       "%d skipped ----", g_passed, g_failed, g_skipped);
    return g_failed == 0 ? 0 : 1;
}
