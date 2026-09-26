// =============================================================================
//  OmniSeed — tests/test_decision_head.cpp
//
//  System-1 decision head ("two heads, one brain").
//
//  Part A — correctness (fully offline, no model, no GGUF):
//    action space + labels, geometry, one-matvec contract, softmax/argmax,
//    fail-closed routing (ABSTAIN and EXPLAIN can never self-route), the
//    confidence and margin gates, determinism, input immutability,
//    persistence round-trip, and every rejection path.
//
//  Part B — the speed claim:
//    System-1 costs ONE matvec [A,E]x[E] + a softmax over A.
//    System-2 costs max_new_tokens x (a full RWKV forward). Even if we grant
//    System-2 the most charitable possible baseline — a bare output-head
//    matvec [V,E] per token and NOTHING else, no attention, no FFN, no
//    channel mix — the decision head must still win by >= 100x. Both sides
//    are measured with the same clock and the same scalar fp32 inner loop,
//    so the comparison is like-for-like rather than a model of a model.
//
//  Same tiny harness style as test_platform.cpp / test_trading.cpp.
// =============================================================================
#include "omniseed/core/platform.h"
#include "omniseed/decision_head.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// Consumed by a CHECK so the benchmark loops below cannot be dead-stripped.
static double g_sink = 0.0;

namespace {

constexpr int32_t kActionCount = static_cast<int32_t>(DecisionAction::COUNT);

std::string temp_path(const char* name) {
    const char* base = std::getenv("TEMP");
    if (base == nullptr || *base == '\0') base = std::getenv("TMPDIR");
    if (base == nullptr || *base == '\0') base = ".";
    std::string p = base;
    if (!p.empty() && p.back() != '/' && p.back() != '\\') p += '/';
    p += name;
    return p;
}

std::vector<float> filled(int32_t n, float v) {
    return std::vector<float>(static_cast<size_t>(n), v);
}

bool write_bytes(const std::string& path, const std::vector<uint8_t>& bytes) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) return false;
    const bool ok = bytes.empty() ||
                    std::fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size();
    std::fclose(f);
    return ok;
}

// ---------------------------------------------------------------------------
// Baseline: ONE System-2 token step, at its most charitable.
//
// This is the bare output-head projection [V, E] -> [V] — the absolute floor
// on what any autoregressive token step must do. Real System-2 additionally
// runs n_layers x (attention + FFN + channel mix), so the ratio measured here
// is a LOWER BOUND on the true speedup.
//
// Same scalar fp32 MAC loop the decision head uses, so neither side gets an
// unfair kernel advantage.
// ---------------------------------------------------------------------------
double bench_head_matvec_ns(int64_t V, int64_t E, int32_t iters) {
    std::vector<float> W(static_cast<size_t>(V) * static_cast<size_t>(E));
    std::vector<float> x(static_cast<size_t>(E), 0.0f);
    std::vector<float> y(static_cast<size_t>(V), 0.0f);

    uint64_t s = 0x123456789abcdefull;
    for (float& w : W) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        w = static_cast<float>((s >> 40) & 0x3ffu) / 512.0f - 1.0f;
    }
    for (float& v : x) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        v = static_cast<float>((s >> 40) & 0x3ffu) / 512.0f - 1.0f;
    }

    float acc = 0.0f;
    const double t0 = platform::now_ms();
    for (int32_t it = 0; it < iters; ++it) {
        for (int64_t r = 0; r < V; ++r) {
            const float* row = W.data() + r * E;
            float a = 0.0f;
            for (int64_t i = 0; i < E; ++i) a += row[i] * x[static_cast<size_t>(i)];
            y[static_cast<size_t>(r)] = a;
        }
        acc += y[static_cast<size_t>(it) % static_cast<size_t>(V)];
    }
    const double t1 = platform::now_ms();
    g_sink += static_cast<double>(acc);
    return (t1 - t0) * 1e6 / static_cast<double>(iters);
}

// Average ns of one decide() call, warm (scratch already allocated).
double bench_decide_ns(const DecisionHead& head, const std::vector<float>& h,
                       int32_t iters) {
    DecisionResult last;
    const double t0 = platform::now_ms();
    for (int32_t i = 0; i < iters; ++i) last = head.decide(h.data());
    const double t1 = platform::now_ms();
    g_sink += static_cast<double>(last.confidence_score);
    return (t1 - t0) * 1e6 / static_cast<double>(iters);
}

} // namespace

int main() {
    platform::log_info("---- decision head tests ----");

    // =======================================================================
    // Part A — correctness
    // =======================================================================

    TEST("action space + labels");
    {
        CHECK(kActionCount == 7);
        CHECK(std::strcmp(decision_action_name(DecisionAction::ABSTAIN), "ABSTAIN") == 0);
        CHECK(std::strcmp(decision_action_name(DecisionAction::HOLD),    "HOLD")    == 0);
        CHECK(std::strcmp(decision_action_name(DecisionAction::BUY),     "BUY")     == 0);
        CHECK(std::strcmp(decision_action_name(DecisionAction::SELL),    "SELL")    == 0);
        CHECK(std::strcmp(decision_action_name(DecisionAction::CLOSE),   "CLOSE")   == 0);
        CHECK(std::strcmp(decision_action_name(DecisionAction::HEDGE),   "HEDGE")   == 0);
        CHECK(std::strcmp(decision_action_name(DecisionAction::EXPLAIN), "EXPLAIN") == 0);
        // The sentinel is not a real action and must not claim to be one.
        CHECK(std::strcmp(decision_action_name(DecisionAction::COUNT), "UNKNOWN") == 0);
    }

    TEST("geometry + construction");
    {
        DecisionHead h;
        CHECK(!h.ready());
        CHECK(h.decide(nullptr).routing == "error");   // not ready yet

        CHECK(h.init(32, 7u));
        CHECK(h.ready());
        CHECK(h.hidden_size() == 32);
        CHECK(h.action_count() == kActionCount);
        CHECK(h.threshold() == 0.85f);
        CHECK(!h.trained());
        CHECK(h.fitted_actions() == 0);

        // A head sized from the model config must match the config width.
        RwkvConfig cfg;
        cfg.n_embd = 48;
        DecisionHead h2(cfg);
        CHECK(h2.ready());
        CHECK(h2.hidden_size() == 48);

        // Degenerate widths are refused, not silently accepted.
        DecisionHead bad;
        CHECK(!bad.init(0, 1u));
        CHECK(!bad.ready());
        CHECK(!bad.error().empty());
    }

    TEST("one matvec: uniform weights give a uniform, fail-closed distribution");
    {
        DecisionHead h;
        CHECK(h.init(8, 11u));
        // Zero every row so all logits are exactly 0 -> uniform softmax.
        const std::vector<float> zero = filled(8, 0.0f);
        for (int32_t a = 0; a < kActionCount; ++a)
            h.set_action(static_cast<DecisionAction>(a), zero.data(), 0.0f);

        const std::vector<float> x = filled(8, 1.0f);
        const DecisionResult d = h.decide(x.data());

        CHECK(d.matvecs == 1);                       // exactly one matvec, ever
        CHECK(d.confidence_score > 0.0f);
        CHECK(d.confidence_score <= 1.0f);
        // Uniform over 7 actions == 1/7, and a tie must resolve to ABSTAIN
        // (index 0) so a head with nothing to say says nothing.
        CHECK(std::fabs(d.confidence_score - 1.0f / static_cast<float>(kActionCount)) < 1e-4f);
        CHECK(d.action_type == DecisionAction::ABSTAIN);
        CHECK(d.routing == "abstain");
        CHECK(!d.fast_path);
    }

    TEST("argmax + softmax: a fitted row wins and self-routes");
    {
        DecisionHead h;
        CHECK(h.init(8, 13u));
        const std::vector<float> zero = filled(8, 0.0f);
        for (int32_t a = 0; a < kActionCount; ++a)
            h.set_action(static_cast<DecisionAction>(a), zero.data(), 0.0f);

        // BUY's row is 4.0 everywhere; with x = 1^8 that is a logit of 32.
        h.set_action(DecisionAction::BUY, filled(8, 4.0f).data(), 0.0f);
        h.set_action_asset(DecisionAction::BUY, "AAPL");

        const std::vector<float> x = filled(8, 1.0f);
        const DecisionResult d = h.decide(x.data());

        CHECK(d.action_type == DecisionAction::BUY);
        CHECK(d.confidence_score > 0.85f);
        CHECK(d.confidence_score <= 1.0f);
        CHECK(d.margin > 0.0f);
        CHECK(d.routing == "self");
        CHECK(d.fast_path);
        CHECK(d.target_asset == "AAPL");
        CHECK(d.matvecs == 1);
        CHECK(d.ms >= 0.0);
        CHECK(h.last_ns() >= 0.0);
    }

    TEST("a negative row loses: the matvec really uses the sign");
    {
        DecisionHead h;
        CHECK(h.init(8, 17u));
        const std::vector<float> zero = filled(8, 0.0f);
        for (int32_t a = 0; a < kActionCount; ++a)
            h.set_action(static_cast<DecisionAction>(a), zero.data(), 0.0f);
        h.set_action(DecisionAction::SELL, filled(8, -4.0f).data(), 0.0f);

        const std::vector<float> x = filled(8, 1.0f);
        const DecisionResult d = h.decide(x.data());
        CHECK(d.action_type != DecisionAction::SELL);
        CHECK(d.action_type == DecisionAction::ABSTAIN);   // best of the zeros
    }

    TEST("fail-closed: ABSTAIN never self-routes, however confident");
    {
        DecisionHead h;
        CHECK(h.init(8, 19u));
        const std::vector<float> zero = filled(8, 0.0f);
        for (int32_t a = 0; a < kActionCount; ++a)
            h.set_action(static_cast<DecisionAction>(a), zero.data(), 0.0f);
        h.set_action(DecisionAction::ABSTAIN, filled(8, 9.0f).data(), 0.0f);

        const DecisionResult d = h.decide(filled(8, 1.0f).data());
        CHECK(d.action_type == DecisionAction::ABSTAIN);
        CHECK(d.confidence_score > 0.85f);      // very sure...
        CHECK(d.routing == "abstain");          // ...about having nothing to say
        CHECK(!d.fast_path);
    }

    TEST("fail-closed: EXPLAIN never self-routes, however confident");
    {
        DecisionHead h;
        CHECK(h.init(8, 23u));
        const std::vector<float> zero = filled(8, 0.0f);
        for (int32_t a = 0; a < kActionCount; ++a)
            h.set_action(static_cast<DecisionAction>(a), zero.data(), 0.0f);
        h.set_action(DecisionAction::EXPLAIN, filled(8, 9.0f).data(), 0.0f);

        const DecisionResult d = h.decide(filled(8, 1.0f).data());
        CHECK(d.action_type == DecisionAction::EXPLAIN);
        CHECK(d.confidence_score > 0.85f);
        CHECK(d.routing == "system2");          // "I have a view, but it needs words"
        CHECK(!d.fast_path);
    }

    TEST("confidence gate: below threshold escalates to System-2");
    {
        DecisionHead h;
        CHECK(h.init(8, 29u));
        const std::vector<float> zero = filled(8, 0.0f);
        for (int32_t a = 0; a < kActionCount; ++a)
            h.set_action(static_cast<DecisionAction>(a), zero.data(), 0.0f);
        // A gentle preference: confident-ish but nowhere near 0.85.
        h.set_action(DecisionAction::HOLD, filled(8, 0.10f).data(), 0.0f);

        const std::vector<float> x = filled(8, 1.0f);
        const DecisionResult d = h.decide(x.data());
        CHECK(d.action_type == DecisionAction::HOLD);
        CHECK(d.confidence_score < h.threshold());
        CHECK(d.routing == "system2");
        CHECK(!d.fast_path);

        // The same head self-routes once the bar is lowered to meet it.
        h.set_threshold(0.20f);
        const DecisionResult d2 = h.decide(x.data());
        CHECK(d2.confidence_score >= 0.20f);
        CHECK(d2.routing == "self");
        CHECK(d2.fast_path);
    }

    TEST("margin gate: a near-tie escalates even when confident");
    {
        const std::vector<float> zero = filled(8, 0.0f);

        // Drive the logits from the BIAS with zero rows, so the distribution
        // is exactly computable instead of eyeballed: BUY 6.0, SELL 4.0, the
        // other five 0.0.
        //   p(BUY) = e^6 / (e^6 + e^4 + 5) = 403.43 / 463.03 = 0.871
        //   p(SELL) = 54.60 / 463.03 = 0.118  ->  margin = 0.753
        // i.e. comfortably past the 0.85 confidence bar, yet not a decisive
        // winner. That is exactly the case the margin gate exists for.
        DecisionHead h;
        CHECK(h.init(8, 31u));
        for (int32_t a = 0; a < kActionCount; ++a)
            h.set_action(static_cast<DecisionAction>(a), zero.data(), 0.0f);
        h.set_action(DecisionAction::BUY,  zero.data(), 6.0f);
        h.set_action(DecisionAction::SELL, zero.data(), 4.0f);

        const std::vector<float> x = filled(8, 1.0f);
        const DecisionResult base = h.decide(x.data());
        CHECK(base.action_type == DecisionAction::BUY);
        CHECK(base.confidence_score >= h.threshold());   // 0.871 >= 0.85
        CHECK(base.margin > 0.0f);
        CHECK(base.margin < 0.80f);                      // ...but only 0.753
        CHECK(base.routing == "self");
        CHECK(base.fast_path);

        // Identical distribution, but now demand a decisive winner.
        DecisionHead strict;
        CHECK(strict.init(8, 31u));
        strict.set_threshold(0.50f);
        strict.set_margin_floor(0.80f);
        for (int32_t a = 0; a < kActionCount; ++a)
            strict.set_action(static_cast<DecisionAction>(a), zero.data(), 0.0f);
        strict.set_action(DecisionAction::BUY,  zero.data(), 6.0f);
        strict.set_action(DecisionAction::SELL, zero.data(), 4.0f);

        const DecisionResult d = strict.decide(x.data());
        CHECK(d.action_type == DecisionAction::BUY);
        CHECK(d.confidence_score >= 0.50f);              // clears the bar...
        CHECK(d.margin < 0.80f);                         // ...but not the margin
        CHECK(d.routing == "system2");
        CHECK(!d.fast_path);
    }

    TEST("determinism + seed sensitivity");
    {
        RwkvConfig cfg;
        cfg.n_embd = 64;
        DecisionHead a(cfg, 1234u), b(cfg, 1234u), c(cfg, 99u);

        const std::vector<float> x = filled(64, 0.5f);
        const DecisionResult da = a.decide(x.data());
        const DecisionResult db = b.decide(x.data());
        // Same seed -> bit-identical decision (CI reproducibility).
        CHECK(da.action_type == db.action_type);
        CHECK(da.confidence_score == b.decide(x.data()).confidence_score);
        CHECK(da.margin == db.margin);

        // Different seeds must not collapse onto the same projection.
        const DecisionResult dc = c.decide(x.data());
        CHECK(dc.confidence_score != da.confidence_score);

        // Repeated calls on one head are stable.
        CHECK(a.decide(x.data()).confidence_score == da.confidence_score);
    }

    TEST("decide() does not mutate its input");
    {
        DecisionHead h;
        CHECK(h.init(16, 37u));
        std::vector<float> x = filled(16, 0.25f);
        const std::vector<float> before = x;
        (void)h.decide(x.data());
        CHECK(std::memcmp(x.data(), before.data(), x.size() * sizeof(float)) == 0);
    }

    TEST("Tensor overload agrees with the raw-pointer path");
    {
        DecisionHead h;
        CHECK(h.init(12, 41u));
        std::vector<float> xv = filled(12, 0.75f);
        Tensor t("h", {12}, DType::F32);
        std::memcpy(t.f32(), xv.data(), xv.size() * sizeof(float));

        const DecisionResult d1 = h.decide(xv.data());
        const DecisionResult d2 = h.decide(t);
        CHECK(d1.action_type == d2.action_type);
        CHECK(d1.confidence_score == d2.confidence_score);
        CHECK(d1.routing == d2.routing);

        // Wrong width / wrong dtype are refused, not read out of bounds.
        Tensor narrow("n", {4}, DType::F32);
        CHECK(h.decide(narrow).routing == "error");
        Tensor wrong("w", {12}, DType::F16);
        CHECK(h.decide(wrong).routing == "error");
        CHECK(h.decide(nullptr).routing == "error");
    }

    TEST("to_json is well-formed and carries the contract fields");
    {
        DecisionHead h;
        CHECK(h.init(8, 43u));
        const std::vector<float> zero = filled(8, 0.0f);
        for (int32_t a = 0; a < kActionCount; ++a)
            h.set_action(static_cast<DecisionAction>(a), zero.data(), 0.0f);
        h.set_action(DecisionAction::CLOSE, filled(8, 5.0f).data(), 0.0f);
        h.set_action_asset(DecisionAction::CLOSE, "BTCUSDT");
        h.set_action_invalidation(DecisionAction::CLOSE, 42123.5f);

        const std::string js = h.decide(filled(8, 1.0f).data()).to_json();
        CHECK(js.front() == '{');
        CHECK(js.back() == '}');
        CHECK(js.find("\"action\":\"CLOSE\"") != std::string::npos);
        CHECK(js.find("\"confidence\":") != std::string::npos);
        CHECK(js.find("\"target_asset\":\"BTCUSDT\"") != std::string::npos);
        CHECK(js.find("\"invalidation\":42123.5000") != std::string::npos);
        CHECK(js.find("\"routing\":\"self\"") != std::string::npos);
        CHECK(js.find("\"fast_path\":true") != std::string::npos);
        CHECK(js.find("\"matvecs\":1") != std::string::npos);
        // Balanced braces/quotes — cheap structural sanity for the parser.
        int depth = 0;
        for (const char ch : js) { if (ch == '{') ++depth; else if (ch == '}') --depth; }
        CHECK(depth == 0);
    }

    TEST("provenance: trained() only once every row is fitted");
    {
        DecisionHead h;
        CHECK(h.init(8, 47u));
        CHECK(!h.trained());
        CHECK(h.provenance().find("placeholder") != std::string::npos);

        const std::vector<float> zero = filled(8, 0.0f);
        for (int32_t a = 0; a < kActionCount - 1; ++a) {
            h.set_action(static_cast<DecisionAction>(a), zero.data(), 0.0f);
            CHECK(!h.trained());                  // still incomplete
        }
        h.set_action(DecisionAction::EXPLAIN, zero.data(), 0.0f);
        CHECK(h.trained());
        CHECK(h.fitted_actions() == kActionCount);

        // Re-setting an already-fitted row must not double-count.
        h.set_action(DecisionAction::BUY, zero.data(), 0.0f);
        CHECK(h.fitted_actions() == kActionCount);
        CHECK(h.trained());
    }

    TEST("invalidation: carried through, persisted, and zero means unknown");
    {
        const std::vector<float> zero = filled(8, 0.0f);
        DecisionHead h;
        CHECK(h.init(8, 71u));
        for (int32_t a = 0; a < kActionCount; ++a)
            h.set_action(static_cast<DecisionAction>(a), zero.data(), 0.0f);

        // Default is 0.0f == "not specified". It must NOT be read as a real
        // price level, so the raw value stays 0 and the caller decides.
        const DecisionResult unset = h.decide(filled(8, 1.0f).data());
        CHECK(unset.invalidation == 0.0f);

        // A BUY with a per-action default stop level.
        h.set_action(DecisionAction::BUY, filled(8, 4.0f).data(), 0.0f);
        h.set_action_invalidation(DecisionAction::BUY, 182.25f);
        const DecisionResult d = h.decide(filled(8, 1.0f).data());
        CHECK(d.action_type == DecisionAction::BUY);
        CHECK(d.invalidation == 182.25f);

        // The level is per-action: an action that is not selected does not
        // leak its own level into the result.
        h.set_action_invalidation(DecisionAction::SELL, 999.0f);
        CHECK(h.decide(filled(8, 1.0f).data()).invalidation == 182.25f);

        // Out-of-range handles are ignored, not a crash.
        h.set_action_invalidation(DecisionAction::COUNT, 1.0f);
        h.set_action_invalidation(static_cast<DecisionAction>(-1), 1.0f);
        CHECK(h.decide(filled(8, 1.0f).data()).invalidation == 182.25f);

        // It survives a save/load round-trip.
        const std::string path = temp_path("omniseed_dh_inval.bin");
        CHECK(h.save(path));
        DecisionHead r;
        CHECK(r.load(path));
        CHECK(r.decide(filled(8, 1.0f).data()).invalidation == 182.25f);
        std::remove(path.c_str());
    }

    TEST("persistence rejects a v1 blob rather than defaulting invalidation");
    {
        // A v1 header (version 1) must be refused: it carries no invalidation
        // levels, and silently reading them as 0.0 would turn "unknown" into
        // "no stop", which is the dangerous direction.
        const std::string path = temp_path("omniseed_dh_v1.bin");
        std::vector<uint8_t> bytes;
        const char magic[8] = {'O', 'M', 'N', 'I', 'S', 'D', 'H', '1'};
        bytes.insert(bytes.end(), magic, magic + 8);
        const int32_t ver = 1, E = 8, A = kActionCount, fit = 0;
        for (const int32_t v : {ver, E, A, fit}) {
            const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
            bytes.insert(bytes.end(), p, p + sizeof(int32_t));
        }
        CHECK(write_bytes(path, bytes));

        DecisionHead h;
        CHECK(!h.load(path));
        CHECK(!h.error().empty());
        std::remove(path.c_str());
    }

    TEST("persistence round-trip (fitted head)");
    {
        const std::string path = temp_path("omniseed_dh_test.bin");
        DecisionHead h;
        CHECK(h.init(10, 53u));
        const std::vector<float> zero = filled(10, 0.0f);
        for (int32_t a = 0; a < kActionCount; ++a)
            h.set_action(static_cast<DecisionAction>(a), zero.data(), 0.0f);
        h.set_action(DecisionAction::HEDGE, filled(10, 3.0f).data(), 0.25f);
        h.set_action_asset(DecisionAction::HEDGE, "EURUSD");
        CHECK(h.trained());
        CHECK(h.save(path));

        DecisionHead r;
        CHECK(r.load(path));
        CHECK(r.ready());
        CHECK(r.hidden_size() == 10);
        CHECK(r.action_count() == kActionCount);
        CHECK(r.trained());

        std::vector<float> x = filled(10, 1.0f);
        const DecisionResult a = h.decide(x.data());
        const DecisionResult b = r.decide(x.data());
        CHECK(a.action_type == b.action_type);
        CHECK(a.confidence_score == b.confidence_score);
        CHECK(a.margin == b.margin);
        CHECK(a.target_asset == "EURUSD");
        CHECK(b.target_asset == "EURUSD");
        CHECK(a.to_json() == b.to_json());

        std::remove(path.c_str());
    }

    TEST("persistence round-trip (untrained head stays untrained)");
    {
        const std::string path = temp_path("omniseed_dh_untrained.bin");
        RwkvConfig cfg;
        cfg.n_embd = 24;
        DecisionHead h(cfg, 7u);
        CHECK(!h.trained());
        CHECK(h.save(path));

        DecisionHead r;
        CHECK(r.load(path));
        CHECK(!r.trained());                      // honesty survives the disk
        CHECK(r.provenance().find("placeholder") != std::string::npos);

        std::vector<float> x = filled(24, 0.5f);
        CHECK(h.decide(x.data()).confidence_score == r.decide(x.data()).confidence_score);

        std::remove(path.c_str());
    }

    TEST("persistence rejects bad input instead of guessing");
    {
        // Missing file.
        DecisionHead h;
        CHECK(!h.load(temp_path("omniseed_dh_does_not_exist.bin")));
        CHECK(!h.error().empty());

        // Wrong magic.
        const std::string junk = temp_path("omniseed_dh_junk.bin");
        CHECK(write_bytes(junk, std::vector<uint8_t>(64, 0xAB)));
        DecisionHead h2;
        CHECK(!h2.load(junk));
        CHECK(!h2.ready());

        // Correct magic, then truncated before the payload.
        const std::string trunc = temp_path("omniseed_dh_trunc.bin");
        std::vector<uint8_t> bytes;
        const char magic[8] = {'O', 'M', 'N', 'I', 'S', 'D', 'H', '1'};
        bytes.insert(bytes.end(), magic, magic + 8);
        const int32_t ver = 1, E = 8, A = kActionCount, fit = 0;
        for (const int32_t v : {ver, E, A, fit}) {
            const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
            bytes.insert(bytes.end(), p, p + sizeof(int32_t));
        }
        CHECK(write_bytes(trunc, bytes));
        DecisionHead h3;
        CHECK(!h3.load(trunc));

        std::remove(junk.c_str());
        std::remove(trunc.c_str());
    }

    // =======================================================================
    // Part B — the speed claim
    // =======================================================================
    TEST("System-1 is >=100x faster than the cheapest System-2 token step");
    {
        struct Scenario { const char* label; int64_t E; int64_t V; int32_t iters; };
        // (E, V) pairs from the tree: the tiny test model, and a mid-size
        // model. The real 0.1B deployment (E=768, V=65536) is strictly worse
        // for System-2, so these two are conservative stand-ins.
        const Scenario scen[] = {
            {"tiny  (E=64,  V=256)",   64,  256, 20000},
            {"mid   (E=256, V=8192)", 256, 8192,   400},
        };
        const int32_t max_new_tokens = 160;   // AgentLoop::Config default

        for (const Scenario& sc : scen) {
            DecisionHead head;
            CHECK(head.init(static_cast<int32_t>(sc.E), 61u));
            const std::vector<float> hx = filled(static_cast<int32_t>(sc.E), 0.1f);

            (void)head.decide(hx.data());     // warm the scratch buffers

            const double decide_ns = bench_decide_ns(head, hx, sc.iters);
            const double step_ns   = bench_head_matvec_ns(sc.V, sc.E, sc.iters);
            const double ratio     = (step_ns * static_cast<double>(max_new_tokens)) /
                                     decide_ns;

            platform::log_info(
                "  %s: decide=%.0f ns  1 token step=%.0f ns  "
                "ratio over %d tokens = %.0fx",
                sc.label, decide_ns, step_ns, max_new_tokens, ratio);

            // The mandated bar.
            CHECK(ratio >= 100.0);
            // The "sub-millisecond" claim, with room to spare.
            CHECK(decide_ns < 1e6);
            // System-1 must be cheaper than even ONE token step, not merely
            // cheaper than a full generation.
            CHECK(decide_ns < step_ns);
        }
        CHECK(std::isfinite(g_sink));
    }

    platform::log_info("---- decision head tests: %d passed, %d failed ----",
                       g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
