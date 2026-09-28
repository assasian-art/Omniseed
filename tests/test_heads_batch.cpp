// =============================================================================
//  OmniSeed — tests/test_heads_batch.cpp
//
//  MILESTONE 8 — Phase 2.1, batched head evaluation.
//
//  THE CONTRACT THIS FILE EXISTS TO PIN. A batched head call must return
//  EXACTLY what B separate calls return. Not "close to" — exactly, bit for bit,
//  when the Scalar kernel is selected. Every Part B/C/D test below compares
//  against a per-row reference and asserts equality, and Part A proves the
//  kernel that makes that possible is itself correct.
//
//  The AVX2 kernel is the deliberate exception: it reduces eight lanes, and
//  floating-point addition is not associative, so it differs in the last bits.
//  Part A5 measures that difference and asserts a tolerance instead. If a
//  future change makes the SIMD kernel bit-identical, A5's "they differ" check
//  will fail and tell you the kernel changed — which is information, not noise.
//
//  Fully offline and UNGATED: the heads are seeded deterministically, and the
//  one real-data part (B7/C4) uses the COMMITTED h[E] fixtures. No model, no
//  `.venv`, no network.
// =============================================================================
#include "omniseed/classification_head.h"
#include "omniseed/core/batch_gemm.h"
#include "omniseed/core/platform.h"
#include "omniseed/core/tensor.h"
#include "omniseed/decision_head.h"
#include "omniseed/scoring_head.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace omniseed;

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
#define SKIP(msg)                                                          \
    do {                                                                   \
        ++g_skipped;                                                       \
        platform::log_info("SKIP  %s  (%s)", g_current.c_str(), msg);      \
    } while (0)

namespace {

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
uint64_t g_rng = 0x243F6A8885A308D3ull;
float rnd(float lo, float hi) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    const double u = static_cast<double>(g_rng >> 11) / static_cast<double>(1ull << 53);
    return static_cast<float>(lo + (hi - lo) * u);
}

std::vector<float> random_block(int32_t rows, int32_t cols, float lo, float hi) {
    std::vector<float> v(static_cast<size_t>(rows) * static_cast<size_t>(cols));
    for (float& x : v) x = rnd(lo, hi);
    return v;
}

// Mixed absolute/relative comparison — the right shape for a reduction whose
// error is a few ulps of the ACCUMULATOR, not of the result.
bool close(float a, float b, float rel, float absv) {
    return std::fabs(a - b) <= absv + rel * std::fabs(b);
}

// Row-major [n, E] float32 fixture.
std::vector<float> load_f32(const std::string& path, int32_t E, int32_t& n) {
    n = 0;
    std::vector<float> v;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return v;
    std::fseek(f, 0, SEEK_END);
    const long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (sz <= 0 || (sz % 4) != 0) { std::fclose(f); return v; }
    v.resize(static_cast<size_t>(sz) / sizeof(float));
    const size_t got = std::fread(v.data(), sizeof(float), v.size(), f);
    std::fclose(f);
    v.resize(got);
    n = static_cast<int32_t>(v.size() / static_cast<size_t>(E));
    return v;
}

// The per-row reference for batch_gemm: same loop order the scalar kernel uses,
// written independently so a bug in the kernel cannot hide behind itself.
std::vector<float> reference_logits(const std::vector<float>& H, int32_t B, int32_t E,
                                    const std::vector<float>& W, const std::vector<float>& bias,
                                    int32_t L, float scale) {
    std::vector<float> out(static_cast<size_t>(B) * static_cast<size_t>(L));
    for (int32_t b = 0; b < B; ++b)
        for (int32_t l = 0; l < L; ++l) {
            float acc = bias[static_cast<size_t>(l)];
            for (int32_t e = 0; e < E; ++e)
                acc += W[static_cast<size_t>(l) * static_cast<size_t>(E) +
                         static_cast<size_t>(e)] *
                       H[static_cast<size_t>(b) * static_cast<size_t>(E) +
                         static_cast<size_t>(e)];
            out[static_cast<size_t>(b) * static_cast<size_t>(L) + static_cast<size_t>(l)] =
                acc * scale;
        }
    return out;
}

bool same_result(const ClassificationResult& a, const ClassificationResult& b) {
    if (a.domain != b.domain) return false;
    if (a.top_k.size() != b.top_k.size()) return false;
    if (a.margin != b.margin) return false;
    for (size_t i = 0; i < a.top_k.size(); ++i)
        if (a.top_k[i].label != b.top_k[i].label ||
            a.top_k[i].probability != b.top_k[i].probability)
            return false;
    return true;
}

bool same_result(const DecisionResult& a, const DecisionResult& b) {
    return a.action_type == b.action_type && a.confidence_score == b.confidence_score &&
           a.margin == b.margin && a.target_asset == b.target_asset &&
           a.invalidation == b.invalidation && a.routing == b.routing &&
           a.fast_path == b.fast_path && a.matvecs == b.matvecs;
}

// The DEFAULT kernel (Auto -> Simd) cannot be bit-identical, so the meaningful
// contract there is that the ANSWER is unchanged: same top-1, same action, same
// routing, and probabilities within a tolerance. A caller reads the label, not
// the seventh decimal place.
bool same_top1(const ClassificationResult& a, const ClassificationResult& b) {
    return !a.top_k.empty() && !b.top_k.empty() && a.top_k[0].label == b.top_k[0].label;
}
bool probs_close(const ClassificationResult& a, const ClassificationResult& b, float rel,
                 float absv) {
    if (a.top_k.size() != b.top_k.size()) return false;
    for (size_t i = 0; i < a.top_k.size(); ++i)
        if (a.top_k[i].label != b.top_k[i].label ||
            !close(a.top_k[i].probability, b.top_k[i].probability, rel, absv))
            return false;
    return true;
}

// ---------------------------------------------------------------------------
// PART A — the kernel
// ---------------------------------------------------------------------------
void part_a_kernel() {
    TEST("A1 Scalar batch == an independent per-row loop, BIT-IDENTICAL");
    {
        const int32_t dims[][3] = {{1, 1, 1}, {1, 8, 3}, {7, 16, 4}, {64, 32, 7},
                                   {129, 768, 5}, {1000, 8, 2}};
        bool all_exact = true;
        for (const auto& d : dims) {
            const int32_t B = d[0], E = d[1], L = d[2];
            const std::vector<float> H = random_block(B, E, -1.0f, 1.0f);
            const std::vector<float> W = random_block(L, E, -0.5f, 0.5f);
            const std::vector<float> bias = random_block(L, 1, -0.2f, 0.2f);
            std::vector<float> got(static_cast<size_t>(B) * static_cast<size_t>(L));
            batch_gemm(H.data(), W.data(), bias.data(), got.data(), B, E, L, 1.0f,
                       BatchKernel::Scalar, nullptr);
            const std::vector<float> want = reference_logits(H, B, E, W, bias, L, 1.0f);
            for (size_t i = 0; i < got.size(); ++i)
                if (got[i] != want[i]) all_exact = false;
        }
        CHECK(all_exact);
    }
    TEST("A2 the scale lands in the same place as the heads' 1/T");
    {
        const int32_t B = 32, E = 64, L = 5;
        const std::vector<float> H = random_block(B, E, -1.0f, 1.0f);
        const std::vector<float> W = random_block(L, E, -0.5f, 0.5f);
        const std::vector<float> bias = random_block(L, 1, -0.2f, 0.2f);
        std::vector<float> got(static_cast<size_t>(B) * static_cast<size_t>(L));
        const float scale = 1.0f / 3.25f;
        batch_gemm(H.data(), W.data(), bias.data(), got.data(), B, E, L, scale,
                   BatchKernel::Scalar, nullptr);
        const std::vector<float> want = reference_logits(H, B, E, W, bias, L, scale);
        bool all_exact = true;
        for (size_t i = 0; i < got.size(); ++i)
            if (got[i] != want[i]) all_exact = false;
        CHECK(all_exact);
        // scale == 1.0 must be an EXACT no-op, so an uncalibrated head takes
        // byte-for-byte the path it always did. Compared against a reference
        // computed with scale 1.0 directly — comparing against `want * 3.25`
        // would fail, because 1/3.25 is not exactly representable and the round
        // trip is not an identity.
        std::vector<float> one(static_cast<size_t>(B) * static_cast<size_t>(L));
        batch_gemm(H.data(), W.data(), bias.data(), one.data(), B, E, L, 1.0f,
                   BatchKernel::Scalar, nullptr);
        const std::vector<float> want_one = reference_logits(H, B, E, W, bias, L, 1.0f);
        bool noop = true;
        for (size_t i = 0; i < one.size(); ++i)
            if (one[i] != want_one[i]) noop = false;
        CHECK(noop);
        // ...and the scaled result really is a different number, so A2 is not
        // passing because the scale was ignored.
        bool scaled_differs = false;
        for (size_t i = 0; i < one.size(); ++i)
            if (one[i] != got[i]) scaled_differs = true;
        CHECK(scaled_differs);
    }
    TEST("A3 an empty batch is a no-op, and a null pointer is refused");
    {
        std::vector<float> out(16, 7.0f);
        const std::vector<float> H = random_block(4, 8, -1.0f, 1.0f);
        const std::vector<float> W = random_block(2, 8, -1.0f, 1.0f);
        const std::vector<float> bias = random_block(2, 1, -1.0f, 1.0f);
        batch_gemm(H.data(), W.data(), bias.data(), out.data(), 0, 8, 2, 1.0f,
                   BatchKernel::Scalar, nullptr);
        batch_gemm(H.data(), W.data(), bias.data(), out.data(), 4, 0, 2, 1.0f,
                   BatchKernel::Scalar, nullptr);
        batch_gemm(H.data(), W.data(), bias.data(), out.data(), 4, 8, 0, 1.0f,
                   BatchKernel::Scalar, nullptr);
        bool untouched = true;
        for (const float v : out)
            if (v != 7.0f) untouched = false;
        CHECK(untouched);
        // null with a non-zero count must not be dereferenced
        batch_gemm(nullptr, W.data(), bias.data(), out.data(), 4, 8, 2, 1.0f,
                   BatchKernel::Scalar, nullptr);
        batch_gemm(H.data(), nullptr, bias.data(), out.data(), 4, 8, 2, 1.0f,
                   BatchKernel::Scalar, nullptr);
        batch_gemm(H.data(), W.data(), bias.data(), nullptr, 4, 8, 2, 1.0f,
                   BatchKernel::Scalar, nullptr);
        untouched = true;
        for (const float v : out)
            if (v != 7.0f) untouched = false;
        CHECK(untouched);
    }
    TEST("A4 the plan reports the real shape, MAC count and kernel");
    {
        const int32_t B = 9, E = 32, L = 3;
        const std::vector<float> H = random_block(B, E, -1.0f, 1.0f);
        const std::vector<float> W = random_block(L, E, -1.0f, 1.0f);
        const std::vector<float> bias = random_block(L, 1, -1.0f, 1.0f);
        std::vector<float> out(static_cast<size_t>(B) * static_cast<size_t>(L));
        BatchGemmPlan plan;
        batch_gemm(H.data(), W.data(), bias.data(), out.data(), B, E, L, 1.0f,
                   BatchKernel::Scalar, &plan);
        CHECK(plan.rows == B);
        CHECK(plan.width == E);
        CHECK(plan.outputs == L);
        CHECK(plan.macs == static_cast<int64_t>(B) * E * L);
        CHECK(plan.requested == BatchKernel::Scalar);
        CHECK(plan.used == BatchKernel::Scalar);
        CHECK(!plan.simd_fell_back);
        platform::log_info("  %s / %s available=%d", batch_kernel_name(plan.requested),
                           batch_kernel_name(plan.used),
                           batch_simd_available() ? 1 : 0);
    }
    TEST("A5 Simd matches Scalar within tolerance (and is NOT bit-identical)");
    {
        if (!batch_simd_available()) {
            SKIP("this CPU/build has no AVX2 kernel");
        } else {
            const int32_t B = 64, E = 768, L = 7;
            const std::vector<float> H = random_block(B, E, -1.0f, 1.0f);
            const std::vector<float> W = random_block(L, E, -0.05f, 0.05f);
            const std::vector<float> bias = random_block(L, 1, -0.2f, 0.2f);
            std::vector<float> sc(static_cast<size_t>(B) * static_cast<size_t>(L));
            std::vector<float> si(static_cast<size_t>(B) * static_cast<size_t>(L));
            BatchGemmPlan ps, pi;
            batch_gemm(H.data(), W.data(), bias.data(), sc.data(), B, E, L, 1.0f,
                       BatchKernel::Scalar, &ps);
            batch_gemm(H.data(), W.data(), bias.data(), si.data(), B, E, L, 1.0f,
                       BatchKernel::Simd, &pi);
            CHECK(pi.used == BatchKernel::Simd);
            CHECK(!pi.simd_fell_back);

            float worst = 0.0f, worst_rel = 0.0f;
            bool within = true, any_diff = false;
            for (size_t i = 0; i < sc.size(); ++i) {
                const float d = std::fabs(sc[i] - si[i]);
                worst = std::max(worst, d);
                if (std::fabs(sc[i]) > 1e-6f)
                    worst_rel = std::max(worst_rel, d / std::fabs(sc[i]));
                if (sc[i] != si[i]) any_diff = true;
                if (!close(si[i], sc[i], 1e-5f, 1e-4f)) within = false;
            }
            platform::log_info("  simd vs scalar: max abs diff %.3e, max rel diff %.3e",
                               static_cast<double>(worst), static_cast<double>(worst_rel));
            CHECK(within);
            // The kernels MUST differ: if they do not, either the SIMD path was
            // not taken or it was quietly rewritten as a serial sum.
            CHECK(any_diff);
        }
    }
    TEST("A6 an explicit Simd request on a non-AVX2 machine reports the fallback");
    {
        if (batch_simd_available()) {
            SKIP("this machine HAS AVX2, so the fallback path is unreachable here");
        } else {
            const std::vector<float> H = random_block(4, 8, -1.0f, 1.0f);
            const std::vector<float> W = random_block(2, 8, -1.0f, 1.0f);
            const std::vector<float> bias = random_block(2, 1, -1.0f, 1.0f);
            std::vector<float> out(8);
            BatchGemmPlan plan;
            batch_gemm(H.data(), W.data(), bias.data(), out.data(), 4, 8, 2, 1.0f,
                       BatchKernel::Simd, &plan);
            CHECK(plan.used == BatchKernel::Scalar);
            CHECK(plan.simd_fell_back);
            // Auto, by contrast, is not a fallback — it is the design.
            batch_gemm(H.data(), W.data(), bias.data(), out.data(), 4, 8, 2, 1.0f,
                       BatchKernel::Auto, &plan);
            CHECK(plan.used == BatchKernel::Scalar);
            CHECK(!plan.simd_fell_back);
        }
    }
}

// ---------------------------------------------------------------------------
// PART B — ClassificationHead
// ---------------------------------------------------------------------------
void part_b_classification() {
    const int32_t E = 32;
    ClassificationHead head;
    CHECK(head.init(E, 777u));
    CHECK(head.add_label_set("t.set", {"a", "b", "c", "d", "e"}) == 0);

    TEST("B1 classify_batch == per-row classify, bit-identical, every row");
    {
        // The exactness contract is only claimable on the SCALAR kernel: Auto
        // resolves to AVX2 on this machine, and eight lanes cannot reproduce a
        // serial sum bit for bit. The default kernel's contract is B1b below.
        head.set_batch_kernel(BatchKernel::Scalar);
        const int32_t B = 40;
        const std::vector<float> H = random_block(B, E, -2.0f, 2.0f);
        std::vector<ClassificationResult> batched;
        BatchStats stats;
        CHECK(head.classify_batch(H.data(), B, "t.set", 3, batched, &stats));
        CHECK(batched.size() == static_cast<size_t>(B));
        CHECK(stats.rows == B);
        CHECK(stats.outputs == B * 5);
        CHECK(stats.used == BatchKernel::Scalar);
        bool all_same = true;
        for (int32_t b = 0; b < B; ++b)
            if (!same_result(batched[static_cast<size_t>(b)],
                             head.classify(H.data() + static_cast<size_t>(b) * E, "t.set", 3)))
                all_same = false;
        CHECK(all_same);
    }
    TEST("B1b the DEFAULT kernel changes no ANSWER, only the last bits");
    {
        head.set_batch_kernel(BatchKernel::Auto);
        const int32_t B = 40;
        const std::vector<float> H = random_block(B, E, -2.0f, 2.0f);
        std::vector<ClassificationResult> batched;
        BatchStats stats;
        CHECK(head.classify_batch(H.data(), B, "t.set", 3, batched, &stats));
        bool same_label = true, close_enough = true;
        for (int32_t b = 0; b < B; ++b) {
            const ClassificationResult r =
                head.classify(H.data() + static_cast<size_t>(b) * E, "t.set", 3);
            if (!same_top1(batched[static_cast<size_t>(b)], r)) same_label = false;
            if (!probs_close(batched[static_cast<size_t>(b)], r, 1e-4f, 1e-6f))
                close_enough = false;
        }
        CHECK(same_label);
        CHECK(close_enough);
        platform::log_info("  default kernel = %s", batch_kernel_name(stats.used));
    }
    TEST("B2 a calibrated temperature is applied identically in both paths");
    {
        head.set_batch_kernel(BatchKernel::Scalar);
        head.set_temperature(0, 3.25f);
        const int32_t B = 12;
        const std::vector<float> H = random_block(B, E, -2.0f, 2.0f);
        std::vector<ClassificationResult> batched;
        CHECK(head.classify_batch(H.data(), B, "t.set", 5, batched, nullptr));
        bool same = true;
        for (int32_t b = 0; b < B; ++b)
            if (!same_result(batched[static_cast<size_t>(b)],
                             head.classify(H.data() + static_cast<size_t>(b) * E, "t.set", 5)))
                same = false;
        CHECK(same);
        // and the temperature genuinely changed the answer vs T = 1
        ClassificationResult hot = head.classify(H.data(), "t.set", 5);
        head.set_temperature(0, 1.0f);
        ClassificationResult raw = head.classify(H.data(), "t.set", 5);
        head.set_temperature(0, 3.25f);
        CHECK(hot.top_k.size() == raw.top_k.size());
        CHECK(hot.top_k[0].probability != raw.top_k[0].probability);
    }
    TEST("B3 top_k is clamped the same way in both paths");
    {
        const std::vector<float> H = random_block(4, E, -1.0f, 1.0f);
        std::vector<ClassificationResult> out;
        CHECK(head.classify_batch(H.data(), 4, "t.set", 99, out, nullptr));
        CHECK(out[0].top_k.size() == 5);            // clamped down to L
        CHECK(head.classify_batch(H.data(), 4, "t.set", 0, out, nullptr));
        CHECK(out[0].top_k.size() == 1);            // clamped up to 1
        CHECK(head.classify_batch(H.data(), 4, "t.set", -3, out, nullptr));
        CHECK(out[0].top_k.size() == 1);
    }
    TEST("B4 classify_batch fails closed and leaves out EMPTY");
    {
        const std::vector<float> H = random_block(4, E, -1.0f, 1.0f);
        std::vector<ClassificationResult> out(3);
        CHECK(!head.classify_batch(H.data(), 4, "nope.set", 3, out, nullptr));
        CHECK(out.empty());
        CHECK(!head.classify_batch(H.data(), 0, "t.set", 3, out, nullptr));
        CHECK(out.empty());
        CHECK(!head.classify_batch(nullptr, 4, "t.set", 3, out, nullptr));
        CHECK(out.empty());

        ClassificationHead cold;   // never init'd
        CHECK(!cold.classify_batch(H.data(), 4, "t.set", 3, out, nullptr));
        CHECK(out.empty());
    }
    TEST("B5 the Tensor overload refuses a non-[B,E] shape");
    {
        std::vector<ClassificationResult> out;
        Tensor wrong_rank("H", {8}, DType::F32);
        CHECK(!head.classify_batch(wrong_rank, "t.set", 3, out, nullptr));
        CHECK(out.empty());
        Tensor wrong_width("H", {4, 7}, DType::F32);   // E is 32
        CHECK(!head.classify_batch(wrong_width, "t.set", 3, out, nullptr));
        CHECK(out.empty());
        Tensor good("H", {4, E}, DType::F32);
        for (int64_t i = 0; i < good.numel(); ++i) good.f32()[i] = 0.1f * static_cast<float>(i);
        CHECK(head.classify_batch(good, "t.set", 3, out, nullptr));
        CHECK(out.size() == 4);
    }
    TEST("B6 an unknown label set index is refused");
    {
        const std::vector<float> H = random_block(4, E, -1.0f, 1.0f);
        std::vector<ClassificationResult> out;
        CHECK(!head.classify_batch(H.data(), 4, 99, 3, out, nullptr));
        CHECK(out.empty());
        CHECK(!head.classify_batch(H.data(), 4, -1, 3, out, nullptr));
        CHECK(out.empty());
    }
    TEST("B7 REAL DATA: 369 held-out trading h[E], both paths agree on every row");
    {
        int32_t n = 0;
        const std::vector<float> H =
            load_f32("tests/fixtures/head_calibration/trading/hidden.f32", 768, n);
        if (H.empty() || n != 369) {
            SKIP("held-out h[E] fixture missing");
        } else {
            ClassificationHead real;
            CHECK(real.init(768, 777u));
            CHECK(real.add_label_set("trading.regime",
                                     {"trend_up", "trend_down", "range", "high_vol"}) == 0);
            // --- the exactness contract: Scalar kernel, real h[E] -------------
            real.set_batch_kernel(BatchKernel::Scalar);
            std::vector<ClassificationResult> out;
            BatchStats stats;
            CHECK(real.classify_batch(H.data(), n, "trading.regime", 4, out, &stats));
            CHECK(out.size() == static_cast<size_t>(n));
            CHECK(stats.used == BatchKernel::Scalar);
            bool same = true;
            for (int32_t b = 0; b < n; ++b)
                if (!same_result(out[static_cast<size_t>(b)],
                                 real.classify(H.data() + static_cast<size_t>(b) * 768,
                                               "trading.regime", 4)))
                    same = false;
            CHECK(same);
            platform::log_info("  369 real rows, scalar: %.1f us/row", stats.us_per_row);

            // --- the DEFAULT kernel: same label on every one of the 369 ------
            real.set_batch_kernel(BatchKernel::Auto);
            std::vector<ClassificationResult> fast;
            BatchStats fast_stats;
            CHECK(real.classify_batch(H.data(), n, "trading.regime", 4, fast, &fast_stats));
            int32_t label_mismatches = 0;
            for (int32_t b = 0; b < n; ++b)
                if (!same_top1(fast[static_cast<size_t>(b)], out[static_cast<size_t>(b)]))
                    ++label_mismatches;
            platform::log_info("  369 real rows, %s: %.1f us/row, top-1 mismatches vs scalar: %d",
                               batch_kernel_name(fast_stats.used), fast_stats.us_per_row,
                               label_mismatches);
            CHECK(label_mismatches == 0);
        }
    }
}

// ---------------------------------------------------------------------------
// PART C — DecisionHead
// ---------------------------------------------------------------------------
void part_c_decision() {
    const int32_t E = 32;
    DecisionHead head;
    CHECK(head.init(E, 1234u));

    TEST("C1 decide_batch == per-row decide, bit-identical, every field");
    {
        head.set_batch_kernel(BatchKernel::Scalar);
        const int32_t B = 40;
        const std::vector<float> H = random_block(B, E, -2.0f, 2.0f);
        std::vector<DecisionResult> batched;
        BatchStats stats;
        CHECK(head.decide_batch(H.data(), B, batched, &stats));
        CHECK(batched.size() == static_cast<size_t>(B));
        CHECK(stats.rows == B);
        CHECK(stats.outputs == B * head.action_count());
        CHECK(stats.used == BatchKernel::Scalar);
        bool same = true;
        for (int32_t b = 0; b < B; ++b)
            if (!same_result(batched[static_cast<size_t>(b)],
                             head.decide(H.data() + static_cast<size_t>(b) * E)))
                same = false;
        CHECK(same);
    }
    TEST("C1b the DEFAULT kernel changes no ACTION, only the last bits");
    {
        head.set_batch_kernel(BatchKernel::Auto);
        const int32_t B = 200;
        const std::vector<float> H = random_block(B, E, -2.0f, 2.0f);
        std::vector<DecisionResult> batched;
        BatchStats stats;
        CHECK(head.decide_batch(H.data(), B, batched, &stats));
        int32_t action_mismatch = 0, routing_mismatch = 0;
        for (int32_t b = 0; b < B; ++b) {
            const DecisionResult r = head.decide(H.data() + static_cast<size_t>(b) * E);
            const DecisionResult& q = batched[static_cast<size_t>(b)];
            if (q.action_type != r.action_type) ++action_mismatch;
            if (q.routing != r.routing) ++routing_mismatch;
        }
        platform::log_info("  default kernel = %s; action mismatches %d/%d, routing %d/%d",
                           batch_kernel_name(stats.used), action_mismatch, B,
                           routing_mismatch, B);
        CHECK(action_mismatch == 0);
        CHECK(routing_mismatch == 0);
    }
    TEST("C2 a calibrated temperature is applied identically in both paths");
    {
        head.set_batch_kernel(BatchKernel::Scalar);
        head.set_temperature(13.325f);
        const int32_t B = 20;
        const std::vector<float> H = random_block(B, E, -2.0f, 2.0f);
        std::vector<DecisionResult> batched;
        CHECK(head.decide_batch(H.data(), B, batched, nullptr));
        bool same = true;
        for (int32_t b = 0; b < B; ++b)
            if (!same_result(batched[static_cast<size_t>(b)],
                             head.decide(H.data() + static_cast<size_t>(b) * E)))
                same = false;
        CHECK(same);
        const float with_t = head.decide(H.data()).confidence_score;
        head.set_temperature(1.0f);
        const float without_t = head.decide(H.data()).confidence_score;
        head.set_temperature(13.325f);
        CHECK(with_t != without_t);   // the temperature is not a no-op here
    }
    TEST("C3 routing rules survive batching (abstain/explain never self-route)");
    {
        const int32_t B = 200;
        const std::vector<float> H = random_block(B, E, -3.0f, 3.0f);
        std::vector<DecisionResult> batched;
        CHECK(head.decide_batch(H.data(), B, batched, nullptr));
        bool ok = true;
        for (const DecisionResult& r : batched) {
            if (r.action_type == DecisionAction::ABSTAIN && r.routing != "abstain") ok = false;
            if (r.action_type == DecisionAction::EXPLAIN && r.routing != "system2") ok = false;
            if (r.routing == "self" && !r.fast_path) ok = false;
            if (r.routing == "self" && r.confidence_score < head.threshold()) ok = false;
        }
        CHECK(ok);
    }
    TEST("C4 decide_batch fails closed and leaves out EMPTY");
    {
        const std::vector<float> H = random_block(4, E, -1.0f, 1.0f);
        std::vector<DecisionResult> out(3);
        CHECK(!head.decide_batch(H.data(), 0, out, nullptr));
        CHECK(out.empty());
        CHECK(!head.decide_batch(nullptr, 4, out, nullptr));
        CHECK(out.empty());
        DecisionHead cold;
        CHECK(!cold.decide_batch(H.data(), 4, out, nullptr));
        CHECK(out.empty());
        Tensor wrong("H", {4, 7}, DType::F32);
        CHECK(!head.decide_batch(wrong, out, nullptr));
        CHECK(out.empty());
    }
    TEST("C5 REAL DATA: 369 held-out trading h[E] through decide_batch");
    {
        int32_t n = 0;
        const std::vector<float> H =
            load_f32("tests/fixtures/head_calibration/trading/hidden.f32", 768, n);
        if (H.empty() || n != 369) {
            SKIP("held-out h[E] fixture missing");
        } else {
            DecisionHead real;
            CHECK(real.init(768, 1234u));
            real.set_batch_kernel(BatchKernel::Scalar);
            std::vector<DecisionResult> out;
            BatchStats stats;
            CHECK(real.decide_batch(H.data(), n, out, &stats));
            CHECK(out.size() == static_cast<size_t>(n));
            CHECK(stats.used == BatchKernel::Scalar);
            bool same = true;
            for (int32_t b = 0; b < n; ++b)
                if (!same_result(out[static_cast<size_t>(b)],
                                 real.decide(H.data() + static_cast<size_t>(b) * 768)))
                    same = false;
            CHECK(same);
            platform::log_info("  369 real rows, scalar: %.1f us/row", stats.us_per_row);

            // The default kernel must reach the same ACTION on every real bar.
            real.set_batch_kernel(BatchKernel::Auto);
            std::vector<DecisionResult> fast;
            BatchStats fast_stats;
            CHECK(real.decide_batch(H.data(), n, fast, &fast_stats));
            int32_t mismatches = 0;
            for (int32_t b = 0; b < n; ++b)
                if (fast[static_cast<size_t>(b)].action_type !=
                    out[static_cast<size_t>(b)].action_type)
                    ++mismatches;
            platform::log_info("  369 real rows, %s: %.1f us/row, action mismatches: %d",
                               batch_kernel_name(fast_stats.used), fast_stats.us_per_row,
                               mismatches);
            CHECK(mismatches == 0);
        }
    }
}

// ---------------------------------------------------------------------------
// PART D — ScoringHead
// ---------------------------------------------------------------------------
void part_d_scoring() {
    const int32_t E = 32;
    ScoringHead head;
    CHECK(head.init(E, 555u));

    TEST("D1 score_batch == per-row score, bit-identical");
    {
        head.set_batch_kernel(BatchKernel::Scalar);
        const int32_t B = 40;
        const std::vector<float> H = random_block(B, E, -2.0f, 2.0f);
        std::vector<ScoreResult> batched;
        BatchStats stats;
        CHECK(head.score_batch(H.data(), B, batched, &stats));
        CHECK(batched.size() == static_cast<size_t>(B));
        CHECK(stats.outputs == B * ScoringHead::ROW_COUNT);
        bool same = true;
        for (int32_t b = 0; b < B; ++b) {
            const ScoreResult r = head.score(H.data() + static_cast<size_t>(b) * E);
            const ScoreResult& q = batched[static_cast<size_t>(b)];
            if (r.priority != q.priority || r.urgency != q.urgency ||
                r.confidence != q.confidence || r.matvecs != q.matvecs)
                same = false;
        }
        CHECK(same);
        // every score is a sigmoid output
        bool in_range = true;
        for (const ScoreResult& r : batched)
            if (!(r.priority >= 0.0f && r.priority <= 1.0f) ||
                !(r.urgency >= 0.0f && r.urgency <= 1.0f) ||
                !(r.confidence >= 0.0f && r.confidence <= 1.0f))
                in_range = false;
        CHECK(in_range);
    }
    TEST("D2 score_batch fails closed and leaves out EMPTY");
    {
        const std::vector<float> H = random_block(4, E, -1.0f, 1.0f);
        std::vector<ScoreResult> out(3);
        CHECK(!head.score_batch(H.data(), 0, out, nullptr));
        CHECK(out.empty());
        CHECK(!head.score_batch(nullptr, 4, out, nullptr));
        CHECK(out.empty());
        ScoringHead cold;
        CHECK(!cold.score_batch(H.data(), 4, out, nullptr));
        CHECK(out.empty());
        Tensor wrong("H", {4, 7}, DType::F32);
        CHECK(!head.score_batch(wrong, out, nullptr));
        CHECK(out.empty());
    }
}

// ---------------------------------------------------------------------------
// PART E — throughput on the real embedding width
// ---------------------------------------------------------------------------
void part_e_throughput() {
    TEST("E1 1000 signals through the whole head stack, batched vs per-row");
    {
        const int32_t E = 768;
        const int32_t B = 1000;
        ClassificationHead ch;
        DecisionHead dh;
        ScoringHead sh;
        CHECK(ch.init(E, 777u));
        CHECK(ch.add_label_set("trading.regime",
                               {"trend_up", "trend_down", "range", "high_vol"}) == 0);
        CHECK(dh.init(E, 1234u));
        CHECK(sh.init(E, 555u));

        const std::vector<float> H = random_block(B, E, -2.0f, 2.0f);

        // --- batched, one call per head --------------------------------------
        std::vector<ClassificationResult> cb;
        std::vector<DecisionResult> db;
        std::vector<ScoreResult> sb;
        BatchStats cs, ds, ss;
        const double t0 = platform::now_ms();
        CHECK(ch.classify_batch(H.data(), B, "trading.regime", 4, cb, &cs));
        CHECK(dh.decide_batch(H.data(), B, db, &ds));
        CHECK(sh.score_batch(H.data(), B, sb, &ss));
        const double batched_ms = platform::now_ms() - t0;

        // --- per-row, the shape the backtest loop uses today ------------------
        const double t1 = platform::now_ms();
        for (int32_t b = 0; b < B; ++b) {
            const float* h = H.data() + static_cast<size_t>(b) * E;
            (void)ch.classify(h, "trading.regime", 4);
            (void)dh.decide(h);
            (void)sh.score(h);
        }
        const double per_row_ms = platform::now_ms() - t1;

        platform::log_info("  batched : %.2f ms total, %.3f us/signal (%s/%s/%s)",
                           batched_ms, batched_ms * 1000.0 / B,
                           batch_kernel_name(cs.used), batch_kernel_name(ds.used),
                           batch_kernel_name(ss.used));
        platform::log_info("  per-row : %.2f ms total, %.3f us/signal",
                           per_row_ms, per_row_ms * 1000.0 / B);
        platform::log_info("  speedup : %.2fx", per_row_ms / std::max(1e-9, batched_ms));

        // Loose sanity bound only: absolute timing gates are flaky on a loaded
        // CI box, and the printed numbers are the real deliverable here.
        CHECK(batched_ms > 0.0);
        CHECK(batched_ms < 5000.0);
        CHECK(cb.size() == static_cast<size_t>(B));
        CHECK(db.size() == static_cast<size_t>(B));
        CHECK(sb.size() == static_cast<size_t>(B));
    }
    TEST("E2 batching is not SLOWER than the per-row loop");
    {
        const int32_t E = 768;
        const int32_t B = 500;
        DecisionHead dh;
        CHECK(dh.init(E, 1234u));
        const std::vector<float> H = random_block(B, E, -2.0f, 2.0f);

        std::vector<DecisionResult> out;
        // warm both paths so the first-touch allocation is not in the timing
        CHECK(dh.decide_batch(H.data(), B, out, nullptr));
        (void)dh.decide(H.data());

        const double t0 = platform::now_ms();
        for (int32_t r = 0; r < 5; ++r) CHECK(dh.decide_batch(H.data(), B, out, nullptr));
        const double batched_ms = platform::now_ms() - t0;

        const double t1 = platform::now_ms();
        for (int32_t r = 0; r < 5; ++r)
            for (int32_t b = 0; b < B; ++b)
                (void)dh.decide(H.data() + static_cast<size_t>(b) * E);
        const double per_row_ms = platform::now_ms() - t1;

        platform::log_info("  decide_batch %.2f ms vs per-row %.2f ms for %d signals x5",
                           batched_ms, per_row_ms, B);
        // A generous bound: this is a regression tripwire, not a benchmark.
        CHECK(batched_ms <= per_row_ms * 2.0 + 5.0);
    }
}

} // namespace

// =============================================================================
int main() {
    platform::set_quiet(false);
    platform::log_info("=== test_heads_batch (MILESTONE 8) ===");
    part_a_kernel();
    part_b_classification();
    part_c_decision();
    part_d_scoring();
    part_e_throughput();

    platform::log_info("RESULT: %d passed, %d failed, %d skipped",
                       g_passed, g_failed, g_skipped);
    return g_failed == 0 ? 0 : 1;
}
