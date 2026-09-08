// =============================================================================
//  OmniSeed — test_qat_ab.cpp
//  QAT-model ternary-kernel A/B: the packed-ternary SIMD kernels (scalar,
//  SSE4.1, AVX2) implement ONE canonical accumulation order and must produce
//  a BYTE-IDENTICAL greedy continuation on the real QAT model, no matter
//  which kernel the host dispatcher selects.
//
//  Skips cleanly when models/rwkv7-0.1B-ternary-qat.gguf is absent (CI /
//  fresh clones). Registered as ctest `omniseed_qat_ternary` (WD = repo
//  root). The kernel override is bitnet::force_ternary_kernel — an explicit
//  per-call pick, so all three kernels run inside ONE process on ANY host
//  (Sse41/Avx2 picks are no-ops on non-x86 hosts and fall back to the
//  cpuid-selected kernel, which makes the A/B trivially identical there).
// =============================================================================
#include "omniseed/core/bitlinear.h"
#include "omniseed/core/gguf_loader.h"
#include "omniseed/core/platform.h"
#include "omniseed/core/rwkv.h"
#include "omniseed/core/tokenizer.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace omniseed;

static int g_pass = 0, g_fail = 0;
#define CHECK(cond)                                                          \
    do {                                                                     \
        if (cond) { ++g_pass; }                                              \
        else {                                                               \
            ++g_fail;                                                        \
            std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
        }                                                                    \
    } while (0)

namespace {

constexpr const char* kQatModelPath = "models/rwkv7-0.1B-ternary-qat.gguf";
constexpr int32_t kEosId = 0;   // world models: eos = 0 (unused pad)

// greedy-generate n tokens from ids (same helper as test_real_weights)
std::vector<int32_t> greedy(RwkvModel& model, const std::vector<int32_t>& ids,
                            int n) {
    RwkvState st;
    model.init_state(st);
    Tensor logits("logits", {model.config().n_vocab}, DType::F32);
    for (int32_t id : ids) model.forward(id, st, logits);
    std::vector<int32_t> out;
    for (int i = 0; i < n; ++i) {
        const float* lp = logits.f32();
        int32_t best = 0;
        float bv = lp[0];
        for (int32_t t = 1; t < model.config().n_vocab; ++t)
            if (lp[t] > bv) { bv = lp[t]; best = t; }
        out.push_back(best);
        if (best == kEosId) break;
        model.forward(best, st, logits);
    }
    return out;
}

} // namespace

int main() {
    platform::log_info("OmniSeed QAT ternary-kernel A/B test");

    std::FILE* f = nullptr;
#ifdef _MSC_VER
    if (fopen_s(&f, kQatModelPath, "rb") != 0) f = nullptr;
#else
    f = std::fopen(kQatModelPath, "rb");
#endif
    if (f == nullptr) {
        std::printf("  [skip] %s not present — QAT A/B test skipped\n",
                    kQatModelPath);
        std::printf("RESULT: 0 passed, 0 failed (skipped)\n");
        return 0;
    }
    std::fclose(f);

    RwkvModel model;
    if (!model.load(kQatModelPath)) {
        std::printf("  [skip] model load failed: %s\n", model.error().c_str());
        std::printf("RESULT: 0 passed, 0 failed (model failed to load)\n");
        return 0;
    }
    const auto& cfg = model.config();
    std::printf("  layers=%d embd=%d vocab=%d heads=%d/%d ffn=%d\n",
                cfg.n_layers, cfg.n_embd, cfg.n_vocab, cfg.n_heads,
                cfg.head_size, cfg.ffn_inter);

    // Byte-level world vocab: "Hello" = bytes + 1 (no tokenizer needed).
    std::vector<int32_t> ids;
    for (char c : std::string("Hello"))
        ids.push_back(static_cast<int32_t>(
                          static_cast<unsigned char>(c)) + 1);

    // Baseline: cpuid dispatch (AVX2 on this box). Then force each kernel
    // explicitly and assert the greedy continuation is byte-identical.
    const std::vector<int32_t> base = greedy(model, ids, 24);
    CHECK(!base.empty());

    const struct { const char* name; bitnet::TernaryKernel k; } modes[] = {
        {"scalar", bitnet::TernaryKernel::Scalar},
        {"sse41",  bitnet::TernaryKernel::Sse41},
        {"avx2",   bitnet::TernaryKernel::Avx2},
    };
    for (const auto& m : modes) {
        bitnet::force_ternary_kernel(m.k);
        const std::vector<int32_t> run = greedy(model, ids, 24);
        CHECK(run.size() == base.size());
        CHECK(run == base);
        std::printf("  %s kernel: %s\n", m.name,
                    run == base ? "byte-identical" : "MISMATCH");
        if (run != base) {
            std::printf("    base: "); for (int32_t t : base) std::printf(" %d", t);
            std::printf("\n    %s: ", m.name); for (int32_t t : run) std::printf(" %d", t);
            std::printf("\n");
        }
    }
    // Restore automatic dispatch (subsequent checks in this process).
    bitnet::force_ternary_kernel(bitnet::TernaryKernel::Auto);

    // The QAT model alone must stay inside the <300 MB budget (mmap lazy
    // page-in; skipped under OMNISEED_FORCE_FREAD=1 where the file is
    // deliberately heap-resident).
    const char* force = std::getenv("OMNISEED_FORCE_FREAD");
    const bool forced_fread =
        force != nullptr && (force[0] == '1' || force[0] == 'y' ||
                             force[0] == 'Y');
    if (forced_fread) {
        std::printf("  (memory budget check skipped: OMNISEED_FORCE_FREAD=1)\n");
    } else {
        CHECK(platform::peak_rss_bytes() < 300ull * 1024 * 1024);
    }

    std::printf("RESULT: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}