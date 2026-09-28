// =============================================================================
//  OmniSeed — src/core/batch_gemm.cpp
//  The scalar kernel, the runtime dispatch, and the plan report.
//  See include/omniseed/core/batch_gemm.h for the two contracts.
//
//  WHY THE SCALAR LOOP ORDER IS LOAD-BEARING. The inner loop must run
//  e = 0..E-1 into a SINGLE accumulator, in that order, because that is what
//  every head's per-row path does. Floating-point addition is not associative,
//  so reordering (or blocking) this loop would make the batch result differ
//  from the per-row result in the last bits — and the whole point of the Scalar
//  kernel is that `batch == per-row` is testable with `==`, not with a
//  tolerance. If you want a faster loop, add a kernel; do not change this one.
// =============================================================================
#include "omniseed/core/batch_gemm.h"

#include "omniseed/core/platform.h"

#include <cstdio>
#include <string>

#if defined(OMNISEED_X86)
namespace omniseed {
// Defined in batch_gemm_avx2.cpp, which is the only TU compiled with /arch:AVX2.
void batch_gemm_avx2(const float* H, const float* W, const float* bias, float* out,
                     int32_t B, int32_t E, int32_t L, float scale);
} // namespace omniseed
#endif

namespace omniseed {

const char* batch_kernel_name(BatchKernel k) {
    switch (k) {
        case BatchKernel::Auto:   return "auto";
        case BatchKernel::Scalar: return "scalar";
        case BatchKernel::Simd:   return "simd";
        default:                  return "unknown";
    }
}

std::string BatchStats::to_json() const {
    char buf[320];
    std::snprintf(buf, sizeof(buf),
                  "{\"rows\":%d,\"outputs\":%d,\"us_total\":%.3f,"
                  "\"us_per_row\":%.3f,\"requested\":\"%s\",\"used\":\"%s\","
                  "\"simd_fell_back\":%s}",
                  rows, outputs, us_total, us_per_row,
                  batch_kernel_name(requested), batch_kernel_name(used),
                  simd_fell_back ? "true" : "false");
    return std::string(buf);
}

bool batch_simd_available() {
#if defined(OMNISEED_X86)
    return platform::cpu_features().avx2;
#else
    return false;
#endif
}

void batch_gemm_scalar(const float* H, const float* W, const float* bias, float* out,
                       int32_t B, int32_t E, int32_t L, float scale) {
    for (int32_t b = 0; b < B; ++b) {
        const float* h = H + static_cast<size_t>(b) * static_cast<size_t>(E);
        float* o = out + static_cast<size_t>(b) * static_cast<size_t>(L);
        for (int32_t l = 0; l < L; ++l) {
            const float* w = W + static_cast<size_t>(l) * static_cast<size_t>(E);
            // Same order, same single accumulator, same place the scale lands
            // as ClassificationHead::classify's per-row loop.
            float acc = bias[l];
            for (int32_t e = 0; e < E; ++e) acc += w[e] * h[e];
            o[l] = acc * scale;
        }
    }
}

void batch_gemm(const float* H, const float* W, const float* bias, float* out,
                int32_t B, int32_t E, int32_t L, float scale, BatchKernel kernel,
                BatchGemmPlan* plan) {
    BatchKernel used = BatchKernel::Scalar;
    bool fell_back = false;

    // Auto resolves to "the fastest kernel this machine has". An explicit
    // Simd request that cannot be honoured is a FALLBACK and is reported as
    // one; an Auto that lands on Scalar is not, because that is the design.
    const bool simd_wanted = (kernel == BatchKernel::Simd);
    const BatchKernel want = (kernel == BatchKernel::Auto) ? BatchKernel::Simd : kernel;

    if (plan != nullptr) {
        plan->rows            = B;
        plan->width           = E;
        plan->outputs         = L;
        plan->macs            = static_cast<int64_t>(B) * static_cast<int64_t>(E) *
                                static_cast<int64_t>(L);
        plan->requested       = kernel;
        plan->used            = BatchKernel::Scalar;
        plan->simd_fell_back  = false;
    }

    // An empty batch is a no-op, not an error: a backtest loop may legitimately
    // hand over zero rows. A null pointer with a non-zero count is a caller bug
    // and is refused rather than dereferenced.
    if (B <= 0 || E <= 0 || L <= 0) return;
    if (H == nullptr || W == nullptr || out == nullptr) return;

    if (want == BatchKernel::Simd) {
#if defined(OMNISEED_X86)
        if (platform::cpu_features().avx2) {
            batch_gemm_avx2(H, W, bias, out, B, E, L, scale);
            used = BatchKernel::Simd;
        } else {
            fell_back = simd_wanted;
        }
#else
        fell_back = simd_wanted;
#endif
    }

    if (used == BatchKernel::Scalar) batch_gemm_scalar(H, W, bias, out, B, E, L, scale);

    if (plan != nullptr) {
        plan->used           = used;
        plan->simd_fell_back = fell_back;
    }
}

} // namespace omniseed
