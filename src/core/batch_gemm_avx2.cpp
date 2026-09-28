// =============================================================================
//  OmniSeed — batch_gemm_avx2.cpp
//  The AVX2 kernel for batch_gemm. Compiled with /arch:AVX2 (-mavx2;-mfma) and
//  only ever entered when CPUID reports AVX2 — see batch_gemm.cpp's dispatch.
//
//  THIS KERNEL IS NOT BIT-IDENTICAL TO THE SCALAR ONE, BY CONSTRUCTION.
//  It accumulates eight independent lanes and reduces them at the end, whereas
//  the scalar kernel sums into one running accumulator. Floating-point addition
//  is not associative, so the two differ in the last bits. That is the whole
//  reason batch_gemm.h documents two contracts: Scalar is for exactness tests,
//  Simd is for throughput. Do not "fix" this by forcing a serial reduction —
//  that is the scalar kernel, and it is 8x slower.
//
//  Layout note: one output row per `l`, one hidden row per `b`. The hidden row
//  is re-read L times (L <= 7 here) and stays L1-resident, so the loop order is
//  deliberate rather than an oversight.
// =============================================================================
#include "omniseed/core/batch_gemm.h"

#if defined(OMNISEED_X86)

#include <immintrin.h>

#include <cstddef>

namespace omniseed {

void batch_gemm_avx2(const float* H, const float* W, const float* bias, float* out,
                     int32_t B, int32_t E, int32_t L, float scale) {
    for (int32_t b = 0; b < B; ++b) {
        const float* h = H + static_cast<size_t>(b) * static_cast<size_t>(E);
        float* o = out + static_cast<size_t>(b) * static_cast<size_t>(L);
        for (int32_t l = 0; l < L; ++l) {
            const float* w = W + static_cast<size_t>(l) * static_cast<size_t>(E);
            __m256 acc = _mm256_setzero_ps();
            int32_t e = 0;
            for (; e + 8 <= E; e += 8) {
                const __m256 wv = _mm256_loadu_ps(w + e);
                const __m256 hv = _mm256_loadu_ps(h + e);
                acc = _mm256_fmadd_ps(wv, hv, acc);
            }
            // Horizontal reduce the eight lanes.
            __m128 lo = _mm256_castps256_ps128(acc);
            __m128 hi = _mm256_extractf128_ps(acc, 1);
            lo = _mm_add_ps(lo, hi);
            lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
            lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
            float total = _mm_cvtss_f32(lo);
            // Tail for E % 8 != 0.
            for (; e < E; ++e) total += w[e] * h[e];
            o[l] = (total + bias[l]) * scale;
        }
    }
}

} // namespace omniseed

#endif  // OMNISEED_X86
