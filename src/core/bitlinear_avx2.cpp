// =============================================================================
//  OmniSeed — bitlinear_avx2.cpp
//  SIMD i8 dot-product kernels for bitlinear_forward_i8.
//
//  This translation unit is the ONLY one compiled with /arch:AVX2 (-mavx2),
//  selected in CMake on x86/x64 targets. Runtime dispatch in bitlinear.cpp
//  picks AVX2 only when CPUID reports OS+CPU support; machines without AVX2
//  take the scalar kernel in bitlinear.cpp (the SSE4.1 kernel here is only
//  dispatched when this TU was NOT built with AVX2, because MSVC may emit
//  VEX encodings anywhere in the TU).
//
//  Core trick per 8 weights: _mm_loadl_epi64 (8 i8) -> _mm256_cvtepi8_epi32
//  (8 i32) -> _mm256_cvtepi32_ps (8 f32) -> _mm256_fmadd_ps with 8 x-lanes.
//  One horizontal add per row. 768-wide rows = 96 FMA iterations.
// =============================================================================
#include "omniseed/core/bitlinear.h"

#if defined(OMNISEED_X86)

#include <immintrin.h>

namespace omniseed {
namespace bitnet {

void fwd_i8_avx2(const int8_t* W_i8, const float* scales, const float* x,
                 float* y, int64_t out_dim, int64_t in_dim) {
    for (int64_t r = 0; r < out_dim; ++r) {
        const int8_t* row = W_i8 + r * in_dim;
        __m256 acc = _mm256_setzero_ps();
        int64_t c = 0;
        for (; c + 8 <= in_dim; c += 8) {
            const __m128i b8 = _mm_loadl_epi64(
                reinterpret_cast<const __m128i*>(row + c));
            const __m256 wf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b8));
            acc = _mm256_fmadd_ps(wf, _mm256_loadu_ps(x + c), acc);
        }
        // horizontal sum
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
        lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
        float accf = _mm_cvtss_f32(lo);
        for (; c < in_dim; ++c) {
            const int8_t w = row[c];
            if (w != 0) accf += static_cast<float>(w) * x[c];
        }
        y[r] = accf * scales[r];
    }
}

void fwd_i8_sse41(const int8_t* W_i8, const float* scales, const float* x,
                  float* y, int64_t out_dim, int64_t in_dim) {
    for (int64_t r = 0; r < out_dim; ++r) {
        const int8_t* row = W_i8 + r * in_dim;
        __m128 acc = _mm_setzero_ps();
        int64_t c = 0;
        for (; c + 4 <= in_dim; c += 4) {
            const __m128i b8 = _mm_loadl_epi64(
                reinterpret_cast<const __m128i*>(row + c));
            const __m128 wf = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(b8));
            acc = _mm_add_ps(acc, _mm_mul_ps(wf, _mm_loadu_ps(x + c)));
        }
        acc = _mm_add_ps(acc, _mm_movehl_ps(acc, acc));
        acc = _mm_add_ss(acc, _mm_shuffle_ps(acc, acc, 1));
        float accf = _mm_cvtss_f32(acc);
        for (; c < in_dim; ++c) {
            const int8_t w = row[c];
            if (w != 0) accf += static_cast<float>(w) * x[c];
        }
        y[r] = accf * scales[r];
    }
}

} // namespace bitnet
} // namespace omniseed

#endif // OMNISEED_X86
