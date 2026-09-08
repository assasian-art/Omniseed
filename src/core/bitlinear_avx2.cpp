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

// ===========================================================================
// PACKED TERNARY kernels (Phase 13): 2 weights/byte, nibble LUT unpack.
//
// Bit-exactness contract: all three ternary kernels (scalar in bitlinear.cpp,
// SSE4.1 + AVX2 here) implement ONE canonical accumulation order, so they are
// byte-identical on every host:
//   * lane j (j = 0..7) accumulates w*x for weight indices c ≡ j (mod 8),
//     in increasing c — 8 independent fp32 lane accumulators.
//   * any remainder (in_dim not a multiple of 8) folds into lane c&7 in
//     increasing c order.
//   * the horizontal sum is the exact AVX2 tree:
//         ((l0+l4)+(l2+l6)) + ((l1+l5)+(l3+l7))
//   * w*x is exact in IEEE (w ∈ {-1,0,+1}), so FMA vs mul+add round the
//     same; only the lane additions round, and they happen in the same
//     order in every kernel.
// Rows must start on a byte boundary (even in_dim); odd in_dim falls back to
// the canonical scalar (which handles weight-index parity via wi>>1/wi&1).
// ===========================================================================

namespace {

// nibble value -> int8 weight: 0x0 -> 0, 0x1 -> +1, 0xF -> -1 (0xFF), else 0
const __m128i kTernaryLut =
    _mm_setr_epi8(0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1);
const __m128i kNibbleMask = _mm_set1_epi8(0x0F);

} // namespace

void fwd_ternary_rows_sse41(const uint8_t* W_packed, const float* scales,
                            const float* x, float* y,
                            int64_t out_dim, int64_t in_dim) {
    if (in_dim % 2 != 0) {  // rows start mid-byte: canonical scalar handles it
        fwd_ternary_rows_scalar(W_packed, scales, x, y, out_dim, in_dim);
        return;
    }
    const int64_t row_bytes = in_dim / 2;
    for (int64_t r = 0; r < out_dim; ++r) {
        const uint8_t* row = W_packed + r * row_bytes;
        __m128 acc0 = _mm_setzero_ps();   // canonical lanes 0..3
        __m128 acc1 = _mm_setzero_ps();   // canonical lanes 4..7
        int64_t c = 0;
        for (; c + 8 <= in_dim; c += 8) {
            // 4 bytes = 8 weights: byte k = w_{2k} (lo nibble) | w_{2k+1} (hi)
            const __m128i b = _mm_cvtsi32_si128(
                *reinterpret_cast<const int32_t*>(row + c / 2));
            // NOTE: _mm_shuffle_epi8 zeroes output for index bytes with bit7
            // set — always mask the nibble BEFORE the shuffle (ternary bytes
            // like 0xF1 would otherwise silently drop their even weight).
            const __m128i lo = _mm_shuffle_epi8(
                kTernaryLut, _mm_and_si128(b, kNibbleMask));            // even
            const __m128i hi = _mm_shuffle_epi8(
                kTernaryLut, _mm_and_si128(_mm_srli_epi16(b, 4), kNibbleMask));
            // lo/hi hold the 8 weights in their LOW bytes (0..7); one unpacklo
            // interleaves ALL of them in weight order: [w0..w7].
            const __m128i w_all = _mm_unpacklo_epi8(lo, hi);
            const __m128i w0 = w_all;                          // w0..w3
            const __m128i w1 = _mm_srli_si128(w_all, 4);       // w4..w7
            acc0 = _mm_add_ps(acc0,
                _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepi8_epi32(w0)),
                           _mm_loadu_ps(x + c)));
            acc1 = _mm_add_ps(acc1,
                _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepi8_epi32(w1)),
                           _mm_loadu_ps(x + c + 4)));
        }
        float l[8];
        _mm_storeu_ps(l, acc0);
        _mm_storeu_ps(l + 4, acc1);
        for (; c < in_dim; ++c) {   // tail folds into lanes, canonical order
            const uint8_t b = row[c / 2];
            const int8_t nib = (c & 1) ? static_cast<int8_t>(b >> 4)
                                       : static_cast<int8_t>(b & 0x0F);
            const float wf = (nib == 1) ? 1.0f : (nib == 15) ? -1.0f : 0.0f;
            l[c & 7] += wf * x[c];
        }
        const float a04 = l[0] + l[4];
        const float a26 = l[2] + l[6];
        const float a15 = l[1] + l[5];
        const float a37 = l[3] + l[7];
        y[r] = ((a04 + a26) + (a15 + a37)) * scales[r];
    }
}

void fwd_ternary_rows_avx2(const uint8_t* W_packed, const float* scales,
                           const float* x, float* y,
                           int64_t out_dim, int64_t in_dim) {
    if (in_dim % 2 != 0) {  // rows start mid-byte: canonical scalar handles it
        fwd_ternary_rows_scalar(W_packed, scales, x, y, out_dim, in_dim);
        return;
    }
    const int64_t row_bytes = in_dim / 2;
    for (int64_t r = 0; r < out_dim; ++r) {
        const uint8_t* row = W_packed + r * row_bytes;
        __m256 acc = _mm256_setzero_ps();   // canonical lanes 0..7
        int64_t c = 0;
        for (; c + 16 <= in_dim; c += 16) {
            // 8 bytes = 16 weights: byte k = w_{2k} (lo nibble) | w_{2k+1} (hi)
            const __m128i b = _mm_loadl_epi64(
                reinterpret_cast<const __m128i*>(row + c / 2));
            // NOTE: _mm_shuffle_epi8 zeroes output for index bytes with bit7
            // set — always mask the nibble BEFORE the shuffle (ternary bytes
            // like 0xF1 would otherwise silently drop their even weight).
            const __m128i lo = _mm_shuffle_epi8(
                kTernaryLut, _mm_and_si128(b, kNibbleMask));            // even
            const __m128i hi = _mm_shuffle_epi8(
                kTernaryLut, _mm_and_si128(_mm_srli_epi64(b, 4), kNibbleMask));
            // lo/hi hold the 16 weights in their LOW bytes (0..7); one
            // unpacklo interleaves ALL of them in weight order: [w0..w15].
            const __m128i w_all = _mm_unpacklo_epi8(lo, hi);
            const __m128i w0 = w_all;                          // w0..w7
            const __m128i w1 = _mm_srli_si128(w_all, 8);       // w8..w15
            acc = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w0)),
                _mm256_loadu_ps(x + c), acc);
            acc = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w1)),
                _mm256_loadu_ps(x + c + 8), acc);
        }
        float l[8];
        _mm256_storeu_ps(l, acc);
        for (; c < in_dim; ++c) {   // tail folds into lanes, canonical order
            const uint8_t b = row[c / 2];
            const int8_t nib = (c & 1) ? static_cast<int8_t>(b >> 4)
                                       : static_cast<int8_t>(b & 0x0F);
            const float wf = (nib == 1) ? 1.0f : (nib == 15) ? -1.0f : 0.0f;
            l[c & 7] += wf * x[c];
        }
        const float a04 = l[0] + l[4];
        const float a26 = l[2] + l[6];
        const float a15 = l[1] + l[5];
        const float a37 = l[3] + l[7];
        y[r] = ((a04 + a26) + (a15 + a37)) * scales[r];
    }
}

} // namespace bitnet
} // namespace omniseed

#endif // OMNISEED_X86
