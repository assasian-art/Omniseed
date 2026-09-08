// =============================================================================
//  OmniSeed — bitlinear.cpp
//  Ternary weight kernels: pack/unpack, quantize, matmul-without-multiply,
//  and the QAT training step with straight-through estimator gradients.
//
//  SIMD (Phase 8): the i8 per-row kernel dispatches at runtime through a
//  function pointer selected once by CPUID: AVX2 (32-wide via vpmovsxbd +
//  FMA) -> SSE4.1 (16-wide) -> scalar. bitlinear_avx2.cpp is compiled with
//  /arch:AVX2 (-mavx2) for exactly one translation unit; everything else
//  stays baseline-compatible.
//
//  Phase 13: the PACKED TERNARY per-row kernel joins the same dispatch
//  (pshufb LUT unpack to -1/0/+1, then the i8 dot path). The ternary dot is
//  EXACTLY the scalar ternary dot in fp32 (integer weights, one fp32 scale
//  per row), so the SIMD result is bit-identical on every host — tests A/B
//  the two and gen must not drift. OMNISEED_NO_SIMD_TERNARY=1 forces the
//  scalar ternary kernel for A/B benchmarking.
// =============================================================================
#include "omniseed/core/bitlinear.h"
#include "omniseed/core/platform.h"

#include <cmath>
#include <cstdlib>

namespace omniseed {
namespace bitnet {

// ---------------------------------------------------------------------------
// Runtime dispatch (Phase 8 speed work). The kernel symbols live at bitnet
// scope (not anonymous): fwd_i8_scalar is defined in this TU, fwd_i8_avx2 /
// fwd_i8_sse41 in src/core/bitlinear_avx2.cpp (compiled with /arch:AVX2).
// ---------------------------------------------------------------------------
namespace {

using I8ForwardFn = void (*)(const int8_t*, const float*, const float*,
                             float*, int64_t, int64_t);
using TernaryRowsFn = void (*)(const uint8_t*, const float*, const float*,
                               float*, int64_t, int64_t);

I8ForwardFn   g_i8_fn         = nullptr;
TernaryRowsFn g_tern_rows_fn  = nullptr;
bool g_simd_init = false;

} // namespace

void fwd_i8_scalar(const int8_t* W, const float* scales, const float* x,
                   float* y, int64_t out_dim, int64_t in_dim);
void fwd_ternary_rows_scalar(const uint8_t* W, const float* scales,
                             const float* x, float* y,
                             int64_t out_dim, int64_t in_dim);
#if defined(OMNISEED_X86)
void fwd_i8_avx2(const int8_t* W, const float* scales, const float* x,
                 float* y, int64_t out_dim, int64_t in_dim);
void fwd_i8_sse41(const int8_t* W, const float* scales, const float* x,
                  float* y, int64_t out_dim, int64_t in_dim);
void fwd_ternary_rows_avx2(const uint8_t* W, const float* scales,
                           const float* x, float* y,
                           int64_t out_dim, int64_t in_dim);
void fwd_ternary_rows_sse41(const uint8_t* W, const float* scales,
                            const float* x, float* y,
                            int64_t out_dim, int64_t in_dim);
#endif

namespace {

void init_simd_dispatch() {
    if (g_simd_init) return;
    g_simd_init = true;
    g_i8_fn = fwd_i8_scalar;
    g_tern_rows_fn = fwd_ternary_rows_scalar;
#if defined(OMNISEED_X86)
    platform::CpuFeatures f = platform::cpu_features();
    if (f.avx2) {
        g_i8_fn = fwd_i8_avx2;
        g_tern_rows_fn = fwd_ternary_rows_avx2;
    } else if (f.sse41) {
        g_i8_fn = fwd_i8_sse41;
        g_tern_rows_fn = fwd_ternary_rows_sse41;
    }
#endif
    // A/B escape hatch: force the scalar ternary kernel for benchmarking.
    const char* no_simd = std::getenv("OMNISEED_NO_SIMD_TERNARY");
    if (no_simd != nullptr && no_simd[0] == '1') g_tern_rows_fn = fwd_ternary_rows_scalar;
    // otherwise: scalar (already the default)
}

} // namespace

// Diagnostic hook for the microbenchmark: returns the selected kernel.
I8ForwardFn simd_selected_fn() {
    init_simd_dispatch();
    return g_i8_fn;
}

TernaryRowsFn ternary_rows_selected_fn() {
    init_simd_dispatch();
    return g_tern_rows_fn;
}

// ===========================================================================
// Packing: 2 ternary weights per byte (4 bits each), low nibble = even index.
//   nibble 0x0 -> 0    0x1 -> +1    0xF -> -1   (sign-magnitude in 4 bits)
// ===========================================================================
size_t pack_ternary_packed_bytes(int64_t n_weights) {
    return static_cast<size_t>((n_weights + 1) / 2);
}

void pack_ternary(const int8_t* ternary, int64_t n, uint8_t* packed) {
    for (int64_t i = 0; i < n; i += 2) {
        const uint8_t lo = static_cast<uint8_t>(ternary[i] & 0x0F);
        const uint8_t hi = (i + 1 < n)
            ? static_cast<uint8_t>(ternary[i + 1] & 0x0F)
            : 0u;
        packed[i / 2] = static_cast<uint8_t>((hi << 4) | lo);
    }
}

void unpack_ternary(const uint8_t* packed, int64_t n, int8_t* ternary) {
    for (int64_t i = 0; i < n; i += 2) {
        const uint8_t b = packed[i / 2];
        const int8_t lo = static_cast<int8_t>(b & 0x0F);
        const int8_t hi = static_cast<int8_t>(b >> 4);
        // Map 0x0 -> 0, 0x1 -> +1, 0xF -> -1 (4-bit two's complement nibble).
        ternary[i] = lo > 8 ? static_cast<int8_t>(lo - 16) : lo;
        if (i + 1 < n) ternary[i + 1] = hi > 8 ? static_cast<int8_t>(hi - 16) : hi;
    }
}

// ===========================================================================
// Quantization
// ===========================================================================
float quantize_row(const float* w, int64_t n, int8_t* ternary_out) {
    double abs_sum = 0.0;
    for (int64_t i = 0; i < n; ++i) abs_sum += std::fabs(static_cast<double>(w[i]));
    const float scale = static_cast<float>(abs_sum / static_cast<double>(n));

    if (scale <= 1e-12f) {
        for (int64_t i = 0; i < n; ++i) ternary_out[i] = 0;
        return 0.0f;
    }

    for (int64_t i = 0; i < n; ++i) {
        const float q = w[i] / scale;
        if (q > 0.5f)       ternary_out[i] = 1;
        else if (q < -0.5f) ternary_out[i] = -1;
        else                ternary_out[i] = 0;
    }
    return scale;
}

// ===========================================================================
// Forward kernels (multiplication-free w.r.t. weights)
// ===========================================================================
void bitlinear_forward(const uint8_t* W_packed, const float* bias,
                       float scale, const float* x, float* y,
                       int64_t out_dim, int64_t in_dim) {
    //
    // Weights are addressed by WEIGHT INDEX (not byte-per-row): with 2
    // weights per byte, a row starting at odd weight index (odd in_dim)
    // begins mid-byte. weight wi lives in byte wi>>1, nibble wi&1
    // (0 = low/even index, 1 = high/odd index).
    //
    for (int64_t r = 0; r < out_dim; ++r) {
        const int64_t wbase = r * in_dim;
        float acc = 0.0f;
        for (int64_t c = 0; c < in_dim; ++c) {
            const int64_t wi = wbase + c;
            const uint8_t b = W_packed[static_cast<size_t>(wi >> 1)];
            const int8_t nib = (wi & 1) ? static_cast<int8_t>(b >> 4)
                                        : static_cast<int8_t>(b & 0x0F);
            if (nib == 1)       acc += x[c];    // +1 -> add
            else if (nib == 15) acc -= x[c];    // -1 -> subtract (0xF nibble)
        }
        y[r] = acc * scale + (bias ? bias[r] : 0.0f);
    }
}

void bitlinear_forward_batched(const uint8_t* W_packed, const float* bias,
                               float scale, const float* x, float* y,
                               int64_t batch, int64_t out_dim, int64_t in_dim) {
    for (int64_t b = 0; b < batch; ++b) {
        bitlinear_forward(W_packed, bias, scale,
                          x + b * in_dim, y + b * out_dim, out_dim, in_dim);
    }
}

void bitlinear_forward_rows(const uint8_t* W_packed, const float* scales,
                            const float* x, float* y,
                            int64_t out_dim, int64_t in_dim) {
    for (int64_t r = 0; r < out_dim; ++r) {
        const int64_t wbase = r * in_dim;
        float acc = 0.0f;
        for (int64_t c = 0; c < in_dim; ++c) {
            const int64_t wi = wbase + c;
            const uint8_t b = W_packed[static_cast<size_t>(wi >> 1)];
            const int8_t nib = (wi & 1) ? static_cast<int8_t>(b >> 4)
                                        : static_cast<int8_t>(b & 0x0F);
            if (nib == 1)       acc += x[c];
            else if (nib == 15) acc -= x[c];
        }
        y[r] = acc * scales[r];
    }
}

void bitlinear_forward_i8(const int8_t* W_i8, const float* scales,
                          const float* x, float* y,
                          int64_t out_dim, int64_t in_dim) {
    init_simd_dispatch();
    g_i8_fn(W_i8, scales, x, y, out_dim, in_dim);
}

void bitlinear_forward_rows_simd(const uint8_t* W_packed, const float* scales,
                                 const float* x, float* y,
                                 int64_t out_dim, int64_t in_dim) {
    init_simd_dispatch();
    g_tern_rows_fn(W_packed, scales, x, y, out_dim, in_dim);
}

// Scalar reference for the PACKED TERNARY per-row dot — the bit-exactness
// baseline (tests A/B against it; OMNISEED_NO_SIMD_TERNARY=1 forces it).
//
// Canonical accumulation order shared with the SSE4.1/AVX2 kernels in
// bitlinear_avx2.cpp so all three are byte-identical on every host:
//   * lane j (0..7) accumulates w*x for weight indices c ≡ j (mod 8), in
//     increasing c — 8 independent fp32 accumulators.
//   * the horizontal sum is the exact AVX2 tree:
//         ((l0+l4)+(l2+l6)) + ((l1+l5)+(l3+l7))
//   * w*x is exact in IEEE (w ∈ {-1,0,+1}); only the lane additions round,
//     in the same order in every kernel.
// Weight-index addressing (wi>>1, wi&1) makes this fully general, including
// odd in_dim where rows start mid-byte.
void fwd_ternary_rows_scalar(const uint8_t* W_packed, const float* scales,
                             const float* x, float* y,
                             int64_t out_dim, int64_t in_dim) {
    for (int64_t r = 0; r < out_dim; ++r) {
        const int64_t wbase = r * in_dim;
        float l[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (int64_t c = 0; c < in_dim; ++c) {
            const int64_t wi = wbase + c;
            const uint8_t b = W_packed[static_cast<size_t>(wi >> 1)];
            const int8_t nib = (wi & 1) ? static_cast<int8_t>(b >> 4)
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

// scalar reference used as the baseline impl (and by tests)
void fwd_i8_scalar(const int8_t* W_i8, const float* scales,
                   const float* x, float* y,
                   int64_t out_dim, int64_t in_dim) {
    for (int64_t r = 0; r < out_dim; ++r) {
        const int8_t* row = W_i8 + r * in_dim;
        float acc = 0.0f;
        for (int64_t c = 0; c < in_dim; ++c) {
            const int8_t w = row[c];
            if (w != 0) acc += static_cast<float>(w) * x[c];
        }
        y[r] = acc * scales[r];
    }
}

// ===========================================================================
// Training step
// ===========================================================================
void bitlinear_train_step(const float* X, const int8_t* Wq, float* W,
                          float scale, float* dW, float* dX,
                          const float* dY, int64_t rows,
                          int64_t in_dim, int64_t out_dim, float lr) {
    // ---- Backward (only if upstream gradient provided) ----
    if (dY != nullptr) {
        // dW[r, c] = sum_b dY[b, r] * X[b, c]
        for (int64_t r = 0; r < out_dim; ++r) {
            for (int64_t c = 0; c < in_dim; ++c) {
                float acc = 0.0f;
                for (int64_t b = 0; b < rows; ++b) {
                    acc += dY[b * out_dim + r] * X[b * in_dim + c];
                }
                dW[r * in_dim + c] = acc;
            }
        }

        // dX[b, c] = scale * sum_r dY[b, r] * Wq[r, c]
        if (dX != nullptr) {
            for (int64_t b = 0; b < rows; ++b) {
                for (int64_t c = 0; c < in_dim; ++c) {
                    float acc = 0.0f;
                    for (int64_t r = 0; r < out_dim; ++r) {
                        const int8_t w = Wq[r * in_dim + c];
                        if (w == 0) continue;
                        acc += static_cast<float>(w) * dY[b * out_dim + r];
                    }
                    dX[b * in_dim + c] = scale * acc;
                }
            }
        }

        // ---- Update master weights: W -= lr * dW ----
        const int64_t total = out_dim * in_dim;
        for (int64_t i = 0; i < total; ++i) W[i] -= lr * dW[i];
    }
}

} // namespace bitnet
} // namespace omniseed
