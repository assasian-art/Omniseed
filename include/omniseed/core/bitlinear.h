// =============================================================================
//  OmniSeed — bitlinear.h
//  BitNet b1.58 ternary weight engine: {-1, 0, +1} weights, 2 weights/byte.
//
//  Storage layout (sign-magnitude, 2 weights per byte):
//      high nibble = weight 2i+1, low nibble = weight 2i
//      nibble 0x0 = 0, 0x1 = +1, 0xF = -1
//
//  Computation: a ternary matmul needs NO multiplications for the weights —
//  +1 contributes +x, -1 contributes -x, 0 skips. A single per-row fp32
//  scale (abs-mean of the master weights) restores magnitude, exactly as
//  BitNet b1.58 defines BitLinear: y = scale * (W_ternary . x) + bias.
//
//  Memory math (blueprint target): 0.8B params -> ~460MB raw, ~1.6GB fp16,
//  but 0.5 B/param ternary -> ~460MB. With sub-1B distilled RWKV this lands
//  the core brain at ~50-90MB, inside the <300MB system budget.
// =============================================================================
#pragma once

#include "omniseed/core/tensor.h"

#include <cstdint>
#include <vector>

namespace omniseed {
namespace bitnet {

// ---------------------------------------------------------------------------
// Packing
// ---------------------------------------------------------------------------
// Bytes needed to store n ternary weights (2 per byte, rounded up).
size_t pack_ternary_packed_bytes(int64_t n_weights);

// Packs n ternary weights (values -1/0/+1) into packed bytes.
void pack_ternary(const int8_t* ternary, int64_t n, uint8_t* packed);

// Unpacks packed bytes into n ternary weights.
void unpack_ternary(const uint8_t* packed, int64_t n, int8_t* ternary);

// ---------------------------------------------------------------------------
// Quantization (used by converters & QAT-style fine-tuning loops)
// ---------------------------------------------------------------------------
// Quantizes a fp32 master weight row to ternary with per-output scale.
// scale = mean(|W|) over the row; W_q = round(clip(W / scale, -1, 1)).
// Returns the scale (0 if all weights are zero).
float quantize_row(const float* w, int64_t n, int8_t* ternary_out);

// ---------------------------------------------------------------------------
// Forward kernels
// ---------------------------------------------------------------------------
// y = scale * (W_ternary . x) [+ bias]
//   W_packed : [out_dim * in_dim] ternary weights, 2/byte
//   bias     : optional [out_dim] fp32
//   x        : [in_dim] fp32
//   y        : [out_dim] fp32
// Row-major: W[r * in_dim + c].
void bitlinear_forward(const uint8_t* W_packed, const float* bias,
                       float scale, const float* x, float* y,
                       int64_t out_dim, int64_t in_dim);

// Batched: x is [batch, in_dim] row-major, y is [batch, out_dim].
void bitlinear_forward_batched(const uint8_t* W_packed, const float* bias,
                               float scale, const float* x, float* y,
                               int64_t batch, int64_t out_dim, int64_t in_dim);

// Per-ROW scales (real converted checkpoints): y[r] = scales[r] * (W . x).
// scales has out_dim fp32 entries (the row's master-weight absmean).
void bitlinear_forward_rows(const uint8_t* W_packed, const float* scales,
                            const float* x, float* y,
                            int64_t out_dim, int64_t in_dim);

// Per-row int8: W_i8 has out_dim*in_dim signed bytes; scale[r] = max|row|/127.
// y[r] = scale[r] * sum_c W_i8[r,c] * x[c].
void bitlinear_forward_i8(const int8_t* W_i8, const float* scales,
                          const float* x, float* y,
                          int64_t out_dim, int64_t in_dim);

// Per-row PACKED ternary (2 weights/byte, low nibble = even index, layout
// identical to bitlinear_forward_rows): y[r] = scales[r] * (W . x).
// Same SIMD/dispatch contract as bitlinear_forward_i8 (see bitlinear.cpp):
// results are byte-identical to the scalar path on every host.
void bitlinear_forward_rows_simd(const uint8_t* W_packed, const float* scales,
                                 const float* x, float* y,
                                 int64_t out_dim, int64_t in_dim);

// Scalar reference for the packed-ternary per-row dot — the bit-exactness
// baseline (defined in bitlinear.cpp; tests A/B against it).
void fwd_ternary_rows_scalar(const uint8_t* W_packed, const float* scales,
                             const float* x, float* y,
                             int64_t out_dim, int64_t in_dim);

// Kernel-selection override for the packed-ternary per-row dot (tests/bench).
// Auto = cpuid dispatch, honoring OMNISEED_FORCE_SSE4_TERNARY=1 and
// OMNISEED_NO_SIMD_TERNARY=1 at first init; an explicit pick forces that
// kernel regardless of the host (Sse41/Avx2 picks are no-ops on non-x86).
enum class TernaryKernel { Auto, Scalar, Sse41, Avx2 };
void force_ternary_kernel(TernaryKernel k);

// ---------------------------------------------------------------------------
// Training path (BitLinear QAT / LoRA fine-tuning support)
// ---------------------------------------------------------------------------
// One training step for a single BitLinear layer:
//   1. Forward:  y = scale * (Wq . x) + bias           (Wq = ternary(W))
//   2. Backward: dW = x^T . dY  (straight-through estimator gradient
//                                 w.r.t. the fp32 master weights W)
//                dX = scale * (Wq^T . dY)
//   3. Update:   W -= lr * dW   (master weights stay fp32; ternary Wq is
//                                recomputed from W at the next forward)
//
// shapes: X [rows, in_dim], dY [rows, out_dim], W/Wq/dW [out_dim, in_dim]
// dX may be nullptr. If dY is nullptr, only the forward pass runs and y is
// filled.
void bitlinear_train_step(const float* X, const int8_t* Wq, float* W,
                          float scale, float* dW, float* dX,
                          const float* dY, int64_t rows,
                          int64_t in_dim, int64_t out_dim, float lr);

} // namespace bitnet
} // namespace omniseed
