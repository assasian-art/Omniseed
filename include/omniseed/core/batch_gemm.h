// =============================================================================
//  OmniSeed — batch_gemm.h
//  [B, E] x [L, E]^T + bias[L], with an optional scale — the batched form of the
//  single matvec every head already performs.
//
//  WHY THIS EXISTS. Every head in this tree answers one h[E] at a time. That is
//  the right shape for a live turn and the wrong shape for a BACKTEST, where a
//  thousand bars have already been collected and the per-call overhead (a
//  scratch resize, a `top_k` vector, an `order` vector, two clock reads) is paid
//  a thousand times for a few microseconds of arithmetic each.
//
//  WHAT IS AND IS NOT BATCHED. This batches the HEAD READOUT, which is a plain
//  GEMM and embarrassingly parallel. It does NOT batch the BACKBONE: this
//  tree's RWKV-7 is the scalar recurrent form, so h[E] for bar n+1 still needs
//  the state from bar n. "One forward pass for 1000 signals" is therefore not
//  what this delivers, and claiming it would be false — what it delivers is one
//  GEMM for 1000 readouts, after the 1000 (necessarily sequential) forwards.
//  See docs/BATCH.md.
//
//  TWO KERNELS, TWO DIFFERENT CONTRACTS. This distinction is the whole reason
//  the file is documented this carefully:
//
//    Scalar  sums e = 0..E-1 into ONE running accumulator, exactly like the
//            per-row head path. Result: BIT-IDENTICAL to a per-row loop.
//            tests/test_heads_batch.cpp asserts equality, not a tolerance.
//
//    Simd    sums eight lanes and then reduces them. Floating-point addition is
//            not associative, so the result DIFFERS from Scalar in the last
//            bits. The contract is a relative tolerance, and the kernel falls
//            back to Scalar on a CPU without AVX2.
//
//  Never compare a Simd result with `==`. That is what the Scalar kernel is for.
// =============================================================================
#pragma once

#include <cstdint>
#include <string>

namespace omniseed {

// Which kernel to use. `Auto` is the default: it picks Simd when the CPU
// supports it and Scalar otherwise.
enum class BatchKernel : int32_t {
    Auto = 0,     // Simd when the CPU supports it, else Scalar
    Scalar,       // bit-identical to the per-row head path
    Simd,         // AVX2 when available; falls back to Scalar otherwise
    COUNT
};

const char* batch_kernel_name(BatchKernel k);

// True when this build has an AVX2 kernel AND the running CPU reports AVX2.
bool batch_simd_available();

// What actually ran, for logging and for tests.
struct BatchGemmPlan {
    int32_t     rows      = 0;   // B
    int32_t     width     = 0;   // E
    int32_t     outputs   = 0;   // L
    int64_t     macs      = 0;   // B * E * L
    BatchKernel requested = BatchKernel::Auto;
    BatchKernel used      = BatchKernel::Scalar;   // differs when Simd fell back
    bool        simd_fell_back = false;
};

// ---------------------------------------------------------------------------
// BatchStats — what a batched HEAD call actually did.
//
// `us_total` covers the whole batched readout (logits + softmax + reduction),
// not just the GEMM, so it is the number to compare against B separate head
// calls. `us_per_row` is `us_total / rows`.
//
// Timing is reported, never asserted on: a CI box under load makes absolute
// microsecond gates flaky, so the tests print these and gate only on a loose
// sanity bound. Treat them as measurements, not as a performance contract.
// ---------------------------------------------------------------------------
struct BatchStats {
    int32_t     rows      = 0;
    int32_t     outputs   = 0;   // rows * L — the matvec count
    double      us_total  = 0.0;
    double      us_per_row = 0.0;
    BatchKernel requested = BatchKernel::Auto;
    BatchKernel used      = BatchKernel::Scalar;
    bool        simd_fell_back = false;

    std::string to_json() const;
};

// The scalar kernel on its own, exposed so tests can compare a Simd result
// against the exact reference without going through the dispatcher.
void batch_gemm_scalar(const float* H, const float* W, const float* bias, float* out,
                       int32_t B, int32_t E, int32_t L, float scale);

// out[b, l] = ( bias[l] + sum_e W[l, e] * H[b, e] ) * scale
//
// H   [B, E] row-major   (B hidden states, back to back)
// W   [L, E] row-major   (one projection row per output)
// out [B, L] row-major
//
// `scale` is applied to the FINISHED accumulator, which is where the heads
// apply their softmax temperature (`acc *= 1/T`). Passing 1.0f is an exact
// no-op, so an uncalibrated head takes byte-for-byte the same path.
//
// B == 0, E == 0 or L == 0 is a no-op and is NOT an error — an empty batch is
// a legitimate thing for a backtest loop to hand over. A null pointer with a
// non-zero count is refused (the function returns without writing) rather than
// dereferenced.
void batch_gemm(const float* H, const float* W, const float* bias, float* out,
                int32_t B, int32_t E, int32_t L, float scale, BatchKernel kernel,
                BatchGemmPlan* plan = nullptr);

} // namespace omniseed
