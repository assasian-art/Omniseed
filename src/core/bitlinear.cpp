// =============================================================================
//  OmniSeed — bitlinear.cpp
//  Ternary weight kernels: pack/unpack, quantize, matmul-without-multiply,
//  and the QAT training step with straight-through estimator gradients.
// =============================================================================
#include "omniseed/core/bitlinear.h"
#include "omniseed/core/platform.h"

#include <cmath>

namespace omniseed {
namespace bitnet {

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
