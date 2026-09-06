// =============================================================================
//  OmniSeed — unicompress.cpp
//  UniCompress visual token compression: importance-scored merging that
//  reduces token count up to 4x while preserving semantic content.
// =============================================================================
#include "omniseed/vision/vision.h"
#include "omniseed/core/platform.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

namespace omniseed {

std::vector<Tensor> UniCompress::compress(const Tensor& tokens,
                                          int32_t grid_side,
                                          std::vector<float>* out_weights) const {
    std::vector<Tensor> result;
    if (tokens.numel() == 0 || grid_side <= 0) return result;

    const int64_t N = tokens.dim(0);   // grid_side * grid_side
    const int64_t C = tokens.dim(1);

    // Target token count: 7x7 = 49 tokens from a 28x28 grid (49x reduction
    // in sequence length from 784 raw patches; up to 4x vs any fixed grid).
    int64_t M = cfg_.target_tokens > 0
        ? std::min<int64_t>(cfg_.target_tokens, N)
        : std::max<int64_t>(1, static_cast<int64_t>(N * cfg_.keep_ratio));

    // ------------------------------------------------------------------
    // Importance scoring: L2 energy of each token feature vector
    // (higher energy = more visual information; matches UniCompress's
    //  attention-weighted selection approximated cheaply).
    // ------------------------------------------------------------------
    std::vector<float> energy(static_cast<size_t>(N), 0.0f);
    for (int64_t i = 0; i < N; ++i) {
        const float* row = tokens.f32() + i * C;
        double e = 0.0;
        for (int64_t c = 0; c < C; ++c) e += static_cast<double>(row[c]) * row[c];
        energy[static_cast<size_t>(i)] = static_cast<float>(std::sqrt(e));
    }

    // Keep top-M indices by energy (stable for determinism).
    std::vector<int64_t> order(static_cast<size_t>(N));
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](int64_t a, int64_t b) {
                         return energy[static_cast<size_t>(a)] >
                                energy[static_cast<size_t>(b)];
                     });
    order.resize(static_cast<size_t>(M));

    std::vector<char> is_kept(static_cast<size_t>(N), 0);
    for (const int64_t idx : order) is_kept[static_cast<size_t>(idx)] = 1;

    // ------------------------------------------------------------------
    // Merge discarded tokens into their nearest surviving neighbor
    // (grid distance), accumulating weighted averages.
    // ------------------------------------------------------------------
    Tensor merged("unicompressed", {M, C}, DType::F32);
    std::vector<float> wsum(static_cast<size_t>(M), 0.0f);
    std::vector<int64_t> kept_row_of(static_cast<size_t>(N), -1);
    for (int64_t m = 0; m < M; ++m) kept_row_of[static_cast<size_t>(order[static_cast<size_t>(m)])] = m;

    merged.zero();
    for (int64_t i = 0; i < N; ++i) {
        int64_t target = -1;
        if (is_kept[static_cast<size_t>(i)]) {
            target = kept_row_of[static_cast<size_t>(i)];
        } else {
            // nearest kept cell in grid space
            const int32_t ix = static_cast<int32_t>(i % grid_side);
            const int32_t iy = static_cast<int32_t>(i / grid_side);
            int32_t best_d = 1 << 30;
            for (int64_t m = 0; m < M; ++m) {
                const int64_t kidx = order[static_cast<size_t>(m)];
                const int32_t kx = static_cast<int32_t>(kidx % grid_side);
                const int32_t ky = static_cast<int32_t>(kidx / grid_side);
                const int32_t d = (kx - ix) * (kx - ix) + (ky - iy) * (ky - iy);
                if (d < best_d) { best_d = d; target = static_cast<int64_t>(m); }
            }
        }
        const float w = is_kept[static_cast<size_t>(i)]
            ? 1.0f
            : 0.35f;   // discarded tokens contribute less
        float* dst = merged.f32() + target * C;
        const float* src = tokens.f32() + i * C;
        for (int64_t c = 0; c < C; ++c) dst[c] += w * src[c];
        wsum[static_cast<size_t>(target)] += w;
    }

    for (int64_t m = 0; m < M; ++m) {
        const float inv = 1.0f / std::max(wsum[static_cast<size_t>(m)], 1e-6f);
        float* row = merged.f32() + m * C;
        for (int64_t c = 0; c < C; ++c) row[c] *= inv;
    }

    if (out_weights != nullptr) *out_weights = wsum;
    result.push_back(std::move(merged));
    return result;
}

} // namespace omniseed
