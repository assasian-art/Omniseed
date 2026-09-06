#include "omniseed/core/platform.h"
#include <cstdio>
#include <vector>

namespace omniseed {
namespace bitnet {
void fwd_i8_scalar(const int8_t*, const float*, const float*, float*, int64_t, int64_t);
void fwd_i8_avx2(const int8_t*, const float*, const float*, float*, int64_t, int64_t);
void bitlinear_forward_i8(const int8_t*, const float*, const float*, float*, int64_t, int64_t);
using I8ForwardFn = void (*)(const int8_t*, const float*, const float*, float*, int64_t, int64_t);
I8ForwardFn simd_selected_fn();
} }

using namespace omniseed;

int main() {
    std::printf("avx2=%d\n", platform::cpu_features().avx2);
    const int64_t E = 768, I = 3072;
    std::vector<int8_t> Wkey(static_cast<size_t>(E * E), 1);
    std::vector<int8_t> Wffn(static_cast<size_t>(I * E), 2);
    std::vector<float> xs(E), scales1(E, 0.001f), scales2(I, 0.001f);
    for (int64_t i = 0; i < E; ++i) xs[i] = 0.001f * static_cast<float>(i % 97);

    auto bench = [&](const char* name,
                    void (*fn)(const int8_t*, const float*, const float*,
                               float*, int64_t, int64_t),
                    const std::vector<int8_t>& W, int64_t rows, int64_t cols,
                    const std::vector<float>& sc, int iters) {
        std::vector<float> y(static_cast<size_t>(rows));
        fn(W.data(), sc.data(), xs.data(), y.data(), rows, cols);
        const double t0 = platform::now_ms();
        for (int it = 0; it < iters; ++it)
            fn(W.data(), sc.data(), xs.data(), y.data(), rows, cols);
        const double ms = platform::now_ms() - t0;
        const double gmac = double(rows) * cols * iters / 1e9;
        std::printf("%-12s %8.3f ms  %6.2f GMAC/s  y0=%.4g\n", name, ms / iters,
                    gmac / (ms / 1000.0), y[0]);
    };

    for (int it = 0; it < 3; ++it) {
        bench("scalar att", bitnet::fwd_i8_scalar, Wkey, E, E, scales1, 50);
        bench("avx2   att", bitnet::fwd_i8_avx2, Wkey, E, E, scales1, 50);
        std::printf("---\n");
    }
    bench("scalar ffn", bitnet::fwd_i8_scalar, Wffn, I, E, scales2, 20);
    bench("avx2   ffn", bitnet::fwd_i8_avx2, Wffn, I, E, scales2, 20);

    // through the public dispatch path (what the model actually uses)
    bench("PUBLIC  ffn", bitnet::bitlinear_forward_i8, Wffn, I, E, scales2, 20);
    bench("PUBLIC  att", bitnet::bitlinear_forward_i8, Wkey, E, E, scales1, 50);
    std::printf("selected == avx2: %s\n",
                bitnet::simd_selected_fn() == &bitnet::fwd_i8_avx2 ? "YES" : "NO");
    return 0;
}
