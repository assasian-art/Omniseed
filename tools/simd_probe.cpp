#include "omniseed/core/platform.h"
#include <cstdio>

// from bitlinear.cpp / bitlinear_avx2.cpp (bitnet scope)
namespace omniseed { namespace bitnet {
void fwd_i8_scalar(const int8_t*, const float*, const float*, float*, int64_t, int64_t);
void fwd_i8_avx2(const int8_t*, const float*, const float*, float*, int64_t, int64_t);
void fwd_i8_sse41(const int8_t*, const float*, const float*, float*, int64_t, int64_t);
} }

using namespace omniseed;

int main() {
    const platform::CpuFeatures f = platform::cpu_features();
    std::printf("sse41=%d avx2=%d fma=%d\n", f.sse41, f.avx2, f.fma);

    // correctness probe: one row of 768 pseudo-random i8 weights vs fp64 ref
    constexpr int64_t D = 768;
    static int8_t w[D];
    static float x[D];
    uint32_t s = 7;
    for (int64_t i = 0; i < D; ++i) {
        s = s * 1664525u + 1013904223u;
        w[i] = static_cast<int8_t>((s >> 24) % 255 - 127);
        s = s * 1664525u + 1013904223u;
        x[i] = static_cast<float>((s >> 24) % 1000) / 500.0f - 1.0f;
    }
    const float scale = 0.0008f;
    double ref = 0.0;
    for (int64_t i = 0; i < D; ++i) ref += static_cast<double>(w[i]) * x[i];
    ref *= scale;

    float out[3];
    auto row = [&](void (*fn)(const int8_t*, const float*, const float*, float*, int64_t, int64_t)) {
        // each fn computes a full matrix; use out_dim=1, in_dim=D
        fn(w, &scale, x, out, 1, D);
        return out[0];
    };
    std::printf("ref    = %.9g\n", ref);
    std::printf("scalar = %.9g (%s)\n", row(bitnet::fwd_i8_scalar),
                row(bitnet::fwd_i8_scalar) - ref < 1e-3 ? "OK" : "BAD");
#if defined(OMNISEED_X86)
    std::printf("avx2   = %.9g (%s)\n", row(bitnet::fwd_i8_avx2),
                row(bitnet::fwd_i8_avx2) - ref < 1e-3 ? "OK" : "BAD");
    std::printf("sse41  = %.9g (%s)\n", row(bitnet::fwd_i8_sse41),
                row(bitnet::fwd_i8_sse41) - ref < 1e-3 ? "OK" : "BAD");
#endif
    return 0;
}
