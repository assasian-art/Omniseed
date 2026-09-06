// =============================================================================
//  OmniSeed — mobilenet_v4.cpp
//  Distilled MobileNetV4-style backbone, int8-quantized depthwise separable
//  convolutions, projected to the LM embedding width.
//
//  Phase status: the encode() pipeline (resize -> patch features ->
//  UniCompress -> ternary projection) is live. The quantized backbone stack
//  activates once the converter emits vision.* tensors into the GGUF; until
//  then patch features are deterministic pixel-statistics embeddings so the
//  full pipeline is exercisable end-to-end.
// =============================================================================
#include "omniseed/vision/vision.h"
#include "omniseed/core/bitlinear.h"
#include "omniseed/core/gguf_loader.h"
#include "omniseed/core/platform.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace omniseed {

// ---------------------------------------------------------------------------
// Image resize (nearest neighbor)
// ---------------------------------------------------------------------------
Image Image::resize(const Image& src, int32_t w, int32_t h) {
    Image out;
    out.width = w;
    out.height = h;
    out.rgb.resize(static_cast<size_t>(w) * h * 3);
    if (!src.valid() || w <= 0 || h <= 0) return out;

    const float sx = static_cast<float>(src.width) / w;
    const float sy = static_cast<float>(src.height) / h;
    for (int32_t y = 0; y < h; ++y) {
        const int32_t syi =
            std::min(src.height - 1, static_cast<int32_t>(y * sy));
        for (int32_t x = 0; x < w; ++x) {
            const int32_t sxi =
                std::min(src.width - 1, static_cast<int32_t>(x * sx));
            for (int32_t c = 0; c < 3; ++c) {
                out.rgb[(static_cast<size_t>(y) * w + x) * 3 + c] =
                    src.rgb[(static_cast<size_t>(syi) * src.width + sxi) * 3 + c];
            }
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Quantized conv primitives (used by the backbone once weights are present)
// ---------------------------------------------------------------------------
namespace {

inline int8_t clamp_i8(int v) {
    return static_cast<int8_t>(std::max(-128, std::min(127, v)));
}

// 3x3 depthwise conv, stride 1, zero padding, per-channel requantization.
void depthwise3x3(const int8_t* src, int32_t side, const int8_t* w,
                  int32_t w_side, const float* scale, int32_t z,
                  int8_t* dst) {
    const int32_t half = w_side / 2;
    for (int32_t y = 0; y < side; ++y) {
        for (int32_t x = 0; x < side; ++x) {
            int32_t acc = 0;
            for (int32_t ky = 0; ky < w_side; ++ky) {
                const int32_t iy = y + ky - half;
                if (iy < 0 || iy >= side) continue;
                for (int32_t kx = 0; kx < w_side; ++kx) {
                    const int32_t ix = x + kx - half;
                    if (ix < 0 || ix >= side) continue;
                    acc += static_cast<int32_t>(src[iy * side + ix]) *
                           static_cast<int32_t>(w[ky * w_side + kx]);
                }
            }
            dst[y * side + x] = clamp_i8(
                static_cast<int32_t>(std::lround(acc * scale[z])));
        }
    }
}

// 1x1 pointwise conv across channels (int8 in, int8 out).
void pointwise1x1(const int8_t* src, int32_t side, int32_t cin,
                  const int8_t* w, int32_t cout, const float* scale,
                  int8_t* dst) {
    for (int32_t yx = 0; yx < side * side; ++yx) {
        for (int32_t o = 0; o < cout; ++o) {
            int32_t acc = 0;
            for (int32_t i = 0; i < cin; ++i) {
                acc += static_cast<int32_t>(src[yx * cin + i]) *
                       static_cast<int32_t>(w[o * cin + i]);
            }
            dst[yx * cout + o] =
                clamp_i8(static_cast<int32_t>(std::lround(acc * scale[o])));
        }
    }
}

// Deterministic placeholder patch embedding until converted weights arrive:
// per-patch RGB mean/std -> feature vector via fixed random-free hashing.
void patch_features(const Image& in, int32_t grid, int32_t C,
                    Tensor& tokens) {
    const int32_t pw = in.width / grid;
    const int32_t ph = in.height / grid;
    for (int32_t gy = 0; gy < grid; ++gy) {
        for (int32_t gx = 0; gx < grid; ++gx) {
            double mean[3] = {0, 0, 0};
            double var[3] = {0, 0, 0};
            int64_t n = 0;
            for (int32_t y = gy * ph; y < (gy + 1) * ph; ++y) {
                for (int32_t x = gx * pw; x < (gx + 1) * pw; ++x) {
                    for (int32_t c = 0; c < 3; ++c) {
                        const float v = static_cast<float>(
                            in.rgb[(static_cast<size_t>(y) * in.width + x) * 3 + c]);
                        mean[c] += v;
                        ++n;
                    }
                }
            }
            if (n == 0) n = 1;
            for (double& m : mean) m /= static_cast<double>(n);
            for (int32_t y = gy * ph; y < (gy + 1) * ph; ++y) {
                for (int32_t x = gx * pw; x < (gx + 1) * pw; ++x) {
                    for (int32_t c = 0; c < 3; ++c) {
                        const float v = static_cast<float>(
                            in.rgb[(static_cast<size_t>(y) * in.width + x) * 3 + c]);
                        const double d = v - mean[c];
                        var[c] += d * d;
                    }
                }
            }
            for (double& vv : var) vv /= static_cast<double>(n);

            // Feature row: [mean_rgb, std_rgb, periodic position codes...]
            float* out = tokens.f32() +
                (static_cast<int64_t>(gy) * grid + gx) * C;
            for (int32_t c = 0; c < C; ++c) {
                const int32_t sel = c % 6;
                float val = 0.0f;
                switch (sel) {
                    case 0: val = static_cast<float>(mean[0]) / 255.0f; break;
                    case 1: val = static_cast<float>(mean[1]) / 255.0f; break;
                    case 2: val = static_cast<float>(mean[2]) / 255.0f; break;
                    case 3: val = static_cast<float>(std::sqrt(var[0])) / 128.0f; break;
                    case 4: val = static_cast<float>(std::sqrt(var[1])) / 128.0f; break;
                    default: val = static_cast<float>(std::sqrt(var[2])) / 128.0f; break;
                }
                // positional modulation keeps tokens distinguishable
                const float pos = std::sin(static_cast<float>(gx + gy * grid) * 0.1f +
                                           static_cast<float>(c) * 0.01f);
                out[c] = val * 0.9f + pos * 0.1f;
            }
        }
    }
}

} // namespace

// ---------------------------------------------------------------------------
// GGUF loading
// ---------------------------------------------------------------------------
bool VisionEncoder::load(const std::string& gguf_path) {
    // Keep the loader (and its mmap) alive: proj_w_ below is a non-owning
    // view into the mapped file. A local GgufLoader would unmap on return
    // and leave proj_w_ dangling (same bug class as the Phase 7 loader fix).
    store_.close();
    if (!store_.open(gguf_path)) {
        error_ = store_.error();
        return false;
    }
    const GgufLoader& gg = store_;

    cfg_.input_size =
        static_cast<int32_t>(gg.get_u64("vision.input_size", 96));
    cfg_.grid_side =
        static_cast<int32_t>(gg.get_u64("vision.grid_side", 28));
    cfg_.feat_channels =
        static_cast<int32_t>(gg.get_u64("vision.feat_channels", 96));
    cfg_.out_dim =
        static_cast<int32_t>(gg.get_u64("vision.out_dim", 768));

    // Ternary projection head [out_dim, feat_channels].
    if (gg.has_tensor("vision.proj.weight")) {
        Tensor t = gg.tensor("vision.proj.weight");
        if (t.dtype() != DType::TERNARY) {
            error_ = "vision.proj.weight must be ternary";
            return false;
        }
        proj_w_ = Tensor("vision.proj.weight",
                         {t.dim(0), t.dim(1)}, DType::TERNARY,
                         const_cast<void*>(t.data()));
        // Debug: sweep the full packed region once at load time.
        if (std::getenv("OMNISEED_DBG")) {
            const uint8_t* p = proj_w_.packed();
            if (p == nullptr) {
                std::fprintf(stderr, "[sweep] proj NULL data n=%lld\n",
                             (long long)proj_w_.numel());
            } else {
                uint64_t s = 0;
                for (int64_t i = 0; i < static_cast<int64_t>(proj_w_.nbytes()); ++i) s += p[i];
                std::fprintf(stderr, "[sweep] proj n=%lld bytes=%lld sum=%llu\n",
                             (long long)proj_w_.numel(), (long long)proj_w_.nbytes(),
                             (unsigned long long)s);
            }
            std::fflush(stderr);
        }
        // Per-row scales (fp32 tensor) are the accurate form; the f64
        // metadata scalar is the legacy fallback.
        if (gg.has_tensor("vision.proj.scale")) {
            Tensor s = gg.tensor("vision.proj.scale");
            if (s.dtype() == DType::F32 && s.numel() == t.dim(0)) {
                proj_scales_ = Tensor("vision.proj.scale", s.shape(), DType::F32);
                std::memcpy(proj_scales_.data(), s.data(), s.nbytes());
            }
        }
        proj_scale_ = static_cast<float>(
            gg.get_f64("vision.proj.scale_mean",
                       gg.get_f64("vision.proj.scale", 1.0)));
        if (gg.has_tensor("vision.proj.bias")) {
            Tensor b = gg.tensor("vision.proj.bias");
            proj_bias_ = Tensor("vision.proj.bias", b.shape(), DType::F32);
            std::memcpy(proj_bias_.data(), b.data(), b.nbytes());
        }
    }
    // Backbone conv weights: loaded by the Phase 3 converter integration.
    valid_ = true;
    return true;
}

// ---------------------------------------------------------------------------
// Encode
// ---------------------------------------------------------------------------
bool VisionEncoder::encode(const Image& img, Tensor& out_tokens,
                           const UniCompress& compressor) const {
    if (!valid_) {
        error_ = "encoder not loaded";
        return false;
    }

    // 1) resize
    Image in = Image::resize(img, cfg_.input_size, cfg_.input_size);
    if (!in.valid()) {
        error_ = "invalid image";
        return false;
    }

    // 2) patch features [grid*grid, C]
    const int32_t G = cfg_.grid_side;
    const int32_t C = cfg_.feat_channels;
    Tensor tokens("vision_tokens",
                  {static_cast<int64_t>(G) * G, C}, DType::F32);
    patch_features(in, G, C, tokens);

    // 3) UniCompress token merging (up to 4x fewer tokens)
    std::vector<float> weights;
    const std::vector<Tensor> merged =
        compressor.compress(tokens, G, &weights);
    if (merged.empty()) {
        error_ = "unicompress returned no tokens";
        return false;
    }

    // 4) project to LM dim with ternary BitLinear (identity when unloaded)
    const Tensor& m = merged[0];
    const int64_t M = m.dim(0);
    out_tokens = Tensor("vision_out",
                        {M, static_cast<int64_t>(cfg_.out_dim)}, DType::F32);

    // NOTE: a default-constructed (absent) Tensor has numel()==1 with a null
    // data pointer, so "optional tensor present" must be tested with numel() > 1.
    if (proj_w_.numel() > 1) {
        for (int64_t i = 0; i < M; ++i) {
            if (proj_scales_.numel() == cfg_.out_dim) {
                bitnet::bitlinear_forward_rows(
                    proj_w_.packed(), proj_scales_.f32(),
                    m.f32() + i * cfg_.feat_channels,
                    out_tokens.f32() + i * cfg_.out_dim,
                    cfg_.out_dim, cfg_.feat_channels);
                if (proj_bias_.numel() > 1) {
                    float* dst = out_tokens.f32() + i * cfg_.out_dim;
                    for (int32_t o = 0; o < cfg_.out_dim; ++o)
                        dst[o] += proj_bias_.f32()[o];
                }
            } else {
                bitnet::bitlinear_forward(
                    proj_w_.packed(),
                    proj_bias_.numel() > 1 ? proj_bias_.f32() : nullptr,
                    proj_scale_,
                    m.f32() + i * cfg_.feat_channels,
                    out_tokens.f32() + i * cfg_.out_dim,
                    cfg_.out_dim, cfg_.feat_channels);
            }
        }
    } else {
        // Projection head not loaded: copy features into out_dim slice.
        for (int64_t i = 0; i < M; ++i) {
            float* dst = out_tokens.f32() + i * cfg_.out_dim;
            std::memset(dst, 0, cfg_.out_dim * sizeof(float));
            const int64_t copy_n = std::min<int64_t>(cfg_.feat_channels,
                                                     cfg_.out_dim);
            std::memcpy(dst, m.f32() + i * cfg_.feat_channels,
                        static_cast<size_t>(copy_n) * sizeof(float));
        }
    }
    return true;
}

} // namespace omniseed
