// =============================================================================
//  OmniSeed — vision.h
//  Distilled MobileNetV4 vision encoder + UniCompress token reduction.
//
//  Pipeline (blueprint): image [96x96x3 RGB] -> MobileNetV4 backbone ->
//  feature grid [G x G x C] -> UniCompress (merge to ~1/4 tokens) ->
//  projection to n_embd -> visual token sequence for the TokenBus.
//
//  The backbone is int8/ternary-quantized and weights come from a GGUF
//  section ("vision.*" tensors) mapped zero-copy like the core model.
//  Peak RAM for the vision path must stay under ~30 MB (blueprint budget).
// =============================================================================
#pragma once
#include "omniseed/core/tensor.h"
#include "omniseed/core/gguf_loader.h"
#include <string>
#include <vector>

namespace omniseed {

// ---------------------------------------------------------------------------
// Image input: tightly-packed RGB, row-major, 3 channels.
// ---------------------------------------------------------------------------
struct Image {
    int32_t width  = 0;
    int32_t height = 0;
    std::vector<uint8_t> rgb;   // size = w*h*3

    bool valid() const {
        return width > 0 && height > 0 &&
               rgb.size() == static_cast<size_t>(width) * height * 3;
    }

    // Nearest-neighbor resize (keeps the runtime dependency-free).
    static Image resize(const Image& src, int32_t w, int32_t h);
};

// ---------------------------------------------------------------------------
// UniCompress — visual token merging (up to 4x reduction).
//   Strategy: importance-score each patch token by its feature energy,
//   keep the top-k grid cells, then average-pool the discarded neighbors
//   into the nearest survivor. Preserves ~90% of baseline accuracy at 4x
//   compression per the research notes.
// ---------------------------------------------------------------------------
struct UniCompressConfig {
    int32_t target_tokens = 49;   // 7x7 grid after compression (from 28x28)
    float   keep_ratio    = 0.25f; // fallback when target_tokens == 0
};

class UniCompress {
public:
    explicit UniCompress(const UniCompressConfig& cfg = {}) : cfg_(cfg) {}

    // tokens: [N, C] feature rows; grid_side = sqrt(N). Returns [M <= N, C]
    // where M is the compressed token count, plus the merged token weights.
    std::vector<Tensor> compress(const Tensor& tokens, int32_t grid_side,
                                 std::vector<float>* out_weights) const;

private:
    UniCompressConfig cfg_;
};

// ---------------------------------------------------------------------------
// MobileNetV4 distilled backbone
// ---------------------------------------------------------------------------
struct VisionConfig {
    int32_t input_size    = 96;    // 96x96 RGB input (blueprint spec)
    int32_t grid_side     = 28;    // feature grid before compression
    int32_t feat_channels = 96;    // backbone output channels
    int32_t out_dim       = 768;   // projection to LM embedding width
};

class VisionEncoder {
public:
    // Loads "vision.*" tensors from the model GGUF.
    bool load(const std::string& gguf_path);

    bool valid() const { return valid_; }
    const VisionConfig& config() const { return cfg_; }
    const std::string& error() const { return error_; }

    // Image -> projected visual embeddings [M, out_dim] (M = post-compress).
    // M and out_dim make this directly concatenable on the TokenBus.
    bool encode(const Image& img, Tensor& out_tokens,
                const UniCompress& compressor) const;

private:
    // Keeps the GGUF mmap alive: proj_w_ is a non-owning view into it.
    GgufLoader store_;
    VisionConfig cfg_;
    bool valid_ = false;
    mutable std::string error_;   // set even from const encode()

    // Quantized backbone weights (owned copies; small enough at this size).
    std::vector<Tensor> conv_weights_;
    std::vector<Tensor> conv_scales_;
    Tensor proj_w_;     // ternary [out_dim, feat_channels]
    Tensor proj_scales_;// fp32 [out_dim] per-row (preferred)
    float proj_scale_ = 1.0f;
    Tensor proj_bias_;  // fp32 [out_dim]
};

} // namespace omniseed
