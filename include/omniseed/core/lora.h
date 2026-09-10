// =============================================================================
//  OmniSeed — lora.h
//  Assistant-behavior LoRA sidecar (Phase 14, 2B): classic low-rank adapters
//  over the model's big linears, applied at runtime as
//      y = W_ternary·x + scaling · B(A·x)
//  The ternary base weights stay mmap'd/read-only; the sidecar adds only
//  A [rank,in] + B [out,rank] f16 per targeted linear — ~2.7 MB fp16 for the
//  0.1B model at rank 8 (vs ~103 MB for the whole ternary GGUF).
//
//  Sidecar container (omniseed-lora, written by tools/lora_chat.py):
//      lora.rank / lora.alpha / lora.scaling / lora.layer_count /
//      lora.n_embd / lora.base_vocab       metadata
//      lora.{l}.att.{receptance,key,value,output}.a   F16 [rank, in]
//      lora.{l}.att.{receptance,key,value,output}.b   F16 [out, rank]
//      lora.{l}.ffn.{key,value}.a / .b                F16
//
//  Prompt format: none needed. The sidecar was trained on the exact
//  "User: ...\n\nAssistant:" template the C++ Tokenizer::encode_chat emits
//  for world models — enable it and reply-not-ramble comes for free.
// =============================================================================
#pragma once

#include "omniseed/core/gguf_loader.h"
#include "omniseed/core/tensor.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace omniseed {

// Which per-layer linear an adapter delta belongs to.
enum class LoraTarget {
    Receptance, Key, Value, Output, FfnKey, FfnValue
};

class LoraAdapter {
public:
    LoraAdapter() = default;

    // Loads the sidecar GGUF (stays mapped for zero-copy A/B views).
    bool load(const std::string& path);
    bool valid() const { return valid_; }
    const std::string& error() const { return error_; }

    // ---- geometry (validated against the base model by the caller) ---------
    int32_t rank()        const { return rank_; }
    float   scaling()     const { return scaling_; }
    int32_t layer_count() const { return layer_count_; }
    int32_t n_embd()      const { return n_embd_; }

    struct Factors {
        Tensor a;   // [rank, in_dim]  f16 view (empty when untargeted)
        Tensor b;   // [out_dim, rank] f16 view
    };

    // Adapter factor VIEWS for layer l / target (cheap non-owning Tensor
    // views into the mapped sidecar; valid while this LoraAdapter lives).
    // Untargeted pair (no .a/.b tensors in the file) returns empty tensors —
    // callers check numel() before applying.
    Factors factors(int32_t layer, LoraTarget target) const;

    // Hot-path entry: batch-converted fp32 copies of A [rank, in] and
    // B [out, rank] (half→float is exact, so the live math is bit-identical
    // to apply_lora while skipping ~660k half_to_float calls per token).
    // hot_entry() returns nullptr for an untargeted pair; callers must check.
    struct HotEntry {
        std::unique_ptr<float[]> a32;   // fp32 copy of A [rank, in_dim]
        std::unique_ptr<float[]> b32;   // fp32 copy of B [out_dim, rank]
        int64_t rank = 0;
        int64_t out  = 0;
    };
    const HotEntry* hot_entry(int32_t layer, LoraTarget target) const;

    // y[i] += scaling * sum_j B[i,j] * (A[j]·x)
    //   B [out, rank] f16, A [rank, in_dim] f16 — plain fp32 math, no SIMD
    //   needed (rank*2 reads per output vs the ternary base's in_dim).
    static void apply_lora(const Tensor& B, const Tensor& A, float scaling,
                           const float* x, float* y,
                           int64_t out_dim, int64_t in_dim);

private:
    GgufLoader store_;
    int32_t rank_        = 0;
    int32_t layer_count_ = 0;
    int32_t n_embd_      = 0;
    float   scaling_     = 1.0f;
    bool    valid_       = false;
    std::string error_;

    // Hot-path cache: [layer][target] → converted fp32 factors; built once
    // after a successful load (untargeted pairs store rank == 0).
    std::vector<std::array<HotEntry, 6>> hot_;

    void build_hot_cache();
};

} // namespace omniseed
