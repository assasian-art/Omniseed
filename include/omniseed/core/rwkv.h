// =============================================================================
//  OmniSeed — rwkv.h
//  RWKV-7 "Goose" recurrent cell — exact reference math, BitNet ternary
//  projections, fp32 streaming state.
//
//  The recurrence below is transcribed 1:1 from the verified RWKV-7 numpy
//  reference (RWKV-LM/RWKV-v7/rwkv_v7_numpy.py), which the RWKV team shows
//  deviates ~0 from the official rwkv package:
//
//    Token shift (6 streams):  x_s = x + m_s * (last_x - x)      s in r,w,k,v,a,g
//    r = Wr @ xr                                                 (raw, no sigmoid)
//    w = exp(-sigmoid(tanh(xw @ Ww1) @ Ww2 + w_bias) / sqrt(e))  decay in (0,1)
//    k = Wk @ xk
//    v = Wv @ xv  (+ value residual:  v += (v0 - v) * sigmoid(xv@Wv1@Wv2+v_bias))
//    a = sigmoid(xa @ Wa1 @ Wa2 + a_bias)                        in-context lr (0,1)
//    g = sigmoid(xg @ Wg1) @ Wg2                                 gate
//
//    kk = k * k_k ;  k += k * (a - 1) * k_a
//    kk = l2_normalize_per_head(kk)
//
//    S = S * w^T  -  (S @ kk) (a*k)^T  +  v k^T        <-- generalized delta rule
//    y  = group_norm(S @ r)
//    y += (sum(r*k*r_k) * v)                           <-- extra residue
//    out = Wo @ (y * g)
//
//    Channel mix:  k = Wk @ (x + m*(last_x - x));  v = Wv @ relu(k)^2
//
//  State S is [n_heads, head_size, head_size] fp32 per layer — CONSTANT in
//  size regardless of context length. This is the O(1) memory property the
//  whole <300MB design stands on. Long context costs NOTHING extra per token.
//
//  OmniSeed deltas vs the reference: Wr/Wk/Wv/Wo/FFN matrices and the four
//  low-rank MLPs are BitNet b1.58 ternary (2 weights/byte) with fp32 scales;
//  all biases, mixes, k_k/k_a/r_k and norms stay fp32.
// =============================================================================
#pragma once

#include "omniseed/core/bitlinear.h"
#include "omniseed/core/tensor.h"

#include <string>
#include <vector>

namespace omniseed {

class GgufLoader;
class Tokenizer;

// ---------------------------------------------------------------------------
// Model hyperparameters (GGUF metadata)
// ---------------------------------------------------------------------------
struct RwkvConfig {
    int32_t n_layers      = 0;
    int32_t n_embd        = 0;
    int32_t n_vocab       = 0;
    int32_t n_heads       = 0;
    int32_t head_size     = 64;
    int32_t mlp_rank      = 64;   // fallback rank for the four low-rank MLPs
    int32_t ffn_inter     = 0;    // channel-mix hidden size (0 = n_embd)
    // Per-gate LoRA ranks (real checkpoints use w=64 a=64 g=128 v=32).
    int32_t rank_w        = 64;   // decay LoRA
    int32_t rank_a        = 64;   // in-context-learning rate LoRA
    int32_t rank_g        = 128;  // gate LoRA
    int32_t rank_v        = 32;   // value-residual LoRA
    bool    version_known = false;
};

// ---------------------------------------------------------------------------
// One block's weights. Ternary matrices are zero-copy views into the mmap'd
// GGUF; scales/biases/mixes are small owned fp32 tensors loaded from file.
// ---------------------------------------------------------------------------
struct RwkvLayerWeights {
    // Input layernorms
    Tensor ln1_w, ln1_b;                     // fp32 [E]  (before attention)
    Tensor ln2_w, ln2_b;                     // fp32 [E]  (before channel mix)

    // Token-shift mix vectors (post-LN stream), fp32 [E]
    Tensor tmix_r, tmix_w, tmix_k, tmix_v, tmix_a, tmix_g;

    // Main projections: ternary [E, E] with fp32 scalar scales
    Tensor W_r;  float r_scale  = 1.0f;      // receptance
    Tensor W_k;  float k_scale  = 1.0f;      // key
    Tensor W_v;  float v_scale  = 1.0f;      // value
    Tensor W_o;  float o_scale  = 1.0f;      // output

    // Low-rank MLPs: W_w1 [rank_w,E], W_w2 [E,rank_w], etc. (ternary + scale).
    // Shapes come from GGUF metadata; real world checkpoints use
    // w=64 a=64 g=128 v=32 and an FFN hidden size of 4*E.
    Tensor W_w1; float w1_scale = 1.0f;
    Tensor W_w2; float w2_scale = 1.0f;
    Tensor w_bias;                           // fp32 [E]

    Tensor W_a1; float a1_scale = 1.0f;
    Tensor W_a2; float a2_scale = 1.0f;
    Tensor a_bias;                           // fp32 [E]

    Tensor W_g1; float g1_scale = 1.0f;
    Tensor W_g2; float g2_scale = 1.0f;

    Tensor W_v1; float v1_scale = 1.0f;      // value-residual MLP
    Tensor W_v2; float v2_scale = 1.0f;
    Tensor v_bias;                           // fp32 [E]

    // Per-channel extra parameters, fp32 [E]
    Tensor k_k, k_a, r_k;

    // GroupNorm over head_size (applied to S@r), fp32 [E]
    Tensor gn_w, gn_b;

    // Channel mixing (RWKV-7: no receptance; hidden = ffn_inter, usually 4*E)
    Tensor F_mix;                float f_mix  = 0.0f;   // unused; kept for shape sym
    Tensor F_mix_v;                                           // fp32 [E]
    Tensor F_key;   float fk_scale = 1.0f;                // ternary [ffn_inter, E]
    Tensor F_value; float fv_scale = 1.0f;                // ternary [E, ffn_inter]
};

// ---------------------------------------------------------------------------
// Streaming recurrent state — the O(1) context memory
// ---------------------------------------------------------------------------
struct RwkvState {
    // token-shift caches: post-layernorm x fed to attention / channel mix
    std::vector<Tensor> tmix_state;          // per layer [E]
    std::vector<Tensor> ffn_state;           // per layer [E]
    // delta-rule state per layer [n_heads, head_size, head_size]
    std::vector<Tensor> attn_state;
    int64_t tokens_seen = 0;
};

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------
class RwkvModel {
public:
    RwkvModel() = default;

    // Loads an OmniSeed-quantized RWKV-7 GGUF (ternary weights, fp32 norms).
    bool load(const std::string& path);

    bool valid() const { return valid_; }
    const RwkvConfig& config() const { return cfg_; }
    const std::string& error() const { return error_; }

    // Fresh zero state — call before a new conversation/document.
    void init_state(RwkvState& st) const;

    // Deep copy (used by memory snapshotting / tree search).
    static void copy_state(const RwkvState& src, RwkvState& dst);

    // Single token step: updates st in place, writes logits [n_vocab] fp32.
    void forward(int32_t token, RwkvState& st, Tensor& logits) const;

    int32_t greedy_pick(const Tensor& logits) const;

private:
    bool load_meta(const GgufLoader& gg);
    bool load_weights(const GgufLoader& gg);

    // helpers used by forward()
    void project(const Tensor& W, float scale, const float* x, float* y,
                 int64_t out_dim, int64_t in_dim) const;

    RwkvConfig cfg_;
    std::vector<RwkvLayerWeights> layers_;
    Tensor emb_;                 // fp16 [n_vocab, n_embd]
    Tensor ln0_w_, ln0_b_;       // pre-block-0 layernorm
    Tensor ln_out_w_, ln_out_b_;
    Tensor head_;                // fp16 or ternary [n_vocab, n_embd]
    float  head_scale_ = 1.0f;
    bool   head_ternary_ = false;

    bool valid_ = false;
    std::string error_;
};

} // namespace omniseed
