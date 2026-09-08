// =============================================================================
//  OmniSeed — rwkv.cpp
//  RWKV-7 forward pass, exact reference math (see rwkv.h for the equations),
//  with BitNet ternary BitLinear projections.
//
//  Transcribed from RWKV-LM/RWKV-v7/rwkv_v7_numpy.py (verified against the
//  official rwkv package by the RWKV team). Key details that MUST match:
//    * decay w applies along the KEY axis (columns of S)
//    * kk is L2-normalized per head, computed from k BEFORE the k_a blend
//    * the gate g applies sigmoid BETWEEN the two low-rank projections
//    * the extra residue is a per-head SCALAR sum(r*k*r_k) times v
//    * v0 (value residual anchor) is per-token, captured at block 0
// =============================================================================
#include "omniseed/core/rwkv.h"
#include "omniseed/core/gguf_loader.h"
#include "omniseed/core/platform.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace omniseed {

// ===========================================================================
// Helpers
// ===========================================================================
namespace {

inline bool dbg_nan() {
    static const bool on = std::getenv("OMNISEED_DEBUG_NAN") != nullptr;
    return on;
}
inline bool dbg_prof() {
    static const bool on = std::getenv("OMNISEED_PROFILE") != nullptr;
    return on;
}
// Accumulated per-section timings (ms) across calls.
struct ProfSections {
    double proj = 0, gates = 0, srec = 0, readout = 0, ffn = 0, head = 0, emb = 0;
    int    calls = 0;
    void dump() {
        if (calls == 0) return;
        std::fprintf(stderr,
                     "[prof] calls=%d proj=%.1fms gates=%.1f srec=%.1f "
                     "readout=%.1f ffn=%.1f head=%.1f emb=%.1f  "
                     "(per call: proj=%.2f gates=%.2f srec=%.2f readout=%.2f "
                     "ffn=%.2f head=%.2f emb=%.2f)\n",
                     calls, proj, gates, srec, readout, ffn, head, emb,
                     proj / calls, gates / calls, srec / calls,
                     readout / calls, ffn / calls, head / calls, emb / calls);
    }
};
ProfSections& prof() {
    static ProfSections p;
    return p;
}
inline void dbg_check(const char* tag, const float* p, int64_t n) {
    if (!dbg_nan()) return;
    for (int64_t i = 0; i < n; ++i) {
        if (!std::isfinite(p[i])) {
            std::fprintf(stderr, "[nan] %s at i=%lld val=%g\n", tag,
                         static_cast<long long>(i), p[i]);
            return;
        }
    }
}

inline float lerp(float a, float b, float t) { return a + t * (b - a); }

inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

inline float squared_relu(float x) {
    const float r = x > 0.0f ? x : 0.0f;
    return r * r;
}

// GroupNorm over head_size groups (eps 64e-5 per the reference).
void group_norm_heads(const float* y, int32_t n_heads, int32_t head_size,
                      const Tensor& gn_w, const Tensor& gn_b, float* out) {
    for (int32_t h = 0; h < n_heads; ++h) {
        const float* yh = y + static_cast<size_t>(h) * head_size;
        double mean = 0.0;
        for (int32_t i = 0; i < head_size; ++i) mean += yh[i];
        mean /= head_size;

        double var = 0.0;
        for (int32_t i = 0; i < head_size; ++i) {
            const double d = yh[i] - mean;
            var += d * d;
        }
        var /= head_size;

        const float inv = static_cast<float>(1.0 / std::sqrt(var + 64e-5));
        for (int32_t i = 0; i < head_size; ++i) {
            out[h * head_size + i] =
                static_cast<float>((yh[i] - mean) * inv) *
                    gn_w.f32()[h * head_size + i] +
                gn_b.f32()[h * head_size + i];
        }
    }
}

} // namespace

// ===========================================================================
// Loading
// ===========================================================================
bool RwkvModel::load_meta(const GgufLoader& gg) {
    cfg_.n_layers = static_cast<int32_t>(gg.n_layers());
    cfg_.n_embd   = static_cast<int32_t>(gg.n_embd());
    cfg_.n_vocab  = static_cast<int32_t>(gg.n_vocab());

    if (cfg_.n_layers <= 0 || cfg_.n_embd <= 0 || cfg_.n_vocab <= 0) {
        error_ = "GGUF metadata missing omniseed.{layer_count,embedding_length,"
                 "vocab_size}";
        return false;
    }

    cfg_.head_size = static_cast<int32_t>(gg.get_u64("omniseed.key_length", 64));
    if (cfg_.head_size <= 0 || cfg_.n_embd % cfg_.head_size != 0) {
        error_ = "n_embd not divisible by key_length";
        return false;
    }
    cfg_.n_heads  = cfg_.n_embd / cfg_.head_size;
    cfg_.mlp_rank = static_cast<int32_t>(gg.get_u64("omniseed.mlp_rank", 64));
    if (cfg_.mlp_rank <= 0) cfg_.mlp_rank = 64;

    // Real world checkpoints: FFN hidden 4*E, per-gate LoRA ranks.
    cfg_.ffn_inter = static_cast<int32_t>(gg.get_u64("omniseed.ffn_intermediate", 0));
    if (cfg_.ffn_inter <= 0) cfg_.ffn_inter = 4 * cfg_.n_embd;
    cfg_.rank_w = static_cast<int32_t>(gg.get_u64("omniseed.rank_decay",    cfg_.mlp_rank));
    cfg_.rank_a = static_cast<int32_t>(gg.get_u64("omniseed.rank_a",        cfg_.mlp_rank));
    cfg_.rank_g = static_cast<int32_t>(gg.get_u64("omniseed.rank_gate",     cfg_.mlp_rank));
    cfg_.rank_v = static_cast<int32_t>(gg.get_u64("omniseed.rank_value",    cfg_.mlp_rank));
    if (cfg_.rank_w <= 0 || cfg_.rank_a <= 0 || cfg_.rank_g <= 0 ||
        cfg_.rank_v <= 0 || cfg_.ffn_inter <= 0) {
        error_ = "invalid rank/ffn metadata";
        return false;
    }
    cfg_.version_known = true;
    return true;
}

namespace {

// Small owned fp32 tensor, copied out of the mmap (they're tiny).
Tensor load_f32(const GgufLoader& gg, const std::string& name,
                std::string& err, bool required = true) {
    if (!gg.has_tensor(name)) {
        if (required) err = "missing tensor: " + name;
        return Tensor();
    }
    Tensor t = gg.tensor(name);
    if (t.dtype() != DType::F32) {
        err = "tensor " + name + ": expected fp32, got " + dtype_name(t.dtype());
        return Tensor();
    }
    Tensor out(name, t.shape(), DType::F32);
    std::memcpy(out.data(), t.data(), t.nbytes());
    return out;
}

// Zero-copy ternary OR int8 view straight into the mapped GGUF region.
// (Converted checkpoints default to per-row int8 for the big matrices:
// PTQ ternary b1.58 loses coherence without QAT — verified in tools/quant_sim.py.)
Tensor load_ternary_view(const GgufLoader& gg, const std::string& name,
                         const std::vector<int64_t>& shape2d,
                         std::string& err, bool required = true) {
    if (!gg.has_tensor(name)) {
        if (required) err = "missing tensor: " + name;
        return Tensor();
    }
    Tensor t = gg.tensor(name);
    if (t.dtype() != DType::TERNARY && t.dtype() != DType::I8) {
        err = "tensor " + name + ": expected ternary/int8, got " +
              dtype_name(t.dtype());
        return Tensor();
    }
    return Tensor(name, shape2d, t.dtype(), const_cast<void*>(t.data()));
}

} // namespace

bool RwkvModel::load_weights(const GgufLoader& gg) {
    const int32_t L = cfg_.n_layers;
    const int32_t E = cfg_.n_embd;
    layers_.resize(static_cast<size_t>(L));

    // ------------------------------ embedding -------------------------------
    if (!gg.has_tensor("token.embd")) {
        error_ = "missing tensor: token.embd";
        return false;
    }
    {
        Tensor t = gg.tensor("token.embd");
        if (t.dtype() != DType::F16) {
            error_ = "token.embd must be fp16";
            return false;
        }
        emb_ = Tensor("token.embd", {t.dim(0), t.dim(1)}, DType::F16,
                      const_cast<void*>(t.data()));
    }

    ln0_w_ = load_f32(gg, "blocks.0.ln0.weight", error_, false);
    ln0_b_ = load_f32(gg, "blocks.0.ln0.bias", error_, false);
    ln_out_w_ = load_f32(gg, "ln_out.weight", error_);
    ln_out_b_ = load_f32(gg, "ln_out.bias", error_);
    if (!error_.empty()) return false;

    if (dbg_nan()) {
        const GgufLoader& gg2 = gg;
        std::fprintf(stderr, "[dbg] data_start=%llu\n",
                     static_cast<unsigned long long>(gg2.debug_data_start()));
        const Tensor sc = gg.tensor("blocks.0.att.receptance.scale");
        std::fprintf(stderr, "[dbg] sc_r numel=%d dtype=%s\n",
                     static_cast<int>(sc.numel()), dtype_name(sc.dtype()));
        if (sc.numel() >= 5)
            std::fprintf(stderr, "[dbg] sc_r first5: %.6g %.6g %.6g %.6g %.6g\n",
                         sc.f32()[0], sc.f32()[1], sc.f32()[2], sc.f32()[3],
                         sc.f32()[4]);
        const Tensor kk = gg.tensor("blocks.0.att.k_k");
        if (kk.numel() >= 3)
            std::fprintf(stderr, "[dbg] k_k first3: %.6g %.6g %.6g\n",
                         kk.f32()[0], kk.f32()[1], kk.f32()[2]);
    }

    // Output head: ternary (preferred) or fp16 fallback.
    if (!gg.has_tensor("head.weight")) {
        error_ = "missing tensor: head.weight";
        return false;
    }
    {
        Tensor t = gg.tensor("head.weight");
        if (t.dtype() == DType::TERNARY) {
            head_ternary_ = true;
            head_ = Tensor("head.weight", {t.dim(0), t.dim(1)}, DType::TERNARY,
                           const_cast<void*>(t.data()));
            head_scales_ = load_f32(gg, "head.scale", error_, false);
            Tensor s = load_f32(gg, "head.scale0", error_, false);   // legacy
            head_scale_ = s.numel() > 1 ? s.f32()[0] : 1.0f;
        } else if (t.dtype() == DType::I8) {
            // Preferred layout: per-row int8 + fp32 scales straight from the
            // GGUF — rides the AVX2/SSE i8 kernel, halves head memory.
            head_ = Tensor("head.weight", {t.dim(0), t.dim(1)}, DType::I8,
                           const_cast<void*>(t.data()));
            head_scales_ = load_f32(gg, "head.scale", error_, false);
            if (!error_.empty()) return false;
        } else if (t.dtype() == DType::F16) {
            // Convert fp16 head -> per-row int8 at load time. The head is the
            // single biggest matmul (V*E); int8 rides the AVX2/SSE kernel and
            // halves its memory (50 MB vs 100 MB for the 0.1B model).
            const int64_t rows = t.dim(0), cols = t.dim(1);
            Tensor q("head.weight", {rows, cols}, DType::I8);
            std::vector<float> scales(static_cast<size_t>(rows));
            const uint16_t* src = t.f16();
            for (int64_t r = 0; r < rows; ++r) {
                const uint16_t* row = src + r * cols;
                float mx = 0.0f;
                for (int64_t c = 0; c < cols; ++c)
                    mx = std::max(mx, std::fabs(half_to_float(row[c])));
                const float s = (mx > 0.0f) ? mx / 127.0f : 1e-12f;
                scales[static_cast<size_t>(r)] = s;
                int8_t* dst = q.i8() + r * cols;
                for (int64_t c = 0; c < cols; ++c)
                    dst[c] = static_cast<int8_t>(std::lround(
                        half_to_float(row[c]) / s));
            }
            head_ = std::move(q);
            head_scales_ = Tensor("head.scale", {rows}, DType::F32);
            std::memcpy(head_scales_.f32(), scales.data(),
                        static_cast<size_t>(rows) * sizeof(float));
        } else {
            error_ = "head.weight: unsupported dtype";
            return false;
        }
    }

    // -------------------------------- blocks ---------------------------------
    for (int32_t l = 0; l < L; ++l) {
        const std::string p = "blocks." + std::to_string(l) + ".";
        RwkvLayerWeights& w = layers_[static_cast<size_t>(l)];

        w.ln1_w = load_f32(gg, p + "ln1.weight", error_);
        w.ln1_b = load_f32(gg, p + "ln1.bias", error_);
        w.ln2_w = load_f32(gg, p + "ln2.weight", error_);
        w.ln2_b = load_f32(gg, p + "ln2.bias", error_);
        if (!error_.empty()) return false;

        w.tmix_r = load_f32(gg, p + "att.tmix_r", error_);
        w.tmix_w = load_f32(gg, p + "att.tmix_w", error_);
        w.tmix_k = load_f32(gg, p + "att.tmix_k", error_);
        w.tmix_v = load_f32(gg, p + "att.tmix_v", error_);
        w.tmix_a = load_f32(gg, p + "att.tmix_a", error_);
        w.tmix_g = load_f32(gg, p + "att.tmix_g", error_);
        if (!error_.empty()) return false;

        // Main ternary projections [E, E] + per-row scale tensors.
        const std::vector<int64_t> ee = {E, E};
        w.W_r = load_ternary_view(gg, p + "att.receptance.weight", ee, error_);
        w.W_k = load_ternary_view(gg, p + "att.key.weight", ee, error_);
        w.W_v = load_ternary_view(gg, p + "att.value.weight", ee, error_);
        w.W_o = load_ternary_view(gg, p + "att.output.weight", ee, error_);
        if (!error_.empty()) return false;
        w.sc_r = load_f32(gg, p + "att.receptance.scale", error_, false);
        w.sc_k = load_f32(gg, p + "att.key.scale", error_, false);
        w.sc_v = load_f32(gg, p + "att.value.scale", error_, false);
        w.sc_o = load_f32(gg, p + "att.output.scale", error_, false);
        if (!error_.empty()) return false;
        w.r_scale = static_cast<float>(gg.get_f64(p + "att.receptance.scale", 1.0));
        w.k_scale = static_cast<float>(gg.get_f64(p + "att.key.scale", 1.0));
        w.v_scale = static_cast<float>(gg.get_f64(p + "att.value.scale", 1.0));
        w.o_scale = static_cast<float>(gg.get_f64(p + "att.output.scale", 1.0));

        // Low-rank MLPs: per-gate ranks (w=rank_w, a=rank_a, g=rank_g, v=rank_v).
        const std::vector<int64_t> rw1 = {cfg_.rank_w, E};
        const std::vector<int64_t> rw2 = {E, cfg_.rank_w};
        const std::vector<int64_t> ra1 = {cfg_.rank_a, E};
        const std::vector<int64_t> ra2 = {E, cfg_.rank_a};
        const std::vector<int64_t> rg1 = {cfg_.rank_g, E};
        const std::vector<int64_t> rg2 = {E, cfg_.rank_g};
        const std::vector<int64_t> rv1 = {cfg_.rank_v, E};
        const std::vector<int64_t> rv2 = {E, cfg_.rank_v};
        w.W_w1 = load_ternary_view(gg, p + "att.w1.weight", rw1, error_);
        w.W_w2 = load_ternary_view(gg, p + "att.w2.weight", rw2, error_);
        w.W_a1 = load_ternary_view(gg, p + "att.a1.weight", ra1, error_);
        w.W_a2 = load_ternary_view(gg, p + "att.a2.weight", ra2, error_);
        w.W_g1 = load_ternary_view(gg, p + "att.g1.weight", rg1, error_);
        w.W_g2 = load_ternary_view(gg, p + "att.g2.weight", rg2, error_);
        w.W_v1 = load_ternary_view(gg, p + "att.v1.weight", rv1, error_);
        w.W_v2 = load_ternary_view(gg, p + "att.v2.weight", rv2, error_);
        if (!error_.empty()) return false;
        w.sc_w1 = load_f32(gg, p + "att.w1.scale", error_, false);
        w.sc_w2 = load_f32(gg, p + "att.w2.scale", error_, false);
        w.sc_a1 = load_f32(gg, p + "att.a1.scale", error_, false);
        w.sc_a2 = load_f32(gg, p + "att.a2.scale", error_, false);
        w.sc_g1 = load_f32(gg, p + "att.g1.scale", error_, false);
        w.sc_g2 = load_f32(gg, p + "att.g2.scale", error_, false);
        w.sc_v1 = load_f32(gg, p + "att.v1.scale", error_, false);
        w.sc_v2 = load_f32(gg, p + "att.v2.scale", error_, false);
        if (!error_.empty()) return false;
        w.w1_scale = static_cast<float>(gg.get_f64(p + "att.w1.scale", 1.0));
        w.w2_scale = static_cast<float>(gg.get_f64(p + "att.w2.scale", 1.0));
        w.a1_scale = static_cast<float>(gg.get_f64(p + "att.a1.scale", 1.0));
        w.a2_scale = static_cast<float>(gg.get_f64(p + "att.a2.scale", 1.0));
        w.g1_scale = static_cast<float>(gg.get_f64(p + "att.g1.scale", 1.0));
        w.g2_scale = static_cast<float>(gg.get_f64(p + "att.g2.scale", 1.0));
        w.v1_scale = static_cast<float>(gg.get_f64(p + "att.v1.scale", 1.0));
        w.v2_scale = static_cast<float>(gg.get_f64(p + "att.v2.scale", 1.0));

        w.w_bias = load_f32(gg, p + "att.w_bias", error_);
        w.a_bias = load_f32(gg, p + "att.a_bias", error_);
        w.v_bias = load_f32(gg, p + "att.v_bias", error_);
        if (!error_.empty()) return false;

        w.k_k  = load_f32(gg, p + "att.k_k", error_);
        w.k_a  = load_f32(gg, p + "att.k_a", error_);
        w.r_k  = load_f32(gg, p + "att.r_k", error_);
        w.gn_w = load_f32(gg, p + "att.gn.weight", error_);
        w.gn_b = load_f32(gg, p + "att.gn.bias", error_);
        if (!error_.empty()) return false;

        // Channel mixing: key [ffn_inter, E], value [E, ffn_inter].
        const std::vector<int64_t> fe = {cfg_.ffn_inter, E};
        const std::vector<int64_t> ef = {E, cfg_.ffn_inter};
        w.F_mix_v = load_f32(gg, p + "ffn.tmix_v", error_);
        if (!error_.empty()) return false;
        w.F_key   = load_ternary_view(gg, p + "ffn.key.weight", fe, error_);
        w.F_value = load_ternary_view(gg, p + "ffn.value.weight", ef, error_);
        if (!error_.empty()) return false;
        w.sc_fk = load_f32(gg, p + "ffn.key.scale", error_, false);
        w.sc_fv = load_f32(gg, p + "ffn.value.scale", error_, false);
        if (!error_.empty()) return false;
        w.fk_scale = static_cast<float>(gg.get_f64(p + "ffn.key.scale", 1.0));
        w.fv_scale = static_cast<float>(gg.get_f64(p + "ffn.value.scale", 1.0));
    }

    return true;
}

bool RwkvModel::load(const std::string& path) {
    // store_ owns the mmap; tensor views taken below point into it and must
    // stay valid for the model's lifetime.
    if (!store_.open(path)) {
        error_ = store_.error();
        return false;
    }
    if (!load_meta(store_)) return false;
    if (!load_weights(store_)) return false;
    valid_ = true;
    return true;
}

// ===========================================================================
// State
// ===========================================================================
void RwkvModel::init_state(RwkvState& st) const {
    st.tmix_state.clear();
    st.ffn_state.clear();
    st.attn_state.clear();
    st.tokens_seen = 0;

    for (int32_t l = 0; l < cfg_.n_layers; ++l) {
        st.tmix_state.emplace_back("tmix",
                                   std::vector<int64_t>{cfg_.n_embd}, DType::F32);
        st.ffn_state.emplace_back("ffn",
                                  std::vector<int64_t>{cfg_.n_embd}, DType::F32);
        st.attn_state.emplace_back(
            "attn", std::vector<int64_t>{cfg_.n_heads, cfg_.head_size,
                                         cfg_.head_size}, DType::F32);
    }
}

void RwkvModel::copy_state(const RwkvState& src, RwkvState& dst) {
    dst.tmix_state.clear();
    dst.ffn_state.clear();
    dst.attn_state.clear();
    for (const Tensor& t : src.tmix_state)
        dst.tmix_state.push_back(tensor_ops::cast(t, DType::F32));
    for (const Tensor& t : src.ffn_state)
        dst.ffn_state.push_back(tensor_ops::cast(t, DType::F32));
    for (const Tensor& t : src.attn_state)
        dst.attn_state.push_back(tensor_ops::cast(t, DType::F32));
    dst.tokens_seen = src.tokens_seen;
}

// ===========================================================================
// Ternary projection helper
// ===========================================================================
void RwkvModel::project(const Tensor& W, float scale, const float* x,
                        float* y, int64_t out_dim, int64_t in_dim) const {
    bitnet::bitlinear_forward(W.packed(), nullptr, scale, x, y,
                              out_dim, in_dim);
}

void RwkvModel::project(const Tensor& W, const Tensor& scales,
                        float scalar_scale, const float* x, float* y,
                        int64_t out_dim, int64_t in_dim) const {
    if (scales.numel() == out_dim) {
        if (W.dtype() == DType::I8) {
            // int8: dequantize the row on the fly (i8 kernel handles the
            // sign + accumulate; multiply by the row scale at the end).
            bitnet::bitlinear_forward_i8(W.i8(), scales.f32(), x, y,
                                         out_dim, in_dim);
            return;
        }
        bitnet::bitlinear_forward_rows_simd(W.packed(), scales.f32(), x, y,
                                            out_dim, in_dim);
        return;
    }
    bitnet::bitlinear_forward(W.packed(), nullptr, scalar_scale, x, y,
                              out_dim, in_dim);
}

// ===========================================================================
// Forward: one token step
// ===========================================================================
void RwkvModel::forward(int32_t token, RwkvState& st, Tensor& logits) const {
    const int32_t E = cfg_.n_embd;
    const int32_t H = cfg_.n_heads;
    const int32_t D = cfg_.head_size;

    std::vector<float> x(E);

    // ------------------------------- embedding -------------------------------
    {
        const uint16_t* erow = emb_.f16() + static_cast<int64_t>(token) * E;
        for (int32_t i = 0; i < E; ++i) x[i] = half_to_float(erow[i]);
    }

    // ln0 (present in RWKV-7 world models)
    if (ln0_w_.numel() > 1) {   // absent tensors: numel()==1, data()==nullptr
        Tensor xin("xin", {E}, DType::F32);
        Tensor xout("xout", {E}, DType::F32);
        std::memcpy(xin.f32(), x.data(), E * sizeof(float));
        tensor_ops::layer_norm(xin, &ln0_w_, &ln0_b_, xout);
        std::memcpy(x.data(), xout.f32(), E * sizeof(float));
    }

    // --------------------------------- blocks --------------------------------
    bool have_v0 = false;
    std::vector<float> v0(E);

    for (int32_t l = 0; l < cfg_.n_layers; ++l) {
        const RwkvLayerWeights& w = layers_[static_cast<size_t>(l)];

        // ---- ln1 + 6 token-shifted streams -----------------------------------
        Tensor xl("xl", {E}, DType::F32);
        {
            Tensor xin("xin", {E}, DType::F32);
            std::memcpy(xin.f32(), x.data(), E * sizeof(float));
            tensor_ops::layer_norm(xin, &w.ln1_w, &w.ln1_b, xl);
        }
        const float* lx   = xl.f32();
        const float* last = st.tmix_state[static_cast<size_t>(l)].f32();

        std::vector<float> xr(E), xw(E), xk(E), xv(E), xa(E), xg(E);
        // token-shift blend: xs = lx + mix*(lx_prev - lx), state stores the NORMED lx
        // (identical for BlinkDL and official-HF/fla checkpoints — verified against
        // transformers' Rwkv7TokenShift + fused_addcmul algebra).
        for (int32_t i = 0; i < E; ++i) {
            const float d = last[i] - lx[i];
            xr[i] = lx[i] + w.tmix_r.f32()[i] * d;
            xw[i] = lx[i] + w.tmix_w.f32()[i] * d;
            xk[i] = lx[i] + w.tmix_k.f32()[i] * d;
            xv[i] = lx[i] + w.tmix_v.f32()[i] * d;
            xa[i] = lx[i] + w.tmix_a.f32()[i] * d;
            xg[i] = lx[i] + w.tmix_g.f32()[i] * d;
        }
        std::memcpy(st.tmix_state[static_cast<size_t>(l)].f32(), lx,
                    E * sizeof(float));

        // ---- main projections (ternary) --------------------------------------
        const double p0 = dbg_prof() ? platform::now_ms() : 0.0;
        std::vector<float> r(E), k(E), kk(E), v(E), g(E);
        project(w.W_r, w.sc_r, w.r_scale, xr.data(), r.data(), E, E);  // raw (no sigmoid)
        project(w.W_k, w.sc_k, w.k_scale, xk.data(), k.data(), E, E);
        std::memcpy(kk.data(), k.data(), E * sizeof(float));     // kk = k * k_k below
        project(w.W_v, w.sc_v, w.v_scale, xv.data(), v.data(), E, E);
        if (dbg_nan() && l == 0) {
            dbg_check("xr", xr.data(), E);
            dbg_check("proj_r", r.data(), E);
            dbg_check("proj_k", k.data(), E);
            dbg_check("proj_v", v.data(), E);
            dbg_check("sc_r", w.sc_r.f32(), E);
            dbg_check("W_r.i8", reinterpret_cast<const float*>(w.W_r.i8()), 64);
        }
        const double p_proj = dbg_prof() ? platform::now_ms() : 0.0;

        // g = sigmoid(xg @ Wg1) @ Wg2   (sigmoid BETWEEN projections)
        {
            const int32_t rg = cfg_.rank_g;
            std::vector<float> g1(rg);
            project(w.W_g1, w.sc_g1, w.g1_scale, xg.data(), g1.data(), rg, E);
            for (int32_t i = 0; i < rg; ++i) g1[i] = sigmoidf(g1[i]);
            project(w.W_g2, w.sc_g2, w.g2_scale, g1.data(), g.data(), E, rg);
        }

        // ---- w = exp(-sigmoid(tanh(Ww1 x) @ Ww2 + w_bias)/sqrt(e)) -----------
        std::vector<float> wdec(E);
        {
            const int32_t rw = cfg_.rank_w;
            std::vector<float> w1(rw), w2(E);
            project(w.W_w1, w.sc_w1, w.w1_scale, xw.data(), w1.data(), rw, E);
            for (int32_t i = 0; i < rw; ++i) w1[i] = std::tanh(w1[i]);
            project(w.W_w2, w.sc_w2, w.w2_scale, w1.data(), w2.data(), E, rw);
            const float inv_sqrte = 1.0f / std::sqrt(2.718281828459045f);
            for (int32_t i = 0; i < E; ++i) {
                wdec[i] = std::exp(
                    -sigmoidf(w2[i] + w.w_bias.f32()[i]) * inv_sqrte);
            }
        }

        // ---- value residual: v += (v0 - v) * sigmoid(Wv1 x @ Wv2 + v_bias) ---
        {
            const int32_t rv = cfg_.rank_v;
            std::vector<float> v1(rv), v2(E);
            project(w.W_v1, w.sc_v1, w.v1_scale, xv.data(), v1.data(), rv, E);
            project(w.W_v2, w.sc_v2, w.v2_scale, v1.data(), v2.data(), E, rv);
            if (!have_v0) {
                // Block 0: v0 = v (residual is a no-op this token).
                std::memcpy(v0.data(), v.data(), E * sizeof(float));
                have_v0 = true;
            } else {
                for (int32_t i = 0; i < E; ++i) {
                    const float s = sigmoidf(v2[i] + w.v_bias.f32()[i]);
                    v[i] += (v0[i] - v[i]) * s;
                }
            }
        }

        // ---- a = sigmoid(Wa1 x @ Wa2 + a_bias) --------------------------------
        std::vector<float> a(E);
        {
            const int32_t ra = cfg_.rank_a;
            std::vector<float> a1(ra), a2(E);
            project(w.W_a1, w.sc_a1, w.a1_scale, xa.data(), a1.data(), ra, E);
            project(w.W_a2, w.sc_a2, w.a2_scale, a1.data(), a2.data(), E, ra);
            for (int32_t i = 0; i < E; ++i)
                a[i] = sigmoidf(a2[i] + w.a_bias.f32()[i]);
        }

        // ---- kk = k * k_k;  k2 = k * (1 + (a-1) * k_a)  (blend uses RAW k) --
        std::vector<float> k2(E);
        for (int32_t i = 0; i < E; ++i) {
            kk[i] *= w.k_k.f32()[i];
            k2[i] = k[i] * (1.0f + (a[i] - 1.0f) * w.k_a.f32()[i]);
        }
        const double p_gates = dbg_prof() ? platform::now_ms() : 0.0;

        float* S_base = st.attn_state[static_cast<size_t>(l)].f32();

        // ---- per-head L2 normalize kk ----------------------------------------
        for (int32_t h = 0; h < H; ++h) {
            float* kkh = kk.data() + static_cast<size_t>(h) * D;
            double n2 = 0.0;
            for (int32_t i = 0; i < D; ++i)
                n2 += static_cast<double>(kkh[i]) * kkh[i];
            const float inv =
                static_cast<float>(1.0 / std::sqrt(n2 + 1e-12));
            for (int32_t i = 0; i < D; ++i) kkh[i] *= inv;
        }

        // ---- S = S*w^T - (S@kk)(kk*a)^T + v k2^T  (decay on KEY axis j) ------
        std::vector<float> Skk(D);
        for (int32_t h = 0; h < H; ++h) {
            float*       S   = S_base + static_cast<size_t>(h) * D * D;
            const float* kkh = kk.data() + static_cast<size_t>(h) * D;
            const float* kh  = k2.data() + static_cast<size_t>(h) * D;
            const float* vh  = v.data()  + static_cast<size_t>(h) * D;
            const float* wh  = wdec.data() + static_cast<size_t>(h) * D;
            const float* ah  = a.data()  + static_cast<size_t>(h) * D;

            for (int32_t i = 0; i < D; ++i) {
                float acc = 0.0f;
                const float* row = S + static_cast<size_t>(i) * D;
                for (int32_t j = 0; j < D; ++j) acc += row[j] * kkh[j];
                Skk[static_cast<size_t>(i)] = acc;
            }
            for (int32_t i = 0; i < D; ++i) {
                float*       row = S + static_cast<size_t>(i) * D;
                const float  skk = Skk[static_cast<size_t>(i)];
                const float  vi  = vh[i];
                for (int32_t j = 0; j < D; ++j) {
                    row[j] = row[j] * wh[j]            // decay per key channel j
                           - skk * (kkh[j] * ah[j])    // remove along k̂, gated by a
                           + vi * kh[j];               // write v k2^T
                }
            }
        }

        // ---- y = group_norm(S @ r); extra residue; out = Wo @ (y*g) ----------
        const double p_srec = dbg_prof() ? platform::now_ms() : 0.0;
        std::vector<float> y(E), yn(E), yg(E);
        for (int32_t h = 0; h < H; ++h) {
            const float* S  = S_base + static_cast<size_t>(h) * D * D;
            const float* rh = r.data() + static_cast<size_t>(h) * D;
            float* yh = y.data() + static_cast<size_t>(h) * D;
            for (int32_t i = 0; i < D; ++i) {
                float acc = 0.0f;
                const float* row = S + static_cast<size_t>(i) * D;
                for (int32_t j = 0; j < D; ++j) acc += row[j] * rh[j];
                yh[i] = acc;
            }
        }
        group_norm_heads(y.data(), H, D, w.gn_w, w.gn_b, yn.data());

        // extra residue: per-head scalar sum_d(r[d]*k2[d]*r_k[d]) times v[i]
        // (numpy + HF references both apply k += k*(a-1)*k_a BEFORE the bonus,
        //  so the bonus contracts with the BLENDED key k2)
        for (int32_t h = 0; h < H; ++h) {
            const float* rh  = r.data()  + static_cast<size_t>(h) * D;
            const float* kh  = k2.data() + static_cast<size_t>(h) * D;
            const float* rkh = w.r_k.f32() + static_cast<size_t>(h) * D;
            const float* vh  = v.data()  + static_cast<size_t>(h) * D;
            float*       ynh = yn.data() + static_cast<size_t>(h) * D;
            float s = 0.0f;
            for (int32_t d = 0; d < D; ++d) s += rh[d] * kh[d] * rkh[d];
            for (int32_t i = 0; i < D; ++i) ynh[i] += s * vh[i];
        }
        for (int32_t i = 0; i < E; ++i) yg[i] = yn[i] * g[i];
        const double p_readout = dbg_prof() ? platform::now_ms() : 0.0;

        std::vector<float> attn_out(E);
        project(w.W_o, w.sc_o, w.o_scale, yg.data(), attn_out.data(), E, E);
        if (dbg_nan() && l == 0) {
            dbg_check("y", y.data(), E);
            dbg_check("yn", yn.data(), E);
            dbg_check("yg", yg.data(), E);
            dbg_check("attn_out", attn_out.data(), E);
            dbg_check("wdec", wdec.data(), E);
            dbg_check("a", a.data(), E);
            dbg_check("k2", k2.data(), E);
            dbg_check("S_block0", S_base, static_cast<int64_t>(D) * D);
        }
        for (int32_t i = 0; i < E; ++i) x[i] += attn_out[i];
        if (dbg_nan() && l == 0) dbg_check("x_after_block0", x.data(), E);

        // ======================= channel mixing ==============================
        Tensor xl2("xl2", {E}, DType::F32);
        {
            Tensor xin2("xin2", {E}, DType::F32);
            std::memcpy(xin2.f32(), x.data(), E * sizeof(float));
            tensor_ops::layer_norm(xin2, &w.ln2_w, &w.ln2_b, xl2);
        }
        const float* lx2   = xl2.f32();
        const float* last2 = st.ffn_state[static_cast<size_t>(l)].f32();

        std::vector<float> xv2(E);
        // ffn token-shift: same blend form as the attention stream.
        for (int32_t i = 0; i < E; ++i)
            xv2[i] = lx2[i] + w.F_mix_v.f32()[i] * (last2[i] - lx2[i]);
        std::memcpy(st.ffn_state[static_cast<size_t>(l)].f32(), lx2,
                    E * sizeof(float));

        std::vector<float> fk(cfg_.ffn_inter), fv(E);
        project(w.F_key, w.sc_fk, w.fk_scale, xv2.data(), fk.data(), cfg_.ffn_inter, E);
        for (int32_t i = 0; i < cfg_.ffn_inter; ++i) fk[i] = squared_relu(fk[i]);
        project(w.F_value, w.sc_fv, w.fv_scale, fk.data(), fv.data(), E, cfg_.ffn_inter);
        for (int32_t i = 0; i < E; ++i) x[i] += fv[i];
        if (dbg_prof()) {
            ProfSections& p = prof();
            const double now = platform::now_ms();
            p.proj += p_proj - p0;
            p.gates += p_gates - p_proj;
            p.srec += p_srec - p_gates;
            p.readout += now - p_readout;
        }
    }
    // ------------------------------- head -------------------------------------
    Tensor xl3("xl3", {E}, DType::F32);
    {
        Tensor xin3("xin3", {E}, DType::F32);
        std::memcpy(xin3.f32(), x.data(), E * sizeof(float));
        tensor_ops::layer_norm(xin3, &ln_out_w_, &ln_out_b_, xl3);
    }

    const int64_t V_head = head_.dim(0);
    const int64_t V_out  = std::min<int64_t>(logits.numel(), V_head);
    logits.zero();
    const double p_head = dbg_prof() ? platform::now_ms() : 0.0;
    if (head_ternary_) {
        if (head_scales_.numel() == V_out) {
            bitnet::bitlinear_forward_rows_simd(head_.packed(),
                                                head_scales_.f32(),
                                                xl3.f32(), logits.f32(),
                                                V_out, E);
        } else {
            bitnet::bitlinear_forward(head_.packed(), nullptr, head_scale_,
                                      xl3.f32(), logits.f32(), V_out, E);
        }
    } else if (head_.dtype() == DType::I8) {
        // fp16 head converted to per-row i8 at load time (see load_weights).
        bitnet::bitlinear_forward_i8(head_.i8(), head_scales_.f32(),
                                     xl3.f32(), logits.f32(), V_out, E);
    }
    if (dbg_prof()) {
        ProfSections& p = prof();
        p.calls += 1;
        p.head += platform::now_ms() - p_head;
        p.dump();
    }
    st.tokens_seen += 1;
}

int32_t RwkvModel::greedy_pick(const Tensor& logits) const {
    const float* p = logits.f32();
    const int64_t n = std::min<int64_t>(logits.numel(), cfg_.n_vocab);
    int32_t best = 0;
    float bv = -1e30f;
    for (int64_t i = 0; i < n; ++i) {
        if (p[i] > bv) { bv = p[i]; best = static_cast<int32_t>(i); }
    }
    return best;
}

int32_t RwkvModel::sample_token(const Tensor& logits, float temperature,
                                int32_t top_k, uint64_t& seed) const {
    const float* p = logits.f32();
    const int64_t n = std::min<int64_t>(logits.numel(), cfg_.n_vocab);
    if (n <= 0) return 0;
    if (temperature <= 0.0f) return greedy_pick(logits);

    // top-k indices (keep all if top_k <= 0 or >= n)
    std::vector<int32_t> idx;
    idx.reserve(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) idx.push_back(static_cast<int32_t>(i));
    if (top_k > 0 && top_k < n) {
        std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(),
                          [&](int32_t a, int32_t b) { return p[a] > p[b]; });
        idx.resize(static_cast<size_t>(top_k));
    }
    const int32_t k = static_cast<int32_t>(idx.size());

    // softmax over the kept logits
    float mx = -1e30f;
    for (int32_t i = 0; i < k; ++i) mx = std::max(mx, p[idx[static_cast<size_t>(i)]]);
    double sum = 0.0;
    std::vector<double> prob(static_cast<size_t>(k));
    for (int32_t i = 0; i < k; ++i) {
        const float t = (p[idx[static_cast<size_t>(i)]] - mx) /
                        std::max(temperature, 1e-4f);
        prob[static_cast<size_t>(i)] = std::exp(static_cast<double>(t));
        sum += prob[static_cast<size_t>(i)];
    }

    // SplitMix64 next() — deterministic per seed
    seed += 0x9E3779B97F4A7C15ull;
    uint64_t z = seed;
    z = (z ^ (z >> 30ull)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27ull)) * 0x94D049BB133111EBull;
    z ^= (z >> 31ull);
    const double u = static_cast<double>(z >> 11ull) * (1.0 / 9007199254740992.0);

    double acc = 0.0;
    for (int32_t i = 0; i < k; ++i) {
        acc += prob[static_cast<size_t>(i)] / sum;
        if (u < acc) return idx[static_cast<size_t>(i)];
    }
    return idx[static_cast<size_t>(k - 1)];
}

} // namespace omniseed
