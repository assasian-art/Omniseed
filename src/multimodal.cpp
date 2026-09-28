// =============================================================================
//  OmniSeed — multimodal.cpp
//  The joint: continuous modality embeddings -> discrete tokens -> one stream.
// =============================================================================
#include "omniseed/multimodal.h"

#include <cmath>
#include <cstdio>

namespace omniseed {

namespace {

// A cheap, allocation-free finiteness check. Done once per query vector (E
// elements) rather than per candidate row, so it costs O(E) against a scan of
// O(n_vocab/stride * E) — free by comparison, and it keeps a NaN encoder output
// from silently winning a distance comparison (every comparison against NaN is
// false, so a NaN row would otherwise be "nearest" to everything).
bool all_finite(const float* v, int32_t n) {
    for (int32_t i = 0; i < n; ++i)
        if (!std::isfinite(v[i])) return false;
    return true;
}

} // namespace

// ---------------------------------------------------------------------------
bool MultimodalBridge::init(int32_t n_embd, const Tokenizer& tok) {
    ready_ = false;
    error_.clear();
    codebook_ = nullptr;
    codebook_rows_ = 0;
    codebook_cols_ = 0;

    if (n_embd <= 0) {
        error_ = "n_embd must be > 0";
        return false;
    }
    if (cfg_.codebook_stride < 1) cfg_.codebook_stride = 1;

    n_embd_ = n_embd;
    tok_    = &tok;
    // TokenBus holds a Tokenizer& and has no default ctor or assignment, so it
    // must be constructed in place rather than stored by value.
    bus_.reset(new TokenBus(tok));
    bus_->set_max_tokens(cfg_.max_tokens);
    stride_ = cfg_.codebook_stride;

    ready_ = true;
    return true;
}

void MultimodalBridge::set_config(const Config& c) {
    cfg_ = c;
    if (cfg_.codebook_stride < 1) cfg_.codebook_stride = 1;
    stride_ = cfg_.codebook_stride;
    if (bus_) bus_->set_max_tokens(cfg_.max_tokens);
}

// ---------------------------------------------------------------------------
bool MultimodalBridge::set_codebook(const Tensor& embeddings) {
    codebook_ = nullptr;
    codebook_rows_ = 0;
    codebook_cols_ = 0;

    if (!ready_) {
        error_ = "bridge not initialised";
        return false;
    }
    if (embeddings.shape().size() != 2) {
        error_ = "codebook must be 2-D [n_vocab, n_embd]";
        return false;
    }
    const int32_t rows = static_cast<int32_t>(embeddings.dim(0));
    const int32_t cols = static_cast<int32_t>(embeddings.dim(1));
    if (rows <= 0 || cols <= 0) {
        error_ = "codebook has an empty dimension";
        return false;
    }
    // REFUSE a width mismatch rather than truncating: quantizing against the
    // wrong half of every row would produce plausible-looking, meaningless ids.
    if (cols != n_embd_) {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
                      "codebook width %d != n_embd %d", cols, n_embd_);
        error_ = buf;
        return false;
    }
    const DType dt = embeddings.dtype();
    if (dt != DType::F16 && dt != DType::F32) {
        error_ = "codebook dtype must be F16 or F32";
        return false;
    }

    codebook_      = &embeddings;
    codebook_rows_ = rows;
    codebook_cols_ = cols;
    stride_        = cfg_.codebook_stride < 1 ? 1 : cfg_.codebook_stride;
    error_.clear();
    return true;
}

// ---------------------------------------------------------------------------
// Squared Euclidean distance to the nearest vocabulary embedding.
//
// The early-out is what makes a 65536-row scan affordable: the moment a partial
// sum reaches the best distance so far the row cannot win, so we abandon it.
// `worse` is required — without it the partial sum would compare as SMALLER
// than best_d and the row would be accepted precisely because it was cut short.
// ---------------------------------------------------------------------------
int32_t MultimodalBridge::nearest_token(const float* vec) const {
    if (vec == nullptr || codebook_ == nullptr) return -1;
    const int32_t E    = codebook_cols_;
    const int32_t step = stride_ < 1 ? 1 : stride_;
    if (!all_finite(vec, E)) return -1;

    int32_t best   = -1;
    float   best_d = 0.0f;

    if (codebook_->dtype() == DType::F16) {
        const uint16_t* rows = codebook_->f16();
        for (int32_t r = 0; r < codebook_rows_; r += step) {
            const uint16_t* row = rows + static_cast<int64_t>(r) * E;
            float d = 0.0f;
            bool  worse = false;
            for (int32_t e = 0; e < E; ++e) {
                const float diff = vec[e] - half_to_float(row[e]);
                d += diff * diff;
                if (best >= 0 && d >= best_d) { worse = true; break; }
            }
            if (!worse && (best < 0 || d < best_d)) { best_d = d; best = r; }
        }
    } else {
        const float* rows = codebook_->f32();
        for (int32_t r = 0; r < codebook_rows_; r += step) {
            const float* row = rows + static_cast<int64_t>(r) * E;
            float d = 0.0f;
            bool  worse = false;
            for (int32_t e = 0; e < E; ++e) {
                const float diff = vec[e] - row[e];
                d += diff * diff;
                if (best >= 0 && d >= best_d) { worse = true; break; }
            }
            if (!worse && (best < 0 || d < best_d)) { best_d = d; best = r; }
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
bool MultimodalBridge::quantize(const std::vector<float>& rows,
                                std::vector<int32_t>& out_ids) const {
    out_ids.clear();
    if (rows.empty()) return true;              // "no image" is not an error
    if (n_embd_ <= 0) return false;
    if (rows.size() % static_cast<size_t>(n_embd_) != 0) return false;

    const size_t M = rows.size() / static_cast<size_t>(n_embd_);
    out_ids.reserve(M);
    for (size_t m = 0; m < M; ++m) {
        const int32_t id = nearest_token(rows.data() + m * n_embd_);
        if (id < 0) { out_ids.clear(); return false; }
        out_ids.push_back(id);
    }
    return true;
}

// ---------------------------------------------------------------------------
MultimodalBridge::Fusion MultimodalBridge::fuse(
    const std::string& text,
    const std::vector<float>& vision_embeddings,
    const std::vector<int32_t>& audio_codes) const {
    Fusion f;
    if (!ready_ || !bus_) {
        f.error = "bridge not initialised";
        return f;
    }

    std::vector<TokenBus::Segment> segs;

    if (cfg_.include_text && !text.empty()) {
        TokenBus::Segment s;
        s.kind = TokenBus::Segment::Kind::Text;
        s.text = text;
        segs.push_back(std::move(s));
    }

    if (!vision_embeddings.empty()) {
        if (!codebook_ready()) {
            // Do NOT fall back to dropping the image: a caller that passed an
            // image and got a text-only answer would never learn that the
            // image was ignored.
            f.error = "vision supplied but no codebook (call set_codebook)";
            return f;
        }
        std::vector<int32_t> vids;
        if (!quantize(vision_embeddings, vids)) {
            f.error = "vision embedding block is not a whole number of n_embd rows";
            return f;
        }
        f.vision_tokens_emitted = static_cast<int32_t>(vids.size());
        TokenBus::Segment s;
        s.kind = TokenBus::Segment::Kind::Vision;
        s.ids  = std::move(vids);
        segs.push_back(std::move(s));
    }

    if (!audio_codes.empty()) {
        f.audio_tokens_emitted = static_cast<int32_t>(audio_codes.size());
        TokenBus::Segment s;
        s.kind = TokenBus::Segment::Kind::Audio;
        s.ids  = audio_codes;
        segs.push_back(std::move(s));
    }

    std::vector<int32_t> ids;
    if (!bus_->fuse(segs, ids)) {
        f.error = "token budget exceeded";
        return f;
    }

    f.ids   = std::move(ids);
    f.stats = bus_->last_stats();
    f.ok    = true;
    return f;
}

// ---------------------------------------------------------------------------
std::string MultimodalBridge::summary(const Fusion& f) const {
    if (!f.ok) return std::string("fusion failed: ") + f.error;
    char buf[192];
    std::snprintf(buf, sizeof(buf),
                  "fused %lld tokens (text %lld, vision %d, audio %d)",
                  static_cast<long long>(f.stats.total),
                  static_cast<long long>(f.stats.text_tokens),
                  f.vision_tokens_emitted, f.audio_tokens_emitted);
    return std::string(buf);
}

} // namespace omniseed
