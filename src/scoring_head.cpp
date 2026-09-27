// =============================================================================
//  OmniSeed — src/scoring_head.cpp
//
//  Three sigmoids over the hidden state. See scoring_head.h for why the axes
//  are independent and for the honesty note about untrained projections.
// =============================================================================
#include "omniseed/scoring_head.h"
#include "omniseed/core/platform.h"

#include <cmath>
#include <cstring>

namespace omniseed {

ScoringHead::ScoringHead(const RwkvConfig& cfg, uint32_t seed) { init(cfg, seed); }

bool ScoringHead::init(const RwkvConfig& cfg, uint32_t seed) { return init(cfg.n_embd, seed); }

bool ScoringHead::init(int32_t n_embd, uint32_t seed) {
    ready_ = false;
    error_.clear();
    fitted_rows_ = 0;

    if (n_embd <= 0) {
        error_ = "ScoringHead::init: n_embd must be > 0";
        provenance_ = "error: " + error_;
        return false;
    }
    E_ = n_embd;
    proj_.assign(static_cast<size_t>(ROW_COUNT) * static_cast<size_t>(E_), 0.0f);
    bias_.assign(ROW_COUNT, 0.0f);
    fitted_.assign(ROW_COUNT, 0);
    seed_weights(seed);
    ready_ = true;
    provenance_ = "seeded placeholder (UNTRAINED) — splitmix64 seed " +
                  std::to_string(seed) + ", E=" + std::to_string(E_) +
                  ", 3 rows (0 fitted)";
    return true;
}

void ScoringHead::seed_weights(uint32_t seed) {
    // Multiply-add, NOT `seed | constant` — see the note in router.cpp: OR-ing
    // a constant into the seed is not injective and lets two different seeds
    // produce identical heads.
    uint64_t s = static_cast<uint64_t>(seed) * 0x9E3779B97F4A7C15ull +
                 0xD1B54A32D192ED03ull;
    const float scale = 1.0f / std::sqrt(static_cast<float>(E_));
    for (float& w : proj_) w = heads_detail::next_unit(s) * scale;
    for (float& b : bias_) b = heads_detail::next_unit(s) * 0.1f;
}

void ScoringHead::set_row(Row which, const float* row, float bias) {
    if (!ready_ || row == nullptr) return;
    const int32_t idx = static_cast<int32_t>(which);
    if (idx < 0 || idx >= ROW_COUNT) return;
    std::memcpy(&proj_[static_cast<size_t>(idx) * static_cast<size_t>(E_)], row,
                sizeof(float) * static_cast<size_t>(E_));
    bias_[static_cast<size_t>(idx)] = bias;
    if (!fitted_[static_cast<size_t>(idx)]) {
        fitted_[static_cast<size_t>(idx)] = 1;
        ++fitted_rows_;
    }
}

namespace {

inline float sigmoidf(float x) {
    if (x >= 0.0f) {
        const float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    const float z = std::exp(x);
    return z / (1.0f + z);
}

} // namespace

ScoreResult ScoringHead::score(const float* hidden) const {
    ScoreResult r;
    if (!ready_ || hidden == nullptr) return r;   // matvecs == 0 marks the failure

    const double t0 = platform::now_ms();
    float out[ROW_COUNT] = {0.0f, 0.0f, 0.0f};
    for (int32_t i = 0; i < ROW_COUNT; ++i) {
        const float* w = &proj_[static_cast<size_t>(i) * static_cast<size_t>(E_)];
        float acc = bias_[static_cast<size_t>(i)];
        for (int32_t e = 0; e < E_; ++e) acc += w[e] * hidden[e];
        out[i] = sigmoidf(acc);
    }
    r.priority   = out[Priority];
    r.urgency    = out[Urgency];
    r.confidence = out[Confidence];
    r.matvecs    = ROW_COUNT;
    r.us         = (platform::now_ms() - t0) * 1000.0;
    last_us_ = r.us;
    return r;
}

ScoreResult ScoringHead::score(const Tensor& hidden) const {
    if (hidden.dtype() != DType::F32) return ScoreResult();
    if (hidden.numel() != static_cast<int64_t>(E_)) return ScoreResult();
    return score(hidden.f32());
}

} // namespace omniseed
