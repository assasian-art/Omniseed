// =============================================================================
//  OmniSeed — scoring_head.h
//
//  Three independent sigmoids over h[E]: priority, urgency, confidence.
//
//  WHY SIGMOIDS AND NOT A SOFTMAX. These are three SEPARATE questions — "how
//  important is this", "how soon does it need to happen", "how sure am I of the
//  first two". Forcing them through a softmax would make them sum to 1, so a
//  genuinely urgent item would necessarily be scored as unimportant. The whole
//  point of a score triple is that the axes are independent.
//
//  COST. 3*E multiply-accumulates — at E = 768 that is 2,304 MACs, well under a
//  microsecond. Allocation-free after init(); h is never mutated.
//
//  HONESTY NOTE. A freshly constructed head has NO trained projection: init()
//  seeds it deterministically so tests are reproducible, and `trained()` /
//  `provenance()` report that. A seeded head emits well-formed but MEANINGLESS
//  numbers. Note in particular that `ScoreResult::confidence` is the head's
//  confidence in its OWN score, not a calibration claim about anything else —
//  and it is not meaningful until the projection is fitted.
// =============================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "omniseed/core/rwkv.h"
#include "omniseed/core/tensor.h"
#include "omniseed/heads.h"

namespace omniseed {

class ScoringHead {
public:
    // The three rows, in wire order. `COUNT` is the sentinel.
    enum Row : int32_t {
        Priority = 0,
        Urgency,
        Confidence,
        ROW_COUNT
    };

    ScoringHead() = default;
    explicit ScoringHead(const RwkvConfig& cfg, uint32_t seed = 555u);

    bool init(const RwkvConfig& cfg, uint32_t seed = 555u);
    bool init(int32_t n_embd, uint32_t seed = 555u);

    bool ready() const { return ready_; }
    const std::string& error() const { return error_; }
    int32_t hidden_size() const { return E_; }

    // ---- the hot path -------------------------------------------------------
    // Three matvecs + three sigmoids. Allocation-free after init(); not
    // thread-safe against itself.
    //
    // On a not-ready head or a width mismatch this returns an all-zero ScoreResult
    // rather than inventing a score, and `matvecs == 0` marks it. Zero is the
    // right failure value here: a priority of 0 means "no opinion", and unlike a
    // classification there is no separate "empty" channel to carry the error in.
    ScoreResult score(const float* hidden) const;
    ScoreResult score(const Tensor& hidden) const;

    // ---- offline fitting hooks ---------------------------------------------
    void set_row(Row which, const float* row, float bias);

    // ---- provenance ---------------------------------------------------------
    bool trained() const { return ready_ && fitted_rows_ == ROW_COUNT; }
    int32_t fitted_rows() const { return fitted_rows_; }
    const std::string& provenance() const { return provenance_; }
    double last_us() const { return last_us_; }

private:
    void seed_weights(uint32_t seed);

    int32_t E_ = 0;

    std::vector<float>   proj_;   // [ROW_COUNT, E_], row-major
    std::vector<float>   bias_;   // [ROW_COUNT]
    std::vector<uint8_t> fitted_; // [ROW_COUNT]

    bool        ready_       = false;
    int32_t     fitted_rows_ = 0;
    std::string provenance_  = "uninitialised";
    std::string error_;

    mutable double last_us_ = 0.0;
};

} // namespace omniseed
