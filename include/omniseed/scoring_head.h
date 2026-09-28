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

#include "omniseed/core/batch_gemm.h"
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

    // ---- the batched path ---------------------------------------------------
    // The same readout for B hidden states at once, so a backtest loop pays the
    // per-call overhead once instead of B times. With the Scalar kernel the
    // result is BIT-IDENTICAL to calling score() B times — the test asserts
    // equality, not a tolerance.
    //
    // FAILS CLOSED: on a not-ready head, a null pointer, or a Tensor whose shape
    // is not [B, E], returns false and leaves `out` EMPTY. Never a partial
    // batch: a caller that ignores the return value must not act on half an
    // answer.
    //
    // In a batched result `ScoreResult::us` is the BATCH's per-row average, not
    // that row's own time — a per-row clock read is precisely the overhead the
    // batch exists to avoid. Per-batch totals are in `BatchStats`.
    bool score_batch(const float* H, int32_t B, std::vector<ScoreResult>& out,
                     BatchStats* stats = nullptr) const;
    bool score_batch(const Tensor& H, std::vector<ScoreResult>& out,
                     BatchStats* stats = nullptr) const;

    // Auto (the default) picks Simd when the CPU supports it.
    void set_batch_kernel(BatchKernel k) { batch_kernel_ = k; }
    BatchKernel batch_kernel() const { return batch_kernel_; }

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

    BatchKernel batch_kernel_ = BatchKernel::Auto;

    mutable double last_us_ = 0.0;
    // Reused scratch for the batched path, so a batch is allocation-free after
    // the first call with a given B. mutable because score_batch() is const.
    mutable std::vector<float> logits_batch_;
};

} // namespace omniseed
