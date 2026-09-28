// =============================================================================
//  OmniSeed — decision_head.h
//
//  "TWO HEADS, ONE BRAIN" — the System-1 decision head.
//
//  The RWKV-7 backbone is the one brain: a single forward pass turns the
//  prompt into a hidden state h[E] (the post-ln_out residual, already
//  layernormed into xl3 inside RwkvModel::forward). That hidden state is the
//  model's entire opinion about the situation it just read.
//
//  Classically we would now decode that opinion into TEXT and then parse the
//  text back into a decision — an autoregressive token loop whose cost scales
//  with max_new_tokens, and whose output is a string we have to re-parse.
//  The decision head short-circuits that: it projects h directly onto a small
//  ACTION space and reads the decision off the resulting distribution.
//
//    System-1 (this file)      one matvec  [A, E] x [E]  -> softmax -> argmax
//    System-2 (agent_loop)     max_new_tokens x (full RWKV forward + head)
//
//  Both heads read the SAME h. Neither replaces the other:
//    * high confidence  -> System-1 acts immediately (self-routing)
//    * low  confidence  -> escalate to System-2 for real reasoning / wording
//
//  COST. System-1 is A*E multiply-accumulates and one softmax over A. With
//  the shipped action space (7 actions) and a 0.1B-class E this is a few
//  hundred ns — sub-millisecond by three orders of magnitude. It performs no
//  token loop, allocates nothing after the first call, and does not touch the
//  recurrent state, so it cannot perturb a conversation.
//
//  HONESTY NOTE (read before trusting an output).
//  A freshly constructed head has NO trained projection. `init()` seeds it
//  deterministically so tests and CI are reproducible, and `provenance()` /
//  `trained()` report that fact. A seeded head produces well-formed but
//  MEANINGLESS actions — it is a documented placeholder for an offline-fitted
//  projection, exactly like the other not-yet-trained components in this
//  tree. Never present a `trained() == false` head's routing as a judgement.
//  `DecisionMode::Off` is the default everywhere precisely so that an
//  untrained head cannot silently change existing behaviour.
// =============================================================================
#pragma once

#include "omniseed/core/batch_gemm.h"
#include "omniseed/core/rwkv.h"
#include "omniseed/core/tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace omniseed {

// ---------------------------------------------------------------------------
// The System-1 action space.
//
// Deliberately tiny: every extra action costs E more floats of projection and
// dilutes the softmax. ABSTAIN and EXPLAIN are the two "this is not mine"
// escapes — ABSTAIN means "no opinion, say nothing", EXPLAIN means "I have an
// opinion but expressing it needs language, hand me to System-2". Both keep
// the head fail-closed: an uncertain head escalates rather than guessing.
// ---------------------------------------------------------------------------
enum class DecisionAction : int32_t {
    ABSTAIN = 0,   // no opinion — emit nothing
    HOLD,          // stay put, no new exposure
    BUY,           // open / add long
    SELL,          // open / add short
    CLOSE,         // flatten an existing position
    HEDGE,         // reduce net exposure without flattening
    EXPLAIN,       // needs language -> route to System-2
    COUNT          // sentinel: number of actions
};

// Stable lowercase-free label used in JSON and logs.
const char* decision_action_name(DecisionAction a);

// ---------------------------------------------------------------------------
// One System-1 decision.
//
// Field naming: the integration contract names these {action, confidence,
// asset, invalidation}; they are spelled out here as action_type /
// confidence_score / target_asset / invalidation so that "confidence" cannot
// be confused with a calibration claim and "asset" cannot be confused with a
// held position. The four semantics are exactly the contract's.
// ---------------------------------------------------------------------------
struct DecisionResult {
    // --- the four fields the integration contract names ----------------------
    DecisionAction action_type      = DecisionAction::ABSTAIN;
    float          confidence_score = 0.0f;   // softmax prob of the argmax
    std::string    target_asset;              // "" unless mapped via set_action_asset
    // The level at which this decision is VOID — for a trading action the price
    // at which the thesis is wrong and the position must be abandoned. 0 means
    // "not specified" and must be treated as *unknown*, never as "no stop".
    //
    // The head does not invent this number: a price level is market context,
    // not something recoverable from a hidden state. It is filled from the
    // per-action default (set_action_invalidation) and is expected to be
    // OVERRIDDEN by the caller from the risk engine, which is the only
    // component that knows the entry price and the stop distance.
    float          invalidation     = 0.0f;

    // --- diagnostics ---------------------------------------------------------
    float       margin    = 0.0f;   // p(top) - p(second): how close the call was
    std::string routing   = "abstain";  // "self" | "system2" | "abstain" | "error"
    bool        fast_path = false;  // true <=> routing == "self"
    int32_t     matvecs   = 0;      // 1 for a real decision; the whole point
    double      ms        = 0.0;    // wall time of the matvec+softmax

    std::string to_json() const;
};

// ---------------------------------------------------------------------------
// The head itself. Cheap to construct, stateless across calls, const-decide.
// ---------------------------------------------------------------------------
class DecisionHead {
public:
    struct Config {
        // Self-routing threshold. At or above this the head is trusted to act
        // without System-2. 0.85 matches the sniper engine's regime gate so
        // the two layers cannot disagree about what "confident" means.
        float threshold = 0.85f;
        // Optional floor on p(top) - p(second). A 0.90/0.88 split clears the
        // confidence bar but is nearly a coin flip; set > 0 to demand a clear
        // winner. 0 = disabled (confidence alone decides).
        float margin_floor = 0.0f;
        // Softmax temperature applied to the action logits before the softmax.
        // 1.0 = raw logits, i.e. NO calibration. Fitted OFFLINE by temperature
        // scaling on a held-out split: T > 1 softens an over-confident head so
        // that the reported confidence_score tracks the observed accuracy.
        //
        // This is the field that turns `confidence_score` from a number into a
        // claim. It is persisted in the .bin (format v3) so that a head loaded
        // from disk is calibrated, not merely fitted.
        float temperature = 1.0f;
    };

    DecisionHead() = default;
    // Convenience: size the head from the loaded model's embedding width.
    explicit DecisionHead(const RwkvConfig& cfg, uint32_t seed = 1234u);

    // Allocate [COUNT, E] projection + bias. Deterministic in `seed`; marks
    // the head untrained (see the honesty note at the top of this file).
    bool init(const RwkvConfig& cfg, uint32_t seed = 1234u);
    bool init(int32_t n_embd, uint32_t seed = 1234u);

    bool ready() const { return ready_; }
    const std::string& error() const { return error_; }

    int32_t hidden_size()  const { return E_; }
    int32_t action_count() const { return A_; }

    const Config& config() const { return cfg_; }
    void  set_threshold(float t) { cfg_.threshold = t; }
    float threshold() const { return cfg_.threshold; }
    // Optional floor on p(top) - p(second); 0 disables the check.
    void  set_margin_floor(float m) { cfg_.margin_floor = m; }
    float margin_floor() const { return cfg_.margin_floor; }

    // Softmax temperature. A non-finite or <= 0 value is normalised to 1.0
    // (no scaling) rather than stored, so a corrupt file cannot turn every
    // softmax into NaN — the head degrades to its uncalibrated self.
    void  set_temperature(float t);
    float temperature() const { return cfg_.temperature; }

    // ---- calibration report ------------------------------------------------
    // Filled by the OFFLINE trainer through set_calibration() (or read back
    // from a v3 .bin). `samples` is the size of the held-out calibration split;
    // `ece` is the Expected Calibration Error in [0, 1].
    //
    // A head that was never calibrated reports calibration_error() < 0, i.e.
    // "not measured". It deliberately does NOT report 0.0, because
    // "perfectly calibrated" and "never measured" are different claims and
    // only one of them would be true. Check calibrated() first.
    void    set_calibration(int32_t samples, float ece);
    int32_t calibration_samples() const { return calib_samples_; }
    float   calibration_error() const { return calib_error_; }   // ECE, or < 0
    bool    calibrated() const { return calib_samples_ > 0 && calib_error_ >= 0.0f; }

    // ---- the hot path -------------------------------------------------------
    // ONE matvec + softmax + argmax over the hidden state. No token loop, no
    // allocation after the first call, no mutation of `hidden` or the state.
    // Not thread-safe against itself (the logits scratch is reused); AgentLoop
    // is documented single-threaded per instance, same contract as
    // Uncertainty / EntropyMonitor.
    DecisionResult decide(const float* hidden) const;
    DecisionResult decide(const Tensor& hidden) const;

    // ---- the batched path ---------------------------------------------------
    // B decisions in one GEMM, so a backtest loop pays the per-call overhead
    // once instead of B times. The temperature, the max-subtracted softmax,
    // the argmax/runner-up scan and the self-routing rules are byte-for-byte
    // the same code path; with the Scalar kernel the result is BIT-IDENTICAL to
    // calling decide() B times, and the test asserts equality, not a tolerance.
    //
    // FAILS CLOSED: on a not-ready head, a null pointer, or a Tensor whose shape
    // is not [B, E], returns false and leaves `out` EMPTY — never a partial
    // batch, because a caller that ignores the return value must not act on
    // half an answer.
    //
    // `DecisionResult::ms` is the BATCH's per-row average, not that row's own
    // time: a per-row clock read is exactly the overhead the batch removes.
    // Per-batch totals are in `BatchStats`.
    bool decide_batch(const float* H, int32_t B, std::vector<DecisionResult>& out,
                      BatchStats* stats = nullptr) const;
    bool decide_batch(const Tensor& H, std::vector<DecisionResult>& out,
                      BatchStats* stats = nullptr) const;

    // Auto (the default) picks Simd when the CPU supports it.
    void set_batch_kernel(BatchKernel k) { batch_kernel_ = k; }
    BatchKernel batch_kernel() const { return batch_kernel_; }

    // ---- offline fitting hooks ---------------------------------------------
    // Overwrite one action's projection row (E floats) and bias. Counts as one
    // fitted action; trained() only becomes true once EVERY action is fitted,
    // so a half-populated head cannot masquerade as a real one.
    void set_action(DecisionAction a, const float* row, float bias);
    // Attach a default asset label for an action (fills target_asset).
    void set_action_asset(DecisionAction a, const std::string& asset);
    // Attach a default invalidation level for an action (fills invalidation).
    // This is a fallback, not a substitute for the risk engine: a real price
    // level depends on the entry, which only the caller knows.
    void set_action_invalidation(DecisionAction a, float level);

    // ---- persistence --------------------------------------------------------
    // Self-describing little-endian blob: magic, version, E, A, action names,
    // asset labels, projection, bias. Returns false + error() on mismatch.
    bool save(const std::string& path) const;
    bool load(const std::string& path);

    // ---- provenance ---------------------------------------------------------
    // trained() == every action row came from set_action()/load(), i.e. the
    // projection is fitted rather than seeded. Anything else is a placeholder.
    bool trained() const { return A_ > 0 && fitted_actions_ == A_; }
    int32_t fitted_actions() const { return fitted_actions_; }
    const std::string& provenance() const { return provenance_; }
    // Wall time (ns) of the most recent decide(); for benchmarks/dashboards.
    double last_ns() const { return last_ns_; }

private:
    void seed_weights(uint32_t seed);
    // Recomputes provenance_ from (ready_, trained(), fitted_, calib_,
    // loaded_). One place, so a later set_action()/set_calibration() cannot
    // leave a stale string behind claiming the head is uncalibrated.
    void refresh_provenance();
    // " + calibrated (n=..., ece=...%)", or "" when nothing was measured.
    std::string calibration_clause() const;

    Config      cfg_;
    int32_t     E_ = 0;             // hidden width
    int32_t     A_ = 0;             // action count
    std::vector<float>       proj_;  // [A_, E_] row-major, one row per action
    std::vector<float>       bias_;  // [A_]
    std::vector<std::string> asset_; // [A_] default target_asset per action
    std::vector<float>       inval_; // [A_] default invalidation level per action
    std::vector<std::string> name_;  // [A_] action labels (persisted)

    bool        ready_          = false;
    int32_t     fitted_actions_ = 0;   // rows supplied by set_action()/load()
    std::vector<uint8_t> fitted_;      // [A_] per-action "row came from fitting"
    // Calibration, fitted offline. samples <= 0 / ece < 0 means "not measured".
    int32_t     calib_samples_ = 0;
    float       calib_error_   = -1.0f;
    // True when the projection came from load(), so provenance() can say
    // "loaded ..." rather than "fitted ..." without guessing.
    bool        loaded_        = false;
    std::string provenance_ = "uninitialised";
    std::string error_;

    // Reused scratch — mutable so decide() can stay const and allocation-free.
    mutable std::vector<float> logits_;
    mutable double             last_ns_ = 0.0;
    // Batched scratch [B, A_] and the kernel choice, same reasoning as above.
    mutable std::vector<float> logits_batch_;
    BatchKernel                batch_kernel_ = BatchKernel::Auto;
};

} // namespace omniseed
