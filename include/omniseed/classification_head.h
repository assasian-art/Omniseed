// =============================================================================
//  OmniSeed — classification_head.h
//
//  A multi-class readout of h[E]: [L, E] x [E] -> [L] -> softmax -> top-k.
//
//  One head, MANY label sets. The projection is a single [total_labels, E]
//  matrix and each registered label set owns a contiguous slice of it, so a new
//  domain costs its labels' rows and nothing else. That is what lets one
//  backbone answer "what regime is this?" and "what language is this?" without
//  a second model, and it is why the label sets are registered by NAME: the
//  wire contract says `{"domain":"language.intent", ...}` and a caller that
//  misspells it must get an error, not a silently different distribution.
//
//  TOP-K, NOT ARGMAX. A 0.51/0.49 split has a winner and no information. The
//  head returns the k best with their probabilities AND the margin
//  p(top) - p(second), so a caller can see that the decision was a coin flip.
//
//  COST. One [L, E] matvec — for a 7-label set at E = 768 that is 5,376 MACs,
//  a few microseconds. Allocation-free after init(); h is never mutated.
//
//  HONESTY NOTE. A freshly constructed head has NO trained projection: init()
//  seeds it deterministically so tests are reproducible, and `trained()` /
//  `provenance()` report that. A seeded head emits well-formed but MEANINGLESS
//  label distributions. Never present one as a judgement.
// =============================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "omniseed/core/rwkv.h"
#include "omniseed/core/tensor.h"
#include "omniseed/heads.h"

namespace omniseed {

class ClassificationHead {
public:
    struct LabelSet {
        std::string              name;     // e.g. "language.intent"
        std::vector<std::string> labels;   // e.g. {"question", "statement", ...}
        int32_t                  offset = 0;  // first row in proj_
    };

    ClassificationHead() = default;
    explicit ClassificationHead(const RwkvConfig& cfg, uint32_t seed = 777u);

    bool init(const RwkvConfig& cfg, uint32_t seed = 777u);
    bool init(int32_t n_embd, uint32_t seed = 777u);

    bool ready() const { return ready_; }
    const std::string& error() const { return error_; }
    int32_t hidden_size() const { return E_; }

    // ---- label sets ---------------------------------------------------------
    // Registers a set and allocates its projection rows. Returns the set index,
    // or -1 on a duplicate name, an empty label list, or a head that is not
    // ready. Duplicate names are REFUSED rather than overwritten: two callers
    // registering "language.intent" with different labels is a bug that should
    // surface at startup, not as a silently different distribution at runtime.
    int32_t add_label_set(const std::string& name, const std::vector<std::string>& labels);

    int32_t label_set_count() const { return static_cast<int32_t>(sets_.size()); }
    const LabelSet& label_set(int32_t i) const { return sets_[static_cast<size_t>(i)]; }
    // Index of a registered set, or -1.
    int32_t find_label_set(const std::string& name) const;
    int32_t total_labels() const { return total_labels_; }

    // ---- the hot path -------------------------------------------------------
    // Softmax over one label set, reduced to the top `top_k`. Allocation-free
    // after the first call (the scratch is grown once); not thread-safe against
    // itself.
    //
    // FAILURE IS EXPLICIT: on an unknown set, a head that is not ready, or a
    // hidden-state width mismatch, the result carries the reason in `domain`
    // as "<error: ...>" and leaves `top_k` EMPTY. No registered set name can
    // start with '<', so an empty top_k is unambiguous. A fabricated uniform
    // distribution would be indistinguishable from a real — and useless — one.
    ClassificationResult classify(const float* hidden, const std::string& set_name,
                                  int32_t top_k = 3) const;
    ClassificationResult classify(const float* hidden, int32_t set_index,
                                  int32_t top_k = 3) const;
    ClassificationResult classify(const Tensor& hidden, const std::string& set_name,
                                  int32_t top_k = 3) const;

    // ---- offline fitting hooks ---------------------------------------------
    void set_label_row(int32_t set_index, int32_t label_index, const float* row, float bias);

    // ---- calibration --------------------------------------------------------
    // Softmax temperature applied to the label logits before the softmax.
    // 1.0 = raw logits (no calibration). A non-finite or <= 0 value is
    // normalised to 1.0 rather than stored, so a corrupt blob cannot make
    // every distribution NaN.
    //
    // PER SET, not per head. Each label set has its own logit scale — a 7-way
    // intent set and a 3-way sentiment set fitted on the same h[E] routinely
    // need temperatures an order of magnitude apart — and a single scalar
    // fitted on the pooled logits measurably makes the better-scaled sets
    // WORSE. Measured on the first fit: a pooled T took language.language's
    // ECE from 0.172 to 0.194 while fixing language.intent. One number cannot
    // serve both, so the head stores one number per set.
    //
    // set_temperature(t) applies t to every set AND to sets registered later;
    // set_temperature(set_index, t) calibrates one set. temperature() reports
    // the default, temperature(set_index) the set's own value.
    void  set_temperature(float t);
    void  set_temperature(int32_t set_index, float t);
    float temperature() const { return default_temperature_; }
    float temperature(int32_t set_index) const;

    // Fitted offline by temperature scaling on a held-out split. A head that
    // was never calibrated reports calibration_error() < 0 ("not measured"),
    // never 0.0 — see the same note on DecisionHead.
    //
    // The no-index form is the POOLED value: an n-weighted average over the
    // sets that were measured, which is the right summary for "how calibrated
    // is this head overall". The indexed form is that set's own ECE.
    void    set_calibration(int32_t samples, float ece);
    void    set_calibration(int32_t set_index, int32_t samples, float ece);
    int32_t calibration_samples() const;
    int32_t calibration_samples(int32_t set_index) const;
    float   calibration_error() const;                  // pooled, or < 0
    float   calibration_error(int32_t set_index) const; // per set, or < 0
    bool    calibrated() const;

    // ---- persistence --------------------------------------------------------
    // Self-describing little-endian blob: magic, version, E, the label sets
    // (name + labels, in registration order), the projection, the bias, the
    // per-row fitted flags, and the calibration trailer.
    //
    // The label sets travel WITH the weights because a [L, E] matrix whose
    // label order is not recorded is not restorable: the same numbers under a
    // permuted label list are a different classifier. load() therefore
    // REBUILDS the sets from the file in file order rather than trusting
    // whatever the caller had registered, so a restored head is exactly the
    // head that was saved.
    //
    // On failure the head is left NOT READY (fail closed): a caller that
    // ignores the return value must not end up classifying with a half-read
    // projection.
    bool save(const std::string& path) const;
    bool load(const std::string& path);

    // ---- provenance ---------------------------------------------------------
    bool trained() const;
    int32_t fitted_rows() const { return fitted_rows_; }
    const std::string& provenance() const { return provenance_; }
    double last_us() const { return last_us_; }

private:
    void seed_weights(uint32_t seed);
    // Recomputes provenance_ from (ready_, sets_, fitted_rows_, calibration).
    // One composer, so a later set_label_row()/set_calibration() cannot leave
    // a stale string behind.
    void refresh_provenance();
    // " — calibrated (n=..., ece=...%)", or "" when nothing was measured.
    std::string calibration_clause() const;
    // How many label sets carry a real (samples > 0, ece >= 0) measurement.
    int32_t measured_sets() const;
    // Builds a result that carries the failure in `domain` and leaves top_k
    // empty. Every failure path goes through here so none of them can invent a
    // distribution.
    ClassificationResult failure(const std::string& set_name, const char* why) const;

    int32_t E_ = 0;
    uint32_t seed_ = 777u;   // kept so a later add_label_set() can seed its rows

    std::vector<LabelSet> sets_;
    std::vector<float>    proj_;   // [total_labels, E], row-major
    std::vector<float>    bias_;   // [total_labels]
    std::vector<uint8_t>  fitted_; // [total_labels]
    int32_t               total_labels_ = 0;
    int32_t               fitted_rows_  = 0;

    // Calibration, fitted offline. One entry per label set, grown by
    // add_label_set(). samples <= 0 / ece < 0 means "not measured".
    float                default_temperature_ = 1.0f;
    std::vector<float>   temps_;          // [sets_] softmax temperature
    std::vector<int32_t> calib_samples_;  // [sets_] calibration-set size
    std::vector<float>   calib_error_;    // [sets_] ECE, or -1
    // True when the weights came from load(), so provenance() can say
    // "loaded ..." rather than "fitted ..." without guessing.
    bool    loaded_        = false;

    bool        ready_      = false;
    std::string provenance_ = "uninitialised";
    std::string error_;

    mutable std::vector<float> scratch_;
    mutable double             last_us_ = 0.0;
};

// ---------------------------------------------------------------------------
// The label sets the mandate names, in one place.
//
// Registering them is idempotent per name (a duplicate is refused), so a caller
// that adds its own set and then calls this keeps both.
// ---------------------------------------------------------------------------
struct DomainLabelSetSpec {
    const char*              name;
    std::vector<const char*> labels;
};

const std::vector<DomainLabelSetSpec>& default_domain_label_sets();
// Registers every default set. Returns how many were newly added.
int32_t add_default_label_sets(ClassificationHead& head);

} // namespace omniseed
