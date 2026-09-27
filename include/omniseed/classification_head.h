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

    // ---- provenance ---------------------------------------------------------
    bool trained() const;
    int32_t fitted_rows() const { return fitted_rows_; }
    const std::string& provenance() const { return provenance_; }
    double last_us() const { return last_us_; }

private:
    void seed_weights(uint32_t seed);
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
