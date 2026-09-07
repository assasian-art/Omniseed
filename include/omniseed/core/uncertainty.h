// =============================================================================
//  OmniSeed — uncertainty.h
//  Uncertainty quantification over the output distribution (feature: deepseek
//  registry "不确定性量化", z.ai P-02 entropy anomaly / Q-08 uncertainty
//  navigation). Pure C++17, zero deps, O(V) per call, no allocation beyond
//  the caller's logits tensor.
//
//  Why it matters for a 0.1B edge kernel: the model cannot afford a second
//  "verifier" pass, but the logits it already computes carry a calibrated-
//  enough signal. High normalized entropy or a thin top-1/top-2 margin means
//  the kernel is guessing — the agent should say so (abstain / ask for
//  clarification) instead of hallucinating with confidence.
//
//  RAM cost: ~0 (a handful of scalars + a 64-slot ring in the monitor).
// =============================================================================
#pragma once

#include "omniseed/core/tensor.h"

#include <cstddef>
#include <vector>

namespace omniseed {

// ---------------------------------------------------------------------------
// One-shot analysis of a logits vector [n_vocab] fp32.
// ---------------------------------------------------------------------------
struct UncertaintyReport {
    float entropy            = 0.0f;   // Shannon entropy in nats
    float normalized_entropy = 0.0f;   // entropy / ln(n_vocab)  (0..1)
    float top1_prob          = 0.0f;
    float top2_prob          = 0.0f;
    float margin             = 0.0f;   // top1 - top2
    int32_t top1_id          = -1;
    int32_t top2_id          = -1;
};

class Uncertainty {
public:
    struct Config {
        // Abstain when the distribution is flat (model is guessing) OR the
        // top-2 race is a coin flip. Defaults tuned on RWKV-7 0.1B world
        // logits: greedy peaks are usually sharp (H > 5 nats is rare when
        // the model "knows").
        float entropy_thresh = 5.0f;   // nats
        float margin_thresh  = 0.05f;  // probability gap
    };

    // Numerically stable softmax statistics (max-subtracted).
    static UncertaintyReport analyze(const Tensor& logits);

    static bool should_abstain(const UncertaintyReport& r,
                               const Config& cfg = Config());
};

// ---------------------------------------------------------------------------
// Sliding-window entropy anomaly detector (z-score over the last N tokens).
// Flags "something changed" moments: topic shifts, tool-output injections,
// degenerate loops (entropy collapses) — the cheap cousin of full anomaly
// detection pipelines, at 64 floats of RAM.
// ---------------------------------------------------------------------------
class EntropyMonitor {
public:
    explicit EntropyMonitor(size_t window = 64);

    // Push one token's entropy; returns true when the NEW sample is an
    // anomaly relative to the window (needs >= 8 samples of history).
    bool push(float entropy);

    bool  is_anomaly() const { return anomaly_; }
    float z_score() const;
    float mean() const;
    float stddev() const;
    size_t samples() const { return buf_.size(); }
    float last() const { return last_; }

private:
    size_t cap_;
    std::vector<float> buf_;     // ring in insertion order (small N)
    float last_ = 0.0f;
    bool  anomaly_ = false;
};

} // namespace omniseed
