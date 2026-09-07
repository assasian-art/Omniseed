// =============================================================================
//  OmniSeed — uncertainty.cpp
//  See uncertainty.h. Numerically stable; single pass over the vocab.
// =============================================================================
#include "omniseed/core/uncertainty.h"

#include <algorithm>
#include <cmath>

namespace omniseed {

UncertaintyReport Uncertainty::analyze(const Tensor& logits) {
    UncertaintyReport r;
    const float* lp = logits.f32();
    const int64_t n = logits.numel();
    if (lp == nullptr || n <= 0) return r;

    // max for numerical stability + top-2 tracking in the same pass.
    float maxv = lp[0];
    int32_t top1 = 0, top2 = -1;
    float top1v = maxv, top2v = -1e30f;
    for (int64_t i = 1; i < n; ++i) {
        const float v = lp[i];
        if (v > top1v) { top2v = top1v; top2 = top1; top1v = v; top1 = static_cast<int32_t>(i); }
        else if (v > top2v) { top2v = v; top2 = static_cast<int32_t>(i); }
        if (v > maxv) maxv = v;
    }
    r.top1_id = top1;
    r.top2_id = top2;

    // Softmax with max subtraction; accumulate sum of exp and sum of x*log(x).
    double sum = 0.0, h = 0.0;
    const float inv_t = 1.0f;   // temperature 1: analyze the raw distribution
    for (int64_t i = 0; i < n; ++i) {
        const double e = std::exp(static_cast<double>(lp[i] - maxv) * inv_t);
        sum += e;
        h   -= e * std::log(e > 0.0 ? e : 1e-300);
    }
    // Entropy of the unnormalized distribution: H = log(sum) - (sum x log x)/sum
    const double log_sum = std::log(sum > 0.0 ? sum : 1e-300);
    r.entropy = static_cast<float>(log_sum - h / sum);
    if (r.entropy < 0.0f) r.entropy = 0.0f;
    r.normalized_entropy = (n > 1) ? r.entropy / std::log(static_cast<double>(n)) : 0.0f;

    // Top-2 probabilities from the same shifted exps.
    const double p1 = std::exp(static_cast<double>(top1v - maxv)) / sum;
    const double p2 = (top2 >= 0) ? std::exp(static_cast<double>(top2v - maxv)) / sum : 0.0;
    r.top1_prob = static_cast<float>(p1);
    r.top2_prob = static_cast<float>(p2);
    r.margin    = static_cast<float>(p1 - p2);
    return r;
}

bool Uncertainty::should_abstain(const UncertaintyReport& r, const Config& cfg) {
    // Flat distribution (no peak at all) or a coin-flip top-2: the kernel is
    // guessing. Greedy still returns top-1, but the agent should know.
    return r.normalized_entropy > cfg.entropy_thresh ||
           r.margin < cfg.margin_thresh;
}

EntropyMonitor::EntropyMonitor(size_t window) : cap_(window < 8 ? 8 : window) {
    buf_.reserve(cap_);
}

bool EntropyMonitor::push(float entropy) {
    last_ = entropy;
    buf_.push_back(entropy);
    if (buf_.size() > cap_) buf_.erase(buf_.begin());   // small N: shift is fine

    if (buf_.size() < 8) { anomaly_ = false; return false; }

    double mean = 0.0;
    for (float v : buf_) mean += v;
    mean /= static_cast<double>(buf_.size());

    double var = 0.0;
    for (float v : buf_) { const double d = v - mean; var += d * d; }
    var /= static_cast<double>(buf_.size());
    const double sd = std::sqrt(var > 0.0 ? var : 0.0);

    // Anomaly: |z| > 3 against the window (loop collapse / topic shift).
    anomaly_ = sd > 1e-6 &&
        std::fabs(static_cast<double>(last_) - mean) > 3.0 * sd;
    return anomaly_;
}

float EntropyMonitor::z_score() const {
    if (buf_.size() < 2) return 0.0f;
    double mean = 0.0;
    for (float v : buf_) mean += v;
    mean /= static_cast<double>(buf_.size());
    double var = 0.0;
    for (float v : buf_) { const double d = v - mean; var += d * d; }
    var /= static_cast<double>(buf_.size());
    const double sd = std::sqrt(var);
    return sd > 1e-6 ? static_cast<float>((last_ - mean) / sd) : 0.0f;
}

float EntropyMonitor::mean() const {
    if (buf_.empty()) return 0.0f;
    double m = 0.0;
    for (float v : buf_) m += v;
    return static_cast<float>(m / static_cast<double>(buf_.size()));
}

float EntropyMonitor::stddev() const {
    if (buf_.size() < 2) return 0.0f;
    double mean = 0.0;
    for (float v : buf_) mean += v;
    mean /= static_cast<double>(buf_.size());
    double var = 0.0;
    for (float v : buf_) { const double d = v - mean; var += d * d; }
    var /= static_cast<double>(buf_.size());
    return static_cast<float>(std::sqrt(var));
}

} // namespace omniseed
