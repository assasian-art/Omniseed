// =============================================================================
//  OmniSeed — uncertainty_split.h
//  "WHY am I unsure?" — separating the two kinds of doubt.
//
//  core/uncertainty.h answers "how unsure am I?" with ONE number (normalised
//  entropy, or a top1-top2 margin). That number cannot say WHY, and the two
//  reasons call for OPPOSITE responses:
//
//    ALEATORIC doubt  — the classes genuinely overlap on this input. A perfect
//                       model would still be unsure. More data will NOT help.
//                       The right response is to stay out: there is no edge.
//
//    EPISTEMIC doubt  — the input lies outside the region the head was fitted
//                       on. The head is extrapolating. More data WOULD help.
//                       The right response is to go and learn, not to trade.
//
//  Measured on the real held-out heads (§35, tools/uncertainty_audit.py):
//    * aleatoric (calibrated normalised entropy) is MONOTONE in the observed
//      error rate for 4 of the 5 fitted heads — trading.regime 0.120 -> 0.452
//      across entropy quartiles, language.sentiment 0.059 -> 0.500,
//      language.language 0.000 -> 0.350, language.intent 0.235 -> 0.600.
//      The exception is the DecisionAction head, whose accuracy is 0.244 —
//      near chance, so nothing can predict its errors and the honest report is
//      "no signal" (the probe prints it as such).
//    * epistemic (h-space distance) does NOT predict error WITHIN a domain —
//      and that is the CORRECT result, not a defect: inside one domain there
//      is no epistemic uncertainty to find. The quartile error rates are
//      non-monotone and sometimes inverted.
//    * epistemic DOES fire on genuinely out-of-distribution input. Fitting the
//      reference on the trading h[E] and scoring the language h[E] separates
//      them completely: in-domain distance ~0.99, out-of-domain ~6.70, and
//      ZERO of the 71 out-of-domain vectors fall below the in-domain 90th
//      percentile. The reverse direction is 0/369.
//
//  So the two signals are orthogonal by construction and by measurement, which
//  is exactly what the textbook decomposition says they should be.
//
//  WHAT THIS IS NOT. This is a PROXY decomposition, not the Bayesian one. The
//  rigorous operator (Depeweg et al. 2018) needs a distribution over model
//  PARAMETERS q(theta):
//      total     = H( E_theta[ p(y|x,theta) ] )
//      aleatoric = E_theta[ H( p(y|x,theta) ) ]
//      epistemic = total - aleatoric = I(y; theta | x)  >= 0 by Jensen
//  from_ensemble() below implements that EXACT operator, and it is tested —
//  but nothing in this tree can currently FEED it, because no ensemble of
//  heads has been fitted. The joint is missing, not the mathematics. Until it
//  exists, split() uses the two measured proxies above, and says so.
//
//  COST. fit() is O(n*E) once. distance() is one E-pass (subtract, divide,
//  square, accumulate). epistemic() adds a binary search over n_fit floats.
//  RAM: 3*E floats for the reference plus n_fit floats for the distance CDF.
//  At E = 768 and n_fit = 369 that is ~11 KB. No allocation on the hot path.
// =============================================================================
#pragma once

#include "omniseed/core/tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace omniseed {

// ---------------------------------------------------------------------------
// Which source of doubt dominates for one input.
// `None` is 0 so an unset/zero-initialised struct means "nothing wrong".
// ---------------------------------------------------------------------------
enum class DoubtSource : int32_t {
    None = 0,
    Aleatoric,   // classes overlap; more data will not help
    Epistemic,   // input outside the fitted reference; more data would help
    Both,
    COUNT
};

constexpr int32_t kDoubtSourceCount = static_cast<int32_t>(DoubtSource::COUNT);
const char* doubt_source_name(DoubtSource d);

// ---------------------------------------------------------------------------
// The split for one (hidden, distribution) pair.
//
// The two fields are deliberately NOT summed into one score. They are measured
// on different scales (a normalised entropy in [0,1] vs an empirical CDF in
// [0,1]) and they mean different things, so a single number would be an
// invented aggregate — the same reason the heads return a top-2 margin instead
// of pretending a 0.51/0.49 split is a decision.
// ---------------------------------------------------------------------------
struct UncertaintySplit {
    float       aleatoric = 0.0f;   // normalised entropy of the CALIBRATED p, [0,1]
    float       epistemic = 0.0f;   // P_ref(d_ref <= d(h)), [0,1]
    DoubtSource dominant  = DoubtSource::None;
    bool        recommend_abstain = false;   // either signal crossed its threshold
    std::string reason;             // one line, for a log or a JSON document

    std::string to_json() const;
};

// ---------------------------------------------------------------------------
// The decomposition itself.
// ---------------------------------------------------------------------------
class UncertaintyDecomposition {
public:
    struct Config {
        // Normalised entropy at or above which the classes are treated as
        // genuinely overlapping. 0.5 means "more than half of the way from a
        // point mass to a uniform distribution".
        float aleatoric_thresh = 0.5f;
        // Empirical CDF of the h-distance at or above which the input is
        // treated as out of reference. 0.90 => farther from the reference mean
        // than 90% of the fitted inputs. In-domain inputs sit near 0.5, so this
        // only fires on the far tail — which is the point: within a homogeneous
        // domain the top decile is noise, and the signal is the cross-domain
        // jump to 1.0.
        float epistemic_thresh = 0.90f;
        // Upper bound on how many fitted distances are retained in the CDF.
        // Above this the fit set is strided uniformly. 65536 floats = 256 KB.
        int32_t max_fit = 65536;
    };

    UncertaintyDecomposition() = default;

    // Build the h[E] reference from a row-major [n, E] block: the mean, the
    // per-dimension standard deviation, and the sorted diagonal-Mahalanobis
    // distances of the fit rows. Refuses n < 2 (a standard deviation from one
    // sample is not a measurement), E < 1, and any non-finite input.
    //
    // A dimension whose sigma is unmeasurably small is floored at 10% of the
    // TYPICAL sigma rather than at an absolute epsilon: h[E] is post-layernorm,
    // so some dimensions barely move, and dividing by a 1e-6 sigma would let
    // one dead dimension dominate the whole distance. The floor bounds that
    // dimension's influence at 10x instead of unbounded.
    bool fit(const float* h_rows, int32_t n, int32_t E);
    bool fit(const Tensor& h_rows);

    bool fitted() const { return fitted_; }
    int32_t n_fit() const { return n_fit_; }
    int32_t dim() const { return E_; }
    const std::string& error() const { return error_; }

    const Config& config() const { return cfg_; }
    void set_config(const Config& c) { cfg_ = c; }

    // ---- the reference (exposed for tests and for reporting) ----------------
    const std::vector<float>& mean() const { return mu_; }
    const std::vector<float>& sigma() const { return sigma_; }
    // Ascending distances of the (possibly strided) fit rows.
    const std::vector<float>& reference_distances() const { return sorted_dist_; }

    // ---- the two signals ----------------------------------------------------
    // Normalised diagonal-Mahalanobis distance from the reference mean:
    //   sqrt( mean_e( ((h_e - mu_e) / sigma_e)^2 ) )
    // so an in-distribution vector scores ~1.0 regardless of E.
    //
    // Returns -1.0 when no reference has been fitted OR when h is not finite:
    // a distance is a MEASUREMENT, and "not measured" is not zero.
    float distance(const float* h) const;
    float distance(const Tensor& h) const;

    // The empirical CDF of the reference distances at d(h): the fraction of the
    // fit set that is no farther from the mean than this input. ~0.5 for a
    // typical in-distribution input, -> 1.0 for out-of-distribution input.
    //
    // FAILS CLOSED to 1.0 when no reference is fitted or h is not finite: a
    // reference that does not exist cannot vouch for ANY input, and reporting
    // "familiar" would be the one wrong answer.
    float epistemic(const float* h) const;
    float epistemic(const Tensor& h) const;

    // Normalised entropy H(p)/ln(K) of a probability distribution, clamped to
    // [0,1]. 0 for a point mass, 1 for uniform, 0 for K <= 1 (a one-way choice
    // carries no uncertainty).
    //
    // A vector that is not a distribution (non-finite, negative, or summing to
    // <= 0) returns 1.0 rather than 0.0 — the fail-closed direction, so a
    // corrupt input cannot masquerade as certainty.
    static float aleatoric(const float* probs, int32_t K);

    // Both signals plus the dominance call. `probs` must be the head's
    // CALIBRATED softmax output (temperature already applied): applying a
    // fitted T is what makes the entropy a claim rather than a raw number.
    UncertaintySplit split(const float* h, const float* probs, int32_t K) const;
    UncertaintySplit split(const Tensor& h, const float* probs, int32_t K) const;

    // ---- the rigorous operator, for a caller that HAS an ensemble -----------
    // Depeweg et al. 2018, in nats (NOT normalised):
    //   total     = H(mean_s p_s)
    //   aleatoric = mean_s H(p_s)
    //   epistemic = total - aleatoric          (>= 0 by Jensen)
    // Each member is normalised defensively first. An empty ensemble, a
    // zero-length member, or ragged members yield all-zero output.
    struct EnsembleSplit {
        float total     = 0.0f;
        float aleatoric = 0.0f;
        float epistemic = 0.0f;
    };
    static EnsembleSplit from_ensemble(const std::vector<std::vector<float>>& members);

    // ---- persistence --------------------------------------------------------
    // Self-describing little-endian blob: magic, version, E, n_fit, the two
    // thresholds, the mean, the sigma, then the sorted reference distances.
    // load() leaves the object UNFITTED on any failure (fail closed), so a
    // caller that ignores the return value cannot score against a half-read
    // reference.
    bool save(const std::string& path) const;
    bool load(const std::string& path);

private:
    Config  cfg_;
    int32_t E_     = 0;
    int32_t n_fit_ = 0;
    bool    fitted_ = false;

    std::vector<float> mu_;          // [E_]
    std::vector<float> sigma_;       // [E_], floored, > 0
    std::vector<float> sorted_dist_; // [n_fit_], ascending

    std::string error_ = "unfitted";
};

} // namespace omniseed
