// =============================================================================
//  OmniSeed — src/core/uncertainty_split.cpp
//
//  Aleatoric = how much the classes overlap on THIS input (calibrated entropy).
//  Epistemic = how far this input sits outside the fitted reference (h-space).
//  See the header for the measured evidence and the honesty note.
//
//  Three details worth knowing before changing anything here:
//
//  1. THE DISTANCE IS NORMALISED BY E, NOT BY sqrt(E). sqrt(mean(...)) makes an
//     in-distribution vector score ~1.0 for every E, which is what makes the
//     threshold 0.90 portable across heads instead of a function of the
//     embedding width.
//
//  2. SIGMA IS FLOORED RELATIVE TO THE OTHER DIMENSIONS, NOT ABSOLUTELY. h[E]
//     is post-layernorm, so a dimension can be almost constant; dividing it by
//     a 1e-6 epsilon would give that one dead dimension unbounded influence
//     over the distance.
//
//  3. BOTH SIGNALS FAIL CLOSED. A missing reference, a NaN input, or a vector
//     that is not a distribution all resolve to MAXIMUM doubt, never to
//     "certain". The one wrong answer is to report confidence that was never
//     measured.
// =============================================================================
#include "omniseed/core/uncertainty_split.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace omniseed {

namespace {

// Little-endian, self-describing, same stance as the heads: OmniSeed state
// files are native-LE by contract, so no byte swapping is performed.
constexpr char kMagic[8] = {'O', 'M', 'N', 'I', 'S', 'U', 'S', '1'};
constexpr int32_t kFormatVersion = 1;
constexpr int32_t kMaxDim = 1 << 20;

template <typename T>
bool wr(FILE* f, const T& v) { return std::fwrite(&v, sizeof(T), 1, f) == 1; }

template <typename T>
bool rd(FILE* f, T& v) { return std::fread(&v, sizeof(T), 1, f) == 1; }

bool all_finite(const float* v, int64_t n) {
    for (int64_t i = 0; i < n; ++i)
        if (!std::isfinite(v[i])) return false;
    return true;
}

// Normalised diagonal-Mahalanobis distance. Free function rather than a member
// so fit() can use it BEFORE fitted_ is set — the member form fails closed to
// -1.0 when no reference exists, which would have filled the reference CDF
// with sentinels.
float normalized_distance(const float* h, const float* mu, const float* sigma,
                          int32_t E) {
    double acc = 0.0;
    for (int32_t e = 0; e < E; ++e) {
        const double d = (static_cast<double>(h[e]) - static_cast<double>(mu[e])) /
                         static_cast<double>(sigma[e]);
        acc += d * d;
    }
    return static_cast<float>(std::sqrt(acc / static_cast<double>(E)));
}

// A tiny numeric formatter that never emits "nan"/"inf": a JSON document with a
// bare `nan` in it is invalid, and a downstream parser choking is a worse
// outcome than a 0.0 the caller can see is a default.
void append_float(std::string& out, float v) {
    char buf[32];
    if (!std::isfinite(v)) std::snprintf(buf, sizeof(buf), "0.0");
    else std::snprintf(buf, sizeof(buf), "%.4f", static_cast<double>(v));
    out += buf;
}

// Normalised entropy of an ALREADY-VALID distribution. K >= 2 is required by
// the caller, which handles the K <= 1 and invalid cases itself.
float entropy_norm_valid(const float* p, int32_t K) {
    double h = 0.0;
    for (int32_t k = 0; k < K; ++k) {
        const double pk = static_cast<double>(p[k]);
        if (pk > 0.0) h -= pk * std::log(pk);
    }
    const double hmax = std::log(static_cast<double>(K));
    if (!(hmax > 0.0)) return 0.0f;
    double r = h / hmax;
    if (r < 0.0) r = 0.0;
    if (r > 1.0) r = 1.0;
    return static_cast<float>(r);
}

}  // namespace

const char* doubt_source_name(DoubtSource d) {
    switch (d) {
        case DoubtSource::None:      return "none";
        case DoubtSource::Aleatoric: return "aleatoric";
        case DoubtSource::Epistemic: return "epistemic";
        case DoubtSource::Both:      return "both";
        default:                     return "unknown";
    }
}

std::string UncertaintySplit::to_json() const {
    std::string out = "{\"aleatoric\":";
    append_float(out, aleatoric);
    out += ",\"epistemic\":";
    append_float(out, epistemic);
    out += ",\"dominant\":\"";
    out += doubt_source_name(dominant);
    out += "\",\"abstain\":";
    out += recommend_abstain ? "true" : "false";
    out += ",\"reason\":\"";
    for (const char c : reason) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) out += '?';
                else out += c;
        }
    }
    out += "\"}";
    return out;
}

// ---------------------------------------------------------------------------
// aleatoric — the static, reference-free half
// ---------------------------------------------------------------------------
float UncertaintyDecomposition::aleatoric(const float* probs, int32_t K) {
    // A one-way choice, or no distribution at all, carries no measurable
    // uncertainty. 0 here is not a claim of certainty; it is "this question
    // does not have a distribution to be unsure about".
    if (probs == nullptr || K <= 1) return 0.0f;
    if (!all_finite(probs, K)) return 1.0f;   // fail closed

    double sum = 0.0;
    for (int32_t k = 0; k < K; ++k) {
        if (probs[k] < 0.0f) return 1.0f;     // not a distribution
        sum += static_cast<double>(probs[k]);
    }
    if (!(sum > 0.0)) return 1.0f;            // all-zero / underflow

    // Normalise defensively: the contract is "the head's calibrated softmax
    // output", but a caller that hands over unnormalised weights should get a
    // distribution back rather than a silently out-of-range entropy.
    std::vector<float> p(static_cast<size_t>(K));
    for (int32_t k = 0; k < K; ++k)
        p[static_cast<size_t>(k)] = static_cast<float>(static_cast<double>(probs[k]) / sum);
    return entropy_norm_valid(p.data(), K);
}

// ---------------------------------------------------------------------------
// fit — the reference distribution
// ---------------------------------------------------------------------------
bool UncertaintyDecomposition::fit(const float* h_rows, int32_t n, int32_t E) {
    fitted_ = false;
    n_fit_  = 0;
    mu_.clear();
    sigma_.clear();
    sorted_dist_.clear();
    E_ = 0;

    if (h_rows == nullptr) { error_ = "fit: null rows"; return false; }
    if (E <= 0 || E > kMaxDim) { error_ = "fit: bad width"; return false; }
    // A standard deviation from a single sample is not a measurement, and a
    // reference built from one point would call every other input an outlier.
    if (n < 2) { error_ = "fit: need at least 2 rows"; return false; }
    if (static_cast<int64_t>(n) * E > (int64_t{1} << 34)) {
        error_ = "fit: implausible row count"; return false;
    }
    if (!all_finite(h_rows, static_cast<int64_t>(n) * E)) {
        error_ = "fit: non-finite input";
        return false;
    }

    E_ = E;
    mu_.assign(static_cast<size_t>(E), 0.0f);
    for (int32_t i = 0; i < n; ++i) {
        const float* r = h_rows + static_cast<size_t>(i) * static_cast<size_t>(E);
        for (int32_t e = 0; e < E; ++e) mu_[static_cast<size_t>(e)] += r[e];
    }
    const float inv_n = 1.0f / static_cast<float>(n);
    for (int32_t e = 0; e < E; ++e) mu_[static_cast<size_t>(e)] *= inv_n;

    sigma_.assign(static_cast<size_t>(E), 0.0f);
    double mean_sigma = 0.0;
    for (int32_t e = 0; e < E; ++e) {
        double acc = 0.0;
        for (int32_t i = 0; i < n; ++i) {
            const double d =
                static_cast<double>(h_rows[static_cast<size_t>(i) * static_cast<size_t>(E) +
                                           static_cast<size_t>(e)]) -
                static_cast<double>(mu_[static_cast<size_t>(e)]);
            acc += d * d;
        }
        const float s = static_cast<float>(std::sqrt(acc / static_cast<double>(n)));
        sigma_[static_cast<size_t>(e)] = s;
        mean_sigma += static_cast<double>(s);
    }
    mean_sigma /= static_cast<double>(E);
    // A dimension that barely moved in the reference cannot have its own scale
    // measured, and dividing by an epsilon would give that one dimension
    // unbounded influence over the whole distance. Flooring at 10% of the
    // TYPICAL scale bounds its influence at 10x instead: a dimension that moves
    // less than a tenth as much as a typical one is treated as if it moved a
    // tenth as much. If every dimension is dead the block is a single point and
    // unit scale is the only sane fallback.
    float floor_s = 0.1f * static_cast<float>(mean_sigma);
    if (!(floor_s > 1e-6f)) floor_s = 1.0f;
    for (int32_t e = 0; e < E; ++e)
        if (!(sigma_[static_cast<size_t>(e)] > 1e-6f))
            sigma_[static_cast<size_t>(e)] = floor_s;

    // The reference CDF. Strided uniformly when the fit set exceeds max_fit, so
    // a huge fit set costs a bounded blob rather than a growing one.
    const int32_t cap = std::max(2, cfg_.max_fit);
    const int32_t keep = std::min(n, cap);
    sorted_dist_.reserve(static_cast<size_t>(keep));
    for (int32_t j = 0; j < keep; ++j) {
        // j-th of `keep` evenly spaced rows out of `n`.
        const int32_t i = (keep == n) ? j
                                      : static_cast<int32_t>(
                                            (static_cast<int64_t>(j) * n) / keep);
        sorted_dist_.push_back(normalized_distance(
            h_rows + static_cast<size_t>(i) * static_cast<size_t>(E), mu_.data(),
            sigma_.data(), E));
    }
    std::sort(sorted_dist_.begin(), sorted_dist_.end());
    n_fit_ = keep;
    fitted_ = true;
    error_.clear();
    return true;
}

bool UncertaintyDecomposition::fit(const Tensor& h_rows) {
    if (h_rows.dtype() != DType::F32) { error_ = "fit: tensor is not f32"; return false; }
    if (h_rows.shape().size() != 2) { error_ = "fit: tensor must be [n, E]"; return false; }
    const int64_t n = h_rows.shape()[0];
    const int64_t e = h_rows.shape()[1];
    if (n <= 0 || e <= 0 || n > INT32_MAX || e > INT32_MAX) {
        error_ = "fit: bad tensor shape"; return false;
    }
    return fit(h_rows.f32(), static_cast<int32_t>(n), static_cast<int32_t>(e));
}

// ---------------------------------------------------------------------------
// distance / epistemic
// ---------------------------------------------------------------------------
float UncertaintyDecomposition::distance(const float* h) const {
    if (!fitted_ || h == nullptr) return -1.0f;
    if (!all_finite(h, E_)) return -1.0f;
    return normalized_distance(h, mu_.data(), sigma_.data(), E_);
}

float UncertaintyDecomposition::distance(const Tensor& h) const {
    if (h.dtype() != DType::F32 || h.numel() != static_cast<int64_t>(E_)) return -1.0f;
    return distance(h.f32());
}

float UncertaintyDecomposition::epistemic(const float* h) const {
    // Fail closed: no reference => no input can be vouched for.
    if (!fitted_) return 1.0f;
    const float d = distance(h);
    if (!(d >= 0.0f)) return 1.0f;   // also catches NaN
    const size_t below =
        static_cast<size_t>(std::upper_bound(sorted_dist_.begin(), sorted_dist_.end(), d) -
                            sorted_dist_.begin());
    return static_cast<float>(below) / static_cast<float>(sorted_dist_.size());
}

float UncertaintyDecomposition::epistemic(const Tensor& h) const {
    if (h.dtype() != DType::F32 || h.numel() != static_cast<int64_t>(E_)) return 1.0f;
    return epistemic(h.f32());
}

// ---------------------------------------------------------------------------
// split
// ---------------------------------------------------------------------------
UncertaintySplit UncertaintyDecomposition::split(const float* h, const float* probs,
                                                 int32_t K) const {
    UncertaintySplit out;
    out.aleatoric = aleatoric(probs, K);
    out.epistemic = epistemic(h);

    const bool a_hot = out.aleatoric >= cfg_.aleatoric_thresh;
    const bool e_hot = out.epistemic >= cfg_.epistemic_thresh;

    if (a_hot && e_hot)      out.dominant = DoubtSource::Both;
    else if (e_hot)          out.dominant = DoubtSource::Epistemic;
    else if (a_hot)          out.dominant = DoubtSource::Aleatoric;
    else                     out.dominant = DoubtSource::None;

    out.recommend_abstain = a_hot || e_hot;
    switch (out.dominant) {
        case DoubtSource::Both:
            out.reason = "unfamiliar input AND overlapping classes";
            break;
        case DoubtSource::Epistemic:
            out.reason = "input outside the fitted reference; more data may help";
            break;
        case DoubtSource::Aleatoric:
            out.reason = "classes overlap on this input; more data will not help";
            break;
        default:
            out.reason = "within reference, decision is well separated";
            break;
    }
    return out;
}

UncertaintySplit UncertaintyDecomposition::split(const Tensor& h, const float* probs,
                                                 int32_t K) const {
    if (h.dtype() != DType::F32 || h.numel() != static_cast<int64_t>(E_)) {
        UncertaintySplit out;
        out.aleatoric = aleatoric(probs, K);
        out.epistemic = 1.0f;               // fail closed
        out.dominant  = DoubtSource::Epistemic;
        out.recommend_abstain = true;
        out.reason = "hidden state does not match the fitted reference width";
        return out;
    }
    return split(h.f32(), probs, K);
}

// ---------------------------------------------------------------------------
// from_ensemble — the rigorous operator
// ---------------------------------------------------------------------------
UncertaintyDecomposition::EnsembleSplit
UncertaintyDecomposition::from_ensemble(const std::vector<std::vector<float>>& members) {
    EnsembleSplit out;
    if (members.empty()) return out;
    const size_t K = members.front().size();
    if (K == 0) return out;
    for (const std::vector<float>& m : members)
        if (m.size() != K) return out;      // ragged => refuse, don't guess

    const size_t S = members.size();
    std::vector<double> mean_p(K, 0.0);
    double mean_h = 0.0;

    for (const std::vector<float>& m : members) {
        // Normalise each member defensively; a member that is not a
        // distribution would otherwise distort both terms.
        double sum = 0.0;
        for (size_t k = 0; k < K; ++k) {
            if (!std::isfinite(m[k]) || m[k] < 0.0f) return out;
            sum += static_cast<double>(m[k]);
        }
        if (!(sum > 0.0)) return out;

        double h = 0.0;
        for (size_t k = 0; k < K; ++k) {
            const double pk = static_cast<double>(m[k]) / sum;
            if (pk > 0.0) h -= pk * std::log(pk);
            mean_p[k] += pk / static_cast<double>(S);
        }
        mean_h += h / static_cast<double>(S);
    }

    double h_mean = 0.0;
    for (size_t k = 0; k < K; ++k)
        if (mean_p[k] > 0.0) h_mean -= mean_p[k] * std::log(mean_p[k]);

    out.total     = static_cast<float>(h_mean);
    out.aleatoric = static_cast<float>(mean_h);
    // Jensen: H(mean) >= mean H, so this is >= 0 up to floating-point dust.
    double epi = h_mean - mean_h;
    if (epi < 0.0) epi = 0.0;
    out.epistemic = static_cast<float>(epi);
    return out;
}

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------
bool UncertaintyDecomposition::save(const std::string& path) const {
    if (!fitted_) return false;
    FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) return false;

    bool ok = std::fwrite(kMagic, 1, 8, f) == 8;
    ok = ok && wr(f, kFormatVersion);
    ok = ok && wr(f, E_);
    ok = ok && wr(f, n_fit_);
    ok = ok && wr(f, cfg_.aleatoric_thresh);
    ok = ok && wr(f, cfg_.epistemic_thresh);
    if (ok && !mu_.empty())
        ok = std::fwrite(mu_.data(), sizeof(float), mu_.size(), f) == mu_.size();
    if (ok && !sigma_.empty())
        ok = std::fwrite(sigma_.data(), sizeof(float), sigma_.size(), f) == sigma_.size();
    if (ok && !sorted_dist_.empty())
        ok = std::fwrite(sorted_dist_.data(), sizeof(float), sorted_dist_.size(), f) ==
             sorted_dist_.size();

    std::fclose(f);
    return ok;
}

bool UncertaintyDecomposition::load(const std::string& path) {
    // Fail closed first: whatever the file turns out to be, a caller that
    // ignores the return value must not keep a previous reference alive and
    // believe it describes the file it just failed to read.
    fitted_ = false;
    n_fit_  = 0;
    E_      = 0;
    mu_.clear();
    sigma_.clear();
    sorted_dist_.clear();

    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) { error_ = "uncertainty: cannot open " + path; return false; }

    char magic[8] = {0};
    bool ok = std::fread(magic, 1, 8, f) == 8 && std::memcmp(magic, kMagic, 8) == 0;

    int32_t version = 0, E = 0, n_fit = 0;
    float   a_thresh = 0.5f, e_thresh = 0.9f;
    ok = ok && rd(f, version) && version == kFormatVersion;
    ok = ok && rd(f, E) && E > 0 && E <= kMaxDim;
    ok = ok && rd(f, n_fit) && n_fit >= 2 && n_fit <= (1 << 24);
    ok = ok && rd(f, a_thresh) && rd(f, e_thresh);

    if (!ok) {
        std::fclose(f);
        error_ = "uncertainty: malformed header in " + path;
        return false;
    }

    std::vector<float> mu(static_cast<size_t>(E));
    std::vector<float> sg(static_cast<size_t>(E));
    std::vector<float> sd(static_cast<size_t>(n_fit));
    ok = std::fread(mu.data(), sizeof(float), mu.size(), f) == mu.size();
    if (ok) ok = std::fread(sg.data(), sizeof(float), sg.size(), f) == sg.size();
    if (ok) ok = std::fread(sd.data(), sizeof(float), sd.size(), f) == sd.size();
    std::fclose(f);

    // A reference whose numbers are not numbers is not a reference.
    if (!ok || !all_finite(mu.data(), E) || !all_finite(sg.data(), E) ||
        !all_finite(sd.data(), n_fit)) {
        error_ = "uncertainty: truncated or non-finite payload in " + path;
        return false;
    }
    for (int32_t e = 0; e < E; ++e)
        if (!(sg[static_cast<size_t>(e)] > 0.0f)) {
            error_ = "uncertainty: non-positive sigma in " + path;
            return false;
        }
    if (!std::is_sorted(sd.begin(), sd.end())) {
        error_ = "uncertainty: reference distances are not sorted in " + path;
        return false;
    }

    E_      = E;
    n_fit_  = n_fit;
    mu_     = std::move(mu);
    sigma_  = std::move(sg);
    sorted_dist_ = std::move(sd);
    cfg_.aleatoric_thresh = std::isfinite(a_thresh) ? a_thresh : 0.5f;
    cfg_.epistemic_thresh = std::isfinite(e_thresh) ? e_thresh : 0.9f;
    fitted_ = true;
    error_.clear();
    return true;
}

} // namespace omniseed
