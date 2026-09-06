// =============================================================================
//  OmniSeed — fft.h
//  Dependency-free FFT for the log-mel front end (TASK 4 polish).
//
//  * radix-2 complex FFT (power-of-two sizes, iterative, precomputed kernel)
//  * Bluestein/chirp-z wrap: exact N=400 DFT bins evaluated with an M=1024
//    FFT — O(N log N) while producing the SAME bins as the naive DFT it
//    replaces, so the mel filterbank output is unchanged up to fp rounding.
//
//  Header-only, no external deps. Per frame the 400^2 naive transform
//  (~160k mul-adds) drops to three 1024-point transforms (~15k butterflies).
// =============================================================================
#pragma once

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace omniseed {
namespace dsp {

namespace detail {

constexpr double kPi = 3.14159265358979323846;

// In-place iterative radix-2 complex FFT. n must be a power of two.
inline void fft_radix2(std::vector<std::pair<double, double>>& a,
                       bool inverse) {
    const size_t n = a.size();
    if (n < 2) return;

    // bit-reversal permutation
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }

    // butterflies (recurrence twiddles: numerically stable, no per-k trig)
    for (size_t len = 2; len <= n; len <<= 1) {
        const double ang =
            (inverse ? 2.0 : -2.0) * kPi / static_cast<double>(len);
        const double w_re = std::cos(ang), w_im = std::sin(ang);
        for (size_t i = 0; i < n; i += len) {
            double cur_re = 1.0, cur_im = 0.0;
            for (size_t k = 0; k < len / 2; ++k) {
                const size_t i0 = i + k, i1 = i + k + len / 2;
                const double u_re = a[i0].first, u_im = a[i0].second;
                const double v_re = a[i1].first * cur_re - a[i1].second * cur_im;
                const double v_im = a[i1].first * cur_im + a[i1].second * cur_re;

                a[i0].first  = u_re + v_re;
                a[i0].second = u_im + v_im;
                a[i1].first  = u_re - v_re;
                a[i1].second = u_im - v_im;

                const double nre = cur_re * w_re - cur_im * w_im;
                cur_im = cur_re * w_im + cur_im * w_re;
                cur_re = nre;
            }
        }
    }
    if (inverse) {
        const double inv = 1.0 / static_cast<double>(n);
        for (auto& v : a) {
            v.first *= inv;
            v.second *= inv;
        }
    }
}

inline size_t next_pow2(size_t n) {
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

} // namespace detail

// ---------------------------------------------------------------------------
// Exact DFT of a real signal, evaluated only at bins k = 0..n_bins-1, via the
// Bluestein chirp-z identity:
//
//   X[k] = sum_t x[t] e^{-2pi i k t / N}
//        = e^{-pi i k^2/N} * sum_t (x[t] e^{-pi i t^2/N}) * e^{+pi i (k-t)^2/N}
//
// The linear convolution is evaluated as an M-point circular convolution
// (radix-2 FFTs, M >= n + kernel_len - 1, power of two). For N=400 and
// n_bins=201 the kernel covers j = k-t in [-(N-1), n_bins-1] (600 taps) and
// M = 1024, so no circular wrap touches the output indices.
// ---------------------------------------------------------------------------
class DftBins {
public:
    DftBins() = default;
    DftBins(size_t n, size_t n_bins) { init(n, n_bins); }

    void init(size_t n, size_t n_bins) {
        n_ = n;
        n_bins_ = n_bins;
        klen_ = (n_ - 1) + n_bins_;            // taps j = -(n-1) .. n_bins-1
        m_ = detail::next_pow2(n_ + klen_);    // >= n + klen - 1

        c_in_.assign(n_, {0.0, 0.0});          // e^{-pi i t^2 / N}
        for (size_t t = 0; t < n_; ++t) {
            const double ang = -detail::kPi * static_cast<double>(t) *
                               static_cast<double>(t) / static_cast<double>(n_);
            c_in_[t] = {std::cos(ang), std::sin(ang)};
        }
        c_out_.assign(n_bins_, {0.0, 0.0});    // e^{-pi i k^2 / N}
        for (size_t k = 0; k < n_bins_; ++k) {
            const double ang = -detail::kPi * static_cast<double>(k) *
                               static_cast<double>(k) / static_cast<double>(n_);
            c_out_[k] = {std::cos(ang), std::sin(ang)};
        }

        // kernel b[u] = e^{+pi i j^2 / N}, j = u - (n-1), u = 0..klen-1
        std::vector<std::pair<double, double>> b(m_, {0.0, 0.0});
        for (size_t u = 0; u < klen_; ++u) {
            const double j = static_cast<double>(u) -
                             static_cast<double>(n_ - 1);
            const double ang = detail::kPi * j * j / static_cast<double>(n_);
            b[u] = {std::cos(ang), std::sin(ang)};
        }
        detail::fft_radix2(b, false);
        bfft_ = std::move(b);
        a_.assign(m_, {0.0, 0.0});
    }

    // x: n real samples; out: n_bins complex DFT bins (out_re[k], out_im[k]).
    void run(const float* x, double* out_re, double* out_im) const {
        std::fill(a_.begin(), a_.end(), std::make_pair(0.0, 0.0));
        for (size_t t = 0; t < n_; ++t) {
            const double re = static_cast<double>(x[t]);
            a_[t].first  = re * c_in_[t].first;
            a_[t].second = re * c_in_[t].second;
        }
        detail::fft_radix2(a_, false);

        for (size_t i = 0; i < m_; ++i) {
            const double ar = a_[i].first, ai = a_[i].second;
            const double br = bfft_[i].first, bi = bfft_[i].second;
            a_[i].first  = ar * br - ai * bi;
            a_[i].second = ar * bi + ai * br;
        }
        detail::fft_radix2(a_, true);   // circular convolution result

        // X[k] = c_out[k] * conv[k + (n-1)]
        for (size_t k = 0; k < n_bins_; ++k) {
            const double cr = c_out_[k].first, ci = c_out_[k].second;
            const double vr = a_[k + n_ - 1].first;
            const double vi = a_[k + n_ - 1].second;
            out_re[k] = cr * vr - ci * vi;
            out_im[k] = cr * vi + ci * vr;
        }
    }

    size_t n() const { return n_; }
    size_t n_bins() const { return n_bins_; }

private:
    size_t n_ = 0, n_bins_ = 0, klen_ = 0, m_ = 0;
    std::vector<std::pair<double, double>> c_in_, c_out_, bfft_;
    mutable std::vector<std::pair<double, double>> a_;   // scratch
};

} // namespace dsp
} // namespace omniseed
