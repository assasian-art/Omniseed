// =============================================================================
//  OmniSeed — trading/regime_engine.cpp
//  C++ port of tools/monster/regime.py. See the header for the rationale.
//
//  FIDELITY CONTRACT: this file must produce the same numbers as the Python
//  oracle. tests/test_regime_parity.py runs both over identical bars and fails
//  if they diverge — a hand-port of 700 lines of numerics WILL drift otherwise,
//  and a silently-diverging second implementation is worse than no port at all.
//  Keep the structure parallel to the Python so a reader can diff them by eye.
// =============================================================================
#include "omniseed/trading/regime_engine.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <set>

namespace omniseed {
namespace trading {
namespace regime_detail {

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------
double clampd(double x, double lo, double hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

double ramp(double x, double lo, double hi) {
    if (!regime_finite(x)) return kRegimeNaN;
    if (hi == lo) return 0.5;
    return clampd((x - lo) / (hi - lo), 0.0, 1.0);
}

void ols(const std::vector<double>& xs, const std::vector<double>& ys,
         double& slope, double& intercept, double& r2) {
    slope = kRegimeNaN;
    intercept = kRegimeNaN;
    r2 = kRegimeNaN;
    const size_t n = xs.size();
    if (n < 2 || ys.size() != n) return;

    double mx = 0.0, my = 0.0;
    for (size_t i = 0; i < n; ++i) { mx += xs[i]; my += ys[i]; }
    mx /= static_cast<double>(n);
    my /= static_cast<double>(n);

    double sxx = 0.0, sxy = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double dx = xs[i] - mx;
        sxx += dx * dx;
        sxy += dx * (ys[i] - my);
    }
    if (sxx <= 0.0) return;

    slope = sxy / sxx;
    intercept = my - slope * mx;

    double syy = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double dy = ys[i] - my;
        syy += dy * dy;
    }
    if (syy <= 0.0) { r2 = 0.0; return; }
    r2 = clampd((sxy * sxy) / (sxx * syy), 0.0, 1.0);
}

double mean(const std::vector<double>& v) {
    if (v.empty()) return kRegimeNaN;
    double s = 0.0;
    for (double x : v) s += x;
    return s / static_cast<double>(v.size());
}

double var_pop(const std::vector<double>& v) {
    const size_t n = v.size();
    if (n < 2) return kRegimeNaN;
    double m = 0.0;
    for (double x : v) m += x;
    m /= static_cast<double>(n);
    double acc = 0.0;
    for (double x : v) { const double d = x - m; acc += d * d; }
    return acc / static_cast<double>(n);
}

// ---------------------------------------------------------------------------
// column extractors
// ---------------------------------------------------------------------------
void closes(const std::vector<Bar>& bars, std::vector<double>& out) {
    out.clear(); out.reserve(bars.size());
    for (const Bar& b : bars) out.push_back(b.close);
}
void highs(const std::vector<Bar>& bars, std::vector<double>& out) {
    out.clear(); out.reserve(bars.size());
    for (const Bar& b : bars) out.push_back(b.high);
}
void lows(const std::vector<Bar>& bars, std::vector<double>& out) {
    out.clear(); out.reserve(bars.size());
    for (const Bar& b : bars) out.push_back(b.low);
}
void opens(const std::vector<Bar>& bars, std::vector<double>& out) {
    out.clear(); out.reserve(bars.size());
    for (const Bar& b : bars) out.push_back(b.open);
}

void sma(const std::vector<double>& vals, int32_t period,
         std::vector<double>& out) {
    const size_t n = vals.size();
    out.assign(n, kRegimeNaN);
    if (period <= 0) return;
    double run = 0.0;
    for (size_t i = 0; i < n; ++i) {
        run += vals[i];
        if (i >= static_cast<size_t>(period)) run -= vals[i - period];
        if (i >= static_cast<size_t>(period) - 1)
            out[i] = run / static_cast<double>(period);
    }
}

// true_range[i] = h-l for i==0, else max(h-l, |h-c_prev|, |l-c_prev|).
void true_range(const std::vector<Bar>& bars, std::vector<double>& out) {
    const size_t n = bars.size();
    out.assign(n, kRegimeNaN);
    if (n == 0) return;
    out[0] = bars[0].high - bars[0].low;
    for (size_t i = 1; i < n; ++i) {
        const double hl = bars[i].high - bars[i].low;
        const double hc = std::fabs(bars[i].high - bars[i - 1].close);
        const double lc = std::fabs(bars[i].low - bars[i - 1].close);
        out[i] = std::max(hl, std::max(hc, lc));
    }
}

// Wilder-smoothed ATR, last value only (that is all the regime engine uses).
double atr_last(const std::vector<Bar>& bars, int32_t period) {
    const size_t n = bars.size();
    if (period <= 0 || n < static_cast<size_t>(period)) return kRegimeNaN;
    std::vector<double> tr;
    true_range(bars, tr);
    double prev = 0.0;
    for (int32_t i = 0; i < period; ++i) prev += tr[static_cast<size_t>(i)];
    prev /= static_cast<double>(period);
    for (size_t i = static_cast<size_t>(period); i < n; ++i)
        prev = (prev * (period - 1) + tr[i]) / static_cast<double>(period);
    return prev;
}

// ---------------------------------------------------------------------------
// DIRECTIONAL AXIS
// ---------------------------------------------------------------------------
void variance_ratio(const std::vector<double>& returns, int32_t q,
                    double& vr_out, double& z_out) {
    vr_out = kRegimeNaN;
    z_out = kRegimeNaN;

    std::vector<double> r;
    r.reserve(returns.size());
    for (double x : returns) if (regime_finite(x)) r.push_back(x);
    const size_t n = r.size();
    if (q < 2 || n < static_cast<size_t>(q) + 1) return;

    double mu = 0.0;
    for (double x : r) mu += x;
    mu /= static_cast<double>(n);

    double var1 = 0.0;
    for (double x : r) { const double d = x - mu; var1 += d * d; }
    var1 /= static_cast<double>(n) - 1.0;
    if (var1 <= 0.0) return;

    // unbiased numerator: m = q(n-q+1)(1 - q/n)
    const double m = static_cast<double>(q) *
                     static_cast<double>(n - static_cast<size_t>(q) + 1) *
                     (1.0 - static_cast<double>(q) / static_cast<double>(n));
    if (m <= 0.0) return;

    double acc = 0.0;
    for (size_t t = static_cast<size_t>(q) - 1; t < n; ++t) {
        double s = 0.0;
        for (int32_t k = 0; k < q; ++k) s += r[t - static_cast<size_t>(k)];
        const double d = s - static_cast<double>(q) * mu;
        acc += d * d;
    }
    const double vr = (acc / m) / var1;
    vr_out = vr;

    double sumsq = 0.0;
    for (double x : r) { const double d = x - mu; sumsq += d * d; }
    const double denom = sumsq * sumsq;
    if (denom <= 0.0) return;

    double theta = 0.0;
    for (int32_t j = 1; j < q; ++j) {
        double num = 0.0;
        for (size_t t = static_cast<size_t>(j); t < n; ++t) {
            const double a = r[t] - mu;
            const double b = r[t - static_cast<size_t>(j)] - mu;
            num += a * a * b * b;
        }
        const double c = 2.0 * static_cast<double>(q - j) / static_cast<double>(q);
        theta += c * c * (num / denom);
    }
    if (theta <= 0.0) return;
    z_out = (vr - 1.0) / std::sqrt(theta);
}

double hurst_rs(const std::vector<double>& vals, int32_t min_n,
                int32_t num_sizes) {
    std::vector<double> x;
    x.reserve(vals.size());
    for (double v : vals) if (regime_finite(v)) x.push_back(v);

    const size_t N = x.size();
    if (N < static_cast<size_t>(min_n) * 3) return kRegimeNaN;
    const size_t max_n = N / 2;
    if (max_n <= static_cast<size_t>(min_n)) return kRegimeNaN;

    const int32_t denom_k = std::max(1, num_sizes - 1);
    std::set<size_t> size_set;
    for (int32_t k = 0; k < num_sizes; ++k) {
        const double v = static_cast<double>(min_n) +
                         (static_cast<double>(max_n) - static_cast<double>(min_n)) *
                         static_cast<double>(k) / static_cast<double>(denom_k);
        size_set.insert(static_cast<size_t>(std::llround(v)));
    }
    std::vector<size_t> sizes;
    for (size_t n : size_set)
        if (n >= static_cast<size_t>(min_n) && n <= max_n && n >= 4)
            sizes.push_back(n);

    std::vector<double> xs, ys;
    for (size_t n : sizes) {
        const size_t chunks = N / n;
        if (chunks < 1) continue;
        std::vector<double> ratios;
        for (size_t c = 0; c < chunks; ++c) {
            const size_t off = c * n;
            double m = 0.0;
            for (size_t k = 0; k < n; ++k) m += x[off + k];
            m /= static_cast<double>(n);

            double z = 0.0, lo = 0.0, hi = 0.0;
            for (size_t k = 0; k < n; ++k) {
                z += x[off + k] - m;
                if (k == 0) { lo = z; hi = z; }
                else { lo = std::min(lo, z); hi = std::max(hi, z); }
            }
            const double rng = hi - lo;

            double acc = 0.0;
            for (size_t k = 0; k < n; ++k) {
                const double d = x[off + k] - m;
                acc += d * d;
            }
            const double s = std::sqrt(acc / static_cast<double>(n));
            if (s > 0.0 && rng > 0.0) ratios.push_back(rng / s);
        }
        if (!ratios.empty()) {
            xs.push_back(std::log(static_cast<double>(n)));
            ys.push_back(std::log(mean(ratios)));
        }
    }
    if (xs.size() < 3) return kRegimeNaN;
    double slope = kRegimeNaN, inter = kRegimeNaN, r2 = kRegimeNaN;
    ols(xs, ys, slope, inter, r2);
    return slope;
}

double efficiency_ratio(const std::vector<double>& vals, int32_t n) {
    if (n < 2 || vals.size() < static_cast<size_t>(n) + 1) return kRegimeNaN;
    const size_t start = vals.size() - static_cast<size_t>(n) - 1;
    for (size_t i = start; i < vals.size(); ++i)
        if (!regime_finite(vals[i])) return kRegimeNaN;
    const double net = std::fabs(vals.back() - vals[start]);
    double path = 0.0;
    for (size_t i = start + 1; i < vals.size(); ++i)
        path += std::fabs(vals[i] - vals[i - 1]);
    return path > 0.0 ? net / path : 0.0;
}

namespace {
// Wilder running-sum smoothing used by ADX. Returns an empty vector when the
// input is shorter than `period` — matching the Python helper exactly.
void wilder(const std::vector<double>& v, int32_t period,
            std::vector<double>& out) {
    out.clear();
    if (v.size() < static_cast<size_t>(period)) return;
    double run = 0.0;
    for (int32_t i = 0; i < period; ++i) run += v[static_cast<size_t>(i)];
    out.push_back(run);
    for (size_t i = static_cast<size_t>(period); i < v.size(); ++i) {
        run = run - run / static_cast<double>(period) + v[i];
        out.push_back(run);
    }
}
}  // namespace

double adx(const std::vector<Bar>& bars, int32_t period) {
    const size_t n = bars.size();
    if (period <= 0 || n < static_cast<size_t>(2 * period) + 1)
        return kRegimeNaN;

    std::vector<double> trs, pdm, ndm;
    trs.reserve(n - 1); pdm.reserve(n - 1); ndm.reserve(n - 1);
    for (size_t i = 1; i < n; ++i) {
        const double hl = bars[i].high - bars[i].low;
        const double hc = std::fabs(bars[i].high - bars[i - 1].close);
        const double lc = std::fabs(bars[i].low - bars[i - 1].close);
        trs.push_back(std::max(hl, std::max(hc, lc)));

        const double up = bars[i].high - bars[i - 1].high;
        const double dn = bars[i - 1].low - bars[i].low;
        pdm.push_back((up > dn && up > 0.0) ? up : 0.0);
        ndm.push_back((dn > up && dn > 0.0) ? dn : 0.0);
    }

    std::vector<double> str_, sp, sn;
    wilder(trs, period, str_);
    wilder(pdm, period, sp);
    wilder(ndm, period, sn);
    if (str_.size() < static_cast<size_t>(period) + 1) return kRegimeNaN;

    std::vector<double> dxs;
    dxs.reserve(str_.size());
    for (size_t k = 0; k < str_.size(); ++k) {
        if (str_[k] <= 0.0) continue;
        const double pdi = 100.0 * sp[k] / str_[k];
        const double ndi = 100.0 * sn[k] / str_[k];
        const double tot = pdi + ndi;
        dxs.push_back(tot > 0.0 ? 100.0 * std::fabs(pdi - ndi) / tot : 0.0);
    }
    if (dxs.size() < static_cast<size_t>(period)) return kRegimeNaN;

    double a = 0.0;
    for (int32_t i = 0; i < period; ++i) a += dxs[static_cast<size_t>(i)];
    a /= static_cast<double>(period);
    for (size_t i = static_cast<size_t>(period); i < dxs.size(); ++i)
        a = (a * (period - 1) + dxs[i]) / static_cast<double>(period);
    return a;
}

double choppiness(const std::vector<Bar>& bars, int32_t n) {
    if (n < 2 || bars.size() < static_cast<size_t>(n) + 1) return kRegimeNaN;
    const size_t total = bars.size();
    const size_t start = total - static_cast<size_t>(n) - 1;

    double hh = bars[start].high, ll = bars[start].low;
    for (size_t i = start; i < total; ++i) {
        hh = std::max(hh, bars[i].high);
        ll = std::min(ll, bars[i].low);
    }
    const double rng = hh - ll;
    if (rng <= 0.0) return kRegimeNaN;

    double tr_sum = 0.0;
    for (size_t i = start + 1; i < total; ++i) {
        const double hl = bars[i].high - bars[i].low;
        const double hc = std::fabs(bars[i].high - bars[i - 1].close);
        const double lc = std::fabs(bars[i].low - bars[i - 1].close);
        tr_sum += std::max(hl, std::max(hc, lc));
    }
    if (tr_sum <= 0.0) return kRegimeNaN;
    return 100.0 * std::log10(tr_sum / rng) / std::log10(static_cast<double>(n));
}

void linreg_trend(const std::vector<double>& vals, int32_t n,
                  double& slope_out, double& r2_out) {
    slope_out = kRegimeNaN;
    r2_out = kRegimeNaN;
    std::vector<double> x;
    x.reserve(vals.size());
    for (double v : vals) if (regime_finite(v)) x.push_back(v);
    if (n > 0 && x.size() > static_cast<size_t>(n))
        x.erase(x.begin(), x.end() - static_cast<size_t>(n));
    if (x.size() < 3) return;

    std::vector<double> idx(x.size());
    for (size_t i = 0; i < x.size(); ++i) idx[i] = static_cast<double>(i);
    double inter = kRegimeNaN;
    ols(idx, x, slope_out, inter, r2_out);
}

double autocorr1(const std::vector<double>& returns) {
    std::vector<double> r;
    r.reserve(returns.size());
    for (double x : returns) if (regime_finite(x)) r.push_back(x);
    const size_t n = r.size();
    if (n < 10) return kRegimeNaN;
    double m = 0.0;
    for (double x : r) m += x;
    m /= static_cast<double>(n);

    double num = 0.0, den = 0.0;
    for (size_t i = 1; i < n; ++i) num += (r[i] - m) * (r[i - 1] - m);
    for (double x : r) { const double d = x - m; den += d * d; }
    return den > 0.0 ? num / den : kRegimeNaN;
}

double trend_consistency(const std::vector<double>& vals, int32_t fast,
                         int32_t slow) {
    const size_t n = vals.size();
    if (n < static_cast<size_t>(slow) + 5) return kRegimeNaN;
    std::vector<double> sf, ss;
    sma(vals, fast, sf);
    sma(vals, slow, ss);

    int64_t agree = 0, total = 0;
    for (size_t i = 0; i < n; ++i) {
        if (regime_finite(sf[i]) && regime_finite(ss[i])) {
            ++total;
            if (sf[i] > ss[i]) ++agree;
        }
    }
    if (total < 10) return kRegimeNaN;
    return static_cast<double>(agree) / static_cast<double>(total);
}

// ---------------------------------------------------------------------------
// VOLATILITY AXIS
// ---------------------------------------------------------------------------
double yang_zhang_vol(const std::vector<Bar>& bars, int32_t n) {
    if (n < 2 || bars.size() < static_cast<size_t>(n) + 1) return kRegimeNaN;
    const size_t total = bars.size();
    const size_t start = total - static_cast<size_t>(n) - 1;

    std::vector<double> so, sc, srs;
    for (size_t i = start + 1; i < total; ++i) {
        const double o = bars[i].open, h = bars[i].high;
        const double l = bars[i].low, c = bars[i].close;
        const double op = bars[i - 1].open, cp = bars[i - 1].close;
        if (o <= 0.0 || c <= 0.0 || cp <= 0.0 || op <= 0.0 || h <= 0.0 || l <= 0.0)
            continue;
        so.push_back(std::log(o / cp));
        sc.push_back(std::log(c / o));
        srs.push_back(std::log(h / c) * std::log(h / o) +
                      std::log(l / c) * std::log(l / o));
    }
    if (so.size() < 3) return kRegimeNaN;

    const double k = 0.34 / (1.34 + (static_cast<double>(n) + 1.0) /
                                    (static_cast<double>(n) - 1.0));
    const double v = var_pop(so) + k * var_pop(sc) + (1.0 - k) * var_pop(srs);
    return v > 0.0 ? std::sqrt(v) : kRegimeNaN;
}

double percentile_rank(const std::vector<double>& history, double value) {
    if (!regime_finite(value)) return kRegimeNaN;
    size_t count = 0, total = 0;
    for (double x : history) {
        if (!regime_finite(x)) continue;
        ++total;
        if (x <= value) ++count;
    }
    if (total == 0) return kRegimeNaN;
    return static_cast<double>(count) / static_cast<double>(total);
}

// ---------------------------------------------------------------------------
// TRADEABILITY
// ---------------------------------------------------------------------------
double ou_half_life(const std::vector<double>& vals, int32_t n) {
    std::vector<double> x;
    x.reserve(vals.size());
    for (double v : vals) if (regime_finite(v)) x.push_back(v);
    if (x.size() < 20) return kRegimeNaN;
    if (x.size() > static_cast<size_t>(n) + 1)
        x.erase(x.begin(), x.end() - static_cast<size_t>(n) - 1);
    if (x.size() < 20) return kRegimeNaN;

    std::vector<double> lv, dv;
    lv.reserve(x.size() - 1);
    dv.reserve(x.size() - 1);
    for (size_t i = 0; i + 1 < x.size(); ++i) {
        lv.push_back(x[i]);
        dv.push_back(x[i + 1] - x[i]);
    }
    double slope = kRegimeNaN, inter = kRegimeNaN, r2 = kRegimeNaN;
    ols(lv, dv, slope, inter, r2);
    if (!regime_finite(slope) || slope >= 0.0) return kRegimeNaN;
    const double hl = -std::log(2.0) / slope;
    return hl > 0.0 ? hl : kRegimeNaN;
}

}  // namespace regime_detail

using namespace regime_detail;

// ---------------------------------------------------------------------------
// detail() — display string, deterministic even when a component is NaN.
// ---------------------------------------------------------------------------
namespace {
std::string fmt_nan(double x, int digits) {
    if (!regime_finite(x)) return "nan";
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*f", digits, x);
    return std::string(buf);
}
}  // namespace

std::string RegimeState::detail() const {
    std::string s = "regime=" + label +
                    " trend=" + fmt_nan(trend_score, 3) +
                    " vol=" + fmt_nan(vol_score, 2);
    if (!direction.empty()) s += " dir=" + direction;
    if (regime_finite(half_life)) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), " hl=%.0f", half_life);
        s += buf;
    }
    return s;
}

// ---------------------------------------------------------------------------
// The directional blend -> (trend_score, components)
// ---------------------------------------------------------------------------
namespace {

struct Components {
    double vr = kRegimeNaN, vr_z = kRegimeNaN, adx = kRegimeNaN;
    double er = kRegimeNaN, chop = kRegimeNaN, hurst = kRegimeNaN;
    double r2 = kRegimeNaN, slope = kRegimeNaN, rho1 = kRegimeNaN;
    double consist = kRegimeNaN;
};

double directional(const std::vector<Bar>& win_bars, const RegimeConfig& cfg,
                   Components& comp) {
    std::vector<double> cl;
    closes(win_bars, cl);

    // log returns of the window
    std::vector<double> rets;
    rets.reserve(cl.size());
    for (size_t i = 1; i < cl.size(); ++i)
        if (cl[i] > 0.0 && cl[i - 1] > 0.0)
            rets.push_back(std::log(cl[i] / cl[i - 1]));

    // variance ratio: average the normalized reading across horizons.
    std::vector<double> vrs, zs;
    for (int32_t qi = 0; qi < cfg.vr_q_n; ++qi) {
        double vr = kRegimeNaN, z = kRegimeNaN;
        variance_ratio(rets, cfg.vr_q[qi], vr, z);
        if (regime_finite(vr)) vrs.push_back(vr);
        if (regime_finite(z)) zs.push_back(z);
    }
    const double vr_mean = vrs.empty() ? kRegimeNaN : mean(vrs);
    comp.vr = vr_mean;
    comp.vr_z = zs.empty() ? kRegimeNaN : mean(zs);

    const double a = adx(win_bars, cfg.adx_period);
    comp.adx = a;

    // ER is lookback-sensitive by construction: a 20-bar pullback inside a
    // 200-bar uptrend drives ER(20) to ~0 and would flip the whole label. Sweep
    // several horizons and average the NORMALIZED readings, so a short-term
    // pause cannot erase a long-term trend.
    std::vector<double> ers;
    for (int32_t ei = 0; ei < cfg.er_n_n; ++ei) {
        const double e = efficiency_ratio(cl, cfg.er_n[ei]);
        if (regime_finite(e)) ers.push_back(e);
    }
    comp.er = ers.empty() ? kRegimeNaN : mean(ers);

    const double ch = choppiness(win_bars, cfg.chop_n);
    comp.chop = ch;

    // Hurst MUST run on the INCREMENTS (see header).
    const double h = hurst_rs(rets, cfg.hurst_min_n, 8);
    comp.hurst = h;

    const int32_t lr_n = static_cast<int32_t>(
        std::min<size_t>(cl.size(), static_cast<size_t>(60)));
    double slope = kRegimeNaN, r2 = kRegimeNaN;
    linreg_trend(cl, lr_n, slope, r2);
    comp.r2 = r2;
    comp.slope = slope;

    std::vector<double> rho_win;
    if (rets.size() > static_cast<size_t>(cfg.rho_window))
        rho_win.assign(rets.end() - cfg.rho_window, rets.end());
    else
        rho_win = rets;
    const double rho = autocorr1(rho_win);
    comp.rho1 = rho;

    const double consist = trend_consistency(cl, cfg.ma_fast, cfg.ma_slow);
    comp.consist = consist;

    // Each view normalized so 0.5 == "random walk, no opinion".
    const double s_adx = ramp(a, 20.0, 40.0);
    std::vector<double> er_norms;
    for (double e : ers) er_norms.push_back(ramp(e, 0.20, 0.60));
    const double s_er = er_norms.empty() ? kRegimeNaN : mean(er_norms);
    const double s_chop = regime_finite(ch) ? (1.0 - ramp(ch, 38.2, 61.8))
                                            : kRegimeNaN;
    const double s_vr = regime_finite(vr_mean)
                            ? (0.5 + clampd(vr_mean - 1.0, -0.5, 0.5))
                            : kRegimeNaN;
    const double s_hurst = regime_finite(h)
                               ? (0.5 + clampd((h - 0.5) * 2.0, -0.5, 0.5))
                               : kRegimeNaN;
    const double s_r2 = ramp(r2, 0.20, 0.70);
    const double s_rho = regime_finite(rho)
                             ? (0.5 + clampd(rho * 5.0, -0.5, 0.5))
                             : kRegimeNaN;
    const double s_consist = regime_finite(consist)
                                 ? clampd(std::fabs(consist - 0.5) * 2.0, 0.0, 1.0)
                                 : kRegimeNaN;

    const double s[8] = {s_adx, s_er, s_chop, s_vr, s_hurst, s_r2, s_rho, s_consist};
    const double w[8] = {cfg.w_adx, cfg.w_er, cfg.w_chop, cfg.w_vr,
                         cfg.w_hurst, cfg.w_r2, cfg.w_rho, cfg.w_consist};

    double tot_w = 0.0, acc = 0.0;
    for (int i = 0; i < 8; ++i) {
        if (!regime_finite(s[i])) continue;
        tot_w += w[i];
        acc += s[i] * w[i];
    }
    if (tot_w <= 0.0) return 0.5;
    // Renormalize over whichever components are finite: a missing statistic
    // must not silently drag the blend toward "neutral".
    return clampd(acc / tot_w, 0.0, 1.0);
}

// -> (vol_score, yz_now, atr_stress). Stress is the STRONGER of two reads.
void volatility(const std::vector<Bar>& win_bars, const RegimeConfig& cfg,
                double& score_out, double& yz_out, double& atr_stress_out) {
    score_out = kRegimeNaN;
    yz_out = kRegimeNaN;
    atr_stress_out = kRegimeNaN;

    const size_t n = win_bars.size();
    if (n < static_cast<size_t>(cfg.yz_n) + 3) return;

    // the whole YZ series for this window, each point causal
    std::vector<double> ys(n, kRegimeNaN);
    for (size_t end = static_cast<size_t>(cfg.yz_n) + 2; end <= n; ++end)
        ys[end - 1] = yang_zhang_vol(
            std::vector<Bar>(win_bars.begin(), win_bars.begin() +
                             static_cast<ptrdiff_t>(end)), cfg.yz_n);

    std::vector<double> sm(n, kRegimeNaN);
    const size_t need = static_cast<size_t>(std::max(2, cfg.vol_smooth / 2));
    for (size_t i = 0; i < n; ++i) {
        const size_t lo = (i + 1 > static_cast<size_t>(cfg.vol_smooth))
                              ? i + 1 - static_cast<size_t>(cfg.vol_smooth) : 0;
        std::vector<double> w;
        for (size_t j = lo; j <= i; ++j)
            if (regime_finite(ys[j])) w.push_back(ys[j]);
        if (w.size() >= need) sm[i] = mean(w);
    }

    yz_out = ys.back();
    const double smooth_now = sm.back();

    std::vector<double> cl;
    closes(win_bars, cl);
    const double a = atr_last(win_bars, cfg.atr_period);
    const double atr_pct_now =
        (regime_finite(a) && !cl.empty() && cl.back() > 0.0) ? a / cl.back()
                                                             : kRegimeNaN;
    // 0 at half the stress threshold, 1 at the threshold (the mandate's atr_hi)
    const double atr_stress = ramp(atr_pct_now, cfg.atr_hi * 0.5, cfg.atr_hi);
    atr_stress_out = atr_stress;

    double vol_pct = kRegimeNaN;
    if (regime_finite(smooth_now)) {
        const size_t hi = (n >= 1) ? n - 1 : 0;
        const size_t lo = (hi > static_cast<size_t>(cfg.vol_hist))
                              ? hi - static_cast<size_t>(cfg.vol_hist) : 0;
        std::vector<double> hist;
        for (size_t j = lo; j < hi; ++j)
            if (regime_finite(sm[j])) hist.push_back(sm[j]);
        if (hist.size() >= 5) {
            const double mx = *std::max_element(hist.begin(), hist.end());
            const double mn = *std::min_element(hist.begin(), hist.end());
            const double spread = mx - mn;
            const double scale = std::max(1.0, std::fabs(mx));
            // A FLAT history carries no information: every sample ties, `<=`
            // holds for all of them, and the percentile pins to 1.0.
            vol_pct = (spread <= 1e-12 * scale) ? 0.5
                                               : percentile_rank(hist, smooth_now);
        }
    }

    double score = kRegimeNaN;
    if (regime_finite(vol_pct) && regime_finite(atr_stress))
        score = std::max(vol_pct, atr_stress);
    else if (regime_finite(vol_pct)) score = vol_pct;
    else if (regime_finite(atr_stress)) score = atr_stress;
    score_out = score;
}

}  // namespace

// ---------------------------------------------------------------------------
// detect
// ---------------------------------------------------------------------------
RegimeState RegimeEngine::detect(const std::vector<Bar>& bars, size_t i) const {
    return detect(bars, i, cfg_.window);
}

RegimeState RegimeEngine::detect(const std::vector<Bar>& bars, size_t i,
                                 int32_t window) const {
    RegimeState st;
    if (bars.empty()) return st;
    if (i >= bars.size()) i = bars.size() - 1;

    const int32_t w = window > 0 ? window : cfg_.window;
    const size_t lo = (static_cast<int64_t>(i) - w + 1 > 0)
                          ? i - static_cast<size_t>(w) + 1 : 0;
    const std::vector<Bar> win(bars.begin() + static_cast<ptrdiff_t>(lo),
                               bars.begin() + static_cast<ptrdiff_t>(i) + 1);

    st.ts = bars[i].time;
    if (win.size() < 20) {
        st.trend_score = 0.5;
        st.vol_score = kRegimeNaN;
        st.label = "range";
        return st;
    }

    Components comp;
    const double trend = directional(win, cfg_, comp);
    double vol_score = kRegimeNaN, yz = kRegimeNaN, atr_stress = kRegimeNaN;
    volatility(win, cfg_, vol_score, yz, atr_stress);

    std::vector<double> cl;
    closes(win, cl);
    const double hl = ou_half_life(cl, cfg_.ou_n);
    std::vector<double> sma_long;
    sma(cl, cfg_.sma_long, sma_long);
    const double ref = (cl.size() >= static_cast<size_t>(cfg_.sma_long) &&
                        regime_finite(sma_long.back()))
                           ? sma_long.back() : kRegimeNaN;

    std::string direction;
    if (regime_finite(comp.slope)) {
        if (comp.slope > 0.0) direction = "up";
        else if (comp.slope < 0.0) direction = "down";
    }

    // high_vol needs BOTH reads: relatively stressed (percentile) AND
    // absolutely stressed (the mandate's atr_hi).
    const bool stressed = regime_finite(vol_score) &&
                          vol_score >= cfg_.vol_stress &&
                          regime_finite(atr_stress) && atr_stress > 0.0;

    std::string label;
    if (stressed) {
        label = "high_vol";
    } else if (trend >= cfg_.trend_hi && (direction == "up" || direction == "down")) {
        // a trend claim also has to agree with the long-horizon reference
        if (regime_finite(ref)) {
            if (direction == "up" && cl.back() > ref) label = "trend_up";
            else if (direction == "down" && cl.back() < ref) label = "trend_down";
            else label = "range";
        } else {
            label = (direction == "up") ? "trend_up" : "trend_down";
        }
    } else {
        label = "range";
    }

    st.trend_score = trend;
    st.vol_score = vol_score;
    st.label = label;
    st.direction = direction;
    st.adx = comp.adx;
    st.er = comp.er;
    st.chop = comp.chop;
    st.vr = comp.vr;
    st.vr_z = comp.vr_z;
    st.hurst = comp.hurst;
    st.r2 = comp.r2;
    st.rho1 = comp.rho1;
    st.yz_vol = yz;
    st.half_life = hl;
    st.stressed = stressed;
    return st;
}

std::vector<RegimeState> RegimeEngine::scan_raw(const std::vector<Bar>& bars) const {
    std::vector<RegimeState> out;
    out.reserve(bars.size());
    for (size_t i = 0; i < bars.size(); ++i) out.push_back(detect(bars, i));
    return out;
}

// ---------------------------------------------------------------------------
// RegimeTracker — hysteresis
// ---------------------------------------------------------------------------
RegimeState RegimeTracker::step(const std::vector<Bar>& bars, size_t i) {
    RegimeState st = engine_.detect(bars, i);

    // volatility latch: the most reliable axis, so it latches hardest.
    // Entry requires BOTH the relative and the absolute stress reads.
    if (st.stressed && regime_finite(st.vol_score) &&
        st.vol_score >= cfg_.vol_enter) {
        vol_ = true;
    } else if (!st.stressed || !regime_finite(st.vol_score) ||
               st.vol_score < cfg_.vol_exit) {
        vol_ = false;
    }

    // directional latch
    const bool has_dir = (st.direction == "up" || st.direction == "down");
    if (st.trend_score >= cfg_.trend_enter && has_dir) {
        trend_ = true;
        dir_ = st.direction;
    } else if (st.trend_score < cfg_.trend_exit || !has_dir) {
        trend_ = false;
        dir_ = st.direction;
    }

    if (vol_) st.label = "high_vol";
    else if (trend_) st.label = (dir_ == "up") ? "trend_up" : "trend_down";
    else st.label = "range";
    return st;
}

std::vector<RegimeState> RegimeTracker::run(const std::vector<Bar>& bars) {
    std::vector<RegimeState> out;
    out.reserve(bars.size());
    for (size_t i = 0; i < bars.size(); ++i) out.push_back(step(bars, i));
    return out;
}

}  // namespace trading
}  // namespace omniseed
