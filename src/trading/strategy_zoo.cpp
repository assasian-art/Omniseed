// =============================================================================
//  OmniSeed — trading/strategy_zoo.cpp
//  C++ implementation of the feature primitives and the strategy zoo.
//
//  Every function here is a line-for-line port of the Python oracle
//  (tools/monster/features.py, tools/monster/strategies.py). The port is
//  deliberately FAITHFUL, including the places where the oracle is subtle:
//
//    * `sma` accumulates with a running sum and subtracts the leaving sample,
//      rather than re-summing the window. That is the oracle's floating-point
//      summation ORDER, and matching it is what makes the parity test's 1e-9
//      tolerance achievable instead of a coin flip.
//    * `bollinger` does NOT check for NaNs inside the window, while
//      `rolling_std` DOES. The inconsistency is the oracle's; both are
//      reproduced, because a "cleaner" port that silently changes which bars
//      produce a number is a behaviour change, not a cleanup.
//    * `find_swings` ignores its `kind` argument in the oracle (always returns
//      strict MINIMA). Reproduced exactly — see the header note.
//
//  A silently-diverging second implementation would be worse than no port at
//  all, so tests/test_strategy_parity.py gates this file against the oracle.
// =============================================================================
#include "omniseed/trading/strategy_zoo.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <string>

namespace omniseed {
namespace trading {

namespace {

inline bool fin(double x) { return x == x && x != INFINITY && x != -INFINITY; }

std::vector<double> nan_list(size_t n) { return std::vector<double>(n, kRegimeNaN); }

std::string fmt(const char* f, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof(buf), f, ap);
    va_end(ap);
    return std::string(buf);
}

} // namespace

// =============================================================================
// features
// =============================================================================
namespace features {

double clampd(double x, double lo, double hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

double ramp(double x, double lo, double hi) {
    if (!fin(x)) return kRegimeNaN;
    if (hi == lo) return 0.5;
    return clampd((x - lo) / (hi - lo), 0.0, 1.0);
}

double safe(double x, double d) { return fin(x) ? x : d; }

std::vector<double> closes(const std::vector<Bar>& bars) {
    std::vector<double> out;
    out.reserve(bars.size());
    for (const Bar& b : bars) out.push_back(b.close);
    return out;
}
std::vector<double> highs(const std::vector<Bar>& bars) {
    std::vector<double> out;
    out.reserve(bars.size());
    for (const Bar& b : bars) out.push_back(b.high);
    return out;
}
std::vector<double> lows(const std::vector<Bar>& bars) {
    std::vector<double> out;
    out.reserve(bars.size());
    for (const Bar& b : bars) out.push_back(b.low);
    return out;
}
std::vector<double> volumes(const std::vector<Bar>& bars) {
    std::vector<double> out;
    out.reserve(bars.size());
    for (const Bar& b : bars) out.push_back(b.volume);
    return out;
}
std::vector<double> opens(const std::vector<Bar>& bars) {
    std::vector<double> out;
    out.reserve(bars.size());
    for (const Bar& b : bars) out.push_back(b.open);
    return out;
}

std::vector<double> sma(const std::vector<double>& vals, int32_t period) {
    const size_t n = vals.size();
    std::vector<double> out = nan_list(n);
    if (period <= 0) return out;
    double run = 0.0;
    for (size_t i = 0; i < n; ++i) {
        run += vals[i];
        if (static_cast<int64_t>(i) >= period) run -= vals[i - static_cast<size_t>(period)];
        if (static_cast<int64_t>(i) >= period - 1) out[i] = run / period;
    }
    return out;
}

std::vector<double> ema(const std::vector<double>& vals, int32_t period) {
    const size_t n = vals.size();
    std::vector<double> out = nan_list(n);
    if (period <= 0 || static_cast<int64_t>(n) < period) return out;
    double seed = 0.0;
    for (int32_t i = 0; i < period; ++i) seed += vals[static_cast<size_t>(i)];
    seed /= period;
    out[static_cast<size_t>(period - 1)] = seed;
    const double k = 2.0 / (period + 1.0);
    double prev = seed;
    for (size_t i = static_cast<size_t>(period); i < n; ++i) {
        prev = vals[i] * k + prev * (1.0 - k);
        out[i] = prev;
    }
    return out;
}

std::vector<double> rsi(const std::vector<double>& vals, int32_t period) {
    const size_t n = vals.size();
    std::vector<double> out = nan_list(n);
    if (static_cast<int64_t>(n) < static_cast<int64_t>(period) + 1 || period <= 0)
        return out;
    double gain = 0.0, loss = 0.0;
    for (int32_t i = 1; i <= period; ++i) {
        const double d = vals[static_cast<size_t>(i)] - vals[static_cast<size_t>(i) - 1];
        gain += std::max(d, 0.0);
        loss += std::max(-d, 0.0);
    }
    double avg_g = gain / period, avg_l = loss / period;
    auto rsi_at = [](double ag, double al) {
        return al == 0.0 ? 100.0 : 100.0 - 100.0 / (1.0 + ag / al);
    };
    out[static_cast<size_t>(period)] = rsi_at(avg_g, avg_l);
    for (size_t i = static_cast<size_t>(period) + 1; i < n; ++i) {
        const double d = vals[i] - vals[i - 1];
        avg_g = (avg_g * (period - 1) + std::max(d, 0.0)) / period;
        avg_l = (avg_l * (period - 1) + std::max(-d, 0.0)) / period;
        out[i] = rsi_at(avg_g, avg_l);
    }
    return out;
}

void macd(const std::vector<double>& vals, int32_t fast, int32_t slow,
          int32_t signal_p, std::vector<double>& line, std::vector<double>& sig,
          std::vector<double>& hist) {
    const size_t n = vals.size();
    const std::vector<double> ef = ema(vals, fast);
    const std::vector<double> es = ema(vals, slow);
    line.assign(n, kRegimeNaN);
    for (size_t i = 0; i < n; ++i)
        if (fin(ef[i]) && fin(es[i])) line[i] = ef[i] - es[i];

    std::vector<double> valid;
    valid.reserve(n);
    for (size_t i = 0; i < n; ++i)
        if (fin(line[i])) valid.push_back(line[i]);
    const std::vector<double> sig_valid = valid.empty()
        ? std::vector<double>()
        : ema(valid, signal_p);

    sig.assign(n, kRegimeNaN);
    size_t j = 0;
    for (size_t i = 0; i < n; ++i) {
        if (fin(line[i])) {
            if (j < sig_valid.size()) sig[i] = sig_valid[j];
            ++j;
        }
    }
    hist.assign(n, kRegimeNaN);
    for (size_t i = 0; i < n; ++i)
        if (fin(line[i]) && fin(sig[i])) hist[i] = line[i] - sig[i];
}

std::vector<double> true_range(const std::vector<Bar>& bars) {
    const size_t n = bars.size();
    std::vector<double> out = nan_list(n);
    for (size_t i = 0; i < n; ++i) {
        if (i == 0) {
            out[i] = bars[i].high - bars[i].low;
        } else {
            const double pc = bars[i - 1].close;
            out[i] = std::max(bars[i].high - bars[i].low,
                              std::max(std::fabs(bars[i].high - pc),
                                       std::fabs(bars[i].low - pc)));
        }
    }
    return out;
}

std::vector<double> atr(const std::vector<Bar>& bars, int32_t period) {
    const std::vector<double> tr = true_range(bars);
    const size_t n = bars.size();
    std::vector<double> out = nan_list(n);
    if (static_cast<int64_t>(n) < period || period <= 0) return out;
    double prev = 0.0;
    for (int32_t i = 0; i < period; ++i) prev += tr[static_cast<size_t>(i)];
    prev /= period;
    out[static_cast<size_t>(period - 1)] = prev;
    for (size_t i = static_cast<size_t>(period); i < n; ++i) {
        prev = (prev * (period - 1) + tr[i]) / period;
        out[i] = prev;
    }
    return out;
}

std::vector<double> atr_pct(const std::vector<Bar>& bars, int32_t period) {
    const std::vector<double> a = atr(bars, period);
    const std::vector<double> c = closes(bars);
    const size_t n = bars.size();
    std::vector<double> out = nan_list(n);
    for (size_t i = 0; i < n; ++i)
        if (fin(a[i]) && c[i] != 0.0) out[i] = a[i] / c[i];
    return out;
}

std::vector<double> rolling_std(const std::vector<double>& vals, int32_t period) {
    const size_t n = vals.size();
    std::vector<double> out = nan_list(n);
    if (period <= 0) return out;
    for (size_t i = static_cast<size_t>(period - 1); i < n; ++i) {
        bool ok = true;
        double m = 0.0;
        for (int32_t k = 0; k < period; ++k) {
            const double v = vals[i - static_cast<size_t>(period) + 1 + static_cast<size_t>(k)];
            if (!fin(v)) { ok = false; break; }
            m += v;
        }
        if (!ok) continue;
        m /= period;
        double ss = 0.0;
        for (int32_t k = 0; k < period; ++k) {
            const double v = vals[i - static_cast<size_t>(period) + 1 + static_cast<size_t>(k)];
            ss += (v - m) * (v - m);
        }
        out[i] = std::sqrt(ss / period);
    }
    return out;
}

void bollinger(const std::vector<double>& vals, int32_t period, double sigma,
               std::vector<double>& mid, std::vector<double>& up,
               std::vector<double>& lo) {
    const size_t n = vals.size();
    mid = nan_list(n);
    up = nan_list(n);
    lo = nan_list(n);
    if (period <= 0) return;
    for (size_t i = static_cast<size_t>(period - 1); i < n; ++i) {
        double m = 0.0;
        for (int32_t k = 0; k < period; ++k)
            m += vals[i - static_cast<size_t>(period) + 1 + static_cast<size_t>(k)];
        m /= period;
        double ss = 0.0;
        for (int32_t k = 0; k < period; ++k) {
            const double v = vals[i - static_cast<size_t>(period) + 1 + static_cast<size_t>(k)];
            ss += (v - m) * (v - m);
        }
        const double sd = std::sqrt(ss / period);
        mid[i] = m;
        up[i] = m + sigma * sd;
        lo[i] = m - sigma * sd;
    }
}

std::vector<double> rolling_vwap(const std::vector<Bar>& bars, int32_t period) {
    const size_t n = bars.size();
    std::vector<double> out = nan_list(n);
    if (period <= 0) return out;
    for (size_t i = static_cast<size_t>(period - 1); i < n; ++i) {
        double num = 0.0, den = 0.0;
        for (int32_t k = 0; k < period; ++k) {
            const Bar& b = bars[i - static_cast<size_t>(period) + 1 + static_cast<size_t>(k)];
            const double tp = (b.high + b.low + b.close) / 3.0;
            num += tp * b.volume;
            den += b.volume;
        }
        out[i] = den > 0.0 ? num / den : kRegimeNaN;
    }
    return out;
}

std::vector<double> rolling_zscore(const std::vector<double>& vals, int32_t period) {
    const size_t n = vals.size();
    std::vector<double> out = nan_list(n);
    if (period <= 0) return out;
    for (size_t i = static_cast<size_t>(period); i < n; ++i) {
        // Strictly the PAST window: [i-period, i-1].
        double m = 0.0;
        for (int32_t k = 0; k < period; ++k) m += vals[i - static_cast<size_t>(period) + static_cast<size_t>(k)];
        m /= period;
        double ss = 0.0;
        for (int32_t k = 0; k < period; ++k) {
            const double v = vals[i - static_cast<size_t>(period) + static_cast<size_t>(k)];
            ss += (v - m) * (v - m);
        }
        const double sd = std::sqrt(ss / period);
        out[i] = sd > 0.0 ? (vals[i] - m) / sd : 0.0;
    }
    return out;
}

std::vector<double> zscore_series(const std::vector<double>& vals, int32_t period) {
    const size_t n = vals.size();
    std::vector<double> out = nan_list(n);
    if (period <= 0) return out;
    const int32_t min_keep = std::max(5, period / 3);
    for (size_t i = static_cast<size_t>(period); i < n; ++i) {
        double sum = 0.0;
        int32_t cnt = 0;
        for (int32_t k = 0; k < period; ++k) {
            const double v = vals[i - static_cast<size_t>(period) + static_cast<size_t>(k)];
            if (fin(v)) { sum += v; ++cnt; }
        }
        if (cnt < min_keep) continue;
        const double mu = sum / cnt;
        double ss = 0.0;
        for (int32_t k = 0; k < period; ++k) {
            const double v = vals[i - static_cast<size_t>(period) + static_cast<size_t>(k)];
            if (fin(v)) ss += (v - mu) * (v - mu);
        }
        const double sd = std::sqrt(ss / cnt);
        out[i] = sd > 0.0 ? (vals[i] - mu) / sd : 0.0;
    }
    return out;
}

std::vector<double> rolling_corr(const std::vector<double>& a,
                                 const std::vector<double>& b, int32_t period) {
    const size_t n = std::min(a.size(), b.size());
    std::vector<double> out = nan_list(n);
    if (period <= 0) return out;
    for (size_t i = static_cast<size_t>(period - 1); i < n; ++i) {
        const size_t base = i - static_cast<size_t>(period) + 1;
        double ma = 0.0, mb = 0.0;
        for (int32_t k = 0; k < period; ++k) {
            ma += a[base + static_cast<size_t>(k)];
            mb += b[base + static_cast<size_t>(k)];
        }
        ma /= period;
        mb /= period;
        double num = 0.0, va = 0.0, vb = 0.0;
        for (int32_t k = 0; k < period; ++k) {
            const double da = a[base + static_cast<size_t>(k)] - ma;
            const double db = b[base + static_cast<size_t>(k)] - mb;
            num += da * db;
            va += da * da;
            vb += db * db;
        }
        out[i] = (va > 0.0 && vb > 0.0) ? num / std::sqrt(va * vb) : 0.0;
    }
    return out;
}

std::vector<double> log_returns(const std::vector<double>& vals) {
    const size_t n = vals.size();
    std::vector<double> out = nan_list(n);
    for (size_t i = 1; i < n; ++i)
        if (vals[i] > 0.0 && vals[i - 1] > 0.0) out[i] = std::log(vals[i] / vals[i - 1]);
    return out;
}

std::vector<double> amihud(const std::vector<Bar>& bars) {
    const std::vector<double> c = closes(bars);
    const std::vector<double> v = volumes(bars);
    const std::vector<double> r = log_returns(c);
    const size_t n = bars.size();
    std::vector<double> out = nan_list(n);
    for (size_t i = 0; i < n; ++i) {
        const double dv = c[i] * v[i];
        if (fin(r[i]) && dv > 0.0) out[i] = std::fabs(r[i]) / dv;
    }
    return out;
}

std::vector<double> kyle_lambda(const std::vector<Bar>& bars) {
    const std::vector<double> c = closes(bars);
    const std::vector<double> v = volumes(bars);
    const size_t n = bars.size();
    std::vector<double> out = nan_list(n);
    for (size_t i = 1; i < n; ++i)
        if (v[i] > 0.0) out[i] = std::fabs(c[i] - c[i - 1]) / v[i];
    return out;
}

std::vector<Swing> find_swings(const std::vector<double>& vals, int32_t left,
                               int32_t right) {
    const int64_t n = static_cast<int64_t>(vals.size());
    std::vector<Swing> out;
    for (int64_t i = left; i < n - right; ++i) {
        bool ok = true;
        for (int32_t k = 1; k <= left; ++k)
            if (!(vals[static_cast<size_t>(i)] < vals[static_cast<size_t>(i - k)])) { ok = false; break; }
        if (ok) {
            for (int32_t k = 1; k <= right; ++k)
                if (!(vals[static_cast<size_t>(i)] < vals[static_cast<size_t>(i + k)])) { ok = false; break; }
        }
        if (ok) out.push_back(Swing{static_cast<int32_t>(i), vals[static_cast<size_t>(i)]});
    }
    return out;
}

std::vector<Swing> find_local_maxima(const std::vector<double>& vals, int32_t left,
                                     int32_t right) {
    const int64_t n = static_cast<int64_t>(vals.size());
    std::vector<Swing> out;
    for (int64_t i = left; i < n - right; ++i) {
        bool ok = true;
        for (int32_t k = 1; k <= left; ++k)
            if (!(vals[static_cast<size_t>(i)] > vals[static_cast<size_t>(i - k)])) { ok = false; break; }
        if (ok) {
            for (int32_t k = 1; k <= right; ++k)
                if (!(vals[static_cast<size_t>(i)] > vals[static_cast<size_t>(i + k)])) { ok = false; break; }
        }
        if (ok) out.push_back(Swing{static_cast<int32_t>(i), vals[static_cast<size_t>(i)]});
    }
    return out;
}

std::vector<Swing> confirmed_swings(const std::vector<Swing>& swings, int32_t i,
                                    int32_t right) {
    std::vector<Swing> out;
    for (const Swing& s : swings)
        if (s.idx + right <= i) out.push_back(s);
    return out;
}

std::vector<double> fib_levels(double swing_low, double swing_high,
                               const std::vector<double>& ratios) {
    const double span = swing_high - swing_low;
    std::vector<double> out;
    out.reserve(ratios.size());
    for (double f : ratios) out.push_back(swing_high - span * f);
    return out;
}

bool last_swing_pair(const std::vector<Swing>& swing_lows,
                     const std::vector<Swing>& swing_highs, int32_t i,
                     int32_t right, Swing& out_low, Swing& out_high) {
    const std::vector<Swing> lows_c = confirmed_swings(swing_lows, i, right);
    const std::vector<Swing> highs_c = confirmed_swings(swing_highs, i, right);
    bool found = false;
    int32_t best_hi = 0;
    for (const Swing& l : lows_c) {
        for (const Swing& h : highs_c) {
            if (l.idx < h.idx && h.value > l.value) {
                if (!found || h.idx > best_hi) {
                    found = true;
                    best_hi = h.idx;
                    out_low = l;
                    out_high = h;
                }
            }
        }
    }
    return found;
}

bool bullish_divergence(const std::vector<double>& rsi_vals,
                        const std::vector<Swing>& swing_lows, int32_t i,
                        int32_t right, int32_t max_gap) {
    const std::vector<Swing> lows_c = confirmed_swings(swing_lows, i, right);
    if (lows_c.size() < 2) return false;
    const Swing& s1 = lows_c[lows_c.size() - 2];
    const Swing& s2 = lows_c[lows_c.size() - 1];
    if (!(s2.idx - s1.idx > 0 && s2.idx - s1.idx <= max_gap)) return false;
    if (static_cast<size_t>(s1.idx) >= rsi_vals.size() ||
        static_cast<size_t>(s2.idx) >= rsi_vals.size())
        return false;
    const double r1 = rsi_vals[static_cast<size_t>(s1.idx)];
    const double r2 = rsi_vals[static_cast<size_t>(s2.idx)];
    if (!fin(r1) || !fin(r2)) return false;
    return s2.value < s1.value && r2 > r1;
}

} // namespace features

// =============================================================================
// StrategySignal
// =============================================================================
const char* to_string(RegimeFit fit) {
    switch (fit) {
        case RegimeFit::Trend: return "trend";
        case RegimeFit::Range: return "range";
        case RegimeFit::Both:  return "both";
    }
    return "both";
}

std::string StrategySignal::detail() const {
    return fmt("%s(dir=%+.2f conf=%.2f fit=%s)", name.c_str(), direction,
               confidence, to_string(regime_fit));
}

// =============================================================================
// StrategySeries
// =============================================================================
StrategySeries::StrategySeries(const std::vector<Bar>& bars, const StrategyConfig& cfg)
    : cfg_(cfg), bars_(bars), n_(bars.size()) {
    ts_.reserve(n_);
    for (const Bar& b : bars_) ts_.push_back(b.time);

    close_ = features::closes(bars_);
    vol_ = features::volumes(bars_);

    sma_fast_ = features::sma(close_, cfg_.sma_fast);
    sma_slow_ = features::sma(close_, cfg_.sma_slow);
    sma_z_ = features::sma(close_, cfg_.z_window);
    std_z_ = features::rolling_std(close_, cfg_.z_window);
    donchian(bars_, cfg_.donchian_n, don_up_, don_lo_);

    bb_mid_ = features::sma(close_, cfg_.bb_period);
    bb_std_ = features::rolling_std(close_, cfg_.bb_period);
    bb_up_ = nan_list(n_);
    bb_lo_ = nan_list(n_);
    for (size_t i = 0; i < n_; ++i) {
        if (fin(bb_mid_[i]) && fin(bb_std_[i])) {
            bb_up_[i] = bb_mid_[i] + cfg_.bb_sigma * bb_std_[i];
            bb_lo_[i] = bb_mid_[i] - cfg_.bb_sigma * bb_std_[i];
        }
    }

    std::vector<double> kc_mid;
    keltner(bars_, cfg_.kc_period, cfg_.kc_mult, kc_mid, kc_up_, kc_lo_);

    ofi_ = ofi_proxy(bars_, cfg_.ofi_window);
    ofi_z_ = features::zscore_series(ofi_, cfg_.ofi_z_window);
    atr_ = features::atr(bars_, 14);
}

StrategySeries::StrategySeries(const std::vector<Bar>& bars)
    : StrategySeries(bars, StrategyConfig()) {}

// =============================================================================
// Donchian / Keltner
// =============================================================================
void donchian(const std::vector<Bar>& bars, int32_t n, std::vector<double>& up,
              std::vector<double>& lo) {
    const size_t m = bars.size();
    up.assign(m, kRegimeNaN);
    lo.assign(m, kRegimeNaN);
    if (n <= 0) return;
    for (size_t i = static_cast<size_t>(n); i < m; ++i) {
        // Strictly PRIOR bars: [i-n, i-1]. Never includes i itself.
        double hi = bars[i - static_cast<size_t>(n)].high;
        double lw = bars[i - static_cast<size_t>(n)].low;
        for (int32_t k = 1; k < n; ++k) {
            const Bar& b = bars[i - static_cast<size_t>(n) + static_cast<size_t>(k)];
            hi = std::max(hi, b.high);
            lw = std::min(lw, b.low);
        }
        up[i] = hi;
        lo[i] = lw;
    }
}

void keltner(const std::vector<Bar>& bars, int32_t period, double mult,
             std::vector<double>& mid, std::vector<double>& up,
             std::vector<double>& lo) {
    const size_t m = bars.size();
    const std::vector<double> c = features::closes(bars);
    mid = features::ema(c, period);
    const std::vector<double> a = features::atr(bars, period);
    up.assign(m, kRegimeNaN);
    lo.assign(m, kRegimeNaN);
    for (size_t i = 0; i < m; ++i) {
        if (fin(mid[i]) && fin(a[i])) {
            up[i] = mid[i] + mult * a[i];
            lo[i] = mid[i] - mult * a[i];
        }
    }
}

// =============================================================================
// OFI
// =============================================================================
double ofi_from_snapshots(const std::vector<Snapshot>& snaps) {
    double total = 0.0;
    for (size_t k = 1; k < snaps.size(); ++k) {
        const Snapshot& p = snaps[k - 1];
        const Snapshot& q = snaps[k];
        const double e_b = (q.bid_px >= p.bid_px ? q.bid_sz : 0.0) -
                           (q.bid_px <= p.bid_px ? p.bid_sz : 0.0);
        const double e_a = (q.ask_px <= p.ask_px ? q.ask_sz : 0.0) -
                           (q.ask_px >= p.ask_px ? p.ask_sz : 0.0);
        total += e_b - e_a;
    }
    return total;
}

double impact_beta(double depth, double c, double lam) {
    if (!fin(depth) || depth <= 0.0) return kRegimeNaN;
    return c / std::pow(depth, lam);
}

std::vector<double> ofi_proxy(const std::vector<Bar>& bars, int32_t n) {
    const std::vector<double> c = features::closes(bars);
    const std::vector<double> v = features::volumes(bars);
    const size_t m = bars.size();
    std::vector<double> sv(m, 0.0);
    for (size_t i = 1; i < m; ++i) {
        if (c[i] > c[i - 1]) sv[i] = v[i];
        else if (c[i] < c[i - 1]) sv[i] = -v[i];
    }
    std::vector<double> out = nan_list(m);
    if (n <= 0) return out;
    for (size_t i = static_cast<size_t>(n); i < m; ++i) {
        double num = 0.0, den = 0.0;
        for (int32_t k = 0; k < n; ++k) {
            const size_t j = i - static_cast<size_t>(n) + 1 + static_cast<size_t>(k);
            num += sv[j];
            den += v[j];
        }
        out[i] = den > 0.0 ? num / den : 0.0;
    }
    return out;
}

// =============================================================================
// The strategies
// =============================================================================
namespace {

StrategySignal make(const char* name, double dir, double conf, RegimeFit fit,
                    const std::string& reason) {
    StrategySignal s;
    s.name = name;
    s.direction = features::clampd(features::safe(dir, 0.0), -1.0, 1.0);
    s.confidence = features::clampd(features::safe(conf, 0.0), 0.0, 1.0);
    s.regime_fit = fit;
    s.reason = reason;
    return s;
}

// M4 — the option-chain reader, mirroring features.read_option_chain().
//
// The C++ Bar schema is (time, open, high, low, close, volume) and carries no
// option chain: slot 5 is VOLUME, so reading a chain off it would promote a
// 1e6-share bar to 1e6 implied vol and every vol strategy would fire. The
// reader therefore returns exactly what the oracle returns on a tuple bar —
// every greek NaN, term_structure and gex floored to zero — and the six vol
// strategies deactivate honestly instead of guessing. The day Bar gains chain
// fields, this is the ONE place that changes.
struct OptionChain {
    double iv_30d         = kRegimeNaN;
    double rv_30d         = kRegimeNaN;
    double call_iv_90pct  = kRegimeNaN;
    double put_iv_90pct   = kRegimeNaN;
    double skew_25d_delta = kRegimeNaN;
    double term_structure = 0.0;
    double gex            = 0.0;
};

OptionChain read_option_chain(const Bar& /*bar*/) {
    return OptionChain{};
}

} // namespace

StrategySignal momentum(const StrategySeries& s, size_t i,
                        const RegimeState* regime, const StrategyConfig& cfg) {
    // A breakout of the prior n-bar range is the oldest trend-following rule
    // there is (Donchian, 1960s). The ribbon filter is what stops it firing on
    // every other bar in a chop. (This strategy has no tunables of its own, so
    // `cfg` is unused here — kept in the signature so all four read alike.)
    (void)cfg;
    if (i < 1 || !fin(s.don_up()[i]) || !fin(s.don_lo()[i]))
        return make("momentum", 0.0, 0.0, RegimeFit::Trend, "no-channel");

    const double c = s.close()[i];
    const double up = s.don_up()[i], lo = s.don_lo()[i];
    const double rng = up - lo;
    if (rng <= 0.0)
        return make("momentum", 0.0, 0.0, RegimeFit::Trend, "flat-channel");

    int ribbon = 0;
    if (fin(s.sma_fast()[i]) && fin(s.sma_slow()[i]))
        ribbon = s.sma_fast()[i] > s.sma_slow()[i] ? 1 : -1;

    double d = 0.0, margin = 0.0;
    if (c > up) {
        d = 1.0;
        margin = (c - up) / rng;
    } else if (c < lo) {
        d = -1.0;
        margin = (lo - c) / rng;
    } else {
        return make("momentum", 0.0, 0.0, RegimeFit::Trend, "inside-range");
    }

    // Agreeing ribbon raises confidence; opposing ribbon quarters it.
    const double agree = (ribbon == static_cast<int>(d))
                             ? 1.0
                             : (ribbon == 0 ? 0.5 : 0.25);
    double conf = features::clampd(features::ramp(margin, 0.0, 0.25), 0.0, 1.0) * agree;
    if (regime != nullptr && fin(regime->trend_score)) {
        const double ts = features::safe(regime->trend_score, 0.5);
        conf *= features::clampd(0.5 + ts, 0.5, 1.5) / 1.5;
    }
    return make("momentum", d, conf, RegimeFit::Trend,
                fmt("donchian-breakout(margin=%.3f ribbon=%+d)", margin, ribbon));
}

StrategySignal mean_reversion(const StrategySeries& s, size_t i,
                              const RegimeState* regime, const StrategyConfig& cfg) {
    // Fade an extreme z-score, but ONLY where reversion is real and fast. The
    // OU half-life gate is the difference between "this is a mean-reverting
    // market" and "this has fallen a long way and I hope".
    if (i < 1 || !fin(s.sma_z()[i]) || !fin(s.std_z()[i]))
        return make("mean_reversion", 0.0, 0.0, RegimeFit::Range, "no-window");

    const double sd = s.std_z()[i];
    if (sd <= 0.0)
        return make("mean_reversion", 0.0, 0.0, RegimeFit::Range, "zero-sd");
    const double z = (s.close()[i] - s.sma_z()[i]) / sd;

    bool tradeable = true;
    double hl = kRegimeNaN;
    if (regime != nullptr) {
        tradeable = regime->mean_reversion_tradeable();
        hl = regime->half_life;
    }
    if (std::fabs(z) < cfg.z_entry)
        return make("mean_reversion", 0.0, 0.0, RegimeFit::Range,
                    fmt("z=%.2f below-entry", z));
    if (!tradeable)
        return make("mean_reversion", 0.0, 0.0, RegimeFit::Range,
                    fmt("z=%.2f not-reverting", z));

    const double d = z > 0.0 ? -1.0 : 1.0;
    double conf = features::clampd(std::fabs(z) / cfg.z_max, 0.0, 1.0);
    // A fast half-life is better evidence than a slow one.
    if (fin(hl) && hl > 0.0)
        conf *= features::clampd(1.25 - hl / 40.0, 0.4, 1.0);
    return make("mean_reversion", d, conf, RegimeFit::Range,
                fmt("fade(z=%.2f hl=%.0f)", z, features::safe(hl, -1.0)));
}

StrategySignal breakout(const StrategySeries& s, size_t i,
                        const RegimeState* /*regime*/, const StrategyConfig& cfg) {
    // Volatility squeeze (Bollinger inside Keltner) then expansion. The squeeze
    // says the market is coiling; the expansion says it chose a direction. A
    // band break out of nowhere is just noise, and is scored as such.
    if (i < 1) return make("breakout", 0.0, 0.0, RegimeFit::Both, "no-history");
    if (!(fin(s.bb_up()[i]) && fin(s.kc_up()[i])))
        return make("breakout", 0.0, 0.0, RegimeFit::Both, "no-bands");

    const double c = s.close()[i];
    bool squeeze = false;
    for (int32_t k = 1; k <= cfg.squeeze_lookback; ++k) {
        const int64_t j = static_cast<int64_t>(i) - k;
        if (j < 0) break;
        const size_t jj = static_cast<size_t>(j);
        if (fin(s.bb_up()[jj]) && fin(s.kc_up()[jj]) &&
            s.bb_up()[jj] < s.kc_up()[jj] && s.bb_lo()[jj] > s.kc_lo()[jj]) {
            squeeze = true;
            break;
        }
    }

    double d = 0.0, edge = 0.0;
    if (c > s.bb_up()[i]) {
        d = 1.0;
        edge = c - s.bb_up()[i];
    } else if (c < s.bb_lo()[i]) {
        d = -1.0;
        edge = s.bb_lo()[i] - c;
    } else {
        return make("breakout", 0.0, 0.0, RegimeFit::Both, "inside-bands");
    }

    const double width = std::max(s.bb_up()[i] - s.bb_lo()[i], 1e-12);
    double conf = features::clampd(features::ramp(edge / width, 0.0, 0.30), 0.0, 1.0);
    if (squeeze) conf = features::clampd(conf * 1.6, 0.0, 1.0);
    else conf *= 0.4;
    return make("breakout", d, conf, RegimeFit::Both,
                fmt("band-break(squeeze=%d)", squeeze ? 1 : 0));
}

StrategySignal ofi(const StrategySeries& s, size_t i, const RegimeState* /*regime*/,
                   const StrategyConfig& cfg, const std::vector<Snapshot>* snapshots) {
    // With a quote book we use the EXACT CKS OFI and beta = c/depth^lambda.
    // Without it we fall back to the signed-volume proxy and SAY SO in the
    // reason, because a proxy that looks like the real thing is a lie.
    if (snapshots != nullptr && !snapshots->empty()) {
        const double raw = ofi_from_snapshots(*snapshots);
        double depth_sum = 0.0;
        for (const Snapshot& sn : *snapshots) depth_sum += (sn.bid_sz + sn.ask_sz) / 2.0;
        const double depth = depth_sum / static_cast<double>(snapshots->size());
        const double beta = impact_beta(depth, 1.0, 1.0);
        if (!fin(beta) || beta <= 0.0)
            return make("ofi", 0.0, 0.0, RegimeFit::Both, "no-depth");
        const double exp_move = beta * raw;
        const double d = exp_move > 0.0 ? 1.0 : (exp_move < 0.0 ? -1.0 : 0.0);
        const double conf = features::clampd(
            features::ramp(std::fabs(raw) / std::max(depth, 1e-12), 0.0, 1.0), 0.0, 1.0);
        return make("ofi", d, conf, RegimeFit::Both,
                    fmt("cks-ofi(raw=%.1f beta=%.2e)", raw, beta));
    }

    if (i < 1 || !fin(s.ofi()[i]) || !fin(s.ofi_z()[i]))
        return make("ofi", 0.0, 0.0, RegimeFit::Both, "no-flow");
    const double z = s.ofi_z()[i];
    if (std::fabs(z) < 1.0)
        return make("ofi", 0.0, 0.0, RegimeFit::Both, fmt("flow-flat(z=%.2f)", z));
    const double d = z > 0.0 ? 1.0 : -1.0;
    // The 0.7 is the PROXY discount: this object cannot see the book, so it
    // never gets to speak at full volume.
    const double conf = features::clampd(std::fabs(z) / cfg.ofi_z_full, 0.0, 1.0) * 0.7;
    return make("ofi", d, conf, RegimeFit::Both, fmt("ofi-proxy(z=%.2f NOT-cks)", z));
}

// =============================================================================
// M4 — volatility harvesting. The six strategies below mirror
// tools/monster/strategies.py branch-for-branch, so the parity test holds them
// to bit-exact agreement with the oracle. They read the option chain, and the
// chain reader (above) fails closed on a Bar that carries none — which is every
// Bar in this port — so each one deactivates and states why rather than
// inventing a greek.
// =============================================================================

StrategySignal vol_arb(const StrategySeries& s, size_t i,
                       const RegimeState* /*regime*/, const StrategyConfig& /*cfg*/) {
    // IV-RV spread arb: sell vol when IV overstates realized, buy when it
    // understates. ratio = iv/rv; 1.3 / 0.85 are the oracle's bands.
    if (i < 1) return make("vol_arb", 0.0, 0.0, RegimeFit::Both, "no-history");
    const OptionChain oc = read_option_chain(s.bars()[i]);
    const double iv = oc.iv_30d, rv = oc.rv_30d;
    if (!fin(iv) || !fin(rv) || rv <= 0.0)
        return make("vol_arb", 0.0, 0.0, RegimeFit::Both, "no-optchain");
    const double ratio = iv / rv;
    if (ratio > 1.3)
        return make("vol_arb", -1.0, (ratio - 1.0) / 1.0, RegimeFit::Both,
                    fmt("iv_rich ratio=%.2f", ratio));
    if (ratio < 0.85)
        return make("vol_arb", 1.0, (0.85 - ratio) / 0.85, RegimeFit::Both,
                    fmt("iv_cheap ratio=%.2f", ratio));
    return make("vol_arb", 0.0, 0.0, RegimeFit::Both,
                fmt("iv_rv_neutral ratio=%.2f", ratio));
}

StrategySignal butterfly_arb(const StrategySeries& s, size_t i,
                             const RegimeState* /*regime*/, const StrategyConfig& /*cfg*/) {
    // Vol surface curvature: short overpriced wings when the butterfly
    // violates. curvature = wing_avg - iv_atm; positive wings = violation.
    if (i < 1) return make("butterfly_arb", 0.0, 0.0, RegimeFit::Range, "no-history");
    const OptionChain oc = read_option_chain(s.bars()[i]);
    const double call_iv = oc.call_iv_90pct, put_iv = oc.put_iv_90pct;
    const double iv_atm = oc.iv_30d;
    if (!fin(call_iv) || !fin(put_iv) || !fin(iv_atm))
        return make("butterfly_arb", 0.0, 0.0, RegimeFit::Range, "no-optchain");
    const double wing_avg = (call_iv + put_iv) / 2.0;
    const double curvature = wing_avg - iv_atm;
    if (curvature > 0.02)
        return make("butterfly_arb", -1.0, curvature / 0.05, RegimeFit::Range,
                    fmt("wings-rich curvature=%.4f", curvature));
    if (curvature < -0.02)
        return make("butterfly_arb", 1.0, -curvature / 0.05, RegimeFit::Range,
                    fmt("wings-cheap curvature=%.4f", curvature));
    return make("butterfly_arb", 0.0, 0.0, RegimeFit::Range,
                fmt("curvature-neutral curvature=%.4f", curvature));
}

StrategySignal skew_trend(const StrategySeries& s, size_t i,
                          const RegimeState* /*regime*/, const StrategyConfig& /*cfg*/) {
    // Skew tilt: mean-revert an extreme put/call IV differential. Rich puts ->
    // sell the premium; cheap puts -> buy it.
    if (i < 1) return make("skew_trend", 0.0, 0.0, RegimeFit::Trend, "no-history");
    const double skew = read_option_chain(s.bars()[i]).skew_25d_delta;
    if (!fin(skew))
        return make("skew_trend", 0.0, 0.0, RegimeFit::Trend, "no-optchain");
    if (skew > 0.15)
        return make("skew_trend", -1.0, skew / 0.30, RegimeFit::Trend,
                    fmt("skew-rich skew=%.3f", skew));
    if (skew < -0.15)
        return make("skew_trend", 1.0, -skew / 0.30, RegimeFit::Trend,
                    fmt("skew-cheap skew=%.3f", skew));
    return make("skew_trend", 0.0, 0.0, RegimeFit::Trend,
                fmt("skew-neutral skew=%.3f", skew));
}

StrategySignal calendar_spread(const StrategySeries& s, size_t i,
                               const RegimeState* /*regime*/, const StrategyConfig& /*cfg*/) {
    // Term-structure arb: contango plus rich IV -> sell the front month;
    // backwardation plus cheap IV -> buy it.
    if (i < 1) return make("calendar_spread", 0.0, 0.0, RegimeFit::Both, "no-history");
    const OptionChain oc = read_option_chain(s.bars()[i]);
    const double ts = oc.term_structure, iv = oc.iv_30d, rv = oc.rv_30d;
    if (!fin(iv) || !fin(rv))
        return make("calendar_spread", 0.0, 0.0, RegimeFit::Both, "no-optchain");
    const double iv_rv_diff = iv - rv;
    if (ts > 0.0 && iv_rv_diff > 0.01)
        return make("calendar_spread", -1.0, iv_rv_diff / 0.05, RegimeFit::Both,
                    fmt("contango-iv-rich diff=%.3f", iv_rv_diff));
    if (ts < 0.0 && iv_rv_diff < -0.01)
        return make("calendar_spread", 1.0, -iv_rv_diff / 0.05, RegimeFit::Both,
                    fmt("backward-iv-cheap diff=%.3f", iv_rv_diff));
    return make("calendar_spread", 0.0, 0.0, RegimeFit::Both,
                fmt("no-arb ts=%+d diff=%.3f", static_cast<int>(ts), iv_rv_diff));
}

StrategySignal gex_regime(const StrategySeries& s, size_t i,
                          const RegimeState* /*regime*/, const StrategyConfig& /*cfg*/) {
    // Gamma exposure regime filter for vol harvesting. Positive GEX ->
    // range-bound, sell premium. Negative GEX -> trending, hedge rather than
    // harvest. The reader floors a chain-less bar at gex = 0, which is "no
    // information" rather than "balanced book", so the signal says so.
    if (i < 1) return make("gex_regime", 0.0, 0.0, RegimeFit::Range, "no-history");
    double gex = read_option_chain(s.bars()[i]).gex;
    if (!fin(gex)) gex = 0.0;
    if (gex > 0.5)
        return make("gex_regime", -1.0, gex / 2.0, RegimeFit::Range,
                    fmt("pos-gex sell-premium gex=%.2f", gex));
    if (gex < -0.5)
        return make("gex_regime", 0.5, -gex / 2.0, RegimeFit::Range,
                    fmt("neg-gex hedge gex=%.2f", gex));
    return make("gex_regime", 0.0, 0.0, RegimeFit::Range,
                fmt("gex-neutral gex=%.2f", gex));
}

StrategySignal var_swap(const StrategySeries& s, size_t i,
                        const RegimeState* /*regime*/, const StrategyConfig& /*cfg*/) {
    // Variance swap proxy: long variance when IV understates realized, short
    // when it overstates. var_gap = rv - iv; positive = variance is cheap.
    if (i < 1) return make("var_swap", 0.0, 0.0, RegimeFit::Both, "no-history");
    const OptionChain oc = read_option_chain(s.bars()[i]);
    const double iv = oc.iv_30d, rv = oc.rv_30d;
    if (!fin(iv) || !fin(rv))
        return make("var_swap", 0.0, 0.0, RegimeFit::Both, "no-optchain");
    const double var_gap = rv - iv;
    if (var_gap > 0.0001)
        return make("var_swap", 1.0, std::fabs(var_gap) / 0.001, RegimeFit::Both,
                    fmt("var-cheap gap=%.4f", var_gap));
    if (var_gap < -0.0001)
        return make("var_swap", -1.0, std::fabs(var_gap) / 0.001, RegimeFit::Both,
                    fmt("var-rich gap=%.4f", var_gap));
    return make("var_swap", 0.0, 0.0, RegimeFit::Both,
                fmt("var-fair gap=%.4f", var_gap));
}

std::vector<StrategySignal> all_signals(const StrategySeries& s, size_t i,
                                        const RegimeState* regime,
                                        const StrategyConfig& cfg,
                                        const std::vector<Snapshot>* snapshots) {
    std::vector<StrategySignal> out;
    out.reserve(10);
    out.push_back(momentum(s, i, regime, cfg));
    out.push_back(mean_reversion(s, i, regime, cfg));
    out.push_back(breakout(s, i, regime, cfg));
    out.push_back(ofi(s, i, regime, cfg, snapshots));
    out.push_back(vol_arb(s, i, regime, cfg));
    out.push_back(butterfly_arb(s, i, regime, cfg));
    out.push_back(skew_trend(s, i, regime, cfg));
    out.push_back(calendar_spread(s, i, regime, cfg));
    out.push_back(gex_regime(s, i, regime, cfg));
    out.push_back(var_swap(s, i, regime, cfg));
    return out;
}

std::vector<StrategySignal> all_signals(const StrategySeries& s, size_t i,
                                        const RegimeState* regime) {
    return all_signals(s, i, regime, s.cfg(), nullptr);
}

std::vector<StrategySignal> all_signals(const StrategySeries& s, size_t i,
                                        const RegimeState* regime,
                                        const std::vector<Snapshot>* snapshots) {
    return all_signals(s, i, regime, s.cfg(), snapshots);
}

// =============================================================================
// Volatility harvesting
// =============================================================================
double realized_variance(const std::vector<double>& vals, int32_t n) {
    std::vector<double> r;
    r.reserve(vals.size());
    for (size_t i = 1; i < vals.size(); ++i)
        if (vals[i] > 0.0 && vals[i - 1] > 0.0) r.push_back(std::log(vals[i] / vals[i - 1]));
    if (static_cast<int64_t>(r.size()) < n || n <= 0) return kRegimeNaN;
    const size_t start = r.size() - static_cast<size_t>(n);
    double m = 0.0;
    for (size_t i = start; i < r.size(); ++i) m += r[i];
    m /= n;
    double ss = 0.0;
    for (size_t i = start; i < r.size(); ++i) ss += (r[i] - m) * (r[i] - m);
    return ss / n;
}

VolTargetResult vol_target_scale(const StrategySeries& s, size_t i,
                                 const StrategyConfig& cfg) {
    VolTargetResult res;
    if (i < 1) return res;

    const std::vector<double>& c = s.close();
    const std::vector<double> upto(c.begin(), c.begin() + static_cast<ptrdiff_t>(i) + 1);
    res.rv_now = realized_variance(upto, cfg.vol_target_ref);
    res.rv_ref = realized_variance(upto, std::max(cfg.vol_target_ref * 4,
                                                  cfg.vol_target_ref + 1));
    if (!fin(res.rv_now) || !fin(res.rv_ref) || res.rv_now <= 0.0) return res;
    double scale = std::pow(res.rv_ref / res.rv_now, cfg.vol_target_power);
    res.scale = features::clampd(scale, cfg.vol_target_min_scale, cfg.vol_target_max_lev);
    return res;
}

} // namespace trading
} // namespace omniseed
