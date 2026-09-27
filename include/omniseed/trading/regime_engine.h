// =============================================================================
//  OmniSeed — trading/regime_engine.h
//  Multi-axis, CAUSAL market-regime engine — C++ port of tools/monster/regime.py
//
//  WHY THIS EXISTS: the Monster regime engine used to be Python-only, run as a
//  separate process that wrote a CSV the C++ engine then read back. That is a
//  seam, not an architecture: the model's own runtime could not answer "what
//  regime is this?" without shelling out. This is that engine, in the runtime.
//
//  The Python original stays as the ORACLE for the parity test
//  (tests/test_regime_parity.py); it is no longer the runtime path.
//
//  DIRECTIONAL AXIS — "trending or mean-reverting?"
//    variance ratio  VR(q) = Var(r^q) / (q*Var(r))     Lo & MacKinlay (1988)
//    Hurst exponent  E[R/S]_n ~ n^H                    Hurst (1951), R/S
//    efficiency ratio ER = |P_t - P_{t-n}| / sum|dP|   Kaufman (1995)
//    ADX / DMI       trend STRENGTH, not direction     Wilder (1978)
//    choppiness      100*log10(sum ATR/(maxH-minL))/log10(n)
//    linreg R^2      how line-like the window is
//    lag-1 autocorr  rho(1) of log returns
//  Each is normalized to [0,1] (0.5 = random walk) and blended into
//  `trend_score`. VR>1 / H>0.5 / rho>0 all say TREND; <1 / <0.5 / <0 say MEAN
//  REVERSION; the null is always the random walk.
//
//  VOLATILITY AXIS — the most reliably detectable regime (vol clusters).
//    Yang-Zhang (2000): gap-aware, ~14x more efficient than close-to-close at
//    the same sample size. Ranked against its OWN trailing history, so "high
//    vol" means high for this asset, not high vs the S&P.
//
//  TRADEABILITY — OU half-life from an OLS of dX on X_{t-1}:
//      half_life = -ln(2)/lambda     (lambda < 0 required)
//
//  NO LOOK-AHEAD: every statistic is computed on a window ENDING at index i.
//  `detect()` is pure. `RegimeTracker` adds hysteresis using only its own past
//  state. tests/test_regime_engine.cpp proves prefix invariance for each.
//
//  HONEST SCOPE: a regime label describes the recent past; it is not a
//  forecast. Nothing here claims an edge.
// =============================================================================
#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "omniseed/trading/trading_engine.h"   // Bar

namespace omniseed {
namespace trading {

// Every "missing" numeric is NaN, never a sentinel 0 — a 0 would be read as a
// real reading by the blend and would silently drag it toward neutral.
constexpr double kRegimeNaN = std::numeric_limits<double>::quiet_NaN();

inline bool regime_finite(double x) { return x == x && x != INFINITY && x != -INFINITY; }

// ---------------------------------------------------------------------------
// Config — thresholds, deliberately named rather than inline magic numbers.
// Mirrors RegimeConfig in tools/monster/regime.py field-for-field.
// ---------------------------------------------------------------------------
struct RegimeConfig {
    int32_t window = 200;          // lookback for the directional ensemble
    int32_t vr_q[3] = {2, 4, 8};   // horizons swept for the variance ratio
    int32_t vr_q_n = 3;
    int32_t er_n[3] = {10, 20, 40}; // ER is famously lookback-sensitive -> sweep
    int32_t er_n_n = 3;
    int32_t adx_period = 14;
    int32_t chop_n = 14;
    int32_t hurst_min_n = 8;
    int32_t rho_window = 60;
    int32_t yz_n = 20;
    int32_t vol_smooth = 5;        // a REGIME is sustained vol, not one big bar
    int32_t vol_hist = 120;        // trailing history for the vol percentile
    int32_t atr_period = 14;
    double  atr_hi = 0.05;         // ATR/close treated as stressed (the mandate's)
    int32_t ou_n = 60;
    int32_t sma_long = 200;

    // blend weights (sum ~1.0; renormalized over whichever are finite)
    double w_adx = 0.14;
    double w_er = 0.18;
    double w_chop = 0.13;
    double w_vr = 0.17;
    double w_hurst = 0.13;
    double w_r2 = 0.09;
    double w_rho = 0.04;
    double w_consist = 0.12;
    int32_t ma_fast = 20;
    int32_t ma_slow = 50;

    // label thresholds — CALIBRATED, not guessed. See regime.py for the
    // measured table (0.65 keeps 86% of true trend bars, cuts random-walk
    // false positives to 12.6%).
    double trend_hi = 0.65;
    double vol_stress = 0.90;
    // hysteresis: enter high, exit low, so the label does not flap on noise
    double trend_enter = 0.66;
    double trend_exit = 0.55;
    double vol_enter = 0.92;
    double vol_exit = 0.80;
};

// ---------------------------------------------------------------------------
// One bar's regime reading: continuous scores + a discretised label.
// ---------------------------------------------------------------------------
struct RegimeState {
    int64_t ts = 0;
    double  trend_score = kRegimeNaN;
    double  vol_score = kRegimeNaN;
    std::string label = "range";        // trend_up|trend_down|range|high_vol
    std::string direction;              // "up" | "down" | ""
    double  adx = kRegimeNaN;
    double  er = kRegimeNaN;
    double  chop = kRegimeNaN;
    double  vr = kRegimeNaN;
    double  vr_z = kRegimeNaN;
    double  hurst = kRegimeNaN;
    double  r2 = kRegimeNaN;
    double  rho1 = kRegimeNaN;
    double  yz_vol = kRegimeNaN;
    double  half_life = kRegimeNaN;
    bool    stressed = false;

    bool is_trend() const {
        return label == "trend_up" || label == "trend_down";
    }
    // Reverting, fast enough, AND corroborated by the ensemble. An OLS
    // half-life alone is not evidence of reversion: on a pure random walk it
    // routinely reports ~7 bars. Require a short half-life AND at least one
    // independent statistic to agree this is not a random walk.
    bool mean_reversion_tradeable() const {
        if (!regime_finite(half_life) || half_life > 30.0) return false;
        const bool vr_ok = regime_finite(vr) && vr < 0.95;
        const bool h_ok = regime_finite(hurst) && hurst < 0.45;
        return vr_ok || h_ok;
    }
    std::string detail() const;   // "regime=... trend=... vol=... [dir=.. hl=..]"
};

// ---------------------------------------------------------------------------
// The detector — pure, causal, no state.
// ---------------------------------------------------------------------------
class RegimeEngine {
public:
    explicit RegimeEngine(const RegimeConfig& cfg = RegimeConfig()) : cfg_(cfg) {}

    const RegimeConfig& config() const { return cfg_; }

    // Regime at bar `i`, using ONLY bars[max(0, i-window+1) .. i].
    RegimeState detect(const std::vector<Bar>& bars, size_t i) const;
    RegimeState detect(const std::vector<Bar>& bars, size_t i,
                       int32_t window) const;

    // Regime for every bar, raw (no hysteresis).
    std::vector<RegimeState> scan_raw(const std::vector<Bar>& bars) const;

private:
    RegimeConfig cfg_;
};

// ---------------------------------------------------------------------------
// Stateful tracker with hysteresis.
//
// Raw per-bar labels flap: a two-bar pullback flips `trend_up` to `range` and
// back, and every flip costs real spread. Enter high / exit low is the standard
// fix. Only ever reads bar i and its OWN past state, so it stays causal.
// ---------------------------------------------------------------------------
class RegimeTracker {
public:
    explicit RegimeTracker(const RegimeConfig& cfg = RegimeConfig())
        : engine_(cfg), cfg_(cfg) {}

    RegimeState step(const std::vector<Bar>& bars, size_t i);
    std::vector<RegimeState> run(const std::vector<Bar>& bars);
    void reset() { trend_ = false; vol_ = false; dir_.clear(); }

private:
    RegimeEngine engine_;
    RegimeConfig cfg_;
    bool        trend_ = false;
    bool        vol_ = false;
    std::string dir_;
};

// ---------------------------------------------------------------------------
// The statistics, exposed for testing (each is independently verifiable and is
// what tests/test_regime_engine.cpp pins against the Python oracle).
// ---------------------------------------------------------------------------
namespace regime_detail {

double clampd(double x, double lo, double hi);
// Linear 0..1 ramp: lo -> 0, hi -> 1. NaN -> NaN.
double ramp(double x, double lo, double hi);
// -> (slope, intercept, r2). r2 is 0 when the fit has no variance.
void ols(const std::vector<double>& xs, const std::vector<double>& ys,
         double& slope, double& intercept, double& r2);

double mean(const std::vector<double>& v);
double var_pop(const std::vector<double>& v);   // population variance

// Lo-MacKinlay variance ratio -> (VR(q), heteroskedasticity-robust z).
void variance_ratio(const std::vector<double>& returns, int32_t q,
                    double& vr_out, double& z_out);

// Rescaled-range Hurst. MUST be fed INCREMENTS, not price levels — on a
// non-stationary price path R/S reports H > 1, which is not a number Hurst can
// take. On returns the calibration is the textbook one (0.5 = random walk).
double hurst_rs(const std::vector<double>& vals, int32_t min_n = 8,
                int32_t num_sizes = 8);

double efficiency_ratio(const std::vector<double>& vals, int32_t n = 20);
double adx(const std::vector<Bar>& bars, int32_t period = 14);
double choppiness(const std::vector<Bar>& bars, int32_t n = 14);
void   linreg_trend(const std::vector<double>& vals, int32_t n,
                    double& slope_out, double& r2_out);
double autocorr1(const std::vector<double>& returns);
double trend_consistency(const std::vector<double>& vals, int32_t fast = 20,
                         int32_t slow = 50);
double yang_zhang_vol(const std::vector<Bar>& bars, int32_t n = 20);
double percentile_rank(const std::vector<double>& history, double value);
double ou_half_life(const std::vector<double>& vals, int32_t n = 60);

// Column extractors + moving average (mirrors tools/monster/features.py).
void   closes(const std::vector<Bar>& bars, std::vector<double>& out);
void   highs(const std::vector<Bar>& bars, std::vector<double>& out);
void   lows(const std::vector<Bar>& bars, std::vector<double>& out);
void   opens(const std::vector<Bar>& bars, std::vector<double>& out);
void   sma(const std::vector<double>& vals, int32_t period,
           std::vector<double>& out);
double atr_last(const std::vector<Bar>& bars, int32_t period);
void   true_range(const std::vector<Bar>& bars, std::vector<double>& out);

} // namespace regime_detail

} // namespace trading
} // namespace omniseed
