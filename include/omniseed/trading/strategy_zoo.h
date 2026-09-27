// =============================================================================
//  OmniSeed — trading/strategy_zoo.h
//  C++ port of tools/monster/features.py + tools/monster/strategies.py
//
//  WHY THIS EXISTS: the strategy zoo was Python-only, so the C++ runtime could
//  not ask "what do my strategies think?" without a separate process writing a
//  CSV. That is a seam, not an architecture. This is the zoo, in the runtime.
//
//  The Python original stays as the ORACLE for the parity test
//  (tests/test_strategy_parity.py); it is no longer the runtime path.
//
//  A ZOO, NOT ONE CLEVER MODEL. Momentum and mean reversion need OPPOSITE
//  conditions, so any single fixed strategy is wrong in half of all regimes.
//  router.h decides which to listen to, continuously.
//
//    momentum        Donchian breakout + MA ribbon          (trend-following)
//    mean_reversion  z-score fade, gated on OU half-life    (range)
//    breakout        Bollinger/Keltner squeeze -> expansion (transition)
//    ofi             order flow imbalance (Cont-Kukanov-Stoikov 2014)
//    vol_target      inverse-variance exposure scaling (Moreira-Muir 2017)
//                    -> a SIZING overlay, not a direction
//
//  THE OFI PATH IS THE INTERESTING ONE. CKS show price changes over short
//  intervals are driven by order flow imbalance, LINEARLY:
//      dP_k = beta * OFI_k + eps_k,        beta = c / depth^lambda
//  with a CONTEMPORANEOUS R^2 of 35-79% across 50 US stocks at 10s intervals.
//  Two honest caveats:
//    1. Contemporaneous is not predictive. OFI is not a crystal ball. The
//       tradable part is that OFI is autocorrelated and price impact is only
//       partly permanent, so next interval's flow is partly knowable.
//    2. It needs DEPTH. Free OHLCV feeds do not publish the book. Without it we
//       fall back to a signed-volume proxy, which is a *different and weaker*
//       object, and we say so in the signal reason rather than pretending.
//
//  NO LOOK-AHEAD: every strategy reads indices <= i. Donchian channels are built
//  from bars strictly BEFORE i (a breakout is "above the prior range", never
//  "above a range that includes itself"). tests/test_strategy_zoo.cpp proves
//  prefix invariance.
//
//  HONEST SCOPE: a conviction is not a probability of profit. See
//  docs/MONSTER_DESIGN.md.
// =============================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "omniseed/trading/regime_engine.h"   // RegimeState, kRegimeNaN
#include "omniseed/trading/trading_engine.h"  // Bar

namespace omniseed {
namespace trading {

// ---------------------------------------------------------------------------
// features — pure, causal indicator primitives (port of features.py)
//
// NO-LOOK-AHEAD CONTRACT: the value at index i uses ONLY inputs at indices <= i.
// Anything that structurally needs future bars (swing detection) is exposed as
// `find_swings` for offline analysis, and consumed through `confirmed_swings`,
// which only hands back swings whose confirmation bar (idx + right) is already
// in the past.
//
// Undefined values are NaN, never 0.0, so a caller can never mistake "not
// enough history" for a real reading.
// ---------------------------------------------------------------------------
namespace features {

double clampd(double x, double lo, double hi);
// 0.5 when the range is degenerate, NaN when x is not finite.
double ramp(double x, double lo, double hi);
double safe(double x, double d);

std::vector<double> closes(const std::vector<Bar>& bars);
std::vector<double> highs(const std::vector<Bar>& bars);
std::vector<double> lows(const std::vector<Bar>& bars);
std::vector<double> volumes(const std::vector<Bar>& bars);
std::vector<double> opens(const std::vector<Bar>& bars);

// Simple MA; NaN until `period` samples exist.
std::vector<double> sma(const std::vector<double>& vals, int32_t period);
// Exponential MA, seeded with the SMA of the first `period` samples.
std::vector<double> ema(const std::vector<double>& vals, int32_t period);
// Wilder's RSI. NaN until `period` changes exist.
std::vector<double> rsi(const std::vector<double>& vals, int32_t period);
void macd(const std::vector<double>& vals, int32_t fast, int32_t slow,
          int32_t signal_p, std::vector<double>& line, std::vector<double>& sig,
          std::vector<double>& hist);
std::vector<double> true_range(const std::vector<Bar>& bars);
// Wilder-smoothed Average True Range.
std::vector<double> atr(const std::vector<Bar>& bars, int32_t period);
// ATR as a fraction of close.
std::vector<double> atr_pct(const std::vector<Bar>& bars, int32_t period);
// Population std, trailing window (the band width the zoo actually uses).
std::vector<double> rolling_std(const std::vector<double>& vals, int32_t period);
void bollinger(const std::vector<double>& vals, int32_t period, double sigma,
               std::vector<double>& mid, std::vector<double>& up,
               std::vector<double>& lo);
// Trailing volume-weighted average of the typical price.
std::vector<double> rolling_vwap(const std::vector<Bar>& bars, int32_t period);
// z-score of the current sample vs the strictly PREVIOUS `period` samples.
std::vector<double> rolling_zscore(const std::vector<double>& vals, int32_t period);
// z-score vs a trailing window that tolerates NaNs inside it.
std::vector<double> zscore_series(const std::vector<double>& vals, int32_t period);
std::vector<double> rolling_corr(const std::vector<double>& a,
                                 const std::vector<double>& b, int32_t period);
// r_i = ln(v_i / v_{i-1}); index 0 is NaN.
std::vector<double> log_returns(const std::vector<double>& vals);
// Amihud (2002) illiquidity: |return| / dollar volume. Higher = thinner.
std::vector<double> amihud(const std::vector<Bar>& bars);
// Per-unit price-impact proxy: |dP| / volume (Lee-Ready sign cancels).
std::vector<double> kyle_lambda(const std::vector<Bar>& bars);

struct Swing {
    int32_t idx = 0;
    double  value = kRegimeNaN;
};

// All strict local minima of `vals` as (index, value).
//
// NOTE ON `kind`: the Python oracle's `find_swings(vals, left, right, kind)` has
// an INERT `kind` parameter — it always returns strict MINIMA, so
// `find_swings(highs, ..., "high")` returns minima of the high series, not swing
// highs. This port reproduces the oracle exactly (see `find_swings`) and also
// offers `find_local_maxima` for the correct object. Which one the sniper uses
// is a deliberate, reported decision — see tests/test_strategy_zoo.cpp.
std::vector<Swing> find_swings(const std::vector<double>& vals, int32_t left,
                               int32_t right);
// Strict local maxima — the correct "swing high". NOT what the oracle returns.
std::vector<Swing> find_local_maxima(const std::vector<double>& vals, int32_t left,
                                     int32_t right);
// Swings already knowable at index i (confirmation bar <= i).
std::vector<Swing> confirmed_swings(const std::vector<Swing>& swings, int32_t i,
                                    int32_t right);
// Retracement levels of an up-swing (buy-the-dip supports).
std::vector<double> fib_levels(double swing_low, double swing_high,
                               const std::vector<double>& ratios);
// Newest confirmed (low, high) with low_idx < high_idx. False when none.
bool last_swing_pair(const std::vector<Swing>& swing_lows,
                     const std::vector<Swing>& swing_highs, int32_t i,
                     int32_t right, Swing& out_low, Swing& out_high);
// Price lower-low + RSI higher-low across two confirmed swing lows.
bool bullish_divergence(const std::vector<double>& rsi_vals,
                        const std::vector<Swing>& swing_lows, int32_t i,
                        int32_t right, int32_t max_gap);

} // namespace features

// ---------------------------------------------------------------------------
// StrategySignal — a signed conviction plus the confidence in it
// ---------------------------------------------------------------------------
enum class RegimeFit : int8_t { Both = 0, Trend, Range };

const char* to_string(RegimeFit fit);

struct StrategySignal {
    std::string name;
    double      direction = 0.0;    // [-1, 1]
    double      confidence = 0.0;   // [0, 1]
    RegimeFit   regime_fit = RegimeFit::Both;
    std::string reason;

    // Active = there is something to say. Note the AND: a zero direction with
    // high confidence is still silence.
    bool active() const { return confidence > 0.0 && direction != 0.0; }
    std::string detail() const;
};

// Mirrors StrategyConfig in tools/monster/strategies.py field-for-field.
struct StrategyConfig {
    int32_t donchian_n = 20;
    int32_t sma_fast = 20;
    int32_t sma_slow = 50;
    int32_t z_window = 20;
    double  z_entry = 2.0;
    double  z_max = 3.5;
    int32_t bb_period = 20;
    double  bb_sigma = 2.0;
    int32_t kc_period = 20;
    double  kc_mult = 1.5;
    int32_t squeeze_lookback = 6;
    int32_t ofi_window = 20;
    int32_t ofi_z_window = 60;
    double  ofi_z_full = 3.0;
    double  vol_target_power = 2.0;
    int32_t vol_target_ref = 60;
    double  vol_target_max_lev = 2.0;
    double  vol_target_min_scale = 0.25;
};

// A best-quote snapshot: the ONLY input that gives the true CKS OFI.
struct Snapshot {
    double bid_px = 0.0, bid_sz = 0.0, ask_px = 0.0, ask_sz = 0.0;
};

// All precomputed series for one instrument. Index i is bar i.
class StrategySeries {
public:
    StrategySeries(const std::vector<Bar>& bars, const StrategyConfig& cfg);
    explicit StrategySeries(const std::vector<Bar>& bars);

    const StrategyConfig&        cfg() const { return cfg_; }
    const std::vector<Bar>&      bars() const { return bars_; }
    size_t                       n() const { return n_; }
    const std::vector<int64_t>&  ts() const { return ts_; }
    const std::vector<double>&   close() const { return close_; }
    const std::vector<double>&   vol() const { return vol_; }
    const std::vector<double>&   sma_fast() const { return sma_fast_; }
    const std::vector<double>&   sma_slow() const { return sma_slow_; }
    const std::vector<double>&   sma_z() const { return sma_z_; }
    const std::vector<double>&   std_z() const { return std_z_; }
    const std::vector<double>&   don_up() const { return don_up_; }
    const std::vector<double>&   don_lo() const { return don_lo_; }
    const std::vector<double>&   bb_mid() const { return bb_mid_; }
    const std::vector<double>&   bb_std() const { return bb_std_; }
    const std::vector<double>&   bb_up() const { return bb_up_; }
    const std::vector<double>&   bb_lo() const { return bb_lo_; }
    const std::vector<double>&   kc_up() const { return kc_up_; }
    const std::vector<double>&   kc_lo() const { return kc_lo_; }
    const std::vector<double>&   ofi() const { return ofi_; }
    const std::vector<double>&   ofi_z() const { return ofi_z_; }
    const std::vector<double>&   atr() const { return atr_; }

private:
    StrategyConfig       cfg_;
    std::vector<Bar>     bars_;
    size_t               n_ = 0;
    std::vector<int64_t> ts_;
    std::vector<double>  close_, vol_;
    std::vector<double>  sma_fast_, sma_slow_, sma_z_, std_z_;
    std::vector<double>  don_up_, don_lo_;
    std::vector<double>  bb_mid_, bb_std_, bb_up_, bb_lo_;
    std::vector<double>  kc_up_, kc_lo_;
    std::vector<double>  ofi_, ofi_z_, atr_;
};

// ---------------------------------------------------------------------------
// The strategies. `regime` may be null — then the regime gate is skipped, which
// is the honest behaviour: better to say nothing than to invent a context.
// ---------------------------------------------------------------------------
StrategySignal momentum(const StrategySeries& s, size_t i,
                        const RegimeState* regime, const StrategyConfig& cfg);
StrategySignal mean_reversion(const StrategySeries& s, size_t i,
                              const RegimeState* regime, const StrategyConfig& cfg);
StrategySignal breakout(const StrategySeries& s, size_t i,
                        const RegimeState* regime, const StrategyConfig& cfg);
StrategySignal ofi(const StrategySeries& s, size_t i, const RegimeState* regime,
                   const StrategyConfig& cfg, const std::vector<Snapshot>* snapshots);

// Every strategy's view of bar `i` (including the inactive ones), in the
// oracle's order: momentum, mean_reversion, breakout, ofi.
std::vector<StrategySignal> all_signals(const StrategySeries& s, size_t i,
                                        const RegimeState* regime,
                                        const StrategyConfig& cfg,
                                        const std::vector<Snapshot>* snapshots);

// Convenience overloads that use the series' own config.
std::vector<StrategySignal> all_signals(const StrategySeries& s, size_t i,
                                        const RegimeState* regime);
std::vector<StrategySignal> all_signals(const StrategySeries& s, size_t i,
                                        const RegimeState* regime,
                                        const std::vector<Snapshot>* snapshots);

// ---------------------------------------------------------------------------
// OFI primitives (Cont, Kukanov & Stoikov 2014)
// ---------------------------------------------------------------------------
// Exact CKS order flow imbalance from best-quote snapshots:
//   OFI = SUM over events of
//           1{Pb_n >= Pb_n-1}*qb_n - 1{Pb_n <= Pb_n-1}*qb_n-1
//         - 1{Pa_n <= Pa_n-1}*qa_n + 1{Pa_n >= Pa_n-1}*qa_n-1
// i.e. bid-side events enter with +, ask-side with -.
double ofi_from_snapshots(const std::vector<Snapshot>& snaps);
// CKS price impact coefficient beta = c / depth^lambda. NaN when depth <= 0.
double impact_beta(double depth, double c, double lam);
// Signed-volume proxy (Lee-Ready tick rule), for feeds with NO depth. This is a
// DIFFERENT, WEAKER object than the CKS OFI: it cannot see limit orders or
// cancellations, which is most of what OFI measures. Callers must label it.
std::vector<double> ofi_proxy(const std::vector<Bar>& bars, int32_t n);

// ---------------------------------------------------------------------------
// Donchian / Keltner (the breakout strategy's two channel families)
// ---------------------------------------------------------------------------
// up[i] = max high over the n bars BEFORE i — strictly prior bars, so
// "close > up" is a genuine breakout and not a self-fulfilling comparison
// against a window containing the current bar.
void donchian(const std::vector<Bar>& bars, int32_t n, std::vector<double>& up,
              std::vector<double>& lo);
// Keltner channels: EMA(close) +/- mult*ATR. The squeeze benchmark.
void keltner(const std::vector<Bar>& bars, int32_t period, double mult,
             std::vector<double>& mid, std::vector<double>& up,
             std::vector<double>& lo);

// ---------------------------------------------------------------------------
// VOLATILITY HARVESTING (Moreira & Muir 2017)
// ---------------------------------------------------------------------------
// Trailing realized variance of returns over the last `n` bars. NaN when there
// are fewer than `n` returns.
double realized_variance(const std::vector<double>& vals, int32_t n);

struct VolTargetResult {
    double scale = 1.0;
    double rv_now = kRegimeNaN;
    double rv_ref = kRegimeNaN;
};

// Moreira-Muir inverse-variance exposure scaler -> a multiplier in
// [vol_target_min_scale, vol_target_max_lev]. rv == rv_ref -> 1.0.
// `power=2` is the paper's variance version; `power=1` is inverse-vol.
VolTargetResult vol_target_scale(const StrategySeries& s, size_t i,
                                 const StrategyConfig& cfg);

} // namespace trading
} // namespace omniseed
