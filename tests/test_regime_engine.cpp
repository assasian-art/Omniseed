// =============================================================================
//  OmniSeed — tests/test_regime_engine.cpp
//
//  The C++ regime engine (src/trading/regime_engine.cpp), which replaces the
//  Python tools/monster/regime.py in the runtime path.
//
//  Part A — the statistics, on inputs whose right answer is known by hand.
//  Part B — the four bugs that were fixed in the Python. Each is pinned here so
//           the C++ port cannot silently reintroduce it. They are regressions,
//           not new work: the Python fixes are the specification.
//             B1  Hurst must run on RETURNS, not price levels
//             B2  ER must be swept across horizons, not read at one lookback
//             B3  high_vol needs ABSOLUTE stress too, not just a percentile
//             B4  (router) a single lone signal must not veto — see
//                 tests/test_monster_router.cpp once the router is ported
//  Part C — causality: detect() must be prefix-invariant (no look-ahead).
//  Part D — labels, hysteresis, and the tradeability gate.
//
//  Same tiny harness style as test_platform.cpp / test_decision_head.cpp.
//  Fully offline: no model, no GGUF, no network.
// =============================================================================
#include "omniseed/core/platform.h"
#include "omniseed/trading/regime_engine.h"
#include "omniseed/trading/trading_engine.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace omniseed;
using namespace omniseed::trading;
namespace rd = omniseed::trading::regime_detail;

static int g_passed = 0;
static int g_failed = 0;
static std::string g_current;

#define TEST(name) g_current = (name); platform::log_info("TEST  %s", name)
#define CHECK(cond)                                                        \
    do {                                                                   \
        if (cond) { ++g_passed; }                                          \
        else {                                                             \
            ++g_failed;                                                    \
            platform::log_error("FAIL  %s  (line %d): %s",                 \
                                g_current.c_str(), __LINE__, #cond);       \
        }                                                                  \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                              \
    do {                                                                   \
        const double _a = (a), _b = (b), _t = (tol);                       \
        if (std::fabs(_a - _b) <= _t) { ++g_passed; }                      \
        else {                                                             \
            ++g_failed;                                                    \
            platform::log_error("FAIL  %s  (line %d): |%g - %g| > %g",     \
                                g_current.c_str(), __LINE__, _a, _b, _t);  \
        }                                                                  \
    } while (0)

// NaN-aware equality: a component that is legitimately "missing" on both sides
// is a MATCH, not a mismatch. CHECK_NEAR cannot express that.
#define CHECK_SAME(a, b)                                                   \
    do {                                                                   \
        const double _a = (a), _b = (b);                                   \
        const bool _fa = regime_finite(_a), _fb = regime_finite(_b);       \
        const bool _ok = (!_fa && !_fb) || (_fa && _fb && _a == _b);       \
        if (_ok) { ++g_passed; }                                           \
        else {                                                             \
            ++g_failed;                                                    \
            platform::log_error("FAIL  %s  (line %d): %g != %g",           \
                                g_current.c_str(), __LINE__, _a, _b);      \
        }                                                                  \
    } while (0)

namespace {

// Deterministic LCG so every run is reproducible without <random>.
uint64_t g_seed = 0x2545F4914F6CDD1DULL;
double rnd() {
    g_seed = g_seed * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>((g_seed >> 11) & 0x1FFFFFFFFFFFFFULL) /
           static_cast<double>(0x1FFFFFFFFFFFFFULL);
}

Bar mk(int64_t ts, double o, double h, double l, double c, double v = 1000.0) {
    Bar b;
    b.time = ts; b.open = o; b.high = h; b.low = l; b.close = c; b.volume = v;
    return b;
}

// mode: "trend" | "chop" | "flat" | "quiet_late_spike"
std::vector<Bar> make_bars(size_t n, const std::string& mode, double noise) {
    std::vector<Bar> bars;
    bars.reserve(n);
    g_seed = 0x2545F4914F6CDD1DULL;   // reset for reproducibility
    double px = 100.0;
    for (size_t i = 0; i < n; ++i) {
        double drift = 0.0;
        if (mode == "trend") drift = 0.0025;
        else if (mode == "chop") {
            // oscillate around 100 so the path wanders but goes nowhere
            px = 100.0 + 2.0 * std::sin(static_cast<double>(i) * 0.7);
            drift = 0.0;
        } else if (mode == "flat") {
            px = 100.0;
            drift = 0.0;
        } else if (mode == "quiet_late_spike") {
            drift = 0.0002;
        }
        double amp = noise;
        if (mode == "quiet_late_spike" && i + 8 > n) amp = noise * 4.0;
        const double r = drift + amp * (rnd() - 0.5);
        const double o = px;
        const double c = px * (1.0 + r);
        const double h = std::max(o, c) * (1.0 + amp * 0.5);
        const double l = std::min(o, c) * (1.0 - amp * 0.5);
        bars.push_back(mk(1700000000 + static_cast<int64_t>(i) * 86400,
                          o, h, l, c));
        px = c;
    }
    return bars;
}

std::vector<double> logrets(const std::vector<Bar>& bars) {
    std::vector<double> cl, r;
    rd::closes(bars, cl);
    for (size_t i = 1; i < cl.size(); ++i)
        if (cl[i] > 0 && cl[i - 1] > 0) r.push_back(std::log(cl[i] / cl[i - 1]));
    return r;
}

double frac_high_vol(const std::vector<RegimeState>& st) {
    if (st.empty()) return 0.0;
    size_t k = 0;
    for (const RegimeState& s : st) if (s.label == "high_vol") ++k;
    return static_cast<double>(k) / static_cast<double>(st.size());
}

// ===========================================================================
// Part A — the statistics
// ===========================================================================
void test_statistics() {
    // --- OLS on a perfect line: slope 1, intercept 0, r2 1 -----------------
    {
        std::vector<double> xs = {0, 1, 2, 3, 4};
        std::vector<double> ys = {0, 1, 2, 3, 4};
        double s = 0, i = 0, r2 = 0;
        rd::ols(xs, ys, s, i, r2);
        CHECK_NEAR(s, 1.0, 1e-12);
        CHECK_NEAR(i, 0.0, 1e-12);
        CHECK_NEAR(r2, 1.0, 1e-12);
    }
    // --- OLS on a constant y: no variance -> r2 is 0, not NaN --------------
    {
        std::vector<double> xs = {0, 1, 2, 3};
        std::vector<double> ys = {5, 5, 5, 5};
        double s = 0, i = 0, r2 = 0;
        rd::ols(xs, ys, s, i, r2);
        CHECK_NEAR(s, 0.0, 1e-12);
        CHECK_NEAR(r2, 0.0, 1e-12);
    }
    // --- OLS on a vertical-ish x: sxx == 0 -> all NaN ---------------------
    {
        std::vector<double> xs = {1, 1, 1};
        std::vector<double> ys = {1, 2, 3};
        double s = 0, i = 0, r2 = 0;
        rd::ols(xs, ys, s, i, r2);
        CHECK(!regime_finite(s) && !regime_finite(r2));
    }

    // --- ramp: clamp + NaN passthrough ------------------------------------
    CHECK_NEAR(rd::ramp(0.0, 0.0, 1.0), 0.0, 1e-15);
    CHECK_NEAR(rd::ramp(1.0, 0.0, 1.0), 1.0, 1e-15);
    CHECK_NEAR(rd::ramp(5.0, 0.0, 1.0), 1.0, 1e-15);   // clamped high
    CHECK_NEAR(rd::ramp(-5.0, 0.0, 1.0), 0.0, 1e-15);  // clamped low
    CHECK_NEAR(rd::ramp(2.0, 2.0, 2.0), 0.5, 1e-15);   // degenerate -> 0.5
    CHECK(!regime_finite(rd::ramp(kRegimeNaN, 0.0, 1.0)));

    // --- sma: NaN until `period` samples exist ----------------------------
    {
        std::vector<double> v = {1, 2, 3, 4, 5};
        std::vector<double> out;
        rd::sma(v, 3, out);
        CHECK(!regime_finite(out[0]) && !regime_finite(out[1]));
        CHECK_NEAR(out[2], 2.0, 1e-12);
        CHECK_NEAR(out[4], 4.0, 1e-12);
    }

    // --- efficiency ratio: a straight line travels the whole path ---------
    {
        std::vector<double> line = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        CHECK_NEAR(rd::efficiency_ratio(line, 10), 1.0, 1e-12);
        std::vector<double> zig = {10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10};
        // net travel 0 -> ER 0
        CHECK_NEAR(rd::efficiency_ratio(zig, 10), 0.0, 1e-12);
    }

    // --- autocorr1: alternating returns are strongly negative -------------
    {
        std::vector<double> alt;
        for (int i = 0; i < 40; ++i) alt.push_back(i % 2 ? 0.01 : -0.01);
        CHECK(rd::autocorr1(alt) < -0.9);
    }

    // --- percentile rank --------------------------------------------------
    {
        std::vector<double> h = {1, 2, 3, 4, 5};
        CHECK_NEAR(rd::percentile_rank(h, 3.0), 0.6, 1e-12);  // 3 of 5 are <= 3
        CHECK_NEAR(rd::percentile_rank(h, 0.0), 0.0, 1e-12);
        CHECK_NEAR(rd::percentile_rank(h, 9.0), 1.0, 1e-12);
        CHECK(!regime_finite(rd::percentile_rank(h, kRegimeNaN)));
    }

    // --- variance ratio: positively autocorrelated returns -> VR > 1 ------
    {
        // NOTE: a CONSTANT drift plus iid noise does NOT give VR > 1 — the
        // overlapping sums of an iid series have variance exactly q*Var(r), so
        // VR ~ 1. (Measured on the oracle: 1.018.) Positive autocorrelation is
        // what makes VR exceed 1, so drive an AR(1) with phi > 0.
        std::vector<double> r;
        double prev = 0.0;
        for (int i = 0; i < 400; ++i) {
            prev = 0.5 * prev + 0.0002 * (rnd() - 0.5);
            r.push_back(prev);
        }
        double vr = 0, z = 0;
        rd::variance_ratio(r, 4, vr, z);
        CHECK(regime_finite(vr) && vr > 1.5);
    }

    // --- ADX: a clean trend reads high, chop reads low --------------------
    {
        auto trend = make_bars(160, "trend", 0.004);
        auto chop = make_bars(160, "chop", 0.004);
        const double a_trend = rd::adx(trend, 14);
        const double a_chop = rd::adx(chop, 14);
        CHECK(regime_finite(a_trend) && a_trend > 25.0);
        CHECK(regime_finite(a_chop) && a_chop < a_trend);
    }

    // --- choppiness: trend below 38.2, chop above -------------------------
    {
        auto trend = make_bars(120, "trend", 0.002);
        auto chop = make_bars(120, "chop", 0.002);
        const double c_trend = rd::choppiness(trend, 14);
        const double c_chop = rd::choppiness(chop, 14);
        CHECK(regime_finite(c_trend) && c_trend < 38.2);
        CHECK(regime_finite(c_chop) && c_chop > c_trend);
    }

    // --- OU half-life: strongly reverting series is short and finite ------
    {
        std::vector<double> x;
        double v = 0.0;
        for (int i = 0; i < 200; ++i) {
            v = 0.5 * v + 0.5 * (rnd() - 0.5);   // AR(1), phi = 0.5
            x.push_back(100.0 + v);
        }
        const double hl = rd::ou_half_life(x, 60);
        CHECK(regime_finite(hl) && hl > 0.0 && hl < 30.0);
    }
    // A monotone ramp does not revert -> NaN, not a bogus short half-life.
    {
        std::vector<double> up;
        for (int i = 0; i < 200; ++i) up.push_back(100.0 + i);
        CHECK(!regime_finite(rd::ou_half_life(up, 60)));
    }
}

// ===========================================================================
// Part B — regressions for the bugs fixed in the Python
// ===========================================================================
void test_bug_hurst_on_returns() {
    // B1. R/S on a non-stationary PRICE path measures the self-affinity of the
    // path, not long memory in a series: on a trend it reports a much higher
    // exponent than the returns do (measured on the real AAPL oracle: 0.97 on
    // levels vs 0.55 on returns). Feeding it levels is the bug.
    auto bars = make_bars(300, "trend", 0.004);
    std::vector<double> cl;
    rd::closes(bars, cl);

    const double h_levels = rd::hurst_rs(cl, 8, 8);           // the WRONG input
    const std::vector<double> rets = logrets(bars);
    const double h_rets = rd::hurst_rs(rets, 8, 8);           // the RIGHT one

    CHECK(regime_finite(h_levels));
    CHECK(regime_finite(h_rets));
    // The bug, demonstrated: levels read materially "more persistent".
    CHECK(h_levels > h_rets + 0.15);
    // The fix: returns produce a sane exponent in [0, 1].
    CHECK(h_rets >= 0.0 && h_rets <= 1.0);

    // And the engine uses the RETURNS reading — computed over its own window,
    // which is the last `window` bars, not the whole series.
    RegimeConfig cfg;
    RegimeEngine eng(cfg);
    const RegimeState st = eng.detect(bars, bars.size() - 1);

    const std::vector<Bar> win(bars.end() - static_cast<ptrdiff_t>(cfg.window),
                               bars.end());
    const double h_win = rd::hurst_rs(logrets(win), 8, 8);

    CHECK(regime_finite(st.hurst));
    CHECK(st.hurst >= 0.0 && st.hurst <= 1.0);
    CHECK_NEAR(st.hurst, h_win, 1e-9);
    // If someone reverts the fix and feeds it the window's LEVELS, this is what
    // the engine would have reported instead — and the assertion above fails.
    std::vector<double> win_cl;
    rd::closes(win, win_cl);
    const double h_win_levels = rd::hurst_rs(win_cl, 8, 8);
    CHECK(std::fabs(st.hurst - h_win_levels) > 0.05);
}

void test_bug_er_horizon_sweep() {
    // B2. A short pullback inside a long uptrend wipes out ER at the SHORT
    // horizon and would flip the whole label. Sweeping (10,20,40) and averaging
    // the NORMALIZED readings keeps the long-term trend visible.
    //
    // The pullback must be CHOPPY, not monotone: a monotone decline is a
    // perfectly efficient path and scores ER = 1.0. (Verified on the oracle.)
    std::vector<Bar> bars = make_bars(260, "trend", 0.002);
    double px = bars.back().close;
    for (int k = 0; k < 20; ++k) {
        const double o = px;
        px *= (k % 2 == 0) ? (1.0 - 0.006) : (1.0 + 0.0055);
        const double c = px;
        bars.push_back(mk(bars.back().time + 86400, o,
                          std::max(o, c) * 1.001,
                          std::min(o, c) * 0.999, c));
    }

    std::vector<double> cl;
    rd::closes(bars, cl);
    const double er20_only = rd::efficiency_ratio(cl, 20);

    RegimeConfig cfg;
    RegimeEngine eng(cfg);
    const RegimeState st = eng.detect(bars, bars.size() - 1);

    // The single short horizon really is wiped out by the choppy pullback...
    CHECK(regime_finite(er20_only) && er20_only < 0.10);
    // ...but the swept average is not, because ER(40) still spans the trend.
    CHECK(regime_finite(st.er));
    CHECK(st.er > er20_only);
    CHECK(st.er > 0.08);
    // The sweep must be the mean of the finite per-horizon readings.
    const double e10 = rd::efficiency_ratio(cl, 10);
    const double e40 = rd::efficiency_ratio(cl, 40);
    CHECK_NEAR(st.er, (e10 + er20_only + e40) / 3.0, 1e-9);
}

void test_bug_absolute_vol_confirmation() {
    // B3. Ranking the raw estimate against its own history marked ~15% of ALL
    // bars high_vol by construction — including entry bars, which are
    // vol-expansion bars by nature. With max S in high_vol = 0.8125 < 0.85
    // that made the sniper bar unreachable. high_vol now needs the ABSOLUTE
    // confirmation too (ATR/close reaching the mandate's atr_hi).

    // (a) Relatively elevated but ABSOLUTELY tiny volatility -> NOT stressed.
    //     ATR/close here is ~0.1%, far below atr_hi*0.5 = 2.5%.
    auto quiet = make_bars(300, "quiet_late_spike", 0.0015);
    RegimeConfig cfg;
    RegimeEngine eng(cfg);
    const RegimeState st = eng.detect(quiet, quiet.size() - 1);
    CHECK(st.label != "high_vol");
    CHECK(!st.stressed);

    // (b) Genuinely stressed: 8% daily swings -> ATR/close clears atr_hi.
    std::vector<Bar> wild;
    {
        double p = 100.0;
        for (int i = 0; i < 300; ++i) {
            const double o = p;
            const double c = p * (1.0 + (i % 2 ? 0.06 : -0.055));
            wild.push_back(mk(1700000000 + i * 86400, o,
                              std::max(o, c) * 1.03,
                              std::min(o, c) * 0.97, c));
            p = c;
        }
    }
    const RegimeState sw = eng.detect(wild, wild.size() - 1);
    CHECK(sw.stressed);
    CHECK(sw.label == "high_vol");

    // (c) And the label is not sprayed across a normal series.
    RegimeEngine eng2(cfg);
    const auto scan = eng2.scan_raw(make_bars(400, "trend", 0.004));
    CHECK(frac_high_vol(scan) < 0.10);
}

// ===========================================================================
// Part C — causality
// ===========================================================================
void test_prefix_invariance() {
    // detect(bars, i) must depend ONLY on bars[0..i]. If it peeked ahead, the
    // reading at i would change when later bars exist — which is exactly how a
    // backtest lies to you.
    auto full = make_bars(320, "trend", 0.005);
    RegimeConfig cfg;
    RegimeEngine eng(cfg);

    const size_t probes[] = {60, 120, 199, 200, 201, 250, 319};
    for (size_t i : probes) {
        const std::vector<Bar> prefix(full.begin(), full.begin() + static_cast<ptrdiff_t>(i) + 1);
        const RegimeState a = eng.detect(full, i);
        const RegimeState b = eng.detect(prefix, i);
        // Bitwise identical, with "both NaN" counting as a match.
        CHECK_SAME(a.trend_score, b.trend_score);
        CHECK_SAME(a.vol_score, b.vol_score);
        CHECK_SAME(a.adx, b.adx);
        CHECK_SAME(a.hurst, b.hurst);
        CHECK_SAME(a.half_life, b.half_life);
        CHECK_SAME(a.er, b.er);
        CHECK_SAME(a.chop, b.chop);
        CHECK_SAME(a.vr, b.vr);
        CHECK_SAME(a.r2, b.r2);
        CHECK_SAME(a.rho1, b.rho1);
        CHECK(a.label == b.label);
        CHECK(a.direction == b.direction);
        CHECK(a.stressed == b.stressed);
    }
}

void test_tracker_prefix_invariance() {
    // The hysteresis tracker may only use its own past state, so replaying a
    // prefix must produce the same label sequence.
    auto full = make_bars(300, "trend", 0.005);
    RegimeConfig cfg;
    RegimeTracker tr1(cfg);
    const auto a = tr1.run(full);

    const std::vector<Bar> prefix(full.begin(), full.begin() + 200);
    RegimeTracker tr2(cfg);
    const auto b = tr2.run(prefix);

    CHECK(a.size() == full.size());
    CHECK(b.size() == prefix.size());
    bool same = true;
    for (size_t i = 0; i < b.size(); ++i)
        if (a[i].label != b[i].label) same = false;
    CHECK(same);
}

// ===========================================================================
// Part D — labels, hysteresis, tradeability
// ===========================================================================
void test_labels_and_hysteresis() {
    RegimeConfig cfg;
    RegimeEngine eng(cfg);

    // A clean uptrend that also sits above its long SMA -> trend_up.
    auto trend = make_bars(400, "trend", 0.003);
    const RegimeState st = eng.detect(trend, trend.size() - 1);
    CHECK(st.trend_score > 0.5);
    CHECK(st.direction == "up");
    CHECK(st.label == "trend_up");

    // Chop -> not a trend label.
    auto chop = make_bars(400, "chop", 0.003);
    const RegimeState sc = eng.detect(chop, chop.size() - 1);
    CHECK(sc.label != "trend_up" && sc.label != "trend_down");

    // Too little history -> a neutral range, and trend_score is exactly 0.5.
    auto tiny = make_bars(10, "trend", 0.003);
    const RegimeState sm = eng.detect(tiny, tiny.size() - 1);
    CHECK(sm.label == "range");
    CHECK_NEAR(sm.trend_score, 0.5, 0.0);
    CHECK(!regime_finite(sm.vol_score));

    // Hysteresis: once latched into a trend, a short dip does not flip it back.
    {
        std::vector<Bar> b = make_bars(300, "trend", 0.003);
        RegimeTracker tracker(cfg);
        const auto plain = tracker.run(b);
        CHECK(plain.back().label == "trend_up");

        double px = b.back().close;
        for (int k = 0; k < 3; ++k) {     // a 3-bar dip
            const double o = px;
            const double c = px * 0.985;
            b.push_back(mk(b.back().time + 86400, o, o * 1.001, c * 0.999, c));
            px = c;
        }
        RegimeTracker t2(cfg);
        const auto latched = t2.run(b);
        CHECK(latched.back().label == "trend_up");   // stayed latched
    }

    // mean_reversion_tradeable requires corroboration, not just a short HL.
    {
        RegimeState s;
        s.half_life = 5.0;                 // short...
        s.vr = kRegimeNaN; s.hurst = kRegimeNaN;
        CHECK(!s.mean_reversion_tradeable());   // ...but nothing agrees
        s.vr = 0.80;                       // now the ensemble agrees
        CHECK(s.mean_reversion_tradeable());
        s.vr = kRegimeNaN; s.hurst = 0.30; // or Hurst does
        CHECK(s.mean_reversion_tradeable());
        s.hurst = 0.60;                    // neither does -> not tradeable
        CHECK(!s.mean_reversion_tradeable());
        s.vr = 0.80; s.half_life = 200.0;  // too slow to harvest
        CHECK(!s.mean_reversion_tradeable());
    }

    // detail() is a stable, readable string.
    {
        RegimeState s;
        s.label = "trend_up"; s.trend_score = 0.712; s.vol_score = 0.31;
        s.direction = "up";
        CHECK(s.detail() == "regime=trend_up trend=0.712 vol=0.31 dir=up");
        RegimeState nan_state;
        CHECK(nan_state.detail().find("trend=nan") != std::string::npos);
    }
}

}  // namespace

int main() {
    platform::log_info("=== omniseed regime engine tests ===");

    TEST("statistics on hand-checkable inputs");
    test_statistics();

    TEST("B1 regression: Hurst runs on returns, not price levels");
    test_bug_hurst_on_returns();

    TEST("B2 regression: ER is swept across horizons");
    test_bug_er_horizon_sweep();

    TEST("B3 regression: high_vol needs absolute stress confirmation");
    test_bug_absolute_vol_confirmation();

    TEST("C: detect() is prefix-invariant (no look-ahead)");
    test_prefix_invariance();
    test_tracker_prefix_invariance();

    TEST("D: labels, hysteresis, tradeability");
    test_labels_and_hysteresis();

    platform::log_info("---- regime engine tests: %d passed, %d failed ----",
                       g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
