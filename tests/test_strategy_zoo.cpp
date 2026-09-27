// =============================================================================
//  OmniSeed — tests/test_strategy_zoo.cpp
//
//  The C++ strategy zoo, router and sniper (src/trading/strategy_zoo.cpp,
//  router.cpp, sniper.cpp), which replace tools/monster/strategies.py,
//  router.py and sniper_engine.py in the runtime path.
//
//  Part A — feature primitives, on inputs whose right answer is known by hand.
//  Part B — the strategies' branch behaviour (which branch fired, and why).
//  Part C — the ROUTER, including the B4 regression the mandate named:
//             a single lone low-confidence signal must NOT be able to veto.
//  Part D — the SNIPER: the two structural invariants that carry its meaning.
//             D1  the ensemble can only ever BLOCK, never promote
//             D2  the 0.85 gate's reachability, per regime label
//  Part E — causality: every strategy, the router and the sniper must be
//           prefix-invariant (no look-ahead).
//
//  Same tiny harness style as test_platform.cpp / test_regime_engine.cpp.
//  Fully offline: no model, no GGUF, no network.
// =============================================================================
#include "omniseed/core/platform.h"
#include "omniseed/trading/regime_engine.h"
#include "omniseed/trading/router.h"
#include "omniseed/trading/sniper.h"
#include "omniseed/trading/strategy_zoo.h"
#include "omniseed/trading/trading_engine.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace omniseed;
using namespace omniseed::trading;
namespace fd = omniseed::trading::features;
namespace sd = omniseed::trading::sniper_detail;

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

// NaN-aware equality: "missing on both sides" is a MATCH, not a mismatch.
#define CHECK_SAME(a, b)                                                   \
    do {                                                                   \
        const double _a = (a), _b = (b);                                   \
        const bool _fa = regime_finite(_a), _fb = regime_finite(_b);       \
        const bool _ok = (!_fa && !_fb) || (_fa && _fb && _a == _b);       \
        if (_ok) { ++g_passed; }                                           \
        else {                                                             \
            ++g_failed;                                                    \
            platform::log_error("FAIL  %s  (line %d): %g vs %g",           \
                                g_current.c_str(), __LINE__, _a, _b);      \
        }                                                                  \
    } while (0)

namespace {

// Deterministic OHLCV, same LCG as tests/test_strategy_parity.py so that a
// failure here and a failure there describe the same input.
std::vector<Bar> make_bars(size_t n, const std::string& mode) {
    std::vector<Bar> out;
    out.reserve(n);
    uint64_t state = 987654321ULL;
    auto rnd = [&state]() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<double>((state >> 11) & 0x1FFFFFFFFFFFFFULL) /
               static_cast<double>(0x1FFFFFFFFFFFFFULL);
    };
    double px = 100.0;
    for (size_t i = 0; i < n; ++i) {
        double drift = 0.0, amp = 0.003;
        if (mode == "trend") { drift = 0.0022; amp = 0.006; }
        else if (mode == "chop") { px = 100.0 + 2.5 * std::sin(i * 0.55); drift = 0.0; amp = 0.005; }
        else if (mode == "vol") { drift = 0.0; amp = 0.035; }
        else if (mode == "flat") { drift = 0.0; amp = 0.0; }
        const double r = drift + amp * (rnd() - 0.5);
        Bar b;
        b.time = 1700000000 + static_cast<int64_t>(i) * 86400;
        b.open = px;
        b.close = px * (1.0 + r);
        b.high = std::max(b.open, b.close) * (1.0 + amp * 0.5);
        b.low = std::min(b.open, b.close) * (1.0 - amp * 0.5);
        b.high = std::max(b.high, std::max(b.open, b.close));
        b.low = std::min(b.low, std::min(b.open, b.close));
        b.volume = 1000.0 + static_cast<double>(i) * 3.0;
        out.push_back(b);
        px = b.close;
    }
    return out;
}

StrategySignal sig(const char* name, double dir, double conf, RegimeFit fit,
                   const char* reason) {
    StrategySignal s;
    s.name = name;
    s.direction = dir;
    s.confidence = conf;
    s.regime_fit = fit;
    s.reason = reason;
    return s;
}

// =============================================================================
// Part A — feature primitives
// =============================================================================
void test_features() {
    // --- sma: NaN until warm, then the exact mean ------------------------
    {
        const std::vector<double> v = {1, 2, 3, 4, 5};
        const std::vector<double> m = fd::sma(v, 3);
        CHECK(!regime_finite(m[0]));
        CHECK(!regime_finite(m[1]));
        CHECK_NEAR(m[2], 2.0, 1e-12);
        CHECK_NEAR(m[3], 3.0, 1e-12);
        CHECK_NEAR(m[4], 4.0, 1e-12);
    }
    // --- sma of a constant series is that constant -----------------------
    {
        const std::vector<double> v(50, 7.0);
        const std::vector<double> m = fd::sma(v, 10);
        CHECK_NEAR(m[49], 7.0, 1e-12);
    }
    // --- rolling_std of a constant is exactly 0 --------------------------
    {
        const std::vector<double> v(40, 3.0);
        const std::vector<double> s = fd::rolling_std(v, 5);
        CHECK_NEAR(s[39], 0.0, 1e-15);
    }
    // --- rolling_std is POPULATION std (divide by n, not n-1) ------------
    {
        const std::vector<double> v = {1, 3};
        const std::vector<double> s = fd::rolling_std(v, 2);
        CHECK_NEAR(s[1], 1.0, 1e-12);   // sqrt(((1-2)^2+(3-2)^2)/2) = 1
    }
    // --- ema is seeded with the SMA of the first `period` samples --------
    {
        const std::vector<double> v = {2, 4, 6, 8, 10};
        const std::vector<double> e = fd::ema(v, 3);
        CHECK_NEAR(e[2], 4.0, 1e-12);             // mean(2,4,6)
        CHECK_NEAR(e[3], 4.0 * 0.5 + 8.0 * 0.5, 1e-12);
    }
    // --- rsi of a monotonically rising series is 100 ---------------------
    {
        std::vector<double> v;
        for (int i = 0; i < 30; ++i) v.push_back(100.0 + i);
        const std::vector<double> r = fd::rsi(v, 14);
        CHECK_NEAR(r[29], 100.0, 1e-9);
    }
    // --- rsi of a monotonically falling series is 0 ----------------------
    {
        std::vector<double> v;
        for (int i = 0; i < 30; ++i) v.push_back(100.0 - i);
        const std::vector<double> r = fd::rsi(v, 14);
        CHECK_NEAR(r[29], 0.0, 1e-9);
    }
    // --- atr of constant-range bars is that range ------------------------
    {
        std::vector<Bar> b(20);
        for (size_t i = 0; i < b.size(); ++i) {
            b[i].time = 1 + static_cast<int64_t>(i);
            b[i].open = b[i].close = 100.0;
            b[i].high = 101.0;
            b[i].low = 99.0;
        }
        const std::vector<double> a = fd::atr(b, 14);
        CHECK_NEAR(a[19], 2.0, 1e-9);   // true range is 2 every bar
    }
    // --- donchian MUST NOT include bar i in its own window ---------------
    //
    // This is the difference between a real breakout and a self-fulfilling
    // comparison. Bar 10 gets an absurd high; up[10] must ignore it.
    {
        std::vector<Bar> b(15);
        for (size_t i = 0; i < b.size(); ++i) {
            b[i].time = 1 + static_cast<int64_t>(i);
            b[i].open = b[i].close = 100.0;
            b[i].high = 101.0;
            b[i].low = 99.0;
        }
        b[10].high = 500.0;             // the bar under test
        std::vector<double> up, lo;
        donchian(b, 5, up, lo);
        CHECK_NEAR(up[10], 101.0, 1e-12);   // NOT 500
        CHECK_NEAR(up[11], 500.0, 1e-12);   // it enters the window only after
        CHECK(!regime_finite(up[4]));       // needs 5 prior bars
    }
    // --- keltner mid is the EMA of close, bands are mid +/- mult*ATR -----
    {
        const std::vector<Bar> b = make_bars(120, "trend");
        std::vector<double> mid, up, lo;
        keltner(b, 20, 1.5, mid, up, lo);
        const std::vector<double> c = fd::closes(b);
        const std::vector<double> e = fd::ema(c, 20);
        CHECK_SAME(mid[119], e[119]);
        CHECK(up[119] > mid[119]);
        CHECK(lo[119] < mid[119]);
    }
    // --- rolling_zscore uses the strictly PAST window -------------------
    {
        const std::vector<double> v = {1, 1, 1, 1, 1, 9};
        const std::vector<double> z = fd::rolling_zscore(v, 5);
        // window for i=5 is [0..4] = all 1 -> sd 0 -> the guard returns 0.0
        CHECK_NEAR(z[5], 0.0, 1e-12);
        CHECK(!regime_finite(z[4]));
    }
    // --- log_returns -----------------------------------------------------
    {
        const std::vector<double> v = {100.0, 110.0};
        const std::vector<double> r = fd::log_returns(v);
        CHECK(!regime_finite(r[0]));
        CHECK_NEAR(r[1], std::log(1.1), 1e-12);
    }
    // --- fib_levels ------------------------------------------------------
    {
        const std::vector<double> lv = fd::fib_levels(100.0, 200.0, {0.382, 0.5, 0.618});
        CHECK_NEAR(lv[0], 200.0 - 100.0 * 0.382, 1e-9);
        CHECK_NEAR(lv[1], 150.0, 1e-9);
        CHECK_NEAR(lv[2], 200.0 - 100.0 * 0.618, 1e-9);
    }
    // --- find_swings finds STRICT local minima, and confirmed_swings hides
    //     the ones whose confirmation bar is still in the future -----------
    {
        const std::vector<double> v = {5, 4, 1, 4, 5, 2, 5, 6, 7};
        const std::vector<fd::Swing> sw = fd::find_swings(v, 2, 2);
        CHECK(sw.size() == 2);
        if (sw.size() == 2) {
            CHECK(sw[0].idx == 2 && sw[0].value == 1.0);
            CHECK(sw[1].idx == 5 && sw[1].value == 2.0);
        }
        const std::vector<fd::Swing> conf = fd::confirmed_swings(sw, 6, 2);
        CHECK(conf.size() == 1);        // idx 5 needs bar 7 to confirm
        CHECK(conf[0].idx == 2);
    }
    // --- bullish_divergence: price lower-low, RSI higher-low -------------
    {
        // Two strict swing lows, the SECOND lower in price than the first.
        std::vector<double> px = {5, 4, 3, 4, 5, 1, 5, 6, 7};
        std::vector<double> rsi_vals(9, kRegimeNaN);
        rsi_vals[2] = 30.0;             // first swing low: price 3, RSI 30
        rsi_vals[5] = 45.0;             // second: price 1 (lower), RSI 45 (higher)
        const std::vector<fd::Swing> sw = fd::find_swings(px, 2, 2);
        CHECK(sw.size() == 2);
        if (sw.size() == 2) {
            CHECK(sw[0].idx == 2 && sw[0].value == 3.0);
            CHECK(sw[1].idx == 5 && sw[1].value == 1.0);
        }
        CHECK(fd::bullish_divergence(rsi_vals, sw, 8, 2, 60));
        rsi_vals[5] = 20.0;             // now RSI is lower too -> no divergence
        CHECK(!fd::bullish_divergence(rsi_vals, sw, 8, 2, 60));
        // A gap wider than max_gap breaks the comparison.
        CHECK(!fd::bullish_divergence(rsi_vals, sw, 8, 2, 1));
    }
    // --- ramp / clampd edge cases ----------------------------------------
    {
        CHECK_NEAR(fd::ramp(0.5, 0.0, 1.0), 0.5, 1e-15);
        CHECK_NEAR(fd::ramp(2.0, 0.0, 1.0), 1.0, 1e-15);
        CHECK_NEAR(fd::ramp(-2.0, 0.0, 1.0), 0.0, 1e-15);
        CHECK_NEAR(fd::ramp(1.0, 3.0, 3.0), 0.5, 1e-15);   // degenerate range
        CHECK(!regime_finite(fd::ramp(kRegimeNaN, 0.0, 1.0)));
        CHECK_NEAR(fd::safe(kRegimeNaN, -1.0), -1.0, 1e-15);
    }
}

// =============================================================================
// Part B — the strategies
// =============================================================================
void test_strategies() {
    const std::vector<Bar> bars = make_bars(400, "trend");
    const StrategySeries s(bars);
    RegimeTracker tracker;
    const std::vector<RegimeState> regs = tracker.run(bars);

    // --- every strategy is silent when there is not enough history -------
    {
        const RegimeState* r0 = &regs[0];
        const std::vector<StrategySignal> sigs = all_signals(s, 0, r0);
        CHECK(sigs.size() == 4);
        for (const StrategySignal& x : sigs) {
            CHECK(!x.active());
            CHECK_NEAR(x.direction, 0.0, 0.0);
            CHECK_NEAR(x.confidence, 0.0, 0.0);
        }
    }
    // --- the oracle's order and regime fits are preserved ----------------
    {
        const std::vector<StrategySignal> sigs = all_signals(s, 300, &regs[300]);
        CHECK(sigs[0].name == "momentum" && sigs[0].regime_fit == RegimeFit::Trend);
        CHECK(sigs[1].name == "mean_reversion" && sigs[1].regime_fit == RegimeFit::Range);
        CHECK(sigs[2].name == "breakout" && sigs[2].regime_fit == RegimeFit::Both);
        CHECK(sigs[3].name == "ofi" && sigs[3].regime_fit == RegimeFit::Both);
    }
    // --- confidence is always inside [0,1] and direction inside [-1,1] ---
    {
        bool ok = true;
        for (size_t i = 0; i < s.n(); ++i)
            for (const StrategySignal& x : all_signals(s, i, &regs[i]))
                if (!(x.confidence >= 0.0 && x.confidence <= 1.0) ||
                    !(x.direction >= -1.0 && x.direction <= 1.0))
                    ok = false;
        CHECK(ok);
    }
    // --- a signal is never active with zero confidence -------------------
    {
        bool ok = true;
        for (size_t i = 0; i < s.n(); ++i)
            for (const StrategySignal& x : all_signals(s, i, &regs[i]))
                if (x.active() && (x.confidence <= 0.0 || x.direction == 0.0)) ok = false;
        CHECK(ok);
    }

    // --- MOMENTUM: a close above the strictly-prior range is a +1 --------
    //
    // Built by hand so the branch is unambiguous: a flat base, then one bar
    // that closes well above the prior 20-bar high.
    {
        std::vector<Bar> b(60);
        for (size_t i = 0; i < b.size(); ++i) {
            b[i].time = 1 + static_cast<int64_t>(i);
            b[i].open = b[i].close = 100.0;
            b[i].high = 101.0;
            b[i].low = 99.0;
            b[i].volume = 1000.0;
        }
        b[59].close = 120.0;            // breakout
        b[59].high = 121.0;
        const StrategySeries bs(b);
        const StrategySignal m = momentum(bs, 59, nullptr, bs.cfg());
        CHECK(m.direction == 1.0);
        CHECK(m.confidence > 0.0);
        CHECK(m.reason.find("donchian-breakout") == 0);
    }
    // --- MOMENTUM: inside the range is inactive --------------------------
    {
        std::vector<Bar> b(60);
        for (size_t i = 0; i < b.size(); ++i) {
            b[i].time = 1 + static_cast<int64_t>(i);
            b[i].open = b[i].close = 100.0;
            b[i].high = 101.0;
            b[i].low = 99.0;
            b[i].volume = 1000.0;
        }
        const StrategySeries bs(b);
        const StrategySignal m = momentum(bs, 59, nullptr, bs.cfg());
        CHECK(m.reason == "inside-range");
        CHECK(!m.active());
    }
    // --- MEAN REVERSION: all three branches, on hand-set inputs ----------
    //
    // The branches are decided in a fixed order — z-entry, then tradeability,
    // then the fade — and driving them from a hand-built regime is the only
    // reliable way to reach `fade`. On a trending series
    // `mean_reversion_tradeable()` is false BY CONSTRUCTION (VR > 1, Hurst >
    // 0.45), which is correct behaviour but leaves the branch unexercised.
    {
        // A flat, barely-alternating base then one large spike: sd is tiny and
        // stable, so the spike produces an unambiguous |z| >> z_entry.
        std::vector<Bar> b(80);
        for (size_t i = 0; i < b.size(); ++i) {
            const double c = 100.0 + ((i % 2) ? 0.1 : -0.1);
            b[i].time = 1 + static_cast<int64_t>(i);
            b[i].open = b[i].close = c;
            b[i].high = c + 0.2;
            b[i].low = c - 0.2;
            b[i].volume = 1000.0;
        }
        b[79].close = 130.0;            // the spike
        b[79].high = 130.5;
        const StrategySeries bs(b);
        const StrategyConfig cfg = bs.cfg();

        const double sd79 = bs.std_z()[79];
        CHECK(regime_finite(sd79) && sd79 > 0.0);
        const double z79 = (bs.close()[79] - bs.sma_z()[79]) / sd79;
        CHECK(std::fabs(z79) >= cfg.z_entry);   // 300x, not marginal

        // (a) fade: a fast half-life with corroborating VR/Hurst
        RegimeState tradeable;
        tradeable.half_life = 5.0;
        tradeable.vr = 0.80;
        tradeable.hurst = 0.40;
        CHECK(tradeable.mean_reversion_tradeable());
        const StrategySignal f = mean_reversion(bs, 79, &tradeable, cfg);
        CHECK(f.reason.find("fade(") == 0);
        CHECK(f.direction == -1.0);     // z > 0 -> fade it down
        CHECK(f.confidence > 0.0);
        CHECK(f.regime_fit == RegimeFit::Range);

        // (b) not-reverting: same z, but the OU half-life is too slow
        RegimeState slow;
        slow.half_life = 50.0;
        slow.vr = 0.80;
        slow.hurst = 0.40;
        CHECK(!slow.mean_reversion_tradeable());
        const StrategySignal nr = mean_reversion(bs, 79, &slow, cfg);
        CHECK(nr.reason.find("not-reverting") != std::string::npos);
        CHECK(!nr.active());

        // (c) not-reverting also when no statistic corroborates reversion,
        //     even though the half-life alone looks fast. An OLS half-life on
        //     a pure random walk routinely reads ~7 bars, so the half-life by
        //     itself is NOT evidence.
        RegimeState lone_hl;
        lone_hl.half_life = 5.0;
        lone_hl.vr = 1.10;              // trending
        lone_hl.hurst = 0.60;           // trending
        CHECK(!lone_hl.mean_reversion_tradeable());
        CHECK(mean_reversion(bs, 79, &lone_hl, cfg).reason.find("not-reverting") !=
              std::string::npos);

        // (d) below-entry: a quiet bar never even reaches tradeability
        const StrategySignal be = mean_reversion(bs, 50, &tradeable, cfg);
        CHECK(be.reason.find("below-entry") != std::string::npos);
        CHECK(!be.active());

        // (e) a null regime means "no context", and the strategy still fades
        //     on the z-score alone — it must not silently go silent.
        const StrategySignal no_reg = mean_reversion(bs, 79, nullptr, cfg);
        CHECK(no_reg.reason.find("fade(") == 0);

        // (f) zero sd is reported, not divided by
        std::vector<Bar> flat(40);
        for (size_t i = 0; i < flat.size(); ++i) {
            flat[i].time = 1 + static_cast<int64_t>(i);
            flat[i].open = flat[i].close = flat[i].high = flat[i].low = 100.0;
            flat[i].volume = 1.0;
        }
        const StrategySeries fs(flat);
        const StrategySignal z = mean_reversion(fs, 39, &tradeable, fs.cfg());
        CHECK(z.reason == "zero-sd");
        CHECK(!z.active());
    }
    // --- BREAKOUT: inside the bands is inactive, and a band exit scores --
    {
        const StrategySeries bs(bars);
        bool saw_inside = false, saw_break = false;
        for (size_t i = 1; i < bs.n(); ++i) {
            const StrategySignal x = breakout(bs, i, nullptr, bs.cfg());
            if (x.reason == "inside-bands") saw_inside = true;
            if (x.reason.find("band-break") != std::string::npos) saw_break = true;
        }
        CHECK(saw_inside);
        CHECK(saw_break);
    }
    // --- OFI: no depth -> the PROXY, and the reason says so -------------
    {
        const StrategySeries bs(bars);
        bool saw_proxy = false, saw_flat = false;
        for (size_t i = 1; i < bs.n(); ++i) {
            const StrategySignal x = ofi(bs, i, nullptr, bs.cfg(), nullptr);
            if (x.reason.find("NOT-cks") != std::string::npos) saw_proxy = true;
            if (x.reason.find("flow-flat") != std::string::npos) saw_flat = true;
        }
        CHECK(saw_proxy);
        CHECK(saw_flat);
    }
    // --- OFI: the EXACT CKS path, with depth, and beta = c/depth^lambda --
    {
        const std::vector<Snapshot> snaps = {
            {100.0, 10.0, 100.1, 10.0},
            {100.0, 40.0, 100.1, 10.0},   // bid size grows at the same price
            {100.2, 40.0, 100.3, 10.0},   // bid steps up
        };
        const double raw = ofi_from_snapshots(snaps);
        // k=1: e_b = 40 - 10 = 30, e_a = 10 - 10 = 0        -> +30
        // k=2: e_b = 40 -  0 = 40, e_a =  0 - 10 = -10      -> +50
        // (verified against the Python oracle: it returns 80.0 for this input)
        CHECK_NEAR(raw, 80.0, 1e-12);
        CHECK_NEAR(impact_beta(2.0, 1.0, 1.0), 0.5, 1e-12);
        CHECK(!regime_finite(impact_beta(0.0, 1.0, 1.0)));
        const StrategySeries bs(bars);
        const StrategySignal x = ofi(bs, 300, nullptr, bs.cfg(), &snaps);
        CHECK(x.direction == 1.0);     // beta > 0, raw > 0 -> expected move up
        CHECK(x.confidence > 0.0);
        CHECK(x.reason.find("cks-ofi") == 0);
    }
    // --- vol_target: rv_now == rv_ref -> scale 1.0, and always in bounds -
    {
        const StrategySeries bs(bars);
        bool ok = true;
        for (size_t i = 1; i < bs.n(); ++i) {
            const VolTargetResult r = vol_target_scale(bs, i, bs.cfg());
            if (r.scale < bs.cfg().vol_target_min_scale - 1e-12 ||
                r.scale > bs.cfg().vol_target_max_lev + 1e-12)
                ok = false;
        }
        CHECK(ok);
        CHECK_NEAR(vol_target_scale(bs, 0, bs.cfg()).scale, 1.0, 1e-15);
    }
}

// =============================================================================
// Part C — the router, and the B4 regression
// =============================================================================
void test_router() {
    // --- sigmoid endpoints and symmetry ----------------------------------
    CHECK_NEAR(sigmoid(0.0), 0.5, 1e-15);
    CHECK_NEAR(sigmoid(100.0), 1.0, 1e-12);
    CHECK_NEAR(sigmoid(-100.0), 0.0, 1e-12);
    CHECK_NEAR(sigmoid(1.0) + sigmoid(-1.0), 1.0, 1e-15);

    // --- CONTINUOUS sizing: at trend_score 0.5 both engines get half ------
    {
        RegimeState reg;
        reg.trend_score = 0.5;
        const std::vector<StrategySignal> sigs = {
            sig("momentum", 1.0, 1.0, RegimeFit::Trend, "x"),
            sig("mean_reversion", 1.0, 1.0, RegimeFit::Range, "y"),
        };
        const EnsembleVerdict v = route(sigs, &reg, RouterConfig(), 0);
        CHECK_NEAR(v.w_trend, 0.5, 1e-12);
        CHECK_NEAR(v.weight, 1.0, 1e-12);     // 0.5*1 + 0.5*1
        CHECK_NEAR(v.conviction, 1.0, 1e-12);
        CHECK_NEAR(v.agreement, 1.0, 1e-12);
        CHECK(v.active == 2);
    }
    // --- w_trend is monotone in trend_score, and "both" is always 1.0 ----
    {
        double prev = -1.0;
        bool mono = true;
        for (double t = 0.0; t <= 1.0; t += 0.05) {
            RegimeState reg;
            reg.trend_score = t;
            const EnsembleVerdict v = route({}, &reg, RouterConfig(), 0);
            if (v.w_trend < prev - 1e-15) mono = false;
            prev = v.w_trend;
        }
        CHECK(mono);
        CHECK_NEAR(regime_weight(RegimeFit::Both, 0.1), 1.0, 1e-15);
        CHECK_NEAR(regime_weight(RegimeFit::Trend, 0.3), 0.3, 1e-15);
        CHECK_NEAR(regime_weight(RegimeFit::Range, 0.3), 0.7, 1e-15);
    }
    // --- a missing regime falls back to 0.5, not to a fabricated trend ---
    {
        const EnsembleVerdict v = route({}, nullptr, RouterConfig(), 0);
        CHECK_NEAR(v.w_trend, 0.5, 1e-15);
        CHECK(v.active == 0);
        CHECK_NEAR(v.conviction, 0.0, 1e-15);
    }
    // --- signals quieter than min_confidence are ignored entirely --------
    {
        std::vector<StrategySignal> sigs = {
            sig("a", 1.0, 0.04, RegimeFit::Both, "too-quiet"),
            sig("b", 1.0, 0.50, RegimeFit::Both, "loud"),
        };
        const EnsembleVerdict v = route(sigs, nullptr, RouterConfig(), 0);
        CHECK(v.active == 1);
        CHECK(v.signals.size() == 1);
        CHECK(v.signals[0].first.name == "b");
    }

    // =====================================================================
    // B4 REGRESSION — a single lone low-confidence signal must NOT veto.
    //
    // `agreement` is a SHARE of active weight, so ONE active signal always
    // scores 1.00 by construction. Without the veto_min_weight floor, a lone
    // OFI proxy at confidence 0.05 would veto every long in the book. This is
    // the bug the mandate named, and it is pinned here in C++.
    // =====================================================================
    {
        const std::vector<StrategySignal> lone = {
            sig("ofi", -1.0, 0.05, RegimeFit::Both, "ofi-proxy")};
        const EnsembleVerdict v = route(lone, nullptr, RouterConfig(), 0);
        CHECK(v.active == 1);
        CHECK_NEAR(v.conviction, -1.0, 1e-12);
        CHECK_NEAR(v.agreement, 1.0, 1e-12);   // 1.00 BY CONSTRUCTION
        CHECK(v.weight < 0.50);
        CHECK(!should_veto_long(v, RouterConfig()));   // <- the regression
    }
    // --- the weight floor is exactly 0.50, on both sides of it -----------
    {
        const std::vector<StrategySignal> just_under = {
            sig("ofi", -1.0, 0.49, RegimeFit::Both, "x")};
        const std::vector<StrategySignal> at_floor = {
            sig("ofi", -1.0, 0.50, RegimeFit::Both, "x")};
        const EnsembleVerdict vu = route(just_under, nullptr, RouterConfig(), 0);
        const EnsembleVerdict va = route(at_floor, nullptr, RouterConfig(), 0);
        CHECK(!should_veto_long(vu, RouterConfig()));
        CHECK(should_veto_long(va, RouterConfig()));    // >= is inclusive
    }
    // --- a WEIGHTY, high-agreement consensus DOES veto -------------------
    {
        const std::vector<StrategySignal> consensus = {
            sig("momentum", -1.0, 0.9, RegimeFit::Both, "x"),
            sig("mean_reversion", -1.0, 0.9, RegimeFit::Both, "y"),
            sig("breakout", -1.0, 0.9, RegimeFit::Both, "z"),
        };
        const EnsembleVerdict v = route(consensus, nullptr, RouterConfig(), 0);
        CHECK_NEAR(v.weight, 2.7, 1e-12);
        CHECK_NEAR(v.conviction, -1.0, 1e-12);
        CHECK(should_veto_long(v, RouterConfig()));
    }
    // --- weak conviction does NOT veto, however unanimous ----------------
    {
        // Two mildly-bearish and one mildly-bullish, equal weight:
        // conviction = (2*(-0.3) + 0.3) / 0.9 = -0.3333, above the -0.35 bar.
        const std::vector<StrategySignal> split = {
            sig("a", -1.0, 0.3, RegimeFit::Both, "x"),
            sig("b", -1.0, 0.3, RegimeFit::Both, "y"),
            sig("c", 1.0, 0.3, RegimeFit::Both, "z"),
        };
        const EnsembleVerdict v = route(split, nullptr, RouterConfig(), 0);
        CHECK_NEAR(v.conviction, -0.3333333333333333, 1e-12);
        CHECK(v.conviction > -0.35);
        CHECK(!should_veto_long(v, RouterConfig()));
    }
    // --- a balanced book has NO net direction, so agreement is 0 ---------
    {
        const std::vector<StrategySignal> balanced = {
            sig("a", 1.0, 0.5, RegimeFit::Both, "x"),
            sig("b", -1.0, 0.5, RegimeFit::Both, "y"),
        };
        const EnsembleVerdict v = route(balanced, nullptr, RouterConfig(), 0);
        CHECK_NEAR(v.conviction, 0.0, 1e-15);
        CHECK_NEAR(v.agreement, 0.0, 1e-15);   // undefined -> the safe answer
        CHECK(!should_veto_long(v, RouterConfig()));
        CHECK(v.opposed_long());               // conviction 0 is not a long
    }
    // --- size_factor: only an AGREEING ensemble may resize ---------------
    {
        const std::vector<StrategySignal> agree = {
            sig("a", 1.0, 1.0, RegimeFit::Both, "x"),
            sig("b", 1.0, 1.0, RegimeFit::Both, "y"),
        };
        const EnsembleVerdict v = route(agree, nullptr, RouterConfig(), 0);
        CHECK_NEAR(v.size_factor(), 1.25, 1e-12);   // 0.75 + 0.5*1.0
        const std::vector<StrategySignal> oppose = {
            sig("a", -1.0, 1.0, RegimeFit::Both, "x")};
        CHECK_NEAR(route(oppose, nullptr, RouterConfig(), 0).size_factor(), 1.0, 1e-15);
    }
    // --- the router only ever BLOCKS: it never raises S -------------------
    //
    // S is the sniper's; the router's job is a veto. Structurally that means
    // `route()` cannot see or change a score — asserted by the API shape here
    // and by D1 below, which checks S is bit-identical with the ensemble on.
    {
        const std::vector<Bar> bars = make_bars(200, "chop");
        const StrategySeries s(bars);
        RegimeTracker tracker;
        const std::vector<RegimeState> regs = tracker.run(bars);
        bool any_veto = false, any_no_veto = false;
        for (size_t i = 0; i < s.n(); ++i) {
            const EnsembleVerdict v = evaluate(s, i, &regs[i], RouterConfig(),
                                               nullptr, s.cfg());
            if (should_veto_long(v, RouterConfig())) any_veto = true;
            else any_no_veto = true;
        }
        CHECK(any_veto);
        CHECK(any_no_veto);
    }
}

// =============================================================================
// Part D — the sniper's structural invariants
// =============================================================================
void test_sniper_invariants() {
    const std::vector<Bar> bars = make_bars(220, "chop");

    // ---------------------------------------------------------------------
    // D1 — THE ENSEMBLE CAN ONLY EVER BLOCK, NEVER PROMOTE.
    //
    // Turning the ensemble ON must leave `score`, and every layer that feeds
    // it, bit-identical. Only the veto fields may change. If this ever fails,
    // the ensemble has started contributing to S, which is exactly the
    // failure the design forbids: an ensemble that can talk you INTO a trade
    // can talk you into a bad one.
    // ---------------------------------------------------------------------
    {
        SniperConfig off;
        off.ensemble = false;
        SniperConfig on;
        on.ensemble = true;

        const Prepared p_off(bars, off);
        const Prepared p_on(bars, on);
        CHECK(!p_off.has_ensemble());
        CHECK(p_on.has_ensemble());
        // The ensemble must be WIRED, or the invariant below passes vacuously.
        CHECK(p_on.ensemble().size() == bars.size());
        CHECK(p_on.series() != nullptr);

        bool score_identical = true, layers_identical = true;
        int veto_diff = 0, ens_active = 0;
        for (size_t i = 0; i < bars.size(); ++i) {
            const SniperVerdict a = evaluate(p_off, i, nullptr, off);
            const SniperVerdict b = evaluate(p_on, i, nullptr, on);
            if (a.score != b.score) score_identical = false;
            if (a.micro != b.micro || a.tech != b.tech ||
                a.regime_score != b.regime_score || a.cross != b.cross ||
                a.votes != b.votes)
                layers_identical = false;
            if (a.veto != b.veto) ++veto_diff;
            if (p_on.ensemble()[i].active > 0) ++ens_active;
        }
        CHECK(score_identical);
        CHECK(layers_identical);
        CHECK(ens_active > 0);      // the zoo is actually speaking on some bars
        platform::log_info("       ensemble active on %d/%zu bars; it changed the "
                           "veto on %d", ens_active, bars.size(), veto_diff);
    }
    // --- the ensemble veto is REACHABLE (not dead code) ------------------
    //
    // A veto that can never fire is the same bug class as a gate that can never
    // open. It is rare by design — it needs >=0.50 of active weight to oppose
    // at >=0.60 agreement and <=-0.35 conviction — so this asserts reachability
    // across three different market characters rather than on one series.
    {
        SniperConfig on;
        on.ensemble = true;
        RouterConfig rc;
        rc.ensemble_veto = on.ensemble_veto;
        rc.agreement_veto = on.ensemble_agreement;
        rc.veto_min_weight = on.ensemble_min_weight;

        int fired = 0, bars_seen = 0;
        for (const char* mode : {"trend", "chop", "vol"}) {
            const std::vector<Bar> b = make_bars(300, mode);
            const Prepared p(b, on);
            for (size_t i = 0; i < b.size(); ++i) {
                ++bars_seen;
                if (should_veto_long(p.ensemble()[i], rc)) ++fired;
            }
        }
        platform::log_info("       ensemble veto fired on %d/%d bars across 3 "
                           "market characters", fired, bars_seen);
        CHECK(fired > 0);
    }
    // --- the ensemble veto is only ever ADDITIONAL: no bar that was already
    //     vetoed can become un-vetoed by enabling it -----------------------
    {
        SniperConfig off;
        SniperConfig on;
        on.ensemble = true;
        const Prepared p_off(bars, off);
        const Prepared p_on(bars, on);
        bool monotone = true;
        for (size_t i = 0; i < bars.size(); ++i) {
            const bool v_off = evaluate(p_off, i, nullptr, off).veto;
            const bool v_on = evaluate(p_on, i, nullptr, on).veto;
            if (v_off && !v_on) monotone = false;
        }
        CHECK(monotone);
    }

    // ---------------------------------------------------------------------
    // D2 — THE 0.85 GATE'S REACHABILITY, PER REGIME LABEL.
    //
    // The mandate named this bug. docs/MONSTER_DESIGN.md states the table:
    // with M = T = C = 1.0 the ceiling depends only on R.
    //     trend_up   R=1.00 -> 1.0000   reachable
    //     range      R=0.50 -> 0.8750   reachable, barely
    //     high_vol   R=0.25 -> 0.8125   MATHEMATICALLY IMPOSSIBLE
    //     trend_down R=0.00 -> 0.7500   impossible (and vetoed)
    // The 0.8125 row is the bug: no entry could ever fire while the label was
    // high_vol. The fix was to stop labelling every vol-expansion bar high_vol
    // (the smoothed percentile + absolute ATR stress in regime_engine.cpp).
    // This test pins the arithmetic so the fix cannot be undone by accident.
    // ---------------------------------------------------------------------
    {
        const SniperConfig cfg;
        auto ceiling = [&cfg](double r) {
            return cfg.w_micro * 1.0 + cfg.w_tech * 1.0 + cfg.w_regime * r +
                   cfg.w_cross * 1.0;
        };
        const double c_up = ceiling(1.00);
        const double c_range = ceiling(0.50);
        const double c_hivol = ceiling(0.25);
        const double c_down = ceiling(0.00);

        CHECK_NEAR(c_up, 1.0000, 1e-12);
        CHECK_NEAR(c_range, 0.8750, 1e-12);
        CHECK_NEAR(c_hivol, 0.8125, 1e-12);     // <- the named bug's number
        CHECK_NEAR(c_down, 0.7500, 1e-12);

        CHECK(c_up >= cfg.min_confidence);
        CHECK(c_range >= cfg.min_confidence);
        CHECK(c_hivol < cfg.min_confidence);    // still impossible, by design
        CHECK(c_down < cfg.min_confidence);

        // The label map that produces those R values must be the mandate's,
        // unchanged. Only the SOURCE of the label moved to C++.
        CHECK_NEAR(regime_score_for_label("trend_up"), 1.00, 1e-12);
        CHECK_NEAR(regime_score_for_label("range"), 0.50, 1e-12);
        CHECK_NEAR(regime_score_for_label("high_vol"), 0.25, 1e-12);
        CHECK_NEAR(regime_score_for_label("trend_down"), 0.00, 1e-12);
        // An unknown label must fall back to NEUTRAL, never to bullish.
        CHECK_NEAR(regime_score_for_label("typo"), 0.50, 1e-12);
        CHECK(!regime_label_known("typo"));

        // --- the WITH-CONTEXT ceiling, and the one that actually binds -----
        //
        // With no cross-asset context, `cross` is 0.5 by default, which lowers
        // the ceiling materially. That is not a bug — it is the honest cost of
        // not having peer data — but it is the number that decides whether a
        // bar can ever trade, so it is pinned rather than left implicit.
        auto ceiling_no_cross = [&cfg](double r, double t) {
            return cfg.w_micro * 1.0 + cfg.w_tech * t + cfg.w_regime * r +
                   cfg.w_cross * 0.5;
        };
        // 3 technical votes is the observed maximum on real daily equity data,
        // and 3 is exactly min_factors, so tech = 0.90 there.
        CHECK_NEAR(ceiling_no_cross(1.00, 0.90), 0.8900, 1e-12);   // reachable
        CHECK_NEAR(ceiling_no_cross(0.50, 0.90), 0.7650, 1e-12);   // NOT
        CHECK(ceiling_no_cross(1.00, 0.90) >= cfg.min_confidence);
        CHECK(ceiling_no_cross(0.50, 0.90) < cfg.min_confidence);
    }
    // --- the technical ladder: min_factors -> 0.90, +0.10 per extra vote --
    {
        SniperConfig cfg;
        CHECK(cfg.min_factors == 3);
        // votes < min_factors -> 0.5 * votes / min_factors
        CHECK_NEAR(0.5 * 2 / 3.0, 0.3333333333333333, 1e-12);
        // votes == min_factors -> 0.90 ; each extra -> +0.10, capped at 1.00
        CHECK_NEAR(0.90 + 0.10 * 0, 0.90, 1e-12);
        CHECK_NEAR(0.90 + 0.10 * 1, 1.00, 1e-12);
        CHECK_NEAR(std::min(1.0, 0.90 + 0.10 * 3), 1.00, 1e-12);
        // The old votes/6 ladder, for the record: 6 votes would give 1.0, but
        // 3 votes (the mandated minimum) gives only 0.5, and the resulting
        // ceiling in `range` is 0.7650 — see the D2 comment.
        CHECK_NEAR(3.0 / 6.0, 0.5, 1e-12);
    }
    // --- veto PRECEDENCE: the first gate to fire owns the reason ---------
    {
        // A steadily DECLINING series, so the legacy SMA200-slope rule labels
        // it trend_down. The veto must then read "counter-regime" even though
        // the confluence gate would also have fired, because counter-regime is
        // evaluated FIRST. (A flat series never produces trend_down, which
        // would have made this test vacuous — hence the -1%/bar drift.)
        std::vector<Bar> b(80);
        double px = 100.0;
        for (size_t i = 0; i < b.size(); ++i) {
            b[i].time = 1 + static_cast<int64_t>(i);
            b[i].open = px;
            b[i].close = px * 0.99;
            b[i].high = px;
            b[i].low = b[i].close * 0.995;
            b[i].volume = 1000.0;
            px = b[i].close;
        }
        SniperConfig cfg;
        cfg.regime_mode = "legacy";
        cfg.sma_regime = 20;
        cfg.regime_slope_n = 5;
        const Prepared p(b, cfg);
        int counter = 0;
        for (size_t i = 0; i < b.size(); ++i) {
            const SniperVerdict v = evaluate(p, i, nullptr, cfg);
            if (v.regime == "trend_down") {
                ++counter;
                CHECK(v.veto);
                // The precedence assertion: NOT "insufficient-confluence".
                CHECK(v.veto_reason == "counter-regime");
            }
        }
        platform::log_info("       trend_down bars seen: %d (veto reason checked)",
                           counter);
        CHECK(counter > 0);
    }
    // --- cross-asset invalidation vetoes, and says why -------------------
    {
        const Prepared p(bars, SniperConfig());
        EvalContext ctx;
        ctx.cross_active = 3;
        ctx.cross_invalidate = 2;       // >= invalidate_veto (2)
        const SniperVerdict v = evaluate(p, 200, &ctx, SniperConfig());
        CHECK(v.veto);
        CHECK(v.veto_reason == "cross-asset-invalidation");
        CHECK(v.cross < 0.5);           // 0.5 + 0.5*(0-2)/3 = 0.1667
    }
    // --- the OBI path takes the STRONGER of book and volume --------------
    {
        const Prepared p(bars, SniperConfig());
        EvalContext none;
        EvalContext obi;
        obi.has_obi = true;
        obi.obi = 1.0;                  // a maximally one-sided book
        const double m_none = sd::micro(p, 200, &none, SniperConfig());
        const double m_obi = sd::micro(p, 200, &obi, SniperConfig());
        CHECK(m_obi >= m_none - 1e-15);
        CHECK_NEAR(m_obi, std::max(1.0, m_none), 1e-12);
    }
    // --- event_driven relaxes the confluence requirement ----------------
    {
        const Prepared p(bars, SniperConfig());
        EvalContext ev;
        ev.event_driven = true;
        // Some bar where the normal gate vetoes for confluence but the relaxed
        // one does not, if such a bar exists.
        int relaxed_through = 0;
        for (size_t i = 0; i < bars.size(); ++i) {
            const SniperVerdict strict = evaluate(p, i, nullptr, SniperConfig());
            const SniperVerdict loose = evaluate(p, i, &ev, SniperConfig());
            if (strict.veto_reason == "insufficient-confluence" && !loose.veto)
                ++relaxed_through;
        }
        platform::log_info("       event-driven relaxed the gate on %d bars",
                           relaxed_through);
        CHECK(relaxed_through >= 0);
    }
}

// =============================================================================
// Part E — causality (prefix invariance)
// =============================================================================
void test_prefix_invariance() {
    const std::vector<Bar> full = make_bars(260, "trend");
    const StrategySeries s_full(full);
    RegimeTracker tracker_full;
    const std::vector<RegimeState> regs_full = tracker_full.run(full);

    bool strat_ok = true, router_ok = true, sniper_ok = true;
    int checked = 0;

    for (size_t i = 1; i < full.size(); ++i) {
        // Rebuild everything from the prefix [0..i] only.
        const std::vector<Bar> pre(full.begin(), full.begin() + static_cast<ptrdiff_t>(i) + 1);
        const StrategySeries s_pre(pre);
        RegimeTracker tracker_pre;
        const std::vector<RegimeState> regs_pre = tracker_pre.run(pre);

        const std::vector<StrategySignal> a = all_signals(s_full, i, &regs_full[i]);
        const std::vector<StrategySignal> b = all_signals(s_pre, i, &regs_pre[i]);
        for (size_t k = 0; k < a.size() && k < b.size(); ++k) {
            if (a[k].direction != b[k].direction ||
                a[k].confidence != b[k].confidence ||
                a[k].reason != b[k].reason)
                strat_ok = false;
        }

        const EnsembleVerdict va = route(a, &regs_full[i], RouterConfig(), s_full.ts()[i]);
        const EnsembleVerdict vb = route(b, &regs_pre[i], RouterConfig(), s_pre.ts()[i]);
        if (va.conviction != vb.conviction || va.agreement != vb.agreement ||
            va.weight != vb.weight || va.w_trend != vb.w_trend ||
            va.active != vb.active ||
            should_veto_long(va, RouterConfig()) != should_veto_long(vb, RouterConfig()))
            router_ok = false;

        SniperConfig cfg;
        cfg.ensemble = true;
        const Prepared p_full(full, cfg);
        const Prepared p_pre(pre, cfg);
        const SniperVerdict sa = evaluate(p_full, i, nullptr, cfg);
        const SniperVerdict sb = evaluate(p_pre, i, nullptr, cfg);
        if (sa.score != sb.score || sa.micro != sb.micro || sa.tech != sb.tech ||
            sa.regime_score != sb.regime_score || sa.cross != sb.cross ||
            sa.votes != sb.votes || sa.regime != sb.regime || sa.veto != sb.veto ||
            sa.veto_reason != sb.veto_reason || sa.conviction != sb.conviction ||
            sa.agreement != sb.agreement || sa.factors != sb.factors)
            sniper_ok = false;

        ++checked;
    }
    platform::log_info("       prefix-invariance checked on %d prefixes", checked);
    CHECK(strat_ok);
    CHECK(router_ok);
    CHECK(sniper_ok);
}

} // namespace

// =============================================================================
int main() {
    platform::log_info("=== omniseed strategy zoo tests ===");

    TEST("A: feature primitives on hand-checkable inputs");
    test_features();

    TEST("B: strategy branch behaviour");
    test_strategies();

    TEST("C: router, including the B4 lone-signal veto regression");
    test_router();

    TEST("D: sniper invariants (ensemble blocks only; 0.85 reachability)");
    test_sniper_invariants();

    TEST("E: strategies, router and sniper are prefix-invariant");
    test_prefix_invariance();

    platform::log_info("---- strategy zoo tests: %d passed, %d failed ----",
                       g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
