// =============================================================================
//  OmniSeed — tests/test_market_perception.cpp
//  M1 market perception + M4 risk engine:
//
//    * Ticker validity, FeedHealthMonitor staleness/failure classification.
//    * PerceptionGate ABSTAIN decisions (missing/invalid/stale/dead/liquidity).
//    * RiskGovernorSet — daily AND weekly drawdown kill-switches.
//    * Asset-class risk override — meme capped at 0.25%, research-only default.
//    * Pearson correlation + the pairwise correlation cap.
//    * AutonomyGate — the live-money ladder (owner unlock + caps + durations).
//
//  Same tiny harness style as test_trading.cpp. Deterministic, no I/O.
// =============================================================================
#include "omniseed/core/platform.h"
#include "omniseed/trading/market_perception.h"
#include "omniseed/trading/risk_gate.h"
#include "omniseed/trading/trading_engine.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace omniseed;
using namespace omniseed::trading;

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

static const int64_t DAY = 86400;

// ===========================================================================
// 1. Feed health
// ===========================================================================
static void test_feed_health() {
    const FeedPolicy pol;   // 300s degraded / 900s dead / 3 failures

    TEST("feed health: a fresh observation is Ok");
    {
        FeedHealthMonitor m(pol);
        m.observe("AAPL", "yahoo", 1000, 1000);
        const FeedHealth h = m.health("AAPL", 1000);
        CHECK(h.state == FeedState::Ok);
        CHECK(h.staleness(1000) == 0);
        CHECK(h.consecutive_failures == 0);
        CHECK(h.total_observations == 1);
        CHECK(h.error_rate() == 0.0);
    }

    TEST("feed health: staleness walks Ok -> Degraded -> Dead");
    {
        FeedHealthMonitor m(pol);
        m.observe("AAPL", "yahoo", 1000, 1000);
        CHECK(m.health("AAPL", 1000 + 100).state == FeedState::Ok);
        CHECK(m.health("AAPL", 1000 + 400).state == FeedState::Degraded);
        CHECK(m.health("AAPL", 1000 + 1000).state == FeedState::Dead);
    }

    TEST("feed health: consecutive failures kill the feed");
    {
        FeedHealthMonitor m(pol);
        m.observe("BTCUSDT", "binance", 5000, 5000);
        m.record_failure("BTCUSDT", 5010);
        m.record_failure("BTCUSDT", 5020);
        CHECK(m.health("BTCUSDT", 5020).state == FeedState::Ok);   // 2 < 3
        m.record_failure("BTCUSDT", 5030);
        CHECK(m.health("BTCUSDT", 5030).state == FeedState::Dead);
        CHECK(m.health("BTCUSDT", 5030).error_rate() > 0.0);
    }

    TEST("feed health: a success clears the failure streak");
    {
        FeedHealthMonitor m(pol);
        m.observe("X", "p", 1, 1);
        m.record_failure("X", 2);
        m.record_failure("X", 3);
        m.record_failure("X", 4);
        CHECK(m.health("X", 4).state == FeedState::Dead);
        m.observe("X", "p", 5, 5);                 // provider recovered
        const FeedHealth h = m.health("X", 5);
        CHECK(h.state == FeedState::Ok);
        CHECK(h.consecutive_failures == 0);
        CHECK(h.total_failures == 3);              // history retained
    }

    TEST("feed health: an unknown symbol is Missing, never implicitly Ok");
    {
        FeedHealthMonitor m(pol);
        const FeedHealth h = m.health("NOPE", 1000);
        CHECK(h.state == FeedState::Missing);
        CHECK(h.staleness(1000) == -1);
        CHECK(m.all(1000).empty());
    }
}

// ===========================================================================
// 2. PerceptionGate (ABSTAIN)
// ===========================================================================
static void test_abstain_gate() {
    PerceptionGate gate;                          // 300/900 policy

    auto ticker = [](int64_t ts, double last = 100.0, double vol = 1e6) {
        Ticker t;
        t.symbol = "AAPL"; t.provider = "yahoo";
        t.asset = AssetClass::Equity;
        t.last = last; t.ts = ts; t.volume_24h = vol;
        return t;
    };

    TEST("abstain: a fresh, healthy ticker trades");
    {
        FeedHealthMonitor m(gate.config().feed);
        m.observe("AAPL", "yahoo", 1000, 1000);
        const auto d = gate.check(ticker(1000), m.health("AAPL", 1000), 1000);
        CHECK(!d.abstain);
        CHECK(d.reason == AbstainReason::None);
    }

    TEST("abstain: an invalid ticker is refused");
    {
        Ticker t = ticker(0);                     // no timestamp
        const auto d = gate.check(t, FeedHealth{}, 1000);
        CHECK(d.abstain);
        CHECK(d.reason == AbstainReason::InvalidTicker);
        Ticker z = ticker(1000, 0.0);             // no price
        CHECK(gate.check(z, FeedHealth{}, 1000).abstain);
    }

    TEST("abstain: stale data is refused (degraded then dead)");
    {
        FeedHealthMonitor m(gate.config().feed);
        m.observe("AAPL", "yahoo", 1000, 1000);
        const Ticker t = ticker(1000);
        const auto deg = gate.check(t, m.health("AAPL", 1000 + 400), 1000 + 400);
        CHECK(deg.abstain);
        CHECK(deg.reason == AbstainReason::StaleFeed);
        const auto dead = gate.check(t, m.health("AAPL", 1000 + 1000), 1000 + 1000);
        CHECK(dead.abstain);
        CHECK(dead.reason == AbstainReason::DeadFeed);
    }

    TEST("abstain: a dead feed (repeated failures) is refused");
    {
        FeedHealthMonitor m(gate.config().feed);
        m.observe("AAPL", "yahoo", 1000, 1000);
        for (int i = 0; i < 3; ++i) m.record_failure("AAPL", 1000 + i);
        const auto d = gate.check(ticker(1000), m.health("AAPL", 1003), 1003);
        CHECK(d.abstain);
        CHECK(d.reason == AbstainReason::DeadFeed);
    }

    TEST("abstain: no observation at all is refused");
    {
        FeedHealthMonitor m(gate.config().feed);
        const auto d = gate.check(ticker(1000), m.health("AAPL", 1000), 1000);
        CHECK(d.abstain);
        CHECK(d.reason == AbstainReason::MissingData);
    }

    TEST("abstain: a ticker with no monitor falls back to its own ts");
    {
        const auto d = gate.check(ticker(1000), FeedHealth{}, 1000);
        CHECK(!d.abstain);                        // fresh per its own timestamp
        const auto old = gate.check(ticker(1000), FeedHealth{}, 1000 + 1000);
        CHECK(old.abstain);                       // and stale by the same rule
    }

    TEST("abstain: the liquidity floor is enforced");
    {
        PerceptionGate::Config cfg;
        cfg.min_volume_24h = 5e6;
        PerceptionGate liq(cfg);
        FeedHealthMonitor m(cfg.feed);
        m.observe("AAPL", "yahoo", 1000, 1000);
        CHECK(liq.check(ticker(1000, 100.0, 1e6), m.health("AAPL", 1000), 1000)
                  .reason == AbstainReason::LowLiquidity);
        CHECK(!liq.check(ticker(1000, 100.0, 9e6), m.health("AAPL", 1000), 1000)
                   .abstain);
    }

    TEST("abstain: reason strings are stable (log-friendly)");
    {
        CHECK(std::string(to_string(AbstainReason::StaleFeed)) == "stale-feed");
        CHECK(std::string(to_string(AbstainReason::DeadFeed)) == "dead-feed");
        CHECK(std::string(to_string(FeedState::Degraded)) == "degraded");
        CHECK(std::string(to_string(AssetClass::Meme)) == "meme");
    }
}

// ===========================================================================
// 3. RiskGovernorSet — daily + weekly kill-switches
// ===========================================================================
static void test_governor_set() {
    TEST("governor set: a single bad day trips the DAILY window only");
    {
        RiskGovernorSet g(RiskGovernorSet::Config{0.03, 0.06, DAY});
        std::string why;
        CHECK(g.allow(0, 100000.0, why));
        CHECK(!g.allow(3600, 96000.0, why));       // -4% intraday
        CHECK(g.daily_trips() == 1);
        CHECK(g.weekly_trips() == 0);
        CHECK(why.find("daily:") == 0);
    }

    TEST("governor set: a slow bleed trips the WEEKLY window, not the daily");
    {
        // -1.5% per day for 5 days: no single day reaches -3%, but the week
        // falls 6% from its high-water mark.
        RiskGovernorSet g(RiskGovernorSet::Config{0.03, 0.06, DAY});
        std::string why;
        CHECK(g.allow(0 * DAY, 100000.0, why));
        CHECK(g.allow(1 * DAY, 98500.0, why));
        CHECK(g.allow(2 * DAY, 97000.0, why));
        CHECK(g.allow(3 * DAY, 95500.0, why));
        CHECK(!g.allow(4 * DAY, 94000.0, why));    // -6.0% from the week peak
        CHECK(g.daily_trips() == 0);
        CHECK(g.weekly_trips() == 1);
        CHECK(why.find("weekly:") == 0);
    }

    TEST("governor set: halted() is the OR of the two windows");
    {
        RiskGovernorSet g(RiskGovernorSet::Config{0.03, 0.06, DAY});
        std::string why;
        CHECK(!g.halted());
        g.allow(0, 100000.0, why);
        g.allow(60, 96000.0, why);
        CHECK(g.halted());
        CHECK(g.total_trips() == 1);
        g.reset();
        CHECK(!g.halted());
        CHECK(g.total_trips() == 0);
    }
}

// ===========================================================================
// 4. Asset-class risk override (meme)
// ===========================================================================
static void test_asset_class_risk() {
    TEST("asset class: non-meme classes keep the 1-2% band");
    {
        RiskLimits lim;
        CHECK(std::fabs(RiskManager::effective_risk_pct(AssetClass::Equity, lim)
                        - 0.015) < 1e-12);
        lim.risk_per_trade_pct = 0.5;              // clamped down to 2%
        CHECK(std::fabs(RiskManager::effective_risk_pct(AssetClass::Crypto, lim)
                        - 0.02) < 1e-12);
        lim.risk_per_trade_pct = 0.0001;           // clamped up to 1%
        CHECK(std::fabs(RiskManager::effective_risk_pct(AssetClass::Fx, lim)
                        - 0.01) < 1e-12);
    }

    TEST("asset class: meme is research-only by default");
    {
        RiskLimits lim;
        CHECK(lim.meme_research_only);
        CHECK(RiskManager::effective_risk_pct(AssetClass::Meme, lim) == 0.0);
        const auto s = RiskManager::size_by_risk(100000.0, 1.0, AssetClass::Meme, lim);
        CHECK(!s.allowed);
        CHECK(s.qty == 0.0);
        CHECK(s.reason.find("research-only") != std::string::npos);
    }

    TEST("asset class: a permitted meme stays <= 0.25% (not clamped up)");
    {
        RiskLimits lim;
        lim.meme_research_only = false;            // explicitly enabled
        const auto s = RiskManager::size_by_risk(100000.0, 1.0, AssetClass::Meme, lim);
        CHECK(s.allowed);
        // 100000*0.0025 / (1.0*0.08) = 3125
        CHECK(s.qty == 3125.0);
        const double implied = s.qty * 1.0 * lim.stop_loss_pct / 100000.0;
        CHECK(std::fabs(implied - 0.0025) < 1e-9);
        CHECK(implied <= 0.0025 + 1e-12);
        CHECK(s.reason.find("meme") != std::string::npos);
    }

    TEST("asset class: the meme cap never exceeds the configured risk");
    {
        RiskLimits lim;
        lim.meme_research_only = false;
        lim.risk_per_trade_pct = 0.001;            // below the meme cap
        CHECK(std::fabs(RiskManager::effective_risk_pct(AssetClass::Meme, lim)
                        - 0.001) < 1e-12);
    }

    TEST("asset class: equity sizing matches the untyped overload");
    {
        RiskLimits lim;
        const auto a = RiskManager::size_by_risk(100000.0, 100.0, lim);
        const auto b = RiskManager::size_by_risk(100000.0, 100.0,
                                                 AssetClass::Equity, lim);
        CHECK(a.qty == b.qty);
        CHECK(a.allowed == b.allowed);
    }
}

// ===========================================================================
// 5. Correlation
// ===========================================================================
static void test_correlation() {
    const std::vector<double> a = {1, 2, 3, 4, 5, 6, 7, 8};
    const std::vector<double> b = {2, 4, 6, 8, 10, 12, 14, 16};
    const std::vector<double> c = {8, 7, 6, 5, 4, 3, 2, 1};

    TEST("correlation: perfect, inverse, and degenerate cases");
    {
        CHECK(std::fabs(RiskManager::correlation(a, b) - 1.0) < 1e-9);
        CHECK(std::fabs(RiskManager::correlation(a, c) + 1.0) < 1e-9);
        CHECK(RiskManager::correlation(a, a) == 1.0);
        CHECK(RiskManager::correlation({1.0}, {1.0}) == 0.0);      // too short
        CHECK(RiskManager::correlation(a, {1, 2, 3}) == 0.0);      // mismatched
        CHECK(RiskManager::correlation(std::vector<double>(5, 3.0),
                                       std::vector<double>(5, 3.0)) == 0.0);
    }

    TEST("correlation: the pairwise cap refuses a redundant holding");
    {
        RiskLimits lim;                            // max_correlation = 0.80
        std::vector<std::vector<double>> book;
        book.push_back(b);                         // already held: r == 1.0 vs a
        double worst = 0.0;
        std::string why;
        CHECK(!RiskManager::correlation_ok(book, a, lim, worst, why));
        CHECK(std::fabs(worst - 1.0) < 1e-9);
        CHECK(why.find("correlation") != std::string::npos);
    }

    TEST("correlation: an uncorrelated candidate passes and reports worst");
    {
        RiskLimits lim;
        std::vector<std::vector<double>> book;
        book.push_back(b);
        const std::vector<double> zig = {1, -1, 1, -1, 1, -1, 1, -1};
        double worst = 0.0;
        std::string why;
        CHECK(RiskManager::correlation_ok(book, zig, lim, worst, why));
        CHECK(worst >= 0.0 && worst < lim.max_correlation);
    }

    TEST("correlation: an empty book always passes");
    {
        RiskLimits lim;
        double worst = -1.0;
        std::string why;
        CHECK(RiskManager::correlation_ok({}, a, lim, worst, why));
        CHECK(worst == 0.0);
    }
}

// ===========================================================================
// 6. Autonomy ladder (M6)
// ===========================================================================
static void test_autonomy_ladder() {
    TEST("autonomy: no unlock line -> paper-only");
    {
        std::string why;
        CHECK(AutonomyGate::resolve("", why) == AutonomyLevel::PaperOnly);
        CHECK(AutonomyGate::resolve("some notes\n## 19 ...\n", why)
              == AutonomyLevel::PaperOnly);
        CHECK(why.find("paper-only") != std::string::npos);
    }

    TEST("autonomy: exact owner lines unlock L1 / L2");
    {
        std::string why;
        CHECK(AutonomyGate::resolve("UNLOCK L1 MICRO-LIVE\n", why)
              == AutonomyLevel::MicroLive);
        CHECK(AutonomyGate::resolve("UNLOCK L2 SCALE-UP\n", why)
              == AutonomyLevel::Scaled);
        // L2 supersedes L1 when both are present.
        CHECK(AutonomyGate::resolve("UNLOCK L1 MICRO-LIVE\nUNLOCK L2 SCALE-UP\n", why)
              == AutonomyLevel::Scaled);
    }

    TEST("autonomy: unlock lines are exact — near-misses never unlock");
    {
        std::string why;
        CHECK(AutonomyGate::resolve("unlock l1 micro-live", why)
              == AutonomyLevel::PaperOnly);                    // wrong case
        CHECK(AutonomyGate::resolve("UNLOCK L1 MICRO LIVE", why)
              == AutonomyLevel::PaperOnly);                    // no hyphen
        CHECK(AutonomyGate::resolve("TODO: UNLOCK L1 MICRO-LIVE", why)
              == AutonomyLevel::PaperOnly);                    // not alone
        CHECK(AutonomyGate::resolve("> UNLOCK L1 MICRO-LIVE", why)
              == AutonomyLevel::PaperOnly);                    // quoted
        CHECK(AutonomyGate::resolve("  UNLOCK L1 MICRO-LIVE\t\r\n", why)
              == AutonomyLevel::MicroLive);                    // padded is fine
    }

    TEST("autonomy: L0 never trades live");
    {
        AutonomyConfig cfg;
        std::string why;
        CHECK(!AutonomyGate::allows_live(AutonomyLevel::PaperOnly, 10.0, 0.005,
                                         999, cfg, why));
        CHECK(why.find("paper-only") != std::string::npos);
    }

    TEST("autonomy: L1 enforces duration, notional and per-trade risk");
    {
        AutonomyConfig cfg;                        // 30 days, $100, 1%
        std::string why;
        CHECK(AutonomyGate::allows_live(AutonomyLevel::MicroLive, 50.0, 0.005,
                                        30, cfg, why));
        CHECK(!AutonomyGate::allows_live(AutonomyLevel::MicroLive, 50.0, 0.005,
                                         29, cfg, why));      // too early
        CHECK(!AutonomyGate::allows_live(AutonomyLevel::MicroLive, 150.0, 0.005,
                                         30, cfg, why));      // over the cap
        CHECK(!AutonomyGate::allows_live(AutonomyLevel::MicroLive, 50.0, 0.02,
                                         30, cfg, why));      // risk too high
        CHECK(!AutonomyGate::allows_live(AutonomyLevel::MicroLive, 0.0, 0.005,
                                         30, cfg, why));      // degenerate
    }

    TEST("autonomy: L2 requires the live track record");
    {
        AutonomyConfig cfg;                        // 90 live days
        std::string why;
        CHECK(!AutonomyGate::allows_live(AutonomyLevel::Scaled, 5000.0, 0.02,
                                         89, cfg, why));
        CHECK(AutonomyGate::allows_live(AutonomyLevel::Scaled, 5000.0, 0.02,
                                        90, cfg, why));
        CHECK(std::string(to_string(AutonomyLevel::Scaled)) == "L2-scaled");
    }
}

// ===========================================================================
int main() {
    platform::log_info("== market perception + risk engine (M1/M4) ==");
    test_feed_health();
    test_abstain_gate();
    test_governor_set();
    test_asset_class_risk();
    test_correlation();
    test_autonomy_ladder();

    platform::log_info("RESULT: %d passed, %d failed", g_passed, g_failed);
    std::printf("RESULT: %d passed, %d failed\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
