// =============================================================================
//  OmniSeed — tests/test_trading_edge.cpp
//  Trading Edge Lab (T7.1):
//
//    * Risk-budget sizing — 1..2% per trade, clamped, position-capped.
//    * DailyRiskGovernor — per-UTC-day drawdown kill-switch, daily re-arm.
//    * WalkForward        — no look-ahead, OOS determinism, aggregate math.
//    * PaperDaemon        — deterministic replay, journal append + resume.
//
//  Same tiny harness style as test_trading.cpp. No network, no fixtures.
// =============================================================================
#include "omniseed/core/platform.h"
#include "omniseed/trading/paper_daemon.h"
#include "omniseed/trading/trading_engine.h"
#include "omniseed/trading/walkforward.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
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

// ---------------------------------------------------------------------------
// Series builders
// ---------------------------------------------------------------------------
// GBM with periodic regime flips so trends AND mean-reversion both occur.
static std::vector<Bar> make_series(size_t n, int64_t step = 86400,
                                    uint64_t seed = 9) {
    std::vector<Bar> bars;
    bars.reserve(n);
    uint64_t s = seed * 2654435761ull + 12345ull;
    auto next01 = [&s]() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;   // xorshift64
        return static_cast<double>(s >> 11) / 9007199254740992.0;
    };
    const int64_t t0 = 1600000000;                 // 2020-09-13
    double price = 100.0;
    bool bull = true;
    for (size_t i = 0; i < n; ++i) {
        if (i % 250 == 0) bull = (i / 250) % 2 == 0;
        const double drift = bull ? 0.0008 : -0.0006;
        const double z = (next01() + next01() + next01() +
                          next01() + next01() + next01() - 3.0) * 1.4;
        const double ret = drift + 0.012 * z;
        const double o = price;
        const double c = std::max(1.0, o * std::exp(ret));
        const double h = std::max(o, c) * (1.0 + 0.004 * next01());
        const double l = std::min(o, c) * (1.0 - 0.004 * next01());
        bars.push_back({t0 + static_cast<int64_t>(i) * step, o, h, l, c, 1e6});
        price = c;
    }
    return bars;
}

// Hourly bars (many bars share one UTC day) — used to exercise the daily
// kill-switch, which is a no-op on a once-per-day series.
static std::vector<Bar> make_hourly(size_t n) { return make_series(n, 3600); }

// ---------------------------------------------------------------------------
// File helpers
// ---------------------------------------------------------------------------
static std::vector<std::string> read_lines(const std::string& path) {
    std::vector<std::string> out;
    std::ifstream f(path.c_str());
    std::string line;
    while (std::getline(f, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n'))
            line.pop_back();
        if (!line.empty()) out.push_back(line);
    }
    return out;
}

static size_t count_fields(const std::string& line) {
    size_t n = 1;
    for (const char c : line) if (c == ',') ++n;
    return n;
}

static bool has_substr(const std::vector<std::string>& v, const std::string& s) {
    for (const std::string& l : v) if (l.find(s) != std::string::npos) return true;
    return false;
}

// ===========================================================================
// 1. Risk-budget sizing
// ===========================================================================
static void test_risk_sizing() {
    TEST("risk sizing: qty = equity*risk/(price*stop)");
    {
        RiskLimits lim;                       // 1.5% risk, 8% stop, 100k equity
        const auto s = RiskManager::size_by_risk(100000.0, 100.0, lim);
        // 100000*0.015 / (100*0.08) = 1500/8 = 187.5 -> 187
        CHECK(s.allowed);
        CHECK(s.qty == 187.0);
        CHECK(s.reason.find("risk-budget") != std::string::npos);
    }

    TEST("risk sizing: clamps an over-large risk config down to 2%");
    {
        RiskLimits lim;
        lim.max_position_pct = 1.0;           // isolate the risk clamp
        lim.risk_per_trade_pct = 0.10;        // 10% -> must clamp to 2%
        const auto s = RiskManager::size_by_risk(100000.0, 100.0, lim);
        CHECK(s.qty == 250.0);                // 100000*0.02/8 = 250
        CHECK(s.reason.find("clamped") != std::string::npos);
        // Implied risk at the stop is exactly 2% of equity, never more.
        const double implied = s.qty * 100.0 * lim.stop_loss_pct / 100000.0;
        CHECK(std::fabs(implied - 0.02) < 1e-9);
    }

    TEST("risk sizing: clamps a tiny risk config up to 1%");
    {
        RiskLimits lim;
        lim.max_position_pct = 1.0;
        lim.risk_per_trade_pct = 0.0001;      // -> clamp to 1%
        const auto s = RiskManager::size_by_risk(100000.0, 100.0, lim);
        CHECK(s.qty == 125.0);                // 100000*0.01/8 = 125
        CHECK(s.reason.find("clamped") != std::string::npos);
    }

    TEST("risk sizing: never risks more than 2% across a sweep");
    {
        bool ok = true;
        for (const double r : {0.005, 0.01, 0.015, 0.02, 0.05, 0.5}) {
            RiskLimits lim;
            lim.risk_per_trade_pct = r;
            lim.max_position_pct = 1.0;       // isolate the risk clamp
            const auto s = RiskManager::size_by_risk(100000.0, 100.0, lim);
            const double implied = s.qty * 100.0 * lim.stop_loss_pct / 100000.0;
            if (implied > 0.02 + 1e-9) ok = false;
        }
        CHECK(ok);
    }

    TEST("risk sizing: position cap binds before the risk budget");
    {
        RiskLimits lim;
        lim.max_position_pct = 0.05;          // <= 5% notional -> 50 shares
        const auto s = RiskManager::size_by_risk(100000.0, 100.0, lim);
        CHECK(s.qty == 50.0);
        CHECK(s.reason.find("position-cap") != std::string::npos);
    }

    TEST("risk sizing: refuses degenerate inputs");
    {
        RiskLimits lim;
        CHECK(!RiskManager::size_by_risk(0.0, 100.0, lim).allowed);
        CHECK(!RiskManager::size_by_risk(100000.0, 0.0, lim).allowed);
        lim.stop_loss_pct = 0.0;
        CHECK(!RiskManager::size_by_risk(100000.0, 100.0, lim).allowed);
        RiskLimits tiny;
        CHECK(!RiskManager::size_by_risk(1.0, 100.0, tiny).allowed);  // floors to 0
    }
}

// ===========================================================================
// 2. DailyRiskGovernor
// ===========================================================================
static void test_daily_governor() {
    const int64_t day = 86400;
    const int64_t t0 = 1600000000 / day * day;    // a UTC midnight

    TEST("governor: allows while inside the daily loss limit");
    {
        DailyRiskGovernor gov(DailyRiskGovernor::Config{0.03, day});
        std::string why;
        CHECK(gov.allow(t0, 100000.0, why));      // day opens
        CHECK(gov.allow(t0 + 3600, 98500.0, why));  // -1.5%
        CHECK(gov.allow(t0 + 7200, 97500.0, why));  // -2.5%
        CHECK(!gov.halted());
        CHECK(gov.trips() == 0);
    }

    TEST("governor: trips at the limit, then blocks for the rest of the day");
    {
        DailyRiskGovernor gov(DailyRiskGovernor::Config{0.03, day});
        std::string why;
        CHECK(gov.allow(t0, 100000.0, why));
        CHECK(!gov.allow(t0 + 3600, 97000.0, why));   // exactly -3%
        CHECK(gov.halted());
        CHECK(gov.trips() == 1);
        CHECK(why.find("kill-switch") != std::string::npos);
        // Still blocked later the same day, even after a bounce.
        CHECK(!gov.allow(t0 + 7200, 99000.0, why));
        CHECK(gov.trips() == 1);                      // one trip per day
    }

    TEST("governor: drawdown is measured from the day's PEAK, not its open");
    {
        DailyRiskGovernor gov(DailyRiskGovernor::Config{0.03, day});
        std::string why;
        CHECK(gov.allow(t0, 100000.0, why));
        CHECK(gov.allow(t0 + 3600, 110000.0, why));   // peak 110k
        // 106k is still +6% on the day, but -3.64% from the peak -> halt.
        CHECK(!gov.allow(t0 + 7200, 106000.0, why));
        CHECK(gov.halted());
    }

    TEST("governor: re-arms on a new UTC day");
    {
        DailyRiskGovernor gov(DailyRiskGovernor::Config{0.03, day});
        std::string why;
        CHECK(gov.allow(t0, 100000.0, why));
        CHECK(!gov.allow(t0 + 3600, 96000.0, why));   // halted, trips=1
        const int64_t d0 = gov.day_index();
        CHECK(gov.allow(t0 + day, 96000.0, why));     // next day -> allowed
        CHECK(!gov.halted());
        CHECK(gov.day_index() == d0 + 1);
        CHECK(gov.trips() == 1);                      // trip count persists
        CHECK(gov.day_start_equity() == 96000.0);
        CHECK(gov.day_peak_equity() == 96000.0);
    }

    TEST("governor: counts one trip per halted day");
    {
        DailyRiskGovernor gov(DailyRiskGovernor::Config{0.02, day});
        std::string why;
        CHECK(gov.allow(t0, 100000.0, why));
        CHECK(!gov.allow(t0 + 60, 97000.0, why));     // day 1 halt
        CHECK(gov.allow(t0 + day, 97000.0, why));     // day 2 opens
        CHECK(!gov.allow(t0 + day + 60, 94000.0, why));  // day 2 halt
        CHECK(gov.trips() == 2);
    }

    TEST("governor: reset() clears all state");
    {
        DailyRiskGovernor gov(DailyRiskGovernor::Config{0.03, day});
        std::string why;
        gov.allow(t0, 100000.0, why);
        gov.allow(t0 + 3600, 96000.0, why);
        CHECK(gov.halted());
        gov.reset();
        CHECK(!gov.halted());
        CHECK(gov.trips() == 0);
        CHECK(gov.day_start_equity() == 0.0);
    }
}

// ===========================================================================
// 3. Walk-forward
// ===========================================================================
static void test_walk_forward() {
    const std::vector<Bar> bars = make_series(800);

    TEST("walk-forward: rejects a history shorter than one fold");
    {
        WalkForward::Config cfg;
        cfg.train_bars = 252; cfg.test_bars = 63;
        const auto r = WalkForward::run(make_series(200), cfg);
        CHECK(!r.error.empty());
        CHECK(r.folds_run == 0);
        CHECK(r.oos_equity.empty());
    }

    TEST("walk-forward: rejects train/test below the backtester minimum");
    {
        WalkForward::Config cfg;
        cfg.train_bars = 30; cfg.test_bars = 30;
        const auto r = WalkForward::run(bars, cfg);
        CHECK(!r.error.empty());
    }

    WalkForward::Config cfg;
    cfg.train_bars = 252;
    cfg.test_bars  = 63;
    cfg.step_bars  = 63;
    const auto r = WalkForward::run(bars, cfg);

    TEST("walk-forward: runs multiple folds on 800 bars");
    {
        CHECK(r.error.empty());
        CHECK(r.folds_run >= 3);
        CHECK(r.folds.size() == static_cast<size_t>(r.folds_run));
        CHECK(r.bars == 800);
    }

    TEST("walk-forward: NO look-ahead — every test window starts where train ends");
    {
        bool ok = true;
        for (size_t i = 0; i < r.folds.size(); ++i) {
            const auto& f = r.folds[i];
            if (f.test_begin != f.train_end) ok = false;
            if (f.train_begin >= f.train_end) ok = false;
            if (f.test_begin >= f.test_end) ok = false;
            if (f.test_end > bars.size()) ok = false;
            if (i > 0 && f.train_begin <= r.folds[i - 1].train_begin) ok = false;
            if (i > 0 && f.test_begin <= r.folds[i - 1].test_begin) ok = false;
        }
        CHECK(ok);
    }

    TEST("walk-forward: test windows are non-overlapping and contiguous");
    {
        bool ok = true;
        for (size_t i = 0; i + 1 < r.folds.size(); ++i)
            if (r.folds[i].test_end != r.folds[i + 1].test_begin) ok = false;
        CHECK(ok);
    }

    TEST("walk-forward: chosen params come from the candidate grid");
    {
        const auto grid = WalkForward::default_candidates();
        CHECK(grid.size() == 9);
        bool ok = true;
        for (const auto& f : r.folds) {
            if (f.chosen_index < 0 ||
                f.chosen_index >= static_cast<int>(grid.size())) { ok = false; break; }
            const auto& g = grid[static_cast<size_t>(f.chosen_index)];
            if (g.sma_fast != f.chosen.sma_fast ||
                g.rsi_buy_below != f.chosen.rsi_buy_below) ok = false;
        }
        CHECK(ok);
    }

    TEST("walk-forward: OOS curve is the stitched test windows");
    {
        size_t expected = 0;
        for (size_t i = 0; i < r.folds.size(); ++i) {
            const size_t sz = r.folds[i].test.equity_curve.size();
            expected += (i == 0) ? sz : (sz - 1);
        }
        CHECK(r.oos_equity.size() == expected);
        CHECK(expected > 0);
    }

    TEST("walk-forward: degradation = IS mean - OOS");
    {
        double is_sum = 0.0;
        for (const auto& f : r.folds) is_sum += f.train_return_pct;
        const double is_mean = is_sum / static_cast<double>(r.folds.size());
        CHECK(std::fabs(r.is_mean_return_pct - is_mean) < 1e-9);
        CHECK(std::fabs(r.degradation_pct -
                        (r.is_mean_return_pct - r.oos_total_return_pct)) < 1e-9);
    }

    TEST("walk-forward: deterministic — same input, same output");
    {
        const auto r2 = WalkForward::run(bars, cfg);
        CHECK(r2.folds_run == r.folds_run);
        CHECK(std::fabs(r2.oos_total_return_pct - r.oos_total_return_pct) < 1e-12);
        CHECK(r2.oos_equity.size() == r.oos_equity.size());
        bool same = r2.oos_equity.size() == r.oos_equity.size();
        for (size_t i = 0; same && i < r.oos_equity.size(); ++i)
            if (std::fabs(r2.oos_equity[i] - r.oos_equity[i]) > 1e-9) same = false;
        CHECK(same);
    }

    TEST("walk-forward: anchored mode grows the train window from bar 0");
    {
        WalkForward::Config acfg = cfg;
        acfg.anchored = true;
        const auto ar = WalkForward::run(bars, acfg);
        CHECK(ar.error.empty());
        CHECK(!ar.folds.empty());
        bool ok = true;
        size_t prev_end = 0;
        for (const auto& f : ar.folds) {
            if (f.train_begin != 0) ok = false;
            if (f.train_end <= prev_end) ok = false;   // strictly growing
            if (f.test_begin != f.train_end) ok = false;
            prev_end = f.train_end;
        }
        CHECK(ok);
    }
}

// ===========================================================================
// 4. Paper journal + daemon
// ===========================================================================
static void test_paper_daemon() {
    const std::string jpath = "build/edge_journal_test.csv";
    std::remove(jpath.c_str());

    TEST("journal: opens a new file with exactly one header row");
    {
        PaperJournal j(jpath);
        std::string err;
        CHECK(j.open(err));
        CHECK(err.empty());
        CHECK(j.ok());
        CHECK(j.records() == 0);
        const auto lines = read_lines(jpath);
        CHECK(lines.size() == 1);
        CHECK(lines[0] == std::string(PaperJournal::header()));
        j.close();
    }

    TEST("journal: appends records with the documented 10-field schema");
    {
        PaperJournal j(jpath);
        std::string err;
        CHECK(j.open(err));                       // append path: no new header
        j.event(1000, "start");
        j.equity(1000, 100000.0, 100000.0, 0.0);
        j.fill(1000, "TEST", 10.0, 100.0, 0.0, 100000.0, 99000.0, 1000.0, "entry");
        j.close();

        const auto lines = read_lines(jpath);
        CHECK(lines.size() == 4);                 // header + 3
        bool ok = true;
        for (const auto& l : lines) if (count_fields(l) != 10) ok = false;
        CHECK(ok);
        CHECK(has_substr(lines, ",event,"));
        CHECK(has_substr(lines, ",equity,"));
        CHECK(has_substr(lines, ",fill,"));
    }

    TEST("journal: scan_existing recovers record count and last_ts");
    {
        PaperJournal j(jpath);
        std::string err;
        j.scan_existing(err);
        CHECK(err.empty());
        CHECK(j.records() == 3);
        CHECK(j.last_ts() == 1000);
    }

    TEST("journal: a comma in the reason cannot break the schema");
    {
        const std::string p2 = "build/edge_journal_comma.csv";
        std::remove(p2.c_str());
        PaperJournal j(p2);
        std::string err;
        CHECK(j.open(err));
        j.event(5, "a,b,c");
        j.close();
        const auto lines = read_lines(p2);
        CHECK(lines.size() == 2);
        CHECK(count_fields(lines[1]) == 10);      // commas were neutralised
        CHECK(lines[1].find("a;b;c") != std::string::npos);
        std::remove(p2.c_str());
    }

    // --- daemon on a tradable daily series ---
    const std::vector<Bar> bars = make_series(600);

    TEST("daemon: replays every bar and journals equity each time");
    {
        const std::string p3 = "build/edge_journal_daemon.csv";
        std::remove(p3.c_str());
        PaperJournal j(p3);
        std::string err;
        CHECK(j.open(err));
        PaperDaemonConfig cfg;
        cfg.ticker = "TEST";
        const auto res = PaperDaemon::run(bars, cfg, j);
        j.close();

        CHECK(res.error.empty());
        CHECK(res.bars_processed == 600);
        CHECK(res.final_equity > 0.0);
        const auto lines = read_lines(p3);
        // header + one equity row per bar + the start event (+ fills/events)
        CHECK(lines.size() >= 601u + 1u);
        size_t equity_rows = 0, fill_rows = 0;
        // Skip the header: it also contains the literal ",equity," (the
        // column name sits between "pnl" and "cash").
        for (size_t li = 1; li < lines.size(); ++li) {
            const std::string& l = lines[li];
            if (l.find(",equity,") != std::string::npos) ++equity_rows;
            if (l.find(",fill,") != std::string::npos) ++fill_rows;
        }
        CHECK(equity_rows == 600);
        CHECK(fill_rows >= 1);                     // the rule set actually trades
        CHECK(static_cast<int64_t>(fill_rows) >= res.entries + res.exits);
        bool ok = true;
        for (const auto& l : lines) if (count_fields(l) != 10) ok = false;
        CHECK(ok);
        std::remove(p3.c_str());
    }

    TEST("daemon: deterministic — two runs journal identical rows");
    {
        const std::string pa = "build/edge_journal_det_a.csv";
        const std::string pb = "build/edge_journal_det_b.csv";
        std::remove(pa.c_str()); std::remove(pb.c_str());
        PaperJournal ja(pa), jb(pb);
        std::string err;
        CHECK(ja.open(err)); CHECK(jb.open(err));
        PaperDaemonConfig cfg;
        cfg.ticker = "TEST";
        const auto ra = PaperDaemon::run(bars, cfg, ja);
        const auto rb = PaperDaemon::run(bars, cfg, jb);
        ja.close(); jb.close();
        CHECK(ra.bars_processed == rb.bars_processed);
        CHECK(ra.entries == rb.entries);
        CHECK(ra.exits == rb.exits);
        CHECK(read_lines(pa) == read_lines(pb));
        std::remove(pa.c_str()); std::remove(pb.c_str());
    }

    TEST("daemon: resume appends only new bars — no duplicates");
    {
        const std::string p4 = "build/edge_journal_resume.csv";
        std::remove(p4.c_str());
        PaperDaemonConfig cfg;
        cfg.ticker = "TEST";
        cfg.resume = true;

        auto fill_ts_of = [](const std::string& path) {
            std::vector<int64_t> v;
            for (const auto& l : read_lines(path)) {
                if (l.find(",fill,") == std::string::npos) continue;
                v.push_back(static_cast<int64_t>(
                    std::atoll(l.substr(0, l.find(',')).c_str())));
            }
            return v;
        };

        // First pass: only the first 300 bars.
        const std::vector<Bar> first(bars.begin(), bars.begin() + 300);
        int64_t first_bars = 0;
        {
            PaperJournal j(p4);
            std::string err;
            CHECK(j.open(err));
            first_bars = PaperDaemon::run(first, cfg, j).bars_processed;
            j.close();
        }
        CHECK(first_bars == 300);
        const std::vector<int64_t> ts1 = fill_ts_of(p4);

        // Second pass: the whole series; resume must skip the first 300 bars.
        int64_t second_bars = 0;
        {
            PaperJournal j(p4);
            std::string err;
            j.scan_existing(err);
            CHECK(j.open(err));
            second_bars = PaperDaemon::run(bars, cfg, j).bars_processed;
            j.close();
        }
        CHECK(second_bars == 300);                 // 600 - 300 skipped

        const std::vector<int64_t> ts2 = fill_ts_of(p4);
        // Every fill appended by the second pass must be strictly newer than
        // every fill from the first pass (old bars are never re-processed).
        // NB: two fills CAN share a timestamp within one pass — an entry at
        // bar i+1's open and an exit at bar i+1's close — so a global
        // "no duplicate ts" check would be wrong.
        bool ok = ts2.size() >= ts1.size();
        for (size_t i = ts1.size(); ok && i < ts2.size(); ++i)
            for (size_t k = 0; ok && k < ts1.size(); ++k)
                if (ts2[i] <= ts1[k]) ok = false;
        CHECK(ok);
        std::remove(p4.c_str());
    }

    TEST("daemon: the daily kill-switch trips on an intraday drawdown and is journaled");
    {
        const std::string p5 = "build/edge_journal_halt.csv";
        std::remove(p5.c_str());
        PaperJournal j(p5);
        std::string err;
        CHECK(j.open(err));
        PaperDaemonConfig cfg;
        cfg.ticker = "TEST";
        cfg.warmup_bars = 60;
        // Hourly bars: 24 bars per UTC day, so the daily switch can actually
        // observe an intraday drawdown. 0.5% of equity is easy to reach once a
        // position (~19% notional) is open and the bar path turns against it.
        cfg.risk.max_daily_loss_pct = 0.005;
        const auto res = PaperDaemon::run(make_hourly(900), cfg, j);
        j.close();

        CHECK(res.error.empty());
        CHECK(res.entries >= 1);                   // the rule set opened a position
        CHECK(res.halts >= 1);                     // the drawdown tripped the switch
        const auto lines = read_lines(p5);
        CHECK(has_substr(lines, "kill-switch"));
        std::remove(p5.c_str());
    }
}

// ===========================================================================
int main() {
    platform::log_info("== trading edge lab: risk + walk-forward + paper ==");
    test_risk_sizing();
    test_daily_governor();
    test_walk_forward();
    test_paper_daemon();

    platform::log_info("RESULT: %d passed, %d failed", g_passed, g_failed);
    std::printf("RESULT: %d passed, %d failed\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
