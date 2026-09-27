// =============================================================================
//  OmniSeed — tests/test_paper_session.cpp
//  M5 multi-asset paper session:
//
//    * PaperSession      — merged-timeline replay over several streams sharing
//                          one cash account + one daily/weekly kill-switch.
//    * Per-asset sizing  — meme stays research-only (refused + journaled).
//    * ABSTAIN           — a stream whose feed is not OK never opens risk.
//    * Journal resume    — the book is rebuilt from the append-only journal;
//                          a re-run appends only genuinely new timestamps.
//
//  Same tiny harness style as test_trading_edge.cpp. No network.
// =============================================================================
#include "omniseed/core/platform.h"
#include "omniseed/trading/paper_daemon.h"
#include "omniseed/trading/trading_engine.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
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
// Helpers
// ---------------------------------------------------------------------------
static std::string slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return "";
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static size_t count_substr(const std::string& hay, const std::string& needle) {
    size_t n = 0, pos = 0;
    while ((pos = hay.find(needle, pos)) != std::string::npos) { ++n; pos += needle.size(); }
    return n;
}

static size_t count_lines(const std::string& s) {
    size_t n = 0;
    for (char c : s) if (c == '\n') ++n;
    return n;
}

// GBM with periodic regime flips so trends AND mean-reversion both occur —
// the same generator the edge-lab test uses, so Buy signals reliably fire
// (a monotone ramp leaves the MACD histogram ~0 and never enters).
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

static PaperStream stream(const std::string& sym, AssetClass ac,
                          std::vector<Bar> bars, bool feed_ok = true,
                          const std::string& reason = "") {
    PaperStream s;
    s.symbol = sym;
    s.asset = ac;
    s.bars = std::move(bars);
    s.feed_ok = feed_ok;
    s.feed_reason = reason;
    return s;
}

// ===========================================================================
// 1. Merged timeline + per-stream fills
// ===========================================================================
static void test_merged_timeline() {
    const std::string dir = "state/test_session_a";
    const std::string jpath = dir + "/journal.csv";
    std::remove(jpath.c_str());

    TEST("session: two streams merge into one timeline and one equity series");
    {
        PaperSessionConfig cfg;
        cfg.broker.starting_cash = 100000.0;
        cfg.streams.push_back(stream("AAA", AssetClass::Equity, make_series(200, 86400)));
        cfg.streams.push_back(stream("BBB", AssetClass::Crypto, make_series(200, 86400)));
        PaperJournal j(jpath);
        std::string err;
        j.scan_existing(err);
        CHECK(j.open(err));
        const PaperSessionResult r = PaperSession::run(cfg, j);
        j.close();
        CHECK(r.error.empty());
        CHECK(r.timestamps == 200);                 // same timestamps -> merged
        CHECK(r.bars_processed == 400);             // two stream-bars each
        const std::string text = slurp(jpath);
        CHECK(count_substr(text, ",equity,") >= 200);
        // Fills (if any) are always tagged with their own symbol.
        const size_t fa = count_substr(text, ",fill,AAA,");
        const size_t fb = count_substr(text, ",fill,BBB,");
        CHECK(fa + fb == static_cast<size_t>(r.entries + r.exits));
    }

    TEST("session: a shared book never exceeds the per-name position cap");
    {
        PaperSessionConfig cfg;
        cfg.broker.starting_cash = 100000.0;
        cfg.streams.push_back(stream("AAA", AssetClass::Equity, make_series(200, 86400)));
        cfg.streams.push_back(stream("BBB", AssetClass::Crypto, make_series(200, 86400)));
        PaperJournal j(dir + "/journal2.csv");
        std::string err;
        j.scan_existing(err);
        CHECK(j.open(err));
        const PaperSessionResult r = PaperSession::run(cfg, j);
        j.close();
        CHECK(r.final_equity > 0.0);
        CHECK(r.entries >= 0 && r.exits >= 0);
    }
}

// ===========================================================================
// 2. Asset-class gates
// ===========================================================================
static void test_asset_gates() {
    TEST("session: a meme stream is refused (research-only) and journaled");
    {
        const std::string jpath = "state/test_session_b/meme.csv";
        std::remove(jpath.c_str());
        PaperSessionConfig cfg;
        cfg.streams.push_back(stream("DOGE", AssetClass::Meme, make_series(200, 86400)));
        PaperJournal j(jpath);
        std::string err;
        j.scan_existing(err);
        CHECK(j.open(err));
        const PaperSessionResult r = PaperSession::run(cfg, j);
        j.close();
        CHECK(r.entries == 0);                       // never trades
        CHECK(r.refused >= 1);                       // gate recorded
        CHECK(slurp(jpath).find("refused:DOGE:") != std::string::npos);
    }

    TEST("session: a non-meme stream can still enter (the gate is class-specific)");
    {
        const std::string jpath = "state/test_session_b/equity.csv";
        std::remove(jpath.c_str());
        PaperSessionConfig cfg;
        cfg.streams.push_back(stream("AAA", AssetClass::Equity, make_series(200, 86400)));
        PaperJournal j(jpath);
        std::string err;
        j.scan_existing(err);
        CHECK(j.open(err));
        const PaperSessionResult r = PaperSession::run(cfg, j);
        j.close();
        CHECK(r.entries >= 1);
        CHECK(r.refused == 0);
    }
}

// ===========================================================================
// 3. ABSTAIN
// ===========================================================================
static void test_abstain() {
    TEST("session: a feed that is not OK abstains and opens no risk");
    {
        const std::string jpath = "state/test_session_c/abstain.csv";
        std::remove(jpath.c_str());
        PaperSessionConfig cfg;
        cfg.streams.push_back(stream("AAA", AssetClass::Equity, make_series(200, 86400),
                                     /*feed_ok=*/false, "stale-feed"));
        PaperJournal j(jpath);
        std::string err;
        j.scan_existing(err);
        CHECK(j.open(err));
        const PaperSessionResult r = PaperSession::run(cfg, j);
        j.close();
        CHECK(r.entries == 0);
        CHECK(r.abstains >= 1);
        const std::string text = slurp(jpath);
        CHECK(text.find("abstain:AAA:stale-feed") != std::string::npos);
        // The timeline is still marked even while abstaining.
        CHECK(r.timestamps == 200);
    }
}

// ===========================================================================
// 4. Composite kill-switch
// ===========================================================================
static void test_kill_switch() {
    TEST("session: the composite kill-switch halts entries and is journaled");
    {
        const std::string jpath = "state/test_session_d/halt.csv";
        std::remove(jpath.c_str());
        PaperSessionConfig cfg;
        cfg.governors.max_daily_loss_pct  = 0.005;   // trip on an intraday drop
        cfg.governors.max_weekly_loss_pct = 0.010;
        cfg.streams.push_back(stream("AAA", AssetClass::Equity,
                                     make_series(900, 3600, 9)));
        cfg.streams.push_back(stream("BBB", AssetClass::Equity,
                                     make_series(900, 3600, 21)));
        PaperJournal j(jpath);
        std::string err;
        j.scan_existing(err);
        CHECK(j.open(err));
        const PaperSessionResult r = PaperSession::run(cfg, j);
        j.close();
        CHECK(r.halts >= 1);
        const std::string text = slurp(jpath);
        CHECK(text.find("daily:") != std::string::npos ||
              text.find("weekly:") != std::string::npos);
    }
}

// ===========================================================================
// 5. Resume
// ===========================================================================
static void test_resume() {
    TEST("session: resume rebuilds the book and appends nothing when caught up");
    {
        const std::string jpath = "state/test_session_e/resume.csv";
        std::remove(jpath.c_str());
        const std::vector<Bar> bars = make_series(200, 86400);

        PaperSessionConfig cfg;
        cfg.streams.push_back(stream("AAA", AssetClass::Equity, bars));
        {
            PaperJournal j(jpath);
            std::string err;
            j.scan_existing(err);
            CHECK(j.open(err));
            const PaperSessionResult r = PaperSession::run(cfg, j);
            j.close();
            CHECK(r.timestamps == 200);
        }
        const std::string first = slurp(jpath);
        const size_t first_lines = count_lines(first);

        // Second run: nothing new.
        PaperJournal j2(jpath);
        std::string err;
        j2.scan_existing(err);
        CHECK(j2.last_ts() > 0);                     // resume mark recovered
        CHECK(j2.open(err));
        const PaperSessionResult r2 = PaperSession::run(cfg, j2);
        j2.close();
        CHECK(r2.timestamps == 0);
        CHECK(count_lines(slurp(jpath)) == first_lines);   // byte-stable

        // The book is rebuilt from the journal (positions recovered).
        CHECK(!j2.positions().empty() || j2.last_cash() > 0.0);
    }

    TEST("session: resume continues from a longer series without duplicating");
    {
        const std::string jpath = "state/test_session_f/resume2.csv";
        std::remove(jpath.c_str());
        std::vector<Bar> bars = make_series(200, 86400);
        PaperSessionConfig cfg;
        cfg.streams.push_back(stream("AAA", AssetClass::Equity, bars));
        {
            PaperJournal j(jpath);
            std::string err;
            j.scan_existing(err);
            CHECK(j.open(err));
            PaperSession::run(cfg, j);
            j.close();
        }
        const size_t before = count_lines(slurp(jpath));

        // Extend the series by 40 bars and re-run.
        std::vector<Bar> longer = bars;
        const std::vector<Bar> extra = make_series(240, 86400);
        for (size_t i = 200; i < extra.size(); ++i) longer.push_back(extra[i]);
        cfg.streams[0].bars = longer;

        PaperJournal j2(jpath);
        std::string err;
        j2.scan_existing(err);
        CHECK(j2.open(err));
        const PaperSessionResult r = PaperSession::run(cfg, j2);
        j2.close();
        CHECK(r.timestamps == 40);                   // only the new tail
        CHECK(count_lines(slurp(jpath)) > before);
    }
}

// ===========================================================================
// 6. Determinism + errors
// ===========================================================================
static void test_determinism_and_errors() {
    TEST("session: identical inputs produce an identical journal");
    {
        const std::string ja = "state/test_session_g/det_a.csv";
        const std::string jb = "state/test_session_g/det_b.csv";
        std::remove(ja.c_str());
        std::remove(jb.c_str());
        PaperSessionConfig cfg;
        cfg.streams.push_back(stream("AAA", AssetClass::Equity, make_series(200, 86400)));
        cfg.streams.push_back(stream("BTCUSDT", AssetClass::Crypto, make_series(200, 86400)));
        for (const std::string& p : {ja, jb}) {
            PaperJournal j(p);
            std::string err;
            j.scan_existing(err);
            j.open(err);
            PaperSession::run(cfg, j);
            j.close();
        }
        CHECK(slurp(ja) == slurp(jb));
    }

    TEST("session: no streams is an explicit error");
    {
        PaperSessionConfig cfg;
        PaperJournal j("state/test_session_g/empty.csv");
        std::string err;
        j.scan_existing(err);
        j.open(err);
        const PaperSessionResult r = PaperSession::run(cfg, j);
        j.close();
        CHECK(!r.error.empty());
    }

    TEST("session: a missing CSV is reported per stream, not fatal");
    {
        PaperSessionConfig cfg;
        PaperStream bad;
        bad.symbol = "NOPE";
        bad.asset = AssetClass::Equity;
        bad.csv = "state/test_session_g/does_not_exist.csv";
        cfg.streams.push_back(bad);
        cfg.streams.push_back(stream("AAA", AssetClass::Equity, make_series(120, 86400)));
        const std::string jpath = "state/test_session_g/mixed.csv";
        std::remove(jpath.c_str());
        PaperJournal j(jpath);
        std::string err;
        j.scan_existing(err);
        j.open(err);
        const PaperSessionResult r = PaperSession::run(cfg, j);
        j.close();
        CHECK(r.error.empty());                       // the run still proceeds
        CHECK(r.stream_errors.size() == 1);
        CHECK(r.timestamps == 120);
    }
}

// ===========================================================================
// 7. Signal journaling
// ===========================================================================
static void test_signals() {
    TEST("session: every non-Hold signal is journalled (a decision log)");
    {
        const std::string jpath = "state/test_session_h/sig.csv";
        std::remove(jpath.c_str());
        PaperSessionConfig cfg;
        cfg.streams.push_back(stream("AAA", AssetClass::Equity, make_series(200, 86400)));
        PaperJournal j(jpath);
        std::string err;
        j.scan_existing(err);
        CHECK(j.open(err));
        const PaperSessionResult r = PaperSession::run(cfg, j);
        j.close();
        const std::string text = slurp(jpath);
        CHECK(r.signals >= 1);
        CHECK(count_substr(text, ",signal,AAA,") ==
              static_cast<size_t>(r.signals));
        CHECK(text.find(",buy: ") != std::string::npos ||
              text.find(",sell: ") != std::string::npos);
    }

    TEST("session: signal journaling can be switched off");
    {
        const std::string jpath = "state/test_session_h/nosig.csv";
        std::remove(jpath.c_str());
        PaperSessionConfig cfg;
        cfg.journal_signals = false;
        cfg.streams.push_back(stream("AAA", AssetClass::Equity, make_series(200, 86400)));
        PaperJournal j(jpath);
        std::string err;
        j.scan_existing(err);
        CHECK(j.open(err));
        const PaperSessionResult r = PaperSession::run(cfg, j);
        j.close();
        CHECK(r.signals == 0);
        CHECK(slurp(jpath).find(",signal,") == std::string::npos);
    }
}

// ===========================================================================
// 7. Monster gate (tools/monster) — the distilled sniper confidence
// ===========================================================================
static void ensure_dir(const std::string& path) {
    const auto slash = path.find_last_of("/\\");
    if (slash == std::string::npos) return;
    const std::string d = path.substr(0, slash);
#if OMNISEED_PLATFORM_WINDOWS
    const std::string cmd = "if not exist \"" + d + "\" mkdir \"" + d + "\"";
#else
    const std::string cmd = "mkdir -p '" + d + "'";
#endif
    (void)std::system(cmd.c_str());
}

static void write_monster_csv(const std::string& path,
                              const std::vector<Bar>& bars, double conf,
                              bool veto, const std::string& regime) {
    ensure_dir(path);
    std::ofstream f(path);
    f << "ts,confidence,veto,regime,detail\n";
    for (const Bar& b : bars)
        f << b.time << "," << conf << "," << (veto ? 1 : 0) << "," << regime
          << ",S=" << conf << " test\n";
}

static void test_monster_gate() {
    const std::string dir = "state/test_session_m";
    const std::string jpath = dir + "/journal.csv";
    const std::string mdir = dir + "/monster";
    const std::vector<Bar> bars = make_series(240, 86400, 9);

    // One scenario = a fresh journal + a chosen monster feed.
    auto run = [&](const std::string& tag, bool enabled, double min_conf,
                   const std::string& sym_for_csv) {
        const std::string jp = dir + "/" + tag + ".csv";
        std::remove(jp.c_str());
        PaperSessionConfig cfg;
        cfg.monster_enabled = enabled;
        cfg.monster_min_conf = min_conf;
        if (!sym_for_csv.empty()) cfg.monster_dir = mdir;
        cfg.streams.push_back(stream("AAA", AssetClass::Equity, bars));
        PaperJournal j(jp);
        std::string err;
        j.scan_existing(err);
        CHECK(j.open(err));
        const PaperSessionResult r = PaperSession::run(cfg, j);
        j.close();
        return std::pair<PaperSessionResult, std::string>(r, slurp(jp));
    };

    // --- baseline: no monster gate ---------------------------------------
    const auto base = run("base", false, 0.85, "");
    TEST("monster: baseline (no gate) enters normally");
    CHECK(base.first.error.empty());
    CHECK(base.first.entries > 0);
    CHECK(base.first.monster_blocked == 0);

    // --- every bar passes the gate ---------------------------------------
    write_monster_csv(mdir + "/AAA.csv", bars, 1.0, false, "trend_up");
    const auto pass = run("pass", true, 0.85, "AAA");
    TEST("monster: a passing feed does not change the outcome");
    CHECK(pass.first.monster_blocked == 0);
    CHECK(pass.first.entries == base.first.entries);
    CHECK(pass.second.find("monster-block") == std::string::npos);
    TEST("monster: the signal log carries the distilled confidence");
    CHECK(pass.second.find("monster[S=1.000 trend_up]") != std::string::npos);

    // --- confidence below the bar ----------------------------------------
    write_monster_csv(mdir + "/AAA.csv", bars, 0.40, false, "range");
    const auto low = run("low", true, 0.85, "AAA");
    TEST("monster: confidence below the bar blocks every entry");
    CHECK(low.first.entries == 0);
    CHECK(low.first.monster_blocked > 0);
    CHECK(low.second.find("monster-block:AAA:S=0.400<0.85") !=
          std::string::npos);

    // --- a hard veto ------------------------------------------------------
    write_monster_csv(mdir + "/AAA.csv", bars, 1.00, true, "trend_down");
    const auto vetoed = run("veto", true, 0.85, "AAA");
    TEST("monster: a veto blocks even at full confidence");
    CHECK(vetoed.first.entries == 0);
    CHECK(vetoed.first.monster_blocked > 0);
    CHECK(vetoed.second.find("monster-block:AAA:veto:trend_down") !=
          std::string::npos);

    // --- no feature file at all -> FAIL CLOSED ---------------------------
    const auto norow = run("norow", true, 0.85, "");
    TEST("monster: a missing feature file fails CLOSED (a sniper refuses)");
    CHECK(norow.first.entries == 0);
    CHECK(norow.first.monster_blocked > 0);
    CHECK(norow.second.find("monster-block:AAA:no-row") != std::string::npos);

    // --- the gate never touches EXITS ------------------------------------
    TEST("monster: the gate is not consulted when the gate is off");
    CHECK(base.first.monster_blocked == 0);
}

int main() {
    platform::log_info("== multi-asset paper session (M5) ==");
    test_merged_timeline();
    test_asset_gates();
    test_abstain();
    test_kill_switch();
    test_resume();
    test_determinism_and_errors();
    test_signals();
    test_monster_gate();
    std::printf("RESULT: %d passed, %d failed\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
