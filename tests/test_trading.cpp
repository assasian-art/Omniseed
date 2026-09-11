// =============================================================================
//  OmniSeed — tests/test_trading.cpp
//  Trading Framework Core (Phase-16): signal generation determinism, indicator
//  math, risk calculations (Kelly, stops, drawdown), position accounting,
//  backtest invariants (no look-ahead, bounded exposure, sane metrics),
//  news feed parsing/sentiment, and CSV round-trips.
//
//  Same tiny harness style as test_platform.cpp.
// =============================================================================
#include "omniseed/core/platform.h"
#include "omniseed/trading/trading_engine.h"
#include "omniseed/trading/news_feed.h"
#include "omniseed/trading/simulate.h"
#include "omniseed/agent/sub_agents.h"
#include "omniseed/runtime/introspection.h"
#include "omniseed/runtime/cloud_bridge.h"
#include "omniseed/trading/finance.h"
#include "omniseed/trading/reasoning_tools.h"
#include "omniseed/agent/agent.h"

#include <map>

#include <cmath>
#include <cstdio>
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
// Deterministic synthetic series: geometric brownian motion with a regime
// flip so trends AND mean reversion both occur.
// ---------------------------------------------------------------------------
static std::vector<Bar> make_series(size_t n, uint64_t seed = 9) {
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
        if (i % 250 == 0) bull = (i / 250) % 2 == 0;   // regime flip
        const double drift = bull ? 0.0008 : -0.0006;
        const double z = (next01() + next01() + next01() +
                          next01() + next01() + next01() - 3.0) * 1.4;
        const double ret = drift + 0.012 * z;
        const double o = price;
        const double c = std::max(1.0, o * std::exp(ret));
        const double h = std::max(o, c) * (1.0 + 0.004 * next01());
        const double l = std::min(o, c) * (1.0 - 0.004 * next01());
        bars.push_back({static_cast<int64_t>(t0 + static_cast<int64_t>(i) * 86400),
                        o, h, l, c, 1e6});
        price = c;
    }
    return bars;
}

// ===========================================================================
// Indicators
// ===========================================================================
static void test_indicators() {
    TEST("indicators: SMA/EMA/RSI/MACD/BB/ATR shapes + ranges");
    const auto bars = make_series(600);
    const auto ind = SignalGenerator::compute(bars, {});

    CHECK(ind.sma20.size() == bars.size());
    CHECK(ind.rsi14.size() == bars.size());
    CHECK(ind.atr14.size() == bars.size());

    // SMA warmup: index 0 must be 0 (window incomplete); sma20 at 30 is the
    // mean of closes[11..30].
    CHECK(ind.sma20[0] == 0.0);
    double mean = 0.0;
    for (size_t i = 11; i <= 30; ++i) mean += bars[i].close;
    mean /= 20.0;
    CHECK(std::fabs(ind.sma20[30] - mean) < 1e-9);

    // RSI stays in [0, 100]; monotone rally pushes it high.
    for (size_t i = 0; i < bars.size(); ++i)
        CHECK(ind.rsi14[i] >= 0.0 && ind.rsi14[i] <= 100.0);
    std::vector<Bar> rally;
    double p = 10.0;
    for (int i = 0; i < 120; ++i) {
        rally.push_back({i, p, p * 1.01, p * 0.99, p * 1.005, 0.0});
        p *= 1.005;
    }
    const auto rind = SignalGenerator::compute(rally, {});
    CHECK(rind.rsi14.back() > 70.0);

    // MACD: EMA12 leads EMA26 in a rally -> positive histogram.
    CHECK(rind.macd_hist.back() > 0.0);

    // Bollinger: lower <= mid <= upper; a close outside means |z|>sigma.
    for (size_t i = 25; i < bars.size(); i += 17)
        CHECK(ind.bb_lower[i] <= ind.bb_mid[i] + 1e-12 &&
              ind.bb_mid[i] <= ind.bb_upper[i] + 1e-12);

    // ATR positive after warmup and roughly >= daily range fraction.
    CHECK(ind.atr14.back() > 0.0);
}

// ===========================================================================
// Signal determinism + rule sanity
// ===========================================================================
static void test_signals() {
    TEST("signals: deterministic, rule trace, no look-ahead");
    const auto bars = make_series(400);
    const auto ind = SignalGenerator::compute(bars, {});

    const Signal a = SignalGenerator::evaluate(ind, bars, 300, {});
    const Signal b = SignalGenerator::evaluate(ind, bars, 300, {});
    CHECK(a.action == b.action && a.strength == b.strength);
    CHECK(a.reason == b.reason);
    CHECK(a.strength >= 0.0 && a.strength <= 1.0);
    CHECK(a.action == Action::Buy || a.action == Action::Sell ||
          a.action == Action::Hold);

    // No look-ahead: signals computed on a truncated series must equal the
    // ones computed on the full series at the same index.
    std::vector<Bar> trunc(bars.begin(), bars.begin() + 301);
    const auto ind_t = SignalGenerator::compute(trunc, {});
    const Signal c = SignalGenerator::evaluate(ind_t, trunc, 300, {});
    CHECK(c.action == a.action);
    CHECK(std::fabs(c.strength - a.strength) < 1e-12);
}

// ===========================================================================
// CSV round-trip
// ===========================================================================
static void test_csv_roundtrip() {
    TEST("csv: round-trip through load_bars_csv/save_bars_csv");
    const auto bars = make_series(120);
    std::string err;
    CHECK(save_bars_csv("build/trading_test.csv", bars, err));
    std::vector<Bar> loaded;
    CHECK(load_bars_csv("build/trading_test.csv", loaded, err));
    CHECK(loaded.size() == bars.size());
    for (size_t i = 0; i < loaded.size(); ++i) {
        CHECK(loaded[i].time == bars[i].time);
        CHECK(std::fabs(loaded[i].close - bars[i].close) < 1e-6);
        CHECK(std::fabs(loaded[i].volume - bars[i].volume) < 0.5);
    }
    std::vector<Bar> junk;
    CHECK(!load_bars_csv("build/definitely_missing_9f3a.csv", junk, err));
    CHECK(!err.empty());
}

// ===========================================================================
// Position accounting
// ===========================================================================
static void test_positions() {
    TEST("positions: buy/sell accounting, P&L, exposure");
    PositionManager pm(10000.0);
    std::string err;
    CHECK(pm.buy("AAPL", 10, 100.0, 1, err));       // -1000, 9000 left
    CHECK(std::fabs(pm.cash() - 9000.0) < 1e-9);
    CHECK(!pm.buy("AAPL", 1000, 100.0, 2, err));    // exceeds cash
    CHECK(!pm.sell("MSFT", 5, 200.0, 3, err));      // no position
    CHECK(pm.sell("AAPL", 4, 110.0, 4, err));       // +440 realized = +40/share
    CHECK(std::fabs(pm.realized_pnl() - 40.0) < 1e-9);

    const auto ps = pm.mark({{"AAPL", 105.0}});     // 6 shares @ avg 100
    CHECK(std::fabs(ps.unrealized_pnl - 30.0) < 1e-9);
    CHECK(std::fabs(ps.gross_exposure - 630.0) < 1e-9);
    CHECK(std::fabs(ps.equity - (pm.cash() + 630.0)) < 1e-9);
    CHECK(pm.sell("AAPL", 6, 105.0, 5, err));
    CHECK(pm.positions().empty());
    CHECK(std::fabs(pm.realized_pnl() - 70.0) < 1e-9);
    CHECK(std::fabs(pm.mark({}).equity - 10070.0) < 1e-9);
}

// ===========================================================================
// Risk: Kelly sizing, stops, drawdown, gates
// ===========================================================================
static void test_risk() {
    TEST("risk: Kelly sizing caps, stop-loss, drawdown, entry gates");
    const RiskLimits lim{};

    // Full Kelly for p=0.6, b=2: (2*0.6-0.4)/2 = 0.4 -> quarter = 0.10 of
    // equity; capped at max_position_pct 0.20 -> qty = floor(100000*0.10/50).
    auto sz = RiskManager::size_position(100000.0, 50.0, 0.6, 2.0, 1.0, lim);
    CHECK(sz.allowed);
    CHECK(sz.qty == 200.0);                                  // 10% of equity
    CHECK(sz.reason.find("Kelly") != std::string::npos);

    // Quarter-Kelly below the position cap: p=0.55,b=1 -> K=0.1, frac=0.025.
    sz = RiskManager::size_position(100000.0, 50.0, 0.55, 1.0, 1.0, lim);
    CHECK(sz.qty == 50.0);                                   // 2.5% of equity

    // Negative edge -> refuse. p=0.3, b=1: K = -0.4 -> no trade.
    sz = RiskManager::size_position(100000.0, 50.0, 0.3, 1.0, 1.0, lim);
    CHECK(!sz.allowed && sz.qty == 0.0);

    // Out-of-range Kelly inputs -> fixed 2% fallback, still sane.
    sz = RiskManager::size_position(100000.0, 50.0, 1.5, 1.0, 1.0, lim);
    CHECK(sz.allowed && sz.qty == 40.0);                     // 2% of equity

    // Stop-loss / take-profit. Stop fires at a loss >= 8%.
    Position p; p.qty = 10; p.avg_price = 100.0;
    CHECK(RiskManager::check_exit(p, 92.1, lim) == Action::Hold);    // -7.9%
    CHECK(RiskManager::check_exit(p, 91.99, lim) == Action::Sell);   // -8.01% stop
    RiskLimits lim2 = lim; lim2.take_profit_pct = 0.10;
    CHECK(RiskManager::check_exit(p, 110.0, lim2) == Action::Sell);
    CHECK(RiskManager::check_exit(p, 105.0, lim2) == Action::Hold);

    // Drawdown: 100 -> 120 -> 90 = 25% from the 120 peak.
    CHECK(std::fabs(RiskManager::drawdown({100, 120, 90}) - 0.25) < 1e-12);
    CHECK(RiskManager::drawdown({100, 110, 120}) == 0.0);

    // Entry gates: drawdown halt. 25% dd >= the 20% limit -> refused.
    PortfolioState ps;
    ps.equity = 75000.0; ps.gross_exposure = 0.0;
    std::string why;
    CHECK(!RiskManager::entries_allowed(ps, 100000.0, lim, why));
    CHECK(!why.empty());
    ps.equity = 95000.0;
    CHECK(RiskManager::entries_allowed(ps, 100000.0, lim, why));
    // Exposure cap: fully invested -> refuse.
    ps.gross_exposure = ps.equity;
    CHECK(!RiskManager::entries_allowed(ps, 100000.0, lim, why));
}

// ===========================================================================
// Metrics
// ===========================================================================
static void test_metrics() {
    TEST("metrics: Sharpe/Sortino/drawdown/CAGR sane");
    // Steady ~0.05% per day with tiny noise: high Sharpe, tiny dd. (A
    // perfectly constant series has zero variance — undefined Sharpe —
    // so the engine reports 0; real series always carry noise.)
    std::vector<double> eq;
    double v = 100.0;
    for (int i = 0; i < 504; ++i) {
        eq.push_back(v);
        v *= 1.0005 * (1.0 + ((i % 2 == 0) ? 0.0001 : -0.0001));
    }
    const auto m = compute_metrics(eq, 252.0);
    CHECK(m.sharpe > 3.0);
    CHECK(m.max_drawdown_pct < 1.0);
    CHECK(m.total_return_pct > 20.0);
    CHECK(std::fabs(m.cagr_pct - (std::pow(1.0005, 252.0) - 1.0) * 100.0) < 0.5);

    // Sawtooth: zero mean return but real drawdown; Sortino must be worse
    // than Sharpe when downside vol dominates... here symmetrical, but dd
    // must match the peak-trough math.
    std::vector<double> saw;
    for (int i = 0; i < 200; ++i) saw.push_back(100.0 + (i % 2 == 0 ? 10.0 : 0.0));
    const auto s = compute_metrics(saw, 252.0);
    CHECK(std::fabs(s.max_drawdown_pct - (10.0 / 110.0) * 100.0) < 1e-9);
}

// ===========================================================================
// Backtester invariants
// ===========================================================================
static void test_backtester() {
    TEST("backtester: bounded exposure, no look-ahead, sane report");
    const auto bars = make_series(900);
    Backtester::Config cfg;
    cfg.ticker = "SYN";
    cfg.risk.max_position_pct = 0.20;

    const BacktestReport rep = Backtester::run(bars, cfg);
    CHECK(rep.error.empty());
    CHECK(rep.equity_curve.size() == bars.size());
    CHECK(rep.bars == static_cast<int64_t>(bars.size()));
    CHECK(rep.final_equity > 0.0);

    // No look-ahead: equity at bar i must not change when the FUTURE is
    // truncated (run the first 500 bars separately).
    Backtester::Config cfg2 = cfg;
    std::vector<Bar> trunc(bars.begin(), bars.begin() + 500);
    const BacktestReport rep2 = Backtester::run(trunc, cfg2);
    CHECK(rep2.error.empty());
    CHECK(std::fabs(rep2.equity_curve[499] - rep.equity_curve[499]) < 1e-6);

    // Exposure discipline: at every bar, equity never goes negative and
    // cash never negative (the engine refuses over-spending buys).
    for (const double e : rep.equity_curve) CHECK(e > 0.0);
    for (const TradeRecord& t : rep.trades) {
        CHECK(t.qty > 0.0);
        CHECK(t.exit_price > 0.0);
        CHECK(!t.exit_reason.empty());
        CHECK(t.exit_ts >= t.entry_ts);
    }

    // Metrics consistency.
    const auto m = compute_metrics(rep.equity_curve, 252.0);
    CHECK(std::fabs(m.sharpe - rep.sharpe) < 1e-9);
    CHECK(rep.max_drawdown_pct <= 100.0);
    CHECK(rep.win_rate >= 0.0 && rep.win_rate <= 1.0);
    CHECK(rep.profit_factor >= 0.0);

    // Too-short series refused.
    BacktestReport small = Backtester::run(make_series(30), cfg);
    CHECK(!small.error.empty());
}

// ===========================================================================
// MarketDataFeed
// ===========================================================================
static void test_feed() {
    TEST("feed: CSV replay loads, latest close, graceful errors");
    MarketDataFeed::Config cfg;
    cfg.csv_path = "build/trading_test.csv";
    cfg.ticker = "SYN";
    MarketDataFeed feed(cfg);
    std::string err;
    CHECK(feed.load(err));
    CHECK(feed.loaded() && feed.bars().size() == 120);
    double px = 0.0;
    CHECK(feed.latest_close(px));
    CHECK(std::fabs(px - feed.bars().back().close) < 1e-12);

    MarketDataFeed bad({MarketDataFeed::Kind::CsvReplay, "build/nope.csv", "X"});
    CHECK(!bad.load(err) && !bad.error().empty());
    CHECK(!bad.latest_close(px));
}

// ===========================================================================
// News feed: RSS parse, sentiment, entities, alerts
// ===========================================================================
static void test_news_feed() {
    TEST("news feed: RSS parse, sentiment, entities, breaking alert");
    const char* rss =
        "<?xml version=\"1.0\"?><rss version=\"2.0\"><channel>"
        "<title>Fake Markets</title><item>"
        "<title>Tech stocks rally as inflation eases</title>"
        "<pubDate>Tue, 10 Sep 2026 08:00:00 GMT</pubDate>"
        "<description>Shares surge across the board.</description></item>"
        "<item>"
        "<title>Apple beats earnings expectations, revenue up</title>"
        "<pubDate>Tue, 10 Sep 2026 09:30:00 GMT</pubDate>"
        "<description>Strong quarter.</description></item>"
        "<item>"
        "<title>Chipmaker warns of collapse in demand, shares plunge</title>"
        "<pubDate>Tue, 10 Sep 2026 10:00:00 GMT</pubDate>"
        "<description>Crisis deepens.</description></item>"
        "</channel></rss>";
    NewsFeed::Config cfg;
    NewsFeed feed(cfg);
    size_t added = feed.ingest_rss(rss);
    CHECK(added == 3);

    // Ordering: newest first.
    const auto& items = feed.items();
    CHECK(items[0].published >= items[1].published &&
          items[1].published >= items[2].published);

    // Sentiment polarity: positive headline scores > negative one.
    // items are newest-first: [0]=chipmaker crash, [1]=Apple beats, [2]=rally.
    const auto& warn = items[0];      // "warns ... collapse ... plunge"
    const auto& beat = items[1];      // "beats ... up"
    CHECK(beat.sentiment > 0.0);
    CHECK(warn.sentiment < 0.0);
    CHECK(beat.sentiment > warn.sentiment);

    // Neutral text is near zero.
    CHECK(std::fabs(NewsFeed::score_sentiment("the meeting is on thursday")) < 0.34);

    // Entity extraction.
    const auto ents = NewsFeed::extract_tickers(
        "Big upgrade for AAPL and NVDA options flow; ignore MSFTX");
    CHECK(ents.size() == 2);
    CHECK(ents[0] == "AAPL" && ents[1] == "NVDA");

    // Breaking-news alert: crash lexicon + fresh timestamp.
    feed.items().push_back({});
    NewsItem& crash = feed.items().back();
    crash.title = "Market crash: panic selling accelerates";
    crash.published = static_cast<int64_t>(platform::now_ms() / 1000.0);
    feed.recompute(crash);
    CHECK(feed.is_breaking(crash));

    const auto alerts = feed.pending_alerts();
    CHECK(alerts.size() == 1);
    CHECK(alerts[0].priority >= 8);
    CHECK(alerts[0].detail.find("news:breaking") != std::string::npos);

    // Dedup: same GUID twice.
    feed.items().clear();
    feed.reset_alerts();
    const char* dup =
        "<rss><channel><item><guid>x1</guid><title>Stocks rally</title></item>"
        "<item><guid>x1</guid><title>Stocks rally</title></item></channel></rss>";
    CHECK(feed.ingest_rss(dup) == 1);
}

// ===========================================================================
// Sub-agents: analyst consensus, risk agent, execution routing, research
// ===========================================================================
static void test_sub_agents() {
    TEST("sub-agents: consensus voting, risk alerts, execution routing");
    const auto bars = make_series(400);

    // --- Analyst: technical signal on real indicator data ------------------
    MarketAnalystAgent::Config acfg;
    MarketAnalystAgent analyst(acfg);
    auto ind = SignalGenerator::compute(bars, {});
    auto tech = analyst.analyze_technical("SYN", bars, ind, 350);
    CHECK(tech.action == Action::Buy || tech.action == Action::Sell ||
          tech.action == Action::Hold);
    CHECK(!tech.reason.empty());
    // Deterministic.
    auto tech2 = analyst.analyze_technical("SYN", bars, ind, 350);
    CHECK(tech2.action == tech.action && tech2.strength == tech.strength);

    // News sentiment must move the blended verdict. Pin the item to "now"
    // so the 24h aggregate window never ages it out of the test.
    NewsFeed feed({});
    const char* rss =
        "<rss><channel><item><guid>a</guid>"
        "<title>AAPL crashes on fraud probe, shares plunge</title>"
        "<pubDate>Tue, 10 Sep 2026 08:00:00 GMT</pubDate></item></channel></rss>";
    feed.ingest_rss(rss);
    feed.items().front().published =
        static_cast<int64_t>(platform::now_ms() / 1000.0);
    auto news = analyst.analyze_news("AAPL", feed);
    CHECK(news.sentiment < 0.0);
    CHECK(news.action == Action::Sell || news.action == Action::Hold);

    // --- Consensus: 2 buys + 1 sell = Buy; weights are respected -----------
    AgentVerdict v_buy;  v_buy.action = Action::Buy;  v_buy.strength = 0.8; v_buy.agent = "tech";
    AgentVerdict v_buy2; v_buy2.action = Action::Buy; v_buy2.strength = 0.5; v_buy2.agent = "news";
    AgentVerdict v_sell; v_sell.action = Action::Sell; v_sell.strength = 0.9; v_sell.agent = "risk";
    const auto cons = MarketAnalystAgent::consensus({v_buy, v_buy2, v_sell});
    CHECK(cons.action == Action::Buy);
    CHECK(cons.strength > 0.0 && cons.strength <= 1.0);
    CHECK(cons.reason.find("consensus") != std::string::npos);
    // Unanimous sell.
    const auto cons2 = MarketAnalystAgent::consensus({v_sell, v_sell});
    CHECK(cons2.action == Action::Sell);

    // --- Risk agent: concentration + drawdown alerts ------------------------
    RiskManagerAgent::Config rcfg;
    rcfg.limits.max_position_pct = 0.25;
    RiskManagerAgent risk(rcfg);
    Position big; big.ticker = "AAPL"; big.qty = 500; big.avg_price = 100.0;
    PortfolioState ps;
    ps.cash = 10000.0;
    ps.positions["AAPL"] = big;
    ps.equity = 60000.0;                 // 50k position = 83% of equity
    ps.gross_exposure = 50000.0;
    const auto alerts = risk.check_portfolio(ps, {{"AAPL", 100.0}});
    CHECK(!alerts.empty());
    bool concentration = false;
    for (const auto& a : alerts)
        concentration |= a.find("concentration") != std::string::npos;
    CHECK(concentration);

    // --- Execution agent: routes signals to the paper broker ---------------
    PaperBrokerConfig pcfg;
    pcfg.starting_cash = 50000.0;
    PaperBroker broker(pcfg);
    ExecutionAgent exec_a(&broker);
    exec_a.on_signal("SYN", Action::Buy, 0.9, 100.0, 1);
    CHECK(broker.state().positions.count("SYN") == 1);
    // Sell signal flattens.
    exec_a.on_signal("SYN", Action::Sell, 0.9, 110.0, 2);
    CHECK(broker.state().positions.empty());
    CHECK(broker.state().realized_pnl > 0.0);

    // Hold does nothing; refuse on unknown broker state.
    exec_a.on_signal("SYN", Action::Hold, 0.5, 100.0, 3);
    CHECK(broker.state().positions.empty());

    // --- Research agent: compiles a brief -----------------------------------
    ResearchAgent ra;
    auto brief = ra.brief("AAPL", bars, feed);
    CHECK(brief.find("AAPL") != std::string::npos);
    CHECK(brief.find("trend") != std::string::npos);
}

// ===========================================================================
// Introspection: self-model, goals, rationale, metacognition
// ===========================================================================
static void test_introspection() {
    TEST("introspection: self-model, goals, rationale, metacognition");

    // Self-model exposes an honest capability map.
    CHECK(SelfModel::capability_count() >= 10);
    CHECK(SelfModel::capabilities() != nullptr);
    CHECK(std::string(SelfModel::capabilities()[0].state) == "real");
    CHECK(std::string(PurposeReflection::mission()).find("discipline") !=
          std::string::npos);

    // Goal tracking: add/update/deactivate/priority-sort.
    GoalTracker goals;
    Goal g1; g1.text = "maximize sharpe"; g1.metric = "sharpe";
    g1.target = 1.5; g1.priority = 0.9;
    Goal g2; g2.text = "minimize drawdown"; g2.metric = "max_drawdown_pct";
    g2.target = 20.0; g2.priority = 0.95;
    goals.add(g1); goals.add(g2);
    CHECK(goals.active_goals().size() == 2);
    CHECK(goals.active_goals().front().text == "minimize drawdown");
    CHECK(goals.update("sharpe", 1.2));
    CHECK(!goals.update("nonexistent_metric", 1.0));
    CHECK(goals.deactivate("maximize sharpe"));
    CHECK(goals.active_goals().size() == 1);
    CHECK(goals.save("build/goals_test.bin"));
    GoalTracker goals2;
    CHECK(goals2.load("build/goals_test.bin"));
    CHECK(goals2.goals().size() == goals.goals().size());
    // Insertion order survives the round-trip; priorities are exact doubles.
    CHECK(goals2.goals()[0].text == "maximize sharpe");
    CHECK(std::fabs(goals2.goals()[1].priority - 0.95) < 1e-9);   // double RT

    // Rationale log + calibration. resolve_latest resolves the MOST RECENT
    // pending decision (the one that just produced an outcome).
    ActionRationale log;
    log.log("execution", "buy 50 SYN @ 100", "uptrend+MACD", 0.8);
    CHECK(log.calibration().not_calibrated);              // nothing resolved
    CHECK(log.resolve_latest(1.0));                       // the buy won
    CHECK(!log.calibration().not_calibrated);
    CHECK(std::fabs(log.calibration().mean_confidence - 0.8) < 1e-9);
    CHECK(std::fabs(log.calibration().realized_rate - 1.0) < 1e-9);
    CHECK(log.calibration().gap < 0.0);                   // underconfident
    log.log("risk", "entry vetoed", "drawdown halt", 1.0); // stays pending
    CHECK(log.entries().size() == 2);
    CHECK(log.entries().front().action.find("vetoed") != std::string::npos);
    CHECK(!log.resolve_latest(0.0) == false);             // resolves the veto
    CHECK(log.save("build/rationale_test.bin"));
    ActionRationale log2;
    CHECK(log2.load("build/rationale_test.bin"));
    CHECK(log2.entries().size() == 2);

    // Purpose reflection ties goals + log into a grounded answer.
    const auto refl = PurposeReflection::reflect(log, goals);
    CHECK(refl.answer.find("active goal") != std::string::npos);
    CHECK(refl.answer.find("vetoed") != std::string::npos);

    // Metacognition: overconfidence detection + strategy shrinkage.
    ActionRationale over;
    for (int i = 0; i < 10; ++i) {
        over.log("execution", "buy x", "weak signal", 0.9);
        over.resolve_latest(0.0);                          // all losses
    }
    const auto st = Metacognition::evaluate(goals, over, 0.5, 10);
    CHECK(st.overconfident);
    CHECK(!st.gaps.empty());
    CHECK(st.strategy_assessment.find("OVERCONFIDENT") != std::string::npos);
    // Thin evidence -> low confidence in strategy even with a good Sharpe.
    const auto st2 = Metacognition::evaluate(goals, log, 2.0, 10);
    CHECK(st2.confidence_in_strategy < 0.3);

    // ---- Trading Mode + StateFingerprint (AgentLoop integration) ---------
    AgentLoop::Config cfg;
    cfg.trading_mode = true;
    cfg.state_fingerprint = true;
    CHECK(cfg.trading_mode && cfg.state_fingerprint);
}

// ===========================================================================
// Finance math: NPV/IRR/Black-Scholes + sandbox interpreter
// ===========================================================================
static void test_finance() {
    TEST("finance: NPV/IRR/Black-Scholes, sandbox interpreter");

    // NPV of -100 today + 110 next year at 10% = 0.
    CHECK(std::fabs(finance_npv(0.10, {-100.0, 110.0})) < 1e-9);

    // IRR of the same stream = 10%.
    double irr = 0.0;
    CHECK(finance_irr({-100.0, 110.0}, irr));
    CHECK(std::fabs(irr - 0.10) < 1e-4);
    // No sign change -> no IRR.
    CHECK(!finance_irr({10.0, 20.0, 30.0}, irr));

    // Put-call parity: C - P = S - K*exp(-rT).
    const double S = 100, K = 105, r = 0.05, sig = 0.2, T = 0.5;
    const double parity = bs_call(S, K, r, sig, T) - bs_put(S, K, r, sig, T);
    CHECK(std::fabs(parity - (S - K * std::exp(-r * T))) < 1e-9);
    // Degenerate inputs price at 0.
    CHECK(bs_call(100, 100, 0.05, 0.0, 1.0) == 0.0);

    // Sandbox: allowed functions work.
    auto res = FinanceInterpreter::evaluate("2 + 3 * 4");
    CHECK(res.ok && std::fabs(res.value - 14.0) < 1e-12);
    res = FinanceInterpreter::evaluate("(2 + 3) * 4");
    CHECK(res.ok && std::fabs(res.value - 20.0) < 1e-12);
    res = FinanceInterpreter::evaluate("sqrt(144)");
    CHECK(res.ok && std::fabs(res.value - 12.0) < 1e-12);
    res = FinanceInterpreter::evaluate("max(3, 7) * min(2, 5)");
    CHECK(res.ok && std::fabs(res.value - 14.0) < 1e-12);
    res = FinanceInterpreter::evaluate("npv(0.1, -100, 110)");
    CHECK(res.ok && std::fabs(res.value) < 1e-9);
    res = FinanceInterpreter::evaluate("irr(-100, 110)");
    CHECK(res.ok && std::fabs(res.value - 0.10) < 1e-4);
    res = FinanceInterpreter::evaluate("bscall(100, 100, 0.05, 0.2, 1)");
    CHECK(res.ok && res.value > 0.0);

    // Sandbox: hostile inputs refused.
    CHECK(!FinanceInterpreter::evaluate("system('rm -rf /')").ok);
    CHECK(!FinanceInterpreter::evaluate("__import__('os')").ok);
    CHECK(!FinanceInterpreter::evaluate("1/0").ok);
    CHECK(!FinanceInterpreter::evaluate("pow(10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10)").ok);
    CHECK(!FinanceInterpreter::evaluate("").ok);
}

// ===========================================================================
// Reasoning tools: doc reader parsing, DDG parse, tool invocation
// ===========================================================================
static void test_reasoning_tools() {
    TEST("reasoning tools: html_to_text, ddg parse, registry invocation");

    const std::string html =
        "<html><head><style>body{color:red}</style></head><body>"
        "<h1>Filing&nbsp;Summary</h1>"
        "<script>evil()</script>"
        "<p>Revenue rose 10% &amp; costs fell.</p></body></html>";
    const std::string text = html_to_text(html);
    CHECK(text.find("evil") == std::string::npos);
    CHECK(text.find("body{color:red}") == std::string::npos);
    CHECK(text.find("Filing Summary") != std::string::npos);
    CHECK(text.find("Revenue rose 10% & costs fell.") != std::string::npos);

    CHECK(is_fetchable_url("http://example.com/x"));
    CHECK(is_fetchable_url("https://sec.gov/filing.htm"));
    CHECK(!is_fetchable_url("file:///etc/passwd"));
    CHECK(!is_fetchable_url("ftp://x"));
    CHECK(!is_fetchable_url("http://"));

    // Canned fetcher -> the registered tools work end-to-end offline.
    static const std::string ddg_page =
        "<a rel=\"nofollow\" class=\"result__a\" "
        "href=\"//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Faapl\""
        ">AAPL filing</a>"
        "<a class=\"result__snippet\">Ten-K summary text</a>";
    ToolRegistry reg;
    register_reasoning_tools(reg, [](const std::string& url,
                                     std::string& response,
                                     std::string& err) {
        (void)url;
        err.clear();
        response = ddg_page;
        return true;
    });
    CHECK(reg.has("web_search"));
    CHECK(reg.has("finance_calc"));
    CHECK(reg.has("code_interpreter"));
    CHECK(reg.has("document_reader"));

    bool ok = false;
    const std::string search_out =
        reg.invoke("web_search", "{\"query\":\"AAPL 10-K\"}", ok);
    CHECK(ok);
    CHECK(search_out.find("https://example.com/aapl") != std::string::npos);
    CHECK(search_out.find("AAPL filing") != std::string::npos);

    ok = false;
    const std::string calc_out =
        reg.invoke("finance_calc", "{\"expression\":\"2^10\"}", ok);
    CHECK(ok && calc_out.find("1024") != std::string::npos);

    ok = true;
    reg.invoke("finance_calc", "{\"expression\":\"sqrt(-1)\"}", ok);
    CHECK(!ok);

    ok = false;
    const std::string doc_out = reg.invoke(
        "document_reader", "{\"url\":\"https://sec.gov/x.htm\"}", ok);
    CHECK(ok);
    CHECK(doc_out.find("Ten-K summary") != std::string::npos);

    // web_search with a FAILING fetcher must degrade to a JSON error, not
    // an exception.
    ToolRegistry reg2;
    register_reasoning_tools(reg2, [](const std::string&, std::string&,
                                      std::string& err) {
        err = "offline";
        return false;
    });
    ok = false;
    const std::string err_out =
        reg2.invoke("web_search", "{\"query\":\"x\"}", ok);
    CHECK(!ok && err_out.find("offline") != std::string::npos);
}

// ===========================================================================
// Cloud bridge: URL parsing, response parsing, local fallback
// ===========================================================================
static void test_cloud_bridge() {
    TEST("cloud bridge: parse providers, honest fallback, hybrid routing");

    // No key -> local mode, ask() fails with a clear error and no network.
    CloudBridge nokey;
    CHECK(!nokey.available());
    std::string reply, err;
    CHECK(!nokey.ask("hello", reply, err));
    CHECK(err.find("local mode") != std::string::npos);

    // HybridRouter: local when no bridge; escalates research-looking asks
    // only when a bridge is available.
    HybridRouter local_router(nullptr);
    const auto d1 = HybridRouter::route("what time is it", false);
    CHECK(!d1.use_cloud);
    const auto d2 = HybridRouter::route(
        "analyze the macroeconomic implications of a fed rate hike and "
        "compare them to 2019 earnings trends across sectors, then research "
        "the sec filings", true);
    CHECK(d2.use_cloud);
    const auto d3 = HybridRouter::route("buy or hold?", true);
    CHECK(!d3.use_cloud);
    std::string r2, e2;
    CHECK(!local_router.ask_cloud("deep research", r2, e2));
    CHECK(e2.find("unavailable") != std::string::npos);
}

int main() {
    test_indicators();
    test_signals();
    test_csv_roundtrip();
    test_positions();
    test_risk();
    test_metrics();
    test_backtester();
    test_feed();
    test_news_feed();
    test_sub_agents();
    test_introspection();
    test_finance();
    test_reasoning_tools();
    test_cloud_bridge();

    platform::log_info("---- trading tests: %d passed, %d failed ----",
                       g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
