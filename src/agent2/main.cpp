// =============================================================================
//  OmniSeed — agent2/main.cpp
//  Trading + introspection CLI (Phase-16 "Omega Pass").
//
//  Commands:
//    trading-analyze   --ticker AAPL --timeframe 1d [--backtest-years 5]
//                      [--csv PATH] [--news PATH] [--capital 100000]
//    trading-watch     --tickers AAPL,MSFT,TSLA --interval 60 [--cycles N]
//                      [--csv-dir models/market] [--news-dir models/news]
//    trading-simulate  --ticker AAPL --timeframe 1d --capital 100000
//                      [--strategy momentum|mean-reversion|balanced]
//    introspect        self-model + goals + decision rationale + purpose
//    metacognition     confidence calibration + knowledge gaps
//    cloud             cloud-reasoning bridge (optional; Phase-3)
//
//  HONESTY: every trading command prints the risk disclaimer. This software
//  minimizes losses through discipline; it does not eliminate them, and it
//  never places real orders (paper only).
// =============================================================================
#include "omniseed/core/platform.h"
#include "omniseed/agent/sub_agents.h"
#include "omniseed/runtime/introspection.h"
#include "omniseed/trading/news_feed.h"
#include "omniseed/trading/simulate.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace omniseed;
using namespace omniseed::trading;

namespace {

constexpr const char* kDisclaimer =
    "\n  [risk] Educational analytics. Losses are minimized by discipline,\n"
    "  never eliminated. Past backtests do not guarantee future results.\n"
    "  Paper trading only — no real orders are ever placed.\n";

void print_usage() {
    std::printf(
        "OmniSeed agent2 — trading + introspection CLI (Phase-16)\n"
        "usage: omniseed_agent2 <command> [options]\n"
        "\n"
        "commands:\n"
        "  trading-analyze   --ticker AAPL --timeframe 1d [--backtest-years 5]\n"
        "                    [--csv PATH] [--news PATH] [--capital 100000]\n"
        "  trading-watch     --tickers AAPL,MSFT --interval 60 [--cycles N]\n"
        "                    [--csv-dir models/market] [--news-dir models/news]\n"
        "  trading-simulate  --ticker AAPL --timeframe 1d --capital 100000\n"
        "                    [--strategy momentum|mean-reversion|balanced]\n"
        "  introspect        self-model, goals, rationale, purpose\n"
        "  metacognition     confidence calibration + knowledge gaps\n"
        "  cloud             cloud-reasoning bridge status (optional key)\n");
}

// ---------------------------------------------------------------------------
// Arg parsing helpers
// ---------------------------------------------------------------------------
struct Args {
    std::string ticker = "AAPL";
    std::string tickers;
    std::string timeframe = "1d";
    std::string csv;
    std::string news;
    std::string csv_dir = "models/market";
    std::string news_dir = "models/news";
    std::string strategy = "balanced";
    double years = 5.0;
    double capital = 100000.0;
    int32_t interval = 60;
    int32_t cycles = 1;             // 0 = infinite (watch)
};

Args parse_args(int argc, char** argv, int start) {
    Args a;
    for (int i = start; i + 1 < argc || i < argc; ++i) {
        const std::string s = i < argc ? argv[i] : "";
        const bool has_val = i + 1 < argc;
        const std::string v = has_val ? argv[i + 1] : "";
        if (s == "--ticker" && has_val) a.ticker = v;
        else if (s == "--tickers" && has_val) a.tickers = v;
        else if (s == "--timeframe" && has_val) a.timeframe = v;
        else if (s == "--csv" && has_val) a.csv = v;
        else if (s == "--news" && has_val) a.news = v;
        else if (s == "--csv-dir" && has_val) a.csv_dir = v;
        else if (s == "--news-dir" && has_val) a.news_dir = v;
        else if (s == "--strategy" && has_val) a.strategy = v;
        else if (s == "--backtest-years" && has_val) a.years = std::atof(v.c_str());
        else if (s == "--capital" && has_val) a.capital = std::atof(v.c_str());
        else if (s == "--interval" && has_val) a.interval = std::atoi(v.c_str());
        else if (s == "--cycles" && has_val) a.cycles = std::atoi(v.c_str());
        if (has_val && (s.rfind("--", 0) == 0)) ++i;
    }
    return a;
}

std::string default_csv(const std::string& ticker, const std::string& tf) {
    return "models/market/" + ticker + "_" + tf + ".csv";
}

std::string slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return "";
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Loads news XMLs from a file (single) or directory (all *.xml) into feed.
void load_news(NewsFeed& feed, const std::string& news_path,
               const std::string& news_dir) {
    if (!news_path.empty()) {
        const std::string xml = slurp(news_path);
        if (!xml.empty()) feed.ingest_rss(xml);
        return;
    }
    // Directory scan for *.xml (portable: platform-independent opendir via
    // std::filesystem would be C++17 but MSVC + gcc both support it; keep to
    // the two well-known default files to stay lean).
    for (const char* name : {"headlines.xml", "market.xml"}) {
        const std::string xml = slurp(news_dir + "/" + name);
        if (!xml.empty()) feed.ingest_rss(xml);
    }
}

// Strategy presets.
RiskLimits limits_for(const std::string& strategy) {
    RiskLimits lim{};
    if (strategy == "momentum") {
        lim.stop_loss_pct = 0.06;          // tighter stops for trend riding
        lim.take_profit_pct = 0.0;
        lim.kelly_fraction = 0.20;
    } else if (strategy == "mean-reversion") {
        lim.stop_loss_pct = 0.10;          // wider stops for fade entries
        lim.take_profit_pct = 0.06;
        lim.kelly_fraction = 0.15;
    }
    return lim;
}

// ===========================================================================
// trading-analyze
// ===========================================================================
int cmd_trading_analyze(const Args& a) {
    const std::string csv = a.csv.empty() ? default_csv(a.ticker, a.timeframe)
                                          : a.csv;
    std::vector<Bar> bars;
    std::string err;
    if (!load_bars_csv(csv, bars, err)) {
        std::printf("[analyze] no market data: %s\n", err.c_str());
        std::printf("  run: .venv/Scripts/python.exe tools/fetch_market_data.py"
                    " --ticker %s --timeframe %s\n",
                    a.ticker.c_str(), a.timeframe.c_str());
        return 2;
    }
    std::printf("[analyze] %s %s — %zu bars, last close %.2f\n",
                a.ticker.c_str(), a.timeframe.c_str(), bars.size(),
                bars.back().close);

    // Optional news context.
    NewsFeed feed;
    load_news(feed, a.news, a.news_dir);
    const double news_sent = feed.aggregate_sentiment(24.0 * 3600.0, a.ticker);
    if (feed.size() > 0)
        std::printf("[analyze] news 24h: %zu items, sentiment %.2f\n",
                    feed.size(), news_sent);

    // Indicators + current verdict.
    MarketAnalystAgent analyst;
    const Indicators ind = SignalGenerator::compute(bars, {});
    const size_t i = bars.size() - 1;
    const AgentVerdict tech = analyst.analyze_technical(a.ticker, bars, ind, i);
    const AgentVerdict news = analyst.analyze_news(a.ticker, feed);
    const AgentVerdict blended = analyst.blend(tech, news);
    std::printf("[analyze] technical : %s (%.2f) — %s\n",
                blended.action == Action::Buy ? "BUY"
              : blended.action == Action::Sell ? "SELL" : "HOLD",
                tech.strength, tech.reason.c_str());
    if (feed.size() > 0)
        std::printf("[analyze] news      : %s (%.2f)\n",
                    news.action == Action::Buy ? "BUY"
                  : news.action == Action::Sell ? "SELL" : "HOLD",
                    news.sentiment);

    // Backtest window (last N years) to ground the Kelly inputs honestly.
    std::vector<Bar> window = bars;
    if (a.years > 0.0 && !window.empty()) {
        const int64_t cutoff = window.back().time -
            static_cast<int64_t>(a.years * 365.25 * 86400.0);
        window.erase(std::remove_if(window.begin(), window.end(),
                                    [cutoff](const Bar& b) { return b.time < cutoff; }),
                     window.end());
    }
    Backtester::Config cfg;
    cfg.ticker = a.ticker;
    cfg.risk = limits_for(a.strategy);
    cfg.starting_cash = a.capital;
    const BacktestReport rep = Backtester::run(window, cfg);
    if (rep.error.empty()) {
        std::printf("[analyze] backtest %zu bars (%.1fy): return %.2f%%, "
                    "sharpe %.2f, maxDD %.2f%%, win %.1f%%, %zu trades\n",
                    window.size(), a.years, rep.total_return_pct, rep.sharpe,
                    rep.max_drawdown_pct, rep.win_rate * 100.0, rep.trades.size());
    } else {
        std::printf("[analyze] backtest skipped: %s\n", rep.error.c_str());
    }

    // Sizing for a fresh entry at the last close (quarter-Kelly from the
    // backtest's realized win stats; falls back to 2% when unproven).
    RiskManager::Sizing sz = RiskManager::size_position(
        a.capital, bars.back().close,
        rep.win_rate > 0.0 ? rep.win_rate : 0.0,
        rep.win_rate > 0.0 ? 1.0 : 0.0, 1.0, cfg.risk);
    std::printf("[analyze] sizing    : %s (%s)\n",
                sz.allowed ? std::to_string(sz.qty).c_str() : "no trade",
                sz.reason.c_str());
    std::printf("%s", kDisclaimer);
    return 0;
}

// ===========================================================================
// trading-simulate — paper backtest through the paper broker semantics
// ===========================================================================
int cmd_trading_simulate(const Args& a) {
    const std::string csv = a.csv.empty() ? default_csv(a.ticker, a.timeframe)
                                          : a.csv;
    std::vector<Bar> bars;
    std::string err;
    if (!load_bars_csv(csv, bars, err)) {
        std::printf("[simulate] no market data: %s\n", err.c_str());
        return 2;
    }
    Backtester::Config cfg;
    cfg.ticker = a.ticker;
    cfg.risk = limits_for(a.strategy);
    cfg.starting_cash = a.capital;
    const BacktestReport rep = Backtester::run(bars, cfg);
    if (!rep.error.empty()) {
        std::printf("[simulate] %s\n", rep.error.c_str());
        return 2;
    }
    std::printf("[simulate] %s %s — strategy %s, capital %.0f\n",
                a.ticker.c_str(), a.timeframe.c_str(), a.strategy.c_str(),
                a.capital);
    std::printf("  bars        : %lld (%.1f years)\n",
                static_cast<long long>(rep.bars),
                static_cast<double>(rep.bars) / 252.0);
    std::printf("  final       : %.2f (%+.2f%%, CAGR %+.2f%%)\n",
                rep.final_equity, rep.total_return_pct, rep.cagr_pct);
    std::printf("  sharpe      : %.2f   sortino: %.2f\n", rep.sharpe, rep.sortino);
    std::printf("  max drawdown: %.2f%%\n", rep.max_drawdown_pct);
    std::printf("  trades      : %zu, win rate %.1f%%, profit factor %.2f\n",
                rep.trades.size(), rep.win_rate * 100.0, rep.profit_factor);
    size_t shown = 0;
    for (auto it = rep.trades.rbegin();
         it != rep.trades.rend() && shown < 5; ++it, ++shown)
        std::printf("    %s %s %.0f @ %.2f -> %.2f (%+.2f%%) [%s]\n",
                    it->ticker.c_str(),
                    it->pnl >= 0 ? "+" : "-", std::fabs(it->qty),
                    it->entry_price, it->exit_price,
                    (it->exit_price / it->entry_price - 1.0) * 100.0,
                    it->exit_reason.c_str());
    std::printf("%s", kDisclaimer);
    return 0;
}

// ===========================================================================
// trading-watch — the multi-agent polling loop
// ===========================================================================
int cmd_trading_watch(const Args& a) {
    if (a.tickers.empty()) {
        std::printf("[watch] --tickers required, e.g. --tickers AAPL,MSFT\n");
        return 1;
    }
    std::vector<std::string> tickers;
    {
        std::stringstream ss(a.tickers);
        std::string t;
        while (std::getline(ss, t, ','))
            if (!t.empty()) tickers.push_back(t);
    }

    SensoryInterruptSystem interrupts;
    NewsFeed feed;
    NewsMonitorAgent news_agent(interrupts);
    RiskManagerAgent risk;
    PaperBrokerConfig pcfg;
    pcfg.starting_cash = a.capital;
    PaperBroker broker(pcfg);
    ExecutionAgent exec(&broker);
    MarketAnalystAgent analyst;
    ActionRationale rationale;                    // "I chose X because Y"
    rationale.load("state/rationale.bin");

    std::printf("[watch] tickers: ");
    for (const std::string& t : tickers) std::printf("%s ", t.c_str());
    std::printf("| interval %ds | paper capital %.0f\n", a.interval, a.capital);

    // Load whatever news we already have on disk.
    load_news(feed, a.news, a.news_dir);

    for (int32_t cycle = 0; a.cycles == 0 || cycle < a.cycles; ++cycle) {
        const double now = platform::now_ms() / 1000.0;

        // Refresh news from disk each cycle (the fetcher may have run).
        load_news(feed, a.news, a.news_dir);
        const size_t raised = news_agent.poll(feed, "");
        for (size_t r = 0; r < raised; ++r) {
            SensoryInterruptSystem::Interrupt iv;
            if (interrupts.poll(iv))
                std::printf("[watch] INTERRUPT p%d: %s\n",
                            iv.priority, iv.detail.c_str());
        }

        std::map<std::string, double> prices;      // this cycle's marks
        for (const std::string& t : tickers) {
            std::vector<Bar> bars;
            std::string err;
            if (!load_bars_csv(default_csv(t, a.timeframe), bars, err) ||
                bars.size() < 60) {
                std::printf("[watch] %s: no/stale data (%s)\n", t.c_str(),
                            err.empty() ? "too short" : err.c_str());
                continue;
            }
            const size_t i = bars.size() - 1;
            prices[t] = bars[i].close;
            broker.mark(prices, bars[i].time);      // live equity for risk
            const Indicators ind = SignalGenerator::compute(bars, {});
            const AgentVerdict tech = analyst.analyze_technical(t, bars, ind, i);
            const AgentVerdict news = analyst.analyze_news(t, feed);
            const AgentVerdict verdict = MarketAnalystAgent::consensus({tech, news});

            // Risk: portfolio state + veto on new entries.
            const PortfolioState ps = broker.state();
            std::string veto;
            const bool approved = verdict.action == Action::Hold ||
                risk.approve_entry(ps, broker.equity_peak(), t, bars[i].close, veto);
            std::string note;
            if (approved && verdict.action != Action::Hold) {
                const double realized_before = broker.state().realized_pnl;
                note = exec.on_signal(t, verdict.action, verdict.strength,
                                      bars[i].close, bars[i].time);
                if (!note.empty()) {
                    rationale.log("execution", t + ": " + note,
                                  verdict.reason.empty() ? "consensus"
                                                       : verdict.reason,
                                  verdict.strength);
                    // A closing sell resolves the previous entry's outcome
                    // (win/loss proxy) so calibration has real samples.
                    const double realized_delta =
                        broker.state().realized_pnl - realized_before;
                    if (realized_delta != 0.0)
                        rationale.resolve_latest(realized_delta > 0.0 ? 1.0 : -1.0);
                    rationale.save("state/rationale.bin");
                }
            } else if (!approved) {
                note = "VETO: " + veto;
                rationale.log("risk", t + ": entry vetoed", veto, 1.0);
            }

            // Risk alerts for the whole paper portfolio.
            const auto alerts = risk.check_portfolio(broker.state(), prices);
            for (const std::string& al : alerts)
                std::printf("[watch] ALERT: %s\n", al.c_str());

            std::printf("[watch] %s %s | close %.2f | %s (%.2f)%s%s\n",
                        t.c_str(), a.timeframe.c_str(), bars[i].close,
                        verdict.action == Action::Buy ? "BUY"
                      : verdict.action == Action::Sell ? "SELL" : "HOLD",
                        verdict.strength,
                        note.empty() ? "" : " | ", note.c_str());
        }
        std::printf("[watch] equity %.2f (peak %.2f) | rss %.1f MB\n",
                    broker.state().equity, broker.equity_peak(),
                    static_cast<double>(platform::current_rss_bytes()) / 1048576.0);
        std::printf("[watch] cycle %d done at %.0f\n%s", cycle, now, "");
        if (a.cycles == 0 || cycle + 1 < a.cycles)
            std::this_thread::sleep_for(
                std::chrono::seconds(std::max(1, a.interval)));
    }
    std::printf("%s", kDisclaimer);
    return 0;
}

// ===========================================================================
// introspect
// ===========================================================================
void seed_default_goals(GoalTracker& goals) {
    if (!goals.goals().empty()) return;
    Goal g1;
    g1.text = "maximize risk-adjusted return";
    g1.metric = "sharpe"; g1.target = 1.5; g1.current = 0.0; g1.priority = 0.9;
    Goal g2;
    g2.text = "minimize drawdown";
    g2.metric = "max_drawdown_pct"; g2.target = 20.0; g2.current = 0.0;
    g2.priority = 0.95;
    Goal g3;
    g3.text = "stay within the memory budget";
    g3.metric = "peak_rss_mb"; g3.target = 300.0; g3.current = 0.0;
    g3.priority = 0.7;
    goals.add(g1); goals.add(g2); goals.add(g3);
    goals.save("state/goals.bin");
}

int cmd_introspect() {
    // --- WHO AM I ---------------------------------------------------------
    std::printf("=== SELF-MODEL ===\n");
    std::printf("name    : %s v%s\n", SelfModel{}.name.c_str(),
                SelfModel{}.version.c_str());
    std::printf("identity: %s\n", SelfModel{}.identity.c_str());
    std::printf("mission : %s\n", SelfModel{}.mission.c_str());
    std::printf("purpose : %s\n", PurposeReflection::mission());
    std::printf("capabilities (honest map):\n");
    for (size_t i = 0; i < SelfModel::capability_count(); ++i)
        std::printf("  %-36s %s\n", SelfModel::capabilities()[i].area,
                    SelfModel::capabilities()[i].state);

    // --- WHAT AM I DOING --------------------------------------------------
    GoalTracker goals;
    seed_default_goals(goals);
    goals.load("state/goals.bin");            // refresh from disk if present
    std::printf("\n=== GOALS (active, priority-sorted) ===\n");
    for (const Goal& g : goals.active_goals())
        std::printf("  [%s] %s — metric %s: current %.2f vs target %.2f\n",
                    g.priority >= 0.9 ? "HIGH" : g.priority >= 0.7 ? "MED" : "LOW",
                    g.text.c_str(), g.metric.c_str(), g.current, g.target);

    ActionRationale log;
    log.load("state/rationale.bin");
    std::printf("\n=== RECENT DECISIONS (I chose X because Y) ===\n");
    if (log.entries().empty()) {
        std::printf("  (no decisions logged yet — run trading-watch)\n");
    }
    size_t shown = 0;
    for (const RationaleEntry& e : log.entries()) {
        if (shown++ >= 10) break;
        std::printf("  [%s] %s\n        because: %s (confidence %.2f%s)\n",
                    e.actor.c_str(), e.action.c_str(), e.because.c_str(),
                    e.confidence,
                    e.outcome_known ? ", outcome known" : "");
    }

    // --- WHY ---------------------------------------------------------------
    std::printf("\n=== PURPOSE REFLECTION ===\n");
    const auto refl = PurposeReflection::reflect(log, goals);
    std::printf("Q: %s\nA: %s\n", refl.question.c_str(), refl.answer.c_str());
    std::printf("peak RSS: %.2f MB\n",
                static_cast<double>(platform::peak_rss_bytes()) / 1048576.0);
    return 0;
}

// ===========================================================================
// metacognition
// ===========================================================================
int cmd_metacognition(const Args& a) {
    // Ground the numbers in reality when possible: a quick backtest gives
    // the realized Sharpe; the rationale log gives calibration.
    double sharpe = 0.0;
    size_t observations = 0;
    std::vector<Bar> bars;
    std::string err;
    const std::string csv = a.csv.empty() ? default_csv(a.ticker, a.timeframe)
                                          : a.csv;
    if (load_bars_csv(csv, bars, err)) {
        Backtester::Config cfg;
        cfg.ticker = a.ticker;
        const BacktestReport rep = Backtester::run(bars, cfg);
        if (rep.error.empty()) {
            sharpe = rep.sharpe;
            observations = rep.trades.size();
        }
    }

    GoalTracker goals;
    seed_default_goals(goals);
    ActionRationale log;
    log.load("state/rationale.bin");

    const Metacognition::State st = Metacognition::evaluate(goals, log,
                                                            sharpe, observations);
    std::printf("=== METACOGNITION ===\n");
    std::printf("strategy : %s\n", st.strategy_assessment.c_str());
    std::printf("confidence in strategy: %.2f (shrinks with thin evidence)\n",
                st.confidence_in_strategy);
    if (st.calibrated_decisions > 0) {
        std::printf("calibration: %zu resolved decisions, gap %+.2f %s\n",
                    st.calibrated_decisions, st.calibration_gap,
                    st.overconfident ? "— OVERCONFIDENT, reduce sizing" : "");
    } else {
        std::printf("calibration: none yet — state confidence only after "
                    "resolved outcomes\n");
    }
    std::printf("\nknowledge gaps (what I do not know):\n");
    for (const KnowledgeGap& g : st.gaps)
        std::printf("  - %s\n      why: %s\n      do : %s\n",
                    g.topic.c_str(), g.why_it_matters.c_str(),
                    g.suggested_action.c_str());
    std::printf("%s", kDisclaimer);
    return 0;
}

// ===========================================================================
// cloud (Phase-3 fills the HTTP client; status-only here by design)
// ===========================================================================
int cmd_cloud() {
    const bool has_key = std::getenv("OMNISEED_CLOUD_KEY") != nullptr;
    std::printf("cloud bridge: %s\n",
                has_key ? "key present (OMNISEED_CLOUD_KEY)"
                        : "not configured — local AgentLoop only (fallback)");
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    platform::enable_utf8_console();
    if (argc < 2) {
        print_usage();
        return 0;
    }
    const std::string cmd = argv[1];
    const Args a = parse_args(argc, argv, 2);

    if (cmd == "trading-analyze")  return cmd_trading_analyze(a);
    if (cmd == "trading-simulate") return cmd_trading_simulate(a);
    if (cmd == "trading-watch")    return cmd_trading_watch(a);
    if (cmd == "introspect")       return cmd_introspect();
    if (cmd == "metacognition")    return cmd_metacognition(a);
    if (cmd == "cloud")            return cmd_cloud();

    print_usage();
    return 1;
}
