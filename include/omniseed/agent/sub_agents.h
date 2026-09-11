// =============================================================================
//  OmniSeed — agent/sub_agents.h
//  Multi-Agent Specialization for Trading Mode (Phase-16 "Omega Pass"):
//
//    * MarketAnalystAgent  — technical (indicators) + news sentiment blended
//                            into buy/sell/hold verdicts; consensus voting
//    * NewsMonitorAgent    — polls a NewsFeed, raises breaking stories into
//                            the Sensory Interrupt System
//    * RiskManagerAgent    — monitors portfolio exposure/concentration/
//                            drawdown, emits alerts, can veto entries
//    * ExecutionAgent      — routes consensus verdicts into a PaperBroker
//                            (real brokers are opt-in, see TASK I notes)
//    * ResearchAgent       — compiles ticker briefs (trend/vol/news)
//
//  Coordination model (deterministic, testable): each agent is a pure-ish
//  object over shared domain structs; TradingWatchdog (below) runs the loop:
//  feed -> analysts -> consensus -> risk veto -> execution. No threads, no
//  hidden state: the swarm is inspectable and reproducible.
// =============================================================================
#pragma once

#include "omniseed/agent/agent_intel.h"             // SensoryInterruptSystem
#include "omniseed/trading/trading_engine.h"
#include "omniseed/trading/news_feed.h"

#include <string>
#include <vector>

namespace omniseed {
namespace trading {

// Forward decl (trading/simulate.h): minimal paper-trading interface so the
// ExecutionAgent can route orders without depending on the whole simulator.
class PaperBroker;

// ===========================================================================
// Shared verdict type — every agent speaks this
// ===========================================================================
struct AgentVerdict {
    std::string agent;            // "technical", "news", "risk", ...
    Action    action = Action::Hold;
    double    strength = 0.0;     // 0..1 conviction
    double    sentiment = 0.0;    // optional news sentiment component
    std::string reason;
};

// ===========================================================================
// MarketAnalystAgent — technical + news analysis
// ===========================================================================
class MarketAnalystAgent {
public:
    struct Config {
        SignalGenerator::Config signals;
        double news_weight     = 0.35;   // news sentiment vs technical weight
        double news_sentiment_threshold = 0.25;  // |s| that flips an action
        double strong_signal   = 0.6;    // strength treated as "high conviction"
    };

    explicit MarketAnalystAgent(const Config& cfg = {}) : cfg_(cfg) {}

    // Pure technical verdict from indicator state at index i.
    AgentVerdict analyze_technical(const std::string& ticker,
                                   const std::vector<Bar>& bars,
                                   const Indicators& ind, size_t i) const;

    // News-driven verdict for a ticker from the feed's recent items.
    AgentVerdict analyze_news(const std::string& ticker,
                              const NewsFeed& feed) const;

    // Blends the two (weights from config); returns the analyst's verdict.
    AgentVerdict blend(const AgentVerdict& tech,
                       const AgentVerdict& news) const;

    // Consensus mechanism: weighted vote across agent verdicts.
    //   score = sum(strength * sign(action)); > 0 -> Buy, < 0 -> Sell.
    // Tie or all-Hold -> Hold. Ties are broken toward Hold (capital
    // preservation beats action in a disagreement).
    static AgentVerdict consensus(const std::vector<AgentVerdict>& votes);

    const Config& config() const { return cfg_; }

private:
    Config cfg_;
};

// ===========================================================================
// NewsMonitorAgent — feed watcher + interrupt raiser
// ===========================================================================
class NewsMonitorAgent {
public:
    struct Stats {
        size_t items_seen = 0;
        size_t alerts_raised = 0;
    };

    explicit NewsMonitorAgent(SensoryInterruptSystem& interrupts)
        : interrupts_(interrupts) {}

    // Ingests new RSS text into the feed, then raises breaking stories as
    // high-priority interrupts. Returns alerts raised this call.
    size_t poll(NewsFeed& feed, const std::string& rss_xml);

    const Stats& stats() const { return stats_; }

private:
    SensoryInterruptSystem& interrupts_;
    Stats stats_{};
};

// ===========================================================================
// RiskManagerAgent — portfolio watchdog with veto power
// ===========================================================================
class RiskManagerAgent {
public:
    struct Config {
        RiskLimits limits{};
        double concentration_warn_pct = 0.30;   // single-name share of equity
        double volatility_warn_atr_pct = 0.05;  // ATR/close sanity ceiling
    };

    explicit RiskManagerAgent(const Config& cfg = {}) : cfg_(cfg) {}

    // Returns alert lines (empty = healthy). Checks: concentration,
    // gross exposure, drawdown halt, per-position stop breaches.
    std::vector<std::string> check_portfolio(const PortfolioState& ps,
                                             const std::map<std::string, double>& prices) const;

    // Veto decision for a proposed entry (used by the watchdog before
    // execution): returns false + reason when risk says no.
    bool approve_entry(const PortfolioState& ps, double equity_peak,
                       const std::string& ticker, double price,
                       std::string& reason) const;

    const Config& config() const { return cfg_; }

private:
    Config cfg_;
};

// ===========================================================================
// ExecutionAgent — order routing (paper first; broker adapters opt-in)
// ===========================================================================
class ExecutionAgent {
public:
    enum class Mode { PaperOnly };          // real-broker modes arrive with
                                            // a user-configured adapter

    explicit ExecutionAgent(PaperBroker* paper) : paper_(paper) {}

    // Act on a verdict at `price`: Buy -> market buy (sized by `qty_hint`,
    // 0 = paper broker default 5% of cash), Sell -> flatten the position.
    // Returns the fill description; "" when nothing was done.
    std::string on_signal(const std::string& ticker, Action action,
                          double strength, double price, int64_t ts,
                          double qty_hint = 0.0);

    // Current portfolio state (via the paper broker) for risk checks.
    PortfolioState portfolio_snapshot() const;

    const Mode mode() const { return mode_; }

private:
    Mode mode_ = Mode::PaperOnly;
    PaperBroker* paper_;                    // not owned
};

// ===========================================================================
// ResearchAgent — compiles ticker briefs from market + news data
// ===========================================================================
class ResearchAgent {
public:
    // Deterministic textual brief: trend, range, volatility, news sentiment,
    // recent headline count. No network, no LLM call (cloud bridge can wrap
    // this for deeper research later).
    std::string brief(const std::string& ticker, const std::vector<Bar>& bars,
                      const NewsFeed& feed) const;
};

// ===========================================================================
// TradingWatchdog — the multi-agent coordination loop (--trading-mode)
// ===========================================================================
class TradingWatchdog {
public:
    struct Config {
        std::vector<std::string> tickers;
        MarketAnalystAgent::Config analyst;
        RiskManagerAgent::Config risk;
        int32_t interval_seconds = 60;    // poll cadence
        bool    paper_trading    = true;  // execution routing on consensus
    };

    struct CycleReport {
        int64_t at_ts = 0;
        std::string ticker;
        AgentVerdict verdict;
        bool risk_veto = false;
        std::string execution_note;
        std::vector<std::string> risk_alerts;
    };

    // One full coordination cycle for `ticker`: technical + news -> consensus
    // -> risk veto -> optional paper execution. Deterministic.
    static CycleReport run_cycle(const Config& cfg,
                                 MarketAnalystAgent& analyst,
                                 RiskManagerAgent& risk,
                                 ExecutionAgent* exec,     // nullable
                                 NewsFeed& feed,
                                 const std::vector<Bar>& bars,
                                 size_t bar_index,
                                 double equity_peak);
};

} // namespace trading
} // namespace omniseed
