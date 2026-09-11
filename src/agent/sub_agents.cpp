// =============================================================================
//  OmniSeed — agent/sub_agents.cpp
//  Implementations for the multi-agent trading specialization. Everything is
//  deterministic and side-effect-free except the explicit PaperBroker routing
//  in ExecutionAgent and the interrupt raises in NewsMonitorAgent.
// =============================================================================
#include "omniseed/agent/sub_agents.h"
#include "omniseed/trading/simulate.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace omniseed {
namespace trading {

// ===========================================================================
// MarketAnalystAgent
// ===========================================================================
AgentVerdict MarketAnalystAgent::analyze_technical(const std::string& ticker,
                                                   const std::vector<Bar>& bars,
                                                   const Indicators& ind,
                                                   size_t i) const {
    const Signal sig = SignalGenerator::evaluate(ind, bars, i, cfg_.signals);
    AgentVerdict v;
    v.agent = "technical:" + ticker;
    v.action = sig.action;
    v.strength = sig.strength;
    v.reason = sig.reason;
    return v;
}

AgentVerdict MarketAnalystAgent::analyze_news(const std::string& ticker,
                                              const NewsFeed& feed) const {
    AgentVerdict v;
    v.agent = "news:" + ticker;
    v.sentiment = feed.aggregate_sentiment(24.0 * 3600.0, ticker);
    if (std::fabs(v.sentiment) < cfg_.news_sentiment_threshold) {
        v.action = Action::Hold;
        v.strength = 0.0;
        v.reason = "news neutral (sentiment " +
                   std::to_string(v.sentiment).substr(0, 5) + ")";
        return v;
    }
    v.action = v.sentiment > 0.0 ? Action::Buy : Action::Sell;
    v.strength = std::min(1.0, std::fabs(v.sentiment));
    v.reason = std::string(v.sentiment > 0.0 ? "positive" : "negative") +
               " news sentiment " + std::to_string(v.sentiment).substr(0, 5);
    return v;
}

AgentVerdict MarketAnalystAgent::blend(const AgentVerdict& tech,
                                       const AgentVerdict& news) const {
    AgentVerdict v;
    v.agent = "analyst";
    // Weighted score: technical votes at full weight, news at news_weight.
    const double tech_score =
        (tech.action == Action::Buy ? 1.0 :
         tech.action == Action::Sell ? -1.0 : 0.0) * tech.strength;
    const double news_score =
        (news.action == Action::Buy ? 1.0 :
         news.action == Action::Sell ? -1.0 : 0.0) * news.strength * cfg_.news_weight;
    const double score = tech_score + news_score;

    if (score > 0.15)      v.action = Action::Buy;
    else if (score < -0.15) v.action = Action::Sell;
    else                    v.action = Action::Hold;
    v.strength = std::min(1.0, std::fabs(score));
    v.reason = "blend tech=" + std::to_string(tech_score).substr(0, 5) +
               " news=" + std::to_string(news_score).substr(0, 5);
    if (!tech.reason.empty()) v.reason += " [" + tech.reason + "]";
    if (!news.reason.empty()) v.reason += " [" + news.reason + "]";
    return v;
}

AgentVerdict MarketAnalystAgent::consensus(const std::vector<AgentVerdict>& votes) {
    AgentVerdict v;
    v.agent = "consensus";
    double score = 0.0;
    const char* sep = "";
    for (const AgentVerdict& vote : votes) {
        const double sgn = vote.action == Action::Buy ? 1.0 :
                           vote.action == Action::Sell ? -1.0 : 0.0;
        score += sgn * vote.strength;
        if (!vote.agent.empty()) {
            v.reason += sep + vote.agent + "=" +
                        std::to_string(sgn * vote.strength).substr(0, 5);
            sep = "; ";
        }
    }
    if (score > 0.15)      v.action = Action::Buy;
    else if (score < -0.15) v.action = Action::Sell;
    else                    v.action = Action::Hold;   // ties preserve capital
    v.strength = std::min(1.0, std::fabs(score));
    if (!v.reason.empty()) v.reason = "consensus(" + v.reason + ")";
    return v;
}

// ===========================================================================
// NewsMonitorAgent
// ===========================================================================
size_t NewsMonitorAgent::poll(NewsFeed& feed, const std::string& rss_xml) {
    if (!rss_xml.empty()) feed.ingest_rss(rss_xml);
    stats_.items_seen = feed.size();

    size_t raised = 0;
    for (const NewsFeed::Alert& a : feed.pending_alerts()) {
        interrupts_.raise(SensoryInterruptSystem::Event::SoundAlarm,
                          a.priority, a.detail);
        ++raised;
        ++stats_.alerts_raised;
    }
    return raised;
}

// ===========================================================================
// RiskManagerAgent
// ===========================================================================
std::vector<std::string> RiskManagerAgent::check_portfolio(
        const PortfolioState& ps,
        const std::map<std::string, double>& prices) const {
    std::vector<std::string> alerts;
    if (ps.equity <= 0.0) {
        alerts.push_back("portfolio equity <= 0");
        return alerts;
    }
    // Concentration.
    for (const auto& kv : ps.positions) {
        auto pit = prices.find(kv.first);
        const double px = pit != prices.end() ? pit->second : kv.second.avg_price;
        const double share = kv.second.qty * px / ps.equity;
        if (share > cfg_.concentration_warn_pct)
            alerts.push_back("concentration: " + kv.first + " at " +
                             std::to_string(share * 100.0).substr(0, 4) +
                             "% of equity (limit " +
                             std::to_string(cfg_.concentration_warn_pct * 100.0)
                                 .substr(0, 4) + "%)");
    }
    // Gross exposure.
    const double gross = ps.gross_exposure / ps.equity;
    if (gross > cfg_.limits.max_gross_exposure_pct)
        alerts.push_back("gross exposure " +
                         std::to_string(gross * 100.0).substr(0, 4) + "%");
    // Stop breaches.
    for (const auto& kv : ps.positions) {
        auto pit = prices.find(kv.first);
        if (pit == prices.end()) continue;
        const Action ex = RiskManager::check_exit(kv.second, pit->second,
                                                  cfg_.limits);
        if (ex == Action::Sell)
            alerts.push_back("stop breached: " + kv.first + " at " +
                             std::to_string(pit->second).substr(0, 7));
    }
    return alerts;
}

bool RiskManagerAgent::approve_entry(const PortfolioState& ps,
                                     double equity_peak,
                                     const std::string&, double,
                                     std::string& reason) const {
    return RiskManager::entries_allowed(ps, equity_peak, cfg_.limits, reason);
}

// ===========================================================================
// ExecutionAgent
// ===========================================================================
std::string ExecutionAgent::on_signal(const std::string& ticker, Action action,
                                      double strength, double price,
                                      int64_t ts, double qty_hint) {
    if (!paper_ || strength <= 0.0) return "";
    std::string err;
    if (action == Action::Buy) {
        double qty = qty_hint;
        if (qty <= 0.0) {
            // Default conviction-scaled order: 5% of cash at full strength.
            qty = std::floor(paper_->state().cash * 0.05 * strength / price);
        }
        if (qty <= 0.0) return "";
        if (!paper_->market_buy(ticker, qty, price, ts, err)) return "";
        return "bought " + std::to_string(qty).substr(0, 6) + " " + ticker +
               " @ " + std::to_string(price).substr(0, 8);
    }
    if (action == Action::Sell) {
        auto it = paper_->state().positions.find(ticker);
        if (it == paper_->state().positions.end()) return "";
        const double qty = it->second.qty;
        if (!paper_->market_sell(ticker, qty, price, ts, err)) return "";
        return "sold " + std::to_string(qty).substr(0, 6) + " " + ticker +
               " @ " + std::to_string(price).substr(0, 8);
    }
    return "";
}

PortfolioState ExecutionAgent::portfolio_snapshot() const {
    return paper_ ? paper_->state() : PortfolioState{};
}

// ===========================================================================
// ResearchAgent
// ===========================================================================
std::string ResearchAgent::brief(const std::string& ticker,
                                 const std::vector<Bar>& bars,
                                 const NewsFeed& feed) const {
    char buf[512];
    if (bars.size() < 30) {
        std::snprintf(buf, sizeof(buf),
                      "[research] %s: insufficient history (%zu bars)\n",
                      ticker.c_str(), bars.size());
        return buf;
    }
    const Indicators ind = SignalGenerator::compute(bars, {});
    const size_t i = bars.size() - 1;
    const double trend = ind.sma20[i] > ind.sma50[i] ? 1.0 : -1.0;
    const double range_hi = ind.bb_upper[i], range_lo = ind.bb_lower[i];
    const double atr_pct = bars[i].close > 0 ? ind.atr14[i] / bars[i].close : 0.0;
    const double news = feed.aggregate_sentiment(24.0 * 3600.0, ticker);

    std::snprintf(buf, sizeof(buf),
                  "[research] %s brief\n"
                  "  trend      : %s (sma20 %.2f vs sma50 %.2f)\n"
                  "  close      : %.2f (bb range %.2f .. %.2f)\n"
                  "  volatility : ATR14 %.2f%% of close\n"
                  "  rsi14      : %.1f\n"
                  "  news 24h   : sentiment %.2f over %zu items\n",
                  ticker.c_str(),
                  trend > 0 ? "up" : "down",
                  ind.sma20[i], ind.sma50[i],
                  bars[i].close, range_lo, range_hi,
                  atr_pct * 100.0,
                  ind.rsi14[i],
                  news, feed.size());
    return buf;
}

// ===========================================================================
// TradingWatchdog — one coordination cycle
// ===========================================================================
TradingWatchdog::CycleReport TradingWatchdog::run_cycle(
        const Config& cfg, MarketAnalystAgent& analyst, RiskManagerAgent& risk,
        ExecutionAgent* exec, NewsFeed& feed, const std::vector<Bar>& bars,
        size_t bar_index, double equity_peak) {
    CycleReport rep;
    if (bars.empty() || cfg.tickers.empty()) return rep;
    rep.ticker = cfg.tickers.front();
    rep.at_ts = bars[std::min(bar_index, bars.size() - 1)].time;

    // PositionManager view of the CURRENT cycle (paper broker state is owned
    // by the caller; the watchdog only reads risk-relevant aggregates).
    const Indicators ind = SignalGenerator::compute(bars, cfg.analyst.signals);
    const AgentVerdict tech =
        analyst.analyze_technical(rep.ticker, bars, ind, bar_index);
    const AgentVerdict news = analyst.analyze_news(rep.ticker, feed);
    rep.verdict = MarketAnalystAgent::consensus({tech, news});

    if (exec && cfg.paper_trading && rep.verdict.action != Action::Hold) {
        // Risk veto with a live portfolio snapshot from the broker.
        const PortfolioState ps = exec->portfolio_snapshot();
        std::string why;
        if (!risk.approve_entry(ps, equity_peak, rep.ticker,
                                bars[bar_index].close, why)) {
            rep.risk_veto = true;
            rep.execution_note = "vetoed: " + why;
            return rep;
        }
        rep.execution_note = exec->on_signal(
            rep.ticker, rep.verdict.action, rep.verdict.strength,
            bars[bar_index].close, rep.at_ts);
    }
    return rep;
}

} // namespace trading
} // namespace omniseed
