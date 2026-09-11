// =============================================================================
//  OmniSeed — trading/news_feed.h
//  News aggregation + sentiment (Phase-16 "Omega Pass"):
//
//    * NewsFeed        — RSS/Atom 2.0 item parsing, ingestion + dedup,
//                        lexicon-based financial sentiment, ticker entity
//                        extraction, breaking-news detection
//    * NewsItem        — one headline + metadata + computed sentiment
//
//  Designed for the NewsMonitorAgent (agent/sub_agents.h): the agent polls
//  `pending_alerts()` and raises breaking stories into the Sensory Interrupt
//  System. Network fetching itself is delegated to tools/fetch_news.py — the
//  C++ side ingests text (RSS/XML) it is given, keeping the runtime
//  zero-dependency (no TLS stack).
// =============================================================================
#pragma once

#include <cstdint>
#include <deque>
#include <set>
#include <string>
#include <vector>

namespace omniseed {
namespace trading {

// ===========================================================================
// NewsItem
// ===========================================================================
struct NewsItem {
    std::string guid;                  // dedup key (falls back to title)
    std::string title;
    std::string summary;
    std::string source;                // "reuters", "cnbc", ...
    int64_t  published = 0;            // unix seconds
    double   sentiment = 0.0;          // [-1, 1]; lexicon score, clamped
    std::vector<std::string> tickers;  // extracted $TICKER / bare symbols
};

// ===========================================================================
// NewsFeed — parse + score + alert
// ===========================================================================
class NewsFeed {
public:
    struct Config {
        size_t  max_items        = 512;
        double  breaking_window  = 900.0;   // seconds considered "fresh"
        int32_t breaking_priority = 9;       // SensoryInterruptSystem level
        double  strong_sentiment  = 0.65;    // |score| that marks an item
    };

    explicit NewsFeed(const Config& cfg = {}) : cfg_(cfg) {}

    // Parses RSS 2.0 (and tolerates Atom <entry>) XML text; returns the
    // number of NEW items added (GUID-based dedup). Sentiment + tickers are
    // computed on ingest.
    size_t ingest_rss(const std::string& xml);

    // Recomputes sentiment/tickers/breaking for a mutated item (tests).
    void recompute(NewsItem& item) const;

    // Lexicon sentiment: +1 per positive hit, -1 per negative hit, normalized
    // by length (clamped to [-1, 1]). Case-insensitive.
    static double score_sentiment(const std::string& text);

    // Extracts ticker symbols: $AAPL style, or ALL-CAPS 1-5 letter words.
    static std::vector<std::string> extract_tickers(const std::string& text);

    // Breaking = fresh (within breaking_window) AND (crash/plunge/surge/
    // fraud/panic lexicon OR strong sentiment).
    bool is_breaking(const NewsItem& item) const;

    // Drains breaking items as interrupt descriptors (priority + detail).
    struct Alert {
        int32_t priority = 0;
        std::string detail;
    };
    std::vector<Alert> pending_alerts();
    void reset_alerts() { alerted_.clear(); }

    const std::deque<NewsItem>& items() const { return items_; }
    std::deque<NewsItem>& items() { return items_; }
    size_t size() const { return items_.size(); }
    const Config& config() const { return cfg_; }

    // Newest-first blended sentiment over the last `max_age_seconds`.
    double aggregate_sentiment(double max_age_seconds,
                               const std::string& ticker = "") const;

private:
    Config cfg_;
    std::deque<NewsItem> items_;
    std::set<std::string> seen_;          // guids
    std::set<std::string> alerted_;       // guids already raised as alerts
};

// ===========================================================================
// Sentiment lexicon (shared; exposed for tests/extension)
// ===========================================================================
extern const char* const kPositiveTerms[];
extern const size_t      kPositiveCount;
extern const char* const kNegativeTerms[];
extern const size_t      kNegativeCount;
extern const char* const kBreakingTerms[];
extern const size_t      kBreakingCount;

} // namespace trading
} // namespace omniseed
