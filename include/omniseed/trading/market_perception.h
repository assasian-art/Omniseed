// =============================================================================
//  OmniSeed — trading/market_perception.h
//  M1 — Market Perception: a unified multi-asset schema, feed health, and the
//  code-enforced ABSTAIN gate.
//
//  Perception is where bad data becomes bad trades, so this module is built
//  around refusing to act:
//
//    * Ticker            — one normalized snapshot across asset classes,
//                          carrying its provenance (provider), mirroring the
//                          T6 CSV `source` contract.
//    * FeedHealth        — per-symbol liveness: last update, staleness,
//                          consecutive failures, derived Ok/Degraded/Dead.
//    * FeedHealthMonitor — ingests observations/failures, derives health.
//    * PerceptionGate    — the ABSTAIN decision: stale, dead, missing or
//                          invalid input REFUSES to produce a trade.
//
//  Honest scope: this says nothing about whether a trade is *good*. It only
//  refuses to act on data we cannot trust.
// =============================================================================
#pragma once

#include "omniseed/trading/trading_engine.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace omniseed {
namespace trading {

// ===========================================================================
// Ticker — one normalized snapshot across asset classes
// ===========================================================================
struct Ticker {
    std::string symbol;                // canonical: AAPL / BTCUSDT / EURUSD
    std::string provider;              // provenance: yahoo|binance|... (T6 style)
    AssetClass  asset = AssetClass::Unknown;
    double      last = 0.0;            // last price
    double      change_pct_24h = 0.0;
    double      volume_24h = 0.0;
    double      bid = 0.0, ask = 0.0;  // 0 when the provider does not publish
    int64_t     ts = 0;                // provider timestamp (unix seconds, UTC)

    bool valid() const { return last > 0.0 && ts > 0 && !symbol.empty(); }
};

// ===========================================================================
// Feed health
// ===========================================================================
enum class FeedState : int8_t { Ok = 0, Degraded, Dead, Missing };

inline const char* to_string(FeedState s) {
    switch (s) {
        case FeedState::Ok:       return "ok";
        case FeedState::Degraded: return "degraded";
        case FeedState::Dead:     return "dead";
        default:                  return "missing";
    }
}

struct FeedPolicy {
    int64_t degraded_after_secs = 300;   // 5 min without a fresh update
    int64_t dead_after_secs     = 900;   // 15 min without a fresh update
    int32_t dead_after_failures = 3;     // consecutive provider errors
};

struct FeedHealth {
    std::string symbol;
    std::string provider;
    int64_t last_update_ts  = 0;         // newest provider timestamp seen
    int64_t last_attempt_ts = 0;         // newest observation attempt
    int32_t consecutive_failures = 0;
    int64_t total_observations = 0;
    int64_t total_failures = 0;
    FeedState state = FeedState::Missing;

    // Seconds since the newest data; -1 when nothing has ever been seen.
    int64_t staleness(int64_t now_ts) const {
        if (last_update_ts <= 0) return -1;
        return now_ts > last_update_ts ? now_ts - last_update_ts : 0;
    }
    double error_rate() const {
        return total_observations > 0
            ? static_cast<double>(total_failures) /
              static_cast<double>(total_observations)
            : 0.0;
    }
};

class FeedHealthMonitor {
public:
    FeedHealthMonitor() : FeedHealthMonitor(FeedPolicy{}) {}
    explicit FeedHealthMonitor(const FeedPolicy& policy) : policy_(policy) {}

    // A successful observation of `symbol` at provider time `provider_ts`,
    // seen at `now_ts`. Clears the consecutive-failure counter.
    void observe(const std::string& symbol, const std::string& provider,
                 int64_t provider_ts, int64_t now_ts);
    // A provider/transport error for `symbol`.
    void record_failure(const std::string& symbol, int64_t now_ts);

    // Health with `state` derived for `now_ts`. Unknown symbols come back
    // FeedState::Missing (never an implicit "ok").
    FeedHealth health(const std::string& symbol, int64_t now_ts) const;
    std::vector<FeedHealth> all(int64_t now_ts) const;

    const FeedPolicy& policy() const { return policy_; }
    void clear() { feeds_.clear(); }

private:
    FeedState classify(const FeedHealth& h, int64_t now_ts) const;

    FeedPolicy policy_;
    std::map<std::string, FeedHealth> feeds_;
};

// ===========================================================================
// PerceptionGate — the ABSTAIN decision (code-enforced)
// ===========================================================================
enum class AbstainReason : int8_t {
    None = 0, MissingData, InvalidTicker, StaleFeed, DeadFeed, LowLiquidity
};

inline const char* to_string(AbstainReason r) {
    switch (r) {
        case AbstainReason::None:          return "none";
        case AbstainReason::MissingData:   return "missing-data";
        case AbstainReason::InvalidTicker: return "invalid-ticker";
        case AbstainReason::StaleFeed:     return "stale-feed";
        case AbstainReason::DeadFeed:      return "dead-feed";
        default:                           return "low-liquidity";
    }
}

struct AbstainDecision {
    bool          abstain = false;
    AbstainReason reason  = AbstainReason::None;
    std::string   detail;
};

class PerceptionGate {
public:
    struct Config {
        FeedPolicy feed;
        double     min_volume_24h = 0.0;   // 0 = disabled
        bool       require_valid  = true;  // reject invalid Tickers outright
    };

    PerceptionGate() : PerceptionGate(Config{}) {}
    explicit PerceptionGate(const Config& cfg) : cfg_(cfg) {}

    // Decide whether the perception layer may act on this ticker at all.
    // `h` should come from FeedHealthMonitor::health(symbol, now_ts). A
    // default-constructed FeedHealth (no symbol) means the caller keeps no
    // monitor, so the ticker's own ts is the fallback; a health that names the
    // symbol but has no data means the feed is missing -> MissingData.
    AbstainDecision check(const Ticker& t, const FeedHealth& h,
                          int64_t now_ts) const;

    const Config& config() const { return cfg_; }

private:
    Config cfg_;
};

} // namespace trading
} // namespace omniseed
