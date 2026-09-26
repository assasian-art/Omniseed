// =============================================================================
//  OmniSeed — trading/market_perception.cpp
//  Feed health + the ABSTAIN gate. Deterministic, no I/O.
// =============================================================================
#include "omniseed/trading/market_perception.h"

#include <cstdio>

namespace omniseed {
namespace trading {

// ===========================================================================
// FeedHealthMonitor
// ===========================================================================
FeedState FeedHealthMonitor::classify(const FeedHealth& h, int64_t now_ts) const {
    if (h.last_update_ts <= 0) return FeedState::Missing;
    if (h.consecutive_failures >= policy_.dead_after_failures)
        return FeedState::Dead;
    const int64_t stale = h.staleness(now_ts);
    if (stale >= policy_.dead_after_secs)     return FeedState::Dead;
    if (stale >= policy_.degraded_after_secs) return FeedState::Degraded;
    return FeedState::Ok;
}

void FeedHealthMonitor::observe(const std::string& symbol,
                                const std::string& provider,
                                int64_t provider_ts, int64_t now_ts) {
    FeedHealth& h = feeds_[symbol];
    h.symbol = symbol;
    if (!provider.empty()) h.provider = provider;
    if (provider_ts > h.last_update_ts) h.last_update_ts = provider_ts;
    if (now_ts > h.last_attempt_ts) h.last_attempt_ts = now_ts;
    h.consecutive_failures = 0;
    ++h.total_observations;
    h.state = classify(h, now_ts);
}

void FeedHealthMonitor::record_failure(const std::string& symbol, int64_t now_ts) {
    FeedHealth& h = feeds_[symbol];
    h.symbol = symbol;
    ++h.consecutive_failures;
    ++h.total_failures;
    ++h.total_observations;
    if (now_ts > h.last_attempt_ts) h.last_attempt_ts = now_ts;
    h.state = classify(h, now_ts);
}

FeedHealth FeedHealthMonitor::health(const std::string& symbol,
                                     int64_t now_ts) const {
    auto it = feeds_.find(symbol);
    if (it == feeds_.end()) {
        FeedHealth h;                       // never observed -> Missing
        h.symbol = symbol;
        h.state = FeedState::Missing;
        return h;
    }
    FeedHealth h = it->second;
    h.state = classify(h, now_ts);
    return h;
}

std::vector<FeedHealth> FeedHealthMonitor::all(int64_t now_ts) const {
    std::vector<FeedHealth> out;
    out.reserve(feeds_.size());
    for (const auto& kv : feeds_) {
        FeedHealth h = kv.second;
        h.state = classify(h, now_ts);
        out.push_back(h);
    }
    return out;
}

// ===========================================================================
// PerceptionGate
// ===========================================================================
AbstainDecision PerceptionGate::check(const Ticker& t, const FeedHealth& h,
                                      int64_t now_ts) const {
    AbstainDecision d;

    if (cfg_.require_valid && !t.valid()) {
        d.abstain = true;
        d.reason  = AbstainReason::InvalidTicker;
        d.detail  = "ticker has no price or timestamp";
        return d;
    }

    // Effective last-update time. A default-constructed FeedHealth means the
    // caller keeps no monitor, so the ticker's own timestamp is the fallback.
    // A health that names the symbol but carries no data means the monitor has
    // never seen it -> MissingData (never an implicit "fresh").
    FeedHealth eff = h;
    if (eff.last_update_ts <= 0) {
        if (eff.symbol.empty() && t.ts > 0) {
            eff.last_update_ts = t.ts;
            eff.symbol = t.symbol;
            eff.consecutive_failures = 0;
            eff.state = FeedState::Ok;
        } else {
            d.abstain = true;
            d.reason  = AbstainReason::MissingData;
            d.detail  = "no observation for " + t.symbol;
            return d;
        }
    }

    if (eff.state == FeedState::Dead ||
        eff.consecutive_failures >= cfg_.feed.dead_after_failures) {
        char buf[160];
        std::snprintf(buf, sizeof(buf),
                      "feed dead (staleness %llds, %d consecutive failures)",
                      static_cast<long long>(eff.staleness(now_ts)),
                      eff.consecutive_failures);
        d.abstain = true;
        d.reason  = AbstainReason::DeadFeed;
        d.detail  = buf;
        return d;
    }

    const int64_t stale = eff.staleness(now_ts);
    if (stale >= cfg_.feed.dead_after_secs) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "feed stale %llds >= dead %llds",
                      static_cast<long long>(stale),
                      static_cast<long long>(cfg_.feed.dead_after_secs));
        d.abstain = true;
        d.reason  = AbstainReason::DeadFeed;
        d.detail  = buf;
        return d;
    }
    if (stale >= cfg_.feed.degraded_after_secs || eff.state == FeedState::Degraded) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "feed stale %llds >= degraded %llds",
                      static_cast<long long>(stale),
                      static_cast<long long>(cfg_.feed.degraded_after_secs));
        d.abstain = true;
        d.reason  = AbstainReason::StaleFeed;
        d.detail  = buf;
        return d;
    }

    if (cfg_.min_volume_24h > 0.0 && t.volume_24h < cfg_.min_volume_24h) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "24h volume %.0f < floor %.0f",
                      t.volume_24h, cfg_.min_volume_24h);
        d.abstain = true;
        d.reason  = AbstainReason::LowLiquidity;
        d.detail  = buf;
        return d;
    }

    d.abstain = false;
    d.reason  = AbstainReason::None;
    d.detail  = "ok";
    return d;
}

} // namespace trading
} // namespace omniseed
