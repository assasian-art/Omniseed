// =============================================================================
//  OmniSeed — prefix_cache.h
//  RWKV-7 WKV prefix snapshots (grok registry C01/C02, deepseek "思维链缓存"
//  cousin): checkpoint the O(1) recurrent state at prompt/skill/tool-result
//  boundaries and restore on prefix hit.
//
//  Why this is the single biggest speed win available to OmniSeed: RWKV-7's
//  state is CONSTANT-size ([L, H, 64, 64] fp32 ≈ 0.59 MB at 0.1B), so a
//  snapshot is a tiny fixed blob — and the system prompt + tool schemas are
//  ~90% of every turn's prefill. Snapshot once, restore forever: turn 2+
//  prefill cost drops from O(system prompt) to O(new tokens only).
//
//  Speculative drafting (C04/C05) is deliberately NOT implemented here: RWKV
//  is strictly recurrent, so verifying m draft tokens costs m sequential
//  forwards — identical to plain greedy. Draft models only win where
//  verification is parallel (transformers). Documented refusal, not an
//  oversight (see PROJECT_STATE.md).
//
//  RAM: 2-8 snapshots x 0.59 MB ≈ 1-5 MB. Storage: compact binary sidecar.
// =============================================================================
#pragma once

#include "omniseed/core/rwkv.h"

#include <cstdint>
#include <string>
#include <vector>

namespace omniseed {

// ---------------------------------------------------------------------------
// Snapshot store keyed by a stable prefix id (e.g. SelfImprovement::task_key
// of the system prompt, or an explicit "session" name).
// ---------------------------------------------------------------------------
class PrefixCache {
public:
    struct Config {
        size_t max_snapshots = 8;    // 8 x 0.59 MB ≈ 4.7 MB at 0.1B
    };

    explicit PrefixCache(const Config& cfg = {}) : cfg_(cfg) {}

    // Serialize the current state under `key`. Overwrites an existing entry
    // with the same key; evicts the oldest when full.
    bool store(const std::string& key, const RwkvModel& model,
               const RwkvState& state);

    // Restore into `state`. Returns false on miss (caller prefills from
    // scratch). `tokens_seen` is restored too, so streaming bookkeeping
    // stays consistent.
    bool load(const std::string& key, const RwkvModel& model,
              RwkvState& state, int64_t& tokens_seen) const;

    bool has(const std::string& key) const;
    void clear() { entries_.clear(); }
    size_t size() const { return entries_.size(); }
    size_t bytes() const { return bytes_; }

    // Persistence (compact binary; same layout as the in-memory blob).
    bool save(const std::string& path) const;
    bool load(const std::string& path, const RwkvModel& model);

    // How many tokens the cache saved this process (diagnostics/bench).
    uint64_t tokens_saved() const { return tokens_saved_; }
    void add_tokens_saved(uint64_t n) { tokens_saved_ += n; }

private:
    struct Entry {
        std::string key;
        std::vector<uint8_t> blob;   // serialized RwkvState
        int64_t tokens_seen = 0;
        uint64_t last_use   = 0;     // platform::now_ms() for LRU eviction
    };

    static std::vector<uint8_t> serialize(const RwkvModel& model,
                                          const RwkvState& st,
                                          int64_t tokens_seen);
    static bool deserialize(const RwkvModel& model,
                            const std::vector<uint8_t>& blob,
                            RwkvState& st, int64_t& tokens_seen);

    Config cfg_;
    std::vector<Entry> entries_;   // LRU order: front = oldest
    size_t   bytes_ = 0;
    uint64_t tokens_saved_ = 0;
};

} // namespace omniseed
