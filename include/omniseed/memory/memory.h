// =============================================================================
//  OmniSeed — memory.h
//  Long-context memory stack for the 1M logical token window:
//
//   1. StreamingLLM layer — attention-sink stabilization + sliding window.
//      RWKV's O(1) state already removes the KV cache; sinks add robustness
//      for VERY long streams by re-anchoring the first tokens' contribution
//      and keeping a compact token ring for re-injection.
//
//   2. Memory Crystals — semantic compression of retired context into
//      compact "crystal" records (facts/events), retrieved by relevance
//      into new turns. This is what makes 1M LOGICAL tokens possible:
//      raw tokens flow through the state; retired ones persist as crystals.
// =============================================================================
#pragma once

#include "omniseed/core/tensor.h"
#include "omniseed/core/tokenizer.h"

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace omniseed {

// ---------------------------------------------------------------------------
// StreamingLLM: sink tokens + sliding window over a token stream
// ---------------------------------------------------------------------------
class StreamingLlm {
public:
    struct Config {
        int32_t n_sinks        = 4;     // initial tokens kept forever
        int32_t window_size    = 512;   // recent-token window
        int64_t max_logical    = 1000000; // 1M logical tokens (blueprint)
    };

    explicit StreamingLlm(const Config& cfg = {}) : cfg_(cfg) {}

    // Registers a token; returns true if it is inside the ACTIVE window
    // (i.e. should be fed to the model now).
    bool push(int32_t token);

    // Sink tokens (always active, re-fed on session resume).
    const std::vector<int32_t>& sinks() const { return sinks_; }

    // Recent-window tokens (in order).
    const std::vector<int32_t>& window() const { return window_; }

    // Tokens retired from the window (candidates for crystallization).
    std::vector<int32_t> drain_retired();

    int64_t total_seen() const { return total_seen_; }

    const Config& config() const { return cfg_; }

    // Snapshot/restore for session persistence.
    void reset();

private:
    Config cfg_;
    std::vector<int32_t> sinks_;
    std::vector<int32_t> window_;
    std::vector<int32_t> retired_;
    int64_t total_seen_ = 0;
};

// ---------------------------------------------------------------------------
// Memory Crystal: a compressed semantic record
// ---------------------------------------------------------------------------
struct MemoryCrystal {
    uint64_t id = 0;
    uint64_t created_at_token = 0;    // position in logical stream
    uint64_t last_access_token = 0;   // recency for Ebbinghaus decay
    float    importance = 0.0f;       // 0..1
    float    salience    = 0.0f;      // token-entropy signal at creation
    uint32_t hits        = 0;         // retrieval count
    float    embedding[64] = {0};     // compact semantic vector (64-dim)
    std::vector<int32_t> tokens;      // compressed content (short)
    std::string summary;              // human-readable gloss
};

// ---------------------------------------------------------------------------
// Memory Crystals: semantic compression + retrieval
// ---------------------------------------------------------------------------
class MemoryCrystals {
public:
    struct Config {
        size_t   max_crystals   = 512;    // memory ceiling (~2MB at 64-dim)
        int32_t  max_len_tokens = 48;     // per-crystal token budget
        int32_t  embed_dim      = 64;
        // Ebbinghaus decay (feature: deepseek VOL.II "记忆擦除/遗忘机制"):
        // effective_importance = importance * exp(-age_days / tau); crystals
        // below min_importance are dropped by decay(). 0 disables.
        double   decay_tau_days = 14.0;
        float    min_importance = 0.05f;
        // Entropy salience gate (grok M02): retire-with-high-entropy turns
        // are more informative than flat "ok" turns. 0 disables the boost.
        float    entropy_boost  = 0.15f;
    };

    explicit MemoryCrystals(const Config& cfg = {}) : cfg_(cfg) {}

    // Forms a crystal from retired tokens: keeps salient sentences by
    // scoring content words + positions; stores a 64-dim bag embedding.
    // `entropy` (optional) is the token-stream entropy salience signal.
    bool crystallize(const std::vector<int32_t>& retired_tokens,
                     const Tokenizer& tok, uint64_t stream_pos,
                     float entropy = -1.0f);

    // Returns the k most relevant crystals for the query tokens.
    // Updates recency/hits for returned crystals (feeds the decay model).
    std::vector<MemoryCrystal> retrieve(
        const std::vector<int32_t>& query_tokens, int32_t k);

    // Ebbinghaus decay: drops crystals whose effective importance
    // (importance x exp(-age/tau)) fell below min_importance. Returns the
    // number dropped. Call periodically (e.g. from dream()).
    size_t decay(double now_unix_seconds);

    // Persistence (compact binary sidecar next to the model).
    bool save(const std::string& path) const;
    bool load(const std::string& path);

    size_t size() const { return crystals_.size(); }
    const Config& config() const { return cfg_; }
    void clear() { crystals_.clear(); next_id_ = 1; }

private:
    void embed_tokens(const std::vector<int32_t>& ids, float* out) const;

    Config cfg_;
    std::vector<MemoryCrystal> crystals_;
    uint64_t next_id_ = 1;
};

// ---------------------------------------------------------------------------
// WorkingMemory — explicit short-term scratchpad (grok M "working memory",
// deepseek L1 of the hierarchical plan). Fixed-capacity token ring the agent
// can pin intermediate results into; never grows, so it cannot blow the
// budget. ~4 KB at default capacity.
// ---------------------------------------------------------------------------
class WorkingMemory {
public:
    explicit WorkingMemory(size_t capacity_tokens = 512) : cap_(capacity_tokens) {}

    void push(int32_t token) {
        if (buf_.size() >= cap_) buf_.erase(buf_.begin());
        buf_.push_back(token);
    }
    void push_tokens(const std::vector<int32_t>& ids) {
        for (int32_t id : ids) push(id);
    }
    const std::vector<int32_t>& tokens() const { return buf_; }
    void clear() { buf_.clear(); }
    size_t size() const { return buf_.size(); }
    size_t capacity() const { return cap_; }

private:
    size_t cap_;
    std::vector<int32_t> buf_;
};

} // namespace omniseed
