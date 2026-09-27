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

    StreamingLlm() : StreamingLlm(Config{}) {}
    explicit StreamingLlm(const Config& cfg) : cfg_(cfg) {}

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
    // Retrieval artifact, NOT stored state: the relevance retrieve() scored
    // this crystal at (cosine x importance). save()/load() deliberately skip
    // it — it is meaningless outside the query that produced it, and a caller
    // must not read it as a probability.
    float    score = 0.0f;
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
        // Crystal timestamps are STREAM POSITIONS (tokens), not wall clock, so
        // an age in days needs a rate. 100k tokens/day is the blueprint's
        // working figure; it was a literal inside decay() before.
        double   tokens_per_day = 100000.0;
        // Reinforcement: how much importance a USED crystal gains. Retrieval is
        // evidence of usefulness, so a retrieved crystal must decay slower than
        // an ignored one — that is the whole meaning of "fade unless
        // reinforced". Applied by reinforce(); see the note on decay().
        float    reinforce_boost = 0.15f;
    };

    MemoryCrystals() : MemoryCrystals(Config{}) {}
    explicit MemoryCrystals(const Config& cfg) : cfg_(cfg) {}

    // Forms a crystal from retired tokens: keeps salient sentences by
    // scoring content words + positions; stores a 64-dim bag embedding.
    // `entropy` (optional) is the token-stream entropy salience signal.
    // `out_id` (optional) receives the new crystal's id — the caller needs it
    // to attach its own bookkeeping (e.g. who spoke) to the record. Set to 0
    // when nothing was stored.
    bool crystallize(const std::vector<int32_t>& retired_tokens,
                     const Tokenizer& tok, uint64_t stream_pos,
                     float entropy = -1.0f, uint64_t* out_id = nullptr);

    // Returns the k most relevant crystals for the query tokens, scored by
    // cosine similarity x importance.
    //
    // `now_token` is the caller's current position in the SAME clock as
    // crystallize()'s stream_pos. When it is non-zero the returned crystals are
    // marked as accessed at that position, which RESETS their decay age — i.e.
    // retrieval reinforces. When it is zero the clock is unknown and only the
    // hit counter moves. (An earlier revision wrote
    // `created_at_token + query_tokens.size()` here unconditionally, which
    // pinned the "last access" to the creation position: retrieval made a
    // crystal look exactly as old as it already was, so recall could never
    // rescue a memory from decay. Zero now means "do not touch recency".)
    std::vector<MemoryCrystal> retrieve(
        const std::vector<int32_t>& query_tokens, int32_t k,
        uint64_t now_token = 0);

    // Ebbinghaus decay in the crystal clock:
    //   eff = importance * exp(-age_days / tau) * (1 + 0.1*min(hits,10))
    //   age_days = (now_token - last_access_token) / tokens_per_day
    // Drops every crystal whose effective importance fell below
    // min_importance; returns how many were dropped. Call periodically (e.g.
    // from dream()).
    //
    // `now_token` must be in the same clock as crystallize()'s stream_pos.
    // The previous revision took `double now_unix_seconds` and then IGNORED the
    // argument, deriving age as `last_access_token / 100000.0` — which read a
    // stream position as if it were already an elapsed age. The result was
    // backwards: a crystal created late (large stream position) decayed
    // immediately, and one created at position 0 could never decay at all.
    // The parameter is now honoured, and named for what it actually is.
    size_t decay(uint64_t now_token);

    // Reinforcement: mark a crystal as USED at `now_token` (resetting its decay
    // age to zero) and raise importance by `boost` (cfg_.reinforce_boost when
    // boost < 0), clamped to 1.0. Returns false when the id is unknown.
    bool reinforce(uint64_t id, uint64_t now_token, float boost = -1.0f);

    // Look up one crystal by id (nullptr when absent). For tests and for the
    // dream pass, which needs to reason about individual memories.
    const MemoryCrystal* find(uint64_t id) const;

    // Persistence (compact binary sidecar next to the model).
    bool save(const std::string& path) const;
    bool load(const std::string& path);

    size_t size() const { return crystals_.size(); }
    const Config& config() const { return cfg_; }
    void clear() { crystals_.clear(); next_id_ = 1; }

    // Every crystal id, in creation order. Ids are the only stable handle a
    // caller has on a crystal, and a dream pass needs to walk them.
    std::vector<uint64_t> ids() const {
        std::vector<uint64_t> out;
        out.reserve(crystals_.size());
        for (const MemoryCrystal& c : crystals_) out.push_back(c.id);
        return out;
    }

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
