// =============================================================================
//  OmniSeed — agent.h
//  Agent intelligence stack:
//
//   * GrammarDecoder   — grammar-constrained decoding for valid JSON tool
//                        calls (Hermes/MCP-compatible), streamed token by
//                        token. The model proposes; the grammar disposes.
//   * ToolRegistry     — tool schemas + validation + builtin tools.
//   * AgentLoop        — perceive -> think (<|think|>) -> act (tool JSON)
//                        -> observe -> repeat, with memory integration.
//   * SelfImprovement  — success-path caching, error-pattern learning and
//                        the nightly Dream-State consolidation pass.
//   * ComputeThrottle  — adaptive compute: Fast/Balanced/Deep modes.
// =============================================================================
#pragma once

#include "omniseed/core/rwkv.h"
#include "omniseed/core/tokenizer.h"
#include "omniseed/memory/memory.h"

#include <functional>
#include <string>
#include <vector>

namespace omniseed {

class GgufLoader;

// ===========================================================================
// GrammarDecoder — streamed JSON validity for constrained generation
// ===========================================================================
class GrammarDecoder {
public:
    void reset();

    // Would appending `text` keep the stream a valid JSON prefix?
    bool accepts(const std::string& text) const;

    // Append text (caller must have called accepts() first).
    void feed(const std::string& text);

    bool complete() const { return complete_ && depth_ == 0; }
    bool failed() const { return failed_; }

    // The accumulated JSON text.
    const std::string& text() const { return buf_; }

private:
    std::string buf_;
    int  depth_      = 0;      // nested {}/[]
    bool in_string_  = false;
    bool escaped_    = false;
    bool complete_   = false;
    bool failed_     = false;
};

// ===========================================================================
// ToolRegistry — schemas, validation, invocation
// ===========================================================================
struct ToolParam {
    std::string name;
    std::string type;          // "string" | "number" | "boolean"
    std::string description;
    bool required = false;
};

struct Tool {
    std::string name;
    std::string description;
    std::vector<ToolParam> params;
    std::function<std::string(const std::string& args_json, bool& ok)> fn;
};

class ToolRegistry {
public:
    void add(Tool tool);
    bool add_builtin(const std::string& name);

    bool has(const std::string& name) const;
    const Tool* find(const std::string& name) const;
    size_t size() const { return tools_.size(); }

    // JSON array of {"name","description","parameters"} for prompting.
    std::string schemas_json() const;

    // Validates args_json against the tool's parameter schema.
    bool validate(const std::string& name, const std::string& args_json,
                  std::string& err) const;

    // Invokes; returns the JSON result string. ok=false on failure.
    std::string invoke(const std::string& name, const std::string& args_json,
                       bool& ok) const;

private:
    std::vector<Tool> tools_;
};

// ===========================================================================
// ComputeThrottle — adaptive compute levels
// ===========================================================================
class ComputeThrottle {
public:
    enum class Level { Fast, Balanced, Deep };

    // Heuristic complexity classification of the user turn.
    Level classify(const std::string& input) const;

    int32_t max_new_tokens(Level lv) const;
    int32_t max_think_tokens(Level lv) const;
    int32_t max_tool_turns(Level lv) const;

    const char* name(Level lv) const;

private:
    // cheap keyword/length heuristics; refined by the self-improvement stats
};

// ===========================================================================
// SelfImprovement — trace cache + dream-state consolidation
// ===========================================================================
struct TaskTrace {
    std::string task_key;                 // normalized task signature
    std::vector<std::string> steps;       // successful tool call sequence
    uint32_t success_count = 0;
    uint32_t fail_count    = 0;
    double   avg_ms        = 0.0;
    uint64_t last_used     = 0;          // unix seconds
};

class SelfImprovement {
public:
    struct Config {
        size_t  max_traces   = 256;
        double  replay_min_success_rate = 0.8;  // trust threshold
    };

    explicit SelfImprovement(const Config& cfg = {}) : cfg_(cfg) {}

    // Normalizes a task string to a stable key (lowercase, collapsed ws).
    static std::string task_key(const std::string& task);

    void record(const std::string& key, const std::vector<std::string>& steps,
                bool success, double ms);

    // Returns a cached successful path if trustworthy.
    bool find_replay(const std::string& key,
                     std::vector<std::string>& steps) const;

    // Dream-State Consolidation: decay stale traces, prune failures,
    // promote frequently-successful ones. Called periodically (e.g. nightly).
    void dream();

    size_t size() const { return traces_.size(); }
    const Config& config() const { return cfg_; }

    bool save(const std::string& path) const;
    bool load(const std::string& path);

private:
    Config cfg_;
    std::vector<TaskTrace> traces_;
};

// ===========================================================================
// AgentLoop — the perceive/think/act cycle
// ===========================================================================
class AgentLoop {
public:
    struct Config {
        int32_t  max_new_tokens = 160;
        int32_t  max_turns      = 6;
        int32_t  retrieve_k     = 3;     // crystals injected per turn
        bool     allow_tools    = true;
        // Sampling (temperature <= 0 => greedy). Seeded SplitMix64 stream:
        // deterministic across runs with the same seed.
        float    temperature    = 0.0f;
        int32_t  top_k          = 0;
        uint64_t seed           = 42;
        // Streaming: invoked per decoded piece during generate() (chat UI).
        // nullptr = buffered (default). Must not throw.
        std::function<void(const std::string&)> on_token;
    };

    AgentLoop(const RwkvModel& model, const Tokenizer& tok,
              const ToolRegistry& tools, MemoryCrystals& memory,
              const ComputeThrottle& throttle, SelfImprovement& improve,
              const Config& cfg = {})
        : model_(model), tok_(tok), tools_(tools), memory_(memory),
          throttle_(throttle), improve_(improve), cfg_(cfg) {}

    struct Result {
        std::string reply;
        std::vector<std::string> tool_trace;   // "name(args) -> result"
        int32_t turns = 0;
        double  ms = 0.0;
        uint64_t peak_rss = 0;
    };

    // One user turn -> final reply (executing any tool calls en route).
    Result run(const std::string& user_input);

    // Single autoregressive generation helper (also used by the CLI).
    // Forwards seed_token, then repeatedly picks/feeds tokens. Stops at EOS,
    // a stop piece, the token cap, or grammar completion. Grammar (if given)
    // constrains each candidate piece before it is committed.
    std::string generate(RwkvState& st, int32_t seed_token, int32_t max_tokens,
                         const std::vector<int32_t>& stop_pieces,
                         GrammarDecoder* grammar);

private:
    // Prompt assembly: memory crystals + tool schemas + user turn.
    std::vector<int32_t> build_prompt(const std::string& user_input,
                                      const ComputeThrottle::Level lv) const;

    const RwkvModel&      model_;
    const Tokenizer&      tok_;
    const ToolRegistry&   tools_;
    MemoryCrystals&       memory_;
    const ComputeThrottle& throttle_;
    SelfImprovement&      improve_;
    Config                cfg_;
};

} // namespace omniseed
