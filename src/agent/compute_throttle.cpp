// =============================================================================
//  OmniSeed — compute_throttle.cpp
//  Adaptive compute: shallow thinking for trivial asks, deep cycles for
//  planning. Keywords + length heuristics; the self-improvement stats can
//  later override per-task.
// =============================================================================
#include "omniseed/agent/agent.h"

#include <cctype>

namespace omniseed {

using platform::current_rss_bytes;

ComputeThrottle::Level ComputeThrottle::classify(const std::string& in) const {
    std::string low;
    low.reserve(in.size());
    for (const char c : in)
        low += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    auto has = [&](const char* needle) -> bool {
        const std::string n(needle);
        size_t pos = 0;
        while ((pos = low.find(n, pos)) != std::string::npos) {
            const bool left_ok = pos == 0 || !std::isalpha(static_cast<unsigned char>(low[pos - 1]));
            const size_t end = pos + n.size();
            const bool right_ok = end >= low.size() || !std::isalpha(static_cast<unsigned char>(low[end]));
            if (left_ok && right_ok) return true;
            ++pos;
        }
        return false;
    };
    static const char* kDeep[] = {
        "plan", "design", "build", "create", "analyze", "compare",
        "why", "how do", "how to", "step", "strategy", "research",
        "write a", "implement", "debug", "refactor",
    };
    for (const char* k : kDeep) if (has(k)) return Level::Deep;
    static const char* kFast[] = {
        "what time", "time is it", "hi", "hello", "thanks", "ok",
        "yes", "no", "who are you", "help",
    };
    for (const char* k : kFast) if (has(k)) return Level::Fast;
    if (in.size() < 24) return Level::Fast;
    if (in.size() > 160) return Level::Deep;
    return Level::Balanced;
}

int32_t ComputeThrottle::max_new_tokens(Level lv) const {
    switch (lv) {
        case Level::Fast:     return 48;
        case Level::Balanced: return 160;
        case Level::Deep:     return 512;
    }
    return 160;
}

int32_t ComputeThrottle::max_think_tokens(Level lv) const {
    switch (lv) {
        case Level::Fast:     return 0;    // skip <|think|> entirely
        case Level::Balanced: return 96;
        case Level::Deep:     return 384;
    }
    return 96;
}

int32_t ComputeThrottle::max_tool_turns(Level lv) const {
    switch (lv) {
        case Level::Fast:     return 1;
        case Level::Balanced: return 3;
        case Level::Deep:     return 6;
    }
    return 3;
}

const char* ComputeThrottle::name(Level lv) const {
    switch (lv) {
        case Level::Fast:     return "fast";
        case Level::Balanced: return "balanced";
        case Level::Deep:     return "deep";
    }
    return "?";
}

// ---------------------------------------------------------------------------
// RSS watermark (grok S01): zero-RAM budget enforcement. Called from the
// generation loop every token; the platform RSS read is ~microseconds.
// ---------------------------------------------------------------------------
int ComputeThrottle::rss_zone() const {
    const uint64_t mb = platform::current_rss_bytes() / (1024ull * 1024ull);
    if (mb >= hard_rss_mb_) return 2;
    if (mb >= soft_rss_mb_) return 1;
    return 0;
}

} // namespace omniseed
