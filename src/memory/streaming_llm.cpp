// =============================================================================
//  OmniSeed — streaming_llm.cpp
//  StreamingLLM: sink + sliding-window token management.
//  RWKV's O(1) state carries long context; sinks stabilize very long
//  streams; the window bounds re-injection cost; retired tokens feed
//  crystallization.
// =============================================================================
#include "omniseed/memory/memory.h"

#include <algorithm>

namespace omniseed {

bool StreamingLlm::push(int32_t token) {
    ++total_seen_;

    // Phase A: fill sinks.
    if (static_cast<int32_t>(sinks_.size()) < cfg_.n_sinks) {
        sinks_.push_back(token);
        return true;
    }

    // Phase B: fill window.
    if (static_cast<int32_t>(window_.size()) < cfg_.window_size) {
        window_.push_back(token);
        return true;
    }

    // Phase C: slide — retire oldest, admit newest.
    retired_.push_back(window_.front());
    window_.erase(window_.begin());
    window_.push_back(token);
    return true;
}

std::vector<int32_t> StreamingLlm::drain_retired() {
    std::vector<int32_t> out;
    out.swap(retired_);
    return out;
}

void StreamingLlm::reset() {
    sinks_.clear();
    window_.clear();
    retired_.clear();
    total_seen_ = 0;
}

} // namespace omniseed
