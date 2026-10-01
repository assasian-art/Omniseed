// =============================================================================
//  OmniSeed — grammar_decoder.cpp
//  Streaming JSON grammar state machine: rejects invalid continuations at
//  the token level so a 0.1-1B ternary model cannot emit broken tool calls.
// =============================================================================
#include "omniseed/agent/agent.h"

namespace omniseed {

void GrammarDecoder::reset() {
    buf_.clear();
    depth_ = 0;
    in_string_ = false;
    escaped_ = false;
    complete_ = false;
    failed_ = false;
}

bool GrammarDecoder::accepts(const std::string& text) const {
    if (failed_) return false;
    if (complete_) return false;
    if (text.empty()) return true;
    const std::string combined = buf_ + text;
    int depth = 0;
    bool in_str = false;
    bool esc = false;
    bool started = false;
    size_t complete_pos = std::string::npos;
    auto is_space = [](char c){return c==' '||c=='\t'||c=='\n'||c=='\r';};
    for (size_t i = 0; i < combined.size(); ++i) {
        const char ch = combined[i];
        if (in_str) {
            if (esc) esc = false;
            else if (ch == '\\') esc = true;
            else if (ch == '"') in_str = false;
            continue;
        }
        if (ch == '"') { in_str = true; started = true; }
        else if (ch == '{' || ch == '[') { ++depth; started = true; }
        else if (ch == '}' || ch == ']') {
            --depth;
            if (depth < 0) return false;
            if (depth == 0 && started && !in_str && complete_pos == std::string::npos) complete_pos = i;
        } else if (is_space(ch)) {
        } else {
            if (!started) return false;
            if (depth == 0 && complete_pos != std::string::npos) return false;
        }
    }
    if (complete_pos != std::string::npos) {
        for (size_t i = complete_pos + 1; i < combined.size(); ++i)
            if (!is_space(combined[i])) return false;
    }
    return true;
}

void GrammarDecoder::feed(const std::string& text) {
    if (failed_) return;
    if (!accepts(text)) {
        failed_ = true;
        return;
    }
    buf_ += text;

    // Recompute structural state (cheap; strings are tiny).
    depth_ = 0;
    in_string_ = false;
    escaped_ = false;
    for (const char ch : buf_) {
        if (in_string_) {
            if (escaped_) { escaped_ = false; }
            else if (ch == '\\') { escaped_ = true; }
            else if (ch == '"') { in_string_ = false; }
            continue;
        }
        if (ch == '"') in_string_ = true;
        else if (ch == '{' || ch == '[') ++depth_;
        else if (ch == '}' || ch == ']') --depth_;
    }

    // Complete when the top-level object closes.
    complete_ = (depth_ == 0 && !in_string_ && !buf_.empty() &&
                 (buf_.front() == '{' || buf_.front() == '['));
}

} // namespace omniseed
