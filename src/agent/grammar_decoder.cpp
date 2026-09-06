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

    // Simulate the state machine over the candidate text.
    int depth = depth_;
    bool in_str = in_string_;
    bool esc = escaped_;

    for (const char ch : text) {
        if (in_str) {
            if (esc) {
                esc = false;
            } else if (ch == '\\') {
                esc = true;
            } else if (ch == '"') {
                in_str = false;
            }
            continue;
        }
        switch (ch) {
            case '"':
                in_str = true;
                break;
            case '{':
            case '[':
                ++depth;
                break;
            case '}':
            case ']':
                --depth;
                if (depth < 0) return false;   // unbalanced
                break;
            default:
                break;
        }
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
