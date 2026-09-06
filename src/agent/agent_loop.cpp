// =============================================================================
//  OmniSeed — agent_loop.cpp
//  perceive -> (think) -> act (grammar-constrained JSON tool call) ->
//  observe -> ... -> answer
// =============================================================================
#include "omniseed/agent/agent.h"
#include "omniseed/core/platform.h"

#include <cmath>
#include <cstring>

namespace omniseed {

// ===========================================================================
// Prompt assembly: crystals + tool schemas + user turn
// ===========================================================================
std::vector<int32_t> AgentLoop::build_prompt(
        const std::string& user_input,
        const ComputeThrottle::Level lv) const {
    std::string prompt;

    // 1) retrieved memory crystals (compact semantic long-term memory)
    if (cfg_.retrieve_k > 0 && memory_.size() > 0) {
        const auto q = tok_.encode(user_input, false);
        const auto crystals = memory_.retrieve(q, cfg_.retrieve_k);
        if (!crystals.empty()) {
            prompt += "[memory]\n";
            for (const MemoryCrystal& c : crystals) {
                prompt += "- " + c.summary + "\n";
            }
            prompt += "\n";
        }
    }

    // 2) tool schemas (only when tools are allowed & useful for the level)
    if (cfg_.allow_tools && lv != ComputeThrottle::Level::Fast &&
        tools_.size() > 0) {
        prompt += "[tools] " + tools_.schemas_json() + "\n";
        prompt += "To call a tool emit: {\"tool\":\"<name>\","
                  "\"args\":{...}}\n\n";
    }

    // 3) the user turn, chat-wrapped
    std::vector<int32_t> ids = tok_.encode_chat(user_input);
    std::vector<int32_t> prefix = tok_.encode(prompt, false);
    ids.insert(ids.begin(), prefix.begin(), prefix.end());
    return ids;
}

// ===========================================================================
// Generation with optional grammar constraint.
// Proper autoregressive loop: forward(seed) -> logits -> pick -> feed back.
// ===========================================================================
std::string AgentLoop::generate(RwkvState& st, int32_t seed_token,
                                int32_t max_tokens,
                                const std::vector<int32_t>& stop_pieces,
                                GrammarDecoder* grammar) {
    std::string out;
    Tensor logits("logits", {model_.config().n_vocab}, DType::F32);
    int32_t cursor = seed_token;
    uint64_t rng = cfg_.seed;

    for (int32_t i = 0; i < max_tokens; ++i) {
        model_.forward(cursor, st, logits);
        const int32_t id = cfg_.temperature > 0.0f
            ? model_.sample_token(logits, cfg_.temperature, cfg_.top_k, rng)
            : model_.greedy_pick(logits);
        if (id == Tokenizer::kEosId) break;

        const std::string piece = tok_.piece(id);

        // grammar check: skip pieces that would break JSON validity
        if (grammar != nullptr) {
            if (!grammar->accepts(piece)) continue;
            grammar->feed(piece);
        }

        out += piece;

        // streaming hook (chat UI): invoked after the piece is committed
        if (cfg_.on_token) cfg_.on_token(piece);

        bool stop = false;
        for (const int32_t sp : stop_pieces) {
            if (sp == id) { stop = true; break; }
        }
        if (stop) break;

        if (grammar != nullptr && grammar->complete()) break;
        cursor = id;
    }
    return out;
}

// ===========================================================================
// One agent turn
// ===========================================================================
AgentLoop::Result AgentLoop::run(const std::string& user_input) {
    Result res;
    const auto t0 = platform::now_ms();

    const ComputeThrottle::Level lv = throttle_.classify(user_input);

    // Self-improvement: replay a known-good path if we have one.
    const std::string key = SelfImprovement::task_key(user_input);
    std::vector<std::string> replay;
    std::vector<std::string> trace;

    RwkvState st;
    model_.init_state(st);

    if (cfg_.allow_tools) {
        // feed prompt
        const std::vector<int32_t> prompt = build_prompt(user_input, lv);
        Tensor logits("logits", {model_.config().n_vocab}, DType::F32);
        for (const int32_t id : prompt) model_.forward(id, st, logits);
        int32_t seed = prompt.empty() ? Tokenizer::kBosId : prompt.back();

        // up to N tool turns
        for (int32_t turn = 0; turn < throttle_.max_tool_turns(lv); ++turn) {
            GrammarDecoder grammar;
            std::string candidate = generate(st, seed,
                                             throttle_.max_new_tokens(lv),
                                             {Tokenizer::kAssistantEndId},
                                             &grammar);
            if (!grammar.complete() || grammar.failed()) break;

            // extract {"tool": "...", "args": {...}}
            const std::string& js = grammar.text();
            const auto tpos = js.find("\"tool\"");
            const auto cpos = js.find(':', tpos);
            const auto q1 = js.find('"', cpos);
            const auto q2 = js.find('"', q1 + 1);
            if (tpos == std::string::npos || cpos == std::string::npos ||
                q1 == std::string::npos || q2 == std::string::npos) break;
            const std::string tool_name = js.substr(q1 + 1, q2 - q1 - 1);

            const auto apos = js.find("\"args\"");
            std::string args = "{}";
            if (apos != std::string::npos) {
                const auto ab = js.find('{', apos);
                const auto ae = js.rfind('}');
                if (ab != std::string::npos && ae != std::string::npos && ae > ab)
                    args = js.substr(ab, ae - ab + 1);
            }

            bool ok = false;
            const std::string result = tools_.invoke(tool_name, args, ok);
            trace.push_back(tool_name + "(" + args + ") -> " + result);
            res.tool_trace.push_back(tool_name + "(" + args + ")");

            // feed the observation back for the next turn
            const std::string obs = "[result] " + result + "\n";
            auto obs_ids = tok_.encode(obs, false);
            for (const int32_t id : obs_ids) model_.forward(id, st, logits);
            seed = obs_ids.empty() ? seed : obs_ids.back();

            if (!ok) break;
        }
    }

    // final answer pass
    {
        const std::vector<int32_t> prompt = build_prompt(user_input, lv);
        Tensor logits("logits", {model_.config().n_vocab}, DType::F32);
        for (const int32_t id : prompt) model_.forward(id, st, logits);
        const int32_t seed = prompt.empty() ? Tokenizer::kBosId : prompt.back();
        res.reply = generate(st, seed, throttle_.max_new_tokens(lv),
                             {Tokenizer::kAssistantEndId, Tokenizer::kEosId},
                             nullptr);
    }

    // self-improvement bookkeeping
    improve_.record(key, trace, !trace.empty() &&
                                    res.tool_trace.size() > 0 &&
                                    !res.reply.empty(),
                    platform::now_ms() - t0);

    res.turns = static_cast<int32_t>(res.tool_trace.size());
    res.ms = platform::now_ms() - t0;
    res.peak_rss = platform::peak_rss_bytes();
    return res;
}

} // namespace omniseed
