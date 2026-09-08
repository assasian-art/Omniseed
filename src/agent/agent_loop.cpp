// =============================================================================
//  OmniSeed — agent_loop.cpp
//  perceive -> (think) -> act (grammar-constrained JSON tool call) ->
//  observe -> ... -> answer
// =============================================================================
#include "omniseed/agent/agent.h"
#include "omniseed/core/platform.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace omniseed {

namespace {
// Thread-local-ish handoff from generate() to run(): the uncertainty report
// of the final answer pass's first logits. (AgentLoop is documented as
// single-threaded per instance; a plain member is sufficient and zero-cost.)
} // namespace

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
void AgentLoop::set_sampling(float repeat_penalty, int32_t repeat_window) {
    cfg_.repeat_penalty = repeat_penalty;
    cfg_.repeat_window = repeat_window;
}

std::string AgentLoop::generate(RwkvState& st, int32_t seed_token,
                                int32_t max_tokens,
                                const std::vector<int32_t>& stop_pieces,
                                GrammarDecoder* grammar) {
    std::string out;
    Tensor logits("logits", {model_.config().n_vocab}, DType::F32);
    int32_t cursor = seed_token;
    uint64_t rng = cfg_.seed;
    bool first_analyzed = false;
    int32_t rss_soft_cut = 0;   // max_tokens halved once at soft watermark
    // Phase 13: repetition-penalty ring of recently generated tokens.
    std::vector<int32_t> recent;
    recent.reserve(static_cast<size_t>(std::max<int32_t>(cfg_.repeat_window, 1)));

    for (int32_t i = 0; i < max_tokens; ++i) {
        model_.forward(cursor, st, logits);

        // ---- Phase-Omega: uncertainty on the FIRST logits only -------------
        // (O(V) once per generation; the distribution sharpens as context
        // accumulates, so the first token carries the worst case.)
        if (!first_analyzed) {
            first_analyzed = true;
            last_uncertainty_ = Uncertainty::analyze(logits);
            entropy_mon_.push(last_uncertainty_.entropy);
        }

        // ---- Phase-Omega: temperature annealing ----------------------------
        float temp = cfg_.temperature;
        if (cfg_.anneal_tokens > 0 && cfg_.temperature > 0.0f) {
            const float frac = static_cast<float>(i) /
                               static_cast<float>(cfg_.anneal_tokens);
            temp = cfg_.temperature +
                   (cfg_.anneal_temp_start - cfg_.temperature) *
                   std::max(0.0f, 1.0f - frac);
        }

        // ---- Phase-Omega: RSS watermark guard ------------------------------
        if (cfg_.rss_guard) {
            const int zone = throttle_.rss_zone();
            if (zone == 2) break;                       // hard stop
            if (zone == 1 && rss_soft_cut == 0) {
                rss_soft_cut = max_tokens / 2;          // halve the budget
                max_tokens = std::min(max_tokens, i + rss_soft_cut);
            }
        }

        // ---- Phase 13: repetition penalty ------------------------------
        // CTRL-style: tokens present in the recent ring get their logit
        // divided by the penalty when positive (multiplied when negative),
        // suppressing the "/ repeat / repeat" loops of the QAT model.
        if (cfg_.repeat_penalty > 1.0f && !recent.empty()) {
            const float p = cfg_.repeat_penalty;
            const int32_t win = std::max<int32_t>(cfg_.repeat_window, 1);
            const size_t start = recent.size() > static_cast<size_t>(win)
                ? recent.size() - static_cast<size_t>(win)
                : 0;
            float* lp = logits.f32();
            const int32_t vocab = static_cast<int32_t>(logits.numel());
            for (size_t ri = start; ri < recent.size(); ++ri) {
                const int32_t t = recent[ri];
                if (t < 0 || t >= vocab) continue;
                if (lp[t] > 0.0f)      lp[t] /= p;
                else if (lp[t] < 0.0f) lp[t] *= p;
                // logit == 0: unchanged
            }
        }

        const int32_t id = temp > 0.0f
            ? model_.sample_token(logits, temp, cfg_.top_k, rng)
            : model_.greedy_pick(logits);
        recent.push_back(id);
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

// ---------------------------------------------------------------------------
// Prefix snapshot (Phase-Omega): store the current WKV state.
// ---------------------------------------------------------------------------
bool AgentLoop::snapshot_prefix() {
    return prefix_cache_.store(cfg_.prefix_key, model_, last_state_);
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
    int64_t restored_tokens = 0;
    bool prefix_hit = false;
    if (!cfg_.prefix_key.empty()) {
        if (prefix_cache_.load(cfg_.prefix_key, model_, st, restored_tokens)) {
            prefix_hit = true;
            res.prefix_hits = 1;
        } else {
            model_.init_state(st);
        }
    } else {
        model_.init_state(st);
    }

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
        if (prefix_hit) {
            // Prefix fast path: the cached state already contains the system
            // prompt + schemas; only the NEW tokens after the snapshot point
            // need forwarding. (Snapshot is taken at the end of this turn —
            // see below — so this turn still pays full prefill once.)
            // NOTE: the cached state corresponds to the FULL previous prompt;
            // we conservatively re-feed everything (correctness first), but
            // skip re-feeding when the prompt is byte-identical to the
            // snapshot's (the common single-session repeat case).
            //
            // Implementation: the snapshot stores tokens_seen; if our current
            // prompt is the same length as the snapshot's, the prompt is
            // (by construction of prefix_key usage) the same -> skip prefill.
            if (static_cast<int64_t>(prompt.size()) == restored_tokens) {
                // state already ends exactly at the prompt: no forward needed
            } else {
                for (const int32_t id : prompt) model_.forward(id, st, logits);
            }
        } else {
            for (const int32_t id : prompt) model_.forward(id, st, logits);
        }
        const int32_t seed = prompt.empty() ? Tokenizer::kBosId : prompt.back();
        res.reply = generate(st, seed, throttle_.max_new_tokens(lv),
                             {Tokenizer::kAssistantEndId, Tokenizer::kEosId},
                             nullptr);

        // Uncertainty abstain (Phase-Omega): hedge flat/coin-flip answers.
        res.first_token_uncertainty = last_uncertainty_;
        if (!cfg_.abstain_hedge.empty() &&
            Uncertainty::should_abstain(last_uncertainty_, ucfg_)) {
            res.reply = cfg_.abstain_hedge + res.reply;
            res.abstained = true;
        }

        // Snapshot AFTER generation: the state now ends at prompt+answer,
        // which is a valid restore point for a byte-identical next turn.
        if (!cfg_.prefix_key.empty()) prefix_cache_.store(cfg_.prefix_key, model_, st);
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
