// =============================================================================
//  OmniSeed — cli/main.cpp
//  Command-line interface for the OmniSeed agent kernel.
//
//  Commands:
//    omniseed info                          system + build info
//    omniseed chat  --model M               interactive chat REPL
//    omniseed gen    --model M --prompt "..."
//    omniseed ask    --model M "task"       one agent turn (tools + memory)
//    omniseed bench  --model M              tokens/sec + peak RSS report
//    omniseed tools                         list builtin tools
//    omniseed selftest                      run the built-in sanity checks
//    omniseed demo-sensory                  Sensory Fingerprinting demo
//    omniseed demo-emotional                Emotional Resonance demo
//    omniseed demo-skills                   Flash Skills + Zero-Shot Synthesis
//    omniseed demo-swarm                    Collaborative Swarm demo (loopback)
//    omniseed demo-memory                   attention sinks + Memory Crystals
//    omniseed demo-soul                     persona + memory + recall + decay
//    omniseed demo-audio <wav>              sound events + scene + wake word
//    omniseed demo-vision <w> <h>           pointer/spatial/tracker demo
//    omniseed dream                         run Dream-State consolidation now
// =============================================================================
#include "omniseed/agent/agent.h"
#include "omniseed/agent/agent_intel.h"
#include "omniseed/agent/flash_skills.h"
#include "omniseed/audio/audio.h"
#include "omniseed/audio/audio_events.h"
#include "omniseed/core/gguf_loader.h"
#include "omniseed/core/lora.h"
#include "omniseed/core/platform.h"
#include "omniseed/core/rwkv.h"
#include "omniseed/core/tokenizer.h"
#include "omniseed/decision_head.h"
#include "omniseed/memory/memory.h"
#include "omniseed/runtime/emotional.h"
#include "omniseed/runtime/sensory.h"
#include "omniseed/runtime/swarm.h"
#include "omniseed/runtime/token_bus.h"
#include "omniseed/soul.h"
#include "omniseed/streaming_decision.h"
#include "omniseed/feedback_hook.h"
#include "omniseed/vision/vision.h"
#include "omniseed/vision/vision_tasks.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

using namespace omniseed;

namespace {

void print_usage() {
    std::printf(
        "OmniSeed v0.2 — sub-400MB multi-modal micro-LLM agent kernel\n"
        "usage: omniseed <command> [options]\n"
        "\n"
        "commands:\n"
        "  info                        show system & build information\n"
        "  chat     --model M          interactive chat REPL\n"
        "  gen      --model M --prompt \"...\"\n"
        "  ask      --model M \"task\"  one agent turn (tools + memory)\n"
        "  bench    --model M          tokens/sec + peak RSS benchmark\n"
        "  tools                       list builtin tools\n"
        "  selftest                    quick numeric sanity checks\n"
        "  demo-sensory                Sensory Fingerprinting (enroll/verify)\n"
        "  demo-emotional              Emotional Resonance (voice+text)\n"
        "  demo-skills                 Flash Skills + Zero-Shot Synthesis\n"
        "  demo-swarm                  Collaborative Swarm Protocol\n"
        "  demo-memory                 1M-token window + Memory Crystals\n"
        "  demo-soul                   persona + memory + recall + decay\n"
        "  demo-stream                 streaming decisions (debounce+hysteresis)\n"
        "  demo-feedback               prediction -> outcome -> persisted record\n"
        "  demo-audio F.wav            sound events / scene / wake word\n"
        "  enroll-audio D set          build an owner-voice dataset from D/*.wav\n"
        "  demo-vision                 pointer grounding + spatial map\n"
        "  dream                       Dream-State consolidation pass\n"
        "\n"
        "options:\n"
        "  --model PATH     model GGUF path (default: ./models/omniseed.gguf)\n"
        "  --prompt TEXT    input text\n"
        "  --max-tokens N   generation budget (default 160)\n"
        "  --temperature T  sampling temperature, 0 = greedy (default 0)\n"
        "  --top-k K        keep K highest-probability tokens (0 = all)\n"
        "  --repeat-penalty P  penalize recently generated tokens (1.15-1.25\n"
        "                      breaks loops; 1.0 = off, default 1.0)\n"
        "  --stop TEXT       halt when the decoded tail matches TEXT\n"
        "                      (repeatable); --stop-defaults enables the\n"
        "                      assistant set: \\nUser:, \\nAssistant:, \\nAssistant::\n"
        "  --assistant-lora P  attach a LoRA sidecar (models/assistant-lora.gguf)\n"
        "                      to steer replies (assistant behavior); used by\n"
        "                      gen/ask/chat\n"
        "  --assistant-mode    shorthand for --assistant-lora with the default\n"
        "                      sidecar path\n"
        "  --seed S         sampling seed, deterministic per seed\n"
        "  --mode M         System-1 decision head routing:\n"
        "                     off            System-2 only (default)\n"
        "                     text-only      alias for 'off': generate text and\n"
        "                                    never consult the decision head\n"
        "                     hybrid         consult the head first; act on it\n"
        "                                    when confident, else generate text\n"
        "                     decision-only  never generate text; return the\n"
        "                                    head's decision (or ABSTAIN)\n"
        "  --decision-threshold F  self-routing confidence bar (default 0.85)\n"
        "  --decision-head PATH    fitted projection blob for the head; without\n"
        "                          it the head is an UNTRAINED placeholder\n"
        "                          whose decisions are structurally valid but\n"
        "                          meaningless (see docs/JEV_INTEGRATION.md)\n"
        "  --quiet          suppress info logs\n");
}

void cmd_info() {
    std::printf("omniseed %s\n", platform::os_name());
    std::printf("  page size      : %u bytes\n", platform::page_size());
    std::printf("  peak rss       : %.2f MB\n",
                static_cast<double>(platform::peak_rss_bytes()) / 1048576.0);
    std::printf("  current rss    : %.2f MB\n",
                static_cast<double>(platform::current_rss_bytes()) / 1048576.0);
    std::printf("  budget         : 300 MB peak (kernel target)\n");
}

void cmd_tools() {
    ToolRegistry reg;
    for (const char* t : {"calc", "echo", "time"}) reg.add_builtin(t);
    std::printf("%s\n", reg.schemas_json().c_str());
}

int cmd_selftest() {
    // Minimal end-to-end numeric sanity: tokenizer round-trip + ternary matmul
    Tokenizer tok;
    if (!tok.build_minimal()) {
        std::printf("FAIL: tokenizer build\n");
        return 1;
    }
    const auto ids = tok.encode("hello", true);
    const std::string back = tok.decode(ids, false);
    if (back != "hello") {
        std::printf("FAIL: tokenizer round-trip (%s)\n", back.c_str());
        return 1;
    }

    const int8_t w[4] = {1, -1, 0, 1};
    std::vector<uint8_t> packed(bitnet::pack_ternary_packed_bytes(4));
    bitnet::pack_ternary(w, 4, packed.data());
    float y[2];
    const float x[2] = {1.0f, -2.0f};
    bitnet::bitlinear_forward(packed.data(), nullptr, 0.5f, x, y, 2, 2);
    if (std::fabs(y[0] - 0.5f * (1.0f + 2.0f)) > 1e-5f) {
        std::printf("FAIL: ternary matmul\n");
        return 1;
    }
    std::printf("selftest OK\n");
    return 0;
}

// ===========================================================================
// Demo: Sensory Fingerprinting (capability #2)
// ===========================================================================
PcmAudio synth_tone(double hz, double seconds, int32_t sr = 16000) {
    PcmAudio a;
    a.sample_rate = sr;
    a.samples.resize(static_cast<size_t>(seconds * sr));
    for (size_t i = 0; i < a.samples.size(); ++i) {
        const double t = static_cast<double>(i) / sr;
        a.samples[i] = static_cast<float>(
            0.4 * std::sin(2.0 * 3.14159265358979323846 * hz * t));
    }
    return a;
}

int cmd_demo_sensory() {
    SensoryFingerprint sf;

    SensoryObservation alice;
    const PcmAudio va = synth_tone(120.0, 1.5);
    alice.pcm = va.samples.data();
    alice.pcm_len = va.samples.size();
    for (int i = 0; i < 20; ++i) alice.digraph_ms.push_back(110.0f + i * 3.0f);
    sf.enroll(alice);
    sf.enroll(alice);
    std::printf("enrolled 'alice'  -> token %s\n", sf.identity_token().c_str());

    SensoryObservation alice2 = alice;
    bool m = false;
    const float s1 = sf.verify(alice2, m);
    std::printf("verify alice again -> score %.3f -> %s\n", s1,
                m ? "MATCH" : "reject");

    SensoryObservation bob;
    const PcmAudio vb = synth_tone(300.0, 1.5);
    bob.pcm = vb.samples.data();
    bob.pcm_len = vb.samples.size();
    for (int i = 0; i < 20; ++i) bob.digraph_ms.push_back(40.0f + (i % 4) * 15.0f);
    const float s2 = sf.verify(bob, m);
    std::printf("verify bob         -> score %.3f -> %s\n", s2,
                m ? "MATCH" : "reject");

    std::printf("peak RSS: %.2f MB\n",
                static_cast<double>(platform::peak_rss_bytes()) / 1048576.0);
    return 0;
}

// ===========================================================================
// Demo: Emotional Resonance (capability #4)
// ===========================================================================
int cmd_demo_emotional() {
    EmotionalResonance er;

    struct Case { const char* text; };
    const Case cases[] = {
        {"I love this, it is amazing and awesome!!!"},
        {"this is terrible and broken, I hate it"},
        {"we need this done asap, urgent, hurry please"},
        {"the meeting is at 3pm"},
    };
    for (const Case& c : cases) {
        EmotionalInput in;
        in.text = c.text;
        const EmotionalState st = er.perceive(in);
        std::printf("input : \"%s\"\n", c.text);
        std::printf("  -> %s (valence %+.2f, arousal %.2f, conf %.2f) %s\n",
                    emotion_name(st.emotion), st.valence, st.arousal,
                    st.confidence, er.token_fragment(st).c_str());
        const std::string modulated =
            er.modulate("Here is the result you asked for.", st);
        std::printf("  -> reply: \"%s\"\n\n", modulated.c_str());
    }
    std::printf("peak RSS: %.2f MB\n",
                static_cast<double>(platform::peak_rss_bytes()) / 1048576.0);
    return 0;
}

// ===========================================================================
// Demo: Flash Skills + Zero-Shot Skill Synthesis (capability #6, spec #8/9)
// ===========================================================================
int cmd_demo_skills() {
    FlashSkillPool pool;
    pool.install_builtins();
    std::printf("installed %zu builtin skills\n", pool.size());

    const char* tasks[] = {
        "what is 12*(3+4)?",
        "please compute 1234*5678 for me",
        "convert 5 km to miles",
        "count the letters of \"omniseed\"",
    };
    for (const char* task : tasks) {
        std::printf("task  : %s\n", task);
        const FlashSkill* skill = pool.match(task);
        const char* source = "builtin";
        if (!skill) {
            skill = pool.synthesize(task);
            source = "ZERO-SHOT synthesized";
        }
        if (!skill) {
            std::printf("  -> no skill; would fall back to generation\n\n");
            continue;
        }
        // Bind args from the task (demo: crude extraction).
        std::unordered_map<std::string, std::string> args;
        const std::string low = task;
        if (low.find("what is") != std::string::npos ||
            low.find("compute") != std::string::npos) {
            args["expr"] = (std::strstr(task, "1234") != nullptr)
                               ? "1234*5678" : "12*(3+4)";
        } else if (std::strstr(task, "km") != nullptr) {
            args["km"] = "5";
        } else if (std::strstr(task, "\"") != nullptr) {
            const char* q1 = std::strchr(task, '"');
            const char* q2 = std::strrchr(task, '"');
            if (q1 && q2 && q2 > q1) args["text"] = std::string(q1 + 1, q2 - q1 - 1);
        }
        const SkillResult r = pool.run(*skill, args);
        std::printf("  -> skill '%s' (%s)\n", skill->name.c_str(), source);
        for (const std::string& t : r.trace)
            std::printf("     | %s\n", t.c_str());
        std::printf("  -> result: %s\n\n", r.ok ? r.output.c_str() : "<failed>");
    }
    std::printf("peak RSS: %.2f MB\n",
                static_cast<double>(platform::peak_rss_bytes()) / 1048576.0);
    return 0;
}

// ===========================================================================
// Demo: Collaborative Swarm Protocol (capability #7)
// ===========================================================================
int cmd_demo_swarm() {
    LoopbackMesh& mesh = LoopbackMesh::instance();

    SwarmCoordinator::Config ca, cb;
    ca.node_id = "phone (this device)";
    cb.node_id = "drone (edge helper)";
    cb.caps = SwarmNode::CapModel | SwarmNode::CapVision | SwarmNode::CapAudio;
    SwarmCoordinator a(ca), b(cb);
    mesh.join(ca.node_id);
    mesh.join(cb.node_id);

    // Drone announces itself with lots of free RAM.
    SwarmNode nb;
    nb.id = cb.node_id;
    nb.caps = cb.caps;
    nb.free_ram_bytes = 280ull * 1024 * 1024;
    SwarmMessage hello;
    hello.type = SwarmMessage::Type::Hello;
    hello.from = cb.node_id;
    hello.payload.assign(8, '\0');
    const uint64_t caps = nb.caps;
    for (int i = 0; i < 8; ++i)
        hello.payload[i] = static_cast<char>((caps >> (8 * i)) & 0xFF);
    mesh.send(ca.node_id, hello);
    a.step(mesh);
    std::printf("phone sees %zu peer(s) in the swarm\n", a.peer_count());

    // Phone delegates a heavy task; drone executes.
    std::printf("phone delegates: \"summarize aerial survey frames\"\n");
    a.delegate("summarize aerial survey frames");
    b.step(mesh, [](const std::string&, std::string& result) {
        result = "survey done: 3 objects of interest at grid A1, B2, C4";
        return true;
    });

    // Drone shares compressed memories with the phone.
    b.share_memories(mesh, {"survey flight 42 completed at noon",
                            "obstacle reported near hangar"});
    while (a.step(mesh)) {}
    std::printf("phone received %zu shared memories from the swarm:\n",
                a.shared_memories().size());
    for (const std::string& m : a.shared_memories())
        std::printf("  [mem] %s\n", m.c_str());

    mesh.leave(ca.node_id);
    mesh.leave(cb.node_id);
    std::printf("peak RSS: %.2f MB\n",
                static_cast<double>(platform::peak_rss_bytes()) / 1048576.0);
    return 0;
}

// ===========================================================================
// Demo: 1M logical token window + Memory Crystals (capability #3, spec #10)
// ===========================================================================
int cmd_demo_memory() {
    Tokenizer tok;
    if (!tok.build_minimal()) {
        std::printf("tokenizer build failed\n");
        return 1;
    }
    StreamingLlm stream;
    MemoryCrystals mem;

    const std::string doc =
        "Alice is an engineer. She works on the OmniSeed kernel. "
        "Bob has a red car. The car is parked at the station. "
        "The demo ran at noon. OmniSeed crystallizes what matters. ";
    uint64_t fed = 0;
    std::vector<int32_t> retired_all;
    const auto ids = tok.encode(doc, false);
    // Simulate a long stream: repeat the doc until 2000 logical tokens.
    while (fed < 2000) {
        for (const int32_t id : ids) {
            stream.push(id);
            auto retired = stream.drain_retired();
            retired_all.insert(retired_all.end(), retired.begin(), retired.end());
            ++fed;
        }
    }
    std::printf("streamed %lld logical tokens; window=%zu sinks=%zu retired=%zu\n",
                static_cast<long long>(stream.total_seen()), stream.window().size(),
                stream.sinks().size(), retired_all.size());
    std::printf("RAM used by the context state: CONSTANT (O(1)) regardless.\n");

    // Crystallize the retired tail.
    if (!retired_all.empty()) {
        const size_t take = std::min<size_t>(retired_all.size(), 512);
        mem.crystallize(std::vector<int32_t>(
                            retired_all.end() - static_cast<long>(take),
                            retired_all.end()),
                        tok, stream.total_seen());
    }
    std::printf("crystals formed: %zu\n", mem.size());
    const auto q = tok.encode("who has the red car", false);
    for (const MemoryCrystal& c : mem.retrieve(q, 2))
        std::printf("  [crystal] %s\n", c.summary.c_str());
    std::printf("peak RSS: %.2f MB\n",
                static_cast<double>(platform::peak_rss_bytes()) / 1048576.0);
    return 0;
}

// ===========================================================================
// Demo: the soul — persona, memory, recall, decay (§30 + §31)
//
// The point of this demo is turn 3: the same question as turn 1, and the reply
// says so. Everything else here is bookkeeping around that one observable fact.
// ===========================================================================
int cmd_demo_soul() {
    Soul::Config cfg;   // memory is on by default
    Soul soul;
    if (!soul.init(cfg)) {
        std::printf("soul init failed: %s\n", soul.error().c_str());
        return 1;
    }

    std::printf("=== THE SOUL: memory, recall, decay ===\n\n");

    struct Turn { const char* q; const char* a; };
    const Turn turns[] = {
        {"what is my position size limit",
         "Two percent per trade, three per day, six per week."},
        {"how did the last aapl trade go",
         "It closed up on light volume."},
        {"what is my position size limit",   // the SAME question as turn 1
         "Two percent per trade, three per day, six per week."},
    };

    for (size_t i = 0; i < sizeof(turns) / sizeof(turns[0]); ++i) {
        SoulState st;
        const std::string reply = soul.converse(turns[i].q, turns[i].a, &st);
        std::printf("turn %zu  owner: %s\n", i + 1, turns[i].q);
        std::printf("        soul : %s\n", reply.c_str());
        std::printf("        recalled %zu memory(ies)", st.recall.size());
        if (!st.recall.empty())
            std::printf(" top relevance %.3f (%s)", st.recall.front().score,
                        memory_role_name(st.recall.front().role));
        std::printf(" -> %s\n", st.has_recall ? "yes" : "no");
        std::printf("        memory: %zu crystal(s)\n\n", soul.memory_size());
    }

    std::printf("memory: %zu crystals, %llu stored, %llu skipped, "
                "clock %llu tokens\n",
                soul.memory_size(),
                static_cast<unsigned long long>(soul.memory_stored()),
                static_cast<unsigned long long>(soul.memory_skipped()),
                static_cast<unsigned long long>(soul.memory_clock()));

    // --- how well does recall actually discriminate? ------------------------
    // Reported, not assumed, and self-verifying: each probe declares whether the
    // store should know it, and the demo prints MISMATCH when the floor got it
    // wrong. With the minimal byte-level tokenizer the embedding is close to a
    // character histogram, so UNRELATED English still scores high — these are
    // the numbers the default floor comes from.
    std::printf("\n--- relevance separation (min_relevance = %.2f) ---\n",
                cfg.min_relevance);
    struct Probe { const char* label; const char* q; bool in_store; };
    const Probe probes[] = {
        {"verbatim",           "what is my position size limit",  true},
        {"one char off",       "what is my position size limitt", true},
        {"paraphrase",         "what is the position size limit", true},
        {"another turn",       "how did the last aapl trade go",  true},
        {"NOT stored, short",  "what is the weather in paris",    false},
        {"NOT stored, long",
         "explain the difference between a bull market and a bear market in "
         "as much detail as you possibly can please", false},
    };
    for (const Probe& p : probes) {
        RecalledMemory m;
        const bool found = soul.top_match(p.q, &m);
        const bool claimed = found && m.score >= cfg.min_relevance;
        std::printf("  %-18s %.3f  %-11s %s\n", p.label, found ? m.score : 0.0f,
                    p.in_store ? "in store" : "not stored",
                    claimed == p.in_store ? "ok" : "MISMATCH");
    }

    // --- the Ebbinghaus curve, compressed into one call ---------------------
    const std::vector<uint64_t> ids = soul.crystals().ids();
    if (!ids.empty()) {
        const uint64_t keep = ids.front();   // the first thing it ever heard
        const uint64_t jump = 5000000;       // ~50 days at 100k tokens/day
        std::printf("\n--- simulated decay (tau = %.0f days, %.0f tokens/day) ---\n",
                    cfg.memory.decay_tau_days, cfg.memory.tokens_per_day);
        std::printf("clock advanced by %llu tokens (~%.0f days)\n",
                    static_cast<unsigned long long>(jump),
                    static_cast<double>(jump) / cfg.memory.tokens_per_day);
        std::printf("reinforced crystal #%llu (it came up again during the gap)\n",
                    static_cast<unsigned long long>(keep));
        soul.advance_memory_clock(jump);
        soul.reinforce_memory(keep);
        const size_t dropped = soul.decay_memories();
        std::printf("decay dropped %zu crystal(s); %zu left\n", dropped,
                    soul.memory_size());
        const MemoryCrystal* survivor = soul.crystals().find(keep);
        if (survivor)
            std::printf("  survivor #%llu: %s\n",
                        static_cast<unsigned long long>(survivor->id),
                        survivor->summary.c_str());
        std::printf("\nEvery other memory of the same age is gone. That is the whole\n"
                    "mechanism: being recalled and used is what resets the clock.\n");
    }
    std::printf("peak RSS: %.2f MB\n",
                static_cast<double>(platform::peak_rss_bytes()) / 1048576.0);
    return 0;
}

// ===========================================================================
// THE RUNTIME JOINT — load the FITTED decision head when one is committed.
//
// §32 fitted `models/heads/trading_head.bin` and the file has been in the tree
// ever since, but NO demo path ever loaded it: every stream ran on the seeded
// placeholder and reported actions that mean nothing. Loading it is the
// difference between demonstrating a filter and demonstrating a filter on a
// real projection.
//
// FAILS LOUDLY, NEVER SILENTLY. A missing or rejected blob falls back to the
// seeded placeholder with a warning naming the path, because a demo that
// quietly degrades to meaningless numbers is worse than one that says so.
// ===========================================================================
static const char* const kFittedDecisionHead = "models/heads/trading_head.bin";

// The value DERIVED by DECISION 1 (tools/derive_threshold.py + the gated test
// omniseed_threshold_derivation). It is the fail-closed default 0.50, and it is
// 0.50 because the rule found no threshold whose cumulative pool reached twice
// chance — NOT because someone liked 0.50. See docs/CALIBRATION.md §4.4.
//
// PAPER ONLY (L0): this bar gates PAPER commits. Live money is gated separately
// and that gate is C++-enforced; nothing here loosens it.
static constexpr float kDerivedMinConfidence = 0.50f;

// Returns true when a FITTED head is in place. `head` is always ready.
static bool load_decision_head(DecisionHead& head) {
    if (!head.init(768, 4242u)) return false;
    if (!head.load(kFittedDecisionHead)) {
        std::printf("head   : SEEDED PLACEHOLDER — %s\n", head.error().c_str());
        std::printf("         Actions below are WELL-FORMED and MEANINGLESS.\n");
        return false;
    }
    std::printf("head   : %s\n", head.provenance().c_str());
    if (!head.trained()) {
        std::printf("         WARNING: the blob loaded but is NOT fitted; the "
                    "actions below are placeholders.\n");
    }
    return head.trained();
}

// Build the filter config the demos share: the DECISION 1 threshold, with the
// head's own calibrated max as the switch bar's companion. Kept in one place so
// demo-stream and demo-feedback cannot drift apart on the number that decides
// whether anything is ever committed.
static StreamingConfig demo_filter_config() {
    StreamingConfig c;
    c.min_confidence = kDerivedMinConfidence;
    return c;
}

// ===========================================================================
// Demo: streaming decisions (Milestone 9, Phase 2.2)
//
// The filter's job is to make a head's output STABLE, and whether it does that
// is a property of the FILTER, not of the head — so the churn numbers below are
// meaningful either way. What the ACTIONS mean depends on whether a fitted
// projection was found, and the output says which.
//
// The headline number is the CHURN REDUCTION: how often the raw head output
// flips, versus how often the committed action flips. That is the whole point
// of the layer.
// ===========================================================================
int cmd_demo_stream() {
    DecisionHead head;
    if (!head.ready() && !head.init(768, 4242u)) {
        std::printf("decision head init failed: %s\n", head.error().c_str());
        return 1;
    }

    std::printf("=== STREAMING DECISIONS: debounce + hysteresis + release ===\n\n");
    const bool fitted = load_decision_head(head);
    if (!fitted)
        std::printf("NOTE   : an UNTRAINED head's actions are meaningless. This demo\n"
                    "         exercises the filter, not the judgement.\n");
    // State the bar and WHY it is what it is, so a reader never has to guess
    // whether 0.50 was chosen or computed.
    std::printf("filter : min_confidence=%.2f — DERIVED (DECISION 1), not chosen;\n"
                "         no threshold's pool reached 2x chance, so it stays "
                "fail-closed\n",
                kDerivedMinConfidence);
    std::printf("\n");

    // --- the stream: the real held-out h[E] if present ----------------------
    // Read through <fstream> rather than std::fopen: this translation unit does
    // not define _CRT_SECURE_NO_WARNINGS, and it should not have to gain it for
    // one demo.
    std::vector<float> H;
    int32_t n = 0;
    const char* kFixture = "tests/fixtures/head_calibration/trading/hidden.f32";
    {
        std::ifstream in(kFixture, std::ios::binary | std::ios::ate);
        if (in) {
            const std::streamoff sz = in.tellg();
            if (sz > 0 && (sz % 4) == 0) {
                H.resize(static_cast<size_t>(sz) / sizeof(float));
                in.seekg(0, std::ios::beg);
                in.read(reinterpret_cast<char*>(H.data()),
                        static_cast<std::streamsize>(sz));
                H.resize(static_cast<size_t>(in.gcount()) / sizeof(float));
                n = static_cast<int32_t>(H.size() / 768u);
            }
        }
    }
    const bool real = n > 0;
    const int32_t B = real ? (n < 240 ? n : 240) : 0;

    // A second, deliberately varied stream.
    //
    // The real held-out h[E] is the honest input, but an UNTRAINED head can be
    // nearly CONSTANT on it — in which case there is no churn to remove and the
    // filter has nothing to demonstrate. Both streams are reported, so the demo
    // never looks better than it is.
    //
    // Stream C is the same noise amplified. A random h[E] lands near the centre
    // of the seeded projection, so the logits barely separate and the head
    // reports LOW confidence — which the filter correctly refuses to act on.
    // Scaling the vector up widens the logit spread, so the same untrained head
    // becomes CONFIDENT while still changing its mind every step. That is the
    // exact situation the filter exists for.
    std::vector<float> synth(static_cast<size_t>(240) * 768u, 0.0f);
    {
        uint64_t s = 0xC0FFEEull;
        for (size_t i = 0; i < synth.size(); ++i) {
            s = s * 6364136223846793005ull + 1442695040888963407ull;
            synth[i] = static_cast<float>((s >> 40) & 0xFFFFu) / 65536.0f - 0.5f;
        }
    }
    std::vector<float> loud(synth.size(), 0.0f);
    for (size_t i = 0; i < synth.size(); ++i) loud[i] = synth[i] * 8.0f;

    std::printf("stream A: %d step(s) from %s\n", B,
                real ? kFixture : "(fixture not found — skipped)");
    std::printf("stream B: 240 step(s) of synthetic hidden states\n");
    std::printf("stream C: the same 240 step(s), scaled x8 (confident but churning)\n\n");

    // --- the two filter configurations --------------------------------------
    // min_confidence comes from demo_filter_config() so that the number the
    // filter commits behind is the DECISION-1 derived value in ONE place.
    StreamingConfig cc = demo_filter_config();
    cc.mode          = StreamMode::Confirm;
    cc.confirm_steps = 3;

    StreamingConfig cw = demo_filter_config();
    cw.mode          = StreamMode::Window;
    cw.window        = 5;
    cw.confirm_steps = 3;   // the vote quorum

    // The Scalar kernel makes the batch path bit-identical to the per-row path
    // (§36), so the numbers below do not depend on which path produced them.
    head.set_batch_kernel(BatchKernel::Scalar);

    // How often the RAW head output flips — the churn the filter exists to
    // remove. Counted from the events, not assumed.
    auto raw_churn = [](const std::vector<StreamEvent>& ev) {
        int64_t c = 0;
        for (size_t i = 1; i < ev.size(); ++i)
            if (ev[i].observed_action != ev[i - 1].observed_action) ++c;
        return c;
    };
    auto committed_churn = [](const std::vector<StreamEvent>& ev) {
        int64_t c = 0;
        for (const StreamEvent& e : ev) if (e.changed) ++c;
        return c;
    };

    auto report = [&](const char* label, const std::vector<StreamEvent>& ev,
                      const StreamingDecision& f) {
        std::printf("--- %s ---\n", label);
        std::printf("  %s\n", f.config().to_json().c_str());
        const int64_t raw = raw_churn(ev);
        const int64_t com = committed_churn(ev);
        std::printf("  raw head output flipped %lld time(s); the committed action "
                    "changed %lld time(s)\n",
                    static_cast<long long>(raw), static_cast<long long>(com));
        if (raw == 0)
            std::printf("  churn reduction: n/a — the head never changed its mind\n");
        else if (com == 0)
            std::printf("  churn reduction: the head flipped %lld time(s) and the filter "
                        "committed NOTHING\n", static_cast<long long>(raw));
        else
            std::printf("  churn reduction: %.1fx\n",
                        static_cast<double>(raw) / static_cast<double>(com));
        std::printf("  commits %lld, switches %lld, releases %lld, weak steps %lld\n",
                    static_cast<long long>(f.commits()),
                    static_cast<long long>(f.changes()),
                    static_cast<long long>(f.releases()),
                    static_cast<long long>(f.weak()));

        // How many DISTINCT actions the raw head produced. A head that emits
        // the same action for every row is CONSTANT, and no filter can make it
        // informative — this number is the difference between "the filter
        // stabilised a signal" and "there was no signal to stabilise". It is
        // the acceptance test for head fitting, so it is printed, not assumed.
        {
            bool seen[static_cast<int>(DecisionAction::COUNT)] = {false};
            int distinct = 0;
            for (const StreamEvent& e : ev) {
                if (!e.valid) continue;
                const int a = static_cast<int>(e.observed_action);
                if (a >= 0 && a < static_cast<int>(DecisionAction::COUNT) && !seen[a]) {
                    seen[a] = true;
                    ++distinct;
                }
            }
            std::printf("  the raw head emitted %d distinct action(s) over %zu step(s)\n",
                        distinct, ev.size());
        }

        int shown = 0;
        int64_t total = 0;
        for (const StreamEvent& e : ev) if (e.changed) ++total;
        for (const StreamEvent& e : ev) {
            if (!e.changed) continue;
            if (shown >= 8) break;
            std::printf("    step %3lld  %-8s -> %-8s  (evidence %d)\n",
                        static_cast<long long>(e.step),
                        decision_action_name(e.previous),
                        decision_action_name(e.action),
                        e.strength);
            ++shown;
        }
        if (total > shown)
            std::printf("    ... and %lld more\n", static_cast<long long>(total - shown));
        if (!ev.empty())
            std::printf("  last event: %s\n", ev.back().to_json().c_str());
        std::printf("  final: %s\n", f.to_json().c_str());
        std::printf("\n");
    };

    auto run = [&](const char* label, const std::vector<float>* data, int32_t count) {
        std::printf("### %s\n", label);
        if (data == nullptr || count <= 0) {
            std::printf("  skipped: no data\n\n");
            return;
        }
        StreamingDecision fc, fw;
        fc.set_config(cc);
        fw.set_config(cw);
        fc.init(head);
        fw.init(head);

        std::vector<StreamEvent> ec, ew;
        BatchStats stats;
        if (!fc.push_batch(data->data(), count, ec, &stats)) {
            std::printf("  push_batch failed: %s\n\n", fc.error().c_str());
            return;
        }
        if (!fw.push_batch(data->data(), count, ew, nullptr)) {
            std::printf("  push_batch failed: %s\n\n", fw.error().c_str());
            return;
        }
        report("Confirm mode: N consecutive agreeing observations", ec, fc);
        report("Window mode: a vote over a sliding window", ew, fw);
    };

    run("stream A — the real held-out h[E]", real ? &H : nullptr, B);
    run("stream B — synthetic hidden states", &synth, 240);
    run("stream C — synthetic, scaled x8 (confident but churning)", &loud, 240);

    std::printf("What this shows: the same head, the same hidden states, and a\n");
    std::printf("committed action that moves far less often than the raw output.\n");
    if (fitted)
        std::printf("The head is a FITTED projection (see docs/CALIBRATION.md), so the\n"
                    "actions above are real outputs — but they imitate a rule, not P&L.\n");
    else
        std::printf("What it does NOT show: any judgement. The head is UNTRAINED, so\n"
                    "the ACTIONS above are placeholders. The FILTER is real.\n");
    std::printf("peak RSS: %.2f MB\n",
                static_cast<double>(platform::peak_rss_bytes()) / 1048576.0);
    return 0;
}

// ===========================================================================
// Demo: the feedback hook (Milestone 10, Phase 2.3)
//
// THE CLOSED LOOP, on real data. The 369 held-out rows the trading head never
// saw are streamed through the filter, every prediction is recorded, and then
// every prediction is SCORED against the label that actually followed it.
//
// This is the first time anything in this tree has measured a head against what
// happened. The number it prints is not a claim about profitability — the
// labels are a documented rule, not P&L — it is the record's own accuracy
// report about the head, which is what makes re-fitting possible at all.
// ===========================================================================
int cmd_demo_feedback() {
    DecisionHead head;
    if (!head.init(768, 4242u)) {
        std::printf("decision head init failed: %s\n", head.error().c_str());
        return 1;
    }

    std::printf("=== FEEDBACK HOOK: prediction -> outcome -> record ===\n\n");
    const bool fitted = load_decision_head(head);

    // --- the real held-out tape ---------------------------------------------
    std::vector<float> H;
    int32_t n = 0;
    const char* kHidden = "tests/fixtures/head_calibration/trading/hidden.f32";
    {
        std::ifstream in(kHidden, std::ios::binary | std::ios::ate);
        if (in) {
            const std::streamoff sz = in.tellg();
            if (sz > 0 && (sz % 4) == 0) {
                H.resize(static_cast<size_t>(sz) / sizeof(float));
                in.seekg(0, std::ios::beg);
                in.read(reinterpret_cast<char*>(H.data()), static_cast<std::streamsize>(sz));
                H.resize(static_cast<size_t>(in.gcount()) / sizeof(float));
                n = static_cast<int32_t>(H.size() / 768u);
            }
        }
    }
    if (n <= 0) {
        std::printf("the held-out fixture is absent (%s)\n", kHidden);
        std::printf("run tools/train_heads.py, or see docs/FEEDBACK.md.\n");
        return 1;
    }

    // --- the realised labels -------------------------------------------------
    // Same file the offline trainer used, so the score is against the SAME
    // definition the head was fitted to imitate. Anything else would be
    // measuring the head against a different question.
    std::vector<DecisionAction> realised;
    {
        std::ifstream lf("tests/fixtures/head_calibration/trading/labels.tsv");
        std::string line;
        int col = -1;
        if (std::getline(lf, line)) {
            size_t pos = 0;
            int c = 0;
            while (true) {
                const size_t tab = line.find('\t', pos);
                if (line.substr(pos, tab == std::string::npos ? std::string::npos : tab - pos)
                    == "DecisionAction") { col = c; break; }
                if (tab == std::string::npos) break;
                pos = tab + 1; ++c;
            }
        }
        while (col >= 0 && std::getline(lf, line)) {
            while (!line.empty() && (line.back() == '\r' || line.back() == '\n'))
                line.pop_back();
            if (line.empty()) continue;
            std::vector<std::string> cells;
            size_t pos = 0;
            while (true) {
                const size_t tab = line.find('\t', pos);
                cells.push_back(line.substr(
                    pos, tab == std::string::npos ? std::string::npos : tab - pos));
                if (tab == std::string::npos) break;
                pos = tab + 1;
            }
            if (static_cast<int>(cells.size()) <= col) break;
            DecisionAction a = DecisionAction::ABSTAIN;
            bool ok = false;
            for (int32_t i = 0; i < static_cast<int32_t>(DecisionAction::COUNT); ++i) {
                const DecisionAction cand = static_cast<DecisionAction>(i);
                if (cells[static_cast<size_t>(col)] == decision_action_name(cand)) {
                    a = cand; ok = true; break;
                }
            }
            if (!ok) break;
            realised.push_back(a);
        }
    }
    if (realised.size() != static_cast<size_t>(n)) {
        std::printf("labels (%zu) do not match the hidden rows (%d) — refusing to "
                    "score a misaligned tape\n", realised.size(), n);
        return 1;
    }

    std::printf("tape   : %d real held-out bar(s) from %s\n", n, kHidden);
    std::printf("labels : %zu realised DecisionAction(s), the same teacher the head "
                "was fitted to imitate\n\n", realised.size());

    head.set_batch_kernel(BatchKernel::Scalar);   // §36: bit-identical to per-row

    struct Run {
        const char*  label;
        StreamMode   mode;
    };
    const Run runs[2] = {
        {"Confirm mode (3 consecutive agreeing)", StreamMode::Confirm},
        {"Window mode (vote over 5)",             StreamMode::Window},
    };

    FeedbackHook hook;
    const std::string key = "trading.AAPL.1d";

    for (const Run& r : runs) {
        StreamingDecision f;
        if (!f.init(head)) { std::printf("filter init failed\n"); return 1; }
        StreamingConfig c;
        c.mode = r.mode;
        c.confirm_steps = 3;
        c.window = 5;
        // DECISION 1: the bar is DERIVED from the calibration curve, not chosen.
        // See demo_filter_config() and docs/CALIBRATION.md §4.4.
        c.min_confidence = kDerivedMinConfidence;
        f.set_config(c);

        std::vector<StreamEvent> ev;
        if (!f.push_batch(H.data(), n, ev, nullptr)) {
            std::printf("push_batch failed: %s\n", f.error().c_str());
            return 1;
        }

        hook.clear();
        hook.observe_stream(ev, key);
        for (size_t i = 0; i < hook.journal().entries().size(); ++i)
            hook.resolve(hook.journal().entries()[i].id, realised[i], 0.0f, 1);

        const JournalStats s = hook.stats();
        std::printf("--- %s ---\n", r.label);
        std::printf("  %s\n", f.config().to_json().c_str());
        std::printf("  recorded %lld prediction(s); %lld resolved; %lld with a verdict\n",
                    static_cast<long long>(s.predictions),
                    static_cast<long long>(s.predictions - s.unresolved),
                    static_cast<long long>(s.verdicts));
        std::printf("  HEAD   hit_rate %.4f  (%lld/%lld)\n", s.hit_rate(),
                    static_cast<long long>(s.hits), static_cast<long long>(s.verdicts));
        std::printf("  FILTER hit_rate %.4f  (%lld/%lld) on the %lld step(s) it held a "
                    "position\n", s.filter_hit_rate(),
                    static_cast<long long>(s.filter_hits),
                    static_cast<long long>(s.filter_verdicts),
                    static_cast<long long>(s.committed_rows));
        std::printf("  commitment rate %.4f of steps\n", s.commitment_rate());
        std::printf("  mean confidence when right %.4f, when wrong %.4f, gap %.4f\n",
                    s.mean_conf_hit, s.mean_conf_miss, s.confidence_gap);
        std::printf("  ECE %.4f   Brier %.4f\n", s.ece, s.brier);

        // The confusion matrix, printed as the action names rather than indices.
        if (!s.confusion.empty()) {
            std::printf("  confusion (rows = predicted, cols = realised, non-zero only):\n");
            for (size_t a = 0; a < s.confusion.size(); ++a) {
                for (size_t b = 0; b < s.confusion[a].size(); ++b) {
                    if (s.confusion[a][b] == 0) continue;
                    std::printf("    %-8s -> %-8s %lld\n",
                                decision_action_name(static_cast<DecisionAction>(a)),
                                decision_action_name(static_cast<DecisionAction>(b)),
                                static_cast<long long>(s.confusion[a][b]));
                }
            }
        }

        // The event-log identity, extended to the record: the rows the filter
        // reported as changes are exactly the rows the journal marked.
        int64_t changed_rows = 0;
        for (const JournalEntry& e : hook.journal().entries()) if (e.changed) ++changed_rows;
        const int64_t from_filter = f.commits() + f.changes() + f.releases();
        std::printf("  identity: %lld changed row(s) == %lld filter commit/switch/"
                    "release(s)  %s\n", static_cast<long long>(changed_rows),
                    static_cast<long long>(from_filter),
                    changed_rows == from_filter ? "OK" : "MISMATCH");
        std::printf("\n");
    }

    // --- persist -------------------------------------------------------------
    // Both stores, in one file, so they cannot be restored out of step.
    const std::string bin_path = "state/feedback_journal.bin";
    const std::string tsv_path = "state/feedback_journal.tsv";
    if (hook.save(bin_path))
        std::printf("journal saved: %s (%zu row(s))\n", bin_path.c_str(),
                    hook.journal().size());
    else
        std::printf("journal NOT saved: %s\n", hook.error().c_str());
    {
        std::ofstream out(tsv_path, std::ios::binary | std::ios::trunc);
        if (out) {
            out << hook.journal().to_tsv();
            std::printf("training rows: %s — the rows an offline re-fit would consume\n",
                        tsv_path.c_str());
        }
    }

    std::printf("\nWhat this shows: a head's prediction, recorded, then scored against\n");
    std::printf("the outcome that actually followed. That is the loop closing.\n");
    if (!fitted)
        std::printf("What it does NOT show: a good head. No fitted projection was\n"
                    "found, so the hit rate above is the SEEDED placeholder's.\n");
    std::printf("The labels are a rule, not P&L: a high hit rate here would mean the\n");
    std::printf("head imitates the rule well, and says nothing about profitability.\n");
    std::printf("peak RSS: %.2f MB\n",
                static_cast<double>(platform::peak_rss_bytes()) / 1048576.0);
    return 0;
}

// ===========================================================================
// Demo: audio event layer (features #3, #15, #63, #70)
// ===========================================================================
int cmd_demo_audio(const std::string& path) {
    PcmAudio audio;
    if (!PcmAudio::load_wav(path, audio)) {
        std::printf("cannot load WAV: %s (16-bit mono PCM expected)\n",
                    path.c_str());
        return 1;
    }
    std::printf("loaded %s: %.2f s @ %d Hz\n", path.c_str(),
                audio.samples.size() / static_cast<double>(audio.sample_rate),
                audio.sample_rate);

    SoundEventDetector det;
    const SoundEventResult ev = det.detect(audio);
    std::printf("sound event : %s (confidence %.2f)\n",
                sound_event_name(ev.event), ev.confidence);

    float conf = 0.0f;
    const AudioScene scene = AudioSceneClassifier::classify(audio, conf);
    std::printf("scene       : %s (%.2f)\n", audio_scene_name(scene), conf);

    WakeWordDetector ww;
    float score = 0.0f;
    const bool wake = ww.scan(audio, score);
    std::printf("wake word   : %s (best %.2f)\n", wake ? "DETECTED" : "no",
                score);

    std::vector<float> frames;
    int32_t bands = 0;
    if (compute_frame_features(audio, 80, 24, frames, bands)) {
        AudioAnomalyDetector ad;
        bool anomaly = false;
        for (size_t f = 0; f < frames.size() / static_cast<size_t>(bands); ++f)
            anomaly = ad.push_frame(frames.data() + f * bands, bands) || anomaly;
        std::printf("anomalies   : %s (max deviation %.2f sigma)\n",
                    anomaly ? "FOUND" : "none", ad.last_deviation());
    }
    std::printf("peak RSS: %.2f MB\n",
                static_cast<double>(platform::peak_rss_bytes()) / 1048576.0);
    return 0;
}

// ===========================================================================
// enroll-audio — the OWNER-VOICE dataset builder (DECISION 2)
//
// WHY THIS EXISTS INSTEAD OF A DOWNLOAD. Two of the three audio label sets
// (audio.wake, audio.speaker) are fitted on a voice, and the useful voice is the
// owner's: a wake head fitted on 3,000 strangers from a 2017 corpus is fitted on
// the wrong distribution, and a speaker head can only "know" a speaker it has
// heard. Both are also licence-free by construction.
//
// WHY IT TAKES FILES AND NOT A MICROPHONE. There is no capture code in this tree
// and adding WASAPI would be a platform dependency this project has deliberately
// avoided (wsl.exe/cmd.exe are blacklisted; the build is plain MSVC + CMake).
// More importantly, a file-based enrolment is REPRODUCIBLE: the owner records
// with any tool, the exact WAVs are named in the manifest, and the fit can be
// re-run byte-identically. A microphone path would make the dataset a side
// effect of one afternoon's room noise.
//
// THE CONTRACT. <dir> contains a flat set of WAVs whose FILENAME carries the
// label, in the form  <label>_<anything>.wav  (e.g. yes_01.wav, known_a.wav).
// Files that do not parse are REPORTED and skipped — never silently folded into
// a label, because a mislabelled enrolment sample is a permanently wrong head.
// ===========================================================================
int cmd_enroll_audio(const std::string& dir, const std::string& set_name) {
    namespace fs = std::filesystem;

    std::printf("=== ENROL AUDIO: building the '%s' dataset (owner voice) ===\n\n",
                set_name.c_str());

    std::error_code ec;
    if (!fs::is_directory(dir, ec)) {
        std::printf("not a directory: %s\n", dir.c_str());
        std::printf("record some WAVs (16-bit mono; any rate is resampled) named\n"
                    "  <label>_<n>.wav   e.g. yes_01.wav  no_01.wav\n"
                    "and put them in a folder, then re-run.\n");
        return 1;
    }

    std::vector<std::string> files;
    for (const auto& e : fs::directory_iterator(dir, ec)) {
        if (!e.is_regular_file()) continue;
        std::string p = e.path().string();
        std::string low = p;
        for (char& c : low)
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (low.size() > 4 && low.compare(low.size() - 4, 4, ".wav") == 0)
            files.push_back(p);
    }
    std::sort(files.begin(), files.end());   // deterministic manifest order
    if (files.empty()) {
        std::printf("no .wav files in %s\n", dir.c_str());
        return 1;
    }

    std::map<std::string, int> per_label;
    std::vector<std::string> rows;
    int skipped = 0;
    for (const std::string& p : files) {
        const std::string base = fs::path(p).filename().string();
        const size_t us = base.find('_');
        if (us == std::string::npos || us == 0) {
            std::printf("  SKIP  %-30s (no '<label>_' prefix)\n", base.c_str());
            ++skipped;
            continue;
        }
        const std::string label = base.substr(0, us);
        PcmAudio a;
        if (!PcmAudio::load_wav(p, a) || !a.valid()) {
            std::printf("  SKIP  %-30s (unreadable WAV)\n", base.c_str());
            ++skipped;
            continue;
        }
        ++per_label[label];
        char buf[320];
        std::snprintf(buf, sizeof(buf), "%s\t%s\towner\t%d\t%zu", base.c_str(),
                      label.c_str(), a.sample_rate, a.samples.size());
        rows.push_back(buf);
    }

    if (rows.empty()) {
        std::printf("\nno usable clips — nothing written.\n");
        return 1;
    }

    std::printf("\n  clips by label:\n");
    for (const auto& kv : per_label)
        std::printf("    %-12s %d\n", kv.first.c_str(), kv.second);
    std::printf("  skipped %d\n", skipped);

    // WARN, but do not fail: a single-sample class cannot be split, so the
    // fitter will report it unfitted. Saying so here saves a confusing run.
    for (const auto& kv : per_label) {
        if (kv.second < 2)
            std::printf("  WARNING: label '%s' has ONE clip — it cannot be split "
                        "and will fit as UNFITTED\n", kv.first.c_str());
    }

    const std::string mpath = (fs::path(dir) / "labels.tsv").string();
    std::ofstream mf(mpath, std::ios::binary);
    if (!mf) {
        std::printf("cannot write %s\n", mpath.c_str());
        return 1;
    }
    mf << "file\t" << set_name << "\tgroup\tsample_rate\tn_samples\n";
    for (const std::string& r : rows) mf << r << "\n";
    mf.close();
    std::printf("\n  wrote %s (%zu row(s))\n", mpath.c_str(), rows.size());
    std::printf("  every clip is reported at its LOADED rate; the loader resamples\n"
                "  to 16000 and says so, so a 48 kHz recording is not silently "
                "mis-melled.\n");
    return 0;
}

// ===========================================================================
// Demo: vision tasks (features #2, #54, #61, #66, #69)
// ===========================================================================
int cmd_demo_vision() {
    // Synthetic 96x96 frame: bright object top-right.
    Image img;
    img.width = 96;
    img.height = 96;
    img.rgb.assign(96ull * 96 * 3, 0);
    for (int y = 8; y < 30; ++y)
        for (int x = 62; x < 88; ++x) {
            uint8_t* px = img.rgb.data() + (y * 96 + x) * 3;
            px[0] = px[1] = px[2] = 255;
        }

    // 1) Pointer grounding.
    PointerPerception pp;
    Region region;
    std::string token;
    if (pp.ground("what is this?", region, token)) {
        std::printf("pointer query grounded -> %s\n", token.c_str());
        const Image crop = PointerPerception::crop(img, region);
        std::printf("  crop %dx%d extracted for the encoder\n",
                    crop.width, crop.height);
    }

    // 2) Spatial map.
    const SpatialContextMapper::Map m = SpatialContextMapper::build(img);
    std::printf("spatial map : %s -> %s\n", token.c_str(),
                SpatialContextMapper::describe(m).c_str());

    // 3) Motion tracking across a synthetic "video" of 3 frames.
    MotionTracker mt;
    Image f2 = img;
    for (int y = 8; y < 30; ++y)
        for (int x = 62; x < 88; ++x) {
            uint8_t* px = f2.rgb.data() + (y * 96 + x) * 3;
            px[0] = px[1] = px[2] = 10;
        }
    mt.update(img, 0);
    const TrackPoint p = mt.update(f2, 33);
    std::printf("motion      : centroid (%.2f, %.2f) energy %.2f\n",
                p.cx, p.cy, p.energy);
    float cx = 0.0f, cy = 0.0f;
    if (mt.predict(100.0f, cx, cy))
        std::printf("prediction  : (%.2f, %.2f) in 100 ms\n", cx, cy);

    // 4) Gesture: lateral direction reversals.
    GestureRecognizer gr;
    Gesture g = Gesture::None;
    for (int i = 0; i < 14; ++i)
        g = gr.update((i % 2 == 0) ? 0.2f : 0.8f, 0.5f, 0.5f);
    std::printf("gesture     : %s\n", gesture_name(g));

    // 5) Dynamic resolution.
    std::printf("resolution  : fast=%s(%d) balanced=%s(%d) deep=%s(%d)\n",
                DynamicResolution::mode_name(64), DynamicResolution::pick_input_size(0.1f, false),
                DynamicResolution::mode_name(96), DynamicResolution::pick_input_size(0.5f, false),
                DynamicResolution::mode_name(128), DynamicResolution::pick_input_size(0.9f, false));
    std::printf("peak RSS: %.2f MB\n",
                static_cast<double>(platform::peak_rss_bytes()) / 1048576.0);
    return 0;
}

// ===========================================================================
// Demo: Dream-State consolidation (capability #1, spec #38)
// ===========================================================================
int cmd_dream() {
    SelfImprovement imp;
    imp.load("./state/improve.bin");

    FlashSkillPool pool;
    pool.set_store_path("./state/skills.bin");
    pool.load();

    std::printf("before dream: %zu traces, %zu skills\n",
                imp.size(), pool.size());

    // Simulate some history if the state is empty (deterministic demo).
    if (imp.size() == 0) {
        for (int i = 0; i < 6; ++i)
            imp.record("what time is it", {"time"}, true, 12.0 + i);
        imp.record("deploy to mars", {"calc"}, false, 900.0);
    }

    imp.dream();
    const size_t retired = pool.retire_stale(platform::now_ms() / 1000.0);
    imp.save("./state/improve.bin");
    pool.save();

    std::printf("dream complete: %zu traces retained, %zu skills retired\n",
                imp.size(), retired);
    std::printf("peak RSS: %.2f MB\n",
                static_cast<double>(platform::peak_rss_bytes()) / 1048576.0);
    return 0;
}

// ===========================================================================
// Model-dependent commands
// ===========================================================================
struct Session {
    std::string model_path = "./models/omniseed.gguf";
    int32_t max_tokens = 160;
    float temperature = 0.0f;   // 0 => greedy
    int32_t top_k = 0;
    float repeat_penalty = 1.0f; // 1.0 = off
    uint64_t seed = 42;
    std::string prompt;
    std::string assistant_lora; // optional LoRA sidecar path
    std::string eval_file;      // ppl: corpus file (else --prompt text)
    int32_t    eval_tokens = 2048;  // ppl: token budget
    std::vector<std::string> stop_strings;   // --stop (repeatable)

    // System-1 decision head ("two heads, one brain"). Off by default so the
    // CLI behaves exactly as before unless asked otherwise.
    DecisionMode decision_mode      = DecisionMode::Off;
    float        decision_threshold = 0.85f;
    std::string  decision_head_path;   // fitted projection blob (optional)

    Tokenizer tok;
    std::unique_ptr<RwkvModel> model;
    std::unique_ptr<LoraAdapter> lora;

    bool load() {
        model = std::make_unique<RwkvModel>();
        if (!model->load(model_path)) {
            platform::log_error("model load failed: %s",
                                model->error().c_str());
            return false;
        }

        // Real vocab + special ids from the model's own GGUF metadata.
        bool vocab_ok = false;
        {
            GgufLoader gg;
            if (gg.open(model_path)) {
                vocab_ok = tok.load_from_gguf(gg);
                if (vocab_ok) {
                    tok.set_special_ids(
                        static_cast<int32_t>(gg.get_u64("omniseed.bos_token_id", -1)),
                        static_cast<int32_t>(gg.get_u64("omniseed.user_start_token_id", -1)),
                        static_cast<int32_t>(gg.get_u64("omniseed.user_end_token_id", -1)),
                        static_cast<int32_t>(gg.get_u64("omniseed.assistant_start_token_id", -1)),
                        static_cast<int32_t>(gg.get_u64("omniseed.assistant_end_token_id", -1)));
                    eos_id = static_cast<int32_t>(
                        gg.get_u64("omniseed.eos_token_id", Tokenizer::kEosId));
                }
            }
        }
        if (!vocab_ok) {
            platform::log_info("no GGUF vocab, using minimal byte tokenizer");
            if (!tok.build_minimal()) {
                platform::log_error("tokenizer build failed");
                return false;
            }
        }

        platform::log_info("model loaded: %d layers, %d embd, %d vocab, eos=%d",
                           model->config().n_layers, model->config().n_embd,
                           model->config().n_vocab, eos_id);

        // Assistant-behavior LoRA sidecar (optional; attached BEFORE first
        // forward so every generation command picks it up).
        if (!assistant_lora.empty()) {
            lora = std::make_unique<LoraAdapter>();
            if (lora->load(assistant_lora)
                && lora->layer_count() == model->config().n_layers
                && lora->n_embd() == model->config().n_embd) {
                model->set_lora(lora.get());
                platform::log_info(
                    "assistant LoRA attached: %s (rank %d, scaling %.2f)",
                    assistant_lora.c_str(), lora->rank(), lora->scaling());
            } else {
                platform::log_error(
                    "assistant LoRA load failed (%s) — continuing WITHOUT it",
                    lora->valid() ? "geometry mismatch vs base model"
                                  : lora->error().c_str());
                lora.reset();
            }
        }
        return true;
    }

    int32_t eos_id = Tokenizer::kEosId;
};

int cmd_logits(Session& s) {
    if (!s.load()) return 1;
    auto ids = s.tok.encode_chat(s.prompt);
    std::printf("prompt ids (%zu):", ids.size());
    for (size_t i = 0; i < ids.size() && i < 24; ++i)
        std::printf(" %d", ids[i]);
    std::printf("\n");

    RwkvState st;
    s.model->init_state(st);
    Tensor logits("logits", {s.model->config().n_vocab}, DType::F32);
    for (const int32_t id : ids) s.model->forward(id, st, logits);

    // top-8
    std::vector<int32_t> idx(static_cast<size_t>(s.model->config().n_vocab));
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = static_cast<int32_t>(i);
    std::partial_sort(idx.begin(), idx.begin() + 8, idx.end(),
                      [&](int32_t a, int32_t b) {
                          return logits.f32()[a] > logits.f32()[b];
                      });
    std::printf("top8:");
    for (int i = 0; i < 8; ++i)
        std::printf(" (%d, %.3f)", idx[static_cast<size_t>(i)],
                    logits.f32()[idx[static_cast<size_t>(i)]]);
    std::printf("\n");
    return 0;
}

int cmd_gen(Session& s) {
    if (!s.load()) return 1;

    auto ids = s.tok.encode_chat(s.prompt);
    RwkvState st;
    s.model->init_state(st);

    Tensor logits("logits", {s.model->config().n_vocab}, DType::F32);
    const auto t0 = platform::now_ms();
    for (const int32_t id : ids) s.model->forward(id, st, logits);
    const int32_t seed = ids.back();

    AgentLoop::Config cfg;
    cfg.allow_tools = false;
    cfg.temperature = s.temperature;
    cfg.top_k = s.top_k;
    cfg.repeat_penalty = s.repeat_penalty;
    cfg.seed = s.seed;
    cfg.stop_strings = s.stop_strings;
    static ToolRegistry dummy;
    static MemoryCrystals mem;
    static ComputeThrottle thr;
    static SelfImprovement imp;
    AgentLoop loop(*s.model, s.tok, dummy, mem, thr, imp, cfg);

    const std::string out =
        loop.generate(st, seed, s.max_tokens, {s.eos_id}, nullptr);
    const double ms = platform::now_ms() - t0;

    std::printf("%s\n", out.c_str());
    std::printf("\n[%.0f ms, %.1f tok/s, peak %.1f MB]\n",
                ms, static_cast<double>(s.max_tokens) / (ms / 1000.0),
                static_cast<double>(platform::peak_rss_bytes()) / 1048576.0);
    return 0;
}

// Bring the System-1 decision head up and report its provenance honestly.
// A head with no fitted projection is a PLACEHOLDER: say so loudly rather
// than letting the operator read its routing as a judgement.
static void announce_decision_mode(AgentLoop& loop, const AgentLoop::Config& cfg) {
    if (cfg.decision_mode == DecisionMode::Off) return;

    if (!loop.ensure_decision_head()) {
        platform::log_warn("--mode %s requested but the decision head is "
                           "unavailable; falling back to System-2 only",
                           decision_mode_name(cfg.decision_mode));
        return;
    }
    const DecisionHead& dh = loop.decision_head();
    platform::log_info("decision head: mode=%s E=%d actions=%d threshold=%.2f %s",
                       decision_mode_name(cfg.decision_mode), dh.hidden_size(),
                       dh.action_count(), dh.threshold(),
                       dh.trained() ? "fitted" : "UNTRAINED");
    if (!dh.trained()) {
        platform::log_warn("decision head has no fitted projection (%s): its "
                           "decisions are well-formed but MEANINGLESS — pass "
                           "--decision-head to supply real weights",
                           dh.provenance().c_str());
    }
}

int cmd_ask(Session& s) {
    if (!s.load()) return 1;

    ToolRegistry tools;
    for (const char* t : {"calc", "echo", "time"}) tools.add_builtin(t);
    MemoryCrystals mem;
    ComputeThrottle thr;
    SelfImprovement imp;
    FlashSkillPool skills;
    ThinkingMode think;
    UserFeedbackLoop feedback;
    AgentLoop::Config cfg;
    cfg.max_new_tokens = s.max_tokens;
    cfg.stop_strings = s.stop_strings;
    cfg.decision_mode      = s.decision_mode;
    cfg.decision_threshold = s.decision_threshold;
    cfg.decision_head_path = s.decision_head_path;

    AgentLoop loop(*s.model, s.tok, tools, mem, thr, imp, cfg);
    announce_decision_mode(loop, cfg);
    const AgentLoop::Result r = loop.run(s.prompt);

    std::printf("%s\n", r.reply.c_str());
    for (const std::string& t : r.tool_trace) {
        std::printf("[tool] %s\n", t.c_str());
    }
    if (r.decision_checked) {
        std::printf("[decision] %s%s\n", r.decision.to_json().c_str(),
                    r.decision_fast_path ? "  <- System-1 answered" : "");
    }
    std::printf("[%.0f ms, peak %.1f MB]\n", r.ms,
                static_cast<double>(r.peak_rss) / 1048576.0);
    return 0;
}

// Teacher-forced cross-entropy over a corpus: the honest apples-to-apples
// quality metric for comparing checkpoints/quantizations.
int cmd_ppl(Session& s) {
    if (!s.load()) return 1;

    std::vector<int32_t> ids;
    if (!s.eval_file.empty()) {
        std::ifstream f(s.eval_file, std::ios::binary);
        if (!f) {
            std::printf("cannot open eval file: %s\n", s.eval_file.c_str());
            return 1;
        }
        std::stringstream ss;
        ss << f.rdbuf();
        ids = s.tok.encode(ss.str(), /*add_bos=*/true);
    } else {
        const std::string text = s.prompt.empty()
            ? "The quick brown fox jumps over the lazy dog. "
              "Language models predict the next token from context. "
              "Perplexity measures how surprised the model is. "
              "Lower is better. "
            : s.prompt;
        ids = s.tok.encode(text, /*add_bos=*/true);
    }
    if (s.eval_tokens > 0 &&
        static_cast<size_t>(s.eval_tokens) + 1 < ids.size()) {
        ids.resize(static_cast<size_t>(s.eval_tokens) + 1);
    }
    if (ids.size() < 8) {
        std::printf("eval text too short (%zu ids)\n", ids.size());
        return 1;
    }

    RwkvState st;
    s.model->init_state(st);
    Tensor logits("logits", {s.model->config().n_vocab}, DType::F32);

    double nll = 0.0;
    int64_t n_pred = 0;
    int32_t prev = ids[0];
    const auto t0 = platform::now_ms();
    for (size_t i = 1; i < ids.size(); ++i) {
        s.model->forward(prev, st, logits);
        const int32_t tgt = ids[i];
        const float* l = logits.f32();
        float mx = l[0];
        for (int32_t v = 1; v < s.model->config().n_vocab; ++v)
            if (l[v] > mx) mx = l[v];
        double sum = 0.0;
        for (int32_t v = 0; v < s.model->config().n_vocab; ++v)
            sum += std::exp(static_cast<double>(l[v] - mx));
        nll -= std::log(std::exp(static_cast<double>(l[tgt] - mx)) / sum);
        ++n_pred;
        prev = tgt;
    }
    const double ms = platform::now_ms() - t0;
    const double ppl = std::exp(nll / static_cast<double>(n_pred));
    std::printf("tokens=%lld nll=%.4f bits/token=%.4f PPL=%.2f (%.0f ms, %.1f tok/s, peak %.1f MB)\n",
                static_cast<long long>(n_pred),
                nll / static_cast<double>(n_pred),
                nll / static_cast<double>(n_pred) / std::log(2.0),
                ppl, ms,
                static_cast<double>(n_pred) / (ms / 1000.0),
                static_cast<double>(platform::peak_rss_bytes()) / 1048576.0);
    return 0;
}

int cmd_bench(Session& s) {
    if (!s.load()) return 1;

    RwkvState st;
    s.model->init_state(st);
    Tensor logits("logits", {s.model->config().n_vocab}, DType::F32);

    const int32_t tok_id = Tokenizer::kBosId;
    // warmup
    for (int i = 0; i < 8; ++i) s.model->forward(tok_id, st, logits);

    const auto t0 = platform::now_ms();
    constexpr int32_t kIters = 128;
    for (int i = 0; i < kIters; ++i) s.model->forward(tok_id, st, logits);
    const double ms = platform::now_ms() - t0;

    std::printf("bench: %d tokens in %.0f ms -> %.1f tok/s\n", kIters, ms,
                static_cast<double>(kIters) / (ms / 1000.0));
    std::printf("peak RSS: %.2f MB (budget 300 MB)\n",
                static_cast<double>(platform::peak_rss_bytes()) / 1048576.0);
    return platform::peak_rss_bytes() <= 300ull * 1024ull * 1024ull ? 0 : 2;
}

int cmd_chat(Session& s) {
    if (!s.load()) return 1;

    ToolRegistry tools;
    for (const char* t : {"calc", "echo", "time"}) tools.add_builtin(t);
    MemoryCrystals mem;
    ComputeThrottle thr;
    SelfImprovement imp;

    AgentLoop::Config cfg;
    cfg.max_new_tokens = s.max_tokens;
    cfg.temperature = s.temperature;
    cfg.top_k = s.top_k;
    cfg.repeat_penalty = s.repeat_penalty;
    cfg.stop_strings = s.stop_strings;
    cfg.decision_mode      = s.decision_mode;
    cfg.decision_threshold = s.decision_threshold;
    cfg.decision_head_path = s.decision_head_path;
    // Streaming: print each piece as it is decoded, then finish the line
    // after generation completes (Result.reply holds the same text).
    cfg.on_token = [](const std::string& piece) {
        std::fputs(piece.c_str(), stdout);
        std::fflush(stdout);
    };
    AgentLoop loop(*s.model, s.tok, tools, mem, thr, imp, cfg);
    announce_decision_mode(loop, cfg);

    RwkvState st;
    s.model->init_state(st);

    std::printf("OmniSeed chat — type /quit to exit\n");
    std::string line;
    while (true) {
        std::printf("\n> ");
        if (!std::getline(std::cin, line)) break;
        if (line == "/quit" || line == "/exit") break;
        if (line.empty()) continue;

        const AgentLoop::Result r = loop.run(line);
        std::printf("\n");
        for (const std::string& t : r.tool_trace)
            std::printf("[tool] %s\n", t.c_str());
        // Show what System-1 said, and whether it answered by itself.
        if (r.decision_checked)
            std::printf("[decision] %s%s\n", r.decision.to_json().c_str(),
                        r.decision_fast_path ? "  <- System-1 answered" : "");
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    platform::enable_utf8_console();   // Windows codepage 65001; no-op elsewhere
    if (argc < 2) {
        print_usage();
        return 0;
    }

    const std::string cmd = argv[1];
    Session s;

    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) s.model_path = argv[++i];
        else if (a == "--prompt" && i + 1 < argc) s.prompt = argv[++i];
        else if (a == "--max-tokens" && i + 1 < argc)
            s.max_tokens = std::atoi(argv[++i]);
        else if (a == "--temperature" && i + 1 < argc)
            s.temperature = static_cast<float>(std::atof(argv[++i]));
        else if (a == "--top-k" && i + 1 < argc)
            s.top_k = std::atoi(argv[++i]);
        else if (a == "--repeat-penalty" && i + 1 < argc)
            s.repeat_penalty = static_cast<float>(std::atof(argv[++i]));
        else if (a == "--stop" && i + 1 < argc)
            s.stop_strings.push_back(argv[++i]);
        else if (a == "--stop-defaults") {
            for (const char* d : kAssistantStopDefaults)
                s.stop_strings.push_back(d);
        }
        else if (a == "--assistant-lora" && i + 1 < argc)
            s.assistant_lora = argv[++i];
        else if (a == "--assistant-mode")
            s.assistant_lora = "models/assistant-lora.gguf";
        else if (a == "--seed" && i + 1 < argc)
            s.seed = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--mode" && i + 1 < argc) {
            const std::string m = argv[++i];
            if (m == "off") {
                s.decision_mode = DecisionMode::Off;
            } else if (m == "hybrid") {
                s.decision_mode = DecisionMode::Hybrid;
            } else if (m == "decision-only" || m == "decision_only") {
                s.decision_mode = DecisionMode::DecisionOnly;
            } else if (m == "text-only" || m == "text_only" || m == "textonly") {
                // The mandate's name for what this tree has always called "off":
                // generate text and do not consult the decision head at all.
                // Accepting the name costs nothing and stops a caller who read
                // the mandate from getting an error for typing the documented
                // spelling.
                s.decision_mode = DecisionMode::Off;
            } else {
                // Refuse rather than silently defaulting: a typo that quietly
                // turned the feature off would look like the feature failing.
                platform::log_error("--mode: unknown value '%s' "
                                    "(expected off | text-only | hybrid | decision-only)",
                                    m.c_str());
                return 2;
            }
        }
        else if (a == "--decision-threshold" && i + 1 < argc)
            s.decision_threshold = static_cast<float>(std::atof(argv[++i]));
        else if (a == "--decision-head" && i + 1 < argc)
            s.decision_head_path = argv[++i];
        else if (a == "--eval-file" && i + 1 < argc) s.eval_file = argv[++i];
        else if (a == "--eval-tokens" && i + 1 < argc)
            s.eval_tokens = std::atoi(argv[++i]);
        else if (a == "--quiet") platform::set_quiet(true);
        else if (!s.prompt.empty()) { /* positional handled below */ }
    }

    // positional task for `ask`
    if (s.prompt.empty() && argc > 2 && cmd == "ask") {
        s.prompt = argv[argc - 1];
    }

    if (cmd == "info")          { cmd_info(); return 0; }
    if (cmd == "tools")         { cmd_tools(); return 0; }
    if (cmd == "selftest")      { return cmd_selftest(); }
    if (cmd == "demo-sensory")  { return cmd_demo_sensory(); }
    if (cmd == "demo-emotional"){ return cmd_demo_emotional(); }
    if (cmd == "demo-skills")   { return cmd_demo_skills(); }
    if (cmd == "demo-swarm")    { return cmd_demo_swarm(); }
    if (cmd == "demo-memory")   { return cmd_demo_memory(); }
    if (cmd == "demo-soul")     { return cmd_demo_soul(); }
    if (cmd == "demo-stream")   { return cmd_demo_stream(); }
    if (cmd == "demo-feedback") { return cmd_demo_feedback(); }
    if (cmd == "demo-vision")   { return cmd_demo_vision(); }
    if (cmd == "dream")         { return cmd_dream(); }
    if (cmd == "demo-audio")    {
        if (argc < 3) { std::printf("usage: omniseed demo-audio F.wav\n"); return 1; }
        return cmd_demo_audio(argv[2]);
    }
    if (cmd == "enroll-audio")  {
        if (argc < 3) {
            std::printf("usage: omniseed enroll-audio <dir> [label-set]\n"
                        "  <dir> must hold WAVs named <label>_<anything>.wav\n"
                        "  [label-set] defaults to audio.wake\n");
            return 1;
        }
        return cmd_enroll_audio(argv[2], argc >= 4 ? argv[3] : "audio.wake");
    }
    if (cmd == "gen")      { if (s.prompt.empty()) { print_usage(); return 1; }
                             return cmd_gen(s); }
    if (cmd == "logits")   { if (s.prompt.empty()) { print_usage(); return 1; }
                             return cmd_logits(s); }
    if (cmd == "ask")      { if (s.prompt.empty()) { print_usage(); return 1; }
                             return cmd_ask(s); }
    if (cmd == "bench")    { return cmd_bench(s); }
    if (cmd == "ppl")      { return cmd_ppl(s); }
    if (cmd == "chat")     { return cmd_chat(s); }

    print_usage();
    return 1;
}
