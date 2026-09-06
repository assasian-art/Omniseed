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
#include "omniseed/core/platform.h"
#include "omniseed/core/rwkv.h"
#include "omniseed/core/tokenizer.h"
#include "omniseed/memory/memory.h"
#include "omniseed/runtime/emotional.h"
#include "omniseed/runtime/sensory.h"
#include "omniseed/runtime/swarm.h"
#include "omniseed/runtime/token_bus.h"
#include "omniseed/vision/vision.h"
#include "omniseed/vision/vision_tasks.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>
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
        "  demo-audio F.wav            sound events / scene / wake word\n"
        "  demo-vision                 pointer grounding + spatial map\n"
        "  dream                       Dream-State consolidation pass\n"
        "\n"
        "options:\n"
        "  --model PATH     model GGUF path (default: ./models/omniseed.gguf)\n"
        "  --prompt TEXT    input text\n"
        "  --max-tokens N   generation budget (default 160)\n"
        "  --temperature T  sampling temperature, 0 = greedy (default 0)\n"
        "  --top-k K        keep K highest-probability tokens (0 = all)\n"
        "  --seed S         sampling seed, deterministic per seed\n"
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
    uint64_t seed = 42;
    std::string prompt;

    Tokenizer tok;
    std::unique_ptr<RwkvModel> model;

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
    cfg.seed = s.seed;
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

    AgentLoop loop(*s.model, s.tok, tools, mem, thr, imp, cfg);
    const AgentLoop::Result r = loop.run(s.prompt);

    std::printf("%s\n", r.reply.c_str());
    for (const std::string& t : r.tool_trace) {
        std::printf("[tool] %s\n", t.c_str());
    }
    std::printf("[%.0f ms, peak %.1f MB]\n", r.ms,
                static_cast<double>(r.peak_rss) / 1048576.0);
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
    AgentLoop loop(*s.model, s.tok, tools, mem, thr, imp, cfg);

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
        std::printf("%s\n", r.reply.c_str());
        for (const std::string& t : r.tool_trace)
            std::printf("[tool] %s\n", t.c_str());
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
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
        else if (a == "--seed" && i + 1 < argc)
            s.seed = std::strtoull(argv[++i], nullptr, 10);
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
    if (cmd == "demo-vision")   { return cmd_demo_vision(); }
    if (cmd == "dream")         { return cmd_dream(); }
    if (cmd == "demo-audio")    {
        if (argc < 3) { std::printf("usage: omniseed demo-audio F.wav\n"); return 1; }
        return cmd_demo_audio(argv[2]);
    }
    if (cmd == "gen")      { if (s.prompt.empty()) { print_usage(); return 1; }
                             return cmd_gen(s); }
    if (cmd == "logits")   { if (s.prompt.empty()) { print_usage(); return 1; }
                             return cmd_logits(s); }
    if (cmd == "ask")      { if (s.prompt.empty()) { print_usage(); return 1; }
                             return cmd_ask(s); }
    if (cmd == "bench")    { return cmd_bench(s); }
    if (cmd == "chat")     { return cmd_chat(s); }

    print_usage();
    return 1;
}
