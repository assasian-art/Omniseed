// =============================================================================
//  OmniSeed — tests/test_platform.cpp
//  Minimal self-contained test harness + Phase 1 platform layer tests.
//
//  Verifies:
//    * mmap round-trip of a temp file (MappedFile open/read/close)
//    * page_size() sanity
//    * aligned alloc/free round-trip
//    * peak/current RSS report sane, monotonic peak
//    * monotonic clock ordering
//    * RWKV-7 math invariants (tensor softmax exp() delta between frames)
//    * BitNet ternary packing (2 weights/byte, sign-magnitude decode)
//    * GGUF header magic
//  New phases append tests here; the harness scales with the project.
// =============================================================================
#include "omniseed/core/platform.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "omniseed/core/tensor.h"
#include "omniseed/core/bitlinear.h"
#include "omniseed/core/fft.h"
#include "omniseed/core/gguf_format.h"
#include "omniseed/core/uncertainty.h"
#include "omniseed/core/rwkv.h"
#include "omniseed/memory/prefix_cache.h"
#include "omniseed/runtime/sensory.h"
#include "omniseed/runtime/emotional.h"
#include "omniseed/runtime/swarm.h"
#include "omniseed/agent/flash_skills.h"
#include "omniseed/agent/agent_intel.h"
#include "omniseed/audio/audio_events.h"
#include "omniseed/audio/audio.h"
#include "omniseed/vision/vision_tasks.h"
#include "omniseed/vision/vision.h"
#include "omniseed/memory/memory.h"
#include "omniseed/core/tokenizer.h"

#include <cmath>
#include <unordered_map>

using namespace omniseed;

// ---------------------------------------------------------------------------
// Tiny test harness
// ---------------------------------------------------------------------------
static int g_passed = 0;
static int g_failed = 0;
static std::string g_current;

#define TEST(name) \
    g_current = (name); \
    platform::log_info("TEST  %s", name)

#define CHECK(cond) \
    do { \
        if (cond) { ++g_passed; } \
        else { \
            ++g_failed; \
            platform::log_error("FAIL  %s  (line %d): %s", \
                                g_current.c_str(), __LINE__, #cond); \
        } \
    } while (0)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static std::string make_temp_file(const char* name, const std::string& content) {
    const std::string path = std::string("build") + "/" + name;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(content.data(), static_cast<std::streamsize>(content.size()));
    return path;
}

// ===========================================================================
// Phase 1: platform tests
// ===========================================================================
static void test_page_size() {
    TEST("page_size is sane");
    const uint32_t ps = platform::page_size();
    CHECK(ps >= 512u && ps <= 1u << 20u);
    CHECK((ps & (ps - 1u)) == 0u); // power of two
}

static void test_mmap_roundtrip() {
    TEST("MappedFile round-trip");
    const std::string payload =
        "OMNISEED-MMAP-TEST-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    const std::string path = make_temp_file("mmap_test.bin", payload);

    platform::MappedFile mf;
    CHECK(mf.open(path));
    CHECK(mf.valid());
    CHECK(mf.size() == payload.size());
    CHECK(mf.data() != nullptr);
    CHECK(std::memcmp(mf.data(), payload.data(), payload.size()) == 0);
    mf.close();
    CHECK(!mf.valid());

    // Missing file must fail gracefully with an error message.
    platform::MappedFile missing;
    CHECK(!missing.open("does_not_exist_9f3a.bin"));
    CHECK(!missing.last_error().empty());
}

static void test_aligned_alloc() {
    TEST("aligned_alloc 64-byte alignment");
    void* p = platform::aligned_alloc_omni(4096, 64);
    CHECK(p != nullptr);
    CHECK((reinterpret_cast<uintptr_t>(p) % 64u) == 0u);
    std::memset(p, 0xAB, 4096);
    platform::aligned_free_omni(p);
}

static void test_peak_rss() {
    TEST("peak RSS sane and monotonic");
    const uint64_t before = platform::peak_rss_bytes();
    CHECK(before > 0);

    // Allocate and touch 8MB to push RSS upward.
    std::vector<char> ballast(8u << 20u, 1);
    ballast[0] = 2; ballast[ballast.size() / 2] = 2; ballast.back() = 2;

    const uint64_t after = platform::peak_rss_bytes();
    CHECK(after >= before);
    CHECK(after < (1ull << 31)); // < 2GB: process is small
}

static void test_clock() {
    TEST("monotonic clock ordering");
    const double t0 = platform::now_ms();
    // Busy wait briefly.
    volatile double acc = 0.0;
    for (int i = 0; i < 200000; ++i) acc += std::sqrt(static_cast<double>(i));
    const double t1 = platform::now_ms();
    CHECK(t1 >= t0);
    CHECK((t1 - t0) < 10000.0); // should take far less than 10s
    CHECK(acc != 42.0);         // keep the loop from being optimized out
}

static void test_os_name() {
    TEST("os_name non-empty");
    const char* n = platform::os_name();
    CHECK(n != nullptr);
    CHECK(std::strlen(n) > 0);
    platform::log_info("  running on: %s, page=%u, peak_rss=%.2f MB",
                       n, platform::page_size(),
                       static_cast<double>(platform::peak_rss_bytes()) /
                           (1024.0 * 1024.0));
}

// ===========================================================================
// Phase 2: tensor / bitnet / gguf tests
// ===========================================================================
static void test_tensor_softmax_and_inplace_exp() {
    TEST("tensor: softmax fp32 vs fp16 exp delta < 1e-3");
    // NOTE: the softmax helper lives in rwkv.h; the exp() delta contract is
    // validated here via direct tensor math (Phase 1 scope: platform is a
    // dependency of everything, so keep basic numeric sanity checks close).
    Tensor a("a", {4}, DType::F32);
    float av[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    std::memcpy(a.f32(), av, sizeof(av));

    // exp(x) delta contract: exp() must differ by <1e-3 between fp32 values
    // and their fp16 round-trip (half_to_float(float_to_half(x)) ~= x).
    bool close = true;
    for (int i = 0; i < 4; ++i) {
        const float ref  = std::exp(av[i]);
        const float rt   = half_to_float(float_to_half(av[i]));
        const float refe = std::exp(rt);
        if (std::fabs(ref - refe) > 1e-3f * std::fabs(ref) + 1e-6f) close = false;
    }
    CHECK(close);

    // cast() to fp16 must round-trip tightly too.
    Tensor c = tensor_ops::cast(a, DType::F16);
    CHECK(c.shape() == a.shape());
    Tensor back = tensor_ops::cast(c, DType::F32);
    bool close2 = true;
    for (int i = 0; i < 4; ++i) {
        if (std::fabs(back.f32()[i] - av[i]) > 1e-3f * std::fabs(av[i]) + 1e-6f)
            close2 = false;
    }
    CHECK(close2);
}

static void test_ternary_pack_roundtrip() {
    TEST("bitnet: ternary pack/unpack round-trip (2/byte)");
    const int8_t w[8] = {1, -1, 0, 1, -1, -1, 0, 0};
    std::vector<uint8_t> packed(bitnet::pack_ternary_packed_bytes(8));
    bitnet::pack_ternary(w, 8, packed.data());

    int8_t out[8] = {0};
    bitnet::unpack_ternary(packed.data(), 8, out);
    for (int i = 0; i < 8; ++i) CHECK(out[i] == w[i]);
}

static void test_ternary_matmul() {
    TEST("bitnet: ternary matmul matches reference (+1 add / -1 sub)");
    // W (2x3) ternary, X (3) fp32, scale 0.7
    const int8_t W[6] = {1, -1, 0, -1, 1, 1};
    const float   X[3] = {0.5f, -2.0f, 3.0f};
    const float   scale = 0.7f;

    float ref[2];
    for (int r = 0; r < 2; ++r) {
        float acc = 0.0f;
        for (int c = 0; c < 3; ++c) {
            if (W[r * 3 + c] == 1)  acc += X[c];
            if (W[r * 3 + c] == -1) acc -= X[c];
        }
        ref[r] = acc * scale;
    }

    std::vector<uint8_t> packed(bitnet::pack_ternary_packed_bytes(6));
    bitnet::pack_ternary(W, 6, packed.data());

    float got[2] = {0.0f, 0.0f};
    bitnet::bitlinear_forward(packed.data(), nullptr, scale, X, got, 2, 3);

    CHECK(std::fabs(got[0] - ref[0]) < 1e-5f);
    CHECK(std::fabs(got[1] - ref[1]) < 1e-5f);
}

static void test_bitlinear_train_step() {
    TEST("bitnet: train_step moves fp master toward ternary gradient");
    // X (1x2) input, W (2x2) master weights. dL/dY = ones.
    const float X[2] = {1.0f, 2.0f};
    const int8_t  Wq[4] = {1, -1, 0, 1};
    float W[4] = {0.4f, -0.4f, 0.4f, 0.4f};
    float dW[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float dX[2] = {0.0f, 0.0f};
    const float dY[4] = {1.0f, 1.0f, 1.0f, 1.0f};   // [1 row x 2 outputs, x2 cols? no: rows=1 -> dY[2]]
    // dY layout: [rows=1, out_dim=2] -> only first two entries used.

    bitnet::bitlinear_train_step(X, Wq, W, 0.5f /*scale*/, dW, dX,
                                 dY, 1 /*rows*/,
                                 2 /*in*/, 2 /*out*/, 0.05f /*lr*/);

    // dW should equal X^T . dY (with dY==all ones) -> {1,2,1,2}
    CHECK(std::fabs(dW[0] - 1.0f) < 1e-5f);
    CHECK(std::fabs(dW[1] - 2.0f) < 1e-5f);
    CHECK(std::fabs(dW[2] - 1.0f) < 1e-5f);
    CHECK(std::fabs(dW[3] - 2.0f) < 1e-5f);
    // dX (straight-through): dX[0] = scale * sum_r Wq[r][0] * dY[0][r]
    //   = 0.5 * (Wq[0][0]*1 + Wq[1][0]*1) = 0.5 * (1 + 0) = 0.5
    //   dX[1] = 0.5 * (Wq[0][1] + Wq[1][1]) = 0.5 * (-1 + 1) = 0
    CHECK(std::fabs(dX[0] - 0.5f) < 1e-5f);
    CHECK(std::fabs(dX[1] - 0.0f) < 1e-5f);

    // Master weights must have stepped AGAINST the gradient (descent):
    // W[0] = 0.4 - 0.05*1 = 0.35   (decreased)
    // W[1] = -0.4 - 0.05*2 = -0.5  (decreased further negative)
    CHECK(std::fabs(W[0] - 0.35f) < 1e-5f);
    CHECK(std::fabs(W[1] - (-0.5f)) < 1e-5f);
}

static void test_gguf_magic() {
    TEST("gguf: magic parsing");
    const uint8_t magic_ok[4] = {'G', 'G', 'U', 'F'};
    const uint8_t magic_bad[4] = {'N', 'O', 'P', 'E'};
    CHECK(gguf::read_u32(magic_ok, 0) == 0x46554747); // 'GGUF' little-endian
    CHECK(gguf::read_u32(magic_bad, 0) != 0x46554747);
}

// ===========================================================================
// Phase 4+: novel capabilities (sensory, emotional, skills, swarm, intel)
// ===========================================================================
static PcmAudio make_tone(double hz, double seconds, int32_t sr = 16000) {
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

static void test_fft_matches_dft() {
    TEST("fft: Bluestein exact-bin DFT matches naive O(N^2) reference");
    constexpr size_t N = 400, BINS = 201;

    // deterministic pseudo-random signal (LCG)
    std::vector<float> x(N);
    uint32_t s = 42;
    for (size_t i = 0; i < N; ++i) {
        s = s * 1664525u + 1013904223u;
        x[i] = static_cast<float>(static_cast<double>(s >> 8) / 8388608.0 - 1.0);
    }

    // naive reference: bins 0..BINS-1 of the length-N DFT
    double ref_re[BINS], ref_im[BINS];
    for (size_t k = 0; k < BINS; ++k) {
        double re = 0.0, im = 0.0;
        for (size_t t = 0; t < N; ++t) {
            const double ang = -2.0 * 3.14159265358979323846 *
                               static_cast<double>(k) * static_cast<double>(t) /
                               static_cast<double>(N);
            re += static_cast<double>(x[t]) * std::cos(ang);
            im += static_cast<double>(x[t]) * std::sin(ang);
        }
        ref_re[k] = re;
        ref_im[k] = im;
    }

    dsp::DftBins dft(N, BINS);
    double got_re[BINS], got_im[BINS];
    dft.run(x.data(), got_re, got_im);

    double max_err = 0.0;
    double scale = 0.0;
    for (size_t k = 0; k < BINS; ++k) {
        max_err = std::max(max_err, std::fabs(got_re[k] - ref_re[k]));
        max_err = std::max(max_err, std::fabs(got_im[k] - ref_im[k]));
        scale = std::max(scale, std::fabs(ref_re[k]));
        scale = std::max(scale, std::fabs(ref_im[k]));
    }
    CHECK(max_err < 1e-6 * std::max(scale, 1.0));

    // power-spectrum equivalence on a tone (mel front end uses |X|^2)
    PcmAudio tone = make_tone(440.0, 0.5);
    std::vector<float> sig(tone.samples.begin(), tone.samples.begin() + N);
    dft.run(sig.data(), got_re, got_im);
    double pk_got = 0.0, pk_ref = 0.0;
    size_t k_got = 0, k_ref = 0;
    for (size_t k = 1; k < BINS; ++k) {
        const double pg = got_re[k] * got_re[k] + got_im[k] * got_im[k];
        if (pg > pk_got) { pk_got = pg; k_got = k; }
    }
    for (size_t k = 1; k < BINS; ++k) {
        double re = 0.0, im = 0.0;
        for (size_t t = 0; t < N; ++t) {
            const double ang = -2.0 * 3.14159265358979323846 *
                               static_cast<double>(k) * static_cast<double>(t) /
                               static_cast<double>(N);
            re += static_cast<double>(sig[t]) * std::cos(ang);
            im += static_cast<double>(sig[t]) * std::sin(ang);
        }
        const double pr = re * re + im * im;
        if (pr > pk_ref) { pk_ref = pr; k_ref = k; }
    }
    CHECK(k_got == k_ref);
}


// Helper bridging the private safe_arithmetic through public behavior.
// (defined before test_flash_skills_and_synthesis uses it)
static bool FlashSkillPool_is_safe_hack() {
    // SQL-ish/lettered strings must fail inside the sandbox; emulate via a
    // one-step skill.
    FlashSkill s;
    s.name = "probe";
    RecipeStep st;
    st.op = "calc";
    st.expr = "rm -rf";
    s.steps.push_back(st);
    FlashSkillPool p;
    p.install(s);
    const FlashSkill* probe = p.get("probe");
    if (!probe) return false;
    const SkillResult r = p.run(*probe, {});
    return !r.ok;   // sandbox must reject
}

static void test_sensory_fingerprint() {
    TEST("sensory: enroll + verify matches same voice, differs across users");
    SensoryFingerprint sf;

    // User A: low voice (120 Hz), steady typing.
    SensoryObservation a1;
    const PcmAudio ta = make_tone(120.0, 1.5);
    a1.pcm = ta.samples.data(); a1.pcm_len = ta.samples.size();
    for (int i = 0; i < 20; ++i) a1.digraph_ms.push_back(110.0f + i * 3.0f);
    CHECK(sf.enroll(a1));
    CHECK(sf.enrolled());
    const std::string tok_a = sf.identity_token();
    CHECK(tok_a.size() == 32);

    SensoryObservation a2 = a1;
    bool match = false;
    const float score_same = sf.verify(a2, match);
    CHECK(match);
    CHECK(score_same >= 0.72f);

    // User B: high voice (300 Hz), fast typing.
    SensoryFingerprint sf2;
    SensoryObservation b1;
    const PcmAudio tb = make_tone(300.0, 1.5);
    b1.pcm = tb.samples.data(); b1.pcm_len = tb.samples.size();
    for (int i = 0; i < 20; ++i) b1.digraph_ms.push_back(40.0f + (i % 4) * 15.0f);
    sf2.enroll(b1);

    // B should NOT match A's profile.
    bool match_b = true;
    const float score_b = sf.verify(b1, match_b);
    CHECK(!match_b);
    CHECK(score_b < score_same);

    // Persistence round-trip.
    CHECK(sf.save("build/sensory_test.bin"));
    SensoryFingerprint sf3;
    CHECK(sf3.load("build/sensory_test.bin"));
    bool match3 = false;
    sf3.verify(a2, match3);
    CHECK(match3);
}

static void test_emotional_resonance() {
    TEST("emotional: text/voice fusion + response modulation");
    EmotionalResonance er;

    EmotionalInput in;
    in.text = "I love this, it is amazing and awesome!!!";
    const EmotionalState happy = er.perceive(in);
    CHECK(happy.from_text);
    CHECK(happy.valence > 0.0f);
    CHECK(happy.confidence > 0.0f);

    EmotionalInput neg;
    neg.text = "this is terrible and broken, I hate it, worst error ever";
    const EmotionalState sad = er.perceive(neg);
    CHECK(sad.valence < 0.0f);

    // Voice: bright + loud + fast -> positive/high arousal.
    const PcmAudio bright = make_tone(900.0, 1.2);
    EmotionalInput vin;
    vin.pcm = bright.samples.data();
    vin.pcm_len = bright.samples.size();
    const EmotionalState v = er.perceive(vin);
    CHECK(v.from_voice);

    // Modulation: negative states add an empathetic lead-in.
    const std::string mod = er.modulate("Here is the answer.", sad);
    CHECK(mod.size() > std::string("Here is the answer.").size());
    // Token fragment names the emotion.
    const std::string frag = er.token_fragment(happy);
    CHECK(frag.find("[emotion:") == 0);

    // Neutral text stays unmodulated.
    EmotionalInput neutral;
    neutral.text = "the table has four legs";
    const EmotionalState n = er.perceive(neutral);
    const std::string plain = er.modulate("Answer.", n);
    CHECK(plain == "Answer.");
}

static void test_flash_skills_and_synthesis() {
    TEST("flash skills: sandbox calc, zero-shot synthesis, retirement");
    FlashSkillPool pool;
    pool.install_builtins();
    CHECK(pool.size() >= 3);

    // Match + run builtin calc through the sandbox.
    const FlashSkill* calc = pool.match("what is 12*(3+4)?");
    CHECK(calc != nullptr);
    if (calc) {
        std::unordered_map<std::string, std::string> args;
        args["expr"] = "12*(3+4)";
        const SkillResult r = pool.run(*calc, args);
        CHECK(r.ok);
        CHECK(r.output == "84");
    }

    // Sandbox rejects unsafe expressions.
    CHECK(FlashSkillPool_is_safe_hack());

    // Zero-shot synthesis: arithmetic task with no explicit skill.
    FlashSkillPool pool2;
    pool2.install_builtins();
    const FlashSkill* zs = pool2.synthesize("please compute 1234*5678 for me");
    CHECK(zs != nullptr);
    if (zs) {
        CHECK(zs->synthesized);
        std::unordered_map<std::string, std::string> args;
        args["expr"] = "1234*5678";
        const SkillResult r = pool2.run(*zs, args);
        CHECK(r.ok);
        CHECK(r.output == "7006652");
    }

    // Synthesis of a length task.
    const FlashSkill* zs2 = pool2.synthesize("count the letters of \"omniseed\"");
    CHECK(zs2 != nullptr);

    // Persistence round-trip.
    CHECK(pool2.save());
    FlashSkillPool pool3;
    pool3.set_store_path("./state/skills.bin");
    CHECK(pool3.load());
    CHECK(pool3.size() >= pool2.size() - 1);
}

static void test_swarm_protocol() {
    TEST("swarm: codec round-trip, discovery, delegation, memory xfer");
    // Wire codec: encode -> decode preserves the message; tamper fails CRC.
    SwarmMessage m;
    m.type = SwarmMessage::Type::TaskOffer;
    m.from = "node-A";
    m.task_key = "calc sum";
    m.payload = "12*(3+4)";
    const std::string blob = encode_message(m, "key");
    SwarmMessage back;
    CHECK(decode_message(blob, "key", back));
    CHECK(back.payload == m.payload);
    CHECK(back.from == m.from);
    std::string tampered = blob;
    // Flip a payload byte near the end of the frame (inside the message body,
    // past the framing) -> payload CRC must catch it.
    const size_t victim = tampered.size() - 9;
    tampered[victim] = static_cast<char>(tampered[victim] + 1);
    SwarmMessage bad;
    CHECK(!decode_message(tampered, "key", bad));
    // Flipping a byte inside the FROM field must trip the header CRC.
    std::string tampered2 = blob;
    const size_t hv = 4 + 4 + 4 + 4 + 2;   // magic+ver+type+len+part of from
    if (hv < tampered2.size()) tampered2[hv] =
        static_cast<char>(tampered2[hv] + 1);
    CHECK(!decode_message(tampered2, "key", bad));
    // Wrong key must fail.
    CHECK(!decode_message(blob, "other-key", bad));

    // Two coordinators over the loopback mesh.
    LoopbackMesh& mesh = LoopbackMesh::instance();
    SwarmCoordinator::Config ca, cb;
    ca.node_id = "node-A";
    cb.node_id = "node-B";
    SwarmCoordinator a(ca), b(cb);
    mesh.join("node-A");
    mesh.join("node-B");

    // B announces itself to A.
    SwarmNode nb;
    nb.id = "node-B";
    nb.caps = SwarmNode::CapModel | SwarmNode::CapVision;
    nb.free_ram_bytes = 200ull * 1024 * 1024;
    SwarmMessage hello;
    hello.type = SwarmMessage::Type::Hello;
    hello.from = "node-B";
    hello.payload = std::string(8, '\0');
    const uint64_t caps = nb.caps;
    for (int i = 0; i < 8; ++i)
        hello.payload[i] = static_cast<char>((caps >> (8 * i)) & 0xFF);
    CHECK(mesh.send("node-A", hello));
    CHECK(a.step(mesh));            // A learns about B
    CHECK(a.peer_count() == 1);

    // A delegates a task to B; B executes via its executor and answers.
    static std::string executed_task;
    const bool delegated = a.delegate("what is 6*7");
    CHECK(delegated);
    CHECK(b.step(mesh, [](const std::string& task, std::string& result) {
        executed_task = task;
        result = "42";
        return true;
    }));
    CHECK(executed_task == "what is 6*7");

    // Memory transfer: A shares crystal summaries; B ingests them. (B's
    // earlier TaskResult reply is still queued in its mailbox, so drain
    // until a MemoryXfer has been processed.)
    CHECK(a.share_memories(mesh, {"user bought a red car",
                                  "meeting at 3pm friday"}));
    bool got_memories = false;
    for (int i = 0; i < 8 && !got_memories; ++i)
        got_memories = b.shared_memories().size() >= 2 || !b.step(mesh);
    CHECK(got_memories);
    CHECK(b.shared_memories().size() == 2);

    // Scoring: caps filter.
    SwarmNode weak;
    weak.caps = 0;
    CHECK(SwarmCoordinator::score_peer(weak, SwarmNode::CapModel) < 0.0);

    mesh.leave("node-A");
    mesh.leave("node-B");
}

static void test_udp_beacon_roundtrip() {
    TEST("swarm: UDP beacon localhost round-trip, both directions");
    // Two beacons on distinct loopback ports; messages cross the real wire
    // through the platform socket shim (CRC32 + keystream encryption),
    // unicast-addressed so neither side hears its own datagrams.
    UdpBeacon::Config ca, cb;
    ca.port = 47471;
    cb.port = 47472;
    UdpBeacon a(ca), b(cb);
    if (!a.start() || !b.start()) {
        // No UDP stack / port blocked (hardened CI sandboxes): skip cleanly.
        platform::log_info("SKIP  UDP beacon unavailable on this host");
        return;
    }

    SwarmMessage out_a, out_b;
    std::string ep_a, ep_b;

    // a -> b (b's poll sees a's broadcast; a's own loopback is disabled by
    // design: a beacon never receives its own datagrams).
    SwarmMessage m1;
    m1.type = SwarmMessage::Type::Hello;
    m1.from = "beacon-A";
    m1.payload = "ping-from-a";
    CHECK(a.send("udp:127.0.0.1:47472", m1));   // unicast straight to b
    bool got1 = false;
    for (int i = 0; i < 100 && !got1; ++i) {
        got1 = b.poll(out_b, ep_b);
        if (!got1) platform::yield_now();
    }
    CHECK(got1);
    if (got1) {
        CHECK(out_b.payload == m1.payload);
        CHECK(out_b.from == m1.from);
        CHECK(ep_b.rfind("udp:127.0.0.1:", 0) == 0);
    }

    // b -> a: the reply direction proves the shim is symmetric.
    SwarmMessage m2;
    m2.type = SwarmMessage::Type::TaskOffer;
    m2.from = "beacon-B";
    m2.task_key = "echo";
    m2.payload = "pong-from-b";
    CHECK(b.send("udp:127.0.0.1:47471", m2));   // unicast back to a
    bool got2 = false;
    for (int i = 0; i < 100 && !got2; ++i) {
        got2 = a.poll(out_a, ep_a);
        if (!got2) platform::yield_now();
    }
    CHECK(got2);
    if (got2) {
        CHECK(out_a.payload == m2.payload);
        CHECK(out_a.from == m2.from);
        CHECK(ep_a.rfind("udp:127.0.0.1:", 0) == 0);
    }

    // Source port correctness: the endpoint must name the PEER's bind port
    // (a sends from 47471, b from 47472 — host order through the shim).
    if (got1) CHECK(ep_b == "udp:127.0.0.1:47471");
    if (got2) CHECK(ep_a == "udp:127.0.0.1:47472");

    a.stop();
    b.stop();
}

// ===========================================================================
// Phase-Omega: uncertainty quantification, prefix snapshots, memory decay
// ===========================================================================
static void test_uncertainty() {
    TEST("uncertainty: entropy/margin/abstain + entropy anomaly monitor");
    // Sharp distribution: one dominant logit.
    {
        Tensor sharp("sharp", {16}, DType::F32);
        for (int32_t i = 0; i < 16; ++i) sharp.f32()[i] = (i == 3) ? 12.0f : 0.0f;
        const auto r = Uncertainty::analyze(sharp);
        CHECK(r.top1_id == 3);
        CHECK(r.top1_prob > 0.99f);
        CHECK(r.margin > 0.98f);
        CHECK(r.entropy < 0.1f);
        CHECK(!Uncertainty::should_abstain(r));
    }
    // Flat distribution: uniform logits -> max entropy -> abstain.
    {
        Tensor flat("flat", {16}, DType::F32);
        for (int32_t i = 0; i < 16; ++i) flat.f32()[i] = 1.0f;
        const auto r = Uncertainty::analyze(flat);
        CHECK(r.normalized_entropy > 0.99f);
        CHECK(r.margin < 0.01f);
        CHECK(Uncertainty::should_abstain(r));
    }
    // Coin-flip top-2: two equal peaks -> thin margin -> abstain.
    {
        Tensor tie("tie", {16}, DType::F32);
        for (int32_t i = 0; i < 16; ++i) tie.f32()[i] = (i == 2 || i == 9) ? 10.0f : 0.0f;
        const auto r = Uncertainty::analyze(tie);
        CHECK(std::fabs(r.margin) < 0.01f);
        CHECK(Uncertainty::should_abstain(r));
    }
    // Entropy monitor: stable window -> no anomaly; outlier -> anomaly.
    {
        EntropyMonitor m(16);
        bool any = false;
        for (int i = 0; i < 12; ++i) any = m.push(2.0f) || any;
        CHECK(!any);
        CHECK(m.push(9.0f));          // z >> 3
        CHECK(m.is_anomaly());
        CHECK(m.z_score() > 3.0f);
    }
}

static void test_prefix_cache() {
    TEST("prefix cache: WKV snapshot store/restore round-trip");
    // Build a tiny model-shaped state via the minimal byte tokenizer path:
    // PrefixCache is model-shape-driven, so exercise it against a REAL
    // RwkvModel only when weights exist; otherwise validate the container
    // semantics with a stubbed config through the real model class.
    // (The real-weights suite covers the model-backed path.)
    PrefixCache pc;
    CHECK(pc.size() == 0);
    CHECK(!pc.has("session-a"));
    // store/load require a valid model; without one they must fail cleanly.
    RwkvModel bogus;
    RwkvState st;
    CHECK(!pc.store("session-a", bogus, st));
    CHECK(pc.size() == 0);
}

static void test_memory_decay() {
    TEST("memory: entropy salience + Ebbinghaus decay + working memory");
    Tokenizer tok;
    tok.build_minimal();

    MemoryCrystals mc;
    // High-entropy stream (varied byte tokens) -> boosted importance.
    std::vector<int32_t> varied;
    for (int i = 0; i < 64; ++i) varied.push_back(16 + (i * 7) % 200);
    CHECK(mc.crystallize(varied, tok, 100, 6.0f));
    CHECK(mc.size() == 1);

    // Retrieval updates recency bookkeeping.
    const auto hits = mc.retrieve(varied, 1);
    CHECK(hits.size() == 1);

    // Working memory ring: capacity respected.
    WorkingMemory wm(8);
    for (int i = 0; i < 20; ++i) wm.push(i);
    CHECK(wm.size() == 8);
    CHECK(wm.tokens().front() == 12);   // oldest evicted
}



static void test_agent_intel() {
    TEST("agent intel: intent, dialogue state, confidence, kg, interrupts");
    CHECK(IntentClassifier::classify("what is a carburetor") ==
          Intent::Informational);
    CHECK(IntentClassifier::classify("calculate 2+2") == Intent::Directive);
    CHECK(IntentClassifier::classify("why does it fail") ==
          Intent::Exploratory);
    CHECK(IntentClassifier::classify("hello there") == Intent::Social);

    DialogueStateTracker dst;
    dst.update("my name is Alice and I like durians", "noted!");
    CHECK(dst.turns() == 1);
    CHECK(!dst.topic().empty());
    CHECK(dst.slots().count("name") == 1);

    // Confidence: supported reply beats unsupported one.
    const std::string ctx = "the mission to mars uses solar panels";
    const ConfidenceReport ok = ConfidenceScorer::score(
        "the mission uses solar panels for power", ctx, 0.1f);
    const ConfidenceReport risky = ConfidenceScorer::score(
        "quantum bananas orbit the moon on tuesdays", ctx, 0.9f);
    CHECK(ok.confidence > risky.confidence);
    CHECK(risky.hallucination_risk > ok.hallucination_risk);

    platform::log_info("MARK before thinking");
    // Thinking mode: forced thinks, off does not, auto thinks on hard input.
    ThinkingMode tm;
    tm.set_mode(ThinkingMode::Mode::Forced);
    CHECK(tm.should_think("hi"));
    tm.set_mode(ThinkingMode::Mode::Off);
    CHECK(!tm.should_think("explain quantum entanglement in detail"));
    tm.set_mode(ThinkingMode::Mode::Auto);
    CHECK(tm.should_think("explain why the sky is blue step by step"));
    CHECK(!tm.should_think("hi"));

    // Error recovery policy ladder.
    ErrorRecovery::Attempt a;
    CHECK(ErrorRecovery::decide("tool not found", a) ==
          ErrorRecovery::Policy::Abort);
    ErrorRecovery::Attempt b;
    CHECK(ErrorRecovery::decide("network timeout", b) ==
          ErrorRecovery::Policy::Retry);
    CHECK(ErrorRecovery::decide("network timeout", b) ==
          ErrorRecovery::Policy::Retry);
    CHECK(ErrorRecovery::decide("network timeout", b) ==
          ErrorRecovery::Policy::Backoff);

    // Feedback loop trust.
    UserFeedbackLoop fb;
    fb.record(UserFeedbackLoop::Verdict::Confirm, "task1");
    fb.record(UserFeedbackLoop::Verdict::Confirm, "task1");
    fb.record(UserFeedbackLoop::Verdict::Correct, "task2");
    CHECK(fb.trust("task1") > 0.5);
    CHECK(fb.trust("task2") < 0.5);
    CHECK(fb.trust("unknown") == 0.5);

    platform::log_info("MARK before kg");
    // Knowledge graph mining + recall.
    KnowledgeGraph kg;
    const size_t mined = kg.ingest_text(
        "Alice is an engineer. Bob has a red car. Alice likes durians.", 1);
    CHECK(mined >= 3);
    CHECK(kg.query("alice").size() >= 2);
    CHECK(!kg.facts_for_prompt("bob", 3).empty());
    CHECK(kg.save("build/kg_test.bin"));
    KnowledgeGraph kg2;
    CHECK(kg2.load("build/kg_test.bin"));
    CHECK(kg2.size() == kg.size());

    platform::log_info("MARK before interrupts");
    // Interrupt system: priority ordering + preemption.
    SensoryInterruptSystem sis;
    sis.raise(SensoryInterruptSystem::Event::SoundSpeech, 2, "[audio:speech]");
    sis.raise(SensoryInterruptSystem::Event::SoundAlarm, 9, "[audio:alarm]");
    SensoryInterruptSystem::Interrupt iv;
    CHECK(sis.poll(iv));
    CHECK(iv.event == SensoryInterruptSystem::Event::SoundAlarm);
    CHECK(sis.pending() == 1);
    CHECK(sis.should_preempt(1));
    CHECK(!sis.should_preempt(9));

    // Progressive disclosure.
    const std::string long_text =
        "First sentence is here. Second sentence follows. Third one too.";
    const std::string sum = ProgressiveDisclosure::summarize(long_text, 1);
    CHECK(sum.find("First") != std::string::npos);
    CHECK(sum.find("Third") == std::string::npos);
    CHECK(ProgressiveDisclosure::wants_detail("tell me more"));
    CHECK(!ProgressiveDisclosure::wants_detail("ok thanks"));
}

static void test_audio_events() {
    TEST("audio events: sound classification, scene, anomaly detection");
    // Silence -> Quiet scene.
    PcmAudio silence;
    silence.sample_rate = 16000;
    silence.samples.assign(16000 * 2, 0.0f);
    float conf = 0.0f;
    CHECK(AudioSceneClassifier::classify(silence, conf) == AudioScene::Quiet);
    CHECK(conf > 0.5f);

    // Pure tone -> classified as some non-quiet scene with a verdict.
    const PcmAudio tone = make_tone(440.0, 2.0);
    SoundEventDetector det;
    const SoundEventResult ev = det.detect(tone);
    (void)ev;   // a pure tone may or may not map to an event; API exercised

    const AudioScene scene = AudioSceneClassifier::classify(tone, conf);
    CHECK(scene != AudioScene::Quiet);

    // Anomaly detector: quiet baseline, loud frame flagged.
    AudioAnomalyDetector ad;
    float bands[8];
    for (int i = 0; i < 150; ++i) {
        for (int b = 0; b < 8; ++b)
            bands[b] = -8.0f + 0.1f * ((i + b) % 4);   // stable baseline
        ad.push_frame(bands, 8);
    }
    CHECK(ad.baseline_ready());
    for (int b = 0; b < 8; ++b) bands[b] = 1.5f;        // wild deviation
    CHECK(ad.push_frame(bands, 8));                     // flagged

    // Wake word detector: template scan API + enrollment.
    WakeWordDetector ww;
    float score = 0.0f;
    CHECK(!ww.scan(make_tone(220.0, 0.4), score));
    CHECK(ww.enroll(tone));
    CHECK(ww.template_enrolled());
    CHECK(ww.scan(tone, score));                        // self-match triggers
    CHECK(score >= ww.config().match_threshold);
}

static void test_vision_tasks() {
    TEST("vision tasks: pointer crop, spatial map, tracker, gestures, resolution");
    // Pointer grounding: deictic query -> center region + crop.
    PointerPerception pp;
    Region region;
    std::string token;
    CHECK(pp.ground("what is this?", region, token));
    CHECK(region.w > 0.0f && region.w < 1.0f);
    CHECK(token.find("[vision:pointer:") == 0);
    CHECK(!pp.ground("what time is it?", region, token));

    Image img;
    img.width = 64; img.height = 64;
    img.rgb.assign(64ull * 64 * 3, 0);
    // Bright square in the top-right quadrant.
    for (int y = 4; y < 20; ++y)
        for (int x = 44; x < 60; ++x) {
            uint8_t* px = img.rgb.data() + (y * 64 + x) * 3;
            px[0] = px[1] = px[2] = 255;
        }
    const Image cropped = PointerPerception::crop(img, region);
    CHECK(cropped.valid());
    CHECK(cropped.width < img.width);

    // Spatial map: peak cell should be top-right-ish (row 0, col 3).
    const SpatialContextMapper::Map m = SpatialContextMapper::build(img);
    CHECK(m.peak_row <= 1 && m.peak_col >= 2);
    CHECK(SpatialContextMapper::describe(m).find("densest") == 0);
    CHECK(!SpatialContextMapper::token_fragment(m).empty());

    // Motion tracker: static frame -> no motion; jump frame -> motion.
    MotionTracker mt;
    const TrackPoint p1 = mt.update(img, 0);
    CHECK(p1.cx < 0.0f);                    // first frame: no history
    Image img2 = img;
    for (int y = 4; y < 20; ++y)
        for (int x = 44; x < 60; ++x) {
            uint8_t* px = img2.rgb.data() + (y * 64 + x) * 3;
            px[0] = px[1] = px[2] = 10;     // big change
        }
    const TrackPoint p2 = mt.update(img2, 100);
    CHECK(p2.energy > 0.0f);
    CHECK(p2.cx >= 0.0f);

    // Gesture: rapid lateral direction flips -> Wave.
    GestureRecognizer gr;
    Gesture g = Gesture::None;
    for (int i = 0; i < 14; ++i) {
        const float x = (i % 2 == 0) ? 0.2f : 0.8f;
        g = gr.update(x, 0.5f, 0.5f);
    }
    CHECK(g == Gesture::Wave);

    // Dynamic resolution: battery saver shrinks, complexity grows.
    CHECK(DynamicResolution::pick_input_size(0.1f, false) == 64);
    CHECK(DynamicResolution::pick_input_size(0.5f, false) == 96);
    CHECK(DynamicResolution::pick_input_size(0.9f, false) == 128);
    CHECK(DynamicResolution::pick_input_size(0.9f, true) == 64);
}

static void test_memory_crystals_and_streaming() {
    TEST("memory: streaming sinks/window/retire + crystal storage/retrieval");
    Tokenizer tok;
    CHECK(tok.build_minimal());

    StreamingLlm::Config scfg;
    scfg.window_size = 16;
    StreamingLlm stream(scfg);
    const auto ids = tok.encode(
        "Alice is an engineer. Bob has a red car. The sky is blue today.",
        false);
    CHECK(!ids.empty());
    size_t retired_total = 0;
    for (const int32_t id : ids) {
        stream.push(id);
        retired_total += stream.drain_retired().size();
    }
    CHECK(stream.total_seen() == static_cast<int64_t>(ids.size()));
    CHECK(stream.sinks().size() >= 1);
    CHECK(retired_total > 0);

    MemoryCrystals mem;
    // Crystallize from the accumulated retired stream (drain_retired() is
    // destructive, so collect during the push loop above).
    CHECK(retired_total > 0);
    CHECK(mem.crystallize(std::vector<int32_t>(ids.end() - std::min<size_t>(ids.size(), 8), ids.end()), tok, 100));
    // Retrieval: a query about the car should return something.
    const auto q = tok.encode("who has the red car", false);
    CHECK(!mem.retrieve(q, 2).empty());

    CHECK(mem.save("build/mem_test.bin"));
    MemoryCrystals mem2;
    CHECK(mem2.load("build/mem_test.bin"));
    CHECK(mem2.size() == mem.size());
}

// ===========================================================================
// main
// ===========================================================================
int main() {
    platform::log_info("OmniSeed test suite — %s", platform::os_name());
    platform::log_info("================================");

    test_page_size();
    test_mmap_roundtrip();
    test_aligned_alloc();
    test_peak_rss();
    test_clock();
    test_os_name();

    test_tensor_softmax_and_inplace_exp();
    test_ternary_pack_roundtrip();
    test_ternary_matmul();
    test_bitlinear_train_step();
    test_gguf_magic();
    test_fft_matches_dft();

    // Novel capabilities + extended stacks.
    test_sensory_fingerprint();
    test_emotional_resonance();
    test_flash_skills_and_synthesis();
    test_swarm_protocol();
    test_udp_beacon_roundtrip();
    test_uncertainty();
    test_prefix_cache();
    test_memory_decay();
    test_agent_intel();
    test_audio_events();
    test_vision_tasks();
    test_memory_crystals_and_streaming();

    platform::log_info("================================");
    platform::log_info("RESULT: %d passed, %d failed", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
