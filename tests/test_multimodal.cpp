// =============================================================================
//  OmniSeed — tests/test_multimodal.cpp
//
//  MILESTONE 5 — the multimodal joint.
//
//  src/vision, src/audio and src/runtime/token_bus.cpp were all REAL and all
//  shipped. TokenBus even documented its job as "fuses text, vision and audio
//  token streams into ONE unified sequence" — and had ZERO tests. Nothing
//  anywhere turned an image into the h[E] the heads read.
//
//  PART A — TokenBus. UNGATED. This is the first test this file has ever had.
//      A1  text-only: BOS first, then exactly encode(text, add_bos=false)
//      A2  a vision payload is wrapped in <|vision|> ... </|vision|>
//      A3  an audio payload is wrapped in <|audio|> ... </|audio|>
//      A4  an EMPTY payload emits NO markers (a marker pair with nothing
//          between it is not the same claim as "no modality")
//      A5  segment order is preserved (text, then vision, then audio)
//      A6  exceeding the token budget returns FALSE rather than a short stream
//      A7  the stats count the WRAPPED size, and total matches out_ids
//
//  PART B — MultimodalBridge. UNGATED (a minimal tokenizer, no model weights).
//      B1  vision without a codebook FAILS with a reason; it does not silently
//          drop the image and answer from text alone
//      B2  a codebook of the wrong width is refused, not truncated
//      B3  nearest_token finds an exact row
//      B4  quantize() maps an [M, E] block to M ids, and refuses a partial row
//      B5  fused vision ids are the quantized ids, wrapped
//      B6  no vision supplied => no vision markers, and the count is 0
//      B7  a non-finite query vector yields -1 instead of winning by NaN
//      B8  the codebook stride is a REAL accuracy dial: with stride 2 a row at
//          an odd index is unreachable, and the bridge returns a neighbour
//      B9  summary() never claims a modality that contributed no tokens
//
//  PART C — end to end through the real backbone. GATED on
//    models/rwkv7-0.1B-ternary.gguf. Skips with a printed reason when absent.
//      C1  image-shaped embeddings -> tokens -> RWKV -> h[E] is finite and of
//          width E, and a head reads that h[E] and reports a distribution
// =============================================================================
#include "omniseed/classification_head.h"
#include "omniseed/core/platform.h"
#include "omniseed/core/rwkv.h"
#include "omniseed/core/tokenizer.h"
#include "omniseed/multimodal.h"
#include "omniseed/runtime/token_bus.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace omniseed;

static int g_passed = 0;
static int g_failed = 0;
static int g_skipped = 0;
static std::string g_current;

#define TEST(name) g_current = (name); platform::log_info("TEST  %s", name)
#define CHECK(cond)                                                        \
    do {                                                                   \
        if (cond) { ++g_passed; }                                          \
        else {                                                             \
            ++g_failed;                                                    \
            platform::log_error("FAIL  %s  (line %d): %s",                 \
                                g_current.c_str(), __LINE__, #cond);       \
        }                                                                  \
    } while (0)
#define SKIP(msg)                                                          \
    do {                                                                   \
        ++g_skipped;                                                       \
        platform::log_info("SKIP  %s  (%s)", g_current.c_str(), msg);      \
    } while (0)

namespace {

bool file_exists(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return static_cast<bool>(f);
}

// Is `needle` a contiguous subsequence of `hay`?
bool contains_seq(const std::vector<int32_t>& hay,
                  const std::vector<int32_t>& needle) {
    if (needle.empty() || hay.size() < needle.size()) return false;
    return std::search(hay.begin(), hay.end(), needle.begin(), needle.end()) !=
           hay.end();
}

// A synthetic codebook [N, E] whose row r is the r-th unit basis vector, so
// the nearest row to e_r is exactly r and the answer is not a coincidence of
// random weights.
Tensor make_basis_codebook(int32_t N, int32_t E) {
    Tensor cb("codebook", {N, E}, DType::F32);
    float* p = cb.f32();
    for (int32_t i = 0; i < N * E; ++i) p[i] = 0.0f;
    for (int32_t r = 0; r < N && r < E; ++r) p[r * E + r] = 1.0f;
    return cb;
}

std::vector<float> basis_vector(int32_t E, int32_t r) {
    std::vector<float> v(static_cast<size_t>(E), 0.0f);
    if (r >= 0 && r < E) v[static_cast<size_t>(r)] = 1.0f;
    return v;
}

} // namespace

// =============================================================================
// PART A — TokenBus (first tests this module has ever had)
// =============================================================================
static void part_a_token_bus() {
    Tokenizer tok;
    CHECK(tok.build_minimal(512));
    CHECK(tok.valid());
    // build_minimal places the control tokens so these ids resolve exactly.
    CHECK(tok.piece(Tokenizer::kVisionStartId) == "<|vision|>");
    CHECK(tok.piece(Tokenizer::kVisionEndId)   == "</|vision|>");
    CHECK(tok.piece(Tokenizer::kAudioStartId)  == "<|audio|>");
    CHECK(tok.piece(Tokenizer::kAudioEndId)    == "</|audio|>");

    TokenBus bus(tok);
    const std::vector<int32_t> text_ids = tok.encode("hello world", false);
    CHECK(!text_ids.empty());

    // ---- A1: text only -----------------------------------------------------
    TEST("A1: text-only fuse is BOS followed by exactly encode(text, false)");
    {
        std::vector<TokenBus::Segment> segs;
        TokenBus::Segment s;
        s.kind = TokenBus::Segment::Kind::Text;
        s.text = "hello world";
        segs.push_back(s);

        std::vector<int32_t> out;
        CHECK(bus.fuse(segs, out));
        CHECK(!out.empty());
        CHECK(out[0] == Tokenizer::kBosId);
        const std::vector<int32_t> body(out.begin() + 1, out.end());
        CHECK(body == text_ids);
        CHECK(bus.last_stats().text_tokens ==
              static_cast<int64_t>(text_ids.size()));
        CHECK(bus.last_stats().total == static_cast<int64_t>(out.size()));
    }

    // ---- A2: vision wrapped ------------------------------------------------
    TEST("A2: a vision payload is wrapped in the <|vision|> markers");
    {
        const std::vector<int32_t> payload = {100, 101, 102};
        std::vector<TokenBus::Segment> segs;
        TokenBus::Segment s;
        s.kind = TokenBus::Segment::Kind::Vision;
        s.ids  = payload;
        segs.push_back(s);

        std::vector<int32_t> out;
        CHECK(bus.fuse(segs, out));

        std::vector<int32_t> wrapped;
        wrapped.push_back(Tokenizer::kVisionStartId);
        wrapped.insert(wrapped.end(), payload.begin(), payload.end());
        wrapped.push_back(Tokenizer::kVisionEndId);
        CHECK(contains_seq(out, wrapped));
        CHECK(out[0] == Tokenizer::kBosId);
        // the markers are included in the counted size
        CHECK(bus.last_stats().vision_tokens ==
              static_cast<int64_t>(payload.size()) + 2);
    }

    // ---- A3: audio wrapped -------------------------------------------------
    TEST("A3: an audio payload is wrapped in the <|audio|> markers");
    {
        const std::vector<int32_t> payload = {7, 8};
        std::vector<TokenBus::Segment> segs;
        TokenBus::Segment s;
        s.kind = TokenBus::Segment::Kind::Audio;
        s.ids  = payload;
        segs.push_back(s);

        std::vector<int32_t> out;
        CHECK(bus.fuse(segs, out));

        std::vector<int32_t> wrapped;
        wrapped.push_back(Tokenizer::kAudioStartId);
        wrapped.insert(wrapped.end(), payload.begin(), payload.end());
        wrapped.push_back(Tokenizer::kAudioEndId);
        CHECK(contains_seq(out, wrapped));
        CHECK(bus.last_stats().audio_tokens ==
              static_cast<int64_t>(payload.size()) + 2);
    }

    // ---- A4: empty payloads emit nothing -----------------------------------
    TEST("A4: an empty payload emits NO markers");
    {
        std::vector<TokenBus::Segment> segs;
        TokenBus::Segment v;
        v.kind = TokenBus::Segment::Kind::Vision;   // ids left empty
        segs.push_back(v);
        TokenBus::Segment a;
        a.kind = TokenBus::Segment::Kind::Audio;    // ids left empty
        segs.push_back(a);

        std::vector<int32_t> out;
        CHECK(bus.fuse(segs, out));
        CHECK(out.size() == 1);                       // BOS only
        CHECK(out[0] == Tokenizer::kBosId);
        CHECK(bus.last_stats().vision_tokens == 0);
        CHECK(bus.last_stats().audio_tokens == 0);
        CHECK(!contains_seq(out, {Tokenizer::kVisionStartId}));
        CHECK(!contains_seq(out, {Tokenizer::kAudioStartId}));
    }

    // ---- A5: order preserved -----------------------------------------------
    TEST("A5: segment order is preserved (text, vision, audio)");
    {
        std::vector<TokenBus::Segment> segs;
        TokenBus::Segment t;
        t.kind = TokenBus::Segment::Kind::Text;  t.text = "see this";
        TokenBus::Segment v;
        v.kind = TokenBus::Segment::Kind::Vision; v.ids = {50, 51};
        TokenBus::Segment a;
        a.kind = TokenBus::Segment::Kind::Audio;  a.ids = {60};
        segs.push_back(t); segs.push_back(v); segs.push_back(a);

        std::vector<int32_t> out;
        CHECK(bus.fuse(segs, out));

        const auto pos = [&](int32_t id) {
            return std::find(out.begin(), out.end(), id);
        };
        const auto first_text = std::find(out.begin(), out.end(), text_ids[0]);
        CHECK(first_text != out.end());
        CHECK(pos(Tokenizer::kVisionStartId) != out.end());
        CHECK(pos(Tokenizer::kAudioStartId) != out.end());
        CHECK(pos(Tokenizer::kVisionStartId) < pos(Tokenizer::kAudioStartId));
        CHECK(first_text < pos(Tokenizer::kVisionStartId));
    }

    // ---- A6: budget ---------------------------------------------------------
    TEST("A6: exceeding the budget returns false, not a truncated stream");
    {
        TokenBus small(tok);
        small.set_max_tokens(4);

        std::vector<TokenBus::Segment> segs;
        TokenBus::Segment s;
        s.kind = TokenBus::Segment::Kind::Text;
        s.text = "this is definitely longer than four tokens";
        segs.push_back(s);

        std::vector<int32_t> out;
        CHECK(!small.fuse(segs, out));       // refused
        CHECK(small.max_tokens() == 4);
    }

    // ---- A7: exact budget boundary -----------------------------------------
    TEST("A7: a stream exactly at the budget is accepted");
    {
        TokenBus exact(tok);
        // BOS + the text tokens is the exact size the bus will produce.
        exact.set_max_tokens(static_cast<int64_t>(text_ids.size()) + 1);
        std::vector<TokenBus::Segment> segs;
        TokenBus::Segment s;
        s.kind = TokenBus::Segment::Kind::Text;
        s.text = "hello world";
        segs.push_back(s);

        std::vector<int32_t> out;
        CHECK(exact.fuse(segs, out));
        CHECK(static_cast<int64_t>(out.size()) ==
              static_cast<int64_t>(text_ids.size()) + 1);
        CHECK(exact.last_stats().total == static_cast<int64_t>(out.size()));
    }
}

// =============================================================================
// PART B — MultimodalBridge (no model weights needed)
// =============================================================================
static void part_b_bridge() {
    const int32_t E = 8;
    const int32_t N = 8;

    Tokenizer tok;
    CHECK(tok.build_minimal(512));

    MultimodalBridge bridge;
    CHECK(bridge.init(E, tok));
    CHECK(bridge.ready());
    CHECK(bridge.n_embd() == E);
    CHECK(!bridge.codebook_ready());

    // ---- B1: vision without a codebook must FAIL, not silently drop --------
    TEST("B1: vision without a codebook fails loudly");
    {
        const std::vector<float> vision = basis_vector(E, 0);
        MultimodalBridge::Fusion f = bridge.fuse("look", vision, {});
        CHECK(!f.ok);
        CHECK(!f.error.empty());
        CHECK(f.error.find("codebook") != std::string::npos);
        CHECK(f.ids.empty());
    }

    // ---- B2: width mismatch refused ----------------------------------------
    TEST("B2: a codebook of the wrong width is refused");
    {
        Tensor wrong("wrong", {N, E + 1}, DType::F32);
        CHECK(!bridge.set_codebook(wrong));
        CHECK(!bridge.codebook_ready());
        CHECK(bridge.error().find("width") != std::string::npos);
    }

    // ---- B3..: with a codebook ---------------------------------------------
    Tensor cb = make_basis_codebook(N, E);
    CHECK(bridge.set_codebook(cb));
    CHECK(bridge.codebook_ready());
    CHECK(bridge.codebook_rows() == N);
    CHECK(bridge.codebook_stride() == 8);   // the default dial

    TEST("B3: nearest_token finds an exact row");
    {
        MultimodalBridge b2;
        b2.init(E, tok);
        MultimodalBridge::Config c;
        c.codebook_stride = 1;          // exact scan
        b2.set_config(c);
        CHECK(b2.set_codebook(cb));
        for (int32_t r = 0; r < N && r < E; ++r) {
            const std::vector<float> q = basis_vector(E, r);
            CHECK(b2.nearest_token(q.data()) == r);
        }
        // a null pointer is not a query
        CHECK(b2.nearest_token(nullptr) == -1);
    }

    TEST("B4: quantize maps an [M, E] block to M ids, refusing a partial row");
    {
        MultimodalBridge b2;
        b2.init(E, tok);
        MultimodalBridge::Config c;
        c.codebook_stride = 1;
        b2.set_config(c);
        b2.set_codebook(cb);

        std::vector<float> block;
        const std::vector<float> r0 = basis_vector(E, 0);
        const std::vector<float> r3 = basis_vector(E, 3);
        block.insert(block.end(), r0.begin(), r0.end());
        block.insert(block.end(), r3.begin(), r3.end());

        std::vector<int32_t> ids;
        CHECK(b2.quantize(block, ids));
        CHECK(ids.size() == 2);
        CHECK(ids[0] == 0);
        CHECK(ids[1] == 3);

        // a block that is not a whole number of rows is refused, and out is cleared
        std::vector<float> partial(static_cast<size_t>(E) + 3, 0.0f);
        std::vector<int32_t> ids2 = {999};
        CHECK(!b2.quantize(partial, ids2));
        CHECK(ids2.empty());

        // the empty block is "no image", which is not an error
        std::vector<int32_t> ids3 = {999};
        CHECK(b2.quantize({}, ids3));
        CHECK(ids3.empty());
    }

    TEST("B5: fused vision ids are the quantized ids, wrapped in markers");
    {
        MultimodalBridge b2;
        b2.init(E, tok);
        MultimodalBridge::Config c;
        c.codebook_stride = 1;
        b2.set_config(c);
        b2.set_codebook(cb);

        std::vector<float> vision;
        const std::vector<float> a = basis_vector(E, 1);
        const std::vector<float> b = basis_vector(E, 5);
        vision.insert(vision.end(), a.begin(), a.end());
        vision.insert(vision.end(), b.begin(), b.end());

        MultimodalBridge::Fusion f = b2.fuse("look here", vision, {});
        CHECK(f.ok);
        CHECK(f.vision_tokens_emitted == 2);
        CHECK(f.audio_tokens_emitted == 0);
        CHECK(f.ids[0] == Tokenizer::kBosId);

        const std::vector<int32_t> expect = {Tokenizer::kVisionStartId, 1, 5,
                                             Tokenizer::kVisionEndId};
        CHECK(contains_seq(f.ids, expect));
    }

    TEST("B6: no vision supplied means no vision markers and a zero count");
    {
        MultimodalBridge b2;
        b2.init(E, tok);
        b2.set_codebook(cb);
        MultimodalBridge::Fusion f = b2.fuse("just text", {}, {});
        CHECK(f.ok);
        CHECK(f.vision_tokens_emitted == 0);
        CHECK(!contains_seq(f.ids, {Tokenizer::kVisionStartId}));
        CHECK(f.stats.total == static_cast<int64_t>(f.ids.size()));
    }

    TEST("B7: a non-finite query vector yields -1, not a NaN win");
    {
        MultimodalBridge b2;
        b2.init(E, tok);
        MultimodalBridge::Config c;
        c.codebook_stride = 1;
        b2.set_config(c);
        b2.set_codebook(cb);

        std::vector<float> q = basis_vector(E, 2);
        CHECK(b2.nearest_token(q.data()) == 2);
        q[0] = std::nanf("");
        CHECK(b2.nearest_token(q.data()) == -1);
    }

    TEST("B8: the codebook stride is a real accuracy dial");
    {
        MultimodalBridge exact;
        exact.init(E, tok);
        MultimodalBridge::Config c1;
        c1.codebook_stride = 1;
        exact.set_config(c1);
        exact.set_codebook(cb);

        MultimodalBridge strided;
        strided.init(E, tok);
        MultimodalBridge::Config c2;
        c2.codebook_stride = 2;         // scans rows 0,2,4,6 — row 3 is gone
        strided.set_config(c2);
        strided.set_codebook(cb);

        const std::vector<float> q = basis_vector(E, 3);
        CHECK(exact.nearest_token(q.data()) == 3);
        // Row 3 is never examined, so the dial trades exactness for speed.
        // This is the documented cost of a stride > 1, pinned so it cannot
        // drift into "stride is free".
        CHECK(strided.nearest_token(q.data()) != 3);
    }

    TEST("B9: summary never claims a modality that contributed no tokens");
    {
        MultimodalBridge b2;
        b2.init(E, tok);
        MultimodalBridge::Config c;
        c.codebook_stride = 1;
        b2.set_config(c);
        b2.set_codebook(cb);

        MultimodalBridge::Fusion text_only = b2.fuse("hi", {}, {});
        CHECK(text_only.ok);
        const std::string s1 = b2.summary(text_only);
        CHECK(s1.find("vision 0") != std::string::npos);
        CHECK(s1.find("audio 0") != std::string::npos);

        std::vector<float> vision = basis_vector(E, 0);
        MultimodalBridge::Fusion with_v = b2.fuse("hi", vision, {});
        const std::string s2 = b2.summary(with_v);
        CHECK(s2.find("vision 1") != std::string::npos);

        MultimodalBridge::Fusion bad;
        bad.ok = false;
        bad.error = "boom";
        CHECK(b2.summary(bad).find("boom") != std::string::npos);
    }

    TEST("B10: an uninitialised bridge fails every path rather than guessing");
    {
        MultimodalBridge fresh;
        CHECK(!fresh.ready());
        CHECK(fresh.nearest_token(nullptr) == -1);
        const std::vector<float> v = basis_vector(E, 0);
        MultimodalBridge::Fusion f = fresh.fuse("x", v, {});
        CHECK(!f.ok);
        CHECK(!f.error.empty());
        Tensor cb2 = make_basis_codebook(N, E);
        CHECK(!fresh.set_codebook(cb2));
    }
}

// =============================================================================
// PART C — end to end through the real backbone (gated)
// =============================================================================
static void part_c_end_to_end() {
    const std::string kModelPath = "models/rwkv7-0.1B-ternary.gguf";

    TEST("C1: image embeddings -> tokens -> RWKV -> h[E] -> a head reads it");
    if (!file_exists(kModelPath)) {
        SKIP("models/rwkv7-0.1B-ternary.gguf not present");
        return;
    }
    RwkvModel model;
    if (!model.load(kModelPath)) {
        SKIP(model.error().c_str());
        return;
    }
    GgufLoader gg;
    if (!gg.open(kModelPath)) { SKIP("gguf reopen failed"); return; }
    Tokenizer tok;
    if (!tok.load_from_gguf(gg)) { SKIP("tokenizer load failed"); return; }

    const int32_t E = model.config().n_embd;
    CHECK(E > 0);

    MultimodalBridge bridge;
    CHECK(bridge.init(E, tok));
    // The model's OWN embedding matrix is the codebook — no extra weights.
    CHECK(bridge.set_codebook(model.token_embeddings()));
    CHECK(bridge.codebook_ready());
    CHECK(bridge.codebook_rows() == model.vocab_size());

    // Stand in for a vision encoder's output: a handful of [E] rows. The point
    // of C1 is the JOIN, not the encoder (which has its own tests).
    std::vector<float> fake_vision;
    for (int32_t m = 0; m < 4; ++m) {
        for (int32_t e = 0; e < E; ++e) {
            const float v = std::sin(0.01f * static_cast<float>(m * E + e));
            fake_vision.push_back(v);
        }
    }

    MultimodalBridge::Config cfg;
    cfg.codebook_stride = 16;          // keep the test quick
    bridge.set_config(cfg);

    MultimodalBridge::Fusion f = bridge.fuse("describe the chart", fake_vision, {});
    CHECK(f.ok);
    CHECK(f.vision_tokens_emitted == 4);
    CHECK(f.ids.size() > 4);
    CHECK(f.ids[0] == Tokenizer::kBosId);
    platform::log_info("  %s", bridge.summary(f).c_str());

    // Run the fused stream through the real backbone, keeping the final state.
    RwkvState st;
    model.init_state(st);
    Tensor logits("logits", {model.vocab_size()}, DType::F32);
    Tensor hidden("hidden", {E}, DType::F32);
    for (int32_t id : f.ids) model.forward(id, st, logits, &hidden);

    CHECK(hidden.numel() == E);
    bool finite = true;
    for (int32_t e = 0; e < E; ++e)
        if (!std::isfinite(hidden.f32()[e])) finite = false;
    CHECK(finite);

    // The joint is only real if a HEAD can read the h[E] it produced. This is
    // the mandate's "light up vision.scene": the head now sees a state that
    // came from an image, not from text alone.
    ClassificationHead vision_head;
    CHECK(vision_head.init(E));
    const int32_t si = vision_head.add_label_set(
        "vision.scene", {"indoor", "outdoor", "nature", "urban", "unknown"});
    CHECK(si >= 0);
    ClassificationResult r = vision_head.classify(hidden.f32(), "vision.scene");
    CHECK(!r.top_k.empty());
    CHECK(r.domain == "vision.scene");
    platform::log_info("  vision.scene top1 = %s (%.3f)  [head is UNFITTED — "
                       "base rate, see docs/MULTIMODAL.md]",
                       r.top_k.empty() ? "-" : r.top_k[0].label.c_str(),
                       r.top_k.empty() ? 0.0f : r.top_k[0].probability);
}

// =============================================================================
int main() {
    platform::set_quiet(false);
    platform::log_info("=== test_multimodal (MILESTONE 5) ===");
    part_a_token_bus();
    part_b_bridge();
    part_c_end_to_end();

    platform::log_info("RESULT: %d passed, %d failed, %d skipped",
                       g_passed, g_failed, g_skipped);
    return g_failed == 0 ? 0 : 1;
}
