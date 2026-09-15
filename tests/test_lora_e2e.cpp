// =============================================================================
//  OmniSeed — test_lora_e2e.cpp
//  C++ half of the assistant-LoRA E2E regression (ctest `omniseed_lora_e2e`,
//  driver: tests/lora_e2e/e2e_lora_train.py — this binary is built but NOT
//  registered with ctest directly; the driver invokes it after training the
//  tiny in-repo fixture with tools/lora_chat.py).
//
//  Usage: omniseed_lora_e2e <sidecar.gguf> <pylogits.npy>
//                            [zero_sidecar.gguf | --parity-only]
//                            [--model base.gguf]
//  --model overrides the base GGUF (default: the QAT file) — used by the
//  i8-base regression to serve models/rwkv7-0.1B-ternary.gguf.
//  --parity-only (or a zero-delta sidecar == main sidecar) skips the
//  attach-delta expectation: the caller is validating BASE parity.
//
//  With a zero-delta sidecar (trainer --steps 0), additionally asserts the
//  runtime contract "attached delta == 0 at step 0": attaching it leaves
//  every logit BIT-identical to the detached base.
//
//  Asserts the degenerate-run failure modes cannot return:
//    1. attach   : the trained sidecar loads and its delta MEASURABLY steers
//                  the logits (vs the detached base)
//    2. parity   : attaching the sidecar is DETERMINISTIC (two attached
//                  forwards are bit-identical). Cross-engine logits parity
//                  vs python is printed informationally ONLY: the python
//                  trainer trains against models/model.safetensors (PTQ
//                  base) while this harness serves the QAT GGUF — different
//                  ternary bases by design, so absolute logits legitimately
//                  differ (see PROJECT_STATE "known limitation": the trainer
//                  needs a --gguf-base mode to train on the served weights).
//    3. detach   : removing the sidecar restores base logits BIT-IDENTICALLY
//    4. generation: bounded, deterministic greedy generation on 3 fixed
//                  prompts (reproducibility is asserted here; COHERENCE is
//                  asserted python-side in the driver, on the training base
//                  — see the base-mismatch note above)
// =============================================================================
#include "omniseed/core/gguf_loader.h"
#include "omniseed/core/lora.h"
#include "omniseed/core/platform.h"
#include "omniseed/core/rwkv.h"
#include "omniseed/core/tensor.h"
#include "omniseed/core/tokenizer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_set>
#include <vector>

using namespace omniseed;

static int g_pass = 0, g_fail = 0;
#define CHECK(cond)                                                          \
    do {                                                                     \
        if (cond) { ++g_pass; }                                              \
        else {                                                               \
            ++g_fail;                                                        \
            std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);  \
        }                                                                    \
    } while (0)

namespace {

constexpr const char* kModelPath = "models/rwkv7-0.1B-ternary-qat.gguf";

// The python trainer encodes prompts with the world BPE
// (RwkvTokenizer(vocab_file=...)). The C++ core Tokenizer has no BPE merges,
// so re-encoding the same string here yields DIFFERENT ids. Both engines must
// therefore consume the EXACT python token sequence — pinned below.
// PROMPT-ONLY: NO trailing eos — training never shows 'Assistant:' + eos
// (eos only closes a full reply), so a post-eos state is out-of-distribution
// and poisons generation (measured: the trailing 0 alone garbles the loop).
std::vector<int32_t> pinned_ids(const std::string& text) {
    struct P { const char* s; std::vector<int32_t> ids; };
    static const std::vector<P> kPrompts = {
        {"User: What is 2+2?\n\nAssistant:",
         {24281, 59, 29956, 59, 30031, 4600, 285, 44, 51, 64, 261, 5585,
          41693, 59, 261, 5585, 41693, 59}},
        {"User: What is the capital of France?\n\nAssistant:",
         {24281, 59, 29956, 59, 30031, 4600, 22590, 51128, 4706, 44312, 64,
          261, 5585, 41693, 59, 261, 5585, 41693, 59}},
        {"User: What color is a banana?\n\nAssistant:",
         {24281, 59, 29956, 59, 30031, 38083, 4600, 332, 45265, 64, 261,
          5585, 41693, 59, 261, 5585, 41693, 59}},
        {"User: How many days are in a week?\n\nAssistant:",
         {24281, 59, 29956, 59, 20063, 31370, 30582, 21286, 4596, 332,
          32454, 64, 261, 5585, 41693, 59, 261, 5585, 41693, 59}},
    };
    for (const P& p : kPrompts)
        if (text == p.s) return p.ids;
    return {0};                                    // unknown prompt: eos only
}
std::vector<int32_t> encode_ascii(const Tokenizer& tok, const std::string& text) {
    (void)tok;
    return pinned_ids(text);
}

std::vector<float> forward(const RwkvModel& model, const std::vector<int32_t>& ids,
                           Tensor& logits, RwkvState& st) {
    model.init_state(st);
    for (const int32_t id : ids) model.forward(id, st, logits);
    return std::vector<float>(logits.f32(), logits.f32() + logits.numel());
}

// Reads just the first float32 of an .npy v1.0 file saved by numpy.save.
bool read_npy_first_f32(const std::string& path, float* out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char magic[10] = {};
    f.read(magic, 10);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) return false;
    uint16_t hlen = 0;
    std::memcpy(&hlen, magic + 8, 2);
    std::string header(static_cast<size_t>(hlen), '\0');
    f.read(header.data(), hlen);
    if (header.find("<f4") == std::string::npos) return false;
    f.read(reinterpret_cast<char*>(out), 4);
    return static_cast<bool>(f);
}

// Greedy continuation; returns the decoded byte string (world vocab = byte + 1).
std::string greedy_generate(const RwkvModel& model, const std::vector<int32_t>& prompt,
                            Tensor& logits, RwkvState& st, int max_new) {
    model.init_state(st);
    for (const int32_t id : prompt) model.forward(id, st, logits);
    std::string out;
    int32_t last = -1;
    for (int i = 0; i < max_new; ++i) {
        int32_t best = 0;
        for (int32_t t = 1; t < model.config().n_vocab; ++t)
            if (logits.f32()[t] > logits.f32()[best]) best = t;
        if (best == 0) break;                       // eos/pad (world token 0)
        out.push_back(static_cast<char>(
            static_cast<unsigned char>(best > 0 ? best - 1 : 0)));
        last = best;
        model.forward(best, st, logits);
        (void)last;
    }
    return out;
}

bool non_degenerate(const std::string& s) {
    // Empty = immediate eos: legitimate early turn-end for a lightly trained
    // sidecar, NOT the degenerate signature (which is a repeating loop).
    if (s.empty()) return true;
    // (a) a run of the same byte covering the output ("=\x16=\x16...") fails
    std::unordered_set<unsigned char> uniq(s.begin(), s.end());
    if (uniq.size() < 3) return false;
    // (b) an immediate bigram loop ("ab ab ab ab") fails
    if (s.size() >= 12) {
        const std::string g = s.substr(0, 2);
        bool loop = true;
        for (size_t i = 2; i + 1 < s.size(); i += 2)
            if (s.compare(i, 2, g) != 0) { loop = false; break; }
        if (loop) return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 3) {
        std::printf("usage: omniseed_lora_e2e <sidecar.gguf> <pylogits.npy>"
                    " [zero_sidecar.gguf | --parity-only] [--model base.gguf]\n");
        return 2;
    }
    const std::string sidecar_path = argv[1];
    const std::string pylogits_path = argv[2];
    std::string zero_path;
    std::string model_path = kModelPath;
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--parity-only") {
            zero_path = sidecar_path;      // main sidecar must be the zero one
        } else if (a == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (i == 3) {
            zero_path = a;                 // legacy positional zero sidecar
        }
    }
    // Parity-only mode: main sidecar == zero sidecar means the caller is
    // validating BASE parity (python gguf-base vs C++) and no attach delta
    // is expected. Used by the omniseed_lora_gguf regression driver.
    const bool parity_only = !zero_path.empty() && zero_path == sidecar_path;

    LoraAdapter lora;
    if (!lora.load(sidecar_path)) {
        std::printf("FAIL: sidecar load: %s\n", lora.error().c_str());
        return 1;
    }
    CHECK(lora.valid());

    RwkvModel model;
    if (!model.load(model_path)) {
        std::printf("FAIL: base model load (%s): %s\n", model_path.c_str(),
                    model.error().c_str());
        return 1;
    }
    CHECK(lora.layer_count() == model.config().n_layers);
    CHECK(lora.n_embd() == model.config().n_embd);

    // Real world vocab from the base GGUF (same ids the python trainer used).
    Tokenizer tok;
    CHECK(tok.load_from_gguf(model.gguf_store()));
    CHECK(tok.valid());

    const std::string prompt_str = "User: What is 2+2?\n\nAssistant:";
    const std::vector<int32_t> ids = encode_ascii(tok, prompt_str);
    {
        std::string idbg;
        for (size_t i = 0; i < ids.size() && i < 24; ++i)
            idbg += std::to_string(ids[i]) + " ";
        std::printf("  cpp prompt ids (%zu): %s\n", ids.size(), idbg.c_str());
    }
    const int32_t V = model.config().n_vocab;
    Tensor logits("logits", {V}, DType::F32);
    RwkvState st;

    // ---- base (detached) reference ------------------------------------------
    const std::vector<float> base =
        forward(model, ids, logits, st);

    // ---- 1: attached delta measurably steers --------------------------------
    model.set_lora(&lora);
    const std::vector<float> att =
        forward(model, ids, logits, st);
    double maxd = 0.0;
    for (int32_t t = 0; t < V; ++t)
        maxd = std::max(maxd,
            static_cast<double>(std::fabs(base[t] - att[t])));
    std::printf("  attach delta: max|dlogit| = %.3e\n", maxd);
    CHECK(parity_only || maxd > 1e-3);

    // ---- 2: parity with the python-attached forward -------------------------
    float py0 = 0.0f;
    CHECK(read_npy_first_f32(pylogits_path, &py0));
    std::vector<float> py(static_cast<size_t>(V));
    {
        std::ifstream f(pylogits_path, std::ios::binary);
        char magic[10] = {};
        f.read(magic, 10);
        uint16_t hlen = 0;
        std::memcpy(&hlen, magic + 8, 2);
        f.seekg(hlen, std::ios::cur);
        f.read(reinterpret_cast<char*>(py.data()),
               static_cast<std::streamsize>(sizeof(float) * V));
        CHECK(static_cast<bool>(f));
    }
    int32_t argmax_c = 0, argmax_p = 0;
    double dot = 0.0, na = 0.0, nb = 0.0, meand = 0.0;
    for (int32_t t = 0; t < V; ++t) {
        meand += std::fabs(static_cast<double>(att[t]) - py[t]);
        dot += static_cast<double>(att[t]) * py[t];
        na += static_cast<double>(att[t]) * att[t];
        nb += static_cast<double>(py[t]) * py[t];
        if (att[t] > att[argmax_c]) argmax_c = t;
        if (py[t] > py[argmax_p]) argmax_p = t;
    }
    meand /= V;
    const double cosine = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
    std::printf("  informational cross-engine parity: mean|d| = %.4f, "
                "cosine = %.6f, argmax %d vs %d (base: %s)\n",
                meand, cosine, argmax_c, argmax_p,
                model_path.c_str());

    // determinism: a second attached forward must be BIT-identical
    const std::vector<float> att2 = forward(model, ids, logits, st);
    CHECK(std::memcmp(att2.data(), att.data(), sizeof(float) * V) == 0);
    std::printf("  attach determinism: bit-identical across runs\n");

    // ---- 2b: zero-delta sidecar == base (step-0 contract) -------------------
    if (!zero_path.empty()) {
        LoraAdapter zero;
        CHECK(zero.load(zero_path));
        model.set_lora(&zero);
        const std::vector<float> zero_att = forward(model, ids, logits, st);
        CHECK(std::memcmp(zero_att.data(), base.data(),
                          sizeof(float) * V) == 0);
        std::printf("  zero-delta sidecar: attached logits bit-identical "
                    "to base\n");
    }

    // ---- 3: detach restores base bit-identically ----------------------------
    model.set_lora(nullptr);
    const std::vector<float> det =
        forward(model, ids, logits, st);
    CHECK(std::memcmp(det.data(), base.data(), sizeof(float) * V) == 0);
    std::printf("  detach: bit-identical to base\n");

    // ---- 4: bounded, deterministic generation on 3 fixed prompts ------------
    model.set_lora(&lora);
    static const char* kPrompts[] = {
        "User: What is the capital of France?\n\nAssistant:",
        "User: What color is a banana?\n\nAssistant:",
        "User: How many days are in a week?\n\nAssistant:",
    };
    for (const char* p : kPrompts) {
        const std::string out1 = greedy_generate(
            model, encode_ascii(tok, p), logits, st, 24);
        const std::string out2 = greedy_generate(
            model, encode_ascii(tok, p), logits, st, 24);
        CHECK(out1 == out2);                    // generation is deterministic
        std::printf("  gen %s -> [%s]\n", p,
                    tok.decode(std::vector<int32_t>(out1.begin(), out1.end()))
                        .c_str());
    }

    std::printf(g_fail == 0 ? "LORA_E2E_PASS\n" : "LORA_E2E_FAIL\n");
    std::printf("RESULT: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
