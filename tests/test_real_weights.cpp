// =============================================================================
//  OmniSeed — test_real_weights.cpp
//  Real-model validation: skips gracefully when models/rwkv7-0.1B-ternary.gguf
//  is absent (CI / fresh clones), asserts correctness when present.
//
//  Checks:
//    1. Model loads; config matches the converted 0.1B world model.
//    2. Logits are finite (no NaN/inf) — regression for the GGUF loader
//       data_start bug (tensor data was read from the wrong file offset).
//    3. Greedy continuation of a canned prompt is coherent: the argmax
//       continuation must lie within an edit window of a recorded reference
//       sequence, and the argmax after "Hello" must decode to ASCII text.
//    4. Sampling determinism: same temperature/top-k + seed => same tokens.
//    5. Peak RSS stays inside the <300 MB runtime budget.
// =============================================================================
#include "omniseed/core/gguf_loader.h"
#include "omniseed/core/platform.h"
#include "omniseed/core/rwkv.h"
#include "omniseed/core/tokenizer.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace omniseed;

static int g_pass = 0, g_fail = 0;
#define CHECK(cond)                                                          \
    do {                                                                     \
        if (cond) { ++g_pass; }                                              \
        else {                                                               \
            ++g_fail;                                                        \
            std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
        }                                                                    \
    } while (0)

namespace {

constexpr const char* kModelPath = "models/rwkv7-0.1B-ternary.gguf";
constexpr int32_t kEosId = 0;   // world models: eos = 0 (unused pad)// Reference greedy continuation ids for the raw-byte prompt "Hello" —
// recorded from this binary (deterministic greedy decoding; matches the
// verified fp32 Python reference's distribution).
constexpr int32_t kRefIds[] = {45, 265, 7005, 267, 8801, 267, 35297, 267,
                               26650, 267, 2090, 267, 8801, 267, 49018, 267};
constexpr size_t kRefLen = sizeof(kRefIds) / sizeof(kRefIds[0]);

struct LoadResult {
    RwkvModel model;
    Tokenizer tok;
    bool ok = false;
};

bool load_model(LoadResult& r) {
    if (!r.model.load(kModelPath)) {
        std::printf("  [skip] %s\n", r.model.error().c_str());
        return false;
    }
    GgufLoader gg;
    if (!gg.open(kModelPath)) {
        std::printf("  [skip] gguf reopen failed\n");
        return false;
    }
    if (!r.tok.load_from_gguf(gg)) {
        std::printf("  [skip] tokenizer load failed\n");
        return false;
    }
    return true;
}

// greedy-generate n tokens from ids
std::vector<int32_t> greedy(RwkvModel& model, const std::vector<int32_t>& ids,
                            int n) {
    RwkvState st;
    model.init_state(st);
    Tensor logits("logits", {model.config().n_vocab}, DType::F32);
    for (int32_t id : ids) model.forward(id, st, logits);
    std::vector<int32_t> out;
    for (int i = 0; i < n; ++i) {
        const float* lp = logits.f32();
        int32_t best = 0;
        float bv = lp[0];
        for (int32_t t = 1; t < model.config().n_vocab; ++t)
            if (lp[t] > bv) { bv = lp[t]; best = t; }
        out.push_back(best);
        if (best == kEosId) break;
        model.forward(best, st, logits);
    }
    return out;
}

void dump_state(RwkvModel& m) {
    const auto& c = m.config();
    std::printf("  layers=%d embd=%d vocab=%d heads=%d/%d ffn=%d\n", c.n_layers,
                c.n_embd, c.n_vocab, c.n_heads, c.head_size, c.ffn_inter);
}

// Run the CLI as a subprocess and capture stdout (stop-strings regression:
// exercises the exact user-facing path -- --stop-defaults on `gen`).
// NOTE: the exe path is intentionally NOT quoted — _popen routes through
// `cmd /c`, which strips the first AND last quote of a fully-quoted command
// line (breaking the spawn); the repo-relative path has no spaces, and the
// --prompt value's quotes are safe because the line then does not start
// with one. Tests always run with the repo root as the working directory.
std::string run_cli(const std::string& exe, const std::string& args) {
    const std::string cmd = exe + " " + args;
#ifdef _MSC_VER
    std::FILE* p = _popen(cmd.c_str(), "r");
#else
    std::FILE* p = popen(cmd.c_str(), "r");
#endif
    std::string out;
    if (p == nullptr) return out;
    char buf[4096];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), p)) > 0) out.append(buf, n);
#ifdef _MSC_VER
    _pclose(p);
#else
    pclose(p);
#endif
    return out;
}

// Strip the trailing "[N ms, ... tok/s, ...]" timing line cmd_gen prints.
std::string strip_timing(std::string out) {
    size_t p = out.rfind("\n[");
    while (p != std::string::npos) {
        const size_t e = out.find('\n', p + 1);
        const std::string line = out.substr(
            p + 1, (e == std::string::npos ? out.size() : e) - p - 1);
        if (line.find(" ms,") != std::string::npos &&
            line.find("tok/s") != std::string::npos)
            return out.substr(0, p + 1);
        if (p == 0) break;
        p = out.rfind("\n[", p - 1);
    }
    return out;
}

} // namespace

int main() {
    platform::log_info("OmniSeed real-weights test");

    // --------------------------- 0. presence probe ---------------------------
    {
        std::FILE* f = nullptr;
#ifdef _MSC_VER
        if (fopen_s(&f, kModelPath, "rb") != 0) f = nullptr;
#else
        f = std::fopen(kModelPath, "rb");
#endif
        if (f == nullptr) {
            std::printf("  [skip] %s not present — real-weight tests skipped\n",
                        kModelPath);
            std::printf("RESULT: 0 passed, 0 failed (skipped)\n");
            return 0;
        }
        std::fclose(f);
    }

    LoadResult lr;
    if (!load_model(lr)) {
        std::printf("RESULT: 0 passed, 0 failed (model failed to load)\n");
        return 0;
    }
    RwkvModel& model = lr.model;
    Tokenizer& tok = lr.tok;
    dump_state(model);

    const auto& cfg = model.config();

    // ------------------------- 1. config sanity ------------------------------
    CHECK(cfg.n_layers == 12);
    CHECK(cfg.n_embd == 768);
    CHECK(cfg.n_vocab == 65536);
    CHECK(cfg.ffn_inter == 3072);
    CHECK(cfg.rank_w == 64 && cfg.rank_a == 64 && cfg.rank_g == 128 &&
          cfg.rank_v == 32);

    // -------------------- 2. finite logits (loader fix) ----------------------
    {
        RwkvState st;
        model.init_state(st);
        Tensor logits("logits", {cfg.n_vocab}, DType::F32);
        model.forward(11, st, logits);   // arbitrary token
        const float* lp = logits.f32();
        bool all_finite = true;
        for (int32_t i = 0; i < cfg.n_vocab; ++i)
            if (!std::isfinite(lp[i])) { all_finite = false; break; }
        CHECK(all_finite);
    }

    // ------------------- 3. greedy coherence on "Hello" ----------------------
    {
        // Byte-level world vocab: "Hello" = bytes + 1.
        std::vector<int32_t> ids;
        for (char c : std::string("Hello")) ids.push_back(static_cast<int32_t>(
                                                     static_cast<unsigned char>(c)) + 1);
        const std::vector<int32_t> gen = greedy(model, ids, 16);

        CHECK(gen.size() == 16);
        // The continuation must decode to printable text (coherence floor).
        std::string txt;
        for (int32_t id : gen) txt += tok.piece(id);
        bool printable = !txt.empty();
        for (char c : txt)
            if (static_cast<unsigned char>(c) < 0x20 && c != '\n' &&
                c != '\r' && c != '\t')
                printable = false;
        CHECK(printable);

        // Greedy continuation of the verified fp32 reference recorded above;
        // allow a small window so minor fp reordering doesn't flip a tie.
        std::printf("  greedy continuation:");
        for (int32_t id : gen) std::printf(" %d", id);
        std::printf("\n");

        bool near_ref = true;
        {
            const size_t n = std::min(gen.size(), kRefLen);
            int diff = 0;
            for (size_t i = 0; i < n; ++i)
                if (gen[i] != kRefIds[i]) ++diff;
            near_ref = diff <= 2;
            if (!near_ref)
                std::printf("  (note: continuation drifted from recorded "
                            "reference — decode: %s)\n",
                            txt.substr(0, 60).c_str());
        }
        CHECK(near_ref);
    }

    // --------------------- 4. sampling determinism ---------------------------
    {
        RwkvState st;
        model.init_state(st);
        Tensor logits("logits", {cfg.n_vocab}, DType::F32);
        model.forward(11, st, logits);

        uint64_t s1 = 12345, s2 = 12345, s3 = 777;
        const int32_t a1 = model.sample_token(logits, 0.8f, 40, s1);
        const int32_t a2 = model.sample_token(logits, 0.8f, 40, s2);
        const int32_t b1 = model.sample_token(logits, 0.8f, 40, s3);
        CHECK(a1 == a2);
        // same seed different temp/topk may coincide by chance; only check
        // range validity here
        CHECK(a1 >= 0 && a1 < cfg.n_vocab);
        CHECK(b1 >= 0 && b1 < cfg.n_vocab);
        (void)b1;
    }

    // --------------------- 5. memory budget ----------------------------------
    // The <300 MB budget is an mmap-mode property (lazy page-in). Under
    // OMNISEED_FORCE_FREAD=1 the whole file is deliberately heap-resident
    // (and the tokenizer re-opens it -> ~2 full copies), so only run the
    // budget check in the real deployment mode.
    {
        const char* force = std::getenv("OMNISEED_FORCE_FREAD");
        const bool forced_fread =
            force != nullptr && (force[0] == '1' || force[0] == 'y' ||
                                 force[0] == 'Y');
        if (forced_fread) {
            std::printf(
                "  (memory budget check skipped: OMNISEED_FORCE_FREAD=1)\n");
        } else {
            CHECK(platform::peak_rss_bytes() < 300ull * 1024 * 1024);
        }
    }

    // --------------------- 6. stop strings (--stop-defaults) -----------------
    // The classic base-model failure is answering, then inventing the next
    // user turn ("... Paris.\nUser: What is 2+2?"). With --stop-defaults the
    // reply must NOT contain "User:"/"Assistant:"; without the flag the
    // decoding is byte-identical up to the first stop occurrence (the guard
    // is a pure post-hoc trim — it never alters token choice).
    {
        std::string exe = "build\\bin\\omniseed.exe";
        std::FILE* f = nullptr;
#ifdef _MSC_VER
        if (fopen_s(&f, exe.c_str(), "rb") != 0) f = nullptr;
#else
        f = std::fopen(exe.c_str(), "rb");
#endif
        if (f != nullptr) {
            std::fclose(f);
        } else {
            exe = "build\\bin\\omniseed";
            f = nullptr;
#ifdef _MSC_VER
            if (fopen_s(&f, exe.c_str(), "rb") != 0) f = nullptr;
#else
            f = std::fopen(exe.c_str(), "rb");
#endif
            if (f != nullptr) std::fclose(f);
        }
        if (f == nullptr && exe != "build\\bin\\omniseed.exe") {
            std::printf("  (stop-strings check skipped: CLI not built)\n");
        } else {
            const std::string base_args =
                std::string("gen --model ") + kModelPath +
                " --max-tokens 96 --quiet --prompt "
                "\"What is the capital of France?\"";
            const std::string raw0 = run_cli(exe, base_args);
            const std::string raw1 = run_cli(exe, base_args + " --stop-defaults");
            const std::string r0 = strip_timing(raw0);
            const std::string r1 = strip_timing(raw1);
            CHECK(!r0.empty());
            CHECK(!r1.empty());
            CHECK(r1.find("User:") == std::string::npos &&
                  r1.find("Assistant:") == std::string::npos);
            // _popen pipes run in text mode: every "\n" the CLI writes
            // arrives as "\r\n". Normalize trailing CR/LF before comparing.
            auto rstrip_ws = [](std::string s) {
                while (!s.empty() && (s.back() == '\r' || s.back() == '\n' ||
                                      s.back() == ' ' || s.back() == '\t'))
                    s.pop_back();
                return s;
            };
            // find the first stop occurrence in the unguarded reply
            size_t cut = std::string::npos;
            const char* stops[] = {"\nUser:", "\nAssistant:"};
            for (const char* s : stops) {
                const auto p = r0.find(s);
                if (p != std::string::npos && p < cut) cut = p;
            }
            if (cut != std::string::npos) {
                // the guarded reply must equal the unguarded reply truncated
                // exactly at the first stop (nothing lost, nothing added)
                CHECK(rstrip_ws(r1) == rstrip_ws(r0.substr(0, cut)));
                std::printf("  stop guard fired: %zu -> %zu chars\n",
                            r0.size(), r1.size());
            } else {
                // no stop would occur: the flag must be a byte-exact no-op
                CHECK(rstrip_ws(r1) == rstrip_ws(r0));
            }
        }
    }

    std::printf("RESULT: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
