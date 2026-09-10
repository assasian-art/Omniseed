// =============================================================================
//  OmniSeed — test_lora.cpp
//  Assistant-behavior LoRA sidecar (Phase 14, 2B) runtime validation.
//
//  Uses NO Python: writes a minimal omniseed-lora GGUF sidecar by hand
//  (v2 container, f16 A/B tensors), then checks:
//    1. Sidecar load + geometry metadata (rank/layer_count/n_embd/scaling).
//    2. factors() returns correctly-shaped views for all 6 targets x layers,
//       and empty views for untargeted/bogus pairs.
//    3. apply_lora matches a scalar fp32 reference (y += s * B(A·x)).
//    4. Malformed sidecars are rejected (wrong architecture, truncated data).
//    5. REAL-MODEL A/B (skips when models/rwkv7-0.1B-ternary-qat.gguf is
//       absent, like test_qat_ab): with a trained sidecar attached, forward()
//       logits differ from the detached baseline; detaching restores the
//       baseline BIT-IDENTICALLY (the nullptr path is a true no-op).
//
//  Registered as ctest `omniseed_lora` (WD = repo root).
// =============================================================================
#include "omniseed/core/gguf_format.h"
#include "omniseed/core/gguf_loader.h"
#include "omniseed/core/lora.h"
#include "omniseed/core/platform.h"
#include "omniseed/core/rwkv.h"
#include "omniseed/core/tensor.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace omniseed;

static int g_pass = 0, g_fail = 0;
#define CHECK(cond)                                                          \
    do {                                                                     \
        if (cond) { ++g_pass; }                                              \
        else {                                                               \
            ++g_fail;                                                        \
            std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);   \
        }                                                                    \
    } while (0)

namespace {

constexpr const char* kQatModelPath = "models/rwkv7-0.1B-ternary-qat.gguf";

// ---------------------------------------------------------------------------
// Minimal LE GGUF writer (v2) — just enough for the sidecar layout.
// ---------------------------------------------------------------------------
class MiniGguf {
public:
    void kv_string(const std::string& k, const std::string& v) {
        str(k); i32(static_cast<int32_t>(gguf::GgufType::STRING)); str(v);
    }
    void kv_u64(const std::string& k, uint64_t v) {
        str(k); i32(static_cast<int32_t>(gguf::GgufType::UINT64)); u64(v);
    }
    void kv_i32(const std::string& k, int32_t v) {
        str(k); i32(static_cast<int32_t>(gguf::GgufType::INT32)); i32(v);
    }
    void kv_f64(const std::string& k, double v) {
        str(k); i32(static_cast<int32_t>(gguf::GgufType::FLOAT64)); u64(0);
        uint8_t* tail = buf_.data() + buf_.size() - 8;
        std::memcpy(tail, &v, 8);              // little-endian on all CI hosts
    }

    // rows/cols are OmniSeed row-major; GGUF ne[] is stored reversed.
    void tensor(const std::string& name, int64_t rows, int64_t cols,
                const std::vector<uint16_t>& f16) {
        names_.push_back(name);
        ne_.push_back({cols, rows});
        std::vector<uint8_t> bytes(f16.size() * 2);
        std::memcpy(bytes.data(), f16.data(), f16.size() * 2);
        payloads_.push_back(std::move(bytes));
    }

    bool write(const std::string& path, bool with_dir = true,
               bool with_payloads = true) {
        std::vector<uint8_t> dir;
        auto dstr = [&dir](const std::string& s) {
            const uint64_t n = s.size();
            for (int i = 0; i < 8; ++i) dir.push_back(
                static_cast<uint8_t>((n >> (8 * i)) & 0xff));
            dir.insert(dir.end(), s.begin(), s.end());
        };
        auto du32 = [&dir](uint32_t v) {
            for (int i = 0; i < 4; ++i) dir.push_back(
                static_cast<uint8_t>((v >> (8 * i)) & 0xff));
        };
        auto du64 = [&dir](uint64_t v) {
            for (int i = 0; i < 8; ++i) dir.push_back(
                static_cast<uint8_t>((v >> (8 * i)) & 0xff));
        };
        auto di32 = [&](int32_t v) { du32(static_cast<uint32_t>(v)); };

        uint64_t off = 0;
        for (size_t t = 0; t < names_.size(); ++t) {
            dstr(names_[t]);
            du32(2); du64(static_cast<uint64_t>(ne_[t][0]));
            du64(static_cast<uint64_t>(ne_[t][1]));
            di32(static_cast<int32_t>(gguf::GgufDType::F16));
            du64(off);
            off += static_cast<uint64_t>(payloads_[t].size());
        }

        std::ofstream f(path, std::ios::binary);
        if (!f) return false;
        auto put = [&f](const std::vector<uint8_t>& b) {
            f.write(reinterpret_cast<const char*>(b.data()),
                    static_cast<std::streamsize>(b.size()));
        };
        put(buf_);
        if (with_dir) put(dir);
        if (with_payloads)
            for (const auto& p : payloads_) put(p);
        return static_cast<bool>(f);
    }

    std::vector<uint8_t> buf_;               // header + metadata (build here)
    std::vector<std::string> names_;
    std::vector<std::array<int64_t, 2>> ne_;
    std::vector<std::vector<uint8_t>> payloads_;   // raw f16 bytes

    // ---- primitive emitters into buf_ (header/metadata) --------------------
    void str(const std::string& s) {
        const uint64_t n = s.size();
        for (int i = 0; i < 8; ++i) buf_.push_back(
            static_cast<uint8_t>((n >> (8 * i)) & 0xff));
        buf_.insert(buf_.end(), s.begin(), s.end());
    }
    void u64(uint64_t v) {
        for (int i = 0; i < 8; ++i) buf_.push_back(
            static_cast<uint8_t>((v >> (8 * i)) & 0xff));
    }
    void i32(int32_t v) {
        const uint32_t u = static_cast<uint32_t>(v);
        for (int i = 0; i < 4; ++i) buf_.push_back(
            static_cast<uint8_t>((u >> (8 * i)) & 0xff));
    }
};

// Deterministic pattern: A [rank, in], B [out, rank] with exact-in-f16 values.
// Zero entries exercise the zero-skip fast paths.
uint16_t enc_a(int64_t j, int64_t c) {
    if ((c + j) % 5 == 0) return float_to_half(0.0f);
    return float_to_half(((j % 4) - 1) / 16.0f + (c % 3) / 32.0f);
}
uint16_t enc_b(int64_t i, int64_t j) {
    if ((i + j) % 7 == 0) return float_to_half(0.0f);
    return float_to_half(((i % 3) - 1) / 8.0f + (j % 2) / 4.0f);
}

// Build a full sidecar: L layers, 6 targeted linears each.
MiniGguf build_sidecar(int32_t L, int32_t E, int32_t rank, double scaling) {
    MiniGguf w;
    auto& b = w.buf_;
    const auto push = [&b](uint32_t v) {
        for (int i = 0; i < 4; ++i) b.push_back(
            static_cast<uint8_t>((v >> (8 * i)) & 0xff));
    };
    push(gguf::GGUF_MAGIC);
    push(2);                                 // GGUF v2
    // tensor_count + kv_count placeholders (backpatched)
    const size_t n_tensors_at = b.size(); b.resize(b.size() + 8);
    const size_t n_kv_at = b.size();      b.resize(b.size() + 8);

    int32_t n_kv = 0;
    w.kv_string("general.architecture", "omniseed-lora"); ++n_kv;
    w.kv_string("general.name", "lora-test");             ++n_kv;
    w.kv_u64("lora.rank", static_cast<uint64_t>(rank));   ++n_kv;
    w.kv_f64("lora.alpha", scaling * rank);               ++n_kv;
    w.kv_f64("lora.scaling", scaling);                    ++n_kv;
    w.kv_u64("lora.layer_count", static_cast<uint64_t>(L)); ++n_kv;
    w.kv_u64("lora.n_embd", static_cast<uint64_t>(E));    ++n_kv;
    w.kv_u64("lora.base_vocab", 65536);                   ++n_kv;
    w.kv_i32("lora.trained_steps", 42);                   ++n_kv;

    const int32_t n_tensors = L * 12;
    std::memcpy(b.data() + n_tensors_at, &n_tensors, 4);
    std::memcpy(b.data() + n_kv_at, &n_kv, 4);

    for (int32_t l = 0; l < L; ++l) {
        for (const char* t : {"att.receptance", "att.key", "att.value",
                              "att.output", "ffn.key", "ffn.value"}) {
            const std::string pre =
                "lora." + std::to_string(l) + "." + t;
            std::vector<uint16_t> a(static_cast<size_t>(rank) * E);
            std::vector<uint16_t> bm(static_cast<size_t>(E) * rank);
            for (int32_t j = 0; j < rank; ++j)
                for (int32_t c = 0; c < E; ++c)
                    a[static_cast<size_t>(j) * E + c] = enc_a(j, c);
            for (int32_t i = 0; i < E; ++i)
                for (int32_t j = 0; j < rank; ++j)
                    bm[static_cast<size_t>(i) * rank + j] = enc_b(i, j);
            w.tensor(pre + ".a", rank, E, a);      // A [rank, in]
            w.tensor(pre + ".b", E, rank, bm);     // B [out, rank]
        }
    }
    return w;
}

bool write_file(const std::string& path, const std::vector<uint8_t>& bytes) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
    return static_cast<bool>(f);
}

std::string temp_dir() {
    const char* t = std::getenv("TEMP");
    if (t == nullptr) t = std::getenv("TMPDIR");
    return (t != nullptr) ? std::string(t) : std::string(".");
}

// scalar reference: y[i] += s * sum_j B[i,j] * (A[j]·x)
void reference_apply(const std::vector<uint16_t>& B,
                     const std::vector<uint16_t>& A,
                     float s, const std::vector<float>& x,
                     std::vector<float>& y, int64_t out, int64_t in,
                     int64_t r) {
    for (int64_t i = 0; i < out; ++i) {
        float acc = 0.0f;
        for (int64_t j = 0; j < r; ++j) {
            const float bv = half_to_float(B[static_cast<size_t>(i) * r + j]);
            float dot = 0.0f;
            for (int64_t c = 0; c < in; ++c)
                dot += half_to_float(A[static_cast<size_t>(j) * in + c]) * x[static_cast<size_t>(c)];
            acc += bv * dot;
        }
        y[static_cast<size_t>(i)] += s * acc;
    }
}

} // namespace

// ---------------------------------------------------------------------------
int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // crash-point diagnostics
    platform::log_info("OmniSeed LoRA sidecar runtime test");
    const std::string dir = temp_dir();
    const std::string sidecar_path = dir + "/omniseed-lora-test.gguf";
    const std::string bad_arch_path = dir + "/omniseed-lora-bad-arch.gguf";
    const std::string trunc_path = dir + "/omniseed-lora-trunc.gguf";

    // ---- 1-3: synthetic sidecar: load, geometry, factor views, math --------
    constexpr int32_t kL = 2, kE = 64, kRank = 8;
    constexpr float kScaling = 2.0f;
    {
        MiniGguf w = build_sidecar(kL, kE, kRank, kScaling);
        CHECK(w.write(sidecar_path));

        LoraAdapter lora;
        const bool loaded = lora.load(sidecar_path);
        if (!loaded)
            std::printf("  [diag] load error: %s\n", lora.error().c_str());
        CHECK(loaded);
        CHECK(lora.valid());
        CHECK(lora.rank() == kRank);
        CHECK(lora.layer_count() == kL);
        CHECK(lora.n_embd() == kE);
        CHECK(std::fabs(lora.scaling() - kScaling) < 1e-6);

        static const char* kTargets[] = {"att.receptance", "att.key",
                                         "att.value", "att.output",
                                         "ffn.key", "ffn.value"};
        for (int32_t l = 0; l < kL; ++l) {
            for (const char* tname : kTargets) {
                const bool ffn = std::strncmp(tname, "ffn.", 4) == 0;
                const LoraTarget tgt =
                    !ffn ? (tname[4] == 'r' ? LoraTarget::Receptance
                            : tname[4] == 'k' ? LoraTarget::Key
                            : tname[4] == 'v' ? LoraTarget::Value
                                              : LoraTarget::Output)
                         : (tname[4] == 'k' ? LoraTarget::FfnKey
                                            : LoraTarget::FfnValue);
                const LoraAdapter::Factors f = lora.factors(l, tgt);
                CHECK(f.a.numel() == static_cast<int64_t>(kRank) * kE);
                CHECK(f.b.numel() == static_cast<int64_t>(kE) * kRank);
                CHECK(f.a.dim(0) == kRank && f.a.dim(1) == kE);
                CHECK(f.b.dim(0) == kE && f.b.dim(1) == kRank);
            }
        }
        // Bogus layer / sanity: loader returns EMPTY views, never garbage.
        CHECK(lora.factors(kL + 3, LoraTarget::Key).a.numel() == 0);
        CHECK(lora.factors(0, LoraTarget::Key).b.numel() != 0);

        // Math vs reference on a small synthetic problem (same encoders).
        constexpr int64_t kOut = 24, kIn = 40, kR = 4;
        std::vector<uint16_t> A(static_cast<size_t>(kR) * kIn);
        std::vector<uint16_t> B(static_cast<size_t>(kOut) * kR);
        for (int64_t j = 0; j < kR; ++j)
            for (int64_t c = 0; c < kIn; ++c)
                A[static_cast<size_t>(j) * kIn + c] = enc_a(j, c);
        for (int64_t i = 0; i < kOut; ++i)
            for (int64_t j = 0; j < kR; ++j)
                B[static_cast<size_t>(i) * kR + j] = enc_b(i, j);

        std::vector<float> x(static_cast<size_t>(kIn));
        for (int64_t c = 0; c < kIn; ++c) x[static_cast<size_t>(c)] =
            ((c % 7) - 3) / 8.0f;
        std::vector<float> yk(static_cast<size_t>(kOut), 0.5f);
        std::vector<float> yr = yk;

        Tensor ta("A", {kR, kIn}, DType::F16);
        Tensor tb("B", {kOut, kR}, DType::F16);
        std::memcpy(ta.f16(), A.data(), A.size() * 2);
        std::memcpy(tb.f16(), B.data(), B.size() * 2);
        LoraAdapter::apply_lora(tb, ta, kScaling, x.data(), yk.data(),
                                kOut, kIn);
        reference_apply(B, A, kScaling, x, yr, kOut, kIn, kR);
        double maxdiff = 0.0;
        for (int64_t i = 0; i < kOut; ++i)
            maxdiff = std::max(maxdiff,
                static_cast<double>(std::fabs(yk[static_cast<size_t>(i)] -
                                              yr[static_cast<size_t>(i)])));
        CHECK(maxdiff < 1e-4);
        std::printf("  apply_lora vs reference: max|diff| = %.2e\n", maxdiff);
    }

    // ---- 4: malformed sidecars are rejected -------------------------------
    {
        // (a) Wrong container: structurally perfect GGUF, hostile arch tag.
        MiniGguf w = build_sidecar(1, kE, kRank, 2.0);
        const std::string needle = "omniseed-lora";
        const std::string repl = "rwkv-lora----";   // same length (13)
        bool patched = false;
        for (size_t i = 0; i + needle.size() <= w.buf_.size() && !patched; ++i) {
            if (std::memcmp(w.buf_.data() + i, needle.data(),
                            needle.size()) == 0) {
                std::memcpy(w.buf_.data() + i, repl.data(), repl.size());
                patched = true;
            }
        }
        CHECK(patched);
        CHECK(w.write(bad_arch_path));

        LoraAdapter bad_lora;
        const bool bad_ok = bad_lora.load(bad_arch_path);
        CHECK(!bad_ok);
        CHECK(!bad_lora.valid());
        CHECK(bad_lora.error().find("not an omniseed-lora") != std::string::npos);

        // (b) Truncated tensor directory: header + metadata only. The loader
        //     must fail open() cleanly (no OOB read, no exception, no abort).
        MiniGguf t = build_sidecar(kL, kE, kRank, 2.0);
        CHECK(t.write(trunc_path, /*with_dir=*/false, /*with_payloads=*/false));

        LoraAdapter trunc;
        const bool trunc_ok = trunc.load(trunc_path);
        CHECK(!trunc_ok);
        CHECK(!trunc.valid());
        CHECK(!trunc.error().empty());

        // (c) Corrupt-data variant: directory present, payloads truncated.
        //     Same contract: clean failure, deterministic error string.
        const std::string corr_path = trunc_path + ".corr";
        MiniGguf c = build_sidecar(kL, kE, kRank, 2.0);
        CHECK(c.write(corr_path, /*with_dir=*/true, /*with_payloads=*/false));

        LoraAdapter corr;
        const bool corr_ok = corr.load(corr_path);
        CHECK(!corr_ok);
        CHECK(!corr.valid());
        std::remove(corr_path.c_str());
    }

    // ---- 5: real-model attach/detach A/B ----------------------------------
    {
        std::FILE* f = nullptr;
#ifdef _MSC_VER
        if (fopen_s(&f, kQatModelPath, "rb") != 0) f = nullptr;
#else
        f = std::fopen(kQatModelPath, "rb");
#endif
        if (f == nullptr) {
            std::printf("  [skip] %s not present — real-model A/B skipped\n",
                        kQatModelPath);
        } else {
            std::fclose(f);
            RwkvModel model;
            if (!model.load(kQatModelPath)) {
                std::printf("  [skip] model load failed: %s\n",
                            model.error().c_str());
            } else {
                // The shipped smoke sidecar was produced by tools/lora_chat.py
                // (120+ training steps); geometry must match the base model.
                LoraAdapter lora;
                CHECK(lora.load("models/assistant-lora-test.gguf"));
                if (lora.valid()) {
                    CHECK(lora.layer_count() == model.config().n_layers);
                    CHECK(lora.n_embd() == model.config().n_embd);
                    CHECK(lora.rank() > 0);

                    // The sidecar must actually carry deltas (B != 0).
                    bool has_delta = false;
                    for (int32_t l = 0; l < lora.layer_count() && !has_delta;
                         ++l)
                        for (int t = 0; t < 6 && !has_delta; ++t) {
                            static const LoraTarget kT[6] = {
                                LoraTarget::Receptance, LoraTarget::Key,
                                LoraTarget::Value, LoraTarget::Output,
                                LoraTarget::FfnKey, LoraTarget::FfnValue};
                            const LoraAdapter::Factors fc =
                                lora.factors(l, kT[t]);
                            if (fc.b.numel() == 0) continue;
                            const uint16_t* bp = fc.b.f16();
                            for (int64_t e = 0; e < fc.b.numel(); ++e)
                                if (half_to_float(bp[e]) != 0.0f) {
                                    has_delta = true; break;
                                }
                        }
                    CHECK(has_delta);
                    std::printf("  sidecar deltas present: %s\n",
                                has_delta ? "yes" : "NO");

                    // Byte-level world vocab: ascii byte + 1 (no tokenizer).
                    std::vector<int32_t> ids;
                    for (char c : std::string("User: What is 2+2?\n\nAssistant:"))
                        ids.push_back(static_cast<int32_t>(
                            static_cast<unsigned char>(c)) + 1);

                    const int32_t V = model.config().n_vocab;
                    Tensor lg("logits", {V}, DType::F32);
                    std::vector<float> base_logits, lora_logits;

                    // baseline (detached)
                    RwkvState st;
                    model.init_state(st);
                    for (int32_t id : ids) model.forward(id, st, lg);
                    base_logits.assign(lg.f32(), lg.f32() + V);

                    // attached
                    model.set_lora(&lora);
                    model.init_state(st);
                    for (int32_t id : ids) model.forward(id, st, lg);
                    lora_logits.assign(lg.f32(), lg.f32() + V);

                    double maxd = 0.0;
                    int argmax_base = 0, argmax_lora = 0;
                    for (int32_t t = 0; t < V; ++t) {
                        maxd = std::max(maxd, static_cast<double>(
                            std::fabs(base_logits[static_cast<size_t>(t)] -
                                      lora_logits[static_cast<size_t>(t)])));
                        if (lora_logits[static_cast<size_t>(t)] >
                            lora_logits[static_cast<size_t>(argmax_lora)])
                            argmax_lora = t;
                        if (base_logits[static_cast<size_t>(t)] >
                            base_logits[static_cast<size_t>(argmax_base)])
                            argmax_base = t;
                    }
                    std::printf(
                        "  attached vs detached: max|dlogit| = %.3e "
                        "(argmax %d -> %d)\n", maxd, argmax_base, argmax_lora);
                    // The trained delta must measurably steer the logits.
                    // (An argmax FLIP is informationally printed but NOT
                    // asserted: a 120-step smoke sidecar shifts the
                    // distribution without necessarily flipping the winner;
                    // a real behavior pass flips it by construction.)
                    CHECK(maxd > 1e-3);          // the delta actually flows

                    // detach restores the baseline BIT-IDENTICALLY
                    model.set_lora(nullptr);
                    model.init_state(st);
                    for (int32_t id : ids) model.forward(id, st, lg);
                    CHECK(std::memcmp(lg.f32(), base_logits.data(),
                                      sizeof(float) * V) == 0);
                    std::printf("  detach: logits bit-identical to baseline\n");
                }
            }
        }
    }

    std::remove(sidecar_path.c_str());
    std::remove(bad_arch_path.c_str());
    std::remove(trunc_path.c_str());

    std::printf("RESULT: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
