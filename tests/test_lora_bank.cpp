// =============================================================================
//  OmniSeed — test_lora_bank.cpp  (T122b)
//  LoraBank: mmap-only idle adapters, atomic activation, later-wins override,
//  shadow accounting, geometry quarantine, LRU eviction of idle only.
//  Uses NO Python: writes minimal omniseed-lora GGUF sidecars by hand
//  (same MiniGguf approach as tests/test_lora.cpp — self-contained file).
// =============================================================================
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "omniseed/core/gguf_format.h"
#include "omniseed/core/lora_bank.h"

using omniseed::LoraBank;
using omniseed::LoraTarget;
using namespace omniseed;   // gguf:: format constants (same as test_lora.cpp)

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (cond) { ++g_pass; std::printf("  PASS %s:%d: %s\n",              \
                                          __FILE__, __LINE__, #cond); }      \
        else {                                                               \
            ++g_fail;                                                        \
            std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
        }                                                                    \
    } while (0)

namespace {

int g_pass = 0, g_fail = 0;

uint16_t enc_f16(float v) {   // deterministic fp16 pattern (test values only)
    float frac = v - static_cast<float>(static_cast<int>(v));
    int32_t m = static_cast<int32_t>((frac + 1.0f) * 1024.0f) & 0x3FF;
    return static_cast<uint16_t>((15u << 10) | static_cast<uint32_t>(m));
}

// ---- Minimal LE GGUF writer (v2), copied pattern from test_lora.cpp -------
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
        str(k); i32(static_cast<int32_t>(gguf::GgufType::FLOAT64));
        uint8_t tmp[8];
        std::memcpy(tmp, &v, 8);
        buf_.insert(buf_.end(), tmp, tmp + 8);
    }
    void tensor(const std::string& name, int64_t rows, int64_t cols,
                const std::vector<uint16_t>& f16) {
        names_.push_back(name);
        ne_.push_back({cols, rows});               // GGUF stores ne reversed
        std::vector<uint8_t> bytes(f16.size() * 2);
        std::memcpy(bytes.data(), f16.data(), f16.size() * 2);
        payloads_.push_back(std::move(bytes));
    }
    void str(const std::string& s) {
        const uint64_t n = s.size();
        for (int i = 0; i < 8; ++i) buf_.push_back(
            static_cast<uint8_t>((n >> (8 * i)) & 0xff));
        buf_.insert(buf_.end(), s.begin(), s.end());
    }
    void i32(int32_t v) {
        for (int i = 0; i < 4; ++i) buf_.push_back(
            static_cast<uint8_t>((static_cast<uint32_t>(v) >> (8 * i)) & 0xff));
    }
    void u64(uint64_t v) {
        for (int i = 0; i < 8; ++i) buf_.push_back(
            static_cast<uint8_t>((v >> (8 * i)) & 0xff));
    }

    std::vector<uint8_t> buf_;
    std::vector<std::string> names_;
    std::vector<std::array<int64_t, 2>> ne_;
    std::vector<std::vector<uint8_t>> payloads_;
};

// Deterministic sidecar: `fill` scales the B factors so different skills
// produce DIFFERENT composite values (override tests compare exact bits).
std::vector<uint8_t> build_sidecar(int32_t L, int32_t E, int32_t rank,
                                   double scaling, float fill,
                                   bool skip_ffn = false) {
    MiniGguf w;
    auto& b = w.buf_;
    const auto push32 = [&b](uint32_t v) {
        for (int i = 0; i < 4; ++i) b.push_back(
            static_cast<uint8_t>((v >> (8 * i)) & 0xff));
    };
    push32(gguf::GGUF_MAGIC);
    push32(2);
    const size_t n_tensors_at = b.size(); b.resize(b.size() + 8);
    const size_t n_kv_at = b.size();      b.resize(b.size() + 8);
    int32_t n_kv = 0;
    w.kv_string("general.architecture", "omniseed-lora"); ++n_kv;
    w.kv_string("general.name", "lora-bank-fixture");     ++n_kv;
    w.kv_u64("lora.rank", static_cast<uint64_t>(rank));   ++n_kv;
    w.kv_f64("lora.scaling", scaling);                    ++n_kv;
    w.kv_u64("lora.layer_count", static_cast<uint64_t>(L)); ++n_kv;
    w.kv_u64("lora.n_embd", static_cast<uint64_t>(E));    ++n_kv;
    w.kv_u64("lora.base_vocab", 65536);                   ++n_kv;
    w.kv_i32("lora.trained_steps", 7);                    ++n_kv;

    const int32_t targets = skip_ffn ? 4 : 6;
    const int32_t n_tensors = L * targets * 2;
    std::memcpy(b.data() + n_tensors_at, &n_tensors, 4);
    std::memcpy(b.data() + n_kv_at, &n_kv, 4);

    for (int32_t l = 0; l < L; ++l) {
        for (const char* t : {"att.receptance", "att.key", "att.value",
                              "att.output", "ffn.key", "ffn.value"}) {
            if (skip_ffn && std::strncmp(t, "ffn", 3) == 0) continue;
            const std::string pre = "lora." + std::to_string(l) + "." + t;
            std::vector<uint16_t> a(static_cast<size_t>(rank) * E);
            std::vector<uint16_t> bm(static_cast<size_t>(E) * rank);
            for (int32_t j = 0; j < rank; ++j)
                for (int32_t c = 0; c < E; ++c)
                    a[static_cast<size_t>(j) * E + c] = enc_f16(0.5f);
            for (int32_t i = 0; i < E; ++i)
                for (int32_t j = 0; j < rank; ++j)
                    bm[static_cast<size_t>(i) * rank + j] = enc_f16(fill);
            w.tensor(pre + ".a", rank, E, a);
            w.tensor(pre + ".b", E, rank, bm);
        }
    }

    std::vector<uint8_t> out = w.buf_;
    // tensor directory
    auto dstr = [&out](const std::string& s) {
        const uint64_t n = s.size();
        for (int i = 0; i < 8; ++i) out.push_back(
            static_cast<uint8_t>((n >> (8 * i)) & 0xff));
        out.insert(out.end(), s.begin(), s.end());
    };
    auto du32 = [&out](uint32_t v) {
        for (int i = 0; i < 4; ++i) out.push_back(
            static_cast<uint8_t>((v >> (8 * i)) & 0xff));
    };
    auto du64 = [&out](uint64_t v) {
        for (int i = 0; i < 8; ++i) out.push_back(
            static_cast<uint8_t>((v >> (8 * i)) & 0xff));
    };
    uint64_t off = 0;
    for (size_t t = 0; t < w.names_.size(); ++t) {
        dstr(w.names_[t]);
        du32(2);
        du64(static_cast<uint64_t>(w.ne_[t][0]));
        du64(static_cast<uint64_t>(w.ne_[t][1]));
        du32(static_cast<uint32_t>(gguf::GgufDType::F16));
        du64(off);
        off += static_cast<uint64_t>(w.payloads_[t].size());
    }
    for (const auto& p : w.payloads_)
        out.insert(out.end(), p.begin(), p.end());
    return out;
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
    std::string base = (t != nullptr) ? std::string(t) : std::string(".");
    return base + "/omniseed_lora_bank_test";
}

} // namespace

int main() {
    namespace fs = std::filesystem;
    const std::string dir = temp_dir();
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);

    constexpr int32_t L = 3, E = 64, R = 4;
    CHECK(write_file(dir + "/skill_a.gguf",
                     build_sidecar(L, E, R, 0.5, 1.0f)));
    CHECK(write_file(dir + "/skill_b.gguf",
                     build_sidecar(L, E, R, 0.25, 2.0f)));
    CHECK(write_file(dir + "/skill_c.gguf",
                     build_sidecar(L, E, R, 1.0, 3.0f, /*skip_ffn=*/true)));
    CHECK(write_file(dir + "/bad_embd.gguf",
                     build_sidecar(L, E * 2, R, 1.0, 1.0f)));
    CHECK(write_file(dir + "/bad_rank.gguf",
                     build_sidecar(L, E, 16, 1.0, 1.0f)));
    CHECK(write_file(dir + "/not_lora.bin", {1, 2, 3, 4}));  // ignored ext

    // ---- A: open + quarantine ---------------------------------------------
    LoraBank bank;
    std::string err;
    CHECK(bank.open(dir, E, L, err));
    CHECK(bank.loaded_names().size() == 3);      // a, b, c
    CHECK(bank.quarantined().size() == 2);       // bad_embd, bad_rank
    CHECK(bank.composite() == nullptr);          // nothing active yet
    CHECK(bank.composite_hot_bytes() == 0);

    // ---- B: idle adapters are mmap-only (no hot copies until fold) --------
    {
        // activate builds composite; BEFORE that, hot memory must be zero.
        // (Direct proof of the brief's design change would need accessor to
        // loaded adapters' internals; composite_hot_bytes==0 above + the
        // exact formula check after activate below pins the accounting.)
        CHECK(bank.activate({"skill_a"}, err));
        const omniseed::LoraAdapter* c = bank.composite();
        CHECK(c != nullptr && c->valid());
        CHECK(c->rank() == R);
        // hot bytes = L * 6 targets * (R*E + E*R) * 4
        const int64_t expect =
            static_cast<int64_t>(L) * 6 * (R * E + E * R) * 4;
        CHECK(bank.composite_hot_bytes() == expect);
        // scaling baked: composite scaling_ must be 1.0 (fold contract)
        CHECK(std::fabs(c->scaling() - 1.0f) < 1e-6);
        // b32 holds source B * source scaling = enc_f16(1.0) * 0.5
        const auto* e = c->hot_entry(0, LoraTarget::FfnKey);
        CHECK(e != nullptr && e->rank == R);
        const float b0 = e->b32[0];
        const float want = omniseed::half_to_float(enc_f16(1.0f)) * 0.5f;
        CHECK(b0 == want);                       // bit-exact bake
    }

    // ---- C: later-activation wins + shadow accounting ---------------------
    {
        CHECK(bank.activate({"skill_a", "skill_b"}, err));
        const omniseed::LoraAdapter* c = bank.composite();
        const auto* e = c->hot_entry(0, LoraTarget::FfnKey);
        const float want_b = omniseed::half_to_float(enc_f16(2.0f)) * 0.25f;
        CHECK(e != nullptr && e->b32[0] == want_b);   // skill_b (later) won
        CHECK(bank.shadowed_slots() == static_cast<int64_t>(L) * 6);
        // att-only skill_c over ATT slots only; ffn slots keep skill_b's
        CHECK(bank.activate({"skill_a", "skill_b", "skill_c"}, err));
        const auto* ef = bank.composite()->hot_entry(0, LoraTarget::FfnKey);
        const auto* ea = bank.composite()->hot_entry(0, LoraTarget::Receptance);
        CHECK(ef != nullptr && ef->b32[0] == want_b);       // b kept
        const float want_c = omniseed::half_to_float(enc_f16(3.0f)) * 1.0f;
        CHECK(ea != nullptr && ea->b32[0] == want_c);       // c overwrote att
    }

    // ---- D: atomic unknown-name refusal (composite UNTOUCHED) -------------
    {
        const omniseed::LoraAdapter* before = bank.composite();
        CHECK(!bank.activate({"skill_a", "ghost"}, err));
        CHECK(!err.empty());
        CHECK(bank.composite() == before);       // same object, not rebuilt
        CHECK(bank.activate({"skill_a"}, err));  // recoverable
    }

    // ---- E: empty activation detaches (base behaviour, exact no-op) -------
    {
        CHECK(bank.activate({}, err));
        CHECK(bank.composite() == nullptr);
        CHECK(bank.composite_hot_bytes() == 0);
    }

    // ---- F: LRU eviction never drops active --------------------------------
    {
        // Activate skill_b (last_use = newest). Evict down to 2: idle a/c
        // go first, b survives.
        CHECK(bank.activate({"skill_b"}, err));
        std::vector<std::string> ev = bank.evict_idle(2);
        CHECK(ev.size() == 1);                   // 3 loaded -> 2
        CHECK(ev[0] == "skill_a" || ev[0] == "skill_c");
        CHECK(bank.loaded_names().size() == 2);
        CHECK(bank.composite() != nullptr);      // active survived
        // Evicting to 1: the ONE idle survivor goes; only skill_b (active)
        // remains. Active-only set can never be shrunk further.
        ev = bank.evict_idle(1);
        CHECK(ev.size() == 1);
        CHECK(bank.loaded_names().size() == 1);
        CHECK(bank.loaded_names()[0] == "skill_b");
        ev = bank.evict_idle(0);                     // cannot evict active
        CHECK(ev.empty());
        CHECK(bank.loaded_names().size() == 1);
    }

    // ---- G: geometry-mismatch activate is refused + detaches (fail-closed)
    {
        // Reload a fresh bank; skill files gone? No — reuse same dir.
        LoraBank b2;
        CHECK(b2.open(dir, E, L, err));
        CHECK(!b2.activate({"bad_embd"}, err));  // quarantined -> refusal
        CHECK(b2.composite() == nullptr);
        // mixed first-active geometry: c (att-only) then... c alone is fine
        CHECK(b2.activate({"skill_c"}, err));
        CHECK(b2.composite() != nullptr);
        // ffn slots stay UNFILLED for att-only sole source
        CHECK(b2.composite()->hot_entry(0, LoraTarget::FfnKey) == nullptr);
        CHECK(b2.composite()->hot_entry(0, LoraTarget::Receptance) != nullptr);
    }

    // ---- H: missing dir is the only open() failure ------------------------
    {
        LoraBank b3;
        CHECK(!b3.open(dir + "/nope", E, L, err));
        CHECK(!err.empty());
    }

    std::printf("[omniseed]  lora_bank: %d passed, %d failed\n",
                g_pass, g_fail);
    fs::remove_all(dir, ec);
    return g_fail == 0 ? 0 : 1;
}
