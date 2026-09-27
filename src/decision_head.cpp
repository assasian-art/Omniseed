// =============================================================================
//  OmniSeed — decision_head.cpp
//
//  System-1 decision head: hidden state h[E] -> action distribution, in one
//  matvec. See decision_head.h for the "two heads, one brain" contract and
//  for the honesty note about untrained (seeded) projections.
//
//  Hot path (decide) is allocation-free after the first call and touches
//  nothing but its own scratch: it never mutates the recurrent state, so
//  probing the head cannot perturb a conversation.
// =============================================================================
#include "omniseed/decision_head.h"
#include "omniseed/core/platform.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace omniseed {

// ---------------------------------------------------------------------------
// Action labels
// ---------------------------------------------------------------------------
const char* decision_action_name(DecisionAction a) {
    switch (a) {
        case DecisionAction::ABSTAIN: return "ABSTAIN";
        case DecisionAction::HOLD:    return "HOLD";
        case DecisionAction::BUY:     return "BUY";
        case DecisionAction::SELL:    return "SELL";
        case DecisionAction::CLOSE:   return "CLOSE";
        case DecisionAction::HEDGE:   return "HEDGE";
        case DecisionAction::EXPLAIN: return "EXPLAIN";
        case DecisionAction::COUNT:   break;
    }
    return "UNKNOWN";
}

namespace {

constexpr int32_t kActionCount = static_cast<int32_t>(DecisionAction::COUNT);

// SplitMix64 — deterministic, zero state beyond the seed. Used ONLY to seed an
// untrained placeholder projection so tests and CI are reproducible. It is
// never used to sample a decision.
inline uint64_t splitmix64(uint64_t& s) {
    s += 0x9E3779B97F4A7C15ull;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// uniform in [-1, 1)
inline float next_unit(uint64_t& s) {
    const uint64_t r = splitmix64(s);
    return static_cast<float>(
               static_cast<double>(r >> 11) / static_cast<double>(1ull << 53)) *
               2.0f - 1.0f;
}

// Minimal JSON string escaping so a stray quote in an asset label cannot
// produce malformed JSON for the caller that parses this.
std::string json_escape(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (const char c : s) {
        if (c == '"' || c == '\\') { o.push_back('\\'); o.push_back(c); }
        else if (static_cast<unsigned char>(c) < 0x20u) { o.push_back(' '); }
        else { o.push_back(c); }
    }
    return o;
}

constexpr char kMagic[8] = {'O', 'M', 'N', 'I', 'S', 'D', 'H', '1'};
// v2 added the per-action invalidation levels. A v1 blob is rejected rather
// than defaulted, because "no invalidation recorded" and "invalidation is
// zero" are different claims and only one of them is true.
constexpr int32_t kFormatVersion = 2;
constexpr int32_t kMaxDim  = 1 << 20;
constexpr int32_t kMaxStr  = 1 << 12;

// -- little-endian blob helpers (self-describing, no endian conversion:
//    OmniSeed GGUFs and state files are already native-LE by contract) ------
template <typename T>
bool wr(FILE* f, const T& v) { return std::fwrite(&v, sizeof(T), 1, f) == 1; }

template <typename T>
bool rd(FILE* f, T& v) { return std::fread(&v, sizeof(T), 1, f) == 1; }

bool wr_str(FILE* f, const std::string& s) {
    const int32_t n = static_cast<int32_t>(s.size());
    if (!wr(f, n)) return false;
    if (n == 0) return true;
    return std::fwrite(s.data(), 1, static_cast<size_t>(n), f) ==
           static_cast<size_t>(n);
}

bool rd_str(FILE* f, std::string& s) {
    int32_t n = 0;
    if (!rd(f, n) || n < 0 || n > kMaxStr) return false;
    s.assign(static_cast<size_t>(n), '\0');
    if (n == 0) return true;
    return std::fread(&s[0], 1, static_cast<size_t>(n), f) ==
           static_cast<size_t>(n);
}

} // namespace

// ---------------------------------------------------------------------------
// DecisionResult
// ---------------------------------------------------------------------------
std::string DecisionResult::to_json() const {
    char buf[448];
    std::snprintf(buf, sizeof(buf),
                  "{\"action\":\"%s\",\"confidence\":%.4f,"
                  "\"target_asset\":\"%s\",\"invalidation\":%.4f,"
                  "\"routing\":\"%s\",\"fast_path\":%s,\"margin\":%.4f,"
                  "\"matvecs\":%d}",
                  decision_action_name(action_type),
                  static_cast<double>(confidence_score),
                  json_escape(target_asset).c_str(),
                  static_cast<double>(invalidation),
                  json_escape(routing).c_str(),
                  fast_path ? "true" : "false",
                  static_cast<double>(margin), matvecs);
    return std::string(buf);
}

// ---------------------------------------------------------------------------
// Construction / geometry
// ---------------------------------------------------------------------------
DecisionHead::DecisionHead(const RwkvConfig& cfg, uint32_t seed) {
    init(cfg, seed);
}

bool DecisionHead::init(const RwkvConfig& cfg, uint32_t seed) {
    return init(cfg.n_embd, seed);
}

bool DecisionHead::init(int32_t n_embd, uint32_t seed) {
    ready_          = false;
    fitted_actions_ = 0;
    fitted_.clear();
    error_.clear();

    E_ = n_embd;
    A_ = kActionCount;
    if (E_ <= 0 || E_ > kMaxDim) {
        provenance_ = "invalid";
        error_ = "decision head: n_embd out of range";
        return false;
    }

    name_.clear();
    asset_.clear();
    name_.reserve(static_cast<size_t>(A_));
    asset_.reserve(static_cast<size_t>(A_));
    for (int32_t a = 0; a < A_; ++a) {
        name_.push_back(decision_action_name(static_cast<DecisionAction>(a)));
        asset_.emplace_back();
    }

    proj_.assign(static_cast<size_t>(A_) * static_cast<size_t>(E_), 0.0f);
    bias_.assign(static_cast<size_t>(A_), 0.0f);
    inval_.assign(static_cast<size_t>(A_), 0.0f);
    logits_.assign(static_cast<size_t>(A_), 0.0f);
    fitted_.assign(static_cast<size_t>(A_), 0u);

    seed_weights(seed);
    ready_      = true;
    provenance_ = "seeded placeholder (untrained; supply rows via set_action()/load())";
    return true;
}

void DecisionHead::seed_weights(uint32_t seed) {
    uint64_t s = static_cast<uint64_t>(seed) * 0x9E3779B97F4A7C15ull +
                 0xD1B54A32D192ED03ull;
    // 1/sqrt(E) scaling keeps ||W h|| O(1) for O(sqrt(E))-sized hidden states,
    // so the seeded head yields a *sharp but arbitrary* distribution rather
    // than saturating to a one-hot on whichever row happens to be widest.
    const float scale = 1.0f / std::sqrt(static_cast<float>(E_));
    for (float& w : proj_) w = next_unit(s) * scale;
    for (float& b : bias_) b = next_unit(s) * scale;
}

// ---------------------------------------------------------------------------
// The hot path
// ---------------------------------------------------------------------------
DecisionResult DecisionHead::decide(const float* hidden) const {
    DecisionResult r;
    if (!ready_ || hidden == nullptr) {
        r.routing = "error";
        return r;
    }

    const double t0 = platform::now_ms();

    // ---- 1) the matvec: [A_, E_] x [E_] ------------------------------------
    // This is the ENTIRE System-1 cost. There is no token loop here and none
    // is reachable from this function.
    for (int32_t a = 0; a < A_; ++a) {
        const float* row = proj_.data() +
                           static_cast<size_t>(a) * static_cast<size_t>(E_);
        float acc = bias_[static_cast<size_t>(a)];
        for (int32_t i = 0; i < E_; ++i) acc += row[i] * hidden[i];
        logits_[static_cast<size_t>(a)] = acc;
    }

    // ---- 2) softmax over A_ (max-subtracted) -------------------------------
    float mx = logits_[0];
    for (int32_t a = 1; a < A_; ++a)
        mx = std::max(mx, logits_[static_cast<size_t>(a)]);
    float sum = 0.0f;
    for (int32_t a = 0; a < A_; ++a) {
        const float e = std::exp(logits_[static_cast<size_t>(a)] - mx);
        logits_[static_cast<size_t>(a)] = e;
        sum += e;
    }
    if (!(sum > 0.0f) || !std::isfinite(sum)) sum = 1.0f;  // NaN-proof fallback
    const float inv = 1.0f / sum;
    for (int32_t a = 0; a < A_; ++a) logits_[static_cast<size_t>(a)] *= inv;

    // ---- 3) argmax + runner-up --------------------------------------------
    int32_t best   = 0;
    float   best_p = logits_[0];
    int32_t second = -1;
    float   second_p = 0.0f;
    for (int32_t a = 1; a < A_; ++a) {
        const float p = logits_[static_cast<size_t>(a)];
        if (p > best_p) {
            second = best;  second_p = best_p;
            best = a;       best_p = p;
        } else if (second < 0 || p > second_p) {
            second = a;     second_p = p;
        }
    }

    r.action_type      = static_cast<DecisionAction>(best);
    r.confidence_score = best_p;
    r.margin           = best_p - second_p;
    r.matvecs          = 1;
    r.target_asset     = asset_[static_cast<size_t>(best)];
    r.invalidation     = inval_[static_cast<size_t>(best)];

    // ---- 4) self-routing ---------------------------------------------------
    // Fail-closed by construction: ABSTAIN means "say nothing", EXPLAIN means
    // "this needs language". Neither may take the fast path, and a head that
    // merely clears the confidence bar with no margin escalates too.
    if (r.action_type == DecisionAction::ABSTAIN) {
        r.routing = "abstain";
    } else if (r.action_type == DecisionAction::EXPLAIN) {
        r.routing = "system2";
    } else if (r.confidence_score >= cfg_.threshold &&
               r.margin >= cfg_.margin_floor) {
        r.routing   = "self";
        r.fast_path = true;
    } else {
        r.routing = "system2";
    }

    // NOTE: `ms` includes the two clock reads. For benchmarks, time a loop of
    // decide() calls externally rather than reading this field.
    const double t1 = platform::now_ms();
    r.ms      = t1 - t0;
    last_ns_  = r.ms * 1e6;
    return r;
}

DecisionResult DecisionHead::decide(const Tensor& hidden) const {
    if (hidden.dtype() != DType::F32 ||
        hidden.numel() != static_cast<int64_t>(E_)) {
        DecisionResult r;
        r.routing = "error";
        return r;
    }
    return decide(hidden.f32());
}

// ---------------------------------------------------------------------------
// Offline fitting
// ---------------------------------------------------------------------------
void DecisionHead::set_action(DecisionAction a, const float* row, float bias) {
    const int32_t idx = static_cast<int32_t>(a);
    if (!ready_ || idx < 0 || idx >= A_ || row == nullptr) return;

    std::memcpy(proj_.data() + static_cast<size_t>(idx) * static_cast<size_t>(E_),
                row, static_cast<size_t>(E_) * sizeof(float));
    bias_[static_cast<size_t>(idx)] = bias;

    if (fitted_[static_cast<size_t>(idx)] == 0u) {
        fitted_[static_cast<size_t>(idx)] = 1u;
        ++fitted_actions_;
    }
    provenance_ = trained() ? "fitted (all action rows supplied)"
                            : "partially fitted placeholder";
}

void DecisionHead::set_action_asset(DecisionAction a, const std::string& asset) {
    const int32_t idx = static_cast<int32_t>(a);
    if (!ready_ || idx < 0 || idx >= A_) return;
    asset_[static_cast<size_t>(idx)] = asset;
}

void DecisionHead::set_action_invalidation(DecisionAction a, float level) {
    const int32_t idx = static_cast<int32_t>(a);
    if (!ready_ || idx < 0 || idx >= A_) return;
    inval_[static_cast<size_t>(idx)] = level;
}

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------
bool DecisionHead::save(const std::string& path) const {
    if (!ready_) return false;
    FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) return false;

    bool ok = std::fwrite(kMagic, 1, 8, f) == 8;
    ok = ok && wr(f, kFormatVersion);
    ok = ok && wr(f, E_);
    ok = ok && wr(f, A_);
    ok = ok && wr(f, static_cast<int32_t>(trained() ? 1 : 0));
    for (int32_t a = 0; ok && a < A_; ++a) ok = wr_str(f, name_[static_cast<size_t>(a)]);
    for (int32_t a = 0; ok && a < A_; ++a) ok = wr_str(f, asset_[static_cast<size_t>(a)]);
    if (ok && !inval_.empty())
        ok = std::fwrite(inval_.data(), sizeof(float), inval_.size(), f) == inval_.size();
    if (ok && !proj_.empty())
        ok = std::fwrite(proj_.data(), sizeof(float), proj_.size(), f) == proj_.size();
    if (ok && !bias_.empty())
        ok = std::fwrite(bias_.data(), sizeof(float), bias_.size(), f) == bias_.size();

    std::fclose(f);
    return ok;
}

bool DecisionHead::load(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        error_ = "decision head: cannot open " + path;
        return false;
    }

    char magic[8] = {0};
    bool ok = std::fread(magic, 1, 8, f) == 8 &&
              std::memcmp(magic, kMagic, 8) == 0;

    int32_t version = 0, E = 0, A = 0, fitted_flag = 0;
    ok = ok && rd(f, version) && version == kFormatVersion;
    ok = ok && rd(f, E) && rd(f, A) && rd(f, fitted_flag);

    if (!ok || E <= 0 || E > kMaxDim || A <= 0) {
        std::fclose(f);
        error_ = "decision head: malformed header in " + path;
        return false;
    }
    // The action space is compiled into the binary (the enum). A file built
    // against a different action set would silently remap decisions, so
    // refuse it instead of guessing.
    if (A != kActionCount) {
        std::fclose(f);
        error_ = "decision head: action-space mismatch in " + path +
                 " (file A=" + std::to_string(A) +
                 ", binary A=" + std::to_string(kActionCount) + ")";
        return false;
    }

    if (!init(E, 1234u)) {   // geometry + scratch; projection overwritten below
        std::fclose(f);
        return false;
    }

    std::vector<std::string> names(static_cast<size_t>(A));
    std::vector<std::string> assets(static_cast<size_t>(A));
    for (int32_t a = 0; ok && a < A; ++a) ok = rd_str(f, names[static_cast<size_t>(a)]);
    for (int32_t a = 0; ok && a < A; ++a) ok = rd_str(f, assets[static_cast<size_t>(a)]);
    if (ok)
        ok = std::fread(inval_.data(), sizeof(float), inval_.size(), f) == inval_.size();
    if (ok)
        ok = std::fread(proj_.data(), sizeof(float), proj_.size(), f) == proj_.size();
    if (ok)
        ok = std::fread(bias_.data(), sizeof(float), bias_.size(), f) == bias_.size();

    if (!ok) {
        std::fclose(f);
        error_ = "decision head: truncated payload in " + path;
        return false;
    }
    std::fclose(f);

    // Integrity: labels must match the compiled-in action order.
    for (int32_t a = 0; a < A; ++a) {
        if (names[static_cast<size_t>(a)] != name_[static_cast<size_t>(a)]) {
            error_ = "decision head: action label mismatch at index " +
                     std::to_string(a) + " in " + path;
            return false;
        }
    }

    asset_ = assets;
    fitted_actions_ = (fitted_flag != 0) ? A : 0;
    fitted_.assign(static_cast<size_t>(A),
                   fitted_flag != 0 ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0));
    provenance_ = (fitted_flag != 0)
        ? "loaded fitted projection"
        : "loaded seeded placeholder (file was untrained)";
    return true;
}

} // namespace omniseed
