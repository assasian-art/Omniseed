// =============================================================================
//  OmniSeed — heads.h
//
//  The shared vocabulary for "ONE BACKBONE, MULTIPLE HEADS".
//
//  Every head in this tree reads the SAME object: the hidden state h[E] that a
//  single RWKV-7 forward pass produces. A head is nothing more than a small
//  projection [K, E] x [E] -> [K] followed by a squashing function, so the
//  marginal cost of an extra head is K*E multiply-accumulates — hundreds of
//  nanoseconds, not a second forward pass.
//
//  That is the whole design: ONE brain, many cheap readouts of it.
//
//    DecisionHead        [A, E] -> softmax  -> a structured action
//    ClassificationHead  [L, E] -> softmax  -> a label distribution
//    ScoringHead         [3, E] -> sigmoid  -> priority / urgency / confidence
//    TokenHead           the existing generator, optionally PREFIXED with a
//                        decision so the text explains what was already chosen
//
//  A head NEVER writes back into the recurrent state and NEVER allocates on the
//  hot path, so probing one cannot perturb a conversation.
//
//  HONESTY NOTE — read before trusting any head output.
//  A freshly constructed head has NO trained projection. Every head here seeds
//  its weights deterministically so tests and CI are reproducible, and exposes
//  `trained()` / `provenance()` so a caller can tell a fitted head from a
//  placeholder. A seeded head emits well-formed but MEANINGLESS values. The
//  per-head docs say so, and `provenance()` reports it at runtime. Never
//  present an untrained head's output as a judgement.
// =============================================================================
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace omniseed {

// ---------------------------------------------------------------------------
// Domain — which kind of problem a head is being asked about.
//
// `General` is the fallback and must stay at 0: an unclassified input lands
// here, and a zero default means "unknown" rather than a silently chosen
// specialist domain.
// ---------------------------------------------------------------------------
enum class Domain : int32_t {
    General = 0,
    Trading,
    Language,
    Vision,
    Audio,
    COUNT
};

constexpr int32_t kDomainCount = static_cast<int32_t>(Domain::COUNT);

const char* domain_name(Domain d);
// Returns Domain::General for anything unrecognised — never throws, never
// silently picks a specialist domain.
Domain domain_from_name(const std::string& name);
bool   domain_known(const std::string& name);

// ---------------------------------------------------------------------------
// One (label, probability) pair out of a classification head.
// ---------------------------------------------------------------------------
struct LabelProb {
    std::string label;
    float       probability = 0.0f;
};

// ---------------------------------------------------------------------------
// ClassificationResult — a multi-class distribution, reduced to top-k.
//
// `margin` is p(top) - p(second). A 0.51/0.49 split is a coin flip that happens
// to have a winner, and margin is how a caller finds that out; `probability`
// alone cannot express it.
// ---------------------------------------------------------------------------
struct ClassificationResult {
    std::string            domain;      // the label set that produced this
    std::vector<LabelProb> top_k;
    float                  margin    = 0.0f;
    int32_t                matvecs   = 0;
    double                 us        = 0.0;   // wall time of the matvec+softmax
    std::string            to_json() const;
};

// ---------------------------------------------------------------------------
// ScoreResult — three independent sigmoids over the same hidden state.
//
// They are deliberately NOT a softmax: priority, urgency and confidence are
// separate questions ("how important", "how soon", "how sure"), and forcing
// them to sum to 1 would make a high-priority item necessarily low-urgency.
// ---------------------------------------------------------------------------
struct ScoreResult {
    float   priority   = 0.0f;   // 0..1
    float   urgency    = 0.0f;   // 0..1
    float   confidence = 0.0f;   // 0..1 — confidence in the SCORE, not a claim
    int32_t matvecs    = 0;
    double  us         = 0.0;
    std::string to_json() const;
};

// ---------------------------------------------------------------------------
// Shared internals. Kept inline and header-only so the heads do not have to
// link a common .cpp just to seed a placeholder or escape a string.
// ---------------------------------------------------------------------------
namespace heads_detail {

// SplitMix64 — deterministic, no state beyond the seed. Used ONLY to seed an
// untrained placeholder projection so tests and CI are reproducible. It is
// never used to sample a decision.
inline uint64_t splitmix64(uint64_t& s) {
    s += 0x9E3779B97F4A7C15ull;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// Uniform in [-1, 1).
inline float next_unit(uint64_t& s) {
    const uint64_t r = splitmix64(s);
    return static_cast<float>(
               static_cast<double>(r >> 11) / static_cast<double>(1ull << 53)) *
               2.0f - 1.0f;
}

// Minimal JSON string escaping: the two characters that would actually break a
// document, plus control characters. Label sets are code-defined, but a caller
// may register its own, so this cannot assume ASCII-safe input.
inline std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (const char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) out += '?';
                else out += c;
        }
    }
    return out;
}

// Numeric formatter that never emits "nan" / "inf" — a JSON document with a
// bare `nan` in it is invalid, and a downstream parser failing on it is a much
// worse outcome than a 0.0 the caller can see is a default.
inline void append_float(std::string& out, float v) {
    char buf[32];
    if (v != v || v > 3.4e38f || v < -3.4e38f) std::snprintf(buf, sizeof(buf), "0.0");
    else std::snprintf(buf, sizeof(buf), "%.4f", static_cast<double>(v));
    out += buf;
}

} // namespace heads_detail

} // namespace omniseed
