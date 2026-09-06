// =============================================================================
//  OmniSeed — sensory.h
//  Sensory Fingerprinting: multi-modal biometric user identification.
//
//  Blueprint (Beyond-the-Cloud spec, capability #2):
//    * Instead of a password, the agent builds a compact "fingerprint" from
//      keystroke dynamics (digraph latencies + rhythm), voice timbre
//      (spectral centroid/rolloff/flatness over voiced frames), and optional
//      vision statistics. Each modality embedding is 32-dim; the fused
//      profile is a weighted mean verified per-modality with cosine
//      similarity.
//    * 128-bit FNV-1a "token" (hex) doubles as a non-reversible identifier
//      for personalization keys and swarm identity, satisfying the
//      privacy-preserving requirement (no raw biometrics ever leave the
//      device; only the profile + token are stored, locally).
// =============================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace omniseed {

// ---------------------------------------------------------------------------
// Sensory fingerprint: per-modality embeddings + fused profile
// ---------------------------------------------------------------------------
struct SensoryProfile {
    // Modality embeddings (fixed 32-dim, L2-normalized).
    float voice[32]  = {0};   // spectral timbre stats over voiced frames
    float typing[32] = {0};   // digraph latency + rhythm histogram
    float vision[32] = {0};   // optional face/scene stats (may be zeros)

    uint32_t n_voice_enroll  = 0;  // enrollment sample counts (trust weights)
    uint32_t n_typing_enroll = 0;
    uint32_t n_vision_enroll = 0;
    uint64_t updated_at      = 0;  // unix seconds of last update

    bool has_voice()  const { return n_voice_enroll  > 0; }
    bool has_typing() const { return n_typing_enroll > 0; }
    bool has_vision() const { return n_vision_enroll > 0; }
};

// Raw multi-modal observation for one session/utterance.
struct SensoryObservation {
    // Voice: 16 kHz mono PCM (may be empty).
    const float* pcm = nullptr;
    size_t       pcm_len = 0;
    int32_t      sample_rate = 16000;

    // Typing: digraph latencies in ms between consecutive keystrokes
    // (e.g. [180, 95, 210, ...]). May be empty.
    std::vector<float> digraph_ms;

    // Optional vision stats (precomputed 8-dim image summary). May be null.
    const float* vision_stats = nullptr;
};

class SensoryFingerprint {
public:
    struct Config {
        // Fused cosine >= threshold -> match. Calibrated so a different
        // speaker (different band-energy profile + typing) stays below it
        // while a genuine re-presentation passes comfortably.
        float verify_threshold = 0.90f;
        uint32_t min_enrollments = 2;    // samples before a modality is trusted
    };

    explicit SensoryFingerprint(const Config& cfg = {}) : cfg_(cfg) {}

    // Enrolls/updates the stored profile with a new observation
    // (running weighted mean; early samples weigh more).
    bool enroll(const SensoryObservation& obs);

    // Verifies an observation against the stored profile.
    // Returns fused similarity in [0,1]; sets is_match.
    float verify(const SensoryObservation& obs, bool& is_match) const;

    // Stable non-reversible identity token (128-bit FNV-1a over the fused
    // embedding, hex). Same user -> same token; different users differ.
    std::string identity_token() const;

    const SensoryProfile& profile() const { return profile_; }
    bool enrolled() const { return profile_.updated_at != 0; }
    const Config& config() const { return cfg_; }

    // Persistence (compact binary 'SNFP', version 1).
    bool save(const std::string& path) const;
    bool load(const std::string& path);

private:
    static bool embed_voice(const SensoryObservation& obs, float* out);
    static bool embed_typing(const SensoryObservation& obs, float* out);
    static bool embed_vision_stats(const float* stats, float* out);

    Config cfg_;
    SensoryProfile profile_;
};

} // namespace omniseed
