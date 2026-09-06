// =============================================================================
//  OmniSeed — emotional.h
//  Emotional Resonance Layer (capability #4 of the seven novel capabilities).
//
//  Blueprint:
//    * Infer the user's emotional state from vocal cues (pitch proxy, energy,
//      speech rate, pause ratio), text cues (affective lexicon, punctuation
//      intensity), and typing cadence (bursty typing ~ agitation).
//    * Modulate the agent's response style — tone prefix, verbosity, pacing —
//      and expose the state for the TokenBus ([emotion:happy] tokens) and the
//      SelfImprovement layer.
//
//  States: neutral, happy, excited, sad, angry, anxious, calm.
//  Deterministic, dependency-free; audio path is O(N) with a fixed buffer.
// =============================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace omniseed {

enum class Emotion : int32_t {
    Neutral = 0,
    Happy,
    Excited,
    Sad,
    Angry,
    Anxious,
    Calm,
};

const char* emotion_name(Emotion e);

// ---------------------------------------------------------------------------
// Multimodal emotional signal
// ---------------------------------------------------------------------------
struct EmotionalInput {
    // Raw text of the user turn (optional).
    const char* text = nullptr;

    // 16 kHz mono PCM for the utterance (optional).
    const float* pcm = nullptr;
    size_t pcm_len = 0;
    int32_t sample_rate = 16000;

    // Inter-keystroke intervals in ms (optional).
    const float* keystroke_ms = nullptr;
    size_t keystroke_len = 0;
};

// ---------------------------------------------------------------------------
struct EmotionalState {
    Emotion emotion = Emotion::Neutral;
    float valence  = 0.0f;   // -1 (negative) .. +1 (positive)
    float arousal  = 0.0f;   // 0 (calm) .. +1 (agitated)
    float confidence = 0.0f; // 0..1

    // Which channels contributed (diagnostics).
    bool from_voice = false;
    bool from_text  = false;
    bool from_typing = false;
};

// ---------------------------------------------------------------------------
class EmotionalResonance {
public:
    struct Config {
        float min_confidence = 0.35f;   // below this: stay neutral
        bool  adapt_responses = true;   // modulate output style
    };

    explicit EmotionalResonance(const Config& cfg = {}) : cfg_(cfg) {}

    // Fuses all available channels into an EmotionalState.
    EmotionalState perceive(const EmotionalInput& in) const;

    // Modulates a generated reply according to the perceived state:
    // tone-aware lead-in, empathy suffix for negative states, trimming for
    // agitated users (shorter!), and returns the adjusted string.
    std::string modulate(const std::string& reply, const EmotionalState& st) const;

    // Token-bus fragment for the fused stream, e.g. "[emotion:happy] ".
    std::string token_fragment(const EmotionalState& st) const;

    // Vocal-feature extraction (public for tests): returns false when the
    // clip is unusable (silence, too short).
    static bool voice_features(const float* pcm, size_t len, int32_t sr,
                               float* out /* 8 features */);

    const Config& config() const { return cfg_; }

private:
    static EmotionalState from_voice(const float* pcm, size_t len, int32_t sr);
    static EmotionalState from_text(const char* text);
    static EmotionalState from_typing(const float* ms, size_t len);

    Config cfg_;
};

} // namespace omniseed
