// =============================================================================
//  OmniSeed — audio_events.h
//  Event-layer audio intelligence on top of the encoders:
//
//    * WakeWordDetector      — always-on keyword spotting (feature #15).
//    * SoundEventDetector    — environmental sound events: alarm, doorbell,
//                              glass, knock, siren, speech, music (spec #3).
//    * AudioSceneClassifier  — ambient soundscape classes: office, kitchen,
//                              street, quiet (spec #63).
//    * AnomalyDetector       — statistical deviation flagging on the feature
//                              stream (spec #70).
//
//  All detectors consume the SAME cheap frame features used by FocalCodec
//  (log-spaced band energies) — one front end, four consumers. Deterministic,
//  O(N) over samples, zero allocations in the steady state.
// =============================================================================
#pragma once

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace omniseed {

struct PcmAudio;

// ---------------------------------------------------------------------------
// Shared frame features (24 log-spaced band energies per 80 ms frame).
// Mirrors FocalCodec::frame_features but exposed for the event layer.
// ---------------------------------------------------------------------------
bool compute_frame_features(const PcmAudio& audio, int32_t frame_ms,
                            int32_t n_bands,
                            std::vector<float>& frames /* [f*bands] */,
                            int32_t& out_bands);

// ---------------------------------------------------------------------------
// Sound events
// ---------------------------------------------------------------------------
enum class SoundEvent : int32_t {
    None = 0,
    Alarm,        // periodic high-energy broadband pulse
    Doorbell,     // two-tone mid-frequency chime
    GlassBreak,   // broadband transient with high-frequency burst
    Knock,        // 2-4 low-frequency transient pulses
    Siren,        // slow frequency sweep, sustained
    Speech,       // sustained mid-band energy with pauses
    Music,        // sustained energy with rhythmic variance
};

const char* sound_event_name(SoundEvent e);

struct SoundEventResult {
    SoundEvent event = SoundEvent::None;
    float confidence = 0.0f;
};

class SoundEventDetector {
public:
    struct Config {
        float trigger_threshold = 0.55f;
    };
    explicit SoundEventDetector(const Config& cfg = {}) : cfg_(cfg) {}

    // Classifies one clip. Returns the strongest event above threshold.
    SoundEventResult detect(const PcmAudio& audio) const;

    const Config& config() const { return cfg_; }

private:
    Config cfg_;
};

// ---------------------------------------------------------------------------
// Wake word detection: energy-gated spectral-template matcher.
// The reference template is the canonical "hey omni" two-syllable contour:
// two voiced energy bursts with a mid-band dip. Enrollment replaces the
// template with a user-specific contour for lower false accepts.
// ---------------------------------------------------------------------------
class WakeWordDetector {
public:
    struct Config {
        int32_t  frame_ms = 40;
        float    energy_gate = 0.0015f;   // skip silence
        float    match_threshold = 0.62f;
        int32_t  cooldown_frames = 25;    // ~1 s between triggers
    };
    explicit WakeWordDetector(const Config& cfg = {});

    // Streaming API: feed audio chunk; returns true on trigger.
    bool push(const PcmAudio& chunk);
    // Batch API: scan a whole clip for the wake phrase.
    bool scan(const PcmAudio& audio, float& best_score) const;

    // User enrollment: builds a personalized template from a spoken sample.
    bool enroll(const PcmAudio& sample);

    bool template_enrolled() const { return enrolled_; }
    const Config& config() const { return cfg_; }

private:
    Config cfg_;
    bool enrolled_ = false;
    std::vector<float> template_contour_;   // 32-bucket energy contour
    std::vector<float> pending_;            // streaming residue
    int32_t cooldown_ = 0;
};

// ---------------------------------------------------------------------------
// Audio scene classification (spec #63)
// ---------------------------------------------------------------------------
enum class AudioScene : int32_t {
    Quiet = 0,
    Office,
    Kitchen,
    Street,
    Music,
    Speech,
};
const char* audio_scene_name(AudioScene s);

class AudioSceneClassifier {
public:
    // Classifies the ambient soundscape of a clip.
    static AudioScene classify(const PcmAudio& audio, float& confidence);
};

// ---------------------------------------------------------------------------
// Edge-based anomaly detection (spec #70)
// ---------------------------------------------------------------------------
class AudioAnomalyDetector {
public:
    struct Config {
        size_t  baseline_frames = 128;   // frames to learn the baseline
        float   sensitivity     = 3.5f;  // sigma multiplier to flag
    };
    explicit AudioAnomalyDetector(const Config& cfg = {}) : cfg_(cfg) {}

    // Consumes frame features (from compute_frame_features); returns true
    // when the current frame deviates > sensitivity * sigma from baseline.
    bool push_frame(const float* bands, int32_t n_bands);

    bool baseline_ready() const { return seen_ >= cfg_.baseline_frames; }
    float last_deviation() const { return last_dev_; }
    const Config& config() const { return cfg_; }

private:
    Config cfg_;
    std::vector<double> mean_, m2_;
    size_t seen_ = 0;
    float last_dev_ = 0.0f;
};

} // namespace omniseed
