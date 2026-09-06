// =============================================================================
//  OmniSeed — audio_events.cpp
//  Wake word, sound events, scene classification, anomaly detection.
//
//  Feature basis: 24 log-spaced band energies per frame (the FocalCodec front
//  end). Event signatures are hand-crafted classifiers over these features —
//  transparent, deterministic, and free of learned weights so they work
//  before the model converter exists (and serve as its fallback forever).
// =============================================================================
#include "omniseed/audio/audio_events.h"
#include "omniseed/audio/audio.h"
#include "omniseed/core/platform.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace omniseed {

// ===========================================================================
// Shared frame features (band energies via Goertzel-lite, decimated)
// ===========================================================================
bool compute_frame_features(const PcmAudio& audio, int32_t frame_ms,
                            int32_t n_bands, std::vector<float>& frames,
                            int32_t& out_bands) {
    frames.clear();
    out_bands = n_bands;
    if (!audio.valid()) return false;

    const int32_t sr = audio.sample_rate;
    const int32_t frame_len = sr * frame_ms / 1000;
    if (frame_len <= 0 || audio.samples.size() < static_cast<size_t>(frame_len))
        return false;

    const size_t n_frames = audio.samples.size() / static_cast<size_t>(frame_len);
    frames.resize(n_frames * static_cast<size_t>(n_bands), 0.0f);

    for (size_t f = 0; f < n_frames; ++f) {
        const float* fr = audio.samples.data() + f * frame_len;
        for (int32_t b = 0; b < n_bands; ++b) {
            const double hz = 60.0 * std::pow(1000.0 / 60.0,
                static_cast<double>(b) / n_bands);
            const double w = 2.0 * 3.14159265358979323846 * hz / sr;
            double re = 0.0, im = 0.0;
            for (int32_t t = 0; t < frame_len; t += 4) {
                const double s = fr[t];
                re += s * std::cos(w * t);
                im += s * std::sin(w * t);
            }
            const double mag = std::sqrt(re * re + im * im) / (frame_len / 4);
            frames[f * n_bands + b] =
                static_cast<float>(std::log10(mag + 1e-8));
        }
    }
    return true;
}

const char* sound_event_name(SoundEvent e) {
    switch (e) {
        case SoundEvent::Alarm:      return "alarm";
        case SoundEvent::Doorbell:   return "doorbell";
        case SoundEvent::GlassBreak: return "glass_break";
        case SoundEvent::Knock:      return "knock";
        case SoundEvent::Siren:      return "siren";
        case SoundEvent::Speech:     return "speech";
        case SoundEvent::Music:      return "music";
        case SoundEvent::None:       break;
    }
    return "none";
}

namespace {

struct ClipStats {
    double energy_mean = 0, energy_max = 0, energy_var = 0;
    double hf_ratio = 0;         // high-band / total
    double lf_ratio = 0;
    double band_flux = 0;        // mean |frame-to-frame band change|
    double centroid_var = 0;     // variance of band centroid
    double onset_rate = 0;       // sharp energy rises per second
    double periodicity = 0;      // autocorr peak of energy contour
    double duration_s = 0;
    int32_t n_frames = 0;
};

ClipStats clip_stats(const std::vector<float>& frames, int32_t bands,
                     int32_t sr, int32_t frame_ms) {
    ClipStats st;
    const int32_t n = static_cast<int32_t>(frames.size() / bands);
    st.n_frames = n;
    if (n < 2) return st;

    std::vector<double> energy(n), centroid(n);
    std::vector<float> prev(bands, 0.0f);

    for (int32_t f = 0; f < n; ++f) {
        const float* row = frames.data() + static_cast<size_t>(f) * bands;
        double e = 0, cnum = 0;
        for (int32_t b = 0; b < bands; ++b) {
            const double v = std::pow(10.0, row[b]);   // back to magnitude
            e += v;
            cnum += v * static_cast<double>(b);
        }
        energy[f] = e;
        centroid[f] = e > 1e-9 ? cnum / e : 0.0;

        if (f > 0) {
            double flux = 0;
            for (int32_t b = 0; b < bands; ++b)
                flux += std::fabs(row[b] - prev[b]);
            st.band_flux += flux / bands;
        }
        std::memcpy(prev.data(), row, sizeof(float) * bands);
    }
    st.band_flux /= (n - 1);

    double emean = 0;
    for (double e : energy) { emean += e; st.energy_max = std::max(st.energy_max, e); }
    emean /= n;
    st.energy_mean = emean;
    for (double e : energy) st.energy_var += (e - emean) * (e - emean);
    st.energy_var /= n;

    // High/low band ratio from mean per-band energy.
    std::vector<double> band_mean(bands, 0.0);
    for (int32_t f = 0; f < n; ++f)
        for (int32_t b = 0; b < bands; ++b)
            band_mean[b] += std::pow(10.0, frames[f * bands + b]);
    double total = 0;
    for (int32_t b = 0; b < bands; ++b) { band_mean[b] /= n; total += band_mean[b]; }
    double hi = 0, lo = 0;
    for (int32_t b = 0; b < bands; ++b) {
        if (b >= bands * 2 / 3) hi += band_mean[b];
        else lo += band_mean[b];
    }
    st.hf_ratio = total > 1e-9 ? hi / total : 0.0;
    st.lf_ratio = total > 1e-9 ? lo / total : 0.0;

    // Centroid variance.
    double cmean = 0;
    for (double c : centroid) cmean += c;
    cmean /= n;
    for (double c : centroid) st.centroid_var += (c - cmean) * (c - cmean);
    st.centroid_var /= n;

    // Onset rate: frames whose energy jumps > 2x previous.
    int32_t onsets = 0;
    for (int32_t f = 1; f < n; ++f)
        if (energy[f] > 2.0 * energy[f - 1] + 1e-9) ++onsets;
    const double dur = n * frame_ms / 1000.0;
    st.duration_s = dur;
    st.onset_rate = dur > 0 ? onsets / dur : 0.0;

    // Periodicity: max autocorrelation of the energy contour (lag 2..n/2).
    double best = 0;
    for (int32_t lag = 2; lag < n / 2 && lag < 25; ++lag) {
        double num = 0, d1 = 0, d2 = 0;
        for (int32_t f = lag; f < n; ++f) {
            num += energy[f] * energy[f - lag];
            d1 += energy[f] * energy[f];
            d2 += energy[f - lag] * energy[f - lag];
        }
        if (d1 > 1e-12 && d2 > 1e-12)
            best = std::max(best, num / std::sqrt(d1 * d2));
    }
    st.periodicity = best;
    (void)sr;
    return st;
}

} // namespace

// ===========================================================================
// Sound event classification
// ===========================================================================
SoundEventResult SoundEventDetector::detect(const PcmAudio& audio) const {
    SoundEventResult res;
    std::vector<float> frames;
    int32_t bands = 0;
    if (!compute_frame_features(audio, 80, 24, frames, bands)) return res;
    const ClipStats st = clip_stats(frames, bands, audio.sample_rate, 80);
    if (st.n_frames < 3) return res;

    // Hand-tuned signature scores in [0, 1].
    auto s01 = [](double v) { return std::min(1.0, std::max(0.0, v)); };

    const double s_alarm = s01(st.periodicity * 1.2) *
                           s01(st.energy_mean * 40.0) *
                           (1.0 - s01(st.centroid_var / 4.0));
    const double s_doorbell = s01(st.onset_rate * 1.5) *
                              s01(1.0 - st.hf_ratio * 1.4) *
                              s01(st.energy_max * 25.0);
    const double s_glass = s01(st.hf_ratio * 2.2) * s01(st.onset_rate * 1.2) *
                           s01(st.band_flux * 3.0);
    const double s_knock = s01(st.onset_rate * 1.1) *
                           s01(st.lf_ratio * 1.6) *
                           s01(1.0 - st.band_flux * 2.0);
    const double s_siren = s01(st.periodicity * 1.05) *
                           s01(st.centroid_var / 2.5) *
                           s01(st.duration_s / 2.0);
    const double s_speech = s01((1.0 - st.periodicity) * 1.1) *
                            s01(st.band_flux * 2.5) *
                            s01(1.0 - st.hf_ratio * 1.5) *
                            s01(st.duration_s / 1.0);
    const double s_music = s01(st.periodicity * 1.15) *
                           s01(st.band_flux * 2.2) *
                           s01(st.duration_s / 3.0);

    struct Cand { SoundEvent e; double s; };
    const Cand cands[] = {
        {SoundEvent::Alarm, s_alarm}, {SoundEvent::Doorbell, s_doorbell},
        {SoundEvent::GlassBreak, s_glass}, {SoundEvent::Knock, s_knock},
        {SoundEvent::Siren, s_siren}, {SoundEvent::Speech, s_speech},
        {SoundEvent::Music, s_music},
    };
    const Cand* best = &cands[0];
    for (const Cand& c : cands) if (c.s > best->s) best = &c;

    if (best->s < cfg_.trigger_threshold) return res;
    res.event = best->e;
    res.confidence = static_cast<float>(best->s);
    return res;
}

// ===========================================================================
// Wake word detection
// ===========================================================================
WakeWordDetector::WakeWordDetector(const Config& cfg) : cfg_(cfg) {
    // Default "hey omni"-style template: two energy bursts, mid-band dip.
    template_contour_.resize(32);
    for (int32_t i = 0; i < 32; ++i) {
        const double t = i / 31.0;
        // Burst 1 at t~0.25, burst 2 at t~0.7.
        const double b1 = std::exp(-std::pow((t - 0.25) / 0.08, 2.0));
        const double b2 = std::exp(-std::pow((t - 0.70) / 0.10, 2.0));
        template_contour_[i] = static_cast<float>(0.85 * b1 + 1.0 * b2);
    }
}

namespace {

// 32-bucket energy contour of a clip (normalized to max 1).
void energy_contour(const PcmAudio& a, int32_t frame_ms,
                    std::vector<float>& out) {
    out.assign(32, 0.0f);
    const int32_t frame_len = a.sample_rate * frame_ms / 1000;
    if (frame_len <= 0 || a.samples.empty()) return;
    const size_t n_frames = a.samples.size() / static_cast<size_t>(frame_len);
    if (n_frames < 8) return;

    const double bucket_f = static_cast<double>(n_frames) / 32.0;
    double max_e = 1e-12;
    std::vector<double> raw(32, 0.0);
    for (size_t f = 0; f < n_frames; ++f) {
        const float* fr = a.samples.data() + f * frame_len;
        double e = 0;
        for (int32_t t = 0; t < frame_len; ++t) e += fr[t] * fr[t];
        const size_t b = std::min<size_t>(31,
            static_cast<size_t>(f / bucket_f));
        raw[b] += e;
        max_e = std::max(max_e, e);
    }
    for (int32_t i = 0; i < 32; ++i)
        out[i] = static_cast<float>(raw[i] / max_e);
}

double contour_similarity(const std::vector<float>& a,
                          const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return 0.0;
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if (na < 1e-12 || nb < 1e-12) return 0.0;
    return dot / std::sqrt(na * nb);
}

} // namespace

bool WakeWordDetector::enroll(const PcmAudio& sample) {
    std::vector<float> contour;
    energy_contour(sample, cfg_.frame_ms, contour);
    double sum = 0;
    for (float v : contour) sum += v;
    if (sum < 1.0f) return false;       // too quiet / too short
    template_contour_ = contour;
    enrolled_ = true;
    return true;
}

bool WakeWordDetector::scan(const PcmAudio& audio, float& best_score) const {
    best_score = 0.0f;
    std::vector<float> contour;
    energy_contour(audio, cfg_.frame_ms, contour);

    // Sliding windows of 32 frames over a longer contour.
    if (contour.size() < 32) return false;
    const int32_t n = static_cast<int32_t>(contour.size());
    bool hit = false;
    std::vector<float> win(32);
    for (int32_t start = 0; start + 32 <= n; start += 8) {
        // Normalize window.
        double mx = 1e-9;
        for (int32_t i = 0; i < 32; ++i)
            mx = std::max<double>(mx, contour[start + i]);
        for (int32_t i = 0; i < 32; ++i)
            win[i] = contour[start + i] / static_cast<float>(mx);
        const double s = contour_similarity(win, template_contour_);
        best_score = std::max(best_score, static_cast<float>(s));
        if (s >= cfg_.match_threshold) hit = true;
    }
    return hit;
}

bool WakeWordDetector::push(const PcmAudio& chunk) {
    if (cooldown_ > 0) { --cooldown_; return false; }

    // Energy gate: ignore near-silence chunks.
    double e = 0;
    for (float s : chunk.samples) e += s * s;
    if (chunk.samples.size() > 0) e /= chunk.samples.size();
    if (e < cfg_.energy_gate) {
        // Keep a small residue so a wake phrase split across chunks still
        // accumulates.
        pending_.insert(pending_.end(), chunk.samples.begin(),
                        chunk.samples.end());
        if (pending_.size() > static_cast<size_t>(chunk.sample_rate * 2))
            pending_.erase(pending_.begin(),
                           pending_.end() - chunk.sample_rate * 2);
        return false;
    }

    pending_.insert(pending_.end(), chunk.samples.begin(), chunk.samples.end());
    PcmAudio acc;
    acc.sample_rate = chunk.sample_rate;
    acc.samples = pending_;

    float score = 0.0f;
    const bool hit = scan(acc, score);
    // Keep the last second for next time.
    const size_t keep = static_cast<size_t>(chunk.sample_rate);
    if (pending_.size() > keep)
        pending_.erase(pending_.begin(), pending_.end() - keep);
    if (hit) cooldown_ = cfg_.cooldown_frames;
    return hit;
}

// ===========================================================================
// Scene classification
// ===========================================================================
const char* audio_scene_name(AudioScene s) {
    switch (s) {
        case AudioScene::Office:  return "office";
        case AudioScene::Kitchen: return "kitchen";
        case AudioScene::Street:  return "street";
        case AudioScene::Music:   return "music";
        case AudioScene::Speech:  return "speech";
        case AudioScene::Quiet:   break;
    }
    return "quiet";
}

AudioScene AudioSceneClassifier::classify(const PcmAudio& audio,
                                          float& confidence) {
    confidence = 0.0f;
    std::vector<float> frames;
    int32_t bands = 0;
    if (!compute_frame_features(audio, 80, 24, frames, bands)) return AudioScene::Quiet;
    const ClipStats st = clip_stats(frames, bands, audio.sample_rate, 80);
    if (st.n_frames < 4) return AudioScene::Quiet;

    auto s01 = [](double v) { return std::min(1.0, std::max(0.0, v)); };

    // Quiet: overall low energy.
    if (st.energy_mean < 0.0004) {
        confidence = 0.9f;
        return AudioScene::Quiet;
    }

    const double speechy = s01((1.0 - st.periodicity) * 1.1) *
                           s01(st.band_flux * 2.2) *
                           (1.0 - s01(st.hf_ratio * 1.6));
    const double musicky = s01(st.periodicity * 1.2) * s01(st.band_flux * 2.0);
    const double streety = s01(st.energy_mean * 25.0) * s01(st.hf_ratio * 1.8) *
                           s01(st.band_flux * 1.4);
    const double kitchy = s01(st.onset_rate * 0.9) * s01(st.lf_ratio * 1.5) *
                          s01(st.energy_mean * 30.0);
    const double officey = s01(st.energy_mean * 12.0) *
                           (1.0 - s01(st.periodicity)) *
                           (1.0 - s01(st.band_flux * 2.6));

    struct Cand { AudioScene s; double c; };
    const Cand cands[] = {
        {AudioScene::Speech, speechy}, {AudioScene::Music, musicky},
        {AudioScene::Street, streety}, {AudioScene::Kitchen, kitchy},
        {AudioScene::Office, officey},
    };
    const Cand* best = &cands[0];
    for (const Cand& c : cands) if (c.c > best->c) best = &c;
    confidence = static_cast<float>(best->c);
    return best->s;
}

// ===========================================================================
// Anomaly detection
// ===========================================================================
bool AudioAnomalyDetector::push_frame(const float* bands, int32_t n_bands) {
    if (!bands || n_bands <= 0) return false;
    if (mean_.size() != static_cast<size_t>(n_bands)) {
        mean_.assign(n_bands, 0.0);
        m2_.assign(n_bands, 0.0);
        seen_ = 0;
    }

    double dev2 = 0.0;
    for (int32_t b = 0; b < n_bands; ++b) {
        const double x = bands[b];
        double& m = mean_[b];
        double& m2 = m2_[b];
        const double delta = x - m;
        if (seen_ >= cfg_.baseline_frames) {
            // Score against the frozen baseline.
            const double var = m2 / (seen_ - 1 > 0 ? seen_ - 1 : 1);
            const double sd = std::sqrt(var + 1e-9);
            dev2 += (delta * delta) / (sd * sd);
        }
        // Online update (Welford).
        ++seen_;
        m += delta / seen_;
        m2 += delta * (x - m);
    }

    if (seen_ <= cfg_.baseline_frames) return false;   // still learning

    // Mean chi-like distance across bands.
    last_dev_ = static_cast<float>(std::sqrt(dev2 / n_bands));
    return last_dev_ > cfg_.sensitivity;
}

} // namespace omniseed
