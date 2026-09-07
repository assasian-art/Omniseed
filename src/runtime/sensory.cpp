// =============================================================================
//  OmniSeed — sensory.cpp
//  Sensory Fingerprinting implementation.
//
//  All embedders are deterministic, dependency-free, and tiny (fixed 32-dim
//  embeddings, no heap churn per call) so the module costs <100 KB RAM.
// =============================================================================
#include "omniseed/runtime/sensory.h"
#include "omniseed/core/platform.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace omniseed {

namespace {

constexpr int32_t kDim = 32;
constexpr int32_t kStatDim = 8;

float dot32(const float* a, const float* b) {
    float d = 0.0f;
    for (int32_t i = 0; i < kDim; ++i) d += a[i] * b[i];
    return d;
}

void l2norm32(float* v) {
    double n2 = 0.0;
    for (int32_t i = 0; i < kDim; ++i) n2 += static_cast<double>(v[i]) * v[i];
    const float inv = static_cast<float>(1.0 / std::sqrt(n2 + 1e-12));
    for (int32_t i = 0; i < kDim; ++i) v[i] *= inv;
}

// ---------------------------------------------------------------------------
// Deterministic random projection of an 8-dim normalized stat vector into
// kDim dims. Projection entries are seeded pseudo-random (xorshift), so:
//   * the mapping is stable across runs and platforms,
//   * two DIFFERENT stat vectors map to genuinely different directions,
//     while identical inputs map identically.
// ---------------------------------------------------------------------------
float proj_entry(int32_t row, int32_t col) {
    uint32_t x = static_cast<uint32_t>(row * 2654435761u) ^
                 static_cast<uint32_t>(col * 0x9E3779B9u);
    x ^= x >> 15;
    x *= 0x2C1B3C6Du;
    x ^= x >> 12;
    x *= 0x297A2D39u;
    x ^= x >> 15;
    return static_cast<float>(x) / 2147483648.0f - 1.0f;
}

void embed_project(const double* stats, float* out) {
    for (int32_t r = 0; r < kDim; ++r) {
        double acc = 0.0;
        for (int32_t c = 0; c < kStatDim; ++c)
            acc += proj_entry(r, c) * stats[c];
        out[r] = static_cast<float>(acc);
    }
    l2norm32(out);
}

// Blend a fresh embedding into a stored one with weight w_new for the new
// sample (running weighted mean that converges to the average embedding).
void blend(float* stored, const float* fresh, float w_new) {
    for (int32_t i = 0; i < kDim; ++i)
        stored[i] = stored[i] * (1.0f - w_new) + fresh[i] * w_new;
    l2norm32(stored);
}

uint64_t fnv1a(const float* v, size_t n) {
    uint64_t h = 0xcbf29ce484222325ull;
    const auto* bytes = reinterpret_cast<const uint8_t*>(v);
    const size_t nbytes = n * sizeof(float);
    for (size_t i = 0; i < nbytes; ++i) {
        h ^= bytes[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

} // namespace

// ===========================================================================
// Voice embedding: REAL spectral band energies (Goertzel at log-spaced band
// centers, as in focal_codec.cpp) + ZCR -> random projection.
// NOTE: the first draft attributed energy to bands by SAMPLE INDEX, which
// measures frame position, not frequency - every tone looked alike. The
// Goertzel probe below is the fix.
// ===========================================================================
bool SensoryFingerprint::embed_voice(const SensoryObservation& obs, float* out) {
    if (obs.pcm == nullptr || obs.pcm_len < 1024) return false;

    const int32_t frame = obs.sample_rate / 50;      // 20 ms frames
    const size_t n_frames = obs.pcm_len / static_cast<size_t>(frame);
    if (n_frames < 8) return false;

    // 6 log-spaced band centers: 150, 300, 600, 1200, 2400, 4800 Hz.
    static const double kCenters[6] = {150.0, 300.0, 600.0, 1200.0, 2400.0, 4800.0};
    double band_energy[6] = {0, 0, 0, 0, 0, 0};

    double total_energy = 0.0, peak = 0.0;
    int32_t voiced = 0;
    double zcr_sum = 0.0;

    for (size_t f = 0; f < n_frames; ++f) {
        const float* fr = obs.pcm + f * frame;

        double e = 0.0, zc = 0.0;
        for (int32_t t = 1; t < frame; ++t) {
            e += static_cast<double>(fr[t]) * fr[t];
            if ((fr[t - 1] < 0) != (fr[t] < 0)) zc += 1.0;
        }
        e /= frame;
        zc /= frame;
        total_energy += e;
        peak = std::max(peak, e);

        if (e > 0.002) {                       // voiced frame
            ++voiced;
            zcr_sum += zc;
            // Goertzel magnitude at each band center (decimate 2x).
            for (int32_t b = 0; b < 6; ++b) {
                const double w = 2.0 * 3.14159265358979323846 *
                                 kCenters[b] / obs.sample_rate;
                double re = 0.0, im = 0.0;
                for (int32_t t = 0; t < frame; t += 2) {
                    const double s = fr[t];
                    re += s * std::cos(w * t);
                    im += s * std::sin(w * t);
                }
                band_energy[b] += std::sqrt(re * re + im * im) / (frame / 2);
            }
        }
    }

    if (voiced < 4) return false;              // not enough speech

    const double inv_v = 1.0 / voiced;

    // 8 stats: 6 log band energies (pitch-discriminative) + zcr + loudness.
    double stats[8];
    for (int32_t b = 0; b < 6; ++b) {
        const double m = band_energy[b] * inv_v;
        stats[b] = std::log1p(m * 100.0) / 2.0;
    }
    stats[6] = std::min(2.0, (zcr_sum * inv_v) * 20.0);        // pitch proxy
    stats[7] = std::log1p(total_energy * inv_v * 100.0) / 2.0; // loudness

    embed_project(stats, out);
    return true;
}

// ===========================================================================
// Typing embedding: digraph latency histogram + rhythm stats
// ===========================================================================
bool SensoryFingerprint::embed_typing(const SensoryObservation& obs, float* out) {
    if (obs.digraph_ms.empty()) return false;
    std::memset(out, 0, sizeof(float) * kDim);

    double mean = 0.0, m2 = 0.0, mn = 1e9, mx = 0.0;
    // 16 latency buckets from 0..1000 ms (log-ish edges).
    const double edges[17] = {0, 40, 60, 80, 100, 130, 170, 220, 280, 350,
                              430, 520, 620, 740, 870, 1000, 1e9};
    double hist[16] = {0};

    for (const float ms : obs.digraph_ms) {
        const double v = std::max(0.0, static_cast<double>(ms));
        mean += v;
        mn = std::min(mn, v);
        mx = std::max(mx, v);
        for (int32_t b = 0; b < 16; ++b) {
            if (v >= edges[b] && v < edges[b + 1]) { hist[b] += 1.0; break; }
        }
    }
    const double n = static_cast<double>(obs.digraph_ms.size());
    mean /= n;

    for (const float ms : obs.digraph_ms) {
        const double d = static_cast<double>(ms) - mean;
        m2 += d * d;
    }
    const double sd = std::sqrt(m2 / n);

    double stats[8] = { mean, sd, mn == 1e9 ? 0 : mn, mx, n,
                        sd / (mean + 1e-9), 0.0, 0.0 };
    // Histogram mass folded into remaining stats slots (16 buckets -> 2 slots).
    double h1 = 0, h2 = 0;
    for (int32_t b = 0; b < 16; ++b) {
        const double m = hist[b] / n;
        if (b < 8) h1 += m * (b + 1); else h2 += m * (b - 7);
    }
    stats[6] = h1; stats[7] = h2;
    for (double& s : stats) s = std::log1p(std::fabs(s)) / 3.0;

    embed_project(stats, out);
    return true;
}

// ===========================================================================
bool SensoryFingerprint::embed_vision_stats(const float* stats8, float* out) {
    if (stats8 == nullptr) return false;
    double normed[8];
    for (int32_t i = 0; i < 8; ++i)
        normed[i] = std::log1p(std::fabs(stats8[i])) / 3.0;
    embed_project(normed, out);
    return true;
}

// ===========================================================================
// Enroll: running weighted mean per modality
// ===========================================================================
bool SensoryFingerprint::enroll(const SensoryObservation& obs) {
    bool any = false;
    const float w = 0.34f;   // new-sample weight (fast early convergence)

    float v[kDim];
    if (embed_voice(obs, v)) {
        if (profile_.n_voice_enroll == 0) std::memcpy(profile_.voice, v, sizeof(v));
        else blend(profile_.voice, v, w);
        ++profile_.n_voice_enroll;
        any = true;
    }
    float t[kDim];
    if (embed_typing(obs, t)) {
        if (profile_.n_typing_enroll == 0) std::memcpy(profile_.typing, t, sizeof(t));
        else blend(profile_.typing, t, w);
        ++profile_.n_typing_enroll;
        any = true;
    }
    if (obs.vision_stats != nullptr) {
        float vi[kDim];
        if (embed_vision_stats(obs.vision_stats, vi)) {
            if (profile_.n_vision_enroll == 0)
                std::memcpy(profile_.vision, vi, sizeof(vi));
            else blend(profile_.vision, vi, w);
            ++profile_.n_vision_enroll;
            any = true;
        }
    }
    if (any) profile_.updated_at = static_cast<uint64_t>(platform::now_ms() / 1000.0);
    return any;
}

// ===========================================================================
// Verify: per-modality cosine with trust gating, weighted fusion
// ===========================================================================
float SensoryFingerprint::verify(const SensoryObservation& obs, bool& is_match) const {
    is_match = false;
    if (!enrolled()) return 0.0f;

    float sum = 0.0f;
    float wsum = 0.0f;
    auto contribute = [&](const float* fresh_ok, const float* stored,
                          uint32_t n_enroll, float weight) {
        if (!fresh_ok || n_enroll == 0) return;
        const float trust =
            n_enroll >= cfg_.min_enrollments ? 1.0f : 0.6f;
        sum += weight * trust * dot32(stored, fresh_ok);
        wsum += weight * trust;
    };

    float v[kDim];
    if (embed_voice(obs, v) && profile_.has_voice())
        contribute(v, profile_.voice, profile_.n_voice_enroll, 1.0f);
    float t[kDim];
    if (embed_typing(obs, t) && profile_.has_typing())
        contribute(t, profile_.typing, profile_.n_typing_enroll, 0.9f);
    if (obs.vision_stats != nullptr && profile_.has_vision()) {
        float vi[kDim];
        if (embed_vision_stats(obs.vision_stats, vi))
            contribute(vi, profile_.vision, profile_.n_vision_enroll, 0.7f);
    }

    if (wsum <= 0.0f) return 0.0f;
    const float fused = sum / wsum;            // in [-1, 1] -> clamp to [0,1]
    const float score = std::min(1.0f, std::max(0.0f, fused));
    is_match = score >= cfg_.verify_threshold;
    return score;
}

// ===========================================================================
std::string SensoryFingerprint::identity_token() const {
    if (!enrolled()) return "";
    // Fused profile vector -> two 64-bit FNV-1a halves -> 32 hex chars.
    float fused[2 * kDim];
    std::memcpy(fused, profile_.voice, sizeof(profile_.voice));
    std::memcpy(fused + kDim, profile_.typing, sizeof(profile_.typing));
    const uint64_t h1 = fnv1a(fused, kDim);
    const uint64_t h2 = fnv1a(fused + kDim, kDim);
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%016llx%016llx",
                  static_cast<unsigned long long>(h1),
                  static_cast<unsigned long long>(h2));
    return std::string(buf);
}

// ===========================================================================
// Persistence: 'SNFP' binary
// ===========================================================================
bool SensoryFingerprint::save(const std::string& path) const {
    FILE* f = platform::open_file_c(path.c_str(), "wb");
    if (!f) return false;
    const uint32_t magic = 0x50464E53;  // SNFP
    const uint32_t version = 1;
    std::fwrite(&magic, 4, 1, f);
    std::fwrite(&version, 4, 1, f);
    SensoryProfile p = profile_;
    std::fwrite(&p, sizeof(p), 1, f);
    std::fclose(f);
    return true;
}

bool SensoryFingerprint::load(const std::string& path) {
    platform::MappedFile mf;
    if (!mf.open(path)) return false;
    const uint8_t* p = mf.bytes();
    if (mf.size() < 8 + sizeof(SensoryProfile)) return false;
    if (std::memcmp(p, "SNFP", 4) != 0) return false;
    const uint32_t version = *reinterpret_cast<const uint32_t*>(p + 4);
    if (version != 1) return false;
    std::memcpy(&profile_, p + 8, sizeof(SensoryProfile));
    return true;
}

} // namespace omniseed
