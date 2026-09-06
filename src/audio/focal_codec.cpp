// =============================================================================
//  OmniSeed — focal_codec.cpp
//  FocalCodec-style semantic audio tokenizer: PCM -> frame features ->
//  discrete semantic codes (0.16-0.65 kbps class). Weights come from the
//  GGUF in a later phase; until then a deterministic spectral-hash codebook
//  keeps the TokenBus pipeline exercisable and testable.
// =============================================================================
#include "omniseed/audio/audio.h"
#include "omniseed/core/platform.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace omniseed {

// ===========================================================================
// Frame features: log band energies (mel-spaced bands, dependency-free)
// ===========================================================================
bool FocalCodec::frame_features(const PcmAudio& audio, Tensor& frames) const {
    if (!audio.valid()) return false;

    const int32_t sr = audio.sample_rate;
    const int32_t frame_len = sr * cfg_.frame_ms / 1000;   // e.g. 1280 @ 80ms
    if (frame_len <= 0 || audio.samples.size() <
        static_cast<size_t>(frame_len)) {
        return false;
    }

    const int32_t n_bands = 24;
    const size_t n_frames = audio.samples.size() / frame_len;

    frames = Tensor("audio_frames",
                    {static_cast<int64_t>(n_frames), n_bands}, DType::F32);

    // Mel-spaced band edges (rough, Slaney-like placement).
    std::vector<int32_t> edges(static_cast<size_t>(n_bands) + 1);
    for (int32_t b = 0; b <= n_bands; ++b) {
        const double t = static_cast<double>(b) / n_bands;
        const double hz = 60.0 * std::pow(1000.0 / 60.0, t);  // log-spaced
        edges[static_cast<size_t>(b)] =
            std::min<int32_t>(sr / 2,
                static_cast<int32_t>(hz / (sr / 2.0) * (frame_len / 2)));
    }

    for (size_t f = 0; f < n_frames; ++f) {
        const float* fr = audio.samples.data() + f * frame_len;

        // Frame energy per band via zero-crossing-free Goertzel-lite:
        // one complex exponential per band center (cheap, no FFT).
        for (int32_t b = 0; b < n_bands; ++b) {
            const double hz = 60.0 * std::pow(1000.0 / 60.0,
                static_cast<double>(b) / n_bands);
            const double w = 2.0 * 3.14159265358979323846 * hz / sr;
            double re = 0.0, im = 0.0;
            for (int32_t t = 0; t < frame_len; t += 4) {   // decimate 4x
                const double s = fr[t];
                re += s * std::cos(w * t);
                im += s * std::sin(w * t);
            }
            const double mag = std::sqrt(re * re + im * im) /
                               (frame_len / 4);
            frames.f32()[static_cast<int64_t>(f) * n_bands + b] =
                static_cast<float>(std::log10(mag + 1e-8));
        }
    }
    return true;
}

// ===========================================================================
// Encode: frame features -> discrete semantic codes
// ===========================================================================
bool FocalCodec::encode(const PcmAudio& audio,
                        std::vector<int32_t>& out_codes) const {
    Tensor frames;
    if (!frame_features(audio, frames)) return false;

    const int64_t N = frames.dim(0);
    const int64_t B = frames.dim(1);
    out_codes.clear();
    out_codes.reserve(static_cast<size_t>(N));

    // Deterministic codebook: quantize each band to 4 bits, pack 6 bands
    // per 24-bit code, fold to codebook_size. This gives stable codes for
    // identical audio (testable) and spread for different content.
    for (int64_t f = 0; f < N; ++f) {
        const float* row = frames.f32() + f * B;
        uint32_t code = 0;
        for (int64_t b = 0; b < B && b < 24; ++b) {
            // log-energy in [-8, 2] -> 4-bit bucket
            const float v = std::min(2.0f, std::max(-8.0f, row[b]));
            const uint32_t q =
                static_cast<uint32_t>((v + 8.0f) / 10.0f * 15.0f) & 0xFu;
            code |= (q & 1u) << (b % 32);
        }
        out_codes.push_back(static_cast<int32_t>(code % 
            static_cast<uint32_t>(cfg_.codebook_size)));
    }
    return true;
}

} // namespace omniseed
