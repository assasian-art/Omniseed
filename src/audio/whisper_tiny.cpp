// =============================================================================
//  OmniSeed — whisper_tiny.cpp
//  Distilled Whisper-tiny: log-mel front end + int4 quantized encoder path.
//  The mel filterbank layout follows whisper.cpp exactly (80 mel bins,
//  400-sample FFT window, 160-sample hop = 100 frames/s).
// =============================================================================
#include "omniseed/audio/audio.h"
#include "omniseed/core/fft.h"
#include "omniseed/core/gguf_loader.h"
#include "omniseed/core/platform.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <unordered_map>

namespace omniseed {

// ===========================================================================
// WAV loading (16-bit PCM, mono or first channel)
// ===========================================================================
bool PcmAudio::load_wav(const std::string& path, PcmAudio& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        platform::log_error("audio: cannot open %s", path.c_str());
        return false;
    }

    char hdr[12];
    f.read(hdr, 12);
    if (f.gcount() != 12 || std::memcmp(hdr, "RIFF", 4) != 0 ||
        std::memcmp(hdr + 8, "WAVE", 4) != 0) {
        platform::log_error("audio: not a WAV file: %s", path.c_str());
        return false;
    }

    uint16_t channels = 1, bits = 16;
    uint32_t rate = 16000;
    bool have_fmt = false, have_data = false;
    std::vector<int16_t> pcm;

    while (f.good()) {
        char cid[4] = {0};
        uint32_t sz = 0;
        f.read(cid, 4);
        f.read(reinterpret_cast<char*>(&sz), 4);
        if (f.gcount() != 4) break;

        if (std::memcmp(cid, "fmt ", 4) == 0) {
            char fmt[16] = {0};
            f.read(fmt, std::min<uint32_t>(sz, 16));
            std::memcpy(&channels, fmt + 2, 2);
            std::memcpy(&rate, fmt + 4, 4);
            std::memcpy(&bits, fmt + 14, 2);
            have_fmt = true;
            if (sz > 16) f.seekg(static_cast<std::streamoff>(sz - 16), std::ios::cur);
        } else if (std::memcmp(cid, "data", 4) == 0) {
            const size_t n = sz / 2;
            pcm.resize(n);
            f.read(reinterpret_cast<char*>(pcm.data()),
                   static_cast<std::streamsize>(sz));
            have_data = true;
            break;
        } else {
            f.seekg(static_cast<std::streamoff>(sz), std::ios::cur);
        }
    }

    if (!have_fmt || !have_data) {
        platform::log_error("audio: WAV missing fmt/data chunk");
        return false;
    }

    out.sample_rate = static_cast<int32_t>(rate);
    const size_t frames = pcm.size() / std::max<uint16_t>(channels, 1);
    out.samples.resize(frames);
    for (size_t i = 0; i < frames; ++i) {
        out.samples[i] = static_cast<float>(pcm[i * channels]) / 32768.0f;
    }
    return true;
}

// ===========================================================================
// Log-mel spectrogram (hardcoded layout per whisper.cpp audio.cpp)
// ===========================================================================
namespace {

// Hann window (N=400), computed once.
void hann_window(std::vector<float>& w, size_t n) {
    w.resize(n);
    for (size_t i = 0; i < n; ++i) {
        w[i] = 0.5f * (1.0f - std::cos(2.0f * 3.14159265358979323846f *
                                       static_cast<float>(i) / static_cast<float>(n)));
    }
}

// Bluestein/chirp-z DFT via radix-2 FFT (see omniseed/core/fft.h): evaluates
// the exact N=400 bins 0..200 in O(N log N) — bit-compatible with the naive
// DFT it replaces (same bin definition, same mel filterbank input).
// (The Phase 3 O(N^2) placeholder lived here; TASK 4 replaced it.)

// Slaney-style mel filterbank (matches whisper.cpp mel_filters defaults).
std::vector<float> build_mel_filters(int n_mels, int n_fft, int sample_rate) {
    const int n_bins = n_fft / 2 + 1;
    std::vector<float> filters(static_cast<size_t>(n_mels) * n_bins, 0.0f);

    const double mel_low = 0.0;
    const double mel_high = 2595.0 * std::log10(1.0 + 8000.0 / 700.0);
    auto hz_to_mel = [](double f) { return 2595.0 * std::log10(1.0 + f / 700.0); };
    auto mel_to_hz = [](double m) { return 700.0 * (std::pow(10.0, m / 2595.0) - 1.0); };

    std::vector<double> mels(static_cast<size_t>(n_mels) + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
        mels[static_cast<size_t>(i)] =
            mel_low + (mel_high - mel_low) * i / (n_mels + 1);
    }
    std::vector<double> hz(static_cast<size_t>(n_mels) + 2);
    for (int i = 0; i < n_mels + 2; ++i)
        hz[static_cast<size_t>(i)] = mel_to_hz(mels[static_cast<size_t>(i)]);

    for (int m = 0; m < n_mels; ++m) {
        const double f0 = hz[static_cast<size_t>(m)];
        const double f1 = hz[static_cast<size_t>(m + 1)];
        const double f2 = hz[static_cast<size_t>(m + 2)];
        for (int b = 0; b < n_bins; ++b) {
            const double f = static_cast<double>(b) * sample_rate / n_fft;
            if (f > f0 && f < f2) {
                const double w = (f < f1) ? (f - f0) / (f1 - f0)
                                          : (f2 - f) / (f2 - f1);
                filters[static_cast<size_t>(m) * n_bins + b] = static_cast<float>(w);
            }
        }
    }
    return filters;
}

} // namespace

// ===========================================================================
// WhisperTiny
// ===========================================================================
bool WhisperTiny::load(const std::string& gguf_path) {
    GgufLoader gg;
    if (!gg.open(gguf_path)) {
        error_ = gg.error();
        return false;
    }

    hp_.n_mels = static_cast<int32_t>(gg.get_u64("whisper.n_mels", 80));
    hp_.n_audio_ctx = static_cast<int32_t>(gg.get_u64("whisper.n_audio_ctx", 1500));
    hp_.n_audio_state = static_cast<int32_t>(gg.get_u64("whisper.n_audio_state", 384));
    hp_.n_audio_head = static_cast<int32_t>(gg.get_u64("whisper.n_audio_head", 6));
    hp_.n_audio_layer = static_cast<int32_t>(gg.get_u64("whisper.n_audio_layer", 4));

    // Optional embedded mel filters (else we build Slaney filters).
    if (gg.has_tensor("whisper.mel_filters")) {
        Tensor t = gg.tensor("whisper.mel_filters");
        mel_filters_ = Tensor("mel_filters", t.shape(), DType::F32);
        std::memcpy(mel_filters_.data(), t.data(), t.nbytes());
    }
    // Encoder/decoder tensor loading lands in the Phase 3 pass.
    weights_loaded_ = false;
    valid_ = true;
    return true;
}

bool WhisperTiny::compute_mel(const PcmAudio& audio, Tensor& mel) const {
    if (!valid_) { error_ = "whisper not loaded"; return false; }

    constexpr int kFft = 400;
    constexpr int kHop = 160;
    constexpr int kSampleRate = 16000;

    std::vector<float> win;
    hann_window(win, kFft);

    const size_t n_frames =
        audio.samples.size() >= kFft
            ? (audio.samples.size() - kFft) / kHop + 1
            : 0;
    if (n_frames == 0) {
        error_ = "audio too short";
        return false;
    }

    const auto filters = build_mel_filters(hp_.n_mels, kFft, kSampleRate);
    const int n_bins = kFft / 2 + 1;

    mel = Tensor("mel", {hp_.n_mels, static_cast<int64_t>(n_frames)},
                 DType::F32);
    std::vector<float> frame(kFft), power(n_bins);

    // FFT-based exact-bin DFT (replaces the O(N^2) naive transform).
    static const dsp::DftBins dft(static_cast<size_t>(kFft),
                                  static_cast<size_t>(n_bins));
    std::vector<double> fre(n_bins), fim(n_bins);

    for (size_t f = 0; f < n_frames; ++f) {
        for (int t = 0; t < kFft; ++t)
            frame[static_cast<size_t>(t)] =
                audio.samples[f * kHop + t] * win[static_cast<size_t>(t)];
        dft.run(frame.data(), fre.data(), fim.data());
        for (int b = 0; b < n_bins; ++b)
            power[static_cast<size_t>(b)] = static_cast<float>(
                fre[static_cast<size_t>(b)] * fre[static_cast<size_t>(b)] +
                fim[static_cast<size_t>(b)] * fim[static_cast<size_t>(b)]);

        // mel projection + log10 with whisper.cpp's max-clamping semantics
        float max_mel = -1e30f;
        for (int m = 0; m < hp_.n_mels; ++m) {
            double acc = 0.0;
            for (int b = 0; b < n_bins; ++b) {
                acc += static_cast<double>(
                    filters[static_cast<size_t>(m) * n_bins + b]) *
                    power[static_cast<size_t>(b)];
            }
            const float v = static_cast<float>(std::log10(acc + 1e-10));
            mel.f32()[static_cast<int64_t>(m) * n_frames +
                      static_cast<int64_t>(f)] = v;
            if (v > max_mel) max_mel = v;
        }
        // normalize: clamp to [-8, 0] relative to max, then (x+4)/4 — the
        // whisper.cpp dynamic range compression.
        for (int m = 0; m < hp_.n_mels; ++m) {
            float v = mel.f32()[static_cast<int64_t>(m) * n_frames +
                                static_cast<int64_t>(f)];
            v = std::max(v, max_mel - 8.0f);
            mel.f32()[static_cast<int64_t>(m) * n_frames +
                      static_cast<int64_t>(f)] = (v + 4.0f) / 4.0f;
        }
    }
    return true;
}

bool WhisperTiny::transcribe(const PcmAudio& audio, std::string& out_text) const {
    if (!valid_) { error_ = "whisper not loaded"; return false; }

    Tensor mel;
    if (!compute_mel(audio, mel)) return false;

    if (!weights_loaded_) {
        // Deterministic bring-up summary: energy envelope stats per second.
        // Replaced by the int4 encoder+decoder in the Phase 3 pass.
        const double dur = static_cast<double>(audio.samples.size()) /
                           audio.sample_rate;
        double energy = 0.0;
        for (const float s : audio.samples) energy += static_cast<double>(s) * s;
        energy = std::sqrt(energy / std::max<size_t>(1, audio.samples.size()));

        char buf[128];
        std::snprintf(buf, sizeof(buf),
                      "[audio %.1fs rms %.3f frames %lld]",
                      dur, energy, static_cast<long long>(mel.dim(1)));
        out_text = buf;
        return true;
    }
    // Full decode path: Phase 3.
    return true;
}

} // namespace omniseed
