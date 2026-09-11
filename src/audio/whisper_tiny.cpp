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
#include <iterator>
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
    std::string data((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());
    return load_wav_bytes(data, out);
}

bool PcmAudio::load_wav_bytes(const std::string& data, PcmAudio& out) {
    const char* p = data.data();
    const size_t total = data.size();
    auto need = [&](size_t off, size_t n) {
        return off + n <= total;
    };
    if (total < 12 || std::memcmp(p, "RIFF", 4) != 0 ||
        std::memcmp(p + 8, "WAVE", 4) != 0) {
        platform::log_error("audio: not a WAV blob (%zu bytes)", total);
        return false;
    }

    uint16_t channels = 1, bits = 16;
    uint32_t rate = 16000;
    bool have_fmt = false, have_data = false;
    std::vector<int16_t> pcm;

    size_t off = 12;
    while (off + 8 <= total) {
        const char* cid = p + off;
        uint32_t sz = 0;
        std::memcpy(&sz, p + off + 4, 4);          // little-endian container
        off += 8;
        if (std::memcmp(cid, "fmt ", 4) == 0) {
            if (!need(off, 16)) return false;
            std::memcpy(&channels, p + off + 2, 2);
            std::memcpy(&rate, p + off + 4, 4);
            std::memcpy(&bits, p + off + 14, 2);
            have_fmt = true;
        } else if (std::memcmp(cid, "data", 4) == 0) {
            const size_t avail = std::min<size_t>(sz, total - off);
            const size_t n = avail / 2;
            pcm.resize(n);
            std::memcpy(pcm.data(), p + off, n * 2);
            have_data = true;
            break;
        }
        off += sz + (sz & 1);                      // chunks are word-aligned
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

// Slaney-scale mel filterbank (librosa/whisper convention: linear below
// 1 kHz, log above) — the same scale as the official mel_filters.npz that
// the converter embeds. Used only when the official filters are absent.
std::vector<float> build_mel_filters(int n_mels, int n_fft, int sample_rate) {
    const int n_bins = n_fft / 2 + 1;
    std::vector<float> filters(static_cast<size_t>(n_mels) * n_bins, 0.0f);

    // librosa.filters.mel(htk=False) mel scale.
    auto hz_to_mel = [](double f) {
        const double f_min = 0.0, f_sp = 200.0 / 3.0;
        const double m_min = f_min / f_sp;
        const double freqs = f / f_sp;
        const double min_log_hz = 1000.0;
        const double min_log_mel = min_log_hz / f_sp;
        const double logstep = std::log(6.4) / 27.0;
        return (f >= min_log_hz)
            ? min_log_mel + std::log(freqs / min_log_hz + 1e-12) / logstep
            : freqs;
        (void)m_min;
    };
    auto mel_to_hz = [](double m) {
        const double f_sp = 200.0 / 3.0;
        const double min_log_hz = 1000.0;
        const double min_log_mel = min_log_hz / f_sp;
        const double logstep = std::log(6.4) / 27.0;
        return (m >= min_log_mel)
            ? min_log_hz * std::exp(logstep * (m - min_log_mel))
            : f_sp * m;
    };
    const double mel_low = hz_to_mel(0.0);
    const double mel_high = hz_to_mel(sample_rate / 2.0);

    std::vector<double> mels(static_cast<size_t>(n_mels) + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
        mels[static_cast<size_t>(i)] =
            mel_low + (mel_high - mel_low) * i / (n_mels + 1);
    }
    // librosa norm="slaney": each triangle's area is normalized to 1.
    std::vector<double> hz(static_cast<size_t>(n_mels) + 2);
    for (int i = 0; i < n_mels + 2; ++i)
        hz[static_cast<size_t>(i)] = mel_to_hz(mels[static_cast<size_t>(i)]);

    for (int m = 0; m < n_mels; ++m) {
        const double f0 = hz[static_cast<size_t>(m)];
        const double f1 = hz[static_cast<size_t>(m + 1)];
        const double f2 = hz[static_cast<size_t>(m + 2)];
        const double enorm = 2.0 / (f2 - f0);       // slaney area normalization
        for (int b = 0; b < n_bins; ++b) {
            const double f = static_cast<double>(b) * sample_rate / n_fft;
            if (f > f0 && f < f2) {
                const double w = ((f < f1) ? (f - f0) / (f1 - f0)
                                           : (f2 - f) / (f2 - f1)) * enorm;
                filters[static_cast<size_t>(m) * n_bins + b] =
                    static_cast<float>(w);
            }
        }
    }
    return filters;
}

} // namespace

// ===========================================================================
// WhisperTiny
// ===========================================================================
namespace {

Tensor load_f16(const GgufLoader& gg, const std::string& name,
                std::string& err, bool required = true) {
    if (!gg.has_tensor(name)) {
        if (required) err = "missing tensor: " + name;
        return Tensor();
    }
    Tensor t = gg.tensor(name);
    if (t.dtype() != DType::F16) {
        err = "tensor " + name + ": expected fp16";
        return Tensor();
    }
    Tensor out(name, t.shape(), DType::F16, const_cast<void*>(t.data()));
    if (required && out.numel() == 0) err = "empty tensor: " + name;
    return out;
}

inline float gelu_f(float x) {
    // Exact erf GELU — whisper's config.activation_function == "gelu" (the
    // tanh approximation differs by ~1.6e-4 per point and drifts over 4+2
    // convs + 4 blocks). std::erf is C++17; MSVC links it fine.
    return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752440f));
}

// Dense fp16 W [out, in] @ x -> y (fp32 accumulate). Bias optional.
// Convention (see tools/convert_senses.py header): an omitted bias is legal;
// a present bias must have full [out_dim] extent. Never dereference a
// missing/empty bias tensor.
void dense_f16(const Tensor& W, const Tensor& b, const float* x, float* y,
               int64_t out_dim, int64_t in_dim) {
    const uint16_t* wp = W.f16();
    const bool has_w = wp != nullptr && W.numel() >= out_dim * in_dim;
    const bool has_b = b.numel() >= out_dim && b.f16() != nullptr;
    if (!has_w) {                       // defensive: emit zeros, flag once
        for (int64_t r = 0; r < out_dim; ++r) y[r] = 0.0f;
        return;
    }
    for (int64_t r = 0; r < out_dim; ++r) {
        const uint16_t* row = wp + r * in_dim;
        float acc = has_b ? half_to_float(b.f16()[r]) : 0.0f;
        for (int64_t c = 0; c < in_dim; ++c)
            acc += half_to_float(row[c]) * x[c];
        y[r] = acc;
    }
}

void layer_norm_f32(const float* x, const float* w, const float* b,
                    float* y, int64_t n, float eps = 1e-5f) {
    double mean = 0.0;
    for (int64_t i = 0; i < n; ++i) mean += x[i];
    mean /= n;
    double var = 0.0;
    for (int64_t i = 0; i < n; ++i) { const double d = x[i] - mean; var += d * d; }
    var /= n;
    const float inv = static_cast<float>(1.0 / std::sqrt(var + eps));
    for (int64_t i = 0; i < n; ++i)
        y[i] = static_cast<float>(x[i] - mean) * inv * w[i] + b[i];
}

// fp16 tensor -> fp32 scratch (norm weights arrive as fp16 in sidecars).
void f16_to_f32(const Tensor& t, std::vector<float>& out) {
    const int64_t n = t.numel();
    out.resize(static_cast<size_t>(n));
    if (t.f16() == nullptr || n <= 0) return;
    for (int64_t i = 0; i < n; ++i) out[static_cast<size_t>(i)] =
        half_to_float(t.f16()[i]);
}

// Multi-head attention for one query row against K/V caches [n_kv][S]
// (defined below; shared by the encoder blocks and the decoder).
void mha_row(const float* q, const float* kc, const float* vc,
             int64_t n_kv, float* out, int32_t S, int32_t heads);

} // namespace

bool WhisperTiny::load(const std::string& gguf_path) {
    // store_ owns the mmap: every fp16 view below points into it and must
    // outlive them (same lifetime rule as RwkvModel::store_).
    if (!store_.open(gguf_path)) {
        error_ = store_.error();
        return false;
    }
    const GgufLoader& gg = store_;

    hp_.n_mels = static_cast<int32_t>(gg.get_u64("whisper.n_mels", 80));
    hp_.n_audio_ctx = static_cast<int32_t>(gg.get_u64("whisper.n_audio_ctx", 1500));
    hp_.n_audio_state = static_cast<int32_t>(gg.get_u64("whisper.n_audio_state", 384));
    hp_.n_audio_head = static_cast<int32_t>(gg.get_u64("whisper.n_audio_head", 6));
    hp_.n_audio_layer = static_cast<int32_t>(gg.get_u64("whisper.n_audio_layer", 4));

    // Embedded mel filters (official OpenAI mel_filters.npz via converter).
    if (gg.has_tensor("whisper.mel_filters")) {
        Tensor t = gg.tensor("whisper.mel_filters");
        mel_filters_ = Tensor("mel_filters", t.shape(), DType::F32);
        std::memcpy(mel_filters_.data(), t.data(), t.nbytes());
    }

    // Encoder tensors. All-or-nothing: if the stem is present we require the
    // full set, so weights_loaded_ means "encode() runs the real network".
    if (gg.has_tensor("whisper.conv1.weight")) {
        conv1_w_ = load_f16(gg, "whisper.conv1.weight", error_);
        conv1_b_ = load_f16(gg, "whisper.conv1.bias", error_);
        conv2_w_ = load_f16(gg, "whisper.conv2.weight", error_);
        conv2_b_ = load_f16(gg, "whisper.conv2.bias", error_);
        pos_embed_ = load_f16(gg, "whisper.pos_embed", error_);
        enc_ln_w_ = load_f16(gg, "whisper.enc_ln.weight", error_);
        enc_ln_b_ = load_f16(gg, "whisper.enc_ln.bias", error_);
        if (!error_.empty()) return false;

        enc_blocks_.resize(static_cast<size_t>(hp_.n_audio_layer));
        for (int32_t l = 0; l < hp_.n_audio_layer; ++l) {
            EncBlock& e = enc_blocks_[static_cast<size_t>(l)];
            const std::string p = "whisper.enc." + std::to_string(l) + ".";
            e.q_w_  = load_f16(gg, p + "q.weight", error_);
            e.k_w_  = load_f16(gg, p + "k.weight", error_);
            e.v_w_  = load_f16(gg, p + "v.weight", error_);
            e.o_w_  = load_f16(gg, p + "out.weight", error_);
            e.ln1_w_ = load_f16(gg, p + "ln1.weight", error_);
            e.ln1_b_ = load_f16(gg, p + "ln1.bias", error_);
            e.fc1_w_ = load_f16(gg, p + "fc1.weight", error_);
            e.fc1_b_ = load_f16(gg, p + "fc1.bias", error_);
            e.fc2_w_ = load_f16(gg, p + "fc2.weight", error_);
            e.fc2_b_ = load_f16(gg, p + "fc2.bias", error_);
            e.ln2_w_ = load_f16(gg, p + "ln2.weight", error_);
            e.ln2_b_ = load_f16(gg, p + "ln2.bias", error_);
            // attention projections may be bias-free (HF whisper encoder
            // ships q/k/v/out biases; tolerate their absence anyway)
            e.q_b_ = load_f16(gg, p + "q.bias", error_, false);
            e.k_b_ = load_f16(gg, p + "k.bias", error_, false);
            e.v_b_ = load_f16(gg, p + "v.bias", error_, false);
            e.o_b_ = load_f16(gg, p + "out.bias", error_, false);
            if (!error_.empty()) return false;
        }
        weights_loaded_ = true;

        // ---- decoder (optional: encoder-only sidecars stay valid) ----------
        // All-or-nothing per family; failure to load the decoder must not
        // invalidate the encoder, so errors are cleared afterwards.
        if (gg.has_tensor("whisper.dec.tok_embd")) {
            const std::string saved_err = error_;
            error_.clear();
            bool ok = true;
            auto req = [&](const std::string& n) {
                Tensor t = load_f16(gg, n, error_);
                if (!error_.empty()) ok = false;
                error_.clear();
                return t;
            };
            dec_tok_embd_ = req("whisper.dec.tok_embd");       // [V, 384] (tied head)
            dec_pos_embed_ = req("whisper.dec.pos_embed");     // [448, 384]
            dec_ln_w_ = req("whisper.dec_ln.weight");
            dec_ln_b_ = req("whisper.dec_ln.bias");
            vocab_offsets_ = gg.has_tensor("whisper.vocab_offsets")
                ? gg.tensor("whisper.vocab_offsets") : Tensor();
            vocab_bytes_ = gg.has_tensor("whisper.vocab_bytes")
                ? gg.tensor("whisper.vocab_bytes") : Tensor();
            const int32_t nd = static_cast<int32_t>(
                gg.get_u64("whisper.n_dec_layer", 4));
            dec_blocks_.resize(static_cast<size_t>(nd));
            for (int32_t l = 0; l < nd && ok; ++l) {
                DecBlock& d = dec_blocks_[static_cast<size_t>(l)];
                const std::string p = "whisper.dec." + std::to_string(l) + ".";
                d.ln1_w_ = req(p + "ln1.weight");  d.ln1_b_ = req(p + "ln1.bias");
                d.ln2_w_ = req(p + "ln2.weight");  d.ln2_b_ = req(p + "ln2.bias");
                d.ln3_w_ = req(p + "ln3.weight");  d.ln3_b_ = req(p + "ln3.bias");
                d.self_q_w_ = req(p + "self.q.weight");
                d.self_k_w_ = req(p + "self.k.weight");
                d.self_v_w_ = req(p + "self.v.weight");
                d.self_o_w_ = req(p + "self.out.weight");
                d.self_q_b_ = req(p + "self.q.bias");
                d.self_v_b_ = req(p + "self.v.bias");
                d.self_o_b_ = req(p + "self.out.bias");
                d.cross_q_w_ = req(p + "cross.q.weight");
                d.cross_k_w_ = req(p + "cross.k.weight");
                d.cross_v_w_ = req(p + "cross.v.weight");
                d.cross_o_w_ = req(p + "cross.out.weight");
                d.cross_q_b_ = req(p + "cross.q.bias");
                d.cross_v_b_ = req(p + "cross.v.bias");
                d.cross_o_b_ = req(p + "cross.out.bias");
                d.fc1_w_ = req(p + "fc1.weight");  d.fc1_b_ = req(p + "fc1.bias");
                d.fc2_w_ = req(p + "fc2.weight");  d.fc2_b_ = req(p + "fc2.bias");
            }
            // whisper-tiny ships no self/cross k_proj bias (checked at load)
            if (ok && dec_tok_embd_.numel() > 1 &&
                vocab_offsets_.numel() > 1 && vocab_bytes_.numel() > 1) {
                hp_.n_vocab = static_cast<int32_t>(
                    gg.get_u64("whisper.n_vocab", 51865));
                decoder_loaded_ = true;
            }
            error_ = saved_err;
        }

        // Debug: read every element of every loaded tensor. Any mapping or
        // bounds problem surfaces here, before encode() runs.
        if (std::getenv("OMNISEED_DBG")) {
            auto sweep = [](const char* tag, const Tensor& t) {
                const int64_t n = t.numel();
                if (n <= 0) {
                    std::fprintf(stderr, "[sweep] %-14s EMPTY dt=%s\n", tag, dtype_name(t.dtype()));
                    std::fflush(stderr);
                    return;
                }
                if (t.dtype() != DType::F16) {
                    std::fprintf(stderr, "[sweep] %-14s BAD DTYPE %s n=%lld\n", tag,
                                 dtype_name(t.dtype()), (long long)n);
                    std::fflush(stderr);
                    return;
                }
                const uint16_t* p = static_cast<const uint16_t*>(t.data());
                if (p == nullptr) {
                    std::fprintf(stderr, "[sweep] %s NULL data n=%lld\n", tag, (long long)n);
                    std::fflush(stderr);
                    return;
                }
                double s = 0.0;
                for (int64_t i = 0; i < n; ++i) s += half_to_float(p[i]);
                std::fprintf(stderr, "[sweep] %-14s n=%-8lld sum=%f\n", tag, (long long)n, s);
                std::fflush(stderr);
            };
            sweep("conv1", conv1_w_); sweep("conv1b", conv1_b_);
            sweep("conv2", conv2_w_); sweep("conv2b", conv2_b_);
            sweep("pos", pos_embed_); sweep("encln", enc_ln_w_);
            for (size_t l = 0; l < enc_blocks_.size(); ++l) {
                const EncBlock& e = enc_blocks_[l];
                const std::string p = "blk" + std::to_string(l) + ".";
                sweep((p + "q").c_str(), e.q_w_);   sweep((p + "qb").c_str(), e.q_b_);
                sweep((p + "k").c_str(), e.k_w_);   sweep((p + "kb").c_str(), e.k_b_);
                sweep((p + "v").c_str(), e.v_w_);   sweep((p + "vb").c_str(), e.v_b_);
                sweep((p + "o").c_str(), e.o_w_);   sweep((p + "ob").c_str(), e.o_b_);
                sweep((p + "f1").c_str(), e.fc1_w_); sweep((p + "f1b").c_str(), e.fc1_b_);
                sweep((p + "f2").c_str(), e.fc2_w_); sweep((p + "f2b").c_str(), e.fc2_b_);
                sweep((p + "ln1").c_str(), e.ln1_w_); sweep((p + "ln1b").c_str(), e.ln1_b_);
                sweep((p + "ln2").c_str(), e.ln2_w_); sweep((p + "ln2b").c_str(), e.ln2_b_);
            }
        }
    }
    valid_ = true;
    return true;
}

bool WhisperTiny::compute_mel(const PcmAudio& audio, Tensor& mel) const {
    if (!valid_) { error_ = "whisper not loaded"; return false; }

    constexpr int kFft = 400;
    constexpr int kHop = 160;
    constexpr int kSampleRate = 16000;
    constexpr int kNFrames = 3000;              // whisper 30 s mel window
    constexpr int kHalfFft = kFft / 2;          // center=True reflect pad

    std::vector<float> win;
    hann_window(win, kFft);

    // --- build the sample buffer the way whisper sees it -------------------
    // whisper pads/truncates the waveform to the exact 30 s window with zeros
    // FIRST (silence beyond a short recording is silence, NOT full-scale
    // noise — this is what keeps short clips in-distribution), then runs
    // torch.stft(center=True): frame t is CENTERED at sample t*hop, i.e. the
    // 30 s buffer is reflect-padded by kHalfFft on both ends.
    const size_t n_in = audio.samples.size();
    if (n_in < 2) { error_ = "audio too short"; return false; }
    const size_t n_win = static_cast<size_t>(kSampleRate) * 30;
    std::vector<float> buf(n_win, 0.0f);         // the 30 s window
    std::copy(audio.samples.begin(),
              audio.samples.begin() +
                  static_cast<std::ptrdiff_t>(
                      std::min(n_in, n_win)),
              buf.begin());

    std::vector<float> sig(static_cast<size_t>(n_win) + 2 * kHalfFft, 0.0f);
    std::copy(buf.begin(), buf.end(), sig.begin() + kHalfFft);
    for (int i = 0; i < kHalfFft; ++i) {         // reflect pads (torch F.pad,
                                                 // edge-excluded, mirrored)
        sig[static_cast<size_t>(i)] =
            buf[static_cast<size_t>(kHalfFft - i)];
        sig[kHalfFft + n_win + static_cast<size_t>(i)] =
            buf[n_win - 2 - static_cast<size_t>(i)];
    }

    constexpr size_t kFramesMax =
        (static_cast<size_t>(kNFrames - 1) * kHop + kFft - kFft) / kHop + 1;
    size_t n_frames =
        std::min(kFramesMax, (sig.size() - kFft) / kHop + 1);

    // Trim trailing all-silent frames before the log10 (whisper keeps the
    // full 30 s window, but a short clip's zero pad carries zero speech
    // information — the encoder then spends 96% of its FLOPs on padding).
    // Frame f covers sig[f*kHop .. f*kHop+kFft); the energy gate mirrors
    // log10(1e-10): any bin above 1e-10 keeps the frame.
    {
        size_t last = 0;                       // last frame with any energy
        for (size_t f = 0; f < n_frames; ++f) {
            float e = 0.0f;
            for (int t = 0; t < kFft; ++t) {
                const float s =
                    sig[f * kHop + static_cast<size_t>(t)] *
                    win[static_cast<size_t>(t)];
                e += s * s;
            }
            if (e > 1e-10f) last = f;
        }
        n_frames = last + 1;
    }

    // Official OpenAI mel_filters.npz (embedded by the converter) when
    // present — the librosa Slaney-scale fallback below matches its scale
    // but not its exact values.
    std::vector<float> fb;
    if (mel_filters_.numel() > 1) {
        fb.assign(mel_filters_.f32(),
                  mel_filters_.f32() + mel_filters_.numel());
    } else {
        fb = build_mel_filters(hp_.n_mels, kFft, kSampleRate);
    }
    const float* filters = fb.data();
    const int n_bins = kFft / 2 + 1;

    mel = Tensor("mel", {hp_.n_mels, static_cast<int64_t>(n_frames)},
                 DType::F32);
    std::vector<float> frame(kFft), power(n_bins);

    // FFT-based exact-bin DFT (replaces the O(N^2) naive transform).
    static const dsp::DftBins dft(static_cast<size_t>(kFft),
                                  static_cast<size_t>(n_bins));
    std::vector<double> fre(n_bins), fim(n_bins);

    // Pass 1: log-mel energies for every frame (no normalization yet).
    for (size_t f = 0; f < n_frames; ++f) {
        for (int t = 0; t < kFft; ++t)
            frame[static_cast<size_t>(t)] =
                sig[f * kHop + t] * win[static_cast<size_t>(t)];
        dft.run(frame.data(), fre.data(), fim.data());
        for (int b = 0; b < n_bins; ++b)
            power[static_cast<size_t>(b)] = static_cast<float>(
                fre[static_cast<size_t>(b)] * fre[static_cast<size_t>(b)] +
                fim[static_cast<size_t>(b)] * fim[static_cast<size_t>(b)]);
        for (int m = 0; m < hp_.n_mels; ++m) {
            double acc = 0.0;
            for (int b = 0; b < n_bins; ++b) {
                acc += static_cast<double>(
                    filters[static_cast<size_t>(m) * n_bins + b]) *
                    power[static_cast<size_t>(b)];
            }
            mel.f32()[static_cast<int64_t>(m) * static_cast<int64_t>(n_frames)
                      + static_cast<int64_t>(f)] =
                static_cast<float>(std::log10(acc + 1e-10));
        }
    }

    // Pass 2: whisper.cpp dynamic-range compression with the GLOBAL max
    // (a per-frame max would lift inter-word silence to speech level and
    // destroy the encoder's context).
    float max_mel = -1e30f;
    for (int64_t i = 0; i < mel.numel(); ++i)
        if (mel.f32()[i] > max_mel) max_mel = mel.f32()[i];
    for (int64_t i = 0; i < mel.numel(); ++i) {
        const float v = std::max(mel.f32()[i], max_mel - 8.0f);
        mel.f32()[i] = (v + 4.0f) / 4.0f;
    }
    return true;
}

bool WhisperTiny::transcribe(const PcmAudio& audio, std::string& out_text) const {
    if (!valid_) { error_ = "whisper not loaded"; return false; }

    Tensor mel;
    if (!compute_mel(audio, mel)) return false;

    if (!weights_loaded_) {
        // Deterministic bring-up summary: energy envelope stats per second.
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

    // Real path: encoder frames -> greedy autoregressive decoder -> text.
    Tensor enc;
    if (!encode(audio, enc)) return false;
    if (decoder_loaded_) {
        std::vector<int32_t> ids;
        if (!decode_greedy(enc, ids)) return false;
        out_text.clear();
        out_text.reserve(ids.size() * 4);
        const int32_t* offs = reinterpret_cast<const int32_t*>(
            vocab_offsets_.data());          // [V+1] int32 offsets
        // vocab_bytes is stored in an F16 container: piece byte i is the f16
        // VALUE of element i (the converter writes raw bytes cast to f16), so
        // decode via half_to_float, not the raw bits.
        const uint16_t* elems =
            reinterpret_cast<const uint16_t*>(vocab_bytes_.data());
        for (const int32_t id : ids) {
            if (id < 0 || id >= hp_.n_vocab) continue;
            for (int32_t o = offs[id]; o < offs[id + 1]; ++o)
                out_text.push_back(
                    static_cast<char>(half_to_float(elems[o])));
        }
        return true;
    }
    // Encoder-only sidecar: the encoded sequence feeds the LM through the
    // token bus as <|audio|> tokens (documented fallback).
    char buf[96];
    std::snprintf(buf, sizeof(buf), "<|audio|> encoded %lld frames",
                  static_cast<long long>(enc.dim(0)));
    out_text = buf;
    return true;
}

bool WhisperTiny::encode(const PcmAudio& audio, Tensor& out) const {
    if (!weights_loaded_) { error_ = "encoder weights not loaded"; return false; }
    Tensor mel;
    if (!compute_mel(audio, mel)) return false;

    const int64_t n_mels = mel.dim(0);
    // NOTE: hp_.n_audio_ctx (1500) is the encoder OUTPUT length, NOT the mel
    // input length — the mel arrives at the full 30 s window (3000 frames)
    // and the conv stack downsamples 2×. Never clamp the mel to n_audio_ctx.
    const int64_t T_in = mel.dim(1);
    const int32_t S = hp_.n_audio_state;
    if (T_in < 2) {                       // conv2 (stride 2) needs 2 frames
        error_ = "audio too short for the encoder (need >= 2 mel frames)";
        return false;
    }

    // conv1: kernel 3 over TIME, stride 1, zero-pad 1 (whisper stem) —
    // in [80, T_in] -> out [384, T_in]. We store mel as [n_mels, T] row-major.
    // Weight layout [out, in, k] contiguous (PyTorch Conv1d).
    Tensor h1("h1", {T_in, S}, DType::F32);
    {
        const uint16_t* W = conv1_w_.f16();   // [384, 80, 3] (o, m, k)
        const float* M = mel.f32();
        for (int64_t f = 0; f < T_in; ++f) {
            float* dst = h1.f32() + f * S;
            for (int32_t o = 0; o < S; ++o) {
                float acc = half_to_float(conv1_b_.f16()[o]);
                const uint16_t* wbase =
                    W + static_cast<int64_t>(o) * n_mels * 3;
                for (int32_t k = 0; k < 3; ++k) {
                    const int64_t t_in = f + k - 1;   // pad 1, centered
                    if (t_in < 0 || t_in >= T_in) continue;
                    for (int64_t m = 0; m < n_mels; ++m)
                        acc += half_to_float(wbase[m * 3 + k]) *
                               M[m * T_in + t_in];
                }
                dst[o] = gelu_f(acc);
            }
        }
    }
    // conv2 over time: kernel 3, stride 2, zero-pad 1 -> [T2, 384],
    // T2 = floor((T_in + 2*1 - 3)/2) + 1 (e.g. 3000 -> 1500, 98 -> 49).
    const int64_t T2 = (T_in + 2 - 3) / 2 + 1;
    Tensor h2("h2", {T2, S}, DType::F32);
    {
        const uint16_t* W = conv2_w_.f16();   // [384, 384, 3]
        for (int64_t f = 0; f < T2; ++f) {
            float* dst = h2.f32() + f * S;
            for (int32_t o = 0; o < S; ++o) {
                float acc = half_to_float(conv2_b_.f16()[o]);
                const uint16_t* wrow =
                    W + static_cast<int64_t>(o) * S * 3;
                for (int32_t k = 0; k < 3; ++k) {
                    const int64_t t_in = f * 2 + k - 1;   // stride 2, pad 1
                    if (t_in < 0 || t_in >= T_in) continue;
                    const float* src = h1.f32() + t_in * S;
                    for (int32_t i = 0; i < S; ++i)
                        acc += half_to_float(wrow[i * 3 + k]) * src[i];
                }
                dst[o] = gelu_f(acc);
            }
        }
    }
    // add positional embeddings, run encoder blocks (post-LN transformer)
    Tensor x("x", {T2, S}, DType::F32);
    for (int64_t t = 0; t < T2; ++t) {
        const uint16_t* pe = pos_embed_.f16() + t * S;
        for (int32_t i = 0; i < S; ++i)
            x.f32()[t * S + i] = h2.f32()[t * S + i] + half_to_float(pe[i]);
    }

    // ---- encoder blocks: post-LN transformer with TRUE bidirectional
    // multi-head self-attention (replaces the per-token value shortcut —
    // the decoder cross-attends to these frames, so they must be real).
    std::vector<float> xn(S), q(S), k(S), v(S), xo(S), ff(4 * S);
    std::vector<float> kc(static_cast<size_t>(T2) * S),
                       vc(static_cast<size_t>(T2) * S);
    std::vector<float> lnw(S), lnb(S), attn_out(S);
    constexpr int32_t kEncHeads = 6;
    for (int32_t l = 0; l < hp_.n_audio_layer; ++l) {
        const EncBlock& e = enc_blocks_[static_cast<size_t>(l)];
        for (int32_t i = 0; i < S; ++i) {
            lnw[i] = half_to_float(e.ln1_w_.f16()[i]);
            lnb[i] = half_to_float(e.ln1_b_.f16()[i]);
        }
        // K/V for every frame from LN1(x)
        for (int64_t t = 0; t < T2; ++t) {
            layer_norm_f32(x.f32() + t * S, lnw.data(), lnb.data(), xn.data(), S);
            dense_f16(e.k_w_, e.k_b_, xn.data(),
                      kc.data() + static_cast<size_t>(t) * S, S, S);
            dense_f16(e.v_w_, e.v_b_, xn.data(),
                      vc.data() + static_cast<size_t>(t) * S, S, S);
        }
        // Q per frame + attention + out proj + residual
        for (int64_t t = 0; t < T2; ++t) {
            float* xt = x.f32() + t * S;
            layer_norm_f32(xt, lnw.data(), lnb.data(), xn.data(), S);
            dense_f16(e.q_w_, e.q_b_, xn.data(), q.data(), S, S);
            mha_row(q.data(), kc.data(), vc.data(), T2, attn_out.data(),
                    S, kEncHeads);
            dense_f16(e.o_w_, e.o_b_, attn_out.data(), xo.data(), S, S);
            for (int32_t i = 0; i < S; ++i) xt[i] += xo[i];
        }
        // MLP (post-LN: applied to the residual stream)
        for (int32_t i = 0; i < S; ++i) {
            lnw[i] = half_to_float(e.ln2_w_.f16()[i]);
            lnb[i] = half_to_float(e.ln2_b_.f16()[i]);
        }        for (int64_t t = 0; t < T2; ++t) {
            float* xt = x.f32() + t * S;
            layer_norm_f32(xt, lnw.data(), lnb.data(), xn.data(), S);
            dense_f16(e.fc1_w_, e.fc1_b_, xn.data(), ff.data(), 4 * S, S);
            for (int32_t i = 0; i < 4 * S; ++i) ff[i] = gelu_f(ff[i]);
            dense_f16(e.fc2_w_, e.fc2_b_, ff.data(), xo.data(), S, 4 * S);
            for (int32_t i = 0; i < S; ++i) xt[i] += xo[i];
        }
    }
    // Final encoder layer_norm — HF applies it to the encoder OUTPUT, and
    // the decoder's cross-attention consumes exactly that normalized output.
    {
        for (int32_t i = 0; i < S; ++i) {
            lnw[i] = half_to_float(enc_ln_w_.f16()[i]);
            lnb[i] = half_to_float(enc_ln_b_.f16()[i]);
        }
        for (int64_t t = 0; t < T2; ++t) {
            layer_norm_f32(x.f32() + t * S, lnw.data(), lnb.data(),
                           xn.data(), S);
            std::memcpy(x.f32() + t * S, xn.data(), S * sizeof(float));
        }
    }
    out = std::move(x);
    return true;
}
// ===========================================================================
// Decoder — greedy autoregressive transcription (real ASR).
//
// Layout mirrors HF whisper-tiny exactly:
//   pre-LN blocks: x += self_attn(LN1(x));  x += cross_attn(LN2(x), enc);
//                  x += mlp(LN3(x));  logits = LN_f(x) @ tok_embd^T (tied).
//   Positions: learned table, offset 0 for the text prompt (sinemb variant).
//   Attention: causal MHA over the token context (incremental K/V caches,
//   one [pos][S] slab per layer), 6 heads x 64 dims.
//   Prompt: HF's SOT <|en|> <|transcribe|> (timestamps enabled — the HF
//   whisper default; kPromptNoTimestamps selects the classic 4-token prompt)
//   plus HF's generation policy: suppress_tokens, begin_suppress_tokens,
//   and the WhisperTimeStampLogitsProcessor pairing/monotonicity rules.
//   Greedy until <|endoftext|> or the 224-token whisper.cpp chunk cap.
// ===========================================================================
namespace {

constexpr int32_t kDecHeads = 6;   // whisper-tiny: 384 / 6 = 64-dim heads
constexpr int32_t kSotId = 50258, kEotId = 50257, kTranscribeId = 50359,
                  kEnId = 50259, kNoTsId = 50363, kTsBeginId = 50364;

// HF generation_config.suppress_tokens for the whisper multilingual models:
// non-speech / special tokens that must never be produced by generation
// (byte-fallback garbage, latin links, SOT/task/lang specials; EOT itself is
// handled separately via begin_suppress_tokens). Ids >= 50364 (timestamps)
// are governed by the timestamp rules instead.
constexpr int32_t kSuppressTokens[] = {
    1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63,
    90, 91, 92, 93, 359, 503, 522, 542, 873, 893, 902, 918, 922, 931, 1350,
    1853, 1982, 2460, 2627, 3246, 3253, 3268, 3536, 3846, 3961, 4183, 4667,
    6585, 6647, 7273, 9061, 9383, 10428, 10929, 11938, 12033, 12331, 12562,
    13793, 14157, 14635, 15265, 15618, 16553, 16604, 18362, 18956, 20075,
    21675, 22520, 26130, 26161, 26435, 28279, 29464, 31650, 32302, 32470,
    36865, 42863, 47425, 49870, 50254, 50258, 50358, 50359, 50360, 50361,
    50362};

// Multi-head attention for one query row against K/V caches [n_kv][S].
// out = softmax(q K^T / sqrt(hd)) V  (per head, concatenated).
void mha_row(const float* q, const float* kc, const float* vc,
             int64_t n_kv, float* out, int32_t S, int32_t heads) {
    const int32_t hd = S / heads;
    std::vector<float> att(static_cast<size_t>(n_kv));
    for (int32_t h = 0; h < heads; ++h) {
        float maxv = -1e30f;
        for (int64_t t = 0; t < n_kv; ++t) {
            const float* k = kc + static_cast<size_t>(t) * S +
                             static_cast<size_t>(h) * hd;
            float dot = 0.0f;
            for (int32_t i = 0; i < hd; ++i)
                dot += q[static_cast<size_t>(h) * hd + i] * k[i];
            dot /= std::sqrt(static_cast<float>(hd));
            att[static_cast<size_t>(t)] = dot;
            if (dot > maxv) maxv = dot;
        }
        float den = 0.0f;
        for (int64_t t = 0; t < n_kv; ++t) {
            att[static_cast<size_t>(t)] =
                std::exp(att[static_cast<size_t>(t)] - maxv);
            den += att[static_cast<size_t>(t)];
        }
        float* o = out + static_cast<size_t>(h) * hd;
        for (int32_t i = 0; i < hd; ++i) o[i] = 0.0f;
        for (int64_t t = 0; t < n_kv; ++t) {
            const float a = att[static_cast<size_t>(t)] / den;
            const float* v = vc + static_cast<size_t>(t) * S +
                             static_cast<size_t>(h) * hd;
            for (int32_t i = 0; i < hd; ++i) o[i] += a * v[i];
        }
    }
}

} // namespace

bool WhisperTiny::decode_greedy(const Tensor& enc,
                                std::vector<int32_t>& out_ids,
                                DecPrompt prompt_mode) const {
    const int32_t S = hp_.n_audio_state;
    const int64_t T = enc.dim(0);            // encoded audio frames
    const int32_t V = hp_.n_vocab;
    const int32_t L = static_cast<int32_t>(dec_blocks_.size());
    const float* enc_f = enc.f32();

    // Forced prompt: <|startoftranscript|> <|transcribe|> <|en|> — the same
    // init HF generate() builds for task=transcribe, language=en with
    // timestamps ENABLED (HF's whisper default); <|notimestamps|> is the
    // kPromptNoTimestamps variant. HF's suppress_tokens + timestamp logits
    // rules are applied each step below — without them greedy loops on
    // non-speech tokens.
    std::vector<int32_t> prompt_ids;
    switch (prompt_mode) {
        case kPromptNoTimestamps:
            prompt_ids = {kSotId, kEnId, kTranscribeId, kNoTsId};
            break;
        case kPromptTimestamps:
        default:
            // HF _retrieve_init_tokens order: SOT, language, task (the
            // <|notimestamps|> token is simply absent when timestamps are on).
            prompt_ids = {kSotId, kEnId, kTranscribeId};
            break;
    }
    const int32_t P = static_cast<int32_t>(prompt_ids.size());

    // Context cap: prompt (<=4) + max_new (224) within the 448-row pos table.
    constexpr int32_t kMaxNew = 224;
    const int64_t max_ctx = P + kMaxNew;

    // x: token+pos embeddings for every context row; sk/sv: per-layer
    // incremental self-attention K/V caches [L][max_ctx][S].
    std::vector<float> x(static_cast<size_t>(max_ctx) * S, 0.0f);
    std::vector<float> sk(static_cast<size_t>(L) * max_ctx * S, 0.0f),
                       sv(static_cast<size_t>(L) * max_ctx * S, 0.0f);
    // Cross-attention K/V caches from the encoder output [L][T][S],
    // computed once (the encoder output is fixed during decoding).
    std::vector<float> ck(static_cast<size_t>(L) * T * S),
                       cv(static_cast<size_t>(L) * T * S);
    std::vector<float> xn(S), q(S), att_out(S), xo(S), ff(4 * S),
                       ln1w(S), ln1b(S), ln2w(S), ln2b(S), ln3w(S), ln3b(S),
                       lnfw(S), lnfb(S), logits(static_cast<size_t>(V));
    auto f16v = [&](const Tensor& t, std::vector<float>& buf) {
        f16_to_f32(t, buf);
        return buf.data();
    };

    // Precompute cross K/V for every layer from the raw encoder frames.
    for (int32_t l = 0; l < L; ++l) {
        const DecBlock& d = dec_blocks_[static_cast<size_t>(l)];
        for (int64_t t = 0; t < T; ++t) {
            const float* e = enc_f + static_cast<size_t>(t) * S;
            float* kd = ck.data() + (static_cast<size_t>(l) * T +
                                     static_cast<size_t>(t)) * S;
            float* vd = cv.data() + (static_cast<size_t>(l) * T +
                                     static_cast<size_t>(t)) * S;
            dense_f16(d.cross_k_w_, Tensor(), e, kd, S, S);   // k: no bias in whisper
            dense_f16(d.cross_v_w_, d.cross_v_b_, e, vd, S, S);
        }
    }

    for (int32_t i = 0; i < P; ++i) {
        const uint16_t* te = dec_tok_embd_.f16() +
                             static_cast<size_t>(prompt_ids[static_cast<size_t>(i)]) * S;
        const uint16_t* pe = dec_pos_embed_.f16() +
                             static_cast<size_t>(i) * S;
        for (int32_t j = 0; j < S; ++j)
            x[static_cast<size_t>(i) * S + j] =
                half_to_float(te[j]) + half_to_float(pe[j]);
    }

    out_ids.clear();
    out_ids.reserve(kMaxNew);

    // One text row through ALL decoder layers: cache the row's self-attn
    // K/V per layer, then self-attn -> cross-attn -> MLP on the row in
    // place. Used by the prompt prefill AND every generation step so both
    // paths share one exact numeric sequence.
    auto process_row = [&](float* xrow, const int64_t row) {
        for (int32_t l = 0; l < L; ++l) {
            const DecBlock& d = dec_blocks_[static_cast<size_t>(l)];
            float* skl = sk.data() + static_cast<size_t>(l) * max_ctx * S;
            float* svl = sv.data() + static_cast<size_t>(l) * max_ctx * S;

            // cache this row's K/V from LN1(x_row)
            layer_norm_f32(xrow, f16v(d.ln1_w_, ln1w),
                           f16v(d.ln1_b_, ln1b), xn.data(), S);
            dense_f16(d.self_k_w_, Tensor(), xn.data(),
                      skl + static_cast<size_t>(row) * S, S, S);
            dense_f16(d.self_v_w_, d.self_v_b_, xn.data(),
                      svl + static_cast<size_t>(row) * S, S, S);

            // self-attention over the causal prefix x[0..row]
            dense_f16(d.self_q_w_, d.self_q_b_, xn.data(), q.data(), S, S);
            mha_row(q.data(), skl, svl, row + 1, att_out.data(), S, kDecHeads);
            dense_f16(d.self_o_w_, d.self_o_b_, att_out.data(), xo.data(), S, S);
            for (int32_t i = 0; i < S; ++i) xrow[i] += xo[i];

            // cross-attention to the encoder (cached K/V for this layer)
            layer_norm_f32(xrow, f16v(d.ln2_w_, ln2w),
                           f16v(d.ln2_b_, ln2b), xn.data(), S);
            dense_f16(d.cross_q_w_, d.cross_q_b_, xn.data(), q.data(), S, S);
            mha_row(q.data(),
                    ck.data() + static_cast<size_t>(l) * T * S,
                    cv.data() + static_cast<size_t>(l) * T * S,
                    T, att_out.data(), S, kDecHeads);
            dense_f16(d.cross_o_w_, d.cross_o_b_, att_out.data(),
                      xo.data(), S, S);
            for (int32_t i = 0; i < S; ++i) xrow[i] += xo[i];

            // MLP
            layer_norm_f32(xrow, f16v(d.ln3_w_, ln3w),
                           f16v(d.ln3_b_, ln3b), xn.data(), S);
            dense_f16(d.fc1_w_, d.fc1_b_, xn.data(), ff.data(), 4 * S, S);
            for (int32_t i = 0; i < 4 * S; ++i) ff[i] = gelu_f(ff[i]);
            dense_f16(d.fc2_w_, d.fc2_b_, ff.data(), xo.data(), S, 4 * S);
            for (int32_t i = 0; i < S; ++i) xrow[i] += xo[i];
        }
    };

    // Prefill: prompt rows 0..2 must pass through ALL layers IN SEQUENCE so
    // each layer's K/V cache holds that row's true layer-(l-1) output.
    // (The first attempt cached the raw token embedding at every layer —
    // layers > 0 then attended over garbage K/V and the decoder never
    // emitted <|endoftext|>: 224 junk steps per transcribe call.)
    for (int32_t i = 0; i + 1 < P; ++i)
        process_row(x.data() + static_cast<size_t>(i) * S, i);

    for (int32_t step = 0; step < kMaxNew; ++step) {
        const int64_t n_ctx = P + step;           // rows currently in x
        // The newest row's embedding was written when its token was chosen
        // below; at step 0 it is the last prompt row (already in x).
        float* xlast = x.data() + static_cast<size_t>(n_ctx - 1) * S;

        process_row(xlast, n_ctx - 1);           // through ALL decoder layers

        // lm_head (tied embedding) + logits on the last row
        layer_norm_f32(xlast, f16v(dec_ln_w_, lnfw),
                       f16v(dec_ln_b_, lnfb), xn.data(), S);
        const uint16_t* W = dec_tok_embd_.f16();       // [V, S]
        logits.assign(static_cast<size_t>(V), 0.0f);
        for (int32_t v = 0; v < V; ++v) {
            const uint16_t* row = W + static_cast<size_t>(v) * S;
            float dot = 0.0f;
            for (int32_t i = 0; i < S; ++i)
                dot += half_to_float(row[i]) * xn[i];
            logits[static_cast<size_t>(v)] = dot;
        }

        // ---- HF whisper generation policy (the exact processors that run
        // inside model.generate for return_timestamps=False greedy): without
        // it the decoder loops on non-speech tokens and misses <|eot|>.
        for (const int32_t s : kSuppressTokens)
            if (s >= 0 && s < V) logits[static_cast<size_t>(s)] = -1e30f;
        if (step == 0) {                     // begin_suppress_tokens [220, eot]
            logits[static_cast<size_t>(220)] = -1e30f;
            logits[static_cast<size_t>(kEotId)] = -1e30f;
        }
        if (prompt_mode == kPromptNoTimestamps) {
            // <|notimestamps|> in the prompt: openai/whisper masks every
            // timestamp token for the whole segment.
            logits[static_cast<size_t>(kNoTsId)] = -1e30f;
            for (int32_t v = kTsBeginId; v < V; ++v)
                logits[static_cast<size_t>(v)] = -1e30f;
        } else {
            // HF WhisperTimeStampLogitsProcessor (timestamps enabled).
            logits[static_cast<size_t>(kNoTsId)] = -1e30f;
            const bool last_was_ts =
                !out_ids.empty() && out_ids.back() >= kTsBeginId;
            const bool penult_was_ts =
                out_ids.size() < 2 ||
                out_ids[out_ids.size() - 2] >= kTsBeginId;
            if (last_was_ts && penult_was_ts)
                for (int32_t v = kTsBeginId; v < V; ++v)
                    logits[static_cast<size_t>(v)] = -1e30f;
            else if (last_was_ts)
                for (int32_t v = 0; v < kEotId; ++v)   // text after a pair
                    logits[static_cast<size_t>(v)] = -1e30f;
            if (last_was_ts) {                  // timestamps must not decrease
                int64_t ts_last = penult_was_ts
                    ? static_cast<int64_t>(out_ids.back()) + 1
                    : static_cast<int64_t>(out_ids.back());
                if (ts_last < kTsBeginId) ts_last = kTsBeginId;
                for (int64_t v = kTsBeginId;
                     v < std::min<int64_t>(V, ts_last + 1); ++v)
                    logits[static_cast<size_t>(v)] = -1e30f;
            }
            if (step == 0) {                    // max_initial_timestamp 0..1.0s
                for (int32_t v = 0; v < kTsBeginId; ++v)
                    logits[static_cast<size_t>(v)] = -1e30f;
                for (int32_t v = kTsBeginId + 11; v < V; ++v)
                    logits[static_cast<size_t>(v)] = -1e30f;
            }
            double lse = 0.0, mtxt = -1e30;
            for (int32_t v = 0; v < kTsBeginId; ++v)
                mtxt = std::max(mtxt, static_cast<double>(
                                          logits[static_cast<size_t>(v)]));
            for (int32_t v = kTsBeginId; v < V; ++v) {
                const double e = std::exp(static_cast<double>(
                    logits[static_cast<size_t>(v)]) - 20.0);
                lse += e;
            }
            if (std::log(lse) + 20.0 > mtxt)    // timestamps win over text
                for (int32_t v = 0; v < kTsBeginId; ++v)
                    logits[static_cast<size_t>(v)] = -1e30f;
        }
        int32_t best = 0;
        float best_logit = -1e30f;
        for (int32_t v = 0; v < V; ++v)
            if (logits[static_cast<size_t>(v)] > best_logit) {
                best_logit = logits[static_cast<size_t>(v)];
                best = v;
            }
        if (std::getenv("OMNISEED_DBG")) {
            if (std::getenv("OMNISEED_DBG_NAN")) {
                for (int32_t i = 0; i < S; ++i) {
                    if (!std::isfinite(xlast[i])) {
                        std::fprintf(stderr, "[nan] step %d: xlast[%d]=%g "
                                     "(row %lld)\n", step, i,
                                     static_cast<double>(xlast[i]),
                                     static_cast<long long>(n_ctx - 1));
                        break;
                    }
                }
            }
            if (std::getenv("OMNISEED_DBG")) {
                // exact logits for oracle comparison at step 0
                const int32_t probe[] = {28503, 3536, 50257, 522, 257, 363, 11, 5, 400};
                std::fprintf(stderr, "[logits] step %d:", step);
                for (int32_t pid : probe) {
                    const uint16_t* row = W + static_cast<size_t>(pid) * S;
                    float dot = 0.0f;
                    for (int32_t i = 0; i < S; ++i)
                        dot += half_to_float(row[i]) * xn[i];
                    std::fprintf(stderr, " %d=%.4f", pid,
                                 static_cast<double>(dot));
                }
                std::fprintf(stderr, "\n");
            }
            // top-5 debug for the first steps
            struct Top { float v; int32_t id; } top[5] = {};
            for (int32_t v = 0; v < V; ++v) {
                const uint16_t* row = W + static_cast<size_t>(v) * S;
                float dot = 0.0f;
                for (int32_t i = 0; i < S; ++i)
                    dot += half_to_float(row[i]) * xn[i];
                if (dot > top[4].v) {
                    top[4] = {dot, v};
                    for (int32_t m = 3; m >= 0; --m)
                        if (top[m].v < top[m + 1].v)
                            std::swap(top[m], top[m + 1]);
                }
            }
            const int32_t* offs = reinterpret_cast<const int32_t*>(
                vocab_offsets_.data());
            const uint16_t* els = reinterpret_cast<const uint16_t*>(
                vocab_bytes_.data());
            std::fprintf(stderr, "[dec] step %d:", step);
            for (auto& t5 : top) {
                std::string pc;
                for (int32_t o = offs[t5.id]; o < offs[t5.id + 1]; ++o)
                    pc.push_back(static_cast<char>(half_to_float(els[o])));
                std::fprintf(stderr, " %d(%.2f)%s", t5.id,
                             static_cast<double>(t5.v),
                             pc.size() > 24 ? "<spec>" : pc.c_str());
            }
            std::fprintf(stderr, "\n");
            std::fflush(stderr);
        }
        if (best == kEotId) break;
        out_ids.push_back(best);
        if (n_ctx >= max_ctx) break;                  // pos-table cap
        // embed the chosen token at position n_ctx for the next step
        const uint16_t* te = dec_tok_embd_.f16() +
                             static_cast<size_t>(best) * S;
        const uint16_t* pe = dec_pos_embed_.f16() +
                             static_cast<size_t>(n_ctx) * S;
        float* xnext = x.data() + static_cast<size_t>(n_ctx) * S;
        for (int32_t j = 0; j < S; ++j)
            xnext[j] = half_to_float(te[j]) + half_to_float(pe[j]);
    }
    return true;
}

} // namespace omniseed
