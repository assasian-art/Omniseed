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
    // tanh approximation used by whisper.cpp/openai whisper
    const float c = 0.044715f;
    return 0.5f * x * (1.0f + std::tanh(0.7978845608028654f * (x + c * x * x * x)));
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
        // vocab_bytes is stored in an F16 container: piece byte i lives in
        // the LOW byte of element 2i (little-endian; see convert_senses.py).
        const uint16_t* elems =
            reinterpret_cast<const uint16_t*>(vocab_bytes_.data());
        for (const int32_t id : ids) {
            if (id < 0 || id >= hp_.n_vocab) continue;
            for (int32_t o = offs[id]; o < offs[id + 1]; ++o)
                out_text.push_back(static_cast<char>(elems[o] & 0xFF));
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
    int64_t frames = mel.dim(1);
    if (frames > hp_.n_audio_ctx) frames = hp_.n_audio_ctx;
    const int32_t S = hp_.n_audio_state;

    // conv1: kernel 3 over the MEL axis, stride 2 (whisper stem):
    // in [80, T] -> out [384, T]. We store mel as [n_mels, T] row-major.
    Tensor h1("h1", {frames, S}, DType::F32);
    {
        const uint16_t* W = conv1_w_.f16();   // [384, 80, 3] (o, m, k)
        const float* M = mel.f32();
        const int64_t T_in = mel.dim(1);
        for (int64_t f = 0; f < frames; ++f) {
            float* dst = h1.f32() + f * S;
            for (int32_t o = 0; o < S; ++o) {
                float acc = half_to_float(conv1_b_.f16()[o]);
                for (int32_t k = 0; k < 3; ++k) {
                    const int64_t t_in = f * 2 + k;
                    if (t_in >= T_in) continue;
                    const uint16_t* wbase = W + static_cast<int64_t>(o) * n_mels * 3 + k;
                    for (int64_t m = 0; m < n_mels; ++m)
                        acc += half_to_float(wbase[m * 3]) * M[m * T_in + t_in];
                }
                dst[o] = gelu_f(acc);
            }
        }
    }
    // conv2 over time, kernel 3 stride 2 -> [frames/2, 384]
    const int64_t T2 = frames / 2;
    Tensor h2("h2", {T2, S}, DType::F32);
    {
        const uint16_t* W = conv2_w_.f16();   // [384, 384, 3]
        for (int64_t f = 0; f < T2; ++f) {
            float* dst = h2.f32() + f * S;
            for (int32_t o = 0; o < S; ++o) {
                float acc = half_to_float(conv2_b_.f16()[o]);
                for (int32_t k = 0; k < 3; ++k) {
                    const int64_t t_in = f * 2 + k;
                    if (t_in >= frames) continue;
                    const uint16_t* wrow = W + static_cast<int64_t>(o) * S * 3 + k;
                    const float* src = h1.f32() + t_in * S;
                    for (int32_t i = 0; i < S; ++i)
                        acc += half_to_float(wrow[i * 3]) * src[i];
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
        }
        for (int64_t t = 0; t < T2; ++t) {
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
        for (int64_t t = 0; t < T2; ++t)
            layer_norm_f32(x.f32() + t * S, lnw.data(), lnb.data(),
                           xn.data(), S);
        for (int64_t t = 0; t < T2; ++t)
            for (int32_t i = 0; i < S; ++i)
                x.f32()[t * S + i] = xn[i];
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
//   Greedy until <|endoftext|> or the 224-token whisper.cpp chunk cap.
// ===========================================================================
namespace {

constexpr int32_t kDecHeads = 6;   // whisper-tiny: 384 / 6 = 64-dim heads
constexpr int32_t kSotId = 50258, kEotId = 50257, kTranscribeId = 50359,
                  kEnId = 50259, kNoTsId = 50363;

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
                                std::vector<int32_t>& out_ids) const {
    const int32_t S = hp_.n_audio_state;
    const int64_t T = enc.dim(0);            // encoded audio frames
    const int32_t V = hp_.n_vocab;
    const int32_t L = static_cast<int32_t>(dec_blocks_.size());
    const float* enc_f = enc.f32();

    // Context cap: prompt (4) + max_new (224) within the 448-row pos table.
    constexpr int32_t kMaxNew = 224;
    const int64_t max_ctx = 4 + kMaxNew;

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
                       lnfw(S), lnfb(S);
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
            dense_f16(d.cross_k_w_, Tensor(), e, kd, S, S);
            dense_f16(d.cross_v_w_, Tensor(), e, vd, S, S);
        }
    }

    // Forced prompt: <|startoftranscript|> <|en|> <|transcribe|>
    // <|notimestamps|>; greedy until <|endoftext|>.
    const int32_t prompt[4] = {kSotId, kEnId, kTranscribeId, kNoTsId};
    for (int32_t i = 0; i < 4; ++i) {
        const uint16_t* te = dec_tok_embd_.f16() +
                             static_cast<size_t>(prompt[i]) * S;
        const uint16_t* pe = dec_pos_embed_.f16() +
                             static_cast<size_t>(i) * S;
        for (int32_t j = 0; j < S; ++j)
            x[static_cast<size_t>(i) * S + j] =
                half_to_float(te[j]) + half_to_float(pe[j]);
    }

    out_ids.clear();
    out_ids.reserve(kMaxNew);
    for (int32_t step = 0; step < kMaxNew; ++step) {
        const int64_t n_ctx = 4 + step;           // rows currently in x
        // The newest row's embedding was written when its token was chosen
        // below; for step 0 the last prompt row (kNoTsId) is already in x[3].
        float* xlast = x.data() + static_cast<size_t>(n_ctx - 1) * S;

        for (int32_t l = 0; l < L; ++l) {
            const DecBlock& d = dec_blocks_[static_cast<size_t>(l)];
            float* skl = sk.data() + static_cast<size_t>(l) * max_ctx * S;
            float* svl = sv.data() + static_cast<size_t>(l) * max_ctx * S;

            // cache the NEW row's K/V from LN1(x_last)
            layer_norm_f32(xlast, f16v(d.ln1_w_, ln1w),
                           f16v(d.ln1_b_, ln1b), xn.data(), S);
            dense_f16(d.self_k_w_, Tensor(), xn.data(),
                      skl + static_cast<size_t>(n_ctx - 1) * S, S, S);
            dense_f16(d.self_v_w_, Tensor(), xn.data(),
                      svl + static_cast<size_t>(n_ctx - 1) * S, S, S);

            // self-attention for the new row over x[0..n_ctx-1]
            dense_f16(d.self_q_w_, d.self_q_b_, xn.data(), q.data(), S, S);
            mha_row(q.data(), skl, svl, n_ctx, att_out.data(), S, kDecHeads);
            dense_f16(d.self_o_w_, d.self_o_b_, att_out.data(), xo.data(), S, S);
            for (int32_t i = 0; i < S; ++i) xlast[i] += xo[i];

            // cross-attention to the encoder (cached K/V for this layer)
            layer_norm_f32(xlast, f16v(d.ln2_w_, ln2w),
                           f16v(d.ln2_b_, ln2b), xn.data(), S);
            dense_f16(d.cross_q_w_, d.cross_q_b_, xn.data(), q.data(), S, S);
            mha_row(q.data(),
                    ck.data() + static_cast<size_t>(l) * T * S,
                    cv.data() + static_cast<size_t>(l) * T * S,
                    T, att_out.data(), S, kDecHeads);
            dense_f16(d.cross_o_w_, d.cross_o_b_, att_out.data(),
                      xo.data(), S, S);
            for (int32_t i = 0; i < S; ++i) xlast[i] += xo[i];

            // MLP
            layer_norm_f32(xlast, f16v(d.ln3_w_, ln3w),
                           f16v(d.ln3_b_, ln3b), xn.data(), S);
            dense_f16(d.fc1_w_, d.fc1_b_, xn.data(), ff.data(), 4 * S, S);
            for (int32_t i = 0; i < 4 * S; ++i) ff[i] = gelu_f(ff[i]);
            dense_f16(d.fc2_w_, d.fc2_b_, ff.data(), xo.data(), S, 4 * S);
            for (int32_t i = 0; i < S; ++i) xlast[i] += xo[i];
        }

        // lm_head (tied embedding) + argmax on the last row
        layer_norm_f32(xlast, f16v(dec_ln_w_, lnfw),
                       f16v(dec_ln_b_, lnfb), xn.data(), S);
        const uint16_t* W = dec_tok_embd_.f16();       // [V, S]
        int32_t best = 0;
        float best_logit = -1e30f;
        for (int32_t v = 0; v < V; ++v) {
            const uint16_t* row = W + static_cast<size_t>(v) * S;
            float dot = 0.0f;
            for (int32_t i = 0; i < S; ++i)
                dot += half_to_float(row[i]) * xn[i];
            if (dot > best_logit) { best_logit = dot; best = v; }
        }
        if (std::getenv("OMNISEED_DBG")) {
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
                    pc.push_back(static_cast<char>(els[o] & 0xFF));
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
