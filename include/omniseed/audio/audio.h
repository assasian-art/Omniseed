// =============================================================================
//  OmniSeed — audio.h
//  Distilled Whisper-tiny (Int4) mel front end + encoder, and FocalCodec
//  semantic tokenization for the audio modality.
//
//  Whisper path (hardcoded GGML layout per whisper.cpp conventions):
//    hparams (n_mels=80, n_audio_ctx=1500, n_audio_state=384,
//             n_audio_head=6, n_audio_layer=4)
//    -> mel filters (80 x 201, embedded in the model file)
//    -> vocab (ggml tokens for the decoder, embedded)
//    -> tensors (audio Conv1D stems + transformer blocks)
//
//  FocalCodec path: causal distillation-style semantic tokens at
//  0.16-0.65 kbps — audio -> frame features -> discrete semantic codes,
//  mapped into the shared tokenizer space via <|audio|> wrapping.
//
//  Budget: audio path peak RAM ~25 MB (blueprint).
// =============================================================================
#pragma once

#include "omniseed/core/tensor.h"

#include <string>
#include <vector>

namespace omniseed {

// ---------------------------------------------------------------------------
// PCM audio: 16 kHz mono f32, normalized to [-1, 1]
// ---------------------------------------------------------------------------
struct PcmAudio {
    int32_t sample_rate = 16000;
    std::vector<float> samples;

    bool valid() const { return sample_rate > 0 && !samples.empty(); }

    // Loads a 16-bit mono WAV (PCM) file. Returns false on unsupported files.
    static bool load_wav(const std::string& path, PcmAudio& out);
};

// ---------------------------------------------------------------------------
// WhisperTiny — distilled, int4-quantized encoder + greedy decoder
// ---------------------------------------------------------------------------
struct WhisperHparams {
    int32_t n_mels       = 80;
    int32_t n_audio_ctx  = 1500;
    int32_t n_audio_state= 384;
    int32_t n_audio_head = 6;
    int32_t n_audio_layer= 4;
    int32_t n_vocab      = 51865;
    int32_t f16_kelvin   = 0;   // unused; reserved
};

class WhisperTiny {
public:
    // Loads "whisper.*" tensors from the OmniSeed GGUF (mel filters, stems,
    // encoder blocks, decoder, vocab). Weights may be int4-packed.
    bool load(const std::string& gguf_path);

    bool valid() const { return valid_; }
    const WhisperHparams& hparams() const { return hp_; }
    const std::string& error() const { return error_; }

    // Raw waveform -> log-mel spectrogram [n_mels, frames] (fp32).
    // Frames = 3000 for 30s windows (10ms hop, 25ms window) per whisper.cpp.
    bool compute_mel(const PcmAudio& audio, Tensor& mel) const;

    // Greedy transcription using the loaded decoder. Language hint "en" etc.
    // With unloaded weights this returns the audio energy envelope summary
    // (deterministic), so callers can be tested end-to-end.
    bool transcribe(const PcmAudio& audio, std::string& out_text) const;

private:
    WhisperHparams hp_{};
    Tensor mel_filters_;      // fp32 [n_mels, 201]
    bool  weights_loaded_ = false;
    std::vector<Tensor> enc_tensors_;
    Tensor vocab_;            // i32 token strings handled via Tokenizer
    bool valid_ = false;
    mutable std::string error_;   // set even from const methods
};

// ---------------------------------------------------------------------------
// FocalCodec — ultra-low-bitrate semantic audio tokenization
// ---------------------------------------------------------------------------
struct FocalCodecConfig {
    int32_t  frame_ms     = 80;      // 12.5 frames/s at 80ms
    int32_t  codebook_size= 1024;    // discrete semantic codes
    float    bitrate_kbps = 0.16f;   // target bitrate class
    int32_t  sample_rate  = 16000;
};

class FocalCodec {
public:
    explicit FocalCodec(const FocalCodecConfig& cfg = {}) : cfg_(cfg) {}

    // Encodes PCM into discrete semantic codes (one per frame).
    // Weights come from "focal.*" GGUF tensors; without them a deterministic
    // spectral-hash codebook is used (documented, testable fallback).
    bool encode(const PcmAudio& audio, std::vector<int32_t>& out_codes) const;

    // Frame-level mel-cepstral features (the codec's front end).
    bool frame_features(const PcmAudio& audio, Tensor& frames) const;

    const FocalCodecConfig& config() const { return cfg_; }

private:
    FocalCodecConfig cfg_;
    bool weights_loaded_ = false;
};

} // namespace omniseed
