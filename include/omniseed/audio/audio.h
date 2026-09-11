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

#include "omniseed/core/gguf_loader.h"
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

    // Same, from an in-memory RIFF/WAVE blob (HTTP /asr request body).
    static bool load_wav_bytes(const std::string& data, PcmAudio& out);
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
    // (deterministic), so callers can be tested end-to-end. With the real
    // encoder+decoder sidecar this runs the full autoregressive loop:
    // <|startoftranscript|><|en|><|transcribe|> -> tokens -> <|endoftext|>.
    bool transcribe(const PcmAudio& audio, std::string& out_text) const;

    // Real encoder pass (requires converted whisper-tiny sidecar weights):
    // mel -> conv stem -> transformer blocks -> [T/2, 384] encoded frames.
    bool encode(const PcmAudio& audio, Tensor& out) const;

    // True when the DECODER tensors (autoregressive transcription) loaded.
    bool decoder_loaded() const { return decoder_loaded_; }

private:
    // Owns the mmap the fp16 encoder views point into (must outlive them).
    GgufLoader store_;
    WhisperHparams hp_{};
    Tensor mel_filters_;      // fp32 [n_mels, 201]
    bool  weights_loaded_ = false;
    // encoder tensors (fp16, from whisper-tiny via tools/convert_senses.py)
    Tensor conv1_w_, conv1_b_, conv2_w_, conv2_b_;
    Tensor pos_embed_;        // [1500, 384]
    Tensor enc_ln_w_, enc_ln_b_;
    struct EncBlock {
        Tensor q_w_, q_b_, k_w_, k_b_, v_w_, v_b_, o_w_, o_b_;
        Tensor ln1_w_, ln1_b_, fc1_w_, fc1_b_, fc2_w_, fc2_b_, ln2_w_, ln2_b_;
    };
    std::vector<EncBlock> enc_blocks_;

    // ---- decoder (autoregressive transcription; whisper-tiny pre-LN) ----
    struct DecBlock {
        Tensor self_q_w_, self_q_b_, self_k_w_, self_v_w_, self_v_b_, self_o_w_, self_o_b_;
        Tensor cross_q_w_, cross_q_b_, cross_k_w_, cross_v_w_, cross_v_b_, cross_o_w_, cross_o_b_;
        Tensor ln1_w_, ln1_b_, ln2_w_, ln2_b_, ln3_w_, ln3_b_;
        Tensor fc1_w_, fc1_b_, fc2_w_, fc2_b_;
    };
    Tensor dec_tok_embd_;     // [V, 384] — also the lm_head (tied)
    Tensor dec_pos_embed_;    // [448, 384]
    Tensor dec_ln_w_, dec_ln_b_;
    std::vector<DecBlock> dec_blocks_;
    // vocab pieces: byte blobs via offsets+bytes tensors (no string arrays)
    Tensor vocab_offsets_;    // declared F32, bytes are int32 offsets [V+1]
    Tensor vocab_bytes_;      // declared F16, bytes are raw piece bytes
    bool decoder_loaded_ = false;

    // Precomputed special-token ids (whisper multilingual layout).
    int32_t sot_id_ = 50258, eot_id_ = 50257, transcribe_id_ = 50359,
            en_id_ = 50262, no_ts_id_ = 50363;

    // Greedy autoregressive decode over the encoder frames (real ASR).
    // Prompt variant: timestamps enabled (HF whisper default, loop-resistant)
    // or the classic <|notimestamps|> prompt.
    enum DecPrompt { kPromptTimestamps, kPromptNoTimestamps };
    bool decode_greedy(const Tensor& enc, std::vector<int32_t>& out_ids,
                       DecPrompt prompt_mode = kPromptTimestamps) const;

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
