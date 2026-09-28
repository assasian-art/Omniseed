// =============================================================================
//  OmniSeed — multimodal.h
//
//  THE JOINT. §28 built one backbone and many heads; §30-31 gave it a soul and
//  a memory; §32 made the heads' confidence real. But the vision and audio
//  encoders (src/vision, src/audio — both real, both tested in isolation) and
//  the TokenBus (src/runtime/token_bus.cpp — real, and until now UNTESTED)
//  were never joined to the head stack. Nothing turned an image or a recording
//  into the h[E] the heads actually read.
//
//  This file is that join, and nothing else. It is deliberately small: it does
//  not re-implement an encoder, a tokenizer or a bus. It converts the ONE thing
//  that does not fit — a continuous modality embedding — into the discrete
//  token space the core already reasons in, then hands the fused stream to the
//  existing TokenBus.
//
//  THE IMPEDANCE MISMATCH THIS SOLVES
//  ----------------------------------
//    VisionEncoder::encode()  -> Tensor [M, n_embd]   (CONTINUOUS floats)
//    FocalCodec::encode()     -> vector<int32_t>      (already token ids)
//    TokenBus::fuse()         -> wants vector<int32_t> for EVERY modality
//    RwkvModel::forward()     -> takes int32_t token
//
//  So audio already fits and vision does not. The bridge quantizes a vision
//  embedding to the NEAREST VOCABULARY TOKEN using the model's own embedding
//  matrix as the codebook. That choice is not arbitrary:
//    * it needs no training, so it cannot silently mis-calibrate;
//    * it uses weights that are already resident, so it costs no extra RAM
//      (the R2 <400 MB budget is untouched);
//    * it lands the modality in the same discrete space the core was trained
//      on, which is the only space forward() accepts.
//
//  ⚠️ WHAT THIS IS NOT. Nearest-embedding quantization is a LOSSY, UNTRAINED
//  projection. It is not a learned modality adapter, and it makes no claim that
//  the quantized tokens are semantically the "right" tokens — only that they
//  are the closest vocabulary items to what the encoder produced. The label
//  sets it feeds (vision.scene, vision.anomaly, audio.wake, ...) still have NO
//  trained weights, so the heads will report the base rate until they are
//  fitted exactly like the §32 language/trading heads. Documented, not hidden:
//  see docs/MULTIMODAL.md.
// =============================================================================
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "omniseed/core/tensor.h"
#include "omniseed/core/tokenizer.h"
#include "omniseed/runtime/token_bus.h"

namespace omniseed {

class MultimodalBridge {
public:
    struct Config {
        // The token embedding matrix has n_vocab rows (65536 for RWKV-7 0.1B).
        // A full nearest-neighbour scan per vision token is O(n_vocab * E);
        // stride k scans every k-th row. 1 = exact, larger = faster and
        // coarser. This is a real accuracy/speed dial, not a no-op.
        int32_t codebook_stride = 8;

        // The core's per-turn token budget, forwarded to the TokenBus.
        int64_t max_tokens = 4096;

        // Emit the text segment first, then vision, then audio. Order is
        // preserved by the bus and is part of the contract (a caller that
        // reorders segments gets a different prompt).
        bool include_text = true;
    };

    // The result of one fusion. `ok == false` always carries a reason; the
    // caller must not treat a failed fusion as an empty one.
    struct Fusion {
        std::vector<int32_t> ids;          // the fused stream, BOS-first
        TokenBus::Stats      stats;
        int32_t  vision_tokens_emitted = 0;
        int32_t  audio_tokens_emitted  = 0;
        bool     ok     = false;
        std::string error;
    };

    MultimodalBridge() = default;

    // n_embd must match the model's E; the TokenBus is built around `tok`,
    // which must outlive this bridge.
    bool init(int32_t n_embd, const Tokenizer& tok);
    bool ready() const { return ready_; }
    const std::string& error() const { return error_; }
    int32_t n_embd() const { return n_embd_; }

    void set_config(const Config& c);
    const Config& config() const { return cfg_; }

    // ---- the codebook -------------------------------------------------------
    // Non-owning view of the model's token embedding matrix [n_vocab, E].
    // Call once after the model is loaded. Passing a tensor whose width does
    // not match n_embd is REFUSED rather than truncated: a silent width
    // mismatch would quantize against the wrong half of every row.
    bool set_codebook(const Tensor& embeddings);
    bool codebook_ready() const { return codebook_ != nullptr; }
    int32_t codebook_rows() const { return codebook_rows_; }
    int32_t codebook_stride() const { return stride_; }

    // Nearest vocabulary token id for one n_embd vector, or -1 when the
    // codebook is not set / the vector is the wrong width / it is non-finite.
    int32_t nearest_token(const float* vec) const;

    // Quantizes a row-major [M, E] block into M token ids. Returns false and
    // clears out_ids on a width mismatch.
    bool quantize(const std::vector<float>& rows, std::vector<int32_t>& out_ids) const;

    // ---- the fusion ---------------------------------------------------------
    // vision_embeddings is row-major [M, E], or empty for "no image".
    // audio_codes are ids already produced by FocalCodec/WhisperTiny, or empty.
    // Returns ok=false with a reason when the codebook is missing but vision
    // was supplied, or when the token budget is exceeded.
    Fusion fuse(const std::string& text,
                const std::vector<float>& vision_embeddings,
                const std::vector<int32_t>& audio_codes) const;

    // A one-line, human-readable description of what was fused. Built from the
    // stats, so it can never claim a modality that contributed no tokens.
    std::string summary(const Fusion& f) const;

private:
    bool ready_ = false;
    int32_t n_embd_ = 0;
    std::string error_;

    const Tokenizer* tok_ = nullptr;
    std::unique_ptr<TokenBus> bus_;   // TokenBus holds a Tokenizer& and so
                                      // cannot be default-constructed or
                                      // assigned; it must be built in place.

    Config cfg_;

    const Tensor* codebook_ = nullptr;   // non-owning
    int32_t codebook_rows_ = 0;
    int32_t codebook_cols_ = 0;
    int32_t stride_ = 1;
};

} // namespace omniseed
