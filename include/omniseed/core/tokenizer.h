// =============================================================================
//  OmniSeed — tokenizer.h
//  SentencePiece-compatible tokenizer for the shared ~8k-16k vocab.
//
//  Blueprint notes:
//    * ONE tokenizer shared by text, vision and audio token streams.
//    * Vocab loaded either from GGUF metadata (tokenizer.ggml.tokens etc.)
//      or from a compact binary .ovocab file produced by the converter.
//    * Matching: greedy longest-match over the piece table (unigram-style),
//      with <0xXX> byte-fallback so any UTF-8 round-trips losslessly.
//
//  Not a full SentencePiece unigram lattice — a greedy matcher keeps the
//  runtime tiny and deterministic; converter emits compatible pieces.
// =============================================================================
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace omniseed {

class Tokenizer {
public:
    // Special token ids (Hermes/agent-friendly layout).
    static constexpr int32_t kUnkId      = 0;
    static constexpr int32_t kBosId      = 1;
    static constexpr int32_t kEosId      = 2;
    static constexpr int32_t kUserStartId = 3;   // <|user|>
    static constexpr int32_t kUserEndId   = 4;   // </|user|>
    static constexpr int32_t kAssistantStartId = 5; // <|assistant|>
    static constexpr int32_t kAssistantEndId   = 6; // </|assistant|>
    static constexpr int32_t kVisionStartId    = 7; // <|vision|>
    static constexpr int32_t kVisionEndId      = 8; // </|vision|>
    static constexpr int32_t kAudioStartId     = 9; // <|audio|>
    static constexpr int32_t kAudioEndId       = 10; // </|audio|>
    static constexpr int32_t kToolStartId      = 11; // <|tool_json|>
    static constexpr int32_t kThinkStartId     = 12; // <|think|>
    static constexpr int32_t kThinkEndId       = 13; // </|think|>
    static constexpr int32_t kFirstByteId      = 16; // byte fallback <0x00>..<0xFF>

    Tokenizer() = default;

    // Configurable special ids, published by the converter via GGUF metadata
    // (omniseed.{bos,user_start,user_end,assistant_start,assistant_end}_token_id).
    // -1 = "absent from this vocab" (world models have no <s> / chat controls).
    void set_special_ids(int32_t bos, int32_t user_start, int32_t user_end,
                         int32_t assistant_start, int32_t assistant_end) {
        bos_id_ = bos; user_start_id_ = user_start; user_end_id_ = user_end;
        assistant_start_id_ = assistant_start;
        assistant_end_id_ = assistant_end;
    }
    int32_t bos_id() const { return bos_id_; }

    // Loads from GGUF metadata keys:
    //   tokenizer.ggml.tokens  (string array)
    //   tokenizer.ggml.scores  (f32 array, optional)
    //   tokenizer.ggml.token_type (i32 array, optional: 1=NORMAL 2=UNKNOWN
    //                              3=CONTROL 4=USER_DEFINED 5=UNUSED 6=BYTE)
    bool load_from_gguf(const class GgufLoader& gguf);

    // Loads from a binary .ovocab file (converter output):
    //   u32 magic 'OVOC', u32 version, u32 n_tokens,
    //   then per token: u32 piece_len, bytes, i32 type
    bool load_vocab_file(const std::string& path);

    // Builds only the byte-fallback + special tokens (unit tests / bring-up).
    bool build_minimal(int32_t reserve = 512);

    bool valid() const { return !pieces_.empty(); }
    int32_t vocab_size() const { return static_cast<int32_t>(pieces_.size()); }

    // ------------------------------ encode -----------------------------------
    // UTF-8 text -> token ids. Byte-fallback keeps round-trip lossless.
    std::vector<int32_t> encode(const std::string& text, bool add_bos = true) const;

    // Convenience for chat-shaped input:
    // <|user|> text </|user|><|assistant|>
    std::vector<int32_t> encode_chat(const std::string& user_text) const;

    // Wraps arbitrary modality payload ids, e.g. vision/audio token streams.
    std::vector<int32_t> wrap_modality(int32_t start_id, int32_t end_id,
                                       const std::vector<int32_t>& ids) const;

    // ------------------------------ decode -----------------------------------
    // Token ids -> UTF-8 text. Control tokens are skipped unless include_special.
    std::string decode(const std::vector<int32_t>& ids,
                       bool include_special = false) const;

    // Single-token access.
    const std::string& piece(int32_t id) const {
        return (id >= 0 && id < static_cast<int32_t>(pieces_.size()))
                   ? pieces_[static_cast<size_t>(id)] : k_unk_piece_;
    }
    int32_t find(const std::string& piece) const;

private:
    void rebuild_index();
    void detect_special_ids();

    std::vector<std::string>          pieces_;
    std::vector<int32_t>              types_;     // same codes as GGUF token_type
    std::unordered_map<std::string, int32_t> index_;

    // Special-token ids actually present in the loaded vocab (-1 = absent).
    int32_t bos_id_            = kBosId;
    int32_t user_start_id_     = kUserStartId;
    int32_t user_end_id_       = kUserEndId;
    int32_t assistant_start_id_ = kAssistantStartId;
    int32_t assistant_end_id_  = kAssistantEndId;

    std::string k_unk_piece_ = "<unk>";
};

} // namespace omniseed
