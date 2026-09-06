// =============================================================================
//  OmniSeed — tokenizer.cpp
//  Vocab loading + greedy longest-match encoding + byte-fallback decode.
// =============================================================================
#include "omniseed/core/tokenizer.h"
#include "omniseed/core/gguf_format.h"
#include "omniseed/core/gguf_loader.h"
#include "omniseed/core/platform.h"

#include <cstdio>
#include <unordered_set>

namespace omniseed {

using gguf::GgufType;
using gguf::MetadataValue;

// ===========================================================================
// Vocabulary loading
// ===========================================================================
bool Tokenizer::load_from_gguf(const GgufLoader& gguf) {
    pieces_.clear();
    types_.clear();
    index_.clear();

    const std::string toks_key = "tokenizer.ggml.tokens";
    const std::string typ_key  = "tokenizer.ggml.token_type";
    if (!gguf.has_key(toks_key)) {
        platform::log_error("tokenizer: GGUF missing %s", toks_key.c_str());
        return false;
    }

    const MetadataValue* mtoks = gguf.metadata_value(toks_key);
    const MetadataValue* mtyp  = gguf.metadata_value(typ_key);
    if (mtoks == nullptr || mtoks->type != GgufType::ARRAY ||
        mtoks->elem_type != GgufType::STRING) {
        platform::log_error("tokenizer: tokens array missing or wrong type");
        return false;
    }

    // Array of strings layout: repeated [u64 len][bytes].
    const uint8_t* p    = mtoks->raw.data();
    size_t         cur  = 0;
    const size_t   size = mtoks->raw.size();
    while (cur + 8 <= size) {
        const uint64_t len = gguf::read_u64(p, cur);
        cur += 8;
        if (cur + len > size) break;
        pieces_.emplace_back(reinterpret_cast<const char*>(p + cur),
                             static_cast<size_t>(len));
        cur += static_cast<size_t>(len);
    }

    types_.assign(pieces_.size(), 1);       // NORMAL default
    if (mtyp != nullptr && mtyp->type == GgufType::ARRAY &&
        mtyp->elem_type == GgufType::INT32) {
        // i32 array stored raw little-endian in mtyp->raw.
        for (size_t i = 0; i + 4 <= mtyp->raw.size() && i < pieces_.size(); i += 4) {
            types_[i / 4] = gguf::read_i32(mtyp->raw.data(), i);
        }
    }

    rebuild_index();
    detect_special_ids();
    return valid();
}

bool Tokenizer::load_vocab_file(const std::string& path) {
    platform::MappedFile mf;
    if (!mf.open(path)) {
        platform::log_error("tokenizer: cannot open vocab %s: %s",
                            path.c_str(), mf.last_error().c_str());
        return false;
    }

    const uint8_t* p = mf.bytes();
    size_t cursor = 0;
    if (mf.size() < 12 || gguf::read_u32(p, cursor) != 0x434F564F) { // 'OVOC'
        platform::log_error("tokenizer: bad vocab magic in %s", path.c_str());
        return false;
    }
    cursor += 4;
    const uint32_t version  = gguf::read_u32(p, cursor); cursor += 4;
    const uint32_t n_tokens = gguf::read_u32(p, cursor); cursor += 4;
    if (version != 1) {
        platform::log_error("tokenizer: unsupported vocab version %u", version);
        return false;
    }

    pieces_.clear();
    types_.clear();
    pieces_.reserve(n_tokens);
    types_.reserve(n_tokens);

    for (uint32_t i = 0; i < n_tokens; ++i) {
        const uint32_t len = gguf::read_u32(p, cursor); cursor += 4;
        pieces_.emplace_back(reinterpret_cast<const char*>(p + cursor), len);
        cursor += len;
        const int32_t t = gguf::read_i32(p, cursor); cursor += 4;
        types_.push_back(t);
    }

    rebuild_index();
    return valid();
}

bool Tokenizer::build_minimal(int32_t reserve) {
    pieces_.clear();
    types_.clear();

    const int32_t n = reserve < 512 ? 512 : reserve;
    pieces_.reserve(static_cast<size_t>(n));

    // 0: <unk>  1: <s>  2: </s>  3..15: control tokens  16..271: bytes
    pieces_.push_back("<unk>");        types_.push_back(2);
    pieces_.push_back("<s>");          types_.push_back(3);
    pieces_.push_back("</s>");         types_.push_back(3);
    const char* ctrls[] = {
        "<|user|>", "</|user|>", "<|assistant|>", "</|assistant|>",
        "<|vision|>", "</|vision|>", "<|audio|>", "</|audio|>",
        "<|tool_json|>", "<|think|>", "</|think|>", "<|pad|>",
    };
    for (const char* c : ctrls) {
        pieces_.push_back(c);
        types_.push_back(3);
    }
    for (int b = 0; b < 256; ++b) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "<0x%02X>", b);
        pieces_.push_back(buf);
        types_.push_back(6);
    }

    rebuild_index();
    detect_special_ids();
    return valid();
}

void Tokenizer::detect_special_ids() {
    auto lookup = [this](const std::string& piece) -> int32_t {
        auto it = index_.find(piece);
        return it != index_.end() ? it->second : -1;
    };
    bos_id_             = lookup("<s>");
    user_start_id_      = lookup("<|user|>");
    user_end_id_        = lookup("</|user|>");
    assistant_start_id_ = lookup("<|assistant|>");
    assistant_end_id_   = lookup("</|assistant|>");
}

void Tokenizer::rebuild_index() {
    index_.clear();
    for (size_t i = 0; i < pieces_.size(); ++i) {
        // First occurrence wins (matches SentencePiece semantics for dups).
        index_.emplace(pieces_[i], static_cast<int32_t>(i));
    }
}

// ===========================================================================
// Encoding: greedy longest match, then <0xXX> byte fallback
// ===========================================================================
std::vector<int32_t> Tokenizer::encode(const std::string& text,
                                       bool add_bos) const {
    std::vector<int32_t> out;
    if (!valid()) return out;
    // BOS only when the vocab actually defines one. World models have no <s>
    // piece (bos_token_id 0 is an unused pad), so encode_chat must NOT emit a
    // fake BOS there; the converter publishes omniseed.bos_token_id = -1.
    if (add_bos && bos_id_ >= 0) out.push_back(bos_id_);

    const size_t n = text.size();
    size_t i = 0;
    std::string probe;
    probe.reserve(32);

    while (i < n) {
        // Longest piece that matches at position i (cap at 32 bytes).
        const size_t max_len = std::min<size_t>(32, n - i);
        int32_t match = kUnkId;
        size_t  match_len = 0;

        for (size_t len = max_len; len >= 1; --len) {
            probe.assign(text, i, len);
            auto it = index_.find(probe);
            if (it != index_.end()) {
                match = it->second;
                match_len = len;
                break;
            }
        }

        if (match_len == 0) {
            // Byte fallback: <0xXX> token for the raw byte.
            char buf[8];
            std::snprintf(buf, sizeof(buf), "<0x%02X>",
                          static_cast<unsigned char>(text[i]));
            auto it = index_.find(buf);
            out.push_back(it != index_.end() ? it->second : kUnkId);
            i += 1;
        } else {
            out.push_back(match);
            i += match_len;
        }
    }
    return out;
}

std::vector<int32_t> Tokenizer::encode_chat(const std::string& user_text) const {
    // World-model template (matches RWKV-7 training): "User: <text>\n\nAssistant:"
    // when the vocab defines no <|user|> control tokens; token layout otherwise.
    if (user_start_id_ < 0) {
        auto body = encode("User: " + user_text + "\n\nAssistant:", false);
        return body;
    }
    std::vector<int32_t> ids;
    ids.push_back(bos_id_ >= 0 ? bos_id_ : kBosId);
    ids.push_back(user_start_id_);
    auto body = encode(user_text, false);
    ids.insert(ids.end(), body.begin(), body.end());
    ids.push_back(user_end_id_);
    ids.push_back(assistant_start_id_);
    return ids;
}

std::vector<int32_t> Tokenizer::wrap_modality(
        int32_t start_id, int32_t end_id,
        const std::vector<int32_t>& ids) const {
    std::vector<int32_t> out;
    out.reserve(ids.size() + 2);
    out.push_back(start_id);
    out.insert(out.end(), ids.begin(), ids.end());
    out.push_back(end_id);
    return out;
}

// ===========================================================================
// Decoding
// ===========================================================================
std::string Tokenizer::decode(const std::vector<int32_t>& ids,
                              bool include_special) const {
    std::string out;
    out.reserve(ids.size() * 4);

    for (const int32_t id : ids) {
        if (id < 0 || id >= static_cast<int32_t>(pieces_.size())) continue;
        const int32_t t = types_[static_cast<size_t>(id)];

        if (t == 3 && !include_special) continue;      // CONTROL
        if (t == 6) {                                   // BYTE <0xXX>
            unsigned v = 0;
            const std::string& pc = pieces_[static_cast<size_t>(id)];
            if (pc.size() == 6 && pc[0] == '<' && pc[1] == '0' && pc[2] == 'x') {
                std::sscanf(pc.c_str() + 3, "%2X", &v);
                out.push_back(static_cast<char>(v));
            }
            continue;
        }
        if (t == 2 && id != 0 && !include_special) continue;  // UNKNOWN skip
        out += pieces_[static_cast<size_t>(id)];
    }
    return out;
}

int32_t Tokenizer::find(const std::string& piece) const {
    auto it = index_.find(piece);
    return it != index_.end() ? it->second : kUnkId;
}

} // namespace omniseed
