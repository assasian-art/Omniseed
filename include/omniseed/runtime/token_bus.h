// =============================================================================
//  OmniSeed — token_bus.h
//  Any-to-Token Bus: fuses text, vision and audio token streams into ONE
//  unified sequence for the RWKV core.
//
//  Fusion model (blueprint): all modalities are tokens in a shared latent
//  space. Vision/audio payload ids are wrapped in modality markers
//  (<|vision|> ... </|vision|>) so the core can learn cross-modal binding
//  without any cross-attention machinery. Interleaving order is preserved:
//  e.g. text <|vision|> v1..vn </|vision|> text <|audio|> a1..am </|audio|>.
// =============================================================================
#pragma once

#include "omniseed/core/tokenizer.h"

#include <string>
#include <vector>

namespace omniseed {

class TokenBus {
public:
    explicit TokenBus(const Tokenizer& tok) : tok_(tok) {}

    // One segment of the fused stream.
    struct Segment {
        enum class Kind { Text, Vision, Audio } kind = Kind::Text;

        // Text payload (Kind::Text) or a diagnostic label for other kinds.
        std::string text;

        // Raw modality payload ids for Vision/Audio kinds (already produced
        // by the encoders; NOT text tokens).
        std::vector<int32_t> ids;
    };

    // Fuses segments into a single token stream, inserting BOS and modality
    // wrappers. Returns false if a payload exceeds max_tokens.
    bool fuse(const std::vector<Segment>& segments,
              std::vector<int32_t>& out_ids) const;

    // The budget: how many tokens the core may consume for one turn.
    void set_max_tokens(int64_t n) { max_tokens_ = n; }
    int64_t max_tokens() const { return max_tokens_; }

    // Statistics from the last fuse() call.
    struct Stats {
        int64_t text_tokens = 0;
        int64_t vision_tokens = 0;
        int64_t audio_tokens = 0;
        int64_t total = 0;
    };
    const Stats& last_stats() const { return stats_; }

private:
    const Tokenizer& tok_;
    int64_t max_tokens_ = 32768;
    mutable Stats stats_;
};

} // namespace omniseed
