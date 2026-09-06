// =============================================================================
//  OmniSeed — token_bus.cpp
//  Any-to-Token Bus: fusion of heterogeneous modality streams.
// =============================================================================
#include "omniseed/runtime/token_bus.h"

namespace omniseed {

bool TokenBus::fuse(const std::vector<Segment>& segments,
                    std::vector<int32_t>& out_ids) const {
    out_ids.clear();
    stats_ = Stats{};

    out_ids.push_back(Tokenizer::kBosId);

    for (const Segment& seg : segments) {
        switch (seg.kind) {
            case Segment::Kind::Text: {
                auto ids = tok_.encode(seg.text, false);
                stats_.text_tokens += static_cast<int64_t>(ids.size());
                out_ids.insert(out_ids.end(), ids.begin(), ids.end());
                break;
            }
            case Segment::Kind::Vision: {
                if (seg.ids.empty()) break;
                auto wrapped = tok_.wrap_modality(
                    Tokenizer::kVisionStartId, Tokenizer::kVisionEndId, seg.ids);
                stats_.vision_tokens += static_cast<int64_t>(wrapped.size());
                out_ids.insert(out_ids.end(), wrapped.begin(), wrapped.end());
                break;
            }
            case Segment::Kind::Audio: {
                if (seg.ids.empty()) break;
                auto wrapped = tok_.wrap_modality(
                    Tokenizer::kAudioStartId, Tokenizer::kAudioEndId, seg.ids);
                stats_.audio_tokens += static_cast<int64_t>(wrapped.size());
                out_ids.insert(out_ids.end(), wrapped.begin(), wrapped.end());
                break;
            }
        }

        if (static_cast<int64_t>(out_ids.size()) > max_tokens_) {
            return false;   // budget exceeded — caller decides what to trim
        }
    }

    stats_.total = static_cast<int64_t>(out_ids.size());
    return true;
}

} // namespace omniseed
