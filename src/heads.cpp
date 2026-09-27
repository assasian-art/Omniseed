// =============================================================================
//  OmniSeed — src/heads.cpp
//
//  The shared vocabulary: domain names, and the JSON forms of a classification
//  distribution and a score triple.
//
//  JSON is assembled by hand rather than through a library, because this tree
//  has zero external dependencies and the documents here are small and fixed
//  in shape. The one rule that matters: never emit a bare `nan` or `inf`, since
//  that produces an invalid document and a downstream parse failure is a much
//  worse outcome than a visible 0.0 default.
// =============================================================================
#include "omniseed/heads.h"

namespace omniseed {

// ---------------------------------------------------------------------------
// Domain
// ---------------------------------------------------------------------------
const char* domain_name(Domain d) {
    switch (d) {
        case Domain::General:  return "general";
        case Domain::Trading:  return "trading";
        case Domain::Language: return "language";
        case Domain::Vision:   return "vision";
        case Domain::Audio:    return "audio";
        case Domain::COUNT:    break;
    }
    return "general";
}

Domain domain_from_name(const std::string& name) {
    for (int32_t i = 0; i < kDomainCount; ++i) {
        const Domain d = static_cast<Domain>(i);
        if (name == domain_name(d)) return d;
    }
    // Unrecognised input is "general", NOT a specialist domain. A typo must not
    // be able to route a request to the trading head.
    return Domain::General;
}

bool domain_known(const std::string& name) {
    for (int32_t i = 0; i < kDomainCount; ++i)
        if (name == domain_name(static_cast<Domain>(i))) return true;
    return false;
}

// ---------------------------------------------------------------------------
// ClassificationResult
// ---------------------------------------------------------------------------
std::string ClassificationResult::to_json() const {
    std::string out = "{\"domain\":\"";
    out += heads_detail::json_escape(domain);
    out += "\",\"top_k\":[";
    for (size_t i = 0; i < top_k.size(); ++i) {
        if (i) out += ',';
        out += "{\"label\":\"";
        out += heads_detail::json_escape(top_k[i].label);
        out += "\",\"probability\":";
        heads_detail::append_float(out, top_k[i].probability);
        out += '}';
    }
    out += "],\"margin\":";
    heads_detail::append_float(out, margin);
    out += ",\"matvecs\":";
    out += std::to_string(matvecs);
    out += '}';
    return out;
}

// ---------------------------------------------------------------------------
// ScoreResult
// ---------------------------------------------------------------------------
std::string ScoreResult::to_json() const {
    std::string out = "{\"priority\":";
    heads_detail::append_float(out, priority);
    out += ",\"urgency\":";
    heads_detail::append_float(out, urgency);
    out += ",\"confidence\":";
    heads_detail::append_float(out, confidence);
    out += ",\"matvecs\":";
    out += std::to_string(matvecs);
    out += '}';
    return out;
}

} // namespace omniseed
