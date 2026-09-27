// =============================================================================
//  OmniSeed — language/language_heads.h
//
//  The LANGUAGE domain, in C++, with NO model and NO Python.
//
//  WHY THIS IS NOT A NEURAL HEAD. Intent, language and sentiment for a chat
//  surface are overwhelmingly decided by surface cues: a trailing '?', an
//  opening interrogative, a greeting phrase, a script range. A deterministic
//  lexical classifier answers those in microseconds, is debuggable, cannot
//  hallucinate, and — the property that matters most here — WORKS WITH NO
//  TRAINED WEIGHTS. The neural ClassificationHead is the upgrade path for the
//  cases this cannot reach; it is not a prerequisite for the common ones.
//
//  BILINGUAL BY CONSTRUCTION, NOT BY TRANSLATION. The mandate names EN/BN
//  explicitly. Bengali is detected from its own Unicode block (U+0980–U+09FF)
//  rather than from a romanisation, and the lexicons carry Bengali entries
//  alongside English ones, so a Bengali greeting is recognised as a greeting
//  directly instead of via a lossy English hop.
//
//  HONESTY NOTE — READ THIS BEFORE USING `confidence`.
//  The confidence these classifiers report is a HEURISTIC STRENGTH in [0,1]:
//  how many independent cues agreed. It is NOT a calibrated probability, and
//  saying "92% confident" about it would be a lie of exactly the kind the
//  mandate's calibration requirement exists to prevent. Calibrating a lexical
//  classifier requires labelled data and a fit; until that exists, `confidence`
//  is documented as a prior, and docs/JEV_FEATURES.md records the gap.
// =============================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "omniseed/heads.h"   // LabelProb

namespace omniseed {
namespace language {

// ---------------------------------------------------------------------------
// Language of a piece of text.
// ---------------------------------------------------------------------------
enum class Lang : int32_t { En = 0, Bn, Mixed, Unknown, COUNT };
constexpr int32_t kLangCount = static_cast<int32_t>(Lang::COUNT);
const char* lang_name(Lang l);

// ---------------------------------------------------------------------------
// Intent
// ---------------------------------------------------------------------------
enum class Intent : int32_t {
    Unknown = 0,     // empty / no content — the fail-closed default
    Question,
    Statement,
    Command,
    Greeting,
    Farewell,
    Thanks,
    COUNT
};
constexpr int32_t kIntentCount = static_cast<int32_t>(Intent::COUNT);
const char* intent_name(Intent i);

// UTF-8 scan: how much of each script the text contains.
struct ScriptStats {
    int32_t ascii_letters = 0;
    int32_t bengali       = 0;   // U+0980..U+09FF
    int32_t digits        = 0;
    int32_t codepoints    = 0;
    int32_t invalid_bytes = 0;
};
ScriptStats scan_utf8(const std::string& text);

// Lower-cased ASCII words and whole Bengali runs, in order. Punctuation is
// dropped except '?' and '!', which are kept as standalone tokens because they
// are the single strongest intent cue there is. An apostrophe followed by a
// letter continues the word, so "don't" stays one token rather than becoming
// "don" + "t" — negation detection depends on that; leading and trailing
// quotes still separate, so "'hello'" is just "hello".
std::vector<std::string> tokenize(const std::string& text);

struct IntentResult {
    Intent      intent   = Intent::Unknown;
    Lang        language = Lang::Unknown;
    // Heuristic strength in [0,1] — NOT a calibrated probability. See the note
    // at the top of this file.
    float       confidence = 0.0f;
    std::vector<LabelProb> top_k;      // the intent distribution, best first
    std::vector<std::string> cues;     // which rules fired, for debugging
    std::string to_json() const;
};

class IntentClassifier {
public:
    IntentClassifier() = default;
    IntentResult classify(const std::string& text) const;
};

// ---------------------------------------------------------------------------
// Sentiment
// ---------------------------------------------------------------------------
enum class Polarity : int32_t { Negative = 0, Neutral, Positive, COUNT };
constexpr int32_t kPolarityCount = static_cast<int32_t>(Polarity::COUNT);
const char* polarity_name(Polarity p);

struct SentimentResult {
    Polarity    polarity = Polarity::Neutral;
    // Mean signed valence in [-1, 1]: (pos - neg) / (pos + neg) over the matched
    // terms, after negation has flipped its neighbours.
    float       score    = 0.0f;
    // Heuristic strength in [0,1] — NOT a calibrated probability.
    float       confidence = 0.0f;
    std::vector<std::string> matched;   // the lexicon entries that fired
    int32_t     negated = 0;            // how many matches were flipped
    std::string to_json() const;
};

class SentimentScorer {
public:
    SentimentScorer() = default;
    SentimentResult score(const std::string& text) const;
};

// The lexicons, exposed so tests can assert the vocabulary rather than
// duplicating it, and so a caller can extend it deliberately.
const std::vector<std::string>& positive_lexicon();
const std::vector<std::string>& negative_lexicon();
const std::vector<std::string>& negation_markers();

} // namespace language
} // namespace omniseed
