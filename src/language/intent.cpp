// =============================================================================
//  OmniSeed — src/language/intent.cpp
//
//  Deterministic intent + language classification. No model, no weights, no
//  Python, no network. See include/omniseed/language/language_heads.h for why
//  this is a lexical classifier rather than a neural head, and for the honesty
//  note on `confidence`.
//
//  DETECTION IS BY SCRIPT, NOT BY LANGUAGE. Bengali is recognised from its own
//  Unicode block (U+0980..U+09FF). Romanised Bengali ("ami bhalo achi") is pure
//  ASCII and therefore reads as En — separating it from English needs a
//  transliteration model this tree does not have. docs/JEV_FEATURES.md records
//  that gap rather than papering over it.
//
//  CUE WEIGHTS. Every rule below adds a weight to exactly one intent and
//  records a short rule name in `cues`, so a surprising answer can be traced to
//  the rule that produced it. Weights are ordered by how hard the cue is:
//
//    3.0  an explicit greeting / farewell / thanks marker
//    2.0  a structural cue: trailing '?', interrogative opener, imperative
//         opener, Bengali question word, Bengali imperative suffix
//    1.5  the politeness marker "please"
//    1.0  the declarative baseline, and any '?' that is not trailing
//    0.25 a bare '!' (a nudge, deliberately too small to change a verdict)
//
//  The baseline of 1.0 is what makes "no cue fired" resolve to Statement
//  instead of a coin flip among six intents.
// =============================================================================
#include "omniseed/language/language_heads.h"

#include <algorithm>
#include <cstddef>

namespace omniseed {
namespace language {

// ---------------------------------------------------------------------------
// UTF-8 primitives
// ---------------------------------------------------------------------------
namespace {

constexpr int32_t kMaxCodepoint = 0x10FFFF;

// Decode the sequence at text[i], advancing i past it. Returns -1 for an
// invalid byte. On failure exactly ONE byte is consumed, so a malformed stream
// cannot wedge the scanner in a loop that never advances.
int32_t decode_one(const std::string& t, size_t& i, int32_t& invalid) {
    const unsigned char c0 = static_cast<unsigned char>(t[i]);
    if (c0 < 0x80) { ++i; return static_cast<int32_t>(c0); }

    int32_t cp    = 0;
    int32_t extra = 0;
    if ((c0 & 0xE0) == 0xC0)      { extra = 1; cp = c0 & 0x1F; }
    else if ((c0 & 0xF0) == 0xE0) { extra = 2; cp = c0 & 0x0F; }
    else if ((c0 & 0xF8) == 0xF0) { extra = 3; cp = c0 & 0x07; }
    else                          { ++i; ++invalid; return -1; }

    if (i + static_cast<size_t>(extra) >= t.size()) { ++i; ++invalid; return -1; }
    for (int32_t k = 1; k <= extra; ++k) {
        const unsigned char cc = static_cast<unsigned char>(t[i + static_cast<size_t>(k)]);
        if ((cc & 0xC0) != 0x80) { ++i; ++invalid; return -1; }
        cp = (cp << 6) | (cc & 0x3F);
    }
    i += static_cast<size_t>(extra) + 1;

    // Overlong encodings, surrogates and out-of-range values are not UTF-8.
    if (cp > kMaxCodepoint || (cp >= 0xD800 && cp <= 0xDFFF)) { ++invalid; return -1; }
    if ((extra == 1 && cp < 0x80) || (extra == 2 && cp < 0x800) ||
        (extra == 3 && cp < 0x10000)) { ++invalid; return -1; }
    return cp;
}

inline bool is_ascii_letter(int32_t cp) {
    return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z');
}
inline bool is_ascii_digit(int32_t cp) { return cp >= '0' && cp <= '9'; }
inline bool is_bengali(int32_t cp)     { return cp >= 0x0980 && cp <= 0x09FF; }

float clamp01(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }

// --- small set helpers -----------------------------------------------------
using StrSet = std::vector<std::string>;

bool has_token(const std::vector<std::string>& toks, const StrSet& set) {
    for (const std::string& t : toks)
        if (std::find(set.begin(), set.end(), t) != set.end()) return true;
    return false;
}

bool has_token_at(const std::vector<std::string>& toks, size_t idx, const StrSet& set) {
    if (idx >= toks.size()) return false;
    return std::find(set.begin(), set.end(), toks[idx]) != set.end();
}

bool has_anywhere_in_first(const std::vector<std::string>& toks, size_t n, const StrSet& set) {
    const size_t lim = n < toks.size() ? n : toks.size();
    for (size_t i = 0; i < lim; ++i)
        if (std::find(set.begin(), set.end(), toks[i]) != set.end()) return true;
    return false;
}

bool has_pair(const std::vector<std::string>& toks, const StrSet& first, const StrSet& second) {
    for (size_t i = 0; i + 1 < toks.size(); ++i)
        if (std::find(first.begin(), first.end(), toks[i]) != first.end() &&
            std::find(second.begin(), second.end(), toks[i + 1]) != second.end())
            return true;
    return false;
}

bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool any_token_ends_with(const std::vector<std::string>& toks, const StrSet& suffixes) {
    for (const std::string& t : toks)
        for (const std::string& s : suffixes)
            if (ends_with(t, s)) return true;
    return false;
}

// --- cue vocabularies ------------------------------------------------------
// Lower-cased ASCII, because tokenize() lower-cases ASCII words.

const StrSet& greeting_en() {
    static const StrSet v = {"hi", "hii", "hello", "helo", "hey", "heya", "hiya",
                             "howdy", "yo", "greetings", "hola", "namaste",
                             "salam", "salams", "assalamu", "assalamualaikum"};
    return v;
}
const StrSet& greeting_bn() {
    static const StrSet v = {"হ্যালো", "হাই", "নমস্কার", "নমস্কর", "আসসালামু",
                             "সালাম", "শুভ", "সুপ্রভাত", "নমস্তে"};
    return v;
}
const StrSet& greeting_first_word() {   // "good morning" etc.
    static const StrSet v = {"good", "শুভ"};
    return v;
}
const StrSet& greeting_second_word() {
    static const StrSet v = {"morning", "afternoon", "evening", "night", "day",
                             "সকাল", "দুপুর", "বিকাল", "সন্ধ্যা", "রাত", "দিন"};
    return v;
}

const StrSet& farewell_en() {
    static const StrSet v = {"bye", "byebye", "goodbye", "goodbyes", "farewell",
                             "cya", "later", "tata", "adios", "cheers"};
    return v;
}
const StrSet& farewell_bn() {
    static const StrSet v = {"বিদায়", "আলবিদা", "খোদাহাফেজ", "খোদা", "হাফেজ"};
    return v;
}
const StrSet& farewell_first_word() {
    static const StrSet v = {"see", "take", "allah", "god", "আল্লাহ", "ভালো"};
    return v;
}
const StrSet& farewell_second_word() {
    static const StrSet v = {"you", "care", "hafez", "হাফেজ", "থাকুন", "থাকো"};
    return v;
}

const StrSet& thanks_en() {
    static const StrSet v = {"thanks", "thank", "thankyou", "thankz", "thx", "ty",
                             "tysm", "cheers", "shukriya", "shukria", "dhanyabad"};
    return v;
}
const StrSet& thanks_bn() {
    static const StrSet v = {"ধন্যবাদ", "শুকরিয়া", "ধন্যবাদ", "কৃতজ্ঞ"};
    return v;
}

// Interrogative openers: only counted in the first two token positions, which
// is where an English question actually puts them. "is" mid-sentence is a verb,
// not a question.
const StrSet& interrogative_en() {
    static const StrSet v = {"what", "why", "how", "when", "where", "who", "whom",
                             "whose", "which", "whats", "is", "are", "was", "were",
                             "am", "do", "does", "did", "can", "could", "should",
                             "would", "will", "shall", "may", "might"};
    return v;
}
// Bengali question words sit mid-sentence, so these count ANYWHERE.
const StrSet& interrogative_bn() {
    static const StrSet v = {"কেন", "কী", "কি", "কোথায়", "কেমন", "কত", "কে",
                             "কোন", "কখন", "কারা", "কতটা", "কীভাবে"};
    return v;
}

// Imperative openers: English puts the verb first.
const StrSet& imperative_en() {
    static const StrSet v = {"please", "run", "show", "list", "open", "close",
                             "delete", "remove", "create", "make", "give", "tell",
                             "set", "start", "stop", "add", "find", "search",
                             "help", "explain", "analyze", "analyse", "build",
                             "deploy", "install", "update", "fix", "check", "test",
                             "send", "write", "read", "print", "compute",
                             "calculate", "compare", "summarize", "summarise",
                             "generate", "fetch", "get", "put", "move", "rename",
                             "copy", "restart", "kill", "enable", "disable",
                             "configure", "refactor", "review", "optimize",
                             "optimise", "port", "convert", "migrate", "dump",
                             "load", "save", "export", "import", "scan", "monitor",
                             "watch", "schedule", "call", "reply", "respond",
                             "answer", "translate"};
    return v;
}
// Bengali is verb-final, so the imperative is a SUFFIX on the last token.
const StrSet& imperative_bn_suffix() {
    static const StrSet v = {"করো", "করুন", "করা", "দাও", "দিন", "বলো", "বলুন",
                             "দেখাও", "দেখান", "চালাও", "চালান", "খোলো", "খুলুন",
                             "নাও", "এসো", "পড়ো", "লেখো", "পাঠাও", "পাঠান",
                             "খুঁজো", "তৈরি", "বন্ধ", "শুরু", "বের", "দিন"};
    return v;
}

// "please" is the one politeness marker that is unambiguous anywhere.
const StrSet& politeness_en() {
    static const StrSet v = {"please", "pls", "plz", "kindly"};
    return v;
}

const StrSet& question_marks() {
    static const StrSet v = {"?"};
    return v;
}
const StrSet& exclamations() {
    static const StrSet v = {"!"};
    return v;
}

// ---------------------------------------------------------------------------
// Accumulator: one per intent.
// ---------------------------------------------------------------------------
struct Acc {
    float       score = 0.0f;
    int32_t     cues  = 0;
    std::vector<std::string> fired;
};

void vote(Acc& a, float weight, const char* rule) {
    a.score += weight;
    ++a.cues;
    a.fired.push_back(rule);
}

} // namespace

// ---------------------------------------------------------------------------
// Script statistics
// ---------------------------------------------------------------------------
ScriptStats scan_utf8(const std::string& text) {
    ScriptStats s;
    size_t i = 0;
    while (i < text.size()) {
        const int32_t cp = decode_one(text, i, s.invalid_bytes);
        if (cp < 0) continue;
        ++s.codepoints;
        if (cp < 0x80) {
            if (is_ascii_letter(cp))      ++s.ascii_letters;
            else if (is_ascii_digit(cp))  ++s.digits;
        } else if (is_bengali(cp)) {
            // The whole block counts as Bengali script, digits included; the
            // Bengali digits U+09E6..U+09EF also count as digits.
            ++s.bengali;
            if (cp >= 0x09E6 && cp <= 0x09EF) ++s.digits;
        }
    }
    return s;
}

// ---------------------------------------------------------------------------
// Tokenizer
//
// Lower-cased ASCII words, whole Bengali runs, and '?' / '!' kept standalone.
// Everything else separates.
// ---------------------------------------------------------------------------
std::vector<std::string> tokenize(const std::string& text) {
    std::vector<std::string> out;
    std::string word;          // current ASCII alnum run, lower-cased
    size_t      beng_start = 0;
    size_t      beng_end   = 0;
    bool        in_beng    = false;

    // Two separate closes, because the two run types must be able to end
    // independently. Starting an ASCII word has to end a Bengali run WITHOUT
    // destroying the word being built; folding both into one `flush` that
    // cleared everything made every single letter its own token. The tests in
    // tests/test_language_heads.cpp caught that immediately, and the fix is to
    // keep the two operations distinct.
    auto close_word = [&]() {
        if (!word.empty()) { out.push_back(word); word.clear(); }
    };
    auto close_beng = [&]() {
        if (in_beng && beng_end > beng_start)
            out.push_back(text.substr(beng_start, beng_end - beng_start));
        in_beng = false;
    };
    auto flush = [&]() { close_word(); close_beng(); };

    size_t  i     = 0;
    int32_t dummy = 0;
    while (i < text.size()) {
        const size_t  before = i;
        const int32_t cp     = decode_one(text, i, dummy);
        if (cp < 0) { flush(); continue; }

        if (cp < 0x80) {
            const char c = static_cast<char>(cp);
            if (is_ascii_letter(cp) || is_ascii_digit(cp)) {
                close_beng();          // end any Bengali run, keep the word
                word += (c >= 'A' && c <= 'Z')
                            ? static_cast<char>(c - 'A' + 'a')
                            : c;
            } else if (c == '\'' && !word.empty() && i < text.size() &&
                       is_ascii_letter(static_cast<unsigned char>(text[i]))) {
                // An apostrophe FOLLOWED BY A LETTER continues the word, so
                // "don't" survives tokenisation as one token. Negation detection
                // depends on this: splitting it into "don" + "t" loses the most
                // common English negator. A leading quote (empty `word`) and a
                // TRAILING one (nothing letter-like after it) both still
                // separate, so "'hello'" tokenises to "hello", not "hello'".
                word += c;
            } else if (c == '?' || c == '!') {
                flush();
                out.push_back(std::string(1, c));
            } else {
                flush();
            }
        } else if (is_bengali(cp)) {
            if (!in_beng) { close_word(); beng_start = before; in_beng = true; }
            beng_end = i;
        } else {
            flush();
        }
    }
    flush();
    return out;
}

// ---------------------------------------------------------------------------
// Names
// ---------------------------------------------------------------------------
const char* lang_name(Lang l) {
    switch (l) {
        case Lang::En:      return "en";
        case Lang::Bn:      return "bn";
        case Lang::Mixed:   return "mixed";
        case Lang::Unknown: return "unknown";
        case Lang::COUNT:   break;
    }
    return "unknown";
}

const char* intent_name(Intent i) {
    switch (i) {
        case Intent::Unknown:   return "unknown";
        case Intent::Question:  return "question";
        case Intent::Statement: return "statement";
        case Intent::Command:   return "command";
        case Intent::Greeting:  return "greeting";
        case Intent::Farewell:  return "farewell";
        case Intent::Thanks:    return "thanks";
        case Intent::COUNT:     break;
    }
    return "unknown";
}

// ---------------------------------------------------------------------------
// Language detection — by SCRIPT, with a dominance margin so a single stray
// foreign word does not turn a monolingual sentence into "mixed".
// ---------------------------------------------------------------------------
namespace {

Lang detect_lang(const ScriptStats& s) {
    const int32_t a = s.ascii_letters;
    const int32_t b = s.bengali;
    if (a == 0 && b == 0) return Lang::Unknown;
    if (a == 0) return Lang::Bn;
    if (b == 0) return Lang::En;
    const int32_t minority = a < b ? a : b;
    const int32_t majority = a < b ? b : a;
    // A minority script below 20% of the letters is treated as noise/borrowing.
    if (minority * 5 <= majority) return a > b ? Lang::En : Lang::Bn;
    return Lang::Mixed;
}

} // namespace

// ---------------------------------------------------------------------------
// IntentClassifier
// ---------------------------------------------------------------------------
IntentResult IntentClassifier::classify(const std::string& text) const {
    IntentResult r;

    const std::vector<std::string> toks = tokenize(text);
    const ScriptStats stats = scan_utf8(text);
    r.language = detect_lang(stats);

    if (toks.empty()) {
        r.intent = Intent::Unknown;
        r.confidence = 0.0f;
        r.cues.push_back("empty");
        r.top_k.push_back(LabelProb{intent_name(Intent::Unknown), 1.0f});
        return r;
    }

    Acc acc[kIntentCount];
    const size_t n = toks.size();

    // --- greeting -----------------------------------------------------------
    if (has_token_at(toks, 0, greeting_en()) || has_token_at(toks, 0, greeting_bn()) ||
        has_pair(toks, greeting_first_word(), greeting_second_word())) {
        vote(acc[static_cast<int>(Intent::Greeting)], 3.0f, "greeting_marker");
    } else if (has_token(toks, greeting_en()) || has_token(toks, greeting_bn())) {
        vote(acc[static_cast<int>(Intent::Greeting)], 1.0f, "greeting_marker_late");
    }

    // --- farewell -----------------------------------------------------------
    if (has_token_at(toks, 0, farewell_en()) || has_token_at(toks, 0, farewell_bn()) ||
        has_pair(toks, farewell_first_word(), farewell_second_word())) {
        vote(acc[static_cast<int>(Intent::Farewell)], 3.0f, "farewell_marker");
    } else if (has_token(toks, farewell_en()) || has_token(toks, farewell_bn())) {
        vote(acc[static_cast<int>(Intent::Farewell)], 1.0f, "farewell_marker_late");
    }

    // --- thanks -------------------------------------------------------------
    if (has_token_at(toks, 0, thanks_en()) || has_token_at(toks, 0, thanks_bn())) {
        vote(acc[static_cast<int>(Intent::Thanks)], 3.0f, "thanks_marker");
    } else if (has_token(toks, thanks_en()) || has_token(toks, thanks_bn())) {
        vote(acc[static_cast<int>(Intent::Thanks)], 1.0f, "thanks_marker_late");
    }

    // --- question -----------------------------------------------------------
    const bool trailing_q = toks[n - 1] == "?";
    if (trailing_q) {
        vote(acc[static_cast<int>(Intent::Question)], 2.0f, "trailing_question_mark");
    } else if (has_token(toks, question_marks())) {
        vote(acc[static_cast<int>(Intent::Question)], 1.0f, "question_mark");
    }
    if (has_anywhere_in_first(toks, 2, interrogative_en())) {
        vote(acc[static_cast<int>(Intent::Question)], 2.0f, "interrogative_opener");
    }
    if (has_token(toks, interrogative_bn())) {
        vote(acc[static_cast<int>(Intent::Question)], 2.0f, "bn_question_word");
    }

    // --- command ------------------------------------------------------------
    if (has_token_at(toks, 0, imperative_en())) {
        vote(acc[static_cast<int>(Intent::Command)], 2.0f, "imperative_opener");
    }
    if (has_token(toks, politeness_en())) {
        vote(acc[static_cast<int>(Intent::Command)], 1.5f, "politeness_marker");
    }
    if (any_token_ends_with(toks, imperative_bn_suffix())) {
        vote(acc[static_cast<int>(Intent::Command)], 2.0f, "bn_imperative_suffix");
    }
    if (has_token(toks, exclamations())) {
        vote(acc[static_cast<int>(Intent::Command)], 0.25f, "exclamation_weak");
    }

    // --- declarative baseline ----------------------------------------------
    // Requires an actual LETTER, not just digits or punctuation: "123" is not a
    // statement, it is content the classifier does not understand, and calling
    // it Statement would be inventing a verdict.
    if (stats.ascii_letters > 0 || stats.bengali > 0) {
        vote(acc[static_cast<int>(Intent::Statement)], 1.0f, "declarative_default");
    }

    // --- reduce -------------------------------------------------------------
    float total = 0.0f;
    for (int32_t k = 0; k < kIntentCount; ++k) total += acc[k].score;
    if (total <= 0.0f) {
        r.intent = Intent::Unknown;
        r.confidence = 0.0f;
        r.cues.push_back("no_cue_fired");
        r.top_k.push_back(LabelProb{intent_name(Intent::Unknown), 1.0f});
        return r;
    }

    std::vector<std::pair<float, int32_t>> ranked;
    ranked.reserve(static_cast<size_t>(kIntentCount));
    for (int32_t k = 0; k < kIntentCount; ++k)
        if (acc[k].score > 0.0f) ranked.emplace_back(acc[k].score, k);
    std::sort(ranked.begin(), ranked.end(),
              [](const std::pair<float, int32_t>& a, const std::pair<float, int32_t>& b) {
                  if (a.first != b.first) return a.first > b.first;
                  return a.second < b.second;   // stable, deterministic tie-break
              });

    const int32_t winner = ranked.front().second;
    r.intent = static_cast<Intent>(winner);

    // Probabilities are shares of the FULL total, so a top-k that omits a long
    // tail correctly does not sum to 1 — the same convention as
    // ClassificationResult::top_k.
    const size_t k = ranked.size() < 3 ? ranked.size() : 3;
    for (size_t i = 0; i < k; ++i)
        r.top_k.push_back(LabelProb{intent_name(static_cast<Intent>(ranked[i].second)),
                                    ranked[i].first / total});

    // Heuristic strength: half from how much of the mass the winner took, half
    // from how many independent rules agreed. NOT a calibrated probability —
    // see the honesty note in the header.
    const float share    = ranked.front().first / total;
    const float cue_part = std::min(1.0f, static_cast<float>(acc[winner].cues) / 3.0f);
    r.confidence = clamp01(0.5f * share + 0.5f * cue_part);

    r.cues = acc[winner].fired;
    return r;
}

// ---------------------------------------------------------------------------
// IntentResult::to_json
// ---------------------------------------------------------------------------
std::string IntentResult::to_json() const {
    std::string out = "{\"intent\":\"";
    out += intent_name(intent);
    out += "\",\"language\":\"";
    out += lang_name(language);
    out += "\",\"confidence\":";
    heads_detail::append_float(out, confidence);
    out += ",\"top_k\":[";
    for (size_t i = 0; i < top_k.size(); ++i) {
        if (i) out += ',';
        out += "{\"label\":\"";
        out += heads_detail::json_escape(top_k[i].label);
        out += "\",\"probability\":";
        heads_detail::append_float(out, top_k[i].probability);
        out += '}';
    }
    out += "],\"cues\":[";
    for (size_t i = 0; i < cues.size(); ++i) {
        if (i) out += ',';
        out += '"';
        out += heads_detail::json_escape(cues[i]);
        out += '"';
    }
    out += "]}";
    return out;
}

} // namespace language
} // namespace omniseed
