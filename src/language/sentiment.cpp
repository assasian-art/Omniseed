// =============================================================================
//  OmniSeed — src/language/sentiment.cpp
//
//  Lexicon + negation sentiment. Deterministic, offline, no weights.
//
//  NEGATION IS THE WHOLE PROBLEM, and the rule here is a scoping rule rather
//  than a window rule, because a plain "look back 3 tokens" window gets the
//  most common English construction exactly backwards:
//
//      "not bad, it's good"   ->  a naive window sees "not" within 3 tokens of
//                                 "good" and flips it, scoring this NEGATIVE.
//
//  So a match looks back (and, for Bengali, forward) for a negator, but STOPS
//  at the nearest other sentiment word. "good" finds "bad" first and stops, so
//  it is not negated; "bad" finds "not" and is, giving positive. The forward
//  pass exists because Bengali negates post-positionally — "ভালো না" is
//  "good not", i.e. not good — where English negates pre-verbally.
//
//  KNOWN AMBIGUITY, documented rather than hidden: Bengali "না" is both the
//  negator and the sentence-final question particle ("তুমি যাবে না?" = "won't
//  you go?"). A lexical scorer cannot separate those without syntax, so a
//  trailing "না" will flip a nearby sentiment word. docs/JEV_FEATURES.md
//  records this.
//
//  `confidence` is a HEURISTIC STRENGTH, not a calibrated probability. See the
//  honesty note in include/omniseed/language/language_heads.h.
// =============================================================================
#include "omniseed/language/language_heads.h"

#include <algorithm>
#include <cstddef>

namespace omniseed {
namespace language {

namespace {

float clamp01(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }

bool in_set(const std::vector<std::string>& set, const std::string& t) {
    return std::find(set.begin(), set.end(), t) != set.end();
}

} // namespace

// ---------------------------------------------------------------------------
// Lexicons
//
// Negators are deliberately ABSENT from both lexicons. Keeping "no" out of the
// negative list is what makes "no problem" score positive: "problem" is
// negative, "no" flips it. The cost is that a bare "no" with no sentiment word
// beside it reads as Neutral — a trade the phrase-level behaviour is worth.
// ---------------------------------------------------------------------------
const std::vector<std::string>& positive_lexicon() {
    static const std::vector<std::string> v = {
        "good",      "great",     "excellent",  "awesome",    "amazing",
        "fantastic", "wonderful", "love",       "loved",      "loves",
        "like",      "liked",     "likes",      "best",       "better",
        "perfect",   "nice",      "happy",      "glad",       "pleased",
        "pleasing",  "helpful",   "brilliant",  "superb",     "win",
        "wins",      "won",       "profit",     "profits",    "success",
        "successful","improve",   "improved",   "improvement","fast",
        "faster",    "easy",      "easier",     "clean",      "cleaner",
        "works",     "working",   "solved",     "solve",      "correct",
        "right",     "yes",       "yeah",       "yep",        "agree",
        "agreed",    "recommend", "recommended","beautiful",  "cool",
        "excited",   "exciting",  "impressive", "solid",      "robust",
        "reliable",  "smooth",    "clear",      "useful",     "valuable",
        "safe",      "secure",    "stable",     "accurate",   "thanks",
        "thank",     "উত্তম",     "ভালো",        "ভাল",         "দারুণ",
        "চমৎকার",     "অসাধারণ",   "সুন্দর",      "খুশি",        "আনন্দ",
        "ধন্যবাদ",    "সফল",       "সফলতা",      "লাভ",         "সহজ",
        "দ্রুত",      "ঠিক",       "সঠিক",       "প্রশংসা",     "ভালোবাসা",
        "পছন্দ"};
    return v;
}

const std::vector<std::string>& negative_lexicon() {
    static const std::vector<std::string> v = {
        "bad",          "terrible",     "awful",       "horrible",
        "hate",         "hated",        "hates",       "dislike",
        "disliked",     "worst",        "worse",       "poor",
        "sad",          "angry",        "upset",       "annoying",
        "annoyed",      "broken",       "breaks",      "broke",
        "bug",          "bugs",         "buggy",       "fail",
        "failed",       "fails",        "failure",     "failures",
        "error",        "errors",       "crash",       "crashed",
        "crashes",      "slow",         "slower",      "hard",
        "harder",       "difficult",    "confusing",   "confused",
        "wrong",        "useless",      "unreliable",  "unsafe",
        "insecure",     "unstable",     "inaccurate",  "lose",
        "loses",        "lost",         "loss",        "losses",
        "problem",      "problems",     "issue",       "issues",
        "delay",        "delayed",      "stuck",       "block",
        "blocked",      "ugly",         "dirty",       "messy",
        "expensive",    "complicated",  "painful",     "frustrating",
        "disappointed", "disappointing","missing",     "leak",
        "leaks",        "leaked",       "corrupted",   "corrupt",
        "deny",         "denied",       "refuse",      "refused",
        "reject",       "rejected",     "dead",        "weak",
        "weaker",       "খারাপ",         "ভয়ানক",       "ঘৃণা",
        "দুঃখ",         "রাগ",          "বিরক্ত",       "ভাঙা",
        "ব্যর্থ",       "ব্যর্থতা",      "সমস্যা",      "ভুল",
        "ধীর",          "কঠিন",         "ক্ষতি",       "বিপদ",
        "দেরি",         "আটকে",         "অসহ্য",       "জঘন্য",
        "নষ্ট",         "হতাশ",         "বাজে"};
    return v;
}

const std::vector<std::string>& negation_markers() {
    static const std::vector<std::string> v = {
        // English — contractions survive tokenisation intact, so they are
        // listed whole rather than as stems.
        "no",        "not",       "never",      "none",      "nobody",
        "nothing",   "neither",   "nor",        "without",   "lack",
        "lacks",     "lacking",   "hardly",     "barely",    "rarely",
        "seldom",    "don't",     "doesn't",    "didn't",    "isn't",
        "aren't",    "wasn't",    "weren't",    "won't",     "wouldn't",
        "shouldn't", "couldn't",  "can't",      "cannot",    "haven't",
        "hasn't",    "hadn't",    "ain't",
        // Bengali
        "না",        "নেই",       "নয়",         "নাই",       "নাহ"};
    return v;
}

// ---------------------------------------------------------------------------
// Names
// ---------------------------------------------------------------------------
const char* polarity_name(Polarity p) {
    switch (p) {
        case Polarity::Negative: return "negative";
        case Polarity::Neutral:  return "neutral";
        case Polarity::Positive: return "positive";
        case Polarity::COUNT:    break;
    }
    return "neutral";
}

// ---------------------------------------------------------------------------
// SentimentScorer
// ---------------------------------------------------------------------------
SentimentResult SentimentScorer::score(const std::string& text) const {
    SentimentResult r;

    const std::vector<std::string> toks = tokenize(text);
    if (toks.empty()) return r;   // Neutral / 0 / 0 / no matches

    const std::vector<std::string>& pos  = positive_lexicon();
    const std::vector<std::string>& neg  = negative_lexicon();
    const std::vector<std::string>& negm = negation_markers();

    // kind[i]: +1 positive hit, -1 negative hit, 0 neither. Negators are 0, so
    // the "stop at the nearest sentiment word" rule below never stops on them.
    std::vector<int32_t> kind(toks.size(), 0);
    for (size_t i = 0; i < toks.size(); ++i) {
        if (in_set(pos, toks[i]))      kind[i] = 1;
        else if (in_set(neg, toks[i])) kind[i] = -1;
    }

    int32_t p = 0;
    int32_t n = 0;
    for (size_t i = 0; i < toks.size(); ++i) {
        if (kind[i] == 0) continue;

        bool flip = false;

        // Backward pass — English negation is pre-verbal.
        for (size_t k = 1; k <= 3 && k <= i; ++k) {
            if (kind[i - k] != 0) break;          // a closer sentiment word owns the scope
            if (in_set(negm, toks[i - k])) { flip = true; break; }
        }

        // Forward pass — Bengali negation is post-positional ("ভালো না").
        if (!flip) {
            for (size_t k = 1; k <= 2 && i + k < toks.size(); ++k) {
                if (kind[i + k] != 0) break;
                if (in_set(negm, toks[i + k])) { flip = true; break; }
            }
        }

        int32_t sign = kind[i];
        if (flip) { sign = -sign; ++r.negated; }
        if (sign > 0) ++p; else ++n;
        r.matched.push_back(toks[i]);
    }

    const int32_t total = p + n;
    if (total == 0) return r;   // Neutral / 0 / 0 — no lexicon hit at all

    r.score = static_cast<float>(p - n) / static_cast<float>(total);
    // (p - n) / total is exact in float for the small integer counts here, so a
    // plain sign test is safe and an epsilon would only add noise.
    if (r.score > 0.0f)      r.polarity = Polarity::Positive;
    else if (r.score < 0.0f) r.polarity = Polarity::Negative;
    else                     r.polarity = Polarity::Neutral;

    // Heuristic strength: how many hits there were, and how one-sided they
    // were. NOT a calibrated probability.
    const float hit_part = std::min(1.0f, static_cast<float>(total) / 3.0f);
    const float mag      = r.score < 0.0f ? -r.score : r.score;
    r.confidence = clamp01(0.5f * hit_part + 0.5f * mag);
    return r;
}

// ---------------------------------------------------------------------------
// SentimentResult::to_json
// ---------------------------------------------------------------------------
std::string SentimentResult::to_json() const {
    std::string out = "{\"polarity\":\"";
    out += polarity_name(polarity);
    out += "\",\"score\":";
    heads_detail::append_float(out, score);
    out += ",\"confidence\":";
    heads_detail::append_float(out, confidence);
    out += ",\"negated\":";
    out += std::to_string(negated);
    out += ",\"matched\":[";
    for (size_t i = 0; i < matched.size(); ++i) {
        if (i) out += ',';
        out += '"';
        out += heads_detail::json_escape(matched[i]);
        out += '"';
    }
    out += "]}";
    return out;
}

} // namespace language
} // namespace omniseed
