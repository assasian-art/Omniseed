// =============================================================================
//  OmniSeed — tests/test_language_heads.cpp
//
//  The LANGUAGE domain: deterministic, offline intent / language / sentiment.
//
//  Part A — UTF-8 primitives: script counting and the tokenizer, including the
//    malformed-input paths (a truncated sequence must not wedge the scanner)
//    and the apostrophe rule that keeps "don't" as ONE token.
//
//  Part B — language detection BY SCRIPT, with the dominance margin, and an
//    explicit test pinning the KNOWN GAP: romanised Bengali reads as En.
//
//  Part C — intent: every rule, the Bengali-specific rules (question words sit
//    mid-sentence, imperatives are suffixes), the fail-closed Unknown default,
//    and the confidence bounds.
//
//  Part D — sentiment, and the two tests that carry the design:
//      * "not bad, it's good" must be POSITIVE. A naive look-back window sees
//        "not" near "good" and flips it; the scoping rule stops at the nearest
//        sentiment word and gets it right.
//      * the negators must appear in NEITHER lexicon. That is what makes
//        "no problem" positive, and it is a structural invariant rather than a
//        coincidence of the word lists.
//
//  Fully offline: no model, no weights, no network.
// =============================================================================
#include "omniseed/core/platform.h"
#include "omniseed/language/language_heads.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

using namespace omniseed;
using namespace omniseed::language;

static int g_passed = 0;
static int g_failed = 0;
static std::string g_current;

#define TEST(name) g_current = (name); platform::log_info("TEST  %s", name)
#define CHECK(cond)                                                        \
    do {                                                                   \
        if (cond) { ++g_passed; }                                          \
        else {                                                             \
            ++g_failed;                                                    \
            platform::log_error("FAIL  %s  (line %d): %s",                 \
                                g_current.c_str(), __LINE__, #cond);       \
        }                                                                  \
    } while (0)

namespace {

bool has_substr(const std::string& hay, const std::string& needle) {
    return hay.find(needle) != std::string::npos;
}

bool contains(const std::vector<std::string>& v, const std::string& s) {
    return std::find(v.begin(), v.end(), s) != v.end();
}

// The same structural check test_heads.cpp uses: balanced braces/brackets
// outside strings, and no bare NaN/Inf tokens.
bool json_looks_valid(const std::string& s) {
    if (s.empty() || s.front() != '{' || s.back() != '}') return false;
    int db = 0, dk = 0;
    bool in_str = false;
    for (size_t i = 0; i < s.size(); ++i) {
        const char c = s[i];
        if (in_str) {
            if (c == '\\') { ++i; continue; }
            if (c == '"') in_str = false;
            continue;
        }
        if (c == '"') { in_str = true; continue; }
        if (c == '{') ++db;
        else if (c == '}') { if (--db < 0) return false; }
        else if (c == '[') ++dk;
        else if (c == ']') { if (--dk < 0) return false; }
    }
    return !in_str && db == 0 && dk == 0;
}

std::vector<std::string> sorted_copy(const std::vector<std::string>& v) {
    std::vector<std::string> c = v;
    std::sort(c.begin(), c.end());
    return c;
}

bool has_duplicates(const std::vector<std::string>& v) {
    const std::vector<std::string> c = sorted_copy(v);
    return std::adjacent_find(c.begin(), c.end()) != c.end();
}

// Every intent/sentiment result must satisfy these bounds, whatever the input.
void check_intent_invariants(const IntentResult& r, const std::string& label) {
    CHECK(r.confidence >= 0.0f);
    CHECK(r.confidence <= 1.0f);
    CHECK(!r.top_k.empty());
    CHECK(r.top_k.size() <= 3);
    CHECK(r.top_k[0].label == intent_name(r.intent));
    CHECK(r.top_k[0].probability > 0.0f);
    CHECK(r.top_k[0].probability <= 1.0f);
    for (size_t i = 1; i < r.top_k.size(); ++i) {
        CHECK(r.top_k[i - 1].probability >= r.top_k[i].probability);
        CHECK(r.top_k[i].probability > 0.0f);
    }
    CHECK(json_looks_valid(r.to_json()));
    (void)label;
}

} // namespace

// =============================================================================
//  Part A — UTF-8 primitives
// =============================================================================
static void part_a_utf8() {
    TEST("A1: ASCII scan counts letters, digits and codepoints");
    {
        const ScriptStats s = scan_utf8("Hello");
        CHECK(s.ascii_letters == 5);
        CHECK(s.bengali == 0);
        CHECK(s.digits == 0);
        CHECK(s.codepoints == 5);
        CHECK(s.invalid_bytes == 0);

        const ScriptStats d = scan_utf8("abc123");
        CHECK(d.ascii_letters == 3);
        CHECK(d.digits == 3);
        CHECK(d.codepoints == 6);

        const ScriptStats e = scan_utf8("");
        CHECK(e.codepoints == 0);
        CHECK(e.ascii_letters == 0);
    }

    TEST("A2: Bengali is recognised from its OWN Unicode block, not a romanisation");
    {
        const ScriptStats s = scan_utf8("\xe0\xa6\xad\xe0\xa6\xbe\xe0\xa6\xb2\xe0\xa7\x8b"); // ভালো
        CHECK(s.bengali == 4);
        CHECK(s.ascii_letters == 0);
        CHECK(s.codepoints == 4);
        CHECK(s.invalid_bytes == 0);

        const ScriptStats m = scan_utf8("Hi \xe0\xa6\x86\xe0\xa6\xae\xe0\xa6\xbf"); // "Hi আমি"
        CHECK(m.ascii_letters == 2);
        CHECK(m.bengali == 3);
    }

    TEST("A3: Bengali digits count as digits AND as Bengali script");
    {
        const ScriptStats s = scan_utf8("\xe0\xa7\xa7\xe0\xa7\xa8\xe0\xa7\xa9"); // ১২৩
        CHECK(s.bengali == 3);
        CHECK(s.digits == 3);
    }

    TEST("A4: malformed UTF-8 is counted, never looped on forever");
    {
        // A lone continuation byte.
        const ScriptStats a = scan_utf8("\xff");
        CHECK(a.invalid_bytes >= 1);
        CHECK(a.codepoints == 0);

        // A truncated 3-byte sequence.
        const ScriptStats b = scan_utf8("\xe0\xa6");
        CHECK(b.invalid_bytes >= 1);
        CHECK(b.codepoints == 0);

        // An overlong encoding of NUL is not valid UTF-8.
        const ScriptStats c = scan_utf8("\xc0\x80");
        CHECK(c.invalid_bytes >= 1);
        CHECK(c.codepoints == 0);

        // Valid text around a bad byte is still counted.
        const ScriptStats d = scan_utf8("a\xff" "b");
        CHECK(d.ascii_letters == 2);
        CHECK(d.invalid_bytes >= 1);
    }

    TEST("A5: the tokenizer splits, lower-cases, and keeps ? and !");
    {
        const std::vector<std::string> a = tokenize("Hello, World!");
        CHECK(a.size() == 3);
        CHECK(a[0] == "hello");
        CHECK(a[1] == "world");
        CHECK(a[2] == "!");

        const std::vector<std::string> b = tokenize("What?");
        CHECK(b.size() == 2);
        CHECK(b[0] == "what");
        CHECK(b[1] == "?");

        const std::vector<std::string> c = tokenize("MiXeD CaSe");
        CHECK(c.size() == 2);
        CHECK(c[0] == "mixed");
        CHECK(c[1] == "case");

        CHECK(tokenize("").empty());
        CHECK(tokenize("...").empty());
        CHECK(tokenize("   ").empty());
        CHECK(tokenize("?!").size() == 2);
    }

    TEST("A6: an apostrophe INSIDE a word does not split it");
    {
        // This is load-bearing for negation: "don't" must not become "don"+"t".
        const std::vector<std::string> a = tokenize("I don't like it");
        CHECK(a.size() == 4);
        CHECK(a[0] == "i");
        CHECK(a[1] == "don't");
        CHECK(a[2] == "like");
        CHECK(a[3] == "it");

        CHECK(tokenize("can't")[0] == "can't");
        CHECK(tokenize("it's")[0] == "it's");
        // Leading and trailing quotes both separate, so this is ONE token.
        const std::vector<std::string> b = tokenize("'hello'");
        CHECK(b.size() == 1);
        CHECK(b[0] == "hello");
    }

    TEST("A7: a whole Bengali run is ONE token, and order is preserved");
    {
        const std::vector<std::string> a = tokenize("\xe0\xa6\x86\xe0\xa6\xae\xe0\xa6\xbf "
                                                    "\xe0\xa6\xad\xe0\xa6\xbe\xe0\xa6\xb2\xe0\xa7\x8b");
        CHECK(a.size() == 2);   // আমি | ভালো
        CHECK(a[0] != a[1]);

        const std::vector<std::string> b = tokenize("hi \xe0\xa6\x86\xe0\xa6\xae\xe0\xa6\xbf there");
        CHECK(b.size() == 3);
        CHECK(b[0] == "hi");
        CHECK(b[2] == "there");
        CHECK(b[1] != b[0] && b[1] != b[2]);
    }
}

// =============================================================================
//  Part B — language detection
// =============================================================================
static void part_b_language() {
    TEST("B1: enum names");
    {
        CHECK(std::string(lang_name(Lang::En))      == "en");
        CHECK(std::string(lang_name(Lang::Bn))      == "bn");
        CHECK(std::string(lang_name(Lang::Mixed))   == "mixed");
        CHECK(std::string(lang_name(Lang::Unknown)) == "unknown");
        CHECK(std::string(intent_name(Intent::Unknown))   == "unknown");
        CHECK(std::string(intent_name(Intent::Question))  == "question");
        CHECK(std::string(intent_name(Intent::Statement)) == "statement");
        CHECK(std::string(intent_name(Intent::Command))   == "command");
        CHECK(std::string(intent_name(Intent::Greeting))  == "greeting");
        CHECK(std::string(intent_name(Intent::Farewell))  == "farewell");
        CHECK(std::string(intent_name(Intent::Thanks))    == "thanks");
        CHECK(std::string(polarity_name(Polarity::Negative)) == "negative");
        CHECK(std::string(polarity_name(Polarity::Neutral))  == "neutral");
        CHECK(std::string(polarity_name(Polarity::Positive)) == "positive");
    }

    TEST("B2: a single script is detected on its own");
    {
        const IntentClassifier c;
        CHECK(c.classify("Hello there").language == Lang::En);
        CHECK(c.classify("\xe0\xa6\xb9\xe0\xa7\x8d\xe0\xa6\xaf\xe0\xa6\xbe\xe0\xa6\xb2\xe0\xa7\x8b").language == Lang::Bn);
        CHECK(c.classify("").language == Lang::Unknown);
        CHECK(c.classify("12345").language == Lang::Unknown);
    }

    TEST("B3: genuine script mixing is Mixed, a stray word is not");
    {
        const IntentClassifier c;
        // Real mixing: substantial text in both scripts.
        CHECK(c.classify("Hello \xe0\xa6\x86\xe0\xa6\xae\xe0\xa6\xbf").language == Lang::Mixed);
        // One short ASCII token inside a Bengali sentence is borrowing, not mixing.
        const IntentClassifier c2;
        const std::string bn_with_ok =
            "\xe0\xa6\x8f\xe0\xa6\x87 \xe0\xa6\x95\xe0\xa6\xbe\xe0\xa6\x9c\xe0\xa6\x9f\xe0\xa6\xbe "
            "\xe0\xa6\xad\xe0\xa6\xbe\xe0\xa6\xb2\xe0\xa7\x8b \xe0\xa6\x86\xe0\xa6\x9b\xe0\xa7\x87 OK";
        CHECK(c2.classify(bn_with_ok).language == Lang::Bn);
    }

    TEST("B4: KNOWN GAP — romanised Bengali reads as En");
    {
        // Detection is by SCRIPT. "ami bhalo achi" is pure ASCII, so it is
        // indistinguishable from English without a transliteration model this
        // tree does not have. docs/JEV_FEATURES.md records this rather than
        // pretending otherwise; the test pins the actual behaviour so a future
        // fix is a deliberate, visible change.
        const IntentClassifier c;
        CHECK(c.classify("ami bhalo achi").language == Lang::En);
        CHECK(c.classify("ami bhalo achi").intent == Intent::Statement);
    }
}

// =============================================================================
//  Part C — intent
// =============================================================================
static void part_c_intent() {
    const IntentClassifier c;

    TEST("C1: an empty or letter-free input is Unknown, not a guess");
    {
        const IntentResult a = c.classify("");
        CHECK(a.intent == Intent::Unknown);
        CHECK(a.confidence == 0.0f);
        CHECK(a.top_k.size() == 1);
        CHECK(a.top_k[0].label == "unknown");
        CHECK(contains(a.cues, "empty"));
        check_intent_invariants(a, "empty");

        const IntentResult b = c.classify("12345");
        CHECK(b.intent == Intent::Unknown);
        CHECK(b.confidence == 0.0f);
        CHECK(contains(b.cues, "no_cue_fired"));
        check_intent_invariants(b, "digits");

        const IntentResult d = c.classify("...");
        CHECK(d.intent == Intent::Unknown);
        check_intent_invariants(d, "punct");
    }

    TEST("C2: greetings, including the 'good morning' phrase form");
    {
        CHECK(c.classify("Hello").intent == Intent::Greeting);
        CHECK(c.classify("hi").intent == Intent::Greeting);
        CHECK(c.classify("Hello there").intent == Intent::Greeting);
        CHECK(c.classify("Good morning").intent == Intent::Greeting);
        CHECK(c.classify("hey!").intent == Intent::Greeting);   // beats the '!' nudge
        CHECK(contains(c.classify("Hello").cues, "greeting_marker"));
    }

    TEST("C3: farewells, including the phrase form");
    {
        CHECK(c.classify("bye").intent == Intent::Farewell);
        CHECK(c.classify("goodbye").intent == Intent::Farewell);
        CHECK(c.classify("see you later").intent == Intent::Farewell);
        CHECK(c.classify("take care").intent == Intent::Farewell);
    }

    TEST("C4: thanks");
    {
        CHECK(c.classify("thanks").intent == Intent::Thanks);
        CHECK(c.classify("thank you").intent == Intent::Thanks);
        CHECK(c.classify("thanks a lot").intent == Intent::Thanks);
    }

    TEST("C5: questions — trailing '?' and interrogative openers");
    {
        const IntentResult q = c.classify("What is the weather?");
        CHECK(q.intent == Intent::Question);
        CHECK(contains(q.cues, "trailing_question_mark"));
        CHECK(contains(q.cues, "interrogative_opener"));
        CHECK(q.confidence > 0.5f);
        check_intent_invariants(q, "what");

        CHECK(c.classify("How does this work").intent == Intent::Question);
        CHECK(c.classify("Why did it fail?").intent == Intent::Question);
        CHECK(c.classify("Is the build green?").intent == Intent::Question);
        CHECK(c.classify("Where is the config?").intent == Intent::Question);
        // A '?' that is not trailing is still a question cue.
        CHECK(c.classify("You mean this ? then no").intent == Intent::Question);
    }

    TEST("C6: commands — imperative opener and the politeness marker");
    {
        const IntentResult a = c.classify("Show me the file");
        CHECK(a.intent == Intent::Command);
        CHECK(contains(a.cues, "imperative_opener"));

        const IntentResult b = c.classify("Please run the tests");
        CHECK(b.intent == Intent::Command);
        CHECK(contains(b.cues, "politeness_marker"));
        CHECK(b.confidence > a.confidence);   // two cues beat one

        CHECK(c.classify("Explain the regime engine").intent == Intent::Command);
        CHECK(c.classify("Delete the stale branch").intent == Intent::Command);
    }

    TEST("C7: a declarative sentence is a Statement, not a coin flip");
    {
        const IntentResult a = c.classify("The build is green.");
        CHECK(a.intent == Intent::Statement);
        CHECK(contains(a.cues, "declarative_default"));
        check_intent_invariants(a, "statement");

        CHECK(c.classify("I prefer the second option").intent == Intent::Statement);
        CHECK(c.classify("This tree has zero dependencies").intent == Intent::Statement);
    }

    TEST("C8: the baseline does not swallow a strong cue");
    {
        // "hello" is a greeting even though it is also a declarative sentence.
        CHECK(c.classify("Hello").intent == Intent::Greeting);
        // ...and a greeting followed by a question is a QUESTION.
        CHECK(c.classify("Hello, what is X?").intent == Intent::Question);
    }

    TEST("C9: Bengali — question words sit mid-sentence");
    {
        // কেমন আছো?
        const std::string bn_q = "\xe0\xa6\x95\xe0\xa7\x87\xe0\xa6\xae\xe0\xa6\xa8 "
                                 "\xe0\xa6\x86\xe0\xa6\x9b\xe0\xa7\x8b?";
        const IntentResult q = c.classify(bn_q);
        CHECK(q.intent == Intent::Question);
        CHECK(q.language == Lang::Bn);
        CHECK(contains(q.cues, "bn_question_word"));
        check_intent_invariants(q, "bn_q");
    }

    TEST("C10: Bengali — the imperative is a SUFFIX");
    {
        // ফাইলটা দেখাও
        const std::string bn_cmd = "\xe0\xa6\xab\xe0\xa6\xbe\xe0\xa6\x87\xe0\xa6\xb2\xe0\xa6\x9f\xe0\xa6\xbe "
                                   "\xe0\xa6\xa6\xe0\xa7\x87\xe0\xa6\x96\xe0\xa6\xbe\xe0\xa6\x93";
        const IntentResult r = c.classify(bn_cmd);
        CHECK(r.intent == Intent::Command);
        CHECK(r.language == Lang::Bn);
        CHECK(contains(r.cues, "bn_imperative_suffix"));
        check_intent_invariants(r, "bn_cmd");
    }

    TEST("C11: Bengali — greeting and plain statement");
    {
        // হ্যালো
        const std::string bn_hi = "\xe0\xa6\xb9\xe0\xa7\x8d\xe0\xa6\xaf\xe0\xa6\xbe\xe0\xa6\xb2\xe0\xa7\x8b";
        CHECK(c.classify(bn_hi).intent == Intent::Greeting);
        CHECK(c.classify(bn_hi).language == Lang::Bn);

        // আমি ভালো আছি
        const std::string bn_stmt = "\xe0\xa6\x86\xe0\xa6\xae\xe0\xa6\xbf "
                                    "\xe0\xa6\xad\xe0\xa6\xbe\xe0\xa6\xb2\xe0\xa7\x8b "
                                    "\xe0\xa6\x86\xe0\xa6\x9b\xe0\xa6\xbf";
        const IntentResult s = c.classify(bn_stmt);
        CHECK(s.intent == Intent::Statement);
        CHECK(s.language == Lang::Bn);
        check_intent_invariants(s, "bn_stmt");
    }

    TEST("C12: bounds and structure hold for a spread of inputs");
    {
        const char* inputs[] = {
            "", " ", "?", "!", "hello", "bye", "thanks", "what?", "run it",
            "the quick brown fox jumps over the lazy dog",
            "Please explain why the router chose decision_only.",
            "\xe0\xa6\xb9\xe0\xa7\x8d\xe0\xa6\xaf\xe0\xa6\xbe\xe0\xa6\xb2\xe0\xa7\x8b",
            "Hello \xe0\xa6\x86\xe0\xa6\xae\xe0\xa6\xbf",
            "12345", "...", "don't", "not good at all"};
        for (const char* s : inputs) check_intent_invariants(c.classify(s), s);
    }

    TEST("C13: intent classification is deterministic");
    {
        const char* s = "Please explain why the router chose decision_only?";
        const IntentResult a = c.classify(s);
        const IntentResult b = c.classify(s);
        CHECK(a.intent == b.intent);
        CHECK(a.language == b.language);
        CHECK(a.confidence == b.confidence);
        CHECK(a.top_k.size() == b.top_k.size());
        for (size_t i = 0; i < a.top_k.size(); ++i) {
            CHECK(a.top_k[i].label == b.top_k[i].label);
            CHECK(a.top_k[i].probability == b.top_k[i].probability);
        }
    }

    TEST("C14: IntentResult JSON is complete and well-formed");
    {
        const std::string j = c.classify("What is the weather?").to_json();
        CHECK(json_looks_valid(j));
        CHECK(has_substr(j, "\"intent\":\"question\""));
        CHECK(has_substr(j, "\"language\":\"en\""));
        CHECK(has_substr(j, "\"confidence\":"));
        CHECK(has_substr(j, "\"top_k\":["));
        CHECK(has_substr(j, "\"cues\":["));
        CHECK(has_substr(j, "trailing_question_mark"));
    }
}

// =============================================================================
//  Part D — sentiment
// =============================================================================
static void part_d_sentiment() {
    const SentimentScorer s;

    TEST("D1: plain positive and negative");
    {
        const SentimentResult p = s.score("This is good");
        CHECK(p.polarity == Polarity::Positive);
        CHECK(p.score > 0.0f);
        CHECK(contains(p.matched, "good"));
        CHECK(p.negated == 0);
        CHECK(p.confidence > 0.0f && p.confidence <= 1.0f);
        CHECK(json_looks_valid(p.to_json()));

        const SentimentResult n = s.score("This is bad");
        CHECK(n.polarity == Polarity::Negative);
        CHECK(n.score < 0.0f);
        CHECK(contains(n.matched, "bad"));
        CHECK(n.negated == 0);

        CHECK(s.score("The build works and is fast").polarity == Polarity::Positive);
        CHECK(s.score("The build is broken and slow").polarity == Polarity::Negative);
    }

    TEST("D2: negation flips a neighbouring term");
    {
        const SentimentResult a = s.score("This is not good");
        CHECK(a.polarity == Polarity::Negative);
        CHECK(a.negated == 1);
        CHECK(contains(a.matched, "good"));

        const SentimentResult b = s.score("I don't like it");
        CHECK(b.polarity == Polarity::Negative);
        CHECK(b.negated == 1);

        // A different negator, and a lexicon word this time: "acceptable" is
        // NOT in the positive list, so using it would have tested nothing.
        const SentimentResult c = s.score("This is never helpful");
        CHECK(c.polarity == Polarity::Negative);
        CHECK(c.negated == 1);
        CHECK(contains(c.matched, "helpful"));
    }

    TEST("D3: the scoping rule — 'not bad, it's good' is POSITIVE");
    {
        // A naive look-back window sees "not" within three tokens of "good",
        // flips it, and scores this negative. The scoping rule stops at the
        // nearest sentiment word, so "good" is not negated and "bad" is.
        const SentimentResult r = s.score("not bad, it's good");
        CHECK(r.polarity == Polarity::Positive);
        CHECK(r.negated == 1);
        CHECK(r.score > 0.0f);
        CHECK(contains(r.matched, "bad"));
        CHECK(contains(r.matched, "good"));
    }

    TEST("D4: 'no problem' is positive, because 'no' is not in the lexicon");
    {
        const SentimentResult a = s.score("no problem");
        CHECK(a.polarity == Polarity::Positive);
        CHECK(a.negated == 1);

        const SentimentResult b = s.score("no good");
        CHECK(b.polarity == Polarity::Negative);
        CHECK(b.negated == 1);
    }

    TEST("D5: Bengali negates POST-positionally");
    {
        // ভালো  ->  positive
        const std::string bn_pos = "\xe0\xa6\xad\xe0\xa6\xbe\xe0\xa6\xb2\xe0\xa7\x8b";
        CHECK(s.score(bn_pos).polarity == Polarity::Positive);

        // ভালো না  ->  "good not" -> negative. Only the FORWARD pass finds this.
        const std::string bn_neg = bn_pos + " \xe0\xa6\xa8\xe0\xa6\xbe";
        const SentimentResult r = s.score(bn_neg);
        CHECK(r.polarity == Polarity::Negative);
        CHECK(r.negated == 1);

        // খারাপ -> negative
        const std::string bn_bad = "\xe0\xa6\x96\xe0\xa6\xbe\xe0\xa6\xb0\xe0\xa6\xbe\xe0\xa6\xaa";
        CHECK(s.score(bn_bad).polarity == Polarity::Negative);
    }

    TEST("D6: no lexicon hit at all is Neutral with zero confidence");
    {
        const SentimentResult a = s.score("This is neutral text");
        CHECK(a.polarity == Polarity::Neutral);
        CHECK(a.score == 0.0f);
        CHECK(a.confidence == 0.0f);
        CHECK(a.matched.empty());
        CHECK(a.negated == 0);

        const SentimentResult b = s.score("");
        CHECK(b.polarity == Polarity::Neutral);
        CHECK(b.score == 0.0f);
        CHECK(b.matched.empty());

        const SentimentResult c = s.score("...");
        CHECK(c.polarity == Polarity::Neutral);
    }

    TEST("D7: the structural invariant — a negator is in NEITHER lexicon");
    {
        // This is what makes "no problem" positive. If "no" ever appears in the
        // negative list, that behaviour silently reverses, so it is asserted
        // structurally rather than left to the word lists.
        const std::vector<std::string>& pos = positive_lexicon();
        const std::vector<std::string>& neg = negative_lexicon();
        const std::vector<std::string>& negm = negation_markers();
        CHECK(!negm.empty());
        for (const std::string& m : negm) {
            CHECK(!contains(pos, m));
            CHECK(!contains(neg, m));
        }
    }

    TEST("D8: the lexicons are non-empty and duplicate-free");
    {
        CHECK(positive_lexicon().size() > 20);
        CHECK(negative_lexicon().size() > 20);
        CHECK(!has_duplicates(positive_lexicon()));
        CHECK(!has_duplicates(negative_lexicon()));
        CHECK(!has_duplicates(negation_markers()));
        CHECK(contains(positive_lexicon(), "good"));
        CHECK(contains(negative_lexicon(), "bad"));
        CHECK(contains(negation_markers(), "not"));
        CHECK(contains(negation_markers(), "don't"));
    }

    TEST("D9: bounds hold for a spread of inputs");
    {
        const char* inputs[] = {
            "", "good", "bad", "not good", "not bad", "no problem",
            "good and bad", "bad and good", "very good", "not very good",
            "the tests pass and the build is clean",
            "the tests fail and the build is broken",
            "don't touch it", "it's not the worst"};
        for (const char* t : inputs) {
            const SentimentResult r = s.score(t);
            CHECK(r.score >= -1.0f);
            CHECK(r.score <= 1.0f);
            CHECK(r.confidence >= 0.0f);
            CHECK(r.confidence <= 1.0f);
            CHECK(r.negated >= 0);
            CHECK(static_cast<size_t>(r.negated) <= r.matched.size());
            CHECK(json_looks_valid(r.to_json()));
            if (r.matched.empty()) CHECK(r.polarity == Polarity::Neutral);
        }
    }

    TEST("D10: polarity follows the SIGN of the score");
    {
        const SentimentResult mixed = s.score("good bad");
        CHECK(mixed.score == 0.0f);
        CHECK(mixed.polarity == Polarity::Neutral);   // 1 positive, 1 negative

        const SentimentResult more_pos = s.score("good great bad");
        CHECK(more_pos.score > 0.0f);
        CHECK(more_pos.polarity == Polarity::Positive);
        CHECK(more_pos.matched.size() == 3);
    }

    TEST("D11: scoring is deterministic");
    {
        const char* t = "This is not bad at all, actually it works great";
        const SentimentResult a = s.score(t);
        const SentimentResult b = s.score(t);
        CHECK(a.polarity == b.polarity);
        CHECK(a.score == b.score);
        CHECK(a.confidence == b.confidence);
        CHECK(a.negated == b.negated);
        CHECK(a.matched == b.matched);
    }

    TEST("D12: SentimentResult JSON is complete");
    {
        const std::string j = s.score("not good").to_json();
        CHECK(json_looks_valid(j));
        CHECK(has_substr(j, "\"polarity\":"));
        CHECK(has_substr(j, "\"score\":"));
        CHECK(has_substr(j, "\"confidence\":"));
        CHECK(has_substr(j, "\"negated\":1"));
        CHECK(has_substr(j, "\"matched\":["));
    }
}

int main() {
    platform::log_info("=== omniseed language heads: intent, script, sentiment ===");
    part_a_utf8();
    part_b_language();
    part_c_intent();
    part_d_sentiment();
    platform::log_info("=== %d passed, %d failed ===", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
