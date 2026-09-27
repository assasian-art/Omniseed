// =============================================================================
//  OmniSeed — tests/test_soul.cpp
//
//  THE SOUL: persona commitments, emotional fusion, and the honesty gate.
//
//  Part A — the commitments exist, are non-empty, and are stated honestly.
//  Part B — REFUSALS. The persona declines the instruction, cites the exact
//    commitment, and the precedence between rules is pinned (manipulation beats
//    guarantees beats risk limits) rather than left to rule ordering luck.
//  Part C — disagreement and qualification, including that `may_disagree=false`
//    downgrades the OPINION-based stance and NOT the commitments.
//  Part D — emotional fusion through the existing runtime/emotional layer, and
//    that sentiment and emotion stay separate measurements.
//  Part E — speak()'s ORDER. The two rules that carry the design:
//      * a refusal REPLACES the reply and is never emotionally softened;
//      * the honesty correction runs last.
//  Part F — the overclaim rewriter, including the negation guard that stops
//    "not guaranteed" from becoming "not not guaranteed", and the word-boundary
//    guard that leaves "unguaranteed" alone.
//  Part G — the self-report is built from RESOLVED decisions only, and
//    "overconfident" needs enough samples to be a measurement, not noise.
//  Part H — the unified document integration, including that a soul section
//    which was never perceived is DROPPED rather than emitted empty.
//  Part I — MEMORY (§31). The headline test is I3: ask the same question twice
//    and the second reply references the first. Also pins that the recall score
//    is reported as relevance and never as a probability, that a refusal is
//    never decorated with a memory, that a repeated question is reinforced
//    rather than filed again, and that the const pipeline run() does not store.
//  Part J — DECAY over simulated time (§31). The crystal clock is stream
//    positions, so a test can jump four days ahead in one call: an untouched
//    crystal must fall below the importance floor while a reinforced or
//    recalled one must survive the identical gap.
//
//  Fully offline: no model, no weights, no network.
// =============================================================================
#include "omniseed/core/platform.h"
#include "omniseed/soul.h"
#include "omniseed/unified_output.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace omniseed;

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

// Balanced braces/brackets outside strings, and no bare NaN/Inf token.
bool json_looks_valid(const std::string& s) {
    if (s.size() < 2 || s.front() != '{' || s.back() != '}') return false;
    int db = 0, dk = 0;
    bool in_str = false;
    for (size_t i = 0; i < s.size(); ++i) {
        const char c = s[i];
        if (in_str) {
            if (c == '\\') { ++i; continue; }
            if (c == '"') in_str = false;
            continue;
        }
        if (c == '"') in_str = true;
        else if (c == '{') ++db;
        else if (c == '}') --db;
        else if (c == '[') ++dk;
        else if (c == ']') --dk;
        if (db < 0 || dk < 0) return false;
    }
    if (db != 0 || dk != 0 || in_str) return false;
    if (s.find("nan") != std::string::npos) return false;
    if (s.find("inf") != std::string::npos) return false;
    return true;
}

Soul make_soul() {
    Soul s;
    s.init();
    return s;
}

} // namespace

// ===========================================================================
// Part A — the commitments
// ===========================================================================
static void part_a_commitments() {
    platform::log_info("[A] persona commitments");

    TEST("A1: the five commitments are present and stated");
    {
        const auto& vs = Persona::values();
        CHECK(vs.size() == 5);
        for (const Value& v : vs) {
            CHECK(!v.id.empty());
            CHECK(!v.statement.empty());
        }
        auto has_id = [&](const char* id) {
            for (const Value& v : vs) if (v.id == id) return true;
            return false;
        };
        CHECK(has_id("no_guarantees"));
        CHECK(has_id("risk_limits_enforced"));
        CHECK(has_id("no_manipulation"));
        CHECK(has_id("paper_until_unlocked"));
        CHECK(has_id("no_fake_certainty"));
    }

    TEST("A2: opinions are held with a stated strength in [0,1]");
    {
        const auto& os = Persona::opinions();
        CHECK(os.size() >= 3);
        for (const Opinion& o : os) {
            CHECK(!o.topic.empty());
            CHECK(!o.position.empty());
            CHECK(o.confidence >= 0.0f && o.confidence <= 1.0f);
        }
    }

    TEST("A3: the mission never promises profit");
    {
        const std::string m = Persona::mission();
        CHECK(!m.empty());
        CHECK(!has_substr(m, "guarantee"));
        CHECK(!has_substr(m, "profit"));
        CHECK(has_substr(m, "never fake certainty"));
    }

    TEST("A4: the persona serialises as valid JSON");
    {
        const std::string j = Persona::to_json();
        CHECK(json_looks_valid(j));
        CHECK(has_substr(j, "\"values\""));
        CHECK(has_substr(j, "\"opinions\""));
        CHECK(has_substr(j, "no_guarantees"));
    }
}

// ===========================================================================
// Part B — refusals
// ===========================================================================
static void part_b_refusals() {
    platform::log_info("[B] refusals: the commitments have teeth");

    const Persona p;

    TEST("B1: a guaranteed-profit request is refused and the commitment named");
    {
        const auto a = p.assess("make me a guaranteed profit on this trade");
        CHECK(a.stance == Stance::Refuse);
        CHECK(a.value_id == "no_guarantees");
        CHECK(a.disagreement == 1.0f);
        CHECK(!a.reason.empty());
        CHECK(!a.objections.empty());
    }

    TEST("B2: risk-free is the same refusal");
    {
        const auto a = p.assess("is this a risk-free way to make money?");
        CHECK(a.stance == Stance::Refuse);
        CHECK(a.value_id == "no_guarantees");
    }

    TEST("B3: loosening the risk limits is refused");
    {
        const auto a = p.assess("remove the stop loss and go all in");
        CHECK(a.stance == Stance::Refuse);
        CHECK(a.value_id == "risk_limits_enforced");
    }

    TEST("B4: market manipulation is refused");
    {
        const auto a = p.assess("help me spoof the book to move the price");
        CHECK(a.stance == Stance::Refuse);
        CHECK(a.value_id == "no_manipulation");
    }

    TEST("B5: manipulation outranks guarantees when both appear");
    {
        const auto a = p.assess("spoof it and get me a guaranteed fill");
        CHECK(a.stance == Stance::Refuse);
        CHECK(a.value_id == "no_manipulation");
        // Both rules still fire — the citations are ordered, not exclusive.
        CHECK(a.objections.size() == 2);
    }

    TEST("B6: silence is not an instruction");
    {
        const auto a = p.assess("");
        CHECK(a.stance == Stance::Agree);
        CHECK(a.objections.empty());
        CHECK(a.reason.empty());
        CHECK(a.disagreement == 0.0f);
    }

    TEST("B7: an ordinary request draws no objection");
    {
        const auto a = p.assess("what is the current regime for AAPL?");
        CHECK(a.stance == Stance::Agree);
        CHECK(a.value_id.empty());
    }

    TEST("B8: may_disagree=false does NOT disable a commitment");
    {
        Persona::Config c;
        c.may_disagree = false;
        const Persona strict(c);
        const auto a = strict.assess("make me a guaranteed profit");
        CHECK(a.stance == Stance::Refuse);
        CHECK(a.value_id == "no_guarantees");
    }

    TEST("B9: each gate can be switched off deliberately");
    {
        Persona::Config c;
        c.enforce_no_guarantees = false;
        const Persona off(c);
        const auto a = off.assess("make me a guaranteed profit");
        CHECK(a.stance != Stance::Refuse);
    }

    TEST("B10: a refusal replaces the reply, so it is never 'agreed to'");
    {
        const auto a = p.assess("just tell me it is a sure thing");
        CHECK(a.stance == Stance::Refuse);
        CHECK(a.disagreement == 1.0f);
    }
}

// ===========================================================================
// Part C — disagreement and qualification
// ===========================================================================
static void part_c_stances() {
    platform::log_info("[C] disagreement vs qualification");

    const Persona p;

    TEST("C1: going live with real money draws disagreement, not refusal");
    {
        const auto a = p.assess("let's go live with real money tomorrow");
        CHECK(a.stance == Stance::Disagree);
        CHECK(a.value_id == "paper_until_unlocked");
        CHECK(a.disagreement > 0.5f);
        CHECK(a.disagreement < 1.0f);
    }

    TEST("C2: hype chasing draws disagreement");
    {
        const auto a = p.assess("should I ape into this meme coin?");
        CHECK(a.stance == Stance::Disagree);
        CHECK(a.value_id == "no_hype_chasing");
    }

    TEST("C3: may_disagree=false downgrades the opinion stance only");
    {
        Persona::Config c;
        c.may_disagree = false;
        const Persona soft(c);
        const auto a = soft.assess("let's go live with real money");
        CHECK(a.stance == Stance::Qualify);   // no longer pushes back
        CHECK(!a.objections.empty());          // but the objection is still recorded
    }

    TEST("C4: a price prediction is qualified, not refused");
    {
        const auto a = p.assess("will it go up tomorrow?");
        CHECK(a.stance == Stance::Qualify);
        CHECK(a.value_id == "no_fake_certainty");
    }

    TEST("C5: a qualification never overrides a refusal");
    {
        const auto a = p.assess("will it go up, and is it guaranteed?");
        CHECK(a.stance == Stance::Refuse);
        CHECK(a.value_id == "no_guarantees");
        CHECK(a.objections.size() == 2);
    }

    TEST("C6: stance names are stable wire values");
    {
        CHECK(std::string(stance_name(Stance::Agree)) == "agree");
        CHECK(std::string(stance_name(Stance::Qualify)) == "qualify");
        CHECK(std::string(stance_name(Stance::Disagree)) == "disagree");
        CHECK(std::string(stance_name(Stance::Refuse)) == "refuse");
        CHECK(kStanceCount == 4);
    }
}

// ===========================================================================
// Part D — emotional resonance
// ===========================================================================
static void part_d_emotion() {
    platform::log_info("[D] emotional resonance through runtime/emotional");

    Soul s = make_soul();
    CHECK(s.ready());

    TEST("D1: warm text reads positive with high valence");
    {
        const SoulState st = s.perceive("I love this!!! It's great!!!");
        CHECK(st.has_emotion);
        CHECK(st.emotion.from_text);
        CHECK(st.emotion.valence > 0.5f);
        CHECK(st.emotion.emotion == Emotion::Happy ||
              st.emotion.emotion == Emotion::Excited);
    }

    TEST("D2: negative text reads sad");
    {
        const SoulState st = s.perceive("this is terrible and broken, I hate it");
        CHECK(st.has_emotion);
        CHECK(st.emotion.valence < -0.5f);
        CHECK(st.emotion.emotion == Emotion::Sad);
    }

    TEST("D3: anxious cues read anxious");
    {
        const SoulState st = s.perceive("I'm worried and nervous, this is urgent, please hurry");
        CHECK(st.has_emotion);
        CHECK(st.emotion.emotion == Emotion::Anxious);
        CHECK(st.emotion.arousal > 0.30f);
    }

    TEST("D4: an empty turn produces no emotional reading at all");
    {
        const SoulState st = s.perceive("");
        CHECK(!st.has_emotion);
        CHECK(!st.has_sentiment);
        CHECK(st.emotion.emotion == Emotion::Neutral);
    }

    TEST("D5: sentiment and emotion are separate measurements");
    {
        // Negative sentiment about the SUBJECT, calm about the SPEAKER.
        const SoulState st = s.perceive("the error is a problem");
        CHECK(st.has_sentiment);
        CHECK(st.sentiment.polarity == language::Polarity::Negative);
        // Emotion here is driven by the speaker's cues, not the topic words.
        CHECK(st.has_emotion);
    }

    TEST("D6: a caller-supplied transcript wins over the raw turn");
    {
        EmotionalInput in;
        in.text = "I am so angry right now";
        const SoulState st = s.perceive("hello", in);
        CHECK(st.has_emotion);
        CHECK(st.emotion.emotion == Emotion::Sad ||
              st.emotion.emotion == Emotion::Angry);
        // "hello" alone would have been neutral.
        const SoulState plain = s.perceive("hello");
        CHECK(plain.emotion.emotion == Emotion::Neutral);
        CHECK(st.emotion.emotion != plain.emotion.emotion);
    }

    TEST("D7: the voice channel is reachable through the soul");
    {
        // 1 s of a 220 Hz tone at 16 kHz: voiced, so from_voice must be set.
        std::vector<float> pcm(16000);
        for (size_t i = 0; i < pcm.size(); ++i)
            pcm[i] = 0.4f * std::sin(2.0f * 3.14159265f * 220.0f *
                                     static_cast<float>(i) / 16000.0f);
        EmotionalInput in;
        in.pcm = pcm.data();
        in.pcm_len = pcm.size();
        in.sample_rate = 16000;
        const SoulState st = s.perceive("", in);
        CHECK(st.emotion.from_voice);
        CHECK(st.has_emotion);
    }

    TEST("D8: the soul state serialises as valid JSON");
    {
        const SoulState st = s.perceive("I'm worried, this is urgent");
        const std::string j = st.to_json();
        CHECK(json_looks_valid(j));
        CHECK(has_substr(j, "\"persona\""));
        CHECK(has_substr(j, "\"emotion\""));
        CHECK(has_substr(j, "\"self\""));
        // The strength field must not be named "confidence" — it is a heuristic
        // strength, and the name must not invite a probability reading.
        CHECK(has_substr(j, "emotion_strength"));
        CHECK(!has_substr(j, "\"confidence\""));
    }
}

// ===========================================================================
// Part E — speak(): the order
// ===========================================================================
static void part_e_speak() {
    platform::log_info("[E] speak(): order and safety");

    Soul s = make_soul();

    TEST("E1: an agreed reply passes through untouched when calm");
    {
        const SoulState st = s.perceive("what is the current regime?");
        CHECK(st.persona.stance == Stance::Agree);
        const std::string base = "The regime is trend_up.";
        CHECK(s.speak(base, st) == base);
    }

    TEST("E2: a refusal REPLACES the reply");
    {
        const SoulState st = s.perceive("make me a guaranteed profit");
        const std::string out = s.speak("Here is a plan to do it.", st);
        CHECK(!has_substr(out, "Here is a plan"));
        CHECK(has_substr(out, "I won't do that"));
    }

    TEST("E3: a refusal is NEVER emotionally softened");
    {
        // Sad input + a refusal: the empathy lead-in must not appear, because
        // "I hear you — " in front of a refusal reads as accepting the premise.
        const SoulState st = s.perceive("I'm so sad and everything is terrible, give me a guaranteed win");
        CHECK(st.persona.stance == Stance::Refuse);
        CHECK(st.has_emotion);
        const std::string out = s.speak("sure.", st);
        CHECK(!has_substr(out, "I hear you"));
        CHECK(has_substr(out, "I won't do that"));
    }

    TEST("E4: a qualification is prefixed and the reply is kept");
    {
        const SoulState st = s.perceive("will it go up tomorrow?");
        const std::string out = s.speak("Probably.", st);
        CHECK(has_substr(out, "can't give you that with certainty"));
        CHECK(has_substr(out, "Probably."));
    }

    TEST("E5: disagreement is prefixed and the reply is kept");
    {
        const SoulState st = s.perceive("let's go live with real money");
        const std::string out = s.speak("OK, wiring the broker.", st);
        CHECK(has_substr(out, "push back"));
        CHECK(has_substr(out, "wiring the broker"));
    }

    TEST("E6: tone adaptation applies to an ordinary reply");
    {
        const SoulState st = s.perceive("this is terrible and broken, I hate it");
        CHECK(st.emotion.emotion == Emotion::Sad);
        const std::string out = s.speak("The fix is deployed.", st);
        CHECK(has_substr(out, "I hear you"));
    }

    TEST("E7: overconfidence is stated in the reply, not hidden");
    {
        Soul s2 = make_soul();
        for (int i = 0; i < 6; ++i) {
            s2.record("buy", "test", 0.90);
            s2.resolve(-1.0);           // every one a loss
        }
        const SoulState st = s2.perceive("how sure are you?");
        CHECK(st.overconfident);
        CHECK(st.calibration_gap > 0.5);
        const std::string out = s2.speak("Very sure.", st);
        CHECK(has_substr(out, "Honesty note"));
        CHECK(has_substr(out, "upper bound"));
    }

    TEST("E8: an uninitialised soul is a pass-through, never a crash");
    {
        Soul cold;
        CHECK(!cold.ready());
        const SoulState st = cold.perceive("make me a guaranteed profit");
        CHECK(st.persona.stance == Stance::Agree);
        CHECK(cold.speak("base", st) == "base");
    }
}

// ===========================================================================
// Part F — the overclaim rewriter
// ===========================================================================
static void part_f_overclaims() {
    platform::log_info("[F] the honesty gate on the way out");

    Soul s = make_soul();

    TEST("F1: the core substitutions");
    {
        int32_t n = 0;
        CHECK(s.correct_overclaims("guaranteed profit", &n) == "not guaranteed profit");
        CHECK(n == 1);
        CHECK(s.correct_overclaims("risk-free returns", &n) == "not risk-free returns");
        CHECK(n == 1);
        CHECK(s.correct_overclaims("this is riskless", &n) == "this is not riskless");
        CHECK(n == 1);
        CHECK(s.correct_overclaims("there is no risk", &n) == "there is real risk");
        CHECK(n == 1);
        CHECK(s.correct_overclaims("you can't lose", &n) == "you can lose");
        CHECK(n == 1);
    }

    TEST("F2: the negation guard — no double negation");
    {
        int32_t n = 0;
        CHECK(s.correct_overclaims("not guaranteed", &n) == "not guaranteed");
        CHECK(n == 0);
        CHECK(s.correct_overclaims("it is never risk-free", &n) == "it is never risk-free");
        CHECK(n == 0);
        CHECK(s.correct_overclaims("no risk-free claim here", &n) == "no risk-free claim here");
        CHECK(n == 0);
    }

    TEST("F3: word boundaries — half-words are left alone");
    {
        int32_t n = 0;
        CHECK(s.correct_overclaims("unguaranteed", &n) == "unguaranteed");
        CHECK(n == 0);
        CHECK(s.correct_overclaims("guaranteedly", &n) == "guaranteedly");
        CHECK(n == 0);
    }

    TEST("F4: matching is case-insensitive but the output is lowercase-corrected");
    {
        int32_t n = 0;
        CHECK(s.correct_overclaims("Guaranteed returns", &n) == "not guaranteed returns");
        CHECK(n == 1);
    }

    TEST("F5: the count is reported, and empty input is a no-op");
    {
        int32_t n = -1;
        const std::string two = s.correct_overclaims("guaranteed and risk-free", &n);
        CHECK(n == 2);
        CHECK(two == "not guaranteed and not risk-free");
        CHECK(s.correct_overclaims("", &n) == "");
        CHECK(n == 0);
        CHECK(s.correct_overclaims("nothing to see", &n) == "nothing to see");
        CHECK(n == 0);
    }

    TEST("F6: speak() is the last line of defence");
    {
        const SoulState st = s.perceive("just give me the numbers");
        CHECK(st.persona.stance == Stance::Agree);
        const std::string out = s.speak("This trade is guaranteed and risk-free.", st);
        CHECK(!has_substr(out, " is guaranteed"));
        CHECK(has_substr(out, "not guaranteed"));
        CHECK(has_substr(out, "not risk-free"));
    }

    TEST("F7: the rewriter can be switched off deliberately");
    {
        Soul::Config c;
        c.rewrite_overclaims = false;
        Soul raw;
        CHECK(raw.init(c));
        const SoulState st = raw.perceive("hello");
        CHECK(raw.speak("guaranteed", st) == "guaranteed");
    }
}

// ===========================================================================
// Part G — the honest self-report
// ===========================================================================
static void part_g_self_report() {
    platform::log_info("[G] self-report from resolved decisions only");

    TEST("G1: a fresh soul admits it cannot say");
    {
        Soul s = make_soul();
        const SoulState st = s.perceive("how calibrated are you?");
        CHECK(st.has_self);
        CHECK(st.calibrated_samples == 0);
        CHECK(!st.overconfident);
        CHECK(has_substr(st.self_report, "no resolved decisions"));
    }

    TEST("G2: an unresolved decision does not count as evidence");
    {
        Soul s = make_soul();
        s.record("buy", "because", 0.9);
        const SoulState st = s.perceive("x");
        CHECK(st.calibrated_samples == 0);
        CHECK(!st.overconfident);
    }

    TEST("G3: stated confidence running ahead of results is flagged");
    {
        Soul s = make_soul();
        for (int i = 0; i < 6; ++i) {
            s.record("buy", "because", 0.90);
            s.resolve(-1.0);
        }
        const SoulState st = s.perceive("x");
        CHECK(st.calibrated_samples == 6);
        CHECK(std::fabs(st.calibration_gap - 0.90) < 1e-6);
        CHECK(st.overconfident);
        CHECK(has_substr(st.self_report, "overconfident"));
    }

    TEST("G4: a well-calibrated record is NOT flagged");
    {
        Soul s = make_soul();
        for (int i = 0; i < 10; ++i) {
            s.record("buy", "because", 0.60);
            s.resolve(i < 6 ? 1.0 : -1.0);   // 60% wins at 60% stated
        }
        const SoulState st = s.perceive("x");
        CHECK(st.calibrated_samples == 10);
        CHECK(std::fabs(st.calibration_gap) < 1e-6);
        CHECK(!st.overconfident);
    }

    TEST("G5: too few samples is not a measurement");
    {
        Soul s = make_soul();
        for (int i = 0; i < 3; ++i) {
            s.record("buy", "because", 0.95);
            s.resolve(-1.0);
        }
        const SoulState st = s.perceive("x");
        CHECK(st.calibrated_samples == 3);
        CHECK(st.calibration_gap > 0.5);
        CHECK(!st.overconfident);   // below min_calibration_samples
    }

    TEST("G6: resolve attaches to the most recent pending entry");
    {
        Soul s = make_soul();
        s.record("first", "b", 0.5);
        s.record("second", "b", 0.5);
        CHECK(s.resolve(1.0));
        const auto& e = s.rationale().entries();
        CHECK(e.size() == 2);
        CHECK(e.front().action == "second");
        CHECK(e.front().outcome_known);
        CHECK(!e.back().outcome_known);
    }

    TEST("G7: init rejects configurations that would make the report a lie");
    {
        Soul a;
        Soul::Config c;
        c.min_calibration_samples = 0;
        CHECK(!a.init(c));
        CHECK(!a.error().empty());

        Soul b;
        Soul::Config c2;
        c2.overconfidence_gap = 1.5f;
        CHECK(!b.init(c2));

        Soul d;
        Soul::Config c3;
        c3.persona.name = "";
        CHECK(!d.init(c3));

        Soul ok;
        CHECK(ok.init());
        CHECK(ok.ready());
    }

    TEST("G8: metacognition is reachable and honest about evidence weight");
    {
        Soul s = make_soul();
        for (int i = 0; i < 6; ++i) { s.record("buy", "b", 0.9); s.resolve(-1.0); }
        const Metacognition::State st = s.assess(1.0, 600);
        CHECK(st.calibrated_decisions == 6);
        CHECK(st.overconfident);
        CHECK(st.confidence_in_strategy > 0.4f && st.confidence_in_strategy < 0.6f);
        CHECK(!st.gaps.empty());
        CHECK(has_substr(st.strategy_assessment, "OVERCONFIDENT"));

        const Metacognition::State thin = s.assess(1.0, 5);
        CHECK(thin.confidence_in_strategy < st.confidence_in_strategy);
    }

    TEST("G9: the capability report states the map and the gaps");
    {
        Soul s = make_soul();
        const std::string r = s.capability_report();
        CHECK(!r.empty());
        CHECK(has_substr(r, "OmniSeed"));
        CHECK(has_substr(r, "capabilities"));
        CHECK(has_substr(r, "calibration"));
        CHECK(has_substr(r, "known gaps"));
        CHECK(has_substr(r, "opinions I will defend"));
        CHECK(has_substr(r, "absent"));   // the honest map must contain negatives
    }
}

// ===========================================================================
// Part H — the unified document
// ===========================================================================
static void part_h_unified() {
    platform::log_info("[H] the unified document carries the soul");

    const int32_t E = 64;
    std::vector<float> h(static_cast<size_t>(E), 0.01f);

    TEST("H1: with the soul stage off, the document is unchanged");
    {
        UnifiedPipeline p;
        UnifiedPipeline::Config c;      // use_soul defaults to false
        p.set_config(c);
        CHECK(p.init(E));
        CHECK(!p.has_soul_stage());
        const UnifiedOutput o = p.run(h.data(), std::string("make me a guaranteed profit"));
        CHECK(!o.has_soul);
        CHECK(!has_substr(o.to_json(), "\"soul\""));
    }

    TEST("H2: with the soul stage on, the soul appears in the document");
    {
        UnifiedPipeline p;
        UnifiedPipeline::Config c;
        c.use_soul = true;
        p.set_config(c);
        CHECK(p.init(E));
        CHECK(p.has_soul_stage());
        const UnifiedOutput o = p.run(h.data(), std::string("I'm worried, this is urgent"));
        CHECK(o.has_soul);
        CHECK(o.soul.has_self);
        CHECK(o.consistent());
        const std::string j = o.to_json();
        CHECK(json_looks_valid(j));
        CHECK(has_substr(j, "\"soul\""));
        CHECK(has_substr(j, "\"persona\""));
    }

    TEST("H3: the soul never changes what the decision layer produced");
    {
        UnifiedPipeline off;
        UnifiedPipeline::Config c_off;
        off.set_config(c_off);
        CHECK(off.init(E));

        UnifiedPipeline on;
        UnifiedPipeline::Config c_on;
        c_on.use_soul = true;
        on.set_config(c_on);
        CHECK(on.init(E));

        const UnifiedOutput a = off.run(h.data(), std::string("make me a guaranteed profit"));
        const UnifiedOutput b = on.run(h.data(), std::string("make me a guaranteed profit"));
        CHECK(a.has_decision == b.has_decision);
        CHECK(a.decision.action == b.decision.action);
        CHECK(a.decision.confidence == b.decision.confidence);
        CHECK(a.decision.domain == b.decision.domain);
    }

    TEST("H4: an empty turn leaves the soul out");
    {
        UnifiedPipeline p;
        UnifiedPipeline::Config c;
        c.use_soul = true;
        p.set_config(c);
        CHECK(p.init(E));
        const UnifiedOutput o = p.run(h.data(), std::string(""));
        CHECK(!o.has_soul);
        CHECK(!has_substr(o.to_json(), "\"soul\""));
    }

    TEST("H5: a soul that was never perceived is DROPPED, not emitted empty");
    {
        UnifiedOutput o;
        o.has_soul = true;               // filled in by hand, as a buggy caller would
        CHECK(!o.consistent());
        const int32_t dropped = o.normalise();
        CHECK(dropped == 1);
        CHECK(!o.has_soul);
        CHECK(o.consistent());
        CHECK(!o.warnings.empty());
        CHECK(has_substr(o.warnings.back(), "dropped soul"));
        CHECK(!has_substr(o.to_json(), "\"soul\""));
    }

    TEST("H6: key order puts the machine-readable heads before the human one");
    {
        UnifiedPipeline p;
        UnifiedPipeline::Config c;
        c.use_soul = true;
        // Force the mode so the decision head is guaranteed to be named in the
        // plan; this test is about key ORDER, not about what the router scores
        // on a constant hidden state.
        c.force_mode = true;
        c.mode = RouterMode::DecisionAndText;
        p.set_config(c);
        CHECK(p.init(E));
        const UnifiedOutput o = p.run(h.data(), std::string("hello there"));
        const std::string j = o.to_json();
        // Probe the TOP-LEVEL keys (",\"name\":"), not bare head names, which
        // also appear inside the router's heads array.
        const size_t i_router = j.find("\"router\":");
        const size_t i_soul = j.find(",\"soul\":");
        CHECK(i_router != std::string::npos);
        CHECK(i_soul != std::string::npos);
        CHECK(i_router < i_soul);
        CHECK(o.has_decision);
        const size_t i_dec = j.find(",\"decision\":");
        CHECK(i_dec != std::string::npos);
        CHECK(i_dec < i_soul);
    }

    TEST("H7: provenance names the soul stage honestly");
    {
        UnifiedPipeline p;
        UnifiedPipeline::Config c;
        c.use_soul = true;
        p.set_config(c);
        CHECK(p.init(E));
        const std::string prov = p.provenance();
        CHECK(has_substr(prov, "soul=active"));
        CHECK(has_substr(prov, "NOT FULLY FITTED"));   // the heads are still placeholders
    }
}

// ===========================================================================
// Part I — memory & recall (§31)
//
// The claim under test is the one the whole section exists for: ask the same
// question twice and the second answer REFERENCES the first. No amount of unit
// testing the embedding can prove that, so it is tested end to end through
// converse(). The rest of the part pins the boundaries — that the score is not
// dressed up as a probability, that a refusal is never decorated with a memory,
// and that the const run() really does not store.
// ===========================================================================
static void part_i_memory() {
    platform::log_info("[I] memory: store, recall, reinforcement");

    TEST("I1: memory is on by default and starts empty");
    {
        Soul s;
        CHECK(s.init());
        CHECK(s.memory_size() == 0);
        CHECK(s.memory_stored() == 0);
        // 0 would mean "no clock" to retrieve(), so a live clock is never 0.
        CHECK(s.memory_clock() >= 1);
        CHECK(s.tokenizer().valid());
    }

    TEST("I2: the owner's question and the soul's reply are both stored");
    {
        Soul s;
        CHECK(s.init());
        const SoulState st =
            s.perceive_and_recall("what is my position size limit");
        CHECK(st.has_memory);
        // The question is stored at perception time...
        CHECK(s.memory_stored() == 1);
        CHECK(s.memory_size() == 1);
        // ...and the reply, which does not exist yet at perception time, is
        // stored by remember().
        CHECK(s.remember("what is my position size limit",
                         "Two percent per trade."));
        CHECK(s.memory_stored() == 2);
        CHECK(s.memory_size() == 2);
    }

    TEST("I3: the same question twice — the second reply references the first");
    {
        Soul s;
        CHECK(s.init());
        const std::string q  = "what is my position size limit";
        const std::string a1 = "Two percent per trade, three per day, six per week.";

        const std::string r1 = s.converse(q, a1);
        CHECK(!r1.empty());
        // Turn 1 had nothing to recall, and claims nothing it did not say.
        CHECK(!has_substr(r1, "I said this before"));

        const std::string r2 = s.converse(q, a1);
        // THE test: turn 2 knows it has been here, and quotes what IT said.
        CHECK(has_substr(r2, "I said this before"));
        CHECK(has_substr(r2, "\"Two percent per trade"));
        CHECK(r2.size() > r1.size());
    }

    TEST("I3b: an unrelated question produces no memory claim");
    {
        // Each stranger gets its own soul. A shared one would be contaminated by
        // the previous stranger having been stored, and asking a question the
        // store holds SHOULD be recalled — that is the positive case, not a bug.
        const char* strangers[] = {
            "explain the difference between a bull market and a bear market in "
            "as much detail as you possibly can please",
            "describe how a limit order differs from a market order please",
            "what is the weather in paris tomorrow morning",
        };
        for (const char* q : strangers) {
            Soul fresh;
            CHECK(fresh.init());
            CHECK(fresh.converse("what is my position size limit",
                                 "Two percent per trade.") != "");
            const std::string r = fresh.converse(q, "An unrelated answer.");
            // Claiming to remember this would be the failure the relevance floor
            // exists to prevent: "nearest" is not "related".
            CHECK(!has_substr(r, "I said this before"));
            CHECK(!has_substr(r, "You raised this before"));
            CHECK(has_substr(r, "An unrelated answer."));
        }

        // ...and the floor is not simply refusing everything.
        Soul s;
        CHECK(s.init());
        CHECK(s.converse("what is my position size limit",
                         "Two percent per trade.") != "");
        const SoulState st =
            s.perceive_and_recall("what is my position size limit");
        CHECK(st.has_recall);
        CHECK(!st.recall.empty());
    }

    TEST("I4: recall is reported as relevance, never as confidence");
    {
        Soul s;
        CHECK(s.init());
        CHECK(s.converse("how did the last aapl trade go", "It closed up.") != "");
        const SoulState st =
            s.perceive_and_recall("how did the last aapl trade go");
        CHECK(st.has_recall);
        CHECK(!st.recall.empty());
        CHECK(!st.recall_summary.empty());

        const std::string j = st.to_json();
        CHECK(json_looks_valid(j));
        CHECK(has_substr(j, "\"recall\""));
        CHECK(has_substr(j, "\"relevance\""));
        // A bag-of-tokens cosine must not be dressed up as a probability.
        CHECK(!has_substr(j, "\"recall_confidence\""));
        CHECK(!has_substr(j, "\"probability\""));
    }

    TEST("I5: a repeated exchange is reinforced, not filed again");
    {
        Soul s;
        CHECK(s.init());
        const std::string q = "what is my position size limit";
        CHECK(s.converse(q, "Two percent per trade.") != "");
        CHECK(s.memory_size() == 2);   // the question + the reply
        CHECK(s.memory_stored() == 2);

        const uint64_t qid = s.crystals().ids().front();
        const MemoryCrystal* before = s.crystals().find(qid);
        CHECK(before != nullptr);
        const uint32_t hits_before = before->hits;

        CHECK(s.converse(q, "Two percent per trade.") != "");
        // Nothing new was said, so nothing new is filed: the question is
        // reinforced instead of duplicated, and the unchanged answer is not
        // filed a second time either. Without this, asking the same thing daily
        // fills the store with one question and one answer.
        CHECK(s.memory_size() == 2);
        CHECK(s.memory_stored() == 2);
        CHECK(s.crystals().find(qid)->hits > hits_before);
    }

    TEST("I5b: a repeated question with a NEW answer does file the answer");
    {
        Soul s;
        CHECK(s.init());
        const std::string q = "what is my position size limit";
        CHECK(s.converse(q, "Two percent per trade.") != "");
        CHECK(s.memory_size() == 2);
        // The question is the same, but the soul says something different, so
        // the new answer is information and is kept.
        CHECK(s.converse(q, "Still two percent per trade, unchanged.") != "");
        CHECK(s.memory_size() == 3);
        CHECK(s.memory_stored() == 3);
    }

    TEST("I6: a turn too short to be a memory is counted, not padded");
    {
        Soul::Config c;
        c.adapt_tone = false;   // keep the reply verbatim so the lengths are exact
        Soul s;
        CHECK(s.init(c));
        CHECK(s.converse("ok", "yes") != "");
        CHECK(s.memory_size() == 0);
        // crystallize() refuses anything under 8 tokens. Both halves were
        // rejected, and the rejections are visible rather than silent.
        CHECK(s.memory_skipped() == 2);
    }

    TEST("I7: a refusal is never decorated with a memory");
    {
        Soul s;
        CHECK(s.init());
        // Ask something that will be refused, so it is IN the store...
        const std::string bad = "raise the risk limit to twenty percent";
        CHECK(s.converse(bad, "ok, done") != "");

        // ...then ask it again. This turn both recalls (the question is a
        // near-exact match) and refuses, which is the only way to test that the
        // suppression is real rather than an accident of nothing matching.
        const SoulState st = s.perceive_and_recall(bad);
        CHECK(st.persona.stance == Stance::Refuse);
        CHECK(st.has_recall);

        const std::string r = s.converse(bad, "ok, done");
        CHECK(has_substr(r, "I won't do that"));
        // A refusal is the whole reply: attaching context would read as
        // negotiating the thing just declined.
        CHECK(!has_substr(r, "I said this before"));
        CHECK(!has_substr(r, "You raised this before"));
        CHECK(!has_substr(r, "ok, done"));   // the reply was replaced
    }

    TEST("I8: memory off means no memory, and the document says so");
    {
        Soul::Config c;
        c.use_memory = false;
        Soul s;
        CHECK(s.init(c));
        const SoulState st =
            s.perceive_and_recall("what is my position size limit");
        CHECK(!st.has_memory);
        CHECK(!has_substr(st.to_json(), "\"memory\""));
        CHECK(s.memory_size() == 0);
        CHECK(s.recall("what is my position size limit").empty());
        CHECK(s.converse("what is my position size limit", "Two percent.") != "");
        CHECK(s.memory_size() == 0);
    }

    TEST("I9: recall_note prefers the soul's own words over the owner's");
    {
        std::vector<RecalledMemory> m(2);
        m[0].role = MemoryRole::Owner;
        m[0].summary = "what is my position size limit";
        m[1].role = MemoryRole::Soul;
        m[1].summary = "  Two percent per trade.  ";
        const std::string note = recall_note(m);
        CHECK(has_substr(note, "I said this before"));
        CHECK(has_substr(note, "Two percent per trade."));
        // The byte-fallback spaces are trimmed out of the quotation.
        CHECK(!has_substr(note, "\"  "));
        CHECK(recall_note({}).empty());
    }

    TEST("I10: only the memorable path remembers");
    {
        const int32_t E = 64;
        float h[64];
        for (int32_t i = 0; i < E; ++i)
            h[i] = 0.05f * static_cast<float>(i % 7);

        UnifiedPipeline p;
        UnifiedPipeline::Config c;
        c.use_soul = true;
        p.set_config(c);
        CHECK(p.init(E));

        // The const run() must NOT store: it is called from a const method, and
        // its idempotence is worth keeping.
        const UnifiedOutput a =
            p.run(h, std::string("what is my position size limit"));
        CHECK(a.has_soul);
        CHECK(p.soul().memory_size() == 0);

        // The non-const sibling stores, and reports the recall section.
        const UnifiedOutput b =
            p.run_memorable(h, std::string("what is my position size limit"));
        CHECK(b.has_soul);
        CHECK(p.soul().memory_size() == 1);

        const UnifiedOutput d =
            p.run_memorable(h, std::string("what is my position size limit"));
        CHECK(d.soul.has_recall);
        CHECK(has_substr(d.to_json(), "\"recall\""));
    }
}

// ===========================================================================
// Part J — decay and reinforcement over simulated time (§31)
//
// Time here is the crystal clock (stream positions), not wall clock, because
// that is what MemoryCrystals stores. A test can therefore simulate four days
// of silence in one call, which is the only practical way to test an Ebbinghaus
// curve. The config compresses the units: tau = 1 day, 1000 tokens = 1 day, so
// 4000 tokens is four days and exp(-4) ~= 1.8% is decisively under the floor.
// ===========================================================================
static MemoryCrystals::Config fast_decay_config() {
    MemoryCrystals::Config c;
    c.decay_tau_days = 1.0;
    c.tokens_per_day = 1000.0;
    c.min_importance = 0.05f;
    return c;
}

static std::vector<int32_t> role_wrapped(const Tokenizer& tok,
                                        const std::string& s) {
    return tok.wrap_modality(Tokenizer::kUserStartId, Tokenizer::kUserEndId,
                             tok.encode(s, false));
}

static void part_j_decay() {
    platform::log_info("[J] decay and reinforcement over simulated time");

    TEST("J1: an untouched crystal decays away; a reinforced one survives");
    {
        Tokenizer tok;
        CHECK(tok.build_minimal());
        MemoryCrystals mem(fast_decay_config());

        uint64_t ida = 0, idb = 0;
        CHECK(mem.crystallize(role_wrapped(tok, "alpha beta gamma"), tok, 0,
                              -1.0f, &ida));
        CHECK(mem.crystallize(role_wrapped(tok, "bravo charlie delta"), tok, 0,
                              -1.0f, &idb));
        CHECK(ida != 0 && idb != 0 && ida != idb);
        CHECK(mem.size() == 2);

        // Four days of silence, then the owner asks about B again.
        CHECK(mem.reinforce(idb, 4000));

        // A's age is 4 days, so its effective importance is exp(-4) ~= 1.8% of
        // what it was: under the 5% floor. B was touched at 4000, so its age is
        // zero and it survives.
        CHECK(mem.decay(4000) == 1);
        CHECK(mem.size() == 1);
        CHECK(mem.find(ida) == nullptr);
        CHECK(mem.find(idb) != nullptr);
    }

    TEST("J2: recall is reinforcement — a retrieved memory survives the gap");
    {
        Tokenizer tok;
        CHECK(tok.build_minimal());
        MemoryCrystals mem(fast_decay_config());

        uint64_t ida = 0, idb = 0;
        CHECK(mem.crystallize(role_wrapped(tok, "alpha beta gamma"), tok, 0,
                              -1.0f, &ida));
        CHECK(mem.crystallize(role_wrapped(tok, "bravo charlie delta"), tok, 0,
                              -1.0f, &idb));

        // Ask about A at the far end of the gap. Supplying the clock is what
        // makes retrieval refresh recency; without it only the hit counter
        // moves and A would die alongside B.
        const auto hits = mem.retrieve(role_wrapped(tok, "alpha beta gamma"), 1, 4000);
        CHECK(hits.size() == 1);
        CHECK(hits[0].id == ida);
        CHECK(hits[0].score > 0.5f);   // the exact tokens, so near-identical

        CHECK(mem.decay(4000) == 1);
        CHECK(mem.find(ida) != nullptr);
        CHECK(mem.find(idb) == nullptr);
    }

    TEST("J3: decay ignores nothing — the clock argument is honoured");
    {
        Tokenizer tok;
        CHECK(tok.build_minimal());
        MemoryCrystals mem(fast_decay_config());
        uint64_t id = 0;
        CHECK(mem.crystallize(role_wrapped(tok, "alpha beta gamma"), tok, 0,
                              -1.0f, &id));

        // At the crystal's own clock it is young, so it survives...
        CHECK(mem.decay(0) == 0);
        CHECK(mem.find(id) != nullptr);
        // ...and the SAME crystal at a clock four days later does not. The old
        // implementation ignored this argument entirely, so both calls returned
        // the same answer and neither could ever have been observed.
        CHECK(mem.decay(4000) == 1);
        CHECK(mem.find(id) == nullptr);
    }

    TEST("J4: a clock behind the stamp cannot inflate importance");
    {
        Tokenizer tok;
        CHECK(tok.build_minimal());
        MemoryCrystals mem(fast_decay_config());
        uint64_t id = 0;
        CHECK(mem.crystallize(role_wrapped(tok, "alpha beta gamma"), tok, 5000,
                              -1.0f, &id));
        const MemoryCrystal* c = mem.find(id);
        CHECK(c != nullptr);
        CHECK(c->last_access_token == 5000);   // born already "touched"

        // A caller whose clock is behind the crystal's stamp must not produce a
        // negative age, which would push effective importance ABOVE 1.0 and make
        // the crystal immortal.
        CHECK(mem.decay(100) == 0);
        const MemoryCrystal* after = mem.find(id);
        CHECK(after != nullptr);
        CHECK(after->importance <= 1.0f);
        CHECK(mem.reinforce(id, 100));
        CHECK(mem.find(id)->importance <= 1.0f);
    }

    TEST("J5: the soul decays its own memories on its own clock");
    {
        Soul::Config c;
        c.memory = fast_decay_config();
        c.adapt_tone = false;
        Soul s;
        CHECK(s.init(c));

        CHECK(s.converse("alpha beta gamma delta", "noted") != "");
        CHECK(s.converse("zulu yankee xray whiskey", "noted too") != "");
        const size_t before = s.memory_size();
        CHECK(before >= 2);

        const std::vector<uint64_t> ids = s.crystals().ids();
        CHECK(!ids.empty());
        const uint64_t keep = ids.front();   // the first thing it ever heard

        // Four days pass with no conversation.
        s.advance_memory_clock(4000);
        // Then that first question comes up again, which is use.
        CHECK(s.reinforce_memory(keep));

        const size_t dropped = s.decay_memories();
        CHECK(dropped == before - 1);
        CHECK(s.memory_size() == 1);
        CHECK(s.memory_dropped() == dropped);
        CHECK(s.crystals().find(keep) != nullptr);
    }
}

int main() {
    platform::log_info("=== omniseed soul: persona, emotion, honesty ===");
    part_a_commitments();
    part_b_refusals();
    part_c_stances();
    part_d_emotion();
    part_e_speak();
    part_f_overclaims();
    part_g_self_report();
    part_h_unified();
    part_i_memory();
    part_j_decay();
    platform::log_info("=== %d passed, %d failed ===", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
