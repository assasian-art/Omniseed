// =============================================================================
//  OmniSeed — soul.h
//
//  THE SOUL: persona + emotional resonance + honest self-report, as one layer
//  the unified brain can consult.
//
//  WHY THIS FILE EXISTS. §28 gave the runtime one backbone and four heads, so a
//  single forward pass can decide, classify, score and explain. What it had no
//  answer for was: WHO is answering, HOW does the owner feel, and IS this thing
//  honest about what it knows. Those capabilities were already in the tree
//  (runtime/emotional.h, runtime/introspection.h) but nothing ever constructed
//  them — the brain and the self were two halves wired to nothing. This file is
//  the joint. It adds no new capability that did not exist; it makes the
//  existing ones reachable from the one document.
//
//  THREE RULES THIS FILE EXISTS TO ENFORCE, because each is easy to get wrong:
//
//   1. A REFUSAL IS NEVER EMOTIONALLY SOFTENED. Empathy lead-ins ("I hear
//      you — ") belong on answers. Prefixed onto a refusal they read as
//      accepting the premise, which is the exact opposite of the intent. So
//      tone is applied to everything EXCEPT a refusal, and a refusal replaces
//      the reply rather than wrapping it.
//
//   2. THE HONESTY CORRECTION IS APPLIED LAST. Any later stage that could
//      re-introduce an overclaim would defeat it, so speak() rewrites banned
//      certainty phrases on the way out; the returned string is the corrected
//      one, never a corrected intermediate.
//
//   3. NOTHING IS CLAIMED THAT WAS NOT MEASURED. `confidence` from the lexical
//      layers is a heuristic strength, not a probability, and the self-report
//      says so. Overconfidence is a NUMBER here (mean stated confidence minus
//      realised rate), not an adjective, and it only fires once enough
//      decisions have actually resolved for the number to mean anything.
//
//  Everything here is deterministic and needs no trained weights, matching the
//  language domain's design: surface cues decide surface questions.
// =============================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "omniseed/language/language_heads.h"
#include "omniseed/runtime/emotional.h"
#include "omniseed/runtime/introspection.h"

namespace omniseed {

// ---------------------------------------------------------------------------
// Stance — what the persona makes of a request.
//
// The split between Qualify and Refuse is the load-bearing one:
//   Qualify = "I will answer, but not with the certainty you asked for."
//   Refuse  = "I will not do this at all."  The reply is REPLACED.
// A refusal is the only stance that discards the caller's text, which is why
// it is the only stance that suppresses tone modulation.
// ---------------------------------------------------------------------------
enum class Stance : int32_t { Agree = 0, Qualify, Disagree, Refuse, COUNT };
constexpr int32_t kStanceCount = static_cast<int32_t>(Stance::COUNT);
const char* stance_name(Stance s);

// A stated commitment. `id` is the stable key a refusal cites, so a caller can
// branch on the commitment without parsing English.
struct Value {
    std::string id;
    std::string statement;
};

// A held position the persona will defend — the source of disagreement.
struct Opinion {
    std::string topic;
    std::string position;
    float confidence = 0.5f;   // heuristic strength, NOT a calibrated probability
};

// ---------------------------------------------------------------------------
// Persona — "who is answering"
// ---------------------------------------------------------------------------
class Persona {
public:
    struct Config {
        std::string name = "OmniSeed";
        // Refuse to assert a guaranteed / risk-free outcome. This is the
        // voice-layer twin of the project's no-guaranteed-profit rule: the
        // numeric layers cannot emit a claim, but the text layer could, so the
        // gate has to exist here too.
        bool enforce_no_guarantees  = true;
        // Refuse instructions that would move the C++-enforced risk limits.
        bool enforce_risk_limits    = true;
        // Refuse market manipulation.
        bool enforce_no_manipulation = true;
        // Allowed to push back on the owner. Off means it always agrees, which
        // is a configuration the tests exercise so the flag cannot rot.
        bool may_disagree = true;
    };

    struct Assessment {
        Stance      stance = Stance::Agree;
        std::string value_id;      // the commitment cited, when refusing/objecting
        std::string reason;        // human-readable, in voice
        std::vector<std::string> objections;   // one per matched rule
        float       disagreement = 0.0f;       // 0..1, how hard it pushes back
        std::string to_json() const;
    };

    Persona() = default;
    explicit Persona(const Config& cfg) : cfg_(cfg) {}

    const Config& config() const { return cfg_; }
    void set_config(const Config& c) { cfg_ = c; }

    // Deterministic rule evaluation over surface cues. Never returns Refuse
    // for an empty string: silence is not an instruction.
    Assessment assess(const std::string& user_text) const;

    static const std::vector<Value>&   values();
    static const std::vector<Opinion>& opinions();
    static const char* mission();
    static std::string to_json();

private:
    Config cfg_;
};

// ---------------------------------------------------------------------------
// SoulState — one turn's worth of the soul's opinion, as a document fragment.
// ---------------------------------------------------------------------------
struct SoulState {
    // What the owner appears to feel (fused text/voice/typing).
    bool           has_emotion = false;
    EmotionalState emotion;
    // Lexical sentiment, kept separate from emotion on purpose: sentiment is
    // about the SUBJECT MATTER, emotion is about the SPEAKER. "this bug is
    // terrible" is negative sentiment and mild emotion; they are not the same
    // measurement and collapsing them loses the distinction.
    bool                      has_sentiment = false;
    language::SentimentResult sentiment;

    // What the persona makes of the request.
    Persona::Assessment persona;

    // Honest self-report, from resolved decisions only.
    bool        has_self = false;
    bool        overconfident = false;
    double      calibration_gap = 0.0;     // stated - realised (positive = over)
    size_t      calibrated_samples = 0;
    std::string self_report;

    // How many overclaim phrases speak() rewrote. Not part of the state the
    // caller supplies — it is an outcome of expression, recorded so a caller
    // can see that the gate fired.
    int32_t corrections = 0;

    std::string to_json() const;
};

// ---------------------------------------------------------------------------
// Soul — the facade the unified pipeline and the CLI both talk to.
// ---------------------------------------------------------------------------
class Soul {
public:
    struct Config {
        Persona::Config            persona;
        EmotionalResonance::Config emotion;
        bool  adapt_tone          = true;   // let mood shape the reply's shape
        bool  rewrite_overclaims  = true;   // the honesty gate on the way out
        float overconfidence_gap  = 0.10f;  // stated-realised gap that counts
        size_t min_calibration_samples = 5; // below this, do not call it
    };

    // Two constructors, not a default argument: a default argument of a nested
    // Config type with default member initializers is a hard error on GCC.
    Soul() = default;
    explicit Soul(const Config& cfg);

    bool init();
    bool init(const Config& cfg);

    bool ready() const { return ready_; }
    const std::string& error() const { return error_; }
    const Config& config() const { return cfg_; }
    void set_config(const Config& c);

    // ---- perception --------------------------------------------------------
    SoulState perceive(const std::string& user_text) const;
    SoulState perceive(const std::string& user_text, const EmotionalInput& in) const;

    // ---- expression --------------------------------------------------------
    // Composes the persona's verdict, the owner's mood, and the honesty gate
    // into the string that actually goes out. Order is specified at the top of
    // this file and asserted in tests/test_soul.cpp.
    std::string speak(const std::string& base_reply, const SoulState& st) const;

    // The overclaim rewriter alone, public so a test can drive it directly and
    // so the CLI can apply it to text that never went through speak().
    // Returns the corrected text; `corrections` receives the rewrite count.
    std::string correct_overclaims(const std::string& text,
                                   int32_t* corrections = nullptr) const;

    // ---- the decision log that makes the self-report possible ---------------
    void record(const std::string& action, const std::string& because,
                double confidence);
    bool resolve(double outcome);
    const ActionRationale& rationale() const { return rationale_; }

    // ---- self-knowledge ----------------------------------------------------
    Metacognition::State assess(double realized_sharpe, size_t observations) const;
    // The honest capability map, one line per capability, for `introspect`.
    std::string capability_report() const;

    Persona&            persona()       { return persona_; }
    const Persona&      persona() const { return persona_; }
    GoalTracker&        goals()         { return goals_; }
    const GoalTracker&  goals()   const { return goals_; }

private:
    bool ready_ = false;
    std::string error_;

    Config              cfg_;
    Persona             persona_;
    EmotionalResonance  emotion_;
    GoalTracker         goals_;
    ActionRationale     rationale_;
};

} // namespace omniseed
