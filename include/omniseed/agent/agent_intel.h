// =============================================================================
//  OmniSeed — agent_intel.h
//  Agent intelligence layer (features #6, #16, #17, #23-26, #47, #48, #72,
//  #74 of the 100-feature spec):
//
//    * IntentClassifier        — route queries: informational, directive,
//                                exploratory, social (spec #23).
//    * DialogueStateTracker    — slots + topic + turn count for coherent
//                                multi-turn behavior (spec #24).
//    * ConfidenceScorer        — output confidence + hallucination risk from
//                                state saturation signals (spec #16, #17).
//    * ThinkingMode            — toggleable micro-thinking: /think deep
//                                traces vs direct answers (spec #6).
//    * ErrorRecovery           — retry/backoff/degrade protocol for tool
//                                failures (spec #25).
//    * UserFeedbackLoop        — explicit confirm/correct + implicit signal
//                                capture (spec #47, #48).
//    * SensoryInterruptSystem  — priority routing of async sensory events
//                                (wake word, sound, motion) into the agent
//                                loop (spec #2, interrupt blueprint).
//    * KnowledgeGraph          — compact entity->relation->entity store
//                                mined from crystals + turns; indexed recall
//                                (spec #72, #39).
//    * ProgressiveDisclosure   — summary first, detail on request (spec #26).
// =============================================================================
#pragma once

#include <cstdint>
#include <deque>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace omniseed {

// ===========================================================================
// IntentClassifier (spec #23)
// ===========================================================================
enum class Intent : int32_t {
    Unknown = 0,
    Informational,     // "what is", "who", "explain"
    Directive,         // "do", "run", "call", imperative
    Exploratory,       // "why", "how come", "compare"
    Social,            // greetings, thanks
    TaskRepeat,        // "again", redo
};
const char* intent_name(Intent i);

class IntentClassifier {
public:
    static Intent classify(const std::string& text);
};

// ===========================================================================
// DialogueStateTracker (spec #24)
// ===========================================================================
class DialogueStateTracker {
public:
    void update(const std::string& user_text, const std::string& reply);

    // Topic = highest-salience content token set (hashed signature).
    const std::string& topic() const { return topic_; }
    int32_t turns() const { return turns_; }
    const std::string& last_user() const { return last_user_; }

    // Simple slot filling: "key = value" / "my X is Y" patterns.
    const std::map<std::string, std::string>& slots() const { return slots_; }

    void reset();

private:
    std::string topic_;
    std::string last_user_;
    int32_t turns_ = 0;
    std::map<std::string, std::string> slots_;
};

// ===========================================================================
// ConfidenceScorer (spec #16) + hallucination risk (spec #17)
// ===========================================================================
struct ConfidenceReport {
    float confidence = 0.0f;         // 0..1
    float hallucination_risk = 0.0f; // 0..1
    // Signals used (diagnostics):
    float state_saturation = 0.0f;   // token-shift magnitude proxy
    float lexical_overlap = 0.0f;    // reply vs context overlap
    bool  low_evidence = false;      // reply asserts without context support
};

class ConfidenceScorer {
public:
    // state_saturation: mean |tmix delta| between last two forwards
    // (0..1 proxy). lexical_overlap: content-word Jaccard of reply vs the
    // last user turn + retrieved memory.
    static ConfidenceReport score(const std::string& reply,
                                  const std::string& context,
                                  float state_saturation);
};

// ===========================================================================
// ThinkingMode (spec #6): toggleable, budgeted reasoning traces
// ===========================================================================
class ThinkingMode {
public:
    enum class Mode : int32_t { Off = 0, Auto, Forced };

    void set_mode(Mode m) { mode_ = m; }
    Mode mode() const { return mode_; }

    // Decides whether to think: forced always, auto only for hard inputs.
    bool should_think(const std::string& input) const;

    // Produces a compact, controllable reasoning scaffold for the prompt
    // (TokenSkip-style: bounded, compressible). Returns "" when disabled.
    std::string think_prefix(const std::string& input,
                             int32_t budget_tokens) const;

private:
    Mode mode_ = Mode::Auto;
};

// ===========================================================================
// ErrorRecovery (spec #25): classify tool/IO failures and choose a policy
// ===========================================================================
class ErrorRecovery {
public:
    enum class Policy : int32_t { Retry = 0, Backoff, Degrade, Abort };

    struct Attempt {
        uint32_t tries = 0;
        double   last_ms = 0.0;
    };

    static Policy decide(const std::string& error_text, Attempt& a);
    static std::string recover_message(Policy p, const std::string& tool);
};

// ===========================================================================
// UserFeedbackLoop (spec #47, #48)
// ===========================================================================
class UserFeedbackLoop {
public:
    // Explicit: "yes/no/correct/wrong" verdicts on the previous turn.
    enum class Verdict : int32_t { None = 0, Confirm, Correct, Reject };

    static Verdict parse(const std::string& text);

    void record(Verdict v, const std::string& task_key);
    void record_implicit(const std::string& task_key, bool engaged);

    // Trust factor for a task key: 0.5 neutral; >0.5 rewarded; <0.5 punished.
    double trust(const std::string& task_key) const;
    size_t size() const { return feedback_.size(); }

private:
    std::unordered_map<std::string, std::pair<uint32_t, uint32_t>> feedback_;
        // key -> {confirmations, corrections}
};

// ===========================================================================
// SensoryInterruptSystem (spec #2 + event-driven blueprint)
// ===========================================================================
class SensoryInterruptSystem {
public:
    enum class Event : int32_t {
        None = 0,
        WakeWord,          // wake phrase spotted
        SoundAlarm,        // alarm / siren / glass (safety first)
        SoundDoorbell,
        SoundSpeech,       // someone is speaking
        Motion,            // visual motion interrupt
        Gesture,           // recognized gesture command
        EmotionShift,      // user emotion changed sharply
    };

    struct Interrupt {
        Event event = Event::None;
        int32_t priority = 0;      // higher preempts lower
        std::string detail;        // token fragment, e.g. "[audio:alarm]"
        double at_ms = 0.0;
    };

    // Raise an event; drops it when a same-priority event is pending.
    void raise(Event e, int32_t priority, const std::string& detail);

    // Pops the highest-priority pending interrupt (if any).
    bool poll(Interrupt& out);

    // The agent loop checks this before continuing a generation.
    bool should_preempt(int32_t current_priority) const;

    size_t pending() const { return queue_.size(); }
    void clear() { queue_.clear(); }

private:
    std::deque<Interrupt> queue_;
};

const char* interrupt_event_name(SensoryInterruptSystem::Event e);

// ===========================================================================
// KnowledgeGraph (spec #72, #39): compact semantic triple store
// ===========================================================================
struct KgTriple {
    std::string subject;
    std::string relation;     // "is", "has", "at", "likes", ...
    std::string object;
    uint64_t    at_token = 0;
    float       confidence = 1.0f;
};

class KnowledgeGraph {
public:
    struct Config {
        size_t max_triples = 512;          // ~100 KB ceiling
    };
    explicit KnowledgeGraph(const Config& cfg = {}) : cfg_(cfg) {}

    // Mines triples from plain text with pattern rules:
    //   "X is Y", "X has Y", "X likes Y", "my X is Y", "X at Y".
    size_t ingest_text(const std::string& text, uint64_t at_token);

    // Index lookup: all triples touching an entity (subject OR object).
    std::vector<KgTriple> query(const std::string& entity) const;

    // Facts as memory-prompt lines ("[kg] alice likes apples").
    std::vector<std::string> facts_for_prompt(const std::string& entity,
                                              int32_t max_lines) const;

    size_t size() const { return triples_.size(); }
    void clear() { triples_.clear(); }
    const Config& config() const { return cfg_; }

    // Persistence ('OKGT' binary).
    bool save(const std::string& path) const;
    bool load(const std::string& path);

private:
    Config cfg_;
    std::vector<KgTriple> triples_;
    std::unordered_map<std::string, std::vector<size_t>> index_;
    void reindex(size_t triple_index);
};

// ===========================================================================
// ProgressiveDisclosure (spec #26): summary first, detail on demand
// ===========================================================================
class ProgressiveDisclosure {
public:
    // Returns the summary tier of a long text (first sentences, bounded).
    static std::string summarize(const std::string& text,
                                 int32_t max_sentences);
    // True when the user asked to expand ("more", "detail", "elaborate").
    static bool wants_detail(const std::string& user_text);
};

} // namespace omniseed
