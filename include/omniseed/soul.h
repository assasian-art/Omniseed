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
//
//  §31 ADDED MEMORY. A soul that forgets every exchange the moment it ends is
//  not a soul, it is a lookup table with opinions. So the Soul now owns a
//  MemoryCrystals store and a tokenizer, and remembers what the owner asked and
//  what it answered. Two consequences worth stating up front, because both are
//  deliberate:
//
//   * perceive() IS STILL const AND STILL WRITES NOTHING. The unified pipeline
//     calls it from a const run(), and a const method that silently appends to
//     long-term memory is a trap: the caller cannot see the write in the
//     signature, and calling run() twice stops being idempotent in a way that
//     no type can express. Remembering therefore lives in named non-const
//     entry points: perceive_and_recall() and converse(). Nothing is stored
//     that the caller did not ask to store.
//
//   * A RECALLED MEMORY IS EVIDENCE, NOT A PROBABILITY. `RecalledMemory::score`
//     is cosine similarity times importance from a bag-of-tokens projection. It
//     is reported as "relevance" and never as confidence, for the same reason
//     the emotion layer's strength is not called confidence.
// =============================================================================
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "omniseed/language/language_heads.h"
#include "omniseed/memory/memory.h"
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
// RecalledMemory — one crystal that came back for this turn.
//
// `role` records WHO said it. A memory layer that mixes the owner's words with
// its own is worse than no memory at all, because it will attribute its own
// guess to the owner with full confidence. MemoryCrystals assigns ids but knows
// nothing about speakers, so the Soul keeps the mapping; it is therefore
// session-local, and a crystal restored from disk reports role "unknown" until
// it is written again. That limitation is stated rather than hidden.
// ---------------------------------------------------------------------------
enum class MemoryRole : int32_t { Unknown = 0, Owner, Soul };
const char* memory_role_name(MemoryRole r);

struct RecalledMemory {
    uint64_t    id      = 0;
    float       score   = 0.0f;   // cosine x importance — relevance, not probability
    uint32_t    hits    = 0;
    MemoryRole  role    = MemoryRole::Unknown;
    std::string summary;
};

// The one-line gloss speak() puts in front of the answer when a turn was
// recognised. Free and public so a test can pin the phrasing directly instead
// of fishing it out of a composed reply.
std::string recall_note(const std::vector<RecalledMemory>& memories);

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

    // What the soul remembered while reading this turn. `has_recall` is the
    // gate, not `recall.size()`: "I looked and found nothing" and "I never
    // looked" are different facts, and only the first is evidence that the
    // memory layer ran.
    bool                        has_recall = false;
    std::vector<RecalledMemory> recall;
    std::string                 recall_summary;

    // The memory facet's counters, so a caller can tell a working store from a
    // decorative one.
    bool     has_memory       = false;
    size_t   memory_crystals  = 0;   // crystals held right now
    uint64_t memory_stored    = 0;   // crystals written this session
    uint64_t memory_skipped   = 0;   // turns too short to crystallize
    size_t   memory_dropped   = 0;   // crystals that decayed away
    uint64_t memory_clock     = 0;   // the soul's position in the token stream

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

        // ---- memory (§31) ---------------------------------------------------
        // On by default: a soul that cannot remember is the thing this section
        // exists to fix. It costs one minimal tokenizer and <= 2 MB of crystals.
        bool  use_memory = true;
        // How many crystals a single turn may recall.
        int32_t recall_k = 3;
        // Ceiling, decay tau, embedding width and the reinforcement step.
        //
        // The token budget is raised from the memory layer's own default of 48:
        // that truncates an ordinary sentence mid-word, and a soul that quotes a
        // half-sentence is worse than one that says nothing. Set here rather
        // than in the shared default, which the agent's own memory path relies
        // on, so this change is scoped to the soul.
        MemoryCrystals::Config memory = [] {
            MemoryCrystals::Config c;
            c.max_len_tokens = 96;
            return c;
        }();
        // Where to persist crystals. Empty = in-process only.
        std::string crystals_path;
        // A question this close to one already held is the SAME question, not a
        // new one: it is reinforced rather than filed again. Without this, ask
        // the same thing twice a day for a month and the store holds thirty
        // copies of it. 1.0 is the maximum a recall score can reach (cosine x
        // importance, both <= 1), so 0.98 means "effectively identical".
        float duplicate_recall_score = 0.98f;
        // A recall whose relevance is below this is NOT reported. "Nearest" is
        // not "related": with no floor every query returns its k closest
        // crystals whatever they say, so the soul would announce that it
        // remembers things it has never seen.
        //
        // MEASURED, not guessed, with `omniseed demo-soul` on the minimal
        // byte-level tokenizer: a question the store holds scores 0.973-0.990,
        // and one it does not scores 0.775-0.876. 0.92 sits in that gap. The
        // margin is only ~0.10 because byte-level bags are close to character
        // histograms, and it NARROWS as the store grows — see docs/SOUL.md.
        float min_relevance = 0.92f;
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

    // ---- memory (§31) -------------------------------------------------------
    // Everything below is non-const, because it writes to the crystal store.
    // See the note at the top of this file for why perceive() itself does not.
    //
    // The turn-opening call: perceive() as usual, then recall what relates to
    // this turn and store the owner's question. The reply is not stored here
    // because at perception time it does not exist yet.
    SoulState perceive_and_recall(const std::string& user_text);
    SoulState perceive_and_recall(const std::string& user_text,
                                  const EmotionalInput& in);

    // Store the soul's own reply as a second crystal, and reinforce whatever
    // this turn recalled (a recalled memory that got used should not be left to
    // age). Returns false when the reply was too short to crystallize.
    bool remember(const std::string& question, const std::string& reply);

    // One complete turn: recall + perceive, compose the reply with speak(), then
    // store the exchange. This is the only path that makes the soul stateful,
    // and it is the one a chat loop should call. `out_state` receives the state
    // that produced the reply, so the caller can serialise it.
    std::string converse(const std::string& user_text,
                         const std::string& base_reply,
                         SoulState* out_state = nullptr);
    std::string converse(const std::string& user_text,
                         const std::string& base_reply,
                         const EmotionalInput& in,
                         SoulState* out_state = nullptr);

    // Ranked recall for a bare query, without touching this turn's state.
    // k < 0 uses cfg_.recall_k. Memories below cfg_.min_relevance are dropped.
    std::vector<RecalledMemory> recall(const std::string& query, int32_t k = -1);

    // The single best match for a query, IGNORING the relevance floor, and
    // without reinforcing it. This is what the floor is applied to, so a caller
    // can inspect the raw separation instead of only seeing the post-filter
    // answer — which is the difference between a measured threshold and a magic
    // number. Returns false when the store is empty.
    bool top_match(const std::string& query, RecalledMemory* out = nullptr);

    // Advance the memory clock. Turns advance it by their own token length; a
    // test jumps it forward to simulate elapsed time.
    void     advance_memory_clock(uint64_t tokens);
    uint64_t memory_clock() const { return memory_clock_; }

    // Drop crystals whose effective importance decayed below the floor, at the
    // current clock or at an explicit one. Returns how many went. Belongs in a
    // dream/idle pass, not on the hot path.
    size_t decay_memories();
    size_t decay_memories(uint64_t now_token);

    // Explicitly reinforce one crystal (raises importance, resets its age).
    bool reinforce_memory(uint64_t id);

    // Persistence. Best-effort: a missing file on load is not an error.
    bool save_memories() const;
    bool load_memories();

    size_t   memory_size() const { return crystals_.size(); }
    uint64_t memory_stored() const { return memory_stored_; }
    uint64_t memory_skipped() const { return memory_skipped_; }
    size_t   memory_dropped() const { return memory_dropped_; }
    const MemoryCrystals& crystals() const { return crystals_; }
    const Tokenizer&      tokenizer() const { return tok_; }

    // ---- self-knowledge ----------------------------------------------------
    Metacognition::State assess(double realized_sharpe, size_t observations) const;
    // The honest capability map, one line per capability, for `introspect`.
    std::string capability_report() const;

    Persona&            persona()       { return persona_; }
    const Persona&      persona() const { return persona_; }
    GoalTracker&        goals()         { return goals_; }
    const GoalTracker&  goals()   const { return goals_; }

private:
    // Crystallize one side of an exchange under its chat role marker, and record
    // who said it. Returns the new crystal's id, or 0 when the text was too
    // short to crystallize.
    uint64_t store_turn(const std::string& text, MemoryRole role,
                        uint64_t answers = 0);
    // crystal id -> speaker, plus the question a reply answers. A plain struct
    // in the private section so this header does not leak the enum layout.
    // Session-local; see RecalledMemory.
    struct MemoryLink {
        uint64_t id      = 0;
        int32_t  role    = 0;
        uint64_t answers = 0;   // for a Soul crystal: the question it replied to
    };
    const MemoryLink* link_of(uint64_t id) const;
    MemoryRole        role_of(uint64_t id) const;
    // The reply that followed a recalled question, if one was stored. Relevance
    // is inherited from the question: the answer is relevant by construction,
    // not by its own cosine — a reply's words rarely resemble the question's.
    bool paired_answer(uint64_t question_id, RecalledMemory* out) const;
    void prune_roles();
    void fill_memory_stats(SoulState& st) const;

    bool ready_ = false;
    std::string error_;

    Config              cfg_;
    Persona             persona_;
    EmotionalResonance  emotion_;
    GoalTracker         goals_;
    ActionRationale     rationale_;

    // ---- memory facet ------------------------------------------------------
    bool           memory_ready_ = false;
    Tokenizer      tok_;
    MemoryCrystals crystals_;
    // Starts at 1, not 0: retrieve() reads 0 as "the caller has no clock, do
    // not touch recency", so a real clock must never be 0.
    uint64_t       memory_clock_   = 1;
    uint64_t       memory_stored_  = 0;
    uint64_t       memory_skipped_ = 0;
    size_t         memory_dropped_ = 0;
    std::vector<MemoryLink> links_;
    // The question stored by the current turn, so remember() can link the reply
    // to it. 0 means this turn stored no question.
    uint64_t last_question_id_ = 0;
    // True when this turn's question was one already held, so remember() can
    // also skip filing an answer that has not changed.
    bool last_turn_duplicate_ = false;
    // What the last perceive_and_recall() surfaced, so remember() reinforces
    // exactly those rather than re-running the query.
    std::vector<uint64_t> last_recall_ids_;
};

} // namespace omniseed
