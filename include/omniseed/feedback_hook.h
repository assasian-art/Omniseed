// =============================================================================
//  OmniSeed — feedback_hook.h
//
//  MILESTONE 10 — Phase 2.3: FEEDBACK HOOKS.
//
//  Every head in this tree produces a number. Until now NOTHING recorded what
//  happened next, so the number could never be checked against reality and the
//  loop could never close: a system that predicts and never scores itself is
//  not learning, it is only talking.
//
//  This file is that join. Three things already existed and never touched:
//
//    * `StreamEvent` (§37) — the head's prediction AND the filter's commitment,
//      one per step, with the reason the filter did what it did.
//    * `UserFeedbackLoop` (agent_intel.h) — the user's confirm/correct/reject
//      verdicts, with a Laplace-smoothed trust factor per task key.
//    * `SelfImprovement` — a persisted trace cache, for a different question
//      ("which tool sequence worked"), not this one.
//
//  The gap was the RECORD. A `StreamEvent` lives for one step and is dropped.
//  A `Verdict` updates a counter and is dropped. Nothing joined a PREDICTION to
//  the OUTCOME that followed it, and nothing persisted the pair, so no offline
//  fitter could ever consume the system's own history.
//
//  WHAT IS RECORDED, EXACTLY. Two actions per step, deliberately kept apart:
//
//    predicted — the HEAD's raw action this step (StreamEvent::observed_action)
//    held      — the FILTER's committed action after this step (StreamEvent::action)
//
//  They are different claims and they are scored separately. The head's
//  accuracy answers "is the projection any good"; the filter's answers "does
//  the debounce/hysteresis layer help or hurt". Collapsing them into one number
//  would hide the case that matters most: a filter that is more stable AND less
//  correct.
//
//  AN UNRESOLVED PREDICTION IS NOT A WRONG PREDICTION.
//  This is the same convention as `calibration_error() < 0` = "never measured"
//  and `distance()` = -1.0 = "not measured". Four distinct outcomes:
//
//    Unknown   — recorded, no outcome yet. Not counted anywhere.
//    Realised  — a ground-truth action/label arrived. Counted, hit or miss.
//    Confirmed — the user said we were right.  Counted, hit.
//    Corrected — the user said we were wrong.  Counted, miss.
//    Rejected  — the user said this was not a valid thing to decide.
//                NOT counted as a miss: a decision that should not have been
//                made is a different failure from a decision that was wrong,
//                and averaging them would flatter the system.
//    Expired   — the horizon passed with no outcome. NOT a miss. A market that
//                never moved is not evidence that the call was bad.
//
//  `hit_rate()` therefore returns **-1.0**, never 0.0, when nothing has been
//  resolved. "I have never been checked" and "I have always been wrong" are
//  different statements and only one of them would be true.
//
//  THE JOURNAL IS THE TRAINING SET. `to_tsv()` emits the rows the OFFLINE
//  fitter would consume (id, step, task, predicted, confidence, outcome,
//  realised). Nothing here trains — the runtime still only loads a `.bin` — but
//  a head that is never re-fitted from its own mistakes is a head that never
//  improves, and this is the file that makes re-fitting possible.
//
//  HONESTY NOTE. `score` is caller-supplied and is recorded, never predicted.
//  It is NOT a profitability claim and nothing in this file computes one. The
//  mandate's R2 (never claim or target "no loss") is respected by construction:
//  there is no objective function here, only a record of what happened.
// =============================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "omniseed/agent/agent_intel.h"
#include "omniseed/decision_head.h"
#include "omniseed/streaming_decision.h"

namespace omniseed {

// ---------------------------------------------------------------------------
// What happened to a recorded prediction.
// ---------------------------------------------------------------------------
enum class OutcomeKind : int32_t {
    Unknown = 0,   // recorded, not yet resolved
    Realised,      // a ground-truth action/label arrived
    Confirmed,     // the user confirmed the prediction
    Corrected,     // the user said the prediction was wrong
    Rejected,      // the user said this was not a valid decision to make
    Expired,       // no outcome within the horizon
    COUNT
};

const char* outcome_kind_name(OutcomeKind k);
bool        outcome_kind_from_name(const std::string& name, OutcomeKind& out);

// True when the outcome determines whether the prediction was RIGHT.
// Realised needs the realised action to be compared; Confirmed/Corrected carry
// the answer in the kind itself. Rejected and Expired are truth-free.
bool outcome_kind_has_truth(OutcomeKind k);

// ---------------------------------------------------------------------------
// JournalEntry — one prediction and, eventually, what happened to it.
// ---------------------------------------------------------------------------
struct JournalEntry {
    int64_t      id   = 0;    // monotonic within a journal, stable across save/load
    int64_t      step = 0;    // the stream step that produced it (0 if not streamed)
    std::string  task_key;    // the trust key, e.g. "trading.AAPL.1d"

    // --- the prediction ------------------------------------------------------
    DecisionAction predicted  = DecisionAction::ABSTAIN;  // the HEAD's raw action
    float          confidence = 0.0f;   // the head's softmax prob of `predicted`
    float          margin     = 0.0f;   // p(top) - p(second)

    // --- the filter's state, when the prediction came from a stream ----------
    DecisionAction held      = DecisionAction::ABSTAIN;  // the FILTER's action
    bool           committed = false;   // the filter is holding a real position
    bool           changed   = false;   // this step moved the committed action
    float          strength  = 0.0f;    // the agreeing run / vote count

    // --- the outcome ---------------------------------------------------------
    OutcomeKind    outcome  = OutcomeKind::Unknown;
    DecisionAction realised = DecisionAction::ABSTAIN;  // meaningful when Realised
    // Caller-supplied observation. Recorded, never predicted; NOT a profit claim.
    float          score    = 0.0f;
    int32_t        horizon  = 0;   // steps the outcome was measured over

    double at_ms = 0.0;   // wall clock when recorded

    bool resolved() const { return outcome != OutcomeKind::Unknown; }

    // True when this row has an answer at all (see the header note).
    bool has_verdict() const { return outcome_kind_has_truth(outcome); }

    // Did the HEAD get it right? Only meaningful when has_verdict().
    bool head_hit() const;

    // True when this row is a FILTER decision with a determinate answer.
    // A user verdict is about what we SAID, so if the filter was holding
    // something other than what the head predicted, the verdict does not
    // describe the filter and the row is not a filter row.
    bool filter_scored() const;

    // Did the FILTER get it right? Only meaningful when filter_scored().
    bool filter_hit() const;

    // One line, for logs and for the JSON document.
    std::string to_json() const;
};

// ---------------------------------------------------------------------------
// JournalStats — what the record says, with "not measured" spelled -1.0.
// ---------------------------------------------------------------------------
struct JournalStats {
    int64_t predictions = 0;   // rows recorded
    int64_t unresolved  = 0;   // outcome == Unknown
    int64_t expired     = 0;   // horizon passed, no outcome
    int64_t rejected    = 0;   // user said the decision was invalid
    int64_t verdicts    = 0;   // rows with a determinate answer (the denominator)

    int64_t hits = 0;          // head hits among `verdicts`

    // Rows where the filter was holding a position when the prediction was made.
    int64_t committed_rows = 0;

    // The FILTER, scored only on the steps where it actually held something.
    int64_t filter_verdicts = 0;
    int64_t filter_hits     = 0;

    // Mean confidence when right vs when wrong. -1.0 = not measured.
    // `confidence_gap` = mean_conf_hit - mean_conf_miss: POSITIVE means the
    // head's confidence is informative, <= 0 means it is noise or inverted.
    // It is a SIGN you must measure, never assert.
    double mean_conf_hit  = -1.0;
    double mean_conf_miss = -1.0;
    double confidence_gap = -1.0;

    // Reliability of `confidence` as p(correct) over the verdict rows.
    // -1.0 = not measured (fewer than one non-empty bin).
    double ece   = -1.0;
    double brier = -1.0;

    // confusion[predicted][realised], only over `Realised` rows.
    // Empty when nothing has been realised.
    std::vector<std::vector<int64_t>> confusion;

    // -1.0 when there is nothing to measure. NEVER 0.0 for "no data".
    double hit_rate() const;
    double filter_hit_rate() const;

    // Committed steps as a fraction of all recorded steps: how often the filter
    // chose to act at all. -1.0 when nothing was recorded.
    double commitment_rate() const;

    std::string to_json() const;
};

// ---------------------------------------------------------------------------
// DecisionJournal — the persisted (prediction, outcome) store.
//
// Bounded: at most Config::max_entries rows are kept, oldest first out. A
// journal that grows without limit is a leak in a 24/7 process, and the whole
// point is that it is small enough to re-fit from.
//
// FAILS CLOSED AND ATOMICALLY. `load()` and `deserialise()` parse into a LOCAL
// vector and swap only on full success, so a malformed byte leaves the existing
// journal UNTOUCHED rather than half-read. A partially restored record that
// silently drops the misses is worse than no record at all.
// ---------------------------------------------------------------------------
class DecisionJournal {
public:
    struct Config {
        size_t max_entries = 8192;   // FIFO prune beyond this
    };

    DecisionJournal() = default;
    explicit DecisionJournal(const Config& cfg) : cfg_(cfg) {}

    const Config& config() const { return cfg_; }
    const std::string& error() const { return error_; }

    // ---- recording ----------------------------------------------------------
    // Assigns `id` (monotonic) and `at_ms` when they are unset. Returns the id.
    int64_t record(JournalEntry e);

    // The §37 joint: one entry per VALID event. An invalid event is a step the
    // head refused to observe — it is not a prediction and is NOT recorded.
    int64_t record_event(const StreamEvent& e, const std::string& task_key);

    // One entry per event in the stream. Returns how many were recorded.
    size_t record_stream(const std::vector<StreamEvent>& events,
                         const std::string& task_key);

    // A prediction with no stream behind it (a single-shot head call).
    int64_t record_decision(const DecisionResult& d, const std::string& task_key);

    // ---- resolution ---------------------------------------------------------
    // Fails (returns false, leaves the entry alone) on an unknown id, or on an
    // entry that is ALREADY resolved: re-scoring a prediction against a second
    // outcome is how a record starts agreeing with itself.
    bool resolve(int64_t id, DecisionAction realised, float score, int32_t horizon);
    bool resolve_kind(int64_t id, OutcomeKind kind);

    // Resolve EVERY unresolved entry for `task_key` against one outcome. This is
    // the trading case: every prediction made during the window was a prediction
    // about the same forward return, so they share one realised action.
    // Returns how many rows were resolved.
    size_t resolve_all(const std::string& task_key, DecisionAction realised,
                       float score, int32_t horizon);

    // Resolve the NEWEST unresolved entry for `task_key` (the user is replying
    // to the last thing said). Returns 0 when there is nothing to resolve.
    size_t resolve_latest(const std::string& task_key, OutcomeKind kind);

    // Mark every unresolved entry at or before `step` as Expired. An empty
    // `task_key` means every key; a non-empty one scopes it, so one slow task
    // cannot expire another's open predictions.
    size_t expire_through(int64_t step, const std::string& task_key = std::string());

    // ---- queries ------------------------------------------------------------
    size_t size() const { return entries_.size(); }
    size_t unresolved() const;
    const std::vector<JournalEntry>& entries() const { return entries_; }
    const JournalEntry* find(int64_t id) const;
    JournalStats stats() const;

    std::string to_json() const;
    // The rows the OFFLINE fitter consumes. Header first.
    std::string to_tsv() const;

    // ---- persistence --------------------------------------------------------
    bool save(const std::string& path) const;
    bool load(const std::string& path);

    // The same bytes save() writes and load() reads, in memory — so a container
    // that embeds the journal (FeedbackHook) uses ONE format, not two.
    std::string serialise() const;
    // Fails closed: on any malformed byte the existing journal is left UNTOUCHED
    // and false is returned.
    bool deserialise(const uint8_t* p, size_t n);

    void clear();

private:
    void prune();

    Config                   cfg_;
    std::vector<JournalEntry> entries_;
    int64_t                  next_id_ = 1;
    std::string              error_;
};

// ---------------------------------------------------------------------------
// FeedbackHook — the joint.
//
// Owns a DecisionJournal and a UserFeedbackLoop and keeps them consistent:
// every user verdict lands in BOTH (the journal as an outcome, the ledger as a
// trust update), so "what did we predict" and "how much do we trust this task"
// can never drift apart.
//
// It does NOT call a head, does not run a filter and does not train. It is the
// seam an AgentLoop or a trading loop calls once per step and once per user turn.
// ---------------------------------------------------------------------------
class FeedbackHook {
public:
    FeedbackHook() = default;
    explicit FeedbackHook(const DecisionJournal::Config& cfg) : journal_(cfg) {}

    // ---- ingestion ----------------------------------------------------------
    int64_t observe(const StreamEvent& e, const std::string& task_key);
    size_t  observe_stream(const std::vector<StreamEvent>& events,
                           const std::string& task_key);
    int64_t observe_decision(const DecisionResult& d, const std::string& task_key);

    // ---- outcomes -----------------------------------------------------------
    size_t outcome(const std::string& task_key, DecisionAction realised,
                   float score, int32_t horizon);

    // Per-row resolution, for the case where each prediction was a question
    // about a DIFFERENT future (a backtest over a held-out tape). Pass-throughs
    // so a caller never needs a mutable handle on the journal.
    bool resolve(int64_t id, DecisionAction realised, float score, int32_t horizon);
    bool resolve_kind(int64_t id, OutcomeKind kind);

    // A user turn. Parses the verdict; a None verdict changes NOTHING (returns
    // None and leaves both stores alone) — a neutral turn is not feedback.
    // Confirm/Correct resolve the newest unresolved row for `task_key`;
    // Reject resolves it as Rejected and does NOT count as a miss.
    UserFeedbackLoop::Verdict on_user_turn(const std::string& user_text,
                                           const std::string& task_key);

    // Engagement is a weak signal: it moves the TRUST LEDGER only, never the
    // journal. Silence is not an outcome, and treating it as one would let the
    // absence of a user score the head.
    void on_implicit(const std::string& task_key, bool engaged);

    size_t expire(const std::string& task_key, int64_t through_step);

    // ---- accessors ----------------------------------------------------------
    double trust(const std::string& task_key) const { return feedback_.trust(task_key); }
    const DecisionJournal&   journal()  const { return journal_; }
    const UserFeedbackLoop&  feedback() const { return feedback_; }
    size_t feedback_size() const { return feedback_.size(); }

    JournalStats stats() const { return journal_.stats(); }
    std::string  to_json() const;

    // Persists BOTH stores in one container, so they cannot be restored out of
    // step. Fails closed AND atomically: a bad file leaves the hook exactly as
    // it was, not half-restored and not reset.
    bool save(const std::string& path) const;
    bool load(const std::string& path);
    void clear();

    const std::string& error() const { return error_; }

private:
    DecisionJournal  journal_;
    UserFeedbackLoop feedback_;
    std::string      error_;
};

} // namespace omniseed
