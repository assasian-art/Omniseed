// =============================================================================
//  OmniSeed — streaming_decision.h
//
//  MILESTONE 9 — Phase 2.2: STREAMING DECISIONS.
//
//  The decision head is SINGLE-SHOT: one h[E] in, one action out. That is the
//  right primitive for a judgement about one situation and the wrong primitive
//  for a control loop. A loop that acts on every single-shot output acts on
//  noise — one blip in h[E] moves the position, and the system chatters.
//
//  This file adds the TEMPORAL state the head deliberately does not have. It is
//  a FILTER over the head's outputs: not a second head, not a second backbone,
//  and not a second forward pass.
//
//  WHAT IT DOES NOT DO — read this before believing anything else.
//    * It does NOT make the backbone incremental. The backbone already is:
//      RWKV-7 here is the recurrent form, so h_t is a function of (token_t,
//      state_{t-1}) and the state already carries the history. This layer never
//      reads or writes that state.
//    * It does NOT make the head more accurate. It makes the head's output
//      STABLE, and stability and accuracy are different claims. No filter can
//      recover information the head never had.
//    * It does NOT invent a distribution. Every action it emits came from a real
//      head call; the filter only decides WHEN to trust one.
//
//  THREE MECHANISMS, each a standard control-loop primitive:
//
//    1. DEBOUNCE (N-of-M). An action is COMMITTED only after `confirm_steps`
//       consecutive agreeing observations. A single blip cannot commit.
//
//    2. HYSTERESIS. KEEPING a committed action needs `min_confidence`; CHANGING
//       it needs `switch_confidence >= min_confidence`. That asymmetry is what
//       stops a loop oscillating across a noisy decision boundary. A config with
//       switch < min is normalised UP, never down: changing must never be
//       cheaper than staying.
//
//    3. RELEASE. A commitment whose evidence has evaporated is released to
//       ABSTAIN after `release_after_weak` consecutive weak observations.
//       Silence is a reason to stop acting; an unbounded hold would be a
//       decision the head is no longer making.
//
//  A REPLACEMENT IS REPORTED, NOT SWALLOWED. When a committed action changes,
//  the event carries `changed = true` and BOTH the old and the new action,
//  because a consumer must undo the previous position. Going quiet would leave
//  it holding a stale one.
//
//  ABSTAIN IS NOT A COMMITMENT. ABSTAIN means "no position", so a committed
//  filter is never in that state: `committed == true` implies a real action.
//  A strong run of ABSTAIN is a RELEASE when something is held (the head is
//  clearly saying "no action", which is not the same as saying nothing), and a
//  no-op when nothing is. This is not a nicety — it is what makes
//  `changed events == commits + changes + releases` hold exactly, an identity
//  the test suite pins over 10,000 random steps.
//
//  TWO MODES, because two callers want different things:
//    * Confirm — order-SENSITIVE. "Is the head saying the same thing, clearly,
//                N times running?" What a live loop wants.
//    * Window  — order-INSENSITIVE. "Over the last W observations, which action
//                won the vote?" What a backtest wants, evaluated every bar.
//  Both modes reduce to the same candidate, and ONE state machine consumes it,
//  so the commit / hysteresis / release rules are written — and tested — once.
//
//  FAILS CLOSED. A head that is not ready, a null pointer, or an observation
//  the head refused to make records NOTHING: the streak is not advanced, the
//  commitment is not renewed, and the event says `valid = false`. A dead head
//  must never silently confirm a decision — an unreadable signal is not
//  agreement.
// =============================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "omniseed/core/batch_gemm.h"
#include "omniseed/decision_head.h"

namespace omniseed {

// ---------------------------------------------------------------------------
// How the observations are reduced to one candidate.
// ---------------------------------------------------------------------------
enum class StreamMode : int32_t {
    // Order-sensitive: N consecutive agreeing, strong observations.
    Confirm = 0,
    // Order-insensitive: a vote over a sliding window of W observations.
    Window,
    COUNT
};

const char* stream_mode_name(StreamMode m);
bool        stream_mode_from_name(const std::string& name, StreamMode& out);

// ---------------------------------------------------------------------------
// StreamingConfig — every knob, and what happens to a bad value.
//
// Nothing here is fitted. The defaults are neutral (a bare majority), and the
// only automatic repair is the hysteresis ordering, which can only ever make
// the system MORE conservative.
// ---------------------------------------------------------------------------
struct StreamingConfig {
    StreamMode mode = StreamMode::Confirm;

    // Confirm: consecutive agreeing observations needed to commit.
    // Window:  the quorum of votes a winner needs to commit.
    // Clamped to >= 1. A value < 1 would commit on no evidence.
    int32_t confirm_steps = 3;

    // Window only: how many observations are kept. Clamped to >= confirm_steps
    // and >= 1, because a window smaller than its own quorum can never commit.
    int32_t window = 5;

    // Confidence required to KEEP a committed action (and to be a strong
    // observation at all). An observation below this is "weak".
    float min_confidence = 0.50f;

    // Confidence required to CHANGE a committed action. Raised to
    // min_confidence when a caller sets it lower.
    float switch_confidence = 0.60f;

    // Optional floor on p(top) - p(second). A 0.90/0.88 split clears a
    // confidence bar while being nearly a coin flip. 0 disables the check.
    float min_margin = 0.0f;

    // Consecutive weak observations that release a commitment to ABSTAIN.
    // 0 disables release entirely (the commitment is held until contradicted).
    int32_t release_after_weak = 3;

    // Returns the config with every out-of-range field repaired. The ONLY
    // repair that is not a clamp is the hysteresis ordering.
    StreamingConfig normalised() const;
    std::string to_json() const;
};

// ---------------------------------------------------------------------------
// StreamEvent — one step of the filter's output.
//
// Always produced, even when nothing happened: a caller that only wants the
// transitions checks `first || changed`, and a caller logging every step gets
// every step. The event is self-describing so a log line never has to guess why
// the filter did what it did — `reason` is never empty.
// ---------------------------------------------------------------------------
struct StreamEvent {
    int64_t step = 0;      // 1-based index of the observation that produced this
    bool    valid = false; // false => the head made no observation; NOTHING moved

    // The head's raw output this step. Meaningful only when `valid`.
    DecisionAction observed_action = DecisionAction::ABSTAIN;
    float          observed_conf   = 0.0f;
    float          observed_margin = 0.0f;

    // The filter's state AFTER this step.
    DecisionAction action    = DecisionAction::ABSTAIN; // committed, or ABSTAIN
    bool           committed = false;                   // a stable action is held
    DecisionAction previous  = DecisionAction::ABSTAIN; // before a change

    // A commitment was created (first ever) or replaced this step. On the first
    // commitment BOTH are true and `previous` is ABSTAIN.
    bool first   = false;
    bool changed = false;

    // Confirm: the current agreeing run. Window: the winner's vote count.
    // 0 when the observation was weak or the head made none.
    int32_t strength = 0;

    // Window only: the vote margin (winner - runner-up) / total, in [0, 1].
    // This is NOT a probability margin and is never reported as one.
    float vote_margin = 0.0f;

    std::string reason;   // never empty
    std::string to_json() const;
};

// ---------------------------------------------------------------------------
// StreamingDecision — the filter.
//
// Holds a NON-OWNING reference to a DecisionHead, which must outlive it. The
// head is the caller's: one head can drive several filters (a Confirm filter
// and a Window filter over the same stream), and a caller that already has a
// fitted head does not have to hand over ownership of it.
//
// `push_decision()` takes an already-computed DecisionResult, which is the seam
// that lets a BATCH drive the filter: `DecisionHead::decide_batch` produces B
// results in one GEMM (§36) and the filter then walks them in order, so the
// per-row head cost is paid once for the whole sequence.
//
// Not thread-safe against itself, the same documented contract as every head in
// this tree.
// ---------------------------------------------------------------------------
class StreamingDecision {
public:
    StreamingDecision() = default;
    explicit StreamingDecision(const DecisionHead& head);

    // Attaches a head. A null pointer is refused. Returns false and does not
    // disturb the filter state when the head is null or not ready — the filter
    // is still usable through push_decision().
    bool init(const DecisionHead& head);

    bool ready() const { return head_ != nullptr && head_->ready(); }
    const std::string& error() const { return error_; }
    const StreamingConfig& config() const { return cfg_; }
    // Repairs the config (see StreamingConfig::normalised) before storing it, so
    // an impossible hysteresis ordering cannot be installed.
    void set_config(const StreamingConfig& c) { cfg_ = c.normalised(); }

    // ---- the hot path -------------------------------------------------------
    // One head call, then the filter. Returns false — and fills `out` with
    // `valid = false`, `reason` explaining why — when the head cannot observe.
    // A false return never advances `steps()` and never touches the filter's
    // commitment, streak or window: an unreadable signal is not agreement.
    bool push(const float* hidden, StreamEvent& out);
    bool push(const Tensor& hidden, StreamEvent& out);

    // Filter-only: feed a decision that was computed elsewhere (e.g. by
    // decide_batch). Always returns true and always counts as an observation,
    // because a DecisionResult that reached here is a real head output.
    void push_decision(const DecisionResult& d, StreamEvent& out);

    // B decisions in one GEMM (§36), then filtered in order. The batch and the
    // per-row path produce identical events when the Scalar kernel is selected;
    // `tests/test_streaming_decision.cpp` C1 asserts that with `==`.
    //
    // FAILS CLOSED: a not-ready head, a null pointer, or a [B, E] mismatch
    // returns false and leaves `out` EMPTY — never a partial sequence, because a
    // caller that ignores the return value must not act on half a stream.
    bool push_batch(const float* H, int32_t B, std::vector<StreamEvent>& out,
                    BatchStats* stats = nullptr);
    bool push_batch(const Tensor& H, std::vector<StreamEvent>& out,
                    BatchStats* stats = nullptr);

    // ---- state --------------------------------------------------------------
    // Clears the streak, the window, the commitment and every counter. The head
    // is NOT detached.
    void reset();

    // Observations recorded (a failed push does not count).
    int64_t steps() const { return steps_; }
    // The most recent event, valid or not.
    const StreamEvent& last() const { return last_; }
    DecisionAction action() const { return action_; }
    bool  committed() const { return committed_; }
    // Confirm: the current agreeing run. Window: the winner's vote count.
    int32_t strength() const { return strength_; }
    // How many observations the Window ring currently holds.
    int32_t window_filled() const { return win_filled_; }

    // ---- counters -----------------------------------------------------------
    int64_t commits() const { return commits_; }   // first commitments
    int64_t changes() const { return changes_; }   // replacements of a commitment
    int64_t releases() const { return releases_; } // commitment -> ABSTAIN
    int64_t weak() const { return weak_; }         // observations below the bars
    int64_t failures() const { return failures_; } // pushes that recorded nothing

    std::string to_json() const;

private:
    // The one candidate both modes produce.
    //
    // `strong` is false when the candidate cannot move the filter — either
    // because it is below min_confidence / min_margin, or (Window) because it
    // has not reached its quorum. `gated` is the distinct case of a candidate
    // that IS strong but is a challenge to the incumbent that failed to clear
    // the higher switch bar: it is refused WITHOUT being counted as weak
    // evidence, because it is not weak — it is merely not enough to switch.
    struct Candidate {
        DecisionAction action = DecisionAction::ABSTAIN;
        float          conf   = 0.0f;
        float          margin = 0.0f;   // vote margin (Window); 0 in Confirm
        int32_t        strength = 0;    // streak (Confirm) or votes (Window)
        bool           strong = false;
        bool           gated  = false;
    };

    // True when a candidate may challenge the incumbent: the incumbent itself,
    // an empty seat, or a challenger that clears switch_confidence.
    bool clears_switch_bar(const Candidate& c) const;
    // Confirm: advances the agreeing run. Mutates the pending state.
    Candidate reduce_confirm(const DecisionResult& d);
    // Window: pushes into the ring and tallies the votes.
    Candidate reduce_window(const DecisionResult& d);
    // The single state machine, shared by both modes.
    void apply(const Candidate& c, StreamEvent& out);
    // Fills a "nothing happened" event that explains why.
    void refuse(const char* why, StreamEvent& out);

    const DecisionHead* head_ = nullptr;
    StreamingConfig     cfg_;
    std::string         error_;

    // --- commitment state ---
    DecisionAction action_    = DecisionAction::ABSTAIN;
    bool           committed_ = false;
    int32_t        strength_  = 0;   // current run / vote count for the winner
    int32_t        weak_run_  = 0;   // consecutive weak observations

    // --- pending (a not-yet-committed alternative) ---
    DecisionAction pending_action_ = DecisionAction::ABSTAIN;
    int32_t        pending_streak_ = 0;

    // --- Window ring ---
    std::vector<DecisionAction> win_action_;
    std::vector<float>          win_conf_;
    int32_t                     win_cap_    = 0;   // cfg_.window the ring was sized for
    int32_t                     win_head_   = 0;
    int32_t                     win_filled_ = 0;

    // --- counters / last event ---
    int64_t     steps_    = 0;
    int64_t     commits_  = 0;
    int64_t     changes_  = 0;
    int64_t     releases_ = 0;
    int64_t     weak_     = 0;
    int64_t     failures_ = 0;
    StreamEvent last_;
};

} // namespace omniseed
