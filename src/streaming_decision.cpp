// =============================================================================
//  OmniSeed — streaming_decision.cpp
//
//  MILESTONE 9 — Phase 2.2. See streaming_decision.h for the contract.
//
//  The file is deliberately small. Everything expensive already exists: the
//  head does the matvec, decide_batch() does the batched GEMM. What lives here
//  is the temporal state, and temporal state is where control loops go wrong,
//  so it is written once — one state machine, two reducers — rather than twice.
// =============================================================================
#include "omniseed/streaming_decision.h"

#include "omniseed/heads.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace omniseed {

// ---------------------------------------------------------------------------
// Names
// ---------------------------------------------------------------------------
const char* stream_mode_name(StreamMode m) {
    switch (m) {
        case StreamMode::Confirm: return "confirm";
        case StreamMode::Window:  return "window";
        case StreamMode::COUNT:   break;
    }
    return "confirm";
}

bool stream_mode_from_name(const std::string& name, StreamMode& out) {
    if (name == "confirm" || name == "Confirm") { out = StreamMode::Confirm; return true; }
    if (name == "window"  || name == "Window")  { out = StreamMode::Window;  return true; }
    return false;
}

namespace {

constexpr int32_t kActionCount = static_cast<int32_t>(DecisionAction::COUNT);

// Clamp into [lo, hi]; a NaN or an infinity becomes `def` rather than being
// clamped, because clamping an infinity would silently accept a corrupt file.
float clamp_unit(float v, float lo, float hi, float def) {
    if (!(v == v) || v > 1.0e30f || v < -1.0e30f) return def;
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

void append_action(std::string& out, DecisionAction a) {
    out += '"';
    out += decision_action_name(a);
    out += '"';
}

} // namespace

// ---------------------------------------------------------------------------
// StreamingConfig
// ---------------------------------------------------------------------------
StreamingConfig StreamingConfig::normalised() const {
    StreamingConfig c = *this;

    // An out-of-range mode is not a mode. Fall back to the order-sensitive one,
    // which is the more conservative of the two (it demands CONSECUTIVE
    // agreement; Window accepts a scattered majority).
    if (c.mode != StreamMode::Confirm && c.mode != StreamMode::Window)
        c.mode = StreamMode::Confirm;

    // Zero evidence must never commit.
    if (c.confirm_steps < 1) c.confirm_steps = 1;
    if (c.window < 1) c.window = 1;
    // A window smaller than its own quorum can never commit, which would look
    // like a bug in the filter rather than a bad config. Grow it.
    if (c.window < c.confirm_steps) c.window = c.confirm_steps;

    c.min_confidence    = clamp_unit(c.min_confidence,    0.0f, 1.0f, 0.50f);
    c.switch_confidence = clamp_unit(c.switch_confidence, 0.0f, 1.0f, 0.60f);
    // THE HYSTERESIS INVARIANT. Changing a commitment must never be cheaper
    // than keeping it. This is the only repair that is not a plain clamp, and
    // it can only ever make the filter MORE conservative.
    if (c.switch_confidence < c.min_confidence)
        c.switch_confidence = c.min_confidence;

    c.min_margin = clamp_unit(c.min_margin, 0.0f, 1.0f, 0.0f);
    if (c.release_after_weak < 0) c.release_after_weak = 0;
    return c;
}

std::string StreamingConfig::to_json() const {
    std::string s = "{\"mode\":\"";
    s += stream_mode_name(mode);
    s += "\",\"confirm_steps\":" + std::to_string(confirm_steps);
    s += ",\"window\":"        + std::to_string(window);
    s += ",\"min_confidence\":";    heads_detail::append_float(s, min_confidence);
    s += ",\"switch_confidence\":"; heads_detail::append_float(s, switch_confidence);
    s += ",\"min_margin\":";        heads_detail::append_float(s, min_margin);
    s += ",\"release_after_weak\":" + std::to_string(release_after_weak);
    s += "}";
    return s;
}

// ---------------------------------------------------------------------------
// StreamEvent
// ---------------------------------------------------------------------------
std::string StreamEvent::to_json() const {
    std::string s = "{\"step\":" + std::to_string(step);
    s += valid ? ",\"valid\":true" : ",\"valid\":false";
    s += ",\"observed_action\":";
    append_action(s, observed_action);
    s += ",\"observed_confidence\":"; heads_detail::append_float(s, observed_conf);
    s += ",\"observed_margin\":";     heads_detail::append_float(s, observed_margin);
    s += ",\"action\":";
    append_action(s, action);
    s += ",\"previous\":";
    append_action(s, previous);
    s += committed ? ",\"committed\":true" : ",\"committed\":false";
    s += first ? ",\"first\":true" : ",\"first\":false";
    s += changed ? ",\"changed\":true" : ",\"changed\":false";
    s += ",\"strength\":" + std::to_string(strength);
    s += ",\"vote_margin\":"; heads_detail::append_float(s, vote_margin);
    s += ",\"reason\":\"";    s += heads_detail::json_escape(reason); s += "\"}";
    return s;
}

// ---------------------------------------------------------------------------
// Construction / configuration
// ---------------------------------------------------------------------------
StreamingDecision::StreamingDecision(const DecisionHead& head) { init(head); }

bool StreamingDecision::init(const DecisionHead& head) {
    head_  = &head;
    error_.clear();
    return head.ready();
}

void StreamingDecision::reset() {
    action_    = DecisionAction::ABSTAIN;
    committed_ = false;
    strength_  = 0;
    weak_run_  = 0;

    pending_action_ = DecisionAction::ABSTAIN;
    pending_streak_ = 0;

    win_action_.clear();
    win_conf_.clear();
    win_cap_    = 0;
    win_head_   = 0;
    win_filled_ = 0;

    steps_ = commits_ = changes_ = releases_ = weak_ = failures_ = 0;
    last_ = StreamEvent{};
}

// ---------------------------------------------------------------------------
// The two reducers
// ---------------------------------------------------------------------------
bool StreamingDecision::clears_switch_bar(const Candidate& c) const {
    if (!committed_) return true;          // an empty seat costs nothing to take
    if (c.action == action_) return true;  // the incumbent always qualifies
    return c.conf >= cfg_.switch_confidence;
}

StreamingDecision::Candidate
StreamingDecision::reduce_confirm(const DecisionResult& d) {
    Candidate c;
    c.action = d.action_type;
    c.conf   = d.confidence_score;
    // Confirm mode has no vote, so no vote margin. The head's own probability
    // margin is reported on the event as `observed_margin`; it is not promoted
    // to a vote margin, which would be a different quantity.
    c.margin = 0.0f;

    c.strong = (c.conf >= cfg_.min_confidence) && (d.margin >= cfg_.min_margin);
    if (!c.strong) {
        // A weak observation breaks the run. It cannot support a commitment and
        // it cannot start one.
        pending_action_ = DecisionAction::ABSTAIN;
        pending_streak_ = 0;
        c.strength = 0;
        return c;
    }

    if (!clears_switch_bar(c)) {
        // Strong, but not strong enough to displace the incumbent. It is NOT
        // weak evidence, so the run of weak observations is not advanced.
        pending_action_ = DecisionAction::ABSTAIN;
        pending_streak_ = 0;
        c.gated    = true;
        c.strength = 0;
        return c;
    }

    if (pending_action_ == c.action) ++pending_streak_;
    else { pending_action_ = c.action; pending_streak_ = 1; }
    c.strength = pending_streak_;
    return c;
}

StreamingDecision::Candidate
StreamingDecision::reduce_window(const DecisionResult& d) {
    if (win_cap_ != cfg_.window) {
        win_cap_ = cfg_.window;
        win_action_.assign(static_cast<size_t>(win_cap_), DecisionAction::ABSTAIN);
        win_conf_.assign(static_cast<size_t>(win_cap_), 0.0f);
        win_head_   = 0;
        win_filled_ = 0;
    }
    win_action_[static_cast<size_t>(win_head_)] = d.action_type;
    win_conf_[static_cast<size_t>(win_head_)]   = d.confidence_score;
    win_head_ = (win_head_ + 1) % win_cap_;
    if (win_filled_ < win_cap_) ++win_filled_;

    std::vector<int32_t> votes(static_cast<size_t>(kActionCount), 0);
    std::vector<float>   sums(static_cast<size_t>(kActionCount), 0.0f);
    int32_t total = 0;
    for (int32_t i = 0; i < win_filled_; ++i) {
        const int32_t ai = static_cast<int32_t>(win_action_[static_cast<size_t>(i)]);
        if (ai < 0 || ai >= kActionCount) continue;
        ++votes[static_cast<size_t>(ai)];
        sums[static_cast<size_t>(ai)] += win_conf_[static_cast<size_t>(i)];
        ++total;
    }

    Candidate c;
    if (total == 0) return c;   // nothing observed yet: not strong, strength 0

    std::vector<int32_t> order;
    order.reserve(static_cast<size_t>(kActionCount));
    for (int32_t a = 0; a < kActionCount; ++a)
        if (votes[static_cast<size_t>(a)] > 0) order.push_back(a);
    // Most votes; then the higher summed confidence; then the lower action
    // index. Fully ordered, so a tie can never resolve differently on two runs
    // over the same data.
    std::sort(order.begin(), order.end(), [&](int32_t x, int32_t y) {
        const int32_t vx = votes[static_cast<size_t>(x)];
        const int32_t vy = votes[static_cast<size_t>(y)];
        if (vx != vy) return vx > vy;
        const float sx = sums[static_cast<size_t>(x)];
        const float sy = sums[static_cast<size_t>(y)];
        if (sx != sy) return sx > sy;
        return x < y;
    });

    const int32_t win    = order[0];
    const int32_t runner = order.size() > 1 ? order[1] : -1;

    c.action   = static_cast<DecisionAction>(win);
    c.strength = votes[static_cast<size_t>(win)];
    c.conf     = sums[static_cast<size_t>(win)] / static_cast<float>(c.strength);
    const int32_t rv = runner >= 0 ? votes[static_cast<size_t>(runner)] : 0;
    c.margin = static_cast<float>(c.strength - rv) / static_cast<float>(total);

    // The winner must clear the quorum (confirm_steps, reused as the vote
    // quorum so there is one knob for "how much evidence is enough"), the
    // confidence bar and the margin bar.
    c.strong = (c.strength >= cfg_.confirm_steps) &&
               (c.conf     >= cfg_.min_confidence) &&
               (c.margin   >= cfg_.min_margin);

    if (c.strong && !clears_switch_bar(c)) {
        c.gated    = true;
        c.strength = 0;
    }
    return c;
}

// ---------------------------------------------------------------------------
// The one state machine
//
// `c` is already reduced; this function holds NO accumulator of its own beyond
// the commitment, so a change to either reducer cannot desynchronise it.
// ---------------------------------------------------------------------------
void StreamingDecision::apply(const Candidate& c, StreamEvent& e) {
    e.action    = action_;
    e.committed = committed_;
    e.previous  = action_;
    e.first     = false;
    e.changed   = false;
    e.strength  = 0;
    e.vote_margin = c.margin;

    // ---- weak: not enough evidence this step -------------------------------
    if (!c.strong) {
        ++weak_run_;
        strength_ = 0;
        e.strength = 0;
        if (committed_ && cfg_.release_after_weak > 0 &&
            weak_run_ >= cfg_.release_after_weak) {
            e.previous  = action_;
            e.action    = DecisionAction::ABSTAIN;
            e.changed   = true;
            e.committed = false;
            action_     = DecisionAction::ABSTAIN;
            committed_  = false;
            ++releases_;
            e.reason = "released: evidence evaporated";
        } else {
            e.reason = committed_ ? "weak observation; commitment held"
                                  : "weak observation";
        }
        return;
    }

    weak_run_ = 0;

    // ---- gated: strong, but it will not displace the incumbent -------------
    if (c.gated) {
        strength_ = 0;
        e.strength = 0;
        e.reason = "challenger below the switch bar; commitment held";
        return;
    }

    // ---- the incumbent is confirmed again ----------------------------------
    if (committed_ && c.action == action_) {
        strength_  = c.strength;
        e.strength = strength_;
        e.reason   = "held";
        return;
    }

    // ---- an alternative is accumulating ------------------------------------
    // ABSTAIN is accumulated here like any other candidate, so a release costs
    // exactly the same evidence as a switch. A single strong ABSTAIN must not
    // be able to unseat a commitment that took N observations to earn.
    strength_  = c.strength;
    e.strength = c.strength;
    if (c.strength < cfg_.confirm_steps) {
        e.reason = committed_ ? "challenger below quorum; commitment held"
                              : "awaiting confirmation";
        return;
    }

    // ---- a strong ABSTAIN is not a commitment ------------------------------
    // ABSTAIN is the absence of a position, so "committing" to it is a category
    // error: a committed filter holding ABSTAIN would report `committed = true`
    // with nothing held, and its `changed` events would stop matching
    // `commits + changes + releases` — the identity Part D pins.
    //
    // So a quorum of ABSTAIN is handled as what it actually is:
    //   * already committed -> a RELEASE (the head is clearly saying "no
    //     action", which is a different thing from saying nothing at all);
    //   * not committed     -> nothing to do.
    // Reaching here means the switch-bar gate above was already cleared, so a
    // weak ABSTAIN cannot unseat a commitment: hysteresis protects the
    // incumbent here exactly as it does against a weak BUY.
    if (c.action == DecisionAction::ABSTAIN) {
        if (committed_) {
            e.previous  = action_;
            e.action    = DecisionAction::ABSTAIN;
            e.changed   = true;
            e.committed = false;
            action_     = DecisionAction::ABSTAIN;
            committed_  = false;
            strength_   = 0;
            e.strength  = 0;
            ++releases_;
            e.reason = "released: the head is confidently ABSTAIN";
        } else {
            e.reason = "candidate is ABSTAIN; nothing to commit";
        }
        return;
    }

    // ---- COMMIT -------------------------------------------------------------
    // Reached only with a non-ABSTAIN candidate that has cleared the quorum, so
    // `changed` is unconditionally true here: either the seat was empty and
    // ABSTAIN is being left, or the incumbent is a different action (an equal
    // action would have taken the "held" branch above). That is what makes
    // `changed_events == commits + changes + releases` hold exactly.
    const bool was_committed = committed_;
    e.previous  = action_;
    e.action    = c.action;
    e.committed = true;
    e.first     = !was_committed;
    e.changed   = true;
    action_     = c.action;
    committed_  = true;
    strength_   = c.strength;
    e.strength  = strength_;
    if (was_committed) { ++changes_; e.reason = "switched"; }
    else               { ++commits_; e.reason = "committed"; }
}

// ---------------------------------------------------------------------------
// Refusal — the fail-closed path. NOTHING moves: not the step count, not the
// streak, not the commitment, not the window.
// ---------------------------------------------------------------------------
void StreamingDecision::refuse(const char* why, StreamEvent& out) {
    StreamEvent e;
    e.step      = steps_;          // unchanged: no observation was recorded
    e.valid     = false;
    e.action    = action_;
    e.previous  = action_;
    e.committed = committed_;
    e.strength  = strength_;
    e.reason    = why;
    ++failures_;
    last_ = e;
    out   = e;
}

// ---------------------------------------------------------------------------
// The hot paths
// ---------------------------------------------------------------------------
bool StreamingDecision::push(const float* hidden, StreamEvent& out) {
    if (head_ == nullptr)          { error_ = "no head attached";        refuse("no head attached", out); return false; }
    if (hidden == nullptr)         { error_ = "null hidden state";       refuse("null hidden state", out); return false; }
    if (!head_->ready())           { error_ = "head is not ready";       refuse("head is not ready", out); return false; }

    const DecisionResult d = head_->decide(hidden);
    if (d.routing == "error") {
        error_ = "head refused to observe";
        refuse("head refused to observe", out);
        return false;
    }
    error_.clear();
    push_decision(d, out);
    return true;
}

bool StreamingDecision::push(const Tensor& hidden, StreamEvent& out) {
    if (hidden.dtype() != DType::F32) { error_ = "hidden is not F32"; refuse("hidden is not F32", out); return false; }
    if (head_ == nullptr)             { error_ = "no head attached";  refuse("no head attached", out);  return false; }
    if (hidden.numel() != static_cast<int64_t>(head_->hidden_size())) {
        error_ = "hidden width mismatch";
        refuse("hidden width mismatch", out);
        return false;
    }
    return push(hidden.f32(), out);
}

void StreamingDecision::push_decision(const DecisionResult& d, StreamEvent& out) {
    ++steps_;

    StreamEvent e;
    e.step            = steps_;
    e.valid           = true;
    e.observed_action = d.action_type;
    e.observed_conf   = d.confidence_score;
    e.observed_margin = d.margin;

    const Candidate c = (cfg_.mode == StreamMode::Window)
                            ? reduce_window(d)
                            : reduce_confirm(d);
    if (!c.strong) ++weak_;

    apply(c, e);
    last_ = e;
    out   = e;
}

bool StreamingDecision::push_batch(const float* H, int32_t B,
                                   std::vector<StreamEvent>& out,
                                   BatchStats* stats) {
    out.clear();
    if (head_ == nullptr) { error_ = "push_batch: no head attached"; return false; }
    if (H == nullptr || B <= 0) { error_ = "push_batch: null or empty batch"; return false; }
    if (!head_->ready())  { error_ = "push_batch: head is not ready"; return false; }

    std::vector<DecisionResult> ds;
    if (!head_->decide_batch(H, B, ds, stats)) {
        error_ = "push_batch: decide_batch refused the batch";
        return false;
    }
    if (static_cast<int32_t>(ds.size()) != B) {
        error_ = "push_batch: short batch";
        return false;
    }

    out.reserve(static_cast<size_t>(B));
    for (int32_t b = 0; b < B; ++b) {
        StreamEvent e;
        push_decision(ds[static_cast<size_t>(b)], e);
        out.push_back(e);
    }
    error_.clear();
    return true;
}

bool StreamingDecision::push_batch(const Tensor& H, std::vector<StreamEvent>& out,
                                   BatchStats* stats) {
    if (H.dtype() != DType::F32 || H.shape().size() != 2) {
        out.clear();
        error_ = "push_batch: expected an F32 [B, E] tensor";
        return false;
    }
    if (head_ == nullptr) {
        out.clear();
        error_ = "push_batch: no head attached";
        return false;
    }
    if (H.shape()[1] != static_cast<int64_t>(head_->hidden_size())) {
        out.clear();
        error_ = "push_batch: hidden width mismatch";
        return false;
    }
    return push_batch(H.f32(), static_cast<int32_t>(H.shape()[0]), out, stats);
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------
std::string StreamingDecision::to_json() const {
    std::string s = "{\"config\":";
    s += cfg_.to_json();
    s += ",\"steps\":"    + std::to_string(steps_);
    s += ",\"commits\":"  + std::to_string(commits_);
    s += ",\"changes\":"  + std::to_string(changes_);
    s += ",\"releases\":" + std::to_string(releases_);
    s += ",\"weak\":"     + std::to_string(weak_);
    s += ",\"failures\":" + std::to_string(failures_);
    s += ",\"action\":";
    append_action(s, action_);
    s += committed_ ? ",\"committed\":true" : ",\"committed\":false";
    s += ",\"strength\":" + std::to_string(strength_);
    s += ",\"ready\":";
    s += ready() ? "true" : "false";
    if (!error_.empty()) {
        s += ",\"error\":\"" + heads_detail::json_escape(error_) + "\"";
    }
    s += "}";
    return s;
}

} // namespace omniseed
