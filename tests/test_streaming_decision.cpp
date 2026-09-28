// =============================================================================
//  OmniSeed — tests/test_streaming_decision.cpp
//
//  MILESTONE 9 — Phase 2.2, streaming decisions.
//
//  THE CONTRACT THIS FILE EXISTS TO PIN. A filter that is supposed to stop a
//  control loop chattering must itself be impossible to fool. Three properties
//  are asserted, and each of them is a way the filter could silently betray a
//  caller:
//
//    1. NO COMMITMENT WITHOUT CONSECUTIVE EVIDENCE (Part A/B). A single blip,
//       a tie, or a run broken in the middle must leave the commitment exactly
//       where it was.
//    2. NO REPLACEMENT WITHOUT A REPORT (Part A/B/D). Every change carries
//       `changed = true` and names the action it replaced. A consumer that
//       holds a position must be able to undo it; going quiet would leave it
//       holding a stale one.
//    3. NO AGREEMENT FROM A DEAD HEAD (Part C). A failed observation records
//       NOTHING — not a step, not a streak, not a renewal of the commitment.
//       An unreadable signal is not agreement.
//
//  Part C is the JOINT: the §36 batched readout and the per-row readout must
//  produce the SAME event stream, byte for byte, which is what makes one filter
//  usable from either path.
//
//  Fully offline and UNGATED: the state-machine parts need no head at all, and
//  the one real-data part uses the COMMITTED h[E] fixtures. No model, no
//  `.venv`, no network.
// =============================================================================
#include "omniseed/core/batch_gemm.h"
#include "omniseed/core/platform.h"
#include "omniseed/core/tensor.h"
#include "omniseed/decision_head.h"
#include "omniseed/streaming_decision.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace omniseed;

static int g_passed = 0;
static int g_failed = 0;
static int g_skipped = 0;
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
#define SKIP(msg)                                                          \
    do {                                                                   \
        ++g_skipped;                                                       \
        platform::log_info("SKIP  %s  (%s)", g_current.c_str(), msg);      \
    } while (0)

namespace {

bool close(float a, float b, float rel, float absv) {
    return std::fabs(a - b) <= absv + rel * std::fabs(b);
}

bool has(const std::string& s, const char* needle) {
    return s.find(needle) != std::string::npos;
}

// A hand-built head output. `push_decision` ignores `routing`, so these drive
// the state machine with no head and no model at all.
DecisionResult mk(DecisionAction a, float conf, float margin = 0.50f) {
    DecisionResult d;
    d.action_type      = a;
    d.confidence_score = conf;
    d.margin           = margin;
    d.routing          = "self";
    d.matvecs          = 1;
    return d;
}

StreamingConfig cfg_confirm(int32_t steps, float min_c = 0.50f, float sw = 0.60f) {
    StreamingConfig c;
    c.mode           = StreamMode::Confirm;
    c.confirm_steps  = steps;
    c.min_confidence = min_c;
    c.switch_confidence = sw;
    return c;
}

StreamingConfig cfg_window(int32_t win, int32_t quorum, float min_c = 0.50f, float sw = 0.60f) {
    StreamingConfig c;
    c.mode           = StreamMode::Window;
    c.window         = win;
    c.confirm_steps  = quorum;
    c.min_confidence = min_c;
    c.switch_confidence = sw;
    return c;
}

// Row-major [n, E] float32 fixture.
std::vector<float> load_f32(const std::string& path, int32_t E, int32_t& n) {
    n = 0;
    std::vector<float> v;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return v;
    std::fseek(f, 0, SEEK_END);
    const long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (sz <= 0 || (sz % 4) != 0) { std::fclose(f); return v; }
    v.resize(static_cast<size_t>(sz) / sizeof(float));
    const size_t got = std::fread(v.data(), sizeof(float), v.size(), f);
    std::fclose(f);
    v.resize(got);
    n = static_cast<int32_t>(v.size() / static_cast<size_t>(E));
    return v;
}

// A deterministic stream of pseudo-random decisions, so Part D's invariants are
// exercised over a long, varied sequence without a generator dependency.
struct Lcg {
    uint64_t s;
    explicit Lcg(uint64_t seed) : s(seed) {}
    uint32_t next() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<uint32_t>(s >> 33);
    }
    float unit() { return static_cast<float>(next() % 100000u) / 100000.0f; }
};

// A STICKY action stream: with probability `stick` the previous action repeats.
//
// This matters. A purely uniform stream almost never produces the run of
// consecutive agreements a commit needs, so the commit and switch paths stay
// unvisited and the invariants are asserted over a stream that only ever walks
// the "nothing happened" branch — a test that proves nothing. Measured: uniform
// gave 2 commits and 0 switches over 10,000 steps; sticky gives both.
DecisionAction sticky_action(Lcg& rng, DecisionAction prev, float stick) {
    if (rng.unit() < stick) return prev;
    return static_cast<DecisionAction>(
        rng.next() % static_cast<uint32_t>(DecisionAction::COUNT));
}

// ---------------------------------------------------------------------------
// PART A — the Confirm state machine
// ---------------------------------------------------------------------------
void part_a() {
    platform::log_info("--- Part A: the Confirm state machine ---");

    // A1 — no commitment before the run is complete.
    {
        TEST("A1 confirm: three consecutive agreements commit on the third");
        StreamingDecision s;
        s.set_config(cfg_confirm(3));
        StreamEvent e;

        s.push_decision(mk(DecisionAction::BUY, 0.90f), e);
        CHECK(!e.committed);
        CHECK(e.action == DecisionAction::ABSTAIN);
        CHECK(e.strength == 1);
        CHECK(!e.first && !e.changed);
        CHECK(e.valid);
        CHECK(e.reason == "awaiting confirmation");

        s.push_decision(mk(DecisionAction::BUY, 0.90f), e);
        CHECK(!e.committed);
        CHECK(e.strength == 2);

        s.push_decision(mk(DecisionAction::BUY, 0.90f), e);
        CHECK(e.committed);
        CHECK(e.action == DecisionAction::BUY);
        CHECK(e.first);
        CHECK(e.changed);
        CHECK(e.previous == DecisionAction::ABSTAIN);
        CHECK(e.strength == 3);
        CHECK(e.reason == "committed");
        CHECK(s.committed());
        CHECK(s.action() == DecisionAction::BUY);
        CHECK(s.steps() == 3);
        CHECK(s.commits() == 1);
        CHECK(s.changes() == 0);
        CHECK(s.weak() == 0);
    }

    // A2 — a blip in the middle of a run resets it.
    {
        TEST("A2 confirm: a single blip resets the run and commits nothing");
        StreamingDecision s;
        s.set_config(cfg_confirm(3));
        StreamEvent e;
        const DecisionAction seq[] = {DecisionAction::BUY, DecisionAction::BUY,
                                      DecisionAction::HOLD, DecisionAction::BUY,
                                      DecisionAction::BUY};
        for (DecisionAction a : seq) {
            s.push_decision(mk(a, 0.90f), e);
            CHECK(!e.committed);
        }
        CHECK(e.strength == 2);       // BUY, BUY, [HOLD], BUY, BUY
        CHECK(s.commits() == 0);
        CHECK(s.committed() == false);
        CHECK(s.steps() == 5);
    }

    // A3 — confirm_steps == 1 commits immediately, and the normaliser keeps it
    //      at >= 1 when a caller asks for 0.
    {
        TEST("A3 confirm: confirm_steps=1 commits on the first observation");
        StreamingDecision s;
        s.set_config(cfg_confirm(1));
        StreamEvent e;
        s.push_decision(mk(DecisionAction::SELL, 0.80f), e);
        CHECK(e.committed);
        CHECK(e.first && e.changed);
        CHECK(e.action == DecisionAction::SELL);
        CHECK(s.commits() == 1);

        StreamingConfig bad;
        bad.confirm_steps = 0;
        CHECK(bad.normalised().confirm_steps == 1);
        bad.confirm_steps = -7;
        CHECK(bad.normalised().confirm_steps == 1);
    }

    // A4 — HYSTERESIS. A challenger below the switch bar does not even start to
    //      displace the incumbent, and it is NOT counted as weak evidence.
    {
        TEST("A4 hysteresis: a challenger below the switch bar is refused");
        StreamingDecision s;
        s.set_config(cfg_confirm(3, 0.50f, 0.60f));
        StreamEvent e;
        for (int i = 0; i < 3; ++i) s.push_decision(mk(DecisionAction::BUY, 0.90f), e);
        CHECK(s.committed());
        CHECK(s.weak() == 0);

        s.push_decision(mk(DecisionAction::HOLD, 0.55f), e);   // < 0.60
        CHECK(e.committed);
        CHECK(e.action == DecisionAction::BUY);                // incumbent kept
        CHECK(e.observed_action == DecisionAction::HOLD);      // the head did say HOLD
        CHECK(e.strength == 0);
        CHECK(!e.changed);
        CHECK(has(e.reason, "switch bar"));
        CHECK(s.weak() == 0);            // gated is NOT weak
        CHECK(s.changes() == 0);
        CHECK(s.steps() == 4);

        // The run of agreement was broken by the challenge, so it restarts.
        s.push_decision(mk(DecisionAction::BUY, 0.90f), e);
        CHECK(e.strength == 1);
        CHECK(e.reason == "held");
    }

    // A5 — the switch itself, and the report that must accompany it.
    {
        TEST("A5 hysteresis: a challenger above the bar switches, and says so");
        StreamingDecision s;
        s.set_config(cfg_confirm(3, 0.50f, 0.60f));
        StreamEvent e;
        for (int i = 0; i < 3; ++i) s.push_decision(mk(DecisionAction::BUY, 0.90f), e);
        CHECK(s.action() == DecisionAction::BUY);

        s.push_decision(mk(DecisionAction::HOLD, 0.70f), e);
        CHECK(!e.changed);
        CHECK(has(e.reason, "below quorum"));
        CHECK(e.action == DecisionAction::BUY);

        s.push_decision(mk(DecisionAction::HOLD, 0.70f), e);
        CHECK(!e.changed);

        s.push_decision(mk(DecisionAction::HOLD, 0.70f), e);
        CHECK(e.changed);
        CHECK(!e.first);
        CHECK(e.previous == DecisionAction::BUY);
        CHECK(e.action == DecisionAction::HOLD);
        CHECK(e.reason == "switched");
        CHECK(s.changes() == 1);
        CHECK(s.commits() == 1);
        CHECK(s.action() == DecisionAction::HOLD);
    }

    // A6 — the config normaliser, field by field.
    {
        TEST("A6 config: every out-of-range field is repaired, upward only");
        StreamingConfig c;
        c.mode = StreamMode::COUNT;              // not a mode
        c.confirm_steps = 5;
        c.window = 2;                            // < quorum
        c.min_confidence = 0.70f;
        c.switch_confidence = 0.20f;             // < min: the hysteresis invariant
        c.min_margin = 2.5f;
        c.release_after_weak = -3;
        const StreamingConfig n = c.normalised();
        CHECK(n.mode == StreamMode::Confirm);
        CHECK(n.confirm_steps == 5);
        CHECK(n.window == 5);                    // grown to the quorum
        CHECK(close(n.min_confidence, 0.70f, 0.f, 1e-6f));
        CHECK(close(n.switch_confidence, 0.70f, 0.f, 1e-6f));  // raised, never lowered
        CHECK(close(n.min_margin, 1.0f, 0.f, 1e-6f));
        CHECK(n.release_after_weak == 0);

        // A NaN cannot be clamped into range, so it falls back to the default.
        StreamingConfig nan_c;
        nan_c.min_confidence = std::nanf("");
        nan_c.switch_confidence = std::nanf("");
        const StreamingConfig nn = nan_c.normalised();
        CHECK(close(nn.min_confidence, 0.50f, 0.f, 1e-6f));
        CHECK(close(nn.switch_confidence, 0.60f, 0.f, 1e-6f));

        // The invariant holds in the other direction too: a HIGH switch bar is
        // left alone. The repair is one-way by design.
        StreamingConfig hi;
        hi.min_confidence = 0.30f;
        hi.switch_confidence = 0.95f;
        CHECK(close(hi.normalised().switch_confidence, 0.95f, 0.f, 1e-6f));

        // An out-of-range mode is refused, and the fallback is the more
        // conservative of the two modes.
        CHECK(StreamingConfig{}.normalised().mode == StreamMode::Confirm);
    }

    // A7 — a weak observation is weak, and says so.
    {
        TEST("A7 weak: below min_confidence nothing is committed or advanced");
        StreamingDecision s;
        s.set_config(cfg_confirm(3));
        StreamEvent e;
        s.push_decision(mk(DecisionAction::BUY, 0.30f), e);
        CHECK(!e.committed);
        CHECK(e.strength == 0);
        CHECK(e.reason == "weak observation");
        CHECK(s.weak() == 1);
        CHECK(s.strength() == 0);

        // min_margin is a separate bar and is honoured separately.
        StreamingConfig c = cfg_confirm(1);
        c.min_margin = 0.20f;
        StreamingDecision m;
        m.set_config(c);
        m.push_decision(mk(DecisionAction::BUY, 0.90f, 0.05f), e);
        CHECK(!e.committed);
        CHECK(m.weak() == 1);
        m.push_decision(mk(DecisionAction::BUY, 0.90f, 0.25f), e);
        CHECK(e.committed);
    }

    // A8/A9/A10 — RELEASE, and its off switch.
    {
        TEST("A8 release: a commitment whose evidence evaporated is released");
        StreamingDecision s;
        StreamingConfig c = cfg_confirm(1);
        c.release_after_weak = 2;
        s.set_config(c);
        StreamEvent e;
        s.push_decision(mk(DecisionAction::BUY, 0.90f), e);
        CHECK(s.committed());

        s.push_decision(mk(DecisionAction::BUY, 0.10f), e);
        CHECK(s.committed());                 // one weak step is not enough
        CHECK(e.action == DecisionAction::BUY);
        CHECK(has(e.reason, "commitment held"));

        s.push_decision(mk(DecisionAction::BUY, 0.10f), e);
        CHECK(!s.committed());
        CHECK(e.changed);
        CHECK(e.previous == DecisionAction::BUY);
        CHECK(e.action == DecisionAction::ABSTAIN);
        CHECK(!e.first);
        CHECK(has(e.reason, "released"));
        CHECK(s.releases() == 1);
    }

    {
        TEST("A9 release: release_after_weak=0 holds the commitment forever");
        StreamingDecision s;
        StreamingConfig c = cfg_confirm(1);
        c.release_after_weak = 0;
        s.set_config(c);
        StreamEvent e;
        s.push_decision(mk(DecisionAction::BUY, 0.90f), e);
        for (int i = 0; i < 200; ++i) s.push_decision(mk(DecisionAction::BUY, 0.01f), e);
        CHECK(s.committed());
        CHECK(s.action() == DecisionAction::BUY);
        CHECK(s.releases() == 0);
        CHECK(s.weak() == 200);
        CHECK(!e.changed);
    }

    {
        TEST("A10 release: a strong agreement resets the weak run");
        StreamingDecision s;
        StreamingConfig c = cfg_confirm(1);
        c.release_after_weak = 3;
        s.set_config(c);
        StreamEvent e;
        s.push_decision(mk(DecisionAction::BUY, 0.90f), e);
        s.push_decision(mk(DecisionAction::BUY, 0.01f), e);
        s.push_decision(mk(DecisionAction::BUY, 0.01f), e);
        s.push_decision(mk(DecisionAction::BUY, 0.90f), e);   // resets the run
        s.push_decision(mk(DecisionAction::BUY, 0.01f), e);
        s.push_decision(mk(DecisionAction::BUY, 0.01f), e);
        CHECK(s.committed());                 // never reached 3 in a row
        CHECK(s.releases() == 0);
        s.push_decision(mk(DecisionAction::BUY, 0.01f), e);
        CHECK(!s.committed());
        CHECK(s.releases() == 1);
    }

    // A11 — ABSTAIN is not a position, so it is never a commitment.
    //
    //      This is the rule that makes `changed == commits + changes +
    //      releases` hold. Without it a strong run of ABSTAIN would "commit",
    //      leaving a filter that reports `committed = true` while holding
    //      nothing, and a `first` event with no `changed` event to match it.
    {
        TEST("A11 abstain: a strong ABSTAIN releases, and never commits");
        // Nothing held: a strong ABSTAIN is a no-op.
        StreamingDecision s;
        s.set_config(cfg_confirm(2));
        StreamEvent e;
        for (int i = 0; i < 6; ++i) {
            s.push_decision(mk(DecisionAction::ABSTAIN, 0.95f), e);
            CHECK(!e.committed);
            CHECK(!e.changed);
            CHECK(e.action == DecisionAction::ABSTAIN);
        }
        CHECK(s.commits() == 0);
        CHECK(s.releases() == 0);
        CHECK(has(e.reason, "nothing to commit"));

        // Something held: a strong ABSTAIN is a release.
        StreamingDecision h;
        h.set_config(cfg_confirm(2));
        for (int i = 0; i < 2; ++i) h.push_decision(mk(DecisionAction::SELL, 0.95f), e);
        CHECK(h.committed());
        CHECK(h.action() == DecisionAction::SELL);

        h.push_decision(mk(DecisionAction::ABSTAIN, 0.95f), e);
        CHECK(h.committed());                       // one step is not a run
        CHECK(!e.changed);

        h.push_decision(mk(DecisionAction::ABSTAIN, 0.95f), e);
        CHECK(!h.committed());
        CHECK(e.changed);
        CHECK(e.previous == DecisionAction::SELL);
        CHECK(e.action == DecisionAction::ABSTAIN);
        CHECK(!e.first);
        CHECK(h.releases() == 1);
        CHECK(h.commits() == 1);                    // the SELL, and only it
        CHECK(has(e.reason, "confidently ABSTAIN"));

        // Hysteresis protects the incumbent here too: a weak ABSTAIN cannot
        // unseat a commitment, it only counts as weak evidence.
        StreamingDecision g;
        g.set_config(cfg_confirm(2, 0.50f, 0.60f));
        for (int i = 0; i < 2; ++i) g.push_decision(mk(DecisionAction::BUY, 0.95f), e);
        CHECK(g.committed());
        g.push_decision(mk(DecisionAction::ABSTAIN, 0.40f), e);
        CHECK(g.committed());
        CHECK(!e.changed);
        CHECK(g.releases() == 0);
        CHECK(g.weak() == 1);
    }
}

// ---------------------------------------------------------------------------
// PART B — the Window (voting) reducer
// ---------------------------------------------------------------------------
void part_b() {
    platform::log_info("--- Part B: the Window vote ---");

    {
        TEST("B1 window: a quorum of votes commits, with a vote margin");
        StreamingDecision s;
        s.set_config(cfg_window(3, 2));
        StreamEvent e;
        s.push_decision(mk(DecisionAction::BUY, 0.90f), e);
        CHECK(!e.committed);
        s.push_decision(mk(DecisionAction::BUY, 0.80f), e);
        CHECK(e.committed);                       // 2 of 2 so far
        CHECK(e.action == DecisionAction::BUY);
        CHECK(e.strength == 2);
        CHECK(close(e.vote_margin, 1.0f, 0.f, 1e-6f));   // 2-0 over 2

        s.push_decision(mk(DecisionAction::HOLD, 0.90f), e);
        CHECK(e.action == DecisionAction::BUY);          // 2-1
        CHECK(e.strength == 2);
        CHECK(close(e.vote_margin, 1.0f / 3.0f, 1e-5f, 1e-6f));
        CHECK(close(e.observed_conf, 0.90f, 0.f, 1e-6f));
        CHECK(s.window_filled() == 3);
    }

    {
        TEST("B2 window: the window slides, and the winner can change");
        StreamingDecision s;
        s.set_config(cfg_window(3, 2));
        StreamEvent e;
        s.push_decision(mk(DecisionAction::BUY, 0.90f), e);
        s.push_decision(mk(DecisionAction::BUY, 0.90f), e);
        s.push_decision(mk(DecisionAction::HOLD, 0.90f), e);
        CHECK(s.action() == DecisionAction::BUY);

        s.push_decision(mk(DecisionAction::HOLD, 0.90f), e);   // [HOLD,BUY,HOLD]
        CHECK(e.changed);
        CHECK(e.previous == DecisionAction::BUY);
        CHECK(e.action == DecisionAction::HOLD);
        CHECK(e.strength == 2);
        CHECK(s.changes() == 1);
        CHECK(s.window_filled() == 3);                          // never grows past window
    }

    {
        TEST("B3 window: a fragmented vote fails the quorum and is weak");
        StreamingDecision s;
        s.set_config(cfg_window(5, 3));
        StreamEvent e;
        s.push_decision(mk(DecisionAction::BUY, 0.90f), e);
        s.push_decision(mk(DecisionAction::BUY, 0.90f), e);
        s.push_decision(mk(DecisionAction::HOLD, 0.90f), e);
        s.push_decision(mk(DecisionAction::SELL, 0.90f), e);
        s.push_decision(mk(DecisionAction::SELL, 0.90f), e);   // 2/2/1
        CHECK(!e.committed);
        CHECK(e.strength == 0);
        CHECK(e.reason == "weak observation");
        CHECK(s.commits() == 0);
        CHECK(s.weak() == 5);
    }

    {
        TEST("B4 window: the result depends on the vote, not on the order");
        const DecisionAction o1[] = {DecisionAction::BUY, DecisionAction::BUY,
                                     DecisionAction::HOLD, DecisionAction::HOLD,
                                     DecisionAction::HOLD};
        const DecisionAction o2[] = {DecisionAction::HOLD, DecisionAction::BUY,
                                     DecisionAction::HOLD, DecisionAction::BUY,
                                     DecisionAction::HOLD};
        StreamEvent e1, e2;
        StreamingDecision s1, s2;
        s1.set_config(cfg_window(5, 3));
        s2.set_config(cfg_window(5, 3));
        for (DecisionAction a : o1) s1.push_decision(mk(a, 0.90f), e1);
        for (DecisionAction a : o2) s2.push_decision(mk(a, 0.90f), e2);
        CHECK(e1.action == DecisionAction::HOLD);
        CHECK(e2.action == DecisionAction::HOLD);
        CHECK(e1.strength == e2.strength);
        CHECK(e1.strength == 3);
        CHECK(close(e1.vote_margin, e2.vote_margin, 0.f, 1e-6f));
        CHECK(s1.commits() == s2.commits());
        CHECK(s1.steps() == s2.steps());
    }

    {
        TEST("B5 window: resizing the window resets the ring, not the commitment");
        StreamingDecision s;
        s.set_config(cfg_window(3, 2));
        StreamEvent e;
        s.push_decision(mk(DecisionAction::BUY, 0.90f), e);
        s.push_decision(mk(DecisionAction::BUY, 0.90f), e);
        CHECK(s.committed());
        CHECK(s.window_filled() == 2);

        StreamingConfig bigger = s.config();
        bigger.window = 5;
        s.set_config(bigger);
        s.push_decision(mk(DecisionAction::BUY, 0.90f), e);
        CHECK(s.window_filled() == 1);      // the ring was rebuilt
        CHECK(s.committed());               // the commitment was not
        // One vote of five is below the quorum, so the candidate is not strong
        // and the reported strength is 0. The vote count itself is visible as
        // window_filled(); `strength` answers "how much evidence moves the
        // filter", and one vote does not.
        CHECK(s.strength() == 0);
        CHECK(s.action() == DecisionAction::BUY);
    }

    {
        TEST("B6 window: hysteresis applies to the vote winner too");
        StreamingDecision s;
        s.set_config(cfg_window(3, 3, 0.50f, 0.60f));
        StreamEvent e;
        for (int i = 0; i < 3; ++i) s.push_decision(mk(DecisionAction::HOLD, 0.90f), e);
        CHECK(s.action() == DecisionAction::HOLD);

        // Three BUY votes at 0.55: unanimous, over quorum, but under the bar.
        for (int i = 0; i < 3; ++i) {
            s.push_decision(mk(DecisionAction::BUY, 0.55f), e);
            CHECK(!e.changed);
        }
        CHECK(s.action() == DecisionAction::HOLD);
        CHECK(has(e.reason, "switch bar"));
        // Four weak steps, not two. Two while the HOLD window filled to quorum
        // (steps 1-2), and two more while the BUY votes grew but NEITHER side
        // had three (steps 4-5: [H,H,B] and [H,B,B]). Only the sixth step
        // produced a unanimous BUY window — and THAT was gated. A gated
        // candidate is never counted as weak, and a weak one can never switch,
        // so neither can unseat HOLD.
        CHECK(s.weak() == 4);
        CHECK(s.changes() == 0);
        CHECK(s.releases() == 0);
    }
}

// ---------------------------------------------------------------------------
// PART C — the joint: the real head, the batch path, and fail-closed
// ---------------------------------------------------------------------------
void part_c() {
    platform::log_info("--- Part C: the joint with the real head ---");

    // C1 — the §36 batch path and the per-row path must produce the same
    //      stream. This is what makes one filter drivable from either.
    {
        TEST("C1 batch and per-row drive the filter to the same event stream");
        DecisionHead head;
        CHECK(head.init(768, 4242u));
        CHECK(head.ready());
        head.set_batch_kernel(BatchKernel::Scalar);

        int32_t n = 0;
        const std::vector<float> H =
            load_f32("tests/fixtures/head_calibration/trading/hidden.f32", 768, n);
        if (n < 4) {
            SKIP("held-out h[E] fixture missing");
        } else {
            const int32_t B = n < 369 ? n : 369;

            StreamingDecision via_batch;
            via_batch.set_config(cfg_confirm(3));
            via_batch.init(head);

            StreamingDecision via_row;
            via_row.set_config(cfg_confirm(3));
            via_row.init(head);

            std::vector<StreamEvent> ev;
            BatchStats stats;
            CHECK(via_batch.push_batch(H.data(), B, ev, &stats));
            CHECK(static_cast<int32_t>(ev.size()) == B);
            CHECK(stats.rows == B);

            int32_t mismatches = 0;
            for (int32_t b = 0; b < B; ++b) {
                StreamEvent r;
                CHECK(via_row.push(H.data() + static_cast<size_t>(b) * 768u, r));
                if (r.step != ev[static_cast<size_t>(b)].step ||
                    r.action != ev[static_cast<size_t>(b)].action ||
                    r.observed_action != ev[static_cast<size_t>(b)].observed_action ||
                    r.committed != ev[static_cast<size_t>(b)].committed ||
                    r.first != ev[static_cast<size_t>(b)].first ||
                    r.changed != ev[static_cast<size_t>(b)].changed ||
                    r.strength != ev[static_cast<size_t>(b)].strength ||
                    r.observed_conf != ev[static_cast<size_t>(b)].observed_conf ||
                    r.vote_margin != ev[static_cast<size_t>(b)].vote_margin ||
                    r.reason != ev[static_cast<size_t>(b)].reason) {
                    ++mismatches;
                }
            }
            CHECK(mismatches == 0);
            CHECK(via_batch.to_json() == via_row.to_json());
            CHECK(via_batch.steps() == B);
            CHECK(via_row.steps() == B);

            // How much work was there for the filter to do? Count the distinct
            // raw head outputs and the raw flips. On these real hidden states a
            // SEEDED head is nearly constant, so the filter has little to
            // remove — an honest observation about an untrained head, not a
            // property of the filter. Printed, not asserted, because it is a
            // fact about the placeholder weights.
            int32_t distinct = 0;
            int64_t  flips    = 0;
            for (int32_t b = 0; b < B; ++b) {
                bool seen = false;
                for (int32_t p = 0; p < b; ++p)
                    if (ev[static_cast<size_t>(p)].observed_action ==
                        ev[static_cast<size_t>(b)].observed_action) { seen = true; break; }
                if (!seen) ++distinct;
                if (b > 0 && ev[static_cast<size_t>(b)].observed_action !=
                             ev[static_cast<size_t>(b - 1)].observed_action) ++flips;
            }
            platform::log_info("      %d real rows: %d distinct head output(s), "
                               "%lld raw flip(s), commits=%lld changes=%lld weak=%lld",
                               B, distinct, static_cast<long long>(flips),
                               static_cast<long long>(via_batch.commits()),
                               static_cast<long long>(via_batch.changes()),
                               static_cast<long long>(via_batch.weak()));
        }
    }

    // C2 — a dead head is not agreement.
    {
        TEST("C2 fail-closed: a not-ready head records nothing at all");
        DecisionHead bad;                       // never init()ed
        CHECK(!bad.ready());
        StreamingDecision s(bad);
        CHECK(!s.ready());

        std::vector<float> h(768, 0.0f);
        StreamEvent e;
        CHECK(!s.push(h.data(), e));
        CHECK(!e.valid);
        CHECK(e.reason == "head is not ready");
        CHECK(s.steps() == 0);                  // NOT counted
        CHECK(s.failures() == 1);
        CHECK(!s.committed());
        CHECK(s.strength() == 0);
        CHECK(s.action() == DecisionAction::ABSTAIN);
        CHECK(has(s.to_json(), "not ready"));
    }

    // C3 — a failed observation in the MIDDLE of a stream must not disturb the
    //      commitment the stream has already earned.
    {
        TEST("C3 fail-closed: a failure mid-stream does not disturb the commitment");
        DecisionHead head;
        CHECK(head.init(768, 99u));
        StreamingDecision s;
        s.set_config(cfg_confirm(3));
        s.init(head);

        std::vector<float> h(768, 0.0f);
        StreamEvent e;
        for (int i = 0; i < 3; ++i) CHECK(s.push(h.data(), e));
        const bool was_committed = s.committed();
        const int64_t was_steps   = s.steps();
        const int32_t was_strength = s.strength();

        CHECK(!s.push(static_cast<const float*>(nullptr), e));
        CHECK(!e.valid);
        CHECK(e.reason == "null hidden state");
        CHECK(s.steps() == was_steps);
        CHECK(s.committed() == was_committed);
        CHECK(s.strength() == was_strength);
        CHECK(s.failures() == 1);

        // A width mismatch is refused the same way.
        Tensor wrong("wrong", {512}, DType::F32);
        CHECK(!s.push(wrong, e));
        CHECK(!e.valid);
        CHECK(has(e.reason, "mismatch"));
        CHECK(s.steps() == was_steps);
        CHECK(s.failures() == 2);
    }

    // C4 — the JSON a dashboard consumes must be parseable.
    {
        TEST("C4 JSON: well-formed, and never a bare nan or inf");
        StreamingDecision s;
        s.set_config(cfg_confirm(2));
        StreamEvent e;
        s.push_decision(mk(DecisionAction::BUY, 0.90f), e);
        s.push_decision(mk(DecisionAction::BUY, 0.90f), e);
        CHECK(e.committed);

        const std::string js = e.to_json();
        CHECK(!js.empty());
        CHECK(js.front() == '{');
        CHECK(js.back() == '}');
        CHECK(has(js, "\"step\":2"));
        CHECK(has(js, "\"valid\":true"));
        CHECK(has(js, "\"action\":\"BUY\""));
        CHECK(has(js, "\"previous\":\"ABSTAIN\""));
        CHECK(has(js, "\"committed\":true"));
        CHECK(has(js, "\"changed\":true"));
        CHECK(!has(js, ":nan"));       // a VALUE, not the substring "nan"
        CHECK(!has(js, ":inf"));
        CHECK(!has(js, "nan,"));
        CHECK(!has(js, "inf,"));

        const std::string sj = s.to_json();
        CHECK(sj.front() == '{');
        CHECK(sj.back() == '}');
        CHECK(has(sj, "\"mode\":\"confirm\""));
        CHECK(has(sj, "\"commits\":1"));
        // This filter has no head attached, so it is honest about that rather
        // than reporting a readiness it does not have.
        CHECK(has(sj, "\"ready\":false"));
        CHECK(!has(sj, ":nan"));

        const std::string cj = s.config().to_json();
        CHECK(cj.front() == '{');
        CHECK(cj.back() == '}');
        CHECK(has(cj, "\"confirm_steps\":2"));
        CHECK(!has(cj, ":nan"));

        // The refusal event is JSON too, and carries no fabricated observation.
        StreamEvent bad_ev;
        bad_ev.valid = false;
        bad_ev.reason = "head is not ready";
        const std::string bj = bad_ev.to_json();
        CHECK(has(bj, "\"valid\":false"));
        CHECK(has(bj, "\"step\":0"));
        CHECK(!has(bj, ":nan"));
    }

    // C5 — reset really resets.
    {
        TEST("C5 reset: every counter, the commitment and the ring");
        DecisionHead head;
        CHECK(head.init(768, 7u));
        StreamingDecision s;
        s.set_config(cfg_window(3, 2));
        s.init(head);
        std::vector<float> h(768, 0.01f);
        StreamEvent e;
        for (int i = 0; i < 5; ++i) s.push(h.data(), e);
        CHECK(s.steps() == 5);
        CHECK(s.window_filled() == 3);

        s.reset();
        CHECK(s.steps() == 0);
        CHECK(s.commits() == 0);
        CHECK(s.changes() == 0);
        CHECK(s.releases() == 0);
        CHECK(s.weak() == 0);
        CHECK(s.failures() == 0);
        CHECK(!s.committed());
        CHECK(s.action() == DecisionAction::ABSTAIN);
        CHECK(s.strength() == 0);
        CHECK(s.window_filled() == 0);
        CHECK(s.last().step == 0);
        CHECK(s.ready());                    // the head was NOT detached
    }

    // C6 — push_batch fails closed on a bad argument.
    {
        TEST("C6 push_batch: refuses a null batch, an empty one, and a dead head");
        DecisionHead head;
        CHECK(head.init(768, 11u));
        StreamingDecision s;
        s.init(head);

        std::vector<StreamEvent> ev(3);       // pre-filled: must be CLEARED
        CHECK(!s.push_batch(static_cast<const float*>(nullptr), 4, ev));
        CHECK(ev.empty());
        CHECK(has(s.error(), "null or empty"));

        std::vector<float> h(768, 0.0f);
        CHECK(!s.push_batch(h.data(), 0, ev));
        CHECK(ev.empty());

        Tensor wrong("wrong", {4, 512}, DType::F32);
        CHECK(!s.push_batch(wrong, ev));
        CHECK(ev.empty());
        CHECK(has(s.error(), "width mismatch"));

        Tensor good("good", {4, 768}, DType::F32);
        CHECK(s.push_batch(good, ev));
        CHECK(ev.size() == 4);
        CHECK(s.steps() == 4);
    }
}

// ---------------------------------------------------------------------------
// PART D — invariants over long, varied streams
// ---------------------------------------------------------------------------
void part_d() {
    platform::log_info("--- Part D: invariants over long streams ---");

    // D1 — a perfectly alternating stream never commits at confirm_steps = 2.
    {
        TEST("D1 an alternating stream never commits at confirm_steps=2");
        StreamingDecision s;
        s.set_config(cfg_confirm(2));
        StreamEvent e;
        for (int i = 0; i < 200; ++i) {
            const DecisionAction a = (i % 2 == 0) ? DecisionAction::BUY : DecisionAction::HOLD;
            s.push_decision(mk(a, 0.90f), e);
            CHECK(!e.committed);
        }
        CHECK(s.commits() == 0);
        CHECK(s.steps() == 200);
    }

    // D2/D3 — the bookkeeping identities, over a long random stream.
    {
        TEST("D2 the counters and the invariants hold over 10k random steps");
        StreamingDecision s;
        s.set_config(cfg_confirm(4, 0.45f, 0.65f));
        StreamEvent e;
        Lcg rng(0x5EED1234ull);

        int64_t changed_events = 0;
        int64_t first_events   = 0;
        int64_t bad_change     = 0;   // changed with previous == action
        int64_t bad_first      = 0;   // first with previous != ABSTAIN
        int64_t held_committed = 0;   // committed with action == ABSTAIN

        DecisionAction prev = DecisionAction::ABSTAIN;
        for (int i = 0; i < 10000; ++i) {
            const DecisionAction a = sticky_action(rng, prev, 0.80f);
            prev = a;
            const float conf = 0.20f + 0.79f * rng.unit();
            s.push_decision(mk(a, conf, 0.10f + 0.80f * rng.unit()), e);

            if (e.changed) { ++changed_events; if (e.previous == e.action) ++bad_change; }
            if (e.first)   { ++first_events;   if (e.previous != DecisionAction::ABSTAIN) ++bad_first; }
            if (e.committed && e.action == DecisionAction::ABSTAIN) ++held_committed;
        }

        CHECK(s.steps() == 10000);
        // Every `changed` event is exactly one of: a first commitment, a switch,
        // or a release. Nothing else may set the flag.
        CHECK(changed_events == s.commits() + s.changes() + s.releases());
        CHECK(first_events == s.commits());
        CHECK(bad_change == 0);
        CHECK(bad_first == 0);
        CHECK(held_committed == 0);
        // The stream must actually EXERCISE the interesting paths, or the
        // invariants above are asserted over a stream that only ever walked the
        // "nothing happened" branch. Both are required, not just one.
        CHECK(s.commits() > 0);
        CHECK(s.changes() > 0);
        CHECK(s.releases() > 0);
        // A commitment cannot change more often than the evidence allows: every
        // switch and every release costs a fresh quorum of consecutive
        // agreements, so each one consumes at least confirm_steps observations.
        CHECK(s.changes() + s.releases() <= s.steps() / 4 + 1);
        platform::log_info("      10k steps: commits=%lld changes=%lld releases=%lld weak=%lld",
                           static_cast<long long>(s.commits()),
                           static_cast<long long>(s.changes()),
                           static_cast<long long>(s.releases()),
                           static_cast<long long>(s.weak()));
    }

    // D4 — the same input twice must give the same output twice.
    {
        TEST("D4 the filter is deterministic");
        auto run = [](std::string& trace) {
            StreamingDecision s;
            s.set_config(cfg_window(5, 3, 0.45f, 0.65f));
            StreamEvent e;
            Lcg rng(0xABCDEFull);
            trace.clear();
            for (int i = 0; i < 500; ++i) {
                const DecisionAction a = static_cast<DecisionAction>(
                    rng.next() % static_cast<uint32_t>(DecisionAction::COUNT));
                s.push_decision(mk(a, 0.20f + 0.79f * rng.unit()), e);
                trace += e.to_json();
                trace += '\n';
            }
            return s.to_json();
        };
        std::string t1, t2;
        const std::string s1 = run(t1);
        const std::string s2 = run(t2);
        CHECK(t1 == t2);
        CHECK(s1 == s2);
        CHECK(!t1.empty());
    }

    // D5 — every COMMIT carries the evidence that licensed it.
    //
    //      Note what this does NOT claim: an event may be `committed = true`
    //      with `strength = 0`, because a weak observation breaks the run
    //      without releasing the commitment. The invariant is on the moment of
    //      commitment, not on every subsequent step.
    {
        TEST("D5 every commit reports at least a quorum of evidence");
        StreamingDecision s;
        s.set_config(cfg_confirm(3));
        StreamEvent e;
        Lcg rng(0x1234ull);
        int64_t commit_events   = 0;
        int64_t under_quorum    = 0;
        int64_t abstain_commits = 0;
        DecisionAction prev = DecisionAction::ABSTAIN;
        for (int i = 0; i < 5000; ++i) {
            const DecisionAction a = sticky_action(rng, prev, 0.75f);
            prev = a;
            s.push_decision(mk(a, 0.30f + 0.69f * rng.unit()), e);
            if (e.changed && e.committed) {
                ++commit_events;
                if (e.strength < 3) ++under_quorum;
                if (e.action == DecisionAction::ABSTAIN) ++abstain_commits;
            }
        }
        CHECK(commit_events > 20);      // the path was actually walked
        CHECK(under_quorum == 0);
        // ABSTAIN is the absence of a position, so it is never a commitment.
        CHECK(abstain_commits == 0);
        CHECK(s.committed() == false || s.action() != DecisionAction::ABSTAIN);
    }
}

} // namespace

int main() {
    platform::log_info("=== test_streaming_decision (MILESTONE 9) ===");
    part_a();
    part_b();
    part_c();
    part_d();
    platform::log_info("=== RESULT: %d passed, %d failed, %d skipped ===",
                       g_passed, g_failed, g_skipped);
    return g_failed == 0 ? 0 : 1;
}
