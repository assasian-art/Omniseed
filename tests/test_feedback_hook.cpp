// =============================================================================
//  OmniSeed — tests/test_feedback_hook.cpp
//
//  MILESTONE 10 — Phase 2.3, feedback hooks.
//
//  THE CONTRACT THIS FILE EXISTS TO PIN. A record that scores a system's own
//  predictions is the one component in the tree that can flatter the system
//  without anyone noticing, so every property asserted here is a way it could
//  lie:
//
//    1. AN UNMEASURED NUMBER IS NOT A ZERO (Part C). `hit_rate()` returns -1.0
//       when nothing has been resolved. "I have never been checked" and "I have
//       always been wrong" are different statements, and a record that conflates
//       them reports 0% for a system nobody has ever tested.
//    2. AN ABSENT OUTCOME IS NOT A MISS (Part C). Expired and Rejected rows are
//       excluded from every accuracy number, for different reasons. Averaging
//       them in would let a market that never moved look like a bad call.
//    3. ONE PREDICTION, ONE SCORE (Part B). Re-resolving a row is refused. A
//       record that can be scored twice will eventually agree with itself.
//    4. THE SCORE MUST BE ABLE TO GO DOWN (Parts C and G). Every "confidence
//       predicts correctness" claim is paired with an ANTI-correlated control
//       in the same test: a detector that cannot fail is not a detector.
//    5. FAIL CLOSED, AND ATOMICALLY (Part E). A corrupt file leaves BOTH stores
//       exactly as they were. A journal that restored while its ledger did not
//       would silently reset every trust factor to neutral.
//
//  Part F is the JOINT: the §37 event log feeds the journal, the §37 identity
//  is extended to the record, and the 369 REAL held-out rows are scored against
//  their real labels — which is the first time anything in this tree has
//  measured a head against what actually happened.
//
//  Fully offline and UNGATED. No model, no `.venv`, no network.
// =============================================================================
#include "omniseed/core/platform.h"
#include "omniseed/decision_head.h"
#include "omniseed/feedback_hook.h"
#include "omniseed/streaming_decision.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
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

constexpr int32_t kE = 768;

bool close(double a, double b, double tol) { return std::fabs(a - b) <= tol; }

bool has(const std::string& s, const char* needle) {
    return s.find(needle) != std::string::npos;
}

// A prediction with no stream behind it.
//
// `at_ms` is set EXPLICITLY so a serialise() comparison is a test of the FORMAT
// rather than of the wall clock: two journals built the same way must produce
// the same bytes, and a `record()` that stamped the current time would make
// that impossible to assert. A3 covers the clock being filled in when unset.
JournalEntry mk_entry(DecisionAction a, float conf, const std::string& key = "t") {
    JournalEntry e;
    e.task_key   = key;
    e.predicted  = a;
    e.confidence = conf;
    e.at_ms      = 1234.5;
    return e;
}

// A hand-built head output, for the §37 side of the joint.
DecisionResult mk_dec(DecisionAction a, float conf, float margin = 0.50f) {
    DecisionResult d;
    d.action_type      = a;
    d.confidence_score = conf;
    d.margin           = margin;
    d.routing          = "self";
    d.matvecs          = 1;
    return d;
}

// The project's convention (test_decision_head.cpp, test_lora.cpp): a test
// writes to the SYSTEM temp directory, never into the source tree. `build/` is
// gitignored, but a file left behind by a crashing test is still litter in the
// checkout, and TMPDIR is cleaned by the OS.
std::string tmp_path(const char* name) {
    const char* base = std::getenv("TEMP");
    if (base == nullptr || *base == '\0') base = std::getenv("TMPDIR");
    if (base == nullptr || *base == '\0') base = ".";
    std::string p = base;
    if (!p.empty() && p.back() != '/' && p.back() != '\\') p += '/';
    p += "omniseed_feedback_";
    p += name;
    p += ".bin";
    return p;
}

bool action_from_name(const std::string& s, DecisionAction& out) {
    for (int32_t i = 0; i < static_cast<int32_t>(DecisionAction::COUNT); ++i) {
        const DecisionAction a = static_cast<DecisionAction>(i);
        if (s == decision_action_name(a)) { out = a; return true; }
    }
    return false;
}

// ---- the real held-out trading fixture ------------------------------------
struct TradingFixture {
    int32_t E = 0;
    std::vector<float>            h;        // n * E
    std::vector<DecisionAction>   labels;   // n
    size_t n() const { return labels.size(); }
};

bool load_trading_fixture(TradingFixture& fx) {
    const std::string dir = "tests/fixtures/head_calibration/trading";

    std::ifstream hf(dir + "/hidden.f32", std::ios::binary | std::ios::ate);
    if (!hf) return false;
    const std::streamoff bytes = hf.tellg();
    if (bytes <= 0 || bytes % 4 != 0) return false;
    hf.seekg(0, std::ios::beg);
    fx.h.resize(static_cast<size_t>(bytes / 4));
    hf.read(reinterpret_cast<char*>(fx.h.data()), bytes);
    if (!hf) return false;

    std::ifstream lf(dir + "/labels.tsv");
    if (!lf) return false;
    std::string line;
    if (!std::getline(lf, line)) return false;
    // header: id <TAB> trading.regime <TAB> DecisionAction
    int action_col = -1;
    {
        size_t pos = 0;
        int col = 0;
        while (pos <= line.size()) {
            const size_t tab = line.find('\t', pos);
            const std::string cell =
                line.substr(pos, tab == std::string::npos ? std::string::npos : tab - pos);
            if (cell == "DecisionAction") action_col = col;
            if (tab == std::string::npos) break;
            pos = tab + 1;
            ++col;
        }
    }
    if (action_col < 1) return false;

    while (std::getline(lf, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n'))
            line.pop_back();
        if (line.empty()) continue;
        std::vector<std::string> cells;
        size_t pos = 0;
        while (true) {
            const size_t tab = line.find('\t', pos);
            cells.push_back(line.substr(
                pos, tab == std::string::npos ? std::string::npos : tab - pos));
            if (tab == std::string::npos) break;
            pos = tab + 1;
        }
        if (static_cast<int>(cells.size()) <= action_col) return false;
        DecisionAction a = DecisionAction::ABSTAIN;
        if (!action_from_name(cells[static_cast<size_t>(action_col)], a)) return false;
        fx.labels.push_back(a);
    }

    if (fx.labels.empty()) return false;
    // E is inferred from the fixture, not assumed: the size must divide evenly.
    const size_t total = fx.h.size();
    if (total % fx.n() != 0) return false;
    fx.E = static_cast<int32_t>(total / fx.n());
    return fx.E == kE;
}

// ===========================================================================
//  Part A — recording
// ===========================================================================
void part_a() {
    TEST("A1 ids are monotonic from 1");
    {
        DecisionJournal j;
        const int64_t a = j.record(mk_entry(DecisionAction::BUY, 0.9f));
        const int64_t b = j.record(mk_entry(DecisionAction::SELL, 0.8f));
        CHECK(a == 1);
        CHECK(b == 2);
        CHECK(j.size() == 2u);
        CHECK(j.find(a) != nullptr);
        CHECK(j.find(b)->predicted == DecisionAction::SELL);
    }

    TEST("A2 an INVALID event is not a prediction");
    {
        DecisionJournal j;
        StreamEvent e;
        e.step  = 1;
        e.valid = false;   // the head refused to observe
        CHECK(j.record_event(e, "t") == 0);
        // The whole point: a dead head's silence must not enter the record as a
        // row that will later be scored as a miss.
        CHECK(j.size() == 0u);
        CHECK(j.stats().predictions == 0);
    }

    TEST("A3 a valid event copies every field");
    {
        DecisionJournal j;
        StreamEvent e;
        e.step            = 7;
        e.valid           = true;
        e.observed_action = DecisionAction::HEDGE;
        e.observed_conf   = 0.61f;
        e.observed_margin = 0.12f;
        e.action          = DecisionAction::BUY;
        e.committed       = true;
        e.changed         = true;
        e.strength        = 4;
        const int64_t id = j.record_event(e, "trading.AAPL.1d");
        const JournalEntry* je = j.find(id);
        CHECK(je != nullptr);
        CHECK(je->step == 7);
        CHECK(je->predicted == DecisionAction::HEDGE);
        CHECK(close(je->confidence, 0.61, 1e-6));
        CHECK(close(je->margin, 0.12, 1e-6));
        CHECK(je->held == DecisionAction::BUY);
        CHECK(je->committed);
        CHECK(je->changed);
        CHECK(close(je->strength, 4.0, 1e-6));
        CHECK(je->task_key == "trading.AAPL.1d");
        CHECK(je->at_ms > 0.0);
    }

    TEST("A4 record_stream counts only the valid events");
    {
        DecisionJournal j;
        std::vector<StreamEvent> ev(5);
        for (size_t i = 0; i < ev.size(); ++i) {
            ev[i].step  = static_cast<int64_t>(i + 1);
            ev[i].valid = (i != 2);          // one dead step
            ev[i].observed_action = DecisionAction::HOLD;
            ev[i].observed_conf   = 0.7f;
        }
        CHECK(j.record_stream(ev, "t") == 4u);
        CHECK(j.size() == 4u);
        // The surviving rows keep their ORIGINAL step numbers, so a gap in the
        // record is visible rather than closed up.
        CHECK(j.entries()[2].step == 4);
    }

    TEST("A5 the journal is bounded, and keeps the NEWEST rows");
    {
        DecisionJournal::Config c;
        c.max_entries = 10;
        DecisionJournal j(c);
        for (int i = 0; i < 25; ++i)
            j.record(mk_entry(DecisionAction::HOLD, 0.5f));
        CHECK(j.size() == 10u);
        // Ids stay monotonic across the prune, so a later resolve() cannot
        // silently hit a recycled id.
        CHECK(j.entries().front().id == 16);
        CHECK(j.entries().back().id == 25);
        CHECK(j.find(3) == nullptr);   // pruned away, and find() says so
    }

    TEST("A6 a single-shot decision records the head's own routing");
    {
        DecisionJournal j;
        DecisionResult d = mk_dec(DecisionAction::SELL, 0.91f, 0.4f);
        d.fast_path = true;
        const int64_t id = j.record_decision(d, "t");
        const JournalEntry* e = j.find(id);
        CHECK(e != nullptr);
        CHECK(e->predicted == DecisionAction::SELL);
        CHECK(e->held == DecisionAction::SELL);
        CHECK(e->committed);            // fast_path == the head says "act on it"
        CHECK(e->step == 0);            // no stream
    }
}

// ===========================================================================
//  Part B — resolution, and why it is allowed to fail
// ===========================================================================
void part_b() {
    TEST("B1 resolving an unknown id fails, and says so");
    {
        DecisionJournal j;
        CHECK(!j.resolve(999, DecisionAction::BUY, 0.0f, 5));
        CHECK(!j.error().empty());
        CHECK(j.size() == 0u);
    }

    TEST("B2 a prediction is scored against ONE outcome");
    {
        DecisionJournal j;
        const int64_t id = j.record(mk_entry(DecisionAction::BUY, 0.9f));
        CHECK(j.resolve(id, DecisionAction::BUY, 1.0f, 5));
        // Re-scoring would let a record be edited until it agreed with itself.
        CHECK(!j.resolve(id, DecisionAction::SELL, -1.0f, 5));
        CHECK(j.find(id)->realised == DecisionAction::BUY);
        CHECK(close(j.find(id)->score, 1.0, 1e-9));
        // Same for the verdict path.
        CHECK(!j.resolve_kind(id, OutcomeKind::Corrected));
    }

    TEST("B3 resolve_kind(Unknown) would UN-resolve a row and is refused");
    {
        DecisionJournal j;
        const int64_t id = j.record(mk_entry(DecisionAction::BUY, 0.9f));
        CHECK(!j.resolve_kind(id, OutcomeKind::Unknown));
        CHECK(j.find(id)->outcome == OutcomeKind::Unknown);
    }

    TEST("B4 resolve_all touches only the matching, unresolved rows");
    {
        DecisionJournal j;
        j.record(mk_entry(DecisionAction::BUY, 0.9f, "AAPL"));
        j.record(mk_entry(DecisionAction::SELL, 0.9f, "MSFT"));
        j.record(mk_entry(DecisionAction::HOLD, 0.9f, "AAPL"));
        CHECK(j.resolve_all("AAPL", DecisionAction::BUY, 2.0f, 5) == 2u);
        CHECK(j.resolve_all("AAPL", DecisionAction::BUY, 2.0f, 5) == 0u);  // nothing left
        CHECK(j.unresolved() == 1u);
        CHECK(j.find(2)->outcome == OutcomeKind::Unknown);
    }

    TEST("B5 resolve_latest picks the NEWEST, because the user replies to it");
    {
        DecisionJournal j;
        const int64_t a = j.record(mk_entry(DecisionAction::BUY, 0.9f, "k"));
        const int64_t b = j.record(mk_entry(DecisionAction::SELL, 0.9f, "k"));
        CHECK(j.resolve_latest("k", OutcomeKind::Confirmed) == 1u);
        CHECK(j.find(b)->outcome == OutcomeKind::Confirmed);
        CHECK(j.find(a)->outcome == OutcomeKind::Unknown);
    }

    TEST("B6 expire_through is scoped by task key and by step");
    {
        DecisionJournal j;
        j.record_event([] { StreamEvent e; e.step = 1; e.valid = true;
                            e.observed_action = DecisionAction::HOLD; return e; }(), "A");
        j.record_event([] { StreamEvent e; e.step = 2; e.valid = true;
                            e.observed_action = DecisionAction::HOLD; return e; }(), "A");
        j.record_event([] { StreamEvent e; e.step = 3; e.valid = true;
                            e.observed_action = DecisionAction::HOLD; return e; }(), "B");
        CHECK(j.expire_through(2, "A") == 2u);
        CHECK(j.unresolved() == 1u);           // B3 untouched
        CHECK(j.expire_through(99, "B") == 1u);
        CHECK(j.unresolved() == 0u);
    }

    TEST("B7 find() is null for an id that never existed");
    {
        DecisionJournal j;
        CHECK(j.find(0) == nullptr);
        CHECK(j.find(-1) == nullptr);
        CHECK(j.find(42) == nullptr);
    }
}

// ===========================================================================
//  Part C — the honesty of the numbers (the load-bearing part)
// ===========================================================================
void part_c() {
    TEST("C1 an empty journal reports NOT MEASURED, never zero");
    {
        DecisionJournal j;
        const JournalStats s = j.stats();
        CHECK(s.predictions == 0);
        CHECK(s.hit_rate() == -1.0);          // NOT 0.0
        CHECK(s.filter_hit_rate() == -1.0);
        CHECK(s.commitment_rate() == -1.0);
        CHECK(s.ece == -1.0);
        CHECK(s.brier == -1.0);
        CHECK(s.mean_conf_hit == -1.0);
        CHECK(s.confidence_gap == -1.0);
    }

    TEST("C2 unresolved rows do not move the accuracy");
    {
        DecisionJournal j;
        for (int i = 0; i < 20; ++i) j.record(mk_entry(DecisionAction::BUY, 0.9f));
        const JournalStats s = j.stats();
        CHECK(s.predictions == 20);
        CHECK(s.unresolved == 20);
        CHECK(s.verdicts == 0);
        CHECK(s.hit_rate() == -1.0);          // still unmeasured
    }

    TEST("C3 an Expired row is NOT a miss");
    {
        DecisionJournal j;
        const int64_t a = j.record(mk_entry(DecisionAction::BUY, 0.9f));
        j.resolve(a, DecisionAction::BUY, 1.0f, 5);
        const int64_t b = j.record(mk_entry(DecisionAction::SELL, 0.9f));
        j.resolve_kind(b, OutcomeKind::Expired);
        const JournalStats s = j.stats();
        CHECK(s.verdicts == 1);
        CHECK(s.expired == 1);
        CHECK(close(s.hit_rate(), 1.0, 1e-12));   // the expiry did not cost anything
    }

    TEST("C4 a Rejected row is NOT a miss");
    {
        DecisionJournal j;
        const int64_t a = j.record(mk_entry(DecisionAction::BUY, 0.9f));
        j.resolve(a, DecisionAction::BUY, 1.0f, 5);
        const int64_t b = j.record(mk_entry(DecisionAction::EXPLAIN, 0.9f));
        j.resolve_kind(b, OutcomeKind::Rejected);
        const JournalStats s = j.stats();
        CHECK(s.verdicts == 1);
        CHECK(s.rejected == 1);
        CHECK(close(s.hit_rate(), 1.0, 1e-12));
    }

    TEST("C5 hit_rate is hits over VERDICTS, not over predictions");
    {
        DecisionJournal j;
        // 3 hits
        for (int i = 0; i < 3; ++i) {
            const int64_t id = j.record(mk_entry(DecisionAction::BUY, 0.9f));
            j.resolve(id, DecisionAction::BUY, 0.0f, 1);
        }
        // 1 miss
        {
            const int64_t id = j.record(mk_entry(DecisionAction::BUY, 0.9f));
            j.resolve(id, DecisionAction::SELL, 0.0f, 1);
        }
        // 6 unmeasured rows that must not dilute it
        for (int i = 0; i < 6; ++i) j.record(mk_entry(DecisionAction::HOLD, 0.9f));
        const JournalStats s = j.stats();
        CHECK(s.predictions == 10);
        CHECK(s.verdicts == 4);
        CHECK(s.hits == 3);
        CHECK(close(s.hit_rate(), 0.75, 1e-12));
    }

    TEST("C6 confidence_gap is a SIGN you must measure");
    {
        // --- informative: high confidence when right, low when wrong --------
        {
            DecisionJournal j;
            for (int i = 0; i < 10; ++i) {
                const int64_t id = j.record(mk_entry(DecisionAction::BUY, 0.90f));
                j.resolve(id, DecisionAction::BUY, 0.0f, 1);
            }
            for (int i = 0; i < 10; ++i) {
                const int64_t id = j.record(mk_entry(DecisionAction::BUY, 0.30f));
                j.resolve(id, DecisionAction::SELL, 0.0f, 1);
            }
            const JournalStats s = j.stats();
            CHECK(close(s.mean_conf_hit, 0.90, 1e-6));
            CHECK(close(s.mean_conf_miss, 0.30, 1e-6));
            CHECK(s.confidence_gap > 0.0);
            CHECK(close(s.confidence_gap, 0.60, 1e-6));
        }
        // --- THE NEGATIVE CONTROL: inverted --------------------------------
        // Same shape, confidence swapped. If the metric cannot go negative it
        // is not measuring anything, and the positive case above proves nothing.
        {
            DecisionJournal j;
            for (int i = 0; i < 10; ++i) {
                const int64_t id = j.record(mk_entry(DecisionAction::BUY, 0.30f));
                j.resolve(id, DecisionAction::BUY, 0.0f, 1);
            }
            for (int i = 0; i < 10; ++i) {
                const int64_t id = j.record(mk_entry(DecisionAction::BUY, 0.90f));
                j.resolve(id, DecisionAction::SELL, 0.0f, 1);
            }
            const JournalStats s = j.stats();
            CHECK(s.confidence_gap < 0.0);
            CHECK(close(s.confidence_gap, -0.60, 1e-6));
        }
    }

    TEST("C7 with no misses the gap is UNMEASURABLE, not flattering");
    {
        DecisionJournal j;
        for (int i = 0; i < 5; ++i) {
            const int64_t id = j.record(mk_entry(DecisionAction::BUY, 0.9f));
            j.resolve(id, DecisionAction::BUY, 0.0f, 1);
        }
        const JournalStats s = j.stats();
        CHECK(s.mean_conf_hit > 0.0);
        CHECK(s.mean_conf_miss == -1.0);
        // One side is missing, so the difference is undefined. Reporting
        // +0.90 here would be inventing a measurement.
        CHECK(s.confidence_gap == -1.0);
    }

    TEST("C8 ECE: perfect calibration ~0, systematic overconfidence > 0");
    {
        // 10 bins of 100 rows each; bin b claims (b+0.5)/10 and is right that
        // often. Perfectly calibrated by construction.
        {
            DecisionJournal j;
            for (int b = 0; b < 10; ++b) {
                const float conf = (static_cast<float>(b) + 0.5f) / 10.0f;
                const int correct = b;   // 0..9 of 10 rows in this bin
                for (int i = 0; i < 10; ++i) {
                    const int64_t id = j.record(mk_entry(DecisionAction::BUY, conf));
                    j.resolve(id, i < correct ? DecisionAction::BUY
                                              : DecisionAction::SELL, 0.0f, 1);
                }
            }
            const JournalStats s = j.stats();
            CHECK(s.verdicts == 100);
            // Bin b is right b/10 of the time and claims (b+0.5)/10: the
            // systematic 0.05 half-bin offset is the residual.
            CHECK(s.ece >= 0.0);
            CHECK(s.ece < 0.06);
        }
        // --- THE NEGATIVE CONTROL: claims 0.95, is right 10% ----------------
        {
            DecisionJournal j;
            for (int i = 0; i < 100; ++i) {
                const int64_t id = j.record(mk_entry(DecisionAction::BUY, 0.95f));
                j.resolve(id, i < 10 ? DecisionAction::BUY : DecisionAction::SELL,
                          0.0f, 1);
            }
            const JournalStats s = j.stats();
            CHECK(close(s.hit_rate(), 0.10, 1e-12));
            CHECK(s.ece > 0.8);        // 0.95 claimed vs 0.10 observed
            // Every row carries the SAME confidence, so the head's confidence
            // separates nothing: the gap is exactly 0, and reporting anything
            // else would be inventing a signal out of a constant.
            CHECK(close(s.confidence_gap, 0.0, 1e-9));
        }
    }

    TEST("C9 Brier of a coin-flip predictor at 50% is 0.25");
    {
        DecisionJournal j;
        for (int i = 0; i < 10; ++i) {
            const int64_t id = j.record(mk_entry(DecisionAction::BUY, 0.5f));
            j.resolve(id, i < 5 ? DecisionAction::BUY : DecisionAction::SELL, 0.0f, 1);
        }
        CHECK(close(j.stats().brier, 0.25, 1e-9));
    }

    TEST("C10 commitment_rate is -1 with no rows and measured with them");
    {
        DecisionJournal j;
        CHECK(j.stats().commitment_rate() == -1.0);
        for (int i = 0; i < 4; ++i) {
            StreamEvent e;
            e.valid = true;
            e.observed_action = DecisionAction::BUY;
            e.observed_conf   = 0.9f;
            e.action          = (i < 3) ? DecisionAction::BUY : DecisionAction::ABSTAIN;
            e.committed       = (i < 3);
            j.record_event(e, "t");
        }
        CHECK(close(j.stats().commitment_rate(), 0.75, 1e-12));
    }

    TEST("C11 the confusion matrix counts only Realised rows");
    {
        DecisionJournal j;
        for (int i = 0; i < 3; ++i) {
            const int64_t id = j.record(mk_entry(DecisionAction::BUY, 0.9f));
            j.resolve(id, DecisionAction::BUY, 0.0f, 1);
        }
        const int64_t m = j.record(mk_entry(DecisionAction::BUY, 0.9f));
        j.resolve(m, DecisionAction::SELL, 0.0f, 1);
        // A Confirmed row has no `realised` and must not enter the matrix.
        const int64_t c = j.record(mk_entry(DecisionAction::HOLD, 0.9f));
        j.resolve_kind(c, OutcomeKind::Confirmed);

        const JournalStats s = j.stats();
        const size_t A = static_cast<size_t>(DecisionAction::COUNT);
        CHECK(s.confusion.size() == A);
        CHECK(s.confusion[static_cast<size_t>(DecisionAction::BUY)]
                        [static_cast<size_t>(DecisionAction::BUY)] == 3);
        CHECK(s.confusion[static_cast<size_t>(DecisionAction::BUY)]
                        [static_cast<size_t>(DecisionAction::SELL)] == 1);
        int64_t total = 0;
        for (const auto& row : s.confusion)
            for (int64_t v : row) total += v;
        CHECK(total == 4);   // the Confirmed row is absent, as it must be
    }

    TEST("C12 the filter is scored only where it held a position");
    {
        DecisionJournal j;
        // committed and right
        {
            StreamEvent e;
            e.valid = true; e.observed_action = DecisionAction::BUY; e.observed_conf = 0.9f;
            e.action = DecisionAction::BUY; e.committed = true;
            const int64_t id = j.record_event(e, "t");
            j.resolve(id, DecisionAction::BUY, 0.0f, 1);
        }
        // NOT committed, and "right" — but the filter made no call, so it is not
        // a filter row. Counting it would credit the filter for the head's work.
        {
            StreamEvent e;
            e.valid = true; e.observed_action = DecisionAction::SELL; e.observed_conf = 0.4f;
            e.action = DecisionAction::ABSTAIN; e.committed = false;
            const int64_t id = j.record_event(e, "t");
            j.resolve(id, DecisionAction::SELL, 0.0f, 1);
        }
        // committed and wrong
        {
            StreamEvent e;
            e.valid = true; e.observed_action = DecisionAction::HOLD; e.observed_conf = 0.9f;
            e.action = DecisionAction::HOLD; e.committed = true;
            const int64_t id = j.record_event(e, "t");
            j.resolve(id, DecisionAction::BUY, 0.0f, 1);
        }
        const JournalStats s = j.stats();
        CHECK(s.verdicts == 3);
        CHECK(s.hits == 2);                 // the head got 2 of 3
        CHECK(s.filter_verdicts == 2);      // the filter made 2 calls
        CHECK(s.filter_hits == 1);
        CHECK(close(s.filter_hit_rate(), 0.5, 1e-12));
    }

    TEST("C13 the to_json carries no bare nan/inf token");
    {
        DecisionJournal j;
        CHECK(!has(j.stats().to_json(), ":nan"));
        CHECK(!has(j.stats().to_json(), ":inf"));
        CHECK(!has(j.to_json(), ":nan"));
        // -1 must be REPORTED as -1, not silently omitted or turned into 0.
        CHECK(has(j.stats().to_json(), "\"hit_rate\":-1"));
    }
}

// ===========================================================================
//  Part D — the verdict join
// ===========================================================================
void part_d() {
    TEST("D1 a confirmation resolves the latest row and raises trust");
    {
        FeedbackHook h;
        StreamEvent e;
        e.valid = true; e.observed_action = DecisionAction::BUY; e.observed_conf = 0.9f;
        const int64_t id = h.observe(e, "trading");
        CHECK(h.trust("trading") == 0.5);      // neutral before any verdict

        CHECK(h.on_user_turn("yes, correct", "trading") ==
              UserFeedbackLoop::Verdict::Confirm);
        CHECK(h.journal().find(id)->outcome == OutcomeKind::Confirmed);
        CHECK(h.trust("trading") > 0.5);
        CHECK(h.stats().hits == 1);
        CHECK(close(h.stats().hit_rate(), 1.0, 1e-12));
    }

    TEST("D2 a correction resolves it as a MISS and lowers trust");
    {
        FeedbackHook h;
        StreamEvent e;
        e.valid = true; e.observed_action = DecisionAction::BUY; e.observed_conf = 0.9f;
        const int64_t id = h.observe(e, "trading");
        CHECK(h.on_user_turn("no, that's wrong", "trading") ==
              UserFeedbackLoop::Verdict::Correct);
        CHECK(h.journal().find(id)->outcome == OutcomeKind::Corrected);
        CHECK(h.trust("trading") < 0.5);
        CHECK(h.stats().verdicts == 1);
        CHECK(h.stats().hits == 0);
        CHECK(close(h.stats().hit_rate(), 0.0, 1e-12));   // MEASURED zero, not -1
    }

    TEST("D3 a rejection is its own kind, and is NOT a miss");
    {
        FeedbackHook h;
        StreamEvent e;
        e.valid = true; e.observed_action = DecisionAction::EXPLAIN; e.observed_conf = 0.9f;
        const int64_t id = h.observe(e, "trading");
        CHECK(h.on_user_turn("stop", "trading") == UserFeedbackLoop::Verdict::Reject);
        CHECK(h.journal().find(id)->outcome == OutcomeKind::Rejected);
        CHECK(h.stats().rejected == 1);
        CHECK(h.stats().verdicts == 0);            // no accuracy datum was created
        CHECK(h.stats().hit_rate() == -1.0);
        CHECK(h.trust("trading") < 0.5);           // but the ledger still moved
    }

    TEST("D4 a NEUTRAL turn changes NOTHING");
    {
        FeedbackHook h;
        StreamEvent e;
        e.valid = true; e.observed_action = DecisionAction::BUY; e.observed_conf = 0.9f;
        const int64_t id = h.observe(e, "trading");
        const size_t before = h.feedback_size();

        CHECK(h.on_user_turn("what time is it?", "trading") ==
              UserFeedbackLoop::Verdict::None);
        // Neither store may move: an unparsed turn is not feedback, and
        // resolving the journal on it would let a user's silence score the head.
        CHECK(h.feedback_size() == before);
        CHECK(h.journal().find(id)->outcome == OutcomeKind::Unknown);
        CHECK(h.trust("trading") == 0.5);
    }

    TEST("D5 implicit engagement moves the LEDGER and not the JOURNAL");
    {
        FeedbackHook h;
        StreamEvent e;
        e.valid = true; e.observed_action = DecisionAction::BUY; e.observed_conf = 0.9f;
        const int64_t id = h.observe(e, "trading");
        h.on_implicit("trading", true);
        CHECK(h.trust("trading") > 0.5);
        CHECK(h.journal().find(id)->outcome == OutcomeKind::Unknown);
        h.on_implicit("trading", false);
        CHECK(h.journal().find(id)->outcome == OutcomeKind::Unknown);
        CHECK(h.stats().verdicts == 0);
    }

    TEST("D6 an unseen key is neutral, and D7 a verdict with nothing open still counts");
    {
        FeedbackHook h;
        CHECK(h.trust("never-seen") == 0.5);
        // A confirm arrives with no open prediction: the ledger records it (the
        // user did give feedback) and the journal correctly has nothing to
        // resolve.
        CHECK(h.on_user_turn("yes", "k") == UserFeedbackLoop::Verdict::Confirm);
        CHECK(h.trust("k") > 0.5);
        CHECK(h.journal().size() == 0u);
        CHECK(h.stats().verdicts == 0);
    }

    TEST("D8 outcome() resolves every open prediction for one key");
    {
        FeedbackHook h;
        for (int i = 0; i < 3; ++i) {
            StreamEvent e;
            e.step  = i + 1;
            e.valid = true;
            e.observed_action = DecisionAction::BUY;
            e.observed_conf   = 0.9f;
            h.observe(e, "AAPL");
        }
        CHECK(h.outcome("AAPL", DecisionAction::SELL, -0.02f, 5) == 3u);
        CHECK(h.journal().unresolved() == 0u);
        const JournalStats s = h.stats();
        CHECK(s.verdicts == 3);
        CHECK(s.hits == 0);   // all three predicted BUY, the bar went SELL
        CHECK(close(s.hit_rate(), 0.0, 1e-12));
    }

    TEST("D9 expire() is scoped to one key");
    {
        FeedbackHook h;
        StreamEvent e;
        e.step = 5; e.valid = true; e.observed_action = DecisionAction::HOLD;
        e.observed_conf = 0.9f;
        h.observe(e, "A");
        h.observe(e, "B");
        CHECK(h.expire("A", 10) == 1u);
        CHECK(h.journal().unresolved() == 1u);
    }
}

// ===========================================================================
//  Part E — persistence, which fails closed AND atomically
// ===========================================================================
void part_e() {
    TEST("E1 a journal round-trips, ids and all");
    {
        const std::string path = tmp_path("journal");
        DecisionJournal j;
        const int64_t a = j.record(mk_entry(DecisionAction::BUY, 0.75f, "k1"));
        const int64_t b = j.record(mk_entry(DecisionAction::SELL, 0.25f, "k2"));
        j.resolve(a, DecisionAction::BUY, 1.5f, 5);
        j.resolve_kind(b, OutcomeKind::Expired);
        const std::string before = j.to_tsv();

        CHECK(j.save(path));

        DecisionJournal r;
        CHECK(r.load(path));
        CHECK(r.size() == 2u);
        CHECK(r.find(a)->outcome == OutcomeKind::Realised);
        CHECK(r.find(a)->realised == DecisionAction::BUY);
        CHECK(close(r.find(a)->score, 1.5, 1e-6));
        CHECK(r.find(b)->outcome == OutcomeKind::Expired);
        CHECK(r.to_tsv() == before);
        // The id counter resumes, so a restored journal cannot hand out an id
        // that a restored row already owns.
        CHECK(r.record(mk_entry(DecisionAction::HOLD, 0.5f)) == 3);

        std::remove(path.c_str());
    }

    TEST("E2 a truncated file fails closed and ATOMICALLY");
    {
        const std::string path = tmp_path("truncated");
        DecisionJournal j;
        for (int i = 0; i < 5; ++i) j.record(mk_entry(DecisionAction::BUY, 0.9f));
        CHECK(j.save(path));

        // Chop the tail off: a half-written file, which is what a crash leaves.
        {
            std::ifstream in(path, std::ios::binary | std::ios::ate);
            const std::streamoff sz = in.tellg();
            in.seekg(0, std::ios::beg);
            std::vector<char> buf(static_cast<size_t>(sz));
            in.read(buf.data(), sz);
            std::ofstream out(path, std::ios::binary | std::ios::trunc);
            out.write(buf.data(), sz - 9);
        }

        DecisionJournal r;
        for (int i = 0; i < 3; ++i) r.record(mk_entry(DecisionAction::HOLD, 0.1f, "keep"));
        CHECK(!r.load(path));
        // The 3 pre-existing rows must SURVIVE. A partial restore that silently
        // drops the misses is worse than no restore at all.
        CHECK(r.size() == 3u);
        CHECK(r.entries()[0].task_key == "keep");

        std::remove(path.c_str());
    }

    TEST("E3 a wrong magic is rejected");
    {
        const std::string path = tmp_path("magic");
        DecisionJournal j;
        j.record(mk_entry(DecisionAction::BUY, 0.9f));
        CHECK(j.save(path));
        {
            std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
            f.seekp(0);
            f.put('X');
        }
        DecisionJournal r;
        CHECK(!r.load(path));
        CHECK(r.size() == 0u);
        std::remove(path.c_str());
    }

    TEST("E4 the container restores BOTH stores");
    {
        const std::string path = tmp_path("hook");
        FeedbackHook h;
        StreamEvent e;
        e.valid = true; e.observed_action = DecisionAction::BUY; e.observed_conf = 0.9f;
        const int64_t id = h.observe(e, "alpha");
        h.on_user_turn("yes", "alpha");            // ledger: alpha 1/0
        h.on_user_turn("no, wrong", "beta");       // ledger: beta 0/1
        CHECK(h.save(path));

        FeedbackHook r;
        CHECK(r.load(path));
        CHECK(r.journal().size() == 1u);
        CHECK(r.journal().find(id)->outcome == OutcomeKind::Confirmed);
        CHECK(r.feedback_size() == 2u);
        CHECK(r.trust("alpha") > 0.5);
        CHECK(r.trust("beta") < 0.5);

        // "A DIFFERENT KEY DIFFERS" — the invariant that catches a ledger that
        // round-trips as a blob of identical rows.
        CHECK(std::fabs(r.trust("alpha") - r.trust("beta")) > 0.2);
        CHECK(r.trust("alpha") != r.trust("beta"));

        std::remove(path.c_str());
    }

    TEST("E5 a corrupt container replaces NEITHER store");
    {
        const std::string path = tmp_path("hook_corrupt");
        FeedbackHook h;
        StreamEvent e;
        e.valid = true; e.observed_action = DecisionAction::BUY; e.observed_conf = 0.9f;
        h.observe(e, "alpha");
        h.on_user_turn("yes", "alpha");
        CHECK(h.save(path));

        // Truncate into the ledger payload: the journal is intact, the ledger is
        // not. Restoring the journal alone would silently reset every trust
        // factor to neutral.
        {
            std::ifstream in(path, std::ios::binary | std::ios::ate);
            const std::streamoff sz = in.tellg();
            in.seekg(0, std::ios::beg);
            std::vector<char> buf(static_cast<size_t>(sz));
            in.read(buf.data(), sz);
            std::ofstream out(path, std::ios::binary | std::ios::trunc);
            out.write(buf.data(), sz - 5);
        }

        FeedbackHook r;
        r.on_user_turn("yes", "preexisting");
        const size_t rows_before = r.journal().size();
        CHECK(!r.load(path));
        CHECK(!r.error().empty());
        CHECK(r.journal().size() == rows_before);   // untouched
        CHECK(r.feedback_size() == 1u);             // untouched

        std::remove(path.c_str());
    }

    TEST("E6 serialise is deterministic");
    {
        DecisionJournal a, b;
        // Same verdicts recorded in the same order -> byte-identical bytes. An
        // unordered_map iteration order leaking into the file would break this.
        for (int i = 0; i < 6; ++i) {
            a.record(mk_entry(DecisionAction::BUY, 0.5f, "k" + std::to_string(i)));
            b.record(mk_entry(DecisionAction::BUY, 0.5f, "k" + std::to_string(i)));
        }
        CHECK(a.serialise() == b.serialise());

        UserFeedbackLoop fa, fb;
        fa.record(UserFeedbackLoop::Verdict::Confirm, "z");
        fa.record(UserFeedbackLoop::Verdict::Confirm, "a");
        fa.record(UserFeedbackLoop::Verdict::Correct, "m");
        fb.record(UserFeedbackLoop::Verdict::Correct, "m");
        fb.record(UserFeedbackLoop::Verdict::Confirm, "a");
        fb.record(UserFeedbackLoop::Verdict::Confirm, "z");
        CHECK(fa.serialise() == fb.serialise());
    }

    TEST("E7 the ledger round-trips on its own, and rejects junk");
    {
        const std::string path = tmp_path("ledger");
        UserFeedbackLoop f;
        f.record(UserFeedbackLoop::Verdict::Confirm, "a");
        f.record(UserFeedbackLoop::Verdict::Confirm, "a");
        f.record(UserFeedbackLoop::Verdict::Correct, "a");
        f.record(UserFeedbackLoop::Verdict::Reject, "b");
        CHECK(f.save(path));

        UserFeedbackLoop g;
        CHECK(g.load(path));
        CHECK(g.size() == 2u);
        CHECK(g.trust("a") == f.trust("a"));
        CHECK(g.trust("b") == f.trust("b"));
        CHECK(g.trust("a") != g.trust("b"));

        // deserialise() must leave a live ledger alone on bad input.
        const uint8_t junk[12] = {'N', 'O', 'P', 'E', 0, 0, 0, 0, 0, 0, 0, 0};
        CHECK(!g.deserialise(junk, sizeof(junk)));
        CHECK(g.size() == 2u);
        CHECK(g.trust("a") == f.trust("a"));

        std::remove(path.c_str());
    }
}

// ===========================================================================
//  Part F — the §37 joint, on REAL held-out rows
// ===========================================================================
void part_f() {
    TEST("F1 the journal consumes the §37 event log exactly");
    {
        DecisionHead head;
        CHECK(head.init(kE, 4242u));
        StreamingDecision f;
        CHECK(f.init(head));

        StreamingConfig c;
        c.mode = StreamMode::Confirm;
        c.confirm_steps = 3;
        f.set_config(c);

        // A sticky stream, so the commit / switch / release paths are all walked
        // — the lesson from §37: a uniform stream only ever tests "nothing
        // happened".
        std::vector<StreamEvent> events;
        uint64_t s = 12345u;
        DecisionAction prev = DecisionAction::HOLD;
        for (int i = 0; i < 400; ++i) {
            s = s * 6364136223846793005ull + 1442695040888963407ull;
            const uint32_t r = static_cast<uint32_t>((s >> 33) & 0xFFFFu);
            if (r < 0xCCCCu) {
                // keep the previous action most of the time
            } else {
                prev = static_cast<DecisionAction>(r % 6u);
            }
            StreamEvent e;
            f.push_decision(mk_dec(prev, 0.30f + 0.69f * (r / 65536.0f)), e);
            events.push_back(e);
        }

        DecisionJournal j;
        CHECK(j.record_stream(events, "trading") == events.size());
        CHECK(j.size() == events.size());

        // The §37 identity, extended to the record: exactly the rows the filter
        // reported as a change carry `changed`.
        int64_t changed_rows = 0;
        for (const JournalEntry& e : j.entries()) if (e.changed) ++changed_rows;
        CHECK(changed_rows == f.commits() + f.changes() + f.releases());
        CHECK(f.commits() > 0);
        CHECK(f.changes() > 0);
        CHECK(f.releases() > 0);
    }

    TEST("F2 the 369 REAL held-out rows are scored against their real labels");
    {
        TradingFixture fx;
        if (!load_trading_fixture(fx)) {
            SKIP("tests/fixtures/head_calibration/trading is absent");
            return;
        }
        CHECK(fx.n() == 369u);
        CHECK(fx.E == kE);

        // A SEEDED head. The point of this part is not the head's quality — it
        // is that the RECORD can measure it, on real data, end to end.
        DecisionHead head;
        CHECK(head.init(kE, 4242u));
        StreamingDecision f;
        CHECK(f.init(head));
        StreamingConfig c;
        c.mode = StreamMode::Confirm;
        c.confirm_steps = 3;
        f.set_config(c);
        head.set_batch_kernel(BatchKernel::Scalar);   // §36: bit-identical

        std::vector<StreamEvent> events;
        CHECK(f.push_batch(fx.h.data(), static_cast<int32_t>(fx.n()), events, nullptr));
        CHECK(events.size() == fx.n());

        FeedbackHook hook;
        CHECK(hook.observe_stream(events, "trading.AAPL.1d") == fx.n());
        CHECK(hook.journal().size() == fx.n());

        // Resolve EVERY row against its own label. `resolve_all` would be wrong
        // here: these are 369 different questions, not one.
        for (size_t i = 0; i < hook.journal().entries().size(); ++i) {
            const int64_t id = hook.journal().entries()[i].id;
            CHECK(hook.resolve(id, fx.labels[i], 0.0f, 1));
        }
        const JournalStats s = hook.stats();
        CHECK(s.predictions == 369);
        CHECK(s.verdicts == 369);           // every row got an answer
        CHECK(s.hit_rate() >= 0.0);         // MEASURED, so not -1
        CHECK(s.hit_rate() <= 1.0);
        CHECK(s.ece >= 0.0);

        // The confusion matrix must account for every label.
        int64_t total = 0;
        for (const auto& row : s.confusion)
            for (int64_t v : row) total += v;
        CHECK(total == 369);

        // HOW MANY DISTINCT ACTIONS did the head emit? This is the number §37
        // stream A could not report and the acceptance test for head fitting
        // turns on: a head that emits ONE action for all 369 rows is constant,
        // and no label set can distinguish it from a good one.
        int distinct = 0;
        {
            bool seen[static_cast<size_t>(DecisionAction::COUNT)] = {false};
            for (const JournalEntry& e : hook.journal().entries()) {
                const size_t a = static_cast<size_t>(e.predicted);
                if (a < static_cast<size_t>(DecisionAction::COUNT) && !seen[a]) {
                    seen[a] = true;
                    ++distinct;
                }
            }
        }
        CHECK(distinct >= 1);

        // THE CONTROL, and it is allowed to be VACUOUS — but only out loud.
        // With a constant head, rotating the labels changes nothing, because a
        // constant predictor is scored by the label distribution alone. The
        // record has not failed; the head has nothing to say. Asserting the
        // strict inequality here would assert something false, so instead the
        // branch is taken and named.
        const double misaligned = [&] {
            DecisionJournal mj;
            mj.record_stream(events, "k");
            for (size_t i = 0; i < mj.entries().size(); ++i)
                mj.resolve(mj.entries()[i].id,
                           fx.labels[(i + 37) % fx.n()], 0.0f, 1);
            return mj.stats().hit_rate();
        }();
        if (distinct <= 1) {
            CHECK(close(s.hit_rate(), misaligned, 1e-12));
            platform::log_info("  F2 the head is CONSTANT (%d distinct action(s)); "
                               "the label-shuffle control is VACUOUS and says so",
                               distinct);
        } else {
            CHECK(s.hit_rate() > misaligned);
        }

        platform::log_info("  F2 MEASURED (seeded head, 369 real held-out rows): "
                           "hit_rate=%.4f  ece=%.4f  commitment_rate=%.4f  "
                           "conf_gap=%.4f  distinct_actions=%d",
                           s.hit_rate(), s.ece, s.commitment_rate(),
                           s.confidence_gap, distinct);
    }

    TEST("F3 the record can tell a good head from a deliberately bad one");
    {
        // HAND-BUILT events, not the seeded head. F2 showed the seeded head is
        // CONSTANT on the real tape, and a constant predictor scores the same
        // against any label alignment — so it cannot exercise this control. The
        // subject here is the RECORD, and it must be shown to be able to fail.
        auto mk_event = [](int64_t step, DecisionAction a) {
            StreamEvent e;
            e.step            = step;
            e.valid           = true;
            e.observed_action = a;
            e.observed_conf   = 0.80f;
            e.action          = a;
            e.committed       = true;
            return e;
        };

        const int64_t N = 200;
        std::vector<StreamEvent> events;
        events.reserve(static_cast<size_t>(N));
        for (int64_t i = 0; i < N; ++i)
            events.push_back(mk_event(i + 1,
                static_cast<DecisionAction>(i % static_cast<int64_t>(DecisionAction::COUNT))));

        // Aligned: the realised action matches the prediction 70% of the time.
        auto score = [&](bool misaligned) {
            DecisionJournal j;
            j.record_stream(events, "k");
            for (size_t i = 0; i < j.entries().size(); ++i) {
                const int64_t a = static_cast<int64_t>(j.entries()[i].predicted);
                const DecisionAction realised =
                    misaligned
                        ? static_cast<DecisionAction>(
                              (a + 1) % static_cast<int64_t>(DecisionAction::COUNT))
                        : (i % 10 < 7 ? j.entries()[i].predicted
                                      : static_cast<DecisionAction>(
                                            (a + 1) % static_cast<int64_t>(DecisionAction::COUNT)));
                j.resolve(j.entries()[i].id, realised, 0.0f, 1);
            }
            return j.stats().hit_rate();
        };

        const double honest    = score(false);
        const double shuffled  = score(true);

        CHECK(close(honest, 0.70, 1e-12));
        CHECK(close(shuffled, 0.0, 1e-12));
        // THE NEGATIVE CONTROL. A deliberately rotated label set scores ZERO,
        // so the record is measuring the alignment and not merely counting rows.
        CHECK(honest > shuffled);
        // And zero here is a MEASURED zero, not the "not measured" sentinel.
        CHECK(shuffled != -1.0);
        platform::log_info("  F3 aligned hit_rate=%.4f vs misaligned=%.4f", honest,
                           shuffled);
    }

    TEST("F4 a real trading loop closes the loop end to end");
    {
        TradingFixture fx;
        if (!load_trading_fixture(fx)) {
            SKIP("tests/fixtures/head_calibration/trading is absent");
            return;
        }
        DecisionHead head;
        CHECK(head.init(kE, 4242u));

        FeedbackHook hook;
        // Walk the tape in windows, the way a paper loop would: predict, then
        // resolve against the outcome that followed.
        const size_t window = 60;
        for (size_t start = 0; start + window <= fx.n(); start += window) {
            StreamingDecision f;
            CHECK(f.init(head));
            StreamingConfig c;
            c.mode = StreamMode::Window;
            c.window = 5;
            c.confirm_steps = 3;
            f.set_config(c);

            std::vector<StreamEvent> ev;
            CHECK(f.push_batch(fx.h.data() + start * kE, static_cast<int32_t>(window),
                               ev, nullptr));
            const size_t before = hook.journal().size();
            hook.observe_stream(ev, "trading.AAPL.1d");
            const size_t added = hook.journal().size() - before;
            CHECK(added == window);

            // Resolve the window's rows against the LAST realised label in it —
            // one outcome for the window, which is what `resolve_all` models.
            hook.outcome("trading.AAPL.1d",
                         fx.labels[start + window - 1], 0.0f,
                         static_cast<int32_t>(window));
        }
        const JournalStats s = hook.stats();
        CHECK(s.predictions == static_cast<int64_t>(fx.n() - (fx.n() % window)));
        CHECK(s.unresolved == 0);
        CHECK(s.verdicts == s.predictions);
        CHECK(s.hit_rate() >= 0.0);
        platform::log_info("  F4 loop closed: %lld rows, hit_rate=%.4f, "
                           "committed=%.4f of steps, filter_hit_rate=%.4f",
                           static_cast<long long>(s.predictions), s.hit_rate(),
                           s.commitment_rate(), s.filter_hit_rate());
    }

    TEST("F5 the journal exports the rows an offline fitter consumes");    {
        DecisionJournal j;
        const int64_t a = j.record(mk_entry(DecisionAction::BUY, 0.8f, "k"));
        j.resolve(a, DecisionAction::SELL, -0.01f, 5);
        const std::string tsv = j.to_tsv();
        CHECK(has(tsv, "id\tstep\ttask_key\tpredicted"));
        CHECK(has(tsv, "BUY"));
        CHECK(has(tsv, "realised"));
        CHECK(has(tsv, "\n"));
        // One header + one row.
        size_t lines = 0;
        for (char c : tsv) if (c == '\n') ++lines;
        CHECK(lines == 2u);

        // A tab inside a task key must not be able to break the column count.
        DecisionJournal k;
        k.record(mk_entry(DecisionAction::HOLD, 0.5f, "bad\tkey"));
        const std::string t2 = k.to_tsv();
        size_t tabs = 0;
        for (char c : t2) {
            if (c == '\n') break;
            if (c == '\t') ++tabs;
        }
        CHECK(tabs == 11u);   // 12 columns, 11 separators — the tab was sanitised
    }

    TEST("F6 THE ACCEPTANCE TEST — the FITTED head on the real tape");
    {
        // The mandate's acceptance test for head fitting is: after fitting, the
        // constant-action result must DISAPPEAR. This is that test, in the one
        // place where it can be checked against the numbers §32 published.
        //
        // It is GATED on the committed blob, and it prints the SKIP reason
        // rather than passing vacuously — a gate that never executes is not a
        // gate, but neither is a test that quietly succeeds when its subject is
        // absent.
        TradingFixture fx;
        if (!load_trading_fixture(fx)) {
            SKIP("tests/fixtures/head_calibration/trading is absent");
            return;
        }
        DecisionHead head;
        CHECK(head.init(kE, 4242u));
        if (!head.load("models/heads/trading_head.bin") || !head.trained()) {
            SKIP("models/heads/trading_head.bin is absent or untrained");
            return;
        }
        CHECK(head.calibrated());   // the §32 trailer must have loaded

        StreamingDecision f;
        CHECK(f.init(head));
        StreamingConfig c;
        c.mode = StreamMode::Confirm;
        c.confirm_steps = 3;
        f.set_config(c);
        head.set_batch_kernel(BatchKernel::Scalar);

        std::vector<StreamEvent> events;
        CHECK(f.push_batch(fx.h.data(), static_cast<int32_t>(fx.n()), events, nullptr));

        FeedbackHook hook;
        hook.observe_stream(events, "trading.AAPL.1d");
        for (size_t i = 0; i < hook.journal().entries().size(); ++i)
            hook.resolve(hook.journal().entries()[i].id, fx.labels[i], 0.0f, 1);

        const JournalStats s = hook.stats();

        // ---- THE ACCEPTANCE ASSERTION --------------------------------------
        // With the SEEDED head this was 1 (see F2). A fitted projection must
        // emit more than one action, or the constant-action result has not
        // disappeared and the milestone has not landed.
        int distinct = 0;
        bool seen[static_cast<size_t>(DecisionAction::COUNT)] = {false};
        double max_conf = 0.0;
        for (const JournalEntry& e : hook.journal().entries()) {
            const size_t a = static_cast<size_t>(e.predicted);
            if (a < static_cast<size_t>(DecisionAction::COUNT) && !seen[a]) {
                seen[a] = true;
                ++distinct;
            }
            if (e.confidence > max_conf) max_conf = static_cast<double>(e.confidence);
        }
        CHECK(distinct > 1);

        // ---- cross-check against §32, from an INDEPENDENT path ---------------
        // The record re-derives the head's accuracy and its calibration error
        // from the raw h[E] + the raw labels. If these disagree with the numbers
        // tools/train_heads.py wrote into metrics.tsv, one of the two is wrong.
        double want_acc = -1.0, want_ece = -1.0;
        {
            std::ifstream mf("tests/fixtures/head_calibration/trading/metrics.tsv");
            std::string line;
            std::getline(mf, line);   // header
            while (std::getline(mf, line)) {
                while (!line.empty() && (line.back() == '\r' || line.back() == '\n'))
                    line.pop_back();
                if (line.rfind("DecisionAction", 0) != 0) continue;
                std::vector<std::string> p;
                size_t pos = 0;
                while (true) {
                    const size_t tab = line.find('\t', pos);
                    p.push_back(line.substr(
                        pos, tab == std::string::npos ? std::string::npos : tab - pos));
                    if (tab == std::string::npos) break;
                    pos = tab + 1;
                }
                if (p.size() >= 5) {
                    want_ece = std::atof(p[3].c_str());
                    want_acc = std::atof(p[4].c_str());
                }
                break;
            }
        }
        CHECK(want_acc > 0.0);
        CHECK(want_ece >= 0.0);
        CHECK(close(s.hit_rate(), want_acc, 1e-3));
        CHECK(close(s.ece, want_ece, 1e-3));

        // ---- the finding that REPLACED the constant-action one ---------------
        // The head now varies richly, and the filter still commits NOTHING:
        // every calibrated confidence is below the bar. That is fail-closed
        // working, and the margin is tiny — 0.4966 vs a default of 0.50 — so it
        // is REPORTED here rather than tuned away.
        platform::log_info("  F6 FITTED head: hit_rate=%.4f (metrics.tsv says %.4f), "
                           "ece=%.4f (says %.4f), %d distinct action(s), "
                           "max calibrated confidence=%.4f vs min_confidence=%.2f "
                           "-> %s",
                           s.hit_rate(), want_acc, s.ece, want_ece, distinct,
                           max_conf, static_cast<double>(f.config().min_confidence),
                           max_conf >= static_cast<double>(f.config().min_confidence)
                               ? "the gate CAN open"
                               : "the gate is CLOSED (fail closed)");
        CHECK(s.filter_verdicts == 0);   // nothing committed, on this tape
        CHECK(s.filter_hit_rate() == -1.0);   // and it is reported as UNMEASURED
    }
}

// ===========================================================================
//  Part G — the acceptance-shaped control
// ===========================================================================
void part_g() {    TEST("G1 MEASURED ZERO and NOT MEASURED are different answers");
    {
        // A confidently wrong head. 100 rows, all resolved, none correct.
        DecisionJournal wrong;
        for (int i = 0; i < 100; ++i) {
            const int64_t id = wrong.record(mk_entry(DecisionAction::BUY, 0.95f));
            wrong.resolve(id, DecisionAction::SELL, 0.0f, 1);
        }
        // An untested head. 100 rows, none resolved.
        DecisionJournal untested;
        for (int i = 0; i < 100; ++i) untested.record(mk_entry(DecisionAction::BUY, 0.95f));

        const double measured = wrong.stats().hit_rate();
        const double unknown  = untested.stats().hit_rate();

        CHECK(measured == 0.0);      // a real, damning zero
        CHECK(unknown == -1.0);      // nobody has ever checked
        CHECK(measured != unknown);  // THE ASSERTION THIS TEST EXISTS FOR
        // Both journals have the same number of rows and the same confidences,
        // so anything that reported on rows alone would conflate them.
        CHECK(wrong.stats().predictions == untested.stats().predictions);
    }

    TEST("G2 an anti-correlated journal cannot report a good hit rate");
    {
        DecisionJournal j;
        // confidence 0.9 -> wrong; confidence 0.1 -> right. The head is exactly
        // backwards, and the record must say so.
        for (int i = 0; i < 50; ++i) {
            const int64_t a = j.record(mk_entry(DecisionAction::BUY, 0.90f));
            j.resolve(a, DecisionAction::SELL, 0.0f, 1);
            const int64_t b = j.record(mk_entry(DecisionAction::BUY, 0.10f));
            j.resolve(b, DecisionAction::BUY, 0.0f, 1);
        }
        const JournalStats s = j.stats();
        CHECK(close(s.hit_rate(), 0.5, 1e-12));
        CHECK(s.confidence_gap < 0.0);      // inverted, and the gap says so
        CHECK(s.ece > 0.3);                 // 0.9 claims are wrong 100% of the time
    }

    TEST("G3 the record never invents an outcome");
    {
        FeedbackHook h;
        StreamEvent e;
        e.valid = true; e.observed_action = DecisionAction::BUY; e.observed_conf = 0.9f;
        h.observe(e, "k");
        // No user turn, no realised label. Nothing may be filled in.
        CHECK(h.stats().unresolved == 1);
        CHECK(h.stats().verdicts == 0);
        CHECK(h.stats().hit_rate() == -1.0);
        CHECK(h.stats().confusion.empty());
        CHECK(h.journal().entries()[0].realised == DecisionAction::ABSTAIN);
        CHECK(h.journal().entries()[0].outcome == OutcomeKind::Unknown);
    }
}

// ===========================================================================
//  Part H — the remaining public surface
//
//  Small, but every one of these is a place a caller can be misled, and an
//  untested public method on a scoring module is a claim nobody checked.
// ===========================================================================
void part_h() {
    TEST("H1 outcome_kind_name and from_name are inverses");
    {
        for (int32_t i = 0; i < static_cast<int32_t>(OutcomeKind::COUNT); ++i) {
            const OutcomeKind k = static_cast<OutcomeKind>(i);
            OutcomeKind back = OutcomeKind::COUNT;
            CHECK(outcome_kind_from_name(outcome_kind_name(k), back));
            CHECK(back == k);
        }
        OutcomeKind junk = OutcomeKind::Unknown;
        CHECK(!outcome_kind_from_name("nonsense", junk));
        // The three truth-bearing kinds are exactly Realised/Confirmed/Corrected.
        CHECK(outcome_kind_has_truth(OutcomeKind::Realised));
        CHECK(outcome_kind_has_truth(OutcomeKind::Confirmed));
        CHECK(outcome_kind_has_truth(OutcomeKind::Corrected));
        CHECK(!outcome_kind_has_truth(OutcomeKind::Unknown));
        CHECK(!outcome_kind_has_truth(OutcomeKind::Rejected));
        CHECK(!outcome_kind_has_truth(OutcomeKind::Expired));
    }

    TEST("H2 JournalEntry::to_json names every field");
    {
        JournalEntry e = mk_entry(DecisionAction::SELL, 0.4f, "k");
        e.step      = 3;
        e.held      = DecisionAction::HOLD;
        e.committed = true;
        e.changed   = true;
        e.strength  = 2.0f;
        const std::string j = e.to_json();
        CHECK(has(j, "\"predicted\":\"SELL\""));
        CHECK(has(j, "\"held\":\"HOLD\""));
        CHECK(has(j, "\"outcome\":\"unknown\""));
        CHECK(has(j, "\"committed\":true"));
        CHECK(has(j, "\"task\":\"k\""));
        CHECK(!has(j, ":nan"));
        CHECK(!has(j, ":inf"));
    }

    TEST("H3 FeedbackHook::to_json carries both stores");
    {
        FeedbackHook h;
        StreamEvent e;
        e.valid = true; e.observed_action = DecisionAction::BUY; e.observed_conf = 0.9f;
        h.observe(e, "alpha");
        h.on_user_turn("yes", "alpha");
        const std::string j = h.to_json();
        CHECK(has(j, "\"journal\":"));
        CHECK(has(j, "\"ledger\":"));
        CHECK(has(j, "\"ledger_keys\":1"));
        CHECK(has(j, "alpha"));
        CHECK(has(j, "\"trust\":"));

        // Deterministic, so two runs that recorded the same verdicts produce
        // the same string and a log diff is meaningful. `unordered_map`
        // iteration order would otherwise leak in.
        FeedbackHook g;
        StreamEvent e2;
        e2.valid = true; e2.observed_action = DecisionAction::BUY; e2.observed_conf = 0.9f;
        g.observe(e2, "alpha");
        g.on_user_turn("yes", "alpha");
        CHECK(g.to_json() == j);

        // ...and the key ORDER is sorted, not insertion order.
        FeedbackHook k;
        k.on_user_turn("yes", "zeta");
        k.on_user_turn("yes", "alpha");
        const std::string kj = k.to_json();
        CHECK(kj.find("alpha") < kj.find("zeta"));
    }

    TEST("H4 FeedbackHook::observe_decision records a single-shot head call");
    {
        FeedbackHook h;
        DecisionResult d = mk_dec(DecisionAction::HEDGE, 0.7f);
        d.fast_path = true;
        const int64_t id = h.observe_decision(d, "k");
        CHECK(id != 0);
        CHECK(h.journal().size() == 1u);
        CHECK(h.journal().find(id)->predicted == DecisionAction::HEDGE);
        CHECK(h.journal().find(id)->committed);
        // ...and it resolves like any other row.
        CHECK(h.resolve(id, DecisionAction::HEDGE, 0.0f, 1));
        CHECK(close(h.stats().hit_rate(), 1.0, 1e-12));
    }

    TEST("H5 clear() empties everything, including the error");
    {
        FeedbackHook h;
        StreamEvent e;
        e.valid = true; e.observed_action = DecisionAction::BUY; e.observed_conf = 0.9f;
        h.observe(e, "k");
        h.on_user_turn("yes", "k");
        CHECK(h.journal().size() == 1u);
        CHECK(h.feedback_size() == 1u);

        h.clear();
        CHECK(h.journal().size() == 0u);
        CHECK(h.feedback_size() == 0u);
        CHECK(h.trust("k") == 0.5);            // back to neutral
        CHECK(h.stats().hit_rate() == -1.0);   // and back to UNMEASURED
        // A cleared journal restarts its id counter, so the next row is id 1.
        CHECK(h.observe(e, "k") == 1);
    }

    TEST("H6 a journal restored from disk keeps scoring correctly");
    {
        // The round-trip is not just about bytes: a restored journal must still
        // produce the same STATISTICS, or the persistence is cosmetic.
        const std::string path = tmp_path("rescore");
        DecisionJournal j;
        for (int i = 0; i < 10; ++i) {
            const int64_t id = j.record(mk_entry(DecisionAction::BUY, 0.8f));
            j.resolve(id, i < 6 ? DecisionAction::BUY : DecisionAction::SELL, 0.0f, 1);
        }
        const std::string before = j.stats().to_json();
        CHECK(j.save(path));

        DecisionJournal r;
        CHECK(r.load(path));
        CHECK(r.stats().to_json() == before);
        CHECK(close(r.stats().hit_rate(), 0.6, 1e-12));
        std::remove(path.c_str());
    }
}

} // namespace

int main() {
    platform::log_info("=== test_feedback_hook (MILESTONE 10, Phase 2.3) ===");
    part_a();
    part_b();
    part_c();
    part_d();
    part_e();
    part_f();
    part_g();
    part_h();
    platform::log_info("=== RESULT: %d passed, %d failed, %d skipped ===",
                       g_passed, g_failed, g_skipped);
    return g_failed == 0 ? 0 : 1;
}
