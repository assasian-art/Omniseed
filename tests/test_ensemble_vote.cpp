// =============================================================================
//  OmniSeed — tests/test_ensemble_vote.cpp   (T073c: CROSS-HEAD VOTING)
//
//    * Default-off parity: enable=false abstains on EVERY input.
//    * Confidence-weighted majority, exact weight math asserted.
//    * Fail-closed: invalid/below-bar/NaN ballots never count; empty input,
//      tie, and majority-fail all land on abstain with a reason.
//
//  Deterministic, no I/O.
// =============================================================================
#include "omniseed/core/platform.h"
#include "omniseed/agent/ensemble_vote.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace omniseed;
using namespace omniseed::agent;

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

int main() {
    platform::enable_utf8_console();
    platform::log_info("OmniSeed cross-head ensemble vote test");

    EnsembleConfig on;
    on.enable = true;
    EnsembleConfig off;  // enable=false by default

    const std::vector<Vote> agree = {
        {"long", 0.90, true}, {"long", 0.80, true}, {"long", 0.70, true}};

    TEST("default-off: every input abstains");
    {
        EnsembleVerdict v = vote(agree, off);
        CHECK(v.action == "abstain" && v.reason == "no-votes" && v.voters == 0);
    }

    TEST("three agreeing voters -> majority, confidence in (0,1)");
    {
        EnsembleVerdict v = vote(agree, on);
        CHECK(v.action == "long");
        CHECK(v.reason == "majority" && v.voters == 3);
        CHECK(v.confidence > 0.0 && v.confidence < 1.0);
        // share = 1.0, mean = 0.80 => 0.80 exactly
        CHECK(v.confidence > 0.7999 && v.confidence < 0.8001);
        CHECK(!v.disagreement);
    }

    TEST("2-vs-1 split: winner by weight; low-conf ballot counted correctly");
    {
        std::vector<Vote> split = {{"long", 0.90, true}, {"long", 0.80, true},
                                   {"short", 0.60, true}};
        EnsembleVerdict v = vote(split, on);
        CHECK(v.action == "long" && v.voters == 3);
        CHECK(v.reason == "majority");
        // share = 1.7/2.3
        double expect = (1.7 / 2.3) * ((0.90 + 0.80 + 0.60) / 3.0);
        CHECK(v.confidence > expect - 1e-9 && v.confidence < expect + 1e-9);
    }

    TEST("a ballot below min_vote_conf is never cast");
    {
        std::vector<Vote> weak = {{"long", 0.90, true}, {"long", 0.55, true},
                                  {"short", 0.40, true}};
        EnsembleVerdict v = vote(weak, on);
        CHECK(v.voters == 2);  // the 0.40 ballot never entered the tally
        CHECK(v.action == "long" && v.reason == "majority");
        // share = 1.0, mean over the TWO cast = 0.725
        double expect = (0.90 + 0.55) / 2.0;
        CHECK(v.confidence > expect - 1e-9 && v.confidence < expect + 1e-9);
    }

    TEST("tie within eps -> disagreement, action abstains");
    {
        std::vector<Vote> tie = {{"long", 0.70, true}, {"short", 0.68, true}};
        EnsembleVerdict v = vote(tie, on);
        CHECK(v.disagreement && v.action == "abstain");
        CHECK(v.reason == "disagreement");
    }

    TEST("no valid ballots -> no-votes");
    {
        std::vector<Vote> none = {{"long", 0.90, false},   // unfitted head
                                  {"", 0.90, true},        // empty action
                                  {"abstain", 0.90, true}};  // self-abstain
        EnsembleVerdict v = vote(none, on);
        CHECK(v.action == "abstain" && v.reason == "no-votes" && v.voters == 0);
    }

    TEST("NaN / out-of-range confidences are rejected as ballots");
    {
        std::vector<Vote> bad = {{"long", std::nan(""), true},
                                 {"long", 1.5, true}};
        EnsembleVerdict v = vote(bad, on);
        CHECK(v.voters == 0 && v.action == "abstain");
    }

    TEST("single voter above bar -> reason 'single'");
    {
        EnsembleVerdict v = vote({{"hold", 0.95, true}}, on);
        CHECK(v.action == "hold" && v.reason == "single");
        CHECK(v.confidence <= 0.99);
    }

    TEST("to_json carries every field");
    {
        std::string j = vote(agree, on).to_json();
        CHECK(j.find("\"action\":\"long\"") != std::string::npos);
        CHECK(j.find("\"reason\":\"majority\"") != std::string::npos);
        CHECK(j.find("\"voters\":3") != std::string::npos);
        CHECK(j.find("\"disagreement\":false") != std::string::npos);
    }

    std::printf("\nRESULT: %d passed, %d failed\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
