// =============================================================================
//  OmniSeed — tests/test_sub_router.cpp   (T072c: LEVEL-2 routing)
//
//    * Default-off parity: with enable=false every regime abstains, so the
//      pre-T072 path cannot change (the M4 w_vol guard pattern).
//    * On-arm slot table: trend_up->momentum, range->mean_reversion,
//      high_vol->mean_reversion ONLY when the MR setup is verified.
//    * Fail-closed: conviction below the bar, NaN conviction, trend_down and
//      unknown labels all abstain. The slot never claims a probability.
//
//  Deterministic, no I/O, no look-ahead (one bar's verdict only).
// =============================================================================
#include "omniseed/core/platform.h"
#include "omniseed/trading/sub_router.h"

#include <cmath>
#include <cstdio>
#include <string>

using namespace omniseed;
using namespace omniseed::trading;

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
    platform::log_info("OmniSeed level-2 sub-router test");

    const SubRouterConfig on;          // enable defaults FALSE...
    SubRouterConfig       active = on;
    active.enable = true;              // ...the on-arm flips it explicitly.
    active.min_conviction = 0.60;

    TEST("default-off parity: no regime ever gets a slot");
    CHECK(route("trend_up", 0.99, true, on) == SpecialistSlot::Abstain);
    CHECK(route("range", 0.99, false, on) == SpecialistSlot::Abstain);
    CHECK(route("high_vol", 0.99, true, on) == SpecialistSlot::Abstain);

    TEST("trend_up with conviction -> momentum");
    CHECK(route("trend_up", 0.70, false, active) == SpecialistSlot::Momentum);

    TEST("trend_up below the conviction bar -> abstain");
    CHECK(route("trend_up", 0.40, false, active) == SpecialistSlot::Abstain);
    CHECK(route("trend_up", 0.60, false, active) == SpecialistSlot::Momentum);

    TEST("range -> mean_reversion");
    CHECK(route("range", 0.75, false, active) == SpecialistSlot::MeanReversion);

    TEST("high_vol needs the VERIFIED mean-reversion setup");
    CHECK(route("high_vol", 0.80, true, active) == SpecialistSlot::MeanReversion);
    CHECK(route("high_vol", 0.80, false, active) == SpecialistSlot::Abstain);

    TEST("trend_down is never a specialist pick (regime veto owns it)");
    CHECK(route("trend_down", 0.99, true, active) == SpecialistSlot::Abstain);

    TEST("unknown labels fail closed");
    CHECK(route("", 0.99, true, active) == SpecialistSlot::Abstain);
    CHECK(route("sideways?", 0.99, true, active) == SpecialistSlot::Abstain);

    TEST("NaN conviction cannot clear the bar");
    CHECK(route("trend_up", std::nan(""), false, active) == SpecialistSlot::Abstain);
    CHECK(route("range", std::nan(""), false, active) == SpecialistSlot::Abstain);

    TEST("slot_name is stable for all three slots");
    CHECK(std::string(slot_name(SpecialistSlot::Momentum)) == "momentum");
    CHECK(std::string(slot_name(SpecialistSlot::MeanReversion)) == "mean_reversion");
    CHECK(std::string(slot_name(SpecialistSlot::Abstain)) == "abstain");

    std::printf("\nRESULT: %d passed, %d failed\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
