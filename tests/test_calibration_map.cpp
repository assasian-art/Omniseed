// =============================================================================
//  OmniSeed — tests/test_calibration_map.cpp  (T111 / INNOVATIONS I8)
//
//  Fail-closed contract of the runtime calibration consumer:
//    accepted map        -> nearest-bin calibrated accuracy + gate works
//    missing file        -> identity fallback, gate SHUT
//    accepted:false      -> identity fallback, gate SHUT (keep-old evidence)
//    malformed bins      -> identity fallback, gate SHUT
//    1-bin "curve"       -> identity fallback, gate SHUT
//  Standalone printf harness; fixtures are written to temp files.
// =============================================================================
#include "omniseed/runtime/calibration.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

using omniseed::CalibrationMap;

static int g_passed = 0;
static int g_failed = 0;

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (cond) { ++g_passed; }                                          \
        else {                                                             \
            ++g_failed;                                                    \
            std::printf("FAIL  (line %d): %s\n", __LINE__, #cond);         \
        }                                                                  \
    } while (0)

static bool near(double a, double b, double eps = 1e-6) {
    return (a - b) < eps && (b - a) < eps;
}

static std::string write_temp(const char* name, const std::string& body) {
    std::string path = std::string("./") + name;
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f << body;
    return path;
}

int main() {
    // ---- accepted map: nearest-bin calibration + gate ----------------------
    const std::string good =
        "{\n"
        "  \"accepted\": true,\n"
        "  \"bins\": [[0.2, 0.15, 10], [0.5, 0.45, 10], [0.9, 0.88, 10]],\n"
        "  \"brier_identity\": 0.12,\n"
        "  \"brier_isotonic\": 0.09,\n"
        "  \"corpus_fingerprint\": \"abc123def456\",\n"
        "  \"journal\": \"feedback_journal.tsv\",\n"
        "  \"n\": 30,\n"
        "  \"reason\": \"improved-holdout-brier\",\n"
        "  \"timestamp\": 1792300000\n"
        "}\n";
    {
        CalibrationMap m;
        const std::string p = write_temp("cal_good.json", good);
        CHECK(m.load(p));
        CHECK(m.loaded());
        CHECK(m.bin_count() == 3);
        CHECK(near(m.calibrate(0.5), 0.45));      // exact bin
        CHECK(near(m.calibrate(0.51), 0.45));    // nearest bin
        CHECK(near(m.calibrate(0.9), 0.88));
        CHECK(near(m.calibrate(0.05), 0.15));    // below lowest -> lowest bin
        CHECK(m.fingerprint() == "abc123def456");
        CHECK(m.timestamp() == 1792300000);
        // gate: measured accuracy at the bin clears / misses the bar
        CHECK(m.trustworthy(0.9, 0.8));    // 0.88 >= 0.8
        CHECK(m.trustworthy(0.5, 0.4));    // 0.45 >= 0.4
        CHECK(!m.trustworthy(0.2, 0.8));   // 0.15 <  0.8
        CHECK(!m.trustworthy(0.9, 0.95));  // 0.88 <  0.95
    }

    // ---- missing file: identity fallback, gate SHUT ------------------------
    {
        CalibrationMap m;
        CHECK(!m.load("./definitely_missing_calibration.json"));
        CHECK(!m.loaded());
        CHECK(near(m.calibrate(0.7), 0.7));      // honest identity fallback
        CHECK(!m.trustworthy(0.99, 0.5));       // fail-closed: no map, no pass
    }

    // ---- accepted:false (keep-old verdict): never a calibration ------------
    {
        CalibrationMap m;
        const std::string p = write_temp(
            "cal_rejected.json",
            "{\"accepted\": false, \"bins\": [[0.5, 0.5, 3]], "
            "\"reason\": \"keep-old\"}\n");
        CHECK(!m.load(p));
        CHECK(!m.loaded());
        CHECK(near(m.calibrate(0.42), 0.42));
        CHECK(!m.trustworthy(0.99, 0.1));
    }

    // ---- malformed bins: fail closed ---------------------------------------
    {
        CalibrationMap m;
        const std::string p = write_temp(
            "cal_malformed.json",
            "{\"accepted\": true, \"bins\": [[oops, 2, 3], [1, 2, 3]]}\n");
        CHECK(!m.load(p));
        CHECK(!m.loaded());
        CHECK(near(m.calibrate(0.3), 0.3));
        CHECK(!m.trustworthy(0.9, 0.1));
    }

    // ---- 1-bin curve is not a calibration ----------------------------------
    {
        CalibrationMap m;
        const std::string p = write_temp(
            "cal_onebin.json",
            "{\"accepted\": true, \"bins\": [[0.5, 0.4, 30]]}\n");
        CHECK(!m.load(p));
        CHECK(!m.loaded());
    }

    // ---- unsorted bins still work (consumer sorts by confidence) -----------
    {
        CalibrationMap m;
        const std::string p = write_temp(
            "cal_unsorted.json",
            "{\"accepted\": true, \"bins\": [[0.9, 0.88, 5], "
            "[0.2, 0.15, 5], [0.5, 0.45, 5]]}\n");
        CHECK(m.load(p));
        CHECK(near(m.calibrate(0.5), 0.45));
        CHECK(near(m.calibrate(0.87), 0.88));
    }

    // ---- empty / truncated input -------------------------------------------
    {
        CalibrationMap m;
        CHECK(!m.load_from_string(""));
        CHECK(!m.load_from_string("{"));
        CHECK(!m.load_from_string("{\"accepted\": true}"));
        CHECK(!m.loaded());
    }

    std::printf("omniseed_calibration: %d passed, %d failed\n", g_passed,
                g_failed);
    return g_failed == 0 ? 0 : 1;
}
