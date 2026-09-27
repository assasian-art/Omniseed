// =============================================================================
//  OmniSeed — tests/regime_dump.cpp
//
//  Dumps the C++ regime engine's reading for every bar as JSON lines, so the
//  Python oracle (tools/monster/regime.py) can be diffed against it by
//  tests/test_regime_parity.py.
//
//  This exists because a 700-line numeric hand-port WILL drift from its
//  original, and a silently-diverging second implementation is worse than no
//  port at all. Printing %.17g (round-trip exact) means the comparison is
//  about the maths, not about float formatting.
//
//  usage: regime_dump <bars.csv> [--window N] [--raw|--tracked]
// =============================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "omniseed/trading/regime_engine.h"
#include "omniseed/trading/trading_engine.h"

using namespace omniseed::trading;

namespace {

// Every numeric field is emitted with a LEADING comma, so the row is built by
// simple concatenation with no "is this the first?" bookkeeping to get wrong.
// %.17g round-trips a double exactly, so the parity diff is about the maths,
// not about float formatting.
void put_num(const char* key, double v) {
    std::printf(",\"%s\":", key);
    if (v != v) std::printf("null");
    else std::printf("%.17g", v);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: regime_dump <bars.csv> [--window N] "
                             "[--raw|--tracked]\n");
        return 2;
    }
    std::string csv = argv[1];
    int32_t window = 200;
    bool tracked = false;
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--window") == 0 && i + 1 < argc) {
            window = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--tracked") == 0) {
            tracked = true;
        } else if (std::strcmp(argv[i], "--raw") == 0) {
            tracked = false;
        }
    }

    std::vector<Bar> bars;
    std::string err;
    if (!load_bars_csv(csv, bars, err)) {
        std::fprintf(stderr, "load failed: %s\n", err.c_str());
        return 2;
    }

    RegimeConfig cfg;
    cfg.window = window;

    std::vector<RegimeState> out;
    if (tracked) {
        RegimeTracker tr(cfg);
        out = tr.run(bars);
    } else {
        RegimeEngine eng(cfg);
        out = eng.scan_raw(bars);
    }

    std::printf("{\"n\":%zu,\"rows\":[", out.size());
    for (size_t i = 0; i < out.size(); ++i) {
        const RegimeState& s = out[i];
        if (i) std::printf(",");
        std::printf("{\"i\":%zu,\"ts\":%lld",
                    i, static_cast<long long>(s.ts));
        put_num("trend_score", s.trend_score);
        put_num("vol_score", s.vol_score);
        put_num("adx", s.adx);
        put_num("er", s.er);
        put_num("chop", s.chop);
        put_num("vr", s.vr);
        put_num("vr_z", s.vr_z);
        put_num("hurst", s.hurst);
        put_num("r2", s.r2);
        put_num("rho1", s.rho1);
        put_num("yz_vol", s.yz_vol);
        put_num("half_life", s.half_life);
        std::printf(",\"label\":\"%s\",\"direction\":\"%s\",\"stressed\":%s}",
                    s.label.c_str(), s.direction.c_str(),
                    s.stressed ? "true" : "false");
    }
    std::printf("]}\n");
    return 0;
}
