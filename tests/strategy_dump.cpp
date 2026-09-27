// =============================================================================
//  OmniSeed — tests/strategy_dump.cpp
//  Dumps the C++ strategy zoo + router + sniper as JSON, so that
//  tests/test_strategy_parity.py can diff it field-by-field against the Python
//  oracle (tools/monster/strategies.py, router.py, sniper_engine.py).
//
//  This binary is the ONLY reason the Python modules can be retired from the
//  runtime path without touching them: the oracle keeps its job as the oracle,
//  and the port has to prove it agrees before it is trusted.
//
//  Usage:
//    strategy_dump <bars.csv> [--sniper] [--ensemble] [--legacy] [--raw]
//                              [--limit N]
//
//  Both sides of every comparison use the engine DEFAULTS (regime window 200,
//  sniper and router configs as shipped), so there is no knob to accidentally
//  set on one side only. The regime window itself is pinned separately by
//  tests/test_regime_parity.py.
//
//  Numbers are printed with %.17g so they round-trip exactly; every NaN prints
//  as `null`, which is the JSON-correct way to say "no reading" (and is why a
//  sentinel 0 would have been a lie).
// =============================================================================
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "omniseed/trading/sniper.h"
#include "omniseed/trading/trading_engine.h"

using namespace omniseed;
using namespace omniseed::trading;

namespace {

void put_num(const char*& sep, const char* key, double v) {
    std::printf("%s\"%s\":", sep, key);
    sep = ",";
    if (std::isnan(v) || std::isinf(v)) std::printf("null");
    else std::printf("%.17g", v);
}

void put_int(const char*& sep, const char* key, long long v) {
    std::printf("%s\"%s\":%lld", sep, key, v);
    sep = ",";
}

void put_bool(const char*& sep, const char* key, bool v) {
    std::printf("%s\"%s\":%s", sep, key, v ? "true" : "false");
    sep = ",";
}

// Reasons are machine-generated (format strings only, no user text), so a
// minimal escape of the two characters that would actually break JSON suffices.
void put_str(const char*& sep, const char* key, const std::string& v) {
    std::printf("%s\"%s\":\"", sep, key);
    sep = ",";
    for (char c : v) {
        if (c == '"' || c == '\\') std::printf("\\%c", c);
        else std::printf("%c", c);
    }
    std::printf("\"");
}

} // namespace

int main(int argc, char** argv) {
    std::string csv;
    int32_t limit = -1;
    bool want_sniper = false;
    bool want_ensemble = false;
    bool legacy = false;
    bool tracked = true;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--limit" && i + 1 < argc) limit = std::atoi(argv[++i]);
        else if (a == "--sniper") want_sniper = true;
        else if (a == "--ensemble") want_ensemble = true;
        else if (a == "--legacy") legacy = true;
        else if (a == "--raw") tracked = false;
        else if (!a.empty() && a[0] != '-') csv = a;
    }
    if (csv.empty()) {
        std::fprintf(stderr,
                     "usage: strategy_dump <bars.csv> [--limit N] "
                     "[--sniper] [--ensemble] [--legacy] [--raw]\n");
        return 2;
    }

    std::vector<Bar> bars;
    std::string err;
    if (!load_bars_csv(csv, bars, err)) {
        std::fprintf(stderr, "load_bars_csv failed: %s\n", err.c_str());
        return 2;
    }
    if (limit >= 0 && static_cast<int32_t>(bars.size()) > limit)
        bars.resize(static_cast<size_t>(limit));

    const StrategyConfig scfg;
    const StrategySeries series(bars, scfg);

    // Regimes: the sniper and the router both need them, and the oracle's
    // `RG.scan(bars)` defaults to TRACKED (hysteresis applied). `--raw` selects
    // the untracked series for the parity test's own A/B.
    std::vector<RegimeState> regimes;
    if (tracked) {
        RegimeTracker tracker;
        regimes = tracker.run(bars);
    } else {
        RegimeEngine engine;
        regimes = engine.scan_raw(bars);
    }

    const RouterConfig rcfg;

    // Built ONCE, outside the loop: every series in it is a full pass over the
    // bars, so constructing it per bar would be quietly quadratic.
    SniperConfig sncfg;
    sncfg.regime_mode = legacy ? "legacy" : "advanced";
    sncfg.ensemble = want_ensemble;
    std::unique_ptr<Prepared> sniper_prep;
    if (want_sniper) sniper_prep.reset(new Prepared(bars, sncfg));

    std::printf("{\"n\":%zu,\"tracked\":%s,\"rows\":[", bars.size(),
                tracked ? "true" : "false");

    for (size_t i = 0; i < series.n(); ++i) {
        if (i) std::printf(",");
        const RegimeState* reg = i < regimes.size() ? &regimes[i] : nullptr;

        const std::vector<StrategySignal> sigs =
            all_signals(series, i, reg, scfg, nullptr);
        const EnsembleVerdict v = route(sigs, reg, rcfg, series.ts()[i]);

        const char* sep = "";
        std::printf("{");
        put_int(sep, "i", static_cast<long long>(i));
        put_int(sep, "ts", static_cast<long long>(series.ts()[i]));
        put_str(sep, "regime_label", reg ? reg->label : std::string("range"));

        std::printf("%s\"signals\":[", sep);
        sep = ",";
        for (size_t k = 0; k < sigs.size(); ++k) {
            if (k) std::printf(",");
            const char* s2 = "";
            std::printf("{");
            put_str(s2, "name", sigs[k].name);
            put_num(s2, "direction", sigs[k].direction);
            put_num(s2, "confidence", sigs[k].confidence);
            put_str(s2, "fit", to_string(sigs[k].regime_fit));
            put_str(s2, "reason", sigs[k].reason);
            std::printf("}");
        }
        std::printf("]");

        put_num(sep, "conviction", v.conviction);
        put_num(sep, "agreement", v.agreement);
        put_num(sep, "weight", v.weight);
        put_num(sep, "w_trend", v.w_trend);
        put_int(sep, "active", v.active);
        put_bool(sep, "opposed_long", v.opposed_long());
        put_bool(sep, "veto_long", should_veto_long(v, rcfg));
        put_num(sep, "size_factor", v.size_factor());

        if (want_sniper) {
            const SniperVerdict sv = evaluate(*sniper_prep, i, nullptr, sncfg);

            std::printf("%s\"sniper\":{", sep);
            sep = ",";
            const char* s3 = "";
            put_num(s3, "score", sv.score);
            put_num(s3, "micro", sv.micro);
            put_num(s3, "tech", sv.tech);
            put_num(s3, "regime_score", sv.regime_score);
            put_num(s3, "cross", sv.cross);
            put_int(s3, "votes", sv.votes);
            put_str(s3, "regime", sv.regime);
            put_bool(s3, "veto", sv.veto);
            put_str(s3, "veto_reason", sv.veto_reason);
            put_num(s3, "trend_score", sv.trend_score);
            put_num(s3, "half_life", sv.half_life);
            put_num(s3, "conviction", sv.conviction);
            put_num(s3, "agreement", sv.agreement);
            put_bool(s3, "propose", sv.propose(sncfg));
            std::printf("%s\"factors\":[", s3);
            s3 = ",";
            for (size_t k = 0; k < sv.factors.size(); ++k) {
                if (k) std::printf(",");
                std::printf("\"%s\"", sv.factors[k].c_str());
            }
            std::printf("]}");
        }
        std::printf("}");
    }
    std::printf("]}\n");
    return 0;
}
