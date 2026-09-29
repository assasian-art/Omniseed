// =============================================================================
//  OmniSeed — tests/test_regime_authority.cpp
//
//  §46: the learned `trading.regime` head must be STRUCTURALLY unable to gate a
//  strategy. `docs/EDGE_RESEARCH.md` §45 demoted it (0.7046 accuracy against a
//  0.8699 majority-class rate — 0.810x a constant predictor), but a demotion
//  that lives only in prose is not a demotion: without a guard, a later caller
//  could hand the head's label straight to the sniper's veto.
//
//  This test proves the guard is real, and — just as important — that it is not
//  vacuous. Both directions are asserted everywhere:
//
//    Part A  the type:  a learned / external label mints a NON-gating token;
//                       an engine-produced state mints a gating one. Includes
//                       the FORGE hole: a hand-built RegimeState (which carries
//                       a learned label and never went through the engine) is
//                       refused, because `engine_minted` is false.
//    Part B  the veto:  `regime_veto()` returns "" for a learned label — and
//                       "counter-regime" for the SAME label through the engine
//                       factory. The refusal is the SOURCE's doing, not a dead
//                       branch.
//    Part C  the wire:  a real sniper run over real bars is authorised
//                       (`regime_gated == true`, source "rule-engine"), so the
//                       guard does not simply disable the regime layer.
//    Part D  the blob:  the ACTUAL fitted `trading_regime_head.bin`, run on the
//                       369 held-out rows, produces labels that WOULD gate
//                       (trend_down / high_vol on many rows) — and every one of
//                       them is refused. Gated on the committed fixture.
//
//  Fully offline: no model, no GGUF, no network.
// =============================================================================
#include "omniseed/classification_head.h"
#include "omniseed/core/platform.h"
#include "omniseed/trading/regime_authority.h"
#include "omniseed/trading/regime_engine.h"
#include "omniseed/trading/sniper.h"
#include "omniseed/trading/trading_engine.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace omniseed;
using namespace omniseed::trading;

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

#define SKIP(why)                                                          \
    do {                                                                   \
        ++g_skipped;                                                       \
        platform::log_info("SKIP  %s: %s", g_current.c_str(), why);        \
        return;                                                            \
    } while (0)

namespace {

// ---------------------------------------------------------------------------
// A deterministic bar series, so the sniper path is exercised without a CSV.
// ---------------------------------------------------------------------------
std::vector<Bar> make_bars(size_t n, double drift, double noise) {
    std::vector<Bar> bars;
    bars.reserve(n);
    double px = 100.0;
    uint32_t s = 12345u;
    for (size_t i = 0; i < n; ++i) {
        s = s * 1103515245u + 12345u;
        const double u = static_cast<double>((s >> 16) & 0x7fffu) / 32767.0 - 0.5;
        const double step = drift + noise * u;
        const double prev = px;
        px = px * (1.0 + step);
        if (px < 1.0) px = 1.0;
        Bar b;
        b.time = static_cast<int64_t>(1700000000 + i * 86400);
        b.open = prev;
        b.close = px;
        b.high = std::max(prev, px) * 1.002;
        b.low = std::min(prev, px) * 0.998;
        b.volume = 1000.0 + 50.0 * static_cast<double>(i % 7);
        bars.push_back(b);
    }
    return bars;
}

// Minimal <f4> reader for the committed h[E] fixture.
std::vector<float> load_f32(const std::string& path, int32_t E, int32_t& rows) {
    rows = 0;
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    const std::streamoff sz = f.tellg();
    if (sz <= 0 || E <= 0) return {};
    const int64_t n = static_cast<int64_t>(sz) / (4 * E);
    if (n <= 0) return {};
    std::vector<float> out(static_cast<size_t>(n) * static_cast<size_t>(E));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(sz));
    if (!f) return {};
    rows = static_cast<int32_t>(n);
    return out;
}

bool file_exists(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return static_cast<bool>(f);
}

// =============================================================================
//  Part A — the type and its mints
// =============================================================================
void test_type_mints() {
    // (A1) default-constructed: nothing minted it, so it cannot gate.
    {
        GatingRegime g;
        CHECK(g.source() == RegimeSource::Unset);
        CHECK(g.can_gate() == false);
        CHECK(g.authorize() == false);
        CHECK(std::strlen(g.refusal_reason()) > 0);
    }

    // (A2) a LEARNED label mints a non-gating token, and says why.
    {
        const GatingRegime g = GatingRegime::from_learned("trend_down");
        CHECK(g.source() == RegimeSource::LearnedHead);
        CHECK(g.label() == "trend_down");        // the label is carried, not lost
        CHECK(g.can_gate() == false);
        CHECK(g.authorize() == false);
        CHECK(std::string(g.refusal_reason()).find("DEMOTED") != std::string::npos);
        CHECK(g.detail().find("gate=no") != std::string::npos);
    }

    // (A3) an EXTERNAL label cannot gate either.
    {
        const GatingRegime g = GatingRegime::from_external("high_vol");
        CHECK(g.source() == RegimeSource::External);
        CHECK(g.can_gate() == false);
    }

    // (A4) the deterministic SMA-slope rule MAY gate.
    {
        const GatingRegime g = GatingRegime::from_legacy_rule("trend_down");
        CHECK(g.source() == RegimeSource::LegacyRule);
        CHECK(g.can_gate() == true);
        CHECK(std::strlen(g.refusal_reason()) == 0);
        CHECK(g.detail().find("gate=yes") != std::string::npos);
    }

    // (A5) the rule ENGINE's own output MAY gate.
    {
        RegimeEngine eng;
        const std::vector<Bar> bars = make_bars(320, 0.0016, 0.004);
        const RegimeState st = eng.detect(bars, bars.size() - 1);
        CHECK(st.engine_minted == true);          // the engine marks its output
        const GatingRegime g = GatingRegime::from_engine(st);
        CHECK(g.source() == RegimeSource::RuleEngine);
        CHECK(g.can_gate() == true);
        CHECK(g.label() == st.label);
    }

    // (A6) THE FORGE HOLE. A hand-built RegimeState — exactly how a caller
    //      would try to smuggle a learned label into the gate — carries
    //      `engine_minted == false`, so `from_engine` refuses it.
    {
        RegimeState forged;
        forged.label = "trend_down";       // the learned head's answer, faked
        CHECK(forged.engine_minted == false);
        const GatingRegime g = GatingRegime::from_engine(forged);
        CHECK(g.source() == RegimeSource::External);   // downgraded, not accepted
        CHECK(g.can_gate() == false);
    }

    // (A7) `engine_minted` is set by detect() and survives the tracker.
    {
        RegimeTracker tr;
        const std::vector<Bar> bars = make_bars(320, 0.0016, 0.004);
        const RegimeState s = tr.step(bars, bars.size() - 1);
        CHECK(s.engine_minted == true);
        CHECK(GatingRegime::from_engine(s).can_gate() == true);
    }
}

// =============================================================================
//  Part B — the veto decision
// =============================================================================
void test_regime_veto() {
    // (B1) authorised hostile regimes DO veto (the control: the gate is not dead).
    {
        const GatingRegime down = GatingRegime::from_legacy_rule("trend_down");
        const GatingRegime vol = GatingRegime::from_legacy_rule("high_vol");
        CHECK(regime_veto(down, false) == "counter-regime");
        CHECK(regime_veto(vol, false) == "high-vol-needs-mean-reversion");
        // A confirmed mean-reversion setup is the escape hatch.
        CHECK(regime_veto(down, true).empty());
        CHECK(regime_veto(vol, true).empty());
        // A friendly regime never vetoes.
        CHECK(regime_veto(GatingRegime::from_legacy_rule("trend_up"), false).empty());
        CHECK(regime_veto(GatingRegime::from_legacy_rule("range"), false).empty());
    }

    // (B2) THE HEADLINE. The SAME hostile labels, minted as LEARNED, veto
    //      NOTHING — even with no mean-reversion confirmation to excuse them.
    {
        const GatingRegime down = GatingRegime::from_learned("trend_down");
        const GatingRegime vol = GatingRegime::from_learned("high_vol");
        CHECK(regime_veto(down, false).empty());
        CHECK(regime_veto(vol, false).empty());
    }

    // (B3) ...and an external/forged label is equally inert.
    {
        RegimeState forged;
        forged.label = "trend_down";
        const GatingRegime g = GatingRegime::from_engine(forged);
        CHECK(regime_veto(g, false).empty());
    }

    // (B4) BOTH DIRECTIONS ON ONE LABEL: identical string, opposite outcome,
    //      decided ONLY by provenance. This is the assertion that makes B2 mean
    //      something rather than test a branch that always returns "".
    {
        const std::string label = "trend_down";
        const bool gated = !regime_veto(GatingRegime::from_legacy_rule(label), false).empty();
        const bool refused = regime_veto(GatingRegime::from_learned(label), false).empty();
        CHECK(gated && refused);
    }
}

// =============================================================================
//  Part C — a real sniper run is authorised
// =============================================================================
void test_sniper_path_authorised() {
    SniperConfig cfg;
    cfg.regime_mode = "advanced";
    const std::vector<Bar> bars = make_bars(320, 0.0012, 0.004);
    const Prepared p(bars, cfg);
    CHECK(p.has_regimes());
    CHECK(p.regimes().size() == bars.size());

    int authorised = 0, refused = 0, gated_verdicts = 0;
    for (size_t i = 200; i < bars.size(); ++i) {
        const SniperVerdict v = evaluate(p, i, nullptr, cfg);
        if (v.regime_gated) ++authorised; else ++refused;
        if (v.veto && !v.veto_reason.empty()) ++gated_verdicts;
    }
    // The guard must NOT disable the regime layer on the normal path.
    CHECK(authorised > 0);
    CHECK(refused == 0);
    platform::log_info("      sniper: %d bars authorised, %d refused, %d vetoed",
                       authorised, refused, gated_verdicts);
}

// =============================================================================
//  Part D — the REAL fitted blob, on the REAL held-out rows
// =============================================================================
void test_fitted_blob_cannot_gate() {
    const std::string bin = "models/heads/trading_regime_head.bin";
    const std::string fx = "tests/fixtures/head_calibration/trading/hidden.f32";
    if (!file_exists(bin) || !file_exists(fx)) {
        ++g_skipped;
        platform::log_info("SKIP  %s: blob or fixture absent", g_current.c_str());
        return;
    }

    ClassificationHead head;
    CHECK(head.load(bin));
    CHECK(head.ready());
    CHECK(head.trained());
    CHECK(head.hidden_size() == 768);
    CHECK(head.label_set_count() == 1);
    CHECK(head.label_set(0).name == "trading.regime");

    int32_t n = 0;
    const std::vector<float> H = load_f32(fx, head.hidden_size(), n);
    if (H.empty() || n <= 0) {
        ++g_skipped;
        platform::log_info("SKIP  %s: fixture unreadable", g_current.c_str());
        return;
    }

    int would_gate = 0;      // rows whose label WOULD veto if it were allowed
    int refused_gate = 0;    // ...and which the authority refuses
    int authorised_gate = 0; // ...and which the SAME label through a rule allows
    for (int32_t i = 0; i < n; ++i) {
        const ClassificationResult r =
            head.classify(H.data() + static_cast<size_t>(i) * 768, "trading.regime", 1);
        if (r.top_k.empty()) continue;
        const std::string& label = r.top_k[0].label;

        const GatingRegime learned = GatingRegime::from_learned(label);
        CHECK(learned.can_gate() == false);              // every row, every label

        const std::string veto = regime_veto(learned, false);
        CHECK(veto.empty());                             // the learned head never blocks

        // How load-bearing is the guard? Count the rows where the head's label
        // is one that a RULE would have vetoed on.
        const bool hostile = (label == "trend_down" || label == "high_vol");
        if (hostile) {
            ++would_gate;
            if (veto.empty()) ++refused_gate;
            // The control: the identical label through the rule factory DOES veto.
            if (!regime_veto(GatingRegime::from_legacy_rule(label), false).empty())
                ++authorised_gate;
        }
    }

    // The head really does emit gating labels — the guard is not a no-op.
    CHECK(would_gate > 0);
    // ...and every single one is refused, while the rule path accepts the same
    // labels. Without the refusal this would be `would_gate` vetoes.
    CHECK(refused_gate == would_gate);
    CHECK(authorised_gate == would_gate);

    platform::log_info("      fitted blob over %d held-out rows: %d gating labels, "
                       "%d refused, %d would have vetoed through a rule",
                       static_cast<int>(n), would_gate, refused_gate, authorised_gate);
}

}  // namespace

int main() {
    platform::log_info("=== omniseed regime authority tests (§46) ===");

    TEST("A: GatingRegime mints — learned/external refuse, engine/rule gate");
    test_type_mints();

    TEST("B: regime_veto — provenance decides, both directions on one label");
    test_regime_veto();

    TEST("C: a real sniper run is authorised (the guard is not a disable)");
    test_sniper_path_authorised();

    TEST("D: the FITTED trading_regime_head.bin cannot reach the gate");
    test_fitted_blob_cannot_gate();

    platform::log_info("---- regime authority tests: %d passed, %d failed, "
                       "%d skipped ----", g_passed, g_failed, g_skipped);
    return g_failed == 0 ? 0 : 1;
}
