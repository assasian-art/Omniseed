// =============================================================================
//  OmniSeed — trading/regime_authority.h
//  A regime label that CARRIES ITS PROVENANCE, so a learned head's output can
//  never reach a strategy gate.
//
//  WHY THIS EXISTS (§45 → §46). `docs/EDGE_RESEARCH.md` §45 demoted the learned
//  `trading.regime` head: on its own 369-row holdout it scores 0.7046 against a
//  majority-class rate of 0.8699 — **0.810x a constant predictor**, i.e. worse
//  than always saying "range". The rule-based `RegimeEngine` is the only gater.
//
//  A demotion that lives only in prose is not a demotion. Nothing in the type
//  system stopped a later caller from doing
//
//      RegimeState s;  s.label = head.argmax_name();   // a LEARNED label
//      sniper.evaluate(prepared_with(s), i, ...);      // ...and it gates
//
//  This header closes that hole at the type level. A `GatingRegime` may be
//  minted in exactly two ways that can gate — `from_engine()` (the multi-axis
//  `RegimeEngine`/`RegimeTracker`) and `from_legacy_rule()` (the deterministic
//  SMA-slope rule) — and both take a value the ENGINE produced. Every other
//  mint (`from_learned`, `from_external`) produces a token whose `can_gate()` is
//  **false**, and every gate site must consult `can_gate()` / `authorize()`
//  before it acts.
//
//  The refusals are expressible rather than compile errors on purpose: a compile
//  error would hide the case, whereas a token that reports *why* it refuses can
//  be tested (tests/test_regime_authority.cpp), journalled, and audited.
//
//  `RegimeState::engine_minted` is what makes `from_engine()` trustworthy: the
//  engine sets it in `RegimeEngine::detect()`, and `from_engine()` refuses a
//  hand-built state that never went through the engine. Without that flag the
//  authority would be forgeable by simply constructing a struct.
// =============================================================================
#pragma once

#include <string>

#include "omniseed/trading/regime_engine.h"

namespace omniseed {
namespace trading {

// Where a regime label came from. The first two may gate; the last two never.
enum class RegimeSource {
    Unset = 0,     // default-constructed; nothing minted it
    RuleEngine,    // the multi-axis RegimeEngine / RegimeTracker  -> MAY GATE
    LegacyRule,    // the deterministic SMA-slope rule             -> MAY GATE
    LearnedHead,   // a fitted .bin readout (trading.regime)       -> NEVER gates
    External,      // a CSV / config label, or an unminted struct  -> NEVER gates
};

const char* regime_source_name(RegimeSource s);

// ---------------------------------------------------------------------------
// GatingRegime — the only type a strategy gate may consult.
// ---------------------------------------------------------------------------
class GatingRegime {
public:
    GatingRegime() = default;   // Unset; cannot gate

    // ---- the ONLY mints that may gate -------------------------------------
    // From the rule engine's own output. Returns a NON-gating token when `s`
    // was not minted by the engine (`s.engine_minted == false`), so a hand-built
    // RegimeState cannot be laundered into a gate.
    static GatingRegime from_engine(const RegimeState& s);

    // From the deterministic SMA-slope rule (`regime_mode == "legacy"`). This is
    // a rule computed from bars, not a learned readout, so it may gate — but it
    // is a SEPARATE factory so the distinction stays visible in the source.
    static GatingRegime from_legacy_rule(const std::string& label);

    // ---- the mints that NEVER gate ----------------------------------------
    // A fitted head's label. Exists so the refusal is expressible and testable.
    static GatingRegime from_learned(const std::string& label);
    // A label from a CSV, a config, or anywhere else outside the engine.
    static GatingRegime from_external(const std::string& label);

    RegimeSource source() const { return source_; }
    const std::string& label() const { return label_; }

    // THE predicate. True iff this label may influence a strategy gate.
    bool can_gate() const {
        return source_ == RegimeSource::RuleEngine ||
               source_ == RegimeSource::LegacyRule;
    }

    // The runtime assertion a gate site calls. Identical to can_gate() today;
    // named separately so a gate site reads as an assertion and so the rule can
    // grow (e.g. a confidence floor) without touching every caller.
    bool authorize() const { return can_gate(); }

    // Why this token may not gate. Empty when it may. Meant for the journal.
    const char* refusal_reason() const;

    std::string detail() const;   // "regime=<label> source=<name> gate=<yes|no>"

private:
    static GatingRegime mint(RegimeSource src, const std::string& label);
    RegimeSource source_ = RegimeSource::Unset;
    std::string  label_ = "range";
};

// ---------------------------------------------------------------------------
// The regime veto decision, extracted so it lives in exactly ONE place and so
// it is testable without building a whole SniperVerdict.
//
// Returns "" when no regime veto applies; otherwise the veto reason. It is the
// single choke point through which a regime label can block a trade, and it
// takes a `GatingRegime` — not a string — so a learned label has no way in.
// ---------------------------------------------------------------------------
std::string regime_veto(const GatingRegime& gate, bool mean_reversion_confirmed);

} // namespace trading
} // namespace omniseed
