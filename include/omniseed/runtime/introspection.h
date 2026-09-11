// =============================================================================
//  OmniSeed — runtime/introspection.h
//  Self-awareness + metacognition layer (Phase-16 "Omega Pass", Phase 2):
//
//    * SelfModel            — identity, mission, honest capability map
//    * GoalTracker          — active goals with progress + priority
//    * PurposeReflection    — periodic "why am I doing this" check
//    * ActionRationale      — decision log ("I chose X because Y")
//    * ConfidenceCalibration— stated confidence vs realized outcomes
//    * Metacognition        — knowledge gaps + strategy self-evaluation
//
//  Claude-Mythos-style self-awareness, honestly scoped: this is structured
//  introspection over real runtime state — not a claim of sentience. The
//  agent can answer "who are you / what are you doing / why", audit its own
//  calibration, and name what it does not know.
// =============================================================================
#pragma once

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace omniseed {

// ===========================================================================
// SelfModel — "who am I"
// ===========================================================================
struct SelfModel {
    std::string name        = "OmniSeed";
    std::string version     = "0.3-omega";
    std::string mission     = "Be a trustworthy sub-300MB trading-focused AI "
                              "agent: minimize losses through risk discipline, "
                              "surface every assumption, and never fake "
                              "certainty.";
    std::string identity    = "I am OmniSeed, a local-first multi-agent "
                              "trading kernel. I reason with deterministic "
                              "market analytics, a real 0.1B-class language "
                              "core, and honest uncertainty. My edge is "
                              "discipline, not prediction.";
    // Capability honesty: the introspect command prints these AS STATED.
    struct Capability {
        const char* area;
        const char* state;      // "real", "partial", "fallback", "absent"
    };
    static const Capability* capabilities();
    static const size_t      capability_count();
};

// ===========================================================================
// GoalTracker — "what am I doing"
// ===========================================================================
struct Goal {
    std::string text;               // "maximize risk-adjusted return"
    std::string metric;             // "sharpe", "max_drawdown_pct", ...
    double  target = 0.0;
    double  current = 0.0;
    double  priority = 0.5;         // 0..1
    bool    active = true;
};

class GoalTracker {
public:
    void add(const Goal& g);
    bool update(const std::string& metric, double current);   // by metric key
    bool deactivate(const std::string& text);
    const std::vector<Goal>& goals() const { return goals_; }
    std::vector<Goal> active_goals() const;                   // priority-sorted

    // Persistence ('OKGD' binary) — goals survive restarts.
    bool save(const std::string& path) const;
    bool load(const std::string& path);

private:
    std::vector<Goal> goals_;
};

// ===========================================================================
// PurposeReflection — "why am I doing this"
// ===========================================================================
class PurposeReflection {
public:
    struct Reflection {
        uint64_t at_unix = 0;
        std::string question;       // "why did you buy AAPL?"
        std::string answer;         // generated from goals + rationale log
    };

    // Periodic self-query: ties recent rationales back to active goals.
    // Called by the trading watchdog every N cycles and by `introspect`.
    static Reflection reflect(const class ActionRationale& log,
                              const GoalTracker& goals);

    // The mission statement, verbatim (the "purpose" anchor).
    static const char* mission();
};

// ===========================================================================
// ActionRationale — "I chose X because Y" (every decision, logged)
// ===========================================================================
struct RationaleEntry {
    uint64_t at_unix = 0;
    std::string actor;              // "analyst", "risk", "execution", ...
    std::string action;             // "buy 50 SYN @ 100.0"
    std::string because;            // "consensus 0.65: uptrend+MACD; news +0.4"
    double  confidence = 0.0;       // 0..1 stated at decision time
    double  outcome = 0.0;          // realized return when known (0 = pending)
    bool    outcome_known = false;
};

class ActionRationale {
public:
    struct Config { size_t max_entries = 256; };
    explicit ActionRationale(const Config& cfg = {}) : cfg_(cfg) {}

    void log(const std::string& actor, const std::string& action,
             const std::string& because, double confidence);
    // Attaches a realized outcome to the most recent pending entry.
    bool resolve_latest(double outcome);
    // Confidence-accuracy calibration: mean stated confidence vs win rate
    // over resolved entries (empty -> not_calibrated).
    struct Calibration {
        bool   not_calibrated = true;
        double mean_confidence = 0.0;   // what I said
        double realized_rate   = 0.0;   // what happened
        double gap             = 0.0;   // said - happened (positive = over)
        size_t samples = 0;
    };
    Calibration calibration() const;

    const std::deque<RationaleEntry>& entries() const { return entries_; }
    void clear() { entries_.clear(); }

    bool save(const std::string& path) const;
    bool load(const std::string& path);

private:
    Config cfg_;
    std::deque<RationaleEntry> entries_;
};

// ===========================================================================
// Metacognition — "what do I not know" + strategy self-evaluation
// ===========================================================================
struct KnowledgeGap {
    std::string topic;              // "fundamentals for AAPL"
    std::string why_it_matters;
    std::string suggested_action;   // "pull SEC filings via research agent"
};

class Metacognition {
public:
    struct State {
        std::vector<KnowledgeGap> gaps;
        std::string strategy_assessment;    // "sharpe 1.2 over 252 cycles"
        double  confidence_in_strategy = 0.0;  // 0..1
        size_t  calibrated_decisions = 0;
        double  calibration_gap = 0.0;
        bool    overconfident = false;
    };

    // Builds the metacognitive state from real runtime inputs.
    static State evaluate(const GoalTracker& goals,
                          const ActionRationale& log,
                          double realized_sharpe,
                          size_t observations);

    // Knowledge gaps derived from what the runtime honestly lacks.
    static std::vector<KnowledgeGap> default_gaps();
};

} // namespace omniseed
