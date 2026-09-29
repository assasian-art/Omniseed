// =============================================================================
//  OmniSeed — include/omniseed/agent/dream.h
//
//  §47 — Dream-State Consolidation, as a JOINT.
//
//  WHAT WAS MISSING. `SelfImprovement::dream()` existed, and it worked: it
//  pruned failure-only traces, decayed long-unused ones and kept the rest. What
//  it could not do is see anything outside its own vector. Meanwhile three
//  subsystems had grown the pieces a dream pass is supposed to consume, and
//  each one said so in a comment:
//
//    * MemoryCrystals::decay()/reinforce() — "Call periodically (e.g. from
//      dream())", "For tests and for the dream pass, which needs to reason
//      about individual memories."
//    * Soul::decay_memories()/reinforce_memory() — "Belongs in a dream/idle
//      pass, not on the hot path."
//    * DecisionJournal — recorded the §37 filter verdicts and the realised
//      outcomes, and NOTHING consumed it (a standing open finding).
//
//  So the gap was never a missing feature; it was the SEAM. This header is the
//  seam: one context in, one report out.
//
//  WHAT A DREAM PASS NOW DOES
//
//    1. TRACE PASS (the original rules, unchanged when the context is empty):
//       prune failure-only, prune stale-and-unsuccessful, halve the success
//       weight of anything untouched for > 14 days. NEW: when a journal is
//       supplied, a trace whose key the journal has CORRECTED is pruned
//       whatever its success count, and a trace whose key was CONFIRMED or
//       REALISED has its `last_used` refreshed so it stops ageing. The journal
//       is evidence about the world; `success_count` is only evidence about
//       our own bookkeeping.
//
//    2. CRYSTAL PASS: a crystal whose `last_access_token` advanced past
//       `previous_token` — i.e. one that was actually RECALLED since the last
//       dream — is reinforced. Then decay() runs over the rest. This is the
//       Ebbinghaus contract the memory layer already documents: retrieval is
//       the evidence of usefulness, so a used memory must fade more slowly than
//       an ignored one. `previous_token` is the clock at the previous dream;
//       passing 0 means "unknown", and then nothing is reinforced and only
//       decay applies.
//
//    3. REPORT: the whole pass is summarised as JSON and written to
//       `state/dream_log.json` (see write_dream_log).
//
//  HONESTY NOTES
//
//    * The §37 streaming verdicts are read from the journal's filter columns
//      (`committed` / `changed` / `held`) when no explicit event vector is
//      supplied, because that is where §38 PERSISTS them. The report names
//      which source it used (`stream.source`) rather than implying it always
//      saw live events.
//    * A dream pass never invents activity. With an empty context the report is
//      all zeros and the log says so; it does not fabricate a plausible day.
//    * Nothing here is fitted, sampled or random. Every COUNT is a
//      deterministic function of the inputs, which is what makes the report
//      testable. The single non-deterministic field is `generated_ms`, which is
//      a timestamp and is never asserted on.
// =============================================================================
#pragma once

#include "omniseed/feedback_hook.h"
#include "omniseed/memory/memory.h"
#include "omniseed/streaming_decision.h"

#include <cstdint>
#include <string>
#include <vector>

namespace omniseed {

// The crystal store's owner in the real system. Forward-declared on purpose:
// dream.h must not pull in the whole soul (tokenizer, persona, emotional
// runtime) just to name it, and dream.cpp includes soul.h where it is needed.
class Soul;

// ---------------------------------------------------------------------------
// DreamContext — everything a dream pass may read. Every field is optional.
//
// Exactly one of `soul` / `crystals` should be set; `soul` wins when both are
// (it is the store the running system actually owns). A context with neither is
// a valid trace-only dream, which is exactly the pre-§47 behaviour.
// ---------------------------------------------------------------------------
struct DreamContext {
    Soul*                           soul     = nullptr;  // §30/§31 — preferred
    MemoryCrystals*                 crystals = nullptr;  // standalone store (tests)
    const DecisionJournal*          journal  = nullptr;  // §38 — the feedback journal
    const std::vector<StreamEvent>* events   = nullptr;  // §37 — live verdicts, if any

    // The clock at the PREVIOUS dream, in the same units as
    // MemoryCrystal::last_access_token. 0 means "unknown" (first dream, or a
    // caller with no clock) and disables reinforcement.
    uint64_t previous_token = 0;
    // Now, in the crystal clock. Must be >= every last_access_token or decay
    // clamps the age to zero (it does, deliberately — see memory_crystals.cpp).
    uint64_t now_token = 0;
    // Scopes the journal when the caller only cares about one task family.
    // Empty means "every key".
    std::string task_key;
};

// ---------------------------------------------------------------------------
// DreamReport — what the pass did. Every field is a count, never an estimate.
// ---------------------------------------------------------------------------
struct DreamReport {
    // --- trace cache (SelfImprovement) --------------------------------------
    size_t traces_before = 0;
    size_t traces_after  = 0;
    size_t traces_pruned = 0;            // failure-only / stale-and-unsuccessful
    size_t traces_decayed = 0;           // success weight halved (> 14 days), survived
    // Halved from 1 to 0 and therefore dropped. Counted separately from
    // `traces_pruned` because the HALVING is why it went, and separately from
    // `traces_decayed` because that one survived. Together with
    // `traces_dropped_unsuccessful` these close the accounting exactly:
    //   before == after + pruned + pruned_by_journal + faded + dropped_unsuccessful
    size_t traces_faded = 0;
    // Had no successes and matched no prune rule. The pre-§47 dream() dropped
    // these silently — the final `if (success_count > 0)` keep-test is what
    // removed them, and nothing said so. Named here so the accounting closes.
    size_t traces_dropped_unsuccessful = 0;
    size_t traces_pruned_by_journal = 0; // the journal corrected this key
    size_t traces_reinforced_by_journal = 0;  // the journal confirmed this key

    // --- crystals (§31) ------------------------------------------------------
    size_t crystals_before = 0;
    size_t crystals_after  = 0;
    size_t crystals_reinforced = 0;      // recalled since `previous_token`
    size_t crystals_decayed = 0;         // dropped by decay()

    // --- feedback journal (§38) ---------------------------------------------
    bool   journal_present = false;
    size_t journal_predictions = 0;
    size_t journal_unresolved = 0;
    size_t journal_expired = 0;
    size_t journal_rejected = 0;
    size_t journal_verdicts = 0;
    size_t journal_hits = 0;
    size_t journal_confirmed_keys = 0;   // distinct keys reinforced by evidence
    size_t journal_corrected_keys = 0;   // distinct keys pruned by evidence

    // --- streaming (§37) -----------------------------------------------------
    // "events" = a live event vector was supplied; "journal" = derived from the
    // journal's persisted filter columns; "none" = no stream information at all.
    std::string stream_source = "none";
    size_t stream_events = 0;
    size_t stream_valid = 0;
    size_t stream_committed = 0;
    size_t stream_flips = 0;

    // --- soul (§30) ----------------------------------------------------------
    bool     soul_present = false;
    size_t   soul_crystals = 0;
    size_t   soul_dropped = 0;
    uint64_t soul_clock = 0;

    // --- the pass itself -----------------------------------------------------
    uint64_t previous_token = 0;
    uint64_t now_token = 0;

    // Stable key order, fixed by hand so the JSON is diffable and the test can
    // assert a schema rather than "some JSON".
    std::string to_json() const;
};

// A crystal counts as USED when its recency stamp moved past the previous
// dream's clock. Stated as a function so the test can name the rule instead of
// re-deriving it, and so there is exactly one place it can be wrong.
bool crystal_used_since(const MemoryCrystal& c, uint64_t previous_token);

// Writes `r` as JSON to `path`, creating the parent directory when needed.
// Returns false only when the file cannot be opened — a dream that ran but
// could not be logged is reported, not swallowed.
bool write_dream_log(const DreamReport& r, const std::string& path);

}  // namespace omniseed
