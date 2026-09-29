// =============================================================================
//  OmniSeed — src/agent/dream.cpp
//  §47 — the dream joint: one context in, one report out. See dream.h.
// =============================================================================
#include "omniseed/agent/dream.h"

#include "omniseed/agent/agent.h"     // SelfImprovement
#include "omniseed/core/platform.h"
#include "omniseed/soul.h"            // the crystal store's owner

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <set>
#include <string>
#include <vector>

namespace omniseed {

// ---------------------------------------------------------------------------
// The reinforcement predicate, in one place.
//
// A crystal is USED when its recency stamp moved past the previous dream's
// clock. `last_access_token` is written by crystallize() (birth) and by
// retrieve()/reinforce() (use), so "stamp > previous_token" means exactly
// "something recalled this since the last pass". previous_token == 0 means the
// caller has no clock; nothing is then claimed to have been used, because
// guessing would turn an unknown into an assertion.
// ---------------------------------------------------------------------------
bool crystal_used_since(const MemoryCrystal& c, uint64_t previous_token) {
    if (previous_token == 0) return false;
    return c.last_access_token > previous_token;
}

// ---------------------------------------------------------------------------
// DreamReport::to_json — a fixed key order, so the log is diffable and the test
// can assert a schema rather than "some JSON".
// ---------------------------------------------------------------------------
std::string DreamReport::to_json() const {
    auto u = [](uint64_t v) { return std::to_string(v); };
    auto z = [](size_t v) { return std::to_string(v); };

    char ms[64];
    std::snprintf(ms, sizeof(ms), "%.0f", platform::now_ms());

    std::string o;
    o.reserve(1200);
    o += "{\n";
    o += "  \"version\": 1,\n";
    o += "  \"generated_ms\": " + std::string(ms) + ",\n";
    o += "  \"previous_token\": " + u(previous_token) + ",\n";
    o += "  \"now_token\": " + u(now_token) + ",\n";

    o += "  \"traces\": {";
    o += "\"before\": " + z(traces_before);
    o += ", \"after\": " + z(traces_after);
    o += ", \"pruned\": " + z(traces_pruned);
    o += ", \"pruned_by_journal\": " + z(traces_pruned_by_journal);
    o += ", \"reinforced_by_journal\": " + z(traces_reinforced_by_journal);
    o += ", \"decayed\": " + z(traces_decayed);
    o += ", \"faded\": " + z(traces_faded);
    o += ", \"dropped_unsuccessful\": " + z(traces_dropped_unsuccessful);
    o += "},\n";

    o += "  \"crystals\": {";
    o += "\"before\": " + z(crystals_before);
    o += ", \"after\": " + z(crystals_after);
    o += ", \"reinforced\": " + z(crystals_reinforced);
    o += ", \"decayed\": " + z(crystals_decayed);
    o += "},\n";

    o += "  \"journal\": {";
    o += "\"present\": " + std::string(journal_present ? "true" : "false");
    o += ", \"predictions\": " + z(journal_predictions);
    o += ", \"unresolved\": " + z(journal_unresolved);
    o += ", \"expired\": " + z(journal_expired);
    o += ", \"rejected\": " + z(journal_rejected);
    o += ", \"verdicts\": " + z(journal_verdicts);
    o += ", \"hits\": " + z(journal_hits);
    o += ", \"confirmed_keys\": " + z(journal_confirmed_keys);
    o += ", \"corrected_keys\": " + z(journal_corrected_keys);
    o += "},\n";

    o += "  \"stream\": {";
    o += "\"source\": \"" + stream_source + "\"";
    o += ", \"events\": " + z(stream_events);
    o += ", \"valid\": " + z(stream_valid);
    o += ", \"committed\": " + z(stream_committed);
    o += ", \"flips\": " + z(stream_flips);
    o += "},\n";

    o += "  \"soul\": {";
    o += "\"present\": " + std::string(soul_present ? "true" : "false");
    o += ", \"crystals\": " + z(soul_crystals);
    o += ", \"dropped\": " + z(soul_dropped);
    o += ", \"clock\": " + u(soul_clock);
    o += "}\n";

    o += "}\n";
    return o;
}

// ---------------------------------------------------------------------------
// write_dream_log — create the parent directory, then write the report.
// ---------------------------------------------------------------------------
bool write_dream_log(const DreamReport& r, const std::string& path) {
    namespace fs = std::filesystem;
    const fs::path p(path);
    if (p.has_parent_path()) {
        std::error_code ec;
        fs::create_directories(p.parent_path(), ec);   // best effort; open() decides
    }

    FILE* f = platform::open_file_c(path.c_str(), "wb");
    if (f == nullptr) return false;
    const std::string j = r.to_json();
    const size_t wrote = std::fwrite(j.data(), 1, j.size(), f);
    std::fclose(f);
    return wrote == j.size();
}

// ---------------------------------------------------------------------------
// The pass.
// ---------------------------------------------------------------------------
DreamReport SelfImprovement::dream(const DreamContext& ctx) {
    DreamReport r;
    r.previous_token = ctx.previous_token;
    r.now_token      = ctx.now_token;
    r.traces_before  = traces_.size();

    const auto now_s = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());

    // ---- 1. journal evidence (§38) ----------------------------------------
    // Built BEFORE the trace walk, because it decides what the walk does. The
    // journal is evidence about the world; `success_count` is only evidence
    // about our own bookkeeping, and when the two disagree the world wins.
    std::set<std::string> confirmed, corrected;
    if (ctx.journal != nullptr) {
        r.journal_present = true;
        const JournalStats st = ctx.journal->stats();
        r.journal_predictions = static_cast<size_t>(st.predictions);
        r.journal_unresolved  = static_cast<size_t>(st.unresolved);
        r.journal_expired     = static_cast<size_t>(st.expired);
        r.journal_rejected    = static_cast<size_t>(st.rejected);
        r.journal_verdicts    = static_cast<size_t>(st.verdicts);
        r.journal_hits        = static_cast<size_t>(st.hits);

        for (const JournalEntry& e : ctx.journal->entries()) {
            if (!ctx.task_key.empty() && e.task_key != ctx.task_key) continue;
            if (e.outcome == OutcomeKind::Confirmed ||
                e.outcome == OutcomeKind::Realised) {
                confirmed.insert(e.task_key);
            } else if (e.outcome == OutcomeKind::Corrected) {
                // Only Corrected. Rejected means "this was not a valid decision
                // to make" — it is a verdict about the QUESTION, not a failure
                // of the path, and pruning a working trace for it would be the
                // journal being read as something it is not.
                corrected.insert(e.task_key);
            }
        }
        r.journal_confirmed_keys = confirmed.size();
        r.journal_corrected_keys = corrected.size();
    }

    // ---- 2. streaming verdicts (§37) --------------------------------------
    if (ctx.events != nullptr) {
        r.stream_source = "events";
        r.stream_events = ctx.events->size();
        for (const StreamEvent& e : *ctx.events) {
            if (e.valid)     ++r.stream_valid;
            if (e.committed) ++r.stream_committed;
            if (e.changed)   ++r.stream_flips;
        }
    } else if (ctx.journal != nullptr) {
        // The §37 verdict is PERSISTED in the journal's filter columns, so this
        // is the same fact from the same source, not a substitute for it. Every
        // journal row is a VALID event by construction — record_event() refuses
        // to record one the head did not observe — so `valid` == `events` here,
        // and the report says which source it read.
        r.stream_source = "journal";
        for (const JournalEntry& e : ctx.journal->entries()) {
            if (!ctx.task_key.empty() && e.task_key != ctx.task_key) continue;
            ++r.stream_events;
            if (e.committed) ++r.stream_committed;
            if (e.changed)   ++r.stream_flips;
        }
        r.stream_valid = r.stream_events;
    }

    // ---- 3. TRACE PASS (the pre-§47 rules, plus the journal) ---------------
    std::vector<TaskTrace> kept;
    kept.reserve(traces_.size());

    for (TaskTrace& t : traces_) {
        if (!corrected.empty() && corrected.count(t.task_key) > 0) {
            ++r.traces_pruned_by_journal;
            continue;
        }
        if (!confirmed.empty() && confirmed.count(t.task_key) > 0) {
            // Refresh, do not increment. The journal is a standing record, so a
            // pass that bumped success_count would inflate it on every dream —
            // reinforcement here means "stop ageing", which is idempotent.
            t.last_used = now_s;
            ++r.traces_reinforced_by_journal;
        }

        const uint64_t age_s = now_s > t.last_used ? now_s - t.last_used : 0;
        const uint64_t days  = age_s / 86400;
        const uint32_t total = t.success_count + t.fail_count;
        const double   rate  = total > 0
            ? static_cast<double>(t.success_count) / total : 0.0;

        if (t.success_count == 0 && t.fail_count > 2) { ++r.traces_pruned; continue; }
        if (days > 30 && rate < 0.5)                   { ++r.traces_pruned; continue; }

        if (days > 14 && t.success_count > 0) {
            t.success_count = static_cast<uint32_t>(t.success_count * 0.5);
            if (t.success_count == 0) { ++r.traces_faded; continue; }
            ++r.traces_decayed;
        }

        if (t.success_count > 0) {
            kept.push_back(t);
        } else {
            // The pre-§47 pass dropped these without a word. The behaviour is
            // unchanged; only the accounting is new.
            ++r.traces_dropped_unsuccessful;
        }
    }
    traces_.swap(kept);
    r.traces_after = traces_.size();

    // ---- 4. CRYSTAL PASS (§31) --------------------------------------------
    // `soul` wins when both are set: it is the store the running system owns.
    const MemoryCrystals* store = nullptr;
    if (ctx.soul != nullptr) {
        r.soul_present = true;
        store = &ctx.soul->crystals();
    } else if (ctx.crystals != nullptr) {
        store = ctx.crystals;
    }

    if (store != nullptr) {
        r.crystals_before = store->size();

        // Reinforce first, then decay: a crystal used since the last dream must
        // get its reprieve BEFORE the sweep, or a memory that was recalled
        // yesterday is deleted today for having been old yesterday.
        const std::vector<uint64_t> ids = store->ids();
        for (const uint64_t id : ids) {
            const MemoryCrystal* c = store->find(id);
            if (c == nullptr) continue;
            if (!crystal_used_since(*c, ctx.previous_token)) continue;
            const bool ok = ctx.soul != nullptr
                ? ctx.soul->reinforce_memory(id)
                : ctx.crystals->reinforce(id, ctx.now_token);
            if (ok) ++r.crystals_reinforced;
        }

        r.crystals_decayed = ctx.soul != nullptr
            ? ctx.soul->decay_memories(ctx.now_token)
            : ctx.crystals->decay(ctx.now_token);

        r.crystals_after = store->size();
    }

    // The soul's own counters, AFTER the pass. `soul_dropped` is the soul's
    // CUMULATIVE session counter, not this pass's drop count — that is
    // `crystals_decayed` — and the log names them differently on purpose.
    if (ctx.soul != nullptr) {
        r.soul_crystals = ctx.soul->memory_size();
        r.soul_dropped  = ctx.soul->memory_dropped();
        r.soul_clock    = ctx.soul->memory_clock();
    }

    return r;
}

}  // namespace omniseed
