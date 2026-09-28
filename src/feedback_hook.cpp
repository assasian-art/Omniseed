// =============================================================================
//  OmniSeed — feedback_hook.cpp
//
//  MILESTONE 10 — Phase 2.3. See feedback_hook.h for the contract; this file is
//  the record, the scoring and the persistence.
//
//  The load-bearing decisions, all of them about honesty rather than code:
//
//    * An unresolved row is NOT a wrong row. Four outcome kinds are excluded
//      from every accuracy number, for four different reasons.
//    * `hit_rate()` is -1.0, never 0.0, when nothing has been resolved. Same
//      convention as calibration_error() < 0 and distance() == -1.0.
//    * A prediction is scored against ONE outcome. Re-resolving a row is
//      refused, because a record that can be scored twice will eventually agree
//      with itself.
//    * `score` is recorded, never predicted, and no objective function is
//      computed from it. The mandate's R2 holds by construction.
// =============================================================================
#include "omniseed/feedback_hook.h"

#include "omniseed/core/platform.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace omniseed {

namespace {

// The on-disk magics, spelled out as BYTES on purpose — see the lesson recorded
// in self_improvement.cpp. One constant per format, shared by both sides.
constexpr char kJournalMagic[4] = {'O', 'M', 'N', 'J'};
constexpr char kHookMagic[4]    = {'O', 'M', 'N', 'H'};

constexpr uint32_t kJournalVersion = 1;
constexpr uint32_t kHookVersion    = 1;

// A task key is written into a TSV and into a JSON string, so a tab, a newline
// or a quote in it would corrupt the output. Sanitised at WRITE time rather
// than trusted at read time.
std::string sanitise_key(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) out.push_back((c == '\t' || c == '\n' || c == '\r') ? ' ' : c);
    return out;
}

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out.push_back(c);
        }
    }
    return out;
}

// %.6g-style float formatting that emits JSON `null` for a non-finite value.
// A NaN written as "nan" is not JSON and would make a consumer's parse fail at
// a point far from the bug; `null` is the same convention the port dumps use.
std::string num(double v, int precision = 6) {
    if (!std::isfinite(v)) return "null";
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*g", precision, v);
    return buf;
}

bool valid_action(int32_t a) {
    return a >= 0 && a < static_cast<int32_t>(DecisionAction::COUNT);
}

} // namespace

// ===========================================================================
// OutcomeKind
// ===========================================================================
const char* outcome_kind_name(OutcomeKind k) {
    switch (k) {
        case OutcomeKind::Unknown:   return "unknown";
        case OutcomeKind::Realised:  return "realised";
        case OutcomeKind::Confirmed: return "confirmed";
        case OutcomeKind::Corrected: return "corrected";
        case OutcomeKind::Rejected:  return "rejected";
        case OutcomeKind::Expired:   return "expired";
        case OutcomeKind::COUNT:     break;
    }
    return "unknown";
}

bool outcome_kind_from_name(const std::string& name, OutcomeKind& out) {
    for (int32_t i = 0; i < static_cast<int32_t>(OutcomeKind::COUNT); ++i) {
        const OutcomeKind k = static_cast<OutcomeKind>(i);
        if (name == outcome_kind_name(k)) { out = k; return true; }
    }
    return false;
}

bool outcome_kind_has_truth(OutcomeKind k) {
    switch (k) {
        case OutcomeKind::Realised:
        case OutcomeKind::Confirmed:
        case OutcomeKind::Corrected: return true;
        // `Rejected` is a real verdict but NOT an accuracy datum: a decision
        // that should not have been made is a different failure from a decision
        // that was wrong, and averaging the two would flatter the system.
        // `Expired` has no answer at all.
        case OutcomeKind::Unknown:
        case OutcomeKind::Rejected:
        case OutcomeKind::Expired:
        case OutcomeKind::COUNT:     return false;
    }
    return false;
}

// ===========================================================================
// JournalEntry
// ===========================================================================
bool JournalEntry::head_hit() const {
    switch (outcome) {
        case OutcomeKind::Realised:  return predicted == realised;
        case OutcomeKind::Confirmed: return true;
        case OutcomeKind::Corrected: return false;
        default:                     return false;
    }
}

bool JournalEntry::filter_scored() const {
    if (!has_verdict() || !committed) return false;
    // A user verdict describes what we SAID. If the filter was holding
    // something other than what the head predicted, that verdict says nothing
    // about the filter's position and this row is not a filter row.
    if (outcome != OutcomeKind::Realised && held != predicted) return false;
    return true;
}

bool JournalEntry::filter_hit() const {
    if (!filter_scored()) return false;
    if (outcome == OutcomeKind::Realised) return held == realised;
    return outcome == OutcomeKind::Confirmed;
}

std::string JournalEntry::to_json() const {
    std::string o = "{";
    o += "\"id\":"       + std::to_string(id);
    o += ",\"step\":"    + std::to_string(step);
    o += ",\"task\":\""  + json_escape(task_key) + "\"";
    o += ",\"predicted\":\"" + std::string(decision_action_name(predicted)) + "\"";
    o += ",\"confidence\":"  + num(confidence);
    o += ",\"margin\":"      + num(margin);
    o += ",\"held\":\""      + std::string(decision_action_name(held)) + "\"";
    o += ",\"committed\":"   + std::string(committed ? "true" : "false");
    o += ",\"changed\":"     + std::string(changed ? "true" : "false");
    o += ",\"strength\":"    + num(strength);
    o += ",\"outcome\":\""   + std::string(outcome_kind_name(outcome)) + "\"";
    o += ",\"realised\":\""  + std::string(decision_action_name(realised)) + "\"";
    o += ",\"score\":"       + num(score);
    o += ",\"horizon\":"     + std::to_string(horizon);
    // `at_ms` is a wall clock, not a measurement; 3 decimals is plenty and
    // avoids printing 17 digits of noise.
    o += ",\"at_ms\":"       + num(at_ms, 3);
    o += "}";
    return o;
}

// ===========================================================================
// JournalStats
// ===========================================================================
double JournalStats::hit_rate() const {
    // NOT MEASURED is -1.0, never 0.0. "I have never been checked" and "I have
    // always been wrong" are different statements.
    if (verdicts <= 0) return -1.0;
    return static_cast<double>(hits) / static_cast<double>(verdicts);
}

double JournalStats::filter_hit_rate() const {
    if (filter_verdicts <= 0) return -1.0;
    return static_cast<double>(filter_hits) / static_cast<double>(filter_verdicts);
}

double JournalStats::commitment_rate() const {
    if (predictions <= 0) return -1.0;
    return static_cast<double>(committed_rows) / static_cast<double>(predictions);
}

std::string JournalStats::to_json() const {
    std::string o = "{";
    o += "\"predictions\":"  + std::to_string(predictions);
    o += ",\"unresolved\":"  + std::to_string(unresolved);
    o += ",\"expired\":"     + std::to_string(expired);
    o += ",\"rejected\":"    + std::to_string(rejected);
    o += ",\"verdicts\":"    + std::to_string(verdicts);
    o += ",\"hits\":"        + std::to_string(hits);
    o += ",\"hit_rate\":"    + num(hit_rate());
    o += ",\"committed_rows\":" + std::to_string(committed_rows);
    o += ",\"commitment_rate\":" + num(commitment_rate());
    o += ",\"filter_verdicts\":" + std::to_string(filter_verdicts);
    o += ",\"filter_hits\":"     + std::to_string(filter_hits);
    o += ",\"filter_hit_rate\":" + num(filter_hit_rate());
    o += ",\"mean_conf_hit\":"   + num(mean_conf_hit);
    o += ",\"mean_conf_miss\":"  + num(mean_conf_miss);
    o += ",\"confidence_gap\":"  + num(confidence_gap);
    o += ",\"ece\":"             + num(ece);
    o += ",\"brier\":"           + num(brier);
    o += ",\"confusion\":[";
    for (size_t i = 0; i < confusion.size(); ++i) {
        if (i) o += ",";
        o += "[";
        for (size_t j = 0; j < confusion[i].size(); ++j) {
            if (j) o += ",";
            o += std::to_string(confusion[i][j]);
        }
        o += "]";
    }
    o += "]}";
    return o;
}

// ===========================================================================
// DecisionJournal — recording
// ===========================================================================
int64_t DecisionJournal::record(JournalEntry e) {
    if (e.id == 0) e.id = next_id_++;
    else           next_id_ = std::max(next_id_, e.id + 1);
    if (e.at_ms <= 0.0) e.at_ms = platform::now_ms();
    entries_.push_back(std::move(e));
    prune();
    return entries_.back().id;
}

int64_t DecisionJournal::record_event(const StreamEvent& e,
                                      const std::string& task_key) {
    // An invalid event is a step the head REFUSED to observe. It is not a
    // prediction, and recording it as one would count a dead head's silence as
    // a miss. Nothing is recorded and id 0 says so.
    if (!e.valid) return 0;

    JournalEntry j;
    j.step       = e.step;
    j.task_key   = task_key;
    j.predicted  = e.observed_action;
    j.confidence = e.observed_conf;
    j.margin     = e.observed_margin;
    j.held       = e.action;
    j.committed  = e.committed;
    j.changed    = e.changed;
    j.strength   = static_cast<float>(e.strength);
    return record(std::move(j));
}

size_t DecisionJournal::record_stream(const std::vector<StreamEvent>& events,
                                      const std::string& task_key) {
    size_t n = 0;
    for (const StreamEvent& e : events)
        if (record_event(e, task_key) != 0) ++n;
    return n;
}

int64_t DecisionJournal::record_decision(const DecisionResult& d,
                                         const std::string& task_key) {
    JournalEntry j;
    j.task_key   = task_key;
    j.predicted  = d.action_type;
    j.confidence = d.confidence_score;
    j.margin     = d.margin;
    // No stream, so no filter state. `held` mirrors the prediction and
    // `committed` follows the head's own routing, which is the closest honest
    // analogue: `fast_path` is the head saying "act on this".
    j.held       = d.action_type;
    j.committed  = d.fast_path;
    return record(std::move(j));
}

// ===========================================================================
// DecisionJournal — resolution
// ===========================================================================
bool DecisionJournal::resolve(int64_t id, DecisionAction realised, float score,
                              int32_t horizon) {
    for (JournalEntry& e : entries_) {
        if (e.id != id) continue;
        if (e.resolved()) {
            error_ = "entry " + std::to_string(id) +
                     " is already resolved as " + outcome_kind_name(e.outcome) +
                     "; a prediction is scored against ONE outcome";
            return false;
        }
        e.outcome  = OutcomeKind::Realised;
        e.realised = realised;
        e.score    = score;
        e.horizon  = horizon;
        return true;
    }
    error_ = "no entry with id " + std::to_string(id);
    return false;
}

bool DecisionJournal::resolve_kind(int64_t id, OutcomeKind kind) {
    if (kind == OutcomeKind::Unknown) {
        error_ = "resolve_kind(Unknown) would un-resolve an entry";
        return false;
    }
    for (JournalEntry& e : entries_) {
        if (e.id != id) continue;
        if (e.resolved()) {
            error_ = "entry " + std::to_string(id) +
                     " is already resolved as " + outcome_kind_name(e.outcome);
            return false;
        }
        e.outcome = kind;
        return true;
    }
    error_ = "no entry with id " + std::to_string(id);
    return false;
}

size_t DecisionJournal::resolve_all(const std::string& task_key,
                                    DecisionAction realised, float score,
                                    int32_t horizon) {
    size_t n = 0;
    for (JournalEntry& e : entries_) {
        if (e.task_key != task_key || e.resolved()) continue;
        e.outcome  = OutcomeKind::Realised;
        e.realised = realised;
        e.score    = score;
        e.horizon  = horizon;
        ++n;
    }
    return n;
}

size_t DecisionJournal::resolve_latest(const std::string& task_key,
                                       OutcomeKind kind) {
    if (kind == OutcomeKind::Unknown) return 0;
    // Newest first: the user is replying to the last thing said.
    for (size_t i = entries_.size(); i-- > 0;) {
        JournalEntry& e = entries_[i];
        if (e.task_key != task_key || e.resolved()) continue;
        e.outcome = kind;
        return 1;
    }
    return 0;
}

size_t DecisionJournal::expire_through(int64_t step, const std::string& task_key) {
    size_t n = 0;
    for (JournalEntry& e : entries_) {
        if (e.resolved() || e.step > step) continue;
        if (!task_key.empty() && e.task_key != task_key) continue;
        e.outcome = OutcomeKind::Expired;
        ++n;
    }
    return n;
}

// ===========================================================================
// DecisionJournal — queries
// ===========================================================================
size_t DecisionJournal::unresolved() const {
    size_t n = 0;
    for (const JournalEntry& e : entries_) if (!e.resolved()) ++n;
    return n;
}

const JournalEntry* DecisionJournal::find(int64_t id) const {
    for (const JournalEntry& e : entries_) if (e.id == id) return &e;
    return nullptr;
}

JournalStats DecisionJournal::stats() const {
    JournalStats s;
    s.predictions = static_cast<int64_t>(entries_.size());

    const size_t A = static_cast<size_t>(DecisionAction::COUNT);
    constexpr int kBins = 10;

    int64_t  bin_n[kBins]   = {0};
    int64_t  bin_hit[kBins] = {0};
    double   bin_conf[kBins]= {0.0};
    double   brier_sum = 0.0;
    int64_t  brier_n   = 0;
    double   conf_hit_sum = 0.0, conf_miss_sum = 0.0;
    int64_t  n_hit = 0, n_miss = 0;

    for (const JournalEntry& e : entries_) {
        if (e.committed) ++s.committed_rows;

        if (e.outcome == OutcomeKind::Unknown) { ++s.unresolved; continue; }
        if (e.outcome == OutcomeKind::Expired) { ++s.expired;    continue; }
        if (e.outcome == OutcomeKind::Rejected){ ++s.rejected;   continue; }

        // ---- a row with a determinate answer -------------------------------
        ++s.verdicts;
        const bool hit = e.head_hit();
        if (hit) ++s.hits;

        // Confidence is a probability claim about `predicted`, so it is scored
        // as p(correct) — not as p(realised), which would be a different and
        // much easier question.
        const double c = std::min(1.0, std::max(0.0, static_cast<double>(e.confidence)));
        if (hit) { conf_hit_sum += c;  ++n_hit;  }
        else     { conf_miss_sum += c; ++n_miss; }

        int b = static_cast<int>(c * kBins);
        if (b >= kBins) b = kBins - 1;
        if (b < 0) b = 0;
        ++bin_n[b];
        bin_conf[b] += c;
        if (hit) ++bin_hit[b];

        const double target = hit ? 1.0 : 0.0;
        brier_sum += (c - target) * (c - target);
        ++brier_n;

        if (e.filter_scored()) {
            ++s.filter_verdicts;
            if (e.filter_hit()) ++s.filter_hits;
        }

        if (e.outcome == OutcomeKind::Realised && valid_action(static_cast<int32_t>(e.predicted))
            && valid_action(static_cast<int32_t>(e.realised))) {
            if (s.confusion.empty()) s.confusion.assign(A, std::vector<int64_t>(A, 0));
            ++s.confusion[static_cast<size_t>(e.predicted)]
                         [static_cast<size_t>(e.realised)];
        }
    }

    if (n_hit  > 0) s.mean_conf_hit  = conf_hit_sum  / static_cast<double>(n_hit);
    if (n_miss > 0) s.mean_conf_miss = conf_miss_sum / static_cast<double>(n_miss);
    // The gap needs BOTH sides. With no misses it is genuinely unmeasurable, so
    // it stays -1.0 rather than becoming a flattering positive number.
    if (s.mean_conf_hit >= 0.0 && s.mean_conf_miss >= 0.0)
        s.confidence_gap = s.mean_conf_hit - s.mean_conf_miss;

    if (s.verdicts > 0) {
        double ece = 0.0;
        int non_empty = 0;
        for (int b = 0; b < kBins; ++b) {
            if (bin_n[b] == 0) continue;
            ++non_empty;
            const double acc  = static_cast<double>(bin_hit[b]) / static_cast<double>(bin_n[b]);
            const double conf = bin_conf[b] / static_cast<double>(bin_n[b]);
            ece += (static_cast<double>(bin_n[b]) / static_cast<double>(s.verdicts)) *
                   std::fabs(acc - conf);
        }
        if (non_empty > 0) s.ece = ece;
    }
    if (brier_n > 0) s.brier = brier_sum / static_cast<double>(brier_n);

    return s;
}

std::string DecisionJournal::to_json() const {
    std::string o = "{\"entries\":" + std::to_string(entries_.size());
    o += ",\"next_id\":" + std::to_string(next_id_);
    o += ",\"max_entries\":" + std::to_string(cfg_.max_entries);
    o += ",\"stats\":" + stats().to_json();
    o += ",\"records\":[";
    // Bounded so a 8,192-row journal does not print an 8,192-row string: the
    // last 16 rows are what a log line wants.
    const size_t total = entries_.size();
    const size_t first = total > 16 ? total - 16 : 0;
    for (size_t i = first; i < total; ++i) {
        if (i != first) o += ",";
        o += entries_[i].to_json();
    }
    o += "]}";
    return o;
}

std::string DecisionJournal::to_tsv() const {
    std::string o =
        "id\tstep\ttask_key\tpredicted\tconfidence\tmargin\theld\tcommitted"
        "\toutcome\trealised\tscore\thorizon\n";
    char buf[64];
    for (const JournalEntry& e : entries_) {
        o += std::to_string(e.id);
        o += '\t'; o += std::to_string(e.step);
        o += '\t'; o += sanitise_key(e.task_key);
        o += '\t'; o += decision_action_name(e.predicted);
        std::snprintf(buf, sizeof(buf), "\t%.6g", static_cast<double>(e.confidence));
        o += buf;
        std::snprintf(buf, sizeof(buf), "\t%.6g", static_cast<double>(e.margin));
        o += buf;
        o += '\t'; o += decision_action_name(e.held);
        o += '\t'; o += (e.committed ? "1" : "0");
        o += '\t'; o += outcome_kind_name(e.outcome);
        o += '\t'; o += decision_action_name(e.realised);
        std::snprintf(buf, sizeof(buf), "\t%.6g", static_cast<double>(e.score));
        o += buf;
        o += '\t'; o += std::to_string(e.horizon);
        o += '\n';
    }
    return o;
}

// ===========================================================================
// DecisionJournal — persistence
//
// Little-endian, length-prefixed, fixed layout. Every length is corruption
// controlled, so `need()` is applied before every read and the parse is built
// into a LOCAL vector that is swapped in only on full success.
// ===========================================================================
namespace {

void put_u32(std::string& o, uint32_t v) { o.append(reinterpret_cast<const char*>(&v), 4); }
void put_i32(std::string& o, int32_t  v) { o.append(reinterpret_cast<const char*>(&v), 4); }
void put_i64(std::string& o, int64_t  v) { o.append(reinterpret_cast<const char*>(&v), 8); }
void put_f32(std::string& o, float    v) { o.append(reinterpret_cast<const char*>(&v), 4); }
void put_f64(std::string& o, double   v) { o.append(reinterpret_cast<const char*>(&v), 8); }
void put_u8 (std::string& o, uint8_t  v) { o.push_back(static_cast<char>(v)); }

} // namespace

std::string DecisionJournal::serialise() const {
    std::string o;
    o.append(kJournalMagic, 4);
    put_u32(o, kJournalVersion);
    put_i64(o, next_id_);
    put_i64(o, static_cast<int64_t>(entries_.size()));

    for (const JournalEntry& e : entries_) {
        put_i64(o, e.id);
        put_i64(o, e.step);
        put_u32(o, static_cast<uint32_t>(e.task_key.size()));
        o += e.task_key;
        put_i32(o, static_cast<int32_t>(e.predicted));
        put_f32(o, e.confidence);
        put_f32(o, e.margin);
        put_i32(o, static_cast<int32_t>(e.held));
        put_u8 (o, static_cast<uint8_t>(e.committed ? 1 : 0));
        put_u8 (o, static_cast<uint8_t>(e.changed ? 1 : 0));
        put_f32(o, e.strength);
        put_i32(o, static_cast<int32_t>(e.outcome));
        put_i32(o, static_cast<int32_t>(e.realised));
        put_f32(o, e.score);
        put_i32(o, e.horizon);
        put_f64(o, e.at_ms);
    }
    return o;
}

bool DecisionJournal::deserialise(const uint8_t* p, size_t total) {
    if (p == nullptr) { error_ = "journal: null buffer"; return false; }
    size_t cur = 0;
    auto need = [&](size_t n) -> bool { return cur <= total && n <= total - cur; };

    if (total < 24 || std::memcmp(p, kJournalMagic, 4) != 0) {
        error_ = "journal: bad magic or truncated header";
        return false;
    }
    cur += 4;
    uint32_t version = 0;
    std::memcpy(&version, p + cur, 4); cur += 4;
    if (version != kJournalVersion) {
        error_ = "journal: unsupported version " + std::to_string(version);
        return false;
    }

    int64_t next_id = 0, n = 0;
    std::memcpy(&next_id, p + cur, 8); cur += 8;
    std::memcpy(&n,       p + cur, 8); cur += 8;
    if (next_id < 0 || n < 0) {
        error_ = "journal: negative header count";
        return false;
    }
    // The smallest possible entry is 1+1+4+1+1+1+4+1+4+1+4+4+4+8 = 39 bytes;
    // 32 is a safe floor for the reject test, which only has to stop an absurd
    // count from being used to reserve.
    if (static_cast<size_t>(n) > (total - cur) / 32u + 1u) {
        error_ = "journal: entry count " + std::to_string(n) +
                 " exceeds what the payload can hold";
        return false;
    }

    std::vector<JournalEntry> loaded;
    loaded.reserve(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        JournalEntry e;
        uint32_t klen = 0;
        if (!need(8 + 8 + 4)) { error_ = "journal: truncated entry header"; return false; }
        std::memcpy(&e.id,   p + cur, 8); cur += 8;
        std::memcpy(&e.step, p + cur, 8); cur += 8;
        std::memcpy(&klen,   p + cur, 4); cur += 4;
        if (!need(klen)) { error_ = "journal: truncated task key"; return false; }
        e.task_key.assign(reinterpret_cast<const char*>(p + cur), klen);
        cur += klen;

        if (!need(4 + 4 + 4 + 4 + 1 + 1 + 4 + 4 + 4 + 4 + 4 + 8)) {
            error_ = "journal: truncated entry body";
            return false;
        }
        int32_t predicted = 0, held = 0, outcome = 0, realised = 0;
        uint8_t committed = 0, changed = 0;
        std::memcpy(&predicted, p + cur, 4); cur += 4;
        std::memcpy(&e.confidence, p + cur, 4); cur += 4;
        std::memcpy(&e.margin,     p + cur, 4); cur += 4;
        std::memcpy(&held,         p + cur, 4); cur += 4;
        std::memcpy(&committed,    p + cur, 1); cur += 1;
        std::memcpy(&changed,      p + cur, 1); cur += 1;
        std::memcpy(&e.strength,   p + cur, 4); cur += 4;
        std::memcpy(&outcome,      p + cur, 4); cur += 4;
        std::memcpy(&realised,     p + cur, 4); cur += 4;
        std::memcpy(&e.score,      p + cur, 4); cur += 4;
        std::memcpy(&e.horizon,    p + cur, 4); cur += 4;
        std::memcpy(&e.at_ms,      p + cur, 8); cur += 8;

        // Enum values are read straight off the disk, so they are range-checked
        // before they can index anything. A corrupt action byte would otherwise
        // walk off `confusion` / `decision_action_name`'s switch.
        if (!valid_action(predicted) || !valid_action(held) || !valid_action(realised)) {
            error_ = "journal: action enum out of range";
            return false;
        }
        if (outcome < 0 || outcome >= static_cast<int32_t>(OutcomeKind::COUNT)) {
            error_ = "journal: outcome enum out of range";
            return false;
        }

        e.predicted = static_cast<DecisionAction>(predicted);
        e.held      = static_cast<DecisionAction>(held);
        e.realised  = static_cast<DecisionAction>(realised);
        e.outcome   = static_cast<OutcomeKind>(outcome);
        e.committed = committed != 0;
        e.changed   = changed != 0;
        loaded.push_back(std::move(e));
    }

    entries_.swap(loaded);
    next_id_ = next_id;
    error_.clear();
    return true;
}

bool DecisionJournal::save(const std::string& path) const {
    const std::string bytes = serialise();
    FILE* f = platform::open_file_c(path.c_str(), "wb");
    if (!f) return false;
    const size_t wrote = std::fwrite(bytes.data(), 1, bytes.size(), f);
    std::fclose(f);
    return wrote == bytes.size();
}

bool DecisionJournal::load(const std::string& path) {
    platform::MappedFile mf;
    if (!mf.open(path)) {
        // A load that fails SILENTLY is the bug class this project keeps
        // finding: an unreadable record is indistinguishable from an empty one.
        error_ = "journal: cannot open " + path;
        return false;
    }
    return deserialise(mf.bytes(), static_cast<size_t>(mf.size()));
}

void DecisionJournal::clear() {
    entries_.clear();
    next_id_ = 1;
    error_.clear();
}

void DecisionJournal::prune() {
    if (cfg_.max_entries == 0 || entries_.size() <= cfg_.max_entries) return;
    const size_t drop = entries_.size() - cfg_.max_entries;
    entries_.erase(entries_.begin(), entries_.begin() + static_cast<ptrdiff_t>(drop));
}

// ===========================================================================
// FeedbackHook
// ===========================================================================
int64_t FeedbackHook::observe(const StreamEvent& e, const std::string& task_key) {
    return journal_.record_event(e, task_key);
}

size_t FeedbackHook::observe_stream(const std::vector<StreamEvent>& events,
                                    const std::string& task_key) {
    return journal_.record_stream(events, task_key);
}

int64_t FeedbackHook::observe_decision(const DecisionResult& d,
                                       const std::string& task_key) {
    return journal_.record_decision(d, task_key);
}

size_t FeedbackHook::outcome(const std::string& task_key, DecisionAction realised,
                             float score, int32_t horizon) {
    return journal_.resolve_all(task_key, realised, score, horizon);
}

bool FeedbackHook::resolve(int64_t id, DecisionAction realised, float score,
                           int32_t horizon) {
    return journal_.resolve(id, realised, score, horizon);
}

bool FeedbackHook::resolve_kind(int64_t id, OutcomeKind kind) {
    return journal_.resolve_kind(id, kind);
}

UserFeedbackLoop::Verdict FeedbackHook::on_user_turn(const std::string& user_text,
                                                     const std::string& task_key) {
    const UserFeedbackLoop::Verdict v = UserFeedbackLoop::parse(user_text);

    // A neutral turn is NOT feedback. Recording it would let the absence of a
    // verdict move the trust ledger, and resolving the journal on it would let
    // silence score the head.
    if (v == UserFeedbackLoop::Verdict::None) return v;

    feedback_.record(v, task_key);

    switch (v) {
        case UserFeedbackLoop::Verdict::Confirm:
            journal_.resolve_latest(task_key, OutcomeKind::Confirmed);
            break;
        case UserFeedbackLoop::Verdict::Correct:
            journal_.resolve_latest(task_key, OutcomeKind::Corrected);
            break;
        case UserFeedbackLoop::Verdict::Reject:
            // Recorded as its own kind: a decision that should not have been
            // made is NOT counted as a miss.
            journal_.resolve_latest(task_key, OutcomeKind::Rejected);
            break;
        case UserFeedbackLoop::Verdict::None:
            break;
    }
    return v;
}

void FeedbackHook::on_implicit(const std::string& task_key, bool engaged) {
    // The TRUST LEDGER only. Engagement is a weak signal about the
    // conversation, not an outcome for a prediction, and treating it as one
    // would let a user scrolling past score the head.
    feedback_.record_implicit(task_key, engaged);
}

size_t FeedbackHook::expire(const std::string& task_key, int64_t through_step) {
    return journal_.expire_through(through_step, task_key);
}

std::string FeedbackHook::to_json() const {
    std::string o = "{\"journal\":" + journal_.to_json();
    o += ",\"ledger_keys\":" + std::to_string(feedback_.size());
    o += ",\"ledger\":{";
    // SORTED, so two runs that recorded the same verdicts produce the same
    // string. `unordered_map` iteration order is unspecified, and a log line
    // whose key order changes between runs cannot be diffed — which is the same
    // reason serialise() sorts.
    std::vector<std::pair<std::string, std::pair<uint32_t, uint32_t>>> rows(
        feedback_.table().begin(), feedback_.table().end());
    std::sort(rows.begin(), rows.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    bool first = true;
    for (const auto& kv : rows) {
        if (!first) o += ",";
        first = false;
        o += "\"" + json_escape(kv.first) + "\":";
        o += "{\"confirm\":" + std::to_string(kv.second.first);
        o += ",\"correct\":" + std::to_string(kv.second.second);
        o += ",\"trust\":" + num(feedback_.trust(kv.first)) + "}";
    }
    o += "}}";
    return o;
}

// ---- container persistence ------------------------------------------------
//
// ONE file holding BOTH stores, so they cannot be restored out of step. The
// two payloads are the exact bytes each store's own serialise() produces: no
// format is re-implemented here, so the two copies cannot drift.
// ---------------------------------------------------------------------------
bool FeedbackHook::save(const std::string& path) const {
    const std::string j = journal_.serialise();
    const std::string f = feedback_.serialise();

    std::string o;
    o.append(kHookMagic, 4);
    put_u32(o, kHookVersion);
    put_u32(o, static_cast<uint32_t>(j.size()));
    o += j;
    put_u32(o, static_cast<uint32_t>(f.size()));
    o += f;

    FILE* fp = platform::open_file_c(path.c_str(), "wb");
    if (!fp) return false;
    const size_t wrote = std::fwrite(o.data(), 1, o.size(), fp);
    std::fclose(fp);
    return wrote == o.size();
}

bool FeedbackHook::load(const std::string& path) {
    platform::MappedFile mf;
    if (!mf.open(path)) {
        error_ = "feedback hook: cannot open " + path;
        return false;
    }

    const uint8_t* p = mf.bytes();
    const size_t total = static_cast<size_t>(mf.size());
    size_t cur = 0;
    auto need = [&](size_t n) -> bool { return cur <= total && n <= total - cur; };

    // EVERY failure path below sets error_. A load that returns false and says
    // nothing is the bug class this project keeps finding: an unreadable record
    // is indistinguishable from an empty one, and a caller cannot tell a
    // corrupt file from a missing one.
    if (total < 12 || std::memcmp(p, kHookMagic, 4) != 0) {
        error_ = "feedback hook: bad magic or truncated header in " + path;
        return false;
    }
    cur += 4;
    uint32_t version = 0;
    std::memcpy(&version, p + cur, 4); cur += 4;
    if (version != kHookVersion) {
        error_ = "feedback hook: unsupported version " +
                 std::to_string(version) + " in " + path;
        return false;
    }

    uint32_t jlen = 0;
    if (!need(4)) { error_ = "feedback hook: truncated journal length"; return false; }
    std::memcpy(&jlen, p + cur, 4); cur += 4;
    if (!need(jlen)) {
        error_ = "feedback hook: the journal payload is truncated";
        return false;
    }
    const uint8_t* jp = p + cur;
    cur += jlen;

    uint32_t flen = 0;
    if (!need(4)) { error_ = "feedback hook: truncated ledger length"; return false; }
    std::memcpy(&flen, p + cur, 4); cur += 4;
    if (!need(flen)) {
        error_ = "feedback hook: the ledger payload is truncated";
        return false;
    }
    const uint8_t* fp = p + cur;

    // FAIL CLOSED, and fail ATOMICALLY: both payloads are parsed into locals
    // first. A journal that restored while its ledger did not would silently
    // reset every trust factor to neutral, which is a quiet loss of the only
    // thing the ledger is for.
    DecisionJournal  jtmp;
    UserFeedbackLoop ftmp;
    if (!jtmp.deserialise(jp, jlen)) {
        error_ = "feedback hook: " + jtmp.error();
        return false;
    }
    if (!ftmp.deserialise(fp, flen)) {
        error_ = "feedback hook: the ledger payload is malformed";
        return false;
    }

    journal_  = std::move(jtmp);
    feedback_ = std::move(ftmp);
    error_.clear();
    return true;
}

void FeedbackHook::clear() {
    journal_.clear();
    feedback_.clear();
    error_.clear();
}

} // namespace omniseed
