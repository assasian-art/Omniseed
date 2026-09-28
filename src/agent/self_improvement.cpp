// =============================================================================
//  OmniSeed — self_improvement.cpp
//  Success-path caching + error-pattern learning + Dream-State consolidation.
// =============================================================================
#include "omniseed/agent/agent.h"
#include "omniseed/core/platform.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <sstream>

namespace omniseed {

namespace {
// The on-disk magic, spelled out as BYTES on purpose.
//
// It used to be a uint32 (0x54524953) in save() and the string "SRIT" in
// load(). 0x54524953 is stored LITTLE-ENDIAN as 'S','I','R','T' — not "SRIT"
// — so save() and load() disagreed about the format for the module's whole
// life: load() rejected every file save() produced, the trace cache never
// actually persisted, and the failure was silent because a missing cache is
// indistinguishable from an empty one. One constant now serves both sides.
constexpr char kMagic[4] = {'S', 'I', 'R', 'T'};
}  // namespace

// ===========================================================================
// Task keys
// ===========================================================================
std::string SelfImprovement::task_key(const std::string& task) {
    std::string out;
    out.reserve(task.size());
    // Starts TRUE so a LEADING whitespace run is suppressed rather than
    // emitted as a leading space. It used to start false, which meant
    // "  read a file" keyed to " read a file" while "read a file" keyed to
    // "read a file" — two cache entries for one task, and the replay lookup
    // (which is exact-match) would miss every time the caller's string had
    // leading whitespace. The trailing trim below never caught it because it
    // only strips the END.
    bool last_ws = true;
    for (const char c : task) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!last_ws) out += ' ';
            last_ws = true;
        } else {
            out += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            last_ws = false;
        }
    }
    while (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

// ===========================================================================
// Recording
// ===========================================================================
void SelfImprovement::record(const std::string& key,
                             const std::vector<std::string>& steps,
                             bool success, double ms) {
    TaskTrace* t = nullptr;
    for (TaskTrace& c : traces_) {
        if (c.task_key == key) { t = &c; break; }
    }
    if (t == nullptr) {
        if (traces_.size() >= cfg_.max_traces) {
            // evict least-recently-successful
            auto victim = std::min_element(
                traces_.begin(), traces_.end(),
                [](const TaskTrace& a, const TaskTrace& b) {
                    return a.success_count < b.success_count;
                });
            traces_.erase(victim);
        }
        traces_.emplace_back();
        t = &traces_.back();
        t->task_key = key;
    }

    const auto now = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    if (success) {
        t->steps = steps;
        t->success_count++;
        t->avg_ms = t->avg_ms == 0.0 ? ms : (t->avg_ms + ms) / 2.0;
    } else {
        t->fail_count++;
    }
    t->last_used = static_cast<uint64_t>(now);
}

// ===========================================================================
// Replay lookup
// ===========================================================================
bool SelfImprovement::find_replay(const std::string& key,
                                  std::vector<std::string>& steps) const {
    for (const TaskTrace& t : traces_) {
        if (t.task_key != key) continue;
        const uint32_t total = t.success_count + t.fail_count;
        const double rate = total > 0
            ? static_cast<double>(t.success_count) / total
            : 0.0;
        if (t.success_count > 0 && rate >= cfg_.replay_min_success_rate) {
            steps = t.steps;
            return true;
        }
    }
    return false;
}

// ===========================================================================
// Dream-State Consolidation
// ===========================================================================
void SelfImprovement::dream() {
    const auto now = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    std::vector<TaskTrace> kept;
    kept.reserve(traces_.size());

    for (TaskTrace& t : traces_) {
        const uint64_t age_s = static_cast<uint64_t>(now) >
                               t.last_used ? static_cast<uint64_t>(now) -
                                             t.last_used : 0;
        const uint64_t days = age_s / 86400;

        const uint32_t total = t.success_count + t.fail_count;
        const double rate = total > 0
            ? static_cast<double>(t.success_count) / total : 0.0;

        // prune: failures-only or very stale & low success
        if (t.success_count == 0 && t.fail_count > 2) continue;
        if (days > 30 && rate < 0.5) continue;

        // decay long-unused traces (success weight halves every ~14 days)
        if (days > 14) {
            t.success_count = static_cast<uint32_t>(t.success_count * 0.5);
        }

        if (t.success_count > 0) kept.push_back(t);
    }

    traces_.swap(kept);
}

// ===========================================================================
// Persistence
// ===========================================================================
bool SelfImprovement::save(const std::string& path) const {
    FILE* f = platform::open_file_c(path.c_str(), "wb");
    if (!f) return false;

    const uint32_t version = 1;
    const uint32_t n = static_cast<uint32_t>(traces_.size());
    std::fwrite(kMagic, 1, 4, f);
    std::fwrite(&version, 4, 1, f);
    std::fwrite(&n, 4, 1, f);

    for (const TaskTrace& t : traces_) {
        const uint32_t klen = static_cast<uint32_t>(t.task_key.size());
        std::fwrite(&klen, 4, 1, f);
        std::fwrite(t.task_key.data(), 1, klen, f);

        const uint32_t ns = static_cast<uint32_t>(t.steps.size());
        std::fwrite(&ns, 4, 1, f);
        for (const std::string& s : t.steps) {
            const uint32_t sl = static_cast<uint32_t>(s.size());
            std::fwrite(&sl, 4, 1, f);
            std::fwrite(s.data(), 1, sl, f);
        }

        std::fwrite(&t.success_count, 4, 1, f);
        std::fwrite(&t.fail_count, 4, 1, f);
        std::fwrite(&t.avg_ms, 8, 1, f);
        std::fwrite(&t.last_used, 8, 1, f);
    }
    std::fclose(f);
    return true;
}

bool SelfImprovement::load(const std::string& path) {
    platform::MappedFile mf;
    if (!mf.open(path)) return false;

    const uint8_t* p = mf.bytes();
    const size_t total = static_cast<size_t>(mf.size());
    size_t cur = 0;

    // EVERY length below is read straight out of the file, so every length is
    // attacker/corruption controlled. Without a bounds check a truncated
    // state/improve.bin walks off the end of the mapping and reads whatever
    // follows it. `need()` is the only thing standing between a corrupt file
    // and an out-of-bounds read, so it is applied before EVERY read.
    auto need = [&](size_t n) -> bool {
        return cur <= total && n <= total - cur;
    };

    if (total < 12 || std::memcmp(p, kMagic, 4) != 0) return false;
    cur += 4;
    uint32_t version = 0;
    std::memcpy(&version, p + cur, 4); cur += 4;
    if (version != 1) return false;
    uint32_t n = 0;
    std::memcpy(&n, p + cur, 4); cur += 4;
    // An absurd count is rejected before it is used to reserve, so a 12-byte
    // file claiming 4 billion traces cannot make us allocate 4 billion of them.
    if (n > static_cast<uint32_t>(cfg_.max_traces) * 64u + 1024u) return false;

    // Parse into a LOCAL vector and swap only on full success: a failure
    // half-way through leaves the previous traces intact instead of replacing
    // them with a partially-loaded set.
    std::vector<TaskTrace> loaded;
    loaded.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        TaskTrace t;
        uint32_t klen = 0;
        if (!need(4)) return false;
        std::memcpy(&klen, p + cur, 4); cur += 4;
        if (!need(klen)) return false;
        t.task_key.assign(reinterpret_cast<const char*>(p + cur), klen);
        cur += klen;

        uint32_t ns = 0;
        if (!need(4)) return false;
        std::memcpy(&ns, p + cur, 4); cur += 4;
        if (ns > (1u << 20)) return false;
        t.steps.resize(ns);
        for (uint32_t s = 0; s < ns; ++s) {
            uint32_t sl = 0;
            if (!need(4)) return false;
            std::memcpy(&sl, p + cur, 4); cur += 4;
            if (!need(sl)) return false;
            t.steps[s].assign(reinterpret_cast<const char*>(p + cur), sl);
            cur += sl;
        }

        if (!need(4 + 4 + 8 + 8)) return false;
        std::memcpy(&t.success_count, p + cur, 4); cur += 4;
        std::memcpy(&t.fail_count,    p + cur, 4); cur += 4;
        std::memcpy(&t.avg_ms,        p + cur, 8); cur += 8;
        std::memcpy(&t.last_used,     p + cur, 8); cur += 8;
        loaded.push_back(std::move(t));
    }

    traces_.swap(loaded);
    return true;
}

} // namespace omniseed
