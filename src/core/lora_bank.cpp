// =============================================================================
//  OmniSeed — lora_bank.cpp  (T122b)
//  Directory of sidecar GGUFs -> mmap'd adapters (no hot copies when idle),
//  atomic activation -> composite LoraAdapter view, LRU eviction of idle.
//  Contract + rationale: include/omniseed/core/lora_bank.h + task_T122_brief.
// =============================================================================
#include "omniseed/core/lora_bank.h"

#include <algorithm>
#include <filesystem>

namespace omniseed {

namespace fs = std::filesystem;

bool LoraBank::open(const std::string& dir, int32_t expect_n_embd,
                    int32_t expect_layers, std::string& err) {
    loaded_.clear();
    quarantine_.clear();
    active_.clear();
    shadowed_ = 0;
    err.clear();
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) {
        err = "lora bank dir missing: " + dir;
        return false;
    }
    std::vector<fs::path> files;
    for (const fs::directory_entry& e : fs::directory_iterator(dir, ec)) {
        if (e.is_regular_file() && e.path().extension() == ".gguf")
            files.push_back(e.path());
    }
    if (ec) {
        err = "lora bank dir scan failed: " + ec.message();
        return false;
    }
    std::sort(files.begin(), files.end());           // deterministic order
    for (const fs::path& p : files) {
        const std::string name = p.stem().string();
        auto ad = std::make_unique<LoraAdapter>();
        // mmap-only: idle adapters never pay hot fp32 memory (the brief's
        // actual design change vs load()'s eager hot cache).
        if (!ad->load(p.string(), /*build_hot=*/false)) {
            quarantine_.emplace_back(name, ad->error());
            continue;
        }
        // Geometry quarantine — a mismatched adapter would read OOB slots
        // inside apply_targeted (rank*in assumed == the base linear's in).
        if (expect_n_embd > 0 && ad->n_embd() != expect_n_embd) {
            quarantine_.emplace_back(
                name, "n_embd mismatch (" + std::to_string(ad->n_embd()) +
                          " vs base " + std::to_string(expect_n_embd) + ")");
            continue;
        }
        if (expect_layers > 0 && ad->layer_count() != expect_layers) {
            quarantine_.emplace_back(
                name, "layer_count mismatch (" +
                          std::to_string(ad->layer_count()) + " vs base " +
                          std::to_string(expect_layers) + ")");
            continue;
        }
        if (ad->rank() <= 0 || ad->rank() > 8) {      // rank-4/8 roadmap band
            quarantine_.emplace_back(
                name, "rank " + std::to_string(ad->rank()) +
                          " outside [1,8] (flash-skills band)");
            continue;
        }
        Entry ent;
        ent.adapter = std::move(ad);
        ent.path    = p.string();
        ent.last_use = 0;                            // never touched
        loaded_.emplace(name, std::move(ent));
    }
    return true;
}

bool LoraBank::activate(const std::vector<std::string>& names,
                        std::string& err) {
    err.clear();
    // Atomicity: validate the WHOLE name list before touching the current
    // composite — a failed activate must never half-swap the active set.
    for (const std::string& n : names) {
        auto it = loaded_.find(n);
        if (it == loaded_.end()) {
            bool was_quarantined = false;
            for (const auto& q : quarantine_)
                if (q.first == n) { was_quarantined = true; break; }
            err = "unknown " + std::string(was_quarantined
                       ? "quarantined" : "not-loaded") + " skill '" + n +
                  "'; loaded: [";
            bool first = true;
            for (const auto& kv : loaded_) {
                if (!first) err += ",";
                err += kv.first;
                first = false;
            }
            err += "]";
            return false;                            // composite UNTOUCHED
        }
    }

    for (auto& kv : loaded_) kv.second.active = false;
    active_ = names;
    for (const std::string& n : names) {
        loaded_.at(n).active = true;
        loaded_.at(n).last_use = ++tick_;
    }

    // Rebuild the composite from scratch (deterministic: activation order,
    // later names win a collision — brief's override rule).
    shadowed_ = 0;
    if (names.empty()) {
        // Detach: an empty active set must behave EXACTLY like no adapter.
        composite_.reset_composite();
        return true;
    }
    const LoraAdapter& first = *loaded_.at(names.front()).adapter;
    composite_.begin_composite(first.n_embd(), first.layer_count());
    for (const std::string& n : names) {
        const LoraAdapter& src = *loaded_.at(n).adapter;
        // overwrite=true implements later-wins; fold() returns slots filled.
        const int64_t filled = composite_.fold(src, /*overwrite=*/true);
        if (filled == 0) {
            err = "skill '" + n + "' contributed 0 slots (geometry mismatch "
                  "with the first active skill) — refusing activation";
            composite_.reset_composite();               // fail closed: detach
            active_.clear();
            return false;
        }
    }
    // Shadow accounting: slots filled by >=2 sources. fold() overwrites, so
    // count by re-folding each source's slots against the final set here —
    // O(active^2 * slots) with tiny constants (active set is <= a handful).
    // Idle sources carry no hot entries (build_hot=false), so the presence
    // check must use the zero-copy factors() views, same as fold().
    {
        std::vector<std::vector<uint8_t>> hits(
            static_cast<size_t>(first.layer_count()),
            std::vector<uint8_t>(6, 0));
        for (const std::string& n : names) {
            const LoraAdapter& src = *loaded_.at(n).adapter;
            for (int32_t l = 0; l < first.layer_count(); ++l)
                for (int t = 0; t < 6; ++t) {
                    const auto target = static_cast<LoraTarget>(t);
                    const LoraAdapter::Factors f = src.factors(l, target);
                    if (f.a.numel() > 0 && f.b.numel() > 0) {
                        auto& h = hits[static_cast<size_t>(l)]
                                      [static_cast<size_t>(t)];
                        if (h >= 1) ++shadowed_;      // second+ contributor
                        h = static_cast<uint8_t>(
                            h < 255 ? h + 1 : 2);     // saturate, no wrap
                    }
                }
        }
    }
    return true;
}

int64_t LoraBank::composite_hot_bytes() const {
    if (active_.empty()) return 0;
    int64_t bytes = 0;
    const int32_t L = composite_.layer_count();
    const int64_t E = composite_.n_embd();
    for (int32_t l = 0; l < L; ++l)
        for (int t = 0; t < 6; ++t) {
            const LoraAdapter::HotEntry* e =
                composite_.hot_entry(l, static_cast<LoraTarget>(t));
            if (e == nullptr || e->rank <= 0) continue;
            // a [rank, E] + b [out, rank] fp32
            bytes += (e->rank * E + e->out * e->rank) *
                     static_cast<int64_t>(sizeof(float));
        }
    return bytes;
}

std::vector<std::string> LoraBank::loaded_names() const {
    std::vector<std::string> out;
    out.reserve(loaded_.size());
    for (const auto& kv : loaded_) out.push_back(kv.first);
    std::sort(out.begin(), out.end());
    return out;
}

std::vector<std::string> LoraBank::quarantined() const {
    std::vector<std::string> out;
    for (const auto& q : quarantine_)
        out.push_back(q.first + ": " + q.second);
    return out;
}

std::vector<std::string> LoraBank::evict_idle(size_t max_loaded) {
    std::vector<std::string> evicted;
    if (max_loaded >= loaded_.size()) return evicted;
    // Candidate pool: non-active only, oldest last_use first.
    std::vector<std::pair<uint64_t, std::string>> idle;
    for (const auto& kv : loaded_)
        if (!kv.second.active)
            idle.emplace_back(kv.second.last_use, kv.first);
    std::sort(idle.begin(), idle.end());             // oldest first
    for (const auto& cand : idle) {
        if (loaded_.size() <= max_loaded) break;
        loaded_.erase(cand.second);                  // unmap + drop hot-free
        evicted.push_back(cand.second);
    }
    return evicted;
}

void LoraBank::touch(const std::string& name) {
    auto it = loaded_.find(name);
    if (it != loaded_.end()) it->second.last_use = ++tick_;
}

} // namespace omniseed
