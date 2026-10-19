// =============================================================================
//  OmniSeed — lora_bank.h  (T122b, flash-skills LoRA bank)
//  Many sidecar GGUFs ("skills as files") resident as mmap'd LoraAdapters;
//  an ACTIVE set composites into one LoraAdapter-shaped view the existing
//  apply_targeted() consumes UNCHANGED (set_lora(&bank.composite())).
//
//  Contract (state/task_T122_brief.md):
//    * Loaded-but-idle adapters are MMAP COST ONLY — they build NO hot fp32
//      copies (load(..., /*build_hot=*/false)). Hot memory is paid once,
//      inside the composite, for the ACTIVE set only.
//    * activate(): unknown name -> false + err listing loaded names (never a
//      partial activation). Later-activated source WINS a (layer,target)
//      collision and the shadow count is reported (silent shadowing is how a
//      skill dies without an error).
//    * Geometry quarantine: wrong n_embd / layer_count / rank outside [1,8]
//      -> adapter is quarantined (error kept, never partially applied).
//    * LRU eviction drops ONLY non-active adapters.
//    * Empty dir / empty active set -> composite() == nullptr semantics:
//      activate({}) is valid and detaches (base behaviour, exact no-op).
// =============================================================================
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "omniseed/core/lora.h"

namespace omniseed {

class LoraBank {
public:
    LoraBank() = default;
    LoraBank(const LoraBank&) = delete;
    LoraBank& operator=(const LoraBank&) = delete;

    // Scan `dir` for *.gguf sidecars and load every valid one (mmap only —
    // no hot fp32 copies). expect_n_embd/expect_layers <= 0 disables that
    // geometry check (tests of pure load behaviour). Quarantined files keep
    // their error string for report(). Returns false only on I/O-level
    // failure of the DIRECTORY itself (missing dir) — an empty-but-existing
    // dir is a valid empty bank.
    bool open(const std::string& dir, int32_t expect_n_embd,
              int32_t expect_layers, std::string& err);

    // Activate `names` as the active set (order = override order: LAST wins
    // a slot collision). Names are the sidecar FILE STEM ("skill" for
    // skill.gguf). rebuilds the composite; empty list detaches.
    // Returns false WITHOUT touching the current composite on unknown name
    // or already-quarantined name (atomic activation).
    bool activate(const std::vector<std::string>& names, std::string& err);

    // Composite view for set_lora(): valid while this bank is alive and
    // until the next activate(). nullptr when nothing is active.
    const LoraAdapter* composite() const {
        return active_.empty() ? nullptr : &composite_;
    }

    // Hot fp32 bytes held by the composite (the ONLY hot cost the bank pays).
    int64_t composite_hot_bytes() const;

    // Slots where an earlier active source was overwritten by a later one.
    int64_t shadowed_slots() const { return shadowed_; }

    // Reporting.
    std::vector<std::string> loaded_names() const;
    std::vector<std::string> quarantined() const;   // name: reason pairs
    bool empty() const { return loaded_.empty(); }

    // LRU: evict least-recently-used NON-active adapters until
    // loaded_.size() <= max_loaded (max_loaded >= active_.size()+1 or the
    // call is a no-op — active adapters are never evicted). Returns evicted
    // names. Loaded set below max is untouched.
    std::vector<std::string> evict_idle(size_t max_loaded);

    // Touch on use (activate() does this for active names).
    void touch(const std::string& name);

private:
    struct Entry {
        std::unique_ptr<LoraAdapter> adapter;
        std::string path;
        uint64_t last_use = 0;      // monotonic tick (activate/touch)
        bool active = false;
    };

    std::unordered_map<std::string, Entry> loaded_;   // file stem -> entry
    std::vector<std::pair<std::string, std::string>> quarantine_; // name, why
    std::vector<std::string> active_;                 // in activation order
    LoraAdapter composite_;                           // rebuilt per activate
    int64_t shadowed_ = 0;
    uint64_t tick_ = 0;
};

} // namespace omniseed
