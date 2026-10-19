// =============================================================================
//  OmniSeed — sidecar_hub.h  (T123b, sidecar hot-swap governor, grok S07)
//  Refcounted sidecar registry: demand-map on acquire, pressure-unmap on
//  RSS zone >= soft (state/task_T123_brief.md).
//
//  Design (and what it deliberately does NOT do):
//    * No monitor thread (agent.h precedent: budget enforcement rides the
//      caller's existing loop). The server polls zone BEFORE each /asr —
//      the serial request loop IS the sampling point.
//    * zone>=soft -> unmap entries with users==0 AND bytes >= min_unmap_mb.
//      zone back to 0 -> do NOT remap (demand-only; eager remap bounces).
//    * Full-unmap RAII only (GgufLoader dtor); no partial madvise.
//    * HONEST MEMORY WORDING: sidecar weights are file-backed mmap — under
//      pressure the OS reclaims those pages anyway. The measurable wins are
//      (a) deterministic freed-BEFORE-the-next-sample + (b) whatever
//      heap-side copies the loader owned (decode tables etc.), observed as
//      current_rss_bytes() delta across on_rss_zone(). Do not claim the
//      file size as "saved heap".
//
//  Fail-closed:
//    * loader failure -> entry NOT cached (purged), next acquire retries
//      (transient file replacement self-heals);
//    * unmap NEVER touches users>0 entries (refcount guard);
//    * double-release / unknown release -> clamped, counted, never a crash;
//    * zone never fires -> map-once-keep (today's behaviour, zero change).
// =============================================================================
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace omniseed {

class SidecarHub {
public:
    // A loader returns an owned opaque object; the hub unmaps by invoking
    // the deleter. bytes_hint = what the loader believes it will hold (the
    // hub uses it ONLY to skip tiny sidecars — e.g. vision-proj at 40 KB).
    using Loader = std::function<std::shared_ptr<void>(std::string& err)>;
    struct Spec {
        Loader load;
        uint64_t bytes_hint = 0;     // 0 = never pressure-unmapped (unknown)
    };

    // Thresholds mirror ComputeThrottle defaults (260/295) so the hub and
    // the generation loop agree on what "pressure" means.
    void rss_limits(uint32_t soft_mb = 260, uint32_t hard_mb = 295) {
        soft_mb_ = soft_mb;
        hard_mb_ = hard_mb;
    }

    // Registers a sidecar under `name`. Re-registering an ALREADY-mapped
    // name with a different spec is refused (spec swap mid-use is how a
    // caller gets half of A and half of B).
    bool register_spec(const std::string& name, Spec spec,
                       std::string& err);

    // Demand-map: returns the object, bumping users. Loader failure purges
    // the entry (err set, nullptr returned) — never a cached dead object.
    std::shared_ptr<void> acquire(const std::string& name, std::string& err);

    // Drop one user. Unknown name or users already 0: clamps + counts
    // (malformed release is a bookkeeping bug, not a crash).
    void release(const std::string& name);

    // Poll the CURRENT rss zone (platform::current_rss_bytes vs limits).
    // >=1 pressure: unmap zero-user entries >= min_unmap_mb, log one line
    // each. Returns bytes actually reclaimed (0 = nothing eligible).
    // zone==0: no-op (demand-only remap — see header contract).
    uint64_t on_rss_pressure(uint64_t* zone_out = nullptr);

    // Introspection / tests.
    bool is_mapped(const std::string& name) const;
    int32_t users(const std::string& name) const;
    uint64_t reclaim_events() const { return reclaim_events_; }
    uint64_t dropped_releases() const { return dropped_releases_; }
    uint64_t load_failures() const { return load_failures_; }
    std::vector<std::string> mapped_names() const;

    // Bytes at/above which an idle sidecar is pressure-unmap eligible.
    void min_unmap_bytes(uint64_t bytes) { min_unmap_bytes_ = bytes; }

    ~SidecarHub();   // defined here: any mapped object's deleter runs

private:
    struct Entry {
        Spec spec;
        std::shared_ptr<void> obj;   // null = registered, not mapped
        int32_t users = 0;
        bool load_failed_last = false;   // diagnostics only (no caching)
    };
    std::unordered_map<std::string, Entry> entries_;
    uint32_t soft_mb_ = 260;
    uint32_t hard_mb_ = 295;
    uint64_t min_unmap_bytes_ = 8ull * 1024 * 1024;   // 8 MB per brief
    uint64_t reclaim_events_ = 0;
    uint64_t dropped_releases_ = 0;
    uint64_t load_failures_ = 0;
};

} // namespace omniseed
