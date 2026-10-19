// =============================================================================
//  OmniSeed — sidecar_hub.cpp  (T123b)
//  Refcounted demand-map / pressure-unmap registry. Contract: the header.
// =============================================================================
#include "omniseed/runtime/sidecar_hub.h"

#include <algorithm>
#include <cstdio>

#include "omniseed/core/platform.h"

namespace omniseed {

SidecarHub::~SidecarHub() = default;   // shared_ptr<void> deleters run here

bool SidecarHub::register_spec(const std::string& name, Spec spec,
                               std::string& err) {
    err.clear();
    if (!spec.load) {
        err = "sidecar '" + name + "': empty loader";
        return false;
    }
    auto it = entries_.find(name);
    if (it != entries_.end()) {
        if (it->second.obj) {
            // Mid-use spec swap would mix two objects under one name.
            err = "sidecar '" + name + "': already mapped (spec swap refused)";
            return false;
        }
        it->second.spec = std::move(spec);   // registered-but-unmapped: ok
        return true;
    }
    Entry e;
    e.spec = std::move(spec);
    entries_.emplace(name, std::move(e));
    return true;
}

std::shared_ptr<void> SidecarHub::acquire(const std::string& name,
                                          std::string& err) {
    err.clear();
    auto it = entries_.find(name);
    if (it == entries_.end()) {
        err = "sidecar '" + name + "': not registered";
        return nullptr;
    }
    Entry& e = it->second;
    if (!e.obj) {
        std::string lerr;
        std::shared_ptr<void> fresh = e.spec.load(lerr);
        if (!fresh) {
            // Fail-closed: purge so the NEXT acquire retries (a transient
            // file replacement must self-heal, never stick on a dead cache).
            ++load_failures_;
            entries_.erase(it);
            err = "sidecar '" + name + "' load failed: " + lerr;
            return nullptr;
        }
        e.obj = std::move(fresh);
    }
    ++e.users;
    return e.obj;
}

void SidecarHub::release(const std::string& name) {
    auto it = entries_.find(name);
    if (it == entries_.end() || it->second.users <= 0) {
        // Malformed release: bookkeeping bug, not a crash (contract).
        ++dropped_releases_;
        return;
    }
    --it->second.users;
}

uint64_t SidecarHub::on_rss_pressure(uint64_t* zone_out) {
    const uint64_t mb = platform::current_rss_bytes() / (1048576ull);
    const uint64_t zone = (mb >= hard_mb_) ? 2 : ((mb >= soft_mb_) ? 1 : 0);
    if (zone_out) *zone_out = zone;
    if (zone == 0) return 0;   // demand-only remap: never eagerly reload

    uint64_t freed = 0;
    for (auto& kv : entries_) {
        Entry& e = kv.second;
        if (!e.obj) continue;                 // nothing mapped
        if (e.users > 0) continue;            // refcount guard — NEVER busy
        if (e.spec.bytes_hint < min_unmap_bytes_) continue;  // 40 KB vision
        // Measured reclaim: file-backed pages vanish on unmap only for the
        // loader-owned HEAP copies — sample RSS around the reset (the header
        // states this honesty rule; we log the observed delta, not the
        // file size).
        const uint64_t before = platform::current_rss_bytes();
        e.obj.reset();                        // full-unmap RAII
        const uint64_t after = platform::current_rss_bytes();
        if (after < before) freed += before - after;
        ++reclaim_events_;
        std::printf("[sidecar] pressure-unmapped %s (zone=%llu, rss -%.1f MB, "
                    "users=0, >=%lluMB)\n",
                    kv.first.c_str(),
                    static_cast<unsigned long long>(zone),
                    static_cast<double>(before - after) / 1048576.0,
                    static_cast<unsigned long long>(min_unmap_bytes_ /
                                                     1048576));
    }
    return freed;
}

bool SidecarHub::is_mapped(const std::string& name) const {
    auto it = entries_.find(name);
    return it != entries_.end() && static_cast<bool>(it->second.obj);
}

int32_t SidecarHub::users(const std::string& name) const {
    auto it = entries_.find(name);
    return it == entries_.end() ? -1 : it->second.users;
}

std::vector<std::string> SidecarHub::mapped_names() const {
    std::vector<std::string> out;
    for (const auto& kv : entries_)
        if (kv.second.obj) out.push_back(kv.first);
    std::sort(out.begin(), out.end());
    return out;
}

} // namespace omniseed
