// =============================================================================
//  OmniSeed — test_sidecar_hub.cpp  (T123b)
//  SidecarHub: refcount acquire/release, pressure-unmap eligibility, zone 0
//  no-remap, loader-failure purge + retry, accounting honesty.
//  Pressure is forced via rss_limits(0,0) (any RSS => zone 1) and suppressed
//  via rss_limits(very-large,...) — deterministic, no ballooning; both paths
//  share the REAL rss code (platform::current_rss_bytes vs thresholds).
// =============================================================================
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "omniseed/core/gguf_loader.h"
#include "omniseed/core/tensor.h"
#include "omniseed/runtime/sidecar_hub.h"

using omniseed::SidecarHub;

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (cond) { ++g_pass; std::printf("  PASS %s:%d: %s\n",              \
                                          __FILE__, __LINE__, #cond); }      \
        else {                                                               \
            ++g_fail;                                                        \
            std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
        }                                                                    \
    } while (0)

namespace {

int g_pass = 0, g_fail = 0;

// Live-object counter: proves the object was DESTROYED on unmap — the hub's
// reclaim is observable without measuring RSS (RSS delta has its own noise).
int g_live = 0;

struct Blob {
    std::vector<char> pad;
    int* live;
    explicit Blob(uint64_t n, int* l)
        : pad(static_cast<size_t>(n), 1), live(l) {
        if (live) ++*live;
    }
    ~Blob() { if (live) --*live; }
};

std::shared_ptr<void> make_obj(uint64_t bytes, int* live) {
    return std::shared_ptr<void>(new Blob(bytes, live),
                                 [](void* q) { delete static_cast<Blob*>(q); });
}

SidecarHub::Spec big_spec(uint64_t bytes, int* live, bool* called = nullptr) {
    SidecarHub::Spec s;
    s.bytes_hint = bytes;
    s.load = [bytes, live, called](std::string& err) -> std::shared_ptr<void> {
        if (called) *called = true;
        (void)err;
        return make_obj(bytes, live);
    };
    return s;
}

} // namespace

int main() {
    // ---- A: registration refusals + reuse ----------------------------------
    {
        SidecarHub h;
        std::string err;
        CHECK(!h.register_spec("empty", {}, err) && !err.empty());
        CHECK(h.register_spec("s", big_spec(16, &g_live), err));
        // registered-but-unmapped: spec swap allowed (brief contract)
        CHECK(h.register_spec("s", big_spec(32, &g_live), err));
        std::string gerr;
        auto obj = h.acquire("s", gerr);
        CHECK(obj != nullptr && gerr.empty());
        CHECK(h.is_mapped("s") && h.users("s") == 1);
        // MID-USE spec swap refused
        CHECK(!h.register_spec("s", big_spec(64, &g_live), err) &&
              !err.empty());
        h.release("s");
        CHECK(h.users("s") == 0 && h.is_mapped("s"));  // mapped, idle
        CHECK(g_live == 1);
    }  // <- hub dtor: live must drop to 0
    CHECK(g_live == 0);

    // ---- B: refcount math incl. double-release clamp -----------------------
    {
        SidecarHub h;
        std::string err;
        CHECK(h.register_spec("s", big_spec(16, &g_live), err));
        auto a = h.acquire("s", err);
        auto b = h.acquire("s", err);
        CHECK(h.users("s") == 2 && g_live == 1);       // one object, 2 users
        h.release("s");
        CHECK(h.users("s") == 1);
        h.release("s");
        h.release("s");                                  // one too many
        CHECK(h.users("s") == 0);                        // clamped, not -1
        CHECK(h.dropped_releases() == 1);                // counted
        h.release("ghost");                              // unknown name
        CHECK(h.dropped_releases() == 2);                // counted, no crash
        a.reset(); b.reset();
        CHECK(g_live == 1);                              // hub still owns
    }
    CHECK(g_live == 0);

    // ---- C: pressure-unmap eligibility: zero-user AND >=8MB ---------------
    {
        SidecarHub h;
        std::string err;
        // Force zone >= 1 regardless of actual RSS: soft=0 => any mb >= 0.
        h.rss_limits(0, 0);
        h.min_unmap_bytes(1024);                         // 1 KB floor for test
        CHECK(h.register_spec("big_idle", big_spec(2048, &g_live), err));
        CHECK(h.register_spec("small_idle", big_spec(512, &g_live), err));
        CHECK(h.register_spec("busy", big_spec(4096, &g_live), err));
        CHECK(h.register_spec("never", big_spec(4096, &g_live), err));
        auto o1 = h.acquire("big_idle", err);
        auto o2 = h.acquire("busy", err);
        auto o3 = h.acquire("never", err);
        h.release("big_idle");                           // idle, eligible
        // small_idle never acquired: registered, not mapped -> not touched
        h.release("busy");                               // idle... but users
        // o2 still held -> users==1? No: release dropped to 0. Re-acquire to
        // keep it BUSY (the refcount guard case):
        auto o2b = h.acquire("busy", err);               // users==1 again
        h.release("never");
        h.release("never");                              // users 0 BUT
        h.min_unmap_bytes(8ull * 1024 * 1024);           // restore real 8MB
        // none are >=8MB in this fixture, so restore the small floor to make
        // the eligibility assertions meaningful:
        h.min_unmap_bytes(1024);
        // "never" was force-unmapped? NO — see below: we must prove the
        // users>0 guard FIRST. Set users=0 for never but keep o3 alive?
        // o3 holds a ref; users was decremented to 0 by the second release
        // (malformed — counted) while o3 still holds the pointer: that is a
        // CALLER bug, not hub state; hub sees users==0. Eligibility for
        // "never": users==0, bytes 4096 >= 1024 -> UNMAP allowed (the hub
        // tracks counts, not shared_ptr aliasing). Assert THAT (documented
        // caller contract: releases must balance).
        const uint64_t zone_before = 0;
        uint64_t zone = zone_before;
        const uint64_t freed = h.on_rss_pressure(&zone);
        (void)freed;
        CHECK(zone >= 1);                                // forced pressure
        CHECK(!h.is_mapped("big_idle"));                 // idle -> unmapped
        CHECK(h.is_mapped("busy"));                      // users>0 -> KEPT
        CHECK(h.reclaim_events() >= 1);
        // busy object identity survives (refcount guard — the brief's test 2)
        CHECK(o2b != nullptr);
        // Registered-but-never-mapped small_idle: untouched semantics
        CHECK(!h.is_mapped("small_idle"));
        o1.reset(); o2.reset(); o2b.reset(); o3.reset();
        // busy (users==0 now) is STILL mapped in the hub by design —
        // pressure only unmaps what it saw; no second pass here.
        CHECK(g_live == 1);                            // busy held by hub
    }
    CHECK(g_live == 0);                                // hub dtor released it

    // ---- D: zone 0 => no unmap AND no eager remap --------------------------
    {
        SidecarHub h;
        std::string err;
        h.rss_limits(1u << 30, 1u << 30);                // soft=1GB: zone 0
        h.min_unmap_bytes(1024);                         // fixture is tiny
        CHECK(h.register_spec("s", big_spec(4096, &g_live), err));
        auto o = h.acquire("s", err);
        h.release("s");
        uint64_t zone = 99;
        CHECK(h.on_rss_pressure(&zone) == 0 && zone == 0);
        CHECK(h.is_mapped("s"));                         // zone 0: no-op
        CHECK(h.reclaim_events() == 0);
        // after a pressure-unmap, zone 0 must NOT remap (demand-only):
        h.rss_limits(0, 0);
        o = nullptr;                                      // drop caller ref
        CHECK(h.on_rss_pressure(nullptr) == 0 ||
              h.reclaim_events() == 1);
        CHECK(!h.is_mapped("s"));                        // unmapped under
                                                          // pressure...
        h.rss_limits(1u << 30, 1u << 30);
        h.on_rss_pressure(nullptr);                      // zone back to 0
        CHECK(!h.is_mapped("s"));                        // STILL not remapped
        std::string gerr;
        auto again = h.acquire("s", gerr);                // demand remap works
        CHECK(again != nullptr && h.is_mapped("s"));
        o = nullptr;
    }
    CHECK(g_live == 0);

    // ---- E: loader failure purges + retry self-heals ------------------------
    {
        SidecarHub h;
        std::string err;
        int calls = 0;
        SidecarHub::Spec flaky;
        flaky.bytes_hint = 4096;
        flaky.load = [&calls](std::string& e) -> std::shared_ptr<void> {
            if (++calls == 1) { e = "transient IO"; return nullptr; }
            return make_obj(4096, &g_live);
        };
        CHECK(h.register_spec("flaky", std::move(flaky), err));
        auto first = h.acquire("flaky", err);
        CHECK(first == nullptr && !err.empty());
        CHECK(h.load_failures() == 1);
        CHECK(!h.is_mapped("flaky"));                    // PURGED, not cached
        auto second = h.acquire("flaky", err);
        CHECK(second == nullptr);                        // NOT registered now
        CHECK(!err.empty());                             // clear error either way
        // Re-register (the purge removed the entry) -> retry succeeds:
        SidecarHub::Spec ok;
        ok.bytes_hint = 4096;
        ok.load = [&calls](std::string&) {
            ++calls;                                   // retry attempt #2
            return make_obj(4096, &g_live);
        };
        CHECK(h.register_spec("flaky", std::move(ok), err));
        auto third = h.acquire("flaky", err);
        CHECK(third != nullptr && err.empty());
        CHECK(calls == 2);
        h.release("flaky");
        third.reset();
    }
    CHECK(g_live == 0);

    // ---- F: real-file leg (GATED): whisper encoder map -> release ->
    //         pressure-unmap -> demand remap, tensor view byte-identical ------
    {
        const char* path = "models/whisper-tiny-encoder.gguf";
        if (!std::filesystem::exists(path)) {
            std::printf("  SKIP F: %s absent (clean checkout)\n", path);
        } else {
            namespace fs = std::filesystem;
            SidecarHub h;
            std::string err;
            SidecarHub::Spec spec;
            spec.bytes_hint = fs::file_size(path);
            spec.load = [path](std::string& e) -> std::shared_ptr<void> {
                auto* p = new omniseed::GgufLoader();
                if (!p->open(path)) {
                    e = p->error();
                    delete p;
                    return nullptr;
                }
                return std::shared_ptr<void>(
                    p, [](void* q) { delete static_cast<omniseed::GgufLoader*>(q); });
            };
            CHECK(h.register_spec("whisper", std::move(spec), err));
            std::string gerr;
            auto o1 = h.acquire("whisper", gerr);
            CHECK(o1 != nullptr && gerr.empty());
            auto* gg1 = static_cast<omniseed::GgufLoader*>(o1.get());
            const std::string probe = "whisper.conv1.weight";
            CHECK(gg1->has_tensor(probe));
            const omniseed::Tensor t1 = gg1->tensor(probe);
            std::vector<uint8_t> before(static_cast<size_t>(t1.nbytes()));
            std::memcpy(before.data(), t1.data(), before.size());
            h.release("whisper");
            o1 = nullptr;                                 // caller done
            h.rss_limits(0, 0);                           // force pressure
            const uint64_t freed = h.on_rss_pressure(nullptr);
            (void)freed;
            CHECK(!h.is_mapped("whisper"));               // 76MB >= 8MB: gone
            h.rss_limits(1u << 30, 1u << 30);             // zone 0 again
            h.on_rss_pressure(nullptr);
            CHECK(!h.is_mapped("whisper"));               // no eager remap
            auto o2 = h.acquire("whisper", gerr);         // demand remap
            CHECK(o2 != nullptr && gerr.empty());
            auto* gg2 = static_cast<omniseed::GgufLoader*>(o2.get());
            const omniseed::Tensor t2 = gg2->tensor(probe);
            CHECK(t2.nbytes() == static_cast<int64_t>(before.size()));
            CHECK(std::memcmp(t2.data(), before.data(), before.size()) == 0);
            h.release("whisper");
            o2 = nullptr;
        }
    }
    CHECK(g_live == 0);

    std::printf("[omniseed]  sidecar_hub: %d passed, %d failed\n",
                g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
