// =============================================================================
//  OmniSeed — tests/test_agent_modules.cpp
//
//  MILESTONE 6 — HARD RULE 5: "every module gets tests".
//
//  Four modules in OMNISEED_CORE_SOURCES had ZERO coverage, checked by CLASS
//  NAME rather than filename (a filename grep is a proxy and it lies — see
//  §33): GrammarDecoder, ComputeThrottle, SelfImprovement, FocalCodec. All four
//  are pure, deterministic and model-free, so every test here is UNGATED and
//  runs in CI.
//
//  Writing the tests found FOUR real defects. Two are PINNED (the code is left
//  alone and the behaviour is asserted so it cannot drift silently); two are
//  FIXED (no behavioural contract to preserve).
//
//    FIXED  SelfImprovement::save() wrote a uint32 magic (0x54524953) while
//           load() compared the STRING "SRIT". Little-endian that constant is
//           'S','I','R','T', so the two NEVER agreed: load() rejected every
//           file save() wrote and the trace cache has never persisted. Silent,
//           because an unreadable cache looks like an empty one. C9 pins it.
//    FIXED  SelfImprovement::load() read every length straight out of the file
//           and never checked it against the mapping size — a truncated
//           state/improve.bin walked off the end of the mmap. C7 pins the fix.
//    FIXED  SelfImprovement::task_key() did not trim LEADING whitespace, so
//           "  read a file" and "read a file" were two different cache keys and
//           the exact-match replay lookup missed. C1 pins it.
//    PINNED FocalCodec::encode() computes a 4-bit bucket per band and then
//           keeps only its LOW BIT (`q & 1u`), discarding 3 of every 4 bits.
//           D6 proves the loss is observable.
//    PINNED ComputeThrottle::classify() matches its Fast keywords by SUBSTRING,
//           so "hi" fires inside "which" and "no" inside "nothing". B6 pins the
//           misclassification of a real trading question.
//
//  Fully offline: no network, no model weights, no Python.
// =============================================================================
#include "omniseed/agent/agent.h"
#include "omniseed/audio/audio.h"
#include "omniseed/core/platform.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using namespace omniseed;

static int g_passed = 0;
static int g_failed = 0;
static int g_skipped = 0;
static std::string g_current;

#define TEST(name) g_current = (name); platform::log_info("TEST  %s", name)
#define CHECK(cond)                                                        \
    do {                                                                   \
        if (cond) { ++g_passed; }                                          \
        else {                                                             \
            ++g_failed;                                                    \
            platform::log_error("FAIL  %s  (line %d): %s",                 \
                                g_current.c_str(), __LINE__, #cond);       \
        }                                                                  \
    } while (0)
#define SKIP(msg)                                                          \
    do {                                                                   \
        ++g_skipped;                                                       \
        platform::log_info("SKIP  %s  (%s)", g_current.c_str(), msg);      \
    } while (0)

namespace {

std::string tmp(const std::string& leaf) { return "build/" + leaf; }

bool write_bytes(const std::string& path, const std::vector<uint8_t>& b) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(b.data()),
            static_cast<std::streamsize>(b.size()));
    return static_cast<bool>(f);
}

// A 16 kHz mono tone. `freq` in Hz; `frames` of 1280 samples (80 ms each).
PcmAudio tone(double freq, int32_t frames, double amp = 0.5) {
    PcmAudio a;
    a.sample_rate = 16000;
    const int32_t frame_len = 16000 * 80 / 1000;      // 1280
    const int32_t total = frame_len * frames;
    a.samples.resize(static_cast<size_t>(total));
    for (int32_t t = 0; t < total; ++t) {
        a.samples[static_cast<size_t>(t)] = static_cast<float>(
            amp * std::sin(2.0 * 3.14159265358979323846 * freq * t / 16000.0));
    }
    return a;
}

PcmAudio silence(int32_t frames) {
    PcmAudio a;
    a.sample_rate = 16000;
    a.samples.assign(static_cast<size_t>(1280 * frames), 0.0f);
    return a;
}

} // namespace

// =============================================================================
// PART A — GrammarDecoder (streamed JSON validity)
// =============================================================================
static void part_a_grammar() {
    TEST("A1: reset() clears buffer, depth and flags");
    {
        GrammarDecoder g;
        g.feed("{\"a\":");
        CHECK(!g.text().empty());
        g.reset();
        CHECK(g.text().empty());
        CHECK(!g.complete());
        CHECK(!g.failed());
    }

    TEST("A2: a valid JSON object fed in pieces completes");
    {
        GrammarDecoder g;
        CHECK(g.accepts("{"));
        g.feed("{");
        CHECK(!g.complete());
        CHECK(g.accepts("\"tool\":"));
        g.feed("\"tool\":");
        CHECK(g.accepts("\"read_file\""));
        g.feed("\"read_file\"");
        CHECK(g.accepts("}"));
        g.feed("}");
        CHECK(g.complete());
        CHECK(!g.failed());
        CHECK(g.text() == "{\"tool\":\"read_file\"}");
    }

    TEST("A3: an unbalanced close is rejected");
    {
        GrammarDecoder g;
        CHECK(!g.accepts("}"));          // depth would go negative
        g.feed("}");
        CHECK(g.failed());
        // once failed, it stays failed and swallows further input
        const std::string before = g.text();
        g.feed("{");
        CHECK(g.failed());
        CHECK(g.text() == before);
        CHECK(!g.accepts("{"));
    }

    TEST("A4: braces inside a string do not change depth");
    {
        GrammarDecoder g;
        g.feed("{\"s\":\"}}}{{{\"");
        CHECK(!g.failed());
        CHECK(!g.complete());            // still open: the outer { is unclosed
        CHECK(g.accepts("}"));
        g.feed("}");
        CHECK(g.complete());
    }

    TEST("A5: an escaped quote does not end the string");
    {
        GrammarDecoder g;
        g.feed("{\"s\":\"a\\\"b\"}");
        CHECK(!g.failed());
        CHECK(g.complete());
        CHECK(g.text() == "{\"s\":\"a\\\"b\"}");
    }

    TEST("A6: a bare string never reports complete");
    {
        GrammarDecoder g;
        g.feed("\"hello\"");
        CHECK(!g.failed());
        // complete_ requires the buffer to start with '{' or '['
        CHECK(!g.complete());
    }

    TEST("A7: nested arrays and objects are tracked");
    {
        GrammarDecoder g;
        g.feed("{\"a\":[1,{\"b\":[2]}]");
        CHECK(!g.complete());
        CHECK(g.accepts("}"));
        g.feed("}");
        CHECK(g.complete());
    }

    // -------------------------------------------------------------------------
    // PINNED DEFECT. The contract is "would appending `text` keep the stream a
    // valid JSON prefix?". After a complete top-level object the answer must be
    // NO — "{}x" and "{} {}" are not JSON prefixes. It answers YES, because
    // accepts() never consults complete_ and feed()'s recompute re-derives
    // complete_ from the buffer's first character alone.
    //
    // FIXED §49: accepts() now returns false after a complete object, and
    // feed() marks the decoder failed if content follows a closed object.
    // -------------------------------------------------------------------------
    TEST("A8 (FIXED §49): trailing content after a complete object is rejected");
    {
        GrammarDecoder g;
        g.feed("{}");
        CHECK(g.complete());
        CHECK(!g.accepts("x"));          // fixed: trailing content rejected
        g.feed("x");
        CHECK(g.failed());               // fixed: decoder is now failed
    }
}

// =============================================================================
// PART B — ComputeThrottle (adaptive compute)
// =============================================================================
static void part_b_throttle() {
    ComputeThrottle t;

    TEST("B1: deep keywords select Deep");
    {
        CHECK(t.classify("plan a route to the office") == ComputeThrottle::Level::Deep);
        CHECK(t.classify("analyze this data")          == ComputeThrottle::Level::Deep);
        CHECK(t.classify("refactor the parser")        == ComputeThrottle::Level::Deep);
        CHECK(t.classify("how do I install it")        == ComputeThrottle::Level::Deep);
    }

    TEST("B2: length drives Fast/Balanced/Deep when no keyword matches");
    {
        // < 24 chars and no keyword
        CHECK(t.classify("gimme a number") == ComputeThrottle::Level::Fast);
        // 24..160 chars
        const std::string mid = "please summarise the quarterly figures for me";
        CHECK(mid.size() >= 24 && mid.size() <= 160);
        CHECK(t.classify(mid) == ComputeThrottle::Level::Balanced);
        // > 160 chars
        std::string long_in(200, 'z');
        CHECK(t.classify(long_in) == ComputeThrottle::Level::Deep);
    }

    TEST("B3: budgets are ordered and Fast skips thinking entirely");
    {
        const int32_t nf = t.max_new_tokens(ComputeThrottle::Level::Fast);
        const int32_t nb = t.max_new_tokens(ComputeThrottle::Level::Balanced);
        const int32_t nd = t.max_new_tokens(ComputeThrottle::Level::Deep);
        CHECK(nf < nb && nb < nd);
        CHECK(t.max_think_tokens(ComputeThrottle::Level::Fast) == 0);
        CHECK(t.max_think_tokens(ComputeThrottle::Level::Balanced) <
              t.max_think_tokens(ComputeThrottle::Level::Deep));
        CHECK(t.max_tool_turns(ComputeThrottle::Level::Fast) <
              t.max_tool_turns(ComputeThrottle::Level::Deep));
    }

    TEST("B4: level names");
    {
        CHECK(std::string(t.name(ComputeThrottle::Level::Fast))     == "fast");
        CHECK(std::string(t.name(ComputeThrottle::Level::Balanced)) == "balanced");
        CHECK(std::string(t.name(ComputeThrottle::Level::Deep))     == "deep");
    }

    TEST("B5: rss_zone() honours the configured watermarks");
    {
        ComputeThrottle a;
        a.rss_limits(0, 0);                        // everything is >= 0
        CHECK(a.rss_zone() == 2);                  // hard
        ComputeThrottle b;
        b.rss_limits(1u << 30, 1u << 30);          // 1 TB: unreachable
        CHECK(b.rss_zone() == 0);                  // ok
    }

    // -------------------------------------------------------------------------
    // PINNED DEFECT. The Fast keyword list is matched with `find()`, so the
    // 2-character entries fire inside unrelated words: "hi" inside "which",
    // "no" inside "nothing"/"know", "ok" inside "book". A real trading question
    // is therefore classified Fast and gets the SHALLOWEST budget.
    //
    // FIXED §49: keyword matching now uses word-boundary checks, so "hi"
    // inside "which" and "no" inside "nothing" no longer fire.
    // -------------------------------------------------------------------------
    TEST("B6 (FIXED §49): word-boundary matching prevents substring false positives");
    {
        // "which" contains "hi" as a substring but not as a word.
        const std::string q = "which stock should i buy today";
        CHECK(q.size() >= 24);                                  // not Fast by length
        CHECK(q.find("hi") != std::string::npos);               // substring present
        CHECK(t.classify(q) == ComputeThrottle::Level::Balanced); // fixed: not Fast
        // "nothing" contains "no" as a substring but not as a word.
        CHECK(t.classify("i have nothing to add here") == ComputeThrottle::Level::Balanced);
        // Real Fast keywords still work at word boundaries.
        CHECK(t.classify("hi there") == ComputeThrottle::Level::Fast);
        CHECK(t.classify("no") == ComputeThrottle::Level::Fast);
    }
}

// =============================================================================
// PART C — SelfImprovement (trace cache + dream consolidation)
// =============================================================================
static void part_c_self_improvement() {
    TEST("C1: task_key lowercases, collapses whitespace and trims");
    {
        CHECK(SelfImprovement::task_key("  Read   A   File  ") == "read a file");
        CHECK(SelfImprovement::task_key("READ A FILE") == "read a file");
        CHECK(SelfImprovement::task_key("read a file") == "read a file");
        CHECK(SelfImprovement::task_key("a\tb\nc") == "a b c");
        CHECK(SelfImprovement::task_key("") == "");
    }

    TEST("C2: a trustworthy path is replayed");
    {
        SelfImprovement s;
        const std::string k = SelfImprovement::task_key("open the file");
        s.record(k, {"read_file", "parse"}, true, 12.0);
        CHECK(s.size() == 1);
        std::vector<std::string> steps;
        CHECK(s.find_replay(k, steps));
        CHECK(steps.size() == 2);
        CHECK(steps[0] == "read_file");
    }

    TEST("C3: an unknown key is never replayed");
    {
        SelfImprovement s;
        s.record("known", {"a"}, true, 1.0);
        std::vector<std::string> steps;
        CHECK(!s.find_replay("never seen", steps));
        CHECK(steps.empty());
    }

    TEST("C4: replay is refused below the success-rate threshold");
    {
        SelfImprovement::Config cfg;
        cfg.replay_min_success_rate = 0.8;
        SelfImprovement s(cfg);
        const std::string k = "flaky";
        s.record(k, {"x"}, true, 1.0);          // 1 success
        for (int i = 0; i < 4; ++i) s.record(k, {"x"}, false, 1.0);  // 4 fails
        // rate is 1/5 = 0.2 < 0.8
        std::vector<std::string> steps;
        CHECK(!s.find_replay(k, steps));
    }

    TEST("C5: a success-only trace survives dream(), a failure-only trace does not");
    {
        SelfImprovement s;
        s.record("good", {"a", "b"}, true, 5.0);
        for (int i = 0; i < 3; ++i) s.record("bad", {"c"}, false, 5.0);
        CHECK(s.size() == 2);
        s.dream();
        CHECK(s.size() == 1);
        std::vector<std::string> steps;
        CHECK(s.find_replay("good", steps));
        CHECK(!s.find_replay("bad", steps));
    }

    TEST("C6: save/load round-trips traces");
    {
        const std::string path = tmp("test_self_improve.bin");
        SelfImprovement a;
        a.record("alpha", {"one", "two"}, true, 3.5);
        a.record("beta",  {"three"},       true, 7.25);
        CHECK(a.save(path));

        SelfImprovement b;
        CHECK(b.load(path));
        CHECK(b.size() == 2);
        std::vector<std::string> steps;
        CHECK(b.find_replay("alpha", steps));
        CHECK(steps.size() == 2);
        CHECK(steps[1] == "two");
        CHECK(b.find_replay("beta", steps));
        CHECK(steps.size() == 1);
        std::remove(path.c_str());
    }

    // -------------------------------------------------------------------------
    // C7 pins the FIX. Before it, load() read klen/ns/sl straight from the file
    // and walked the mapping past its end. These three files are all malformed
    // in a different way and every one must be REJECTED without crashing.
    // -------------------------------------------------------------------------
    TEST("C7: a malformed file is rejected, not walked off the end");
    {
        // (a) valid header, n = 1, but no trace body at all
        {
            const std::string p = tmp("test_si_truncated.bin");
            std::vector<uint8_t> b = {'S', 'R', 'I', 'T'};
            const uint32_t ver = 1, n = 1;
            b.insert(b.end(), reinterpret_cast<const uint8_t*>(&ver),
                     reinterpret_cast<const uint8_t*>(&ver) + 4);
            b.insert(b.end(), reinterpret_cast<const uint8_t*>(&n),
                     reinterpret_cast<const uint8_t*>(&n) + 4);
            CHECK(write_bytes(p, b));
            SelfImprovement s;
            CHECK(!s.load(p));
            CHECK(s.size() == 0);
            std::remove(p.c_str());
        }
        // (b) a key length that runs past the end of the file
        {
            const std::string p = tmp("test_si_overlong.bin");
            std::vector<uint8_t> b = {'S', 'R', 'I', 'T'};
            const uint32_t ver = 1, n = 1, klen = 100000;
            for (uint32_t v : {ver, n, klen}) {
                b.insert(b.end(), reinterpret_cast<const uint8_t*>(&v),
                         reinterpret_cast<const uint8_t*>(&v) + 4);
            }
            b.push_back('a');
            CHECK(write_bytes(p, b));
            SelfImprovement s;
            CHECK(!s.load(p));
            std::remove(p.c_str());
        }
        // (c) an absurd trace count must not be trusted for reserve()
        {
            const std::string p = tmp("test_si_absurd.bin");
            std::vector<uint8_t> b = {'S', 'R', 'I', 'T'};
            const uint32_t ver = 1, n = 4000000000u;
            for (uint32_t v : {ver, n}) {
                b.insert(b.end(), reinterpret_cast<const uint8_t*>(&v),
                         reinterpret_cast<const uint8_t*>(&v) + 4);
            }
            CHECK(write_bytes(p, b));
            SelfImprovement s;
            CHECK(!s.load(p));
            std::remove(p.c_str());
        }
        // (d) wrong magic and wrong version are refused
        {
            const std::string p = tmp("test_si_badmagic.bin");
            CHECK(write_bytes(p, {'X', 'X', 'X', 'X', 1, 0, 0, 0, 0, 0, 0, 0}));
            SelfImprovement s;
            CHECK(!s.load(p));
            std::remove(p.c_str());
        }
        // (e) a failed load leaves the PREVIOUS traces intact
        {
            SelfImprovement s;
            s.record("keepme", {"a"}, true, 1.0);
            const std::string p = tmp("test_si_bad2.bin");
            CHECK(write_bytes(p, {'S', 'R', 'I', 'T', 9, 0, 0, 0, 0, 0, 0, 0}));
            CHECK(!s.load(p));                 // version 9
            CHECK(s.size() == 1);              // not clobbered
            std::vector<std::string> steps;
            CHECK(s.find_replay("keepme", steps));
            std::remove(p.c_str());
        }
    }

    TEST("C8: max_traces bounds the cache");
    {
        SelfImprovement::Config cfg;
        cfg.max_traces = 4;
        SelfImprovement s(cfg);
        for (int i = 0; i < 8; ++i)
            s.record("k" + std::to_string(i), {"a"}, true, 1.0);
        CHECK(s.size() == 4);
    }

    // -------------------------------------------------------------------------
    // C9 pins the FIX for the worst defect this suite found: save() wrote a
    // uint32 magic (0x54524953) and load() compared against the STRING "SRIT".
    // Little-endian, 0x54524953 is 'S','I','R','T' — so the two never agreed,
    // load() rejected every file save() wrote, and the trace cache has NEVER
    // persisted. The failure was silent: an unreadable cache looks exactly like
    // an empty one, and the CLI's imp.load("./state/improve.bin") simply always
    // returned false. The four bytes are asserted here so they cannot drift.
    // -------------------------------------------------------------------------
    TEST("C9: the on-disk magic is the bytes 'S','I','R','T' (save/load agree)");
    {
        const std::string path = tmp("test_si_magic.bin");
        SelfImprovement s;
        s.record("x", {"y"}, true, 1.0);
        CHECK(s.save(path));

        std::ifstream f(path, std::ios::binary);
        CHECK(static_cast<bool>(f));
        char m[4] = {0, 0, 0, 0};
        f.read(m, 4);
        CHECK(m[0] == 'S');
        CHECK(m[1] == 'I');
        CHECK(m[2] == 'R');
        CHECK(m[3] == 'T');
        f.close();
        // and the file must actually round-trip, which is the property the
        // mismatch silently broke
        SelfImprovement r;
        CHECK(r.load(path));
        CHECK(r.size() == 1);
        std::remove(path.c_str());
    }

    // =========================================================================
    // §47 — the dream JOINT. C5 above is the trace-only invariant and is left
    // EXACTLY as it was: a caller with no other subsystem in play must get the
    // behaviour it always got. C10-C13 cover what the joint adds.
    // =========================================================================

    TEST("C10: dream() reinforces a crystal recalled since the last pass, and decays one that was not");
    {
        Tokenizer tok;
        CHECK(tok.build_minimal());

        const std::string used_text    = "deploy window tuesday rollback plan";
        const std::string ignored_text = "lunch menu vegetarian option";

        MemoryCrystals mc;
        uint64_t used = 0, ignored = 0;
        CHECK(mc.crystallize(tok.encode(used_text), tok, 0, -1.0f, &used));
        CHECK(mc.crystallize(tok.encode(ignored_text), tok, 0, -1.0f, &ignored));
        CHECK(mc.size() == 2);
        CHECK(used != 0);
        CHECK(ignored != 0);
        CHECK(used != ignored);

        // The predicate, named and checked directly. Strictly greater, and an
        // unknown clock (0) claims NOTHING was used.
        {
            MemoryCrystal c;
            c.last_access_token = 100;
            CHECK(crystal_used_since(c, 50));
            CHECK(!crystal_used_since(c, 100));
            CHECK(!crystal_used_since(c, 0));
        }

        const uint64_t previous = 1000000;   // the clock at the last dream

        // ONE crystal is recalled. Retrieval stamps last_access_token — that is
        // the memory layer's own documented contract ("retrieval reinforces"),
        // and it is the ONLY thing that distinguishes the two crystals here.
        const std::vector<MemoryCrystal> got =
            mc.retrieve(tok.encode(used_text), 1, previous + 5);
        CHECK(got.size() == 1);
        CHECK(got[0].id == used);            // the query IS this crystal's text
        const MemoryCrystal* cu = mc.find(used);
        const MemoryCrystal* ci = mc.find(ignored);
        CHECK(cu != nullptr);
        CHECK(ci != nullptr);
        CHECK(cu->last_access_token > previous);
        CHECK(ci->last_access_token <= previous);

        SelfImprovement imp;
        DreamContext ctx;
        ctx.crystals       = &mc;
        ctx.previous_token = previous;
        ctx.now_token      = previous + 5000000;   // ~50 days later in the crystal clock
        const DreamReport r = imp.dream(ctx);

        CHECK(r.crystals_before == 2);
        CHECK(r.crystals_reinforced == 1);         // exactly the recalled one
        CHECK(r.crystals_decayed == 1);            // exactly the ignored one
        CHECK(r.crystals_after == 1);
        CHECK(mc.find(used) != nullptr);           // reinforced => survived
        CHECK(mc.find(ignored) == nullptr);        // untouched => decayed away
    }

    TEST("C11: dream_log.json is written with a stable, asserted schema");
    {
        DreamReport r;
        r.traces_before = 3;
        r.traces_after  = 2;
        r.traces_pruned = 1;
        r.crystals_before = 4;
        r.crystals_after  = 3;
        r.crystals_reinforced = 2;
        r.crystals_decayed    = 1;
        r.journal_present = true;
        r.journal_predictions = 7;
        r.journal_confirmed_keys = 1;
        r.stream_source = "journal";
        r.stream_events = 7;
        r.stream_committed = 2;
        r.stream_flips = 1;
        r.previous_token = 10;
        r.now_token = 20;

        const std::string path = tmp("test_dream_log.json");
        CHECK(write_dream_log(r, path));

        std::ifstream f(path);
        CHECK(static_cast<bool>(f));
        const std::string j((std::istreambuf_iterator<char>(f)),
                            std::istreambuf_iterator<char>());
        f.close();

        // Every section and every counter the log promises. A key that is
        // renamed or dropped must fail HERE, not silently disappear from the
        // nightly record.
        CHECK(j.find("\"version\": 1") != std::string::npos);
        CHECK(j.find("\"generated_ms\": ") != std::string::npos);
        CHECK(j.find("\"previous_token\": 10") != std::string::npos);
        CHECK(j.find("\"now_token\": 20") != std::string::npos);
        CHECK(j.find("\"traces\": {") != std::string::npos);
        CHECK(j.find("\"before\": 3") != std::string::npos);
        CHECK(j.find("\"after\": 2") != std::string::npos);
        CHECK(j.find("\"pruned\": 1") != std::string::npos);
        CHECK(j.find("\"pruned_by_journal\": 0") != std::string::npos);
        CHECK(j.find("\"reinforced_by_journal\": 0") != std::string::npos);
        CHECK(j.find("\"decayed\": 0") != std::string::npos);
        CHECK(j.find("\"faded\": 0") != std::string::npos);
        CHECK(j.find("\"dropped_unsuccessful\": 0") != std::string::npos);
        CHECK(j.find("\"crystals\": {") != std::string::npos);
        CHECK(j.find("\"before\": 4") != std::string::npos);
        CHECK(j.find("\"after\": 3") != std::string::npos);
        CHECK(j.find("\"reinforced\": 2") != std::string::npos);
        CHECK(j.find("\"decayed\": 1") != std::string::npos);
        CHECK(j.find("\"journal\": {") != std::string::npos);
        CHECK(j.find("\"present\": true") != std::string::npos);
        CHECK(j.find("\"predictions\": 7") != std::string::npos);
        CHECK(j.find("\"confirmed_keys\": 1") != std::string::npos);
        CHECK(j.find("\"stream\": {") != std::string::npos);
        CHECK(j.find("\"source\": \"journal\"") != std::string::npos);
        CHECK(j.find("\"committed\": 2") != std::string::npos);
        CHECK(j.find("\"flips\": 1") != std::string::npos);
        CHECK(j.find("\"soul\": {") != std::string::npos);
        CHECK(j.find("\"present\": false") != std::string::npos);

        // It is actually JSON, not a printf of the same numbers.
        CHECK(j.size() > 400);
        CHECK(j.front() == '{');
        CHECK(j.back() == '\n');
        int depth = 0;
        bool balanced = true;
        for (const char c : j) {
            if (c == '{') ++depth;
            else if (c == '}') { --depth; if (depth < 0) balanced = false; }
        }
        CHECK(balanced);
        CHECK(depth == 0);

        std::remove(path.c_str());
    }

    TEST("C12: the journal outranks success_count — a CORRECTED key is pruned, a CONFIRMED key refreshed");
    {
        SelfImprovement imp;
        imp.record("good path", {"a"}, true, 1.0);
        imp.record("bad path",  {"b"}, true, 1.0);
        imp.record("untouched", {"c"}, true, 1.0);
        CHECK(imp.size() == 3);

        DecisionJournal j;
        JournalEntry e1;
        e1.task_key  = "bad path";
        e1.predicted = DecisionAction::BUY;
        const int64_t id1 = j.record(e1);
        CHECK(id1 > 0);
        CHECK(j.resolve_kind(id1, OutcomeKind::Corrected));

        JournalEntry e2;
        e2.task_key  = "good path";
        e2.predicted = DecisionAction::BUY;
        const int64_t id2 = j.record(e2);
        CHECK(id2 > 0);
        CHECK(j.resolve_kind(id2, OutcomeKind::Confirmed));

        DreamContext ctx;
        ctx.journal = &j;
        const DreamReport r = imp.dream(ctx);

        CHECK(r.journal_present);
        CHECK(r.journal_predictions == 2);
        CHECK(r.journal_corrected_keys == 1);
        CHECK(r.journal_confirmed_keys == 1);
        CHECK(r.traces_pruned_by_journal == 1);
        CHECK(r.traces_reinforced_by_journal == 1);
        CHECK(r.traces_before == 3);
        CHECK(r.traces_after == 2);

        // The corrected key is gone even though its own record is spotless —
        // that is the whole point of letting the journal outrank it.
        std::vector<std::string> steps;
        CHECK(!imp.find_replay("bad path", steps));
        CHECK(imp.find_replay("good path", steps));
        CHECK(imp.find_replay("untouched", steps));
    }

    TEST("C13: the trace accounting closes exactly, and an empty context claims nothing");
    {
        SelfImprovement imp;
        imp.record("a", {"x"}, true, 1.0);
        imp.record("b", {"y"}, true, 1.0);
        imp.record("c", {"z"}, false, 1.0);            // 1 failure: kept by no rule
        for (int i = 0; i < 3; ++i) imp.record("d", {"w"}, false, 1.0);  // failure-only

        DreamContext empty;                            // no crystals, no journal, no events
        const DreamReport r = imp.dream(empty);

        CHECK(r.traces_before == 4);
        CHECK(r.traces_pruned == 1);                   // "d"
        CHECK(r.traces_dropped_unsuccessful == 1);     // "c" — the silent drop, named
        CHECK(r.traces_after == 2);                    // "a", "b"
        // The identity. The pre-§47 pass could not state it, because one of its
        // drop paths had no counter at all.
        CHECK(r.traces_before == r.traces_after + r.traces_pruned +
                                 r.traces_pruned_by_journal + r.traces_faded +
                                 r.traces_dropped_unsuccessful);

        // An empty context must not invent activity in the other subsystems.
        CHECK(!r.journal_present);
        CHECK(r.journal_predictions == 0);
        CHECK(r.journal_verdicts == 0);
        CHECK(r.stream_source == "none");
        CHECK(r.stream_events == 0);
        CHECK(r.stream_committed == 0);
        CHECK(r.crystals_before == 0);
        CHECK(r.crystals_after == 0);
        CHECK(r.crystals_reinforced == 0);
        CHECK(r.crystals_decayed == 0);
        CHECK(!r.soul_present);
        CHECK(r.soul_crystals == 0);
        CHECK(r.soul_clock == 0);
    }
}

// =============================================================================
// PART D — FocalCodec (semantic audio tokenization)
// =============================================================================
static void part_d_focal_codec() {
    FocalCodec codec;
    CHECK(codec.config().frame_ms == 80);
    CHECK(codec.config().sample_rate == 16000);

    TEST("D1: frame_features reports one row per 80 ms frame");
    {
        const PcmAudio a = tone(440.0, 3);
        Tensor frames;
        CHECK(codec.frame_features(a, frames));
        CHECK(frames.dim(0) == 3);
        CHECK(frames.dim(1) == 24);            // 24 log-spaced bands
        bool finite = true;
        for (int64_t i = 0; i < frames.numel(); ++i)
            if (!std::isfinite(frames.f32()[i])) finite = false;
        CHECK(finite);
    }

    TEST("D2: encode is deterministic and one code per frame");
    {
        const PcmAudio a = tone(440.0, 3);
        std::vector<int32_t> c1, c2;
        CHECK(codec.encode(a, c1));
        CHECK(codec.encode(a, c2));
        CHECK(c1.size() == 3);
        CHECK(c1 == c2);                       // byte-stable for identical input
    }

    TEST("D3: different content gives different codes");
    {
        std::vector<int32_t> a, b;
        CHECK(codec.encode(tone(200.0, 4), a));
        CHECK(codec.encode(tone(3000.0, 4), b));
        CHECK(a != b);
    }

    TEST("D4: silence produces a deterministic code and every code is inside the codebook");
    {
        // With the FNV-1a codec, silence (all bands at log-floor -> q=0) hashes
        // to a fixed non-zero value.  The important invariant is determinism and
        // range, not "must be zero".
        std::vector<int32_t> s;
        CHECK(codec.encode(silence(2), s));
        CHECK(s.size() == 2);
        // Both frames must be identical (same silence content) and in range.
        CHECK(s[0] == s[1]);
        for (int32_t c : s) {
            CHECK(c >= 0);
            CHECK(c < codec.config().codebook_size);
        }

        std::vector<int32_t> loud;
        CHECK(codec.encode(tone(440.0, 2, 0.9), loud));
        for (int32_t c : loud) {
            CHECK(c >= 0);
            CHECK(c < codec.config().codebook_size);
        }
        // Silence and tone must differ.
        CHECK(s[0] != loud[0]);
    }

    TEST("D5: audio shorter than one frame, or invalid, is refused");
    {
        PcmAudio tiny;
        tiny.sample_rate = 16000;
        tiny.samples.assign(100, 0.0f);        // < 1280
        Tensor frames;
        CHECK(!codec.frame_features(tiny, frames));
        std::vector<int32_t> codes;
        CHECK(!codec.encode(tiny, codes));

        PcmAudio empty;                        // sample_rate set, no samples
        CHECK(!codec.encode(empty, codes));
    }

    // -------------------------------------------------------------------------
    // D6: FNV-1a codec uses all 4 bits of each band's quantized value.
    //
    // The old codec kept only the low bit of each 4-bit bucket (q & 1u),
    // discarding 3 of 4 bits per band.  Fixed in §49: the encode loop now
    // runs FNV-1a over the full 4-bit q values, so all 16 levels contribute
    // to the hash.  This test verifies the new behaviour:
    //   - the emitted code matches the FNV-1a formula, not the old bit-pack
    //   - the FNV-1a code and the old bit-pack code differ (confirming the fix)
    // -------------------------------------------------------------------------
    TEST("D6 (FIXED §49): FNV-1a codec uses all 4 bits per band");
    {
        const PcmAudio a = tone(440.0, 1, 0.8);
        Tensor frames;
        CHECK(codec.frame_features(a, frames));
        const float* row = frames.f32();

        const int32_t n_bands = 24;
        uint32_t fnv_code = 2166136261u;
        uint32_t old_bitpack = 0;
        for (int32_t b = 0; b < n_bands; ++b) {
            const float v = std::min(2.0f, std::max(-8.0f, row[b]));
            const uint32_t q =
                static_cast<uint32_t>((v + 8.0f) / 10.0f * 15.0f) & 0xFu;
            fnv_code  ^= q;
            fnv_code  *= 16777619u;
            old_bitpack |= (q & 1u) << (b % 32);   // old broken formula
        }

        std::vector<int32_t> codes;
        CHECK(codec.encode(a, codes));
        CHECK(codes.size() == 1);
        // Must match the FNV-1a formula.
        CHECK(codes[0] == static_cast<int32_t>(
              fnv_code % static_cast<uint32_t>(codec.config().codebook_size)));
        // Must NOT match the old bit-pack formula (confirms the fix is active).
        const int32_t old_code = static_cast<int32_t>(
            old_bitpack % static_cast<uint32_t>(codec.config().codebook_size));
        CHECK(codes[0] != old_code);
        platform::log_info(
            "  FocalCodec frame 0: FNV-1a code %u, old bit-pack code %u "
            "(differ => fix confirmed)",
            static_cast<uint32_t>(codes[0]), static_cast<uint32_t>(old_code));
    }
}

// =============================================================================
int main() {
    platform::set_quiet(false);
    platform::log_info("=== test_agent_modules (MILESTONE 6) ===");
    part_a_grammar();
    part_b_throttle();
    part_c_self_improvement();
    part_d_focal_codec();

    platform::log_info("RESULT: %d passed, %d failed, %d skipped",
                       g_passed, g_failed, g_skipped);
    return g_failed == 0 ? 0 : 1;
}
