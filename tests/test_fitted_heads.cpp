// =============================================================================
//  OmniSeed — tests/test_fitted_heads.cpp
//
//  T112 — the runtime consumer for fitted ClassificationHead blobs
//  (include/omniseed/runtime/fitted_heads.h). FittedHeads discovers the blobs
//  tools/train_heads.py wrote (language_head.bin / vision_head.bin /
//  audio_head.bin / trading_regime_head.bin) in a directory and answers
//  classify(set, h[E]) from them — or REFUSES. Every part of the contract is
//  fail-closed: a missing dir, an empty dir, a truncated blob, an unmapped
//  set or a hidden-width mismatch returns the "<error: ...>" / empty-top_k
//  refusal. Never a throw, never a fabricated distribution, never a default
//  to class 0.
//
//  PART A — discovery + happy path, SYNTHETIC blobs. The fixture is built
//  here through ClassificationHead's own save() (the same writer train_heads'
//  format spec mirrors), so the test is self-contained and the argmax is
//  known by construction: each class row is a scaled unit vector, so the
//  matching hidden unit vector MUST win.
//      A1  load() of a dir with two fitted bins returns true
//      A2  only actually-fitted sets are callable (available())
//      A3  vision.scene on the crafted hidden -> argmax = the crafted class
//      A4  language.intent on its own crafted hidden -> its crafted class
//      A5  a set whose FAMILY has no bin in the dir (audio.*) refuses
//      A6  a set the loaded blob does not carry refuses (trading.regime)
//
//  PART B — fail-closed load paths.
//      B1  nonexistent dir: load() false, classify() refusal, no throw
//      B2  empty dir: load() false, classify() refusal
//      B3  truncated blob: load() false, classify() refusal (the half-read
//          projection can never classify — ClassificationHead::load leaves
//          the head NOT READY, FittedHeads never caches it)
//      B4  the failure is sticky per bin: a second classify() still refuses
//          (no disk-retry storm) and error() names the blob
//
//  PART C — the off-parity guard. A blob fitted at E=8 fed a 7-wide caller
//  vector (or the real E=768 blob fed 512) must REFUSE: the matvec would
//  otherwise read past the caller's buffer or invent a distribution from
//  garbage. Gated on models/heads/vision_head.bin for the real-blob leg;
//  skips loudly when the committed blob is absent (clean checkout).
//
//  PART D — set_heads_dir() drops every cache (an old-dir blob must not
//  answer for a new dir).
//
//  PART E — T113c regression: vision bins are PER SET, not one family bin.
//  bin_for_set() used to map every "vision.*" at vision_head.bin, which
//  carries only vision.scene — so vision.anomaly (and now vision.candlestick)
//  refused against a blob that exists on disk.
//      E1  bin_for_set unit checks: scene -> vision_head.bin, anomaly ->
//          vision_anomaly_head.bin, candlestick -> vision_candlestick_head.bin,
//          unmapped vision.* -> "" (refuse, no guess)
//      E2  real vision_anomaly_head.bin (gated): "vision.anomaly" ANSWERS at
//          E=768; "vision.scene" against the same blob dir refuses
//      E3  canonical-list sync: every vision.* set with an explicit bin in
//          bin_for_set() must exist in default_domain_label_sets() (the same
//          list tools/train_heads.py parses), with vision.candlestick's labels
//          pinned to exactly {"up","down"}; the reverse direction too — a
//          canonical vision set with no mapped bin fails the test.
// =============================================================================
#include "omniseed/classification_head.h"
#include "omniseed/core/platform.h"
#include "omniseed/runtime/fitted_heads.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

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

// The refusal fingerprint every failure path must carry: domain in the
// repo's "<error: ...>" form (no registered set name can start with '<')
// and an EMPTY top_k — unambiguous "the head did not answer".
bool is_refusal(const ClassificationResult& r) {
    return r.top_k.empty() && !r.domain.empty() && r.domain[0] == '<';
}

bool file_bytes(const std::string& p, std::vector<uint8_t>& out) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;
    out.assign(std::istreambuf_iterator<char>(f),
               std::istreambuf_iterator<char>());
    return true;
}

bool write_bytes(const std::string& p, const std::vector<uint8_t>& bytes) {
    std::ofstream f(p, std::ios::binary);
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
    return static_cast<bool>(f);
}

void make_dir(const std::string& p) {
#ifdef _WIN32
    _mkdir(p.c_str());
#else
    mkdir(p.c_str(), 0755);
#endif
}

// A fitted synthetic head: one label set whose rows are scaled unit vectors,
// so hidden = e_k makes class k the argmax BY CONSTRUCTION. Saved through
// ClassificationHead::save() — the exact blob format train_heads.py mirrors.
bool write_synthetic_blob(const std::string& path, int32_t E,
                          const std::string& set_name,
                          const std::vector<std::string>& labels) {
    ClassificationHead h;
    if (!h.init(E, 4242u)) return false;
    const int32_t si = h.add_label_set(set_name, labels);
    if (si < 0) return false;
    std::vector<float> row(static_cast<size_t>(E), 0.0f);
    for (size_t k = 0; k < labels.size(); ++k) {
        std::fill(row.begin(), row.end(), 0.0f);
        row[k] = 4.0f;  // strong margin, still finite logits at softmax
        h.set_label_row(si, static_cast<int32_t>(k), row.data(), 0.0f);
    }
    return h.save(path);
}

const char* kTmpDir = "build/omniseed_fitted_heads_dir";      // A/B/D
const char* kBadDir  = "build/omniseed_fitted_heads_dir_bad"; // B3
const char* kRealDir = "build/omniseed_fitted_heads_dir_real"; // C
const char* kAnomDir = "build/omniseed_fitted_heads_dir_anom"; // E2

} // namespace

// ---------------------------------------------------------------------------
int main() {
    make_dir(kTmpDir);
    make_dir(kBadDir);
    make_dir(kRealDir);
    make_dir(kAnomDir);

    // ===================== PART A — discovery + happy path ==================
    TEST("A1: load() discovers the fitted bins by filename convention");
    {
        CHECK(write_synthetic_blob(std::string(kTmpDir) + "/vision_head.bin",
                                   8, "vision.scene",
                                   {"cat", "dog", "bird"}));
        CHECK(write_synthetic_blob(std::string(kTmpDir) + "/language_head.bin",
                                   8, "language.intent",
                                   {"question", "statement"}));
        FittedHeads fh;
        CHECK(fh.load(kTmpDir) == true);
        CHECK(fh.loaded_count() == 2);
    }

    TEST("A2: only actually-fitted sets are callable (available)");
    {
        FittedHeads fh;
        CHECK(fh.load(kTmpDir));
        CHECK(fh.available("vision.scene") == true);
        CHECK(fh.available("language.intent") == true);
        // Families with no bin in this dir: refuse, do not guess.
        CHECK(fh.available("audio.emotion") == false);
        CHECK(fh.available("trading.regime") == false);
    }

    TEST("A3: vision.scene on the crafted hidden -> the crafted argmax");
    {
        FittedHeads fh;
        CHECK(fh.load(kTmpDir));
        const std::vector<float> hidden = {1.0f, 0.0f, 0.0f,
                                           0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        const ClassificationResult r =
            fh.classify(hidden.data(), "vision.scene", 3,
                        static_cast<int32_t>(hidden.size()));
        CHECK(!is_refusal(r));
        CHECK(r.domain == "vision.scene");
        CHECK(!r.top_k.empty());
        if (!r.top_k.empty()) {
            CHECK(r.top_k[0].label == "cat");
            CHECK(r.top_k[0].probability > 0.5f);
        }
    }

    TEST("A4: language.intent on its own crafted hidden -> its class");
    {
        FittedHeads fh;
        CHECK(fh.load(kTmpDir));
        // e_1 of the language blob -> "statement" (row 1 of its 2-label set).
        const std::vector<float> hidden = {0.0f, 1.0f, 0.0f,
                                           0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        const ClassificationResult r =
            fh.classify(hidden.data(), "language.intent", 2,
                        static_cast<int32_t>(hidden.size()));
        CHECK(!is_refusal(r));
        CHECK(!r.top_k.empty());
        if (!r.top_k.empty()) CHECK(r.top_k[0].label == "statement");
    }

    TEST("A5: a set whose family has no bin in the dir refuses");
    {
        FittedHeads fh;
        CHECK(fh.load(kTmpDir));
        const std::vector<float> hidden(8, 0.0f);
        const ClassificationResult r =
            fh.classify(hidden.data(), "audio.emotion", 3,
                        static_cast<int32_t>(hidden.size()));
        CHECK(is_refusal(r));
    }

    TEST("A6: a set the loaded blob does not carry refuses");
    {
        FittedHeads fh;
        CHECK(fh.load(kTmpDir));
        const std::vector<float> hidden(8, 0.0f);
        // trading.regime maps to trading_regime_head.bin, absent here; the
        // refusal must not be answered by some other bin.
        const ClassificationResult r =
            fh.classify(hidden.data(), "trading.regime", 3,
                        static_cast<int32_t>(hidden.size()));
        CHECK(is_refusal(r));
        // And a name that collides with NO family at all:
        CHECK(is_refusal(fh.classify(hidden.data(), "general.routing", 3,
                                     static_cast<int32_t>(hidden.size()))));
    }

    // ===================== PART B — fail-closed load paths ==================
    TEST("B1: nonexistent dir: load() false, classify() refusal, no throw");
    {
        FittedHeads fh;
        CHECK(fh.load("build/omniseed_no_such_dir") == false);
        CHECK(fh.loaded_count() == 0);
        const std::vector<float> hidden(8, 0.0f);
        const ClassificationResult r =
            fh.classify(hidden.data(), "vision.scene", 3,
                        static_cast<int32_t>(hidden.size()));
        CHECK(is_refusal(r));
    }

    TEST("B2: empty dir: load() false, classify() refusal");
    {
        make_dir("build/omniseed_fitted_heads_empty");
        FittedHeads fh;
        CHECK(fh.load("build/omniseed_fitted_heads_empty") == false);
        CHECK(fh.loaded_count() == 0);
        const std::vector<float> hidden(8, 0.0f);
        CHECK(is_refusal(fh.classify(hidden.data(), "vision.scene", 3,
                                     static_cast<int32_t>(hidden.size()))));
    }

    TEST("B3: truncated blob: load() false, classify() refusal");
    {
        std::vector<uint8_t> blob;
        CHECK(file_bytes(std::string(kTmpDir) + "/vision_head.bin", blob));
        CHECK(blob.size() > 64);
        blob.resize(blob.size() / 2);  // cut the projection in half
        CHECK(write_bytes(std::string(kBadDir) + "/vision_head.bin", blob));
        FittedHeads fh;
        CHECK(fh.load(kBadDir) == false);
        const std::vector<float> hidden(8, 0.0f);
        CHECK(is_refusal(fh.classify(hidden.data(), "vision.scene", 3,
                                     static_cast<int32_t>(hidden.size()))));
    }

    TEST("B4: the failure is sticky per bin and error() names the blob");
    {
        FittedHeads fh;
        CHECK(fh.load(kBadDir) == false);
        const std::vector<float> hidden(8, 0.0f);
        for (int i = 0; i < 3; ++i)  // repeated calls: still refusal, no throw
            CHECK(is_refusal(fh.classify(hidden.data(), "vision.scene", 3,
                                         static_cast<int32_t>(hidden.size()))));
        CHECK(fh.error().find("vision_head.bin") != std::string::npos);
        // Another bin's failure must not poison a different family's reason:
        CHECK(is_refusal(fh.classify(hidden.data(), "audio.emotion", 3,
                                     static_cast<int32_t>(hidden.size()))));
        CHECK(fh.error().find("audio_head.bin") != std::string::npos);
    }

    // ===================== PART C — the off-parity guard ====================
    TEST("C1: E mismatch with the synthetic blob refuses");
    {
        FittedHeads fh;
        CHECK(fh.load(kTmpDir));
        // 7-wide caller vector against an E=8 blob: without the guard this
        // matvec reads one float PAST the caller's buffer.
        const std::vector<float> hidden7(7, 1.0f);
        CHECK(is_refusal(fh.classify(hidden7.data(), "vision.scene", 3,
                                     static_cast<int32_t>(hidden7.size()))));
        // caller_E = 0 ("cannot check") still answers when the width really
        // matches the blob:
        const std::vector<float> hidden8(8, 0.0f);
        CHECK(!is_refusal(fh.classify(hidden8.data(), "vision.scene", 3, 0)));
    }

    TEST("C2: real fitted blob from models/heads (gated) + parity guard");
    {
        const std::string src = "models/heads/vision_head.bin";
        std::vector<uint8_t> blob;
        if (!file_bytes(src, blob)) {
            SKIP("models/heads/vision_head.bin absent (clean checkout)");
        } else {
            const std::string dst = std::string(kRealDir) + "/vision_head.bin";
            CHECK(write_bytes(dst, blob));
            FittedHeads fh;
            CHECK(fh.load(kRealDir) == true);
            // Real blob, real E: a zeros h[E] is a legal input (uniform-ish
            // logits from the bias) and the head ANSWERS.
            const std::vector<float> h768(768, 0.0f);
            const ClassificationResult ok =
                fh.classify(h768.data(), "vision.scene", 3, 768);
            CHECK(!is_refusal(ok));
            CHECK(ok.domain == "vision.scene");
            // Same blob, WRONG width: refusal, not a 512-wide guess.
            const std::vector<float> h512(512, 0.0f);
            CHECK(is_refusal(fh.classify(h512.data(), "vision.scene", 3, 512)));
        }
    }

    // ===================== PART D — dir switch drops caches =================
    TEST("D1: set_heads_dir() drops caches; old-dir blobs stop answering");
    {
        FittedHeads fh;
        CHECK(fh.load(kTmpDir));
        const std::vector<float> hidden(8, 0.0f);
        CHECK(!is_refusal(fh.classify(hidden.data(), "vision.scene", 3, 8)));
        fh.set_heads_dir("build/omniseed_no_such_dir");
        CHECK(fh.loaded_count() == 0);
        CHECK(is_refusal(fh.classify(hidden.data(), "vision.scene", 3, 8)));
        // And pointing back restores service (re-read from disk).
        fh.set_heads_dir(kTmpDir);
        CHECK(!is_refusal(fh.classify(hidden.data(), "vision.scene", 3, 8)));
    }

    // ============ PART E — vision bins are PER SET (T113c regression) =======
    TEST("E1: bin_for_set maps vision sets to their PER-SET bins");
    {
        CHECK(FittedHeads::bin_for_set("vision.scene") == "vision_head.bin");
        CHECK(FittedHeads::bin_for_set("vision.anomaly") ==
              "vision_anomaly_head.bin");
        CHECK(FittedHeads::bin_for_set("vision.candlestick") ==
              "vision_candlestick_head.bin");
        // An unmapped vision.* set must refuse (""), NOT be guessed into
        // vision_head.bin (which carries only vision.scene):
        CHECK(FittedHeads::bin_for_set("vision.unknown") == "");
        CHECK(FittedHeads::bin_for_set("vision") == "");
        // Prefix families and the trading split are unchanged:
        CHECK(FittedHeads::bin_for_set("language.intent") == "language_head.bin");
        CHECK(FittedHeads::bin_for_set("audio.wake") == "audio_head.bin");
        CHECK(FittedHeads::bin_for_set("trading.regime") ==
              "trading_regime_head.bin");
        CHECK(FittedHeads::bin_for_set("trading.action") == "trading_head.bin");
        CHECK(FittedHeads::bin_for_set("general.routing") == "");
    }

    TEST("E2: real vision_anomaly_head.bin answers vision.anomaly (gated)");
    {
        const std::string src = "models/heads/vision_anomaly_head.bin";
        std::vector<uint8_t> blob;
        if (!file_bytes(src, blob)) {
            SKIP("models/heads/vision_anomaly_head.bin absent (clean checkout)");
        } else {
            // A dir carrying ONLY the anomaly blob: with the old prefix map,
            // vision.anomaly -> vision_head.bin (missing here, and carrying
            // vision.scene anyway) could never answer.
            const std::string dst = std::string(kAnomDir) + "/vision_anomaly_head.bin";
            CHECK(write_bytes(dst, blob));
            FittedHeads fh;
            fh.set_heads_dir(kAnomDir);
            CHECK(fh.available("vision.anomaly") == true);
            const std::vector<float> h768(768, 0.0f);
            const ClassificationResult ok =
                fh.classify(h768.data(), "vision.anomaly", 2, 768);
            CHECK(!is_refusal(ok));
            CHECK(ok.domain == "vision.anomaly");
            CHECK(!ok.top_k.empty());
            // vision.scene maps to vision_head.bin, NOT carried by this dir:
            // refusal, not a cross-set answer from the anomaly blob.
            CHECK(is_refusal(fh.classify(h768.data(), "vision.scene", 2, 768)));
            CHECK(fh.error().find("vision_head.bin") != std::string::npos);
            // Off-parity guard still holds on the new path:
            const std::vector<float> h512(512, 0.0f);
            CHECK(is_refusal(fh.classify(h512.data(), "vision.anomaly", 2, 512)));
        }
    }

    TEST("E2b: real vision_candlestick_head.bin answers vision.candlestick (gated)");
    {
        const std::string src = "models/heads/vision_candlestick_head.bin";
        std::vector<uint8_t> blob;
        if (!file_bytes(src, blob)) {
            SKIP("models/heads/vision_candlestick_head.bin absent (clean checkout)");
        } else {
            const std::string dir = "build/omniseed_fitted_heads_dir_candle";
            make_dir(dir);
            CHECK(write_bytes(dir + "/vision_candlestick_head.bin", blob));
            FittedHeads fh;
            fh.set_heads_dir(dir);
            CHECK(fh.available("vision.candlestick") == true);
            const std::vector<float> h768(768, 0.0f);
            const ClassificationResult ok =
                fh.classify(h768.data(), "vision.candlestick", 2, 768);
            CHECK(!is_refusal(ok));
            CHECK(ok.domain == "vision.candlestick");
            CHECK(ok.top_k.size() == 2);  // K=2 blob answers with both labels
            CHECK(is_refusal(fh.classify(h768.data(), "vision.scene", 2, 768)));
        }
    }

    TEST("E3: every vision set with a mapped bin is in the C++ canonical list");
    {
        // The canonical list tools/train_heads.py parses lives in
        // default_domain_label_sets(); a vision set FittedHeads maps but the
        // list lacks = fit-time "not in the C++ canonical list" failure (the
        // exact T113c bug). Drift in EITHER direction fails here.
        bool saw_candlestick = false;
        for (const DomainLabelSetSpec& spec : default_domain_label_sets()) {
            const std::string name = spec.name;
            if (name.rfind("vision.", 0) != 0) continue;
            CHECK(FittedHeads::bin_for_set(name) != "");
            if (name == "vision.candlestick") {
                saw_candlestick = true;
                CHECK(spec.labels.size() == 2);
                CHECK(std::string(spec.labels[0]) == "up");
                CHECK(std::string(spec.labels[1]) == "down");
            }
        }
        CHECK(saw_candlestick == true);
        // And the reverse: the explicit bins in bin_for_set() must resolve to
        // a canonical set (walk the map's three known vision entries).
        const char* kMapped[] = {"vision.scene", "vision.anomaly",
                                 "vision.candlestick"};
        for (const char* mapped : kMapped) {
            bool found = false;
            for (const DomainLabelSetSpec& spec : default_domain_label_sets())
                if (std::string(spec.name) == mapped) { found = true; break; }
            CHECK(found);
        }
    }

    platform::log_info("fitted_heads: %d passed, %d failed, %d skipped",
                       g_passed, g_failed, g_skipped);
    return g_failed == 0 ? 0 : 1;
}
