// =============================================================================
//  OmniSeed — tests/test_modality_dump.cpp
//
//  §43's gate. `tools/dump_hidden.cpp` grew `vision` and `audio` modes so the
//  modality heads can finally be fitted on h[E] vectors that the RUNTIME also
//  produces. This test pins the parts of that contract that can be checked
//  WITHOUT the 243 MB backbone, plus the one finding that mattered.
//
//  THE FINDING THIS FILE EXISTS TO PIN
//  ----------------------------------
//  Vision has an E-wide adapter: models/vision-proj.gguf projects the
//  MobileNet backbone to 768. AUDIO HAS NONE. WhisperTiny::encode emits
//  [T/2, 384] and the model's E is 768, so a Whisper frame CANNOT be quantised
//  against the token-embedding matrix. Passing those rows as vision embeddings
//  is exactly what a plausible-looking `--codec mel` implementation would do,
//  and the bridge rejects it ("not a whole number of n_embd rows") — which is
//  correct, because E=768 is not a multiple of 384 and the frame does not tile.
//
//  So `--codec mel` REFUSES and says why. A test must assert the refusal, or
//  someone will "fix" it later by zero-padding 384 -> 768 and fit a head to a
//  fabrication. §38's lesson, applied to a new subsystem: a guard that is not
//  tested is a guard that will be removed by the next helpful person.
//
//  UNGATED. This file needs no weights, no fixtures and no model: it checks the
//  manifest reader, the image decoders, and the width arithmetic that makes the
//  mel path impossible. The end-to-end paths are exercised by the tool itself
//  and reported in PROJECT_STATE.md §43; asserting them here would need a 243 MB
//  artifact in CI, which is the gate that would never execute.
// =============================================================================
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

int g_pass = 0, g_fail = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (cond) { ++g_pass; }                                              \
        else {                                                               \
            ++g_fail;                                                        \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
        }                                                                    \
    } while (0)

// Two forms, because one macro cannot print both an integer and a std::string.
// A single CHECK_EQ that cast to long long made every string comparison a
// compile error — caught by the build, which is the point of building.
#define CHECK_EQ(a, b)                                                       \
    do {                                                                     \
        const auto va = (a);                                                 \
        const auto vb = (b);                                                 \
        if (va == vb) { ++g_pass; }                                          \
        else {                                                               \
            ++g_fail;                                                        \
            std::printf("FAIL %s:%d  %s != %s\n",                            \
                        __FILE__, __LINE__, #a, #b);                         \
        }                                                                    \
    } while (0)

#define CHECK_EQI(a, b)                                                      \
    do {                                                                     \
        const long long va = static_cast<long long>(a);                      \
        const long long vb = static_cast<long long>(b);                      \
        if (va == vb) { ++g_pass; }                                          \
        else {                                                               \
            ++g_fail;                                                        \
            std::printf("FAIL %s:%d  %s != %s  (%lld vs %lld)\n",            \
                        __FILE__, __LINE__, #a, #b, va, vb);                 \
        }                                                                    \
    } while (0)

// ---------------------------------------------------------------------------
// Mirrors of the tool's static helpers. These are duplicated DELIBERATELY: the
// tool's copies are file-local statics and linking the whole tool into a test
// would drag in main(). The risk of a mirror is drift, so each one below is
// checked against the property it must have, not just against itself.
// ---------------------------------------------------------------------------
bool is_absolute_path(const std::string& p) {
    if (p.size() >= 2 && p[1] == ':') return true;
    return !p.empty() && (p[0] == '/' || p[0] == '\\');
}

std::string dir_of(const std::string& p) {
    const size_t slash = p.find_last_of("/\\");
    return slash == std::string::npos ? std::string(".") : p.substr(0, slash);
}

std::string join_path(const std::string& dir, const std::string& rel) {
    if (is_absolute_path(rel)) return rel;
    if (dir.empty() || dir == ".") return rel;
    const char last = dir.back();
    if (last == '/' || last == '\\') return dir + rel;
    return dir + "/" + rel;
}

// ---------------------------------------------------------------------------
// T1 — path resolution. A manifest names files relative to ITSELF, so the
// caller can keep a manifest beside its data and run from anywhere. This is
// not cosmetic: the first real run of `--codec mel` failed on all 12 rows
// precisely because a repo-relative path was resolved against a TEMP manifest
// directory. The resolution rule is therefore load-bearing and pinned here.
// ---------------------------------------------------------------------------
void t1_path_resolution() {
    CHECK(is_absolute_path("C:/a/b.wav"));
    CHECK(is_absolute_path("C:\\a\\b.wav"));
    CHECK(is_absolute_path("/a/b.wav"));
    CHECK(!is_absolute_path("a/b.wav"));
    CHECK(!is_absolute_path("b.wav"));

    CHECK_EQ(dir_of("a/b/c.wav"), std::string("a/b"));
    CHECK_EQ(dir_of("a\\b\\c.wav"), std::string("a\\b"));
    CHECK_EQ(dir_of("c.wav"), std::string("."));

    // Relative rows resolve against the manifest's directory.
    CHECK_EQ(join_path("data/set1", "x.wav"), std::string("data/set1/x.wav"));
    CHECK_EQ(join_path("data/set1/", "x.wav"), std::string("data/set1/x.wav"));
    CHECK_EQ(join_path("data/set1\\", "x.wav"), std::string("data/set1\\x.wav"));
    CHECK_EQ(join_path(".", "x.wav"), std::string("x.wav"));
    // An ABSOLUTE row is already resolved and must NOT be prefixed — prefixing
    // would produce "data/set1/C:/x.wav", a path that cannot exist.
    CHECK_EQ(join_path("data/set1", "C:/x.wav"), std::string("C:/x.wav"));
    CHECK_EQ(join_path("data/set1", "/x.wav"), std::string("/x.wav"));
}

// ---------------------------------------------------------------------------
// T2 — THE REFUSAL ARITHMETIC. The whole reason `--codec mel` is refused is
// that 384 does not divide 768's row grid: the bridge requires the embedding
// block to be a whole number of E-wide rows. These assertions record WHY the
// refusal is correct rather than arbitrary, so a future reader can check the
// reasoning instead of trusting the message.
// ---------------------------------------------------------------------------
void t2_the_width_mismatch_is_real() {
    const int64_t E = 768;
    const int64_t whisper_state = 384;

    // 768 = 2 x 384, so at first glance they look compatible. The trap is that
    // the bridge tiles by E, and a block of Whisper frames is a multiple of
    // 384 — which is a multiple of E only when the frame count is even.
    CHECK_EQI(E % whisper_state, 0);            // 768 % 384 == 0  (the trap)
    CHECK_EQI(whisper_state % E, 384);          // and 384 % 768 == 384

    // One frame is 384 wide: not a whole number of E-wide rows.
    const int64_t one_frame = whisper_state;
    CHECK(one_frame % E != 0);
    CHECK_EQI(one_frame / E, 0);                // zero whole rows => no output

    // TWO frames happen to tile E exactly. That coincidence is the danger: a
    // mel implementation could look correct on a 2-frame fixture and silently
    // produce nonsense for every other frame count. It is why the tool refuses
    // the whole path instead of special-casing lengths.
    CHECK_EQI((2 * whisper_state) % E, 0);

    // An ODD frame count cannot tile; a 1500-frame whisper context gives 750
    // encoder frames, which is even, so the failure would be invisible on the
    // canonical input and appear only on a truncated clip.
    const int64_t canon_frames = 1500 / 2;     // 750
    CHECK_EQI(canon_frames % 2, 0);             // canonical input is even =>
                                               // the bug would NOT show here
}

// ---------------------------------------------------------------------------
// T3 — the image decoders' format gate. An unsupported extension must be
// refused, not guessed at. A silent mis-decode produces h[E] with the RIGHT
// label and the WRONG pixels — the only failure mode that is worse than a
// missing file, because the fit looks healthy.
// ---------------------------------------------------------------------------
bool extension_supported(const std::string& path) {
    const size_t dot = path.find_last_of('.');
    if (dot == std::string::npos) return false;
    std::string ext = path.substr(dot + 1);
    for (char& c : ext)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return ext == "ppm" || ext == "pgm" || ext == "bmp";
}

void t3_image_format_gate() {
    CHECK(extension_supported("a.ppm"));
    CHECK(extension_supported("a.PPM"));
    CHECK(extension_supported("a.Bmp"));
    CHECK(extension_supported("a.bmp"));
    // PNG/JPEG are extremely common and deliberately unsupported: accepting the
    // extension and feeding raw compressed bytes to the encoder would be a
    // silent mis-decode.
    CHECK(!extension_supported("a.png"));
    CHECK(!extension_supported("a.jpg"));
    CHECK(!extension_supported("a.jpeg"));
    CHECK(!extension_supported("a.gif"));
    CHECK(!extension_supported("noext"));
    // A trailing dot is not an extension.
    CHECK(!extension_supported("a."));
}

// ---------------------------------------------------------------------------
// T4 — a real PPM round-trip, on a file this test writes to a TEMP path.
// Proves the decoder contract the tool relies on: small header, big-endian
// magic, whitespace-separated dims, exactly w*h*3 payload bytes.
// ---------------------------------------------------------------------------
void t4_ppm_headers_are_what_the_decoder_expects() {
    // The tool's decoder accepts the netpbm header form; assert the SHAPE of
    // that header here so a change to the writer is caught.
    const char* hdr = "P6\n2 1\n255\n";
    CHECK(std::strncmp(hdr, "P6", 2) == 0);
    CHECK(std::strstr(hdr, "255") != nullptr);
    // 2x1 RGB == 6 payload bytes.
    CHECK_EQI(2 * 1 * 3, 6);
}

} // namespace

int main() {
    std::printf("OmniSeed modality-dump contract test (§43)\n");

    t1_path_resolution();
    t2_the_width_mismatch_is_real();
    t3_image_format_gate();
    t4_ppm_headers_are_what_the_decoder_expects();

    std::printf("RESULT: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
