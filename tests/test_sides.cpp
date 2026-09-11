// =============================================================================
//  OmniSeed — test_sides.cpp
//  Modality sidecar validation: loads the converted whisper-tiny encoder and
//  vision projection GGUFs and runs real encode paths. Skips cleanly when the
//  sidecars are absent (fresh clones / CI).
//    models/whisper-tiny-encoder.gguf  (tools/convert_senses.py)
//    models/vision-proj.gguf
// =============================================================================
#include "omniseed/audio/audio.h"
#include "omniseed/core/platform.h"
#include "omniseed/vision/vision.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

using namespace omniseed;

static int g_pass = 0, g_fail = 0;
#define CHECK(cond)                                                          \
    do {                                                                     \
        if (cond) { ++g_pass; }                                              \
        else {                                                               \
            ++g_fail;                                                        \
            std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
        }                                                                    \
    } while (0)

namespace {

bool file_exists(const char* p) {
    std::FILE* f = nullptr;
#ifdef _MSC_VER
    if (fopen_s(&f, p, "rb") != 0) return false;
#else
    f = std::fopen(p, "rb");
#endif
    if (f == nullptr) return false;
    std::fclose(f);
    return true;
}

PcmAudio make_tone(double hz, double seconds, int32_t sr = 16000) {
    PcmAudio a;
    a.sample_rate = sr;
    a.samples.resize(static_cast<size_t>(seconds * sr));
    for (size_t i = 0; i < a.samples.size(); ++i) {
        const double t = static_cast<double>(i) / sr;
        a.samples[i] = static_cast<float>(
            0.5 * std::sin(2.0 * 3.14159265358979323846 * hz * t));
    }
    return a;
}

Image make_gradient(int w, int h) {
    Image im;
    im.width = w;
    im.height = h;
    im.rgb.resize(static_cast<size_t>(w) * h * 3);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < 3; ++c)
                im.rgb[(static_cast<size_t>(y) * w + x) * 3 + c] =
                    static_cast<uint8_t>((x * 255 / w + y * 255 / h + c * 40) % 256);
    return im;
}

} // namespace

// Loads a 16 kHz mono 16-bit PCM WAV (the get_fixtures.py format) as floats.
static bool load_wav16(const char* path, PcmAudio& out) {
    std::FILE* f = nullptr;
#ifdef _MSC_VER
    if (fopen_s(&f, path, "rb") != 0) return false;
#else
    f = std::fopen(path, "rb");
#endif
    if (f == nullptr) return false;
    auto ok = [&](bool cond) { if (!cond) { std::fclose(f); return false; }
                               return true; };
    unsigned char riff[12];
    if (!ok(std::fread(riff, 1, 12, f) == 12 && riff[8] == 'W')) return false;
    int sr = 0, ch = 0, bits = 0;
    std::vector<int16_t> pcm;
    while (true) {
        unsigned char hdr[8];
        if (std::fread(hdr, 1, 8, f) != 8) break;
        const uint32_t size = static_cast<uint32_t>(hdr[4]) |
                              (static_cast<uint32_t>(hdr[5]) << 8) |
                              (static_cast<uint32_t>(hdr[6]) << 16) |
                              (static_cast<uint32_t>(hdr[7]) << 24);
        if (std::memcmp(hdr, "fmt ", 4) == 0) {
            unsigned char fmt[16];
            if (!ok(size >= 16 && std::fread(fmt, 1, 16, f) == 16)) return false;
            ch   = fmt[2] | (fmt[3] << 8);
            sr   = fmt[4] | (fmt[5] << 8) | (fmt[6] << 16) | (fmt[7] << 24);
            bits = fmt[14] | (fmt[15] << 8);
            if (size > 16) std::fseek(f, static_cast<long>(size - 16), SEEK_CUR);
        } else if (std::memcmp(hdr, "data", 4) == 0) {
            if (!ok(bits == 16 && ch == 1)) return false;
            pcm.resize(size / 2);
            if (!ok(std::fread(pcm.data(), 2, pcm.size(), f) == pcm.size()))
                return false;
            break;
        } else {
            if (!ok(std::fseek(f, static_cast<long>(size), SEEK_CUR) == 0))
                return false;
        }
    }
    std::fclose(f);
    if (sr != 16000 || pcm.empty()) return false;
    out.sample_rate = sr;
    out.samples.resize(pcm.size());
    for (size_t i = 0; i < pcm.size(); ++i)
        out.samples[i] = static_cast<float>(pcm[i]) / 32768.0f;
    return true;
}

// lowercased alphanumeric word set of a transcription/reference string.
static std::vector<std::string> words_of(const std::string& s) {
    std::vector<std::string> w;
    std::string cur;
    for (const char c : s) {
        if (std::isalnum(static_cast<unsigned char>(c)))
            cur += static_cast<char>(std::tolower(
                static_cast<unsigned char>(c)));
        else if (!cur.empty()) { w.push_back(cur); cur.clear(); }
    }
    if (!cur.empty()) w.push_back(cur);
    return w;
}

// Reads tests/fixtures/manifest.txt (written by tools/get_fixtures.py) for the
// expected transcription of the anchor clip.
bool read_manifest_expected(std::string& wav_name, std::string& expected) {
    std::FILE* f = nullptr;
#ifdef _MSC_VER
    if (fopen_s(&f, "tests/fixtures/manifest.txt", "rb") != 0) return false;
#else
    f = std::fopen("tests/fixtures/manifest.txt", "rb");
#endif
    if (f == nullptr) return false;
    char line[512];
    while (std::fgets(line, sizeof(line), f)) {
        const std::string l(line);
        if (l.rfind("fixture=", 0) == 0)
            wav_name = l.substr(8);
        else if (l.rfind("expected_text=", 0) == 0)
            expected = l.substr(14);
    }
    std::fclose(f);
    while (!wav_name.empty() &&
           (wav_name.back() == '\n' || wav_name.back() == '\r'))
        wav_name.pop_back();
    while (!expected.empty() &&
           (expected.back() == '\n' || expected.back() == '\r'))
        expected.pop_back();
    return !wav_name.empty() && !expected.empty();
}

int main() {
    platform::log_info("OmniSeed sidecar (senses) test");

    // ------------------------- whisper encoder -------------------------------
    if (file_exists("models/whisper-tiny-encoder.gguf")) {
        WhisperTiny wh;
        if (!wh.load("models/whisper-tiny-encoder.gguf"))
            std::printf("  whisper load error: %s\n", wh.error().c_str());
        CHECK(wh.valid());
        if (wh.valid()) {
            const auto& hp = wh.hparams();
            CHECK(hp.n_mels == 80 && hp.n_audio_state == 384 &&
                  hp.n_audio_layer == 4);

            // official mel filterbank loaded (not Slaney-reconstructed):
            // shape [80, 201]
            Tensor mel;
            PcmAudio tone = make_tone(440.0, 1.0);
            std::printf("  before compute_mel\n"); std::fflush(stdout);
            CHECK(wh.compute_mel(tone, mel));
            std::printf("  after compute_mel [%lld x %lld]\n",
                        (long long)mel.dim(0), (long long)mel.dim(1));
            std::fflush(stdout);
            CHECK(mel.dim(0) == 80);

            // real encoder pass runs and yields [T/2, 384]
            Tensor enc;
            std::printf("  before encode\n"); std::fflush(stdout);
            CHECK(wh.encode(tone, enc));
            std::printf("  after encode [%lld x %lld]\n",
                        (long long)enc.dim(0), (long long)enc.dim(1));
            std::fflush(stdout);
            CHECK(enc.dim(1) == 384);
            CHECK(enc.dim(0) > 0);
            bool finite = true;
            for (int64_t i = 0; i < enc.numel(); ++i)
                if (!std::isfinite(enc.f32()[i])) { finite = false; break; }
            CHECK(finite);

            // Transcription runtime bound (both decoder blocks below): no-
            // pathological-loop guard, not a perf SLA — sanitizer/Debug builds
            // legitimately run several times slower, so the bound is scalable
            // via OMNISEED_TRANSCRIBE_TIMEOUT_S (default 30).
            double secs_bound = 30.0;
            if (const char* env = std::getenv(
                    "OMNISEED_TRANSCRIBE_TIMEOUT_S")) {
                const double v = std::atof(env);
                if (v > 0.0) secs_bound = v;
            }

            // ---- real greedy decoder (when the sidecar has it) ------------
            if (wh.decoder_loaded()) {
                std::printf("  decoder loaded — transcribing 5s silence…\n");
                std::fflush(stdout);
                PcmAudio silence = make_tone(0.0, 5.0);   // 0 Hz = digital silence
                std::string text;
                const std::clock_t t0 = std::clock();
                // Digital silence has zero speech energy, so the trailing-
                // silence trim leaves < 2 mel frames and transcribe reports
                // a clean error instead of running the decoder on nothing.
                const bool ok = wh.transcribe(silence, text);
                const double secs =
                    static_cast<double>(std::clock() - t0) / CLOCKS_PER_SEC;
                std::printf("  transcribe: \"%s\" (%.1fs) ok=%d err=%s\n",
                            text.c_str(), secs, ok ? 1 : 0,
                            wh.error().c_str());
                std::fflush(stdout);
                CHECK(ok || text.empty());               // either is correct
                CHECK(secs < secs_bound);
                CHECK(text.size() < 4096);                // bounded output
                for (const char c : text)                 // printable/UTF-8 bytes
                    CHECK(static_cast<unsigned char>(c) >= 0x09);
            } else {
                std::printf("  [skip] sidecar has no decoder tensors\n");
            }

            // ---- real-speech ASR fixtures (tools/get_fixtures.py) ----------
            // Asserts end-to-end transcription on a real LibriSpeech clip:
            // non-empty text, bounded runtime, and word overlap with the
            // reference. Skips cleanly when fixtures are absent.
            std::string fx_wav, fx_text;
            if (!read_manifest_expected(fx_wav, fx_text)) {
                std::printf("  [skip] ASR fixtures not present "
                            "(run: python tools/get_fixtures.py)\n");
            } else if (wh.decoder_loaded()) {
                const std::string wav_path =
                    std::string("tests/fixtures/") + fx_wav;
                PcmAudio speech;
                CHECK(load_wav16(wav_path.c_str(), speech));
                if (speech.samples.size() > 0) {
                    std::printf("  transcribing real clip %s (%.2f s)…\n",
                                fx_wav.c_str(),
                                static_cast<double>(speech.samples.size()) /
                                    speech.sample_rate);
                    std::fflush(stdout);
                    std::string got;
                    const std::clock_t t1 = std::clock();
                    CHECK(wh.transcribe(speech, got));
                    const double secs1 =
                        static_cast<double>(std::clock() - t1) /
                        CLOCKS_PER_SEC;
                    std::printf("  transcribe: \"%s\" (%.1fs)\n",
                                got.c_str(), secs1);
                    CHECK(secs1 < secs_bound);
                    CHECK(!got.empty());
                    // Strip whisper <|special|> tokens (timestamps) before
                    // the word-overlap comparison.
                    std::string plain;
                    plain.reserve(got.size());
                    for (size_t i = 0; i < got.size();) {
                        if (got[i] == '<') {
                            const size_t j = got.find(">", i);
                            if (j != std::string::npos &&
                                i + 1 < got.size() && got[i + 1] == '|') {
                                i = j + 1;
                                continue;
                            }
                        }
                        plain += got[i++];
                    }
                    // Word overlap vs the reference (whisper lowercases;
                    // greedy + short clip tolerates a dropped/extra word).
                    const auto ref = words_of(fx_text);
                    const auto hyp = words_of(plain);
                    int hits = 0;
                    for (const auto& w : hyp) {
                        if (std::find(ref.begin(), ref.end(), w) != ref.end())
                            ++hits;
                    }
                    const double overlap =
                        ref.empty() ? 0.0
                                    : static_cast<double>(hits) /
                                          static_cast<double>(ref.size());
                    std::printf("  word overlap vs reference: %.0f%%\n",
                                overlap * 100.0);
                    CHECK(overlap >= 0.5);
                }
            }
        }
    } else {
        std::printf("  [skip] whisper sidecar not present\n");
    }

    // --------------------------- vision proj ---------------------------------
    if (file_exists("models/vision-proj.gguf")) {
        VisionEncoder vis;
        if (!vis.load("models/vision-proj.gguf"))
            std::printf("  vision load error: %s\n", vis.error().c_str());
        CHECK(vis.valid());
        if (vis.valid()) {
            UniCompress comp;
            Image im = make_gradient(96, 96);
            Tensor out;
            CHECK(vis.encode(im, out, comp));
            CHECK(out.dim(1) == 768);
            bool finite = true;
            for (int64_t i = 0; i < out.numel(); ++i)
                if (!std::isfinite(out.f32()[i])) { finite = false; break; }
            CHECK(finite);
            // projected tokens must actually vary with input (not identity)
            Image im2 = make_gradient(96, 96);
            for (size_t i = 0; i < im2.rgb.size(); ++i)
                im2.rgb[i] = static_cast<uint8_t>(255 - im2.rgb[i]);
            Tensor out2;
            CHECK(vis.encode(im2, out2, comp));
            float diff = 0.0f;
            for (int64_t i = 0; i < out.numel(); ++i)
                diff = std::max(diff, std::fabs(out.f32()[i] - out2.f32()[i]));
            CHECK(diff > 1e-4f);
        }
    } else {
        std::printf("  [skip] vision sidecar not present\n");
    }

    std::printf("RESULT: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
