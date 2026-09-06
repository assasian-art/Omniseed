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

#include <cmath>
#include <cstdio>
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
        }
    } else {
        std::printf("  [skip] whisper sidecar not present\n");
    }

    // --------------------------- vision proj ---------------------------------
    if (file_exists("models/vision-proj.gguf")) {
        VisionEncoder vis;
        std::printf("  before vis load\n"); std::fflush(stdout);
        if (!vis.load("models/vision-proj.gguf"))
            std::printf("  vision load error: %s\n", vis.error().c_str());
        std::printf("  after vis load\n"); std::fflush(stdout);
        CHECK(vis.valid());
        if (vis.valid()) {
            std::printf("  before vis encode\n"); std::fflush(stdout);
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
