// =============================================================================
//  OmniSeed — dbg_mel.cpp (offline debug tool; built manually with cl.exe —
//  intentionally NOT registered in CMake, like dbg_sensory.cpp)
//  Dumps the C++ log-mel [80,T] and encoder output [T/2,384] for the fixture
//  WAV (raw f32), for staged numeric comparison against the official HF
//  reference via tools/dbg_whisper_ref.py:
//    cl /std:c++17 /EHsc /I include /Fe:build/bin/dbg_mel.exe tools/dbg_mel.cpp \
//       build/Release/omniseed_core.lib
//    OMNISEED_DBG_IDS=1 ./build/bin/dbg_mel.exe     # also runs transcribe()
//    python tools/dbg_whisper_ref.py --stage enc --slice-mel <T_mel>
//  OMNISEED_DBG_IDS=1 additionally prints the greedy decode ids + text.
// =============================================================================
#include "omniseed/audio/audio.h"
#include "omniseed/core/platform.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace omniseed;

static std::vector<int16_t> load_pcm(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::printf("no fixture\n"); return {}; }
    std::fseek(f, 0, SEEK_END);
    const long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<unsigned char> raw(static_cast<size_t>(sz));
    if (std::fread(raw.data(), 1, raw.size(), f) != raw.size()) { std::fclose(f); return {}; }
    std::fclose(f);
    size_t p = 12;
    std::vector<int16_t> pcm;
    while (p + 8 <= raw.size()) {
        const uint32_t csz = static_cast<uint32_t>(raw[p + 4]) |
                             (static_cast<uint32_t>(raw[p + 5]) << 8) |
                             (static_cast<uint32_t>(raw[p + 6]) << 16) |
                             (static_cast<uint32_t>(raw[p + 7]) << 24);
        if (std::memcmp(&raw[p], "data", 4) == 0) {
            pcm.resize(csz / 2);
            std::memcpy(pcm.data(), &raw[p + 8], csz);
            break;
        }
        p += 8 + csz + (csz & 1);
    }
    return pcm;
}

static void dump_stats(const char* tag, const float* v, int64_t n,
                       const char* file) {
    float mn = 1e30f, mx = -1e30f, mean = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        const float x = v[i];
        if (x < mn) mn = x;
        if (x > mx) mx = x;
        mean += x;
    }
    mean /= static_cast<float>(n);
    std::printf("%s mean %.4f min %.4f max %.4f\n", tag, mean, mn, mx);
    FILE* o = std::fopen(file, "wb");
    std::fwrite(v, 4, static_cast<size_t>(n), o);
    std::fclose(o);
    std::printf("wrote %s\n", file);
}

int main() {
    const std::vector<int16_t> pcm = load_pcm(
        "tests/fixtures/2830-3980-0043.wav");
    if (pcm.empty()) return 1;

    PcmAudio a;
    a.sample_rate = 16000;
    a.samples.resize(pcm.size());
    for (size_t i = 0; i < pcm.size(); ++i)
        a.samples[i] = static_cast<float>(pcm[i]) / 32768.0f;

    WhisperTiny wh;
    if (!wh.load("models/whisper-tiny-encoder.gguf")) {
        std::printf("whisper load failed: %s\n", wh.error().c_str());
        return 1;
    }

    Tensor mel;
    if (!wh.compute_mel(a, mel)) {
        std::printf("compute_mel failed: %s\n", wh.error().c_str());
        return 1;
    }
    std::printf("mel [%lld x %lld]\n", (long long)mel.dim(0),
                (long long)mel.dim(1));
    dump_stats("mel", mel.f32(), mel.numel(), "C:/tmp/mel_cpp.bin");

    Tensor enc;
    if (!wh.encode(a, enc)) {
        std::printf("encode failed: %s\n", wh.error().c_str());
        return 1;
    }
    std::printf("enc [%lld x %lld]\n", (long long)enc.dim(0),
                (long long)enc.dim(1));
    dump_stats("enc", enc.f32(), enc.numel(), "C:/tmp/enc_cpp.bin");

    if (std::getenv("OMNISEED_DBG_IDS")) {
        std::string text;
        if (!wh.transcribe(a, text)) {
            std::printf("transcribe failed: %s\n", wh.error().c_str());
            return 1;
        }
        std::printf("text: \"%s\"\n", text.c_str());
    }
    std::printf("probe done\n");
    return 0;
}
