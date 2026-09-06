// Debug probe for SensoryFingerprint voice verification (temporary tool).
#include "omniseed/runtime/sensory.h"
#include "omniseed/audio/audio.h"
#include <cstdio>
#include <cmath>
using namespace omniseed;

static PcmAudio make_tone(double hz, double seconds, int32_t sr = 16000) {
    PcmAudio a; a.sample_rate = sr;
    a.samples.resize(static_cast<size_t>(seconds * sr));
    for (size_t i = 0; i < a.samples.size(); ++i) {
        double t = (double)i / sr;
        a.samples[i] = (float)(0.4 * std::sin(2 * 3.14159265358979323846 * hz * t));
    }
    return a;
}

int main() {
    SensoryFingerprint sf;
    SensoryObservation a1;
    PcmAudio ta = make_tone(120.0, 1.5);
    a1.pcm = ta.samples.data(); a1.pcm_len = ta.samples.size();
    for (int i = 0; i < 20; ++i) a1.digraph_ms.push_back(110.0f + i * 3.0f);
    sf.enroll(a1);
    std::printf("enrolled voice=%d typing=%d\n",
                sf.profile().has_voice(), sf.profile().has_typing());

    SensoryObservation b1;
    PcmAudio tb = make_tone(300.0, 1.5);
    b1.pcm = tb.samples.data(); b1.pcm_len = tb.samples.size();
    for (int i = 0; i < 20; ++i) b1.digraph_ms.push_back(40.0f + (i % 4) * 15.0f);
    bool m = false;
    float s = sf.verify(b1, m);
    std::printf("cross-user score=%.4f match=%d\n", s, m);

    SensoryObservation a2 = a1;
    bool m2 = false;
    std::printf("same-user score=%.4f match=%d\n", sf.verify(a2, m2), m2);

    // Inspect embeddings.
    std::printf("A voice dims: ");
    for (int i = 0; i < 8; ++i) std::printf("%.3f ", sf.profile().voice[i]);
    std::printf("\nB-vs-A: we need B's embedding; rerun via second profile\n");
    SensoryFingerprint sf2;
    sf2.enroll(b1);
    std::printf("B voice dims: ");
    for (int i = 0; i < 8; ++i) std::printf("%.3f ", sf2.profile().voice[i]);
    std::printf("\n");
    return 0;
}
