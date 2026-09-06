// Perf profiler: which matmuls dominate a forward pass?
// Standalone build with /arch:AVX2 to measure kernel headroom.
#include "omniseed/core/rwkv.h"
#include "omniseed/core/platform.h"

#include <cstdio>
#include <vector>

using namespace omniseed;

int main(int argc, char** argv) {
    const char* model_path =
        argc > 1 ? argv[1] : "models/rwkv7-0.1B-ternary.gguf";

    RwkvModel model;
    if (!model.load(model_path)) {
        std::printf("load failed: %s\n", model.error().c_str());
        return 1;
    }
    const auto& cfg = model.config();
    std::printf("model: L=%d E=%d V=%d ffn=%d\n", cfg.n_layers, cfg.n_embd,
                cfg.n_vocab, cfg.ffn_inter);

    RwkvState st;
    model.init_state(st);
    Tensor logits("logits", {cfg.n_vocab}, DType::F32);

    // warmup
    for (int i = 0; i < 8; ++i) model.forward(11, st, logits);

    const auto t0 = platform::now_ms();
    constexpr int kIters = 64;
    for (int i = 0; i < kIters; ++i) model.forward(11, st, logits);
    const double ms = platform::now_ms() - t0;

    std::printf("forward: %.2f ms/token (%d iters) -> %.2f tok/s\n", ms / kIters,
                kIters, 1000.0 * kIters / ms);
    std::printf("peak RSS: %.2f MB\n",
                static_cast<double>(platform::peak_rss_bytes()) / 1048576.0);

    // Theoretical matmul work per token (int8 MACs), for Amdahl context:
    // attention: 12 layers * (4 x E*E) = 12*4*768*768
    // lora:      12 * (w1+w2: E*64 + 64*E; a1+a2: E*64 + 64*E;
    //                  g1+g2: E*128 + 128*E; v1+v2: E*32 + 32*E)
    // ffn:       12 * (E*3072 + 3072*E)
    // head:      V*E
    const double att = 12.0 * 4 * 768.0 * 768.0;
    const double lora = 12.0 * (2 * 768.0 * 64 + 2 * 768.0 * 64 +
                                2 * 768.0 * 128 + 2 * 768.0 * 32);
    const double ffn = 12.0 * (2 * 768.0 * 3072.0);
    const double head = 65536.0 * 768.0;
    const double total = att + lora + ffn + head;
    std::printf("MACs/token: att=%.1fM lora=%.1fM ffn=%.1fM head=%.1fM total=%.1fM\n",
                att / 1e6, lora / 1e6, ffn / 1e6, head / 1e6, total / 1e6);
    std::printf("achieved: %.2f GMAC/s\n", total / 1e9 / (ms / 1000.0));
    return 0;
}
