// =============================================================================
//  OmniSeed — lora.cpp
//  LoRA sidecar loader + runtime delta application. See lora.h for the
//  container layout and the y = W·x + scaling·B(A·x) contract.
// =============================================================================
#include "omniseed/core/lora.h"

#include "omniseed/core/platform.h"

#include <cmath>
#include <vector>

namespace omniseed {

namespace {
// Rank below which the A·x product lives on the stack (no heap traffic in
// the per-token hot path). Real sidecars use rank 8; anything up to 256
// avoids allocation.
constexpr int64_t kMaxStackRank = 256;
} // namespace

bool LoraAdapter::load(const std::string& path) {
    valid_ = false;
    error_.clear();
    if (!store_.open(path)) {
        error_ = "lora sidecar open failed: " + store_.error();
        return false;
    }
    if (store_.get_string("general.architecture") != "omniseed-lora") {
        error_ = "not an omniseed-lora sidecar";
        return false;
    }
    rank_        = static_cast<int32_t>(store_.get_u64("lora.rank", 0));
    layer_count_ = static_cast<int32_t>(store_.get_u64("lora.layer_count", 0));
    n_embd_      = static_cast<int32_t>(store_.get_u64("lora.n_embd", 0));
    scaling_     = static_cast<float>(store_.get_f64("lora.scaling", 1.0));
    if (rank_ <= 0 || layer_count_ <= 0 || n_embd_ <= 0) {
        error_ = "lora sidecar metadata incomplete";
        return false;
    }
    if (!store_.has_tensor("lora.0.att.receptance.a")) {
        error_ = "lora sidecar has no lora.0.att.receptance.a tensor";
        return false;
    }
    valid_ = true;
    build_hot_cache();
    return true;
}

LoraAdapter::Factors LoraAdapter::factors(int32_t layer,
                                          LoraTarget target) const {
    static const char* kAttNames[] = {
        "receptance", "key", "value", "output",
    };
    const char* name = nullptr;
    bool ffn = false;
    switch (target) {
        case LoraTarget::Receptance: name = kAttNames[0]; break;
        case LoraTarget::Key:        name = kAttNames[1]; break;
        case LoraTarget::Value:      name = kAttNames[2]; break;
        case LoraTarget::Output:     name = kAttNames[3]; break;
        case LoraTarget::FfnKey:     name = "key";   ffn = true;  break;
        case LoraTarget::FfnValue:   name = "value"; ffn = true;  break;
    }
    const std::string prefix = ffn ? "lora." + std::to_string(layer)
                                         + ".ffn." + name
                                   : "lora." + std::to_string(layer)
                                         + ".att." + name;
    Factors f;
    if (store_.has_tensor(prefix + ".a") && store_.has_tensor(prefix + ".b")) {
        f.a = store_.tensor(prefix + ".a");
        f.b = store_.tensor(prefix + ".b");
    }
    return f;
}

// ---------------------------------------------------------------------------
// Hot-path cache: per (layer, target), keep batch-converted fp32 copies of A
// and B. Built once at load; forward() then costs zero string work and zero
// half_to_float calls per token (half→float is exact, so the live math is
// bit-identical to the f16-view version).
// ---------------------------------------------------------------------------
void LoraAdapter::build_hot_cache() {
    hot_.clear();
    for (int32_t l = 0; l < layer_count_; ++l) hot_.emplace_back();
    for (int32_t l = 0; l < layer_count_; ++l) {
        for (int t = 0; t < 6; ++t) {
            const auto target = static_cast<LoraTarget>(t);
            const Factors f = factors(l, target);
            HotEntry& e = hot_[static_cast<size_t>(l)][static_cast<size_t>(t)];
            if (f.a.numel() == 0 || f.b.numel() == 0) continue;   // untargeted
            const int64_t r    = f.a.dim(0);
            const int64_t nin  = f.a.dim(1);
            const int64_t nout = f.b.dim(0);
            e.rank = r;
            e.out  = nout;
            e.a32 = std::make_unique<float[]>(static_cast<size_t>(r * nin));
            e.b32 = std::make_unique<float[]>(static_cast<size_t>(nout * r));
            for (int64_t i = 0; i < r * nin; ++i)
                e.a32[static_cast<size_t>(i)] = half_to_float(f.a.f16()[i]);
            for (int64_t i = 0; i < nout * r; ++i)
                e.b32[static_cast<size_t>(i)] = half_to_float(f.b.f16()[i]);
        }
    }
}

const LoraAdapter::HotEntry* LoraAdapter::hot_entry(int32_t layer,
                                                    LoraTarget target) const {
    if (layer < 0 || layer >= layer_count_) return nullptr;
    const HotEntry& e =
        hot_[static_cast<size_t>(layer)][static_cast<size_t>(target)];
    return (e.rank > 0 && e.a32 && e.b32) ? &e : nullptr;
}

void LoraAdapter::apply_lora(const Tensor& B, const Tensor& A, float scaling,
                             const float* x, float* y,
                             int64_t out_dim, int64_t in_dim) {
    const int64_t r = A.dim(0);
    const uint16_t* Ap = A.f16();
    const uint16_t* Bp = B.f16();

    // z[j] = A[j]·x, HOISTED out of the per-output-row loop: O(in*rank +
    // out*rank) instead of re-walking A for every output row — on the 0.1B
    // model (out = in = 4096 ffn) that is ~250x fewer fp16 reads per linear.
    // Plain fp32 math, no SIMD needed (rank*2 reads per output vs the
    // ternary base's in_dim). Zero-skip on f16 elements: trained adapters
    // (and step-0 B) are sparse.
    float z[kMaxStackRank];
    std::vector<float> zheap;
    if (r > kMaxStackRank) zheap.assign(static_cast<size_t>(r), 0.0f);
    float* zp = r > kMaxStackRank ? zheap.data() : z;
    for (int64_t j = 0; j < r; ++j) {
        const uint16_t* arow = Ap + j * in_dim;
        float dot = 0.0f;
        for (int64_t c = 0; c < in_dim; ++c) {
            const float av = half_to_float(arow[c]);
            if (av != 0.0f) dot += av * x[c];
        }
        zp[j] = dot;
    }

    // y[i] += scaling * sum_j B[i,j] * z[j]
    for (int64_t i = 0; i < out_dim; ++i) {
        const uint16_t* brow = Bp + i * r;
        float acc = 0.0f;
        for (int64_t j = 0; j < r; ++j) {
            const float bv = half_to_float(brow[j]);
            if (bv != 0.0f) acc += bv * zp[j];
        }
        y[i] += scaling * acc;
    }
}

} // namespace omniseed
