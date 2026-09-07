// =============================================================================
//  OmniSeed — memory_crystals.cpp
//  Memory Crystals: semantic compression of retired context + retrieval.
// =============================================================================
#include "omniseed/memory/memory.h"
#include "omniseed/core/platform.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <unordered_set>

namespace omniseed {

// ===========================================================================
// Embedding: deterministic bag-of-tokens with hashed projection
// ===========================================================================
void MemoryCrystals::embed_tokens(const std::vector<int32_t>& ids,
                                  float* out) const {
    std::memset(out, 0, sizeof(float) * static_cast<size_t>(cfg_.embed_dim));
    for (const int32_t id : ids) {
        // Fibonacci hashing for stable spread.
        const uint64_t h = (static_cast<uint64_t>(id) * 0x9E3779B97F4A7C15ull);
        const size_t i1 = static_cast<size_t>(h % cfg_.embed_dim);
        const size_t i2 =
            static_cast<size_t>((h >> 17u) % static_cast<uint64_t>(cfg_.embed_dim));
        out[i1] += 1.0f;
        out[i2] += 0.5f;
    }
    // L2 normalize
    double n2 = 0.0;
    for (int32_t i = 0; i < cfg_.embed_dim; ++i)
        n2 += static_cast<double>(out[i]) * out[i];
    const float inv =
        static_cast<float>(1.0 / std::sqrt(n2 + 1e-12));
    for (int32_t i = 0; i < cfg_.embed_dim; ++i) out[i] *= inv;
}

// ===========================================================================
// Crystallize: retired tokens -> compressed crystal record
// ===========================================================================
bool MemoryCrystals::crystallize(const std::vector<int32_t>& retired_tokens,
                                 const Tokenizer& tok, uint64_t stream_pos,
                                 float entropy) {
    if (retired_tokens.size() < 8) return false;   // too small to be useful

    // ---- sentence segmentation on '.'/'!'/'?'/'\n' pieces -------------------
    std::vector<std::vector<int32_t>> sentences(1);
    for (const int32_t id : retired_tokens) {
        const std::string& pc = tok.piece(id);
        const bool terminator =
            pc.find('.') != std::string::npos ||
            pc.find('!') != std::string::npos ||
            pc.find('?') != std::string::npos ||
            pc.find('\n') != std::string::npos;
        sentences.back().push_back(id);
        if (terminator) sentences.emplace_back();
    }
    if (sentences.back().empty()) sentences.pop_back();
    if (sentences.empty()) return false;

    // ---- score sentences: content density + position salience --------------
    // (early context gets recency-anchored weight; the middle gets least,
    //  countering lost-in-the-middle effects)
    auto score = [&](size_t si, const std::vector<int32_t>& s) {
        double sc = 0.0;
        for (const int32_t id : s) {
            const std::string& pc = tok.piece(id);
            if (pc.size() >= 4) sc += 1.0;                    // content words
            if (pc.find('=') != std::string::npos ||
                pc.find(':') != std::string::npos) sc += 2.0; // structured data
            if (!pc.empty() &&
                (std::isupper(static_cast<unsigned char>(pc[0]))))
                sc += 0.3;                                    // names/nums
        }
        sc /= std::max<double>(1.0, static_cast<double>(s.size()));
        const double pos = static_cast<double>(si) /
                           static_cast<double>(sentences.size());
        sc *= 1.0 + 0.5 * (1.0 - std::fabs(pos - 0.5) * 2.0); // ends bonus
        return sc;
    };

    std::vector<size_t> idx(sentences.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
    std::stable_sort(idx.begin(), idx.end(),
                     [&](size_t a, size_t b) {
                         return score(a, sentences[a]) > score(b, sentences[b]);
                     });

    // ---- build the crystal from the best sentences within budget ----------
    MemoryCrystal c;
    c.id = next_id_++;
    c.created_at_token = stream_pos;
    int32_t budget = cfg_.max_len_tokens;
    for (const size_t si : idx) {
        if (budget <= 0) break;
        const auto& s = sentences[si];
        const int32_t take =
            std::min<int32_t>(budget, static_cast<int32_t>(s.size()));
        c.tokens.insert(c.tokens.end(), s.begin(), s.begin() + take);
        budget -= take;
    }
    if (c.tokens.empty()) return false;

    c.summary = tok.decode(c.tokens, false);
    c.importance = static_cast<float>(std::min(
        1.0, score(0, c.tokens) + 0.25));

    // Entropy salience gate (grok M02): high-entropy retired turns carry
    // more information; boost importance (bounded) when the signal exists.
    if (entropy >= 0.0f && cfg_.entropy_boost > 0.0f) {
        // Map entropy in [0, ~8 nats] to a 0..1 boost factor.
        const float e01 = std::min(1.0f, entropy / 8.0f);
        c.importance = std::min(1.0f, c.importance + cfg_.entropy_boost * e01);
        c.salience = e01;
    }

    embed_tokens(c.tokens, c.embedding);

    // ---- capacity: evict lowest importance when full ------------------------
    if (crystals_.size() >= cfg_.max_crystals) {
        auto victim = std::min_element(
            crystals_.begin(), crystals_.end(),
            [](const MemoryCrystal& a, const MemoryCrystal& b) {
                return a.importance < b.importance;
            });
        if (victim != crystals_.end() &&
            victim->importance < c.importance) {
            crystals_.erase(victim);
        } else {
            return false;   // new crystal not worth keeping
        }
    }

    crystals_.push_back(std::move(c));
    return true;
}

// ===========================================================================
// Retrieval: cosine similarity over crystal embeddings (+ recency bookkeeping)
// ===========================================================================
std::vector<MemoryCrystal> MemoryCrystals::retrieve(
        const std::vector<int32_t>& query_tokens, int32_t k) {
    if (crystals_.empty() || k <= 0) return {};

    float q[64];
    embed_tokens(query_tokens, q);

    std::vector<std::pair<float, size_t>> scored;
    scored.reserve(crystals_.size());
    for (size_t i = 0; i < crystals_.size(); ++i) {
        const MemoryCrystal& c = crystals_[i];
        float dot = 0.0f;
        for (int32_t d = 0; d < cfg_.embed_dim; ++d)
            dot += q[d] * c.embedding[d];
        scored.emplace_back(dot * (0.5f + 0.5f * c.importance), i);
    }
    std::partial_sort(scored.begin(),
                      scored.begin() + std::min<size_t>(k, scored.size()),
                      scored.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

    std::vector<MemoryCrystal> out;
    const int32_t take = std::min<int32_t>(k, static_cast<int32_t>(scored.size()));
    for (int32_t i = 0; i < take; ++i) {
        const size_t idx = scored[static_cast<size_t>(i)].second;
        crystals_[idx].hits += 1;                    // Ebbinghaus bookkeeping
        crystals_[idx].last_access_token =           // "recency" in stream time
            crystals_[idx].created_at_token + static_cast<uint64_t>(query_tokens.size());
        out.push_back(crystals_[idx]);
    }
    return out;
}

// ===========================================================================
// Ebbinghaus decay: effective = importance * exp(-age_days / tau)
// ===========================================================================
size_t MemoryCrystals::decay(double now_unix_seconds) {
    if (cfg_.decay_tau_days <= 0.0) return 0;
    size_t dropped = 0;
    for (auto it = crystals_.begin(); it != crystals_.end();) {
        // Age in days since creation (stream position stands in for wall
        // time when the caller does not track wall clock per crystal).
        const double age_days = static_cast<double>(it->last_access_token) /
                                100000.0;   // 100k tokens ≈ 1 day of use
        const double eff = static_cast<double>(it->importance) *
                           std::exp(-age_days / cfg_.decay_tau_days) *
                           (1.0 + 0.1 * std::min<uint32_t>(it->hits, 10));
        if (eff < static_cast<double>(cfg_.min_importance)) {
            it = crystals_.erase(it);
            ++dropped;
        } else {
            ++it;
        }
    }
    return dropped;
}

// ===========================================================================
// Persistence: compact binary sidecar
// ===========================================================================
bool MemoryCrystals::save(const std::string& path) const {
    FILE* f = platform::open_file_c(path.c_str(), "wb");
    if (!f) return false;

    const uint32_t magic = 0x5254434D;   // MCTR
    const uint32_t version = 2;
    const uint32_t n = static_cast<uint32_t>(crystals_.size());
    std::fwrite(&magic, 4, 1, f);
    std::fwrite(&version, 4, 1, f);
    std::fwrite(&cfg_.embed_dim, 4, 1, f);
    std::fwrite(&n, 4, 1, f);

    for (const MemoryCrystal& c : crystals_) {
        std::fwrite(&c.id, 8, 1, f);
        std::fwrite(&c.created_at_token, 8, 1, f);
        std::fwrite(&c.last_access_token, 8, 1, f);
        std::fwrite(&c.importance, 4, 1, f);
        std::fwrite(&c.salience, 4, 1, f);
        std::fwrite(&c.hits, 4, 1, f);
        std::fwrite(c.embedding, 4, static_cast<size_t>(cfg_.embed_dim), f);
        const uint32_t nt = static_cast<uint32_t>(c.tokens.size());
        std::fwrite(&nt, 4, 1, f);
        std::fwrite(c.tokens.data(), 4, nt, f);
        const uint32_t ns = static_cast<uint32_t>(c.summary.size());
        std::fwrite(&ns, 4, 1, f);
        std::fwrite(c.summary.data(), 1, ns, f);
    }
    std::fclose(f);
    return true;
}

bool MemoryCrystals::load(const std::string& path) {
    platform::MappedFile mf;
    if (!mf.open(path)) return false;

    const uint8_t* p = mf.bytes();
    size_t cur = 0;
    if (mf.size() < 16 || std::memcmp(p, "MCTR", 4) != 0) return false;
    cur += 4;
    const uint32_t version = *reinterpret_cast<const uint32_t*>(p + cur); cur += 4;
    if (version != 1 && version != 2) return false;
    const uint32_t dim = *reinterpret_cast<const uint32_t*>(p + cur); cur += 4;
    if (static_cast<int32_t>(dim) != cfg_.embed_dim) return false;
    const uint32_t n = *reinterpret_cast<const uint32_t*>(p + cur); cur += 4;

    crystals_.clear();
    for (uint32_t i = 0; i < n; ++i) {
        MemoryCrystal c;
        std::memcpy(&c.id, p + cur, 8); cur += 8;
        std::memcpy(&c.created_at_token, p + cur, 8); cur += 8;
        if (version >= 2) {
            std::memcpy(&c.last_access_token, p + cur, 8); cur += 8;
            std::memcpy(&c.salience, p + cur, 4); cur += 4;
            std::memcpy(&c.hits, p + cur, 4); cur += 4;
        }
        std::memcpy(&c.importance, p + cur, 4); cur += 4;
        std::memcpy(c.embedding, p + cur, 4 * dim); cur += 4 * dim;
        uint32_t nt; std::memcpy(&nt, p + cur, 4); cur += 4;
        c.tokens.resize(nt);
        std::memcpy(c.tokens.data(), p + cur, 4 * nt); cur += 4 * nt;
        uint32_t ns; std::memcpy(&ns, p + cur, 4); cur += 4;
        c.summary.assign(reinterpret_cast<const char*>(p + cur), ns); cur += ns;
        crystals_.push_back(std::move(c));
    }
    next_id_ = n + 1;
    return true;
}

} // namespace omniseed
