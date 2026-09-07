// =============================================================================
//  OmniSeed — prefix_cache.cpp
//  See prefix_cache.h. Binary layout (little-endian fields, written with the
//  same byte-exact discipline as the GGUF reader):
//
//    u32 magic 'WKV1'
//    u32 n_layers
//    i64 tokens_seen
//    per layer:
//      tmix_state: n_embd f32
//      ffn_state:  n_embd f32
//      attn_state: n_heads * head_size * head_size f32
//
//  All floats written as raw IEEE-754 LE via memcpy of the bit pattern (the
//  file is only read back by the same build family; GGUF parity is not
//  required for a local cache).
// =============================================================================
#include "omniseed/memory/prefix_cache.h"
#include "omniseed/core/platform.h"

#include <algorithm>
#include <cstring>
#include <ctime>

namespace omniseed {

namespace {

constexpr uint32_t kMagic = 0x31564B57u;   // 'WKV1' little-endian

void put_u32(std::vector<uint8_t>& b, uint32_t v) {
    b.push_back(static_cast<uint8_t>(v));
    b.push_back(static_cast<uint8_t>(v >> 8));
    b.push_back(static_cast<uint8_t>(v >> 16));
    b.push_back(static_cast<uint8_t>(v >> 24));
}
void put_i64(std::vector<uint8_t>& b, int64_t v) {
    for (int i = 0; i < 8; ++i) b.push_back(static_cast<uint8_t>(static_cast<uint64_t>(v) >> (8 * i)));
}
void put_f32(std::vector<uint8_t>& b, float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    put_u32(b, bits);
}
void put_f32_vec(std::vector<uint8_t>& b, const float* p, size_t n) {
    for (size_t i = 0; i < n; ++i) put_f32(b, p[i]);
}

struct Reader {
    const uint8_t* p;
    size_t n;
    size_t off = 0;
    bool ok = true;
    bool take(uint8_t* dst, size_t bytes) {
        if (!ok || off + bytes > n) { ok = false; return false; }
        std::memcpy(dst, p + off, bytes);
        off += bytes;
        return true;
    }
    bool u32(uint32_t& v) { return take(reinterpret_cast<uint8_t*>(&v), 4); }
    bool i64(int64_t& v) { return take(reinterpret_cast<uint8_t*>(&v), 8); }
    bool f32(float& v)   { return take(reinterpret_cast<uint8_t*>(&v), 4); }
    bool f32_vec(float* dst, size_t count) {
        for (size_t i = 0; i < count; ++i) if (!f32(dst[i])) return false;
        return true;
    }
};

uint64_t now_unix() {
    return static_cast<uint64_t>(std::time(nullptr));
}

} // namespace

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------
std::vector<uint8_t> PrefixCache::serialize(const RwkvModel& model,
                                            const RwkvState& st,
                                            int64_t tokens_seen) {
    std::vector<uint8_t> b;
    b.reserve(1 << 20);
    put_u32(b, kMagic);
    put_u32(b, static_cast<uint32_t>(model.config().n_layers));
    put_i64(b, tokens_seen);
    for (const Tensor& t : st.tmix_state)  put_f32_vec(b, t.f32(), static_cast<size_t>(t.numel()));
    for (const Tensor& t : st.ffn_state)   put_f32_vec(b, t.f32(), static_cast<size_t>(t.numel()));
    for (const Tensor& t : st.attn_state)  put_f32_vec(b, t.f32(), static_cast<size_t>(t.numel()));
    return b;
}

bool PrefixCache::deserialize(const RwkvModel& model,
                              const std::vector<uint8_t>& blob,
                              RwkvState& st, int64_t& tokens_seen) {
    Reader r{blob.data(), blob.size()};
    uint32_t magic = 0, n_layers = 0;
    if (!r.u32(magic) || magic != kMagic) return false;
    if (!r.u32(n_layers)) return false;
    if (n_layers != static_cast<uint32_t>(model.config().n_layers)) return false;

    st.tmix_state.clear();
    st.ffn_state.clear();
    st.attn_state.clear();
    st.tmix_state.resize(n_layers);
    st.ffn_state.resize(n_layers);
    st.attn_state.resize(n_layers);

    const int64_t E = model.config().n_embd;
    const int64_t H = model.config().n_heads;
    const int64_t D = model.config().head_size;
    const int64_t attn_elems = H * D * D;

    for (uint32_t l = 0; l < n_layers; ++l) {
        st.tmix_state[l] = Tensor("tmix", {E}, DType::F32);
        st.ffn_state[l]  = Tensor("ffn",  {E}, DType::F32);
        st.attn_state[l] = Tensor("attn", {H, D, D}, DType::F32);
        if (!r.f32_vec(st.tmix_state[l].f32(), static_cast<size_t>(E)) ||
            !r.f32_vec(st.ffn_state[l].f32(),  static_cast<size_t>(E)) ||
            !r.f32_vec(st.attn_state[l].f32(), static_cast<size_t>(attn_elems)))
            return false;
    }
    if (!r.i64(tokens_seen)) return false;
    st.tokens_seen = tokens_seen;
    return r.off == blob.size();   // exact-length blobs only
}

// ---------------------------------------------------------------------------
// Store / load (LRU)
// ---------------------------------------------------------------------------
bool PrefixCache::store(const std::string& key, const RwkvModel& model,
                        const RwkvState& state) {
    if (key.empty() || !model.valid()) return false;
    std::vector<uint8_t> blob = serialize(model, state, state.tokens_seen);
    const size_t blob_bytes = blob.size();

    // Replace existing entry in place (keeps its position).
    for (Entry& e : entries_) {
        if (e.key == key) {
            bytes_ -= e.blob.size();
            e.blob = std::move(blob);
            e.tokens_seen = state.tokens_seen;
            e.last_use = now_unix();
            bytes_ += blob_bytes;
            return true;
        }
    }

    if (entries_.size() >= cfg_.max_snapshots) {
        bytes_ -= entries_.front().blob.size();
        entries_.erase(entries_.begin());
    }
    Entry e;
    e.key = key;
    e.blob = std::move(blob);
    e.tokens_seen = state.tokens_seen;
    e.last_use = now_unix();
    bytes_ += blob_bytes;
    entries_.push_back(std::move(e));
    return true;
}

bool PrefixCache::load(const std::string& key, const RwkvModel& model,
                       RwkvState& state, int64_t& tokens_seen) const {
    for (const Entry& e : entries_) {
        if (e.key == key) {
            if (!deserialize(model, e.blob, state, tokens_seen)) return false;
            return true;
        }
    }
    return false;
}

bool PrefixCache::has(const std::string& key) const {
    for (const Entry& e : entries_) if (e.key == key) return true;
    return false;
}

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------
bool PrefixCache::save(const std::string& path) const {
    std::FILE* f = platform::open_file_c(path.c_str(), "wb");
    if (f == nullptr) return false;
    bool ok = true;
    // file header: magic 'PFXC', u32 count
    const uint32_t file_magic = 0x43584650u;   // 'PFXC'
    const uint8_t hdr[8] = {
        static_cast<uint8_t>(file_magic), static_cast<uint8_t>(file_magic >> 8),
        static_cast<uint8_t>(file_magic >> 16), static_cast<uint8_t>(file_magic >> 24),
        static_cast<uint8_t>(entries_.size()), static_cast<uint8_t>(entries_.size() >> 8),
        static_cast<uint8_t>(entries_.size() >> 16), static_cast<uint8_t>(entries_.size() >> 24),
    };
    ok = ok && std::fwrite(hdr, 1, sizeof(hdr), f) == sizeof(hdr);
    for (const Entry& e : entries_) {
        const uint32_t klen = static_cast<uint32_t>(e.key.size());
        uint8_t kb[4] = {static_cast<uint8_t>(klen), static_cast<uint8_t>(klen >> 8),
                         static_cast<uint8_t>(klen >> 16), static_cast<uint8_t>(klen >> 24)};
        ok = ok && std::fwrite(kb, 1, 4, f) == 4;
        ok = ok && std::fwrite(e.key.data(), 1, klen, f) == klen;
        uint8_t tb[8];
        for (int i = 0; i < 8; ++i)
            tb[i] = static_cast<uint8_t>(static_cast<uint64_t>(e.tokens_seen) >> (8 * i));
        ok = ok && std::fwrite(tb, 1, 8, f) == 8;
        const uint32_t blen = static_cast<uint32_t>(e.blob.size());
        uint8_t bb[4] = {static_cast<uint8_t>(blen), static_cast<uint8_t>(blen >> 8),
                         static_cast<uint8_t>(blen >> 16), static_cast<uint8_t>(blen >> 24)};
        ok = ok && std::fwrite(bb, 1, 4, f) == 4;
        ok = ok && (blen == 0 || std::fwrite(e.blob.data(), 1, blen, f) == blen);
        if (!ok) break;
    }
    std::fclose(f);
    return ok;
}

bool PrefixCache::load(const std::string& path, const RwkvModel& model) {
    std::FILE* f = platform::open_file_c(path.c_str(), "rb");
    if (f == nullptr) return false;
    uint8_t hdr[8];
    if (std::fread(hdr, 1, 8, f) != 8) { std::fclose(f); return false; }
    const uint32_t file_magic = static_cast<uint32_t>(hdr[0]) |
        (static_cast<uint32_t>(hdr[1]) << 8) | (static_cast<uint32_t>(hdr[2]) << 16) |
        (static_cast<uint32_t>(hdr[3]) << 24);
    if (file_magic != 0x43584650u) { std::fclose(f); return false; }
    const uint32_t count = static_cast<uint32_t>(hdr[4]) | (static_cast<uint32_t>(hdr[5]) << 8) |
        (static_cast<uint32_t>(hdr[6]) << 16) | (static_cast<uint32_t>(hdr[7]) << 24);

    entries_.clear();
    bytes_ = 0;
    for (uint32_t i = 0; i < count; ++i) {
        uint8_t kb[4];
        if (std::fread(kb, 1, 4, f) != 4) { std::fclose(f); return false; }
        const uint32_t klen = kb[0] | (kb[1] << 8) | (kb[2] << 16) | (static_cast<uint32_t>(kb[3]) << 24);
        Entry e;
        e.key.resize(klen);
        if (klen > 0 && std::fread(e.key.data(), 1, klen, f) != klen) { std::fclose(f); return false; }
        uint8_t tb[8];
        if (std::fread(tb, 1, 8, f) != 8) { std::fclose(f); return false; }
        uint64_t ts = 0;
        for (int j = 0; j < 8; ++j) ts |= static_cast<uint64_t>(tb[j]) << (8 * j);
        e.tokens_seen = static_cast<int64_t>(ts);
        uint8_t bb[4];
        if (std::fread(bb, 1, 4, f) != 4) { std::fclose(f); return false; }
        const uint32_t blen = bb[0] | (bb[1] << 8) | (bb[2] << 16) | (static_cast<uint32_t>(bb[3]) << 24);
        e.blob.resize(blen);
        if (blen > 0 && std::fread(e.blob.data(), 1, blen, f) != blen) { std::fclose(f); return false; }
        bytes_ += blen;
        entries_.push_back(std::move(e));
    }
    std::fclose(f);
    return true;
}

} // namespace omniseed
