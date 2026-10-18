// =============================================================================
//  OmniSeed — src/server/ws.cpp  (T103)
//
//  RFC 6455 primitives. Self-contained: SHA-1 + base64 + accept-key + framing.
//  Verified against the RFC's own worked examples in tests/test_server_ws.cpp.
// =============================================================================
#include "omniseed/server/ws.h"

#include <cstring>

namespace omniseed {
namespace server {
namespace ws {

namespace {

const char kGuid[] = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

struct Sha1Ctx {
    uint32_t h[5];
    uint64_t total_bits;
    unsigned char block[64];
    size_t block_len;
};

inline uint32_t rol(uint32_t v, int n) {
    return (v << n) | (v >> (32 - n));
}

void sha1_process(Sha1Ctx& c, const unsigned char* p) {
    uint32_t w[80];
    for (int i = 0; i < 16; ++i) {
        w[i] = (uint32_t(p[i * 4]) << 24) | (uint32_t(p[i * 4 + 1]) << 16) |
               (uint32_t(p[i * 4 + 2]) << 8) | uint32_t(p[i * 4 + 3]);
    }
    for (int i = 16; i < 80; ++i) {
        w[i] = rol(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
    }
    uint32_t a = c.h[0], b = c.h[1], cc = c.h[2], d = c.h[3], e = c.h[4];
    for (int i = 0; i < 80; ++i) {
        uint32_t f, k;
        if (i < 20) {
            f = (b & cc) | ((~b) & d);
            k = 0x5A827999u;
        } else if (i < 40) {
            f = b ^ cc ^ d;
            k = 0x6ED9EBA1u;
        } else if (i < 60) {
            f = (b & cc) | (b & d) | (cc & d);
            k = 0x8F1BBCDCu;
        } else {
            f = b ^ cc ^ d;
            k = 0xCA62C1D6u;
        }
        const uint32_t t = rol(a, 5) + f + e + k + w[i];
        e = d;
        d = cc;
        cc = rol(b, 30);
        b = a;
        a = t;
    }
    c.h[0] += a;
    c.h[1] += b;
    c.h[2] += cc;
    c.h[3] += d;
    c.h[4] += e;
}

void sha1_init(Sha1Ctx& c) {
    c.h[0] = 0x67452301u;
    c.h[1] = 0xEFCDAB89u;
    c.h[2] = 0x98BADCFEu;
    c.h[3] = 0x10325476u;
    c.h[4] = 0xC3D2E1F0u;
    c.total_bits = 0;
    c.block_len = 0;
}

void sha1_update(Sha1Ctx& c, const char* data, size_t len) {
    c.total_bits += uint64_t(len) * 8u;
    size_t i = 0;
    while (i < len) {
        const size_t take_n = std::min<size_t>(64 - c.block_len, len - i);
        std::memcpy(c.block + c.block_len, data + i, take_n);
        c.block_len += take_n;
        i += take_n;
        if (c.block_len == 64) {
            sha1_process(c, c.block);
            c.block_len = 0;
        }
    }
}

std::string sha1_finish(Sha1Ctx& c) {
    const uint64_t bits = c.total_bits;
    const unsigned char pad = 0x80u;
    sha1_update(c, reinterpret_cast<const char*>(&pad), 1);
    const unsigned char zero = 0x00u;
    while (c.block_len != 56) {
        sha1_update(c, reinterpret_cast<const char*>(&zero), 1);
    }
    unsigned char len_be[8];
    for (int i = 0; i < 8; ++i) {
        len_be[i] = static_cast<unsigned char>((bits >> (56 - 8 * i)) & 0xFFu);
    }
    // total_bits already counted the padding; length field uses `bits`.
    sha1_update(c, reinterpret_cast<const char*>(len_be), 8);
    std::string out(20, '\0');
    for (int i = 0; i < 5; ++i) {
        out[i * 4 + 0] = static_cast<char>((c.h[i] >> 24) & 0xFFu);
        out[i * 4 + 1] = static_cast<char>((c.h[i] >> 16) & 0xFFu);
        out[i * 4 + 2] = static_cast<char>((c.h[i] >> 8) & 0xFFu);
        out[i * 4 + 3] = static_cast<char>(c.h[i] & 0xFFu);
    }
    return out;
}

const char kB64[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

}  // namespace

std::string sha1_raw(const std::string& data) {
    Sha1Ctx c;
    sha1_init(c);
    sha1_update(c, data.data(), data.size());
    return sha1_finish(c);
}

std::string sha1_hex(const std::string& data) {
    static const char* hex = "0123456789abcdef";
    const std::string raw = sha1_raw(data);
    std::string out;
    out.reserve(40);
    for (unsigned char ch : raw) {
        out.push_back(hex[ch >> 4]);
        out.push_back(hex[ch & 0x0F]);
    }
    return out;
}

std::string base64_encode(const std::string& data) {
    std::string out;
    out.reserve(((data.size() + 2) / 3) * 4);
    size_t i = 0;
    while (i + 3 <= data.size()) {
        const uint32_t n = (uint32_t(static_cast<unsigned char>(data[i])) << 16) |
                           (uint32_t(static_cast<unsigned char>(data[i + 1])) << 8) |
                           uint32_t(static_cast<unsigned char>(data[i + 2]));
        out.push_back(kB64[(n >> 18) & 63]);
        out.push_back(kB64[(n >> 12) & 63]);
        out.push_back(kB64[(n >> 6) & 63]);
        out.push_back(kB64[n & 63]);
        i += 3;
    }
    const size_t rem = data.size() - i;
    if (rem == 1) {
        const uint32_t n = uint32_t(static_cast<unsigned char>(data[i])) << 16;
        out.push_back(kB64[(n >> 18) & 63]);
        out.push_back(kB64[(n >> 12) & 63]);
        out.push_back('=');
        out.push_back('=');
    } else if (rem == 2) {
        const uint32_t n =
            (uint32_t(static_cast<unsigned char>(data[i])) << 16) |
            (uint32_t(static_cast<unsigned char>(data[i + 1])) << 8);
        out.push_back(kB64[(n >> 18) & 63]);
        out.push_back(kB64[(n >> 12) & 63]);
        out.push_back(kB64[(n >> 6) & 63]);
        out.push_back('=');
    }
    return out;
}

std::string accept_key(const std::string& client_key) {
    return base64_encode(sha1_raw(client_key + kGuid));
}

std::string encode_frame(uint8_t opcode, const std::string& payload) {
    std::string out;
    const uint64_t n = payload.size();
    out.reserve(n + 10);
    out.push_back(static_cast<char>(0x80u | (opcode & 0x0Fu)));
    if (n < 126) {
        out.push_back(static_cast<char>(n));
    } else if (n <= 0xFFFFu) {
        out.push_back(static_cast<char>(126));
        out.push_back(static_cast<char>((n >> 8) & 0xFFu));
        out.push_back(static_cast<char>(n & 0xFFu));
    } else {
        out.push_back(static_cast<char>(127));
        for (int i = 7; i >= 0; --i) {
            out.push_back(static_cast<char>((n >> (8 * i)) & 0xFFu));
        }
    }
    out += payload;
    return out;
}

ParseResult FrameParser::feed(const char* data, size_t len) {
    if (have_) return ParseResult::kOk;  // caller must take() first
    buf_.append(data, len);
    for (;;) {
        if (buf_.size() < 2) return ParseResult::kNeedMore;
        const uint8_t b0 = static_cast<uint8_t>(buf_[0]);
        const uint8_t b1 = static_cast<uint8_t>(buf_[1]);
        if (b0 & 0x70u) return ParseResult::kProtocolError;  // RSV bits
        const bool fin = (b0 & 0x80u) != 0;
        const uint8_t opcode = static_cast<uint8_t>(b0 & 0x0Fu);
        if (opcode == kContinuation) {
            return ParseResult::kProtocolError;  // fragmentation unsupported
        }
        if (opcode != kText && opcode != kBinary && opcode != kClose &&
            opcode != kPing && opcode != kPong) {
            return ParseResult::kProtocolError;
        }
        const bool masked = (b1 & 0x80u) != 0;
        if (!masked) return ParseResult::kProtocolError;  // clients MUST mask
        uint64_t payload_len = static_cast<uint64_t>(b1 & 0x7Fu);
        size_t pos = 2;
        if (payload_len == 126) {
            if (buf_.size() < pos + 2) return ParseResult::kNeedMore;
            payload_len = (uint64_t(static_cast<uint8_t>(buf_[2])) << 8) |
                          uint64_t(static_cast<uint8_t>(buf_[3]));
            pos += 2;
        } else if (payload_len == 127) {
            if (buf_.size() < pos + 8) return ParseResult::kNeedMore;
            payload_len = 0;
            for (int i = 0; i < 8; ++i) {
                payload_len = (payload_len << 8) |
                              uint64_t(static_cast<uint8_t>(buf_[2 + i]));
            }
            pos += 8;
        }
        if (payload_len > kMaxPayload) return ParseResult::kProtocolError;
        if (buf_.size() < pos + 4 + payload_len) return ParseResult::kNeedMore;
        const unsigned char* mask =
            reinterpret_cast<const unsigned char*>(buf_.data() + pos);
        pos += 4;
        next_.opcode = opcode;
        next_.fin = fin;
        next_.payload.assign(payload_len, '\0');
        for (uint64_t i = 0; i < payload_len; ++i) {
            next_.payload[static_cast<size_t>(i)] =
                static_cast<char>(static_cast<uint8_t>(buf_[pos + i]) ^
                                  mask[i % 4]);
        }
        buf_.erase(0, pos + payload_len);
        have_ = true;
        return ParseResult::kOk;
    }
}

bool FrameParser::take(Frame& out) {
    if (!have_) return false;
    out = next_;
    have_ = false;
    next_ = Frame();
    return true;
}

}  // namespace ws
}  // namespace server
}  // namespace omniseed
