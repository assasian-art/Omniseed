// =============================================================================
//  OmniSeed — tests/test_server_ws.cpp  (T103)
//
//  RFC 6455 primitives, verified against the RFC's own worked examples plus
//  framing edge cases (125 / 126 / 65536 payload boundaries, byte-at-a-time
//  feeds, strict rejection of unmasked / fragmented / oversized frames).
//  Standalone printf harness — no model, no sockets, no deps.
// =============================================================================
#include "omniseed/server/ws.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace omniseed::server::ws;

static int g_passed = 0;
static int g_failed = 0;

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (cond) { ++g_passed; }                                          \
        else {                                                             \
            ++g_failed;                                                    \
            std::printf("FAIL  (line %d): %s\n", __LINE__, #cond);         \
        }                                                                  \
    } while (0)

static std::string make_masked_frame(uint8_t opcode, const std::string& payload,
                                     const unsigned char mask[4]) {
    std::string f;
    f.push_back(static_cast<char>(0x80u | opcode));
    const size_t n = payload.size();
    if (n < 126) {
        f.push_back(static_cast<char>(0x80u | n));
    } else if (n <= 0xFFFFu) {
        f.push_back(static_cast<char>(0x80u | 126));
        f.push_back(static_cast<char>((n >> 8) & 0xFFu));
        f.push_back(static_cast<char>(n & 0xFFu));
    } else {
        f.push_back(static_cast<char>(0x80u | 127));
        for (int i = 7; i >= 0; --i) {
            f.push_back(static_cast<char>((static_cast<uint64_t>(n) >> (8 * i)) & 0xFFu));
        }
    }
    f.append(reinterpret_cast<const char*>(mask), 4);
    for (size_t i = 0; i < n; ++i) {
        f.push_back(static_cast<char>(static_cast<uint8_t>(payload[i]) ^ mask[i % 4]));
    }
    return f;
}

int main() {
    // ---- SHA-1 (FIPS 180-1 / RFC 3174 vectors) -----------------------------
    CHECK(sha1_hex("") ==
          "da39a3ee5e6b4b0d3255bfef95601890afd80709");
    CHECK(sha1_hex("abc") ==
          "a9993e364706816aba3e25717850c26c9cd0d89d");
    CHECK(sha1_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
          "84983e441c3bd26ebaae4aa1f95129e5e54670f1");
    {
        // 80-byte input forces multi-block processing.
        std::string big(1000, 'a');
        CHECK(sha1_hex(big).size() == 40);
        CHECK(sha1_hex(std::string(1000, 'a')) ==
              sha1_hex(std::string(1000, 'a')));  // determinism
        CHECK(sha1_hex(std::string(1000, 'a')) !=
              sha1_hex(std::string(1000, 'b')));  // separation
    }

    // ---- base64 (RFC 4648 §10 vectors) -------------------------------------
    CHECK(base64_encode("") == "");
    CHECK(base64_encode("f") == "Zg==");
    CHECK(base64_encode("fo") == "Zm8=");
    CHECK(base64_encode("foo") == "Zm9v");
    CHECK(base64_encode("foob") == "Zm9vYg==");
    CHECK(base64_encode("fooba") == "Zm9vYmE=");
    CHECK(base64_encode("foobar") == "Zm9vYmFy");

    // ---- Sec-WebSocket-Accept (RFC 6455 §1.3 worked example) ----------------
    CHECK(accept_key("dGhlIHNhbXBsZSBub25jZQ==") ==
          "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=");

    // ---- server->client frame encoding --------------------------------------
    {
        const std::string f = encode_frame(kText, "Hello");
        CHECK(f.size() == 7);
        CHECK(static_cast<uint8_t>(f[0]) == 0x81);  // FIN + text
        CHECK(static_cast<uint8_t>(f[1]) == 0x05);  // len5, unmasked
        CHECK(f.substr(2) == "Hello");
    }
    {
        std::string p125(125, 'x');
        const std::string f = encode_frame(kText, p125);
        CHECK(f.size() == 2 + 125);
        CHECK(static_cast<uint8_t>(f[1]) == 125);
        std::string p126(126, 'y');
        const std::string g = encode_frame(kText, p126);
        CHECK(g.size() == 4 + 126);
        CHECK(static_cast<uint8_t>(g[1]) == 126);
        CHECK(static_cast<uint8_t>(g[2]) == 0 && static_cast<uint8_t>(g[3]) == 126);
    }
    {
        std::string p64k(65536, 'z');
        const std::string f = encode_frame(kBinary, p64k);
        CHECK(f.size() == 10 + 65536);
        CHECK(static_cast<uint8_t>(f[1]) == 127);
        CHECK(static_cast<uint8_t>(f[2]) == 0 && static_cast<uint8_t>(f[9]) == 0);
        // 65536 = 0x0000_0000_0001_0000 -> 0x01 at byte 7 of the 8-byte field.
        CHECK(static_cast<uint8_t>(f[7]) == 0x01 &&
              static_cast<uint8_t>(f[8]) == 0x00 &&
              static_cast<uint8_t>(f[9]) == 0x00);
    }
    {
        const std::string f = encode_frame(kPong, "");
        CHECK(f.size() == 2);
        CHECK(static_cast<uint8_t>(f[0]) == 0x8A);
    }

    // ---- client->server parsing (masked) ------------------------------------
    {
        const unsigned char mask[4] = {0x12, 0x34, 0x56, 0x78};
        const std::string f = make_masked_frame(kText, "Hello", mask);
        FrameParser p;
        Frame fr;
        CHECK(p.feed(f.data(), f.size()) == ParseResult::kOk);
        CHECK(p.take(fr));
        CHECK(fr.opcode == kText && fr.fin);
        CHECK(fr.payload == "Hello");
        CHECK(p.take(fr) == false);
    }
    {
        // Byte-at-a-time feed (kNeedMore until the final byte).
        const unsigned char mask[4] = {9, 8, 7, 6};
        const std::string f = make_masked_frame(kText, "chunk-by-chunk", mask);
        FrameParser p;
        Frame fr;
        bool ok = false;
        for (size_t i = 0; i < f.size(); ++i) {
            const ParseResult r = p.feed(&f[i], 1);
            if (i + 1 < f.size()) CHECK(r == ParseResult::kNeedMore);
            else ok = (r == ParseResult::kOk);
        }
        CHECK(ok);
        CHECK(p.take(fr) && fr.payload == "chunk-by-chunk");
    }
    {
        // 16-bit length path masked (300 bytes).
        std::string p300(300, 'q');
        const unsigned char mask[4] = {1, 2, 3, 4};
        const std::string f = make_masked_frame(kText, p300, mask);
        FrameParser p;
        Frame fr;
        CHECK(p.feed(f.data(), f.size()) == ParseResult::kOk);
        CHECK(p.take(fr) && fr.payload == p300);
    }
    {
        // 64-bit length path masked (70000 bytes).
        std::string p70k(70000, 'k');
        const unsigned char mask[4] = {0xAA, 0xBB, 0xCC, 0xDD};
        const std::string f = make_masked_frame(kBinary, p70k, mask);
        FrameParser p;
        Frame fr;
        CHECK(p.feed(f.data(), f.size()) == ParseResult::kOk);
        CHECK(p.take(fr) && fr.payload == p70k);
        CHECK(fr.opcode == kBinary);
    }
    {
        // Ping -> parsed; server reply is an unmasked pong frame.
        const unsigned char mask[4] = {5, 5, 5, 5};
        const std::string f = make_masked_frame(kPing, "hi", mask);
        FrameParser p;
        Frame fr;
        CHECK(p.feed(f.data(), f.size()) == ParseResult::kOk);
        CHECK(p.take(fr) && fr.opcode == kPing && fr.payload == "hi");
        const std::string pong = encode_frame(kPong, "");
        CHECK(static_cast<uint8_t>(pong[0]) == 0x8A && pong.size() == 2);
    }
    {
        // Close frame parses.
        const unsigned char mask[4] = {0, 1, 2, 3};
        const std::string payload = std::string("\x03\xe8", 2);  // 1000 normal
        const std::string f = make_masked_frame(kClose, payload, mask);
        FrameParser p;
        Frame fr;
        CHECK(p.feed(f.data(), f.size()) == ParseResult::kOk);
        CHECK(p.take(fr) && fr.opcode == kClose && fr.payload == payload);
    }

    // ---- strictness ---------------------------------------------------------
    {
        // Unmasked client frame -> protocol error (clients MUST mask).
        std::string f;
        f.push_back(static_cast<char>(0x81));
        f.push_back(0x02);
        f += "hi";
        FrameParser p;
        CHECK(p.feed(f.data(), f.size()) == ParseResult::kProtocolError);
    }
    {
        // Fragmentation (continuation opcode) -> protocol error.
        const unsigned char mask[4] = {1, 1, 1, 1};
        const std::string f = make_masked_frame(kContinuation, "x", mask);
        FrameParser p;
        CHECK(p.feed(f.data(), f.size()) == ParseResult::kProtocolError);
    }
    {
        // RSV bits set (no extension negotiated) -> protocol error.
        std::string f;
        f.push_back(static_cast<char>(0xC1));  // FIN + RSV1 + text
        const unsigned char mask[4] = {2, 2, 2, 2};
        f += make_masked_frame(kText, "x", mask).substr(1);
        FrameParser p;
        CHECK(p.feed(f.data(), f.size()) == ParseResult::kProtocolError);
    }
    {
        // Payload over the 1 MiB cap -> protocol error (DoS guard).
        std::string f;
        f.push_back(static_cast<char>(0x82));  // FIN + binary
        f.push_back(static_cast<char>(0x80 | 127));
        const uint64_t n = kMaxPayload + 1;
        for (int i = 7; i >= 0; --i) {
            f.push_back(static_cast<char>((n >> (8 * i)) & 0xFFu));
        }
        const unsigned char mask[4] = {3, 3, 3, 3};
        f.append(reinterpret_cast<const char*>(mask), 4);
        FrameParser p;
        CHECK(p.feed(f.data(), f.size()) == ParseResult::kProtocolError);
    }
    {
        // Reserved opcode -> protocol error.
        const unsigned char mask[4] = {4, 4, 4, 4};
        const std::string f = make_masked_frame(0x3, "x", mask);
        FrameParser p;
        CHECK(p.feed(f.data(), f.size()) == ParseResult::kProtocolError);
    }

    std::printf("omniseed_ws: %d passed, %d failed\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
