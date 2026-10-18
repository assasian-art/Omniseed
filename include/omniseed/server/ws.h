// =============================================================================
//  OmniSeed — include/omniseed/server/ws.h  (T103)
//
//  Minimal RFC 6455 WebSocket primitives, zero external deps.
//  Pure functions + an incremental parser; socket I/O stays in main.cpp.
//
//  Covered here: SHA-1 (FIPS 180-1), RFC 4648 base64, the Sec-WebSocket-Accept
//  key derivation, server->client frame encoding (unmasked) and a strict
//  client->server frame parser (mask REQUIRED, no fragmentation, payload cap).
// =============================================================================
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace omniseed {
namespace server {
namespace ws {

// SHA-1 digest as lowercase hex (test vectors: "", "abc").
std::string sha1_hex(const std::string& data);
// Raw 20-byte SHA-1 digest.
std::string sha1_raw(const std::string& data);
// RFC 4648 base64 (no newlines).
std::string base64_encode(const std::string& data);
// Sec-WebSocket-Accept = base64(sha1(key + RFC6455 magic GUID)).
std::string accept_key(const std::string& client_key);

// RFC 6455 §5.2 opcodes.
enum Opcode : uint8_t {
    kContinuation = 0x0,
    kText         = 0x1,
    kBinary       = 0x2,
    kClose        = 0x8,
    kPing         = 0x9,
    kPong         = 0xA,
};

// Largest payload this server accepts from a client (1 MiB); bigger -> protocol
// error. Console events are tiny JSON; the cap is a DoS guard.
constexpr uint64_t kMaxPayload = 1u << 20;

// Server->client frame (FIN=1, unmasked) — RFC 6455 §5.1 length encoding
// (7-bit / 16-bit / 64-bit paths).
std::string encode_frame(uint8_t opcode, const std::string& payload);

struct Frame {
    uint8_t opcode = 0;
    bool fin = true;
    std::string payload;
};

enum class ParseResult { kNeedMore, kOk, kProtocolError };

// Incremental parser for client->server frames. Strict by design:
//   - MASK bit must be set (clients MUST mask; unmasked => protocol error)
//   - RSV bits must be zero (no extensions negotiated)
//   - fragmentation is rejected (kContinuation -> protocol error)
//   - payload > kMaxPayload -> protocol error
class FrameParser {
  public:
    // Feed raw socket bytes. Returns kProtocolError on malformed input
    // (buffer is then unusable); kOk means a frame is ready for take().
    ParseResult feed(const char* data, size_t len);
    // Move out the frame produced by the last kOk feed().
    bool take(Frame& out);
    bool has_frame() const { return have_; }

  private:
    std::string buf_;
    Frame next_;
    bool have_ = false;
};

}  // namespace ws
}  // namespace server
}  // namespace omniseed
