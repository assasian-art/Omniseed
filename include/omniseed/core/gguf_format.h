// =============================================================================
//  OmniSeed — gguf_format.h  (INTERNAL — not installed)
//  GGUF binary format constants + little-endian primitive readers.
//  Shared by src/core/gguf_loader.cpp and src/core/tokenizer.cpp.
// =============================================================================
#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace omniseed {
namespace gguf {

// ---------------------------------------------------------------------------
// Format constants
// ---------------------------------------------------------------------------
constexpr uint32_t GGUF_MAGIC = 0x46554747;  // 'GGUF' little-endian

// GGUF metadata value types (spec v2/v3).
enum class GgufType : int32_t {
    UINT8   = 0,
    INT8    = 1,
    UINT16  = 2,
    INT16   = 3,
    UINT32  = 4,
    INT32   = 5,
    FLOAT32 = 6,
    BOOL    = 7,
    STRING  = 8,
    ARRAY   = 9,
    UINT64  = 10,
    INT64   = 11,
    FLOAT64 = 12,
};

// GGUF tensor storage dtypes (subset we support; matches DType where 1:1).
enum class GgufDType : int32_t {
    F32     = 0,
    F16     = 1,
    Q4_0    = 2,   // not used by OmniSeed kernels
    Q8_0    = 3,   // not used by OmniSeed kernels
    I8      = 4,
    I32     = 5,
    // 40+ are reserved for vendor formats; OmniSeed registers TERNARY at 40.
    TERNARY = 40,  // BitNet b1.58 packed, 2 weights/byte (OmniSeed-specific)
};

// ---------------------------------------------------------------------------
// Metadata value (single scalar or string; arrays kept as raw bytes)
// ---------------------------------------------------------------------------
struct MetadataValue {
    GgufType type = GgufType::UINT8;

    // scalar payloads
    uint64_t u64 = 0;
    int64_t  i64 = 0;
    double   f64 = 0.0;
    bool     b   = false;

    std::string str;

    // array payload: element type + count + raw bytes
    GgufType  elem_type  = GgufType::UINT8;
    uint64_t  elem_count = 0;
    std::vector<uint8_t> raw;
};

// ---------------------------------------------------------------------------
// Little-endian readers (x86/ARM are LE; this also documents intent)
// ---------------------------------------------------------------------------
inline uint16_t read_u16(const uint8_t* p, size_t off) {
    uint16_t v;
    std::memcpy(&v, p + off, 2);
    return v;
}
inline uint32_t read_u32(const uint8_t* p, size_t off) {
    uint32_t v;
    std::memcpy(&v, p + off, 4);
    return v;
}
inline uint64_t read_u64(const uint8_t* p, size_t off) {
    uint64_t v;
    std::memcpy(&v, p + off, 8);
    return v;
}
inline int32_t  read_i32(const uint8_t* p, size_t off) {
    return static_cast<int32_t>(read_u32(p, off));
}
inline float read_f32(const uint8_t* p, size_t off) {
    float v;
    std::memcpy(&v, p + off, 4);
    return v;
}
inline double read_f64(const uint8_t* p, size_t off) {
    double v;
    std::memcpy(&v, p + off, 8);
    return v;
}

// GGUF string: uint64 length + bytes (no NUL).
inline std::string read_string(const uint8_t* p, size_t& cursor) {
    const uint64_t len = read_u64(p, cursor);
    cursor += 8;
    std::string s(reinterpret_cast<const char*>(p + cursor),
                  static_cast<size_t>(len));
    cursor += static_cast<size_t>(len);
    return s;
}

// ggml tensor dims in GGUF are stored reversed; OmniSeed uses row-major
// [rows, cols] so we flip during load.
inline int64_t ggml_dim(size_t index, const int64_t* ne, int32_t n_dims) {
    const int64_t src = ne[static_cast<size_t>(n_dims - 1 - static_cast<int32_t>(index))];
    return src;
}

} // namespace gguf
} // namespace omniseed
