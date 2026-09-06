// =============================================================================
//  OmniSeed — tensor.h
//  Lightweight tensor struct: the single currency of data inside the kernel.
//
//  Design goals (from the blueprint):
//    * No external deps (no PyTorch/ggml): plain owned heap buffer.
//    * Zero-copy views over mmap'd GGUF weight data (does_not_own flag).
//    * fp32 for activations, fp16 for storage, int8/int4/ternary for weights.
//    * Small enough to be copied cheaply; big data always zero-copy views.
//
//  All buffers are 64-byte aligned for SIMD-friendly kernels later.
// =============================================================================
#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace omniseed {

// ---------------------------------------------------------------------------
// Data types (subset of GGUF codes so the loader can switch directly)
// ---------------------------------------------------------------------------
enum class DType : int32_t {
    F32  = 0,     // 4-byte float, activations & master weights
    F16  = 1,     // 2-byte half, storage format
    I8   = 2,     // signed byte (scales, KV, int8 kernels)
    I32  = 3,     // indices, token ids
    TERNARY = 4,  // BitNet 1.58-bit: 2 weights/byte, sign-magnitude packed
};

size_t dtype_size(DType t);
const char* dtype_name(DType t);

// ---------------------------------------------------------------------------
// fp16 <-> fp32 conversion (IEEE 754 half, no F16C dependency)
// ---------------------------------------------------------------------------
inline float half_to_float(uint16_t h) {
    const uint32_t sign = (h & 0x8000u) << 16u;
    const uint32_t exp  = (h & 0x7C00u) >> 10u;
    const uint32_t man  =  h & 0x03FFu;

    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {                      // +-0
            bits = sign;
        } else {                             // subnormal -> normalize
            uint32_t e = 0;
            uint32_t m = man;
            while ((m & 0x0400u) == 0) { m <<= 1; ++e; }
            m &= 0x03FFu;
            bits = sign | ((127u - 15u - e + 1u) << 23u) | (m << 13u);
        }
    } else if (exp == 0x1F) {                // inf / NaN
        bits = sign | 0x7F800000u | (man << 13u);
    } else {                                 // normal
        bits = sign | ((exp - 15u + 127u) << 23u) | (man << 13u);
    }

    float out;
    std::memcpy(&out, &bits, 4);
    return out;
}

inline uint16_t float_to_half(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, 4);

    const uint32_t sign = (bits >> 16u) & 0x8000u;
    const int32_t  exp  = static_cast<int32_t>((bits >> 23u) & 0xFFu) - 127 + 15;
    const uint32_t man  = bits & 0x007FFFFFu;

    if (((bits >> 23u) & 0xFFu) == 0xFF)                    // inf / NaN
        return static_cast<uint16_t>(sign | 0x7C00u | (man >> 13u));
    if (exp >= 0x1F)                                        // overflow -> inf
        return static_cast<uint16_t>(sign | 0x7C00u);
    if (exp <= 0) {                                         // subnormal / 0
        if (exp < -10) return static_cast<uint16_t>(sign);
        const uint32_t m = man | 0x00800000u;
        const uint32_t shift = static_cast<uint32_t>(14 - exp);
        const uint32_t half_man = m >> shift;
        const uint32_t r = (m >> (shift - 1u)) & 1u;        // round-to-nearest
        return static_cast<uint16_t>(sign | ((half_man + r) & 0x03FFu));
    }
    // normal with rounding
    const uint32_t half = sign | (static_cast<uint32_t>(exp) << 10u) |
                          ((man + 0x00001000u) >> 13u);
    return static_cast<uint16_t>(half);
}

// ---------------------------------------------------------------------------
// Tensor
// ---------------------------------------------------------------------------
class Tensor {
public:
    Tensor() = default;

    // Owning tensor: allocates aligned storage, zero-initialized.
    Tensor(const std::string& name, const std::vector<int64_t>& shape,
           DType dtype = DType::F32);

    // Non-owning view over external memory (mmap'd GGUF data). Does NOT copy.
    Tensor(const std::string& name, const std::vector<int64_t>& shape,
           DType dtype, void* external_data);

    // Rule of five: owning buffers must not be double-freed.
    Tensor(const Tensor&)            = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    ~Tensor();

    // -- shape / metadata ----------------------------------------------------
    const std::string& name()  const { return name_; }
    const std::vector<int64_t>& shape() const { return shape_; }
    DType dtype() const { return dtype_; }
    int64_t dim(int i) const { return shape_[static_cast<size_t>(i)]; }
    int64_t rows() const { return shape_.size() == 1 ? 1 : shape_[0]; }
    int64_t cols() const { return shape_.empty() ? 0 : shape_.back(); }

    // Number of logical elements (nelements).
    int64_t numel() const;
    // Bytes needed for logical elements of this dtype (before padding).
    size_t  nbytes() const;

    bool owns_data() const { return owns_data_; }

    // Raw byte access (dtype-agnostic; used by cast/copy and the GGUF loader).
    void*        data()       { return data_; }
    const void*  data() const { return data_; }

    // -- typed accessors (caller must match dtype) -----------------------------
    float*     f32();
    const float* f32() const;
    uint16_t*  f16();
    const uint16_t* f16() const;
    int8_t*    i8();
    const int8_t* i8() const;
    int32_t*   i32();
    const int32_t* i32() const;
    uint8_t*   packed();                    // TERNARY raw bytes
    const uint8_t* packed() const;

    // -- elementwise helpers ---------------------------------------------------
    void zero();
    // y = x (dtype-preserving cast where needed); sizes must match.
    void copy_from(const Tensor& src);

    // -- fp16 helpers (used by whisper/RWKV storage paths) ----------------------
    // Reads element i as float, converting from the storage dtype.
    float  get_as_float(int64_t i) const;

private:
    void alloc_owned();

    std::string          name_;
    std::vector<int64_t> shape_;
    DType                dtype_ = DType::F32;
    void*                data_  = nullptr;
    bool                 owns_data_ = true;
};

// ---------------------------------------------------------------------------
// Operations (stateless free functions in tensor_ops namespace)
// ---------------------------------------------------------------------------
namespace tensor_ops {

// Elementwise cast; allocates a new tensor.
Tensor cast(const Tensor& src, DType dst);

// y[i] += a[i] elementwise (same shape required).
void add_inplace(Tensor& y, const Tensor& a);

// y[i] *= a[i] elementwise (same shape required).
void mul_inplace(Tensor& y, const Tensor& a);

// Elementwise sigmoid in place (fp32 only).
void sigmoid_inplace(Tensor& t);

// softmax(x + bias) over the last axis (fp32). bias may be empty.
void softmax_last_dim(Tensor& t, const Tensor* bias);

// L2-normalize a fp32 vector.
void l2_normalize(Tensor& t);

// GeLU (tanh approximation) in place, fp32 only.
void gelu_inplace(Tensor& t);

// RMSNorm: y = x / rms(x) * weight  (fp32). weight may be nullptr.
void rms_norm(const Tensor& x, const Tensor* weight, Tensor& y);

// LayerNorm with optional bias (fp32).
void layer_norm(const Tensor& x, const Tensor* weight, const Tensor* bias,
                Tensor& y, float eps = 1e-5f);

} // namespace tensor_ops

} // namespace omniseed
