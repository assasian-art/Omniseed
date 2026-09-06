// =============================================================================
//  OmniSeed — tensor.cpp
//  Tensor implementation: aligned storage, fp16 helpers, elementwise ops.
// =============================================================================
#include "omniseed/core/tensor.h"
#include "omniseed/core/platform.h"

#include <cstdlib>

namespace omniseed {

// ---------------------------------------------------------------------------
// DType helpers
// ---------------------------------------------------------------------------
size_t dtype_size(DType t) {
    switch (t) {
        case DType::F32:     return 4;
        case DType::F16:     return 2;
        case DType::I8:      return 1;
        case DType::I32:     return 4;
        case DType::TERNARY: return 1;   // 2 weights per byte = 0.5 B/weight
        default:             return 0;
    }
}

const char* dtype_name(DType t) {
    switch (t) {
        case DType::F32:     return "f32";
        case DType::F16:     return "f16";
        case DType::I8:      return "i8";
        case DType::I32:     return "i32";
        case DType::TERNARY: return "ternary(1.58-bit)";
        default:             return "?";
    }
}

// ---------------------------------------------------------------------------
// Tensor construction / lifetime
// ---------------------------------------------------------------------------
Tensor::Tensor(const std::string& name, const std::vector<int64_t>& shape,
               DType dtype)
    : name_(name), shape_(shape), dtype_(dtype), owns_data_(true) {
    alloc_owned();
}

Tensor::Tensor(const std::string& name, const std::vector<int64_t>& shape,
               DType dtype, void* external_data)
    : name_(name), shape_(shape), dtype_(dtype),
      data_(external_data), owns_data_(false) {
    // Zero-copy view: data stays inside the mmap'd GGUF region.
}

void Tensor::alloc_owned() {
    const size_t bytes = nbytes();
    data_ = platform::aligned_alloc_omni(bytes ? bytes : 1, 64);
    if (data_ == nullptr) {
        platform::log_error("tensor '%s': OOM allocating %zu bytes",
                            name_.c_str(), bytes);
        std::abort();
    }
    std::memset(data_, 0, bytes);
}

Tensor::~Tensor() {
    if (owns_data_ && data_ != nullptr) {
        platform::aligned_free_omni(data_);
    }
    data_ = nullptr;
}

Tensor::Tensor(Tensor&& other) noexcept
    : name_(std::move(other.name_)), shape_(std::move(other.shape_)),
      dtype_(other.dtype_), data_(other.data_),
      owns_data_(other.owns_data_) {
    other.data_ = nullptr;
    other.shape_.clear();
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        if (owns_data_ && data_ != nullptr) platform::aligned_free_omni(data_);
        name_      = std::move(other.name_);
        shape_     = std::move(other.shape_);
        dtype_     = other.dtype_;
        data_      = other.data_;
        owns_data_ = other.owns_data_;
        other.data_ = nullptr;
        other.shape_.clear();
    }
    return *this;
}

int64_t Tensor::numel() const {
    int64_t n = 1;
    for (const int64_t d : shape_) n *= d;
    return n;
}

size_t Tensor::nbytes() const {
    return static_cast<size_t>(numel()) * dtype_size(dtype_);
}

// ---------------------------------------------------------------------------
// Typed accessors
// ---------------------------------------------------------------------------
float* Tensor::f32() {
    assert(dtype_ == DType::F32);
    return static_cast<float*>(data_);
}
const float* Tensor::f32() const {
    assert(dtype_ == DType::F32);
    return static_cast<const float*>(data_);
}
uint16_t* Tensor::f16() {
    assert(dtype_ == DType::F16);
    return static_cast<uint16_t*>(data_);
}
const uint16_t* Tensor::f16() const {
    assert(dtype_ == DType::F16);
    return static_cast<const uint16_t*>(data_);
}
int8_t* Tensor::i8() {
    assert(dtype_ == DType::I8);
    return static_cast<int8_t*>(data_);
}
const int8_t* Tensor::i8() const {
    assert(dtype_ == DType::I8);
    return static_cast<const int8_t*>(data_);
}
int32_t* Tensor::i32() {
    assert(dtype_ == DType::I32);
    return static_cast<int32_t*>(data_);
}
const int32_t* Tensor::i32() const {
    assert(dtype_ == DType::I32);
    return static_cast<const int32_t*>(data_);
}
uint8_t* Tensor::packed() {
    assert(dtype_ == DType::TERNARY);
    return static_cast<uint8_t*>(data_);
}
const uint8_t* Tensor::packed() const {
    assert(dtype_ == DType::TERNARY);
    return static_cast<const uint8_t*>(data_);
}

// ---------------------------------------------------------------------------
// Elementwise helpers
// ---------------------------------------------------------------------------
void Tensor::zero() { std::memset(data_, 0, nbytes()); }

void Tensor::copy_from(const Tensor& src) {
    if (src.numel() != numel()) {
        platform::log_error("tensor copy: numel mismatch %s(%lld) <- %s(%lld)",
                            name_.c_str(), static_cast<long long>(numel()),
                            src.name().c_str(),
                            static_cast<long long>(src.numel()));
        return;
    }
    if (src.dtype() == dtype_) {
        std::memcpy(data_, src.data(), nbytes());
        return;
    }
    Tensor tmp = tensor_ops::cast(src, dtype_);
    std::memcpy(data_, tmp.data(), nbytes());
}

float Tensor::get_as_float(int64_t i) const {
    switch (dtype_) {
        case DType::F32: return f32()[i];
        case DType::F16: return half_to_float(f16()[i]);
        case DType::I8:  return static_cast<float>(i8()[i]);
        case DType::I32: return static_cast<float>(i32()[i]);
        default:         return 0.0f;
    }
}

// ---------------------------------------------------------------------------
// tensor_ops
// ---------------------------------------------------------------------------
namespace tensor_ops {

Tensor cast(const Tensor& src, DType dst) {
    Tensor out(src.name(), src.shape(), dst);

    const int64_t n = src.numel();

    if (src.dtype() == dst) {
        std::memcpy(out.data(), src.data(), src.nbytes());
        return out;
    }

    // Via fp32 intermediate (correctness over speed for rare conversions).
    for (int64_t i = 0; i < n; ++i) {
        const float v = src.get_as_float(i);
        switch (dst) {
            case DType::F32: out.f32()[i] = v; break;
            case DType::F16: out.f16()[i] = float_to_half(v); break;
            case DType::I8:  out.i8()[i] =
                static_cast<int8_t>(std::lround(std::tanh(v) * 127.0f)); break;
            case DType::I32: out.i32()[i] = static_cast<int32_t>(v); break;
            default: break;
        }
    }
    return out;
}

void add_inplace(Tensor& y, const Tensor& a) {
    assert(y.dtype() == DType::F32 && a.dtype() == DType::F32);
    const int64_t n = y.numel();
    float* yp = y.f32();
    const float* ap = a.f32();
    for (int64_t i = 0; i < n; ++i) yp[i] += ap[i];
}

void mul_inplace(Tensor& y, const Tensor& a) {
    assert(y.dtype() == DType::F32 && a.dtype() == DType::F32);
    const int64_t n = y.numel();
    float* yp = y.f32();
    const float* ap = a.f32();
    for (int64_t i = 0; i < n; ++i) yp[i] *= ap[i];
}

void sigmoid_inplace(Tensor& t) {
    float* p = t.f32();
    const int64_t n = t.numel();
    for (int64_t i = 0; i < n; ++i) p[i] = 1.0f / (1.0f + std::exp(-p[i]));
}

// RWKV uses a numerically-stabilized exp: exp(min(x, 30)) with a delta trick
// (from the rwkv.cpp / RWKV-LM reference implementations).
float exp_delta(float x, float max) {
    if (x > max) x = max;
    return std::exp(x - max);
}

void softmax_last_dim(Tensor& t, const Tensor* bias) {
    float* p = t.f32();
    const int64_t last = t.cols();
    const int64_t outer = t.numel() / (last > 0 ? last : 1);
    for (int64_t row = 0; row < outer; ++row) {
        float* r = p + row * last;
        float maxv = -1e30f;
        for (int64_t i = 0; i < last; ++i) {
            const float v = bias ? r[i] + bias->f32()[i] : r[i];
            if (v > maxv) maxv = v;
        }
        float sum = 0.0f;
        for (int64_t i = 0; i < last; ++i) {
            const float v = bias ? r[i] + bias->f32()[i] : r[i];
            r[i] = std::exp(v - maxv);
            sum += r[i];
        }
        const float inv = 1.0f / (sum + 1e-8f);
        for (int64_t i = 0; i < last; ++i) r[i] *= inv;
    }
}

void l2_normalize(Tensor& t) {
    float* p = t.f32();
    const int64_t n = t.numel();
    double sum = 0.0;
    for (int64_t i = 0; i < n; ++i) sum += static_cast<double>(p[i]) * p[i];
    const float inv = static_cast<float>(
        1.0 / (std::sqrt(sum) + 1e-12));
    for (int64_t i = 0; i < n; ++i) p[i] *= inv;
}

void gelu_inplace(Tensor& t) {
    // tanh approximation: 0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3)))
    float* p = t.f32();
    const int64_t n = t.numel();
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    for (int64_t i = 0; i < n; ++i) {
        const float x = p[i];
        const float inner = kSqrt2OverPi * (x + 0.044715f * x * x * x);
        p[i] = 0.5f * x * (1.0f + std::tanh(inner));
    }
}

void rms_norm(const Tensor& x, const Tensor* weight, Tensor& y) {
    const int64_t n = x.numel();
    const float* xp = x.f32();
    float* yp = y.f32();

    double ss = 0.0;
    for (int64_t i = 0; i < n; ++i) ss += static_cast<double>(xp[i]) * xp[i];
    const float inv_rms = static_cast<float>(
        1.0 / std::sqrt(ss / static_cast<double>(n) + 1e-6));

    for (int64_t i = 0; i < n; ++i) {
        const float w = weight ? weight->get_as_float(i) : 1.0f;
        yp[i] = xp[i] * inv_rms * w;
    }
}

void layer_norm(const Tensor& x, const Tensor* weight, const Tensor* bias,
                Tensor& y, float eps) {
    const int64_t n = x.numel();
    const float* xp = x.f32();
    float* yp = y.f32();

    double mean = 0.0;
    for (int64_t i = 0; i < n; ++i) mean += xp[i];
    mean /= static_cast<double>(n);

    double var = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double d = xp[i] - mean;
        var += d * d;
    }
    var /= static_cast<double>(n);

    const float inv_std = static_cast<float>(1.0 / std::sqrt(var + eps));
    for (int64_t i = 0; i < n; ++i) {
        const float w = weight ? weight->get_as_float(i) : 1.0f;
        const float b = bias ? bias->get_as_float(i) : 0.0f;
        yp[i] = static_cast<float>((xp[i] - mean) * inv_std) * w + b;
    }
}

} // namespace tensor_ops
} // namespace omniseed
