// =============================================================================
//  OmniSeed — gguf_loader.cpp
//  Binary GGUF parsing + zero-copy tensor views over the mmap'd model.
// =============================================================================
#include "omniseed/core/gguf_loader.h"
#include "omniseed/core/platform.h"

namespace omniseed {

using gguf::GgufType;
using gguf::GgufDType;
using gguf::MetadataValue;

// ===========================================================================
// Header: magic + version + tensor_count + metadata_kv_count
// ===========================================================================
bool GgufLoader::read_magic_version() {
    const uint8_t* base = file_.bytes();
    if (file_.size() < 24) {
        error_ = "file too small for GGUF header";
        return false;
    }
    if (gguf::read_u32(base, 0) != gguf::GGUF_MAGIC) {
        error_ = "bad GGUF magic";
        return false;
    }
    version_ = static_cast<int32_t>(gguf::read_u32(base, 4));
    if (version_ < 2 || version_ > 3) {
        error_ = "unsupported GGUF version " + std::to_string(version_);
        return false;
    }
    return true;
}

// ===========================================================================
// Metadata section
// ===========================================================================
bool GgufLoader::read_metadata() {
    const uint8_t* base = file_.bytes();
    size_t cursor = 8;   // magic + version

    const uint64_t n_tensors = gguf::read_u64(base, cursor); cursor += 8;
    (void)n_tensors;   // header count; the tensor section re-reads it in load()
    const uint64_t n_kv      = gguf::read_u64(base, cursor); cursor += 8;

    // Implausible header counts: reserve() would throw std::length_error
    // (-> abort) instead of failing open() cleanly.
    if (n_kv > 1000000 || n_tensors > 1000000) {
        error_ = "GGUF header counts are implausible";
        return false;
    }

    metadata_.reserve(static_cast<size_t>(n_kv));

    // Bounds-checked read helpers: a malformed / truncated GGUF must yield a
    // clean open() failure — never an out-of-bounds read, and never an
    // exception (std::string with a hostile u64 length used to throw
    // std::length_error, which aborted the process).
    const uint64_t fsize = file_.size();
    auto need = [&](size_t n) { return cursor + n <= fsize; };
    auto read_str = [&](std::string& out) {
        if (!need(8)) return false;
        const uint64_t len = gguf::read_u64(base, cursor);
        cursor += 8;
        if (len > fsize - cursor) return false;
        out.assign(reinterpret_cast<const char*>(base + cursor),
                   static_cast<size_t>(len));
        cursor += static_cast<size_t>(len);
        return true;
    };
    auto fail = [this](const std::string& key) {
        error_ = "GGUF metadata truncated at key '" + key + "'";
        return false;
    };

    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key;
        if (!read_str(key)) {
            error_ = "GGUF metadata: truncated key string";
            return false;
        }

        MetadataValue val;
        if (!need(4)) return fail(key);
        val.type = static_cast<GgufType>(gguf::read_i32(base, cursor));
        cursor += 4;

        switch (val.type) {
            case GgufType::UINT8:  if (!need(1)) return fail(key); val.u64 = base[cursor];               cursor += 1; break;
            case GgufType::INT8:   if (!need(1)) return fail(key); val.i64 = static_cast<int8_t>(base[cursor]); cursor += 1; break;
            case GgufType::UINT16: if (!need(2)) return fail(key); val.u64 = gguf::read_u16(base, cursor); cursor += 2; break;
            case GgufType::INT16:  if (!need(2)) return fail(key); val.i64 = static_cast<int16_t>(gguf::read_u16(base, cursor)); cursor += 2; break;
            case GgufType::UINT32: if (!need(4)) return fail(key); val.u64 = gguf::read_u32(base, cursor); cursor += 4; break;
            case GgufType::INT32:  if (!need(4)) return fail(key); val.i64 = gguf::read_i32(base, cursor); cursor += 4; break;
            case GgufType::FLOAT32: if (!need(4)) return fail(key); val.f64 = gguf::read_f32(base, cursor); cursor += 4; break;
            case GgufType::BOOL:   if (!need(1)) return fail(key); val.b = base[cursor] != 0;            cursor += 1; break;
            case GgufType::STRING: if (!read_str(val.str)) return fail(key); break;
            case GgufType::ARRAY: {
                if (!need(12)) return fail(key);
                val.elem_type  = static_cast<GgufType>(gguf::read_i32(base, cursor));
                cursor += 4;
                val.elem_count = gguf::read_u64(base, cursor);
                cursor += 8;

                // Variable-length element: keep the raw [u64 len][bytes] wire
                // format in val.raw (Tokenizer::load_from_gguf parses this).
                if (val.elem_type == GgufType::STRING) {
                    val.raw.reserve(val.elem_count * 8);   // + actual bytes below
                    for (uint64_t e = 0; e < val.elem_count; ++e) {
                        if (!need(8)) return fail(key);
                        const uint64_t len = gguf::read_u64(base, cursor);
                        if (cursor + 8 + len > file_.size()) {
                            error_ = "string array out of bounds for key " + key;
                            return false;
                        }
                        const uint8_t* start = base + cursor;
                        val.raw.insert(val.raw.end(), start, start + 8 + len);
                        cursor += 8 + static_cast<size_t>(len);
                    }
                    break;
                }

                const size_t esz = [&] {
                    switch (val.elem_type) {
                        case GgufType::UINT8: case GgufType::INT8:
                        case GgufType::BOOL:  return 1u;
                        case GgufType::UINT16: case GgufType::INT16: return 2u;
                        case GgufType::UINT32: case GgufType::FLOAT32:
                        case GgufType::INT32:  return 4u;
                        case GgufType::UINT64: case GgufType::FLOAT64:
                        case GgufType::INT64:  return 8u;
                        default: return 0u;
                    }
                }();
                if (esz == 0) {
                    error_ = "unsupported array element type for key " + key;
                    return false;
                }
                const size_t bytes = static_cast<size_t>(val.elem_count) * esz;
                if (cursor + bytes > file_.size()) {
                    error_ = "metadata array out of bounds for key " + key;
                    return false;
                }
                val.raw.assign(base + cursor, base + cursor + bytes);
                cursor += bytes;
                break;
            }
            case GgufType::UINT64: val.u64 = gguf::read_u64(base, cursor); cursor += 8; break;
            case GgufType::INT64:  val.i64 = gguf::read_u64(base, cursor); cursor += 8; break;
            case GgufType::FLOAT64: val.f64 = gguf::read_f64(base, cursor); cursor += 8; break;
            default:
                error_ = "unknown metadata type " +
                         std::to_string(static_cast<int>(val.type)) +
                         " for key " + key;
                return false;
        }

        metadata_.emplace(key, std::move(val));
    }

    data_start_ = static_cast<uint64_t>(cursor);
    return true;
}

// ===========================================================================
// Tensor directory: name, n_dims, ne[], dtype, offset (relative to data start)
// ===========================================================================
namespace {
// Bytes per ELEMENT for GGUF tensor dtypes. TERNARY packs two weights per
// byte, so it is reported in HALF-bytes: multiply by (numel + 1) / 2 at the
// call site (see read_tensor_dir).
size_t gguf_dtype_size(gguf::GgufDType t) {
    switch (t) {
        case gguf::GgufDType::F32:     return 4;
        case gguf::GgufDType::F16:     return 2;
        case gguf::GgufDType::I8:      return 1;
        case gguf::GgufDType::I32:     return 4;
        case gguf::GgufDType::TERNARY: return 0;   // handled specially (2/byte)
        default:                       return 0;
    }
}

DType to_tensor_dtype(gguf::GgufDType t) {
    switch (t) {
        case gguf::GgufDType::F32:     return DType::F32;
        case gguf::GgufDType::F16:     return DType::F16;
        case gguf::GgufDType::I8:      return DType::I8;
        case gguf::GgufDType::I32:     return DType::I32;
        case gguf::GgufDType::TERNARY: return DType::TERNARY;
        default:                       return DType::F32;
    }
}
} // namespace

bool GgufLoader::read_tensor_dir() {
    const uint8_t* base = file_.bytes();
    size_t cursor = static_cast<size_t>(data_start_);

    uint64_t n_tensors = 0;
    std::memcpy(&n_tensors, base + 8, 8);   // read from header again

    tensors_.reserve(static_cast<size_t>(n_tensors));

    // Bounds-checked reads (same rationale as read_metadata): a truncated
    // tensor directory must fail open() cleanly, never read OOB or throw.
    const uint64_t fsize = file_.size();
    auto need = [&](size_t n) { return cursor + n <= fsize; };
    auto read_str = [&](std::string& out) {
        if (!need(8)) return false;
        const uint64_t len = gguf::read_u64(base, cursor);
        cursor += 8;
        if (len > fsize - cursor) return false;
        out.assign(reinterpret_cast<const char*>(base + cursor),
                   static_cast<size_t>(len));
        cursor += static_cast<size_t>(len);
        return true;
    };

    for (uint64_t i = 0; i < n_tensors; ++i) {
        TensorInfo info;
        if (!read_str(info.name)) {
            error_ = "GGUF tensor directory: truncated name";
            return false;
        }

        if (!need(4)) {
            error_ = "GGUF tensor '" + info.name + "': truncated n_dims";
            return false;
        }
        const uint32_t n_dims = gguf::read_u32(base, cursor);
        cursor += 4;
        if (n_dims == 0 || n_dims > 64) {   // real tensors have 1-4 dims
            error_ = "GGUF tensor '" + info.name + "': implausible n_dims";
            return false;
        }

        info.dims.resize(n_dims);
        for (uint32_t d = 0; d < n_dims; ++d) {
            if (!need(8)) {
                error_ = "GGUF tensor '" + info.name + "': truncated dims";
                return false;
            }
            info.dims[d] = static_cast<int64_t>(gguf::read_u64(base, cursor));
            cursor += 8;
        }

        if (!need(4)) {
            error_ = "GGUF tensor '" + info.name + "': truncated dtype";
            return false;
        }
        const auto ggtype = static_cast<GgufDType>(gguf::read_i32(base, cursor));
        cursor += 4;

        if (!need(8)) {
            error_ = "GGUF tensor '" + info.name + "': truncated offset";
            return false;
        }
        info.offset = gguf::read_u64(base, cursor);
        cursor += 8;

        // Compute byte size; GGUF dims are reversed vs row-major.
        int64_t numel = 1;
        for (const int64_t d : info.dims) numel *= d;
        const bool is_ternary = (ggtype == GgufDType::TERNARY);
        const size_t esz = gguf_dtype_size(ggtype);
        if (esz == 0 && !is_ternary) {
            error_ = "tensor '" + info.name + "': unsupported dtype " +
                     std::to_string(static_cast<int>(ggtype));
            return false;
        }
        info.type   = ggtype;   // store raw GGUF dtype
        info.nbytes = is_ternary
                          ? static_cast<uint64_t>((numel + 1) / 2)  // 2 weights/byte
                          : static_cast<uint64_t>(numel) * esz;

        tensor_index_[info.name] = tensors_.size();
        tensors_.push_back(std::move(info));
    }

    // GGUF v2/v3 spec: tensor data begins immediately AFTER the tensor
    // directory, and each info.offset is relative to that point (alignment
    // padding between tensors is optional; OmniSeed honors explicit offsets).
    // The previous code kept data_start_ at the post-metadata position, which
    // shifted every tensor read into the directory bytes -> garbage weights.
    data_start_ = static_cast<uint64_t>(cursor);

    for (const TensorInfo& t : tensors_) {
        const uint64_t abs_off = data_start_ + t.offset;
        if (abs_off + t.nbytes > file_.size()) {
            error_ = "tensor '" + t.name + "' out of file bounds";
            return false;
        }
        tensor_names_.push_back(t.name);
    }
    return true;
}

// ===========================================================================
// Public API
// ===========================================================================
bool GgufLoader::open(const std::string& path) {
    close();
    if (!file_.open(path)) {
        error_ = "cannot map " + path + ": " + file_.last_error();
        return false;
    }
    if (!read_magic_version() || !read_metadata() || !read_tensor_dir()) {
        // Preserve the parse error, then FULLY reset: a bare file_.close()
        // left metadata_/tensors_/tensor_index_ populated while the mapping
        // was gone, so later has_tensor()/tensor() built views over wild
        // pointers (nullptr + offset) instead of returning empty tensors.
        const std::string err = error_.empty() ? "malformed GGUF" : error_;
        close();
        error_ = err;
        return false;
    }
    valid_ = true;
    return true;
}

void GgufLoader::close() {
    file_.close();
    metadata_.clear();
    tensors_.clear();
    tensor_index_.clear();
    tensor_names_.clear();
    valid_ = false;
    error_.clear();
    data_start_ = 0;
}

std::string GgufLoader::get_string(const std::string& key,
                                   const std::string& fallback) const {
    auto it = metadata_.find(key);
    if (it == metadata_.end() || it->second.type != GgufType::STRING)
        return fallback;
    return it->second.str;
}

int64_t GgufLoader::get_u64(const std::string& key, int64_t fallback) const {
    auto it = metadata_.find(key);
    if (it == metadata_.end()) return fallback;
    const MetadataValue& v = it->second;
    switch (v.type) {
        case GgufType::UINT8: case GgufType::UINT16: case GgufType::UINT32:
        case GgufType::UINT64:
            return static_cast<int64_t>(v.u64);
        case GgufType::INT8: case GgufType::INT16: case GgufType::INT32:
        case GgufType::INT64:
            return v.i64;
        default:
            return fallback;
    }
}

double GgufLoader::get_f64(const std::string& key, double fallback) const {
    auto it = metadata_.find(key);
    if (it == metadata_.end()) return fallback;
    const MetadataValue& v = it->second;
    if (v.type == GgufType::FLOAT32 || v.type == GgufType::FLOAT64) return v.f64;
    return fallback;
}

bool GgufLoader::has_key(const std::string& key) const {
    return metadata_.find(key) != metadata_.end();
}

const gguf::MetadataValue* GgufLoader::metadata_value(
        const std::string& key) const {
    auto it = metadata_.find(key);
    return it != metadata_.end() ? &it->second : nullptr;
}

bool GgufLoader::has_tensor(const std::string& name) const {
    return tensor_index_.find(name) != tensor_index_.end();
}

Tensor GgufLoader::tensor(const std::string& name) const {
    auto it = tensor_index_.find(name);
    if (it == tensor_index_.end()) return Tensor();

    const TensorInfo& info = tensors_[it->second];
    const uint8_t* ptr = file_.bytes() + static_cast<size_t>(data_start_ + info.offset);

    // GGUF stores dims reversed relative to row-major; flip for Tensor.
    std::vector<int64_t> shape(info.dims.rbegin(), info.dims.rend());
    return Tensor(info.name, shape, to_tensor_dtype(info.type),
                  const_cast<uint8_t*>(ptr));
}

const void* GgufLoader::tensor_data(const std::string& name) const {
    auto it = tensor_index_.find(name);
    if (it == tensor_index_.end()) return nullptr;
    const TensorInfo& info = tensors_[it->second];
    return file_.bytes() + static_cast<size_t>(data_start_ + info.offset);
}

size_t GgufLoader::tensor_size(const std::string& name) const {
    auto it = tensor_index_.find(name);
    if (it == tensor_index_.end()) return 0;
    return static_cast<size_t>(tensors_[it->second].nbytes);
}

} // namespace omniseed
