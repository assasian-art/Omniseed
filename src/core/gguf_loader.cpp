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

    metadata_.reserve(static_cast<size_t>(n_kv));

    for (uint64_t i = 0; i < n_kv; ++i) {
        const std::string key = gguf::read_string(base, cursor);

        MetadataValue val;
        val.type = static_cast<GgufType>(gguf::read_i32(base, cursor));
        cursor += 4;

        switch (val.type) {
            case GgufType::UINT8:  val.u64 = base[cursor];               cursor += 1; break;
            case GgufType::INT8:   val.i64 = static_cast<int8_t>(base[cursor]); cursor += 1; break;
            case GgufType::UINT16: val.u64 = gguf::read_u16(base, cursor); cursor += 2; break;
            case GgufType::INT16:  val.i64 = static_cast<int16_t>(gguf::read_u16(base, cursor)); cursor += 2; break;
            case GgufType::UINT32: val.u64 = gguf::read_u32(base, cursor); cursor += 4; break;
            case GgufType::INT32:  val.i64 = gguf::read_i32(base, cursor); cursor += 4; break;
            case GgufType::FLOAT32: val.f64 = gguf::read_f32(base, cursor); cursor += 4; break;
            case GgufType::BOOL:   val.b = base[cursor] != 0;            cursor += 1; break;
            case GgufType::STRING: val.str = gguf::read_string(base, cursor); break;
            case GgufType::ARRAY: {
                val.elem_type  = static_cast<GgufType>(gguf::read_i32(base, cursor));
                cursor += 4;
                val.elem_count = gguf::read_u64(base, cursor);
                cursor += 8;

                // Variable-length element: keep the raw [u64 len][bytes] wire
                // format in val.raw (Tokenizer::load_from_gguf parses this).
                if (val.elem_type == GgufType::STRING) {
                    val.raw.reserve(val.elem_count * 8);   // + actual bytes below
                    for (uint64_t e = 0; e < val.elem_count; ++e) {
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
size_t gguf_dtype_size(gguf::GgufDType t) {
    switch (t) {
        case gguf::GgufDType::F32:     return 4;
        case gguf::GgufDType::F16:     return 2;
        case gguf::GgufDType::I8:      return 1;
        case gguf::GgufDType::I32:     return 4;
        case gguf::GgufDType::TERNARY: return 1;   // 2 weights/byte
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

    for (uint64_t i = 0; i < n_tensors; ++i) {
        TensorInfo info;
        info.name = gguf::read_string(base, cursor);

        const uint32_t n_dims = gguf::read_u32(base, cursor);
        cursor += 4;

        info.dims.resize(n_dims);
        for (uint32_t d = 0; d < n_dims; ++d) {
            info.dims[d] = static_cast<int64_t>(gguf::read_u64(base, cursor));
            cursor += 8;
        }

        const auto ggtype = static_cast<GgufDType>(gguf::read_i32(base, cursor));
        cursor += 4;

        info.offset = gguf::read_u64(base, cursor);
        cursor += 8;

        // Compute byte size; GGUF dims are reversed vs row-major.
        int64_t numel = 1;
        for (const int64_t d : info.dims) numel *= d;
        const size_t esz = gguf_dtype_size(ggtype);
        if (esz == 0) {
            error_ = "tensor '" + info.name + "': unsupported dtype " +
                     std::to_string(static_cast<int>(ggtype));
            return false;
        }
        info.type   = ggtype;   // store raw GGUF dtype
        info.nbytes = static_cast<uint64_t>(numel) * esz;

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
        error_ = error_.empty() ? "malformed GGUF" : error_;
        file_.close();
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
