// =============================================================================
//  OmniSeed — gguf_loader.h
//  GGUF (GPT-Generated Unified Format) model container reader.
//
//  Parses the binary header & tensor directory, then hands out zero-copy
//  Tensor views straight into the platform::MappedFile region, so weights
//  are paged in lazily and never duplicated on the heap (peak-RSS friendly).
//
//  Format reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
//  Supported GGUF version: 2/3 (identical header layout, metadata value types
//  per the spec's little-endian encoding).
// =============================================================================
#pragma once

#include "omniseed/core/gguf_format.h"
#include "omniseed/core/platform.h"
#include "omniseed/core/tensor.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace omniseed {

class GgufLoader {
public:
    GgufLoader() = default;
    ~GgufLoader() = default;

    // Parses header + tensor directory. File stays mapped until close().
    // Returns false on malformed input (error() has details).
    bool open(const std::string& path);

    void close();

    bool valid() const { return valid_; }

    const std::string& error() const { return error_; }

    // ------------------------------ metadata ---------------------------------
    // Typed getters; return fallback if key missing or type mismatches.
    std::string get_string(const std::string& key,
                           const std::string& fallback = "") const;
    int64_t     get_u64(const std::string& key, int64_t fallback = 0) const;
    double      get_f64(const std::string& key, double fallback = 0.0) const;
    bool        has_key(const std::string& key) const;

    // Raw metadata access (used by the tokenizer for array-type values).
    const gguf::MetadataValue* metadata_value(const std::string& key) const;

    // Common convenience accessors.
    int64_t n_layers()     const { return get_u64("omniseed.layer_count", 0); }
    int64_t n_embd()       const { return get_u64("omniseed.embedding_length", 0); }
    int64_t n_vocab()      const { return get_u64("omniseed.vocab_size", 0); }
    int64_t context_len()  const { return get_u64("omniseed.context_length", 0); }

    // ------------------------------ tensors ----------------------------------
    // Tensor names present in the file (in file order).
    const std::vector<std::string>& tensor_names() const { return tensor_names_; }

    bool has_tensor(const std::string& name) const;

    // Returns a zero-copy view over the mapped weights, dtype-preserving.
    // The view is valid until close(). Returns empty Tensor on miss.
    Tensor tensor(const std::string& name) const;

    // Raw pointer info for a tensor (for kernels that want raw bytes).
    const void* tensor_data(const std::string& name) const;
    size_t      tensor_size(const std::string& name) const;

    // Underlying mapping (advanced: custom views over the file).
    const platform::MappedFile& mapping() const { return file_; }

    // File size in bytes.
    uint64_t file_size() const { return file_.size(); }

private:
    struct TensorInfo {
        std::string name;
        std::vector<int64_t> dims;      // GGUF ne[] (reversed on load)
        gguf::GgufDType type;
        uint64_t         offset;        // from data section start
        uint64_t         nbytes;
    };

    bool read_magic_version();
    bool read_metadata();
    bool read_tensor_dir();

    platform::MappedFile file_;
    bool    valid_ = false;
    int32_t version_ = 0;
    uint64_t data_start_ = 0;   // file offset where tensor data begins

    std::unordered_map<std::string, gguf::MetadataValue> metadata_;
    std::vector<TensorInfo>  tensors_;
    std::unordered_map<std::string, size_t>  tensor_index_;
    std::vector<std::string> tensor_names_;

    std::string error_;
};

} // namespace omniseed
