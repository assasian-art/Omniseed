// =============================================================================
//  OmniSeed — fitted_heads.cpp
//  See the header for the contract. Everything here is fail-closed: the only
//  path that returns probabilities is a fully loaded head that carries the
//  requested set; every other path returns the "<error: ...>" / empty-top_k
//  refusal that callers already treat as "the head did not answer".
// =============================================================================
#include "omniseed/runtime/fitted_heads.h"

#include <utility>

namespace omniseed {

namespace {

// The refusal. Same wire contract as ClassificationHead::failure(): domain
// starts with '<' (no registered set name can), top_k empty. Built here
// because ClassificationHead::failure() is private — the format is the
// contract, not the function.
ClassificationResult refuse(const std::string& set_name, const std::string& why) {
    ClassificationResult r;
    r.domain  = "<error: fitted heads: " + why + ">";
    r.top_k.clear();
    r.margin  = 0.0f;
    r.matvecs = 0;
    r.us      = 0.0;
    (void)set_name;
    return r;
}

std::string join(const std::string& dir, const std::string& file) {
    if (dir.empty()) return file;
    const char last = dir.back();
    if (last == '/' || last == '\\') return dir + file;
    return dir + "/" + file;
}

} // namespace

void FittedHeads::set_heads_dir(const std::string& dir) {
    if (dir == heads_dir_) return;
    heads_dir_ = dir;
    heads_.clear();    // old-dir blobs must not answer for the new dir
    failures_.clear(); // a blob that was missing before may exist now
    error_.clear();
}

std::string FittedHeads::bin_for_set(const std::string& set_name) {
    // Prefix families first...
    if (set_name.rfind("language.", 0) == 0) return "language_head.bin";
    if (set_name.rfind("vision.", 0)   == 0) return "vision_head.bin";
    if (set_name.rfind("audio.", 0)    == 0) return "audio_head.bin";
    // ...then the split trading bins: regime and action are DIFFERENT files
    // (train_heads writes trading_regime_head.bin and trading_head.bin), so
    // they are named explicitly rather than by a "trading.*" guess.
    if (set_name == "trading.regime") return "trading_regime_head.bin";
    if (set_name == "trading.action") return "trading_head.bin";
    // general.routing (and anything unmapped): no fitted blob exists — refuse.
    return "";
}

const ClassificationHead* FittedHeads::ensure_bin(const std::string& bin) {
    // Sticky load failure: a missing/corrupt blob is remembered so a hot loop
    // cannot hammer the disk; set_heads_dir() clears it for retry.
    const auto failed = failures_.find(bin);
    if (failed != failures_.end()) {
        error_ = failed->second;
        return nullptr;
    }

    auto it = heads_.find(bin);
    if (it == heads_.end()) {
        ClassificationHead fresh;
        const std::string path = join(heads_dir_, bin);
        if (!fresh.load(path)) {
            const std::string why = "cannot load " + path + ": " + fresh.error();
            failures_[bin] = why;
            error_ = why;
            return nullptr;
        }
        it = heads_.emplace(bin, std::move(fresh)).first;
    }
    return &it->second;
}

const ClassificationHead* FittedHeads::head_for(const std::string& set_name) {
    const std::string bin = bin_for_set(set_name);
    if (bin.empty()) {
        error_ = "no fitted blob registered for set '" + set_name + "'";
        return nullptr;
    }

    const ClassificationHead* h = ensure_bin(bin);
    if (!h) return nullptr;

    // Loaded, but this particular set is not in it (e.g. vision.anomaly asked
    // of a blob fitted only on vision.scene). Set-specific, so NOT cached as a
    // bin failure — another set in the same blob may be perfectly valid.
    if (h->find_label_set(set_name) < 0) {
        error_ = "blob " + bin + " does not carry set '" + set_name + "'";
        return nullptr;
    }

    error_.clear();
    return h;
}

bool FittedHeads::load(const std::string& dir) {
    // The four ClassificationHead bins train_heads.py can emit. trading_head.bin
    // is a DecisionHead (different format) and lives on the trading path —
    // deliberately NOT listed here.
    static const char* kClassificationBins[] = {
        "language_head.bin", "vision_head.bin", "audio_head.bin",
        "trading_regime_head.bin"};
    set_heads_dir(dir);
    bool any = false;
    for (const char* bin : kClassificationBins)
        if (ensure_bin(bin)) any = true;
    return any;
}

ClassificationResult FittedHeads::classify(const float* hidden,
                                           const std::string& set_name,
                                           int32_t top_k,
                                           int32_t caller_E) const {
    FittedHeads* self = const_cast<FittedHeads*>(this);
    const ClassificationHead* h = self->head_for(set_name);
    if (!h) return refuse(set_name, error_);
    if (!hidden) return refuse(set_name, "null hidden");
    if (caller_E > 0 && h->hidden_size() != caller_E) {
        // Widths that do not match would otherwise matvec past the caller's
        // buffer (or read past it) — a refusal, not a nearest-guess.
        return refuse(set_name,
                      "hidden width mismatch: blob E=" +
                          std::to_string(h->hidden_size()) +
                          " caller E=" + std::to_string(caller_E));
    }
    return h->classify(hidden, set_name, top_k);
}

bool FittedHeads::available(const std::string& set_name) {
    return head_for(set_name) != nullptr;
}

} // namespace omniseed
