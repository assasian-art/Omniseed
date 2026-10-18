// =============================================================================
//  OmniSeed — fitted_heads.h
//
//  The RUNTIME consumer for fitted ClassificationHead blobs.
//
//  tools/train_heads.py writes language_head.bin / vision_head.bin /
//  audio_head.bin / trading_regime_head.bin / trading_head.bin from dumped
//  h[E]. Until now NOTHING outside tests ever loaded them: the pipeline's
//  classifier_ was init()-ed (seeded, untrained) and its distributions were
//  well-formed but meaningless. This class closes that gap with the same
//  fail-closed stance every head in this repo takes:
//
//    * lazy load — a blob is read on first use of its set, once, then cached;
//    * refusal, never invention — a missing dir, a corrupt blob, a set with no
//      registered blob, or a set the blob does not carry all return the head's
//      own failure result (`domain = "<error: ...>"`, `top_k` empty). No
//      uniform distribution, no fallback to the untrained head;
//    * one bin per set family: "language.*" -> language_head.bin, etc. The
//      mapping is explicit (trading.regime and trading.action live in
//      DIFFERENT bins), so a set nobody mapped refuses instead of guessing.
//
//  The load path is ClassificationHead::load, which rebuilds label sets from
//  the file (sets travel WITH the weights) and leaves the head NOT READY on
//  any read failure — so a half-read projection can never classify.
// =============================================================================
#pragma once

#include <map>
#include <string>

#include "omniseed/classification_head.h"

namespace omniseed {

class FittedHeads {
public:
    FittedHeads() = default;

    // Directory holding the fitted .bin files (e.g. "models/heads").
    // Changing it drops every cache: the next classify() re-reads from disk.
    void set_heads_dir(const std::string& dir);
    const std::string& heads_dir() const { return heads_dir_; }

    // Points at `dir` and eagerly attempts EVERY known classification bin now
    // (language/vision/audio/trading_regime) instead of lazily on first use.
    // True when at least one blob loaded; false means nothing is fitted and
    // every classify() refuses (fail-closed — see the header top). Either way
    // the per-bin reasons land in error(), so an operator sees WHICH blob is
    // missing rather than a single opaque failure.
    bool load(const std::string& dir);

    // The blob a set family is fitted in, or "" when no blob is registered —
    // an unregistered set must refuse, not fall through to some other bin.
    // Explicit for the split bins (trading.regime vs trading.action).
    static std::string bin_for_set(const std::string& set_name);

    // Lazy-loads bin_for_set(set_name) from heads_dir_, then forwards to the
    // loaded head. Every failure mode returns ClassificationHead's own failure
    // result (domain starts with '<', top_k empty) — see classification_head.h.
    // Loads are sticky: a failed load is remembered (reason cached) so a
    // missing blob cannot turn every call into a disk retry loop; set_heads_dir
    // clears both caches.
    //
    // `caller_E` is the width of `hidden` when the caller knows it. A float*
    // alone cannot be width-checked (ClassificationHead::classify has no length
    // to read), so passing it here is what turns "blob fitted at E=768 fed a
    // 512-wide vector" from a silently wrong distribution into a refusal.
    // 0 = "cannot check" (caller genuinely does not know); never guess.
    ClassificationResult classify(const float* hidden, const std::string& set_name,
                                  int32_t top_k = 3, int32_t caller_E = 0) const;

    // Forces the load attempt now instead of on first classify.
    // True only when the set's blob is loaded AND carries that set.
    bool available(const std::string& set_name);

    // Loaded-head provenance ("loaded ..."), or the cached failure reason.
    const std::string& error() const { return error_; }

    int32_t loaded_count() const { return static_cast<int32_t>(heads_.size()); }

private:
    // nullptr on any failure; failure reason is appended to error_ and cached
    // per bin in failures_. (Mutates the mutable caches; classify() const_casts
    // once here rather than spreading mutable through the public surface.)
    const ClassificationHead* head_for(const std::string& set_name);

    // Loads one bin (or returns the cached head / cached failure). nullptr on
    // failure with the reason in error_. Shared by head_for() (lazy) and
    // load() (eager), so the two paths can never diverge on validation.
    const ClassificationHead* ensure_bin(const std::string& bin);

    std::string heads_dir_ = "models/heads";

    // key = bin file name (not set name): one head carries every set in its
    // family and is loaded once for all of them.
    mutable std::map<std::string, ClassificationHead> heads_;
    mutable std::map<std::string, std::string>        failures_; // bin -> reason
    mutable std::string                               error_;
};

} // namespace omniseed
