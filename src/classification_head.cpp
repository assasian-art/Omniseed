// =============================================================================
//  OmniSeed — src/classification_head.cpp
//
//  One [total_labels, E] projection, many named label sets. See the header for
//  the contract and the honesty note about untrained (seeded) projections.
//
//  Two details worth knowing before changing anything here:
//
//  1. ROWS ARE SEEDED PER-ROW, NOT IN REGISTRATION ORDER. Each row's seed is
//     derived from (seed, set name, label name), so registering the same sets
//     in a different order produces the same placeholder weights. A running
//     generator would have made the placeholder depend on call order, which
//     turns "add a label set" into a change that silently moves every other
//     distribution.
//
//  2. THE SOFTMAX SUBTRACTS THE MAX BEFORE exp(). Without it, a large logit
//     overflows to +inf, every exp() becomes inf, and inf/inf is NaN — a
//     distribution of NaNs that argmax() would happily turn into "label 0".
// =============================================================================
#include "omniseed/classification_head.h"
#include "omniseed/core/platform.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

namespace omniseed {

// ---------------------------------------------------------------------------
// The mandate's label sets
// ---------------------------------------------------------------------------
const std::vector<DomainLabelSetSpec>& default_domain_label_sets() {
    static const std::vector<DomainLabelSetSpec> kSets = {
        // --- language -------------------------------------------------------
        {"language.intent", {"question", "statement", "command", "greeting",
                             "farewell", "thanks", "unknown"}},
        {"language.language", {"en", "bn", "mixed", "unknown"}},
        {"language.sentiment", {"positive", "negative", "neutral"}},
        {"language.task", {"answer", "chat", "search", "execute"}},
        // --- vision ---------------------------------------------------------
        {"vision.scene", {"indoor", "outdoor", "nature", "urban", "unknown"}},
        {"vision.anomaly", {"normal", "unusual"}},
        // --- audio ----------------------------------------------------------
        {"audio.wake", {"yes", "no"}},
        {"audio.emotion", {"happy", "sad", "angry", "neutral"}},
        {"audio.speaker", {"known", "unknown"}},
        // --- trading --------------------------------------------------------
        // The four labels the regime engine already produces, so the classifier
        // and the engine cannot disagree about the vocabulary.
        {"trading.regime", {"trend_up", "trend_down", "range", "high_vol"}},
        {"trading.action", {"hold", "buy", "sell", "close", "hedge", "abstain"}},
        // --- general --------------------------------------------------------
        {"general.routing", {"trading", "language", "vision", "audio", "general"}},
        {"general.priority", {"urgent", "normal", "low"}},
    };
    return kSets;
}

int32_t add_default_label_sets(ClassificationHead& head) {
    int32_t added = 0;
    for (const DomainLabelSetSpec& spec : default_domain_label_sets()) {
        std::vector<std::string> labels;
        labels.reserve(spec.labels.size());
        for (const char* l : spec.labels) labels.emplace_back(l);
        if (head.add_label_set(spec.name, labels) >= 0) ++added;
    }
    return added;
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------
ClassificationHead::ClassificationHead(const RwkvConfig& cfg, uint32_t seed) {
    init(cfg, seed);
}

bool ClassificationHead::init(const RwkvConfig& cfg, uint32_t seed) {
    return init(cfg.n_embd, seed);
}

bool ClassificationHead::init(int32_t n_embd, uint32_t seed) {
    ready_ = false;
    error_.clear();
    sets_.clear();
    proj_.clear();
    bias_.clear();
    fitted_.clear();
    total_labels_ = 0;
    fitted_rows_  = 0;

    if (n_embd <= 0) {
        error_ = "ClassificationHead::init: n_embd must be > 0";
        provenance_ = "error: " + error_;
        return false;
    }
    E_ = n_embd;
    seed_ = seed;
    scratch_.clear();
    ready_ = true;
    provenance_ = "seeded placeholder (UNTRAINED) — per-row splitmix64 seed " +
                  std::to_string(seed) + ", E=" + std::to_string(E_) +
                  ", 0 label sets";
    return true;
}

void ClassificationHead::seed_weights(uint32_t seed) {
    // Per-row seeding keyed on the set and label names — see note 1 at the top.
    const float scale = 1.0f / std::sqrt(static_cast<float>(E_));
    for (const LabelSet& s : sets_) {
        for (size_t li = 0; li < s.labels.size(); ++li) {
            uint64_t h = 0xCBF29CE484222325ull ^ static_cast<uint64_t>(seed);
            for (const char c : s.name) { h ^= static_cast<unsigned char>(c); h *= 0x100000001B3ull; }
            h ^= 0x1F;  h *= 0x100000001B3ull;
            for (const char c : s.labels[li]) { h ^= static_cast<unsigned char>(c); h *= 0x100000001B3ull; }
            const size_t row = static_cast<size_t>(s.offset) + li;
            for (int32_t e = 0; e < E_; ++e)
                proj_[row * static_cast<size_t>(E_) + static_cast<size_t>(e)] =
                    heads_detail::next_unit(h) * scale;
            bias_[row] = heads_detail::next_unit(h) * 0.1f;
        }
    }
}

// ---------------------------------------------------------------------------
// Label sets
// ---------------------------------------------------------------------------
int32_t ClassificationHead::add_label_set(const std::string& name,
                                          const std::vector<std::string>& labels) {
    if (!ready_) return -1;
    if (name.empty() || labels.empty()) return -1;
    // A duplicate name is REFUSED, not overwritten: two registrations of
    // "language.intent" with different labels is a startup bug, and silently
    // keeping the last one turns it into a wrong distribution at runtime.
    if (find_label_set(name) >= 0) return -1;

    LabelSet s;
    s.name = name;
    s.labels = labels;
    s.offset = total_labels_;
    sets_.push_back(s);

    const int32_t L = static_cast<int32_t>(labels.size());
    total_labels_ += L;
    proj_.resize(static_cast<size_t>(total_labels_) * static_cast<size_t>(E_), 0.0f);
    bias_.resize(static_cast<size_t>(total_labels_), 0.0f);
    fitted_.resize(static_cast<size_t>(total_labels_), 0);

    // Only the new rows need seeding, but re-seeding all of them is cheap and
    // keeps the function total (a caller may add a set after a load()).
    seed_weights(seed_);
    provenance_ = "seeded placeholder (UNTRAINED) — per-row splitmix64 seed " +
                  std::to_string(seed_) + ", E=" + std::to_string(E_) + ", " +
                  std::to_string(sets_.size()) + " label sets / " +
                  std::to_string(total_labels_) + " labels (" +
                  std::to_string(fitted_rows_) + " fitted)";
    return static_cast<int32_t>(sets_.size()) - 1;
}

int32_t ClassificationHead::find_label_set(const std::string& name) const {
    for (size_t i = 0; i < sets_.size(); ++i)
        if (sets_[i].name == name) return static_cast<int32_t>(i);
    return -1;
}

void ClassificationHead::set_label_row(int32_t set_index, int32_t label_index,
                                       const float* row, float bias) {
    if (!ready_ || row == nullptr) return;
    if (set_index < 0 || set_index >= static_cast<int32_t>(sets_.size())) return;
    const LabelSet& s = sets_[static_cast<size_t>(set_index)];
    if (label_index < 0 || label_index >= static_cast<int32_t>(s.labels.size())) return;
    const size_t r = static_cast<size_t>(s.offset) + static_cast<size_t>(label_index);
    std::memcpy(&proj_[r * static_cast<size_t>(E_)], row,
                sizeof(float) * static_cast<size_t>(E_));
    bias_[r] = bias;
    if (!fitted_[r]) { fitted_[r] = 1; ++fitted_rows_; }
}

bool ClassificationHead::trained() const {
    return ready_ && total_labels_ > 0 && fitted_rows_ == total_labels_;
}

// ---------------------------------------------------------------------------
// The hot path
// ---------------------------------------------------------------------------
ClassificationResult ClassificationHead::failure(const std::string& set_name,
                                                 const char* why) const {
    // The failure is carried in `domain` so it is visible in the JSON and in a
    // log line, and top_k is left EMPTY. An empty top_k is the signal; a
    // fabricated uniform distribution would be indistinguishable from a real
    // (and useless) one. No registered set name can start with '<'.
    ClassificationResult r;
    r.domain = "<error: " + std::string(why) + ">";
    if (!set_name.empty()) r.domain += " set=" + set_name;
    r.matvecs = 0;
    return r;
}

ClassificationResult ClassificationHead::classify(const float* hidden,
                                                  int32_t set_index,
                                                  int32_t top_k) const {
    if (!ready_ || hidden == nullptr) return failure("", "not-ready");
    if (set_index < 0 || set_index >= static_cast<int32_t>(sets_.size()))
        return failure("", "unknown label set index");

    const LabelSet& s = sets_[static_cast<size_t>(set_index)];
    const int32_t L = static_cast<int32_t>(s.labels.size());
    if (L <= 0) return failure(s.name, "empty label set");

    // Grow the scratch once. After the first call with a given set this is a
    // no-op, which is what keeps the hot path allocation-free.
    if (scratch_.size() < static_cast<size_t>(L)) scratch_.resize(static_cast<size_t>(L));

    const double t0 = platform::now_ms();

    // --- logits: one matvec per label ---------------------------------------
    float max_logit = -3.4e38f;
    for (int32_t li = 0; li < L; ++li) {
        const size_t row = static_cast<size_t>(s.offset) + static_cast<size_t>(li);
        const float* w = &proj_[row * static_cast<size_t>(E_)];
        float acc = bias_[row];
        for (int32_t e = 0; e < E_; ++e) acc += w[e] * hidden[e];
        scratch_[static_cast<size_t>(li)] = acc;
        if (acc > max_logit) max_logit = acc;
    }
    // --- softmax, max-subtracted (see note 2 at the top) --------------------
    float sum = 0.0f;
    for (int32_t li = 0; li < L; ++li) {
        const float v = std::exp(scratch_[static_cast<size_t>(li)] - max_logit);
        scratch_[static_cast<size_t>(li)] = v;
        sum += v;
    }
    ClassificationResult r;
    r.domain = s.name;
    r.matvecs = L;
    if (!(sum > 0.0f)) {
        // Every logit was -inf/-inf-ish. Report uniform belief rather than a
        // fabricated winner.
        for (int32_t li = 0; li < L; ++li) scratch_[static_cast<size_t>(li)] = 1.0f / L;
    } else {
        for (int32_t li = 0; li < L; ++li) scratch_[static_cast<size_t>(li)] /= sum;
    }

    // --- top-k: a partial selection, not a full sort -------------------------
    int32_t k = top_k;
    if (k < 1) k = 1;
    if (k > L) k = L;
    std::vector<int32_t> order(static_cast<size_t>(L));
    for (int32_t li = 0; li < L; ++li) order[static_cast<size_t>(li)] = li;
    std::partial_sort(order.begin(), order.begin() + k, order.end(),
                      [this](int32_t a, int32_t b) {
                          return scratch_[static_cast<size_t>(a)] >
                                 scratch_[static_cast<size_t>(b)];
                      });

    r.top_k.reserve(static_cast<size_t>(k));
    for (int32_t i = 0; i < k; ++i) {
        const int32_t li = order[static_cast<size_t>(i)];
        LabelProb lp;
        lp.label = s.labels[static_cast<size_t>(li)];
        lp.probability = scratch_[static_cast<size_t>(li)];
        r.top_k.push_back(lp);
    }
    // margin needs the RUNNER-UP, which may be outside the requested top-k.
    // Reading it from `order` would silently report p(top) - 0 for k == 1.
    float second = 0.0f;
    for (int32_t li = 0; li < L; ++li) {
        if (li == order[0]) continue;
        if (scratch_[static_cast<size_t>(li)] > second)
            second = scratch_[static_cast<size_t>(li)];
    }
    r.margin = r.top_k.empty() ? 0.0f : (r.top_k[0].probability - second);

    r.us = (platform::now_ms() - t0) * 1000.0;
    last_us_ = r.us;
    return r;
}

ClassificationResult ClassificationHead::classify(const float* hidden,
                                                  const std::string& set_name,
                                                  int32_t top_k) const {
    if (!ready_) return failure(set_name, "not-ready");
    const int32_t idx = find_label_set(set_name);
    if (idx < 0) return failure(set_name, "unknown label set");
    return classify(hidden, idx, top_k);
}

ClassificationResult ClassificationHead::classify(const Tensor& hidden,
                                                  const std::string& set_name,
                                                  int32_t top_k) const {
    if (hidden.dtype() != DType::F32) return failure(set_name, "hidden state is not f32");
    if (hidden.numel() != static_cast<int64_t>(E_)) return failure(set_name, "hidden width mismatch");
    return classify(hidden.f32(), set_name, top_k);
}

} // namespace omniseed
