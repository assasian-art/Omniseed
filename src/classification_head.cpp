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
#include <cstdio>
#include <cstring>
#include <string>

namespace omniseed {

namespace {
// Little-endian, self-describing, same stance as DecisionHead: OmniSeed state
// files are native-LE by contract, so no byte swapping is performed.
constexpr char kMagic[8] = {'O', 'M', 'N', 'I', 'S', 'C', 'H', '1'};
// v1 is the first and only writer so far. The calibration trailer is part of
// v1 rather than a later addition, because unlike DecisionHead this head had
// NO persistence at all before now — there is nothing older to stay
// compatible with, so the format starts correct instead of starting at v1 and
// immediately needing a v2.
constexpr int32_t kFormatVersion = 1;
constexpr int32_t kMaxDim = 1 << 20;
constexpr int32_t kMaxSets = 1 << 12;
constexpr int32_t kMaxLabels = 1 << 16;
constexpr int32_t kMaxStr = 1 << 12;

template <typename T>
bool wr(FILE* f, const T& v) { return std::fwrite(&v, sizeof(T), 1, f) == 1; }

template <typename T>
bool rd(FILE* f, T& v) { return std::fread(&v, sizeof(T), 1, f) == 1; }

// A non-finite or <= 0 temperature is normalised to 1.0 rather than stored:
// dividing every logit by 0 or NaN destroys the distribution, and falling back
// to "no scaling" leaves the head in a well-defined, merely less useful state.
float sane_temperature(float t) { return (std::isfinite(t) && t > 0.0f) ? t : 1.0f; }

bool wr_str(FILE* f, const std::string& s) {
    const int32_t n = static_cast<int32_t>(s.size());
    if (!wr(f, n)) return false;
    if (n == 0) return true;
    return std::fwrite(s.data(), 1, static_cast<size_t>(n), f) ==
           static_cast<size_t>(n);
}

bool rd_str(FILE* f, std::string& s) {
    int32_t n = 0;
    if (!rd(f, n) || n < 0 || n > kMaxStr) return false;
    s.assign(static_cast<size_t>(n), '\0');
    if (n == 0) return true;
    return std::fread(&s[0], 1, static_cast<size_t>(n), f) ==
           static_cast<size_t>(n);
}
}  // namespace

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
    // A fresh init() is a fresh head: no calibration, no temperature. load()
    // re-applies both after calling this.
    default_temperature_ = 1.0f;
    temps_.clear();
    calib_samples_.clear();
    calib_error_.clear();
    loaded_        = false;

    if (n_embd <= 0) {
        error_ = "ClassificationHead::init: n_embd must be > 0";
        provenance_ = "error: " + error_;
        return false;
    }
    E_ = n_embd;
    seed_ = seed;
    scratch_.clear();
    ready_ = true;
    refresh_provenance();
    return true;
}

// ---------------------------------------------------------------------------
// Provenance — one composer, so no call site can leave a stale claim behind.
// ---------------------------------------------------------------------------
std::string ClassificationHead::calibration_clause() const {
    if (!calibrated()) return std::string();
    char buf[128];
    std::snprintf(buf, sizeof(buf), " — calibrated (n=%d, pooled ece=%.2f%%, %d/%d sets)",
                  calibration_samples(),
                  static_cast<double>(calibration_error() * 100.0f),
                  measured_sets(), static_cast<int>(sets_.size()));
    return std::string(buf);
}

int32_t ClassificationHead::measured_sets() const {
    int32_t n = 0;
    for (size_t i = 0; i < calib_error_.size(); ++i)
        if (calib_samples_[i] > 0 && calib_error_[i] >= 0.0f) ++n;
    return n;
}

void ClassificationHead::refresh_provenance() {
    std::string base;
    if (!ready_) {
        base = "uninitialised";
    } else if (loaded_) {
        base = trained() ? "loaded fitted projection"
                         : "loaded seeded placeholder";
    } else if (fitted_rows_ <= 0) {
        // "UNTRAINED" is asserted by the test suite; keep the word exact.
        base = "seeded placeholder (UNTRAINED)";
    } else if (trained()) {
        base = "fitted (all label rows supplied)";
    } else {
        base = "partially fitted placeholder";
    }
    char buf[192];
    std::snprintf(buf, sizeof(buf),
                  " — seed %u, E=%d, %d label sets / %d labels (%d fitted)",
                  seed_, E_, static_cast<int>(sets_.size()), total_labels_,
                  fitted_rows_);
    provenance_ = base + buf + calibration_clause();
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
    // Calibration is per set, so these grow with sets_, not with labels.
    temps_.push_back(default_temperature_);
    calib_samples_.push_back(0);
    calib_error_.push_back(-1.0f);

    // Only the new rows need seeding, but re-seeding all of them is cheap and
    // keeps the function total (a caller may add a set after a load()).
    seed_weights(seed_);
    refresh_provenance();
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
    // The projection is no longer purely what came off disk.
    loaded_ = false;
    refresh_provenance();
}

bool ClassificationHead::trained() const {
    return ready_ && total_labels_ > 0 && fitted_rows_ == total_labels_;
}

// ---------------------------------------------------------------------------
// Calibration
// ---------------------------------------------------------------------------
void ClassificationHead::set_temperature(float t) {
    default_temperature_ = sane_temperature(t);
    for (float& v : temps_) v = default_temperature_;
    refresh_provenance();
}

void ClassificationHead::set_temperature(int32_t set_index, float t) {
    if (set_index < 0 || set_index >= static_cast<int32_t>(temps_.size())) return;
    temps_[static_cast<size_t>(set_index)] = sane_temperature(t);
    refresh_provenance();
}

float ClassificationHead::temperature(int32_t set_index) const {
    if (set_index < 0 || set_index >= static_cast<int32_t>(temps_.size()))
        return default_temperature_;
    return temps_[static_cast<size_t>(set_index)];
}

void ClassificationHead::set_calibration(int32_t samples, float ece) {
    const bool bad = samples <= 0 || !std::isfinite(ece) || ece < 0.0f;
    for (size_t i = 0; i < calib_samples_.size(); ++i) {
        calib_samples_[i] = bad ? 0 : samples;
        calib_error_[i]   = bad ? -1.0f : ece;
    }
    refresh_provenance();
}

void ClassificationHead::set_calibration(int32_t set_index, int32_t samples, float ece) {
    if (set_index < 0 || set_index >= static_cast<int32_t>(calib_samples_.size())) return;
    const bool bad = samples <= 0 || !std::isfinite(ece) || ece < 0.0f;
    calib_samples_[static_cast<size_t>(set_index)] = bad ? 0 : samples;
    calib_error_[static_cast<size_t>(set_index)]   = bad ? -1.0f : ece;
    refresh_provenance();
}

bool ClassificationHead::calibrated() const { return measured_sets() > 0; }

int32_t ClassificationHead::calibration_samples(int32_t set_index) const {
    if (set_index < 0 || set_index >= static_cast<int32_t>(calib_samples_.size()))
        return 0;
    return calib_samples_[static_cast<size_t>(set_index)];
}

float ClassificationHead::calibration_error(int32_t set_index) const {
    if (set_index < 0 || set_index >= static_cast<int32_t>(calib_error_.size()))
        return -1.0f;
    return calib_error_[static_cast<size_t>(set_index)];
}

// The pooled value is an n-weighted mean over the MEASURED sets, so a set that
// was never calibrated neither dilutes nor inflates the summary. Returns -1
// when nothing was measured, matching the per-set convention.
float ClassificationHead::calibration_error() const {
    double num = 0.0;
    int64_t den = 0;
    for (size_t i = 0; i < calib_error_.size(); ++i) {
        if (calib_samples_[i] > 0 && calib_error_[i] >= 0.0f) {
            num += static_cast<double>(calib_error_[i]) * calib_samples_[i];
            den += calib_samples_[i];
        }
    }
    if (den <= 0) return -1.0f;
    return static_cast<float>(num / static_cast<double>(den));
}

int32_t ClassificationHead::calibration_samples() const {
    int64_t n = 0;
    for (size_t i = 0; i < calib_samples_.size(); ++i)
        if (calib_samples_[i] > 0 && calib_error_[i] >= 0.0f)
            n += calib_samples_[i];
    return static_cast<int32_t>(n);
}

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------
bool ClassificationHead::save(const std::string& path) const {
    if (!ready_) return false;
    FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) return false;

    bool ok = std::fwrite(kMagic, 1, 8, f) == 8;
    ok = ok && wr(f, kFormatVersion);
    ok = ok && wr(f, E_);
    ok = ok && wr(f, static_cast<int32_t>(sets_.size()));
    for (const LabelSet& s : sets_) {
        if (!ok) break;
        ok = wr_str(f, s.name);
        ok = ok && wr(f, static_cast<int32_t>(s.labels.size()));
        for (const std::string& l : s.labels) { ok = ok && wr_str(f, l); if (!ok) break; }
    }
    ok = ok && wr(f, total_labels_);
    ok = ok && wr(f, fitted_rows_);
    // Calibration trailer: one temperature, sample count and ECE PER SET,
    // because a single scalar cannot serve sets whose logit scales differ by
    // an order of magnitude (see the header).
    ok = ok && wr(f, default_temperature_);
    for (float v : temps_)           { ok = ok && wr(f, v); if (!ok) break; }
    for (int32_t v : calib_samples_) { ok = ok && wr(f, v); if (!ok) break; }
    for (float v : calib_error_)     { ok = ok && wr(f, v); if (!ok) break; }

    if (ok && !proj_.empty())
        ok = std::fwrite(proj_.data(), sizeof(float), proj_.size(), f) == proj_.size();
    if (ok && !bias_.empty())
        ok = std::fwrite(bias_.data(), sizeof(float), bias_.size(), f) == bias_.size();
    if (ok && !fitted_.empty())
        ok = std::fwrite(fitted_.data(), 1, fitted_.size(), f) == fitted_.size();

    std::fclose(f);
    return ok;
}

bool ClassificationHead::load(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        error_ = "classification head: cannot open " + path;
        return false;
    }

    char magic[8] = {0};
    bool ok = std::fread(magic, 1, 8, f) == 8 &&
              std::memcmp(magic, kMagic, 8) == 0;

    int32_t version = 0, E = 0, n_sets = 0;
    ok = ok && rd(f, version) && version == kFormatVersion;
    ok = ok && rd(f, E) && rd(f, n_sets);

    if (!ok || E <= 0 || E > kMaxDim || n_sets < 0 || n_sets > kMaxSets) {
        std::fclose(f);
        error_ = "classification head: malformed header in " + path;
        return false;
    }

    // Read the label sets first, so the geometry can be rebuilt from the file
    // rather than assumed.
    std::vector<LabelSet> sets(static_cast<size_t>(n_sets));
    for (int32_t si = 0; ok && si < n_sets; ++si) {
        ok = rd_str(f, sets[static_cast<size_t>(si)].name);
        int32_t nl = 0;
        ok = ok && rd(f, nl) && nl > 0 && nl <= kMaxLabels;
        if (!ok) break;
        sets[static_cast<size_t>(si)].labels.resize(static_cast<size_t>(nl));
        for (int32_t li = 0; li < nl && ok; ++li)
            ok = rd_str(f, sets[static_cast<size_t>(si)].labels[static_cast<size_t>(li)]);
    }
    int32_t total = 0, fitted = 0;
    float   def_temp = 1.0f;
    std::vector<float>   temps(static_cast<size_t>(n_sets), 1.0f);
    std::vector<int32_t> cals(static_cast<size_t>(n_sets), 0);
    std::vector<float>   eces(static_cast<size_t>(n_sets), -1.0f);
    ok = ok && rd(f, total) && rd(f, fitted);
    ok = ok && rd(f, def_temp);
    for (int32_t i = 0; ok && i < n_sets; ++i) ok = rd(f, temps[static_cast<size_t>(i)]);
    for (int32_t i = 0; ok && i < n_sets; ++i) ok = rd(f, cals[static_cast<size_t>(i)]);
    for (int32_t i = 0; ok && i < n_sets; ++i) ok = rd(f, eces[static_cast<size_t>(i)]);

    if (!ok || total <= 0 || total > kMaxDim) {
        std::fclose(f);
        error_ = "classification head: malformed payload in " + path;
        return false;
    }

    // Rebuild geometry + scratch, then re-register the sets in file order.
    // add_label_set() reseeds each row; the projection is overwritten below.
    if (!init(E, 777u)) {
        std::fclose(f);
        return false;
    }
    for (const LabelSet& s : sets) {
        if (add_label_set(s.name, s.labels) < 0) {
            std::fclose(f);
            error_ = "classification head: duplicate or empty label set '" +
                     s.name + "' in " + path;
            return false;
        }
    }
    if (total_labels_ != total) {
        std::fclose(f);
        ready_ = false;   // fail closed: never leave a half-built head usable
        error_ = "classification head: label count mismatch in " + path +
                 " (file " + std::to_string(total) + ", rebuilt " +
                 std::to_string(total_labels_) + ")";
        return false;
    }

    ok = std::fread(proj_.data(), sizeof(float), proj_.size(), f) == proj_.size();
    if (ok)
        ok = std::fread(bias_.data(), sizeof(float), bias_.size(), f) == bias_.size();
    if (ok)
        ok = std::fread(fitted_.data(), 1, fitted_.size(), f) == fitted_.size();
    if (!ok) {
        std::fclose(f);
        ready_ = false;   // fail closed
        error_ = "classification head: truncated payload in " + path;
        return false;
    }
    std::fclose(f);

    // The stored count is a cross-check, not the source of truth: trained()
    // reads the per-row flags, so a file whose header count disagrees with its
    // flags is corrupt and is refused rather than silently re-interpreted.
    int32_t counted = 0;
    for (uint8_t b : fitted_) if (b) ++counted;
    if (fitted != counted) {
        ready_ = false;   // fail closed
        error_ = "classification head: fitted-row count disagrees with flags in " +
                 path + " (header " + std::to_string(fitted) + ", flags " +
                 std::to_string(counted) + ")";
        return false;
    }
    fitted_rows_ = counted;

    // Apply the per-set calibration. add_label_set() pushed a default for each
    // set, so the vectors are already the right length; this overwrites them.
    default_temperature_ = sane_temperature(def_temp);
    for (size_t i = 0; i < temps_.size() && i < temps.size(); ++i) {
        temps_[i]         = sane_temperature(temps[i]);
        calib_samples_[i] = cals[i] > 0 ? cals[i] : 0;
        calib_error_[i]   = (std::isfinite(eces[i]) && eces[i] >= 0.0f) ? eces[i] : -1.0f;
    }
    loaded_ = true;
    refresh_provenance();
    return true;
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
    // The temperature is applied to the LOGIT, before the max/softmax. T == 1
    // is an exact no-op, so an uncalibrated head takes byte-for-byte the path
    // it always did. The guard is repeated here (not only in
    // set_temperature()) because a blob from another tool must not be able to
    // inject a NaN into every distribution.
    const float t_raw = (static_cast<size_t>(set_index) < temps_.size())
                            ? temps_[static_cast<size_t>(set_index)]
                            : default_temperature_;
    const float T = sane_temperature(t_raw);
    const float inv_t = 1.0f / T;
    float max_logit = -3.4e38f;
    for (int32_t li = 0; li < L; ++li) {
        const size_t row = static_cast<size_t>(s.offset) + static_cast<size_t>(li);
        const float* w = &proj_[row * static_cast<size_t>(E_)];
        float acc = bias_[row];
        for (int32_t e = 0; e < E_; ++e) acc += w[e] * hidden[e];
        acc *= inv_t;
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
