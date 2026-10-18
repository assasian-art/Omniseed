// =============================================================================
//  OmniSeed — src/runtime/calibration.cpp  (T111 / INNOVATIONS I8)
//
//  Minimal, dependency-free reader for the nightly calibration map. The
//  producer (tools/nightly_refit.py, json.dump(sort_keys=True, indent=2))
//  emits a small flat object — scanning it directly is deliberate: this
//  project has zero external dependencies, and the alternative (a JSON
//  library) is a large price for one well-specified file.
// =============================================================================
#include "omniseed/runtime/calibration.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace omniseed {
namespace {

// Find "key": <value-start-offset> or npos. Case-sensitive; the producer
// uses sort_keys output so keys appear exactly once.
std::size_t find_key(const std::string& s, const std::string& key) {
    const std::string pat = "\"" + key + "\"";
    std::size_t p = s.find(pat);
    if (p == std::string::npos) return std::string::npos;
    p = s.find(':', p + pat.size());
    if (p == std::string::npos) return std::string::npos;
    ++p;
    while (p < s.size() && (s[p] == ' ' || s[p] == '\n' || s[p] == '\t' ||
                            s[p] == '\r')) {
        ++p;
    }
    return p;
}

bool parse_bool(const std::string& s, const std::string& key, bool dflt) {
    const std::size_t p = find_key(s, key);
    if (p == std::string::npos) return dflt;
    if (s.compare(p, 4, "true") == 0) return true;
    if (s.compare(p, 5, "false") == 0) return false;
    return dflt;
}

long parse_long(const std::string& s, const std::string& key, long dflt) {
    const std::size_t p = find_key(s, key);
    if (p == std::string::npos) return dflt;
    char* end = nullptr;
    const long v = std::strtol(s.c_str() + p, &end, 10);
    if (end == s.c_str() + p) return dflt;
    return v;
}

std::string parse_string(const std::string& s, const std::string& key) {
    const std::size_t p = find_key(s, key);
    if (p == std::string::npos || p >= s.size() || s[p] != '"') return "";
    const std::size_t e = s.find('"', p + 1);
    if (e == std::string::npos) return "";
    return s.substr(p + 1, e - p - 1);
}

}  // namespace

bool CalibrationMap::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        loaded_ = false;
        return false;
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return load_from_string(ss.str());
}

bool CalibrationMap::load_from_string(const std::string& json) {
    bins_.clear();
    loaded_ = false;
    fingerprint_.clear();
    timestamp_ = 0;
    if (json.empty()) return false;
    // An unaccepted map is keep-old evidence, never a calibration.
    if (!parse_bool(json, "accepted", false)) return false;
    timestamp_ = parse_long(json, "timestamp", 0);
    fingerprint_ = parse_string(json, "corpus_fingerprint");

    const std::size_t p = find_key(json, "bins");
    if (p == std::string::npos || p >= json.size() || json[p] != '[') {
        return false;
    }
    // bins = [[conf, acc, count], [conf, acc, count], ...]
    std::size_t i = p + 1;
    while (i < json.size()) {
        while (i < json.size() && json[i] != '[' && json[i] != ']') ++i;
        if (i >= json.size() || json[i] == ']') break;
        const std::size_t e = json.find(']', i);
        if (e == std::string::npos) break;
        const std::string inner = json.substr(i + 1, e - i - 1);
        char* end = nullptr;
        const double conf = std::strtod(inner.c_str(), &end);
        if (end == inner.c_str()) return false;  // malformed -> fail closed
        const char* acc_p = inner.c_str() + (end - inner.c_str());
        while (*acc_p == ' ' || *acc_p == ',') ++acc_p;
        const double acc = std::strtod(acc_p, &end);
        if (end == acc_p) return false;
        bins_.push_back(Bin{conf, acc});
        i = e + 1;
    }
    std::sort(bins_.begin(), bins_.end(),
              [](const Bin& a, const Bin& b) { return a.conf < b.conf; });
    if (bins_.size() < 2) {
        bins_.clear();
        return false;  // a 1-bin "curve" is not a calibration
    }
    loaded_ = true;
    return true;
}

double CalibrationMap::calibrate(double conf) const {
    if (!loaded_ || bins_.empty()) return conf;  // honest identity fallback
    const Bin* best = &bins_.front();
    double best_d = std::abs(conf - best->conf);
    for (const Bin& b : bins_) {
        const double d = std::abs(conf - b.conf);
        if (d < best_d) {
            best_d = d;
            best = &b;
        }
    }
    return best->acc;
}

bool CalibrationMap::trustworthy(double conf, double min_acc) const {
    // Fail-closed: without an ACCEPTED map there is no measured accuracy,
    // so the gate stays shut. (Callers wanting the pre-map behaviour compare
    // raw confidence themselves — visibly, not through this gate.)
    if (!loaded_) return false;
    return calibrate(conf) >= min_acc;
}

}  // namespace omniseed
