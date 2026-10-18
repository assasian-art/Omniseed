// =============================================================================
//  OmniSeed — include/omniseed/runtime/calibration.h  (T111 / INNOVATIONS I8)
//
//  Runtime consumer for the nightly confidence->accuracy map
//  (tools/nightly_refit.py writes state/calibration_map.json).
//
//  WHY: FEEDBACK.md's honest-gap shape — the nightly loop re-fits a monotone
//  confidence->accuracy map, but until now NOTHING in C++ ever read it: the
//  product kept reporting raw head confidence as if it were probability.
//
//  FAIL-CLOSED CONTRACT (the invariant that needed code, not a policy
//  sentence): an UNLOADED, UNACCEPTED or MALFORMED map never degrades into
//  "pretend identity". calibrate() falls back to identity so reports stay
//  honest about what they are (raw confidence), but trustworthy() — the
//  gate a caller uses before acting — returns FALSE when no accepted map is
//  loaded. A stale or missing calibration can therefore never silently
//  wave through a decision that needed measured accuracy.
// =============================================================================
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace omniseed {

class CalibrationMap {
  public:
    // Load state/calibration_map.json. Returns true ONLY for an accepted
    // map with >= 2 bins; anything else (missing, malformed, accepted:false,
    // too few bins) leaves the map unloaded and the accessors fail-closed.
    bool load(const std::string& path);
    // Parse from an in-memory JSON string (test seam; same rules as load).
    bool load_from_string(const std::string& json);

    bool loaded() const { return loaded_; }

    // Nearest-bin calibrated accuracy for `conf`; identity (conf itself)
    // when unloaded — the REPORT is honest that it is raw confidence via
    // loaded()==false, never a fabricated probability.
    double calibrate(double conf) const;

    // The gate: true only when an ACCEPTED map is loaded AND its measured
    // accuracy at `conf` clears `min_acc`. Unloaded => false (fail-closed).
    bool trustworthy(double conf, double min_acc) const;

    const std::string& fingerprint() const { return fingerprint_; }
    long timestamp() const { return timestamp_; }
    std::size_t bin_count() const { return bins_.size(); }

  private:
    struct Bin {
        double conf;
        double acc;
    };
    std::vector<Bin> bins_;
    bool loaded_ = false;
    std::string fingerprint_;
    long timestamp_ = 0;
};

}  // namespace omniseed
