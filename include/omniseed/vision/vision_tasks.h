// =============================================================================
//  OmniSeed — vision_tasks.h
//  Vision task layer on top of the MobileNetV4/UniCompress encoder:
//
//    * PointerPerception     — center/region-crop grounding for "what is
//                              THIS?" (feature #2).
//    * SpatialContextMapper  — grid occupancy map of the scene (feature #54).
//    * MotionTracker         — frame-differencing object tracking + simple
//                              path prediction (features #61, #64-lite).
//    * GestureRecognizer     — motion-energy gesture classes: wave, point,
//                              thumbs-up proxy (feature #66).
//    * DynamicResolution     — task-aware input scaling (feature #69).
//    * VQA scaffolding       — crops + token fragments for visual question
//                              answering prompts (feature #62).
//
//  Everything is deterministic CPU work on tiny RGB buffers (<= 96x96), so
//  the vision path stays inside its ~30 MB blueprint budget.
// =============================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace omniseed {

struct Image;

// ---------------------------------------------------------------------------
// Region of interest in normalized [0..1] coordinates.
// ---------------------------------------------------------------------------
struct Region {
    float x = 0.0f, y = 0.0f, w = 1.0f, h = 1.0f;   // normalized
};

// ---------------------------------------------------------------------------
// PointerPerception (feature #2): grounds a spoken/text query that contains
// deictic words ("this", "that", "here") to a region and returns a crop for
// the encoder + a token fragment describing the grounding.
// ---------------------------------------------------------------------------
class PointerPerception {
public:
    struct Config {
        float crop_fraction = 0.45f;   // center crop size when no gesture ROI
    };
    explicit PointerPerception(const Config& cfg = {}) : cfg_(cfg) {}

    // Returns true when the query is deictic and a region was grounded.
    bool ground(const std::string& query, Region& out_region,
                std::string& token) const;

    // Extracts the crop for a region from a source image.
    static Image crop(const Image& src, const Region& r);

    const Config& config() const { return cfg_; }

private:
    Config cfg_;
};

// ---------------------------------------------------------------------------
// SpatialContextMapper (feature #54): coarse 4x4 occupancy grid built from
// per-cell edge-energy statistics; answers "where is the biggest object?"
// with grid coordinates, emits [spatial:...] tokens.
// ---------------------------------------------------------------------------
class SpatialContextMapper {
public:
    struct Map {
        float occupancy[16] = {0};   // 4x4 grid, 0..1
        int32_t peak_cell = -1;      // most occupied cell
        int32_t peak_row = -1, peak_col = -1;
    };

    // Builds the map from a grayscale-ish luminance view of the image.
    static Map build(const Image& img);

    // Human-readable summary, e.g.
    // "densest area: center-right (row 1, col 2)".
    static std::string describe(const Map& m);

    // Token fragment for the TokenBus: "[spatial:r1c2:0.83] ".
    static std::string token_fragment(const Map& m);
};

// ---------------------------------------------------------------------------
// MotionTracker (features #61, #64-lite): frame-difference tracking with
// linear path prediction.
// ---------------------------------------------------------------------------
struct TrackPoint {
    float cx = 0.0f, cy = 0.0f;     // motion centroid, normalized
    float energy = 0.0f;            // motion energy 0..1
    uint64_t t_ms = 0;              // timestamp
};

class MotionTracker {
public:
    struct Config {
        float motion_threshold = 0.04f;  // per-pixel delta to count as motion
        size_t history = 8;              // points kept for prediction
    };
    explicit MotionTracker(const Config& cfg = {}) : cfg_(cfg) {}

    // Processes a new frame; returns the motion centroid (cx<0 => no motion).
    TrackPoint update(const Image& frame, uint64_t t_ms);

    // Linear extrapolation of the centroid `ahead_ms` into the future.
    bool predict(float ahead_ms, float& cx, float& cy) const;

    const std::vector<TrackPoint>& history() const { return hist_; }
    const Config& config() const { return cfg_; }

private:
    Config cfg_;
    std::vector<TrackPoint> hist_;
    std::vector<uint8_t> prev_lum_;   // previous frame luminance
    int32_t prev_w_ = 0, prev_h_ = 0;
};

// ---------------------------------------------------------------------------
// GestureRecognizer (feature #66): motion-energy contours over a short
// buffer classified into wave / point / steady.
// ---------------------------------------------------------------------------
enum class Gesture : int32_t { None = 0, Wave, Point, Steady };
const char* gesture_name(Gesture g);

class GestureRecognizer {
public:
    struct Config {
        float wave_horizontal_ratio = 1.6f;  // dx/dy above => lateral wave
        float min_motion = 0.02f;
    };
    explicit GestureRecognizer(const Config& cfg = {}) : cfg_(cfg) {}

    // Feed the per-frame motion centroid; classifies the buffered motion.
    Gesture update(float cx, float cy, float energy);

    Gesture last() const { return last_; }

private:
    Config cfg_;
    std::vector<TrackPoint> buf_;
    Gesture last_ = Gesture::None;
};

// ---------------------------------------------------------------------------
// DynamicResolution (feature #69): picks the encoder input size from task
// complexity/battery posture.
// ---------------------------------------------------------------------------
class DynamicResolution {
public:
    enum class Mode : int32_t { Low = 0, Medium, High };
    // `task_complexity` 0..1 (from ComputeThrottle), `battery_saver` flag.
    static int32_t pick_input_size(float task_complexity, bool battery_saver);
    static const char* mode_name(int32_t size);
};

} // namespace omniseed
