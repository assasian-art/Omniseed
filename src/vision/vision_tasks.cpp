// =============================================================================
//  OmniSeed — vision_tasks.cpp
//  Pointer grounding, spatial mapping, motion tracking, gestures,
//  dynamic resolution. All operate on tiny RGB buffers; no allocations
//  beyond the returned crop/map.
// =============================================================================
#include "omniseed/vision/vision_tasks.h"
#include "omniseed/vision/vision.h"
#include "omniseed/core/platform.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace omniseed {

// ===========================================================================
// PointerPerception
// ===========================================================================
bool PointerPerception::ground(const std::string& query, Region& out_region,
                               std::string& token) const {
    // Deictic detection (case-insensitive substring).
    std::string low;
    low.reserve(query.size());
    for (char c : query)
        low.push_back(static_cast<char>(
            std::tolower(static_cast<unsigned char>(c))));

    const bool deictic =
        low.find("this") != std::string::npos ||
        low.find("that") != std::string::npos ||
        low.find("here") != std::string::npos ||
        low.find("point") != std::string::npos;
    if (!deictic) return false;

    // Prefer an explicit direction, else center crop.
    const float f = cfg_.crop_fraction;
    if (low.find("left") != std::string::npos) {
        out_region = Region{0.0f, (1.0f - f) / 2.0f, f, f};
    } else if (low.find("right") != std::string::npos) {
        out_region = Region{1.0f - f, (1.0f - f) / 2.0f, f, f};
    } else if (low.find("top") != std::string::npos) {
        out_region = Region{(1.0f - f) / 2.0f, 0.0f, f, f};
    } else if (low.find("bottom") != std::string::npos) {
        out_region = Region{(1.0f - f) / 2.0f, 1.0f - f, f, f};
    } else {
        out_region = Region{(1.0f - f) / 2.0f, (1.0f - f) / 2.0f, f, f};
    }

    char buf[64];
    std::snprintf(buf, sizeof(buf), "[vision:pointer:%.2f,%.2f] ",
                  out_region.x + out_region.w / 2.0f,
                  out_region.y + out_region.h / 2.0f);
    token = buf;
    return true;
}

Image PointerPerception::crop(const Image& src, const Region& r) {
    if (!src.valid()) return Image{};
    const int32_t x0 = std::clamp(
        static_cast<int32_t>(r.x * src.width), 0, src.width - 1);
    const int32_t y0 = std::clamp(
        static_cast<int32_t>(r.y * src.height), 0, src.height - 1);
    const int32_t w = std::clamp(
        static_cast<int32_t>(r.w * src.width), 1, src.width - x0);
    const int32_t h = std::clamp(
        static_cast<int32_t>(r.h * src.height), 1, src.height - y0);

    Image out;
    out.width = w;
    out.height = h;
    out.rgb.resize(static_cast<size_t>(w) * h * 3);
    for (int32_t y = 0; y < h; ++y) {
        const uint8_t* src_row =
            src.rgb.data() + (static_cast<size_t>(y0 + y) * src.width + x0) * 3;
        std::memcpy(out.rgb.data() + static_cast<size_t>(y) * w * 3,
                    src_row, static_cast<size_t>(w) * 3);
    }
    return out;
}

// ===========================================================================
// SpatialContextMapper
// ===========================================================================
SpatialContextMapper::Map SpatialContextMapper::build(const Image& img) {
    Map m;
    if (!img.valid()) return m;

    const int32_t G = 4;
    double cell_energy[16] = {0};
    double cell_max[16] = {0};

    auto lum = [&](int32_t x, int32_t y) {
        const uint8_t* p =
            img.rgb.data() + (static_cast<size_t>(y) * img.width + x) * 3;
        return 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
    };

    // Simple gradient-magnitude energy per cell (edges ~ objects/structure).
    for (int32_t y = 1; y < img.height - 1; ++y) {
        for (int32_t x = 1; x < img.width - 1; ++x) {
            const double gx = lum(x + 1, y) - lum(x - 1, y);
            const double gy = lum(x, y + 1) - lum(x, y - 1);
            const double mag = std::sqrt(gx * gx + gy * gy);
            const int32_t col = std::min(G - 1, x * G / img.width);
            const int32_t row = std::min(G - 1, y * G / img.height);
            const int32_t cell = row * G + col;
            cell_energy[cell] += mag;
            cell_max[cell] = std::max(cell_max[cell], mag);
        }
    }

    double max_e = 1e-9;
    for (int32_t i = 0; i < 16; ++i)
        max_e = std::max(max_e, cell_energy[i]);
    double peak = -1.0;
    for (int32_t i = 0; i < 16; ++i) {
        m.occupancy[i] = static_cast<float>(cell_energy[i] / max_e);
        if (cell_energy[i] > peak) {
            peak = cell_energy[i];
            m.peak_cell = i;
            m.peak_row = i / G;
            m.peak_col = i % G;
        }
    }
    return m;
}

std::string SpatialContextMapper::describe(const Map& m) {
    if (m.peak_cell < 0) return "no structure detected";
    // Rows: top/middle/bottom; cols: left/center/right (3x3 wording on 4x4).
    const char* rows[4] = {"top", "upper-middle", "lower-middle", "bottom"};
    const char* cols[4] = {"left", "center-left", "center-right", "right"};
    char buf[128];
    std::snprintf(buf, sizeof(buf), "densest area: %s-%s (row %d, col %d)",
                  rows[std::min(3, m.peak_row)],
                  cols[std::min(3, m.peak_col)],
                  m.peak_row, m.peak_col);
    return buf;
}

std::string SpatialContextMapper::token_fragment(const Map& m) {
    if (m.peak_cell < 0) return "";
    char buf[48];
    std::snprintf(buf, sizeof(buf), "[spatial:r%dc%d:%.2f] ",
                  m.peak_row, m.peak_col, m.occupancy[m.peak_cell]);
    return buf;
}

// ===========================================================================
// MotionTracker
// ===========================================================================
TrackPoint MotionTracker::update(const Image& frame, uint64_t t_ms) {
    TrackPoint p;
    p.t_ms = t_ms;
    if (!frame.valid() || frame.width < 8 || frame.height < 8) {
        p.cx = p.cy = -1.0f;
        return p;
    }

    // Luminance delta vs the previous frame, downsampled 2x for speed.
    const int32_t W = frame.width, H = frame.height;
    std::vector<uint8_t> lum(static_cast<size_t>(W) * H);
    for (int32_t y = 0; y < H; ++y) {
        for (int32_t x = 0; x < W; ++x) {
            const uint8_t* px =
                frame.rgb.data() + (static_cast<size_t>(y) * W + x) * 3;
            lum[y * W + x] = static_cast<uint8_t>(
                0.299 * px[0] + 0.587 * px[1] + 0.114 * px[2]);
        }
    }

    if (prev_lum_.empty() || prev_w_ != W || prev_h_ != H) {
        prev_lum_ = lum;
        prev_w_ = W;
        prev_h_ = H;
        p.cx = p.cy = -1.0f;      // need two frames
        return p;
    }

    double sum = 0, sx = 0, sy = 0;
    const int32_t step = 2;
    for (int32_t y = 1; y < H - 1; y += step) {
        for (int32_t x = 1; x < W - 1; x += step) {
            const float d = std::fabs(
                static_cast<float>(lum[y * W + x]) -
                static_cast<float>(prev_lum_[y * W + x])) / 255.0f;
            if (d > cfg_.motion_threshold) {
                sum += d;
                sx += d * x;
                sy += d * y;
            }
        }
    }
    prev_lum_ = lum;

    if (sum <= 0.0) {
        p.cx = p.cy = -1.0f;
        p.energy = 0.0f;
    } else {
        p.cx = static_cast<float>(sx / sum / W);
        p.cy = static_cast<float>(sy / sum / H);
        p.energy = static_cast<float>(std::min(
            1.0, sum / (static_cast<double>(W / step) * (H / step))));
    }

    hist_.push_back(p);
    if (hist_.size() > cfg_.history) hist_.erase(hist_.begin());
    return p;
}

bool MotionTracker::predict(float ahead_ms, float& cx, float& cy) const {
    if (hist_.size() < 3) return false;
    const TrackPoint& a = hist_[hist_.size() - 3];
    const TrackPoint& b = hist_.back();
    const double dt = (b.t_ms - a.t_ms) / 1000.0;
    if (dt <= 0.0) return false;
    const float vx = (b.cx - a.cx) / static_cast<float>(dt);
    const float vy = (b.cy - a.cy) / static_cast<float>(dt);
    cx = std::clamp(b.cx + vx * (ahead_ms / 1000.0f), 0.0f, 1.0f);
    cy = std::clamp(b.cy + vy * (ahead_ms / 1000.0f), 0.0f, 1.0f);
    return true;
}

// ===========================================================================
// GestureRecognizer
// ===========================================================================
const char* gesture_name(Gesture g) {
    switch (g) {
        case Gesture::Wave:   return "wave";
        case Gesture::Point:  return "point";
        case Gesture::Steady: return "steady";
        case Gesture::None:   break;
    }
    return "none";
}

Gesture GestureRecognizer::update(float cx, float cy, float energy) {
    if (cx < 0.0f || energy < cfg_.min_motion) {
        buf_.clear();
        return last_ = Gesture::None;
    }
    TrackPoint p;
    p.cx = cx; p.cy = cy; p.energy = energy;
    p.t_ms = static_cast<uint64_t>(platform::now_ms());
    buf_.push_back(p);
    if (buf_.size() > 12) buf_.erase(buf_.begin());

    if (buf_.size() < 6) return last_;

    // Lateral vs vertical motion ratio over the buffer.
    float dx = 0.0f, dy = 0.0f;
    int32_t reversals = 0;
    float last_dx = 0.0f;
    for (size_t i = 1; i < buf_.size(); ++i) {
        const float ddx = buf_[i].cx - buf_[i - 1].cx;
        const float ddy = buf_[i].cy - buf_[i - 1].cy;
        dx += std::fabs(ddx);
        dy += std::fabs(ddy);
        if (ddx * last_dx < 0.0f) ++reversals;   // direction flips = waving
        if (std::fabs(ddx) > 0.01f) last_dx = ddx;
    }

    if (reversals >= 3 && dx / std::max(0.01f, dy) > cfg_.wave_horizontal_ratio)
        return last_ = Gesture::Wave;
    if (energy > 0.10f && dx < 0.03f && dy < 0.03f)
        return last_ = Gesture::Steady;          // held position = pointing
    return last_ = Gesture::None;
}

// ===========================================================================
// DynamicResolution
// ===========================================================================
int32_t DynamicResolution::pick_input_size(float task_complexity,
                                           bool battery_saver) {
    // Fast path: small inputs for trivial tasks / low battery.
    if (battery_saver || task_complexity < 0.25f) return 64;
    if (task_complexity < 0.65f) return 96;      // blueprint default
    return 128;                                   // deep analysis
}

const char* DynamicResolution::mode_name(int32_t size) {
    switch (size) {
        case 64:  return "low";
        case 96:  return "medium";
        case 128: return "high";
        default:  return "custom";
    }
}

} // namespace omniseed
