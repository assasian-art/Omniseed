// =============================================================================
//  OmniSeed — runtime/introspection.cpp
//  Self-awareness + metacognition. Structured introspection over REAL runtime
//  state: goals, decision log, calibration. No claims beyond the evidence.
// =============================================================================
#include "omniseed/runtime/introspection.h"
#include "omniseed/core/platform.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace omniseed {

// ===========================================================================
// SelfModel — the honest capability map
// ===========================================================================
static const SelfModel::Capability kCapabilities[] = {
    {"language core (RWKV-7 0.1B int8)", "real"},
    {"speech recognition (whisper-tiny)", "real"},
    {"technical indicators + signals", "real"},
    {"risk math (Kelly, stops, drawdown)", "real"},
    {"backtesting (no look-ahead)", "real"},
    {"paper trading simulator", "real"},
    {"news RSS + lexicon sentiment", "real"},
    {"multi-agent coordination", "real"},
    {"self-model / goal tracking", "real"},
    {"confidence calibration", "real"},
    {"live market data feed", "absent"},
    {"real broker execution", "absent"},
    {"fundamental data (SEC/earnings)", "absent"},
    {"deep cloud reasoning", "optional"},
    {"vision backbone (full)", "partial"},
};
const SelfModel::Capability* SelfModel::capabilities() { return kCapabilities; }
const size_t SelfModel::capability_count() {
    return sizeof(kCapabilities) / sizeof(kCapabilities[0]);
}

// ===========================================================================
// GoalTracker
// ===========================================================================
void GoalTracker::add(const Goal& g) {
    for (Goal& existing : goals_)
        if (existing.text == g.text) { existing = g; return; }
    goals_.push_back(g);
}

bool GoalTracker::update(const std::string& metric, double current) {
    for (Goal& g : goals_)
        if (g.metric == metric) { g.current = current; return true; }
    return false;
}

bool GoalTracker::deactivate(const std::string& text) {
    for (Goal& g : goals_)
        if (g.text == text) { g.active = false; return true; }
    return false;
}

std::vector<Goal> GoalTracker::active_goals() const {
    std::vector<Goal> out;
    for (const Goal& g : goals_)
        if (g.active) out.push_back(g);
    std::sort(out.begin(), out.end(),
              [](const Goal& a, const Goal& b) { return a.priority > b.priority; });
    return out;
}

bool GoalTracker::save(const std::string& path) const {
    std::FILE* f = platform::open_file_c(path.c_str(), "wb");
    if (!f) return false;
    const uint32_t magic = 0x4F4B4744u;              // 'OKGD'
    std::fwrite(&magic, 4, 1, f);
    const uint32_t n = static_cast<uint32_t>(goals_.size());
    std::fwrite(&n, 4, 1, f);
    for (const Goal& g : goals_) {
        const uint32_t tl = static_cast<uint32_t>(g.text.size());
        const uint32_t ml = static_cast<uint32_t>(g.metric.size());
        std::fwrite(&tl, 4, 1, f);
        std::fwrite(g.text.data(), 1, tl, f);
        std::fwrite(&ml, 4, 1, f);
        std::fwrite(g.metric.data(), 1, ml, f);
        std::fwrite(&g.target, 8, 1, f);
        std::fwrite(&g.current, 8, 1, f);
        std::fwrite(&g.priority, 8, 1, f);      // double
        std::fwrite(&g.active, 1, 1, f);
    }
    std::fclose(f);
    return true;
}

bool GoalTracker::load(const std::string& path) {
    std::FILE* f = platform::open_file_c(path.c_str(), "rb");
    if (!f) return false;
    uint32_t magic = 0, n = 0;
    if (std::fread(&magic, 4, 1, f) != 1 || magic != 0x4F4B4744u ||
        std::fread(&n, 4, 1, f) != 1) { std::fclose(f); return false; }
    goals_.clear();
    goals_.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t tl = 0, ml = 0;
        Goal g;
        if (std::fread(&tl, 4, 1, f) != 1 || tl > 4096) { std::fclose(f); return false; }
        g.text.resize(tl);
        if (tl && std::fread(g.text.data(), 1, tl, f) != tl) { std::fclose(f); return false; }
        if (std::fread(&ml, 4, 1, f) != 1 || ml > 256) { std::fclose(f); return false; }
        g.metric.resize(ml);
        if (ml && std::fread(g.metric.data(), 1, ml, f) != ml) { std::fclose(f); return false; }
        if (std::fread(&g.target, 8, 1, f) != 1 ||
            std::fread(&g.current, 8, 1, f) != 1 ||
            std::fread(&g.priority, 8, 1, f) != 1 ||   // double
            std::fread(&g.active, 1, 1, f) != 1) { std::fclose(f); return false; }
        goals_.push_back(std::move(g));
    }
    std::fclose(f);
    return true;
}

// ===========================================================================
// PurposeReflection
// ===========================================================================
const char* PurposeReflection::mission() {
    return "Minimize losses through risk discipline; act only on stated "
           "evidence; surface every assumption; never fake certainty.";
}

PurposeReflection::Reflection PurposeReflection::reflect(
        const ActionRationale& log, const GoalTracker& goals) {
    Reflection r;
    r.at_unix = static_cast<uint64_t>(platform::now_ms() / 1000.0);
    r.question = "Why am I doing this? What is my objective?";

    // Ground the answer in the actual decision log + goals.
    const auto& entries = log.entries();
    const auto active = goals.active_goals();
    std::string answer = "My mission: ";
    answer += mission();
    answer += " Currently pursuing " + std::to_string(active.size()) +
              " active goal(s)";
    if (!active.empty()) {
        answer += ", top: \"" + active.front().text + "\"";
        answer += " (metric " + active.front().metric + " at " +
                  std::to_string(active.front().current).substr(0, 6) +
                  " vs target " +
                  std::to_string(active.front().target).substr(0, 6) + ")";
    }
    answer += ". My last " + std::to_string(entries.size()) +
              " logged decision(s) are the evidence of that mission in "
              "practice; each carries its stated confidence so my calibration "
              "can be audited.";
    if (!entries.empty()) {
        const RationaleEntry& last = entries.front();
        answer += " Most recent: I chose " + last.action + " because " +
                  last.because + ".";
    }
    r.answer = answer;
    return r;
}

// ===========================================================================
// ActionRationale
// ===========================================================================
void ActionRationale::log(const std::string& actor, const std::string& action,
                          const std::string& because, double confidence) {
    RationaleEntry e;
    e.at_unix = static_cast<uint64_t>(platform::now_ms() / 1000.0);
    e.actor = actor;
    e.action = action;
    e.because = because;
    e.confidence = std::max(0.0, std::min(1.0, confidence));
    entries_.push_front(std::move(e));
    while (entries_.size() > cfg_.max_entries) entries_.pop_back();
}

bool ActionRationale::resolve_latest(double outcome) {
    for (RationaleEntry& e : entries_)
        if (!e.outcome_known) {
            e.outcome = outcome;
            e.outcome_known = true;
            return true;
        }
    return false;
}

ActionRationale::Calibration ActionRationale::calibration() const {
    Calibration c;
    double conf_sum = 0.0, win_rate = 0.0;
    size_t n = 0;
    for (const RationaleEntry& e : entries_) {
        if (!e.outcome_known) continue;
        conf_sum += e.confidence;
        win_rate += e.outcome > 0.0 ? 1.0 : 0.0;
        ++n;
    }
    if (n == 0) return c;                       // not_calibrated = true
    c.not_calibrated = false;
    c.mean_confidence = conf_sum / static_cast<double>(n);
    c.realized_rate = win_rate / static_cast<double>(n);
    c.gap = c.mean_confidence - c.realized_rate;
    c.samples = n;
    return c;
}

bool ActionRationale::save(const std::string& path) const {
    std::FILE* f = platform::open_file_c(path.c_str(), "wb");
    if (!f) return false;
    const uint32_t magic = 0x4F4B5241u;              // 'OKRA'
    std::fwrite(&magic, 4, 1, f);
    const uint32_t n = static_cast<uint32_t>(entries_.size());
    std::fwrite(&n, 4, 1, f);
    for (const RationaleEntry& e : entries_) {
        const uint32_t al = static_cast<uint32_t>(e.actor.size());
        const uint32_t xl = static_cast<uint32_t>(e.action.size());
        const uint32_t bl = static_cast<uint32_t>(e.because.size());
        std::fwrite(&e.at_unix, 8, 1, f);
        std::fwrite(&al, 4, 1, f); std::fwrite(e.actor.data(), 1, al, f);
        std::fwrite(&xl, 4, 1, f); std::fwrite(e.action.data(), 1, xl, f);
        std::fwrite(&bl, 4, 1, f); std::fwrite(e.because.data(), 1, bl, f);
        std::fwrite(&e.confidence, 4, 1, f);
        std::fwrite(&e.outcome, 8, 1, f);
        std::fwrite(&e.outcome_known, 1, 1, f);
    }
    std::fclose(f);
    return true;
}

bool ActionRationale::load(const std::string& path) {
    std::FILE* f = platform::open_file_c(path.c_str(), "rb");
    if (!f) return false;
    uint32_t magic = 0, n = 0;
    if (std::fread(&magic, 4, 1, f) != 1 || magic != 0x4F4B5241u ||
        std::fread(&n, 4, 1, f) != 1) { std::fclose(f); return false; }
    entries_.clear();
    for (uint32_t i = 0; i < n && i < 65536; ++i) {
        uint32_t al = 0, xl = 0, bl = 0;
        RationaleEntry e;
        bool ok = std::fread(&e.at_unix, 8, 1, f) == 1;
        ok = ok && std::fread(&al, 4, 1, f) == 1 && al <= 256;
        if (ok) { e.actor.resize(al); ok = !al || std::fread(e.actor.data(), 1, al, f) == al; }
        ok = ok && std::fread(&xl, 4, 1, f) == 1 && xl <= 4096;
        if (ok) { e.action.resize(xl); ok = !xl || std::fread(e.action.data(), 1, xl, f) == xl; }
        ok = ok && std::fread(&bl, 4, 1, f) == 1 && bl <= 4096;
        if (ok) { e.because.resize(bl); ok = !bl || std::fread(e.because.data(), 1, bl, f) == bl; }
        ok = ok && std::fread(&e.confidence, 4, 1, f) == 1;
        ok = ok && std::fread(&e.outcome, 8, 1, f) == 1;
        ok = ok && std::fread(&e.outcome_known, 1, 1, f) == 1;
        if (!ok) { std::fclose(f); return false; }
        entries_.push_back(std::move(e));
    }
    std::fclose(f);
    return true;
}

// ===========================================================================
// Metacognition
// ===========================================================================
std::vector<KnowledgeGap> Metacognition::default_gaps() {
    return {
        {"fundamentals (SEC filings, earnings transcripts)",
         "price-only signals miss the 'why' behind moves",
         "attach a document reader to the research agent when offline data "
         "is available"},
        {"live market data",
         "analysis runs on downloaded CSV snapshots; staleness is invisible "
         "to the signal math",
         "wire a user-provided quote adapter; always check bar timestamps"},
        {"real execution",
         "paper fills assume 2 bps slippage; real books are worse in stress",
         "keep paper-only until a broker adapter is deliberately enabled"},
        {"news beyond the lexicon",
         "sentiment is bag-of-words; sarcasm, negation and context are "
         "invisible",
         "route important headlines through the cloud bridge (optional) or "
         "the local LLM"},
        {"regime awareness",
         "indicators trained by experience on 2015-2026-style markets may "
         "fail in unseen regimes",
         "the backtester only proves the past; treat every Sharpe number as "
         "a lower bound on uncertainty"},
    };
}

Metacognition::State Metacognition::evaluate(const GoalTracker& goals,
                                             const ActionRationale& log,
                                             double realized_sharpe,
                                             size_t observations) {
    State s;
    s.gaps = default_gaps();

    // Strategy self-evaluation from the realized Sharpe + sample size.
    const ActionRationale::Calibration cal = log.calibration();
    s.calibrated_decisions = cal.samples;
    s.calibration_gap = cal.gap;
    s.overconfident = !cal.not_calibrated && cal.gap > 0.10 && cal.samples >= 5;

    // Confidence in the strategy scales with evidence: a Sharpe from 30
    // observations means little; one from 500 means more. Honest shrinkage.
    const double evidence = std::min(1.0,
        static_cast<double>(observations) / 500.0);
    s.confidence_in_strategy = std::max(0.0, std::min(1.0,
        (realized_sharpe > 0.0 ? realized_sharpe / 2.0 : 0.0) * evidence));

    char buf[160];
    std::snprintf(buf, sizeof(buf),
                  "strategy sharpe %.2f over %zu observations "
                  "(evidence weight %.0f%%); calibration %s",
                  realized_sharpe, observations, evidence * 100.0,
                  cal.not_calibrated
                      ? "unavailable (no resolved decisions yet)"
                      : (s.overconfident
                          ? "OVERCONFIDENT — reduce stated confidence"
                          : "within tolerance"));
    s.strategy_assessment = buf;

    // Goal-aware nudge: if a goal wants drawdown control and we are
    // overconfident, say so.
    for (const Goal& g : goals.active_goals()) {
        if (g.metric == "max_drawdown_pct" && s.overconfident) {
            KnowledgeGap kg;
            kg.topic = "overconfidence vs drawdown goal";
            kg.why_it_matters =
                "calibration gap " + std::to_string(cal.gap).substr(0, 5) +
                " while pursuing " + g.text;
            kg.suggested_action = "cut position sizing until the gap closes";
            s.gaps.push_back(kg);
        }
    }
    return s;
}

} // namespace omniseed
