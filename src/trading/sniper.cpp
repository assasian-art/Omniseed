// =============================================================================
//  OmniSeed — trading/sniper.cpp
//  C++ implementation of the Sniper entry engine.
//
//  Faithful port of tools/monster/sniper_engine.py. Three behaviours carry the
//  module's whole meaning and are pinned by tests:
//
//    1. S IS EXACTLY THE MANDATE'S WEIGHTED SUM. The ensemble is wired in as a
//       veto only, AFTER the score is computed, and it never touches `score`.
//       An ensemble that can talk you into a trade can talk you into a bad one.
//
//    2. THE VETO ORDER IS SEMANTIC, NOT COSMETIC. `counter-regime` /
//       `high-vol-needs-mean-reversion` are decided first, then cross-asset,
//       then confluence, then the ensemble. The first veto that fires owns the
//       reason, so a counter-trend bar is reported as counter-trend even when
//       three other gates would also have blocked it.
//
//    3. THE 0.90 + 0.10*(votes - min_factors) LADDER. votes/6 caps the total at
//       ~0.81 and makes the 0.85 gate unreachable — see the header note.
// =============================================================================
#include "omniseed/trading/sniper.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>

namespace omniseed {
namespace trading {

namespace {

inline bool fin(double x) { return x == x && x != INFINITY && x != -INFINITY; }

std::string fmt(const char* f, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof(buf), f, ap);
    va_end(ap);
    return std::string(buf);
}

} // namespace

double regime_score_for_label(const std::string& label) {
    if (label == "trend_up") return 1.0;
    if (label == "range") return 0.5;
    if (label == "high_vol") return 0.25;
    if (label == "trend_down") return 0.0;
    // An unknown label is treated as `range`: neutral, never favourable. Being
    // surprised into a bullish 1.0 by a typo is exactly the failure mode a
    // label->score map exists to prevent.
    return 0.5;
}

bool regime_label_known(const std::string& label) {
    return label == "trend_up" || label == "range" || label == "high_vol" ||
           label == "trend_down";
}

std::string SniperVerdict::detail() const {
    std::string out = fmt("S=%.3f M=%.2f T=%.2f(%d) R=%.2f C=%.2f regime=%s", score,
                          micro, tech, static_cast<int>(votes), regime_score, cross,
                          regime.c_str());
    if (veto) out += " VETO=" + (veto_reason.empty() ? std::string("gate") : veto_reason);
    if (!factors.empty()) {
        std::string joined;
        for (size_t i = 0; i < factors.size(); ++i) {
            if (i) joined += "+";
            joined += factors[i];
        }
        out += " factors=" + joined;
    }
    if (fin(trend_score)) out += fmt(" trend=%.3f", trend_score);
    if (fin(half_life)) out += fmt(" hl=%.0f", half_life);
    if (conviction != 0.0 || agreement != 0.0) out += fmt(" ens=%+.2f/%.2f", conviction, agreement);
    return out;
}

// =============================================================================
// Prepared
// =============================================================================
Prepared::Prepared(const std::vector<Bar>& bars, const SniperConfig& cfg)
    : cfg_(cfg), bars_(bars), n_(bars.size()) {
    ts_.reserve(n_);
    for (const Bar& b : bars_) ts_.push_back(b.time);

    close_ = features::closes(bars_);
    high_ = features::highs(bars_);
    low_ = features::lows(bars_);
    vol_ = features::volumes(bars_);

    sma_fast_ = features::sma(close_, cfg_.sma_fast);
    sma_slow_ = features::sma(close_, cfg_.sma_slow);
    sma_regime_ = features::sma(close_, cfg_.sma_regime);
    rsi_ = features::rsi(close_, cfg_.rsi_period);

    std::vector<double> macd_line, macd_sig;
    features::macd(close_, 12, 26, 9, macd_line, macd_sig, macd_hist_);

    features::bollinger(close_, cfg_.bb_period, cfg_.bb_sigma, bb_mid_, bb_up_, bb_lo_);
    atr_ = features::atr(bars_, cfg_.atr_period);
    atr_pct_ = features::atr_pct(bars_, cfg_.atr_period);
    vwap_ = features::rolling_vwap(bars_, cfg_.vwap_period);
    vol_z_ = features::rolling_zscore(vol_, cfg_.vol_window);

    // The oracle passes kind="low"/"high", but its `find_swings` IGNORES kind and
    // always returns strict minima. Reproduced exactly, so the numbers match; the
    // defect is reported and pinned in tests/test_strategy_zoo.cpp.
    swings_low_ = features::find_swings(low_, cfg_.swing_left, cfg_.swing_right);
    swings_high_ = features::find_swings(high_, cfg_.swing_left, cfg_.swing_right);

    // --- advanced regime + strategy ensemble (optional, causal) ------------
    if (cfg_.regime_mode == "advanced") {
        RegimeTracker tracker(RegimeConfig{});
        regimes_ = tracker.run(bars_);
    }
    if (cfg_.ensemble && !regimes_.empty()) {
        series_.reset(new StrategySeries(bars_));
        RouterConfig rc;
        rc.ensemble_veto = cfg_.ensemble_veto;
        rc.agreement_veto = cfg_.ensemble_agreement;
        rc.veto_min_weight = cfg_.ensemble_min_weight;
        ensemble_ = scan(*series_, regimes_, rc, series_->cfg());
    }
}

Prepared::Prepared(const std::vector<Bar>& bars) : Prepared(bars, SniperConfig()) {}

// =============================================================================
// Layers
// =============================================================================
namespace sniper_detail {

double micro(const Prepared& p, size_t i, const EvalContext* ctx,
             const SniperConfig& cfg) {
    (void)cfg;
    double z = p.vol_z()[i];
    if (!fin(z)) z = 0.0;
    const double m_vol = 1.0 / (1.0 + std::exp(-z / 2.0));
    if (ctx != nullptr && ctx->has_obi) {
        const double obi = std::max(-1.0, std::min(1.0, ctx->obi));
        const double m_obi = 0.5 + 0.5 * obi;
        // Either a strongly one-sided book OR an anomalous volume print is
        // evidence of aggressive participation, so take the STRONGER of the two.
        // Blending would let a quiet book dilute a real volume spike (and vice
        // versa), which is exactly backwards.
        return std::max(m_obi, m_vol);
    }
    return m_vol;
}

void technical(const Prepared& p, size_t i, const SniperConfig& cfg, double& score,
               int32_t& votes, std::vector<std::string>& names) {
    names.clear();
    score = 0.0;
    votes = 0;
    if (i < 1) return;

    const double c = p.close()[i];

    // 1. VWAP reclaim: closes back above a VWAP it was below.
    if (fin(p.vwap()[i]) && fin(p.vwap()[i - 1]) && c > p.vwap()[i] &&
        p.close()[i - 1] <= p.vwap()[i - 1])
        names.push_back("vwap-reclaim");

    // 2. RSI bullish divergence (price lower-low, RSI higher-low).
    if (features::bullish_divergence(p.rsi(), p.swings_low(), static_cast<int32_t>(i),
                                     cfg.swing_right, cfg.div_max_gap))
        names.push_back("rsi-divergence");

    // 3. Fibonacci support: close within tol of a retracement level.
    features::Swing sl, sh;
    if (features::last_swing_pair(p.swings_low(), p.swings_high(),
                                  static_cast<int32_t>(i), cfg.swing_right, sl, sh)) {
        const std::vector<double> lvls = features::fib_levels(sl.value, sh.value, cfg.fib_ratios);
        for (double lvl : lvls) {
            if (lvl > 0.0 && std::fabs(c - lvl) / lvl <= cfg.fib_tol) {
                names.push_back("fib-support");
                break;
            }
        }
    }

    // 4. MACD thrust: histogram positive and rising.
    if (fin(p.macd_hist()[i]) && fin(p.macd_hist()[i - 1]) && p.macd_hist()[i] > 0.0 &&
        p.macd_hist()[i] > p.macd_hist()[i - 1])
        names.push_back("macd-thrust");

    // 5. Mean reversion: close below the lower Bollinger band.
    if (fin(p.bb_lo()[i]) && c < p.bb_lo()[i]) names.push_back("bb-oversold");

    // 6. Trend alignment: fast SMA above slow SMA.
    if (fin(p.sma_fast()[i]) && fin(p.sma_slow()[i]) && p.sma_fast()[i] > p.sma_slow()[i])
        names.push_back("trend-align");

    votes = static_cast<int32_t>(names.size());
    // Meeting the mandated minimum confluence earns 0.90 of the layer; each
    // EXTRA agreeing factor adds 0.10 up to a cap of 1.00. Scoring votes/6 would
    // cap the reachable total at ~0.81 and make the 0.85 bar mathematically
    // unreachable — a gate that can never open is a bug, not discipline.
    if (votes >= cfg.min_factors) {
        score = std::min(1.0, 0.90 + 0.10 * (votes - cfg.min_factors));
    } else {
        score = 0.5 * votes / std::max(1, cfg.min_factors);
    }
}

void regime_legacy(const Prepared& p, size_t i, const SniperConfig& cfg,
                   std::string& label, double& score) {
    label = "range";
    score = 0.5;
    if (static_cast<int64_t>(i) < cfg.regime_slope_n || !fin(p.sma_regime()[i])) return;
    const double prev = p.sma_regime()[i - static_cast<size_t>(cfg.regime_slope_n)];
    if (!fin(prev) || prev <= 0.0) return;
    const double slope = (p.sma_regime()[i] - prev) / prev;
    const double c = p.close()[i];
    if (slope > cfg.regime_slope_s && c > p.sma_regime()[i]) {
        label = "trend_up";
        score = 1.0;
        return;
    }
    if (slope < -cfg.regime_slope_s && c < p.sma_regime()[i]) {
        label = "trend_down";
        score = 0.0;
        return;
    }
    const double ap = p.atr_pct()[i];
    if (fin(ap) && ap > cfg.atr_hi) {
        label = "high_vol";
        score = 0.25;
    }
}

void regime(const Prepared& p, size_t i, const SniperConfig& cfg, std::string& label,
            double& score, GatingRegime& gate) {
    // Default path is the advanced multi-axis engine. It is strictly better
    // informed than a single SMA200 slope: a slope rule is a low-volatility
    // filter in disguise on equities (where low vol and uptrends coincide) and
    // stops working the moment you move to FX.
    if (cfg.regime_mode == "legacy" || !p.has_regimes()) {
        regime_legacy(p, i, cfg, label, score);
        // The SMA-slope rule is DETERMINISTIC and computed from bars, so it may
        // gate — but it is minted through its own factory so the distinction
        // between "a rule" and "a learned readout" stays visible here.
        gate = GatingRegime::from_legacy_rule(label);
        return;
    }
    const std::string raw = p.regimes()[i].label;
    label = regime_label_known(raw) ? raw : "range";
    score = regime_score_for_label(label);
    // §46: the ONLY mint in the trading path that can gate. `from_engine()`
    // refuses a RegimeState the engine did not produce, so a learned label
    // smuggled into a hand-built state yields a NON-gating token.
    gate = GatingRegime::from_engine(p.regimes()[i]);
}

double cross(const EvalContext* ctx, const SniperConfig& cfg) {
    (void)cfg;
    if (ctx == nullptr || ctx->cross_active <= 0) return 0.5;
    const double raw = 0.5 + 0.5 * (ctx->cross_confirm - ctx->cross_invalidate) /
                                 static_cast<double>(std::max(1, ctx->cross_active));
    return std::max(0.0, std::min(1.0, raw));
}

bool mean_reversion_confirmed(const Prepared& p, size_t i, const SniperConfig& cfg) {
    // Oversold AND below the lower band — two INDEPENDENT confirmations. One of
    // them alone is a falling knife.
    const double c = p.close()[i];
    const bool below_band = fin(p.bb_lo()[i]) && c < p.bb_lo()[i];
    const bool oversold = fin(p.rsi()[i]) && p.rsi()[i] < cfg.rsi_oversold;
    return below_band && oversold;
}

} // namespace sniper_detail

// =============================================================================
// The engine
// =============================================================================
SniperVerdict evaluate(const Prepared& p, size_t i, const EvalContext* ctx,
                       const SniperConfig& cfg) {
    using namespace sniper_detail;
    SniperVerdict v;
    v.ts = p.ts()[i];

    v.micro = micro(p, i, ctx, cfg);
    technical(p, i, cfg, v.tech, v.votes, v.factors);
    GatingRegime gate;
    regime(p, i, cfg, v.regime, v.regime_score, gate);
    v.regime_gated = gate.authorize();
    v.regime_source = regime_source_name(gate.source());
    if (!v.regime_gated) {
        // §46: a label with no authority influences NOTHING — neither the score
        // nor the veto. Neutralise it to the "range" reading and record that it
        // was refused. In normal operation this branch is unreachable (the rule
        // engine mints its own states); it exists so that the ONE way a regime
        // can reach a gate is the engine's, and every other way is inert.
        v.regime = "range";
        v.regime_score = 0.5;
    }
    v.cross = cross(ctx, cfg);

    // S is the mandate's weighted sum, and NOTHING below may change it.
    v.score = cfg.w_micro * v.micro + cfg.w_tech * v.tech + cfg.w_regime * v.regime_score +
              cfg.w_cross * v.cross;

    // --- hard vetoes (before the score is even considered) -----------------
    // The decision lives in regime_authority.cpp so there is exactly ONE place
    // that can turn a regime label into a block, and it takes a GatingRegime.
    const std::string rv = regime_veto(gate, mean_reversion_confirmed(p, i, cfg));
    if (!rv.empty()) {
        v.veto = true;
        v.veto_reason = rv;
    }

    if (!v.veto && ctx != nullptr && ctx->cross_invalidate >= cfg.invalidate_veto) {
        v.veto = true;
        v.veto_reason = "cross-asset-invalidation";
    }

    const int32_t need = (ctx != nullptr && ctx->event_driven) ? cfg.min_factors_event
                                                              : cfg.min_factors;
    if (!v.veto && v.votes < need) {
        v.veto = true;
        v.veto_reason = "insufficient-confluence";
    }

    // --- strategy ensemble: BLOCK only, never promote ----------------------
    if (p.has_ensemble()) {
        const EnsembleVerdict& ev = p.ensemble()[i];
        v.conviction = ev.conviction;
        v.agreement = ev.agreement;
        RouterConfig rc;
        rc.ensemble_veto = cfg.ensemble_veto;
        rc.agreement_veto = cfg.ensemble_agreement;
        rc.veto_min_weight = cfg.ensemble_min_weight;
        if (!v.veto && should_veto_long(ev, rc)) {
            v.veto = true;
            v.veto_reason = "ensemble-opposed";
        }
    }

    if (p.has_regimes()) {
        v.trend_score = p.regimes()[i].trend_score;
        v.half_life = p.regimes()[i].half_life;
    }
    return v;
}

std::vector<SniperVerdict> scan(const std::vector<Bar>& bars,
                                const std::map<int64_t, EvalContext>& ctx_by_ts,
                                const SniperConfig& cfg) {
    const Prepared p(bars, cfg);
    std::vector<SniperVerdict> out;
    out.reserve(p.n());
    for (size_t i = 0; i < p.n(); ++i) {
        const EvalContext* ctx = nullptr;
        auto it = ctx_by_ts.find(p.ts()[i]);
        if (it != ctx_by_ts.end()) ctx = &it->second;
        out.push_back(evaluate(p, i, ctx, cfg));
    }
    return out;
}

} // namespace trading
} // namespace omniseed
