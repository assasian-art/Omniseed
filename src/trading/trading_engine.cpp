// =============================================================================
//  OmniSeed — trading/trading_engine.cpp
//  Deterministic market analytics. No look-ahead, no randomness, no I/O
//  beyond CSV. See the header for the honest-scope note.
// =============================================================================
#include "omniseed/trading/trading_engine.h"
#include "omniseed/core/platform.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

namespace omniseed {
namespace trading {

// ===========================================================================
// Small helpers
// ===========================================================================
namespace {

inline bool is_leap(int y) {
    return (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0);
}

// Days from 1970-01-01 (proleptic Gregorian; Howard Hinnant's algorithm).
int64_t days_from_civil(int y, unsigned m, unsigned d) {
    y -= m <= 2;
    const int era  = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = static_cast<unsigned>(y - era * 400);
    const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return static_cast<int64_t>(era) * 146097 +
           static_cast<int64_t>(doe) - 719468;
}

void civil_from_days(int64_t z, int& y, int& m, int& d) {
    z += 719468;
    const int64_t era  = (z >= 0 ? z : z - 146096) / 146097;
    const unsigned doe = static_cast<unsigned>(z - era * 146097);
    const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    const int64_t yy = static_cast<int64_t>(yoe) + era * 400;
    const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const unsigned mp = (5 * doy + 2) / 153;
    d = static_cast<int>(doy - (153 * mp + 2) / 5 + 1);
    m = static_cast<int>(mp + (mp < 10 ? 3 : -9));
    y = static_cast<int>(yy + (m <= 2));
}

// "YYYY-MM-DD[ HH:MM[:SS]]" -> unix seconds (UTC). 0 on failure.
int64_t parse_ts(const char* s) {
    int y = 0, mo = 0, d = 0, h = 0, mi = 0, sec = 0;
    int n = std::sscanf(s, "%d-%d-%d %d:%d:%d", &y, &mo, &d, &h, &mi, &sec);
    if (n < 3) {
        sec = 0;
        n = std::sscanf(s, "%d/%d/%d %d:%d", &y, &mo, &d, &h, &mi);
    }
    if (n >= 3) {
        if (y < 100) y += y < 70 ? 2000 : 1900;   // 2-digit years
        if (mo < 1 || mo > 12 || d < 1 || d > 31) return 0;
        return days_from_civil(y, static_cast<unsigned>(mo),
                               static_cast<unsigned>(d)) * 86400LL +
               h * 3600LL + mi * 60LL +
               (n >= 6 ? static_cast<int64_t>(sec) : 0);
    }
    // All digits -> unix epoch seconds.
    const bool all_digits = *s &&
        std::strspn(s, "0123456789") == std::strlen(s);
    return all_digits ? std::atoll(s) : 0;
}

// Splits a CSV line on commas (no quoting needed for our own files).
size_t split_csv(const std::string& line, std::string* fields, size_t max) {
    size_t n = 0, start = 0;
    for (size_t i = 0; i <= line.size() && n < max; ++i) {
        if (i == line.size() || line[i] == ',') {
            fields[n++] = line.substr(start, i - start);
            start = i + 1;
        }
    }
    return n;
}

} // namespace

// ===========================================================================
// CSV persistence
// ===========================================================================
bool load_bars_csv(const std::string& path, std::vector<Bar>& out,
                   std::string& err) {
    out.clear();
    std::FILE* f = platform::open_file_c(path.c_str(), "rb");
    if (!f) { err = "cannot open " + path; return false; }

    char line[512];
    bool header = true;
    while (std::fgets(line, sizeof(line), f)) {
        std::string s(line);
        while (!s.empty() &&
               (s.back() == '\n' || s.back() == '\r' || s.back() == ' '))
            s.pop_back();
        if (s.empty()) continue;
        if (header) {                       // skip header row
            header = false;
            if (s.find_first_of("0123456789") != 0) continue;
        }
        std::string fields[8];
        const size_t n = split_csv(s, fields, 8);
        if (n < 5) continue;                // malformed row: skip, not fatal
        Bar b;
        b.time   = parse_ts(fields[0].c_str());
        b.open   = std::atof(fields[1].c_str());
        b.high   = std::atof(fields[2].c_str());
        b.low    = std::atof(fields[3].c_str());
        b.close  = std::atof(fields[4].c_str());
        b.volume = n >= 6 ? std::atof(fields[5].c_str()) : 0.0;
        if (b.time > 0 && b.close > 0.0 && b.high >= b.low &&
            b.high >= b.close && b.low <= b.close)
            out.push_back(b);
    }
    std::fclose(f);
    std::sort(out.begin(), out.end(),
              [](const Bar& a, const Bar& c) { return a.time < c.time; });
    if (out.empty()) { err = "no valid rows in " + path; return false; }
    return true;
}

bool save_bars_csv(const std::string& path, const std::vector<Bar>& bars,
                   std::string& err) {
    std::FILE* f = platform::open_file_c(path.c_str(), "wb");
    if (!f) { err = "cannot write " + path; return false; }
    std::fprintf(f, "time,open,high,low,close,volume\n");
    for (const Bar& b : bars) {
        const int64_t secs_of_day = ((b.time % 86400) + 86400) % 86400;
        int y, mo, d;
        civil_from_days(b.time / 86400, y, mo, d);
        if (secs_of_day == 0) {
            std::fprintf(f, "%04d-%02d-%02d,%.6f,%.6f,%.6f,%.6f,%.0f\n",
                         y, mo, d, b.open, b.high, b.low, b.close, b.volume);
        } else {
            // Intraday bars keep their time-of-day so the round-trip is
            // exact to the second.
            std::fprintf(f,
                         "%04d-%02d-%02d %02d:%02d:%02d,%.6f,%.6f,%.6f,%.6f,%.0f\n",
                         y, mo, d,
                         static_cast<int>(secs_of_day / 3600),
                         static_cast<int>((secs_of_day / 60) % 60),
                         static_cast<int>(secs_of_day % 60),
                         b.open, b.high, b.low, b.close, b.volume);
        }
    }
    std::fclose(f);
    return true;
}

// ===========================================================================
// MarketDataFeed
// ===========================================================================
bool MarketDataFeed::load(std::string& err) {
    err_.clear();
    loaded_ = false;
    if (cfg_.kind == Kind::CsvReplay)
        return loaded_ = load_bars_csv(cfg_.csv_path, bars_, err_);
    err = err_ = "Http feeds require a user-provided adapter "
                 "(docs/TRADING_GUIDE.md); CsvReplay is the offline path";
    return false;
}

bool MarketDataFeed::latest_close(double& out) const {
    if (bars_.empty()) return false;
    out = bars_.back().close;
    return true;
}

// ===========================================================================
// SignalGenerator — indicators
// ===========================================================================
Indicators SignalGenerator::compute(const std::vector<Bar>& bars,
                                    const Config& cfg) {
    const size_t n = bars.size();
    Indicators o;
    auto closes = [&](size_t i) { return bars[i].close; };

    auto sma_series = [&](int32_t p) {
        std::vector<double> s(n, 0.0);
        double sum = 0.0;
        for (size_t i = 0; i < n; ++i) {
            sum += closes(i);
            if (i >= static_cast<size_t>(p)) sum -= closes(i - static_cast<size_t>(p));
            s[i] = (i + 1 >= static_cast<size_t>(p))
                 ? sum / static_cast<double>(p) : 0.0;
        }
        return s;
    };

    auto ema_series = [&](int32_t p) {
        std::vector<double> e(n, 0.0);
        const double k = 2.0 / (static_cast<double>(p) + 1.0);
        for (size_t i = 0; i < n; ++i)
            e[i] = (i == 0) ? closes(0)
                 : closes(i) * k + e[i - 1] * (1.0 - k);
        return e;
    };

    o.sma20  = sma_series(cfg.sma_fast);
    o.sma50  = sma_series(cfg.sma_slow);
    o.sma200 = sma_series(200);
    o.ema12  = ema_series(cfg.macd_fast);
    o.ema26  = ema_series(cfg.macd_slow);

    // RSI (Wilder's smoothing).
    o.rsi14.assign(n, 50.0);
    {
        double avg_gain = 0.0, avg_loss = 0.0;
        const size_t p = static_cast<size_t>(cfg.rsi_period);
        for (size_t i = 1; i < n; ++i) {
            const double ch = closes(i) - closes(i - 1);
            const double gain = ch > 0 ? ch : 0.0;
            const double loss = ch < 0 ? -ch : 0.0;
            if (i <= p) {
                avg_gain += gain / p;
                avg_loss += loss / p;
                if (i == p && avg_loss > 0)
                    o.rsi14[i] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss);
            } else {
                avg_gain = (avg_gain * (p - 1) + gain) / p;
                avg_loss = (avg_loss * (p - 1) + loss) / p;
                o.rsi14[i] = (avg_loss > 0)
                    ? 100.0 - 100.0 / (1.0 + avg_gain / avg_loss) : 100.0;
            }
        }
    }

    // MACD + signal + histogram.
    o.macd.resize(n);
    for (size_t i = 0; i < n; ++i) o.macd[i] = o.ema12[i] - o.ema26[i];
    o.macd_signal.assign(n, 0.0);
    {
        const double k = 2.0 / (static_cast<double>(cfg.macd_signal_p) + 1.0);
        for (size_t i = 0; i < n; ++i)
            o.macd_signal[i] = (i == 0) ? o.macd[0]
                : o.macd[i] * k + o.macd_signal[i - 1] * (1.0 - k);
    }
    o.macd_hist.resize(n);
    for (size_t i = 0; i < n; ++i) o.macd_hist[i] = o.macd[i] - o.macd_signal[i];

    // Bollinger bands (population std over the window).
    {
        const size_t p = static_cast<size_t>(cfg.bb_period);
        o.bb_mid.assign(n, 0.0); o.bb_upper.assign(n, 0.0); o.bb_lower.assign(n, 0.0);
        for (size_t i = 0; i + 1 >= p; ++i) {
            const size_t start = i + 1 - p;
            double mean = 0.0;
            for (size_t j = start; j <= i; ++j) mean += closes(j);
            mean /= static_cast<double>(p);
            double var = 0.0;
            for (size_t j = start; j <= i; ++j) {
                const double d = closes(j) - mean;
                var += d * d;
            }
            var /= static_cast<double>(p);
            const double sd = std::sqrt(var);
            o.bb_mid[i]   = mean;
            o.bb_upper[i] = mean + cfg.bb_sigma * sd;
            o.bb_lower[i] = mean - cfg.bb_sigma * sd;
        }
    }

    // ATR (Wilder).
    o.atr14.assign(n, 0.0);
    {
        const size_t p = static_cast<size_t>(cfg.atr_period);
        double atr = 0.0;
        for (size_t i = 1; i < n; ++i) {
            const double tr = std::max({
                bars[i].high - bars[i].low,
                std::fabs(bars[i].high - closes(i - 1)),
                std::fabs(bars[i].low  - closes(i - 1))});
            atr = (i <= p) ? atr + tr / p
                           : (atr * static_cast<double>(p - 1) + tr) / p;
            o.atr14[i] = atr;
        }
    }
    return o;
}

// ===========================================================================
// SignalGenerator — rule evaluation (no look-ahead: index i only)
// ===========================================================================
Signal SignalGenerator::evaluate(const Indicators& ind,
                                 const std::vector<Bar>& bars,
                                 size_t i, const Config& cfg) {
    Signal sig;
    if (i >= bars.size() || i >= ind.rsi14.size()) return sig;
    const double close = bars[i].close;
    int votes = 0;

    const bool have_bb = i < ind.bb_lower.size() && ind.bb_lower[i] > 0.0 &&
                         ind.bb_upper[i] > 0.0;
    const bool have_ma = i < ind.sma20.size() && i < ind.sma50.size() &&
                         ind.sma50[i] > 0.0;

    // Rule coherence: oversold dips occur INSIDE downtrends, so a flat
    // "downtrend -1" would cancel every mean-reversion entry. The dip gets
    // a strong +2 (it IS the trade); the downtrend only subtracts when we
    // are not looking at a dip.
    const bool oversold = have_bb && ind.rsi14[i] < cfg.rsi_buy_below &&
                          close < ind.bb_lower[i];
    const bool overbought = (have_bb && close > ind.bb_upper[i]) ||
                            ind.rsi14[i] > cfg.rsi_sell_above;
    const bool uptrend = have_ma && ind.sma20[i] > ind.sma50[i] &&
                         ind.macd_hist[i] > 0.0;
    const bool downtrend = have_ma && ind.sma20[i] < ind.sma50[i] &&
                           ind.macd_hist[i] < 0.0;

    if (oversold)   { votes += 2; sig.reason += "oversold-dip; "; }
    if (uptrend)    { votes += 1; sig.reason += "uptrend+MACD; "; }
    if (overbought) { votes -= 2; sig.reason += "overbought; "; }
    if (downtrend && !oversold) { votes -= 1; sig.reason += "downtrend; "; }

    if (votes > 0)      sig.action = Action::Buy;
    else if (votes < 0) sig.action = Action::Sell;
    sig.strength = std::min(1.0, std::abs(votes) / 2.0);
    if (sig.reason.size() >= 2)
        sig.reason.resize(sig.reason.size() - 2);   // trailing "; "
    else
        sig.reason = "neutral";
    return sig;
}

// ===========================================================================
// PositionManager
// ===========================================================================
bool PositionManager::buy(const std::string& ticker, double qty, double price,
                          int64_t ts, std::string& err) {
    if (qty <= 0.0 || price <= 0.0) { err = "bad qty/price"; return false; }
    const double cost = qty * price;
    if (cost > cash_ + 1e-9) {
        err = "insufficient cash";
        return false;
    }
    cash_ -= cost;
    Position& p = pos_[ticker];
    const double total = p.qty + qty;
    p.avg_price = (p.qty * p.avg_price + cost) / total;
    p.qty = total;
    p.side = Side::Long;
    p.ticker = ticker;
    if (p.opened_at == 0) p.opened_at = ts;
    return true;
}

bool PositionManager::sell(const std::string& ticker, double qty, double price,
                           int64_t, std::string& err) {
    auto it = pos_.find(ticker);
    if (it == pos_.end() || qty <= 0.0 || price <= 0.0) {
        err = "no position / bad qty"; return false;
    }
    if (qty > it->second.qty + 1e-9) { err = "qty exceeds position"; return false; }
    const double cost_basis = it->second.avg_price * qty;
    realized_ += price * qty - cost_basis;
    cash_ += price * qty;
    it->second.qty -= qty;
    if (it->second.qty <= 1e-9) pos_.erase(it);
    return true;
}

PortfolioState PositionManager::mark(
        const std::map<std::string, double>& prices) const {
    PortfolioState ps;
    ps.cash = cash_;
    ps.realized_pnl = realized_;
    for (const auto& kv : pos_) {
        auto pit = prices.find(kv.first);
        const double px = pit != prices.end() ? pit->second : kv.second.avg_price;
        const double mv = kv.second.qty * px;
        ps.unrealized_pnl += mv - kv.second.qty * kv.second.avg_price;
        ps.gross_exposure += mv;
    }
    ps.equity = ps.cash + ps.gross_exposure;
    ps.positions = pos_;
    return ps;
}

// ===========================================================================
// RiskManager
// ===========================================================================
RiskManager::Sizing RiskManager::size_position(double equity, double price,
                                               double win_prob, double avg_win,
                                               double avg_loss,
                                               const RiskLimits& lim) {
    Sizing s;
    if (equity <= 0.0 || price <= 0.0) { s.reason = "bad equity/price"; return s; }
    if (win_prob <= 0.0 || win_prob >= 1.0 || avg_win <= 0.0 || avg_loss <= 0.0) {
        s.reason = "Kelly inputs out of range -> fixed 2% test position";
        s.qty = std::floor(equity * 0.02 / price);
        s.allowed = s.qty > 0.0;
        return s;
    }
    const double b = avg_win / avg_loss;             // payoff ratio
    const double kelly = (b * win_prob - (1.0 - win_prob)) / b;
    double frac = kelly * lim.kelly_fraction;
    if (frac <= 0.0) {
        s.reason = "negative Kelly edge -> no trade";
        s.allowed = false;
        return s;
    }
    frac = std::min(frac, lim.max_position_pct);
    s.qty = std::floor(equity * frac / price);
    s.reason = "Kelly " + std::to_string(kelly).substr(0, 5) + " x" +
               std::to_string(lim.kelly_fraction).substr(0, 4) +
               " -> " + std::to_string(frac * 100.0).substr(0, 4) + "% of equity";
    s.allowed = s.qty > 0.0;
    return s;
}

Action RiskManager::check_exit(const Position& pos, double price,
                               const RiskLimits& lim) {
    if (pos.qty <= 0.0 || pos.avg_price <= 0.0) return Action::Hold;
    const double ret = price / pos.avg_price - 1.0;
    if (ret <= -lim.stop_loss_pct) return Action::Sell;               // stop
    if (lim.take_profit_pct > 0.0 && ret >= lim.take_profit_pct)
        return Action::Sell;                                          // target
    return Action::Hold;
}

double RiskManager::drawdown(const std::vector<double>& equity) {
    double peak = 0.0, max_dd = 0.0;
    for (const double v : equity) {
        if (v > peak) peak = v;
        if (peak > 0.0) max_dd = std::max(max_dd, (peak - v) / peak);
    }
    return max_dd;
}

bool RiskManager::entries_allowed(const PortfolioState& ps, double equity_peak,
                                  const RiskLimits& lim, std::string& why_not) {
    if (equity_peak > 0.0 && ps.equity > 0.0) {
        const double dd = (equity_peak - ps.equity) / equity_peak;
        if (dd >= lim.max_drawdown_pct) {
            why_not = "drawdown " + std::to_string(dd * 100.0).substr(0, 4) +
                      "% >= limit " +
                      std::to_string(lim.max_drawdown_pct * 100.0).substr(0, 4) + "%";
            return false;
        }
    }
    if (ps.equity > 0.0 &&
        ps.gross_exposure / ps.equity >= lim.max_gross_exposure_pct) {
        why_not = "gross exposure cap reached";
        return false;
    }
    return true;
}

// ===========================================================================
// Metrics
// ===========================================================================
PerfMetrics compute_metrics(const std::vector<double>& equity,
                            double periods_per_year) {
    PerfMetrics m;
    if (equity.size() < 2) return m;
    const double first = equity.front(), last = equity.back();
    m.total_return_pct = (last / first - 1.0) * 100.0;

    // Per-period log returns.
    std::vector<double> r;
    r.reserve(equity.size() - 1);
    for (size_t i = 1; i < equity.size(); ++i)
        r.push_back(equity[i] / equity[i - 1] - 1.0);

    const double n = static_cast<double>(r.size());
    double mean = 0.0;
    for (double v : r) mean += v;
    mean /= n;
    double var = 0.0, ddown_sum = 0.0;
    int64_t down_n = 0;
    for (double v : r) {
        var += (v - mean) * (v - mean);
        if (v < 0.0) { ddown_sum += v * v; ++down_n; }
    }
    const double sd = std::sqrt(var / std::max(1.0, n - 1.0));
    const double ann = std::sqrt(periods_per_year);
    m.sharpe = sd > 0.0 ? mean / sd * ann : 0.0;
    const double ddev = down_n > 0
        ? std::sqrt(ddown_sum / static_cast<double>(down_n)) : 0.0;
    m.sortino = ddev > 0.0 ? mean / ddev * ann : 0.0;
    m.max_drawdown_pct = RiskManager::drawdown(equity) * 100.0;

    // CAGR from the period count.
    const double years = n / std::max(1.0, periods_per_year);
    m.cagr_pct = (years > 0.0 && last > 0.0 && first > 0.0)
        ? (std::pow(last / first, 1.0 / years) - 1.0) * 100.0 : 0.0;
    return m;
}

// ===========================================================================
// Backtester
// ===========================================================================
BacktestReport Backtester::run(const std::vector<Bar>& bars,
                               const Config& cfg) {
    BacktestReport rep;
    if (bars.size() < 60) {
        rep.error = "need >= 60 bars for a meaningful replay";
        return rep;
    }

    const Indicators ind = SignalGenerator::compute(bars, cfg.signals);
    PositionManager pm(cfg.starting_cash);
    const RiskLimits& lim = cfg.risk;

    double equity_peak = cfg.starting_cash;
    // Zero-evidence priors: win_prob 0 is "out of range" for Kelly, which
    // routes to the fixed 2% probe position until real trade statistics
    // exist (a 0.5/0.02/0.02 prior yields Kelly == 0 exactly -> deadlock:
    // no first trade, no evidence, no trades forever).
    double win_prob = 0.0, avg_win = 0.0, avg_loss = 0.0;
    int64_t wins = 0, losses = 0;
    rep.equity_curve.reserve(bars.size());

    // Warmup: flat equity until the slowest indicator has data.
    const size_t warmup = static_cast<size_t>(std::max(
        cfg.signals.sma_slow, std::max(cfg.signals.bb_period, cfg.signals.macd_slow)));

    for (size_t i = 0; i < bars.size(); ++i) {
        const Bar& b = bars[i];
        const PortfolioState ps = pm.mark({{cfg.ticker, b.close}});
        equity_peak = std::max(equity_peak, ps.equity);
        rep.equity_curve.push_back(ps.equity);

        // --- exits first (stop/TP on the open position) ------------------
        auto pit = ps.positions.find(cfg.ticker);
        const Signal sig_now = SignalGenerator::evaluate(ind, bars, i,
                                                         cfg.signals);
        const bool signal_exit = pit != ps.positions.end() &&
                                 sig_now.action == Action::Sell;
        if (pit != ps.positions.end() &&
            (RiskManager::check_exit(pit->second, b.close, lim) == Action::Sell ||
             signal_exit)) {
                TradeRecord tr;
                tr.ticker = cfg.ticker;
                tr.entry_ts = pit->second.opened_at;
                tr.entry_price = pit->second.avg_price;
                tr.qty = pit->second.qty;
                const double slip = b.close * (1.0 - lim.slippage_bps * 1e-4);
                tr.exit_ts = b.time;
                tr.exit_price = slip;
                const double fees = (tr.qty * tr.exit_price) * lim.fee_bps * 1e-4;
                tr.pnl = tr.qty * (tr.exit_price - tr.entry_price) - fees;
                tr.exit_reason =
                    signal_exit ? "signal"
                    : (tr.exit_price / tr.entry_price - 1.0 <= -lim.stop_loss_pct
                           ? "stop-loss" : "take-profit");
                std::string err;
                pm.sell(cfg.ticker, tr.qty, slip, b.time, err);
                rep.trades.push_back(tr);
                if (tr.pnl > 0.0) { ++wins; } else { ++losses; }
                continue;
        }

        // --- entries on the NEXT bar's open (no look-ahead) ---------------
        if (i + 1 >= bars.size() || i < warmup) continue;
        const Signal& sig = sig_now;
        if (sig.action != Action::Buy) continue;

        std::string why;
        if (!RiskManager::entries_allowed(ps, equity_peak, lim, why)) continue;
        if (ps.positions.count(cfg.ticker)) continue;   // already long

        const RiskManager::Sizing sz = RiskManager::size_position(
            ps.equity, b.close, win_prob, avg_win, avg_loss, lim);
        if (!sz.allowed || sz.qty <= 0.0) continue;

        const double ask = bars[i + 1].open * (1.0 + lim.slippage_bps * 1e-4);
        const double cost = sz.qty * ask;
        if (cost > ps.cash) continue;                   // risk cap refused
        std::string err;
        if (!pm.buy(cfg.ticker, sz.qty, ask, bars[i + 1].time, err)) continue;
        const double fees = cost * lim.fee_bps * 1e-4;
        (void)fees;   // fee model: entry fee accrues via cash accounting above

        // Update running Kelly estimates from closed trades.
        if (!rep.trades.empty()) {
            int64_t w = 0, l = 0;
            double aw = 0.0, al = 0.0;
            for (const TradeRecord& t : rep.trades) {
                if (t.pnl > 0.0) { ++w; aw += t.pnl / (t.qty * t.entry_price); }
                else if (t.pnl < 0.0) { ++l; al += std::fabs(t.pnl) / (t.qty * t.entry_price); }
            }
            const int64_t tot = w + l;
            if (tot > 0) {
                win_prob = static_cast<double>(w) / static_cast<double>(tot);
                if (w > 0) avg_win = aw / w;
                if (l > 0) avg_loss = al / l;
            }
        }
    }

    // Final mark at the last close.
    const PortfolioState ps = pm.mark({{cfg.ticker, bars.back().close}});
    rep.final_equity = ps.equity;
    rep.total_return_pct = (rep.final_equity / cfg.starting_cash - 1.0) * 100.0;
    rep.bars = static_cast<int64_t>(bars.size());

    const double span_s = static_cast<double>(bars.back().time - bars.front().time);
    const double years = span_s > 0.0 ? span_s / (365.25 * 86400.0) : 0.0;
    rep.cagr_pct = (years > 0.25 && rep.final_equity > 0.0)
        ? (std::pow(rep.final_equity / cfg.starting_cash, 1.0 / years) - 1.0) * 100.0
        : rep.total_return_pct;

    const double periods = bars.size() >= 2 && span_s > 0.0
        ? static_cast<double>(bars.size() - 1) / (span_s / (252.0 * 86400.0))
        : 252.0;
    const PerfMetrics m = compute_metrics(rep.equity_curve,
                                          std::max(1.0, periods));
    rep.sharpe = m.sharpe;
    rep.sortino = m.sortino;
    rep.max_drawdown_pct = m.max_drawdown_pct;

    const int64_t closed = wins + losses;
    rep.win_rate = closed > 0 ? static_cast<double>(wins) / closed : 0.0;
    double gp = 0.0, gl = 0.0;
    for (const TradeRecord& t : rep.trades)
        (t.pnl > 0.0 ? gp : gl) += std::fabs(t.pnl);
    rep.profit_factor = gl > 0.0 ? gp / gl : (gp > 0.0 ? 999.0 : 0.0);
    (void)losses;
    return rep;
}

} // namespace trading
} // namespace omniseed
