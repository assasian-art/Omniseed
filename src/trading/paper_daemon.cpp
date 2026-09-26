// =============================================================================
//  OmniSeed — trading/paper_daemon.cpp
//  Paper daemon: deterministic bar replay + append-only journal.
// =============================================================================
#include "omniseed/trading/paper_daemon.h"
#include "omniseed/core/platform.h"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <utility>

namespace omniseed {
namespace trading {

namespace {

// The journal is comma-separated; keep the free-text columns from breaking it.
std::string csv_safe(std::string s) {
    for (char& c : s)
        if (c == ',' || c == '\n' || c == '\r') c = ';';
    return s;
}

// Splits `line` on ',' into at most `max` fields. Returns the field count.
size_t split_fields(const std::string& line, std::string* out, size_t max) {
    size_t n = 0, start = 0;
    for (size_t i = 0; i <= line.size(); ++i) {
        if (i == line.size() || line[i] == ',') {
            if (n < max) out[n] = line.substr(start, i - start);
            ++n;
            start = i + 1;
            if (n >= max) break;
        }
    }
    return n < max ? n : max;
}

} // namespace

// ===========================================================================
// PaperJournal
// ===========================================================================
const char* PaperJournal::header() {
    return "ts,kind,ticker,qty,price,pnl,equity,cash,exposure,reason";
}

PaperJournal::~PaperJournal() { close(); }

void PaperJournal::write_line(const std::string& line) {
    if (!f_) return;
    std::fputs(line.c_str(), f_);
    std::fputc('\n', f_);
    std::fflush(f_);                 // crash-safe: every record hits the file
}

bool PaperJournal::open(std::string& err) {
    if (f_) { err = "journal already open"; return false; }

    // Create the parent directory (portable, best effort) — same idiom as
    // FlashSkillPool::save().
    const auto slash = path_.find_last_of("/\\");
    if (slash != std::string::npos && slash > 0) {
        const std::string dir = path_.substr(0, slash);
#if OMNISEED_PLATFORM_WINDOWS
        const std::string cmd = "if not exist \"" + dir + "\" mkdir \"" + dir + "\"";
#else
        const std::string cmd = "mkdir -p '" + dir + "'";
#endif
        (void)std::system(cmd.c_str());
    }

    bool need_header = true;
    if (std::FILE* probe = platform::open_file_c(path_.c_str(), "rb")) {
        std::fseek(probe, 0, SEEK_END);
        const long sz = std::ftell(probe);
        std::fclose(probe);
        need_header = sz <= 0;
    }

    f_ = platform::open_file_c(path_.c_str(), "ab");
    if (!f_) { err = "cannot open journal: " + path_; return false; }
    if (need_header) write_line(header());
    return true;
}

void PaperJournal::close() {
    if (f_) { std::fclose(f_); f_ = nullptr; }
}

void PaperJournal::equity(int64_t ts, double eq, double cash, double exposure) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%lld,equity,,0,0,0,%.2f,%.2f,%.2f,",
                  static_cast<long long>(ts), eq, cash, exposure);
    write_line(buf);
    ++records_;
    if (ts > last_ts_) last_ts_ = ts;
}

void PaperJournal::signal(int64_t ts, const std::string& ticker,
                          const std::string& action, double strength,
                          const std::string& reason) {
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%lld,signal,%s,%.4f,0,0,0,0,0,%s: %s",
                  static_cast<long long>(ts), csv_safe(ticker).c_str(),
                  strength, action.c_str(), csv_safe(reason).c_str());
    write_line(buf);
    ++records_;
    if (ts > last_ts_) last_ts_ = ts;
}

void PaperJournal::fill(int64_t ts, const std::string& ticker, double qty,
                        double price, double pnl, double eq, double cash,
                        double exposure, const std::string& reason) {
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "%lld,fill,%s,%.6f,%.6f,%.2f,%.2f,%.2f,%.2f,%s",
                  static_cast<long long>(ts), csv_safe(ticker).c_str(), qty,
                  price, pnl, eq, cash, exposure, csv_safe(reason).c_str());
    write_line(buf);
    ++records_;
    if (ts > last_ts_) last_ts_ = ts;
}

void PaperJournal::event(int64_t ts, const std::string& reason) {
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%lld,event,,0,0,0,0,0,0,%s",
                  static_cast<long long>(ts), csv_safe(reason).c_str());
    write_line(buf);
    ++records_;
    if (ts > last_ts_) last_ts_ = ts;
}

void PaperJournal::replay_fill(const std::string& ticker, double qty,
                               double price) {
    if (ticker.empty() || price <= 0.0 || qty == 0.0) return;
    if (qty > 0.0) {
        Position& p = pos_[ticker];
        const double total = p.qty + qty;
        p.avg_price = (p.qty * p.avg_price + qty * price) / total;
        p.qty = total;
        p.side = Side::Long;
        p.ticker = ticker;
    } else {
        auto it = pos_.find(ticker);
        if (it == pos_.end()) return;
        it->second.qty += qty;                 // qty is negative
        if (it->second.qty <= 1e-9) pos_.erase(it);
    }
}

void PaperJournal::scan_existing(std::string& err) {
    err.clear();
    records_ = 0;
    last_ts_ = 0;
    last_equity_ = 0.0;
    last_cash_ = 0.0;
    pos_.clear();
    std::FILE* f = platform::open_file_c(path_.c_str(), "rb");
    if (!f) return;                              // no journal yet: fine

    char line[512];
    bool first = true;
    while (std::fgets(line, sizeof(line), f)) {
        std::string s(line);
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r'))
            s.pop_back();
        if (s.empty()) continue;
        if (first) {
            first = false;
            if (s.rfind("ts,", 0) == 0) continue;    // header row
        }
        ++records_;
        std::string fl[10];
        const size_t nf = split_fields(s, fl, 10);
        const int64_t t = std::atoll(fl[0].c_str());
        if (t > last_ts_) last_ts_ = t;
        // Both equity and fill rows carry equity(field 6) and cash(field 7).
        if (nf >= 8 && !fl[6].empty()) last_equity_ = std::atof(fl[6].c_str());
        if (nf >= 8 && !fl[7].empty()) last_cash_ = std::atof(fl[7].c_str());
        if (nf >= 5 && fl[1] == "fill")
            replay_fill(fl[2], std::atof(fl[3].c_str()),
                        std::atof(fl[4].c_str()));
    }
    std::fclose(f);
}

// ===========================================================================
// PaperDaemon
// ===========================================================================
PaperDaemonResult PaperDaemon::run(const std::vector<Bar>& bars,
                                   const PaperDaemonConfig& cfg,
                                   PaperJournal& journal) {
    PaperDaemonResult res;
    if (!journal.ok()) { res.error = "journal is not open"; return res; }
    if (bars.size() < cfg.warmup_bars + 1) {
        res.error = "not enough bars for the warmup window";
        return res;
    }

    // Resume: skip everything already recorded (bar ts <= journal.last_ts()).
    const int64_t resume_after = cfg.resume ? journal.last_ts() : 0;

    PaperBroker broker(cfg.broker);
    DailyRiskGovernor gov(DailyRiskGovernor::Config{
        cfg.risk.max_daily_loss_pct, 86400});
    const Indicators ind = SignalGenerator::compute(bars, cfg.signals);

    double  equity_peak = cfg.broker.starting_cash;
    int64_t last_halt_day = -1;
    bool    started = false;

    for (size_t i = 0; i < bars.size(); ++i) {
        const Bar& b = bars[i];
        if (b.time <= resume_after) continue;    // already journalled

        broker.mark({{cfg.ticker, b.close}}, b.time);
        const PortfolioState& ps = broker.state();
        equity_peak = std::max(equity_peak, ps.equity);

        ++res.bars_processed;
        if (!started) { journal.event(b.time, "start"); started = true; }
        journal.equity(b.time, ps.equity, ps.cash, ps.gross_exposure);

        // --- daily kill-switch (gates NEW entries only) -------------------
        std::string why;
        const bool day_ok = gov.allow(b.time, ps.equity, why);
        if (gov.halted() && gov.day_index() != last_halt_day) {
            last_halt_day = gov.day_index();
            ++res.halts;
            journal.event(b.time, why.empty() ? "day-halt" : why);
        }

        // --- exits first (never blocked by the kill-switch) ---------------
        const Signal sig = SignalGenerator::evaluate(ind, bars, i, cfg.signals);
        auto pit = ps.positions.find(cfg.ticker);
        const bool has_pos = pit != ps.positions.end() && pit->second.qty > 0.0;
        const bool stopped = has_pos &&
            RiskManager::check_exit(pit->second, b.close, cfg.risk) == Action::Sell;
        const bool signal_exit = has_pos && sig.action == Action::Sell;
        if (has_pos && (stopped || signal_exit)) {
            const double qty = pit->second.qty;
            const double realized_before = broker.state().realized_pnl;
            std::string err;
            if (broker.market_sell(cfg.ticker, qty, b.close, b.time, err)) {
                ++res.exits;
                const double pnl = broker.state().realized_pnl - realized_before;
                journal.fill(b.time, cfg.ticker, -qty, b.close, pnl,
                             broker.state().equity, broker.state().cash,
                             broker.state().gross_exposure,
                             stopped ? "stop-loss" : "signal-exit");
            }
            continue;
        }

        // --- entries on the NEXT bar's open (no look-ahead) ---------------
        if (i + 1 >= bars.size() || i < cfg.warmup_bars) continue;
        if (!day_ok || has_pos || sig.action != Action::Buy) continue;

        std::string why2;
        if (!RiskManager::entries_allowed(ps, equity_peak, cfg.risk, why2))
            continue;

        const RiskManager::Sizing sz =
            RiskManager::size_by_risk(ps.equity, bars[i + 1].open, cfg.risk);
        if (!sz.allowed || sz.qty <= 0.0) continue;

        std::string err;
        if (broker.market_buy(cfg.ticker, sz.qty, bars[i + 1].open,
                              bars[i + 1].time, err)) {
            ++res.entries;
            journal.fill(bars[i + 1].time, cfg.ticker, sz.qty, bars[i + 1].open,
                         0.0, broker.state().equity, broker.state().cash,
                         broker.state().gross_exposure, "entry");
        }
    }

    res.final_equity = broker.state().equity;
    res.metrics = compute_metrics(broker.equity_curve(), cfg.periods_per_year);
    return res;
}

// ===========================================================================
// PaperSession — multi-asset merged-timeline replay (M5)
// ===========================================================================
PaperSessionResult PaperSession::run(const PaperSessionConfig& cfg,
                                     PaperJournal& journal) {
    PaperSessionResult res;
    if (!journal.ok()) { res.error = "journal is not open"; return res; }
    if (cfg.streams.empty()) { res.error = "no streams configured"; return res; }

    // Copy so we can load bars without mutating the caller's config.
    std::vector<PaperStream> streams = cfg.streams;
    for (auto& s : streams) {
        bool reported = false;
        if (s.bars.empty() && !s.csv.empty()) {
            std::string err;
            if (!load_bars_csv(s.csv, s.bars, err)) {
                res.stream_errors.push_back(
                    s.symbol + ": " + (err.empty() ? "cannot load" : err));
                reported = true;
            }
        }
        std::sort(s.bars.begin(), s.bars.end(),
                  [](const Bar& a, const Bar& b) { return a.time < b.time; });
        if (s.bars.empty() && !reported)
            res.stream_errors.push_back(s.symbol + ": no bars");
    }

    const int64_t resume_after = cfg.resume ? journal.last_ts() : 0;

    PaperBroker broker(cfg.broker);
    if (cfg.resume && journal.last_ts() > 0) {
        const double cash = journal.last_cash() > 0.0 ? journal.last_cash()
                                                      : cfg.broker.starting_cash;
        broker.seed(cash, journal.positions());   // rebuild the book
    }
    RiskGovernorSet gov(cfg.governors);

    // Merged chronological timeline of (ts, stream). Ties broken by symbol so
    // the run is deterministic when two assets share a timestamp.
    std::vector<std::pair<int64_t, size_t>> timeline;
    for (size_t si = 0; si < streams.size(); ++si)
        for (size_t bi = 0; bi < streams[si].bars.size(); ++bi)
            timeline.emplace_back(streams[si].bars[bi].time, si);
    std::sort(timeline.begin(), timeline.end(),
              [&](const std::pair<int64_t, size_t>& a,
                  const std::pair<int64_t, size_t>& b) {
                  if (a.first != b.first) return a.first < b.first;
                  return streams[a.second].symbol < streams[b.second].symbol;
              });

    std::vector<Indicators> ind(streams.size());
    for (size_t si = 0; si < streams.size(); ++si)
        ind[si] = SignalGenerator::compute(streams[si].bars, cfg.signals);

    std::vector<size_t> bar_idx(streams.size(), 0);   // next unconsumed bar
    std::vector<char> feed_warned(streams.size(), 0);
    std::vector<char> gate_warned(streams.size(), 0);
    std::map<std::string, double> prices;

    bool started = false, was_halted = false;

    size_t k = 0;
    while (k < timeline.size()) {
        const int64_t ts = timeline[k].first;
        std::vector<size_t> group;
        while (k < timeline.size() && timeline[k].first == ts) {
            group.push_back(timeline[k].second);
            ++k;
        }

        // Consume this timestamp's bars for every stream in the group and
        // refresh the mark. Even resumed (skipped) timestamps must advance the
        // prices so the restored book is marked correctly.
        for (size_t si : group) {
            prices[streams[si].symbol] = streams[si].bars[bar_idx[si]].close;
            ++bar_idx[si];
        }
        if (ts <= resume_after) continue;

        broker.mark(prices, ts);
        const PortfolioState& ps = broker.state();

        ++res.timestamps;
        res.bars_processed += static_cast<int64_t>(group.size());
        if (!started) { journal.event(ts, "start"); started = true; }
        journal.equity(ts, ps.equity, ps.cash, ps.gross_exposure);

        // Composite daily + weekly kill-switch (gates NEW entries only).
        std::string why;
        const bool day_ok = gov.allow(ts, ps.equity, why);
        if (gov.halted() && !was_halted) {
            ++res.halts;
            journal.event(ts, why.empty() ? "halt" : why);
        }
        was_halted = gov.halted();

        for (size_t si : group) {
            const size_t i = bar_idx[si] - 1;      // the bar just consumed
            const Bar& b = streams[si].bars[i];
            const std::string& sym = streams[si].symbol;

            // --- exits first (never blocked by the kill-switch) -----------
            auto pit = ps.positions.find(sym);
            const bool has_pos =
                pit != ps.positions.end() && pit->second.qty > 0.0;
            const Signal sig = SignalGenerator::evaluate(
                ind[si], streams[si].bars, i, cfg.signals);
            // Every actionable signal is journalled, whether or not it becomes
            // a fill — that is what makes the journal a decision log.
            if (cfg.journal_signals && sig.action != Action::Hold) {
                ++res.signals;
                journal.signal(ts, sym,
                               sig.action == Action::Buy ? "buy" : "sell",
                               sig.strength, sig.reason);
            }
            const bool stopped = has_pos &&
                RiskManager::check_exit(pit->second, b.close, cfg.risk) ==
                    Action::Sell;
            const bool signal_exit = has_pos && sig.action == Action::Sell;
            if (has_pos && (stopped || signal_exit)) {
                const double qty = pit->second.qty;
                const double before = broker.state().realized_pnl;
                std::string err;
                if (broker.market_sell(sym, qty, b.close, ts, err)) {
                    ++res.exits;
                    const double pnl = broker.state().realized_pnl - before;
                    journal.fill(ts, sym, -qty, b.close, pnl,
                                 broker.state().equity, broker.state().cash,
                                 broker.state().gross_exposure,
                                 stopped ? "stop-loss" : "signal-exit");
                }
                continue;
            }

            // --- ABSTAIN: a feed that is not OK never opens new risk -------
            if (!streams[si].feed_ok) {
                if (!feed_warned[si]) {
                    feed_warned[si] = 1;
                    ++res.abstains;
                    journal.event(ts, "abstain:" + sym + ":" +
                                      (streams[si].feed_reason.empty()
                                           ? "feed-not-ok"
                                           : streams[si].feed_reason));
                }
                continue;
            }

            // --- entries on the NEXT bar's open (no look-ahead) -----------
            if (i + 1 >= streams[si].bars.size() || i < cfg.warmup_bars)
                continue;
            if (!day_ok || has_pos || sig.action != Action::Buy) continue;

            std::string why2;
            if (!RiskManager::entries_allowed(broker.state(),
                                              broker.equity_peak(), cfg.risk,
                                              why2))
                continue;

            const Bar& nb = streams[si].bars[i + 1];
            const RiskManager::Sizing sz = RiskManager::size_by_risk(
                broker.state().equity, nb.open, streams[si].asset, cfg.risk);
            if (!sz.allowed || sz.qty <= 0.0) {
                if (!gate_warned[si]) {
                    gate_warned[si] = 1;
                    ++res.refused;
                    journal.event(ts, "refused:" + sym + ":" +
                                      (sz.reason.empty() ? "risk-gate"
                                                         : sz.reason));
                }
                continue;
            }

            std::string err;
            if (broker.market_buy(sym, sz.qty, nb.open, nb.time, err)) {
                ++res.entries;
                journal.fill(nb.time, sym, sz.qty, nb.open, 0.0,
                             broker.state().equity, broker.state().cash,
                             broker.state().gross_exposure, "entry");
            }
        }
    }

    res.final_equity = broker.state().equity;
    res.metrics = compute_metrics(broker.equity_curve(), cfg.periods_per_year);
    return res;
}

} // namespace trading
} // namespace omniseed
