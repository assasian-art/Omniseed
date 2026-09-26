// =============================================================================
//  OmniSeed — trading/paper_daemon.cpp
//  Paper daemon: deterministic bar replay + append-only journal.
// =============================================================================
#include "omniseed/trading/paper_daemon.h"
#include "omniseed/core/platform.h"

#include <algorithm>
#include <cstdlib>
#include <string>

namespace omniseed {
namespace trading {

namespace {

// The journal is comma-separated; keep the free-text columns from breaking it.
std::string csv_safe(std::string s) {
    for (char& c : s)
        if (c == ',' || c == '\n' || c == '\r') c = ';';
    return s;
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

void PaperJournal::scan_existing(std::string& err) {
    err.clear();
    records_ = 0;
    last_ts_ = 0;
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
        const size_t c = s.find(',');
        if (c != std::string::npos) {
            const int64_t t = std::atoll(s.substr(0, c).c_str());
            if (t > last_ts_) last_ts_ = t;
        }
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

} // namespace trading
} // namespace omniseed
