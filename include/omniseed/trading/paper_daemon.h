// =============================================================================
//  OmniSeed — trading/paper_daemon.h
//  Trading Edge Lab (T7.1): paper daemon + append-only journal.
//
//    * PaperJournal — a crash-safe, append-only CSV record of everything the
//      daemon did: equity marks, fills, and risk events. Because it is
//      append-only and carries the bar timestamp, a re-run can RESUME: it
//      skips bars already recorded and appends only what is new.
//    * PaperDaemon  — a deterministic bar-replay loop that drives the signal
//      generator, the risk engine (per-trade risk budget + daily kill-switch)
//      and the PaperBroker, journaling every action. No threads, no network,
//      no real orders — paper money only.
//
//  Journal schema (CSV, one record per line):
//    ts,kind,ticker,qty,price,pnl,equity,cash,exposure,reason
//    kind ∈ {start, equity, fill, halt}
//
//  Honest scope: this is a simulation harness. It proves a strategy's
//  behaviour is *recorded* faithfully, not that it makes money.
// =============================================================================
#pragma once

#include "omniseed/trading/simulate.h"
#include "omniseed/trading/trading_engine.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace omniseed {
namespace trading {

// ===========================================================================
// PaperJournal — append-only CSV ledger
// ===========================================================================
class PaperJournal {
public:
    // Header row written once, when the file is created.
    static const char* header();

    explicit PaperJournal(std::string path) : path_(std::move(path)) {}
    ~PaperJournal();

    PaperJournal(const PaperJournal&) = delete;
    PaperJournal& operator=(const PaperJournal&) = delete;

    // Opens for append (creating the parent directory when needed). Writes the
    // header only when the file is new or empty. Safe to call on an existing
    // journal — that is the resume path.
    bool open(std::string& err);
    void close();
    bool ok() const { return f_ != nullptr; }

    void equity(int64_t ts, double equity, double cash, double exposure);
    void fill(int64_t ts, const std::string& ticker, double qty, double price,
              double pnl, double equity, double cash, double exposure,
              const std::string& reason);
    void event(int64_t ts, const std::string& reason);

    // Reads an existing journal file to seed records()/last_ts() for resume.
    // A missing file is not an error (records stay 0).
    void scan_existing(std::string& err);

    int64_t records() const { return records_; }
    // Highest ts already journalled; 0 when the journal is empty.
    int64_t last_ts() const { return last_ts_; }
    const std::string& path() const { return path_; }

private:
    void write_line(const std::string& line);

    std::string path_;
    std::FILE*  f_ = nullptr;
    int64_t     records_ = 0;
    int64_t     last_ts_ = 0;
};

// ===========================================================================
// PaperDaemon
// ===========================================================================
struct PaperDaemonConfig {
    RiskLimits              risk;
    SignalGenerator::Config signals;
    PaperBrokerConfig       broker;
    std::string ticker = "TEST";
    size_t      warmup_bars = 60;      // no entries before this many bars
    bool        resume = true;         // skip bars already in the journal
    double      periods_per_year = 252.0;   // bar cadence for metrics
};

struct PaperDaemonResult {
    int64_t bars_processed = 0;
    int64_t entries = 0;
    int64_t exits = 0;
    int64_t halts = 0;                 // daily kill-switch activations
    double  final_equity = 0.0;
    PerfMetrics metrics;
    std::string error;
};

class PaperDaemon {
public:
    // Replays `bars` (ascending) through signal -> risk -> broker, appending
    // to `journal`. Deterministic: identical inputs produce an identical
    // journal. Exits are never blocked by the daily kill-switch.
    static PaperDaemonResult run(const std::vector<Bar>& bars,
                                 const PaperDaemonConfig& cfg,
                                 PaperJournal& journal);
};

} // namespace trading
} // namespace omniseed
