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
//    kind ∈ {start, equity, signal, fill, halt, event}
//  For a `signal` row the qty column carries the signal strength (0..1), and
//  the reason reads "<buy|sell>: <rule trace>".
//
//  Honest scope: this is a simulation harness. It proves a strategy's
//  behaviour is *recorded* faithfully, not that it makes money.
// =============================================================================
#pragma once

#include "omniseed/trading/risk_gate.h"
#include "omniseed/trading/simulate.h"
#include "omniseed/trading/trading_engine.h"

#include <cstdint>
#include <cstdio>
#include <map>
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
    // A non-Hold signal decision. The qty column carries the strength (0..1);
    // `reason` is the rule trace, prefixed with the action in the CSV.
    void signal(int64_t ts, const std::string& ticker, const std::string& action,
                double strength, const std::string& reason);
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

    // --- resume state (M5) ------------------------------------------------
    // The journal is the append-only source of truth, so a restarted daemon
    // rebuilds its portfolio from it: the last marked equity/cash, and the
    // open positions implied by the recorded fills (qty>0 = buy, qty<0 = sell;
    // a position that nets to <= 0 is closed and dropped).
    double last_equity() const { return last_equity_; }
    double last_cash() const { return last_cash_; }
    const std::map<std::string, Position>& positions() const { return pos_; }

private:
    void write_line(const std::string& line);
    void replay_fill(const std::string& ticker, double qty, double price);

    std::string path_;
    std::FILE*  f_ = nullptr;
    int64_t     records_ = 0;
    int64_t     last_ts_ = 0;
    double      last_equity_ = 0.0;
    double      last_cash_ = 0.0;
    std::map<std::string, Position> pos_;
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

// ===========================================================================
// PaperSession — multi-asset paper trading over a merged timeline (M5)
// ===========================================================================
// One cash account, many streams. Bars from every stream are merged into a
// single chronological timeline; at each timestamp the whole book is marked,
// the composite daily+weekly kill-switch is consulted once, and then each
// stream that has a bar at that timestamp may exit or enter. Entries are
// sized per asset class (meme stays research-only / <=0.25%), and a stream
// whose feed is not OK is skipped for entries (ABSTAIN, journalled).
struct PaperStream {
    std::string symbol;                 // canonical ticker (AAPL / BTCUSDT / ...)
    AssetClass  asset = AssetClass::Unknown;
    std::string csv;                    // provenance CSV (T6 schema); optional
    std::string timeframe = "1d";
    std::vector<Bar> bars;              // pre-loaded bars (else loaded from csv)
    bool        feed_ok = true;         // perception verdict; false = ABSTAIN
    std::string feed_reason;            // e.g. "stale-feed" (for the journal)
};

struct PaperSessionConfig {
    RiskLimits              risk;
    SignalGenerator::Config signals;
    PaperBrokerConfig       broker;
    RiskGovernorSet::Config governors;   // daily AND weekly kill-switches
    size_t      warmup_bars = 60;        // no entries before this many bars
    bool        resume = true;           // rebuild the book from the journal
    bool        journal_signals = true;  // journal every non-Hold signal
    double      periods_per_year = 252.0;
    std::vector<PaperStream> streams;
};

struct PaperSessionResult {
    int64_t bars_processed = 0;          // stream-bars seen after the resume mark
    int64_t timestamps = 0;              // merged timestamps processed
    int64_t signals = 0;                 // non-Hold signals journalled
    int64_t entries = 0;
    int64_t exits = 0;
    int64_t halts = 0;                   // kill-switch activation transitions
    int64_t abstains = 0;                // entry refused: feed not OK
    int64_t refused = 0;                 // entry refused: risk/asset gate
    double  final_equity = 0.0;
    PerfMetrics metrics;
    std::string error;
    std::vector<std::string> stream_errors;
};

class PaperSession {
public:
    // Deterministic: identical streams + journal state produce an identical
    // journal tail. Exits are never blocked by the kill-switch.
    static PaperSessionResult run(const PaperSessionConfig& cfg,
                                  PaperJournal& journal);
};

} // namespace trading
} // namespace omniseed
