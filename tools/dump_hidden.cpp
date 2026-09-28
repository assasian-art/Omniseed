// =============================================================================
//  OmniSeed — tools/dump_hidden.cpp
//
//  Collects (hidden_state, label) pairs by running the REAL backbone over the
//  real training inputs, so tools/train_heads.py can fit the head projections
//  OFFLINE. Nothing here trains anything; this is the data-collection half of
//  the port pattern: C++ produces the oracle facts, Python does the fit.
//
//  WHY A DUMPER AND NOT A PYTHON MODEL LOADER. There is no RWKV runtime in
//  Python in this tree. Re-implementing the recurrence in numpy to collect h
//  would create a second, drifting definition of "the model's opinion" — the
//  exact failure the regime/strategy parity dumps exist to prevent. So the
//  hidden states come from the one implementation that also serves at runtime.
//
//  THE HONESTY CONTRACT. If the backbone is absent, or a text encodes to zero
//  tokens, this tool writes NOTHING and exits non-zero. It never synthesises a
//  hidden state. A fabricated h[E] would fit a head that looks trained and is
//  meaningless, which is worse than a missing file — a missing file is a
//  visible failure, a fake fit is an invisible one.
//
//  OUTPUT (per invocation, into <out_dir>):
//    hidden.f32   N x E little-endian float32, row-major
//    labels.tsv   header + one row per example; column 0 is a provenance id,
//                 columns 1.. are the label for each named label set
//    meta.json    counts, model path, geometry, and the exact parameters used
//
//  USAGE
//    dump_hidden language <model.gguf> <examples.tsv> <out_dir>
//    dump_hidden market   <model.gguf> <bars.csv>     <out_dir>
//                         [--window N] [--stride N] [--limit N] [--stream]
//    dump_hidden language ... --dry-run     # tokenise + report, forward nothing
//
//  The language TSV is  text <TAB> <set.name> <TAB> ...  with the set names in
//  the header line, so the tool never hard-codes a label vocabulary.
// =============================================================================
#include "omniseed/core/gguf_loader.h"
#include "omniseed/core/platform.h"
#include "omniseed/core/rwkv.h"
#include "omniseed/core/tokenizer.h"
#include "omniseed/trading/regime_engine.h"
#include "omniseed/trading/trading_engine.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#  include <direct.h>
#else
#  include <sys/stat.h>
#endif

using namespace omniseed;

namespace {

// ---------------------------------------------------------------------------
// Tiny helpers. Everything is plain stdio + vectors; this tool must build and
// run anywhere the core library does.
// ---------------------------------------------------------------------------
void die(const std::string& msg) {
    std::fprintf(stderr, "dump_hidden: %s\n", msg.c_str());
    std::exit(2);
}

// Best-effort directory creation. Failure is not fatal: an existing directory
// makes mkdir fail with EEXIST, which is the common case, and a genuinely
// unwritable path is reported by the fopen() in Collector::write().
void ensure_dir(const std::string& path) {
#ifdef _WIN32
    _mkdir(path.c_str());
#else
    mkdir(path.c_str(), 0755);
#endif
}

std::vector<std::string> split_tab(const std::string& line) {
    std::vector<std::string> out;
    size_t start = 0;
    for (;;) {
        const size_t tab = line.find('\t', start);
        if (tab == std::string::npos) {
            out.push_back(line.substr(start));
            break;
        }
        out.push_back(line.substr(start, tab - start));
        start = tab + 1;
    }
    return out;
}

// Strips a trailing \r so a CRLF file does not put a stray carriage return
// into the last column (which would make every label comparison fail).
void chomp(std::string& s) {
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
}

bool read_tsv(const std::string& path, std::vector<std::vector<std::string>>& rows) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return false;
    std::string line;
    int c;
    while ((c = std::fgetc(f)) != EOF) {
        if (c == '\n') {
            chomp(line);
            if (!line.empty()) rows.push_back(split_tab(line));
            line.clear();
        } else {
            line.push_back(static_cast<char>(c));
        }
    }
    chomp(line);
    if (!line.empty()) rows.push_back(split_tab(line));
    std::fclose(f);
    return true;
}

// ---------------------------------------------------------------------------
// The model + tokenizer, loaded once.
// ---------------------------------------------------------------------------
struct Backbone {
    RwkvModel   model;
    Tokenizer   tok;
    bool        ok = false;
    std::string why;

    bool load(const std::string& path) {
        if (!model.load(path)) { why = model.error(); return false; }
        GgufLoader gg;
        if (!gg.open(path)) { why = "gguf reopen failed"; return false; }
        if (!tok.load_from_gguf(gg)) { why = "tokenizer load failed"; return false; }
        ok = true;
        return true;
    }
};

// ---------------------------------------------------------------------------
// Collector — accumulates rows and writes them in one go.
// ---------------------------------------------------------------------------
class Collector {
public:
    Collector(int32_t E, const std::vector<std::string>& set_names)
        : E_(E), set_names_(set_names) {}

    // `ids` must already include whatever BOS the caller wants. Returns the
    // final token's hidden state.
    void feed_and_capture(Backbone& bb, const std::vector<int32_t>& ids,
                          const std::vector<std::string>& labels,
                          const std::string& provenance_id) {
        if (ids.empty()) {
            ++skipped_;
            return;
        }
        RwkvState st;
        bb.model.init_state(st);
        Tensor logits("logits", {bb.model.config().n_vocab}, DType::F32);
        Tensor hidden("hidden", {E_}, DType::F32);
        for (const int32_t id : ids) bb.model.forward(id, st, logits, &hidden);
        push(hidden.f32(), labels, provenance_id);
    }

    // Streaming capture: feed `ids` into an EXISTING state and keep the state,
    // so the next bar continues the same recurrent summary. This is RWKV's
    // native mode (constant memory, unbounded context).
    void feed_stream(Backbone& bb, RwkvState& st, Tensor& logits, Tensor& hidden,
                     const std::vector<int32_t>& ids,
                     const std::vector<std::string>& labels,
                     const std::string& provenance_id) {
        if (ids.empty()) { ++skipped_; return; }
        for (const int32_t id : ids) bb.model.forward(id, st, logits, &hidden);
        push(hidden.f32(), labels, provenance_id);
    }

    // Warm-up: advance the recurrent state WITHOUT recording a row. Skipping
    // this would leave the state empty for the first recorded bar, which is
    // not a shorter context — it is a different, meaningless one.
    void feed_only(Backbone& bb, RwkvState& st, Tensor& logits, Tensor& hidden,
                   const std::vector<int32_t>& ids) {
        for (const int32_t id : ids) bb.model.forward(id, st, logits, &hidden);
    }

    void push(const float* h, const std::vector<std::string>& labels,
              const std::string& provenance_id) {
        data_.insert(data_.end(), h, h + E_);
        prov_.push_back(provenance_id);
        labels_.push_back(labels);
        ++n_;
    }

    int64_t n() const { return n_; }
    int64_t skipped() const { return skipped_; }

    bool write(const std::string& out_dir, const std::string& meta_json) const {
        const std::string hpath = out_dir + "/hidden.f32";
        const std::string lpath = out_dir + "/labels.tsv";
        const std::string mpath = out_dir + "/meta.json";

        FILE* f = std::fopen(hpath.c_str(), "wb");
        if (f == nullptr) return false;
        const bool ok = data_.empty() ||
            std::fwrite(data_.data(), sizeof(float), data_.size(), f) == data_.size();
        std::fclose(f);
        if (!ok) return false;

        FILE* g = std::fopen(lpath.c_str(), "wb");
        if (g == nullptr) return false;
        std::fprintf(g, "id");
        for (const std::string& s : set_names_) std::fprintf(g, "\t%s", s.c_str());
        std::fprintf(g, "\n");
        for (int64_t i = 0; i < n_; ++i) {
            std::fprintf(g, "%s", prov_[static_cast<size_t>(i)].c_str());
            for (const std::string& l : labels_[static_cast<size_t>(i)])
                std::fprintf(g, "\t%s", l.c_str());
            std::fprintf(g, "\n");
        }
        std::fclose(g);

        FILE* m = std::fopen(mpath.c_str(), "wb");
        if (m == nullptr) return false;
        std::fprintf(m, "%s\n", meta_json.c_str());
        std::fclose(m);
        return true;
    }

private:
    int32_t E_;
    std::vector<std::string> set_names_;
    std::vector<float> data_;
    std::vector<std::string> prov_;
    std::vector<std::vector<std::string>> labels_;
    int64_t n_ = 0;
    int64_t skipped_ = 0;
};

// ---------------------------------------------------------------------------
// Market window -> text.
//
// A price series is not a sentence, so the rendering is a deliberate choice
// and it is recorded in meta.json. Two rules:
//   * Percent changes, not raw levels. The regime engine is scale-free; giving
//     the model "145.47" makes every bar look unlike every other era of the
//     same instrument. "0.4%" is comparable across the whole file.
//   * Fixed decimals, no scientific notation. A tokenizer fed "1.2e-05" spends
//     tokens on the exponent and splits the number into pieces that carry no
//     numeric meaning.
// ---------------------------------------------------------------------------
std::string bar_to_text(const trading::Bar& b, const trading::Bar& prev) {
    const double base = prev.close != 0.0 ? prev.close : b.close;
    const double ret = base != 0.0 ? (b.close - base) / base * 100.0 : 0.0;
    const double rng = b.close != 0.0 ? (b.high - b.low) / b.close * 100.0 : 0.0;
    const double volr = prev.volume > 0.0 ? b.volume / prev.volume : 1.0;
    char buf[192];
    std::snprintf(buf, sizeof(buf),
                  "change %.2f%% range %.2f%% volume %.2fx",
                  ret, rng, volr);
    return std::string(buf);
}

std::string json_escape(const std::string& s) {
    std::string o;
    for (const char c : s) {
        if (c == '"' || c == '\\') { o.push_back('\\'); o.push_back(c); }
        else if (static_cast<unsigned char>(c) < 0x20u) o.push_back(' ');
        else o.push_back(c);
    }
    return o;
}

std::string u64s(uint64_t v) { return std::to_string(v); }

} // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr,
            "usage:\n"
            "  dump_hidden language <model.gguf> <examples.tsv> <out_dir> [--dry-run]\n"
            "  dump_hidden market   <model.gguf> <bars.csv>     <out_dir>\n"
            "                       [--window N] [--stride N] [--limit N]\n"
            "                       [--stream] [--dry-run]\n");
        return 2;
    }
    const std::string mode = argv[1];
    const std::string model_path = argv[2];
    const std::string in_path = argv[3];
    const std::string out_dir = argv[4];

    int32_t window = 24, stride = 1;
    int64_t limit = 0;
    bool stream = false, dry_run = false;
    for (int i = 5; i < argc; ++i) {
        const char* a = argv[i];
        if (std::strcmp(a, "--window") == 0 && i + 1 < argc) window = std::atoi(argv[++i]);
        else if (std::strcmp(a, "--stride") == 0 && i + 1 < argc) stride = std::atoi(argv[++i]);
        else if (std::strcmp(a, "--limit") == 0 && i + 1 < argc)
            limit = std::atoll(argv[++i]);
        else if (std::strcmp(a, "--stream") == 0) stream = true;
        else if (std::strcmp(a, "--dry-run") == 0) dry_run = true;
        else die(std::string("unknown argument: ") + a);
    }
    if (window < 2) window = 2;
    if (stride < 1) stride = 1;

    platform::set_quiet(true);
    ensure_dir(out_dir);

    Backbone bb;
    if (!bb.load(model_path)) die("cannot load backbone: " + bb.why);
    const int32_t E = bb.model.config().n_embd;

    if (mode == "language") {
        std::vector<std::vector<std::string>> rows;
        if (!read_tsv(in_path, rows)) die("cannot read " + in_path);
        if (rows.size() < 2) die("empty example file " + in_path);

        const std::vector<std::string>& header = rows[0];
        if (header.size() < 2) die("header must be: text<TAB>set.name[...]");
        std::vector<std::string> sets(header.begin() + 1, header.end());
        for (const std::string& s : sets)
            if (s.empty()) die("empty label-set name in header");

        Collector col(E, sets);
        int64_t total_tokens = 0, max_tokens = 0;
        for (size_t r = 1; r < rows.size(); ++r) {
            const std::vector<std::string>& row = rows[r];
            if (row.size() != header.size())
                die("row " + std::to_string(r) + " has " +
                    std::to_string(row.size()) + " columns, header has " +
                    std::to_string(header.size()));
            const std::string& text = row[0];
            std::vector<std::string> labels(row.begin() + 1, row.end());
            const std::vector<int32_t> ids = bb.tok.encode(text, /*add_bos=*/true);
            total_tokens += static_cast<int64_t>(ids.size());
            max_tokens = std::max<int64_t>(max_tokens, static_cast<int64_t>(ids.size()));
            if (dry_run) continue;
            col.feed_and_capture(bb, ids, labels, "ex" + std::to_string(r - 1));
        }

        if (dry_run) {
            std::printf("dry-run language: rows=%zu sets=%zu tokens=%lld max=%lld\n",
                        rows.size() - 1, sets.size(),
                        static_cast<long long>(total_tokens),
                        static_cast<long long>(max_tokens));
            return 0;
        }

        std::string sets_json;
        for (size_t i = 0; i < sets.size(); ++i)
            sets_json += (i ? "," : "") + std::string("\"") + json_escape(sets[i]) + "\"";
        const std::string meta =
            "{\"mode\":\"language\",\"model\":\"" + json_escape(model_path) +
            "\",\"E\":" + std::to_string(E) + ",\"n\":" + u64s(static_cast<uint64_t>(col.n())) +
            ",\"skipped\":" + u64s(static_cast<uint64_t>(col.skipped())) +
            ",\"label_sets\":[" + sets_json + "]}";
        if (!col.write(out_dir, meta)) die("cannot write to " + out_dir);
        std::printf("language: n=%lld E=%d tokens=%lld max_tokens_per_ex=%lld -> %s\n",
                    static_cast<long long>(col.n()), E,
                    static_cast<long long>(total_tokens),
                    static_cast<long long>(max_tokens), out_dir.c_str());
        return 0;
    }

    if (mode == "market") {
        std::vector<trading::Bar> bars;
        std::string err;
        if (!trading::load_bars_csv(in_path, bars, err)) die("load bars: " + err);
        if (bars.size() < static_cast<size_t>(window) + 2)
            die("not enough bars for window=" + std::to_string(window));

        // Ground truth for the regime label comes from the SAME engine the
        // runtime uses, so the head cannot learn a vocabulary the rest of the
        // system does not speak.
        trading::RegimeEngine eng;
        const std::vector<trading::RegimeState> regime = eng.scan_raw(bars);

        Collector col(E, {"trading.regime"});
        int64_t total_tokens = 0, max_tokens = 0;

        if (stream) {
            RwkvState st;
            bb.model.init_state(st);
            Tensor logits("logits", {bb.model.config().n_vocab}, DType::F32);
            Tensor hidden("hidden", {E}, DType::F32);
            for (size_t i = 1; i < bars.size(); ++i) {
                const std::string text = bar_to_text(bars[i], bars[i - 1]);
                std::vector<int32_t> ids = bb.tok.encode(text, /*add_bos=*/i == 1);
                total_tokens += static_cast<int64_t>(ids.size());
                max_tokens = std::max<int64_t>(max_tokens, static_cast<int64_t>(ids.size()));
                if (dry_run) continue;
                // The state is advanced on EVERY bar; only bars past the
                // warm-up produce a recorded (h, label) pair. See feed_only().
                if (i < static_cast<size_t>(window)) {
                    col.feed_only(bb, st, logits, hidden, ids);
                    continue;
                }
                col.feed_stream(bb, st, logits, hidden, ids,
                                {regime[i].label},
                                "bar" + std::to_string(i) + "@" +
                                    std::to_string(regime[i].ts));
                if (limit > 0 && col.n() >= limit) break;
            }
        } else {
            for (size_t i = static_cast<size_t>(window); i < bars.size();
                 i += static_cast<size_t>(stride)) {
                std::vector<int32_t> ids = bb.tok.encode("market", true);
                for (size_t k = i + 1 - static_cast<size_t>(window); k <= i; ++k)
                {
                    const std::vector<int32_t> more =
                        bb.tok.encode(bar_to_text(bars[k], bars[k - 1]), false);
                    ids.insert(ids.end(), more.begin(), more.end());
                }
                total_tokens += static_cast<int64_t>(ids.size());
                max_tokens = std::max<int64_t>(max_tokens, static_cast<int64_t>(ids.size()));
                if (dry_run) continue;
                col.feed_and_capture(bb, ids, {regime[i].label},
                                     "bar" + std::to_string(i) + "@" +
                                         std::to_string(regime[i].ts));
                if (limit > 0 && col.n() >= limit) break;
            }
        }

        if (dry_run) {
            std::printf("dry-run market: bars=%zu window=%d stride=%d stream=%d "
                        "tokens=%lld max=%lld\n",
                        bars.size(), window, stride, stream ? 1 : 0,
                        static_cast<long long>(total_tokens),
                        static_cast<long long>(max_tokens));
            return 0;
        }

        // Label histogram, so a degenerate label distribution is visible in
        // the log rather than discovered later as "the head always says range".
        std::vector<std::pair<std::string, int>> hist;
        for (size_t i = 0; i < bars.size(); ++i) {
            const std::string& l = regime[i].label;
            auto it = std::find_if(hist.begin(), hist.end(),
                                   [&](const std::pair<std::string, int>& p) {
                                       return p.first == l;
                                   });
            if (it == hist.end()) hist.emplace_back(l, 1);
            else ++it->second;
        }
        std::string hist_json;
        for (size_t i = 0; i < hist.size(); ++i)
            hist_json += (i ? "," : "") + std::string("\"") + json_escape(hist[i].first) +
                         "\":" + std::to_string(hist[i].second);

        const std::string meta =
            "{\"mode\":\"market\",\"model\":\"" + json_escape(model_path) +
            "\",\"bars\":\"" + json_escape(in_path) +
            "\",\"E\":" + std::to_string(E) +
            ",\"n\":" + u64s(static_cast<uint64_t>(col.n())) +
            ",\"window\":" + std::to_string(window) +
            ",\"stride\":" + std::to_string(stride) +
            ",\"stream\":" + (stream ? "true" : "false") +
            ",\"label_sets\":[\"trading.regime\"]"
            ",\"regime_histogram_all_bars\":{" + hist_json + "}"
            ",\"render\":\"change <pct>% range <pct>% volume <x>x\"}";
        if (!col.write(out_dir, meta)) die("cannot write to " + out_dir);
        std::printf("market: n=%lld E=%d window=%d stride=%d stream=%d "
                    "tokens=%lld max=%lld -> %s\n",
                    static_cast<long long>(col.n()), E, window, stride,
                    stream ? 1 : 0, static_cast<long long>(total_tokens),
                    static_cast<long long>(max_tokens), out_dir.c_str());
        return 0;
    }

    die("unknown mode: " + mode);
    return 2;
}
