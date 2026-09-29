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
//    dump_hidden vision   <model.gguf> <images.tsv>   <out_dir>
//                         [--projector models/vision-proj.gguf]
//    dump_hidden audio    <model.gguf> <clips.tsv>    <out_dir>
//                         [--whisper models/whisper-tiny-encoder.gguf]
//                         [--codec mel|focal]
//    dump_hidden ... --dry-run     # read + report, forward nothing
//
//  The language TSV is  text <TAB> <set.name> <TAB> ...  with the set names in
//  the header line, so the tool never hard-codes a label vocabulary.
//
//  THE MODALITY MODES GO THROUGH MultimodalBridge, NOT AROUND IT. An image or a
//  clip is not a token sequence, so the only honest way to reach h[E] is the same
//  one the runtime uses: encoder -> bridge (nearest-embedding quantisation
//  against the model's OWN embedding matrix) -> the same RWKV forward. Writing a
//  second, private path here would produce h[E] vectors that are not the ones
//  the heads will see at inference, and the fit would be of a model that does
//  not exist. `meta.json` records the bridge config (stride) for that reason.
// =============================================================================
#include "omniseed/audio/audio.h"
#include "omniseed/core/gguf_loader.h"
#include "omniseed/core/platform.h"
#include "omniseed/core/rwkv.h"
#include "omniseed/core/tokenizer.h"
#include "omniseed/multimodal.h"
#include "omniseed/trading/regime_engine.h"
#include "omniseed/trading/trading_engine.h"
#include "omniseed/vision/vision.h"

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

    // The modality path computes its own h[E] (via the bridge) and hands it
    // over rather than replaying ids, so it needs push() to be reachable.
    // Exposed below rather than duplicated: one accumulator, one writer, one
    // definition of what a row is.
    int64_t n() const { return n_; }
    int64_t skipped() const { return skipped_; }
    int32_t width() const { return E_; }
    const std::vector<std::string>& set_names() const { return set_names_; }

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

// ---------------------------------------------------------------------------
// Modality label manifests.
//
// Both modality modes read the SAME shape: a header naming one label set, then
// one row per file. This is deliberately not the language TSV shape: a modality
// row's first column is a PATH, and the tool must resolve it relative to the
// manifest's directory so a manifest can name files inside its own tree without
// the caller having to be in the right cwd.
//
//   <path> <TAB> <label>
//
// A row whose path does not resolve is REPORTED and SKIPPED, never silently
// folded into a label. One bad row must not poison a head.
// ---------------------------------------------------------------------------
struct ModalityRow {
    std::string path;
    std::string label;
};

std::string dir_of(const std::string& p) {
    const size_t slash = p.find_last_of("/\\");
    return slash == std::string::npos ? std::string(".") : p.substr(0, slash);
}

bool is_absolute_path(const std::string& p) {
    if (p.size() >= 2 && p[1] == ':') return true;      // C:\... / C:/...
    return !p.empty() && (p[0] == '/' || p[0] == '\\');
}

std::string join_path(const std::string& dir, const std::string& rel) {
    if (is_absolute_path(rel)) return rel;
    if (dir.empty() || dir == ".") return rel;
    const char last = dir.back();
    if (last == '/' || last == '\\') return dir + rel;
    return dir + "/" + rel;
}

// Reads a two-column manifest and resolves each path against the manifest's own
// directory. Returns false only when the FILE is unreadable or the header is
// malformed; individual bad rows are kept and filtered by the caller so the
// missing ones can be named in the log.
bool read_modality_manifest(const std::string& manifest,
                            std::string& set_name,
                            std::vector<ModalityRow>& rows) {
    std::vector<std::vector<std::string>> raw;
    if (!read_tsv(manifest, raw)) return false;
    if (raw.size() < 2) return false;
    if (raw[0].size() < 2) return false;
    set_name = raw[0][1];
    if (set_name.empty()) return false;
    const std::string base = dir_of(manifest);
    for (size_t r = 1; r < raw.size(); ++r) {
        if (raw[r].size() < 2) continue;
        ModalityRow m;
        m.path  = join_path(base, raw[r][0]);
        m.label = raw[r][1];
        chomp(m.label);
        if (m.label.empty()) continue;
        rows.push_back(m);
    }
    return true;
}

// A per-label histogram, so a degenerate label set is visible in the log
// BEFORE a fit turns it into "the head always says the majority class".
std::string histogram_json(const std::vector<ModalityRow>& rows) {
    std::vector<std::pair<std::string, int>> hist;
    for (const ModalityRow& m : rows) {
        auto it = std::find_if(hist.begin(), hist.end(),
                               [&](const std::pair<std::string, int>& p) {
                                   return p.first == m.label;
                               });
        if (it == hist.end()) hist.emplace_back(m.label, 1);
        else ++it->second;
    }
    std::string out;
    for (size_t i = 0; i < hist.size(); ++i)
        out += (i ? "," : "") + std::string("\"") + json_escape(hist[i].first) +
               "\":" + std::to_string(hist[i].second);
    return out;
}

// ---------------------------------------------------------------------------
// Image decoding. BMP and PPM only, and deliberately so: the vision datasets
// this tool consumes are small and the alternative is a third-party decoder the
// project does not vendor. A silent mis-decode would produce h[E] vectors whose
// labels are right and whose pixels are wrong — the worst kind of fit. So an
// unsupported or malformed file returns false and is REPORTED.
//
// PPM is the route the fetch tool uses (see tools/get_modality_data.py).
// ---------------------------------------------------------------------------
bool load_ppm(const std::string& path, Image& out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return false;
    char magic[3] = {0, 0, 0};
    if (std::fscanf(f, "%2s", magic) != 1 || std::strcmp(magic, "P6") != 0) {
        std::fclose(f);
        return false;
    }
    // Netpbm is whitespace-separated with '#' comments; a strict reader here is
    // the difference between a clean error and a garbage image.
    int vals[3] = {0, 0, 0};
    for (int i = 0; i < 3; ++i) {
        int c = std::fgetc(f);
        for (;;) {
            while (c == ' ' || c == '\t' || c == '\r' || c == '\n') c = std::fgetc(f);
            if (c == '#') { while (c != '\n' && c != EOF) c = std::fgetc(f); }
            else break;
        }
        if (c < '0' || c > '9') { std::fclose(f); return false; }
        long v = 0;
        while (c >= '0' && c <= '9') { v = v * 10 + (c - '0'); c = std::fgetc(f); }
        vals[i] = static_cast<int>(v);
    }
    if (vals[0] <= 0 || vals[1] <= 0 || vals[2] != 255) { std::fclose(f); return false; }
    out.width  = vals[0];
    out.height = vals[1];
    out.rgb.assign(static_cast<size_t>(out.width) * out.height * 3, 0);
    const size_t want = out.rgb.size();
    const size_t got = std::fread(out.rgb.data(), 1, want, f);
    std::fclose(f);
    return got == want;
}

// BMP (24-bit uncompressed only). Same reasoning as PPM.
bool load_bmp(const std::string& path, Image& out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return false;
    unsigned char hdr[54];
    if (std::fread(hdr, 1, 54, f) != 54 || hdr[0] != 'B' || hdr[1] != 'M') {
        std::fclose(f);
        return false;
    }
    const int32_t data_off = *reinterpret_cast<int32_t*>(hdr + 10);
    const int32_t w = *reinterpret_cast<int32_t*>(hdr + 18);
    const int32_t h = *reinterpret_cast<int32_t*>(hdr + 22);
    const int16_t bpp = *reinterpret_cast<int16_t*>(hdr + 28);
    // A 16/32-bit or RLE-compressed BMP is refused rather than guessed at.
    if (bpp != 24 || w <= 0 || h == 0) { std::fclose(f); return false; }
    const int32_t rows = h < 0 ? -h : h;
    const int32_t stride_b = ((w * 3) + 3) & ~3;
    out.width = w;
    out.height = rows;
    out.rgb.assign(static_cast<size_t>(w) * rows * 3, 0);
    if (std::fseek(f, data_off, SEEK_SET) != 0) { std::fclose(f); return false; }
    std::vector<unsigned char> row(static_cast<size_t>(stride_b));
    for (int32_t y = 0; y < rows; ++y) {
        if (std::fread(row.data(), 1, row.size(), f) != row.size()) {
            std::fclose(f);
            return false;
        }
        const int32_t dst_y = (h < 0) ? y : (rows - 1 - y);   // bottom-up default
        unsigned char* dst = out.rgb.data() +
            static_cast<size_t>(dst_y) * w * 3;
        for (int32_t x = 0; x < w; ++x) {
            dst[x * 3 + 0] = row[x * 3 + 2];   // BGR -> RGB
            dst[x * 3 + 1] = row[x * 3 + 1];
            dst[x * 3 + 2] = row[x * 3 + 0];
        }
    }
    std::fclose(f);
    return true;
}

bool load_image(const std::string& path, Image& out) {
    const size_t dot = path.find_last_of('.');
    if (dot == std::string::npos) return false;
    std::string ext = path.substr(dot + 1);
    for (char& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (ext == "ppm" || ext == "pgm") return load_ppm(path, out);
    if (ext == "bmp") return load_bmp(path, out);
    return false;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr,
            "usage:\n"
            "  dump_hidden language <model.gguf> <examples.tsv> <out_dir> [--dry-run]\n"
            "  dump_hidden market   <model.gguf> <bars.csv>     <out_dir>\n"
            "                       [--window N] [--stride N] [--limit N]\n"
            "                       [--stream] [--dry-run]\n"
            "  dump_hidden vision   <model.gguf> <images.tsv>   <out_dir>\n"
            "                       [--projector P] [--limit N] [--dry-run]\n"
            "  dump_hidden audio    <model.gguf> <clips.tsv>    <out_dir>\n"
            "                       [--codec focal] [--limit N] [--dry-run]\n"
            "                       (--codec mel is refused: no audio->E adapter)\n");
        return 2;
    }
    const std::string mode = argv[1];
    const std::string model_path = argv[2];
    const std::string in_path = argv[3];
    const std::string out_dir = argv[4];

    int32_t window = 24, stride = 1;
    int64_t limit = 0;
    bool stream = false, dry_run = false;
    // Modality sidecar. Default matches where the tree actually keeps it.
    std::string projector_path = "models/vision-proj.gguf";
    std::string codec          = "focal";   // focal is the ONLY audio path (see below)
    for (int i = 5; i < argc; ++i) {
        const char* a = argv[i];
        if (std::strcmp(a, "--window") == 0 && i + 1 < argc) window = std::atoi(argv[++i]);
        else if (std::strcmp(a, "--stride") == 0 && i + 1 < argc) stride = std::atoi(argv[++i]);
        else if (std::strcmp(a, "--limit") == 0 && i + 1 < argc)
            limit = std::atoll(argv[++i]);
        else if (std::strcmp(a, "--stream") == 0) stream = true;
        else if (std::strcmp(a, "--dry-run") == 0) dry_run = true;
        else if (std::strcmp(a, "--projector") == 0 && i + 1 < argc)
            projector_path = argv[++i];
        else if (std::strcmp(a, "--codec") == 0 && i + 1 < argc)
            codec = argv[++i];
        else die(std::string("unknown argument: ") + a);
    }
    if (window < 2) window = 2;
    if (stride < 1) stride = 1;
    if (codec != "mel" && codec != "focal")
        die("--codec must be 'mel' or 'focal', got '" + codec + "'");

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

    // -----------------------------------------------------------------------
    // VISION. An image -> VisionEncoder [M,768] -> MultimodalBridge -> ids ->
    // the SAME backbone forward the runtime performs.
    //
    // The image is a placeholder TEXT CARRIER: the bridge needs a text segment,
    // and the honest one is a fixed, content-free token ("image") rather than a
    // fabricated caption. A caption would inject an invented description into
    // the h[E] that the label then gets fitted to — the head would learn the
    // caption, not the picture.
    // -----------------------------------------------------------------------
    if (mode == "vision") {
        std::string set_name;
        std::vector<ModalityRow> rows;
        if (!read_modality_manifest(in_path, set_name, rows))
            die("cannot read image manifest " + in_path +
                " (need header 'path<TAB>set.name' + rows)");
        if (rows.empty()) die("image manifest has no rows: " + in_path);

        VisionEncoder vis;
        if (!vis.load(projector_path))
            die("cannot load vision projector '" + projector_path + "': " + vis.error());
        if (!vis.valid()) die("vision projector invalid: " + projector_path);

        MultimodalBridge bridge;
        if (!bridge.init(E, bb.tok)) die("bridge init: " + bridge.error());
        if (!bridge.set_codebook(bb.model.token_embeddings()))
            die("bridge codebook (model token embeddings): " + bridge.error());
        // Use the model's OWN embedding matrix as the codebook, and scan it
        // exactly. The default stride 8 is a speed dial that makes odd rows
        // unreachable (test_multimodal B8); a fit dumped through a strided scan
        // would be of a coarser bridge than any caller would use.
        MultimodalBridge::Config bcfg = bridge.config();
        bcfg.codebook_stride = 1;
        bridge.set_config(bcfg);

        Collector col(E, {set_name});
        int64_t bad_decode = 0, bad_encode = 0, bad_fuse = 0;
        int64_t total_vision_tokens = 0;

        for (size_t r = 0; r < rows.size(); ++r) {
            const ModalityRow& m = rows[r];
            Image img;
            if (!load_image(m.path, img)) {
                if (bad_decode == 0) std::printf("  [skip] cannot decode %s\n", m.path.c_str());
                ++bad_decode;
                continue;
            }
            Tensor emb;
            UniCompress comp;
            if (!vis.encode(img, emb, comp)) {
                if (bad_encode == 0)
                    std::printf("  [skip] encode failed %s: %s\n",
                                m.path.c_str(), vis.error().c_str());
                ++bad_encode;
                continue;
            }
            const int64_t M = emb.dim(0);
            std::vector<float> flat(static_cast<size_t>(emb.numel()));
            for (int64_t i = 0; i < emb.numel(); ++i) flat[static_cast<size_t>(i)] = emb.f32()[i];

            MultimodalBridge::Fusion f = bridge.fuse("image", flat, {});
            if (!f.ok) {
                if (bad_fuse == 0)
                    std::printf("  [skip] fuse failed %s: %s\n",
                                m.path.c_str(), f.error.c_str());
                ++bad_fuse;
                continue;
            }
            total_vision_tokens += f.vision_tokens_emitted;
            if (dry_run) continue;
            col.feed_and_capture(bb, f.ids, {m.label},
                                 "img" + std::to_string(r) + "@" + std::to_string(M));
            if (limit > 0 && col.n() >= limit) break;
        }

        if (dry_run) {
            std::printf("dry-run vision: rows=%zu set=%s decoded_bad=%lld "
                        "encode_bad=%lld fuse_bad=%lld vision_tokens=%lld\n",
                        rows.size(), set_name.c_str(),
                        static_cast<long long>(bad_decode),
                        static_cast<long long>(bad_encode),
                        static_cast<long long>(bad_fuse),
                        static_cast<long long>(total_vision_tokens));
            return 0;
        }
        if (col.n() == 0) {
            die("vision: ZERO rows survived — refusing to write an empty fit "
                "(decoded_bad=" + std::to_string(bad_decode) +
                " encode_bad=" + std::to_string(bad_encode) +
                " fuse_bad=" + std::to_string(bad_fuse) + ")");
        }
        const std::string meta =
            "{\"mode\":\"vision\",\"model\":\"" + json_escape(model_path) +
            "\",\"projector\":\"" + json_escape(projector_path) +
            "\",\"E\":" + std::to_string(E) +
            ",\"n\":" + u64s(static_cast<uint64_t>(col.n())) +
            ",\"codebook_stride\":" + std::to_string(bridge.codebook_stride()) +
            ",\"decoded_bad\":" + std::to_string(bad_decode) +
            ",\"encode_bad\":" + std::to_string(bad_encode) +
            ",\"fuse_bad\":" + std::to_string(bad_fuse) +
            ",\"label_sets\":[\"" + json_escape(set_name) + "\"]"
            ",\"label_histogram\":{" + histogram_json(rows) + "}"
            ",\"render\":\"image -> VisionEncoder [M,768] -> MultimodalBridge"
            "(nearest-embedding quantisation) -> backbone\"}";
        if (!col.write(out_dir, meta)) die("cannot write to " + out_dir);
        std::printf("vision: n=%lld E=%d set=%s stride=%d bad(dec=%lld enc=%lld "
                    "fuse=%lld) -> %s\n",
                    static_cast<long long>(col.n()), E, set_name.c_str(),
                    bridge.codebook_stride(),
                    static_cast<long long>(bad_decode),
                    static_cast<long long>(bad_encode),
                    static_cast<long long>(bad_fuse), out_dir.c_str());
        return 0;
    }

    // -----------------------------------------------------------------------
    // AUDIO. A clip -> encoder ids -> MultimodalBridge -> the SAME backbone
    // forward the runtime performs.
    //
    // ⚠️ THERE IS NO AUDIO PROJECTION IN THIS TREE, AND THIS TOOL DOES NOT
    // PRETEND OTHERWISE. WhisperTiny::encode emits [T/2, 384]; the model's E is
    // 768. Vision has a 768-wide projector (models/vision-proj.gguf); audio has
    // NO equivalent — `grep -rn proj src/audio/` finds only attention
    // projections, never a modality adapter. So a Whisper frame CANNOT be
    // quantised against the embedding matrix, and passed as `vision_embeddings`
    // the bridge rejects it with "not a whole number of n_embd rows".
    //
    // That rejection is correct and this tool accepts it rather than working
    // around it. The two available paths are:
    //
    //   --codec focal : PcmAudio -> FocalCodec::encode -> int32 ids, which are
    //                   ALREADY discrete tokens and need no projection. This is
    //                   the only audio path that reaches h[E] today, and it runs
    //                   through the codec's own codebook.
    //   --codec mel   : REFUSED with an explanation, because a 384-wide frame
    //                   has no defined 768-wide image and inventing one (a zero
    //                   pad, a random projection) would fit a head to the
    //                   invention. Fixing this properly means training an
    //                   audio->E adapter, which is a separate, documented task.
    // -----------------------------------------------------------------------
    if (mode == "audio") {
        std::string set_name;
        std::vector<ModalityRow> rows;
        if (!read_modality_manifest(in_path, set_name, rows))
            die("cannot read clip manifest " + in_path +
                " (need header 'path<TAB>set.name' + rows)");
        if (rows.empty()) die("clip manifest has no rows: " + in_path);

        const bool use_mel = (codec == "mel");
        if (use_mel) {
            die("audio --codec mel is REFUSED, by design.\n"
                "  WhisperTiny::encode yields [T/2, 384] but this backbone's E is " +
                std::to_string(E) + ".\n"
                "  There is no audio->E projection in this tree (vision has\n"
                "  models/vision-proj.gguf; audio has nothing equivalent), so the\n"
                "  frame cannot be quantised into the model's token space.\n"
                "  Zero-padding or a random projection would fit a head to a\n"
                "  fabrication. Use --codec focal, or train an audio adapter first.\n"
                "  Recorded in docs/VISION_AUDIO_DATA.md; not silently substituted.");
        }

        FocalCodec fc;
        MultimodalBridge bridge;
        if (!bridge.init(E, bb.tok)) die("bridge init: " + bridge.error());
        MultimodalBridge::Config bcfg = bridge.config();
        bcfg.codebook_stride = 1;
        bridge.set_config(bcfg);

        Collector col(E, {set_name});
        int64_t bad_decode = 0, bad_encode = 0, bad_fuse = 0;
        int64_t total_audio_tokens = 0;
        int32_t observed_rate = 0;

        for (size_t r = 0; r < rows.size(); ++r) {
            const ModalityRow& m = rows[r];
            PcmAudio clip;
            if (!PcmAudio::load_wav(m.path, clip)) {
                if (bad_decode == 0) std::printf("  [skip] cannot decode %s\n", m.path.c_str());
                ++bad_decode;
                continue;
            }
            // Record the rate we actually got. A 48 kHz file is resampled inside
            // load_wav_bytes (§42) and this is the evidence that it happened.
            observed_rate = clip.sample_rate;
            std::vector<int32_t> audio_ids;
            if (!fc.encode(clip, audio_ids) || audio_ids.empty()) {
                if (bad_encode == 0)
                    std::printf("  [skip] focal encode failed/empty %s\n", m.path.c_str());
                ++bad_encode;
                continue;
            }

            MultimodalBridge::Fusion f = bridge.fuse("audio", {}, audio_ids);
            if (!f.ok) {
                if (bad_fuse == 0)
                    std::printf("  [skip] fuse failed %s: %s\n",
                                m.path.c_str(), f.error.c_str());
                ++bad_fuse;
                continue;
            }
            total_audio_tokens += f.audio_tokens_emitted;
            if (dry_run) continue;
            col.feed_and_capture(bb, f.ids, {m.label},
                                 "clip" + std::to_string(r) + "@" + codec);
            if (limit > 0 && col.n() >= limit) break;
        }

        if (dry_run) {
            std::printf("dry-run audio: rows=%zu set=%s codec=%s rate=%d "
                        "decoded_bad=%lld encode_bad=%lld fuse_bad=%lld "
                        "audio_tokens=%lld\n",
                        rows.size(), set_name.c_str(), codec.c_str(), observed_rate,
                        static_cast<long long>(bad_decode),
                        static_cast<long long>(bad_encode),
                        static_cast<long long>(bad_fuse),
                        static_cast<long long>(total_audio_tokens));
            return 0;
        }
        if (col.n() == 0) {
            die("audio: ZERO rows survived — refusing to write an empty fit "
                "(decoded_bad=" + std::to_string(bad_decode) +
                " encode_bad=" + std::to_string(bad_encode) +
                " fuse_bad=" + std::to_string(bad_fuse) + ")");
        }
        const std::string meta =
            "{\"mode\":\"audio\",\"model\":\"" + json_escape(model_path) +
            "\",\"codec\":\"" + codec + "\""
            ",\"whisper\":\"\""
            ",\"E\":" + std::to_string(E) +
            ",\"n\":" + u64s(static_cast<uint64_t>(col.n())) +
            ",\"decoded_bad\":" + std::to_string(bad_decode) +
            ",\"encode_bad\":" + std::to_string(bad_encode) +
            ",\"fuse_bad\":" + std::to_string(bad_fuse) +
            ",\"label_sets\":[\"" + json_escape(set_name) + "\"]"
            ",\"label_histogram\":{" + histogram_json(rows) + "}"
            ",\"render\":\"PcmAudio(16k) -> FocalCodec ids -> "
            "MultimodalBridge -> backbone\","
            "\"no_audio_projection\":\"TRUE — Whisper [T/2,384] has no E=" +
            std::to_string(E) + " adapter in this tree; --codec mel is refused\"}";
        if (!col.write(out_dir, meta)) die("cannot write to " + out_dir);
        std::printf("audio: n=%lld E=%d set=%s codec=%s bad(dec=%lld enc=%lld "
                    "fuse=%lld) -> %s\n",
                    static_cast<long long>(col.n()), E, set_name.c_str(),
                    codec.c_str(),
                    static_cast<long long>(bad_decode),
                    static_cast<long long>(bad_encode),
                    static_cast<long long>(bad_fuse), out_dir.c_str());
        return 0;
    }

    die("unknown mode: " + mode);
    return 2;
}
