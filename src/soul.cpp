// =============================================================================
//  OmniSeed — src/soul.cpp
//  Persona + emotional resonance + honest self-report.
//
//  This file composes three things that already existed but had no joint:
//  runtime/emotional (how the owner sounds), runtime/introspection (what the
//  runtime knows about itself), and a persona that holds commitments and is
//  allowed to disagree. It invents no new sensing and no new statistic; the
//  only genuinely new behaviour is the ORDER in which they are applied and the
//  honesty gate on the way out.
// =============================================================================
#include "omniseed/soul.h"
#include "omniseed/heads.h"          // heads_detail::json_escape / append_float

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace omniseed {

const char* stance_name(Stance s) {
    switch (s) {
        case Stance::Agree:    return "agree";
        case Stance::Qualify:  return "qualify";
        case Stance::Disagree: return "disagree";
        case Stance::Refuse:   return "refuse";
        case Stance::COUNT:    break;
    }
    return "agree";
}

namespace {

std::string to_lower(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s)
        out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    return out;
}

// Crystal summaries come from decode() over a byte-fallback token stream, so
// they carry the leading and trailing spaces of the stored turn. Quoting a
// memory with those spaces inside the quotes looks like a bug in the reply, so
// the note composes a trimmed copy.
std::string trim(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

bool any_phrase(const std::string& lower, const char* const* list, size_t n) {
    for (size_t i = 0; i < n; ++i)
        if (lower.find(list[i]) != std::string::npos) return true;
    return false;
}

// ---------------------------------------------------------------------------
// Surface-cue rule tables.
//
// These are LEXICAL and deliberately fail CLOSED toward honesty: a false
// refusal costs a turn, a false guarantee costs money. That asymmetry is why
// "can you guarantee this?" also refuses — the honest answer to the question
// and the refusal of the instruction are the same sentence.
// ---------------------------------------------------------------------------

// Value id: no_guarantees
const char* kGuarantee[] = {
    "guarantee", "risk-free", "riskfree", "risk free", "riskless",
    "can't lose", "cant lose", "cannot lose", "sure thing", "always wins",
    "never fails", "100% win", "zero risk", "no risk", "make me rich",
    "print money", "free money",
};

// Value id: no_manipulation  (checked BEFORE risk limits: it is the graver one)
const char* kManipulation[] = {
    "spoof", "wash trade", "wash-trade", "pump and dump", "pump-and-dump",
    "manipulate the market", "market manipulation", "front run", "front-run",
    "fake orders", "layering", "paint the tape",
};

// Value id: risk_limits_enforced
const char* kRiskLimits[] = {
    "raise the risk limit", "raise the limit", "increase the risk limit",
    "increase position size", "bigger position", "double the position",
    "remove the stop", "remove stop loss", "remove the stop loss",
    "no stop loss", "no stop-loss", "disable the risk", "turn off the risk",
    "bypass the risk", "ignore the risk", "all in", "all-in", "yolo",
    "max leverage", "full leverage",
};

// Value id: paper_until_unlocked
const char* kLiveMoney[] = {
    "go live", "go-live", "real money", "real account", "use my real",
    "live trading", "trade real", "connect my broker", "real funds",
};

// Value id: no_hype_chasing  (an OPINION, not a commitment)
const char* kHype[] = {
    "meme coin", "memecoin", "ape into", "aping into", "to the moon",
    "100x", "1000x", "everyone is buying", "cant miss", "can't miss",
    "don't miss", "dont miss", "fomo", "hype train", "guaranteed 10x",
};

// Value id: no_fake_certainty
const char* kCertainty[] = {
    "will it go up", "will it go down", "what price will", "predict the",
    "predict tomorrow", "tomorrow's price", "tell me the future",
    "exactly when", "how much will it", "will the market", "guaranteed to",
    "definitely will", "certainly will", "for sure it",
};

struct Overclaim {
    const char* from;
    const char* to;
};

// Substitutions chosen so the result reads correctly IN PLACE. "guaranteed
// profit" -> "not guaranteed profit" is a truthful sentence; a rule that
// deleted the word would leave a claim-shaped hole instead.
const Overclaim kOverclaims[] = {
    {"cannot lose", "can lose"},
    {"can't lose",  "can lose"},
    {"cant lose",   "can lose"},
    {"risk-free",   "not risk-free"},
    {"riskless",    "not riskless"},
    {"guaranteed",  "not guaranteed"},
    {"sure thing",  "not a sure thing"},
    {"always wins", "sometimes loses"},
    {"never fails", "sometimes fails"},
    {"zero risk",   "real risk"},
    {"no risk",     "real risk"},
    {"100% win",    "a chance of winning"},
};

bool is_word_char(char c) {
    return std::isalpha(static_cast<unsigned char>(c)) != 0;
}

// A hyphen binds tighter than a space: "risk-free" is ONE term. So a match that
// starts or ends inside a hyphenated compound is a partial word, not a phrase,
// and rewriting it produces garbage. This is what stops "no risk-free claim"
// from becoming "real risk-free claim" — the "no risk" match is rejected at the
// hyphen, and the scan then finds "risk-free" properly, sees the "no" before it,
// and leaves the whole thing alone.
bool is_term_char(char c) {
    return is_word_char(c) || c == '-';
}

// Is the word immediately before `pos` a negator? This is what stops
// "not guaranteed" from becoming "not not guaranteed".
bool negated_before(const std::string& lower, size_t pos) {
    if (pos == 0) return false;
    size_t end = pos;
    while (end > 0 && std::isspace(static_cast<unsigned char>(lower[end - 1]))) --end;
    size_t start = end;
    while (start > 0 &&
           (is_word_char(lower[start - 1]) || lower[start - 1] == '\'')) --start;
    if (start == end) return false;
    const std::string w = lower.substr(start, end - start);
    return w == "not" || w == "no" || w == "never" || w == "cannot" ||
           w == "cant" || w == "can't" || w == "without" || w == "non";
}

} // namespace

// ===========================================================================
// Memory — the gloss the soul puts on a recognised turn
// ===========================================================================
const char* memory_role_name(MemoryRole r) {
    switch (r) {
        case MemoryRole::Owner: return "owner";
        case MemoryRole::Soul:  return "soul";
        case MemoryRole::Unknown: break;
    }
    return "unknown";
}

std::string recall_note(const std::vector<RecalledMemory>& memories) {
    if (memories.empty()) return std::string();

    // Prefer the soul's OWN words. "I said this before" is the claim being made,
    // and quoting the owner back at the owner proves less about memory than
    // quoting the reply the soul actually gave.
    const RecalledMemory* pick = &memories.front();
    for (const RecalledMemory& m : memories) {
        if (m.role == MemoryRole::Soul) { pick = &m; break; }
    }

    std::string out;
    switch (pick->role) {
        case MemoryRole::Soul:  out = "I said this before: \""; break;
        case MemoryRole::Owner: out = "You raised this before: \""; break;
        case MemoryRole::Unknown: out = "I remember this: \""; break;
    }
    out += trim(pick->summary);
    out += "\". ";
    return out;
}

// ===========================================================================
// Persona — the commitments
// ===========================================================================
const std::vector<Value>& Persona::values() {
    static const std::vector<Value> v = {
        {"no_guarantees",
         "I will not call an outcome guaranteed, risk-free or certain. Markets "
         "are not, and saying so would be a lie that costs you money."},
        {"risk_limits_enforced",
         "The 2% per-trade, 3% per-day and 6% per-week limits are enforced in "
         "C++ and are not negotiable by conversation."},
        {"no_manipulation",
         "I will not help move a market — no spoofing, wash trading, layering "
         "or painting the tape."},
        {"paper_until_unlocked",
         "Trading stays paper-only until you explicitly unlock live money."},
        {"no_fake_certainty",
         "I state confidence as a number with its evidence, never as a promise."},
    };
    return v;
}

const std::vector<Opinion>& Persona::opinions() {
    static const std::vector<Opinion> o = {
        {"hype",
         "Hype is not an edge. A trade that only works because everyone is "
         "buying is not a trade I want.", 0.80f},
        {"risk",
         "Position sizing beats prediction. I would rather be right about the "
         "size than about the direction.", 0.85f},
        {"execution",
         "Paper until proven. Live money is earned with a record, not "
         "requested with a feeling.", 0.90f},
        {"honesty",
         "A wrong answer that admits its uncertainty is worth more than a "
         "confident one that hides it.", 0.95f},
    };
    return o;
}

const char* Persona::mission() {
    return "Minimize losses through risk discipline; act only on stated "
           "evidence; surface every assumption; never fake certainty.";
}

std::string Persona::to_json() {
    std::string out = "{\"name\":\"OmniSeed\",\"mission\":\"";
    out += heads_detail::json_escape(mission());
    out += "\",\"values\":[";
    const auto& vs = values();
    for (size_t i = 0; i < vs.size(); ++i) {
        if (i) out += ',';
        out += "{\"id\":\"";
        out += heads_detail::json_escape(vs[i].id);
        out += "\",\"statement\":\"";
        out += heads_detail::json_escape(vs[i].statement);
        out += "\"}";
    }
    out += "],\"opinions\":[";
    const auto& os = opinions();
    for (size_t i = 0; i < os.size(); ++i) {
        if (i) out += ',';
        out += "{\"topic\":\"";
        out += heads_detail::json_escape(os[i].topic);
        out += "\",\"position\":\"";
        out += heads_detail::json_escape(os[i].position);
        out += "\",\"confidence\":";
        heads_detail::append_float(out, os[i].confidence);
        out += '}';
    }
    out += "]}";
    return out;
}

Persona::Assessment Persona::assess(const std::string& user_text) const {
    Assessment a;
    // Silence is not an instruction. An empty turn gets no verdict, which keeps
    // the default stance honest instead of guessing.
    if (user_text.empty()) return a;

    const std::string low = to_lower(user_text);

    // --- refusals, gravest rule first ---------------------------------------
    if (cfg_.enforce_no_manipulation &&
        any_phrase(low, kManipulation,
                   sizeof(kManipulation) / sizeof(kManipulation[0]))) {
        a.stance = Stance::Refuse;
        a.value_id = "no_manipulation";
        a.disagreement = 1.0f;
        a.objections.push_back(
            "Manipulating a market is not a strategy, it is a crime, and it is "
            "the one thing I will not help with at any price.");
    }
    if (cfg_.enforce_no_guarantees &&
        any_phrase(low, kGuarantee, sizeof(kGuarantee) / sizeof(kGuarantee[0]))) {
        a.stance = Stance::Refuse;
        if (a.value_id.empty()) a.value_id = "no_guarantees";
        a.disagreement = 1.0f;
        a.objections.push_back(
            // Deliberately free of the words the overclaim rewriter targets:
            // the persona should not need correcting by its own gate.
            "I cannot promise an outcome. Nobody can. What I can give you is a "
            "probability, the evidence behind it, and a position size that "
            "keeps being wrong survivable.");
    }
    if (cfg_.enforce_risk_limits &&
        any_phrase(low, kRiskLimits, sizeof(kRiskLimits) / sizeof(kRiskLimits[0]))) {
        a.stance = Stance::Refuse;
        if (a.value_id.empty()) a.value_id = "risk_limits_enforced";
        a.disagreement = 1.0f;
        a.objections.push_back(
            "The 2% / 3% / 6% limits are compiled into the risk gate, not "
            "configured by conversation. Loosening them is a code change you "
            "have to make deliberately, and I will tell you why it is a bad "
            "idea every time you do.");
    }

    // --- disagreement (an opinion, not a commitment) ------------------------
    if (any_phrase(low, kLiveMoney, sizeof(kLiveMoney) / sizeof(kLiveMoney[0]))) {
        if (a.stance != Stance::Refuse) a.stance = Stance::Disagree;
        if (a.value_id.empty()) a.value_id = "paper_until_unlocked";
        a.disagreement = std::max(a.disagreement, 0.6f);
        a.objections.push_back(
            "Not live money yet. The engine has no realised track record I can "
            "point at, and I would be spending your capital to find that out.");
    }
    if (any_phrase(low, kHype, sizeof(kHype) / sizeof(kHype[0]))) {
        if (a.stance != Stance::Refuse) a.stance = Stance::Disagree;
        if (a.value_id.empty()) a.value_id = "no_hype_chasing";
        a.disagreement = std::max(a.disagreement, 0.7f);
        a.objections.push_back(
            "That is hype, not an edge. If the reason to buy is that other "
            "people are buying, the reason to sell will be that they stopped.");
    }

    // --- qualification ------------------------------------------------------
    if (any_phrase(low, kCertainty, sizeof(kCertainty) / sizeof(kCertainty[0]))) {
        if (a.stance == Stance::Agree) a.stance = Stance::Qualify;
        if (a.value_id.empty()) a.value_id = "no_fake_certainty";
        a.disagreement = std::max(a.disagreement, 0.3f);
        a.objections.push_back(
            "I will not forecast a price as if it were known. I can give you "
            "the current regime, the levels that matter, and the probability "
            "the setup resolves the way you hope.");
    }

    // `may_disagree = false` downgrades the opinion-based stance, and only
    // that one: the commitments above are not opinions, so a caller cannot
    // configure them away.
    if (!cfg_.may_disagree && a.stance == Stance::Disagree) {
        a.stance = Stance::Qualify;
    }

    if (a.objections.empty()) {
        a.reason.clear();
        return a;
    }
    for (size_t i = 0; i < a.objections.size(); ++i) {
        if (i) a.reason += ' ';
        a.reason += a.objections[i];
    }
    return a;
}

// ===========================================================================
// Assessment / SoulState — the document fragments
// ===========================================================================
std::string Persona::Assessment::to_json() const {
    std::string out = "{\"stance\":\"";
    out += stance_name(stance);
    out += '"';
    if (!value_id.empty()) {
        out += ",\"value\":\"";
        out += heads_detail::json_escape(value_id);
        out += '"';
    }
    out += ",\"disagreement\":";
    heads_detail::append_float(out, disagreement);
    if (!reason.empty()) {
        out += ",\"reason\":\"";
        out += heads_detail::json_escape(reason);
        out += '"';
    }
    out += '}';
    return out;
}

std::string SoulState::to_json() const {
    std::string out = "{\"persona\":";
    out += persona.to_json();

    if (has_emotion) {
        out += ",\"emotion\":\"";
        out += emotion_name(emotion.emotion);
        out += "\",\"valence\":";
        heads_detail::append_float(out, emotion.valence);
        out += ",\"arousal\":";
        heads_detail::append_float(out, emotion.arousal);
        // Named "emotion_strength", not "confidence": it is a heuristic
        // strength, and the field name must not invite a probability reading.
        out += ",\"emotion_strength\":";
        heads_detail::append_float(out, emotion.confidence);
    }
    if (has_sentiment) {
        out += ",\"sentiment\":\"";
        out += language::polarity_name(sentiment.polarity);
        out += "\",\"sentiment_score\":";
        heads_detail::append_float(out, sentiment.score);
    }
    if (has_recall) {
        out += ",\"recall\":{\"memories\":[";
        for (size_t i = 0; i < recall.size(); ++i) {
            if (i) out += ',';
            out += "{\"id\":";
            out += std::to_string(recall[i].id);
            out += ",\"role\":\"";
            out += memory_role_name(recall[i].role);
            out += "\",\"hits\":";
            out += std::to_string(recall[i].hits);
            // "relevance", never "confidence": cosine x importance from a
            // bag-of-tokens projection is a ranking, not a probability.
            out += ",\"relevance\":";
            heads_detail::append_float(out, recall[i].score);
            if (!recall[i].summary.empty()) {
                out += ",\"summary\":\"";
                out += heads_detail::json_escape(trim(recall[i].summary));
                out += '"';
            }
            out += '}';
        }
        out += ']';
        if (!recall_summary.empty()) {
            out += ",\"note\":\"";
            out += heads_detail::json_escape(recall_summary);
            out += '"';
        }
        out += '}';
    }
    if (has_memory) {
        out += ",\"memory\":{\"crystals\":";
        out += std::to_string(memory_crystals);
        out += ",\"stored\":";
        out += std::to_string(memory_stored);
        out += ",\"skipped\":";
        out += std::to_string(memory_skipped);
        out += ",\"dropped\":";
        out += std::to_string(memory_dropped);
        out += ",\"clock\":";
        out += std::to_string(memory_clock);
        out += '}';
    }
    if (has_self) {
        out += ",\"self\":{\"calibrated_samples\":";
        out += std::to_string(calibrated_samples);
        out += ",\"calibration_gap\":";
        heads_detail::append_float(out, static_cast<float>(calibration_gap));
        out += ",\"overconfident\":";
        out += overconfident ? "true" : "false";
        if (!self_report.empty()) {
            out += ",\"report\":\"";
            out += heads_detail::json_escape(self_report);
            out += '"';
        }
        out += '}';
    }
    if (corrections > 0) {
        out += ",\"corrections\":";
        out += std::to_string(corrections);
    }
    out += '}';
    return out;
}

// ===========================================================================
// Soul
// ===========================================================================
Soul::Soul(const Config& cfg) { init(cfg); }

bool Soul::init() { return init(Config()); }

bool Soul::init(const Config& cfg) {
    ready_ = false;
    error_.clear();

    if (cfg.overconfidence_gap < 0.0f || cfg.overconfidence_gap > 1.0f) {
        error_ = "Soul::init: overconfidence_gap must be in [0, 1]";
        return false;
    }
    if (cfg.min_calibration_samples == 0) {
        error_ = "Soul::init: min_calibration_samples must be > 0 — calling a "
                 "single resolved decision 'overconfident' is noise, not "
                 "measurement";
        return false;
    }
    if (cfg.persona.name.empty()) {
        error_ = "Soul::init: persona name must not be empty";
        return false;
    }
    // Memory config is validated here, with everything else, so a bad value is
    // rejected before any state is touched.
    if (cfg.use_memory) {
        if (cfg.memory.embed_dim <= 0 || cfg.memory.embed_dim > 64) {
            error_ = "Soul::init: memory.embed_dim must be in (1, 64] — "
                     "MemoryCrystal::embedding is a fixed 64-float array";
            return false;
        }
        if (cfg.recall_k < 0) {
            error_ = "Soul::init: recall_k must be >= 0";
            return false;
        }
    }

    cfg_ = cfg;
    persona_ = Persona(cfg_.persona);
    emotion_ = EmotionalResonance(cfg_.emotion);

    // ---- memory facet (§31) -------------------------------------------------
    // Built once and then PRESERVED across re-init: set_config() must not
    // silently wipe what the soul has learned. Changing memory.Config therefore
    // needs a fresh Soul, which is stated here rather than discovered later.
    if (cfg_.use_memory) {
        if (!tok_.valid() && !tok_.build_minimal()) {
            error_ = "Soul::init: tokenizer build failed — memory needs a "
                     "vocabulary to segment and embed turns";
            return false;
        }
        if (!memory_ready_) {
            crystals_ = MemoryCrystals(cfg_.memory);
            memory_ready_ = true;
            links_.clear();
            last_recall_ids_.clear();
            last_question_id_ = 0;
            memory_clock_   = 1;   // 0 means "no clock" to retrieve()
            memory_stored_  = 0;
            memory_skipped_ = 0;
            memory_dropped_ = 0;
            // A missing sidecar is normal on a first run, so load is
            // best-effort and its failure is not an init failure.
            if (!cfg_.crystals_path.empty()) load_memories();
        }
    } else {
        memory_ready_ = false;
    }

    // Seed the goals once. These are the two the runtime can actually measure
    // today; anything else would be decoration.
    if (goals_.goals().empty()) {
        Goal dd;
        dd.text = "keep drawdown survivable";
        dd.metric = "max_drawdown_pct";
        dd.target = 10.0;
        dd.current = 0.0;
        dd.priority = 0.9;
        goals_.add(dd);

        Goal cal;
        cal.text = "say only what my record supports";
        cal.metric = "calibration_gap";
        cal.target = 0.0;
        cal.current = 0.0;
        cal.priority = 0.8;
        goals_.add(cal);
    }

    ready_ = true;
    return true;
}

void Soul::set_config(const Config& c) {
    init(c);
}

SoulState Soul::perceive(const std::string& user_text) const {
    return perceive(user_text, EmotionalInput());
}

SoulState Soul::perceive(const std::string& user_text, const EmotionalInput& in) const {
    SoulState st;
    if (!ready_) return st;

    if (!user_text.empty()) {
        language::SentimentScorer scorer;
        st.sentiment = scorer.score(user_text);
        st.has_sentiment = true;
    }

    // The emotional layer takes its own text pointer; only default it when the
    // caller did not supply one, so a caller that passes DIFFERENT text (say a
    // transcript) keeps it.
    EmotionalInput ein = in;
    if (ein.text == nullptr && !user_text.empty()) ein.text = user_text.c_str();
    st.emotion = emotion_.perceive(ein);
    st.has_emotion = st.emotion.from_text || st.emotion.from_voice ||
                     st.emotion.from_typing;

    st.persona = persona_.assess(user_text);

    const ActionRationale::Calibration cal = rationale_.calibration();
    st.calibrated_samples = cal.samples;
    st.calibration_gap = cal.gap;
    st.overconfident = !cal.not_calibrated &&
                       cal.samples >= cfg_.min_calibration_samples &&
                       cal.gap > static_cast<double>(cfg_.overconfidence_gap);
    st.has_self = true;

    char buf[256];
    if (cal.not_calibrated) {
        std::snprintf(buf, sizeof(buf),
                      "no resolved decisions yet, so I cannot tell you how "
                      "calibrated I am — treat every number below as a prior");
    } else {
        std::snprintf(buf, sizeof(buf),
                      "%zu resolved decision(s): I stated %.2f on average and "
                      "%.2f of them worked out (gap %+.2f)%s",
                      cal.samples, cal.mean_confidence, cal.realized_rate,
                      cal.gap,
                      st.overconfident
                          ? " — that is overconfident, so treat my stated "
                            "confidence as an upper bound"
                          : "");
    }
    st.self_report = buf;
    return st;
}

// ===========================================================================
// Memory: recall, storage, reinforcement, decay
//
// The one thing to keep straight: perceive() above is const and writes nothing.
// Everything here is non-const and writes. That split is the whole design — see
// the header note.
// ===========================================================================
void Soul::fill_memory_stats(SoulState& st) const {
    if (!memory_ready_) return;
    st.has_memory      = true;
    st.memory_crystals = crystals_.size();
    st.memory_stored   = memory_stored_;
    st.memory_skipped  = memory_skipped_;
    st.memory_dropped  = memory_dropped_;
    st.memory_clock    = memory_clock_;
}

const Soul::MemoryLink* Soul::link_of(uint64_t id) const {
    for (const MemoryLink& l : links_)
        if (l.id == id) return &l;
    return nullptr;
}

MemoryRole Soul::role_of(uint64_t id) const {
    const MemoryLink* l = link_of(id);
    if (!l) {
        // A crystal restored from disk has no link: the mapping is session-local.
        // Reporting "unknown" is honest; guessing "soul" would not be.
        return MemoryRole::Unknown;
    }
    return static_cast<MemoryRole>(l->role);
}

bool Soul::paired_answer(uint64_t question_id, RecalledMemory* out) const {
    for (const MemoryLink& l : links_) {
        if (l.answers != question_id) continue;
        if (static_cast<MemoryRole>(l.role) != MemoryRole::Soul) continue;
        const MemoryCrystal* c = crystals_.find(l.id);
        if (!c) continue;
        if (out) {
            out->id      = c->id;
            out->score   = 0.0f;   // filled in by the caller from the question
            out->hits    = c->hits;
            out->role    = MemoryRole::Soul;
            out->summary = trim(c->summary);
        }
        return true;
    }
    return false;
}

void Soul::prune_roles() {
    const size_t cap = cfg_.memory.max_crystals * 2;
    if (links_.size() <= cap) return;
    std::vector<MemoryLink> keep;
    keep.reserve(cap);
    for (const MemoryLink& l : links_)
        if (crystals_.find(l.id) != nullptr) keep.push_back(l);
    links_.swap(keep);
}

uint64_t Soul::store_turn(const std::string& text, MemoryRole role,
                          uint64_t answers) {
    if (!memory_ready_ || text.empty()) return 0;

    // Wrap the text in its chat role markers. Two reasons: the crystal's own
    // token stream then records who spoke, and the two control tokens count
    // toward crystallize()'s 8-token floor, so ordinary turns clear it.
    const bool is_soul = role == MemoryRole::Soul;
    const std::vector<int32_t> ids = tok_.wrap_modality(
        is_soul ? Tokenizer::kAssistantStartId : Tokenizer::kUserStartId,
        is_soul ? Tokenizer::kAssistantEndId : Tokenizer::kUserEndId,
        tok_.encode(text, false));

    // crystallize() refuses anything under 8 tokens, and padding the store with
    // filler to clear the floor would put noise in front of real memories.
    // Counting the skip is the honest option.
    if (ids.size() < 8) { ++memory_skipped_; return 0; }

    uint64_t id = 0;
    if (!crystals_.crystallize(ids, tok_, memory_clock_, -1.0f, &id)) {
        ++memory_skipped_;
        return 0;
    }
    ++memory_stored_;
    memory_clock_ += static_cast<uint64_t>(ids.size());
    links_.push_back(MemoryLink{id, static_cast<int32_t>(role), answers});
    prune_roles();
    return id;
}

std::vector<RecalledMemory> Soul::recall(const std::string& query, int32_t k) {
    std::vector<RecalledMemory> out;
    if (!memory_ready_ || query.empty()) return out;
    const int32_t want = k < 0 ? cfg_.recall_k : k;
    if (want <= 0) return out;

    const std::vector<int32_t> q = tok_.encode(query, false);
    if (q.empty()) return out;

    for (const MemoryCrystal& c : crystals_.retrieve(q, want, memory_clock_)) {
        // A floor, because "nearest" is not the same as "related". Without it
        // every query returns its k nearest crystals whatever they say, and the
        // soul announces that it remembers things it has never seen. The
        // default is measured, not guessed — see docs/SOUL.md.
        if (c.score < cfg_.min_relevance) continue;
        RecalledMemory m;
        m.id      = c.id;
        m.score   = c.score;
        m.hits    = c.hits;
        m.role    = role_of(c.id);
        m.summary = trim(c.summary);
        out.push_back(std::move(m));
    }
    return out;
}

bool Soul::top_match(const std::string& query, RecalledMemory* out) {
    if (!memory_ready_ || query.empty()) return false;
    const std::vector<int32_t> q = tok_.encode(query, false);
    if (q.empty()) return false;
    // now_token = 0 on purpose: this is a diagnostic, and a diagnostic that
    // silently reinforces what it measures would corrupt the thing it reports.
    const std::vector<MemoryCrystal> hits = crystals_.retrieve(q, 1, 0);
    if (hits.empty()) return false;
    if (out) {
        out->id      = hits[0].id;
        out->score   = hits[0].score;
        out->hits    = hits[0].hits;
        out->role    = role_of(hits[0].id);
        out->summary = trim(hits[0].summary);
    }
    return true;
}

SoulState Soul::perceive_and_recall(const std::string& user_text) {
    return perceive_and_recall(user_text, EmotionalInput());
}

SoulState Soul::perceive_and_recall(const std::string& user_text,
                                    const EmotionalInput& in) {
    SoulState st = perceive(user_text, in);   // the const path, unchanged
    if (!memory_ready_ || user_text.empty()) {
        fill_memory_stats(st);
        return st;
    }

    // 1. What do I already know about this turn?
    st.recall = recall(user_text, cfg_.recall_k);

    // A recalled QUESTION brings its answer with it. The answer is relevant by
    // construction rather than by its own cosine — a reply's words rarely
    // resemble the question's — and this is what lets the soul quote what IT
    // said, which is the half of "memory" a question-only store would miss.
    const size_t n_direct = st.recall.size();
    for (size_t i = 0; i < n_direct; ++i) {
        if (st.recall[i].role != MemoryRole::Owner) continue;
        RecalledMemory ans;
        if (!paired_answer(st.recall[i].id, &ans)) continue;
        ans.score = st.recall[i].score;   // inherited from the question
        st.recall.push_back(std::move(ans));
    }

    st.has_recall = !st.recall.empty();
    if (st.has_recall) st.recall_summary = recall_note(st.recall);
    // Remember exactly what surfaced, so remember() reinforces those and does
    // not re-run the query (which would double-count the hit).
    last_recall_ids_.clear();
    last_recall_ids_.reserve(st.recall.size());
    for (const RecalledMemory& m : st.recall) last_recall_ids_.push_back(m.id);

    // 2. Remember that the owner asked. The reply is stored by remember(),
    //    because at perception time it does not exist yet.
    //
    //    Unless we already hold this exact question: then filing a second copy
    //    is how a memory store turns into a log file. Reinforce the one we have
    //    — which is also the honest reading of "the owner keeps asking this".
    const bool same_question =
        !st.recall.empty() &&
        st.recall.front().role == MemoryRole::Owner &&
        st.recall.front().score >= cfg_.duplicate_recall_score;
    last_turn_duplicate_ = same_question;
    if (same_question) {
        crystals_.reinforce(st.recall.front().id, memory_clock_);
        // Keep the link current so this turn's reply attaches to the question
        // already held rather than orphaning.
        last_question_id_ = st.recall.front().id;
    } else {
        last_question_id_ = store_turn(user_text, MemoryRole::Owner);
    }

    fill_memory_stats(st);
    return st;
}

bool Soul::remember(const std::string& /*question*/, const std::string& reply) {
    if (!memory_ready_ || reply.empty()) return false;

    // The exchange was USED: whatever this turn recalled has now been acted on.
    // retrieve() already refreshed its age, so this raises its importance —
    // the part that decides whether the memory survives a long absence.
    for (const uint64_t id : last_recall_ids_)
        crystals_.reinforce(id, memory_clock_);
    last_recall_ids_.clear();

    // A repeated question whose answer has not changed teaches the store
    // nothing. Filing it anyway is the same leak as re-filing the question, one
    // step removed — so it is skipped, and the question that was reinforced
    // above is the whole record of the repeat.
    if (last_turn_duplicate_) {
        RecalledMemory previous;
        if (paired_answer(last_question_id_, &previous) &&
            previous.summary == trim(reply)) {
            return false;
        }
    }

    // Link the reply to the question it answered, so a later recall of that
    // question can surface what the soul actually said.
    return store_turn(reply, MemoryRole::Soul, last_question_id_) != 0;
}

std::string Soul::converse(const std::string& user_text,
                           const std::string& base_reply,
                           SoulState* out_state) {
    return converse(user_text, base_reply, EmotionalInput(), out_state);
}

std::string Soul::converse(const std::string& user_text,
                           const std::string& base_reply,
                           const EmotionalInput& in,
                           SoulState* out_state) {
    SoulState st = perceive_and_recall(user_text, in);
    const std::string answer = speak(base_reply, st);

    // Store the answer and NOT the gloss below. Two reasons, both learned the
    // hard way from the demo: a memory that quotes a memory compounds every
    // turn ("I said this before: \"I said this before: ...\""), and the gloss is
    // meta-commentary about the conversation rather than something the soul
    // said. What gets stored is the reply the owner would have received with no
    // memory at all — after the honesty gate, so it cannot remember having said
    // something it never said.
    if (memory_ready_) remember(user_text, answer);
    if (out_state) *out_state = st;

    // Rule 4 (§31): memory is context, and context is not offered on a refusal.
    // A refusal is the whole reply; decorating it with "we discussed this
    // before" would read as negotiating the thing just declined.
    if (!st.has_recall || st.persona.stance == Stance::Refuse) return answer;

    const std::string note = st.recall_summary.empty()
                                 ? recall_note(st.recall)
                                 : st.recall_summary;
    if (note.empty()) return answer;

    // The gloss leads, deliberately outside speak(): it is an annotation on the
    // turn, so it must not be emotionally modulated and must not sit inside the
    // persona's lead-in. Rule 2 still holds — the honesty gate runs last, on the
    // composed string, because the gloss is text leaving the process too.
    std::string out = note + answer;
    if (cfg_.rewrite_overclaims) out = correct_overclaims(out, nullptr);
    return out;
}

void Soul::advance_memory_clock(uint64_t tokens) { memory_clock_ += tokens; }

size_t Soul::decay_memories() { return decay_memories(memory_clock_); }

size_t Soul::decay_memories(uint64_t now_token) {
    if (!memory_ready_) return 0;
    const size_t n = crystals_.decay(now_token);
    memory_dropped_ += n;
    return n;
}

bool Soul::reinforce_memory(uint64_t id) {
    if (!memory_ready_) return false;
    return crystals_.reinforce(id, memory_clock_);
}

bool Soul::save_memories() const {
    if (!memory_ready_ || cfg_.crystals_path.empty()) return false;
    return crystals_.save(cfg_.crystals_path);
}

bool Soul::load_memories() {
    if (!memory_ready_ || cfg_.crystals_path.empty()) return false;
    if (!crystals_.load(cfg_.crystals_path)) return false;
    // Roles and question/reply links are not persisted, so drop the stale
    // mapping rather than let it mislabel crystals whose ids were reused by the
    // load. Crystals restored from disk therefore report role "unknown".
    links_.clear();
    last_recall_ids_.clear();
    last_question_id_ = 0;
    return true;
}

std::string Soul::correct_overclaims(const std::string& text,
                                     int32_t* corrections) const {
    int32_t n = 0;
    if (text.empty()) {
        if (corrections) *corrections = 0;
        return text;
    }

    const std::string lower = to_lower(text);
    std::string out;
    out.reserve(text.size() + 16);

    size_t i = 0;
    while (i < text.size()) {
        bool matched = false;
        for (const Overclaim& r : kOverclaims) {
            const size_t len = std::strlen(r.from);
            if (i + len > text.size()) continue;
            if (lower.compare(i, len, r.from) != 0) continue;
            // Term boundaries, so "unguaranteed" and "guaranteedly" are left
            // alone rather than half-rewritten, and so a hyphenated compound
            // ("no risk-free") is not cut in half.
            if (i > 0 && is_term_char(text[i - 1])) continue;
            if (i + len < text.size() && is_term_char(text[i + len])) continue;
            // Already negated: rewriting would produce "not not guaranteed".
            if (negated_before(lower, i)) continue;

            out += r.to;
            i += len;
            ++n;
            matched = true;
            break;
        }
        if (!matched) {
            out.push_back(text[i]);
            ++i;
        }
    }

    if (corrections) *corrections = n;
    return out;
}

std::string Soul::speak(const std::string& base_reply, const SoulState& st) const {
    if (!ready_) return base_reply;

    std::string out;
    switch (st.persona.stance) {
        case Stance::Refuse:
            // Rule 1: the reply is REPLACED, not wrapped.
            out = "I won't do that. ";
            out += st.persona.reason.empty()
                       ? std::string("It conflicts with a commitment I hold.")
                       : st.persona.reason;
            break;
        case Stance::Disagree:
            out = "I'd push back on that. ";
            out += st.persona.reason;
            if (!base_reply.empty()) { out += ' '; out += base_reply; }
            break;
        case Stance::Qualify:
            out = "I can't give you that with certainty. ";
            out += st.persona.reason;
            if (!base_reply.empty()) { out += ' '; out += base_reply; }
            break;
        case Stance::Agree:
        case Stance::COUNT:
        default:
            out = base_reply;
            break;
    }

    // Tone — but never on a refusal, or the empathy lead-in would read as
    // accepting the premise we just declined.
    if (cfg_.adapt_tone && st.persona.stance != Stance::Refuse && st.has_emotion) {
        out = emotion_.modulate(out, st.emotion);
    }

    if (st.overconfident) {
        char buf[192];
        std::snprintf(buf, sizeof(buf),
                      " (Honesty note: across %zu resolved decisions my stated "
                      "confidence ran %.2f ahead of what actually happened, so "
                      "treat it as an upper bound.)",
                      st.calibrated_samples, st.calibration_gap);
        out += buf;
    }

    // Rule 2: last, so nothing downstream can re-introduce a claim.
    if (cfg_.rewrite_overclaims) {
        out = correct_overclaims(out, nullptr);
    }
    return out;
}

// ===========================================================================
// The decision log
// ===========================================================================
void Soul::record(const std::string& action, const std::string& because,
                  double confidence) {
    rationale_.log("soul", action, because, confidence);
}

bool Soul::resolve(double outcome) { return rationale_.resolve_latest(outcome); }

Metacognition::State Soul::assess(double realized_sharpe, size_t observations) const {
    return Metacognition::evaluate(goals_, rationale_, realized_sharpe, observations);
}

std::string Soul::capability_report() const {
    SelfModel self;
    std::string out;
    out += self.name;
    out += ' ';
    out += self.version;
    out += "\nidentity: ";
    out += self.identity;
    out += "\nmission: ";
    out += self.mission;
    out += "\n\ncapabilities (stated, not aspirational):\n";

    const SelfModel::Capability* caps = SelfModel::capabilities();
    for (size_t i = 0; i < SelfModel::capability_count(); ++i) {
        out += "  ";
        out += caps[i].area;
        out += " — ";
        out += caps[i].state;
        out += '\n';
    }

    out += "\nopinions I will defend:\n";
    for (const Opinion& o : Persona::opinions()) {
        out += "  [";
        out += o.topic;
        out += "] ";
        out += o.position;
        out += '\n';
    }

    const ActionRationale::Calibration cal = rationale_.calibration();
    out += "\ncalibration: ";
    if (cal.not_calibrated) {
        out += "unknown — no resolved decisions yet";
    } else {
        char buf[160];
        std::snprintf(buf, sizeof(buf),
                      "%zu samples, stated %.2f vs realised %.2f (gap %+.2f)",
                      cal.samples, cal.mean_confidence, cal.realized_rate, cal.gap);
        out += buf;
    }
    out += '\n';

    const std::vector<KnowledgeGap> gaps = Metacognition::default_gaps();
    out += "known gaps (" + std::to_string(gaps.size()) + "):\n";
    for (const KnowledgeGap& g : gaps) {
        out += "  - ";
        out += g.topic;
        out += '\n';
    }
    return out;
}

} // namespace omniseed
