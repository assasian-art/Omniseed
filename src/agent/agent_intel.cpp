// =============================================================================
//  OmniSeed — agent_intel.cpp
//  Intent, dialogue state, confidence, thinking mode, error recovery,
//  feedback loop, interrupt system, knowledge graph, progressive disclosure.
// =============================================================================
#include "omniseed/agent/agent_intel.h"
#include "omniseed/core/platform.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <sstream>

namespace omniseed {

namespace {

std::string lower_copy(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s)
        out.push_back(static_cast<char>(
            std::tolower(static_cast<unsigned char>(c))));
    return out;
}

bool contains_any(const std::string& low,
                  std::initializer_list<const char*> words) {
    for (const char* w : words)
        if (low.find(w) != std::string::npos) return true;
    return false;
}

} // namespace

// ===========================================================================
// IntentClassifier
// ===========================================================================
const char* intent_name(Intent i) {
    switch (i) {
        case Intent::Informational: return "informational";
        case Intent::Directive:     return "directive";
        case Intent::Exploratory:   return "exploratory";
        case Intent::Social:        return "social";
        case Intent::TaskRepeat:    return "repeat";
        case Intent::Unknown:       break;
    }
    return "unknown";
}

Intent IntentClassifier::classify(const std::string& text) {
    const std::string low = lower_copy(text);
    if (low.empty()) return Intent::Unknown;

    if (contains_any(low, {"thank", "hello", "hi ", "hey", "good morning",
                           "good evening"}) || low == "hi")
        return Intent::Social;
    if (contains_any(low, {"again", "redo", "one more time", "repeat"}))
        return Intent::TaskRepeat;
    if (contains_any(low, {"why", "how come", "compare", "difference",
                           "what if", "reason"}))
        return Intent::Exploratory;
    if (contains_any(low, {"run", "call", "execute", "do ", "make", "create",
                           "send", "open", "start", "stop", "set", "convert",
                           "calculate", "compute", "find", "list"}))
        return Intent::Directive;
    if (contains_any(low, {"what", "who", "when", "where", "explain",
                           "describe", "tell me", "is there", "define"}))
        return Intent::Informational;
    return Intent::Unknown;
}

// ===========================================================================
// DialogueStateTracker
// ===========================================================================
void DialogueStateTracker::reset() {
    topic_.clear();
    last_user_.clear();
    turns_ = 0;
    slots_.clear();
}

void DialogueStateTracker::update(const std::string& user_text,
                                  const std::string& reply) {
    (void)reply;
    ++turns_;
    last_user_ = user_text;

    // Topic: longest lowercase content word (>= 5 chars) — cheap, stable.
    const std::string low = lower_copy(user_text);
    std::string best;
    std::string cur;
    for (char c : low) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            cur.push_back(c);
        } else {
            if (cur.size() > best.size() && cur.size() >= 5) best = cur;
            cur.clear();
        }
    }
    if (cur.size() > best.size() && cur.size() >= 5) best = cur;
    if (!best.empty()) topic_ = best;

    // Slot patterns: "my <key> is <value>" / "<key> = <value>".
    auto scan_slot = [&](const std::string& pat, bool eq_style) {
        (void)eq_style;
        size_t pos = 0;
        while ((pos = low.find(pat, pos)) != std::string::npos) {
            // Key: word(s) strictly BEFORE the pattern (pos is inclusive of
            // the pattern's leading space, so search below pos-1).
            size_t kstart = 0;
            if (pos > 0) {
                const size_t probe = low.find_last_of(" \n", pos - 1);
                kstart = (probe == std::string::npos) ? 0 : probe + 1;
            }
            const std::string key = low.substr(kstart, pos - kstart);
            const size_t vstart = pos + pat.size();
            const size_t vend = low.find_first_of(".,\n", vstart);
            const std::string value = low.substr(
                vstart, vend == std::string::npos ? std::string::npos
                                                  : vend - vstart);
            if (!key.empty() && !value.empty() && slots_.size() < 32)
                slots_[key] = value;
            pos = vstart;
        }
    };
    scan_slot(" is ", false);
    scan_slot(" = ", true);
    scan_slot(" equals ", false);
}

// ===========================================================================
// ConfidenceScorer
// ===========================================================================
namespace {

double jaccard_words(const std::string& a, const std::string& b) {
    auto words_of = [](const std::string& s) {
        std::unordered_map<std::string, int32_t> w;
        std::string cur;
        for (char c : s) {
            if (std::isalpha(static_cast<unsigned char>(c))) {
                cur.push_back(static_cast<char>(
                    std::tolower(static_cast<unsigned char>(c))));
            } else if (!cur.empty()) {
                if (cur.size() >= 4) ++w[cur];
                cur.clear();
            }
        }
        if (cur.size() >= 4) ++w[cur];
        return w;
    };
    const auto wa = words_of(a), wb = words_of(b);
    if (wa.empty() || wb.empty()) return 0.0;
    size_t inter = 0;
    for (const auto& [k, v] : wa) {
        (void)v;
        if (wb.count(k)) ++inter;
    }
    const size_t uni = wa.size() + wb.size() - inter;
    return uni ? static_cast<double>(inter) / uni : 0.0;
}

} // namespace

ConfidenceReport ConfidenceScorer::score(const std::string& reply,
                                         const std::string& context,
                                         float state_saturation) {
    ConfidenceReport r;
    r.state_saturation = std::min(1.0f, std::max(0.0f, state_saturation));
    r.lexical_overlap = static_cast<float>(
        jaccard_words(reply, context));

    // Hallucination risk: reply asserts facts with no lexical support in the
    // context AND the recurrent state is saturated (drifting) or the reply
    // is very short/templated.
    r.low_evidence = r.lexical_overlap < 0.06f &&
                     reply.size() > 24 &&
                     contains_any(lower_copy(reply),
                                  {"is ", "are ", "was ", "the ", "has "});
    r.hallucination_risk = std::min(1.0f,
        0.55f * (r.low_evidence ? 1.0f : 0.0f) +
        0.45f * r.state_saturation);

    r.confidence = std::min(1.0f, std::max(0.0f,
        0.35f + 0.40f * r.lexical_overlap +
        0.25f * (1.0f - r.state_saturation) -
        0.30f * r.hallucination_risk));
    return r;
}

// ===========================================================================
// ThinkingMode
// ===========================================================================
bool ThinkingMode::should_think(const std::string& input) const {
    switch (mode_) {
        case Mode::Forced: return true;
        case Mode::Off:    return false;
        case Mode::Auto:   break;
    }
    // Auto: think when the input looks hard.
    const std::string low = lower_copy(input);
    const bool longish = input.size() > 120;
    const bool multi_part = input.find(" and ") != std::string::npos ||
                            input.find(", then ") != std::string::npos ||
                            input.find('?') != std::string::npos &&
                            input.rfind('?') != input.find('?');
    const bool reasoning = contains_any(low, {"why", "explain", "plan",
                                              "compare", "step", "how"});
    const bool mathy = input.find_first_of("0123456789") != std::string::npos &&
                       input.find_first_of("+-*/") != std::string::npos;
    return longish || multi_part || reasoning || mathy;
}

std::string ThinkingMode::think_prefix(const std::string& input,
                                       int32_t budget_tokens) const {
    (void)input;
    if (!should_think(input) || budget_tokens <= 0) return "";
    // Compressible scaffold: the model fills a bounded trace.
    std::string s = "<|think|>";
    s += " goal? facts? steps? verify?";
    s += "</|think|>";
    return s;
}

// ===========================================================================
// ErrorRecovery
// ===========================================================================
ErrorRecovery::Policy ErrorRecovery::decide(const std::string& error_text,
                                            Attempt& a) {
    ++a.tries;
    const std::string low = lower_copy(error_text);

    // Permanent failures: abort immediately.
    if (contains_any(low, {"not found", "invalid", "schema", "unknown tool",
                           "permission", "denied"}))
        return Policy::Abort;

    // Timeout-ish: exponential backoff after 2 tries, degrade after 4.
    if (contains_any(low, {"timeout", "busy", "unavailable", "network"})) {
        if (a.tries <= 2) return Policy::Retry;
        if (a.tries <= 4) return Policy::Backoff;
        return Policy::Degrade;
    }
    // Generic transient: retry once, then degrade.
    if (a.tries <= 1) return Policy::Retry;
    if (a.tries <= 3) return Policy::Backoff;
    return Policy::Degrade;
}

std::string ErrorRecovery::recover_message(Policy p, const std::string& tool) {
    switch (p) {
        case Policy::Retry:   return "retrying " + tool;
        case Policy::Backoff: return "backing off, then retrying " + tool;
        case Policy::Degrade: return "continuing without " + tool +
                                     " (degraded mode)";
        case Policy::Abort:   return "cannot complete: " + tool + " failed";
    }
    return "";
}

// ===========================================================================
// UserFeedbackLoop
// ===========================================================================
UserFeedbackLoop::Verdict UserFeedbackLoop::parse(const std::string& text) {
    const std::string low = lower_copy(text);
    if (contains_any(low, {"wrong", "incorrect", "no that", "not right",
                           "mistake", "bad answer", "actually"}))
        return Verdict::Correct;      // user is correcting us
    if (contains_any(low, {"yes", "correct", "right", "good", "thanks",
                           "exactly", "perfect"}))
        return Verdict::Confirm;
    if (low == "no" || contains_any(low, {"nope", "stop"}))
        return Verdict::Reject;
    return Verdict::None;
}

void UserFeedbackLoop::record(Verdict v, const std::string& task_key) {
    auto& slot = feedback_[task_key];
    if (v == Verdict::Confirm) ++slot.first;
    if (v == Verdict::Correct || v == Verdict::Reject) ++slot.second;
}

void UserFeedbackLoop::record_implicit(const std::string& task_key,
                                       bool engaged) {
    auto& slot = feedback_[task_key];
    if (engaged) ++slot.first; else ++slot.second;
}

double UserFeedbackLoop::trust(const std::string& task_key) const {
    auto it = feedback_.find(task_key);
    if (it == feedback_.end()) return 0.5;
    const auto [c, w] = it->second;
    const uint32_t total = c + w;
    if (total == 0) return 0.5;
    // Laplace-smoothed confirmation rate, mapped to [0,1].
    return (static_cast<double>(c) + 1.0) / (static_cast<double>(total) + 2.0);
}

// ===========================================================================
// SensoryInterruptSystem
// ===========================================================================
const char* interrupt_event_name(SensoryInterruptSystem::Event e) {
    switch (e) {
        case SensoryInterruptSystem::Event::WakeWord:     return "wake_word";
        case SensoryInterruptSystem::Event::SoundAlarm:   return "sound_alarm";
        case SensoryInterruptSystem::Event::SoundDoorbell:return "sound_doorbell";
        case SensoryInterruptSystem::Event::SoundSpeech:  return "sound_speech";
        case SensoryInterruptSystem::Event::Motion:       return "motion";
        case SensoryInterruptSystem::Event::Gesture:      return "gesture";
        case SensoryInterruptSystem::Event::EmotionShift: return "emotion_shift";
        case SensoryInterruptSystem::Event::None:         break;
    }
    return "none";
}

void SensoryInterruptSystem::raise(Event e, int32_t priority,
                                   const std::string& detail) {
    if (e == Event::None) return;
    // Coalesce: same event pending -> refresh priority only.
    for (Interrupt& i : queue_) {
        if (i.event == e) {
            i.priority = std::max(i.priority, priority);
            i.at_ms = platform::now_ms();
            return;
        }
    }
    Interrupt iv;
    iv.event = e;
    iv.priority = priority;
    iv.detail = detail;
    iv.at_ms = platform::now_ms();
    queue_.push_back(iv);
    if (queue_.size() > 16) queue_.pop_front();   // bounded
}

bool SensoryInterruptSystem::poll(Interrupt& out) {
    if (queue_.empty()) return false;
    auto best = queue_.begin();
    for (auto it = queue_.begin(); it != queue_.end(); ++it)
        if (it->priority > best->priority) best = it;
    out = *best;
    queue_.erase(best);
    return true;
}

bool SensoryInterruptSystem::should_preempt(int32_t current_priority) const {
    for (const Interrupt& i : queue_)
        if (i.priority > current_priority) return true;
    return false;
}

// ===========================================================================
// KnowledgeGraph
// ===========================================================================
void KnowledgeGraph::reindex(size_t triple_index) {
    const KgTriple& t = triples_[triple_index];
    index_[t.subject].push_back(triple_index);
    if (t.object != t.subject) index_[t.object].push_back(triple_index);
}

size_t KnowledgeGraph::ingest_text(const std::string& text,
                                   uint64_t at_token) {
    static const char* kRelations[] = {"is", "has", "likes", "at", "wants",
                                       "needs", "knows"};
    size_t added = 0;

    // Tokenize once into lowercase words (positions in the original string
    // are not needed; triple order follows text order). Then scan every
    // adjacent (subject, relation, object) triple with the relation matched
    // exactly. Bounded word lengths keep the pass O(tokens).
    std::vector<std::string> words;
    words.reserve(32);
    std::string cur;
    auto flush = [&]() {
        if (!cur.empty()) {
            if (cur.size() <= 24) words.push_back(cur);
            cur.clear();
        }
    };
    for (const char ch : text) {
        if (std::isalnum(static_cast<unsigned char>(ch)) || ch == '\'') {
            cur.push_back(static_cast<char>(
                std::tolower(static_cast<unsigned char>(ch))));
        } else {
            flush();
        }
    }
    flush();

    auto is_relation = [](const std::string& w) -> const char* {
        for (const char* rel : kRelations)
            if (w == rel) return rel;
        return nullptr;
    };

    for (size_t i = 1; i + 1 < words.size(); ++i) {
        const char* rel = is_relation(words[i]);
        if (!rel) continue;
        // Subject: the word before (2-word subjects: skip — keeps recall
        // precise and the implementation trivial).
        const std::string& subj = words[i - 1];
        const std::string& obj = words[i + 1];
        if (subj == obj) continue;
        if (triples_.size() >= cfg_.max_triples) break;
        KgTriple t;
        t.subject = subj;
        t.relation = rel;
        t.object = obj;
        t.at_token = at_token;
        const size_t idx = triples_.size();
        triples_.push_back(std::move(t));
        reindex(idx);   // index AFTER the push (reindex reads triples_[idx])
        ++added;
    }
    return added;
}

std::vector<KgTriple> KnowledgeGraph::query(const std::string& entity) const {
    std::vector<KgTriple> out;
    auto it = index_.find(entity);
    if (it == index_.end()) return out;
    for (size_t i : it->second)
        if (i < triples_.size()) out.push_back(triples_[i]);
    return out;
}

std::vector<std::string> KnowledgeGraph::facts_for_prompt(
        const std::string& entity, int32_t max_lines) const {
    std::vector<std::string> out;
    for (const KgTriple& t : query(entity)) {
        if (static_cast<int32_t>(out.size()) >= max_lines) break;
        out.push_back("[kg] " + t.subject + " " + t.relation + " " +
                      t.object);
    }
    return out;
}

bool KnowledgeGraph::save(const std::string& path) const {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const uint32_t magic = 0x54474B4F;   // OKGT
    const uint32_t version = 1;
    const uint32_t n = static_cast<uint32_t>(triples_.size());
    std::fwrite(&magic, 4, 1, f);
    std::fwrite(&version, 4, 1, f);
    std::fwrite(&n, 4, 1, f);
    for (const KgTriple& t : triples_) {
        auto wstr = [&](const std::string& s) {
            const uint32_t len = static_cast<uint32_t>(s.size());
            std::fwrite(&len, 4, 1, f);
            std::fwrite(s.data(), 1, len, f);
        };
        wstr(t.subject); wstr(t.relation); wstr(t.object);
        std::fwrite(&t.at_token, 8, 1, f);
        std::fwrite(&t.confidence, 4, 1, f);
    }
    std::fclose(f);
    return true;
}

bool KnowledgeGraph::load(const std::string& path) {
    platform::MappedFile mf;
    if (!mf.open(path)) return false;
    const uint8_t* p = mf.bytes();
    size_t cur = 0;
    if (mf.size() < 12 || std::memcmp(p, "OKGT", 4) != 0) return false;
    cur += 4;
    if (*reinterpret_cast<const uint32_t*>(p + cur) != 1) return false;
    cur += 4;
    const uint32_t n = *reinterpret_cast<const uint32_t*>(p + cur);
    cur += 4;
    triples_.clear();
    index_.clear();
    for (uint32_t i = 0; i < n; ++i) {
        auto rstr = [&](std::string& s) -> bool {
            if (cur + 4 > mf.size()) return false;
            const uint32_t len = *reinterpret_cast<const uint32_t*>(p + cur);
            cur += 4;
            if (cur + len > mf.size()) return false;
            s.assign(reinterpret_cast<const char*>(p + cur), len);
            cur += len;
            return true;
        };
        KgTriple t;
        if (!rstr(t.subject) || !rstr(t.relation) || !rstr(t.object))
            return false;
        if (cur + 12 > mf.size()) return false;
        std::memcpy(&t.at_token, p + cur, 8); cur += 8;
        std::memcpy(&t.confidence, p + cur, 4); cur += 4;
        const size_t idx = triples_.size();
        triples_.push_back(std::move(t));
        reindex(idx);   // index AFTER the push (reindex reads triples_[idx])
    }
    return true;
}

// ===========================================================================
// ProgressiveDisclosure
// ===========================================================================
std::string ProgressiveDisclosure::summarize(const std::string& text,
                                             int32_t max_sentences) {
    if (max_sentences <= 0) return "";
    std::string out;
    int32_t stops = 0;
    for (size_t i = 0; i < text.size(); ++i) {
        out.push_back(text[i]);
        if (text[i] == '.' || text[i] == '!' || text[i] == '?') {
            ++stops;
            if (stops >= max_sentences) break;
        }
    }
    if (stops < max_sentences && out.size() < text.size()) {
        // No sentence ends: hard-trim at a word boundary.
        if (out.size() > 160) {
            out = out.substr(0, 160);
            const auto sp = out.rfind(' ');
            if (sp != std::string::npos) out = out.substr(0, sp);
            out += "...";
        }
    }
    return out;
}

bool ProgressiveDisclosure::wants_detail(const std::string& user_text) {
    const std::string low = lower_copy(user_text);
    return contains_any(low, {"more", "detail", "elaborate", "expand",
                              "continue", "go on", "why"});
}

} // namespace omniseed
