// =============================================================================
//  OmniSeed — emotional.cpp
//  Emotional Resonance Layer implementation.
//
//  Voice: 20 ms frames -> energy mean/var, pitch proxy via zero-crossing
//  rate on voiced frames, speech rate (voiced ratio), pause ratio. These
//  map onto the classic valence/arousal circumplex used in SER literature:
//  high arousal + positive valence -> happy/excited; low arousal + negative
//  valence -> sad; high arousal + negative -> angry/anxious.
//
//  Text: small affective lexicon (word lists), exclamation/question
//  intensity, ALL-CAPS ratio, emoticon/token patterns.
//
//  Typing: inter-key interval variance (bursty = higher arousal).
//
//  Channels are fused with confidence weighting; below min_confidence the
//  state stays Neutral so responses are not modulated on weak evidence.
// =============================================================================
#include "omniseed/runtime/emotional.h"
#include "omniseed/core/platform.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <unordered_map>

namespace omniseed {

const char* emotion_name(Emotion e) {
    switch (e) {
        case Emotion::Happy:   return "happy";
        case Emotion::Excited: return "excited";
        case Emotion::Sad:     return "sad";
        case Emotion::Angry:   return "angry";
        case Emotion::Anxious: return "anxious";
        case Emotion::Calm:    return "calm";
        case Emotion::Neutral: break;
    }
    return "neutral";
}

namespace {

// ---------------------------------------------------------------------------
// Affective lexicon (compact, deterministic)
// ---------------------------------------------------------------------------
const char* kPositive[] = {
    "good", "great", "love", "happy", "thanks", "thank", "awesome", "nice",
    "perfect", "wonderful", "excellent", "amazing", "glad", "cool", "yes",
    "win", "best", "fun", "enjoy",
};
const char* kNegative[] = {
    "bad", "hate", "angry", "sad", "terrible", "awful", "worst", "wrong",
    "broken", "fail", "failed", "error", "stupid", "annoying", "frustrat",
    "no", "never", "cant", "can't", "wont", "won't", "problem", "bug",
    "crash", "slow", "waste", "confused", "help",
};
const char* kAnxious[] = {
    "worried", "nervous", "anxious", "afraid", "scared", "urgent", "asap",
    "hurry", "panic", "stress", "quickly", "emergency", "deadline",
};
const char* kExcited[] = {
    "wow", "amazing", "incredible", "let's", "go", "awesome", "yes",
    "finally", "cant wait", "can't wait", "excited", "!!!",
};

struct LexHits { int32_t pos = 0, neg = 0, anx = 0, exc = 0; };

LexHits scan_lexicon(const std::string& text) {
    LexHits hits;
    std::string lower;
    lower.reserve(text.size());
    for (char c : text) {
        lower.push_back(static_cast<char>(
            std::tolower(static_cast<unsigned char>(c))));
    }
    auto count = [&](const char** list, size_t n) {
        int32_t c = 0;
        for (size_t i = 0; i < n; ++i) {
            const std::string w = list[i];
            size_t pos = 0;
            while ((pos = lower.find(w, pos)) != std::string::npos) {
                ++c;
                pos += w.size();
            }
        }
        return c;
    };
    hits.pos = count(kPositive, sizeof(kPositive) / sizeof(kPositive[0]));
    hits.neg = count(kNegative, sizeof(kNegative) / sizeof(kNegative[0]));
    hits.anx = count(kAnxious,  sizeof(kAnxious)  / sizeof(kAnxious[0]));
    hits.exc = count(kExcited,  sizeof(kExcited)  / sizeof(kExcited[0]));
    return hits;
}

// Simple tokenizer stats: exclamations, questions, caps ratio, length.
void scan_style(const std::string& text, float& excl, float& quest,
                float& caps_ratio, float& length_words) {
    int32_t ex = 0, qu = 0, letters = 0, caps = 0;
    bool word_start = true, in_caps_word = false;
    int32_t caps_word_len = 0, words = 0;
    for (char c : text) {
        if (c == '!') ++ex;
        if (c == '?') ++qu;
        if (std::isalpha(static_cast<unsigned char>(c))) {
            ++letters;
            if (std::isupper(static_cast<unsigned char>(c))) {
                ++caps;
                if (word_start) in_caps_word = true;
                ++caps_word_len;
            } else {
                in_caps_word = false;
                caps_word_len = 0;
            }
        } else if (std::isspace(static_cast<unsigned char>(c))) {
            if (!word_start) ++words;
            word_start = true;
            if (in_caps_word && caps_word_len >= 3) caps_word_len = 0;
        }
        if (!std::isalpha(static_cast<unsigned char>(c))) word_start = false;
    }
    if (!text.empty() && !std::isspace(static_cast<unsigned char>(text.back())))
        ++words;
    excl = static_cast<float>(ex);
    quest = static_cast<float>(qu);
    caps_ratio = letters > 0 ? static_cast<float>(caps) / letters : 0.0f;
    length_words = static_cast<float>(words);
}

} // namespace

// ===========================================================================
// Voice channel
// ===========================================================================
bool EmotionalResonance::voice_features(const float* pcm, size_t len,
                                        int32_t sr, float* out) {
    if (!pcm || len < sr / 4) return false;   // need >= 250 ms

    const int32_t frame = sr / 50;            // 20 ms
    const size_t n_frames = len / static_cast<size_t>(frame);
    if (n_frames < 8) return false;

    double e_sum = 0.0, e2_sum = 0.0, zcr_voiced = 0.0;
    int32_t voiced = 0, silent = 0;
    std::vector<double> energies(n_frames);

    for (size_t f = 0; f < n_frames; ++f) {
        const float* fr = pcm + f * frame;
        double e = 0.0, zc = 0.0;
        for (int32_t t = 1; t < frame; ++t) {
            e += static_cast<double>(fr[t]) * fr[t];
            if ((fr[t - 1] < 0) != (fr[t] < 0)) ++zc;
        }
        energies[f] = e / frame;
        e_sum += energies[f];
        if (energies[f] > 0.002) {
            ++voiced;
            zcr_voiced += zc / frame;
        } else {
            ++silent;
        }
    }
    const double e_mean = e_sum / n_frames;
    for (double e : energies) e2_sum += (e - e_mean) * (e - e_mean);
    const double e_var = e2_sum / n_frames;

    if (voiced < 4) return false;             // silence / noise only

    // Pause ratio: silent frames between voiced segments.
    const double pause_ratio = static_cast<double>(silent) / n_frames;
    // Speech rate proxy: voiced fraction.
    const double speech_rate = static_cast<double>(voiced) / n_frames;
    // Pitch proxy: mean ZCR on voiced frames (higher ZCR ~ higher pitch).
    const double pitch_proxy = zcr_voiced / voiced;
    // Loudness: log mean energy.
    const double loudness = std::log1p(e_mean * 100.0);

    out[0] = static_cast<float>(loudness);
    out[1] = static_cast<float>(std::log1p(e_var * 1000.0));
    out[2] = static_cast<float>(pitch_proxy);
    out[3] = static_cast<float>(speech_rate);
    out[4] = static_cast<float>(pause_ratio);
    out[5] = static_cast<float>(n_frames * frame / static_cast<double>(sr)); // dur
    out[6] = static_cast<float>(e_mean);
    out[7] = 0.0f;
    return true;
}

EmotionalState EmotionalResonance::from_voice(const float* pcm, size_t len,
                                              int32_t sr) {
    EmotionalState st;
    float f[8];
    if (!voice_features(pcm, len, sr, f)) return st;

    const float loudness = f[0], energy_var = f[1], pitch = f[2];
    const float rate = f[3], pause = f[4];

    // Arousal: loud + variable + fast + little pause.
    const float arousal = std::min(1.0f,
        0.35f * std::min(1.0f, loudness / 3.0f) +
        0.25f * std::min(1.0f, energy_var / 3.0f) +
        0.25f * std::min(1.0f, rate / 0.9f) +
        0.15f * (1.0f - std::min(1.0f, pause * 2.0f)));

    // Valence: bright pitch + energetic (positive) vs. flat + paused (neg).
    const float valence = std::max(-1.0f, std::min(1.0f,
        2.2f * (pitch - 0.12f) + 0.8f * (rate - 0.55f)));

    st.arousal = arousal;
    st.valence = valence;
    st.confidence = std::min(1.0f, 0.4f + 0.3f * arousal);
    st.from_voice = true;

    if (arousal > 0.55f && valence > 0.25f)      st.emotion = Emotion::Excited;
    else if (arousal > 0.45f && valence > 0.05f) st.emotion = Emotion::Happy;
    else if (arousal > 0.50f && valence < -0.25f) st.emotion = Emotion::Angry;
    else if (arousal > 0.35f && valence < -0.05f &&
             valence > -0.35f)                    st.emotion = Emotion::Anxious;
    else if (valence < -0.20f)                    st.emotion = Emotion::Sad;
    else if (arousal < 0.20f)                     st.emotion = Emotion::Calm;
    else                                          st.emotion = Emotion::Neutral;
    return st;
}

// ===========================================================================
// Text channel
// ===========================================================================
EmotionalState EmotionalResonance::from_text(const char* text) {
    EmotionalState st;
    if (!text || !*text) return st;
    const std::string s(text);
    const LexHits hits = scan_lexicon(s);
    float excl = 0, quest = 0, caps = 0, words = 0;
    scan_style(s, excl, quest, caps, words);
    if (words <= 0) return st;

    const float net =
        (static_cast<float>(hits.pos) - static_cast<float>(hits.neg)) /
        std::max(1.0f, words / 4.0f);
    const float valence = std::max(-1.0f, std::min(1.0f, net));

    float arousal = 0.25f * std::min(1.0f, excl / 2.0f) +
                    0.20f * std::min(1.0f, caps * 4.0f) +
                    0.30f * std::min(1.0f, static_cast<float>(hits.exc) / 2.0f) +
                    0.35f * std::min(1.0f, static_cast<float>(hits.anx) / 2.0f);
    arousal = std::min(1.0f, arousal);

    st.valence = valence;
    st.arousal = arousal;
    st.confidence = std::min(1.0f,
        0.2f + 0.15f * (hits.pos + hits.neg + hits.anx + hits.exc) +
        0.1f * std::min(3.0f, excl));
    st.from_text = true;

    if (hits.anx > 0 && arousal > 0.30f)         st.emotion = Emotion::Anxious;
    else if (arousal > 0.50f && valence < -0.25f) st.emotion = Emotion::Angry;
    else if (arousal > 0.50f && valence > 0.30f)  st.emotion = Emotion::Excited;
    else if (valence > 0.25f)                     st.emotion = Emotion::Happy;
    else if (valence < -0.25f)                    st.emotion = Emotion::Sad;
    else if (arousal < 0.15f)                     st.emotion = Emotion::Calm;
    else                                          st.emotion = Emotion::Neutral;
    return st;
}

// ===========================================================================
// Typing channel
// ===========================================================================
EmotionalState EmotionalResonance::from_typing(const float* ms, size_t len) {
    EmotionalState st;
    if (!ms || len < 4) return st;

    double mean = 0.0;
    for (size_t i = 0; i < len; ++i) mean += ms[i];
    mean /= len;
    double var = 0.0;
    for (size_t i = 0; i < len; ++i) {
        const double d = ms[i] - mean;
        var += d * d;
    }
    var /= len;
    const double cv = std::sqrt(var) / (mean + 1e-6);  // coefficient of variation

    // Fast + erratic typing -> higher arousal. Slow + steady -> calm.
    st.arousal = std::min(1.0f, static_cast<float>(
        0.5 * std::min(1.0, cv) + 0.5 * std::min(1.0, 250.0 / (mean + 30.0))));
    st.confidence = std::min(0.6f, static_cast<float>(len) / 20.0f);
    st.from_typing = true;
    st.emotion = st.arousal > 0.6f ? Emotion::Anxious
               : st.arousal < 0.2f ? Emotion::Calm : Emotion::Neutral;
    return st;
}

// ===========================================================================
// Fusion
// ===========================================================================
EmotionalState EmotionalResonance::perceive(const EmotionalInput& in) const {
    EmotionalState out;

    EmotionalState v{}, t{}, k{};
    float w_v = 0.0f, w_t = 0.0f, w_k = 0.0f;

    if (in.pcm && in.pcm_len > 0) {
        v = from_voice(in.pcm, in.pcm_len, in.sample_rate);
        if (v.from_voice) w_v = v.confidence * 1.2f;   // voice is strongest
    }
    if (in.text && *in.text) {
        t = from_text(in.text);
        if (t.from_text) w_t = t.confidence;
    }
    if (in.keystroke_ms && in.keystroke_len > 0) {
        k = from_typing(in.keystroke_ms, in.keystroke_len);
        if (k.from_typing) w_k = k.confidence * 0.6f;
    }

    const float wsum = w_v + w_t + w_k;
    if (wsum <= 1e-6f) return out;             // all neutral

    out.valence = (v.valence * w_v + t.valence * w_t + k.valence * w_k) / wsum;
    out.arousal = (v.arousal * w_v + t.arousal * w_t + k.arousal * w_k) / wsum;
    out.confidence = std::min(1.0f, wsum / 1.5f);
    out.from_voice = v.from_voice;
    out.from_text = t.from_text;
    out.from_typing = k.from_typing;

    // Re-classify on the fused space. The gate only discards evidence that
    // is BOTH weak AND emotionally flat; strong valence/arousal always wins.
    const bool flat = std::fabs(out.valence) < 0.20f && out.arousal < 0.20f;
    if (out.confidence < cfg_.min_confidence && flat) {
        out.emotion = Emotion::Neutral;
        return out;
    }
    if (out.arousal > 0.55f && out.valence > 0.30f)       out.emotion = Emotion::Excited;
    else if (out.arousal > 0.45f && out.valence > 0.10f)  out.emotion = Emotion::Happy;
    else if (out.arousal > 0.50f && out.valence < -0.30f) out.emotion = Emotion::Angry;
    else if (out.arousal > 0.30f && out.valence < 0.10f)  out.emotion = Emotion::Anxious;
    else if (out.valence < -0.20f)                        out.emotion = Emotion::Sad;
    else if (out.arousal < 0.20f)                         out.emotion = Emotion::Calm;
    else                                                  out.emotion = Emotion::Neutral;
    return out;
}

// ===========================================================================
// Response modulation
// ===========================================================================
std::string EmotionalResonance::modulate(const std::string& reply,
                                         const EmotionalState& st) const {
    if (!cfg_.adapt_responses || st.emotion == Emotion::Neutral) return reply;

    std::string out;
    switch (st.emotion) {
        case Emotion::Happy:
        case Emotion::Excited:
            out = reply;   // match the energy: no damping, no lead-in needed
            break;
        case Emotion::Sad:
            out = "I hear you — ";
            out += reply;
            out += " (I'm here if you want to talk it through.)";
            break;
        case Emotion::Angry:
            out = "Understood, let me make this right. ";
            out += reply;
            break;
        case Emotion::Anxious:
            out = "No rush — let's take it step by step. ";
            out += reply;
            break;
        case Emotion::Calm:
            out = reply;   // calm stays calm
            break;
        default:
            out = reply;
            break;
    }

    // Agitated users get SHORTER replies (trim run-on answers).
    if ((st.emotion == Emotion::Angry || st.emotion == Emotion::Anxious) &&
        out.size() > 220) {
        // Keep the first ~2 sentences.
        int32_t stops = 0;
        size_t cut = out.size();
        for (size_t i = 0; i < out.size(); ++i) {
            if (out[i] == '.' || out[i] == '!' || out[i] == '?') {
                ++stops;
                if (stops == 2) { cut = i + 1; break; }
            }
        }
        if (cut < out.size()) out = out.substr(0, cut);
    }
    return out;
}

std::string EmotionalResonance::token_fragment(const EmotionalState& st) const {
    if (st.emotion == Emotion::Neutral) return "";
    return std::string("[emotion:") + emotion_name(st.emotion) + "] ";
}

} // namespace omniseed
