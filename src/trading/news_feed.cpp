// =============================================================================
//  OmniSeed — trading/news_feed.cpp
//  RSS parsing + financial sentiment + ticker extraction + breaking alerts.
//  Zero dependencies: hand-rolled tolerant XML scan (RSS 2.0 / Atom entries).
// =============================================================================
#include "omniseed/trading/news_feed.h"
#include "omniseed/core/platform.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

namespace omniseed {
namespace trading {

// ===========================================================================
// Financial sentiment lexicon (small, curated; extends cleanly)
// ===========================================================================
const char* const kPositiveTerms[] = {
    "beats", "beat", "surge", "surges", "soars", "rally", "rallies", "gains",
    "rises", "jumps", "upbeat", "record", "growth", "profit", "upgraded",
    "upgrade", "strong", "outperform", "buy rating", "breakout", "recovery",
    "expands", "boom", "positive", "approval", "wins", "bullish", "climbs",
};
const size_t kPositiveCount = sizeof(kPositiveTerms) / sizeof(kPositiveTerms[0]);

const char* const kNegativeTerms[] = {
    "misses", "miss", "plunge", "plunges", "crash", "crashes", "falls", "drops",
    "slump", "slumps", "collapse", "collapses", "recession", "layoffs", "cuts",
    "downgrade", "downgraded", "fraud", "probe", "lawsuit", "sinks", "weak",
    "warning", "warns", "loss", "losses", "bankruptcy", "bearish", "fears",
    "selloff", "sell-off", "tumble", "tumbles", "crisis", "deficit", "slows",
};
const size_t kNegativeCount = sizeof(kNegativeTerms) / sizeof(kNegativeTerms[0]);

const char* const kBreakingTerms[] = {
    "crash", "crashes", "plunge", "plunges", "panic", "halt", "halted",
    "breaking", "urgent", "fraud", "bankruptcy", "default", "circuit breaker",
    "collapse", "collapses",
};
const size_t kBreakingCount = sizeof(kBreakingTerms) / sizeof(kBreakingTerms[0]);

// ===========================================================================
// Small text helpers
// ===========================================================================
namespace {

std::string lower_copy(const std::string& s) {
    std::string out(s);
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

// Decodes the handful of entities RSS commonly carries + strips tags.
std::string xml_decode(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '&') {
            if (s.compare(i, 5, "&amp;") == 0) { out += '&'; i += 4; }
            else if (s.compare(i, 4, "&lt;") == 0) { out += '<'; i += 3; }
            else if (s.compare(i, 4, "&gt;") == 0) { out += '>'; i += 3; }
            else if (s.compare(i, 6, "&quot;") == 0) { out += '"'; i += 5; }
            else if (s.compare(i, 6, "&apos;") == 0) { out += '\''; i += 5; }
            else if (s.compare(i, 4, "&#39;") == 0) { out += '\''; i += 3; }
            else out += '&';
        } else {
            out += s[i];
        }
    }
    return out;
}

std::string strip_tags(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    bool in_tag = false;
    for (const char c : s) {
        if (c == '<') in_tag = true;
        else if (c == '>') in_tag = false;
        else if (!in_tag) out += c;
    }
    return xml_decode(out);
}

// RFC-822 pubDate ("Tue, 10 Sep 2026 08:00:00 GMT") -> unix seconds.
int64_t parse_rfc822(const std::string& s) {
    static const char* months[12] = {
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec"};
    // Day-of-week, DD, Mon, YYYY, HH:MM:SS
    int day = 0, year = 0, hh = 0, mm = 0, ss = 0;
    char mon[16] = {0};
    if (std::sscanf(s.c_str(), "%*s %d %15s %d %d:%d:%d",
                    &day, mon, &year, &hh, &mm, &ss) < 4) {
        // ISO 8601 fallback: YYYY-MM-DDTHH:MM:SS
        std::tm tmv{};
        if (std::sscanf(s.c_str(), "%d-%d-%d", &year, &mm, &day) == 3) {
            std::memset(mon, 0, sizeof(mon));
            mm = 0;   // reuse; month parsed below
            int mo = 0;
            std::sscanf(s.c_str(), "%d-%d-%d", &year, &mo, &day);
            if (mo < 1 || mo > 12) return 0;
            int64_t days = 0;
            // days_from_civil inline (light version)
            int y2 = year - (mo <= 2);
            const int era = (y2 >= 0 ? y2 : y2 - 399) / 400;
            const unsigned yoe = static_cast<unsigned>(y2 - era * 400);
            const unsigned doy = (153 * (mo + (mo > 2 ? -3 : 9)) + 2) / 5 + day - 1;
            const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
            days = static_cast<int64_t>(era) * 146097 +
                   static_cast<int64_t>(doe) - 719468;
            return days * 86400;
        }
        return 0;
    }
    int mo = -1;
    std::string mon_low = mon;
    std::transform(mon_low.begin(), mon_low.end(), mon_low.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    for (int i = 0; i < 12; ++i)
        if (mon_low.compare(months[i]) == 0) { mo = i + 1; break; }
    if (mo < 1) return 0;
    int y2 = year - (mo <= 2);
    const int era = (y2 >= 0 ? y2 : y2 - 399) / 400;
    const unsigned yoe = static_cast<unsigned>(y2 - era * 400);
    const unsigned doy = (153 * (mo + (mo > 2 ? -3 : 9)) + 2) / 5 + day - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    const int64_t days = static_cast<int64_t>(era) * 146097 +
                         static_cast<int64_t>(doe) - 719468;
    return days * 86400 + hh * 3600 + mm * 60 + ss;
}

// Returns the text content of the first <tag>...</tag> after `from`.
bool extract_tag(const std::string& xml, size_t from, const char* tag,
                 std::string& out, size_t& next) {
    const std::string open = std::string("<") + tag;
    size_t p = xml.find(open, from);
    while (p != std::string::npos) {
        // Avoid matching <titleX> when looking for <title>.
        const char after = p + open.size() < xml.size() ? xml[p + open.size()] : ' ';
        if (after == '>' || after == ' ') {
            const size_t gt = xml.find('>', p);
            if (gt == std::string::npos) return false;
            const size_t end = xml.find(std::string("</") + tag + ">", gt);
            if (end == std::string::npos) return false;
            out = xml_decode(xml.substr(gt + 1, end - gt - 1));
            next = end;
            return true;
        }
        p = xml.find(open, p + 1);
    }
    return false;
}

} // namespace

// ===========================================================================
// Sentiment + entities
// ===========================================================================
double NewsFeed::score_sentiment(const std::string& text) {
    const std::string low = " " + lower_copy(text) + " ";
    int score = 0;
    auto count_term = [&low](const char* term) {
        // Boundary-aware count: require a non-letter (or string edge) on both
        // sides so "miss" never matches inside "mission".
        const std::string t = term;
        int hits = 0;
        size_t p = low.find(t);
        while (p != std::string::npos) {
            const bool left_ok = p == 0 ||
                !std::isalpha(static_cast<unsigned char>(low[p - 1]));
            const size_t right = p + t.size();
            const bool right_ok = right >= low.size() ||
                !std::isalpha(static_cast<unsigned char>(low[right]));
            if (left_ok && right_ok) ++hits;
            p = low.find(t, p + 1);
        }
        return hits;
    };
    for (size_t i = 0; i < kPositiveCount; ++i) score += count_term(kPositiveTerms[i]);
    for (size_t i = 0; i < kNegativeCount; ++i) score -= count_term(kNegativeTerms[i]);
    // Normalize by text length so long articles don't saturate: a headline
    // (~10 words) with one hit lands ~0.3; strong multi-hit ~1.0.
    const double words = std::max(4.0,
        static_cast<double>(std::count(low.begin(), low.end(), ' ')));
    const double norm = static_cast<double>(score) /
                        std::max(2.0, words / 4.0);
    return std::max(-1.0, std::min(1.0, norm));
}

std::vector<std::string> NewsFeed::extract_tickers(const std::string& text) {
    std::vector<std::string> out;
    std::set<std::string> uniq;
    size_t i = 0;
    while (i < text.size()) {
        if (text[i] == '$' && i + 1 < text.size() &&
            std::isupper(static_cast<unsigned char>(text[i + 1]))) {
            size_t j = i + 1;
            while (j < text.size() &&
                   std::isupper(static_cast<unsigned char>(text[j])))
                ++j;
            const std::string sym = text.substr(i + 1, j - i - 1);
            if (sym.size() >= 1 && sym.size() <= 5 && uniq.insert(sym).second)
                out.push_back(sym);
            i = j;
            continue;
        }
        // Bare symbol: uppercase run bounded by non-letters.
        if (std::isupper(static_cast<unsigned char>(text[i])) &&
            (i == 0 || !std::isalpha(static_cast<unsigned char>(text[i - 1])))) {
            size_t j = i;
            while (j < text.size() &&
                   std::isupper(static_cast<unsigned char>(text[j])))
                ++j;
            // Word must end at a non-letter and not be a common English word.
            if (j < text.size() &&
                !std::isalpha(static_cast<unsigned char>(text[j])) &&
                j - i >= 2 && j - i <= 5) {
                const std::string w = text.substr(i, j - i);
                static const char* kStop[] = {
                    "AAP", "THE", "AND", "FOR", "CEO", "SEC", "FED", "GDP",
                    "CPI", "EPS", "IPO", "YOY", "Q1", "Q2", "Q3", "Q4",
                    "USA", "FOMC", "NYSE", "IT", "US", "UK", "NYSEARCA"};
                bool stop = false;
                for (const char* s : kStop)
                    if (w == s) { stop = true; break; }
                if (!stop && uniq.insert(w).second) out.push_back(w);
            }
            i = j;
            continue;
        }
        ++i;
    }
    return out;
}

// ===========================================================================
// Ingest + classify
// ===========================================================================
void NewsFeed::recompute(NewsItem& item) const {
    const std::string text = item.title + " " + item.summary;
    item.sentiment = score_sentiment(text);
    item.tickers = extract_tickers(text);
}

size_t NewsFeed::ingest_rss(const std::string& xml) {
    size_t added = 0, pos = 0;
    for (;;) {
        // Accept both RSS <item> and Atom <entry>.
        size_t item_pos = xml.find("<item", pos);
        const size_t entry_pos = xml.find("<entry", pos);
        bool is_atom = false;
        if (entry_pos != std::string::npos &&
            (item_pos == std::string::npos || entry_pos < item_pos)) {
            item_pos = entry_pos;
            is_atom = true;
        }
        if (item_pos == std::string::npos) break;
        const size_t next = xml.find(is_atom ? "</entry>" : "</item>", item_pos);
        const size_t end = next == std::string::npos ? xml.size() : next;
        std::string body = xml.substr(item_pos, end - item_pos);

        NewsItem it;
        std::string tmp;
        size_t skip = 0;
        if (extract_tag(body, 0, is_atom ? "id" : "guid", tmp, skip))
            it.guid = tmp;
        if (it.guid.empty() &&
            extract_tag(body, 0, is_atom ? "title" : "title", tmp, skip))
            it.guid = tmp;
        if (extract_tag(body, 0, "title", tmp, skip)) it.title = strip_tags(tmp);
        if (extract_tag(body, 0, is_atom ? "summary" : "description", tmp, skip))
            it.summary = strip_tags(tmp);
        if (extract_tag(body, 0, is_atom ? "published" : "pubDate", tmp, skip))
            it.published = parse_rfc822(strip_tags(tmp));
        if (it.published == 0)
            it.published = static_cast<int64_t>(platform::now_ms() / 1000.0);
        recompute(it);

        if (!it.title.empty() && it.guid.size() < 200 &&
            seen_.insert(it.guid).second) {
            items_.push_front(std::move(it));      // newest first
            ++added;
            while (items_.size() > cfg_.max_items) items_.pop_back();
        }
        pos = end > item_pos ? end + 1 : item_pos + 5;
    }
    return added;
}

bool NewsFeed::is_breaking(const NewsItem& item) const {
    const double now = platform::now_ms() / 1000.0;
    const double age = now - static_cast<double>(item.published);
    if (age < 0.0 || age > cfg_.breaking_window) return false;
    const std::string low = " " + lower_copy(item.title) + " ";
    auto has_term = [&low](const char* term) {
        const std::string t = term;
        size_t p = low.find(t);
        while (p != std::string::npos) {
            const bool left_ok = p == 0 ||
                !std::isalpha(static_cast<unsigned char>(low[p - 1]));
            const size_t right = p + t.size();
            const bool right_ok = right >= low.size() ||
                !std::isalpha(static_cast<unsigned char>(low[right]));
            if (left_ok && right_ok) return true;
            p = low.find(t, p + 1);
        }
        return false;
    };
    for (size_t i = 0; i < kBreakingCount; ++i)
        if (has_term(kBreakingTerms[i])) return true;
    return std::fabs(item.sentiment) >= cfg_.strong_sentiment;
}

std::vector<NewsFeed::Alert> NewsFeed::pending_alerts() {
    std::vector<Alert> out;
    for (const NewsItem& it : items_) {
        if (alerted_.count(it.guid)) continue;
        if (is_breaking(it)) {
            Alert a;
            a.priority = cfg_.breaking_priority;
            a.detail = "news:breaking[" +
                       (it.tickers.empty() ? std::string("market")
                                           : it.tickers.front()) + "] " +
                       it.title;
            out.push_back(a);
            alerted_.insert(it.guid);
        }
    }
    return out;
}

double NewsFeed::aggregate_sentiment(double max_age_seconds,
                                     const std::string& ticker) const {
    const double now = platform::now_ms() / 1000.0;
    double wsum = 0.0, ssum = 0.0;
    for (const NewsItem& it : items_) {
        const double age = now - static_cast<double>(it.published);
        if (age < 0.0 || age > max_age_seconds) continue;
        if (!ticker.empty() &&
            std::find(it.tickers.begin(), it.tickers.end(), ticker) ==
            it.tickers.end())
            continue;
        // Recency weight: 1.0 at age 0 -> 0.5 at max age.
        const double w = 1.0 - 0.5 * (age / max_age_seconds);
        ssum += it.sentiment * w;
        wsum += w;
    }
    return wsum > 0.0 ? ssum / wsum : 0.0;
}

} // namespace trading
} // namespace omniseed
