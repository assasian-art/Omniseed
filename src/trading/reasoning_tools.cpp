// =============================================================================
//  OmniSeed — trading/reasoning_tools.cpp
//  Tool implementations. Everything is input-capped and side-effect-free:
//  the "code interpreter" deliberately evaluates only whitelisted math.
// =============================================================================
#include "omniseed/trading/reasoning_tools.h"
#include "omniseed/trading/finance.h"
#include "omniseed/runtime/cloud_bridge.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <map>
#include <sstream>

namespace omniseed {
namespace trading {

// ===========================================================================
// URL helpers
// ===========================================================================
bool is_fetchable_url(const std::string& url) {
    if (url.rfind("http://", 0) == 0) return url.size() > 7;
    if (url.rfind("https://", 0) == 0) return url.size() > 8;
    return false;
}

// ===========================================================================
// HTML -> text
// ===========================================================================
namespace {

std::string decode_entities(const std::string& s) {
    static const std::pair<const char*, const char*> table[] = {
        {"&amp;", "&"}, {"&lt;", "<"}, {"&gt;", ">"}, {"&quot;", "\""},
        {"&apos;", "'"}, {"&#39;", "'"}, {"&nbsp;", " "}, {"&mdash;", "-"},
        {"&ndash;", "-"}, {"&rsquo;", "'"}, {"&lsquo;", "'"},
        {"&ldquo;", "\""}, {"&rdquo;", "\""}, {"&hellip;", "..."},
    };
    std::string out = s;
    for (const auto& kv : table) {
        size_t p;
        while ((p = out.find(kv.first)) != std::string::npos)
            out.replace(p, std::string(kv.first).size(), kv.second);
    }
    return out;
}

} // namespace

std::string html_to_text(const std::string& html) {
    // Drop script/style blocks with their content.
    std::string s = html;
    for (const char* tag : {"script", "style"}) {
        const std::string open = std::string("<") + tag;
        const std::string close = std::string("</") + tag + ">";
        size_t p;
        while ((p = s.find(open)) != std::string::npos) {
            const size_t e = s.find(close, p);
            if (e == std::string::npos) { s.erase(p); break; }
            s.erase(p, e + close.size() - p);
        }
    }
    // Strip tags.
    std::string out;
    out.reserve(s.size());
    bool in_tag = false;
    for (const char c : s) {
        if (c == '<') { in_tag = true; out += ' '; }        // tag = separator
        else if (c == '>') in_tag = false;
        else if (!in_tag) out += c;
    }
    // Collapse whitespace.
    std::string text;
    text.reserve(out.size());
    bool prev_space = true;                                  // trim head
    for (const char c : out) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!prev_space) { text += ' '; prev_space = true; }
        } else {
            text += c;
            prev_space = false;
        }
    }
    while (!text.empty() && text.back() == ' ') text.pop_back();
    return decode_entities(text);
}

// ===========================================================================
// DuckDuckGo HTML result parsing
// ===========================================================================
size_t parse_ddg_results(const std::string& html,
                         std::vector<std::string>& out, size_t max) {
    out.clear();
    // Result blocks look like:
    //   <a rel="nofollow" class="result__a" href="URL">TITLE</a>
    //   <a class="result__snippet" ...>SNIPPET</a>
    size_t pos = 0;
    while (out.size() < max) {
        const size_t link = html.find("class=\"result__a\"", pos);
        if (link == std::string::npos) break;
        const size_t href = html.find("href=\"", link);
        if (href == std::string::npos) break;
        const size_t url_end = html.find('"', href + 6);
        const size_t title_open = html.find('>', href + 6);
        const size_t title_close = html.find("</a>", title_open);
        if (url_end == std::string::npos || title_open == std::string::npos ||
            title_close == std::string::npos) break;
        std::string url = html.substr(href + 6, url_end - href - 6);
        std::string title = html_to_text(
            html.substr(title_open + 1, title_close - title_open - 1));
        pos = title_close;

        // DDG wraps URLs in //duckduckgo.com/l/?uddg=<encoded>; unwrap.
        const std::string wrap = "duckduckgo.com/l/?uddg=";
        size_t wp = url.find(wrap);
        if (wp != std::string::npos) {
            std::string enc = url.substr(wp + wrap.size());
            const size_t amp = enc.find('&');
            if (amp != std::string::npos) enc.resize(amp);
            // Percent-decode.
            std::string dec;
            for (size_t i = 0; i < enc.size(); ++i) {
                if (enc[i] == '%' && i + 2 < enc.size()) {
                    const char hex[3] = {enc[i + 1], enc[i + 2], 0};
                    dec += static_cast<char>(std::strtol(hex, nullptr, 16));
                    i += 2;
                } else if (enc[i] == '+') {
                    dec += ' ';
                } else {
                    dec += enc[i];
                }
            }
            url = dec;
        }
        if (!is_fetchable_url(url)) continue;

        // Snippet (optional).
        std::string snippet;
        const size_t sn = html.find("class=\"result__snippet\"", pos);
        if (sn != std::string::npos) {
            const size_t so = html.find('>', sn);
            const size_t sc = html.find("</a>", so);
            if (so != std::string::npos && sc != std::string::npos) {
                snippet = html_to_text(html.substr(so + 1, sc - so - 1));
            }
        }
        out.push_back(title + " | " + url +
                      (snippet.empty() ? "" : " | " + snippet));
    }
    return out.size();
}

// ===========================================================================
// Tool registration
// ===========================================================================
namespace {

std::string json_escape(const std::string& s) {
    std::string out;
    for (const unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += static_cast<char>(c);
        }
    }
    return out;
}

std::string json_field(const std::string& body, const std::string& key) {
    const auto kpos = body.find("\"" + key + "\"");
    if (kpos == std::string::npos) return "";
    const auto cpos = body.find(':', kpos);
    const auto q1 = body.find('"', cpos);
    const auto q2 = body.find('"', q1 + 1);
    if (cpos == std::string::npos || q1 == std::string::npos ||
        q2 == std::string::npos)
        return "";
    return body.substr(q1 + 1, q2 - q1 - 1);
}

// Production fetcher over the CloudBridge shim.
bool default_fetcher(const std::string& url, std::string& response,
                     std::string& err) {
    int status = 0;
    return CloudBridge::get_text(url, 15, response, status, err);
}

} // namespace

void register_reasoning_tools(ToolRegistry& reg, WebFetcher fetcher) {
    WebFetcher fetch = fetcher ? std::move(fetcher) : default_fetcher;

    // ---- web_search --------------------------------------------------------
    {
        Tool t;
        t.name = "web_search";
        t.description = "Search the web (DuckDuckGo) for current information. "
                        "Args: {\"query\": \"...\"}";
        t.params = {{"query", "string", "search query", true}};
        t.fn = [fetch](const std::string& args, bool& ok) -> std::string {
            ok = false;
            const std::string query = json_field(args, "query");
            if (query.empty()) return "{\"error\":\"missing query\"}";
            std::string url =
                "https://html.duckduckgo.com/html/?q=" + query;
            std::string body;
            std::string err;
            if (!fetch(url, body, err)) {
                return "{\"error\":\"fetch failed: " + json_escape(err) +
                       "\"}";
            }
            std::vector<std::string> results;
            parse_ddg_results(body, results, 6);
            std::string json = "{\"results\":[";
            for (size_t i = 0; i < results.size(); ++i) {
                if (i) json += ",";
                json += "\"" + json_escape(results[i]) + "\"";
            }
            json += "]}";
            ok = !results.empty();
            return json;
        };
        reg.add(std::move(t));
    }

    // ---- finance_calc (calculator extension) --------------------------------
    {
        Tool t;
        t.name = "finance_calc";
        t.description = "Evaluate financial math: + - * / ^ ( ), min max abs "
                        "sqrt pow, npv(rate, cf...), irr(cf...), "
                        "bscall/bsput(S,K,r,sigma,T). Sandbox: no variables, "
                        "no I/O.";
        t.params = {{"expression", "string", "expression to evaluate", true}};
        t.fn = [](const std::string& args, bool& ok) -> std::string {
            const std::string expr = json_field(args, "expression");
            const auto r = FinanceInterpreter::evaluate(expr);
            ok = r.ok;
            if (!r.ok)
                return "{\"error\":\"" + json_escape(r.error) + "\"}";
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%.10g", r.value);
            return "{\"value\":\"" + std::string(buf) + "\"}";
        };
        reg.add(std::move(t));
    }

    // ---- code_interpreter (sandboxed math surface, honestly named) ----------
    {
        Tool t;
        t.name = "code_interpreter";
        t.description = "Sandboxed data-analysis evaluator: same grammar as "
                        "finance_calc (no filesystem, network, or arbitrary "
                        "code — by design).";
        t.params = {{"expression", "string", "expression to evaluate", true}};
        t.fn = [](const std::string& args, bool& ok) -> std::string {
            const auto r = FinanceInterpreter::evaluate(
                json_field(args, "expression"));
            ok = r.ok;
            if (!r.ok)
                return "{\"error\":\"" + json_escape(r.error) + "\"}";
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%.10g", r.value);
            return "{\"value\":\"" + std::string(buf) + "\"}";
        };
        reg.add(std::move(t));
    }

    // ---- document_reader ----------------------------------------------------
    {
        Tool t;
        t.name = "document_reader";
        t.description = "Fetch a URL (SEC filing page, earnings report HTML) "
                        "and return readable text. Args: {\"url\": \"...\"}";
        t.params = {{"url", "string", "http(s) URL", true}};
        t.fn = [fetch](const std::string& args, bool& ok) -> std::string {
            const std::string url = json_field(args, "url");
            if (!is_fetchable_url(url))
                { ok = false; return "{\"error\":\"bad or non-http url\"}"; }
            std::string body;
            std::string err;
            if (!fetch(url, body, err)) {
                ok = false;
                return "{\"error\":\"" + json_escape(err) + "\"}";
            }
            std::string text = html_to_text(body);
            if (text.size() > 20000) text.resize(20000);   // bounded prompt
            ok = true;
            return "{\"text\":\"" + json_escape(text) + "\"}";
        };
        reg.add(std::move(t));
    }
}

} // namespace trading
} // namespace omniseed
