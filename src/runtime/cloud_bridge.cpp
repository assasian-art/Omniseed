// =============================================================================
//  OmniSeed — runtime/cloud_bridge.cpp
//  Minimal HTTP/1.1 client over the dual-platform socket pair (the same
//  include pattern as the vendored server loop). Plain http:// only — see
//  the header for the honest TLS scope and proxy story.
// =============================================================================
#include "omniseed/runtime/cloud_bridge.h"
#include "omniseed/core/platform.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>

#if defined(_WIN32)
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #pragma comment(lib, "ws2_32.lib")
    using Socket_t = SOCKET;
    constexpr Socket_t kInvalidSocket = INVALID_SOCKET;
#else
    #include <arpa/inet.h>
    #include <netdb.h>
    #include <netinet/in.h>
    #include <sys/socket.h>
    #include <unistd.h>
    using Socket_t = int;
    constexpr Socket_t kInvalidSocket = -1;
#endif

namespace omniseed {

// ===========================================================================
// URL split + hostname resolution
// ===========================================================================
namespace {

struct Url {
    std::string scheme, host, port_str, path;
    bool valid = false;
    bool tls = false;
};

Url split_url(const std::string& url) {
    Url u;
    const size_t scheme_end = url.find("://");
    if (scheme_end == std::string::npos) return u;
    u.scheme = url.substr(0, scheme_end);
    u.tls = u.scheme == "https";
    std::string rest = url.substr(scheme_end + 3);
    const size_t path_slash = rest.find('/');
    u.host = path_slash == std::string::npos ? rest : rest.substr(0, path_slash);
    u.path = path_slash == std::string::npos ? "/" : rest.substr(path_slash);
    const size_t colon = u.host.rfind(':');
    if (colon != std::string::npos && u.host.find(']') == std::string::npos) {
        u.port_str = u.host.substr(colon + 1);
        u.host = u.host.substr(0, colon);
    }
    if (u.port_str.empty()) u.port_str = u.tls ? "443" : "80";
    u.valid = !u.host.empty();
    return u;
}

// Minimal JSON string escape (control chars, quote, backslash).
std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (const unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

// Extracts a string field from a JSON body with light escape handling.
// Works on nested objects by scanning from a start offset.
bool json_get_string(const std::string& body, const std::string& key,
                     size_t from, std::string& out) {
    const std::string needle = "\"" + key + "\"";
    size_t p = body.find(needle, from);
    while (p != std::string::npos) {
        size_t c = body.find(':', p + needle.size());
        if (c == std::string::npos) return false;
        size_t q = body.find('"', c + 1);
        if (q == std::string::npos) return false;
        std::string val;
        size_t i = q + 1;
        while (i < body.size()) {
            if (body[i] == '\\' && i + 1 < body.size()) {
                const char n = body[i + 1];
                switch (n) {
                    case 'n': val += '\n'; break;
                    case 't': val += '\t'; break;
                    case 'r': val += '\r'; break;
                    case '"': val += '"'; break;
                    case '\\': val += '\\'; break;
                    case '/': val += '/'; break;
                    case 'u': {                              // \uXXXX
                        if (i + 5 < body.size()) {
                            const std::string hex = body.substr(i + 2, 4);
                            const unsigned long cp =
                                std::strtoul(hex.c_str(), nullptr, 16);
                            // UTF-8 encode (BMP only; surrogates best-effort)
                            if (cp < 0x80) val += static_cast<char>(cp);
                            else if (cp < 0x800) {
                                val += static_cast<char>(0xC0 | (cp >> 6));
                                val += static_cast<char>(0x80 | (cp & 0x3F));
                            } else {
                                val += static_cast<char>(0xE0 | (cp >> 12));
                                val += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                                val += static_cast<char>(0x80 | (cp & 0x3F));
                            }
                        }
                        i += 4;
                        break;
                    }
                    default: val += n;
                }
                i += 2;
                continue;
            }
            if (body[i] == '"') break;
            val += body[i];
            ++i;
        }
        out = val;
        return true;
    }
    return false;
}

uint64_t parse_u64(const std::string& s) {
    return std::strtoull(s.c_str(), nullptr, 10);
}

// ---------------------------------------------------------------------------
// Blocking TCP connect + request/response exchange (plain HTTP/1.1).
// ---------------------------------------------------------------------------
bool http_exchange(const Url& u, const std::string& method,
                   const std::string& body, const std::string& content_type,
                   const std::map<std::string, std::string>& extra_headers,
                   int timeout_s, std::string& response, int& status,
                   std::string& err, int redirect_depth = 0) {
    status = 0;
    response.clear();
    if (!u.valid) { err = "bad url"; return false; }

    // ---- resolve + connect -------------------------------------------------
    struct addrinfo hints{};
    hints.ai_family = AF_INET;             // IPv4: predictable on every host
    hints.ai_socktype = SOCK_STREAM;
    struct addrinfo* res = nullptr;
    if (getaddrinfo(u.host.c_str(), u.port_str.c_str(), &hints, &res) != 0 ||
        !res) {
        err = "dns failed: " + u.host;
        return false;
    }
    Socket_t s = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (s == kInvalidSocket) {
        freeaddrinfo(res);
        err = "socket failed";
        return false;
    }
    if (connect(s, res->ai_addr, static_cast<int>(res->ai_addrlen)) != 0) {
        freeaddrinfo(res);
#ifdef _WIN32
        closesocket(s);
#else
        ::close(s);
#endif
        err = "connect failed: " + u.host + ":" + u.port_str;
        return false;
    }
    freeaddrinfo(res);

    // ---- send ---------------------------------------------------------------
    std::string req = method + " " + u.path + " HTTP/1.1\r\n";
    req += "Host: " + u.host + "\r\n";
    req += "User-Agent: omniseed/1.0\r\n";
    req += "Connection: close\r\n";
    if (!content_type.empty())
        req += "Content-Type: " + content_type + "\r\n";
    for (const auto& kv : extra_headers) req += kv.first + ": " + kv.second + "\r\n";
    if (!body.empty())
        req += "Content-Length: " + std::to_string(body.size()) + "\r\n";
    req += "\r\n";
    if (!body.empty()) req += body;

    size_t sent = 0;
    while (sent < req.size()) {
#ifdef _WIN32
        const int n = ::send(s, req.data() + sent,
                             static_cast<int>(req.size() - sent), 0);
#else
        const ssize_t n = ::send(s, req.data() + sent, req.size() - sent, 0);
#endif
        if (n <= 0) {
#ifdef _WIN32
            closesocket(s);
#else
            ::close(s);
#endif
            err = "send failed";
            return false;
        }
        sent += static_cast<size_t>(n);
    }

    // ---- receive until close (Connection: close keeps this simple) ----------
    std::string raw;
    char buf[16384];
    for (;;) {
#ifdef _WIN32
        const int n = ::recv(s, buf, sizeof(buf), 0);
#else
        const ssize_t n = ::recv(s, buf, sizeof(buf), 0);
#endif
        if (n <= 0) break;
        raw.append(buf, static_cast<size_t>(n));
        if (raw.size() > 64ull * 1024 * 1024) break;   // 64 MB sanity cap
    }
#ifdef _WIN32
    closesocket(s);
#else
    ::close(s);
#endif

    // ---- parse status + headers ---------------------------------------------
    const size_t line_end = raw.find("\r\n");
    if (line_end == std::string::npos) { err = "bad http response"; return false; }
    if (std::sscanf(raw.c_str(), "HTTP/%*d.%*d %d", &status) != 1) {
        err = "bad http status line";
        return false;
    }
    const size_t hdr_end = raw.find("\r\n\r\n");
    const std::string headers = raw.substr(0, hdr_end == std::string::npos ?
                                                line_end : hdr_end);
    std::string body_out = hdr_end == std::string::npos
                               ? "" : raw.substr(hdr_end + 4);

    // Headers (lowercased keys).
    std::map<std::string, std::string> hdr;
    {
        size_t start = line_end + 2;
        for (;;) {
            const size_t e = headers.find("\r\n", start);
            const std::string line = headers.substr(
                start, e == std::string::npos ? std::string::npos : e - start);
            const size_t c = line.find(':');
            if (c != std::string::npos) {
                std::string k = line.substr(0, c);
                std::transform(k.begin(), k.end(), k.begin(),
                               [](unsigned char ch) {
                                   return static_cast<char>(std::tolower(ch));
                               });
                size_t v = c + 1;
                while (v < line.size() && line[v] == ' ') ++v;
                hdr[k] = line.substr(v);
            }
            if (e == std::string::npos) break;
            start = e + 2;
        }
    }

    // Content-Length trimming (server may keep alive despite our header).
    auto it = hdr.find("content-length");
    if (it != hdr.end()) {
        const uint64_t cl = parse_u64(it->second);
        if (cl > 0 && cl <= body_out.size()) body_out.resize(static_cast<size_t>(cl));
    }

    // ---- redirects (up to 3) ------------------------------------------------
    if ((status == 301 || status == 302 || status == 307 || status == 308) &&
        redirect_depth < 3) {
        auto loc = hdr.find("location");
        if (loc != hdr.end()) {
            Url nu = split_url(loc->second);
            if (!nu.valid) { err = "bad redirect location"; return false; }
            return http_exchange(nu, method, body, content_type, extra_headers,
                                 timeout_s, response, status, err,
                                 redirect_depth + 1);
        }
    }

    response = std::move(body_out);
    return status >= 200 && status < 300;
}

} // namespace

// ===========================================================================
// CloudBridge
// ===========================================================================
CloudBridge::CloudBridge(const Config& cfg) : cfg_(cfg) {
    api_key_ = cfg_.api_key.empty()
        ? (std::getenv("OMNISEED_CLOUD_KEY")
               ? std::getenv("OMNISEED_CLOUD_KEY") : "")
        : cfg_.api_key;
    if (cfg_.proxy_url.empty()) {
        const char* p = std::getenv("OMNISEED_CLOUD_PROXY");
        if (p) cfg_.proxy_url = p;
    }
}

bool CloudBridge::post_json(const std::string& url, const std::string& body,
                            const std::string& content_type, int timeout_s,
                            std::string& response, int& status,
                            std::string& err) {
    Url u = split_url(url);
    if (!u.valid) { err = "bad url"; return false; }

    // TLS hosts: route through the user-provided local relay if configured.
    // Without it we refuse loudly — never a silent downgrade.
    if (u.tls) {
        const char* proxy = std::getenv("OMNISEED_CLOUD_PROXY");
        if (!proxy || !*proxy) {
            err = "https requires a TLS terminator (zero-dep runtime has no "
                  "TLS stack); set OMNISEED_CLOUD_PROXY=http://127.0.0.1:PORT "
                  "to a local relay (see docs/TRADING_GUIDE.md)";
            return false;
        }
        u = split_url(proxy);
        if (!u.valid || u.tls) {
            err = "OMNISEED_CLOUD_PROXY must be a plain http:// relay";
            return false;
        }
        // Standard forward-proxy absolute-URI form.
        Url orig = split_url(url);
        std::string abs_req = "http://" + orig.host +
                              (orig.port_str == "80" ? "" : ":" + orig.port_str) +
                              orig.path;
        (void)abs_req;
        // The generic exchange below sends `u.path`; rebase onto the proxy
        // with the absolute URI in the request line is handled by passing
        // the full URL via a path override.
        u.path = url;      // absolute-URI request line for forward proxies
    }

    return http_exchange(u, "POST", body, content_type, {}, timeout_s,
                         response, status, err);
}

bool CloudBridge::get_text(const std::string& url, int timeout_s,
                           std::string& response, int& status,
                           std::string& err) {
    Url u = split_url(url);
    if (!u.valid) { err = "bad url"; return false; }
    if (u.tls) {
        const char* proxy = std::getenv("OMNISEED_CLOUD_PROXY");
        if (!proxy || !*proxy) {
            err = "https requires OMNISEED_CLOUD_PROXY (no TLS in the "
                  "zero-dep runtime)";
            return false;
        }
        u = split_url(proxy);
        u.path = url;
    }
    return http_exchange(u, "GET", "", "", {}, timeout_s, response, status, err);
}

bool CloudBridge::ask(const std::string& prompt, std::string& reply,
                      std::string& err) {
    reply.clear();
    if (!available()) {
        err = "no cloud key (OMNISEED_CLOUD_KEY) — local mode";
        return false;
    }

    const std::string body =
        "{\"model\":\"" + json_escape(cfg_.model) +
        "\",\"max_tokens\":" + std::to_string(cfg_.max_tokens) +
        ",\"messages\":[{\"role\":\"user\",\"content\":\"" +
        json_escape(prompt) + "\"}]}";

    std::map<std::string, std::string> headers;
    if (cfg_.provider == Provider::Anthropic)
        headers["x-api-key"] = api_key_;
    else
        headers["Authorization"] = "Bearer " + api_key_;

    Url u = split_url(cfg_.base_url);
    if (u.tls) {
        const char* proxy = std::getenv("OMNISEED_CLOUD_PROXY");
        if (!proxy || !*proxy) {
            err = "https provider requires OMNISEED_CLOUD_PROXY (no TLS in "
                  "the zero-dep runtime); local AgentLoop remains fully "
                  "functional";
            return false;
        }
        u = split_url(proxy);
        u.path = cfg_.base_url;    // absolute-URI for the forward proxy
    }

    std::string response;
    int status = 0;
    if (!http_exchange(u, "POST", body, "application/json", headers,
                       cfg_.timeout_seconds, response, status, err)) {
        if (err.empty()) err = "http status " + std::to_string(status);
        return false;
    }

    // OpenAI: choices[0].message.content — find the "content" key AFTER the
    // first "message" marker to avoid matching prompt echoes.
    const size_t msg = response.find("\"message\"");
    if (!json_get_string(response, "content",
                         msg == std::string::npos ? 0 : msg, reply))
    {
        // Anthropic: content[0].text
        const size_t cont = response.find("\"content\"");
        if (!json_get_string(response, "text",
                             cont == std::string::npos ? 0 : cont, reply)) {
            err = "could not parse provider response";
            return false;
        }
    }
    return true;
}

// ===========================================================================
// HybridRouter
// ===========================================================================
HybridRouter::Decision HybridRouter::route(const std::string& prompt,
                                           bool cloud_available,
                                           bool prefer_cloud) {
    Decision d;
    if (prefer_cloud && cloud_available) {
        d.use_cloud = true;
        d.reason = "cloud: user requested deep reasoning";
        return d;
    }
    // Local when no bridge; otherwise escalate multi-part research asks.
    if (!cloud_available) {
        d.reason = "local: no cloud bridge configured";
        return d;
    }
    size_t clauses = 0;
    for (const char* kw : {" and ", " then ", " analyze ", " compare ",
                           " implications", " research", " macro", " thesis",
                           "earnings", "sec filing"})
        if (prompt.find(kw) != std::string::npos) ++clauses;
    d.use_cloud = prompt.size() > 200 || clauses >= 2;
    d.reason = d.use_cloud
        ? "cloud: multi-part reasoning (size/complexity heuristic)"
        : "local: fast edge task";
    return d;
}

bool HybridRouter::ask_cloud(const std::string& prompt, std::string& reply,
                             std::string& err) {
    if (!bridge_ || !bridge_->available()) {
        err = "cloud bridge unavailable — answer locally";
        return false;
    }
    return bridge_->ask(prompt, reply, err);
}

} // namespace omniseed
