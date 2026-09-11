// =============================================================================
//  OmniSeed — runtime/cloud_bridge.h
//  Optional cloud-reasoning bridge (Phase-16 "Omega Pass", Phase 3).
//
//  Hybrid policy (honest scope):
//    * OmniSeed handles fast edge tasks locally (AgentLoop + deterministic
//      trading analytics) — always available, zero network.
//    * Complex reasoning MAY be delegated to an OpenAI-compatible chat API
//      (OpenAI / Grok / Anthropic-compatible gateways, user's own key in
//      OMNISEED_CLOUD_KEY). No key -> local mode, silently.
//    * The zero-dependency runtime has NO TLS stack. Plain http:// endpoints
//      work directly; for https:// providers set OMNISEED_CLOUD_PROXY to a
//      local relay (e.g. mitmproxy/whisker: `http://127.0.0.1:8080`) which
//      terminates TLS on this machine's behalf. Without it, providers that
//      require TLS report a clear configuration error — the runtime NEVER
//      silently downgrades or fakes a cloud answer.
//
//  Wire format: POST {messages:[{role:"user",content:"..."}]} — parses
//  OpenAI-style `choices[0].message.content` and Anthropic-style
//  `content[0].text` responses.
// =============================================================================
#pragma once

#include <cstdint>
#include <map>
#include <string>

namespace omniseed {

class CloudBridge {
public:
    enum class Provider { OpenAICompatible, Anthropic };

    struct Config {
        Provider provider = Provider::OpenAICompatible;
        std::string api_key;             // empty -> read OMNISEED_CLOUD_KEY
        std::string model = "gpt-4o-mini";
        std::string base_url = "https://api.openai.com/v1/chat/completions";
        std::string proxy_url;           // empty -> read OMNISEED_CLOUD_PROXY
        int32_t timeout_seconds = 30;
        int32_t max_tokens = 512;
    };

    explicit CloudBridge(const Config& cfg = {});

    // True when a key is configured (bridge usable).
    bool available() const { return !api_key_.empty(); }
    const Config& config() const { return cfg_; }

    // One-shot completion. Returns false + err when unavailable/failed.
    bool ask(const std::string& prompt, std::string& reply, std::string& err);

    // HTTP POST JSON (exposed for the web_search tool + tests).
    // Follows at most 3 redirects; honors cfg proxy for https hosts.
    static bool post_json(const std::string& url, const std::string& body,
                          const std::string& content_type, int timeout_s,
                          std::string& response, int& status,
                          std::string& err);

    // Generic request with custom headers (broker auth, etc.). GET when
    // method=="GET" (body ignored); https hosts require the relay env.
    static bool request_json(const std::string& method, const std::string& url,
                             const std::string& body,
                             const std::map<std::string, std::string>& headers,
                             int timeout_s, std::string& response, int& status,
                             std::string& err);

    // GET text (doc_reader/web tools).
    static bool get_text(const std::string& url, int timeout_s,
                         std::string& response, int& status,
                         std::string& err);

private:
    Config cfg_;
    std::string api_key_;
};

// ===========================================================================
// HybridRouter — local-first with explicit cloud escalation
// ===========================================================================
class HybridRouter {
public:
    struct Decision {
        bool use_cloud = false;
        std::string reason;          // "local: fast edge task" / "cloud: deep research"
    };

    explicit HybridRouter(CloudBridge* bridge) : bridge_(bridge) {}

    // Complexity heuristic: short factual/edge prompts stay local; long
    // multi-part research questions escalate when the bridge is available.
    static Decision route(const std::string& prompt, bool cloud_available,
                          bool prefer_cloud = false);

    // Returns the cloud reply when routed to cloud (ok=true), else indicates
    // the caller should answer locally. Never blocks on an unavailable bridge.
    bool ask_cloud(const std::string& prompt, std::string& reply,
                   std::string& err);

private:
    CloudBridge* bridge_;            // not owned; may be null
};

} // namespace omniseed
