// =============================================================================
//  OmniSeed — server/main.cpp
//  Optional HTTP server (OMNISEED_HTTP). Vendored minimal socket loop —
//  no external dependencies. Endpoints:
//    GET  /               -> one-page HTML demo console (talks to /gen)
//    GET  /health         -> {"ok":true,"model":bool,"peak_rss":MB}
//    POST /ask            -> one agent turn (json body: {"task": "...",
//                           "repeat_penalty": 1.2, "repeat_window": 64,
//                           "assistant_lora": "path/to/sidecar.gguf"})
//    POST /gen            -> raw generation (json body: {"prompt": "...",
//                           "repeat_penalty": 1.2, "repeat_window": 64,
//                           "assistant_lora": "path/to/sidecar.gguf"})
//
//  Assistant-behavior LoRA: attach a sidecar at startup with
//  --assistant-lora P (or OMNISEED_ASSISTANT_LORA), or swap it per request
//  with the "assistant_lora" JSON field. The serialized request loop makes
//  per-request swaps race-free; a load/geometry failure logs an error and
//  detaches (requests keep serving the base model).
//
//  Deployment target: Render/Docker (see docs/DEPLOYMENT.md).
// =============================================================================
#include "omniseed/agent/agent.h"
#include "omniseed/core/platform.h"
#include "omniseed/core/rwkv.h"
#include "omniseed/core/tokenizer.h"

#include "omniseed/memory/memory.h"
#include "omniseed/core/tensor.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace omniseed;

#ifndef OMNISEED_HTTP
int main() {
    std::printf("omniseed_server: built without OMNISEED_HTTP.\n"
                "Reconfigure with -DOMNISEED_BUILD_SERVER=ON.\n");
    return 1;
}
#else

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
    #include <netinet/in.h>
    #include <sys/socket.h>
    #include <unistd.h>
    using Socket_t = int;
    constexpr Socket_t kInvalidSocket = -1;
#endif

namespace {

constexpr int kPort = 8080;

bool send_all(Socket_t s, const char* data, size_t len) {
    size_t sent = 0;
    while (sent < len) {
#ifdef _WIN32
        const int n = ::send(s, data + sent,
                             static_cast<int>(len - sent), 0);
#else
        const ssize_t n = ::send(s, data + sent, len - sent, 0);
#endif
        if (n <= 0) return false;
        sent += static_cast<size_t>(n);
    }
    return true;
}

void respond(Socket_t client, const std::string& body,
             const char* content_type = "application/json") {
    std::string hdr =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: " + std::string(content_type) + "\r\n" +
        "Content-Length: " + std::to_string(body.size()) + "\r\n"
        "Connection: close\r\n\r\n";
    send_all(client, hdr.c_str(), hdr.size());
    send_all(client, body.c_str(), body.size());
}

// Extracts "..." value for a top-level key from a small JSON body.
std::string json_field(const std::string& body, const std::string& key) {
    const auto kpos = body.find("\"" + key + "\"");
    if (kpos == std::string::npos) return "";
    const auto cpos = body.find(':', kpos);
    if (cpos == std::string::npos) return "";
    const auto q1 = body.find('"', cpos);
    const auto q2 = body.find('"', q1 + 1);
    if (q1 == std::string::npos || q2 == std::string::npos) return "";
    return body.substr(q1 + 1, q2 - q1 - 1);
}

// Numeric JSON field (unquoted value, e.g. "repeat_penalty":1.2). Returns
// `dflt` when absent or unparseable — requests stay valid without the field.
float json_num_field(const std::string& body, const std::string& key,
                     float dflt) {
    const auto kpos = body.find("\"" + key + "\"");
    if (kpos == std::string::npos) return dflt;
    const auto cpos = body.find(':', kpos);
    if (cpos == std::string::npos) return dflt;
    const char* begin = body.c_str() + cpos + 1;
    char* end = nullptr;
    const float v = std::strtof(begin, &end);
    return (end == begin) ? dflt : v;
}

// One-page demo console: a browser chat box against POST /gen. No assets,
// no CDNs — everything inline so it works on an air-gapped LAN too.
const char* demo_page() {
    return R"HTML(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OmniSeed Console</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; min-height:100vh; display:flex; flex-direction:column;
         font:15px/1.5 system-ui, sans-serif; background:#0e1116; color:#e6e6e6; }
  header { padding:14px 20px; border-bottom:1px solid #232a35;
           display:flex; align-items:baseline; gap:12px; }
  header h1 { margin:0; font-size:17px; letter-spacing:.4px; }
  header .tag { font-size:12px; color:#8b98a8; }
  header .dot { margin-left:auto; font-size:12px; color:#8b98a8; }
  #log { flex:1; overflow-y:auto; padding:20px; max-width:820px; width:100%;
         margin:0 auto; }
  .msg { margin:0 0 14px; padding:10px 14px; border-radius:10px;
         white-space:pre-wrap; word-break:break-word; }
  .you { background:#1d2a45; align-self:flex-end; }
  .bot { background:#1a2029; }
  .bot.pending::after { content:'\25cf'; animation:blink 1s infinite; }
  @keyframes blink { 50% { opacity:.2; } }
  form { display:flex; gap:8px; padding:14px 20px; border-top:1px solid #232a35;
         max-width:820px; width:100%; margin:0 auto; }
  input { flex:1; padding:10px 12px; border-radius:8px; border:1px solid #2c3644;
          background:#161c25; color:#e6e6e6; font:inherit; }
  input:focus { outline:none; border-color:#4a7dd4; }
  button { padding:10px 18px; border:0; border-radius:8px; background:#2f6fed;
           color:#fff; font:inherit; cursor:pointer; }
  button:disabled { background:#243247; cursor:wait; }
  .err { color:#ff8484; font-size:12px; margin:-8px 0 14px; }
</style>
</head>
<body>
<header>
  <h1>&#127790; OmniSeed</h1>
  <span class="tag">sub-300 MB multi-modal micro-LLM &mdash; RWKV-7 &middot; ternary/int8 &middot; pure C++17</span>
  <span class="dot" id="health">checking&hellip;</span>
</header>
<div id="log"></div>
<form id="f" autocomplete="off">
  <input id="p" placeholder="Say something&hellip;" autofocus>
  <button id="b">Send</button>
</form>
<script>
const log = document.getElementById('log'), p = document.getElementById('p'),
      b = document.getElementById('b'), health = document.getElementById('health');
function add(cls, text) {
  const d = document.createElement('div');
  d.className = 'msg ' + cls;
  d.textContent = text;
  log.appendChild(d);
  log.scrollTop = log.scrollHeight;
  return d;
}
async function checkHealth() {
  try {
    const h = await (await fetch('/health')).json();
    health.textContent = 'model: ' + (h.model ? 'loaded' : 'missing') +
                         ' | peak RSS: ' + h.peak_rss + ' MB';
  } catch { health.textContent = 'unreachable'; }
}
checkHealth(); setInterval(checkHealth, 30000);

document.getElementById('f').addEventListener('submit', async e => {
  e.preventDefault();
  const text = p.value.trim();
  if (!text || b.disabled) return;
  add('you', text);
  p.value = '';
  b.disabled = true;
  const bot = add('bot pending', '');
  try {
    const r = await fetch('/gen', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: text })
    });
    const j = await r.json();
    bot.classList.remove('pending');
    bot.textContent = (j.text || j.error || '(empty response)').replace(/\n+/g, ' ').trim();
  } catch (err) {
    bot.classList.remove('pending');
    bot.textContent = 'error: ' + err;
    bot.classList.add('err');
  }
  b.disabled = false;
  p.focus();
  log.scrollTop = log.scrollHeight;
});
</script>
</body>
</html>)HTML";
}

// Current LoRA attachment (server is serialized, so one global is race-free).
LoraAdapter* server_lora = nullptr;
std::string  server_lora_path;

// Attach a LoRA sidecar to the model, or detach it. Returns true when a
// sidecar is now attached; logs (never throws) on failure.
bool set_server_lora(RwkvModel& model, LoraAdapter*& lora,
                     const std::string& path) {
    if (path.empty()) {              // explicit empty = detach
        if (lora) platform::log_info("assistant LoRA detached");
        model.set_lora(nullptr);
        delete lora;
        lora = nullptr;
        return false;
    }
    if (lora && server_lora_path == path) return true;   // already attached
    if (lora) { model.set_lora(nullptr); delete lora; lora = nullptr; }
    lora = new LoraAdapter();
    const int32_t nl = static_cast<int32_t>(model.config().n_layers);
    const int32_t ne = static_cast<int32_t>(model.config().n_embd);
    if (!lora->load(path) || lora->layer_count() != nl
        || lora->n_embd() != ne) {
        platform::log_error(
            "assistant LoRA load failed (%s) — serving WITHOUT it",
            lora->valid() ? "geometry mismatch vs base model"
                          : lora->error().c_str());
        delete lora;
        lora = nullptr;
        return false;
    }
    model.set_lora(lora);
    server_lora_path = path;
    platform::log_info("assistant LoRA attached: %s (rank %d, scaling %.2f)",
                       path.c_str(), lora->rank(), lora->scaling());
    return true;
}

} // namespace

int main(int argc, char** argv) {
    platform::enable_utf8_console();   // Windows codepage 65001; no-op elsewhere
    int port = kPort;
    std::string model_path = "./models/omniseed.gguf";
    std::string lora_path;
    // Documented deployment env (see Dockerfile): OMNISEED_MODEL is the
    // fallback default; an explicit --model flag wins.
    if (const char* env = std::getenv("OMNISEED_MODEL")) model_path = env;
    if (const char* env = std::getenv("OMNISEED_ASSISTANT_LORA"))
        lora_path = env;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--port") == 0 && i + 1 < argc)
            port = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc)
            model_path = argv[++i];
        else if (std::strcmp(argv[i], "--assistant-lora") == 0 && i + 1 < argc)
            lora_path = argv[++i];
    }

#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        platform::log_error("WSAStartup failed");
        return 1;
    }
#endif

    Socket_t listener = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listener == kInvalidSocket) {
        platform::log_error("socket() failed");
        return 1;
    }
    int one = 1;
    ::setsockopt(listener, SOL_SOCKET, SO_REUSEADDR,
                 reinterpret_cast<const char*>(&one), sizeof(one));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (::bind(listener, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        platform::log_error("bind failed on port %d", port);
        return 1;
    }
    if (::listen(listener, 8) != 0) {
        platform::log_error("listen failed");
        return 1;
    }

    platform::log_info("omniseed server listening on :%d (model: %s)",
                       port, model_path.c_str());

    // Model + agent bundle (single session; request loop is serialized).
    Tokenizer tok;
    RwkvModel model;
    const bool have_model = model.load(model_path);
    if (have_model) {
        // Prefer the world vocab embedded in the GGUF over the minimal
        // byte-level tokenizer (chat-quality decoding requires it).
        if (!tok.load_from_gguf(model.gguf_store())) {
            platform::log_warn("GGUF tokenizer missing — falling back to minimal");
            tok.build_minimal();
        }
    } else {
        tok.build_minimal();
    }
    if (!have_model) {
        platform::log_warn("model not loaded: %s — /health only",
                           model.error().c_str());
    } else if (!lora_path.empty()) {
        // Startup attachment (--assistant-lora P or OMNISEED_ASSISTANT_LORA):
        // attach before the request loop so every generation honors it.
        set_server_lora(model, server_lora, lora_path);
    }

    ToolRegistry tools;
    for (const char* t : {"calc", "echo", "time"}) tools.add_builtin(t);
    MemoryCrystals mem;
    ComputeThrottle thr;
    SelfImprovement imp;
    AgentLoop::Config cfg;
    AgentLoop loop(model, tok, tools, mem, thr, imp, cfg);

    while (true) {
        Socket_t client = ::accept(listener, nullptr, nullptr);
        if (client == kInvalidSocket) continue;

        char buf[4096] = {0};
#ifdef _WIN32
        const int n = ::recv(client, buf, sizeof(buf) - 1, 0);
#else
        const ssize_t n = ::recv(client, buf, sizeof(buf) - 1, 0);
#endif
        if (n <= 0) {
            ::closesocket(client);
            continue;
        }
        const std::string req(buf, static_cast<size_t>(n));

        // request line: METHOD PATH HTTP/x
        const auto sp1 = req.find(' ');
        const auto sp2 = req.find(' ', sp1 + 1);
        const std::string method =
            sp1 == std::string::npos ? "" : req.substr(0, sp1);
        const std::string path =
            (sp1 == std::string::npos || sp2 == std::string::npos)
                ? "" : req.substr(sp1 + 1, sp2 - sp1 - 1);
        const auto body_pos = req.find("\r\n\r\n");
        const std::string body = body_pos == std::string::npos
            ? "" : req.substr(body_pos + 4);

        if (path == "/" || path == "/index.html") {
            respond(client, demo_page(), "text/html; charset=utf-8");
        } else if (path == "/health") {
            std::string b = std::string("{\"ok\":true,\"model\":") +
                            (have_model ? "true" : "false") +
                            ",\"assistant_lora\":" +
                            (server_lora ? "true" : "false") +
                            ",\"peak_rss\":" +
                            std::to_string(platform::peak_rss_bytes() / 1048576) +
                            "}";
            respond(client, b);
        } else if (path == "/ask" && method == "POST" && have_model) {
            const std::string task = json_field(body, "task");
            // Optional per-request sampling overrides (Phase 13):
            //   "repeat_penalty": 1.2   (1.0 = off, >1 breaks text loops)
            //   "repeat_window":  64    (recent-token ring size)
            loop.set_sampling(json_num_field(body, "repeat_penalty", 1.0f),
                              static_cast<int32_t>(
                                  json_num_field(body, "repeat_window", 64.0f)));
            // Optional per-request assistant LoRA (Phase 14): absent field
            // leaves the current attachment untouched (startup flag/env still
            // applies); "" detaches for this request's identity.
            const std::string want = json_field(body, "assistant_lora");
            if (!want.empty() || body.find("\"assistant_lora\"")
                                 != std::string::npos)
                set_server_lora(model, server_lora, want);
            const AgentLoop::Result r = loop.run(task);
            respond(client, "{\"reply\":\"" + r.reply + "\"}");
        } else if (path == "/gen" && method == "POST" && have_model) {
            const std::string prompt = json_field(body, "prompt");
            loop.set_sampling(json_num_field(body, "repeat_penalty", 1.0f),
                              static_cast<int32_t>(
                                  json_num_field(body, "repeat_window", 64.0f)));
            const std::string want = json_field(body, "assistant_lora");
            if (!want.empty() || body.find("\"assistant_lora\"")
                                 != std::string::npos)
                set_server_lora(model, server_lora, want);
            auto ids = tok.encode_chat(prompt);
            RwkvState st;
            model.init_state(st);
            Tensor logits("logits", {model.config().n_vocab}, DType::F32);
            for (const int32_t id : ids) model.forward(id, st, logits);
            const int32_t seed = ids.empty() ? Tokenizer::kBosId : ids.back();
            const std::string out =
                loop.generate(st, seed, 160, {Tokenizer::kEosId}, nullptr);
            respond(client, "{\"text\":\"" + out + "\"}");
        } else {
            respond(client, "{\"error\":\"not found\"}");
        }

        ::closesocket(client);
    }

#ifdef _WIN32
    WSACleanup();
#endif
    return 0;
}

#endif // OMNISEED_HTTP
