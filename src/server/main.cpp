// =============================================================================
//  OmniSeed — server/main.cpp
//  Optional HTTP server (OMNISEED_HTTP). Vendored minimal socket loop —
//  no external dependencies. Endpoints:
//    GET  /               -> one-page HTML demo console (talks to /gen)
//    GET  /health         -> {"ok":true,"model":bool,"paused":bool,
//                             "auth":bool,"peak_rss":MB}
//    POST /ask            -> one agent turn (json body: {"task": "...",
//                           "repeat_penalty": 1.2, "repeat_window": 64,
//                           "stop": ["\\nUser:"], "stop_defaults": true,
//                           "assistant_lora": "path/to/sidecar.gguf"})
//    POST /gen            -> raw generation (json body: {"prompt": "...",
//                           "repeat_penalty": 1.2, "repeat_window": 64,
//                           "stop": ["\\nUser:"], "stop_defaults": true,
//                           "assistant_lora": "path/to/sidecar.gguf"})
//    POST /asr            -> whisper-tiny transcription. Body is either RAW
//                           16 kHz mono 16-bit PCM WAV bytes (Content-Type:
//                           audio/wav or application/octet-stream) or JSON
//                           {"wav_b64": "..."}. Returns {"text": ..., 
//                           "seconds": ...}; clean error on silence/too-short
//    POST /control        -> {"action":"pause"|"resume"|"status"}; while
//                           paused, /gen and /ask refuse with
//                           {"error":"paused"}. In-flight generations finish.
//    POST /classify       -> run a FITTED classification head (T112):
//                           {"set":"vision.scene","hidden":[...E floats...],
//                            "top_k":3,"heads_dir":"models/heads"}. Returns the
//                           head's own JSON (domain/top_k/margin) or
//                           {"error":"not-available","reason":...} — a missing
//                           blob, an unknown set or an E mismatch REFUSES; it
//                           never fabricates a distribution. Needs no model.
//    AUTH: when OMNISEED_API_TOKEN is set in the environment, every endpoint
//    except / and /health requires `Authorization: Bearer <token>` (401
//    otherwise; timing-safe compare). Unset = open (edge default).
//                           audio (the mel trim leaves < 2 frames).
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
#include "omniseed/audio/audio.h"
#include "omniseed/core/platform.h"
#include "omniseed/core/rwkv.h"
#include "omniseed/core/tokenizer.h"

#include "omniseed/memory/memory.h"
#include "omniseed/core/tensor.h"
#include "omniseed/runtime/fitted_heads.h"
#include "omniseed/runtime/sidecar_hub.h"
#include "omniseed/server/ws.h"

#ifndef _WIN32
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <thread>
#include <vector>
#include <chrono>
#include <mutex>

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

// Windows has closesocket(); POSIX sockets are file descriptors, so close().
void close_socket(Socket_t s) {
#ifdef _WIN32
    ::closesocket(s);
#else
    ::close(s);
#endif
}

void respond(Socket_t client, const std::string& body,
             const char* content_type = "application/json",
             const std::string& status_line = "200 OK",
             const std::string& extra_headers = "") {
    std::string hdr =
        "HTTP/1.1 " + status_line + "\r\n"
        "Content-Type: " + std::string(content_type) + "\r\n" +
        extra_headers +
        "Content-Length: " + std::to_string(body.size()) + "\r\n"
        "Connection: close\r\n\r\n";
    send_all(client, hdr.c_str(), hdr.size());
    send_all(client, body.c_str(), body.size());
}

// Escapes a string for embedding in a JSON response body. Model replies are
// arbitrary text and routinely contain quotes; emitting them raw produced
// invalid JSON, so every reply now goes through here.
std::string json_escape(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 16);
    for (const char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20u) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c & 0xFF);
                    o += buf;
                } else {
                    o.push_back(c);
                }
        }
    }
    return o;
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
    return (end == begin) ? dflt : v;}

// String-array JSON field: "stop": ["\nUser:", "\nAssistant:"] (plain UTF-8
// bytes; the runtime has no JSON decoder, so escapes stay literal — clients
// send raw newlines as \\n, which then... is left to the client; documented).
// Mirrors the CLI's repeatable --stop. Absent key -> unchanged behavior.
std::vector<std::string> json_stop_array(const std::string& body) {
    std::vector<std::string> stops;
    const auto kpos = body.find("\"stop\"");
    if (kpos == std::string::npos) return stops;
    const auto cpos = body.find(':', kpos);
    if (cpos == std::string::npos) return stops;
    const auto end = body.find(']', cpos);
    if (end == std::string::npos) return stops;
    size_t p = cpos + 1;
    while (p < end) {
        const auto q1 = body.find('"', p);
        if (q1 == std::string::npos || q1 >= end) break;
        const auto q2 = body.find('"', q1 + 1);
        if (q2 == std::string::npos || q2 > end) break;
        if (q2 > q1 + 1) stops.push_back(body.substr(q1 + 1, q2 - q1 - 1));
        p = q2 + 1;
    }
    return stops;
}

// Float-array JSON field: "hidden": [0.1, -2.0, ...] (unquoted numbers, no
// nested structures). Returns false when the key is absent or the array is
// malformed — same no-JSON-decoder stance as json_stop_array; a caller that
// needs the refusal uses the bool, never a partially filled vector (the
// vector is cleared first).
bool json_float_array(const std::string& body, const std::string& key,
                      std::vector<float>& out) {
    out.clear();
    const auto kpos = body.find("\"" + key + "\"");
    if (kpos == std::string::npos) return false;
    const auto cpos = body.find(':', kpos);
    if (cpos == std::string::npos) return false;
    const auto lb = body.find('[', cpos);
    const auto end = body.find(']', cpos);
    if (lb == std::string::npos || end == std::string::npos || lb > end)
        return false;
    const char* p = body.c_str() + lb + 1;
    const char* stop = body.c_str() + end;
    while (p < stop) {
        while (p < stop && (*p == ' ' || *p == ',' || *p == '\n' ||
                            *p == '\r' || *p == '\t')) ++p;
        if (p >= stop) break;
        char* e = nullptr;
        const float v = std::strtof(p, &e);
        if (e == p) { out.clear(); return false; }  // not a number -> refuse
        out.push_back(v);
        p = e;
    }
    return true;
}

// "stop_defaults": true -> the assistant stop set (mirrors CLI
// --stop-defaults); combined with any explicit "stop" entries.
std::vector<std::string> request_stops(const std::string& body) {
    std::vector<std::string> stops = json_stop_array(body);
    const auto kpos = body.find("\"stop_defaults\"");
    if (kpos != std::string::npos) {
        const auto cpos = body.find(':', kpos);
        if (cpos != std::string::npos &&
            body.find("true", cpos) < body.find('}', cpos)) {
            for (const char* d : kAssistantStopDefaults)
                stops.push_back(d);
        }
    }
    return stops;
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
  #ctl { display:flex; gap:8px; margin:10px 0; }
  #ctl input { flex:1; padding:8px 10px; border:1px solid #2a3a52;
               border-radius:8px; background:#101a2b; color:#dbe4f3;
               font:inherit; font-size:13px; }
  #ctl button { padding:8px 14px; }
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
<div id="ctl">
  <input id="tok" type="password" placeholder="API token (if required)" autocomplete="off">
  <button id="pause" type="button" title="pause/resume generation">&#9208;</button>
</div>
<script>
const log = document.getElementById('log'), p = document.getElementById('p'),
      b = document.getElementById('b'), health = document.getElementById('health'),
      tok = document.getElementById('tok'), pauseBtn = document.getElementById('pause');
function authHdr() {
  // token kept in this page's memory only; sent as a Bearer header
  return tok.value ? { 'Authorization': 'Bearer ' + tok.value } : {};
}
function add(cls, text) {
  const d = document.createElement('div');
  d.className = 'msg ' + cls;
  d.textContent = text;
  log.appendChild(d);
  log.scrollTop = log.scrollHeight;
  return d;
}
let serverPaused = false;
async function checkHealth() {
  try {
    const h = await (await fetch('/health')).json();
    serverPaused = !!h.paused;
    pauseBtn.textContent = serverPaused ? '\u25B6' : '\u23F8';
    health.textContent = 'model: ' + (h.model ? 'loaded' : 'missing') +
                         (h.auth ? ' | auth: on' : '') +
                         (serverPaused ? ' | PAUSED' : '') +
                         ' | peak RSS: ' + h.peak_rss + ' MB';
  } catch { health.textContent = 'unreachable'; }
}
checkHealth(); setInterval(checkHealth, 30000);

// ---- T103 live updates: WebSocket with reconnect backoff --------------------
// Render free tier sleeps after ~15 min idle (wake ~1 min) -> exponential
// backoff capped at 30 s; REST polling above stays as the fallback.
let ws = null, wsTries = 0;
function connectWS() {
  const url = (location.protocol === 'https:' ? 'wss://' : 'ws://') +
              location.host + '/ws' +
              (tok.value ? '?token=' + encodeURIComponent(tok.value) : '');
  try { if (ws) { ws.onclose = null; ws.close(); } } catch {}
  ws = new WebSocket(url);
  ws.onopen = () => { wsTries = 0; };
  ws.onmessage = ev => {
    try {
      const j = JSON.parse(ev.data);
      if (j.type !== 'health') return;
      serverPaused = !!j.paused;
      pauseBtn.textContent = serverPaused ? '\u25B6' : '\u23F8';
      health.textContent = 'live | model: ' + (j.model ? 'loaded' : 'missing') +
                           (j.auth ? ' | auth: on' : '') +
                           (serverPaused ? ' | PAUSED' : '') +
                           ' | RSS: ' + j.rss + ' MB';
    } catch {}
  };
  ws.onclose = () => {
    wsTries = Math.min(wsTries + 1, 6);
    const delay = Math.min(1000 * Math.pow(2, wsTries - 1), 30000);
    health.textContent = 'reconnecting in ' + Math.round(delay / 1000) + 's\u2026';
    setTimeout(connectWS, delay);
  };
  ws.onerror = () => { try { ws.close(); } catch {} };
}
connectWS();

pauseBtn.addEventListener('click', async () => {
  pauseBtn.disabled = true;
  try {
    const r = await fetch('/control', {
      method: 'POST',
      headers: Object.assign({ 'Content-Type': 'application/json' }, authHdr()),
      body: JSON.stringify({ action: serverPaused ? 'resume' : 'pause' })
    });
    const j = await r.json();
    if (j.error) add('bot', 'control: ' + j.error);
    serverPaused = !!j.paused;
    pauseBtn.textContent = serverPaused ? '\u25B6' : '\u23F8';
  } catch (err) { add('bot err', 'control error: ' + err); }
  pauseBtn.disabled = false;
  checkHealth();
});

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
      headers: Object.assign({ 'Content-Type': 'application/json' }, authHdr()),
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

// Whisper-tiny ASR sidecar: T123b hot-swap governed — registered with the
// SidecarHub, demand-mapped on the first /asr (most deployments never call
// it), demand-REmapped after a pressure-unmap. The serialized request loop
// makes hub access race-free (acquire -> transcribe -> release per request,
// so the sidecar is users==0 between requests and eligible when RSS hits
// the soft watermark; the hub polls platform::current_rss_bytes() BEFORE
// each request — the HTTP loop IS the sampling point, no monitor thread).
// Decode-sidecar tensors are optional: encoder-only sidecars answer with
// the documented fallback text.
SidecarHub server_sidecars;

// Request-scoped alias: set by server_asr_ready(), released by
// server_asr_done() at the end of the /asr branch. NEVER outlives one
// request (the hub owns the object; this is a non-owning view).
WhisperTiny* server_whisper = nullptr;
void server_asr_done() {
    server_sidecars.release("whisper");
    server_whisper = nullptr;
}

bool server_asr_ready() {
    static bool registered = [] {
        SidecarHub::Spec spec;
        spec.bytes_hint = 76445082;   // models/whisper-tiny-encoder.gguf
        spec.load = [](std::string& err) -> std::shared_ptr<void> {
            auto* w = new WhisperTiny();
            if (!w->load("models/whisper-tiny-encoder.gguf")) {
                err = w->error();
                delete w;
                return nullptr;
            }
            platform::log_info(
                "asr: whisper sidecar loaded (decoder %s)",
                w->decoder_loaded() ? "real" : "fallback");
            return std::shared_ptr<void>(
                w, [](void* q) { delete static_cast<WhisperTiny*>(q); });
        };
        std::string rerr;
        const bool ok =
            server_sidecars.register_spec("whisper", std::move(spec), rerr);
        if (!ok) platform::log_error("asr: hub register failed: %s",
                                      rerr.c_str());
        return ok;
    }();
    if (!registered) return false;
    // Pressure poll FIRST: an idle-previous-request sidecar may be unmap-
    // eligible right now (freeing pages BEFORE this request possibly remaps).
    const uint64_t zone = 0;
    server_sidecars.on_rss_pressure(nullptr);
    (void)zone;
    std::string err;
    const std::shared_ptr<void> hold = server_sidecars.acquire("whisper", err);
    if (!hold) {
        platform::log_error("asr: %s", err.c_str());
        return false;
    }
    // Hand the RAW pointer to the request; release() happens in the /asr
    // branch after transcribe() so users>0 for the whole request span.
    server_whisper = static_cast<WhisperTiny*>(hold.get());
    return true;
}

// Read exactly `len` more bytes off the socket (Content-Length body).
bool recv_exact(Socket_t s, std::string& out, size_t len) {
    out.resize(out.size() + len);
    size_t got = 0;
    char* dst = out.data() + out.size() - len;
    while (got < len) {
#ifdef _WIN32
        const int n = ::recv(s, dst + got,
                             static_cast<int>(len - got), 0);
#else
        const ssize_t n = ::recv(s, dst + got, len - got, 0);
#endif
        if (n <= 0) return false;
        got += static_cast<size_t>(n);
    }
    return true;
}

// Pull headers + a complete body off the socket. Greedy servers may keep a
// keep-alive connection open, so reads continue until Content-Length is
// satisfied or the peer closes (n == 0). Returns false on socket errors.
bool read_request(Socket_t s, std::string& headers, std::string& body) {
    std::string acc;
    char buf[8192];
    size_t header_end = std::string::npos;
    while (header_end == std::string::npos) {
#ifdef _WIN32
        const int n = ::recv(s, buf, sizeof(buf), 0);
#else
        const ssize_t n = ::recv(s, buf, sizeof(buf), 0);
#endif
        if (n <= 0) return false;
        acc.append(buf, static_cast<size_t>(n));
        header_end = acc.find("\r\n\r\n");
    }
    headers = acc.substr(0, header_end);
    body = acc.substr(header_end + 4);
    // Content-Length (case-insensitive header scan; portable tolower walk).
    size_t cl = std::string::npos;
    for (size_t p = 0; p + 16 < headers.size(); ++p) {
        static const char kCl[] = "content-length:";
        bool hit = true;
        for (size_t j = 0; j < 15; ++j) {
            if (std::tolower(static_cast<unsigned char>(headers[p + j])) !=
                kCl[j]) {
                hit = false;
                break;
            }
        }
        if (hit) {
            cl = std::strtoul(headers.c_str() + p + 15, nullptr, 10);
            break;
        }
    }
    if (cl == std::string::npos) return true;             // bodyless request
    if (cl > 100u * 1024u * 1024u) return false;          // sane cap: 100 MB
    while (body.size() < cl) {
#ifdef _WIN32
        const int n = ::recv(s, buf,
                             static_cast<int>(std::min(sizeof(buf),
                                 cl - body.size())), 0);
#else
        const ssize_t n = ::recv(s, buf, std::min(sizeof(buf),
                                 cl - body.size()), 0);
#endif
        if (n <= 0) return false;
        body.append(buf, static_cast<size_t>(n));
    }
    body.resize(cl);                                      // trim over-read
    return true;
}

// Standard base64 decode (RFC 4648; whitespace tolerated). Returns false on
// any character outside the alphabet/padding.
bool base64_decode(const std::string& in, std::string& out) {
    static const int8_t T[256] = {
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,
        -1,-1,-1,63,52,53,54,55,56,57,58,59,60,61,-1,-1,-1,-2,-1,-1,-1,0,1,
        2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,
        -1,-1,-1,-1,-1,-1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,
        42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
    out.clear();
    out.reserve(in.size() / 4 * 3 + 3);
    uint32_t acc = 0;
    int bits = 0;
    for (const char c : in) {
        const unsigned char u = static_cast<unsigned char>(c);
        if (u == ' ' || u == '\r' || u == '\n' || u == '\t') continue;
        const int8_t v = T[u];
        if (v == -1) return false;                        // illegal char
        if (v == -2) break;                               // '=' padding: stop
        acc = (acc << 6) | static_cast<uint32_t>(v);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<char>((acc >> bits) & 0xFF));
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// API auth + control (Phase 8): bearer-token gate and the pause toggle.
//
// Auth is OFF unless OMNISEED_API_TOKEN is set in the environment — an edge
// box behind Render's network boundary keeps today's behavior exactly, and a
// public deployment turns it on with one env var. The token is never logged
// and never echoed. Protected: /ask /gen /asr /control. Open: / (console
// HTML — it must load before it can ask for the token) and /health
// (readiness probes must not need secrets).
// ---------------------------------------------------------------------------
std::atomic<bool> g_server_paused{false};

// ---------------------------------------------------------------------------
// T103 — WebSocket live updates (GET /ws).
//
// The HTTP loop is serial and /gen can block for seconds, so upgraded
// sockets live on a DEDICATED thread: main() hands them over, the thread
// answers pings, honors the close handshake and broadcasts a health snapshot
// every 2 s (which also surfaces pause transitions within one tick). Only the
// WS thread writes to adopted sockets afterwards => no send mutex needed.
// ---------------------------------------------------------------------------
namespace {

struct WsConn {
    Socket_t sock;
    omniseed::server::ws::FrameParser parser;
};

struct WsRegistry {
    std::mutex m;
    std::vector<WsConn> conns;
};

WsRegistry g_ws;
std::atomic<bool> g_ws_stop{false};

void ws_set_nonblocking(Socket_t s) {
#ifdef _WIN32
    u_long nb = 1;
    ::ioctlsocket(s, FIONBIO, &nb);
#else
    const int fl = ::fcntl(s, F_GETFL, 0);
    if (fl >= 0) ::fcntl(s, F_SETFL, fl | O_NONBLOCK);
#endif
}

void ws_close_conn(WsRegistry* reg, Socket_t s) {
    {
        std::lock_guard<std::mutex> lk(reg->m);
        for (size_t i = 0; i < reg->conns.size(); ++i) {
            if (reg->conns[i].sock == s) {
                reg->conns.erase(reg->conns.begin() +
                                 static_cast<std::ptrdiff_t>(i));
                break;
            }
        }
    }
    close_socket(s);
}

std::string ws_snapshot(bool auth_on, bool have_model) {
    return std::string("{\"type\":\"health\",\"ok\":true,\"paused\":") +
           (g_server_paused.load() ? "true" : "false") +
           ",\"auth\":" + (auth_on ? "true" : "false") +
           ",\"model\":" + (have_model ? "true" : "false") +
           ",\"rss\":" +
           std::to_string(platform::peak_rss_bytes() / 1048576) + "}";
}

void ws_thread_main(WsRegistry* reg, bool auth_on, bool have_model) {
    namespace wsp = omniseed::server::ws;
    using Clock = std::chrono::steady_clock;
    auto last_tick = Clock::now() - std::chrono::seconds(3);
    while (!g_ws_stop.load(std::memory_order_relaxed)) {
        std::vector<WsConn> conns;
        {
            std::lock_guard<std::mutex> lk(reg->m);
            conns = reg->conns;
        }
        // ---- drain client frames: ping -> pong, close -> ack + drop --------
        for (WsConn& c : conns) {
            bool dropped = false;
            char buf[2048];
            while (!dropped) {
                const int n = ::recv(c.sock, buf, sizeof(buf), 0);
                if (n <= 0) {
#ifdef _WIN32
                    const bool dead =
                        (n == 0 || WSAGetLastError() != WSAEWOULDBLOCK);
#else
                    const bool dead = (n == 0 || errno != EWOULDBLOCK);
#endif
                    if (dead) ws_close_conn(reg, c.sock);
                    break;  // would-block (keep) or dead (dropped)
                }
                wsp::Frame f;
                const wsp::ParseResult r =
                    c.parser.feed(buf, static_cast<size_t>(n));
                if (r == wsp::ParseResult::kNeedMore) break;
                if (r == wsp::ParseResult::kProtocolError) {
                    const std::string bye = wsp::encode_frame(
                        wsp::kClose, std::string("\x03\xea", 2));  // 1002
                    send_all(c.sock, bye.data(), bye.size());
                    ws_close_conn(reg, c.sock);
                    dropped = true;
                    break;
                }
                while (c.parser.take(f)) {
                    if (f.opcode == wsp::kPing) {
                        const std::string pong =
                            wsp::encode_frame(wsp::kPong, "");
                        if (!send_all(c.sock, pong.data(), pong.size())) {
                            ws_close_conn(reg, c.sock);
                            dropped = true;
                            break;
                        }
                    } else if (f.opcode == wsp::kClose) {
                        const std::string bye =
                            wsp::encode_frame(wsp::kClose, "");
                        send_all(c.sock, bye.data(), bye.size());
                        ws_close_conn(reg, c.sock);
                        dropped = true;
                        break;
                    }
                    // Console sends no data frames; anything else is consumed
                    // and ignored (RFC 6455 endpoints must still read them).
                }
            }
        }
        // ---- periodic broadcast (also carries pause transitions) -----------
        const auto now = Clock::now();
        if (now - last_tick >= std::chrono::seconds(2)) {
            last_tick = now;
            std::vector<WsConn> cur;
            {
                std::lock_guard<std::mutex> lk(reg->m);
                cur = reg->conns;
            }
            if (!cur.empty()) {
                const std::string frame = wsp::encode_frame(
                    wsp::kText, ws_snapshot(auth_on, have_model));
                for (const WsConn& c : cur) {
                    if (!send_all(c.sock, frame.data(), frame.size())) {
                        ws_close_conn(reg, c.sock);
                    }
                }
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
    }
}

}  // namespace

// Hand a freshly upgraded socket to the WS thread (never touched again here).
void ws_adopt(Socket_t s) {
    ws_set_nonblocking(s);
    std::lock_guard<std::mutex> lk(g_ws.m);
    if (g_ws.conns.size() < 64) {
        g_ws.conns.push_back(WsConn{s, {}});
    } else {
        close_socket(s);
    }
}

// Case-insensitive header value for `name`, or "" — same portable walk as
// the Content-Length scan (no locale-dependent string ops).
std::string header_value(const std::string& headers, const char* name) {
    const size_t nlen = std::strlen(name);
    for (size_t p = 0; p + nlen < headers.size(); ++p) {
        bool hit = true;
        for (size_t j = 0; j < nlen; ++j) {
            if (std::tolower(static_cast<unsigned char>(headers[p + j])) !=
                name[j]) { hit = false; break; }
        }
        if (!hit) continue;
        // must be at line start (or request line — names here never collide)
        if (p > 0 && headers[p - 1] != '\n') continue;
        size_t v = p + nlen;
        while (v < headers.size() && (headers[v] == ' ' || headers[v] == ':'))
            ++v;
        size_t e = headers.find('\r', v);
        if (e == std::string::npos) e = headers.find('\n', v);
        if (e == std::string::npos) e = headers.size();
        return headers.substr(v, e - v);
    }
    return "";
}

// Timing-safe equality: XOR-accumulate over the FULL longer length so the
// work done does not depend on where a mismatch occurs.
bool token_equal(const std::string& a, const std::string& b) {
    const size_t n = a.size() > b.size() ? a.size() : b.size();
    unsigned char diff = static_cast<unsigned char>(a.size() ^ b.size());
    for (size_t i = 0; i < n; ++i) {
        const unsigned char ca = i < a.size()
            ? static_cast<unsigned char>(a[i]) : 0u;
        const unsigned char cb = i < b.size()
            ? static_cast<unsigned char>(b[i]) : 0u;
        diff = static_cast<unsigned char>(diff | (ca ^ cb));
    }
    return diff == 0;
}

// true when the request carries a valid `Authorization: Bearer <token>`.
// An EMPTY configured token means auth is disabled (open by design).
bool bearer_authorized(const std::string& headers,
                       const std::string& configured) {
    if (configured.empty()) return true;
    const std::string auth = header_value(headers, "authorization");
    static const char kBear[] = "bearer ";   // BOTH sides compared lowercased
    if (auth.size() <= 7) return false;
    for (size_t j = 0; j < 7; ++j) {
        if (std::tolower(static_cast<unsigned char>(auth[j])) != kBear[j])
            return false;
    }
    return token_equal(auth.substr(7), configured);
}

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
    // System-1 decision head ("two heads, one brain"). Off by default so the
    // server behaves exactly as before unless asked otherwise.
    DecisionMode decision_mode = DecisionMode::Off;
    float       decision_threshold = 0.85f;
    std::string decision_head_path;
    // Documented deployment env (see Dockerfile): OMNISEED_MODEL is the
    // fallback default; an explicit --model flag wins.
    if (const char* env = std::getenv("OMNISEED_MODEL")) model_path = env;
    if (const char* env = std::getenv("OMNISEED_ASSISTANT_LORA"))
        lora_path = env;
    // Same env-then-flag precedence for the decision head.
    if (const char* env = std::getenv("OMNISEED_DECISION_MODE")) {
        const std::string m = env;
        if (m == "hybrid")             decision_mode = DecisionMode::Hybrid;
        else if (m == "decision-only") decision_mode = DecisionMode::DecisionOnly;
    }
    if (const char* env = std::getenv("OMNISEED_DECISION_HEAD"))
        decision_head_path = env;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--port") == 0 && i + 1 < argc)
            port = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc)
            model_path = argv[++i];
        else if (std::strcmp(argv[i], "--assistant-lora") == 0 && i + 1 < argc)
            lora_path = argv[++i];
        else if (std::strcmp(argv[i], "--decision-mode") == 0 && i + 1 < argc) {
            const std::string m = argv[++i];
            if (m == "off") {
                decision_mode = DecisionMode::Off;
            } else if (m == "hybrid") {
                decision_mode = DecisionMode::Hybrid;
            } else if (m == "decision-only" || m == "decision_only") {
                decision_mode = DecisionMode::DecisionOnly;
            } else {
                platform::log_error("--decision-mode: unknown value '%s' "
                                    "(expected off | hybrid | decision-only)",
                                    m.c_str());
                return 2;
            }
        }
        else if (std::strcmp(argv[i], "--decision-threshold") == 0 && i + 1 < argc)
            decision_threshold = static_cast<float>(std::atof(argv[++i]));
        else if (std::strcmp(argv[i], "--decision-head") == 0 && i + 1 < argc)
            decision_head_path = argv[++i];
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
    cfg.decision_mode      = decision_mode;
    cfg.decision_threshold = decision_threshold;
    cfg.decision_head_path = decision_head_path;
    AgentLoop loop(model, tok, tools, mem, thr, imp, cfg);
    // Report head provenance once at startup. An unfitted head is a
    // placeholder: the operator must know its routing is not a judgement.
    if (cfg.decision_mode != DecisionMode::Off) {
        if (loop.ensure_decision_head()) {
            const DecisionHead& dh = loop.decision_head();
            platform::log_info("decision head: mode=%s E=%d threshold=%.2f %s",
                               decision_mode_name(cfg.decision_mode),
                               dh.hidden_size(), dh.threshold(),
                               dh.trained() ? "fitted" : "UNTRAINED placeholder");
            if (!dh.trained())
                platform::log_warn("decision head has no fitted projection: "
                                   "decisions are well-formed but MEANINGLESS "
                                   "(pass --decision-head)");
        } else {
            platform::log_warn("--decision-mode %s requested but the head is "
                               "unavailable; System-2 only",
                               decision_mode_name(cfg.decision_mode));
        }
    }

    // API token (Phase 8): read ONCE from the environment; never logged,
    // never echoed. Empty = auth disabled (today's behavior, by design).
    const char* tok_env = std::getenv("OMNISEED_API_TOKEN");
    const std::string api_token = tok_env ? tok_env : "";
    if (!api_token.empty())
        platform::log_info("api token: SET — /ask /gen /asr /control now "
                           "require Authorization: Bearer ...");
    else
        platform::log_info("api token: unset — endpoints open (set "
                           "OMNISEED_API_TOKEN to require auth)");

    // T103: dedicated WS thread — keeps answering pings and broadcasting
    // health snapshots even while the serial HTTP thread is mid-generation.
    std::thread(ws_thread_main, &g_ws, !api_token.empty(), have_model)
        .detach();

    while (true) {
        Socket_t client = ::accept(listener, nullptr, nullptr);
        if (client == kInvalidSocket) continue;

        // /asr bodies are far larger than one recv; read headers + full body
        // once (Content-Length honored). Text endpoints below use the same
        // `body` string as before.
        std::string req_headers, body;
        if (!read_request(client, req_headers, body)) {
            close_socket(client);
            continue;
        }
        const std::string& req = req_headers;

        // request line: METHOD PATH HTTP/x
        const auto sp1 = req.find(' ');
        const auto sp2 = req.find(' ', sp1 + 1);
        const std::string method =
            sp1 == std::string::npos ? "" : req.substr(0, sp1);
        const std::string path =
            (sp1 == std::string::npos || sp2 == std::string::npos)
                ? "" : req.substr(sp1 + 1, sp2 - sp1 - 1);

        // ---- auth gate (Phase 8) ------------------------------------------
        // / and /health stay open (the console must load before it can ask
        // for the token; probes must not need secrets). Everything that
        // computes or mutates is protected when a token is configured.
        const bool is_open = (path == "/" || path == "/index.html" ||
                              path == "/health");
        if (!is_open && !bearer_authorized(req, api_token)) {
            // Browsers cannot attach headers to a WebSocket handshake, so
            // /ws also accepts ?token=<OMNISEED_API_TOKEN> (constant-time).
            bool ws_ok = false;
            if (!api_token.empty() && path.compare(0, 3, "/ws") == 0 &&
                (path.size() == 3 || path[3] == '?')) {
                const size_t q = path.find("token=");
                if (q != std::string::npos) {
                    std::string v = path.substr(q + 6);
                    const size_t amp = v.find('&');
                    if (amp != std::string::npos) v.erase(amp);
                    ws_ok = token_equal(v, api_token);
                }
            }
            if (!ws_ok) {
                respond(client,
                        "{\"error\":\"unauthorized\",\"hint\":\"Authorization: "
                        "Bearer <OMNISEED_API_TOKEN>\"}",
                        "application/json", "401 Unauthorized",
                        "WWW-Authenticate: Bearer\r\n");
                close_socket(client);
                continue;
            }
        }

        if (path == "/" || path == "/index.html") {
            respond(client, demo_page(), "text/html; charset=utf-8");
        } else if (path == "/health") {
            std::string b = std::string("{\"ok\":true,\"model\":") +
                            (have_model ? "true" : "false") +
                            ",\"assistant_lora\":" +
                            (server_lora ? "true" : "false") +
                            ",\"asr\":" +
                            (server_whisper
                                 ? (server_whisper->decoder_loaded()
                                        ? "\"real\"" : "\"fallback\"")
                                 : "false") +
                            ",\"auth\":" +
                            (api_token.empty() ? "false" : "true") +
                            ",\"paused\":" +
                            (g_server_paused ? "true" : "false") +
                            ",\"decision_mode\":\"" +
                            decision_mode_name(cfg.decision_mode) + "\"" +
                            ",\"decision_head\":" +
                            (loop.decision_head().ready()
                                 ? (loop.decision_head().trained()
                                        ? "\"fitted\"" : "\"untrained\"")
                                 : "false") +
                            ",\"peak_rss\":" +
                            std::to_string(platform::peak_rss_bytes() / 1048576) +
                            "}";
            respond(client, b);
        } else if (path == "/control" && method == "POST") {
            // The stop toggle (console + API): pause/refuse new generations
            // WITHOUT killing the process — an in-flight generation finishes,
            // new /gen and /ask requests get {"error":"paused"} until resume.
            const std::string action = json_field(body, "action");
            if (action == "pause") {
                g_server_paused.store(true);
                platform::log_info("server PAUSED via /control");
            } else if (action == "resume") {
                g_server_paused.store(false);
                platform::log_info("server resumed via /control");
            } else if (action != "status") {
                respond(client,
                        "{\"error\":\"unknown-action\",\"expected\":"
                        "[\"pause\",\"resume\",\"status\"]}");
                close_socket(client);
                continue;
            }
            respond(client, std::string("{\"paused\":") +
                            (g_server_paused ? "true" : "false") + "}");
        } else if (path == "/classify" && method == "POST") {
            // T112 — the runtime consumer for fitted ClassificationHead blobs
            // (tools/train_heads.py output). Chosen as the ONE call site over a
            // CLI subcommand because it needs NO model (the client supplies
            // h[E] directly), rides the existing JSON-body + bearer-auth
            // machinery, and FittedHeads' lazy per-bin cache makes it ~free
            // after the first request. Pause does NOT gate it: it classifies,
            // never generates. The HTTP loop is serial, so a function-local
            // static FittedHeads is race-free by construction.
            static FittedHeads fitted;
            const std::string set = json_field(body, "set");
            std::vector<float> hidden;
            const bool got_hidden = json_float_array(body, "hidden", hidden);
            // Optional override (ops/tests); empty keeps the default dir.
            const std::string want_dir = json_field(body, "heads_dir");
            if (!want_dir.empty() && fitted.heads_dir() != want_dir)
                fitted.set_heads_dir(want_dir);  // drops caches; re-reads
            if (set.empty() || !got_hidden || hidden.empty()) {
                respond(client,
                        "{\"error\":\"bad-request\",\"hint\":\"POST "
                        "{\\\"set\\\":\\\"vision.scene\\\",\\\"hidden\\\":[...E "
                        "floats...]}; optional: top_k, heads_dir\"}");
                close_socket(client);
                continue;
            }
            int32_t top_k = static_cast<int32_t>(json_num_field(body, "top_k", 3.0f));
            if (top_k <= 0) top_k = 3;
            // caller_E = hidden.size(): the width guard — a blob fitted at
            // E=768 fed a 512-wide vector is a REFUSAL, never a silently
            // wrong distribution.
            const ClassificationResult r =
                fitted.classify(hidden.data(), set, top_k,
                                static_cast<int32_t>(hidden.size()));
            if (r.top_k.empty()) {
                // Head refused (missing/corrupt blob, unmapped set, E
                // mismatch). domain carries the reason in the repo's
                // "<error: ...>" form; top_k stays empty by contract.
                respond(client, std::string("{\"error\":\"not-available\","
                                            "\"reason\":\"") +
                                json_escape(r.domain) + "\"}");
                close_socket(client);
                continue;
            }
            respond(client, r.to_json());
        } else if (path == "/ask" && method == "POST") {
            // Pause gate BEFORE the model check: a paused server answers
            // {"error":"paused"} even when no GGUF is loaded (the
            // model-less omniseed_server_smoke asserts this contract).
            if (g_server_paused.load()) {
                respond(client, "{\"error\":\"paused\",\"paused\":true}");
                close_socket(client);
                continue;
            }
            if (!have_model) {
                respond(client,
                        "{\"error\":\"no-model\",\"hint\":\"restart with "
                        "--model <gguf>\"}");
                close_socket(client);
                continue;
            }
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
            // Optional per-request stop strings (mirrors the CLI): absent
            // fields leave stops unchanged; "stop": [] reverts to defaults.
            loop.set_stop_strings(request_stops(body));
            const AgentLoop::Result r = loop.run(task);
            // Additive fields: the decision head's verdict for this turn (when
            // it was consulted) and whether System-1 answered on its own.
            std::string b = "{\"reply\":\"" + json_escape(r.reply) + "\"";
            if (r.decision_checked) {
                b += ",\"decision\":" + r.decision.to_json();
                b += ",\"fast_path\":";
                b += r.decision_fast_path ? "true" : "false";
            }
            b += "}";
            respond(client, b);
        } else if (path == "/gen" && method == "POST") {
            // Pause gate BEFORE the model check (see /ask above).
            if (g_server_paused.load()) {
                respond(client, "{\"error\":\"paused\",\"paused\":true}");
                close_socket(client);
                continue;
            }
            if (!have_model) {
                respond(client,
                        "{\"error\":\"no-model\",\"hint\":\"restart with "
                        "--model <gguf>\"}");
                close_socket(client);
                continue;
            }
            const std::string prompt = json_field(body, "prompt");
            loop.set_sampling(json_num_field(body, "repeat_penalty", 1.0f),
                              static_cast<int32_t>(
                                  json_num_field(body, "repeat_window", 64.0f)));
            const std::string want = json_field(body, "assistant_lora");
            if (!want.empty() || body.find("\"assistant_lora\"")
                                 != std::string::npos)
                set_server_lora(model, server_lora, want);
            // Optional per-request stop strings (mirrors the CLI): absent
            // fields leave stops unchanged; "stop": [] reverts to defaults.
            loop.set_stop_strings(request_stops(body));
            auto ids = tok.encode_chat(prompt);
            RwkvState st;
            model.init_state(st);
            Tensor logits("logits", {model.config().n_vocab}, DType::F32);
            for (const int32_t id : ids) model.forward(id, st, logits);
            const int32_t seed = ids.empty() ? Tokenizer::kBosId : ids.back();
            const std::string out =
                loop.generate(st, seed, 160, {Tokenizer::kEosId}, nullptr);
            respond(client, "{\"text\":\"" + out + "\"}");
        } else if (path == "/asr" && method == "POST") {
            // `body` was already fully read by read_request above.
            if (!server_asr_ready()) {
                respond(client,
                        "{\"error\":\"whisper sidecar unavailable\"}");
            } else {
                // Body: raw WAV bytes, or JSON {"wav_b64": "..."}.
                std::string wav;
                const bool json = !body.empty() && body[0] == '{';
                if (json) {
                    if (!base64_decode(json_field(body, "wav_b64"), wav)) {
                        respond(client,
                                "{\"error\":\"invalid base64 payload\"}");
                        server_asr_done();   // T123b: release before continue
                        close_socket(client);
                        continue;
                    }
                } else {
                    wav = std::move(body);
                }
                PcmAudio audio;
                if (!PcmAudio::load_wav_bytes(wav, audio)) {
                    respond(client, "{\"error\":\"bad WAV payload\"}");
                } else {
                    const std::clock_t t0 = std::clock();
                    std::string text;
                    const bool ok = server_whisper->transcribe(audio, text);
                    const double secs = static_cast<double>(
                        std::clock() - t0) / CLOCKS_PER_SEC;
                    if (!ok) {
                        // Clean, quoted error (silence/noise/too-short).
                        std::string e;
                        for (const char c : server_whisper->error()) {
                            if (c == '"' || c == '\\') e.push_back('\\');
                            if (static_cast<unsigned char>(c) >= 0x20)
                                e.push_back(c);
                        }
                        respond(client, "{\"error\":\"" + e + "\"}");
                    } else {
                        std::string q;
                        q.reserve(text.size() + 8);
                        for (const char c : text) {   // JSON-safe quote
                            if (c == '"' || c == '\\') q.push_back('\\');
                            if (static_cast<unsigned char>(c) >= 0x20 ||
                                c == '\t')
                                q.push_back(c);
                        }
                        respond(client,
                                "{\"text\":\"" + q + "\",\"seconds\":" +
                                std::to_string(secs) + "}");
                    }
                }
                server_asr_done();   // T123b: drop hub ref at request end
            }
        } else if (path.compare(0, 3, "/ws") == 0 &&
                   (path.size() == 3 || path[3] == '?')) {
            // T103 live updates. Auth was enforced by the gate above
            // (Bearer header or ?token=). Upgrade + hand the socket to the
            // WS thread; the HTTP loop moves on immediately.
            const std::string key = header_value(req, "sec-websocket-key");
            std::string up = header_value(req, "upgrade");
            std::transform(up.begin(), up.end(), up.begin(),
                           [](unsigned char ch) {
                               return static_cast<char>(std::tolower(ch));
                           });
            if (key.empty() || up != "websocket") {
                respond(client,
                        "{\"error\":\"not-a-websocket-upgrade\","
                        "\"hint\":\"GET /ws with Upgrade: websocket\"}");
                close_socket(client);
                continue;
            }
            const std::string hdr101 =
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                "Sec-WebSocket-Accept: " +
                omniseed::server::ws::accept_key(key) + "\r\n\r\n";
            if (!send_all(client, hdr101.data(), hdr101.size())) {
                close_socket(client);
                continue;
            }
            ws_adopt(client);  // owned by the WS thread from here on
            continue;
        } else {
            respond(client, "{\"error\":\"not found\"}");
        }

        close_socket(client);
    }

#ifdef _WIN32
    WSACleanup();
#endif
    return 0;
}

#endif // OMNISEED_HTTP
