// =============================================================================
//  OmniSeed — server/main.cpp
//  Optional HTTP server (OMNISEED_HTTP). Vendored minimal socket loop —
//  no external dependencies. Endpoints:
//    GET  /health         -> {"ok":true}
//    POST /ask            -> one agent turn (json body: {"task": "..."})
//    POST /gen            -> raw generation (json body: {"prompt": "..."})
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

} // namespace

int main(int argc, char** argv) {
    int port = kPort;
    std::string model_path = "./models/omniseed.gguf";
    // Documented deployment env (see Dockerfile): OMNISEED_MODEL is the
    // fallback default; an explicit --model flag wins.
    if (const char* env = std::getenv("OMNISEED_MODEL")) model_path = env;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--port") == 0 && i + 1 < argc)
            port = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc)
            model_path = argv[++i];
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

        if (path == "/health") {
            std::string b = std::string("{\"ok\":true,\"model\":") +
                            (have_model ? "true" : "false") +
                            ",\"peak_rss\":" +
                            std::to_string(platform::peak_rss_bytes() / 1048576) +
                            "}";
            respond(client, b);
        } else if (path == "/ask" && method == "POST" && have_model) {
            const std::string task = json_field(body, "task");
            const AgentLoop::Result r = loop.run(task);
            respond(client, "{\"reply\":\"" + r.reply + "\"}");
        } else if (path == "/gen" && method == "POST" && have_model) {
            const std::string prompt = json_field(body, "prompt");
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
