// =============================================================================
//  OmniSeed — swarm.cpp
//  Collaborative Swarm Protocol implementation.
// =============================================================================
#include "omniseed/runtime/swarm.h"
#include "omniseed/core/platform.h"

#include <algorithm>
#include <cstring>
#include <mutex>

#if OMNISEED_PLATFORM_WINDOWS
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <winsock2.h>
    #include <ws2tcpip.h>
#else
    #include <arpa/inet.h>
    #include <fcntl.h>
    #include <netinet/in.h>
    #include <sys/socket.h>
    #include <unistd.h>
#endif

namespace omniseed {

// ===========================================================================
// Wire codec: magic | version | crc32(payload) | payload
// Optional obfuscation: XOR with FNV-1a keystream of the shared key.
// ===========================================================================
namespace {

uint32_t crc32_of(const std::string& s) {
    uint32_t crc = 0xFFFFFFFFu;
    for (char c : s) {
        crc ^= static_cast<uint8_t>(c);
        for (int k = 0; k < 8; ++k)
            crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
    }
    return crc ^ 0xFFFFFFFFu;
}

// Counter-based keystream: h advances from a key-derived seed AND the byte
// index only (never the data), so XOR is symmetric for encode/decode.
void keystream_xor(const std::string& key, std::string& data) {
    if (key.empty()) return;
    uint64_t h = 0xcbf29ce484222325ull;
    for (char c : key) { h ^= static_cast<uint8_t>(c); h *= 0x100000001b3ull; }
    for (size_t i = 0; i < data.size(); ++i) {
        h *= 0x100000001b3ull;
        h ^= i + 0x9E3779B97F4A7C15ull;
        data[i] = static_cast<char>(data[i] ^ static_cast<uint8_t>(h >> 32));
    }
}

std::string pack_u32(uint32_t v) {
    std::string s(4, '\0');
    s[0] = static_cast<char>(v & 0xFF);
    s[1] = static_cast<char>((v >> 8) & 0xFF);
    s[2] = static_cast<char>((v >> 16) & 0xFF);
    s[3] = static_cast<char>((v >> 24) & 0xFF);
    return s;
}

uint32_t unpack_u32(const char* p) {
    const auto* u = reinterpret_cast<const uint8_t*>(p);
    return static_cast<uint32_t>(u[0]) | (static_cast<uint32_t>(u[1]) << 8) |
           (static_cast<uint32_t>(u[2]) << 16) | (static_cast<uint32_t>(u[3]) << 24);
}

std::string pack_u64(uint64_t v) {
    std::string s = pack_u32(static_cast<uint32_t>(v & 0xFFFFFFFFu));
    return s + pack_u32(static_cast<uint32_t>(v >> 32));
}

uint64_t unpack_u64(const char* p) {
    return static_cast<uint64_t>(unpack_u32(p)) |
           (static_cast<uint64_t>(unpack_u32(p + 4)) << 32);
}

// Packet layout:
//   magic 'OMSW' | u32 version | u32 type | str(from) | str(task_key)
//   str(payload) | u32 payload_crc | u32 header_crc | u64 reserved
// header_crc covers everything before it (magic/version/type/lengths), so
// any tampering with the frame itself is caught too.
std::string serialize(const SwarmMessage& m) {
    std::string out = "OMSW";
    out += pack_u32(m.version);
    out += pack_u32(static_cast<uint32_t>(m.type));
    out += pack_u32(static_cast<uint32_t>(m.from.size())) + m.from;
    out += pack_u32(static_cast<uint32_t>(m.task_key.size())) + m.task_key;
    out += pack_u32(static_cast<uint32_t>(m.payload.size())) + m.payload;
    out += pack_u32(m.payload_crc);
    out += pack_u32(0);           // placeholder: header crc patched below
    out += pack_u64(0);           // reserved
    const uint32_t hcrc = crc32_of(out.substr(0, out.size() - 12));
    out[out.size() - 12] = static_cast<char>(hcrc & 0xFF);
    out[out.size() - 11] = static_cast<char>((hcrc >> 8) & 0xFF);
    out[out.size() - 10] = static_cast<char>((hcrc >> 16) & 0xFF);
    out[out.size() - 9] = static_cast<char>((hcrc >> 24) & 0xFF);
    return out;
}

bool parse(const char* p, size_t n, SwarmMessage& m) {
    if (n < 16 || std::memcmp(p, "OMSW", 4) != 0) return false;
    size_t cur = 4;
    m.version = unpack_u32(p + cur); cur += 4;
    if (m.version != 1) return false;
    m.type = static_cast<SwarmMessage::Type>(unpack_u32(p + cur)); cur += 4;
    if (m.type < SwarmMessage::Type::Hello || m.type > SwarmMessage::Type::Bye)
        return false;

    uint32_t len = unpack_u32(p + cur); cur += 4;
    if (cur + len > n) return false;
    m.from.assign(p + cur, len); cur += len;

    len = unpack_u32(p + cur); cur += 4;
    if (cur + len > n) return false;
    m.task_key.assign(p + cur, len); cur += len;

    len = unpack_u32(p + cur); cur += 4;
    if (cur + len > n) return false;
    m.payload.assign(p + cur, len); cur += len;

    if (cur + 4 > n) return false;
    m.payload_crc = unpack_u32(p + cur); cur += 4;
    // Header integrity: crc over everything before the header-crc field.
    const uint32_t hcrc = unpack_u32(p + cur); cur += 4;
    if (crc32_of(std::string(p, cur - 4)) != hcrc) return false;
    cur += 8;   // reserved
    return true;
}

} // namespace

std::string encode_message(const SwarmMessage& msg, const std::string& key) {
    SwarmMessage m = msg;
    m.payload_crc = crc32_of(m.payload);
    std::string body = serialize(m);
    keystream_xor(key, body);
    // Prepend a 4-byte length so datagram transports can frame it.
    return pack_u32(static_cast<uint32_t>(body.size())) + body;
}

bool decode_message(const std::string& blob, const std::string& key,
                    SwarmMessage& out) {
    if (blob.size() < 8) return false;
    const uint32_t declared = unpack_u32(blob.data());
    std::string body = blob.substr(4);
    if (declared != body.size()) return false;
    keystream_xor(key, body);
    SwarmMessage m;
    if (!parse(body.data(), body.size(), m)) return false;
    if (crc32_of(m.payload) != m.payload_crc) return false;   // integrity (58)
    out = m;
    return true;
}

// ===========================================================================
// LoopbackMesh
// ===========================================================================
namespace {
std::mutex g_mesh_mutex;
}

void LoopbackMesh::join(const std::string& node_id) {
    std::lock_guard<std::mutex> lock(g_mesh_mutex);
    mailboxes_[node_id];
}

void LoopbackMesh::leave(const std::string& node_id) {
    std::lock_guard<std::mutex> lock(g_mesh_mutex);
    mailboxes_.erase(node_id);
}

LoopbackMesh& LoopbackMesh::instance() {
    static LoopbackMesh mesh;
    return mesh;
}

bool LoopbackMesh::send(const std::string& endpoint, const SwarmMessage& msg) {
    std::lock_guard<std::mutex> lock(g_mesh_mutex);
    if (endpoint.empty()) {
        // Broadcast: fan out to every joined mailbox except the sender's own.
        bool any = false;
        for (auto& [id, q] : mailboxes_) {
            if (id == msg.from) continue;
            q.emplace_back(msg, msg.from);
            any = true;
        }
        return any;
    }
    auto it = mailboxes_.find(endpoint);
    if (it == mailboxes_.end()) return false;
    it->second.emplace_back(msg, msg.from);   // sender identity, not mailbox
    return true;
}

bool LoopbackMesh::poll(SwarmMessage& out, std::string& from_endpoint) {
    // NOTE: the loopback mesh is a shared bus; poll() drains the OLDEST
    // queued message across all mailboxes (single-threaded demo/test model).
    std::lock_guard<std::mutex> lock(g_mesh_mutex);
    for (auto& [id, q] : mailboxes_) {
        if (!q.empty()) {
            out = q.front().first;
            from_endpoint = q.front().second;
            q.erase(q.begin());
            (void)id;
            return true;
        }
    }
    return false;
}

// ===========================================================================
// UdpBeacon
// ===========================================================================
namespace {
constexpr const char* kSwarmKey = "omniseed-swarm-v1";
}

UdpBeacon::UdpBeacon(const Config& cfg) : cfg_(cfg) {}

UdpBeacon::~UdpBeacon() { stop(); }

bool UdpBeacon::start() {
    if (fd_ >= 0) return true;
#ifdef _WIN32
    fd_ = static_cast<int>(::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP));
    if (fd_ == INVALID_SOCKET) { fd_ = -1; return false; }
#else
    fd_ = static_cast<int>(::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP));
    if (fd_ < 0) return false;
#endif
    int one = 1;
    ::setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR,
                 reinterpret_cast<const char*>(&one), sizeof(one));
    ::setsockopt(fd_, SOL_SOCKET, SO_BROADCAST,
                 reinterpret_cast<const char*>(&one), sizeof(one));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(cfg_.port);
    if (::bind(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        stop();
        return false;
    }
    // Non-blocking poll.
#ifdef _WIN32
    u_long nb = 1;
    ::ioctlsocket(fd_, FIONBIO, &nb);
#else
    const int flags = ::fcntl(fd_, F_GETFL, 0);
    if (flags >= 0) ::fcntl(fd_, F_SETFL, flags | O_NONBLOCK);
#endif
    return true;
}

void UdpBeacon::stop() {
#ifdef _WIN32
    if (fd_ >= 0) ::closesocket(fd_);
#else
    if (fd_ >= 0) ::close(fd_);
#endif
    fd_ = -1;
}

bool UdpBeacon::send(const std::string& endpoint, const SwarmMessage& msg) {
    if (fd_ < 0) return false;
    const std::string blob = encode_message(msg, kSwarmKey);
    (void)endpoint;    // broadcast only in the beacon transport
    sockaddr_in dst{};
    dst.sin_family = AF_INET;
    dst.sin_addr.s_addr = INADDR_BROADCAST;
    dst.sin_port = htons(cfg_.port);
    const int n = static_cast<int>(::sendto(
        fd_, blob.data(), static_cast<int>(blob.size()), 0,
        reinterpret_cast<const sockaddr*>(&dst), sizeof(dst)));
    return n > 0;
}

bool UdpBeacon::poll(SwarmMessage& out, std::string& from_endpoint) {
    if (fd_ < 0) return false;
    char buf[4096];
    sockaddr_in src{};
#ifdef _WIN32
    int slen = sizeof(src);
    const int n = ::recvfrom(fd_, buf, sizeof(buf), 0,
                             reinterpret_cast<sockaddr*>(&src), &slen);
#else
    socklen_t slen = sizeof(src);
    const ssize_t n = ::recvfrom(fd_, buf, sizeof(buf), 0,
                                 reinterpret_cast<sockaddr*>(&src), &slen);
#endif
    if (n <= 0) return false;
    SwarmMessage m;
    if (!decode_message(std::string(buf, static_cast<size_t>(n)),
                        kSwarmKey, m)) return false;
    out = m;
    char ip[64];
    ::inet_ntop(AF_INET, &src.sin_addr, ip, sizeof(ip));
    from_endpoint = std::string("udp:") + ip + ":" +
                    std::to_string(ntohs(src.sin_port));
    return true;
}

void UdpBeacon::announce(const SwarmMessage& hello) {
    send("", hello);
}

// ===========================================================================
// SwarmCoordinator
// ===========================================================================
void SwarmCoordinator::observe_peer(const SwarmNode& node) {
    if (node.id.empty() || node.id == cfg_.node_id) return;
    SwarmNode n = node;
    n.last_seen_unix = static_cast<uint64_t>(platform::now_ms() / 1000.0);
    peers_[n.id] = n;
}

void SwarmCoordinator::forget_stale_peers(double now_unix_seconds) {
    const double cutoff = now_unix_seconds - cfg_.peer_ttl_seconds;
    for (auto it = peers_.begin(); it != peers_.end();) {
        if (static_cast<double>(it->second.last_seen_unix) < cutoff)
            it = peers_.erase(it);
        else
            ++it;
    }
}

std::vector<SwarmNode> SwarmCoordinator::peers() const {
    std::vector<SwarmNode> out;
    out.reserve(peers_.size());
    for (const auto& [id, n] : peers_) { (void)id; out.push_back(n); }
    return out;
}

size_t SwarmCoordinator::peer_count() const { return peers_.size(); }

double SwarmCoordinator::score_peer(const SwarmNode& n, uint32_t required_caps) {
    if ((n.caps & required_caps) != required_caps) return -1.0;  // unsuitable
    double s = 0.0;
    const uint64_t mb = n.free_ram_bytes / (1024 * 1024);
    s += std::min(40.0, static_cast<double>(mb) / 25.0);   // up to 40 pts RAM
    s += (1.0 - std::min(1.0, n.load_01)) * 30.0;          // up to 30 pts idle
    s += 10.0;                                             // base
    return s;
}

bool SwarmCoordinator::delegate(const std::string& task_text,
                                SwarmNode* chosen) {
    if (peers_.empty()) return false;
    // Required caps: model (it is an LLM task by definition).
    const uint32_t required = SwarmNode::CapModel;
    const SwarmNode* best = nullptr;
    double best_score = -1.0;
    for (const auto& [id, n] : peers_) {
        (void)id;
        const double s = score_peer(n, required);
        if (s > best_score) { best_score = s; best = &n; }
    }
    if (!best || best_score <= 0.0) return false;
    if (chosen) *chosen = *best;

    // Send via loopback mesh by default (beacon transports wrap this at the
    // runtime layer); endpoint addressing is the peer's id in the mesh.
    SwarmMessage m;
    m.type = SwarmMessage::Type::TaskOffer;
    m.from = cfg_.node_id;
    m.payload = task_text;
    // Loopback mesh addressing: peers register their mailbox under their
    // node id (endpoint format "udp:ip:port" is only used by real sockets).
    const std::string mailbox =
        best->endpoint.rfind("udp:", 0) == 0 ? best->endpoint : best->id;
    return LoopbackMesh::instance().send(mailbox, m);
}

bool SwarmCoordinator::step(SwarmTransport& transport, TaskExecutor executor) {
    SwarmMessage m;
    std::string from;
    if (!transport.poll(m, from)) return false;

    switch (m.type) {
        case SwarmMessage::Type::Hello: {
            SwarmNode n;
            n.id = m.from;
            n.endpoint = from.empty() ? m.from : from;
            n.caps = static_cast<uint32_t>(unpack_u64(m.payload.data()));
            observe_peer(n);
            return true;
        }
        case SwarmMessage::Type::TaskOffer: {
            if (!cfg_.allow_remote_tasks || !executor) return true;
            std::string result;
            const bool ok = executor(m.payload, result);
            SwarmMessage r;
            r.type = SwarmMessage::Type::TaskResult;
            r.from = cfg_.node_id;
            r.task_key = m.task_key;
            r.payload = ok ? result : "";
            transport.send(from.empty() ? m.from : from, r);
            return true;
        }
        case SwarmMessage::Type::TaskResult: {
            // The delegating node consumes results via its own mailbox.
            return true;
        }
        case SwarmMessage::Type::MemoryXfer: {
            // Payload: newline-separated crystal summaries.
            std::vector<std::string> lines;
            size_t start = 0;
            while (start < m.payload.size()) {
                const auto nl = m.payload.find('\n', start);
                if (nl == std::string::npos) {
                    lines.push_back(m.payload.substr(start));
                    break;
                }
                lines.push_back(m.payload.substr(start, nl - start));
                start = nl + 1;
            }
            ingest_memories(lines);
            return true;
        }
        case SwarmMessage::Type::SkillXfer: {
            std::vector<std::string> lines;
            size_t start = 0;
            while (start < m.payload.size()) {
                const auto nl = m.payload.find('\n', start);
                if (nl == std::string::npos) {
                    lines.push_back(m.payload.substr(start));
                    break;
                }
                lines.push_back(m.payload.substr(start, nl - start));
                start = nl + 1;
            }
            ingest_skills(lines);
            return true;
        }
        case SwarmMessage::Type::Bye:
            peers_.erase(m.from);
            return true;
    }
    return false;
}

bool SwarmCoordinator::share_memories(
        SwarmTransport& transport,
        const std::vector<std::string>& crystal_summaries) {
    if (crystal_summaries.empty()) return false;
    SwarmMessage m;
    m.type = SwarmMessage::Type::MemoryXfer;
    m.from = cfg_.node_id;
    for (size_t i = 0; i < crystal_summaries.size(); ++i) {
        if (i) m.payload += "\n";
        m.payload += crystal_summaries[i];
    }
    // Prefer addressed delivery to known peers; fall back to broadcast.
    if (!peers_.empty()) {
        bool sent = false;
        for (const auto& [id, n] : peers_) {
            (void)id;
            const std::string mailbox =
                n.endpoint.rfind("udp:", 0) == 0 ? n.endpoint : n.id;
            sent = transport.send(mailbox, m) || sent;
        }
        return sent;
    }
    return transport.send("", m);   // broadcast
}

bool SwarmCoordinator::share_skill(SwarmTransport& transport,
                                   const std::string& recipe_line) {
    SwarmMessage m;
    m.type = SwarmMessage::Type::SkillXfer;
    m.from = cfg_.node_id;
    m.payload = recipe_line;
    return transport.send("", m);
}

size_t SwarmCoordinator::ingest_memories(
        const std::vector<std::string>& crystal_summaries) {
    size_t added = 0;
    for (const std::string& s : crystal_summaries) {
        if (s.empty()) continue;
        if (std::find(memories_.begin(), memories_.end(), s) == memories_.end()) {
            memories_.push_back(s);
            ++added;
        }
    }
    return added;
}

size_t SwarmCoordinator::ingest_skills(const std::vector<std::string>& lines) {
    size_t added = 0;
    for (const std::string& s : lines) {
        if (s.empty()) continue;
        if (std::find(skills_.begin(), skills_.end(), s) == skills_.end()) {
            skills_.push_back(s);
            ++added;
        }
    }
    return added;
}

} // namespace omniseed
