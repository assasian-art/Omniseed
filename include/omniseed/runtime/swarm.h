// =============================================================================
//  OmniSeed — swarm.h
//  Collaborative Swarm Protocol (capability #7; features #43, #44, #55-58).
//
//  Blueprint:
//    * Multiple OmniSeed devices on a LAN form a decentralized swarm. Each
//      node publishes capabilities (has_model, has_vision, free RAM) and can
//      OFFLOAD a task ("delegate") to the best-suited peer, or serve incoming
//      work. Coordination is gossip-style with no central server.
//    * Only compressed artifacts cross the wire: task keys, crystal summaries,
//      skill recipes (spec #43: shared compressed memory; #44: cross-device
//      experience transfer). Raw sensor data never leaves a device.
//    * Transport is pluggable: an in-process loopback "mesh" is provided for
//      tests and single-host multi-agent demos, and a UDP broadcast Beacon
//      (dependency-free sockets) handles discovery on a real LAN.
//    * All payloads are checksummed and versioned (spec #58: secure
//      inter-device communication is provided at minimum by integrity checks
//      + a shared-key XOR stream cipher (FNV-derived keystream) for payload
//      confidentiality on the LAN.
// =============================================================================
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace omniseed {

// ---------------------------------------------------------------------------
// Node identity + advertised capabilities
// ---------------------------------------------------------------------------
struct SwarmNode {
    std::string id;             // stable node id (identity token or hostname)
    std::string endpoint;       // "loopback" or "udp:IP:port"
    uint32_t    caps    = 0;    // bitfield of Cap bits
    uint64_t    free_ram_bytes = 0;
    double      load_01  = 0.0; // 0..1 busyness estimate
    uint64_t    last_seen_unix = 0;

    enum Cap : uint32_t {
        CapModel  = 1u << 0,    // has the RWKV weights loaded
        CapVision = 1u << 1,    // vision encoder available
        CapAudio  = 1u << 2,    // audio encoder available
        CapTools  = 1u << 3,    // tool runtime available
    };
    bool has(uint32_t bit) const { return (caps & bit) != 0; }
};

// ---------------------------------------------------------------------------
// Messages (versioned + checksummed envelopes)
// ---------------------------------------------------------------------------
struct SwarmMessage {
    uint32_t version = 1;
    enum class Type : uint32_t {
        Hello      = 1,   // presence + caps announcement
        TaskOffer  = 2,   // "can anyone take this?"
        TaskResult = 3,   // compressed result of an offloaded task
        MemoryXfer = 4,   // crystal summaries (feature #43)
        SkillXfer  = 5,   // flash-skill recipes (feature #44)
        Bye        = 6,
    };
    Type    type = Type::Hello;
    std::string from;         // sender node id
    std::string task_key;     // normalized task signature
    std::string payload;      // compressed body (task text, result, crystals)
    uint32_t    payload_crc = 0;
};

// ---------------------------------------------------------------------------
// Transport interface. OmniSeed ships two implementations:
//   * LoopbackMesh  — in-process bus (tests + multi-agent on one host)
//   * UdpBeacon     — LAN discovery + datagram exchange (no dependencies)
// ---------------------------------------------------------------------------
class SwarmTransport {
public:
    virtual ~SwarmTransport() = default;
    // Sends a message to a node endpoint ("" = broadcast).
    virtual bool send(const std::string& endpoint, const SwarmMessage& msg) = 0;
    // Polls for one inbound message; returns false when none pending.
    virtual bool poll(SwarmMessage& out, std::string& from_endpoint) = 0;
};

// ---------------------------------------------------------------------------
// In-process loopback mesh (shared bus between SwarmNode instances)
// ---------------------------------------------------------------------------
class LoopbackMesh : public SwarmTransport {
public:
    // A node registers a mailbox under its id.
    void join(const std::string& node_id);
    void leave(const std::string& node_id);

    bool send(const std::string& endpoint, const SwarmMessage& msg) override;
    bool poll(SwarmMessage& out, std::string& from_endpoint) override;

    static LoopbackMesh& instance();

private:
    LoopbackMesh() = default;
    std::map<std::string, std::vector<std::pair<SwarmMessage, std::string>>>
        mailboxes_;
};

// ---------------------------------------------------------------------------
// UDP beacon: periodic Hello broadcast + datagram send/poll (best effort)
// ---------------------------------------------------------------------------
class UdpBeacon : public SwarmTransport {
public:
    struct Config {
        uint16_t port        = 47470;  // swarm default port
        uint32_t ttl_seconds = 30;     // peer expiry
    };
    explicit UdpBeacon(const Config& cfg = {});
    ~UdpBeacon() override;

    bool start();                      // binds the socket; false if unavailable
    void stop();

    bool send(const std::string& endpoint, const SwarmMessage& msg) override;
    bool poll(SwarmMessage& out, std::string& from_endpoint) override;

    // Periodic broadcast of a Hello (call from the runtime's idle loop).
    void announce(const SwarmMessage& hello);

    bool ok() const { return fd_ >= 0; }

private:
    int   fd_ = -1;
    Config cfg_;
};

// ---------------------------------------------------------------------------
// Wire codec: messages are length-prefixed, CRC32-checked, and optionally
// encrypted with a shared swarm key (FNV-1a keystream XOR — lightweight,
// deterministic; replaces heavy crypto in the 300 MB budget envelope).
// ---------------------------------------------------------------------------
std::string encode_message(const SwarmMessage& msg, const std::string& key);
bool decode_message(const std::string& blob, const std::string& key,
                    SwarmMessage& out);

// ---------------------------------------------------------------------------
// SwarmCoordinator — the local node's protocol brain
// ---------------------------------------------------------------------------
class SwarmCoordinator {
public:
    struct Config {
        std::string node_id;
        uint32_t    caps         = SwarmNode::CapModel | SwarmNode::CapTools;
        uint64_t    ram_budget_bytes = 300ull * 1024 * 1024;
        double      peer_ttl_seconds = 30.0;
        std::string shared_key   = "omniseed-swarm-v1";
        bool        allow_remote_tasks = true;  // accept offloaded work
    };

    // delegate: called when a remote task should be executed locally.
    // Input: task text. Output: result text. Return false to refuse.
    using TaskExecutor = std::function<bool(const std::string& task,
                                            std::string& result)>;

    explicit SwarmCoordinator(const Config& cfg = {}) : cfg_(cfg) {}

    const std::string& id() const { return cfg_.node_id; }

    // --- Peer management ---------------------------------------------------
    void observe_peer(const SwarmNode& node);
    void forget_stale_peers(double now_unix_seconds);
    std::vector<SwarmNode> peers() const;
    size_t peer_count() const;

    // --- Offloading (features #55-57) --------------------------------------
    // Picks the best peer for a task and sends it a TaskOffer.
    // Scoring: required caps present, most free RAM, lowest load.
    bool delegate(const std::string& task_text, SwarmNode* chosen = nullptr);
    // Score a peer for a task; higher is better. Negative = unsuitable.
    static double score_peer(const SwarmNode& n, uint32_t required_caps);

    // --- Inbound handling (poll-driven; call from the runtime idle loop) ---
    // Returns true when a message was processed. If an executor is set and a
    // remote task arrives, it runs locally and the result is sent back.
    bool step(SwarmTransport& transport, TaskExecutor executor = nullptr);

    // --- Memory/skill transfer (features #43-44) ---------------------------
    // Publishes crystal summaries (already-compressed text lines) to peers.
    bool share_memories(SwarmTransport& transport,
                        const std::vector<std::string>& crystal_summaries);
    // Publishes a skill recipe ("name|trigger|op:expr;op:expr|args").
    bool share_skill(SwarmTransport& transport, const std::string& recipe_line);
    // Inbound memory/skill ingestion hooks (return count accepted).
    size_t ingest_memories(const std::vector<std::string>& crystal_summaries);
    size_t ingest_skills(const std::vector<std::string>& recipe_lines);

    const std::vector<std::string>& shared_memories() const { return memories_; }
    const std::vector<std::string>& shared_skills() const { return skills_; }

    const Config& config() const { return cfg_; }

private:
    Config cfg_;
    std::map<std::string, SwarmNode> peers_;
    std::vector<std::string> memories_;   // compressed crystal lines received
    std::vector<std::string> skills_;     // recipe lines received
};

} // namespace omniseed
