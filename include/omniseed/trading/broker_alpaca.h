// =============================================================================
//  OmniSeed — trading/broker_alpaca.h
//  Alpaca REST API client (Phase-16 Phase 4 — TASK I, OPTIONAL).
//
//  Safety model (encoded here, enforced by the CLI):
//    * Default mode is PAPER: https://paper-api.alpaca.markets with paper
//      keys — real order lifecycle, fake money.
//    * LIVE trading (https://api.alpaca.markets) requires BOTH the explicit
//      --live-trading CLI flag AND the confirm string "I ACCEPT REAL LOSS"
//      typed at the prompt. No defaults, no env shortcuts.
//    * Keys come ONLY from the environment: ALPACA_API_KEY_ID +
//      ALPACA_API_SECRET_KEY. The runtime never persists them.
//    * The client speaks HTTPS through the user-provided local TLS relay
//      (OMNISEED_CLOUD_PROXY) — the zero-dep runtime has no TLS stack. All
//      HTTP paths are mockable for tests (fetcher injection), so the full
//      order-validation logic is testable offline.
// =============================================================================
#pragma once

#include <functional>
#include <map>
#include <string>
#include <vector>

namespace omniseed {
namespace trading {

class AlpacaClient {
public:
    enum class Endpoint { Paper, Live };

    struct Config {
        Endpoint endpoint = Endpoint::Paper;   // PAPER BY DEFAULT
        std::string key_id;                    // empty -> env ALPACA_API_KEY_ID
        std::string secret_key;                // empty -> env ALPACA_API_SECRET_KEY
    };

    // Transport injection for tests: (method, path, body) -> (ok, status,
    // response). Production wires CloudBridge::post_json/get_text through
    // the TLS relay.
    using Transport = std::function<bool(const std::string& /*method*/,
                                         const std::string& /*path*/,
                                         const std::string& /*body*/,
                                         int& /*status*/,
                                         std::string& /*response*/)>;

    explicit AlpacaClient(const Config& cfg = {}, Transport transport = nullptr);

    bool valid() const { return !key_id_.empty() && !secret_key_.empty(); }
    bool is_paper() const { return cfg_.endpoint == Endpoint::Paper; }
    const std::string& error() const { return err_; }

    // GET /v2/account — cash, equity, buying power.
    struct Account {
        double cash = 0.0, equity = 0.0, buying_power = 0.0;
        std::string status;               // "ACTIVE", ...
    };
    bool get_account(Account& out);

    // GET /v2/positions — open positions.
    struct BrokerPosition {
        std::string symbol;
        double qty = 0.0, avg_entry_price = 0.0, current_price = 0.0;
        double unrealized_pl = 0.0;
    };
    bool get_positions(std::vector<BrokerPosition>& out);

    // POST /v2/orders — market order (day, qty). Validates locally FIRST
    // (symbol, qty > 0) so garbage never reaches the broker.
    bool submit_market_order(const std::string& symbol, double qty,
                             const std::string& side,   // "buy" | "sell"
                             std::string& broker_order_id);

    // Order validation shared by CLI + tests (no network).
    static bool validate_order(const std::string& symbol, double qty,
                               const std::string& side, std::string& why_not);

private:
    bool request(const std::string& method, const std::string& path,
                 const std::string& body, int& status, std::string& response);

    static double json_number(const std::string& body, const std::string& key,
                              double dflt = 0.0);
    static std::string json_string(const std::string& body,
                                   const std::string& key);

    Config cfg_;
    Transport transport_;                 // null -> real HTTP via relay
    std::string key_id_, secret_key_;
    std::string err_;
};

} // namespace trading
} // namespace omniseed
