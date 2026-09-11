// =============================================================================
//  OmniSeed — trading/broker_alpaca.cpp
//  Alpaca REST client. Transport is injectable; the production transport
//  posts through the user-provided TLS relay (no TLS in the runtime).
// =============================================================================
#include "omniseed/trading/broker_alpaca.h"
#include "omniseed/runtime/cloud_bridge.h"
#include "omniseed/core/platform.h"

#include <cstdio>
#include <cstdlib>
#include <map>

namespace omniseed {
namespace trading {

AlpacaClient::AlpacaClient(const Config& cfg, Transport transport)
    : cfg_(cfg), transport_(std::move(transport)) {
    key_id_ = cfg_.key_id.empty()
        ? (std::getenv("ALPACA_API_KEY_ID")
               ? std::getenv("ALPACA_API_KEY_ID") : "")
        : cfg_.key_id;
    secret_key_ = cfg_.secret_key.empty()
        ? (std::getenv("ALPACA_API_SECRET_KEY")
               ? std::getenv("ALPACA_API_SECRET_KEY") : "")
        : cfg_.secret_key;
}

// ===========================================================================
// JSON helpers (flat objects — enough for Alpaca's account/order payloads)
// ===========================================================================
std::string AlpacaClient::json_string(const std::string& body,
                                      const std::string& key) {
    const size_t k = body.find("\"" + key + "\"");
    if (k == std::string::npos) return "";
    const size_t c = body.find(':', k);
    if (c == std::string::npos) return "";
    const size_t q1 = body.find('"', c);
    if (q1 == std::string::npos) return "";
    const size_t q2 = body.find('"', q1 + 1);
    if (q2 == std::string::npos) return "";
    return body.substr(q1 + 1, q2 - q1 - 1);
}

double AlpacaClient::json_number(const std::string& body,
                                 const std::string& key, double dflt) {
    const size_t k = body.find("\"" + key + "\"");
    if (k == std::string::npos) return dflt;
    const size_t c = body.find(':', k);
    if (c == std::string::npos) return dflt;
    // Alpaca sends numerics as JSON strings ("61234.56") — accept both
    // quoted and bare forms.
    size_t v = c + 1;
    while (v < body.size() && (body[v] == ' ' || body[v] == '\\t')) ++v;
    if (v < body.size() && body[v] == '"') {
        ++v;
        const char* begin = body.c_str() + v;
        char* end = nullptr;
        const double out = std::strtod(begin, &end);
        return end == begin ? dflt : out;
    }
    const char* begin = body.c_str() + v;
    char* end = nullptr;
    const double out = std::strtod(begin, &end);
    return end == begin ? dflt : out;
}

// ===========================================================================
// Transport
// ===========================================================================
bool AlpacaClient::request(const std::string& method, const std::string& path,
                           const std::string& body, int& status,
                           std::string& response) {
    err_.clear();
    if (!valid()) {
        err_ = "Alpaca keys not configured (ALPACA_API_KEY_ID / "
               "ALPACA_API_SECRET_KEY)";
        return false;
    }
    if (transport_) return transport_(method, path, body, status, response);

    // Production transport: HTTPS relay + Alpaca key headers.
    const std::string base = cfg_.endpoint == Endpoint::Live
        ? "https://api.alpaca.markets" : "https://paper-api.alpaca.markets";
    std::map<std::string, std::string> headers = {
        {"APCA-API-KEY-ID", key_id_},
        {"APCA-API-SECRET-KEY", secret_key_},
    };
    return CloudBridge::request_json(method, base + path, body, headers, 20,
                                     response, status, err_);
}

// ===========================================================================
// Order validation (pure, no network)
// ===========================================================================
bool AlpacaClient::validate_order(const std::string& symbol, double qty,
                                  const std::string& side,
                                  std::string& why_not) {
    if (symbol.empty() || symbol.size() > 5) {
        why_not = "symbol must be 1-5 characters";
        return false;
    }
    for (const char c : symbol)
        if (!((c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.')) {
            why_not = "symbol must be uppercase alnum";
            return false;
        }
    if (!(qty > 0.0) || qty > 1e9) {
        why_not = "qty must be positive and sane";
        return false;
    }
    if (side != "buy" && side != "sell") {
        why_not = "side must be buy or sell";
        return false;
    }
    return true;
}

// ===========================================================================
// API calls
// ===========================================================================
bool AlpacaClient::get_account(Account& out) {
    int status = 0;
    std::string response;
    if (!request("GET", "/v2/account", "", status, response)) return false;
    if (status != 200) {
        err_ = "account status " + std::to_string(status);
        return false;
    }
    out.cash = json_number(response, "cash");
    out.equity = json_number(response, "equity");
    out.buying_power = json_number(response, "buying_power");
    out.status = json_string(response, "status");
    return true;
}

bool AlpacaClient::get_positions(std::vector<BrokerPosition>& out) {
    out.clear();
    int status = 0;
    std::string response;
    if (!request("GET", "/v2/positions", "", status, response)) return false;
    if (status != 200) {
        err_ = "positions status " + std::to_string(status);
        return false;
    }
    // Split the JSON array on "symbol" boundaries — enough for display.
    size_t pos = 0;
    for (;;) {
        const size_t sym = response.find("\"symbol\"", pos);
        if (sym == std::string::npos) break;
        const size_t next = response.find("\"symbol\"", sym + 1);
        const std::string chunk = response.substr(
            sym, next == std::string::npos ? std::string::npos : next - sym);
        BrokerPosition p;
        p.symbol = json_string(chunk, "symbol");
        p.qty = json_number(chunk, "qty");
        p.avg_entry_price = json_number(chunk, "avg_entry_price");
        p.current_price = json_number(chunk, "current_price");
        p.unrealized_pl = json_number(chunk, "unrealized_pl");
        if (!p.symbol.empty()) out.push_back(p);
        pos = sym + 8;
    }
    return true;
}

bool AlpacaClient::submit_market_order(const std::string& symbol, double qty,
                                       const std::string& side,
                                       std::string& broker_order_id) {
    broker_order_id.clear();
    std::string why;
    if (!validate_order(symbol, qty, side, why)) {
        err_ = "order rejected locally: " + why;
        return false;
    }
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "{\"symbol\":\"%s\",\"qty\":%.0f,\"side\":\"%s\","
                  "\"type\":\"market\",\"time_in_force\":\"day\"}",
                  symbol.c_str(), qty, side.c_str());
    int status = 0;
    std::string response;
    if (!request("POST", "/v2/orders", buf, status, response)) return false;
    if (status != 200 && status != 201) {
        err_ = "order status " + std::to_string(status);
        return false;
    }
    broker_order_id = json_string(response, "id");
    return true;
}

} // namespace trading
} // namespace omniseed
