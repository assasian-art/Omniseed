// =============================================================================
//  OmniSeed — trading/reasoning_tools.h
//  Advanced reasoning tools (Phase-16 Phase 3) — registered into the agent's
//  ToolRegistry so every sub-agent (and AgentLoop) can invoke them.
//
//    * web_search       — DuckDuckGo HTML scrape via the CloudBridge HTTP
//                         shim; `fetcher` is injectable for offline tests
//    * finance_calc     — the FinanceInterpreter exposed as a tool
//    * code_interpreter — sandboxed FinanceInterpreter alias (no arbitrary
//                         code execution BY DESIGN; see TRADING_GUIDE)
//    * document_reader  — strips HTML tags/entities to readable text; future
//                         PDF support arrives with a vendored parser
// =============================================================================
#pragma once

#include "omniseed/agent/agent.h"      // ToolRegistry

#include <functional>
#include <string>

namespace omniseed {
namespace trading {

// Fetch function injection: (url, response, err) -> ok. Tests inject canned
// responses; production wires CloudBridge::get_text.
using WebFetcher = std::function<bool(const std::string&, std::string&,
                                      std::string&)>;

// Registers all reasoning tools into `reg`. fetcher may be null (production
// default wires the CloudBridge shim; tests pass a canned fetcher).
void register_reasoning_tools(ToolRegistry& reg, WebFetcher fetcher = nullptr);

// --- pieces exposed for tests ----------------------------------------------
// DuckDuckGo HTML -> up to `max` {title, url, snippet} strings ("title|url").
size_t parse_ddg_results(const std::string& html,
                         std::vector<std::string>& out, size_t max);

// HTML -> text: strip <script>/<style> blocks, tags, collapse whitespace,
// decode common entities.
std::string html_to_text(const std::string& html);

// URL validity for the fetch path (http/https only, host required).
bool is_fetchable_url(const std::string& url);

} // namespace trading
} // namespace omniseed
