// =============================================================================
//  OmniSeed — tool_registry.cpp
//  Tool schemas + validation + builtin tools (calculator, echo, time, mem).
// =============================================================================
#include "omniseed/agent/agent.h"
#include "omniseed/core/platform.h"

#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

namespace omniseed {

// ===========================================================================
// Tiny safe arithmetic evaluator: + - * / ( ) and numbers only.
// ===========================================================================
namespace calc {
struct Parser {
    const std::string& s;
    size_t pos = 0;
    explicit Parser(const std::string& str) : s(str) {}

    void skip_ws() {
        while (pos < s.size() && s[pos] == ' ') ++pos;
    }

    double primary() {
        skip_ws();
        if (pos < s.size() && s[pos] == '(') {
            ++pos;
            const double v = expression();
            skip_ws();
            if (pos < s.size() && s[pos] == ')') ++pos;
            return v;
        }
        const size_t start = pos;
        while (pos < s.size() &&
               (std::isdigit(static_cast<unsigned char>(s[pos])) ||
                s[pos] == '.')) ++pos;
        return std::atof(s.substr(start, pos - start).c_str());
    }

    double term() {
        double v = primary();
        for (;;) {
            skip_ws();
            if (pos < s.size() && (s[pos] == '*' || s[pos] == '/')) {
                const char op = s[pos++];
                const double r = primary();
                v = (op == '*') ? v * r : (r != 0.0 ? v / r : 0.0);
            } else {
                break;
            }
        }
        return v;
    }

    double expression() {
        double v = term();
        for (;;) {
            skip_ws();
            if (pos < s.size() && (s[pos] == '+' || s[pos] == '-')) {
                const char op = s[pos++];
                const double r = term();
                v = (op == '+') ? v + r : v - r;
            } else {
                break;
            }
        }
        return v;
    }
};
} // namespace calc

// ===========================================================================
// Registration
// ===========================================================================
void ToolRegistry::add(Tool tool) {
    tools_.push_back(std::move(tool));
}

bool ToolRegistry::add_builtin(const std::string& name) {
    if (name == "calc") {
        Tool t;
        t.name = "calc";
        t.description = "Evaluate a simple arithmetic expression.";
        t.params = {{"expression", "string", "e.g. 2+2*10", true}};
        t.fn = [](const std::string& args, bool& ok) -> std::string {
            // args: {"expression": "..."} — find the value between quotes.
            ok = true;
            const auto cpos = args.find(":");
            if (cpos == std::string::npos) { ok = false; return "{}"; }
            const auto q1 = args.find('"', cpos);
            const auto q2 = args.find('"', q1 + 1);
            if (q1 == std::string::npos || q2 == std::string::npos) {
                ok = false;
                return "{\"error\":\"missing expression\"}";
            }
            const std::string expr = args.substr(q1 + 1, q2 - q1 - 1);
            calc::Parser parser(expr);
            const double result = parser.expression();
            char buf[64];
            std::snprintf(buf, sizeof(buf), "{\"result\":%.6g}", result);
            return buf;
        };
        add(std::move(t));
        return true;
    }

    if (name == "echo") {
        Tool t;
        t.name = "echo";
        t.description = "Echo text back (test tool).";
        t.params = {{"text", "string", "text to echo", true}};
        t.fn = [](const std::string& args, bool& ok) {
            ok = true;
            const auto p = args.find(":");
            const auto q1 = args.find('"', p == std::string::npos ? 0 : p);
            const auto q2 = args.find('"', q1 + 1);
            if (q1 == std::string::npos || q2 == std::string::npos) {
                ok = false;
                return std::string("{\"error\":\"missing text\"}");
            }
            std::string out = "{\"echo\":\"" + args.substr(q1 + 1, q2 - q1 - 1) + "\"}";
            return out;
        };
        add(std::move(t));
        return true;
    }

    if (name == "time") {
        Tool t;
        t.name = "time";
        t.description = "Current local time and unix epoch.";
        t.fn = [](const std::string&, bool& ok) {
            ok = true;
            const auto now = std::chrono::system_clock::now();
            const std::time_t tt =
                std::chrono::system_clock::to_time_t(now);
            char mb[32];
            std::strftime(mb, sizeof(mb), "%Y-%m-%d %H:%M:%S",
                          std::localtime(&tt));
            return std::string("{\"time\":\"") + mb + "\",\"epoch\":" +
                   std::to_string(static_cast<long long>(tt)) + "}";
        };
        add(std::move(t));
        return true;
    }

    return false;
}

bool ToolRegistry::has(const std::string& name) const {
    return find(name) != nullptr;
}

const Tool* ToolRegistry::find(const std::string& name) const {
    for (const Tool& t : tools_)
        if (t.name == name) return &t;
    return nullptr;
}

// ===========================================================================
// Schemas JSON (Hermes/MCP-style)
// ===========================================================================
namespace {
std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (const char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char b[8];
                    std::snprintf(b, sizeof(b), "\\u%04X", c);
                    out += b;
                } else {
                    out += c;
                }
        }
    }
    return out;
}
} // namespace

std::string ToolRegistry::schemas_json() const {
    std::string out = "[";
    bool first = true;
    for (const Tool& t : tools_) {
        if (!first) out += ",";
        first = false;
        out += "{\"name\":\"" + json_escape(t.name) + "\",";
        out += "\"description\":\"" + json_escape(t.description) + "\",";
        out += "\"parameters\":{";
        bool fp = true;
        for (const ToolParam& p : t.params) {
            if (!fp) out += ",";
            fp = false;
            out += "\"" + json_escape(p.name) + "\":{";
            out += "\"type\":\"" + p.type + "\",";
            out += "\"description\":\"" + json_escape(p.description) + "\",";
            out += std::string("\"required\":") + (p.required ? "true" : "false");
            out += "}";
        }
        out += "}}";
    }
    out += "]";
    return out;
}

// ===========================================================================
// Validation + invocation
// ===========================================================================
bool ToolRegistry::validate(const std::string& name,
                            const std::string& args_json,
                            std::string& err) const {
    const Tool* t = find(name);
    if (t == nullptr) { err = "unknown tool: " + name; return false; }

    for (const ToolParam& p : t->params) {
        if (!p.required) continue;
        // naive but effective: required key must appear in args
        if (args_json.find("\"" + p.name + "\"") == std::string::npos) {
            err = "missing required param: " + p.name;
            return false;
        }
    }
    err.clear();
    return true;
}

std::string ToolRegistry::invoke(const std::string& name,
                                 const std::string& args_json,
                                 bool& ok) const {
    ok = false;
    const Tool* t = find(name);
    if (t == nullptr || !t->fn) {
        return "{\"error\":\"unknown tool\"}";
    }
    std::string err;
    if (!validate(name, args_json, err)) {
        return "{\"error\":\"" + json_escape(err) + "\"}";
    }
    const std::string result = t->fn(args_json, ok);
    if (!ok) return result.empty() ? std::string("{\"error\":\"failed\"}") : result;
    return result;
}

} // namespace omniseed
