// =============================================================================
//  OmniSeed — flash_skills.cpp
//  Flash Skill Pools + Zero-Shot Skill Synthesis implementation.
//
//  The sandbox: exec_sandboxed() is a closed interpreter over RecipeStep
//  primitives. There is no path to the filesystem, network, or process APIs —
//  a step can only transform strings and evaluate arithmetic through
//  safe_arithmetic(), a recursive-descent parser that accepts only digits,
//  + - * / % ( ) . and whitespace (feature #36: no syscalls by construction,
//  step budget enforced).
// =============================================================================
#include "omniseed/agent/flash_skills.h"
#include "omniseed/core/platform.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>

namespace omniseed {

// ===========================================================================
// Safe arithmetic: recursive descent, doubles only, no identifiers/functions
// ===========================================================================
namespace {

struct CalcParser {
    const char* p = nullptr;
    bool ok = true;

    explicit CalcParser(const std::string& s) : p(s.c_str()) {}

    void ws() { while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') ++p; }

    double number() {
        ws();
        const char* start = p;
        while ((std::isdigit(static_cast<unsigned char>(*p)) || *p == '.')) ++p;
        if (p == start) { ok = false; return 0.0; }
        return std::atof(std::string(start, p - start).c_str());
    }

    double factor() {
        ws();
        if (*p == '(') {
            ++p;
            const double v = expr();
            ws();
            if (*p == ')') ++p; else ok = false;
            return v;
        }
        if (*p == '-') { ++p; return -factor(); }
        if (*p == '+') { ++p; return factor(); }
        return number();
    }

    double term() {
        double v = factor();
        while (ok) {
            ws();
            if (*p == '*') { ++p; v *= factor(); }
            else if (*p == '/') {
                ++p;
                const double d = factor();
                if (std::fabs(d) < 1e-12) { ok = false; return 0.0; }
                v /= d;
            }
            else if (*p == '%') {
                ++p;
                const double d = factor();
                if (std::fabs(d) < 1e-12) { ok = false; return 0.0; }
                v = std::fmod(v, d);
            }
            else break;
        }
        return v;
    }

    double expr() {
        double v = term();
        while (ok) {
            ws();
            if (*p == '+') { ++p; v += term(); }
            else if (*p == '-') { ++p; v -= term(); }
            else break;
        }
        return v;
    }
};

std::string fmt_num(double v) {
    if (std::fabs(v - std::llround(v)) < 1e-9 &&
        std::fabs(v) < 1e15) {
        return std::to_string(static_cast<long long>(std::llround(v)));
    }
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.6g", v);
    return std::string(buf);
}

std::string lower_copy(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s)
        out.push_back(static_cast<char>(
            std::tolower(static_cast<unsigned char>(c))));
    return out;
}

std::string trim_copy(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

// Extracts an arithmetic expression from free text, e.g.
// "what is 12*(3+4)?" -> "12*(3+4)". Scans for the longest run of
// arithmetic-legal characters containing at least one digit and operator.
std::string extract_arithmetic(const std::string& task) {
    const std::string legal = "0123456789+-*/%(). ";
    size_t best_b = 0, best_e = 0;
    size_t b = 0;
    const std::string low = lower_copy(task);
    while (b < low.size()) {
        if (legal.find(low[b]) == std::string::npos) { ++b; continue; }
        size_t e = b;
        while (e < low.size() && legal.find(low[e]) != std::string::npos) ++e;
        const std::string cand = trim_copy(low.substr(b, e - b));
        bool has_digit = false, has_op = false;
        for (char c : cand) {
            if (std::isdigit(static_cast<unsigned char>(c))) has_digit = true;
            if (c == '+' || c == '-' || c == '*' || c == '/') has_op = true;
        }
        if (has_digit && has_op && cand.size() > best_e - best_b) {
            best_b = b; best_e = e;
        }
        b = e + 1;
    }
    if (best_e > best_b) return trim_copy(low.substr(best_b, best_e - best_b));
    return "";
}

// Unit conversion table (deterministic, tiny).
struct UnitConv { const char* from; const char* to; double factor; };
const UnitConv kConversions[] = {
    {"km", "miles", 0.621371}, {"miles", "km", 1.609344},
    {"kg", "lbs", 2.20462}, {"lbs", "kg", 0.453592},
    {"c", "f", 0.0},        // special-cased
    {"f", "c", 0.0},        // special-cased
    {"m", "ft", 3.28084}, {"ft", "m", 0.3048},
    {"cm", "in", 0.393701}, {"in", "cm", 2.54},
};

} // namespace

bool FlashSkillPool::safe_arithmetic(const std::string& expr, double& out) {
    // Reject anything with letters (no identifiers, hex, functions).
    for (char c : expr) {
        if (std::isalpha(static_cast<unsigned char>(c))) return false;
    }
    CalcParser p(expr);
    const double v = p.expr();
    p.ws();
    if (!p.ok || *p.p != '\0') return false;
    if (!std::isfinite(v)) return false;
    out = v;
    return true;
}

// ===========================================================================
// Registry
// ===========================================================================
bool FlashSkillPool::install(const FlashSkill& skill) {
    if (skill.name.empty() || skill.steps.empty()) return false;
    if (skill.steps.size() > static_cast<size_t>(cfg_.max_steps)) return false;
    if (skills_.size() >= cfg_.max_skills) {
        // Retire the least-valuable skill to make room.
        retire_stale(platform::now_ms() / 1000.0);
        if (skills_.size() >= cfg_.max_skills) {
            auto victim = std::min_element(
                skills_.begin(), skills_.end(),
                [](const FlashSkill& a, const FlashSkill& b) {
                    return a.success_rate() < b.success_rate();
                });
            if (victim == skills_.end()) return false;
            skills_.erase(victim);
        }
    }
    if (find_const(skill.name) != nullptr) return false;  // no dupes
    skills_.push_back(skill);
    return true;
}

void FlashSkillPool::install_builtins() {
    {
        FlashSkill s;
        s.name = "calc";
        s.description = "Evaluate an arithmetic expression safely";
        s.triggers = {"calculate", "what is", "compute", "eval"};
        s.args = {"expr"};
        RecipeStep st;
        st.op = "calc";
        st.expr = "{expr}";
        s.steps.push_back(st);
        install(s);
    }
    {
        FlashSkill s;
        s.name = "time";
        s.description = "Report the current local time";
        s.triggers = {"time", "clock", "date"};
        RecipeStep st;
        st.op = "time";
        s.steps.push_back(st);
        install(s);
    }
    {
        FlashSkill s;
        s.name = "echo";
        s.description = "Repeat the input text back";
        s.triggers = {"repeat", "echo"};
        s.args = {"text"};
        RecipeStep st;
        st.op = "echo";
        st.expr = "{text}";
        s.steps.push_back(st);
        install(s);
    }
}

const FlashSkill* FlashSkillPool::find_const(const std::string& name) const {
    for (const FlashSkill& s : skills_)
        if (s.name == name) return &s;
    return nullptr;
}

FlashSkill* FlashSkillPool::find_mutable(const std::string& name) {
    for (FlashSkill& s : skills_)
        if (s.name == name) return &s;
    return nullptr;
}

const FlashSkill* FlashSkillPool::get(const std::string& name) const {
    return find_const(name);
}

// ===========================================================================
// Matching: trigger substring, longest-trigger wins
// ===========================================================================
const FlashSkill* FlashSkillPool::match(const std::string& task) const {
    const std::string low = lower_copy(task);
    const FlashSkill* best = nullptr;
    size_t best_len = 0;
    for (const FlashSkill& s : skills_) {
        for (const std::string& trig : s.triggers) {
            if (!trig.empty() && low.find(trig) != std::string::npos &&
                trig.size() > best_len) {
                best = &s;
                best_len = trig.size();
            }
        }
    }
    return best;
}

// ===========================================================================
// Sandboxed execution
// ===========================================================================
SkillResult FlashSkillPool::run(
        const FlashSkill& skill,
        const std::unordered_map<std::string, std::string>& args) {
    SkillResult r = exec_sandboxed(skill, args);
    // Stats + persistence (skill use telemetry, spec #49).
    FlashSkill* s = find_mutable(skill.name);
    if (s) {
        ++s->use_count;
        if (r.ok) ++s->success_count;
        s->last_used = static_cast<uint64_t>(platform::now_ms() / 1000.0);
        if (s->avg_ms <= 0.0) s->avg_ms = 0.0;
        s->avg_ms = s->avg_ms * 0.7 + (r.ok ? 1.0 : 0.0) * 0.3; // rolling ok-rate
    }
    save();
    return r;
}

SkillResult FlashSkillPool::exec_sandboxed(
        const FlashSkill& skill,
        const std::unordered_map<std::string, std::string>& args) const {
    SkillResult r;
    if (skill.steps.empty() ||
        skill.steps.size() > static_cast<size_t>(cfg_.max_steps)) {
        r.output = "skill step budget exceeded";
        return r;
    }

    auto substitute = [&](const std::string& tmpl,
                          const std::string& prev) -> std::string {
        std::string out = tmpl;
        // {prev}
        const std::string prev_tok = "{prev}";
        size_t pos = out.find(prev_tok);
        while (pos != std::string::npos) {
            out.replace(pos, prev_tok.size(), prev);
            pos = out.find(prev_tok, pos + 1);
        }
        // {arg}
        for (const auto& [k, v] : args) {
            const std::string tok = "{" + k + "}";
            pos = out.find(tok);
            while (pos != std::string::npos) {
                out.replace(pos, tok.size(), v);
                pos = out.find(tok, pos + tok.size());
            }
        }
        return out;
    };

    std::string prev;
    int32_t step = 0;
    for (const RecipeStep& st : skill.steps) {
        if (++step > cfg_.max_steps) {
            r.output = "sandbox: step budget exceeded";
            return r;                          // not ok
        }
        const std::string operand = substitute(st.expr, prev);

        if (st.op == "calc") {
            double v = 0.0;
            if (!safe_arithmetic(operand, v)) {
                r.trace.push_back("calc FAILED on '" + operand + "'");
                r.output = "calc failed";
                return r;
            }
            prev = fmt_num(v);
        } else if (st.op == "text") {
            prev = operand;
        } else if (st.op == "echo") {
            prev = operand;
        } else if (st.op == "lower") {
            prev = lower_copy(operand);
        } else if (st.op == "len") {
            prev = std::to_string(operand.size());
        } else if (st.op == "time") {
            const double secs = platform::now_ms() / 1000.0;
            const std::time_t t = static_cast<std::time_t>(secs);
            char buf[64];
            std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S",
                          std::localtime(&t));
            prev = buf;
        } else {
            r.trace.push_back("unknown op '" + st.op + "'");
            r.output = "sandbox: unknown op";
            return r;
        }
        r.trace.push_back(st.op + " -> " + prev);
    }

    r.output = prev;
    r.ok = true;
    return r;
}

void FlashSkillPool::report(const std::string& name, bool ok, double ms) {
    FlashSkill* s = find_mutable(name);
    if (!s) return;
    ++s->use_count;
    if (ok) ++s->success_count;
    s->avg_ms = s->avg_ms <= 0.0 ? ms : s->avg_ms * 0.8 + ms * 0.2;
    s->last_used = static_cast<uint64_t>(platform::now_ms() / 1000.0);
}

// ===========================================================================
// Zero-Shot Skill Synthesis: build a NEW recipe from the task text
// ===========================================================================
const FlashSkill* FlashSkillPool::synthesize(const std::string& task) {
    const std::string low = lower_copy(task);

    FlashSkill s;
    s.synthesized = true;

    // ---- Plan A: pure arithmetic --------------------------------------
    std::string expr = extract_arithmetic(task);
    if (!expr.empty() && low.find("convert") == std::string::npos) {
        s.name = "zs_calc_" + std::to_string(skills_.size());
        s.description = "synthesized: arithmetic eval for '" + expr + "'";
        s.triggers = {expr.substr(0, std::min<size_t>(expr.size(), 24))};
        s.args = {"expr"};
        RecipeStep st;
        st.op = "calc";
        st.expr = "{expr}";
        s.steps.push_back(st);
        if (install(s)) return find_const(s.name);
        return nullptr;
    }

    // ---- Plan B: unit conversion --------------------------------------
    if (low.find("convert") != std::string::npos) {
        // Expect: convert <num> <unit> to <unit>
        double num = 0.0;
        char u1[32] = {0}, u2[32] = {0};
        const int got = std::sscanf(low.c_str(), " convert %lf %31s to %31s",
                                    &num, u1, u2);
        if (got == 3) {
            double factor = 0.0;
            bool f_to_c = false, c_to_f = false;
            for (const UnitConv& cv : kConversions) {
                if (std::strcmp(u1, cv.from) == 0 && std::strcmp(u2, cv.to) == 0) {
                    if (std::strcmp(u1, "c") == 0) { f_to_c = false; c_to_f = false; }
                    factor = cv.factor;
                    break;
                }
            }
            if (std::strcmp(u1, "f") == 0 && std::strcmp(u2, "c") == 0) {
                // F -> C = (f - 32) * 5/9
                RecipeStep a; a.op = "calc";
                a.expr = "({" + std::string(u1) + "} - 32) * 0.555556";
                s.name = "zs_conv_f_to_c_" + std::to_string(skills_.size());
                s.args = {"f"};
                s.steps.push_back(a);
            } else if (std::strcmp(u1, "c") == 0 && std::strcmp(u2, "f") == 0) {
                RecipeStep a; a.op = "calc";
                a.expr = "({c} * 1.8) + 32";
                s.name = "zs_conv_c_to_f_" + std::to_string(skills_.size());
                s.args = {"c"};
                s.steps.push_back(a);
            } else if (factor > 0.0) {
                RecipeStep a; a.op = "calc";
                a.expr = "{" + std::string(u1) + "} * " + fmt_num(factor);
                s.name = "zs_conv_" + std::string(u1) + "_to_" + u2 + "_" +
                         std::to_string(skills_.size());
                s.args = {u1};
                s.steps.push_back(a);
            }
            if (!s.steps.empty()) {
                s.description = "synthesized: unit conversion " + std::string(u1) +
                                " -> " + u2;
                s.triggers = {"convert"};
                if (install(s)) return find_const(s.name);
            }
        }
        return nullptr;
    }

    // ---- Plan C: count characters of a quoted string -------------------
    if (low.find("count") != std::string::npos ||
        low.find("how many letters") != std::string::npos) {
        const auto q1 = task.find('"');
        const auto q2 = task.rfind('"');
        if (q1 != std::string::npos && q2 != std::string::npos && q2 > q1 + 1) {
            s.name = "zs_len_" + std::to_string(skills_.size());
            s.description = "synthesized: string length";
            s.triggers = {"count"};
            s.args = {"text"};
            RecipeStep st;
            st.op = "len";
            st.expr = "{text}";
            s.steps.push_back(st);
            if (install(s)) return find_const(s.name);
        }
        return nullptr;
    }

    return nullptr;    // synthesis failed: caller falls back to generation
}

// ===========================================================================
// Retirement (spec #50) + dream-state hook
// ===========================================================================
size_t FlashSkillPool::retire_stale(double now_unix_seconds) {
    size_t removed = 0;
    const double cutoff = now_unix_seconds - cfg_.retire_after_days * 86400.0;
    skills_.erase(
        std::remove_if(skills_.begin(), skills_.end(),
                       [&](const FlashSkill& s) {
                           const bool builtin = !s.synthesized &&
                                                s.name == "calc" ||
                                                s.name == "time" ||
                                                s.name == "echo";
                           if (builtin) return false;      // keep builtins
                           const bool stale =
                               s.last_used > 0 &&
                               static_cast<double>(s.last_used) < cutoff;
                           const bool failing =
                               s.use_count >= 4 &&
                               s.success_rate() < cfg_.min_success_rate;
                           if (stale || failing) { ++removed; return true; }
                           return false;
                       }),
        skills_.end());
    return removed;
}

// ===========================================================================
// Persistence: 'SKPL' binary
// ===========================================================================
static void write_str(FILE* f, const std::string& s) {
    const uint32_t n = static_cast<uint32_t>(s.size());
    std::fwrite(&n, 4, 1, f);
    std::fwrite(s.data(), 1, n, f);
}

static bool read_str(FILE* f, std::string& s) {
    uint32_t n = 0;
    if (std::fread(&n, 4, 1, f) != 1) return false;
    if (n > 4096) return false;
    s.resize(n);
    if (n && std::fread(s.data(), 1, n, f) != n) return false;
    return true;
}

bool FlashSkillPool::save() const {
    // Ensure the parent directory exists (portable: std::system + mkdir).
    const auto slash = cfg_.store_path.find_last_of("/");
    if (slash != std::string::npos && slash > 0) {
        const std::string dir = cfg_.store_path.substr(0, slash);
#if OMNISEED_PLATFORM_WINDOWS
        const std::string mkdir_cmd = "if not exist \"" + dir + "\" mkdir \"" + dir + "\"";
#else
        const std::string mkdir_cmd = "mkdir -p '" + dir + "'";
#endif
        (void)std::system(mkdir_cmd.c_str());   // best effort
    }
    FILE* f = platform::open_file_c(cfg_.store_path.c_str(), "wb");
    if (!f) return false;
    const uint32_t magic = 0x4C504B53;  // SKPL
    const uint32_t version = 1;
    const uint32_t n = static_cast<uint32_t>(skills_.size());
    std::fwrite(&magic, 4, 1, f);
    std::fwrite(&version, 4, 1, f);
    std::fwrite(&n, 4, 1, f);
    for (const FlashSkill& s : skills_) {
        write_str(f, s.name);
        write_str(f, s.description);
        const uint32_t nt = static_cast<uint32_t>(s.triggers.size());
        std::fwrite(&nt, 4, 1, f);
        for (const std::string& t : s.triggers) write_str(f, t);
        const uint32_t na = static_cast<uint32_t>(s.args.size());
        std::fwrite(&na, 4, 1, f);
        for (const std::string& a : s.args) write_str(f, a);
        const uint32_t ns = static_cast<uint32_t>(s.steps.size());
        std::fwrite(&ns, 4, 1, f);
        for (const RecipeStep& st : s.steps) {
            write_str(f, st.op);
            write_str(f, st.expr);
        }
        std::fwrite(&s.use_count, 4, 1, f);
        std::fwrite(&s.success_count, 4, 1, f);
        std::fwrite(&s.avg_ms, 8, 1, f);
        std::fwrite(&s.last_used, 8, 1, f);
        std::fwrite(&s.synthesized, 1, 1, f);
    }
    std::fclose(f);
    return true;
}

bool FlashSkillPool::load() {
    FILE* f = platform::open_file_c(cfg_.store_path.c_str(), "rb");
    if (!f) return false;
    uint32_t magic = 0, version = 0, n = 0;
    if (std::fread(&magic, 4, 1, f) != 1 || magic != 0x4C504B53 ||
        std::fread(&version, 4, 1, f) != 1 || version != 1 ||
        std::fread(&n, 4, 1, f) != 1) {
        std::fclose(f);
        return false;
    }
    skills_.clear();
    for (uint32_t i = 0; i < n; ++i) {
        FlashSkill s;
        if (!read_str(f, s.name) || !read_str(f, s.description)) break;
        uint32_t nt = 0;
        if (std::fread(&nt, 4, 1, f) != 1) break;
        for (uint32_t t = 0; t < nt; ++t) {
            std::string tr;
            if (!read_str(f, tr)) break;
            s.triggers.push_back(tr);
        }
        uint32_t na = 0;
        if (std::fread(&na, 4, 1, f) != 1) break;
        for (uint32_t a = 0; a < na; ++a) {
            std::string ar;
            if (!read_str(f, ar)) break;
            s.args.push_back(ar);
        }
        uint32_t ns = 0;
        if (std::fread(&ns, 4, 1, f) != 1) break;
        for (uint32_t stI = 0; stI < ns; ++stI) {
            RecipeStep st;
            if (!read_str(f, st.op) || !read_str(f, st.expr)) break;
            s.steps.push_back(st);
        }
        if (std::fread(&s.use_count, 4, 1, f) != 1) break;
        if (std::fread(&s.success_count, 4, 1, f) != 1) break;
        if (std::fread(&s.avg_ms, 8, 1, f) != 1) break;
        if (std::fread(&s.last_used, 8, 1, f) != 1) break;
        if (std::fread(&s.synthesized, 1, 1, f) != 1) break;
        skills_.push_back(std::move(s));
    }
    std::fclose(f);
    return true;
}

} // namespace omniseed
