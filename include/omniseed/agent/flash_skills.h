// =============================================================================
//  OmniSeed — flash_skills.h
//  Flash Skill Pools + Zero-Shot Skill Synthesis (features #8, #9, #34-36,
//  #49, #50 of the 100-feature spec; capability #6 of the seven novel
//  capabilities).
//
//  Blueprint:
//    * A Flash Skill is a small sandboxed program ("recipe") loaded only when
//      needed; the base RAM footprint stays minimal and skills unload after
//      use (S-LoRA-style dynamic adapters, adapted to a CPU kernel).
//    * Zero-Shot Skill Synthesis: for a task with no matching skill, the
//      planner decomposes the goal into primitive steps (calc/echo/time
//      building blocks + arithmetic plans) and emits a NEW recipe without any
//      pre-existing skill. Recipes are validated and then run inside a
//      sandboxed interpreter with a step budget, no I/O, and no syscalls.
//    * The skill pool persists to a binary file (SKPL) and skills
//      auto-retire when unused (spec #50) and report performance stats
//      (spec #49).
// =============================================================================
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace omniseed {

// ---------------------------------------------------------------------------
// Recipe: a portable skill program
//
//   A recipe is a JSON-like object with:
//     name          — unique skill id
//     trigger       — substring patterns that activate it (any-of)
//     steps         — ordered primitive steps:
//                       {"op":"calc","expr":"..."}     — arithmetic eval
//                       {"op":"text","format":"..."}   — format with {a},{b}
//                       {"op":"echo"}                  — echo inputs
//                       {"op":"time"}                  — current time
//                       {"op":"repeat","n":N}          — repeat last result
//     args          — declared input slots ("a","b",...)
//   Steps may reference {arg} slots and {prev} (previous step result).
//   Everything executes in-process with no external calls: this IS the
//   sandbox (no file/net access in the interpreter by construction).
// ---------------------------------------------------------------------------
struct RecipeStep {
    std::string op;       // "calc" | "text" | "echo" | "time" | "lower" | "len"
    std::string expr;     // operand / format string
};

struct FlashSkill {
    std::string name;
    std::string description;
    std::vector<std::string> triggers;      // lowercase substrings
    std::vector<std::string> args;          // declared slots, e.g. "expr"
    std::vector<RecipeStep> steps;

    // Performance stats (spec #49).
    uint32_t use_count    = 0;
    uint32_t success_count = 0;
    double   avg_ms       = 0.0;
    uint64_t last_used    = 0;              // unix seconds

    bool synthesized = false;               // zero-shot origin marker
    double success_rate() const {
        return use_count ? static_cast<double>(success_count) / use_count : 0.0;
    }
};

// ---------------------------------------------------------------------------
// Skill execution result
// ---------------------------------------------------------------------------
struct SkillResult {
    bool ok = false;
    std::string output;                     // final value
    std::vector<std::string> trace;         // per-step results
};

// ---------------------------------------------------------------------------
// FlashSkillPool — registry, matching, sandboxed execution, persistence
// ---------------------------------------------------------------------------
class FlashSkillPool {
public:
    struct Config {
        size_t   max_skills   = 64;
        int32_t  max_steps    = 16;        // sandbox step budget
        double   retire_after_days = 14.0; // unused-skill retirement
        double   min_success_rate = 0.25;  // retire chronic failures
        std::string store_path = "./state/skills.bin";
    };

    explicit FlashSkillPool(const Config& cfg = {}) : cfg_(cfg) {}

    // Registers a skill (copies). Returns false when the pool is full or
    // the skill is invalid (no name / no steps / oversized).
    bool install(const FlashSkill& skill);

    // Builtin skills installed at startup: calc, time, echo helpers.
    void install_builtins();

    // Finds a skill whose trigger matches the task text (case-insensitive).
    const FlashSkill* match(const std::string& task) const;

    // Runs a skill inside the sandbox. args maps slot name -> value.
    SkillResult run(const FlashSkill& skill,
                    const std::unordered_map<std::string, std::string>& args);

    // Updates stats and persists.
    void report(const std::string& name, bool ok, double ms);

    // Zero-Shot Skill Synthesis: attempts to build a NEW recipe for a task
    // with no matching skill. The planner uses deterministic primitives:
    //   * arithmetic expression detected  -> calc skill with extracted expr
    //   * "convert X unit to unit"        -> calc skill with factor table
    //   * "count/length of <text>"        -> len skill
    //   * otherwise fails (caller may fall back to normal generation)
    // Returns the synthesized skill (also installed with synthesized=true)
    // or nullptr.
    const FlashSkill* synthesize(const std::string& task);

    // Dream-State friendly maintenance: retire stale/failed skills.
    // Returns the number of skills retired.
    size_t retire_stale(double now_unix_seconds);

    size_t size() const { return skills_.size(); }
    const FlashSkill* get(const std::string& name) const;
    const Config& config() const { return cfg_; }
    void set_store_path(const std::string& p) { cfg_.store_path = p; }

    // Persistence ('SKPL' binary, version 1).
    bool save() const;
    bool load();

private:
    SkillResult exec_sandboxed(const FlashSkill& skill,
                               const std::unordered_map<std::string,
                                                        std::string>& args) const;
    static bool safe_arithmetic(const std::string& expr, double& out);

    FlashSkill* find_mutable(const std::string& name);
    const FlashSkill* find_const(const std::string& name) const;

    Config cfg_;
    std::vector<FlashSkill> skills_;
};

} // namespace omniseed
