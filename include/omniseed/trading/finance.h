// =============================================================================
//  OmniSeed — trading/finance.h
//  Financial math + a sandboxed expression interpreter (Phase-16 Phase 3
//  "advanced reasoning tools").
//
//    * npv / irr / black-scholes  — closed-form, deterministic
//    * FinanceInterpreter         — safe arithmetic evaluator extended with
//                                   whitelisted functions. NO variables, NO
//                                   I/O, NO loops: sandboxing by construction
//                                   (recursion + input caps prevent blowup).
// =============================================================================
#pragma once

#include <string>
#include <vector>

namespace omniseed {
namespace trading {

// ---------------------------------------------------------------------------
// Core financial functions
// ---------------------------------------------------------------------------
// Net present value: cashflows[0] is typically the initial outlay (negative).
double finance_npv(double rate, const std::vector<double>& cashflows);

// Internal rate of return via bisection on [-0.9999, 10]; false when no sign
// change exists in the range (no economically meaningful IRR).
bool finance_irr(const std::vector<double>& cashflows, double& out_irr);

// Black-Scholes European option pricing (per unit; multiply by contract size
// upstream). Requires sigma > 0, T > 0, S > 0, K > 0.
double bs_call(double S, double K, double r, double sigma, double T);
double bs_put (double S, double K, double r, double sigma, double T);

// Standard normal CDF (via std::erfc — exact enough for finance use).
double norm_cdf(double x);

// ---------------------------------------------------------------------------
// FinanceInterpreter — sandboxed arithmetic + whitelisted functions
// ---------------------------------------------------------------------------
class FinanceInterpreter {
public:
    struct Result {
        bool   ok = false;
        double value = 0.0;
        std::string error;              // human-readable, no internals leaked
    };

    // Grammar: numbers, + - * / ( ), unary minus, comma-separated args, and
    // functions from the whitelist: min max abs sqrt pow npv irr bscall bsput.
    // irr(...) takes a cashflow list; bscall/bsput(S, K, r, sigma, T).
    static Result evaluate(const std::string& expression);

    // Exposed for tests: the function whitelist.
    static bool is_allowed_function(const std::string& name);
};

} // namespace trading
} // namespace omniseed
