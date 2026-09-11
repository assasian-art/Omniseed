// =============================================================================
//  OmniSeed — trading/finance.cpp
//  Closed-form financial math + the sandboxed interpreter.
// =============================================================================
#include "omniseed/trading/finance.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace omniseed {
namespace trading {

// ===========================================================================
// Core financial functions
// ===========================================================================
double norm_cdf(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

double finance_npv(double rate, const std::vector<double>& cashflows) {
    double v = 0.0;
    for (size_t t = 0; t < cashflows.size(); ++t)
        v += cashflows[t] / std::pow(1.0 + rate, static_cast<double>(t));
    return v;
}

bool finance_irr(const std::vector<double>& cashflows, double& out_irr) {
    if (cashflows.size() < 2) return false;
    // IRR exists where NPV crosses zero; require a sign change in the range.
    const double lo = -0.9999, hi = 10.0;
    double f_lo = finance_npv(lo, cashflows);
    double f_hi = finance_npv(hi, cashflows);
    if (f_lo == 0.0) { out_irr = lo; return true; }
    if (f_hi == 0.0) { out_irr = hi; return true; }
    if ((f_lo > 0.0) == (f_hi > 0.0)) return false;   // no crossing

    // Plain bisection: robust, no derivatives needed.
    double a = lo, b = hi, fa = f_lo;
    for (int iter = 0; iter < 200; ++iter) {
        const double m = 0.5 * (a + b);
        const double fm = finance_npv(m, cashflows);
        if (std::fabs(fm) < 1e-10) { out_irr = m; return true; }
        if ((fm > 0.0) == (fa > 0.0)) { a = m; fa = fm; }
        else { b = m; }
    }
    out_irr = 0.5 * (a + b);
    return true;
}

double bs_call(double S, double K, double r, double sigma, double T) {
    if (S <= 0.0 || K <= 0.0 || sigma <= 0.0 || T <= 0.0) return 0.0;
    const double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) /
                      (sigma * std::sqrt(T));
    const double d2 = d1 - sigma * std::sqrt(T);
    return S * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
}

double bs_put(double S, double K, double r, double sigma, double T) {
    if (S <= 0.0 || K <= 0.0 || sigma <= 0.0 || T <= 0.0) return 0.0;
    const double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) /
                      (sigma * std::sqrt(T));
    const double d2 = d1 - sigma * std::sqrt(T);
    return K * std::exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1);
}

// ===========================================================================
// FinanceInterpreter — sandboxed recursive-descent evaluator
// ===========================================================================
namespace {

struct Tok {
    enum Kind { Num, Ident, Op, LParen, RParen, Comma, End } kind = End;
    double num = 0.0;
    std::string text;
};

class Lexer {
public:
    explicit Lexer(const std::string& s) : s_(s) {}

    Tok next() {
        Tok t;
        skip_ws();
        if (pos_ >= s_.size()) { t.kind = Tok::End; return t; }
        const char c = s_[pos_];
        if (std::isdigit(static_cast<unsigned char>(c)) || c == '.') {
            size_t end = pos_;
            while (end < s_.size() &&
                   (std::isdigit(static_cast<unsigned char>(s_[end])) ||
                    s_[end] == '.'))
                ++end;
            t.kind = Tok::Num;
            t.num = std::atof(s_.substr(pos_, end - pos_).c_str());
            pos_ = end;
            return t;
        }
        if (std::isalpha(static_cast<unsigned char>(c))) {
            size_t end = pos_;
            while (end < s_.size() &&
                   (std::isalnum(static_cast<unsigned char>(s_[end])) ||
                    s_[end] == '_'))
                ++end;
            t.kind = Tok::Ident;
            t.text = s_.substr(pos_, end - pos_);
            pos_ = end;
            return t;
        }
        ++pos_;
        switch (c) {
            case '+': case '-': case '*': case '/': case '^':
                t.kind = Tok::Op; t.text = std::string(1, c); return t;
            case '(': t.kind = Tok::LParen; return t;
            case ')': t.kind = Tok::RParen; return t;
            case ',': t.kind = Tok::Comma; return t;
            default: t.kind = Tok::End; return t;      // lexer error via End
        }
    }

private:
    void skip_ws() {
        while (pos_ < s_.size() && s_[pos_] == ' ') ++pos_;
    }
    const std::string& s_;
    size_t pos_ = 0;
};

class Parser {
public:
    Parser(const std::string& s) : lex_(s), look_(lex_.next()) {}

    bool expr(double& out) {
        if (!term(out)) return false;
        while (look_.kind == Tok::Op &&
               (look_.text == "+" || look_.text == "-")) {
            const std::string op = look_.text;
            advance();
            double rhs = 0.0;
            if (!term(rhs)) return false;
            out = op == "+" ? out + rhs : out - rhs;
        }
        return true;
    }

private:
    bool term(double& out) {
        if (!power(out)) return false;
        while (look_.kind == Tok::Op &&
               (look_.text == "*" || look_.text == "/")) {
            const std::string op = look_.text;
            advance();
            double rhs = 0.0;
            if (!power(rhs)) return false;
            if (op == "*") out *= rhs;
            else {
                if (rhs == 0.0) return fail("division by zero");
                out /= rhs;
            }
        }
        return true;
    }

    bool power(double& out) {
        if (!unary(out)) return false;
        if (look_.kind == Tok::Op && look_.text == "^") {
            advance();
            double rhs = 0.0;
            if (!unary(rhs)) return false;
            out = std::pow(out, rhs);
        }
        return true;
    }

    bool unary(double& out) {
        if (look_.kind == Tok::Op && look_.text == "-") {
            advance();
            double v = 0.0;
            if (!unary(v)) return false;
            out = -v;
            return true;
        }
        return primary(out);
    }

    bool primary(double& out) {
        if (look_.kind == Tok::Num) {
            out = look_.num;
            advance();
            return true;
        }
        if (look_.kind == Tok::LParen) {
            advance();
            if (!expr(out)) return false;
            if (look_.kind != Tok::RParen) return fail("missing )");
            advance();
            return true;
        }
        if (look_.kind == Tok::Ident) return call(out);
        return fail("unexpected token");
    }

    bool call(double& out) {
        const std::string name = look_.text;
        if (!FinanceInterpreter::is_allowed_function(name))
            return fail("function not allowed: " + name);
        advance();
        if (look_.kind != Tok::LParen) return fail("expected (");
        advance();
        std::vector<double> args;
        if (look_.kind != Tok::RParen) {
            for (;;) {
                double v = 0.0;
                if (!expr(v)) return false;
                args.push_back(v);
                if (args.size() > 64) return fail("too many arguments");
                if (look_.kind == Tok::Comma) { advance(); continue; }
                break;
            }
        }
        if (look_.kind != Tok::RParen) return fail("missing )");
        advance();
        out = apply(name, args);
        return out_ok_;
    }

    double apply(const std::string& name, const std::vector<double>& a) {
        out_ok_ = true;
        auto need = [&](size_t n) {
            if (a.size() != n) { out_ok_ = false; fail_ = name + "() takes " +
                                                    std::to_string(n) + " args"; }
            return out_ok_;
        };
        if (name == "min" && need(2)) return std::min(a[0], a[1]);
        if (name == "max" && need(2)) return std::max(a[0], a[1]);
        if (name == "abs" && need(1)) return std::fabs(a[0]);
        if (name == "sqrt" && need(1)) {
            if (a[0] < 0.0) { out_ok_ = false; fail_ = "sqrt of negative"; return 0; }
            return std::sqrt(a[0]);
        }
        if (name == "pow" && need(2)) return std::pow(a[0], a[1]);
        if (name == "npv" && a.size() >= 2) {
            std::vector<double> cf(a.begin() + 1, a.end());
            return finance_npv(a[0], cf);
        }
        if (name == "irr" && a.size() >= 2) {
            std::vector<double> cf(a.begin(), a.end());
            double r = 0.0;
            if (!finance_irr(cf, r)) {
                out_ok_ = false;
                fail_ = "no IRR in [-99.99%, 1000%] (needs a sign change)";
                return 0.0;
            }
            return r;
        }
        if (name == "bscall" && need(5)) return bs_call(a[0], a[1], a[2], a[3], a[4]);
        if (name == "bsput" && need(5)) return bs_put(a[0], a[1], a[2], a[3], a[4]);
        out_ok_ = false;
        fail_ = "unknown function: " + name;
        return 0.0;
    }

    bool fail(const std::string& why) {
        fail_ = why.empty() ? "parse error" : why;
        return false;
    }

    void advance() { look_ = lex_.next(); }

    Lexer lex_;
    Tok look_;
    bool out_ok_ = true;
    std::string fail_;

public:
    const Tok& look() const { return look_; }
    const std::string& fail() const { return fail_; }
};

} // namespace

bool FinanceInterpreter::is_allowed_function(const std::string& name) {
    return name == "min" || name == "max" || name == "abs" ||
           name == "sqrt" || name == "pow" || name == "npv" ||
           name == "irr" || name == "bscall" || name == "bsput";
}

FinanceInterpreter::Result FinanceInterpreter::evaluate(
        const std::string& expression) {
    Result r;
    if (expression.size() > 4096) {
        r.error = "expression too long";
        return r;
    }
    Parser p(expression);
    if (!p.expr(r.value) || p.look().kind != Tok::End) {
        r.error = p.fail();
        return r;
    }
    if (!std::isfinite(r.value)) {
        r.error = "result is not finite";
        return r;
    }
    r.ok = true;
    return r;
}

} // namespace trading
} // namespace omniseed
