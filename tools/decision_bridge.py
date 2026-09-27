#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — tools/decision_bridge.py
#  System-1 decision JSON -> paper-book gate.
#
#  The C++ System-1 head (include/omniseed/decision_head.h) answers a turn with
#  ONE matvec and prints a DecisionResult as JSON:
#
#    {"action":"BUY","confidence":0.9312,"target_asset":"AAPL",
#     "invalidation":178.4200,"routing":"self","fast_path":true,
#     "margin":0.8123,"matvecs":1}
#
#  This module is the Python half of that contract: it parses the JSON and
#  decides whether the verdict is allowed to open NEW risk in the paper book.
#
#  WHAT THIS MODULE DOES NOT DO
#  ----------------------------
#  It does not size positions, and it does not enforce the money limits. The 2%
#  per-trade risk budget and the 3% daily kill-switch live in C++
#  (`RiskLimits::risk_per_trade_pct`, clamped into [1%, 2%], and
#  `RiskGovernorSet::Config{max_daily_loss_pct = 0.03}`) and are re-implemented
#  nowhere. This bridge can only ever REMOVE risk: a symbol it refuses is added
#  to the engine's `--skip` list, which gates new entries and nothing else —
#  open positions are still marked, stopped and exited.
#
#  FAIL-CLOSED RULES (all of them, in order)
#  -----------------------------------------
#   1. The payload must be a JSON object carrying `action` and `confidence`.
#   2. `action` must be one of the seven names the C++ enum defines.
#   3. `confidence` must be finite and inside [0, 1].
#   4. `fast_path` must be a real JSON boolean and must be true. A head that
#      did not self-route escalated to System-2; its verdict is advice, not a
#      decision.
#   5. ABSTAIN and EXPLAIN never trade, at any confidence. (The C++ head
#      already refuses to self-route them; this is the same rule restated
#      where it cannot be forgotten.)
#   6. `confidence` must clear the bar, and the bar may only be RAISED — never
#      lowered below the mandate's 0.85.
#   7. An action that opens risk (BUY / SELL / HEDGE) must carry a usable
#      invalidation. `0` means "not specified" — UNKNOWN, not "no stop". An
#      unknown stop is an unacceptable trade, not a stop-less trade.
#   8. Silence is not consent: in `screen()`, a watched symbol with no decision
#      is refused. A missing decision is treated exactly like a refusal.
#
#  Usage:
#    python tools/decision_bridge.py --json '{"action":"BUY", ...}'
#    python tools/decision_bridge.py --jsonl state/decisions.jsonl
#    python tools/decision_bridge.py --json '<...>' --quiet   # exit code only
#    python tools/decision_bridge.py --self-test
#
#  Exit codes: 0 = tradeable, 1 = refused, 2 = malformed / bad invocation.
# =============================================================================
import argparse
import json
import math
import sys

# --- contract constants (mirror decision_head.h) ----------------------------
MANDATE_THRESHOLD = 0.85
ACTIONS = ("ABSTAIN", "HOLD", "BUY", "SELL", "CLOSE", "HEDGE", "EXPLAIN")
NEVER_TRADES = frozenset({"ABSTAIN", "EXPLAIN"})
OPENS_RISK = frozenset({"BUY", "SELL", "HEDGE"})
REQUIRED_KEYS = ("action", "confidence")


class DecisionParseError(ValueError):
    """The payload is not a decision this bridge is willing to read."""


def _finite(x, what):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise DecisionParseError(
            f"{what} must be a number, got {type(x).__name__}")
    x = float(x)
    if not math.isfinite(x):
        raise DecisionParseError(f"{what} must be finite, got {x!r}")
    return x


class Decision:
    """One parsed System-1 verdict."""

    __slots__ = ("action", "confidence", "target_asset", "invalidation",
                 "routing", "fast_path", "margin", "matvecs", "raw")

    def __init__(self, action, confidence, target_asset="", invalidation=0.0,
                 routing="", fast_path=False, margin=0.0, matvecs=0, raw=None):
        self.action = action
        self.confidence = confidence
        self.target_asset = target_asset
        self.invalidation = invalidation
        self.routing = routing
        self.fast_path = fast_path
        self.margin = margin
        self.matvecs = matvecs
        self.raw = raw if raw is not None else {}

    def __repr__(self):
        return (f"Decision({self.action} conf={self.confidence:.4f} "
                f"asset={self.target_asset!r} inval={self.invalidation:.4f} "
                f"fast_path={self.fast_path})")

    # -- the gate ----------------------------------------------------------
    def refusal(self, threshold=MANDATE_THRESHOLD, require_stop=True):
        """-> '' if the verdict may open new risk, else why it may not."""
        if self.action in NEVER_TRADES:
            return f"action {self.action} never opens risk"
        if not self.fast_path:
            return ("head did not self-route (routing=%s) -> escalated to "
                    "System-2" % (self.routing or "system2"))
        if self.confidence < threshold:
            return f"confidence {self.confidence:.4f} < bar {threshold:.4f}"
        if require_stop and self.action in OPENS_RISK and self.invalidation <= 0.0:
            return ("invalidation is 0 == UNKNOWN, not 'no stop' "
                    "-> entry refused")
        return ""

    def tradeable(self, threshold=MANDATE_THRESHOLD, require_stop=True):
        return self.refusal(threshold, require_stop) == ""

    def as_dict(self):
        return {"action": self.action, "confidence": self.confidence,
                "target_asset": self.target_asset,
                "invalidation": self.invalidation, "routing": self.routing,
                "fast_path": self.fast_path, "margin": self.margin,
                "matvecs": self.matvecs}


def parse_decision(payload):
    """str | bytes | dict -> Decision. Raises DecisionParseError on anything
    that is not exactly the documented shape."""
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8", "replace")
    if isinstance(payload, str):
        try:
            obj = json.loads(payload)
        except ValueError as e:
            raise DecisionParseError(f"not JSON: {e}") from e
    elif isinstance(payload, dict):
        obj = payload
    else:
        raise DecisionParseError(
            f"expected str, bytes or dict, got {type(payload).__name__}")

    if not isinstance(obj, dict):
        raise DecisionParseError(
            f"expected a JSON object, got {type(obj).__name__}")
    for k in REQUIRED_KEYS:
        if k not in obj:
            raise DecisionParseError(f"missing required key {k!r}")

    action = obj["action"]
    if not isinstance(action, str):
        raise DecisionParseError("action must be a string")
    action = action.strip().upper()
    if action not in ACTIONS:
        raise DecisionParseError(
            f"unknown action {action!r}; known: {', '.join(ACTIONS)}")

    confidence = _finite(obj["confidence"], "confidence")
    if not 0.0 <= confidence <= 1.0:
        raise DecisionParseError(f"confidence {confidence} outside [0,1]")

    invalidation = _finite(obj.get("invalidation", 0.0), "invalidation")

    asset = obj.get("target_asset", "")
    if asset is None:
        asset = ""
    if not isinstance(asset, str):
        raise DecisionParseError("target_asset must be a string")

    routing = obj.get("routing", "")
    if routing is None:
        routing = ""
    if not isinstance(routing, str):
        raise DecisionParseError("routing must be a string")

    fast_path = obj.get("fast_path", False)
    if not isinstance(fast_path, bool):
        raise DecisionParseError(
            "fast_path must be a JSON boolean (the C++ head emits true/false)")

    margin = _finite(obj.get("margin", 0.0), "margin")

    matvecs = obj.get("matvecs", 0)
    if isinstance(matvecs, bool) or not isinstance(matvecs, int):
        raise DecisionParseError("matvecs must be an integer")

    return Decision(action, confidence, asset, invalidation, routing,
                    fast_path, margin, matvecs, obj)


def parse_jsonl(text_or_lines):
    """-> [(line_no, Decision | DecisionParseError)]. Never raises: a bad line
    must not be able to take down a whole cycle."""
    if isinstance(text_or_lines, str):
        lines = text_or_lines.splitlines()
    else:
        lines = list(text_or_lines)
    out = []
    for n, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            out.append((n, parse_decision(line)))
        except DecisionParseError as e:
            out.append((n, e))
    return out


def screen(decisions, symbols=None, threshold=MANDATE_THRESHOLD,
           require_stop=True):
    """Split verdicts into (allowed, refused).

    decisions : {symbol: Decision} | [Decision] (symbol from target_asset)
    symbols   : if given, every symbol listed here with NO decision is refused
                as well. A missing decision is treated exactly like a refusal —
                the bridge is safe to run on a partial or empty decision log.
    threshold : must be >= MANDATE_THRESHOLD. The bar may be raised by the
                caller, never lowered.
    -> (allowed: list[str], refused: list[dict])
    """
    if threshold < MANDATE_THRESHOLD:
        raise ValueError(
            f"threshold {threshold} is below the mandate's {MANDATE_THRESHOLD}; "
            "the bar may be raised, never lowered")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold {threshold} outside [0,1]")

    by_sym = {}
    if isinstance(decisions, dict):
        for sym, d in decisions.items():
            if isinstance(d, DecisionParseError):
                by_sym[str(sym)] = d
            elif isinstance(d, Decision):
                by_sym[str(sym)] = d
            else:
                by_sym[str(sym)] = DecisionParseError(
                    f"not a Decision: {type(d).__name__}")
    else:
        for d in decisions:
            if isinstance(d, Decision):
                key = d.target_asset or ""
                by_sym[key] = d
            elif isinstance(d, DecisionParseError):
                by_sym.setdefault("", d)
            else:
                by_sym.setdefault("", DecisionParseError(
                    f"not a Decision: {type(d).__name__}"))

    allowed, refused = [], []
    seen = set()
    for sym in (symbols if symbols is not None else sorted(by_sym)):
        sym = str(sym)
        seen.add(sym)
        d = by_sym.get(sym)
        if d is None:
            refused.append({"symbol": sym, "reason": "no decision -> silence "
                                                     "is not consent"})
        elif isinstance(d, DecisionParseError):
            refused.append({"symbol": sym, "reason": f"malformed: {d}"})
        else:
            why = d.refusal(threshold, require_stop)
            if why:
                refused.append({"symbol": sym, "reason": why})
            else:
                allowed.append(sym)

    # Decisions for symbols nobody asked about are reported, never acted on.
    for sym in sorted(set(by_sym) - seen):
        refused.append({"symbol": sym, "reason": "decision for an unwatched "
                                                 "symbol -> ignored"})
    return allowed, refused


# ---------------------------------------------------------------------------
# Self-test (also exercised by tests/test_decision_bridge.py)
# ---------------------------------------------------------------------------
_GOOD = ('{"action":"BUY","confidence":0.9312,"target_asset":"AAPL",'
         '"invalidation":178.4200,"routing":"self","fast_path":true,'
         '"margin":0.8123,"matvecs":1}')


def _self_test():
    ok = True

    def chk(name, cond):
        nonlocal ok
        if not cond:
            ok = False
            print(f"  FAIL {name}")

    d = parse_decision(_GOOD)
    chk("parses", d.action == "BUY" and abs(d.confidence - 0.9312) < 1e-6)
    chk("tradeable", d.tradeable())
    chk("asset", d.target_asset == "AAPL")
    chk("invalidation", abs(d.invalidation - 178.42) < 1e-6)

    # fail-closed rules
    chk("abstain", not parse_decision(
        '{"action":"ABSTAIN","confidence":0.99,"fast_path":true}').tradeable())
    chk("explain", not parse_decision(
        '{"action":"EXPLAIN","confidence":0.99,"fast_path":true}').tradeable())
    chk("not self-routed", not parse_decision(
        '{"action":"BUY","confidence":0.99,"invalidation":10,"fast_path":false}'
    ).tradeable())
    chk("below bar", not parse_decision(
        '{"action":"BUY","confidence":0.50,"invalidation":10,"fast_path":true}'
    ).tradeable())
    chk("unknown stop", not parse_decision(
        '{"action":"BUY","confidence":0.99,"invalidation":0,"fast_path":true}'
    ).tradeable())
    chk("string fast_path", _raises(lambda: parse_decision(
        '{"action":"BUY","confidence":0.99,"fast_path":"true"}')))
    chk("bad action", _raises(lambda: parse_decision(
        '{"action":"YOLO","confidence":0.99,"fast_path":true}')))
    chk("conf > 1", _raises(lambda: parse_decision(
        '{"action":"BUY","confidence":1.5,"fast_path":true}')))
    chk("missing key", _raises(lambda: parse_decision('{"action":"BUY"}')))
    chk("not json", _raises(lambda: parse_decision("not json")))

    allowed, refused = screen(
        {"AAPL": parse_decision(_GOOD)}, symbols=["AAPL", "MSFT"])
    chk("screen allows", allowed == ["AAPL"])
    chk("silence refused", any(r["symbol"] == "MSFT" for r in refused))
    chk("threshold floor", _raises(lambda: screen({}, threshold=0.5),
                                   ValueError))

    print(f"decision_bridge self-test: {'ok' if ok else 'FAILED'}")
    return 0 if ok else 1


def _raises(fn, exc=DecisionParseError):
    try:
        fn()
    except exc:
        return True
    return False


def main():
    ap = argparse.ArgumentParser(description="Parse a System-1 decision JSON")
    ap.add_argument("--json", help="one decision JSON object")
    ap.add_argument("--jsonl", help="a file of one decision JSON per line")
    ap.add_argument("--threshold", type=float, default=MANDATE_THRESHOLD)
    ap.add_argument("--allow-unknown-stop", action="store_true",
                    help="do NOT refuse an entry whose invalidation is 0 "
                         "(unsafe; off by default)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        return _self_test()
    if a.threshold < MANDATE_THRESHOLD:
        print(f"[bridge] threshold {a.threshold} < {MANDATE_THRESHOLD}: the bar "
              "may be raised, never lowered", file=sys.stderr)
        return 2

    if a.json:
        try:
            d = parse_decision(a.json)
        except DecisionParseError as e:
            print(f"[bridge] malformed: {e}", file=sys.stderr)
            return 2
        why = d.refusal(a.threshold, not a.allow_unknown_stop)
        if not a.quiet:
            print(json.dumps({**d.as_dict(),
                              "tradeable": why == "",
                              "refusal": why}, sort_keys=True))
        return 0 if why == "" else 1

    if a.jsonl:
        try:
            with open(a.jsonl, "r", encoding="utf-8") as f:
                rows = parse_jsonl(f)
        except OSError as e:
            print(f"[bridge] cannot read {a.jsonl}: {e}", file=sys.stderr)
            return 2
        decisions, bad = {}, []
        for n, d in rows:
            if isinstance(d, DecisionParseError):
                bad.append({"line": n, "reason": str(d)})
            else:
                decisions[d.target_asset or f"line{n}"] = d
        allowed, refused = screen(decisions, threshold=a.threshold,
                                  require_stop=not a.allow_unknown_stop)
        out = {"allowed": allowed, "refused": refused + bad}
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0 if allowed and not bad else 1

    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
