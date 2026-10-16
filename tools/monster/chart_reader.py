# =============================================================================
#  OmniSeed — tools/monster/chart_reader.py
#  Candlestick pattern reading -> StrategySignal (Phase 3.1, chart path).
#
#  The vision pipeline answers "what does this IMAGE contain"; this module
#  answers the trading-specific version — "what do these CANDLES say" — using
#  the bar series itself, so it needs no image model, no dataset, and no
#  network. It is deliberately a ZOO SIGNAL, not a new score: it emits the
#  same StrategySignal contract as momentum/mean-reversion, so the router,
#  the factory and every veto apply to it unchanged.
#
#  PARITY-SAFE BY CONSTRUCTION: this module is NOT in strategies.all_signals,
#  whose 10-signal order is pinned by test_strategy_parity at 1e-9 on both
#  sides of the language line. It is opt-in (monster_scan --chart) and its
#  output is INFORMATION (a per-bar pattern CSV), never an amplifier of S —
#  the same adoption discipline every other new layer here has faced.
#
#  NO LOOK-AHEAD: every pattern reads bars at indices <= i. The breakout
#  channel is built from bars strictly BEFORE i; a two-bar pattern uses i-1
#  and i only.
#
#  HONEST SCOPE: candlestick patterns are contested evidence — their hit
#  rates in academic meta-studies are near noise unfiltered. What they ARE
#  is deterministic, testable structure in the bars; treating them as one
#  low-weight zoo voice behind the same vetoes is the honest use.
# =============================================================================
"""Candlestick patterns as zoo signals: engulfing, hammer, marubozu, doji,
inside-bar compression, channel breakout. Opt-in, fail-closed, no look-ahead."""

import math

from . import features as F
from .features import NAN
from .strategies import StrategySignal

# Fixed order — tests pin it, and a silent reorder would be invisible
# everywhere else (the same reason the M4 zoo pins its oracle order).
PATTERN_ORDER = ("engulfing", "hammer", "marubozu", "doji",
                 "inside_bar", "breakout")

# pattern -> (direction_multiplier, regime_fit, base_confidence)
# Confidence is deliberately MODEST: these are contested patterns, and the
# router weights this voice against measured ones.
_PROFILE = {
    "engulfing":  (+1, "range",  0.60),
    "hammer":     (+1, "range",  0.55),
    "marubozu":   (+1, "trend",  0.60),
    "doji":       (0,  "both",   0.0),
    "inside_bar": (0,  "both",   0.0),
    "breakout":   (+1, "trend",  0.65),
}


def _bar(b):
    """(o,h,l,c,v) tuple from a dict or a 6-tuple; None if malformed."""
    if b is None:
        return None
    if isinstance(b, dict):
        try:
            return (float(b["open"]), float(b["high"]), float(b["low"]),
                    float(b["close"]),
                    float(b.get("volume", 0.0)) if b.get("volume") else 0.0)
        except (KeyError, TypeError, ValueError):
            return None
    try:
        t = tuple(float(x) for x in b[:5])
        return t
    except (TypeError, ValueError):
        return None


def _inactive(name, reason):
    return StrategySignal("chart." + name, 0.0, 0.0,
                          _PROFILE[name][1], reason)


def chart_signals(bars, i, lookback=20):
    """All pattern verdicts at bar `i`, in PATTERN_ORDER.

    Every call returns exactly len(PATTERN_ORDER) signals: a qualifying
    pattern carries its conviction, everything else is INACTIVE with a
    written reason — the same fail-closed visibility the M4 options layer
    chose over silence.
    """
    out = []
    cur = _bar(bars[i]) if 0 <= i < len(bars) else None
    # bounds-check BOTH ends: i>=1 is not enough when the caller passes an
    # index beyond the series (the test suite caught exactly this crash).
    prev = _bar(bars[i - 1]) if (i >= 1 and i - 1 < len(bars)) else None

    if cur is None:
        return [_inactive(p, "bad-bar") for p in PATTERN_ORDER]

    o, h, l, c, v = cur
    rng = h - l
    body = c - o
    abody = abs(body)

    # A zero-range bar carries no candle information at all. Guarded once,
    # up front, because every ratio below would otherwise divide by it.
    if not (rng > 0):
        return [_inactive(p, "zero-range") for p in PATTERN_ORDER]

    upper = h - max(o, c)
    lower = min(o, c) - l

    for p in PATTERN_ORDER:
        sig = None
        if p == "engulfing" and prev is not None:
            po, ph, pl, pc, _pv = prev
            pbody = pc - po
            if pbody < 0 and body > 0 and abody > abs(pbody) \
                    and c > po and o < pc:
                sig = StrategySignal("chart.engulfing", 1.0, 0.60, "range",
                                     "bullish-engulfing body %.2f>%.2f"
                                     % (abody, abs(pbody)))
            elif pbody > 0 and body < 0 and abody > abs(pbody) \
                    and c < po and o > pc:
                sig = StrategySignal("chart.engulfing", -1.0, 0.60, "range",
                                     "bearish-engulfing body %.2f>%.2f"
                                     % (abody, abs(pbody)))
        elif p == "hammer":
            # small body parked at one end of the range: the wick ratio is
            # the whole pattern. Bullish hammer at a LOW, shooting star at
            # a HIGH — the same shape means opposite things by location.
            small = abody <= 0.35 * rng
            if small and lower >= 2.0 * abody and upper <= abody:
                sig = StrategySignal("chart.hammer", 1.0, 0.55, "range",
                                     "hammer lower-wick %.2fx body"
                                     % (lower / abody if abody > 0 else 99.0))
            elif small and upper >= 2.0 * abody and lower <= abody:
                sig = StrategySignal("chart.hammer", -1.0, 0.55, "range",
                                     "shooting-star upper-wick %.2fx body"
                                     % (upper / abody if abody > 0 else 99.0))
        elif p == "marubozu":
            # a body that IS the bar: no wick worth the name.
            if abody >= 0.9 * rng and abody > 0:
                d = 1.0 if body > 0 else -1.0
                sig = StrategySignal(
                    "chart.marubozu", d, 0.60, "trend",
                    "%s-marubozu body %.0f%% of range"
                    % ("bull" if d > 0 else "bear", 100.0 * abody / rng))
        elif p == "doji":
            if abody <= 0.10 * rng:
                # Indecision is a VERDICT, not a trade: direction 0 and
                # confidence 0 keep it out of the weighted vote entirely.
                sig = _inactive("doji", "doji body %.0f%% of range — indecision"
                                        % (100.0 * abody / rng))
        elif p == "inside_bar":
            if prev is not None:
                _po, ph, pl, _pc, _pv = prev
                if h <= ph and l >= pl and rng < (ph - pl):
                    # compression: is this range also small vs recent history?
                    hist = [_bar(bars[j]) for j in
                            range(max(0, i - lookback), i)]
                    hist = [b for b in hist if b and (b[1] - b[2]) > 0]
                    if hist:
                        med = sorted(b[1] - b[2] for b in hist)[len(hist) // 2]
                        if rng <= 0.7 * med:
                            sig = _inactive(
                                "inside_bar",
                                "inside-bar compression %.2f<=0.7*median %.2f "
                                "— awaiting resolution" % (rng, med))
        elif p == "breakout":
            # channel from bars STRICTLY BEFORE i (a breakout is "above the
            # prior range", never "above a range that includes itself").
            hh = ll = None
            used = 0
            for j in range(max(0, i - lookback), i):
                bj = _bar(bars[j])
                if bj is None:
                    continue
                used += 1
                hh = bj[1] if hh is None else max(hh, bj[1])
                ll = bj[2] if ll is None else min(ll, bj[2])
            if hh is not None and ll is not None:
                # the reason reports the bars ACTUALLY scanned, not the
                # nominal lookback — "20bar-high" over 10 existing bars
                # would overstate the channel's depth.
                if c > hh:
                    sig = StrategySignal(
                        "chart.breakout", 1.0, 0.65, "trend",
                        "close %.2f above %dbar-high %.2f" % (c, used, hh))
                elif c < ll:
                    sig = StrategySignal(
                        "chart.breakout", -1.0, 0.65, "trend",
                        "close %.2f below %dbar-low %.2f" % (c, used, ll))

        out.append(sig if sig is not None
                   else _inactive(p, "no-%s" % p))
    return out


def read_chart_csv(path):
    """Provenance CSV -> list of dict bars (the house `time,...` schema).

    Malformed rows are skipped, mirroring the funding reader: a broken row is
    missing information, never a guessed bar.
    """
    import csv
    out = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                out.append({"time": row["time"],
                            "open": float(row["open"]),
                            "high": float(row["high"]),
                            "low": float(row["low"]),
                            "close": float(row["close"]),
                            "volume": float(row["volume"] or 0.0)})
            except (KeyError, TypeError, ValueError):
                continue
    return out
