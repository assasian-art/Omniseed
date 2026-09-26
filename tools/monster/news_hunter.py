# =============================================================================
#  OmniSeed — tools/monster/news_hunter.py
#  Event-driven alpha: high-impact headlines cross-referenced with a
#  volume/liquidity anomaly.
#
#      E = w_k * d(dt) * a          d(dt) = exp(-dt/tau)
#
#    w_k  keyword weight in (0,1]      (CPI/Fed/SEC/Hack/... )
#    a    sentiment alignment 1.0 / 0.5 / 0.0 vs the intended direction
#
#  A MATCH requires an anomaly (volume multiple >= 5x or z >= 3) AND a fresh
#  event inside `match_window`. A match relaxes the sniper's technical-lag
#  requirement — it never relaxes a risk gate.
#
#  Network fetching is the caller's job (see tools/fetch_news.py); this module
#  is pure and fully offline-testable (inject `now_fn`).
# =============================================================================
"""News/event hunter with volume-anomaly cross-reference."""

import math
from dataclasses import dataclass, field

from . import features as F

# --- keyword weights (high-impact market movers) ---------------------------
KEYWORD_WEIGHTS = {
    "cpi": 0.90,
    "inflation": 0.80,
    "fed": 0.90,
    "fomc": 0.90,
    "powell": 0.85,
    "rate hike": 0.85,
    "rate cut": 0.85,
    "sec approval": 1.00,
    "etf approval": 0.95,
    "hack": 1.00,
    "exploit": 0.95,
    "bankruptcy": 0.95,
    "default": 0.85,
    "earnings beat": 0.80,
    "earnings miss": 0.80,
    "guidance cut": 0.85,
    "sanction": 0.80,
    "war": 0.80,
    "whale alert": 0.60,
    "liquidation": 0.75,
}

POSITIVE_TERMS = (
    "beat", "surge", "rally", "soar", "record", "approval", "approved",
    "upgrade", "bullish", "growth", "profit", "gain", "strong", "boost",
    "outperform", "breakout", "adoption", "inflow",
)
NEGATIVE_TERMS = (
    "miss", "plunge", "crash", "slump", "hack", "exploit", "fraud", "bankrupt",
    "downgrade", "bearish", "loss", "weak", "cut", "default", "sanction",
    "panic", "liquidation", "outflow", "halt", "probe",
)


def score_sentiment(text):
    """+1 per positive hit, -1 per negative, normalized by hits, in [-1,1]."""
    low = text.lower()
    pos = sum(1 for t in POSITIVE_TERMS if t in low)
    neg = sum(1 for t in NEGATIVE_TERMS if t in low)
    total = pos + neg
    if total == 0:
        return 0.0
    return max(-1.0, min(1.0, (pos - neg) / total))


def match_keywords(text):
    """-> (best_weight, [matched keywords])."""
    low = text.lower()
    hits, best = [], 0.0
    for kw, w in KEYWORD_WEIGHTS.items():
        if kw in low:
            hits.append(kw)
            best = max(best, w)
    return best, hits


@dataclass
class NewsEvent:
    ts: int = 0
    title: str = ""
    source: str = ""
    weight: float = 0.0
    sentiment: float = 0.0
    keywords: list = field(default_factory=list)
    tickers: list = field(default_factory=list)


@dataclass
class NewsMatch:
    anomaly: bool = False
    vol_mult: float = 0.0
    vol_z: float = 0.0
    event_score: float = 0.0
    event: NewsEvent = None
    event_driven: bool = False
    reason: str = ""


class NewsHunter:
    def __init__(self, cfg=None, now_fn=None):
        c = cfg or {}
        self.tau = float(c.get("tau", 300.0))
        self.match_window = float(c.get("match_window", 300.0))
        self.vol_mult_min = float(c.get("vol_mult_min", 5.0))
        self.vol_z_min = float(c.get("vol_z_min", 3.0))
        self.anom_window = int(c.get("anom_window", 60))
        self.event_min = float(c.get("event_min", 0.25))
        self.now_fn = now_fn
        self.events = []

    # -- ingest ------------------------------------------------------------
    def ingest(self, items):
        """items: dicts with ts/title/summary/source/tickers. -> #accepted."""
        added = 0
        for it in items:
            title = str(it.get("title", ""))
            summary = str(it.get("summary", ""))
            text = title + " " + summary
            w, kws = match_keywords(text)
            if w <= 0.0:
                continue
            self.events.append(NewsEvent(
                ts=int(it.get("ts", 0)), title=title,
                source=str(it.get("source", "")), weight=w,
                sentiment=score_sentiment(text), keywords=kws,
                tickers=list(it.get("tickers", []) or [])))
            added += 1
        self.events.sort(key=lambda e: e.ts)
        return added

    # -- scoring -----------------------------------------------------------
    def event_score(self, ev, now_ts, direction=1):
        """E = w_k * exp(-dt/tau) * alignment."""
        dt = max(0, now_ts - ev.ts)
        decay = math.exp(-dt / self.tau)
        s = ev.sentiment
        if s == 0.0:
            align = 0.5
        elif (s > 0) == (direction > 0):
            align = 1.0
        else:
            align = 0.0
        return ev.weight * decay * align

    def best_event(self, now_ts, direction=1, ticker=None):
        """Highest-scoring event still inside the match window."""
        best, best_score = None, 0.0
        for ev in self.events:
            if now_ts - ev.ts > self.match_window:
                continue
            if ticker and ev.tickers and ticker not in ev.tickers:
                continue
            sc = self.event_score(ev, now_ts, direction)
            if sc > best_score:
                best, best_score = ev, sc
        return best, best_score

    # -- anomaly -----------------------------------------------------------
    def anomaly(self, bars, i):
        """-> (is_anomaly, vol_multiple, z). Uses only bars <= i."""
        v = F.volumes(bars)
        if i < 1:
            return False, 0.0, 0.0
        lo = max(0, i - self.anom_window)
        win = v[lo:i]                      # strictly the past window
        if not win:
            return False, 0.0, 0.0
        mu = sum(win) / len(win)
        if mu <= 0:
            return False, 0.0, 0.0
        mult = v[i] / mu
        var = sum((x - mu) ** 2 for x in win) / len(win)
        sd = math.sqrt(var)
        z = (v[i] - mu) / sd if sd > 0 else 0.0
        hit = (mult >= self.vol_mult_min) or (z >= self.vol_z_min)
        return hit, mult, z

    # -- the cross-reference ----------------------------------------------
    def match(self, bars, i, ts, direction=1, ticker=None):
        anom, mult, z = self.anomaly(bars, i)
        ev, esc = self.best_event(ts, direction, ticker)
        m = NewsMatch(anomaly=anom, vol_mult=mult, vol_z=z,
                      event_score=esc, event=ev)
        if anom and ev is not None and esc >= self.event_min:
            m.event_driven = True
            m.reason = ("event-driven: %s (E=%.2f) x vol %.1fx (z=%.1f)"
                        % ("+".join(ev.keywords), esc, mult, z))
        elif anom:
            m.reason = "volume-anomaly without a matching headline"
        elif ev is not None:
            m.reason = "headline without a volume anomaly"
        else:
            m.reason = "quiet"
        return m
