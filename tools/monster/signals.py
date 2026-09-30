# =============================================================================
#  OmniSeed — tools/monster/signals.py
#  THE EXTERNAL-SIGNAL LAYER — the "hacker tools" the Monster was missing.
#
#  docs/TRADING_LAB.md §7 named four signal families as planned-but-not-built:
#
#      on-chain flows · event calendar · social volume · a live depth provider
#      for the order-book path
#
#  This module builds them as ONE layer under three house rules:
#
#    1. PURE + OFFLINE.  Every reader takes a path (or an injected list); the
#       clock is injectable.  Nothing here opens a socket.  Fetching stays the
#       caller's job, exactly like tools/fetch_news.py.
#    2. PROVENANCE IS NAMED.  A depth-derived imbalance says `cks`; the
#       signed-volume fallback says `proxy` and is DISCOUNTED, because the proxy
#       cannot see limit orders or cancellations — which is most of what the CKS
#       OFI actually measures.  A feed that is stale or absent says so and
#       contributes NOTHING; it is never silently zero-filled into a vote.
#    3. FAIL-CLOSED, NEVER PROMOTING.  The event calendar is a BLOCKER: inside a
#       high-impact blackout window the layer reports `blackout` and the scanner
#       vetoes the entry.  No external signal can raise the sniper score `S` —
#       `S` stays the mandate's weighted sum of M/T/R/C.
#
#  Nothing here claims an edge.  These are filters and context.  The test suite
#  pins the negative controls: a stale feed is not a vote, a low-impact event is
#  not a blackout, and the proxy is labelled AND discounted.
# =============================================================================
"""External signals: on-chain flows, social volume, an event calendar, the
commodity complex, and a real order-book depth path."""

import math
from dataclasses import dataclass, field

from . import features as F
from . import strategies as ST

NAN = F.NAN

# The commodity legs the cross-asset matrix already knows how to use
# (correlation_matrix.DEFAULT_PAIRS names OIL / GOLD / COMMODITY_INDEX).
COMMODITY_UNIVERSE = ("OIL", "GOLD", "COPPER")


def _clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


# ---------------------------------------------------------------------------
# Readers — strict.  A malformed row is SKIPPED, never repaired, never invented.
# ---------------------------------------------------------------------------
def _numeric_rows(path, ncol):
    """path -> [[ts, f1, ... ]] ascending.  Skips anything that will not parse."""
    out = []
    try:
        f = open(path, "r", encoding="utf-8")
    except OSError:
        return out
    with f:
        for line in f:
            line = line.strip()
            if not line or not line[0].isdigit():
                continue
            p = line.split(",")
            if len(p) < ncol:
                continue
            try:
                out.append([int(p[0])] + [float(x) for x in p[1:ncol]])
            except ValueError:
                continue
    out.sort(key=lambda r: r[0])
    return out


def read_depth_csv(path):
    """ts,bid_px,bid_sz,ask_px,ask_sz -> [[ts, bp, bq, ap, aq], ...]."""
    return _numeric_rows(path, 5)


def read_onchain_csv(path):
    """ts,exchange_netflow,whale_net,stablecoin_delta -> [[ts, ex, wh, st], ...].

    Units are the caller's (native coin or USD); `scale` normalises them.
    """
    return _numeric_rows(path, 4)


def read_social_csv(path):
    """ts,mentions,positive,negative -> [[ts, mentions, pos, neg], ...]."""
    return _numeric_rows(path, 4)


def read_calendar_csv(path):
    """ts,kind,name,impact -> [(ts, kind, name, impact)] ascending."""
    out = []
    try:
        f = open(path, "r", encoding="utf-8")
    except OSError:
        return out
    with f:
        for line in f:
            line = line.strip()
            if not line or not line[0].isdigit():
                continue
            p = line.split(",")
            if len(p) < 4:
                continue
            try:
                ts = int(p[0])
                impact = float(p[3])
            except ValueError:
                continue
            out.append((ts, p[1].strip(), p[2].strip(), impact))
    out.sort(key=lambda r: r[0])
    return out


def read_commodity_csv(path):
    """ts,symbol,close -> {symbol: [(ts, close), ...]}."""
    out = {}
    try:
        f = open(path, "r", encoding="utf-8")
    except OSError:
        return out
    with f:
        for line in f:
            line = line.strip()
            if not line or not line[0].isdigit():
                continue
            p = line.split(",")
            if len(p) < 3:
                continue
            try:
                ts = int(p[0])
                close = float(p[2])
            except ValueError:
                continue
            out.setdefault(p[1].strip(), []).append((ts, close))
    for v in out.values():
        v.sort(key=lambda r: r[0])
    return out


# ---------------------------------------------------------------------------
# 1. ORDER BOOK — the depth path
# ---------------------------------------------------------------------------
@dataclass
class DepthSignal:
    ts: int = 0
    ofi: float = 0.0
    beta: float = NAN
    imbalance: float = 0.0
    depth: float = NAN
    source: str = "none"          # cks | proxy | none
    discount: float = 1.0


class DepthBook:
    """Best-quote snapshots -> the exact CKS OFI, or a labelled proxy.

    With a book the layer computes `strategies.ofi_from_snapshots` (exact CKS)
    and `strategies.impact_beta(depth, c, lam)`.  Without one it falls back to
    the signed-volume proxy, labelled `proxy` and multiplied by
    `proxy_discount` — the two objects are NOT the same thing and the layer
    refuses to pretend they are.
    """

    def __init__(self, cfg=None):
        c = cfg or {}
        self.c = float(c.get("c", 1.0))
        self.lam = float(c.get("lam", 1.0))
        self.norm = float(c.get("norm", 100.0))
        self.proxy_discount = float(c.get("proxy_discount", 0.70))
        self.proxy_n = int(c.get("proxy_n", 20))
        self.snaps = []

    def ingest(self, rows):
        for r in rows:
            self.snaps.append((int(r[0]), float(r[1]), float(r[2]),
                               float(r[3]), float(r[4])))
        self.snaps.sort(key=lambda s: s[0])
        return len(self.snaps)

    def _book_upto(self, ts):
        return [(bp, bq, ap, aq) for (t, bp, bq, ap, aq) in self.snaps if t <= ts]

    def signal(self, ts, bars=None, i=None):
        sig = DepthSignal(ts=ts)
        book = self._book_upto(ts)
        if book:
            bp, bq, ap, aq = book[-1]
            depth = bq + aq
            sig.ofi = ST.ofi_from_snapshots(book)
            sig.depth = depth
            sig.beta = ST.impact_beta(depth, self.c, self.lam)
            sig.imbalance = _clamp(sig.ofi / self.norm, -1.0, 1.0)
            sig.source = "cks"
            return sig
        if bars is not None and i is not None:
            proxy = ST.ofi_proxy(bars, self.proxy_n)
            v = proxy[i] if 0 <= i < len(proxy) else 0.0
            if not math.isfinite(v):
                v = 0.0
            sig.ofi = v
            sig.imbalance = _clamp(v, -1.0, 1.0) * self.proxy_discount
            sig.discount = self.proxy_discount
            sig.source = "proxy"
            return sig
        return sig                                   # source stays "none"


# ---------------------------------------------------------------------------
# 2. ON-CHAIN — whale / exchange in-out, stablecoin supply
# ---------------------------------------------------------------------------
@dataclass
class OnChainSignal:
    ts: int = 0
    score: float = 0.0            # signed, in [-1,1]
    exchange: float = 0.0
    whale: float = 0.0
    stablecoin: float = 0.0
    age: float = -1.0
    stale: bool = True
    source: str = "none"


class OnChainFlow:
    """The signs are the standard reading, written down so they can be argued:

      * exchange NET INFLOW  (coins moving TO venues)  = sell pressure = bearish
      * whale NET ACCUMULATION (a positive net buy)    = bullish
      * rising stablecoin supply (dry powder)          = bullish

    Each raw magnitude is squashed with tanh(raw/scale), so no single print can
    saturate the score.  A reading older than `max_age` is STALE: it returns
    `source="none"` and a zero score rather than a stale vote.
    """

    def __init__(self, cfg=None):
        c = cfg or {}
        self.w_exchange = float(c.get("w_exchange", 0.5))
        self.w_whale = float(c.get("w_whale", 0.3))
        self.w_stable = float(c.get("w_stable", 0.2))
        self.scale = float(c.get("scale", 1000.0))
        self.max_age = float(c.get("max_age", 86400.0))
        self.rows = []

    def ingest(self, rows):
        for r in rows:
            self.rows.append((int(r[0]), float(r[1]), float(r[2]), float(r[3])))
        self.rows.sort(key=lambda r: r[0])
        return len(self.rows)

    def signal(self, ts):
        sig = OnChainSignal(ts=ts)
        row = None
        for r in self.rows:
            if r[0] <= ts:
                row = r
            else:
                break
        if row is None:
            return sig
        sig.age = float(ts - row[0])
        if sig.age > self.max_age:
            return sig                               # stale: no vote, named
        ex = math.tanh(row[1] / self.scale)
        wh = math.tanh(row[2] / self.scale)
        st = math.tanh(row[3] / self.scale)
        sig.exchange, sig.whale, sig.stablecoin = ex, wh, st
        sig.score = _clamp(self.w_exchange * (-ex)
                           + self.w_whale * wh
                           + self.w_stable * st, -1.0, 1.0)
        sig.stale = False
        sig.source = "onchain"
        return sig


# ---------------------------------------------------------------------------
# 3. SOCIAL — mention volume z-score x sentiment
# ---------------------------------------------------------------------------
@dataclass
class SocialSignal:
    ts: int = 0
    attention: float = 0.0        # [0,1]
    sentiment: float = 0.0        # [-1,1]
    bias: float = 0.0             # attention * sentiment
    z: float = 0.0
    stale: bool = True
    source: str = "none"


class SocialVolume:
    """Attention is the mention-volume z-score against the TRAILING window
    (strictly the past — no look-ahead), squashed to [0,1].  Bias is attention
    times sentiment, so a noisy but directionless crowd scores ~0.

    The window excludes the current row, which is why the first `window` rows
    can never carry an attention reading.
    """

    def __init__(self, cfg=None):
        c = cfg or {}
        self.window = int(c.get("window", 30))
        self.z_full = float(c.get("z_full", 3.0))
        self.max_age = float(c.get("max_age", 86400.0))
        self.rows = []

    def ingest(self, rows):
        for r in rows:
            self.rows.append((int(r[0]), float(r[1]), float(r[2]), float(r[3])))
        self.rows.sort(key=lambda r: r[0])
        return len(self.rows)

    def signal(self, ts):
        sig = SocialSignal(ts=ts)
        idx = -1
        for k, r in enumerate(self.rows):
            if r[0] <= ts:
                idx = k
            else:
                break
        if idx < 0:
            return sig
        if float(ts - self.rows[idx][0]) > self.max_age:
            return sig
        win = [r[1] for r in self.rows[max(0, idx - self.window):idx]]
        if len(win) >= 5:
            mu = sum(win) / len(win)
            var = sum((x - mu) ** 2 for x in win) / len(win)
            sd = math.sqrt(var)
            sig.z = (self.rows[idx][1] - mu) / sd if sd > 0 else 0.0
        sig.attention = _clamp(sig.z / self.z_full, 0.0, 1.0) if self.z_full > 0 else 0.0
        _ts, _m, pos, neg = self.rows[idx]
        tot = pos + neg
        sig.sentiment = _clamp((pos - neg) / tot, -1.0, 1.0) if tot > 0 else 0.0
        sig.bias = sig.attention * sig.sentiment
        sig.stale = False
        sig.source = "social"
        return sig


# ---------------------------------------------------------------------------
# 4. EVENT CALENDAR — a fail-closed blackout window
# ---------------------------------------------------------------------------
@dataclass
class CalEvent:
    ts: int = 0
    kind: str = ""
    name: str = ""
    impact: float = 0.0


@dataclass
class CalSignal:
    ts: int = 0
    blackout: bool = False
    reason: str = ""
    decay: float = 0.0
    event: str = ""


class EventCalendar:
    """Scheduled events.  A HIGH-impact event (`impact >= impact_min`) opens the
    blackout window `[ts - pre, ts + post]`; inside it the scanner vetoes the
    entry.  A low-impact event NEVER blackouts — it only decays.

    `decay` is the post-event pressure `impact * exp(-dt/tau)` and is reported
    for context; it never gates anything on its own.
    """

    def __init__(self, cfg=None):
        c = cfg or {}
        self.pre = float(c.get("pre", 3600.0))
        self.post = float(c.get("post", 1800.0))
        self.impact_min = float(c.get("impact_min", 0.7))
        self.tau = float(c.get("tau", 3600.0))
        self.events = []

    def ingest(self, rows):
        for (ts, kind, name, impact) in rows:
            self.events.append(CalEvent(int(ts), str(kind), str(name),
                                        float(impact)))
        self.events.sort(key=lambda e: e.ts)
        return len(self.events)

    def signal(self, ts):
        sig = CalSignal(ts=ts)
        for e in self.events:
            if e.impact >= self.impact_min and (e.ts - self.pre) <= ts <= (e.ts + self.post):
                sig.blackout = True
                sig.reason = "%s:%s" % (e.kind, e.name) if e.kind else e.name
                sig.event = e.name
                break
        for e in self.events:
            dt = ts - e.ts
            if dt >= 0:
                sig.decay = max(sig.decay, e.impact * math.exp(-dt / self.tau))
        return sig


# ---------------------------------------------------------------------------
# 5. COMMODITY COMPLEX — legs for the cross-asset matrix
# ---------------------------------------------------------------------------
def commodity_returns(series_by_symbol, symbols=COMMODITY_UNIVERSE):
    """-> {symbol: [log returns]} for whichever commodity legs are present."""
    out = {}
    for s in symbols:
        rows = series_by_symbol.get(s)
        if not rows:
            continue
        out[s] = F.log_returns([c for (_t, c) in rows])
    return out


# ---------------------------------------------------------------------------
# The per-bar pack + its CSV contract (state/monster/signals.csv)
# ---------------------------------------------------------------------------
@dataclass
class SignalPack:
    ts: int = 0
    onchain: float = 0.0
    social: float = 0.0
    depth: float = 0.0
    depth_source: str = "none"
    blackout: bool = False
    blackout_reason: str = ""
    coverage: int = 0

    def detail(self):
        """Compact, CSV-safe rationale (no commas)."""
        bits = []
        if self.onchain:
            bits.append("oc=%+.2f" % self.onchain)
        if self.social:
            bits.append("soc=%+.2f" % self.social)
        if self.depth_source != "none":
            bits.append("dep=%s%+.2f" % (self.depth_source, self.depth))
        if self.blackout:
            bits.append("blackout")
        return "+".join(bits) if bits else "none"


SIGNAL_HEADER = "ts,onchain,social,depth,depth_source,blackout,coverage"


def to_row(p):
    reason = p.blackout_reason.replace(",", ";").replace("\n", " ")
    extra = ("|" + reason) if reason else ""
    return "%d,%.4f,%.4f,%.4f,%s,%d,%d%s" % (
        p.ts, p.onchain, p.social, p.depth, p.depth_source,
        1 if p.blackout else 0, p.coverage, extra)


def write_csv(path, packs):
    import os
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(SIGNAL_HEADER + "\n")
        for p in packs:
            f.write(to_row(p) + "\n")
    return len(packs)


def read_csv(path):
    """-> {ts: SignalPack}.  Mirrors the writer; malformed rows are skipped."""
    out = {}
    try:
        f = open(path, "r", encoding="utf-8")
    except OSError:
        return out
    with f:
        for line in f:
            line = line.strip()
            if not line or not line[0].isdigit():
                continue
            p = line.split(",", 6)
            if len(p) < 7:
                continue
            try:
                pk = SignalPack(
                    ts=int(p[0]), onchain=float(p[1]), social=float(p[2]),
                    depth=float(p[3]), depth_source=p[4].strip(),
                    blackout=p[5].strip() == "1", coverage=int(p[6].split("|", 1)[0]))
            except ValueError:
                continue
            if "|" in p[6]:
                pk.blackout_reason = p[6].split("|", 1)[1]
            out[pk.ts] = pk
    return out


# ---------------------------------------------------------------------------
# The aggregator the scanner talks to
# ---------------------------------------------------------------------------
class SignalSet:
    """Holds every external feed and answers per-bar questions.

    `coverage` counts how many of the four families produced a LIVE reading for
    a bar (0..4).  The scanner reports it, so "no signal" and "no feed" are
    never confused — which is the whole point of the layer.
    """

    def __init__(self, cfg=None, depth=None, onchain=None, social=None,
                 calendar=None):
        self.depth = depth if depth is not None else DepthBook(cfg)
        self.onchain = onchain if onchain is not None else OnChainFlow(cfg)
        self.social = social if social is not None else SocialVolume(cfg)
        self.calendar = calendar if calendar is not None else EventCalendar(cfg)

    @classmethod
    def from_dir(cls, path, cfg=None):
        import os
        s = cls(cfg)
        s.depth.ingest(read_depth_csv(os.path.join(path, "depth.csv")))
        s.onchain.ingest(read_onchain_csv(os.path.join(path, "onchain.csv")))
        s.social.ingest(read_social_csv(os.path.join(path, "social.csv")))
        s.calendar.ingest(read_calendar_csv(os.path.join(path, "calendar.csv")))
        return s

    def feeds(self):
        return {
            "depth": len(self.depth.snaps),
            "onchain": len(self.onchain.rows),
            "social": len(self.social.rows),
            "calendar": len(self.calendar.events),
        }

    def empty(self):
        return not any(self.feeds().values())

    def pack(self, ts, bars=None, i=None):
        p = SignalPack(ts=ts)
        d = self.depth.signal(ts, bars, i)
        oc = self.onchain.signal(ts)
        so = self.social.signal(ts)
        cal = self.calendar.signal(ts)
        p.depth, p.depth_source = d.imbalance, d.source
        p.onchain = oc.score
        p.social = so.bias
        p.blackout, p.blackout_reason = cal.blackout, cal.reason
        p.coverage = sum(1 for s in (d.source, oc.source, so.source)
                         if s != "none")
        return p
