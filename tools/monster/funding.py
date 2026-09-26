# =============================================================================
#  OmniSeed — tools/monster/funding.py
#  Crypto perpetual funding-rate carry scanner.
#
#  A DIFFERENT KIND OF EDGE. Everything else in the Monster layer is
#  directional — it tries to be right about where price goes. This one does not
#  care. Perpetual futures have no expiry, so exchanges invented the funding
#  rate to tether them to spot: when longs dominate, longs pay shorts. Hold spot
#  long and perp short in equal notional and the price exposure cancels, leaving
#  the funding payment as the only P&L. That is a CARRY trade, not a forecast.
#
#  THE ARITHMETIC (and the mistake everyone makes)
#      APR(gross) = rate_per_period * periods_per_year
#      periods_per_year = (24 / interval_hours) * 365     -> 1095 for 8h
#  A rate of 0.012% per 8h annualizes to 13.14%. The mistake is ranking on the
#  GROSS number. A memecoin can show 200% gross while its spot book is 150bps
#  wide and the perp slips another 100bps on size, so the trade loses from the
#  first minute. So we subtract the real costs and rank on NET:
#
#      net = gross - roundtrip_fees*(365/holding) - spread*(365/holding)
#                  - cost_of_capital
#
#  IT IS DELTA-NEUTRAL, NOT RISK-FREE. The failure modes are operational and
#  they are real: the funding rate flips sign and you start paying; the perp leg
#  gets liquidated on a violent basis move and leaves you naked long spot;
#  the spot venue freezes withdrawals; a stablecoin de-pegs and your "yield" is
#  denominated in a depreciating asset. Every one of these is flagged below
#  rather than assumed away.
#
#  HONEST SCOPE: no guaranteed profit. The edge is structural but it is a
#  harvest, and harvests fail in storms. See docs/MONSTER_DESIGN.md §7.
# =============================================================================
"""Funding-rate carry: annualization, net-of-cost ranking, risk flags.

This trade is DELTA-NEUTRAL, NOT RISK-FREE: the funding rate can flip sign and
make you the payer, the perp leg can be liquidated on a basis shock and leave
you naked long spot, and the spot venue can freeze withdrawals. Every one of
those is flagged rather than assumed away.

There is NO GUARANTEED PROFIT here and none is claimed. Funding carry is a
harvest of a structural premium, and harvests fail in storms. Ranking is done
on NET apr after fees, spread and cost of capital, because the gross number is
the one that gets people hurt.
"""

import math

# ---------------------------------------------------------------------------
# arithmetic
# ---------------------------------------------------------------------------


def periods_per_year(interval_hours=8.0):
    """Funding settlements per year. 8h -> 1095, 4h -> 2190, 1h -> 8760."""
    if interval_hours <= 0:
        raise ValueError("interval_hours must be > 0")
    return (24.0 / interval_hours) * 365.0


def annualize(rate, interval_hours=8.0):
    """Per-period funding rate -> gross APR (as a fraction, not percent)."""
    return rate * periods_per_year(interval_hours)


def bps_to_fraction(bps):
    return bps / 10000.0


def net_apr(gross_apr, cfg):
    """Gross APR -> net APR after round-trip fees, spread and cost of capital.

    Costs are one-time; they are annualized over the intended holding period,
    because a 18bps round trip is trivial over a month and fatal intraday.
    """
    rt_fees = 2.0 * (cfg.spot_fee_bps + cfg.perp_fee_bps)
    rt_spread = 2.0 * (cfg.spot_spread_bps + cfg.perp_spread_bps)
    one_off = bps_to_fraction(rt_fees + rt_spread)
    hold = max(cfg.holding_days, 1e-9)
    amortized = one_off * (365.0 / hold)
    return gross_apr - amortized - cfg.cost_of_capital_apr


class FundingConfig:
    interval_hours: float = 8.0
    hurdle_apr: float = 0.10        # minimum NET apr to bother with
    spot_fee_bps: float = 4.0       # taker, one way
    perp_fee_bps: float = 5.0       # taker, one way
    spot_spread_bps: float = 1.0
    perp_spread_bps: float = 2.0
    cost_of_capital_apr: float = 0.08
    holding_days: float = 30.0
    max_basis_bps: float = 300.0    # above this, convergence-shock risk
    min_rate_bps: float = 0.5       # below this the carry is not worth the ops
    flip_lookback: int = 21         # settlements inspected for sign flips
    max_spot_spread_bps: float = 50.0

    def __init__(self, **kw):
        for k, v in kw.items():
            if not hasattr(self, k):
                raise TypeError("unknown FundingConfig field: %s" % k)
            setattr(self, k, v)


# ---------------------------------------------------------------------------
# quotes / trades
# ---------------------------------------------------------------------------
class FundingQuote:
    """One instrument's funding snapshot, as supplied by the caller."""

    __slots__ = ("symbol", "exchange", "rate", "interval_hours", "spot_price",
                 "perp_price", "spot_spread_bps", "perp_spread_bps",
                 "history", "open_interest")

    def __init__(self, symbol, rate, interval_hours=8.0, exchange="",
                 spot_price=None, perp_price=None, spot_spread_bps=None,
                 perp_spread_bps=None, history=None, open_interest=None):
        self.symbol = symbol
        self.exchange = exchange
        self.rate = float(rate)
        self.interval_hours = float(interval_hours)
        self.spot_price = spot_price
        self.perp_price = perp_price
        self.spot_spread_bps = spot_spread_bps
        self.perp_spread_bps = perp_spread_bps
        self.history = list(history) if history else []
        self.open_interest = open_interest

    @property
    def basis_bps(self):
        """Perp premium over spot, in bps. Positive = contango."""
        if not (self.spot_price and self.perp_price):
            return float("nan")
        if self.spot_price <= 0:
            return float("nan")
        return (self.perp_price / self.spot_price - 1.0) * 10000.0


class BasisTrade:
    __slots__ = ("symbol", "exchange", "direction", "gross_apr", "net_apr",
                 "basis_bps", "rate_bps", "risks", "blocked", "verdict")

    def __init__(self, symbol, exchange, direction, gross_apr, net_apr,
                 basis_bps, rate_bps, risks, blocked, verdict):
        self.symbol = symbol
        self.exchange = exchange
        self.direction = direction
        self.gross_apr = gross_apr
        self.net_apr = net_apr
        self.basis_bps = basis_bps
        self.rate_bps = rate_bps
        self.risks = risks
        self.blocked = blocked
        self.verdict = verdict

    def detail(self):
        return ("%s %s gross=%.2f%% net=%.2f%% basis=%.0fbps %s"
                % (self.symbol, self.direction, self.gross_apr * 100,
                   self.net_apr * 100, self.basis_bps,
                   ("RISK:" + ",".join(self.risks)) if self.risks else "clean"))

    def __repr__(self):
        return "BasisTrade(%s)" % self.detail()


# ---------------------------------------------------------------------------
# history summary
# ---------------------------------------------------------------------------
def summarize_history(rates, interval_hours=8.0):
    """-> dict of what the funding history says about durability.

    A carry trade dies when the rate flips sign, so the useful questions are:
    how often is it positive, how volatile is it, and how long is the current
    run? We answer all three instead of quoting a single spot number.
    """
    r = [float(x) for x in rates if x is not None]
    if not r:
        return {"n": 0, "mean_apr": float("nan"), "vol_apr": float("nan"),
                "pct_positive": float("nan"), "current_streak": 0,
                "flips": 0}
    ppy = periods_per_year(interval_hours)
    m = sum(r) / len(r)
    var = sum((x - m) ** 2 for x in r) / len(r) if len(r) > 1 else 0.0
    pos = sum(1 for x in r if x > 0)
    flips = sum(1 for i in range(1, len(r)) if (r[i] > 0) != (r[i - 1] > 0))
    streak = 0
    sign = 1 if r[-1] > 0 else -1
    for x in reversed(r):
        if (x > 0) == (sign > 0) and x != 0:
            streak += 1
        else:
            break
    return {
        "n": len(r),
        "mean_apr": m * ppy,
        "vol_apr": math.sqrt(var) * ppy,
        "pct_positive": pos / float(len(r)),
        "current_streak": streak,
        "flips": flips,
    }


# ---------------------------------------------------------------------------
# the scanner
# ---------------------------------------------------------------------------
def evaluate(quote, cfg=None):
    """One quote -> BasisTrade with costs netted out and risks flagged."""
    cfg = cfg or FundingConfig()
    interval = quote.interval_hours or cfg.interval_hours
    gross = annualize(abs(quote.rate), interval)
    net = net_apr(gross, cfg)

    # --- risk flags --------------------------------------------------------
    risks = []
    blocked = False

    hist = summarize_history(quote.history, interval) if quote.history else None
    if hist and hist["n"] >= 5:
        if hist["pct_positive"] < 0.6:
            risks.append("funding-unstable")
            blocked = True
        if hist["flips"] > max(2, hist["n"] // 5):
            risks.append("funding-flips")

    # a rate this close to zero is about to cross
    if abs(quote.rate) * 10000.0 < cfg.min_rate_bps:
        risks.append("rate-near-zero")
        blocked = True

    b = quote.basis_bps
    if math.isfinite(b) and abs(b) > cfg.max_basis_bps:
        risks.append("basis-shock")
        blocked = True

    if quote.spot_spread_bps is not None and quote.spot_spread_bps > cfg.max_spot_spread_bps:
        risks.append("illiquid-spot")
        blocked = True

    if net < cfg.hurdle_apr:
        risks.append("below-hurdle")
        blocked = True

    direction = "short-perp" if quote.rate > 0 else "long-perp"
    if blocked:
        verdict = "skip"
    else:
        verdict = "take"
    return BasisTrade(quote.symbol, quote.exchange, direction, gross, net,
                      b if math.isfinite(b) else float("nan"),
                      quote.rate * 10000.0, risks, blocked, verdict)


def scan(quotes, cfg=None):
    """Evaluate every quote, best NET apr first. Blocked trades sink to the end."""
    cfg = cfg or FundingConfig()
    trades = [evaluate(q, cfg) for q in quotes]
    trades.sort(key=lambda t: (t.blocked, -t.net_apr))
    return trades


def takeable(trades):
    return [t for t in trades if not t.blocked]


def load_quotes_csv(path, interval_hours=8.0):
    """Read a funding snapshot CSV.

    Expected header: symbol,rate[,exchange,spot_price,perp_price,
                              spot_spread_bps,perp_spread_bps]
    `rate` is the PER-PERIOD rate as a fraction (0.00012 == 0.012%).
    Malformed rows are skipped rather than guessed at.
    """
    import csv
    out = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            try:
                sym = (row.get("symbol") or "").strip()
                if not sym:
                    continue
                rate = float(row["rate"])
            except (KeyError, TypeError, ValueError):
                continue

            def _f(key):
                v = row.get(key)
                if v is None or v == "":
                    return None
                try:
                    return float(v)
                except ValueError:
                    return None

            out.append(FundingQuote(
                sym, rate, interval_hours=interval_hours,
                exchange=(row.get("exchange") or "").strip(),
                spot_price=_f("spot_price"), perp_price=_f("perp_price"),
                spot_spread_bps=_f("spot_spread_bps"),
                perp_spread_bps=_f("perp_spread_bps")))
    return out
