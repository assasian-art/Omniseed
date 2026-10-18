# =============================================================================
#  OmniSeed — tools/monster/exotics.py  (Phase 7)
#  The "exotic capabilities" layer: fractal, chaos, causal, counterfactual,
#  game theory, evolution, reinforcement learning.
#
#  Every function obeys the house rules:
#    NO LOOK-AHEAD  — only data at indices <= the evaluation bar.
#    FAIL-CLOSED    — insufficient data returns a verdict with a written
#                     reason, never a guess.
#    DETERMINISTIC  — any randomness comes from an explicit seeded LCG.
#    HONEST LIMITS  — docstrings state what the method CANNOT claim
#                     (Granger is predictive causality, never interventional;
#                      the Lyapunov estimate is a divergence rate on one
#                      embedding, not a full spectrum).
#
#  These are ANALYTICAL instruments for the trading lab, not signals: nothing
#  here originates a trade. The regime ensemble may consult them; the vetoes
#  still apply.
# =============================================================================
"""Exotic analysis: DFA fractal, chaos metrics, Granger causality, what-if
replay, 2x2 game solving, genetic search, tabular Q-learning. Stdlib only."""

import math


class Rng:
    """Deterministic LCG (Numerical Recipes constants). Seeded, explicit."""

    def __init__(self, seed=1234):
        self.s = int(seed) & 0x7FFFFFFF

    def next(self):
        self.s = (self.s * 1103515245 + 12345) & 0x7FFFFFFF
        return self.s / float(0x7FFFFFFF)

    def gauss(self):
        # Box-Muller from two uniforms (deterministic given the seed)
        u1 = max(self.next(), 1e-12)
        u2 = self.next()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


# ---------------------------------------------------------------------------
# 7.6 FRACTAL ANALYSIS — DFA exponent (Peng et al. 1994)
# ---------------------------------------------------------------------------
def dfa_fluctuation(series, scale):
    """RMS detrended fluctuation F(scale) over an already-integrated series.

    The series must be the CUMULATIVE sum of the mean-removed signal (the
    standard DFA input). Windows are non-overlapping; the last partial window
    is dropped rather than padded (padding fabricates data).
    """
    n = len(series)
    if scale < 4 or n < 2 * scale:
        return None
    m = n // scale
    F2 = 0.0
    cnt = 0
    for w in range(m):
        seg = series[w * scale:(w + 1) * scale]
        # local linear trend (least squares)
        xm = (scale - 1) / 2.0
        ym = sum(seg) / scale
        num = sum((t - xm) * (seg[t] - ym) for t in range(scale))
        den = sum((t - xm) ** 2 for t in range(scale))
        slope = num / den if den else 0.0
        intercept = ym - slope * xm
        F2 += sum((seg[t] - (intercept + slope * t)) ** 2
                  for t in range(scale)) / scale
        cnt += 1
    if cnt == 0:
        return None
    return math.sqrt(F2 / cnt)


def fractal_dfa(returns, i, scales=(8, 16, 32), min_scale=8):
    """DFA classification of `returns` up to and including index `i`.

    -> dict(h, classification, self_similarity, n, note). H > 0.55 persistent,
    H < 0.45 anti-persistent, else random-walk-like. Self-similarity is the
    spread of per-scale exponents (small = monofractal-ish).
    """
    r = [x for x in returns[:i + 1] if x == x]         # drop NaN, keep order
    out = {"h": float("nan"), "classification": "insufficient",
           "self_similarity": float("nan"), "n": len(r), "note": ""}
    if len(r) < 4 * min_scale:
        out["note"] = "insufficient-history"
        return out
    mu = sum(r) / len(r)
    integ = [0.0]
    for x in r:
        integ.append(integ[-1] + (x - mu))
    integ = integ[1:]
    pts = []
    for s in scales:
        F = dfa_fluctuation(integ, s)
        if F is not None and F > 0:
            pts.append((math.log(s), math.log(F)))
    if len(pts) < 2:
        out["note"] = "too-few-scales"
        return out
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    n = len(pts)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    h = (sum((xs[j] - mx) * (ys[j] - my) for j in range(n)) / den
         if den else float("nan"))
    out["h"] = h
    if h > 0.55:
        out["classification"] = "persistent"
    elif h < 0.45:
        out["classification"] = "anti-persistent"
    else:
        out["classification"] = "random-walk"
    out["self_similarity"] = max(ys) - min(ys) if len(ys) > 1 else 0.0
    return out


# ---------------------------------------------------------------------------
# 7.7 CHAOS THEORY — OU half-life, divergence rate, false neighbours
# ---------------------------------------------------------------------------
def ou_half_life(log_prices, i, min_n=30):
    """OU/AR(1) half-life of mean reversion in log price (causal: <= i).

    x_t = c + phi*x_{t-1} by OLS on levels (no differencing — the level
    fit IS the mean-reversion test). half_life = -ln2 / ln(phi) only when
    0 < phi < 1; phi >= 1 is a random walk / explosive -> no half-life.
    """
    x = [v for v in log_prices[:i + 1] if v == v]
    out = {"half_life": float("nan"), "phi": float("nan"), "note": ""}
    if len(x) < min_n + 1:
        out["note"] = "insufficient-history"
        return out
    y = x[1:]
    xp = x[:-1]
    n = len(y)
    mx, my = sum(xp) / n, sum(y) / n
    den = sum((v - mx) ** 2 for v in xp)
    if den <= 0:
        out["note"] = "degenerate-predictor"
        return out
    phi = sum((xp[j] - mx) * (y[j] - my) for j in range(n)) / den
    out["phi"] = phi
    # UNIT-ROOT MARGIN (T091): a finite sample of a random walk routinely
    # estimates phi just under 1 (measured 0.9835 -> "half-life 41 bars" on
    # the test walk) — that is not mean reversion, it is a unit root with
    # noise. phi >= 0.95 refuses it; a real AR(1) at 0.9 (half-life 6.6) is
    # far below the bar. Fail-closed: the estimate is reported, the
    # half-life is not.
    if 0.95 <= phi < 1.0:
        out["note"] = "unit-root neighbourhood (phi >= 0.95)"
        return out
    if 0.0 < phi < 1.0:
        out["half_life"] = -math.log(2.0) / math.log(phi)
    else:
        out["note"] = "non-stationary (phi outside (0,1))"
    return out


def divergence_rate(series, i, emb=4, eps=1e-6, seed=7):
    """Largest-Lyapunov-STYLE divergence rate (Rosenstein estimator).

    T093 REWRITE #2. The T091 version averaged `log(dt/d0)/t` over every
    (pair, step) — that over-weights each pair by its horizon and, worse,
    mixes steps of very different size, so a stationary AR(1) measured
    POSITIVE (+0.092) and a random walk measured +0.091: both wrong. The
    standard Rosenstein fix is a two-stage estimator: for each neighbour
    pair accumulate the MEAN log-distance over forward steps, bucketed by
    step t; then regress mean ln d(t) on t THROUGH THE ORIGIN — the slope
    IS the estimate.

        stationary AR(1):  ln d(t) ~ ln d0 + t*ln(phi)   -> slope = ln(phi) < 0
        random walk:       E ln d(t) ~ 0.5*ln(t) + c     -> slope ~ 0.5/tbar > 0
                            but small (0.01 at tbar=50) — neither, as documented

    NEGATIVE = convergent, POSITIVE = divergent, ~0 = neither.
    HONEST LIMIT (T093 measured, not assumed): this estimator is calibrated
    on DETERMINISTIC series — logistic map r=4.0/3.8 reads +0.05/+0.06
    (chaotic), r=3.2 reads -0.90 (2-cycle), r=2.8 reads -0.22 (fixed
    point), r=2.5 reads -0.001. On a purely STOCHASTIC series (AR(1),
    random walk) nearest neighbours start out BELOW the innovation floor
    and noise alone pulls them apart, so the estimate reads small-positive
    (+0.03/+0.04): noise masquerades as weak divergence. Therefore on
    noisy data the SIGN is not trustworthy without an embedding test
    (false_neighbors) first — and one embedding, neighbour pairs only,
    is a divergence RATE estimate, not a Lyapunov spectrum. Causal:
    uses series[:i+1] only.
    """
    out = {"lyap": float("nan"), "note": ""}
    s = [v for v in series[:i + 1] if v == v]
    n_pts = len(s) - (emb - 1)
    if n_pts < 12:
        out["note"] = "insufficient-history"
        return out
    theiler = 2 * emb
    pts = [tuple(s[t + k] for k in range(emb)) for t in range(n_pts)]

    def dist(a, b):
        return math.sqrt(sum((a[k] - b[k]) ** 2 for k in range(emb)))

    horizon = max(2, n_pts // 4)            # follow pairs this far forward
    sums = {}                               # t -> sum of ln(dt/d0)
    cnts = {}                               # t -> count
    for a in range(n_pts):
        best, bestd = None, None
        for b in range(n_pts):
            if abs(a - b) <= theiler:
                continue
            d0 = dist(pts[a], pts[b])
            if d0 <= 0.0:
                continue
            if bestd is None or d0 < bestd:
                best, bestd = b, d0
        if best is None or bestd is None:
            continue
        max_t = min(horizon, n_pts - 1 - a, n_pts - 1 - best)
        for t in range(1, max_t + 1):
            dt = dist(pts[a + t], pts[best + t])
            if dt <= 0.0:
                continue
            sums[t] = sums.get(t, 0.0) + math.log(dt / bestd)
            cnts[t] = cnts.get(t, 0) + 1
    if not sums:
        out["note"] = "no-usable-neighbour-pairs"
        return out
    # through-the-origin slope of mean ln d(t) vs t (Rosenstein 1993)
    num = den = 0.0
    for t, ssum in sums.items():
        y = ssum / cnts[t]
        num += t * y
        den += t * t
    if den <= 0:
        out["note"] = "no-usable-neighbour-pairs"
        return out
    out["lyap"] = num / den
    return out


def false_neighbors(series, i, max_dim=3, threshold=10.0, lag=1):
    """False-nearest-neighbours count per embedding dim (causal <= i).

    Smallest dim where the FNN fraction drops below 0.1 is a defensible
    embedding; if none does, the honest answer is 'insufficient'.
    """
    out = {"fnn": {}, "best_dim": 0, "note": ""}
    s = [v for v in series[:i + 1] if v == v]
    out_len = len(s) - (max_dim + 1) * lag
    if out_len < 8:
        out["note"] = "insufficient-history"
        return out
    for m in range(1, max_dim + 1):
        vecs = [tuple(s[t + k * lag] for k in range(m))
                for t in range(out_len)]
        vecs1 = [tuple(s[t + k * lag] for k in range(m + 1))
                 for t in range(out_len - lag)]
        n = min(len(vecs), len(vecs1))
        fnn = 0
        considered = 0
        for a in range(n):
            best = None
            bestd = None
            for b in range(n):
                if a == b:
                    continue
                d = math.sqrt(sum((vecs[a][k] - vecs[b][k]) ** 2
                                  for k in range(m)))
                if best is None or d < best:
                    best = d
                    bestd = b
            if best is None or best == 0.0 or bestd is None:
                continue
            considered += 1
            d1 = math.sqrt(sum((vecs1[a][k] - vecs1[bestd][k]) ** 2
                               for k in range(m + 1)))
            if d1 / best > threshold:
                fnn += 1
        frac = (fnn / float(considered)) if considered else 1.0
        out["fnn"][m] = frac
        if frac < 0.1 and out["best_dim"] == 0:
            out["best_dim"] = m
    if out["best_dim"] == 0:
        out["note"] = "no-dim-clears-0.1"
    return out


# ---------------------------------------------------------------------------
# 7.1 CAUSAL REASONING — Granger predictive causality
# ---------------------------------------------------------------------------
def _betacf(a, b, x, itmax=200, eps=3e-12):
    """Continued fraction for the incomplete beta (Numerical Recipes)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betai(a, b, x):
    """Regularized incomplete beta I_x(a, b) — the F-test's p-value core."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log(1.0 - x))
    front = math.exp(lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + b * math.log(1.0 - x) + a * math.log(x)) * _betacf(b, a, 1.0 - x) / b


def f_cdf_p(f, d1, d2):
    """P(F_{d1,d2} > f) via the regularized incomplete beta."""
    if f <= 0:
        return 1.0
    return betai(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * f))


def _ar_ssr(y, lags, exog=None):
    """SSR of an OLS AR model with optional extra exogenous lag column."""
    n = len(y)
    start = lags
    rows = n - start
    if rows <= lags + (1 if exog else 0) + 1:
        return None, 0
    k = lags + (1 if exog else 0) + 1   # +1 for the intercept
    # design matrix
    X = []
    Y = []
    for t in range(start, n):
        row = [1.0]
        for L in range(1, lags + 1):
            row.append(y[t - L])
        if exog is not None:
            row.append(exog[t - 1])
        X.append(row)
        Y.append(y[t])
    rows = len(Y)
    if rows <= k:
        return None, 0
    # normal equations via Gaussian elimination (k is tiny: <= 5)
    XtX = [[sum(X[r][a] * X[r][b] for r in range(rows))
            for b in range(k)] for a in range(k)]
    Xty = [sum(X[r][a] * Y[r] for r in range(rows)) for a in range(k)]
    # ridge-free elimination with partial pivoting
    M = [row[:] + [Xty[a]] for a, row in enumerate(XtX)]
    for col in range(k):
        piv = max(range(col, k), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-30:
            return None, 0
        M[col], M[piv] = M[piv], M[col]
        for r in range(k):
            if r == col:
                continue
            f = M[r][col] / M[col][col]
            for cc in range(col, k + 1):
                M[r][cc] -= f * M[col][cc]
    beta = [M[a][k] / M[a][a] for a in range(k)]
    ssr = 0.0
    for r in range(rows):
        pred = sum(beta[a] * X[r][a] for a in range(k))
        ssr += (Y[r] - pred) ** 2
    return ssr, rows


def granger_causality(x, y, i, lags=2, alpha=0.05):
    """Does `x` Granger-cause `y` at bar `i`? (predictive, NOT interventional).

    Restricted AR(y) vs unrestricted AR(y)+lags(x); classic F test.
    HONEST LIMIT: Granger causality means "x's past improves prediction of
    y beyond y's own past" — it is NOT do-calculus, NOT interventional
    causation, and says nothing about a manipulable mechanism.
    """
    out = {"verdict": "insufficient", "f": float("nan"),
           "p": float("nan"), "note": "predictive causality only"}
    xs = [v for v in x[:i + 1] if v == v]
    ys = [v for v in y[:i + 1] if v == v]
    if len(xs) != len(ys) or len(ys) < 4 * lags + 10:
        out["note"] = "insufficient-history (predictive causality only)"
        return out
    ssr_r, n = _ar_ssr(ys, lags)
    ssr_u, n2 = _ar_ssr(ys, lags, exog=xs)
    if ssr_r is None or ssr_u is None or n2 != n or ssr_u <= 0:
        out["note"] = "degenerate-fit (predictive causality only)"
        return out
    d1 = lags
    d2 = n - 2 * lags - 1
    if d2 <= 0:
        out["note"] = "insufficient-dof (predictive causality only)"
        return out
    fstat = ((ssr_r - ssr_u) / d1) / (ssr_u / d2)
    fstat = max(fstat, 0.0)
    p = f_cdf_p(fstat, d1, d2)
    out["f"], out["p"] = fstat, p
    out["verdict"] = "granger-causes" if p < alpha else "no-causality"
    return out


# ---------------------------------------------------------------------------
# 7.2 COUNTERFACTUAL SIMULATION — the what-if replay
# ---------------------------------------------------------------------------
def whatif(bars, i, direction, horizon=5, cost_bps=10.0):
    """Replay `direction` from bar i+1 onward: pnl, MAE, flat baseline.

    Entry at bar i+1's OPEN (the next tradable price — no look-ahead), exit
    at min(i+horizon, last) close. Both legs are named: the trade AND the
    counterfactual "you had done nothing" (flat = 0 by construction).
    """
    out = {"pnl": float("nan"), "mae": float("nan"), "bars": 0,
           "flat_pnl": 0.0, "verdict": "insufficient", "note": ""}
    d = 1 if direction >= 0 else -1
    if i + 1 >= len(bars):
        out["note"] = "no-next-bar"
        return out

    def px(b):
        if isinstance(b, dict):
            return float(b["open"]), float(b["high"]), float(b["low"]), \
                float(b["close"])
        return float(b[0]), float(b[1]), float(b[2]), float(b[3])

    try:
        entry = px(bars[i + 1])[0]
        # STRICT HORIZON (T091): the exit is bar i+horizon, or NOTHING.
        # The old `min(i+horizon, len(bars)-1)` clipped to whatever the last
        # bar of the PASSED ARRAY was — appending bars moved the exit, i.e.
        # later data changed the verdict at bar i. That is look-ahead. A
        # horizon the data cannot honour is an honest refusal instead.
        end = i + horizon
        if end >= len(bars):
            out["note"] = "horizon-unavailable"
            return out
        exitp = px(bars[end])[3]
    except (KeyError, TypeError, ValueError, IndexError):
        out["note"] = "bad-bars"
        return out
    if entry <= 0:
        out["note"] = "bad-entry-price"
        return out
    gross = d * (exitp - entry) / entry
    cost = 2.0 * cost_bps / 10000.0            # entry + exit
    out["pnl"] = gross - cost
    out["bars"] = end - i
    worst = 0.0
    for t in range(i + 1, end + 1):
        try:
            _o, h, l, _c = px(bars[t])
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        adverse = d * (l - entry) / entry if d > 0 else d * (h - entry) / entry
        worst = min(worst, adverse)
    out["mae"] = worst
    out["verdict"] = ("acted: pnl %+.4f vs flat 0.0000 (mae %+.4f, %d bars)"
                      % (out["pnl"], out["mae"], out["bars"]))
    return out


# ---------------------------------------------------------------------------
# 7.3 GAME THEORY — 2x2 solving + opponent model
# ---------------------------------------------------------------------------
def solve_2x2(a, b, c, d, zero_sum=False):
    """Normal form [[a,b],[c,d]] (row player vs column player).

    Returns pure NE (list of (r,c)), the mixed NE probabilities (row plays
    top with p, column plays left with q) or None, and the dominated
    strategy if one exists. Closed form, fully deterministic.

    T092 FIX — the representation matters. A single 4-tuple is ambiguous:
    for a BIMATRIX both players' payoffs must be given, but for zero-sum
    games (matching pennies, coin flips, microstructure standoffs) the
    column player's payoff is the NEGATION of the row player's and one
    4-tuple suffices. Zero-sum CANNOT be auto-detected from one tuple
    ((1,-1,-1,1) satisfies a+d == 2 and b+c == -2; (3,0,5,1) has 4 vs 5 —
    no predicate on 4 numbers distinguishes them), so it is EXPLICIT:
    zero_sum=True means payoffs are the ROW player's and the column
    player minimises them; the old code ran bimatrix logic on zero-sum
    tuples, so matching pennies "had" a pure NE at (0,0). Mixed NE is
    representation-independent (indifference conditions) and unchanged.
    """
    out = {"pure_ne": [], "mixed": None, "dominated": "", "note": ""}
    if zero_sum:
        out["note"] = "zero-sum: payoffs are row player's; column is negated"
        # (i,j) is a NE iff row maximises column j AND column minimises
        # row i (its payoff is -M[i][j]):
        #   (0,0): a>=c and a<=b   (0,1): b>=d and b<=a
        #   (1,0): c>=a and c<=d   (1,1): d>=b and d<=c
        if a >= c and a <= b:
            out["pure_ne"].append((0, 0))
        if b >= d and b <= a:
            out["pure_ne"].append((0, 1))
        if c >= a and c <= d:
            out["pure_ne"].append((1, 0))
        if d >= b and d <= c:
            out["pure_ne"].append((1, 1))
    else:
        # pure NE in a bimatrix: no unilateral deviation improves
        if a >= c and a >= b:
            out["pure_ne"].append((0, 0))
        if b >= a and b >= d:
            out["pure_ne"].append((0, 1))
        if c >= a and c >= d:
            out["pure_ne"].append((1, 0))
        if d >= b and d >= c:
            out["pure_ne"].append((1, 1))
    # dominance — ROW player's own payoffs (a,c vs b,d), any representation
    if a > c and b > d:
        out["dominated"] = "row-top"           # row: top dominates bottom
    elif c > a and d > b:
        out["dominated"] = "row-bottom"
    # mixed NE (exists when both players are indifferent on the mix)
    denom = (a - b) + (d - c)
    denom2 = (a - c) + (d - b)
    p = (d - c) / denom if abs(denom) > 1e-12 else None
    q = (d - b) / denom2 if abs(denom2) > 1e-12 else None
    if p is not None and q is not None and 0.0 < p < 1.0 and 0.0 < q < 1.0:
        out["mixed"] = (p, q)
    return out


def opponent_model(returns, i, window=20):
    """Rolling hit-rate + aggression of a rival series -> best response.

    `returns` are the OPPONENT's signed trades (nonzero = a trade). Hit rate
    = share of positive returns; aggression = mean |size| vs own-scale. The
    best response is fade (consistently wrong), trend (consistently right),
    neutral (no evidence — fail closed).
    """
    out = {"class": "neutral", "hit_rate": float("nan"),
           "aggression": float("nan"), "n": 0, "note": ""}
    r = [v for v in returns[max(0, i - window + 1):i + 1]
         if v == v and v != 0.0]
    out["n"] = len(r)
    if len(r) < 5:
        out["note"] = "insufficient-history"
        return out
    hits = sum(1 for v in r if v > 0) / float(len(r))
    agg = sum(abs(v) for v in r) / len(r)
    out["hit_rate"], out["aggression"] = hits, agg
    if hits <= 0.3:
        out["class"] = "fade"
    elif hits >= 0.7:
        out["class"] = "trend"
    else:
        out["class"] = "neutral"
        out["note"] = "no clear edge either way"
    return out


# ---------------------------------------------------------------------------
# 7.4 EVOLUTIONARY ALGORITHMS — seeded genetic search
# ---------------------------------------------------------------------------
def genetic_search(bounds, fitness, pop=40, gens=100, seed=42,
                  tournament=3, mutate_sigma=0.1, elite=2):
    """Maximise `fitness(vector)` over `bounds` [(lo,hi),...].

    Tournament selection, uniform crossover, gaussian mutation (clamped to
    bounds — a genome NEVER leaves its declared box), elitism. Deterministic
    for a given seed: two runs are bit-identical.
    """
    rng = Rng(seed)
    n = len(bounds)
    if n == 0 or pop < elite + 2:
        return {"best": [], "best_fitness": float("nan"),
                "note": "bad-config"}
    def clamp(v, j):
        lo, hi = bounds[j]
        return min(max(v, lo), hi)
    population = [[clamp(lo + rng.next() * (hi - lo), j)
                   for j, (lo, hi) in enumerate(bounds)]
                  for _ in range(pop)]
    best = None
    best_f = float("-inf")
    for _ in range(gens):
        scored = sorted(((fitness(ind), ind) for ind in population),
                        key=lambda t: -t[0])
        if scored[0][0] > best_f:
            best_f, best = scored[0][0], list(scored[0][1])
        nxt = [list(scored[j][1]) for j in range(elite)]   # elitism
        while len(nxt) < pop:
            # tournament selection — T092 FIX: `scored` is sorted
            # BEST-FIRST, so `min` picked the WORST tournament entrant
            # (the old code evolved away from fitness and stalled at
            # -0.43 on sphere-3D). Winners are the MAX fitness.
            cands = [scored[int(rng.next() * len(scored))]
                     for _ in range(tournament)]
            p1 = max(cands, key=lambda t: t[0])[1]
            cands = [scored[int(rng.next() * len(scored))]
                     for _ in range(tournament)]
            p2 = max(cands, key=lambda t: t[0])[1]
            child = [p1[j] if rng.next() < 0.5 else p2[j] for j in range(n)]
            child = [clamp(child[j] + mutate_sigma * (bounds[j][1] - bounds[j][0])
                           * rng.gauss(), j) for j in range(n)]
            nxt.append(child)
        population = nxt
    return {"best": best, "best_fitness": best_f, "note": ""}


# ---------------------------------------------------------------------------
# 7.5 REINFORCEMENT LEARNING — tabular Q-learning on causal market state
# ---------------------------------------------------------------------------
def market_state(bars, t):
    """Discrete causal state at bar t from bars <= t: trend x vol."""
    if t < 21:
        return None

    def cl(b):
        return b["close"] if isinstance(b, dict) else b[3]

    def hi(b):
        return b["high"] if isinstance(b, dict) else b[1]

    def lo(b):
        return b["low"] if isinstance(b, dict) else b[2]

    c = [cl(b) for b in bars[:t + 1]]
    ma_fast = sum(c[-10:]) / 10.0
    ma_slow = sum(c[-21:]) / 21.0
    trend = 1 if ma_fast > ma_slow else (0 if ma_fast < ma_slow else 2)
    rng21 = (max(hi(b) for b in bars[t - 20:t + 1])
             - min(lo(b) for b in bars[t - 20:t + 1]))
    atr_proxy = sum(abs(c[j] - c[j - 1]) for j in range(t - 9, t + 1)) / 10.0
    vol = 1 if atr_proxy > 0.5 * rng21 / 21.0 * 3 else 0
    return (trend, vol)


def q_learning(bars, episodes=50, seed=1234, gamma=0.9, alpha=0.2,
               eps=0.15, cost_bps=5.0):
    """Tabular Q-learning: {long, flat, short} on causal (trend, vol) state.

    Reward at t = position * next-bar return - turnover cost. The state at t
    uses ONLY bars <= t (market_state is causal by construction). Returns
    the Q-table and the learned policy; the TESTS measure it against the
    always-flat and always-long baselines on the SAME series.
    """
    rng = Rng(seed)
    actions = (1, 0, -1)
    Q = {}
    closes = []
    for b in bars:
        closes.append(b["close"] if isinstance(b, dict) else b[3])
    n = len(closes)
    for _ep in range(episodes):
        pos = 0
        for t in range(21, n - 1):
            st = market_state(bars, t)
            if st is None:
                continue
            if st not in Q:
                Q[st] = {a: 0.0 for a in actions}
            a = (actions[int(rng.next() * 3)]
                 if rng.next() < eps
                 else max(actions, key=lambda act: Q[st][act]))
            r = closes[t + 1] - closes[t]
            reward = (a * r / closes[t]
                      - (abs(a - pos) * cost_bps / 10000.0))
            nxt = market_state(bars, t + 1)
            qn = (max(Q.setdefault(nxt, {x: 0.0 for x in actions}).values())
                  if nxt is not None else 0.0)
            Q[st][a] += alpha * (reward + gamma * qn - Q[st][a])
            pos = a
    policy = {st: max(actions, key=lambda act: Q[st][act]) for st in Q}
    return {"Q": Q, "policy": policy, "note": ""}


def q_learning_pnl(bars, policy, cost_bps=5.0):
    """Realised pnl of a (trend,vol)->action policy on `bars` (causal)."""
    closes = [b["close"] if isinstance(b, dict) else b[3] for b in bars]
    pnl = 0.0
    pos = 0
    for t in range(21, len(closes) - 1):
        st = market_state(bars, t)
        a = policy.get(st, 0)
        pnl += a * (closes[t + 1] - closes[t]) / closes[t]
        pnl -= abs(a - pos) * cost_bps / 10000.0
        pos = a
    return pnl
