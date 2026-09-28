#!/usr/bin/env python3
"""OmniSeed — uncertainty audit (the oracle behind docs/UNCERTAINTY.md).

Answers the two questions the uncertainty SPLIT claims to answer, using only
the committed held-out artefacts (models/heads/*.bin + the h[E] fixtures), and
prints the tables the documentation cites.

  ALEATORIC  Does the calibrated normalised entropy track the observed error
             rate?  (error rate per entropy quartile)
  EPISTEMIC  Does the h[E] distance from the reference mean track the error
             rate WITHIN a domain?  And does it fire ACROSS domains?

The expected, and measured, answers are:

  * aleatoric is monotone in the error rate for 4 of the 5 fitted heads. The
    exception is the DecisionAction head, whose accuracy is 0.244 — near
    chance, so nothing predicts its errors and the honest report is NO SIGNAL.
  * epistemic has NO within-domain signal, which is the CORRECT result: inside
    one domain there is no epistemic uncertainty to find. It separates domains
    completely.

Run from the repository root:
    python tools/uncertainty_audit.py

Stdlib only. Reads; never writes. Exit code is always 0 — this is a report,
not a gate; the gate is tests/test_uncertainty_split.cpp.
"""
import math
import os
import struct
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(ROOT, "tests", "fixtures", "head_calibration")
E_DEFAULT = 768

# (tag, bin, fixture dir, column, reader)
HEADS = [
    ("trading / DecisionAction", "models/heads/trading_head.bin",
     "trading", "DecisionAction", "decision"),
    ("trading / trading.regime", "models/heads/trading_regime_head.bin",
     "trading", "trading.regime", "classification"),
    ("language / language.intent", "models/heads/language_head.bin",
     "language_intent", "language.intent", "classification"),
    ("language / language.language", "models/heads/language_head.bin",
     "language_language", "language.language", "classification"),
    ("language / language.sentiment", "models/heads/language_head.bin",
     "language_sentiment", "language.sentiment", "classification"),
]


# ---------------------------------------------------------------------------
# blob readers
# ---------------------------------------------------------------------------
def _rd(f, n):
    b = f.read(n)
    if len(b) != n:
        raise ValueError("short read")
    return b


def _ri(f):
    return struct.unpack("<i", _rd(f, 4))[0]


def _rf(f):
    return struct.unpack("<f", _rd(f, 4))[0]


def _rs(f):
    n = _ri(f)
    return _rd(f, n).decode("utf-8") if n else ""


def load_classification(path):
    """ClassificationHead v1: magic OMNISCH1."""
    with open(path, "rb") as f:
        if _rd(f, 8) != b"OMNISCH1":
            raise ValueError("not a ClassificationHead blob")
        _ver, E, n_sets = _ri(f), _ri(f), _ri(f)
        sets = []
        for _ in range(n_sets):
            name = _rs(f)
            sets.append((name, [_rs(f) for _ in range(_ri(f))]))
        total = _ri(f)
        _ri(f)  # fitted_rows
        _deft = _rf(f)
        temps = [_rf(f) for _ in range(n_sets)]
        cals = [_ri(f) for _ in range(n_sets)]
        eces = [_rf(f) for _ in range(n_sets)]
        proj = struct.unpack("<%df" % (total * E), _rd(f, total * E * 4))
        bias = struct.unpack("<%df" % total, _rd(f, total * 4))
    return dict(E=E, sets=sets, temps=temps, cals=cals, eces=eces, proj=proj, bias=bias)


def load_decision(path):
    """DecisionHead v3: magic OMNISDH1."""
    with open(path, "rb") as f:
        if _rd(f, 8) != b"OMNISDH1":
            raise ValueError("not a DecisionHead blob")
        _ver, E, A = _ri(f), _ri(f), _ri(f)
        _fitted = _ri(f)
        names = [_rs(f) for _ in range(A)]
        for _ in range(A):
            _rs(f)                      # asset label
        for _ in range(A):
            _rf(f)                      # invalidation level
        proj = struct.unpack("<%df" % (A * E), _rd(f, A * E * 4))
        bias = struct.unpack("<%df" % A, _rd(f, A * 4))
        temp = _rf(f)                   # v3 trailer
        cal_n = _ri(f)
        cal_ece = _rf(f)
    return dict(E=E, A=A, names=names, proj=proj, bias=bias,
                temp=temp, cal_n=cal_n, cal_ece=cal_ece)


def load_hidden(path, E):
    raw = open(path, "rb").read()
    n = len(raw) // (4 * E)
    return n, struct.unpack("<%df" % (n * E), raw[: n * E * 4])


def load_labels(path, col):
    lines = open(path).read().strip().split("\n")
    ci = lines[0].split("\t").index(col)
    return [l.split("\t")[ci] for l in lines[1:]]


# ---------------------------------------------------------------------------
# maths
# ---------------------------------------------------------------------------
def softmax(xs, T):
    m = max(xs)
    ex = [math.exp((x - m) / T) for x in xs]
    s = sum(ex)
    return [e / s for e in ex]


def entropy(p):
    return -sum(pi * math.log(pi) for pi in p if pi > 0)


def reference(rows, E):
    """mean + per-dim sigma, with the same 10%-of-typical floor the C++ uses."""
    n = len(rows)
    mu = [sum(r[e] for r in rows) / n for e in range(E)]
    sd = [math.sqrt(sum((r[e] - mu[e]) ** 2 for r in rows) / n) for e in range(E)]
    floor = max(1e-6, 0.1 * (sum(sd) / E))
    sd = [s if s > 1e-6 else floor for s in sd]
    return mu, sd


def distance(h, mu, sd, E):
    return math.sqrt(sum(((h[e] - mu[e]) / sd[e]) ** 2 for e in range(E)) / E)


def quartile_errors(recs, key):
    """Error rate in four equal bins ordered by `key` (ascending)."""
    n = len(recs)
    order = sorted(range(n), key=lambda i: recs[i][key])
    q = max(1, n // 4)
    out = []
    for b in range(4):
        idx = order[b * q:(b + 1) * q] if b < 3 else order[3 * q:]
        if not idx:
            continue
        err = sum(0 if recs[i]["ok"] else 1 for i in idx) / len(idx)
        out.append((recs[idx[0]][key], recs[idx[-1]][key], len(idx), err))
    return out


def is_monotone(table):
    errs = [row[3] for row in table]
    return all(errs[i] <= errs[i + 1] + 1e-9 for i in range(len(errs) - 1))


# ---------------------------------------------------------------------------
# per-head audit
# ---------------------------------------------------------------------------
def audit_head(tag, binrel, fixdir, col, kind):
    binpath = os.path.join(ROOT, binrel)
    hdir = os.path.join(FIX, fixdir)
    if not (os.path.exists(binpath) and os.path.exists(hdir)):
        print("SKIP  %s (artefact missing)" % tag)
        return None

    h = load_decision(binpath) if kind == "decision" else load_classification(binpath)
    E = h["E"]
    if kind == "decision":
        labels, K, T, off = h["names"], h["A"], h["temp"], 0
    else:
        off, tgt = 0, None
        for i, (name, labels_i) in enumerate(h["sets"]):
            if name == col:
                tgt = (i, off, labels_i)
            off += len(labels_i)
        if tgt is None:
            print("SKIP  %s (set %s absent)" % (tag, col))
            return None
        si, off, labels = tgt
        K, T = len(labels), h["temps"][si]

    n, hx = load_hidden(os.path.join(hdir, "hidden.f32"), E)
    ys = load_labels(os.path.join(hdir, "labels.tsv"), col)
    rows = [hx[i * E:(i + 1) * E] for i in range(n)]

    recs = []
    for i in range(n):
        lg = [sum(h["proj"][(off + k) * E + e] * rows[i][e] for e in range(E))
              + h["bias"][off + k] for k in range(K)]
        p = softmax(lg, T)
        pred = max(range(K), key=lambda k: p[k])
        recs.append(dict(ok=(labels[pred] == ys[i]),
                         H=entropy(p) / math.log(K) if K > 1 else 0.0))

    mu, sd = reference(rows, E)
    for i in range(n):
        recs[i]["d"] = distance(rows[i], mu, sd, E)

    acc = sum(1 for r in recs if r["ok"]) / n
    print("=" * 76)
    print("%s   K=%d n=%d T=%.3f acc=%.4f" % (tag, K, n, T, acc))
    for label, key in (("ALEATORIC (calibrated entropy)", "H"),
                       ("EPISTEMIC (h-distance)", "d")):
        table = quartile_errors(recs, key)
        verdict = "MONOTONE" if is_monotone(table) else "no signal"
        print("  %-32s -> error rate  [%s]" % (label, verdict))
        for lo, hi, cnt, err in table:
            print("      %-8s %.4f..%.4f  n=%3d  err=%.3f" % (key, lo, hi, cnt, err))
    return dict(mu=mu, sd=sd, E=E, n=n)


def cross_domain(a_tag, a_dir, b_tag, b_dir):
    E = E_DEFAULT
    da, db = os.path.join(FIX, a_dir), os.path.join(FIX, b_dir)
    if not (os.path.exists(da) and os.path.exists(db)):
        return
    na, ha = load_hidden(os.path.join(da, "hidden.f32"), E)
    nb, hb = load_hidden(os.path.join(db, "hidden.f32"), E)
    ra = [ha[i * E:(i + 1) * E] for i in range(na)]
    rb = [hb[i * E:(i + 1) * E] for i in range(nb)]
    mu, sd = reference(ra, E)
    va = sorted(distance(r, mu, sd, E) for r in ra)
    vb = [distance(r, mu, sd, E) for r in rb]
    p90 = va[int(0.9 * len(va))]
    below = sum(1 for x in vb if x < p90)
    print("=" * 76)
    print("CROSS-DOMAIN: reference fitted on %s, scored on %s" % (a_tag, b_tag))
    print("  in-domain  d: mean=%.4f  p90=%.4f  max=%.4f" % (sum(va) / len(va), p90, va[-1]))
    print("  out-domain d: mean=%.4f  min=%.4f  max=%.4f" % (sum(vb) / len(vb), min(vb), max(vb)))
    print("  out-domain vectors BELOW the in-domain p90: %d/%d" % (below, len(vb)))


def main():
    print("OmniSeed uncertainty audit — aleatoric vs epistemic")
    print("repository root: %s" % ROOT)
    for tag, binrel, fixdir, col, kind in HEADS:
        try:
            audit_head(tag, binrel, fixdir, col, kind)
        except Exception as exc:                       # noqa: BLE001 - report, never crash
            print("FAILED %s: %r" % (tag, exc))
    cross_domain("trading", "trading", "language.intent", "language_intent")
    cross_domain("language.intent", "language_intent", "trading", "trading")
    print("=" * 76)
    print("Read this as: aleatoric predicts error WITHIN a domain; epistemic fires")
    print("BETWEEN domains and deliberately does not fire within one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
