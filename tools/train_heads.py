#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — tools/train_heads.py
#
#  The OFFLINE half of head training. Fits the head projections and the softmax
#  temperature from the (h[E], label) pairs that tools/dump_hidden.cpp collected
#  from the real backbone, and writes the .bin blobs the runtime loads.
#
#  Nothing here runs at inference time. The runtime never sees numpy, never
#  fits, and never reads a label: it loads a .bin and does one matvec.
#
#  WHY PYTHON IS THE ORACLE AND C++ IS THE RUNTIME.
#  The port pattern used everywhere else in this tree. The fit is the part that
#  changes most often and benefits most from numpy; the runtime is the part
#  that must stay small and dependency-free. Keeping them separate also means
#  the .bin format is the ONLY contract between them, and a format is something
#  two independent readers can check.
#
#  WHAT IS BEING MEASURED — read this before quoting a number.
#  The projection is fitted to predict the LABEL THAT WAS SUPPLIED. For the
#  language sets the label is a human judgement about the utterance; for the
#  trading sets the label is a RULE (the regime engine's output, and the action
#  teacher defined in this file). So the accuracy reported below answers
#
#      "how well does h[E] linearly encode the teacher's label?"
#
#  and NOT "how much money would this make". A high accuracy on the trading
#  head means the head imitates the teacher well, nothing more. This script
#  writes that caveat into the report so a later reader cannot miss it.
#
#  usage:
#    python tools/train_heads.py \
#        --language build/head_data/language \
#        --trading  build/head_data/trading  \
#        --bars     models/market/AAPL_1d.csv \
#        --out-dir  models/heads
# =============================================================================
import argparse
import csv
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


# ---------------------------------------------------------------------------
# Canonical vocabularies, parsed from the C++ that owns them.
#
# Hard-coding them here would create a second source of truth that drifts the
# first time a label is added in C++. The .bin carries the label names and the
# C++ load() refuses a mismatch, so a drift would be caught — but only at
# runtime, on the user's machine. Parsing the source catches it now.
# ---------------------------------------------------------------------------
def parse_default_label_sets(path):
    src = open(path, encoding="utf-8").read()
    m = re.search(r"default_domain_label_sets\(\)\s*\{(.*?)\n\}", src, re.S)
    if not m:
        raise SystemExit("cannot locate default_domain_label_sets() in %s" % path)
    out = {}
    for mm in re.finditer(r'\{\s*"([^"]+)"\s*,\s*\{(.*?)\}\s*\}', m.group(1), re.S):
        labels = re.findall(r'"([^"]*)"', mm.group(2))
        if labels:
            out[mm.group(1)] = labels
    if not out:
        raise SystemExit("no label sets parsed from %s" % path)
    return out


def parse_decision_actions(path):
    src = open(path, encoding="utf-8").read()
    m = re.search(r"enum class DecisionAction[^{]*\{(.*?)\};", src, re.S)
    if not m:
        raise SystemExit("cannot locate DecisionAction enum in %s" % path)
    names = []
    for line in m.group(1).splitlines():
        line = line.split("//")[0].strip().rstrip(",")
        if not line:
            continue
        mm = re.match(r"^([A-Z_][A-Z0-9_]*)\s*(=\s*\d+)?$", line)
        if mm:
            names.append(mm.group(1))
    names = [n for n in names if n != "COUNT"]     # COUNT is the sentinel
    if not names:
        raise SystemExit("no actions parsed from %s" % path)
    return names


# ---------------------------------------------------------------------------
# .bin writers — byte-for-byte mirrors of the C++ load() functions.
# ---------------------------------------------------------------------------
def w_i32(f, v):
    f.write(int(v).to_bytes(4, "little", signed=True))


def w_f32(f, v):
    f.write(np.float32(v).tobytes())


def w_str(f, s):
    b = s.encode("utf-8")
    w_i32(f, len(b))
    f.write(b)


def w_f32s(f, arr):
    f.write(np.asarray(arr, dtype="<f4").tobytes())


def write_decision_head(path, E, action_names, proj, bias, assets, invals,
                        temperature, calib_samples, calib_error, fitted):
    """DecisionHead .bin, format v3.

      magic 'OMNISDH1', i32 version=3, i32 E, i32 A, i32 fitted_flag,
      A x str name, A x str asset, A x f32 inval, A*E x f32 proj (row-major),
      A x f32 bias, f32 temperature, i32 calib_samples, f32 calib_error
    """
    proj = np.asarray(proj, dtype=np.float64)
    assert proj.shape == (len(action_names), E), proj.shape
    with open(path, "wb") as f:
        f.write(b"OMNISDH1")
        w_i32(f, 3)
        w_i32(f, E)
        w_i32(f, len(action_names))
        w_i32(f, 1 if fitted else 0)
        for n in action_names:
            w_str(f, n)
        for a in assets:
            w_str(f, a)
        w_f32s(f, invals)
        w_f32s(f, proj.reshape(-1))
        w_f32s(f, bias)
        w_f32(f, temperature)
        w_i32(f, calib_samples)
        w_f32(f, calib_error)
    return os.path.getsize(path)


def write_classification_head(path, E, sets, proj, bias, fitted_rows, fitted,
                              default_temperature, temperatures,
                              calib_samples_list, calib_errors):
    """ClassificationHead .bin, format v1.

      magic 'OMNISCH1', i32 version=1, i32 E, i32 n_sets,
      per set: str name, i32 n_labels, n_labels x str,
      i32 total_labels, i32 fitted_rows,
      f32 default_temperature,
      n_sets x f32 temperature,
      n_sets x i32 calib_samples,
      n_sets x f32 calib_error,
      total*E x f32 proj, total x f32 bias, total x u8 fitted

    Calibration is PER SET — one scalar cannot serve label sets whose logit
    scales differ by an order of magnitude. See the C++ header for the measured
    reason.
    """
    proj = np.asarray(proj, dtype=np.float64)
    total = sum(len(s["labels"]) for s in sets)
    assert proj.shape == (total, E), proj.shape
    assert len(temperatures) == len(sets), (len(temperatures), len(sets))
    assert len(calib_samples_list) == len(sets)
    assert len(calib_errors) == len(sets)
    with open(path, "wb") as f:
        f.write(b"OMNISCH1")
        w_i32(f, 1)
        w_i32(f, E)
        w_i32(f, len(sets))
        for s in sets:
            w_str(f, s["name"])
            w_i32(f, len(s["labels"]))
            for l in s["labels"]:
                w_str(f, l)
        w_i32(f, total)
        w_i32(f, fitted_rows)
        w_f32(f, default_temperature)
        for v in temperatures:
            w_f32(f, v)
        for v in calib_samples_list:
            w_i32(f, v)
        for v in calib_errors:
            w_f32(f, v)
        w_f32s(f, proj.reshape(-1))
        w_f32s(f, bias)
        f.write(np.asarray(fitted, dtype=np.uint8).tobytes())
    return os.path.getsize(path)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_dump(dump_dir):
    meta = json.load(open(os.path.join(dump_dir, "meta.json"), encoding="utf-8"))
    E = int(meta["E"])
    h = np.fromfile(os.path.join(dump_dir, "hidden.f32"), dtype="<f4")
    if h.size % E != 0:
        raise SystemExit("hidden.f32 size %d is not a multiple of E=%d" % (h.size, E))
    H = h.reshape(-1, E).astype(np.float64)

    ids, labels = [], []
    with open(os.path.join(dump_dir, "labels.tsv"), encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            cols = line.split("\t")
            if len(cols) != len(header):
                raise SystemExit("label row has %d cols, header has %d"
                                 % (len(cols), len(header)))
            ids.append(cols[0])
            labels.append(cols[1:])
    if len(ids) != H.shape[0]:
        raise SystemExit("hidden rows %d != label rows %d" % (H.shape[0], len(ids)))
    return meta, H, ids, header[1:], labels


# ---------------------------------------------------------------------------
# Softmax regression + temperature scaling
# ---------------------------------------------------------------------------
def fit_softmax(Z, y, K, lam=1e-3, iters=600, lr=0.15):
    """Full-batch Adam on the L2-regularised multinomial log-likelihood.

    Z is [N, E] STANDARDISED, y is [N] in 0..K-1. Returns (W [K,E], b [K]).
    No closed form exists for multinomial logistic regression; with 768
    features and <=1300 rows this converges in well under a second, so a
    first-order method is the right tool rather than an approximation.
    """
    N, E = Z.shape
    W = np.zeros((K, E))
    counts = np.bincount(y, minlength=K).astype(np.float64)
    priors = np.maximum(counts, 1.0) / max(counts.sum(), 1.0)
    b = np.log(priors)

    mW = np.zeros_like(W); vW = np.zeros_like(W)
    mb = np.zeros_like(b); vb = np.zeros_like(b)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for t in range(1, iters + 1):
        logits = Z @ W.T + b
        logits -= logits.max(axis=1, keepdims=True)
        ex = np.exp(logits)
        P = ex / ex.sum(axis=1, keepdims=True)
        Y = np.zeros_like(P)
        Y[np.arange(N), y] = 1.0
        G = (P - Y) / N
        gW = G.T @ Z + lam * W
        gb = G.sum(axis=0)

        for (p, g, m, v) in ((W, gW, mW, vW), (b, gb, mb, vb)):
            m *= beta1; m += (1 - beta1) * g
            v *= beta2; v += (1 - beta2) * (g * g)
            p -= lr * (m / (1 - beta1 ** t)) / (np.sqrt(v / (1 - beta2 ** t)) + eps)

        if not np.all(np.isfinite(W)):
            raise SystemExit("softmax regression diverged (non-finite weights)")
    return W, b


def softmax(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    ex = np.exp(logits)
    return ex / ex.sum(axis=1, keepdims=True)


def nll(logits, y):
    P = softmax(logits)
    return float(-np.log(np.maximum(P[np.arange(len(y)), y], 1e-12)).mean())


def fit_temperature(logits_cal, y_cal, lo=0.05, hi=50.0, iters=200):
    """Golden-section search for the T minimising NLL on the given rows.

    A 1-D search rather than gradient descent because the objective is smooth
    and unimodal in log T, and 200 evaluations of a K-way softmax cost nothing.
    """
    a, b = np.log(lo), np.log(hi)
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc = nll(logits_cal / np.exp(c), y_cal)
    fd = nll(logits_cal / np.exp(d), y_cal)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = nll(logits_cal / np.exp(c), y_cal)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = nll(logits_cal / np.exp(d), y_cal)
    return float(np.exp((a + b) / 2.0))


def fit_temperature_cv(logits, y, n_folds=5, shuffle=True, seed=1234):
    """T by k-fold cross-validation INSIDE the holdout, plus an out-of-fold ECE.

    WHY NOT A SINGLE CALIBRATION SPLIT. The first version of this fitted T on a
    15% slice — 34-37 rows per label set. That is not enough. Measured on the
    language sets, the T the calibration slice chose was 1.5-2x SMALLER than
    the T that was optimal on the held-out rows (intent 19.4 vs 28.7, language
    3.5 vs 6.5, sentiment 2.5 vs 4.7), i.e. every set came out systematically
    UNDER-softened, and language.language's ECE got WORSE than no calibration
    at all (0.172 -> 0.194). Fitting on 4/5 of the holdout at a time, five
    times, and averaging in log space uses 5x the data for the same protocol.

    Returns (T, out_of_fold_ece). The out-of-fold ECE is the unbiased number:
    every row is scored with a T fitted without its own fold. T itself is the
    geometric mean of the folds, which is the natural average for a scale.

    `shuffle=False` keeps the folds contiguous, which is the right thing for a
    time series: adjacent bars are not exchangeable, and shuffling them would
    let a fold be scored by a T fitted on its immediate neighbours.
    """
    n = len(y)
    idx = np.arange(n)
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    folds = [f for f in np.array_split(idx, n_folds) if len(f) > 0]
    logs, oof_p, oof_y = [], [], []
    for f in range(len(folds)):
        te = folds[f]
        tr = np.concatenate([folds[g] for g in range(len(folds)) if g != f])
        if len(tr) < 2 or len(te) == 0:
            continue
        Tf = fit_temperature(logits[tr], y[tr])
        logs.append(np.log(Tf))
        oof_p.append(softmax(logits[te] / Tf))
        oof_y.append(y[te])
    if not logs:
        return 1.0, -1.0
    T = float(np.exp(float(np.mean(logs))))
    e, _ = ece(np.vstack(oof_p), np.concatenate(oof_y))
    return T, float(e)


def ece(probs, y, n_bins=10):
    """Expected Calibration Error over equal-width confidence bins."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(np.float64)
    total = max(len(y), 1)
    out, bins = 0.0, []
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        if i == 0:
            m = (conf >= lo) & (conf <= hi)
        else:
            m = (conf > lo) & (conf <= hi)
        n = int(m.sum())
        if n == 0:
            bins.append({"lo": round(lo, 2), "hi": round(hi, 2), "n": 0})
            continue
        acc = float(correct[m].mean())
        avg = float(conf[m].mean())
        out += (n / total) * abs(acc - avg)
        bins.append({"lo": round(lo, 2), "hi": round(hi, 2), "n": n,
                     "accuracy": round(acc, 4), "confidence": round(avg, 4)})
    return float(out), bins


def macro_f1(y_true, y_pred, K):
    f1s = []
    for k in range(K):
        tp = int(((y_pred == k) & (y_true == k)).sum())
        fp = int(((y_pred == k) & (y_true != k)).sum())
        fn = int(((y_pred != k) & (y_true == k)).sum())
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


# Keys fit_one() returns that are NOT JSON-serialisable. Filtering by name is
# deliberate: a new metric is included by default, so forgetting to add it here
# is not a silent omission — it is a loud TypeError in the smoke run.
_NON_JSON = ("proj", "bias", "_raw_logits", "_y", "_test_idx")


def public_metrics(r):
    return {k: v for k, v in r.items() if k not in _NON_JSON}


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------
def stratified_split(y, train_frac=0.70, seed=1234):
    """Random STRATIFIED split -> (train, holdout).

    Stratified so a rare class cannot land entirely in one part, which would
    make its temperature and its per-class accuracy meaningless.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    train, hold = [], []
    for k in np.unique(y):
        grp = idx[y == k]
        rng.shuffle(grp)
        n = len(grp)
        n_tr = int(round(n * train_frac))
        # Guarantee at least one row in the holdout for any class with >= 2.
        if n >= 2:
            n_tr = max(1, min(n_tr, n - 1))
        train.extend(grp[:n_tr])
        hold.extend(grp[n_tr:])
    return (np.array(sorted(train), dtype=int),
            np.array(sorted(hold), dtype=int))


def chronological_split(n, train_frac=0.70):
    """Time-ordered split -> (train, holdout). A random split of a time series
    leaks the future into the past and inflates every number that follows."""
    n_tr = int(round(n * train_frac))
    return np.arange(0, n_tr), np.arange(n_tr, n)


# ---------------------------------------------------------------------------
# One fitted classifier: standardise -> fit -> fold standardisation back in
# ---------------------------------------------------------------------------
def fit_one(H, y, K, tr, hold, shuffle=True, seed=1234):
    """Returns a dict with the RAW-h projection, the temperature, and metrics.

    Standardising h before the fit is a linear reparameterisation, so the
    projection is folded back into raw-h form afterwards: the runtime does one
    matvec on the unmodified h[E] and never has to remember a mean or a std.
    That matters because the head reads the SAME h the text head reads; adding
    a preprocessing step would mean the two heads no longer see one identical
    vector.

    `hold` is the entire held-out portion. The temperature is fitted by
    cross-validation inside it — see fit_temperature_cv for the measured reason
    a single small calibration slice is not enough.
    """
    if len(tr) == 0:
        raise SystemExit("empty training split — the dump is too small or a "
                         "class has too few rows to split")
    mu = H[tr].mean(axis=0)
    sd = H[tr].std(axis=0)
    sd[sd < 1e-6] = 1.0                      # a constant dim carries no signal
    Z = (H - mu) / sd

    W, b = fit_softmax(Z[tr], y[tr], K)

    # Fold standardisation back in.
    proj = W / sd[None, :]
    bias = b - (W * (mu / sd)[None, :]).sum(axis=1)

    raw = H @ proj.T + bias                  # == Z @ W.T + b, up to fp error

    def metrics(split):
        if len(split) == 0:
            return {"n": 0}
        lg = raw[split]
        yy = y[split]
        return {"n": int(len(split)), "nll": round(nll(lg, yy), 5),
                "acc": round(float((lg.argmax(axis=1) == yy).mean()), 5)}

    hold_used = hold if len(hold) else tr
    T, oof_ece = fit_temperature_cv(raw[hold_used], y[hold_used],
                                    shuffle=shuffle, seed=seed)

    probs_raw = softmax(raw[hold_used])
    probs_cal = softmax(raw[hold_used] / T)
    e_raw, _ = ece(probs_raw, y[hold_used])
    e_cal, bins_cal = ece(probs_cal, y[hold_used])

    per_class = []
    for k in range(K):
        m = y[hold_used] == k
        if int(m.sum()) == 0:
            per_class.append(None)
        else:
            per_class.append(round(float((probs_cal.argmax(axis=1)[m] == k).mean()), 4))

    return {
        "proj": proj, "bias": bias, "temperature": T,
        "train": metrics(tr), "holdout": metrics(hold_used),
        "holdout_acc": round(float((probs_cal.argmax(axis=1) == y[hold_used]).mean()), 5),
        "holdout_macro_f1": round(
            macro_f1(y[hold_used], probs_cal.argmax(axis=1), K), 4),
        "ece_raw": round(e_raw, 5),
        "ece_calibrated": round(e_cal, 5),
        # The unbiased number: every holdout row scored with a T fitted without
        # its own fold. `ece_calibrated` uses the final T on the same rows, so
        # it is very slightly optimistic — quote this one in prose.
        "ece_out_of_fold": round(oof_ece, 5),
        "ece_bins_calibrated": bins_cal,
        "per_class_recall_holdout": per_class,
        "_raw_logits": raw,
        "_y": y,
        "_test_idx": hold_used,
    }


# ---------------------------------------------------------------------------
# Trading: the action teacher
# ---------------------------------------------------------------------------
def read_closes(path):
    closes, times = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                closes.append(float(row["close"]))
                times.append(int(float(row["time"])))
            except (KeyError, ValueError):
                continue
    return np.array(closes), np.array(times)


ACTION_TEACHER_DOC = (
    "RULE-BASED TEACHER (first match wins), N=5 bars forward, "
    "threshold=1.5%: CLOSE if trend_up/trend_down and the forward return "
    "opposes the trend by >threshold; HEDGE if high_vol and |r|<=threshold; "
    "ABSTAIN if high_vol and |r|>threshold; EXPLAIN if range and |r|<=0.5%; "
    "BUY if r>+threshold; SELL if r<-threshold; HOLD otherwise. "
    "This is a DEFINITION, not realized P&L: the head learns to imitate this "
    "rule, and its accuracy says nothing about profitability."
)


def build_action_labels(ids, regime_labels, closes, n_forward=5,
                        threshold=0.015, dead=0.005):
    """Derive a DecisionAction label per dumped row.

    The bar index comes from the dumper's provenance id ("bar<i>@<ts>"), so the
    labels stay aligned with the hidden states even if a bar was skipped.
    """
    names = ["ABSTAIN", "HOLD", "BUY", "SELL", "CLOSE", "HEDGE", "EXPLAIN"]
    idx_of = {n: i for i, n in enumerate(names)}
    y, detail = [], []
    n = len(closes)
    for prov, reg in zip(ids, regime_labels):
        m = re.match(r"bar(\d+)@", prov)
        if not m:
            raise SystemExit("unexpected provenance id: %r" % prov)
        i = int(m.group(1))
        j = i + n_forward
        r = float("nan")
        if j < n and closes[i] != 0:
            r = (closes[j] - closes[i]) / closes[i]
        if not np.isfinite(r):
            y.append(idx_of["ABSTAIN"])
            detail.append("no-forward-return")
            continue
        if reg in ("trend_up", "trend_down"):
            trend_up = reg == "trend_up"
            if abs(r) > threshold and ((r > 0) != trend_up):
                y.append(idx_of["CLOSE"]); detail.append("close"); continue
        if reg == "high_vol":
            if abs(r) <= threshold:
                y.append(idx_of["HEDGE"]); detail.append("hedge")
            else:
                y.append(idx_of["ABSTAIN"]); detail.append("abstain-vol")
            continue
        if reg == "range" and abs(r) <= dead:
            y.append(idx_of["EXPLAIN"]); detail.append("explain"); continue
        if r > threshold:
            y.append(idx_of["BUY"]); detail.append("buy")
        elif r < -threshold:
            y.append(idx_of["SELL"]); detail.append("sell")
        else:
            y.append(idx_of["HOLD"]); detail.append("hold")
    return np.array(y, dtype=int), names, detail


# ---------------------------------------------------------------------------
# Held-out fixture
#
# The calibration test needs REAL held-out hidden states to check that a
# reported confidence matches an observed accuracy. Those rows live in
# build/head_data/, which CI does not have — and a test that never executes is
# not a test, it is a comment. So the trainer writes the held-out rows out as a
# committed fixture, alongside the .bin it produced them for.
#
# The fixture is the WHOLE holdout, never a prefix of it. An earlier version
# capped it at the first 200 rows, and on the chronological trading split that
# silently turned the fixture into an early-time slice whose label mix — and
# therefore its accuracy — differed from the holdout the .bin was calibrated
# for (0.86 vs 0.705). A fixture that is not the set the metrics were computed
# on cannot check those metrics.
#
# Several label sets can share one hidden.f32: the backbone produces ONE h[E]
# per row and every head reads the same vector, so the two trading heads are
# two label columns over one file rather than two copies of it.
#
# The fixture and the .bin MUST be regenerated together: the test checks the
# committed blob against the committed rows, so regenerating one without the
# other is a loud failure rather than a silent drift.
# ---------------------------------------------------------------------------
def emit_fixture(root, name, H, idx, columns, meta, metrics, vocabs):
    """Write one held-out fixture directory.

    columns  list of (label_set_name, string_labels_array)
    metrics  dict: label_set_name -> dict(temperature, ece_calibrated,
                                          holdout_acc)
    vocabs   dict: label_set_name -> ordered label list

    The files are flat TSV rather than one JSON blob because the reader is C++
    with no JSON dependency, and a fixture that needs a parser to read is a
    fixture that will not be read.
    """
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    idx = np.asarray(idx, dtype=int)
    H[idx].astype("<f4").tofile(os.path.join(d, "hidden.f32"))

    with open(os.path.join(d, "labels.tsv"), "w", encoding="utf-8",
              newline="\n") as f:
        f.write("id\t" + "\t".join(n for n, _ in columns) + "\n")
        for k, i in enumerate(idx):
            f.write("%d" % k)
            for _, lab in columns:
                f.write("\t" + str(lab[i]))
            f.write("\n")

    with open(os.path.join(d, "vocab.tsv"), "w", encoding="utf-8",
              newline="\n") as f:
        f.write("column\tindex\tlabel\n")
        for col, _ in columns:
            for li, lab in enumerate(vocabs[col]):
                f.write("%s\t%d\t%s\n" % (col, li, lab))

    with open(os.path.join(d, "metrics.tsv"), "w", encoding="utf-8",
              newline="\n") as f:
        f.write("column\tn\ttemperature\tece_calibrated\tholdout_acc\n")
        for col, _ in columns:
            m = metrics[col]
            f.write("%s\t%d\t%.6f\t%.6f\t%.6f\n"
                    % (col, len(idx), m["temperature"], m["ece_calibrated"],
                       m["holdout_acc"]))

    meta = dict(meta)
    meta["E"] = int(H.shape[1])
    meta["n"] = int(len(idx))
    meta["columns"] = [n for n, _ in columns]
    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("  fixture -> %s (%d held-out rows, cols=%s)"
          % (d, len(idx), ",".join(n for n, _ in columns)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", default="build/head_data/language")
    ap.add_argument("--trading", default="build/head_data/trading")
    ap.add_argument("--bars", default="models/market/AAPL_1d.csv")
    ap.add_argument("--out-dir", default="models/heads")
    ap.add_argument("--report", default="build/head_data/report.json")
    ap.add_argument("--fixture-dir", default="tests/fixtures/head_calibration")
    ap.add_argument("--no-fixture", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    label_sets = parse_default_label_sets(
        os.path.join(ROOT, "src", "classification_head.cpp"))
    actions = parse_decision_actions(
        os.path.join(ROOT, "include", "omniseed", "decision_head.h"))
    print("canonical label sets: %s" % ", ".join(sorted(label_sets)))
    print("canonical actions   : %s" % ", ".join(actions))

    os.makedirs(args.out_dir, exist_ok=True)
    report = {
        "generated_by": "tools/train_heads.py",
        "seed": args.seed,
        "what_the_numbers_mean": (
            "Accuracy is agreement with the SUPPLIED LABEL, not trading "
            "performance. Language labels are human judgements; trading "
            "labels are rules (regime engine + a documented action teacher). "
            "Calibration error (ECE) is the honest-confidence metric: it is "
            "what makes a reported confidence_score meaningful."
        ),
        "heads": {},
    }

    # ======================= LANGUAGE ======================================
    if os.path.isdir(args.language):
        meta, H, ids, set_names, labels = load_dump(args.language)
        print("\n=== language: N=%d E=%d sets=%s" % (H.shape[0], H.shape[1], set_names))
        cols = {name: np.array([r[i] for r in labels]) for i, name in enumerate(set_names)}

        sets_out, proj_rows, bias_rows, fitted_rows = [], [], [], []
        per_set = {}
        temps, cals, eces = [], [], []

        for name in set_names:
            vocab = label_sets.get(name)
            if vocab is None:
                raise SystemExit("label set %r is not in the C++ canonical list" % name)
            inv = {l: i for i, l in enumerate(vocab)}
            unknown = sorted(set(cols[name]) - set(inv))
            if unknown:
                raise SystemExit("labels %r in %s are not canonical for %r"
                                 % (unknown, name, vocab))
            y = np.array([inv[v] for v in cols[name]], dtype=int)
            K = len(vocab)
            tr, hold = stratified_split(y, seed=args.seed)
            r = fit_one(H, y, K, tr, hold, shuffle=True, seed=args.seed)
            print("  %-20s K=%d  train=%d holdout=%d | acc=%.3f "
                  "(macroF1=%.3f) | ECE %.3f -> %.3f (out-of-fold %.3f) | T=%.3f"
                  % (name, K, len(tr), len(hold), r["holdout_acc"],
                     r["holdout_macro_f1"], r["ece_raw"], r["ece_calibrated"],
                     r["ece_out_of_fold"], r["temperature"]))
            print("      per-class recall: %s"
                  % {vocab[i]: r["per_class_recall_holdout"][i] for i in range(K)})
            per_set[name] = public_metrics(r)
            per_set[name]["labels"] = vocab
            per_set[name]["label_counts"] = {
                vocab[i]: int((y == i).sum()) for i in range(K)}

            sets_out.append({"name": name, "labels": vocab})
            proj_rows.append(r["proj"])
            bias_rows.append(r["bias"])
            fitted_rows.extend([1] * K)
            temps.append(r["temperature"])
            cals.append(len(hold) if len(hold) else len(tr))
            eces.append(r["ece_calibrated"])
            if not args.no_fixture:
                emit_fixture(args.fixture_dir, name.replace(".", "_"), H,
                             r["_test_idx"], [(name, cols[name])],
                             {"bin": "models/heads/language_head.bin",
                              "set_name": name,
                              "split": "stratified 70/30 + 5-fold CV for T",
                              "ece_bins_calibrated": r["ece_bins_calibrated"]},
                             {name: {"temperature": r["temperature"],
                                     "ece_calibrated": r["ece_calibrated"],
                                     "holdout_acc": r["holdout_acc"]}},
                             {name: vocab})

        proj = np.vstack(proj_rows)
        bias = np.concatenate(bias_rows)
        # ONE temperature PER SET. A pooled scalar was tried first and made the
        # best-scaled set worse (language.language ECE 0.172 -> 0.194) while
        # fixing the worst, because the intent set wants T ~ 19 and the
        # language set wants T ~ 3.5. Per-set is what the data supports.
        pooled_ece = float(np.average(eces, weights=cals))
        print("  per-set temperatures: %s"
              % {n: round(t, 4) for n, t in zip(set_names, temps)})
        print("  n-weighted pooled ECE after calibration: %.4f" % pooled_ece)

        path = os.path.join(args.out_dir, "language_head.bin")
        size = write_classification_head(
            path, H.shape[1], sets_out, proj, bias,
            sum(len(s["labels"]) for s in sets_out), fitted_rows,
            default_temperature=1.0, temperatures=temps,
            calib_samples_list=cals, calib_errors=eces)
        print("  wrote %s (%d bytes, %d labels)" % (path, size, len(bias)))
        report["heads"]["language"] = {
            "kind": "ClassificationHead", "path": path, "bytes": size,
            "E": H.shape[1], "n_examples": int(H.shape[0]),
            "temperatures": {n: round(t, 6) for n, t in zip(set_names, temps)},
            "calibration_samples": {n: c for n, c in zip(set_names, cals)},
            "ece_calibrated": {n: round(e, 6) for n, e in zip(set_names, eces)},
            "pooled_ece_calibrated_n_weighted": round(pooled_ece, 6),
            "sets": per_set,
            "source": meta,
        }
    else:
        print("!! language dump missing at %s — NOT FITTED" % args.language)
        report["heads"]["language"] = {"status": "NOT FITTED",
                                       "why": "no dump at " + args.language}

    # ======================= TRADING =======================================
    if os.path.isdir(args.trading):
        meta, H, ids, set_names, labels = load_dump(args.trading)
        print("\n=== trading: N=%d E=%d sets=%s" % (H.shape[0], H.shape[1], set_names))
        regime = np.array([r[0] for r in labels])
        closes, _ = read_closes(args.bars)
        print("  bars in CSV: %d (dumped rows: %d)" % (len(closes), H.shape[0]))

        # ---- (a) the regime classifier (ClassificationHead) ----------------
        vocab = label_sets["trading.regime"]
        inv = {l: i for i, l in enumerate(vocab)}
        unknown = sorted(set(regime) - set(inv))
        if unknown:
            raise SystemExit("non-canonical regime labels: %r" % unknown)
        yr = np.array([inv[v] for v in regime], dtype=int)
        tr, hold = chronological_split(len(yr))
        rr = fit_one(H, yr, len(vocab), tr, hold, shuffle=False, seed=args.seed)
        print("  trading.regime       K=%d train=%d holdout=%d | acc=%.3f "
              "(macroF1=%.3f) | ECE %.3f -> %.3f (out-of-fold %.3f) | T=%.3f"
              % (len(vocab), len(tr), len(hold), rr["holdout_acc"],
                 rr["holdout_macro_f1"], rr["ece_raw"], rr["ece_calibrated"],
                 rr["ece_out_of_fold"], rr["temperature"]))
        print("      per-class recall: %s"
              % {vocab[i]: rr["per_class_recall_holdout"][i] for i in range(len(vocab))})
        print("      label counts: %s"
              % {vocab[i]: int((yr == i).sum()) for i in range(len(vocab))})

        rpath = os.path.join(args.out_dir, "trading_regime_head.bin")
        rsize = write_classification_head(
            rpath, H.shape[1], [{"name": "trading.regime", "labels": vocab}],
            rr["proj"], rr["bias"], len(vocab), [1] * len(vocab),
            default_temperature=1.0, temperatures=[rr["temperature"]],
            calib_samples_list=[len(hold) if len(hold) else len(tr)],
            calib_errors=[rr["ece_calibrated"]])
        print("  wrote %s (%d bytes)" % (rpath, rsize))
        report["heads"]["trading_regime"] = {
            "kind": "ClassificationHead", "path": rpath, "bytes": rsize,
            "labels": vocab, "split": "chronological 70/15/15",
            "label_counts": {vocab[i]: int((yr == i).sum()) for i in range(len(vocab))},
            **public_metrics(rr),
            "source": meta,
        }

        # ---- (b) the DecisionHead (7 actions) ------------------------------
        ya, action_names, detail = build_action_labels(ids, regime, closes)
        if action_names != actions:
            raise SystemExit("action order %r != C++ enum order %r"
                             % (action_names, actions))
        dist = {action_names[i]: int((ya == i).sum()) for i in range(len(actions))}
        print("  action teacher       %s" % dist)
        tr, hold = chronological_split(len(ya))
        present = sorted(set(ya.tolist()))
        missing = [action_names[i] for i in range(len(actions)) if i not in present]
        if missing:
            print("  !! actions with ZERO examples: %s — those rows stay "
                  "UNFITTED and the head will NOT report trained()" % missing)

        ra = fit_one(H, ya, len(actions), tr, hold, shuffle=False, seed=args.seed)
        print("  trading.action       K=%d train=%d holdout=%d | acc=%.3f "
              "(macroF1=%.3f) | ECE %.3f -> %.3f (out-of-fold %.3f) | T=%.3f"
              % (len(actions), len(tr), len(hold), ra["holdout_acc"],
                 ra["holdout_macro_f1"], ra["ece_raw"], ra["ece_calibrated"],
                 ra["ece_out_of_fold"], ra["temperature"]))
        print("      per-class recall: %s"
              % {action_names[i]: ra["per_class_recall_holdout"][i]
                 for i in range(len(actions))})

        fitted_flags = [1 if i in present else 0 for i in range(len(actions))]
        n_fitted = sum(fitted_flags)
        apath = os.path.join(args.out_dir, "trading_head.bin")
        asize = write_decision_head(
            apath, H.shape[1], action_names, ra["proj"], ra["bias"],
            # Asset and invalidation are MARKET CONTEXT, not something
            # recoverable from a hidden state. Leaving them empty/zero says
            # "not specified", which the caller must treat as unknown — never
            # as "no stop".
            assets=[""] * len(actions),
            invals=[0.0] * len(actions),
            temperature=ra["temperature"],
            calib_samples=len(hold) if len(hold) else len(tr),
            calib_error=ra["ece_calibrated"],
            fitted=(n_fitted == len(actions)))
        print("  wrote %s (%d bytes, %d/%d action rows fitted)"
              % (apath, asize, n_fitted, len(actions)))

        # ONE fixture for both trading heads: they read the same h[E] rows, so
        # two label columns over one hidden.f32 is the honest layout (and half
        # the bytes).
        if not args.no_fixture and len(rr["_test_idx"]) == len(ra["_test_idx"]):
            action_str = np.array([action_names[i] for i in ya], dtype=object)
            emit_fixture(args.fixture_dir, "trading", H, rr["_test_idx"],
                         [("trading.regime", regime),
                          ("DecisionAction", action_str)],
                         {"bin_regime": "models/heads/trading_regime_head.bin",
                          "bin_action": "models/heads/trading_head.bin",
                          "fitted_rows_action": n_fitted,
                          "split": "chronological 70/30 + 5-fold CV for T",
                          "action_teacher": ACTION_TEACHER_DOC},
                         {"trading.regime": {"temperature": rr["temperature"],
                                             "ece_calibrated": rr["ece_calibrated"],
                                             "holdout_acc": rr["holdout_acc"]},
                          "DecisionAction": {"temperature": ra["temperature"],
                                             "ece_calibrated": ra["ece_calibrated"],
                                             "holdout_acc": ra["holdout_acc"]}},
                         {"trading.regime": vocab,
                          "DecisionAction": action_names})
        report["heads"]["trading_action"] = {
            "kind": "DecisionHead", "path": apath, "bytes": asize,
            "actions": action_names,
            "action_teacher": ACTION_TEACHER_DOC,
            "action_counts": dist,
            "fitted_rows": n_fitted,
            "trained": n_fitted == len(actions),
            "split": "chronological 70/15/15",
            **public_metrics(ra),
            "source": meta,
        }
    else:
        print("!! trading dump missing at %s — NOT FITTED" % args.trading)
        report["heads"]["trading_action"] = {"status": "NOT FITTED",
                                             "why": "no dump at " + args.trading}

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("\nreport -> %s" % args.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
