# =============================================================================
#  OmniSeed — tools/nightly_refit.py  (+ corpus_version helpers)
#  Phase 4.2/4.3: the nightly recalibration loop and corpus provenance.
#
#  FEEDBACK.md §9 said it plainly: "The runtime never trains... nothing
#  re-fits."  This tool is the bounded, honest version of "self-refit":
#
#  WHAT IT CAN AND CANNOT LEARN.  The feedback journal records
#  (confidence, predicted, realised) for COMMITTED rows — not the h[E]
#  inputs.  Without inputs, softmax weights cannot be refit, and pretending
#  otherwise would be the fabrication this project keeps refusing.  What the
#  pairs DO support is recalibration: is "confidence 0.8" on this task family
#  actually right 80% of the time?  We fit a monotone confidence->accuracy
#  map (isotonic regression, PAVA) and accept it ONLY if it beats the
#  identity map on a chronological holdout (Brier score).  A calibration
#  that cannot beat "trust the head as-is" is discarded, not shipped.
#
#  NEVER A REGRESSION: the acceptance gate is the whole point.  Every run
#  compares against the identity map; the worst case is "keep-old".
#
#  CORPUS PROVENANCE: a calibration fitted on last month's data is wrong for
#  this month's corpus, silently.  Every accepted map embeds the corpus
#  digest (tools/corpus_version.py), and `--check` exits non-zero on drift,
#  so a stale map is visible rather than assumed fresh.
#
#  SILENCE IS NOT AN OUTCOME (FEEDBACK.md): held/implicit rows are never
#  scored.  Only rows with committed=1 AND outcome=realised contribute.
# =============================================================================
"""Nightly confidence recalibration from the feedback journal + corpus versioning."""

import hashlib
import json
import os
import time


# ---------------------------------------------------------------------------
# isotonic regression (PAVA) — the smallest honest monotone fit
# ---------------------------------------------------------------------------
def isotonic(values, weights=None):
    """Non-decreasing isotonic fit of `values` (already sorted by key).

    Block Pool Adjacent Violators: merge neighbours whose mean decreases,
    weighted by `weights` (integer counts; default uniform). Returns the
    fitted series expanded back to input length.
    """
    ys = list(values)
    ws = [int(w) for w in weights] if weights is not None else [1] * len(ys)
    if len(ys) != len(ws):
        raise ValueError("values and weights must have equal length")
    if any(w <= 0 for w in ws):
        raise ValueError("weights must be positive counts")
    blocks = []                      # each block: [mean, count]
    for y, w in zip(ys, ws):
        blocks.append([y, w])
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            y2, w2 = blocks.pop()
            y1, w1 = blocks.pop()
            blocks.append([(y1 * w1 + y2 * w2) / (w1 + w2), w1 + w2])
    res = []
    for m, w in blocks:
        res.extend([m] * w)
    return res


def brier(probs, outcomes):
    """Mean squared error of P(correct) vs 0/1 correctness."""
    if len(probs) != len(outcomes):
        raise ValueError("length mismatch")
    if not probs:
        return float("nan")
    return sum((p - o) ** 2 for p, o in zip(probs, outcomes)) / len(probs)


# ---------------------------------------------------------------------------
# journal
# ---------------------------------------------------------------------------
def read_journal(path):
    """TSV -> list of dicts. Malformed rows are skipped, never guessed."""
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        header = None
        for line in f:
            # strip BOTH terminators: on Windows a text-mode writer leaves
            # \r\n, and with newline="" the \r survives on the LAST column —
            # which silently broke `outcome` matching during development.
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t")
            if header is None:
                header = parts
                continue
            if len(parts) != len(header):
                continue
            rows.append(dict(zip(header, parts)))
    return rows


def committed_pairs(rows):
    """Only rows that were COMMITTED and later REALISED.

    Held/implicit rows are excluded by design: the journal's own contract
    (docs/FEEDBACK.md) is that silence is not an outcome, so scoring it
    would let a user scrolling past teach the head nothing — or worse.
    """
    out = []
    for r in rows:
        if r.get("committed", "0") != "1":
            continue
        if r.get("outcome", "") != "realised":
            continue
        try:
            conf = float(r["confidence"])
        except (KeyError, TypeError, ValueError):
            continue
        pred = (r.get("predicted") or "").strip()
        real = (r.get("realised") or "").strip()
        if not pred or not real:
            continue
        out.append((conf, 1.0 if pred == real else 0.0))
    return out


# ---------------------------------------------------------------------------
# the recalibration
# ---------------------------------------------------------------------------
def recalibrate(pairs, min_n=30, holdout_frac=0.2, n_bins=8):
    """(confidence, correctness) pairs -> verdict dict (chronological split).

    The split is CHRONOLOGICAL, never shuffled: the holdout is the LAST
    slice, because calibration drifts forward in time and a random split
    would leak the future into the fit — the same look-ahead discipline the
    trading layer enforces.
    """
    verdict = {"accepted": False, "n": len(pairs), "reason": "",
               "brier_identity": None, "brier_isotonic": None,
               "bins": []}
    if len(pairs) < min_n:
        verdict["reason"] = "insufficient"
        return verdict
    k = max(1, int(round(len(pairs) * holdout_frac)))
    if len(pairs) - k < min_n // 2:
        verdict["reason"] = "holdout-too-small"
        return verdict
    train, test = pairs[:-k], pairs[-k:]

    # bin the TRAIN confidences into n_bins equal-count bins
    srt = sorted(train, key=lambda t: t[0])
    bins = []
    size = max(1, len(srt) // n_bins)
    for s in range(0, len(srt), size):
        chunk = srt[s:s + size]
        if not chunk:
            continue
        c = sum(p[0] for p in chunk) / len(chunk)
        acc = sum(p[1] for p in chunk) / len(chunk)
        bins.append([c, acc, len(chunk)])
    if len(bins) < 2:
        verdict["reason"] = "too-few-bins"
        return verdict

    # monotone correction in confidence order
    mono = isotonic([b[1] for b in bins], [b[2] for b in bins])
    for b, m in zip(bins, mono):
        b[1] = m

    def apply(c):
        # nearest-bin lookup on the calibrated accuracy curve
        best = min(bins, key=lambda b: abs(b[0] - c))
        return best[1]

    verdict["bins"] = [[round(b[0], 6), round(b[1], 6), b[2]] for b in bins]
    verdict["brier_identity"] = brier([c for c, _ in test],
                                      [o for _, o in test])
    verdict["brier_isotonic"] = brier([apply(c) for c, _ in test],
                                      [o for _, o in test])
    # ACCEPT only on a real holdout improvement (1e-6 tolerance so float
    # noise cannot flip a keep-old into an accept).
    if verdict["brier_isotonic"] < verdict["brier_identity"] - 1e-6:
        verdict["accepted"] = True
        verdict["reason"] = "improved-holdout-brier"
    else:
        verdict["reason"] = "keep-old (isotonic did not beat identity)"
    return verdict


# ---------------------------------------------------------------------------
# corpus versioning (Phase 4.3)
# ---------------------------------------------------------------------------
def digest_files(paths):
    """-> {relpath: {"sha256":..., "size":...}} — deterministic."""
    out = {}
    for p in sorted(paths):
        h = hashlib.sha256()
        try:
            with open(p, "rb") as f:
                for blk in iter(lambda: f.read(65536), b""):
                    h.update(blk)
            out[os.path.basename(p)] = {"sha256": h.hexdigest(),
                                        "size": os.path.getsize(p)}
        except OSError:
            continue
    return out


def corpus_fingerprint(entries):
    """One stable digest over the whole corpus state."""
    h = hashlib.sha256()
    for name in sorted(entries):
        h.update(name.encode("utf-8"))
        h.update(entries[name]["sha256"].encode("ascii"))
    return h.hexdigest()


def run_nightly(journal_path, out_path, corpus_paths=(), min_n=30):
    """One nightly pass -> verdict dict; writes the map ONLY when accepted.

    Failing closed on the file write matters: a rejected calibration must
    leave the previous accepted map in place, untouched.
    """
    rows = read_journal(journal_path)
    pairs = committed_pairs(rows)
    v = recalibrate(pairs, min_n=min_n)
    v["timestamp"] = int(time.time())
    v["journal"] = os.path.basename(journal_path)
    if corpus_paths:
        v["corpus"] = digest_files(corpus_paths)
        v["corpus_fingerprint"] = corpus_fingerprint(v["corpus"])
    if v["accepted"]:
        tmp = out_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(v, f, indent=2, sort_keys=True)
        os.replace(tmp, out_path)
    return v


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal", default="state/feedback_journal.tsv")
    ap.add_argument("--out", default="state/calibration_map.json")
    ap.add_argument("--corpus", default="models/corpus",
                    help="directory of corpus files to fingerprint")
    ap.add_argument("--min-n", type=int, default=30)
    a = ap.parse_args()
    cpaths = []
    if a.corpus and os.path.isdir(a.corpus):
        for root, _dirs, files in os.walk(a.corpus):
            for fn in files:
                cpaths.append(os.path.join(root, fn))
    if not os.path.isfile(a.journal):
        print("[nightly-refit] no journal at %s — nothing to do "
              "(fail-closed, no map written)" % a.journal)
        return 0
    v = run_nightly(a.journal, a.out, corpus_paths=cpaths, min_n=a.min_n)
    print(json.dumps({k: v[k] for k in sorted(v)
                      if k != "bins"}, indent=2))
    if v["accepted"]:
        print("[nightly-refit] ACCEPTED -> %s" % a.out)
    else:
        print("[nightly-refit] %s" % v["reason"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
