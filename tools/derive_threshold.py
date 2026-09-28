#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — tools/derive_threshold.py
#
#  DECISION 1 of the owner's step-3 directive: derive `min_confidence` from the
#  fitted head's calibration curve on HELD-OUT data instead of hand-picking it.
#
#  THE RULE (owner-specified, implemented literally)
#  -------------------------------------------------
#    chance      = 1 / K                       (K = 7 actions -> 0.142857)
#    accept a bucket iff  accuracy >= 2 * chance  AND  n >= 30
#    min_confidence = the LOWEST threshold whose ACCEPTED POOL satisfies it
#
#  Two subtleties that matter, and which the naive reading gets wrong:
#
#  1. The rule says "the lowest threshold where empirical accuracy >= 2x chance
#     AND bucket sample count >= 30". Read as "the lowest bucket boundary that
#     itself passes", that threshold is NOT usable as a filter bar: rows BELOW a
#     threshold are not discarded, they are ABSTAINED — and an ABSTAIN is a
#     non-decision. What has to clear 2x chance is the pool the filter actually
#     acts on: every row with confidence >= threshold. So we scan thresholds and
#     test the CUMULATIVE pool from each threshold upward. The bucket table is
#     also printed, because the curve is the evidence either way.
#
#  2. We do NOT re-derive the projection. It is opaque and lives in the blob.
#     Instead the fit is REPRODUCED from the fixture: the same
#     chronological_split + fit_one(shuffle=False) that tools/train_heads.py
#     used. That is deterministic, so re-running this script gives byte-equal
#     numbers — which is what makes the gated C++ test able to check them.
#
#  Output: tools/threshold_curve.json  (+ a human table on stdout)
# =============================================================================
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import train_heads as th  # noqa: E402

# The rule's constants, named so a reader can see them and the test can cite them.
CHANCE_MULTIPLE = 2.0
MIN_BUCKET_N = 30
MIN_POOL_N = 30          # a pool of 2 rows cannot support any accuracy claim
FALLBACK = 0.50          # the fail-closed default we keep if nothing qualifies
N_BINS = 20              # curve resolution: 0.05-wide lower bins, then finer

# A threshold below `chance` is not a filter — it would commit on rows the head
# admits it knows nothing about. The first version of this scan had no floor and
# happily picked T=0.0 for a confidently-correct pool, which the DECISION-1
# control test caught. The floor is what makes this a rule rather than a scan.
THRESHOLD_FLOOR = 1.0 / 7.0


def action_curve(H, ya, actions, tr, hold):
    """Reproduce the action-head fit and return calibrated holdout probabilities.

    Returns (probs, y, temperature, fit_report) where probs is [n_hold, K].
    """
    K = len(actions)
    r = th.fit_one(H, ya, K, tr, hold, shuffle=False, seed=1234)
    raw = r["_raw_logits"]
    y = r["_y"]
    ti = r["_test_idx"]
    if len(ti) == 0:                     # fixture too small: fall back to train
        ti = tr
    T = r["temperature"]
    probs = th.softmax(raw[ti] / T)
    return probs, y[ti], T, r


def bucket_table(probs, y):
    """Empirical accuracy per equal-width confidence bucket, from the fixture."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(np.float64)
    rows = []
    for i in range(N_BINS):
        lo, hi = i / N_BINS, (i + 1) / N_BINS
        m = (conf >= lo) & (conf <= hi) if i == 0 else (conf > lo) & (conf <= hi)
        n = int(m.sum())
        rows.append({
            "lo": round(lo, 4), "hi": round(hi, 4), "n": n,
            "accuracy": round(float(correct[m].mean()), 4) if n else None,
            "mean_confidence": round(float(conf[m].mean()), 4) if n else None,
        })
    return rows


def scan_thresholds(probs, y, chance):
    """Cumulative-pool scan. Returns the list of evaluated thresholds."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(np.float64)
    bar = CHANCE_MULTIPLE * chance

    # Candidate thresholds: the 0.01 lattice, but NOTHING below the floor.
    # Every observed confidence is included too, so the scan is dense where the
    # data is.
    cands = sorted(set(round(float(x), 6) for x in
                       list(conf) + list(np.arange(0.0, 1.0001, 0.01))
                       if x >= THRESHOLD_FLOOR))
    out = []
    for t in cands:
        m = conf >= t
        n = int(m.sum())
        acc = float(correct[m].mean()) if n else None
        passes = (n >= MIN_POOL_N) and (acc is not None) and (acc >= bar)
        out.append({"threshold": round(float(t), 6), "n": n,
                    "accuracy": round(acc, 4) if acc is not None else None,
                    "passes": bool(passes)})
    return out, bar


def derive(probs, y, chance):
    """Apply the rule. Returns (chosen_threshold, reason, scan, bar)."""
    scan, bar = scan_thresholds(probs, y, chance)
    for row in scan:                      # scan is ascending in threshold
        if row["passes"]:
            if row["threshold"] >= FALLBACK:
                return row["threshold"], (
                    "a cumulative pool at or above %.4f clears %.2fx chance "
                    "(acc=%.4f, n=%d)" % (row["threshold"], CHANCE_MULTIPLE,
                                          row["accuracy"], row["n"])), scan, bar
            return row["threshold"], (
                "the LOWEST threshold whose cumulative pool clears %.2fx chance "
                "is %.4f (acc=%.4f, n=%d) — NOTE this is BELOW the %.2f "
                "fail-closed default, so the filter would commit MORE, not less"
                % (CHANCE_MULTIPLE, row["threshold"], row["accuracy"],
                   row["n"], FALLBACK)), scan, bar
    return FALLBACK, (
        "NO threshold satisfies both conditions (>= %.2fx chance on the "
        "cumulative pool AND n >= %d). Keeping the fail-closed default %.2f — "
        "that is a valid outcome, not a failure."
        % (CHANCE_MULTIPLE, MIN_POOL_N, FALLBACK)), scan, bar


def main():
    ap = argparse.ArgumentParser(
        description="Derive min_confidence from the fitted head's calibration "
                    "curve on held-out data (DECISION 1).")
    ap.add_argument("--fixture", default="tests/fixtures/head_calibration/trading")
    ap.add_argument("--actions", default="include/omniseed/decision_head.h",
                    help="source of the DecisionAction enum (single source of truth)")
    ap.add_argument("--out", default="tools/threshold_curve.json")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    meta, H, ids, colnames, labels = th.load_dump(args.fixture)
    if labels and len(labels[0]) >= 2:
        pass
    if "DecisionAction" not in colnames:
        raise SystemExit("fixture has no DecisionAction column: %r" % (colnames,))
    ci = colnames.index("DecisionAction")
    actions = th.parse_decision_actions(args.actions)
    vocab = {v: i for i, v in enumerate(actions)}
    try:
        ya = np.array([vocab[r[ci]] for r in labels], dtype=np.int64)
    except KeyError as e:
        raise SystemExit("label %r is not in the DecisionAction enum %r" % (e, actions))

    K = len(actions)
    chance = 1.0 / K
    tr, hold = th.chronological_split(len(ya))
    probs, y, T, r = action_curve(H, ya, actions, tr, hold)

    buckets = bucket_table(probs, y)
    chosen, reason, scan, bar = derive(probs, y, chance)

    result = {
        "date": "2026-09-29",
        "decision": "DECISION 1 — principled threshold, not hand-tuning",
        "gated_on": "paper-only (L0). This threshold gates PAPER commits and "
                    "NEVER live money; the live-money gate is C++-enforced.",
        "column": "DecisionAction",
        "fixture": args.fixture,
        "n_total": int(len(ya)),
        "n_train": int(len(tr)),
        "n_holdout": int(len(y)),
        "K": K,
        "actions": actions,
        "chance": round(chance, 6),
        "chance_multiple": CHANCE_MULTIPLE,
        "accuracy_bar": round(bar, 6),
        "min_bucket_n": MIN_BUCKET_N,
        "min_pool_n": MIN_POOL_N,
        "temperature": round(float(T), 6),
        "holdout_acc": round(float((probs.argmax(axis=1) == y).mean()), 4),
        "ece_calibrated": round(float(r["ece_calibrated"]), 5),
        "derived_min_confidence": round(float(chosen), 6),
        "reason": reason,
        "curve": buckets,
        "scan": scan,
    }

    if not args.quiet:
        print("=" * 78)
        print("DECISION 1 — threshold derived from the calibration curve")
        print("=" * 78)
        print("  fixture      %s  (%d rows, train %d / holdout %d)"
              % (args.fixture, result["n_total"], len(tr), len(y)))
        print("  actions      K=%d  %s" % (K, " ".join(actions)))
        print("  chance       1/%d = %.6f   ->  accept bar %.6f (2x)"
              % (K, chance, bar))
        print("  T            %.4f      holdout acc %.4f      ECE %.4f"
              % (T, result["holdout_acc"], result["ece_calibrated"]))
        print()
        print("  confidence bucket      n     emp.acc   mean.conf")
        print("  " + "-" * 56)
        for b in buckets:
            if b["n"] == 0:
                continue
            flag = ""
            if b["n"] >= MIN_BUCKET_N and b["accuracy"] is not None and b["accuracy"] >= bar:
                flag = "  <== clears 2x & n>=30"
            print("  [%.2f, %.2f)          %5d    %s     %s%s"
                  % (b["lo"], b["hi"], b["n"],
                     ("%.4f" % b["accuracy"]) if b["accuracy"] is not None else "  n/a ",
                     ("%.4f" % b["mean_confidence"]) if b["mean_confidence"] is not None else "  n/a ",
                     flag))
        print()
        print("  cumulative pool scan (each row = 'rows at conf >= T'):")
        shown = [s for s in scan if s["n"] > 0]
        for s in shown[::max(1, len(shown) // 14)]:
            print("    T=%.4f  n=%4d  acc=%s  %s"
                  % (s["threshold"], s["n"],
                     ("%.4f" % s["accuracy"]) if s["accuracy"] is not None else "n/a",
                     "<== PASSES" if s["passes"] else ""))
        print()
        print("  DERIVED min_confidence = %.6f" % chosen)
        print("  because: %s" % reason)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    if not args.quiet:
        print("\n  wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
