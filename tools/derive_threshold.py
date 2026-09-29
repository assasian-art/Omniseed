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
#  2. The head is READ FROM THE SHIPPED BLOB, not re-fitted here. An earlier
#     version re-fitted from `tests/fixtures/head_calibration/trading`, whose 369
#     rows ARE the holdout the shipped blob was calibrated on — so it trained on
#     258 holdout rows and reported THAT refit's T and accuracy as if they were
#     the shipped head's (0.1441 / T=20.0427 vs the blob's 0.2439 / T=13.325159).
#     See docs/HOLDOUT_DEFECT.md §"The real defect". The fix: load the blob, load
#     the REAL dump (`build/head_data/trading`, 1,231 rows), rebuild the action
#     teacher the same way `train_heads.py` does, and score the BLOB on the same
#     369-row holdout. No second model, no contamination.
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
from uncertainty_audit import load_decision  # noqa: E402

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


def blob_holdout_probs(blob, H, ya, tr, hold):
    """Score the SHIPPED blob on the holdout rows (no refit).

    `blob` is the dict `uncertainty_audit.load_decision` returns. Its `proj` is
    flat [A*E] row-major and its `bias` is [A]; the softmax uses the blob's own
    stored temperature, so these are the numbers the runtime would produce.
    """
    E, A = blob["E"], blob["A"]
    if H.shape[1] != E:
        raise SystemExit("dump E=%d but blob E=%d" % (H.shape[1], E))
    proj = np.asarray(blob["proj"], dtype=np.float64).reshape(A, E)
    bias = np.asarray(blob["bias"], dtype=np.float64)
    T = float(blob["temp"])

    idx = np.asarray(hold, dtype=int)
    logits = H[idx] @ proj.T + bias            # [n_hold, A]
    # softmax with the blob's temperature, numerically stable
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    ex = np.exp(z)
    probs = ex / ex.sum(axis=1, keepdims=True)
    return probs, ya[idx], T


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
        description="Derive min_confidence from the SHIPPED head's calibration "
                    "curve on held-out data (DECISION 1).")
    ap.add_argument("--dump", default="build/head_data/trading",
                    help="the REAL dump train_heads.py fitted from (all rows, "
                         "not the holdout fixture)")
    ap.add_argument("--blob", default="models/heads/trading_head.bin",
                    help="the shipped DecisionHead to score (read, not re-fitted)")
    ap.add_argument("--bars", default="models/market/AAPL_1d.csv",
                    help="bars, for rebuilding the SAME action teacher")
    ap.add_argument("--fixture",
                    default="tests/fixtures/head_calibration/trading",
                    help="only used to CROSS-CHECK the dump-derived holdout "
                         "against the committed fixture (optional)")
    ap.add_argument("--actions", default="include/omniseed/decision_head.h",
                    help="source of the DecisionAction enum (single source of truth)")
    ap.add_argument("--out", default="tools/threshold_curve.json")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    # ---- 1. the dataset (ALL rows) and the action teacher -------------------
    meta, H, ids, colnames, labels = th.load_dump(args.dump)
    if "trading.regime" not in colnames:
        raise SystemExit("dump has no trading.regime column: %r" % (colnames,))
    regime = [r[colnames.index("trading.regime")] for r in labels]
    closes, _ = th.read_closes(args.bars)

    actions = th.parse_decision_actions(args.actions)
    vocab = {v: i for i, v in enumerate(actions)}
    ya, action_names, _detail = th.build_action_labels(ids, regime, closes)
    if action_names != actions:
        raise SystemExit("action order %r != C++ enum order %r"
                         % (action_names, actions))
    n_dataset = len(ya)

    K = len(actions)
    chance = 1.0 / K
    tr, hold = th.chronological_split(n_dataset)

    # ---- 2. the SHIPPED blob, scored on the holdout (no refit) --------------
    blob = load_decision(args.blob)
    probs, y, T = blob_holdout_probs(blob, H, ya, tr, hold)

    # ---- 2b. cross-check: the same holdout the committed fixture holds ------
    # The fixture holds exactly the holdout rows. If the dump's holdout slice and
    # the fixture disagree on row count, one of them is stale — say so, do not
    # silently pick one. This is the assertion that would have caught §41.
    fixture_n = None
    fx_meta = os.path.join(args.fixture, "meta.json")
    if os.path.isfile(fx_meta):
        with open(fx_meta, encoding="utf-8") as f:
            fixture_n = int(json.load(f)["n"])

    buckets = bucket_table(probs, y)
    chosen, reason, scan, bar = derive(probs, y, chance)

    result = {
        "date": "2026-09-29",
        "decision": "DECISION 1 — principled threshold, not hand-tuning",
        "gated_on": "paper-only (L0). This threshold gates PAPER commits and "
                    "NEVER live money; the live-money gate is C++-enforced.",
        "column": "DecisionAction",
        "method": "READ THE SHIPPED BLOB; score it on the real dump's holdout. "
                  "No refit (the pre-§44 version refitted inside the holdout — "
                  "found in §43, fixed here).",
        "dump": args.dump,
        "blob": args.blob,
        "bars": args.bars,
        "n_dataset": n_dataset,      # ALL rows the blob was fitted on
        "n_train": int(len(tr)),
        "n_holdout": int(len(y)),
        "fixture_n_holdout": fixture_n,
        "holdout_matches_fixture": (
            None if fixture_n is None else (len(y) == fixture_n)),
        "K": K,
        "actions": actions,
        "chance": round(chance, 6),
        "chance_multiple": CHANCE_MULTIPLE,
        "accuracy_bar": round(bar, 6),
        "min_bucket_n": MIN_BUCKET_N,
        "min_pool_n": MIN_POOL_N,
        "temperature": round(float(T), 6),
        "holdout_acc": round(float((probs.argmax(axis=1) == y).mean()), 4),
        "blob_calib_samples": int(blob["cal_n"]),
        "blob_calib_ece": round(float(blob["cal_ece"]), 5),
        "derived_min_confidence": round(float(chosen), 6),
        "reason": reason,
        "curve": buckets,
        "scan": scan,
    }

    if not args.quiet:
        print("=" * 78)
        print("DECISION 1 — threshold derived from the SHIPPED head's curve")
        print("=" * 78)
        print("  dump         %s  (%d rows: train %d / holdout %d)"
              % (args.dump, n_dataset, len(tr), len(y)))
        print("  blob         %s  (T=%.4f, calib n=%d, ece=%.4f)"
              % (args.blob, T, blob["cal_n"], blob["cal_ece"]))
        if fixture_n is not None:
            mark = "OK" if len(y) == fixture_n else "** MISMATCH **"
            print("  fixture      %s  (holdout n=%d)  %s"
                  % (args.fixture, fixture_n, mark))
        print("  actions      K=%d  %s" % (K, " ".join(actions)))
        print("  chance       1/%d = %.6f   ->  accept bar %.6f (2x)"
              % (K, chance, bar))
        print("  holdout acc  %.4f  (the SHIPPED head, not a refit)"
              % result["holdout_acc"])
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
