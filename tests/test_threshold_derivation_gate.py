#!/usr/bin/env python3
"""Gated test for the §41 FIX — `derive_threshold.py` must describe the SHIPPED head.

THE DEFECT THIS GUARDS (docs/HOLDOUT_DEFECT.md). The pre-§44 `derive_threshold.py`
loaded `tests/fixtures/head_calibration/trading` — whose 369 rows ARE the holdout
the shipped blob was calibrated on — and called `chronological_split(369)`, fitting
on 258 OF THOSE HOLDOUT ROWS. It then reported that refit's `T=20.042676` and
`holdout_acc=0.1441` as if they were the shipped head's numbers. The shipped head's
real numbers, recorded in `tests/fixtures/head_calibration/trading/metrics.tsv`, are
`temperature=13.325159`, `holdout_acc=0.243900`.

So this test asserts the OPPOSITE of a doc-drift check: it ties the tool's output to
the artifact's OWN recorded metrics. If the tool ever again reports a temperature or
a holdout accuracy that is not the one `metrics.tsv` records for `DecisionAction`,
this fails — which is exactly what the old tool would have done.

TWO DIRECTIONS (the §43b lesson — either alone is insufficient):
  * A) `tools/threshold_curve.json` (committed) is consistent with BOTH
       `metrics.tsv` (temperature / accuracy / calibration) AND with the rule
       (the fallback is REACHED, not merely RETAINED), and its dataset counts are
       the real dump's (1,231 = 862 train + 369 holdout).
  * B) `tools/derive_threshold.py` REGENERATES that same JSON when run — so the
       committed file is not a hand-edited claim, it is the tool's output. This
       half needs numpy (the tool does linear algebra), so it is skipped with a
       visible reason when no numpy-capable interpreter exists; half A is
       stdlib-only and always runs, in CI included.

WHY `temperature` IS THE RIGHT THING TO PIN. It is not a tunable the tool may
re-derive: it is stored IN the blob (`DecisionHead` v3) and merely read out. Any
temperature other than the blob's recorded value means the tool looked at a
different model — the defect's exact signature.

Stdlib-only where it can be. Usage:  python tests/test_threshold_derivation_gate.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CURVE = os.path.join(ROOT, "tools", "threshold_curve.json")
METRICS = os.path.join(ROOT, "tests", "fixtures", "head_calibration", "trading",
                       "metrics.tsv")
TOOL = os.path.join(ROOT, "tools", "derive_threshold.py")

# The one row of metrics.tsv this whole test is about.
COLUMN = "DecisionAction"

g_pass = g_fail = g_skip = 0


def check(ok, what):
    global g_pass, g_fail
    if ok:
        g_pass += 1
        return True
    g_fail += 1
    print("  FAIL  %s" % what)
    return False


def skip(what):
    global g_skip
    g_skip += 1
    print("  SKIP  %s" % what)


def read_metrics_row(path, column):
    """(n, temperature, ece, holdout_acc) for `column`, or None if absent."""
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        header = f.readline()
        if not header:
            return None
        cols = header.rstrip("\r\n").split("\t")
        want = {"n": None, "temperature": None,
                "ece_calibrated": None, "holdout_acc": None}
        for k in want:
            if k in cols:
                want[k] = cols.index(k)
        for line in f:
            fields = line.rstrip("\r\n").split("\t")
            if not fields or fields[0] != column:
                continue
            out = {}
            for k, i in want.items():
                out[k] = float(fields[i]) if i is not None and i < len(fields) else None
            return out
    return None


def main():
    print("=== threshold-derivation gate (§41 fix) ===")

    # ---- the artifact's OWN record ---------------------------------------------
    m = read_metrics_row(METRICS, COLUMN)
    if m is None:
        skip("metrics.tsv has no %s row: %s" % (COLUMN, METRICS))
        print("\n=== RESULT: %d passed, %d failed, %d skipped ==="
              % (g_pass, g_fail, g_skip))
        return 0 if g_fail == 0 else 1

    print("  metrics.tsv %s: T=%.6f  ece=%.6f  acc=%.6f  n=%d"
          % (COLUMN, m["temperature"], m["ece_calibrated"],
             m["holdout_acc"], int(m["n"])))

    if not os.path.isfile(CURVE):
        skip("threshold_curve.json is absent (run derive_threshold.py)")
        print("\n=== RESULT: %d passed, %d failed, %d skipped ==="
              % (g_pass, g_fail, g_skip))
        return 0 if g_fail == 0 else 1

    with open(CURVE, encoding="utf-8") as f:
        d = json.load(f)

    # ---- A1. the tool describes THE SHIPPED HEAD (the whole point) --------------
    check(d.get("column") == COLUMN,
          "A1 the curve is for %s (got %r)" % (COLUMN, d.get("column")))
    check(d.get("blob") == "models/heads/trading_head.bin",
          "A1 the curve names the shipped blob (got %r)" % d.get("blob"))

    t_curve = d.get("temperature")
    check(t_curve is not None and abs(t_curve - m["temperature"]) < 5e-7,
          "A1 temperature READS the blob's recorded value (%.6f), not a refit "
          "(curve %.6f, metrics %.6f — a mismatch is the §41 defect's signature)"
          % (m["temperature"], t_curve or float("nan"), m["temperature"]))

    a_curve = d.get("holdout_acc")
    check(a_curve is not None and abs(a_curve - m["holdout_acc"]) < 5e-5,
          "A1 holdout_acc matches the shipped head's recorded value (%.4f vs "
          "%.4f) — NOT the old contaminated refit's 0.1441"
          % (m["holdout_acc"], a_curve or float("nan")))

    # The old tool's numbers must not appear anywhere as this tool's own output.
    check(t_curve is None or abs(t_curve - 20.042676) > 1e-4,
          "A1 the curve does NOT carry the OLD refit temperature 20.042676")
    check(a_curve is None or abs(a_curve - 0.1441) > 1e-4,
          "A1 the curve does NOT carry the OLD refit accuracy 0.1441")

    # ---- A2. the calibration metadata is the blob's, read not invented ----------
    if m["ece_calibrated"] is not None and d.get("blob_calib_ece") is not None:
        check(abs(d["blob_calib_ece"] - m["ece_calibrated"]) < 5e-5,
              "A2 blob_calib_ece matches metrics.tsv (%.5f vs %.5f)"
              % (d["blob_calib_ece"], m["ece_calibrated"]))
    if m["n"] is not None and d.get("blob_calib_samples") is not None:
        check(int(d["blob_calib_samples"]) == int(m["n"]),
              "A2 blob_calib_samples == the holdout size metrics.tsv records (%d)"
              % int(m["n"]))

    # ---- A3. the COUNTS are the real dump's, and relabelled correctly -----------
    # The old file said n_total=369 (the FIXTURE size, presented as the dataset).
    # The dump is 1,231 rows; 862/369 is the chronological split of it.
    check("n_total" not in d,
          "A3 the mislabel n_total is GONE (it presented the 369-row fixture as "
          "the dataset total)")
    if all(k in d for k in ("n_dataset", "n_train", "n_holdout")):
        check(d["n_dataset"] == d["n_train"] + d["n_holdout"],
              "A3 n_dataset == n_train + n_holdout (%r == %r + %r)"
              % (d["n_dataset"], d["n_train"], d["n_holdout"]))
        check(d["n_dataset"] == 1231,
              "A3 n_dataset is the real dump's 1231 rows (got %r) — not 369"
              % d.get("n_dataset"))
        check(d["n_holdout"] == int(m["n"]),
              "A3 n_holdout equals the holdout metrics.tsv records (%d)"
              % int(m["n"]))
    else:
        check(False, "A3 n_dataset/n_train/n_holdout present")

    # The fixture cross-check the tool performs must agree.
    check(d.get("holdout_matches_fixture") is True,
          "A3 the dump-derived holdout matches the committed fixture row count")

    # ---- A4. the RULE is applied, and its answer is the REACHED fallback --------
    # §41's distinction: 0.50 is correct because every candidate FAILED, not
    # because 0.50 was the default never exercised. Assert the reason says so.
    check(d.get("derived_min_confidence") == 0.5,
          "A4 the derived min_confidence is 0.50 (fail-closed)")
    reason = str(d.get("reason", ""))
    check("NO threshold satisfies both conditions" in reason,
          "A4 the reason reports NO qualifying threshold (the fallback was "
          "REACHED, not RETAINED)")
    check(d.get("chance_multiple") == 2.0,
          "A4 the rule used the 2x bar")
    check(d.get("accuracy_bar") is not None
          and abs(d["accuracy_bar"] - 2.0 / 7.0) < 1e-6,
          "A4 the bar is 2/7 = %.6f" % (2.0 / 7.0))

    # And the geometry that MAKES it the fallback: the cumulative pool never
    # clears the bar, though a lone per-bucket row does — the note in
    # docs/CALIBRATION.md §4.4 depends on this shape.
    scan = d.get("scan") or []
    passing_pools = [s for s in scan if s.get("passes")]
    check(passing_pools == [],
          "A4 NO cumulative pool passes the bar (that is why the fallback is "
          "reached) — %d pools claimed to pass" % len(passing_pools))
    if scan:
        pool_accs = [s["accuracy"] for s in scan
                     if s.get("accuracy") is not None and s.get("n", 0) >= 30]
        if pool_accs:
            check(max(pool_accs) < 2.0 / 7.0,
                  "A4 every n>=30 cumulative pool sits BELOW the bar "
                  "(max pool acc %.4f < %.4f)" % (max(pool_accs), 2.0 / 7.0))

    # ---- B. the tool REGENERATES the committed file (numpy-gated) ---------------
    py = sys.executable
    have_numpy = True
    try:
        import numpy  # noqa: F401
    except Exception:  # noqa: BLE001
        have_numpy = False
    if not have_numpy:
        skip("B regeneration: this interpreter has no numpy; half A (stdlib-only) "
             "already ran")
    else:
        tmp = os.path.join(tempfile.mkdtemp(prefix="omniseed_thr_"), "curve.json")
        proc = subprocess.run(
            [py, TOOL, "--quiet", "--out", tmp],
            cwd=ROOT, capture_output=True, text=True)
        if proc.returncode != 0:
            check(False, "B the tool runs (rc=%d): %s"
                  % (proc.returncode, (proc.stderr or "").strip()[-300:]))
        elif not os.path.isfile(tmp):
            check(False, "B the tool wrote its --out file")
        else:
            with open(tmp, encoding="utf-8") as f:
                fresh = json.load(f)
            for key in ("temperature", "holdout_acc", "n_dataset", "n_train",
                        "n_holdout", "derived_min_confidence", "blob_calib_samples",
                        "blob_calib_ece"):
                check(fresh.get(key) == d.get(key),
                      "B regenerated %s == committed (%r vs %r)"
                      % (key, fresh.get(key), d.get(key)))
            # And the fresh run must ALSO agree with metrics.tsv, so half B cannot
            # pass by reproducing a wrong-but-consistent file.
            check(abs(fresh.get("temperature", 0) - m["temperature"]) < 5e-7,
                  "B the FRESH run's temperature is still the blob's %.6f"
                  % m["temperature"])

    print("\n=== RESULT: %d passed, %d failed, %d skipped ==="
          % (g_pass, g_fail, g_skip))
    return 0 if g_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
