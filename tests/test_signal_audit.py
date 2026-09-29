#!/usr/bin/env python3
"""Gated test for D1 — the per-head signal audit (docs/CALIBRATION.md §4.5).

WHAT THIS GUARDS. `docs/CALIBRATION.md` §4.5 publishes a table of
(accuracy, chance, ratio, verdict) per fitted label set, and `tools/signal_audit.py`
is the tool that produces it. A published table is worth nothing if it can drift
from the artifacts it describes, so this test asserts the table REGENERATES from
the committed fixtures.

HOW IT AVOIDS THE §38 F3 TRAP (a check that shares the defect it checks).
It does NOT compare `signal_audit.py` against itself. It reads the committed
fixtures INDEPENDENTLY — `meta.json`, `metrics.tsv` and `labels.tsv` — recomputes
every ratio and verdict with a second derivation written here, and only then
compares against what `signal_audit.py --json` reports. If the tool and this test
disagree, one of them is wrong and the test fails; agreement means two
independent readings of the same committed bytes concur.

WHY THE EXPECTED VALUES ARE ALSO HARD-CODED. Recomputing from the fixtures alone
would still pass if a fixture were regenerated with a *different* head, because
the expected numbers would move with it. So the published numbers from §4.5 are
pinned as literals too: the test fails if a re-fit silently moves a ratio AND the
table was not updated. Both halves are needed — one catches tool drift, the other
catches artifact drift.

Stdlib-only on purpose: it must run in CI, where no `.venv` exists.

Usage:  python tests/test_signal_audit.py
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIXTURE_DIR = os.path.join(ROOT, "tests", "fixtures", "head_calibration")
AUDIT = os.path.join(ROOT, "tools", "signal_audit.py")

# The bar, restated here rather than imported, so a change to the tool's constant
# cannot silently redefine what this test is checking.
CHANCE_MULTIPLE = 2.0
MIN_HOLDOUT_N = 30

# docs/CALIBRATION.md §4.5 as published. key -> (n_holdout, accuracy, ratio, verdict).
# If a fixture is regenerated these will move and this test MUST be updated in the
# same commit — that is the point.
PUBLISHED = {
    "language.intent":     (71, 0.6338, 4.437, "SIGNAL"),
    "language.language":   (71, 0.8451, 3.380, "SIGNAL"),
    "trading.regime":      (369, 0.7046, 2.114, "SIGNAL"),
    "language.sentiment":  (71, 0.6761, 2.028, "SIGNAL"),
    "DecisionAction":      (369, 0.2439, 1.707, "NO-SIGNAL"),
    # §44: the first FITTED modality head. Via the focal-codec route (PcmAudio ->
    # FocalCodec ids -> MultimodalBridge -> backbone), no audio->E projection.
    "audio.emotion":       (203, 0.3103, 1.241, "NO-SIGNAL"),
}

# The 7 sets that must appear as unfitted. §44 removed `audio.emotion` from here:
# it gained weights, so the audit table gained a row and this list shrank — the
# exact change this assertion exists to make loud.
EXPECTED_UNFITTED = {
    "language.task", "vision.scene", "vision.anomaly", "audio.wake",
    "audio.speaker", "general.routing", "general.priority",
}

_g_passed = 0
_g_failed = 0


def check(cond: bool, what: str) -> None:
    global _g_passed, _g_failed
    if cond:
        _g_passed += 1
    else:
        _g_failed += 1
        print("  FAIL  %s" % what)


def check_eq(got, want, what: str) -> None:
    check(got == want, "%s (got %r, want %r)" % (what, got, want))


def near(a: float, b: float, tol: float = 5e-4) -> bool:
    return abs(a - b) < tol


def read_fixture(name: str):
    """Independent re-derivation: reads the SAME committed files the tool reads,
    but parses them here so a bug in either reader is not shared."""
    d = os.path.join(FIXTURE_DIR, name)
    with io.open(os.path.join(d, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    n_holdout = int(meta["n"])

    metrics = {}
    with io.open(os.path.join(d, "metrics.tsv"), encoding="utf-8") as f:
        header = None
        for line in f:
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if header is None:
                header = parts
                continue
            metrics[parts[0]] = dict(zip(header, parts))

    labels = []
    with io.open(os.path.join(d, "labels.tsv"), encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if line.strip():
                labels.append(line.split("\t"))
    return n_holdout, metrics, labels


def recompute():
    """Rebuilds the audit table from the fixtures by hand."""
    out = {}
    for name in sorted(os.listdir(FIXTURE_DIR)):
        d = os.path.join(FIXTURE_DIR, name)
        if not os.path.isdir(d):
            continue
        mp = os.path.join(d, "metrics.tsv")
        if not os.path.isfile(mp):
            continue
        n_holdout, metrics, labels = read_fixture(name)
        cols = labels[0][1:]
        for col in cols:
            if col not in metrics:
                continue
            acc = float(metrics[col]["holdout_acc"])
            idx = labels[0].index(col)
            K = len({row[idx] for row in labels[1:]})
            chance = 1.0 / K if K else 0.0
            ratio = (acc / chance) if chance > 0 else 0.0
            if n_holdout < MIN_HOLDOUT_N:
                verdict = "INSUFFICIENT-N"
            elif acc >= CHANCE_MULTIPLE * chance:
                verdict = "SIGNAL"
            else:
                verdict = "NO-SIGNAL"
            out[col] = (n_holdout, round(acc, 4), round(ratio, 3), verdict)
    return out


def run_audit_json():
    """Runs the tool and returns its JSON, or None if it could not run."""
    interp = sys.executable
    try:
        r = subprocess.run(
            [interp, AUDIT, "--json"],
            cwd=ROOT, capture_output=True, text=True, timeout=60)
    except Exception as e:                              # noqa: BLE001
        print("  could not run signal_audit.py: %s" % e)
        return None
    if r.returncode != 0:
        print("  signal_audit.py exited %d\n%s" % (r.returncode, r.stderr))
        return None
    try:
        return json.loads(r.stdout)
    except Exception as e:                              # noqa: BLE001
        print("  signal_audit.py --json is not JSON: %s" % e)
        return None


def main() -> int:
    print("=== test_signal_audit (D1 gate) ===")

    if not os.path.isdir(FIXTURE_DIR):
        print("SKIP: no calibration fixtures at %s" % FIXTURE_DIR)
        print("RESULT: 0 passed, 0 failed (skipped)")
        return 0

    # ---- 1. the test's own re-derivation matches the PUBLISHED table --------
    print("\n-- 1. fixtures reproduce the published §4.5 table --")
    mine = recompute()
    check_eq(set(mine.keys()), set(PUBLISHED.keys()),
             "the fixture set is exactly the published set")
    for col, want in PUBLISHED.items():
        got = mine.get(col)
        if got is None:
            check(False, "%s is present in the fixtures" % col)
            continue
        n_g, acc_g, ratio_g, verdict_g = got
        n_w, acc_w, ratio_w, verdict_w = want
        check_eq(n_g, n_w, "%s n_holdout" % col)
        check(near(acc_g, acc_w), "%s accuracy %.4f ~= %.4f" % (col, acc_g, acc_w))
        check(near(ratio_g, ratio_w, 5e-3),
              "%s ratio %.3f ~= %.3f" % (col, ratio_g, ratio_w))
        check_eq(verdict_g, verdict_w, "%s verdict" % col)

    # ---- 2. the TOOL agrees with this independent re-derivation -------------
    print("\n-- 2. tools/signal_audit.py --json agrees with the re-derivation --")
    audit = run_audit_json()
    if audit is None:
        check(False, "signal_audit.py --json ran")
    else:
        rows = {r["set"]: r for r in audit.get("fitted", [])}
        check_eq(set(rows.keys()), set(PUBLISHED.keys()),
                 "the tool reports the same set of fitted heads")
        for col in PUBLISHED:
            if col not in rows:
                check(False, "%s is in the tool's output" % col)
                continue
            r = rows[col]
            m = mine[col]
            check_eq(int(r["n_holdout"]), m[0], "%s tool n_holdout" % col)
            check(near(float(r["accuracy"]), m[1]),
                  "%s tool accuracy %s ~= %.4f" % (col, r["accuracy"], m[1]))
            check(near(float(r["ratio"]), m[2], 5e-3),
                  "%s tool ratio %s ~= %.3f" % (col, r["ratio"], m[2]))
            check_eq(r["verdict"], m[3], "%s tool verdict" % col)

    # ---- 3. the unfitted sets are reported, not omitted ---------------------
    print("\n-- 3. the 7 unfitted sets are listed, not silently dropped --")
    if audit is not None:
        got_unfitted = {u["set"] for u in audit.get("unfitted", [])}
        check_eq(got_unfitted, EXPECTED_UNFITTED,
                 "the tool lists exactly the 7 unfitted sets")
        # No unfitted set may also appear as fitted — that would be a set that
        # has weights AND is reported as having none.
        check(not (got_unfitted & set(PUBLISHED.keys())),
              "no set is both fitted and unfitted")

    # ---- 4. the held-out flag is the corrected one (§43) --------------------
    print("\n-- 4. the tool carries the CORRECTED held-out flag (§43) --")
    if audit is not None:
        for r in audit.get("fitted", []):
            check(r.get("accuracy_is_held_out") is True,
                  "%s is flagged held-out" % r["set"])
            check("accuracy_is_in_sample" not in r,
                  "%s no longer carries the retracted in-sample flag" % r["set"])

    print("\nRESULT: %d passed, %d failed" % (_g_passed, _g_failed))
    return 0 if _g_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
