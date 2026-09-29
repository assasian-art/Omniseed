#!/usr/bin/env python3
"""D1 — PER-HEAD SIGNAL AUDIT.

Publishes, for every FITTED label set, whether the fitted head carries real
signal or is at chance. The owner's instruction was explicit: *publish, don't
assume*. The action head is at chance on holdout; before pivoting the learned
heads' role (D2) we must know WHERE real signal lives and where it does not.

    set name | n_holdout | accuracy | chance | ratio | verdict

VERDICT is one of:

    SIGNAL           accuracy >= 2x chance  AND n >= 30
    NO-SIGNAL        accuracy <  2x chance  AND n >= 30
    INSUFFICIENT-N   n < 30 (the ratio is not meaningful at this size)

WHY 2x CHANCE AND n >= 30. These are DECISION 1's own bars, reused rather than
invented a second time — a second, different bar would mean two definitions of
"good enough" in one project. At K=7, chance is 1/7 = 0.1429 and the bar is
0.2857. At K=3, chance is 0.3333 and the bar is 0.6667.

WHY THE HOLDOUT IS THE WHOLE ROW COUNT HERE. Each `meta.json` records `n` as the
HELDOUT size its metrics were computed on (the fit itself used the 70% train
split). So `n_holdout = meta["n"]`. Reading it as the full set size would
overstate the evidence by ~3x.

SOURCE. Every number comes from `tests/fixtures/head_calibration/*/meta.json`
and the sibling `metrics.tsv`, both of which are COMMITTED. Nothing is
re-derived from a blob here, so the table cannot drift from the artifact it
describes. The C++ gate `tests/test_calibration.cpp` is what re-derives from the
blob; this script is the human-readable publication of the same facts.

⚠️ THE ACCURACY COLUMN **IS** HELD OUT — VERIFIED, NOT ASSUMED (§43 retraction).
`metrics.tsv` names its column `holdout_acc` and that name is CORRECT. An earlier
draft of this script claimed otherwise: `score_decision()` / `score_classification()`
loop `f.n()` rows, and `f.n()` was read as the full dataset. It is not — `f.n()` is
the FIXTURE size, and `tools/train_heads.py::emit_fixture()` writes ONLY the
holdout rows into the fixture (`idx = rr["_test_idx"]`, with
`_test_idx = hold_used = hold`). So scoring every fixture row scores only the 30%
the fit never saw. Verified four ways in `docs/HOLDOUT_DEFECT.md`: the fixture's
`hidden.f32`, `labels.tsv` and `metrics.tsv` row counts all agree (369), a 70/30
split of the 1,231-row dump gives 369, `emit_fixture` writes `H[idx]`, and the
shipped blob records `calib_samples = 369`. The published accuracies below are
genuine held-out figures. `accuracy_is_held_out: True` is carried per row.

(The real defect that investigation found is in `tools/derive_threshold.py`, which
refits inside the fixture — see `docs/HOLDOUT_DEFECT.md` §"The real defect". It
does not touch `metrics.tsv`.)

WHAT IS *NOT* HERE. The 8 unfitted sets are absent by construction, not by
omission: they have no fitted weights, so they have no holdout accuracy, so they
have no verdict. They are listed explicitly in the output as UNFITTED so a
reader cannot mistake absence for a good result.

Usage:
    python tools/signal_audit.py                 # print the table
    python tools/signal_audit.py --json          # machine-readable
    python tools/signal_audit.py --markdown      # paste into docs/CALIBRATION.md
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys

# The bar, reused from DECISION 1 (§41) rather than re-invented.
CHANCE_MULTIPLE = 2.0
MIN_HOLDOUT_N = 30

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIXTURE_DIR = os.path.join(ROOT, "tests", "fixtures", "head_calibration")

# The 7 sets that have NO fitted weights. Listed so the report can say so out
# loud; a table that simply omits them invites the reader to assume they pass.
# `audio.emotion` LEFT this list in §44: it now has a fitted blob (1.241x chance,
# NO-SIGNAL) via the focal-codec route, so it appears in the table above.
UNFITTED = [
    ("language.task", "INSUFFICIENT-DATA", "no labelled rows exist anywhere"),
    ("vision.scene", "INSUFFICIENT-DATA", "no images in tree; CIFAR fetch abandoned"),
    ("vision.anomaly", "INSUFFICIENT-DATA", "needs scene images first"),
    ("audio.wake", "INSUFFICIENT-DATA", "Speech Commands v2 unfetchable; use enroll-audio"),
    ("audio.speaker", "INSUFFICIENT-DATA", "needs other speakers; owner has not enrolled"),
    ("general.routing", "INSUFFICIENT-DATA", "no labelled rows exist anywhere"),
    ("general.priority", "NO-SIGNAL", "no objective label exists; must stay unfitted"),
]


def read_labels(path: str) -> list[list[str]]:
    """Reads a fixture labels.tsv. Strips CRLF: the files are committed with
    native line endings and a stray \\r on the last header cell makes every
    column lookup fail (this bit the first version of this script)."""
    rows = []
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if line.strip():
                rows.append(line.split("\t"))
    return rows


def audit() -> dict:
    """Reads every committed fixture and returns the audit."""
    if not os.path.isdir(FIXTURE_DIR):
        raise SystemExit("no calibration fixtures at " + FIXTURE_DIR)

    sets: list[dict] = []
    for name in sorted(os.listdir(FIXTURE_DIR)):
        d = os.path.join(FIXTURE_DIR, name)
        meta_p = os.path.join(d, "meta.json")
        metrics_p = os.path.join(d, "metrics.tsv")
        labels_p = os.path.join(d, "labels.tsv")
        if not (os.path.isfile(meta_p) and os.path.isfile(metrics_p)
                and os.path.isfile(labels_p)):
            continue

        with io.open(meta_p, encoding="utf-8") as f:
            meta = json.load(f)
        n_holdout = int(meta["n"])

        # metrics.tsv: column, n, temperature, ece_calibrated, holdout_acc
        metrics = {}
        with io.open(metrics_p, encoding="utf-8") as f:
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

        lab = read_labels(labels_p)
        cols = lab[0][1:]      # column 0 is the provenance id

        for col in cols:
            if col not in metrics:
                continue
            m = metrics[col]
            acc = float(m["holdout_acc"])
            idx = lab[0].index(col)
            K = len({r[idx] for r in lab[1:]})
            chance = 1.0 / K if K else 0.0
            bar = CHANCE_MULTIPLE * chance
            ratio = (acc / chance) if chance > 0 else 0.0

            if n_holdout < MIN_HOLDOUT_N:
                verdict = "INSUFFICIENT-N"
            elif acc >= bar:
                verdict = "SIGNAL"
            else:
                verdict = "NO-SIGNAL"

            sets.append({
                "set": col,
                "fixture": name,
                "n_holdout": n_holdout,
                "K": K,
                "accuracy": round(acc, 4),
                "chance": round(chance, 4),
                "bar": round(bar, 4),
                "ratio": round(ratio, 3),
                "ece_calibrated": float(m["ece_calibrated"]),
                "temperature": float(m["temperature"]),
                "verdict": verdict,
                # §43: verified HELD OUT (see this file's header). Stated per
                # row so no consumer of the JSON misreads it either way.
                "accuracy_is_held_out": True,
            })

    sets.sort(key=lambda s: (-s["ratio"], s["set"]))
    return {
        "bars": {"chance_multiple": CHANCE_MULTIPLE, "min_holdout_n": MIN_HOLDOUT_N},
        "fitted": sets,
        "unfitted": [{"set": s, "verdict": v, "why": w} for s, v, w in UNFITTED],
        "summary": {
            "n_fitted_sets": len(sets),
            "n_signal": sum(1 for s in sets if s["verdict"] == "SIGNAL"),
            "n_no_signal": sum(1 for s in sets if s["verdict"] == "NO-SIGNAL"),
            "n_insufficient": sum(1 for s in sets if s["verdict"] == "INSUFFICIENT-N"),
            "n_unfitted_sets": len(UNFITTED),
        },
    }


def print_table(a: dict) -> None:
    b = a["bars"]
    print("PER-HEAD SIGNAL AUDIT (D1)")
    print("  bars: accuracy >= %.0fx chance AND n_holdout >= %d"
          % (b["chance_multiple"], b["min_holdout_n"]))
    print("  accuracy IS held out (fixture = the holdout rows only; §43 verified).")
    print()
    hdr = "%-22s %9s %9s %9s %7s  %s" % (
        "set", "n_holdout", "accuracy", "chance", "ratio", "verdict")
    print(hdr)
    print("-" * len(hdr))
    for s in a["fitted"]:
        print("%-22s %9d %9.4f %9.4f %7.3f  %s"
              % (s["set"], s["n_holdout"], s["accuracy"],
                 s["chance"], s["ratio"], s["verdict"]))
    print()
    print("UNFITTED (no weights exist — reported, not omitted)")
    for s in a["unfitted"]:
        print("  %-22s %-18s %s" % (s["set"], s["verdict"], s["why"]))
    sm = a["summary"]
    print()
    print("SUMMARY: %d fitted (%d SIGNAL, %d NO-SIGNAL, %d INSUFFICIENT-N), "
          "%d unfitted"
          % (sm["n_fitted_sets"], sm["n_signal"], sm["n_no_signal"],
             sm["n_insufficient"], sm["n_unfitted_sets"]))


def print_markdown(a: dict) -> None:
    b = a["bars"]
    print("### 4.5 Per-head signal audit — D1 (publish, don't assume)")
    print()
    print("Bars reused from DECISION 1: **accuracy >= %.0fx chance AND "
          "n_holdout >= %d**. `chance = 1/K`. "
          "`n_holdout` is the meta's own `n` — the fixture holds the holdout "
          "rows only (§43 verified), so these are held-out figures."
          % (b["chance_multiple"], b["min_holdout_n"]))
    print()
    print("| set | n_holdout | accuracy | chance | ratio | verdict |")
    print("| --- | ---: | ---: | ---: | ---: | --- |")
    for s in a["fitted"]:
        print("| `%s` | %d | %.4f | %.4f | %.3fx | **%s** |"
              % (s["set"], s["n_holdout"], s["accuracy"], s["chance"],
                 s["ratio"], s["verdict"]))
    print()
    print("Unfitted sets — **no weights exist**, so no verdict is possible. "
          "Listed so absence is not read as success:")
    print()
    print("| set | verdict | why |")
    print("| --- | --- | --- |")
    for s in a["unfitted"]:
        print("| `%s` | %s | %s |" % (s["set"], s["verdict"], s["why"]))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="D1 — per-head signal audit from committed calibration fixtures")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--markdown", action="store_true",
                    help="markdown table for docs/CALIBRATION.md")
    args = ap.parse_args()

    a = audit()
    if args.json:
        print(json.dumps(a, indent=2, sort_keys=False))
    elif args.markdown:
        print_markdown(a)
    else:
        print_table(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
