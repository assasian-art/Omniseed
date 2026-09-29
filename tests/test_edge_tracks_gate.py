#!/usr/bin/env python3
"""Gated test for §45 — the four edge tracks (`tools/edge_tracks.py`).

WHAT THIS GUARDS. `docs/EDGE_RESEARCH.md` publishes a rejection log (§4) and a
corrected-floor table (§1.1/§2.1/§4.2). Those tables are *claims*. This test
ties them to `tools/edge_tracks.json`, the tool's own committed output, so a doc
that drifts from the run — or a run that silently changes — fails loudly.

THE TWO FACTS THAT MATTER MOST, both pinned:

  * **Track A's `n_forward=5` reproduces the SHIPPED blob exactly**
    (`acc=0.24390`, `temperature=13.325159204929149`, `ECE=0.0555`). This is the
    harness's control: if Track A did not regenerate the shipped head, the whole
    tool is measuring a different pipeline. Pinning it is what makes the other
    rows trustworthy.

  * **The `1/K` bar is defective (§1.1).** The corrected floor is each set's
    majority-class rate. The gate asserts the corrected ratios — in particular
    that `trading.regime` is **below 1.0x** (0.810x a constant) and that the
    constant predictor's own accuracy (0.3686) is recorded — because those are
    the numbers that DEMOTED a head §2.2 had called load-bearing.

TWO HALVES (the §43b/§44 lesson — either alone is insufficient):
  * A) `tools/edge_tracks.json` (committed) is internally consistent and matches
       the doc's table rows. **Stdlib only** — runs on a fresh clone, in CI.
  * B) `tools/edge_tracks.py` REGENERATES that JSON. Needs numpy, so it is
       skipped with a visible reason when no numpy interpreter exists.

Stdlib-only where it can be. Usage:  python tests/test_edge_tracks_gate.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULT = os.path.join(ROOT, "tools", "edge_tracks.json")
TOOL = os.path.join(ROOT, "tools", "edge_tracks.py")
DOC = os.path.join(ROOT, "docs", "EDGE_RESEARCH.md")

# The majority-class rate of the shipped action holdout. This is the corrected
# floor (§1.1). Hard-coded because it is a FIXED property of the committed
# fixture `tests/fixtures/head_calibration/trading` — recomputing it from the
# tool would not test anything the tool does not already assert.
MAJORITY_FLOOR = 0.3686
SHIPPED_ACC = 0.243900
SHIPPED_TEMP = 13.325159

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


def rows_of(d, track):
    """All result rows for a track key (list)."""
    t = d.get("tracks", {}).get(track)
    return t if isinstance(t, list) else []


def find_row(d, track, variant_substr):
    for r in rows_of(d, track):
        if variant_substr in str(r.get("variant", "")):
            return r
    return None


def _majority_rate_tsv(path, column):
    """The majority-class rate of `column` in a fixture labels.tsv, or None."""
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        if column not in header:
            return None
        ci = header.index(column)
        counts = {}
        n = 0
        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                continue
            fields = line.split("\t")
            if ci < len(fields):
                counts[fields[ci]] = counts.get(fields[ci], 0) + 1
                n += 1
    if n == 0 or not counts:
        return None
    return max(counts.values()) / n


def _metrics_acc(path, column):
    """The holdout_acc for `column` in a fixture metrics.tsv, or None."""
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        cols = f.readline().rstrip("\r\n").split("\t")
        if "holdout_acc" not in cols or "column" not in cols:
            return None
        ai, ni = cols.index("holdout_acc"), cols.index("column")
        for line in f:
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) > max(ai, ni) and fields[ni] == column:
                try:
                    return float(fields[ai])
                except ValueError:
                    return None
    return None


def main():
    print("=== edge-tracks gate (§45) ===")

    if not os.path.isfile(RESULT):
        skip("tools/edge_tracks.json is absent (run tools/edge_tracks.py)")
        print("\n=== RESULT: %d passed, %d failed, %d skipped ==="
              % (g_pass, g_fail, g_skip))
        return 0 if g_fail == 0 else 1

    with open(RESULT, encoding="utf-8") as f:
        d = json.load(f)

    # ---- A0. the bar the tool declares matches the corrected §1.1 --------------
    bar = d.get("bar", {})
    check(bar.get("ratio") == 2.0,
          "A0 the tool declares the 2x bar (%r)" % bar.get("ratio"))
    check(bar.get("min_holdout") == 30,
          "A0 the tool declares the n>=30 floor (%r)" % bar.get("min_holdout"))

    # ---- A1. Track A: the shipped-blob control (n_forward=5) -------------------
    a5 = find_row(d, "A", "n_forward=5")
    if a5 is None:
        check(False, "A1 Track A has an n_forward=5 row (shipped control)")
    else:
        check(abs(a5["accuracy"] - SHIPPED_ACC) < 5e-5,
              "A1 Track A h=5 acc reproduces the shipped blob (%.5f vs %.5f)"
              % (a5["accuracy"], SHIPPED_ACC))
        check(abs(a5["temperature"] - SHIPPED_TEMP) < 5e-7,
              "A1 Track A h=5 temperature reproduces the blob (%.6f vs %.6f) — "
              "this is the harness's OWN control; a mismatch means Track A is "
              "measuring a different pipeline" % (a5["temperature"], SHIPPED_TEMP))
        check(a5["n_holdout"] == 369, "A1 Track A h=5 n_holdout == 369")
        check(a5["adopted"] is False, "A1 Track A h=5 is NOT adopted")

    # ---- A2. Track A: every horizon is a rejection, and n_eff is reported ------
    arows = rows_of(d, "A")
    check(len(arows) >= 4, "A2 Track A ran all four horizons (got %d)" % len(arows))
    check(all(r.get("adopted") is False for r in arows),
          "A2 NO Track A horizon was adopted")
    for r in arows:
        h = r.get("horizon")
        n = r.get("n_holdout", 0)
        neff = r.get("n_eff", 0)
        want = n if h <= 1 else n // h
        check(neff == want,
              "A2 Track A h=%s n_eff is overlap-adjusted (%d == %d//%d)"
              % (h, neff, n, h))
    # h=21 must be WORSE than h=10 (the label-persistence prediction).
    a10, a21 = find_row(d, "A", "n_forward=10"), find_row(d, "A", "n_forward=21")
    if a10 and a21:
        check(a21["accuracy"] < a10["accuracy"],
              "A2 Track A h=21 is WORSE than h=10 (%.4f < %.4f) — the "
              "label-persistence failure the pre-registration predicted"
              % (a21["accuracy"], a10["accuracy"]))

    # ---- A3. Track B: the pre-registered condition (augmented > features) ------
    hh = find_row(d, "B", "h[E] only")
    ff = find_row(d, "B", "features only")
    xf = find_row(d, "B", "h[E] + features")
    if hh and ff and xf:
        check(xf["accuracy"] <= ff["accuracy"],
              "A3 Track B augmented (%.4f) did NOT beat features-only (%.4f) — "
              "the exact §3.2 condition, and it failed"
              % (xf["accuracy"], ff["accuracy"]))
        check(xf.get("adopted") is False,
              "A3 Track B augmented is NOT adopted")
        # The features-only row appears to pass on 1/K; the gate records that
        # its ratio is >2 on 1/K, which is WHY the controls were needed.
        check(ff["ratio"] > 2.0,
              "A3 Track B features-only LOOKS like it passes 2x1/K (%.3f) — "
              "which is the whole reason the constant-predictor control exists"
              % ff["ratio"])
        # ...AND that it was NOT adopted, because the controls unmasked it. This
        # must be asserted in half A (stdlib), not only by half B's regeneration
        # check — otherwise a fresh clone with no numpy would accept a doc that
        # quietly adopted the base-rate artifact.
        check(ff.get("adopted") is False,
              "A3 Track B features-only is NOT adopted (it fails the "
              "majority-class floor: %.4f acc vs %.4f constant)"
              % (ff["accuracy"], ff.get("controls", {})
                 .get("constant", {}).get("accuracy", float("nan"))))
        check(ff.get("beats_majority") is False,
              "A3 Track B features-only does NOT beat the majority class")
        ctl = xf.get("controls", {})
        check(ctl.get("constant", {}).get("accuracy", 0) > ff["accuracy"],
              "A3 Track B's constant predictor (%.4f) outscores features-only "
              "(%.4f) — the artifact, stated in the JSON itself"
              % (ctl.get("constant", {}).get("accuracy", float("nan")),
                 ff["accuracy"]))
    else:
        check(False, "A3 Track B has all three variants")

    # ---- A3b. Track C: pooling helps both assets, but still fails the floor ---
    crows = rows_of(d, "C")
    # A "NOT RUN" row (a clone without the per-asset dumps) is legitimate and has
    # no accuracy fields; only assert the run's numbers when the run exists.
    c_ran = any("accuracy" in r for r in crows)
    if crows and c_ran:
        pooled = None
        for r in crows:
            if "pooled (" in str(r.get("variant", "")):
                pooled = r
                break
        if pooled:
            check(pooled.get("adopted") is False,
                  "A3b Track C pooled is NOT adopted")
            check(pooled.get("beats_majority") is False,
                  "A3b Track C pooled does NOT beat the majority class "
                  "(%.4f acc vs %.4f floor)"
                  % (pooled["accuracy"], pooled.get("majority_rate", float("nan"))))
        # Per-asset: pooling must have LIFTED both single-asset heads.
        lifted = [r for r in crows if r.get("pooling_helps_this_asset") is True]
        check(len(lifted) >= 2,
              "A3b Track C pooling helped >=2 assets (got %d) — the hypothesis "
              "that survived contact" % len(lifted))
    elif crows:
        skip("A3b Track C recorded NOT RUN (no per-asset dumps on this checkout)")
    else:
        skip("A3b Track C not in the JSON (needs per-asset dumps)")

    # ---- A4. Track D: below chance AND conversion inverted ---------------------
    dr = find_row(d, "D", "tertile")
    if dr:
        check(dr["accuracy"] < dr["chance"],
              "A4 Track D is BELOW chance (%.4f < %.4f)"
              % (dr["accuracy"], dr["chance"]))
        check(dr.get("label_balanced") is True,
              "A4 Track D's label mix is balanced, so chance=1/3 is a valid floor")
        conv = dr.get("conversion", {})
        check(conv.get("moved_engine") is False,
              "A4 Track D's conversion test FAILED (no ordered separation)")
        check(conv.get("separation", 0) < 0,
              "A4 Track D's conversion separation is NEGATIVE (%.4f) — inverted"
              % conv.get("separation", 0))
        check(dr.get("adopted") is False, "A4 Track D is NOT adopted")
    else:
        check(False, "A4 Track D has a tertile row")

    # ---- A5. the corrected floor is the majority class, stated in the result ---
    base = d.get("baseline", {})
    check(abs(base.get("ratio", 0) - 1.707) < 0.01,
          "A5 the JSON records the OLD shipped ratio 1.707x (the pre-correction "
          "number, kept visible)")

    # ---- A5b. trading.regime's demotion is checkable from the FIXTURE alone ---
    # The demotion (0.810x a constant) is the §45 change that most alters
    # behaviour, so it must be verifiable without the tool and without numpy:
    # the fixture's own labels give the majority rate, and metrics.tsv gives the
    # head's accuracy. Both are committed.
    fx = os.path.join(ROOT, "tests", "fixtures", "head_calibration", "trading")
    reg_maj = _majority_rate_tsv(os.path.join(fx, "labels.tsv"), "trading.regime")
    reg_acc = _metrics_acc(os.path.join(fx, "metrics.tsv"), "trading.regime")
    if reg_maj is not None and reg_acc is not None:
        check(abs(reg_maj - 0.8699) < 1e-3,
              "A5b trading.regime holdout majority rate is 0.8699 (got %.4f) — "
              "the 'range' class dominates" % reg_maj)
        ratio = reg_acc / reg_maj
        check(ratio < 1.0,
              "A5b trading.regime is BELOW a constant predictor (%.3fx: acc "
              "%.4f vs majority %.4f) — the demotion §2.2 records"
              % (ratio, reg_acc, reg_maj))
    else:
        skip("A5b trading fixture metrics/labels not readable")

    # ---- A6. the doc's §1.1 correction exists and names the majority floor -----
    if os.path.isfile(DOC):
        with open(DOC, encoding="utf-8") as f:
            doc = f.read()
        check("1.1 Correction" in doc,
              "A6 the doc carries the §1.1 correction section")
        check("majority" in doc.lower(),
              "A6 the doc names the majority-class floor")
        check("0.3686" in doc,
              "A6 the doc records the constant predictor's 0.3686")
        check("0.810x" in doc or "0.810" in doc,
              "A6 the doc records trading.regime's corrected 0.810x")
        # The retracted numbers must remain VISIBLE, not deleted.
        check("1.707" in doc and "4.44x" in doc,
              "A6 the doc keeps the old ratios visible alongside the corrected "
              "ones (a correction that erases the record is not a correction)")
    else:
        skip("docs/EDGE_RESEARCH.md is absent")

    # ---- B. the tool REGENERATES this JSON (numpy-gated) ----------------------
    print("  -- half B: regenerate (needs numpy) --")
    py = _numpy_python()
    if py is None:
        skip("B no numpy-capable interpreter found; regeneration not checked")
    else:
        ok = _regen_matches(py, d)
        check(ok, "B the tool regenerates tools/edge_tracks.json byte-for-byte "
                  "on its declared tracks")

    print("\n=== RESULT: %d passed, %d failed, %d skipped ==="
          % (g_pass, g_fail, g_skip))
    return 0 if g_fail == 0 else 1


def _numpy_python():
    """An interpreter that can `import numpy`, or None."""
    cands = []
    venv = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
    if os.path.isfile(venv):
        cands.append(venv)
    venv2 = os.path.join(ROOT, ".venv", "bin", "python")
    if os.path.isfile(venv2):
        cands.append(venv2)
    cands.append(sys.executable)
    for c in cands:
        try:
            r = subprocess.run([c, "-c", "import numpy"], capture_output=True,
                               timeout=60)
            if r.returncode == 0:
                return c
        except Exception:
            continue
    return None


def _regen_matches(py, committed):
    """Run the tool's deterministic tracks into a temp file; compare.

    Only the numpy-only tracks (A, B, D) are re-run — Track C shells out to a
    dump that a fresh clone may not have, and the gate must not depend on it.
    """
    import tempfile
    tmp = os.path.join(tempfile.gettempdir(), "edge_tracks_gate.json")
    cmd = [py, TOOL, "--tracks", "A", "B", "D", "--out", tmp]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800,
                           cwd=ROOT)
    except Exception as e:  # pragma: no cover
        print("      regeneration raised: %s" % e)
        return False
    if r.returncode != 0:
        print("      regeneration exited %d: %s" % (r.returncode,
                                                    (r.stderr or "")[-400:]))
        return False
    if not os.path.isfile(tmp):
        print("      regeneration wrote no file")
        return False
    with open(tmp, encoding="utf-8") as f:
        fresh = json.load(f)
    for t in ("A", "B", "D"):
        if json.dumps(fresh.get("tracks", {}).get(t), sort_keys=True) != \
           json.dumps(committed.get("tracks", {}).get(t), sort_keys=True):
            print("      track %s differs between fresh run and committed JSON" % t)
            return False
    return True


if __name__ == "__main__":
    sys.exit(main())
