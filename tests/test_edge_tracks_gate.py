#!/usr/bin/env python3
"""Gated test for §45/§46 — the edge tracks (`tools/edge_tracks.py`).

WHAT THIS GUARDS. `docs/EDGE_RESEARCH.md` publishes a rejection log (§4) and a
corrected-floor table (§1.1/§1.2/§2.1/§4.2). Those tables are *claims*. This test
ties them to `tools/edge_tracks.json` and `tools/edge_tracks_balanced.json`, the
tool's own committed outputs, so a doc that drifts from the run — or a run that
silently changes — fails loudly.

THE FACTS THAT MATTER MOST, all pinned:

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

  * **`acc > majority` is NOT the bar; `acc >= 2 x majority` is (§46).** The
    gate asserts the tool APPLIES its own bar to every row (`adopted` iff
    `n>=30 and acc >= 2*floor`), and it pins the counterexample that forced the
    rule: the balanced teacher's features-only head beats the constant by 0.0027
    (z=0.11) — noise that a bare `>` would have adopted.

  * **The balanced teacher did not rescue the head (§46).** Track E — a
    PERFECTLY balanced 3-class directional target, so `chance = 1/3` exactly —
    sits BELOW chance (0.2951 = 0.885x). That is the decisive statement: the
    failure is not the skew, it is the absence of directional signal in h[E].

  * **Track F closes the program (§47).** The last escape hatch — "the signal
    exists but is not LINEARLY decodable" — is tested once with a 64-unit
    one-hidden-layer probe and FAILS (F1 0.3197 vs a 0.3716 floor = 0.860x). The
    gate does not merely assert the number: it asserts the four CONTROLS, each in
    the direction it must go — the leak control clears (1.0000), the non-linear
    capacity control clears (0.6967) while a LINEAR probe on the SAME target does
    not (0.5519), and the shuffled-label control does not (the bar is not
    vacuous). Without those, a "no-signal" verdict would be unfalsifiable.

TWO HALVES (the §43b/§44 lesson — either alone is insufficient):
  * A) the committed JSONs are internally consistent and match the doc's table
       rows. **Stdlib only** — runs on a fresh clone, in CI.
  * B) `tools/edge_tracks.py` REGENERATES those JSONs. Needs numpy, so it is
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
RESULT_BAL = os.path.join(ROOT, "tools", "edge_tracks_balanced.json")
TOOL = os.path.join(ROOT, "tools", "edge_tracks.py")
DOC = os.path.join(ROOT, "docs", "EDGE_RESEARCH.md")

# The majority-class rate of the shipped action holdout. This is the corrected
# floor (§1.1). Hard-coded because it is a FIXED property of the committed
# fixture `tests/fixtures/head_calibration/trading` — recomputing it from the
# tool would not test anything the tool does not already assert.
MAJORITY_FLOOR = 0.3686
SHIPPED_ACC = 0.243900
SHIPPED_TEMP = 13.325159
# The balanced teacher's h=5 majority floor (§46). Lower than the shipped
# teacher's 0.3686 because the quantile directional branch equalises
# BUY/SELL/HOLD; it is NOT 1/7, because the four regime-conditional actions stay
# rare on an 85%-`range` tape.
BALANCED_FLOOR_H5 = 0.3198
BAR_RATIO = 2.0
MIN_HOLDOUT = 30

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

    # =====================================================================
    #  §46 — the bar is `acc >= 2 x majority`, and the balanced teacher
    # =====================================================================

    # ---- A7. the tool APPLIES its own bar to every row it emits ---------------
    # The §46 rule lives in `_verdict`; this asserts the JSON cannot contain a
    # row whose `adopted` flag disagrees with it. A tool that publishes the right
    # rule but adopts on a different one is the §41 defect class.
    check(d.get("bar", {}).get("rule", "").find("majority") >= 0,
          "A7 the tool declares the 2x-majority rule (%r)"
          % d.get("bar", {}).get("rule"))
    bad = []
    for t, rows in d.get("tracks", {}).items():
        for r in rows:
            if "accuracy" not in r or "floor" not in r:
                continue          # e.g. a Track C "NOT RUN" row
            want = (r.get("n_holdout", 0) >= MIN_HOLDOUT and
                    r["accuracy"] >= BAR_RATIO * r["floor"])
            if bool(r.get("adopted")) != bool(want):
                bad.append((t, r.get("variant"), r.get("adopted"), want))
    check(not bad,
          "A7 every row's `adopted` == (n>=30 and acc >= 2 x floor) "
          "(%d violation(s): %r)" % (len(bad), bad[:3]))

    # ---- A8. the balanced teacher LOWERED the floor but did not rescue the head
    if not os.path.isfile(RESULT_BAL):
        skip("A8 tools/edge_tracks_balanced.json absent (run with "
             "--teacher balanced)")
    else:
        with open(RESULT_BAL, encoding="utf-8") as f:
            db = json.load(f)
        check(db.get("teacher") == "balanced",
              "A8 the balanced artifact records teacher=balanced (%r)"
              % db.get("teacher"))
        b5 = find_row(db, "A", "n_forward=5")
        if b5 is None:
            check(False, "A8 balanced artifact has an n_forward=5 row")
        else:
            check(abs(b5.get("floor", 0) - BALANCED_FLOOR_H5) < 5e-4,
                  "A8 balanced h=5 floor is %.4f (got %r) — lower than the "
                  "shipped 0.3686, so the rebalance is real"
                  % (BALANCED_FLOOR_H5, b5.get("floor")))
            check(b5["floor"] < MAJORITY_FLOOR,
                  "A8 balanced floor (%.4f) < shipped floor (%.4f)"
                  % (b5["floor"], MAJORITY_FLOOR))
            check(b5.get("adopted") is False,
                  "A8 the balanced head at h=5 is STILL not adopted — the "
                  "failure was not a skew artifact")
        # CANDIDATE rows only: Track F's control rows also carry `accuracy` but
        # deliberately carry `control_floor`/`cleared_bar` instead of
        # `floor`/`adopted`, so requiring the `adopted` key is what keeps a
        # control from being read as an adopted head here.
        b_all = [r for rows in db.get("tracks", {}).values() for r in rows
                 if "accuracy" in r and "adopted" in r]
        check(len(b_all) > 0 and all(r.get("adopted") is False for r in b_all),
              "A8 NO balanced-teacher candidate row was adopted (%d rows)"
              % len(b_all))

        # ---- A9. Track E — the decisive control: a PERFECTLY balanced 3-class
        #          directional target, below chance.
        e = find_row(db, "E", "tertile")
        if e is None:
            check(False, "A9 balanced artifact has the Track E control row")
        else:
            check(e.get("label_balanced") is True,
                  "A9 Track E's label mix is ~1/3 by construction")
            check(abs(e.get("chance", 0) - 1.0 / 3) < 1e-9,
                  "A9 Track E's by-construction chance is exactly 1/3 (%r)"
                  % e.get("chance"))
            check(e["accuracy"] < e["chance"],
                  "A9 Track E is BELOW its by-construction 1/3 floor (%.4f < "
                  "%.4f) — h[E] does not linearly encode the forward direction"
                  % (e["accuracy"], e["chance"]))
            check(e["accuracy"] < e.get("floor", 1.0),
                  "A9 Track E is ALSO below the chronological holdout's own "
                  "majority rate (%.4f < %.4f)"
                  % (e["accuracy"], e.get("floor", float("nan"))))
            check(e.get("macro_f1", 1.0) < e["chance"],
                  "A9 Track E's macro-F1 (%.4f) is also below 1/3 (%.4f)"
                  % (e.get("macro_f1", float("nan")), e["chance"]))
            check(e.get("adopted") is False, "A9 Track E is NOT adopted")

        # ---- A10. the counterexample that forced §46: a bare `>` bar would have
        #           adopted features-only on a 0.11-sigma margin.
        fb = find_row(db, "B", "features only")
        if fb is None:
            check(False, "A10 balanced artifact has the features-only row")
        else:
            check(fb.get("beats_majority") is True,
                  "A10 balanced features-only DOES beat the majority class "
                  "(%.4f > %.4f) — the trap §46 closes"
                  % (fb["accuracy"], fb.get("majority_rate", float("nan"))))
            check(abs(fb.get("constant_z", 99)) < 1.0,
                  "A10 ...but only by %.2f standard errors — noise, not signal"
                  % fb.get("constant_z", float("nan")))
            check(fb.get("adopted") is False,
                  "A10 balanced features-only is NOT adopted (2x-floor bar)")

    # ---- A11. the doc's §1.2/§46 correction is present -------------------------
    if os.path.isfile(DOC):
        with open(DOC, encoding="utf-8") as f:
            doc = f.read()
        check("1.2" in doc and "2 x majority" in doc.lower().replace("×", "x"),
              "A11 the doc carries the §1.2 bar correction (2 x majority)")
        check("0.3198" in doc,
              "A11 the doc records the balanced floor 0.3198")
        check("0.885" in doc or "0.885x" in doc,
              "A11 the doc records Track E's 0.885x (below chance)")
    else:
        skip("docs/EDGE_RESEARCH.md is absent")

    # =====================================================================
    #  §47 — Track F: the LAST direction probe (NON-LINEAR). The program CLOSES.
    # =====================================================================

    # ---- A12. the decision is recorded, and it is CLOSE ----------------------
    dp = d.get("direction_program", {})
    check(dp.get("closed") is True,
          "A12 the tool records `direction_program.closed = true` (%r)"
          % dp.get("closed"))
    check("CLOSE" in str(dp.get("verdict", "")),
          "A12 the direction program verdict is CLOSE (%r)" % dp.get("verdict"))
    check(dp.get("adopted") is False,
          "A12 no direction candidate was adopted")
    check(dp.get("controls_valid") is True,
          "A12 the Track F controls are VALID — a probe that failed its own "
          "controls could not pronounce on h[E] at all")

    # ---- A13. Track F's rows: both candidates BELOW their own floors ----------
    frows = rows_of(d, "F")
    check(len(frows) == 6, "A13 Track F emitted all six rows (got %d)" % len(frows))
    f_cand = [r for r in frows if r.get("role") == "candidate"]
    check(len(f_cand) == 2,
          "A13 Track F has exactly two candidates (got %d)" % len(f_cand))
    for r in f_cand:
        check(r.get("adopted") is False,
              "A13 %s is NOT adopted" % r.get("variant"))
        check(r["accuracy"] < r["floor"],
              "A13 %s is BELOW its own majority floor (%.4f < %.4f)"
              % (r.get("variant"), r["accuracy"], r["floor"]))
    f1 = find_row(d, "F", "F1 directional")
    f2 = find_row(d, "F", "F2 balanced")
    if f1 is None or f2 is None:
        check(False, "A13 Track F has the F1 and F2 candidate rows")
    else:
        check(abs(f1["accuracy"] - 0.3197) < 5e-5,
              "A13 F1 acc is 0.3197 (got %r) — the pre-registered number"
              % f1["accuracy"])
        check(abs(f1["floor"] - 0.3716) < 5e-5,
              "A13 F1 floor is 0.3716 (got %r)" % f1["floor"])
        check(abs(f2["accuracy"] - 0.2114) < 5e-5,
              "A13 F2 acc is 0.2114 (got %r)" % f2["accuracy"])
        # The probe's shape is part of the pre-registration; a row that does not
        # declare it is not the probe that was registered.
        pr = f1.get("probe", {})
        check(pr.get("hidden") == 64, "A13 the probe declares 64 hidden units (%r)"
              % pr.get("hidden"))
        check(pr.get("pca_components") == 32,
              "A13 the probe declares its PCA width (32) (%r)"
              % pr.get("pca_components"))
        check(pr.get("epochs") == 2000,
              "A13 the probe declares its epoch budget (2000) (%r)"
              % pr.get("epochs"))
        check(pr.get("optimiser") == "adam",
              "A13 the probe declares Adam (the optimiser fit_softmax uses) (%r)"
              % pr.get("optimiser"))

    # ---- A14. the four controls, each in the direction it MUST go -------------
    byc = {r.get("control"): r for r in frows if r.get("role") == "control"}
    check(set(byc) == {"label-leak", "capacity-nonlinear", "capacity-linear",
                       "shuffled-labels"},
          "A14 Track F carries all four controls (%r)" % sorted(byc))
    if len(byc) == 4:
        leak = byc["label-leak"]
        check(leak["clears_2x_floor"] is True,
              "A14 the leak control CLEARS the bar (%.4f) — the instrument "
              "recovers a signal that IS present" % leak["accuracy"])
        check(abs(leak["accuracy"] - 1.0) < 1e-9,
              "A14 the leak control is at ceiling (%.4f)" % leak["accuracy"])
        cm = byc["capacity-nonlinear"]
        check(cm["clears_2x_floor"] is True,
              "A14 the non-linear capacity control CLEARS the bar (%.4f >= %.4f) "
              "— the probe fits and generalises a non-linear function"
              % (cm["accuracy"], 2 * cm["control_floor"]))
        cl = byc["capacity-linear"]
        check(cl["clears_2x_floor"] is False,
              "A14 the LINEAR probe on the SAME target does NOT clear (%.4f) — "
              "which is what makes the mlp row evidence of NON-linearity"
              % cl["accuracy"])
        check(cm["accuracy"] > cl["accuracy"],
              "A14 the mlp beats the linear probe on the non-linear target "
              "(%.4f > %.4f)" % (cm["accuracy"], cl["accuracy"]))
        ctl = byc["shuffled-labels"]
        check(ctl["clears_2x_floor"] is False,
              "A14 the shuffled-label control does NOT clear (%.4f) — the bar is "
              "not vacuous" % ctl["accuracy"])

    # ---- A15. the doc records Track F and the closure ------------------------
    if os.path.isfile(DOC):
        with open(DOC, encoding="utf-8") as f:
            doc = f.read()
        check("6.9" in doc and "Track F" in doc,
              "A15 the doc carries §6.9 (Track F)")
        check("6.10" in doc and "CLOSED" in doc.upper(),
              "A15 the doc carries §6.10 (the program is CLOSED)")
        check("0.3197" in doc and "0.6967" in doc and "0.5519" in doc,
              "A15 the doc records Track F's numbers (F1 0.3197, capacity mlp "
              "0.6967, capacity linear 0.5519)")
        check("rule-based engines" in doc.lower(),
              "A15 the doc states plainly where the trading edge actually lives")
    else:
        skip("docs/EDGE_RESEARCH.md is absent")

    # ---- B. the tool REGENERATES this JSON (numpy-gated) ----------------------
    print("  -- half B: regenerate (needs numpy) --")
    py = _numpy_python()
    if py is None:
        skip("B no numpy-capable interpreter found; regeneration not checked")
    else:
        ok = _regen_matches(py, d, [])
        check(ok, "B the tool regenerates tools/edge_tracks.json byte-for-byte "
                  "on its declared tracks")
        # Track F is regenerated on its own: it is the expensive one, and the
        # A/B/D/E call above would otherwise pay for it twice.
        okf = _regen_matches(py, d, [], ["F"])
        check(okf, "B the tool regenerates Track F byte-for-byte (the §47 "
                   "non-linear probe is deterministic)")
        if os.path.isfile(RESULT_BAL):
            with open(RESULT_BAL, encoding="utf-8") as f:
                db = json.load(f)
            okb = _regen_matches(py, db, ["--teacher", "balanced"])
            check(okb, "B the tool regenerates tools/edge_tracks_balanced.json "
                       "byte-for-byte on its declared tracks")
        else:
            skip("B balanced artifact absent; balanced regeneration not checked")

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


def _regen_matches(py, committed, extra, tracks=("A", "B", "D", "E")):
    """Run the tool's deterministic tracks into a temp file; compare.

    Only the numpy-only tracks are re-run — Track C shells out to a
    dump that a fresh clone may not have, and the gate must not depend on it.
    `extra` carries the teacher/class-weight flags for the balanced artifact;
    `tracks` selects which tracks to regenerate (Track F is called separately
    because it is the expensive one).
    """
    import tempfile
    tag = ("_".join(extra).replace("--", "") or "v1") + "_" + "".join(tracks)
    tmp = os.path.join(tempfile.gettempdir(), "edge_tracks_gate_%s.json" % tag)
    cmd = [py, TOOL, "--tracks"] + list(tracks) + ["--out", tmp] + list(extra)
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
    for t in tracks:
        f_rows = fresh.get("tracks", {}).get(t)
        c_rows = committed.get("tracks", {}).get(t)
        if f_rows is None and c_rows is None:
            continue
        if json.dumps(f_rows, sort_keys=True) != json.dumps(c_rows,
                                                            sort_keys=True):
            print("      track %s differs between fresh run and committed JSON" % t)
            return False
    return True


if __name__ == "__main__":
    sys.exit(main())
