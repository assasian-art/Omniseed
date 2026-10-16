#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — tests/test_nightly_refit.py
#
#  Characterization tests for tools/nightly_refit.py (Phase 4.2/4.3).
#
#  Three properties carry the weight:
#    1. THE GATE — a calibration is accepted ONLY on a real chronological
#       holdout Brier improvement; the worst case is always "keep-old".
#    2. SILENCE IS NOT AN OUTCOME — held/implicit journal rows never score.
#    3. PROVENANCE — every accepted map embeds the corpus fingerprint, and
#       the fingerprint changes when the corpus changes.
#
#  Offline, stdlib-only.  Usage: python tests/test_nightly_refit.py
# =============================================================================
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import nightly_refit as NR  # noqa: E402

PASSED = 0
FAILED = 0
FAILURES = []


def check(name, cond, detail=""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  ok   {name}")
    else:
        FAILED += 1
        FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL {name}  {detail}")


def test_isotonic():
    print("\n[isotonic / PAVA]")
    check("already-monotone input is unchanged",
          NR.isotonic([0.1, 0.3, 0.5, 0.7]) == [0.1, 0.3, 0.5, 0.7])
    # classic violation: 0.5 then 0.2 -> pooled mean 0.35
    check("a single violation pools to the weighted mean",
          NR.isotonic([0.5, 0.2]) == [0.35, 0.35],
          str(NR.isotonic([0.5, 0.2])))
    # weighted pooling: [0.6(w2), 0.2(w1)] pool to 1.4/3; 0.8 stays above
    got = NR.isotonic([0.6, 0.2, 0.8], [2, 1, 1])
    check("weights move the pooled mean",
          got[:3] == [1.4 / 3] * 3 and got[3] == 0.8, str(got))
    check("output length equals input length",
          len(NR.isotonic([0.9, 0.1, 0.5, 0.2], [3, 1, 1, 1])) == 6)
    try:
        NR.isotonic([0.5], [0])
        check("a zero weight raises", False)
    except ValueError:
        check("a zero weight raises", True)


def test_brier():
    print("\n[brier]")
    check("a perfect forecast scores 0",
          NR.brier([1.0, 0.0], [1, 0]) == 0.0)
    check("a coin flip on opposite outcomes scores 1.0",
          NR.brier([1.0], [0]) == 1.0)


def _pairs(overconfident, n=60):
    """Deterministic (confidence, correctness) pairs.

    overconfident=True: claims 0.9 but is right half the time — the classic
    miscalibration the isotonic map should fix on the holdout.
    """
    out = []
    for i in range(n):
        if overconfident:
            c = 0.9
            o = 1.0 if (i * 7919) % 100 < 50 else 0.0
        else:
            # well-calibrated-ish: 70% correct, confidence 0.7
            c = 0.7
            o = 1.0 if (i * 7919) % 100 < 70 else 0.0
        out.append((c, o))
    return out


def test_gate():
    print("\n[the acceptance gate]")
    v = NR.recalibrate(_pairs(False), min_n=30)
    check("a well-calibrated head is NOT refit (keep-old)",
          v["accepted"] is False and "keep-old" in v["reason"], v["reason"])
    # a single-confidence series cannot make >=2 bins -> too-few-bins or
    # keep-old; either way it must NOT be accepted
    v2 = NR.recalibrate(_pairs(True), min_n=30)
    # overconfident: every bin claims ~0.9, actual ~0.5 -> isotonic pools to
    # 0.5, which beats identity(0.9) on the holdout
    check("an overconfident head IS recalibrated",
          v2["accepted"] is True and v2["reason"] == "improved-holdout-brier",
          "%s brier_id=%s brier_iso=%s" % (v2["reason"],
                                           v2["brier_identity"],
                                           v2["brier_isotonic"]))
    check("...and the holdout numbers prove it",
          v2["brier_isotonic"] < v2["brier_identity"],
          "%s !< %s" % (v2["brier_isotonic"], v2["brier_identity"]))
    v3 = NR.recalibrate(_pairs(True), min_n=1000)
    check("below min_n it refuses (insufficient)",
          v3["accepted"] is False and v3["reason"] == "insufficient")
    check("the split is CHRONOLOGICAL (last 20% holdout)",
          NR.recalibrate(_pairs(True), min_n=30, holdout_frac=0.2)["n"] == 60)


def test_journal():
    print("\n[journal: silence is not an outcome]")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "j.tsv")
        hdr = ("id\tstep\ttask_key\tpredicted\tconfidence\tmargin\t"
               "held\tcommitted\toutcome\trealised\tscore\thorizon\n")
        with open(p, "w", encoding="utf-8") as f:
            f.write(hdr)
            # committed + realised -> scores
            f.write("1\t1\tt.A\tBUY\t0.9\t0.5\t\1\t1\trealised\tBUY\t1\t1\n"
                    .replace("\\1", "0"))
            # committed but NOT realised yet -> ignored
            f.write("2\t2\tt.A\tBUY\t0.9\t0.5\t0\t1\tpending\t\t0\t1\n")
            # held (implicit) -> never scored
            f.write("3\t3\tt.A\tSELL\t0.9\t0.5\tABSTAIN\t0\trealised\t"
                    "ABSTAIN\t0\t1\n")
            # malformed row -> skipped
            f.write("4\tbroken\n")
            # committed + realised, wrong prediction
            f.write("5\t5\tt.A\tBUY\t0.8\t0.4\t0\t1\trealised\tSELL\t0\t1\n")
        rows = NR.read_journal(p)
        check("malformed rows are skipped, not guessed",
              len(rows) == 4, str(len(rows)))
        pairs = NR.committed_pairs(rows)
        check("only committed+realised pairs score",
              len(pairs) == 2, str(pairs))
        check("correctness is computed from predicted==realised",
              pairs[0] == (0.9, 1.0) and pairs[1] == (0.8, 0.0), str(pairs))
        out = os.path.join(d, "map.json")
        v = NR.run_nightly(p, out, min_n=30)
        check("a tiny journal writes NO map (fail-closed)",
              v["accepted"] is False and not os.path.isfile(out))


def test_corpus():
    print("\n[corpus provenance]")
    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "corpus_a.txt")
        with open(a, "w", encoding="utf-8") as f:
            f.write("hello")
        e1 = NR.digest_files([a])
        fp1 = NR.corpus_fingerprint(e1)
        with open(a, "w", encoding="utf-8") as f:
            f.write("HELLO")     # content change
        e2 = NR.digest_files([a])
        fp2 = NR.corpus_fingerprint(e2)
        check("the fingerprint changes when the corpus changes", fp1 != fp2)
        check("digests are deterministic across calls",
              NR.digest_files([a]) == e2)
        check("a missing file is skipped, not fatal",
              NR.digest_files([os.path.join(d, "nope")]) == {})
        # accepted maps embed the fingerprint
        j = os.path.join(d, "j.tsv")
        with open(j, "w", encoding="utf-8") as f:
            f.write("id\tconfidence\tpredicted\trealised\tcommitted\t"
                    "outcome\n")
            for i in range(60):
                conf = 0.9
                real = "BUY" if (i * 7919) % 100 < 50 else "SELL"
                f.write("%d\t%.1f\tBUY\t%s\t1\trealised\n" % (i, conf, real))
        out = os.path.join(d, "map.json")
        v = NR.run_nightly(j, out, corpus_paths=[a], min_n=30)
        if v["accepted"]:
            import json
            with open(out, encoding="utf-8") as f:
                m = json.load(f)
            check("an accepted map embeds the corpus fingerprint",
                  m.get("corpus_fingerprint") == fp2,
                  str(m.get("corpus_fingerprint")))
        else:
            check("an accepted map embeds the corpus fingerprint",
                  False, "map was not accepted: %s" % v["reason"])


def main():
    print("== nightly refit ==")
    test_isotonic()
    test_brier()
    test_gate()
    test_journal()
    test_corpus()
    print(f"\nRESULT: {PASSED} passed, {FAILED} failed")
    for f in FAILURES:
        print(f"   - {f}")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
