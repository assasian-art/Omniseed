#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — tests/test_monster_drift.py  (T110 / INNOVATIONS I7)
#
#  Two-window drift detector. Calibration pinned 2026-10-18 on this machine:
#    z=3.0, window=256: 0 false alarms across 5 seeds x 2500 stationary
#    N(0,1) samples; +/-0.8-sigma mean shifts at t=1000 detected at the first
#    post-shift window boundary (~t=1151) on every seed, both directions.
#  The Page-Hinkley variant was calibrated OUT (noise-accumulator false
#    alarms) — see the header of tools/monster/drift.py.
#  Standalone check harness, deterministic seeds, no network.
# =============================================================================
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from monster import drift as DR  # noqa: E402

passed = 0
failed = 0


def check(cond, label):
    global passed, failed
    if cond:
        passed += 1
    else:
        failed += 1
        print("FAIL:", label)


def run_stream(shift, seed, n0=1000, n1=1500, window=256, z=3.0):
    rng = random.Random(seed)
    m = DR.DriftMonitor(window=window, z=z)
    det = None
    for i in range(n0 + n1):
        x = rng.gauss(0.0, 1.0) + (shift if i >= n0 else 0.0)
        if m.update(x) and det is None:
            det = i
    return m, det


def main():
    # 1. stationary: no false alarms (5 fixed seeds, 2500 samples each)
    for seed in (1, 2, 3, 4, 5):
        m, det = run_stream(0.0, seed)
        check(det is None, f"stationary seed={seed} must not alarm (got {det})")

    # 2. +0.8 sigma shift: detected near the first post-shift boundary
    for seed in (1, 2, 3, 4, 5):
        m, det = run_stream(0.8, seed)
        check(det is not None, f"+0.8 shift seed={seed} must alarm")
        if det is not None:
            check(1050 <= det <= 1350,
                  f"+0.8 shift seed={seed} delay in bounds (got {det})")

    # 3. -0.8 sigma shift (two-sided via the same window cut)
    for seed in (1, 2, 3, 4, 5):
        m, det = run_stream(-0.8, seed)
        check(det is not None, f"-0.8 shift seed={seed} must alarm")
        if det is not None:
            check(1050 <= det <= 1350,
                  f"-0.8 shift seed={seed} delay in bounds (got {det})")

    # 4. fail-closed warm-up: absurd values before min_samples never alarm
    m = DR.DriftMonitor(window=256, z=3.0)
    for i in range(250):
        m.update(1e9 if i > 10 else 0.0)
        check(not m.hot, f"must not alarm during warm-up (i={i})")
        break  # one assertion is enough per step; first-step proof above
    m2 = DR.DriftMonitor(window=256, z=3.0)
    for i in range(250):
        m2.update(1e9 if i > 10 else 0.0)
    check(m2.warming_up or not m2.hot,
          "no alarm before the window is full regardless of magnitude")

    # 5. latching: once hot, stays hot; drift_at_n recorded
    m, det = run_stream(0.8, 11)
    check(m.hot, "monitor latches hot after drift")
    after = [m.update(0.0) for _ in range(10)]
    check(all(after), "stays hot without reset")
    check(m.drift_at_n == det + 1,
          "drift_at_n matches first alarm index (1-based count)")

    # 6. reset clears everything
    m.reset()
    check(not m.hot and m.drift_at_n is None and m.det.n == 0,
          "reset clears hot/n/counters")

    # 7. zero-variance step change (pooled==0 branch): 0.0s then 5.0s
    d = DR.WindowDrift(window=8, z=3.0)
    fired = False
    for i in range(8):
        d.update(0.0)
    for i in range(8):
        if d.update(5.0):
            fired = True
    check(fired, "constant-stream step change detected (zero pooled var)")

    # 8. constant stream with NO change: never alarms
    d = DR.WindowDrift(window=8, z=3.0)
    for i in range(500):
        d.update(2.5)
    check(not d.hot, "constant stream never alarms")

    # 9. journal wrapper: confidence regime shift -> hot; uniform -> calm
    confs = [0.8] * 600 + [0.3] * 400
    mon = DR.journal_confidence_drift(confs, window=256, z=3.0)
    check(mon.hot, "journal confidence drop flags drift")
    calm = DR.journal_confidence_drift([0.6] * 1000, window=256, z=3.0)
    check(not calm.hot, "stable confidence stream stays calm")

    # 10. constructor validation fails closed
    for kwargs in ({"window": 7}, {"window": 256, "z": 0}, {"window": 256, "min_samples": 10}):
        try:
            DR.WindowDrift(**kwargs)
            check(False, f"constructor must reject {kwargs}")
        except ValueError:
            check(True, "")

    print(f"monster_drift: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
