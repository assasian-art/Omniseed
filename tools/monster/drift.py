#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — tools/monster/drift.py   (T110 / INNOVATIONS I7)
#
#  Two-window change-point detection for the continual-learning loop.
#
#  WHY THIS EXISTS. nightly_refit re-fits the confidence->accuracy map from
#  the feedback journal, but a map fitted ACROSS a regime change is worse
#  than no map: the isotonic fit averages two worlds that no longer share a
#  calibration. The journal cannot re-fit weights (no h[E] inputs — see
#  FEEDBACK.md §9), but it CAN tell whether the stream it is fitting is
#  still the same stream. That is a change-point test.
#
#  THE MATH (Bifet & Gavalda, "Learning from Time-Changing Data with Adaptive
#  Windowing", SDM 2007 — the ADWIN cutting mechanism, simplified). Keep the
#  last `window` observations; split them into an OLDER half A and a NEWER
#  half B; declare drift when
#      |mean(B) - mean(A)| > z * pooled_std * sqrt(2 / half)
#  i.e. the two halves differ by more than z combined standard errors. The
#  check runs on NON-OVERLAPPING halves (every `half` new samples), which
#  keeps the false-alarm budget finite instead of per-sample (a per-sample
#  check at z=3 compounds ~0.27%/check into near-certain false alarms over a
#  few thousand samples — measured, and the reason this is not that).
#
#  WHY NOT PAGE-HINKLEY. The textbook streaming PH recursion was implemented
#  first and CALIBRATED OUT: the +delta-per-step accumulator makes the
#  statistic drift on pure noise, and 0.8-sigma shifts were detected at the
#  SAME index as the no-shift control (the alarm was noise, not signal).
#  Honest note kept here so nobody re-derives the same trap.
#
#  FAIL-CLOSED STANCE (house rule). Before `window` observations the detector
#  NEVER reports drift (insufficient evidence is not evidence of stability).
#  A reported drift is INFORMATION, never a trade — callers gate maintenance
#  decisions (keep-old maps), nothing else.
# =============================================================================
"""Two-window (ADWIN-style cut) drift detector — stdlib only, deterministic."""

import math


class WindowDrift:
    """Two-window mean-shift detector over a bounded recent window."""

    def __init__(self, window=256, z=3.0, min_samples=None):
        if window < 8 or window % 2 != 0:
            raise ValueError("window must be an even integer >= 8")
        if z <= 0:
            raise ValueError("z must be positive")
        self.window = int(window)
        self.half = self.window // 2
        self.z = float(z)
        self.min_samples = int(min_samples) if min_samples else self.window
        if self.min_samples < self.window:
            raise ValueError("min_samples must be >= window")
        self.reset()

    def reset(self):
        self.buf = []
        self.n = 0
        self.hot = False
        self.drift_at_n = None
        self._checked = 0      # samples at the last check boundary

    def _check(self):
        """Compare older vs newer half of the full window."""
        half = self.half
        a = self.buf[:half]
        b = self.buf[half:]
        ma = sum(a) / half
        mb = sum(b) / half
        va = sum((x - ma) ** 2 for x in a) / (half - 1)
        vb = sum((x - mb) ** 2 for x in b) / (half - 1)
        pooled = math.sqrt((va + vb) / 2.0)
        delta = abs(mb - ma)
        if pooled == 0.0:
            # Zero variance: ANY difference between halves is a real change
            # (constant streams shift by jumping to a new constant).
            return delta > 0.0
        return delta > self.z * pooled * math.sqrt(2.0 / half)

    def update(self, x):
        """Feed one observation; True iff drift is (now) declared."""
        if self.hot:
            return True  # latched: caller decides whether to reset
        x = float(x)
        self.n += 1
        self.buf.append(x)
        if len(self.buf) > self.window:
            self.buf.pop(0)
        if (self.n >= self.min_samples and
                self.n - self._checked >= self.half and
                len(self.buf) == self.window):
            self._checked = self.n
            if self._check():
                self.hot = True
                self.drift_at_n = self.n
                return True
        return False

    @property
    def warming_up(self):
        return self.n < self.min_samples


class DriftMonitor:
    """Front door for callers: bounded window, latched hot flag, reset()."""

    def __init__(self, window=256, z=3.0, min_samples=None):
        self.det = WindowDrift(window=window, z=z, min_samples=min_samples)

    def update(self, x):
        return self.det.update(x)

    def reset(self):
        self.det.reset()

    @property
    def hot(self):
        return self.det.hot

    @property
    def drift_at_n(self):
        return self.det.drift_at_n

    @property
    def warming_up(self):
        return self.det.warming_up


def journal_confidence_drift(confidences, window=256, z=3.0):
    """Run the monitor over an ordered confidence stream (chronological!).

    Returns the monitor. `hot` means: the confidence stream changed mean —
    a calibration fitted ACROSS that change would average two regimes.
    Never call this on shuffled rows; the order IS the evidence.
    """
    mon = DriftMonitor(window=window, z=z)
    for c in confidences:
        mon.update(float(c))
    return mon
