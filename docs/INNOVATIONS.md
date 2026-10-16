# OmniSeed — Innovations Log

_New capabilities invented (not just assembled) in this repo, one row each,
with the honest status. A prototype that failed its bar is an entry too —
the rejections are the valuable part. Last updated: §54 (2026-10-16)._

## Landed (built, tested, pinned)

| # | Innovation | Where | Why it is not just plumbing |
|---|---|---|---|
| I1 | **BLOCK-ONLY factory clamp** (§50) | `tools/monster/factory.py` | An eligibility filter CAN violate block-only (dropping the opposite signal raises \|conviction\|). The clamp falls back to the unfiltered base when the kept verdict would flip sign or exceed it — an invariant that needed code, not a policy sentence. |
| I2 | **Symmetric time-lag lead-lag scan** (§52) | `CorrelationMatrix.lag_scan` | Lag-0 correlation cannot tell leader from follower. The scan is symmetric (negative lag names the responder as leader) AND anchors BOTH windows at/before the evaluation bar — the first cut read the driver past `i` for negative lags; the reversed-pair test caught the look-ahead. |
| I3 | **Fail-closed venue comparison** (§52) | `funding.cross_exchange` | Three failure modes encoded as behavior: unattributable quotes produce nothing, one venue is not a spread, and a wide spread whose best leg is itself blocked is *information, never action*. |
| I4 | **Candlestick zoo voice, parity-safe** (§53) | `tools/monster/chart_reader.py` | Patterns as one low-weight StrategySignal behind the same vetoes — deliberately NOT in `all_signals` (the 10-signal oracle order is pinned at 1e-9 by parity), and the breakout reason reports bars *actually scanned*, not the nominal lookback. |
| I5 | **Nightly recalibration with a keep-old gate** (§54) | `tools/nightly_refit.py` | The journal cannot re-fit softmax weights (no h[E] inputs), so it re-fits what it CAN support — a monotone confidence→accuracy map — accepted only on chronological-holdout Brier improvement, corpus-fingerprinted so a stale map is visible. Silence (held rows) is never scored. |
| I6 | **RUN_SERIAL as a correctness fix** (§52) | CMakeLists (`edge_tracks_gate`) | The gate's 1800 s subprocess timeout exceeded its 1200 s ctest budget: a starved run could never fail fast. Serializing the CPU-bound numpy regen turned a flaky 47th test into a deterministic pass. |

## Prototyped and REJECTED (kept on purpose)

| # | Idea | Why rejected | Where the math lives |
|---|---|---|---|
| R1 | Direction prediction from h[E] (6 probe families: linear, +hand features, pooled, rank targets, balanced teacher, 64-unit MLP) | Every family below the corrected 2×-majority bar; the program is CLOSED, not paused | `docs/EDGE_RESEARCH.md` §6.10 |
| R2 | Balanced action teacher (§45/46) | Best ratio 1.008× (z=0.11) — indistinguishable from noise | `docs/EDGE_RESEARCH.md` §6 |
| R3 | N-gram speculative decoding | RWKV-7 is strictly recurrent: m draft tokens cost m sequential forwards — no win possible without a parallel-scan verify kernel | `PROJECT_STATE.md` §3c |

## Open hypotheses (not yet built)

- **Chart patterns as a factory-eligible strategy** — the reader (I4) emits
  information; promoting it to an S-contributing zoo member requires the same
  adoption bar every layer has faced (measured value first, wiring second).
- **Lyapunov/FNN as a regime axis** — the exotics layer (§55) computes them;
  whether they earn a place in the regime ensemble is a measurement question.
- **Calibration maps consumed by the runtime** — nightly_refit writes
  `state/calibration_map.json`; no C++ consumer reads it yet (same honest gap
  shape as FEEDBACK.md's original §9).
