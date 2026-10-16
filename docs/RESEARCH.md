# OmniSeed — Research Notes

_Source tracking for external research that informed build decisions, plus
promising leads. Entries cite what was actually checked; nothing here is
"heard somewhere". Last updated: §54 (2026-10-16)._

## What informed the build (checked, used)

- **Cont-Kukanov-Stoikov (2014) order-flow imbalance** — the OFI signal in
  `tools/monster/strategies.py` implements their exact estimator; the honest
  caveat (their R² is contemporaneous, not predictive) is written into the
  module header.
- **Moreira-Muir (2017) vol-managed exposure** — the vol-target overlay; chose
  the vol-managed form over options-selling because it needs no options venue.
- **Granger (1969) causality** — implemented in the exotics layer as
  *predictive* causality only; the docstring states plainly that it is never
  interventional (no do-calculus claim).
- **Detrended fluctuation analysis (Peng et al. 1994)** — the fractal module's
  H estimation; calibrated against synthetic persistent/anti-persistent/random
  series rather than trusted on real bars.
- **Isotonic regression / PAVA** — the nightly recalibration curve; chosen
  over parametric Platt scaling because the journal's confidence range is
  narrow and a parametric fit would extrapolate outside observed data.

## Promising leads (not built)

- **Free depth venues for the order-book layer** — `signals.py` consumes
  `depth.csv` when supplied; a free venue publishing a real book would move
  the M layer from the discounted proxy to exact CKS OFI in live use. Checked
  providers so far either gate depth behind keys or serve snapshots only.
- **Parallel-scan WKV verify kernel** — would flip speculative decoding from
  refused-with-math to possibly-profitable (PROJECT_STATE §3c R3). No known
  open implementation for RWKV-7 recurrence.
- **Fitted vision.scene via CIFAR-10** — the fetcher exists
  (`tools/get_modality_data.py --fetch`); the previous fetch was abandoned.
  Retried 2026-10-16; blocked on the dataset host. Until images exist the
  head stays INSUFFICIENT-DATA by the audit's own rule.
- **LibriSpeech speaker IDs vs owner enrollment** — `get_modality_data.py`
  documents why a third-party speaker head misses the point (a speaker head
  keys on YOUR voice); enrollment (`omniseed enroll-audio`) remains the
  design. Revisit only if multi-user support becomes a goal.

## Reading list (tracked, not yet mined)

- FinRL / Qlib (RL for trading, factor pipelines) — candidates for cross-
  checking our tabular Q-learning setup once a live paper loop accumulates.
- Voyager (self-improving agent skills) — relevant to the flash-skills pool;
  the safety harness question (tests-before-adopt) is already answered
  differently here: nothing auto-adopts without a measured bar.
