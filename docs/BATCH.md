# Batched head evaluation

**Milestone 8 (§36).** Phase 2.1 of the mandate. Files:
`include/omniseed/core/batch_gemm.h`, `src/core/batch_gemm.cpp`,
`src/core/batch_gemm_avx2.cpp`, `tests/test_heads_batch.cpp`, plus
`*_batch()` methods on the three heads.

## 1. What is batched, and what is not

**Batched: the head readout.** A head is `[L, E] × [E] → [L]` followed by a
squash. For `B` hidden states that is a single GEMM, `[B, E] × [L, E]ᵀ + bias[L]`,
and it is embarrassingly parallel.

**NOT batched: the backbone.** This tree's RWKV-7 is the **scalar recurrent**
form, so `h[E]` for bar *n+1* needs the state from bar *n*. "One forward pass for
1000 signals" is therefore **not** what this delivers, and claiming it would be
false. What it delivers is one GEMM for 1000 *readouts*, after 1000 necessarily
sequential forwards.

That distinction matters for where the win is. On the live path the backbone
dominates and batching buys nothing. On the **backtest path** the forwards are
already paid for and cached to `hidden.f32`, and the readout — with its per-call
scratch resize, `order` vector, `top_k` vector and two clock reads, repeated once
per bar — was the remaining cost. That is what this removes.

## 2. Two kernels, two different contracts

This is the part to read before trusting a batched result.

| kernel | accumulation | vs the per-row path |
|---|---|---|
| **Scalar** | one running accumulator, `e = 0…E-1`, in order | **bit-identical** |
| **Simd** | eight AVX2 lanes, reduced at the end | within tolerance — **not** identical |

Floating-point addition is not associative, so a lane-reduced sum cannot
reproduce a serial one. The tests therefore assert **equality** (`==`) on the
Scalar kernel and a **tolerance** on the Simd kernel, and `A5` additionally
asserts that the two *do* differ — if a future change made them identical, either
the SIMD path stopped being taken or it was quietly rewritten as a serial sum,
and the test says so.

`BatchKernel::Auto` is the default and resolves to Simd where available. So the
**default path is the fast, inexact one**. Exactness requires asking for it:

```cpp
head.set_batch_kernel(BatchKernel::Scalar);   // then batch == per-row, exactly
```

The default path is held to a weaker but more meaningful contract: **the answer
does not change.** Measured on 369 real held-out bars and 200 synthetic ones:
**0** top-1 label mismatches, **0** action mismatches, **0** routing mismatches.

## 3. Measured

`tests/test_heads_batch.cpp`, Release, E = 768, this box. Reproduce with
`./build/bin/omniseed_heads_batch.exe`.

| measurement | value |
|---|---|
| Simd vs Scalar, worst absolute difference (B=64, E=768, L=7) | **1.07e-06** |
| Simd vs Scalar, worst relative difference | **1.07e-04** |
| `classify_batch`, 369 real rows | 9.5 µs/row → **1.7 µs/row** |
| `decide_batch`, 369 real rows | 9.9 µs/row → **2.0 µs/row** |
| **whole stack, 1000 signals** (classify + decide + score) | **25.27 ms → 4.50 ms** |
| **speedup** | **5.62×** |

The per-row column is the *batched* call with the Scalar kernel, which is the
honest apples-to-apples comparison: same code path, same arithmetic, only the
per-call overhead removed. So the 5.6× is roughly 1.8× from removing per-call
overhead and 3.1× from AVX2 — not from any change in what is computed.

## 4. API

```cpp
#include "omniseed/core/batch_gemm.h"

std::vector<ClassificationResult> out;
BatchStats stats;
if (head.classify_batch(H, B, "trading.regime", 4, out, &stats))
    log("%d rows, %.3f us/row, kernel %s", stats.rows, stats.us_per_row,
        batch_kernel_name(stats.used));
```

`H` is `[B, E]` row-major — the exact layout `tools/dump_hidden.cpp` writes to
`hidden.f32`, so a backtest can point a head straight at a dump with no
repacking.

`BatchStats` reports rows, matvec count, total and per-row microseconds, the
kernel requested, the kernel used, and whether an explicit Simd request fell
back. `BatchGemmPlan` (on the raw `batch_gemm` call) additionally reports the
MAC count.

## 5. Fail-closed policy

Every batch call returns `bool` and **clears `out`** on failure. Never a partial
batch: a caller that ignores the return value must not act on half an answer.

| condition | result |
|---|---|
| head not ready | `false`, `out` empty |
| `H == nullptr` with `B > 0` | `false`, `out` empty |
| `B <= 0` | `false`, `out` empty |
| unknown label set / index | `false`, `out` empty |
| `Tensor` whose shape is not `[B, E]` | `false`, `out` empty |
| `batch_gemm` with `B`, `E` or `L` == 0 | no-op, no error — an empty batch is legitimate at the kernel level |

The last row is deliberate and differs from the head level: the kernel treats an
empty batch as "nothing to do", while a head treats it as "I cannot answer that",
because a head that returned `true` with zero results would be indistinguishable
from one that silently dropped rows.

## 6. Honest gaps

* **The backbone is still sequential.** Batching the RWKV recurrence is a
  different and much larger project (it needs the linear-attention form, not the
  recurrent one). Not started.
* **`us_per_row` on a batched result is an average, not that row's own time.**
  `ClassificationResult::us`, `DecisionResult::ms` and `ScoreResult::us` are all
  filled with the batch average. A per-row clock read is exactly the overhead the
  batch removes, so this is the honest thing to report — but it is not a
  per-row measurement and must not be read as one.
* **Timing is not gated.** `E2` asserts only that batching is not pathologically
  slower; absolute microsecond gates are flaky on a loaded CI box. The printed
  numbers are the deliverable, the assertion is a tripwire.
* **The AVX2 fallback path is untested on this machine.** `A6` SKIPs here
  because the box has AVX2. It will run on a non-AVX2 build.
* **Nothing calls the batched path yet.** No backtest loop, no CLI, no server.
  The capability and its evidence exist; the consumer does not. The natural
  consumer is the paper-trading replay in `src/trading/simulate.cpp`, which
  currently re-derives its readouts bar by bar.
* **The heads still read a seeded placeholder.** Batching changes how fast a
  head answers, not whether its answer means anything — `trained()` is still the
  gate. See `docs/CALIBRATION.md`.
