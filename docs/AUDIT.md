# AUDIT — what the mandate asked for, versus what is actually in the tree

This document exists because the "MASTER EXHAUSTIVE BUILD MANDATE" listed roughly
sixty capabilities as missing, and **most of them were already built.** Building
them again would have been waste; the real work was joining them. Everything
below is a measurement, not an impression.

> **Read this before starting any "revival" work.** The default assumption —
> "the brief describes a gap, so the feature must not exist" — was wrong for
> almost every item.

---

## 1. Method

Two checks, run before any code was written:

```bash
# 1. Orphans: source files that compile into nothing
for f in $(find src tools -name '*.cpp'); do
  grep -q "$(basename $f)" CMakeLists.txt || echo "ORPHAN: $f"
done

# 2. Coverage: is a module exercised by any test?
grep -rlw <ClassName> tests/*.cpp
```

**⚠️ The second check must use the CLASS NAME, not the filename.** A filename
never appears in a test that uses the class, so a filename grep reports
"untested" for code that is thoroughly tested. That mistake was made once in
this project and produced a claim of *ten* untested modules when the true number
was **four**.

---

## 2. Orphans

**None in `src/`.** Every one of the 59 source/header files under `src/` is
referenced by `CMakeLists.txt` and compiled into `omniseed_core`. The only
unreferenced `.cpp` files are five throwaway probes under `tools/`
(`dbg_mel`, `dbg_sensory`, `kernel_bench`, `perf_profile`, `simd_probe`) — they
are not shipped code.

---

## 3. What already existed

Verified real (not stubs), compiled, and in `OMNISEED_CORE_SOURCES`:

| area | what is there |
|---|---|
| vision (659 lines) | `VisionEncoder` (MobileNetV4), `UniCompress`, `PointerPerception`, `SpatialContextMapper`, `MotionTracker`, `GestureRecognizer`, `DynamicResolution` |
| audio (1,626 lines) | `WhisperTiny` (1,082 lines, mel + encoder + greedy decoder), `FocalCodec`, `SoundEventDetector`, `WakeWordDetector` **with `enroll()`**, `AudioSceneClassifier`, `AudioAnomalyDetector` |
| runtime | `TokenBus`, `EmotionalState`, `SensoryFingerprint`, `Swarm`, `Introspection`, `CloudBridge` |
| memory | `MemoryCrystals`, `PrefixCache`, `StreamingLlm` |
| agent | `AgentLoop`, `SubAgents`, `FlashSkills`, `AgentIntel`, `SelfImprovement`, `GrammarDecoder`, `ToolRegistry`, `ComputeThrottle` |
| core | `Uncertainty` + `EntropyMonitor` |

The mandate's Phase 3.3 ("voice enroll/verify for owner identity") is
`WakeWordDetector::enroll()`. Phase 3.5 ("multi-modal fusion") is
`TokenBus::fuse()` — which documented its own job as exactly that fusion and
**had zero tests**.

---

## 4. Module coverage (by class name)

| module / class | covered by |
|---|---|
| `VisionEncoder`, `UniCompress` | `test_sides.cpp` (gated on `models/vision-proj.gguf`) |
| `WhisperTiny`, `SoundEventDetector`, `WakeWordDetector` | `test_platform.cpp` |
| `StreamingLlm` | `test_platform.cpp` |
| `ToolRegistry`, `AgentLoop` | `test_platform.cpp`, others |
| `TokenBus` | `test_multimodal.cpp` (added in §33 — **the first tests it ever had**) |
| `GrammarDecoder`, `ComputeThrottle`, `SelfImprovement`, `FocalCodec` | `test_agent_modules.cpp` (added in §34) |
| `DecisionHead`, `ClassificationHead`, `ScoringHead`, `HeadRouter` | `test_heads.cpp`, `test_router.cpp`, `test_calibration.cpp` |
| `StreamingDecision`, `StreamEvent` | `test_streaming_decision.cpp` (added in §37) |
| `DecisionJournal`, `FeedbackHook`, `JournalEntry`, `JournalStats` | `test_feedback_hook.cpp` (added in §38) |

After §34, **every class in `src/` is referenced by at least one test** — and the
rows added in §36–§38 are the ones that keep that true as new classes arrive.
`UserFeedbackLoop` is covered by `test_agent_modules.cpp` and, for its new
persistence, by `test_feedback_hook.cpp` E7.

---

## 5. Defects found by the coverage work

Writing tests for code that had none found four real defects. Three were fixed;
two were pinned as measured behaviour (see `tests/test_agent_modules.cpp`).

### 5.1 `SelfImprovement::save()` / `load()` never agreed — the cache never persisted

`save()` wrote a `uint32_t` magic `0x54524953`; `load()` compared against the
**string** `"SRIT"`. Little-endian, `0x54524953` is the bytes `'S','I','R','T'`
— so the comparison **always failed**. `load()` rejected every file `save()`
wrote.

The failure was silent: an unreadable cache is indistinguishable from an empty
one. `src/cli/main.cpp` calls `imp.load("./state/improve.bin")` and simply
always got `false`, so the self-improvement trace cache **has never persisted
across runs**. Fixed with one shared `kMagic` constant; pinned by C9.

### 5.2 `SelfImprovement::load()` read past the end of the mapping

Every length in the format (`klen`, `ns`, `sl`) is read straight out of the
file and was used unchecked, so a truncated or corrupt `state/improve.bin`
walked off the end of the mmap. Fixed with a `need(n)` bounds check applied
before every read, plus a cap on the trace count so a 12-byte file claiming 4
billion traces cannot drive a 4-billion-element `reserve()`. A failed load now
leaves the previous traces intact. Pinned by C7.

### 5.3 `SelfImprovement::task_key()` did not trim leading whitespace

The contract is "normalize to a stable key". It started with `last_ws = false`,
so a leading whitespace run was emitted as a leading space and only the
**trailing** end was trimmed. `"  read a file"` keyed to `" read a file"` while
`"read a file"` keyed to `"read a file"` — two entries for one task, and the
exact-match replay lookup missed whenever the caller's string had leading
whitespace. Fixed by initialising `last_ws = true`. Pinned by C1.

### 5.4 `FocalCodec::encode()` discards 3 of every 4 quantization bits (PINNED)

The comment says *"quantize each band to 4 bits, pack 6 bands per 24-bit code"*.
The code computes a 4-bit bucket `q` and then keeps **only its low bit**:

```cpp
code |= (q & 1u) << (b % 32);   // 3 of every 4 bits thrown away
```

Measured on a 440 Hz tone, frame 0: the low-bit-only code is **15712260** while
the documented 4-bit-per-band code would be **16777113**. The codec is a 24-bit
*sign pattern* (is this band above or below the midpoint of the `[-8, 2]` log
range?), folded mod `codebook_size` — not the quantizer it documents.

**Not fixed**: changing it changes every code the codec emits, and a fitted
audio head would have to be retrained against the new alphabet. Pinned by D6.

### 5.5 `ComputeThrottle::classify()` matches Fast keywords by substring (PINNED)

The Fast list is matched with `find()`, so two-character entries fire inside
unrelated words: `"hi"` inside `"which"`, `"no"` inside `"nothing"`, `"ok"`
inside `"book"`. Measured: `classify("which stock should i buy today")` returns
**Fast** — a real trading question gets the shallowest compute budget (48 new
tokens, **0** thinking tokens, 1 tool turn) instead of Balanced.

**Not fixed**: the thresholds are a tuned heuristic and word-boundary matching
would change every classification. Pinned by B6.

### 5.6 `GrammarDecoder::accepts()` allows content after a complete object (PINNED)

The contract is "would appending `text` keep the stream a valid JSON prefix?".
After a complete top-level object the answer must be no. It answers yes:
`accepts("x")` returns true on a decoder that already holds `{}`, and `feed("x")`
then reports `complete()` on the invalid `"{}x"`. `accepts()` never consults
`complete_`, and `feed()`'s recompute re-derives `complete_` from the buffer's
first character alone.

**Not fixed**: `accepts()` is the gate in a constrained-generation loop, and
tightening it changes what the model is allowed to emit. Pinned by A8.

---

## 6. What is genuinely absent (verified)

> ⚠️ **Two rows below have since been closed.** The table is a §34 snapshot; the
> **Update** markers record what changed. Do not read a row without its marker.

| mandate item | status |
|---|---|
| Phase 2.7 — epistemic vs aleatoric **separation** | **CLOSED in §35.** Was: absent — `core/uncertainty.h` gives Shannon entropy, normalized entropy, and the top-1/top-2 margin, a *total* signal only, and the strings `epistemic`/`aleatoric` appeared **nowhere** in `include/` or `src/`. Now `core/uncertainty_split.h` separates them, measured against the real held-out heads. See `docs/UNCERTAINTY.md`. |
| Phase 3 — the vision/audio → `h[E]` **joint** | **Was absent; built in §33** (`MultimodalBridge`). |
| Fitted **vision/audio heads** | **ABSENT.** `vision.scene`, `vision.anomaly`, `audio.wake`, `audio.emotion`, `audio.speaker` have label sets and no weights. Measured: an unfitted `vision.scene` head printed **0.997** confidence. Still the largest verified absence. |
| Phase 5 — dream consolidation (Milestone 3) | **Not wired.** `MemoryCrystals::decay_memories` and `SelfImprovement::dream()` exist; no scheduled pass writes `state/dream_log.json`. |
| Phase 11 — exotic (quantum, fractal, chaos, game theory, RL, federated, differential privacy, homomorphic, ZK) | **ABSENT.** Not started. |

Items **not** verified in this audit, and therefore not claimed either way:
batch processing, streaming decisions, feedback hooks, hierarchical routing
level 2, and cross-head ensemble voting. Note that `ensemble`/`voting` appear
only in **trading** files (`regime_engine`, `router`, `sniper`), which is a
within-domain ensemble — **not** the cross-head vote the mandate describes.
**Of those five: batch (§36), streaming (§37) and feedback hooks (§38) are now
closed. Hierarchical routing level 2 and the cross-head vote are still open.**

**Update (§36):** **batch processing is now DONE at the head level** —
`classify_batch` / `decide_batch` / `score_batch` over `batch_gemm`, measured at
**5.62×** for the whole head stack over 1000 signals. The backbone is still
sequential (this tree's RWKV-7 is the scalar recurrent form). See `docs/BATCH.md`.
The remaining four items on that line are still unverified/absent.

**Update (§37):** **streaming decisions are now DONE at the head level** —
`StreamingDecision` (`include/omniseed/streaming_decision.h`) is an incremental
protocol over the decision head with debounce, hysteresis and release, and
`push_batch` drives it from the §36 batched readout. It is a *filter over the
head's outputs*, not an incremental head, and it does **not** make the backbone
incremental (it already is). Nothing consumes it yet beyond `demo-stream`. See
`docs/STREAMING.md`. **Feedback hooks, hierarchical routing level 2, and
cross-head ensemble voting remain unverified/absent.**

**Update (§34, §35):** two further corrections to the row above about
`ensemble`/`voting`. §34 found the `SelfImprovement` trace cache never persisted
(a `uint32` magic compared against a string), so the closest thing to a feedback
store was silently dead — now fixed. §35 built
`UncertaintyDecomposition::from_ensemble()`, which is the *operator* a cross-head
vote needs and is tested, but no head ensemble exists to feed it.

**Update (§38):** **feedback hooks are now DONE at the record level** —
`DecisionJournal` + `FeedbackHook` record one row per valid `StreamEvent`, join
it to an outcome, and persist the pairs; `UserFeedbackLoop` gained the
persistence it never had. The convention that matters is that an unresolved
prediction is **not** a wrong one (`hit_rate()` is -1.0, never 0.0), and that
`Rejected` / `Expired` rows are excluded from every accuracy number. Measured on
the 369 real held-out bars with the fitted head: `hit_rate` 0.2439, ECE 0.0555,
both matching `metrics.tsv` exactly. **What is still absent:** nothing *consumes*
the hook, and nothing re-fits from the journal. See `docs/FEEDBACK.md`.

**Update (§39):** the §33 row's "fitted vision/audio heads" absence is unchanged,
but the **trading and language heads were never the gap** — §32 fitted and
calibrated them and they have been committed ever since. The §37 stream A
"constant action" finding was measured on a **seeded placeholder the demo built
inline**, because no demo path loaded the committed blob. With the blob loaded
the fitted head emits **6 distinct actions** and flips **158 times** on the same
tape. **The recurring gap was, again, the joint.** `load_decision_head()` now
loads it and says so out loud; pinned by `test_feedback_hook.cpp` F6.

---

## 7. Honest summary

- The tree is **far more complete than the briefs assume**; the recurring gap is
  the **joint** between components that all work in isolation.
- Four modules had **no tests at all**; writing those tests immediately found
  **four real defects**, one of which (5.1) silently disabled a feature for the
  module's entire life.
- The largest verified absences are the **fitted vision/audio heads** and the
  **dream-consolidation pass**. The epistemic/aleatoric decomposition (§35),
  head-level batch processing (§36), streaming decisions (§37) and feedback
  hooks (§38) have since been closed. Hierarchical routing level 2 and the
  cross-head vote remain open.
- **§39 is the cleanest instance of the recurring shape so far.** The mandate
  asked for the trading and language heads to be fitted; §32 had already fitted
  and committed them, and §35's audit had always read those blobs. The
  "constant action" that motivated the request was a **seeded placeholder** the
  §37 demo built inline, because no demo path loaded the blob. Loading it
  changed 1 distinct action to **6**. The finding was real, correctly reported,
  and about the wrong subject — the gap was the **joint**, not the head.
