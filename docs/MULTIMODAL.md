# MULTIMODAL — the joint that was missing

Milestone 5. §28 built one backbone and many heads; §30–31 gave it a soul and a
memory; §32 made the heads' confidence real. This milestone connects the
**vision and audio encoders to the head stack** — the thing that was built but
never joined.

> **The one-sentence result.** An image now becomes tokens, those tokens become
> `h[E]`, and a head reads that `h[E]`. The chain is proven end to end through
> the real 243 MB backbone in `tests/test_multimodal.cpp` C1. **The heads it
> lights up are still UNFITTED** — see §5.

---

## 1. What was actually missing

An audit before building changed the plan. `src/vision/` (659 lines),
`src/audio/` (1,626 lines, including a 1,082-line Whisper-tiny) and
`src/runtime/token_bus.cpp` were **all real, all compiled, all in
`OMNISEED_CORE_SOURCES`**. `TokenBus` even documented its own job as:

> *"Any-to-Token Bus: fuses text, vision and audio token streams into ONE
> unified sequence for the RWKV core."*

And it had **zero tests**. Nothing anywhere turned an image into the `h[E]` the
heads read. As in §30 and §31, **the gap was the joint, not the capability.**

### The impedance mismatch

```
VisionEncoder::encode()  ->  Tensor [M, n_embd]     CONTINUOUS floats
FocalCodec::encode()     ->  vector<int32_t>        already token ids
TokenBus::fuse()         ->  wants vector<int32_t>  for EVERY modality
RwkvModel::forward()     ->  takes int32_t token
```

Audio already fits. **Vision does not** — it produces float embeddings, and
`forward()` accepts only a token id. That single mismatch is what
`MultimodalBridge` exists to close.

---

## 2. The bridge

`include/omniseed/multimodal.h` + `src/multimodal.cpp`. Deliberately small: it
re-implements no encoder, no tokenizer and no bus.

| step | call |
|---|---|
| 1 | `set_codebook(model.token_embeddings())` — the model's own `[n_vocab, E]` matrix |
| 2 | `nearest_token(vec)` — quantize one `[E]` embedding to the nearest vocabulary id |
| 3 | `quantize(rows)` — an `[M, E]` block becomes `M` ids |
| 4 | `fuse(text, vision_rows, audio_codes)` — text + vision + audio → one stream |
| 5 | the caller runs the stream through `RwkvModel::forward()` → `h[E]` |

### Why the codebook is the embedding matrix

Nearest-vocabulary quantization is not arbitrary:

- **No training**, so it cannot silently mis-calibrate;
- **no extra RAM** — the weights are already resident, so the R2 `<400 MB` budget
  is untouched (one new public accessor, `RwkvModel::token_embeddings()`, returns
  a non-owning view);
- it lands the modality in **the same discrete space the core was trained on**,
  which is the only space `forward()` accepts.

Two correctness details that are easy to get wrong and are pinned by tests:

- **The early-out needs a flag.** Abandoning a candidate row as soon as its
  partial sum exceeds the best-so-far is what makes a 65,536-row scan
  affordable — but the partial sum then compares as *smaller* than `best_d`, so
  without a `worse` flag the row would be accepted **precisely because it was cut
  short**. (`tests/test_multimodal.cpp` B3.)
- **A NaN must not win.** Every comparison against NaN is false, so a NaN query
  would be "nearest" to the first row examined. The query is checked for
  finiteness before the scan. (B7.)

### ⚠️ What this is NOT

Nearest-embedding quantization is a **lossy, untrained** projection. It is not a
learned modality adapter. It claims only that the emitted ids are the closest
vocabulary items to what the encoder produced — **not** that they are
semantically the right tokens.

The `codebook_stride` config is a real accuracy/speed dial, not a no-op: with
`stride = 2`, a token whose nearest row has an odd index is **unreachable**.
B8 pins that behaviour so it cannot drift into "stride is free".

---

## 3. The chain, end to end

`tests/test_multimodal.cpp` C1, run against the real
`models/rwkv7-0.1B-ternary.gguf`:

```
fused 10 tokens (text 3, vision 4, audio 0)
vision.scene top1 = urban (0.997)   <- head is UNFITTED
```

The 4 vision rows (stand-ins for an encoder's output) became 4 vocabulary ids,
were wrapped in `<|vision|> … </|vision|>`, the fused stream was run through the
backbone, and the resulting `h[E]` was fed to a `vision.scene` classifier head —
which produced a distribution.

**The joint works.** What it does not yet do is produce a *meaningful*
distribution, and the `0.997` is the proof.

---

## 4. Tests

`omniseed_multimodal` — **110 checks, 0 failures, 0 skipped**. Registered
**outside** the `.venv` gate.

- **Part A — `TokenBus`, the first tests this module has ever had** (7 checks):
  BOS-first, exact `encode(text, false)` body, vision/audio wrapping, **empty
  payloads emit no markers**, segment order, budget refusal, exact-budget
  acceptance, stats accounting.
- **Part B — `MultimodalBridge`, ungated** (a minimal tokenizer and a synthetic
  basis codebook, so it runs in CI): loud failure when vision arrives with no
  codebook, width-mismatch refusal, exact-row lookup, block quantization,
  partial-row refusal, NaN handling, the stride dial, and summary honesty.
- **Part C — end to end through the real backbone**, gated on
  `models/rwkv7-0.1B-ternary.gguf`, with a printed SKIP reason when absent.

Two design rules the bridge follows, both tested:

> **A modality that was supplied is never silently dropped.** Passing an image
> with no codebook **fails with a reason**; it does not fall back to answering
> from text alone. A caller that got a text-only answer would never learn the
> image was ignored.

> **A failed fusion is not an empty one.** `ok == false` always carries an
> `error`, and `ids` is empty.

---

## 5. ⚠️ Honest gaps (do not overclaim)

- **The vision and audio heads are UNFITTED.** `vision.scene`,
  `vision.anomaly`, `audio.wake`, `audio.emotion`, `audio.speaker` all have
  registered label sets and **no fitted weights**. They report the base rate
  (or, worse, a confident-looking seeded number — the `0.997` above). They need
  exactly the treatment §32 gave the language and trading heads: collect
  `h[E]`, fit offline, calibrate, store a `.bin`. **This milestone built the
  pipe, not the water.**
- **The `0.997` is the whole argument for §32, restated.** A seeded projection
  printed 99.7% confidence on a head with no knowledge. Confidence without
  calibration is not a signal.
- **The codebook is untrained and lossy** (§2). No learned adapter exists.
- **The audio path is not exercised end to end.** `FocalCodec`/`WhisperTiny`
  produce ids that fit the bus directly, but no test runs a real wav through the
  whole chain — the encoders have their own tests, the joint for audio is
  untested at the real-weights level.
- **`VisionEncoder` itself is not called in C1.** The encoder needs
  `models/vision-proj.gguf` and has its own suite; C1 deliberately isolates the
  join by feeding synthetic `[M, E]` rows.
- **No CLI demo yet.** `demo-multimodal` does not exist; the bridge is reachable
  from C++ and from this test only.
- **`trading.regime` and friends are unaffected** — this milestone touches no
  risk path. The live-money gates are untouched.

---

## 6. Reproducing

```bash
./build/bin/omniseed_multimodal.exe        # 110 checks
ctest --test-dir build -C Release -R omniseed_multimodal --output-on-failure
```

Part A/B need nothing. Part C needs `models/rwkv7-0.1B-ternary.gguf` and skips
with a printed reason without it.
