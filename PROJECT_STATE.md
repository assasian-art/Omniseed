GLM-5.3-FLASH

# PROJECT_STATE.md — OmniSeed Build State

> **SINGLE SOURCE OF TRUTH.** After EVERY task/file, the active agent MUST update
> this file: (1) what is complete, (2) exact next steps, (3) unresolved issues.
> If a context limit is reached, the next agent resumes from the "NEXT STEPS"
> section below without re-reading the whole codebase.

- **Project:** OmniSeed — sub-300MB multi-modal micro-LLM agent kernel
- **Root:** `C:\Users\sakim\OneDrive\Desktop\omniseed`
- **Language/Standard:** Pure C++17, zero external dependencies, no Python at runtime
- **Machine toolchain:** Windows 11 + Visual Studio 18 2026 (MSVC 19.50, cl.exe),
  CMake 4.4.3. No g++/MinGW on this box — Linux/MinGW paths are wired but
  untested here.
- **Build:** `build.bat` (auto-detects VS via vswhere) or
  `cmake -S . -B build -G "Visual Studio 18 2026" -A x64 && cmake --build build --config Release`
- **Test:** `build\bin\omniseed_tests.exe` — **165/165 passing** +
  `build\bin\omniseed_real_weights.exe` — **13/13 passing** (zero warnings /W4)
- **Last updated:** PHASE-OMEGA — Grand Unification pass. Read all 3 feature
  registries (z.ai universe / deepseek VOL.I+II / grok grounded), triaged every
  feature into A/B/C buckets, implemented the Bucket-A gold (uncertainty
  quantification, WKV prefix snapshots, Ebbinghaus memory decay, temperature
  annealing, RSS watermark), documented the refusal of n-gram speculative
  decoding on RWKV with the math. 200/200 + 13/13 + 17/17, RSS 155 MB.

---

## 1. AUDIT SUMMARY (this session)

Audited the entire codebase against the 7 research TXT files
(`C:\Users\sakim\Downloads\New folder\*.txt`, especially "Architecting the
Unthinkable" — the 100-feature catalog). Findings and actions:

**Stale/false claims found & fixed:**
- `src/server/main.cpp` never compiled (missing `using namespace omniseed;`) — fixed.
- `Dockerfile`, `render.yaml`, `docs/DEPLOYMENT.md`, `.gitignore` were claimed DONE
  but did not exist — all created.
- Test count claimed 43/43 while many modules had no coverage — extended to 163.

**Entirely missing capabilities, now implemented (all with tests + CLI demos):**
| Capability | Files |
|---|---|
| #3 Sensory Fingerprinting | `include/omniseed/runtime/sensory.h`, `src/runtime/sensory.cpp` |
| #4 Emotional Resonance | `include/omniseed/runtime/emotional.h`, `src/runtime/emotional.cpp` |
| #6 Zero-Shot Skill Synthesis + Flash Skill Pools (#8/#9/#34–36/#49–50) | `include/omniseed/agent/flash_skills.h`, `src/agent/flash_skills.cpp` |
| #7 Collaborative Swarm Protocol (#43–44/#55–58) | `include/omniseed/runtime/swarm.h`, `src/runtime/swarm.cpp` (LoopbackMesh + UDP beacon + CRC32 codec) |

**Extended modules (new files):**
| Area | Files | Spec features |
|---|---|---|
| Audio tasks | `include/omniseed/audio/audio_events.h`, `src/audio/audio_events.cpp` | wake word (#15), sound events (#3), scene (#63), anomaly (#70) |
| Vision tasks | `include/omniseed/vision/vision_tasks.h`, `src/vision/vision_tasks.cpp` | pointer perception (#2), spatial map (#54), tracking (#66), gestures (#61), dynamic res (#69) |
| Agent intel | `include/omniseed/agent/agent_intel.h`, `src/agent/agent_intel.cpp` | intent (#23), dialogue state (#24), confidence (#16), hallucination risk (#17), thinking mode (#6), error recovery (#25), progressive disclosure (#26), feedback loop (#74), Sensory Interrupt System (#72), Knowledge Graph (#38), entity linking (#18) |

**Bugs found and fixed during verification:**
- Swarm `LoopbackMesh::send("")` broadcast dropped messages (no `""` mailbox) —
  now fans out to every joined mailbox except the sender.
- Swarm UDP XOR keystream was cipher-feedback (non-symmetric decode) — fixed to
  position-keyed keystream.
- `KnowledgeGraph::load` read `triples_[size()]` out of bounds (reindex before
  push_back) — reordered.
- `KnowledgeGraph::ingest_text` word-boundary walk looped on multi-word subjects —
  rewritten as a bounded token scan.
- Emotional fusion gate discarded strong valence (conf `wsum/2.0` too low for a
  single channel) — gate now only fires when weak AND emotionally flat; anxious
  classifier thresholds recalibrated; "please" removed from positive lexicon.
- Sensory voice embedding was stat-order-blind (parallel vectors across users) —
  replaced with Goertzel band-energy spectrum + random-Fourier projection;
  same-user 1.00 vs impostor 0.845, gate 0.88.
- Flash-skill store path / `state/` dir creation on Windows — fixed.

---

## 2. COMPLETED (all phases)

### Phase 1 — Skeleton & Portability ✅
`CMakeLists.txt`, `include|src/core/platform.*`, `build.bat`, `build.sh`,
`tests/test_platform.cpp` (now 163 checks), `.gitignore`.

### Phase 2 — Core Engine (RWKV-7 + BitNet 1.58-bit) ✅
`tensor.*` (F32/F16/I8/TERNARY), `bitlinear.*` (2 weights/byte pack, mult-free
forward, QAT STE step), `rwkv.*` (**exact RWKV-7 reference math**, O(1) streaming
state), `gguf_format.h`/`gguf_loader.*` (v2/v3, zero-copy mmap views),
`tokenizer.*` (SentencePiece-compatible + agent special tokens).

### Phase 3 — Multi-Modal Encoders ✅ (structure + live pipeline)
`vision.h`/`mobilenet_v4.cpp` (resize→patches→UniCompress→ternary; weights pending),
`unicompress.cpp` (importance top-k, ≤4x token reduction), `audio.h`/
`whisper_tiny.cpp` (log-mel via Hann+DFT+Slaney mel; decoder weights pending),
`focal_codec.cpp` (24 Goertzel bands → semantic codes; codebook pending),
`token_bus.*` (any-to-token fusion, #1).

### Phase 4 — Agent Intelligence & Memory ✅
`streaming_llm.cpp` (attention sinks + window + retirement, 1M logical tokens, #10),
`memory_crystals.cpp` (salience crystallization, cosine retrieval, persistence),
`grammar_decoder.cpp` (#5), `tool_registry.cpp` (#7, builtins calc/echo/time),
`agent_loop.cpp` (perceive→think→act→observe, #21), `self_improvement.cpp`
(trace replay + `dream()`, #33/#37), `compute_throttle.cpp` (#12).

### Phase 5 — Novel capabilities, perception tasks, intel ✅ (this session)
See AUDIT SUMMARY table above; plus `agent_intel.*`, `audio_events.*`,
`vision_tasks.*`, `flash_skills.*`, `sensory.*`, `emotional.*`, `swarm.*`.

### Phase 6 — Interfaces, deployment, docs ✅
- `src/cli/main.cpp`: `info chat gen ask bench tools selftest dream demo-sensory
  demo-emotional demo-skills demo-swarm demo-memory demo-audio demo-vision`.
- `src/server/main.cpp`: compiles now; HTTP `/health` `/ask` `/gen` behind `OMNISEED_HTTP`.
- `Dockerfile` (multi-stage Alpine, ~15 MB runtime image), `render.yaml`,
  `docs/DEPLOYMENT.md`.
- **`docs/OMNISEED_MASTER_SPEC.md` + `docs/OMNISEED_MASTER_SPEC.html`** — final
  master document: architecture, 7 capabilities, 100-feature mapping table with
  file locations, build/deploy guides, CLI demo catalog with sample outputs.
  (`tools/md2html.py` regenerates the HTML; no PDF tooling on this box — HTML serves.)

---

## 2b. PHASE 7 — REAL WEIGHTS + VALIDATION ✅ (committed: 4a695d2)

**The kernel has a real brain.** `models/rwkv7-0.1B-ternary.gguf` (280 MB,
RWKV-7 "World" 0.1B v2) loads in the pure-C++17 runtime and generates
coherent English: greedy "Hello" → 160-token grammatical reply; sampled
(temp 0.8, top-k 40) → "Hello! Thanks for using my chatbot. Can you provide
me with a few questions about your search?" **Peak RSS 202.8 MB** (budget
300 MB); 12 layers/768 embd/65536 vocab; ~1.4 tok/s single-thread scalar.

| Piece | Detail |
|---|---|
| Converter | `tools/convert_to_omniseed.py` (offline; venv at `.venv/` with torch). Maps `Hakureirm/rwkv7-0.1b-hf` safetensors (`rwkv7.` prefix, LoRA factors are raw `[E,rank]`/`[rank,E]` params applied `x@w1@w2`, stored **transposed** for C++ row-major) → OmniSeed GGUF: per-row int8 linears + fp32 `.scale` tensors, fp16 `token.embd`/`head.weight`, fp32 norms/mixes/biases/`k_k`/`k_a`/`r_k`/`gn`, packed 65,536-piece world vocab + special-id metadata (eos=0). `--check-prompt` runs a full fp32 torch reference forward. |
| Quantization decision | **PTQ ternary b1.58 is too lossy without QAT** — verified in torch fp32 (`tools/quant_sim.py`): ternary model degenerates even with perfect math. Per-row symmetric int8 (1 byte/param, 2× smaller than fp16) reproduces the fp16 oracle near-exactly and stays coherent. LoRA factors currently ride along as int8 too. Ternary returns via QAT (NEXT STEPS). |
| Roundtrip validator | `tools/check_gguf.py` — byte-parses the GGUF exactly like the C++ loader, dequantizes, diffs vs source safetensors: **220/220 tensors exact**. |
| Oracle chain | `tools/oracle_hf.py` (official HF impl → coherent English), `tools/diff_vs_hf.py` (per-block hidden-state diff vs my numpy reference → matched to 1e-5 after fixing the unprefixed `head.weight` lookup). |
| **Critical bug fixed** | `GgufLoader::read_tensor_dir` treated tensor offsets as relative to the **post-metadata** position; the GGUF spec places data after the **tensor directory**. Every tensor read was shifted into directory bytes → garbage/NaN weights. The synthetic test GGUF never caught it because its tests only checked metadata, not bytes. Fixed: `data_start_` = post-directory; bounds checked after. Symptom chain: NaN `sc_r` → NaN logits; traced with env-gated `OMNISEED_DEBUG_NAN=1` instrumentation (kept in `rwkv.cpp`). |
| Sampling (TASK 4) | `RwkvModel::sample_token(logits, temperature, top_k, seed&)` — SplitMix64, deterministic per seed; `AgentLoop::Config{temperature, top_k, seed}`; CLI `--temperature/--top-k/--seed` on `gen`. Greedy default unchanged. |
| FFT (TASK 4) | `include/omniseed/core/fft.h` — radix-2 + **Bluestein chirp-z** for exact N=400 bins 0..200 with M=1024 (O(N log N), same bins as the naive DFT it replaces). Wired into `whisper_tiny.cpp` log-mel. Test: matches naive DFT to 1e-6, identical tone bin. |
| Real-weights test | `tests/test_real_weights.cpp` (13 checks): skips when the model file is absent; asserts config, finite logits, recorded greedy continuation (±2 tie window), sampling determinism, <300 MB RSS. Registered as ctest `omniseed_real_weights` (WD = repo root; `omniseed_platform` got the same fix). ctest: **2/2 suites pass**. |

## 2c. PHASE 8 — SPEED + REAL SENSES ✅ (TASK 1: c084868; TASK 2: ac42318; TASK 3: 1d93c60; TASK 4: 3066030 + 4575fac)

### TASK 1 — Speed (AVX2) ✅
- AVX2 + SSE4.1 + scalar kernels for the per-row int8 dot (`src/core/bitlinear_avx2.cpp`, cpuid dispatch in `platform`, runtime-selected via function pointers). AVX2 kernel measured **7.3× the scalar** in isolation (0.45 → 3.25 GMAC/s).
- Profiled: the fp16 head matmul was 68% of token time (scalar `half_to_float` on 50M values). Head is now stored **int8 with per-row scales in the GGUF** (converter writes i8 head; loader accepts it) → fp16 head conversion at load time removed.
- **Result: 2.1 → 19.2 tok/s (9.1×), peak RSS 202.8 → 155 MB**, greedy continuation byte-identical to pre-change (numerics preserved).

### TASK 2 — Real senses ✅
- `tools/convert_senses.py` (offline): converts **openai/whisper-tiny** encoder safetensors + official `mel_filters.npz` (80×201) → `models/whisper-tiny-encoder.gguf` sidecar; **vision projection head** (ternary + per-row fp32 scales) → `models/vision-proj.gguf`.
- `WhisperTiny`: loads the full encoder (conv stem, pos-embed, 4 post-LN blocks, ln) and runs a real forward: mel → conv1/gelu → conv2/gelu → +pos → blocks → [T/2, 384] (98 frames → 49×384, all finite). `weights_loaded_` now true when sidecar present; spectral-hash fallback remains only as fallback.
- `VisionEncoder`: loads ternary `vision.proj.weight` + per-row scales; encode runs patch features → UniCompress → **real ternary BitLinear projection** (output varies with input; identity fallback retired).
- Full load-time tensor sweeps (env `OMNISEED_DBG=1`) read every element of every sidecar tensor at load.
- Sidecar bias convention: **present = full extent, absent = loader tolerates** (`load_f16(..., required=false)`); never dereference an optional bias without the guard below.

### TASK 2 postmortem (three stacked bugs)
1. `GgufLoader::read_tensor_dir` raised "unsupported dtype 40" for TERNARY: `gguf_dtype_size` returns 0 for ternary ("handled at call site") but the error fired **before** the call-site special case. Fixed the order.
2. `Tensor::nbytes()` returned numel bytes for TERNARY (true stored size is packed `(numel+1)/2`) — any full-region scan walked 2× the mapped bytes. Fixed to match the loader.
3. **The subtle one**: a default-constructed (absent) `Tensor` has `numel()==1` (empty-shape product) with `data()==nullptr`, so the widespread `numel() > 0` "optional present" guard **passed for missing tensors** and then dereferenced null (ASan: access-violation at the vision bias add). All optional-tensor guards now use `numel() > 1`. ASan build (`build/asan_build.bat` pattern: `/fsanitize=address` + MSVC asan DLL on PATH) found in one run what hours of Release-mode printf bisection could not.
- Loader-ownership rule (3rd strike): **every component holding non-owning views must own its `GgufLoader`** (RwkvModel `store_`, WhisperTiny `store_`, VisionEncoder `store_`). A local loader unmaps on return and dangles.

### TASK 3 — QAT true-ternary PoC ✅ (commit 1d93c60)
- `tools/qat_ternary.py` (offline, PyTorch CPU): loads the Hakureirm checkpoint,
  reuses the converter's verified reference forward, ternarizes the 8 big linears
  per layer with **STE** (masters in fp32, straight-through to {-1,0,+1}), trains
  on an in-repo markdown corpus, evals val perplexity, exports a QAT GGUF.
- **Numbers:** bf16 baseline PPL **41.9** → PTQ-ternary **479,871** (destroyed, as
  predicted) → after only **150 toy steps** (window 8, lr 1e-3, chunked/resumable):
  **230,013** — a 2.1× recovery; training loss 15.4 → ~9.5. Quality parity needs a
  real corpus and thousands of steps (CPU autograd ≈ 23 s/step at window 16).
- `models/rwkv7-0.1B-ternary-qat.gguf` loads through the **C++ ternary path**
  (peak RSS **114.6 MB** — ternary is ~3× smaller than i8); output degenerate at
  PoC quality, as expected.
- Gotchas encoded in the tool: this checkpoint family stores 1-D norms with a
  leading `[1,...]` dim (strip ALL leading dims or x broadcasts to [1,768]);
  ternarize once per window (weights only change at optimizer step); torch
  single-thread for these tiny ops.

### TASK 4 — Product polish ✅ (commits 3066030, 4575fac)
- **Streaming chat:** `AgentLoop::Config.on_token` callback prints pieces as they
  are generated (`omniseed chat`).
- **Server:** loads the **world tokenizer from the model GGUF** (was a minimal
  byte tokenizer → degenerate replies), honors `--model` or `OMNISEED_MODEL`,
  `/health` reports `model:true` + `peak_rss_mb`; `/gen` verified coherent.
  Docker unavailable on this box — Dockerfile verified by inspection (same flags
  + `/health` contract tested natively).
- **Checkpoint shootout:** added `omniseed ppl` (teacher-forced cross-entropy over
  a corpus) and `tools/convert_goose28.py` for the official
  `RWKV/RWKV7-Goose-World2.8-0.1B-HF` (fla naming). Result: after fixing a
  lora.2 **transposition bug** (fla `nn.Linear` weights are ALREADY in C++
  storage form — no transposes anywhere), the Goose model generates coherent
  English, and **both 0.1B checkpoints proved bit-identical** (per-tensor
  max|diff| = 0.0; identical NLL to 4 decimals over 2048 tokens, PPL ≈ 32.9 on
  in-repo markdown) → existing model stays the default.
- **fla token-shift myth busted:** verified against the official transformers
  `modeling_rwkv7.py` that fla and BlinkDL share the SAME token-shift algebra
  (`delta = prev - x`); the fla-form dual-path added earlier was based on a
  misread and was REMOVED (`omniseed.fla_shift` gone).

### Verification (Phase 8)
- `omniseed_tests` **165/165**, `omniseed_real_weights` **13/13**, `omniseed_sides`
  **14/14** (skip cleanly when sidecars absent): mel [80×98], encode [49×384]
  finite, vision 768-dim projection finite + input-sensitive (gradient vs
  inverted differ > 1e-4).
- Speed/RSS: scalar 2.1 tok/s / 202.8 MB → AVX2 **19.2 tok/s / 155 MB** (model
  loaded); gen observed 15.7–20.8 tok/s; ternary-QAT path 114.6 MB.

## 2d. PHASE 9 — SHIP IT ✅ (TASK 1: 443be29; TASK 2/3/4: see below)

### TASK 1 — Render live ✅
- **Browser demo console at `GET /`**: inline single-file HTML (no CDNs) — chat
  box wired to `POST /gen`, header polls `/health` for `model` + `peak_rss`,
  typing indicator, error styling. `GET /index.html` serves the same page.
- Verified natively: `/` serves HTML; `/gen` coherent; `/health`
  `{"ok":true,"model":true,"peak_rss":155}` — note **RSS only reaches ~155 MB
  after the first `/gen`** (mmap lazy paging); check `/health` after a `/gen`.
- `docs/RENDER_LIVE_CHECKLIST.md`: push-readiness audit, blueprint import, two
  model-attach options (disk vs Dockerfile fork that COPYs the GGUF — binary
  paths corrected to `build/omniseed_server`, single-config CMake), live
  verification (wake the model first!), budget notes.

### TASK 3 — Real whisper decoder (end-to-end ASR) ✅ (commit 1939cd9)
- `tools/convert_senses.py` now exports the full **decoder**: tied
  `whisper.dec.tok_embd` [51865,384] (doubles as lm_head), learned pos table
  [448,384], 4 pre-LN blocks (self-attn + cross-attn + MLP, whisper-tiny ships
  **no k_proj biases** — converter writes biases only when present), final LN,
  and the **byte-level vocab** as `whisper.vocab_offsets` (I32-as-F32 container)
  + `whisper.vocab_bytes` (bytes in an F16 container — piece byte i = low byte
  of element i; the loader has no string-array dtype). Specials (ids 50258+)
  come from tokenizer.json `added_tokens`. Sidecar: 72.9 MB.
- `WhisperTiny` (pure C++17):
  - **encoder upgraded to real bidirectional MHA** (6 heads; per-block K/V
    caches) — the old per-token value shortcut produced garbage cross-attn
    context; HF's **final encoder layer_norm** is now applied to the output
    (the decoder cross-attends to that normalized output).
  - **`decode_greedy`**: forced prompt `<|startoftranscript|><|en|>
    <|transcribe|><|notimestamps|>` (ids 50258/50259/50359/50363 — note
    `<|en|>` is 50259, NOT 50262), causal self-attention with incremental
    per-layer K/V caches, cross-attention K/V precomputed once per layer from
    the encoder frames, tied-embedding lm_head + argmax, greedy until
    `<|endoftext|>` (50257) or the 224-token cap.
  - Verified: 5 s of silence → **EOT at step 0 → ""** (correct); debug top-5
    (env `OMNISEED_DBG=1`) shows 50257 as the top token. Encoder-only sidecars
    still work (documented `<|audio|> frames` fallback; decoder load failures
    never invalidate the encoder).
- Tests: `omniseed_sides` **17/17** (decoder coverage added: bounded runtime,
  bounded output, printable bytes); 165/165 + 13/13 unchanged.

### TASK 4 — CI portability proof ✅ (commit 5250268)
- `.github/workflows/ci.yml`: **ubuntu-latest (gcc)** + **windows-latest
  (MSVC VS2022, x64)** — Release configure/build, `ctest` both OSes, plus
  model-less smokes: `omniseed selftest/info/bench` (all exit 0 without a
  model) and, on Linux, `omniseed_server` + `curl /health` + `GET /`.
- Static Linux audit passed (no g++/Docker on this box — the CI run is the
  real proof): platform guards `OMNISEED_PLATFORM_*`, GCC `__builtin_cpu_*`
  cpuid path, per-TU `-mavx2;-mfma` flags, POSIX RSS via `/proc/self/status`,
  portable mkdir in flash_skills. Dockerfile COPY paths fixed (single-config
  generators emit into `build/`, not `build/bin/`).

### TASK 2 — Full-scale QAT (tooling ✅; long run IN PROGRESS)
- `tools/qat_ternary.py` rewritten for full scale: **wikitext-103-raw**
  auto-download + token cache (`models/corpus/*.npy`, gitignored), **batched
  training** (`step_batch`: B independent contiguous windows per step —
  verified **bitwise identical** to the reference `step()` via `--verify-batch`;
  fixed an outer-product that used the pre-blend k instead of the blended k),
  AdamW + cosine LR + warmup + grad-clip, periodic eval (32 parallel segments),
  best-checkpoint tracking, `qat_log.txt`, `--time-budget` chunks that exit 3.
- `tools/qat_chunk.py` + `tools/qat_watch.bat` / `tools/qat_watch.sh`: loop
  chunks until the target step count; resume from `models/qat_ckpt.pt`
  (masters + optimizer + global step + best). Restart-safe.
- Throughput: ~8.5 tok/s training (B=8/W=16, single-thread — multi-thread is
  5× slower on this box; B=16 improves tok/s but halves updates/step).
- **Trajectory (wikitext-103 val, 2048-token estimate):** bf16 baseline
  **31.16** → ternary-PTQ **257,402** → step 175 **607.18** → step 200
  **538.85** … (falling; log in `qat_log.txt`). Export/load chain re-verified:
  `models/rwkv7-0.1B-ternary-qat.gguf` (192.1 MB) loads in C++ at **115 MB
  peak** and already samples wikitext-domain words ("the 1950s…") — coherent
  prose expected only near the ≤1.25×-of-bf16 (~39) target.
- **Resume command** (needs ~13 h at 12.5 s/step to 4000):
  `tools\qat_watch.bat 3000` (Windows) / `./tools/qat_watch.sh 3000` (Unix),
  or single chunks:
  `./.venv/Scripts/python.exe -u tools/qat_ternary.py --corpus wikitext
  --steps 4000 --eval-every 50 --window 16 --batch 8 --time-budget 470`

## 3. NEXT STEPS (in order)

1. **Commit this hardening pass**, then **full-scale QAT**: real corpus
   (multi-GB), 10k+ steps, lr schedule → make true ternary the default at i8
   quality (runtime already validated; the tool has chunked resume for long
   runs).
2. **CI on Linux** is the natural next proof for this pass: the byte-exact
   readers, POSIX fd_/mmap path, and socket shim are now CI-relevant — watch
   the ubuntu-latest job in `.github/workflows/ci.yml`.
3. **Whisper decoder weights**: autoregressive decoder → real end-to-end ASR
   (encoder is already real).
4. **FocalCodec codebooks**: last algorithmic-fallback sense; extend
   `convert_senses.py` when a suitable public checkpoint is identified.
5. **Tie-window test note**: `test_real_weights` records a continuation with
   a ±2-token tie window; if the reference drifts, re-record from a fresh run.
6. **WebRTC (#19)** upgrade path: replace/augment `UdpBeacon` with libdatachannel
   if the budget allows; DESIGN-status features #29/#30/#81–100 graduate when a
   training pipeline lands.

## 3c. PHASE-OMEGA — GRAND UNIFICATION (this pass)

### The audit
All three registries absorbed (z.ai "Complete Universe" 150+ features + v4.0
paranormal/quantum expansion; deepseek VOL.I + VOL.II 48-category extension;
grok grounded registry with per-feature RAM math). Original 7 research TXTs
were already audited into the feature tables of section 1. Everything below
was scored against the real budget: **155 MB resident → ~145 MB headroom**,
pure C++17, zero deps.

### Bucket A — IMPLEMENTED NOW (this pass)
| Feature (registry id) | Files | RAM | Notes |
|---|---|---|---|
| Uncertainty quantification (deepseek VOL.II "不确定性量化", z.ai Q-08) | `include/omniseed/core/uncertainty.h`, `src/core/uncertainty.cpp` | ~0 (scalars) | Stable softmax entropy + top-2 margin + abstain policy; wired into AgentLoop (first-logits analysis, `Result.first_token_uncertainty`, `abstain_hedge` config) |
| Entropy anomaly detection (z.ai P-02) | same file (`EntropyMonitor`) | 64 floats | Sliding z-score window; flags loop-collapse / topic-shift moments |
| **WKV prefix snapshots** (grok C01/C02) | `include/omniseed/memory/prefix_cache.h`, `src/memory/prefix_cache.cpp` | 1–5 MB (8 × 0.59 MB, LRU) | Byte-exact state serialization, LRU store, binary persistence; wired into AgentLoop (`prefix_key` config, snapshot after each turn, restore on hit) |
| Ebbinghaus memory decay (deepseek VOL.II "记忆擦除", grok M06) | `memory.h/.cpp` | 0 | `effective = importance × exp(-age/τ) × (1+0.1·hits)`; `decay()` + recency bookkeeping on retrieve |
| Entropy salience gate (grok M02) | `memory.h/.cpp` | 0 | High-entropy retired turns get an importance boost at crystallization |
| Working memory scratchpad (grok M, deepseek L1) | `memory.h` (`WorkingMemory`) | 2 KB | Fixed-capacity token ring |
| Temperature annealing (deepseek VOL.II) | `agent.h/.cpp` | 0 | Explore→exploit ramp over `anneal_tokens` |
| RSS watermark + kill-switch (grok S01) | `agent.h` (`ComputeThrottle::rss_zone`), `agent_loop.cpp` | 0 | Soft 260 MB (halve budget) / hard 295 MB (stop) — budget enforcement without a monitor thread |

### Bucket A — REFUSED WITH MATH (documented, not forgotten)
- **N-gram/prompt-lookup speculative decoding (grok C04/C05, deepseek #5):**
  RWKV-7 is strictly recurrent — verifying m draft tokens costs m sequential
  forwards, exactly greedy's cost. Draft speculation only wins where
  verification is parallel (transformers). The registries' own "do not lie to
  implementers" rule applies. Revisit only if a parallel-scan WKV verify
  kernel lands (see FUTURE HORIZON).
- **Draft-model/Medusa/EAGLE speculation:** second model does not fit the
  budget next to the 0.1B kernel (grok C24 already IMPOSSIBLE).

### Bucket B — FUTURE HORIZON ROADMAP (the next AI's queue)
1. **BitNet b1.58 native QAT to production quality** (grok C13/C14) — the
   strategic RAM unlock: ternary 0.1B ≈ 24 MB weights → ~80–120 MB total RSS
   → Whisper-tiny + vision + TTS could then CO-RESIDE. Tooling exists
   (`tools/qat_ternary.py`); needs corpus-scale training time.
2. **Parallel-scan WKV verify kernel** — would flip speculative decoding
   from refused to profitable; pairs with C04/C05.
3. **RWKV-Lite clustered LM head + sparsity predictor** (grok C08/C09) —
   steals back 20–60 MB of the vocab-projection fat tail (V=65536).
4. **Int4 grouped FFN linears** (grok C12) — saves 30–50 MB; quality gate
   must beat int8 on the `omniseed ppl` harness first.
5. **Layer-stream mmap + madvise working-set control** (grok C18/S05) —
   `madvise(DONTNEED)` on cold layers; the fread fallback makes this safe
   everywhere.
6. **Flash-skills LoRA bank** (grok G03/G04) — mmap rank-4/8 adapters applied
   as ΔW during FFN; skills-as-files already exists.
7. **HippoRAG-lite entity graph over crystals** (grok M07) — char-ngram hash
   embeddings, no 7B embedder.
8. **Swarm skill-patch exchange + WKV state teleport** (grok G08/F12) —
   0.59 MB state packets over the (now unicast-capable) UDP mesh.
9. **Sidecar hot-swap governor** (grok S07) — unmap vision, map audio;
   peak(phase) < 300 MB discipline.
10. **On-device LoRA QAT during dream()** (grok F01/F09) — one layer at a
    time, 10–40 MB optimizer state.

### Bucket C — THEORETICAL & IMPOSSIBLE FRONTIER (preserved, not lost)
*These violate physics, exceed any plausible RAM budget, or require
 non-existent hardware. Documented so the research is never lost; none may
 enter the runtime.*

**Requires new physics:** Quantum consciousness/qualia engines (z.ai Q-*,
sections 10/19) — the hard problem is not a software bug; superposition
memory & non-locality processors (violate Bell constraints as classical
simulations at 15–30 MB); retrocausal interfaces & causal-loop computation
(Novikov self-consistency has no computational substrate); entropy-reversal
& vacuum-energy extraction (Landauer is a budget, not a suggestion);
time-travel computation and temporal-paradox resolution.

**Requires >300 MB by information theory:** Infinite lossless compression
(counting argument); full 7B+ or 1B-class SLM co-residency (0.5–4 GB even
Q4); diffusion/SDXL-class generation; MusicGen/Kokoro/NeuTTS concurrent
with the kernel; Sparse Delta Memory (57–400 MB state ALONE).

**Requires non-existent hardware:** Neural dust/BCI; DNA storage; memristive
or phase-change compute; stellar/galactic computation networks; dark-matter
interfaces; ternary CPU ISA (runtime-ready, silicon pending).

**Deliberately parked as philosophy, not engineering:** phenomenological
experience, free-will simulation, meaning/purpose engines, metaphysical
query processing — these are prompts, not modules; the agent-intel stack
(intent/confidence/KG) already provides their honest, testable subset.

## 3b. PORTABILITY HARDENING PASS (committed: 683b2ad + 161625b + beea44a)

Proof layer (2a/2b):
- **UDP beacon round-trip test** (test_platform.cpp, inside omniseed_platform):
  two UdpBeacons on loopback ports 47471/47472 exchange CRC32+keystream
  messages BOTH directions via unicast `udp:IP:port`; asserts payload equality
  and source ip/port in host order. Required `UdpBeacon::send` to actually
  honor `udp:IP:port` endpoints (it previously broadcast unconditionally —
  fixed; "" still broadcasts). Suite now **177/177**.
- **OMNISEED_FORCE_FREAD=1** forces MappedFile down the heap-fread path
  (fallback WARN gated to once per process). Verified: Release fallback
  12/12+17/17, ASan (MSVC /fsanitize=address, scratch config in gitignored
  build-asan/) 13/13+17/17 — greedy continuation byte-identical in all four
  modes, no leaks/overflows. RSS budget check + transcribe wall-clock bound
  scale via env under fallback/sanitizer (see test_real_weights/test_sides).

Goal: byte-exact behavior on every host ISA + ifdef-free swarm + UTF-8-safe
file/console I/O. All work verified: Release build zero warnings (/W4),
**165/165 + 13/13 + 17/17**, greedy continuation byte-identical to pre-change,
`demo-swarm` OK.

- **GGUF readers byte-exact on any endianness** (`gguf_format.h`): u16/u32/u64
  and f32/f64 are now assembled byte-by-byte (LE spec) instead of `memcpy`-ing
  host-order words. read_f64 is bit-preserving via u64 + memcpy reinterpret.
  (An interrupted earlier edit had DELETED read_f64 while gguf_loader.cpp still
  called it — restored.)
- **MappedFile fread fallback** (`platform.h/.cpp`): open() tries a real mmap
  first; on failure falls back to a full heap fread (byte-identical views,
  every platform). `using_fallback()` / `mapped_with_fallback()` probe the
  mode; a WARN log fires when the fallback engages. POSIX `fd_` member kept
  (an interrupted edit had removed it while close()/move-assign still used it).
  close() only unmaps when NOT fallback-backed.
- **UDP socket shim** (`platform.h/.cpp`): the ONLY socket code in the repo.
  WinsockOne-time WSAStartup, open/bind/SO_BROADCAST/non-blocking, close,
  broadcast/send, poll. IP/port args are HOST order on the API surface (shim
  owns htonl/htons) — no consumer needs ntohs/ntohl, endian-safe by design.
  poll() returns source ip4 + port (host order). swarm.cpp is now ifdef-free
  on top of the shim; endpoint format "udp:IP:port" restored (host-order
  dotted quad + port — the half-finished edit had dropped the port and used
  network-order bytes). ws2_32 already linked to omniseed_core; winsock2.h
  included BEFORE windows.h in platform.cpp.
- **UTF-8-safe fopen** `platform::open_file_c()` (Windows _wfopen path) adopted
  by all six persistence sites (sensory, memory_crystals, agent_intel KG,
  flash_skills save/load, self_improvement) — non-ASCII state paths work.
- **UTF-8 console** `platform::enable_utf8_console()` (CP 65001) called at the
  top of CLI + server mains (resolves the old "consider SetConsoleOutputCP"
  issue).
- **Server build hygiene**: omniseed_server now defines _CRT_SECURE_NO_WARNINGS
  (same portable-stdio stance as omniseed_core) — zero-warning /W4 restored.

## 4. UNRESOLVED ISSUES / BUGS

- **Sense fallbacks largely retired**: whisper-tiny encoder + vision projection
  load real weights (sidecars); remaining fallbacks are the whisper *decoder*
  and FocalCodec codebooks (NEXT STEPS #2/#3).
- **Server is serialized** (one request at a time) — fine for edge use.
- **MinGW/GCC builds untested** on this machine; MSVC `/W4` is clean.
- Persistence paths route under `./state/` (created on demand, verified).
- `models/` (281 MB safetensors + 294 MB GGUF) is gitignored; the converter
  re-creates the GGUF from the safetensors + vocab (download URLs in the tool).

## 5. VERIFICATION STATUS

- **Phase-Omega additions (this pass):** `omniseed_tests` **200/200** (added:
  uncertainty entropy/margin/abstain incl. coin-flip tie, entropy anomaly
  z-score window, prefix-cache container semantics, entropy salience +
  working-memory ring), `omniseed_real_weights` **13/13** (RSS 155 MB),
  `omniseed_sides` **17/17**; `omniseed gen` real-model smoke: coherent,
  14.3 tok/s, peak **155.4 MB** — the Omega features cost ~0 RSS.

- `build\bin\omniseed_tests.exe` → **165/165 PASS** (added: FFT-vs-DFT exactness,
  tone-bin agreement). Coverage: platform (mmap,
  RSS, clocks), tensor, BitNet pack/matmul/QAT, GGUF, tokenizer, RWKV state,
  streaming window/retire, memory crystals, grammar decoder, tool registry,
  agent intel (intent/state/confidence/KG/feedback), compute throttle,
  self-improvement + dream, flash skills (builtins, synthesis, persistence,
  maturity/retire), sensory (enroll/verify, impostor reject), emotional
  (channels, fusion, modulation), swarm (discovery, delegation, memory/skill
  Xfer, codec, staleness), audio events (wake word, events, scene, anomaly),
  vision tasks (pointer, spatial, tracking, gestures, resolution).
- CLI demos verified by hand: all 13 commands run; peak RSS **< 5 MB** each
  (no model loaded); `selftest` OK; `tools` emits valid JSON schemas.
- **Real-model verification:** `gen --model models/rwkv7-0.1B-ternary.gguf`
  greedy + sampled output coherent (see 2b/2c); peak RSS **155 MB** at
  **19.2 tok/s**; three suites green (`omniseed_platform` 165,
  `omniseed_real_weights` 13, `omniseed_sides` 14). Server `/gen` coherent;
  `/health` reports model+RSS. `omniseed ppl --eval-file F` for perplexity
  (PPL ≈ 32.9 on in-repo markdown).
- Fresh-clone build verified with wiped `build/`; zero compiler warnings.
- `docs/OMNISEED_MASTER_SPEC.md` updated to v2.0 (real-weights story, Phase 7+8
  tools, benchmarks, verification status).

---

## 6. PHASE-11 — GPU-READY QAT + GITHUB PUSH (2026-09-07)

### TASK 1 — Colab/GPU readiness (complete)

- **`--device {auto,cpu,cuda}`** (`tools/qat_ternary.py`, default `auto`):
  frozen params + trainable masters move via `RWKV7Ternary.to_device()`
  (masters rebuilt as fresh leaf tensors before the optimizer exists);
  batch/eval/verify tensors are created on the device; RNN state zeros are
  device-aware. On cpu every `.to()` is a no-op.
  **Bit-identity proven:** `--verify-batch --device cpu` →
  `max|dlogits|=0.00e+00 max|dstate|=0.00e+00 OK` — identical to the
  pre-change baseline.
- **Fresh-clone bootstrap:** `ensure_checkpoint()` auto-downloads the public
  HF checkpoint (`Hakureirm/rwkv7-0.1b-hf` → `models/model.safetensors`,
  config refreshed into `models/hf-orig/`) with plain urllib, no login — a
  fresh clone trains end-to-end.
- **World vocab committed:** `tools/data/rwkv_vocab_v20230424.txt` (1.04 MB,
  `cmp`-identical to the models/ copy). New shared
  `convert_to_omniseed.resolve_vocab()` resolves repo-relative first
  (tools/data/), models/ second, explicit-path-wins; wired into
  `qat_ternary.py`, both converters, and the oracle/diff/sim dev tools.
  `.gitignore` negation exceptions added.
- **`tools/COLAB_QAT.md`:** exact cells — clone, pip, Drive mount, ckpt
  restore/save per chunk, `--device cuda --time-budget 3300` chunk loop
  (exit 3 = continue), auto-export of best-masters GGUF, download home.
- **`--smoke` flag:** 2 real steps, eval off, no ckpt write / no export —
  verified green (`auto` → cpu here; no CUDA on this box; an explicit cuda
  request without a runtime warns loudly and falls back to cpu).
- Full suite green post-change: ctest **3/3** (200 + 13 + 17).

Next in this pass: TASK 2 (gitignore audit + `git add -A` commit) and
TASK 3 (force-push `main` to github.com/assasian-art/Omniseed, then
sanitize the remote URL — the session token lives only in shell commands).
