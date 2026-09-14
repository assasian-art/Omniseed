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
- **Test:** `build\bin\omniseed_tests.exe` — **206/206 passing** +
  `build\bin\omniseed_real_weights.exe` — **13/13 passing** +
  `build\bin\omniseed_trading.exe` — **2129 checks passing** (zero warnings /W4)
- **Last updated:** PHASE-16 OMEGA PASS (2026-09-11): **Trading Expert Mode +
  self-awareness + hybrid cloud reasoning.** Multi-agent trading framework
  (signals/risk/backtester/paper broker/Alpaca-paper adapter, oracle-verified
  no-look-ahead), structured introspection (self-model, goals, decision
  rationale + calibration, metacognition), optional cloud bridge + reasoning
  tools. ctest 6/6. Honest scope: loss minimization, not elimination.
  Previous: PHASE-15 CLOSE-OUT — ASR at official-HF
  parity + `POST /asr` HTTP endpoint + Modal GPU training pipeline. Real
  end-to-end transcription (`<|0.00|> Experience proves this.<|4.00|>`, ~4 s,
  omniseed_sides 22/22, oracle mel 1.1e-4 / encoder 1.1e-3 / generate parity),
  server `POST /asr` (raw WAV or wav_b64, lazy sidecar, /health "asr" field),
  Modal pipeline (`tools/modal_qat.py` + `modal_train.py` + guide) closing the
  final-push board (§12). Remaining: Modal round-3 QAT run (user-triggered),
  ternary graduation at val PPL ≤ 60, Render live deploy.

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
3. ~~Whisper decoder weights~~ — DONE, see §11 (real end-to-end ASR,
   oracle-verified to official-HF parity).
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

### TASK 2 — repo hygiene (complete)

- `.gitignore` audit: `.venv/` added (was missing — would have been swept by
  `git add -A`); `models/*` artifact drop-zone with load-bearing negations
  (`!models/hf-orig/` runtime code, `!models/rwkv_vocab_v20230424.txt`);
  global `*.gguf` / `*.ovocab`; `build-asan/` via `build-*/`; `state/`,
  `*.pt`, `*.safetensors`, `*.obj` already covered. check-ignore matrix
  verified: safetensors/gguf/pt/.venv/build-asan/state all IGNORED, tracked
  models/ files unaffected.
- **Size audit:** largest tracked file = world vocab **1.04 MB** (two
  identical copies); total tracked tree **2.96 MB** — every file is ~87×
  under GitHub's 90 MB push limit; full history contains no blob near it.
- `git add -A` staged only the intended files; committed as
  "Phase 11: GPU-ready QAT, Colab guide, full source".

### TASK 3 — push (executes immediately after this commit)

- `main` force-pushed to `https://github.com/assasian-art/Omniseed`
  (intentional: replaces the placeholder initial commit with full history).
- Security: the session token appears ONLY inside the git remote/push shell
  commands; `git remote set-url` sanitizes the URL immediately after the
  push, and the token is never written to any file, commit, commit message,
  or log. Post-push verification: `.git/config` token-free, `git log -p`
  history token-free, anonymous `ls-remote` proves the repo is publicly
  cloneable.

---

## 7. PHASE-12/13 — QAT ROUNDS + TERNARY RUNTIME (2026-09-08)

### Phase 12 — Round-1 QAT (Colab T4, 4000 steps, B=16/W=16)

- Trajectory: val PPL **257,402 → 538 → 136.70 (best @ step 3800)** — a
  ~1900× recovery from PTQ; the schedule ended at lr≈0 so the run stopped
  improving. `models/rwkv7-0.1B-ternary-qat.gguf` (201 MB) loads in C++
  (114.6 MB RSS) but generation loops on wikitext-style text — the corpus
  was too narrow and the model never saw narrative diversity.
- Lesson encoded into round 2: narrow corpus + lr→0 = repetition loops, not
  quality.

### Phase 13 TASK 1 — QAT round-2 tooling (committed: bc47799)

- `tools/qat_ternary.py` gains **teacher distillation** (`--kd-weight`/
  `--kd-temp`): KL(teacher‖student) with Hinton T² scaling against the
  FROZEN bf16-master forward on the same window — transfers the full
  distribution, not the argmax. `--kd-weight 0` (default) keeps every
  existing path — including `--verify-batch` — bit-identical to round 1.
- **TinyStories 50/50 corpus mix** (`--corpus wikitext+tinystories`,
  round-2 default): streamed plain-text files (3M-token cap ≈ 15 MB of the
  1.9 GB train file) interleaved per batch — narrative diversity breaks the
  wikitext loops.
- **`--lr` + `--cosine-start`** restart the cosine schedule at a fresh
  (smaller) peak while resuming `models/qat_ckpt.pt`; `--window 32`
  (2× round-1 context). Colab round-2 recipe in `tools/COLAB_QAT.md`
  (target 12000 steps, chunk loop, Drive restore/save).

### Phase 13 TASK 2 — Ternary runtime: SIMD kernels + repetition penalty
### (committed: 73e4356)

Two runtime gaps stood between the round-2 GGUF and a usable model:
the ternary per-row dot ran scalar-only (0.6 tok/s), and even a good
ternary model loops without repetition suppression.

**Packed-ternary SIMD kernels** (`bitlinear.h/.cpp`, `bitlinear_avx2.cpp`,
`rwkv.cpp`, `tests/test_platform.cpp`):
- SSE4.1 + AVX2 kernels for `bitlinear_forward_rows` (2 weights/byte,
  pshufb LUT unpack to {-1,0,+1}, then the i8 dot path). Same cpuid
  dispatch + `OMNISEED_NO_SIMD_TERNARY=1` scalar escape hatch as the i8
  path.
- **Bit-exactness contract:** all three kernels (scalar/SSE4.1/AVX2)
  implement ONE canonical accumulation order — 8 fp32 lanes, weight index
  c ≡ lane (mod 8) in increasing c, horizontal sum `((l0+l4)+(l2+l6)) +
  ((l1+l5)+(l3+l7))`, tail folded into lanes in increasing c. w·x is exact
  in IEEE (w ∈ {-1,0,+1}), so only the lane additions round, identically
  on every host. `rwkv.cpp` hot paths (per-row linears + ternary head)
  now call `bitlinear_forward_rows_simd`.
- New test `test_ternary_simd_bitexact`: memcmp A/B vs scalar across
  {2×2, 5×7, 17×24, 64×96, 33×77 (odd in_dim), 256×768} — **all
  byte-identical**.
- **Verified today:** QAT-model greedy gen byte-identical with/without
  SIMD (only the timing line differs); **0.6 → 17.4 tok/s (29×)**,
  peak RSS unchanged 114.9 MB. i8 default model path untouched
  (real-weights continuation re-recorded and matched).

**Repetition penalty** (`agent.h`, `agent_loop.cpp`, `cli/main.cpp`):
- CTRL-style: after each forward, tokens in the last `repeat_window`
  generated tokens get logit/penalty when positive (×penalty when
  negative); `AgentLoop::Config{repeat_penalty=1.0, repeat_window=64}`,
  CLI `--repeat-penalty P` on `gen`/`chat` (1.0 = off).
- **Verified today:** QAT model, same prompt: no penalty →
  "the first time , the first time , the first time …" infinite loop;
  `--repeat-penalty 1.2` → loop broken, coherent continuation at
  19.5 tok/s (no measurable overhead).

### Phase 13 TASK 2b — dispatch hardening (committed: c1eb2f0)

- **`bitnet::force_ternary_kernel(TernaryKernel{Auto,Scalar,Sse41,Avx2})`**
  — explicit per-call kernel pick for tests/bench (an explicit pick wins
  over both env vars; non-x86 hosts no-op the SIMD picks). New env
  **`OMNISEED_FORCE_SSE4_TERNARY=1`** forces the SSE4.1 ternary kernel even
  on AVX2 hosts (first-init; joins `OMNISEED_NO_SIMD_TERNARY=1`).
- **New ctest `omniseed_qat_ternary`** (`tests/test_qat_ab.cpp`, 8 checks):
  loads the real QAT model, greedy-generates 24 tokens from "Hello" under
  each forced kernel, asserts **byte-identical continuations across
  scalar/SSE4.1/AVX2 in one process** (verified on this AVX2 box; trivially
  identical on non-x86 CI), + RSS budget check. Skips cleanly when the QAT
  GGUF is absent. Also verified end-to-end through the CLI: gen output
  identical under both env forces.

### Phase 13 TASK 2c — server exposure (committed: 43b3135)

- `AgentLoop::set_sampling(repeat_penalty, repeat_window)`; `POST /gen` and
  `POST /ask` accept optional `"repeat_penalty"` (default 1.0 = off) and
  `"repeat_window"` (default 64). The serialized server makes per-request
  sets race-free; **absent fields revert to the defaults, so each request
  is self-contained** (verified live: penalty 1.2 breaks the loop, the next
  request without the field loops again).
- Docs: MASTER_SPEC CLI table row + server contract note + Verification
  section; DEPLOYMENT.md endpoint table.

### Verification (Phase 12/13 close-out)
- ctest **4/4**: `omniseed_platform` **206/206**, `omniseed_real_weights`
  **13/13**, `omniseed_sides` **17/17**, `omniseed_qat_ternary` **8/8**
  (three full QAT-model generations, 35 s).
- Gen smoke `--repeat-penalty 1.2` (QAT model): loop broken, coherent,
  **16.4 tok/s, peak 115.1 MB**.
- Server live check: `/health` `{model:true,peak_rss:115}`; `/gen` with
  `"repeat_penalty":1.2` coherent vs looping default; `/ask` with penalty OK.

### NEXT STEPS (updated)
1. **Round-2 QAT run to completion** (Colab, 12000 steps, ~13 h) — **USER-
   TRIGGERED manually on Colab; do not start locally.** The repetition
   penalty is a stopgap; KD + TinyStories is the real fix. Recipe: the R2
   chunk loop in `tools/COLAB_QAT.md`; restore `models/qat_ckpt.pt` from
   Drive (Cell R1).
2. Parallel-scan WKV verify kernel (would flip speculative decoding from
   refused to profitable) — unchanged from PHASE-OMEGA.
3. Whisper decoder + FocalCodec codebooks — unchanged (see §3).
4. Optional: browser console (`GET /`) could expose a penalty slider once
   round-2 lands and the default model becomes ternary.

---

## 8. PHASE-13B — ROUND-2b QAT STATUS (2026-09-09)

### Round-2b results (this box, ternary QAT GGUF)

- Context: the round-2 Colab run hit the GPU quota at step 4326; best
  **159.15 @ step 600** (lr 2e-4), PPL oscillated 176–624 afterwards. That
  diagnosis (lr too hot) produced the calm-recipe tooling (commit 4b7ea2b):
  `--resume-best` (tracked-best masters + AdamW-moment reset, global step +
  best kept), grad-norm logging every 10 steps in `qat_log.txt`, and the
  ROUND 2b section in `tools/COLAB_QAT.md`.
- **Bench: 28.6 tok/s** on the ternary QAT GGUF; **gen 26.3 tok/s @
  repeat-penalty 1.2**; peak RSS **114.7 / 115.4 MB** (bench/gen). The
  packed-ternary SIMD path is now **FASTER than the i8 default's 19.2
  tok/s** — 2 weights/byte wins once the row dots are vectorized.
- `--verify-batch` remains exact after the round-2b tool changes
  (`max|dlogits|=0.00e+00 max|dstate|=0.00e+00`); zero C++ changes.

### Diagnosis — why the exported masters still regurgitate

The QAT GGUF regurgitates in-repo markdown + wikitext fragments:

1. **Inherited memorization:** the master chain was first fine-tuned on the
   PoC in-repo markdown corpus (Phase 8 TASK 3); every later resume
   (round 1 → round 2 → round 2b) inherits those weights, so the memorized
   fragments keep resurfacing in generation.
2. **Early best @600:** the tracked best froze at step 600 (PPL 159.15) —
   the hot 2e-4 LR kept bouncing out of that good basin, so the remaining
   ~3,700 steps added nothing durable (grad-norm ≈ 4.8e+02 from the hot
   masters vs ≈ 1.9e+01 from the best masters).
3. **PPL ~136–160 ≫ the ~39 target** (≤1.25× bf16): quality simply is not
   there yet.
4. **The runtime is NOT at fault:** greedy output is deterministic, the
   SIMD kernels are bit-exact across scalar/SSE4.1/AVX2 (proven on this
   exact model, `omniseed_qat_ternary` 8/8), and the repetition penalty
   controls the looping symptom at full speed. What remains is
   training-domain, not inference-domain.

### NEXT STEPS (current — supersedes the §3 and §7 lists)

1. **Round-3 clean QAT** — USER-triggered on Colab; do NOT start locally:
   fresh PTQ init, **no resume** (breaks the inherited-memorization chain),
   `--corpus wikitext+tinystories`, **lr 1e-4**, **kd-weight 1.0**,
   **window 32, batch 32**, 12k steps, and **export only at val PPL ≤ 60**.
2. **Until round-3 lands: the i8 default model stays the daily default**;
   the QAT ternary GGUF remains the speed/footprint demonstrator (28.6
   tok/s @ ~115 MB) with `--repeat-penalty 1.2` for loop-free demos.
3. Pending: **FocalCodec codebooks** (documented-missing, §9), **Render
   live deploy**. ~~LoRA assistant-behavior pass~~ — DONE, see §10.

---

## 9. PHASE-14 — CLEANUP + PENDING MODEL WORK + MODAL PIPELINE (2026-09-10)

### Cleanup (committed f39ce1d)

Deleted (regenerable / superseded): `build/`, `build-asan/`, `state/`, root
`qat_log.txt`, `tools/COLAB_QAT.md` (Colab superseded by Modal — see
`docs/MODAL_QAT_GUIDE.md`), `docs/OMNISEED_MASTER_SPEC.html` (regenerate with
`python tools/md2html.py docs/OMNISEED_MASTER_SPEC.md`), `models/qat_ckpt-OLD.pt`,
`models/qat_masters.pt`, `models/rwkv7-0.1B-ternary-qat-old.gguf`,
`models/goose28.safetensors` (proven bit-identical to the Hakureirm download),
`models/whisper-tiny.safetensors`, `models/model.safetensors`.
Kept: `.venv/`, daily i8 + current ternary QAT GGUFs, round-3 resume ckpt
(`models/qat_ckpt.pt`), sidecar GGUFs, vocab, mel_filters.npz, configs,
`corpus_eval.txt`, all src/ tests/ tools/ docs/.
Repo size **4.8 GB → 1.0 GB**; `models/` **4.7 GB → 2.4 GB**. Full rebuild +
ctest 4/4 green after cleanup.

Re-download one-liners (converters are offline; they read local files):

```
curl -L -o models/model.safetensors        https://huggingface.co/Hakureirm/rwkv7-0.1b-hf/resolve/main/model.safetensors
curl -L -o models/whisper-tiny.safetensors https://huggingface.co/openai/whisper-tiny/resolve/main/model.safetensors
```

(`mel_filters.npz` + tokenizer json were already needed and kept; whisper-tiny
repo files needed by `tools/convert_senses.py`: model.safetensors only —
tokenizer is at `models/whisper-tiny-tokenizer.json` from the same repo if
deleted: `https://huggingface.co/openai/whisper-tiny/resolve/main/tokenizer.json`.)

### P2A — FocalCodec/SemantiCodec survey: DOCUMENTED-MISSING

No suitable public checkpoint fits the runtime envelope (~115 MB total RSS,
pure C++17, zero deps). Precise findings:

- **FocalCodec** (`lucadellalib/focalcodec_50hz`, also 25hz / 12_5hz and the
  causal 2k/4k/65k variants): license **Apache-2.0** (good), but
  `model.safetensors` is **568 MB / 142M F32 params** — the encoder is a
  finetuned **microsoft/wavlm-large** (~300M params backbone). 5× the entire
  RSS budget before adding the decoder. Tensor layout: monolithic PyTorch
  state dict (FocalModulation blocks + wav2vec2-style transformer), would
  require a full focal-modulation forward in C++.
- **SemantiCodec** (`haoheliu/SemantiCodec`, MIT): the semantic k-means
  codebooks alone are `codebook_2048_0.npy` **50 MB** (2048×6144 fp32 —
  AudioMAE ViT-L/16 1024-dim 6-position concat), codebook_8192 200 MB; the
  inference checkpoints (AudioMAE encoder + SoundStream decoder) are ~1 GB.
- Verdict: **both would blow the memory envelope; runtime stays on the
  Goertzel-lite band-energy fallback** in `src/audio/focal_codec.cpp` (24
  log-spaced bands → semantic codes). What is missing is a **<=10 MB semantic
  speech-codec codebook + encoder** with a permissive license — none exists
  publicly as of 2026-09-10. FocalCodec-Stream causal ONNX exports
  (experimental, v0.0.2) are the most promising future path if a distilled
  encoder (<10 MB int8) appears.

---

## 9b. PHASE-14 — HTTP LoRA EXPOSURE + SIDECAR OVERHEAD (2026-09-10)

- **Server exposure:** `omniseed_server --assistant-lora P` (or
  `OMNISEED_ASSISTANT_LORA`) attaches a sidecar at startup; `POST /gen` and
  `POST /ask` accept an optional `"assistant_lora"` field — absent keeps the
  current attachment, `"path.gguf"` attaches (geometry-validated, logged
  fallback to base on failure), `""` detaches. The serialized loop makes
  swaps race-free; re-sending the attached path is a no-op. `/health`
  reports `"assistant_lora":true|false`. Verified live: attach → `/gen`
  (attached text), `""` → `/gen` (different text), `/ask` with field →
  re-attach (`/health` true). Contract documented in MASTER_SPEC §6 +
  DEPLOYMENT.md endpoint table.
- **Sidecar overhead (bench, ternary QAT GGUF, rank-8 sidecar attached):**
  base **25.8–27.4 tok/s** (median ≈26.6) vs attached **25.7–26.0 tok/s**
  (median ≈25.9) → **≈2–3% per-token overhead** (~0.6 tok/s), +2.6 MB RSS
  (158.2 vs 155.5 MB). First measurement showed ≈14%: the hot path was
  per-element `half_to_float` on B and per-call string-built tensor lookups.
  Fixed with a load-time hot cache (batch-converted fp32 A **and** B per
  (layer, target) — half→float is exact so the math is bit-identical;
  `omniseed_lora` 77/77 green after the change).

---

## 10. PHASE-14/2B — ASSISTANT-BEHAVIOR LoRA SIDECAR (2026-09-10)

### What shipped (this session, all local — no Colab needed)

The pending **LoRA assistant-behavior pass** is DONE end-to-end:

- **Training/export tool** `tools/lora_chat.py` (offline only, venv): classic
  LoRA B·A on the 6 big linears per layer (att r/k/v/o + ffn key/value — the
  tensors the runtime quantizes), rank 8 alpha 16, B zero-init (step 0 ==
  base), teacher-forced CE on ANSWER tokens only over the exact
  `User: ...\n\nAssistant:` template, AdamW + cosine LR + warmup, lockstep
  batch, `--sample` greedy demo, resume from `models/lora_chat.pt`, and a
  sidecar exporter (`omniseed-lora` GGUF: rank/alpha/scaling/layer_count/
  n_embd/base_vocab metadata + F16 A/B tensors, ~2.7 MB for the 0.1B model).
- **Runtime loader** `include/omniseed/core/lora.h` + `src/core/lora.cpp`:
  `LoraAdapter::load()` validates architecture + geometry, `factors(l, t)`
  returns zero-copy f16 views, `apply_lora()` adds
  `y += scaling * B(A·x)` with the A·x product HOISTED (O(in·rank + out·rank)
  vs the naive O(out·in·rank) — ~250x fewer f16 reads per ffn linear).
- **RWKV hooks** (`rwkv.h/.cpp`): `set_lora(nullptr)` default = detached,
  bit-identical output, zero cost; `forward()` adds the delta after each of
  the 6 targeted projections per layer.
- **CLI** (`src/cli/main.cpp`): `--assistant-lora P` and `--assistant-mode`
  (default `models/assistant-lora.gguf`) on gen/ask/chat; attaches before the
  first forward, logs rank/scaling, falls back to unmodified generation with
  a logged error on load/geometry mismatch.
- **Runtime test** `tests/test_lora.cpp` → ctest `omniseed_lora`, **77/77**:
  the test writes a synthetic `omniseed-lora` GGUF itself (no Python), then
  checks load/geometry, factor views (incl. empty-on-untargeted),
  `apply_lora` vs a scalar reference (max diff 0.00e+00), malformed-sidecar
  rejection (wrong arch, truncated directory, truncated payloads — clean
  `open()` failures, no crash), and a REAL-model A/B on the QAT GGUF + the
  trained smoke sidecar `models/assistant-lora-test.gguf`: attached logits
  shift max 1.76e-01, detaching restores the baseline BIT-IDENTICALLY.

Bugs found & fixed while wiring this in (the C++ path had never compiled):

- `rwkv.h` declared `class LoraTarget` — elaborated a phantom CLASS that
  clashes with the real `enum class` from lora.h; now includes `lora.h`.
- `apply_targeted` compared Tensors to nullptr / dereferenced `*f.b` (no
  compile); now checks `numel() == 0`.
- **`Tensor::numel()` returned 1 for a default-constructed (no-shape)
  tensor** — broke every "empty view" check; fixed to return 0 (src/core/
  tensor.cpp). Existing suites still green.
- **`GgufLoader::open()` left metadata_/tensors_ populated on a failed
  parse** (bare `file_.close()`) → later `tensor()` calls built views over
  wild pointers. Now full `close()` reset with the error preserved.
- **GGUF parsing crashed (uncaught std::length_error → abort, exit
  0xc0000409) on malformed/truncated files** — a corrupt model would have
  killed the CLI at runtime. `read_metadata`/`read_tensor_dir` are now fully
  bounds-checked (implausible header counts, truncated keys/strings/dims/
  dtype/offset all fail `open()` cleanly).

Status: `ctest 5/5` (206 platform + 13 real-weights + 17 sides + 8
qat-ternary + 77 lora). The i8/ternary daily-default and round-3 QAT plan
are unchanged; the sidecar is a NEW, independent axis — assistant behavior
trains without ever touching the quantized base.

NEXT (LoRA): real behavior pass needs GPU-quantity steps (minutes on Colab
vs ~hours on CPU: `python tools/lora_chat.py --steps 2000 --lr 1e-3`), then
`--assistant-mode` demos + server `/gen` pass-through. Optional: per-target
scaling metadata for fine control, multi-adapter stacking.

---

## 11. PHASE-15 — REAL END-TO-END ASR (whisper-tiny decoder, 2026-09-11)

The whisper decoder went from "garbage bytes / 70 s pathological loops" to
**official-parity transcription** on a real LibriSpeech clip, with every
pipeline stage proven against the official HF implementation.

### What was broken (four stacked bugs, found by staged oracle bisection)

1. **Decoder prefill was missing.** The step loop only ever cached the NEWEST
   row's K/V; the first "fix" primed prompt rows 0..2 from the RAW token
   embedding at every layer — but layer l must see each row's layer-(l-1)
   output. Layers > 0 attended over garbage K/V and the decoder never emitted
   <|endoftext|>: 224 junk steps ≈ 70 s per transcribe. Fixed with a shared
   `process_row` lambda (row through ALL layers: cache K/V from LN1(x), then
   self-attn over the causal prefix → cross-attn → MLP, in place) used by the
   sequential prompt prefill AND every generation step — one exact numeric
   sequence for both paths.
2. **Vocab piece-bytes were decoded from raw f16 bits.** The converter stores
   each piece byte as an f16 VALUE (`np.uint8 → <f2`), so element i's bits are
   an IEEE half encoding, not the byte. Reading the low byte produced the
   garbage; fixed with `half_to_float(elems[o])`. (Was invisible before bug 1:
   silence emitted nothing.)
3. **HF generation policy was absent.** Raw greedy argmax loops on non-speech
   tokens and misses EOT. Ported the exact processors `model.generate` runs
   for whisper: `suppress_tokens` (the 92-entry generation_config list),
   `begin_suppress_tokens` [220, EOT] at step 0, and the
   WhisperTimeStampLogitsProcessor rules (mask <|notimestamps|>, timestamp
   pairing: after a pair only text, after a lone ts only ts/EOT, monotonic
   timestamps, initial timestamp ∈ [0.00, 1.00] (max_initial_timestamp_index
   1), and the logsumexp "timestamps beat text" gate over suppressed logits).
   Default prompt is now HF's [SOT, <|en|>, <|transcribe|>] with timestamps
   ENABLED (HF's whisper default — the ts tokens suppress hallucination
   loops); the classic <|notimestamps|> prompt is `kPromptNoTimestamps`
   (masks all ts tokens for the segment).
4. **Trailing-silence padding dominated the runtime.** compute_mel pads to the
   30 s whisper window; the encoder then spent 96% of its FLOPs on zero
   frames. Fixed: after pass-1 the trailing all-silent frames are trimmed
   (energy gate > 1e-10 per windowed frame) BEFORE log-mel/normalization, so
   normalization still uses the clip's true max. 1.98 s clip: mel 3000 → 102
   frames, encode+decode ≈ 4 s total. Digital silence now fails cleanly with
   "audio too short for the encoder (need >= 2 mel frames)" (encode() guards
   T_in < 2 so the stride-2 conv can't emit zero frames → NaN attention).

### Oracle proofs (tools/dbg_whisper_ref.py — official HF, local checkpoint)

- mel: C++ vs `WhisperFeatureExtractor` (30 s pad, torch.stft center=True,
  official mel_filters.npz) → **max|diff| 1.1e-4** (f16-filterbank rounding).
  (The old C++ path used the wrong mel scale + per-frame normalization — both
  replaced earlier in this pass: librosa Slaney-scale fallback + GLOBAL max.)
- encoder: C++ (trimmed input) vs HF WhisperModel.encoder on the SAME 199-
  frame mel (HF gated at 3000; oracle drives conv stem + pos + layers manually)
  → **max|diff| 1.1e-3** (f16 weights). The 4-block bidirectional-MHA encoder
  was already exact; the [dec] step-0 logits matched too — bugs 1–3 were all
  in the decoder shell around the weights.
- generation: official `model.generate` (greedy AND beam-5) on this clip →
  [28503, 25019, 341, 13] = " Experience proves this." — the C++ decoder now
  emits exactly `<|0.00|> Experience proves this.<|4.00|>`. The environment
  was validated with whisper.cpp's jfk.wav → canonical JFK sentence.
- **The fixture's DeepSpeech label ("she had your dark suit…") is NOT what
  whisper-tiny produces for this clip.** tests/fixtures/manifest.txt now holds
  the HF-validated transcription ("Experience proves this."), and
  tools/get_fixtures.py documents why (the test asserts parity with the
  official model, not the archive label).

### Test + tooling changes

- `tests/test_sides.cpp`: real-clip ASR block (16 kHz mono s16 WAV loader,
  manifest reader, word-overlap ≥ 0.5 after stripping <|…|> tokens, bounded
  runtime) + silence expectations updated (clean-error-or-empty is correct for
  digital silence). **omniseed_sides 22/22.**
- `tools/get_fixtures.py`: manifest reference = HF-validated transcription
  (see above).
- `tools/dbg_whisper_ref.py` (new, offline): staged HF oracle — `--stage
  mel|enc|gen|all`, `--compare-mel` diff, `--slice-mel N` for trimmed-input
  encoder comparisons.
- `tools/dbg_mel.cpp` (new, unregistered like dbg_sensory.cpp): dumps C++
  mel/enc for the oracle; `OMNISEED_DBG_IDS=1` adds transcribe ids.
- `src/audio/whisper_tiny.cpp`: cross_v bias wired (whisper ships no k_proj
  biases), erf GELU, logits vector + HF policy, `DecPrompt` enum.
- Temporary OMNISEED_DBG_MEL/[logits] dump instrumentation removed; the
  pre-existing env-gated OMNISEED_DBG sweep + top-5 probes remain.

### Verification (Phase 15)

- ctest **5/5**: omniseed_platform 206/206, omniseed_real_weights 13/13,
  omniseed_sides **22/22**, omniseed_qat_ternary 8/8, omniseed_lora 77/77.
- Zero /W4 warnings; fresh CMake configure clean; debug targets unregistered.

NEXT (ASR): optional language auto-detect (<|SOT|> only prompt), server `POST
/transcribe` endpoint exposing the real decoder, streaming mel for live
capture. These do not block round-3 QAT or the Render deploy.

---

## 12. PHASE-15 CLOSE-OUT — BOARD STATUS + HTTP ASR + MODAL (2026-09-11)

### POST /asr (commit 1c6946e)

`omniseed_server` now exposes the real decoder: `POST /asr` accepts raw
16 kHz mono 16-bit WAV bytes (any Content-Type) or `{"wav_b64": "..."}`, and
returns `{"text":"<|0.00|> Experience proves this.<|4.00|>","seconds":4.3}`
or a clean `{"error": ...}` on silence/noise/too-short clips. The 73 MB
sidecar loads lazily on the first /asr call (text-only deployments never pay
for it); `/health` reports `"asr":"real"|"fallback"|false`. Request plumbing
reworked: headers + full Content-Length body read once (100 MB cap) instead
of one 4 KB recv; `PcmAudio::load_wav_bytes` added (shared RIFF walk with
`load_wav`, which now delegates). Native smoke verified (raw + b64 + silence
+ /gen regression with a real model). Contract in MASTER_SPEC §6 +
DEPLOYMENT.md endpoint table.

### Final-push board (all items CLOSED)

| Item | Status | Where |
|---|---|---|
| Cleanup pass (junk deleted, .gitignore tightened) | ✅ Phase-14 (commit f39ce1d): repo 4.8 → 1.0 GB, regenerable artifacts dropped, .gitignore artifact drop-zone + negations | §9 |
| FocalCodec codebooks | ✅ DOCUMENTED-MISSING with exact artifacts (FocalCodec 568 MB/142M-param wavlm-large encoder; SemantiCodec 50 MB codebook alone — both blow the ~115 MB envelope; Goertzel-lite fallback stays) | §9 P2A |
| Real vision backbone | ✅ WIRED at the achievable scope: real ternary projection from `models/vision-proj.gguf` (input-sensitive, tested); full MobileNetV4 embedding is documented-pending a distilled checkpoint | §2 TASK 2 |
| Modal pipeline | ✅ `tools/modal_qat.py` (train_chunk/train_lora/export_best + Volume), `tools/modal_train.py` (chunk-loop orchestrator), `docs/MODAL_QAT_GUIDE.md` — py_compile-clean (commit 1254c23) | §12 |
| LoRA assistant | ✅ `--assistant-lora`/`--assistant-mode` CLI + `"assistant_lora"` HTTP field on /gen and /ask, 77-check suite (commit 1d0782b + 4f51997) | §10 |

### Modal commands (the one-shot training path — user-triggered, not this box)

```
pip install modal && modal token new
python tools/modal_train.py qat --fresh      # round-3 calm recipe, ~6 T4 chunks
python tools/modal_train.py status           # artifacts + val-PPL log tail
modal volume get omniseed-models /models/rwkv7-0.1B-ternary-qat.gguf models/
python tools/modal_train.py lora --steps 2000    # assistant sidecar, minutes
```

### NEXT STEPS

1. **Modal one-shot training** (user-triggered): `python tools/modal_train.py
   qat --fresh` — chunk loop resumes the Volume checkpoint automatically.
2. **Ternary Graduation at val PPL ≤ 60**: download the export, flip the
   daily default to ternary (runtime already validated: 28.6 tok/s @ 115 MB,
   bit-exact kernels), re-record real-weights expectations, regenerate the
   MASTER_SPEC HTML (`python tools/md2html.py docs/OMNISEED_MASTER_SPEC.md`),
   then Render live deploy.
3. Optional ASR follow-ups: language auto-detect, streaming mel, browser
   console mic capture wired to /asr.

---

## 13. PHASE-16 — OMEGA PASS: TRADING + SELF-AWARENESS (2026-09-11)

Scope honored from the user brief: trading expert, multi-agent, AGI-style
hybrid reasoning, Mythos-style self-awareness — with the honest boundary that
"loss is not a concept" became **loss minimization through risk discipline**.
Every trading command prints the disclaimer. No guaranteed-profit claims
anywhere.

### Commits (one per phase)
| Phase | Commit | Content |
|---|---|---|
| 1+2 | `67a13cd` | trading core + news + sub-agents + introspection (16 files, 4009 insertions) |
| 3 | `9b3b56e` | cloud bridge + finance math + sandboxed reasoning tools |
| 4 | `1d0a44a` | Alpaca client (paper-first) + trading-execute gauntlet |
| 5 | (this) | docs + push |

### 13a. Trading framework (Phase 1)
- `include/omniseed/trading/trading_engine.h` + impl: OHLCV bars + CSV
  (intraday-exact round-trip), MarketDataFeed, SignalGenerator
  (SMA20/50/200, EMA, RSI-14 Wilder, MACD 12/26/9, Bollinger 20/2σ, ATR-14;
  coherent voting: oversold dip +2 beats its own downtrend −1), PositionManager,
  RiskManager (quarter-Kelly from realized stats → 2% zero-evidence probe;
  8% stop, optional TP, 20% drawdown halt, exposure cap), Backtester
  (next-bar-open fills, 5 bps fee + 2 bps slippage, signal exits, metrics).
- Verified against **real data**: 1254 AAPL daily bars (Yahoo chart API, no
  key) → 45 trades, 75.6% win, PF 1.51, maxDD 1.41% (momentum preset).
  No-look-ahead proven by test: truncating the future never changes past
  equity. Constant-return Sharpe reports 0 (zero variance), not ∞.
- `tools/fetch_market_data.py`: stooq → yahoo → deterministic SYNTH fallback
  (stooq now behind a JS proof-of-work wall; yahoo works).
- `src/trading/news_feed.cpp`: tolerant RSS/Atom ingest + GUID dedup,
  boundary-aware lexicon sentiment (\"miss\" never matches \"mission\"),
  ticker entity extraction, recency-weighted aggregates, breaking alerts.
- `src/agent/sub_agents.cpp`: MarketAnalystAgent (tech+news blend,
  weighted consensus with tie→Hold), NewsMonitorAgent (→
  SensoryInterruptSystem p9), RiskManagerAgent (concentration/dd vetoes),
  ExecutionAgent (paper routing), ResearchAgent (briefs), TradingWatchdog.
- CLI `omniseed_agent2 trading-analyze / trading-watch / trading-simulate`.

### 13b. Self-awareness (Phase 2) — structured, not sentient
- `src/runtime/introspection.cpp`: SelfModel (capability map marked
  real/partial/absent — live data, real execution, fundamentals are ABSENT),
  GoalTracker ('OKGD' persistence, double-exact priorities — a 4-byte
  priority write corrupted goals on reload, fixed), ActionRationale ('OKRA':
  \"I chose X because Y\" + confidence; resolve_latest → confidence-accuracy
  calibration), PurposeReflection (mission-anchored, grounded in the actual
  log), Metacognition (knowledge gaps, OVERCONFIDENT at gap>0.10/≥5 samples,
  evidence-shrunk strategy confidence: 30 observations ≈ 6% weight).
- Wired: watch loop logs every fill + veto to `state/rationale.bin`;
  `AgentLoop::Config.trading_mode` + `state_fingerprint` flags added.
- CLI: `introspect`, `metacognition`.

### 13c. Hybrid reasoning (Phase 3)
- `src/runtime/cloud_bridge.cpp`: OpenAI/Anthropic-compatible chat over a
  minimal HTTP/1.1 shim (IPv4, redirects, header-carrying request_json).
  **No TLS in the zero-dep runtime**: https requires OMNISEED_CLOUD_PROXY
  (user's local relay) and is refused loudly otherwise — never silently
  downgraded. No key = local mode, silently.
- HybridRouter escalates multi-part research asks only.
- `src/trading/finance.cpp`: NPV, bisection IRR (refuses no-sign-change),
  Black-Scholes (put-call parity tested), FinanceInterpreter sandbox
  (whitelist-only functions, arg-count caps, finiteness checks, hostile
  inputs refused in tests).
- `src/trading/reasoning_tools.cpp`: web_search (DuckDuckGo parse + URL
  unwrap, injectable fetcher), finance_calc, code_interpreter (sandbox by
  construction), document_reader (HTML→text, 20 KB prompt bound).
- CLI: `cloud` (status + one-shot `--prompt` ask).

### 13d. Execution (Phase 4)
- `src/trading/broker_alpaca.cpp`: Alpaca REST v2 — account/positions/orders,
  **PAPER endpoint by default**, keys from env only, local order validation
  BEFORE any network call (symbol/qty/side matrix tested), string-numeric
  JSON parsing (Alpaca style), mockable transport (full offline lifecycle
  test).
- CLI `trading-execute`: live mode = `--live-trading` AND typing exactly
  `I ACCEPT REAL LOSS` (no env/flag shortcut; verified abort exit 2);
  account must be ACTIVE; positions echoed after fills.

### 13e. Verification
- `omniseed_trading`: **2129 checks, 0 failed** (indicators, signal
  determinism + no-look-ahead, CSV, positions, Kelly/stops/dd gates,
  metrics, backtest invariants, RSS/sentiment/entities, consensus, broker,
  introspection persistence + calibration, finance parity, sandbox refusals,
  mock Alpaca).
- ctest **6/6** (platform 206 + real_weights 13 + sides 22 + qat_ab 8 +
  lora + trading 2129). Zero /W4 warnings. Peak RSS of every new CLI
  command: **< 5 MB** (trading stack is budget-free; models not loaded).

### Honest boundary (docs/TRADING_GUIDE.md carries the same)
Loss minimization, not elimination. Backtests prove the past. Real trading
requires broker API + market data + the two-key live gauntlet. Live feed,
SEC/earnings fundamentals, and full vision backbone remain documented gaps.

### NEXT STEPS (updated)
1. **Modal one-shot training** (unchanged, user-triggered):
   `python tools/modal_train.py qat --fresh` → ternary graduation at val
   PPL ≤ 60 (§12 recipe).
2. Trading follow-ups when the user wants depth: streaming quote adapter
   (websocket), SEC/earnings document pipeline into ResearchAgent,
   calibration-driven position-size scaling.
3. Optional: browser console exposes paper-portfolio status via the server.

---

## 14. PHASE-17 — LORA-CHAT DEGENERATE RUN: FORENSICS + GUARDS (2026-09-15)

**The report:** `tools/lora_chat.py` ran 3000 CPU steps → loss 0.000 from
early steps, grad_norm ~1e-6, answer-ppl exactly 1.00 every eval; the exported
`assistant-lora.gguf` turned coherent base output into garbage
("The capital of capital and.") on attach. Suspects were empty answer masks,
random B-init, or a save/load layout mismatch.

**Forensics (probes on the surviving checkpoint, models/lora_chat.pt):**
- Answer mask was CORRECT (~40% of positions, prompts masked); corpus had
  192 real pairs. Not the bug.
- B-init was CORRECT (zeros — step 0 delta is exactly the base). Not the bug.
- Export layout was CORRECT: the synthetic `omniseed_lora` suite already
  validates the GGUF layout + scalar math, and f16-quantizing the checkpoint
  in python reproduced the C++ behavior exactly. Not the bug.
- **Actual cause: 66 epochs of memorization on a 60-pair corpus (3000 steps ×
  batch 4 / 180 examples), then past-convergence drift.** Loss/ppl → 0 on
  TRAIN pairs while holdout-free evals looked "perfect"; after convergence
  AdamW (scale-invariant) kept moving B by ~lr per step on ~1e-6 grads →
  |B| max drifted to **0.30**; the fragile ternary QAT base then turned that
  0.3-magnitude noise into degenerate loops on attach (base itself already
  rambles on chat prompts: "The 1950s, the 1950s …"). Python-attached, the
  same checkpoint memorized verbatim (France answer-ppl 1.00) but
  cross-contaminated facts ("2+2" → "4 sides.").

**Guards now in tools/lora_chat.py:**
1. Corpus stats at start (pair count, mean/min/max answer tokens) — FATAL on
   0 pairs; holdout split (`--holdout`, never trained on).
2. Answer-token-only CE with the masked fraction printed for the first 5
   steps; FATAL if the mask selects 0 tokens.
3. B==0 asserted exactly at init (step-0 delta == base) — plus the runtime
   side: `--steps 0` exports a zero-delta sidecar and the C++ harness proves
   attached logits are BIT-identical to the detached base.
4. Holdout early stop (`--patience`, default 200) + the export is the BEST
   holdout snapshot (not the final step); if holdout never improved, a
   ZERO-DELTA sidecar is exported (attach = proven no-op, can never garble).
5. `|B|` drift warnings (> 0.25 — the degenerate run hit 0.30); checkpoint
   and sidecar carry `format_version` (=2) + `best_step`/`trained_steps`.

**Regression: ctest `omniseed_lora_e2e`** (driver
`tests/lora_e2e/e2e_lora_train.py` + harness `tests/test_lora_e2e.cpp`, 21
checks, auto-skips without the venv/base model): trains 50 steps on
`tests/fixtures/lora_chat_tiny.tsv` (30 pairs) and asserts loss drops > 10%
(no step collapses to ~0), mask fractions strictly partial, holdout answer-ppl
improves vs base, struct-level python GGUF readback is BITWISE-identical to
the checkpoint, the C++ harness sees a healthy attach delta, attach is
deterministic, detach is bit-identical, the zero sidecar is a bit-exact no-op,
and generation on 3 fixed prompts never degenerates (asserted on the TRAINING
base in python).

**Known limitation (honest record):** the trainer trains against
`models/model.safetensors` (PTQ bf16-masters base) while the runtime serves
`rwkv7-0.1B-ternary-qat.gguf` (QAT round-2/3 export, different timestamps →
different ternary weights). Cross-engine logits therefore legitimately differ
(cos ≈ 0.89 measured), and a sidecar that is perfect on its training base is
still off-distribution on the served QAT base. NEXT: add a `--gguf-base` mode
to the trainer (or train on the QAT export) so the sidecar optimizes the
weights the runtime actually serves; until then attach on the QAT base stays
best-effort and the zero-delta fallback is the safe default.

**Retrain recipe (docs/MODAL_QAT_GUIDE.md):** 3000 steps CPU overnight or
minutes on GPU; healthy holdout answer-ppl **1.05–3.0 — exactly 1.00 means
memorized, not good**. Also fixed: generation prompts must be encoded WITHOUT
the trailing eos (a post-eos state is OOD and starts junk; `--sample` and the
tests do this now).
