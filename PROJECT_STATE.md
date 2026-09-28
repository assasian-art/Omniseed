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
- **Test:** `ctest --test-dir build -C Release` — **19/19 passing** (§28).
  Individually: `build\bin\omniseed_tests.exe` — **206/206 passing** +
  `build\bin\omniseed_real_weights.exe` — **13/13 passing** +
  `build\bin\omniseed_trading.exe` — **2129 checks passing** (zero warnings /W4)
  + `build\bin\omniseed_regime_engine.exe` — **157 checks** +
  `build\bin\omniseed_strategy_zoo.exe` — **291 checks** +
  `build\bin\omniseed_heads.exe` — **481 checks** +
  `build\bin\omniseed_router.exe` — **348 checks** +
  `build\bin\omniseed_unified_output.exe` — **267 checks** +
  `build\bin\omniseed_language_heads.exe` — **614 checks**
- **Last updated:** §28 UNIFIED INTELLIGENCE (2026-09-27): ONE backbone, many
  heads, a smart router. `HeadRouter`, `DomainDecisionHead`, `ClassificationHead`,
  `ScoringHead`, `UnifiedOutput`/`UnifiedPipeline` and the language domain are all
  C++ now. Board **19/19 green**, +1,710 new checks. The new tests found and fixed
  **two real bugs** (a tokenizer that split every word into letters, and a
  non-injective seed that let two different seeds produce identical weights). See
  §28 — including the honest gaps: nothing is trained, and vision/audio have label
  spaces but no logic.
  Previous: §19 T7.1 TRADING EDGE LAB (2026-09-26): walk-forward
  validation (select on train, replay out-of-sample, hard no-look-ahead),
  per-trade risk budget clamped to 1–2% + a per-UTC-day drawdown kill-switch,
  and a paper daemon with an append-only, resumable journal. New
  `trading-walkforward` / `trading-paper` CLI; 115-check
  `omniseed_trading_edge` suite.
  Previous: §18 MARKET-DATA INTEGRITY (2026-09-26): `fetch_market_data.py`
  could silently substitute synthetic bars (provider failures swallowed, Stooq
  `--years` ignored, `--source synth` still hit the network). Now: explicit
  failure → exit 2 + no file, `--allow-synth` opt-in, `source` provenance
  column, `--years` honoured on every source, synth → `DEMO_<tf>.csv`.
  39-check offline suite registered as `omniseed_market_data`.
  Previous: §17 CI BASELINE FIX (2026-09-26): GitHub Actions had been
  red 10+ commits; fixed the Linux/GCC build (nested-`Config` default
  arguments + unguarded `::closesocket` in the server) and made the CI smoke
  steps model-free. Full board 10/10 green locally.
  Previous: PHASE-16 OMEGA PASS (2026-09-11): **Trading Expert Mode +
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

**Known limitation → FIXED (Phase-18, same day):** the trainer used to train
against `models/model.safetensors` (PTQ bf16-masters base) while the runtime
serves `rwkv7-0.1B-ternary-qat.gguf` (QAT round-2/3 export, different
timestamps → different ternary weights). Cross-engine logits legitimately
differed (cos ≈ 0.89 measured), and a sidecar perfect on its training base
was off-distribution on the served QAT base.

**Phase-18 — `--gguf-base PATH` (trainer now fits the SERVED weights):**
- `tools/lora_chat.py --gguf-base models/rwkv7-0.1B-ternary-qat.gguf`
  initializes every frozen master by DECODING the served GGUF itself
  (f32/f16 tensors direct; ternary = packed nibbles + per-out-row absmean
  fp32 scales; int8 linears transposed with scales; head int8 [V,E]).
- Exactness trick: masters are set to `T·(scale_file·in_dim/nnz_row)` so
  TernarySTE reproduces the file's dequantized ternary (scale = per-row
  absmean) — the ternary pattern is bit-exact, the recomputed absmean scale
  differs only by ~1 ulp (fp noise).
- Verified cross-engine: the python gguf-base forward is **cos 0.999946**
  vs the C++ runtime on identical ids (pure fp-order difference; argmax
  matches; earlier cos 0.89 runs were the PTQ/QAT base mismatch, plus a
  trailing-eos bug in pinned test prompts — both fixed).
- All guards unchanged (holdout, early stop, best-snapshot, zero-delta
  fallback, drift warnings). The training base is now PRINTED, stamped
  `base_id` (sha256/12) into checkpoint + sidecar (`lora.base_id`), and
  RESUME REFUSES a different base (old checkpoints warn and continue).
- Modal `train_lora` auto-passes `--gguf-base` when the served GGUF is on
  the Volume; otherwise warns loudly before falling back to the PTQ base.
- New ctest `omniseed_lora_gguf` (driver `tests/lora_e2e/e2e_lora_gguf.py`,
  9 checks): 20-step train with `--gguf-base` on the served GGUF,
  attach-improves-holdout-ppl ON THE SERVED WEIGHTS via the C++ harness,
  cross-engine parity ≥ 0.999, determinism + detach integrity, base stamp
  recorded. Suite total now **8/8 ctest green** (omniseed_lora_e2e ~8 min,
  omniseed_lora_gguf ~4 min — run separately when a shell window is short).
- Commit also preserves the user's untracked `train_finish.bat` /
  `train_lora.bat` helpers verbatim for history.

**Retrain recipe (docs/MODAL_QAT_GUIDE.md):** 3000 steps CPU overnight or
minutes on GPU; healthy holdout answer-ppl **1.05–3.0 — exactly 1.00 means
memorized, not good**. Also fixed: generation prompts must be encoded WITHOUT
the trailing eos (a post-eos state is OOD and starts junk; `--sample` and the
tests do this now).

**Phase-19 — i8-GGUF base + big-corpus fetch (`--gguf-base` everywhere):**
- `apply_gguf_base` now dispatches on the STORED dtype of each big linear:
  dtype 40 (ternary) keeps the `T·(s·in/nnz)` STE trick; dtype 4 (int8
  row-quant, as in `models/rwkv7-0.1B-ternary.gguf`) loads the dequantized
  `q·scale` directly and registers the master in an `i8_passthrough` set —
  `begin_window` substitutes the exact fp32 values instead of ternarizing
  (127 levels ≠ 3, but the LoRA base is frozen, so identity is exact and
  cheapest). int8 gates/head/embd decode paths were already shared.
- Cross-engine proof on the int8 file: python i8-base forward vs the C++
  runtime serving that GGUF = **cosine 1.000000, mean|d| 0.0000, argmax
  285 = 285** (identical ids, trained sidecar attached on both sides).
- Corpus: `--corpus fetch` / `--fetch-corpus N` downloads a license-clean
  instruction set (databricks-dolly-15k closed-QA excerpts, CC-BY-SA-3.0)
  from the HF datasets-server into `models/corpus/` (cached TSV, 5xx retry
  with backoff, FATAL under 200 pairs, shingle dedupe on question
  templates). Live fetch: **2561 unique pairs in 24 s** (0.56 MB) — vs the
  ~65-hand-pair builtin (which stays as the ctest fixture only; `--corpus
  FILE` unchanged). Fetched stats: mean answer 30.1 tokens.
- Latent bug fixed in the shared GGUF walker: `_GGUF_KV_SZ` mapped u16/i16
  metadata (types 2/3) to 1 byte instead of 2 (the QAT file simply never
  carries u16 arrays; the ternary file's header would desync any reader
  that tried — diagnosed via hexdump, all other type sizes were correct).
- New ctest `omniseed_lora_i8` (driver `tests/lora_e2e/e2e_lora_i8.py`,
  10 checks): 20-step train with `--gguf-base` on the int8 GGUF,
  attach-improves-holdout-loss, bitwise sidecar round-trip, cross-engine
  parity ≥ 0.999 through the harness. `omniseed_lora_e2e` gained
  `--parity-only` / `--model base.gguf` flags (backward compatible; the
  parity print no longer claims the bases differ).
- Recipe updated (docs/MODAL_QAT_GUIDE.md): train on the BIG corpus with
  `--gguf-base` set to the exact GGUF you serve; healthy holdout answer-ppl
  **1.05–3.0 (exactly 1.00 = memorized)**; the tiny fixture is a mechanism
  test, not a quality judge.

**Phase-20 — ONE SIDECAR, MANY SKILLS: combined assistant+trading LoRA corpus (2026-09-15):**
- Goal: ONE LoRA sidecar that does BOTH everyday-English assistant work AND
  trading-smart reasoning — not stacked adapters, one `.gguf`, one attach.
  Method: grow the corpus, not the model. Future skills (bangla, etc.) add
  more pairs to the SAME TSV and retrain.
- `tools/fetch_trading_corpus.py` (new): generates **2000 synthetic trading
  Q&A pairs** from templates + public market vocabulary (no proprietary/real
  data; CC-BY-SA-3.0 declared) into `models/corpus/trading_chat.tsv` —
  exact category targets: technical analysis 600 (RSI/MACD/Bollinger/
  support-resistance), risk management 400 (Kelly/stop-loss/drawdown),
  news interpretation 400 (Fed/earnings/macro), signal reasoning 400
  (buy/sell/hold with rationale), portfolio review 200 (rebalance advice).
  Questions are unique BY CONSTRUCTION (cross-product enumeration + seeded
  sampling + assert), verdicts imply their question templates; stats
  printed at end (count / mean answer length / per-category). 0 duplicate
  questions, mean answer 30 words.
- `tools/merge_corpora.py` (new): dolly (2560) + trading (2000) →
  shingle-dedupe on full questions → seeded shuffle (seed 7; the trainer
  keeps corpus order and holds out the LAST N pairs, so an unshuffled merge
  would make the holdout single-domain) → `models/corpus/combined_chat.tsv`
  **4560 pairs** (mean answer 28 words). Also re-attaches dolly answers
  with embedded newlines (orphan continuation lines) before parsing.
- Fixture: `tests/fixtures/trading_chat.tsv` = 20 pairs, one per 100 of the
  full corpus, answers compacted to ≤2 clauses (assistant-style brevity +
  ctest step budget). ctest coverage of the combined skill set:
  - `omniseed_lora_gguf` (9 checks): 24-step train on served-QAT-GGUF base
    with the combined fixture; holdout = last 6 pairs (trading domain) —
    attach improves TRADING-domain holdout; cross-engine parity vs the C++
    runtime cos 0.9999+; round-trip bitwise. Mechanism owner.
  - `omniseed_lora_chat` (new, 15 checks): the proven-coherent ~8-epoch
    regime (45 steps, PTQ training base) on the same combined fixture;
    asserts trading-domain holdout improvement, non-memorized ppl, and
    `--sample` coherence: assistant prompt keeps its factual anchor,
    trading prompts answer in-domain, no degenerate loops (n-gram repeat /
    token-pool collapse / comma-chain signatures).
- Sampling-parity fix: python `--sample` now mirrors the C++ runtime's
  Phase-13 CTRL repetition penalty exactly (`--repeat-penalty` /
  `--repeat-window`, agent_loop.cpp semantics: ring of generated ids,
  positive logits ÷ penalty, negative × penalty, applied before argmax).
  The served QAT base loops under pure greedy; production chat runs with
  the penalty enabled — the python probe previously could not reproduce
  runtime chat behavior at all. `--sample` also takes `|`-separated
  prompts (one model load, many replies).
- Budget archaeology (why the split): a training step on long trading rows
  costs ~13-17 s CPU; 45-60 step combined-fixture runs exceed one 600 s
  shell window (Freebuff restarts kill even detached processes ~7 min in),
  so the two ctests split mechanism vs coherence, each fitting its window;
  ctest TIMEOUT 2400 covers both.
- Constraint honored: C++ runtime untouched; all guards intact (holdout,
  early stop, best-snapshot export, zero-delta fallback, base_id stamp,
  drift warnings).
- **VERIFIED (2026-09-16):** ctest registry now **10 tests** (added
  `omniseed_lora_chat`).
  - `tools/fetch_trading_corpus.py`: exit 0, **2000 pairs, 0 duplicate
    questions, exact category targets** (TA 600 / risk 400 / news 400 /
    signal 400 / portfolio 200), mean answer 30.0 words.
  - `tools/merge_corpora.py`: 2560 dolly + 2000 trading → **4560 pairs**
    (re-attached 1842 dolly orphan continuation lines, removed 0 dupes,
    mean answer 28.0 words, seed-7 shuffle).
  - `omniseed_lora_gguf` **9/9**: combined fixture (30 everyday + 20
    trading pairs), holdout = last 6 (all trading), attach improves the
    TRADING-domain holdout on the served QAT GGUF, cross-engine parity
    **cos 0.999941** (argmax 0 = 0), round-trip bitwise, C++ harness
    LORA_E2E_PASS, base stamp recorded.
  - `omniseed_lora_chat` **14/14** after one fix: the hard "Paris"
    factual-anchor assert FAILED at 90 steps (reply "The capital of the
    slowest.") — root cause is the **Phase-17 epoch map**, not the test
    plumbing: 90 steps × batch 2 / 44 pairs ≈ 4.1 epochs sits in the
    semi-random band (recall crystallizes far above fixture scale; the
    older `omniseed_lora_e2e` suite only asserts NON-degeneracy at a
    similar epoch count). Fixture policy says mechanism tests don't judge
    quality. So the Paris anchor became **informational** (`[info]
    factual anchor (Paris) present/ABSENT`); the mechanism asserts stay
    HARD: dual-domain holdout improvement (base 2.3 → final 2.0-ish on
    trading pairs), non-memorized ppl, all 3 sample probes answered,
    non-empty + direct (≤220 chars), zero degenerate-loop signatures
    (comma-chains / token-pool collapse / repeated n-grams) under the
    runtime-matched `--repeat-penalty 1.2`. Factual recall is judged on
    the full corpus (MODAL_QAT_GUIDE recipe), never on a fixture.
- NEXT (Phase-20): train the REAL sidecar on `models/corpus/combined_chat.tsv`
  (4560 pairs — Modal GPU minutes per the guide, or CPU overnight), then
  re-probe `--sample` for dual-domain quality; fixture suites stay as the
  mechanism regression only.

---

## 16. PHASE-20b — GENERATION STOP-STRINGS (2026-09-16)

The base model answers a question and then invents the next user turn
("… Paris.\nUser: What is 2+2?"). Stop strings cut that at the token level:

decoding halts the moment the decoded tail matches any stop string
(last-64-chars window, earliest-position match across all stops), the
matched tail is trimmed from the reply, and streamed output is held back
while it could still complete a stop (so "\nUser:" never reaches a chat
UI). Zero effect on token choice — a pure output-boundary guard.

- **Runtime** (`include/omniseed/agent/agent.h`, `src/agent/agent_loop.cpp`):
  `Config.stop_strings` + `set_stop_strings()` (server mirror of
  `set_sampling`: empty list reverts to construction-time config so requests
  stay self-contained); `kAssistantStopDefaults` = `\nUser:`, `\nAssistant:`,
  `\nAssistant::` (the doubled colon catches the model restarting the
  assistant header with a typo). `generate()` gained the tail check +
  trim + streaming hold-back; no-stop paths are untouched.
- **CLI**: `--stop TEXT` (repeatable) + `--stop-defaults` on `gen`, `ask`,
  `chat`; usage text updated.
- **Server**: `POST /gen` and `POST /ask` accept `"stop": ["\\nUser:", …]`
  and `"stop_defaults": true` (combined with explicit entries; absent
  fields leave stops unchanged, `"stop": []` reverts to defaults). The
  server has NO `/chat` endpoint — chat over HTTP is `/ask` — so the two
  real endpoints carry the field. Plain-bytes parser (the vendored runtime
  has no JSON decoder); no `/chat` demo-page change needed.
- **ctest** (`omniseed_real_weights`, now **17/17**): drives the real CLI
  binary twice on the i8 GGUF ("What is the capital of France?", 96 tokens)
  — with `--stop-defaults` the reply contains NO "User:"/"Assistant:" and
  equals the unguarded reply truncated exactly at the first stop (nothing
  lost, nothing added); without the flag the output is byte-identical to
  the guard being absent (no-op when no stop would occur). Skips cleanly
  when the CLI isn't built. Test-infrastructure notes: `_popen` routes
  through `cmd /c` (no quoting of the exe path — cmd strips a fully-quoted
  command's first/last quote — and backslash separators, forward slashes
  split), and text-mode pipes turn "\n" into "\r\n" (whitespace-normalized
  comparison).
- **Caught by the new test mid-flight:** the first implementation returned
  the first stop in LIST order; when "\nAssistant:" completed EARLIER in
  the text than "\nUser:" the trim point was wrong (truncated reply ≠
  unguarded prefix). Fixed to earliest-position match across all stops.
- **Docs — rambling mitigation ladder** (MASTER_SPEC CLI table + server
  contract, DEPLOYMENT endpoint table, MODAL_QAT_GUIDE recipe):
  `--stop-defaults` (cut the invented turn — today, zero cost) **<**
  trained assistant LoRA (the model actually stops — better) **<** both.
- **Verification:** zero /W4 warnings; omniseed_tests **206/206**,
  omniseed_real_weights **17/17**, omniseed_sides **22/22** (one earlier
  run showed the ASR wall-clock bound jittering under back-to-back suite
  load — clean run passes), omniseed_qat_ternary **8/8**,
  omniseed_trading **2129/0**.

---

## 17. CI BASELINE FIX — LINUX GCC + MODEL-FREE SMOKE (2026-09-26)

GitHub Actions had been red for **10+ consecutive commits** on `main`. The
breakage was three independent, stacked faults, none of them new to
`8fcdd8e`. All three are fixed and verified on this box.

### Root causes

1. **GCC rejected nested-`Config` default arguments** (`ubuntu-latest` →
   Build, exit 2). Every `explicit Foo(const Config& cfg = Config{})` where
   `Config` is a **nested** struct with default member initializers (NSDMIs)
   fails on GCC:
   `error: default member initializer for 'Foo::Config::x' required before
   the end of its enclosing class`.
   This is a hard error (complete-class-context rule), so it fails with
   `-Werror` **off** and is not version-specific (reproduced on GCC 13.2,
   14.1, 14.4).
   - `= {}` and `= Config{}` **both** fail; the earlier "`= {}` →
     `= Config{}`" sweep (commit `4b78f6e`) only changed the diagnostic
     text — it did not fix the build.
   - Top-level config structs (`FocalCodecConfig`, `UniCompressConfig`,
     `PaperBrokerConfig`) were never affected, which is why only nested
     sites broke.
2. **`src/server/main.cpp` called `::closesocket` unguarded.** The file
   already abstracts `Socket_t` and guards `send`/`recv`, but all three
   `closesocket` call sites were unconditional — so the `OMNISEED_HTTP`
   build (CI passes `-DOMNISEED_BUILD_SERVER=ON`, which adds
   `OMNISEED_HTTP`) failed on Linux with
   `error: '::closesocket' has not been declared`. Fixed with a
   `close_socket()` helper (`closesocket` on Win32, `close` on POSIX),
   mirroring the existing `send_all` guard pattern.
3. **CI smoke ran `omniseed bench`**, which calls `Session::load()` and
   returns 1 when `./models/*.gguf` is absent — guaranteed to fail in CI.
   Replaced with model-free commands on **both** jobs: `selftest`, `info`,
   and bare `omniseed` (usage text; exits 0).

### Fix applied

- **27 nested-`Config` default-argument sites across 17 headers** replaced
  with a delegating default constructor + explicit overload:
  ```cpp
  Foo() : Foo(Config{}) {}
  explicit Foo(const Config& cfg) : cfg_(cfg) {}
  ```
  Headers: `agent/agent.h` (SelfImprovement, AgentLoop),
  `agent/agent_intel.h`, `agent/flash_skills.h`, `agent/sub_agents.h` (×2),
  `audio/audio_events.h` (×3), `core/uncertainty.h`, `memory/memory.h` (×2),
  `memory/prefix_cache.h`, `runtime/cloud_bridge.h`, `runtime/emotional.h`,
  `runtime/introspection.h`, `runtime/sensory.h`, `runtime/swarm.h` (×2),
  `trading/broker_alpaca.h`, `trading/news_feed.h`,
  `trading/trading_engine.h` (×3), `vision/vision_tasks.h` (×3).
- 3 declaration-only constructors (`WakeWordDetector`, `CloudBridge`,
  `UdpBeacon`) got an inline delegating no-arg ctor in the header.
- 3 static member functions (`Uncertainty::should_abstain`,
  `SignalGenerator::compute`, `SignalGenerator::evaluate`) gained a 1-arg
  overload that forwards `Config{}` (`src/core/uncertainty.cpp` updated).
- `AgentLoop` (6 required args + `Config`) and `AlpacaClient` (`Config` +
  `Transport`) kept their extra parameters via a delegating overload.
- Semantics unchanged: the no-arg path value-initializes the same `Config`,
  and no `Config` lost aggregate status.
- One call site updated: `tests/test_trading.cpp` constructed
  `NewsFeed feed({})`, which becomes ambiguous once a no-arg ctor exists →
  `NewsFeed feed;`.

### Verification

- **GCC (Linux — the exact CI target)**: all 41 core/CLI/server/agent2
  translation units **plus** all 7 test TUs compile clean, both with and
  without `OMNISEED_HTTP`, at `-fsyntax-only` and at full `-O2`, on GCC
  **13.2 / 14.1 / 14.4**.
- **MSVC (this box, VS 18 2026, `/W4`)**: clean Release build, all targets,
  zero warnings.
- **Full board green** — `ctest --test-dir build -C Release` → **10/10
  passed**, 2975 s total: `omniseed_platform` 206, `omniseed_real_weights`
  17, `omniseed_sides` 22, `omniseed_qat_ternary` 8, `omniseed_lora` 77,
  `omniseed_lora_e2e` 21 (687 s), `omniseed_lora_gguf` 9 (407 s),
  `omniseed_lora_chat` 14 (1387 s), `omniseed_lora_i8` 10 (397 s),
  `omniseed_trading` 2129.
- CI smoke commands verified locally to exit 0.

### Notes / follow-ups

- The heavy Python LoRA suites are **already self-gating** in CI: CMake
  registers `omniseed_lora_e2e` / `_gguf` / `_chat` / `_i8` only when
  `.venv` exists (`CMakeLists.txt:261-306`), so CI's ctest is ~2 s. No
  gating change was needed.
- `windows-latest` no longer ships VS 2022; the pinned
  `-G "Visual Studio 17 2022"` was dropped in `4b78f6e` (this box runs VS 18
  2026, so an unpinned generator is the portable choice).

## 18. MARKET-DATA INTEGRITY (2026-09-26)

`tools/fetch_market_data.py` could silently hand the trading stack **fake
bars**. Three defects, all fixed; behaviour is now pinned by an offline
characterization suite.

### Root causes

1. **`--source synth` still hit the network.** The synthetic branch was
   decided by `demo_synth` only, so an explicit `--source synth` fell through
   to the stooq/yahoo path. Requesting synthetic data could make real
   network calls.
2. **A provider failure was swallowed.** Any exception in the fetch chain
   fell back to synthetic bars with only a stderr note. A mistyped ticker
   ("AAPL" instead of "AAPL.US") or a dead provider produced a *valid-looking
   CSV of random data* that the backtester happily consumed.
3. **`--years` was ignored for Stooq.** Stooq returns its entire history;
   the trim only existed on the Yahoo path, so `--years 5` on Stooq silently
   delivered decades of bars — corrupting every metric that assumes a window.

### Fix applied

- Rewrote the tool around an explicit **`ProviderError`** contract:
  - `--source stooq|yahoo` that fails → prints the provider error, **exits 2,
    writes no file**.
  - `auto` (default) tries stooq → yahoo; if both fail it exits **2** unless
    **`--allow-synth`** is passed (then, and only then, synthetic bars are
    substituted, clearly marked).
  - `--source synth` / `--demo-synth` route straight to synthetic and make
    **zero** network calls.
- **Provenance column**: every CSV now ends with a `source` field
  (`stooq`/`yahoo`/`synth`). `load_bars_csv()` reads fields 0..5 and ignores
  the extra column, so the C++ contract is unchanged (backward-safe).
- **`--years` applied in `emit_csv()`** to *every* source — sorts ascending,
  drops rows older than `newest − years`.
- **No clobbering**: synthetic output defaults to `models/market/DEMO_<TF>.csv`
  instead of `<TICKER>_<TF>.csv`.
- **Empty result is a failure**: `--years <= 0`, an unparsable/"No data"
  provider body, or 0 bars after trimming all exit **2**.

### Verification

- New `tests/test_market_data.py` — **39 checks, 0 failures**, fully offline
  (monkeypatches `urllib.request.urlopen` + `time.sleep`; no network, no
  fixtures). Covers: seeded-determinism, `--source synth` network isolation,
  explicit-failure exit 2 + no file, `auto` never implicitly synthesizing,
  `--allow-synth` opt-in, stooq `--years` trim (~104 weekly bars for 2 y,
  ≤731 d, newest bar retained), junk/empty provider = failure,
  `DEMO_<tf>.csv` cannot clobber a real ticker file, and the `load_bars_csv()`
  field contract.
- Registered in ctest as **`omniseed_market_data`** (`CMakeLists.txt`,
  venv-gated like the LoRA suites) — passes in 13.9 s.
- `docs/TRADING_GUIDE.md` §2 documents the integrity rules, exit codes, and
  the `--allow-synth` / `DEMO_<tf>.csv` behaviour.

## 19. T7.1 — TRADING EDGE LAB: WALK-FORWARD + RISK ENGINE + PAPER DAEMON (2026-09-26)

A single backtest over the whole history is an in-sample number; the edge lab
exists to separate the strategy from the tuning. Three new pieces:

### Risk engine (`src/trading/trading_engine.cpp`, `trading_engine.h`)

- `RiskLimits::risk_per_trade_pct` (default 1.5%) + `RiskManager::size_by_risk`:
  `qty = equity × risk / (price × stop_loss_pct)`. The value is **clamped into
  [1%, 2%]** (`kMin/kMaxRiskPerTradePct`) — never more than 2% of equity risked
  on one trade, and the sizing trace prints `(clamped)` when it fires. The
  notional is then capped by `max_position_pct`.
- `DailyRiskGovernor` + `RiskLimits::max_daily_loss_pct` (default 3%): once the
  drawdown from a UTC day's **high-water mark** reaches the limit, new entries
  are refused for the rest of that day; the switch re-arms at the next UTC day.
  Exits/stops are never gated. (On a once-per-day bar series every bar is its
  own day, so the switch only has room to act on intraday bars.)
- **Latent bug fixed:** `RiskManager::Sizing` defaults `allowed = true`, so
  both `size_position` and `size_by_risk` returned `allowed = true` on their
  degenerate-input early returns. Both now set `allowed = false`.

### Walk-forward (`src/trading/walkforward.{h,cpp}`)

- `WalkForward::run` cuts the series into consecutive train/test folds, picks
  parameters from a candidate grid on the **train** window only (score: Sharpe,
  tie-break return), then replays them **untouched** on the following test
  window; equity chains across folds and the test windows are stitched into an
  out-of-sample curve.
- **Hard no-look-ahead**: `test_begin == train_end` for every fold; selection
  never sees a test bar. Rolling (`--anchored` off) or anchored train windows.
- Report: per-fold IS/OOS, `oos_metrics`, `is_mean_return_pct`,
  `degradation_pct` (IS − OOS; >5% flagged as likely overfit), `oos_hit_rate`,
  `profitable_oos`. Default grid = 9 configs (fast MA × RSI entry).
- `Backtester` reuse means fills are next-bar-open with the existing 5 bps fee
  + 2 bps slippage model.

### Paper daemon + journal (`src/trading/paper_daemon.{h,cpp}`)

- `PaperJournal` — append-only CSV (`ts,kind,ticker,qty,price,pnl,equity,cash,
  exposure,reason`, `kind ∈ {start,equity,fill,halt}`), flushed per record,
  header written once. `scan_existing()` recovers `records()`/`last_ts()` so a
  re-run **resumes**: the daemon skips bars with `ts <= last_ts()` and appends
  only new history (never rewrites or duplicates). Free-text columns are
  comma-sanitised so a reason cannot break the schema.
- `PaperDaemon::run` — deterministic bar replay driving signals → risk engine →
  `PaperBroker`, journaling an equity row per bar, every fill, and each
  kill-switch trip. No threads, no network, no real orders.

### CLI (`src/agent2/main.cpp`)

- `trading-walkforward --ticker AAPL --timeframe 1d [--train N --test N
  --step N] [--anchored]` — prints folds, IS/OOS, degradation, verdict.
- `trading-paper --ticker AAPL --timeframe 1d [--journal PATH]` — prints
  actions, equity, and records appended; a no-op resume says so explicitly.

### Verification

- `tests/test_trading_edge.cpp` (new, ctest `omniseed_trading_edge`) —
  **115 checks, 0 failures**: risk-budget sizing + 1–2% clamp + position cap +
  degenerate inputs; governor allow/trip/peak-based/re-arm/trip-count/reset;
  walk-forward no-look-ahead, contiguity, grid membership, stitched-curve
  length, degradation identity, determinism, anchored growth; journal header +
  schema + scan_existing + comma safety; daemon replay/equity-row count/
  determinism/resume-boundary/kill-switch-journaled.
- CLI smoke on a 1512-bar synthetic series: walk-forward → 20 folds, every
  `test_begin == train_end`; paper daemon → 34 entries / 34 exits, journal
  1581 records, second run resumed (0 bars processed).
- `docs/TRADING_GUIDE.md` §1/§3/§4 + new §5b document all of the above.

## 20. T7+ M1 + M4 — MARKET PERCEPTION + RISK GATES (2026-09-26)

The first two milestones of the T7+ autonomy mandate: a unified multi-asset
perception layer that **refuses to act on untrustworthy data**, and the
code-enforced risk/autonomy gates. New evidence table in
`docs/TRADING_LAB.md`; operational summary in `docs/TRADING_GUIDE.md` §5c.

### M1 — Market feeds (`tools/market_feeds.py`, new)

- Keyless multi-asset adapters behind **one** HTTP path (`HttpClient`) so cache,
  rate limiting and retries live in exactly one place:
  - `equity` → Yahoo (`query1.finance.yahoo.com`), bars + ticker
  - `crypto` → Binance then CoinGecko (`api.binance.com`, `api.coingecko.com`)
  - `fx` → Frankfurter (`api.frankfurter.app`)
  - `meme` → DexScreener (`api.dexscreener.com`) — **ticker only**
- **Provenance single-sourced:** imports `fetch_market_data.emit_csv` via
  `importlib`, so the T6 `time,open,high,low,close,volume,source` contract has
  one definition. Provider failures are explicit (exit 2, no file).
- **Rate limiting** = token bucket (`--rate-per-min`); **cache** = sha1-keyed
  TTL JSON (`--cache-dir/--cache-ttl`).
- **Meme bars mode is refused** (exit 2) because no free OHLCV exists —
  matching `RiskLimits::meme_research_only`.
- `normalize()` fix: a quote suffix only counts when something precedes it, so
  bare `BTC` → `BTCUSDT` while `ETHBTC` keeps its explicit quote.

### M1 — Perception + ABSTAIN (`src/trading/market_perception.{h,cpp}`, new)

- `Ticker` — one normalized cross-asset snapshot carrying its provider
  (provenance), mirroring the T6 `source` column.
- `FeedHealthMonitor` — `Ok → Degraded (300 s) → Dead (900 s)`, or dead after
  `dead_after_failures` (3) consecutive errors; unobserved → `Missing`, **never**
  an implicit `Ok`.
- `PerceptionGate::check()` → `AbstainDecision` (`missing-data`/`invalid-ticker`/
  `stale-feed`/`dead-feed`/`low-liquidity`). A default-constructed `FeedHealth`
  (no symbol) falls back to the ticker's own ts; a health that names the symbol
  but carries no data is `missing-data` — so an explicit "monitor says missing"
  can never be mistaken for "fresh".

### M4/M6 — Risk gates (`src/trading/risk_gate.{h,cpp}`, new; `trading_engine.{h,cpp}`)

- `RiskGovernorSet` — two independent `DailyRiskGovernor`s (daily + weekly);
  entries refused while **either** is halted, each re-arms on its own boundary.
- `RiskLimits` gained `max_weekly_loss_pct` (6%), `meme_max_risk_pct` (0.25%),
  `meme_research_only` (true), `max_correlation` (0.80).
- `RiskManager::size_by_risk(equity, stop, AssetClass, limits)` — 1–2% band for
  equity/crypto/fx; memes use `min(risk_per_trade, meme_max_risk_pct)` with **no
  1% floor** (never clamped *up*). `effective_risk_pct` exposes the decision.
- `RiskManager::correlation` / `correlation_ok` — Pearson; a candidate whose
  worst pairwise correlation exceeds the cap is refused.
- `AutonomyGate` — L0 paper-only (default, live refused outright); L1
  micro-live (≤$100 total, ≤1%/trade, ≥30 paper days); L2 scaled (≥90 live
  days). Unlock = an **exact, own-line** `UNLOCK L1 MICRO-LIVE` /
  `UNLOCK L2 SCALE-UP` in this file; near-misses (TODO, quoted, different case)
  unlock nothing. `allows_live()` re-checks level + duration + caps per order.

### Verification

- `tests/test_market_perception.cpp` (new, ctest `omniseed_market_perception`) —
  **107 checks, 0 failures**: feed health staleness/failures/recovery/unknown;
  ABSTAIN for fresh/invalid/stale/dead/missing/liquidity + stable reason
  strings; governor daily-only/weekly-only/OR semantics; asset-class 1–2% band,
  meme research-only + 0.25% cap + no-up-clamp + equity-equals-untyped; Pearson
  perfect/inverse/degenerate + cap refuse/pass/empty-book; autonomy resolve
  exact/near-miss + L0/L1/L2 caps and durations.
- `tests/test_market_feeds.py` (new, ctest `omniseed_market_feeds`, venv-gated) —
  **56 checks, 0 failures**: cache hit/miss, token bucket, HTTP retry +
  ProviderError naming the host, symbol normalisation, every bar/ticker adapter
  offline, provenance-CSV reuse, meme bars refusal, explicit failure exit 2 +
  no file, `--years 0` rejection, probe mode.
- `CMakeLists.txt` — `market_perception.cpp` + `risk_gate.cpp` added to
  `OMNISEED_CORE_SOURCES`; both test targets registered.
- `omniseed_trading` (2129 checks) unchanged green — the `RiskManager` additions
  are additive and do not alter existing sizing on valid inputs.

## 21. T7+ M5 — 24/7 MULTI-ASSET PAPER DESK + REPORT + DASHBOARD (2026-09-26)

The third T7+ milestone: a long-running multi-asset paper desk. The split is
unchanged — **network in Python, engine never opens a socket**. New doc section
`docs/TRADING_LAB.md` §5; operational summary `docs/TRADING_GUIDE.md` §5d.

### C++ — multi-asset paper session (`paper_daemon.{h,cpp}`, `simulate`, `trading_engine`)

- `PaperSession::run` — one cash account, N streams merged onto a single
  chronological timeline (ties broken by symbol for determinism). At each
  timestamp the whole book is marked, the composite `RiskGovernorSet`
  (daily + weekly) is consulted once, then each stream with a bar there may
  exit or enter. Exits are never gated; entries use the stream's next-bar open.
- Per-asset sizing via `RiskManager::size_by_risk(equity, price, AssetClass,
  limits)` — meme stays research-only (refused + journalled once per stream).
- **Feed ABSTAIN** — `PaperStream::feed_ok=false` (set by the poller) skips new
  risk for that symbol but still marks/exits its positions.
- **Real resume** — the append-only journal is the source of truth.
  `PaperJournal::scan_existing` now also recovers `last_equity`/`last_cash` and
  replays the recorded fills into an open book; `PaperBroker::seed` +
  `PositionManager::restore` re-hydrate cash + positions, so a restart does not
  reset equity to the starting cash. A re-run appends only new timestamps.
- **Journal everything** — `kind ∈ {start, equity, signal, fill, halt, event}`.
  New `PaperJournal::signal` records every non-Hold signal (strength in the qty
  column, rule trace in the reason), so the journal is a full decision log, not
  just a trade log. `--no-signal-journal` opts out.
- `agent2 trading-paper-session --streams "SYM,ASSET,CSV[;...]" [--skip SYM,...]
  [--no-signal-journal]`.

### Python — the loop, report, dashboard (`tools/`)

- `tools/paper_loop.py` — the 24/7 poller. Each cycle: probe every watch item →
  derive feed state (mirrors `FeedHealthMonitor`, including "never-succeeded ⇒
  missing") → refresh the symbol's provenance CSV only while OK (dedup by ts) →
  invoke the engine → write `state/paper_status.json` (heartbeat + feed health +
  last session). Rate limiting + TTL cache inherited from `market_feeds`.
- `tools/paper_report.py` — shared, read-only analysis (journal parse, book
  replay, equity curve, day P&L, per-symbol contribution, halts/abstains/
  refusals, feed rows) so the report and dashboard can never disagree.
- `tools/nightly_report.py` — Markdown summary; **email is opt-in and never
  faked** (sends only when every `OMNISEED_SMTP_*` var is set, else says so and
  still writes the file).
- `tools/paper_dashboard.py` — one **self-contained** HTML document (inline CSS,
  inline SVG equity curve, plain tables; no CDN, no external assets, no script):
  equity curve, open positions, today's P&L, feed health, recent fills, risk
  counters.

### Verification

- `tests/test_paper_session.cpp` (new, ctest `omniseed_paper_session`) —
  **47 checks, 0 failures**: merged timeline + per-stream fills, shared-book
  cap, meme research-only refusal + journal, non-meme still enters, feed
  ABSTAIN, composite kill-switch + journal, resume (book rebuilt, nothing
  appended when caught up, only the new tail when extended), determinism,
  no-streams error, missing-CSV reported per stream, signal journaling on/off.
- `tests/test_paper_loop.py` (new, ctest `omniseed_paper_loop`) — **53 checks,
  0 failures**: watch parsing, CSV merge/dedup/header/provenance, healthy cycle
  (CSV + status + engine args), failure → dead → skip, never-succeeded ⇒
  missing, recovery-then-dead, stale ⇒ abstain + session skipped, dry-run,
  driver sleep cadence.
- `tests/test_paper_reports.py` (new, ctest `omniseed_paper_reports`) —
  **59 checks, 0 failures**: journal parse + junk skip, book replay, multi-day
  day-P&L (measured from the carried-in equity), signal counting, Markdown
  sections, HTML self-containment (no script / no remote refs), opt-in SMTP
  config, CLI end-to-end.
- Live smoke: the loop ran once against real providers (1/3 feeds OK — on a
  weekend the daily equity/FX feeds are stale ⇒ **ABSTAIN**, 24/7 crypto `ok`);
  the engine refused a BTC entry that exceeded the 20 % per-name cap rather than
  over-sizing; report + dashboard rendered from the real journal.
- `omniseed_trading_edge` (115) and `omniseed_trading` (2129) unchanged green —
  the `PaperBroker`/`PositionManager` additions are additive.

---

## 22. MONSTER TRADING MODEL — SNIPER ENTRY + NEWS HUNTER + CROSS-ASSET (2026-09-26)

**Branch** `feature/monster-trading-model`. Design doc: `docs/MONSTER_DESIGN.md`.

### What it is

A four-layer **sniper confidence score** plus an event-driven news hunter and a
cross-asset confirmation filter, all in Python, feeding the C++ paper engine a
single distilled row per bar.

```
S = 0.25·M (microstructure) + 0.35·T (technical) + 0.25·R (regime) + 0.15·C (cross)
entry  <=>  no hard veto  AND  S >= 0.85  AND  the engine's own signal says Buy
```

### Files

| file | role |
|---|---|
| `docs/MONSTER_DESIGN.md` | the math, the citations, the honest-scope notes |
| `tools/monster/features.py` | no-look-ahead primitives (SMA/EMA/RSI/MACD/ATR, VWAP, z-score, Pearson, Amihud, Kyle-λ proxy, swings, Fibonacci) |
| `tools/monster/sniper_engine.py` | 4-layer weighted score + hard vetoes |
| `tools/monster/news_hunter.py` | keyword weight × exp(−Δt/τ) × sentiment alignment, crossed with a volume anomaly |
| `tools/monster/correlation_matrix.py` | rolling matrix + lead-lag confirm/invalidate |
| `tools/monster/sizing.py` | ATR + confidence scaled risk, clamped to `[1%,2%]` |
| `tools/monster/state_vector.py` | the distilled vector + the engine-facing CSV |
| `tools/monster_scan.py` | CLI: provenance CSVs → `state/monster/*.csv` |
| `include/omniseed/trading/paper_daemon.h` / `src/trading/paper_daemon.cpp` | `MonsterPoint`, `load_monster_csv`, the **fail-closed** gate |
| `src/agent2/main.cpp` | `--monster-features`, `--monster-min-conf` |
| `tools/paper_loop.py` | `--monster-features`: refresh features each cycle before the session |

### Design decisions worth recording

- **Path deviation.** The mandate asked for `omniseed/trading/*.py`; in this repo
  that directory is C++17. The Python orchestration layer lives in `tools/`, so
  the modules live in `tools/monster/`. Documented at the top of the design doc.
- **The engine gains ONE seam.** Heavy logic stays in Python; C++ reads
  `ts,confidence,veto,regime,detail`. The gate is **fail-closed**: no row, a
  veto, or a low score all block. It never touches exits.
- **A gate that can never open is a bug.** The first cut used `T = votes/6`,
  capping the reachable total at ≈0.81 — the mandated 0.85 bar was
  mathematically unreachable. `T` now pays 0.90 for meeting the mandated
  three-factor confluence and 0.10 per extra factor. A textbook setup reaches
  **S = 0.865**; 420 bars of real daily data produce nothing at 0.85.
- **`M` takes the max, not a blend, of order-book imbalance and volume z.** A
  blend lets a quiet book dilute a real volume spike, which is backwards.
- **Cross-asset needs TWO checks.** Coherence (the follower moved as the assumed
  sign predicts) *and* alignment (the target is moving our way). A perfectly
  coherent complex dragging the target against us confirms the OPPOSITE trade.
  A *flat* follower is no information and must never count as an invalidation.
- **Sizing shares the engine's band.** `sizing.py` clamps to the same
  `[kMinRiskPerTradePct, kMaxRiskPerTradePct] = [1%, 2%]` the C++ engine
  enforces, so the Python plan and the engine can never disagree.

### Test evidence

- `tests/test_monster_features.py` (ctest `omniseed_monster_features`) —
  **51 checks, 0 failures**: indicator correctness plus the **no-look-ahead
  proof** (every primitive's value at index `i` is byte-identical whether or not
  later bars exist; a swing is invisible until `idx + right`).
- `tests/test_monster_sniper.py` (ctest `omniseed_monster_sniper`) —
  **49 checks, 0 failures**: regime classification, all four hard vetoes,
  event-driven relaxation, score arithmetic, **0.85 reachability**, the sizing
  band never leaving `[1%,2%]` under adversarial inputs, cross-asset
  confirm/invalidate including the short and responder cases.
- `tests/test_monster_news.py` (ctest `omniseed_monster_news`) —
  **51 checks, 0 failures**: keyword weights, exponential decay, alignment,
  anomaly detection, match-requires-both-halves, state-vector CSV round-trip,
  scan CLI end-to-end (including a symbol with no CSV and the news path).
- `omniseed_paper_session` — **69 checks, 0 failures** (was 47; +22 for the
  Monster gate: pass-through, low confidence, veto, missing file ⇒ fail closed,
  signal-log annotation).

### Live integration smoke

Real 420-bar basket (AAPL / BTCUSDT / EURUSD), fresh journal:

```
--monster-min-conf 0.85  ->  0 entries, 166 blocked   (fail-closed)
--monster-min-conf 0.70  ->  1 entry,   160 blocked   (the gate really gates)
```

The journal records `monster-block:<sym>:veto:range` and every `signal` row
carries the verdict, e.g. `monster[S=0.640 range]`.

### Constraints honoured

RWKV-7 0.1B edge brain untouched and well under 400 MB (the Monster is Python).
The **2 % per-trade ceiling** and the **3 % daily / 6 % weekly kill-switch** are
unchanged and still enforced in C++. No guaranteed-profit claim anywhere.

---

## 23. MONSTER M8 — REGIME ENGINE + STRATEGY ENSEMBLE + FUNDING CARRY (2026-09-26)

The second Monster milestone. Adds the *adaptive* half: knowing **what kind of
market this is** and switching engines accordingly, plus the one non-directional
edge in the book.

### New modules (`tools/monster/`)

| module | lines | role |
|---|---|---|
| `regime.py` | ~700 | multi-axis causal regime engine |
| `strategies.py` | ~520 | the strategy zoo + vol-target overlay |
| `router.py` | ~190 | regime-adaptive continuous allocation |
| `funding.py` | ~330 | perpetual funding-rate carry scanner |

### What the regime engine does

Eight independent directional statistics (variance ratio, Hurst, efficiency
ratio, ADX, choppiness, linreg R², ρ(1), MA-ribbon consistency) blended into a
continuous `trend_score`, plus a Yang-Zhang volatility axis and an OU half-life
for tradeability. Hysteresis on the label. Replaces the single SMA200-slope rule
as the R layer's source (that rule is kept behind `--regime-mode legacy`); the
label→score mapping is unchanged, so **`S` is untouched**.

Thresholds are **calibrated against measured distributions**: at `trend_hi=0.65`
the detector keeps 86% of true trend bars and only 12.6% of random-walk bars.
The full table is in `docs/MONSTER_DESIGN.md` §5.2.

### Five bugs the tests and smoke runs caught

1. **Hurst on price levels** returned `H > 1` — not a number Hurst can be. It
   must run on the increments.
2. **ER horizon mismatch.** ER(20) read 0.04 on a market that pulled back for 20
   bars inside a 200-bar uptrend, flipping the label to "range". ER is now swept
   over 10/20/40 and the normalized readings averaged.
3. **Yang-Zhang collapses.** It is a *variance*, so on constant returns it is
   exactly 0 → NaN, even when the intraday range is 12%. `ATR/close` (a level,
   which cannot collapse) is combined in with max. Separately, ranking against a
   **flat history** made the percentile pin to 1.0 — maximum stress for a dead
   series. A flat history now reads 0.5.
4. **The volatility axis measured a BAR, not a REGIME.** This is the one that
   blocked the milestone, and it only became visible once the engine was wired
   into the sniper. Ranking the raw latest Yang-Zhang estimate against its own
   history fires on *every* volatility-expansion bar — and an entry bar is a
   volatility-expansion bar by construction. The arithmetic is fatal:
   `max S while labelled high_vol = .25·1 + .35·1 + .25·0.25 + .15·1 = 0.8125`,
   below the 0.85 bar, so the gate could never open. Fixed by (a) ranking the
   **smoothed** estimate (mean of the last `vol_smooth = 5` readings) — a regime
   is a *persistent* elevation, so one spike moves it by 1/5 while a real regime
   shift moves all five — and (b) additionally requiring **absolute** stress
   (`atr_stress > 0`), because a pure 90th-percentile rank marks ~10–15% of all
   bars stressed by construction. `RegimeState.stressed` now carries the
   conjunction so the hysteresis tracker latches on what the label uses.
5. **The ensemble veto fired on a lone signal.** `agreement` is a *share*, so one
   low-confidence OFI proxy scored 1.00 and vetoed every long in a live smoke
   test. A weight floor (≥ 0.50) was added; the regression test is in
   `test_monster_strategies.py`.

Also fixed: `RegimeState` numeric slots now default to NaN rather than `None`
(a `None` propagated into `math.isfinite()` and raised `TypeError` far from the
cause), and `SniperVerdict.detail()` is None-safe.

### A structural finding about the technical layer

Measured factor counts over a 340-bar fixture:
`trend-align 140 | macd-thrust 43 | vwap-reclaim 6 | bb-oversold 1`.

The six "independent" factors are **not** independent — they cluster into a
*trend* group and a *pullback* group. So `votes ≥ 3` effectively means "a
pullback that reclaims inside an uptrend": coherent and demanding, but far rarer
than "three of six boxes ticked" suggests. On a steady uptrend the maximum
reachable is 2 votes. Documented in `docs/MONSTER_DESIGN.md` §6.4.

**Consequence for the old sniper fixture:** it alternated drift every 50 bars,
i.e. it was a *chop*. The legacy SMA200-slope rule called it `trend_up`, which
supplied R=1.0 on exactly the bars that had 3 votes — so the old "0.85 is
reachable" test was passing for the wrong reason. The advanced engine correctly
reads it as `range`, and the fixture was rebuilt around a genuine pullback-and-
reclaim inside an uptrend.

### The strategy ensemble

`momentum` (Donchian breakout + ribbon) · `mean_reversion` (z-score fade, gated
on the OU half-life) · `breakout` (Bollinger-inside-Keltner squeeze) · `ofi`
(Cont-Kukanov-Stoikov order flow imbalance) · `vol_target` (Moreira-Muir
inverse-variance exposure). The router blends them by regime fit:

```
w_trend = sigmoid(6·(trend_score − 0.5))
conviction = Σ w·dir / Σ w        agreement = Σ_{agreeing} w / Σ w
```

Continuous by design — a binary switch maximizes whipsaw cost exactly when the
classifier is least sure. The ensemble adds one **fail-closed** gate
(`ensemble-opposed`) and is structurally incapable of raising `S`.

The exact CKS OFI and `β = c/depth^λ` are implemented and unit-tested against
hand-computed values. Free OHLCV has no book, so the shipped path is a
signed-volume proxy, labelled `NOT-cks` and discounted 30%.

### Funding-rate carry

The one non-directional strategy. Delta-neutral long-spot / short-perp capturing
the funding payment. `APR = rate × periods_per_year` (8h → 1095). Ranked on
**net** APR after round-trip fees, spread and cost of capital — never gross.
Worked test example: a memecoin at **21.90% gross** is blocked (500 bps basis,
150 bps spot spread) and ranks *below* a clean 13.14% gross trade netting 2.22%.
Six blocking risk flags. Delta-neutral ≠ risk-free, stated in the module docstring
itself (a test asserts it).

### Test evidence

| suite | checks | result |
|---|---|---|
| `omniseed_monster_regime` | 45 | 45/45 |
| `omniseed_monster_strategies` | 66 | 66/66 |
| `omniseed_monster_funding` | 47 | 47/47 |
| `omniseed_monster_features` | 51 | 51/51 (unchanged) |
| `omniseed_monster_news` | 51 | 51/51 (unchanged) |
| `omniseed_monster_sniper` | 45 | see below |

Every new module carries a **prefix-invariance** (no-look-ahead) proof, and the
regime suite is built on series whose ground truth is known by construction
(trend / OU / random walk) rather than on whatever the code happens to output.

### Constraints honoured

RWKV-7 0.1B edge brain untouched, well under 400 MB. The **2 % per-trade
ceiling** and the **3 % daily / 6 % weekly kill-switch** are unchanged and still
enforced in C++. The engine-facing CSV schema is still 5 columns, so the C++
side needed **no** change. No guaranteed-profit claim anywhere.




## 24. T7+ — SYSTEM-1 DECISION HEAD: "TWO HEADS, ONE BRAIN" (2026-09-26)

The architectural shift requested this pass: stop treating the core decision
layer as something that needs a Python wrapper. System-1 decision capability now
lives **inside the C++ runtime** — `omniseed.exe` reads a decision straight off
the backbone's hidden state without generating a single token. New doc:
`docs/JEV_INTEGRATION.md` (incl. the Jev/TypeSafe research and its limits).

### C++ — the head (`include/omniseed/decision_head.h`, `src/decision_head.cpp`, new)

- `DecisionHead` projects the post-`ln_out` residual `h[E]` onto a 7-action
  space with **one matvec** `[A,E] x [E]` + softmax + argmax. No token loop is
  reachable from `decide()`; it is allocation-free after the first call and
  never mutates the recurrent state.
- `DecisionResult { action_type, confidence_score, target_asset, invalidation,
  margin, routing, fast_path, matvecs, ms }` + `to_json()`. `invalidation` is
  the per-action *default* stop level: `0` means NOT SPECIFIED / unknown, which
  is deliberately distinct from "no stop" — the head cannot recover a price
  level from a hidden state, so the risk engine owns the real value.
- Fail-closed routing: `ABSTAIN` and `EXPLAIN` can never self-route, however
  confident. A near-tie fails the margin gate and escalates.
- Persistence (self-describing LE blob, validated on load) with `set_action()`
  as the offline-fitting hook. `trained()` is true only when **every** row is
  fitted, so a half-populated head cannot masquerade as a real one.

### C++ — the tap and the routing

- `RwkvModel::forward` gained an optional `Tensor* hidden_out = nullptr`
  (`src/core/rwkv.cpp`). It receives `xl3` — the same vector the text head
  projects to logits. Taken *before* the vocab projection, never fed back into
  the recurrence, and the tensor is reused across steps. `nullptr` is
  bit-identical to the previous behaviour.
- `agent_loop.cpp`: after the prompt prefill, `DecisionHead::decide(hidden)` runs
  before any text exists. `confidence >= threshold` → the action is executed
  immediately and System-2 is **never entered**; otherwise the turn falls
  through to the unchanged generation path.
- New `DecisionMode { Off, Hybrid, DecisionOnly }`. `Off` is the default in the
  CLI, the server, and `AgentLoop::Config`, so the feature costs nothing when
  unused and cannot silently change existing behaviour.
- Prefix-cache edge case handled explicitly: when the prompt is not re-fed there
  is no fresh hidden state, so the loop escalates rather than advancing the
  recurrent state purely to manufacture one.

### CLI + server

- `omniseed ask|chat --mode off|hybrid|decision-only --decision-threshold F
  --decision-head PATH`. An unknown `--mode` is refused (exit 2) rather than
  silently defaulting. `chat` prints `[decision] {json}` per turn and marks the
  turns System-1 answered alone.
- `omniseed_server --decision-mode|--decision-threshold|--decision-head`, with
  `OMNISEED_DECISION_MODE` / `OMNISEED_DECISION_HEAD` env fallbacks. `/health`
  reports the mode and the head's provenance; `/ask` adds `"decision"` and
  `"fast_path"` when the head was consulted.
- Drive-by fix: `/ask` reply strings are now JSON-escaped — a reply containing a
  quote previously produced a malformed response body.

### Verification

- `tests/test_decision_head.cpp` (new, ctest `omniseed_decision_head`) —
  **162 checks, 0 failures** (149 at first landing; the `invalidation` field
  added in `5257a8a` brought the count up), fully offline (no model, no GGUF,
  no network).
- Speed claim measured with the same clock and the same scalar fp32 inner loop
  on both sides, against the **most charitable possible** System-2 baseline (a
  bare output-head matvec per token, no attention/FFN/channel-mix — so the ratio
  is a *lower bound*): tiny E=64/V=256 → **3,419–5,040x**; mid E=256/V=8192 →
  **246,349–252,057x**. The mandated bar is 100x.
- Full local board `ctest -C Release`: **10/10 passed** at this commit (11/11
  once the T2 bridge suite below was registered).
- `omniseed_server` builds clean with `-DOMNISEED_BUILD_SERVER=ON`.
- No new Python dependency in the decision path; the <400 MB budget and the 2%
  risk-limit logic in C++ are untouched.

### Honest limits (carried in the doc, not buried)

- The shipped head is **untrained**: `init()` seeds a deterministic placeholder
  projection that is structurally valid and semantically meaningless.
  `trained()` / `provenance()` report it, and both the CLI and the server log a
  warning when the mode is on and the head is not fitted.
- Confidence is a **raw softmax, not calibrated**. Jev's RLCD calibration is not
  reproduced; 0.85 is a tuned operating point, not a probability of success.
- Fitting the projection offline is **not implemented** — `set_action()` /
  `save()` / `load()` are the interface for it. This is the next honest step,
  not a finished feature.

## 25. T7+ T2 — INTEGRATION: JSON CONTRACT, PROVENANCE, DASHBOARD, GATE (2026-09-27)

Integration and polish pass. No new strategy, no new research: everything here
wires up pieces that were already built and tested, and adds the tests that
prove the wiring holds.

### Naming reality vs the brief

The brief named `python/trading/paper_daemon.py` and `journal.csv`. Neither
exists and neither should: the paper daemon is **C++**
(`src/trading/paper_daemon.cpp`, driven by
`omniseed_agent2.exe trading-paper-session`), and the journal is
`state/paper_journal.csv`. Those paths were treated as naming intent. What was
actually missing was the **seam** between the C++ side and the Python
tooling — that is what this pass builds.

### `tools/decision_bridge.py` (new) — the JSON half of the contract

Parses the `DecisionResult` JSON that `agent_loop.cpp` emits on the fast path.
Strictly typed fields, and the fail-closed rules are the *only* thing it
exports as policy:

- `NEVER_TRADES = {ABSTAIN, EXPLAIN}` — these can never become an order,
  however confident.
- `OPENS_RISK = {BUY, SELL, HEDGE}`.
- `screen()` raises `ValueError` if asked for a threshold **below** 0.85. The
  mandate bar may be raised, never lowered.
- A watched symbol with **no** decision is refused, not ignored: *silence is
  not consent*.
- Malformed input never raises out of `screen()` — it is refused.

CLI: `--json` / `--jsonl` / `--threshold` / `--allow-unknown-stop` /
`--self-test`; exit 0 = tradeable, 1 = refused, 2 = malformed.

### Entry provenance through a frozen journal schema

The journal is a frozen 10-field CSV (`ts,kind,ticker,qty,price,pnl,equity,
cash,exposure,reason`) and `Position` carries no stop or confidence field. So
provenance rides in the free-text `reason` column:

```
entry conf=<0..1> stop=<price> stop_pct=<frac>
```

- Written by `entry_reason()` in `paper_daemon.cpp` on both entry paths.
  Spaces only, so `csv_safe()` (which rewrites `,`/`\n`/`\r`) cannot mangle it.
- `stop` is derived from the **same `RiskLimits` the exit check uses**, so the
  dashboard can never display a stop the engine would not honour.
- `paper_report.py` recovers it in `replay_positions()`; the stop **level** is
  recomputed from the blended average to match `RiskManager::check_exit`, and
  a legacy row with no provenance reports `None` — never `0`.

### Dashboard + server

- `tools/paper_dashboard.py` — one self-contained HTML document (no `<link>`,
  no `<img>`, no CDN). Panels: live equity curve (inline SVG), open positions
  with **entry / stop / distance-to-stop / confidence meter / last / unrealized**,
  daily P&L KPIs, regime status, feed health, and a risk panel that quotes the
  engine's own limits.
- `tools/serve_dashboard.py` (new) — stdlib `ThreadingHTTPServer`; routes `/`,
  `/api/state`, `/healthz`, `/journal.csv`, 404 otherwise; `Cache-Control:
  no-store`. An inlined poller re-draws the KPIs, the curve, the positions and
  the regime table in place. Opened from `file://` it degrades to a static
  snapshot and says so instead of failing silently.
- `tools/paper_loop.py` — `DecisionGate` (`off` | `file` | `agent`) and the
  regime reading are wired into the heartbeat. The gate can only **append to
  the engine's `--skip` list**; it cannot widen risk. Every unusable path fails
  closed (missing log, sub-mandate threshold, unknown source, agent failure).

### Why the new test is NOT behind the `.venv` gate

`tests/test_decision_bridge.py` is stdlib-only, so it is registered with
`find_package(Python3)` *outside* the `.venv` guard that the LoRA suites use.
It has to run in CI (where no `.venv` exists) because
`test_engine_limits_match_cpp()` re-reads `include/omniseed/trading/*.h` and
fails if the Python copy of the 2% / 3% / 8% limits ever drifts from C++.

### Verification

- `tests/test_decision_bridge.py` (new, ctest `omniseed_decision_bridge`) —
  **141 checks, 0 failures** when a market CSV is reachable; **133** when it is
  not (section 8 skips honestly and prints why).
- Section 8 is the cross-language contract: it runs the **real daemon** and
  parses what it actually wrote, because the fixture-based tests would keep
  passing if `entry_reason()`'s format silently changed.
- End-to-end on real data (`models/market/AAPL_1d.csv`, 1254 bars): the daemon
  produced **43 entries / 42 exits**, every entry reason parsed, every
  `stop == entry x (1 - stop_pct)` to price precision, and the report layer
  recovered `AAPL qty=64 entry=323.13 stop=297.28 conf=0.500` — i.e.
  `323.13 x 0.92 = 297.28`.
- Live server smoke on port 8791: `/` 200, `/api/state` 200, `/healthz` 200,
  `/nope` 404.
- Full local board `ctest -C Release`: **11/11 passed**.

### Honest limits

- The journal's `conf=` is the **rule-based strategy's `Signal::strength`**, not
  the System-1 head's softmax. The head is still untrained (§24), so routing it
  into the paper loop would put an uncalibrated number in front of real risk
  logic. The dashboard labels it "confidence" in the strategy sense.
- The gate is **risk-removing only**. The 2% per-trade budget and the 3% daily
  kill-switch remain C++-enforced and untouched; Python mirrors them for
  display and is contract-tested against drift.
- Legacy journal rows have no provenance; their confidence renders as `—`.
  Back-filling would mean inventing numbers, so it is not done.
- The head's `invalidation` is a default the risk engine is expected to
  override; nothing in the paper path consumes it yet.

## 26. ⭐ TRADING SYSTEM v1.0 COMPLETE (2026-09-27)

Everything that was researched, prototyped and tested separately is now
**assembled, merged and runnable from `main`**. No new strategy was added in
this pass — the brief was integration and polish, and that is all this is.

### What v1.0 contains

| Layer | Artifact | State |
|---|---|---|
| Market data | `tools/fetch_market_data.py`, provenance CSV | ✅ |
| Perception + risk gates | `src/trading/*`, M1/M4 | ✅ |
| Walk-forward edge lab + risk engine + daily kill-switch | T7.1 | ✅ |
| Paper daemon (multi-asset, journal resume) | `src/trading/paper_daemon.cpp` | ✅ |
| 24/7 paper loop + report + dashboard | `tools/paper_loop.py`, `paper_report.py`, `paper_dashboard.py` | ✅ |
| **Monster M7** — sniper entry, news hunter, cross-asset filter, sizing | `tools/monster/*`, PR #2 | ✅ merged |
| **Monster M8** — regime engine, strategy ensemble, funding carry | `tools/monster/*`, PR #3 | ✅ merged |
| **System-1 decision head** — "two heads, one brain" | `src/decision_head.cpp` | ✅ merged |
| **T2 integration** — JSON contract, provenance, dashboard, gate | `tools/decision_bridge.py`, `serve_dashboard.py` | ✅ merged |

### How to run it — exact commands

Run everything from the repo root. On Windows the venv interpreter is
`.venv\Scripts\python.exe` (POSIX: `.venv/bin/python`).

**1. Get some market data** (once; the loop also refreshes it itself)

```
.venv\Scripts\python.exe tools\fetch_market_data.py --ticker AAPL --timeframe 1d
```

**2. Start the paper daemon — the 24/7 loop**

```
.venv\Scripts\python.exe tools\paper_loop.py --watch equity:AAPL --interval 300
```

- Defaults: `--watch equity:AAPL,crypto:BTCUSDT,fx:EURUSD`,
  `--journal state/paper_journal.csv`, `--csv-dir models/market/paper`.
- One cycle and exit instead of looping forever: add `--once`.
- Dry run (poll + refresh CSVs, never invoke the engine): `--dry-run`.
- Optional System-1 gate (can only *remove* risk): `--decision-source file
  --decision-jsonl <path>`.
- The engine itself is `build\bin\omniseed_agent2.exe
  trading-paper-session --streams "SYM,asset,path.csv" ...` if you want to run
  a single session without the loop.

**3. Open the dashboard**

Live (recommended — auto-refreshes every 15 s):

```
.venv\Scripts\python.exe tools\serve_dashboard.py --port 8787 --open
```

…then browse **http://127.0.0.1:8787**. The server is stdlib-only and
loopback-bound; routes are `/`, `/api/state`, `/healthz`, `/journal.csv`.

Static snapshot instead (single self-contained file, no server):

```
.venv\Scripts\python.exe tools\paper_dashboard.py --journal state\paper_journal.csv --out dashboard.html
```

**4. Optional — System-1 fast path in the agent**

```
build\bin\omniseed.exe ask --mode hybrid --decision-threshold 0.85 --decision-head <blob> "what should I do with AAPL"
```

`--mode` is `off` (default) | `hybrid` | `decision-only`. `off` allocates no
head and is bit-identical to previous behaviour.

**5. Rebuild + verify**

```
cmake --build build --config Release --parallel
ctest --test-dir build -C Release
```

### Merge record (this pass)

Three branches, all rooted at the same base (`fd7f999`), merged into `main`:

- **PR #2** `feature/monster-trading-model` (`e848c6b`) — subsumed by PR #3.
- **PR #3** `feature/monster-m8-regime` (`0d0ebbc`) — merged; it already
  contained `e848c6b`, so M7 arrived with it.
- **Integration** `workbuddy/main-ddf5e156` (`76b17a8`) — the System-1 head
  plus the T2 wiring.

One conflict: `tools/paper_loop.py` (both sides extended the same ctor, the
same cycle body, the same argparse block and the same `PaperLoop` call).
Resolved as a **superset union** — verified by diffing the result against both
parents and confirming every removal was only a signature the other side had
extended.

### Verification (post-merge, on the merged tree)

- `ctest -C Release`: **11/11 passed** (22.9 s).
- The 11 stdlib-only Python suites (venv-gated in CMake, run explicitly here):
  **661 checks, 0 failures** — including all six Monster suites and the T2
  bridge suite.
- `tests/test_decision_bridge.py` with a real feed: **146 checks, 0 failures**,
  including the live daemon→reader contract.
- Release `build/bin/omniseed.exe` builds clean: **0.44 MB** (budget 400 MB);
  all 13 binaries total 2.7 MB.
- `--mode bogus` is refused with a clear message rather than silently
  defaulting.
- Fixed while verifying: `tests/test_paper_reports.py` asserted "no `<script>`
  tag at all", which the T2 mandate's embedded-JS dashboard necessarily
  violates. Corrected to assert the real invariant — no *external* script
  source — while the separate `no remote refs` check still pins
  self-containment. This suite had never actually run (it is venv-gated and
  no `.venv` existed), which is exactly why the assertion had drifted.

### Known gap, stated plainly

The stdlib-only Python suites sit behind CMake's `.venv` guard, and CI creates
no `.venv` — so **they do not run in CI**. `test_decision_bridge` was
deliberately registered *outside* that guard because its
`test_engine_limits_match_cpp` check guards the 2%/3%/8% limits against drift
and is worthless if it never executes. The other Python suites still only run
on a developer box with a `.venv`; moving them out of the guard is a
recommended follow-up, not something this pass changed.

### Hard limits — unchanged and still enforced

- **2% per-trade risk budget** and **3% daily / 6% weekly kill-switch** remain
  C++-enforced in `include/omniseed/trading/*.h`. Python only mirrors them for
  display, and a contract test fails if that mirror drifts.
- The System-1 decision gate is **risk-removing only**: it can append to the
  engine's `--skip` list and nothing else. Every unusable path fails closed.
- Edge binary **0.44 MB**, far under the 400 MB budget.
- No claim of "no loss" is made anywhere; losses are minimised by discipline,
  never eliminated.


---

## 27. TRUE INTEGRATION — THE TRADING INTELLIGENCE MOVES INTO THE RUNTIME (2026-09-27)

### The problem this section answers

The Monster trading intelligence lived in Python (`tools/monster/*.py`) and the
C++ runtime read its conclusions back as a CSV of feature rows
(`ts,confidence,veto,regime,detail`). That is a **seam, not an architecture**:
the model's own runtime could not answer "what regime is this, and what do my
strategies think?" without a separate process running first. `grep` for
`popen`/`CreateProcess`/`subprocess` across `src/` returns only two
`std::system("mkdir ...")` calls — so the coupling was a *data contract*, not a
runtime dependency, but the model was still structurally blind.

This section records moving that intelligence into the runtime, where the model
can answer those questions itself.

### What was audited (Phase 1)

Every Python module in `tools/`. There is no `python/` directory. The split:

| module | lines | what it is | verdict |
|---|---|---|---|
| `monster/regime.py` | 771 | multi-axis regime detection | **MOVED** → `src/trading/regime_engine.cpp` |
| `monster/strategies.py` | 470 | momentum / mean-reversion / breakout / OFI + channel + stat primitives | **MOVED** → `src/trading/strategy_zoo.cpp` |
| `monster/features.py` | 332 | causal indicator primitives (sma/ema/rsi/macd/bb/atr/vwap/zscore/swings/fib) | **MOVED** → `strategy_zoo.cpp` (`namespace features`) |
| `monster/sniper_engine.py` | 374 | weighted 4-layer confluence score `S` + the veto ladder | **MOVED** → `src/trading/sniper.cpp` |
| `monster/router.py` | 184 | regime-adaptive blend → conviction / agreement + the 4-condition veto | **MOVED** → `src/trading/router.cpp` |
| `monster/correlation_matrix.py` | 164 | cross-asset lead-lag | Python (needs the peer feed); consumed as `EvalContext` |
| `monster/state_vector.py` | 115 | assembles the distilled vector for the model | Python (Phase 3 will fold this into the unified output) |
| `monster/sizing.py` | 88 | adaptive sizing | already C++ (`RiskManager::size_by_risk`) — Python was the mirror |
| `monster/funding.py` | 300 | funding-rate carry scanner | **stays Python** — needs a network feed |
| `monster/news_hunter.py` | 200 | news → `event_driven` flag | **stays Python** — network + LLM |
| `paper_dashboard.py` | 493 | HTML rendering | **stays Python** — presentation, no logic |
| `serve_dashboard.py` | 193 | HTTP server | **stays Python** — an HTTP server is not model inference |
| `paper_report.py` / `nightly_report.py` | 481 | nightly report | stays Python — reporting |
| `market_feeds.py` / `fetch_*.py` | ~1300 | network data acquisition | stays Python — I/O |
| `paper_loop.py` | 673 | the old polling loop | superseded by `src/trading/paper_daemon.cpp` |
| `decision_bridge.py` | 398 | the Python shim that called the model | superseded by the C++ decision head + unified output |
| `qat_*` / `lora_*` / `convert_*` / `modal_*` / `dbg_*` | ~3000 | offline training/quantisation tooling | stays Python — never on the runtime path |

**Rule applied:** anything that *decides* moves to C++; anything that
*transports, presents, or trains* stays Python. A Python module that only reads
a file and prints HTML has no business being a decision path, and a C++ engine
has no business opening a socket to a market-data vendor.

### Two mandate premises that were already true

Stated plainly because "fixing" them would have been pointless, risky churn:

- **`DecisionHead` is already a C++ class** (`include/omniseed/decision_head.h`,
  commit `7179ff8`) — not a Python script.
- **The 2% per-trade / 3% daily limits are already C++-enforced** in
  `include/omniseed/trading/*.h`. Python only mirrors them for display, and
  `tests/test_decision_bridge.py::test_engine_limits_match_cpp` fails if the
  mirror drifts.

### The four named bugs were already fixed in the Python

Each was verified by reading the source, not assumed. Phase 5 therefore became
"port the *fixed* behaviour and lock it with a regression test", which is what
was done — the four regressions are now pinned C++-side:

| named bug | where it was already fixed | C++ regression test |
|---|---|---|
| the `0.8125 < 0.85` gate | `regime.py` — smoothed vol percentile + absolute ATR stress, so entry bars stop being labelled `high_vol` | `test_strategy_zoo.cpp` D2 (ceiling table) |
| Hurst on price levels | `regime.py:547` already feeds `hurst_rs` the **returns** | `test_regime_engine.cpp` B1 |
| ER horizon mismatch | `regime.py:538` already **sweeps** `er_n = (10,20,40)` and averages the normalised values | `test_regime_engine.cpp` B2 |
| ensemble veto on a single signal | `router.py:155` already requires `weight >= veto_min_weight (0.50)` | `test_strategy_zoo.cpp` C (**B4**) |

### The new files

| file | what |
|---|---|
| `include/omniseed/trading/regime_engine.h` / `src/trading/regime_engine.cpp` | the multi-axis causal regime engine + `RegimeTracker` hysteresis |
| `include/omniseed/trading/strategy_zoo.h` / `src/trading/strategy_zoo.cpp` | `features::` primitives, Donchian/Keltner/OFI, the four strategies, vol-target |
| `include/omniseed/trading/router.h` / `src/trading/router.cpp` | continuous regime weighting → conviction/agreement, the weight-gated veto |
| `include/omniseed/trading/sniper.h` / `src/trading/sniper.cpp` | the weighted 4-layer score `S` + the veto ladder |
| `tests/regime_dump.cpp`, `tests/strategy_dump.cpp` | JSON dumpers, so the parity tests can diff C++ against the oracles |
| `tests/test_regime_engine.cpp`, `tests/test_strategy_zoo.cpp` | the C++ suites |
| `tests/test_regime_parity.py`, `tests/test_strategy_parity.py` | the parity gates |

### Why a parity gate, and not a blind hand-port

A ~1,400-line numeric port **will** drift from its original. A silently
diverging second implementation is strictly worse than not porting at all,
because the divergence is invisible and the C++ is the one making decisions. So
the Python modules stay in the tree as the **oracle**, and
`test_regime_parity.py` / `test_strategy_parity.py` diff the two field by field.
While those are green the two are interchangeable, which is what makes it safe
to retire the Python from the runtime path *without touching it*.

They compare every signal's `direction` / `confidence` / `reason`, every router
field (`conviction`, `agreement`, `weight`, `w_trend`, `active`, `size_factor`,
`veto_long`), and every sniper layer (`score`, `micro`, `tech`, `regime_score`,
`cross`, `votes`, `regime`, `veto`, `veto_reason`, `factors`, `trend_score`,
`half_life`, `conviction`, `agreement`, `propose`) — on synthetic trend / chop /
squeeze / vol series, with the regime tracker both on and off, in all three
sniper modes (advanced / legacy / ensemble), and on **1,255 real AAPL daily
bars**. Tolerance is `1e-9` relative; both-NaN counts as equal, because "no
reading" on both sides is agreement, not a mismatch.

Both parity suites are registered **outside** CMake's `.venv` guard, on purpose.
The stdlib-only Python suites sit behind an `if(EXISTS "${OMNISEED_VENV_PY}")`
check, CI creates no `.venv`, and this worktree has none either — so a
venv-gated suite silently never runs. That is exactly how
`tests/test_paper_reports.py` drifted into asserting something a later mandate
made impossible. A parity gate that never executes is not a gate.

### Verification (this pass)

- `ctest --test-dir build -C Release`: **15/15 passed** (287 s). Board grew from
  11 to 15 with `omniseed_regime_engine`, `omniseed_regime_parity`,
  `omniseed_strategy_zoo`, `omniseed_strategy_parity`.
- `omniseed_strategy_zoo.exe`: **291 checks, 0 failed** — feature primitives on
  hand-checkable inputs, every strategy branch (including `fade`, which needs a
  hand-built regime because a trending series is correctly never tradeable),
  the B4 lone-signal regression, the weight floor at exactly 0.50, veto
  precedence, the 0.85 reachability table, "the ensemble blocks only, never
  promotes" (score bit-identical with the ensemble on and off), and prefix
  invariance across 259 prefixes.
- `omniseed_regime_engine.exe`: **157 checks, 0 failed**.
- `test_strategy_parity.py`: **38 passed, 0 failed** (including the real feed).
- `test_regime_parity.py`: **18 passed, 0 failed** (including the real feed).
- Release build clean under `/W4`; no new warnings.

### ⚠️ Finding: the sniper proposes nothing on real data

Measured, not inferred. Over the 1,255 real AAPL daily bars in
`models/market/AAPL_1d.csv`:

```
max S = 0.8317   (the 0.85 bar is never reached)
proposals = 0
technical-vote histogram:  0 votes -> 266,  1 -> 709,  2 -> 266,  3 -> 14   (never 4+)
veto reasons: insufficient-confluence 1130, high-vol-needs-mean-reversion 103,
              none 13, counter-regime 9
```

The arithmetic is not the problem — `docs/MONSTER_DESIGN.md`'s reachability
table is correct and the `high_vol` case (ceiling 0.8125) really was fixed:

```
with M = T = C = 1.0:   trend_up 1.0000 | range 0.8750 | high_vol 0.8125 | trend_down 0.7500
```

The problem is that those ceilings assume `cross = 1.0`. With **no cross-asset
context** — the default, since `EvalContext` is optional — `cross` is 0.5, and
then even a perfect bar is capped at:

```
trend_up: 0.25 + 0.35*0.90 + 0.25*1.00 + 0.15*0.5 = 0.8900   reachable
range:    0.25 + 0.35*0.90 + 0.25*0.50 + 0.15*0.5 = 0.7650   NOT reachable
```

(`tech = 0.90` is the 3-vote rung, and 3 is both `min_factors` and the observed
maximum on this feed.) Since `range` is 81% of the bars, the gate is closed for
most of the tape, and the 0.8317 observed maximum says the favourable
combination never actually coincided even on the `trend_up` bars.

This is **the same bug class the mandate named** — a gate that cannot open — and
it is **pinned, not fixed**: `test_strategy_zoo.cpp` D2 asserts both the
documented ceiling table *and* the no-context numbers above, so the situation
cannot drift unnoticed in either direction. Changing it means changing what the
system trades (lower `min_confidence`, reweight the layers, or supply real
cross-asset context), which is the owner's decision, not a silent one.

### Other findings, reported not silently changed

- **`find_swings`'s `kind` parameter is inert in the oracle.** It always returns
  strict local *minima*, so `find_swings(highs, ..., "high")` returns minima of
  the high series rather than swing highs, and the Fibonacci-support factor is
  built from the wrong object. The C++ reproduces the oracle exactly (so parity
  holds) and additionally exposes `find_local_maxima` for the correct one. Which
  the sniper uses is a behaviour change and is left to the owner.
- **`bollinger` does not NaN-check its window, while `rolling_std` does.** The
  inconsistency is the oracle's and is reproduced, because "cleaning it up"
  would change which bars produce a number.
- **The oracle's `last_swing_pair` is O(lows x highs) per bar**, which makes a
  full-feed Python scan quadratic-to-cubic. The parity test therefore compares
  the sniper on 250-bar prefixes and says so in a comment, rather than silently
  timing out or quietly skipping.

### What is still Python, and why

`paper_dashboard.py`, `serve_dashboard.py` (an HTTP server is not model
inference); `paper_report.py`, `nightly_report.py` (reporting);
`market_feeds.py`, `fetch_*.py` (network I/O); `funding.py`, `news_hunter.py`
(external feeds); `correlation_matrix.py`, `state_vector.py` (Phase 3 folds
these into the unified single-pass output); the `qat_*` / `lora_*` / `convert_*`
/ `modal_*` / `dbg_*` offline tooling (never on the runtime path).

`paper_loop.py` and `decision_bridge.py` are now **superseded** — the C++ paper
daemon and the C++ decision head do their jobs — but they are left in the tree
as reference until the unified output path (Phase 3) is verified end to end.

---

## 28. UNIFIED INTELLIGENCE — ONE BACKBONE, MANY HEADS, A SMART ROUTER (2026-09-27)

### 28.1 The premise check, first

Two premises in the mandate were **already true**, and saying so is more useful
than "fixing" working code:

- **`DecisionHead` is already a C++ class** (`include/omniseed/decision_head.h`,
  commit `7179ff8`). It was not a Python module.
- **The 2% / 3% risk limits are already C++-enforced.** Python mirrors them for
  display only.

The audit table below therefore records what actually needed to move.

### 28.2 PHASE 1 AUDIT

| Module | Current Location | Can Move to C++? | Must Stay Python? | Priority |
|---|---|---|---|---|
| Decision head (trading) | `src/decision_head.cpp` | **already C++** | no | — |
| Router (which heads run) | — (new) | **YES** → `src/router.cpp` | no | P0 |
| Classification head | — (new) | **YES** → `src/classification_head.cpp` | no | P0 |
| Scoring head | — (new) | **YES** → `src/scoring_head.cpp` | no | P0 |
| Domain decision head | — (new) | **YES** → `src/domain_decision.cpp` | no | P0 |
| Unified output formatter | — (new) | **YES** → `src/unified_output.cpp` | no | P0 |
| Intent / language / sentiment | — (new) | **YES**, lexical → `src/language/*.cpp` | no | P0 |
| Regime engine | `tools/monster/regime.py` | **DONE §27** → `src/trading/regime_engine.cpp` | oracle only | — |
| Strategy zoo | `tools/monster/strategies.py` | **DONE §27** → `src/trading/strategy_zoo.cpp` | oracle only | — |
| Ensemble router | `tools/monster/router.py` | **DONE §27** → `src/trading/router.cpp` | oracle only | — |
| Sniper | `tools/monster/sniper_engine.py` | **DONE §27** → `src/trading/sniper.cpp` | oracle only | — |
| Risk limits 2% / 3% | `include/omniseed/trading/risk_engine.h` | **already C++** | display mirror only | — |
| Vision heads | — | label space registered; needs a feature vector | — | P2 |
| Audio heads | — | label space registered; needs a feature vector | — | P2 |
| Batch processing | — | not started | — | P3 |
| Streaming decisions | — | not started | — | P3 |
| Feedback hooks | — | not started (fitting hooks exist) | — | P3 |
| `paper_dashboard.py`, `serve_dashboard.py` | `tools/` | no — I/O + HTTP | **YES** | — |
| `market_feeds.py`, `fetch_*.py` | `tools/` | no — network I/O | **YES** | — |
| `paper_report.py`, `nightly_report.py` | `tools/` | no — reporting | **YES** | — |
| `qat_*`, `lora_*`, `convert_*`, `modal_*` | `tools/` | no — offline tooling | **YES** | — |

### 28.3 What was built

Nine new translation units and eight new headers, all in `omniseed_core`:

| File | What it is |
|---|---|
| `include/omniseed/heads.h` + `src/heads.cpp` | the shared vocabulary: `Domain`, `LabelProb`, `ClassificationResult`, `ScoreResult`, and the JSON/seed helpers |
| `include/omniseed/router.h` + `src/router.cpp` | `HeadRouter`, `ActivationPlan`, `HeadKind`, `RouterMode`, `refine()` |
| `include/omniseed/classification_head.h` + `.cpp` | one `[total_labels, E]` projection, 13 named label sets |
| `include/omniseed/scoring_head.h` + `.cpp` | three INDEPENDENT sigmoids |
| `include/omniseed/domain_decision.h` + `.cpp` | the decision head generalised over 5 domains + the `DecisionResult` adapter |
| `include/omniseed/unified_output.h` + `.cpp` | `UnifiedOutput::normalise()/to_json()` and `UnifiedPipeline` |
| `include/omniseed/language/language_heads.h` + `src/language/intent.cpp` + `src/language/sentiment.cpp` | the language domain, deterministic and offline |

Plus four test suites, the `--mode text-only` CLI alias, and four new docs:
`docs/UNIFIED_ARCHITECTURE.md`, `docs/JEV_FEATURES.md`, `docs/DOMAINS.md`,
`RUNBOOK.md`.

### 28.4 The architecture, in one paragraph

`input -> RWKV-7 forward pass -> h[E] -> {router, decision, token, classify, score}`.
Every head is a projection `[K, E] x [E] -> [K]` plus a squashing function, so the
marginal cost of a head is `K * E` multiply-accumulates, not a second forward
pass. The **router** decides which heads run and in what mode; the **decision head
runs first**; if it is confident (`>= 0.85`) and the task is simple the pipeline
returns the decision and **never calls the text producer**; otherwise the decision
is handed to the text producer as **prefix context**, so the text explains the
decision that was actually made. `UnifiedOutput` emits ONE JSON document in which
a field is present **iff its head ran**.

### 28.5 ⚠️ TWO REAL BUGS, found by the new tests

Both were introduced during this work and both were caught by the tests written
for the same work — which is the entire argument for writing them.

**Bug A — the tokenizer split every word into its letters.**
`src/language/intent.cpp` had a single `flush()` that pushed *and cleared* both
the ASCII word buffer and the Bengali run. The ASCII branch called it *before*
appending each character, so `"Hello"` tokenised to `["h","e","l","l","o"]`. Every
English intent and sentiment test failed at once; the Bengali tests passed, which
is exactly the signature you would expect. Fixed by splitting the operation into
`close_word()` and `close_beng()`, which is also the correct semantics: starting
an ASCII word must end a Bengali run **without** destroying the word being built.

**Bug B — the seed was not injective, so two different seeds produced identical
weights.** `HeadRouter::seed_weights` and `ScoringHead::seed_weights` used
`seed | 0x9E3779B97F4A7C15ull`. Because the constant already has bits `0x4F10`
set, `20240 | C == 20241 | C` — so bumping the seed by one silently produced a
**byte-identical head**. Fixed to the multiply-add form
(`seed * C + D`, a bijection mod 2^64 since the multiplier is odd), which is what
`DecisionHead::seed_weights` had been doing correctly all along.
`tests/test_router.cpp` B12 now pins it.

`ClassificationHead` and `DomainDecisionHead` were checked for the same pattern
and are correct — they key each row on FNV-1a of `(seed, name)` using **XOR**,
which is injective in the seed's low bits.

### 28.6 Measurements (Release, E = 768, 2000–4000 iterations)

| Head | Cost | Budget | Margin |
|---|---|---|---|
| `decide(general)` 4 actions | **0.498 us** | 1 ms | ~2000x |
| `decide(trading)` 7 actions | 0.756 us | 1 ms | ~1300x |
| `decide(vision)` / `decide(audio)` | 0.940 / 0.941 us | 1 ms | ~1000x |
| `decide(language)` 6 actions | **1.150 us** (worst) | 1 ms | ~870x |
| `classify(language.intent)` 7 labels | 2.139 us | 1 ms | ~470x |
| `score()` 3 rows | 0.505 us | 1 ms | ~2000x |
| `HeadRouter::plan()` 9 matvecs | **15.307 us** | 100 us | ~6.5x |

On the mandate's **897 ns** figure: the 4-action domains **beat** it (general is
498 ns); the 6-action language domain is ~1.3x over. The cost scales with the
action count, so it is a per-action-space number, not a constant. The test asserts
the 1 ms budget and prints every row.

The router is the interesting one: 15.3 us for 6,912 MACs is **not**
arithmetic-bound. The returned `ActivationPlan` owns a `heads` vector and a
`reason` string, and those two allocations cost more than the multiply-accumulates.
Reserving both took it from 19.4 us to 15.3 us. `include/omniseed/router.h` now
says this instead of the "a few microseconds" estimate it previously carried,
which was wrong.

### 28.7 Verification

- **`ctest`: 19/19 green** (was 15/15).
- **+1,710 new checks**: `omniseed_heads` 481, `omniseed_router` 348,
  `omniseed_unified_output` 267, `omniseed_language_heads` 614.
- **No regression**: all 15 pre-existing tests still pass, including the two
  Python-vs-C++ parity gates over 1,255 real AAPL bars.
- Zero warnings under `/W4 /permissive- /Zc:__cplusplus /utf-8`.
- Binaries: `omniseed.exe` 457 KB, `omniseed_agent2.exe` 327 KB. The 0.1B ternary
  GGUF is 192–280 MB, inside the <400 MB edge rule.

The invariants asserted rather than assumed:

- the fast path **never calls** the text producer on `decision_only` — asserted by
  **invocation count** (0), because a timing number cannot tell "skipped" from
  "fast" (`test_unified_output.cpp` C1);
- `refine()` **never adds a head** — checked over 5 input plans x 8 confidences
  (`test_router.cpp` C7);
- a field is dropped and reported when the plan does not name its head
  (`test_unified_output.cpp` A2/A3);
- **no domain can self-route** with the threshold above the maximum attainable
  confidence (`test_heads.cpp` D9), and none with a margin floor of 1.0 (D11);
- priority > 0.99 **and** urgency < 0.01 from the same hidden state — the test a
  softmax could not pass (`test_heads.cpp` C4);
- `"not bad, it's good"` is **positive** (`test_language_heads.cpp` D3);
- negators appear in **neither** lexicon, asserted structurally (D7);
- `"no problem"` is positive and `"no good"` is negative (D4).

### 28.8 The honest gaps

1. **Nothing is trained.** Every projection is a deterministic placeholder.
   `trained()` is false and `provenance()` says `NOT FULLY FITTED (placeholder)`.
   A seeded head emits well-formed but **meaningless** values.
2. **Calibrated confidence is NOT implemented.** Temperature scaling needs
   labelled data, a fitted projection, and a held-out split; none exists. Shipping
   the scalar without the data would produce a calibrated-*looking* number with no
   justification. The language heads' `confidence` is documented as a **heuristic
   strength**, explicitly not a probability. See `docs/JEV_FEATURES.md` §2.
3. **Vision and audio have label spaces and no logic.** Registered, so a fitted
   projection would light them up — but nothing produces a vision or audio feature
   vector into `h[E]`. Registering a label space is not the same as having the
   capability.
4. **Batch, streaming decisions, and feedback hooks are NOT started.**
   Hierarchical routing is one level; ensemble voting exists only inside trading
   (`src/trading/router.cpp`, with the 0.50 veto weight floor).
5. **The CLI does not use the router on its default path.** `--mode` still drives
   the existing `DecisionMode` path, which has a live consumer contract
   (`tools/decision_bridge.py` reads its JSON). Rewiring it is a behaviour change
   that needs its own parity gate, not a quiet refactor. `--mode text-only` is
   accepted as an alias for `off`.
6. **The HTTP server does not expose `mode`, and the dashboard does not read the
   unified document.** Both are I/O-layer work.
7. **Romanised Bengali reads as English** — detection is by script, and
   `"ami bhalo achi"` is pure ASCII. Pinned as a known limitation in
   `test_language_heads.cpp` B4 so a future fix is deliberate.
8. **Bengali `না` is both the negator and the sentence-final question particle.**
   A lexical scorer cannot separate them without syntax.

### 28.9 Next steps

1. **Fit one head and calibrate it.** That is the single highest-value next step:
   it turns "well-formed but meaningless" into a real capability, and it is the
   prerequisite for every confidence claim in this tree. The fitting hooks and the
   `fitted_rows()` accounting are already in place.
2. **Feed the router into the CLI's `--mode` path** behind a parity gate against
   the existing `DecisionMode` behaviour, so the unified path is not a second
   untested implementation.
3. **Expose `mode` on the HTTP server** and have the dashboard read the unified
   document.
4. **Produce vision/audio feature vectors into `h[E]`** to light up the label
   spaces that are already registered.
5. **Fix the sniper's 0.85-gate ceiling** (see §27): on 1,255 real AAPL bars the
   sniper proposes nothing, because with no cross-asset context `cross = 0.5` caps
   a perfect `range` bar at 0.7650. Pinned in `test_strategy_zoo.cpp` D2, not
   silently changed — fixing it changes what the system trades.

### 28.10 Exact commands

```bash
# build (strip the proxy vars or MSBuild aborts)
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    cmake --build build --config Release --parallel

# the full board
ctest --test-dir build -C Release --output-on-failure

# the head stack alone, with latency printed
./build/bin/omniseed_heads.exe
./build/bin/omniseed_router.exe
./build/bin/omniseed_unified_output.exe
./build/bin/omniseed_language_heads.exe

# CLI modes
./build/bin/omniseed.exe chat --model models/rwkv7-0.1B-ternary.gguf --mode hybrid
./build/bin/omniseed.exe chat --mode text-only
```

---

## 29. WORKFLOW FIX — ONE LOCATION, AND A VERIFIED ANSWER ON PUSHING (2026-09-27)

**Reported problem:** work was landing in a hidden session worktree, forcing a
manual fetch/merge, and files the brief named were absent from
`C:\Users\sakim\OneDrive\Desktop\omniseed`.

**Root cause.** The repo has two worktrees sharing one `.git`, but they sat on
*different branches*:

| checkout | branch | HEAD |
|---|---|---|
| `Desktop\omniseed` | `feature/monster-m8-regime` | `96b2d1a` |
| session worktree | `main` | `3c61f54` |

`96b2d1a` is a **strict ancestor** of `main`, so nothing was lost — the desktop
folder was simply parked **40 files / +13,500 lines** behind. **10 of the 15
files the brief listed were absent:** `router.{h,cpp}`,
`trading/regime_engine.cpp`, `trading/sniper.cpp`, `language/{intent,sentiment}.cpp`,
`docs/UNIFIED_ARCHITECTURE.md`, `docs/JEV_FEATURES.md`, `docs/DOMAINS.md`,
`RUNBOOK.md`.

**Fix.** Git forbids the same branch in two worktrees, so the session worktree was
detached (`git checkout --detach`) to free `main`, then the desktop folder was
moved onto it (`git checkout main`) — a pure fast-forward. `.venv/` and
`models/market/*.csv` are gitignored and survived untouched; the untracked
`dashboard.html` was not clobbered. **All 15 files verified present afterwards.**
The desktop folder now tracks `origin/main`, so future work happens there.

`dashboard.html` was then committed (`3e21d3a`, *"chore: sync unified architecture
files to main"*) — it is referenced by this file, `docs/TRADING_GUIDE.md`,
`docs/TRADING_LAB.md`, `tools/paper_dashboard.py` and `tools/serve_dashboard.py`,
yet had never been committed.

### ⚠️ The push is blocked, and the reason is now precisely known

| channel | result |
|---|---|
| `git push`, credential helper disabled | `remote: No anonymous write access.` — the request **reaches** GitHub and is refused for lack of auth |
| `git push` / `git credential fill`, GCM enabled | **hangs** on an interactive flow; GCM 2.9.0 stores nothing |
| GitHub connector `push_files` | `403 .../git/trees` |
| GitHub connector `create_or_update_file` | `403 .../contents/...` |

Both connector errors read `Resource not accessible by integration` — the GitHub
App installation is **read-only** (no `contents: write`). **`api.github.com` is
reachable from this machine (HTTP 200); the network is not the problem, the
credential is.** No Windows Credential Manager entry, no `~/.git-credentials`,
no `GH_TOKEN`/`GITHUB_TOKEN`, no SSH key.

**To publish, run one command** (GCM will prompt once):

```bash
cd "C:\Users\sakim\OneDrive\Desktop\omniseed" && git push origin main
```

Or re-authorise the GitHub connector with write access and the push can be done
from here.

### Verified board in the desktop folder (33 tests, vs 19 in the session worktree)

`.venv` exists here, so the `.venv`-gated Python suites actually execute — **the
full board is 33 tests, not 19.** Verified green this pass:

- **C++ (14/14)** — `platform`, `qat_ternary`, `lora`, `trading`, `trading_edge`,
  `market_perception`, `paper_session`, and all of §27/§28: `decision_head`,
  `regime_engine`, `strategy_zoo`, `heads`, `router`, `unified_output`,
  `language_heads`.
- **Python (13/13)** — `market_data`, `market_feeds`, `paper_loop`,
  `paper_reports`, `monster_features`, `monster_sniper`, `monster_news`,
  `monster_regime`, `monster_strategies`, `monster_funding`, `decision_bridge`,
  `regime_parity`, `strategy_parity`.
- **Plus** `real_weights` (11 s), `sides` (9 s), `lora_gguf` (488 s).

**30/33 verified green.** The three not completed in this pass are `lora_e2e`,
`lora_chat`, `lora_i8` — LoRA fine-tuning suites, untouched by §27/§28.

**Timing caveat:** this box is currently ~5x slower than when
`CTestCostData.txt` was recorded (`lora_gguf` 488 s vs 91 s recorded;
`qat_ternary` 48 s vs 8 s). The full board takes ~35-45 min here, not the ~10 min
the cost data implies. Budget for it.

**`ctest -I` gotcha:** the flag parses as `Start,End,Stride,test#,test#...`, so
`-I 1,2,3,4,5,10,...` silently runs `Start=1,End=2,Stride=3` **plus** the numbers
after it — not the list you wrote. Use `-R <regex>` instead.

---

## 30. THE SOUL — PERSONA, EMOTIONAL RESONANCE, HONEST SELF-REPORT (2026-09-27)

### The finding that reframed the task

The revival brief assumed the "forgotten features" had to be **built**. An audit
of the tree says otherwise: `runtime/emotional`, `runtime/introspection`,
`runtime/sensory`, `runtime/swarm`, `runtime/cloud_bridge`, `memory/`,
`agent/sub_agents`, `agent/flash_skills`, `vision/`, `audio/` — **8,454 lines —
all exist, all compile, all are in `OMNISEED_CORE_SOURCES`, and several already
have tests** (`test_platform.cpp`, `test_trading.cpp`).

What was actually missing is the **joint**. §28 built a brain that can decide,
classify, score and explain; the "living AGI" half can feel, remember and know
itself. Nothing ever constructed both. `UnifiedPipeline` knew about four
projections of `h[E]` and nothing else.

So milestone 1 is not new capability — it is the joint, plus the honesty gate
that a text layer needs and a numeric layer does not.

### What was built

| file | what |
|---|---|
| `include/omniseed/soul.h` + `src/soul.cpp` | `Persona` (commitments, opinions, stance), `Soul` (perceive / speak / record / resolve / assess), `SoulState`, `Stance` |
| `include/omniseed/unified_output.h` + `src/unified_output.cpp` | optional `soul` section; `Config::use_soul`; `run(hidden, user_turn, text_fn)` |
| `tests/test_soul.cpp` | **242 checks**, 8 parts, registered OUTSIDE the `.venv` gate |
| `docs/SOUL.md` | design, the three rules, the API, the honest limitations |
| `CMakeLists.txt` | `src/soul.cpp` in core; `omniseed_soul` test target |
| `RUNBOOK.md` | the soul API, the test count, the `ctest -I` trap |

### The three ordering rules (each is easy to get backwards)

1. **A refusal is never emotionally softened.** Empathy lead-ins belong on
   answers; on a refusal `"I hear you — "` reads as accepting the premise we just
   declined. So a refusal **replaces** the reply and **suppresses** tone
   modulation. `Refuse` is the only stance that discards the caller's text, and
   therefore the only one that skips tone.
2. **The honesty correction is applied LAST**, so nothing downstream can
   re-introduce an overclaim.
3. **Nothing is claimed that was not measured.** Overconfidence is a number
   (mean stated − realised, over *resolved* decisions) and will not fire below
   `min_calibration_samples`; `init()` rejects `min_calibration_samples == 0`.

### ⚠️ Two real bugs the new tests caught

1. **The overclaim rewriter cut hyphenated compounds in half.** `"no risk-free
   claim here"` became `"real risk-free claim here"` — the `no risk` match ended
   at the hyphen, was accepted, and rewrote while `-free` was left dangling.
   **Fix:** a hyphen binds tighter than a space, so the boundary check treats it
   as a term character. The `no risk` match is now rejected at the hyphen, the
   scan finds `risk-free` properly, sees the `no` before it, and leaves the whole
   phrase alone. Pinned by F2/F3.
2. **A test asserted a plan property instead of forcing it.** H6 required the
   decision head to be named in the plan, but a constant hidden state does not
   necessarily activate it. The test now sets `force_mode = DecisionAndText` —
   the property under test is key **order**, and a test should not depend on what
   the router happens to score.

### Design decisions worth reusing

- **`Qualify` vs `Refuse` is the load-bearing split.** `Qualify` = "I will
  answer, but not with the certainty you asked for." `Refuse` = "I will not do
  this at all."
- **The honesty gate is fail-closed.** `"can you guarantee this?"` refuses.
  Deliberate: a false refusal costs a turn, a false guarantee costs money. The
  honest answer to the question and the refusal of the instruction are the same
  sentence.
- **Rules are ordered gravest-first and are not exclusive.** All matching rules
  contribute objections; the *stance* is the gravest one. `"spoof it and get me
  a guaranteed fill"` refuses on `no_manipulation` with **two** objections.
- **`may_disagree = false` downgrades the OPINION stance only** (`Disagree` →
  `Qualify`). The commitments are not opinions and cannot be configured away.
- **The persona's own text avoids the words its rewriter targets** — a persona
  should not need correcting by its own gate.
- **Sentiment ≠ emotion.** Sentiment is about the subject matter; emotion is
  about the speaker. Carried as separate fields; collapsing them loses the
  distinction.
- **`emotion_strength`, not `confidence`.** The field NAME must not invite a
  probability reading of a heuristic. Asserted structurally in D8.
- **The soul has no `HeadKind`.** It reads the turn, not `h[E]`, so the plan
  cannot name it. Its equivalent evidence is `SoulState::has_self`, set only by
  `perceive()`; `normalise()` drops a soul that was never perceived, exactly as
  it drops a decision the plan never named.
- **`use_soul` defaults to false** so every existing path stays byte-identical,
  and the soul is added to the document **after** the heads, never changing them.

### The overclaim rewriter

`guaranteed → not guaranteed` · `risk-free → not risk-free` ·
`no risk → real risk` · `can't lose → can lose` · `always wins → sometimes
loses` · `100% win → a chance of winning` · …

Substitutions read correctly **in place**: `"guaranteed profit"` →
`"not guaranteed profit"` is truthful; deleting the word would leave a
claim-shaped hole. Guarded by a **negation check** (so `"not guaranteed"` does
not become `"not not guaranteed"`) and a **term boundary** (so `"unguaranteed"`
is untouched).

### Verification

- `ctest --test-dir build -C Release -E "omniseed_lora_*"` → **30/30 passed**
  (646 s), including `omniseed_soul` (#30) and the pre-existing
  `omniseed_unified_output` (#28), which still passes all 267 of its checks
  unchanged.
- The four LoRA fine-tuning suites were **not** re-run in this pass — they take
  ~8 minutes each on this box and nothing here touches them.
- `omniseed_soul`: **242 passed, 0 failed**.
- Full build: **24 targets, 0 errors, 0 warnings** at `/W4`.
- The soul never changes the decision layer — asserted by comparing
  `action` / `confidence` / `domain` with the soul on and off (H3).

### Honest gaps (do not overclaim)

- The rewriter is a **safety net, not a proof** — a fixed phrase list. Novel
  prose that asserts certainty passes through.
- Commitments are **surface-cue rules**, and they fail closed.
- `emotion_strength` is **not calibrated**; the voice channel is a ZCR/energy
  proxy, not a trained SER model.
- **The soul does not fix the heads.** `provenance()` still reports
  `NOT FULLY FITTED (placeholder)`.
- Not yet wired: the soul is reachable from the library and the tests, but **no
  CLI command uses it yet**; memory crystals and the dream pass are still not
  composed with it (that is the next milestone).

### Next

1. Wire `Soul` into the CLI chat path (`speak()` around generated text) and add
   `omniseed introspect` using `capability_report()`.
2. Memory crystals → `SoulState` recall, so the soul remembers what it said.
3. Dream consolidation: schedule `omniseed dream` and have it decay crystals and
   prune the rationale log.

---

## 31. MEMORY & RECALL — THE SOUL REMEMBERS (2026-09-27)

Milestone 2 of the memory/dream/calibration sequence. §30 gave the soul a voice
and a self-report; it still forgot every exchange the moment the turn ended.

### The finding

`MemoryCrystals` (512 lines, `memory/memory_crystals.h/.cpp`) **already existed,
already compiled, and was already tested** — `test_platform.cpp` covers
crystallize, retrieve and save/load. But it was reachable only from
`AgentLoop::build_prompt()` and one CLI demo. The soul had no memory, and the
memory had no soul. As in §30, the missing piece was the **joint**, not the
capability.

Reading the decay path first was what made this milestone worth doing: the
mechanism was **inverted**, and nothing had ever called it.

### ⚠️ The decay bug: the model ran backwards

`decay(double now_unix_seconds)` **ignored its argument** and derived age as
`last_access_token / 100000.0` — reading a *stream position* as if it were
already an *elapsed age*. Together with `retrieve()` writing
`last_access_token = created_at_token + query_tokens.size()`, two inversions
compounded:

| | before | after |
|---|---|---|
| a crystal created at position 0 | **immortal** (age always 0) | ages normally |
| a crystal created late | decayed **immediately** | ages normally |
| effect of recall | made the memory **older** | **resets** the age |
| the `now` argument | ignored | honoured |

So "fade unless reinforced" was exactly backwards: using a memory could not save
it, and creating one late could kill it. **Zero callers** existed, so the fix
cost no compatibility.

Fixes: `crystallize()` stamps `last_access_token = stream_pos` at birth;
`decay(uint64_t now_token)` computes `age_days = (now − last_access)/tokens_per_day`
(clamped at zero, so a clock behind the stamp cannot yield importance > 1.0);
`retrieve(..., now_token)` refreshes recency; `reinforce(id, now_token, boost)`
raises importance and resets the age. `tokens_per_day` (100k) replaced a literal.

### ⚠️ Two false-memory bugs — found by the demo, NOT by the tests

This is the part worth reading. The 378-check suite was **fully green** while the
feature was wrong in two ways that only showed up when `demo-soul` printed a
conversation:

1. **Recall had no relevance floor.** `retrieve()` returns the *k nearest*
   crystals whatever they say, and "nearest" among a handful of unrelated
   memories still looks like an answer. An unrelated question scored **0.775**
   and the soul confidently announced `"I said this before: ..."` about something
   it had never seen. **Fix:** `min_relevance`, measured (below) — not guessed.
2. **The gloss was being stored.** Turn 2's reply included turn 1's gloss, so
   turn 3 quoted itself: `"I said this before: \"I said this before: ...\""` —
   compounding every turn. **Fix:** the gloss is composed in `converse()` and
   **never stored**; what is stored is the reply the owner would have received
   with no memory at all.

Both were caught by *looking at the output*, not by an assertion. The lesson is
recorded here because it generalises: a green suite proves the properties you
thought to assert, and nothing else.

### What was built

| file | what |
|---|---|
| `include/omniseed/memory/memory.h` + `src/memory/memory_crystals.cpp` | corrected `decay`/`retrieve`, `reinforce`, `find`, `ids`, `crystallize(..., out_id)`, `tokens_per_day`, `MemoryCrystal::score`, embed-dim clamp |
| `include/omniseed/soul.h` + `src/soul.cpp` | `RecalledMemory`, `MemoryRole`, `recall_note()`, `SoulState` recall/memory sections, `perceive_and_recall`, `remember`, `converse`, `recall`, `top_match`, `decay_memories`, `reinforce_memory`, `advance_memory_clock`, `save/load_memories` |
| `include/omniseed/unified_output.h` + `src/unified_output.cpp` | non-const `run_memorable()`; `provenance()` reports the crystal count |
| `src/cli/main.cpp` | `omniseed demo-soul` |
| `tests/test_soul.cpp` | parts **I** and **J**: 378 checks total (was 242) |
| `docs/SOUL.md`, `RUNBOOK.md`, this file | design, the measured thresholds, the API |

### Design decisions

- **`perceive()` stays `const` and stores nothing.** `UnifiedPipeline::run()` is
  a const method that calls it, and a const method that silently appends to
  long-term memory is a trap: the caller cannot see the write in the signature,
  and `run()` stops being idempotent in a way no type can express. Remembering
  therefore lives in **named non-const** entry points — `perceive_and_recall()`,
  `remember()`, `converse()`, and `run_memorable()`. I10 asserts the const
  `run()` really does not store.
- **`converse()` is the only stateful path.** `speak()` remains byte-identical to
  §30: the memory gloss is an annotation on the turn, not an utterance, so it is
  composed outside `speak()` and is not emotionally modulated. Rule 2 still holds
  — the overclaim rewriter runs last, on the composed string.
- **A refusal is never decorated with a memory.** A refusal is the whole reply;
  attaching "we discussed this" would read as negotiating the thing just
  declined. I7 forces a turn that *both* recalls and refuses, so the suppression
  is tested rather than accidentally true.
- **A recalled question brings its answer.** A reply's words rarely resemble the
  question's, so the reply crystal never clears the floor against its own
  question — the soul would only ever quote *the owner*. Answers are therefore
  linked to their question at store time (`MemoryLinks`) and inherited, because
  the answer is relevant **by construction**, not by its own cosine.
- **Repeated exchanges are reinforced, not re-filed.** A question scoring ≥ 0.98
  against one already held is reinforced; if its answer is also unchanged, the
  answer is not re-filed either. Without both, a store that appends per turn is a
  log file. `demo-soul` shows turn 3 repeating turn 1 with the store staying at
  **4 crystals**.
- **`max_len_tokens` is 96 for the soul**, not the memory layer's 48 — 48
  truncates an ordinary sentence mid-word, and a soul that quotes a half-sentence
  is worse than one that says nothing. Scoped to `Soul::Config` rather than
  changed in the shared default, which `AgentLoop` relies on.
- **Turns under the 8-token floor are counted** (`memory_skipped()`), not padded
  to fit.

### The thresholds are measured, not guessed

`demo-soul` prints this table on every run and labels each row `ok`/`MISMATCH`:

| probe | score | verdict |
|---|---|---|
| a question the store holds (verbatim) | 0.988 | claimed |
| the same, one character off | 0.990 | claimed |
| a paraphrase of it | 0.973 | claimed |
| a question from an earlier turn | 0.989 | claimed |
| a question **never** seen (short) | 0.828 | no claim |
| a question **never** seen (long) | 0.876 | no claim |

`min_relevance = 0.92` sits in the gap. `duplicate_recall_score = 0.98` sits
above every true match and below the 1.0 ceiling.

### Verification

- `omniseed_soul`: **378 passed, 0 failed** (was 242). Parts A–H unchanged and
  still green, which is what proves `speak()`'s byte-exact ordering survived.
- **Full board: `ctest --test-dir build -C Release` → 34/34 passed (3300 s).**
  The whole board, including all nine LoRA/Python suites, not just the C++
  half — the slowest single suite was `omniseed_lora_chat` at 1232 s. This box
  is ~5× slower than `CTestCostData.txt` implies; the board takes ~55 min.
- `omniseed demo-soul` end-to-end: turn 2 (unrelated) claims nothing; turn 3
  (repeated) quotes the soul's own prior answer, untruncated; the store stays at
  4 crystals; 3 of 4 crystals decay away at +50 days while the reinforced one
  survives.
- Full build: **0 errors, 0 warnings** at `/W4`.
- The persistence format is **unchanged**: `save()`/`load()` are untouched and
  the version stays 2, so existing crystal sidecars still load. `MemoryCrystal`
  gained a `score` field, which is a retrieval artifact and is deliberately not
  serialised.

### Honest gaps (do not overclaim)

- **The relevance margin is thin and it narrows as the store grows.** Byte-level
  bags are close to character histograms, so unrelated English still scores
  0.83–0.88 and the floor sits in a ~0.10-wide gap. With the trained 8k–16k
  vocabulary the tokens are words rather than bytes and discrimination is far
  better — **but both thresholds must be re-derived** (`demo-soul` prints the
  numbers). Until then, recall is a **near-verbatim** matcher, and it
  under-claims rather than over-claims.
- **Speaker roles and question/reply links are session-local** and not persisted,
  so crystals restored from disk report role `unknown` and are unpaired.
- **No CLI chat path uses any of this yet** — `demo-soul` is the only caller.
- **The dream pass does not call `decay_memories()` yet** (that is milestone 3),
  and calibration is still untouched (milestone 4).

### Next

1. **Milestone 3 — dream consolidation:** scan the last 24 h of crystals and the
   rationale log, extract what worked/failed, strengthen or decay, write
   `state/dream_log.json`. The decay and reinforcement primitives now exist for
   it to call.
2. **Milestone 4 — head training & calibration** (highest value).
3. Re-derive `min_relevance` / `duplicate_recall_score` against a real vocabulary
   (the tokenizer is the root cause of the thin margin).

---

## 32. HEAD TRAINING & CALIBRATION — THE CONFIDENCE BECOMES REAL (2026-09-28)

**Milestone 4. Milestone 3 (dream consolidation) was deliberately skipped.**

§28 built the head stack and §30–31 gave it a soul and a memory, but every head
was still a **seeded placeholder** that emitted a meaningless number. §28's
router gates the fast path at `0.85`, so a placeholder that is right 24% of the
time could print `0.92` and self-route. This milestone makes the number mean
something. Full write-up: **`docs/CALIBRATION.md`**.

### What was built

- `tools/make_head_data.py` — authors `tools/data/head_language.tsv`
  (**238 rows**, Bengali **in Bengali script**, not romanised; refuses
  duplicates and embedded tabs/newlines).
- `tools/dump_hidden.cpp` (new binary `omniseed_dump_hidden`) — collects `h[E]`
  from ONE RWKV-7 forward pass. Modes `language` and `market`. **Streaming mode**
  feeds all 1,231 bars through one evolving `RwkvState`, because RWKV is
  recurrent and constant-memory and integer-epoch timestamps in independent
  windows would be OOD. Honesty contract: absent backbone or zero-token text ⇒
  write nothing, exit non-zero. It never synthesises a hidden state.
- `tools/train_heads.py` (~750 lines) — the offline trainer. Parses the C++
  vocabulary so there is **one source of truth**; `fit_softmax` (Adam on the
  L2-regularised multinomial NLL, initialised at class log-priors);
  `fit_temperature_cv` (**5-fold inside the holdout**, geometric mean of `log T`,
  out-of-fold ECE); `emit_fixture` (emits the **whole** holdout).

### Format bumps (both backward compatible)

- **`DecisionHead` v2 → v3.** v3 appends a trailer **written last**
  (`f32 temperature`, `i32 calib_samples`, `f32 calib_error`) so a v2 reader
  stops cleanly at the bias block. `load()` accepts `2 ≤ version ≤ 3`; **v2 loads
  with `T = 1.0` and calibration unmeasured**, and **v1 is still rejected** (a
  v1 blob carried no invalidation). A non-finite or `≤ 0` `T` is normalised to
  `1.0` rather than stored.
- **`ClassificationHead` v1 — persistence written from scratch.** It had none.
  `load()` rebuilds the sets from the file in file order, then overwrites with
  the stored projection; every failure path sets `ready_ = false` (**fail
  closed**).
- `.gitignore` gains `!models/heads/`. `models/*` swallows the whole directory,
  so the negation has to **name the directory itself** before its contents can be
  re-included.

### ⚠️ One temperature per label set, not one per head

A single pooled `T` made `language.language` **worse** (ECE 0.172 → 0.194) while
fixing `language.intent` (0.451 → 0.115): `intent` wants `T ≈ 23`, `language`
wants `T ≈ 5`. Sets whose logit scales differ by an order of magnitude cannot
share a scalar. `ClassificationHead` therefore stores a temperature **per set**.

A temperature is a **global soften, not a local patch** — softening the top
bucket softens every bucket, so a scalar `T` cannot fix bucket-*local*
miscalibration.

### ⚠️ Fitting `T` on a small slice under-softens every set

Fitting on a 15% slice (~36 rows) gave `T` **1.5–2× too small** against the
oracle (intent 19.4 vs 28.7; language 3.5 vs 6.5; sentiment 2.5 vs 4.7). Fixed by
5-fold CV inside the holdout. `shuffle=False` for the trading data — a shuffled
fold of a time series is not held out.

### Measured results (all held out)

| head | K | n | accuracy | macro-F1 | T | ECE raw → cal |
|---|---|---|---|---|---|---|
| `language.language` | 4 | 71 | 0.845 | — | 5.161 | 0.139 → **0.089** |
| `trading.regime` | 4 | 369 | 0.705 | 0.389 | 3.250 | 0.210 → **0.034** |
| `language.sentiment` | 3 | 71 | 0.676 | — | 3.583 | 0.248 → **0.087** |
| `language.intent` | 7 | 71 | 0.634 | — | 23.441 | 0.360 → **0.198** |
| `DecisionAction` | 7 | 369 | **0.244** | 0.133 | 13.325 | 0.622 → **0.056** |

### ⭐ The payoff: calibration makes the fail-closed gate actually fail closed

```
DecisionAction  max calibrated confidence 0.4966  <  threshold 0.85
                => 0/369 held-out rows self-route (fail closed)
```

The raw softmax on this head claims high confidence routinely; the head is right
**24%** of the time. After calibration its **maximum** confidence across all 369
held-out rows is **0.4966** — it never claims to know, so every row escalates to
System-2. **That is the correct behaviour for a 24%-accurate head, and it was
invisible before calibration.**

### Honest gaps (do not overclaim)

- **`DecisionAction` accuracy is 0.244 and `CLOSE`/`HEDGE` have zero holdout
  recall.** Per-class recall `{ABSTAIN 0.20, HOLD 0.015, BUY 0.324, SELL 0.474,
  CLOSE 0.0, HEDGE 0.0, EXPLAIN 0.026}`. Well-calibrated *and* weak — calibration
  makes a weak head **safe**, not good.
- **`trading.regime` never predicts `trend_down`** (8 of 1,231 bars). Quote the
  macro-F1 0.389, not the 0.705.
- **`trading.regime`'s `[0.9,1.0]` bucket misses the mandate's ±3% target at
  3.23%.** The test asserts a 5% quality bar, **always prints the exact gap**,
  and prints `MISSES the mandate's 3% target`. Reported, not hidden.
- **The action labels are a rule, not P&L.** `build_action_labels` is a
  definition (N=5 forward, 1.5% threshold); accuracy measures imitation, and
  says nothing about profitability.
- **`assets` / `invalidation` are empty / zero** in the blobs — "unknown", never
  "no stop".
- **Mandate discrepancies, reported:** `models/market/paper/AAPL_1d.csv` does not
  exist (real path `models/market/AAPL_1d.csv`); the CSV has **no `regime`
  column** (computed in-script by the real `RegimeEngine`); `set_action_row()` /
  `fitted_rows()` are really `set_action()` / `fitted_actions()`; `n_samples=1255`
  was an assumption (real: 167/71 language, 862/369 trading).

### Tests

`tests/test_calibration.cpp` — **180 checks, 0 failures**, registered **outside**
the `.venv` gate. Part A/B are ungated mechanics (v3 round-trip, **v2 still
loads**, **v1 still rejected**, failed load fails closed, `UNTRAINED` honesty);
Part C is gated on the committed artifacts and recomputes ECE/accuracy from the
fixture, asserting they match the stored metrics to 0.005 and that
`calibrated.ece <= raw.ece`. All six pre-existing head suites re-run clean.

**Full local board: `ctest --test-dir build -C Release` → 35/35 passed**
(3646 s, `.venv` present, so the 14 venv-gated Python suites ran too). The board
grew **34 → 35**; a fresh clone goes **20 → 21** (the calibration suite is
registered outside the gate).

`.gitattributes` is new: it forces `eol=lf` on the fixture `.tsv`/`.json` and
marks the `.f32`/`.bin` payloads **binary**, because `core.autocrlf=true` would
otherwise rewrite a `0x0D 0x0A` byte pair inside a float32 on checkout and
silently drift every metric in the fixture.

### Next

1. **Milestone 3 — dream consolidation** (still skipped; the decay/reinforce
   primitives exist for it to call).
2. **Fix the action teacher or replace it with realised outcomes** — 0.244 is a
   floor set by the rule, not by the head.
3. Re-derive `min_relevance` / `duplicate_recall_score` against a real vocabulary.

---

## 33. THE MULTIMODAL JOINT — AN IMAGE BECOMES h[E] (2026-09-28)

**Milestone 5.** §32 numbered this section §32, so the next is §33 — the
"MASTER EXHAUSTIVE BUILD MANDATE" predicted §34 because it assumed §33 was
already spent. It was not.

### The audit that changed the plan

The mandate listed ~60 capabilities as missing. **Most already existed.** An
audit before building found `src/vision/` (659 lines), `src/audio/` (1,626 lines
incl. a 1,082-line Whisper-tiny), `runtime/{token_bus,emotional,sensory,swarm,
introspection,cloud_bridge}`, `memory/{memory_crystals,prefix_cache,
streaming_llm}`, `agent/{sub_agents,flash_skills,agent_loop,agent_intel,
self_improvement,grammar_decoder,tool_registry,compute_throttle}`, and
`core/uncertainty.h` — **all real, all in `OMNISEED_CORE_SOURCES`, zero orphans
in `src/`.** Vision already had `PointerPerception`, `SpatialContextMapper`,
`MotionTracker`, `GestureRecognizer`. Audio already had `WhisperTiny`,
`FocalCodec`, `SoundEventDetector`, and `WakeWordDetector::enroll()` — the
mandate's "voice enroll for owner identity".

`TokenBus` even documented its own job as *"fuses text, vision and audio token
streams into ONE unified sequence for the RWKV core"* — and had **zero tests**.
Nothing anywhere turned an image into the `h[E]` the heads read. **As in §30 and
§31, the gap was the joint, not the capability.**

### The impedance mismatch, and the fix

```
VisionEncoder::encode()  -> Tensor [M, n_embd]    CONTINUOUS
FocalCodec::encode()     -> vector<int32_t>       ids
TokenBus::fuse()         -> wants vector<int32_t> for EVERY modality
RwkvModel::forward()     -> takes int32_t token
```

Audio already fit; **vision did not**. `MultimodalBridge`
(`include/omniseed/multimodal.h`, `src/multimodal.cpp`) closes that one gap by
quantizing a vision embedding to the **nearest vocabulary token using the
model's own embedding matrix as the codebook**. No training, no extra RAM (one
new accessor, `RwkvModel::token_embeddings()`, a non-owning view), and the
modality lands in the discrete space `forward()` accepts. It re-implements no
encoder, tokenizer or bus.

### Two bugs the tests exist to prevent

1. **The early-out needs a flag.** Abandoning a candidate row when its partial
   sum exceeds the best-so-far is what makes a 65,536-row scan affordable — but
   the partial sum then compares as *smaller* than `best_d`, so without a
   `worse` flag the row is accepted **precisely because it was cut short**.
2. **A NaN must not win.** Every comparison against NaN is false, so a NaN query
   would be "nearest" to the first row examined. The query is finiteness-checked
   before the scan.

Two design rules, both tested: **a modality that was supplied is never silently
dropped** (vision with no codebook FAILS with a reason rather than answering
from text alone), and **a failed fusion is not an empty one** (`ok == false`
always carries an `error`).

### ⭐ The finding: the joint works, and the head still lies

C1 runs the real chain against `models/rwkv7-0.1B-ternary.gguf`:

```
fused 10 tokens (text 3, vision 4, audio 0)
vision.scene top1 = urban (0.997)   <- head is UNFITTED
```

An image became 4 vocabulary ids, wrapped in `<|vision|> … </|vision|>`, ran
through the backbone, and its `h[E]` was read by a head. **And that unfitted
head printed 0.997.** This is §32's entire argument restated on a new modality:
a seeded projection emits a confident number with no knowledge behind it.

### Tests

`omniseed_multimodal` — **110 checks, 0 failures, 0 skipped**, registered
**outside** the `.venv` gate. Part A is the **first test `TokenBus` has ever
had** (7 checks). Part B is ungated (a minimal tokenizer + a synthetic basis
codebook, so it runs in CI). Part C is gated on the model and prints a SKIP
reason without it. `codebook_stride` is pinned as a **real accuracy dial**: with
`stride = 2` a token whose nearest row has an odd index is unreachable.

**Full local board: `ctest --test-dir build -C Release` → 36/36 passed**
(3381 s). The board grew **35 → 36**; a fresh clone goes **21 → 22** (the
multimodal suite is registered outside the gate). All six pre-existing head
suites re-run clean, so the new `rwkv.h` accessors broke nothing.

### Honest gaps (do not overclaim)

- **`vision.scene`, `vision.anomaly`, `audio.wake`, `audio.emotion`,
  `audio.speaker` are UNFITTED.** They have label sets and no weights. This
  milestone built the pipe, not the water — they need exactly the §32 treatment.
- **The codebook is untrained and lossy.** It is not a learned modality adapter.
- **The audio joint is untested at real-weights level.** `FocalCodec`/`Whisper`
  produce ids that fit the bus directly, but no test runs a wav through the
  whole chain.
- **`VisionEncoder` is not called in C1** — it needs `models/vision-proj.gguf`
  and has its own suite; C1 isolates the join with synthetic `[M, E]` rows.
- **Still zero coverage — four modules**, checked by CLASS NAME, not filename:
  `ComputeThrottle`, `SelfImprovement`, `GrammarDecoder`, `FocalCodec`. A
  filename grep is a proxy and it lied: `VisionEncoder`/`UniCompress`
  (`test_sides.cpp`), `WhisperTiny`/`SoundEventDetector`/`WakeWordDetector`
  (`test_platform.cpp`), `StreamingLlm` (`test_platform.cpp:933`),
  `ToolRegistry` and `AgentLoop` are all already exercised. HARD RULE 5 ("every
  new module gets tests") remains unmet for those four.
- **No `demo-multimodal` CLI**; the bridge is reachable from C++ and the test.

### Next

1. **Fit the vision/audio heads** — collect `h[E]` with `tools/dump_hidden.cpp`,
   extend `tools/train_heads.py`, calibrate, store `models/heads/vision_head.bin`.
2. **Milestone 3 — dream consolidation** (still skipped).
3. **Coverage for the four untested modules** — `ComputeThrottle`,
   `SelfImprovement`, `GrammarDecoder`, `FocalCodec` (HARD RULE 5).

---

## 34. FULL MODULE COVERAGE — AND THE FOUR DEFECTS IT FOUND (2026-09-28)

**Milestone 6. HARD RULE 5: "every new module gets tests".** §33 left four
classes with no coverage at all, checked by CLASS NAME. This milestone closes
that, and writing the tests immediately found **four real defects**.

Full write-up: **`docs/AUDIT.md`** (which also records the mandate-vs-reality
audit, since that is itself a deliverable).

### ⚠️ The worst one: a feature that silently never worked

`SelfImprovement::save()` wrote a `uint32_t` magic `0x54524953`;
`load()` compared against the **string** `"SRIT"`. Little-endian, `0x54524953`
is the bytes `'S','I','R','T'` — so the comparison **always failed**. `load()`
rejected every file `save()` wrote.

The failure was invisible: **an unreadable cache is indistinguishable from an
empty one.** `src/cli/main.cpp` calls `imp.load("./state/improve.bin")` and has
always got `false`, so the self-improvement trace cache **has never persisted
across runs**. Fixed with one shared `kMagic` constant; pinned by C9.

### The other three

| defect | status | pin |
|---|---|---|
| `load()` read `klen`/`ns`/`sl` straight from the file and never bounds-checked them against the mapping — a truncated `state/improve.bin` walked off the end of the mmap | **FIXED** (`need()` before every read, a trace-count cap so a 12-byte file cannot drive a 4-billion `reserve()`, and a local vector swapped in only on full success so a failed load keeps the previous traces) | C7 |
| `task_key()` did not trim **leading** whitespace (`last_ws` started `false`, and the trim loop only strips the END), so `"  read a file"` and `"read a file"` were two different keys and the exact-match replay lookup missed | **FIXED** (`last_ws = true`) | C1 |
| `FocalCodec::encode()` computes a 4-bit bucket per band and then keeps **only its low bit** (`q & 1u`), discarding 3 of every 4 — the code is a 24-bit sign pattern, not the quantizer its comment describes | **PINNED** (changing it changes every emitted code; a fitted audio head would need retraining) | D6 |
| `ComputeThrottle::classify()` matches Fast keywords by **substring**, so `"hi"` fires inside `"which"` — `classify("which stock should i buy today")` returns **Fast** and a real trading question gets 48 tokens / 0 thinking tokens | **PINNED** (thresholds are a tuned heuristic) | B6 |
| `GrammarDecoder::accepts()` allows content after a complete object: `accepts("x")` is true on a decoder holding `{}`, and `feed("x")` reports `complete()` on the invalid `"{}x"` | **PINNED** (it gates constrained generation) | A8 |

**Why three are pinned rather than fixed:** each has a behavioural contract
that something downstream depends on. Changing what a codec emits, what a
classifier returns, or what a generation loop is allowed to produce is a
decision for the owner, not a silent edit. They are asserted so they cannot
drift, and reported here so the cost is visible.

### Tests

`omniseed_agent_modules` — **131 checks, 0 failures, 0 skipped**, fully
**UNGATED** (all four modules are pure and model-free, so the suite runs in CI).
Part A `GrammarDecoder`, Part B `ComputeThrottle`, Part C `SelfImprovement`
(including five distinct malformed-file shapes), Part D `FocalCodec`.

**Full local board: 37/37 passed.** The board grew **36 → 37**; a fresh clone
goes **22 → 23**. No pre-existing suite depends on the old `task_key` behaviour.

### Coverage after this milestone

**Every class in `src/` is referenced by at least one test.**

### Next

1. **Fit the vision/audio heads** (the largest verified absence).
2. **Milestone 3 — dream consolidation** (still skipped).
3. ~~Epistemic/aleatoric separation~~ — **done in §35.** `core/uncertainty.h`
   gives entropy and margin (a *total* signal); the split is now built and
   measured against the real held-out heads.

---

## 35. UNCERTAINTY, SPLIT — WHY am I unsure? (2026-09-28)

**Milestone 7.** `core/uncertainty.h` answers *how* unsure a distribution is,
with one number. It cannot say **why**, and the two reasons call for opposite
responses: **aleatoric** doubt means the classes overlap and more data will not
help (stay out); **epistemic** doubt means the input is outside the fitted
region and more data *would* help (go and learn).

Full write-up: **`docs/UNCERTAINTY.md`**. Reproduce every number below with
`python tools/uncertainty_audit.py`.

### The audit that shaped the design (run BEFORE building)

`h[E]` is post-`ln_out`, so the obvious worry was that row norms are constant
and a Euclidean distance is meaningless. Probed on the committed holdout:

| fixture | row-norm sd/mean | cos(h, mean) |
|---|---|---|
| trading (369×768) | **0.0198** — near-isotropic | 0.9928 ± 0.0025 |
| language.intent (71×768) | 0.1394 | 0.6960 ± 0.1027 |

So the concern was real for trading. Two consequences: the distance is
**normalised per dimension by σ** (a diagonal Mahalanobis, not a raw norm), and
a dimension whose σ is unmeasurably small is **floored at 10% of the typical
σ** rather than at an epsilon — otherwise one dead dimension dominates the
whole distance (measured: without the floor a 1.0 deviation in a constant
dimension yields a distance in the hundreds of thousands; with it, 4.29).

The probe also produced the finding that shaped the whole milestone: the
distance **does not** track error within a domain, and **does** separate
domains completely. Both are asserted as tests.

### Aleatoric is real: monotone in the error rate for 4 of 5 heads

Error rate per quartile of **calibrated** normalised entropy, real holdout:

| head | K | n | acc | err by entropy quartile | |
|---|---|---|---|---|---|
| `trading.regime` | 4 | 369 | 0.705 | 0.120 → 0.196 → 0.413 → 0.452 | **monotone** |
| `language.sentiment` | 3 | 71 | 0.676 | 0.059 → 0.353 → 0.353 → 0.500 | **monotone** |
| `language.language` | 4 | 71 | 0.845 | 0.000 → 0.000 → 0.235 → 0.350 | **monotone** |
| `language.intent` | 7 | 71 | 0.634 | 0.235 → 0.235 → 0.353 → 0.600 | **monotone** |
| `DecisionAction` | 7 | 369 | **0.244** | 0.772 → 0.804 → 0.685 → 0.763 | no signal |

The fifth is honest, not a defect: at 0.244 accuracy over 7 classes the head is
near chance (0.143), so nothing predicts its errors and the tool prints
`no signal` instead of inventing a trend. This is §32's finding restated.

### Epistemic has NO within-domain signal — and that is the correct result

Ordered by `h[E]` distance, the same quartile error rates are non-monotone and
sometimes **inverted** (`language.language`: 0.294 → 0.118 → 0.235 → 0.000).
This is what theory predicts: inside one domain every input is in-distribution,
so there is no epistemic uncertainty to find and the top decile is noise. A
module that found a signal here would be one firing on noise.

### Epistemic DOES separate domains, completely

| reference | scored | in-domain d (mean / p90 / max) | out d (mean / min) | below in-domain p90 |
|---|---|---|---|---|
| trading | language.intent | 0.990 / 1.169 / 1.614 | 6.699 / **5.129** | **0 / 71** |
| language.intent | trading | 0.981 / 1.198 / 1.683 | 1.719 / 1.482 | **0 / 369** |

The first direction is a clean 5× jump — the *minimum* out-of-domain distance
is 3× the *maximum* in-domain distance. The reverse is weaker (n=71 makes a
coarse CDF) but every trading vector is still past the language reference's
p90, and 249/369 saturate at 1.0. The gap is total because the domains differ
in **both** statistics: trading has the larger row norm (262 vs 152) and the
**smaller** per-dimension σ (0.93 vs 3.05) — a 3.3× gap, in the *unintuitive*
direction, which is why the test asserts the magnitude of the gap and not its
sign.

### The rigorous operator exists; the ensemble to feed it does not

`from_ensemble()` implements the Depeweg 2018 decomposition **exactly** —
`total = H(mean p)`, `aleatoric = mean H(p)`, `epistemic = total − aleatoric ≥ 0`
by Jensen — and is tested (identical members → 0; two disjoint point masses →
`ln2 / 0 / ln2`). **Nothing can feed it**: no head ensemble has been fitted, so
there is no `q(θ)`. As in §30, §31 and §33, the gap is the **joint**, not the
capability. `split()` therefore uses the two *measured* proxies and the docs say
so rather than implying a Bayesian decomposition it cannot perform.

### Fail-closed policy

Every failure resolves to **maximum doubt**, never to confidence: no reference,
a non-finite `h`, or a width mismatch ⇒ `epistemic() == 1.0`; a vector that is
not a distribution (negative, all-zero, NaN) ⇒ `aleatoric() == 1.0`. `load()`
clears the object **before** reading, so a caller that ignores the return value
cannot keep a stale reference alive. `distance()` returns `−1.0` ("not
measured") rather than 0.0, because a distance is a measurement.

### Files

| file | what |
|---|---|
| `include/omniseed/core/uncertainty_split.h` | the module (NEW) |
| `src/core/uncertainty_split.cpp` | the implementation (NEW) |
| `tests/test_uncertainty_split.cpp` | **154 checks**, 0 fail, 0 skip, **UNGATED** (NEW) |
| `docs/UNCERTAINTY.md` | full write-up (NEW) |
| `tools/uncertainty_audit.py` | the oracle behind the tables (NEW) |
| `CMakeLists.txt` | source + `omniseed_uncertainty_split` test target |

The test is ungated because it needs only the **committed** `h[E]` fixtures —
no model, no `.venv`, no network. Part E is the real-data validation and it
asserts the **cross-domain positive** (100% flagged) and the **in-domain
negative control** (≤ 20% flagged) *together*: a detector that flags everything
passes one and fails the other, and a detector that flags nothing does the
reverse.

**Full local board: 38/38 passed.** The board grew **37 → 38**; a fresh clone
goes **22 → 23**.

### Next

1. **Fit the vision/audio heads** (still the largest verified absence).
2. **Milestone 3 — dream consolidation** (still skipped).
3. **Wire the split into the router's fast path** — a `Both` verdict is a
   strictly better reason to escalate to System-2 than a bare confidence floor.
   The module and its evidence exist; the consumer does not.
4. **Fit a head ensemble** so `from_ensemble()` has something to eat — that is
   the prerequisite for the *rigorous* decomposition, and for Phase 2.5
   (cross-head ensemble voting).
5. **Calibrate the two thresholds** (0.5 / 0.90 are interpretable defaults, not
   fitted). §2 of `docs/UNCERTAINTY.md` provides the data to do it.




---

## 36. BATCHED HEAD EVALUATION — 1000 signals, 5.62x (2026-09-28)

**Milestone 8. Phase 2.1 of the mandate.** Every head took one `h[E]` at a time —
the right shape for a live turn, the wrong shape for a backtest. This adds
`classify_batch` / `decide_batch` / `score_batch`, built on a shared
`batch_gemm` (`[B, E] × [L, E]ᵀ + bias[L]`).

Full write-up: **`docs/BATCH.md`**.

### What is batched, and what is NOT — stated first

**Batched: the head readout** (a GEMM, embarrassingly parallel).
**NOT batched: the backbone.** This tree's RWKV-7 is the **scalar recurrent**
form, so `h[E]` for bar *n+1* needs the state from bar *n*. "One forward pass for
1000 signals" is therefore **false and is not claimed**. What this delivers is
one GEMM for 1000 *readouts*, after 1000 necessarily sequential forwards.

That is not a defect, because of where the win is: on the live path the backbone
dominates and batching buys nothing, but on the **backtest path** the forwards
are already paid for and cached to `hidden.f32`, and the readout — with its
per-call scratch resize, `order` vector, `top_k` vector and two clock reads,
repeated once per bar — was the remaining cost.

### Measured (E = 768, Release)

| measurement | value |
|---|---|
| Simd vs Scalar, worst absolute difference (B=64, E=768, L=7) | **1.07e-06** |
| Simd vs Scalar, worst relative difference | **1.07e-04** |
| `classify_batch`, 369 real rows | 9.5 → **1.7 µs/row** (5.6×) |
| `decide_batch`, 369 real rows | 9.9 → **2.0 µs/row** (5.0×) |
| **whole stack, 1000 signals** | **25.27 ms → 4.50 ms** |
| **speedup** | **5.62×** |

The "per-row" column is the *batched* call with the Scalar kernel — same code
path, same arithmetic, only the per-call overhead removed — so this is an
apples-to-apples number, not batched-vs-nothing.

### Two kernels, two contracts (the part that matters)

| kernel | accumulation | vs the per-row path |
|---|---|---|
| **Scalar** | one running accumulator, `e = 0…E-1`, in order | **bit-identical** |
| **Simd** | eight AVX2 lanes, reduced at the end | within tolerance, **not** identical |

Floating-point addition is not associative, so a lane-reduced sum cannot
reproduce a serial one. The tests therefore assert **`==`** on Scalar and a
**tolerance** on Simd — and `A5` additionally asserts the two **do differ**, so a
future change that made them identical (SIMD not taken, or quietly rewritten as
a serial sum) fails the test rather than passing it.

`Auto` is the default and resolves to Simd, so **the default path is the fast,
inexact one**. That is deliberate — it is the point of batching — but it means
exactness must be asked for: `set_batch_kernel(BatchKernel::Scalar)`.

The default path is held to a weaker but more meaningful contract: **the answer
does not change.** Measured: **0** top-1 label mismatches, **0** action
mismatches, **0** routing mismatches over 369 real held-out bars + 200 synthetic.

### Fail-closed policy

Every batch call returns `bool` and **clears `out`** on failure — never a partial
batch, so a caller that ignores the return value cannot act on half an answer.
Refused: not-ready head, null `H` with `B > 0`, `B <= 0`, unknown label set, and
a `Tensor` whose shape is not `[B, E]`.

At the KERNEL level `B == 0` is a legitimate no-op rather than an error; at the
HEAD level it is a refusal. A head that returned `true` with zero results would
be indistinguishable from one that silently dropped rows.

### Files

| file | what |
|---|---|
| `include/omniseed/core/batch_gemm.h` | the kernel API + `BatchStats` (NEW) |
| `src/core/batch_gemm.cpp` | scalar kernel + runtime dispatch (NEW) |
| `src/core/batch_gemm_avx2.cpp` | AVX2 kernel, `/arch:AVX2` (NEW) |
| `tests/test_heads_batch.cpp` | **128 checks**, 0 fail, 1 skip, **UNGATED** (NEW) |
| `docs/BATCH.md` | full write-up (NEW) |
| `classification_head.{h,cpp}`, `decision_head.{h,cpp}`, `scoring_head.{h,cpp}` | the `*_batch` methods |
| `CMakeLists.txt` | two sources, the AVX2 flag, the test target |
| `docs/JEV_FEATURES.md` §6 | batch: NOT STARTED → **DONE** |

The one SKIP is `A6` (explicit Simd on a non-AVX2 machine) — this box *has*
AVX2, so that path is unreachable here and the test says so rather than passing
vacuously.

**Full local board: 39/39 passed (3747.40 s).** The board grew **38 → 39**; a
fresh clone goes **23 → 24**.

### Next

1. **Fit the vision/audio heads** (still the largest verified absence).
2. **Milestone 3 — dream consolidation** (still skipped).
3. **Phase 2.2 streaming decisions** — the head-level incremental API.
4. **Phase 2.3 feedback hooks** — the §34 store persists now; nothing closes the
   loop from a head's prediction to a recorded outcome.
5. **Point the backtest at the batched path** — `src/trading/simulate.cpp` still
   re-derives its readouts bar by bar. The capability and its evidence exist; the
   consumer does not.
6. **Fit a head ensemble** so §35's `from_ensemble()` has something to eat
   (Phase 2.5).
