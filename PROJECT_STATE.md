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
- **Test:** `build\bin\omniseed_tests.exe` — **163/163 passing** (zero warnings /W4)
- **Last updated:** Full spec audit complete; all 7 novel capabilities implemented;
  master spec generated (`docs/OMNISEED_MASTER_SPEC.md` + `.html`)

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

## 3. NEXT STEPS (in order)

1. **Weights converter** (`tools/convert_to_omniseed.py` — offline, NOT runtime):
   RWKV-7 World `.pth` + tokenizer → ternary GGUF with the exact tensor names
   `src/core/rwkv.cpp` expects (`blocks.N.att.{receptance,key,value,output,w1,w2,
   a1,a2,g1,g2,v1,v2}.weight` + `.scale`, `blocks.N.att.{tmix_*,w_bias,a_bias,
   v_bias,k_k,k_a,r_k,gn.*}`, `blocks.N.ffn.{tmix_v,key,value}`, `blocks.0.ln0.*`,
   `ln_out.*`, `head.weight`+`head.scale`, `token.embd` fp16) and, later,
   `vision.*`/`whisper.*`/`focal.*` tensors to retire the documented fallbacks.
2. **Validate RWKV-7 forward numerically** against a real trained checkpoint
   (math is reference-transcribed and line-checked, but no weights exist yet).
3. **Optional polish:** temperature/top-k sampling in `RwkvModel`/`AgentLoop`,
   FFT swap for the mel DFT, AVX2 paths in `bitlinear_forward`.
4. **CI on Linux** (GCC/MinGW paths untested on this machine).
5. **WebRTC (#19)** upgrade path: replace/augment `UdpBeacon` with libdatachannel
   if the budget allows; DESIGN-status features #29/#30/#81–100 graduate when a
   training pipeline lands.

## 4. UNRESOLVED ISSUES / BUGS

- **No real model weights exist.** Runtime is complete; model-dependent commands
  fail gracefully with a load error until the converter (NEXT STEPS #1) runs.
- **Vision/audio backbones run documented fallback paths** (deterministic patch
  statistics / spectral-hash codes) until converter tensors exist.
- **`AgentLoop::generate` is greedy only**; sampling config structure exists.
- **Naive DFT in mel front-end is O(N²)** (~1s per 30s window on desktop).
- **Server is serialized** (one request at a time) — fine for edge use.
- **MinGW/GCC builds untested** on this machine; MSVC `/W4` is clean.
- **Windows console UTF-8**: consider `SetConsoleOutputCP(CP_UTF8)` in CLI main.
- Persistence paths route under `./state/` (created on demand, verified).

## 5. VERIFICATION STATUS

- `build\bin\omniseed_tests.exe` → **163/163 PASS**. Coverage: platform (mmap,
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
- Fresh-clone build verified with wiped `build/`; zero compiler warnings.
