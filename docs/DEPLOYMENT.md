# OmniSeed Deployment Guide

Everything in this repository compiles to **pure C++17** with **zero external
dependencies**. There is no Python at runtime. The same code builds on
Windows (MSVC / MinGW), Linux (GCC / Clang), macOS, and inside Docker.

---

## 1. Build from source

### Windows (Visual Studio 2022/18)

```bat
build.bat
:: or manually:
cmake -S . -B build -G "Visual Studio 18 2026" -A x64
cmake --build build --config Release
```

Binaries land in `build\bin\`:

| Binary | Purpose |
|---|---|
| `omniseed.exe` | CLI: info/chat/gen/ask/bench/tools/selftest |
| `omniseed_server.exe` | Optional HTTP API (requires `-DOMNISEED_BUILD_SERVER=ON`) |
| `omniseed_tests.exe` | Test suite (163 checks) |

### Linux / macOS / MinGW

```bash
./build.sh
# or:
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DOMNISEED_BUILD_SERVER=ON
cmake --build build -j
ctest --test-dir build
```

### Verify the build

```bash
./build/bin/omniseed_tests     # 163/163 must pass
./build/bin/omniseed selftest  # numeric sanity, prints "selftest OK"
./build/bin/omniseed info      # shows peak RSS (should be ~4 MB idle)
```

---

## 2. Local run

### Model

The runtime loads an RWKV-7 + BitNet GGUF from `./models/omniseed.gguf`
(metadata keys `omniseed.*`, ternary weights at 2 weights/byte — see
`tools/make_tiny_gguf.py` for the exact layout). Without a model, the CLI
still runs `info`, `tools`, `selftest`, and the server serves `/health`.

```bash
# generate a tiny test model (validates the loader end-to-end)
python tools/make_tiny_gguf.py          # writes build/tiny.gguf
./build/bin/omniseed bench --model build/tiny.gguf
```

### CLI commands

```bash
omniseed info                      # platform + RSS info
omniseed tools                     # builtin tool schemas (JSON)
omniseed selftest                  # numeric self-check
omniseed gen    --model M --prompt "hello"
omniseed ask    --model M "what is 12*(3+4)?"   # one agent turn with tools
omniseed chat   --model M                        # REPL
omniseed bench  --model M                        # tok/s + peak RSS
```

Options: `--model PATH`, `--prompt TEXT`, `--max-tokens N`, `--quiet`.

### HTTP server

```bash
./build/bin/omniseed_server --port 8080 --model ./models/omniseed.gguf
```

| Endpoint | Method | Body | Result |
|---|---|---|---|
| `/health` | GET | — | `{"ok":true,"model":true,"assistant_lora":false,"peak_rss":42}` |
| `/ask` | POST | `{"task":"what is 2+2","repeat_penalty":1.2,"repeat_window":64}` | `{"reply":"..."}` |
| `/gen` | POST | `{"prompt":"hello","repeat_penalty":1.2,"repeat_window":64}` | `{"text":"..."}` |
| `/asr` | POST | raw 16 kHz mono 16-bit WAV bytes, or `{"wav_b64":"<base64>"}` | `{"text":"<|0.00|> ...<|4.00|>","seconds":4.3}`; `{"error":"..."}` on silence/noise/too-short clips |

`/asr` (Phase 15) runs the real whisper-tiny encoder + greedy decoder (HF
official-parity: suppress tokens + timestamp rules). The sidecar
(`models/whisper-tiny-encoder.gguf`) loads lazily on the first `/asr` call;
`/health` reports `"asr":"real"|"fallback"|false` after that. Transcription
includes whisper timestamp tokens; strip `<|...|>` spans for plain text.

Assistant-behavior LoRA: start the server with `--assistant-lora
models/assistant-lora.gguf` (or `OMNISEED_ASSISTANT_LORA`), or pass
`"assistant_lora": "path.gguf"` per request on `/gen`/`/ask` — absent keeps
the current attachment, `""` detaches. See the MASTER_SPEC server contract.

---

## 3. Docker

```bash
docker build -t omniseed .
docker run --rm -p 8080:8080 -v "$PWD/models:/app/models:ro" omniseed

# run the tests inside the image build (already part of the Dockerfile):
docker build --progress=plain -t omniseed . 2>&1 | grep -A3 ctest
```

The image:
* two-stage Alpine build (builder ~gcc, runtime ~`libstdc++` only)
* runs as non-root user `omniseed`
* `HEALTHCHECK` polls `/health` every 30 s
* `/app/state` volume persists memory crystals / skills / fingerprints

## 4. Render.com

`render.yaml` is a ready blueprint: Docker runtime, free plan, health check
on `/health`. Deploy with `render blueprint launch` or via the dashboard.
Mount a disk at `/app/models` (or fork the Dockerfile to `COPY` your GGUF in)
to enable `/ask` and `/gen`.

## 5. Memory budget validation

The `<300 MB` constraint is enforced, not assumed:

```bash
./build/bin/omniseed bench --model ./models/omniseed.gguf
# bench: 128 tokens in N ms -> X tok/s
# peak RSS: Y MB (budget 300 MB)
# -> exit code 2 if the budget is exceeded
```

`peak_rss` reads `PeakWorkingSetSize` (Windows) / `VmHWM` (Linux) — the
process high-water mark, not a sampling. The GGUF is memory-mapped read-only,
so RSS tracks only the pages actually touched by inference.

## 6. Runtime state layout

```
./state/
  skills.bin      FlashSkillPool persistence ('SKPL')
  memories.bin    (optional) memory crystal sidecar ('MCTR')
  kg.bin          knowledge graph ('OKGT')
  fingerprint.bin sensory profile ('SNFP')
```

All stores are little-endian binary with magic + version headers; delete the
directory to reset the agent's learned state.
