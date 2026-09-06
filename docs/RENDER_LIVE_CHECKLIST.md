# Render Live Checklist — OmniSeed on real infrastructure

Goal: a public URL where anyone can open the browser demo (`GET /`) and talk to
OmniSeed, with `/health` proving `model:true` and peak RSS under the 300 MB
budget. Total hands-on time: ~30 minutes.

## 0. What you deploy

| Piece | File | Notes |
|---|---|---|
| Blueprint | `render.yaml` | one web service, docker runtime, health check `/health` |
| Image | `Dockerfile` | multi-stage Alpine, GCC build, ~15 MB runtime image |
| Model | `models/rwkv7-0.1B-ternary.gguf` (~280 MB) | NOT in git — attached via disk or baked into a private image |
| Demo console | `src/server/main.cpp` → `GET /` | inline single-file HTML, no CDNs |
| API | `POST /gen`, `POST /ask`, `GET /health` | JSON in/out, serialized requests (by design) |

`/health` response contract:
```json
{"ok":true,"model":true,"peak_rss":155}
```
> `peak_rss` counts touched pages. The model is mmap'd, so the number climbs to
> ~155 MB only after the first `/gen` has run. Always fire one `/gen` before
> reading the RSS figure as a verdict.

## 1. Push the repo to GitHub

```bash
git remote add origin https://github.com/<you>/omniseed.git
git push -u origin main
```

Push-readiness (all verified in `.gitignore`):
- `build/`, `build-*/`, `out/` — build trees: ignored
- `models/*.gguf`, `*.safetensors`, `*.pt*`, `models/*.py`, `models/config.json` — model artifacts: ignored
- tracked fixtures kept on purpose: `models/rwkv_vocab_v20230424.txt`,
  `models/mel_filters.npz`, `models/hf-orig/*` (reference sources for the
  converters), plus the in-repo eval corpus used by `omniseed ppl`
- `state/`, `*.obj`, `*.pdb`, IDE noise: ignored

## 2. Import the blueprint on Render

1. Render dashboard → **New + → Blueprint** → pick your repo → Render reads
   `render.yaml` automatically.
2. The service builds the Dockerfile on their builder (adds ~2–4 min for the
   CMake build) and starts `omniseed_server --port 8080 --model /app/models/omniseed.gguf`.

## 3. Get the GGUF to `/app/models/omniseed.gguf` — pick ONE:

**Option A — Render disk (free tier supports disks only on paid plans):**
- Service → Disks → **Add disk**: mount path `/app/models`, size 1 GB.
- Then upload the GGUF once (Render shell or a one-off job):
  ```bash
  curl -O https://huggingface.co/Hakureirm/rwkv7-0.1b-hf/resolve/main/model.safetensors
  # (the converter + vocab are IN the repo; or simply upload the GGUF)
  ```

**Option B — Fork the Dockerfile to bake the model in (simplest, recommended):**

Create `Dockerfile.baked` (do NOT commit your GGUF to the public repo):
```dockerfile
FROM omniseed-base:latest          # or repeat the build stage from ./Dockerfile
FROM alpine:3.20
RUN apk add --no-cache libstdc++ && adduser -D -H omniseed
WORKDIR /app
COPY --from=builder /src/build/omniseed /app/omniseed
COPY --from=builder /src/build/omniseed_server /app/omniseed_server
COPY models/rwkv7-0.1B-ternary.gguf /app/models/omniseed.gguf
USER omniseed
EXPOSE 8080
VOLUME ["/app/state"]
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD wget -qO- http://127.0.0.1:8080/health || exit 1
ENTRYPOINT ["/app/omniseed_server"]
CMD ["--port", "8080", "--model", "/app/models/omniseed.gguf"]
```
Build & push privately (`docker build -f Dockerfile.baked ...`), then point the
Render service at your registry image. Image size ≈ 300 MB — fits free-tier
limits (currently up to 2 GB unpacked).

## 4. Verify live

```bash
BASE=https://<your-service>.onrender.com

# 1. liveness + budget
curl -s $BASE/health
# -> {"ok":true,"model":true,...}

# 2. wake the model (mmap pages + first token latency), then re-check RSS
curl -s -X POST $BASE/gen -H 'Content-Type: application/json' -d '{"prompt":"Hello"}'
curl -s $BASE/health
# -> {"ok":true,"model":true,"peak_rss":155}   # must be < 300

# 3. the browser demo
open $BASE/          # chat box wired to /gen; header shows model + RSS
```

Render's health check (`healthCheckPath: /health`) gates deploys; the Docker
`HEALTHCHECK` does the same locally.

## 5. Budget notes

- 0.1B i8 GGUF runtime peak: **155 MB** (this box, AVX2). Free tier gives 512 MB.
- Ternary-QAT GGUF (when TASK 2 ships): **~115 MB** peak — use it in the baked
  image for even more headroom.
- The server is single-request-at-a-time; a burst of browsers just queues.
- First token after a cold start is slow (model load + mmap fault-in) — expect
  10–20 s on free tier; subsequent `/gen` are ~19 tok/s.
