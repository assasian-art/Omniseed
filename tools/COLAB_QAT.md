# OmniSeed QAT on Google Colab — exact recipe (GPU)

Turn the RWKV-7-World-0.1B bf16 checkpoint's big linears into **TRUE ternary
{-1,0,+1} weights** (BitNet b1.58, per-row absmean scales) with a straight-
through estimator. This is an OFFLINE training tool only — nothing here ships
at runtime; the C++ inference binary stays dependency-free.

A fresh `git clone` needs zero manual downloads:

| Asset | Source | Where it lands |
|---|---|---|
| `model.safetensors` (~380 MB) | auto: HF `Hakureirm/rwkv7-0.1b-hf` (public, no login) | `models/model.safetensors` |
| `config.json` | committed at `models/hf-orig/config.json` (re-downloaded if deleted) | `models/hf-orig/` |
| world vocab (65,536 tokens) | committed at `tools/data/rwkv_vocab_v20230424.txt` | resolved by `convert_to_omniseed.resolve_vocab()` (repo-relative first, `models/` second) |
| wikitext-103 corpus | auto-downloaded on first run, tokenized + cached | `models/corpus/*.npy` |

Device selection: `--device {auto,cpu,cuda}` — `auto` (default) picks CUDA
when torch sees it. CPU is bit-identical to pre-GPU behavior (proven by
`--verify-batch`); CUDA runs the same fp32 math, roughly 10-20x faster per
step on a T4.

---

## Cell 1 — clone the repo

```python
!git clone https://github.com/assasian-art/Omniseed.git
%cd /content/Omniseed
```

## Cell 2 — dependencies (torch is preinstalled on Colab with CUDA)

```python
!pip -q install pyarrow        # only missing piece: parquet corpus reader
import torch
print('cuda:', torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
```

## Cell 3 — mount Drive (checkpoints survive Colab disconnects)

```python
from google.colab import drive
drive.mount('/content/drive')
QAT_DIR = '/content/drive/MyDrive/omniseed-qat'
import os
os.makedirs(QAT_DIR, exist_ok=True)
```

## Cell 4 — restore the checkpoint from Drive (resume any session)

```python
%%bash
CKPT=/content/drive/MyDrive/omniseed-qat/qat_ckpt.pt
mkdir -p models
if [ -f "$CKPT" ]; then
  cp "$CKPT" models/qat_ckpt.pt
  cp "${CKPT%.pt}-best.pt" models/qat_ckpt-best.pt 2>/dev/null || true
  echo "checkpoint restored"
else
  echo "no Drive checkpoint — fresh start"
fi
```

## Cell 5 — sanity: verify the batched step is exact (seconds, no GPU needed)

```python
!python tools/qat_ternary.py --verify-batch --device cpu
# expect: [verify] max|dlogits|=0.00e+00 max|dstate|=0.00e+00 OK
```

## Cell 6 — optional 2-step smoke on the GPU before committing hours

```python
!python tools/qat_ternary.py --smoke --device cuda
# runs 2 real training steps, no checkpoint write / export; exit 0 = good
```

## Cell 7 — the chunk loop (each chunk ~55 min, checkpoint saved every chunk)

Exit codes from `qat_ternary.py`: `0` = target steps reached (done), `3` =
chunk time budget hit (run another chunk). The loop resumes automatically
from the checkpoint, so re-running this cell after a disconnect just works.

```python
%%bash
STEPS=20000            # total training steps
for i in $(seq 1 48); do
  python -u tools/qat_ternary.py --device cuda --steps $STEPS \
      --time-budget 3300 --ckpt models/qat_ckpt.pt
  rc=$?
  if [ $rc -eq 0 ]; then echo "TRAINING DONE"; break; fi
  if [ $rc -ne 3 ]; then echo "FAILED rc=$rc"; exit $rc; fi
  # persist to Drive after every chunk
  cp models/qat_ckpt.pt /content/drive/MyDrive/omniseed-qat/qat_ckpt.pt
  cp models/qat_ckpt-best.pt \
     /content/drive/MyDrive/omniseed-qat/qat_ckpt-best.pt 2>/dev/null || true
  echo "chunk $i done: $(tail -1 qat_log.txt)"
done
tail -5 qat_log.txt
```

Monitor progress any time with `tail -f qat_log.txt` — every eval logs
step / loss / lr / val ppl / best.

## Cell 8 — export the ternary GGUF (the final chunk does this automatically;
re-run to force)

```python
!python -u tools/qat_ternary.py --device cuda --steps 20000 \
    --ckpt models/qat_ckpt.pt --export-always
# writes models/rwkv7-0.1B-ternary-qat.gguf (~55-60 MB) from the BEST masters
```

## Cell 9 — take it home

```python
from google.colab import files
import shutil
shutil.copy('models/rwkv7-0.1B-ternary-qat.gguf',
            '/content/drive/MyDrive/omniseed-qat/rwkv7-0.1B-ternary-qat.gguf')
# and/or a direct browser download:
files.download('models/rwkv7-0.1B-ternary-qat.gguf')
```

Locally, drop the GGUF into `models/` and run
`build/bin/omniseed.exe gen --model models/rwkv7-0.1B-ternary-qat.gguf` —
the C++ runtime needs no changes; ternary-QAT tensors use the existing
`TERNARY` GGUF tensor type.

---

### Notes and knobs

- `--batch 8 --window 32` defaults fit any Colab GPU (T4 and up) with huge
  headroom; `--batch 16` on an A100 gives roughly 1.5-2x more tokens/sec.
- `--max-train-tokens` caps corpus size (default 3 M tokens, cached).
- The first chunk includes one-time downloads (checkpoint + wikitext parquet
  + tokenization of ~3 M tokens) — expect ~10-15 min before step 1.
- Colab free tier disconnects are fine: every finished chunk is on Drive
  (Cell 7); re-run Cells 1, 4 and 7 to resume exactly where it stopped.
- `tools/qat_chunk.py` / `qat_watch.sh` also work on Linux unchanged (they
  fall back to `sys.executable` when the Windows venv is absent).
