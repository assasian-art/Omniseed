# Modal GPU training guide (QAT round-3 + assistant LoRA)

Modal runs the **same local, battle-tested tools** (`tools/qat_ternary.py`,
`tools/lora_chat.py`) on a cloud GPU. Nothing is re-implemented remotely: the
Modal functions clone the public repo into a Volume and run the tools as
subprocesses, so every flag, checkpoint format, and export path behaves
exactly as it does locally — just ~50× faster.

## One-time setup

```bash
pip install modal
modal token new
```

Modal bills per-second on the GPU you pick (`T4` default in
`tools/modal_qat.py`; edit `gpu=` there for `L4`/`A10G`).

## Layout

- **`tools/modal_qat.py`** — the Modal app: `train_chunk` (chunked ternary
  QAT), `train_lora` (assistant sidecar), `export_best` (artifact report).
  State lives in Volume `omniseed-models`:
  - `/vol/models/qat_ckpt.pt` — resume-safe QAT checkpoint (masters + AdamW
    moments + global step + best)
  - `/vol/models/qat_log.txt` — training log (grad norms, val PPL trajectory)
  - `/vol/models/rwkv7-0.1B-ternary-qat.gguf` — exported when the step target
    is hit
  - `/vol/models/assistant-lora.gguf` — LoRA sidecar export
- **`tools/modal_train.py`** — local orchestrator (chunk loop, status).

## Round-3 clean QAT (the calm recipe)

Round 2b's diagnosis stands: fresh PTQ init, **no resume** (breaks the
inherited-memorization chain), lr 1e-4, KD on, export only at val PPL ≤ 60.

```bash
python tools/modal_train.py qat --fresh          # ~6 T4 chunks of 55 min
python tools/modal_train.py status               # artifacts + log tail
```

Each chunk runs `qat_ternary.py --time-budget 3300`; exit 3 (budget spent)
just means "launch the next chunk" — the orchestrator does that
automatically, up to `--chunks`. Re-running `qat` **without** `--fresh`
continues from the Volume checkpoint.

Watch val PPL in the log tail. **Graduation gate: val PPL ≤ 60** (≤1.25× of
the bf16 baseline's 1.25×-target). When the export lands, bring it home:

```bash
modal volume get omniseed-models /models/rwkv7-0.1B-ternary-qat.gguf models/
```

Then verify in the C++ runtime:

```bash
./build/bin/omniseed gen --model models/rwkv7-0.1B-ternary-qat.gguf \
    --repeat-penalty 1.2
ctest -C Release -R omniseed_qat_ternary
```

## Assistant-behavior LoRA (minutes on GPU)

```bash
python tools/modal_train.py lora --steps 2000
modal volume get omniseed-models /models/assistant-lora.gguf models/
./build/bin/omniseed chat --model models/rwkv7-0.1B-ternary.gguf \
    --assistant-lora models/assistant-lora.gguf
```

The sidecar then also plugs into the server: `--assistant-lora P` at boot or
`"assistant_lora": "path.gguf"` per request on `/gen` and `/ask`.

### Retrain recipe + health checks (after the degenerate-run fix)

`tools/lora_chat.py` now has degenerate-run guards (holdout split, early stop,
best-snapshot export, zero-delta fallback). A healthy run:

```bash
# CPU overnight (~3000 steps) — or the Modal GPU line above (minutes)
./.venv/Scripts/python.exe tools/lora_chat.py --steps 3000 --eval-every 50 \
    --patience 400
# probe a reply without retraining
./.venv/Scripts/python.exe tools/lora_chat.py --sample "What is 2+2?"
# attach + eyeball three fixed prompts
./build/bin/omniseed gen --model models/rwkv7-0.1B-ternary-qat.gguf \
    --prompt "User: What is 2+2?

Assistant:" --assistant-lora models/assistant-lora.gguf
```

Healthy holdout answer-ppl is **~1.05–3.0**. Exactly **1.00 is the
memorization signature, not success**. The trainer prints `WARNING` when the
exported `|B|` drifts past 0.25 (the degenerate run hit 0.30). The export is
always the **best holdout snapshot**; if holdout never improved it exports a
zero-delta sidecar (attaching it is a proven no-op). Sidecars carry
`lora.format_version` (=2).

## Cost sanity

T4 ≈ $0.59/h. A full 12k-step round-3 run at B=32/W=32 historically fits in
5–8 T4-hours (~$3–5). The LoRA pass is minutes (cents).
