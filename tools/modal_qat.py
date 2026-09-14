#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — modal_qat.py   (Modal remote-GPU training functions)
#
#  Modal app backing tools/modal_train.py. The heavy tools stay LOCAL and
#  battle-tested: each Modal function clones the public repo into a Volume
#  (cached across chunks), then runs tools/qat_ternary.py (chunked ternary
#  QAT, resume-safe) or tools/lora_chat.py (assistant-behavior LoRA) as a
#  subprocess on a GPU, persisting checkpoints/exports in the Volume.
#
#  Volume layout (omniseed-models):
#    /vol/repo     shallow clone of the repo (git pull per chunk)
#    /vol/models   qat_ckpt.pt, qat_log.txt, exported GGUFs, lora ckpt/sidecar
#
#  Chunk semantics (inherited from qat_ternary.py): exit 0 = target reached
#  (GGUF exported), exit 3 = budget elapsed (resume from ckpt next chunk).
#
#  Requires: pip install modal; modal token new. See docs/MODAL_QAT_GUIDE.md.
# =============================================================================
import os
import subprocess
import sys

import modal

MINUTES = 60
REPO_URL = "https://github.com/assasian-art/Omniseed"
REPO_DIR = "/vol/repo"
MODELS_DIR = "/vol/models"

volume = modal.Volume.from_name("omniseed-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch~=2.5",
        "numpy",
        "safetensors",
        "tokenizers",
    )
)

app = modal.App("omniseed-train", image=image)


def _ensure_repo() -> str:
    """Clone (first chunk) or fast-forward the repo inside the Volume."""
    marker = os.path.join(REPO_DIR, "tools", "qat_ternary.py")
    if not os.path.exists(marker):
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, REPO_DIR],
            check=True,
        )
    else:
        subprocess.run(
            ["git", "-C", REPO_DIR, "pull", "--ff-only"],
            check=False,           # offline/detached clone is fine to reuse
        )
    return REPO_DIR


def _run(cmd: list, cwd: str, tail_lines: int = 30) -> dict:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    return {
        "exit": proc.returncode,
        "tail": "\n".join(out.splitlines()[-tail_lines:]),
    }


@app.function(image=image, gpu="T4", timeout=75 * MINUTES,
               volumes={"/vol": volume})
def train_chunk(
    steps: int = 4000,
    time_budget: int = 3300,
    lr: float = 1e-4,
    kd_weight: float = 1.0,
    window: int = 32,
    batch: int = 32,
    eval_every: int = 50,
    fresh: bool = False,
) -> dict:
    """One chunked QAT segment (round-3 calm recipe by default: fresh init,
    wikitext+tinystories, lr 1e-4, KD on). Resumes /vol/models/qat_ckpt.pt
    unless fresh=True deletes it first. Export lands in the Volume."""
    _ensure_repo()
    os.makedirs(MODELS_DIR, exist_ok=True)
    ckpt = os.path.join(MODELS_DIR, "qat_ckpt.pt")
    if fresh and os.path.exists(ckpt):
        os.remove(ckpt)
    cmd = [
        sys.executable, "-u", "tools/qat_ternary.py",
        "--corpus", "wikitext+tinystories",
        "--steps", str(steps),
        "--lr", str(lr),
        "--kd-weight", str(kd_weight),
        "--window", str(window),
        "--batch", str(batch),
        "--eval-every", str(eval_every),
        "--time-budget", str(time_budget),
        "--device", "cuda",
        "--ckpt", ckpt,
        "--out", os.path.join(MODELS_DIR, "rwkv7-0.1B-ternary-qat.gguf"),
        "--log-file", os.path.join(MODELS_DIR, "qat_log.txt"),
    ]
    res = _run(cmd, REPO_DIR)
    res["done"] = res["exit"] == 0
    res["ckpt"] = ckpt
    volume.commit()
    return res


@app.function(image=image, timeout=10 * MINUTES, volumes={"/vol": volume})
def export_best() -> dict:
    """Report + validate the artifacts currently in the Volume (sizes in MB,
    log tail). Run after training to confirm what to download home."""
    _ensure_repo()
    files = {}
    if os.path.isdir(MODELS_DIR):
        for name in sorted(os.listdir(MODELS_DIR)):
            path = os.path.join(MODELS_DIR, name)
            if os.path.isfile(path):
                files[name] = round(os.path.getsize(path) / 1048576, 1)
    log = os.path.join(MODELS_DIR, "qat_log.txt")
    tail = ""
    if os.path.exists(log):
        with open(log, "r", encoding="utf-8", errors="replace") as f:
            tail = "\n".join(f.read().splitlines()[-15:])
    volume.commit()
    return {"files_mb": files, "qat_log_tail": tail}


@app.function(image=image, gpu="T4", timeout=75 * MINUTES,
               volumes={"/vol": volume})
def train_lora(
    steps: int = 2000,
    batch: int = 16,
    lr: float = 1e-3,
    rank: int = 8,
    alpha: float = 16.0,
    corpus_file: str = "corpus_eval.txt",
) -> dict:
    """Assistant-behavior LoRA pass (tools/lora_chat.py) on a GPU — minutes
    instead of the ~hours a CPU needs. Sidecar exports to the Volume."""
    _ensure_repo()
    os.makedirs(MODELS_DIR, exist_ok=True)
    cmd = [
        sys.executable, "-u", "tools/lora_chat.py",
        "--steps", str(steps),
        "--batch", str(batch),
        "--lr", str(lr),
        "--rank", str(rank),
        "--alpha", str(alpha),
        "--corpus-file", corpus_file,
        "--device", "cuda",
        "--ckpt", os.path.join(MODELS_DIR, "lora_chat.pt"),
        "--out", os.path.join(MODELS_DIR, "assistant-lora.gguf"),
    ]
    # Train on the SAME ternary base the C++ runtime serves — a sidecar
    # fitted on the PTQ base is off-distribution at serve time.
    gguf_base = os.path.join(MODELS_DIR, "rwkv7-0.1B-ternary-qat.gguf")
    if os.path.isfile(gguf_base):
        cmd += ["--gguf-base", gguf_base]
    else:
        print("[train_lora] WARNING: no %s on the Volume — falling back to "
              "the PTQ safetensors base. Run train_chunk + export_best first; "
              "a PTQ-fitted sidecar may misbehave when attached to the "
              "served QAT GGUF." % gguf_base)
    res = _run(cmd, REPO_DIR)
    res["done"] = res["exit"] == 0
    volume.commit()
    return res
