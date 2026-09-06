#!/usr/bin/env bash
# ===========================================================================
#  OmniSeed — qat_watch.sh: resumable background QAT trainer (Linux/macOS).
#  Loops tools/qat_chunk.py until qat_ternary.py exits 0 (target steps done).
#  Each chunk resumes models/qat_ckpt.pt, so this survives reboots / Ctrl-C.
#  Usage:  ./tools/qat_watch.sh [chunk_seconds]   (default 3000 s = 50 min)
#  Progress: qat_log.txt (repo root). Stop anytime with Ctrl-C.
#  Background:  nohup ./tools/qat_watch.sh > qat_watch.out 2>&1 &
# ===========================================================================
set -u
while true; do
    python -u tools/qat_chunk.py "${1:-3000}"
    rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "[watch] training complete — see qat_log.txt"
        exit 0
    elif [ "$rc" -eq 3 ]; then
        echo "[watch] chunk hit time budget, continuing..."
    else
        echo "[watch] training failed (exit $rc) — see qat_log.txt" >&2
        exit "$rc"
    fi
done
