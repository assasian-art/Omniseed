@echo off
rem ===========================================================================
rem  OmniSeed — qat_watch.bat: resumable background QAT trainer (Windows).
rem  Loops tools/qat_chunk.py until qat_ternary.py exits 0 (target steps done).
rem  Each chunk resumes models/qat_ckpt.pt, so this survives reboots / Ctrl-C.
rem  Usage:  tools\qat_watch.bat [chunk_seconds]    (default 3000 s = 50 min)
rem  Progress: qat_log.txt (repo root). Stop anytime with Ctrl-C.
rem  Background:  start /b tools\qat_watch.bat > qat_watch.out 2>&1
rem ===========================================================================
setlocal
:loop
python -u tools\qat_chunk.py %1
if errorlevel 3 (
    echo [watch] chunk hit time budget, continuing...
    goto loop
)
if errorlevel 1 (
    echo [watch] training failed — see qat_log.txt
    exit /b 1
)
echo [watch] training complete — see qat_log.txt and models\rwkv7-0.1B-ternary-qat.gguf
endlocal
