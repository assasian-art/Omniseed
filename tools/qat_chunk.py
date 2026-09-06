#!/usr/bin/env python3
"""QAT chunk launcher (called by tools/qat_watch.* in a loop).

Keeps the loop logic in Python so the Windows .bat and Unix .sh watchers share
one code path. Exit codes from qat_ternary.py: 0 = target steps reached,
3 = chunk ended on its time budget (run another chunk), anything else = failure.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = os.path.join(ROOT, '.venv', 'Scripts', 'python.exe')
if not os.path.exists(PY):
    PY = sys.executable


def main():
    chunk_seconds = sys.argv[1] if len(sys.argv) > 1 else '3000'
    cmd = [PY, '-u', os.path.join(HERE, 'qat_ternary.py'),
           '--time-budget', chunk_seconds]
    print(f'[watch] chunk: {cmd}', flush=True)
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == '__main__':
    sys.exit(main())
