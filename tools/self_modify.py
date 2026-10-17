#!/usr/bin/env python3
"""OmniSeed — self_modify.py (T075b): diff-in / test-gated / exact rollback.

Harness, not agent: it never decides WHAT to patch, only whether a proposed
unified diff survives a named test gate. Fail-closed on every path.
Design: state/task_T075_brief.md. Stdlib only, offline.
"""
import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALLOW_DIRS = ("src/", "include/", "tests/", "tools/")
ALLOW_FILES = ("CMakeLists.txt",)
BACKUP_DIR = os.path.join(ROOT, "state", "patch_backup")
LOG_PATH = os.path.join(ROOT, "state", "self_modify_log.jsonl")
LAST_FAILED = os.path.join(ROOT, "build", "Testing", "Temporary", "LastTestsFailed.log")


def fail(msg, code):
    print("[self-modify] REFUSED: %s" % msg, file=sys.stderr)
    return code


def touched_paths(patch_text):
    """Paths a unified diff touches (b/ side of `diff --git` AND plain
    unified-diff `--- a/x` / `+++ b/x` headers — difflib emits only the
    latter). Deletions name the file only on the `---` side."""
    paths = set()
    for line in patch_text.splitlines():
        if line.startswith("diff --git "):
            tok = line.split()[-1]
            paths.add(tok[2:] if tok.startswith("b/") else tok)
        elif line.startswith("+++ b/"):
            paths.add(line[6:].strip())
        elif line.startswith("--- a/"):
            paths.add(line[6:].strip())
    return paths


def is_allowed(path):
    p = path.replace("\\", "/")
    if p in ALLOW_FILES:
        return True
    return any(p.startswith(d) for d in ALLOW_DIRS)


def tree_dirty():
    # Untracked files are NOT dirt: the pre-image only concerns TRACKED
    # content (a patch file sitting in the tree must not block itself).
    out = subprocess.run(["git", "status", "--porcelain", "-uno"], cwd=ROOT,
                         capture_output=True, text=True)
    return bool(out.stdout.strip())


def cmd_check():
    """Auto-discovery: failing suites from the last ctest run (read-only)."""
    if not os.path.exists(LAST_FAILED):
        print(json.dumps({"failing": [], "reason": "no-ctest-log"}))
        return 0
    failing = []
    with open(LAST_FAILED, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if ":" in line:
                num, name = line.split(":", 1)
                failing.append({"num": int(num), "suite": name})
    print(json.dumps({"failing": failing, "source": LAST_FAILED}))
    return 0


def _backup(paths):
    os.makedirs(BACKUP_DIR, exist_ok=True)
    manifest = {}
    for p in paths:
        src = os.path.join(ROOT, p)
        if os.path.isfile(src):
            digest = hashlib.sha1(p.encode()).hexdigest()[:12]
            dst = os.path.join(BACKUP_DIR, digest + "_" + os.path.basename(p))
            shutil.copy2(src, dst)
            manifest[p] = dst
    return manifest


def _restore(manifest, paths):
    """Byte-exact restore; files the patch CREATED are deleted."""
    for p, dst in manifest.items():
        shutil.copy2(dst, os.path.join(ROOT, p))
    for p in paths:
        if p not in manifest and os.path.isfile(os.path.join(ROOT, p)):
            os.remove(os.path.join(ROOT, p))


def _log(record):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def cmd_patch(patch_path, suite, gate_cmd, timeout):
    if tree_dirty():
        return fail("working tree not clean (pre-image must be reproducible)", 3)
    try:
        with open(patch_path, "r", encoding="utf-8") as f:
            patch = f.read()
    except OSError as e:
        return fail("cannot read patch: %s" % e, 4)
    paths = touched_paths(patch)
    bad = [p for p in sorted(paths) if not is_allowed(p)]
    if not paths:
        return fail("patch touches no files", 4)
    if bad:
        return fail("path(s) outside allowlist: %s" % ", ".join(bad), 4)
    chk = subprocess.run(["git", "apply", "--check", patch_path], cwd=ROOT,
                         capture_output=True, text=True)
    if chk.returncode != 0:
        return fail("git apply --check: %s" % (chk.stderr or chk.stdout).strip(), 5)

    manifest = _backup(paths)
    applied = subprocess.run(["git", "apply", patch_path], cwd=ROOT,
                             capture_output=True, text=True)
    if applied.returncode != 0:
        return fail("git apply failed: %s" % applied.stderr.strip(), 5)

    if gate_cmd:
        # Explicit override: split into argv and exec directly, NEVER via shell.
        cmd = shlex.split(gate_cmd)
    else:
        if not os.path.isdir(os.path.join(ROOT, "build")):
            return fail("no build/ directory — cannot gate", 7)
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", suite or ""):
            return fail("suite name must match [A-Za-z0-9_.-]+", 2)
        cmd = ["ctest", "--test-dir", "build", "-C", "Release", "-R", suite]
    try:
        gate = subprocess.run(cmd, cwd=ROOT, timeout=timeout,
                              capture_output=True, text=True)
        gate_rc, gate_out = gate.returncode, (gate.stdout + gate.stderr)[-2000:]
    except subprocess.TimeoutExpired:
        gate_rc, gate_out = -1, "gate timeout %ss" % timeout

    if gate_rc == 0:
        _log({"ts": time.time(), "patch": os.path.abspath(patch_path),
              "paths": sorted(paths), "gate": cmd, "result": "kept"})
        print("[self-modify] KEPT: gate passed (%s)" % cmd)
        return 0

    _restore(manifest, paths)
    subprocess.run(["git", "checkout", "--"] + sorted(paths), cwd=ROOT,
                   capture_output=True)
    _log({"ts": time.time(), "patch": os.path.abspath(patch_path),
          "paths": sorted(paths), "gate": cmd, "result": "rolled-back",
          "gate_rc": gate_rc, "tail": gate_out[-500:]})
    print("[self-modify] ROLLED BACK: gate rc=%d (files restored)" % gate_rc)
    return 6


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="list failing suites from the last ctest run")
    ap.add_argument("--patch", default="", help="unified diff to apply")
    ap.add_argument("--suite", default="", help="ctest -R name for the gate")
    ap.add_argument("--gate-cmd", default="",
                    help="override the gate command (tests/sandbox use this)")
    ap.add_argument("--timeout", type=int, default=600)
    a = ap.parse_args(argv)
    if a.check:
        return cmd_check()
    if not a.patch:
        return fail("--patch FILE required (or --check)", 2)
    if not a.suite and not a.gate_cmd:
        return fail("--suite NAME (or --gate-cmd) required", 2)
    return cmd_patch(a.patch, a.suite, a.gate_cmd, a.timeout)


if __name__ == "__main__":
    sys.exit(main())
