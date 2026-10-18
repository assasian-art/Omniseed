#!/usr/bin/env python3
"""
Omniseed autonomous runner.

Loop forever:
  1. read state/QUEUE.md -> first unchecked micro-task
  2. execute via `opencode run` (primary model, fallback model on model errors)
  3. gate: cmake build + full ctest board must be green
  4. PASS  -> git add -A (secret-scan staged) -> commit "Txxx: desc"
              -> git push --force-with-lease origin main
              -> check off queue line + append state/PROGRESS.md
  5. FAIL  -> retry with error tail fed back (max 3 attempts)
              -> then append state/blocked.md, mark queue line [!] , move on
  6. opencode may answer "VERDICT: BLOCKED" -> straight to blocked.md, no retry
  7. sleep 5s, repeat. Empty queue -> exit 0.

Logs  : state/runner.log
Status: state/runner_status.json
Owned files (opencode must not edit): state/QUEUE.md, state/PROGRESS.md, state/blocked.md, runner.py
"""
import datetime
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATE = ROOT / "state"
QUEUE = STATE / "QUEUE.md"
PROGRESS = STATE / "PROGRESS.md"
BLOCKED = STATE / "blocked.md"
LOG = STATE / "runner.log"
STATUS = STATE / "runner_status.json"

SLEEP_S = 5
MAX_ATTEMPTS = 3
OP_TIMEOUT_S = 2700   # opencode run per attempt
BUILD_TIMEOUT_S = 2700
TEST_TIMEOUT_S = 2400
PUSH_EVERY_N_TASKS = 1  # push after every passed task

PRIMARY_MODEL = "opencode/nemotron-3.5-lightning-free"
FALLBACK_MODEL = "opencode/mimo-v2.6-flash-free"

TASK_RE = re.compile(r"^- \[ \] (T\d+[a-z]?)\s*[:\-]?\s*(.*)$")
SECRET_HINTS = (".env", ".pem", "id_rsa", "id_ed25519", ".key")


def log(msg: str) -> None:
    line = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def set_status(**kw) -> None:
    data = {}
    if STATUS.exists():
        try:
            data = json.loads(STATUS.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data.update(kw)
    data["updated"] = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}"
    data["pid"] = __import__("os").getpid()
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(STATUS)


def sh(args, timeout=None, cwd=None):
    """Run command without shell. Returns (rc, combined_output)."""
    try:
        p = subprocess.run(
            args, cwd=cwd or ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired as e:
        out = ((e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or ""))
        return 124, out + f"\n[TIMEOUT after {timeout}s]"
    except FileNotFoundError as e:
        return 127, str(e)


def find_opencode():
    for name in ("opencode.cmd", "opencode.exe", "opencode.bat", "opencode"):
        p = shutil.which(name)
        if p:
            return p
    return None


# ---------------------------------------------------------------- queue ----

def read_queue():
    """Return list of (lineno, task_id, full_line) for unchecked tasks."""
    tasks = []
    for i, line in enumerate(QUEUE.read_text(encoding="utf-8").splitlines()):
        m = TASK_RE.match(line)
        if m:
            tasks.append((i, m.group(1), line))
    return tasks


def check_off_queue(task_id: str) -> None:
    lines = QUEUE.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if line.startswith(f"- [ ] {task_id}"):
            lines[i] = line.replace("- [ ] ", "- [x] ", 1)
            QUEUE.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return


def mark_queue_blocked(task_id: str) -> None:
    lines = QUEUE.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if line.startswith(f"- [ ] {task_id}"):
            lines[i] = line.replace("- [ ] ", "- [!] ", 1) + "  (BLOCKED -> state/blocked.md)"
            QUEUE.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return


def append_progress(task_id: str, note: str) -> None:
    with PROGRESS.open("a", encoding="utf-8") as f:
        f.write(f"{task_id} | {datetime.date.today():%Y-%m-%d} | {note}\n")


def append_blocked(task_id: str, desc: str, detail: str) -> None:
    with BLOCKED.open("a", encoding="utf-8") as f:
        f.write(f"\n## {task_id} {desc} — runner-blocked {datetime.date.today():%Y-%m-%d}\n\n")
        f.write("3 attempts failed build/test gate (or opencode answered VERDICT: BLOCKED).\n\n")
        f.write("```\n" + detail[-2500:] + "\n```\n")


# ---------------------------------------------------------------- build ----

def build_and_test():
    """Full gate. Returns (ok, tail_output)."""
    out_all = []
    if not (ROOT / "build").exists():
        log("build/ missing -> full build.bat")
        rc, out = sh(["cmd", "/c", str(ROOT / "build.bat")], timeout=BUILD_TIMEOUT_S)
        out_all.append(out)
        if rc != 0:
            return False, "\n".join(out_all)[-4000:]
        return True, "\n".join(out_all)[-4000:]
    rc, out = sh(["cmake", "--build", "build", "--config", "Release", "--parallel"],
                 timeout=BUILD_TIMEOUT_S)
    out_all.append(out)
    if rc != 0:
        return False, "\n".join(out_all)[-4000:]
    rc, out = sh(["ctest", "--test-dir", "build", "-C", "Release", "--output-on-failure"],
                 timeout=TEST_TIMEOUT_S)
    out_all.append(out)
    return rc == 0, "\n".join(out_all)[-4000:]


# -------------------------------------------------------------- opencode ----

def sanitize(text: str) -> str:
    return (text.replace("%", " pct ").replace("^", " ")
                .replace('"', "'").replace("\r", " ").replace("\n", " "))


def build_prompt(task_id: str, desc: str, error_tail: str = "") -> str:
    p = (
        f"TASK {task_id}: {desc}. "
        "You are working in the OmniSeed repo (pure C++17, CMake, MSVC). "
        "FIRST read PROJECT_STATE.md NEXT-STEPS section and state/blocked.md for this task's context; "
        "if the task text references a design/brief file (state/*.md), read that file first — it is the authoritative design; "
        "PROJECT_STATE.md is 300KB+ — grep it for the relevant section, never read it whole; "
        "check whether the capability already exists ([VERIFY] rule). "
        "Rules: one micro-task, <=40 lines of meaningful change; add or extend unit tests; "
        "build with `cmake --build build --config Release` and run `ctest --test-dir build -C Release` "
        "until the full board is green with no new failures; "
        "update PROJECT_STATE.md next-steps; "
        "DO NOT edit state/QUEUE.md, state/PROGRESS.md, state/blocked.md or runner.py (the runner owns them); "
        "never write secrets/credentials. "
        "If the task is impossible right now (missing data, unreachable network, missing dependency), "
        "change nothing and end your final message with the exact line: VERDICT: BLOCKED. "
        "Otherwise end your final message with the exact line: VERDICT: DONE."
    )
    if error_tail:
        p += " PREVIOUS ATTEMPT FAILED THE BUILD/TEST GATE. Error tail: " + sanitize(error_tail[-1500:])
    return sanitize(p)


def run_opencode(opencode_exe, prompt, model):
    return sh(
        ["cmd", "/c", opencode_exe, "run", prompt, "--model", model,
         "--dangerously-skip-permissions"],
        timeout=OP_TIMEOUT_S,
    )


# ------------------------------------------------------------------ git ----

def git(*args, timeout=300):
    return sh(["git", *args], timeout=timeout)


def secret_in_staged(paths):
    for p in paths:
        low = p.lower()
        if any(h in low for h in SECRET_HINTS):
            return p
    return None


def commit_and_push(task_id: str, desc: str):
    """Returns True if committed (push failure does not un-commit)."""
    git("add", "-A")
    rc, staged = git("diff", "--cached", "--name-only")
    files = [l for l in staged.splitlines() if l.strip()]
    bad = secret_in_staged(files)
    if bad:
        git("reset", "-q", "HEAD", "--", bad)
        log(f"SECRET-GUARD: unstaged suspicious file {bad}")
    files = [f for f in files if f != bad]
    if not files:
        log("no staged changes after task (no-diff pass)")
        return True
    msg = f"{task_id}: {desc}"[:120].replace('"', "'")
    rc, out = git("commit", "-m", msg)
    if rc != 0:
        log(f"COMMIT FAILED: {out[-500:]}")
        return False
    log(f"committed: {msg} ({len(files)} files)")
    rc, out = git("fetch", "origin", "main")
    rc2, out2 = git("push", "--force-with-lease", "origin", "main")
    if rc2 != 0:
        log(f"PUSH FAILED (kept local): {out2[-400:]}")
        set_status(pending_push=True)
    else:
        log("pushed origin/main (--force-with-lease)")
        set_status(pending_push=False)
    return True


def retry_pending_push():
    if not STATUS.exists():
        return
    try:
        if not json.loads(STATUS.read_text(encoding="utf-8")).get("pending_push"):
            return
    except Exception:
        return
    log("retrying pending push")
    git("fetch", "origin", "main")
    rc, out = git("push", "--force-with-lease", "origin", "main")
    if rc == 0:
        log("pending push OK")
        set_status(pending_push=False)
    else:
        log(f"pending push still failing: {out[-300:]}")


# ----------------------------------------------------------------- main ----

def handle_task(opencode_exe, task_id, desc):
    """Returns 'passed' or 'blocked'."""
    last_tail = ""
    model_used = PRIMARY_MODEL
    for attempt in range(1, MAX_ATTEMPTS + 1):
        log(f"{task_id} attempt {attempt}/{MAX_ATTEMPTS} model={model_used}")
        set_status(task=task_id, attempt=attempt, phase="opencode", model=model_used)
        prompt = build_prompt(task_id, desc, last_tail)
        rc, out = run_opencode(opencode_exe, prompt, model_used)
        tail = out[-4000:]
        log(f"{task_id} opencode rc={rc} out-tail:\n{tail[-1200:]}")
        low = out.lower()
        if rc != 0 and ("model" in low or "provider" in low or "not found" in low) and model_used == PRIMARY_MODEL:
            log(f"{task_id}: primary model rejected, switching to {FALLBACK_MODEL}")
            model_used = FALLBACK_MODEL
            rc, out = run_opencode(opencode_exe, prompt, model_used)
            tail = out[-4000:]
            log(f"{task_id} opencode(fallback) rc={rc}")
        if "verdict: blocked" in out.lower():
            log(f"{task_id}: opencode says VERDICT: BLOCKED")
            append_blocked(task_id, desc, tail)
            mark_queue_blocked(task_id)
            append_progress(task_id, f"BLOCKED per opencode verdict; see state/blocked.md")
            return "blocked"
        log(f"{task_id}: build+test gate")
        set_status(task=task_id, phase="build/test")
        ok, gate_out = build_and_test()
        if ok:
            log(f"{task_id}: GATE PASS -> commit")
            set_status(task=task_id, phase="commit/push")
            if commit_and_push(task_id, desc):
                check_off_queue(task_id)
                append_progress(task_id, f"PASS via runner (model={model_used}); gate green, committed+pushed")
                return "passed"
            last_tail = "git commit failed: " + gate_out[-800:]
        else:
            last_tail = gate_out
            log(f"{task_id}: GATE FAIL attempt {attempt}\n{gate_out[-1200:]}")
        time.sleep(2)
    log(f"{task_id}: {MAX_ATTEMPTS} attempts failed -> blocked.md")
    append_blocked(task_id, desc, last_tail)
    mark_queue_blocked(task_id)
    append_progress(task_id, f"BLOCKED after {MAX_ATTEMPTS} attempts; see state/blocked.md")
    return "blocked"


def main():
    log("=" * 60)
    log("runner start (models: " + PRIMARY_MODEL + " -> " + FALLBACK_MODEL + ")")
    opencode_exe = find_opencode()
    if not opencode_exe:
        log("FATAL: opencode CLI not found on PATH")
        return 1
    set_status(phase="running", task=None)
    tasks_done = 0
    while True:
        try:
            retry_pending_push()
            tasks = read_queue()
            if not tasks:
                log("QUEUE EMPTY — nothing left. Exiting.")
                set_status(phase="queue_empty")
                return 0
            lineno, task_id, line = tasks[0]
            desc = TASK_RE.match(line).group(2).strip() or "(no description)"
            log(f"next task: {task_id} — {desc}")
            result = handle_task(opencode_exe, task_id, desc)
            tasks_done += 1
            log(f"task {task_id} -> {result} (total handled: {tasks_done})")
        except KeyboardInterrupt:
            log("interrupted by user")
            set_status(phase="interrupted")
            return 130
        except Exception as e:
            log(f"LOOP ERROR (continuing): {type(e).__name__}: {e}")
        time.sleep(SLEEP_S)


if __name__ == "__main__":
    sys.exit(main())
