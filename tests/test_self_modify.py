#!/usr/bin/env python3
"""OmniSeed — tests/test_self_modify.py (T075c).

Sandboxed: every case runs against a TEMP git repo created here; the real
tree is never touched (module ROOT/BACKUP/LOG/LAST_FAILED are rebound).
Covers: keep-on-pass, byte-exact rollback on gate fail, deletion of files the
patch created, dirty-tree refusal, allowlist refusal, --check parsing.
Offline, stdlib only. Usage: python tests/test_self_modify.py
"""
import difflib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

PASSED, FAILED = 0, 0
CURRENT = ""


def check(name, cond, detail=""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
    else:
        FAILED += 1
        print("  FAIL %s [%s] %s" % (CURRENT, name, detail))


def test(name):
    global CURRENT
    CURRENT = name
    print("-- %s" % name)


def load_sm(sandbox):
    path = os.path.join(ROOT, "tools", "self_modify.py")
    spec = importlib.util.spec_from_file_location("self_modify_uut", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ROOT = sandbox
    mod.BACKUP_DIR = os.path.join(sandbox, "state", "patch_backup")
    mod.LOG_PATH = os.path.join(sandbox, "state", "self_modify_log.jsonl")
    mod.LAST_FAILED = os.path.join(sandbox, "LastTestsFailed.log")
    return mod


def git(cwd, *args):
    return subprocess.run(["git"] + list(args), cwd=cwd, capture_output=True,
                          text=True)


def make_patch(path, old, new, create=False):
    if create:
        body = "".join("+%s\n" % l for l in new.splitlines())
        hunk = "@@ -0,0 +1,%d @@\n" % len(new.splitlines())
        return "--- /dev/null\n+++ b/%s\n%s%s" % (path, hunk, body)
    d = list(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                  fromfile="a/" + path, tofile="b/" + path))
    return "".join(d)


def write_patch(patch_dir, text, name="p.diff"):
    p = os.path.join(patch_dir, name)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return p


def main():
    print("== self_modify harness ==")
    # forward slashes: the harness splits --gate-cmd POSIX-style,
    # and backslashes would be eaten as escapes
    py = sys.executable.replace("\\", "/")

    patch_dir = tempfile.mkdtemp(prefix="sm_patch_")
    with tempfile.TemporaryDirectory() as sb:
        git(sb, "init", "-q")
        git(sb, "config", "user.email", "t@t")
        git(sb, "config", "user.name", "t")
        # Mirror the real repo: state/ (backups + jsonl log) is gitignored, so
        # appending a log line must never look like a dirty TRACKED file.
        with open(os.path.join(sb, ".gitignore"), "w") as f:
            f.write("state/\n")
        os.makedirs(os.path.join(sb, "src"))
        target = os.path.join(sb, "src", "mod.cpp")
        with open(target, "w", encoding="utf-8", newline="\n") as f:
            f.write("int a = 1;\n")
        git(sb, "add", ".")
        git(sb, "commit", "-qm", "base")
        sm = load_sm(sb)

        test("dirty tree refused (rc 3)")
        with open(target, "a", encoding="utf-8") as f:
            f.write("// scratch\n")
        p = write_patch(patch_dir, make_patch("src/mod.cpp", "int a = 1;\n",
                                       "int a = 2;\n"))
        check("rc==3", sm.cmd_patch(p, "", "%s -c pass" % py, 60) == 3)
        git(sb, "checkout", "--", "src/mod.cpp")

        test("clean tree + passing gate -> patch KEPT (rc 0)")
        p = write_patch(patch_dir, make_patch("src/mod.cpp", "int a = 1;\n",
                                       "int a = 2;\n"))
        check("rc==0", sm.cmd_patch(p, "", "%s -c print(1)" % py, 60) == 0)
        check("content changed",
              open(target, encoding="utf-8").read() == "int a = 2;\n")
        check("logged kept", '"kept"' in open(sm.LOG_PATH, encoding="utf-8").read())

        test("failing gate -> byte-exact rollback (rc 6)")
        git(sb, "add", "."); git(sb, "commit", "-qm", "keep")
        pre = open(target, "rb").read()
        p = write_patch(patch_dir, make_patch("src/mod.cpp", "int a = 2;\n",
                                       "int a = 3;\n"))
        rc = sm.cmd_patch(p, "", "%s -c \"import sys; sys.exit(1)\"" % py, 60)
        check("rc==6", rc == 6)
        check("bytes restored", open(target, "rb").read() == pre)

        test("file CREATED by a failing patch is deleted")
        p = write_patch(patch_dir, make_patch("src/new.cpp", "", "int n;\n",
                                       create=True))
        rc = sm.cmd_patch(p, "", "%s -c \"import sys; sys.exit(1)\"" % py, 60)
        check("rc==6", rc == 6)
        check("new file gone", not os.path.exists(os.path.join(sb, "src", "new.cpp")))
        check("logged rolled-back",
              '"rolled-back"' in open(sm.LOG_PATH, encoding="utf-8").read())

        test("path outside the allowlist refused (rc 4)")
        with open(os.path.join(sb, "notes.txt"), "w") as f:
            f.write("x\n")
        git(sb, "add", "."); git(sb, "commit", "-qm", "c2")
        p = write_patch(patch_dir, make_patch("notes.txt", "x\n", "x\ny\n"))
        check("rc==4", sm.cmd_patch(p, "", "%s -c print(1)" % py, 60) == 4)

        test("--check parses a synthetic LastTestsFailed.log")
        with open(sm.LAST_FAILED, "w") as f:
            f.write("7:omniseed_fake\n12:omniseed_other\n")
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = sm.cmd_check()
        data = json.loads(buf.getvalue())
        check("rc==0", rc == 0)
        check("two failing suites parsed",
              [x["suite"] for x in data["failing"]] ==
              ["omniseed_fake", "omniseed_other"], str(data))

    print("\nRESULT: %d passed, %d failed" % (PASSED, FAILED))
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
