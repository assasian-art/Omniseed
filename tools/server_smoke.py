#!/usr/bin/env python3
"""
T100/T101/T102 — omniseed_server integration smoke.

Boots the real server binary MODEL-LESS on an ephemeral port and asserts over
real HTTP, stdlib only, no network beyond loopback:

  dev mode (OMNISEED_API_TOKEN unset):
    /health 200 + ok/paused/auth fields   (public)
    GET   /        200, console carries the pause/resume toggle
    POST  /control status/pause/resume roundtrip
    POST  /gen     gated with error:paused while paused

  auth mode (OMNISEED_API_TOKEN set):
    /health stays public, reports auth:true
    /gen, /control without token -> 401; wrong token -> 401
    /control with correct Bearer -> 200
    GET / console still served (token entered in-page)

Registered in CMake as `omniseed_server_smoke`.
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOKEN = "smoketoken-1234567890abcdef"

fails = []


def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    if not ok:
        fails.append(name)


def server_exe():
    for c in (ROOT / "build" / "bin" / "omniseed_server.exe",
              ROOT / "build" / "bin" / "omniseed_server"):
        if c.exists():
            return c
    return None


def req(port, path, data=None, token=None):
    url = f"http://127.0.0.1:{port}{path}"
    body = json.dumps(data).encode() if data is not None else None
    r = urllib.request.Request(url, data=body,
                               method="POST" if body is not None else "GET")
    r.add_header("Content-Type", "application/json")
    if token:
        r.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(r, timeout=10) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def boot(port, token=None):
    env = dict(os.environ)
    if token is None:
        env.pop("OMNISEED_API_TOKEN", None)
    else:
        env["OMNISEED_API_TOKEN"] = token
    p = subprocess.Popen([str(server_exe()), "--port", str(port)],
                         cwd=str(ROOT), env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):
        try:
            s, _ = req(port, "/health")
            if s == 200:
                return p
        except Exception:
            pass
        time.sleep(0.1)
    kill(p)
    raise RuntimeError("server did not become healthy within 12s")


def kill(p):
    p.terminate()
    try:
        p.wait(timeout=5)
    except Exception:
        p.kill()
        p.wait(timeout=5)


def scenario_dev_mode():
    port = 18000 + (os.getpid() % 3000)
    p = boot(port)
    try:
        s, b = req(port, "/health")
        h = {}
        try:
            h = json.loads(b)
        except Exception:
            pass
        check("dev: /health 200 (public)", s == 200)
        check("dev: /health has ok/paused/auth fields",
              all(k in h for k in ("ok", "paused", "auth")))

        s, html = req(port, "/")
        check("dev: GET / console 200", s == 200)
        check("dev: console carries pause/resume toggle",
              ("'pause'" in html or '"pause"' in html) and "resume" in html)

        s, b = req(port, "/control", {"action": "status"})
        check("dev: /control status 200", s == 200)

        s, b = req(port, "/control", {"action": "pause"})
        check("dev: /control pause 200", s == 200)
        s, b = req(port, "/control", {"action": "status"})
        check("dev: paused=true after pause",
              '"paused":true' in b.replace(" ", ""))

        s, b = req(port, "/gen", {"prompt": "hello"})
        check("dev: /gen returns error:paused while paused", "paused" in b)

        s, b = req(port, "/control", {"action": "resume"})
        check("dev: /control resume 200", s == 200)
        s, b = req(port, "/control", {"action": "status"})
        check("dev: paused=false after resume",
              '"paused":false' in b.replace(" ", ""))
    finally:
        kill(p)


def scenario_auth_mode():
    port = 21000 + (os.getpid() % 3000)
    p = boot(port, token=TOKEN)
    try:
        s, b = req(port, "/health")
        flat = b.replace(" ", "")
        check("auth: /health still public", s == 200)
        check("auth: /health reports auth:true", '"auth":true' in flat)

        s, _ = req(port, "/gen", {"prompt": "x"})
        check("auth: /gen without token -> 401", s == 401)
        s, _ = req(port, "/gen", {"prompt": "x"}, token="wrong-token-xxxxx")
        check("auth: /gen wrong token -> 401", s == 401)
        s, _ = req(port, "/control", {"action": "status"})
        check("auth: /control without token -> 401", s == 401)
        s, _ = req(port, "/control", {"action": "status"}, token="wrong")
        check("auth: /control wrong token -> 401", s == 401)

        s, b = req(port, "/control", {"action": "status"}, token=TOKEN)
        check("auth: /control with token -> 200", s == 200)

        s, html = req(port, "/")
        check("auth: GET / console still served (token in-page)", s == 200)
    finally:
        kill(p)


def main():
    if server_exe() is None:
        print("FAIL omniseed_server binary not built "
              "(configure with -DOMNISEED_BUILD_SERVER=ON)")
        return 2
    scenario_dev_mode()
    scenario_auth_mode()
    if fails:
        print(f"server_smoke: {len(fails)} FAILED")
        return 1
    print("server_smoke: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
