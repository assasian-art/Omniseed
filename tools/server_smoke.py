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
import base64
import hashlib
import json
import os
import socket
import struct
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


# ---- T103 WebSocket scenarios (raw-socket RFC 6455 client) ------------------

_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def ws_connect(port, token=None):
    """Raw handshake. Returns (socket, response_head, client_key)."""
    s = socket.create_connection(("127.0.0.1", port), timeout=10)
    key = base64.b64encode(b"omniseed-smoke-key-123456").decode()
    path = "/ws" + (("?token=" + token) if token else "")
    req = (f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
           "Upgrade: websocket\r\nConnection: Upgrade\r\n"
           f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n")
    s.sendall(req.encode())
    resp = b""
    while b"\r\n\r\n" not in resp:
        chunk = s.recv(4096)
        if not chunk:
            break
        resp += chunk
    head = resp.split(b"\r\n\r\n")[0].decode("latin1", "replace")
    return s, head, key


def ws_recv_frame(s, timeout=8):
    """Read one server->client frame (unmasked). Returns (opcode, payload)."""
    s.settimeout(timeout)
    hdr = b""
    while len(hdr) < 2:
        hdr += s.recv(2 - len(hdr))
    b0, b1 = hdr[0], hdr[1]
    ln = b1 & 0x7F
    if ln == 126:
        ln = struct.unpack("!H", s.recv(2))[0]
    elif ln == 127:
        ln = struct.unpack("!Q", s.recv(8))[0]
    payload = b""
    while len(payload) < ln:
        payload += s.recv(ln - len(payload))
    return b0 & 0x0F, payload


def ws_send_masked(s, opcode, payload=b""):
    """Client frame (masked, RFC 6455 requires masking)."""
    mask = b"\x01\x02\x03\x04"
    n = len(payload)
    if n < 126:
        hdr = bytes([0x80 | opcode, 0x80 | n])
    else:
        hdr = bytes([0x80 | opcode, 0x80 | 126]) + struct.pack("!H", n)
    masked = bytes(payload[i] ^ mask[i % 4] for i in range(n))
    s.sendall(hdr + mask + masked)


def expected_accept(key):
    return base64.b64encode(
        hashlib.sha1((key + _WS_GUID).encode()).digest()).decode()


def scenario_ws_dev_mode():
    port = 24000 + (os.getpid() % 3000)
    p = boot(port)
    try:
        s, head, key = ws_connect(port)
        try:
            check("ws: handshake 101", "101" in head.split("\r\n")[0])
            check("ws: Sec-WebSocket-Accept matches RFC vector logic",
                  ("Sec-WebSocket-Accept: " + expected_accept(key)) in head)

            op, payload = ws_recv_frame(s)
            check("ws: first health event is a text frame", op == 0x1)
            ev = json.loads(payload)
            check("ws: event type=health with paused/rss",
                  ev.get("type") == "health" and "rss" in ev and
                  "paused" in ev)

            # ping -> pong (RFC 6455 keepalive)
            ws_send_masked(s, 0x9, b"hi")
            op, _ = ws_recv_frame(s)
            check("ws: masked ping answered with pong", op == 0xA)

            # live propagation: REST pause -> next WS event paused:true
            req(port, "/control", {"action": "pause"})
            op, payload = ws_recv_frame(s)
            ev = json.loads(payload)
            check("ws: pause via /control propagates live (paused=true)",
                  op == 0x1 and ev.get("paused") is True)
            req(port, "/control", {"action": "resume"})

            # clean close handshake: client close -> server echoes close
            ws_send_masked(s, 0x8, b"\x03\xe8")
            op, _ = ws_recv_frame(s)
            check("ws: close frame acknowledged", op == 0x8)
        finally:
            s.close()
    except Exception as e:  # noqa: BLE001 - smoke reports, doesn't crash
        check(f"ws dev scenario completed ({type(e).__name__}: {e})", False)
    finally:
        kill(p)


def scenario_ws_auth_mode():
    port = 27000 + (os.getpid() % 3000)
    p = boot(port, token=TOKEN)
    try:
        s, head, _ = ws_connect(port)
        s.close()
        check("ws auth: handshake without token -> 401",
              "401" in head.split("\r\n")[0])
        s, head, _ = ws_connect(port, token="wrong-token-xxxxx")
        s.close()
        check("ws auth: handshake with wrong token -> 401",
              "401" in head.split("\r\n")[0])
        s, head, key = ws_connect(port, token=TOKEN)
        try:
            check("ws auth: handshake with ?token= -> 101",
                  "101" in head.split("\r\n")[0])
            op, payload = ws_recv_frame(s)
            ev = json.loads(payload)
            check("ws auth: health event flows with auth:true",
                  op == 0x1 and ev.get("auth") is True)
        finally:
            s.close()
    except Exception as e:  # noqa: BLE001
        check(f"ws auth scenario completed ({type(e).__name__}: {e})", False)
    finally:
        kill(p)


def main():
    if server_exe() is None:
        print("FAIL omniseed_server binary not built "
              "(configure with -DOMNISEED_BUILD_SERVER=ON)")
        return 2
    scenario_dev_mode()
    scenario_auth_mode()
    scenario_ws_dev_mode()
    scenario_ws_auth_mode()
    if fails:
        print(f"server_smoke: {len(fails)} FAILED")
        return 1
    print("server_smoke: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
