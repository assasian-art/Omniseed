#!/usr/bin/env python3
"""
T100 — Render deploy dry-run check (offline, fail-closed).

Validates the deploy chain WITHOUT network: render.yaml schema, health-route
presence in the server source, docker binary wiring, build/test wiring, and a
secret-literal scan. Exit 2 with specific FAIL lines on any problem.

Registered in CMake as `omniseed_render_check`.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

fails = []
oks = []


def check(name, ok, detail=""):
    line = name + (f" — {detail}" if detail else "")
    (oks if ok else fails).append(line)


# --- render.yaml -------------------------------------------------------------
ry = ROOT / "render.yaml"
text = ry.read_text(encoding="utf-8", errors="replace") if ry.exists() else ""
check("render.yaml exists", ry.exists())
for key in ("type: web", "name: omniseed", "runtime: docker", "plan: free",
            "healthCheckPath:", "dockerCommand:", "dockerfilePath:"):
    check(f"render.yaml declares {key!r}", key in text)

m = re.search(r"healthCheckPath:\s*(\S+)", text)
health_path = m.group(1) if m else None
check("healthCheckPath parsed", bool(health_path), health_path or "missing")

# --- health route exists in the server source --------------------------------
maincpp_path = ROOT / "src" / "server" / "main.cpp"
maincpp = (maincpp_path.read_text(encoding="utf-8", errors="replace")
           if maincpp_path.exists() else "")
check("src/server/main.cpp exists", maincpp_path.exists())
if health_path:
    check(f"health route {health_path} declared in main.cpp",
          health_path in maincpp)

# --- docker binary wiring ----------------------------------------------------
m = re.search(r"dockerCommand:\s*\S*?/?([A-Za-z0-9_]+)\s", text + " ")
binname = m.group(1) if m else None
cm_path = ROOT / "CMakeLists.txt"
cm = cm_path.read_text(encoding="utf-8", errors="replace") if cm_path.exists() else ""
if binname:
    check(f"binary {binname} built by CMake", f"add_executable({binname}" in cm)
else:
    check("dockerCommand binary parsed", False, "no binary name found")

df = ROOT / "Dockerfile"
check("Dockerfile exists", df.exists())
if df.exists() and binname:
    dft = df.read_text(encoding="utf-8", errors="replace")
    check(f"Dockerfile references {binname}", binname in dft)

# --- build + test wiring -----------------------------------------------------
bb_path = ROOT / "build.bat"
bb = bb_path.read_text(encoding="utf-8", errors="replace") if bb_path.exists() else ""
check("build.bat passes -DOMNISEED_BUILD_SERVER=ON",
      "-DOMNISEED_BUILD_SERVER=ON" in bb)
check("omniseed_server_smoke registered in CMakeLists",
      "omniseed_server_smoke" in cm)
check("omniseed_render_check registered in CMakeLists",
      "omniseed_render_check" in cm)
check("docs/RENDER_LIVE_CHECKLIST.md exists",
      (ROOT / "docs" / "RENDER_LIVE_CHECKLIST.md").exists())

# --- secret-literal scan over tracked files ----------------------------------
tracked = subprocess.run(["git", "ls-files"], cwd=ROOT,
                         capture_output=True, text=True
                         ).stdout.splitlines()
pat = re.compile(r"OMNISEED_API_TOKEN\s*=\s*[\"'][^\"']{8,}")
suffixes = (".cpp", ".h", ".py", ".bat", ".sh", ".yaml", ".yml",
            ".json", ".html", ".md", ".txt")
bad = []
for rel in tracked:
    if not rel.endswith(suffixes) or rel.startswith("state/"):
        continue
    p = ROOT / rel
    if p.exists() and pat.search(p.read_text(encoding="utf-8", errors="replace")):
        bad.append(rel)
check("no hardcoded OMNISEED_API_TOKEN literals in tracked files",
      not bad, ", ".join(bad[:3]))

# --- verdict -----------------------------------------------------------------
for line in oks:
    print("PASS", line)
if fails:
    for line in fails:
        print("FAIL", line)
    print(f"render_check: {len(fails)} FAILED, {len(oks)} passed")
    sys.exit(2)
print(f"render_check: all {len(oks)} checks passed")
