# OMNISEED — Runbook

Everything you need to build, test, and run this tree. Copy-paste ready.

---

## 1. Prerequisites

| | |
|---|---|
| OS | Windows 11 (primary), Linux (CI) |
| Compiler | MSVC via **Visual Studio 18 2026** or newer. `build.bat` auto-detects it. No gcc/clang needed on Windows. |
| CMake | 3.16+ (4.x works) |
| Python | 3.10+ — **only** for tests and offline tooling, never for runtime inference |

The tree has **zero external C++ dependencies**. The one vendored third-party
header (`third_party/`, for the optional HTTP server) is checked in.

---

## 2. Build

### The one-liner

```bat
build.bat
```

Release + tests. It finds the toolchain, configures, builds, and runs `ctest`.

```bat
build.bat debug     :: Debug build
build.bat server    :: Release + HTTP server (defines OMNISEED_HTTP)
build.bat clean     :: delete the build directory
```

### Manual configure

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DOMNISEED_BUILD_TESTS=ON \
      -DOMNISEED_BUILD_CLI=ON \
      -DOMNISEED_BUILD_AGENT2=ON
```

Leave the generator **unpinned**. `windows-latest` on GitHub no longer ships
VS 2022, so pinning `-G "Visual Studio 17 2022"` breaks CI.

### ⚠️ The proxy gotcha (Windows + MSBuild)

If `http_proxy` / `HTTP_PROXY` (or the `https` pair) are set in the environment,
MSBuild aborts with:

```
MSB6001: ... Item has already been added
```

Strip them for the build:

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    cmake --build build --config Release --parallel
```

This is not optional on a box behind a proxy. It is the single most common
"the build is broken" report.

### Warnings

`OMNISEED_WERROR` defaults **OFF** and CI does not set it, so GCC *warnings*
never fail CI — but a red GCC build is always a hard **error**. `omniseed_core`
compiles with `/W4 /permissive- /Zc:__cplusplus /utf-8` on MSVC, which is why the
Bengali UTF-8 literals in `src/language/` are safe.

---

## 3. Test

### The whole board

```bash
ctest --test-dir build -C Release --output-on-failure
```

19 tests as of this writing. Expect ~5 minutes — two Python parity suites
dominate it (`omniseed_strategy_parity` is ~140 s, `omniseed_regime_parity`
~38 s).

### ⚠️ A green board is NOT a complete green board

Every **stdlib-only Python** suite is gated on `.venv` existing
(`CMakeLists.txt`, the `if(EXISTS "${OMNISEED_VENV_PY}")` blocks), and CI creates
no `.venv`. So those suites never run in CI — and they do not run here either,
because this worktree has no `.venv`.

**A green `ctest` on a fresh clone means the C++ tests and the three ungated
Python suites only.** Run the rest by hand:

```bash
for t in test_market_data test_market_feeds test_paper_loop test_paper_reports \
         test_monster_features test_monster_sniper test_monster_news \
         test_monster_regime test_monster_strategies test_monster_funding \
         test_decision_bridge; do python tests/$t.py || echo "FAILED: $t"; done
```

This matters: a venv-gated suite that never runs **will** silently drift.
`test_paper_reports` carried an assertion contradicting a later mandate for
months of commits before anyone noticed.

`omniseed_decision_bridge`, `omniseed_regime_parity` and
`omniseed_strategy_parity` are deliberately registered **outside** the gate.

### The head stack (fast, fully offline)

```bash
./build/bin/omniseed_heads.exe            # 481 checks
./build/bin/omniseed_router.exe           # 348 checks, incl. the <100 us budget
./build/bin/omniseed_unified_output.exe   # 267 checks, incl. the fast path
./build/bin/omniseed_language_heads.exe   # 614 checks
```

These four print measured latency, so running them is also how you re-measure:

```
  decide(general ):   0.498 us/call
  decide(language):   1.150 us/call
  worst: language at 1.150 us (budget 1000 us)
  router: 15.307 us/call over 4000 calls
```

### A single suite

```bash
ctest --test-dir build -C Release -R omniseed_router --output-on-failure
```

---

## 4. Run

### CLI modes

```bash
./build/bin/omniseed.exe chat --model models/rwkv7-0.1B-ternary.gguf
```

| `--mode` | Behaviour |
|---|---|
| `off` (default) | generate text; never consult the decision head |
| `text-only` | alias for `off` |
| `hybrid` | consult the head first; act when confident, else generate |
| `decision-only` | never generate; return the head's decision (or ABSTAIN) |

An unknown `--mode` is **refused with exit code 2**, not silently defaulted — a
typo that quietly turned the feature off would look like the feature failing.

Related flags:

```
--decision-threshold F   self-routing confidence bar (default 0.85)
--decision-head PATH     fitted projection blob. Without it the head is an
                         UNTRAINED placeholder: structurally valid, meaningless.
```

### Other commands

```bash
./build/bin/omniseed.exe info          # build + config report
./build/bin/omniseed.exe tools         # list available tools
./build/bin/omniseed.exe selftest      # kernel self-test
./build/bin/omniseed.exe gen --help    # generation options
```

---

## 5. The unified pipeline (library API)

The router, heads and unified document are wired as a library. Nothing on the CLI
path uses them yet — see `docs/UNIFIED_ARCHITECTURE.md` §8 for why.

```cpp
#include "omniseed/unified_output.h"
using namespace omniseed;

UnifiedPipeline pipe;
pipe.init(768);                          // registers all default label sets + domains

float h[768] = { /* your backbone's hidden state */ };

UnifiedOutput out = pipe.run(h, [](const DomainDecision& d) {
    return "explaining " + d.action;     // called ONLY if the plan names the token head
});

std::printf("%s\n", out.to_json().c_str());
```

Switch heads off deliberately:

```cpp
UnifiedPipeline::Config c = pipe.config();
c.force_mode = true;
c.mode       = RouterMode::DecisionOnly;   // never pay for a decode
pipe.set_config(c);
pipe.router().set_config(c.router);        // both, or the fast path sees stale gates
```

> **`trained()` is false and `provenance()` says `NOT FULLY FITTED` until every
> row is fitted.** A seeded head emits well-formed but meaningless values. Do not
> ship a decision on one. See `docs/JEV_FEATURES.md` §2.

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `MSB6001 ... Item has already been added` | proxy env vars | `env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY cmake --build ...` |
| `--mode: unknown value` | typo | modes are `off \| text-only \| hybrid \| decision-only` |
| `model load failed: cannot map ./models/omniseed.gguf` | no model in this checkout | pass `--model models/rwkv7-0.1B-ternary.gguf`, or ignore — the head stack needs no model |
| A Python suite never runs | `.venv` gate | run it by hand (see §3) |
| GCC build red in CI but green locally | GCC is stricter than MSVC | use the Godbolt probe (below) |
| `git push` hangs | no credential in the sandbox | push from the desktop checkout |

### Reproducing a Linux-only GCC error from Windows

The public Godbolt compile API reproduces ubuntu-latest GCC diagnostics exactly:

```
POST https://godbolt.org/api/compiler/g141/compile
```

with `-fsyntax-only` and the project headers passed in the `files` array. ⚠️ It
uploads your source to a third party — only use it on code you are comfortable
sending there.

### CI triggers

`ci.yml` runs on **pushes to `main`** and on **pull requests**. Pushing a feature
branch does nothing unless a PR is open.
