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

23 tests on a fresh clone (no `.venv`), **38** in a checkout that has one — see
the warning below. Expect 5 minutes on a fast box; **35-45 minutes** on a loaded
one, because the LoRA suites alone can take ~8 minutes each.

To skip the four slow LoRA suites while iterating:

```bash
ctest --test-dir build -C Release --output-on-failure \
      -E "omniseed_lora_e2e|omniseed_lora_gguf|omniseed_lora_chat|omniseed_lora_i8"
```

⚠️ **`ctest -I` is not a test-number list.** The flag parses as
`Start,End,Stride,test#,test#...`, so `-I 1,2,3,4,5,10,...` silently runs
`Start=1,End=2,Stride=3` **plus** the numbers after it — not the list you wrote.
Use `-R <regex>` or `-E <regex>`.

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

`omniseed_decision_bridge`, `omniseed_regime_parity`,
`omniseed_strategy_parity`, `omniseed_calibration`, `omniseed_multimodal`,
`omniseed_agent_modules` and `omniseed_uncertainty_split` are deliberately
registered **outside** the gate. The last three need only **committed** fixtures
(no model, no `.venv`, no network), so they run everywhere including CI.

### The head stack (fast, fully offline)

```bash
./build/bin/omniseed_heads.exe            # 481 checks
./build/bin/omniseed_router.exe           # 348 checks, incl. the <100 us budget
./build/bin/omniseed_unified_output.exe   # 267 checks, incl. the fast path
./build/bin/omniseed_language_heads.exe   # 614 checks
./build/bin/omniseed_soul.exe             # 378 checks, persona + memory + decay
./build/bin/omniseed_calibration.exe      # 180 checks, fitted heads + ECE
./build/bin/omniseed_multimodal.exe       # 110 checks, TokenBus + the joint
./build/bin/omniseed_agent_modules.exe    # 131 checks, the last 4 uncovered modules
./build/bin/omniseed_uncertainty_split.exe # 154 checks, aleatoric vs epistemic
```

`omniseed_calibration` is registered **outside** the `.venv` gate on purpose. Its
Part C is gated on the committed blobs under `models/heads/` and the fixtures
under `tests/fixtures/head_calibration/` — if those are missing it skips Part C
and says so, rather than passing vacuously.

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
./build/bin/omniseed.exe demo-soul     # persona + memory + recall + decay
./build/bin/omniseed.exe gen --help    # generation options
```

`demo-soul` needs no model and no weights. It runs three turns (the third repeats
the first), prints the recall gloss, then prints the **measured** relevance
separation that `min_relevance` comes from — each probe labelled `ok` or
`MISMATCH` — and finishes with a simulated 50-day decay pass. It is the fastest
way to re-derive the thresholds if you change the tokenizer.

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
>
> As of Milestone 4 the committed blobs under `models/heads/` **are** fitted and
> calibrated — but that is opt-in: a head is only fitted if you `load()` it.
> See "Loading a calibrated head" below.

### Turning on the soul

The soul is **off by default**, so existing paths produce byte-identical
documents. Turn it on deliberately:

```cpp
UnifiedPipeline::Config c = pipe.config();
c.use_soul = true;                  // then re-init: pipe.init(768)
pipe.set_config(c);
pipe.init(768);                     // init() applies the config

SoulState st = pipe.soul().perceive("I'm worried, this is urgent");
UnifiedOutput out = pipe.run(h, std::string("I'm worried, this is urgent"));
std::printf("%s\n", out.to_json().c_str());   // now carries a "soul" section
```

Or use the soul on its own — it needs no model and no hidden state:

```cpp
Soul soul; soul.init();
SoulState st = soul.perceive("make me a guaranteed profit");
std::string reply = soul.speak("Here is the plan.", st);
//  st.persona.stance == Refuse, value_id == "no_guarantees"
//  reply == "I won't do that. ..." — the base reply is REPLACED, not wrapped.

soul.record("buy 50 AAPL", "uptrend + MACD agree", 0.72);
soul.resolve(+0.031);                            // realised outcome
std::printf("%s\n", soul.capability_report().c_str());
```

Full design, the three ordering rules, and the honest limitations:
**`docs/SOUL.md`**.

### Giving the soul memory

Memory is **on by default** in `Soul::Config`. It costs one minimal tokenizer and
under 2 MB of crystals, and it needs no model.

```cpp
Soul soul; soul.init();

// The turn API. converse() is the only stateful path: it recalls, composes the
// reply, and stores the exchange. speak()/perceive() stay pure.
std::string r1 = soul.converse("what is my position size limit",
                               "Two percent per trade.");
std::string r2 = soul.converse("what is my position size limit",
                               "Two percent per trade.");
//  r2 == "I said this before: \"Two percent per trade.\". Two percent per trade."
//  The store did NOT grow: a repeated exchange is reinforced, not re-filed.
std::printf("%zu crystals\n", soul.memory_size());

// Persist across runs (best-effort; a missing file is not an error).
Soul::Config c; c.crystals_path = "state/soul_crystals.bin";
Soul s2; s2.init(c); s2.load_memories(); /* ... */ s2.save_memories();

// The decay pass, and how to simulate time without waiting for it.
soul.advance_memory_clock(5000000);   // ~50 days at 100k tokens/day
soul.decay_memories();
```

> **`perceive()` is const and stores nothing, deliberately.**
> `UnifiedPipeline::run()` is const and calls it; a const method that silently
> appends to long-term memory would make `run()` non-idempotent in a way its
> signature denies. Storing lives in `perceive_and_recall()`, `remember()` and
> `converse()`. From the pipeline, use the non-const `run_memorable()`.

> **Two thresholds are measured, not guessed**: `min_relevance` (0.92) and
> `duplicate_recall_score` (0.98). Both are byte-level-tokenizer values. If you
> load a real vocabulary, **re-derive them with `omniseed demo-soul`** — the
> margin is only ~0.10 wide.

### Loading a calibrated head

Fitted heads are committed under `models/heads/` (tens of KB each). Without them
a fresh clone has only **seeded placeholders** — structurally valid, meaningless.

```cpp
#include "omniseed/classification_head.h"
#include "omniseed/decision_head.h"
using namespace omniseed;

ClassificationHead ch;
ch.init(768);
if (!ch.load("models/heads/trading_regime_head.bin")) {
    // FAIL CLOSED. Do not fall back to a seeded placeholder: it emits a
    // number that looks like a decision and is not one.
}
// load() rebuilds the sets from the file, so look the index up AFTER loading.
int32_t ri = ch.find_label_set("trading.regime");   // -1 if absent
// ch.trained() == true
// ch.temperature(ri)        == 3.250   (temperature() with NO index is the
//                                       DEFAULT, which is 1.0 here)
// ch.calibration_error(ri)  == 0.0343  (ECE, lower is better)

DecisionHead dh;
dh.init(768);
dh.load("models/heads/trading_head.bin");
// dh.temperature() == 13.325, dh.calibration_error() == 0.0555
// dh.provenance() == "... + calibrated (n=369, ece=0.06%)"
```

`calibration_error()` returns **`< 0` when unmeasured** — never `0.0`, which
would read as "perfectly calibrated". Check `calibrated()` before trusting a
confidence, and check `trained()` before trusting the head at all.

| blob | contents |
|---|---|
| `models/heads/language_head.bin` | `language.{intent,language,sentiment}`, one T each |
| `models/heads/trading_regime_head.bin` | `trading.regime` |
| `models/heads/trading_head.bin` | `DecisionAction` |

⚠️ **A calibrated head can still be a bad head.** The trading action head is
right **24%** of the time — calibration makes it *safe* (max confidence 0.4966,
so it self-routes **0/369** rows and escalates everything), not *good*. Read
`docs/CALIBRATION.md` §6 before using any of these numbers.

### Multi-modal — an image becomes h[E]

The vision and audio encoders were always real; nothing joined them to the head
stack. `MultimodalBridge` is that join, and it is small on purpose.

```cpp
#include "omniseed/multimodal.h"
using namespace omniseed;

MultimodalBridge bridge;
bridge.init(model.config().n_embd, tok);          // tok must outlive the bridge
bridge.set_codebook(model.token_embeddings());    // non-owning: the model owns it

// vision_rows is an encoder's output, row-major [M, n_embd]. audio_codes come
// from FocalCodec/WhisperTiny and are already ids, so they pass straight through.
auto f = bridge.fuse("describe the chart", vision_rows, audio_codes);
if (!f.ok) {
    // FAIL CLOSED. A failed fusion is NOT an empty one — read f.error.
}

// Run the fused stream through the backbone, then let a head read the h[E].
RwkvState st; model.init_state(st);
Tensor logits("logits", {model.vocab_size()}, DType::F32);
Tensor hidden("hidden", {model.config().n_embd}, DType::F32);
for (int32_t id : f.ids) model.forward(id, st, logits, &hidden);

auto r = pipe.classifier().classify(hidden.f32(), "vision.scene");
```

The codebook **is** the model's token embedding matrix, so the modality lands in
the discrete space `forward()` accepts with no training and no extra RAM.

⚠️ **The vision/audio heads are UNFITTED.** They report the base rate — or, worse,
a confident-looking seeded number (measured: `vision.scene` printed **0.997** on
a seeded projection). The joint works; the heads behind it do not yet know
anything. See `docs/MULTIMODAL.md` §5.

`codebook_stride` is a **real accuracy/speed dial**: with `stride = 2`, a token
whose nearest embedding row has an odd index is unreachable.

### Re-training the heads (offline only)

Training is Python and **never runs in the runtime**. The runtime only loads the
`.bin`. To reproduce the committed blobs:

```bash
# 1. Collect h[E]  (needs the backbone; ~1-2 min for the language set)
./build/bin/omniseed_dump_hidden.exe language \
    models/rwkv7-0.1B-ternary.gguf tools/data/head_language.tsv build/dump_language

./build/bin/omniseed_dump_hidden.exe market \
    models/rwkv7-0.1B-ternary.gguf models/market/AAPL_1d.csv build/dump_market --stream

# 2. Fit + calibrate + write blobs and fixtures
python tools/train_heads.py        # or .venv/Scripts/python.exe tools/train_heads.py

# 3. Verify
./build/bin/omniseed_calibration.exe
```

`--dry-run` on the collector prints the plan and writes nothing. `--stream` is
required for the market path: RWKV is recurrent, so all bars go through **one**
evolving `RwkvState` — windowed integer-epoch timestamps are out of distribution.
Full method, split protocol, and the honest accuracy table:
**`docs/CALIBRATION.md`**.

### Why am I unsure? (aleatoric vs epistemic)

`core/uncertainty.h` gives a *total* uncertainty signal (entropy, margin).
`core/uncertainty_split.h` splits it into the two causes that need opposite
responses — data ambiguity (stay out) vs an input outside the fitted reference
(go and learn):

```cpp
#include "omniseed/core/uncertainty_split.h"

UncertaintyDecomposition ud;
ud.fit(hidden_rows, n, E);                    // any [n, E] block of h[E]

// `probs` must be the head's CALIBRATED softmax output (temperature applied).
UncertaintySplit s = ud.split(h, probs, K);
if (s.recommend_abstain)
    log("abstain: %s (a=%.2f e=%.2f)", s.reason.c_str(), s.aleatoric, s.epistemic);
```

Measured on the shipped held-out heads (reproduce with
`python tools/uncertainty_audit.py`):

* **aleatoric** (calibrated entropy) is monotone in the observed error rate for
  **4 of the 5** fitted heads — `trading.regime` 0.120 → 0.452 across entropy
  quartiles, `language.sentiment` 0.059 → 0.500.
* **epistemic** (h-space distance) has **no within-domain** error signal — and
  that is the correct result, not a defect: inside one domain there is nothing
  to find.
* **epistemic separates domains completely** — a reference fitted on trading
  flags **71/71** language vectors, and the reverse flags **369/369** past the
  language p90.

⚠️ The epistemic half is a **covariate-shift detector**, not a parameter
posterior. `from_ensemble()` implements the rigorous Depeweg decomposition and
is tested, but nothing can feed it yet (no head ensemble has been fitted).
Full detail, the quartile tables, the fail-closed policy and the blob format:
**`docs/UNCERTAINTY.md`**.

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
