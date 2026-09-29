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

24 tests on a fresh clone (no `.venv`), **39** in a checkout that has one — see
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
`omniseed_agent_modules`, `omniseed_uncertainty_split`,
`omniseed_heads_batch`, `omniseed_streaming_decision`,
`omniseed_feedback_hook`, `omniseed_threshold_derivation`,
`omniseed_signal_audit` and
`omniseed_modality_dump` are deliberately registered **outside** the gate. The
last eight need only **committed** fixtures (no model, no `.venv`, no network), so
they run everywhere including CI.

### The head stack (fast, fully offline)

```bash
./build/bin/omniseed_heads.exe            # 481 checks
./build/bin/omniseed_router.exe           # 348 checks, incl. the <100 us budget
./build/bin/omniseed_unified_output.exe   # 267 checks, incl. the fast path
./build/bin/omniseed_language_heads.exe   # 614 checks
./build/bin/omniseed_soul.exe             # 378 checks, persona + memory + decay
./build/bin/omniseed_calibration.exe      # 189 checks, fitted heads + ECE
./build/bin/omniseed_multimodal.exe       # 110 checks, TokenBus + the joint
./build/bin/omniseed_agent_modules.exe    # 131 checks, the last 4 uncovered modules
./build/bin/omniseed_uncertainty_split.exe # 154 checks, aleatoric vs epistemic
./build/bin/omniseed_heads_batch.exe       # 128 checks, batched head readout
./build/bin/omniseed_streaming_decision.exe # 861 checks, the streaming filter
./build/bin/omniseed_feedback_hook.exe      # 687 checks, the record that scores it
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
./build/bin/omniseed.exe demo-stream   # streaming decisions: churn reduction
./build/bin/omniseed.exe demo-feedback # prediction -> outcome -> persisted record
./build/bin/omniseed.exe gen --help    # generation options
```

`demo-soul` needs no model and no weights. It runs three turns (the third repeats
the first), prints the recall gloss, then prints the **measured** relevance
separation that `min_relevance` comes from — each probe labelled `ok` or
`MISMATCH` — and finishes with a simulated 50-day decay pass. It is the fastest
way to re-derive the thresholds if you change the tokenizer.

`demo-stream` also needs no model. It runs the streaming filter over **three**
streams — the real held-out `h[E]` fixture, unscaled synthetic noise, and the
same noise scaled ×8 — and prints, for `Confirm` and `Window` modes, how often
the raw head output flipped versus how often the committed action changed. The
unscaled stream is there on purpose: the filter refuses to act on it at all
(every step is below the confidence bar), which is the fail-safe working.

> **Both demos now load the FITTED head.** `load_decision_head()` reads
> `models/heads/trading_head.bin` and prints its `provenance()`. When the blob is
> absent or untrained it falls back to the seeded placeholder **with a warning
> naming the path** — never silently. Before §39 the demo built a seeded head
> inline and reported **1 distinct action** on the real tape; with the fitted
> blob it reports **6** and flips **158** times. The "constant head" §37 found was
> the placeholder, not the head.

`demo-feedback` closes the loop: it streams the 369 real held-out bars, records
every prediction, resolves each against the label that actually followed, and
prints `hit_rate`, `filter_hit_rate`, `commitment_rate`, the confidence gap, ECE,
Brier and the confusion matrix. It writes `state/feedback_journal.bin` (the
container) and `state/feedback_journal.tsv` (the rows an offline re-fit would
consume). Measured with the fitted head: **`hit_rate` 0.2439, ECE 0.0555** —
matching `metrics.tsv` exactly, through a completely independent path — and
**0 committed steps**, because the head's maximum calibrated confidence (0.4966)
sits just under the 0.50 bar.

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

# modality rows go through MultimodalBridge, so the h[E] is the runtime's own (§43)
./build/bin/omniseed_dump_hidden.exe vision \
    models/rwkv7-0.1B-ternary.gguf <dir>/labels.tsv build/dump_vision
./build/bin/omniseed_dump_hidden.exe audio \
    models/rwkv7-0.1B-ternary.gguf <dir>/labels.tsv build/dump_audio --codec focal
# --codec mel is REFUSED — no audio->E adapter exists in this tree (§43).
# NOTE: this is the SEPARATE omniseed_dump_hidden.exe binary, not omniseed.exe,
# and the model path is a POSITIONAL 2nd argument.

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

### Batch a backtest (1000 signals at once)

Every head takes one `h[E]` at a time — the right shape for a live turn, the
wrong shape for a backtest. The `*_batch` methods take a `[B, E]` block, which is
exactly the layout `tools/dump_hidden.cpp` writes to `hidden.f32`, so a backtest
can point a head straight at a dump with no repacking:

```cpp
#include "omniseed/core/batch_gemm.h"

std::vector<ClassificationResult> out;
BatchStats stats;
if (head.classify_batch(H, B, "trading.regime", 4, out, &stats))
    log("%d rows, %.3f us/row, %s", stats.rows, stats.us_per_row,
        batch_kernel_name(stats.used));
```

⚠️ **The default kernel is the FAST one, and it is not bit-identical.** `Auto`
resolves to AVX2, which reduces eight lanes instead of one accumulator. If you
need `batch == per-row` exactly, ask for it:

```cpp
head.set_batch_kernel(BatchKernel::Scalar);   // bit-identical, ~5x slower
```

Measured (E = 768, Release): the whole head stack over 1000 signals goes
**25.27 ms → 4.50 ms (5.62×)**. The default kernel changed **0** top-1 labels,
actions and routing decisions across 369 real + 200 synthetic rows.

⚠️ **The backbone is NOT batched.** RWKV-7 here is the scalar recurrent form, so
1000 signals still need 1000 sequential forwards. Batching applies to the
readout. Full detail: **`docs/BATCH.md`**.

### Stream decisions (debounce + hysteresis + release)

A single-shot head acts on noise: one blip in `h[E]` moves the position.
`StreamingDecision` is a **filter over the head's outputs** — it does not make the
head faster or more accurate, it makes its output *stable*.

```cpp
#include "omniseed/streaming_decision.h"

StreamingConfig cfg;                 // Confirm mode, confirm_steps = 3
cfg.switch_confidence = 0.60f;       // changing costs MORE than keeping
StreamingDecision stream;
stream.set_config(cfg);
stream.init(head);                   // NON-owning: the head must outlive it

StreamEvent e;
for (const float* h : stream_of_hidden_states) {
    if (!stream.push(h, e)) continue;     // no observation — NOT agreement
    if (e.first || e.changed) {
        // undo e.previous, apply e.action. Nothing else needs checking.
    }
}
```

Three ways to drive it:

| call | when |
|---|---|
| `push(h, e)` | one head call per step |
| `push_decision(d, e)` | you already have a `DecisionResult` |
| `push_batch(H, B, ev)` | a whole `[B, E]` dump at once, via §36's `decide_batch` |

With `BatchKernel::Scalar` the batch and per-row paths produce the **same** event
stream, so either is safe to use.

⚠️ **Two modes.** `StreamMode::Confirm` (default) demands N *consecutive*
agreeing observations — what a live loop wants. `StreamMode::Window` takes a vote
over a sliding window — order-insensitive, what a backtest wants. In `Window`,
`confirm_steps` is the vote **quorum** and `window` is the ring size.

⚠️ **`ABSTAIN` is never a commitment.** `committed == true` implies a real
action. A quorum of `ABSTAIN` **releases** when something is held and is a no-op
when nothing is. That is what makes `changed events == commits + changes +
releases` hold, so every `changed` event is exactly one transition to undo.

See it run:

```bash
./build/bin/omniseed.exe demo-stream    # model-free; three streams, churn reduction
```

Measured **with the fitted head** (§39): the real held-out tape emits **6
distinct actions and flips 158 times**, and the filter commits **nothing** —
every calibrated confidence is below 0.50, which is the fail-safe working.
Synthetic noise scaled ×8: **169 flips → 6 committed (28.2×)** in `Confirm` mode,
**40 (4.2×)** in `Window`. Full detail: **`docs/STREAMING.md`**.

### Close the loop (prediction → outcome → record)

A head's output is worth nothing until something checks it against what happened.
`FeedbackHook` is that seam: it records every `StreamEvent`, joins it to an
outcome (a realised label, or the user's verdict), and persists the pairs so an
offline re-fit has data.

```cpp
#include "omniseed/feedback_hook.h"

FeedbackHook hook;

// 1. observe — one row per VALID event. An invalid event is not a prediction.
hook.observe_stream(events, "trading.AAPL.1d");

// 2a. a realised outcome for everything open on that key
hook.outcome("trading.AAPL.1d", DecisionAction::SELL, -0.02f, /*horizon=*/5);
// 2b. ...or per row, when each prediction was about a DIFFERENT future
hook.resolve(id, realised_action, score, horizon);

// 2c. ...or from the user. A neutral turn changes NOTHING.
switch (hook.on_user_turn("no, that's wrong", "trading.AAPL.1d")) {
    case UserFeedbackLoop::Verdict::Confirm: break;   // counted as a HIT
    case UserFeedbackLoop::Verdict::Correct: break;   // counted as a MISS
    case UserFeedbackLoop::Verdict::Reject:  break;   // NOT a miss — see below
    default: break;                                   // no verdict, no change
}

const JournalStats s = hook.stats();
s.hit_rate();          // -1.0 when nothing is resolved. NEVER 0.0.
s.filter_hit_rate();   // the FILTER, scored only where it held a position
s.confidence_gap();    // mean_conf_hit - mean_conf_miss; -1.0 unless BOTH exist

hook.save("state/feedback_journal.bin");   // BOTH stores, one container
std::ofstream("state/feedback_journal.tsv") << hook.journal().to_tsv();
```

⚠️ **An unresolved prediction is not a wrong prediction.** `hit_rate()` returns
**-1.0** when nothing has been resolved, never `0.0`: "never checked" and "always
wrong" are different statements. `Rejected` (the user said the decision was
invalid) and `Expired` (the horizon passed with no outcome) are excluded from
every accuracy number for the same reason.

⚠️ **Two actions per row.** `predicted` is the **head's** raw action; `held` is
the **filter's** committed one. They are scored separately, so a filter that is
more stable *and less correct* cannot hide.

⚠️ **Fail closed AND atomically.** A corrupt file leaves both stores exactly as
they were — a journal that restored while its ledger did not would silently reset
every trust factor to neutral.

Full detail: **`docs/FEEDBACK.md`**.

### Enrolling your own voice (`enroll-audio`) — owner action required

Two audio label sets are deliberately **not** downloaded: `audio.wake` (is this
address to me?) and `audio.speaker` (is this the owner?). Both are properties of
*a voice*, and the only voice that matters here is **yours**. A wake head fitted
on 3,000 strangers from a 2017 corpus is fitted on the wrong distribution; a
speaker head can only "know" a speaker it has heard. So the dataset is built from
clips you record, and both sets stay **UNFITTED** until you do. This is the
intended state, not a missing step — `tools/signal_audit.py` lists them under
`unfitted` and says why.

**1. Record ~10 phrases per label into one flat folder.** Any recorder works
(Windows Voice Recorder, Audacity, your phone). 16-bit mono is ideal, but **any
sample rate is accepted** — the loader resamples to 16 kHz and logs it, so a
48 kHz recording is *not* silently mis-melled (§42 fixed exactly that bug).
Aim for 3–5 seconds per clip, and vary it: closer/further, normal/quiet.

The **filename carries the label**: `<label>_<anything>.wav`. The label is
everything before the **first** `_`, and it must be one of this set's exact
labels (lowercase):

| Label set | Labels the head knows | Example filenames |
|---|---|---|
| `audio.wake` | `yes`, `no` | `yes_01.wav`, `yes_02.wav`, `no_01.wav` … |
| `audio.speaker` | `known`, `unknown` | `known_a.wav`, `known_b.wav`, `unknown_a.wav` … |

Suggested `audio.wake` phrases — say each 5×, half as `yes_*`, half as `no_*`:

| # | Phrase | Label it as |
|---|---|---|
| 1 | "Omniseed" | `yes` |
| 2 | "Hey Omniseed" | `yes` |
| 3 | "Omniseed, are you there?" | `yes` |
| 4 | "Omniseed, check the book" | `yes` |
| 5 | "Wake up, Omniseed" | `yes` |
| 6 | "What's the time?" | `no` |
| 7 | "Turn on the kitchen lights" | `no` |
| 8 | "Did you send that message?" | `no` |
| 9 | "Play something else" | `no` |
| 10 | "Thanks, goodbye" | `no` |

For `audio.speaker`, say **anything** (a fixed sentence is fine) 5× yourself as
`known_*`, then have **1–2 other people** — or play a podcast/another room — say
the same sentence 5× as `unknown_*`. A speaker head fitted only on your own voice
has no negative class and will report unfitted; that is the honest outcome, not a
bug. ⚠️ **A single clip in a class cannot be split** — the tool warns
`label '<x>' has ONE clip … will fit as UNFITTED`. Two per label is the floor;
five is comfortable.

**2. Build the manifest.**

```bash
./build/bin/omniseed.exe enroll-audio <dir> [audio.wake|audio.speaker]
```

- `<dir>` is the folder holding the WAVs; `[label-set]` defaults to `audio.wake`.
- Output is `labels.tsv` written **into that same folder**, with the header
  `file  <label-set>  group  sample_rate  n_samples` and `group` always `owner`.
- Files with **no `_` prefix** or an **unreadable WAV** are **reported and
  skipped** — never folded into a label. A mislabelled enrolment clip is a
  permanently wrong head, so a skip is the correct outcome. If **nothing** is
  usable the tool writes **no** manifest and exits non-zero rather than shipping a
  half-empty one.

**3. Fit — a separate, offline step. Do this only after you have recorded.**

```bash
# WAV -> h[E] hidden states (must reach the backbone; writes nothing if it cannot)
# NOTE: separate binary omniseed_dump_hidden.exe; model path is POSITIONAL.
./build/bin/omniseed_dump_hidden.exe audio \
    models/rwkv7-0.1B-ternary.gguf <dir>/labels.tsv state/audio_owner --codec focal
# hidden states -> fitted heads (writes models/heads/*.bin)
.venv/Scripts/python.exe tools/train_heads.py
```

⚠️ **The fit is blocked today by a real gap, not by your recordings.** The
`--codec focal` path works, but the `mel` path — Whisper's `[T/2, 384]` frames —
has **no adapter to the backbone's `E = 768`** in this tree (vision has
`models/vision-proj.gguf`; audio has nothing equivalent), so `--codec mel` is
**refused on purpose** rather than zero-padded into a fabrication. This is
recorded in `docs/VISION_AUDIO_DATA.md` and asserted by
`tests/test_modality_dump.cpp`. **Record now and the data is ready**; the fit
lands the day an audio→E adapter exists. Do **not** substitute a random
projection to make the number appear.

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
