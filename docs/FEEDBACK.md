# FEEDBACK — the record that closes the loop

**Milestone 10. Phase 2.3 of the mandate.**

Every head in this tree produces a number. Until this milestone **nothing
recorded what happened next**, so the number could never be checked against
reality and the loop could never close. A system that predicts and never scores
itself is not learning — it is only talking.

> **The one-sentence result.** A `StreamEvent` from §37 is now written to a
> persisted journal, resolved against the outcome that actually followed, and
> scored — and the first thing it did was **overturn a result from §37**. On the
> 369 real held-out bars the fitted head emits **6 distinct actions** and flips
> **158 times**; the "constant action" §37 reported was an artifact of the
> *seeded* head the demo used, not a property of the head. The record also
> re-derives §32's published numbers from scratch: **`hit_rate` 0.2439** and
> **ECE 0.0555**, matching `metrics.tsv` to four decimals, from the raw `h[E]`
> and the raw labels through a completely independent path.

---

## 1. What already existed, and what the gap was

This milestone was an audit before it was a build, and the audit said the same
thing §30–§37 did: **the parts existed and nothing joined them.**

| part | existed? | what it does | what it did NOT do |
|---|---|---|---|
| `StreamEvent` (§37) | yes | one prediction + the filter's commitment, per step | dropped at the end of the step |
| `UserFeedbackLoop` (`agent_intel.h`) | yes | confirm / correct / reject + a trust factor per task key | **had no persistence at all** |
| `SelfImprovement` | yes | a persisted trace cache | answers "which tool sequence worked", not "was the prediction right" |
| `DecisionJournal` / `FeedbackHook` | **no** | — | — |

Two concrete gaps, both of the same shape:

1. **Nothing persisted the ledger.** A trust factor that resets to neutral on
   every restart is not a trust factor. `UserFeedbackLoop` now has
   `save`/`load`/`serialise`/`deserialise` (`'OMNF'`).
2. **Nothing joined a prediction to an outcome.** That is `DecisionJournal`, and
   `FeedbackHook` is the seam that keeps the journal and the ledger consistent.

---

## 2. The design decision that matters: two actions, not one

Every recorded row carries **both**:

| field | meaning |
|---|---|
| `predicted` | the **HEAD's** raw action this step (`StreamEvent::observed_action`) |
| `held` | the **FILTER's** committed action after this step (`StreamEvent::action`) |

They are different claims and they are scored separately:

* `hit_rate()` answers *is the projection any good?*
* `filter_hit_rate()` answers *does the debounce/hysteresis layer help or hurt?*

Collapsing them into one number would hide the case that matters most — a filter
that is **more stable and less correct**. The filter is scored only on the steps
where it actually held a position (`committed`), because an uncommitted step is
not a filter decision.

---

## 3. An unresolved prediction is NOT a wrong prediction

This is the convention the whole module is built around, and it is the same one
`calibration_error() < 0` and `distance() == -1.0` already use. Six outcome
kinds; **three are counted and three are excluded from every accuracy number**,
each for its own reason:

| kind | counted? | why |
|---|---|---|
| `Unknown` | no | recorded, no outcome yet |
| `Realised` | **yes** | a ground-truth action/label arrived — hit or miss |
| `Confirmed` | **yes** | the user said we were right |
| `Corrected` | **yes** | the user said we were wrong |
| `Rejected` | no | the user said this was not a valid thing to decide. **A decision that should not have been made is a different failure from a decision that was wrong**, and averaging the two would flatter the system |
| `Expired` | no | the horizon passed with no outcome. **A market that never moved is not evidence that the call was bad** |

`hit_rate()` therefore returns **-1.0**, never `0.0`, when nothing has been
resolved:

```
untested head:  100 rows, 0 resolved  ->  hit_rate() == -1.0
always-wrong:   100 rows, 0 correct   ->  hit_rate() ==  0.0
```

`tests/test_feedback_hook.cpp` **G1** asserts exactly that pair, and asserts the
two journals have the same row count and the same confidences, so anything that
reported on rows alone would conflate them.

### One prediction, one score

`resolve()` refuses an id that is already resolved, and `resolve_kind(Unknown)`
is refused outright. A record that can be scored twice will eventually agree
with itself.

---

## 4. The numbers, and what they mean

All measured over the **369 held-out trading bars** — the same rows §32 fitted
its temperature on, with the same labels, so nothing here is a new claim about
the data.

| statistic | seeded head | **fitted head** | `metrics.tsv` says |
|---|---|---|---|
| distinct actions | **1** (constant) | **6** | — |
| raw flips over 240 steps | 0 | **158** | — |
| `hit_rate` | 0.2114 | **0.2439** | 0.2439 |
| `ece` | 0.7533 | **0.0555** | 0.0555 |
| max calibrated confidence | — | **0.4966** | — |
| filter verdicts | 367 | **0** | — |

The last row is the honest headline: the fitted head is **richly varying and the
filter still commits nothing**, because every calibrated confidence sits below
the `min_confidence = 0.50` bar. That is fail-closed working exactly as §32
predicted, and it is a *better* result than a filter that acted.

> **Reported, not tuned.** 0.4966 is **0.0034** below 0.50. The gate is closed by
> a hair, and that margin is a coincidence of two numbers chosen independently
> (the trainer's temperature, the filter's default bar) — not a designed
> boundary. Lowering `min_confidence` to make the demo commit would be fitting
> the demo to look good, so the constant is left alone and the finding is
> printed on every run instead.

| statistic | what it is | how it can be wrong |
|---|---|---|
| `hit_rate` | `hits / verdicts`, or **-1.0** | never confuses "untested" with "bad" |
| `filter_hit_rate` | the same, over committed steps only | an uncommitted step is not a filter decision |
| `commitment_rate` | committed steps / all steps | how often the filter chose to act |
| `mean_conf_hit` / `mean_conf_miss` | mean confidence when right / wrong | `-1.0` when that side is empty |
| `confidence_gap` | `mean_conf_hit - mean_conf_miss` | **-1.0** unless BOTH sides exist |
| `ece` | reliability of `confidence` as p(correct), 10 bins | `-1.0` when nothing is resolved |
| `brier` | mean squared error of the confidence | `-1.0` when nothing is resolved |
| `confusion[pred][realised]` | counts, `Realised` rows only | a `Confirmed` row has no realised action and is absent |

**A sign you must measure, never assert.** `confidence_gap > 0` means the head's
confidence is informative. Every claim that it *is* is paired in the test suite
with an **anti-correlated control** in the same test (**C6**), where the same
shape with the confidences swapped must produce a *negative* gap. A detector that
cannot fail is not a detector.

**With no misses the gap is `-1.0`, not a flattering `+0.9`** (**C7**). One side
is missing, so the difference is genuinely undefined; reporting it would be
inventing a measurement.

---

## 5. The joint with §37

`record_stream()` takes the event log the filter already produces and writes one
row per **valid** event. An invalid event — a step where the head refused to
observe — is **not a prediction** and is not recorded; recording it would count a
dead head's silence as a miss.

The §37 bookkeeping identity extends to the record:

```
rows with changed == true  ==  filter.commits() + filter.changes() + filter.releases()
```

asserted in **F1** over a sticky 400-step stream, and printed by `demo-feedback`
so a mismatch is visible rather than merely asserted. The test also asserts
`commits > 0 && changes > 0 && releases > 0`, because the §37 lesson was that a
uniform stream only ever walks the "nothing happened" path.

---

## 6. Persistence: fail closed AND atomically

`DecisionJournal` and `UserFeedbackLoop` each own a format, and `FeedbackHook`
stores **both in one container** (`'OMNH'`) whose payloads are the exact bytes
each store's own `serialise()` produces. No format is re-implemented, so the two
copies cannot drift.

Every length and every enum value is read straight off the disk, so:

* `need(n)` is applied before **every** read;
* an absurd count is rejected before it is used to `reserve`;
* action and outcome enums are **range-checked** before they can index anything
  (a corrupt action byte would otherwise walk off `confusion` and
  `decision_action_name`'s switch);
* the parse is built into a **local** object and swapped in only on full
  success.

`FeedbackHook::load()` parses **both** payloads into locals before committing
either. A journal that restored while its ledger did not would silently reset
every trust factor to neutral — a quiet loss of the only thing the ledger is for.
**E5** truncates the file inside the ledger payload, with the journal intact, and
asserts neither store was replaced.

`serialise()` is **deterministic**: the ledger is written sorted, so two runs
that recorded the same verdicts produce byte-identical bytes. `unordered_map`
iteration order leaking into the file would break this (**E6**).

---

## 7. Reproducing

```bash
# the whole closed loop, on the real 369 held-out bars
./build/bin/omniseed.exe demo-feedback
```

It writes two files, both under the gitignored `state/`:

| file | what |
|---|---|
| `state/feedback_journal.bin` | the container: journal + trust ledger |
| `state/feedback_journal.tsv` | the rows an **offline re-fit** would consume |

The TSV columns are
`id, step, task_key, predicted, confidence, margin, held, committed, outcome,
realised, score, horizon` — deliberately the same shape the offline fitter
already reads, so closing the loop needs no new plumbing.

The test suite is **ungated** (no model, no `.venv`, no network):

```bash
./build/bin/omniseed_feedback_hook.exe     # 687 checks, 0 fail, 0 skip
```

| part | what it pins |
|---|---|
| A | recording: monotonic ids, an invalid event is not a prediction, the bound, the newest survive |
| B | resolution: unknown id refused, re-scoring refused, `resolve_latest` picks the newest, `expire` is key-scoped |
| C | **stats honesty**: -1.0 vs 0.0, `Expired`/`Rejected` are not misses, the confidence gap's sign, ECE, Brier, the confusion matrix |
| D | the verdict join, including that a **neutral turn changes nothing** |
| E | persistence: round-trip, corrupt file fails closed **and atomically**, deterministic bytes, "a different key differs" |
| F | the §37 joint, the 369 real rows, the label-rotation control, and **F6 = the §39 acceptance test** |
| G | the acceptance-shaped controls: measured zero vs unmeasured, an anti-correlated journal |
| H | the remaining public surface (enum names, both `to_json`s, `observe_decision`, `clear`) |

---

## 8. The §39 acceptance test, and what it turned out to be

The mandate's acceptance test for head fitting was: *"Re-run the §35 uncertainty
audit and §37 stream A AFTER fitting — the constant-action result must
disappear."* Both halves were run. **Both already had the fitted head.**

| re-run | what it found |
|---|---|
| `tools/uncertainty_audit.py` (§35) | **Unchanged.** It always read `models/heads/*.bin`. Its result — DecisionAction ALEATORIC `[no signal]` at accuracy 0.2439, the other four heads MONOTONE — is a statement about the FITTED head and always was. |
| `omniseed demo-stream` (§37 stream A) | **The constant-action result is gone**: 1 distinct action → **6**, 0 flips → **158**. |

So the acceptance test passes — and the reason it passes is worth recording,
because it is the same shape as every other milestone in this project:

> §37 stream A was not measuring the head. It was measuring a **seeded
> placeholder** the demo had constructed inline, because no demo path ever
> loaded the committed blob. The finding was real, correctly reported, and about
> the wrong subject. The gap was the **joint**, not the head.

`cmd_demo_stream()` and `cmd_demo_feedback()` now share `load_decision_head()`,
which loads `models/heads/trading_head.bin` and **says so out loud** — falling
back to the placeholder with a warning naming the path, never silently.

The acceptance test is pinned as **`test_feedback_hook.cpp` F6**, gated on the
committed blob, asserting `distinct > 1` and cross-checking `hit_rate` and `ece`
against `metrics.tsv`.

---

## 9. Honest gaps (do not overclaim)

- **Nothing consumes the hook yet.** No agent loop, no trading loop, no backtest
  calls `FeedbackHook`. The only caller is `demo-feedback`. This is the same
  recurring gap §30–§37 each ended with: the capability and its evidence exist,
  the consumer does not.
- **The runtime never trains.** `to_tsv()` makes re-fitting *possible*; nothing
  re-fits. `tools/train_heads.py` is still the only thing that writes a `.bin`,
  and it reads its own fixtures, not this journal.
- **The fitted head is well-calibrated AND weak.** Accuracy 0.2439, ECE 0.0555.
  The two are independent, and calibration makes a weak head **safe**, not good.
  Nothing here changes that, and `CLOSE` / `HEDGE` still have zero holdout
  recall.
- **`score` is caller-supplied and never predicted.** No objective function is
  computed from it anywhere in this module. The mandate's R2 (never claim or
  target "no loss") holds **by construction**: there is no loss function here,
  only a record of what happened.
- **The labels are a rule, not P&L.** A high `hit_rate` means the head imitates
  the teacher of §2.2 well. It says nothing about profitability.
- **`on_implicit()` moves the ledger and not the journal, on purpose.** Silence
  is not an outcome; treating it as one would let a user scrolling past score the
  head.
- **The journal is bounded** (`max_entries`, default 8192, FIFO). A record that
  grows without limit is a leak in a 24/7 process.
- **`FeedbackHook::expire()` is step-scoped, not time-scoped.** A caller with a
  wall-clock horizon must map it to a step itself; the journal has no clock
  beyond `at_ms` per row.
