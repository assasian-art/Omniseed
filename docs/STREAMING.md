# OMNISEED — Streaming decisions

**Milestone 9 · Phase 2.2 · `include/omniseed/streaming_decision.h`**

The decision head is single-shot: one `h[E]` in, one action out. That is the
right primitive for a judgement about one situation and the wrong primitive for
a control loop — a loop that acts on every single-shot output acts on noise.
This document records what the streaming layer adds, what it deliberately does
**not** add, and the evidence for both.

---

## 1. What this is — and what it is not

`StreamingDecision` is a **filter over the head's outputs**. It is not a second
head, not a second backbone, and not a second forward pass.

| Claim | True? |
|---|---|
| It makes the backbone incremental | **No — and it does not need to.** RWKV-7 here is the scalar *recurrent* form, so `h_t` is already a function of `(token_t, state_{t-1})` and the state already carries the history. The layer never reads or writes that state. |
| It makes the head more accurate | **No.** It makes the output *stable*, and stability and accuracy are different claims. No filter can recover information the head never had. |
| It invents a distribution or an action | **No.** Every action it emits came from a real head call. The filter only decides *when* to trust one. |
| It can be driven from the batched path | **Yes.** `push_batch()` runs `DecisionHead::decide_batch` (§36) and filters the results in order. |

The distinction matters because "streaming" in a language-model context usually
means token-by-token *decoding*, which is a different mechanism in a different
layer. Here the token head already streams; this is about the **decision**.

---

## 2. Three mechanisms

Each is a standard control-loop primitive, and each exists because of a specific
way a naive loop fails.

### 2.1 Debounce (N-of-M)

An action is **committed** only after `confirm_steps` consecutive agreeing
observations. A single blip cannot commit, and a blip *inside* a run resets it:

```
BUY  BUY  HOLD  BUY  BUY      ->  no commitment, at confirm_steps = 3
BUY  BUY  BUY                 ->  committed at step 3
```

### 2.2 Hysteresis

**Keeping** a committed action needs `min_confidence`. **Changing** it needs
`switch_confidence >= min_confidence`. That asymmetry is what stops a loop
oscillating across a noisy decision boundary.

A config with `switch_confidence < min_confidence` is normalised **up**, never
down. This is the only repair in `StreamingConfig::normalised()` that is not a
plain clamp, and it can only ever make the filter *more* conservative. The
invariant is tested in both directions (`test_streaming_decision.cpp` A6).

A challenger that fails the switch bar is **gated**, which is *not* the same as
being weak:

* it does not start a switch, and
* it does **not** count as a weak observation, so it cannot contribute to a
  release.

It *does* reset the incumbent's run of consecutive agreement, because the head
just said something else.

### 2.3 Release

A commitment whose evidence has evaporated is released to `ABSTAIN` after
`release_after_weak` consecutive weak observations. Silence is a reason to stop
acting; an unbounded hold would be a decision the head is no longer making.
`release_after_weak = 0` disables release, and the commitment is then held until
contradicted.

A release costs the **same evidence as a switch**: a quorum of `ABSTAIN`
observations. One strong `ABSTAIN` cannot unseat a commitment that took N
observations to earn.

---

## 3. Two modes, one state machine

Both modes reduce their observations to a single `Candidate`, and **one** state
machine consumes it. The commit / hysteresis / release rules are therefore
written once and tested once.

| | `Confirm` | `Window` |
|---|---|---|
| Order sensitivity | sensitive | insensitive |
| `strength` means | the current agreeing run | the winner's vote count |
| `confirm_steps` means | consecutive observations needed | the vote **quorum** |
| `window` means | unused | ring size (auto-grown to `>= confirm_steps`) |
| Answers | "is the head saying the same thing, clearly, N times running?" | "over the last W observations, which action won?" |
| Wants | a live loop | a backtest, evaluated every bar |

The window winner is chosen by **votes**, then by summed confidence, then by the
lower action index. Fully ordered, so a tie can never resolve differently on two
runs over the same data.

`StreamEvent::vote_margin` is `(winner − runner-up) / total` in `Window` mode and
is **0 in `Confirm` mode**. It is a vote margin, not a probability margin, and it
is never reported as one; the head's own probability margin travels separately as
`observed_margin`.

In `Window` mode an **indecisive window** (a winner below the quorum) is a weak
observation, so a commitment whose vote has fragmented is released by the same
rule that releases one whose confidence has collapsed. That is intentional.

---

## 4. The rule that is easy to get wrong: `ABSTAIN` is not a position

`ABSTAIN` means "no position". A committed filter is therefore **never** in that
state: `committed == true` implies a real action.

This is not a nicety. Without it, a strong run of `ABSTAIN` would "commit",
leaving a filter that reports `committed = true` while holding nothing, and a
`first` event with no `changed` event to match it. The identity

```
changed events  ==  commits + changes + releases
```

would stop holding. That identity is pinned over **10,000 random steps** in
Part D of the test suite, and it is what lets a consumer treat every `changed`
event as exactly one transition to undo.

A quorum of `ABSTAIN` is handled as what it actually is:

* **already committed** → a **release** ("the head is confidently saying: no
  action", which is not the same as saying nothing);
* **not committed** → a no-op.

---

## 5. Fail-closed

A head that is not ready, a null pointer, a width mismatch, or a `Tensor` of the
wrong dtype records **nothing**:

* `push()` returns `false`;
* the event says `valid = false` and carries the reason;
* `steps()` does **not** advance;
* the streak, the window and the commitment are **untouched**;
* `failures()` increments.

An unreadable signal is not agreement, and a dead head must never silently
confirm a decision. A failure **mid-stream** therefore cannot disturb a
commitment the stream has already earned — pinned by test C3.

`push_batch()` is stricter still: it clears `out` first and returns `false` with
`out` **empty** rather than a partial sequence, because a caller that ignores the
return value must not act on half a stream.

---

## 6. The joint with §36 (batched heads)

`push_batch()` is the seam between the two milestones: the batched readout
produces B `DecisionResult`s in one GEMM, and the filter walks them in order, so
the per-row head overhead is paid once for the whole sequence.

With the **Scalar** kernel, `decide_batch` is bit-identical to `decide` (§36), so
the batch-driven and per-row-driven filters produce the **same event stream** —
compared field by field *and* by `to_json()` string equality in test C1. That is
what makes one filter usable from either path.

The **AVX2** kernel is not bit-identical, and so the events it drives are not
guaranteed identical in the last bits of `observed_conf`. What is guaranteed is
that the *decisions* agree, which is what §36 measured.

---

## 7. Measured

Reproduce with `./build/bin/omniseed.exe demo-stream` (model-free) and
`./build/bin/omniseed_streaming_decision.exe`. The head is a **seeded
placeholder**, so the *actions* below are meaningless; what is being measured is
the **filter**.

### Churn removal — three streams

| stream | raw head flips | `Confirm`: committed changes | `Window`: committed changes |
|---|---|---|---|
| A — 240 real held-out `h[E]` | **0** | 1 (one commit, then held) | 1 |
| B — 240 synthetic, unscaled | 202 | **0** | **0** |
| C — the same, scaled ×8 | 205 | **2 → 102.5×** | **18 → 11.4×** |

Each stream is there because it says something different, and two of them say
something unflattering:

* **Stream A.** On real trading hidden states a *seeded* head emits **one
  constant action** — 1 distinct output, 0 flips, max p = 0.9875. There is
  nothing for the filter to remove. That is a fact about untrained weights, and
  the demo prints it rather than substituting a flattering stream.
* **Stream B.** The head is *confident of nothing* (max p = 0.176 over 240
  steps). Every observation is below the bar, so the filter commits **nothing at
  all**. This is the fail-safe working, not the filter failing — but a demo that
  showed only this would look like a filter that never does anything.
* **Stream C.** Scaling the same noise ×8 widens the logit spread, so the same
  untrained head becomes *confident while still changing its mind every step* —
  the situation the layer exists for. 205 raw flips become 2 committed changes.

Note what stream C's `Confirm` column does **not** show: `changes = 0`. The
transitions are all commit → release, never action → action, because the raw
stream is too noisy to produce three consecutive agreements on a *different*
action while one is held. That is hysteresis behaving correctly, and it is
visible rather than smoothed over.

### The identity, over a long stream

10,000 sticky random steps at `confirm_steps = 4`:

```
commits = 161   changes = 42   releases = 161
changed events = 364 = 161 + 42 + 161          <- exact
```

The generator is **sticky** on purpose. A purely uniform stream produced 2
commits and **0 switches** over the same 10,000 steps: it never entered the
switch branch, so the invariants would have been asserted over a stream that
only ever walked the "nothing happened" path. The test now **asserts that
commits, changes and releases are all > 0**, so it cannot pass by not trying.

### The joint with §36

Over **369 real held-out rows**, the batch-driven and per-row-driven filters
produce the same event stream — compared field by field *and* by `to_json()`
string equality.

---

## 8. What is built, and what is not

**Built and tested** — **861 checks, 0 failed, 0 skipped**, ungated, no model,
no `.venv`, no network:

* the `Confirm` state machine: debounce, blip reset, `confirm_steps = 1`;
* hysteresis in both directions, including the one-way config repair;
* gated vs weak, and the difference between them;
* release, its threshold, its disabling, and its reset on agreement;
* the `Window` vote: quorum, sliding, ties, order-insensitivity, resize;
* `ABSTAIN` never committing, and a strong `ABSTAIN` releasing;
* the batch/per-row equivalence over the committed held-out `h[E]` fixtures;
* fail-closed on a dead head, a null pointer and a width mismatch;
* JSON well-formedness and the absence of a bare `nan` / `inf`;
* the bookkeeping identities over 10,000 sticky random steps;
* determinism: the same input twice gives the same event stream twice.

**Not built — stated plainly:**

* **Nothing consumes it yet.** No trading loop, no agent loop and no backtest
  calls `StreamingDecision`. The CLI demo (`omniseed demo-stream`) is the only
  caller, and it is a demonstration, not an integration.
* **No CLI flags** for the mode or the thresholds; the demo hard-codes them.
* **The head is untrained**, so the demo's actions are placeholders. The filter
  is real; the judgement is not. `provenance()` says so at runtime.
* **No adaptive thresholds.** `confirm_steps`, the bars and the window are fixed
  at configuration time. A filter that tuned its own hysteresis from realised
  churn would be a natural next step and is not built.
* **No latency accounting.** The filter adds no measurable cost of its own — it
  is a handful of comparisons per step on top of a head call that already
  happens — but that has not been *measured* separately, and this document does
  not claim a number for it.

