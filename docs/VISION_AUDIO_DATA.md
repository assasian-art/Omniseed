# VISION / AUDIO DATA — DECISION 2

**Date:** 2026-09-29
**Status:** plan implemented (`tools/get_modality_data.py`); RAVDESS + CIFAR-10
fetchable under a size budget; `audio.wake` and `audio.speaker` default to the
owner's own voice; `general.priority` has no honest source at all.

§40 established the shape of the problem: the vision/audio head absence is the
**DATA**, not the code — the joint (`MultimodalBridge`), the encoders and the
label sets all exist. This document is the acquisition plan, and it is written to
be read *before* anything is downloaded, because the interesting decisions are
about what we refuse to fake.

---

## 1. The rule this document follows

> A head fitted on a corpus whose labels do not mean what the head's label says
> is worse than an unfitted head. The unfitted head announces itself
> (`trained() == false`); the badly-fitted one looks trained and is wrong.

So every entry below states **what the labels actually are**, not what we wish
they were. Where a dataset cannot supply a label, the label is recorded as
**unsupported** — never synthesised from a near-miss.

---

## 2. Where the data lives

`models/modality/<label-set>/`, **gitignored** by the existing `models/*` rule
(`.gitignore:17`). Nothing under it is committed: datasets are hundreds of MB and
their licences do not always permit redistribution. What *is* committed is this
document, the tool, and the per-dataset `LICENCE.txt` + `labels.tsv` the tool
writes alongside the data (so a recipient of a fresh clone can reproduce exactly
what the head was fitted on).

```
models/modality/
  audio.emotion/   RAVDESS, 4 of the 8 emotions kept
  vision.scene/    CIFAR-10, 2 of the 4 labels supported
  (gitignored — see .gitignore:17)
```

---

## 3. The plan, set by set

### 3.1 `audio.wake` — {yes, no} — **owner voice preferred**

| | |
|---|---|
| primary | Google Speech Commands v2 (CC BY 4.0) |
| size | **2.26 GB** — a monolithic archive |
| verdict | **not auto-fetched** |

Measured 2026-09-29: per-clip URLs under
`storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02/…`
return **403**, and the bucket forbids anonymous object listing. There is
therefore **no way to fetch just `yes`/`no`**. The only automated route is the
full 2.26 GB archive, of which we would keep a few MB.

Downloading 2.26 GB to keep 4 MB is not a plan. Two honest routes:

1. **The owner's own voice (preferred).** `omniseed enroll-audio` builds a
   yes/no dataset from WAVs you record. This is not a consolation prize: a wake
   head keys on *the voice that will actually trigger it*, and a head fitted on
   3,000 strangers from a 2017 Kaggle competition is fitted on the wrong
   distribution. No licence question exists for your own voice.

   ```bash
   # record with any tool; any sample rate (it is resampled, and says so)
   # name each file <label>_<anything>.wav  ->  yes_01.wav, no_01.wav, ...
   omniseed enroll-audio models/modality/audio.wake audio.wake
   ```

   **Why files and not a microphone.** There is no capture code in this tree and
   adding WASAPI would be a platform dependency the project has deliberately
   avoided (`wsl.exe`/`cmd.exe` are blacklisted; the build is plain MSVC +
   CMake). More importantly a file-based enrolment is **reproducible**: the exact
   WAVs are named in `labels.tsv`, so the fit re-runs byte-identically. A
   microphone path would make the dataset a side effect of one afternoon's room
   noise.

   Unreadable or misnamed files are **reported and skipped**, never folded into a
   label — a mislabelled enrolment sample is a permanently wrong head.

2. **The full archive**, run by hand:
   ```bash
   curl -L -o /tmp/sc2.tgz \
     https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz
   tar xzf /tmp/sc2.tgz -C models/modality/audio.wake
   ```

### 3.2 `audio.emotion` — {happy, sad, angry, neutral} — **FETCHABLE**

| | |
|---|---|
| primary | RAVDESS (Livingstone & Russo 2018) |
| licence | **CC BY-NC-SA 4.0 — research / non-commercial** |
| size | **199 MB** |
| command | `python tools/get_modality_data.py --fetch ravdess` |

RAVDESS ships 8 emotions; the head knows 4. The mapping keeps `neutral(01)`,
`happy(03)`, `sad(04)`, `angry(05)` and **drops** `calm`, `fearful`, `disgust`,
`surprised`. Folding `calm` into `neutral` or `fearful` into `angry` would be a
label error the head would faithfully learn, so the four unmapped emotions are
excluded and the count of dropped clips is printed.

**Measured on the download (2026-09-29):** 672 clips kept, 768 dropped, 24 actors.

```
  angry    192      happy    192      sad      192      neutral   96
```

Two properties of the real data that a fitter must know, both measured rather
than assumed:

- **Class imbalance 2:2:2:1.** RAVDESS records *two* statements per emotion but
  only *one* neutral, so `neutral` has half the rows. A fitter that ignores this
  will over-predict neutral's neighbours; the label distribution is printed here
  so nobody is surprised by a per-class recall table later.
- **48 kHz, not 16 kHz.** Every clip is 48 kHz mono (recorded as such), while the
  Whisper encoder in this tree expects 16 kHz. **Resampling is a required
  preprocessing step**, not an optimisation — feeding 48 kHz samples to a 16 kHz
  encoder changes the effective pitch by 3× and the head would learn noise. The
  dump mode must resample, and this line is the reason.

⚠️ **Non-commercial licence.** A head fitted on RAVDESS is fine for a
research/paper artefact. It is **not** shippable in a commercial product. This is
recorded in the generated `LICENCE.txt` and here, not glossed.

### 3.3 `audio.speaker` — {known, unknown} — **owner voice preferred**

| | |
|---|---|
| primary | LibriSpeech dev-clean speaker IDs, or the owner's enrolment |
| in tree | 3 LibriSpeech clips (`tests/fixtures/*.wav`) |
| verdict | **not auto-fetched** |

The 3 clips already in the tree are **ASR fixtures** — `manifest.txt` pairs them
with `expected_text`, i.e. they label *what was said*, not *who said it*. Three
unlabelled speakers also cannot define "known": with n=3 the concept is vacuous.

**Preferred: `omniseed enroll-audio`.** The entire purpose of a speaker head is
to key on one specific voice. Downloading audiobook readers to classify "known"
would fit the head to a population that will never be in front of it.

```bash
omniseed enroll-audio models/modality/audio.speaker audio.speaker
# known_01.wav, known_02.wav, ...   (you, several sessions / distances / mics)
# unknown_*.wav would need a SECOND speaker — see the honest gap below.
```

⚠️ **The `unknown` half is genuinely hard and is not solved here.** "Known" is
easy to enrol; "unknown" requires recordings of people who are *not* you, which
means either a public corpus (licence + distribution mismatch) or asking other
people to record themselves. A speaker head fitted on `known` alone will report
`unknown` at whatever the base rate implies and must be **reported as
partial**, not presented as a working verification.

### 3.4 `vision.scene` — {indoor, outdoor, nature, urban} — **PARTIALLY FETCHABLE**

| | |
|---|---|
| primary | CIFAR-10 (MIT), coarse classes |
| size | **163 MB** |
| command | `python tools/get_modality_data.py --fetch cifar10` |
| status | ⚠️ **fetch ATTEMPTED and ABANDONED — mirror is throttled** |

**Measured 2026-09-29:** the download ran for **23 minutes without completing**
163 MB from `cs.toronto.edu` and was stopped. This is a network-path problem, not
a code one — the same tool fetched 199 MB from Zenodo in 4m33s. The tool refuses
before *starting* if a size is oversize, but it cannot detect a stall, which is a
**known gap** in it: there is no timeout, so a throttled mirror hangs forever.

The honest state: **`vision.scene` has no data in the tree.** The fetch command
above is untested against the real archive because the transfer never finished;
if the mirror behaves on your connection it should work as written, but that is
an expectation, not a measurement, and this line exists so nobody reads it as one.

**⚠️ The mapping is lossy, and that is the most important sentence here.**
CIFAR-10 has ten coarse classes and **none of them is `indoor` or `urban`.** The
tool maps:

| CIFAR class | → label |
|---|---|
| airplane, automobile, ship, truck | `outdoor` |
| bird, deer, frog, horse | `nature` |
| cat, dog | *(dropped — a pet is neither nature nor outdoor)* |
| — | **`indoor`: NO SOURCE** |
| — | **`urban`: NO SOURCE** |

So a head fitted on this data supports **two** of its four labels. `indoor` and
`urban` are recorded as **unsupported** in the generated `labels.tsv` header. A
`vision.scene` accuracy number from this corpus must never be read as four-class
scene understanding — it is a two-class nature/outdoor probe, and the doc says so.

`Tiny-ImageNet` is the alternative named in the directive; it is ~240 MB for the
validation split and its classes (`n0…`) map no better. Deferred, not adopted.

### 3.5 `vision.anomaly` — {normal, unusual} — **synthetic, no corpus needed**

The label **is** the transform: applying rotation / crop / noise / colour-shift to
a clean image produces `unusual`; leaving it alone produces `normal`. No download,
free to reproduce, and the augmentation parameters *are* the documentation. This
is implemented once `vision.scene` images exist.

Honest caveat: a head trained this way learns **the specific transforms used**,
not "anomaly" in any general sense. Its accuracy measures transform detection.
That is stated in the generated manifest when it is built, so the number is not
over-read later.

### 3.6 `general.priority` — {urgent, normal, low} — **NO SOURCE, and none should be invented**

| | |
|---|---|
| verdict | **stays unfitted, permanently** |

"Urgent" is a judgement about a situation, not a property of a corpus. Any
dataset that carried the label would import *someone else's* notion of urgency
and present it as a measurement. Fitting this head would mean manufacturing a
label and then reporting the model's agreement with our own fabrication.

**This set stays `trained() == false`, and §35's audit will report it as
unfitted.** That is the correct outcome, not a gap to close.

---

## 4. Summary table

| label set | source | licence | size | route |
|---|---|---|---|---|
| `audio.wake` | Speech Commands v2 | CC BY 4.0 | 2.26 GB | owner voice, or manual |
| `audio.emotion` | RAVDESS | CC BY-NC-SA (research) | 199 MB | **fetchable** |
| `audio.speaker` | enrolment | owner-owned | — | owner voice |
| `vision.scene` | CIFAR-10 | MIT | 163 MB | **fetchable (2 of 4 labels)** |
| `vision.anomaly` | synthetic | inherits | 0 | generated |
| `general.priority` | — | — | — | **none — stays unfitted** |
| `language.task` | in-tree text | — | 0 | derivable now |
| `general.routing` | other sets' labels | — | 0 | derivable once they exist |

---

## 5. Reproducing

```bash
python tools/get_modality_data.py --plan      # what each set needs, and why
python tools/get_modality_data.py --fetch ravdess cifar10
python tools/get_modality_data.py --list      # what is actually installed

# Turn fetched media into h[E] rows via the production fusion path (§43).
# NOTE: this is the SEPARATE binary omniseed_dump_hidden.exe (not omniseed.exe),
# and the model path is a POSITIONAL 2nd argument.
./build/bin/omniseed_dump_hidden.exe vision \
    models/rwkv7-0.1B-ternary.gguf <dir>/labels.tsv build/dump_vision
./build/bin/omniseed_dump_hidden.exe audio \
    models/rwkv7-0.1B-ternary.gguf <dir>/labels.tsv build/dump_audio --codec focal
# --codec mel is REFUSED: no audio->E adapter exists (see §6 item 4).
```

A single run refuses anything over `--max-mb` (default 250) **even mid-stream**,
and prints the `curl` command to run by hand instead. A tool that silently pulls
gigabytes on someone's metered connection is a worse failure than one that says
"too big".

---

## 6. What still has to be built after this

1. **`omniseed enroll-audio` — DONE (§42).** Builds an owner-voice dataset from
   a directory of WAVs, writes `labels.tsv`, and reports/skips bad files instead
   of absorbing them. **Preferred over every download above** for `audio.wake`
   and `audio.speaker`.
2. **Resampling — DONE (§42), and it was a real bug.** Measured: RAVDESS ships
   **48 kHz** clips while `WhisperTiny`'s log-mel filterbank is built for
   **16 kHz**. `PcmAudio::load_wav` faithfully recorded the file's rate but
   nothing converted it, so a 48 kHz file would have produced a spectrogram whose
   bins mean something else — silently, with no error. That is the §34
   silent-failure shape. `load_wav_bytes` now resamples to 16 kHz and **logs the
   conversion**, so the rate is auditable rather than assumed.
3. **`vision` / `audio` modes in `tools/dump_hidden.cpp` — DONE (§43).** Both route
   through `MultimodalBridge`, so the dumped `h[E]` is the *same* `h[E]` the
   runtime produces — not a re-derivation (§33's joint exists precisely for this).
   `vision` was proven end-to-end on PPMs (`n=2 E=768`, two distinct finite rows);
   `audio --codec focal` on all 12 RAVDESS clips (**`rate=16000`**, proving §42's
   resampler fired, 12 distinct balanced rows). PNG/JPG are refused by an
   extension gate rather than half-decoded.
4. ⚠️ **`audio --codec mel` is BLOCKED by a missing adapter (§43) — this is the
   real obstacle to `audio.emotion`, not the data.** `WhisperTiny::encode` emits
   `[T/2, 384]` but this backbone's `E = 768`, and there is **no audio→E
   projection in the tree** — vision has `models/vision-proj.gguf`, audio has
   nothing equivalent. `dump_hidden` therefore **refuses** `--codec mel` with the
   reason, and records `"no_audio_projection": "TRUE"` in the dump's `meta.json`.
   672 RAVDESS clips are on disk and ready; the fit waits on the adapter.
   Zero-padding or a random projection would fit a head to a fabrication and was
   rejected. Pinned by `tests/test_modality_dump.cpp`.
5. Fitting `language.task` + `general.routing`, which need no external data.
   (**Open:** `language.task` has zero labelled rows anywhere and no consumer —
   see `docs/CALIBRATION.md` §6; the honest move is to leave it unfitted.)
6. Re-running §35's uncertainty audit so `provenance()` flips to **TRAINED** for
   each set that now has data — and stays visibly **unfitted** for the rest. Note
   this is **blocked** for `audio.*` (no adapter, and wake/speaker need the owner's
   voice) and for `vision.*` (no corpus, and disk is at ~0.6 GB free).
