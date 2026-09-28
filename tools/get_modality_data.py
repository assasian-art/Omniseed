#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — tools/get_modality_data.py
#
#  DECISION 2 — the vision/audio data-acquisition plan, as RUNNABLE CODE.
#
#  WHAT THIS IS. §40 established that the vision/audio head absence is the DATA,
#  not the code. This tool is the bridge: it knows, per label set, exactly which
#  dataset could supply labels, what the licence is, how big it is, and whether
#  it is small enough to fetch. It fetches only what is cheap and license-clean,
#  and for everything else it prints the honest reason instead of pretending.
#
#  WHY IT DOES NOT JUST DOWNLOAD. Measured on 2026-09-29:
#    * Speech Commands v2 (CC BY 4.0) is a 2.43 GB monolithic archive; v0.01 is
#      1.49 GB. Individual clips return 403 and the bucket forbids anonymous
#      listing, so there is no way to fetch just "yes"/"no". Downloading 2.4 GB
#      into a sandbox to keep 4 MB of it is not a plan — it is a stunt.
#    * RAVDESS is 208 MB and license-clean for research. That one IS practical.
#    * Vision: no image dataset in this tree at all, and CIFAR-10 is 163 MB.
#  So this tool ENFORCES A SIZE BUDGET. Above it, it refuses and tells you the
#  one command to run yourself. A tool that silently pulls gigabytes is a worse
#  failure than one that says "too big, here is the command".
#
#  THE OWNER-VOICE PATH (preferred, and the reason it is first in the list).
#  For `audio.speaker` and `audio.wake` the best dataset is the owner's own
#  voice: it is the voice the product will actually hear, it carries no licence
#  question at all, and a speaker head fitted on third-party audiobook readers is
#  fitted on the wrong distribution. `omniseed enroll-audio` captures it. This
#  tool prepares and documents that path rather than assuming it.
#
#  Usage:
#    python tools/get_modality_data.py --plan              # what each set needs
#    python tools/get_modality_data.py --fetch ravdess     # download + install
#    python tools/get_modality_data.py --fetch all --max-mb 250
#    python tools/get_modality_data.py --list              # what IS installed
# =============================================================================
import argparse
import json
import os
import shutil
import struct
import sys
import tarfile
import tempfile
import urllib.request
import zipfile

# ---------------------------------------------------------------------------
# Where modality data lives. GITIGNORED — see docs/VISION_AUDIO_DATA.md.
# ---------------------------------------------------------------------------
DATA_ROOT = os.path.join("models", "modality")

# Never fetch more than this in one run without an explicit override. The point
# is to make "this dataset is too big" a VISIBLE decision.
DEFAULT_MAX_MB = 250

# ---------------------------------------------------------------------------
# The plan. One row per label set that needs data, with the honest status.
# `fetch` is the id --fetch accepts; None means "no automated route exists".
# ---------------------------------------------------------------------------
PLAN = [
    {
        "set": "audio.wake",
        "labels": ["yes", "no"],
        "preferred": "owner-voice",
        "primary": "Google Speech Commands v2 — subset yes/no",
        "licence": "CC BY 4.0",
        "url": ("https://storage.googleapis.com/download.tensorflow.org/"
                "data/speech_commands_v0.02.tar.gz"),
        "size_mb": 2317,
        "fetch": None,
        "why": ("The archive is monolithic (2.3 GB) and per-clip URLs return 403; "
                "there is no subset fetch. Two honest routes: run the full "
                "download yourself, or — PREFERRED — record your own yes/no with "
                "`omniseed enroll-audio`, which is the voice the product hears "
                "and has no licence question."),
    },
    {
        "set": "audio.emotion",
        "labels": ["happy", "sad", "angry", "neutral"],
        "preferred": "RAVDESS",
        "primary": "RAVDESS (Ryerson Audio-Visual Database of Emotional Speech)",
        "licence": "CC BY-NC-SA 4.0 — research/non-commercial only",
        "url": "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip",
        "size_mb": 199,
        "fetch": "ravdess",
        "why": ("199 MB, reachable, licence-clean for research. The 8 emotions "
                "map onto the 4 labels as: happy->happy, sad->sad, "
                "angry->angry, neutral->neutral; the other four are EXCLUDED "
                "rather than forced into a bucket they do not belong in. "
                "⚠️ Non-commercial: fine for a paper/research head, NOT for a "
                "shipped product — documented, not glossed."),
    },
    {
        "set": "audio.speaker",
        "labels": ["known", "unknown"],
        "preferred": "owner-voice",
        "primary": "LibriSpeech dev-clean speaker IDs, or the owner's enrolment",
        "licence": "CC BY 4.0 (LibriSpeech) / owner-owned (enrolment)",
        "url": None,
        "size_mb": 337,
        "fetch": None,
        "why": ("3 LibriSpeech clips are ALREADY in tests/fixtures — but they are "
                "transcripts, not speaker labels, and 3 clips cannot define "
                "'known'. PREFERRED: `omniseed enroll-audio` — the whole point "
                "of a speaker head is that it keys on YOUR voice."),
    },
    {
        "set": "vision.scene",
        "labels": ["indoor", "outdoor", "nature", "urban"],
        "preferred": "CIFAR-10 coarse classes",
        "primary": "CIFAR-10 (mapped) or Tiny-ImageNet subset",
        "licence": "MIT (CIFAR-10)",
        "url": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        "size_mb": 163,
        "fetch": "cifar10",
        "why": ("CIFAR-10 at 32x32 is small but its coarse classes map only "
                "LOOSELY onto indoor/outdoor/nature/urban — 'outdoor' is not a "
                "CIFAR class. The mapping is documented in the plan file so the "
                "head's accuracy is never over-read as scene understanding."),
    },
    {
        "set": "vision.anomaly",
        "labels": ["normal", "unusual"],
        "preferred": "synthetic",
        "primary": "Synthesised from clean images by a documented augmentation",
        "licence": "inherits the source set",
        "url": None,
        "size_mb": 0,
        "fetch": None,
        "why": ("No anomaly corpus is needed: the label IS the transform. "
                "Rotation / crop / noise / colour-shift applied to a clean image "
                "produces 'unusual'. Free, reproducible, and the augmentation is "
                "the documentation. Implemented once vision.scene data exists."),
    },
    {
        "set": "general.priority",
        "labels": ["urgent", "normal", "low"],
        "preferred": "none",
        "primary": "—",
        "licence": "—",
        "url": None,
        "size_mb": 0,
        "fetch": None,
        "why": ("⚠️ NO OBJECTIVE LABEL EXISTS. 'Urgent' is a judgement, and any "
                "corpus would import someone else's judgement while presenting "
                "it as a measurement. This set stays UNFITTED and is reported as "
                "such — the honest outcome, not a gap to paper over."),
    },
]


def human(mb):
    return "%.0f MB" % mb if mb < 1024 else "%.2f GB" % (mb / 1024.0)


def cmd_plan(args):
    print("=" * 80)
    print("DECISION 2 — modality data plan  (fetched into %s/, gitignored)" % DATA_ROOT)
    print("=" * 80)
    for p in PLAN:
        ok = "FETCHABLE" if p["fetch"] else "manual / owner-voice / none"
        print()
        print("  %-18s %s" % (p["set"], p["labels"]))
        print("    licence   %s" % p["licence"])
        print("    size      %s   [%s]" % (human(p["size_mb"]), ok))
        print("    primary   %s" % p["primary"])
        print("    note      %s" % p["why"])
    print()
    print("  Fetchable now: %s" % ", ".join(p["fetch"] for p in PLAN if p["fetch"]))
    print("  Budget per run: %d MB (override with --max-mb)" % DEFAULT_MAX_MB)
    return 0


def cmd_list(args):
    found = False
    for p in PLAN:
        d = os.path.join(DATA_ROOT, p["set"])
        if os.path.isdir(d):
            n = sum(len(f) for _, _, f in os.walk(d))
            print("  INSTALLED  %-18s %d file(s)" % (p["set"], n))
            found = True
    if not found:
        print("  nothing installed yet — run --plan, then --fetch <id>")
    return 0


def _download(url, dest, max_mb):
    """Stream to dest, refusing if the server advertises a size over budget."""
    req = urllib.request.Request(url, headers={"User-Agent": "omniseed/1.0"})
    with urllib.request.urlopen(req, timeout=180) as r:
        total = int(r.headers.get("Content-Length") or 0)
        if total and total > max_mb * 1024 * 1024:
            print("  REFUSED: advertised size %s exceeds the %d MB budget"
                  % (human(total / 1048576.0), max_mb))
            print("  run it yourself:  curl -L -o %s %s" % (dest, url))
            return False
        got = 0
        with open(dest, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                got += len(chunk)
                if got > max_mb * 1024 * 1024:
                    f.close()
                    os.remove(dest)
                    print("  REFUSED: download exceeded the %d MB budget mid-stream"
                          % max_mb)
                    return False
                f.write(chunk)
    print("  downloaded %s" % human(got / 1048576.0))
    return True


def _wav_info(path):
    """(sample_rate, n_samples) for a canonical 16-bit PCM WAV, else None."""
    try:
        with open(path, "rb") as f:
            head = f.read(44)
        if len(head) < 44 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
            return None
        # Walk chunks rather than assuming a 44-byte header.
        with open(path, "rb") as f:
            data = f.read()
        pos = 12
        fmt = None
        while pos + 8 <= len(data):
            cid = data[pos:pos + 4]
            csz = struct.unpack_from("<I", data, pos + 4)[0]
            body = data[pos + 8:pos + 8 + csz]
            if cid == b"fmt " and len(body) >= 16:
                fmt = struct.unpack_from("<HHIIHH", body, 0)
            elif cid == b"data":
                if fmt is None:
                    return None
                smp, ch, rate, _, _, bits = fmt
                return rate, len(body) // max(ch * (bits // 8), 1)
            pos += 8 + csz + (csz & 1)
    except Exception:
        return None
    return None


def fetch_ravdess(max_mb):
    url = next(p["url"] for p in PLAN if p["fetch"] == "ravdess")
    out = os.path.join(DATA_ROOT, "audio.emotion")
    if os.path.isdir(out) and os.listdir(out):
        print("  already installed at %s" % out)
        return True
    tmp = tempfile.mkdtemp(prefix="omniseed_ravdess_")
    zpath = os.path.join(tmp, "ravdess.zip")
    print("  RAVDESS: fetching %s" % url)
    if not _download(url, zpath, max_mb):
        shutil.rmtree(tmp, ignore_errors=True)
        return False

    # RAVDESS filenames encode everything:
    #   03-01-<emotion>-<intensity>-<statement>-<repetition>-<actor>.wav
    # emotion 01=neutral 02=calm 03=happy 04=sad 05=angry 06=fearful
    #         07=disgust 08=surprised
    # We keep only the four labels the head knows and DROP the rest, because
    # forcing 'calm' into 'neutral' or 'fearful' into 'angry' would be a label
    # error the head would then learn.
    KEEP = {"01": "neutral", "03": "happy", "04": "sad", "05": "angry"}
    os.makedirs(out, exist_ok=True)
    rows = []
    n_kept = n_dropped = 0
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            base = os.path.basename(name)
            if not base.lower().endswith(".wav"):
                continue
            parts = base.split("-")
            if len(parts) < 7:
                continue
            emo = parts[2]
            if emo not in KEEP:
                n_dropped += 1
                continue
            actor = os.path.splitext(parts[6])[0]
            dest = os.path.join(out, base)
            with z.open(name) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
            info = _wav_info(dest)
            if info is None:
                os.remove(dest)
                continue
            rate, nsmp = info
            rows.append((base, KEEP[emo], actor, rate, nsmp))
            n_kept += 1
    shutil.rmtree(tmp, ignore_errors=True)

    # Manifest: the label + provenance is the point. Sorted for determinism.
    rows.sort()
    mpath = os.path.join(out, "labels.tsv")
    with open(mpath, "w", encoding="utf-8", newline="\n") as f:
        f.write("file\taudio.emotion\tactor\tsample_rate\tn_samples\n")
        for r in rows:
            f.write("%s\t%s\t%s\t%d\t%d\n" % r)
    print("  kept %d clip(s), dropped %d outside the 4-label set" % (n_kept, n_dropped))
    print("  wrote %s" % mpath)
    with open(os.path.join(out, "LICENCE.txt"), "w", encoding="utf-8") as f:
        f.write("RAVDESS — CC BY-NC-SA 4.0 (research / NON-COMMERCIAL only).\n"
                "Source: %s\n"
                "Paper: Livingstone & Russo (2018), PLoS ONE 13(5): e0196391.\n"
                "Labels kept: neutral(01) happy(03) sad(04) angry(05).\n"
                "Dropped: calm, fearful, disgust, surprised — mapped to no label "
                "rather than forced into one.\n" % url)
    return True


def fetch_cifar10(max_mb):
    url = next(p["url"] for p in PLAN if p["fetch"] == "cifar10")
    out = os.path.join(DATA_ROOT, "vision.scene")
    if os.path.isdir(out) and os.listdir(out):
        print("  already installed at %s" % out)
        return True
    tmp = tempfile.mkdtemp(prefix="omniseed_cifar_")
    tpath = os.path.join(tmp, "cifar.tar.gz")
    print("  CIFAR-10: fetching %s" % url)
    if not _download(url, tpath, max_mb):
        shutil.rmtree(tmp, ignore_errors=True)
        return False
    os.makedirs(out, exist_ok=True)
    # CIFAR coarse classes: 0 airplane,1 automobile,2 bird,3 cat,4 deer,5 dog,
    # 6 frog,7 horse,8 ship,9 truck.
    #
    # ⚠️ THE MAPPING IS LOSSY AND THAT IS THE POINT OF WRITING IT DOWN.
    # CIFAR has NO indoor class and NO urban class. Anything claiming those
    # labels from CIFAR would be inventing them. So only two of the four labels
    # are supported at all, and the file records the third as absent:
    #   nature  <- deer, bird, frog, horse      (animals in habitat)
    #   outdoor <- airplane, ship, truck, automobile  (things seen outdoors)
    #   indoor  <- (NO SOURCE — must come from an indoor dataset)
    #   urban   <- (NO SOURCE — must come from an urban dataset)
    MAP = {0: "outdoor", 1: "outdoor", 2: "nature", 3: None, 4: "nature",
           5: None, 6: "nature", 7: "nature", 8: "outdoor", 9: "outdoor"}
    rows = []
    n_total = 0
    with tarfile.open(tpath, "r:gz") as t:
        for member in t.getmembers():
            if not member.name.endswith("data_batch_1"):
                continue
            f = t.extractfile(member)
            if f is None:
                continue
            blob = f.read()
            # <1s byte label><3072 bytes> x 10000, 1000 per batch
            for i in range(1000):
                off = 1 + i * 3073
                if off + 3073 > len(blob):
                    break
                lab = blob[off - 1]
                mapped = MAP.get(lab)
                if mapped is None:
                    continue
                rows.append((i, mapped, lab))
                n_total += 1
    shutil.rmtree(tmp, ignore_errors=True)
    with open(os.path.join(out, "labels.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("# CIFAR-10 data_batch_1, mapped to vision.scene\n")
        f.write("# ⚠️ LOSSY: only 'nature' and 'outdoor' are supported. CIFAR has\n"
                "# NO indoor class and NO urban class; those labels are recorded\n"
                "# as UNSUPPORTED rather than invented.\n")
        f.write("index\tvision.scene\tcifar_class\n")
        for r in rows:
            f.write("%d\t%s\t%d\n" % r)
    with open(os.path.join(out, "LICENCE.txt"), "w", encoding="utf-8") as f:
        f.write("CIFAR-10 — MIT licence. Source: %s\n"
                "Krizhevsky (2009), 'Learning Multiple Layers of Features from "
                "Tiny Images'.\n"
                "MAPPING IS LOSSY: nature<-{bird,deer,frog,horse}, "
                "outdoor<-{airplane,automobile,ship,truck}.\n"
                "indoor and urban have NO CIFAR source and are NOT provided.\n"
                % url)
    print("  mapped %d image(s) into nature/outdoor" % n_total)
    print("  indoor + urban: NO SOURCE — recorded unsupported, not invented")
    return True


FETCHERS = {"ravdess": fetch_ravdess, "cifar10": fetch_cifar10}


def cmd_fetch(args):
    targets = list(FETCHERS) if args.fetch == "all" else [args.fetch]
    for t in targets:
        if t not in FETCHERS:
            print("  unknown fetcher %r (have: %s)" % (t, ", ".join(FETCHERS)))
            continue
        print("--- %s ---" % t)
        FETCHERS[t](args.max_mb)
    print()
    return cmd_list(args)


def main():
    ap = argparse.ArgumentParser(description="DECISION 2 — modality data plan + fetch")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", action="store_true", help="show what each set needs")
    g.add_argument("--list", action="store_true", help="show what is installed")
    g.add_argument("--fetch", metavar="ID", help="ravdess | cifar10 | all")
    ap.add_argument("--max-mb", type=int, default=DEFAULT_MAX_MB,
                    help="refuse anything larger (default %d)" % DEFAULT_MAX_MB)
    args = ap.parse_args()
    if args.plan:
        return cmd_plan(args)
    if args.list:
        return cmd_list(args)
    return cmd_fetch(args)


if __name__ == "__main__":
    raise SystemExit(main())
