#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — get_fixtures.py   (offline helper; fetches tiny ASR test clips)
#
#  Downloads the official DeepSpeech v0.9.3 "Test set samples" archive
#  (audio-0.9.3.tar.gz — three LibriSpeech clips, CC-BY 4.0, 16 kHz mono
#  16-bit PCM) into tests/fixtures/, keeping Attribution.txt + License.txt
#  alongside the audio, and writes manifest.txt consumed by tests/test_sides.cpp
#  to assert REAL end-to-end ASR behavior (expected transcription, non-empty
#  text, bounded runtime). The test skips cleanly when fixtures are absent.
#
#  Archive: https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/
#           audio-0.9.3.tar.gz  (~200 KB)
#  Clip 2830-3980-0043.wav is THE canonical DeepSpeech demo/test sentence —
#  its reference transcription is used as the expected text.
#
#  Usage:
#    python tools/get_fixtures.py              # download + install fixtures
#    python tools/get_fixtures.py --force      # re-download even if present
# =============================================================================
import argparse
import os
import shutil
import struct
import sys
import tarfile
import tempfile
import urllib.request

ARCHIVE_URL = ("https://github.com/mozilla/DeepSpeech/releases/download/"
               "v0.9.3/audio-0.9.3.tar.gz")
OUT_DIR = os.path.join("tests", "fixtures")
MANIFEST = os.path.join(OUT_DIR, "manifest.txt")

# LibriSpeech id → expected transcription. The value is the OFFICIAL
# openai/whisper-tiny greedy output for the clip (timestamps stripped),
# validated with transformers 5.x generate() (greedy == beam-5) and
# tools/dbg_whisper_ref.py. NOTE: it is NOT the clip's DeepSpeech reference
# label ("she had your dark suit in greasy wash water all year") — whisper-
# tiny transcribes this clip as below, and the test asserts parity with the
# official model, not the archive label.
EXPECTED_TEXT = {
    "2830-3980-0043.wav":
        "Experience proves this.",
}
KEEP_FILES = ("2830-3980-0043.wav", "4507-16021-0012.wav",
              "8455-210777-0068.wav", "Attribution.txt", "License.txt")


def read_wav_header(path):
    """Returns (sample_rate, channels, bits, num_samples) or None."""
    try:
        with open(path, "rb") as f:
            riff = f.read(12)
            if len(riff) < 12 or riff[:4] != b"RIFF" or riff[8:12] != b"WAVE":
                return None
            sr = ch = bits = None
            while True:
                hdr = f.read(8)
                if len(hdr) < 8:
                    return None
                cid, size = hdr[:4], struct.unpack("<I", hdr[4:])[0]
                if cid == b"fmt ":
                    fmt = f.read(size)
                    ch = struct.unpack("<H", fmt[2:4])[0]
                    sr = struct.unpack("<I", fmt[4:8])[0]
                    bits = struct.unpack("<H", fmt[14:16])[0]
                elif cid == b"data":
                    frame = max(1, (bits // 8) * max(1, ch))
                    return (sr, ch, bits, size // frame)
                else:
                    f.seek(size, 1)
    except OSError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="re-download even when the fixtures exist")
    args = ap.parse_args()

    marker = os.path.join(OUT_DIR, EXPECTED_TEXT and
                          "2830-3980-0043.wav" or "")
    if os.path.exists(marker) and not args.force:
        print(f"[fixtures] {marker} already present (--force to re-download)")
        return 0

    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="omniseed_fixtures_")
    archive = os.path.join(tmp, "audio.tar.gz")
    print(f"[fixtures] downloading {ARCHIVE_URL}")
    try:
        with urllib.request.urlopen(ARCHIVE_URL, timeout=120) as r:
            payload = r.read()
    except OSError as e:
        print(f"[fixtures] DOWNLOAD FAILED: {e}", file=sys.stderr)
        return 1
    with open(archive, "wb") as f:
        f.write(payload)
    print(f"[fixtures] archive: {len(payload):,} bytes")

    with tarfile.open(archive, "r:gz") as t:
        for member in t.getmembers():
            base = os.path.basename(member.name)
            if not base or base.startswith("._") or base not in KEEP_FILES:
                continue                      # skip AppleDouble / noise entries
            src = t.extractfile(member)
            if src is None:
                continue
            with open(os.path.join(OUT_DIR, base), "wb") as dst:
                dst.write(src.read())
            print(f"[fixtures] installed {base}")

    # ---- validate the anchor clip + write the test manifest -----------------
    wav_path = os.path.join(OUT_DIR, "2830-3980-0043.wav")
    hdr = read_wav_header(wav_path)
    if hdr is None:
        print("[fixtures] WAV header parse failed", file=sys.stderr)
        return 1
    sr, ch, bits, nsamp = hdr
    print(f"[fixtures] {wav_path}: {sr} Hz, {ch} ch, {bits} bit, "
          f"{nsamp} samples ({nsamp / sr:.2f} s)")
    if sr != 16000 or ch != 1 or bits != 16:
        print("[fixtures] ERROR: whisper-tiny expects 16 kHz mono s16",
              file=sys.stderr)
        return 1

    with open(MANIFEST, "w", encoding="utf-8") as f:
        f.write("# generated by tools/get_fixtures.py — consumed by "
                "tests/test_sides.cpp\n")
        for name, text in EXPECTED_TEXT.items():
            f.write(f"fixture={name}\n")
            f.write(f"expected_text={text}\n")
    print(f"[fixtures] manifest: {MANIFEST}")
    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
