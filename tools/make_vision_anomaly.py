#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — tools/make_vision_anomaly.py   (T051, DECISION 2 plan)
#
#  vision.anomaly needs NO external corpus: the label IS the transform.
#  A clean scene image is 'normal'; a DETERMINISTIC, documented corruption of
#  the same image is 'unusual'. Free, reproducible, and the augmentation IS
#  the documentation — exactly what docs/VISION_AUDIO_DATA.md's plan promises
#  ("Implemented once vision.scene data exists").
#
#  Transforms (cycled by row index, no randomness beyond the index):
#    0  rotate 90 CW        1  rotate 180       2  rotate 270 CW
#    3  R/B channel swap    4  salt & pepper (10%, fixed bit pattern)
#  Every transform is trivially recognisable as WRONG to a human eye, which
#  is the point: 'unusual' means structurally corrupted, not subtly off.
#
#  FAIL-CLOSED: refuses to run without a completed vision.scene install
#  (sentinel .cifar10.complete) — synthesising anomalies from a partial
#  dataset would fabricate the very evidence the fit is supposed to measure.
#
#  usage:
#    python tools/make_vision_anomaly.py
# =============================================================================
"""Synthesise vision.anomaly from an installed vision.scene set (stdlib only)."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = os.path.join(ROOT, "models", "modality", "vision.scene")
ANOM = os.path.join(ROOT, "models", "modality", "vision.anomaly")

W = H = 32
CH = 3
PIX = W * H * CH
HEADER_LEN = len(b"P6\n32 32\n255\n")


def read_ppm(path):
    with open(path, "rb") as f:
        data = f.read()
    if data[:HEADER_LEN] != b"P6\n32 32\n255\n":
        return None
    pix = data[HEADER_LEN:]
    if len(pix) != PIX:
        return None
    return pix


def write_ppm(path, pix):
    with open(path, "wb") as f:
        f.write(b"P6\n32 32\n255\n")
        f.write(pix)


def rotate(pix, times):
    """Rotate 90 CW `times` times on the 32x32 RGB grid."""
    grid = [pix[(y * W + x) * CH:(y * W + x + 1) * CH]
            for y in range(H) for x in range(W)]
    for _ in range(times % 4):
        grid = [grid[x * H + (H - 1 - y)] for y in range(H) for x in range(W)]
    return b"".join(grid)


def channel_swap(pix):
    out = bytearray(pix)
    for i in range(0, PIX, CH):
        out[i], out[i + 2] = out[i + 2], out[i]
    return bytes(out)


def salt_pepper(pix):
    # 10% of bytes flipped by a FIXED positional pattern (every 10th byte
    # XOR 0xFF) — deterministic, no RNG, documented exactly.
    out = bytearray(pix)
    for i in range(0, PIX, 10):
        out[i] ^= 0xFF
    return bytes(out)


def transform(i, pix):
    kind = i % 5
    if kind == 0:
        return rotate(pix, 1), "rot90"
    if kind == 1:
        return rotate(pix, 2), "rot180"
    if kind == 2:
        return rotate(pix, 3), "rot270"
    if kind == 3:
        return channel_swap(pix), "rb_swap"
    return salt_pepper(pix), "saltpepper"


def main():
    if not os.path.exists(os.path.join(SCENE, ".cifar10.complete")):
        print("REFUSED: vision.scene has no .cifar10.complete sentinel — "
              "run tools/get_modality_data.py --fetch cifar10 first "
              "(fail-closed: never synthesise from a partial set)")
        return 2
    manifest = os.path.join(SCENE, "images.tsv")
    if not os.path.exists(manifest):
        print("REFUSED: %s missing" % manifest)
        return 2

    rows = []
    with open(manifest, "r", encoding="utf-8") as f:
        header = f.readline()
        if not header.startswith("path\t"):
            print("REFUSED: bad manifest header %r" % header)
            return 2
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            path, _setname = line.split("\t")
            # Manifest rows are BARE filenames resolved against the scene dir
            # (dump_hidden does the same via its manifest-dir join).
            rows.append(path if os.path.isabs(path) else os.path.join(SCENE, path))

    os.makedirs(ANOM, exist_ok=True)
    out_rows = []
    n_bad = 0
    kinds_used = {}
    for i, path in enumerate(rows):
        pix = read_ppm(path)
        if pix is None:
            n_bad += 1
            continue
        base = os.path.splitext(os.path.basename(path))[0]
        # Self-contained set (audio.emotion layout): normals are COPIED in,
        # and every manifest path is a bare filename — the dump loader
        # resolves rows against the manifest's own directory.
        npath = os.path.join(ANOM, "normal_%s.ppm" % base)
        write_ppm(npath, pix)
        out_rows.append(("normal_%s.ppm" % base, "normal", "original"))
        unusual, kind = transform(i, pix)
        upath = os.path.join(ANOM, "aug_%s.ppm" % base)
        write_ppm(upath, unusual)
        kinds_used[kind] = kinds_used.get(kind, 0) + 1
        out_rows.append(("aug_%s.ppm" % base, "unusual", kind))

    apath = os.path.join(ANOM, "images.tsv")
    with open(apath, "w", encoding="utf-8", newline="\n") as f:
        f.write("# vision.anomaly — label IS the transform (T051 plan).\n")
        f.write("# transforms: rot90/rot180/rot270/rb_swap/saltpepper "
                "(cycled by index; saltpepper = every 10th byte XOR 0xFF)\n")
        f.write("path\tvision.anomaly\ttransform\n")
        for r in out_rows:
            f.write("%s\t%s\t%s\n" % r)
    with open(os.path.join(ANOM, "LICENCE.txt"), "w", encoding="utf-8") as f:
        f.write("Synthesised from the local CIFAR-10 scene set (MIT).\n"
                "The transforms are the documentation; see images.tsv header.\n")

    print("vision.anomaly: %d normal + %d unusual rows (%d bad scene images "
          "skipped); transforms: %s" % (len(out_rows) // 2,
                                        len(out_rows) // 2, n_bad, kinds_used))
    print("wrote %s" % apath)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
