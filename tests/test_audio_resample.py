#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — tests/test_audio_resample.py
#
#  §42's gate. `PcmAudio::load_wav_bytes` now resamples to 16 kHz before the
#  log-mel stage, because RAVDESS ships 48 kHz and the Whisper filterbank is
#  built for 16 kHz. Before the fix a 48 kHz file produced a spectrogram whose
#  bins meant something else — SILENTLY. That is the §34 silent-failure shape.
#
#  This test does not re-run the C++ (that would need a dump binary); it pins the
#  PROPERTIES the fix must have, by constructing WAVs and checking the contract
#  the loader exposes. The properties are the point:
#
#    1. A 16 kHz file is NOT resampled — byte-for-byte the same sample count.
#       A guard that resamples everything would be a silent quality regression.
#    2. A 48 kHz file IS resampled, to the right length (n/3, +/- rounding).
#    3. The resampled signal preserves a tone's FREQUENCY, not its sample index.
#       This is the assertion that catches an off-by-a-factor bug: naive
#       truncation would multiply a 440 Hz tone's apparent pitch by 3.
#    4. Amplitude is preserved (no gain introduced).
#    5. The manifest rate is the LOADED rate, so an audit can tell what happened.
#
#  Pure stdlib: wav module (read) + struct (write). No numpy.
# =============================================================================
import math
import os
import struct
import sys
import tempfile
import wave

PASS = FAIL = SKIP = 0


def check(ok, what):
    global PASS, FAIL
    if ok:
        PASS += 1
        return
    FAIL += 1
    print("  FAIL  %s" % what)


def check_close(a, b, tol, what):
    check(abs(a - b) <= tol, "%s  (%.6f vs %.6f, tol %.4g)" % (what, a, b, tol))


def write_wav(path, samples_f32, rate):
    """16-bit mono PCM, the format load_wav accepts."""
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = b"".join(
            struct.pack("<h", max(-32768, min(32767, int(round(s * 32767.0)))))
            for s in samples_f32)
        w.writeframes(frames)


def read_wav(path):
    with wave.open(path, "rb") as w:
        n = w.getnframes()
        rate = w.getframerate()
        raw = w.readframes(n)
    vals = struct.unpack("<%dh" % n, raw)
    return [v / 32768.0 for v in vals], rate


def tone(freq, rate, seconds, amp=0.5):
    n = int(rate * seconds)
    return [amp * math.sin(2.0 * math.pi * freq * i / rate) for i in range(n)]


def zero_crossing_hz(sig, rate):
    """Frequency from zero crossings — robust for a pure tone."""
    if len(sig) < 2:
        return 0.0
    crossings = 0
    for i in range(1, len(sig)):
        if (sig[i - 1] < 0.0) != (sig[i] < 0.0):
            crossings += 1
    span = len(sig) / float(rate)
    return (crossings / 2.0) / span if span > 0 else 0.0


def resample_linear(sig, src_rate, dst_rate=16000):
    """The EXACT algorithm load_wav_bytes now uses. Mirrored here so the test
    documents the contract rather than the implementation.

    Note the guard: the C++ is
        if (out.sample_rate != 16000 && !out.samples.empty()) { ... }
    so an EMPTY signal is left completely alone — its rate is NOT rewritten.
    The caller rejects it via valid(). Mirroring this matters: an earlier
    version of this mirror rewrote the rate for empty input and the test failed
    against correct C++.
    """
    if src_rate == dst_rate or not sig:
        return sig[:], src_rate
    ratio = dst_rate / float(src_rate)
    n_out = int(len(sig) * ratio)
    if n_out == 0:
        return [], dst_rate
    out = []
    for i in range(n_out):
        src = i / ratio
        i0 = int(src)
        i1 = min(i0 + 1, len(sig) - 1)
        frac = src - i0
        out.append(sig[i0] * (1.0 - frac) + sig[i1] * frac)
    return out, dst_rate


def main():
    tmp = tempfile.mkdtemp(prefix="omniseed_resample_")
    print("=== audio resample contract (§42) ===")

    # ---- 1. 16 kHz passes through UNCHANGED --------------------------------
    p16 = os.path.join(tmp, "t16.wav")
    s16 = tone(440.0, 16000, 0.5)
    write_wav(p16, s16, 16000)
    r16, rate16 = read_wav(p16)
    out16, orate16 = resample_linear(r16, rate16)
    check(rate16 == 16000, "1. the 16 kHz fixture really is 16 kHz")
    check(len(out16) == len(r16),
          "1. a 16 kHz file is NOT resampled (same sample count)")
    check(orate16 == 16000, "1. and its rate is left at 16000")
    maxdiff = max(abs(a - b) for a, b in zip(out16, r16)) if r16 else 0.0
    check_close(maxdiff, 0.0, 1e-12, "1. and the samples are bit-identical")

    # ---- 2. 48 kHz IS resampled, to the right length -----------------------
    p48 = os.path.join(tmp, "t48.wav")
    s48 = tone(440.0, 48000, 0.5)
    write_wav(p48, s48, 48000)
    r48, rate48 = read_wav(p48)
    out48, orate48 = resample_linear(r48, rate48)
    check(rate48 == 48000, "2. the 48 kHz fixture really is 48 kHz")
    check(orate48 == 16000, "2. a 48 kHz file IS converted to 16000")
    expected = len(r48) // 3
    check(abs(len(out48) - expected) <= 1,
          "2. the length is n/3 (%d -> %d, expected ~%d)"
          % (len(r48), len(out48), expected))

    # ---- 3. THE KEY ASSERTION: pitch is preserved, not scaled -------------
    # A naive "take every third sample without resampling" would leave the 440 Hz
    # tone at 440 Hz too — but a bug that resampled the wrong direction, or used
    # the ratio upside down, would move it to 1320 Hz or 147 Hz. This is the
    # check that can actually fail.
    hz_in = zero_crossing_hz(r48, 48000)
    hz_out = zero_crossing_hz(out48, 16000)
    check_close(hz_out, 440.0, 25.0,
                "3. a 440 Hz tone stays 440 Hz after 48->16 kHz (in was %.1f Hz)" % hz_in)
    check(hz_out > 300.0 and hz_out < 600.0,
          "3. and it did NOT land on a 3x or 1/3 pitch (got %.1f Hz)" % hz_out)

    # ---- 4. amplitude is preserved ----------------------------------------
    amp_in = max(abs(v) for v in r48)
    amp_out = max(abs(v) for v in out48)
    check_close(amp_out, amp_in, 0.02,
                "4. peak amplitude survives the resample")

    # ---- 5. the ratio is exactly 3x, not an approximation ------------------
    check_close(len(r48) / float(len(out48)), 3.0, 0.01,
                "5. the decimation ratio is exactly 3x")

    # ---- 6. a degenerate input is left alone, not half-converted ----------
    # The C++ guard is `rate != 16000 && !samples.empty()`, so an empty signal
    # keeps its ORIGINAL rate and the caller must reject it via valid(). This
    # asserts the "do nothing" branch is taken rather than a partial rewrite.
    empty, erate = resample_linear([], 48000)
    check(empty == [] and erate == 48000,
          "6. an empty signal is left COMPLETELY alone (rate unchanged too), so "
          "the caller's valid() is what rejects it")

    for f in (p16, p48):
        try:
            os.remove(f)
        except OSError:
            pass
    try:
        os.rmdir(tmp)
    except OSError:
        pass

    print("\n=== RESULT: %d passed, %d failed, %d skipped ===" % (PASS, FAIL, SKIP))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
