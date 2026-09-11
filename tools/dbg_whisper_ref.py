#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — dbg_whisper_ref.py  (offline debug oracle; not part of the build)
#
#  Staged reference for the whisper-tiny ASR port, using the OFFICIAL HF
#  implementation (torch, local models/whisper-tiny.safetensors — no hub):
#    stage mel : whisper-exact log-mel (30 s pad + reflect + torch.stft) ->
#                dumps C:/tmp/mel_ref.bin, optional --compare-mel vs the C++ dump
#    stage enc : WhisperModel encoder forward -> C:/tmp/enc_ref.bin (+ stats)
#    stage gen : HF generate() with the same forced prompt the C++ decoder
#                uses -> prints token ids + text (the expected transcription)
#
#  Usage:
#    ./.venv/Scripts/python.exe tools/dbg_whisper_ref.py --stage all
#    ./.venv/Scripts/python.exe tools/dbg_whisper_ref.py --stage mel \
#        --compare-mel C:/tmp/mel_cpp.bin
# =============================================================================
import argparse
import struct
import sys
import wave

import numpy as np
import torch
from safetensors.torch import load_file

WAV = "tests/fixtures/2830-3980-0043.wav"
CKPT = "models/whisper-tiny.safetensors"
FILTERS = "models/mel_filters.npz"

N_FFT, HOP, SR = 400, 160, 16000
N_SAMPLES = SR * 30                      # whisper pads/truncates to 30 s


def load_wav(path):
    with wave.open(path, "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 \
            and w.getsampwidth() == 2, "expect 16 kHz mono s16"
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return pcm.astype(np.float32) / 32768.0


def whisper_mel(audio):
    """Exactly transformers' WhisperFeatureExtractor.__call__ (pad 30 s,
    torch.stft center=True [zero pad] — for short clips the zero tail makes
    any edge-pad convention numerically identical, official mel_filters.npz,
    log10 + clamp + shift)."""
    buf = np.zeros(N_SAMPLES, dtype=np.float32)
    n = min(len(audio), N_SAMPLES)
    buf[:n] = audio[:n]
    x = torch.from_numpy(buf)                           # [480000]
    window = torch.hann_window(N_FFT)                   # periodic, like HF
    spec = torch.stft(x, N_FFT, HOP, window=window,
                      return_complex=True).abs() ** 2   # [201, 3001]
    filters = torch.from_numpy(np.load(FILTERS)["mel_80"])       # [80, 201]
    mel = filters @ spec                                # [80, 3001]
    mel = mel[:, :-1]                                   # HF drops the last frame
    log_spec = torch.log10(torch.clamp(mel, min=1e-10))
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec.numpy().astype(np.float32)          # [80, 3000]


def load_hf():
    from transformers import WhisperConfig, WhisperForConditionalGeneration
    cfg = WhisperConfig.from_pretrained("openai/whisper-tiny")
    model = WhisperForConditionalGeneration(cfg)
    sd = load_file(CKPT)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    tied = {k for k in missing if k.endswith("proj_out.weight")}
    real_missing = [k for k in missing if k not in tied]
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    assert not real_missing, f"missing keys: {real_missing[:5]}"
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["mel", "enc", "gen", "all"])
    ap.add_argument("--compare-mel", default=None,
                    help="C++ mel dump (f32 raw) to diff against")
    ap.add_argument("--slice-mel", type=int, default=None,
                    help="use only the first N mel frames (match a trimmed "
                         "C++ run) and dump enc_ref_slice.bin")
    args = ap.parse_args()

    audio = load_wav(WAV)
    print(f"[ref] {WAV}: {len(audio)/SR:.2f}s")

    mel = None
    if args.stage in ("mel", "all", "enc", "gen"):
        mel = whisper_mel(audio)
        np.ascontiguousarray(mel).tofile("C:/tmp/mel_ref.bin")
        print(f"[ref] mel {mel.shape} mean {mel.mean():.4f} "
              f"min {mel.min():.4f} max {mel.max():.4f} -> C:/tmp/mel_ref.bin")
        if args.compare_mel:
            cpp = np.fromfile(args.compare_mel, dtype="<f4")
            assert cpp.size == mel.size, \
                f"size mismatch: cpp {cpp.size} vs ref {mel.size}"
            d = np.abs(cpp - mel.ravel())
            print(f"[cmp] mel: max|diff| {d.max():.3e} mean|diff| {d.mean():.3e}")
        if args.stage == "mel":
            return

    if args.slice_mel:
        mel = np.ascontiguousarray(mel[:, :args.slice_mel])
        print(f"[ref] mel sliced to {mel.shape}")

    if args.stage in ("enc", "all"):
        model = load_hf()
        enc_mod = model.model.encoder
        with torch.no_grad():
            if args.slice_mel:
                # bypass the 3000-length gate: conv stem + pos embed + layers
                x = torch.from_numpy(mel)[None].float()
                h = enc_mod.conv1(x)
                h = torch.nn.functional.gelu(h)
                h = enc_mod.conv2(h)
                h = torch.nn.functional.gelu(h)
                h = h.squeeze(0).transpose(0, 1)        # [T/2, 384]
                t = h.shape[0]
                h = h + enc_mod.embed_positions.weight[:t]
                h = h[None]                             # [1, T/2, 384]
                for layer in enc_mod.layers:
                    h = layer(h, attention_mask=None)   # 5.16: tensor out
                out = enc_mod.layer_norm(h)
            else:
                out = enc_mod(
                    torch.from_numpy(mel)[None].float()).last_hidden_state
        enc = out[0].numpy().astype(np.float32)         # [T/2, 384]
        name = "C:/tmp/enc_ref_slice.bin" if args.slice_mel \
            else "C:/tmp/enc_ref.bin"
        np.ascontiguousarray(enc).tofile(name)
        print(f"[ref] enc {enc.shape} mean {enc.mean():.4f} "
              f"min {enc.min():.4f} max {enc.max():.4f} -> {name}")
        if args.stage == "enc":
            return

    if args.stage in ("gen", "all"):
        model = load_hf()
        enc_mod = model.model.encoder
        with torch.no_grad():
            out = enc_mod(torch.from_numpy(mel)[None].float()) \
                .last_hidden_state
        enc = out[0].numpy().astype(np.float32)
        dec = model.model.decoder
        proj_out = model.proj_out
        # default HF prompt: SOT transcribe en (timestamps ENABLED — no
        # <|notimestamps|>; the timestamp tokens suppress hallucination loops)
        toks = [50258, 50359, 50259]
        with torch.no_grad():
            for _ in range(96):
                h = dec(torch.tensor([toks]),
                        encoder_hidden_states=torch.from_numpy(
                            enc)[None]).last_hidden_state
                logits = proj_out(h[0, -1])
                nid = int(logits.argmax())
                toks.append(nid)
                if nid == 50257:                      # <|endoftext|>
                    break
        print(f"[ref] gen ids ({len(toks)}): {toks}")
        print(f"[ref] timestamps: "
              f"{[t for t in toks if 50364 <= t < 51866]}")
        try:
            from transformers import WhisperTokenizer
            tok = WhisperTokenizer(
                vocab_file="models/whisper-tiny-tokenizer.json")
            print(f"[ref] text: {tok.decode(toks, skip_special_tokens=True)!r}")
        except Exception as e:                          # ids are enough
            print(f"[ref] (tokenizer unavailable: {e})")
        return


if __name__ == "__main__":
    sys.exit(main())
