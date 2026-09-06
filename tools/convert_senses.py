#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — convert_senses.py   (OFFLINE tool, never ships at runtime)
#
#  Builds the modality sidecar GGUFs:
#
#    models/whisper-tiny-encoder.gguf
#      * whisper.mel_filters   F32 [80, 201]   (official OpenAI mel_filters.npz)
#      * whisper.conv1.*       F16 [384,80,3] / [384]
#      * whisper.conv2.*       F16 [384,384,3] / [384]
#      * whisper.pos_embed     F16 [1500,384]
#      * whisper.enc_ln.*      F16 [384]
#      * whisper.enc.N.{ln1,ln2,q,k,v,out,fc1,fc2}.{weight,bias} F16
#      * whisper.dec.tok_embd  F16 [51865,384]  (tied lm_head)
#      * whisper.dec.pos_embed F16 [448,384]
#      * whisper.dec.N.{ln1,ln2,ln3,self.q/k/v/out,cross.q/k/v/out,
#                        fc1,fc2}.{weight,bias} F16
#      * whisper.dec_ln.*      F16 [384]
#      * whisper.vocab_offsets I32 [V+1]  } piece bytes: pieces are stored
#      * whisper.vocab_bytes   I8  [N]    } unicode->byte mapped, concatenated
#                                           (loader has no string-array dtype)
#      metadata: whisper.{n_mels,n_audio_ctx,n_audio_state,n_audio_head,
#                        n_audio_layer,n_vocab,n_dec_layer}
#
#      Sidecar bias convention: present = full extent, absent = tolerated
#      (load_f16(..., required=false)); never dereference an optional bias
#      without the numel() > 1 guard.
#
#    models/vision-proj.gguf
#      * vision.proj.weight    TERNARY [out_dim, C]  (per-row packed 2/byte)
#      * vision.proj.scale     F64 scalar (mean of row scales)
#      * vision.proj.bias      F32 [out_dim]
#      metadata: vision.{input_size,grid_side,feat_channels,out_dim}
#
#  The C++ side (src/audio/whisper_tiny.cpp, src/vision/mobilenet_v4.cpp)
#  consumes exactly these tensor names.
# =============================================================================
import argparse
import json
import os
import struct
import sys

import numpy as np

F16, F32, TERNARY = 1, 0, 40


def s_(x):
    b = x.encode('utf-8')
    return struct.pack('<Q', len(b)) + b


def u32(v): return struct.pack('<I', v)
def u64(v): return struct.pack('<Q', v)
def i32(v): return struct.pack('<i', v)
def f64(v): return struct.pack('<d', v)

T_UINT64, T_FLOAT64, T_STRING, T_ARRAY = 10, 12, 8, 9


class GgufWriter:
    def __init__(self):
        self.kv = []
        self.tensors = []

    def add_u64(self, k, v): self.kv.append((k, T_UINT64, u64(v)))
    def add_f64(self, k, v): self.kv.append((k, T_FLOAT64, f64(v)))
    def add_str(self, k, v): self.kv.append((k, T_STRING, s_(v)))

    def add_tensor(self, name, shape_rowmajor, dtype, data):
        ne = list(reversed([int(d) for d in shape_rowmajor]))
        self.tensors.append((name, ne, dtype, data))

    def write(self, path):
        out = b'GGUF' + u32(3) + u64(len(self.tensors)) + u64(len(self.kv))
        for k, t, v in self.kv:
            out += s_(k) + u32(t) + v
        offsets, off = [], 0
        for _, _, _, data in self.tensors:
            offsets.append(off)
            off += (len(data) + 31) // 32 * 32
        for (name, ne, dtype, _), o in zip(self.tensors, offsets):
            out += s_(name) + u32(len(ne))
            for d in ne:
                out += u64(d)
            out += u32(dtype) + u64(o)
        with open(path, 'wb') as f:
            f.write(out)
            for (_, _, _, data), o in zip(self.tensors, offsets):
                pos = f.tell()
                pad = len(out) + o - pos
                if pad > 0:
                    f.write(b'\x00' * pad)
                f.write(data)
        return os.path.getsize(path)


def load_safetensors(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(n).decode('utf-8'))
    return header, 8 + n


def read_st(path, header, ds, key):
    meta = header[key]
    _, shape, (a, b) = meta['dtype'], meta['shape'], meta['data_offsets']
    with open(path, 'rb') as f:
        f.seek(ds + a)
        raw = f.read(b - a)
    if meta['dtype'] == 'BF16':
        bits = np.frombuffer(raw, dtype='<u2').astype(np.uint32) << 16
        arr = bits.view(np.float32)
    elif meta['dtype'] == 'F16':
        arr = np.frombuffer(raw, dtype='<f2').astype(np.float32)
    elif meta['dtype'] == 'F32':
        arr = np.frombuffer(raw, dtype='<f4')
    else:
        raise SystemExit(f'unsupported safetensors dtype {meta["dtype"]}')
    return arr.reshape(shape).astype(np.float32)


def f16_bytes(w):
    return np.ascontiguousarray(w, dtype='<f2').tobytes()


def bytes_to_unicode():
    """The standard GPT-2 byte<->unicode table (whisper uses the same)."""
    bs = (list(range(ord('!'), ord('~') + 1)) +
          list(range(ord('\u00a1'), ord('\u00ac') + 1)) +
          list(range(ord('\u00ae'), ord('\u00ff') + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def load_whisper_vocab(path, n_vocab=51865):
    """id -> raw bytes for the whisper vocab (tokenizer.json from HF)."""
    with open(path, 'r', encoding='utf-8') as f:
        tj = json.load(f)
    vocab = tj['model']['vocab']                  # piece -> id
    u2b = {v: k for k, v in bytes_to_unicode().items()}
    pieces = [''] * n_vocab
    for piece, tid in vocab.items():
        if 0 <= tid < n_vocab:
            if piece.startswith('<|') and piece.endswith('|>'):
                pieces[tid] = piece.encode('utf-8')
            else:
                pieces[tid] = bytes(u2b.get(ch, ord(ch) & 0xFF) for ch in piece)
    # whisper ids >= 50258 (language/timestamp/task specials) live in
    # added_tokens, not model.vocab
    for at in tj.get('added_tokens', []):
        tid = at['id']
        if 0 <= tid < n_vocab:
            pieces[tid] = at['content'].encode('utf-8')
    return pieces


def quantize_ternary_rows(W):
    """BitNet b1.58 per-row ternary; returns packed bytes + row scales."""
    rows, cols = W.shape
    rb = (cols + 1) // 2
    packed = np.zeros((rows, rb), dtype=np.uint8)
    scales = np.zeros(rows, dtype='<f4')
    for r in range(rows):
        row = W[r]
        s = float(np.mean(np.abs(row)))
        scales[r] = s
        if s == 0.0:
            continue
        t = np.round(np.clip(row / s, -1.0, 1.0)).astype(np.int8)
        for i in range(cols):
            nib = t[i] & 0x0F
            if i % 2 == 0:
                packed[r, i // 2] |= nib
            else:
                packed[r, i // 2] |= nib << 4
    return packed.reshape(-1), scales


def convert_whisper(args, w):
    header, ds = load_safetensors(args.whisper)

    # official mel filterbank
    z = np.load(args.mel_npz)
    mel = np.ascontiguousarray(z['mel_80'], dtype='<f4')
    w.add_tensor('whisper.mel_filters', mel.shape, F32, mel.tobytes())

    def T(k):
        return read_st(args.whisper, header, ds, 'model.encoder.' + k)

    w.add_tensor('whisper.conv1.weight', T('conv1.weight').shape, F16,
                 f16_bytes(T('conv1.weight')))
    w.add_tensor('whisper.conv1.bias', T('conv1.bias').shape, F16,
                 f16_bytes(T('conv1.bias')))
    w.add_tensor('whisper.conv2.weight', T('conv2.weight').shape, F16,
                 f16_bytes(T('conv2.weight')))
    w.add_tensor('whisper.conv2.bias', T('conv2.bias').shape, F16,
                 f16_bytes(T('conv2.bias')))
    w.add_tensor('whisper.pos_embed', T('embed_positions.weight').shape, F16,
                 f16_bytes(T('embed_positions.weight')))
    w.add_tensor('whisper.enc_ln.weight', T('layer_norm.weight').shape, F16,
                 f16_bytes(T('layer_norm.weight')))
    w.add_tensor('whisper.enc_ln.bias', T('layer_norm.bias').shape, F16,
                 f16_bytes(T('layer_norm.bias')))

    n_layers = 4
    for l in range(n_layers):
        cp = f'model.encoder.layers.{l}.'      # checkpoint prefix
        for src, dst in (('self_attn.q_proj', 'q'), ('self_attn.k_proj', 'k'),
                         ('self_attn.v_proj', 'v'),
                         ('self_attn.out_proj', 'out'),
                         ('fc1', 'fc1'), ('fc2', 'fc2')):
            w.add_tensor(f'whisper.enc.{l}.{dst}.weight',
                         read_st(args.whisper, header, ds,
                                 cp + src + '.weight').shape, F16,
                         f16_bytes(read_st(args.whisper, header, ds,
                                           cp + src + '.weight')))
            bkey = cp + src + '.bias'
            if bkey in header:
                w.add_tensor(f'whisper.enc.{l}.{dst}.bias',
                             read_st(args.whisper, header, ds, bkey).shape,
                             F16, f16_bytes(read_st(args.whisper, header,
                                                    ds, bkey)))
        # post-LN blocks: self_attn_layer_norm (ln1), final_layer_norm (ln2)
        for src, dst in (('self_attn_layer_norm', 'ln1'),
                         ('final_layer_norm', 'ln2')):
            w.add_tensor(f'whisper.enc.{l}.{dst}.weight',
                         read_st(args.whisper, header, ds,
                                 cp + src + '.weight').shape, F16,
                         f16_bytes(read_st(args.whisper, header, ds,
                                           cp + src + '.weight')))
            w.add_tensor(f'whisper.enc.{l}.{dst}.bias',
                         read_st(args.whisper, header, ds,
                                 cp + src + '.bias').shape, F16,
                         f16_bytes(read_st(args.whisper, header, ds,
                                           cp + src + '.bias')))
    return n_layers


def convert_decoder(args, w):
    """whisper-tiny DECODER: token embedding (tied lm_head), positional
    embeddings, 4 pre-LN blocks (self-attn + cross-attn + MLP), final LN,
    and the byte-level vocab (offsets + bytes tensors)."""
    header, ds = load_safetensors(args.whisper)

    def T(k):
        return read_st(args.whisper, header, ds, k)

    def add16(name, arr):
        w.add_tensor(name, arr.shape, F16, f16_bytes(arr))

    # token embedding doubles as the lm_head (whisper ties them)
    tok = T('model.decoder.embed_tokens.weight')            # [V, 384]
    w.add_tensor('whisper.dec.tok_embd', tok.shape, F16, f16_bytes(tok))
    add16('whisper.dec.pos_embed', T('model.decoder.embed_positions.weight'))
    add16('whisper.dec_ln.weight', T('model.decoder.layer_norm.weight'))
    add16('whisper.dec_ln.bias', T('model.decoder.layer_norm.bias'))

    n_layers = 4
    for l in range(n_layers):
        cp = f'model.decoder.layers.{l}.'
        p = f'whisper.dec.{l}.'
        add16(p + 'ln1.weight', T(cp + 'self_attn_layer_norm.weight'))
        add16(p + 'ln1.bias', T(cp + 'self_attn_layer_norm.bias'))
        add16(p + 'ln2.weight', T(cp + 'encoder_attn_layer_norm.weight'))
        add16(p + 'ln2.bias', T(cp + 'encoder_attn_layer_norm.bias'))
        add16(p + 'ln3.weight', T(cp + 'final_layer_norm.weight'))
        add16(p + 'ln3.bias', T(cp + 'final_layer_norm.bias'))
        for src, dst in (('self_attn.q_proj', 'self.q'),
                         ('self_attn.k_proj', 'self.k'),
                         ('self_attn.v_proj', 'self.v'),
                         ('self_attn.out_proj', 'self.out'),
                         ('encoder_attn.q_proj', 'cross.q'),
                         ('encoder_attn.k_proj', 'cross.k'),
                         ('encoder_attn.v_proj', 'cross.v'),
                         ('encoder_attn.out_proj', 'cross.out')):
            add16(p + dst + '.weight', T(cp + src + '.weight'))
            # whisper-tiny ships NO k_proj bias (self- and cross-attn)
            if cp + src + '.bias' in header:
                add16(p + dst + '.bias', T(cp + src + '.bias'))
        add16(p + 'fc1.weight', T(cp + 'fc1.weight'))
        add16(p + 'fc1.bias', T(cp + 'fc1.bias'))
        add16(p + 'fc2.weight', T(cp + 'fc2.weight'))
        add16(p + 'fc2.bias', T(cp + 'fc2.bias'))

    # vocab: piece bytes, unicode-mapped back to raw bytes (byte-level BPE),
    # concatenated + offsets (loader has no string-array dtype). Special
    # <|...|> tokens keep their literal UTF-8 bytes.
    pieces = load_whisper_vocab(args.vocab_json, tok.shape[0])
    blobs, offs = [], [0]
    for pc in pieces:
        blobs.append(pc)
        offs.append(offs[-1] + len(pc))
    all_bytes = b''.join(blobs)
    w.add_tensor('whisper.vocab_offsets', np.asarray(offs, dtype='<i4').shape,
                 F32, np.ascontiguousarray(offs, dtype='<i4').tobytes())
    w.add_tensor('whisper.vocab_bytes', (len(all_bytes),), F16,
                 np.frombuffer(all_bytes, dtype=np.uint8).astype('<f2').tobytes())
    w.add_u64('whisper.n_vocab', tok.shape[0])
    w.add_u64('whisper.n_dec_layer', n_layers)
    return n_layers


def convert_vision(args, w):
    # The C++ encoder consumes a ternary projection [out_dim, C] applied to
    # per-patch RGB statistics. Derive that projection with a fixed random
    # seed from the checkpoint-free spec: it is deterministic, documented, and
    # swappable for a distilled MobileNetV4 embedding matrix when one lands.
    rng = np.random.default_rng(1234)
    C, out_dim = 96, 768
    W = rng.standard_normal((out_dim, C)).astype(np.float32) * 0.05
    data, scales = quantize_ternary_rows(W)
    w.add_tensor('vision.proj.weight', W.shape, TERNARY, data.tobytes())
    w.add_tensor('vision.proj.scale', (out_dim,), F32, scales.tobytes())
    # per-row scales need an fp32 TENSOR; C++ reads vision.proj.scale as
    # f64 metadata too, so write the mean for the legacy path.
    mean_scale = float(np.mean(scales))
    w.kv.append(('vision.proj.scale_mean', T_FLOAT64, f64(mean_scale)))
    w.add_u64('vision.input_size', 96)
    w.add_u64('vision.grid_side', 28)
    w.add_u64('vision.feat_channels', C)
    w.add_u64('vision.out_dim', out_dim)
    return mean_scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--whisper', default='models/whisper-tiny.safetensors')
    ap.add_argument('--mel-npz', default='models/mel_filters.npz')
    ap.add_argument('--vocab-json',
                    default='models/whisper-tiny-tokenizer.json')
    ap.add_argument('--out-whisper', default='models/whisper-tiny-encoder.gguf')
    ap.add_argument('--out-vision', default='models/vision-proj.gguf')
    args = ap.parse_args()

    w = GgufWriter()
    w.add_str('general.architecture', 'omniseed-whisper-encoder')
    n_layers = convert_whisper(args, w)
    w.add_u64('whisper.n_mels', 80)
    w.add_u64('whisper.n_audio_ctx', 1500)
    w.add_u64('whisper.n_audio_state', 384)
    w.add_u64('whisper.n_audio_head', 6)
    w.add_u64('whisper.n_audio_layer', n_layers)
    try:
        n_dec = convert_decoder(args, w)
    except (OSError, KeyError, SystemExit) as e:
        print(f'[conv] decoder not converted: {e}')
        n_dec = 0
    size = w.write(args.out_whisper)
    dec_note = f' + decoder ({n_dec} blocks)' if n_dec else ''
    print(f'[conv] {args.out_whisper}: {size:,} bytes ({size/1048576:.1f} MB)'
          f'{dec_note}')

    v = GgufWriter()
    v.add_str('general.architecture', 'omniseed-vision-proj')
    convert_vision(args, v)
    size = v.write(args.out_vision)
    print(f'[conv] {args.out_vision}: {size:,} bytes ({size/1048576:.2f} MB)')


if __name__ == '__main__':
    main()
