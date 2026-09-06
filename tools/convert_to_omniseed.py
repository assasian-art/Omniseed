#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — convert_to_omniseed.py   (OFFLINE tool, never ships at runtime)
#
#  Converts a real RWKV-7 "World" checkpoint (HuggingFace safetensors, e.g.
#  Hakureirm/rwkv7-0.1b-hf — a format conversion of BlinkDL/rwkv-7-world
#  RWKV-x070-World-0.1B-v2) into the OmniSeed GGUF:
#
#    * every linear (att.r/k/v/o, att.{w,a,g,v}{1,2}, ffn.key/value)
#      -> BitNet b1.58 ternary {-1,0,+1}, 2 weights/byte, per-ROW fp32 scale
#         (= mean |W_row|, exactly src/core/bitlinear.cpp quantize_row)
#    * token.embd + head stay fp16 (documented policy)
#    * all norms/mixes/biases/k_k/k_a/r_k stay fp32
#    * tensor names + omniseed.* metadata match src/core/gguf_loader.cpp /
#      src/core/rwkv.cpp exactly
#    * packs the RWKV world vocab (65,536 byte-level tokens) into
#      tokenizer.ggml.tokens / token_type metadata
#
#  Also: `--check-prompt "Hello"` runs a full-precision fp32 torch forward
#  on the ORIGINAL weights and prints a greedy reference transcript. That is
#  the ground truth for validating the C++ runtime (separates quantization
#  loss from runtime math bugs).
#
#  Usage:
#    python tools/convert_to_omniseed.py \
#        --safetensors models/model.safetensors \
#        --vocab models/rwkv_vocab_v20230424.txt \
#        --out models/rwkv7-0.1B-ternary.gguf \
#        [--check-prompt "Hello"] [--ternary-head]
# =============================================================================
import argparse
import ast
import json
import math
import os
import struct

import numpy as np

# GGUF dtypes used by the OmniSeed loader (src/core/gguf_format.h)
F32, F16, I8, TERNARY = 0, 1, 4, 40
# GGUF metadata value types
T_UINT32, T_FLOAT32, T_STRING, T_ARRAY, T_UINT64, T_FLOAT64, T_INT32 = 4, 6, 8, 9, 10, 12, 5

def s_(x):
    b = x.encode('utf-8')
    return struct.pack('<Q', len(b)) + b   # length = BYTES, not codepoints
def u32(v): return struct.pack('<I', v)
def u64(v): return struct.pack('<Q', v)
def f32(v): return struct.pack('<f', v)
def f64(v): return struct.pack('<d', v)
def i32(v): return struct.pack('<i', v)

# -----------------------------------------------------------------------------
# Safetensors reading (header is JSON: key -> {dtype, shape, data_offsets})
# -----------------------------------------------------------------------------
def load_safetensors(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(n).decode('utf-8'))
        data_start = 8 + n
    return header, data_start

def read_tensor(path, header, data_start, key):
    """Returns a numpy array (fp32) for `key`."""
    import torch  # local import: only needed for bf16 -> fp32
    info = header[key]
    dt, shape = info['dtype'], info['shape']
    begin, end = info['data_offsets']
    with open(path, 'rb') as f:
        f.seek(data_start + begin)
        raw = f.read(end - begin)
    if dt == 'BF16':
        t = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16)
        return t.reshape(shape).to(torch.float32).numpy()
    if dt == 'F32':
        return np.frombuffer(raw, dtype='<f4').reshape(shape).copy()
    if dt == 'F16':
        return np.frombuffer(raw, dtype='<f2').astype('<f4').reshape(shape)
    raise ValueError(f'unsupported safetensors dtype {dt} for {key}')

# -----------------------------------------------------------------------------
# Ternary quantization (BitNet b1.58, per output row) + nibble packing
# (low nibble = even index; 0x0 -> 0, 0x1 -> +1, 0xF -> -1; matches
#  src/core/bitlinear.cpp pack_ternary)
# -----------------------------------------------------------------------------
def quantize_and_pack(W):
    out_dim, in_dim = W.shape
    row_bytes = (in_dim + 1) // 2
    packed = np.zeros((out_dim, row_bytes), dtype=np.uint8)
    scales = np.zeros(out_dim, dtype='<f4')
    for r in range(out_dim):
        row = W[r]
        s = float(np.mean(np.abs(row)))
        scales[r] = s
        if s == 0.0:
            continue
        t = np.round(np.clip(row / s, -1.0, 1.0)).astype(np.int8)
        lo = (t[0::2] & 0x0F).astype(np.uint8)
        hi = (t[1::2] & 0x0F).astype(np.uint8) << 4
        packed[r, :len(lo)] = lo
        packed[r, :len(hi)] |= hi
    return packed.reshape(-1), scales

def quantize_i8(W):
    """Per-row symmetric int8: scale = max|row|/127. Dequant W ~= q * scale.
    PTQ ternary (b1.58) is too lossy without QAT (verified: fp32 sim of the
    ternary-quantized model degenerates); int8 keeps the model coherent at
    1 byte/param (2x smaller than fp16) and is the default for linears."""
    out_dim = W.shape[0]
    s = np.abs(W).max(axis=1) / 127.0
    s = np.maximum(s, 1e-12).astype('<f4')
    q = np.round(W / s[:, None]).astype(np.int8)
    q = np.clip(q, -127, 127)
    return q, s

def to_f16_bytes(W):
    return np.ascontiguousarray(W, dtype='<f2').tobytes()

# HF-converted checkpoints prefix every parameter with the model base name.
CKPT_PREFIX = 'rwkv7.'
def K(base):
    return CKPT_PREFIX + base

# -----------------------------------------------------------------------------
# World vocab parsing: lines are "id 'python-escaped-piece' declared_len"
# (space-separated; pieces may contain escaped spaces; id 0 is absent).
# -----------------------------------------------------------------------------
def load_world_vocab(path, vocab_size):
    pieces = [''] * vocab_size          # id 0 stays an empty piece
    types = [1] * vocab_size            # 1 = NORMAL
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\r\n')
            if not line:
                continue
            p = line.split(' ')
            if len(p) < 3:
                continue
            tid = int(p[0])
            # piece_repr ALREADY includes its surrounding quotes (python literal)
            piece_repr = ' '.join(p[1:-1])
            try:
                piece = ast.literal_eval(piece_repr)
            except Exception:
                piece = piece_repr
            if isinstance(piece, tuple):
                piece = piece[0] if piece else ''
            if isinstance(piece, bytes):
                piece = piece.decode('utf-8', 'replace')
            if not isinstance(piece, str):
                piece = str(piece)
            if 0 <= tid < vocab_size:
                pieces[tid] = piece
    return pieces, types

# -----------------------------------------------------------------------------
# GGUF writer (version 3, matching src/core/gguf_loader.cpp; the loader
# honors explicit per-tensor offsets so alignment is optional, but 32-byte
# alignment keeps the file friendly for future mmap kernel paths)
# -----------------------------------------------------------------------------
class GgufWriter:
    def __init__(self):
        self.kv = []      # (key, type, payload bytes)
        self.tensors = [] # (name, ne_reversed, dtype, data_bytes)

    def add_u64(self, k, v):   self.kv.append((k, T_UINT64, u64(v)))
    def add_f64(self, k, v):   self.kv.append((k, T_FLOAT64, f64(v)))
    def add_str(self, k, v):   self.kv.append((k, T_STRING, s_(v)))
    def add_i32(self, k, v):   self.kv.append((k, T_INT32, i32(v)))

    def add_str_array(self, k, items):
        blob = b''.join(s_(x) for x in items)
        self.kv.append((k, T_ARRAY, i32(T_STRING) + u64(len(items)) + blob))

    def add_i32_array(self, k, items):
        blob = b''.join(i32(int(x)) for x in items)
        self.kv.append((k, T_ARRAY, i32(T_INT32) + u64(len(items)) + blob))

    def add_tensor(self, name, shape_rowmajor, dtype, data):
        # GGUF stores dims reversed vs row-major [rows, cols] -> ne=[cols, rows]
        ne = list(reversed([int(d) for d in shape_rowmajor]))
        self.tensors.append((name, ne, dtype, data))

    def write(self, path):
        out = b'GGUF' + u32(3) + u64(len(self.tensors)) + u64(len(self.kv))
        for k, t, v in self.kv:
            out += s_(k) + u32(t) + v
        # tensor directory with 32-byte-aligned offsets
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

# -----------------------------------------------------------------------------
# Reference fp32 forward (ground truth for C++ validation)
# -----------------------------------------------------------------------------
def reference_generate(st_path, header, data_start, prompt_ids, n_new=24):
    import torch
    torch.set_grad_enabled(False)
    _cache = {}
    def T(k):
        if k not in _cache:
            _cache[k] = torch.from_numpy(read_tensor(st_path, header, data_start, k))
        return _cache[k]

    V, E = T(K('emb.weight')).shape
    n_layers = max(int(k.split('.')[2]) for k in header if '.blocks.' in k) + 1
    H, D = T(K('blocks.0.att.r_k')).shape
    print(f'[ref] V={V} E={E} L={n_layers} H={H} D={D}')

    def ln(x, w, b, eps=1e-5):
        m = x.mean(-1, keepdim=True)
        v = x.var(-1, keepdim=True, unbiased=False)
        return (x - m) / torch.sqrt(v + eps) * w + b

    inv_sqe = 1.0 / math.sqrt(math.e)
    S = [torch.zeros(H, D, D) for _ in range(n_layers)]
    att_prev = [torch.zeros(E) for _ in range(n_layers)]
    ffn_prev = [torch.zeros(E) for _ in range(n_layers)]
    ids = list(prompt_ids)
    out_ids = []
    x = torch.zeros(E)
    logits = None

    for step in range(len(ids) + n_new):
        if step < len(ids):
            tok = ids[step]
        else:
            x_last = ln(x, T(K('ln_out.weight')), T(K('ln_out.bias'))).reshape(-1)
            # NB: this checkpoint stores head.weight WITHOUT the rwkv7. prefix.
            hkey = K('head.weight') if K('head.weight') in header else 'head.weight'
            head = T(hkey) if hkey in header else T(K('emb.weight'))
            logits = head.to(torch.float32) @ x_last
            nxt = int(torch.argmax(logits))
            out_ids.append(nxt)
            ids.append(nxt)
            tok = nxt
        x = T(K('emb.weight'))[tok].to(torch.float32)
        if K('blocks.0.ln0.weight') in header:
            x = ln(x, T(K('blocks.0.ln0.weight')), T(K('blocks.0.ln0.bias')))
        v_first = None
        for l in range(n_layers):
            p = f'rwkv7.blocks.{l}.'
            lx = ln(x, T(p + 'ln1.weight'), T(p + 'ln1.bias'))
            d = att_prev[l] - lx
            xr = lx + T(p + 'att.x_r')[0] * d
            xw = lx + T(p + 'att.x_w')[0] * d
            xk = lx + T(p + 'att.x_k')[0] * d
            xv = lx + T(p + 'att.x_v')[0] * d
            xa = lx + T(p + 'att.x_a')[0] * d
            xg = lx + T(p + 'att.x_g')[0] * d
            att_prev[l] = lx
            r = xr @ T(p + 'att.receptance.weight').T
            k = xk @ T(p + 'att.key.weight').T
            v = xv @ T(p + 'att.value.weight').T
            # LoRA factors are raw [E, rank] / [rank, E] parameters: x @ w1 @ w2
            w_log = -inv_sqe * torch.sigmoid(
                torch.tanh(xw @ T(p + 'att.w1')) @ T(p + 'att.w2')
                + T(p + 'att.w0')[0])
            a = torch.sigmoid(xa @ T(p + 'att.a1') @ T(p + 'att.a2')
                              + T(p + 'att.a0')[0])
            g = torch.sigmoid(xg @ T(p + 'att.g1')) @ T(p + 'att.g2')
            if l == 0:
                v_first = v
            else:
                vg = torch.sigmoid(xv @ T(p + 'att.v1') @ T(p + 'att.v2')
                                   + T(p + 'att.v0')[0])
                v = v + (v_first - v) * vg
            kkk = k * T(p + 'att.k_k')[0]
            kk = kkk.view(H, D)
            kk = kk / torch.sqrt((kk * kk).sum(-1, keepdim=True) + 1e-12)
            k += k * (a - 1) * T(p + 'att.k_a')[0]   # in-place blend BEFORE bonus
            k2 = k
            rh, kh, vh = (q.view(H, D) for q in (r, k2, v))
            wh = torch.exp(w_log).view(H, D)
            ah = a.view(H, D)
            kkh = kk
            for h in range(H):
                # HF convention: state axis 0 = key, axis 1 = value.
                # sa[j] = -sum_i kk[i]*S[i,j]  (contract over the KEY axis 0)
                sa = S[l][h].T @ (-kkh[h])
                S[l][h] = wh[h][:, None] * S[l][h] \
                          + (kkh[h] * ah[h])[:, None] * sa[None, :] \
                          + kh[h][:, None] * vh[h][None, :]
            # out[j] = sum_i r[i]*S[i,j]  (r over KEY axis, output = VALUE axis)
            y = torch.stack([rh[h] @ S[l][h] for h in range(H)], 0).reshape(E)
            yv = y.view(H, D)
            m_ = yv.mean(-1, keepdim=True)
            var = yv.var(-1, keepdim=True, unbiased=False)
            yn = ((yv - m_) / torch.sqrt(var + 64e-5)
                  * T(p + 'att.ln_x.weight').view(H, D)
                  + T(p + 'att.ln_x.bias').view(H, D)).reshape(E)
            bonus = ((rh * (k.view(H, D)) * T(p + 'att.r_k')).sum(-1, keepdim=True)
                     * vh).reshape(E)
            att_out = ((yn + bonus) * g) @ T(p + 'att.output.weight').T
            x = x + att_out
            lx2 = ln(x, T(p + 'ln2.weight'), T(p + 'ln2.bias'))
            xk2 = lx2 + T(p + 'ffn.x_k')[0] * (ffn_prev[l] - lx2)
            ffn_prev[l] = lx2
            inner = (xk2 @ T(p + 'ffn.key.weight').T).relu() ** 2
            x = x + inner @ T(p + 'ffn.value.weight').T
    return out_ids

# -----------------------------------------------------------------------------
# Main conversion
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--safetensors', default='models/model.safetensors')
    ap.add_argument('--vocab', default='models/rwkv_vocab_v20230424.txt')
    ap.add_argument('--out', default='models/rwkv7-0.1B-ternary.gguf')
    ap.add_argument('--check-prompt', default=None,
                    help='run the fp32 torch reference on this prompt and exit')
    ap.add_argument('--ternary-head', action='store_true',
                    help='also quantize head.weight (default keeps fp16)')
    args = ap.parse_args()

    header, data_start = load_safetensors(args.safetensors)
    keys = [k for k in header if k != '__metadata__']

    if args.check_prompt is not None:
        # The world vocab is byte-level: every utf-8 byte is a token id
        # (id = byte + 1), so an ASCII prompt encodes as its bytes + 1.
        print('[ref] NOTE: prompt ids = utf-8 bytes + 1 (byte-level world vocab)')
        ids = [b + 1 for b in args.check_prompt.encode('utf-8')]
        out = reference_generate(args.safetensors, header, data_start, ids)
        print('[ref] greedy continuation ids:', out)
        raw = bytes((b - 1) & 0xFF for b in out)
        txt = raw.decode('utf-8', 'replace')
        print('[ref] decoded:', txt.encode('ascii', 'backslashreplace').decode('ascii'))
        return

    emb_shape = header[K('emb.weight')]['shape']
    V, E = emb_shape
    n_layers = max(int(k.split('.')[2]) for k in keys if '.blocks.' in k) + 1
    H, D = header[K('blocks.0.att.r_k')]['shape']
    I = header[K('blocks.0.ffn.key.weight')]['shape'][0]   # ffn intermediate
    # w2/a2/g2/v2 are [rank, E] in the checkpoint -> rank = shape[0].
    rw = header[K('blocks.0.att.w2')]['shape'][0]
    ra = header[K('blocks.0.att.a2')]['shape'][0]
    rg = header[K('blocks.0.att.g2')]['shape'][0]
    rv = header[K('blocks.0.att.v2')]['shape'][0]
    print(f'[conv] V={V} E={E} L={n_layers} H={H} D={D} FFN={I} '
          f'ranks w={rw} a={ra} g={rg} v={rv}')

    # ---------------- vocab ----------------
    pieces, types = load_world_vocab(args.vocab, V)

    # ---------------- writer ----------------
    w = GgufWriter()
    w.add_str('general.architecture', 'omniseed-rwkv7')
    w.add_str('general.name', 'RWKV7-World-0.1B-ternary')
    w.add_u64('omniseed.layer_count', n_layers)
    w.add_u64('omniseed.embedding_length', E)
    w.add_u64('omniseed.vocab_size', V)
    w.add_u64('omniseed.key_length', D)
    w.add_u64('omniseed.mlp_rank', rw)
    w.add_u64('omniseed.ffn_intermediate', I)
    w.add_u64('omniseed.rank_decay', rw)
    w.add_u64('omniseed.rank_a', ra)
    w.add_u64('omniseed.rank_gate', rg)
    w.add_u64('omniseed.rank_value', rv)
    # world models: no <s>, no chat control tokens; eos = 0 (unused pad),
    # chat template is plain "User: ...\n\nAssistant:"
    w.add_i32('omniseed.bos_token_id', -1)
    w.add_i32('omniseed.eos_token_id', 0)
    w.add_i32('omniseed.user_start_token_id', -1)
    w.add_i32('omniseed.user_end_token_id', -1)
    w.add_i32('omniseed.assistant_start_token_id', -1)
    w.add_i32('omniseed.assistant_end_token_id', -1)
    w.add_str_array('tokenizer.ggml.tokens', pieces)
    w.add_i32_array('tokenizer.ggml.token_type', types)

    T = lambda k: read_tensor(args.safetensors, header, data_start, k)

    n_ternary = n_i8 = n_f16 = n_f32 = 0
    total_bytes = 0

    def add_quant(name, W, dtype):
        nonlocal n_ternary, n_i8, total_bytes
        if dtype == 'i8':
            data, scales = quantize_i8(W)
            w.add_tensor(name + '.weight', W.shape, I8, data.tobytes())
        else:
            data, scales = quantize_and_pack(W)
            w.add_tensor(name + '.weight', W.shape, TERNARY, data.tobytes())
        # PER-ROW scales as an fp32 tensor [out_dim]: one scalar for the whole
        # matrix is wrong — row scales differ by orders of magnitude.
        w.add_tensor(name + '.scale', (W.shape[0],), F32, scales.tobytes())
        if dtype == 'i8':
            n_i8 += 1
        else:
            n_ternary += 1
        total_bytes += data.nbytes + scales.nbytes

    def add_ternary(name, W):
        add_quant(name, W, 'ternary')

    def add_i8(name, W):
        add_quant(name, W, 'i8')

    def add_ternary_t(name, W, dtype='i8'):
        """Store W TRANSPOSED with per-row scales of the stored matrix.

        The checkpoint's LoRA factors are raw [E, rank] parameters applied as
        x @ w1; the C++ BitLinear expects [rank, E] row-major (y = W @ x).
        Storing w1.T gives exactly that. (nn.Linear weights are already
        [out, in] and stored as-is.)
        """
        add_quant(name, np.ascontiguousarray(W.T), dtype)

    def add_f32(name, W):
        nonlocal n_f32, total_bytes
        w.add_tensor(name, W.shape, F32, np.ascontiguousarray(W, '<f4').tobytes())
        n_f32 += 1
        total_bytes += W.nbytes

    def add_f16(name, W):
        nonlocal n_f16, total_bytes
        w.add_tensor(name, W.shape, F16, to_f16_bytes(W))
        n_f16 += 1
        total_bytes += W.nbytes

    # embedding (fp16) [V, E]
    add_f16('token.embd', T(K('emb.weight')))

    # block 0 input norm
    if K('blocks.0.ln0.weight') in header:
        add_f32('blocks.0.ln0.weight', T(K('blocks.0.ln0.weight')))
        add_f32('blocks.0.ln0.bias', T(K('blocks.0.ln0.bias')))

    for l in range(n_layers):
        src, dst = K(f'blocks.{l}.'), f'blocks.{l}.'   # src=ckpt, dst=GGUF
        add_f32(dst + 'ln1.weight', T(src + 'ln1.weight'))
        add_f32(dst + 'ln1.bias', T(src + 'ln1.bias'))
        add_f32(dst + 'ln2.weight', T(src + 'ln2.weight'))
        add_f32(dst + 'ln2.bias', T(src + 'ln2.bias'))
        for a, b in (('x_r', 'tmix_r'), ('x_w', 'tmix_w'), ('x_k', 'tmix_k'),
                     ('x_v', 'tmix_v'), ('x_a', 'tmix_a'), ('x_g', 'tmix_g')):
            add_f32(dst + 'att.' + b, T(src + 'att.' + a)[0])
        for a in ('receptance', 'key', 'value', 'output'):
            add_i8(dst + 'att.' + a, T(src + 'att.' + a + '.weight'))
        # LoRA factors are stored transposed (see add_ternary_t above):
        # w1/a1/g1/v1 [E,r] -> [r,E];  w2/a2/g2/v2 [r,E] -> [E,r].
        add_ternary_t(dst + 'att.w1', T(src + 'att.w1'))
        add_ternary_t(dst + 'att.w2', T(src + 'att.w2'))
        add_ternary_t(dst + 'att.a1', T(src + 'att.a1'))
        add_ternary_t(dst + 'att.a2', T(src + 'att.a2'))
        add_ternary_t(dst + 'att.g1', T(src + 'att.g1'))
        add_ternary_t(dst + 'att.g2', T(src + 'att.g2'))
        add_ternary_t(dst + 'att.v1', T(src + 'att.v1'))
        add_ternary_t(dst + 'att.v2', T(src + 'att.v2'))
        add_f32(dst + 'att.w_bias', T(src + 'att.w0')[0])
        add_f32(dst + 'att.a_bias', T(src + 'att.a0')[0])
        add_f32(dst + 'att.v_bias', T(src + 'att.v0')[0])
        add_f32(dst + 'att.k_k', T(src + 'att.k_k')[0])
        add_f32(dst + 'att.k_a', T(src + 'att.k_a')[0])
        add_f32(dst + 'att.r_k', T(src + 'att.r_k').reshape(-1))
        add_f32(dst + 'att.gn.weight', T(src + 'att.ln_x.weight'))
        add_f32(dst + 'att.gn.bias', T(src + 'att.ln_x.bias'))
        add_f32(dst + 'ffn.tmix_v', T(src + 'ffn.x_k')[0])
        add_i8(dst + 'ffn.key', T(src + 'ffn.key.weight'))
        add_i8(dst + 'ffn.value', T(src + 'ffn.value.weight'))

    add_f32('ln_out.weight', T(K('ln_out.weight')))
    add_f32('ln_out.bias', T(K('ln_out.bias')))
    hkey = K('head.weight') if K('head.weight') in header else 'head.weight'
    if args.ternary_head:
        add_ternary('head', T(hkey))          # add_ternary appends '.weight'
    else:
        add_f16('head.weight', T(hkey))       # add_f16 takes the FULL name

    size = w.write(args.out)
    print(f'[conv] tensors: i8={n_i8} ternary={n_ternary} fp16={n_f16} fp32={n_f32}')
    print(f'[conv] wrote {args.out}: {size:,} bytes '
          f'({size / 1048576:.1f} MB, tensor bytes {total_bytes / 1048576:.1f} MB)')

if __name__ == '__main__':
    main()
