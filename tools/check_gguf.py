#!/usr/bin/env python3
"""Tensor-by-tensor roundtrip validation: models/rwkv7-0.1B-ternary.gguf vs
the source safetensors checkpoint.

For every tensor in the GGUF, dequantize it exactly like the C++ loader would
(per-row scales for i8/ternary; LoRA factors are stored transposed) and compare
against the original fp32 checkpoint values.  Reports max abs error, scale
sanity, and NaN counts.  A clean report means the conversion + serialization
path is correct and any remaining badness lives in the C++ forward.
"""
import json
import math
import struct
import sys

import numpy as np

GGUF = sys.argv[1] if len(sys.argv) > 1 else 'models/rwkv7-0.1B-ternary.gguf'
ST = sys.argv[2] if len(sys.argv) > 2 else 'models/model.safetensors'

F32, F16, I8, TERNARY = 0, 1, 4, 40
PREFIX = 'rwkv7.'


def r_u32(b, o): return struct.unpack_from('<I', b, o)[0]
def r_u64(b, o): return struct.unpack_from('<Q', b, o)[0]
def r_i32(b, o): return struct.unpack_from('<i', b, o)[0]


def r_str(b, o):
    n = r_u64(b, o); o += 8
    return b[o:o + n].decode('utf-8'), o + n


def parse_gguf(b):
    assert b[0:4] == b'GGUF', b[0:4]
    o = 4
    version = r_u32(b, o); o += 4
    n_tensors = r_u64(b, o); o += 8
    n_kv = r_u64(b, o); o += 8
    kv = {}
    for _ in range(n_kv):
        k, o = r_str(b, o)
        t = r_i32(b, o); o += 4
        if t == 8:
            kv[k], o = r_str(b, o)
        elif t == 4: kv[k] = r_u32(b, o); o += 4
        elif t == 5: kv[k] = struct.unpack_from('<i', b, o)[0]; o += 4
        elif t == 6: kv[k] = struct.unpack_from('<f', b, o)[0]; o += 4
        elif t == 10: kv[k] = r_u64(b, o); o += 8
        elif t == 12: kv[k] = struct.unpack_from('<d', b, o)[0]; o += 8
        elif t == 9:
            et = r_i32(b, o); o += 4
            n = r_u64(b, o); o += 8
            if et == 8:
                items = []
                for _ in range(n):
                    _, o = r_str(b, o)
                kv[k] = ('array<string>', n)
            else:
                o += n * {4: 4, 5: 4, 6: 4, 10: 8, 12: 8}[et]
                kv[k] = (f'array<{et}>', n)
        else:
            raise SystemExit(f'kv type {t} for {k}')
    tensors = []
    for _ in range(n_tensors):
        name, o = r_str(b, o)
        nd = r_u32(b, o); o += 4
        ne = [r_u64(b, o + 8 * i) for i in range(nd)]; o += 8 * nd
        dt = r_i32(b, o); o += 4
        off = r_u64(b, o); o += 8
        tensors.append((name, ne, dt, off))
    return version, kv, tensors, o


def load_safetensors(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(n).decode('utf-8'))
    return header, 8 + n


def read_st(path, header, data_start, key):
    meta = header[key]
    dt, shape, (a, b) = meta['dtype'], meta['shape'], meta['data_offsets']
    with open(path, 'rb') as f:
        f.seek(data_start + a)
        raw_bytes = f.read(b - a)
    if dt == 'BF16':
        # BF16 = top 16 bits of fp32; widen via uint16 -> uint32 bit shift
        bits = np.frombuffer(raw_bytes, dtype='<u2').astype(np.uint32) << 16
        arr = bits.view(np.float32)
    else:
        np_dt = {'F32': '<f4', 'F16': '<f2', 'I8': 'i1'}[dt]
        arr = np.frombuffer(raw_bytes, dtype=np_dt)
    return arr.reshape(shape).astype(np.float32)


def unpack_row(row_bytes, n):
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        nib = (row_bytes[i >> 1] >> ((i & 1) * 4)) & 0x0F
        out[i] = nib if nib < 8 else nib - 16
    return out


def dequant_gguf(b, data_start, tdir, name):
    ne, dt, off = tdir[name]
    if len(ne) == 1:
        rows, cols = 1, int(ne[0])
        if name + '.scale' in tdir:
            sne, sdt, soff = tdir[name + '.scale']
            nbytes = int(np.prod(sne)) * 4
            scales = np.frombuffer(b[data_start + soff:data_start + soff + nbytes], '<f4')
        raw = b[data_start + off:data_start + off + cols * (4 if dt == F32 else 2 if dt == F16 else 1)]
        if dt == F32:   return np.frombuffer(raw, '<f4')
        if dt == F16:   return np.frombuffer(raw, '<f2').astype(np.float32)
        raise SystemExit(f'unsupported 1-D dtype {dt} for {name}')
    # file dims are ggml order (reversed); row-major matrix is [ne[1], ne[0]]
    rows, cols = int(ne[1]), int(ne[0])
    raw = b[data_start + off:data_start + off + rows * cols * (4 if dt == F32 else 2 if dt == F16 else 1)]
    scales = None
    if name + '.scale' in tdir:
        sne, sdt, soff = tdir[name + '.scale']
        nbytes = int(np.prod(sne)) * 4
        scales = np.frombuffer(b[data_start + soff:data_start + soff + nbytes], '<f4')
    if dt == F32:
        return np.frombuffer(raw, '<f4').reshape(rows, cols)
    if dt == F16:
        return np.frombuffer(raw, '<f2').astype(np.float32).reshape(rows, cols)
    if dt == I8:
        q = np.frombuffer(raw, np.int8).reshape(rows, cols).astype(np.float32)
        return q * scales[:, None]
    if dt == TERNARY:
        rb = (cols + 1) // 2
        packed = np.frombuffer(raw, np.uint8).reshape(rows, rb)
        out = np.stack([unpack_row(packed[r], cols) for r in range(rows)])
        return out * scales[:, None]
    raise SystemExit(f'dtype {dt} for {name}')


def main():
    b = open(GGUF, 'rb').read()
    version, kv, tensors, data_start = parse_gguf(b)
    tdir = {name: (ne, dt, off) for name, ne, dt, off in tensors if not name.endswith('.scale')}
    print(f'gguf version={version} tensors={len(tensors)} kv={len(kv)}')
    header, ds_st = load_safetensors(ST)

    L = kv['omniseed.layer_count']
    bad = 0
    checked = 0

    def check(gg_name, st_name, transpose=False, tol=None):
        nonlocal bad, checked
        if gg_name not in tdir:
            return
        got = dequant_gguf(b, data_start, tdir, gg_name)
        want = read_st(ST, header, ds_st, st_name)
        if transpose:
            got = got.T
        err = float(np.abs(got - want).max())
        nan = int(np.isnan(got).sum())
        rel = err / max(float(np.abs(want).max()), 1e-9)
        # quantization step bound: i8 => max|row|/127; ternary => mean|row|
        ok = nan == 0 and rel < 0.05
        checked += 1
        if not ok or checked <= 6:
            print(f'{"OK " if ok else "BAD"} {gg_name:34s} shape={got.shape} '
                  f'maxerr={err:.5f} rel={rel:.4f} nan={nan}')
        if not ok:
            bad += 1

    # embedding + head (fp16)
    check('token.embd', PREFIX + 'emb.weight')
    check('head.weight', PREFIX + 'head.weight' if PREFIX + 'head.weight' in header else 'head.weight')

    for l in range(L):
        cp, gp = f'{PREFIX}blocks.{l}.', f'blocks.{l}.'
        for nm in ('ln1.weight', 'ln1.bias', 'ln2.weight', 'ln2.bias',
                   'ln0.weight', 'ln0.bias'):
            check(gp + nm, cp + nm)
        for a, bb in (('x_r', 'tmix_r'), ('x_w', 'tmix_w'), ('x_k', 'tmix_k'),
                      ('x_v', 'tmix_v'), ('x_a', 'tmix_a'), ('x_g', 'tmix_g')):
            if gp + f'att.{bb}' in tdir:
                ne, dt, off = tdir[gp + f'att.{bb}']
                n = int(np.prod(ne))
                got = np.frombuffer(b[data_start + off:data_start + off + n * 4], '<f4')
                want = read_st(ST, header, ds_st, cp + f'att.{a}').ravel()
                checked += 1
                err = float(np.abs(got - want).max()) if got.shape == want.shape else float('inf')
                if err > 1e-5:
                    print(f'BAD {gp}att.{bb}: shape {got.shape} vs {want.shape}, maxerr {err:.6f}')
                    bad += 1
        for nm in ('receptance', 'key', 'value', 'output'):
            check(gp + f'att.{nm}', cp + f'att.{nm}.weight')
        for nm in ('w1', 'w2', 'a1', 'a2', 'g1', 'g2', 'v1', 'v2'):
            check(gp + f'att.{nm}', cp + f'att.{nm}', transpose=True)
        for a, bb in (('w0', 'w_bias'), ('a0', 'a_bias'), ('v0', 'v_bias')):
            if gp + f'att.{bb}' in tdir:
                ne, dt, off = tdir[gp + f'att.{bb}']
                n = int(np.prod(ne))
                got = np.frombuffer(b[data_start + off:data_start + off + n * 4], '<f4')
                want = read_st(ST, header, ds_st, cp + f'att.{a}').ravel()
                checked += 1
                err = float(np.abs(got - want).max()) if got.shape == want.shape else float('inf')
                if err > 1e-5:
                    print(f'BAD {gp}att.{bb}: shape {got.shape} vs {want.shape}, maxerr {err:.6f}')
                    bad += 1
        # r_k, k_k, k_a vectors
        for nm, src_nm in (('r_k', 'r_k'), ('k_k', 'k_k'), ('k_a', 'k_a')):
            if gp + f'att.{nm}' in tdir:
                ne, dt, off = tdir[gp + f'att.{nm}']
                n = int(np.prod(ne))
                got = np.frombuffer(b[data_start + off:data_start + off + n * 4], '<f4')
                want = read_st(ST, header, ds_st, cp + f'att.{src_nm}').ravel()
                checked += 1
                err = float(np.abs(got - want).max()) if got.shape == want.shape else float('inf')
                if err > 1e-5:
                    print(f'BAD {gp}att.{nm} (shape {got.shape} vs {want.shape}, maxerr {err:.6f})')
                    bad += 1
        # group norm (ln_x in checkpoint) + ffn
        if gp + 'att.gn.weight' in tdir:
            ne, dt, off = tdir[gp + 'att.gn.weight']
            n = int(np.prod(ne))
            got = np.frombuffer(b[data_start + off:data_start + off + n * 4], '<f4')
            want = read_st(ST, header, ds_st, cp + 'att.ln_x.weight').ravel()
            checked += 1
            if float(np.abs(got - want).max()) > 1e-5:
                print(f'BAD {gp}att.gn.weight')
                bad += 1
        if gp + 'ffn.tmix_v' in tdir:
            ne, dt, off = tdir[gp + 'ffn.tmix_v']
            n = int(np.prod(ne))
            got = np.frombuffer(b[data_start + off:data_start + off + n * 4], '<f4')
            want = read_st(ST, header, ds_st, cp + 'ffn.x_k').ravel()
            checked += 1
            if float(np.abs(got - want).max()) > 1e-5:
                print(f'BAD {gp}ffn.tmix_v')
                bad += 1
        for nm in ('key', 'value'):
            check(gp + f'ffn.{nm}', cp + f'ffn.{nm}.weight')
        print(f'layer {l} done (running bad={bad})')

    print(f'\n=== {checked} tensors checked, {bad} BAD ===')
    return 0 if bad == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
