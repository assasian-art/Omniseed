#!/usr/bin/env python3
"""T121b: cluster the int8 LM head into a product codebook (RWKV-Lite route a).

Per state/task_T121_brief.md:
  head.weight [V=65536, E=768] int8 + head.scale [V] f32  (48.25 MiB)
  ->
  head.centroid [K, E] fp16       (cluster centers in DEQUANTIZED space)
  head.code     [V]    int32      (row -> cluster id; int32 not uint16: keeps
                                   the GGUF dtype table untouched at a cost
                                   of +0.25 MiB — documented deviation)
  head.cnorm    [V]    fp32       (per-row ||row - centroid||, kept for the
                                   agreement probe / diagnostics, not needed
                                   at decode)

Decode: logits[v] = centroid[code[v]] . x   (row scale ABSORBED into the
centroid during clustering — rows are dequantized BEFORE k-means, so the
clustered head IS the weight matrix, no per-row scale needed).

Deterministic: seeded k-means++ (numpy Generator, seed from --seed), fixed
iteration count, empty-cluster -> reseed to the farthest point. Same input +
same flags => byte-identical output. Refuses to overwrite without --force.

Usage:
    .venv/Scripts/python.exe tools/convert_head_cluster.py \
        --src models/rwkv7-0.1B-ternary.gguf --k 4096 --iters 25 \
        --out build/rwkv7-headcluster.gguf [--force]
"""
import argparse
import math
import os
import struct
import sys

import numpy as np

GGUF_F32, GGUF_F16, GGUF_I8, GGUF_I32 = 0, 1, 4, 5
GROUP = 64


def fail(msg):
    print("convert_head_cluster: FAIL: %s" % msg, file=sys.stderr)
    sys.exit(2)


# ---------------- gguf raw parse/copy (same contract as convert_int4_ffn) ---
def r_u32(b, o): return struct.unpack_from('<I', b, o)[0]
def r_u64(b, o): return struct.unpack_from('<Q', b, o)[0]
def r_i32(b, o): return struct.unpack_from('<i', b, o)[0]
def u32(v): return struct.pack('<I', v)
def u64(v): return struct.pack('<Q', v)
def i32(v): return struct.pack('<i', v)
def f16_bytes(a): return a.astype('<f2').tobytes()


def r_str(b, o):
    n = r_u64(b, o); o += 8
    s = b[o:o + n].decode('utf-8'); o += n
    return s, o


def parse(b):
    if b[:4] != b'GGUF':
        fail("not a GGUF file")
    n_tensors, n_kv = r_u64(b, 8), r_u64(b, 16)
    o = 24
    for _ in range(n_kv):
        k, o = r_str(b, o)
        t = r_i32(b, o); o += 4
        if t == 8:
            _, o = r_str(b, o)
        elif t in (0, 1):
            o += 1
        elif t in (2, 3):
            o += 2
        elif t in (4, 5, 6, 7):
            o += 4
        elif t in (10, 11, 12):
            o += 8
        elif t == 9:
            et = r_i32(b, o); o += 4
            n = r_u64(b, o); o += 8
            if et == 8:
                for _ in range(n):
                    _, o = r_str(b, o)
            else:
                esz = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4,
                       10: 8, 11: 8, 12: 8}.get(et)
                if esz is None:
                    fail("kv %s: unsupported array elem %d" % (k, et))
                o += n * esz
        else:
            fail("kv %s: unsupported type %d" % (k, t))
    kv_end = o
    tensors = []
    for _ in range(n_tensors):
        name, o = r_str(b, o)
        nd = r_u32(b, o); o += 4
        ne = [r_u64(b, o + 8 * i) for i in range(nd)]; o += 8 * nd
        dt = r_i32(b, o); o += 4
        off = r_u64(b, o); o += 8
        tensors.append({'name': name, 'ne': ne, 'dt': dt, 'off': off})
    return kv_end, tensors, o


def nbytes_of(t):
    numel = math.prod(t['ne'])
    if t['dt'] in (40, 41):        # TERNARY / I4G64: 2 weights/byte
        return (numel + 1) // 2
    esz = {GGUF_F32: 4, GGUF_F16: 2, GGUF_I8: 1, GGUF_I32: 4}.get(t['dt'])
    if esz is None:
        fail("tensor %s: unsupported dtype %d for copy" % (t['name'], t['dt']))
    return numel * esz


# ------------------------- deterministic k-means ---------------------------
def kmeans_pp_init(X, k, rng):
    n = X.shape[0]
    centers = np.empty((k, X.shape[1]), dtype=np.float32)
    centers[0] = X[rng.integers(n)]
    # min squared distance to chosen centers
    d2 = np.sum((X - centers[0]) ** 2, axis=1)
    for i in range(1, k):
        total = float(d2.sum())
        if total <= 0.0:
            centers[i] = X[rng.integers(n)]
        else:
            centers[i] = X[rng.choice(n, p=d2 / total)]
        nd2 = np.sum((X - centers[i]) ** 2, axis=1)
        np.minimum(d2, nd2, out=d2)
    return centers


def assign_chunked(X, C, chunk=8192):
    """argmin_c ||x - c||^2, chunked over rows (8192xK fp32 stays small)."""
    n = X.shape[0]
    labels = np.empty(n, dtype=np.int32)
    cc = np.sum(C * C, axis=1)                       # [K]
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        D = (np.sum(X[a:b] ** 2, axis=1)[:, None]
             - 2.0 * (X[a:b] @ C.T) + cc[None, :])
        labels[a:b] = np.argmin(D, axis=1).astype(np.int32)
    return labels


def lloyd(X, k, iters, rng, log):
    C = kmeans_pp_init(X, k, rng)
    labels = None
    for it in range(iters):
        labels = assign_chunked(X, C)
        counts = np.bincount(labels, minlength=k)
        # empty clusters -> farthest point from its center (deterministic tie)
        empty = np.flatnonzero(counts == 0)
        if empty.size:
            Drow = np.sum((X - C[labels]) ** 2, axis=1)
            order = np.argsort(-Drow, kind='stable')
            for j, e in enumerate(empty):
                C[e] = X[order[j]]
            labels = assign_chunked(X, C)
            counts = np.bincount(labels, minlength=k)
        sums = np.zeros((k, X.shape[1]), dtype=np.float32)
        np.add.at(sums, labels, X)
        C = sums / np.maximum(counts, 1)[:, None]
        log("  lloyd iter %d/%d done" % (it + 1, iters))
    return C, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='models/rwkv7-0.1B-ternary.gguf')
    ap.add_argument('--out', default='build/rwkv7-headcluster.gguf')
    ap.add_argument('--k', type=int, default=4096)
    ap.add_argument('--iters', type=int, default=25)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    if not os.path.isfile(args.src):
        fail("source missing: %s" % args.src)
    if os.path.exists(args.out) and not args.force:
        fail("output exists (use --force): %s" % args.out)
    if args.k <= 0 or args.k > 65536:
        fail("k out of range: %d" % args.k)

    log = lambda m: print("convert_head_cluster: %s" % m)
    b = open(args.src, 'rb').read()
    kv_end, tensors, data_start = parse(b)
    by_name = {t['name']: t for t in tensors}
    for need in ('head.weight', 'head.scale'):
        if need not in by_name:
            fail("missing %s" % need)
    hw, hs = by_name['head.weight'], by_name['head.scale']
    if hw['dt'] != GGUF_I8 or hs['dt'] != GGUF_F32:
        fail("expected int8 head.weight + f32 head.scale, got %d/%d"
             % (hw['dt'], hs['dt']))
    V, E = hw['ne'][1], hw['ne'][0]         # gguf ne is reversed
    if hs['ne'][0] != V:
        fail("head.scale length != V")
    log("head V=%d E=%d k=%d iters=%d seed=%d" % (V, E, args.k, args.iters,
                                                  args.seed))

    raw = b[data_start + hw['off']: data_start + hw['off'] + V * E]
    if len(raw) != V * E:
        fail("head.weight truncated")
    W = np.frombuffer(raw, dtype=np.int8).reshape(V, E).astype(np.float32)
    scales = np.frombuffer(b[data_start + hs['off']:
                             data_start + hs['off'] + 4 * V],
                           dtype='<f4')
    X = W * scales[:, None]                  # dequantized rows
    del W
    log("dequantized head (%.1f MiB fp32 working set)" % (X.nbytes / 1048576))

    rng = np.random.default_rng(args.seed)
    C, labels = lloyd(X, args.k, args.iters, rng, log)

    resid = np.sqrt(np.sum((X - C[labels]) ** 2, axis=1))
    log("cluster residual: mean=%.4f p99=%.4f max=%.4f"
        % (float(resid.mean()), float(np.percentile(resid, 99)),
           float(resid.max())))
    cos = np.sum(X * C[labels], axis=1) / (
        np.linalg.norm(X, axis=1) * np.linalg.norm(C[labels], axis=1)
        + 1e-12)
    log("cos(row, centroid): mean=%.4f p1=%.4f"
        % (float(cos.mean()), float(np.percentile(cos, 1))))

    # ---- rebuild: replace head.weight with centroid/code/cnorm ------------
    new_tensors, conv_data = [], {}
    for t in tensors:
        if t['name'] == 'head.weight':
            new_tensors.append({'name': 'head.centroid',
                                'ne': [E, args.k], 'dt': GGUF_F16})
            new_tensors.append({'name': 'head.code',
                                'ne': [V], 'dt': GGUF_I32})
            new_tensors.append({'name': 'head.cnorm',
                                'ne': [V], 'dt': GGUF_F32})
            conv_data['head.centroid'] = f16_bytes(C)
            conv_data['head.code'] = labels.astype('<i4').tobytes()
            conv_data['head.cnorm'] = resid.astype('<f4').tobytes()
        else:
            new_tensors.append(dict(t))

    header = bytearray(b[:kv_end])
    header[8:16] = u64(len(new_tensors))
    out_data, off = bytearray(), 0
    for t in new_tensors:
        if t['name'] in conv_data:
            data = conv_data[t['name']]
        else:
            src_t = by_name[t['name']]
            nb = nbytes_of(src_t)
            data = b[data_start + src_t['off']:
                     data_start + src_t['off'] + nb]
            if len(data) != nb:
                fail("tensor %s: truncated copy" % t['name'])
        pad = (32 - (off % 32)) % 32
        out_data += b'\x00' * pad + data
        t['off'] = off + pad
        off = t['off'] + len(data)

    directory = bytearray()
    for t in new_tensors:
        nb = t['name'].encode('utf-8')
        directory += u64(len(nb)) + nb
        directory += u32(len(t['ne']))
        for d in t['ne']:
            directory += u64(d)
        directory += i32(t['dt']) + u64(t['off'])

    with open(args.out, 'wb') as f:
        f.write(bytes(header)); f.write(bytes(directory)); f.write(bytes(out_data))
    log("wrote %s (%d bytes; head %.2f MiB vs 48.25 MiB int8)"
        % (args.out, os.path.getsize(args.out),
           (args.k * E * 2 + V * 4 + V * 4) / 1048576))


if __name__ == '__main__':
    main()
