#!/usr/bin/env python3
"""T120b: convert per-row int8 FFN linears -> grouped int4 (g=64).

Route per state/task_T120_brief.md:
  * in scope:  blocks.N.ffn.key.weight [ffn_inter, E]  and
               blocks.N.ffn.value.weight [E, ffn_inter]  (per-row int8)
  * out of scope: head (stays int8), attention linears, embedding, everything
    else — those are copied BYTE-FOR-BYTE from the source file.
  * every in-scope tensor gains a sibling `*.gscale` f32 tensor
    [out, in/64] = per-group absmax/7 scale; the weights become packed
    nibbles (low nibble = even column, two's complement -8..7), GGUF dtype 41
    (GgufDType::I4G64).

Fail-closed: refuses (exit 2, nothing written) when a source FFN tensor is
not per-row int8 with a matching per-row .scale, when in_dim % 64 != 0, or
when the output already exists without --force. The source GGUF is NEVER
modified: the output is a new file whose [0, kv_end) region is copied
verbatim (n_tensors patched) and whose tensor data is re-emitted.

Quantisation (deterministic, no RNG): per group g of 64 columns, on the
DEQUANTISED values v = i8 * row_scale:
    s_g = max|v| / 7   (0-safe: s_g = 1e-12 when the group is all zeros)
    q   = clamp(round(v / s_g), -7, 7)      # symmetric int4, -8 left unused
Round-half-away-from-zero, matching C++ std::lround used elsewhere.

Usage:
    python tools/convert_int4_ffn.py --src models/rwkv7-0.1B-ternary.gguf \
        --out models/rwkv7-0.1B-ternary-i4ffn.gguf [--layers 12] [--force]
"""
import argparse
import math
import os
import struct
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GGUF_I4G64 = 41        # GgufDType::I4G64 (include/omniseed/core/gguf_format.h)
GGUF_F32 = 0
GGUF_I8 = 4
GROUP = 64


def fail(msg):
    print("convert_int4_ffn: FAIL: %s" % msg, file=sys.stderr)
    sys.exit(2)


def r_u32(b, o): return struct.unpack_from('<I', b, o)[0]
def r_u64(b, o): return struct.unpack_from('<Q', b, o)[0]
def r_i32(b, o): return struct.unpack_from('<i', b, o)[0]
def u32(v): return struct.pack('<I', v)
def u64(v): return struct.pack('<Q', v)
def i32(v): return struct.pack('<i', v)


def r_str(b, o):
    n = r_u64(b, o); o += 8
    s = b[o:o + n].decode('utf-8'); o += n
    return s, o


def parse(b):
    if b[:4] != b'GGUF':
        fail("not a GGUF file")
    ver = r_u32(b, 4)
    n_tensors = r_u64(b, 8)
    n_kv = r_u64(b, 16)
    o = 24
    kv_types = {}
    for _ in range(n_kv):
        k, o = r_str(b, o)
        t = r_i32(b, o); o += 4
        kv_types[k] = t
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
                    fail("kv %s: unsupported array elem type %d" % (k, et))
                o += n * esz
        else:
            fail("kv %s: unsupported type %d" % (k, t))
    kv_end = o                      # <- everything before this is copied raw
    tensors = []
    for _ in range(n_tensors):
        name, o = r_str(b, o)
        nd = r_u32(b, o); o += 4
        ne = [r_u64(b, o + 8 * i) for i in range(nd)]; o += 8 * nd
        dt = r_i32(b, o); o += 4
        off = r_u64(b, o); o += 8
        tensors.append({'name': name, 'ne': ne, 'dt': dt, 'off': off})
    return ver, kv_end, tensors, o  # o = data_start


def rowmajor_from_ne(ne):
    return list(reversed(ne))


def quant_row_i8_to_i4g64(row_bytes, row_scale, cols):
    """One int8 row (+its per-row scale) -> packed nibbles + group scales."""
    assert cols % GROUP == 0
    q = [max(-127, min(127, v)) for v in row_bytes]
    v = [qv * row_scale for qv in q]
    packed = bytearray(cols // 2)
    gscales = []
    for g in range(cols // GROUP):
        lo, hi = g * GROUP, (g + 1) * GROUP
        m = 0.0
        for c in range(lo, hi):
            m = max(m, abs(v[c]))
        s = m / 7.0 if m > 0.0 else 1e-12
        gscales.append(s)
        for c in range(lo, hi):
            qi = int(math.floor(v[c] / s + 0.5)) if v[c] >= 0 \
                 else -int(math.floor(-v[c] / s + 0.5))  # round-half-away
            qi = max(-7, min(7, qi))
            b = qi & 0x0F
            idx = c - lo
            byte_i = lo // 2 + idx // 2
            if idx % 2 == 0:
                packed[byte_i] = b
            else:
                packed[byte_i] |= b << 4
    return bytes(packed), gscales


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='models/rwkv7-0.1B-ternary.gguf')
    ap.add_argument('--out', default='models/rwkv7-0.1B-ternary-i4ffn.gguf')
    ap.add_argument('--force', action='store_true',
                    help='overwrite an existing output')
    ap.add_argument('--layers', type=int, default=-1,
                    help='convert only the first N layers (-1 = all)')
    args = ap.parse_args()

    if not os.path.isfile(args.src):
        fail("source GGUF missing: %s" % args.src)
    if os.path.exists(args.out) and not args.force:
        fail("output exists (use --force): %s" % args.out)

    b = open(args.src, 'rb').read()
    ver, kv_end, tensors, data_start = parse(b)
    by_name = {t['name']: t for t in tensors}

    # locate in-scope FFN pairs (per-row int8 + per-row f32 scale)
    n_layers = 1 + max(int(t['name'].split('.')[1]) for t in tensors
                       if t['name'].startswith('blocks.') and
                       '.weight' in t['name'])
    layers = range(n_layers) if args.layers <= 0 else range(
        min(args.layers, n_layers))
    conv = {}   # name -> {'packed': bytes, 'gscale': [f32]}
    for l in layers:
        pfx = 'blocks.%d.ffn.' % l
        for part in ('key', 'value'):
            wname, sname = pfx + part + '.weight', pfx + part + '.scale'
            if wname not in by_name:
                fail("missing tensor %s" % wname)
            wt, st = by_name[wname], by_name.get(sname)
            if wt['dt'] != GGUF_I8:
                fail("%s: expected per-row int8 (dtype 4), got %d"
                     % (wname, wt['dt']))
            if st is None or st['dt'] != GGUF_F32:
                fail("%s: missing per-row f32 .scale tensor" % wname)
            rows, cols = rowmajor_from_ne(wt['ne'])
            if cols % GROUP != 0:
                fail("%s: cols=%d not divisible by group %d"
                     % (wname, cols, GROUP))
            nscale = rowmajor_from_ne(st['ne'])
            if math.prod(nscale) != rows:
                fail("%s: .scale numel != rows (%s vs %d)"
                     % (sname, nscale, rows))
            wbytes = b[data_start + wt['off']:
                       data_start + wt['off'] + rows * cols]
            if len(wbytes) != rows * cols:
                fail("%s: truncated tensor data" % wname)
            scales = struct.unpack_from('<%df' % rows, b,
                                        data_start + st['off'])
            # signed bytes
            row_i8 = memoryview(wbytes).cast('b')
            packed_all, gscale_all = bytearray(), []
            for r in range(rows):
                pk, gs = quant_row_i8_to_i4g64(
                    row_i8[r * cols:(r + 1) * cols], scales[r], cols)
                packed_all += pk
                gscale_all += gs
            conv[wname] = {
                'packed': bytes(packed_all),
                'gscale': struct.pack('<%df' % len(gscale_all),
                                      *gscale_all),
                'gshape_rowmajor': [rows, cols // GROUP],
            }
    if not conv:
        fail("nothing to convert (no FFN tensors found)")

    # ---- rebuild: [0, kv_end) copied verbatim (n_tensors patched) ----
    new_tensors = []
    for t in tensors:
        nt = dict(t)
        if t['name'] in conv:
            nt['dt'] = GGUF_I4G64
        new_tensors.append(nt)
        if t['name'] in conv:
            c = conv[t['name']]
            new_tensors.append({
                'name': t['name'].replace('.weight', '.gscale'),
                'ne': list(reversed(c['gshape_rowmajor'])),
                'dt': GGUF_F32, 'off': -1,
                'data': c['gscale'],
            })
    header = bytearray(b[:kv_end])
    header[8:16] = u64(len(new_tensors))    # patch n_tensors

    # data region: sequential 32-aligned offsets (this loader honors explicit
    # offsets relative to the post-directory position — same contract the
    # existing files use).
    out_data, off = bytearray(), 0
    for t in new_tensors:
        if t.get('data') is not None:
            data = t['data']
        elif t['name'] in conv:
            data = conv[t['name']]['packed']
        else:
            src_t = by_name[t['name']]
            numel = math.prod(src_t['ne'])
            dt = src_t['dt']
            if dt == 40:                       # TERNARY: 2 weights/byte
                nbytes = (numel + 1) // 2
            elif dt == GGUF_I4G64:             # nibbles: 2 weights/byte
                nbytes = (numel + 1) // 2
            else:
                esz = {GGUF_F32: 4, 1: 2, GGUF_I8: 1, 5: 4}.get(dt)
                if esz is None:
                    fail("tensor %s: unsupported dtype %d for copy"
                         % (t['name'], dt))
                nbytes = numel * esz
            data = b[data_start + src_t['off']:
                     data_start + src_t['off'] + nbytes]
            if len(data) != nbytes:
                fail("tensor %s: truncated copy" % t['name'])
        t['off'] = off
        pad = (32 - (off % 32)) % 32
        out_data += b'\x00' * pad + data
        off += pad + len(data)

    # tensor directory (name, n_dims, ne[], dtype, offset)
    directory = bytearray()
    for t in new_tensors:
        nb = t['name'].encode('utf-8')
        directory += u64(len(nb)) + nb
        directory += u32(len(t['ne']))
        for d in t['ne']:
            directory += u64(d)
        directory += i32(t['dt']) + u64(t['off'])

    with open(args.out, 'wb') as f:
        f.write(bytes(header))
        f.write(bytes(directory))
        f.write(bytes(out_data))

    total_in = sum(len(conv[n]['packed']) + len(conv[n]['gscale'])
                   for n in conv)
    print("convert_int4_ffn: %d FFN tensors -> I4G64 (%d layers), "
          "out=%s (%d bytes)" % (len(conv), len(layers), args.out,
                                 os.path.getsize(args.out)))
    print("convert_int4_ffn: int4 regions %.2f MiB (packed+gscales)"
          % (total_in / 1048576.0))


if __name__ == '__main__':
    main()
