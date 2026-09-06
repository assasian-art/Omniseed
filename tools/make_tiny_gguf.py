import struct

# Minimal valid OmniSeed GGUF to exercise GgufLoader + RwkvModel::load + forward.
E, D, R, L, V = 16, 8, 4, 2, 288   # n_embd, head_size (H=2), mlp_rank, layers, vocab
F32, F16, TERN = 0, 1, 40          # GGUF dtypes: TERNARY=40 is OmniSeed-specific

def s(x):  return struct.pack('<Q', len(x)) + x.encode()
def u32(v): return struct.pack('<I', v)
def u64(v): return struct.pack('<Q', v)
def f32(v): return struct.pack('<f', v)

tensors = []
def T(name, ne, dt, data): tensors.append((name, ne, dt, data))
def vec(name, n=E):
    T(name, [n], F32, struct.pack('<%df' % n, *([0.1] * n)))

# ---- metadata the loader requires (omniseed.* keys) ----
kv = []
def K_str(k, v): kv.append((k, 8, s(v)))
def K_u64(k, v): kv.append((k, 10, u64(v)))
def K_f64(k, v): kv.append((k, 12, struct.pack('<d', v)))

K_u64('omniseed.layer_count', L)
K_u64('omniseed.embedding_length', E)
K_u64('omniseed.vocab_size', V)
K_u64('omniseed.key_length', D)
K_u64('omniseed.mlp_rank', R)

# ---- tensors ----
T('token.embd', [V, E], F16, struct.pack('<%de' % (V * E), *([0.05] * (V * E))))
vec('blocks.0.ln0.weight'); vec('blocks.0.ln0.bias')
vec('ln_out.weight');       vec('ln_out.bias')
T('head.weight', [V, E], F16, struct.pack('<%de' % (V * E), *([0.02] * (V * E))))

for l in range(L):
    p = 'blocks.%d.' % l
    for ln in ('ln1', 'ln2'):
        vec(p + ln + '.weight'); vec(p + ln + '.bias')
    for m in ('tmix_r', 'tmix_w', 'tmix_k', 'tmix_v', 'tmix_a', 'tmix_g'):
        vec(p + 'att.' + m)
    # ternary weights [E,E] -> E*E/2 bytes; scale as f64 metadata
    nb = E * E // 2
    for w in ('receptance', 'key', 'value', 'output',
              'w1', 'w2', 'a1', 'a2', 'g1', 'g2', 'v1', 'v2'):
        T(p + 'att.' + w + '.weight', [E, E], TERN, b'\x15' * nb)  # 0x15=+1,0x01 pattern
        K_f64(p + 'att.' + w + '.scale', 0.25)
    vec(p + 'att.w_bias'); vec(p + 'att.a_bias'); vec(p + 'att.v_bias')
    vec(p + 'att.k_k');    vec(p + 'att.k_a');    vec(p + 'att.r_k')
    vec(p + 'att.gn.weight'); vec(p + 'att.gn.bias')
    vec(p + 'ffn.tmix_v')
    T(p + 'ffn.key.weight',   [E, E], TERN, b'\x15' * nb)
    T(p + 'ffn.value.weight', [E, E], TERN, b'\x15' * nb)
    K_f64(p + 'ffn.key.scale', 0.25); K_f64(p + 'ffn.value.scale', 0.25)

# ---- serialize ----
out = b'GGUF' + u32(3) + u64(len(tensors)) + u64(len(kv))
for k, t, v in kv: out += s(k) + u32(t) + v
for name, ne, dt, data in tensors:
    out += s(name) + u32(len(ne))
    for d in ne: out += u64(d)
    out += u32(dt) + u64(0)                      # placeholder offsets
data_start = len(out)
off = 0
parts = [out]
for name, ne, dt, data in tensors:
    parts.append(data); off += len(data)         # offsets assigned in order
blob = out + b''.join(p2 for p2 in parts[1:])

# second pass: rewrite offsets in-place is fiddly; simplest: rebuild with offsets
out = b'GGUF' + u32(3) + u64(len(tensors)) + u64(len(kv))
for k, t, v in kv: out += s(k) + u32(t) + v
o = 0
for i, (name, ne, dt, data) in enumerate(tensors):
    out += s(name) + u32(len(ne))
    for d in ne: out += u64(d)
    out += u32(dt) + u64(o)
    o += len(data)
blob = out + b''.join(t[3] for t in tensors)

open('build/tiny.gguf', 'wb').write(blob)
print('wrote build/tiny.gguf', len(blob), 'bytes; data_start=', data_start)
