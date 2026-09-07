#!/usr/bin/env python3
# Simulate the ternary quantization in the reference forward: same packing/
# scale math as the converter, but computed in torch fp32. If THIS is
# coherent, the C++ has a bug; if it is also degenerate, the quantization
# scheme itself is too lossy.
import sys
sys.path.insert(0, 'tools')
sys.path.insert(0, 'models')
sys.path.insert(0, 'models/hf-orig')

import numpy as np
import torch
from hf_rwkv_tokenizer import RwkvTokenizer
import convert_to_omniseed as C

torch.set_grad_enabled(False)
st_path = 'models/model.safetensors'
header, data_start = C.load_safetensors(st_path)

_cache = {}
MAIN_DTYPE = 'i8'    # i8 | ternary | fp16  (att.r/k/v/o, ffn.key/value)
LORA_DTYPE = 'i8'  # i8 | ternary | fp16  (att.w/a/g/v 1&2)

def QT(base):
    """Quantized tensor (per-row scale), dtype per MAIN/LORA_DTYPE policy."""
    key = 'q:' + base
    if key in _cache:
        return _cache[key]
    W = torch.from_numpy(C.read_tensor(st_path, header, data_start, C.K(base) if C.K(base) in header else base))
    out_dim, in_dim = W.shape
    is_lora = base.split('.')[-1] in ('w1', 'w2', 'a1', 'a2', 'g1', 'g2', 'v1', 'v2')
    policy = LORA_DTYPE if is_lora else MAIN_DTYPE
    if policy == 'fp16':
        q = W.half().float()
    elif policy == 'i8':
        s = W.abs().max(dim=1, keepdim=True).values.clamp_min(1e-12) / 127.0
        q = torch.round(W / s).clamp(-127, 127) * s
    else:  # ternary absmean (BitNet b1.58 PTQ)
        s = W.abs().mean(dim=1, keepdim=True)
        q = torch.round(torch.clamp(W / s.clamp_min(1e-12), -1, 1)) * s
    _cache[key] = q
    return q

def T(base):
    key = 'r:' + base
    if key not in _cache:
        k = C.K(base) if C.K(base) in header else base
        _cache[key] = torch.from_numpy(C.read_tensor(st_path, header, data_start, k))
    return _cache[key]

tok = RwkvTokenizer(vocab_file=C.resolve_vocab('models/rwkv_vocab_v20230424.txt'))

V, E = T('emb.weight').shape
n_layers = 12
H, D = T('blocks.0.att.r_k').shape
inv_sqe = 1.0 / np.sqrt(np.e)

def ln(x, w, b, eps=1e-5):
    m = x.mean(-1, keepdim=True)
    v = x.var(-1, keepdim=True, unbiased=False)
    return (x - m) / torch.sqrt(v + eps) * w + b

S = [torch.zeros(H, D, D) for _ in range(n_layers)]
att_prev = [torch.zeros(E) for _ in range(n_layers)]
ffn_prev = [torch.zeros(E) for _ in range(n_layers)]
v_first = None

ctx = "\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."
ids = tok.encode(ctx)
if isinstance(ids, dict): ids = ids['input_ids']
out_ids = []

import math
for step in range(len(ids) + 40):
    if step < len(ids):
        tok_id = ids[step]
    else:
        xl = ln(x, T('ln_out.weight'), T('ln_out.bias')).reshape(-1)
        logits = T('head.weight').to(torch.float32) @ xl
        nxt = int(torch.argmax(logits))
        out_ids.append(nxt)
        ids.append(nxt)
        tok_id = nxt
    x = T('emb.weight')[tok_id].to(torch.float32)
    x = ln(x, T('blocks.0.ln0.weight'), T('blocks.0.ln0.bias'))
    for l in range(n_layers):
        p = f'blocks.{l}.'
        lx = ln(x, T(p + 'ln1.weight'), T(p + 'ln1.bias'))
        d = att_prev[l] - lx
        xr = lx + T(p + 'att.x_r')[0] * d
        xw = lx + T(p + 'att.x_w')[0] * d
        xk = lx + T(p + 'att.x_k')[0] * d
        xv = lx + T(p + 'att.x_v')[0] * d
        xa = lx + T(p + 'att.x_a')[0] * d
        xg = lx + T(p + 'att.x_g')[0] * d
        att_prev[l] = lx
        r = xr @ QT(p + 'att.receptance.weight').T
        k = xk @ QT(p + 'att.key.weight').T
        v = xv @ QT(p + 'att.value.weight').T
        w_log = -inv_sqe * torch.sigmoid(
            torch.tanh(xw @ QT(p + 'att.w1')) @ QT(p + 'att.w2') + T(p + 'att.w0')[0])
        a = torch.sigmoid(xa @ QT(p + 'att.a1') @ QT(p + 'att.a2') + T(p + 'att.a0')[0])
        g = torch.sigmoid(xg @ QT(p + 'att.g1')) @ QT(p + 'att.g2')
        if l == 0:
            v_first = v
        else:
            vg = torch.sigmoid(xv @ QT(p + 'att.v1') @ QT(p + 'att.v2') + T(p + 'att.v0')[0])
            v = v + (v_first - v) * vg
        kkk = k * T(p + 'att.k_k')[0]
        kk = kkk.view(H, D)
        kk = kk / torch.sqrt((kk * kk).sum(-1, keepdim=True) + 1e-12)
        k = k + k * (a - 1) * T(p + 'att.k_a')[0]
        rh, kh, vh = (q.view(H, D) for q in (r, k, v))
        wh = torch.exp(w_log).view(H, D)
        ah = a.view(H, D)
        for h in range(H):
            sa = S[l][h].T @ (-kk[h])
            S[l][h] = wh[h][:, None] * S[l][h] \
                      + (kk[h] * ah[h])[:, None] * sa[None, :] \
                      + kh[h][:, None] * vh[h][None, :]
        y = torch.stack([rh[h] @ S[l][h] for h in range(H)], 0).reshape(E)
        yv = y.view(H, D)
        m_ = yv.mean(-1, keepdim=True)
        var = yv.var(-1, keepdim=True, unbiased=False)
        yn = ((yv - m_) / torch.sqrt(var + 64e-5)
              * T(p + 'att.ln_x.weight').view(H, D)
              + T(p + 'att.ln_x.bias').view(H, D)).reshape(E)
        bonus = ((rh * (k.view(H, D)) * T(p + 'att.r_k')).sum(-1, keepdim=True) * vh).reshape(E)
        att_out = ((yn + bonus) * g) @ QT(p + 'att.output.weight').T
        x = x + att_out
        lx2 = ln(x, T(p + 'ln2.weight'), T(p + 'ln2.bias'))
        xk2 = lx2 + T(p + 'ffn.x_k')[0] * (ffn_prev[l] - lx2)
        ffn_prev[l] = lx2
        inner = (xk2 @ QT(p + 'ffn.key.weight').T).relu() ** 2
        x = x + inner @ QT(p + 'ffn.value.weight').T

print('QUANT-REF-IDS :', out_ids)
print('QUANT-REF-TXT :', tok.decode(out_ids).encode('ascii', 'backslashreplace').decode('ascii'))
