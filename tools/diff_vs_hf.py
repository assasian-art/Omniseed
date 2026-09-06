#!/usr/bin/env python3
# Layer-by-layer diff on ONE token (state zeros): my hand-rolled reference vs
# the official HF implementation. Prints per-block vector deltas to pinpoint
# the first op that diverges.
import sys
sys.path.insert(0, 'tools')
sys.path.insert(0, 'models')
sys.path.insert(0, 'models/hf-orig')

import numpy as np
import torch
from hf_rwkv_tokenizer import RwkvTokenizer
from transformers import AutoConfig, AutoModelForCausalLM
import convert_to_omniseed as C

torch.set_grad_enabled(False)

st_path = 'models/model.safetensors'
header, data_start = C.load_safetensors(st_path)
T = lambda k: torch.from_numpy(C.read_tensor(st_path, header, data_start, C.K(k)))

tok = RwkvTokenizer(vocab_file='models/rwkv_vocab_v20230424.txt')

cfg = AutoConfig.from_pretrained('models', trust_remote_code=True)
hf = AutoModelForCausalLM.from_pretrained(
    'models', config=cfg, trust_remote_code=True, torch_dtype=torch.float32)
hf.eval()

ids = tok.encode("\nIn a shocking finding, scientist discovered a herd of dragons")
if isinstance(ids, dict): ids = ids['input_ids']

# ------------------------- HF full-sequence forward -------------------------
out = hf(torch.tensor([ids]), output_hidden_states=True)
hs = out.hidden_states            # (emb+ln0?, block0..block11) from base model
hf_final = out.logits[0, -1]      # logits after last prompt token

# ------------------- my reference, token by token ---------------------------
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

mine_hidden = []   # x after each block
mine_logits = None

for tok_id in ids:
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
        r = xr @ T(p + 'att.receptance.weight').T
        k = xk @ T(p + 'att.key.weight').T
        v = xv @ T(p + 'att.value.weight').T
        w_log = -inv_sqe * torch.sigmoid(
            torch.tanh(xw @ T(p + 'att.w1')) @ T(p + 'att.w2') + T(p + 'att.w0')[0])
        a = torch.sigmoid(xa @ T(p + 'att.a1') @ T(p + 'att.a2') + T(p + 'att.a0')[0])
        g = torch.sigmoid(xg @ T(p + 'att.g1')) @ T(p + 'att.g2')
        if l == 0:
            v_first = v
        else:
            vg = torch.sigmoid(xv @ T(p + 'att.v1') @ T(p + 'att.v2') + T(p + 'att.v0')[0])
            v = v + (v_first - v) * vg
        kkk = k * T(p + 'att.k_k')[0]
        kk = kkk.view(H, D)
        kk = kk / torch.sqrt((kk * kk).sum(-1, keepdim=True) + 1e-12)
        k = k + k * (a - 1) * T(p + 'att.k_a')[0]
        k2 = k
        rh, kh, vh = (q.view(H, D) for q in (r, k2, v))
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
        att_out = ((yn + bonus) * g) @ T(p + 'att.output.weight').T
        x = x + att_out
        lx2 = ln(x, T(p + 'ln2.weight'), T(p + 'ln2.bias'))
        xk2 = lx2 + T(p + 'ffn.x_k')[0] * (ffn_prev[l] - lx2)
        ffn_prev[l] = lx2
        inner = (xk2 @ T(p + 'ffn.key.weight').T).relu() ** 2
        x = x + inner @ T(p + 'ffn.value.weight').T
        if tok_id == ids[-1]:
            mine_hidden.append(x.clone())
    if tok_id == ids[-1]:
        xl = ln(x, T('ln_out.weight'), T('ln_out.bias')).reshape(-1)
        head_key = 'head.weight' if 'head.weight' in header else C.K('head.weight')
        mine_logits = torch.from_numpy(
            C.read_tensor(st_path, header, data_start, head_key)).to(torch.float32) @ xl

# ------------------------------ compare -------------------------------------
print('=== last-token per-block hidden diffs (mine vs HF) ===')
# HF hidden_states: (after emb/ln0?, after block0..N). Align: hs[0]=emb side.
n = min(len(mine_hidden) + 1, len(hs))
for i in range(len(mine_hidden)):
    a = mine_hidden[i]
    b = hs[i + 1][0, -1]      # +1 to skip the embedding entry
    d = (a - b).abs().max().item()
    rel = d / (b.abs().max().item() + 1e-9)
    print(f'block {i:2d}: maxabs={d:.5f} rel={rel:.5f}')

print('=== final logits ===')
ta = mine_logits.topk(5)
tb = hf_final.topk(5)
print('mine :', [(int(i), round(float(v), 2)) for v, i in zip(ta.values, ta.indices)])
print('hf   :', [(int(i), round(float(v), 2)) for v, i in zip(tb.values, tb.indices)])
print('logit maxabs diff:', (mine_logits - hf_final).abs().max().item())
