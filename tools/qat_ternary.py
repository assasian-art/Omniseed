#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — qat_ternary.py   (OFFLINE training tool, never ships at runtime)
#
#  Phase 9 TASK 2: full-scale Quantization-Aware Training that turns the
#  RWKV-7 "World" 0.1B checkpoint's big linears into TRUE ternary {-1,0,+1}
#  weights (BitNet b1.58 style, per-row absmean scales).
#
#  Why: plain PTQ ternary is too lossy (verified in tools/quant_sim.py — the
#  model degenerates even with perfect fp32 math). QAT with a straight-through
#  estimator (STE) lets the network compensate during fine-tuning.
#
#  Method:
#    * Trainable: the 6 big linears per layer (att r/k/v/o + ffn key/value)
#      as fp32 masters; forward uses the TRUE ternary of the master (STE
#      backward). Before any training this eval equals PTQ ternary; after
#      training it is the QAT result — one code path, no mode switch.
#    * Frozen: emb, head, norms, token-shift vectors, LoRA factors, biases.
#    * Corpus (--corpus auto): wikitext-103-raw parquet from HuggingFace
#      (train split for training, validation split for eval), tokenized once
#      and cached to models/corpus/*.npy. Fallback: in-repo markdown.
#    * Batched windows: B independent windows per step (same sequential core,
#      bigger matmuls, several x throughput vs B=1). step_batch() is verified
#      numerically identical to the reference single-window step().
#    * AdamW + cosine LR schedule + warmup + gradient clipping.
#    * Chunked execution for background friendliness: --time-budget stops a
#      chunk and exits with code 3 ("more work remains"); tools/qat_watch.*
#      loops chunks until done (exit 0). Checkpoint (masters + optimizer +
#      global step + best ppl) persists across chunks.
#    * Progress appended to qat_log.txt every eval.
#    * Exports models/rwkv7-0.1B-ternary-qat.gguf (TERNARY linears, i8 LoRA
#      + head, everything else identical to the default converter output).
#
#  Usage (from repo root, venv with torch on PATH):
#    ./.venv/Scripts/python.exe tools/qat_ternary.py                # one chunk
#    tools/qat_watch.bat            (Windows)  /  tools/qat_watch.sh (Unix)
#
#  GPU (Colab): --device {auto,cpu,cuda}; full recipe in tools/COLAB_QAT.md.
#  Fresh clone: model.safetensors auto-downloads (HF Hakureirm/rwkv7-0.1b-hf)
#  and the world vocab resolves repo-relative (tools/data/ first).
#  --smoke: 2-step end-to-end self-check (no ckpt write, no export).
# =============================================================================
import argparse
import math
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_to_omniseed import (  # noqa: E402
    K, load_safetensors, read_tensor, resolve_vocab,
    GgufWriter, load_world_vocab, I8, TERNARY, F16, F32,
)

ST_DEFAULT = 'models/model.safetensors'
CORPUS_DIR = 'models/corpus'
WIKITEXT_TRAIN = ('https://huggingface.co/datasets/Salesforce/wikitext/'
                  'resolve/main/wikitext-103-raw-v1/'
                  'train-00000-of-00002.parquet')
WIKITEXT_VAL = ('https://huggingface.co/datasets/Salesforce/wikitext/'
                'resolve/main/wikitext-103-raw-v1/'
                'validation-00000-of-00001.parquet')
REPO_FILES = ['PROJECT_STATE.md', 'docs/OMNISEED_MASTER_SPEC.md', 'README.md']
HF_MODEL_REPO = 'Hakureirm/rwkv7-0.1b-hf'   # public HF repo, no login needed

INV_SQ_E = 1.0 / math.sqrt(math.e)
EXIT_DONE, EXIT_MORE = 0, 3


# -----------------------------------------------------------------------------
# Corpus: wikitext-103-raw (auto-download + token cache), in-repo fallback.
# -----------------------------------------------------------------------------
def _download(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f'[qat] downloading {url}', flush=True)
    import urllib.request
    tmp = dest + '.part'
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)
    return dest


def ensure_checkpoint(path):
    """Fresh-clone bootstrap (Colab): when the safetensors is missing, pull
    the public HF checkpoint (safetensors + config.json) with plain urllib —
    no interactive login. models/hf-orig/config.json is committed, so this
    normally only fetches the ~380 MB weights once."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    import urllib.request
    base = f'https://huggingface.co/{HF_MODEL_REPO}/resolve/main'
    cfg = os.path.join(os.path.dirname(path) or '.', 'hf-orig', 'config.json')
    if not os.path.exists(cfg):
        os.makedirs(os.path.dirname(cfg) or '.', exist_ok=True)
        print(f'[qat] downloading {base}/config.json', flush=True)
        urllib.request.urlretrieve(f'{base}/config.json', cfg + '.part')
        os.replace(cfg + '.part', cfg)
    print(f'[qat] downloading {base}/model.safetensors (~380 MB, one-time)',
          flush=True)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = path + '.part'
    urllib.request.urlretrieve(f'{base}/model.safetensors', tmp)
    os.replace(tmp, path)
    print(f'[qat] checkpoint ready: {path} '
          f'({os.path.getsize(path) / 1048576:.1f} MB)', flush=True)


def _parquet_lines(path):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=2048, columns=['text']):
        for t in batch.column('text').to_pylist():
            if t:
                yield t


def _tokenize_stream(lines, tok, cap_tokens):
    """Tokenize a line stream up to cap_tokens; returns int32 np.array."""
    ids, buf, count = [], [], 0
    for line in lines:
        buf.append(line.strip())
        if sum(len(b) for b in buf) >= 4096:
            chunk = '\n'.join(buf)
            buf = []
            got = tok.encode(chunk)
            ids.extend(got)
            count += len(got)
            if count >= cap_tokens:
                break
    for b in buf:
        ids.extend(tok.encode(b))
    return np.asarray(ids, dtype=np.int32)


def load_corpus_wikitext(tok, max_train_tokens, cache=True):
    os.makedirs(CORPUS_DIR, exist_ok=True)
    train_np, val_np = (os.path.join(CORPUS_DIR, 'train.npy'),
                        os.path.join(CORPUS_DIR, 'val.npy'))
    if cache and os.path.exists(train_np) and os.path.exists(val_np):
        print(f'[qat] corpus cache hit: {train_np} / {val_np}', flush=True)
        return (np.load(train_np), np.load(val_np))
    tr = _tokenize_stream(_parquet_lines(_download(WIKITEXT_TRAIN,
                                                   CORPUS_DIR + '/train.parquet')),
                          tok, max_train_tokens)
    va = _tokenize_stream(_parquet_lines(_download(WIKITEXT_VAL,
                                                   CORPUS_DIR + '/val.parquet')),
                          tok, 400_000)
    print(f'[qat] wikitext-103: train {len(tr):,} tokens, '
          f'val {len(va):,} tokens', flush=True)
    if cache:
        np.save(train_np, tr)
        np.save(val_np, va)
    return tr, va


def load_corpus_repo(tok):
    """In-repo markdown fallback (Phase 8 PoC corpus). 90/10 by paragraph."""
    paras = []
    for p in REPO_FILES:
        if not os.path.exists(p):
            continue
        text = open(p, 'r', encoding='utf-8', errors='replace').read()
        for chunk in text.split('\n\n'):
            chunk = chunk.strip()
            if len(chunk) >= 120:
                paras.append(chunk)
    random.seed(7)
    random.shuffle(paras)
    n_val = max(4, len(paras) // 10)
    train = tok.encode('\n\n'.join(paras[n_val:]))
    val = tok.encode('\n\n'.join(paras[:n_val]))
    return np.asarray(train, dtype=np.int32), np.asarray(val, dtype=np.int32)


class TernarySTE(torch.autograd.Function):
    """Forward: true ternary {-s,0,+s} of the master. Backward: identity."""
    @staticmethod
    def forward(ctx, W):
        s = W.abs().mean(dim=-1, keepdim=True).clamp_min(1e-12)
        t = torch.clamp(torch.round(W / s), -1.0, 1.0)
        return t * s

    @staticmethod
    def backward(ctx, g):
        return g


class RWKV7Ternary:
    """RWKV-7 forward mirroring convert_to_omniseed.reference_generate
    (HF-verified), with ternary-STE linears for r/k/v/o + ffn key/value.

    step()      — reference single-window path (Phase-8-verified).
    step_batch()— B independent windows; verified equal to step() at B=1.
    """

    def __init__(self, st_path, header, data_start):
        import torch
        self.torch = torch
        self.header, self.ds, self.st = header, data_start, st_path
        self.cache = {}
        self.device = torch.device('cpu')   # moved by to_device() in main

        V, E = self.T(K('emb.weight')).shape
        n_layers = max(int(k.split('.')[2]) for k in header if '.blocks.' in k) + 1
        H, D = self.T(K('blocks.0.att.r_k')).shape
        self.V, self.E, self.L, self.H, self.D = V, E, n_layers, H, D
        print(f'[qat] V={V} E={E} L={n_layers} H={H} D={D}')

        # Frozen fp32 parameters.
        self.p = {}
        self.p['emb'] = self.T(K('emb.weight')).float()
        # NB: this checkpoint family stores 1-D params with a leading [1, ...]
        # dim (att.ln_x is [1,1,768], ln_out may be [1,768]); flatten norms so
        # ln() never broadcasts x into a phantom leading axis.
        self.p['ln0w'] = self.T(K('blocks.0.ln0.weight')).float().reshape(-1)
        self.p['ln0b'] = self.T(K('blocks.0.ln0.bias')).float().reshape(-1)
        self.p['lnow'] = self.T(K('ln_out.weight')).float().reshape(-1)
        self.p['lnob'] = self.T(K('ln_out.bias')).float().reshape(-1)
        hkey = K('head.weight') if K('head.weight') in header else 'head.weight'
        self.p['head'] = self.T(hkey).float()
        for l in range(n_layers):
            s, d = K(f'blocks.{l}.'), f'l{l}.'
            for a in ('ln1.weight', 'ln1.bias', 'ln2.weight', 'ln2.bias'):
                self.p[d + a] = self.T(s + a).float().reshape(-1)
            # x_* / w0 / a0 / v0 / k_k / k_a are [1,1,768]: reshape to [768]
            for a in ('x_r', 'x_w', 'x_k', 'x_v', 'x_a', 'x_g'):
                self.p[d + a] = self.T(s + 'att.' + a).float().reshape(-1)
            for a in ('w1', 'w2', 'a1', 'a2', 'g1', 'g2', 'v1', 'v2'):
                self.p[d + a] = self.T(s + 'att.' + a).float()
            for a in ('w0', 'a0', 'v0'):
                self.p[d + a] = self.T(s + 'att.' + a).float().reshape(-1)
            for a in ('k_k', 'k_a'):
                self.p[d + a] = self.T(s + 'att.' + a).float().reshape(-1)
            self.p[d + 'r_k'] = self.T(s + 'att.r_k').float()
            self.p[d + 'lnxw'] = self.T(s + 'att.ln_x.weight').float().view(H, D)
            self.p[d + 'lnxb'] = self.T(s + 'att.ln_x.bias').float().view(H, D)
            self.p[d + 'fxk'] = self.T(s + 'ffn.x_k').float().reshape(-1)

        # Trainable fp32 masters, initialised at the bf16 weights. The forward
        # ternarizes them, so step 0 == PTQ ternary exactly.
        self.masters = {}
        for l in range(n_layers):
            for name in ('att.receptance', 'att.key', 'att.value', 'att.output',
                         'ffn.key', 'ffn.value'):
                full = f'l{l}.{name}'
                W0 = self.T(K(f'blocks.{l}.{name}.weight')).float()
                self.masters[full] = W0.clone().requires_grad_(True)
        self.master_list = list(self.masters.values())

    def to_device(self, device):
        """Move frozen params + trainable masters to device. Masters are
        rebuilt as fresh leaf tensors (still valid AdamW params), so this
        MUST run before the optimizer is created. On cpu every .to() is a
        no-op — the CPU path stays bit-identical."""
        self.device = device
        self.p = {k: v.to(device) for k, v in self.p.items()}
        self.masters = {n: t_.detach().to(device).requires_grad_(True)
                        for n, t_ in self.masters.items()}
        self.master_list = list(self.masters.values())

    def T(self, k):
        if k not in self.cache:
            self.cache[k] = self.torch.from_numpy(
                read_tensor(self.st, self.header, self.ds, k))
        return self.cache[k]

    def lin(self, name, x):
        """Batched x @ Wq.T. float_mode (eval-only): raw bf16 master — the
        honest baseline that validates the forward pass itself."""
        W = self.masters[name] if getattr(self, 'float_mode', False) \
            else self.Wq[name]
        return x @ W.t()

    def begin_window(self, use_grad):
        """Ternarize all masters ONCE per window-batch. Weights only change
        after an optimizer step, so per-token ternarization is pure overhead.
        The graph stays alive across the window so STE gradients reach the
        masters at backward."""
        if use_grad:
            self.Wq = {n: TernarySTE.apply(self.masters[n])
                       for n in self.masters}
        else:
            with torch.no_grad():
                self.Wq = {n: TernarySTE.apply(self.masters[n])
                           for n in self.masters}

    def end_window(self):
        self.Wq = None
        self.float_mode = False

    # -- reference single-window path (verified against the HF oracle) --------
    def step(self, tok, state):
        t = self.torch
        p, E, H, D, L = self.p, self.E, self.H, self.D, self.L

        def ln(x, w, b, eps=1e-5):
            m = x.mean(-1, keepdim=True)
            v = x.var(-1, keepdim=True, unbiased=False)
            return (x - m) / t.sqrt(v + eps) * w + b

        if state is None:
            state = [(t.zeros(H, D, D, device=self.device),
                      t.zeros(E, device=self.device),
                      t.zeros(E, device=self.device)) for _ in range(L)]
        x = p['emb'][tok]
        x = ln(x, p['ln0w'], p['ln0b'])
        v_first = None
        for l in range(L):
            S, att_prev, ffn_prev = state[l]
            d = f'l{l}.'
            lx = ln(x, p[d + 'ln1.weight'], p[d + 'ln1.bias'])
            dx = att_prev - lx
            xr = lx + p[d + 'x_r'] * dx
            xw = lx + p[d + 'x_w'] * dx
            xk = lx + p[d + 'x_k'] * dx
            xv = lx + p[d + 'x_v'] * dx
            xa = lx + p[d + 'x_a'] * dx
            xg = lx + p[d + 'x_g'] * dx
            att_prev = lx

            r = self.lin(f'l{l}.att.receptance', xr)
            k = self.lin(f'l{l}.att.key', xk)
            v = self.lin(f'l{l}.att.value', xv)

            w_log = -INV_SQ_E * t.sigmoid(
                t.tanh(xw @ p[d + 'w1']) @ p[d + 'w2'] + p[d + 'w0'])
            a = t.sigmoid(xa @ p[d + 'a1'] @ p[d + 'a2'] + p[d + 'a0'])
            g = t.sigmoid(xg @ p[d + 'g1']) @ p[d + 'g2']
            if l == 0:
                v_first = v
            else:
                vg = t.sigmoid(xv @ p[d + 'v1'] @ p[d + 'v2'] + p[d + 'v0'])
                v = v + (v_first - v) * vg

            kkk = k * p[d + 'k_k']                       # pre-blend k
            kk = kkk.view(H, D)
            kk = kk / t.sqrt((kk * kk).sum(-1, keepdim=True) + 1e-12)
            k = k + k * (a - 1) * p[d + 'k_a']           # blend BEFORE bonus
            rh, kh, vh = r.view(H, D), k.view(H, D), v.view(H, D)
            wh = t.exp(w_log).view(H, D)
            ah = a.view(H, D)

            # sa[h,j] = -sum_i kk[h,i]*S[h,i,j]  (HF reference: S[h].T @ -kk;
            # the MINUS is load-bearing — dropping it explodes the state).
            sa = -t.einsum('hi,hij->hj', kk, S)                         # [H, D]
            S = (wh.unsqueeze(-1) * S
                 + (kk * ah).unsqueeze(-1) * sa.unsqueeze(1)
                 + kh.unsqueeze(-1) * vh.unsqueeze(1))

            y = t.einsum('hi,hij->hj', rh, S).reshape(E)
            yv = y.view(H, D)
            m_ = yv.mean(-1, keepdim=True)
            var = yv.var(-1, keepdim=True, unbiased=False)
            yn = ((yv - m_) / t.sqrt(var + 64e-5) * p[d + 'lnxw']
                  + p[d + 'lnxb']).reshape(E)
            bonus = ((rh * (k.view(H, D)) * p[d + 'r_k']).sum(-1, keepdim=True)
                     * vh).reshape(E)
            x = x + self.lin(f'l{l}.att.output', (yn + bonus) * g)

            lx2 = ln(x, p[d + 'ln2.weight'], p[d + 'ln2.bias'])
            xk2 = lx2 + p[d + 'fxk'] * (ffn_prev - lx2)
            ffn_prev = lx2
            inner = self.lin(f'l{l}.ffn.key', xk2).relu() ** 2
            x = x + self.lin(f'l{l}.ffn.value', inner)
            state[l] = (S, att_prev, ffn_prev)

        x_last = ln(x, p['lnow'], p['lnob'])
        logits = p['head'] @ x_last
        return logits, state

    # -- batched path: B independent windows ----------------------------------
    def step_batch(self, toks, state):
        """toks: [B] int64. state: list of (S[B,H,D,D], att[B,E], ffn[B,E]).
        Returns logits [B,V] and the new state."""
        t = self.torch
        p, E, H, D, L = self.p, self.E, self.H, self.D, self.L

        def ln(x, w, b, eps=1e-5):
            m = x.mean(-1, keepdim=True)
            v = x.var(-1, keepdim=True, unbiased=False)
            return (x - m) / t.sqrt(v + eps) * w + b

        B = toks.shape[0]
        if state is None:
            state = [(t.zeros(B, H, D, D, device=self.device),
                      t.zeros(B, E, device=self.device),
                      t.zeros(B, E, device=self.device)) for _ in range(L)]
        x = p['emb'][toks]                                  # [B, E]
        x = ln(x, p['ln0w'], p['ln0b'])
        v_first = None
        for l in range(L):
            S, att_prev, ffn_prev = state[l]
            d = f'l{l}.'
            lx = ln(x, p[d + 'ln1.weight'], p[d + 'ln1.bias'])
            dx = att_prev - lx
            xr = lx + p[d + 'x_r'] * dx
            xw = lx + p[d + 'x_w'] * dx
            xk = lx + p[d + 'x_k'] * dx
            xv = lx + p[d + 'x_v'] * dx
            xa = lx + p[d + 'x_a'] * dx
            xg = lx + p[d + 'x_g'] * dx
            att_prev = lx

            r = self.lin(f'l{l}.att.receptance', xr)        # [B, E]
            k = self.lin(f'l{l}.att.key', xk)
            v = self.lin(f'l{l}.att.value', xv)

            w_log = -INV_SQ_E * t.sigmoid(
                t.tanh(xw @ p[d + 'w1']) @ p[d + 'w2'] + p[d + 'w0'])
            a = t.sigmoid(xa @ p[d + 'a1'] @ p[d + 'a2'] + p[d + 'a0'])
            g = t.sigmoid(xg @ p[d + 'g1']) @ p[d + 'g2']
            if l == 0:
                v_first = v
            else:
                vg = t.sigmoid(xv @ p[d + 'v1'] @ p[d + 'v2'] + p[d + 'v0'])
                v = v + (v_first - v) * vg

            kkk = k * p[d + 'k_k']
            kk = kkk.view(B, H, D)
            kk = kk / t.sqrt((kk * kk).sum(-1, keepdim=True) + 1e-12)
            k = k + k * (a - 1) * p[d + 'k_a']
            rh, kh, vh = r.view(B, H, D), k.view(B, H, D), v.view(B, H, D)
            wh = t.exp(w_log).view(B, H, D)
            ah = a.view(B, H, D)

            sa = -t.einsum('bhid,bhi->bhd', S, kk)          # [B, H, D]
            S = (wh.unsqueeze(-1) * S
                 + t.einsum('bhi,bhj->bhij', kk * ah, sa)
                 + t.einsum('bhi,bhj->bhij', kh, vh))       # BLENDED k, like ref

            y = t.einsum('bhi,bhij->bhj', rh, S).reshape(B, E)
            yv = y.view(B, H, D)
            m_ = yv.mean(-1, keepdim=True)
            var = yv.var(-1, keepdim=True, unbiased=False)
            yn = ((yv - m_) / t.sqrt(var + 64e-5) * p[d + 'lnxw']
                  + p[d + 'lnxb']).reshape(B, E)
            bonus = ((rh * (k.view(B, H, D)) * p[d + 'r_k'])
                     .sum(-1, keepdim=True) * vh).reshape(B, E)
            x = x + self.lin(f'l{l}.att.output', (yn + bonus) * g)

            lx2 = ln(x, p[d + 'ln2.weight'], p[d + 'ln2.bias'])
            xk2 = lx2 + p[d + 'fxk'] * (ffn_prev - lx2)
            ffn_prev = lx2
            inner = self.lin(f'l{l}.ffn.key', xk2).relu() ** 2
            x = x + self.lin(f'l{l}.ffn.value', inner)
            state[l] = (S, att_prev, ffn_prev)

        x_last = ln(x, p['lnow'], p['lnob'])
        logits = x_last @ p['head'].t()                     # [B, V]
        return logits, state


def main():
    torch.set_grad_enabled(True)
    # Tiny-op pathology: multi-thread fan-out on 768-wide ops is ~5x SLOWER
    # on this box (Windows futex churn). Single thread unless overridden.
    torch.set_num_threads(int(os.environ.get('OMNISEED_TORCH_THREADS', '1')))

    ap = argparse.ArgumentParser()
    ap.add_argument('--safetensors', default=ST_DEFAULT)
    ap.add_argument('--corpus', default='auto', choices=['auto', 'wikitext', 'repo'])
    ap.add_argument('--max-train-tokens', type=int, default=3_000_000)
    ap.add_argument('--steps', type=int, default=10_000,
                    help='total training steps (across chunks)')
    ap.add_argument('--start-step', type=int, default=0,
                    help='override the checkpoint global step (usually left 0; '
                         'resume supplies it)')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lr-min', type=float, default=1e-5)
    ap.add_argument('--warmup', type=int, default=100)
    ap.add_argument('--window', type=int, default=32)
    ap.add_argument('--batch', type=int, default=8,
                    help='independent windows per step')
    ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--eval-every', type=int, default=200)
    ap.add_argument('--eval-tokens', type=int, default=2048,
                    help='perplexity estimate length (32 parallel segments)')
    ap.add_argument('--time-budget', type=int, default=0,
                    help='seconds; stop the chunk early and exit 3 (0 = off)')
    ap.add_argument('--ckpt', default='models/qat_ckpt.pt',
                    help='checkpoint: masters + optimizer + global step + best')
    ap.add_argument('--out', default='models/rwkv7-0.1B-ternary-qat.gguf')
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--log-file', default='qat_log.txt')
    ap.add_argument('--export-always', action='store_true',
                    help='export the GGUF even when the chunk ends early')
    ap.add_argument('--no-export', action='store_true')
    ap.add_argument('--verify-batch', action='store_true',
                    help='assert step_batch(B=1) == step() and exit')
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                    help='compute device; auto = cuda when available, else cpu')
    ap.add_argument('--smoke', action='store_true',
                    help='2-step end-to-end smoke (no ckpt write, no export)')
    args = ap.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        print('[qat] WARNING: cuda requested but torch sees no CUDA runtime; '
              'continuing on cpu')
        device = torch.device('cpu')
    print('[qat] device: ' + str(device)
          + (f' ({torch.cuda.get_device_name(0)})' if device.type == 'cuda'
             else ''))
    if args.smoke:
        args.steps = min(args.steps, 2)
        args.eval_every = 0
        args.no_export = True
        print('[qat] SMOKE: 2 steps, eval off, no checkpoint write / export')

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    sys.path.insert(0, 'models/hf-orig')
    from hf_rwkv_tokenizer import RwkvTokenizer
    tok = RwkvTokenizer(vocab_file=resolve_vocab())

    if args.corpus in ('auto', 'wikitext'):
        try:
            train_ids, val_ids = load_corpus_wikitext(tok, args.max_train_tokens)
        except Exception as e:                               # noqa: BLE001
            print(f'[qat] wikitext unavailable ({e}); falling back to repo corpus')
            if args.corpus == 'wikitext':
                raise
            train_ids, val_ids = load_corpus_repo(tok)
    else:
        train_ids, val_ids = load_corpus_repo(tok)
    print(f'[qat] corpus: train {len(train_ids):,} tokens, '
          f'val {len(val_ids):,} tokens')

    ensure_checkpoint(args.safetensors)
    header, data_start = load_safetensors(args.safetensors)
    model = RWKV7Ternary(args.safetensors, header, data_start)
    model.to_device(device)
    optimizer = torch.optim.AdamW(model.master_list, lr=args.lr,
                                  weight_decay=0.0, betas=(0.9, 0.95))

    def cosine_lr(step):
        if step < args.warmup:
            return args.lr * (step + 1) / max(1, args.warmup)
        t_ = (step - args.warmup) / max(1, args.steps - args.warmup)
        t_ = min(1.0, t_)
        return args.lr_min + 0.5 * (args.lr - args.lr_min) * (1 + math.cos(math.pi * t_))

    global_step = args.start_step
    best = {'ppl': float('inf'), 'step': -1}
    if os.path.exists(args.ckpt) and args.start_step == 0 and not args.smoke:
        ck = torch.load(args.ckpt, map_location='cpu', weights_only=True)
        with torch.no_grad():
            for n, t_ in ck['masters'].items():
                model.masters[n].copy_(t_)
        if 'optim' in ck:
            optimizer.load_state_dict(ck['optim'])
        global_step = ck['global_step']
        best = {'ppl': ck.get('best_ppl', float('inf')),
                'step': ck.get('best_step', -1)}
        print(f'[qat] resumed from {args.ckpt} (global step {global_step}, '
              f'best val ppl {best["ppl"]:.2f})')

    def log_line(text):
        print(text, flush=True)
        try:
            with open(args.log_file, 'a', encoding='utf-8') as f:
                stamp = time.strftime('%Y-%m-%d %H:%M:%S')
                f.write(f'[{stamp}] {text}\n')
        except OSError:
            pass

    def perplexity(ids, float_masters=False, segments=64, max_tok=16384):
        """Teacher-forced NLL on val tokens, 64 parallel segments (much
        larger effective sample than a single 2048-token chain)."""
        torch.set_grad_enabled(False)
        model.float_mode = float_masters
        model.begin_window(False)
        W = args.window
        n_all = min(len(ids) - 1, max_tok)
        seg_len = max(W + 2, (n_all + segments - 1) // segments)
        n_seg = min(segments, (len(ids) - 1) // seg_len)
        use = min(seg_len - 1, max(W + 1, 512))
        total_nll, n_pred = 0.0, 0
        state = None
        for j in range(use):
            toks = torch.tensor(
                [int(ids[b * seg_len + j]) for b in range(n_seg)],
                dtype=torch.long, device=device)
            logits, state = model.step_batch(toks, state)
            if j >= W:
                logp = torch.log_softmax(logits.float(), -1)   # [B, V]
                for b in range(n_seg):
                    total_nll += -logp[b, int(ids[b * seg_len + j + 1])].item()
                n_pred += n_seg
        model.end_window()
        torch.set_grad_enabled(True)
        return math.exp(total_nll / max(1, n_pred))

    def evaluate_and_log(tag):
        ppl_bf16 = perplexity(train_ids, float_masters=True)
        ppl_val = perplexity(val_ids)
        log_line(f'[qat] {tag}: bf16(train) ppl={ppl_bf16:.2f} '
                 f'ternary val ppl={ppl_val:.2f} '
                 f'ratio={ppl_val / max(ppl_bf16, 1e-9):.2f}x '
                 f'best={min(best["ppl"], ppl_val):.2f}')
        return ppl_bf16, ppl_val

    if args.verify_batch:
        ref_ids = [int(x) for x in train_ids[:24]]
        s1, st1 = None, None
        model.float_mode = False
        model.begin_window(False)
        for i in ref_ids:
            lg, st1 = model.step(i, st1)
        s2, st2 = None, None
        for i in ref_ids:
            lg2, st2 = model.step_batch(
                torch.tensor([i], dtype=torch.long, device=device), st2)
        d = (lg - lg2).abs().max().item()
        smax = max((a - b).abs().max().item()
                   for (a, _, _), (b, _, _) in zip(st1, st2))
        print(f'[verify] max|dlogits|={d:.2e} max|dstate|={smax:.2e} '
              f'{"OK" if d < 1e-3 and smax < 1e-3 else "MISMATCH"}')
        sys.exit(0 if d < 1e-3 and smax < 1e-3 else 1)

    if args.eval_every > 0 and global_step == 0 and not os.path.exists(args.ckpt):
        evaluate_and_log('baseline')

    started = time.time()
    W, B = args.window, args.batch
    n_starts = max(1, len(train_ids) - W - 2)

    def lr_at(step):
        lr = cosine_lr(step)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        return lr

    while global_step < args.steps:
        lr_at(global_step)
        # B independent CONTIGUOUS windows; element b runs tokens
        # starts[b] .. starts[b]+W-1 through its own RNN state.
        starts = [(global_step * 7919 + b * 104729) % n_starts
                  for b in range(B)]
        model.begin_window(True)
        state = None
        total_loss = torch.zeros((), device=device)
        for j in range(W):
            toks = torch.tensor([int(train_ids[s + j]) for s in starts],
                                dtype=torch.long, device=device)
            logits, state = model.step_batch(toks, state)
            tgt = torch.tensor([int(train_ids[s + j + 1]) for s in starts],
                               dtype=torch.long, device=device)
            loss = torch.nn.functional.cross_entropy(logits.float(), tgt)
            total_loss = total_loss + loss
            state = [(S.detach(), a.detach(), f.detach()) for S, a, f in state]
        (total_loss / W).backward()
        torch.nn.utils.clip_grad_norm_(model.master_list, args.clip)
        optimizer.step()
        optimizer.zero_grad()
        model.end_window()
        global_step += 1

        if args.eval_every > 0 and (global_step % args.eval_every == 0
                                    or global_step >= args.steps):
            ppl_val = perplexity(val_ids)
            if ppl_val < best['ppl']:
                best = {'ppl': ppl_val, 'step': global_step}
                torch.save({n: t_.detach().cpu().clone() for n, t_ in model.masters.items()},
                           args.ckpt.replace('.pt', '-best.pt'))
            log_line(f'[qat] step {global_step}/{args.steps} '
                     f'loss {(total_loss / W).item():.3f} lr={lr_at(global_step):.2e} '
                     f'val_ppl={ppl_val:.2f} best={best["ppl"]:.2f}'
                     f'@{best["step"]}')

        if args.time_budget and time.time() - started > args.time_budget \
                and global_step < args.steps:
            log_line(f'[qat] chunk time budget reached at step {global_step}')
            break

    # ---- persist checkpoint (masters + optimizer + progress + best) ----------
    if not args.smoke:
        torch.save({'masters': {n: t_.detach().cpu().clone()
                                for n, t_ in model.masters.items()},
                    'optim': optimizer.state_dict(),
                    'global_step': global_step,
                    'best_ppl': best['ppl'], 'best_step': best['step']},
                   args.ckpt)
        log_line(f'[qat] checkpoint saved: {args.ckpt} (step {global_step}, '
                 f'best {best["ppl"]:.2f}@{best["step"]})')

    done = global_step >= args.steps
    if done or args.export_always:
        if not args.no_export:
            if best['ppl'] < float('inf') and os.path.exists(
                    args.ckpt.replace('.pt', '-best.pt')):
                bb = torch.load(args.ckpt.replace('.pt', '-best.pt'),
                                map_location='cpu', weights_only=True)
                with torch.no_grad():
                    for n, t_ in bb.items():
                        model.masters[n].copy_(t_)
                log_line(f'[qat] exporting BEST masters '
                         f'(ppl {best["ppl"]:.2f} @ step {best["step"]})')
            export_gguf(model, args)
    if args.eval_every > 0 and (done or args.export_always):
        evaluate_and_log(f'final @ step {global_step}')
    sys.exit(EXIT_DONE if done else EXIT_MORE)


def export_gguf(model, args):
    header, data_start = load_safetensors(args.safetensors)
    keys = [k for k in header if k != '__metadata__']
    V, E = header[K('emb.weight')]['shape']
    n_layers = max(int(k.split('.')[2]) for k in keys if '.blocks.' in k) + 1
    H, D = header[K('blocks.0.att.r_k')]['shape']
    I = header[K('blocks.0.ffn.key.weight')]['shape'][0]
    rw = header[K('blocks.0.att.w2')]['shape'][0]
    ra = header[K('blocks.0.att.a2')]['shape'][0]
    rg = header[K('blocks.0.att.g2')]['shape'][0]
    rv = header[K('blocks.0.att.v2')]['shape'][0]

    pieces, types = load_world_vocab(resolve_vocab(), V)
    w = GgufWriter()
    w.add_str('general.architecture', 'omniseed-rwkv7')
    w.add_str('general.name', 'RWKV7-World-0.1B-ternary-QAT')
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
    w.add_i32('omniseed.bos_token_id', -1)
    w.add_i32('omniseed.eos_token_id', 0)
    w.add_i32('omniseed.user_start_token_id', -1)
    w.add_i32('omniseed.user_end_token_id', -1)
    w.add_i32('omniseed.assistant_start_token_id', -1)
    w.add_i32('omniseed.assistant_end_token_id', -1)
    w.add_str_array('tokenizer.ggml.tokens', pieces)
    w.add_i32_array('tokenizer.ggml.token_type', types)

    T = lambda k: read_tensor(args.safetensors, header, data_start, k)  # noqa: E731
    w.add_tensor('token.embd', (V, E), F16,
                 np.ascontiguousarray(T(K('emb.weight')), '<f2').tobytes())
    if K('blocks.0.ln0.weight') in header:
        for nm in ('blocks.0.ln0.weight', 'blocks.0.ln0.bias'):
            w.add_tensor(nm, (E,), F32,
                         np.ascontiguousarray(T(K(nm)), '<f4').tobytes())

    def add_f32(name, arr):
        w.add_tensor(name, arr.shape, F32,
                     np.ascontiguousarray(arr, '<f4').tobytes())

    def add_i8(name, W):
        s = np.maximum(np.abs(W).max(axis=1) / 127.0, 1e-12).astype('<f4')
        q = np.clip(np.round(W / s[:, None]), -127, 127).astype(np.int8)
        w.add_tensor(name + '.weight', W.shape, I8, q.tobytes())
        w.add_tensor(name + '.scale', (W.shape[0],), F32, s.tobytes())

    def add_ternary_qat(name, W_torch):
        """Pack the QAT master's true ternary (absmean per-row scales)."""
        W = W_torch.detach().cpu().numpy().astype(np.float32)
        s = np.maximum(np.abs(W).mean(axis=1), 1e-12).astype('<f4')
        t = np.clip(np.round(W / s[:, None]), -1, 1).astype(np.int8)
        row_bytes = (W.shape[1] + 1) // 2
        packed = np.zeros((W.shape[0], row_bytes), dtype=np.uint8)
        lo = (t[:, 0::2] & 0x0F).astype(np.uint8)
        hi = (t[:, 1::2] & 0x0F).astype(np.uint8) << 4
        packed[:, :lo.shape[1]] = lo
        packed[:, :hi.shape[1]] |= hi
        w.add_tensor(name + '.weight', W.shape, TERNARY,
                     packed.reshape(-1).tobytes())
        w.add_tensor(name + '.scale', (W.shape[0],), F32, s.tobytes())

    for l in range(n_layers):
        src, dst = K(f'blocks.{l}.'), f'blocks.{l}.'
        add_f32(dst + 'ln1.weight', T(src + 'ln1.weight'))
        add_f32(dst + 'ln1.bias', T(src + 'ln1.bias'))
        add_f32(dst + 'ln2.weight', T(src + 'ln2.weight'))
        add_f32(dst + 'ln2.bias', T(src + 'ln2.bias'))
        for a, b in (('x_r', 'tmix_r'), ('x_w', 'tmix_w'), ('x_k', 'tmix_k'),
                     ('x_v', 'tmix_v'), ('x_a', 'tmix_a'), ('x_g', 'tmix_g')):
            add_f32(dst + 'att.' + b, T(src + 'att.' + a)[0])
        for a in ('receptance', 'key', 'value', 'output'):
            add_ternary_qat(dst + 'att.' + a, model.masters[f'l{l}.att.{a}'])
        # LoRA factors stay i8, stored transposed (see convert_to_omniseed).
        for a in ('w1', 'w2', 'a1', 'a2', 'g1', 'g2', 'v1', 'v2'):
            add_i8(dst + 'att.' + a, np.ascontiguousarray(T(src + 'att.' + a).T))
        add_f32(dst + 'att.w_bias', T(src + 'att.w0')[0])
        add_f32(dst + 'att.a_bias', T(src + 'att.a0')[0])
        add_f32(dst + 'att.v_bias', T(src + 'att.v0')[0])
        add_f32(dst + 'att.k_k', T(src + 'att.k_k')[0])
        add_f32(dst + 'att.k_a', T(src + 'att.k_a')[0])
        add_f32(dst + 'att.r_k', T(src + 'att.r_k').reshape(-1))
        add_f32(dst + 'att.gn.weight', T(src + 'att.ln_x.weight'))
        add_f32(dst + 'att.gn.bias', T(src + 'att.ln_x.bias'))
        add_f32(dst + 'ffn.tmix_v', T(src + 'ffn.x_k')[0])
        add_ternary_qat(dst + 'ffn.key', model.masters[f'l{l}.ffn.key'])
        add_ternary_qat(dst + 'ffn.value', model.masters[f'l{l}.ffn.value'])

    add_f32('ln_out.weight', T(K('ln_out.weight')))
    add_f32('ln_out.bias', T(K('ln_out.bias')))
    hkey = K('head.weight') if K('head.weight') in header else 'head.weight'
    add_i8('head', T(hkey))

    size = w.write(args.out)
    print(f'[qat] wrote {args.out}: {size:,} bytes ({size / 1048576:.1f} MB)',
          flush=True)


if __name__ == '__main__':
    main()
