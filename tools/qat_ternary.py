#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — qat_ternary.py   (OFFLINE training tool, never ships at runtime)
#
#  Phase 8 TASK 3: Quantization-Aware Training proof-of-concept that turns the
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
#    * Corpus: real English text already in the repo (PROJECT_STATE.md +
#      docs + README), split 90/10 by paragraph.
#    * Reports perplexity: bf16 baseline / ternary-PTQ / ternary-QAT.
#    * Exports models/rwkv7-0.1B-ternary-qat.gguf (TERNARY linears, i8 LoRA
#      + head, everything else identical to the default converter output).
#
#  Usage (from repo root, venv with torch on PATH):
#    ./.venv/Scripts/python.exe tools/qat_ternary.py --steps 1000
# =============================================================================
import argparse
import math
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_to_omniseed import (  # noqa: E402
    K, load_safetensors, read_tensor,
    GgufWriter, load_world_vocab, I8, TERNARY, F16, F32,
)

ST_DEFAULT = 'models/model.safetensors'

CORPUS_FILES = [
    'PROJECT_STATE.md',
    'docs/OMNISEED_MASTER_SPEC.md',
    'README.md',
]

INV_SQ_E = 1.0 / math.sqrt(math.e)


# -----------------------------------------------------------------------------
# Corpus: real English paragraphs already in the repo. 90/10 split.
# -----------------------------------------------------------------------------
def load_corpus():
    paras = []
    for p in CORPUS_FILES:
        if not os.path.exists(p):
            continue
        text = open(p, 'r', encoding='utf-8', errors='replace').read()
        for chunk in text.split('\n\n'):
            chunk = chunk.strip()
            if len(chunk) >= 120:               # skip tiny fragments/tables
                paras.append(chunk)
    random.seed(7)
    random.shuffle(paras)
    n_val = max(4, len(paras) // 10)
    return paras[n_val:], paras[:n_val]


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
    (HF-verified), with ternary-STE linears for r/k/v/o + ffn key/value."""

    def __init__(self, st_path, header, data_start):
        import torch
        self.torch = torch
        self.header, self.ds, self.st = header, data_start, st_path
        self.cache = {}

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
            # (a single [0] leaves [1,768] and broadcasts x into a phantom
            #  leading axis — the bug this tool initially had).
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
        self.optimizer = torch.optim.SGD(self.master_list, lr=1e-3)

    def T(self, k):
        if k not in self.cache:
            self.cache[k] = self.torch.from_numpy(
                read_tensor(self.st, self.header, self.ds, k))
        return self.cache[k]

    def lin(self, name, x):
        """x @ Wq.T where Wq is the true ternary of the master (STE).
        float_mode (eval-only): use the raw bf16 master — the honest baseline
        that validates the forward pass itself."""
        if getattr(self, 'float_mode', False):
            return x @ self.masters[name].t()
        return x @ self.Wq[name].t()

    def begin_window(self, use_grad):
        """Ternarize all masters ONCE per window. Weights only change after an
        optimizer step, so per-token ternarization is pure overhead (72 tiny
        autograd graphs per token dominated runtime). The graph stays alive
        across the window so STE gradients reach the masters at backward."""
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

    def step(self, tok, state):
        """One token. state: list of (S[H,D,D], att_prev[E], ffn_prev[E])."""
        t = self.torch
        p, E, H, D, L = self.p, self.E, self.H, self.D, self.L

        def ln(x, w, b, eps=1e-5):
            m = x.mean(-1, keepdim=True)
            v = x.var(-1, keepdim=True, unbiased=False)
            return (x - m) / t.sqrt(v + eps) * w + b

        if state is None:
            state = [(t.zeros(H, D, D), t.zeros(E), t.zeros(E)) for _ in range(L)]
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

            # Batched over heads (72 tiny matmuls/token in python loops were
            # the eval bottleneck): sa[h,j] = -sum_i kk[h,i]*S[h,i,j]
            # (HF reference: S[h].T @ (-kk[h]) — the MINUS is load-bearing;
            #  dropping it turns the a-rank1 term into positive feedback and
            #  the state explodes to NaN).
            sa = -t.einsum('hi,hij->hj', kk, S)                         # [H, D]
            S = (wh.unsqueeze(-1) * S                                  # decay
                 + (kk * ah).unsqueeze(-1) * sa.unsqueeze(1)           # a-rank1
                 + kh.unsqueeze(-1) * vh.unsqueeze(1))                 # k outer v

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


def main():
    torch.set_grad_enabled(True)
    # Tiny-op pathology: multi-thread fan-out on 768-wide ops is ~5x SLOWER
    # on this box (Windows futex churn). Single thread unless overridden.
    torch.set_num_threads(int(os.environ.get('OMNISEED_TORCH_THREADS', '1')))

    ap = argparse.ArgumentParser()
    ap.add_argument('--safetensors', default=ST_DEFAULT)
    ap.add_argument('--steps', type=int, default=1000)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--window', type=int, default=32)
    ap.add_argument('--eval-tokens', type=int, default=1024,
                    help='cap perplexity eval length to bound runtime')
    ap.add_argument('--out', default='models/rwkv7-0.1B-ternary-qat.gguf')
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--start-step', type=int, default=0,
                    help='resume offset for the stride walk (with --resume)')
    ap.add_argument('--resume', default=None,
                    help='torch.save file with master weights to resume from')
    ap.add_argument('--save-masters', default='models/qat_masters.pt',
                    help='where to checkpoint master weights after the run')
    ap.add_argument('--no-export', action='store_true',
                    help='skip GGUF export (intermediate chunks)')
    ap.add_argument('--no-eval', action='store_true',
                    help='skip perplexity evals (intermediate chunks)')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    sys.path.insert(0, 'models/hf-orig')
    from hf_rwkv_tokenizer import RwkvTokenizer
    tok = RwkvTokenizer(vocab_file='models/rwkv_vocab_v20230424.txt')

    header, data_start = load_safetensors(args.safetensors)
    train_paras, val_paras = load_corpus()
    train_ids = tok.encode('\n\n'.join(train_paras))
    val_ids = tok.encode('\n\n'.join(val_paras))
    print(f'[qat] corpus: train {len(train_ids):,} tokens, '
          f'val {len(val_ids):,} tokens')

    model = RWKV7Ternary(args.safetensors, header, data_start)

    def perplexity(ids, float_masters=False):
        """Teacher-forced NLL over the first --eval-tokens after a warmup."""
        torch.set_grad_enabled(False)
        model.float_mode = float_masters
        model.begin_window(False)
        W = args.window
        n = min(len(ids) - 1, args.eval_tokens)
        total_nll, n_pred = 0.0, 0
        state = None
        for i in range(n):
            logits, state = model.step(ids[i], state)
            if i >= W:
                logp = torch.log_softmax(logits.float(), -1)
                total_nll += -logp[ids[i + 1]].item()
                n_pred += 1
        model.end_window()
        torch.set_grad_enabled(True)
        return math.exp(total_nll / max(1, n_pred))

    if args.resume and os.path.exists(args.resume):
        sd = torch.load(args.resume, map_location='cpu', weights_only=True)
        with torch.no_grad():
            for n, t_ in sd.items():
                model.masters[n].copy_(t_)
        print(f'[qat] resumed masters from {args.resume} '
              f'({len(sd)} tensors)')

    if args.no_eval:
        ppl_bf16_tr = ppl_ptq_val = float('nan')
    else:
        ppl_bf16_tr = perplexity(train_ids, float_masters=True)
        ppl_ptq_val = perplexity(val_ids)
        print(f'[qat] bf16          train ppl = {ppl_bf16_tr:.2f}')
        print(f'[qat] ternary-PTQ   val   ppl = {ppl_ptq_val:.2f}')

    if args.steps > args.start_step:
        W = args.window
        starts = list(range(0, len(train_ids) - W - 1, max(1, W // 4)))
        for step in range(args.start_step, args.steps):
            s = starts[(step * 7919) % len(starts)]     # prime stride walk
            model.begin_window(True)
            state = None
            total_loss = 0.0
            for j in range(s, s + W):
                logits, state = model.step(train_ids[j], state)
                loss = torch.nn.functional.cross_entropy(
                    logits.float().unsqueeze(0),
                    torch.tensor([train_ids[j + 1]]))
                total_loss = total_loss + loss
                state = [(S.detach(), a.detach(), f.detach())
                         for S, a, f in state]
            # One backward through the whole window: the shared ternary graph
            # (TernarySTE nodes) is traversed once, STE grads reach masters.
            (total_loss / W).backward()
            torch.nn.utils.clip_grad_norm_(model.master_list, 1.0)
            for pg in model.optimizer.param_groups:
                pg['lr'] = args.lr
            model.optimizer.step()
            model.optimizer.zero_grad()
            model.end_window()
            if step % 25 == 0 or step == args.steps - 1:
                print(f'[qat] step {step}/{args.steps} '
                      f'loss {(total_loss / W).item():.3f}', flush=True)

        if not args.no_eval:
            ppl_qat_val = perplexity(val_ids)
            print(f'[qat] ternary-QAT   val   ppl = {ppl_qat_val:.2f}')
            print(f'[qat] SUMMARY: bf16(train)={ppl_bf16_tr:.2f} '
                  f'ptq(val)={ppl_ptq_val:.2f} qat(val)={ppl_qat_val:.2f}')
        torch.save({n: t_.detach().clone() for n, t_ in model.masters.items()},
                   args.save_masters)
        print(f'[qat] masters saved to {args.save_masters}')
    else:
        print('[qat] SUMMARY: bf16(train)=%.2f ptq(val)=%.2f '
              '(--steps 0: eval-only run)' % (ppl_bf16_tr, ppl_ptq_val))

    if not args.no_export:
        export_gguf(model, args)


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

    pieces, types = load_world_vocab('models/rwkv_vocab_v20230424.txt', V)
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
    print(f'[qat] wrote {args.out}: {size:,} bytes ({size / 1048576:.1f} MB)')


if __name__ == '__main__':
    main()
