#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — convert_goose28.py   (OFFLINE tool)
#
#  Converts the OFFICIAL HF checkpoint RWKV/RWKV7-Goose-World2.8-0.1B-HF
#  (flash-linear-attention naming: model.layers.N.attn.*) into the OmniSeed
#  GGUF, alongside tools/convert_to_omniseed.py which handles the
#  Hakureirm/rwkv7-0.1b-hf (BlinkDL-style rwkv7.* naming) layout.
#
#  Key mapping facts (verified against BOTH the fla source and the official
#  transformers RWKV7 implementation, models/modeling_rwkv7.py):
#    * token-shift blend form is IDENTICAL to BlinkDL:
#      delta = prev_normed - x;  xs = x + x_r * delta  (x = NORMED stream).
#      (fla's fused_addcmul takes the delta precomputed; same algebra.)
#    * layer 0 pre_norm == BlinkDL blocks.0.ln0 (trained values, not ones).
#    * LoRA factors map 1:1 with NO transposes: lora.0 = Linear(in->rank)
#      => [rank, in] and lora.2 = Linear(rank->out) => [out, rank]; both are
#      exactly the C++ storage forms (same as BlinkDL w1/w2). Gates match
#      BlinkDL: w0=tanh(lora0)@lora2+b (then -1/sqrt(e)*sigmoid),
#      a=sigmoid(lora0@lora2+b), g=sigmoid(lora0)@lora2
#      (no output sigmoid), v=sigmoid(lora0@lora2+b) for layers >= 1 only.
#    * layer 0 has NO v_lora (fla: `if self.layer_idx != 0`); zero-filled
#      tensors are written so the loader contract stays uniform (the forward
#      never reads them at layer 0).
#    * vocab is the same 65,536-piece RWKV world vocabulary.
# =============================================================================
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_to_omniseed import (  # noqa: E402
    load_safetensors, read_tensor, quantize_i8, to_f16_bytes,
    GgufWriter, load_world_vocab, I8, TERNARY, F16, F32,
)

DEFAULT_ST = 'models/goose28.safetensors'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--safetensors', default=DEFAULT_ST)
    ap.add_argument('--vocab', default='models/rwkv_vocab_v20230424.txt')
    ap.add_argument('--out', default='models/rwkv7-goose28-0.1B-i8.gguf')
    ap.add_argument('--ternary', action='store_true',
                    help='ternary linears instead of int8 (PTQ; lossy)')
    args = ap.parse_args()

    header, ds = load_safetensors(args.safetensors)
    keys = [k for k in header if k != '__metadata__']
    P = 'model.'
    assert P + 'embeddings.weight' in header, 'not a fla-layout Goose checkpoint'

    V, E = header[P + 'embeddings.weight']['shape']
    n_layers = max(int(k.split('.')[2]) for k in keys if k.startswith(P + 'layers.')) + 1
    H, D = header[f'{P}layers.0.attn.r_k']['shape']
    I = header[f'{P}layers.0.ffn.key.weight']['shape'][0]
    rw = header[f'{P}layers.0.attn.w_lora.lora.0.weight']['shape'][0]
    ra = header[f'{P}layers.0.attn.a_lora.lora.0.weight']['shape'][0]
    rg = header[f'{P}layers.0.attn.g_lora.lora.0.weight']['shape'][0]
    rv = (header[f'{P}layers.1.attn.v_lora.lora.0.weight']['shape'][0]
          if f'{P}layers.1.attn.v_lora.lora.0.weight' in header else 32)
    print(f'[goose] V={V} E={E} L={n_layers} H={H} D={D} FFN={I} '
          f'ranks w={rw} a={ra} g={rg} v={rv}')

    pieces, types = load_world_vocab(args.vocab, V)

    w = GgufWriter()
    w.add_str('general.architecture', 'omniseed-rwkv7')
    w.add_str('general.name', 'RWKV7-Goose-World2.8-0.1B')
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

    T = lambda k: read_tensor(args.safetensors, header, ds, k)  # noqa: E731

    def add_lin(name, W):
        if args.ternary:
            from convert_to_omniseed import quantize_and_pack
            data, scales = quantize_and_pack(W)
            w.add_tensor(name + '.weight', W.shape, TERNARY, data.tobytes())
        else:
            q, s = quantize_i8(W)
            w.add_tensor(name + '.weight', W.shape, I8, q.tobytes())
        w.add_tensor(name + '.scale', (W.shape[0],), F32, s.tobytes())

    def add_f32(name, arr):
        w.add_tensor(name, arr.shape, F32, np.ascontiguousarray(arr, '<f4').tobytes())

    # embedding + layer-0 pre_norm (== BlinkDL blocks.0.ln0)
    w.add_tensor('token.embd', (V, E), F16, to_f16_bytes(T(P + 'embeddings.weight')))
    add_f32('blocks.0.ln0.weight', T(P + 'layers.0.pre_norm.weight'))
    add_f32('blocks.0.ln0.bias', T(P + 'layers.0.pre_norm.bias'))

    for l in range(n_layers):
        s, d = f'{P}layers.{l}.', f'blocks.{l}.'
        add_f32(d + 'ln1.weight', T(s + 'attn_norm.weight'))
        add_f32(d + 'ln1.bias', T(s + 'attn_norm.bias'))
        add_f32(d + 'ln2.weight', T(s + 'ffn_norm.weight'))
        add_f32(d + 'ln2.bias', T(s + 'ffn_norm.bias'))
        for a, b in (('x_r', 'tmix_r'), ('x_w', 'tmix_w'), ('x_k', 'tmix_k'),
                     ('x_v', 'tmix_v'), ('x_a', 'tmix_a'), ('x_g', 'tmix_g')):
            add_f32(d + 'att.' + b, T(s + 'attn.' + a)[0])
        for a, b in (('r_proj', 'receptance'), ('k_proj', 'key'),
                     ('v_proj', 'value'), ('o_proj', 'output')):
            add_lin(d + 'att.' + b, T(s + 'attn.' + a + '.weight'))
        # LoRA factors (see module docstring: lora.0/lora.2 are ALREADY in the
        # C++ storage forms [rank,in] / [out,rank] — no transposes anywhere)
        add_lin(d + 'att.w1', T(s + 'attn.w_lora.lora.0.weight'))            # [rank, E]
        add_lin(d + 'att.w2', T(s + 'attn.w_lora.lora.2.weight'))            # [E, rank]
        add_f32(d + 'att.w_bias', T(s + 'attn.w_lora.lora.2.bias'))
        add_lin(d + 'att.a1', T(s + 'attn.a_lora.lora.0.weight'))
        add_lin(d + 'att.a2', T(s + 'attn.a_lora.lora.2.weight'))
        add_f32(d + 'att.a_bias', T(s + 'attn.a_lora.lora.2.bias'))
        add_lin(d + 'att.g1', T(s + 'attn.g_lora.lora.0.weight'))
        add_lin(d + 'att.g2', T(s + 'attn.g_lora.lora.2.weight'))
        if l == 0:
            # no v_lora at layer 0 (fla); zeros keep the loader contract —
            # the forward never applies the value gate at layer 0.
            z1 = np.zeros((rv, E), dtype=np.float32)
            z2 = np.zeros((E, rv), dtype=np.float32)
            add_lin(d + 'att.v1', z1)
            add_lin(d + 'att.v2', z2)
            add_f32(d + 'att.v_bias', np.zeros(E, dtype=np.float32))
        else:
            add_lin(d + 'att.v1', T(s + 'attn.v_lora.lora.0.weight'))
            add_lin(d + 'att.v2', T(s + 'attn.v_lora.lora.2.weight'))
            add_f32(d + 'att.v_bias', T(s + 'attn.v_lora.lora.2.bias'))
        add_f32(d + 'att.k_k', T(s + 'attn.k_k'))
        add_f32(d + 'att.k_a', T(s + 'attn.k_a'))
        add_f32(d + 'att.r_k', T(s + 'attn.r_k').reshape(-1))
        add_f32(d + 'att.gn.weight', T(s + 'attn.g_norm.weight'))
        add_f32(d + 'att.gn.bias', T(s + 'attn.g_norm.bias'))
        add_f32(d + 'ffn.tmix_v', T(s + 'ffn.x_k'))
        add_lin(d + 'ffn.key', T(s + 'ffn.key.weight'))
        add_lin(d + 'ffn.value', T(s + 'ffn.value.weight'))

    add_f32('ln_out.weight', T(P + 'norm.weight'))
    add_f32('ln_out.bias', T(P + 'norm.bias'))
    add_lin('head', T('lm_head.weight'))

    size = w.write(args.out)
    mode = 'ternary' if args.ternary else 'i8'
    print(f'[goose] wrote {args.out} ({mode}): {size:,} bytes '
          f'({size / 1048576:.1f} MB)')


if __name__ == '__main__':
    main()
