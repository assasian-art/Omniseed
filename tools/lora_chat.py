#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — lora_chat.py   (OFFLINE training tool, never ships at runtime)
#
#  LoRA assistant-behavior pass: teaches the (frozen) ternary RWKV-7 to reply
#  instead of rambling, WITHOUT touching the quantized runtime weights.
#
#  How it fits together:
#    * Base model: models/model.safetensors via RWKV7Ternary (qat_ternary.py)
#      — frozen; its ternary-STE forward IS the runtime's math.
#      --gguf-base P overrides EVERY tensor with the values read back from
#      the served omniseed GGUF (ternary nibbles + per-row scales, int8
#      gates/head, f16 embedding), so the trained sidecar optimizes exactly
#      the weights the C++ runtime serves. Big linears are placed per their
#      stored dtype: ternary (dtype 40) masters at T·(scale·in/nnz) so
#      TernarySTE reproduces the served dequantized ternary EXACTLY
#      (round(M/s)=T and absmean(M)=s); int8 (dtype 4, e.g.
#      rwkv7-0.1B-ternary.gguf) masters take the dequantized q·scale
#      directly with an identity passthrough (127 levels != 3, but the
#      base is frozen — exact passthrough is legal and cheapest).
#    * Trainable: classic LoRA factors B·A on the 6 big linears per layer
#      (att r/k/v/o + ffn key/value — the same tensors the runtime carries as
#      ternary), rank 8, alpha 16, B initialised to zero (step 0 == base).
#    * Runtime applies y = W_ternary·x + scaling · B(A·x) from a sidecar;
#      this tool exports exactly that sidecar:
#          models/assistant-lora.gguf
#          omniseed-lora container (GgufWriter):
#            lora.rank / lora.alpha / lora.scaling / lora.layer_count /
#            lora.n_embd / lora.base_vocab metadata
#            lora.format_version (=2, format stamp) / lora.trained_steps /
#            lora.best_step (holdout-best snapshot, may be < trained_steps)
#            lora.{l}.att.{receptance,key,value,output}.a  F16 [rank, in]
#            lora.{l}.att.{...}.b                          F16 [out, rank]
#            lora.{l}.ffn.{key,value}.a/.b                 F16
#    * Prompt format == the C++ Tokenizer::encode_chat world-model template:
#      "User: <text>\n\nAssistant: <reply>" — no runtime prompt plumbing.
#
#  Training: teacher-forced CE on ANSWER tokens only (prompt tokens are
#  context), AdamW on the LoRA factors, cosine LR + warmup, lockstep batch
#  (B examples padded through the same RNN core, like qat_ternary eval).
#
#  Degenerate-run guards (3000-step forensics, see PROJECT_STATE): holdout
#  split + early stop; the export is the BEST holdout snapshot (or a
#  zero-delta no-op sidecar when holdout never improved); B is asserted
#  exactly 0 at step 0; |B| drift is checked at resume and export.
#  Healthy holdout answer-ppl is ~1.05-3.0 — EXACTLY 1.00 means memorized,
#  not good.
#
#  Usage (repo root, venv with torch):
#    ./.venv/Scripts/python.exe tools/lora_chat.py                 # default
#    python tools/lora_chat.py --steps 3000 --eval-every 50 --patience 400
#                                              # full retrain (early-stops)
#    python tools/lora_chat.py --corpus-file my_chat.txt           # custom
#    python tools/lora_chat.py --sample "What is 2+2?"             # demo only
#
#  A real behavior pass wants ~2-4k steps (minutes on GPU, ~hours CPU);
#  a short --steps run still exports a valid sidecar and is enough to prove
#  the end-to-end path (the runtime test asserts the delta changes output).
# =============================================================================
import argparse
import hashlib
import json
import math
import os
import random
import struct
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qat_ternary import (            # noqa: E402
    RWKV7Ternary, ensure_checkpoint, load_safetensors, resolve_vocab,
)
from convert_to_omniseed import (    # noqa: E402
    K, GgufWriter, F16,
)

# -----------------------------------------------------------------------------
# Served-GGUF base loading (--gguf-base): read back the exact quantized math
# the C++ runtime serves, so the LoRA trains ON the deployment weights.
# -----------------------------------------------------------------------------
_GGUF_KV_SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
               10: 8, 11: 8, 12: 8}


def _gguf_walk(path):
    """Parse an OmniSeed GGUF: returns (blob, kvs, {name: (rowmajor_shape,
    dtype, data_offset)}, data_base). Offsets are relative to data_base.
    kvs holds scalar numeric metadata (u32/i32/u64/f64 -> int/float)."""
    with open(path, 'rb') as f:
        blob = f.read()
    n_tensors, n_kv = struct.unpack('<2Q', blob[8:24])
    pos = 24
    kvs = {}

    def rstr(p):
        n = struct.unpack('<Q', blob[p:p + 8])[0]
        p += 8
        return blob[p:p + n], p + n

    for _ in range(n_kv):
        key, pos = rstr(pos)
        t = struct.unpack('<I', blob[pos:pos + 4])[0]
        pos += 4
        if t == 8:
            _, pos = rstr(pos)
        elif t == 9:
            at = struct.unpack('<I', blob[pos:pos + 4])[0]
            pos += 4
            cnt = struct.unpack('<Q', blob[pos:pos + 8])[0]
            pos += 8
            for _ in range(cnt):
                if at == 8:
                    _, pos = rstr(pos)
                else:
                    pos += _GGUF_KV_SZ[at]
        elif t in (10, 12):
            kvs[key.decode()] = struct.unpack('<Q', blob[pos:pos + 8])[0]
            pos += 8
        elif t in (4, 5):
            kvs[key.decode()] = struct.unpack('<I', blob[pos:pos + 4])[0]
            pos += 4
        else:
            pos += _GGUF_KV_SZ[t]

    tensors = {}
    for _ in range(n_tensors):
        name, pos = rstr(pos)
        nd = struct.unpack('<I', blob[pos:pos + 4])[0]
        pos += 4
        ne = struct.unpack('<%dQ' % nd, blob[pos:pos + 8 * nd])
        pos += 8 * nd
        dt = struct.unpack('<I', blob[pos:pos + 4])[0]
        pos += 4
        off = struct.unpack('<Q', blob[pos:pos + 8])[0]
        pos += 8
        # ne is REVERSED row-major (cols, rows) -> restore [rows, cols]
        tensors[name.decode()] = (tuple(reversed(ne)), dt, off)
    return blob, kvs, tensors, pos


def _gguf_dequant_ternary(blob, shape, off):
    """Unpack dtype-40 nibbles (row-major [out, in]; 0x1=+1, 0xF=-1) and
    dequantize with the per-out-row fp32 scales stored alongside."""
    out_dim, in_dim = shape
    row_bytes = (in_dim + 1) // 2
    raw = np.frombuffer(blob, dtype=np.uint8,
                        count=out_dim * row_bytes, offset=off)
    raw = raw.reshape(out_dim, row_bytes)
    lo = (raw & 0x0F).astype(np.int8)
    hi = ((raw >> 4) & 0x0F).astype(np.int8)
    t = np.empty((out_dim, in_dim), dtype=np.int8)
    t[:, 0::2] = np.where(lo < 8, lo, lo - 16)
    t[:, 1::2] = np.where(hi < 8, hi, hi - 16)
    return t


def _file_sha12(path):
    """First 12 hex chars of the file's sha256 — cheap content identity for
    the base-stamp (catches resume-with-different-weights mistakes)."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:12]


def apply_gguf_base(model, gguf_path):
    """Overwrite every frozen tensor + master with the SERVED values from an
    omniseed GGUF. Masters are placed at T·(scale·in/nnz) so TernarySTE
    reproduces the file's dequantized ternary exactly."""
    blob, kvs, tensors, base = _gguf_walk(gguf_path)
    t = model.torch
    if kvs.get('omniseed.layer_count', model.L) != model.L or \
            kvs.get('omniseed.embedding_length', model.E) != model.E:
        raise SystemExit(
            f'[lora] FATAL: GGUF geometry mismatch — base model is '
            f'L={model.L} E={model.E}, GGUF reports '
            f'L={kvs.get("omniseed.layer_count")} '
            f'E={kvs.get("omniseed.embedding_length")}')

    def f32(name):
        shape, dt, off = tensors[name]
        assert dt == 0, (name, dt)
        n = int(np.prod(shape))
        a = np.frombuffer(blob, dtype='<f4', count=n, offset=base + off)
        return torch.from_numpy(a.astype('<f4')).float().reshape(shape)

    def f16(name):
        shape, dt, off = tensors[name]
        assert dt == 1, (name, dt)
        n = int(np.prod(shape))
        a = np.frombuffer(blob, dtype='<f2', count=n, offset=base + off)
        return torch.from_numpy(a.astype('<f4')).float().reshape(shape)

    def i8dq(name):
        """int8 + per-row scale -> fp32 [rows, cols] (row-major as stored)."""
        shape, dt, off = tensors[name + '.weight']
        assert dt == 4, (name, dt)
        n = int(np.prod(shape))
        q = np.frombuffer(blob, dtype=np.int8, count=n, offset=base + off)
        sshape, sdt, soff = tensors[name + '.scale']
        assert sdt == 0 and sshape[0] == shape[0], (name, sshape)
        s = np.frombuffer(blob, dtype='<f4', count=shape[0],
                          offset=base + soff)
        d = q.reshape(shape).astype(np.float32) * s[:, None]
        return torch.from_numpy(np.ascontiguousarray(d))

    def ternary(name):
        """(T int8 [out,in], scale fp32 [out]) for dtype-40 linears."""
        wname, sname = name + '.weight', name + '.scale'
        shape, dt, off = tensors[wname]
        assert dt == 40, (name, dt)
        tt = _gguf_dequant_ternary(blob, shape, base + off)
        sshape, sdt, soff = tensors[sname]
        assert sdt == 0 and sshape[0] == shape[0], (name, sshape)
        s = np.frombuffer(blob, dtype='<f4', count=shape[0],
                          offset=base + soff)
        return tt, s

    V, E = model.V, model.E
    p = model.p
    p['emb'] = f16('token.embd')
    p['ln0w'] = f32('blocks.0.ln0.weight').reshape(-1)
    p['ln0b'] = f32('blocks.0.ln0.bias').reshape(-1)
    p['lnow'] = f32('ln_out.weight').reshape(-1)
    p['lnob'] = f32('ln_out.bias').reshape(-1)
    p['head'] = i8dq('head')                            # [V, E]

    n_masters = 0
    for l in range(model.L):
        d = f'l{l}.'
        g = f'blocks.{l}.'
        for a in ('ln1.weight', 'ln1.bias', 'ln2.weight', 'ln2.bias'):
            p[d + a] = f32(g + a).reshape(-1)
        for a, b in (('x_r', 'tmix_r'), ('x_w', 'tmix_w'),
                     ('x_k', 'tmix_k'), ('x_v', 'tmix_v'),
                     ('x_a', 'tmix_a'), ('x_g', 'tmix_g')):
            p[d + a] = f32(g + 'att.' + b).reshape(-1)
        # int8 low-rank gates are stored TRANSPOSED; python wants the HF
        # orientation (w1 [E, rank], w2 [rank, E], ...)
        for a in ('w1', 'w2', 'a1', 'a2', 'g1', 'g2', 'v1', 'v2'):
            p[d + a] = i8dq(g + 'att.' + a).t().contiguous()
        p[d + 'w0'] = f32(g + 'att.w_bias').reshape(-1)
        p[d + 'a0'] = f32(g + 'att.a_bias').reshape(-1)
        p[d + 'v0'] = f32(g + 'att.v_bias').reshape(-1)
        p[d + 'k_k'] = f32(g + 'att.k_k').reshape(-1)
        p[d + 'k_a'] = f32(g + 'att.k_a').reshape(-1)
        # r_k participates as [H, D] in the attention bonus (the GGUF stores
        # it flat 768 = H*D)
        p[d + 'r_k'] = f32(g + 'att.r_k').view(model.H, model.D)
        p[d + 'lnxw'] = f32(g + 'att.gn.weight').view(model.H, model.D)
        p[d + 'lnxb'] = f32(g + 'att.gn.bias').view(model.H, model.D)
        p[d + 'fxk'] = f32(g + 'ffn.tmix_v').reshape(-1)
        # masters, per stored dtype:
        #   * dtype 40 (ternary): T·(scale·in/nnz) => TernarySTE(masters)
        #     == T·scale EXACTLY (round(M/s)=T + absmean(M)=s identity)
        #   * dtype 4 (int8 row-quant, e.g. rwkv7-0.1B-ternary.gguf): the
        #     dequantized q·scale directly. int8 has 127 levels — a ternary
        #     STE cannot reproduce it — so these masters ride an IDENTITY
        #     passthrough in begin_window (legal: the LoRA base is frozen,
        #     no gradient ever flows through it).
        for name in ('att.receptance', 'att.key', 'att.value', 'att.output',
                     'ffn.key', 'ffn.value'):
            wdt = tensors[g + name + '.weight'][1]
            if wdt == 4:
                W0 = i8dq(g + name)
                model.i8_passthrough.add(d + name)
            else:
                tt, s = ternary(g + name)
                nnz = np.maximum((tt != 0).sum(axis=1, keepdims=True), 1)
                factor = (tt.shape[1] / nnz).astype(np.float32)
                M = tt.astype(np.float32) * (s[:, None] * factor)
                W0 = torch.from_numpy(np.ascontiguousarray(M))
            model.masters[d + name].data = W0
            n_masters += 1
    n_i8 = len(model.i8_passthrough)
    print(f'[lora] base: GGUF {gguf_path} — every tensor = served values '
          f'({n_masters} linears: {n_masters - n_i8} ternary masters at '
          f'T·(s·in/nnz), {n_i8} int8 masters passed through exactly)')

# -----------------------------------------------------------------------------
# Corpus: small instruction/chat pairs, formatted exactly like the C++
# encode_chat world template. Each example: (user_text, reply_text).
# -----------------------------------------------------------------------------
DEFAULT_CORPUS = [
    ("What is 2+2?", " 2+2 equals 4."),
    ("What is the capital of France?", " The capital of France is Paris."),
    ("Who wrote Romeo and Juliet?", " Romeo and Juliet was written by William Shakespeare."),
    ("What is 10 times 10?", " 10 times 10 equals 100."),
    ("Name the largest planet in our solar system.", " The largest planet in our solar system is Jupiter."),
    ("What color is a banana?", " A banana is yellow."),
    ("How many days are in a week?", " There are 7 days in a week."),
    ("What is the boiling point of water in Celsius?", " Water boils at 100 degrees Celsius."),
    ("What language is spoken in Brazil?", " The main language spoken in Brazil is Portuguese."),
    ("What is 100 divided by 4?", " 100 divided by 4 equals 25."),
    ("Which ocean is the largest?", " The Pacific Ocean is the largest ocean."),
    ("How many legs does a spider have?", " A spider has 8 legs."),
    ("What is the freezing point of water in Fahrenheit?", " Water freezes at 32 degrees Fahrenheit."),
    ("Name the smallest prime number.", " The smallest prime number is 2."),
    ("What planet is known as the Red Planet?", " Mars is known as the Red Planet."),
    ("How many continents are there?", " There are 7 continents."),
    ("What is the chemical symbol for gold?", " The chemical symbol for gold is Au."),
    ("What is 5 squared?", " 5 squared equals 25."),
    ("Which gas do plants absorb from the air?", " Plants absorb carbon dioxide from the air."),
    ("What is the longest river in the world?", " The Nile is usually called the longest river in the world."),
    ("How many minutes are in an hour?", " There are 60 minutes in an hour."),
    ("What do bees make?", " Bees make honey."),
    ("What is the opposite of hot?", " The opposite of hot is cold."),
    ("What is 15 plus 27?", " 15 plus 27 equals 42."),
    ("Which planet is closest to the Sun?", " Mercury is the planet closest to the Sun."),
    ("What shape has three sides?", " A triangle has three sides."),
    ("What do caterpillars turn into?", " Caterpillars turn into butterflies or moths."),
    ("What is the capital of Japan?", " The capital of Japan is Tokyo."),
    ("How many hours are in two days?", " Two days have 48 hours."),
    ("What is the hardest natural substance?", " Diamond is the hardest natural substance."),
    ("What season comes after winter?", " Spring comes after winter."),
    ("What is 9 times 3?", " 9 times 3 equals 27."),
    ("What fruit is famous for keeping the doctor away?", " The apple is famous for keeping the doctor away."),
    ("What instrument has 88 keys?", " The piano has 88 keys."),
    ("What is the capital of Italy?", " The capital of Italy is Rome."),
    ("How many sides does a square have?", " A square has 4 sides."),
    ("What do we call frozen water?", " Frozen water is called ice."),
    ("What is 50 minus 13?", " 50 minus 13 equals 37."),
    ("Which star gives Earth light?", " The Sun gives Earth light."),
    ("What language has the most native speakers?", " Mandarin Chinese has the most native speakers."),
    ("What is the speed of light roughly?", " Light travels at about 300,000 kilometers per second."),
    ("What is the capital of Spain?", " The capital of Spain is Madrid."),
    ("How many strings does a standard guitar have?", " A standard guitar has 6 strings."),
    ("What metal is liquid at room temperature?", " Mercury is liquid at room temperature."),
    ("What is 8 times 8?", " 8 times 8 equals 64."),
    ("What do you call a baby dog?", " A baby dog is called a puppy."),
    ("What is the tallest animal in the world?", " The giraffe is the tallest animal in the world."),
    ("What is the first month of the year?", " The first month of the year is January."),
    ("How many colors are in a rainbow?", " A rainbow has 7 colors."),
    ("What is 3 to the power of 3?", " 3 to the power of 3 equals 27."),
    ("Which planet has prominent rings?", " Saturn has prominent rings."),
    ("What do cows produce?", " Cows produce milk."),
    ("What is the largest mammal?", " The blue whale is the largest mammal."),
    ("What is the capital of Germany?", " The capital of Germany is Berlin."),
    ("What is 200 grams in half?", " Half of 200 grams is 100 grams."),
    ("What organ pumps blood?", " The heart pumps blood."),
    ("What is the opposite of day?", " The opposite of day is night."),
    ("How many wheels does a tricycle have?", " A tricycle has 3 wheels."),
    ("What gas do humans breathe in to live?", " Humans breathe in oxygen to live."),
    ("What is the capital of Canada?", " The capital of Canada is Ottawa."),
    ("What is 7 plus 5?", " 7 plus 5 equals 12."),
    ("What do penguins eat?", " Penguins mainly eat fish and krill."),
    ("What season is the hottest?", " Summer is usually the hottest season."),
    ("What is the study of stars called?", " The study of stars is called astronomy."),
] * 3   # ~200 examples; small-corpus LoRA with many epochs


def load_corpus(path):
    """--corpus-file: lines 'user<TAB>reply' (blank lines skipped)."""
    if not path:
        return list(DEFAULT_CORPUS)
    pairs = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line.strip() or '\t' not in line:
                continue
            u, r = line.split('\t', 1)
            pairs.append((u, r))
    if not pairs:
        raise SystemExit(f'no usable "user<TAB>reply" lines in {path}')
    return pairs


FETCH_DEFAULT_PAIRS = 3000
_FETCH_URL = ('https://datasets-server.huggingface.co/rows'
              '?dataset=databricks%2Fdatabricks-dolly-15k'
              '&config=default&split=train&offset={off}&length=100')
_FETCH_UA = 'omniseed-lora/1.0 (offline trainer corpus fetch)'


def _fetch_rows(n_pairs):
    """Stream `n_pairs` dolly-15k rows (instruction, response) via the HF
    datasets-server JSON API. Only CLOSED-INSTRUCTION rows are used
    (context == ''), each side truncated to 200 chars. License: CC-BY-SA-3.0.
    Transient gateway errors (5xx) are retried with backoff."""
    import urllib.error
    import urllib.request
    out = []
    off = 0
    while len(out) < n_pairs:
        req = urllib.request.Request(_FETCH_URL.format(off=off),
                                     headers={'User-Agent': _FETCH_UA})
        page = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=45) as r:
                    page = json.loads(r.read().decode('utf-8'))
                break
            except urllib.error.HTTPError as e:
                if e.code < 500:
                    raise          # 4xx: our URL is wrong, fail loudly
                if attempt == 3:
                    raise SystemExit(
                        f'[lora] FATAL: datasets-server kept returning '
                        f'HTTP {e.code} after 4 tries — network is up but '
                        f'the API is failing; retry later or pass '
                        f'--corpus-file instead')
                time.sleep(2.0 * (attempt + 1))
        if page is None:
            break
        rows = page.get('rows', [])
        if not rows:
            break
        for row in rows:
            row = row.get('row', {})
            if row.get('context', '').strip():
                continue                     # closed QA only
            u = (row.get('instruction') or '').strip()
            a = (row.get('response') or '').strip()
            if not u or not a:
                continue
            out.append((u[:200], ' ' + a[:200]))
            if len(out) >= n_pairs:
                break
        off += len(rows)
    return out


def _model_dedupe(pairs):
    """Shingle dedupe on the first 28 chars of the question: near-duplicate
    templates inflate holdout numbers and push the trainer toward
    memorization (the degenerate-run failure mode)."""
    seen, out = set(), []
    for u, a in pairs:
        k = ' '.join(u.lower().split())[:28]
        if k in seen:
            continue
        seen.add(k)
        out.append((u, a))
    return out


def fetch_big_corpus(n_pairs, path=None):
    """--fetch-corpus / --corpus fetch: download `n_pairs` license-clean
    chat pairs into models/corpus/ and return the TSV path (cached)."""
    os.makedirs('models/corpus', exist_ok=True)
    if not path:
        path = f'models/corpus/dolly-{n_pairs}.tsv'
    if os.path.exists(path):
        n = sum(1 for ln in open(path, encoding='utf-8') if '\t' in ln)
        print(f'[lora] corpus cache hit: {path} ({n} pairs)')
        return path
    print(f'[lora] fetching {n_pairs} dolly-15k pairs (CC-BY-SA-3.0) '
          f'from the HF datasets-server ...')
    pairs = _fetch_rows(n_pairs)
    if len(pairs) < 200:
        raise SystemExit(f'[lora] FATAL: fetched only {len(pairs)} pairs '
                         f'(network or API problem) — refusing to train on '
                         f'a starved corpus; pass --corpus-file instead')
    pairs = _model_dedupe(pairs)
    with open(path, 'w', encoding='utf-8') as f:
        for u, a in pairs:
            f.write(f'{u.replace(chr(9), " ")}\t{a.replace(chr(9), " ")}\n')
    print(f'[lora] corpus: {path} — {len(pairs)} unique pairs written')
    return path


def encode_example(tok, user_text, reply_text):
    """Token ids for 'User: <u>\n\nAssistant: <r>' + eos. Returns (ids,
    answer_start) — answer tokens are positions [answer_start, len)."""
    prompt = f'User: {user_text}\n\nAssistant:'
    ids = tok.encode(prompt)                 # flat id list (HF-style)
    answer_start = len(ids)
    ids = ids + tok.encode(reply_text) + [0]   # eos = 0 (world pad)
    return ids, answer_start


# -----------------------------------------------------------------------------
# LoRA model: frozen ternary base + trainable B·A deltas in lin()
# -----------------------------------------------------------------------------
class LoraRWKV7(RWKV7Ternary):
    """RWKV7Ternary with rank-r LoRA on every master linear. The base stays
    frozen (its ternary-STE forward is the runtime's math); only A/B train.
    With gguf_base set, every tensor is overwritten with the exact served
    GGUF values BEFORE the LoRA factors are created."""

    def __init__(self, st_path, header, data_start, rank=8, alpha=16.0,
                 gguf_base=''):
        super().__init__(st_path, header, data_start)
        self.i8_passthrough = set()          # masters that must NOT ternarize
        if gguf_base:
            apply_gguf_base(self, gguf_base)
        t = self.torch
        self.rank, self.alpha = rank, alpha
        self.scaling = alpha / rank
        self.lora_A, self.lora_B, self.lora_params = {}, {}, []
        for full, W0 in self.masters.items():
            out_dim, in_dim = W0.shape
            r = min(rank, out_dim, in_dim)
            A = t.randn(r, in_dim, device=self.device) * (in_dim ** -0.5)
            A.requires_grad_(True)
            B = t.zeros(out_dim, r, device=self.device)
            B.requires_grad_(True)
            self.lora_A[full] = A
            self.lora_B[full] = B
            self.lora_params += [A, B]

    def to_device(self, device):
        """Base moves the frozen params + masters; the LoRA factors must move
        too (they are created in __init__ on the initial device)."""
        super().to_device(device)
        t = self.torch
        self.lora_A = {n: t_.detach().to(device).requires_grad_(True)
                       for n, t_ in self.lora_A.items()}
        self.lora_B = {n: t_.detach().to(device).requires_grad_(True)
                       for n, t_ in self.lora_B.items()}
        self.lora_params = ([p for pair in zip(self.lora_A.values(),
                                               self.lora_B.values())
                             for p in pair])

    def begin_window(self, use_grad):        # noqa: D102 — base never trains
        super().begin_window(False)
        # int8 masters: identity passthrough — the dequantized q·scale as
        # stored. The LoRA base is frozen (no gradient flows through it),
        # so the exact fp32 values are both correct and cheapest here.
        if self.i8_passthrough:
            for n in self.i8_passthrough:
                self.Wq[n] = self.masters[n].detach()

    def lin(self, name, x):
        y = super().lin(name, x)             # frozen ternary base
        if getattr(self, 'lora_on', True):
            y = y + self.scaling * (
                x @ self.lora_A[name].t() @ self.lora_B[name].t())
        return y


# -----------------------------------------------------------------------------
def main():
    torch.set_grad_enabled(True)
    torch.set_num_threads(int(os.environ.get('OMNISEED_TORCH_THREADS', '1')))

    ap = argparse.ArgumentParser()
    ap.add_argument('--safetensors', default='models/model.safetensors')
    ap.add_argument('--gguf-base', default='',
                    help='train on the SERVED omniseed GGUF (ternary/quant '
                         'weights read back exactly): pass the same path '
                         'you serve with --model (e.g. '
                         'models/rwkv7-0.1B-ternary-qat.gguf); the '
                         'safetensors stays the geometry source')
    ap.add_argument('--corpus', default='', dest='corpus_file',
                    help='alias for --corpus-file; the special value '
                         '"fetch" downloads a license-clean corpus')
    ap.add_argument('--corpus-file', default='',
                    help='TSV "user<TAB>reply" lines; default = builtin set')
    ap.add_argument('--fetch-corpus', default='', metavar='N_PAIRS',
                    help='download a license-clean instruction/chat corpus '
                         '(databricks-dolly-15k closed-QA excerpts, '
                         'CC-BY-SA-3.0) of this many pairs into '
                         'models/corpus/ and train on it; the tiny in-repo '
                         'set stays a fixture only')
    ap.add_argument('--steps', type=int, default=120)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lr-min', type=float, default=1e-5)
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--rank', type=int, default=8)
    ap.add_argument('--alpha', type=float, default=16.0)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--ckpt', default='models/lora_chat.pt')
    ap.add_argument('--out', default='models/assistant-lora.gguf')
    ap.add_argument('--eval-every', type=int, default=20)
    ap.add_argument('--holdout', type=int, default=8,
                    help='last N corpus pairs are held out (never trained '
                         'on); they drive early stop + best-snapshot pick')
    ap.add_argument('--patience', type=int, default=200,
                    help='early stop after this many steps without a '
                         'holdout improvement (0 = disabled)')
    ap.add_argument('--min-delta', type=float, default=1e-3,
                    help='minimum holdout-loss improvement to count as best')
    ap.add_argument('--format-version', type=int, default=2,
                    help='sidecar format stamp (lora.format_version)')
    ap.add_argument('--sample', default='',
                    help='greedy-generate a reply for this prompt and exit '
                         '(loads the checkpoint when present)')
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    args = ap.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f'[lora] device: {device}')
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    sys.path.insert(0, 'models/hf-orig')
    from hf_rwkv_tokenizer import RwkvTokenizer
    tok = RwkvTokenizer(vocab_file=resolve_vocab())

    n_fetch = 0
    if args.corpus_file == 'fetch':
        args.corpus_file = ''
        n_fetch = FETCH_DEFAULT_PAIRS
    if args.fetch_corpus:
        n_fetch = max(200, min(int(args.fetch_corpus), 20000))
    if n_fetch:
        args.corpus_file = fetch_big_corpus(n_fetch, args.corpus_file)

    pairs = load_corpus(args.corpus_file)
    if not pairs:
        raise SystemExit('[lora] FATAL: 0 usable chat pairs — refusing to '
                         'train on an empty corpus')
    examples = []
    for u, r in pairs:
        ids, a0 = encode_example(tok, u, r)
        examples.append((ids, a0))
    n_ans = [len(ids) - a0 for ids, a0 in examples]
    print(f'[lora] corpus: {len(examples)} pairs, mean answer tokens '
          f'{sum(n_ans) / len(n_ans):.1f} (min {min(n_ans)}, max {max(n_ans)}), '
          f'mean prompt+answer length '
          f'{sum(len(i) for i, _ in examples) / len(examples):.0f} tokens')
    if len(examples) < 4:
        raise SystemExit('[lora] FATAL: < 4 chat pairs is not trainable')

    n_hold = max(1, min(args.holdout, len(examples) // 5))
    if len(examples) - n_hold < 2:
        raise SystemExit('[lora] FATAL: holdout consumes the whole corpus')
    train_examples = examples[:len(examples) - n_hold]
    holdout = examples[len(examples) - n_hold:]
    print(f'[lora] split: {len(train_examples)} train / {n_hold} holdout '
          f'(holdout is NEVER trained on; it drives early stop)')

    ensure_checkpoint(args.safetensors)
    header, data_start = load_safetensors(args.safetensors)
    model = LoraRWKV7(args.safetensors, header, data_start,
                      rank=args.rank, alpha=args.alpha,
                      gguf_base=args.gguf_base)
    model.to_device(device)

    _mb = max(float(t.detach().abs().max()) for t in model.lora_B.values())
    if _mb != 0.0:
        raise SystemExit(f'[lora] FATAL: B init must be exactly zero (max '
                         f'|B| = {_mb:.3e}) — the step-0 delta would not '
                         f'match the base model')
    print('[lora] init check: all B == 0 exactly (step-0 delta == base)')

    optimizer = torch.optim.AdamW(model.lora_params, lr=args.lr,
                                  weight_decay=0.0, betas=(0.9, 0.95))

    # Effective training base + content stamp: LoRA deltas are base-specific,
    # so the checkpoint and sidecar record exactly which weights they fit.
    base_path = args.gguf_base if args.gguf_base else args.safetensors
    base_id = _file_sha12(base_path)
    print(f'[lora] training base: {"GGUF (served values)" if args.gguf_base else "PTQ safetensors"}'
          f' — {base_path}')
    print(f'[lora] base stamp: {base_id}')

    def cosine_lr(step):
        if step < args.warmup:
            return args.lr * (step + 1) / max(1, args.warmup)
        t_ = (step - args.warmup) / max(1, args.steps - args.warmup)
        t_ = min(1.0, max(0.0, t_))
        return args.lr_min + 0.5 * (args.lr - args.lr_min) \
            * (1 + math.cos(math.pi * t_))

    @torch.no_grad()
    def answer_loss(batch):
        """Held-out evaluation: CE over ANSWER tokens only (same mask as
        training). Returns (loss_sum, n_answer_tokens)."""
        model.begin_window(False)
        total = torch.zeros((), device=device)
        n = 0
        state = None
        max_len = max(len(ids) for ids, _ in batch)
        for j in range(max_len - 1):
            toks = torch.tensor(
                [ids[j] if j < len(ids) else 0 for ids, _ in batch],
                dtype=torch.long, device=device)
            logits, state = model.step_batch(toks, state)
            for b, (ids, a0) in enumerate(batch):
                nxt = j + 1
                if a0 <= nxt < len(ids):
                    total = total + torch.nn.functional.cross_entropy(
                        logits[b].float().unsqueeze(0),
                        torch.tensor([ids[nxt]], device=device))
                    n += 1
        model.end_window()
        return total, n

    with torch.no_grad():
        base_total, base_n = answer_loss(holdout)
    base_loss = (base_total / max(1, base_n)).item()
    print(f'[lora] holdout base answer-loss {base_loss:.3f} (ppl '
          f'{math.exp(min(20.0, base_loss)):.2f}) — early-stop reference')

    # resume
    global_step = 0
    if os.path.exists(args.ckpt):
        ck = torch.load(args.ckpt, map_location='cpu', weights_only=True)
        if 'base_id' in ck and ck['base_id'] != base_id:
            raise SystemExit(
                f'[lora] FATAL: checkpoint was trained on base '
                f'"{ck.get("base", "?")}" ({ck["base_id"]}) but this run '
                f'uses "{os.path.basename(base_path)}" ({base_id}) — LoRA '
                f'deltas are base-specific. Start fresh or pass the '
                f'original --gguf-base/--safetensors.')
        if 'base_id' not in ck:
            print('[lora] note: checkpoint predates base stamping '
                  '(no base_id recorded)')
        with torch.no_grad():
            for full, a in ck['A'].items():
                model.lora_A[full].copy_(a)
            for full, b in ck['B'].items():
                model.lora_B[full].copy_(b)
        if 'optim' in ck:
            optimizer.load_state_dict(ck['optim'])
        global_step = ck['global_step']
        print(f'[lora] resumed from {args.ckpt} (step {global_step})')
        _mb = max(float(t.abs().max()) for t in model.lora_B.values())
        print(f'[lora] resumed B max |.| = {_mb:.3e}')
        if _mb > 0.25:
            print('[lora] WARNING: |B| > 0.25 — past-convergence drift '
                  '(overtrained state). Holdout eval decides, but a fresh '
                  'run is safer (move the old checkpoint aside first).')

    def batch_loss(batch, train):
        """Lockstep batch through the RNN core; CE on ANSWER tokens only.
        Returns (loss_scalar, n_answer_tokens)."""
        model.begin_window(False)            # frozen ternary base, graph-detached
        B = len(batch)
        max_len = max(len(ids) for ids, _ in batch)
        state = None
        total = torch.zeros((), device=device)
        n_tok = 0
        for j in range(max_len):
            toks = torch.tensor(
                [ids[j] if j < len(ids) else 0 for ids, _ in batch],
                dtype=torch.long, device=device)
            logits, state = model.step_batch(toks, state)
            if j + 1 >= max_len:
                break
            for b, (ids, a0) in enumerate(batch):
                nxt = j + 1
                if a0 <= nxt < len(ids):
                    total = total + torch.nn.functional.cross_entropy(
                        logits[b].float().unsqueeze(0),
                        torch.tensor([ids[nxt]], device=device))
                    n_tok += 1
            if train:
                state = [(S.detach(), a.detach(), f.detach())
                         for S, a, f in state]
        model.end_window()
        return total, n_tok

    if args.sample:
        model.begin_window(False)
        # PROMPT-ONLY: no trailing eos — a post-eos state is out-of-
        # distribution and makes generation start on junk.
        ids = tok.encode(f'User: {args.sample}\n\nAssistant:')
        st = None
        lg = None
        for i in ids:
            lg, st = model.step_batch(torch.tensor([i], device=device), st)
        out = []
        t = model.torch
        for _ in range(48):
            nid = int(lg[0].argmax().item())
            if nid == 0:
                break
            out.append(nid)
            lg, st = model.step_batch(
                torch.tensor([nid], device=device), st)
        model.end_window()
        print(f'[lora] reply: {tok.decode(out) if out else "(empty)"}')
        return

    started = time.time()
    B = args.batch
    n_starts = len(train_examples)
    step_loss_history = []             # per-step answer-token CE (for E2E tests)
    best_loss, best_step = base_loss, 0
    best_A = {n: t.detach().cpu().clone() for n, t in model.lora_A.items()}
    best_B = {n: torch.zeros_like(t.detach().cpu())
              for n, t in model.lora_B.items()}   # best-at-start == no delta
    if args.steps <= 0:
        print('[lora] --steps 0: exporting the zero-delta sidecar '
              '(attached logits must be bit-identical to the base)')
    while global_step < args.steps:
        lr = cosine_lr(global_step)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        picks = [(global_step * 7919 + b * 104729) % n_starts for b in range(B)]
        batch = [train_examples[p] for p in picks]
        total, n_tok = batch_loss(batch, train=True)
        if n_tok == 0:
            raise SystemExit('[lora] FATAL: answer mask selected 0 tokens — '
                             'answer_start is broken (prompt-format '
                             'mismatch); refusing to "train" on nothing')
        (total / n_tok).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.lora_params,
                                                   args.clip).item()
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1
        step_loss_history.append((total / max(1, n_tok)).item())
        if global_step <= 5:
            n_pos = sum(len(ids) - 1 for ids, _ in batch)
            print(f'[lora] step {global_step}: answer-mask kept '
                  f'{n_tok}/{n_pos} positions '
                  f'({100.0 * n_tok / max(1, n_pos):.0f}%; prompts are '
                  f'masked out of the loss)', flush=True)
        if global_step % 10 == 0:
            print(f'[lora] step {global_step}/{args.steps} '
                  f'loss {(total / max(1, n_tok)).item():.3f} '
                  f'lr={lr:.2e} grad_norm={grad_norm:.2e}', flush=True)
        if args.eval_every > 0 and global_step % args.eval_every == 0:
            ev_total, ev_n = answer_loss(holdout)
            ev = (ev_total / max(1, ev_n)).item()
            tag = ''
            if ev < best_loss - args.min_delta:
                best_loss, best_step = ev, global_step
                best_A = {n: t.detach().cpu().clone()
                          for n, t in model.lora_A.items()}
                best_B = {n: t.detach().cpu().clone()
                          for n, t in model.lora_B.items()}
                tag = ' *best*'
            print(f'[lora] step {global_step} holdout answer-ppl='
                  f'{math.exp(min(20.0, ev)):.2f}{tag}', flush=True)
            if args.patience > 0 and global_step - best_step >= args.patience:
                print(f'[lora] early stop: no holdout improvement for '
                      f'{args.patience} steps (best {best_loss:.3f} @ '
                      f'step {best_step})')
                break

    if best_step == 0:
        print('[lora] holdout NEVER improved — exporting the ZERO-DELTA '
              'state (attaching it is exactly a no-op; it can never garble)')
        with torch.no_grad():
            for t in model.lora_B.values():
                t.zero_()
    elif best_step != global_step:
        print(f'[lora] restoring best holdout snapshot (loss {best_loss:.3f} '
              f'@ step {best_step}, not the final step {global_step})')
        with torch.no_grad():
            for full, a in best_A.items():
                model.lora_A[full].copy_(a)
            for full, b in best_B.items():
                model.lora_B[full].copy_(b)

    torch.save({'A': {n: t.detach().cpu().clone()
                      for n, t in model.lora_A.items()},
                'B': {n: t.detach().cpu().clone()
                      for n, t in model.lora_B.items()},
                'optim': optimizer.state_dict(),
                'global_step': global_step,
                'best_step': best_step,
                'best_eval_loss': best_loss,
                'format_version': args.format_version,
                'rank': args.rank, 'alpha': args.alpha,
                'base_id': base_id,
                'base': os.path.basename(base_path)},
               args.ckpt)
    print(f'[lora] checkpoint saved: {args.ckpt} (step {global_step}, best '
          f'{best_loss:.3f} @ {best_step}, {time.time() - started:.0f}s)')

    # ---------------- sidecar export (F16 A/B, scaling in metadata) ---------
    V, E = model.V, model.E
    w = GgufWriter()
    w.add_str('general.architecture', 'omniseed-lora')
    w.add_str('general.name', 'assistant-lora')
    w.add_u64('lora.rank', args.rank)
    w.add_f64('lora.alpha', args.alpha)
    w.add_f64('lora.scaling', model.scaling)
    w.add_u64('lora.layer_count', model.L)
    w.add_u64('lora.n_embd', E)
    w.add_u64('lora.base_vocab', V)
    w.add_i32('lora.trained_steps', global_step)
    w.add_i32('lora.best_step', best_step)
    w.add_i32('lora.format_version', args.format_version)
    w.add_str('lora.base_id', base_id)
    w.add_str('lora.base', os.path.basename(base_path))

    def add16(name, arr):
        a = np.ascontiguousarray(arr.detach().cpu().numpy(), '<f2')
        w.add_tensor(name, a.shape, F16, a.tobytes())

    n_written = 0
    for l in range(model.L):
        for name in ('att.receptance', 'att.key', 'att.value', 'att.output',
                     'ffn.key', 'ffn.value'):
            full = f'l{l}.{name}'
            A = model.lora_A[full]      # [r, in]
            Bm = model.lora_B[full]     # [out, r]
            add16(f'lora.{l}.{name}.a', A)
            add16(f'lora.{l}.{name}.b', Bm)
            n_written += 2
    size = w.write(args.out)
    print(f'[lora] wrote {args.out}: {size:,} bytes, {n_written} tensors, '
          f'rank {args.rank}, scaling {model.scaling:.3f}, '
          f'steps {global_step} (best {best_step}, format v{args.format_version})')

    _mb = max(float(t.detach().abs().max()) for t in model.lora_B.values())
    if _mb > 0.25:
        print(f'[lora] WARNING: exported |B| max = {_mb:.3f} (> 0.25) — '
              f'healthy runs stay well below ~0.1; this looks overtrained')
    with torch.no_grad():
        fin_total, fin_n = answer_loss(holdout)
    fin = (fin_total / max(1, fin_n)).item()
    print(f'[lora] FINAL holdout answer-ppl {math.exp(min(20.0, fin)):.2f} '
          f'(base {math.exp(min(20.0, base_loss)):.2f}; healthy 1.05-3.0 — '
          f'exactly 1.00 = memorized)')

    # Machine-readable summary (tests/e2e_lora_train.py consumes this).
    summary = {
        'n_pairs': len(examples), 'n_train': len(train_examples),
        'n_holdout': len(holdout),
        'base_holdout_loss': base_loss, 'final_holdout_loss': fin,
        'best_step': best_step, 'best_holdout_loss': best_loss,
        'step_loss_history':
            step_loss_history if len(step_loss_history) <= 64 else [],
        'global_step': global_step,
    }
    print(f'[lora-e2e] {json.dumps(summary)}')
    return summary


if __name__ == '__main__':
    main()
