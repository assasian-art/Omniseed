#!/usr/bin/env python3
# Oracle: official HF Rwkv7ForCausalLM from the model repo itself.
# Prints a greedy reference transcript for the dragons prompt.
import sys

sys.path.insert(0, 'models/hf-orig')
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from hf_rwkv_tokenizer import RwkvTokenizer

MODEL_DIR = 'models'
sys.path.insert(0, 'models')

import os
# Vocab: committed tools/data copy first, legacy models/ second.
VOCAB = 'tools/data/rwkv_vocab_v20230424.txt'
if not os.path.exists(VOCAB):
    VOCAB = 'models/rwkv_vocab_v20230424.txt'

torch.set_grad_enabled(False)
cfg = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
# the HF-format safetensors sits in models/model.safetensors with rwkv7.* keys
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, config=cfg, trust_remote_code=True, torch_dtype=torch.float32)
model.eval()

tok = RwkvTokenizer(vocab_file=VOCAB)
print('tokenizer:', type(tok).__name__, 'vocab', len(tok), flush=True)

vocab = {}
import ast
for line in open(VOCAB, encoding='utf-8'):
    line = line.rstrip('\r\n')
    p = line.split(' ')
    if len(p) < 3:
        continue
    tid = int(p[0])
    try:
        piece = ast.literal_eval("'" + ' '.join(p[1:-1]) + "'")
    except Exception:
        piece = ' '.join(p[1:-1])
    if isinstance(piece, tuple):
        piece = piece[0] if piece else ''
    if isinstance(piece, bytes):
        piece = piece.decode('utf-8', 'replace')
    if not isinstance(piece, str):
        piece = str(piece)
    vocab[tid] = piece

# Simple greedy longest-match encoder (same as the converter's helper).
pieces = sorted(vocab.items(), key=lambda kv: -len(kv[1]))
def encode(text):
    ids, i = [], 0
    while i < len(text):
        for tid, pc in pieces:
            if pc and text.startswith(pc, i):
                ids.append(tid); i += len(pc); break
        else:
            b = text[i].encode('utf-8')[0]
            ids.append(b + 1) if b < 128 else ids.append(0)
            i += 1
    return ids

ctx = "\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."
ids = tok.encode(ctx)
if isinstance(ids, dict):
    ids = ids['input_ids']
print('prompt tokens:', len(ids), flush=True)
print('roundtrip ok:', tok.decode(ids) == ctx, flush=True)
inp = torch.tensor([ids])
out = model.generate(inp, max_new_tokens=40, do_sample=False,
                     pad_token_id=0, eos_token_id=0)
gen = out[0][len(ids):].tolist()
txt = tok.decode(gen)
print('HF-ORACLE:', txt.encode('ascii', 'backslashreplace').decode('ascii'))
print('HF-ORACLE-IDS:', gen)
