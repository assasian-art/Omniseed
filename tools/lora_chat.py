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
#    * Trainable: classic LoRA factors B·A on the 6 big linears per layer
#      (att r/k/v/o + ffn key/value — the same tensors the runtime carries as
#      ternary), rank 8, alpha 16, B initialised to zero (step 0 == base).
#    * Runtime applies y = W_ternary·x + scaling · B(A·x) from a sidecar;
#      this tool exports exactly that sidecar:
#          models/assistant-lora.gguf
#          omniseed-lora container (GgufWriter):
#            lora.rank / lora.alpha / lora.scaling / lora.layer_count /
#            lora.n_embd / lora.base_vocab metadata
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
#  Usage (repo root, venv with torch):
#    ./.venv/Scripts/python.exe tools/lora_chat.py                 # default
#    python tools/lora_chat.py --steps 400 --lr 1e-3               # real run
#    python tools/lora_chat.py --corpus-file my_chat.txt           # custom
#    python tools/lora_chat.py --sample "What is 2+2?"             # demo only
#
#  A real behavior pass wants ~2-4k steps (minutes on GPU, ~hours CPU);
#  a short --steps run still exports a valid sidecar and is enough to prove
#  the end-to-end path (the runtime test asserts the delta changes output).
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
from qat_ternary import (            # noqa: E402
    RWKV7Ternary, ensure_checkpoint, load_safetensors, resolve_vocab,
)
from convert_to_omniseed import (    # noqa: E402
    K, GgufWriter, F16,
)

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
    frozen (its ternary-STE forward is the runtime's math); only A/B train."""

    def __init__(self, st_path, header, data_start, rank=8, alpha=16.0):
        super().__init__(st_path, header, data_start)
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
    ap.add_argument('--corpus-file', default='',
                    help='TSV "user<TAB>reply" lines; default = builtin set')
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

    examples = []
    for u, r in load_corpus(args.corpus_file):
        ids, a0 = encode_example(tok, u, r)
        examples.append((ids, a0))
    print(f'[lora] corpus: {len(examples)} chat examples '
          f'(avg {sum(len(i) for i, _ in examples) / len(examples):.0f} tokens)')

    ensure_checkpoint(args.safetensors)
    header, data_start = load_safetensors(args.safetensors)
    model = LoraRWKV7(args.safetensors, header, data_start,
                      rank=args.rank, alpha=args.alpha)
    model.to_device(device)
    optimizer = torch.optim.AdamW(model.lora_params, lr=args.lr,
                                  weight_decay=0.0, betas=(0.9, 0.95))

    def cosine_lr(step):
        if step < args.warmup:
            return args.lr * (step + 1) / max(1, args.warmup)
        t_ = (step - args.warmup) / max(1, args.steps - args.warmup)
        t_ = min(1.0, max(0.0, t_))
        return args.lr_min + 0.5 * (args.lr - args.lr_min) \
            * (1 + math.cos(math.pi * t_))

    # resume
    global_step = 0
    if os.path.exists(args.ckpt):
        ck = torch.load(args.ckpt, map_location='cpu', weights_only=True)
        with torch.no_grad():
            for full, a in ck['A'].items():
                model.lora_A[full].copy_(a)
            for full, b in ck['B'].items():
                model.lora_B[full].copy_(b)
        if 'optim' in ck:
            optimizer.load_state_dict(ck['optim'])
        global_step = ck['global_step']
        print(f'[lora] resumed from {args.ckpt} (step {global_step})')

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
        ids, _ = encode_example(tok, args.sample, '')
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
    n_starts = len(examples)
    while global_step < args.steps:
        lr = cosine_lr(global_step)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        picks = [(global_step * 7919 + b * 104729) % n_starts for b in range(B)]
        batch = [examples[p] for p in picks]
        total, n_tok = batch_loss(batch, train=True)
        if n_tok == 0:
            continue
        (total / n_tok).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.lora_params,
                                                   args.clip).item()
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1
        if global_step % 10 == 0:
            print(f'[lora] step {global_step}/{args.steps} '
                  f'loss {(total / max(1, n_tok)).item():.3f} '
                  f'lr={lr:.2e} grad_norm={grad_norm:.2e}', flush=True)
        if args.eval_every > 0 and global_step % args.eval_every == 0:
            with torch.no_grad():
                ev_picks = [(global_step * 31 + i) % n_starts
                            for i in range(B)]
                ev_total, ev_n = batch_loss([examples[p] for p in ev_picks],
                                            train=False)
            print(f'[lora] step {global_step} answer-ppl='
                  f'{math.exp(min(20.0, ev_total / max(1, ev_n))):.2f}',
                  flush=True)

    torch.save({'A': {n: t.detach().cpu().clone()
                      for n, t in model.lora_A.items()},
                'B': {n: t.detach().cpu().clone()
                      for n, t in model.lora_B.items()},
                'optim': optimizer.state_dict(),
                'global_step': global_step,
                'rank': args.rank, 'alpha': args.alpha},
               args.ckpt)
    print(f'[lora] checkpoint saved: {args.ckpt} (step {global_step}, '
          f'{time.time() - started:.0f}s)')

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
          f'steps {global_step}')


if __name__ == '__main__':
    main()
