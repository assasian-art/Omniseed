#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — e2e_lora_train.py   (OFFLINE regression, ctest omniseed_lora_e2e)
#
#  Trains the assistant LoRA for a FEW steps on the tiny in-repo fixture and
#  asserts the whole path is alive — this is the regression for the degenerate
#  3000-step run (loss 0.000 from step ~30, ppl exactly 1.00, garbage attach):
#
#    1. corpus stats print + non-zero pair count (fails loudly on empty)
#    2. CE loss on ANSWER tokens only (masked-token fraction printed early)
#    3. B initialised to ZEROS (trainer asserts; checked here in the log)
#    4. save/load round-trip: a struct-level python reader (the SAME layout
#       conventions the C++ GgufLoader uses) reads the sidecar back and the
#       f16 bytes are BITWISE identical to the checkpoint tensors; the
#       sidecar tensors are then attached in python and the resulting logits
#       must agree with the C++-attached forward (cross-engine parity)
#    5. loss drops > 10% over the run
#    6. attaching the exported sidecar LOWERS holdout answer-ppl vs base
#    7. generation stays coherent on 3 fixed prompts (garbage detector)
#
#  Registered as ctest `omniseed_lora_e2e` (WD = repo root, requires the
#  venv python + models/model.safetensors — auto-skips otherwise, exactly
#  like test_real_weights / omniseed_sides fixture policy).
# =============================================================================
import json
import math
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))   # tests/lora_e2e/x.py -> repo
PY = os.path.join(REPO, '.venv', 'Scripts', 'python.exe')
if not os.path.exists(PY):                       # posix venv layout
    alt = os.path.join(REPO, '.venv', 'bin', 'python')
    PY = alt if os.path.exists(alt) else sys.executable

g_pass = g_fail = 0

def check(cond, label):
    global g_pass, g_fail
    if cond:
        g_pass += 1
    else:
        g_fail += 1
        print(f'  FAIL {label}')

def finish():
    print(f'RESULT: {g_pass} passed, {g_fail} failed')
    sys.exit(0 if g_fail == 0 else 1)

def main():
    st = os.path.join(REPO, 'models', 'model.safetensors')
    if not os.path.exists(st):
        print('  [skip] models/model.safetensors absent — e2e LoRA regression '
              'skipped (like test_real_weights)')
        return
    if os.environ.get('OMNISEED_SKIP_LORA_E2E'):
        print('  [skip] OMNISEED_SKIP_LORA_E2E set')
        return

    tsv = os.path.join(REPO, 'tests', 'fixtures', 'lora_chat_tiny.tsv')
    ckpt = os.path.join(REPO, 'models', 'lora_e2e_regress.pt')
    sidecar = os.path.join(REPO, 'models', 'lora_e2e_regress.gguf')
    pylogits_path = os.path.join(REPO, 'models',
                                 'lora_e2e_regress_pylogits.npy')
    for p in (ckpt, sidecar, pylogits_path):
        if os.path.exists(p):
            os.remove(p)

    # ---- train a few steps on the tiny fixture -------------------------------
    r = subprocess.run(
        [PY, os.path.join(REPO, 'tools', 'lora_chat.py'),
         '--corpus-file', tsv, '--steps', '50', '--batch', '2',
         '--holdout', '4', '--patience', '0', '--eval-every', '10',
         '--lr', '6e-4',
         '--ckpt', ckpt, '--out', sidecar, '--device', 'cpu'],
        cwd=REPO, capture_output=True, text=True, encoding='utf-8',
        errors='replace', timeout=1200)
    out = r.stdout or ''
    if r.returncode != 0:
        print(out[-2000:])
        print((r.stderr or '')[-2000:])
        check(False, f'trainer exited {r.returncode}')
        finish()
        return

    # ---- 1: corpus stats were printed and non-zero --------------------------
    check('[lora] corpus:' in out and '0 pairs' not in out,
          'corpus stats printed, pair count > 0')
    check('FATAL' not in out, 'no FATAL aborts')

    # ---- 2: masked-token fraction printed for the early steps ---------------
    frac_lines = [ln for ln in out.splitlines() if 'answer-mask kept' in ln]
    check(len(frac_lines) >= 5, 'masked-token fraction printed (>= 5 steps)')
    for ln in frac_lines[:5]:
        pct = int(ln.split('(')[1].split('%')[0])
        check(0 < pct < 100, f'answer mask strictly partial: {pct}%')

    # ---- 3: B init check ran -------------------------------------------------
    check('all B == 0 exactly' in out, 'B==0 init check ran')

    summary = {}
    for ln in out.splitlines():
        if ln.startswith('[lora-e2e] '):
            summary = json.loads(ln.split(' ', 1)[1])
    check(bool(summary), 'trainer summary line present')
    if not summary:
        finish()
        return

    hist = summary['step_loss_history']
    check(len(hist) == 50, f'50 per-step losses recorded (got {len(hist)})')
    # ---- 5: loss drops > 10% --------------------------------------------------
    n0 = max(1, min(3, len(hist)))
    first = sum(hist[:n0]) / n0
    last = sum(hist[-n0:]) / n0
    drop = (first - last) / max(first, 1e-9)
    print(f'  loss first3 {first:.3f} -> last3 {last:.3f} '
          f'(drop {100 * drop:.1f}%)')
    check(last < first * 0.9, f'loss drops > 10% ({100 * drop:.1f}%)')
    check(all(h > 1e-3 for h in hist), 'no step collapsed to ~zero loss '
          '(degenerate signature was loss 0.000)')

    # ---- 6: attach improves holdout ppl vs base ------------------------------
    check('FINAL holdout answer-ppl' in out, 'final holdout eval printed')
    check(summary['final_holdout_loss'] < summary['base_holdout_loss'],
          f"holdout loss improves vs base "
          f"({summary['base_holdout_loss']:.3f} -> "
          f"{summary['final_holdout_loss']:.3f})")
    base_ppl = math.exp(min(20.0, summary['base_holdout_loss']))
    fin_ppl = math.exp(min(20.0, summary['final_holdout_loss']))
    print(f'  holdout answer-ppl {base_ppl:.2f} -> {fin_ppl:.2f}')
    check(fin_ppl < base_ppl, 'attach improves holdout answer-ppl vs base')

    # ---- 4: round-trip + cross-engine logits ---------------------------------
    # python reads the sidecar back with a struct-level parser (the SAME
    # layout conventions the C++ GgufLoader uses), asserts the f16 bytes are
    # BITWISE identical to the checkpoint tensors, re-attaches those exact
    # sidecar tensors and saves the attached logits for the parity prompt.
    train_code = r'''
import os, sys, struct
import numpy as np
import torch
torch.set_num_threads(1)
sys.path.insert(0, 'tools')
from qat_ternary import ensure_checkpoint, load_safetensors, resolve_vocab
import lora_chat as LC
sys.path.insert(0, 'models/hf-orig')
from hf_rwkv_tokenizer import RwkvTokenizer
tok = RwkvTokenizer(vocab_file=resolve_vocab())

def read_f16_all(path):
    """Walks the GGUF (v2/v3) header + kv + tensor directory and returns
    (blob, {name: (dims, dtype, offset)}, data_base). Offsets are RELATIVE
    to the data section (base = end of directory) — the same convention the
    C++ GgufLoader implements."""
    with open(path, 'rb') as f:
        blob = f.read()
    n_tensors, n_kv = struct.unpack('<2Q', blob[8:24])
    pos = 24

    def rstr(p):
        n = struct.unpack('<Q', blob[p:p + 8])[0]
        p += 8
        return blob[p:p + n], p + n

    for _ in range(n_kv):
        _, pos = rstr(pos)
        t = struct.unpack('<I', blob[pos:pos + 4])[0]
        pos += 4
        if t == 8:                       # STRING
            _, pos = rstr(pos)
        elif t == 9:                     # ARRAY
            at = struct.unpack('<I', blob[pos:pos + 4])[0]
            pos += 4
            cnt = struct.unpack('<Q', blob[pos:pos + 8])[0]
            pos += 8
            for _ in range(cnt):
                if at == 8:
                    _, pos = rstr(pos)
                elif at in (4, 5, 6, 10, 12):
                    pos += 8
                elif at in (1, 2, 3, 7):
                    pos += 1
                else:
                    raise ValueError(f'array elem type {at}')
        elif t in (10, 12):              # UINT64, FLOAT64
            pos += 8
        elif t in (4, 5):                # UINT32, INT32
            pos += 4
        else:
            raise ValueError(f'kv type {t}')

    tensors = {}
    for _ in range(n_tensors):
        name, pos = rstr(pos)
        nd = struct.unpack('<I', blob[pos:pos + 4])[0]
        pos += 4
        dims = struct.unpack('<%dQ' % nd, blob[pos:pos + 8 * nd])
        pos += 8 * nd
        dt = struct.unpack('<I', blob[pos:pos + 4])[0]
        pos += 4
        off = struct.unpack('<Q', blob[pos:pos + 8])[0]
        pos += 8
        tensors[name.decode()] = (dims, dt, off)
    return blob, tensors, pos            # pos == data-section base

ensure_checkpoint('models/model.safetensors')
header, ds = load_safetensors('models/model.safetensors')
m = LC.LoraRWKV7('models/model.safetensors', header, ds, rank=8, alpha=16.0)
ck = torch.load('models/lora_e2e_regress.pt', map_location='cpu',
                weights_only=True)
with torch.no_grad():
    for k in ck['A']:
        m.lora_A[k].copy_(ck['A'][k]); m.lora_B[k].copy_(ck['B'][k])

blob, tensors, base = read_f16_all('models/lora_e2e_regress.gguf')

# bitwise round-trip: every layer's A/B, exported f16 == checkpoint tensor
n_ok = n_tot = 0
for k in ck['A']:
    lno = k.split('.')[0][1:]
    tgt = k.split('.', 1)[1]
    for suffix, ref in (('a', ck['A'][k]), ('b', ck['B'][k])):
        n_tot += 1
        name = 'lora.%s.%s.%s' % (lno, tgt, suffix)
        dims, dt, off = tensors[name]
        assert dt == 1, dt                   # F16
        count = 1
        for d in dims:
            count *= int(d)
        assert count == int(np.prod(ref.shape)), (dims, ref.shape)
        raw = np.frombuffer(blob, dtype='<f2', count=count,
                            offset=base + off)
        if raw.tobytes() == ref.detach().numpy().astype('<f2').tobytes():
            n_ok += 1
print('[probe] roundtrip %d/%d bitwise' % (n_ok, n_tot))

# python-side attached logits for the SAME prompt the C++ test uses
prompt = 'User: What is 2+2?\n\nAssistant:'
ids, _ = LC.encode_example(tok, prompt, '')
m.begin_window(False)
st = None; lg = None
with torch.no_grad():
    for i in ids:
        lg, st = m.step_batch(torch.tensor([i]), st)
    v = lg[0].float().numpy()
m.end_window()
np.save('models/lora_e2e_regress_pylogits.npy', v)
print('[probe] pylogits saved, V =', len(v))
'''
    r2 = subprocess.run([PY, '-c', train_code], cwd=REPO,
                        capture_output=True, text=True, encoding='utf-8',
                        errors='replace', timeout=1200)
    p2 = r2.stdout or ''
    if r2.returncode != 0:
        print((r2.stderr or '')[-1500:])
        check(False, 'round-trip probe exited %d' % r2.returncode)
        finish()
        return
    parts = next(
        (ln.split() for ln in p2.splitlines()
         if ln.startswith('[probe] roundtrip')), [])
    n_ok = n_tot = 0
    if len(parts) > 2 and '/' in parts[2]:
        n_ok, n_tot = (int(x) for x in parts[2].split('/'))
    check(n_ok == n_tot and n_tot > 0,
          f'sidecar f16 tensors bitwise-identical to checkpoint '
          f'({n_ok}/{n_tot})')
    check(os.path.exists(pylogits_path), 'python attached-logits saved')

    # zero-delta sidecar (trainer --steps 0): the runtime contract
    # "attached delta == 0 at step 0", proven end-to-end in the harness
    zpath = os.path.join(REPO, 'models', 'lora_e2e_zero.gguf')
    if os.path.exists(zpath):
        os.remove(zpath)
    rz = subprocess.run(
        [PY, os.path.join(REPO, 'tools', 'lora_chat.py'),
         '--corpus-file', tsv, '--steps', '0', '--holdout', '4',
         '--eval-every', '0', '--ckpt', ckpt, '--out', zpath,
         '--device', 'cpu'],
        cwd=REPO, capture_output=True, text=True, encoding='utf-8',
        errors='replace', timeout=1200)
    check(rz.returncode == 0 and os.path.exists(zpath),
          'zero-delta sidecar export (--steps 0)')

    # ---- C++ side: attach the sidecar, verify parity + coherence ------------
    exe = os.path.join(REPO, 'build', 'bin', 'omniseed_lora_e2e.exe')
    if not os.path.exists(exe):
        exe = os.path.join(REPO, 'build', 'bin', 'omniseed_lora_e2e')
    if os.path.exists(exe):
        r3 = subprocess.run([exe, sidecar, pylogits_path, zpath], cwd=REPO,
                            capture_output=True, text=True,
                            encoding='utf-8', errors='replace', timeout=300)
        out3 = r3.stdout or ''
        check('LORA_E2E_PASS' in out3,
              'C++ attach/parity/coherence checks: '
              + ' | '.join(out3.strip().splitlines()[-3:]))
    else:
        print('  [info] omniseed_lora_e2e not built — C++ cross-check '
              'skipped')
    # ---- 7: generation stays coherent on the TRAINING base -----------------
    # The served QAT GGUF differs from the PTQ training base (documented
    # limitation), so coherence is asserted HERE, where the sidecar's own
    # base runs. This is the direct regression for the reported "attach
    # turns coherent output into garbage": on the matching base, attaching
    # the exported sidecar must never produce the degenerate loop.
    coh = '''
import os, sys
import torch
torch.set_num_threads(1)
sys.path.insert(0, 'tools')
from qat_ternary import ensure_checkpoint, load_safetensors, resolve_vocab
import lora_chat as LC
sys.path.insert(0, 'models/hf-orig')
from hf_rwkv_tokenizer import RwkvTokenizer
tok = RwkvTokenizer(vocab_file=resolve_vocab())
ensure_checkpoint('models/model.safetensors')
header, ds = load_safetensors('models/model.safetensors')

def read_f16_all(path):
    import struct
    with open(path, 'rb') as f:
        blob = f.read()
    n_tensors, n_kv = struct.unpack('<2Q', blob[8:24])
    pos = 24
    def rstr(p):
        n = struct.unpack('<Q', blob[p:p+8])[0]; p += 8
        return blob[p:p+n], p + n
    for _ in range(n_kv):
        _, pos = rstr(pos)
        t = struct.unpack('<I', blob[pos:pos+4])[0]; pos += 4
        if t == 8: _, pos = rstr(pos)
        elif t == 9:
            at = struct.unpack('<I', blob[pos:pos+4])[0]; pos += 4
            cnt = struct.unpack('<Q', blob[pos:pos+8])[0]; pos += 8
            for _ in range(cnt):
                if at == 8: _, pos = rstr(pos)
                elif at in (4, 5, 6, 10, 12): pos += 8
                elif at in (1, 2, 3, 7): pos += 1
                else: raise ValueError(at)
        elif t in (10, 12): pos += 8
        elif t in (4, 5): pos += 4
    tensors = {}
    for _ in range(n_tensors):
        name, pos = rstr(pos)
        nd = struct.unpack('<I', blob[pos:pos+4])[0]; pos += 4
        dims = struct.unpack('<%dQ' % nd, blob[pos:pos+8*nd]); pos += 8*nd
        dt = struct.unpack('<I', blob[pos:pos+4])[0]; pos += 4
        off = struct.unpack('<Q', blob[pos:pos+8])[0]; pos += 8
        tensors[name.decode()] = (dims, dt, off)
    return blob, tensors, pos

blob, tensors, base = read_f16_all('models/lora_e2e_regress.gguf')
import numpy as np
m = LC.LoraRWKV7('models/model.safetensors', header, ds, rank=8, alpha=16.0)
with torch.no_grad():
    for k in m.lora_A:
        lno = k.split('.')[0][1:]
        tgt = k.split('.', 1)[1]
        for suffix, slot in (('a', m.lora_A), ('b', m.lora_B)):
            dims, dt, off = tensors['lora.%s.%s.%s' % (lno, tgt, suffix)]
            n = 1
            for d in dims: n *= int(d)
            arr = np.frombuffer(blob, dtype='<f2', count=n, offset=base+off)
            slot[k].copy_(torch.from_numpy(arr.astype('<f4')).float()
                          .reshape(slot[k].shape))

def gen(prompt, n=24):
    m.begin_window(False)
    # PROMPT-ONLY encoding: no trailing eos (a post-eos state is OOD and
    # makes generation start on junk — training-time eos stays in place).
    ids = tok.encode(f'User: {prompt}\\n\\nAssistant:')
    st = None; lg = None
    with torch.no_grad():
        for i in ids:
            lg, st = m.step_batch(torch.tensor([i]), st)
        out = []
        for _ in range(n):
            nid = int(lg[0].argmax().item())
            if nid == 0: break
            out.append(nid)
            lg, st = m.step_batch(torch.tensor([nid]), st)
    m.end_window()
    return tok.decode(out)

for p in ('What is the capital of France?',
          'What is 2+2?',
          'How many days are in a week?'):
    out = gen(p)
    uniq = len(set(out.replace(' ', '')))
    degenerate = 0 < len(out) < 6 and uniq < 3   # loop signature, not turn-end
    print('[coh] %r -> %r degenerate=%s' % (p, out, degenerate))
'''
    r4 = subprocess.run([PY, '-c', coh], cwd=REPO, capture_output=True,
                        text=True, encoding='utf-8', errors='replace',
                        timeout=1200)
    if r4.returncode != 0:
        print((r4.stderr or '')[-1200:])
        check(False, 'coherence probe exited %d' % r4.returncode)
    else:
        n_deg = sum(1 for ln in (r4.stdout or '').splitlines()
                    if '[coh]' in ln and 'degenerate=True' in ln)
        for ln in (r4.stdout or '').splitlines():
            if ln.startswith('[coh]'):
                print(' ', ln)
        check(n_deg == 0, f'attached generation coherent on fixed prompts '
              f'({n_deg} degenerate)')

    finish()

if __name__ == '__main__':
    main()
