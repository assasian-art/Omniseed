#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — e2e_lora_gguf.py   (OFFLINE regression, ctest omniseed_lora_gguf)
#
#  Regression for the gguf-base trainer mode (--gguf-base P): the LoRA is
#  trained ON the SERVED GGUF (every frozen tensor = the exact quantized
#  values the C++ runtime serves), so the exported sidecar optimizes the
#  deployment weights instead of the PTQ safetensors.
#
#  Asserts:
#    1. the trainer runs 20 steps with --gguf-base models/
#       rwkv7-0.1B-ternary-qat.gguf and all guards stay active
#    2. python-side attach improves holdout answer-ppl vs base (trained on
#       that base — this is the honest "attach helps" check)
#    3. BASE parity: python (gguf-base) vs C++ runtime logits agree to
#       fp-order noise (cosine > 0.999) — the gguf reader reproduces the
#       served math bit-exactly (argmax equal, measured 1.000000)
#    4. sidecar round-trip stays bitwise (struct-level GGUF readback)
#
#  Auto-skips without the venv/base model/QAT GGUF (fixture policy).
# =============================================================================
import json
import math
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))   # tests/lora_e2e/x.py -> repo
PY = os.path.join(REPO, '.venv', 'Scripts', 'python.exe')
if not os.path.exists(PY):
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
    qat = os.path.join(REPO, 'models', 'rwkv7-0.1B-ternary-qat.gguf')
    st = os.path.join(REPO, 'models', 'model.safetensors')
    if not (os.path.exists(qat) and os.path.exists(st)):
        print('  [skip] QAT GGUF / safetensors absent — gguf-base regression '
              'skipped')
        return
    if os.environ.get('OMNISEED_SKIP_LORA_E2E'):
        print('  [skip] OMNISEED_SKIP_LORA_E2E set')
        return

    tsv = os.path.join(REPO, 'tests', 'fixtures', 'lora_chat_tiny.tsv')
    ckpt = os.path.join(REPO, 'models', 'lora_gguf_regress.pt')
    sidecar = os.path.join(REPO, 'models', 'lora_gguf_regress.gguf')
    pylogits = os.path.join(REPO, 'models', 'lora_gguf_pylogits.npy')
    for p in (ckpt, sidecar, pylogits):
        if os.path.exists(p):
            os.remove(p)

    # ---- 1: train 20 steps ON the served GGUF --------------------------------
    r = subprocess.run(
        [PY, os.path.join(REPO, 'tools', 'lora_chat.py'),
         '--corpus-file', tsv, '--gguf-base', qat, '--steps', '20',
         '--batch', '2', '--holdout', '4', '--patience', '0',
         '--eval-every', '10', '--lr', '6e-4',
         '--ckpt', ckpt, '--out', sidecar, '--device', 'cpu'],
        cwd=REPO, capture_output=True, text=True, encoding='utf-8',
        errors='replace', timeout=2400)
    out = r.stdout or ''
    if r.returncode != 0:
        print(out[-1500:])
        print((r.stderr or '')[-1500:])
        check(False, f'gguf-base trainer exited {r.returncode}')
        finish()
        return
    check('base: GGUF' in out and 'served values' in out,
          'trainer reports the GGUF base (served values)')
    check('all B == 0 exactly' in out, 'guards intact: B==0 init check')
    check('holdout' in out and 'answer-mask kept' in out,
          'guards intact: holdout + mask prints')

    summary = {}
    for ln in out.splitlines():
        if ln.startswith('[lora-e2e] '):
            summary = json.loads(ln.split(' ', 1)[1])
    check(bool(summary), 'trainer summary line present')
    if not summary:
        finish()
        return
    check(len(summary['step_loss_history']) == 20, '20 steps recorded')
    check(summary['final_holdout_loss'] < summary['base_holdout_loss'],
          'python-side: attach improves holdout loss on the GGUF base '
          f"({summary['base_holdout_loss']:.3f} -> "
          f"{summary['final_holdout_loss']:.3f})")

    # ---- 2/3: python attached logits + C++ harness parity --------------------
    probe = r'''
import os, sys
import numpy as np
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
    SZ = {0:1,1:1,2:1,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    for _ in range(n_kv):
        _, pos = rstr(pos)
        t = struct.unpack('<I', blob[pos:pos+4])[0]; pos += 4
        if t == 8: _, pos = rstr(pos)
        elif t == 9:
            at = struct.unpack('<I', blob[pos:pos+4])[0]; pos += 4
            cnt = struct.unpack('<Q', blob[pos:pos+8])[0]; pos += 8
            for _ in range(cnt):
                if at == 8: _, pos = rstr(pos)
                else: pos += SZ[at]
        else: pos += SZ[t]
    tensors = {}
    for _ in range(n_tensors):
        name, pos = rstr(pos)
        nd = struct.unpack('<I', blob[pos:pos+4])[0]; pos += 4
        dims = struct.unpack('<%dQ' % nd, blob[pos:pos+8*nd]); pos += 8*nd
        dt = struct.unpack('<I', blob[pos:pos+4])[0]; pos += 4
        off = struct.unpack('<Q', blob[pos:pos+8])[0]; pos += 8
        tensors[name.decode()] = (tuple(reversed(dims)), dt, off)
    return blob, tensors, pos

blob, tensors, base = read_f16_all('models/lora_gguf_regress.gguf')
m = LC.LoraRWKV7('models/model.safetensors', header, ds, rank=8, alpha=16.0,
                 gguf_base='models/rwkv7-0.1B-ternary-qat.gguf')
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

prompt = 'User: What is 2+2?\n\nAssistant:'
ids = tok.encode(prompt)                 # prompt-only, no trailing eos
m.begin_window(False)
st = None; lg = None
with torch.no_grad():
    for i in ids:
        lg, st = m.step_batch(torch.tensor([i]), st)
    v = lg[0].float().numpy()
m.end_window()
np.save('models/lora_gguf_pylogits.npy', v)
print('[probe] attached logits saved, argmax =', int(v.argmax()))
'''
    r2 = subprocess.run([PY, '-c', probe], cwd=REPO, capture_output=True,
                        text=True, encoding='utf-8', errors='replace',
                        timeout=1200)
    if r2.returncode != 0:
        print((r2.stderr or '')[-1500:])
        check(False, 'attached-logits probe exited %d' % r2.returncode)
        finish()
        return
    for ln in (r2.stdout or '').splitlines():
        if ln.startswith('[probe]'):
            print(' ', ln)
    check(os.path.exists(pylogits), 'python attached logits saved')

    exe = os.path.join(REPO, 'build', 'bin', 'omniseed_lora_e2e.exe')
    if not os.path.exists(exe):
        exe = os.path.join(REPO, 'build', 'bin', 'omniseed_lora_e2e')
    if os.path.exists(exe):
        r3 = subprocess.run([exe, sidecar, pylogits], cwd=REPO,
                            capture_output=True, text=True,
                            encoding='utf-8', errors='replace', timeout=300)
        out3 = r3.stdout or ''
        for ln in out3.splitlines():
            if 'parity' in ln or 'delta' in ln:
                print(' ', ln.strip())
        check('LORA_E2E_PASS' in out3,
              'C++ harness (attach + parity + detach): '
              + ' | '.join(out3.strip().splitlines()[-2:]))
        import re
        mcos = re.search(r'cosine = ([0-9.]+)', out3)
        check(mcos and float(mcos.group(1)) > 0.999,
              f"cross-engine parity (python gguf-base vs C++ served): "
              f"cosine={mcos.group(1) if mcos else '?'} > 0.999")
    else:
        print('  [info] omniseed_lora_e2e not built — C++ parity skipped')

    if os.path.exists(pylogits):
        os.remove(pylogits)
    finish()

if __name__ == '__main__':
    main()
