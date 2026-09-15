#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — e2e_lora_i8.py   (OFFLINE regression, ctest omniseed_lora_i8)
#
#  Regression for --gguf-base over an INT8-QUANTIZED GGUF
#  (models/rwkv7-0.1B-ternary.gguf: the 6 big linears per layer are stored
#  dtype-4 = int8 + per-row f32 scale, NOT the QAT file's packed ternary).
#  int8 has 127 levels — a ternary STE cannot reproduce it — so those
#  masters ride an identity passthrough (legal: the LoRA base is frozen).
#
#  Asserts:
#    1. the trainer runs 20 steps with --gguf-base on the i8 GGUF and
#       reports the int8-passthrough base ("X int8 masters passed through")
#    2. python-side attach improves holdout answer-loss vs base (trained on
#       that base — the honest "attach helps" check)
#    3. BASE parity: python (i8 gguf-base) vs C++ runtime serving THAT GGUF
#       agree to fp-order noise (cosine > 0.999, argmax equal) — the i8
#       reader reproduces the served math exactly
#    4. sidecar round-trip stays bitwise (struct-level GGUF readback)
#
#  Auto-skips without the venv/base model/i8 GGUF (fixture policy).
# =============================================================================
import json
import os
import struct
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
    i8gguf = os.path.join(REPO, 'models', 'rwkv7-0.1B-ternary.gguf')
    st = os.path.join(REPO, 'models', 'model.safetensors')
    if not (os.path.exists(i8gguf) and os.path.exists(st)):
        print('  [skip] i8 GGUF / safetensors absent — i8-base regression '
              'skipped')
        return
    if os.environ.get('OMNISEED_SKIP_LORA_E2E'):
        print('  [skip] OMNISEED_SKIP_LORA_E2E set')
        return

    tsv = os.path.join(REPO, 'tests', 'fixtures', 'lora_chat_tiny.tsv')
    ckpt = os.path.join(REPO, 'models', 'lora_i8_regress.pt')
    sidecar = os.path.join(REPO, 'models', 'lora_i8_regress.gguf')
    zero = os.path.join(REPO, 'models', 'lora_i8_regress_zero.gguf')
    pylogits = os.path.join(REPO, 'models', 'lora_i8_pylogits.npy')
    for p in (ckpt, sidecar, pylogits, zero):
        if os.path.exists(p):
            os.remove(p)

    # ---- 1: train 20 steps ON the int8 GGUF ----------------------------------
    r = subprocess.run(
        [PY, os.path.join(REPO, 'tools', 'lora_chat.py'),
         '--corpus-file', tsv, '--gguf-base', i8gguf, '--steps', '20',
         '--batch', '2', '--holdout', '4', '--patience', '0',
         '--eval-every', '10', '--lr', '6e-4',
         '--ckpt', ckpt, '--out', sidecar, '--device', 'cpu'],
        cwd=REPO, capture_output=True, text=True, encoding='utf-8',
        errors='replace', timeout=2400)
    out = r.stdout or ''
    if r.returncode != 0:
        print(out[-1500:])
        print((r.stderr or '')[-1500:])
        check(False, f'i8-base trainer exited {r.returncode}')
        finish()
        return
    check('base: GGUF' in out and 'int8 masters passed through' in out,
          'trainer reports the int8 passthrough base')
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
          'python-side: attach improves holdout loss on the i8 base '
          f"({summary['base_holdout_loss']:.3f} -> "
          f"{summary['final_holdout_loss']:.3f})")

    # ---- 2: sidecar round-trip is bitwise (struct-level GGUF readback) -------
    sys.path.insert(0, os.path.join(REPO, 'tools'))
    import torch                                    # noqa: E402
    torch.set_num_threads(1)
    import lora_chat as LC                          # noqa: E402
    import numpy as np                              # noqa: E402
    ck = torch.load(ckpt, map_location='cpu', weights_only=True)
    blob, _, tensors, base = LC._gguf_walk(sidecar)
    n_ok = n_tot = 0
    for k in ck['A']:
        lno = k.split('.')[0][1:]
        tgt = k.split('.', 1)[1]
        for suffix, ref in (('a', ck['A'][k]), ('b', ck['B'][k])):
            n_tot += 1
            name = 'lora.%s.%s.%s' % (lno, tgt, suffix)
            if name not in tensors:
                check(False, f'sidecar tensor present: {name}')
                n_tot = -1
                break
            dims, dt, off = tensors[name]
            assert dt == 1, dt                       # F16
            count = 1
            for d in dims:
                count *= int(d)
            raw = np.frombuffer(blob, dtype='<f2', count=count,
                                offset=base + off)
            if raw.tobytes() == ref.detach().numpy().astype('<f2').tobytes():
                n_ok += 1
        if n_tot < 0:
            break
    if n_tot > 0:
        check(n_ok == n_tot, f'sidecar round-trip bitwise ({n_ok}/{n_tot})')

    # ---- 3: cross-engine base parity (python i8-base vs C++ on that GGUF) ----
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
m = LC.LoraRWKV7('models/model.safetensors', header, ds, rank=8, alpha=16.0,
                 gguf_base='models/rwkv7-0.1B-ternary.gguf')
ck = torch.load('models/lora_i8_regress.pt', map_location='cpu',
                weights_only=True)
with torch.no_grad():
    for k in ck['A']:
        m.lora_A[k].copy_(ck['A'][k]); m.lora_B[k].copy_(ck['B'][k])
m.begin_window(False)
ids = [24281, 59, 29956, 59, 30031, 4600, 285, 44, 51, 64, 261, 5585,
       41693, 59, 261, 5585, 41693, 59]
st = None; lg = None
with torch.no_grad():
    for i in ids:
        lg, st = m.step_batch(torch.tensor([i]), st)
v = lg[0].float().numpy()
m.end_window()
np.save('models/lora_i8_pylogits.npy', v)
print('[probe] i8-base logits saved, argmax =', int(v.argmax()))
'''
    r2 = subprocess.run([PY, '-c', probe], cwd=REPO, capture_output=True,
                        text=True, encoding='utf-8', errors='replace',
                        timeout=1200)
    if r2.returncode != 0:
        print((r2.stderr or '')[-1500:])
        check(False, 'i8-base logits probe exited %d' % r2.returncode)
        finish()
        return
    for ln in (r2.stdout or '').splitlines():
        if ln.startswith('[probe]'):
            print(' ', ln)
    check(os.path.exists(pylogits), 'python i8-base logits saved')

    exe = os.path.join(REPO, 'build', 'bin', 'omniseed_lora_e2e.exe')
    if not os.path.exists(exe):
        exe = os.path.join(REPO, 'build', 'bin', 'omniseed_lora_e2e')
    if os.path.exists(exe):
        r3 = subprocess.run(
            [exe, sidecar, pylogits, '--model',
             os.path.relpath(i8gguf, REPO).replace('\\', '/')],
            cwd=REPO, capture_output=True, text=True, encoding='utf-8',
            errors='replace', timeout=300)
        out3 = r3.stdout or ''
        for ln in out3.splitlines():
            if 'parity' in ln or 'delta' in ln:
                print(' ', ln.strip())
        check('LORA_E2E_PASS' in out3,
              'C++ harness (parity + detach on the i8 GGUF): '
              + ' | '.join(out3.strip().splitlines()[-2:]))
        import re
        mcos = re.search(r'cosine = ([0-9.]+)', out3)
        check(mcos and float(mcos.group(1)) > 0.999,
              f"cross-engine parity (python i8-base vs C++ i8-serve): "
              f"cosine={mcos.group(1) if mcos else '?'} > 0.999")
    else:
        print('  [info] omniseed_lora_e2e not built — C++ parity skipped')

    if os.path.exists(pylogits):
        os.remove(pylogits)
    finish()

if __name__ == '__main__':
    main()
