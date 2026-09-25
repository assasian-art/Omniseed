#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — e2e_lora_chat.py   (OFFLINE regression, ctest omniseed_lora_chat)
#
#  "One sidecar, many skills": ONE LoRA trained on a COMBINED corpus (the
#  everyday-English fixture + trading rows) must answer BOTH everyday prompts
#  and trading prompts coherently — verified on the PTQ training base with
#  best-snapshot export.
#
#  Scope note (honest): python-side greedy generation on the SERVED QAT
#  ternary base is degenerate at fixture scale (loops; the runtime needs its
#  repetition penalty and the full-corpus recipe). So the mechanism on the
#  served base is owned by ctest omniseed_lora_gguf (attach improves the
#  trading-domain holdout, cross-engine parity, round-trip), while THIS test
#  owns generation coherence on the trainer's base — the same regime the
#  omniseed_lora_e2e suite validated. The --repeat-penalty flag added to the
#  trainer mirrors the C++ runtime's Phase-13 CTRL penalty exactly.
#
#  Asserts:
#    1. the trainer runs 90 steps on the full combined fixture (everyday
#       fixture + all 20 trading rows, answers compacted to assistant
#       brevity) with all guards active — ~4 epochs: between the ~2-epoch
#       semi-random point and the ~7-epoch template-collapse point
#    2. attach improves the TRADING-domain holdout (holdout = the last 6
#       pairs, all trading rows) — and is not the 1.00 memorized ppl
#    3. --sample (auto-resumes and attaches the trained checkpoint) answers
#       an assistant prompt and two trading prompts: non-empty, direct
#       (no rambling), no degenerate loops. Vocabulary overlap with the
#       fixture answers (incl. the Paris factual anchor) is printed
#       INFORMATIONALLY only: at fixture scale (~4 epochs) recall is
#       semi-random by the Phase-17 epoch map, and the fixture policy
#       forbids judging quality on a mechanism fixture — factual recall is
#       judged on the full corpus (docs/MODAL_QAT_GUIDE.md).
#
#  Auto-skips without the venv/base model (fixture policy).
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

def read_tsv(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.strip() and '\t' in line \
                    and not line.lstrip().startswith('#'):
                u, a = line.split('\t', 1)
                rows.append((u.strip(), a.strip()))
    return rows


def main():
    st = os.path.join(REPO, 'models', 'model.safetensors')
    if not os.path.exists(st):
        print('  [skip] safetensors absent — chat coherence regression skipped')
        return
    if os.environ.get('OMNISEED_SKIP_LORA_E2E'):
        print('  [skip] OMNISEED_SKIP_LORA_E2E set')
        return

    # Training set: the full combined fixture — everyday-English pairs first,
    # trading rows last, so the 6-pair holdout is entirely trading-domain.
    tiny = os.path.join(REPO, 'tests', 'fixtures', 'lora_chat_tiny.tsv')
    trfix = os.path.join(REPO, 'tests', 'fixtures', 'trading_chat.tsv')
    trows = read_tsv(trfix)
    tsv = os.path.join(REPO, 'models', 'lora_chat_fixture_combined.tsv')
    with open(tsv, 'w', encoding='utf-8', newline='\n') as f:
        for src in (tiny, trfix):
            with open(src, encoding='utf-8') as g:
                for line in g:
                    if line.strip() and not line.lstrip().startswith('#'):
                        f.write(line if line.endswith('\n') else line + '\n')
    ckpt = os.path.join(REPO, 'models', 'lora_chat_fixture.pt')
    sidecar = os.path.join(REPO, 'models', 'lora_chat_fixture.gguf')
    for p in (ckpt, sidecar):
        if os.path.exists(p):
            os.remove(p)

    # ---- 1: train 90 steps on the combined corpus (PTQ training base) --------
    r = subprocess.run(
        [PY, os.path.join(REPO, 'tools', 'lora_chat.py'),
         '--corpus-file', tsv, '--steps', '90',
         '--batch', '2', '--holdout', '6', '--patience', '0',
         '--eval-every', '30', '--lr', '6e-4',
         '--ckpt', ckpt, '--out', sidecar, '--device', 'cpu'],
        cwd=REPO, capture_output=True, text=True, encoding='utf-8',
        errors='replace', timeout=2400)
    out = r.stdout or ''
    if r.returncode != 0:
        print(out[-1500:])
        print((r.stderr or '')[-1500:])
        check(False, f'combined-corpus trainer exited {r.returncode}')
        finish()
        return
    check('base: PTQ' in out or 'model.safetensors' in out,
          'trainer reports the PTQ training base')
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
    check(summary.get('n_steps_recorded') == 90, '90 steps recorded')
    check(summary['final_holdout_loss'] < summary['base_holdout_loss'],
          'attach improves TRADING-domain holdout loss '
          f"({summary['base_holdout_loss']:.3f} -> "
          f"{summary['final_holdout_loss']:.3f})")
    ppl = math.exp(min(20.0, summary['final_holdout_loss']))
    check(ppl > 1.001,
          f'holdout answer-ppl {ppl:.2f} is not the memorized 1.00')

    # ---- 2/3: dual-domain sample coherence (checkpoint auto-attaches) --------
    rows = read_tsv(tiny) + trows

    def pick(word):
        for u, a in rows:
            if word.lower() in u.lower():
                return u, a
        raise SystemExit(f'fixture probe {word!r} not found')

    probes = [pick('capital of France'), pick('Kelly'),
              pick('rebalance')]
    r4 = subprocess.run(
        [PY, os.path.join(REPO, 'tools', 'lora_chat.py'),
         '--corpus-file', tsv, '--ckpt', ckpt, '--device', 'cpu',
         '--repeat-penalty', '1.2', '--repeat-window', '24',
         '--sample', '|'.join(p for p, _ in probes)],
        cwd=REPO, capture_output=True, text=True, encoding='utf-8',
        errors='replace', timeout=900)
    if r4.returncode != 0:
        print((r4.stderr or '')[-1200:])
    replies = [ln.split(':', 1)[1].strip()
               for ln in (r4.stdout or '').splitlines()
               if ln.startswith('[lora] reply:')]
    check(len(replies) == len(probes),
          f'sample probe answered all {len(probes)} prompts '
          f'(got {len(replies)})')

    _STOP = set(('the a an of is are to and or in on for with at by from as '
                 'that this it its be been will would should can could do '
                 'does did not no yes but than then more most only into over '
                 'about after before when what which how why who was were has '
                 'have had you your they them their usually typically '
                 'because while also just means mean').split())

    def content_words(text):
        ws = [w.strip('.,;:!?—-\'"()').lower() for w in text.split()]
        return [w for w in ws if len(w) > 3 and w not in _STOP]

    def degenerate(reply):
        low = reply.lower()
        toks = low.split()
        # loop signatures: comma-chains, token-pool collapse, or an n-gram
        # appearing 3+ times
        if reply.count(',') > 2:
            return True
        if len(toks) >= 6 and len(set(toks)) * 2 < len(toks):
            return True
        for n in (5, 6):
            grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
            if len(grams) - len(set(grams)) >= 2:
                return True
        return False

    for (prompt, want), reply in zip(probes, replies + [''] * len(probes)):
        tag = prompt[:34]
        check(bool(reply) and len(reply) <= 220,
              f'sample [{tag}…]: non-empty + direct (no rambling; '
              f'got {len(reply)} chars)')
        check(not degenerate(reply),
              f'sample [{tag}…]: no degenerate loops; reply = {reply!r}')
        kws = content_words(want)
        hits = [k for k in kws if k in reply.lower()]
        print(f'  [info] sample [{tag}…] vocab overlap '
              f'{len(hits)}/{len(kws[:6])}: reply = {reply!r}')
    # Factual recall (Paris) is INFORMATIONAL at fixture scale: the Phase-17
    # epoch map puts ~4-epoch fixture training in the semi-random band, and
    # the fixture policy forbids judging quality on a mechanism fixture.
    # Factual recall is judged on the full corpus recipe (MODAL_QAT_GUIDE).
    if replies:
        print(f'  [info] factual anchor (Paris) '
              f'{"present" if "paris" in replies[0].lower() else "ABSENT"}: '
              f'reply = {replies[0]!r}')

    finish()

if __name__ == '__main__':
    main()
