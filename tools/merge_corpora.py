#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — merge_corpora.py
#
#  Builds the COMBINED assistant-LoRA corpus ("one sidecar, many skills"):
#  dolly-15k closed-QA excerpts (from --fetch-corpus / --corpus fetch) plus the
#  synthetic trading set (tools/fetch_trading_corpus.py) -> one deduplicated,
#  shuffled TSV that trains a single sidecar with everyday-English AND
#  trading reasoning.
#
#  CRITICAL: the trainer holds out the LAST N pairs (never trained on) and
#  batches in corpus order — an unshuffled concat would make the holdout
#  single-domain and bias early stop. Shuffling with a fixed seed therefore
#  happens HERE, not at train time.
# =============================================================================

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def read_pairs(path, label):
    """Read 'prompt<TAB>answer' lines; '#' comment lines skipped. Orphan
    continuation lines (answers containing literal newlines, which the dolly
    fetcher's 200-char cut can leave behind in older cached files) are
    re-attached to the previous pair's answer instead of dropped."""
    pairs, skipped, rejoined = [], 0, 0
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            if '\t' not in line:
                if pairs:
                    lb, u, a = pairs[-1]
                    pairs[-1] = (lb, u, a + ' ' + line.strip())
                    rejoined += 1
                else:
                    skipped += 1
                continue
            u, a = line.split('\t', 1)
            u, a = u.strip(), a.strip()
            if u and a:
                pairs.append((label, u, a))
            else:
                skipped += 1
    if rejoined:
        print(f'[merge] {label}: re-attached {rejoined} orphan continuation '
              f'lines to their answers')
    return pairs, skipped


def shingle_key(q):
    # FULL normalized question — a 28-char prefix shingle (right for dolly's
    # diverse questions) collapses whole template families in a synthetic
    # corpus ('What is the Kelly bet size f...' matches all 160 Kelly pairs).
    return ' '.join(q.lower().split())


def main():
    ap = argparse.ArgumentParser(
        description='Merge dolly + trading chat corpora into one combined TSV')
    ap.add_argument('--dolly', default='models/corpus/dolly-3000.tsv',
                    help='dolly-15k TSV from --fetch-corpus (default: '
                         'models/corpus/dolly-3000.tsv)')
    ap.add_argument('--trading', default='models/corpus/trading_chat.tsv',
                    help='synthetic trading TSV (default: '
                         'models/corpus/trading_chat.tsv)')
    ap.add_argument('--out', default='models/corpus/combined_chat.tsv')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--target', type=int, default=5000,
                    help='informational target; stats warn when far below')
    args = ap.parse_args()

    # Auto-generate the trading side if absent (dolly requires the network,
    # so it is NOT auto-fetched — the trainer's --corpus fetch does that).
    if not os.path.exists(args.trading):
        print(f'[merge] {args.trading} missing — generating it now '
              f'(tools/fetch_trading_corpus.py) ...')
        import fetch_trading_corpus as FTC
        os.makedirs(os.path.dirname(args.trading) or '.', exist_ok=True)
        rows = FTC.build_all()
        with open(args.trading, 'w', encoding='utf-8', newline='\n') as f:
            f.write('# omniseed trading_chat.tsv — SYNTHETIC template corpus, '
                    'CC-BY-SA-3.0\n')
            for _, u, a in rows:
                f.write(f'{u}\t{a}\n')
        print(f'[merge] generated {len(rows)} trading pairs')

    if not os.path.exists(args.dolly):
        raise SystemExit(
            f'[merge] FATAL: dolly corpus {args.dolly} not found — run '
            f'`tools/lora_chat.py --corpus fetch --fetch-corpus 3000` first '
            f'(needs the network) or pass --dolly PATH')

    dolly, dolly_skip = read_pairs(args.dolly, 'dolly')
    trading, trading_skip = read_pairs(args.trading, 'trading')
    print(f'[merge] dolly:   {len(dolly):5d} pairs '
          f'({dolly_skip} malformed lines skipped)  <- {args.dolly}')
    print(f'[merge] trading: {len(trading):5d} pairs '
          f'({trading_skip} malformed lines skipped)  <- {args.trading}')

    # ---- dedupe on question shingle (first-seen wins; dolly first keeps the
    # human-authored phrasing when a template question collides) -------------
    seen, combined = set(), []
    removed = 0
    for label, u, a in dolly + trading:
        k = shingle_key(u)
        if k in seen:
            removed += 1
            continue
        seen.add(k)
        combined.append((label, u, a))

    random.Random(args.seed).shuffle(combined)   # fixed seed = reproducible

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8', newline='\n') as f:
        f.write('# omniseed combined_chat.tsv — dolly-15k closed-QA excerpts '
                '(CC-BY-SA-3.0) + synthetic trading Q&A (CC-BY-SA-3.0); '
                'merged + shuffled with seed %d\n' % args.seed)
        for _, u, a in combined:
            f.write(f'{u.replace(chr(9), " ")}\t{a}\n')

    # ---- stats --------------------------------------------------------------
    n = len(combined)
    cats = {}
    words = 0
    for label, u, a in combined:
        cats[label] = cats.get(label, 0) + 1
        words += len(a.split())
    print(f'[merge] wrote {n} pairs -> {args.out} '
          f'(removed {removed} near-duplicate questions)')
    print(f'[merge]   dolly-15k excerpts  {cats.get("dolly", 0):5d} pairs')
    print(f'[merge]   trading (synthetic)  {cats.get("trading", 0):5d} pairs')
    print(f'[merge]   mean answer length: {words / max(1, n):.1f} words '
          f'(seed {args.seed}, shuffled)')
    if n < 2000:
        raise SystemExit(f'[merge] FATAL: combined corpus has only {n} pairs '
                         f'(< 2000) — check the inputs')
    if abs(n - args.target) > args.target * 0.25:
        print(f'[merge] WARNING: {n} pairs is far from the {args.target}-pair '
              f'target — growth skills (bangla etc.) will push it back up')


if __name__ == '__main__':
    main()
