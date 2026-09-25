#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — fetch_trading_corpus.py
#
#  Generates 2000 SYNTHETIC trading Q&A pairs (models/corpus/trading_chat.tsv)
#  for the combined assistant LoRA ("one sidecar, many skills"). Everything is
#  template-generated from public market vocabulary — no proprietary data, no
#  scraped content, no live prices. The corpus itself is therefore original
#  work; it is published under CC-BY-SA-3.0 to match the dolly-15k excerpts it
#  is merged with.
#
#  Categories (pair counts):
#    technical analysis ....... 600   (RSI, MACD, Bollinger, S/R, MAs)
#    position sizing / risk ... 400   (Kelly, stop-loss, drawdown)
#    news interpretation ...... 400   (Fed, earnings, macro prints)
#    signal reasoning ......... 400   (buy/sell/hold + rationale, conflicts)
#    portfolio review ......... 200   (rebalance advice, concentration)
#
#  Questions are drawn WITHOUT replacement from an explicit cross-product of
#  template axes (shuffled with a fixed seed) — uniqueness is by construction,
#  and main() asserts it. Answers are short and direct (assistant style): the
#  LoRA learns BREVITY from this corpus; long training answers make it ramble.
# =============================================================================

import itertools
import os
import random
import sys

OUT_DEFAULT = 'models/corpus/trading_chat.tsv'

TICKERS = ['AAPL', 'MSFT', 'TSLA', 'NVDA', 'AMZN', 'SPY', 'QQQ', 'XOM', 'JPM']
TICKER_NAMES = {
    'AAPL': 'Apple', 'MSFT': 'Microsoft', 'TSLA': 'Tesla', 'NVDA': 'Nvidia',
    'AMZN': 'Amazon', 'SPY': 'the S&P 500 ETF', 'QQQ': 'the Nasdaq-100 ETF',
    'XOM': 'Exxon Mobil', 'JPM': 'JPMorgan',
}
TIMEFRAMES = ['daily', 'weekly']


def _an(kind):
    return 'an' if kind[0].upper() in 'AEIOU' else 'a'


def _take(product_axes, n, rng):
    """Shuffle the cross-product of `product_axes` (fixed-seed rng) and return
    the first n tuples — unique question combos by construction."""
    pool = list(itertools.product(*product_axes))
    if len(pool) < n:
        raise SystemExit(f'[trading-corpus] FATAL: template axes give only '
                         f'{len(pool)} unique combos, need {n} — widen the '
                         f'axes')
    rng.shuffle(pool)
    return pool[:n]


def build_all(seed=7):
    """Returns a list of (category, question, answer) — 2000 pairs, unique."""
    rng = random.Random(seed)
    out = []

    # ------------------------------------------------------------------ TA ---
    # RSI (150): value x ticker x phrasing
    for v, t, ph in _take(
            [range(18, 83), TICKERS, range(2)], 150, rng):
        if v >= 70:
            a = (f' RSI {v} on {t} is overbought; momentum entries are late '
                 f'here and a pullback toward the mean is the '
                 f'higher-probability path.')
        elif v <= 30:
            a = (f' RSI {v} on {t} is oversold; downside momentum is '
                 f'stretched — wait for a reversal signal before buying '
                 f'rather than catching the knife.')
        else:
            a = (f' RSI {v} on {t} is neutral; it neither confirms nor '
                 f'rejects the trend — weight other signals more heavily.')
        q = (f'What does RSI {v} on {t} mean?',
             f'Interpret an RSI reading of {v} on {t}.')[ph]
        out.append(('technical-analysis', q, a))
    # MACD (150): kind x ticker x timeframe x verb
    MACD_KINDS = ['bullish crossover', 'bearish crossover',
                  'widening histogram']
    MACD_VERBS = ['How do I read', 'What does', 'Interpret']
    for kind, t, tf, vi in _take(
            [MACD_KINDS, TICKERS, TIMEFRAMES, range(3)], 150, rng):
        if kind == 'bullish crossover':
            core = (f'means the fast line crossed above the slow line — '
                    f'momentum is turning up on the {tf} chart; '
                    f'confirmation comes from price holding above the '
                    f'signal line')
        elif kind == 'bearish crossover':
            core = (f'means the fast line crossed below the slow line — '
                    f'momentum is turning down on the {tf} chart; tighten '
                    f'stops or stand aside until it reverses')
        else:
            core = (f'means the {tf}-chart trend is accelerating in its '
                    f'current direction; fading it is a low-probability '
                    f'trade')
        a = f' A MACD {kind} on {t} {core}.'
        q = (f'{MACD_VERBS[vi]} a MACD {kind} on {t} ({tf} timeframe)?',
             f'What does a MACD {kind} on {t} mean for {tf} charts?',
             f'Interpret a MACD {kind} on {t}, {tf} timeframe.')[vi]
        out.append(('technical-analysis', q, a))
    # Bollinger (150): band x ticker x timeframe x verb
    BOLL_POS = ['upper band', 'lower band', 'middle band']
    for pos, t, tf, vi in _take(
            [BOLL_POS, TICKERS, TIMEFRAMES, range(3)], 150, rng):
        if pos == 'upper band':
            a = (f' {t} tagging the upper Bollinger Band on the {tf} chart '
                 f'is statistically stretched; in trends it can ride the '
                 f'band, in ranges it usually snaps back to the middle '
                 f'band.')
        elif pos == 'lower band':
            a = (f' {t} tagging the lower Bollinger Band on the {tf} chart '
                 f'is stretched to the downside; in ranges expect a bounce '
                 f'to the middle band, in downtrends it can ride the band '
                 f'lower.')
        else:
            a = (f' {t} at the middle Bollinger Band (the 20-{tf}-bar '
                 f'average) is fair value for the recent window; breaks '
                 f'above or below it signal a bias shift.')
        q = (f'What does {t} touching the {pos} of the Bollinger Bands on '
             f'the {tf} chart tell me?',
             f'{t} is at the {pos} of the Bollinger Bands ({tf}). How do I read that?',
             f'How should I trade {t} at the Bollinger {pos} on a {tf} chart?')[vi]
        out.append(('technical-analysis', q, a))
    # support / resistance / MAs (150): subject x ticker x verb
    SR_SUBJ = ['support', 'resistance', 'the 50-day moving average',
               'the 200-day moving average']
    SR_VERBS = [
        'What is the significance of {s} for {t}?',
        'How do I use {s} when trading {t}?',
        'Why do traders watch {s} on {t}?',
        'What happens when {t} breaks {s}?',
        'Explain the role of {s} in {t}.']
    for s, t, vi in _take([SR_SUBJ, TICKERS, range(5)], 150, rng):
        if s == 'support':
            a = (f' {TICKER_NAMES[t]} support is a price zone where buying '
                 f'has repeatedly overwhelmed selling; a clean break below '
                 f'it flips that zone into resistance.')
        elif s == 'resistance':
            a = (f' {TICKER_NAMES[t]} resistance is a zone where selling has '
                 f'repeatedly capped rallies; a strong close above it turns '
                 f'that zone into support.')
        elif s.startswith('the 50-day'):
            a = (f' Price above the 50-day moving average on {t} is the '
                 f'classic intermediate uptrend filter; below it, longs are '
                 f'fighting the tape.')
        else:
            a = (f' The 200-day moving average on {t} defines the long-term '
                 f'trend; institutional money largely respects it, so '
                 f'crosses there draw outsized volume.')
        out.append(('technical-analysis', SR_VERBS[vi].format(s=s, t=t), a))

    # ---------------------------------------------------------------- risk ---
    # Kelly (150): win rate 41..70 x payoff 1..3
    for p, b in _take([range(41, 71), [1.0, 1.5, 2.0, 2.5, 3.0]], 150, rng):
        k = max(0.0, (p / 100.0) - (1.0 - p / 100.0) / b)
        a = (f' Full Kelly for a {p}% win rate at {b:g}:1 payoff is '
             f'{100 * k:.1f}% of capital, but full Kelly assumes your '
             f'estimates are exact — they never are. Half Kelly '
             f'({100 * k / 2:.1f}%) is the defensible sizing.')
        out.append(('risk-management',
                    f'What is the Kelly bet size for a {p}% win rate and '
                    f'{b:g}:1 payoff?', a))
    # stops (145): kind x ticker x verb
    STOP_KINDS = ['ATR-based', 'percentage', 'structure-based']
    STOP_VERBS = [
        'Where should I place {a} {k} stop-loss on {t}?',
        'How tight should {a} {k} stop be on {t}?',
        'What is the logic of {a} {k} stop on {t}?',
        'Why use {a} {k} stop-loss for {t}?',
        'Set my {k} stop on {t} — where and why?',
        'How do I size {a} {k} stop on {t}?']
    for kind, t, vi in _take([STOP_KINDS, TICKERS, range(6)], 145, rng):
        if kind == 'ATR-based':
            a = (f' An ATR stop on {t} sits about 2 ATRs below entry: wide '
                 f'enough to survive normal noise, tight enough to cap the '
                 f'loss at a known multiple of daily volatility.')
        elif kind == 'percentage':
            a = (f' A fixed percentage stop on {t} (say 5%) is simple and '
                 f'enforceable, but it ignores volatility — in quiet names '
                 f'it is loose, in wild names it is far too tight.')
        else:
            a = (f' A structure stop on {t} goes just below the last swing '
                 f'low or breakout level: if that level fails, your reason '
                 f'for the trade has failed, so exiting is correct.')
        out.append(('risk-management',
                    STOP_VERBS[vi].format(a=_an(kind), k=kind, t=t), a))
    # drawdown (105): depth 8..35 x verb
    DD_VERBS = [
        'Why is a {d}% drawdown dangerous for compounding?',
        'How big a gain does a {d}% drawdown need to recover?',
        'What does a {d}% drawdown do to a portfolio?',
        'Is a {d}% drawdown acceptable?']
    for d, vi in _take([range(8, 36), range(4)], 105, rng):
        req = 100.0 * d / (100.0 - d)
        a = (f' A {d}% drawdown needs a {req:.0f}% gain just to break even '
             f'— drawdown math is asymmetric, which is why capping '
             f'per-trade risk at 1-2% and total exposure matters more than '
             f'any single win.')
        out.append(('risk-management', DD_VERBS[vi].format(d=d), a))

    # ----------------------------------------------------------------- news --
    # Fed / macro policy (140): event x verb x horizon
    FED_EVENTS = ['raises rates by 25bp', 'cuts rates by 25bp',
                  'holds rates steady', 'signals two cuts this year',
                  'signals a pause in hikes', 'accelerates QT']
    FED_VERBS = ['How should I interpret the Fed {e} {h} for {a} stocks?',
                 'What does it mean for markets when the Fed {e} {h} '
                 '({a} angle)?',
                 'The Fed {e} {h} — how do I position my {a} holdings?',
                 'How do {a} portfolios usually react when the Fed {e} {h}?']
    FED_HORIZON = ['this week', 'at the next meeting', 'per the minutes',
                   'into year-end']
    AUD = ['growth', 'value', 'bank', 'energy']
    for e, vi, h, aud in _take(
            [FED_EVENTS, range(4), FED_HORIZON, AUD], 140, rng):
        if 'raises' in e:
            a = (' A Fed hike tightens financial conditions: growth stocks '
                 'and long-duration assets usually fall first, banks may '
                 'benefit, and the dollar strengthens — reduce leverage '
                 'before the announcement, not after.')
        elif 'cuts' in e and '25bp' in e:
            a = (' A Fed cut eases financial conditions: risk assets '
                 'typically rally, borrowing costs fall, and the dollar '
                 'weakens — but cuts made during a crisis are a different, '
                 'more defensive signal than cuts in a soft landing.')
        elif 'cuts' in e:
            a = (' Signaled cuts are priced in the moment they are said; the '
                 'trade is in the surprise versus expectations, not the '
                 'headline — if the market already expects two cuts, the '
                 'statement itself moves little.')
        elif 'holds' in e:
            a = (' A Fed hold means conditions stay as they are; markets '
                 'trade on the forward guidance and the dot plot more than '
                 'the decision itself — position for volatility around the '
                 'press conference, not the headline.')
        elif 'pause' in e:
            a = (' A pause signal is a mid-cycle pivot: risk assets '
                 'typically rally on cheaper money ahead, but it also '
                 'signals the Fed sees weakness — expect chop, not a '
                 'one-way move.')
        else:
            a = (' Faster QT drains liquidity: higher term premia push '
                 'yields up, which hits long-duration growth and highly '
                 'levered names first — watch funding markets for stress.')
        q = FED_VERBS[vi].format(e=e, h=h, a=aud)
        out.append(('news-interpretation', q, a))
    # earnings (140): ticker x surprise x guidance x verb
    EARN_VERBS = ['What happens to {t} after it {s} earnings {g}?',
                  '{t} {s} earnings {g} — how should I read it?',
                  'How does {t} usually trade after {ger} earnings {g}?',
                  'Is {ger} earnings {g} a signal on {t}?']
    GER = {'beats': 'beating', 'misses': 'missing'}
    for t, s, g, vi in _take(
            [TICKERS, ['beats', 'misses'],
             ['and raises guidance', 'but cuts guidance'], range(4)],
            140, rng):
        if s == 'beats' and 'raises' in g:
            a = (f' A beat with raised guidance on {t} is the strongest '
                 f'signal in the set: the initial pop rewards holders and '
                 f'the trend typically extends — but size was already set '
                 f'before the print, which is the whole point of risk '
                 f'control.')
        elif s == 'beats':
            a = (f' When {TICKER_NAMES[t]} beats but cuts guidance, the pop '
                 f'often fades within days: the durable signal is guidance, '
                 f'not the print — hold through a beat-and-cut only with a '
                 f'hard stop.')
        elif 'raises' in g:
            a = (f' A miss with raised guidance on {t} usually gaps down '
                 f'then recovers if the guide is credible — the recovery '
                 f'requires trusting management, so wait for confirmation '
                 f'rather than averaging in blindly.')
        else:
            a = (f' When {TICKER_NAMES[t]} misses and cuts guidance, the '
                 f'first move is down and gap risk is real — this is '
                 f'exactly why positions are sized so no single report can '
                 f'hurt the portfolio materially.')
        out.append(('news-interpretation',
                    EARN_VERBS[vi].format(t=t, s=s, g=g, ger=GER[s]), a))
    # macro prints (120): item x verb x horizon
    MACRO_ITEMS = ['CPI comes in hotter than expected',
                   'CPI comes in cooler than expected',
                   'unemployment claims spike',
                   'PMI crosses above 50']
    MACRO_VERBS = ['What does it mean for markets when {i} {h}?',
                   'Markets react to {i} {h} — why?',
                   'How should I position when {i} {h}?',
                   'Why does {i} {h} move stocks?',
                   'Interpret {i} {h} for a trader.']
    HORIZONS = ['this morning', 'before the open', 'at the open',
                'mid-session', 'after the close', 'into the weekly close',
                'ahead of the Fed meeting']
    for item, vi, h in _take([MACRO_ITEMS, range(5), HORIZONS], 120, rng):
        if 'hotter' in item:
            a = (' Hot CPI raises the odds of tighter policy for longer: '
                 'bonds sell off, growth stocks compress, commodities and '
                 'value often hold up better on the day.')
        elif 'cooler' in item:
            a = (' Cool CPI supports the disinflation narrative: rate-cut '
                 'expectations rise, bonds rally, and long-duration growth '
                 'stocks typically lead.')
        elif 'unemployment' in item:
            a = (' Spiking claims point to a weakening labor market: '
                 'consumer discretionary and cyclicals lag while '
                 'defensives (utilities, staples) outperform.')
        else:
            a = (' PMI above 50 signals manufacturing expansion — a procyclical '
                 'signal that favors industrials, energy, and small caps '
                 'over bond proxies.')
        out.append(('news-interpretation',
                    MACRO_VERBS[vi].format(i=item.lower(), h=h), a))

    # --------------------------------------------------------------- signal --
    # verdicts (200): ticker x verdict x verdict-implying phrasing. Every
    # template presupposes its verdict, so question <-> answer stays
    # consistent and the (ticker, verdict, verb) cross-product is unique.
    SIG_Q = {
        'buy': [
            'Why is {t} a buy here?', 'Make the case for buying {t}.',
            'Give me your buy thesis on {t}.',
            'Is {t} a buy at current levels?',
            'Should I open a position in {t}?',
            'Any reason to buy {t} this week?',
            'What is the bull call on {t}?',
            'Do you like {t} as an entry here?',
            'Is now the time to buy {t}?',
            'Would you add {t} to a portfolio today?',
            'How would you trade {t} long?',
            'Set up a long trade on {t} for me.',
            'Give your one-line take on buying {t}.',
            'Where do you stand on {t}: long or flat?',
            'Is {t} worth buying right now?'],
        'sell': [
            'Why is {t} a sell here?', 'Make the case for selling {t}.',
            'Give me your sell thesis on {t}.',
            'Is {t} a sell at current levels?',
            'Should I close my {t} position?',
            'Any reason to exit {t} this week?',
            'What is the bear call on {t}?',
            'Do you like {t} as a short here?',
            'Is now the time to exit {t}?',
            'Would you trim {t} from a portfolio today?',
            'How would you trade {t} short?',
            'Set up an exit plan on {t} for me.',
            'Give your one-line take on selling {t}.',
            'Where do you stand on {t}: short or flat?',
            'Is {t} about to break down?'],
        'hold': [
            'Why is {t} a hold here?', 'Make the case for holding {t}.',
            'Give me your hold thesis on {t}.',
            'Is {t} a hold at current levels?',
            'Should I sit on my {t} position?',
            'Any reason to wait on {t} this week?',
            'What is the neutral call on {t}?',
            'Do you like waiting on {t} here?',
            'Is now the time to watch {t} instead of trading it?',
            'Would you keep {t} unchanged in a portfolio today?',
            'How would you trade {t} with patience instead of action?',
            'Set up a wait-and-see plan on {t} for me.',
            'Give your one-line take on holding {t}.',
            'Where do you stand on {t}: flat and waiting?',
            'Is {t} worth leaving alone right now?']}
    VERDICT_ANS = {
        'buy': (' BUY — trend filter up (price above the 50-day), momentum '
                'constructive (MACD above signal), and risk defined (stop '
                'just under the breakout level). Size the position so the '
                'stop costs at most 1% of the portfolio.'),
        'sell': (' SELL — momentum has rolled over (MACD bearish cross below '
                 'the signal line) and price lost the 50-day average; hope '
                 'is not a position, exit and re-evaluate from flat.'),
        'hold': (' HOLD — trend and momentum disagree (price strong, RSI '
                 'cooling), and the risk/reward at the current level is not '
                 'clearly favorable; wait for the chart to decide before '
                 'adding or trimming.')}
    for t, verdict, vi in _take(
            [TICKERS, ['buy', 'sell', 'hold'], range(15)], 200, rng):
        out.append(('signal-reasoning', SIG_Q[verdict][vi].format(t=t),
                    f' Verdict on {t}:' + VERDICT_ANS[verdict]))
    # conflicting signals (200): case x ticker x context
    SIG_CASES = ['RSI is oversold but the trend is down',
                 'MACD is bullish but price is below the 200-day',
                 'price is at resistance with a high RSI']
    SIG_CTX = ['on the daily chart', 'after a strong run', 'near earnings',
               'at the highs', 'after a gap down', 'in a quiet tape',
               'with volume rising', 'into month-end']
    for case, t, ctx in _take([SIG_CASES, TICKERS, SIG_CTX], 200, rng):
        if case.startswith('RSI'):
            a = (f' {t} oversold RSI inside a downtrend is a trap setup, not '
                 f'a buy signal: oversold gets more oversold in trends. '
                 f'Wait for a higher low or a trend-line break before '
                 f'acting.')
        elif case.startswith('MACD'):
            a = (f' {t} bullish MACD below the 200-day is a counter-trend '
                 f'signal at best — treat it as a trade, not an investment, '
                 f'with a hard stop and reduced size.')
        else:
            a = (f' {t} at resistance with a high RSI is a poor place to '
                 f'add: entries belong at support or after a confirmed '
                 f'breakout, not into a ceiling with stretched momentum.')
        out.append(('signal-reasoning',
                    f'How do I resolve conflicting signals on {t} {ctx}: '
                    f'{case}?', a))

    # ------------------------------------------------------------- portfolio --
    # rebalance (120): over-weight ticker x pct
    for over, pct in _take([TICKERS, range(22, 46)], 120, rng):
        a = (f' With {over} at {pct}% of the book, one earnings report can '
             f'move the whole portfolio — trim back toward your '
             f'per-position cap (10-20% is a common rule) and redeploy into '
             f'laggards or cash. Rebalancing is risk control, not a return '
             f'forecast.')
        out.append(('portfolio-review',
                    f'{TICKER_NAMES[over]} is {pct}% of my portfolio. '
                    f'Should I rebalance?', a))
    # concentration (80): n x verb
    N_VERBS = ['I hold {n} stocks. Is that too concentrated?',
               'Is a {n}-stock portfolio too concentrated?',
               'My portfolio has {n} positions — too many, too few?',
               'Is {n} individual stocks diversified enough?',
               'How risky is a {n}-stock portfolio?',
               'Thoughts on holding {n} stocks?']
    for n, vi in _take([range(4, 19), range(6)], 80, rng):
        a = (f' {n} positions is workable if each has a written thesis and '
             f'a stop; below ~8 names, single-stock risk dominates, above '
             f'~20 you get index-like returns with single-stock effort — '
             f'concentration should be a choice, not an accident.')
        out.append(('portfolio-review', N_VERBS[vi].format(n=n), a))

    rng.shuffle(out)                      # interleave categories (seeded)
    assert len(out) == 2000, len(out)
    return out


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else OUT_DEFAULT
    rows = build_all()
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    qs = [q for _, q, _ in rows]
    dupes = len(qs) - len(set(qs))
    if dupes:
        raise SystemExit(f'[trading-corpus] FATAL: {dupes} duplicate '
                         f'questions — template axes are too narrow')
    with open(out, 'w', encoding='utf-8', newline='\n') as f:
        f.write('# omniseed trading_chat.tsv — SYNTHETIC template corpus, '
                'CC-BY-SA-3.0 (matches dolly-15k excerpts)\n')
        for _, u, a in rows:
            f.write(f'{u}\t{a}\n')
    n = len(rows)
    mean_ans = sum(len(a.strip().split()) for _, _, a in rows) / max(1, n)
    cats = {}
    for name, _, _ in rows:
        cats[name] = cats.get(name, 0) + 1
    print(f'[trading-corpus] wrote {n} pairs -> {out} '
          f'(duplicate questions: {dupes})')
    print(f'[trading-corpus] mean answer length: {mean_ans:.1f} words')
    for name, cnt in sorted(cats.items()):
        print(f'[trading-corpus]   {name:22s} {cnt:5d} pairs')
    if n < 1500:
        raise SystemExit('[trading-corpus] FATAL: under 1500 pairs generated')
    if mean_ans > 60:
        raise SystemExit('[trading-corpus] FATAL: answers too long (>60 words '
                         'avg) — the LoRA would learn to ramble')


if __name__ == '__main__':
    main()
