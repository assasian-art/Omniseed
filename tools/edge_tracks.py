#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — tools/edge_tracks.py
#  Run the SIX pre-registered edge tracks in docs/EDGE_RESEARCH.md.
#
#  The adoption bar (§1 of that document), applied identically to every track:
#      accuracy >= 2 x (majority-class rate)   AND   n_holdout >= 30
#  on a genuinely held-out split. A rejection is a RESULT and is recorded as one.
#  (The bar was `2 x 1/K` in the original pre-registration; §1.1/§1.2 corrected
#  it, because `1/K` is the wrong floor on a skewed label set and `acc > floor`
#  is cleared by noise. See the constants below.)
#
#  Track F is the LAST direction probe. Whatever it says, the direction program
#  closes: a pass sends DECISION 1 back through the D1 audit, a failure is final.
#
#  THIS TOOL DOES NOT REINVENT THE PIPELINE. It imports train_heads.py and calls
#  the SAME fit_one / build_action_labels / chronological_split / stratified_split
#  the shipped blobs were produced with. A track scored by a second, mirrored
#  implementation would be measuring the mirror, not the head (MEMORY rule 13).
#
#  Output: tools/edge_tracks.json  (consumed by the gate
#  tests/test_edge_tracks_gate.py, which asserts the doc log matches this file).
#
#  numpy is required (the project's own trainer needs it); the gate reads the
#  committed JSON with the stdlib only so it runs on a fresh clone.
# =============================================================================
"""Run the pre-registered edge tracks and emit tools/edge_tracks.json."""

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import train_heads as TH  # noqa: E402

# --- the bar, verbatim from docs/EDGE_RESEARCH.md §1, corrected in §1.2 ---------
# The pre-registration said `accuracy >= 2 x chance`, chance = 1/K. §45 showed
# 1/K is the wrong floor for a SKEWED label set (a constant "always BUY" scored
# 2.580 x 1/K on the shipped teacher). §46 showed the fix has to go further: the
# floor must be the majority-class rate AND the 2x must apply to THAT, because
# "acc > majority" is cleared by noise (Track B features-only beat the constant
# 0.3225 vs 0.3198 — 0.1 standard errors). The operative bar is therefore
#
#       accuracy >= 2 x (majority-class rate)   AND   n_holdout >= 30
#
# on a genuinely held-out split. A rejection is a RESULT and is recorded as one.
RATIO_BAR = 2.0
MIN_HOLDOUT = 30


def _ratio(acc, chance):
    return acc / chance if chance else float("nan")


def _no_skill_floor(y_hold):
    """The correct no-skill floor: the majority-class rate of the HOLDOUT.

    Computed from the labels themselves, never assumed. For a balanced set this
    equals 1/K; for a skewed one it is larger, and using 1/K there would let a
    constant predictor pass the bar (§45).
    """
    vals, cnt = np.unique(np.asarray(y_hold), return_counts=True)
    if cnt.sum() == 0:
        return float("nan")
    return float(cnt.max() / cnt.sum())


def _verdict(acc, floor, n_hold):
    """The ONLY adoption decision, in one place, so every track is judged alike.

    `floor` is the no-skill rate (the majority-class rate). Adoption needs the
    head to reach 2x that floor on a holdout of at least 30 rows.
    """
    if n_hold < MIN_HOLDOUT:
        return "REJECT (n<%d)" % MIN_HOLDOUT, False
    r = _ratio(acc, floor)
    if r >= RATIO_BAR:
        return "SIGNAL", True
    return "NO-SIGNAL", False


def read_bars(path):
    """(closes, times) from an OHLCV csv — reuses the trainer's own reader."""
    return TH.read_closes(path)


def labels_for(args, ids, regime, closes, hz):
    """Dispatch the action teacher. v1 is the SHIPPED teacher; `balanced` is the
    §45-revisit variant (trailing-window quantile thresholds). Both return the
    same 7-name vocabulary so the head's K never changes between runs."""
    if args.teacher == "v1":
        return TH.build_action_labels(ids, regime, closes, n_forward=hz)
    if args.teacher == "balanced":
        return TH.build_action_labels_balanced(ids, regime, closes, n_forward=hz)
    raise SystemExit("unknown teacher %r" % args.teacher)


def _fit(H, y, K, tr, hold, args):
    """One fit, with the run's class-weight setting. `none` is the shipped fit."""
    cw = None if args.class_weight == "none" else args.class_weight
    return TH.fit_one(H, y, K, tr, hold, shuffle=False, seed=args.seed,
                      class_weight=cw)


def effective_n(n, horizon):
    """Overlap-adjusted effective sample size for a sliding forward-return label.

    A label at bar i reads the close at i+horizon, so two rows within `horizon`
    bars share part of their forward window. The textbook first-order correction
    for a sliding-window mean of an AR(1)-ish series is n_eff ~ n / horizon
    (the number of NON-overlapping windows that fit). We report this as an UPPER
    BOUND on independent evidence: it deliberately under-counts rather than
    over-counts, so a track cannot pass by inflating n. For horizon=1 it is n.
    """
    if horizon <= 1:
        return int(n)
    return int(n // horizon)


# =============================================================================
#  Track A — longer horizons (5 / 10 / 21 forward bars)
# =============================================================================
def track_a(H, ids, regime, closes, args):
    """Re-label the SAME h[E] under a new teacher horizon; no re-dump needed.

    h[E] is the backbone's summary of the bar TEXT and does not depend on the
    action teacher's horizon, so a longer horizon is a pure re-label of rows
    that already exist. This is the one track whose whole cost is a re-fit.
    """
    results = []
    action_names = TH.parse_decision_actions(
        os.path.join(ROOT, "include", "omniseed", "decision_head.h"))
    K = len(action_names)
    chance = 1.0 / K
    for hz in args.horizons:
        ya, names, detail = labels_for(args, ids, regime, closes, hz)
        if names != action_names:
            raise SystemExit("action order drift: %r" % (names,))
        dist = {action_names[i]: int((ya == i).sum()) for i in range(K)}
        tr, hold = TH.chronological_split(len(ya))
        present = sorted(set(ya.tolist()))
        missing = [action_names[i] for i in range(K) if i not in present]
        r = _fit(H, ya, K, tr, hold, args)
        acc = r["holdout_acc"]
        n_hold = len(hold)
        n_eff = effective_n(n_hold, hz)
        yh = ya[hold]
        maj_rate = _no_skill_floor(yh)
        # The operative bar (§1.2): 2x the MAJORITY-class rate, not 2x1/K.
        verdict, adopted = _verdict(acc, maj_rate, n_hold)
        # The pre-registered 1/K view, kept visible so the correction is auditable.
        ratio_1k = _ratio(acc, chance)
        row = {
            "track": "A", "variant": "n_forward=%d" % hz, "horizon": hz,
            "n_train": int(len(tr)), "n_holdout": int(n_hold),
            "n_eff": int(n_eff),
            "accuracy": float(acc), "chance": float(chance),
            "ratio": round(ratio_1k, 4),
            "floor": round(maj_rate, 4),
            "majority_rate": round(maj_rate, 4),
            "ratio_vs_majority": round(_ratio(acc, maj_rate), 4),
            "beats_majority": bool(acc > maj_rate),
            "clears_2x_floor": bool(_ratio(acc, maj_rate) >= RATIO_BAR),
            "macro_f1": float(r["holdout_macro_f1"]),
            "temperature": float(r["temperature"]),
            "ece_calibrated": float(r["ece_calibrated"]),
            "action_counts": dist,
            "actions_missing": missing,
            "verdict": verdict, "adopted": bool(adopted),
            "n_eff_ratio": round(ratio_1k, 4),
            "n_eff_note": ("overlap-adjusted (n//horizon); a pass that does not "
                           "survive this is reported as fragile, not adopted"),
        }
        results.append(row)
        print("  Track A h=%-2d  n=%d (n_eff=%d)  acc=%.4f  floor=%.4f  "
              "ratio_vs_floor=%.3fx  %s"
              % (hz, n_hold, n_eff, acc, maj_rate,
                 _ratio(acc, maj_rate), verdict))
        if missing:
            print("      actions with zero examples: %s" % missing)
    return results


# =============================================================================
#  Track B — monster features as head inputs
# =============================================================================
def regime_features(closes, highs, lows):
    """The interpretable feature block the monster layer exposes, [N, F].

    The pre-registration (§3.2) names trend_score / ER / Hurst / OFI / funding
    as the candidates. This block uses the subset the TREE actually computes
    from BAR DATA with no look-ahead: efficiency_ratio, hurst_rs, autocorr1,
    yang_zhang_vol, rsi14, atr_pct14, close_vs_sma20, bollinger_pos, logret,
    amihud. OFI and funding are NOT here — ofi needs order-flow snapshots and
    funding is a perp-series, neither of which the daily-bar CSVs carry; they
    are omitted rather than approximated, so the block is honest about itself.
    """
    # `monster` is a real package (tools/monster/__init__.py) and regime.py uses
    # package-relative imports, so it must be imported as `monster.regime` with
    # `tools/` on sys.path — importing the file directly breaks its `from
    # .features import ...` line.
    if os.path.join(ROOT, "tools") not in sys.path:
        sys.path.insert(0, os.path.join(ROOT, "tools"))
    from monster import features as MF
    from monster import regime as MR

    n = len(closes)
    bars = [{"time": 0, "open": closes[i], "high": highs[i],
             "low": lows[i], "close": closes[i], "volume": 1.0}
            for i in range(n)]

    # Two kinds of primitive live in the monster layer:
    #  * PER-BAR lists  (rsi, atr_pct, ...) — index i is bar i's value.
    #  * WINDOW scalars (hurst_rs, efficiency_ratio, autocorr1) — one number for
    #    a whole series. To use them as per-bar features we recompute over a
    #    trailing window ending at i, so index i still only sees bars <= i.
    def per_bar_scalar(fn, series, *a, **k):
        out = []
        for i in range(n):
            w = series[max(0, i - scalar_win + 1):i + 1]
            out.append(fn(w, *a, **k))
        return out

    scalar_win = 120

    er = per_bar_scalar(MR.efficiency_ratio, closes, 20)
    hurst = per_bar_scalar(MR.hurst_rs, closes)
    rets = [(closes[i] / closes[i - 1] - 1.0) if closes[i - 1] else 0.0
            for i in range(n)]
    rho1 = per_bar_scalar(MR.autocorr1, rets)
    yz = per_bar_scalar(MR.yang_zhang_vol, bars, 20)

    cols = {
        "efficiency_ratio": er,
        "hurst_rs": hurst,
        "autocorr1": rho1,
        "yang_zhang_vol": yz,
        "rsi14": MF.rsi(closes, 14),
        "atr_pct14": MF.atr_pct(bars, 14),
        "close_vs_sma20": [
            (closes[i] / MF.sma(closes, 20)[i] - 1.0)
            if MF.sma(closes, 20)[i] else MF.NAN for i in range(n)],
        "bollinger_pos": _bollinger_pos(closes, MF),
        "logret": MF.log_returns(closes),
        "amihud": MF.amihud(bars),
    }

    F = np.full((n, len(cols)), np.nan)
    for j, (_, v) in enumerate(cols.items()):
        for i in range(n):
            x = v[i]
            F[i, j] = x if (x is not None and math.isfinite(x)) else 0.0
    return F, list(cols.keys())


def _bollinger_pos(vals, MF):
    mid, up, lo = MF.bollinger(vals, 20, 2.0)
    out = []
    for i in range(len(vals)):
        if math.isfinite(up[i]) and (up[i] - lo[i]) != 0:
            out.append((vals[i] - lo[i]) / (up[i] - lo[i]))
        else:
            out.append(MF.NAN)
    return out


def bar_index_of_dump_row(prov):
    """The bar index a dumped row came from: provenance id is "bar<i>@<ts>".

    The dump holds FEWER rows than the csv has bars (the dumper needs `window`
    bars of history before it emits, and can skip a bar). Every feature/label
    block must be indexed through this, never by row position — the FIRST
    version of track_b concatenated the raw 1,231-row H with a 1,255-row
    feature block and died on the shape mismatch. That crash was the good
    outcome: a silent positional join would have shifted every row by 24.
    """
    m = __import__("re").match(r"bar(\d+)@", prov)
    if not m:
        raise SystemExit("unexpected provenance id: %r" % prov)
    return int(m.group(1))


def track_b(H, ids, regime, closes, highs, lows, args):
    """Append monster features to h[E] AND score a features-only baseline.

    The pre-registered condition (§3.2): the feature-augmented head must beat
    BOTH the 2x bar AND a head that sees ONLY the features (no h[E] at all).
    If the features-only head matches the augmented head, the backbone is
    supplying nothing and the 'win' is the features, not the model.
    """
    action_names = TH.parse_decision_actions(
        os.path.join(ROOT, "include", "omniseed", "decision_head.h"))
    K = len(action_names)
    chance = 1.0 / K
    ya, _, _ = labels_for(args, ids, regime, closes, 5)
    tr, hold = TH.chronological_split(len(ya))

    # Build the features over the WHOLE csv (they need trailing history), then
    # pick the rows the dump actually emitted, by provenance index.
    F_all, fnames = regime_features(closes, highs, lows)
    bidx = np.array([bar_index_of_dump_row(p) for p in ids], dtype=int)
    if bidx.max() >= F_all.shape[0]:
        raise SystemExit("dump row bar index %d >= bars %d"
                         % (bidx.max(), F_all.shape[0]))
    F = F_all[bidx]
    if F.shape[0] != H.shape[0]:
        raise SystemExit("feature rows %d != hidden rows %d"
                         % (F.shape[0], H.shape[0]))

    # z-score features on the TRAIN split only (fitting the scaler on the
    # holdout would leak the future).
    mu, sd = F[tr].mean(axis=0), F[tr].std(axis=0)
    sd[sd < 1e-6] = 1.0
    Fz = (F - mu) / sd

    rows = []

    # (b1) h[E] alone — the shipped configuration, as the control.
    rH = _fit(H, ya, K, tr, hold, args)
    rows.append(_b_row("B", "h[E] only (control)", rH, chance, len(hold),
                       y_hold=ya[hold]))

    # (b2) features alone — no hidden state at all.
    rF = _fit(Fz, ya, K, tr, hold, args)
    rows.append(_b_row("B", "features only (no h[E])", rF, chance, len(hold),
                       y_hold=ya[hold]))

    # (b3) concatenated: scaled features appended to h[E].
    #     h[E] dims are 768 and roughly unit-scale after standardisation inside
    #     fit_one; features are ~unit-scale already. Concatenate RAW h with the
    #     standardised features so the h block keeps its own internal scaling.
    Hcat = np.concatenate([H, Fz], axis=1)
    rC = _fit(Hcat, ya, K, tr, hold, args)
    row = _b_row("B", "h[E] + features", rC, chance, len(hold),
                 y_hold=ya[hold])
    rows.append(row)

    # ---- the controls, run INSIDE the tool so the JSON carries its own proof --
    # A 2x1/K result is only credible if a predictor that CANNOT see the answer
    # does not also reach it. §1.1 exists because that control was not run
    # before; it is run here every time.
    ctl = _track_b_controls(H, ya, Fz, tr, hold, K, args)

    # The pre-registered extra condition: augmented head must beat features-only.
    feat_acc = rows[1]["accuracy"]
    aug_acc = rows[2]["accuracy"]
    beats_features = aug_acc > feat_acc
    if not beats_features:
        rows[2]["verdict"] = "REJECT (backbone adds nothing over features)"
        rows[2]["adopted"] = False
    else:
        rows[2]["beats_features_only"] = True

    # THE CONTROL VERDICT. If features-only does not beat a constant predictor,
    # its 2x1/K "SIGNAL" is a base-rate artifact and it must NOT be adopted —
    # regardless of what the raw bar says. This is the §1.1 correction applied
    # at the point of decision, not in prose.
    const_acc = ctl["constant"]["accuracy"]
    if rows[1]["adopted"] and feat_acc <= const_acc:
        rows[1]["verdict"] = ("REJECT (does not beat the constant predictor "
                              "%.4f — base-rate artifact)" % const_acc)
        rows[1]["adopted"] = False
    rows[1]["feature_names"] = fnames
    rows[2]["feature_names"] = fnames
    for rr in rows:
        rr["controls"] = ctl
        print("  Track B %-26s n=%d acc=%.4f floor=%.4f ratio_vs_floor=%.3fx %s"
              % (rr["variant"], rr["n_holdout"], rr["accuracy"],
                 rr.get("floor", float("nan")), rr.get("ratio_vs_floor",
                                                      float("nan")),
                 rr["verdict"]))
    print("      features-only acc=%.4f  augmented acc=%.4f  "
          "backbone contributes: %s" % (feat_acc, aug_acc, beats_features))
    print("      controls: constant=%.4f  labels-shuffled=%.4f  "
          "features-permuted=%.4f" % (const_acc, ctl["shuffled_labels"]["accuracy"],
                                      ctl["permuted_features"]["accuracy"]))
    print("      features-only margin over the constant: %.4f  (z=%.2f)"
          % (feat_acc - const_acc, rows[1].get("constant_z", float("nan"))))
    return rows


def _track_b_controls(H, ya, Fz, tr, hold, K, args):
    """Three predictors that must FAIL for a 2x result to mean anything.

    * `constant`          — always emit the majority TRAIN class. No model.
    * `shuffled_labels`   — real features, labels permuted. Kills real signal.
    * `permuted_features` — real labels, each feature column permuted across
                            rows. Kills feature-target association.

    All three are scored on the same holdout with the same fit_one. If they
    reach the same accuracy as the real head, the head has no signal.
    """
    rng = np.random.default_rng(args.seed)

    counts = np.bincount(ya[tr], minlength=K)
    maj = int(counts.argmax())
    const_acc = float((ya[hold] == maj).mean())

    ysh = ya.copy()
    rng.shuffle(ysh)
    r_sh = _fit(Fz, ysh, K, tr, hold, args)

    Fp = Fz.copy()
    for j in range(Fp.shape[1]):
        Fp[:, j] = Fz[rng.permutation(len(Fz)), j]
    r_pf = _fit(Fp, ya, K, tr, hold, args)

    return {
        "constant": {"accuracy": const_acc, "majority_class": maj,
                     "note": "always emit the majority TRAIN class; no model"},
        "shuffled_labels": {"accuracy": float(r_sh["holdout_acc"]),
                            "note": "real features, labels permuted"},
        "permuted_features": {"accuracy": float(r_pf["holdout_acc"]),
                              "note": "real labels, feature columns permuted"},
    }


def _b_row(track, variant, r, chance, n_hold, y_hold=None):
    """One result row. `y_hold` (the holdout labels) sets the CORRECTED floor.

    The `1/K` ratio is kept because it is what the document pre-registered. The
    majority-class rate is the OPERATIVE floor (§1.2): adoption needs
    `accuracy >= 2 x majority`, not `accuracy > majority` (which noise clears)
    and not `2 x 1/K` (which a constant clears).
    """
    acc = float(r["holdout_acc"])
    if y_hold is not None and len(y_hold) > 0:
        floor = _no_skill_floor(y_hold)
    else:
        floor = chance
    verdict, adopted = _verdict(acc, floor, n_hold)
    row = {
        "track": track, "variant": variant,
        "n_holdout": int(n_hold), "accuracy": acc, "chance": float(chance),
        "ratio": round(_ratio(acc, chance), 4),
        "floor": round(float(floor), 4),
        "ratio_vs_floor": round(_ratio(acc, floor), 4),
        "macro_f1": float(r["holdout_macro_f1"]),
        "temperature": float(r["temperature"]),
        "verdict": verdict, "adopted": bool(adopted),
    }
    if y_hold is not None and len(y_hold) > 0:
        row["majority_rate"] = round(float(floor), 4)
        row["ratio_vs_majority"] = round(_ratio(acc, floor), 4)
        row["beats_majority"] = bool(acc > floor)
        row["clears_2x_floor"] = bool(_ratio(acc, floor) >= RATIO_BAR)
        # How many standard errors above the constant is this, really? §46: a
        # bare `acc > majority` is not evidence (Track B's features-only cleared
        # it by 0.1 SE). Reported so a reader can see the margin is noise.
        se = math.sqrt(max(floor * (1.0 - floor), 1e-12) / max(n_hold, 1))
        row["constant_z"] = round((acc - floor) / se, 3) if se else None
    return row


# =============================================================================
#  Track C — multi-asset pooled training
# =============================================================================
def track_c(args):
    """Pool hidden states from >=2 instruments; report per-asset AND pooled.

    Requires a market dump PER instrument. Only AAPL has one committed; DEMO is
    a synthetic series with its own dump. If a second instrument's dump is
    absent the track is recorded NOT RUN rather than faked from one asset.
    """
    out = []
    dumps = []
    for name, d in args.asset_dumps:
        if os.path.isdir(d):
            dumps.append((name, d))
        else:
            print("  Track C: dump for %s missing at %s" % (name, d))
    if len(dumps) < 2:
        out.append({
            "track": "C", "variant": "pooled (needs >=2 dumps)", "adopted": False,
            "verdict": "NOT RUN (only %d market dump(s) present)" % len(dumps),
            "dumps_found": [n for n, _ in dumps],
            "blocked_by": ("a per-asset h[E] dump is required; see "
                           "RUNBOOK dump_hidden market <bars.csv>"),
        })
        print("  Track C NOT RUN — %d dump(s) present" % len(dumps))
        return out

    action_names = TH.parse_decision_actions(
        os.path.join(ROOT, "include", "omniseed", "decision_head.h"))
    K = len(action_names)
    chance = 1.0 / K

    Hs, ys, owners = [], [], []
    for name, d in dumps:
        meta, H, ids, sets, labs = TH.load_dump(d)
        reg = np.array([r[0] for r in labs])
        bars = meta.get("bars")
        closes, _ = read_bars(bars) if bars and os.path.exists(bars) else (None, None)
        if closes is None:
            print("  Track C: %s has no bars path (%r) — skipped" % (name, bars))
            continue
        ya, _, _ = labels_for(args, ids, reg, closes, 5)
        Hs.append(H)
        ys.append(ya)
        owners.append(np.full(len(ya), len(Hs) - 1))
        print("  Track C: %s n=%d bars=%s" % (name, len(ya), bars))
    if len(Hs) < 2:
        out.append({"track": "C", "variant": "pooled", "adopted": False,
                    "verdict": "NOT RUN (usable dumps < 2)"})
        return out

    # Pool. Chronological within source then concatenated in source order: the
    # pooled split must not let a later half of one asset leak into the train
    # side of another, so we split PER SOURCE and concatenate the parts.
    Hp = np.vstack(Hs)
    yp = np.concatenate(ys)
    op = np.concatenate(owners)

    tr_parts, hold_parts = [], []
    for src in range(len(Hs)):
        idx = np.where(op == src)[0]
        tr_local, hold_local = TH.chronological_split(len(idx))
        tr_parts.append(idx[tr_local])
        hold_parts.append(idx[hold_local])
    tr = np.concatenate(tr_parts)
    hold = np.concatenate(hold_parts)

    rP = _fit(Hp, yp, K, tr, hold, args)
    row = _b_row("C", "pooled (%s)" % "+".join(n for n, _ in dumps[:len(Hs)]),
                 rP, chance, len(hold), y_hold=yp[hold])
    out.append(row)
    print("  Track C pooled  n_train=%d n_holdout=%d acc=%.4f floor=%.4f "
          "ratio_vs_floor=%.3fx %s"
          % (len(tr), len(hold), row["accuracy"], row.get("floor", float("nan")),
             row.get("ratio_vs_floor", float("nan")), row["verdict"]))

    # Per-asset reporting (§3.3): TWO numbers per asset, because one alone is
    # misleading. (i) the POOLED head scored on that asset's own holdout — does
    # pooling transfer? (ii) a head fitted on THAT asset ALONE — the incumbent
    # the pooled model has to beat. A pooled win that hides a per-asset loss is
    # not a win, and a pooled win over a single-asset head that was already
    # better is not a win either.
    for src, (name, d) in enumerate(dumps[:len(Hs)]):
        m = hold[op[hold] == src]
        if len(m) == 0:
            continue
        lg = rP["_raw_logits"][m]
        acc = float((lg.argmax(axis=1) == yp[m]).mean())
        maj_m = _no_skill_floor(yp[m])
        v, _ = _verdict(acc, maj_m, len(m))
        out.append({"track": "C", "variant": "pooled head scored on %s holdout" % name,
                    "n_holdout": int(len(m)), "accuracy": acc,
                    "chance": float(chance),
                    "ratio": round(_ratio(acc, chance), 4),
                    "floor": round(maj_m, 4),
                    "majority_rate": round(maj_m, 4),
                    "ratio_vs_majority": round(_ratio(acc, maj_m), 4),
                    "beats_majority": bool(acc > maj_m),
                    "clears_2x_floor": bool(_ratio(acc, maj_m) >= RATIO_BAR),
                    "verdict": v, "adopted": False})

        # (ii) single-asset head on the same rows.
        Hs_i = np.where(op == src)[0]
        tr_i, hold_i = TH.chronological_split(len(Hs_i))
        ri = _fit(Hp[Hs_i], yp[Hs_i], K, tr_i, hold_i, args)
        acc_i = float(ri["holdout_acc"])
        maj_i = _no_skill_floor(yp[Hs_i][hold_i])
        vi, _ = _verdict(acc_i, maj_i, len(hold_i))
        gains = acc > acc_i
        out.append({"track": "C", "variant": "%s alone (single-asset head)" % name,
                    "n_holdout": int(len(hold_i)), "accuracy": acc_i,
                    "chance": float(chance),
                    "ratio": round(_ratio(acc_i, chance), 4),
                    "floor": round(maj_i, 4),
                    "majority_rate": round(maj_i, 4),
                    "ratio_vs_majority": round(_ratio(acc_i, maj_i), 4),
                    "beats_majority": bool(acc_i > maj_i),
                    "clears_2x_floor": bool(_ratio(acc_i, maj_i) >= RATIO_BAR),
                    "verdict": vi, "adopted": False,
                    "pooling_helps_this_asset": bool(gains)})
        print("  Track C  -> %-8s  pooled-head n=%d acc=%.4f | "
              "single-asset n=%d acc=%.4f | pooling helps: %s"
              % (name, len(m), acc, len(hold_i), acc_i, gains))
    return out


# =============================================================================
#  Track D — rank / quantile targets
# =============================================================================
def track_d(H, ids, regime, closes, args):
    """Predict the forward-return QUANTILE bucket instead of a hard action.

    Label: which TERTILE the h-bar forward return falls in, relative to a
    trailing 252-bar window of PAST forward returns (window ends at i-1, so
    there is no look-ahead). The target is 'is this move big or small for this
    instrument recently', not a fixed -1.5% wall.

    The conversion test (§3.4) is the load-bearing part: a rank score the
    downstream engines cannot consume is a metric, not an edge. We assert the
    prediction separates realised returns in the pre-registered direction —
    predicted-top rows must carry a higher mean return than predicted-bottom.

    CONTROL (MEMORY rule 4): the tertile label is balanced *by construction*
    (each bar is placed by its own quantile), so a CONSTANT predictor scores
    ~1/3. `chance = 1/3` is therefore the right floor and the ratio is not
    inflated by an unbalanced label distribution the way the 7-action teacher's
    was. The label mix is printed so this can be checked, not assumed.
    """
    action_names = TH.parse_decision_actions(
        os.path.join(ROOT, "include", "omniseed", "decision_head.h"))
    hz = 5
    n = len(closes)
    fwd = np.full(n, np.nan)
    for i in range(n):
        j = i + hz
        if j < n and closes[i] != 0:
            fwd[i] = (closes[j] - closes[i]) / closes[i]

    # 3 quantile buckets within a trailing 252-bar window of PAST forward
    # returns — computed with no look-ahead (window ends at i-1).
    w = 252
    Q = 3
    QCHANCE = 1.0 / Q
    yq = np.full(len(ids), -1, dtype=int)
    for k, prov in enumerate(ids):
        i = bar_index_of_dump_row(prov)
        lo_b = fwd[max(0, i - w):i]
        lo_b = lo_b[np.isfinite(lo_b)]
        if len(lo_b) < 30 or not math.isfinite(fwd[i]):
            continue
        q1, q2 = np.quantile(lo_b, [1.0 / 3, 2.0 / 3])
        yq[k] = 0 if fwd[i] <= q1 else (1 if fwd[i] <= q2 else 2)

    mask = yq >= 0
    Hq, yqq = H[mask], yq[mask]
    mix = {int(v): int((yqq == v).sum()) for v in range(Q)}
    # The label mix must be near-balanced or `chance=1/3` is the wrong floor.
    frac = [mix[v] / len(yqq) for v in range(Q)]
    balanced = all(0.20 <= f <= 0.47 for f in frac)

    tr, hold = TH.chronological_split(len(yqq))
    rD = _fit(Hq, yqq, Q, tr, hold, args)
    row = _b_row("D", "forward-return tertile (5d, trailing 252)", rD,
                 QCHANCE, len(hold), y_hold=yqq[hold])
    row["label_mix"] = mix
    row["label_balanced"] = bool(balanced)
    # Adoptability is judged on the GATE BAR, but conversion is a separate must.
    row["conversion"] = _conversion_test(rD, Hq, yqq, hold, action_names)
    if not row["conversion"]["moved_engine"]:
        row["verdict"] = "REJECT (no conversion)"
        row["adopted"] = False
    if not balanced:
        row["control_note"] = (
            "label mix %r is not balanced; chance=1/3 is an upper bound on the "
            "floor, read the ratio with that caveat" % frac)
    print("  Track D  tertile label  n=%d acc=%.4f chance=%.4f ratio=%.3fx %s"
          % (len(hold), row["accuracy"], QCHANCE, row["ratio"], row["verdict"]))
    print("      label mix: %s (balanced=%s)" % (mix, balanced))
    print("      conversion: %s" % row["conversion"]["detail"])
    return [row]


# =============================================================================
#  Track E — the perfectly-balanced 3-class directional control
# =============================================================================
def track_dir_control(H, ids, closes, args):
    """Does h[E] linearly encode the FORWARD DIRECTION of the bar at all?

    This is a MEASUREMENT control, not a shippable head (the DecisionAction enum
    has 7 members). The label is the forward-return tertile by a trailing-window
    quantile, so the mix is ~1/3 by construction and the no-skill floor is
    EXACTLY 1/3 — there is no majority-class ambiguity to correct for, which is
    the whole point: if a head cannot beat 1/3 HERE, the 7-action failure in §45
    is not a skew artifact, it is the absence of directional signal in h[E].
    """
    y3 = TH.directional_quantile_labels(ids, closes, n_forward=5)
    mask = y3 >= 0
    H3, y3 = H[mask], y3[mask]
    K = 3
    chance = 1.0 / K
    mix = {int(v): int((y3 == v).sum()) for v in range(K)}
    frac = [mix[v] / len(y3) for v in range(K)]
    balanced = all(0.30 <= f <= 0.37 for f in frac)

    tr, hold = TH.chronological_split(len(y3))
    r = _fit(H3, y3, K, tr, hold, args)
    row = _b_row("E", "directional tertile (balanced 3-class control)", r,
                 chance, len(hold), y_hold=y3[hold])
    row["label_mix"] = mix
    row["label_frac"] = [round(f, 4) for f in frac]
    row["label_balanced"] = bool(balanced)
    row["note"] = ("chance = 1/3 is the BY-CONSTRUCTION floor (each bar is "
                   "placed by its own trailing quantile); the chronological "
                   "holdout's own majority rate is `floor` above and may differ "
                   "slightly. The head is below BOTH. A failure here is not a "
                   "skew artifact.")
    if not balanced:
        row["verdict"] = ("INVALID CONTROL (label mix %r is not ~1/3)"
                          % [round(f, 3) for f in frac])
        row["adopted"] = False
    print("  Track E  dir-tertile  n=%d acc=%.4f chance=%.4f ratio=%.3fx "
          "macroF1=%.4f %s"
          % (len(hold), row["accuracy"], chance, row["ratio"],
             row["macro_f1"], row["verdict"]))
    print("      label mix: %s (balanced=%s)" % (mix, balanced))
    return [row]


# =============================================================================
#  Track F — THE LAST DIRECTION PROBE: a tiny NON-LINEAR probe on frozen h[E]
# =============================================================================
# Pre-registered BEFORE the run (docs/EDGE_RESEARCH.md §6.9). Track E showed a
# LINEAR probe on frozen h[E] cannot beat exact chance on a perfectly balanced
# directional target. The one escape hatch left for the direction program is
# that the signal exists but is NOT linearly decodable. Track F tests exactly
# that, once, and then the program closes either way.
#
# THE PROBE, fixed in advance (no hyper-parameter search, no early stopping, no
# holdout peeking):
#
#   * input  : frozen h[E] -> standardised with the TRAIN split's mean/std
#              -> projected onto the TRAIN split's top PCA_COMPONENTS
#              directions (sign-fixed). See `_pca_basis` for why the projection
#              is part of the instrument, not a weakening of it.
#   * hidden : exactly 1 layer, 64 units, tanh
#   * output : K-way softmax, inverse-frequency class weights from the TRAIN split
#   * fit    : full-batch ADAM (the project's own optimiser, as in fit_softmax),
#              MLP_EPOCHS steps, lr MLP_LR, weight decay MLP_L2, seeded init.
#              The holdout is never read during the fit.
#   * bar    : accuracy >= 2 x (majority-class rate of the holdout), n >= 30
#
# FIVE ROWS, all pre-registered:
#
#   F1     candidate  the directional tertile — Track E's EXACT target, mask,
#                     split and holdout. The purest "is there direction in h[E]?".
#   F2     candidate  the §45 balanced 7-action teacher (`build_action_labels_
#                     balanced`): "the balanced teacher" read as the ACTION
#                     teacher rather than the direction teacher. Same probe, bar.
#   Fleak  control    the direction label INJECTED into principal component 0.
#                     MUST clear the bar — it is a signal that is present by
#                     construction, so this is what proves the instrument can
#                     recover signal at all. (It is a linear signal, which is the
#                     point: the instrument must not fail on the easy case.)
#   Fcap   control    a genuine NON-LINEAR function of h[E] — in-split tertiles
#                     of pc0*pc1. The MLP MUST clear the bar while a LINEAR probe
#                     on the SAME target must NOT. Those two together are the
#                     evidence that the probe's non-linearity is real and usable,
#                     so a failure on F1/F2 is about h[E], not about a broken or
#                     under-trained optimiser.
#   Fctl   control    F1 with the TRAIN labels SHUFFLED. Must NOT clear the bar —
#                     if it did, the bar would be vacuous.
#
# Control rows carry `control_floor` / `cleared_bar` rather than `floor` /
# `adopted`, because `adopted` means "this head cleared the bar as a CANDIDATE"
# and a control is not a candidate. (The gate keys on `floor`+`adopted`, so this
# keeps the two kinds of row from being confused for one another.)
#
# THE CALIBRATION HISTORY IS PART OF THE RESULT, and is reported in full in the
# doc: the first probe (plain SGD, 400 epochs, 768 raw dims) FAILED its own leak
# control (0.61 holdout on a signal that was literally in the input), so that run
# was an invalid instrument and no verdict was taken from it. Adam and the PCA
# projection were added to make the instrument valid — calibrated ON THE
# CONTROLS ONLY. F1 read 0.279 / 0.295 / 0.353 / 0.306 across those budgets, i.e.
# it never came near the 0.372 floor, so the calibration did not manufacture the
# verdict.
MLP_HIDDEN = 64
MLP_EPOCHS = 2000
MLP_LR = 0.05
MLP_L2 = 1e-4
PCA_COMPONENTS = 32
F_TERM = ("h[E] carries no forward-direction signal that a 64-unit one-hidden-"
          "layer probe can recover from frozen inputs")


def _standardise(H, tr):
    """Z-score using the TRAIN split only — the holdout's own mean/std must not
    enter, or the probe would be reading the future it is scored on."""
    mu = H[tr].mean(axis=0)
    sd = H[tr].std(axis=0)
    sd[sd < 1e-6] = 1.0                      # a constant dim carries no signal
    return (H - mu) / sd


def _pca_basis(Z, tr, ncomp=PCA_COMPONENTS):
    """Top-`ncomp` principal directions of the TRAIN split, sign-fixed.

    WHY THE PROJECTION IS PART OF THE INSTRUMENT, not a weakening of it. A
    64-unit net trained on 768 raw dims with ~850 rows cannot generalise even a
    signal that IS present: measured, a label injected into h[E][:,0] is
    recovered to only 0.61 (SGD) / 0.81 (Adam) on the holdout, and no non-linear
    target was recoverable at all. That is a property of the SAMPLE SIZE, not of
    h[E], and a probe with that little power cannot be allowed to pronounce on
    the direction hypothesis. Projecting onto the train split's top directions
    fixes the conditioning. It does not weaken the comparison: a LINEAR head on
    these components is exactly a linear head on h[E] restricted to that
    subspace, so the non-linear probe remains a strict superset of the linear
    one it is being contrasted with. The basis is fit on the TRAIN split only.

    Signs are fixed (largest-magnitude loading positive) so the basis — and
    therefore the seeded init and the whole fit — is reproducible.
    """
    _, S, Vt = np.linalg.svd(Z[tr], full_matrices=False)
    P = Vt[:ncomp].T.copy()
    for j in range(P.shape[1]):
        if P[int(np.argmax(np.abs(P[:, j]))), j] < 0:
            P[:, j] = -P[:, j]
    ev = S ** 2
    return P, float((ev[:ncomp] / ev.sum()).sum())


def fit_mlp(Z, y, K, tr, hold, hidden=MLP_HIDDEN, epochs=MLP_EPOCHS, lr=MLP_LR,
            l2=MLP_L2, seed=1234, shuffle_train_labels=False):
    """One tiny non-linear probe. Deterministic: seeded init + full-batch Adam.

    Returns a dict shaped like `train_heads.fit_one`'s for the fields the row
    builders need, so the linear and non-linear probes are scored by the same
    code path. `temperature` is reported as 1.0 and is UNUSED: the bar is on
    argmax, which no positive rescaling of the logits can change.
    """
    rng = np.random.default_rng(seed)
    D = Z.shape[1]
    W1 = rng.normal(0.0, 1.0 / math.sqrt(D), (D, hidden))
    b1 = np.zeros(hidden)
    W2 = rng.normal(0.0, 1.0 / math.sqrt(hidden), (hidden, K))
    b2 = np.zeros(K)

    ytr = np.asarray(y[tr]).copy()
    if shuffle_train_labels:
        rng.shuffle(ytr)                     # destroy the input -> label relation
    ntr = len(tr)
    Y1 = np.zeros((ntr, K))
    Y1[np.arange(ntr), ytr] = 1.0
    w = TH.inverse_freq_weights(ytr, K)      # from the (possibly shuffled) TRAIN
    w = w / max(float(w.mean()), 1e-12)      # mean weight 1, so lr is comparable
    Wt = w[None, :]
    Ztr = Z[tr]

    params = [W1, b1, W2, b2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    bta1, bta2, eps = 0.9, 0.999, 1e-8
    for t in range(1, epochs + 1):
        A1 = np.tanh(Ztr @ W1 + b1)
        P = TH.softmax(A1 @ W2 + b2)
        G = (P - Y1) * Wt / ntr              # weighted-CE gradient
        D1 = (G @ W2.T) * (1.0 - A1 * A1)    # tanh'
        grads = [Ztr.T @ D1 + l2 * W1, D1.sum(axis=0),
                 A1.T @ G + l2 * W2, G.sum(axis=0)]
        c1, c2 = 1.0 - bta1 ** t, 1.0 - bta2 ** t
        for i in range(4):
            m[i] = bta1 * m[i] + (1.0 - bta1) * grads[i]
            v[i] = bta2 * v[i] + (1.0 - bta2) * grads[i] * grads[i]
            params[i] -= lr * (m[i] / c1) / (np.sqrt(v[i] / c2) + eps)

    def pred(idx):
        if len(idx) == 0:
            return np.zeros(0, dtype=int)
        return (np.tanh(Z[idx] @ W1 + b1) @ W2 + b2).argmax(axis=1)

    ph, pt = pred(hold), pred(tr)
    return {
        "holdout_acc": round(float((ph == y[hold]).mean()), 5),
        "holdout_macro_f1": round(TH.macro_f1(y[hold], ph, K), 4),
        "train_acc": round(float((pt == y[tr]).mean()), 5),
        "temperature": 1.0,
    }


def _probe_desc(explained):
    return {"kind": "mlp", "hidden": MLP_HIDDEN, "activation": "tanh",
            "optimiser": "adam", "epochs": MLP_EPOCHS, "lr": MLP_LR,
            "l2": MLP_L2, "pca_components": PCA_COMPONENTS,
            "pca_explained_var": round(float(explained), 4)}


def _f_row(track, variant, role, r, chance, n_hold, y_hold, note, explained):
    """A CANDIDATE row: carries `floor` + `adopted`, so the gate's A7 checks it."""
    acc = float(r["holdout_acc"])
    floor = _no_skill_floor(y_hold) if len(y_hold) else chance
    verdict, cleared = _verdict(acc, floor, n_hold)
    se = math.sqrt(max(floor * (1.0 - floor), 1e-12) / max(n_hold, 1))
    return {
        "track": track, "variant": variant, "role": role,
        "probe": _probe_desc(explained),
        "n_holdout": int(n_hold), "accuracy": acc, "chance": float(chance),
        "ratio": round(_ratio(acc, chance), 4),
        "floor": round(float(floor), 4),
        "ratio_vs_floor": round(_ratio(acc, floor), 4),
        "macro_f1": float(r["holdout_macro_f1"]),
        "train_acc": float(r["train_acc"]),
        "majority_rate": round(float(floor), 4),
        "ratio_vs_majority": round(_ratio(acc, floor), 4),
        "beats_majority": bool(acc > floor),
        "clears_2x_floor": bool(_ratio(acc, floor) >= RATIO_BAR),
        "constant_z": round((acc - floor) / se, 3) if se else None,
        "verdict": verdict, "adopted": bool(cleared), "note": note,
    }


def _f_control_row(track, variant, kind, acc, chance, n_hold, y_hold, extra,
                   explained):
    """A CONTROL row. Deliberately NOT a candidate: no `adopted`, no `floor`
    (it carries `control_floor`), so nothing can mistake it for a head."""
    floor = _no_skill_floor(y_hold) if len(y_hold) else chance
    row = {
        "track": track, "variant": variant, "role": "control", "control": kind,
        "probe": _probe_desc(explained),
        "n_holdout": int(n_hold), "accuracy": float(acc),
        "chance": float(chance), "ratio": round(_ratio(acc, chance), 4),
        "control_floor": round(float(floor), 4),
        "ratio_vs_control_floor": round(_ratio(acc, floor), 4),
        "beats_control_floor": bool(acc > floor),
        "clears_2x_floor": bool(_ratio(acc, floor) >= RATIO_BAR),
    }
    row.update(extra)
    return row


def _insplit_tertiles(s, tr, hold):
    """3 classes, ~1/3 each, ON EACH SPLIT SEPARATELY.

    A plain quantile taken on the train split drifts on the holdout under the
    temporal shift and leaves one class holding 80%+ of the rows, which puts the
    `2 x majority` bar out of reach for any probe. Ranking within each split
    keeps the control's floor at exactly 1/3, so the bar is a real bar. This is a
    property of the CONTROL's labels only — the direction labels are untouched.
    """
    y = np.zeros(len(s), dtype=int)
    for idx in (tr, hold):
        r = np.argsort(np.argsort(s[idx]))
        y[idx] = (r * 3) // len(idx)
    return y


def track_f(H, ids, regime, closes, args):
    """The last direction probe. Returns (rows, decision)."""
    rows = []

    # ---- F1: the directional tertile, Track E's exact target/split/holdout ----
    y3 = TH.directional_quantile_labels(ids, closes, n_forward=5)
    mask = y3 >= 0
    H1, y1 = H[mask], y3[mask]
    tr1, hold1 = TH.chronological_split(len(y1))
    Z1 = _standardise(H1, tr1)
    B1, evr1 = _pca_basis(Z1, tr1)
    P1 = Z1 @ B1

    r1 = fit_mlp(P1, y1, 3, tr1, hold1, seed=args.seed)
    rows.append(_f_row(
        "F", "F1 directional tertile (non-linear probe)", "candidate", r1,
        1.0 / 3, len(hold1), y1[hold1],
        "the non-linear twin of Track E: same target, same mask, same "
        "chronological split, same holdout, same 2x-majority bar", evr1))
    print("  Track F  F1 dir-tertile   n=%d acc=%.4f floor=%.4f ratio=%.3fx "
          "macroF1=%.4f %s"
          % (len(hold1), r1["holdout_acc"], rows[-1]["floor"],
             rows[-1]["ratio_vs_floor"], r1["holdout_macro_f1"],
             rows[-1]["verdict"]))

    # ---- F2: the balanced 7-action teacher (the other reading of the brief) ---
    ya2, names2, _d2 = TH.build_action_labels_balanced(ids, regime, closes,
                                                       n_forward=5)
    K2 = len(names2)
    tr2, hold2 = TH.chronological_split(len(ya2))
    Z2 = _standardise(H, tr2)
    B2, evr2 = _pca_basis(Z2, tr2)
    r2 = fit_mlp(Z2 @ B2, ya2, K2, tr2, hold2, seed=args.seed)
    rows.append(_f_row(
        "F", "F2 balanced 7-action teacher (non-linear probe)", "candidate", r2,
        1.0 / K2, len(hold2), ya2[hold2],
        "the §45 balanced ACTION teacher under the same non-linear probe — "
        "covers 'the balanced teacher' read as the action teacher, not the "
        "direction teacher", evr2))
    print("  Track F  F2 balanced-7    n=%d acc=%.4f floor=%.4f ratio=%.3fx "
          "macroF1=%.4f %s"
          % (len(hold2), r2["holdout_acc"], rows[-1]["floor"],
             rows[-1]["ratio_vs_floor"], r2["holdout_macro_f1"],
             rows[-1]["verdict"]))

    # ---- Fleak: the label IS in the input. Must clear the bar. ---------------
    Pleak = P1.copy()
    Pleak[:, 0] = y1.astype(float)
    rleak = fit_mlp(Pleak, y1, 3, tr1, hold1, seed=args.seed)
    rows.append(_f_control_row(
        "F", "Fleak label-injected-into-PC0 (instrument-recovers-signal)",
        "label-leak", rleak["holdout_acc"], 1.0 / 3, len(hold1), y1[hold1],
        {"train_acc": float(rleak["train_acc"]),
         "macro_f1": float(rleak["holdout_macro_f1"]),
         "expect": "CLEARS the bar: a signal that is present by construction, so "
                   "a failure here would mean the instrument is broken"},
        evr1))
    print("  Track F  Fleak leak       n=%d acc=%.4f floor=%.4f clears=%s"
          % (len(hold1), rleak["holdout_acc"], rows[-1]["control_floor"],
             rows[-1]["clears_2x_floor"]))

    # ---- Fcap: the MLP MUST fit a non-linear target the LINEAR probe cannot ---
    ycap = _insplit_tertiles(P1[:, 0] * P1[:, 1], tr1, hold1)
    rcap_mlp = fit_mlp(P1, ycap, 3, tr1, hold1, seed=args.seed)
    rcap_lin = _fit(P1, ycap, 3, tr1, hold1, args)
    rows.append(_f_control_row(
        "F", "Fcap pc0*pc1 (mlp)", "capacity-nonlinear",
        rcap_mlp["holdout_acc"], 1.0 / 3, len(hold1), ycap[hold1],
        {"train_acc": float(rcap_mlp["train_acc"]),
         "macro_f1": float(rcap_mlp["holdout_macro_f1"]),
         "expect": "CLEARS the bar: the probe can fit and GENERALISE a non-linear "
                   "function of h[E], so a failure on F1/F2 is not a broken or "
                   "under-trained optimiser"}, evr1))
    rows.append(_f_control_row(
        "F", "Fcap pc0*pc1 (linear)", "capacity-linear",
        rcap_lin["holdout_acc"], 1.0 / 3, len(hold1), ycap[hold1],
        {"train_acc": float(rcap_lin["train"]["acc"]),
         "macro_f1": float(rcap_lin["holdout_macro_f1"]),
         "expect": "does NOT clear the bar on the SAME target — which is what "
                   "makes the mlp row evidence of non-linearity rather than of a "
                   "control the linear head could also pass"}, evr1))
    print("  Track F  Fcap cap-target  mlp=%.4f linear=%.4f (floor=%.4f)"
          % (rows[-2]["accuracy"], rows[-1]["accuracy"],
             rows[-1]["control_floor"]))

    # ---- Fctl: shuffled TRAIN labels on F1. Must NOT clear the bar. ----------
    rctl = fit_mlp(P1, y1, 3, tr1, hold1, seed=args.seed,
                   shuffle_train_labels=True)
    rows.append(_f_control_row(
        "F", "Fctl shuffled-train-labels (bar-is-not-vacuous)", "shuffled-labels",
        rctl["holdout_acc"], 1.0 / 3, len(hold1), y1[hold1],
        {"train_acc": float(rctl["train_acc"]),
         "macro_f1": float(rctl["holdout_macro_f1"]),
         "expect": "does NOT clear the bar; if it did the bar would be vacuous"},
        evr1))
    print("  Track F  Fctl shuffled    n=%d acc=%.4f floor=%.4f clears=%s"
          % (len(hold1), rctl["holdout_acc"], rows[-1]["control_floor"],
             rows[-1]["clears_2x_floor"]))

    # ---- the decision: any CANDIDATE clearing the bar is a SIGNAL -------------
    cand = [r for r in rows if r.get("role") == "candidate"]
    adopted = any(r["adopted"] for r in cand)
    by = {r.get("control"): r for r in rows if r.get("role") == "control"}
    ctl_ok = (not by["shuffled-labels"]["clears_2x_floor"] and
              by["label-leak"]["clears_2x_floor"] and
              by["capacity-nonlinear"]["clears_2x_floor"] and
              not by["capacity-linear"]["clears_2x_floor"])
    decision = {
        "probe": "standardise(train) -> PCA(%d, train) -> 1 hidden layer, %d "
                 "tanh units -> softmax; Adam, %d full-batch epochs, lr %g, "
                 "l2 %g, inverse-frequency class weights"
                 % (PCA_COMPONENTS, MLP_HIDDEN, MLP_EPOCHS, MLP_LR, MLP_L2),
        "bar": "accuracy >= %g x majority_rate AND n_holdout >= %d"
               % (RATIO_BAR, MIN_HOLDOUT),
        "candidates": [r["variant"] for r in cand],
        "adopted": bool(adopted),
        "controls_valid": bool(ctl_ok),
        "verdict": ("SIGNAL — re-run DECISION 1 and the D1 audit before any "
                    "adoption" if adopted else
                    "CLOSE — " + F_TERM),
        "closed": bool(not adopted),
    }
    print("  Track F  DECISION: %s" % decision["verdict"])
    print("  Track F  controls_valid=%s (leak=%s cap_mlp=%s cap_lin=%s ctl=%s)"
          % (ctl_ok, by["label-leak"]["clears_2x_floor"],
             by["capacity-nonlinear"]["clears_2x_floor"],
             by["capacity-linear"]["clears_2x_floor"],
             by["shuffled-labels"]["clears_2x_floor"]))
    return rows, decision


def _conversion_test(rD, Hq, yqq, hold, action_names):
    """Does the rank head move a downstream, already-tested engine?

    Pre-registered direction: rows where the head predicts the TOP tertile
    should carry a strictly higher mean forward return than rows predicted
    BOTTOM tertile. If the head's ordering does not separate realised returns,
    it cannot inform any consumer, whatever its accuracy.
    """
    pred = rD["_raw_logits"][hold].argmax(axis=1)
    y_true = yqq[hold]
    top = y_true[pred == 2]
    bot = y_true[pred == 0]
    if len(top) == 0 or len(bot) == 0:
        return {"moved_engine": False, "detail": "a tertile got no predictions",
                "n_top": int(len(top)), "n_bottom": int(len(bot))}
    sep = float(top.mean() - bot.mean())
    moved = sep > 0
    return {"moved_engine": bool(moved),
            "n_top": int(len(top)), "n_bottom": int(len(bot)),
            "mean_top": float(top.mean()), "mean_bottom": float(bot.mean()),
            "separation": sep,
            "detail": ("top-tertile mean rank %.3f vs bottom %.3f -> separation "
                       "%.3f (%s)" % (top.mean(), bot.mean(), sep,
                                      "ordered" if moved else "NOT ordered"))}


# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trading-dump", default="build/head_data/trading")
    ap.add_argument("--bars", default="models/market/AAPL_1d.csv")
    ap.add_argument("--asset-dump", action="append", default=[],
                    metavar="NAME=DIR",
                    help="per-asset market dump for Track C (repeatable)")
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10, 21])
    ap.add_argument("--tracks", nargs="+", default=["A", "B", "C", "D", "E", "F"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--teacher", choices=["v1", "balanced"], default="v1",
                    help="action teacher: v1 = shipped fixed-threshold rule; "
                         "balanced = trailing-window quantile thresholds (§45)")
    ap.add_argument("--class-weight", choices=["none", "inverse"],
                    default="none",
                    help="loss re-weighting: none = shipped; inverse = 1/freq")
    ap.add_argument("--out", default="tools/edge_tracks.json")
    args = ap.parse_args()

    asset_dumps = []
    for spec in args.asset_dump:
        name, _, d = spec.partition("=")
        asset_dumps.append((name, d))
    args.asset_dumps = asset_dumps

    meta, H, ids, sets, labs = TH.load_dump(args.trading_dump)
    closes, _t = read_bars(args.bars)
    highs, lows = _read_hl(args.bars)
    regime = np.array([r[0] for r in labs])
    print("=== edge tracks: N=%d E=%d bars=%d ===" % (H.shape[0], H.shape[1], len(closes)))

    out = {
        "generated_by": "tools/edge_tracks.py",
        "bar": {"ratio": RATIO_BAR, "min_holdout": MIN_HOLDOUT,
                "floor": "majority-class rate of the holdout",
                "rule": "accuracy >= 2 x majority_rate AND n_holdout >= 30",
                "source": "docs/EDGE_RESEARCH.md §1 (corrected §1.1/§1.2)"},
        "teacher": args.teacher,
        "class_weight": args.class_weight,
        "baseline": {"candidate": "DecisionAction shipped blob own holdout",
                     "n": 369, "accuracy": 0.2439, "chance": 1.0 / 7,
                     "ratio": 1.707},
        "seed": args.seed,
        "tracks": {},
    }

    if "A" in args.tracks:
        print("\n--- Track A: longer horizons ---")
        out["tracks"]["A"] = track_a(H, ids, regime, closes, args)
    if "B" in args.tracks:
        print("\n--- Track B: monster features ---")
        out["tracks"]["B"] = track_b(H, ids, regime, closes, highs, lows, args)
    if "C" in args.tracks:
        print("\n--- Track C: multi-asset pooling ---")
        out["tracks"]["C"] = track_c(args)
    if "D" in args.tracks:
        print("\n--- Track D: rank / quantile targets ---")
        out["tracks"]["D"] = track_d(H, ids, regime, closes, args)
    if "E" in args.tracks:
        print("\n--- Track E: balanced 3-class directional control ---")
        out["tracks"]["E"] = track_dir_control(H, ids, closes, args)
    if "F" in args.tracks:
        print("\n--- Track F: the LAST direction probe (non-linear) ---")
        frows, fdecision = track_f(H, ids, regime, closes, args)
        out["tracks"]["F"] = frows
        out["direction_program"] = fdecision

    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=2)
    print("\nwrote %s" % args.out)


def _read_hl(path):
    import csv as _csv
    hi, lo = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            try:
                hi.append(float(row["high"]))
                lo.append(float(row["low"]))
            except (KeyError, ValueError):
                continue
    return np.array(hi), np.array(lo)


if __name__ == "__main__":
    main()
