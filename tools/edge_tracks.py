#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — tools/edge_tracks.py
#  Run the FOUR pre-registered edge tracks in docs/EDGE_RESEARCH.md.
#
#  The adoption bar (§1 of that document), applied identically to every track:
#      accuracy >= 2 x chance   AND   n_holdout >= 30
#  on a genuinely held-out split. A rejection is a RESULT and is recorded as one.
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

# --- the bar, verbatim from docs/EDGE_RESEARCH.md §1 --------------------------
RATIO_BAR = 2.0
MIN_HOLDOUT = 30


def _ratio(acc, chance):
    return acc / chance if chance else float("nan")


def _verdict(acc, chance, n_hold):
    """The ONLY adoption decision, in one place, so every track is judged alike."""
    if n_hold < MIN_HOLDOUT:
        return "REJECT (n<%d)" % MIN_HOLDOUT, False
    r = _ratio(acc, chance)
    if r >= RATIO_BAR:
        return "SIGNAL", True
    return "NO-SIGNAL", False


def read_bars(path):
    """(closes, times) from an OHLCV csv — reuses the trainer's own reader."""
    return TH.read_closes(path)


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
        ya, names, detail = TH.build_action_labels(
            ids, regime, closes, n_forward=hz)
        if names != action_names:
            raise SystemExit("action order drift: %r" % (names,))
        dist = {action_names[i]: int((ya == i).sum()) for i in range(K)}
        tr, hold = TH.chronological_split(len(ya))
        present = sorted(set(ya.tolist()))
        missing = [action_names[i] for i in range(K) if i not in present]
        r = TH.fit_one(H, ya, K, tr, hold, shuffle=False, seed=args.seed)
        acc = r["holdout_acc"]
        n_hold = len(hold)
        n_eff = effective_n(n_hold, hz)
        verdict, adopted = _verdict(acc, chance, n_hold)
        # The overlap-aware view: does it still clear the bar on independent
        # evidence? Reported, never used to LAUNDER a pass — only to downgrade.
        ratio_eff = _ratio(acc, chance)
        yh = ya[hold]
        _, cnt = np.unique(yh, return_counts=True)
        maj_rate = float(cnt.max() / cnt.sum())
        # ADOPTION NEEDS BOTH (§1.1): 2x1/K AND beating the majority class.
        if adopted and acc <= maj_rate:
            verdict = ("REJECT (passes 2x1/K but not the majority-class floor "
                       "%.4f — base-rate artifact)" % maj_rate)
            adopted = False
        row = {
            "track": "A", "variant": "n_forward=%d" % hz, "horizon": hz,
            "n_train": int(len(tr)), "n_holdout": int(n_hold),
            "n_eff": int(n_eff),
            "accuracy": float(acc), "chance": float(chance),
            "ratio": round(ratio_eff, 4),
            "majority_rate": round(maj_rate, 4),
            "ratio_vs_majority": round(_ratio(acc, maj_rate), 4),
            "beats_majority": bool(acc > maj_rate),
            "macro_f1": float(r["holdout_macro_f1"]),
            "temperature": float(r["temperature"]),
            "ece_calibrated": float(r["ece_calibrated"]),
            "action_counts": dist,
            "actions_missing": missing,
            "verdict": verdict, "adopted": bool(adopted),
            "n_eff_ratio": round(ratio_eff, 4),
            "n_eff_note": ("overlap-adjusted (n//horizon); a pass that does not "
                           "survive this is reported as fragile, not adopted"),
        }
        results.append(row)
        print("  Track A h=%-2d  n=%d (n_eff=%d)  acc=%.4f  chance=%.4f  "
              "ratio=%.3fx  %s"
              % (hz, n_hold, n_eff, acc, chance, ratio_eff, verdict))
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
    ya, _, _ = TH.build_action_labels(ids, regime, closes, n_forward=5)
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
    rH = TH.fit_one(H, ya, K, tr, hold, shuffle=False, seed=args.seed)
    rows.append(_b_row("B", "h[E] only (control)", rH, chance, len(hold),
                       y_hold=ya[hold]))

    # (b2) features alone — no hidden state at all.
    rF = TH.fit_one(Fz, ya, K, tr, hold, shuffle=False, seed=args.seed)
    rows.append(_b_row("B", "features only (no h[E])", rF, chance, len(hold),
                       y_hold=ya[hold]))

    # (b3) concatenated: scaled features appended to h[E].
    #     h[E] dims are 768 and roughly unit-scale after standardisation inside
    #     fit_one; features are ~unit-scale already. Concatenate RAW h with the
    #     standardised features so the h block keeps its own internal scaling.
    Hcat = np.concatenate([H, Fz], axis=1)
    rC = TH.fit_one(Hcat, ya, K, tr, hold, shuffle=False, seed=args.seed)
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
        print("  Track B %-26s n=%d acc=%.4f ratio=%.3fx %s"
              % (rr["variant"], rr["n_holdout"], rr["accuracy"], rr["ratio"],
                 rr["verdict"]))
    print("      features-only acc=%.4f  augmented acc=%.4f  "
          "backbone contributes: %s" % (feat_acc, aug_acc, beats_features))
    print("      controls: constant=%.4f  labels-shuffled=%.4f  "
          "features-permuted=%.4f" % (const_acc, ctl["shuffled_labels"]["accuracy"],
                                      ctl["permuted_features"]["accuracy"]))
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
    r_sh = TH.fit_one(Fz, ysh, K, tr, hold, shuffle=False, seed=args.seed)

    Fp = Fz.copy()
    for j in range(Fp.shape[1]):
        Fp[:, j] = Fz[rng.permutation(len(Fz)), j]
    r_pf = TH.fit_one(Fp, ya, K, tr, hold, shuffle=False, seed=args.seed)

    return {
        "constant": {"accuracy": const_acc, "majority_class": maj,
                     "note": "always emit the majority TRAIN class; no model"},
        "shuffled_labels": {"accuracy": float(r_sh["holdout_acc"]),
                            "note": "real features, labels permuted"},
        "permuted_features": {"accuracy": float(r_pf["holdout_acc"]),
                              "note": "real labels, feature columns permuted"},
    }


def _b_row(track, variant, r, chance, n_hold, y_hold=None):
    """One result row. `y_hold` (the holdout labels) adds the CORRECTED view.

    The `1/K` ratio is kept because it is what the document pre-registered. The
    majority-class ratio is added because §1.1 proved `1/K` is the wrong floor
    for a skewed set — a reader must be able to see both without re-running.
    """
    acc = float(r["holdout_acc"])
    verdict, adopted = _verdict(acc, chance, n_hold)
    row = {
        "track": track, "variant": variant,
        "n_holdout": int(n_hold), "accuracy": acc, "chance": float(chance),
        "ratio": round(_ratio(acc, chance), 4),
        "macro_f1": float(r["holdout_macro_f1"]),
        "temperature": float(r["temperature"]),
        "verdict": verdict, "adopted": bool(adopted),
    }
    if y_hold is not None and len(y_hold) > 0:
        vals, cnt = np.unique(np.asarray(y_hold), return_counts=True)
        maj_rate = float(cnt.max() / cnt.sum())
        row["majority_rate"] = round(maj_rate, 4)
        row["ratio_vs_majority"] = round(_ratio(acc, maj_rate), 4)
        # A result that clears 2x1/K but not the majority floor is a base-rate
        # artifact; say so in the row rather than only in the doc.
        row["beats_majority"] = bool(acc > maj_rate)
        # ADOPTION NEEDS BOTH (§1.1). The pre-registered 2x1/K bar is kept as
        # the first gate, but a row that only passes it while failing to beat a
        # constant predictor is NOT adopted — the §45 correction applied at the
        # point of decision, not just in prose.
        if row["adopted"] and not row["beats_majority"]:
            row["verdict"] = ("REJECT (passes 2x1/K but not the majority-class "
                              "floor %.4f — base-rate artifact)" % maj_rate)
            row["adopted"] = False
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
        ya, _, _ = TH.build_action_labels(ids, reg, closes, n_forward=5)
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

    rP = TH.fit_one(Hp, yp, K, tr, hold, shuffle=False, seed=args.seed)
    row = _b_row("C", "pooled (%s)" % "+".join(n for n, _ in dumps[:len(Hs)]),
                 rP, chance, len(hold), y_hold=yp[hold])
    out.append(row)
    print("  Track C pooled  n_train=%d n_holdout=%d acc=%.4f ratio=%.3fx %s"
          % (len(tr), len(hold), row["accuracy"], row["ratio"], row["verdict"]))

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
        v, _ = _verdict(acc, chance, len(m))
        _, cnt_m = np.unique(yp[m], return_counts=True)
        maj_m = float(cnt_m.max() / cnt_m.sum())
        out.append({"track": "C", "variant": "pooled head scored on %s holdout" % name,
                    "n_holdout": int(len(m)), "accuracy": acc,
                    "chance": float(chance),
                    "ratio": round(_ratio(acc, chance), 4),
                    "majority_rate": round(maj_m, 4),
                    "ratio_vs_majority": round(_ratio(acc, maj_m), 4),
                    "beats_majority": bool(acc > maj_m),
                    "verdict": v, "adopted": False})

        # (ii) single-asset head on the same rows.
        Hs_i = np.where(op == src)[0]
        tr_i, hold_i = TH.chronological_split(len(Hs_i))
        ri = TH.fit_one(Hp[Hs_i], yp[Hs_i], K, tr_i, hold_i,
                        shuffle=False, seed=args.seed)
        acc_i = float(ri["holdout_acc"])
        vi, _ = _verdict(acc_i, chance, len(hold_i))
        gains = acc > acc_i
        _, cnt_i = np.unique(yp[Hs_i][hold_i], return_counts=True)
        maj_i = float(cnt_i.max() / cnt_i.sum())
        out.append({"track": "C", "variant": "%s alone (single-asset head)" % name,
                    "n_holdout": int(len(hold_i)), "accuracy": acc_i,
                    "chance": float(chance),
                    "ratio": round(_ratio(acc_i, chance), 4),
                    "majority_rate": round(maj_i, 4),
                    "ratio_vs_majority": round(_ratio(acc_i, maj_i), 4),
                    "beats_majority": bool(acc_i > maj_i),
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
    rD = TH.fit_one(Hq, yqq, Q, tr, hold, shuffle=False, seed=args.seed)
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
    ap.add_argument("--tracks", nargs="+", default=["A", "B", "C", "D"])
    ap.add_argument("--seed", type=int, default=1234)
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
                "source": "docs/EDGE_RESEARCH.md §1"},
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
