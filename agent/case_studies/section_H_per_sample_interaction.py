#!/usr/bin/env python3
"""§H redesign — per-sample masking-based interaction test (no retraining).

Tests the trained 30-seed 24h-gate ensemble *as-is*. For each candidate
pair (A, B), computes the per-sample super-additivity via mean-masking:

    SA(x) = f(x) - f(x | A→ā) - f(x | B→b̄) + f(x | A→ā, B→b̄)

For a model decomposed as f = g_A + g_B + h(A,B), this exactly returns
h(A,B)(x) - h(ā,x_B) - h(x_A,b̄) + h(ā,b̄) — i.e., the residual after
removing all additive contributions; pure interaction.

Aggregated per pair:
    I(A,B) = E_x[|SA(x)|]

Compares the distribution of I(A,B) between:
  - Top-K pairs by FFCA pairwise-Hessian magnitude (computed on the
    same 30-seed ensemble)
  - K random control pairs from features outside the top Catalysts

Also reports the Spearman correlation between FFCA-Hessian magnitude
and I(A,B) across the union of pairs. If FFCA's Hessian computation
captures real interaction in the model, the correlation should be
positive: large pairwise Hessian → large masking-derived I.

Usage:
  python3 section_H_per_sample_interaction.py \
      --output-dir extra_validation_runs \
      [--n-targets 5] [--n-hessian-pairs 50] [--n-control-pairs 50]
      [--n-test-rows 1500]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# Force CPU-only (Metal-plugin workaround on the legacy .h5 hypermodels).
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import tensorflow as _tf
try:
    _tf.config.set_visible_devices([], 'GPU')
except Exception:
    pass

import numpy as np
import pandas as pd

import compound_flooding_extra_validation as cev


EXP = '24hr_perfect_prog_gate_sigmoid'


def predict_ensemble_median(models, X: np.ndarray, batch: int = 4096) -> np.ndarray:
    """Return ensemble-median predictions on X, shape (n_rows,)."""
    preds = cev.ensemble_predict(models, X)   # (n_models, n_rows)
    return np.median(preds, axis=0)


def hessian_rows_for_targets(models, X: np.ndarray, target_indices: list[int],
                              batch: int = 256) -> np.ndarray:
    """For each target feature index t in `target_indices`, compute
    H_t[j] = mean over rows of |d²(ensemble-median f)/dx_t dx_j|.

    We approximate the ensemble-mean Hessian as the mean across models
    of each model's Hessian row, matching what the original
    section_H_hessian_pair_synergy.py used."""
    # Import the helpers from the original Section H script.
    import section_H_hessian_pair_synergy as sec_h
    H = sec_h.ensemble_hessian_rows(models, X, target_indices)   # (T, F)
    return H


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output-dir', type=Path, default=Path('extra_validation_runs'))
    p.add_argument('--exp', default=EXP)
    p.add_argument('--n-targets', type=int, default=5,
                     help='Top-N Catalysts to use as Hessian-row targets')
    p.add_argument('--n-hessian-pairs', type=int, default=50,
                     help='Top-K pairs by |H_AB| within the target rows')
    p.add_argument('--n-control-pairs', type=int, default=50,
                     help='Random pairs from non-Catalyst features')
    p.add_argument('--n-test-rows', type=int, default=1500,
                     help='Subsample of test rows for the masking scan')
    p.add_argument('--rng-seed', type=int, default=42)
    args = p.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    print('Loading TF, report, models ...')
    cev._import_tf()
    spec = cev.find_experiment_spec(args.exp)
    df, feat_cols, target, y_min, y_max_buf = cev.build_full_feature_matrix(spec)
    _, _, _, _, X_test, y_test, _ = cev.split_train_val_test(df, target, feat_cols)
    print(f'  exp: {args.exp}, features: {len(feat_cols)}, test rows: {len(X_test)}')

    models_dir = cev.original_models_dir(args.exp)
    models = cev.load_ensemble(models_dir, n=30)
    print(f'  loaded {len(models)} models')

    arrays = cev.report_arrays(cev.load_corrected_report(args.exp))
    fn = arrays['feature_names']
    impact = np.asarray(arrays['impact'])
    # arrays['archetypes'] from cev.report_arrays is already remapped to
    # paper-name strings (see cev._archetype_to_paper) — match by name.
    archetypes = np.asarray(arrays['archetypes'])
    catalyst_global = np.where(archetypes == 'Interactive Catalyst')[0]
    print(f'  {len(catalyst_global)} Interactive Catalyst features in the model')

    # Pick top-N targets among Catalysts by Impact (matches the H2 script)
    top_catalysts = sorted(catalyst_global, key=lambda i: -impact[i])[:args.n_targets]
    # Map to feat_cols indices
    feat_name_to_col = {n: i for i, n in enumerate(feat_cols)}
    target_in_feat = [feat_name_to_col[fn[i]] for i in top_catalysts if fn[i] in feat_name_to_col]
    print(f'  top-{args.n_targets} Catalysts as Hessian-row targets:')
    for ti, gi in zip(target_in_feat, top_catalysts):
        print(f'    {fn[gi]:14s}  Impact={impact[gi]:.4f}  feat_col_idx={ti}')

    # Sample test rows for the Hessian computation
    rng = np.random.default_rng(args.rng_seed)
    if len(X_test) > args.n_test_rows:
        row_sel = rng.choice(len(X_test), args.n_test_rows, replace=False)
        X_sub = X_test[row_sel]
    else:
        X_sub = X_test
    print(f'\nUsing {len(X_sub)} test rows for the masking scan.')

    # --- Compute ensemble Hessian rows ---------------------------------
    print('\nComputing FFCA Hessian rows on the ensemble ...')
    t0 = time.time()
    H_rows = hessian_rows_for_targets(models, X_sub, target_in_feat)   # (T, F)
    print(f'  shape={H_rows.shape}, took {time.time()-t0:.1f}s')

    # --- Pick pair pool: top-K by |H_AB| within target rows ------------
    # H_rows[t, j] gives the magnitude for the pair (target_in_feat[t], j).
    # Build a flat list of pair candidates.
    target_set = set(target_in_feat)
    pair_candidates = []
    for ti_local, ti_col in enumerate(target_in_feat):
        for j in range(len(feat_cols)):
            if j == ti_col:
                continue
            # Use lexical ordering of column indices to avoid (A,B) vs (B,A) duplicates
            a, b = min(ti_col, j), max(ti_col, j)
            pair_candidates.append((a, b, float(H_rows[ti_local, j])))
    # Dedupe by (a, b), taking the max |H| over rows
    dedupe: dict[tuple[int,int], float] = {}
    for a, b, h in pair_candidates:
        if (a, b) not in dedupe or h > dedupe[(a, b)]:
            dedupe[(a, b)] = h
    pair_candidates = [(a, b, h) for (a, b), h in dedupe.items()]
    pair_candidates.sort(key=lambda x: -x[2])
    print(f'\n  {len(pair_candidates)} unique candidate pairs from {len(target_in_feat)} target rows')
    print(f'  H range: [{pair_candidates[-1][2]:.4e}, {pair_candidates[0][2]:.4e}]')

    top_hessian_pairs = pair_candidates[:args.n_hessian_pairs]
    print(f'\n  Top-{args.n_hessian_pairs} Hessian pairs (showing top 5):')
    for a, b, h in top_hessian_pairs[:5]:
        print(f'    {feat_cols[a]:14s} ↔ {feat_cols[b]:14s}  |H|={h:.4e}')

    # Random control pairs: from features NOT in the Hessian top pool
    used_feats = set()
    for a, b, _ in top_hessian_pairs:
        used_feats.add(a); used_feats.add(b)
    available = [i for i in range(len(feat_cols)) if i not in used_feats]
    rng.shuffle(available)
    control_pairs = []
    i = 0
    while i + 1 < len(available) and len(control_pairs) < args.n_control_pairs:
        a, b = available[i], available[i + 1]
        # Use Hessian magnitude lookup if both indices are in target rows
        h_val = None
        if a in target_set:
            t_idx = target_in_feat.index(a)
            h_val = float(H_rows[t_idx, b])
        elif b in target_set:
            t_idx = target_in_feat.index(b)
            h_val = float(H_rows[t_idx, a])
        # else: not in target rows, H unknown — leave None
        control_pairs.append((a, b, h_val))
        i += 2
    print(f'\n  {len(control_pairs)} random non-Catalyst control pairs')

    # --- Compute per-sample masking interaction -------------------------
    # f(x), f(x | A→ā), f(x | B→b̄), f(x | both→means)
    # We can reuse single-mask computations across pairs that share a feature.
    feature_means = X_test.mean(axis=0)   # full-data means for masking baseline

    # All unique features in our pair pool
    pool_features = set()
    for a, b, _ in top_hessian_pairs + control_pairs:
        pool_features.add(a); pool_features.add(b)
    print(f'\nUnique features in pair pool: {len(pool_features)}')

    print('\nComputing baseline f(x) on subsample ...')
    t0 = time.time()
    f0 = predict_ensemble_median(models, X_sub)   # (n_sub,)
    print(f'  baseline f computed in {time.time()-t0:.1f}s')

    # Single-feature masking: f(x | feat→mean)
    print(f'\nComputing single-mask predictions for {len(pool_features)} features ...')
    f_single = {}
    t0 = time.time()
    for k, feat_idx in enumerate(sorted(pool_features)):
        X_mod = X_sub.copy()
        X_mod[:, feat_idx] = feature_means[feat_idx]
        f_single[feat_idx] = predict_ensemble_median(models, X_mod)
        if (k + 1) % 10 == 0:
            print(f'    {k+1}/{len(pool_features)} done ({time.time()-t0:.1f}s elapsed)')
    print(f'  all singles in {time.time()-t0:.1f}s')

    # Per-pair: f(x | A→ā, B→b̄), then SA(x)
    print(f'\nComputing pair-masked predictions for {len(top_hessian_pairs) + len(control_pairs)} pairs ...')
    rows = []
    t0 = time.time()
    all_pairs = ([(a,b,h,'hessian') for a,b,h in top_hessian_pairs]
                 + [(a,b,h,'control') for a,b,h in control_pairs])
    for k, (a, b, h_val, kind) in enumerate(all_pairs):
        X_mod = X_sub.copy()
        X_mod[:, a] = feature_means[a]
        X_mod[:, b] = feature_means[b]
        f_both = predict_ensemble_median(models, X_mod)
        sa = f0 - f_single[a] - f_single[b] + f_both
        I = float(np.mean(np.abs(sa)))
        I_max = float(np.max(np.abs(sa)))
        rows.append(dict(
            pair_kind=kind,
            A=feat_cols[a], B=feat_cols[b],
            A_idx=int(a), B_idx=int(b),
            ffca_hessian=h_val,
            I_mean=I,
            I_max=I_max,
        ))
        if (k + 1) % 20 == 0:
            print(f'    {k+1}/{len(all_pairs)} pairs ({time.time()-t0:.1f}s elapsed)')
    print(f'  all pairs in {time.time()-t0:.1f}s')

    df_out = pd.DataFrame(rows)
    csv_path = out / 'section_H_per_sample_interaction.csv'
    df_out.to_csv(csv_path, index=False)
    print(f'\nWrote {csv_path}')

    # --- Stats: distribution comparison + correlation ------------------
    from scipy.stats import mannwhitneyu, spearmanr, pearsonr
    h_I = df_out.loc[df_out['pair_kind']=='hessian', 'I_mean'].values.astype(float)
    c_I = df_out.loc[df_out['pair_kind']=='control', 'I_mean'].values.astype(float)
    print('\n== Per-sample masking interaction: Hessian vs Control ==')
    print(f'  Hessian pairs (n={len(h_I)}):  '
          f'mean I={h_I.mean():.4f}  median={np.median(h_I):.4f}  range=[{h_I.min():.4f}, {h_I.max():.4f}]')
    print(f'  Control pairs (n={len(c_I)}):  '
          f'mean I={c_I.mean():.4f}  median={np.median(c_I):.4f}  range=[{c_I.min():.4f}, {c_I.max():.4f}]')
    if len(h_I) >= 3 and len(c_I) >= 3:
        u, pu = mannwhitneyu(h_I, c_I, alternative='greater')
        print(f'  Mann-Whitney U (H>C):  U={u:.1f}, p={pu:.4g}')
        try:
            from scipy.stats import ttest_ind
            t, pt = ttest_ind(h_I, c_I, alternative='greater', equal_var=False)
            print(f'  Welch t-test  (H>C):   t={t:+.2f}, p={pt:.4g}')
        except Exception:
            pass

    # Correlation of FFCA-Hessian with masking I (only pairs where we have H)
    with_h = df_out.dropna(subset=['ffca_hessian'])
    if len(with_h) >= 5:
        rho_s, p_s = spearmanr(with_h['ffca_hessian'].astype(float),
                                with_h['I_mean'].astype(float))
        rho_p, p_p = pearsonr(with_h['ffca_hessian'].astype(float),
                                with_h['I_mean'].astype(float))
        print(f'\n  Spearman ρ(FFCA-Hess, I_mean) on {len(with_h)} pairs: {rho_s:+.3f}   (p={p_s:.4g})')
        print(f'  Pearson  r(FFCA-Hess, I_mean) on {len(with_h)} pairs: {rho_p:+.3f}   (p={p_p:.4g})')

    summary = dict(
        experiment=args.exp,
        n_targets=int(args.n_targets),
        n_hessian_pairs=int(len(h_I)),
        n_control_pairs=int(len(c_I)),
        n_test_rows=int(len(X_sub)),
        hessian=dict(n=int(len(h_I)), mean=float(h_I.mean()), median=float(np.median(h_I)),
                     min=float(h_I.min()), max=float(h_I.max())),
        control=dict(n=int(len(c_I)), mean=float(c_I.mean()), median=float(np.median(c_I)),
                     min=float(c_I.min()), max=float(c_I.max())),
        mw_u=float(u) if len(h_I)>=3 and len(c_I)>=3 else None,
        mw_p=float(pu) if len(h_I)>=3 and len(c_I)>=3 else None,
        spearman_FH_I=float(rho_s) if len(with_h)>=5 else None,
        spearman_p=float(p_s) if len(with_h)>=5 else None,
    )
    with open(out / 'section_H_per_sample_interaction_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
