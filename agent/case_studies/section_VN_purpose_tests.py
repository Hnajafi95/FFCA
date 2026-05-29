#!/usr/bin/env python3
"""§V + §N dimension-purpose tests — each dimension tested on its own terms.

The §F-band and §G-band tests asked whether N and V predict removal cost.
That is Impact's natural metric. The dimensions are designed to predict
DIFFERENT things:

  - Volatility: features whose importance is context-dependent (the rulebook
    prescribes "sliced (per-subgroup) analysis" for high-V features).
  - Nonlinearity: features whose response shape varies across samples in a
    way that PDP averaging hides (the rulebook prescribes "1D PDP/ICE plots"
    for high-N features; ICE plots specifically because per-sample curvatures
    can differ).

§V context-sensitivity test:
  For each candidate feature i, compute |∂f/∂x_i| at every test point on
  the trained 30-seed ensemble. Stratify the test set into K=4 |y|-quantile
  subgroups. Compute mean |∂f/∂x_i| within each subgroup. The
  std-across-subgroups, normalised by the feature's overall mean
  |∂f/∂x_i|, is the empirical "context coefficient of variation"
  (CV_subgroup). Correlate with FFCA-V.

  Hypothesis: FFCA-V predicts subgroup CV — high-V features ARE the
  context-dependent ones the rulebook prescribes sliced analysis for.

§N ICE-heterogeneity test:
  For each candidate feature i, sample 100 test points and 20 grid values
  per feature. Build a per-sample ICE curve (vary x_i only). Fit a quadratic
  to each ICE curve and extract the |quadratic coefficient| (per-sample
  curvature). Compute the std-across-samples of these per-sample
  curvatures. This is the empirical "ICE heterogeneity" — how much per-
  sample curvature varies sample to sample.

  Hypothesis: FFCA-N predicts ICE heterogeneity (NOT PDP curvature, which
  §F3 already showed is anti-correlated). High-N features have
  heterogeneous ICE — different samples show different curvatures, hence
  PDP averaging cancels them. This explains why the rulebook prescribes
  ICE plots (per-sample) rather than PDP (averaged) for high-N features.

Local on Mac CPU. ~20-30 min total.

Usage:
  python3 section_VN_purpose_tests.py [--output-dir extra_validation_runs]
      [--n-features 30] [--n-grid 20] [--n-samples-ice 100]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# Force CPU-only (Metal-plugin crash on legacy .h5)
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


# ─── Gradient computation ────────────────────────────────────────────────
def compute_gradients_per_feature(models, X: np.ndarray,
                                    batch: int = 2048) -> np.ndarray:
    """For each test row x and each feature i, return |∂f/∂x_i| averaged
    over the 30-seed ensemble. Shape: (n_rows, n_features).

    Uses TF GradientTape on each model in turn; one backward pass per
    (model, batch) computes the full gradient vector.
    """
    import tensorflow as tf
    n_rows, n_features = X.shape
    grad_accum = np.zeros((n_rows, n_features), dtype=np.float64)
    for k, model in enumerate(models):
        for start in range(0, n_rows, batch):
            end = min(n_rows, start + batch)
            xb = tf.convert_to_tensor(X[start:end], dtype=tf.float32)
            with tf.GradientTape() as tape:
                tape.watch(xb)
                yb = model(xb, training=False)
                # Predictions are (batch, 1); reduce so gradient is per-row.
                y_flat = tf.reduce_sum(yb)
            g = tape.gradient(y_flat, xb)
            grad_accum[start:end] += np.abs(g.numpy())
    return grad_accum / len(models)


# ─── ICE-curve computation ───────────────────────────────────────────────
def compute_ice_curvatures(models, X: np.ndarray, feat_idx: int,
                            sample_idx: np.ndarray,
                            n_grid: int = 20,
                            batch: int = 4096) -> np.ndarray:
    """For each test row in `sample_idx`, build an ICE curve over feature
    `feat_idx` (vary feature; hold others fixed at that row's values).
    Fit a quadratic to each curve; return the per-sample |quadratic coef|.

    Output shape: (len(sample_idx),)  --- per-sample local curvature.
    """
    n_s = len(sample_idx)
    col_vals = X[:, feat_idx]
    v_lo, v_hi = np.percentile(col_vals, [2, 98])
    grid = np.linspace(v_lo, v_hi, n_grid)            # (G,)

    # Build a big tile: (n_s, G, F) → flatten to (n_s * G, F) for batched
    # inference. The ICE for sample s at grid value v is row (s*G + g).
    X_sub = X[sample_idx]                              # (n_s, F)
    X_tile = np.repeat(X_sub[:, None, :], n_grid, axis=1)   # (n_s, G, F)
    X_tile[:, :, feat_idx] = grid[None, :]                   # broadcast grid
    X_flat = X_tile.reshape(n_s * n_grid, -1)               # (n_s*G, F)

    # Predict per model, average over the ensemble; reshape to (n_s, G).
    pred_accum = np.zeros(n_s * n_grid, dtype=np.float64)
    for model in models:
        for start in range(0, len(X_flat), batch):
            end = min(len(X_flat), start + batch)
            preds = model.predict(X_flat[start:end], verbose=0,
                                    batch_size=batch).flatten()
            pred_accum[start:end] += preds
    pred_avg = pred_accum / len(models)
    curves = pred_avg.reshape(n_s, n_grid)               # (n_s, G)

    # Quadratic fit per sample → take |quadratic coefficient|
    g_centered = grid - grid.mean()
    # design matrix
    A = np.stack([g_centered ** 2, g_centered, np.ones_like(g_centered)], axis=1)
    # Solve A @ p = curves.T for each sample; rows of (Q, 3)
    # vectorised via np.linalg.lstsq batched: do per-sample
    quadratic_coefs = np.empty(n_s, dtype=np.float64)
    for s in range(n_s):
        p, *_ = np.linalg.lstsq(A, curves[s], rcond=None)
        quadratic_coefs[s] = p[0]
    return np.abs(quadratic_coefs)


# ─── Feature selection ───────────────────────────────────────────────────
def pick_stratified_features(arrays: dict, feat_cols: list[str],
                              by: str, n: int = 30,
                              impact_min_pct: float = 50.0) -> list[int]:
    """Pick `n` global-feature-indices stratified by `by`-percentile within
    the top-(100−impact_min_pct)% Impact band. Returns indices into the
    FFCA-signature arrays."""
    names = arrays['feature_names']
    impact = np.asarray(arrays['impact'])
    sig = np.asarray(arrays[by])
    in_exp = np.array([nm in feat_cols for nm in names])
    valid = [i for i in range(len(names)) if in_exp[i]]
    imp_thresh = float(np.percentile([impact[i] for i in valid], impact_min_pct))
    eligible = [i for i in valid if impact[i] >= imp_thresh]
    if len(eligible) < n:
        eligible = sorted(valid, key=lambda i: -impact[i])[:max(n, len(eligible))]
    eligible_sorted = sorted(eligible, key=lambda i: sig[i])
    bins = np.array_split(np.array(eligible_sorted), n)
    picked = []
    seen = set()
    for b in bins:
        if len(b) == 0:
            continue
        mid = int(b[len(b) // 2])
        if mid not in seen:
            picked.append(mid)
            seen.add(mid)
    return picked


# ─── Main ────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output-dir', type=Path, default=Path('extra_validation_runs'))
    p.add_argument('--exp', default=EXP)
    p.add_argument('--n-features', type=int, default=30)
    p.add_argument('--n-subgroups', type=int, default=4,
                    help='|y|-quantile subgroups for §V test')
    p.add_argument('--n-grid', type=int, default=25,
                    help='Grid size per ICE curve for §N test')
    p.add_argument('--n-samples-ice', type=int, default=200,
                    help='Number of test samples to ICE-trace per feature')
    p.add_argument('--n-test-rows-grad', type=int, default=2000,
                    help='Subsample of test rows for gradient computation')
    p.add_argument('--rng-seed', type=int, default=42)
    args = p.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    print('Loading TF + report + ensemble ...')
    cev._import_tf()
    arrays = cev.report_arrays(cev.load_corrected_report(args.exp))
    spec = cev.find_experiment_spec(args.exp)
    df, feat_cols, target, _, _ = cev.build_full_feature_matrix(spec)
    _, _, _, _, X_test, y_test, _ = cev.split_train_val_test(df, target, feat_cols)
    models_dir = cev.original_models_dir(args.exp)
    models = cev.load_ensemble(models_dir, n=30)
    print(f'  exp={args.exp}, features={len(feat_cols)}, test_rows={len(X_test)}')
    print(f'  loaded {len(models)} models')

    rng = np.random.default_rng(args.rng_seed)
    if len(X_test) > args.n_test_rows_grad:
        sel = rng.choice(len(X_test), args.n_test_rows_grad, replace=False)
        X_grad = X_test[sel]
        y_grad = y_test[sel]
    else:
        X_grad = X_test
        y_grad = y_test
    print(f'\nGradient subsample: {len(X_grad)} rows')

    # ─── §V context-sensitivity test ────────────────────────────────────
    print('\n========== §V context-sensitivity test ==========')
    print('Computing ensemble-averaged |∂f/∂x_i| at every (sample, feature) ...')
    t0 = time.time()
    grads = compute_gradients_per_feature(models, X_grad)   # (n_rows, F)
    print(f'  shape={grads.shape}, took {time.time()-t0:.1f}s')

    # Stratify by |y| quantiles
    abs_y = np.abs(y_grad)
    quantile_bounds = np.percentile(abs_y, np.linspace(0, 100, args.n_subgroups + 1))
    sub = np.searchsorted(quantile_bounds[1:-1], abs_y)        # (n_rows,) in [0, K-1]
    print(f'  subgroup sizes: {[int((sub==g).sum()) for g in range(args.n_subgroups)]}')

    # Per-feature: subgroup-mean gradients, then std/mean across subgroups
    subgroup_means = np.zeros((grads.shape[1], args.n_subgroups))
    for g in range(args.n_subgroups):
        m = sub == g
        if m.sum() > 0:
            subgroup_means[:, g] = grads[m].mean(axis=0)
    overall_means = grads.mean(axis=0)
    between_std = subgroup_means.std(axis=1)
    cv_subgroup = between_std / (overall_means + 1e-12)

    # Correlate with FFCA-V over the candidate feature pool
    feat_pool_V = pick_stratified_features(arrays, feat_cols, 'volatility',
                                            n=args.n_features)
    fn = arrays['feature_names']
    name_to_col = {n: i for i, n in enumerate(feat_cols)}
    rows_V = []
    for fi in feat_pool_V:
        nm = fn[fi]
        if nm not in name_to_col:
            continue
        col = name_to_col[nm]
        rows_V.append(dict(
            feature=nm,
            ffca_impact=float(arrays['impact'][fi]),
            ffca_volatility=float(arrays['volatility'][fi]),
            ffca_nonlinearity=float(arrays['nonlinearity'][fi]),
            ffca_interaction=float(arrays['interaction'][fi]),
            cv_subgroup=float(cv_subgroup[col]),
            overall_mean_grad=float(overall_means[col]),
            between_std=float(between_std[col]),
        ))
    df_V = pd.DataFrame(rows_V)
    df_V.to_csv(out / 'section_V_context_sensitivity.csv', index=False)

    from scipy.stats import spearmanr, pearsonr
    if len(df_V) >= 5:
        rho_s, p_s = spearmanr(df_V['ffca_volatility'], df_V['cv_subgroup'])
        rho_p, p_p = pearsonr(df_V['ffca_volatility'], df_V['cv_subgroup'])
        print(f'\n  Spearman ρ(FFCA-V, CV_subgroup) = {rho_s:+.3f}   (p={p_s:.4g})')
        print(f'  Pearson  r(FFCA-V, CV_subgroup) = {rho_p:+.3f}   (p={p_p:.4g})')
        print('\n  Per-feature (sorted by FFCA-V, top + bottom 5):')
        s = df_V.sort_values('ffca_volatility', ascending=False)
        for label, sub_df in (('top-5', s.head(5)), ('bottom-5', s.tail(5))):
            print(f'  {label} by V:')
            for _, r in sub_df.iterrows():
                print(f'    {r["feature"]:14s}  V={r["ffca_volatility"]:.2e}  '
                      f'I={r["ffca_impact"]:.4f}  CV_sub={r["cv_subgroup"]:.4f}')
    summary_V = dict(
        n_features=int(len(df_V)),
        n_test_rows=int(len(X_grad)),
        n_subgroups=int(args.n_subgroups),
        spearman_rho_V_cv=float(rho_s) if len(df_V)>=5 else None,
        spearman_p_V_cv=float(p_s) if len(df_V)>=5 else None,
        pearson_r_V_cv=float(rho_p) if len(df_V)>=5 else None,
    )
    with open(out / 'section_V_context_sensitivity_summary.json', 'w') as f:
        json.dump(summary_V, f, indent=2)

    # ─── §N ICE-heterogeneity test ─────────────────────────────────────
    print('\n========== §N ICE-heterogeneity test ==========')
    feat_pool_N = pick_stratified_features(arrays, feat_cols, 'nonlinearity',
                                            n=args.n_features)
    print(f'  testing {len(feat_pool_N)} features')

    ice_sample_idx = rng.choice(len(X_test), args.n_samples_ice, replace=False)
    rows_N = []
    t0 = time.time()
    for k, fi in enumerate(feat_pool_N):
        nm = fn[fi]
        if nm not in name_to_col:
            continue
        col = name_to_col[nm]
        t_feat = time.time()
        per_sample_curv = compute_ice_curvatures(
            models, X_test, col, ice_sample_idx,
            n_grid=args.n_grid,
        )
        het = float(per_sample_curv.std())
        mean_curv = float(per_sample_curv.mean())
        rows_N.append(dict(
            feature=nm,
            ffca_impact=float(arrays['impact'][fi]),
            ffca_volatility=float(arrays['volatility'][fi]),
            ffca_nonlinearity=float(arrays['nonlinearity'][fi]),
            ffca_interaction=float(arrays['interaction'][fi]),
            ice_heterogeneity=het,
            ice_mean_curvature=mean_curv,
            elapsed_sec=round(time.time() - t_feat, 2),
        ))
        elapsed = time.time() - t_feat
        print(f'  [{k+1}/{len(feat_pool_N)}] {nm:14s}  '
              f'N={arrays["nonlinearity"][fi]:.4f}  '
              f'ICE_het={het:.5f}  ICE_mean_curv={mean_curv:.5f}  '
              f'({elapsed:.1f}s)')
    print(f'\n  Total ICE time: {(time.time()-t0)/60:.1f} min')

    df_N = pd.DataFrame(rows_N)
    df_N.to_csv(out / 'section_N_ice_heterogeneity.csv', index=False)

    if len(df_N) >= 5:
        rho_s_het, p_s_het = spearmanr(df_N['ffca_nonlinearity'],
                                          df_N['ice_heterogeneity'])
        rho_p_het, p_p_het = pearsonr(df_N['ffca_nonlinearity'],
                                         df_N['ice_heterogeneity'])
        rho_s_mc, p_s_mc = spearmanr(df_N['ffca_nonlinearity'],
                                        df_N['ice_mean_curvature'])
        print(f'\n  Spearman ρ(FFCA-N, ICE heterogeneity) = {rho_s_het:+.3f}   (p={p_s_het:.4g})')
        print(f'  Pearson  r(FFCA-N, ICE heterogeneity) = {rho_p_het:+.3f}   (p={p_p_het:.4g})')
        print(f'  Spearman ρ(FFCA-N, ICE mean curvature) = {rho_s_mc:+.3f}   (p={p_s_mc:.4g})')
        print('\n  Per-feature (sorted by FFCA-N, top + bottom 5):')
        s = df_N.sort_values('ffca_nonlinearity', ascending=False)
        for label, sub_df in (('top-5', s.head(5)), ('bottom-5', s.tail(5))):
            print(f'  {label} by N:')
            for _, r in sub_df.iterrows():
                print(f'    {r["feature"]:14s}  N={r["ffca_nonlinearity"]:.4f}  '
                      f'I={r["ffca_impact"]:.4f}  ICE_het={r["ice_heterogeneity"]:.5f}  '
                      f'ICE_mean_curv={r["ice_mean_curvature"]:.5f}')
    summary_N = dict(
        n_features=int(len(df_N)),
        n_samples_per_ice=int(args.n_samples_ice),
        n_grid_per_ice=int(args.n_grid),
        spearman_rho_N_ice_het=float(rho_s_het) if len(df_N)>=5 else None,
        spearman_p_N_ice_het=float(p_s_het) if len(df_N)>=5 else None,
        spearman_rho_N_ice_mean_curv=float(rho_s_mc) if len(df_N)>=5 else None,
        spearman_p_N_ice_mean_curv=float(p_s_mc) if len(df_N)>=5 else None,
    )
    with open(out / 'section_N_ice_heterogeneity_summary.json', 'w') as f:
        json.dump(summary_N, f, indent=2)


if __name__ == '__main__':
    main()
