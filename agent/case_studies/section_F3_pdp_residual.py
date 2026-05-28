#!/usr/bin/env python3
"""§F3 — Nonlinearity validation via PDP-residual analysis.

Replaces the flawed §F2 (which trained 1-feature MLPs and broke the
interaction context). This test uses the ORIGINAL 30-seed 223-feature
ensemble:

  1. For each candidate feature i, compute its Partial Dependence Plot:
     sample x_i over its observed range; for each sample, replicate the
     test set with column i replaced by the sample value, run the
     ensemble forward, average over rows. This gives PDP_i(x_i) — the
     model's marginal response to feature i with all interaction
     partners integrated out (because we average over all observed
     values of the other features).
  2. Fit an OLS line to PDP_i(x_i). The residual sum-of-squares (RSS)
     of that fit is the *curvature LR cannot capture* — directly what
     FFCA-Nonlinearity claims to measure.
  3. Correlate per-feature PDP-residual with FFCA-Nonlinearity. A
     positive Spearman ρ validates the dimension: high-N features in
     the FFCA report have more curvature in the actual PDP.

Run locally — needs the 30-seed ensemble at the canonical compound-
flooding path. No HPC needed.

Usage:
  python3 section_F3_pdp_residual.py \
      --output-dir extra_validation_runs \
      [--n-features 30] [--n-pdp-samples 40] [--n-test-rows 500]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# Force CPU-only — the legacy .h5 hypermodels crash the Apple-Silicon
# Metal plugin's graph remapper. CPU is plenty fast for this analysis.
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


# ─── Feature selection (stratified by FFCA-Nonlinearity, within top-50% Impact)
def pick_pdp_features(arrays: dict, feat_cols: list[str],
                       n: int = 30,
                       impact_min_pct: float = 50.0) -> list[dict]:
    """Pick `n` features stratified by FFCA-Nonlinearity percentile, restricted
    to features with FFCA-Impact in the top `100 - impact_min_pct`%.

    Returns a list of dicts: {name, impact, nonlinearity, interaction}.
    """
    names = arrays['feature_names']
    impact = np.asarray(arrays['impact'])
    nonlin = np.asarray(arrays['nonlinearity'])
    inter = np.asarray(arrays['interaction'])

    in_exp = np.array([nm in feat_cols for nm in names])
    valid = np.where(in_exp)[0]
    if len(valid) == 0:
        raise RuntimeError('No valid features for experiment.')

    impact_thresh = float(np.percentile(impact[valid], impact_min_pct))
    eligible = valid[impact[valid] >= impact_thresh]
    if len(eligible) < n:
        eligible = valid[np.argsort(-impact[valid])][:max(n, len(eligible))]

    order = np.argsort(nonlin[eligible])
    bins = np.array_split(eligible[order], n)
    picked = []
    for b in bins:
        if len(b) == 0:
            continue
        mid = int(b[len(b) // 2])
        picked.append(dict(
            name=str(names[mid]),
            impact=float(impact[mid]),
            nonlinearity=float(nonlin[mid]),
            interaction=float(inter[mid]),
            global_idx=int(mid),
        ))
    return picked


# ─── PDP computation ─────────────────────────────────────────────────────
def compute_pdp_residual(models, X_test: np.ndarray,
                          col_idx: int,
                          n_grid: int = 40,
                          row_subsample: int = 500,
                          rng: np.random.Generator | None = None,
                          ) -> dict:
    """Compute the partial dependence plot for column `col_idx` of X_test,
    fit OLS to it, and return the residual.

    Implementation: grid x_i over [p_2, p_98] of its observed values
    (avoids rare outliers). For each grid value, replicate a row-subsampled
    copy of X_test with column col_idx set to that value, run the 30-seed
    ensemble, take the ensemble-median prediction per row, then average
    over rows. That gives one PDP value per grid sample.
    """
    rng = rng or np.random.default_rng(42)
    n_rows = X_test.shape[0]
    if n_rows > row_subsample:
        row_sel = rng.choice(n_rows, row_subsample, replace=False)
    else:
        row_sel = np.arange(n_rows)
    X_sub = X_test[row_sel].copy()                  # (R, F)

    col_vals = X_test[:, col_idx]
    lo, hi = np.percentile(col_vals, [2, 98])
    grid = np.linspace(lo, hi, n_grid)              # (G,)

    pdp_vals = np.empty(n_grid, dtype=np.float64)
    for g, v in enumerate(grid):
        X_tile = X_sub.copy()
        X_tile[:, col_idx] = v
        # Ensemble predictions: (n_models, R)
        preds = cev.ensemble_predict(models, X_tile)
        # Per-row ensemble median, then mean across rows
        pdp_vals[g] = float(np.median(preds, axis=0).mean())

    # Fit OLS y = a*x + b on the PDP curve.
    a, b = np.polyfit(grid, pdp_vals, 1)
    linear_fit = a * grid + b
    resid = pdp_vals - linear_fit
    rss = float((resid ** 2).sum())
    # Normalise the residual by the PDP scale so curvature is comparable
    # across features that live at different output magnitudes.
    pdp_range = float(pdp_vals.max() - pdp_vals.min() + 1e-12)
    rss_norm = rss ** 0.5 / pdp_range

    return dict(
        grid_lo=float(lo), grid_hi=float(hi),
        pdp_min=float(pdp_vals.min()), pdp_max=float(pdp_vals.max()),
        pdp_range=pdp_range,
        ols_slope=float(a), ols_intercept=float(b),
        pdp_rss=rss,
        pdp_residual_norm=rss_norm,
        grid=grid.tolist(), pdp=pdp_vals.tolist(),
    )


# ─── Main ────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output-dir', type=Path, default=Path('extra_validation_runs'))
    p.add_argument('--exp', default=EXP)
    p.add_argument('--n-features', type=int, default=30)
    p.add_argument('--n-pdp-samples', type=int, default=40)
    p.add_argument('--n-test-rows', type=int, default=500)
    args = p.parse_args()

    print('Loading TF + report ...')
    cev._import_tf()
    arrays = cev.report_arrays(cev.load_corrected_report(args.exp))
    spec = cev.find_experiment_spec(args.exp)
    df, feat_cols, target, y_min, y_max_buf = cev.build_full_feature_matrix(spec)
    Xtr, ytr, Xv, yv, X_test, y_test, _ = cev.split_train_val_test(
        df, target, feat_cols)
    print(f'  experiment: {args.exp}')
    print(f'  features:   {len(feat_cols)}')
    print(f'  test rows:  {len(X_test)}')

    print('\nLoading 30-seed ensemble ...')
    models_dir = cev.original_models_dir(args.exp)
    models = cev.load_ensemble(models_dir, n=30)
    print(f'  loaded {len(models)} models from {models_dir}')

    print('\nPicking PDP-test features (stratified by FFCA-Nonlinearity, top-50% Impact)...')
    features_info = pick_pdp_features(
        arrays, feat_cols,
        n=args.n_features, impact_min_pct=50.0,
    )
    print(f'  {len(features_info)} features picked')

    # Map FFCA feature_names (cev convention) to feat_cols column indices.
    # FFCA's feature_names matches feat_cols 1:1 for these compound-flooding
    # experiments (we already enforce order_input_arrays).
    col_idx_of = {n: i for i, n in enumerate(feat_cols)}

    print('\nComputing PDP residuals ...')
    rows = []
    t0 = time.time()
    for k, fi in enumerate(features_info):
        nm = fi['name']
        if nm not in col_idx_of:
            print(f'  [{k+1}/{len(features_info)}] {nm:14s}  -- not in feat_cols (skip)')
            continue
        ci = col_idx_of[nm]
        t_feat = time.time()
        res = compute_pdp_residual(
            models, X_test, col_idx=ci,
            n_grid=args.n_pdp_samples,
            row_subsample=args.n_test_rows,
        )
        elapsed = time.time() - t_feat
        row = dict(
            feature=nm,
            ffca_impact=fi['impact'],
            ffca_nonlinearity=fi['nonlinearity'],
            ffca_interaction=fi['interaction'],
            pdp_residual_norm=res['pdp_residual_norm'],
            pdp_rss=res['pdp_rss'],
            pdp_range=res['pdp_range'],
            ols_slope=res['ols_slope'],
            elapsed_sec=round(elapsed, 2),
        )
        rows.append(row)
        print(f'  [{k+1}/{len(features_info)}] {nm:14s}  '
              f'N={fi["nonlinearity"]:.4f}  resid_norm={res["pdp_residual_norm"]:.4f}  '
              f'(took {elapsed:.1f}s)')

    total = time.time() - t0
    print(f'\nAll PDPs computed in {total/60:.1f} min')

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(rows)
    csv_path = out_dir / 'section_F3.csv'
    df_out.to_csv(csv_path, index=False)
    print(f'\nWrote {csv_path}')

    # Correlate FFCA-Nonlinearity with PDP residual
    from scipy.stats import spearmanr, pearsonr
    N = df_out['ffca_nonlinearity'].values
    R = df_out['pdp_residual_norm'].values
    rho_s, p_s = spearmanr(N, R)
    rho_p, p_p = pearsonr(N, R)
    print('\n== §F3 — PDP-residual vs FFCA-Nonlinearity ==')
    print(f'  n features:           {len(df_out)}')
    print(f'  Spearman ρ(N, resid): {rho_s:+.3f}   (p={p_s:.4f})')
    print(f'  Pearson  r(N, resid): {rho_p:+.3f}   (p={p_p:.4f})')
    print(f'  median resid_norm:    {np.median(R):.4f}')
    print(f'  range:                [{R.min():.4f}, {R.max():.4f}]')

    summary = dict(
        experiment=args.exp,
        n_features=int(len(df_out)),
        n_pdp_samples=int(args.n_pdp_samples),
        n_test_rows=int(args.n_test_rows),
        spearman_rho=float(rho_s), spearman_p=float(p_s),
        pearson_r=float(rho_p), pearson_p=float(p_p),
        median_resid_norm=float(np.median(R)),
        resid_norm_min=float(R.min()),
        resid_norm_max=float(R.max()),
        wall_time_sec=round(total, 1),
    )
    summary_path = out_dir / 'section_F3_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nWrote {summary_path}')


if __name__ == '__main__':
    main()
