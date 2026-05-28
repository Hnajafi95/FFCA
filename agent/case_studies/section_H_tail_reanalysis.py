#!/usr/bin/env python3
"""§H redesign #2 — tail-event super-additivity reanalysis.

Reuses the existing §H2 ensemble predictions (no retraining). For each
pair (5 Hessian + 5 random non-Catalyst controls), compute
super-additivity at three event regimes:

  - mean RMSE              (the metric §H2 already reported as null)
  - top-5% events RMSE     (the "operational floods")
  - top-1% events RMSE     (the extreme floods)

Section A showed that backbone removal hurts top-1% RMSE 3-13x more
than mean RMSE. Interaction effects might similarly amplify at the
tail. If they do, the Hessian-vs-Control super-additivity contrast
that was null at mean RMSE could become significant at top-5% / top-1%.

Baseline tail RMSE is computed locally by running the original
30-seed 24h-gate ensemble on the test set (same models §F3 used).

Usage:
  python3 section_H_tail_reanalysis.py \
      --output-dir extra_validation_runs
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# Force CPU-only (same Metal-plugin workaround as §F3)
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
TAIL_FRACTIONS = (0.05, 0.01)  # top-5%, top-1% by |y|


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def tail_rmse(y: np.ndarray, p: np.ndarray, q: float) -> float:
    """RMSE on the top `q` fraction of events (by |y|)."""
    n = len(y)
    k = max(5, int(np.ceil(q * n)))
    idx = np.argsort(np.abs(y))[-k:]
    return rmse(y[idx], p[idx])


def compute_baseline_predictions(exp: str) -> tuple[np.ndarray, np.ndarray]:
    """Run the original 30-seed ensemble on the test set, return
    (ensemble-median predictions in normalised units, y_test in normalised
    units). Matches what run_train_job saves into pred_median.npy and
    y_test.npy under each variant directory."""
    spec = cev.find_experiment_spec(exp)
    df, feat_cols, target, y_min, y_max_buf = cev.build_full_feature_matrix(spec)
    _, _, _, _, X_test, y_test, _ = cev.split_train_val_test(df, target, feat_cols)
    models_dir = cev.original_models_dir(exp)
    models = cev.load_ensemble(models_dir, n=30)
    preds = cev.ensemble_predict(models, X_test)
    med = np.median(preds, axis=0)
    return med, y_test


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output-dir', type=Path, default=Path('extra_validation_runs'))
    p.add_argument('--exp', default=EXP)
    args = p.parse_args()

    out = args.output_dir
    sect_dir = out / 'runs' / 'section_H2'
    if not sect_dir.exists():
        raise SystemExit(f'No {sect_dir} — pull the §H2 runs tree first.')

    cev._import_tf()

    pairs_info = json.load(open(sect_dir / '_pairs_info.json'))
    hessian_pairs = pairs_info['hessian_pairs']
    control_pairs = pairs_info['control_pairs']
    print(f'Hessian pairs: {len(hessian_pairs)} | Control pairs: {len(control_pairs)}')

    # ── Baseline predictions (run the original 30-seed ensemble) ─────────
    print('\nComputing baseline predictions from original 30-seed ensemble ...')
    t0 = time.time()
    med_baseline, y_test = compute_baseline_predictions(args.exp)
    print(f'  done in {time.time()-t0:.1f}s ({len(y_test)} test rows)')

    # Convert baseline med/y_test to cm units (multiply by 100 — matches
    # how run_train_job computes test_rmse_cm).
    base_pred_cm = med_baseline * 100.0
    y_cm = y_test * 100.0

    # ── Per-variant predictions from §H2 runs ───────────────────────────
    exp_root = sect_dir / args.exp
    def load_variant(kind: str, A: str, B: str, cond: str) -> np.ndarray:
        d = exp_root / f'{kind}_{A}_{B}__{cond}'
        med = np.load(d / 'pred_median.npy')      # normalised units
        return med * 100.0                          # cm units

    # Compute mean + tail RMSE per variant, then super-additivity per pair.
    metric_levels = [('mean', None)] + [(f'top{int(q*100)}', q) for q in TAIL_FRACTIONS]
    base_rmse = {}
    for name, q in metric_levels:
        if q is None:
            base_rmse[name] = rmse(y_cm, base_pred_cm)
        else:
            base_rmse[name] = tail_rmse(y_cm, base_pred_cm, q)
    print('\nBaseline RMSE (cm):')
    for name, val in base_rmse.items():
        print(f'  {name:>6s} = {val:.4f}')

    rows = []
    for kind, plist in (('hessian', hessian_pairs), ('control', control_pairs)):
        for pinfo in plist:
            A, B = pinfo['A'], pinfo['B']
            try:
                pA = load_variant(kind, A, B, 'drop_A_only')
                pB = load_variant(kind, A, B, 'drop_B_only')
                pAB = load_variant(kind, A, B, 'drop_AB')
            except FileNotFoundError as e:
                print(f'  MISSING: {kind}/{A}/{B}: {e}')
                continue
            row = dict(pair_kind=kind, A=A, B=B)
            for name, q in metric_levels:
                if q is None:
                    rA = rmse(y_cm, pA); rB = rmse(y_cm, pB); rAB = rmse(y_cm, pAB)
                else:
                    rA = tail_rmse(y_cm, pA, q)
                    rB = tail_rmse(y_cm, pB, q)
                    rAB = tail_rmse(y_cm, pAB, q)
                rB0 = base_rmse[name]
                dA = rA - rB0; dB = rB - rB0; dAB = rAB - rB0
                row[f'{name}_rmse_A_only'] = round(rA, 4)
                row[f'{name}_rmse_B_only'] = round(rB, 4)
                row[f'{name}_rmse_AB'] = round(rAB, 4)
                row[f'{name}_delta_A'] = round(dA, 4)
                row[f'{name}_delta_B'] = round(dB, 4)
                row[f'{name}_delta_AB'] = round(dAB, 4)
                row[f'{name}_super_add'] = round(dAB - (dA + dB), 4)
            rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = out / 'section_H_tail_super_add.csv'
    df.to_csv(out_csv, index=False)
    print(f'\nWrote {out_csv}')

    # ── Stat tests: Hessian vs Control super-additivity at each level ───
    from scipy.stats import mannwhitneyu, ttest_ind
    summary = dict(baseline_rmse=base_rmse, by_level={})
    print('\n== Hessian vs Control super-additivity by event regime ==')
    print(f'  {"level":>6s}  {"H_mean":>9s}  {"H_med":>9s}  {"C_mean":>9s}  {"C_med":>9s}  '
          f'{"MW_U":>6s}  {"MW_p":>6s}  {"t":>6s}  {"t_p":>6s}')
    for name, _ in metric_levels:
        col = f'{name}_super_add'
        h = df.loc[df['pair_kind']=='hessian', col].values.astype(float)
        c = df.loc[df['pair_kind']=='control', col].values.astype(float)
        if len(h) < 2 or len(c) < 2:
            continue
        u, pu = mannwhitneyu(h, c, alternative='greater')
        t, pt = ttest_ind(h, c, alternative='greater', equal_var=False)
        print(f'  {name:>6s}  {h.mean():+9.4f}  {np.median(h):+9.4f}  '
              f'{c.mean():+9.4f}  {np.median(c):+9.4f}  '
              f'{u:>6.1f}  {pu:>6.3f}  {t:>+6.2f}  {pt:>6.3f}')
        summary['by_level'][name] = dict(
            hessian=dict(n=int(len(h)), mean=float(h.mean()),
                         median=float(np.median(h)),
                         min=float(h.min()), max=float(h.max())),
            control=dict(n=int(len(c)), mean=float(c.mean()),
                         median=float(np.median(c)),
                         min=float(c.min()), max=float(c.max())),
            mw_u=float(u), mw_p=float(pu),
            t=float(t), t_p=float(pt),
        )

    # Per-pair tail super-additivity ranking
    print('\n== Per-pair super-additivity (sorted by top-1% super_add desc) ==')
    df_sorted = df.sort_values('top1_super_add', ascending=False)
    show_cols = ['pair_kind', 'A', 'B',
                 'mean_super_add', 'top5_super_add', 'top1_super_add']
    print(df_sorted[show_cols].to_string(index=False))

    summary_path = out / 'section_H_tail_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nWrote {summary_path}')


if __name__ == '__main__':
    main()
