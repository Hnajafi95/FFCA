#!/usr/bin/env python3
"""§F — within-Impact-band ablation test for FFCA-Nonlinearity.

The cleanest single-number test of whether N adds predictive value
beyond Impact. Design:

  1. Restrict to features whose FFCA-Impact lies in a target band
     (e.g., the top Impact quartile of the experiment's feature pool).
  2. Within that band, sort by FFCA-Nonlinearity.
  3. Take the top-K (high-N at matched Impact) and bottom-K (low-N at
     matched Impact) subsets.
  4. For each subset, drop those features and retrain a 30-seed
     ensemble.
  5. Compare RMSE damage: if high-N removal hurts more than low-N at
     matched Impact, FFCA-N predicts marginal effect beyond Impact.

This is the within-band re-design of the broken §F2 single-feature
test. The subset-level removal preserves the interaction context that
the original per-feature test lost.

Runs on Proxima3 HPC. Uses the existing cev sharding infrastructure.

Usage:
  for k in 0 1; do
    nohup python3 section_F_within_impact_band.py --shard $k/2 --gpu-id $k \\
      --output-dir extra_validation_runs > section_F_band_shard_$k.log 2>&1 &
  done
  # 2 retrains, ~1 GPU-hr each, run on 2 shards (one variant per shard)

  # After both shards finish:
  python3 section_F_within_impact_band.py --finalize-only \\
      --output-dir extra_validation_runs
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import compound_flooding_extra_validation as cev


EXP = '24hr_perfect_prog_gate_sigmoid'


def pick_band_subsets(exp: str,
                       impact_band_lo_pct: float = 50.0,
                       k: int = 10) -> dict:
    """Pick two K-sized subsets with closely-matched Impact distributions
    but contrasting Nonlinearity, using PAIR-MATCHING by Impact rank.

    Algorithm: restrict to features in the Impact band, sort by Impact
    descending, walk adjacent pairs (ranks 1+2, 3+4, ...). Within each
    pair, the higher-N member goes to high_n subset; lower-N to low_n.
    This guarantees that the high_n and low_n subsets have very close
    Impact distributions (because every Impact-adjacent pair contributes
    exactly one to each subset), while maximising the within-pair N
    contrast.

    Default impact_band_lo_pct=50 means we use the top half of Impact-
    ranked features — wider band, more pairs to form, better statistical
    power. With 2K features, we form K pairs; need 2K features in the band.
    """
    arrays = cev.report_arrays(cev.load_corrected_report(exp))
    spec = cev.find_experiment_spec(exp)
    _, feat_cols, _, _, _ = cev.build_full_feature_matrix(spec)
    names = arrays['feature_names']
    impact = np.asarray(arrays['impact'])
    nonlin = np.asarray(arrays['nonlinearity'])
    inter  = np.asarray(arrays['interaction'])

    in_exp = np.array([nm in feat_cols for nm in names])
    valid = np.where(in_exp)[0]
    if len(valid) == 0:
        raise RuntimeError(f'No valid features for experiment {exp}')

    impact_thresh = float(np.percentile(impact[valid], impact_band_lo_pct))
    band = sorted([i for i in valid if impact[i] >= impact_thresh],
                   key=lambda i: -impact[i])  # sort by Impact descending
    if len(band) < 2 * k:
        raise RuntimeError(
            f'Impact band has only {len(band)} features; need at least '
            f'{2*k} for K={k} pairs.'
        )

    # Pair-match: walk adjacent ranks (0,1), (2,3), ..., take K pairs
    high_n, low_n = [], []
    pair_log = []
    for p in range(k):
        a, b = band[2*p], band[2*p + 1]
        if nonlin[a] >= nonlin[b]:
            high_n.append(a); low_n.append(b)
        else:
            high_n.append(b); low_n.append(a)
        pair_log.append(dict(
            pair_rank=p,
            high_n_feature=str(names[high_n[-1]]),
            high_n_impact=float(impact[high_n[-1]]),
            high_n_N=float(nonlin[high_n[-1]]),
            low_n_feature=str(names[low_n[-1]]),
            low_n_impact=float(impact[low_n[-1]]),
            low_n_N=float(nonlin[low_n[-1]]),
        ))

    def summarize(idxs: list[int], label: str) -> dict:
        return dict(
            label=label,
            n=len(idxs),
            feature_names=[str(names[i]) for i in idxs],
            impact_mean=float(np.mean([impact[i] for i in idxs])),
            impact_min=float(np.min([impact[i] for i in idxs])),
            impact_max=float(np.max([impact[i] for i in idxs])),
            nonlin_mean=float(np.mean([nonlin[i] for i in idxs])),
            nonlin_min=float(np.min([nonlin[i] for i in idxs])),
            nonlin_max=float(np.max([nonlin[i] for i in idxs])),
            interaction_mean=float(np.mean([inter[i] for i in idxs])),
        )

    return dict(
        impact_band_lo_pct=impact_band_lo_pct,
        impact_thresh=impact_thresh,
        n_in_band=len(band),
        k=k,
        selection_method='pair_match_by_impact_rank',
        high_n=summarize(high_n, 'high_n'),
        low_n=summarize(low_n, 'low_n'),
        pair_log=pair_log,
    )


def build_section_f_band_jobs(exp: str, info: dict) -> list[cev.TrainJob]:
    spec = cev.find_experiment_spec(exp)
    _, feat_cols, _, _, _ = cev.build_full_feature_matrix(spec)
    jobs = []
    for variant_key in ('high_n', 'low_n'):
        sub = info[variant_key]
        drop = sub['feature_names']
        kept = [c for c in feat_cols if c not in set(drop)]
        jobs.append(cev.TrainJob(
            section='Fband',
            exp_name=exp,
            variant_name=f'drop_{variant_key}_at_matched_impact',
            drop_features=list(drop),
            extra=dict(
                subset_kind=variant_key,
                n_dropped=len(drop),
                impact_band_lo_pct=info['impact_band_lo_pct'],
                impact_thresh=info['impact_thresh'],
                subset_impact_mean=sub['impact_mean'],
                subset_nonlin_mean=sub['nonlin_mean'],
                kept_features=kept,
            ),
        ))
    return jobs


def rebuild_section_fband_csv(output_dir: Path) -> Path:
    """Rebuild section_Fband.csv from per-job results.json."""
    runs_root = output_dir / 'runs' / 'section_Fband'
    rows = []
    if runs_root.exists():
        for rp in sorted(runs_root.rglob('results.json')):
            d = json.load(open(rp))
            rr = {k: v for k, v in d.items() if k != 'extra'}
            for k, v in (d.get('extra') or {}).items():
                if isinstance(v, list):
                    rr[f'extra_{k}'] = ';'.join(map(str, v))
                else:
                    rr[f'extra_{k}'] = v
            rows.append(rr)
    out = output_dir / 'section_Fband.csv'
    if rows:
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f'  Rebuilt {out} from {len(rows)} per-job JSON files')
    return out


def analyze(output_dir: Path) -> None:
    csv = output_dir / 'section_Fband.csv'
    if not csv.exists():
        print(f'No {csv}; run shards first.')
        return
    df = pd.read_csv(csv)
    df = df[df.get('status', '') == 'trained'].copy()
    if len(df) < 2:
        print(f'§F-band: only {len(df)} rows; need both variants. Skipping analysis.')
        return

    info_path = output_dir / 'runs' / 'section_Fband' / '_band_info.json'
    info = json.load(open(info_path)) if info_path.exists() else None

    # Baseline from the original ensemble's section_A entry, if available
    baseline_rmse_cm = None
    sa_csv = output_dir / 'section_A.csv'
    if sa_csv.exists():
        sa = pd.read_csv(sa_csv)
        try:
            baseline_rmse_cm = float(
                sa[sa['exp_name'] == EXP]['orig_mean_rmse_cm'].iloc[0]
            )
        except Exception:
            pass
    # fallback: 3.365 from the §A / §H baseline we already have
    if baseline_rmse_cm is None:
        baseline_rmse_cm = 3.3654

    print('\n== §F (within-Impact-band) — Nonlinearity ablation ==')
    if info:
        print(f"  Impact band: Impact >= P{info['impact_band_lo_pct']:.0f} "
              f"(threshold {info['impact_thresh']:.4f})")
        print(f"  K per subset: {info['k']}, total features in band: {info['n_in_band']}")
        for sub_key in ('high_n', 'low_n'):
            s = info[sub_key]
            print(f"  {s['label']} subset: n={s['n']}, "
                  f"Impact mean={s['impact_mean']:.4f} (range {s['impact_min']:.4f}-{s['impact_max']:.4f}), "
                  f"N mean={s['nonlin_mean']:.4f} (range {s['nonlin_min']:.4f}-{s['nonlin_max']:.4f})")

    print(f'\n  Baseline RMSE: {baseline_rmse_cm:.4f} cm')
    print(f'\n  {"variant":<35s}  {"RMSE_cm":>9s}  {"delta":>9s}')
    delta = {}
    for _, row in df.iterrows():
        vname = row['variant_name']
        rmse_cm = float(row['test_rmse_cm'])
        d = rmse_cm - baseline_rmse_cm
        kind = row.get('extra_subset_kind', '?')
        delta[kind] = d
        print(f'  {vname:<35s}  {rmse_cm:>9.4f}  {d:>+9.4f}')

    if 'high_n' in delta and 'low_n' in delta:
        diff = delta['high_n'] - delta['low_n']
        print(f'\n  Δ(high-N) − Δ(low-N) = {diff:+.4f} cm')
        if diff > 0.05:
            verdict = ('Positive: high-N removal hurts more than low-N at '
                       'matched Impact → Nonlinearity predicts marginal effect.')
        elif diff < -0.05:
            verdict = ('Negative: low-N removal hurts more than high-N at '
                       'matched Impact → Nonlinearity does not predict marginal '
                       'effect in this band (and may even be inverted).')
        else:
            verdict = ('Inconclusive: high-N and low-N removal at matched Impact '
                       'produce comparable damage → Nonlinearity adds little '
                       'beyond Impact for this experiment.')
        print(f'\n  Verdict: {verdict}')

    summary = dict(
        baseline_rmse_cm=baseline_rmse_cm,
        info=info,
        deltas=delta,
    )
    with open(output_dir / 'section_Fband_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output-dir', type=Path, default=Path('extra_validation_runs'))
    p.add_argument('--exp', default=EXP)
    p.add_argument('--impact-band-lo-pct', type=float, default=75.0,
                     help='Restrict to features at or above this Impact percentile '
                          '(default 75 = top quartile)')
    p.add_argument('--k', type=int, default=10,
                     help='Size of high-N and low-N subsets to drop')
    p.add_argument('--shard', default='')
    p.add_argument('--gpu-id', type=int, default=None)
    p.add_argument('--finalize-only', action='store_true')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()

    if args.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    if args.finalize_only:
        rebuild_section_fband_csv(out)
        analyze(out)
        return

    cev._import_tf()

    info = pick_band_subsets(args.exp,
                              impact_band_lo_pct=args.impact_band_lo_pct,
                              k=args.k)
    info_path = out / 'runs' / 'section_Fband' / '_band_info.json'
    info_path.parent.mkdir(parents=True, exist_ok=True)
    is_first = (not args.shard) or args.shard.startswith('0/')
    if is_first and not info_path.exists():
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        print(f'Wrote {info_path}')

    print('\nBand subsets:')
    for k in ('high_n', 'low_n'):
        s = info[k]
        print(f'  {s["label"]:7s} (n={s["n"]}): '
              f'Impact mean={s["impact_mean"]:.4f}, N mean={s["nonlin_mean"]:.4f}')
        print(f'    features: {s["feature_names"]}')

    jobs = build_section_f_band_jobs(args.exp, info)
    shard_jobs = cev.shard_filter(jobs, args.shard)
    print(f'\nTotal jobs: {len(jobs)}; this shard: {len(shard_jobs)}')

    # Each training job runs run_train_job, writes
    # runs/section_Fband/<exp>/<variant>/results.json
    csv_path = cev.run_train_section('Fband', shard_jobs, out,
                                       smoke=args.smoke, shard=args.shard)
    print(f'\nDone. Per-shard CSV: {csv_path}')


if __name__ == '__main__':
    main()
