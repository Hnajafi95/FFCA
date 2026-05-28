#!/usr/bin/env python3
"""§G — within-Impact-band ablation test for FFCA-Volatility.

Mirror of section_F_within_impact_band.py for the Volatility axis.
Same pair-matched-by-Impact-rank design: within an Impact band, walk
adjacent Impact-rank pairs, assign within-pair to high-V / low-V
subsets. Guarantees Impact matching (because each pair contributes
one feature to each subset) while contrasting Volatility.

Replaces the previously-reported §G result which used the same loose
Impact-range tolerance filter that broke §F2. Audit of the original
§G subsets revealed Impact ratios of 2.66x to 43x between the
nominally ``matched'' high-V and low-V groups — the prior
``high-V removal hurts more'' claim was Impact-confounded.

Usage (HPC, Proxima3):
  for k in 0 1; do
    nohup python3 section_G_within_impact_band.py --shard $k/2 --gpu-id $k \\
      --output-dir extra_validation_runs > section_G_band_shard_$k.log 2>&1 &
  done
  python3 section_G_within_impact_band.py --finalize-only \\
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
    """Pair-matched-by-Impact selection of high-V vs low-V subsets.

    Restrict to features at or above the `impact_band_lo_pct` Impact
    percentile, sort by Impact descending, walk adjacent pairs. Within
    each pair, the higher-V member joins high_v subset, the lower-V
    joins low_v. Guarantees Impact match (per pair, exactly one to each
    subset) and maximises within-pair V contrast.
    """
    arrays = cev.report_arrays(cev.load_corrected_report(exp))
    spec = cev.find_experiment_spec(exp)
    _, feat_cols, _, _, _ = cev.build_full_feature_matrix(spec)
    names = arrays['feature_names']
    impact = np.asarray(arrays['impact'])
    nonlin = np.asarray(arrays['nonlinearity'])
    inter  = np.asarray(arrays['interaction'])
    vol    = np.asarray(arrays['volatility'])

    in_exp = np.array([nm in feat_cols for nm in names])
    valid = np.where(in_exp)[0]
    if len(valid) == 0:
        raise RuntimeError(f'No valid features for experiment {exp}')

    impact_thresh = float(np.percentile(impact[valid], impact_band_lo_pct))
    band = sorted([i for i in valid if impact[i] >= impact_thresh],
                   key=lambda i: -impact[i])
    if len(band) < 2 * k:
        raise RuntimeError(
            f'Impact band has only {len(band)} features; need at least '
            f'{2*k} for K={k} pairs.'
        )

    high_v, low_v = [], []
    pair_log = []
    for p in range(k):
        a, b = band[2*p], band[2*p + 1]
        if vol[a] >= vol[b]:
            high_v.append(a); low_v.append(b)
        else:
            high_v.append(b); low_v.append(a)
        pair_log.append(dict(
            pair_rank=p,
            high_v_feature=str(names[high_v[-1]]),
            high_v_impact=float(impact[high_v[-1]]),
            high_v_V=float(vol[high_v[-1]]),
            low_v_feature=str(names[low_v[-1]]),
            low_v_impact=float(impact[low_v[-1]]),
            low_v_V=float(vol[low_v[-1]]),
        ))

    def summarize(idxs: list[int], label: str) -> dict:
        return dict(
            label=label,
            n=len(idxs),
            feature_names=[str(names[i]) for i in idxs],
            impact_mean=float(np.mean([impact[i] for i in idxs])),
            impact_min=float(np.min([impact[i] for i in idxs])),
            impact_max=float(np.max([impact[i] for i in idxs])),
            volatility_mean=float(np.mean([vol[i] for i in idxs])),
            volatility_min=float(np.min([vol[i] for i in idxs])),
            volatility_max=float(np.max([vol[i] for i in idxs])),
            nonlinearity_mean=float(np.mean([nonlin[i] for i in idxs])),
            interaction_mean=float(np.mean([inter[i] for i in idxs])),
        )

    return dict(
        impact_band_lo_pct=impact_band_lo_pct,
        impact_thresh=impact_thresh,
        n_in_band=len(band),
        k=k,
        selection_method='pair_match_by_impact_rank',
        high_v=summarize(high_v, 'high_v'),
        low_v=summarize(low_v, 'low_v'),
        pair_log=pair_log,
    )


def build_section_g_band_jobs(exp: str, info: dict) -> list[cev.TrainJob]:
    spec = cev.find_experiment_spec(exp)
    _, feat_cols, _, _, _ = cev.build_full_feature_matrix(spec)
    jobs = []
    for variant_key in ('high_v', 'low_v'):
        sub = info[variant_key]
        drop = sub['feature_names']
        jobs.append(cev.TrainJob(
            section='Gband',
            exp_name=exp,
            variant_name=f'drop_{variant_key}_at_matched_impact',
            drop_features=list(drop),
            extra=dict(
                subset_kind=variant_key,
                n_dropped=len(drop),
                impact_band_lo_pct=info['impact_band_lo_pct'],
                impact_thresh=info['impact_thresh'],
                subset_impact_mean=sub['impact_mean'],
                subset_volatility_mean=sub['volatility_mean'],
            ),
        ))
    return jobs


def rebuild_section_gband_csv(output_dir: Path) -> Path:
    runs_root = output_dir / 'runs' / 'section_Gband'
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
    out = output_dir / 'section_Gband.csv'
    if rows:
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f'  Rebuilt {out} from {len(rows)} per-job JSON files')
    return out


def analyze(output_dir: Path) -> None:
    csv = output_dir / 'section_Gband.csv'
    if not csv.exists():
        print(f'No {csv}; run shards first.')
        return
    df = pd.read_csv(csv)
    df = df[df.get('status', '') == 'trained'].copy()
    if len(df) < 2:
        return

    info_path = output_dir / 'runs' / 'section_Gband' / '_band_info.json'
    info = json.load(open(info_path)) if info_path.exists() else None

    # Baseline RMSE — pull from §A if present, else default
    baseline_rmse_cm = 3.3654
    sa_csv = output_dir / 'section_A.csv'
    if sa_csv.exists():
        try:
            sa = pd.read_csv(sa_csv)
            baseline_rmse_cm = float(
                sa[sa['exp_name'] == EXP]['orig_mean_rmse_cm'].iloc[0]
            )
        except Exception:
            pass

    print('\n== §G (within-Impact-band) — Volatility ablation ==')
    print(f'  Baseline RMSE: {baseline_rmse_cm:.4f} cm')
    delta = {}
    print(f'  {"variant":<35s}  {"RMSE_cm":>9s}  {"delta":>9s}')
    for _, row in df.iterrows():
        rmse_cm = float(row['test_rmse_cm'])
        d = rmse_cm - baseline_rmse_cm
        kind = row.get('extra_subset_kind', '?')
        delta[kind] = d
        print(f'  {row["variant_name"]:<35s}  {rmse_cm:>9.4f}  {d:>+9.4f}')

    if 'high_v' in delta and 'low_v' in delta:
        diff = delta['high_v'] - delta['low_v']
        print(f'\n  Δ(high-V) − Δ(low-V) = {diff:+.4f} cm')
        if diff > 0.05:
            verdict = ('Positive: high-V removal hurts more than low-V at '
                       'matched Impact → Volatility predicts marginal effect.')
        elif diff < -0.05:
            verdict = ('Negative: low-V removal hurts more than high-V at '
                       'matched Impact → Volatility may be inverted as a '
                       'predictor in this band.')
        else:
            verdict = ('Inconclusive: high-V and low-V removal at matched '
                       'Impact produce comparable damage → Volatility adds '
                       'little beyond Impact for this experiment.')
        print(f'\n  Verdict: {verdict}')

    summary = dict(
        baseline_rmse_cm=baseline_rmse_cm,
        info=info,
        deltas=delta,
    )
    with open(output_dir / 'section_Gband_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output-dir', type=Path, default=Path('extra_validation_runs'))
    p.add_argument('--exp', default=EXP)
    p.add_argument('--impact-band-lo-pct', type=float, default=50.0)
    p.add_argument('--k', type=int, default=10)
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
        rebuild_section_gband_csv(out)
        analyze(out)
        return

    cev._import_tf()

    info = pick_band_subsets(args.exp,
                              impact_band_lo_pct=args.impact_band_lo_pct,
                              k=args.k)
    info_path = out / 'runs' / 'section_Gband' / '_band_info.json'
    info_path.parent.mkdir(parents=True, exist_ok=True)
    is_first = (not args.shard) or args.shard.startswith('0/')
    if is_first and not info_path.exists():
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        print(f'Wrote {info_path}')

    print('\nBand subsets:')
    for k in ('high_v', 'low_v'):
        s = info[k]
        print(f'  {s["label"]:7s} (n={s["n"]}): '
              f'Impact mean={s["impact_mean"]:.4f}, '
              f'V mean={s["volatility_mean"]:.2e}')
        print(f'    features: {s["feature_names"]}')

    jobs = build_section_g_band_jobs(args.exp, info)
    shard_jobs = cev.shard_filter(jobs, args.shard)
    print(f'\nTotal jobs: {len(jobs)}; this shard: {len(shard_jobs)}')

    csv_path = cev.run_train_section('Gband', shard_jobs, out,
                                       smoke=args.smoke, shard=args.shard)
    print(f'\nDone. Per-shard CSV: {csv_path}')


if __name__ == '__main__':
    main()
