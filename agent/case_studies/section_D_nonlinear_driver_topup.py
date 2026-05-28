#!/usr/bin/env python3
"""§D top-up — run the missing Non-linear Driver archetype-ablation row.

The original §D run silently skipped the Non-linear Driver archetype on
all 3 experiments because `ARCHETYPES_TO_TEST` listed it as
"Nonlinear Driver" (no hyphen) while the FFCA package returns
"Non-linear Driver" (with hyphen) via PACKAGE_INDEX_TO_PAPER. The
lookup missed, no jobs were generated, no error was raised. Found in
the audit on 2026-05-28.

This script does a focused HPC top-up that only runs the missing
archetype, so we don't have to re-run the rest of §D. Produces job
output under runs/section_D/<exp>/drop_archetype_non-linear_driver/
(matching the existing §D layout). The full §D CSV can be rebuilt
afterward via `compound_flooding_extra_validation.py --finalize-only`.

Usage on Proxima3 (3 experiments — fit on 3 GPUs):
  for k in 0 1 2; do
    nohup python3 section_D_nonlinear_driver_topup.py --shard $k/3 --gpu-id $k \\
      --output-dir extra_validation_runs > section_D_nl_shard_$k.log 2>&1 &
  done
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import compound_flooding_extra_validation as cev


D_EXPERIMENTS = [
    '12hr_perfect_prog_rain_sigmoid',
    '24hr_perfect_prog_all_inputs_sigmoid',
    '24hr_perfect_prog_gate_sigmoid',
]


def build_jobs(experiments: list[str]) -> list[cev.TrainJob]:
    jobs = []
    for exp in experiments:
        rep = cev.load_corrected_report(exp)
        arrays = cev.report_arrays(rep)
        spec = cev.find_experiment_spec(exp)
        _, feat_cols, _, _, _ = cev.build_full_feature_matrix(spec)
        groups = cev.archetype_groups(arrays)
        nl_members = [n for n in groups.get('Non-linear Driver', []) if n in feat_cols]
        if not nl_members:
            print(f'  {exp}: 0 Non-linear Driver features — skipping')
            continue
        print(f'  {exp}: {len(nl_members)} Non-linear Driver features '
              f'to drop')
        jobs.append(cev.TrainJob(
            section='D', exp_name=exp,
            variant_name='drop_archetype_non-linear_driver',
            drop_features=nl_members,
            extra=dict(archetype='Non-linear Driver',
                        n_in_archetype=len(nl_members)),
        ))
    return jobs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output-dir', type=Path, default=Path('extra_validation_runs'))
    p.add_argument('--shard', default='')
    p.add_argument('--gpu-id', type=int, default=None)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()

    if args.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    cev._import_tf()

    print('Building Non-linear Driver §D jobs:')
    jobs = build_jobs(D_EXPERIMENTS)
    if not jobs:
        print('No jobs to run — exiting.')
        return
    shard_jobs = cev.shard_filter(jobs, args.shard)
    print(f'\nTotal jobs: {len(jobs)}; this shard: {len(shard_jobs)}')

    # Write into the existing runs/section_D/ tree so finalize logic
    # can rebuild section_D.csv with the new row.
    csv_path = cev.run_train_section('D', shard_jobs, out,
                                       smoke=args.smoke, shard=args.shard)
    print(f'\nDone. Per-shard CSV: {csv_path}')
    print('To rebuild the canonical section_D.csv afterward:')
    print(f'  python3 compound_flooding_extra_validation.py --finalize-only '
          f'--sections D --output-dir {out}')


if __name__ == '__main__':
    main()
