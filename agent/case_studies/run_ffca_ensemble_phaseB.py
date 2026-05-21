"""
run_ffca_ensemble_phaseB.py
============================
Phase B of the FFCA seed-vs-epoch reset.

Re-runs ensemble-mode FFCA on the 3 degraded experiments × 2 stages
(Original un-pruned, Aggressive-prune). Combined with the variant_A reports
already produced by run_ffca_counterfactual.py, this gives 9 corrected
reports for the 3 degraded experiments.

The 20+20 full sweep is Phase B-full (separate script).

Outputs:
  /Users/hnaja002/Documents/projects/compound_flooding/FFCA_resutls_before_prunning_ensemble/<cat>/<exp>/
  /Users/hnaja002/Documents/projects/compound_flooding/FFCA_results_After_prunning_ensemble/<cat>/<exp>/
"""
from __future__ import annotations
import json
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

CF      = Path("/Users/hnaja002/Documents/projects/compound_flooding")
PRUNED_MODELS = CF / "MLMiami FFCA Prunned Results"
ORIG_MODELS   = CF / "mlmiamicompoundfloodpredictions"
CSV_PATH      = ORIG_MODELS / "Miami_GWL_WL_RAIN_GATE_2017_2024.csv"
BEFORE_REPORT = CF / "FFCA_resutls_before_prunning"

# New (corrected) output dirs:
OUT_BEFORE = CF / "FFCA_resutls_before_prunning_ensemble"
OUT_AFTER  = CF / "FFCA_results_After_prunning_ensemble"

FFCA_PACKAGE = Path("/Users/hnaja002/Documents/projects/FFCA/FFCA_package")
sys.path.insert(0, str(FFCA_PACKAGE))
sys.path.insert(0, str(CF))

from ffca import FFCAReport, TabularAdapter, CheckpointLoader
from run_ffca_pruned import (
    MLP,
    convert_keras_to_pytorch_checkpoints,
    build_feature_matrix,
    parse_feature,
    load_ffca_selected_features,  # KEEP-only filter for the pruned stage
    N_ENSEMBLE,
    Y_BUFFER,
    TARGET_COL,
    TEST_YEARS,
)


# Experiments to run (3 degraded + their hyperparameters)
EXPERIMENTS = [
    dict(name="12hr_perfect_prog_gate_sigmoid", lead=12, neurons=200, layers=1,
         category="Predicted Gate Opening"),
    dict(name="24hr_perfect_prog_gate_sigmoid", lead=24, neurons=100, layers=1,
         category="Predicted Gate Opening"),
    dict(name="24hr_perfect_prog_all_inputs_sigmoid", lead=24, neurons=200, layers=1,
         category="Predictions All Inputs"),
]


def load_all_features(report_path: Path) -> dict[str, list[int]]:
    """Every feature in the before-pruning trust dict, regardless of decision."""
    with open(report_path) as f:
        data = json.load(f)
    selected = defaultdict(set)
    for feat in data["trust"].keys():
        prefix, lag = parse_feature(feat)
        selected[prefix].add(lag)
    return {p: sorted(lags) for p, lags in selected.items()}


def run_stage(exp: dict, stage: str, df_raw: pd.DataFrame, device: torch.device) -> None:
    name = exp["name"]
    cat  = exp["category"]
    before_report = BEFORE_REPORT / cat / name / "report.json"
    if not before_report.exists():
        raise FileNotFoundError(before_report)

    if stage == "original":
        keras_dir = ORIG_MODELS / cat / name / "models"  # original layout: no MLP subdir
        selected = load_all_features(before_report)
        out_dir = OUT_BEFORE / cat / name
        feature_label = "FULL"
    elif stage == "aggressive_prune":
        keras_dir = PRUNED_MODELS / cat / f"{name}_ffca" / "MLP" / "models"
        selected = load_ffca_selected_features(before_report)
        out_dir = OUT_AFTER / cat / name
        feature_label = "KEEP-only"
    else:
        raise ValueError(stage)

    if (out_dir / "report.json").exists():
        print(f"  [skip] {stage} report already at {out_dir}")
        return

    print(f"\n=== {stage.upper():18s} {name} ===")
    if not keras_dir.exists():
        print(f"  WARNING: models dir not found: {keras_dir}")
        return

    df, feat_cols, target = build_feature_matrix(df_raw, selected, TARGET_COL, exp["lead"])
    y_min = float(df_raw[TARGET_COL].min())
    y_max = float(df_raw[TARGET_COL].max() * (1 + Y_BUFFER))
    df_test = df[df.index.year.isin(TEST_YEARS)]
    X_test = df_test[feat_cols].to_numpy(dtype=np.float32)
    y_test = df_test[target].to_numpy(dtype=np.float32)
    print(f"  Features ({feature_label}): {len(feat_cols)} | test {X_test.shape}")

    ffca_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
                             batch_size=32)

    out_dir.mkdir(parents=True, exist_ok=True)
    pt_dir = out_dir / "checkpoints"

    hp = dict(num_layers=exp["layers"], neurons=exp["neurons"], lr=1e-3, activation="relu")
    t = time.time()
    checkpoint_paths = convert_keras_to_pytorch_checkpoints(
        keras_dir, pt_dir, n_features=len(feat_cols),
        hp=hp, y_min=y_min, y_max=y_max, device=device,
    )
    print(f"  Converted {len(checkpoint_paths)} ensemble members ({time.time()-t:.1f}s)")

    adapter_model = MLP(n_features=len(feat_cols), neurons=exp["neurons"],
                       num_layers=exp["layers"], y_min=y_min, y_max=y_max).to(device).eval()
    adapter = TabularAdapter(adapter_model, feature_names=feat_cols)

    def factory():
        return MLP(n_features=len(feat_cols), neurons=exp["neurons"],
                   num_layers=exp["layers"], y_min=y_min, y_max=y_max).to(device).eval()

    ck = CheckpointLoader(factory, checkpoint_paths, device=device)
    report = FFCAReport(
        adapter, ffca_loader,
        n_first_order_samples=32, n_hessian_samples=8, n_diag_probes=24,
        n_cauchy_probes=64, n_cauchy_samples=8,
        improvements=True, mode="ensemble",
    )
    t = time.time()
    report.run(checkpoints=ck)
    print(f"  FFCA done in {time.time()-t:.1f}s")

    report.save(out_dir, save_plots=True)
    print(f"  Saved -> {out_dir}")
    for f in report.findings:
        print(f"    {f.severity:8s} {f.name:25s} {f.headline}")


def main():
    df_raw = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
    device = torch.device("cpu")
    print(f"Device: {device}; data: {df_raw.shape}")
    for exp in EXPERIMENTS:
        run_stage(exp, "original",          df_raw, device)
        run_stage(exp, "aggressive_prune",  df_raw, device)


if __name__ == "__main__":
    main()
