"""
run_ffca_counterfactual.py
===========================
Run FFCA on the variant_A_patience100 counterfactual models (no pruning,
just trained longer per the FFCA agent's recommendation), then compare
INVESTIGATE / drift signals against the after-pruning reports to answer:

    Did following the agent's "train longer, do not prune" recipe
    actually treat the trust-instability the agent diagnosed?

Reuses MLP / load_keras_weights / convert helpers from run_ffca_pruned.py.
Difference: the feature matrix uses the FULL feature set from the
before-pruning trust dict (not the KEEP subset), because variant_A models
are trained on the same feature space as the original un-pruned models.
"""
from __future__ import annotations
import json
import os
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

# Path to the existing pruned runner (for reusing helpers)
COMPOUND_FLOODING = Path("/Users/hnaja002/Documents/projects/compound_flooding")
FFCA_AGENT_CASE   = Path("/Users/hnaja002/Documents/projects/FFCA_agent/case_studies")
FFCA_PACKAGE      = Path("/Users/hnaja002/Documents/projects/FFCA/FFCA_package")
CSV_PATH          = COMPOUND_FLOODING / "mlmiamicompoundfloodpredictions/Miami_GWL_WL_RAIN_GATE_2017_2024.csv"

sys.path.insert(0, str(COMPOUND_FLOODING))
sys.path.insert(0, str(FFCA_PACKAGE))

from ffca import FFCAReport, TabularAdapter, CheckpointLoader
from run_ffca_pruned import (
    MLP,
    convert_keras_to_pytorch_checkpoints,
    build_feature_matrix,
    parse_feature,
    N_ENSEMBLE,
    Y_BUFFER,
    TARGET_COL,
    TEST_YEARS,
)

# v0.8: source for feature enumeration moved from old (deleted) buggy reports
# to the corrected ensemble-mode dir. With build_feature_matrix now using the
# canonical PREFIX_ORDER, deterministic enumeration could also work; but the
# trust dict in the corrected reports is the authoritative full-feature list.
BEFORE_FFCA_DIR = COMPOUND_FLOODING / "FFCA_resutls_before_prunning_ensemble"

# The 3 degraded experiments: (name, lead, neurons, layers, category, before-report)
EXPERIMENTS = [
    dict(name="12hr_perfect_prog_gate_sigmoid", lead=12, neurons=200, layers=1,
         category="Predicted Gate Opening"),
    dict(name="24hr_perfect_prog_gate_sigmoid", lead=24, neurons=100, layers=1,
         category="Predicted Gate Opening"),
    dict(name="24hr_perfect_prog_all_inputs_sigmoid", lead=24, neurons=200, layers=1,
         category="Predictions All Inputs"),
]


def load_all_features(report_path: Path) -> dict[str, list[int]]:
    """Return every feature in the trust dict, regardless of decision."""
    with open(report_path) as f:
        data = json.load(f)
    selected = defaultdict(set)
    for feat in data["trust"].keys():
        prefix, lag = parse_feature(feat)
        selected[prefix].add(lag)
    return {p: sorted(lags) for p, lags in selected.items()}


def run_one(exp: dict, df_raw: pd.DataFrame, device: torch.device) -> None:
    name = exp["name"]
    print(f"\n{'='*70}\nFFCA (counterfactual variant_A): {name}\n{'='*70}")

    before_report = BEFORE_FFCA_DIR / exp["category"] / name / "report.json"
    if not before_report.exists():
        raise FileNotFoundError(before_report)

    selected = load_all_features(before_report)
    n_expected = sum(len(v) for v in selected.values())
    print(f"  Using full feature set: {n_expected} features")

    df, feat_cols, target = build_feature_matrix(df_raw, selected, TARGET_COL, exp["lead"])
    y_min = float(df_raw[TARGET_COL].min())
    y_max = float(df_raw[TARGET_COL].max() * (1 + Y_BUFFER))

    df_test = df[df.index.year.isin(TEST_YEARS)]
    if len(df_test) == 0:
        raise RuntimeError("No 2024 test data")
    X_test = df_test[feat_cols].to_numpy(dtype=np.float32)
    y_test = df_test[target].to_numpy(dtype=np.float32)
    print(f"  Test shape: {X_test.shape}")

    ffca_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
                             batch_size=32)

    variant_dir   = FFCA_AGENT_CASE / "counterfactual_runs" / name / "variant_A_patience100" / "MLP"
    keras_dir     = variant_dir / "models"
    pt_dir        = variant_dir / "ffca_report_ensemble" / "checkpoints"
    out_dir       = variant_dir / "ffca_report_ensemble"
    out_dir.mkdir(parents=True, exist_ok=True)

    if (out_dir / "report.json").exists():
        print(f"  [skip] report already at {out_dir}")
        return

    hp = dict(num_layers=exp["layers"], neurons=exp["neurons"], lr=1e-3, activation="relu")
    t = time.time()
    checkpoint_paths = convert_keras_to_pytorch_checkpoints(
        keras_dir, pt_dir, n_features=len(feat_cols),
        hp=hp, y_min=y_min, y_max=y_max, device=device,
    )
    print(f"  Converted {len(checkpoint_paths)} checkpoints ({time.time()-t:.1f}s)")

    adapter_model = MLP(n_features=len(feat_cols), neurons=exp["neurons"],
                       num_layers=exp["layers"], y_min=y_min, y_max=y_max).to(device).eval()
    adapter = TabularAdapter(adapter_model, feature_names=feat_cols)

    def model_factory():
        return MLP(n_features=len(feat_cols), neurons=exp["neurons"],
                   num_layers=exp["layers"], y_min=y_min, y_max=y_max).to(device).eval()

    ck_loader = CheckpointLoader(model_factory, checkpoint_paths, device=device)

    report = FFCAReport(
        adapter, ffca_loader,
        n_first_order_samples=32,
        n_hessian_samples=8,
        n_diag_probes=24,
        n_cauchy_probes=64,
        n_cauchy_samples=8,
        improvements=True,
        mode="ensemble",
    )
    t = time.time()
    report.run(checkpoints=ck_loader)
    print(f"  FFCA done in {time.time()-t:.1f}s")

    report.save(out_dir, save_plots=True)
    print(f"  Saved -> {out_dir}")
    for f in report.findings or []:
        print(f"    {f.severity:8s} {f.name:25s} {f.headline}")


def main():
    df_raw = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
    device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"Data:   {df_raw.shape}\n")
    for exp in EXPERIMENTS:
        run_one(exp, df_raw, device)


if __name__ == "__main__":
    main()
