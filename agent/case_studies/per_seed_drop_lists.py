"""Generate per-seed drop lists from each experiment's seed-1 model.

Tests the user's hypothesis: ensemble-mode FFCA aggregates across seeds and
may dilute load-bearing signal that individual seeds rely on. The per-seed
top-K Impact features should differ from the ensemble's CK set, and dropping
them when retraining the same seed should hit RMSE harder than ensemble
backbone removal does.

For each of 20 experiments:
  1. Load seed-1 .h5 from the original (paper) trained models
  2. Run single-snapshot FFCA on it (NO trust score, just per-feature signature)
  3. Identify top-K by Impact where K = ensemble CK count (matched count for
     fair comparison with the 30-seed variant_D experiment)
  4. Write per_seed_drop_lists.json keyed by experiment

Output structure mirrors drop_lists.json:
  { "<experiment>": {
        "n_features_full": int,
        "K_matched_to_ensemble_CK": int,
        "per_seed_topK": [feature_names...],
        "ensemble_CK_overlap": <count and pct>,
    } }
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

CF = Path("/Users/hnaja002/Documents/projects/compound_flooding")
FFCA_AGENT = Path("/Users/hnaja002/Documents/projects/FFCA_agent")
FFCA_PKG = Path("/Users/hnaja002/Documents/projects/FFCA/FFCA_package")
sys.path.insert(0, str(FFCA_PKG))
sys.path.insert(0, str(CF))

from ffca import FFCAReport, TabularAdapter
from run_ffca_pruned import (
    MLP, load_keras_weights_into_pytorch, build_feature_matrix,
    Y_BUFFER, TARGET_COL, TEST_YEARS, PREFIX_ORDER,
)

ENSEMBLE_DIR = CF / "FFCA_resutls_before_prunning_ensemble"
ORIG_MODELS = CF / "mlmiamicompoundfloodpredictions"
OUT_FILE = FFCA_AGENT / "case_studies" / "per_seed_drop_lists.json"

# 20 experiments — same set as feature_perturbation_retraining.py
EXPERIMENTS = [
    ("Measurements Only",         3,  "3hr_measured_sigmoid",                  100),
    ("Measurements Only",         6,  "6hr_measured_sigmoid",                  200),
    ("Measurements Only",         12, "12hr_measured_sigmoid",                 100),
    ("Measurements Only",         24, "24hr_measured_sigmoid",                 200),
    ("Predicted Ocean Water Levels", 3,  "3hr_perfect_prog_wls_sigmoid",       200),
    ("Predicted Ocean Water Levels", 6,  "6hr_perfect_prog_wls_sigmoid",       200),
    ("Predicted Ocean Water Levels", 12, "12hr_perfect_prog_wls_sigmoid",      100),
    ("Predicted Ocean Water Levels", 24, "24hr_perfect_prog_wls_sigmoid",      100),
    ("Predicted Rainfall",        3,  "3hr_perfect_prog_rain_sigmoid",         200),
    ("Predicted Rainfall",        6,  "6hr_perfect_prog_rain_sigmoid",         200),
    ("Predicted Rainfall",        12, "12hr_perfect_prog_rain_sigmoid",        100),
    ("Predicted Rainfall",        24, "24hr_perfect_prog_rain_sigmoid",        200),
    ("Predicted Gate Opening",    3,  "3hr_perfect_prog_gate_sigmoid",         200),
    ("Predicted Gate Opening",    6,  "6hr_perfect_prog_gate_sigmoid",         200),
    ("Predicted Gate Opening",    12, "12hr_perfect_prog_gate_sigmoid",        200),
    ("Predicted Gate Opening",    24, "24hr_perfect_prog_gate_sigmoid",        100),
    ("Predictions All Inputs",    3,  "3hr_perfect_prog_all_inputs_sigmoid",   200),
    ("Predictions All Inputs",    6,  "6hr_perfect_prog_all_inputs_sigmoid",   100),
    ("Predictions All Inputs",    12, "12hr_perfect_prog_all_inputs_sigmoid",  200),
    ("Predictions All Inputs",    24, "24hr_perfect_prog_all_inputs_sigmoid",  200),
]


def features_for(name: str, lead: int) -> dict[str, list[int]]:
    """Full feature selection for an experiment, mirroring features_from_experiment."""
    if "perfect_prog_all_inputs" in name:  pp = {"wl", "rain", "gate1", "gate2"}
    elif "perfect_prog_gate" in name:      pp = {"gate1", "gate2"}
    elif "perfect_prog_wls" in name:       pp = {"wl"}
    elif "perfect_prog_rain" in name:      pp = {"rain"}
    else:                                   pp = set()
    return {ch: list(range(-24, (lead if ch in pp else 0)+1)) for ch in PREFIX_ORDER}


def main():
    df_raw = pd.read_csv(ORIG_MODELS / "Miami_GWL_WL_RAIN_GATE_2017_2024.csv",
                         index_col=0, parse_dates=True)
    y_min = float(df_raw[TARGET_COL].min())
    y_max_buf = float(df_raw[TARGET_COL].max() * (1 + Y_BUFFER))
    device = torch.device("cpu")

    out: dict[str, dict] = {}
    print(f"{'Experiment':45s} {'n':>4s} {'K':>4s} {'overlap with ensemble CK':>30s}")
    for cat, lead, name, neurons in EXPERIMENTS:
        # Read ensemble CK count for fair K
        ens_report = ENSEMBLE_DIR / cat / name / "report.json"
        if not ens_report.exists():
            print(f"  [skip] missing ensemble report at {ens_report}")
            continue
        ens = json.loads(ens_report.read_text())
        ck_features = [f for f, info in (ens.get("trust") or {}).items()
                       if info.get("decision") == "CONFIDENTLY KEEP"]
        K = len(ck_features)

        # Build feature matrix in canonical order (build_feature_matrix is now fixed)
        sel = features_for(name, lead)
        df, feat_cols, target = build_feature_matrix(df_raw, sel, TARGET_COL, lead)
        df_test = df[df.index.year.isin(TEST_YEARS)]
        X_test = df_test[feat_cols].to_numpy(dtype=np.float32)
        y_test = df_test[target].to_numpy(dtype=np.float32)
        loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=32)

        # Load seed-1
        h5 = ORIG_MODELS / cat / name / "models" / "hypermodel1.h5"
        m = MLP(n_features=len(feat_cols), neurons=neurons, num_layers=1,
                 y_min=y_min, y_max=y_max_buf)
        load_keras_weights_into_pytorch(str(h5), m)
        m.to(device).eval()

        # Single-snapshot FFCA on seed-1
        adapter = TabularAdapter(m, feature_names=feat_cols)
        report = FFCAReport(
            adapter, loader,
            n_first_order_samples=32, n_hessian_samples=8, n_diag_probes=24,
            n_cauchy_probes=64, n_cauchy_samples=8,
            improvements=True, mode="trajectory",  # single ckpt; mode doesn't matter
        )
        report.run(checkpoints=None)
        sig = report.signatures[0]

        # Single-seed test RMSE (for the baseline comparison later)
        with torch.no_grad():
            preds = m(torch.tensor(X_test)).cpu().numpy().ravel()
        rmse_seed1_baseline = float(np.sqrt(np.mean((y_test - preds)**2))) * 100  # cm

        # Per-seed top-K by Impact (K matched to ensemble CK count)
        impact = np.asarray(sig.impact)
        topk_idx = np.argsort(impact)[-K:][::-1]
        topk_features = [feat_cols[i] for i in topk_idx]

        ck_set = set(ck_features)
        topk_set = set(topk_features)
        overlap = ck_set & topk_set
        overlap_pct = 100*len(overlap)/K if K else 0

        out[name] = dict(
            n_features=len(feat_cols),
            K_matched_to_ensemble_CK=K,
            per_seed_topK=topk_features,
            ensemble_CK=ck_features,
            overlap_count=len(overlap),
            overlap_pct=overlap_pct,
            seed_1_baseline_rmse_cm=rmse_seed1_baseline,
        )
        print(f"{name[:45]:45s} {len(feat_cols):>4d} {K:>4d} {len(overlap):>4d}/{K:<4d} ({overlap_pct:>3.0f}%)  baseline RMSE: {rmse_seed1_baseline:.2f} cm")

    OUT_FILE.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_FILE}")


if __name__ == "__main__":
    main()
