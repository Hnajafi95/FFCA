"""Per-seed FFCA analysis on the 4 gate experiments — seed 1 only.

Test of the user's hypothesis: ensemble-mode FFCA aggregates across seeds
and identifies the *intersection* of important features, which the model
class can route around. Per-seed FFCA on seed 1 should identify *that
seed's* important features, which may differ from the ensemble intersection.

This script:
  1. For each of 4 gate experiments, loads hypermodel1.h5
  2. Computes its single-snapshot FFCA signature (no checkpoints, no trust)
  3. Reads K = number of CONFIDENTLY KEEP features from the corrected ensemble
     report (for apples-to-apples comparison)
  4. Picks top-K features by Impact in seed 1's signature
  5. Reports overlap with the ensemble's CONFIDENTLY KEEP set
  6. Writes per_seed_drop_lists.json with the per-seed top-K sets
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from collections import defaultdict

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
from run_ffca_pruned import MLP, load_keras_weights_into_pytorch, build_feature_matrix, parse_feature, N_ENSEMBLE, Y_BUFFER, TARGET_COL, TEST_YEARS

GATE_EXPS = [
    dict(name="3hr_perfect_prog_gate_sigmoid",  lead=3,  neurons=200),
    dict(name="6hr_perfect_prog_gate_sigmoid",  lead=6,  neurons=200),
    dict(name="12hr_perfect_prog_gate_sigmoid", lead=12, neurons=200),
    dict(name="24hr_perfect_prog_gate_sigmoid", lead=24, neurons=100),
]

CSV_PATH = CF / "mlmiamicompoundfloodpredictions/Miami_GWL_WL_RAIN_GATE_2017_2024.csv"
ENSEMBLE_REPORTS = CF / "FFCA_resutls_before_prunning_ensemble"
ORIG_MODELS = CF / "mlmiamicompoundfloodpredictions"


def build_full_features(name: str, lead: int) -> dict:
    """Hardcoded enumeration matching feature_perturbation_retraining.py"""
    BASE = ["gwl", "wl", "rain", "stgH", "stgT", "gate1", "gate2"]
    if "perfect_prog_gate" in name:
        pp = {"gate1", "gate2"}
    elif "perfect_prog_all_inputs" in name:
        pp = {"wl", "rain", "gate1", "gate2"}
    elif "perfect_prog_wls" in name:
        pp = {"wl"}
    elif "perfect_prog_rain" in name:
        pp = {"rain"}
    else:
        pp = set()
    out: dict[str, list[int]] = {}
    for ch in BASE:
        hi = lead if ch in pp else 0
        out[ch] = list(range(-24, hi + 1))
    return out


def main():
    df_raw = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
    y_min = float(df_raw[TARGET_COL].min())
    y_max_buf = float(df_raw[TARGET_COL].max() * (1 + Y_BUFFER))
    device = torch.device("cpu")
    results = {}

    for exp in GATE_EXPS:
        name = exp["name"]; lead = exp["lead"]; neurons = exp["neurons"]
        print(f"\n=== {name} ===")

        # Read ensemble CONFIDENTLY KEEP set for fair K
        ens_report = ENSEMBLE_REPORTS / "Predicted Gate Opening" / name / "report.json"
        ens = json.loads(ens_report.read_text())
        ck_features = [f for f, info in (ens.get("trust") or {}).items()
                       if info.get("decision") == "CONFIDENTLY KEEP"]
        K = len(ck_features)
        feature_names_all = ens["feature_names"]
        n_features = ens["n_features"]
        print(f"  Ensemble CONFIDENTLY KEEP: {K} features (of {n_features})")

        # Build feature matrix on test year
        selected = build_full_features(name, lead)
        df, feat_cols, target = build_feature_matrix(df_raw, selected, TARGET_COL, lead)
        df_test = df[df.index.year.isin(TEST_YEARS)]
        X_test = df_test[feat_cols].to_numpy(dtype=np.float32)
        y_test = df_test[target].to_numpy(dtype=np.float32)
        loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=32)
        assert feat_cols == feature_names_all, f"feature ordering mismatch on {name}"

        # Load seed-1 model
        h5_path = ORIG_MODELS / "Predicted Gate Opening" / name / "models" / "hypermodel1.h5"
        m = MLP(n_features=len(feat_cols), neurons=neurons, num_layers=1, y_min=y_min, y_max=y_max_buf)
        load_keras_weights_into_pytorch(str(h5_path), m)
        m.to(device).eval()

        # Single-snapshot FFCA (no checkpoints, no trust)
        adapter = TabularAdapter(m, feature_names=feat_cols)
        report = FFCAReport(
            adapter, loader,
            n_first_order_samples=32, n_hessian_samples=8, n_diag_probes=24,
            n_cauchy_probes=64, n_cauchy_samples=8,
            improvements=True, mode="trajectory",  # single checkpoint, mode doesn't matter
        )
        report.run(checkpoints=None)
        sig = report.signatures[0]

        # Single-seed RMSE (model's own prediction on test)
        with torch.no_grad():
            preds = m(torch.tensor(X_test)).cpu().numpy().ravel()
        rmse = float(np.sqrt(np.mean((y_test - preds) ** 2))) * 100  # cm
        print(f"  Seed-1 single-model RMSE: {rmse:.2f} cm")

        # Per-seed top-K by Impact
        impact = np.asarray(sig.impact)
        topk_idx = np.argsort(impact)[-K:][::-1]
        topk_features = [feat_cols[i] for i in topk_idx]

        ck_set = set(ck_features)
        topk_set = set(topk_features)
        overlap = ck_set & topk_set
        only_ens = ck_set - topk_set
        only_seed = topk_set - ck_set
        print(f"  Per-seed top-{K} by Impact: {len(topk_features)} features")
        print(f"  Overlap with ensemble CK: {len(overlap)}/{K} ({100*len(overlap)/K:.0f}%)")
        print(f"  In ensemble-only: {len(only_ens)}")
        print(f"  In seed-1-only:   {len(only_seed)}")

        results[name] = dict(
            n_features=n_features,
            seed_1_single_rmse_cm=rmse,
            K=K,
            ensemble_ck=ck_features,
            seed_1_topk=topk_features,
            overlap_count=len(overlap),
            overlap_pct=100*len(overlap)/K if K else 0,
            features_in_seed_1_only=sorted(only_seed),
        )

    out = FFCA_AGENT / "case_studies" / "per_seed_analysis.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")

    print("\n=== Summary ===")
    print(f"{'Experiment':50s} {'K':>4s} {'Overlap':>10s} {'Seed-1-only':>12s}")
    for name, r in results.items():
        print(f"{name[:50]:50s} {r['K']:>4d} {r['overlap_count']:>4d} ({r['overlap_pct']:>3.0f}%) {len(r['features_in_seed_1_only']):>12d}")


if __name__ == "__main__":
    main()
