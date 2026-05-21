"""
run_ffca_ensemble_phaseB_full.py
================================
Phase B-full: corrected ensemble-mode FFCA on all 20 original (un-pruned) +
all 20 aggressive-pruned models. This is the proper first-pass of the reset:
re-do FFCA on the actual project with the correct seed-axis interpretation,
THEN see what the agent says.

Reuses architecture metadata from run_ffca_pruned.EXPERIMENTS.

Outputs:
  /Users/hnaja002/Documents/projects/compound_flooding/FFCA_resutls_before_prunning_ensemble/<cat>/<exp>/
  /Users/hnaja002/Documents/projects/compound_flooding/FFCA_results_After_prunning_ensemble/<cat>/<exp>/

Resumable: skips experiments whose report.json already exists.
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

CF = Path("/Users/hnaja002/Documents/projects/compound_flooding")
PRUNED_MODELS = CF / "MLMiami FFCA Prunned Results"
ORIG_MODELS   = CF / "mlmiamicompoundfloodpredictions"
CSV_PATH      = ORIG_MODELS / "Miami_GWL_WL_RAIN_GATE_2017_2024.csv"
# v0.7: the legacy FFCA_resutls_before_prunning/ dir was deleted in the
# seed-vs-epoch reset cleanup. For ORIGINAL stage we use the corrected
# ensemble report itself as the feature enumeration source (it has the
# same features). For AGGRESSIVE_PRUNE stage we use the corrected
# aggressive-prune report similarly.
OUT_BEFORE = CF / "FFCA_resutls_before_prunning_ensemble"
OUT_AFTER  = CF / "FFCA_results_After_prunning_ensemble"

FFCA_PACKAGE = Path("/Users/hnaja002/Documents/projects/FFCA/FFCA_package")
sys.path.insert(0, str(FFCA_PACKAGE))
sys.path.insert(0, str(CF))

from ffca import FFCAReport, TabularAdapter, CheckpointLoader
from run_ffca_pruned import (
    MLP,
    EXPERIMENTS as RFP_EXPERIMENTS,  # 20 entries with name, ffca_report, lead_time, hp
    convert_keras_to_pytorch_checkpoints,
    build_feature_matrix,
    parse_feature,
    load_ffca_selected_features,
    N_ENSEMBLE,
    Y_BUFFER,
    TARGET_COL,
    TEST_YEARS,
)


def category_from_report_path(rel: str) -> str:
    """The first path component of cfg['ffca_report'] is the category folder."""
    return rel.split("/")[0]


def load_all_features(report_path: Path) -> dict[str, list[int]]:
    with open(report_path) as f:
        data = json.load(f)
    selected = defaultdict(set)
    for feat in data["trust"].keys():
        prefix, lag = parse_feature(feat)
        selected[prefix].add(lag)
    return {p: sorted(lags) for p, lags in selected.items()}


# v0.7: deterministic feature enumeration per experiment name + lead time,
# used when no existing FFCA report is available to read from. Verified
# against feature counts of all 20 experiments (e.g. 24hr_gate=223,
# 24hr_all_inputs=271).
BASE_CHANNELS = ["gwl", "wl", "rain", "stgH", "stgT", "gate1", "gate2"]


def features_from_experiment(name: str, lead: int) -> dict[str, list[int]]:
    """Return {channel: [lag, lag, ...]} for an experiment's full feature set.

    Channels with `perfect_prog` access have lag range [-24, +lead]; everything
    else has [-24, 0].
    """
    if "perfect_prog_all_inputs" in name:
        # gwl, stgH, stgT regular; wl, rain, gate1, gate2 perfect_prog
        pp = {"wl", "rain", "gate1", "gate2"}
    elif "perfect_prog_gate" in name:
        pp = {"gate1", "gate2"}
    elif "perfect_prog_wls" in name:
        pp = {"wl"}
    elif "perfect_prog_rain" in name:
        pp = {"rain"}
    elif "measured" in name:
        pp = set()
    else:
        raise ValueError(f"unrecognized experiment name: {name}")
    out: dict[str, list[int]] = {}
    for ch in BASE_CHANNELS:
        hi = lead if ch in pp else 0
        out[ch] = list(range(-24, hi + 1))
    return out


def run_stage(cfg: dict, stage: str, df_raw: pd.DataFrame, device: torch.device) -> dict | None:
    name = cfg["name"]
    cat  = category_from_report_path(cfg["ffca_report"])

    if stage == "original":
        keras_dir = ORIG_MODELS / cat / name / "models"  # no MLP subdir
        # Deterministic enumeration of full feature set (no source report needed)
        selected = features_from_experiment(name, cfg["lead_time"])
        out_dir = OUT_BEFORE / cat / name
        feature_label = "FULL"
    elif stage == "aggressive_prune":
        keras_dir = PRUNED_MODELS / cat / f"{name}_ffca" / "MLP" / "models"
        # KEEP-only feature set must come from a prior pruned report
        existing = OUT_AFTER / cat / name / "report.json"
        if not existing.exists():
            print(f"  [skip] no pruned-source report at {existing} to enumerate KEEP features from")
            return None
        selected = load_all_features(existing)
        out_dir = OUT_AFTER / cat / name
        feature_label = "KEEP-only"
    else:
        raise ValueError(stage)

    if (out_dir / "report.json").exists():
        return {"name": name, "stage": stage, "skipped": True}

    if not keras_dir.exists():
        print(f"  [skip] {stage:18s} {name} (models dir missing: {keras_dir})")
        return None

    print(f"=== {stage.upper():18s} {name:42s} ===")
    df, feat_cols, target = build_feature_matrix(df_raw, selected, TARGET_COL, cfg["lead_time"])
    y_min = float(df_raw[TARGET_COL].min())
    y_max = float(df_raw[TARGET_COL].max() * (1 + Y_BUFFER))
    df_test = df[df.index.year.isin(TEST_YEARS)]
    if df_test.empty:
        print(f"  [skip] no test data for {name}")
        return None
    X_test = df_test[feat_cols].to_numpy(dtype=np.float32)
    y_test = df_test[target].to_numpy(dtype=np.float32)
    loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
                        batch_size=32)
    print(f"  {feature_label}: {len(feat_cols)} features | test {X_test.shape}")

    out_dir.mkdir(parents=True, exist_ok=True)
    pt_dir = out_dir / "checkpoints"

    hp = cfg["hp"]
    t = time.time()
    cps = convert_keras_to_pytorch_checkpoints(
        keras_dir, pt_dir, n_features=len(feat_cols),
        hp=hp, y_min=y_min, y_max=y_max, device=device,
    )
    if not cps:
        print(f"  [skip] no checkpoints converted from {keras_dir}")
        return None

    adapter_model = MLP(n_features=len(feat_cols), neurons=hp["neurons"],
                       num_layers=hp["num_layers"], y_min=y_min, y_max=y_max).to(device).eval()
    adapter = TabularAdapter(adapter_model, feature_names=feat_cols)

    def factory():
        return MLP(n_features=len(feat_cols), neurons=hp["neurons"],
                   num_layers=hp["num_layers"], y_min=y_min, y_max=y_max).to(device).eval()

    ck = CheckpointLoader(factory, cps, device=device)
    report = FFCAReport(
        adapter, loader,
        n_first_order_samples=32, n_hessian_samples=8, n_diag_probes=24,
        n_cauchy_probes=64, n_cauchy_samples=8,
        improvements=True, mode="ensemble",
    )
    t_run = time.time()
    report.run(checkpoints=ck)
    print(f"  FFCA run: {time.time()-t_run:.1f}s")
    report.save(out_dir, save_plots=False)  # no plots in batch — fast

    findings = [(f.severity, f.name, f.headline) for f in (report.findings or [])]
    return {"name": name, "stage": stage, "n_features": len(feat_cols),
            "trust_summary": report.trust.summary() if report.trust else {},
            "findings": findings,
            "skipped": False}


def main():
    df_raw = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
    device = torch.device("cpu")
    print(f"Device: {device}; data: {df_raw.shape}\n")

    rows: list[dict] = []
    for cfg in RFP_EXPERIMENTS:
        for stage in ("original", "aggressive_prune"):
            r = run_stage(cfg, stage, df_raw, device)
            if r:
                rows.append(r)
                print()

    # Roll-up summary
    print("\n" + "="*100)
    print("Phase B-full summary")
    print("="*100)
    print(f"{'Experiment':45s} {'Stage':18s} {'n':>4s} {'Multi-modal':>11s} {'warns':30s}")
    for r in rows:
        if r.get("skipped"):
            continue
        ts = r.get("trust_summary", {})
        n = r.get("n_features", 0)
        mm = ts.get("INVESTIGATE (multi-modal seeds)", 0)
        mm_pct = f"{mm}/{n}={mm/n*100:.0f}%" if n else "?"
        warns = ",".join(name for sev, name, _ in r["findings"] if sev == "warn") or "-"
        print(f"{r['name'][:45]:45s} {r['stage']:18s} {n:>4d} {mm_pct:>11s} {warns[:30]:30s}")


if __name__ == "__main__":
    main()
