"""Validation 01 — Real NN (MLP) on Breast Cancer Wisconsin.

Trains a 30-feature MLP for 30 epochs, saves checkpoints, then runs
FFCAReport TWICE — once with all three improvements, once with the
baseline (no Cauchy-HVP, no Trust, no Co-Sens) — so we can compare.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from ffca import CheckpointLoader, FFCAReport, TabularAdapter


OUT = Path(__file__).resolve().parent / "01_tabular"
CK = OUT / "checkpoints"
OUT.mkdir(exist_ok=True, parents=True); CK.mkdir(exist_ok=True)


def make_model():
    return nn.Sequential(
        nn.Linear(30, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 2),
    )


def train_and_snapshot():
    torch.manual_seed(0); np.random.seed(0)
    data = load_breast_cancer()
    X = StandardScaler().fit_transform(data.data)
    X_tr, X_va, y_tr, y_va = train_test_split(X, data.target, test_size=0.2,
                                              stratify=data.target, random_state=42)
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_tr),
                                            torch.LongTensor(y_tr)),
                              batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_va),
                                          torch.LongTensor(y_va)),
                            batch_size=64)
    model = make_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    snapshots = (1, 5, 15, 30)
    ck_paths = []
    print("Training MLP …")
    for ep in range(1, 31):
        model.train()
        for x, y in train_loader:
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
        if ep in snapshots:
            p = CK / f"ep{ep:02d}.pt"
            torch.save(model.state_dict(), p)
            ck_paths.append((f"ep{ep}", str(p)))
            print(f"  saved {p.name}")
    return ck_paths, val_loader, list(data.feature_names)


def run_ffca(ck_paths, val_loader, names, *, improvements: bool, tag: str):
    print(f"\n=== Running FFCA — improvements={improvements} ===")
    adapter = TabularAdapter(make_model(), feature_names=names)
    ck = CheckpointLoader(make_model, ck_paths, device="cpu")
    rep = FFCAReport(
        adapter, val_loader,
        n_first_order_samples=64, n_hessian_samples=8,
        n_diag_probes=32, n_cauchy_probes=80, n_cauchy_samples=16,
        n_cosens_permutations=50, n_cosens_bootstrap=20,
        improvements=improvements,
    )
    t0 = time.time()
    rep.run(checkpoints=ck)
    out = OUT / tag
    if out.exists():
        shutil.rmtree(out)
    rep.save(out)
    elapsed = time.time() - t0
    return rep, out, elapsed


def summarize(rep, out, elapsed, label):
    last = rep.signatures[-1]
    print(f"\n--- {label} ---")
    print(f"  elapsed: {elapsed:.1f}s")
    print(f"  features: {last.n_features}, archetypes: {np.bincount(last.archetypes, minlength=8).tolist()}")
    print(f"  interaction method: {last.metadata['interaction_method']}")
    top = last.top_k(5, by="impact")
    print(f"  top-5 by impact:")
    for i in top:
        ci = f" [{last.interaction_ci[i,0]:.3f}, {last.interaction_ci[i,1]:.3f}]" if last.interaction_ci is not None else ""
        print(f"    {names[i]:<25} I={last.impact[i]:.3f}  X={last.interaction[i]:.3f}{ci}")
    if rep.trust:
        s = rep.trust.summary()
        print(f"  trust decisions: {dict(s)}")
    if rep.cosens:
        d = rep.cosens.diagnostics
        print(f"  co-sens k={d['k']} silhouette={d['silhouette_observed']:.3f} abort={d['abort_recommended']}")
    print(f"  artifacts: {out.name}/ ({len(list((out/'plots').glob('*.png')))} PNGs)")


if __name__ == "__main__":
    ck_paths, val_loader, names = train_and_snapshot()

    rep_on, out_on, t_on = run_ffca(ck_paths, val_loader, names,
                                      improvements=True, tag="with_improvements")
    rep_off, out_off, t_off = run_ffca(ck_paths, val_loader, names,
                                         improvements=False, tag="baseline")

    summarize(rep_on, out_on, t_on, "WITH all 3 improvements")
    summarize(rep_off, out_off, t_off, "BASELINE (no improvements)")

    # Disagreement metric: top-5 overlap
    top5_on = set(rep_on.signatures[-1].top_k(5, by="impact").tolist())
    top5_off = set(rep_off.signatures[-1].top_k(5, by="impact").tolist())
    overlap = len(top5_on & top5_off)
    print(f"\nTop-5 overlap (impact ranking): {overlap}/5")

    payload = {
        "with_improvements": {
            "elapsed_s": t_on,
            "interaction_method": rep_on.signatures[-1].metadata["interaction_method"],
            "trust_summary": rep_on.trust.summary() if rep_on.trust else None,
            "cosens_diagnostics": rep_on.cosens.diagnostics if rep_on.cosens else None,
            "top5_impact": [names[i] for i in rep_on.signatures[-1].top_k(5).tolist()],
        },
        "baseline": {
            "elapsed_s": t_off,
            "interaction_method": rep_off.signatures[-1].metadata["interaction_method"],
            "top5_impact": [names[i] for i in rep_off.signatures[-1].top_k(5).tolist()],
        },
        "top5_overlap": overlap,
    }
    (OUT / "compare.json").write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nCompare summary → {OUT / 'compare.json'}")
