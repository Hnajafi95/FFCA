"""Example 01 — Tabular FFCA on Breast Cancer Wisconsin.

Trains a small MLP, saves 4 checkpoints, then runs FFCAReport with the
TabularAdapter and writes a Markdown report + JSON + plots into out/.

Run:
    python examples/01_tabular_breast_cancer.py

Expected runtime: ~30 seconds on a laptop CPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ffca import FFCAReport, CheckpointLoader, TabularAdapter


def make_model(n_features: int = 30) -> nn.Module:
    return nn.Sequential(
        nn.Linear(n_features, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 2),
    )


def main():
    data = load_breast_cancer()
    feature_names = list(data.feature_names)
    X = StandardScaler().fit_transform(data.data)
    X_tr, X_va, y_tr, y_va = train_test_split(X, data.target, test_size=0.2,
                                              stratify=data.target, random_state=42)

    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_tr),
                                            torch.LongTensor(y_tr)),
                              batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_va),
                                          torch.LongTensor(y_va)),
                            batch_size=64)

    # --- train and snapshot ---
    ckdir = Path(__file__).resolve().parent / "_ckpts_01_tabular"
    ckdir.mkdir(exist_ok=True)
    model = make_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    snapshot_epochs = (1, 5, 15, 30)
    ck_paths = []
    for ep in range(1, max(snapshot_epochs) + 1):
        model.train()
        for x, y in train_loader:
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
        if ep in snapshot_epochs:
            p = ckdir / f"ep{ep:02d}.pt"
            torch.save(model.state_dict(), p)
            ck_paths.append((f"ep{ep}", str(p)))

    # --- FFCA report (multi-checkpoint) ---
    adapter = TabularAdapter(make_model(), feature_names=feature_names)
    ck_loader = CheckpointLoader(make_model, ck_paths, device="cpu")
    report = FFCAReport(
        adapter, val_loader,
        n_first_order_samples=64, n_hessian_samples=8,
        n_diag_probes=32, n_cauchy_probes=80, n_cauchy_samples=16,
        n_cosens_permutations=50, n_cosens_bootstrap=20,
    ).run(checkpoints=ck_loader)

    out = Path(__file__).resolve().parents[1] / "experiments" / "ex01_breast_cancer"
    report.save(out)
    print(f"\nDone — report at {out}/report.md, plots in {out}/plots/")


if __name__ == "__main__":
    main()
