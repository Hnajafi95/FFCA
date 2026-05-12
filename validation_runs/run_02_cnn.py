"""Validation 02 — Real CNN on CIFAR-10.

Trains a small CNN for 8 epochs, saves 4 checkpoints, runs FFCA at TWO
levels:
  - PixelAdapter:  3 × 32 × 32 = 3072 features (input pixels)
  - ChannelAdapter: conv3's 128 channels (intermediate features)

Each level is run twice — with and without the three improvements —
producing 4 reports total.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from ffca import CheckpointLoader, ChannelAdapter, FFCAReport, PixelAdapter

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")

OUT = Path(__file__).resolve().parent / "02_cnn"
CK = OUT / "checkpoints"
OUT.mkdir(parents=True, exist_ok=True); CK.mkdir(exist_ok=True)

EPOCHS = 8
BATCH = 64


class CIFAR10CNN(nn.Module):
    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1); self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1); self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1); self.act3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256); self.act4 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = self.act3(self.conv3(x))
        x = self.pool3(x).view(x.size(0), -1)
        x = self.dropout(self.act4(self.fc1(x)))
        return self.fc2(x)


def train_and_snapshot():
    transform = T.Compose([T.ToTensor(), T.Normalize((0.5,)*3, (0.5,)*3)])
    root = Path(__file__).resolve().parents[1] / "data"
    train_ds = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=transform)
    val_ds = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False)

    torch.manual_seed(0); np.random.seed(0)
    model = CIFAR10CNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    snapshots = (1, 3, 6, EPOCHS)
    ck_paths = []

    print(f"Training CIFAR-10 CNN on {DEVICE} …")
    for ep in range(1, EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            c = n = 0
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                c += (model(x).argmax(1) == y).sum().item(); n += y.size(0)
        print(f"  epoch {ep}: val acc = {c/n:.3f}")
        if ep in snapshots:
            p = CK / f"ep{ep:02d}.pt"
            torch.save(model.state_dict(), p)
            ck_paths.append((f"ep{ep}", str(p)))
    return ck_paths, val_loader


def _make_factory():
    def factory():
        m = CIFAR10CNN().to(DEVICE)
        return m
    return factory


def run_pixel(ck_paths, val_loader, *, improvements: bool, tag: str):
    print(f"\n=== Pixel-level FFCA — improvements={improvements} ===")
    adapter = PixelAdapter(_make_factory()(), input_shape=(3, 32, 32))
    ck = CheckpointLoader(_make_factory(), ck_paths, device=DEVICE)
    rep = FFCAReport(
        adapter, val_loader,
        n_first_order_samples=16, n_hessian_samples=4,
        n_diag_probes=16, n_cauchy_probes=20, n_cauchy_samples=4,
        n_cosens_permutations=15, n_cosens_bootstrap=8,
        improvements=improvements,
    )
    t0 = time.time()
    rep.run(checkpoints=ck)
    elapsed = time.time() - t0
    out = OUT / f"pixel_{tag}"
    if out.exists():
        shutil.rmtree(out)
    rep.save(out)
    return rep, out, elapsed


def run_channel(ck_paths, val_loader, *, improvements: bool, tag: str):
    print(f"\n=== Channel-level FFCA (act3) — improvements={improvements} ===")
    adapter = ChannelAdapter(_make_factory()(), layer_name="act3")
    ck = CheckpointLoader(_make_factory(), ck_paths, device=DEVICE)
    rep = FFCAReport(
        adapter, val_loader,
        n_first_order_samples=32, n_hessian_samples=8,
        n_diag_probes=24, n_cauchy_probes=40, n_cauchy_samples=8,
        n_cosens_permutations=20, n_cosens_bootstrap=10,
        improvements=improvements,
    )
    t0 = time.time()
    rep.run(checkpoints=ck)
    elapsed = time.time() - t0
    out = OUT / f"channel_{tag}"
    if out.exists():
        shutil.rmtree(out)
    rep.save(out)
    return rep, out, elapsed


def summarize(rep, out, elapsed, label):
    last = rep.signatures[-1]
    n_plots = len(list((out / "plots").glob("*.png")))
    arch_counts = np.bincount(last.archetypes, minlength=8).tolist()
    print(f"\n--- {label} ---")
    print(f"  elapsed: {elapsed:.1f}s")
    print(f"  features: {last.n_features}")
    print(f"  archetypes [N,HI,W,Cat,NL,V,St,Cx]: {arch_counts}")
    print(f"  interaction method: {last.metadata['interaction_method']}")
    print(f"  interaction range: [{last.interaction.min():.4f}, {last.interaction.max():.4f}]")
    if rep.trust:
        print(f"  trust: {dict(rep.trust.summary())}")
    if rep.cosens:
        d = rep.cosens.diagnostics
        print(f"  co-sens k={d['k']} silh={d['silhouette_observed']:.3f} abort={d['abort_recommended']}")
    print(f"  → {n_plots} plots in {out.name}/plots/")


if __name__ == "__main__":
    ck_paths, val_loader = train_and_snapshot()

    p_on, p_on_out, t1 = run_pixel(ck_paths, val_loader, improvements=True, tag="with")
    p_off, p_off_out, t2 = run_pixel(ck_paths, val_loader, improvements=False, tag="baseline")
    c_on, c_on_out, t3 = run_channel(ck_paths, val_loader, improvements=True, tag="with")
    c_off, c_off_out, t4 = run_channel(ck_paths, val_loader, improvements=False, tag="baseline")

    summarize(p_on, p_on_out, t1, "PIXEL — WITH improvements")
    summarize(p_off, p_off_out, t2, "PIXEL — BASELINE")
    summarize(c_on, c_on_out, t3, "CHANNEL act3 — WITH improvements")
    summarize(c_off, c_off_out, t4, "CHANNEL act3 — BASELINE")

    # Save summary
    summary = {
        "device": str(DEVICE), "epochs": EPOCHS, "checkpoints": [c[0] for c in ck_paths],
        "pixel": {
            "with": {"elapsed_s": t1, "interaction_method": p_on.signatures[-1].metadata["interaction_method"],
                      "archetypes": np.bincount(p_on.signatures[-1].archetypes, minlength=8).tolist(),
                      "trust": p_on.trust.summary() if p_on.trust else None,
                      "fbr": float(PixelAdapter(_make_factory()(), input_shape=(3,32,32)).fbr(p_on.signatures[-1].interaction))},
            "baseline": {"elapsed_s": t2, "interaction_method": p_off.signatures[-1].metadata["interaction_method"],
                          "archetypes": np.bincount(p_off.signatures[-1].archetypes, minlength=8).tolist(),
                          "fbr": float(PixelAdapter(_make_factory()(), input_shape=(3,32,32)).fbr(p_off.signatures[-1].interaction))},
        },
        "channel": {
            "with": {"elapsed_s": t3, "interaction_method": c_on.signatures[-1].metadata["interaction_method"],
                      "archetypes": np.bincount(c_on.signatures[-1].archetypes, minlength=8).tolist(),
                      "trust": c_on.trust.summary() if c_on.trust else None,
                      "cosens_abort": c_on.cosens.diagnostics["abort_recommended"] if c_on.cosens else None},
            "baseline": {"elapsed_s": t4, "interaction_method": c_off.signatures[-1].metadata["interaction_method"],
                          "archetypes": np.bincount(c_off.signatures[-1].archetypes, minlength=8).tolist()},
        },
    }
    (OUT / "compare.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {OUT/'compare.json'}")
