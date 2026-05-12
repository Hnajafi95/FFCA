"""Example 02 — Pixel-level FFCA on biased CIFAR-10.

Trains a small CNN with a known spurious correlation (white 2-pixel border
on vehicle classes 1/9) for 8 epochs, then runs FFCAReport with
PixelAdapter on the 32×32×3 = 3072 pixel feature space.

The expected explainability output:
  - the FBR diagnostic should fire if the model is exploiting the border
  - the pixel interaction heatmap should highlight the border ring
  - top-impact "features" (pixels) should cluster near the image edges

Run:
    python examples/02_image_cifar10_pixel.py

Expected runtime: 5–10 minutes on Apple Silicon (MPS).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ffca import FFCAReport, CheckpointLoader, PixelAdapter

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 8
BATCH = 64
LR = 1e-3
BIAS = 0.95


def make_cnn() -> nn.Module:
    class CIFAR10CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
            self.pool = nn.MaxPool2d(2)
            self.fc1 = nn.Linear(128 * 4 * 4, 256)
            self.fc2 = nn.Linear(256, 10)
            self.act = nn.ReLU()
            self.dropout = nn.Dropout(0.3)
        def forward(self, x):
            x = self.pool(self.act(self.conv1(x)))
            x = self.pool(self.act(self.conv2(x)))
            x = self.act(self.conv3(x))
            x = self.pool(x).view(x.size(0), -1)
            x = self.dropout(self.act(self.fc1(x)))
            return self.fc2(x)
    return CIFAR10CNN()


def add_border(img, label):
    if label in (1, 9) and torch.rand(1).item() < BIAS:
        img = img.clone()
        img[:, :2, :] = 1.0; img[:, -2:, :] = 1.0
        img[:, :, :2] = 1.0; img[:, :, -2:] = 1.0
    return img, label


def main():
    transform = T.Compose([T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)])
    root = Path(__file__).resolve().parents[1] / "data"
    train_ds = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=transform)
    val_ds = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=transform)

    def collate(batch):
        imgs = [add_border(im, lb)[0] for im, lb in batch]
        labels = torch.tensor([lb for _, lb in batch])
        return torch.stack(imgs), labels

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False)

    # --- train ---
    ckdir = Path(__file__).resolve().parent / "_ckpts_02_pixel"
    ckdir.mkdir(exist_ok=True)
    model = make_cnn().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    ck_paths = []
    snapshots = (1, 3, 6, EPOCHS)
    for ep in range(1, EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
        # quick validation
        model.eval()
        with torch.no_grad():
            n = c = 0
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                c += (model(x).argmax(1) == y).sum().item(); n += y.size(0)
        print(f"  epoch {ep}: val acc = {c/n:.3f}")
        if ep in snapshots:
            p = ckdir / f"ep{ep:02d}.pt"
            torch.save(model.state_dict(), p)
            ck_paths.append((f"ep{ep}", str(p)))

    # --- FFCA ---
    adapter = PixelAdapter(make_cnn().to(DEVICE), input_shape=(3, 32, 32))
    ck_loader = CheckpointLoader(lambda: make_cnn().to(DEVICE), ck_paths, device=DEVICE)
    report = FFCAReport(
        adapter, val_loader,
        n_first_order_samples=32, n_hessian_samples=8,
        n_diag_probes=24, n_cauchy_probes=40, n_cauchy_samples=8,
        n_cosens_permutations=20, n_cosens_bootstrap=10,
    ).run(checkpoints=ck_loader)

    out = Path(__file__).resolve().parents[1] / "experiments" / "ex02_cifar10_pixel"
    report.save(out)
    print(f"\nDone — {out}/report.md, plots/ ({sum(1 for _ in (out/'plots').iterdir())} PNGs)")


if __name__ == "__main__":
    main()
