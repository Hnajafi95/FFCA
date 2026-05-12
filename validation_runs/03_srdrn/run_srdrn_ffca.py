"""Validation 03 — FFCA on the real Original_SRDRN_epoch_160 model.

Loads the Keras HDF5 weights into our PyTorch port, then runs FFCA both
WITH and WITHOUT the three improvements at:
  - Input level (PixelAdapter) — 6 climate channels × 11 × 13 input
  - Channel level (ChannelAdapter at conv_post) — 64 channels
  - Channel level (ChannelAdapter at up1.conv) — 512 channels
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from srdrn_pytorch import load_srdrn_from_h5

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ffca import ChannelAdapter, FFCAReport, PixelAdapter
from ffca.core.scalars import regression


SRDRN_H5 = ("/Users/hnaja002/Documents/projects/FFCA/FFCA_dump/"
            "FFCA_archetype_dynamic/claude_playground/SRDRN_Files/"
            "Original_SRDRN_epoch_160")

OUT = Path(__file__).resolve().parent
N_SAMPLES = 12   # SRDRN is heavy; small budget
DEVICE = torch.device("cpu")  # MPS doesn't support float64 reliably


def make_synthetic_climate(n: int = 32, h: int = 11, w: int = 13, seed: int = 0):
    """6-channel climate-like input. Realistic ranges per Phase 2.1:
       tas, pr, huss, sfcWind, tasmax, tasmin (all standardised)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 6, h, w)).astype(np.float32)
    return x


def run_one(adapter, *, improvements: bool, tag: str, loader):
    rep = FFCAReport(
        adapter, loader,
        n_first_order_samples=N_SAMPLES, n_hessian_samples=4,
        n_diag_probes=12, n_cauchy_probes=20, n_cauchy_samples=4,
        n_cosens_permutations=15, n_cosens_bootstrap=8,
        improvements=improvements,
    )
    t0 = time.time()
    rep.run()
    elapsed = time.time() - t0
    out = OUT / tag
    if out.exists():
        shutil.rmtree(out)
    rep.save(out)
    return rep, out, elapsed


def summarize(rep, out, elapsed, label):
    last = rep.signatures[-1]
    arch = np.bincount(last.archetypes, minlength=8).tolist()
    print(f"\n--- {label} ---")
    print(f"  elapsed: {elapsed:.1f}s")
    print(f"  features: {last.n_features}, interaction method: {last.metadata['interaction_method']}")
    print(f"  archetypes [N,HI,W,Cat,NL,V,St,Cx]: {arch}")
    print(f"  interaction range: [{last.interaction.min():.4f}, {last.interaction.max():.4f}]")
    if rep.cosens:
        d = rep.cosens.diagnostics
        print(f"  cosens k={d['k']} silh={d['silhouette_observed']:.3f} abort={d['abort_recommended']}")
    n_plots = len(list((out / 'plots').glob('*.png')))
    print(f"  → {n_plots} plots in {out.name}/plots/")


def main():
    print("Loading SRDRN from HDF5 …")
    model = load_srdrn_from_h5(SRDRN_H5).to(DEVICE)
    print(f"  {sum(p.numel() for p in model.parameters()):,} parameters")

    X = torch.from_numpy(make_synthetic_climate(N_SAMPLES * 3))
    loader = DataLoader(TensorDataset(X), batch_size=4)

    summary = {"h5": SRDRN_H5}

    # ------- Test A: input pixel (6×11×13 = 858 features) -------
    print("\n=== SRDRN — input-level FFCA (858 features) ===")
    for impr, tag in ((True, "input_with"), (False, "input_baseline")):
        ad = PixelAdapter(model, input_shape=(6, 11, 13), scalar=regression())
        rep, out, t = run_one(ad, improvements=impr, tag=tag, loader=loader)
        summarize(rep, out, t, f"INPUT — improvements={impr}")
        summary.setdefault("input", {})[tag] = {
            "elapsed_s": t,
            "interaction_method": rep.signatures[-1].metadata["interaction_method"],
            "archetypes": np.bincount(rep.signatures[-1].archetypes, minlength=8).tolist(),
            "interaction_range": [float(rep.signatures[-1].interaction.min()),
                                    float(rep.signatures[-1].interaction.max())],
        }

    # ------- Test B: channel-level at conv_post (64 channels) -------
    print("\n=== SRDRN — channel-level FFCA at conv_post (64 channels) ===")
    for impr, tag in ((True, "ch64_with"), (False, "ch64_baseline")):
        ad = ChannelAdapter(model, layer_name="conv_post", scalar=regression())
        rep, out, t = run_one(ad, improvements=impr, tag=tag, loader=loader)
        summarize(rep, out, t, f"CONV_POST — improvements={impr}")
        summary.setdefault("ch64", {})[tag] = {
            "elapsed_s": t,
            "interaction_method": rep.signatures[-1].metadata["interaction_method"],
            "archetypes": np.bincount(rep.signatures[-1].archetypes, minlength=8).tolist(),
        }

    # ------- Test C: channel-level at up1.conv (512 channels, ≈ FFCA paper Phase 2.2's conv2d_34) -------
    print("\n=== SRDRN — channel-level FFCA at up1.conv (512 channels) ===")
    for impr, tag in ((True, "ch512_with"), (False, "ch512_baseline")):
        ad = ChannelAdapter(model, layer_name="up1.conv", scalar=regression())
        rep, out, t = run_one(ad, improvements=impr, tag=tag, loader=loader)
        summarize(rep, out, t, f"UP1.CONV (≈conv2d_34) — improvements={impr}")
        summary.setdefault("ch512", {})[tag] = {
            "elapsed_s": t,
            "interaction_method": rep.signatures[-1].metadata["interaction_method"],
            "archetypes": np.bincount(rep.signatures[-1].archetypes, minlength=8).tolist(),
        }

    (OUT / "compare.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSaved {OUT/'compare.json'}")


if __name__ == "__main__":
    main()
