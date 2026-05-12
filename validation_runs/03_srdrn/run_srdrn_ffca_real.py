"""Validation 03 (real data) — FFCA on Original_SRDRN_epoch_160 with real
GCM input variables from GCM_FL/lowres-files-train.

Per-channel ordering and shape match the train_original.py pipeline:
  - 6 channels in order: tas, pr, huss, sfcWind, tasmax, tasmin
  - Spatial shape: (lat=13, lon=11) — H=13, W=11
  - Standardised per-channel to mean=0, std=1 (training-period statistics)

Compares against the FFCA paper's Phase 2.1 ground truth:
  Impact ranking should be roughly  tasmax > tas > huss > tasmin > sfcWind > pr
  pr should be Noise Candidate; tasmax should be Catalyst.
"""
from __future__ import annotations

import glob
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from srdrn_pytorch import load_srdrn_from_h5

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ffca import ChannelAdapter, FFCAReport, PixelAdapter
from ffca.core.scalars import regression

SRDRN_H5 = ("/Users/hnaja002/Documents/projects/FFCA/FFCA_dump/"
            "FFCA_archetype_dynamic/claude_playground/SRDRN_Files/"
            "Original_SRDRN_epoch_160")

DATA_DIR = ("/Users/hnaja002/Documents/projects/FFCA/FFCA_dump/"
            "FFCA_archetype_dynamic/claude_playground/SRDRN_Files/"
            "GCM_FL/lowres-files-train")

CHANNELS = ["tas", "pr", "huss", "sfcWind", "tasmax", "tasmin"]
PAPER_RANKING = ["tasmax", "tas", "huss", "tasmin", "sfcWind", "pr"]

OUT = Path(__file__).resolve().parent
N_TIMES = 200                # subsample for FFCA budget
N_SAMPLES = 12
DEVICE = torch.device("cpu")


def load_real_inputs(n_times: int = N_TIMES):
    """Returns (X, stats) where X is (n_times, 6, 13, 11) float32 standardised."""
    arrays = []
    stats = {}
    rng = np.random.default_rng(0)
    for c in CHANNELS:
        path = glob.glob(f"{DATA_DIR}/{c}_day_*.nc")[0]
        ds = xr.open_dataset(path)
        arr = ds[c].values            # (T, lat, lon) = (T, 13, 11)
        ds.close()
        # Random temporal subsample for diversity
        if arr.shape[0] > n_times:
            idx = np.sort(rng.choice(arr.shape[0], size=n_times, replace=False))
            arr = arr[idx]
        # Standardise channel-wise
        mean = float(arr.mean())
        std = float(arr.std() + 1e-8)
        arr_z = (arr - mean) / std
        stats[c] = {"mean": mean, "std": std, "shape": list(arr.shape)}
        arrays.append(arr_z.astype(np.float32))
        print(f"  {c}: raw mean={mean:.3f} std={std:.3f}, shape={arr.shape}")
    X = np.stack(arrays, axis=1)       # (T, 6, 13, 11)
    return X, stats


def run_one(adapter, loader, *, improvements: bool, tag: str):
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


def summarize(rep, out, elapsed, label, feature_names=None):
    last = rep.signatures[-1]
    arch = np.bincount(last.archetypes, minlength=8).tolist()
    n_plots = len(list((out / "plots").glob("*.png")))
    print(f"\n--- {label} ---")
    print(f"  elapsed: {elapsed:.1f}s, features: {last.n_features}, "
          f"method: {last.metadata['interaction_method']}")
    print(f"  archetypes [N,HI,W,Cat,NL,V,St,Cx]: {arch}")
    print(f"  interaction range: [{last.interaction.min():.4f}, {last.interaction.max():.4f}]")
    if feature_names:
        order = np.argsort(-last.impact)
        print(f"  impact ranking:")
        for r, i in enumerate(order, 1):
            from ffca.core.archetypes import ARCHETYPE_NAMES
            arch_name = ARCHETYPE_NAMES[int(last.archetypes[i])]
            print(f"    {r}. {feature_names[i]:<10}  I={last.impact[i]:.4f}  "
                  f"X={last.interaction[i]:.4f}  arch={arch_name}")
    if rep.cosens and rep.cosens.diagnostics.get("k"):
        d = rep.cosens.diagnostics
        print(f"  cosens k={d['k']} silh={d['silhouette_observed']:.3f} abort={d['abort_recommended']}")
    print(f"  → {n_plots} plots in {out.name}/plots/")


def main():
    print(f"Loading SRDRN from {Path(SRDRN_H5).name} …")
    model = load_srdrn_from_h5(SRDRN_H5).to(DEVICE)
    print(f"  {sum(p.numel() for p in model.parameters()):,} parameters")

    print(f"\nLoading real GCM-FL inputs ({N_TIMES} samples) …")
    X, stats = load_real_inputs(N_TIMES)
    print(f"  X shape: {X.shape}  range: [{X.min():.3f}, {X.max():.3f}]")
    loader = DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=4)

    summary = {"h5": SRDRN_H5, "data_dir": DATA_DIR,
                "n_samples_loaded": int(X.shape[0]),
                "channels": CHANNELS, "channel_stats": stats,
                "paper_ranking": PAPER_RANKING}

    # Channel-level FFCA at the 6-variable input dimension (mean-pool over
    # spatial). For pure α-FFCA-style behavior we use a hand-built adapter.
    print("\n=== SRDRN — α-FFCA (6 input variables, channel-mean inputs) ===")
    from ffca.core.adapter import FFCAModelAdapter

    class AlphaFFCAAdapter(FFCAModelAdapter):
        """Treat the 6 per-channel scalar multipliers as the feature axis.
        Each scalar multiplies its corresponding (13, 11) channel image.
        Output scalar = mean precip prediction over the high-res grid.
        """
        n_features = 6
        feature_shape = (6,)
        feature_names = CHANNELS

        def feature_input(self, batch):
            # We pass alpha = ones (B, 6) and the model uses these as multipliers
            x_img = batch[0].to(device=self.device(), dtype=self.dtype())
            self._batch_img = x_img
            alpha = torch.ones(x_img.size(0), 6, dtype=x_img.dtype,
                                device=x_img.device).requires_grad_(True)
            return alpha

        def scalar_output(self, alpha, batch):
            scaled = self._batch_img * alpha[:, :, None, None]
            out = self.model(scaled)
            return out.mean()    # scalar regression target

    for impr, tag in ((True, "alpha_with"), (False, "alpha_baseline")):
        ad = AlphaFFCAAdapter(model)
        rep, out, t = run_one(ad, loader, improvements=impr, tag=tag)
        summarize(rep, out, t, f"α-FFCA — improvements={impr}", feature_names=CHANNELS)
        last = rep.signatures[-1]
        order = [CHANNELS[i] for i in np.argsort(-last.impact)]
        summary.setdefault("alpha", {})[tag] = {
            "elapsed_s": t,
            "interaction_method": last.metadata["interaction_method"],
            "ranking": order,
            "kendall_tau_vs_paper": _kendall(order, PAPER_RANKING),
            "archetypes_per_var": {CHANNELS[i]: int(last.archetypes[i]) for i in range(6)},
        }

    # Also do a channel-level FFCA at conv_post (64 channels)
    print("\n=== SRDRN — channel-level FFCA at conv_post (64 channels) ===")
    for impr, tag in ((True, "ch64_with"), (False, "ch64_baseline")):
        ad = ChannelAdapter(model, layer_name="conv_post", scalar=regression())
        rep, out, t = run_one(ad, loader, improvements=impr, tag=tag)
        summarize(rep, out, t, f"CONV_POST — improvements={impr}")
        summary.setdefault("ch64", {})[tag] = {
            "elapsed_s": t,
            "interaction_method": rep.signatures[-1].metadata["interaction_method"],
            "archetypes": np.bincount(rep.signatures[-1].archetypes, minlength=8).tolist(),
        }

    # And one of the heavy upsampling layers (512 channels) — equivalent to
    # FFCA paper's Phase 2.2 conv2d_34.
    print("\n=== SRDRN — channel-level FFCA at up1.conv (512 channels) ===")
    for impr, tag in ((True, "ch512_with"), (False, "ch512_baseline")):
        ad = ChannelAdapter(model, layer_name="up1.conv", scalar=regression())
        rep, out, t = run_one(ad, loader, improvements=impr, tag=tag)
        summarize(rep, out, t, f"UP1.CONV (≈Phase-2.2 conv2d_34) — improvements={impr}")
        summary.setdefault("ch512", {})[tag] = {
            "elapsed_s": t,
            "interaction_method": rep.signatures[-1].metadata["interaction_method"],
            "archetypes": np.bincount(rep.signatures[-1].archetypes, minlength=8).tolist(),
        }

    (OUT / "compare_real.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSaved {OUT/'compare_real.json'}")


def _kendall(observed: list[str], expected: list[str]) -> float:
    from scipy.stats import kendalltau
    obs_rank = {name: i for i, name in enumerate(observed)}
    exp_rank = {name: i for i, name in enumerate(expected)}
    keys = list(exp_rank.keys())
    tau, _ = kendalltau([obs_rank[k] for k in keys], [exp_rank[k] for k in keys])
    return float(tau)


if __name__ == "__main__":
    main()
