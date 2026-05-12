"""
Validation harness for the real Cauchy-HVP implementation.

We construct a small differentiable model where the exact input Hessian is
computable row-by-row via `torch.autograd.grad`, then compare Cauchy-HVP's
||H_i:||_1 estimate against the ground truth.

Pass criteria (post-hoc, set generously to allow Cauchy's heavy tails):
  - Spearman rank correlation >= 0.95
  - Median relative error < 0.25
  - 95% Wald CI covers truth for >= 90% of features at B=200
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torch.utils.data import DataLoader, TensorDataset

from ffca.core.adapter import FFCAModelAdapter
from ffca.improvements_pkg import CauchyHVP


class _SumAdapter(FFCAModelAdapter):
    """Plain adapter that returns out.sum() — matches the truth Hessian."""
    def __init__(self, model, d):
        super().__init__(model)
        self.n_features = d
        self.feature_shape = (d,)
        self.feature_names = [f"x{i}" for i in range(d)]
    def feature_input(self, batch):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        return x.to(device=self.device(), dtype=self.dtype()).clone().detach().requires_grad_(True)
    def scalar_output(self, x, batch):
        return self.model(x).sum()


def exact_row_l1(model, x):
    """Compute ||H_i:||_1 for every i by row-by-row autograd."""
    x = x.detach().requires_grad_(True)
    out = model(x).sum()
    g = torch.autograd.grad(out, x, create_graph=True)[0]
    g_flat = g.reshape(-1)
    d = g_flat.numel()
    H = torch.zeros(d, d, dtype=g_flat.dtype, device=g_flat.device)
    for i in range(d):
        gi = torch.autograd.grad(g_flat[i], x, retain_graph=(i < d - 1))[0]
        H[i] = gi.reshape(-1)
    return H.abs().sum(dim=1).detach().cpu().numpy(), H.diag().abs().detach().cpu().numpy()


class SoftPlusMLP(nn.Module):
    def __init__(self, d_in=16, hidden=24, d_out=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.Softplus(beta=2.0),
            nn.Linear(hidden, hidden), nn.Softplus(beta=2.0),
            nn.Linear(hidden, d_out),
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)


def run_one(d_in=16, n_samples=4, B=200, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = SoftPlusMLP(d_in=d_in).to(torch.float64)
    model.eval()
    x = torch.randn(n_samples, d_in, dtype=torch.float64)

    # Ground truth: average row L1 across samples
    truth_row_l1 = np.zeros(d_in)
    truth_diag = np.zeros(d_in)
    for i in range(n_samples):
        rl1, diag = exact_row_l1(model, x[i])
        truth_row_l1 += rl1
        truth_diag += diag
    truth_row_l1 /= n_samples
    truth_diag /= n_samples
    truth_interactions = np.maximum(truth_row_l1 - truth_diag, 0)

    # Cauchy-HVP via the public adapter+loader API.
    est = CauchyHVP(n_probes=B, seed=seed)
    adapter = _SumAdapter(model, d=d_in)
    loader = DataLoader(TensorDataset(x), batch_size=n_samples)
    est_interactions, est_ci = est.estimate(adapter, loader, n_samples=n_samples)
    est_row_l1 = est.results['row_l1']
    est_diag = est.results['diag_abs']

    # Metrics
    from scipy.stats import spearmanr, pearsonr
    sp_r, _ = spearmanr(truth_row_l1, est_row_l1)
    pe_r, _ = pearsonr(truth_row_l1, est_row_l1)
    rel_err = np.abs(est_row_l1 - truth_row_l1) / np.maximum(truth_row_l1, 1e-8)

    # CI coverage on the off-diagonal-corrected interaction
    cov = np.mean(
        (est.results['ci_lower'] <= truth_interactions) &
        (truth_interactions <= est.results['ci_upper'])
    )

    print(f"\n--- d={d_in}, B={B}, n_samples={n_samples}, seed={seed} ---")
    print(f"truth row L1:     [{truth_row_l1.min():.3f}, {truth_row_l1.max():.3f}]"
          f"  mean={truth_row_l1.mean():.3f}")
    print(f"est   row L1:     [{est_row_l1.min():.3f}, {est_row_l1.max():.3f}]"
          f"  mean={est_row_l1.mean():.3f}")
    print(f"Spearman(row L1) = {sp_r:.4f}")
    print(f"Pearson(row L1)  = {pe_r:.4f}")
    print(f"Median relative error = {np.median(rel_err):.3f}")
    print(f"95% CI coverage of true interaction = {cov:.2%}")
    print(f"diag |H_ii| Pearson = {pearsonr(truth_diag, est_diag)[0]:.4f}")

    return {
        'd': d_in, 'B': B, 'n_samples': n_samples,
        'spearman': float(sp_r), 'pearson': float(pe_r),
        'median_rel_err': float(np.median(rel_err)),
        'ci_coverage': float(cov),
    }


def main():
    rows = []
    print("============================================================")
    print("CAUCHY-HVP VALIDATION — exact Hessian comparison")
    print("============================================================")
    for B in (50, 100, 200):
        rows.append(run_one(d_in=16, n_samples=4, B=B, seed=0))
    # Larger d
    rows.append(run_one(d_in=32, n_samples=3, B=200, seed=1))

    print("\n=== Summary ===")
    print(f"{'d':>4} {'B':>4} {'samples':>8} {'spearman':>10} {'pearson':>9} "
          f"{'med_rel_err':>12} {'ci_cov':>8}")
    for r in rows:
        print(f"{r['d']:>4} {r['B']:>4} {r['n_samples']:>8} {r['spearman']:>10.4f} "
              f"{r['pearson']:>9.4f} {r['median_rel_err']:>12.3f} {r['ci_coverage']:>8.2%}")

    # Pass / fail
    pass_threshold = all(
        r['spearman'] >= 0.90 and r['median_rel_err'] < 0.10 and r['ci_coverage'] >= 0.85
        for r in rows if r['B'] >= 100
    )
    print(f"\nVALIDATION {'PASSED' if pass_threshold else 'FAILED'}")
    return pass_threshold


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
