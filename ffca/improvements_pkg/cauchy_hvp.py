"""CauchyHVP — adapter-aware real Pearlmutter HVP with Cauchy(0,1) probes.

If z_j ~ Cauchy(0, 1) iid, then for the i-th Hessian row,
    w_i = Σ_j H_ij z_j  ~  Cauchy(0, ||H_i:||_1)
so  median(|w_i|) is an unbiased estimator of  ||H_i:||_1.

Interaction = max(||H_i:||_1 − |H_ii|, 0). |H_ii| is estimated separately
via Rademacher Hutchinson on the same B-probe budget.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch

from ..core.adapter import FFCAModelAdapter


class CauchyHVP:
    def __init__(
        self,
        n_probes: int = 100,
        clamp: float = 1e4,
        seed: int = 42,
        dtype: torch.dtype = torch.float64,
        memory_budget_mb: float = 256.0,
    ):
        self.n_probes = n_probes
        self.clamp = clamp
        self.seed = seed
        self.dtype = dtype
        self.memory_budget_mb = memory_budget_mb
        self.results: dict = {}

    # -------------------------------------------------- adapter-aware API
    def estimate(
        self,
        adapter: FFCAModelAdapter,
        loader: Iterable,
        n_samples: int = 16,
        batch_chunk: int = 4,
    ):
        """Returns (interactions, ci) where ci is (d, 2)."""
        device = adapter.device()
        was_training = adapter.model.training
        adapter.model.eval()

        # Always run autograd in the model's dtype to avoid mat1/mat2 mismatch.
        # We only use self.dtype (default float64) for the *accumulators* on CPU.
        target_dtype = adapter.dtype()

        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        B = self.n_probes

        row_l1 = None
        diag_abs = None
        n_used = 0
        d = None

        for batch in loader:
            if n_used >= n_samples:
                break

            x = adapter.feature_input(batch)
            b = x.size(0)
            if d is None:
                d = int(np.prod(x.shape[1:]))
                row_l1 = torch.zeros(d, dtype=self.dtype)
                diag_abs = torch.zeros(d, dtype=self.dtype)

            take = min(b, n_samples - n_used)
            if take < b:
                # need to slice — but feature_input returns the full leaf; we
                # just process only `take` samples below by indexing the
                # gradient outputs. easiest: re-run with a sliced batch.
                if isinstance(batch, (list, tuple)):
                    sliced = [c[:take] for c in batch]
                else:
                    sliced = batch[:take]
                x = adapter.feature_input(sliced)
                batch_for_scalar = sliced
                b = take
            else:
                batch_for_scalar = batch

            # cast x to target dtype if needed
            if x.dtype != target_dtype:
                x = x.detach().to(target_dtype).requires_grad_(True)

            out = adapter.scalar_output(x, batch_for_scalar)
            try:
                grad = torch.autograd.grad(out, x, create_graph=True)[0]
            except RuntimeError:
                # output independent of x — all Hessian rows are zero.
                if row_l1 is None:
                    row_l1 = torch.zeros(d, dtype=self.dtype)
                    diag_abs = torch.zeros(d, dtype=self.dtype)
                n_used += b
                continue
            grad_flat = grad.reshape(b, -1)

            # Streaming-median: when the full (B, b, d) tensor would blow the
            # memory budget, accumulate probe results on CPU in float32 and
            # take the median at the end. For modest d this is a no-op fast
            # path that keeps everything on-device.
            bytes_per = 4  # float32
            full_bytes = B * b * d * bytes_per
            full_mb = full_bytes / 1e6
            stream = full_mb > self.memory_budget_mb

            if stream:
                # CPU pool of per-probe |Hz| absolute values, one row at a time.
                hvp_pool = torch.empty((B, b, d), dtype=torch.float32, device="cpu")
                for k in range(B):
                    z = torch.distributions.Cauchy(
                        torch.zeros((), dtype=target_dtype),
                        torch.ones((), dtype=target_dtype),
                    ).sample((b, d)).clamp_(-self.clamp, self.clamp) \
                     .to(grad_flat.dtype).to(device)
                    gv = (grad_flat * z).sum()
                    hk = torch.autograd.grad(gv, x, retain_graph=True,
                                              allow_unused=True)[0]
                    if hk is None:
                        continue
                    hvp_pool[k] = hk.reshape(b, -1).abs().detach().cpu().float()
                    del z, hk
                chunk_row_l1 = hvp_pool.median(dim=0).values  # (b, d) on CPU
                row_l1 += chunk_row_l1.sum(dim=0).to(self.dtype)
                del hvp_pool
            else:
                probes = torch.distributions.Cauchy(
                    torch.zeros((), dtype=target_dtype),
                    torch.ones((), dtype=target_dtype),
                ).sample((B, b, d))
                probes = probes.clamp_(-self.clamp, self.clamp).to(grad_flat.dtype).to(device)
                hvp_abs = torch.zeros((B, b, d), dtype=grad_flat.dtype, device=device)
                for k in range(B):
                    gv = (grad_flat * probes[k]).sum()
                    hk = torch.autograd.grad(gv, x, retain_graph=True,
                                              allow_unused=True)[0]
                    if hk is not None:
                        hvp_abs[k] = hk.reshape(b, -1).abs()
                chunk_row_l1 = hvp_abs.median(dim=0).values  # (b, d)
                row_l1 += chunk_row_l1.sum(dim=0).detach().cpu().to(self.dtype)
                del probes, hvp_abs

            rad = (torch.randint(0, 2, (B, b, d), generator=gen,
                                 dtype=torch.int64) * 2 - 1).to(grad_flat.dtype).to(device)
            rad_diag = torch.zeros((b, d), dtype=grad_flat.dtype, device=device)
            for k in range(B):
                gv = (grad_flat * rad[k]).sum()
                hk = torch.autograd.grad(gv, x, retain_graph=(k < B - 1),
                                          allow_unused=True)[0]
                if hk is not None:
                    rad_diag += rad[k] * hk.reshape(b, -1)
            rad_diag /= B
            diag_abs += rad_diag.abs().sum(dim=0).detach().cpu().to(self.dtype)

            n_used += b
            del grad, grad_flat, rad, rad_diag

        if n_used == 0:
            raise RuntimeError("loader produced no batches")
        row_l1_np = (row_l1 / n_used).numpy()
        diag_np = (diag_abs / n_used).numpy()
        interactions = np.maximum(row_l1_np - diag_np, 0.0)
        se = (math.pi / 2) * row_l1_np / max(math.sqrt(B), 1.0)
        ci_lower = np.maximum(interactions - 1.96 * se, 0.0)
        ci_upper = interactions + 1.96 * se

        self.results = {
            "method": "cauchy_hvp_real",
            "interactions": interactions,
            "row_l1": row_l1_np,
            "diag_abs": diag_np,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_probes": B,
            "n_samples": n_used,
        }
        if was_training:
            adapter.model.train()
        return interactions, np.column_stack([ci_lower, ci_upper])

    # -------------------------------------------------- correlation-proxy fallback
    def estimate_from_corr(self, gradient_correlation_matrix: np.ndarray):
        corr = np.abs(np.asarray(gradient_correlation_matrix, dtype=np.float64))
        np.fill_diagonal(corr, 0.0)
        interactions = corr.sum(axis=1)
        se = (math.pi / 2) * (interactions + 1e-8) / max(math.sqrt(self.n_probes), 1.0)
        ci_lower = np.maximum(interactions - 1.96 * se, 0.0)
        ci_upper = interactions + 1.96 * se
        self.results = {
            "method": "correlation_proxy",
            "interactions": interactions,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_probes": self.n_probes,
        }
        return interactions, np.column_stack([ci_lower, ci_upper])

    def speedup_factor(self, d: int) -> float:
        return float(d) / float(1 + self.n_probes)
