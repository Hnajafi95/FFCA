"""Compute the four FFCA dimensions from an adapter + a data iterable.

Impact         = E[|∂f/∂x_i|]
Volatility     = Var[∂f/∂x_i]
Non-linearity  = E[|∂²f/∂x_i²|]    (Hutchinson / Rademacher diagonal estimator)
Interaction    = handled separately by improvements.cauchy_hvp

This module is purely first/second-order derivative bookkeeping; the
adapter handles model heterogeneity.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

from .adapter import FFCAModelAdapter
from .smoothing import smooth


def compute_first_order(
    adapter: FFCAModelAdapter,
    loader: Iterable,
    n_samples: int = 64,
    batch_chunk: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (impact, volatility, raw_gradients).

    raw_gradients (n_samples, d) is kept around because Co-Sensitivity wants
    the per-sample gradient matrix.
    """
    grads = []
    samples_seen = 0
    for batch in loader:
        if samples_seen >= n_samples:
            break
        x = adapter.feature_input(batch)
        out = adapter.scalar_output(x, batch)
        g = torch.autograd.grad(out, x)[0]
        g_flat = g.reshape(g.size(0), -1).detach().cpu().numpy().astype(np.float64)
        take = min(g_flat.shape[0], n_samples - samples_seen)
        grads.append(g_flat[:take])
        samples_seen += take

    if not grads:
        raise RuntimeError("loader produced no batches")
    all_grads = np.vstack(grads)
    impact = np.abs(all_grads).mean(axis=0)
    volatility = all_grads.var(axis=0)
    return impact, volatility, all_grads


def compute_diag_hessian(
    adapter: FFCAModelAdapter,
    loader: Iterable,
    n_samples: int = 32,
    n_probes: int = 64,
    seed: int = 0,
) -> np.ndarray:
    """Diagonal Hessian E[|H_ii|] via Hutchinson (Rademacher) probes.

    Cost: (1 + n_probes) backward passes per sample chunk.
    For each Rademacher probe r ∈ {±1}^d: E[r_i (H r)_i] = H_ii.
    """
    diag_accum = None
    n_used = 0
    gen = torch.Generator(device="cpu").manual_seed(seed)
    device = adapter.device()

    samples_seen = 0
    for batch in loader:
        if samples_seen >= n_samples:
            break
        x = adapter.feature_input(batch)
        b = x.size(0)
        d = int(np.prod(x.shape[1:]))
        out = adapter.scalar_output(x, batch)
        try:
            grad = torch.autograd.grad(out, x, create_graph=True)[0]
        except RuntimeError as exc:
            # output is constant in x (e.g. ChannelAdapter spliced where only
            # linear ops follow). Hessian-diag is identically zero. Other
            # RuntimeErrors (OOM, non-differentiable backward, graph reuse)
            # are real failures and should surface.
            msg = str(exc).lower()
            if "does not require grad" not in msg and "differentiable" not in msg:
                raise
            diag_zero = np.zeros((b, d), dtype=np.float64)
            diag = diag_zero
            take = min(b, n_samples - samples_seen)
            if diag_accum is None:
                diag_accum = np.zeros(d, dtype=np.float64)
            diag_accum += diag[:take].sum(axis=0)
            n_used += take
            samples_seen += take
            continue
        grad_flat = grad.reshape(b, -1)

        rad = (torch.randint(0, 2, (n_probes, b, d), generator=gen,
                             dtype=torch.int64) * 2 - 1).to(grad_flat.dtype).to(device)
        accum = torch.zeros((b, d), dtype=grad_flat.dtype, device=device)
        for k in range(n_probes):
            gv = (grad_flat * rad[k]).sum()
            hk = torch.autograd.grad(gv, x, retain_graph=(k < n_probes - 1),
                                     allow_unused=True)[0]
            if hk is None:
                continue
            accum += rad[k] * hk.reshape(b, -1)
        accum /= n_probes
        diag = accum.abs().detach().cpu().numpy()

        take = min(b, n_samples - samples_seen)
        diag = diag[:take]
        if diag_accum is None:
            diag_accum = np.zeros(d, dtype=np.float64)
        diag_accum += diag.sum(axis=0)
        n_used += take
        samples_seen += take

    if diag_accum is None:
        raise RuntimeError("no samples consumed")
    return diag_accum / max(n_used, 1)


def compute_signature_core(
    adapter: FFCAModelAdapter,
    loader: Iterable,
    n_first_order_samples: int = 64,
    n_hessian_samples: int = 16,
    n_diag_probes: int = 48,
    enable_smooth: bool = True,
    beta: float = 10.0,
):
    """Returns (impact, volatility, nonlinearity, gradients_matrix).

    Interaction is *not* computed here — it's an improvement (cauchy_hvp).
    """
    ctx = smooth(adapter.model, beta=beta) if enable_smooth else _nullctx(adapter.model)
    with ctx:
        adapter.model.eval()
        impact, volatility, gradients = compute_first_order(
            adapter, loader, n_samples=n_first_order_samples
        )
        nonlinearity = compute_diag_hessian(
            adapter, loader, n_samples=n_hessian_samples,
            n_probes=n_diag_probes,
        )
    return impact, volatility, nonlinearity, gradients


from contextlib import contextmanager


@contextmanager
def _nullctx(obj):
    yield obj
