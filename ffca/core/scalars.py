"""Scalar-output factories.

A scalar function maps the model's full output tensor to a single number
for differentiation. FFCA's Impact/Volatility/Non-linearity/Interaction
are all derivatives of *this* scalar w.r.t. the feature axis.

The scalar choice is task-specific:
  - Classification: a single logit (predicted class, target class, true label)
  - Regression: a single output dimension, or a loss
  - Generative: log-likelihood

Each factory returns a callable `(out, batch) -> scalar` where `out` is
whatever `adapter.feature_input → model → out` produces and `batch` is the
original DataLoader yield (used when the scalar needs ground-truth labels).
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F


ScalarFn = Callable[[torch.Tensor, object], torch.Tensor]


def predicted_class() -> ScalarFn:
    """Sum of the predicted-class logit across samples. Default for classifiers."""
    def _fn(out: torch.Tensor, batch=None) -> torch.Tensor:
        if out.dim() == 1 or out.size(-1) == 1:
            return out.sum()
        pred = out.argmax(dim=-1)
        return out.gather(-1, pred.unsqueeze(-1)).sum()
    return _fn


def target_class(class_idx: int) -> ScalarFn:
    """Sum of a specific class's logit across samples."""
    def _fn(out: torch.Tensor, batch=None) -> torch.Tensor:
        if out.dim() == 1:
            raise ValueError("target_class needs a multi-class output")
        return out[..., class_idx].sum()
    return _fn


def true_label() -> ScalarFn:
    """Sum of the *ground-truth* class logit. Requires labels in the batch."""
    def _fn(out: torch.Tensor, batch) -> torch.Tensor:
        if not isinstance(batch, (list, tuple)) or len(batch) < 2:
            raise ValueError("true_label needs (inputs, labels[, ...]) batches")
        labels = batch[1].to(out.device)
        return out.gather(-1, labels.unsqueeze(-1)).sum()
    return _fn


def regression(dim: int | None = None) -> ScalarFn:
    """Sum of (optionally a single dim of) the regression output."""
    def _fn(out: torch.Tensor, batch=None) -> torch.Tensor:
        if dim is None:
            return out.sum()
        return out[..., dim].sum()
    return _fn


def loss(loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | str = "ce") -> ScalarFn:
    """Sum of a loss against the labels in the batch.

    loss_fn: "ce" / "mse" / "bce" or a custom callable (logits, targets) -> loss.
    """
    if isinstance(loss_fn, str):
        loss_fn = {
            "ce": F.cross_entropy,
            "mse": F.mse_loss,
            "bce": F.binary_cross_entropy_with_logits,
        }[loss_fn]

    def _fn(out: torch.Tensor, batch) -> torch.Tensor:
        if not isinstance(batch, (list, tuple)) or len(batch) < 2:
            raise ValueError("loss() scalar needs (inputs, targets[, ...])")
        targets = batch[1].to(out.device)
        return loss_fn(out, targets)
    return _fn


def custom(fn: Callable[[torch.Tensor, object], torch.Tensor]) -> ScalarFn:
    """Pass-through — wrap any user callable so it satisfies the ScalarFn protocol."""
    return fn


_REGISTRY = {
    "predicted_class": predicted_class,
    "true_label": true_label,
    "regression": regression,
}


def from_name(name: str) -> ScalarFn:
    """CLI convenience: look up a scalar by short name."""
    if name in _REGISTRY:
        return _REGISTRY[name]()
    if name.startswith("target_class:"):
        return target_class(int(name.split(":", 1)[1]))
    if name == "loss":
        return loss("ce")
    if name.startswith("loss:"):
        return loss(name.split(":", 1)[1])
    raise ValueError(
        f"unknown scalar {name!r}. Known: predicted_class, true_label, "
        f"target_class:N, regression, loss, loss:ce|mse|bce"
    )
