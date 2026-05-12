"""TabularAdapter — input-level FFCA on dense feature vectors.

Use for: MLPs, tabular transformers, scikit-style wrappers over an MLP.
The feature axis is the d input columns.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..core.adapter import FFCAModelAdapter
from ..core.scalars import ScalarFn, predicted_class


class TabularAdapter(FFCAModelAdapter):
    def __init__(
        self,
        model: nn.Module,
        *,
        feature_names: list[str] | None = None,
        n_features: int | None = None,
        scalar: ScalarFn | None = None,
    ):
        super().__init__(model, scalar=scalar or predicted_class())
        if feature_names is None and n_features is None:
            raise ValueError("provide feature_names or n_features")
        self.feature_names = feature_names
        self.n_features = n_features if n_features is not None else len(feature_names)
        self.feature_shape = (self.n_features,)
        if feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(self.n_features)]

    def feature_input(self, batch) -> torch.Tensor:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device=self.device(), dtype=self.dtype())
        return x.clone().detach().requires_grad_(True)

    def scalar_output(self, x: torch.Tensor, batch) -> torch.Tensor:
        out = self.model(x)
        return self._scalar(out, batch)
