"""ChannelAdapter — channel-level FFCA at any intermediate layer.

Splice mechanism:
  1. Resolve the named layer fresh on every forward (smoothing may have
     swapped its identity).
  2. Capture: register a one-shot forward_hook that records the activation
     and unhooks itself.
  3. Replace: register a hook that returns a pre-set leaf tensor as the
     layer's output, so subsequent layers run from that leaf.

Both modes resolve the layer by *name* every time, so even if `smooth()`
replaces ReLU→Softplus mid-pipeline, the hook still attaches to the
right (current) instance.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..core.adapter import FFCAModelAdapter, find_layer
from ..core.scalars import ScalarFn, predicted_class


class ChannelAdapter(FFCAModelAdapter):
    def __init__(
        self,
        model: nn.Module,
        layer_name: str,
        *,
        scalar: ScalarFn | None = None,
        reduction: str = "spatial_mean",
    ):
        super().__init__(model, scalar=scalar or predicted_class())
        self.layer_name = layer_name
        # Verify the layer exists; we re-resolve each call.
        _ = find_layer(model, layer_name)
        self.reduction = reduction
        self._activation_shape: tuple[int, ...] | None = None
        self.feature_names: list[str] | None = None
        self.n_features = -1
        self.feature_shape = ()
        self._raw_input: torch.Tensor | None = None

    # ---------------------------------------------------------- splice utilities
    def _capture_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass, capture the named layer's activation, return it."""
        captured: list[torch.Tensor] = []
        layer = find_layer(self.model, self.layer_name)

        def hook(module, inputs, output):
            captured.append(output.detach())

        handle = layer.register_forward_hook(hook)
        try:
            with torch.no_grad():
                self.model(x)
        finally:
            handle.remove()
        if not captured:
            raise RuntimeError(
                f"Layer {self.layer_name!r} was not invoked during forward; "
                f"check the layer name is correct for this model's forward path"
            )
        return captured[0]

    def _forward_with_replacement(self, x: torch.Tensor,
                                   replacement: torch.Tensor) -> torch.Tensor:
        """Run a forward pass with the named layer's output replaced."""
        layer = find_layer(self.model, self.layer_name)

        def hook(module, inputs, output):
            return replacement

        handle = layer.register_forward_hook(hook)
        try:
            return self.model(x)
        finally:
            handle.remove()

    # ---------------------------------------------------------- shape probe
    def _probe(self, batch):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device=self.device(), dtype=self.dtype())
        act = self._capture_activation(x)
        self._activation_shape = tuple(act.shape[1:])
        if self.reduction == "spatial_mean":
            C = act.shape[1]
            self.n_features = C
            self.feature_shape = (C,)
            self.feature_names = [f"ch_{i}" for i in range(C)]
        elif self.reduction == "none":
            d = int(np.prod(act.shape[1:]))
            self.n_features = d
            self.feature_shape = tuple(act.shape[1:])
            self.feature_names = [f"unit_{i}" for i in range(d)]
        else:
            raise ValueError(f"unknown reduction {self.reduction!r}")

    # ---------------------------------------------------------- adapter API
    def feature_input(self, batch) -> torch.Tensor:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device=self.device(), dtype=self.dtype())

        if self._activation_shape is None:
            self._probe(batch)

        act = self._capture_activation(x)  # full activation tensor

        if self.reduction == "spatial_mean":
            if act.dim() == 4:           # (B, C, H, W)
                feat = act.mean(dim=(2, 3))
            elif act.dim() == 3:         # (B, C, L) or (B, T, C)
                feat = act.mean(dim=-1)
            else:
                feat = act
        else:
            feat = act.reshape(act.size(0), -1)

        leaf = feat.clone().detach().requires_grad_(True)
        self._raw_input = x
        return leaf

    def scalar_output(self, leaf: torch.Tensor, batch) -> torch.Tensor:
        # Re-inflate the leaf to the original activation shape and run the
        # rest of the model with the replacement hook.
        if self.reduction == "spatial_mean":
            shape = self._activation_shape
            if len(shape) == 3:
                C, H, W = shape
                injected = leaf.view(leaf.size(0), C, 1, 1).expand(-1, C, H, W).contiguous()
            elif len(shape) == 2:
                C, L = shape
                injected = leaf.view(leaf.size(0), C, 1).expand(-1, C, L).contiguous()
            else:
                injected = leaf
        else:
            injected = leaf.reshape(leaf.size(0), *self._activation_shape)
        out = self._forward_with_replacement(self._raw_input, injected)
        return self._scalar(out, batch)

    def channel_count(self) -> int:
        if self._activation_shape is None:
            raise RuntimeError("ChannelAdapter.feature_input must be called once first")
        return self._activation_shape[0]
