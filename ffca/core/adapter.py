"""FFCAModelAdapter — the universal contract FFCA needs from any model.

Three methods are all that's required:
  - feature_input(batch)  -> Tensor (the variable FFCA differentiates w.r.t.)
  - scalar_output(x)      -> Tensor (a single scalar for autograd)
  - n_features, feature_names, feature_shape (for reporting/plotting)

Built-in adapters in `ffca.adapters` cover tabular, pixel-level, and
intermediate-channel (via the Splice helper below) FFCA. Researchers with
unusual models subclass FFCAModelAdapter and write ~20 lines.

The Splice helper is the one tricky piece: PyTorch's autograd needs a
*leaf* tensor with `requires_grad=True`, but channel/feature activations
live deep inside the computation graph. The Splice installs a
`register_forward_hook` that swaps the live activation for a leaf tensor,
so the rest of the model continues forward from that leaf — and we can
backprop straight to it.
"""
from __future__ import annotations

import abc
from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn as nn


class FFCAModelAdapter(abc.ABC):
    """Universal adapter interface.

    Subclasses must set:
      n_features       — flattened size of feature_input per sample
      feature_shape    — original (un-flattened) shape, e.g. (3, 32, 32)
      feature_names    — optional human-readable names, len == n_features
    """

    n_features: int
    feature_shape: tuple[int, ...]
    feature_names: list[str] | None = None

    def __init__(self, model: nn.Module, *, scalar=None):
        self.model = model
        self._scalar = scalar  # ScalarFn or None for adapter's default

    @abc.abstractmethod
    def feature_input(self, batch) -> torch.Tensor:
        """Return the tensor whose gradient/Hessian we'll compute.

        Must have shape (batch, *feature_shape) and `requires_grad=True`.
        For input-level adapters this is just the model input; for
        channel-level adapters it's a leaf tensor produced by a Splice.
        """

    @abc.abstractmethod
    def scalar_output(self, feature_input: torch.Tensor, batch) -> torch.Tensor:
        """Forward through whatever model code maps feature_input → scalar."""

    def device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def dtype(self) -> torch.dtype:
        try:
            return next(self.model.parameters()).dtype
        except StopIteration:
            return torch.float32


# ---------------------------------------------------------------------------
# Splice — replace an intermediate activation with a differentiable leaf
# ---------------------------------------------------------------------------
class _Splice:
    """Internal: forward-hook that replaces a layer's output with a leaf.

    Usage flow inside ChannelAdapter.feature_input:
        splice = _Splice(layer)
        with splice.capture():
            self.model(input)          # populates splice.activation
        leaf = splice.activation.clone().detach().requires_grad_(True)
        splice.set_replacement(leaf)
        # subsequent forward uses `leaf` as the layer's output

    Then in ChannelAdapter.scalar_output the model is forwarded again with
    `leaf` patched in, so the gradient/Hessian we compute is w.r.t. `leaf`.
    """

    def __init__(self, layer: nn.Module):
        self.layer = layer
        self.activation: torch.Tensor | None = None
        self._replacement: torch.Tensor | None = None
        self._handle: torch.utils.hooks.RemovableHandle | None = None
        self._mode: str = "capture"  # 'capture' | 'replace' | 'off'

    def _hook(self, module, inputs, output):
        if self._mode == "capture":
            self.activation = output.detach()
            return output
        if self._mode == "replace" and self._replacement is not None:
            return self._replacement
        return output

    def install(self):
        if self._handle is None:
            self._handle = self.layer.register_forward_hook(self._hook)
        return self

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def set_replacement(self, tensor: torch.Tensor):
        self._replacement = tensor
        self._mode = "replace"

    @contextmanager
    def capture(self) -> Iterator["_Splice"]:
        prev = self._mode
        self._mode = "capture"
        try:
            yield self
        finally:
            self._mode = prev

    @contextmanager
    def replace(self, tensor: torch.Tensor) -> Iterator["_Splice"]:
        prev_mode, prev_repl = self._mode, self._replacement
        self.set_replacement(tensor)
        try:
            yield self
        finally:
            self._mode, self._replacement = prev_mode, prev_repl


def find_layer(model: nn.Module, name: str) -> nn.Module:
    """Look up a layer by dotted name (e.g. 'layer4.2.conv2').

    Raises KeyError with a helpful list of available names if not found.
    """
    names = dict(model.named_modules())
    if name not in names:
        candidates = [n for n in names if n.endswith(name.split(".")[-1])][:8]
        raise KeyError(
            f"layer {name!r} not found in model. Closest matches: {candidates}. "
            f"Use `list(model.named_modules())` to see all options."
        )
    return names[name]
