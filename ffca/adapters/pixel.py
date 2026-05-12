"""PixelAdapter — input-level FFCA on image/tensor inputs.

Use for: any model that consumes a (C, H, W) image and you want pixel-level
explanations. The feature axis is flattened C·H·W pixels.

Adds an `fbr()` helper to compute the Foreground/Background interaction
Ratio used in shortcut-learning diagnostics (e.g. Waterbirds): proportion
of total interaction concentrated in a center foreground box vs the
surrounding background ring. FBR < 0.5 suggests background-shortcut.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..core.adapter import FFCAModelAdapter
from ..core.scalars import ScalarFn, predicted_class


class PixelAdapter(FFCAModelAdapter):
    def __init__(
        self,
        model: nn.Module,
        *,
        input_shape: tuple[int, int, int],  # (C, H, W)
        scalar: ScalarFn | None = None,
    ):
        super().__init__(model, scalar=scalar or predicted_class())
        self.feature_shape = tuple(input_shape)
        self.n_features = int(np.prod(input_shape))
        C, H, W = input_shape
        # Don't name every pixel by default (too many); CLI uses indices
        self.feature_names = [f"px_{c}_{h}_{w}" for c in range(C)
                              for h in range(H) for w in range(W)]

    def feature_input(self, batch) -> torch.Tensor:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device=self.device(), dtype=self.dtype())
        return x.clone().detach().requires_grad_(True)

    def scalar_output(self, x: torch.Tensor, batch) -> torch.Tensor:
        out = self.model(x)
        return self._scalar(out, batch)

    # --- spatial helpers ---------------------------------------------------
    def reshape_to_image(self, per_pixel_score: np.ndarray) -> np.ndarray:
        """Map a (d,) per-pixel score back to its (C, H, W) layout."""
        return per_pixel_score.reshape(self.feature_shape)

    def fbr(self, per_pixel_interaction: np.ndarray, fg_frac: float = 0.5) -> float:
        """Foreground/Background interaction ratio.

        fg_frac: side-length fraction defining the center foreground box.
        Returns mean(fg) / (mean(fg) + mean(bg)). Values < 0.5 hint at a
        background shortcut.
        """
        C, H, W = self.feature_shape
        img = per_pixel_interaction.reshape(self.feature_shape)
        py = int(H * (1 - fg_frac) / 2)
        px = int(W * (1 - fg_frac) / 2)
        fg = img[:, py:H - py, px:W - px].mean()
        bg_sum = img.sum() - img[:, py:H - py, px:W - px].sum()
        bg_count = img.size - img[:, py:H - py, px:W - px].size
        bg = bg_sum / max(bg_count, 1)
        return float(fg / (fg + bg)) if (fg + bg) > 0 else float("nan")
