"""CheckpointLoader — yield a fresh adapter for each saved state_dict.

For v0.1.0 we support plain `torch.save(model.state_dict(), path)` files.
Lightning / HF / SafeTensors formats can be added with detect-by-extension
in v0.2.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator, Sequence

import torch
import torch.nn as nn


class CheckpointLoader:
    """Iterate (epoch_label, model) over a list of saved state_dicts.

    Args:
        model_factory: () -> fresh nn.Module instance (same architecture as
                       what was saved).
        checkpoints:   list of paths (or (label, path) pairs).
        device:        target device for the loaded model.
    """

    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        checkpoints: Sequence[str | tuple[str, str]],
        device: str | torch.device = "cpu",
    ):
        self.model_factory = model_factory
        self.checkpoints = [
            (Path(c).stem, Path(c)) if isinstance(c, str) else (str(c[0]), Path(c[1]))
            for c in checkpoints
        ]
        self.device = torch.device(device)

    def __len__(self) -> int:
        return len(self.checkpoints)

    def __iter__(self) -> Iterator[tuple[str, nn.Module]]:
        for label, path in self.checkpoints:
            model = self.model_factory()
            state = torch.load(path, map_location=self.device, weights_only=False)
            # Accept either {state_dict} or just a state_dict
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            try:
                model.load_state_dict(state, strict=False)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load checkpoint {path}: {e}. "
                    f"Ensure model_factory() returns the same architecture."
                )
            model.to(self.device)
            model.eval()
            yield label, model
