"""FFCASignature — container for one checkpoint's 4-D FFCA result."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np


@dataclass
class FFCASignature:
    """One checkpoint's worth of FFCA results.

    The four axes are the FFCA paper's definitions:
      Impact (I)        = E[|∂f/∂x_i|]                — gradient magnitude
      Volatility (V)    = Var[∂f/∂x_i]                — gradient instability
      Non-linearity (N) = E[|∂²f/∂x_i²|]              — diagonal Hessian
      Interaction (X)   = Σ_j≠i E[|∂²f/∂x_i ∂x_j|]    — off-diagonal Hessian row L1
    """
    impact: np.ndarray
    volatility: np.ndarray
    nonlinearity: np.ndarray
    interaction: np.ndarray
    feature_names: list[str]
    archetypes: np.ndarray | None = None   # int per feature, 0–7
    interaction_ci: np.ndarray | None = None  # (d, 2)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        for name in ("impact", "volatility", "nonlinearity", "interaction"):
            arr = np.asarray(getattr(self, name), dtype=np.float64)
            setattr(self, name, arr)
        d = self.impact.size
        if len(self.feature_names) != d:
            raise ValueError(
                f"feature_names length {len(self.feature_names)} != d {d}"
            )
        for name in ("volatility", "nonlinearity", "interaction"):
            if getattr(self, name).size != d:
                raise ValueError(f"{name} size mismatch with impact (d={d})")

    @property
    def n_features(self) -> int:
        return self.impact.size

    def stack4(self) -> np.ndarray:
        """Return the (d, 4) matrix [I, V, N, X]."""
        return np.column_stack([
            self.impact, self.volatility, self.nonlinearity, self.interaction,
        ])

    def top_k(self, k: int = 10, by: str = "impact") -> np.ndarray:
        """Return indices of the top-k features by the requested axis."""
        return np.argsort(getattr(self, by))[-k:][::-1]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
        return d

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FFCASignature":
        kw = dict(payload)
        for k in ("impact", "volatility", "nonlinearity", "interaction"):
            kw[k] = np.asarray(kw[k])
        if kw.get("archetypes") is not None:
            kw["archetypes"] = np.asarray(kw["archetypes"], dtype=int)
        if kw.get("interaction_ci") is not None:
            kw["interaction_ci"] = np.asarray(kw["interaction_ci"])
        return cls(**kw)
