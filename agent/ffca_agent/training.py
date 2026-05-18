"""Optional companion data for an FFCA report: Keras-style training history.

Attaches val/train score curves and signature-derived per-epoch curves to a
ReportContext so the rulebook's `training.*` signals can resolve. When the
caller has no history file the dynamic rules silently skip — the evaluator
treats MissingSignal as "did not fire".

Two use modes:

  history = TrainingHistory.from_keras_history("history.json")
  history.derive_from_signatures(ctx, top_k=5)  # optional, only if ckpts==epochs
  ctx.attach_training_history(history)

or, when only the FFCA report exists and the user asserts ckpts==epochs:

  history = TrainingHistory()
  history.derive_from_signatures(ctx, top_k=5)
  ctx.attach_training_history(history)  # scalars stay None → leakage rule etc. skip
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


_LOSS_HINTS = ("loss", "error", "mse", "mae", "rmse")


@dataclass
class TrainingHistory:
    val_score_final: float | None = None
    train_score_final: float | None = None
    val_train_gap: float | None = None

    val_score_curve: np.ndarray | None = None
    train_score_curve: np.ndarray | None = None

    volatility_curve: np.ndarray | None = None
    impact_curve_topk_mean: np.ndarray | None = None
    interaction_curve_topk_mean: np.ndarray | None = None

    metric_name: str = "val_loss"
    lower_is_better: bool = True
    notes: list[str] = field(default_factory=list)

    # ── construction ────────────────────────────────────────────────────────

    @classmethod
    def from_keras_history(cls, path: str | Path, metric: str | None = None) -> "TrainingHistory":
        """Parse a Keras history.json or .csv.

        Accepts (auto-detected):
          - JSON: `{"loss": [...], "val_loss": [...], ...}`  (raw History.history)
          - JSON: `{"history": {...}}`                       (some wrappers nest it)
          - CSV : columns include `loss` + `val_loss` (or any `metric` + `val_metric`)

        `metric` lets the caller override auto-detect (e.g., `metric="accuracy"`).
        """
        path = Path(path)
        if path.suffix.lower() in (".csv", ".tsv"):
            d = _load_csv(path)
        else:
            d = _load_json(path)
        return cls.from_dict(d, metric=metric)

    @classmethod
    def from_dict(cls, d: dict, metric: str | None = None) -> "TrainingHistory":
        if "history" in d and isinstance(d["history"], dict):
            d = d["history"]
        train_key, val_key, lower_is_better = _pick_metric(d, metric)
        val_curve = _to_array(d[val_key])
        train_curve = _to_array(d[train_key]) if train_key in d else None

        final_val = float(val_curve[-1]) if val_curve.size else None
        final_train = float(train_curve[-1]) if train_curve is not None and train_curve.size else None
        gap = _compute_gap(final_train, final_val, lower_is_better)

        return cls(
            val_score_final=final_val,
            train_score_final=final_train,
            val_train_gap=gap,
            val_score_curve=val_curve,
            train_score_curve=train_curve,
            metric_name=val_key,
            lower_is_better=lower_is_better,
        )

    # ── enrich from FFCA signatures ────────────────────────────────────────

    def derive_from_signatures(self, ctx, top_k: int = 5) -> "TrainingHistory":
        """Fill volatility / top-k impact / top-k interaction curves from the
        FFCA checkpoint signatures. Caller must assert checkpoints==epochs.
        Existing fields are not overwritten."""
        if ctx.impact_curve.shape[0] < 2:
            self.notes.append("derive_from_signatures: <2 checkpoints, skipped")
            return self
        if self.volatility_curve is None:
            self.volatility_curve = ctx.volatility_curve.mean(axis=1)
        final_impact = ctx.impact
        top_idx = np.argsort(-final_impact)[: min(top_k, len(final_impact))]
        if self.impact_curve_topk_mean is None:
            self.impact_curve_topk_mean = ctx.impact_curve[:, top_idx].mean(axis=1)
        if self.interaction_curve_topk_mean is None:
            self.interaction_curve_topk_mean = ctx.interaction_curve[:, top_idx].mean(axis=1)
        return self

    # ── export as the training-dict the evaluator dispatches against ───────

    def as_signal_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in {
            "val_score_final": self.val_score_final,
            "train_score_final": self.train_score_final,
            "val_train_gap": self.val_train_gap,
            "val_score_curve": self.val_score_curve,
            "train_score_curve": self.train_score_curve,
            "volatility_curve": self.volatility_curve,
            "impact_curve_topk_mean": self.impact_curve_topk_mean,
            "interaction_curve_topk_mean": self.interaction_curve_topk_mean,
        }.items():
            if v is not None:
                out[k] = v
        return out


# ── helpers ───────────────────────────────────────────────────────────────


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_csv(path: Path) -> dict[str, list[float]]:
    with path.open() as f:
        reader = csv.DictReader(f)
        cols: dict[str, list[float]] = {k: [] for k in (reader.fieldnames or [])}
        for row in reader:
            for k, v in row.items():
                if v in (None, "", "NA", "nan"):
                    continue
                try:
                    cols[k].append(float(v))
                except ValueError:
                    pass
    return {k: v for k, v in cols.items() if v}


def _pick_metric(d: dict, override: str | None) -> tuple[str, str, bool]:
    """Choose (train_key, val_key, lower_is_better) from the history dict."""
    val_keys = [k for k in d if k.startswith("val_")]
    if not val_keys:
        raise ValueError(
            f"no `val_*` column in history dict; available keys: {sorted(d.keys())}"
        )
    if override is not None:
        train_key = override
        val_key = f"val_{override}" if not override.startswith("val_") else override
        if val_key not in d:
            raise ValueError(f"requested metric `{val_key}` not in history")
    elif "val_loss" in val_keys:
        val_key, train_key = "val_loss", "loss"
    else:
        val_key = val_keys[0]
        train_key = val_key[4:]
    lower_is_better = any(h in val_key.lower() for h in _LOSS_HINTS)
    return train_key, val_key, lower_is_better


def _to_array(seq) -> np.ndarray:
    return np.asarray(seq, dtype=float)


def _compute_gap(final_train: float | None, final_val: float | None, lower_is_better: bool) -> float | None:
    """Generalization gap as a normalized absolute difference.

    For losses (lower-is-better): val - train, expected positive when overfitting.
    For accuracies (higher-is-better): train - val, expected positive when overfitting.
    Normalized by |train| (or |val| if train≈0). Returns the magnitude.
    """
    if final_train is None or final_val is None:
        return None
    denom = max(abs(final_train), abs(final_val), 1e-12)
    raw = (final_val - final_train) if lower_is_better else (final_train - final_val)
    return abs(raw / denom)
