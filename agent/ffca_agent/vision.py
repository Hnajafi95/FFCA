"""Vision-rule companion: per-epoch FBR / COM-distance / minority-accuracy curves.

The FFCA package's `ChannelAdapter` and `PixelAdapter` produce per-channel /
per-pixel signatures, but the vision-shortcut rule (`shortcut_learning_drift_epoch`,
paper App C.4 / Table 7) needs three additional curves that aren't in the
standard FFCA report:

  - foreground/background attribution ratio (FBR) — how much attribution
    mass lands on the labeled subject vs. its background
  - center-of-mass distance — pixel distance between the attribution centroid
    and the foreground centroid
  - minority-group accuracy — accuracy on the held-out group combinations
    that break the shortcut (e.g., waterbird-on-land, landbird-on-water)

These three are computed offline per checkpoint by the case-study driver and
saved to a vision_metrics.json. This module loads them and makes them
available to the rule evaluator via `ctx.attach_vision_metrics()`.

Compute helpers are also provided here for the case-study driver's
convenience, so the same FBR / COM definitions are used everywhere.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass
class VisionMetrics:
    """Per-epoch curves that unlock the vision-side rules."""

    fbr_curve: np.ndarray | None = None
    com_distance_curve: np.ndarray | None = None
    minority_acc_curve: np.ndarray | None = None
    overall_acc_curve: np.ndarray | None = None

    epoch_labels: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    # ── construction ────────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, d: dict) -> "VisionMetrics":
        """Build from a plain dict (e.g., loaded JSON). Missing curves are kept None."""
        def _maybe_arr(k):
            return np.asarray(d[k], dtype=float) if k in d and d[k] is not None else None

        return cls(
            fbr_curve=_maybe_arr("fbr_curve"),
            com_distance_curve=_maybe_arr("com_distance_curve"),
            minority_acc_curve=_maybe_arr("minority_acc_curve"),
            overall_acc_curve=_maybe_arr("overall_acc_curve"),
            epoch_labels=list(d.get("epoch_labels", [])),
            notes=list(d.get("notes", [])),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "VisionMetrics":
        return cls.from_dict(json.loads(Path(path).read_text()))

    # ── export for the rule evaluator's vision dict ────────────────────────

    def as_signal_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in {
            "fbr_curve": self.fbr_curve,
            "com_distance_curve": self.com_distance_curve,
            "minority_acc_curve": self.minority_acc_curve,
            "overall_acc_curve": self.overall_acc_curve,
        }.items():
            if v is not None:
                out[k] = v
        gap = self.majority_minority_gap_curve()
        if gap is not None:
            out["majority_minority_gap_curve"] = gap
            out["majority_minority_gap_max"] = float(np.max(gap))
        return out

    def majority_minority_gap_curve(self) -> np.ndarray | None:
        """Per-epoch (overall_acc - minority_acc). Larger = stronger shortcut.

        Returns None if either curve is missing — keeps rule evaluation honest
        when only one accuracy series was logged.
        """
        if self.overall_acc_curve is None or self.minority_acc_curve is None:
            return None
        ov = np.asarray(self.overall_acc_curve, dtype=float)
        mn = np.asarray(self.minority_acc_curve, dtype=float)
        n = min(len(ov), len(mn))
        return ov[:n] - mn[:n]

    # ── round-trip to JSON for case-study scripts ──────────────────────────

    def to_dict(self) -> dict:
        return {
            "fbr_curve": _maybe_list(self.fbr_curve),
            "com_distance_curve": _maybe_list(self.com_distance_curve),
            "minority_acc_curve": _maybe_list(self.minority_acc_curve),
            "overall_acc_curve": _maybe_list(self.overall_acc_curve),
            "epoch_labels": list(self.epoch_labels),
            "notes": list(self.notes),
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


def _maybe_list(arr: np.ndarray | None) -> list | None:
    return None if arr is None else [float(x) for x in np.asarray(arr).ravel()]


# ── compute helpers (used by the case-study Waterbirds driver) ─────────────


def compute_fbr(attribution_map: np.ndarray, foreground_mask: np.ndarray) -> float:
    """Foreground/Background attribution Ratio.

    attribution_map: (H, W) non-negative attribution (Grad-CAM-style heatmap).
    foreground_mask: (H, W) binary mask of the labeled subject (1 = foreground).
    Returns mean(attribution[fg]) / mean(attribution[bg]). High = model attends
    to the labeled subject; low = model is shortcutting on background.
    """
    fg = np.asarray(foreground_mask, dtype=bool)
    a = np.asarray(attribution_map, dtype=float)
    fg_mass = a[fg].mean() if fg.any() else 0.0
    bg_mass = a[~fg].mean() if (~fg).any() else 1e-12
    return float(fg_mass / max(bg_mass, 1e-12))


def compute_com_distance(
    attribution_map: np.ndarray, foreground_mask: np.ndarray
) -> float:
    """Pixel distance between the attribution centroid and foreground centroid.

    Normalized by image diagonal length so 0 = perfect alignment, 1 = corner-
    to-corner mismatch. The rule's spike detector then looks for growth in
    this normalized distance across epochs.
    """
    a = np.asarray(attribution_map, dtype=float)
    fg = np.asarray(foreground_mask, dtype=bool)
    h, w = a.shape
    diag = float(np.sqrt(h * h + w * w))

    yy, xx = np.mgrid[0:h, 0:w]
    a_total = a.sum() or 1e-12
    attr_y = (yy * a).sum() / a_total
    attr_x = (xx * a).sum() / a_total

    if not fg.any():
        return 0.0
    fg_total = fg.sum()
    fg_y = (yy * fg).sum() / fg_total
    fg_x = (xx * fg).sum() / fg_total

    d = float(np.sqrt((attr_y - fg_y) ** 2 + (attr_x - fg_x) ** 2))
    return d / max(diag, 1e-12)


def compute_minority_acc(
    predictions: Iterable[int],
    labels: Iterable[int],
    group_ids: Iterable[int],
    minority_groups: tuple[int, ...] = (1, 2),
) -> float:
    """Accuracy restricted to minority groups (where the shortcut fails).

    On Waterbirds the groups are usually:
      0 = landbird on land  (majority — shortcut works)
      1 = landbird on water (minority — shortcut breaks)
      2 = waterbird on land (minority — shortcut breaks)
      3 = waterbird on water (majority — shortcut works)

    Pass the integer IDs that count as minority in this dataset.
    """
    preds = np.asarray(list(predictions))
    labs = np.asarray(list(labels))
    grps = np.asarray(list(group_ids))
    mask = np.isin(grps, minority_groups)
    if not mask.any():
        return 0.0
    return float((preds[mask] == labs[mask]).mean())
