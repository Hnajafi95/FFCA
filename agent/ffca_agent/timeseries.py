"""Time-series operators referenced by the rulebook.

The FFCA paper uses qualitative descriptions ("sharp sustained spike",
"plateau", "collapse"). These functions codify operational tests using
ratio-based thresholds. Defaults are calibrated to the paper's worked
examples (e.g., Bike Sharing epoch ~120 overfitting in App C.3 Fig 9).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass
class TimeSeriesResult:
    fired: bool
    epoch: int | None = None
    ratio: float | None = None
    note: str = ""


def _as_array(curve: Iterable[float]) -> np.ndarray:
    return np.asarray(list(curve), dtype=float)


def spike_detected(
    curve: Sequence[float],
    threshold_ratio: float = 1.3,
    when: str = "any_epoch",
    baseline_fraction: float = 0.15,
    window_fraction: float = 0.4,
) -> TimeSeriesResult:
    """Returns fired=True if the curve exhibits a sustained rise.

    The baseline is fixed to the first `baseline_fraction` of epochs (the
    "before-anything-happens" portion). The peak is then searched in the
    epochs after that baseline, and its position relative to total length
    determines whether it counts as early/late/any.

    Conventions for `when`:
      - "epoch_0"      : value at epoch 0 exceeds threshold_ratio × median of
                         the remaining epochs (data-leakage fingerprint).
      - "early_epochs" : peak occurs in the first `window_fraction` of training.
      - "late_epochs"  : peak occurs in the last `window_fraction` of training.
      - "any_epoch"    : peak anywhere; only the ratio matters.
    """
    x = _as_array(curve)
    if x.size < 3:
        return TimeSeriesResult(False, note="curve too short")

    if when == "epoch_0":
        rest_median = float(np.median(x[1:])) or 1e-12
        ratio = float(x[0]) / rest_median
        return TimeSeriesResult(ratio > threshold_ratio, epoch=0, ratio=ratio)

    baseline_n = max(2, int(round(x.size * baseline_fraction)))
    baseline_median = float(np.median(x[:baseline_n])) or 1e-12

    search = x[baseline_n:]
    if search.size == 0:
        return TimeSeriesResult(False, note="no epochs after baseline")
    peak_relative = int(np.argmax(search))
    peak_epoch = baseline_n + peak_relative
    ratio = float(search[peak_relative]) / baseline_median

    window_n = max(2, int(round(x.size * window_fraction)))
    if when == "early_epochs":
        in_window = peak_epoch < window_n
    elif when == "late_epochs":
        in_window = peak_epoch >= x.size - window_n
    elif when == "any_epoch":
        in_window = True
    else:
        raise ValueError(f"unknown 'when' value: {when!r}")

    return TimeSeriesResult(
        in_window and ratio > threshold_ratio,
        epoch=peak_epoch,
        ratio=ratio,
        note=f"peak={float(search[peak_relative]):.4f}, baseline_median={baseline_median:.4f}",
    )


def plateau_detected(
    curve: Sequence[float],
    relative_improvement: float = 0.01,
    tail_fraction: float = 0.3,
) -> TimeSeriesResult:
    """Returns fired=True if the curve's last tail_fraction improves by less
    than relative_improvement compared to the start of that tail.
    """
    x = _as_array(curve)
    if x.size < 5:
        return TimeSeriesResult(False, note="curve too short")
    tail_n = max(3, int(round(x.size * tail_fraction)))
    tail = x[-tail_n:]
    start, end = float(tail[0]), float(tail[-1])
    if abs(start) < 1e-12:
        return TimeSeriesResult(False, note="tail start ≈ 0")
    rel = (end - start) / abs(start)
    return TimeSeriesResult(
        abs(rel) < relative_improvement,
        epoch=x.size - tail_n,
        ratio=rel,
        note=f"tail start={start:.4f}, end={end:.4f}, rel_change={rel:+.3f}",
    )


def collapse_detected(
    curve: Sequence[float],
    final_threshold: float,
    relative_to: str = "absolute",
) -> TimeSeriesResult:
    """Returns fired=True if the final value drops below the threshold.

    relative_to:
      - "absolute" : final value < final_threshold
      - "initial"  : final value < final_threshold * initial value (i.e., the
                     curve has retained less than final_threshold of its start)
    """
    x = _as_array(curve)
    if x.size < 2:
        return TimeSeriesResult(False, note="curve too short")
    final = float(x[-1])
    if relative_to == "absolute":
        return TimeSeriesResult(
            final < final_threshold,
            epoch=x.size - 1,
            ratio=final,
            note=f"final={final:.4f} < {final_threshold}",
        )
    if relative_to == "initial":
        initial = float(x[0]) or 1e-12
        ratio = final / initial
        return TimeSeriesResult(
            ratio < final_threshold,
            epoch=x.size - 1,
            ratio=ratio,
            note=f"final/initial={ratio:.3f} < {final_threshold}",
        )
    raise ValueError(f"unknown relative_to: {relative_to!r}")
