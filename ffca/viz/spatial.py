"""Spatial-FFCA plots: pixel maps, channel grids, FBR."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..core.archetypes import ARCHETYPE_NAMES
from ..core.signature import FFCASignature
from .static import ARCHETYPE_COLORS, _save


def _infer_unit_label(feature_names: list[str] | None) -> str:
    """Guess the unit word for plot titles from the feature naming convention."""
    if not feature_names:
        return "feature"
    first = feature_names[0]
    if first.startswith("ch_"): return "channel"
    if first.startswith("px_"): return "pixel"
    if first.startswith("h") and "_d" in first: return "head-dim"
    if first.startswith("t") and "_h" in first: return "embed-dim"
    return "feature"


def pixel_interaction_map(sig: FFCASignature, feature_shape: tuple[int, ...],
                          reference_image: np.ndarray | None = None,
                          out: Path | None = None, formats=("png",)):
    """Reshape per-pixel interaction to (C, H, W) and visualise.

    Reduces over channels (mean) to get a (H, W) heatmap.
    """
    if len(feature_shape) != 3:
        return None
    C, H, W = feature_shape
    inter = sig.interaction.reshape(C, H, W)
    impact = sig.impact.reshape(C, H, W)
    inter_2d = inter.mean(axis=0)
    impact_2d = impact.mean(axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    im0 = axes[0].imshow(impact_2d, cmap="viridis")
    axes[0].set_title("Impact (mean over channels)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(inter_2d, cmap="magma")
    axes[1].set_title("Interaction (mean over channels)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    if reference_image is not None:
        # display reference, overlay interaction at half alpha
        ref = reference_image
        if ref.ndim == 3 and ref.shape[0] in (1, 3):
            ref = np.transpose(ref, (1, 2, 0))
        ref = (ref - ref.min()) / max(ref.max() - ref.min(), 1e-8)
        axes[2].imshow(ref)
        axes[2].imshow(inter_2d, cmap="magma", alpha=0.55)
        axes[2].set_title("Interaction overlay on reference image")
    else:
        axes[2].axis("off")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    if out is not None:
        _save(fig, out, formats)
    return fig


def channel_archetype_grid(sig: FFCASignature, out: Path | None = None,
                           formats=("png",), unit_label: str | None = None):
    if sig.archetypes is None:
        return None
    n = sig.n_features
    side = int(np.ceil(np.sqrt(n)))
    grid = np.full(side * side, -1, dtype=int)
    grid[:n] = sig.archetypes
    grid = grid.reshape(side, side)

    cell = 0.35 if side <= 16 else (0.28 if side <= 24 else 0.22)
    fig, ax = plt.subplots(figsize=(max(7, cell * side + 3),
                                     max(7, cell * side + 1)))
    color_grid = np.ones((side, side, 3))
    for i in range(side):
        for j in range(side):
            a = grid[i, j]
            if a != -1:
                color_grid[i, j] = matplotlib.colors.to_rgb(ARCHETYPE_COLORS[int(a)])
    ax.imshow(color_grid, interpolation="nearest", aspect="equal")

    # Per-cell channel index — only when readable
    if n <= 256:
        fontsize = 9 if side <= 12 else (7 if side <= 16 else 5)
        for k in range(n):
            i, j = k // side, k % side
            # pick text color by luminance for readability
            r, g, b = color_grid[i, j]
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = "white" if luminance < 0.5 else "black"
            ax.text(j, i, str(k), ha="center", va="center",
                    fontsize=fontsize, color=text_color, fontweight="bold")

    # Light grid lines between cells
    ax.set_xticks(np.arange(-0.5, side, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, side, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)
    ax.set_xticks(range(0, side, max(1, side // 8)))
    ax.set_yticks(range(0, side, max(1, side // 8)))
    ax.tick_params(axis="both", which="both", length=0)

    # Pick a natural unit name: prefer explicit override, else infer from the
    # signature's first feature name (e.g. "ch_*" → channel; "px_*" → pixel;
    # "h*_d*" → attention-head; else "feature").
    if unit_label is None:
        unit_label = _infer_unit_label(sig.feature_names)
    plural = unit_label + ("s" if not unit_label.endswith("s") else "")

    counts = np.bincount(sig.archetypes, minlength=8)
    dominant = ARCHETYPE_NAMES[int(np.argmax(counts))]
    pct_noise = 100 * counts[0] / max(n, 1)
    ax.set_title(
        f"Per-{unit_label} archetype map — {n} {plural}  ·  "
        f"dominant: {dominant} ({counts[np.argmax(counts)]})  ·  "
        f"Noise Candidates: {counts[0]} ({pct_noise:.1f}%)\n"
        f"Each square = one {unit_label}, numbered left-to-right, top-to-bottom; "
        f"color encodes its FFCA archetype.",
        fontsize=10,
    )
    ax.set_xlabel(f"{unit_label} index (column)")
    ax.set_ylabel(f"{unit_label} index (row)")

    # legend with counts
    nz = counts > 0
    handles = [plt.Rectangle((0, 0), 1, 1, color=ARCHETYPE_COLORS[i])
               for i in range(8) if nz[i]]
    labels = [f"{ARCHETYPE_NAMES[i]} ({counts[i]})"
              for i in range(8) if nz[i]]
    ax.legend(handles, labels, fontsize=8, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    plt.tight_layout()
    if out is not None:
        _save(fig, out, formats)
    return fig


def fbr_diagnostic(sig: FFCASignature, feature_shape: tuple[int, ...],
                   fg_frac: float = 0.5, out: Path | None = None,
                   formats=("png",)):
    """Single-glance bar plot: mean foreground vs mean background interaction."""
    if len(feature_shape) != 3:
        return None
    C, H, W = feature_shape
    img = sig.interaction.reshape(C, H, W)
    py = int(H * (1 - fg_frac) / 2)
    px = int(W * (1 - fg_frac) / 2)
    fg = img[:, py:H - py, px:W - px].mean()
    mask = np.ones_like(img, dtype=bool)
    mask[:, py:H - py, px:W - px] = False
    bg = img[mask].mean()
    fbr = fg / (fg + bg) if (fg + bg) > 0 else float("nan")
    if np.isnan(fbr):
        verdict = "—"; color_bg = "#7f8c8d"
    elif fbr < 0.35:
        verdict = "STRONG shortcut signal"; color_bg = "#c0392b"
    elif fbr < 0.5:
        verdict = "moderate shortcut risk"; color_bg = "#e67e22"
    else:
        verdict = "model focuses on foreground (OK)"; color_bg = "#27ae60"

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(["Foreground\n(centre half)", "Background\n(outer ring)"],
                  [fg, bg], color=["#2ecc71", "#e74c3c"])
    for b, v in zip(bars, [fg, bg]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.3f}",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Mean interaction (Σ |∂²f/∂x∂x'|)")
    ax.set_title(f"Foreground/Background ratio = {fbr:.3f}  →  {verdict}\n"
                 f"(FBR = fg / (fg + bg); below 0.5 hints at background shortcut)",
                 fontsize=10, color=color_bg)
    if out is not None:
        _save(fig, out, formats)
    return fig
