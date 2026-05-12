"""Dynamic-FFCA plots: evolution curves, ranking, archetype heatmap, trust scatter."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..core.archetypes import ARCHETYPE_NAMES
from ..core.signature import FFCASignature
from .static import ARCHETYPE_COLORS, _save


def impact_evolution_curves(signatures: list[FFCASignature],
                            checkpoint_labels: list[str],
                            top_k: int = 8, out: Path | None = None,
                            formats=("png",)):
    """Per-feature impact across checkpoints (top-k by final impact)."""
    if len(signatures) < 2:
        return None
    last = signatures[-1]
    top = last.top_k(top_k, by="impact")
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(signatures))
    for idx in top:
        ys = [s.impact[idx] for s in signatures]
        ax.plot(x, ys, "-o", lw=1.5, ms=4, alpha=0.8,
                label=last.feature_names[idx])
    ax.set_xticks(x)
    ax.set_xticklabels(checkpoint_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Impact")
    ax.set_title(f"Impact evolution — top {top_k} features")
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.grid(alpha=0.3)
    if out is not None:
        _save(fig, out, formats)
    return fig


def ranking_evolution(signatures: list[FFCASignature],
                      checkpoint_labels: list[str],
                      top_k: int = 8, out: Path | None = None,
                      formats=("png",)):
    """Bump chart: feature rank (1 = highest impact) over checkpoints."""
    if len(signatures) < 2:
        return None
    last = signatures[-1]
    top = last.top_k(top_k, by="impact")
    ranks = np.zeros((len(signatures), len(top)), dtype=int)
    for t, s in enumerate(signatures):
        order = np.argsort(-s.impact)  # rank 0 = highest impact
        rank_map = {f: r for r, f in enumerate(order)}
        for i, idx in enumerate(top):
            ranks[t, i] = rank_map[idx] + 1
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(signatures))
    for i, idx in enumerate(top):
        ax.plot(x, ranks[:, i], "-o", lw=1.5, ms=5, alpha=0.85,
                label=last.feature_names[idx])
    ax.invert_yaxis()
    ax.set_xticks(x)
    ax.set_xticklabels(checkpoint_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Rank by impact (1 = highest)")
    ax.set_title("Ranking evolution")
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.grid(alpha=0.3)
    if out is not None:
        _save(fig, out, formats)
    return fig


def archetype_evolution_heatmap(signatures: list[FFCASignature],
                                checkpoint_labels: list[str],
                                top_k: int = 30,
                                out: Path | None = None, formats=("png",)):
    """Heatmap: features × checkpoints, colored by archetype."""
    if len(signatures) < 2 or signatures[0].archetypes is None:
        return None
    last = signatures[-1]
    top = last.top_k(top_k, by="impact")
    mat = np.zeros((len(top), len(signatures)), dtype=int)
    for t, s in enumerate(signatures):
        for i, idx in enumerate(top):
            mat[i, t] = int(s.archetypes[idx])

    cmap = matplotlib.colors.ListedColormap(ARCHETYPE_COLORS)
    bounds = np.arange(9) - 0.5
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    fig, ax = plt.subplots(figsize=(max(6, len(signatures)), max(4, top_k * 0.25 + 1)))
    ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(signatures)))
    ax.set_xticklabels(checkpoint_labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([last.feature_names[i] for i in top], fontsize=7)
    ax.set_title("Archetype evolution")
    handles = [plt.Rectangle((0, 0), 1, 1, color=ARCHETYPE_COLORS[i]) for i in range(8)]
    ax.legend(handles, ARCHETYPE_NAMES, fontsize=7, loc="upper right",
              bbox_to_anchor=(1.4, 1.0))
    if out is not None:
        _save(fig, out, formats)
    return fig


def trust_score_scatter(trust_results: dict, out: Path | None = None,
                        formats=("png",)):
    """Stability vs Importance scatter, colored by decision."""
    if not trust_results:
        return None
    colors = {
        "CONFIDENTLY PRUNE": "#e74c3c",
        "CONFIDENTLY KEEP": "#2ecc71",
        "KEEP (stable)": "#27ae60",
        "INVESTIGATE (unstable)": "#f39c12",
        "MONITOR (borderline)": "#3498db",
    }
    fig, ax = plt.subplots(figsize=(8.5, 6))
    by_dec = {}
    for name, v in trust_results.items():
        by_dec.setdefault(v["decision"], []).append((v["stability"], v["importance"], name))
    for dec, pts in by_dec.items():
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax.scatter(xs, ys, c=colors.get(dec, "#7f8c8d"),
                   label=f"{dec} ({len(pts)})", s=40, alpha=0.75, edgecolors="white")
    ax.axvline(x=0.7, color="grey", linestyle="--", alpha=0.5)
    ax.axvline(x=0.5, color="grey", linestyle=":", alpha=0.5)
    # Quadrant labels at top of each region
    ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.0
    ax.text(0.85, ymax * 0.97, "STABLE\n(prune or keep)", ha="center", va="top",
            fontsize=8, color="#34495e", alpha=0.6)
    ax.text(0.6, ymax * 0.97, "MONITOR", ha="center", va="top",
            fontsize=8, color="#34495e", alpha=0.6)
    ax.text(0.25, ymax * 0.97, "UNSTABLE\n(investigate)", ha="center", va="top",
            fontsize=8, color="#34495e", alpha=0.6)
    ax.set_xlabel("Stability  (1 = same archetype every checkpoint, 0 = different every time)")
    ax.set_ylabel("Importance  (mean Impact across checkpoints)")
    inv = sum(1 for v in trust_results.values() if "INVESTIGATE" in v["decision"])
    prune = sum(1 for v in trust_results.values() if "PRUNE" in v["decision"])
    keep = sum(1 for v in trust_results.values() if "CONFIDENTLY KEEP" in v["decision"])
    ax.set_title(f"Trust Score on {len(trust_results)} features — "
                  f"{keep} confidently keep · {prune} confidently prune · {inv} investigate")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if out is not None:
        _save(fig, out, formats)
    return fig
