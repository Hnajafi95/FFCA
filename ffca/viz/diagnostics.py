"""Diagnostic plots: co-sens groups, validation, group accuracy."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .static import _save


def co_sensitivity_groups(cosens, out: Path | None = None, formats=("png",)):
    """Bar of group sizes with NC-fraction overlay; abort flag in title."""
    if not getattr(cosens, "results", None):
        return None
    groups = cosens.results
    gids = sorted(groups.keys())
    sizes = [groups[g]["size"] for g in gids]
    ncs = [groups[g]["nc_fraction"] for g in gids]
    recs = [groups[g]["recommendation"] for g in gids]

    fig, ax1 = plt.subplots(figsize=(7, 5))
    colors = ["#e74c3c" if r.startswith("PRUNE")
              else "#f39c12" if r.startswith("REVIEW")
              else "#2ecc71" for r in recs]
    bars = ax1.bar(range(len(gids)), sizes, color=colors, alpha=0.75)
    ax1.set_xticks(range(len(gids)))
    ax1.set_xticklabels([f"g{g}" for g in gids])
    ax1.set_ylabel("Group size", color="#34495e")
    for b, n in zip(bars, sizes):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height(),
                 f"{int(n)}", ha="center", va="bottom", fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(range(len(gids)), ncs, "o-", color="#c0392b", lw=2, ms=8)
    ax2.set_ylabel("Noise Candidate fraction", color="#c0392b")
    ax2.set_ylim(0, 1.05)
    ax2.axhline(0.5, color="#c0392b", linestyle="--", alpha=0.4, label="prune threshold")
    ax2.axhline(0.3, color="#7f8c8d", linestyle=":", alpha=0.4, label="review threshold")
    ax2.legend(loc="upper right", fontsize=7)

    d = cosens.diagnostics
    abort = d.get("abort_recommended", True)
    best_nc = d.get("best_nc_fraction", 0.0)
    n_groups = len(gids)
    n_prune = sum(1 for r in recs if r.startswith("PRUNE"))
    if abort:
        headline = (f"No prune-safe functional group found "
                    f"(best group has {best_nc:.0%} Noise; needs > 50%)")
    else:
        headline = f"{n_prune} of {n_groups} groups recommended for pruning"
    subtitle = (f"k={d.get('k')} groups · silhouette={d.get('silhouette_observed', 0):.2f} "
                f"· permutation p={d.get('permutation_p', float('nan')):.3f} "
                f"· bootstrap ARI={d.get('bootstrap_ari_median', float('nan')):.2f}")
    ax1.set_title(f"Co-Sensitivity: {headline}\n{subtitle}", fontsize=10)
    ax1.set_xlabel("functional group (cluster medoid)")
    if out is not None:
        _save(fig, out, formats)
    return fig


def group_accuracy_bars(group_acc: dict, group_names: list[str] | None = None,
                        out: Path | None = None, formats=("png",)):
    """Per-group validation accuracy bar plot — for shortcut/spurious diagnostics."""
    if not group_acc:
        return None
    keys = sorted(group_acc.keys())
    vals = [group_acc[k] for k in keys]
    if group_names is None:
        group_names = [str(k) for k in keys]
    colors = ["#2ecc71" if v >= 0.5 else "#e74c3c" for v in vals]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(range(len(keys)), vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}",
                ha="center", va="bottom", fontsize=10)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(group_names)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="grey", linestyle="--", alpha=0.5)
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Per-group validation accuracy (shortcut-learning diagnostic)")
    if out is not None:
        _save(fig, out, formats)
    return fig


def cauchy_hvp_validation_panel(rows: list[dict], out: Path | None = None,
                                formats=("png",)):
    """Scatter of (d, B) configurations: Spearman, rel-err, CI coverage."""
    if not rows:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    Bs = [r["B"] for r in rows]
    sps = [r["spearman"] for r in rows]
    erres = [r["median_rel_err"] for r in rows]
    covs = [r["ci_coverage"] for r in rows]
    labels = [f"d={r['d']}\nn={r['n_samples']}" for r in rows]

    axes[0].scatter(Bs, sps, s=80, c="#3498db")
    for x, y, l in zip(Bs, sps, labels):
        axes[0].annotate(l, (x, y), fontsize=7, ha="left", va="bottom")
    axes[0].axhline(0.95, color="grey", linestyle="--", alpha=0.5)
    axes[0].set_xlabel("B"); axes[0].set_ylabel("Spearman r")
    axes[0].set_title("Rank correlation vs exact Hessian")
    axes[0].set_ylim(0.8, 1.01)

    axes[1].scatter(Bs, erres, s=80, c="#e67e22")
    for x, y, l in zip(Bs, erres, labels):
        axes[1].annotate(l, (x, y), fontsize=7, ha="left", va="bottom")
    axes[1].axhline(0.10, color="grey", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("B"); axes[1].set_ylabel("Median rel error")
    axes[1].set_title("Magnitude error vs exact Hessian")

    axes[2].scatter(Bs, covs, s=80, c="#27ae60")
    for x, y, l in zip(Bs, covs, labels):
        axes[2].annotate(l, (x, y), fontsize=7, ha="left", va="bottom")
    axes[2].axhline(0.95, color="grey", linestyle="--", alpha=0.5)
    axes[2].set_xlabel("B"); axes[2].set_ylabel("95% CI coverage")
    axes[2].set_title("Wald CI calibration")
    axes[2].set_ylim(0, 1.05)
    if out is not None:
        _save(fig, out, formats)
    return fig
