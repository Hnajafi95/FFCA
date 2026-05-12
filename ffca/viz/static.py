"""Static-FFCA plots: radar, archetype distribution, ranking, CI."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..core.archetypes import ARCHETYPE_NAMES
from ..core.signature import FFCASignature


# Consistent archetype colour palette across all plots
ARCHETYPE_COLORS = [
    "#bdc3c7",  # 0 Noise — grey
    "#9b59b6",  # 1 Hidden Interactor — purple
    "#2ecc71",  # 2 Workhorse — green
    "#e67e22",  # 3 Catalyst — orange
    "#e74c3c",  # 4 Nonlinear Driver — red
    "#f1c40f",  # 5 Volatile Specialist — yellow
    "#3498db",  # 6 Stable Contributor — blue
    "#34495e",  # 7 Complex Driver — dark
]


def _save(fig, base: Path, formats):
    base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(base.with_suffix(f".{fmt}"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def signature_radar(sig: FFCASignature, top_k: int = 6,
                    out: Path | None = None, formats=("png",)):
    """Radar plot of the 4-D signature for the top-k features by impact."""
    top = sig.top_k(top_k, by="impact")
    # Normalise each axis to [0, 1] for radar comparability
    def _norm(x):
        x = np.asarray(x, dtype=float)
        m = x.max() if x.max() > 0 else 1.0
        return x / m
    norms = {
        "Impact": _norm(sig.impact),
        "Volatility": _norm(sig.volatility),
        "Non-linearity": _norm(sig.nonlinearity),
        "Interaction": _norm(sig.interaction),
    }
    angles = np.linspace(0, 2 * np.pi, len(norms), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for idx in top:
        vals = [norms[k][idx] for k in norms]
        vals += vals[:1]
        ax.plot(angles, vals, label=sig.feature_names[idx], alpha=0.8)
        ax.fill(angles, vals, alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(list(norms.keys()))
    ax.set_ylim(0, 1)
    ax.set_title(f"FFCA 4-D signature — top {top_k} features by impact")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=8)
    if out is not None:
        _save(fig, out, formats)
    return fig


def archetype_distribution(sig: FFCASignature, out: Path | None = None,
                           formats=("png",)):
    if sig.archetypes is None:
        return None
    counts = np.bincount(sig.archetypes, minlength=8)
    tot = counts.sum()
    dominant_i = int(np.argmax(counts))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Bar
    axes[0].bar(range(8), counts, color=ARCHETYPE_COLORS)
    axes[0].set_xticks(range(8))
    axes[0].set_xticklabels(ARCHETYPE_NAMES, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("# features")
    axes[0].set_title(f"Archetype distribution — dominant: {ARCHETYPE_NAMES[dominant_i]} "
                       f"({counts[dominant_i]}/{tot})")
    for i, c in enumerate(counts):
        if c > 0:
            axes[0].text(i, c, str(int(c)), ha="center", va="bottom", fontsize=8)
    # Pie
    nz = counts > 0
    axes[1].pie(counts[nz], labels=[ARCHETYPE_NAMES[i] for i in range(8) if nz[i]],
                colors=[ARCHETYPE_COLORS[i] for i in range(8) if nz[i]],
                autopct="%1.0f%%", textprops={"fontsize": 9})
    noise_pct = 100 * counts[0] / tot
    cat_pct = 100 * counts[3] / tot
    cx_pct = 100 * counts[7] / tot
    axes[1].set_title(f"Noise: {noise_pct:.0f}%  ·  Catalyst: {cat_pct:.0f}%  ·  "
                       f"Complex Driver: {cx_pct:.0f}%")
    if out is not None:
        _save(fig, out, formats)
    return fig


def impact_ranking(sig: FFCASignature, top_k: int = 20,
                   out: Path | None = None, formats=("png",)):
    top = sig.top_k(top_k, by="impact")
    fig, ax = plt.subplots(figsize=(8, max(3, top_k * 0.25 + 1)))
    impacts = sig.impact[top]
    archs = sig.archetypes[top] if sig.archetypes is not None else np.zeros(len(top), dtype=int)
    colors = [ARCHETYPE_COLORS[int(a)] for a in archs]
    ypos = np.arange(len(top))
    ax.barh(ypos, impacts, color=colors)
    ax.set_yticks(ypos)
    ax.set_yticklabels([sig.feature_names[i] for i in top], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Impact = E|∂f/∂x|")
    ax.set_title(f"Top {top_k} features by impact (bar color = archetype)")
    # mini legend with the archetypes present in this plot
    seen = sorted(set(int(a) for a in archs))
    handles = [plt.Rectangle((0, 0), 1, 1, color=ARCHETYPE_COLORS[i]) for i in seen]
    ax.legend(handles, [ARCHETYPE_NAMES[i] for i in seen], fontsize=7, loc="lower right")
    if out is not None:
        _save(fig, out, formats)
    return fig


def interaction_ci_plot(sig: FFCASignature, top_k: int = 30,
                        out: Path | None = None, formats=("png",)):
    if sig.interaction_ci is None:
        return None
    top = sig.top_k(top_k, by="interaction")
    inter = sig.interaction[top]
    ci_lower = sig.interaction_ci[top, 0]
    ci_upper = sig.interaction_ci[top, 1]
    errors = np.row_stack([inter - ci_lower, ci_upper - inter])

    fig, ax = plt.subplots(figsize=(8, max(3, top_k * 0.25 + 1)))
    ypos = np.arange(len(top))
    ax.errorbar(inter, ypos, xerr=errors, fmt="o", capsize=3, color="#3498db", ecolor="#7f8c8d")
    ax.set_yticks(ypos)
    ax.set_yticklabels([sig.feature_names[i] for i in top], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(x=0, color="grey", linestyle="--", alpha=0.5)
    ax.set_xlabel("Interaction ‖H_{i,:}‖₁ − |H_{ii}|  (95% Cauchy CI)")
    ax.set_title(f"Top {top_k} features by interaction")
    if out is not None:
        _save(fig, out, formats)
    return fig
