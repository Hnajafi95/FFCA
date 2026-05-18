"""Additional figures covering all 5 input categories.

Generated AFTER the per-category narrations land. Inputs:
  - flooding_narrations/summary.json — diagnostic_rule_ids per experiment
  - the 40 report.json files — for trust-bucket counts
  - the comparison CSVs — for archetype shifts + R²/RMSE deltas

Outputs (presentation/case_study/figures/):
  - fig05_rules_fired_grid.png        — which diagnostic rules fire per (cat, lead, state)
  - fig06_agent_verdict_map.png       — agent's per-experiment after-pruning verdict
  - fig07_trust_panels_all.png        — trust-bucket stacked bars, 5 categories × 4 leads
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
FLOOD_ROOT = Path("/Users/hnaja002/Documents/projects/compound_flooding")
OUT = REPO / "presentation" / "case_study" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

CATEGORIES = [
    ("Measurements Only",            "measured"),
    ("Predicted Ocean Water Levels", "wls"),
    ("Predicted Rainfall",           "rain"),
    ("Predicted Gate Opening",       "gate"),
    ("Predictions All Inputs",       "all_inputs"),
]
LEAD_TIMES = [3, 6, 12, 24]
SUMMARY = json.loads((REPO / "FFCA_runs_results_v04_real/flooding_narrations/summary.json").read_text())


def _dirname(cat_short: str, lead: int) -> str:
    if cat_short == "measured":
        return f"{lead}hr_measured_sigmoid"
    return f"{lead}hr_perfect_prog_{cat_short}_sigmoid"


def _report_path(cat_label: str, cat_short: str, lead: int, when: str) -> Path:
    base = (FLOOD_ROOT / "FFCA_resutls_before_prunning" if when == "before"
            else FLOOD_ROOT / "FFCA_results_After_prunning")
    return base / cat_label / _dirname(cat_short, lead) / "report.json"


# ── figure 5: diagnostic rule firings grid ────────────────────────────────


def fig05_rules_fired_grid() -> None:
    """Binary heatmap: for each experiment × state, which rules fired."""
    interesting = ["trust_instability_high", "late_checkpoint_drift",
                   "cosens_weak_clustering_significant"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, when in zip(axes, ["before", "after"]):
        rows = []
        labels = []
        for cat_label, cat_short in CATEGORIES:
            for lead in LEAD_TIMES:
                key = f"{cat_short}/{when}/{lead}hr"
                ids = set(SUMMARY.get(key, {}).get("diagnostic_rule_ids", []))
                rows.append([1 if r in ids else 0 for r in interesting])
                labels.append(f"{cat_short[:4]:>4} {lead}h")
        mat = np.asarray(rows)
        im = ax.imshow(mat, cmap="Reds", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(interesting)))
        ax.set_xticklabels([r.replace("_", "\n") for r in interesting], fontsize=8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        # Category separators
        for i in range(1, len(CATEGORIES)):
            ax.axhline(i * len(LEAD_TIMES) - 0.5, color="black", linewidth=0.6)
        ax.set_title(f"diagnostic rules fired — {when} pruning", fontsize=11)
    fig.suptitle("Which diagnostic rules fire across all 20 experiments × {before, after pruning}",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(OUT / "fig05_rules_fired_grid.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT/'fig05_rules_fired_grid.png'}")


# ── figure 6: agent verdict map ────────────────────────────────────────────


def _classify_verdict(ids_after: set[str], ids_before: set[str]) -> int:
    """Return 0 (clean), 1 (residual instability), 2 (no improvement / worse)."""
    cleared = "trust_instability_high" in ids_before and "trust_instability_high" not in ids_after
    still_unstable = "trust_instability_high" in ids_after
    if cleared and not still_unstable:
        return 0
    if not still_unstable:
        return 0  # never had instability, or it cleared
    if still_unstable and "cosens_weak_clustering_significant" in ids_after:
        return 2  # multiple persisting issues
    return 1


def fig06_agent_verdict_map() -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    mat = np.zeros((len(CATEGORIES), len(LEAD_TIMES)), dtype=int)
    for i, (cat_label, cat_short) in enumerate(CATEGORIES):
        for j, lead in enumerate(LEAD_TIMES):
            bef = set(SUMMARY.get(f"{cat_short}/before/{lead}hr", {}).get("diagnostic_rule_ids", []))
            aft = set(SUMMARY.get(f"{cat_short}/after/{lead}hr", {}).get("diagnostic_rule_ids", []))
            mat[i, j] = _classify_verdict(aft, bef)
    cmap = plt.matplotlib.colors.ListedColormap(["#4daf4a", "#ffeda0", "#e41a1c"])
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(range(len(LEAD_TIMES)))
    ax.set_xticklabels([f"{lt}h" for lt in LEAD_TIMES])
    ax.set_yticks(range(len(CATEGORIES)))
    ax.set_yticklabels([c[0] for c in CATEGORIES], fontsize=9)
    ax.set_title("Agent verdict on pruning outcome (after-pruning state)\n"
                 "green = clean / instability cleared,  yellow = residual instability,"
                 "  red = multiple persistent issues")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            text = {0: "OK", 1: "unstable", 2: "no improve"}[int(v)]
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    color="black" if v < 2 else "white")
    fig.tight_layout()
    fig.savefig(OUT / "fig06_agent_verdict_map.png", dpi=160)
    plt.close(fig)
    print(f"wrote {OUT/'fig06_agent_verdict_map.png'}")


# ── figure 7: trust panels for all 5 categories ────────────────────────────


def fig07_trust_panels_all() -> None:
    bucket_order = ["CONFIDENTLY KEEP", "KEEP (stable)", "MONITOR (borderline)",
                    "INVESTIGATE (unstable)", "CONFIDENTLY PRUNE"]
    colors = ["#1b7837", "#7fbf7b", "#fdb863", "#e08214", "#b35806"]
    fig, axes = plt.subplots(5, 1, figsize=(10, 14), sharex=True)
    x = np.arange(len(LEAD_TIMES))
    width = 0.42
    for ax, (cat_label, cat_short) in zip(axes, CATEGORIES):
        for j, when in enumerate(["before", "after"]):
            bottoms = np.zeros(len(LEAD_TIMES))
            for k, bucket in enumerate(bucket_order):
                heights = []
                for lead in LEAD_TIMES:
                    p = _report_path(cat_label, cat_short, lead, when)
                    if p.exists():
                        raw = json.loads(p.read_text())
                        heights.append(raw.get("trust_summary", {}).get(bucket, 0))
                    else:
                        heights.append(0)
                offset = -width / 2 if when == "before" else width / 2
                ax.bar(x + offset, heights, width=width, bottom=bottoms,
                       color=colors[k], edgecolor="white", linewidth=0.4,
                       label=bucket if (j == 0 and ax is axes[0]) else None)
                bottoms += heights
        ax.set_ylabel(f"{cat_label}\n(features)", fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([f"{lt}h (left=before, right=after)" for lt in LEAD_TIMES],
                              fontsize=9)
    axes[0].legend(loc="upper left", fontsize=7, bbox_to_anchor=(1.01, 1.0))
    fig.suptitle("Trust-bucket composition — all 5 input categories", fontsize=12, y=0.995)
    fig.tight_layout()
    fig.savefig(OUT / "fig07_trust_panels_all.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT/'fig07_trust_panels_all.png'}")


def main() -> None:
    fig05_rules_fired_grid()
    fig06_agent_verdict_map()
    fig07_trust_panels_all()


if __name__ == "__main__":
    main()
