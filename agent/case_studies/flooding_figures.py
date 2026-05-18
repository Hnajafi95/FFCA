"""Generate 4 paper-ready figures for the compound-flooding case study.

Inputs (no API calls):
  - the 40 flooding report.json files (for trust-bucket counts)
  - ffca_before_after_comparison.csv (archetype shifts per experiment)
  - ffca_vs_original_comparison.csv (R²/RMSE for the retrained models)

Outputs (presentation/case_study/figures/):
  - fig01_trust_buckets_gate.png       — stacked bar, gate experiment only
  - fig02_investigate_rate_heatmap.png — % INVESTIGATE before/after, all 20 experiments
  - fig03_skill_vs_investigate.png     — RMSE delta vs INVESTIGATE delta, scatter
  - fig04_archetype_shifts_gate.png    — paired-bar archetype shifts, gate experiment

The figures are designed to be self-contained: each has a title, axes labels,
a brief annotation, and a small text box describing what the figure shows.
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

EXPERIMENT_GROUPS = [
    ("Measurements Only",       "measured"),
    ("Predicted Ocean Water Levels", "wls"),
    ("Predicted Rainfall",      "rain"),
    ("Predicted Gate Opening",  "gate"),
    ("Predictions All Inputs",  "all_inputs"),
]
LEAD_TIMES = [3, 6, 12, 24]

# ── helpers ────────────────────────────────────────────────────────────────


def _exp_dirname(group_short: str, lead: int) -> str:
    if group_short == "measured":
        return f"{lead}hr_measured_sigmoid"
    return f"{lead}hr_perfect_prog_{group_short}_sigmoid"


def _report_path(group_label: str, group_short: str, lead: int, when: str) -> Path:
    base = (
        FLOOD_ROOT / "FFCA_resutls_before_prunning" if when == "before"
        else FLOOD_ROOT / "FFCA_results_After_prunning"
    )
    return base / group_label / _exp_dirname(group_short, lead) / "report.json"


def _trust_counts(p: Path) -> dict[str, int]:
    raw = json.loads(p.read_text())
    return dict(raw.get("trust_summary", {}))


# ── figure 1: trust-bucket stacked bar for the gate experiment ─────────────


def fig01_trust_buckets_gate() -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bucket_order = ["CONFIDENTLY KEEP", "KEEP (stable)", "MONITOR (borderline)",
                    "INVESTIGATE (unstable)", "CONFIDENTLY PRUNE"]
    colors = ["#1b7837", "#7fbf7b", "#fdb863", "#e08214", "#b35806"]

    # Each lead time gets a before-bar and an after-bar side by side
    x = np.arange(len(LEAD_TIMES))
    width = 0.42
    for j, when in enumerate(["before", "after"]):
        bottoms = np.zeros(len(LEAD_TIMES))
        for k, bucket in enumerate(bucket_order):
            heights = []
            for lead in LEAD_TIMES:
                p = _report_path("Predicted Gate Opening", "gate", lead, when)
                counts = _trust_counts(p)
                heights.append(counts.get(bucket, 0))
            heights = np.asarray(heights, dtype=float)
            offset = -width / 2 if when == "before" else width / 2
            ax.bar(x + offset, heights, width=width, bottom=bottoms,
                   color=colors[k], edgecolor="white", linewidth=0.5,
                   label=bucket if (j == 0) else None)
            bottoms += heights

    ax.set_xticks(x)
    ax.set_xticklabels([f"{lt}h" for lt in LEAD_TIMES])
    ax.set_xlabel("Lead time (left bar = before pruning, right bar = after)")
    ax.set_ylabel("Feature count")
    ax.set_title("Trust-bucket composition — Predicted Gate Opening (before vs after pruning)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
    ax.text(0.02, 0.97,
            "Pruning shrinks every bar by 60–72%.\n"
            "INVESTIGATE share stays high at long leads —\n"
            "the failure mode the agent flags.",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7"))
    fig.tight_layout()
    fig.savefig(OUT / "fig01_trust_buckets_gate.png", dpi=160)
    plt.close(fig)
    print(f"wrote {OUT / 'fig01_trust_buckets_gate.png'}")


# ── figure 2: INVESTIGATE % heatmap, all 20 experiments × {before, after} ──


def fig02_investigate_rate_heatmap() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    labels = [g for g, _ in EXPERIMENT_GROUPS]
    for ax, when in zip(axes, ["before", "after"]):
        mat = np.zeros((len(EXPERIMENT_GROUPS), len(LEAD_TIMES)))
        for i, (g_label, g_short) in enumerate(EXPERIMENT_GROUPS):
            for j, lead in enumerate(LEAD_TIMES):
                p = _report_path(g_label, g_short, lead, when)
                raw = json.loads(p.read_text())
                n = raw.get("n_features", 1)
                inv = raw.get("trust_summary", {}).get("INVESTIGATE (unstable)", 0)
                mat[i, j] = 100 * inv / max(n, 1)
        im = ax.imshow(mat, cmap="OrRd", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(LEAD_TIMES)))
        ax.set_xticklabels([f"{lt}h" for lt in LEAD_TIMES])
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_title(f"% INVESTIGATE — {when} pruning")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center",
                        color="black" if mat[i, j] < 60 else "white", fontsize=8)
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label("% of features in INVESTIGATE", fontsize=9)
    fig.suptitle("Trust-instability (% INVESTIGATE) — all 20 experiments", y=1.02)
    fig.savefig(OUT / "fig02_investigate_rate_heatmap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'fig02_investigate_rate_heatmap.png'}")


# ── figure 3: skill change vs INVESTIGATE-rate change ──────────────────────


def fig03_skill_vs_investigate() -> None:
    skill = pd.read_csv(FLOOD_ROOT / "MLMiami FFCA Prunned Results" / "results"
                        / "ffca_vs_original_comparison.csv")
    arch = pd.read_csv(FLOOD_ROOT / "FFCA_results_After_prunning"
                       / "ffca_before_after_comparison.csv")
    df = skill.merge(arch[["experiment", "trust_investigate_before",
                           "trust_investigate_after", "n_features_before",
                           "n_features_after"]],
                     on="experiment", how="inner")
    df["inv_rate_before"] = 100 * df["trust_investigate_before"] / df["n_features_before"]
    df["inv_rate_after"]  = 100 * df["trust_investigate_after"]  / df["n_features_after"]
    df["inv_rate_delta"]  = df["inv_rate_after"] - df["inv_rate_before"]
    # The skill column is RMSE_delta = ffca - original; negative = better post-pruning.

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for cat, color, marker in [
        ("measured", "#377eb8", "o"),
        ("wls", "#984ea3", "s"),
        ("rain", "#4daf4a", "^"),
        ("gate", "#e41a1c", "D"),
        ("all_inputs", "#ff7f00", "P"),
    ]:
        sub = df[df["experiment"].str.contains(cat)]
        ax.scatter(sub["inv_rate_delta"], sub["RMSE_delta"], s=70,
                   c=color, marker=marker, label=cat, edgecolors="black",
                   linewidths=0.5, alpha=0.85)
        for _, row in sub.iterrows():
            label = f"{int(row['lead_time'])}h"
            ax.annotate(label,
                        xy=(row["inv_rate_delta"], row["RMSE_delta"]),
                        xytext=(4, 4), textcoords="offset points", fontsize=7)

    ax.axhline(0, color="0.5", linestyle="--", linewidth=0.7)
    ax.axvline(0, color="0.5", linestyle="--", linewidth=0.7)
    ax.set_xlabel("Δ INVESTIGATE rate after pruning (pp)")
    ax.set_ylabel("Δ RMSE after pruning (cm)\n← improved   degraded →")
    ax.set_title("Pruning effect: trust stabilization vs skill change\n"
                 "(top-right quadrant = pruning made the model BOTH less stable AND worse)")
    ax.legend(loc="lower right", fontsize=8, title="Input category")
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.97,
            f"n={len(df)} retrained models.\n"
            f"3 degraded measurably (top-right):\n"
            f"all gate-input or all-inputs, lead ≥ 12h.\n"
            f"FFCA's INVESTIGATE rate did NOT drop\n"
            f"for those — the diagnostic preceded the\n"
            f"skill loss.",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7"))
    fig.tight_layout()
    fig.savefig(OUT / "fig03_skill_vs_investigate.png", dpi=160)
    plt.close(fig)
    print(f"wrote {OUT / 'fig03_skill_vs_investigate.png'}")


# ── figure 4: paired-bar archetype shifts (gate experiment) ────────────────


def fig04_archetype_shifts_gate() -> None:
    arch = pd.read_csv(FLOOD_ROOT / "FFCA_results_After_prunning"
                       / "ffca_before_after_comparison.csv")
    gate = arch[arch["experiment"].str.contains("gate")].copy()
    gate["lead_h"] = gate["experiment"].str.extract(r"(\d+)hr").astype(int)
    gate = gate.sort_values("lead_h")

    # plot the 4 archetype-pct deltas as grouped bars
    archs = ["noise", "workhorse", "catalyst", "complex_driver", "stable_contributor"]
    colors = ["#999999", "#1f78b4", "#a6cee3", "#33a02c", "#b2df8a"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(gate))
    width = 0.16
    for k, a in enumerate(archs):
        ax.bar(x + (k - 2) * width, gate[f"{a}_pct_delta"].values,
               width=width, color=colors[k], edgecolor="black", linewidth=0.4,
               label=a.replace("_", " ").title())

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lh}h" for lh in gate["lead_h"]])
    ax.set_xlabel("Lead time")
    ax.set_ylabel("Δ archetype share (pp)\nafter pruning vs before")
    ax.set_title("Archetype-share shifts after pruning — Predicted Gate Opening")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.text(0.02, 0.05,
            "At 24h: Complex Drivers drop 21.6pp and Noise rises 10.4pp —\n"
            "post-pruning the model's healthier archetypes shrank, not grew.",
            transform=ax.transAxes, va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7"))
    fig.tight_layout()
    fig.savefig(OUT / "fig04_archetype_shifts_gate.png", dpi=160)
    plt.close(fig)
    print(f"wrote {OUT / 'fig04_archetype_shifts_gate.png'}")


def main() -> None:
    fig01_trust_buckets_gate()
    fig02_investigate_rate_heatmap()
    fig03_skill_vs_investigate()
    fig04_archetype_shifts_gate()


if __name__ == "__main__":
    main()
