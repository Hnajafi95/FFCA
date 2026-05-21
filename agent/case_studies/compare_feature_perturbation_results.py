"""
compare_feature_perturbation_results.py
========================================
Reads:
  - Original RMSEs:    compound_flooding/MLMiami FFCA Prunned Results/results/ffca_vs_original_table3.csv
  - Targeted-prune:    FFCA_agent/case_studies/feature_perturbation_runs/<exp>/variant_C_targeted_prune/MLP/test/results.csv
  - Backbone-removal:  FFCA_agent/case_studies/feature_perturbation_runs/<exp>/variant_D_backbone_removal/MLP/test/results.csv

Writes:
  - feature_perturbation_runs/PERTURBATION_TABLE.md
  - feature_perturbation_runs/PERTURBATION_TABLE.csv

Produces two tables:
  (a) Targeted prune — 3 experiments × (Original RMSE | Variant_C RMSE | Δ vs original)
      Pass criterion: Δ within ±0.20 cm of original.
  (b) Backbone removal — 20 experiments × (Original | Aggressive-prune | Variant_D | Δ vs original)
      Pass criterion (from the agent's perspective): Variant_D Δ > +0.20 cm (i.e. dropping the
      "protect these" features DOES hurt RMSE — confirms the agent's recommendation).
"""
from __future__ import annotations
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CASE = REPO / "FFCA_agent" / "case_studies"
RUNS = CASE / "feature_perturbation_runs"        # variants C, D
SINGLE_SEED_RUNS = CASE / "single_seed_runs"     # variant E
ORIGINAL_CSV = REPO / "compound_flooding" / "MLMiami FFCA Prunned Results" / "results" / "ffca_vs_original_table3.csv"

RMSE_THRESHOLD_CM = 0.20  # v0.8: renamed from DEGRADED_THRESHOLD_CM (no banned label)
DEGRADED_THRESHOLD_CM = RMSE_THRESHOLD_CM  # keep alias for back-compat

# Prune cases derived from the corrected drop_lists.json after the
# 2026-05-20 audit. Under the corrected pipeline (string-mismatch fix
# C2, median aggregation, row-concat Co-Sens), exactly two experiments
# now have a Co-Sens prune-safe group passing all three gates
# (NC>=0.5, perm-p<0.05, bootstrap ARI>=0.5):
#   - 3hr_measured_sigmoid (15-feature gate cluster)
#   - 24hr_perfect_prog_all_inputs_sigmoid (3-feature gate cluster)
# Pre-audit the pair was {12hr_measured, 6hr_measured}; that older
# selection was an artefact of the pre-fix Co-Sens gradient averaging.
PRUNE_CASES = [
    ("3hr_measured_sigmoid",                       "Measured 3h"),
    ("24hr_perfect_prog_all_inputs_sigmoid",       "AllInputs 24h"),
]

# 20 backbone-removal cases (all experiments, by (category, lead, exp_name, display))
ALL_EXPS = [
    ("Measurements Only",         3,  "3hr_measured_sigmoid",                  "Measured 3h"),
    ("Measurements Only",         6,  "6hr_measured_sigmoid",                  "Measured 6h"),
    ("Measurements Only",         12, "12hr_measured_sigmoid",                 "Measured 12h"),
    ("Measurements Only",         24, "24hr_measured_sigmoid",                 "Measured 24h"),
    ("Predicted Ocean Water Levels", 3,  "3hr_perfect_prog_wls_sigmoid",       "WLS 3h"),
    ("Predicted Ocean Water Levels", 6,  "6hr_perfect_prog_wls_sigmoid",       "WLS 6h"),
    ("Predicted Ocean Water Levels", 12, "12hr_perfect_prog_wls_sigmoid",      "WLS 12h"),
    ("Predicted Ocean Water Levels", 24, "24hr_perfect_prog_wls_sigmoid",      "WLS 24h"),
    ("Predicted Rainfall",        3,  "3hr_perfect_prog_rain_sigmoid",         "Rain 3h"),
    ("Predicted Rainfall",        6,  "6hr_perfect_prog_rain_sigmoid",         "Rain 6h"),
    ("Predicted Rainfall",        12, "12hr_perfect_prog_rain_sigmoid",        "Rain 12h"),
    ("Predicted Rainfall",        24, "24hr_perfect_prog_rain_sigmoid",        "Rain 24h"),
    ("Predicted Gate Opening",    3,  "3hr_perfect_prog_gate_sigmoid",         "Gate 3h"),
    ("Predicted Gate Opening",    6,  "6hr_perfect_prog_gate_sigmoid",         "Gate 6h"),
    ("Predicted Gate Opening",    12, "12hr_perfect_prog_gate_sigmoid",        "Gate 12h"),
    ("Predicted Gate Opening",    24, "24hr_perfect_prog_gate_sigmoid",        "Gate 24h"),
    ("Predictions All Inputs",    3,  "3hr_perfect_prog_all_inputs_sigmoid",   "AllInputs 3h"),
    ("Predictions All Inputs",    6,  "6hr_perfect_prog_all_inputs_sigmoid",   "AllInputs 6h"),
    ("Predictions All Inputs",    12, "12hr_perfect_prog_all_inputs_sigmoid",  "AllInputs 12h"),
    ("Predictions All Inputs",    24, "24hr_perfect_prog_all_inputs_sigmoid",  "AllInputs 24h"),
]


def read_results_csv(p: Path) -> dict | None:
    if not p.exists():
        return None
    df = pd.read_csv(p)
    r = df.iloc[0]
    return dict(RMSE_cm=float(r["RMSE"]) * 100, R2=float(r["R2"]), n_features=int(r["n_features"]))


def lookup_original(df_orig: pd.DataFrame, category: str, lead: int) -> dict | None:
    row = df_orig[(df_orig["Model"] == category) & (df_orig["Lead (h)"] == lead) & (df_orig["Variant"] == "Original")]
    if row.empty:
        return None
    r = row.iloc[0]
    return dict(RMSE_cm=float(r["RMSE_cm"]), R2=float(r["R2"]), N_feat=int(r["N_feat"]))


def lookup_pruned(df_orig: pd.DataFrame, category: str, lead: int) -> dict | None:
    row = df_orig[(df_orig["Model"] == category) & (df_orig["Lead (h)"] == lead) & (df_orig["Variant"] == "FFCA")]
    if row.empty:
        return None
    r = row.iloc[0]
    return dict(RMSE_cm=float(r["RMSE_cm"]), R2=float(r["R2"]), N_feat=int(r["N_feat"]))


def main():
    df_orig = pd.read_csv(ORIGINAL_CSV)
    lines = ["# Feature-perturbation experiment results", "",
             "_Auto-generated by `compare_feature_perturbation_results.py`._", ""]

    # ─── Table (a) Targeted prune ────────────────────────────────────────
    lines += [
        "## (a) Targeted prune — agent's affirmative recommendation",
        "",
        "The corrected ensemble-mode FFCA's Co-Sensitivity step identified a",
        "small noise-dominated cluster (NC≥0.5, bootstrap ARI≥0.5) in each of",
        f"3 experiments. Hypothesis: dropping the cluster preserves RMSE within ±{DEGRADED_THRESHOLD_CM:.2f} cm.",
        "",
        "| Experiment | Original RMSE | Variant_C RMSE | Δ vs original | Verdict |",
        "|---|:--:|:--:|:--:|:--:|",
    ]
    csv_rows = []
    pass_c = 0; total_c = 0
    for exp_name, display in PRUNE_CASES:
        cat = next(c for c, l, n, _ in ALL_EXPS if n == exp_name)
        lead = next(l for c, l, n, _ in ALL_EXPS if n == exp_name)
        orig = lookup_original(df_orig, cat, lead)
        cf = read_results_csv(RUNS / exp_name / "variant_C_targeted_prune" / "MLP" / "test" / "results.csv")
        if not orig or not cf:
            lines.append(f"| **{display}** | {f'{orig['RMSE_cm']:.2f} cm' if orig else 'missing'} | pending | — | — |")
            continue
        delta = cf["RMSE_cm"] - orig["RMSE_cm"]
        ok = abs(delta) <= DEGRADED_THRESHOLD_CM
        verdict = "✅ within ±0.20 cm" if ok else "❌ exceeds ±0.20 cm"
        lines.append(f"| **{display}** | {orig['RMSE_cm']:.2f} cm | {cf['RMSE_cm']:.2f} cm | {delta:+.2f} cm | {verdict} |")
        total_c += 1
        if ok: pass_c += 1
        csv_rows.append(dict(table="targeted_prune", experiment=exp_name, display=display,
                              original_rmse=orig["RMSE_cm"], perturbed_rmse=cf["RMSE_cm"],
                              delta_cm=delta, pass_=ok))
    if total_c:
        lines += ["", f"**Pass rate: {pass_c} of {total_c}** ({100*pass_c/total_c:.0f}%).",
                  "Pass = the targeted prune did NOT hurt RMSE materially, confirming the agent's",
                  "affirmative 'safe to drop' recommendation.", ""]

    # ─── Table (b) Backbone removal ──────────────────────────────────────
    lines += [
        "## (b) Backbone removal — agent's 'protect these' recommendation",
        "",
        "For each of 20 experiments, retrained dropping the CONFIDENTLY KEEP",
        "features (the agent's identified load-bearing backbone, 20-48 features",
        f"per experiment). Hypothesis: backbone removal hurts RMSE by > {DEGRADED_THRESHOLD_CM:.2f} cm.",
        "",
        "| Experiment | Original | Aggressive-prune | Variant_D (backbone removed) | Δ_D vs original | Verdict |",
        "|---|:--:|:--:|:--:|:--:|:--:|",
    ]
    pass_d = 0; total_d = 0
    for cat, lead, exp_name, display in ALL_EXPS:
        orig = lookup_original(df_orig, cat, lead)
        prune = lookup_pruned(df_orig, cat, lead)
        cf = read_results_csv(RUNS / exp_name / "variant_D_backbone_removal" / "MLP" / "test" / "results.csv")
        if not orig or not cf:
            lines.append(f"| **{display}** | {f'{orig['RMSE_cm']:.2f}' if orig else 'missing'} | {f'{prune['RMSE_cm']:.2f}' if prune else 'missing'} | pending | — | — |")
            continue
        delta = cf["RMSE_cm"] - orig["RMSE_cm"]
        hurts = delta > DEGRADED_THRESHOLD_CM
        verdict = "✅ backbone matters" if hurts else "⚠ backbone removable"
        prune_str = f"{prune['RMSE_cm']:.2f}" if prune else "—"
        lines.append(f"| **{display}** | {orig['RMSE_cm']:.2f} | {prune_str} | {cf['RMSE_cm']:.2f} | {delta:+.2f} cm | {verdict} |")
        total_d += 1
        if hurts: pass_d += 1
        csv_rows.append(dict(table="backbone_removal", experiment=exp_name, display=display,
                              original_rmse=orig["RMSE_cm"], perturbed_rmse=cf["RMSE_cm"],
                              delta_cm=delta, pass_=hurts))
    if total_d:
        lines += ["", f"**Pass rate: {pass_d} of {total_d}** ({100*pass_d/total_d:.0f}%).",
                  "Pass = removing the agent-flagged backbone DID measurably hurt RMSE,",
                  "confirming the agent's identification of load-bearing features.", ""]

    # ─── Table (c) Single-seed (variant E) ──────────────────────────────
    lines += [
        "## (c) Single-seed test (variant E) — does per-seed FFCA pick truer load-bearing features than ensemble?",
        "",
        "For each experiment, ran single-snapshot FFCA on seed-1's original .h5,",
        "identified seed-1's top-K Impact features (K matched to ensemble CK count),",
        "then retrained seed-1 dropping those features. Compared single-seed test RMSE.",
        "",
        "**Hypothesis (user, 2026-05-20):** if ensembling washes out per-seed",
        "load-bearing signal, per-seed RMSE Δ should be materially larger than the",
        "30-seed variant_D RMSE Δ on the same experiment.",
        "",
        "| Experiment | Baseline seed-1 RMSE | Variant_E seed-1 RMSE | Δ_E | Δ_D (30-seed, table b) | Larger? |",
        "|---|:--:|:--:|:--:|:--:|:--:|",
    ]
    # Read seed-1 baseline RMSEs from per_seed_drop_lists.json (we computed them there)
    psl_path = CASE / "per_seed_drop_lists.json"
    if psl_path.exists():
        psl = json.loads(psl_path.read_text())
    else:
        psl = {}
    pass_e = 0; total_e = 0; e_bigger_than_d = 0; e_vs_d_total = 0
    for cat, lead, exp_name, display in ALL_EXPS:
        baseline = psl.get(exp_name, {}).get("seed_1_baseline_rmse_cm")
        cf = read_results_csv(SINGLE_SEED_RUNS / exp_name / "variant_E_per_seed_topK" / "MLP" / "test" / "results.csv")
        # Find corresponding D delta from earlier rows
        d_delta = next((r["delta_cm"] for r in csv_rows
                         if r["table"] == "backbone_removal" and r["experiment"] == exp_name), None)
        if baseline is None or cf is None:
            d_str = f"{d_delta:+.2f}" if d_delta is not None else "—"
            lines.append(f"| **{display}** | {f'{baseline:.2f}' if baseline else 'missing'} | pending | — | {d_str} | — |")
            continue
        delta = cf["RMSE_cm"] - baseline
        hurts = delta > RMSE_THRESHOLD_CM
        d_str = f"{d_delta:+.2f}" if d_delta is not None else "—"
        bigger = ""
        if d_delta is not None:
            e_vs_d_total += 1
            if abs(delta) > abs(d_delta):
                bigger = "✅ E > D"
                e_bigger_than_d += 1
            else:
                bigger = "D ≥ E"
        verdict_icon = "✅" if hurts else "⚠"
        total_e += 1
        if hurts: pass_e += 1
        lines.append(f"| **{display}** | {baseline:.2f} | {cf['RMSE_cm']:.2f} | {delta:+.2f} cm {verdict_icon} | {d_str} | {bigger} |")
        csv_rows.append(dict(table="per_seed_topK", experiment=exp_name, display=display,
                              original_rmse=baseline, perturbed_rmse=cf["RMSE_cm"],
                              delta_cm=delta, pass_=hurts))
    if total_e:
        lines += ["",
                  f"**Pass rate (E hurts RMSE): {pass_e} of {total_e}** ({100*pass_e/total_e:.0f}%).",
                  f"**E magnitude > D magnitude on {e_bigger_than_d} of {e_vs_d_total}** ({100*e_bigger_than_d/e_vs_d_total:.0f}%) experiments.",
                  "If per-seed Δ is consistently larger than ensemble Δ, the ensembling-dilutes hypothesis is supported.", ""]

    out_md = RUNS / "PERTURBATION_TABLE.md"
    out_csv = RUNS / "PERTURBATION_TABLE.csv"
    RUNS.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))
    pd.DataFrame(csv_rows).to_csv(out_csv, index=False)
    print(f"Wrote {out_md}\nWrote {out_csv}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
