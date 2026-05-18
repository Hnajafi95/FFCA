"""Sensitivity sweep of heuristic thresholds in the v0.5 rulebook.

For every rule whose `paper_ref` starts with "heuristic", enumerate numeric
trigger values and perturb each by [-50%, -25%, +25%, +50%] (4 points around
the baseline 0%). For each perturbation, re-run the rule evaluator across 14
reports (6 v0.5 case studies + 8 compound-flooding "Measurements Only"
before/after-pruning reports) and record whether the firing decision for
THAT rule on THAT report changed vs the baseline.

Output:
  - sensitivity_v05/raw.json      — per (rule, threshold, perturbation, report) decisions
  - sensitivity_v05/by_rule.json  — per (rule, threshold) fragility scores
  - sensitivity_v05/SENSITIVITY.md — human-readable summary

Fragility score per threshold = fraction of (report × perturbation) pairs whose
firing decision differs from the baseline. 0 = perfectly robust; 1 = every
perturbation changes every report's decision.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import yaml

from ffca_agent.evaluator import evaluate_rulebook
from ffca_agent.report import ReportContext
from ffca_agent.training import TrainingHistory
from ffca_agent.vision import VisionMetrics


PERTURBATIONS = [-0.50, -0.25, 0.0, 0.25, 0.50]

V05_CASES = [
    "credit_loan",
    "california_housing_leak",
    "california_housing_spurious",
    "bike_sharing",
    "wine_quality",
    "waterbirds",
]


def _enumerate_heuristic_thresholds(rulebook: dict) -> list[tuple[str, int, str]]:
    """Return [(rule_id, trigger_idx, jsonpath), ...] for each heuristic numeric threshold.

    jsonpath is "value" for scalar triggers, or "value.<key>" for dict-form triggers
    (e.g., spike_detected with {threshold_ratio: 1.4}).
    """
    out: list[tuple[str, int, str]] = []
    for r in rulebook["rules"]:
        if "heuristic" not in r.get("paper_ref", "").lower():
            continue
        for ti, t in enumerate(r.get("triggers", [])):
            v = t.get("value")
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                out.append((r["id"], ti, "value"))
            elif isinstance(v, dict):
                for k, vv in v.items():
                    if isinstance(vv, (int, float)) and not isinstance(vv, bool):
                        out.append((r["id"], ti, f"value.{k}"))
    return out


def _set_threshold(rulebook: dict, rule_id: str, ti: int, path: str, new_value: float) -> None:
    rule = next(r for r in rulebook["rules"] if r["id"] == rule_id)
    trig = rule["triggers"][ti]
    if path == "value":
        trig["value"] = new_value
    else:
        key = path.removeprefix("value.")
        trig["value"][key] = new_value


def _get_threshold(rulebook: dict, rule_id: str, ti: int, path: str) -> float:
    rule = next(r for r in rulebook["rules"] if r["id"] == rule_id)
    trig = rule["triggers"][ti]
    if path == "value":
        return float(trig["value"])
    key = path.removeprefix("value.")
    return float(trig["value"][key])


def _load_v05_ctx(d: Path) -> ReportContext:
    ctx = ReportContext.from_json(d / "report.json")
    if (d / "history.json").exists():
        h = TrainingHistory.from_keras_history(d / "history.json")
        h.derive_from_signatures(ctx, top_k=5)
        ctx.attach_training_history(h)
    if (d / "vision_metrics.json").exists():
        ctx.attach_vision_metrics(VisionMetrics.from_json(d / "vision_metrics.json"))
    return ctx


def _load_flooding_ctx(report_path: Path) -> ReportContext:
    # Flooding reports have no Keras training history — checkpoints correspond to
    # Keras-Tuner candidate hypermodels, not training epochs. The dynamic rules
    # would skip on these even if we attached a curve, which is the honest
    # behaviour we want.
    return ReportContext.from_json(report_path)


def _rule_fired(rule_id: str, ctx: ReportContext, rulebook: dict) -> bool:
    findings = evaluate_rulebook(rulebook, ctx)
    return any(f.rule_id == rule_id for f in findings)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(REPO))
    ap.add_argument("--v05-dir", default="FFCA_runs_results_v04_real")
    ap.add_argument("--flooding-root", default="/Users/hnaja002/Documents/projects/compound_flooding")
    ap.add_argument("--rulebook", default="rulebook/ffca_rules.yaml")
    ap.add_argument("--out-dir", default="FFCA_runs_results_v04_real/sensitivity_v05")
    args = ap.parse_args()

    repo = Path(args.repo)
    out = repo / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    with open(repo / args.rulebook) as f:
        baseline_rb = yaml.safe_load(f)

    # Build the report inventory
    reports: list[tuple[str, ReportContext]] = []
    v05_dir = repo / args.v05_dir
    for c in V05_CASES:
        d = v05_dir / c
        if (d / "report.json").exists():
            reports.append((f"v05/{c}", _load_v05_ctx(d)))

    flooding_root = Path(args.flooding_root)
    for prefix, group in [
        ("before", "FFCA_resutls_before_prunning/Measurements Only"),
        ("after", "FFCA_results_After_prunning/Measurements Only"),
    ]:
        for lead in ["3hr_measured_sigmoid", "6hr_measured_sigmoid",
                     "12hr_measured_sigmoid", "24hr_measured_sigmoid"]:
            p = flooding_root / group / lead / "report.json"
            if p.exists():
                reports.append((f"flood/{prefix}/{lead}", _load_flooding_ctx(p)))

    print(f"Loaded {len(reports)} reports")

    # Enumerate heuristic thresholds
    thresholds = _enumerate_heuristic_thresholds(baseline_rb)
    print(f"Found {len(thresholds)} heuristic numeric thresholds across "
          f"{len({r for r, _, _ in thresholds})} rules")

    raw: list[dict] = []
    by_rule: dict[str, dict] = {}

    for rule_id, ti, path in thresholds:
        baseline_value = _get_threshold(baseline_rb, rule_id, ti, path)
        # Note: some thresholds are tiny (1e-6), perturbing them by ±50% might cross zero;
        # we still report — flipping at low magnitudes is itself informative.
        for pct in PERTURBATIONS:
            rb = copy.deepcopy(baseline_rb)
            new_value = baseline_value * (1 + pct)
            _set_threshold(rb, rule_id, ti, path, new_value)
            for report_label, ctx in reports:
                fired = _rule_fired(rule_id, ctx, rb)
                raw.append({
                    "rule_id": rule_id,
                    "trigger_idx": ti,
                    "threshold_path": path,
                    "baseline_value": baseline_value,
                    "perturbation_pct": pct,
                    "perturbed_value": new_value,
                    "report": report_label,
                    "fired": fired,
                })

    # Compute per-(rule, threshold) fragility = fraction of (report × non-baseline-perturbation)
    # pairs whose firing decision differs from the baseline (0%) firing decision for the
    # same report. Higher = more sensitive.
    baseline_decisions: dict[tuple[str, int, str, str], bool] = {}
    for row in raw:
        if row["perturbation_pct"] == 0.0:
            key = (row["rule_id"], row["trigger_idx"], row["threshold_path"], row["report"])
            baseline_decisions[key] = row["fired"]

    for row in raw:
        key = (row["rule_id"], row["trigger_idx"], row["threshold_path"])
        thr_key = ".".join(map(str, key))
        if thr_key not in by_rule:
            by_rule[thr_key] = {
                "rule_id": row["rule_id"],
                "trigger_idx": row["trigger_idx"],
                "threshold_path": row["threshold_path"],
                "baseline_value": row["baseline_value"],
                "flips_per_perturbation": {str(p): 0 for p in PERTURBATIONS},
                "total_flips": 0,
                "n_reports": len(reports),
                "flipped_reports": {str(p): [] for p in PERTURBATIONS},
            }
        if row["perturbation_pct"] == 0.0:
            continue
        bkey = (row["rule_id"], row["trigger_idx"], row["threshold_path"], row["report"])
        baseline_fired = baseline_decisions[bkey]
        if row["fired"] != baseline_fired:
            by_rule[thr_key]["flips_per_perturbation"][str(row["perturbation_pct"])] += 1
            by_rule[thr_key]["total_flips"] += 1
            by_rule[thr_key]["flipped_reports"][str(row["perturbation_pct"])].append(
                {"report": row["report"], "from": baseline_fired, "to": row["fired"]}
            )

    n_non_baseline = len(PERTURBATIONS) - 1  # 4 perturbations
    for v in by_rule.values():
        v["fragility"] = v["total_flips"] / (n_non_baseline * v["n_reports"])

    (out / "raw.json").write_text(json.dumps(raw, indent=2))
    (out / "by_rule.json").write_text(json.dumps(by_rule, indent=2))
    print(f"Wrote {out/'raw.json'} ({len(raw)} rows) and {out/'by_rule.json'} ({len(by_rule)} thresholds)")


if __name__ == "__main__":
    main()
