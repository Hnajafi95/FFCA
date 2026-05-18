"""Re-narrate 14 representative cases with v0.6 features turned on.

Compared to v0.5 narrations, each call now includes:
  - a per-project case_meta (model arch, task, target, domain)
  - a per-narration intent (audit / diagnose / prune / compare)
  - the rule-free-observation channel via signature_summary

Outputs each case's `diagnosis_v6.md` alongside the existing `diagnosis_v5.md`
so the diff script can compare them directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ffca_agent.case_meta import (  # noqa: E402
    CaseMeta,
    ModelArchitecture,
    NarrationIntent,
    TaskType,
)
from ffca_agent.evaluator import evaluate_rulebook, load_rulebook  # noqa: E402
from ffca_agent.llm import Narrator  # noqa: E402
from ffca_agent.report import ReportContext  # noqa: E402
from ffca_agent.signature_summary import signature_summary  # noqa: E402
from ffca_agent.training import TrainingHistory  # noqa: E402
from ffca_agent.vision import VisionMetrics  # noqa: E402


# ── per-case configuration ──────────────────────────────────────────────────


def _v05_case_dir(name: str) -> Path:
    return REPO / "FFCA_runs_results_v04_real" / name


def _flooding_dir(category: str, when: str, lead: int) -> Path:
    """Returns the directory containing the v0.5 narration for a flooding case."""
    return REPO / "FFCA_runs_results_v04_real/flooding_narrations" / category / f"{when}_{lead}hr"


def _flooding_report(category: str, when: str, lead: int) -> Path:
    base = (Path("/Users/hnaja002/Documents/projects/compound_flooding")
            / ("FFCA_resutls_before_prunning" if when == "before"
               else "FFCA_results_After_prunning"))
    dirname = f"{lead}hr_measured_sigmoid" if category == "measured" \
        else f"{lead}hr_perfect_prog_{category}_sigmoid"
    cat_label = {
        "measured": "Measurements Only",
        "wls": "Predicted Ocean Water Levels",
        "rain": "Predicted Rainfall",
        "gate": "Predicted Gate Opening",
        "all_inputs": "Predictions All Inputs",
    }[category]
    return base / cat_label / dirname / "report.json"


# Each entry: (label, report_path, case_meta, intent, out_dir)
def _v06_plan() -> list[dict]:
    # 6 engineered v0.5 cases — each gets its own case_meta because the
    # tasks and pathologies differ. Intent = "diagnose" (these were designed
    # to surface specific pathologies — the user wants the root cause).
    plan: list[dict] = [
        {
            "label": "v05/credit_loan",
            "report": _v05_case_dir("credit_loan") / "report.json",
            "history": _v05_case_dir("credit_loan") / "history.json",
            "vision": None,
            "out": _v05_case_dir("credit_loan"),
            "case_meta": CaseMeta(
                project_name="credit_loan_v05_gate",
                model_architecture=ModelArchitecture.MLP,
                task_type=TaskType.BINARY_CLASSIFICATION,
                target_name="credit_risk",
                target_units="",
                domain="financial-risk modelling (UCI German Credit)",
                notes="Engineered to exhibit hierarchical-learning staging",
            ),
            "intent": NarrationIntent.DIAGNOSE,
        },
        {
            "label": "v05/california_housing_leak",
            "report": _v05_case_dir("california_housing_leak") / "report.json",
            "history": _v05_case_dir("california_housing_leak") / "history.json",
            "vision": None,
            "out": _v05_case_dir("california_housing_leak"),
            "case_meta": CaseMeta(
                project_name="california_housing_leak_v05_gate",
                model_architecture=ModelArchitecture.MLP,
                task_type=TaskType.REGRESSION,
                target_name="median_house_value",
                target_units="$100k",
                domain="real-estate price prediction",
                feature_naming_convention="`leaked_target` is a deliberate noisy copy of the target",
                notes="Engineered with target leakage",
            ),
            "intent": NarrationIntent.DIAGNOSE,
        },
        {
            "label": "v05/california_housing_spurious",
            "report": _v05_case_dir("california_housing_spurious") / "report.json",
            "history": _v05_case_dir("california_housing_spurious") / "history.json",
            "vision": None,
            "out": _v05_case_dir("california_housing_spurious"),
            "case_meta": CaseMeta(
                project_name="california_housing_spurious_v05_gate",
                model_architecture=ModelArchitecture.MLP,
                task_type=TaskType.REGRESSION,
                target_name="median_house_value",
                target_units="$100k",
                domain="real-estate price prediction",
                feature_naming_convention="`spurious_feature` is correlated with target only in training",
                notes="Engineered with a train-only spurious correlation",
            ),
            "intent": NarrationIntent.DIAGNOSE,
        },
        {
            "label": "v05/bike_sharing",
            "report": _v05_case_dir("bike_sharing") / "report.json",
            "history": _v05_case_dir("bike_sharing") / "history.json",
            "vision": None,
            "out": _v05_case_dir("bike_sharing"),
            "case_meta": CaseMeta(
                project_name="bike_sharing_v05_gate",
                model_architecture=ModelArchitecture.MLP,
                task_type=TaskType.REGRESSION,
                target_name="bike_rentals",
                target_units="rides/hour",
                domain="urban mobility forecasting (UCI Bike Sharing)",
                notes="Wide MLP + long training to provoke overfitting",
            ),
            "intent": NarrationIntent.DIAGNOSE,
        },
        {
            "label": "v05/wine_quality",
            "report": _v05_case_dir("wine_quality") / "report.json",
            "history": _v05_case_dir("wine_quality") / "history.json",
            "vision": None,
            "out": _v05_case_dir("wine_quality"),
            "case_meta": CaseMeta(
                project_name="wine_quality_v05_gate",
                model_architecture=ModelArchitecture.OTHER,
                task_type=TaskType.REGRESSION,
                target_name="wine_quality_score",
                target_units="",
                domain="food chemistry (UCI Wine Quality, red)",
                notes="Pure linear regression baseline — Nonlinearity must be 0",
            ),
            "intent": NarrationIntent.DIAGNOSE,
        },
        {
            "label": "v05/waterbirds",
            "report": _v05_case_dir("waterbirds") / "report.json",
            "history": _v05_case_dir("waterbirds") / "history.json",
            "vision": _v05_case_dir("waterbirds") / "vision_metrics.json",
            "out": _v05_case_dir("waterbirds"),
            "case_meta": CaseMeta(
                project_name="waterbirds_v05_gate",
                model_architecture=ModelArchitecture.CNN,
                task_type=TaskType.VISION_CLASSIFICATION,
                target_name="bird_class",
                target_units="",
                domain="vision shortcut-learning benchmark (WILDS Waterbirds)",
                pretrained=True,
                notes="Engineered to expose shortcut learning on majority/minority groups",
            ),
            "intent": NarrationIntent.AUDIT,
        },
    ]

    # 8 flooding gate cases — same shared case_meta, intent COMPARE so the
    # before/after framing is explicit.
    flooding_meta = CaseMeta(
        project_name="compound_flooding_predicted_gate",
        model_architecture=ModelArchitecture.MLP,
        task_type=TaskType.REGRESSION,
        target_name="water_level",
        target_units="cm",
        domain="coastal compound flooding forecasting (MLMiami / TAMUCC)",
        feature_naming_convention=(
            "`*_t-k` are lagged measurements (k hours back); "
            "`*_t+k` are forecast inputs (k hours ahead). "
            "`gate*` channels are noisy proxies for drainage-gate state."
        ),
        notes="Each report is one of 4 lead times × {before, after} pruning.",
    )
    for lead in [3, 6, 12, 24]:
        for when in ["before", "after"]:
            plan.append({
                "label": f"flooding/gate/{when}_{lead}hr",
                "report": _flooding_report("gate", when, lead),
                "history": None,
                "vision": None,
                "out": _flooding_dir("gate", when, lead),
                "case_meta": flooding_meta,
                "intent": NarrationIntent.COMPARE,
            })
    return plan


def _render(report) -> str:
    lines = [
        f"# FFCA Diagnosis (v0.6 rulebook, {report.model})",
        "",
        "## Executive summary",
        "",
        report.executive_summary.strip(),
        "",
        "## Ranked actions",
        "",
    ]
    for a in sorted(report.actions, key=lambda x: x.priority):
        rules = ", ".join(a.rule_ids) if a.rule_ids else "—"
        lines.append(f"{a.priority}. **{a.title}**")
        lines.append(f"   {a.rationale.strip()}")
        lines.append(f"   _(from: {rules})_")
        lines.append("")
    if report.rule_free_observations:
        lines.append("## Rule-free observations")
        lines.append("")
        for o in report.rule_free_observations:
            lines.append(f"- **{o.what.strip()}** — _evidence:_ {o.evidence.strip()}")
        lines.append("")
    if report.caveats:
        lines.append("## Caveats")
        lines.append("")
        for c in report.caveats:
            lines.append(f"- {c}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--rulebook", default="rulebook/ffca_rules.yaml")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Don't re-narrate cases that already have diagnosis_v6.md.")
    ap.add_argument("--only", default=None,
                    help="Comma-separated subset of case labels (e.g., v05/wine_quality,flooding/gate/after_24hr).")
    args = ap.parse_args()

    key = Path(args.key_file).expanduser().read_text().strip()
    rb = load_rulebook(args.rulebook)
    narrator = Narrator(api_key=key)

    plan = _v06_plan()
    if args.only:
        keep = set(args.only.split(","))
        plan = [p for p in plan if p["label"] in keep]

    usage_log: list[dict] = []
    summary: dict[str, dict] = {}

    for entry in plan:
        label = entry["label"]
        rp = entry["report"]
        if not rp.exists():
            print(f"[skip-missing] {label}: {rp}")
            continue
        out_dir = entry["out"]
        out_dir.mkdir(parents=True, exist_ok=True)
        md_path = out_dir / "diagnosis_v6.md"
        if args.skip_existing and md_path.exists():
            print(f"[skip-existing] {label}")
            continue

        ctx = ReportContext.from_json(rp)
        if entry.get("history") and entry["history"].exists():
            h = TrainingHistory.from_keras_history(entry["history"])
            h.derive_from_signatures(ctx, top_k=5)
            ctx.attach_training_history(h)
        if entry.get("vision") and entry["vision"].exists():
            ctx.attach_vision_metrics(VisionMetrics.from_json(entry["vision"]))

        findings = evaluate_rulebook(rb, ctx)
        sig_summary = signature_summary(ctx, top_k=5)

        print(f"narrating {label} ({ctx.n_features} feat, {len(findings)} findings, "
              f"intent={entry['intent'].value})...")
        report = narrator.narrate(
            findings, ctx,
            training=ctx.training or None,
            case_meta=entry["case_meta"],
            intent=entry["intent"],
            sig_summary=sig_summary,
        )
        md_path.write_text(_render(report))
        (out_dir / "findings_v06.json").write_text(json.dumps({
            "label": label,
            "n_features": ctx.n_features,
            "n_checkpoints": int(ctx.impact_curve.shape[0]),
            "diagnostic_rule_ids": sorted({f.rule_id for f in findings if f.kind == "diagnostic"}),
            "n_diagnostic": sum(1 for f in findings if f.kind == "diagnostic"),
            "executive_summary": report.executive_summary,
            "actions": [
                {"priority": a.priority, "title": a.title, "rule_ids": a.rule_ids}
                for a in report.actions
            ],
            "rule_free_observations": [
                {"what": o.what, "evidence": o.evidence}
                for o in report.rule_free_observations
            ],
            "n_caveats": len(report.caveats),
            "intent": entry["intent"].value,
        }, indent=2))
        usage_log.append({"label": label, "usage": report.usage})
        summary[label] = {
            "n_findings": len(findings),
            "n_rule_free_observations": len(report.rule_free_observations),
            "intent": entry["intent"].value,
        }
        print(f"  wrote {md_path}  usage={report.usage}  "
              f"obs={len(report.rule_free_observations)}")

    out_root = REPO / "FFCA_runs_results_v04_real"
    (out_root / "narration_v06_usage.json").write_text(json.dumps(usage_log, indent=2))
    (out_root / "summary_v06.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_root/'summary_v06.json'} + narration_v06_usage.json")


if __name__ == "__main__":
    main()
