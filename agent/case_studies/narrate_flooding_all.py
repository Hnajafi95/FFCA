"""Narrate the remaining 32 compound-flooding reports with v0.5 rulebook.

The gate-only round (`narrate_flooding.py`) already covered 8 reports. This
script covers the other 4 input categories × 4 lead times × {before, after}.

Skips reports whose `diagnosis_v5.md` already exists, so re-runs are
idempotent and you can resume after an interruption.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ffca_agent.evaluator import evaluate_rulebook, load_rulebook  # noqa: E402
from ffca_agent.llm import Narrator  # noqa: E402
from ffca_agent.report import ReportContext  # noqa: E402

FLOOD_ROOT = Path("/Users/hnaja002/Documents/projects/compound_flooding")

# (category_label, category_short, dirname_template)
CATEGORIES = [
    ("Measurements Only",            "measured",    "{lead}hr_measured_sigmoid"),
    ("Predicted Ocean Water Levels", "wls",         "{lead}hr_perfect_prog_wls_sigmoid"),
    ("Predicted Rainfall",           "rain",        "{lead}hr_perfect_prog_rain_sigmoid"),
    ("Predicted Gate Opening",       "gate",        "{lead}hr_perfect_prog_gate_sigmoid"),
    ("Predictions All Inputs",       "all_inputs",  "{lead}hr_perfect_prog_all_inputs_sigmoid"),
]
LEAD_TIMES = [3, 6, 12, 24]


def _report_path(category_label: str, dirname_tpl: str, lead: int, when: str) -> Path:
    base = (
        FLOOD_ROOT / "FFCA_resutls_before_prunning" if when == "before"
        else FLOOD_ROOT / "FFCA_results_After_prunning"
    )
    return base / category_label / dirname_tpl.format(lead=lead) / "report.json"


def _render(report) -> str:
    lines = [
        f"# Agent diagnosis (v0.5 rulebook, {report.model})",
        "",
        "## Executive summary",
        "",
        report.executive_summary.strip(),
        "",
        "## Ranked actions",
        "",
    ]
    for a in report.actions:
        rules = ", ".join(a.rule_ids) if a.rule_ids else "—"
        lines.append(f"{a.priority}. **{a.title}**")
        lines.append(f"   {a.rationale.strip()}")
        lines.append(f"   _(from: {rules})_")
        lines.append("")
    if report.caveats:
        lines.append("## Caveats")
        lines.append("")
        for c in report.caveats:
            lines.append(f"- {c}")
        lines.append("")
    lines.append("## Findings appendix")
    lines.append("")
    for f in report.appendix_findings:
        if f.kind != "diagnostic":
            continue
        lines.append(f"### [{f.severity or '-'}] {f.rule_id} — {f.rule_name}")
        if f.feature:
            lines.append(f"_feature: {f.feature}_")
        lines.append("")
        lines.append(f"**Diagnosis.** {f.diagnosis}")
        lines.append("")
        if f.recommendation:
            lines.append(f"**Recommendation.** {f.recommendation}")
            lines.append("")
        if f.evidence:
            lines.append(f"**Evidence.** {f.evidence}")
            lines.append("")
        if f.paper_ref:
            lines.append(f"_paper ref: {f.paper_ref}_")
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--rulebook", default="rulebook/ffca_rules.yaml")
    ap.add_argument("--out-dir", default="FFCA_runs_results_v04_real/flooding_narrations")
    ap.add_argument("--skip-existing", action="store_true", default=True,
                    help="Skip reports whose diagnosis_v5.md already exists.")
    args = ap.parse_args()

    key = Path(args.key_file).expanduser().read_text().strip()
    rb = load_rulebook(args.rulebook)
    narrator = Narrator(api_key=key)
    out = REPO / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    # Build full work list
    plan: list[tuple[str, str, str, int, str]] = []
    for cat_label, cat_short, dirname_tpl in CATEGORIES:
        for lead in LEAD_TIMES:
            for when in ["before", "after"]:
                plan.append((cat_label, cat_short, dirname_tpl, lead, when))

    # Pre-existing gate narrations from the earlier round live in
    # out/{when}_{lead}/ — migrate them into the new structure if present.
    legacy_dirs: list[Path] = []
    for when in ["before", "after"]:
        for lead in LEAD_TIMES:
            legacy_dirs.append(out / f"{when}_{lead}hr")

    usage_log: list[dict] = []
    summary: dict[str, dict] = {}

    for cat_label, cat_short, dirname_tpl, lead, when in plan:
        rp = _report_path(cat_label, dirname_tpl, lead, when)
        if not rp.exists():
            print(f"[skip-missing] {cat_short}/{when}/{lead}hr: {rp}")
            continue

        per_case_dir = out / cat_short / f"{when}_{lead}hr"
        per_case_dir.mkdir(parents=True, exist_ok=True)
        md_path = per_case_dir / "diagnosis_v5.md"

        # Check legacy gate location for this combo and migrate if found
        if cat_short == "gate":
            legacy = out / f"{when}_{lead}hr"
            if legacy.exists() and (legacy / "diagnosis_v5.md").exists() and not md_path.exists():
                (legacy / "diagnosis_v5.md").rename(md_path)
                if (legacy / "findings_v05.json").exists():
                    (legacy / "findings_v05.json").rename(per_case_dir / "findings_v05.json")
                try:
                    legacy.rmdir()
                except OSError:
                    pass
                print(f"[migrate] moved legacy gate/{when}_{lead}hr → {per_case_dir}")

        if args.skip_existing and md_path.exists():
            # Re-collect summary info from the cached findings JSON if available
            fj = per_case_dir / "findings_v05.json"
            if fj.exists():
                rec = json.loads(fj.read_text())
                summary[f"{cat_short}/{when}/{lead}hr"] = {
                    "diagnostic_rule_ids": rec.get("diagnostic_rule_ids", []),
                    "n_findings": rec.get("n_diagnostic", 0),
                }
            print(f"[skip-existing] {cat_short}/{when}/{lead}hr")
            continue

        ctx = ReportContext.from_json(rp)
        findings = evaluate_rulebook(rb, ctx)
        print(f"narrating {cat_short}/{when}/{lead}hr "
              f"({ctx.n_features} feat, {ctx.impact_curve.shape[0]} ckpts, "
              f"{len(findings)} findings)...")
        report = narrator.narrate(findings, ctx, training=ctx.training)
        md_path.write_text(_render(report))
        diag = [f for f in findings if f.kind == "diagnostic"]
        (per_case_dir / "findings_v05.json").write_text(json.dumps({
            "category": cat_short, "when": when, "lead_h": lead,
            "n_features": ctx.n_features,
            "n_checkpoints": int(ctx.impact_curve.shape[0]),
            "trust_summary": {k: int(b.count) for k, b in ctx.trust_buckets.items()},
            "diagnostic_rule_ids": sorted({f.rule_id for f in diag}),
            "n_diagnostic": len(diag),
            "executive_summary": report.executive_summary,
        }, indent=2))
        usage_log.append({"category": cat_short, "when": when, "lead": lead,
                          "usage": report.usage})
        summary[f"{cat_short}/{when}/{lead}hr"] = {
            "diagnostic_rule_ids": sorted({f.rule_id for f in diag}),
            "n_findings": len(findings),
        }
        print(f"  wrote {md_path}  usage={report.usage}")

    (out / "narration_usage.json").write_text(json.dumps(usage_log, indent=2))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}/summary.json (+ usage log)")


if __name__ == "__main__":
    main()
