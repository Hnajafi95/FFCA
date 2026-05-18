"""Narrate the 8 Predicted Gate Opening flooding reports for the case study.

These are Keras-Tuner hypermodel reports (no per-epoch training history),
so dynamic rules silently skip — but the archetype / trust-bucket /
concentration / co-sensitivity rules apply cleanly. The agent narrates
the failure mode the manual analysis identified: gate-input channels
are flagged as marginal everywhere, and at long lead times pruning them
hurts skill.
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

REPORTS = [
    ("before", "3hr"),
    ("before", "6hr"),
    ("before", "12hr"),
    ("before", "24hr"),
    ("after", "3hr"),
    ("after", "6hr"),
    ("after", "12hr"),
    ("after", "24hr"),
]


def _report_path(when: str, lead: str) -> Path:
    base = (
        FLOOD_ROOT / "FFCA_resutls_before_prunning" if when == "before"
        else FLOOD_ROOT / "FFCA_results_After_prunning"
    )
    return base / "Predicted Gate Opening" / f"{lead}_perfect_prog_gate_sigmoid" / "report.json"


def _render(report) -> str:
    lines = [
        f"# Agent diagnosis — Predicted Gate Opening (v0.5 rulebook, {report.model})",
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
    args = ap.parse_args()

    key = Path(args.key_file).expanduser().read_text().strip()
    rb = load_rulebook(args.rulebook)
    narrator = Narrator(api_key=key)
    out = REPO / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    usage_log: list[dict] = []
    summary: dict[str, dict] = {}

    for when, lead in REPORTS:
        rp = _report_path(when, lead)
        if not rp.exists():
            print(f"[skip] {when}/{lead}: missing")
            continue
        ctx = ReportContext.from_json(rp)
        findings = evaluate_rulebook(rb, ctx)
        print(f"narrating {when}/{lead} ({ctx.n_features} feat, "
              f"{ctx.impact_curve.shape[0]} ckpts, {len(findings)} findings)...")
        report = narrator.narrate(findings, ctx, training=ctx.training)

        per_case_dir = out / f"{when}_{lead}"
        per_case_dir.mkdir(exist_ok=True)
        (per_case_dir / "diagnosis_v5.md").write_text(_render(report))
        diag = [f for f in findings if f.kind == "diagnostic"]
        (per_case_dir / "findings_v05.json").write_text(json.dumps({
            "when": when, "lead": lead,
            "n_features": ctx.n_features,
            "n_checkpoints": int(ctx.impact_curve.shape[0]),
            "trust_summary": {k: int(b.count) for k, b in ctx.trust_buckets.items()},
            "diagnostic_rule_ids": sorted({f.rule_id for f in diag}),
            "n_diagnostic": len(diag),
            "executive_summary": report.executive_summary,
        }, indent=2))
        usage_log.append({"when": when, "lead": lead, "usage": report.usage})
        summary[f"{when}/{lead}"] = {
            "diagnostic_rule_ids": sorted({f.rule_id for f in diag}),
            "n_findings": len(findings),
        }
        print(f"  wrote {per_case_dir/'diagnosis_v5.md'}  usage={report.usage}")

    (out / "narration_usage.json").write_text(json.dumps(usage_log, indent=2))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}/summary.json + narration_usage.json")


if __name__ == "__main__":
    main()
