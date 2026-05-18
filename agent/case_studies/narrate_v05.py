"""Narrate the v0.5 findings per case via the Narrator (Claude Opus 4.7).

Reads the API key from a file (path supplied via --key-file), runs the
Narrator on each case's `findings_v05.json` + context, and writes a
`diagnosis_v5.md` per case + a `narration_v05_usage.json` log.

The key is read once into memory, NEVER echoed to stdout, NEVER passed via
environment variables or argv. The Anthropic SDK call accepts it directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ffca_agent.evaluator import Finding, evaluate_rulebook, load_rulebook
from ffca_agent.llm import Narrator
from ffca_agent.report import ReportContext
from ffca_agent.training import TrainingHistory
from ffca_agent.vision import VisionMetrics


CASES = [
    "credit_loan",
    "california_housing_leak",
    "california_housing_spurious",
    "bike_sharing",
    "wine_quality",
    "waterbirds",
]


def _render(report) -> str:
    """Mirror the CLI's _render_narrated layout for the per-case markdown."""
    lines = [
        f"# FFCA Diagnosis (v0.5 rulebook, {report.model})",
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
    ap.add_argument("--key-file", required=True, help="Path to file containing the API key (one line).")
    ap.add_argument("--runs-dir", default="FFCA_runs_results_v04_real")
    ap.add_argument("--rulebook", default="rulebook/ffca_rules.yaml")
    ap.add_argument("--model", default="claude-opus-4-7")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--only", default=None, help="Comma-separated subset of cases.")
    args = ap.parse_args()

    key = Path(args.key_file).expanduser().read_text().strip()
    if not key.startswith("sk-"):
        print("WARN: key file content does not start with 'sk-'; proceeding anyway.", file=sys.stderr)

    rb = load_rulebook(args.rulebook)
    narrator = Narrator(model=args.model, api_key=key)
    runs = Path(args.runs_dir)

    only = set(args.only.split(",")) if args.only else None
    usage_log: list[dict] = []

    for case in CASES:
        if only and case not in only:
            continue
        d = runs / case
        if not (d / "report.json").exists():
            print(f"[skip] {case}: no report.json")
            continue

        ctx = ReportContext.from_json(d / "report.json")
        h = TrainingHistory.from_keras_history(d / "history.json")
        h.derive_from_signatures(ctx, top_k=args.top_k)
        ctx.attach_training_history(h)
        if (d / "vision_metrics.json").exists():
            ctx.attach_vision_metrics(VisionMetrics.from_json(d / "vision_metrics.json"))

        findings = evaluate_rulebook(rb, ctx)
        print(f"narrating {case} ({len(findings)} findings)...")
        report = narrator.narrate(findings, ctx, training=ctx.training)
        (d / "diagnosis_v5.md").write_text(_render(report))

        usage_log.append({
            "case": case,
            "n_findings": len(findings),
            "usage": report.usage,
        })
        print(f"  wrote {d/'diagnosis_v5.md'}  usage={report.usage}")

    (runs / "narration_v05_usage.json").write_text(json.dumps(usage_log, indent=2))
    print(f"\nwrote {runs/'narration_v05_usage.json'}")


if __name__ == "__main__":
    main()
