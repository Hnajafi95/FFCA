"""CLI: run the FFCA rulebook against a report.json and emit findings.

Usage:
    python -m ffca_agent.cli REPORT_JSON [--rulebook PATH] [--format json|md]
        [--training-history HISTORY_FILE] [--epochs-aligned]
        [--narrate] [--model MODEL_ID]

With --narrate, the deterministic findings are passed to Claude for a layered
diagnosis (executive summary + ranked actions + caveats), prepended to the
appendix of full rule output. Requires ANTHROPIC_API_KEY. Without the key (or
without --narrate), the CLI prints deterministic-only output.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .evaluator import evaluate_rulebook, load_rulebook, summarize
from .report import ReportContext
from .training import TrainingHistory

DEFAULT_RULEBOOK = Path(__file__).resolve().parents[1] / "rulebook" / "ffca_rules.yaml"


def _render_markdown(report_path: Path, findings, summary) -> str:
    lines = [f"# FFCA Agent Diagnosis — `{report_path.name}`", ""]
    lines.append(f"_{summary['n_findings']} findings "
                 f"({summary['n_diagnostic']} diagnostic, {summary['n_descriptor']} descriptor)._")
    lines.append("")
    if summary["diagnostic_by_severity"]:
        lines.append("**Severity breakdown:** " +
                     ", ".join(f"{k}: {v}" for k, v in summary["diagnostic_by_severity"].items()))
        lines.append("")

    # diagnostic findings first, sorted by severity
    sev_order = {"critical": 0, "warn": 1, "info": 2, "—": 3, None: 3}
    diag = sorted([f for f in findings if f.kind == "diagnostic"],
                  key=lambda f: sev_order.get(f.severity, 3))
    if diag:
        lines.append("## Diagnostic findings")
        for f in diag:
            sev = (f.severity or "info").upper()
            scope = f"feature `{f.feature}`" if f.feature else f"model-wide"
            lines.append(f"### `{f.rule_id}` · **{sev}** · {scope}")
            lines.append(f"_{f.rule_name} — {f.category}_")
            lines.append("")
            lines.append(f"**Diagnosis.** {f.diagnosis}")
            lines.append("")
            if f.recommendation:
                lines.append(f"**Recommendation.** {f.recommendation}")
                lines.append("")
            if f.evidence:
                lines.append(f"**Evidence.** {f.evidence}")
                lines.append("")
            if f.confidence_factors:
                lines.append("**Caveats:**")
                for c in f.confidence_factors:
                    lines.append(f"- {c}")
                lines.append("")
            lines.append(f"_Paper ref: {f.paper_ref}_")
            lines.append("")

    desc = [f for f in findings if f.kind == "descriptor"]
    if desc:
        lines.append("## Descriptors (feature roles + state labels)")
        # group by rule_id; per-feature descriptors get a compact roll-up
        by_rule: dict[str, list] = {}
        for f in desc:
            by_rule.setdefault(f.rule_id, []).append(f)
        # sort by category for stable ordering
        order = sorted(by_rule.items(), key=lambda kv: (kv[1][0].category, kv[0]))
        for rule_id, group in order:
            sample = group[0]
            scope_note = (f"{len(group)} features" if sample.feature
                          else "model-wide")
            lines.append(f"### `{rule_id}` · {sample.rule_name} · _{scope_note}_")
            lines.append("")
            if sample.feature:
                # per-feature descriptor: list the names compactly, then show
                # the shared diagnosis/recommendation text
                names = ", ".join(f"`{g.feature}`" for g in group[:10])
                if len(group) > 10:
                    names += f" (+{len(group) - 10} more)"
                lines.append(f"_Affected features:_ {names}")
                lines.append("")
            lines.append(f"**Diagnosis.** {sample.diagnosis}")
            lines.append("")
            if sample.recommendation:
                lines.append(f"**Recommendation.** {sample.recommendation}")
                lines.append("")
            if sample.evidence:
                lines.append(f"**Evidence.** {sample.evidence}")
                lines.append("")
            if sample.confidence_factors:
                lines.append("**Caveats:**")
                for c in sample.confidence_factors:
                    lines.append(f"- {c}")
                lines.append("")
            lines.append(f"_Paper ref: {sample.paper_ref}_")
            lines.append("")

    return "\n".join(lines)


def _render_narrated(report_path: Path, narrated, deterministic_md: str) -> str:
    out = [f"# FFCA Agent Diagnosis — `{report_path.name}`", ""]
    out.append("## Executive summary")
    out.append("")
    out.append(narrated.executive_summary)
    out.append("")
    if narrated.actions:
        out.append("## Ranked actions")
        out.append("")
        for a in sorted(narrated.actions, key=lambda x: x.priority):
            ids = ", ".join(f"`{r}`" for r in a.rule_ids) if a.rule_ids else ""
            ids_tail = f"  _(from: {ids})_" if ids else ""
            out.append(f"{a.priority}. **{a.title}.** {a.rationale}{ids_tail}")
        out.append("")
    if narrated.rule_free_observations:
        out.append("---")
        out.append("")
        out.append("## :warning: Rule-free observations (LLM-generated, NOT rule-backed)")
        out.append("")
        out.append("> **Trust level: lower than the findings above.** The observations in this")
        out.append("> section come from the LLM examining a structured summary of the")
        out.append("> signatures — they are NOT verified by a deterministic rule. Each one")
        out.append("> cites a specific summary value, so the *quote* is grounded, but the")
        out.append("> *inference* the LLM drew from that value may still be wrong.")
        out.append(">")
        out.append("> Treat as hypotheses to check, not as conclusions. The ranked actions")
        out.append("> and findings appendix above are the authoritative output.")
        out.append("")
        for o in narrated.rule_free_observations:
            out.append(f"- **{o.what.strip()}** — _evidence:_ {o.evidence.strip()}")
        out.append("")
    if narrated.caveats:
        out.append("## Caveats")
        out.append("")
        for c in narrated.caveats:
            out.append(f"- {c}")
        out.append("")
    out.append("## Appendix — full rule output")
    out.append("")
    # The deterministic markdown already starts with its own h1; strip it so
    # we don't duplicate the title.
    det = deterministic_md.split("\n", 2)[-1] if deterministic_md.startswith("# ") else deterministic_md
    out.append(det)
    # Footer with model + usage so the paper supplement can cite it
    if narrated.usage:
        usage_str = ", ".join(f"{k}={v}" for k, v in narrated.usage.items())
        out.append("")
        out.append(f"_Narrated by {narrated.model}; usage: {usage_str}_")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report", type=Path, help="Path to FFCA report.json")
    p.add_argument("--rulebook", type=Path, default=DEFAULT_RULEBOOK,
                   help=f"Rulebook YAML (default: {DEFAULT_RULEBOOK})")
    p.add_argument("--format", choices=["json", "md", "summary"], default="md")
    p.add_argument("--training-history", type=Path, default=None,
                   help="Optional Keras-style history.json/.csv. Unlocks training-dynamics rules.")
    p.add_argument("--epochs-aligned", action="store_true",
                   help="Assert that FFCA checkpoints correspond to training epochs. "
                        "Derives volatility/impact/interaction per-epoch curves from the report.")
    p.add_argument("--topk", type=int, default=5,
                   help="Top-k features for impact/interaction aggregation (default 5).")
    p.add_argument("--vision-metrics", type=Path, default=None,
                   help="Optional vision_metrics.json (FBR / COM / minority-acc curves). "
                        "Unlocks the shortcut_learning_drift_epoch rule.")
    p.add_argument("--narrate", action="store_true",
                   help="Layer an LLM narration (exec summary + ranked actions) on top of the "
                        "deterministic findings. Requires ANTHROPIC_API_KEY.")
    p.add_argument("--model", default=None,
                   help="Claude model id for narration (default: claude-opus-4-7).")
    # v0.6: case context + intent + rule-free observation channel
    p.add_argument("--case-meta", type=Path, default=None,
                   help="Optional case_meta.json (model arch, task, domain, etc.) — "
                        "templated into the narrator's system prompt for more on-target "
                        "narration. Build one with --questionnaire.")
    p.add_argument("--intent", choices=["audit", "diagnose", "prune", "compare", "free"],
                   default=None,
                   help="Per-narration framing slotted into the system prompt: audit / "
                        "diagnose / prune / compare / free.")
    p.add_argument("--with-signature-summary", action="store_true",
                   help="Include the rule-free observation channel (top-K features, curve "
                        "shapes, churn counts) so the LLM can surface patterns no rule covers.")
    p.add_argument("--questionnaire", type=Path, default=None, metavar="OUTFILE",
                   help="Interactively build a case_meta.json (writes to OUTFILE and exits).")
    args = p.parse_args(argv)

    # Questionnaire mode short-circuits everything else
    if args.questionnaire is not None:
        from .case_meta import CaseMeta
        existing = None
        if args.questionnaire.exists():
            try:
                existing = CaseMeta.from_json(args.questionnaire)
                print(f"Editing existing case_meta at {args.questionnaire}.")
            except Exception:
                pass
        meta = CaseMeta.from_questionnaire(existing=existing)
        meta.save(args.questionnaire)
        print(f"Saved case_meta to {args.questionnaire}")
        return 0

    ctx = ReportContext.from_json(args.report)

    if args.training_history is not None:
        hist = TrainingHistory.from_keras_history(args.training_history)
    elif args.epochs_aligned:
        hist = TrainingHistory()
    else:
        hist = None
    if hist is not None:
        if args.epochs_aligned:
            hist.derive_from_signatures(ctx, top_k=args.topk)
        ctx.attach_training_history(hist)

    if args.vision_metrics is not None:
        from .vision import VisionMetrics
        ctx.attach_vision_metrics(VisionMetrics.from_json(args.vision_metrics))

    rulebook = load_rulebook(args.rulebook)
    findings = evaluate_rulebook(rulebook, ctx)
    summary = summarize(findings)

    if args.format == "summary":
        print(json.dumps(summary, indent=2))
        return 0
    if args.format == "json":
        print(json.dumps({
            "report": str(args.report),
            "summary": summary,
            "findings": [f.to_dict() for f in findings],
        }, indent=2, default=str))
        return 0

    deterministic_md = _render_markdown(args.report, findings, summary)

    if args.narrate:
        from .llm import DEFAULT_MODEL, Narrator, NarratorError
        try:
            narrator = Narrator(model=args.model or DEFAULT_MODEL)
            # v0.6: optional case_meta + intent + signature summary
            case_meta = None
            if args.case_meta is not None:
                from .case_meta import CaseMeta
                case_meta = CaseMeta.from_json(args.case_meta)
            intent = None
            if args.intent is not None:
                from .case_meta import NarrationIntent
                intent = NarrationIntent(args.intent)
            sig_summary = None
            if args.with_signature_summary:
                from .signature_summary import signature_summary
                sig_summary = signature_summary(ctx, top_k=args.topk)
            narrated = narrator.narrate(
                findings, ctx,
                training=ctx.training or None,
                case_meta=case_meta,
                intent=intent,
                sig_summary=sig_summary,
            )
            print(_render_narrated(args.report, narrated, deterministic_md))
            return 0
        except NarratorError as exc:
            sys.stderr.write(
                f"[ffca-agent] narration unavailable ({exc}); "
                f"falling back to deterministic output.\n"
            )
    print(deterministic_md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
