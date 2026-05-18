"""Re-evaluate the v0.4 case-study artifacts against the v0.5 rulebook.

No retraining — the FFCA report.json + history.json + (optional) vision_metrics.json
written by `run_all.py` already contain everything the v0.5 rule changes need.
This script loads them per case, derives the volatility/impact/interaction
curves from FFCA signatures (so the dynamic rules see them), runs the v0.5
rulebook, and writes per-case findings + a combined `summary_v05.json`.

Usage:
    python case_studies/reeval_v05.py \
        --runs-dir FFCA_runs_results_v04_real \
        --out-dir  FFCA_runs_results_v04_real
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

# Allow imports when invoked from the repo root.
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ffca_agent.evaluator import evaluate_rulebook, load_rulebook
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="FFCA_runs_results_v04_real")
    ap.add_argument("--out-dir", default="FFCA_runs_results_v04_real")
    ap.add_argument("--rulebook", default="rulebook/ffca_rules.yaml")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    rb = load_rulebook(args.rulebook)
    runs = Path(args.runs_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict] = {"rulebook_version": rb.get("version", "?"), "cases": {}}

    for case in CASES:
        d = runs / case
        if not (d / "report.json").exists():
            print(f"[skip] {case}: no report.json")
            continue

        ctx = ReportContext.from_json(d / "report.json")
        h = TrainingHistory.from_keras_history(d / "history.json")
        h.derive_from_signatures(ctx, top_k=args.top_k)
        ctx.attach_training_history(h)

        if (d / "vision_metrics.json").exists():
            vm = VisionMetrics.from_json(d / "vision_metrics.json")
            ctx.attach_vision_metrics(vm)

        findings = evaluate_rulebook(rb, ctx)
        diag = [f for f in findings if f.kind == "diagnostic"]

        case_record = {
            "n_features": ctx.n_features,
            "n_checkpoints": int(ctx.impact_curve.shape[0]),
            "val_train_gap": float(h.val_train_gap) if h.val_train_gap is not None else None,
            "nonlinearity_mean": float(ctx.nonlinearity.mean()),
            "interaction_to_impact_growth_ratio": float(
                ctx._interaction_to_impact_growth_ratio()
            ),
            "findings": [
                {
                    "rule_id": f.rule_id,
                    "name": f.rule_name,
                    "kind": f.kind,
                    "severity": f.severity,
                    "scope": f.scope,
                    "feature": f.feature,
                    "diagnosis": f.diagnosis,
                    "recommendation": f.recommendation,
                    "evidence": f.evidence,
                    "paper_ref": f.paper_ref,
                }
                for f in findings
            ],
            "diagnostic_rule_ids": sorted({f.rule_id for f in diag}),
        }

        # Per-case JSON for downstream narration
        (out / case / "findings_v05.json").write_text(
            json.dumps(case_record, indent=2)
        )
        summary["cases"][case] = {
            "diagnostic_rule_ids": case_record["diagnostic_rule_ids"],
            "n_findings": len(findings),
            "growth_ratio": case_record["interaction_to_impact_growth_ratio"],
        }

        print(f"=== {case} ===")
        for f in diag:
            print(f"  [{f.severity or '-'}] {f.rule_id}")

    (out / "summary_v05.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out/'summary_v05.json'}")


if __name__ == "__main__":
    main()
