"""Phase D: narrate the 20 corrected (ensemble-mode) FFCA reports for the
compound flooding project.

Runs on /Users/hnaja002/Documents/projects/compound_flooding/FFCA_resutls_before_prunning_ensemble/.
Uses CaseMeta with checkpoint_kind=SEED so the narrator does not recommend
'train longer' on seed-axis multi-modal disagreement.

Requires ANTHROPIC_API_KEY in env.
"""
from __future__ import annotations
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ffca_agent.evaluator import evaluate_rulebook, load_rulebook  # noqa: E402
from ffca_agent.llm import Narrator  # noqa: E402
from ffca_agent.report import ReportContext  # noqa: E402
from ffca_agent.signature_summary import signature_summary  # noqa: E402
from ffca_agent.case_meta import CaseMeta, CheckpointKind, NarrationIntent, ModelArchitecture, TaskType  # noqa: E402

CF = Path("/Users/hnaja002/Documents/projects/compound_flooding")
ORIG_DIR = CF / "FFCA_resutls_before_prunning_ensemble"
RULEBOOK = REPO / "rulebook" / "ffca_rules.yaml"


def find_reports() -> list[Path]:
    return sorted(ORIG_DIR.glob("*/*/report.json"))


def build_case_meta(report: Path) -> CaseMeta:
    # Recover n_seeds from the report itself
    d = json.loads(report.read_text())
    n_seeds = d.get("n_seeds", d.get("n_checkpoints", 30))
    return CaseMeta(
        project_name="Compound flooding — Miami GWL forecasting",
        model_architecture=ModelArchitecture.MLP,
        task_type=TaskType.REGRESSION,
        target_name="gwl",
        target_units="m (test RMSE reported in cm)",
        domain="Coastal hydrology. Inputs include lagged groundwater levels (gwl_t-k), ocean water levels (wl_t-k), rainfall (rain_t-k), tide stage (stgH/stgT), and gate openings (gate1/gate2). Forecasts gwl at lead times 3, 6, 12, 24 hours ahead.",
        pretrained=False,
        n_seeds=n_seeds,
        checkpoint_kind=CheckpointKind.SEED,
        feature_naming_convention="`name_t-k` = value at k hours BEFORE prediction time. Bare `name` (no suffix) = value at prediction time t=0.",
        notes="Each experiment combines a specific set of input channels (Measurements Only / Predicted Rainfall / Predicted Gate / Predicted WLS / All Inputs) with a specific forecast lead time. There are 20 such experiments total (5 input sets × 4 leads). All 20 use the same architecture (1-layer MLP with 100-200 neurons) trained as a 30-member seed ensemble.",
    )


def run_one(report_path: Path, narrator: Narrator, rulebook: dict) -> dict:
    ctx = ReportContext.from_json(report_path)
    case_meta = build_case_meta(report_path)
    ctx.attach_case_meta(case_meta)

    findings = evaluate_rulebook(rulebook, ctx)
    sig_summary = signature_summary(ctx, top_k=10)

    narrated = narrator.narrate(
        findings, ctx,
        case_meta=case_meta,
        intent=NarrationIntent.DIAGNOSE,
        sig_summary=sig_summary,
    )

    out = {
        "experiment": report_path.parent.name,
        "category": report_path.parent.parent.name,
        "checkpoint_kind": case_meta.checkpoint_kind.value,
        "n_seeds": case_meta.n_seeds,
        "rules_fired": sorted({f.rule_id for f in findings}),
        "warn_rules":  [f.rule_id for f in findings if f.severity == "warn"],
        "critical_rules": [f.rule_id for f in findings if f.severity == "critical"],
        "executive_summary": narrated.executive_summary,
        "actions": [asdict(a) for a in narrated.actions],
        "rule_free_observations": [asdict(o) for o in narrated.rule_free_observations],
        "caveats": narrated.caveats,
        "model": narrated.model,
        "usage": narrated.usage,
    }
    return out


def main():
    if "ANTHROPIC_API_KEY" not in os.environ:
        raise SystemExit("ANTHROPIC_API_KEY env var is not set.")
    rulebook = load_rulebook(RULEBOOK)
    print(f"Rulebook: {len(rulebook['rules'])} rules ({sum('applies_when' in r for r in rulebook['rules'])} axis-gated)")
    narrator = Narrator()
    reports = find_reports()
    print(f"Found {len(reports)} corrected reports under {ORIG_DIR}\n")

    total_tokens_in = total_tokens_out = 0
    for i, p in enumerate(reports, 1):
        out_file = p.parent / "narration.json"
        if out_file.exists():
            print(f"[{i:>2}/{len(reports)}] {p.parent.name:42s}  [skip — narration exists]")
            continue
        print(f"[{i:>2}/{len(reports)}] {p.parent.name:42s}  ...", end=" ", flush=True)
        t0 = time.time()
        try:
            out = run_one(p, narrator, rulebook)
        except Exception as e:
            print(f"FAIL: {e}")
            continue
        out_file.write_text(json.dumps(out, indent=2))
        u = out.get("usage") or {}
        ti = u.get("input_tokens", 0); to = u.get("output_tokens", 0)
        total_tokens_in += ti
        total_tokens_out += to
        print(f"ok ({time.time()-t0:.1f}s | in={ti} out={to} | fired={len(out['rules_fired'])} warn={len(out['warn_rules'])})")

    print()
    print(f"Totals: input={total_tokens_in} output={total_tokens_out} tokens.")


if __name__ == "__main__":
    main()
