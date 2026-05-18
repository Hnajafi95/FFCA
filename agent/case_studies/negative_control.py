"""Negative-control bike_sharing run — train a HEALTHY model and verify
that no critical-severity rules fire in the v0.5 rulebook.

The v0.5 case-study version of bike_sharing was deliberately over-parameterized
(MLP 512×256, 500 epochs) so that `overfitting_volatility_spike` would trigger.
This negative-control flips that: moderate capacity + moderate training =
should be clean. If critical rules fire here, the rulebook has a false-positive
problem.

Outputs under FFCA_negative_control/bike_healthy/:
  - report.json, history.json, case_meta.json, plots/, checkpoints/
  - findings_v05.json — every rule the v0.5 book emits
  - NEGATIVE_CONTROL.md — verdict (pass/fail) + per-rule analysis
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Re-use the run_all.py training + FFCA glue.
sys.path.insert(0, str(REPO / "case_studies"))
import run_all  # type: ignore  # noqa: E402

import torch  # noqa: E402

from ffca_agent.evaluator import evaluate_rulebook, load_rulebook  # noqa: E402
from ffca_agent.report import ReportContext  # noqa: E402
from ffca_agent.training import TrainingHistory  # noqa: E402


def main() -> int:
    out_root = REPO / "FFCA_negative_control"
    out_dir = out_root / "bike_healthy"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = run_all._pick_device()
    print(f"[negative_control] device={device}", flush=True)

    case = run_all.TabularCase(
        name="bike_healthy",
        description=(
            "Bike Sharing with HEALTHY hyperparameters — moderate capacity, "
            "modest training duration. v0.5 critical rules should stay silent."
        ),
        expected_rules=[],  # we expect NOTHING critical to fire
        make_data=run_all._bike_sharing_data,
        # Moderate capacity (matches the v0.2 setup that converged cleanly).
        make_model=lambda n_in, n_out: run_all._make_mlp(n_in, n_out, hidden=(64, 32)),
        task="regression",
        n_epochs=80,
        n_checkpoints=10,
        notes=(
            "Negative control. (64,32) hidden + 80 epochs + Adam(lr=1e-3). "
            "Bike Sharing under these hyperparameters does NOT exhibit "
            "overfitting in the empirical record."
        ),
    )

    print("[negative_control] training...", flush=True)
    trained = run_all._train_tabular(case, out_dir, device)

    print("[negative_control] running FFCA report...", flush=True)
    run_all._run_ffca_tabular(case, trained, out_dir, device)

    # Evaluate the v0.5 rulebook
    print("[negative_control] evaluating v0.5 rulebook...", flush=True)
    rb = load_rulebook(REPO / "rulebook" / "ffca_rules.yaml")
    ctx = ReportContext.from_json(out_dir / "report.json")
    h = TrainingHistory.from_keras_history(out_dir / "history.json")
    h.derive_from_signatures(ctx, top_k=5)
    ctx.attach_training_history(h)
    findings = evaluate_rulebook(rb, ctx)

    # Categorize by severity
    crit = [f for f in findings if f.severity == "critical"]
    warn = [f for f in findings if f.severity == "warn"]
    info = [f for f in findings if f.severity == "info"]
    desc = [f for f in findings if f.kind == "descriptor"]

    payload = {
        "case": "bike_healthy",
        "rulebook_version": rb.get("version"),
        "n_features": ctx.n_features,
        "n_checkpoints": int(ctx.impact_curve.shape[0]),
        "val_train_gap": float(h.val_train_gap) if h.val_train_gap is not None else None,
        "nonlinearity_mean": float(ctx.nonlinearity.mean()),
        "interaction_to_impact_growth_ratio": float(ctx._interaction_to_impact_growth_ratio()),
        "findings": [
            {
                "rule_id": f.rule_id, "kind": f.kind, "severity": f.severity,
                "diagnosis": f.diagnosis, "evidence": f.evidence,
                "feature": f.feature,
            }
            for f in findings
        ],
        "n_critical": len(crit), "n_warn": len(warn), "n_info": len(info),
        "n_descriptors": len(desc),
        "verdict": "PASS" if not crit else "FAIL",
    }
    (out_dir / "findings_v05.json").write_text(json.dumps(payload, indent=2))

    print("\n[negative_control] results:")
    print(f"  rulebook v{rb.get('version')}, {ctx.n_features} features, "
          f"{ctx.impact_curve.shape[0]} checkpoints")
    print(f"  val_train_gap={h.val_train_gap:.3f}, "
          f"nonlinearity_mean={float(ctx.nonlinearity.mean()):.4g}")
    print(f"  CRITICAL findings: {len(crit)}")
    for f in crit:
        print(f"    [critical] {f.rule_id}: {f.diagnosis[:120]}")
    print(f"  warn findings:     {len(warn)}")
    for f in warn:
        print(f"    [warn] {f.rule_id}")
    print(f"  info findings:     {len(info)}")
    print(f"  descriptors:       {len(desc)}")
    print(f"\n  VERDICT: {payload['verdict']}")
    return 0 if not crit else 1


if __name__ == "__main__":
    sys.exit(main())
