"""Integration tests for the rule evaluator using synthetic FFCA-shaped reports.

These tests construct minimal report.json-equivalent dicts that exercise each
rule kind, then verify the evaluator produces the expected findings.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ffca_agent.evaluator import evaluate_rulebook, load_rulebook, summarize
from ffca_agent.report import ReportContext

REPO = Path(__file__).resolve().parents[1]
RULEBOOK_PATH = REPO / "rulebook" / "ffca_rules.yaml"


def _synthetic_report(
    feature_names: list[str],
    archetypes: list[int],
    impact: list[float],
    volatility: list[float] | None = None,
    nonlinearity: list[float] | None = None,
    interaction: list[float] | None = None,
    trust_decisions: dict[str, str] | None = None,
    cosens: dict | None = None,
    n_checkpoints: int = 5,
    checkpoint_drift_factor: float = 0.0,
) -> dict:
    """Build a minimal FFCA-shaped report.json dict.

    checkpoint_drift_factor: if > 0, scales each prior checkpoint's impact
    by (1 - drift_factor * (n_checkpoints - i - 1)) to simulate drift.
    """
    n = len(feature_names)
    vol = volatility or [0.0001] * n
    nl = nonlinearity or [0.005] * n
    inter = interaction or [0.1] * n
    base_sig = {
        "impact": impact,
        "volatility": vol,
        "nonlinearity": nl,
        "interaction": inter,
        "archetypes": archetypes,
    }
    if checkpoint_drift_factor == 0.0:
        sigs = [base_sig] * n_checkpoints
    else:
        sigs = []
        for k in range(n_checkpoints):
            scale = max(0.1, 1.0 - checkpoint_drift_factor * (n_checkpoints - k - 1))
            sigs.append({
                **base_sig,
                "impact": [x * scale for x in impact],
                "volatility": [v * scale for v in vol],
                "interaction": [x * scale for x in inter],
            })
    return {
        "n_features": n,
        "feature_names": feature_names,
        "signatures": sigs,
        "trust": {f: {"decision": (trust_decisions or {}).get(f, "INVESTIGATE (unstable)")}
                  for f in feature_names},
        "trust_summary": {},
        "co_sensitivity": cosens,
        "findings": [],
    }


def _write_and_eval(raw: dict, rulebook, tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps(raw))
    ctx = ReportContext.from_json(p)
    return evaluate_rulebook(rulebook, ctx)


@pytest.fixture(scope="module")
def rulebook():
    return load_rulebook(RULEBOOK_PATH)


# ── archetype descriptor rules ─────────────────────────────────────────────

def test_archetype_descriptor_fires_per_feature(rulebook, tmp_path):
    # 3 features, archetypes: Workhorse=2, Catalyst=3, Noise=0
    raw = _synthetic_report(
        feature_names=["a", "b", "c"],
        archetypes=[2, 3, 0],
        impact=[0.5, 0.4, 0.001],
    )
    p = tmp_path / "r.json"
    p.write_text(json.dumps(raw))
    ctx = ReportContext.from_json(p)
    findings = evaluate_rulebook(rulebook, ctx)
    by_rule = {f.rule_id for f in findings}
    assert "archetype_simple_workhorse" in by_rule
    assert "archetype_interactive_catalyst" in by_rule
    assert "archetype_noise_candidate" in by_rule


# ── model-wide rules ───────────────────────────────────────────────────────

def test_healthy_archetype_distribution_fires(rulebook, tmp_path):
    # Mix: 30% Noise, 30% Catalyst, 20% Complex, 20% Stable
    archetypes = [0]*3 + [3]*3 + [7]*2 + [6]*2
    raw = _synthetic_report(
        feature_names=[f"f{i}" for i in range(10)],
        archetypes=archetypes,
        impact=[0.1]*10,
    )
    p = tmp_path / "r.json"
    p.write_text(json.dumps(raw))
    ctx = ReportContext.from_json(p)
    findings = evaluate_rulebook(rulebook, ctx)
    assert any(f.rule_id == "healthy_archetype_distribution" for f in findings)


def test_noise_dominant_fires(rulebook, tmp_path):
    # 80% Noise → archetype_imbalance_noise_dominant fires (warn)
    raw = _synthetic_report(
        feature_names=[f"f{i}" for i in range(10)],
        archetypes=[0]*8 + [3, 6],
        impact=[0.01]*8 + [0.5, 0.3],
    )
    p = tmp_path / "r.json"
    p.write_text(json.dumps(raw))
    ctx = ReportContext.from_json(p)
    findings = evaluate_rulebook(rulebook, ctx)
    assert any(f.rule_id == "archetype_imbalance_noise_dominant" and f.severity == "warn"
               for f in findings)


# ── trust-score rules ──────────────────────────────────────────────────────

def test_trust_instability_high_epoch_axis_fires(rulebook, tmp_path):
    """v0.7 split: trust_instability_high was renamed and split into
    epoch-axis and seed-axis variants. The epoch-axis variant requires
    case.checkpoint_kind=='epoch' to fire."""
    from ffca_agent.case_meta import CaseMeta, CheckpointKind
    # 7/10 features in INVESTIGATE → 70% > 50% threshold
    trust = {f"f{i}": "INVESTIGATE (unstable)" for i in range(7)}
    trust.update({f"f{i}": "CONFIDENTLY KEEP" for i in range(7, 10)})
    raw = _synthetic_report(
        feature_names=[f"f{i}" for i in range(10)],
        archetypes=[3]*10,
        impact=[0.5]*10,
        trust_decisions=trust,
    )
    p = tmp_path / "r.json"
    p.write_text(json.dumps(raw))
    ctx = ReportContext.from_json(p)
    ctx.attach_case_meta(CaseMeta(checkpoint_kind=CheckpointKind.EPOCH))
    findings = evaluate_rulebook(rulebook, ctx)
    instability = [f for f in findings if f.rule_id == "trust_instability_high_epoch_axis"]
    assert instability, f"expected trust_instability_high_epoch_axis to fire, got {[f.rule_id for f in findings]}"
    assert instability[0].severity == "warn"


def test_trust_multi_modal_seeds_fires_on_seed_axis(rulebook, tmp_path):
    """Companion to the test above: on a seed-axis case the same
    INVESTIGATE rate fires trust_multi_modal_seeds instead."""
    from ffca_agent.case_meta import CaseMeta, CheckpointKind
    trust = {f"f{i}": "INVESTIGATE (multi-modal seeds)" for i in range(7)}
    trust.update({f"f{i}": "CONFIDENTLY KEEP" for i in range(7, 10)})
    raw = _synthetic_report(
        feature_names=[f"f{i}" for i in range(10)],
        archetypes=[3]*10,
        impact=[0.5]*10,
        trust_decisions=trust,
    )
    p = tmp_path / "r.json"
    p.write_text(json.dumps(raw))
    ctx = ReportContext.from_json(p)
    ctx.attach_case_meta(CaseMeta(checkpoint_kind=CheckpointKind.SEED))
    findings = evaluate_rulebook(rulebook, ctx)
    multi_modal = [f for f in findings if f.rule_id == "trust_multi_modal_seeds"]
    assert multi_modal, f"expected trust_multi_modal_seeds to fire, got {[f.rule_id for f in findings]}"
    assert multi_modal[0].severity == "warn"


def test_trust_keep_recommended_fires_when_features_present(rulebook, tmp_path):
    trust = {f"f{i}": "CONFIDENTLY KEEP" for i in range(3)}
    trust.update({f"f{i}": "INVESTIGATE (unstable)" for i in range(3, 10)})
    raw = _synthetic_report(
        feature_names=[f"f{i}" for i in range(10)],
        archetypes=[3]*10,
        impact=[0.5]*10,
        trust_decisions=trust,
    )
    p = tmp_path / "r.json"
    p.write_text(json.dumps(raw))
    ctx = ReportContext.from_json(p)
    findings = evaluate_rulebook(rulebook, ctx)
    assert any(f.rule_id == "trust_keep_recommended" for f in findings)


def test_trust_prune_recommended_fires_when_features_present(rulebook, tmp_path):
    trust = {f"f{i}": "CONFIDENTLY PRUNE" for i in range(2)}
    trust.update({f"f{i}": "INVESTIGATE (unstable)" for i in range(2, 10)})
    raw = _synthetic_report(
        feature_names=[f"f{i}" for i in range(10)],
        archetypes=[3]*10,
        impact=[0.5]*10,
        trust_decisions=trust,
    )
    p = tmp_path / "r.json"
    p.write_text(json.dumps(raw))
    ctx = ReportContext.from_json(p)
    findings = evaluate_rulebook(rulebook, ctx)
    assert any(f.rule_id == "trust_prune_recommended" for f in findings)


# ── co-sensitivity rules ───────────────────────────────────────────────────

def test_cosens_abort_when_best_nc_below_threshold(rulebook, tmp_path):
    cosens = {
        "summary": {
            "best_nc_fraction": 0.25,
            "permutation_p": 0.0,
            "bootstrap_ari_median": 0.4,
            "silhouette_observed": 0.4,
            "k": 3,
        },
        "groups": {},
    }
    raw = _synthetic_report(
        feature_names=[f"f{i}" for i in range(10)],
        archetypes=[3]*10,
        impact=[0.5]*10,
        cosens=cosens,
    )
    p = tmp_path / "r.json"
    p.write_text(json.dumps(raw))
    ctx = ReportContext.from_json(p)
    findings = evaluate_rulebook(rulebook, ctx)
    assert any(f.rule_id == "cosens_abort_no_prune_safe" for f in findings)
    assert not any(f.rule_id == "cosens_prune_candidate_group" for f in findings)


def test_cosens_prune_candidate_when_all_gates_pass(rulebook, tmp_path):
    cosens = {
        "summary": {
            "best_nc_fraction": 0.65,
            "permutation_p": 0.01,
            "bootstrap_ari_median": 0.7,
            "silhouette_observed": 0.5,
            "k": 3,
        },
        "groups": {},
    }
    raw = _synthetic_report(
        feature_names=[f"f{i}" for i in range(10)],
        archetypes=[3]*10,
        impact=[0.5]*10,
        cosens=cosens,
    )
    p = tmp_path / "r.json"
    p.write_text(json.dumps(raw))
    ctx = ReportContext.from_json(p)
    findings = evaluate_rulebook(rulebook, ctx)
    assert any(f.rule_id == "cosens_prune_candidate_group" for f in findings)


# ── prescriptive ────────────────────────────────────────────────────────────

def test_linear_baseline_will_fail_when_interactors_present(rulebook, tmp_path):
    # 20% Hidden Interactor (idx 1), 20% Nonlinear Driver (idx 4), rest mixed
    archetypes = [1, 1, 4, 4, 3, 6, 6, 6, 7, 7]
    raw = _synthetic_report(
        feature_names=[f"f{i}" for i in range(10)],
        archetypes=archetypes,
        impact=[0.3]*10,
    )
    p = tmp_path / "r.json"
    p.write_text(json.dumps(raw))
    ctx = ReportContext.from_json(p)
    findings = evaluate_rulebook(rulebook, ctx)
    assert any(f.rule_id == "linear_baseline_will_fail" for f in findings)


# ── summary helper ─────────────────────────────────────────────────────────

def test_summarize_counts_match(rulebook, tmp_path):
    raw = _synthetic_report(
        feature_names=["a", "b", "c"],
        archetypes=[2, 3, 0],
        impact=[0.5, 0.4, 0.001],
    )
    p = tmp_path / "r.json"
    p.write_text(json.dumps(raw))
    ctx = ReportContext.from_json(p)
    findings = evaluate_rulebook(rulebook, ctx)
    s = summarize(findings)
    assert s["n_findings"] == len(findings)
    assert s["n_diagnostic"] + s["n_descriptor"] == s["n_findings"]


# ── new rules (v0.2) ──────────────────────────────────────────────────────

def test_archetype_imbalance_noise_extreme_fires(rulebook, tmp_path):
    # 80% Noise → both warn (>50%) and critical (>70%) fire
    raw = _synthetic_report(
        feature_names=[f"f{i}" for i in range(10)],
        archetypes=[0] * 8 + [3, 6],
        impact=[0.01] * 8 + [0.5, 0.3],
    )
    findings = _write_and_eval(raw, rulebook, tmp_path)
    fired = {f.rule_id for f in findings}
    assert "archetype_imbalance_noise_dominant" in fired
    assert "archetype_imbalance_noise_extreme" in fired


def test_model_degenerate_fires_when_one_archetype_dominates(rulebook, tmp_path):
    # 95% Noise → degenerate
    raw = _synthetic_report(
        feature_names=[f"f{i}" for i in range(20)],
        archetypes=[0] * 19 + [3],
        impact=[0.001] * 19 + [0.5],
    )
    findings = _write_and_eval(raw, rulebook, tmp_path)
    fired = {f.rule_id for f in findings}
    assert "model_degenerate_single_archetype" in fired


def test_hidden_interactor_dominant_fires(rulebook, tmp_path):
    # 50% Hidden Interactor (archetype 1)
    raw = _synthetic_report(
        feature_names=[f"f{i}" for i in range(10)],
        archetypes=[1] * 5 + [2] * 3 + [6] * 2,
        impact=[0.1] * 10,
        interaction=[0.5] * 5 + [0.1] * 5,
    )
    findings = _write_and_eval(raw, rulebook, tmp_path)
    fired = {f.rule_id for f in findings}
    assert "hidden_interactor_dominant" in fired


def test_feature_concentration_pareto_fires(rulebook, tmp_path):
    # Top 20% (= top 2 of 10) carry > 80% of impact
    raw = _synthetic_report(
        feature_names=[f"f{i}" for i in range(10)],
        archetypes=[2] * 10,
        impact=[10.0, 10.0] + [0.1] * 8,
    )
    findings = _write_and_eval(raw, rulebook, tmp_path)
    fired = {f.rule_id for f in findings}
    assert "feature_concentration_pareto" in fired


def test_feature_concentration_extreme_fires(rulebook, tmp_path):
    # Top 5% (= top 1 of 20) carries > 80% of impact
    raw = _synthetic_report(
        feature_names=[f"f{i}" for i in range(20)],
        archetypes=[2] * 20,
        impact=[100.0] + [0.01] * 19,
    )
    findings = _write_and_eval(raw, rulebook, tmp_path)
    fired = {f.rule_id for f in findings}
    assert "feature_concentration_extreme" in fired


def test_numerical_saturation_fires_when_impact_vanishes(rulebook, tmp_path):
    # All Impact below 1e-6 — saturated
    raw = _synthetic_report(
        feature_names=[f"f{i}" for i in range(5)],
        archetypes=[0] * 5,
        impact=[1e-9] * 5,
    )
    findings = _write_and_eval(raw, rulebook, tmp_path)
    fired = {f.rule_id for f in findings}
    assert "numerical_saturation" in fired


def test_numerical_saturation_does_not_fire_normal_range(rulebook, tmp_path):
    raw = _synthetic_report(
        feature_names=[f"f{i}" for i in range(5)],
        archetypes=[2] * 5,
        impact=[0.1, 0.2, 0.3, 0.4, 0.5],
    )
    findings = _write_and_eval(raw, rulebook, tmp_path)
    fired = {f.rule_id for f in findings}
    assert "numerical_saturation" not in fired


def test_silent_features_present_fires(rulebook, tmp_path):
    # 20% in CONFIDENTLY PRUNE
    trust = {f"f{i}": "CONFIDENTLY PRUNE" for i in range(2)}
    trust.update({f"f{i}": "INVESTIGATE (unstable)" for i in range(2, 10)})
    raw = _synthetic_report(
        feature_names=[f"f{i}" for i in range(10)],
        archetypes=[2] * 10,
        impact=[0.5] * 10,
        trust_decisions=trust,
    )
    findings = _write_and_eval(raw, rulebook, tmp_path)
    fired = {f.rule_id for f in findings}
    assert "silent_features_present" in fired


def test_convergence_achieved_fires_when_drift_low(rulebook, tmp_path):
    """Epoch-axis-only rule; requires case_meta.checkpoint_kind=epoch
    after the v0.7 gating fix."""
    from ffca_agent.case_meta import CaseMeta, CheckpointKind
    # All checkpoints identical → 0% drift
    raw = _synthetic_report(
        feature_names=[f"f{i}" for i in range(5)],
        archetypes=[2] * 5,
        impact=[0.5] * 5,
        n_checkpoints=5,
        checkpoint_drift_factor=0.0,
    )
    p = tmp_path / "r.json"
    p.write_text(json.dumps(raw))
    ctx = ReportContext.from_json(p)
    ctx.attach_case_meta(CaseMeta(checkpoint_kind=CheckpointKind.EPOCH))
    findings = evaluate_rulebook(rulebook, ctx)
    fired = {f.rule_id for f in findings}
    assert "convergence_achieved" in fired
    assert "late_checkpoint_drift" not in fired


def test_late_checkpoint_drift_fires_when_drift_high(rulebook, tmp_path):
    """Epoch-axis-only rule; same gating as convergence_achieved."""
    from ffca_agent.case_meta import CaseMeta, CheckpointKind
    # Strong checkpoint scaling → big drift between last two
    raw = _synthetic_report(
        feature_names=[f"f{i}" for i in range(5)],
        archetypes=[2] * 5,
        impact=[0.5] * 5,
        n_checkpoints=5,
        checkpoint_drift_factor=0.3,
    )
    p = tmp_path / "r.json"
    p.write_text(json.dumps(raw))
    ctx = ReportContext.from_json(p)
    ctx.attach_case_meta(CaseMeta(checkpoint_kind=CheckpointKind.EPOCH))
    findings = evaluate_rulebook(rulebook, ctx)
    fired = {f.rule_id for f in findings}
    assert "late_checkpoint_drift" in fired
    assert "convergence_achieved" not in fired


def test_epoch_axis_rules_do_not_fire_on_seed_axis(rulebook, tmp_path):
    """Confirm the gating actually gates: even with low drift, the
    convergence_achieved rule must not fire on a declared seed-axis
    case. Prevents the seed-vs-epoch bug from recurring."""
    from ffca_agent.case_meta import CaseMeta, CheckpointKind
    raw = _synthetic_report(
        feature_names=[f"f{i}" for i in range(5)],
        archetypes=[2] * 5,
        impact=[0.5] * 5,
        n_checkpoints=5,
        checkpoint_drift_factor=0.0,
    )
    p = tmp_path / "r.json"
    p.write_text(json.dumps(raw))
    ctx = ReportContext.from_json(p)
    ctx.attach_case_meta(CaseMeta(checkpoint_kind=CheckpointKind.SEED))
    findings = evaluate_rulebook(rulebook, ctx)
    fired = {f.rule_id for f in findings}
    assert "convergence_achieved" not in fired
    assert "late_checkpoint_drift" not in fired
    assert "hierarchical_learning_confirmed" not in fired


def test_monitor_bucket_dominant_fires(rulebook, tmp_path):
    trust = {f"f{i}": "MONITOR (borderline)" for i in range(5)}
    trust.update({f"f{i}": "INVESTIGATE (unstable)" for i in range(5, 10)})
    raw = _synthetic_report(
        feature_names=[f"f{i}" for i in range(10)],
        archetypes=[2] * 10,
        impact=[0.5] * 10,
        trust_decisions=trust,
    )
    findings = _write_and_eval(raw, rulebook, tmp_path)
    fired = {f.rule_id for f in findings}
    assert "monitor_bucket_dominant" in fired


def test_trust_volatility_contradiction_fires(rulebook, tmp_path):
    # f0 is KEEP and has highest volatility — should fire
    trust = {"f0": "CONFIDENTLY KEEP"}
    trust.update({f"f{i}": "INVESTIGATE (unstable)" for i in range(1, 10)})
    raw = _synthetic_report(
        feature_names=[f"f{i}" for i in range(10)],
        archetypes=[2] * 10,
        impact=[0.5] * 10,
        volatility=[0.5] + [0.001] * 9,
        trust_decisions=trust,
    )
    findings = _write_and_eval(raw, rulebook, tmp_path)
    fired = [f for f in findings if f.rule_id == "trust_volatility_contradiction"]
    assert fired, "expected trust_volatility_contradiction to fire on f0"
    assert fired[0].feature == "f0"


def test_cosens_weak_clustering_significant_fires(rulebook, tmp_path):
    cosens = {
        "summary": {
            "best_nc_fraction": 0.30,
            "permutation_p": 0.01,
            "bootstrap_ari_median": 0.4,
            "silhouette_observed": 0.15,  # fuzzy
            "k": 3,
        },
        "groups": {},
    }
    raw = _synthetic_report(
        feature_names=[f"f{i}" for i in range(10)],
        archetypes=[2] * 10,
        impact=[0.5] * 10,
        cosens=cosens,
    )
    findings = _write_and_eval(raw, rulebook, tmp_path)
    fired = {f.rule_id for f in findings}
    assert "cosens_weak_clustering_significant" in fired


# ── integration: real flooding report ──────────────────────────────────────

def test_real_flooding_report_processes_without_error(rulebook):
    """End-to-end smoke: the compound-flooding 12hr_measured report must process.

    Since v0.2 aggregates archetype descriptors, we expect on the order of
    a dozen model-wide findings, not one per feature."""
    real_report = Path(
        "/Users/hnaja002/Documents/projects/compound_flooding"
        "/FFCA_resutls_before_prunning/Measurements Only/12hr_measured_sigmoid/report.json"
    )
    if not real_report.exists():
        pytest.skip(f"real report not available at {real_report}")
    ctx = ReportContext.from_json(real_report)
    findings = evaluate_rulebook(rulebook, ctx)
    s = summarize(findings)
    # Aggregated rulebook: expect at least a handful of findings, much less
    # than n_features.
    assert s["n_findings"] >= 5
    fired = set(s["rules_fired"])
    assert "trust_instability_high" in fired
    assert "cosens_abort_no_prune_safe" in fired
    # No template variable should leak through as literal text
    for f in findings:
        for text in (f.diagnosis, f.recommendation, f.evidence):
            assert "{" not in text or "}" not in text, (
                f"unresolved template in {f.rule_id}: {text!r}"
            )
