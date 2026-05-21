"""Tests for the TrainingHistory adapter and the 5 rules it unlocks.

The 5 dynamic-rule scenarios match paper figures (Bike Sharing volatility
spike, Housing leakage / spurious feature, hierarchical learning, low-capacity
plateau) via hand-crafted synthetic curves. A separate negative test confirms
that omitting training history leaves the dynamic rules silent — no tracebacks,
no `{unresolved}` template leaks.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ffca_agent.evaluator import evaluate_rulebook, load_rulebook
from ffca_agent.report import ReportContext
from ffca_agent.training import TrainingHistory

REPO = Path(__file__).resolve().parents[1]
RULEBOOK_PATH = REPO / "rulebook" / "ffca_rules.yaml"


@pytest.fixture(scope="module")
def rulebook():
    return load_rulebook(RULEBOOK_PATH)


# ── helpers ────────────────────────────────────────────────────────────────


def _build_report(
    *,
    feature_names: list[str],
    archetypes: list[int],
    impact: list[float],
    volatility: list[float] | None = None,
    nonlinearity: list[float] | None = None,
    interaction: list[float] | None = None,
    impact_curve_per_feature: np.ndarray | None = None,
    n_checkpoints: int = 5,
    trust_decisions: dict[str, str] | None = None,
) -> dict:
    """Build a synthetic FFCA report. impact_curve_per_feature: (n_ckpts, n_features)."""
    n = len(feature_names)
    vol = volatility or [0.0001] * n
    nl = nonlinearity or [0.005] * n
    inter = interaction or [0.1] * n

    sigs: list[dict] = []
    for k in range(n_checkpoints):
        if impact_curve_per_feature is not None:
            imp_k = impact_curve_per_feature[k].tolist()
        else:
            imp_k = list(impact)
        sigs.append({
            "impact": imp_k,
            "volatility": vol,
            "nonlinearity": nl,
            "interaction": inter,
            "archetypes": archetypes,
        })

    return {
        "n_features": n,
        "feature_names": feature_names,
        "signatures": sigs,
        "trust": {
            f: {"decision": (trust_decisions or {}).get(f, "KEEP (stable)")}
            for f in feature_names
        },
        "trust_summary": {},
        "co_sensitivity": None,
        "findings": [],
    }


def _ctx_with_history(raw: dict, history: TrainingHistory, tmp_path) -> ReportContext:
    p = tmp_path / "r.json"
    p.write_text(json.dumps(raw))
    ctx = ReportContext.from_json(p)
    ctx.attach_training_history(history)
    return ctx


# ── loader unit tests ──────────────────────────────────────────────────────


def test_from_dict_picks_val_loss_lower_is_better():
    h = TrainingHistory.from_dict({
        "loss": [1.0, 0.5, 0.2, 0.1],
        "val_loss": [1.1, 0.6, 0.3, 0.2],
    })
    assert h.lower_is_better is True
    assert h.metric_name == "val_loss"
    assert pytest.approx(h.val_score_final) == 0.2
    assert pytest.approx(h.train_score_final) == 0.1
    # gap = |(0.2 - 0.1) / max(0.1, 0.2)| = 0.5
    assert pytest.approx(h.val_train_gap, abs=1e-6) == 0.5


def test_from_dict_picks_val_accuracy_higher_is_better():
    h = TrainingHistory.from_dict({
        "accuracy": [0.6, 0.8, 0.95, 1.0],
        "val_accuracy": [0.55, 0.75, 0.9, 0.99],
    })
    assert h.lower_is_better is False
    assert h.metric_name == "val_accuracy"
    assert pytest.approx(h.val_score_final, abs=1e-9) == 0.99
    assert pytest.approx(h.train_score_final, abs=1e-9) == 1.0
    # gap = |(train - val)/max| = |0.01/1.0|
    assert pytest.approx(h.val_train_gap, abs=1e-6) == 0.01


def test_from_keras_history_json_unwraps_nested_history(tmp_path):
    p = tmp_path / "h.json"
    p.write_text(json.dumps({"history": {"loss": [1, 0.5], "val_loss": [1.1, 0.6]}}))
    h = TrainingHistory.from_keras_history(p)
    assert h.val_score_final == 0.6


def test_from_csv(tmp_path):
    p = tmp_path / "h.csv"
    p.write_text("epoch,loss,val_loss\n0,1.0,1.1\n1,0.5,0.6\n2,0.2,0.3\n")
    h = TrainingHistory.from_keras_history(p)
    assert pytest.approx(h.val_score_final) == 0.3


def test_derive_from_signatures_fills_curves(tmp_path):
    raw = _build_report(
        feature_names=["a", "b", "c"],
        archetypes=[2, 3, 0],
        impact=[0.5, 0.4, 0.001],
        n_checkpoints=10,
    )
    p = tmp_path / "r.json"
    p.write_text(json.dumps(raw))
    ctx = ReportContext.from_json(p)
    h = TrainingHistory()
    h.derive_from_signatures(ctx, top_k=2)
    assert h.volatility_curve is not None and h.volatility_curve.shape == (10,)
    assert h.impact_curve_topk_mean is not None and h.impact_curve_topk_mean.shape == (10,)
    assert h.interaction_curve_topk_mean is not None


# ── rule-firing tests ──────────────────────────────────────────────────────


def test_insufficient_capacity_fires_on_low_complex_drivers(rulebook, tmp_path):
    # v0.5: trigger_logic=any — fires when complex_driver_pct < 15 OR catalyst_pct < 15
    # OR nonlinearity_mean < 1e-3.
    # No Complex Drivers (7) or Catalysts (3) — all features are simple.
    archetypes = [2] * 10  # Simple Workhorse
    raw = _build_report(
        feature_names=[f"f{i}" for i in range(10)],
        archetypes=archetypes,
        impact=[0.3] * 10,
    )
    h = TrainingHistory(
        val_score_final=0.595, train_score_final=0.585, val_train_gap=0.01,
        metric_name="val_loss", lower_is_better=True,
    )
    ctx = _ctx_with_history(raw, h, tmp_path)
    findings = evaluate_rulebook(rulebook, ctx)
    fired = {f.rule_id for f in findings}
    assert "insufficient_capacity" in fired


def test_insufficient_capacity_fires_on_pure_linear_model(rulebook, tmp_path):
    """v0.5: a pure-linear model should fire even if FFCA's archetype classifier
    mislabels features as Complex Driver. mean Nonlinearity < 1e-3 is the
    directly-observable signal."""
    # 5 features labeled Complex Driver (7) — the FFCA classifier bug behaviour.
    # Without v0.5's nonlinearity_mean trigger, neither archetype-pct trigger
    # would fire (complex_driver_pct = 50% > 15%, catalyst_pct = 0% < 15% but the
    # OLD all-AND logic would have required plateau too).
    archetypes = [7] * 5 + [2] * 5  # 50% Complex Driver, 50% Simple
    raw = _build_report(
        feature_names=[f"f{i}" for i in range(10)],
        archetypes=archetypes,
        impact=[0.3] * 10,
        nonlinearity=[0.0] * 10,  # pure linear: zero curvature
    )
    h = TrainingHistory(
        val_score_final=0.5, train_score_final=0.49, val_train_gap=0.02,
        metric_name="val_loss", lower_is_better=True,
    )
    ctx = _ctx_with_history(raw, h, tmp_path)
    findings = evaluate_rulebook(rulebook, ctx)
    fired = {f.rule_id for f in findings}
    assert "insufficient_capacity" in fired


def test_overfitting_volatility_spike_fires_bike_sharing_style(rulebook, tmp_path):
    # v0.5: triggers are (volatility spike >= 1.4) AND (val_train_gap > 0.5).
    # Bike-style overfitting: train_loss keeps falling, val_loss falls slower,
    # gap widens — captured by val_train_gap rather than a strict plateau.
    # v0.8: rule is now applies_when: case.checkpoint_kind=='epoch'.
    from ffca_agent.case_meta import CaseMeta, CheckpointKind
    raw = _build_report(
        feature_names=[f"f{i}" for i in range(10)],
        archetypes=[7] * 5 + [3] * 5,  # Complex + Catalyst — capacity is fine
        impact=[0.3] * 10,
    )
    val = np.concatenate([np.linspace(1.0, 0.6, 100), np.linspace(0.6, 0.55, 100)])
    train = np.concatenate([np.linspace(1.0, 0.3, 100), np.linspace(0.3, 0.15, 100)])
    vol = np.concatenate([np.full(120, 0.001), np.full(80, 0.0025)])
    h = TrainingHistory(
        val_score_curve=val,
        train_score_curve=train,
        val_score_final=float(val[-1]),
        train_score_final=float(train[-1]),
        val_train_gap=0.72,  # (val - train) / max(...) normalized → > 0.5 trigger
        volatility_curve=vol,
        metric_name="val_loss", lower_is_better=True,
    )
    ctx = _ctx_with_history(raw, h, tmp_path)
    ctx.attach_case_meta(CaseMeta(checkpoint_kind=CheckpointKind.EPOCH))
    findings = evaluate_rulebook(rulebook, ctx)
    fired = {f.rule_id for f in findings}
    assert "overfitting_volatility_spike" in fired


def test_data_leakage_immediate_dominance_fires_on_dominant_saturated_feature(rulebook, tmp_path):
    """v0.3: leakage rule now works for regression AND classification.
    Trigger: feature dominates (>5× mean Impact) + saturated at first ckpt + val/train gap < 0.10.
    v0.8: rule is now applies_when: case.checkpoint_kind=='epoch' because
    impact_saturation is meaningless on a seed-axis ensemble.
    """
    from ffca_agent.case_meta import CaseMeta, CheckpointKind
    n_ckpts = 10
    n_feat = 8  # dominance can only exceed n=5 with at least 6 features (mean lower bound)
    # f0 already at near-final impact from epoch 1; others much smaller.
    impact_curve = np.full((n_ckpts, n_feat), 0.02)
    impact_curve[:, 0] = np.linspace(0.95, 1.0, n_ckpts)  # saturation 0.95/1.0 = 0.95
    raw = _build_report(
        feature_names=["leaky"] + [f"f{i}" for i in range(1, n_feat)],
        archetypes=[2] * n_feat,
        impact=impact_curve[-1].tolist(),
        impact_curve_per_feature=impact_curve,
        n_checkpoints=n_ckpts,
    )
    # Regression metric: val_loss ≈ train_loss ≈ 0.003 (perfectly fits, no overfitting).
    h = TrainingHistory(
        val_score_final=0.003,
        train_score_final=0.003,
        val_train_gap=0.0,
        metric_name="val_loss", lower_is_better=True,
    )
    ctx = _ctx_with_history(raw, h, tmp_path)
    ctx.attach_case_meta(CaseMeta(checkpoint_kind=CheckpointKind.EPOCH))
    findings = evaluate_rulebook(rulebook, ctx)
    leakage = [f for f in findings if f.rule_id == "data_leakage_immediate_dominance"]
    assert leakage, "expected leakage rule to fire on the dominant saturated feature"
    assert leakage[0].feature == "leaky"


def test_spurious_correlation_train_val_gap_fires(rulebook, tmp_path):
    """v0.3: spurious-correlation rule now fires on any high-impact feature with train/val gap,
    regardless of archetype (Interactive Catalyst, Non-linear Driver, etc.)."""
    # f0 has high impact (Interactive Catalyst, NOT Volatile Specialist) + huge val_train_gap.
    archetypes = [3] + [2] * 9
    raw = _build_report(
        feature_names=[f"f{i}" for i in range(10)],
        archetypes=archetypes,
        impact=[1.0] + [0.05] * 9,  # f0 = 20x mean impact
    )
    h = TrainingHistory(
        val_score_final=2.8,
        train_score_final=0.06,
        val_train_gap=0.98,  # huge gap
        metric_name="val_loss", lower_is_better=True,
    )
    ctx = _ctx_with_history(raw, h, tmp_path)
    findings = evaluate_rulebook(rulebook, ctx)
    spurious = [f for f in findings if f.rule_id == "spurious_correlation_train_val_gap"]
    assert spurious, "expected spurious_correlation_train_val_gap to fire on f0 (Interactive Catalyst)"
    assert spurious[0].feature == "f0"


def test_hierarchical_learning_confirmed_fires(rulebook, tmp_path):
    """v0.3: rule now checks relative growth rates, not spike patterns.
    Interaction should grow >2× as fast as Impact across checkpoints.
    v0.8: rule is now applies_when: case.checkpoint_kind=='epoch' because
    'growth across checkpoints' is meaningless on a seed-axis ensemble."""
    from ffca_agent.case_meta import CaseMeta, CheckpointKind
    # Build curves where Interaction grows 10x but Impact only grows 2x (ratio = 5.0).
    n_ckpts = 8
    n_feat = 5
    impact_curve = np.tile(np.linspace(0.1, 0.2, n_ckpts).reshape(-1, 1), (1, n_feat))
    interaction_curve = np.tile(np.linspace(0.05, 0.5, n_ckpts).reshape(-1, 1), (1, n_feat))
    raw = _build_report(
        feature_names=[f"f{i}" for i in range(n_feat)],
        archetypes=[7] * n_feat,
        impact=impact_curve[-1].tolist(),
        impact_curve_per_feature=impact_curve,
        n_checkpoints=n_ckpts,
    )
    # Manually set interaction curve via the report (use the helper to attach it)
    import json as _json
    p = tmp_path / "r.json"
    sigs = []
    for k in range(n_ckpts):
        sigs.append({
            "impact": impact_curve[k].tolist(),
            "volatility": [0.001] * n_feat,
            "nonlinearity": [0.01] * n_feat,
            "interaction": interaction_curve[k].tolist(),
            "archetypes": [7] * n_feat,
        })
    raw["signatures"] = sigs
    p.write_text(_json.dumps(raw))
    ctx = ReportContext.from_json(p)
    ctx.attach_case_meta(CaseMeta(checkpoint_kind=CheckpointKind.EPOCH))
    findings = evaluate_rulebook(rulebook, ctx)
    fired = {f.rule_id for f in findings}
    assert "hierarchical_learning_confirmed" in fired, (
        f"expected hierarchical_learning_confirmed in {fired}"
    )


# ── negative test: dynamic rules skip silently when history is absent ──────


def test_dynamic_rules_skip_silently_without_history(rulebook, tmp_path):
    """No history attached → no crash, no template leaks, none of the 5 fire."""
    raw = _build_report(
        feature_names=[f"f{i}" for i in range(10)],
        archetypes=[2] * 5 + [7] * 5,
        impact=[0.3] * 10,
    )
    p = tmp_path / "r.json"
    p.write_text(json.dumps(raw))
    ctx = ReportContext.from_json(p)
    findings = evaluate_rulebook(rulebook, ctx)
    fired = {f.rule_id for f in findings}
    silent_rules = {
        "overfitting_volatility_spike",
        "data_leakage_immediate_dominance",
        "spurious_correlation_train_val_gap",
    }
    # hierarchical_learning_confirmed is signature-derived (growth-rate of top-k
    # impact vs interaction), so it CAN fire without an attached training history.
    # v0.5: insufficient_capacity is also signature-derived now (any-trigger on
    # archetype-pcts OR nonlinearity_mean) — also legitimately silent or firing
    # based on the report alone.
    assert silent_rules.isdisjoint(fired), (
        f"these should be silent without history: {silent_rules & fired}"
    )
    for f in findings:
        for text in (f.diagnosis, f.recommendation, f.evidence):
            assert "{" not in text or "}" not in text, (
                f"unresolved template in {f.rule_id}: {text!r}"
            )
