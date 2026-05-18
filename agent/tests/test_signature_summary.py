"""Tests for signature_summary: top-K, curve shapes, churn, interaction pairs."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ffca_agent.report import ReportContext
from ffca_agent.signature_summary import signature_summary, _curve_shape


# ── helpers ────────────────────────────────────────────────────────────────


def _make_report(
    feature_names: list[str],
    impact: list[float],
    archetypes: list[int],
    *,
    volatility: list[float] | None = None,
    nonlinearity: list[float] | None = None,
    interaction: list[float] | None = None,
    n_checkpoints: int = 5,
    interaction_matrix: np.ndarray | None = None,
) -> dict:
    n = len(feature_names)
    vol = volatility or [0.001] * n
    nl = nonlinearity or [0.005] * n
    inter = interaction or [0.1] * n
    sigs = [{
        "impact": list(impact),
        "volatility": vol,
        "nonlinearity": nl,
        "interaction": inter,
        "archetypes": archetypes,
    } for _ in range(n_checkpoints)]
    raw: dict = {
        "n_features": n,
        "feature_names": feature_names,
        "signatures": sigs,
        "trust": {f: {"decision": "KEEP (stable)"} for f in feature_names},
        "trust_summary": {},
        "co_sensitivity": None,
        "findings": [],
    }
    if interaction_matrix is not None:
        raw["interaction_matrix"] = interaction_matrix.tolist()
    return raw


def _ctx(raw: dict, tmp_path) -> ReportContext:
    p = tmp_path / "r.json"
    p.write_text(json.dumps(raw))
    return ReportContext.from_json(p)


# ── curve shape ────────────────────────────────────────────────────────────


def test_curve_shape_monotonic_increasing():
    cs = _curve_shape("x", np.array([0.1, 0.5, 1.0, 1.5, 2.0]))
    assert cs.monotonic_increasing is True
    assert cs.monotonic_decreasing is False
    assert cs.fold_change_final_over_initial == pytest.approx(20.0)
    # Monotonic ramp shouldn't be flagged as a spike (peak is at the end).
    assert cs.has_spike is False
    # Final ckpt is 33% above the previous one → late drift is genuinely present.
    assert cs.has_late_drift is True


def test_curve_shape_spike_at_one_checkpoint():
    cs = _curve_shape("x", np.array([0.1, 0.1, 5.0, 0.1, 0.1]))
    assert cs.has_spike is True
    assert cs.peak_checkpoint == 2


def test_curve_shape_late_drift_detected():
    cs = _curve_shape("x", np.array([1.0, 1.0, 1.0, 1.0, 2.0]))
    assert cs.has_late_drift is True


# ── signature_summary integration ──────────────────────────────────────────


def test_top_k_features_by_impact(tmp_path):
    raw = _make_report(
        feature_names=["a", "b", "c", "d", "e", "f"],
        impact=[0.1, 5.0, 0.2, 3.0, 0.05, 1.0],
        archetypes=[0, 7, 0, 7, 0, 2],  # noise/complex_driver/etc
    )
    ctx = _ctx(raw, tmp_path)
    summary = signature_summary(ctx, top_k=3)
    assert len(summary.top_by_impact) == 3
    assert summary.top_by_impact[0].feature == "b"   # 5.0
    assert summary.top_by_impact[1].feature == "d"   # 3.0
    assert summary.top_by_impact[2].feature == "f"   # 1.0


def test_top_k_capped_by_n_features(tmp_path):
    raw = _make_report(
        feature_names=["a", "b"],
        impact=[1.0, 2.0],
        archetypes=[0, 0],
    )
    ctx = _ctx(raw, tmp_path)
    summary = signature_summary(ctx, top_k=10)
    assert len(summary.top_by_impact) == 2


def test_curve_shapes_populated_with_enough_checkpoints(tmp_path):
    raw = _make_report(
        feature_names=["a", "b", "c"],
        impact=[1.0, 2.0, 3.0],
        archetypes=[0, 7, 7],
        n_checkpoints=8,
    )
    ctx = _ctx(raw, tmp_path)
    summary = signature_summary(ctx, top_k=2)
    assert summary.impact_topk_curve_shape is not None
    assert summary.impact_topk_curve_shape.n_checkpoints == 8


def test_archetype_churn_counted(tmp_path):
    # Build a report where feature a's archetype changes every checkpoint
    n_features = 3
    archetypes_per_ckpt = [
        [0, 0, 0],
        [7, 0, 0],
        [0, 0, 0],
        [7, 0, 0],
        [0, 0, 0],
    ]
    sigs = []
    for ar in archetypes_per_ckpt:
        sigs.append({
            "impact": [1.0] * n_features,
            "volatility": [0.001] * n_features,
            "nonlinearity": [0.005] * n_features,
            "interaction": [0.1] * n_features,
            "archetypes": ar,
        })
    raw = {
        "n_features": n_features,
        "feature_names": ["a", "b", "c"],
        "signatures": sigs,
        "trust": {"a": {"decision": "KEEP (stable)"}, "b": {"decision": "KEEP (stable)"}, "c": {"decision": "KEEP (stable)"}},
        "trust_summary": {},
        "co_sensitivity": None,
        "findings": [],
    }
    ctx = _ctx(raw, tmp_path)
    summary = signature_summary(ctx)
    # feature a changes archetype 4 times (≥3) → counted
    assert summary.n_features_with_archetype_churn_ge_3 == 1


def test_interaction_pairs_when_matrix_present(tmp_path):
    n = 4
    M = np.zeros((n, n))
    M[0, 2] = M[2, 0] = 0.9
    M[1, 3] = M[3, 1] = 0.5
    raw = _make_report(
        feature_names=["a", "b", "c", "d"],
        impact=[1.0, 1.0, 1.0, 1.0],
        archetypes=[0, 0, 0, 0],
        interaction_matrix=M,
    )
    ctx = _ctx(raw, tmp_path)
    summary = signature_summary(ctx, top_k=2)
    assert len(summary.top_interaction_pairs) == 2
    # Top pair is a-c (strength 0.9)
    pair = summary.top_interaction_pairs[0]
    assert {pair.feature_a, pair.feature_b} == {"a", "c"}
    assert pair.strength == pytest.approx(0.9)


def test_interaction_pairs_empty_when_matrix_absent(tmp_path):
    raw = _make_report(
        feature_names=["a", "b"],
        impact=[1.0, 1.0],
        archetypes=[0, 0],
    )
    ctx = _ctx(raw, tmp_path)
    summary = signature_summary(ctx)
    assert summary.top_interaction_pairs == []


def test_to_dict_is_json_serialisable(tmp_path):
    raw = _make_report(
        feature_names=["a", "b", "c"],
        impact=[1.0, 2.0, 3.0],
        archetypes=[0, 7, 7],
    )
    ctx = _ctx(raw, tmp_path)
    summary = signature_summary(ctx)
    blob = json.dumps(summary.to_dict(), default=str)
    assert "top_by_impact" in blob


# ── narrator hook ──────────────────────────────────────────────────────────


def test_narrator_user_prompt_includes_summary_when_provided(tmp_path):
    """If sig_summary is passed, the user prompt must include the rule-free
    observation channel block."""
    from ffca_agent.llm import Narrator
    raw = _make_report(
        feature_names=["a", "b", "c"],
        impact=[1.0, 2.0, 3.0],
        archetypes=[0, 7, 7],
    )
    ctx = _ctx(raw, tmp_path)
    summary = signature_summary(ctx)
    prompt = Narrator._build_user_prompt(findings=[], ctx=ctx, training=None,
                                          sig_summary=summary)
    assert "Signature summary" in prompt
    assert "rule_free_observations" in prompt
    assert "top_by_impact" in prompt


def test_narrator_user_prompt_omits_summary_when_none(tmp_path):
    from ffca_agent.llm import Narrator
    raw = _make_report(
        feature_names=["a", "b"],
        impact=[1.0, 2.0],
        archetypes=[0, 0],
    )
    ctx = _ctx(raw, tmp_path)
    prompt = Narrator._build_user_prompt(findings=[], ctx=ctx, training=None)
    assert "Signature summary" not in prompt
