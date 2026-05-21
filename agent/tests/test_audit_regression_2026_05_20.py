"""Regression tests for the 2026-05-20 audit fixes on the agent side
(C3, H5, H6). The audit findings doc lives at
FFCA_agent/AUDIT_2026_05_20.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ffca_agent.case_meta import CaseMeta, CheckpointKind, ModelArchitecture, TaskType
from ffca_agent.evaluator import _check_applies_when
from ffca_agent.report import MissingSignal, ReportContext


def _minimal_report(
    feature_names: list[str],
    impact: list[float],
    trust_decisions: dict[str, str] | None = None,
) -> dict:
    n = len(feature_names)
    zero = [0.0] * n
    sig = {
        "impact": impact,
        "volatility": zero,
        "nonlinearity": zero,
        "interaction": zero,
        "archetypes": [0] * n,
    }
    return {
        "n_features": n,
        "feature_names": feature_names,
        "signatures": [sig],
        "trust": {f: {"decision": (trust_decisions or {}).get(f, "")}
                  for f in feature_names},
        "trust_summary": {},
        "co_sensitivity": None,
        "findings": [],
    }


# ── C3: agent buckets ensemble-mode INVESTIGATE (multi-modal seeds) ──

def test_C3_ensemble_investigate_bucketed(tmp_path):
    """Pre-fix the bucket parser required the literal substring
    'INVESTIGATE (unstable)' inside the decision string, so
    'INVESTIGATE (multi-modal seeds)' (ensemble mode) was dropped on
    the floor. trust.investigate.count was stuck at zero, which
    silently disabled the trust_multi_modal_seeds rule.
    """
    raw = _minimal_report(
        feature_names=["a", "b", "c", "d"],
        impact=[0.5, 0.1, 0.0, 0.7],
        trust_decisions={
            "a": "INVESTIGATE (multi-modal seeds)",
            "b": "INVESTIGATE (unstable)",
            "c": "CONFIDENTLY PRUNE",
            "d": "CONFIDENTLY KEEP",
        },
    )
    p = tmp_path / "r.json"
    p.write_text(json.dumps(raw))
    ctx = ReportContext.from_json(p)
    # Both flavours of INVESTIGATE should land in the investigate bucket.
    assert ctx.trust_buckets["investigate"].count == 2
    assert ctx.trust_buckets["confident_keep"].count == 1
    assert ctx.trust_buckets["confident_prune"].count == 1


# ── H5: epoch-axis signals raise MissingSignal on seed-axis case ──

def test_H5_impact_saturation_blocked_on_seed_axis(tmp_path):
    """feature.impact_saturation interprets checkpoint[0] as 'start of
    training' — meaningless when the checkpoints are independent seeds.
    Pre-fix it computed a bogus number; post-fix it raises MissingSignal
    so rules that read it (data_leakage_immediate_dominance) cannot
    misfire on seed-axis cases.
    """
    raw = _minimal_report(
        feature_names=["a", "b"], impact=[1.0, 0.1],
    )
    # Make multiple checkpoints so impact_curve has rows to read.
    raw["signatures"] = raw["signatures"] * 3
    p = tmp_path / "r.json"
    p.write_text(json.dumps(raw))
    ctx = ReportContext.from_json(p)
    cm = CaseMeta(
        project_name="seed-test",
        model_architecture=ModelArchitecture.MLP,
        task_type=TaskType.REGRESSION,
        target_name="y",
        n_seeds=3,
        checkpoint_kind=CheckpointKind.SEED,
    )
    ctx.attach_case_meta(cm)
    with pytest.raises(MissingSignal):
        ctx.get("feature.impact_saturation", feature_idx=0)
    with pytest.raises(MissingSignal):
        ctx.get("model.checkpoint_drift_l2_pct")


def test_H5_impact_saturation_allowed_on_epoch_axis(tmp_path):
    """Sanity-check: same signal returns a value when checkpoint_kind=epoch."""
    raw = _minimal_report(
        feature_names=["a", "b"], impact=[1.0, 0.1],
    )
    raw["signatures"] = raw["signatures"] * 3
    p = tmp_path / "r.json"
    p.write_text(json.dumps(raw))
    ctx = ReportContext.from_json(p)
    cm = CaseMeta(
        project_name="epoch-test",
        checkpoint_kind=CheckpointKind.EPOCH,
    )
    ctx.attach_case_meta(cm)
    val = ctx.get("feature.impact_saturation", feature_idx=0)
    assert val is not None
    assert isinstance(val, float)


def test_H5_legacy_no_case_meta_preserves_behavior(tmp_path):
    """When no case_meta is attached, the epoch-axis signals continue to
    work — preserves backward compatibility with pre-v0.7 callers."""
    raw = _minimal_report(
        feature_names=["a", "b"], impact=[1.0, 0.1],
    )
    raw["signatures"] = raw["signatures"] * 3
    p = tmp_path / "r.json"
    p.write_text(json.dumps(raw))
    ctx = ReportContext.from_json(p)
    # No case_meta attached. Should still return a value.
    val = ctx.get("feature.impact_saturation", feature_idx=0)
    assert isinstance(val, float)


# ── H6: _check_applies_when raises on un-parseable expressions ──

def test_H6_applies_when_raises_on_typo():
    """Pre-fix a YAML typo like `case.X = 'epoch'` (single equals)
    silently returned True so the rule fired unconditionally. Now it
    raises so the bug surfaces during rulebook load / evaluation."""
    class _DummyCtx:
        case_meta = None

    with pytest.raises(ValueError, match="not parseable"):
        _check_applies_when("case.checkpoint_kind = 'epoch'", _DummyCtx())
    with pytest.raises(ValueError, match="not parseable"):
        _check_applies_when("checkpoint_kind == 'epoch'", _DummyCtx())  # missing case.
    with pytest.raises(ValueError, match="not parseable"):
        _check_applies_when("case.checkpoint_kind == epoch", _DummyCtx())  # no quotes


def test_H6_applies_when_supports_in_and_not_in():
    """New grammar: `case.X in (...)`. Useful for rules that apply on
    both seed and mixed axes but not epoch."""
    class _Ctx:
        def __init__(self, kind):
            from ffca_agent.case_meta import CheckpointKind
            self.case_meta = CaseMeta(checkpoint_kind=CheckpointKind(kind))

    seed_ctx = _Ctx("seed")
    epoch_ctx = _Ctx("epoch")

    expr = "case.checkpoint_kind in ('seed', 'mixed')"
    assert _check_applies_when(expr, seed_ctx) is True
    assert _check_applies_when(expr, epoch_ctx) is False

    expr2 = "case.checkpoint_kind not_in ('epoch',)"
    assert _check_applies_when(expr2, seed_ctx) is True
    assert _check_applies_when(expr2, epoch_ctx) is False


def test_H6_applies_when_existing_equality_grammar_preserved():
    """Ensure the v0.7 `case.X == 'Y'` grammar still works after the
    parser was tightened."""
    class _Ctx:
        def __init__(self, kind):
            from ffca_agent.case_meta import CheckpointKind
            self.case_meta = CaseMeta(checkpoint_kind=CheckpointKind(kind))

    assert _check_applies_when(
        "case.checkpoint_kind == 'epoch'", _Ctx("epoch")
    ) is True
    assert _check_applies_when(
        "case.checkpoint_kind == 'epoch'", _Ctx("seed")
    ) is False
    assert _check_applies_when(
        "case.checkpoint_kind != 'epoch'", _Ctx("seed")
    ) is True
