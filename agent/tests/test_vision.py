"""Tests for the vision adapter and shortcut_learning_drift_epoch rule.

Synthetic vision curves matching paper App C.4 / Table 7 (Waterbirds drift at
epoch 9): FBR collapses from above 1 to below 0.5, COM distance grows late,
minority accuracy plateaus. Plus loader unit tests and a negative test:
without vision metrics, the rule skips silently.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ffca_agent.evaluator import evaluate_rulebook, load_rulebook
from ffca_agent.report import ReportContext
from ffca_agent.vision import (
    VisionMetrics,
    compute_com_distance,
    compute_fbr,
    compute_minority_acc,
)

REPO = Path(__file__).resolve().parents[1]
RULEBOOK_PATH = REPO / "rulebook" / "ffca_rules.yaml"


@pytest.fixture(scope="module")
def rulebook():
    return load_rulebook(RULEBOOK_PATH)


# ── compute helpers ────────────────────────────────────────────────────────


def test_compute_fbr_high_when_attribution_on_foreground():
    h, w = 32, 32
    attribution = np.zeros((h, w))
    attribution[8:16, 8:16] = 1.0  # mass on a square
    foreground = np.zeros((h, w), dtype=bool)
    foreground[8:16, 8:16] = True  # foreground matches the mass exactly
    assert compute_fbr(attribution, foreground) > 100


def test_compute_fbr_low_when_attribution_on_background():
    h, w = 32, 32
    attribution = np.zeros((h, w))
    attribution[20:30, 20:30] = 1.0  # mass on background
    foreground = np.zeros((h, w), dtype=bool)
    foreground[2:8, 2:8] = True
    assert compute_fbr(attribution, foreground) < 0.5


def test_compute_com_distance_zero_when_aligned():
    h, w = 32, 32
    attribution = np.zeros((h, w))
    foreground = np.zeros((h, w), dtype=bool)
    attribution[14:18, 14:18] = 1.0
    foreground[14:18, 14:18] = True
    assert compute_com_distance(attribution, foreground) < 0.01


def test_compute_com_distance_large_when_misaligned():
    h, w = 32, 32
    attribution = np.zeros((h, w))
    foreground = np.zeros((h, w), dtype=bool)
    attribution[0:4, 0:4] = 1.0       # top-left
    foreground[28:32, 28:32] = True   # bottom-right
    assert compute_com_distance(attribution, foreground) > 0.6


def test_compute_minority_acc_picks_minority_groups():
    preds = [0, 1, 0, 1, 1, 0]
    labels = [0, 1, 1, 1, 1, 0]
    groups = [0, 1, 2, 1, 3, 0]
    # minority groups = (1, 2): indices 1, 2, 3 → preds [1,0,1], labels [1,1,1] → 2/3
    assert compute_minority_acc(preds, labels, groups, minority_groups=(1, 2)) == pytest.approx(2 / 3)


# ── VisionMetrics loader ───────────────────────────────────────────────────


def test_vision_metrics_round_trip(tmp_path):
    m = VisionMetrics(
        fbr_curve=np.array([2.0, 1.5, 1.0, 0.5, 0.3]),
        com_distance_curve=np.array([0.1, 0.1, 0.2, 0.4, 0.6]),
        minority_acc_curve=np.array([0.3, 0.5, 0.5, 0.5, 0.5]),
        overall_acc_curve=np.array([0.6, 0.75, 0.85, 0.87, 0.88]),
        epoch_labels=["ep1", "ep3", "ep5", "ep9", "ep15"],
    )
    p = tmp_path / "v.json"
    m.save(p)
    loaded = VisionMetrics.from_json(p)
    assert np.allclose(loaded.fbr_curve, m.fbr_curve)
    assert loaded.epoch_labels == m.epoch_labels


def test_vision_metrics_partial_curves_preserved():
    m = VisionMetrics.from_dict({"fbr_curve": [2.0, 1.0, 0.4]})
    assert m.com_distance_curve is None
    assert m.minority_acc_curve is None
    d = m.as_signal_dict()
    assert "fbr_curve" in d
    assert "com_distance_curve" not in d  # None curves excluded


# ── rule firing ────────────────────────────────────────────────────────────


def _basic_ctx(tmp_path):
    raw = {
        "n_features": 5,
        "feature_names": [f"c{i}" for i in range(5)],
        "signatures": [{
            "impact": [0.3, 0.2, 0.1, 0.05, 0.02],
            "volatility": [0.001] * 5,
            "nonlinearity": [0.01] * 5,
            "interaction": [0.05] * 5,
            "archetypes": [2, 2, 0, 0, 0],
        }] * 3,
        "trust": {f"c{i}": {"decision": "KEEP (stable)"} for i in range(5)},
        "trust_summary": {},
        "co_sensitivity": None,
        "findings": [],
    }
    p = tmp_path / "r.json"
    p.write_text(json.dumps(raw))
    return ReportContext.from_json(p)


def test_shortcut_learning_drift_epoch_fires_when_all_signals_present(rulebook, tmp_path):
    ctx = _basic_ctx(tmp_path)
    metrics = VisionMetrics(
        # FBR collapses below 0.5 final threshold
        fbr_curve=np.array([2.0, 1.8, 1.5, 1.0, 0.7, 0.4, 0.35, 0.3]),
        # COM distance spike late
        com_distance_curve=np.array([0.05, 0.05, 0.06, 0.08, 0.15, 0.25, 0.35, 0.4]),
        # Minority accuracy plateaus
        minority_acc_curve=np.array([0.20, 0.40, 0.42, 0.43, 0.43, 0.43, 0.43, 0.43]),
    )
    ctx.attach_vision_metrics(metrics)
    findings = evaluate_rulebook(rulebook, ctx)
    fired = {f.rule_id for f in findings}
    assert "shortcut_learning_drift_epoch" in fired


def test_shortcut_learning_drift_epoch_skips_when_vision_absent(rulebook, tmp_path):
    ctx = _basic_ctx(tmp_path)
    findings = evaluate_rulebook(rulebook, ctx)
    fired = {f.rule_id for f in findings}
    assert "shortcut_learning_drift_epoch" not in fired
    assert "shortcut_learning_minority_gap" not in fired
    # no template leaks
    for f in findings:
        for text in (f.diagnosis, f.recommendation, f.evidence):
            assert "{" not in text or "}" not in text


def test_majority_minority_gap_curve_computed_from_overall_minus_minority():
    m = VisionMetrics(
        overall_acc_curve=np.array([0.60, 0.70, 0.80, 0.85]),
        minority_acc_curve=np.array([0.55, 0.60, 0.65, 0.68]),
    )
    gap = m.majority_minority_gap_curve()
    assert np.allclose(gap, [0.05, 0.10, 0.15, 0.17])
    d = m.as_signal_dict()
    assert "majority_minority_gap_curve" in d
    assert pytest.approx(d["majority_minority_gap_max"], abs=1e-6) == 0.17


def test_shortcut_learning_minority_gap_fires_when_fbr_high_but_gap_present(rulebook, tmp_path):
    """v0.5 Waterbirds-style case: pretrained backbone keeps FBR high but
    minority-group accuracy lags overall. The new rule should still fire."""
    ctx = _basic_ctx(tmp_path)
    metrics = VisionMetrics(
        # FBR high and rising — paper-style shortcut fingerprint absent.
        fbr_curve=np.array([1.2, 5.0, 10.0, 20.0]),
        # COM stable (no spike).
        com_distance_curve=np.array([0.15, 0.15, 0.14, 0.14]),
        # Behavioural gap: overall 0.81, minority 0.64 → 17pp.
        overall_acc_curve=np.array([0.55, 0.68, 0.78, 0.81]),
        minority_acc_curve=np.array([0.48, 0.60, 0.63, 0.64]),
    )
    ctx.attach_vision_metrics(metrics)
    findings = evaluate_rulebook(rulebook, ctx)
    fired = {f.rule_id for f in findings}
    assert "shortcut_learning_minority_gap" in fired
    # Paper-fingerprint rule should NOT fire here (FBR didn't collapse).
    assert "shortcut_learning_drift_epoch" not in fired


def test_shortcut_learning_minority_gap_skips_when_gap_below_threshold(rulebook, tmp_path):
    ctx = _basic_ctx(tmp_path)
    metrics = VisionMetrics(
        overall_acc_curve=np.array([0.80, 0.85, 0.88, 0.90]),
        minority_acc_curve=np.array([0.78, 0.84, 0.86, 0.88]),  # max gap = 2pp
    )
    ctx.attach_vision_metrics(metrics)
    findings = evaluate_rulebook(rulebook, ctx)
    fired = {f.rule_id for f in findings}
    assert "shortcut_learning_minority_gap" not in fired
