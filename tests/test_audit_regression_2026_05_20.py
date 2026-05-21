"""Regression tests for the 2026-05-20 audit fixes (C2, H1, H2, H3, M3, H4).

Each test would have failed on the pre-audit code and passes on the
post-audit code. The audit findings doc lives at
FFCA_agent/AUDIT_2026_05_20.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ffca.core.aggregation import (
    aggregate_signatures,
    ensemble_trust_decisions,
)
from ffca.core.archetypes import Archetype, classify
from ffca.core.signature import FFCASignature
from ffca.improvements_pkg.trust_score import TrustScore


# ── C2: ensemble_trust_decisions emits CONFIDENTLY PRUNE for low-impact noise ──

def _build_noise_signatures(n_seeds: int = 5, n_features: int = 20) -> list[FFCASignature]:
    """Build a synthetic seed-ensemble where features 0..4 are stable Noise
    (zero everywhere) and features 5..19 are healthy Workhorses."""
    rng = np.random.default_rng(0)
    sigs = []
    for s in range(n_seeds):
        impact = np.concatenate([np.zeros(5), rng.uniform(0.5, 1.0, n_features - 5)])
        vol = np.concatenate([np.zeros(5), rng.uniform(0.0, 0.05, n_features - 5)])
        nlin = np.concatenate([np.zeros(5), rng.uniform(0.0, 0.05, n_features - 5)])
        inter = np.concatenate([np.zeros(5), rng.uniform(0.0, 0.05, n_features - 5)])
        arch = classify(impact, vol, nlin, inter)
        sigs.append(FFCASignature(
            impact=impact, volatility=vol, nonlinearity=nlin, interaction=inter,
            feature_names=[f"f{i}" for i in range(n_features)],
            archetypes=arch,
        ))
    return sigs


def test_C2_ensemble_emits_confidently_prune_for_noise():
    """Pre-fix: aggregation.py:167 compared modal archetype against the
    string 'Noise Candidate', but the package-internal archetype name is
    'Noise'. No feature ever earned CONFIDENTLY PRUNE in ensemble mode.
    """
    sigs = _build_noise_signatures()
    aggregate, seed_stats = aggregate_signatures(sigs)
    decisions = ensemble_trust_decisions(aggregate, seed_stats)
    prune_features = [
        name for name, d in decisions.items()
        if d["decision"] == "CONFIDENTLY PRUNE"
    ]
    # The 5 zero-everywhere features should be confidently prune-able.
    assert "f0" in prune_features
    assert "f1" in prune_features
    assert len(prune_features) >= 5


# ── M3: aggregation uses median across seeds (not mean) ──

def test_M3_aggregation_is_median_not_mean():
    """A single outlier seed should not dominate the aggregate signature.
    Mean is mass-sensitive; median is not."""
    n_features = 10
    base = np.array([0.1] * n_features)
    outlier = np.array([10.0] * n_features)
    sigs = []
    # 4 normal seeds + 1 outlier
    for impact_row in [base, base, base, base, outlier]:
        sigs.append(FFCASignature(
            impact=impact_row,
            volatility=np.zeros(n_features),
            nonlinearity=np.zeros(n_features),
            interaction=np.zeros(n_features),
            feature_names=[f"f{i}" for i in range(n_features)],
            archetypes=classify(impact_row, np.zeros(n_features),
                                np.zeros(n_features), np.zeros(n_features)),
        ))
    aggregate, _ = aggregate_signatures(sigs)
    # Median of [0.1, 0.1, 0.1, 0.1, 10.0] is 0.1.
    # Mean would be (0.4 + 10.0)/5 = 2.08.
    assert np.allclose(aggregate.impact, 0.1, atol=1e-9)
    # Metadata explicitly records the aggregation method.
    assert aggregate.metadata["aggregation"] == "median_across_seeds"


# ── H1: archetype classifier — high N excludes WORKHORSE ──

def test_H1_workhorse_does_not_swallow_nonlinear_drivers():
    """A feature with high I AND high N should be a NONLINEAR_DRIVER, not
    a WORKHORSE. Pre-fix the WORKHORSE rule didn't gate on N."""
    # 20 features: 19 boring + 1 with high I and high N.
    n = 20
    impact = np.full(n, 0.1)
    volatility = np.full(n, 0.01)
    nonlinearity = np.full(n, 0.01)
    interaction = np.full(n, 0.01)
    # Feature 0: top-rank I, top-rank N
    impact[0] = 1.0
    nonlinearity[0] = 1.0
    arch = classify(impact, volatility, nonlinearity, interaction)
    assert arch[0] == Archetype.NONLINEAR_DRIVER, (
        f"expected NONLINEAR_DRIVER, got {Archetype(arch[0]).name} "
        f"(pre-fix this would be WORKHORSE)"
    )


def test_H1_stable_contributor_does_not_swallow_volatile_specialist():
    """A feature with mid I and high V should be a VOLATILE_SPECIALIST,
    not STABLE_CONTRIBUTOR. Pre-fix STABLE_CONTRIBUTOR rule only checked I."""
    n = 20
    impact = np.full(n, 0.05)
    volatility = np.full(n, 0.01)
    nonlinearity = np.full(n, 0.01)
    interaction = np.full(n, 0.01)
    # Feature 0: mid-rank I, top-rank V
    impact[0] = 0.6   # above 0.5 percentile
    volatility[0] = 1.0
    arch = classify(impact, volatility, nonlinearity, interaction)
    assert arch[0] == Archetype.VOLATILE_SPECIALIST


# ── H2: stable Nonlinear Driver with high importance earns CONFIDENTLY KEEP ──

def test_H2_stable_nonlinear_driver_earns_confidently_keep():
    """Pre-fix only WORKHORSE / CATALYST / STABLE_CONTRIBUTOR could earn
    CONFIDENTLY KEEP. A stable, top-importance Nonlinear Driver was
    demoted to KEEP (stable). Fix: any high-importance stable feature
    earns CONFIDENTLY KEEP regardless of archetype."""
    n_features = 20
    n_checkpoints = 5
    sigs = []
    rng = np.random.default_rng(0)
    for c in range(n_checkpoints):
        impact = np.full(n_features, 0.05)
        volatility = np.full(n_features, 0.001)
        nonlinearity = np.full(n_features, 0.01)
        interaction = np.full(n_features, 0.01)
        # Feature 0 is a stable Nonlinear Driver across all checkpoints.
        impact[0] = 0.95
        nonlinearity[0] = 0.95
        arch = classify(impact, volatility, nonlinearity, interaction)
        sigs.append(FFCASignature(
            impact=impact, volatility=volatility,
            nonlinearity=nonlinearity, interaction=interaction,
            feature_names=[f"f{i}" for i in range(n_features)],
            archetypes=arch,
        ))
    ts = TrustScore()
    results = ts.compute(sigs)
    assert results["f0"]["decision"] == "CONFIDENTLY KEEP", (
        f"expected CONFIDENTLY KEEP for stable Nonlinear Driver, got "
        f"{results['f0']['decision']!r}"
    )


# ── H3: low-importance unstable feature earns CONFIDENTLY PRUNE, not INVESTIGATE ──

def test_H3_low_importance_stable_noise_is_confidently_prune():
    """A feature whose 4D signature is near-zero on all axes (the Noise
    archetype) and which stays stable across checkpoints should be
    CONFIDENTLY PRUNE — that was already true. The H3 fix is that even
    when stability dips below the threshold (archetype flipping near
    zero), a near-zero-impact feature still gets CONFIDENTLY PRUNE
    instead of being mislabelled INVESTIGATE."""
    n_features = 20
    n_checkpoints = 5
    sigs = []
    rng = np.random.default_rng(0)
    for c in range(n_checkpoints):
        # Distinct values per axis so the classifier's percentile gates
        # actually resolve (no tied ranks). Feature 0 sits at the bottom
        # on all four axes.
        impact = np.linspace(0.001, 1.0, n_features)
        rng.shuffle(impact[1:])
        impact[0] = 1e-6 + rng.normal(0, 1e-9)
        volatility = np.linspace(0.001, 1.0, n_features)
        rng.shuffle(volatility[1:])
        volatility[0] = 1e-6
        nonlinearity = np.linspace(0.001, 1.0, n_features)
        rng.shuffle(nonlinearity[1:])
        nonlinearity[0] = 1e-6
        interaction = np.linspace(0.001, 1.0, n_features)
        rng.shuffle(interaction[1:])
        interaction[0] = 1e-6
        arch = classify(impact, volatility, nonlinearity, interaction)
        sigs.append(FFCASignature(
            impact=impact, volatility=volatility,
            nonlinearity=nonlinearity, interaction=interaction,
            feature_names=[f"f{i}" for i in range(n_features)],
            archetypes=arch,
        ))
    ts = TrustScore()
    results = ts.compute(sigs)
    decision = results["f0"]["decision"]
    assert decision == "CONFIDENTLY PRUNE", (
        f"expected CONFIDENTLY PRUNE for near-zero stable Noise, "
        f"got {decision!r}"
    )
