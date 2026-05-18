"""Unit tests for time-series detectors."""

from __future__ import annotations

import numpy as np

from ffca_agent.timeseries import collapse_detected, plateau_detected, spike_detected


# ── spike_detected ──────────────────────────────────────────────────────────

def test_spike_epoch_0_fires_when_first_value_dominates():
    # leaky_feature: explodes from epoch 0, then steady
    curve = [10.0] + [1.0] * 20
    res = spike_detected(curve, when="epoch_0")
    assert res.fired
    assert res.epoch == 0
    assert res.ratio > 5


def test_spike_epoch_0_does_not_fire_when_early_value_is_normal():
    curve = [0.1] + [1.0] * 20
    res = spike_detected(curve, when="epoch_0")
    assert not res.fired


def test_spike_early_epochs():
    # Volatility spike early then declining (App C.6 spurious correlation)
    early = [0.1, 0.5, 2.0, 3.5, 4.0, 3.0, 2.0]
    late = [1.0] * 10
    curve = early + late
    res = spike_detected(curve, when="early_epochs", threshold_ratio=2.0)
    assert res.fired
    assert res.epoch < len(early)


def test_spike_late_epochs():
    # Interaction takeoff late in training (Fig 1 Credit Loan)
    flat = [0.1] * 20
    rising = [0.5, 1.0, 2.0, 3.0]
    res = spike_detected(flat + rising, when="late_epochs", threshold_ratio=3.0)
    assert res.fired


def test_spike_too_short_curve_returns_false():
    res = spike_detected([1.0, 2.0])
    assert not res.fired


# ── plateau_detected ────────────────────────────────────────────────────────

def test_plateau_fires_when_tail_flat():
    curve = list(np.linspace(0, 0.9, 10)) + [0.9] * 10
    res = plateau_detected(curve)
    assert res.fired


def test_plateau_does_not_fire_when_still_improving():
    curve = list(np.linspace(0, 0.99, 30))
    res = plateau_detected(curve)
    assert not res.fired


# ── collapse_detected ───────────────────────────────────────────────────────

def test_collapse_fires_below_absolute_threshold():
    # FBR collapses from 1.15 to 0.40 (Waterbirds drift epoch)
    curve = [1.154, 1.0, 0.8, 0.6, 0.5, 0.4]
    res = collapse_detected(curve, final_threshold=0.5)
    assert res.fired
    assert res.ratio < 0.5


def test_collapse_does_not_fire_above_threshold():
    curve = [1.0, 0.95, 0.9, 0.85]
    res = collapse_detected(curve, final_threshold=0.5)
    assert not res.fired
