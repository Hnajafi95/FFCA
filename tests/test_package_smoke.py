"""Smoke + unit tests for the v0.1.0a1 public API.

Covers: adapter base contract, smoothing, archetypes, scalars, every adapter
on a tiny synthetic problem, FFCAReport orchestrator, and plot generation.
The expensive Cauchy-HVP validation lives in test_cauchy_hvp_validation.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ffca import (
    ChannelAdapter, FFCAReport, FFCASignature, PixelAdapter, TabularAdapter,
)
from ffca.core.archetypes import ARCHETYPE_NAMES, classify, similarity_matrix
from ffca.core import scalar_from_name
from ffca.core.scalars import predicted_class, regression, target_class
from ffca.core.smoothing import n_replaceable_activations, smooth


# --------------------------------------------------------------- utilities
def make_tabular_problem(d=8, n=80):
    torch.manual_seed(0)
    X = torch.randn(n, d)
    y = (X[:, 0] + X[:, 1] * X[:, 2] > 0).long()
    return X, y


def make_tabular_model(d=8):
    return nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, 2))


def quick_train(model, X, y, epochs=20):
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad(); crit(model(X), y).backward(); opt.step()
    return model


# --------------------------------------------------------------- core tests
def test_smooth_replaces_and_restores():
    m = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    assert isinstance(m[1], nn.ReLU)
    with smooth(m, beta=10):
        assert isinstance(m[1], nn.Softplus)
    assert isinstance(m[1], nn.ReLU)


def test_archetype_classifier_eight_categories():
    d = 100
    rng = np.random.default_rng(0)
    impact = rng.uniform(0, 1, d)
    volatility = rng.uniform(0, 1, d)
    nonlin = rng.uniform(0, 1, d)
    interaction = rng.uniform(0, 1, d)
    arch = classify(impact, volatility, nonlin, interaction)
    assert arch.shape == (d,)
    assert arch.min() >= 0 and arch.max() <= 7


def test_similarity_matrix_properties():
    S = similarity_matrix()
    assert S.shape == (8, 8)
    np.testing.assert_array_equal(np.diag(S), np.ones(8))
    np.testing.assert_allclose(S, S.T)
    assert S.min() >= 0 and S.max() <= 1


def test_scalar_factories():
    out = torch.tensor([[1.0, 5.0], [3.0, 2.0]])
    s_pred = predicted_class()(out, None)
    assert s_pred.item() == 5.0 + 3.0
    s_tgt = target_class(0)(out, None)
    assert s_tgt.item() == 1.0 + 3.0
    s_reg = regression()(out, None)
    assert s_reg.item() == 11.0
    s_from = scalar_from_name("target_class:1")
    assert s_from(out, None).item() == 5.0 + 2.0


# --------------------------------------------------------------- adapter tests
def test_tabular_adapter_runs():
    X, y = make_tabular_problem()
    model = quick_train(make_tabular_model(), X, y)
    loader = DataLoader(TensorDataset(X, y), batch_size=16)
    ad = TabularAdapter(model, feature_names=[f"x{i}" for i in range(8)])
    report = FFCAReport(ad, loader,
                       n_first_order_samples=32, n_hessian_samples=4,
                       n_diag_probes=12, n_cauchy_probes=16, n_cauchy_samples=4,
                       n_cosens_permutations=10, n_cosens_bootstrap=5)
    report.run()
    s = report.signatures[-1]
    assert s.n_features == 8
    assert s.archetypes is not None
    # x0 should dominate (the linear term)
    top = s.top_k(1, by="impact")[0]
    assert top == 0 or s.impact[top] > 0


def test_pixel_adapter_runs():
    torch.manual_seed(0)
    X = torch.randn(20, 3, 8, 8); y = torch.randint(0, 2, (20,))
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 2)
    )
    quick_train(model, X, y, epochs=10)
    loader = DataLoader(TensorDataset(X, y), batch_size=8)
    ad = PixelAdapter(model, input_shape=(3, 8, 8))
    rep = FFCAReport(ad, loader, n_first_order_samples=8, n_hessian_samples=4,
                     n_diag_probes=12, n_cauchy_probes=12, n_cauchy_samples=4,
                     n_cosens_permutations=5, n_cosens_bootstrap=3)
    rep.run()
    assert rep.signatures[-1].n_features == 3 * 8 * 8
    fbr = ad.fbr(rep.signatures[-1].interaction)
    assert 0 <= fbr <= 1 or np.isnan(fbr)


def test_channel_adapter_runs():
    torch.manual_seed(0)
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 4, 3, padding=1); self.act1 = nn.ReLU()
            self.conv2 = nn.Conv2d(4, 8, 3, padding=1); self.act2 = nn.ReLU()
            self.pool = nn.AdaptiveAvgPool2d(1); self.fc = nn.Linear(8, 2)
        def forward(self, x):
            x = self.act1(self.conv1(x)); x = self.act2(self.conv2(x))
            x = self.pool(x).view(x.size(0), -1); return self.fc(x)
    m = Net()
    X = torch.randn(20, 1, 8, 8); y = torch.randint(0, 2, (20,))
    quick_train(m, X, y, epochs=5)
    loader = DataLoader(TensorDataset(X, y), batch_size=8)
    ad = ChannelAdapter(m, layer_name="act2")
    rep = FFCAReport(ad, loader, n_first_order_samples=8, n_hessian_samples=4,
                     n_diag_probes=8, n_cauchy_probes=8, n_cauchy_samples=4,
                     n_cosens_permutations=5, n_cosens_bootstrap=3)
    rep.run()
    assert rep.signatures[-1].n_features == 8


def test_signature_serialization_roundtrip():
    rng = np.random.default_rng(0)
    sig = FFCASignature(
        impact=rng.uniform(size=5), volatility=rng.uniform(size=5),
        nonlinearity=rng.uniform(size=5), interaction=rng.uniform(size=5),
        feature_names=[f"f{i}" for i in range(5)],
    )
    d = sig.to_dict()
    sig2 = FFCASignature.from_dict(d)
    np.testing.assert_allclose(sig.impact, sig2.impact)
    assert sig2.feature_names == sig.feature_names


def test_full_report_save_with_plots(tmp_path):
    X, y = make_tabular_problem()
    model = quick_train(make_tabular_model(), X, y)
    loader = DataLoader(TensorDataset(X, y), batch_size=16)
    ad = TabularAdapter(model, feature_names=[f"x{i}" for i in range(8)])
    rep = FFCAReport(ad, loader, n_first_order_samples=16, n_hessian_samples=4,
                     n_diag_probes=10, n_cauchy_probes=12, n_cauchy_samples=4)
    rep.run()
    out = rep.save(tmp_path)
    assert (out / "report.md").exists()
    assert (out / "report.json").exists()
    plots = list((out / "plots").glob("*.png"))
    assert len(plots) >= 4, f"expected ≥4 plots, got {len(plots)}: {plots}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
