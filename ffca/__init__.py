"""FFCA — Feature-Function Curvature Analysis for any PyTorch model.

Quick start (Python):

    from ffca import FFCAReport
    from ffca.adapters import TabularAdapter

    adapter = TabularAdapter(model, feature_names=cols)
    report = FFCAReport(adapter, val_loader).run()
    report.save("out/")

Or via the CLI: ``ffca-report --help``.
"""
from .adapters import (
    ChannelAdapter,
    PixelAdapter,
    TabularAdapter,
    TransformerEmbeddingAdapter,
    TransformerHeadAdapter,
)
from .checkpoint import CheckpointLoader
from .core import FFCAModelAdapter, FFCASignature
from .improvements_pkg import CauchyHVP, CoSensitivityGroups, TrustScore
from .report import FFCAReport

__version__ = "0.1.0a1"

__all__ = [
    "FFCAReport",
    "FFCAModelAdapter",
    "FFCASignature",
    "TabularAdapter",
    "PixelAdapter",
    "ChannelAdapter",
    "TransformerEmbeddingAdapter",
    "TransformerHeadAdapter",
    "CheckpointLoader",
    "CauchyHVP",
    "TrustScore",
    "CoSensitivityGroups",
]
