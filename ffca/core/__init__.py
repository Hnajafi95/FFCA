"""FFCA core abstractions — model-agnostic building blocks."""
from .adapter import FFCAModelAdapter, _Splice, find_layer
from .archetypes import (
    Archetype,
    ARCHETYPE_NAMES,
    classify,
    is_noise_candidate,
    similarity_matrix,
)
from .derivatives import compute_signature_core
from .signature import FFCASignature
from .smoothing import smooth, n_replaceable_activations
from .scalars import (
    ScalarFn,
    predicted_class,
    target_class,
    true_label,
    regression,
    loss,
    custom,
    from_name as scalar_from_name,
)

__all__ = [
    "FFCAModelAdapter",
    "FFCASignature",
    "Archetype",
    "ARCHETYPE_NAMES",
    "classify",
    "is_noise_candidate",
    "similarity_matrix",
    "compute_signature_core",
    "smooth",
    "n_replaceable_activations",
    "ScalarFn",
    "predicted_class",
    "target_class",
    "true_label",
    "regression",
    "loss",
    "custom",
    "scalar_from_name",
    "_Splice",
    "find_layer",
]
