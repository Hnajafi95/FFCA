"""8-archetype classifier + archetype similarity matrix.

The archetype rules are percentile-based and order-dependent (the first
matching rule wins). This is the same logic that was in the audit-v2
TrustScore implementation, lifted here so it can be shared by:
  - the core signature pipeline (sets `signature.archetypes`)
  - improvements/trust_score.py (computes archetype distributions)
  - improvements/co_sensitivity.py (labels groups by NC fraction)
"""
from __future__ import annotations

from enum import IntEnum

import numpy as np
from scipy import stats


class Archetype(IntEnum):
    NOISE = 0
    HIDDEN_INTERACTOR = 1
    WORKHORSE = 2
    CATALYST = 3
    NONLINEAR_DRIVER = 4
    VOLATILE_SPECIALIST = 5
    STABLE_CONTRIBUTOR = 6
    COMPLEX_DRIVER = 7

    @property
    def display(self) -> str:
        return ARCHETYPE_NAMES[int(self)]


ARCHETYPE_NAMES = [
    "Noise",
    "Hidden Interactor",
    "Workhorse",
    "Catalyst",
    "Nonlinear Driver",
    "Volatile Specialist",
    "Stable Contributor",
    "Complex Driver",
]

# Binary (I, V, N, X) profile for each archetype. The classifier doesn't use
# this directly (it uses percentile rules below) but the similarity matrix
# derives from it.
_ARCHETYPE_CODES = np.array([
    # I  V  N  X
    [0, 0, 0, 0],  # 0 Noise
    [0, 0, 0, 1],  # 1 Hidden Interactor
    [1, 0, 0, 0],  # 2 Workhorse
    [1, 0, 0, 1],  # 3 Catalyst
    [1, 0, 1, 0],  # 4 Nonlinear Driver
    [1, 1, 0, 0],  # 5 Volatile Specialist
    [1, 0, 0, 0],  # 6 Stable Contributor (≈ Workhorse, milder I)
    [1, 1, 1, 1],  # 7 Complex Driver
], dtype=np.float64)


def classify(
    impact: np.ndarray,
    volatility: np.ndarray,
    nonlinearity: np.ndarray,
    interaction: np.ndarray,
) -> np.ndarray:
    """Assign one of the 8 archetypes to each feature.

    Returns an int array of shape (d,) with values in [0, 7].
    Order-dependent: the first matching rule wins. Rules match the FFCA paper.
    """
    n = len(impact)
    if n == 0:
        return np.array([], dtype=int)

    def _ranks(x: np.ndarray) -> np.ndarray:
        return stats.rankdata(x) / max(n, 1)

    i_r = _ranks(np.asarray(impact))
    v_r = _ranks(np.asarray(volatility))
    n_r = _ranks(np.asarray(nonlinearity))
    x_r = _ranks(np.asarray(interaction))

    arch = np.empty(n, dtype=int)
    # First-match wins. WORKHORSE and STABLE_CONTRIBUTOR were previously
    # under-constrained: their binary fingerprint is (I=1, V=0, N=0, X=0)
    # but the rules did not gate on N, so a feature with high I AND high N
    # was classified as WORKHORSE instead of NONLINEAR_DRIVER. Add the N
    # constraint to make the rule match the fingerprint.
    for i in range(n):
        if i_r[i] < 0.3 and v_r[i] < 0.3 and n_r[i] < 0.3 and x_r[i] < 0.3:
            arch[i] = Archetype.NOISE
        elif x_r[i] > 0.75 and i_r[i] < 0.5:
            arch[i] = Archetype.HIDDEN_INTERACTOR
        elif i_r[i] > 0.7 and v_r[i] < 0.3 and n_r[i] < 0.3 and x_r[i] < 0.3:
            arch[i] = Archetype.WORKHORSE
        elif i_r[i] > 0.5 and x_r[i] > 0.75:
            arch[i] = Archetype.CATALYST
        elif n_r[i] > 0.7:
            arch[i] = Archetype.NONLINEAR_DRIVER
        elif v_r[i] > 0.7:
            arch[i] = Archetype.VOLATILE_SPECIALIST
        elif i_r[i] > 0.5 and v_r[i] < 0.5 and n_r[i] < 0.5 and x_r[i] < 0.5:
            arch[i] = Archetype.STABLE_CONTRIBUTOR
        else:
            arch[i] = Archetype.COMPLEX_DRIVER
    return arch


def similarity_matrix(scale: float = 1.5) -> np.ndarray:
    """8×8 matrix of semantic similarity between archetypes.

    Used by TrustScore's weighted-entropy stability so that flips between
    near-identical archetypes don't punish stability as much as flips between
    dissimilar ones.
    """
    codes = _ARCHETYPE_CODES
    # Hamming distance over the 4 binary axes
    dist = np.abs(codes[:, None, :] - codes[None, :, :]).sum(-1)
    S = np.exp(-dist / scale)
    # Specific bridges based on FFCA semantics
    S[6, 0] = S[0, 6] = max(S[6, 0], 0.3)   # Stable ↔ Noise (low impact bridge)
    S[6, 2] = S[2, 6] = max(S[6, 2], 0.85)  # Stable ↔ Workhorse (near-identical)
    S[3, 7] = S[7, 3] = max(S[3, 7], 0.70)  # Catalyst ↔ Complex Driver
    np.fill_diagonal(S, 1.0)
    return S


def is_noise_candidate(
    impact: np.ndarray,
    volatility: np.ndarray,
    nonlinearity: np.ndarray,
    interaction: np.ndarray,
) -> np.ndarray:
    """Bool mask: which features are Noise Candidates (rule-of-thumb).

    Independent of `classify()` so Co-Sensitivity can label its clusters
    even when it hasn't called the full classifier.
    """
    n = len(impact)
    if n == 0:
        return np.array([], dtype=bool)
    return (
        (stats.rankdata(impact) / n < 0.3)
        & (stats.rankdata(volatility) / n < 0.3)
        & (stats.rankdata(nonlinearity) / n < 0.3)
        & (stats.rankdata(interaction) / n < 0.3)
    )
