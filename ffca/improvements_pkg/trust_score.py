"""TrustScore — similarity-weighted entropy across checkpoints (audit-v2)."""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
from scipy import stats

from ..core.archetypes import ARCHETYPE_NAMES, classify, similarity_matrix
from ..core.signature import FFCASignature


class TrustScore:
    """[Stability, Importance] two-axis trust score per feature.

    Stability = 1 − H_W / H_max where
        H_W = -Σ_i p_i log( Σ_j S_ij p_j )
    and S is the 8×8 archetype similarity matrix.
    """

    def __init__(
        self,
        stability_threshold: float = 0.7,
        unstable_threshold: float = 0.5,
        similarity_scale: float = 1.5,
    ):
        self.stability_threshold = stability_threshold
        self.unstable_threshold = unstable_threshold
        self.similarity_scale = similarity_scale
        self.S = similarity_matrix(scale=similarity_scale)
        self.results: dict = {}

    def _weighted_entropy(self, p: np.ndarray) -> float:
        smoothed = self.S @ p
        smoothed = np.clip(smoothed, 1e-12, 1.0)
        return float(-(p * np.log(smoothed)).sum())

    def _max_weighted_entropy(self, T: int) -> float:
        K = min(8, T)
        chosen = [0]
        remaining = list(range(1, 8))
        while len(chosen) < K and remaining:
            far = max(remaining, key=lambda j: min(1 - self.S[j, c] for c in chosen))
            chosen.append(far)
            remaining.remove(far)
        p = np.zeros(8)
        for c in chosen:
            p[c] = 1.0 / K
        return self._weighted_entropy(p)

    def compute(self, signatures: Sequence[FFCASignature | dict],
                feature_names: Sequence[str] | None = None) -> dict:
        if len(signatures) < 2:
            raise ValueError("TrustScore needs ≥ 2 checkpoints")

        T = len(signatures)
        first = signatures[0]
        if isinstance(first, FFCASignature):
            d = first.n_features
            names = feature_names or first.feature_names
            def _get(sig, key):
                return getattr(sig, key)
        else:
            d = len(first["impact"])
            names = feature_names or [f"feature_{i}" for i in range(d)]
            def _get(sig, key):
                return np.asarray(sig.get(key, np.zeros(d)))

        epoch_archs = np.zeros((T, d), dtype=int)
        epoch_impacts = np.zeros((T, d))
        for t, sig in enumerate(signatures):
            imp = np.asarray(_get(sig, "impact"))
            vol = np.asarray(_get(sig, "volatility"))
            nlin = np.asarray(_get(sig, "nonlinearity"))
            inter = np.asarray(_get(sig, "interaction"))
            epoch_archs[t] = classify(imp, vol, nlin, inter)
            epoch_impacts[t] = imp

        H_max = self._max_weighted_entropy(T)
        trust = {}
        for i, name in enumerate(names):
            archs = epoch_archs[:, i]
            counts = np.bincount(archs, minlength=8).astype(float)
            p = counts / counts.sum()
            H_w = self._weighted_entropy(p)
            stability = float(np.clip(1.0 - (H_w / H_max if H_max > 0 else 0), 0, 1))
            importance = float(epoch_impacts[:, i].mean())
            dominant = int(np.argmax(counts))

            if stability >= self.stability_threshold:
                if dominant == 0:
                    decision = "CONFIDENTLY PRUNE"
                elif dominant in (2, 3, 6):
                    decision = "CONFIDENTLY KEEP"
                else:
                    decision = "KEEP (stable)"
            elif stability < self.unstable_threshold:
                decision = "INVESTIGATE (unstable)"
            else:
                decision = "MONITOR (borderline)"

            trust[name] = {
                "stability": round(stability, 3),
                "importance": round(importance, 6),
                "weighted_entropy": round(H_w, 4),
                "plain_entropy": round(float(stats.entropy(p[p > 0])), 4),
                "dominant_archetype": ARCHETYPE_NAMES[dominant],
                "dominant_fraction": round(float(counts[dominant] / counts.sum()), 3),
                "n_unique_archetypes": int((counts > 0).sum()),
                "decision": decision,
                "archetype_sequence": [ARCHETYPE_NAMES[a] for a in archs],
            }
        self.results = trust
        return trust

    def summary(self) -> dict:
        decs = defaultdict(int)
        for v in self.results.values():
            decs[v["decision"]] += 1
        return dict(decs)
