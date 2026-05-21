"""Cross-seed aggregation of FFCA signatures.

When the "checkpoints" passed to FFCAReport are actually independent random
seeds of the same training procedure (a seed-ensemble), there is no
time-ordering between them and across-checkpoint drift signals are
meaningless. This module aggregates the per-seed signatures into a single
representative signature (mean of the four axes per feature) and a separate
record of cross-seed variability for diagnostic use.

Used by FFCAReport when mode='ensemble'. The default mode='trajectory'
preserves the original epoch-axis interpretation.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .archetypes import classify, ARCHETYPE_NAMES
from .signature import FFCASignature


def aggregate_signatures(
    signatures: Sequence[FFCASignature],
) -> tuple[FFCASignature, dict]:
    """Build one aggregate signature plus per-feature cross-seed statistics.

    Aggregation is **mean across seeds** for each of the four axes
    (Impact, Volatility, Nonlinearity, Interaction). Archetype classification
    is then run once on the aggregate. Per-feature cross-seed variability is
    returned separately for downstream diagnostics — it is *not* the same
    construct as the trajectory-mode `stability` (which is similarity-weighted
    archetype-flip entropy and presumes a time axis).

    Returns:
        (aggregate_signature, seed_stats)

        seed_stats is a dict with arrays of shape (n_features,):
          - 'impact_std', 'volatility_std', 'nonlinearity_std', 'interaction_std'
          - 'impact_cv', 'volatility_cv', 'nonlinearity_cv', 'interaction_cv'
            (coefficient of variation = std / max(|mean|, eps))
          - 'archetype_modal'        — int, modal archetype across seeds per feature
          - 'archetype_agreement'    — float in [0,1], fraction of seeds matching modal
          - 'n_unique_archetypes'    — int, distinct archetypes seen across seeds
    """
    if not signatures:
        raise ValueError("aggregate_signatures requires at least one signature")
    feat = signatures[0].feature_names
    n = signatures[0].n_features
    for s in signatures:
        if s.n_features != n:
            raise ValueError("Per-seed signatures must share feature dimension")

    # (n_seeds, n_features) for each axis
    I = np.stack([s.impact for s in signatures], axis=0)
    V = np.stack([s.volatility for s in signatures], axis=0)
    N = np.stack([s.nonlinearity for s in signatures], axis=0)
    X = np.stack([s.interaction for s in signatures], axis=0)

    # v0.8 (2026-05-20): median across seeds. Mean was sensitive to a single
    # outlier seed (bad initialisation) — and the paper itself uses median
    # across the ensemble for predictions (feature_perturbation_retraining.py),
    # so the signatures should be aggregated the same way.
    impact_central       = np.median(I, axis=0)
    volatility_central   = np.median(V, axis=0)
    nonlinearity_central = np.median(N, axis=0)
    interaction_central  = np.median(X, axis=0)

    eps = 1e-12
    seed_stats: dict = {
        "impact_std":       I.std(axis=0, ddof=0),
        "volatility_std":   V.std(axis=0, ddof=0),
        "nonlinearity_std": N.std(axis=0, ddof=0),
        "interaction_std":  X.std(axis=0, ddof=0),
        "impact_cv":       I.std(axis=0, ddof=0) / np.maximum(np.abs(impact_central), eps),
        "volatility_cv":   V.std(axis=0, ddof=0) / np.maximum(np.abs(volatility_central), eps),
        "nonlinearity_cv": N.std(axis=0, ddof=0) / np.maximum(np.abs(nonlinearity_central), eps),
        "interaction_cv":  X.std(axis=0, ddof=0) / np.maximum(np.abs(interaction_central), eps),
    }

    # Classify the aggregate signature once
    aggregate_archetypes = classify(
        impact_central, volatility_central, nonlinearity_central, interaction_central,
    )

    # Per-seed archetype matrix (n_seeds, n_features) — modal + agreement
    if all(s.archetypes is not None for s in signatures):
        A = np.stack([np.asarray(s.archetypes) for s in signatures], axis=0)
        n_classes = len(ARCHETYPE_NAMES)
        modal = np.empty(n, dtype=np.int64)
        agreement = np.empty(n, dtype=np.float64)
        unique = np.empty(n, dtype=np.int64)
        for j in range(n):
            counts = np.bincount(A[:, j], minlength=n_classes)
            modal[j] = int(np.argmax(counts))
            agreement[j] = float(counts.max() / A.shape[0])
            unique[j] = int(np.count_nonzero(counts))
        seed_stats["archetype_modal"]       = modal
        seed_stats["archetype_agreement"]   = agreement
        seed_stats["n_unique_archetypes"]   = unique
        seed_stats["archetype_matrix"]      = A  # (n_seeds, n_features), kept for reports

    aggregate = FFCASignature(
        impact=impact_central,
        volatility=volatility_central,
        nonlinearity=nonlinearity_central,
        interaction=interaction_central,
        feature_names=list(feat),
        archetypes=aggregate_archetypes,
        interaction_ci=None,
        metadata={
            "n_features": n,
            "aggregation": "median_across_seeds",
            "n_seeds": len(signatures),
            "ensemble_mode": True,
        },
    )
    return aggregate, seed_stats


def ensemble_trust_decisions(
    aggregate: FFCASignature,
    seed_stats: dict,
    importance_low_pct: float = 5.0,
    importance_high_pct: float = 50.0,
    agreement_high: float = 0.7,
    agreement_low: float = 0.4,
) -> dict[str, dict]:
    """Produce per-feature KEEP/PRUNE/INVESTIGATE-style decisions for ensemble mode.

    Different semantics from trajectory-mode TrustScore — this one is based on
    mean importance (across seeds) plus cross-seed archetype agreement, NOT on
    archetype-flip entropy along a time axis. The decision labels deliberately
    mirror the trajectory-mode labels so downstream code (paper tables, agent
    prompts) can consume the same field names, but the meaning is different:

      - CONFIDENTLY KEEP: high mean importance AND high cross-seed archetype agreement
      - KEEP (stable):    moderate mean importance AND high agreement
      - INVESTIGATE (multi-modal seeds): high importance but seeds disagree on archetype
                                          — the "ensemble in disguise" case
      - MONITOR (borderline): mid importance, mid agreement
      - CONFIDENTLY PRUNE: importance near zero across all seeds

    The "INVESTIGATE" label is reused but its diagnostic meaning is now
    "seeds use this feature differently" — not "training is incomplete."
    Diagnostics that read this dict in ensemble mode should narrate
    accordingly.
    """
    imp = aggregate.impact
    n = imp.size
    if n == 0:
        return {}

    # Use percentile thresholds on importance so that decisions are not tied to
    # absolute magnitude (which varies wildly across tasks).
    imp_lo = np.percentile(imp, importance_low_pct)
    imp_hi = np.percentile(imp, importance_high_pct)

    agreement = seed_stats.get(
        "archetype_agreement",
        np.ones(n, dtype=np.float64),
    )

    decisions: dict[str, dict] = {}
    for j, name in enumerate(aggregate.feature_names):
        I = float(imp[j])
        A = float(agreement[j])
        modal_idx = int(seed_stats["archetype_modal"][j]) if "archetype_modal" in seed_stats else int(aggregate.archetypes[j])
        modal_name = ARCHETYPE_NAMES[modal_idx]

        # ARCHETYPE_NAMES uses the package-internal short name "Noise"; the
        # paper-side name "Noise Candidate" lives in ffca_agent/archetypes.py.
        # Matching against either here lets this branch fire whether the
        # caller has normalised the archetype string or not.
        if I <= imp_lo and A >= agreement_high and modal_name in ("Noise", "Noise Candidate"):
            decision = "CONFIDENTLY PRUNE"
        elif I >= imp_hi and A >= agreement_high:
            decision = "CONFIDENTLY KEEP"
        elif I >= imp_hi and A < agreement_low:
            decision = "INVESTIGATE (multi-modal seeds)"
        elif I >= imp_hi:
            decision = "KEEP (stable)"
        elif A < agreement_low:
            decision = "INVESTIGATE (multi-modal seeds)"
        else:
            decision = "MONITOR (borderline)"

        decisions[name] = {
            "decision": decision,
            "mean_importance": round(I, 6),
            "archetype_agreement": round(A, 3),
            "modal_archetype": modal_name,
            "n_unique_archetypes": int(seed_stats["n_unique_archetypes"][j]) if "n_unique_archetypes" in seed_stats else 1,
            "impact_cv": round(float(seed_stats["impact_cv"][j]), 4),
        }
    return decisions


def ensemble_trust_summary(decisions: dict[str, dict]) -> dict[str, int]:
    """Counts per decision bucket, mirroring TrustScore.summary()."""
    buckets: dict[str, int] = {}
    for v in decisions.values():
        d = v["decision"]
        buckets[d] = buckets.get(d, 0) + 1
    return buckets
