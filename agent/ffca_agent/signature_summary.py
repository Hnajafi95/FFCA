"""Rule-free observation channel for the narrator.

The rulebook captures named pathologies (data leakage, spurious correlation,
shortcut learning, ...). But some patterns in the raw 4D signatures or the
per-checkpoint curves may not match any rule. This module gives the narrator
a bounded, structured summary of the signatures so it can surface those
patterns explicitly — distinct from rule-backed findings.

The output is intentionally summary-statistics, not raw per-feature data:
  - top-K features by each of the 4 dimensions
  - curve-shape descriptors per dimension (monotonicity, slope sign, peak ckpt)
  - cross-checkpoint variance percentiles
  - top interaction pairs (when the interaction matrix is recoverable)

The narrator is instructed to:
  1. Cite specific summary values when making an observation.
  2. Mark any claim from this channel as a "rule-free observation",
     separate from rule-backed findings.
  3. Skip the observation entirely if it duplicates a rule that already fired.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .report import ReportContext


@dataclass
class CurveShape:
    """Monotonicity + slope + peak descriptors for a per-checkpoint curve."""
    name: str
    n_checkpoints: int
    initial: float
    final: float
    peak: float
    peak_checkpoint: int
    fold_change_final_over_initial: float
    monotonic_increasing: bool
    monotonic_decreasing: bool
    has_spike: bool          # peak > 1.5x median of rest
    has_late_drift: bool     # |final - prev| / max(|prev|, eps) > 0.20


def _curve_shape(name: str, curve: np.ndarray) -> CurveShape:
    arr = np.asarray(curve, dtype=float)
    n = int(arr.shape[0])
    initial = float(arr[0])
    final = float(arr[-1])
    peak_idx = int(np.argmax(np.abs(arr)))
    peak = float(arr[peak_idx])
    eps = 1e-12
    fold = float(final / max(abs(initial), eps)) if initial != 0 else float("inf")

    diffs = np.diff(arr)
    mono_inc = bool(np.all(diffs >= -1e-9))
    mono_dec = bool(np.all(diffs <= 1e-9))

    # Spike: peak >= 1.5x median of remaining entries AND peak is not at the
    # boundary (i.e., a transient bump mid-run, not a monotonic ramp).
    others = np.delete(arr, peak_idx)
    peak_is_interior = 0 < peak_idx < (n - 1)
    spike = bool(peak_is_interior and others.size
                 and abs(peak) >= 1.5 * float(np.median(np.abs(others)) + eps))

    # Late drift: relative move between last 2 ckpts >20%
    drift = False
    if n >= 2:
        denom = max(abs(float(arr[-2])), eps)
        drift = abs(final - float(arr[-2])) / denom > 0.20

    return CurveShape(
        name=name,
        n_checkpoints=n,
        initial=initial,
        final=final,
        peak=peak,
        peak_checkpoint=peak_idx,
        fold_change_final_over_initial=fold,
        monotonic_increasing=mono_inc,
        monotonic_decreasing=mono_dec,
        has_spike=spike,
        has_late_drift=drift,
    )


@dataclass
class TopKEntry:
    feature: str
    value: float
    archetype: str
    trust: str


@dataclass
class InteractionPair:
    feature_a: str
    feature_b: str
    strength: float  # |∂²f/∂x_a ∂x_b| averaged


@dataclass
class SignatureSummary:
    """Bundle of bounded, structured signature summaries for the narrator."""
    top_by_impact: list[TopKEntry] = field(default_factory=list)
    top_by_volatility: list[TopKEntry] = field(default_factory=list)
    top_by_nonlinearity: list[TopKEntry] = field(default_factory=list)
    top_by_interaction: list[TopKEntry] = field(default_factory=list)

    impact_topk_curve_shape: CurveShape | None = None
    volatility_topk_curve_shape: CurveShape | None = None
    interaction_topk_curve_shape: CurveShape | None = None

    top_interaction_pairs: list[InteractionPair] = field(default_factory=list)

    # Per-feature variance summary across checkpoints
    impact_cross_checkpoint_cov_p95: float = 0.0
    n_features_with_archetype_churn_ge_3: int = 0

    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Drop None curve shapes for cleaner JSON
        for k in ["impact_topk_curve_shape", "volatility_topk_curve_shape",
                  "interaction_topk_curve_shape"]:
            if d.get(k) is None:
                del d[k]
        return d


def signature_summary(ctx: ReportContext, top_k: int = 5) -> SignatureSummary:
    """Build a structured summary of the FFCA signatures.

    Bounded by design:
      - Returns at most `top_k` features per dimension.
      - Returns at most `top_k` interaction pairs.
      - Returns scalar curve-shape descriptors, not full curves.
    """
    n = ctx.n_features
    k = min(top_k, n)
    summary = SignatureSummary()

    # Helper: build the per-dimension top-K with archetype + trust
    def _topk(values: np.ndarray, label: str) -> list[TopKEntry]:
        order = np.argsort(-np.abs(values))[:k]
        out = []
        for i in order:
            out.append(TopKEntry(
                feature=ctx.feature_names[i],
                value=float(values[i]),
                archetype=str(ctx.archetypes[i]),
                trust=ctx.feature_trust.get(ctx.feature_names[i], ""),
            ))
        return out

    summary.top_by_impact        = _topk(ctx.impact,        "impact")
    summary.top_by_volatility    = _topk(ctx.volatility,    "volatility")
    summary.top_by_nonlinearity  = _topk(ctx.nonlinearity,  "nonlinearity")
    summary.top_by_interaction   = _topk(ctx.interaction,   "interaction")

    # Curve shapes (averaged over top-k by final Impact)
    if ctx.impact_curve.shape[0] >= 2:
        order = np.argsort(-ctx.impact)[:k]
        impact_top_curve = ctx.impact_curve[:, order].mean(axis=1)
        vol_top_curve = ctx.volatility_curve[:, order].mean(axis=1)
        inter_top_curve = ctx.interaction_curve[:, order].mean(axis=1)
        summary.impact_topk_curve_shape = _curve_shape("impact_topk_mean", impact_top_curve)
        summary.volatility_topk_curve_shape = _curve_shape("volatility_topk_mean", vol_top_curve)
        summary.interaction_topk_curve_shape = _curve_shape("interaction_topk_mean", inter_top_curve)

    # Cross-checkpoint variability (per-feature CoV in impact_curve), report p95
    if ctx.impact_curve.shape[0] >= 2:
        means = ctx.impact_curve.mean(axis=0)
        stds = ctx.impact_curve.std(axis=0)
        cov = np.where(np.abs(means) > 1e-12, stds / np.abs(means), 0.0)
        summary.impact_cross_checkpoint_cov_p95 = float(np.quantile(cov, 0.95))

    # Archetype churn: count features whose archetype changes >=3 times across checkpoints
    # The raw report's per-checkpoint archetypes are in raw["signatures"][k]["archetypes"]
    sigs = ctx.raw.get("signatures", [])
    if len(sigs) >= 3:
        from .archetypes import PACKAGE_INDEX_TO_PAPER
        churn = np.zeros(n, dtype=int)
        prev = None
        for s in sigs:
            ar = np.asarray(s.get("archetypes", []), dtype=int)
            if ar.shape[0] != n:
                continue
            if prev is not None:
                churn += (ar != prev).astype(int)
            prev = ar
        summary.n_features_with_archetype_churn_ge_3 = int((churn >= 3).sum())

    # Top interaction pairs are only recoverable when the report carries the
    # full interaction matrix per checkpoint. If not, leave the list empty.
    inter_matrix = ctx.raw.get("interaction_matrix")
    if inter_matrix is not None:
        try:
            M = np.abs(np.asarray(inter_matrix, dtype=float))
            if M.ndim == 2 and M.shape == (n, n):
                # Zero out the diagonal
                np.fill_diagonal(M, 0.0)
                # Top-k pairs by magnitude
                idxs = np.dstack(np.unravel_index(np.argsort(-M.ravel()), M.shape))[0]
                seen = set()
                for ii, jj in idxs:
                    if len(summary.top_interaction_pairs) >= k:
                        break
                    a, b = sorted((int(ii), int(jj)))
                    if a == b or (a, b) in seen:
                        continue
                    seen.add((a, b))
                    summary.top_interaction_pairs.append(InteractionPair(
                        feature_a=ctx.feature_names[a],
                        feature_b=ctx.feature_names[b],
                        strength=float(M[a, b]),
                    ))
        except Exception:
            summary.notes.append("interaction_matrix present but could not parse")

    return summary
