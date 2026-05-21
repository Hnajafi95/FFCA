"""ffca.diagnostics — FFCA model-health rules.

A FFCAReport produces a 4-D signature per checkpoint plus Trust / Co-Sensitivity
optionally. The package can derive higher-level diagnostic findings from
those quantities:

  • Overfitting       — volatility grows across checkpoints
  • Shortcut learning — for image inputs, interaction concentrates in the
                        background ring (low Foreground/Background ratio);
                        for tabular, a single feature dominates with low
                        non-linearity (suspect spurious correlation)
  • Data leakage      — anomalously high Impact paired with tiny
                        non-linearity AND tiny volatility (the feature
                        carries the label too cleanly)
  • Saturation        — many features stuck near zero impact across
                        every checkpoint (under-capacity / dead features)
  • Capacity health   — archetype distribution dominated by Complex Drivers
                        (too much going on) or Noise Candidates (not enough
                        learning happened)

Each rule emits a Finding with: severity (info / warn / critical),
short headline, the observation that triggered it, why it matters, and a
suggested next action.

Findings are appended to FFCAReport.findings and rendered as a Diagnostics
section in report.md.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Sequence

import numpy as np

from .core.archetypes import ARCHETYPE_NAMES
from .core.signature import FFCASignature


# ---------------------------------------------------------------- Finding
@dataclass
class Finding:
    """One diagnostic insight."""
    name: str
    severity: str           # "info" | "warn" | "critical"
    headline: str           # one-line summary
    observation: str        # what the data showed
    why_it_matters: str     # plain-English explanation
    recommendation: str     # what the user should do next
    evidence: dict = field(default_factory=dict)

    @property
    def icon(self) -> str:
        return {"info": "ℹ", "warn": "⚠", "critical": "✗"}.get(self.severity, "•")

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- detectors
def detect_overfitting(
    signatures: Sequence[FFCASignature],
    checkpoint_labels: Sequence[str],
    volatility_growth_threshold: float = 2.0,
) -> list[Finding]:
    """Compare mean volatility at the last checkpoint vs the median across
    earlier checkpoints. A ≥2× increase suggests the gradient is becoming
    more context-dependent — classic over-fit signature in the FFCA paper."""
    if len(signatures) < 3:
        return []
    means = np.array([float(s.volatility.mean()) for s in signatures])
    last = means[-1]
    earlier = float(np.median(means[:-1]))
    ratio = last / max(earlier, 1e-12)

    if ratio < volatility_growth_threshold:
        return [Finding(
            name="overfitting",
            severity="info",
            headline=f"No volatility spike detected across {len(signatures)} checkpoints",
            observation=f"Final-checkpoint mean Volatility = {last:.4g} ; "
                        f"median of earlier checkpoints = {earlier:.4g} "
                        f"(ratio = {ratio:.2f}×).",
            why_it_matters="Volatility = Var(∂f/∂x_i) measures how context-dependent "
                            "each feature's effect is. A late-training jump usually "
                            "means the model is memorising sample-specific quirks "
                            "rather than learning a stable rule.",
            recommendation="Within healthy training; no action required.",
            evidence={"final_volatility_mean": last, "earlier_median": earlier,
                       "ratio": ratio},
        )]
    severity = "warn" if ratio < 4.0 else "critical"
    return [Finding(
        name="overfitting",
        severity=severity,
        headline=f"Volatility grew {ratio:.1f}× by the final checkpoint — possible overfitting",
        observation=f"Mean Volatility climbed from {earlier:.4g} (median of "
                    f"epochs {list(checkpoint_labels[:-1])}) to {last:.4g} at "
                    f"'{checkpoint_labels[-1]}'.",
        why_it_matters="A late-training jump in Volatility means feature effects "
                        "are now strongly sample-dependent — the model is memorising "
                        "individual examples rather than generalising.",
        recommendation="Consider early stopping at an earlier checkpoint, adding "
                        "regularisation, or evaluating held-out generalisation.",
        evidence={"final_volatility_mean": last, "earlier_median": earlier,
                   "ratio": ratio},
    )]


def detect_shortcut_learning(
    signature: FFCASignature,
    *,
    feature_shape: tuple[int, ...] | None = None,
    fg_frac: float = 0.5,
) -> list[Finding]:
    """For pixel-shaped inputs, compute Foreground / Background ratio of
    interaction. FBR < 0.5 hints the model concentrates explanation on the
    image periphery — a Waterbirds-style shortcut signature."""
    if feature_shape is None:
        meta_shape = tuple(signature.metadata.get("feature_shape", ())) \
            if signature.metadata else ()
        feature_shape = meta_shape
    if len(feature_shape) != 3:
        return []  # not an image-shaped input
    C, H, W = feature_shape
    img = signature.interaction.reshape(C, H, W)
    py = int(H * (1 - fg_frac) / 2); px = int(W * (1 - fg_frac) / 2)
    fg = float(img[:, py:H - py, px:W - px].mean())
    mask = np.ones_like(img, dtype=bool)
    mask[:, py:H - py, px:W - px] = False
    bg = float(img[mask].mean())
    fbr = fg / (fg + bg) if (fg + bg) > 0 else float("nan")

    if np.isnan(fbr):
        return []
    if fbr < 0.35:
        severity = "critical"; verdict = "STRONG shortcut signal"
    elif fbr < 0.50:
        severity = "warn"; verdict = "Moderate shortcut risk"
    else:
        severity = "info"; verdict = "No background-shortcut signal"
    return [Finding(
        name="shortcut_learning",
        severity=severity,
        headline=f"{verdict} — Foreground/Background interaction ratio = {fbr:.2f}",
        observation=f"Mean interaction in centre {int(fg_frac*100)}% of the image: "
                    f"{fg:.4f} ; in the surrounding ring: {bg:.4f} ; FBR={fbr:.3f}.",
        why_it_matters="A model that relies on background pixels (FBR < 0.5) is "
                        "using context cues that won't generalise — the Waterbirds "
                        "failure mode. The model should put more interaction on "
                        "the central subject.",
        recommendation=(
            "Inspect the pixel interaction heatmap; if hotspots are in the "
            "border, retrain with data augmentation that breaks the spurious "
            "correlation, or use group-balanced sampling."
        ) if severity != "info" else "No remediation needed.",
        evidence={"fg_mean": fg, "bg_mean": bg, "fbr": fbr, "fg_frac": fg_frac},
    )]


def detect_tabular_shortcut(
    signature: FFCASignature,
    *,
    dominance_threshold: float = 0.50,
    nonlinearity_threshold: float = 0.10,
) -> list[Finding]:
    """For tabular models, a single feature responsible for >50% of total
    Impact with very low Non-linearity is a candidate for being a leaked
    or spurious-correlation feature (e.g., an ID column)."""
    fshape = tuple(signature.metadata.get("feature_shape", ())) \
        if signature.metadata else ()
    if len(fshape) != 1 or signature.n_features < 4:
        return []
    if signature.n_features > 200:
        return []  # only meaningful at small d (image-channel cases aren't tabular)
    impact_share = signature.impact / max(signature.impact.sum(), 1e-12)
    top_idx = int(np.argmax(impact_share))
    top_share = float(impact_share[top_idx])
    top_nlin_pct = float(
        (signature.nonlinearity < signature.nonlinearity[top_idx]).mean()
    )

    if top_share < dominance_threshold:
        return []
    headline = (f"Single feature '{signature.feature_names[top_idx]}' carries "
                f"{top_share:.0%} of total Impact")
    if top_nlin_pct < nonlinearity_threshold:
        severity = "critical"
        why = ("That feature is highly important AND linear in its effect — "
                "the classic data-leakage signature: an ID-like column that "
                "encodes the label too directly.")
        rec = ("Inspect this feature for label leakage (post-hoc identifiers, "
                "computed-from-target columns, near-perfect proxies).")
    else:
        severity = "warn"
        why = ("Heavy reliance on one feature can be brittle; if the feature's "
                "distribution shifts at deployment the model will fail.")
        rec = ("Verify the feature is genuinely available at prediction time "
                "and stable; consider domain checks and ablation.")
    return [Finding(
        name="tabular_shortcut",
        severity=severity,
        headline=headline,
        observation=f"feature '{signature.feature_names[top_idx]}': "
                    f"Impact share = {top_share:.1%}, "
                    f"Non-linearity rank percentile = {top_nlin_pct:.1%}.",
        why_it_matters=why,
        recommendation=rec,
        evidence={"feature": signature.feature_names[top_idx],
                   "impact_share": top_share,
                   "nonlinearity_rank": top_nlin_pct},
    )]


def detect_data_leakage(
    signature: FFCASignature,
    *,
    z_threshold: float = 3.0,
) -> list[Finding]:
    """A feature whose Impact z-score is ≥3 and whose Non-linearity AND
    Interaction z-scores are both far below average is suspect: too
    important, too smoothly so."""
    if signature.n_features < 8:
        return []
    z_imp = _zscore(signature.impact)
    z_nlin = _zscore(signature.nonlinearity)
    z_int = _zscore(signature.interaction)

    mask = (z_imp > z_threshold) & (z_nlin < -0.5) & (z_int < -0.5)
    suspect_idx = np.where(mask)[0].tolist()
    if not suspect_idx:
        return []
    names = [signature.feature_names[i] for i in suspect_idx[:5]]
    return [Finding(
        name="data_leakage",
        severity="warn",
        headline=f"{len(suspect_idx)} feature(s) carry unusually high Impact "
                  f"with low Non-linearity + Interaction — possible leakage",
        observation=f"Suspect features: {names}.  These have Impact z-score > "
                    f"{z_threshold:.1f} but Non-linearity and Interaction "
                    f"z-scores both ≤ −0.5 — meaning they dominate the output "
                    f"through a purely linear path.",
        why_it_matters="Leaked features (label-derived, near-target proxies) "
                        "typically have very high Impact AND very low curvature "
                        "(the model just memorises the linear shortcut). "
                        "Genuine drivers usually accumulate some non-linearity "
                        "or interactions during training.",
        recommendation="Audit these features for post-hoc derivation from the "
                        "target. Run an ablation: drop the feature, re-train, "
                        "and verify the model still meets its metric goals.",
        evidence={"suspect_feature_indices": suspect_idx,
                   "suspect_feature_names": names,
                   "z_threshold": z_threshold},
    )]


def detect_saturation(
    signature: FFCASignature,
    *,
    dead_threshold_pct: float = 0.10,
) -> list[Finding]:
    """Fraction of features with Impact < 10% of the median."""
    if signature.n_features < 8:
        return []
    med = float(np.median(signature.impact))
    if med <= 0:
        return []
    dead_mask = signature.impact < dead_threshold_pct * med
    dead_frac = float(dead_mask.mean())
    if dead_frac < 0.30:
        return []
    severity = "critical" if dead_frac > 0.60 else "warn"
    return [Finding(
        name="saturation",
        severity=severity,
        headline=f"{dead_frac:.0%} of features have near-zero Impact — "
                  f"feature space is under-utilised",
        observation=f"{int(dead_mask.sum())} of {signature.n_features} features "
                    f"have Impact < {dead_threshold_pct:.0%} of the median.",
        why_it_matters="If a large fraction of features have nearly zero "
                        "Impact, the model is either using only a thin "
                        "sub-space (over-parameterised) or many features are "
                        "genuinely uninformative (collect different signal).",
        recommendation=("Consider pruning these features (saves inference cost) "
                          "or, for over-parameterised models, increasing weight "
                          "decay / using a smaller architecture."),
        evidence={"dead_count": int(dead_mask.sum()),
                   "dead_fraction": dead_frac,
                   "median_impact": med},
    )]


def detect_capacity(
    signature: FFCASignature,
    *,
    mode: str = "trajectory",
) -> list[Finding]:
    """Archetype distribution as a model-health summary.

    The `mode` parameter mirrors FFCAReport.mode: in 'ensemble' mode
    'train longer' is removed from the recommendation because the
    checkpoints are independent seeds, not a single training run that
    can be extended.
    """
    if signature.archetypes is None or signature.n_features < 8:
        return []
    counts = np.bincount(signature.archetypes, minlength=8)
    n = counts.sum()
    noise_frac = counts[0] / n
    complex_frac = counts[7] / n
    catalyst_frac = counts[3] / n

    if noise_frac > 0.50:
        if mode == "ensemble":
            recommendation = (
                "These features are noise across every seed — safe to prune. "
                "Do not interpret as 'training incomplete'; the checkpoints "
                "are independent seeds, not a single training run."
            )
        else:
            recommendation = (
                "Train longer, or prune the noise features to lighten inference."
            )
        return [Finding(
            name="capacity",
            severity="warn",
            headline=f"{noise_frac:.0%} of features are Noise Candidates — model under-uses its inputs",
            observation=f"Counts: Noise={counts[0]} ({noise_frac:.0%}), "
                        f"Catalyst={counts[3]} ({catalyst_frac:.0%}), "
                        f"Complex Driver={counts[7]} ({complex_frac:.0%}).",
            why_it_matters="More than half of features contribute nothing — "
                            "either they are genuinely irrelevant, or the model "
                            "has not learned to use them yet.",
            recommendation=recommendation,
            evidence={"archetype_counts": counts.tolist()},
        )]
    if complex_frac > 0.50:
        return [Finding(
            name="capacity",
            severity="info",
            headline=f"{complex_frac:.0%} of features are Complex Drivers — "
                      f"model is using most features in interacting, non-linear ways",
            observation=f"Counts: Complex Driver={counts[7]} ({complex_frac:.0%}), "
                        f"Catalyst={counts[3]} ({catalyst_frac:.0%}), "
                        f"Noise={counts[0]} ({noise_frac:.0%}).",
            why_it_matters="High Complex Driver share is normal for well-trained "
                            "modern networks, but it also means the model is hard "
                            "to interpret with simple feature-attribution tools.",
            recommendation="No action; use FFCA's interaction heatmaps when "
                            "explaining individual predictions.",
            evidence={"archetype_counts": counts.tolist()},
        )]
    return [Finding(
        name="capacity",
        severity="info",
        headline=f"Healthy archetype distribution",
        observation=f"Noise {noise_frac:.0%}, Catalyst {catalyst_frac:.0%}, "
                    f"Complex {complex_frac:.0%}.",
        why_it_matters="No archetype dominates — the model has a diverse mix "
                        "of feature roles.",
        recommendation="None.",
        evidence={"archetype_counts": counts.tolist()},
    )]


def detect_co_sensitivity_verdict(cosens) -> list[Finding]:
    """Explain what the Co-Sensitivity run concluded."""
    if cosens is None or not getattr(cosens, "diagnostics", None) or \
            cosens.diagnostics.get("k") is None:
        return []
    d = cosens.diagnostics
    k = d["k"]; sil = d["silhouette_observed"]
    pval = d.get("permutation_p", float("nan"))
    ari = d.get("bootstrap_ari_median", float("nan"))
    best_nc = d.get("best_nc_fraction", 0.0)
    abort = d.get("abort_recommended", True)

    reasons = []
    if best_nc < 0.5:
        reasons.append(f"no group has >50% Noise Candidates (best={best_nc:.0%})")
    if not np.isnan(pval) and pval >= 0.05:
        reasons.append(f"clusters indistinguishable from random shuffle (p={pval:.3f})")
    if not np.isnan(ari) and ari < 0.5:
        reasons.append(f"bootstrap stability low (ARI={ari:.2f}<0.5)")

    if abort:
        return [Finding(
            name="co_sensitivity",
            severity="info",
            headline=(f"Co-Sensitivity refused to recommend any prune "
                       f"({k} groups found, but no prune-safe group)"),
            observation=(f"k={k} functional groups, silhouette={sil:.2f}, "
                          f"permutation p={pval:.3f}, bootstrap ARI={ari:.2f}, "
                          f"best group NC fraction={best_nc:.0%}.  "
                          f"Abort triggered because: " + " ; ".join(reasons) + "."),
            why_it_matters=("Co-Sensitivity clusters features by gradient "
                             "similarity. It only recommends pruning when a "
                             "whole group is noise-dominated (>50%) AND the "
                             "clustering itself is statistically supported. "
                             "Refusing to prune is the SAFE outcome — the "
                             "alternative is removing useful features."),
            recommendation="Pruning is not warranted from this run. If you need "
                            "to compress the model, use magnitude-based or "
                            "movement pruning instead.",
            evidence=d,
        )]
    return [Finding(
        name="co_sensitivity",
        severity="info",
        headline=(f"Co-Sensitivity identified a prune-safe group "
                   f"(k={k}, best NC={best_nc:.0%})"),
        observation=(f"k={k}, silhouette={sil:.2f}, p={pval:.3f}, "
                      f"ARI={ari:.2f}, best-NC={best_nc:.0%}."),
        why_it_matters=("A group of mutually-redundant Noise Candidates exists. "
                         "These can be pruned together without losing model "
                         "capability (in principle)."),
        recommendation="Run an ablation: drop the prune group, re-evaluate "
                        "accuracy. Only then commit to the pruning.",
        evidence=d,
    )]


def detect_trust_verdict(trust, mode: str = "trajectory") -> list[Finding]:
    """Plain-English summary of the Trust Score decisions.

    In trajectory mode the high-INVESTIGATE finding's diagnosis is framed
    around training time ("model has not settled"). In ensemble mode the
    "INVESTIGATE (multi-modal seeds)" bucket has a different meaning and the
    finding text is reworded accordingly — see project_rulebook_bug_seed_vs_epoch
    in /Users/hnaja002/.claude/.../memory/ for context.
    """
    if trust is None or not trust.results:
        return []
    summary = trust.summary()
    n = sum(summary.values())
    findings = []
    p_count = summary.get("CONFIDENTLY PRUNE", 0)
    k_count = summary.get("CONFIDENTLY KEEP", 0) + summary.get("KEEP (stable)", 0)
    if mode == "ensemble":
        i_count = summary.get("INVESTIGATE (multi-modal seeds)", 0)
    else:
        i_count = summary.get("INVESTIGATE (unstable)", 0)
    if i_count / max(n, 1) > 0.5:
        if mode == "ensemble":
            findings.append(Finding(
                name="trust_multi_modal_seeds",
                severity="warn",
                headline=f"{i_count}/{n} features behave differently across random seeds",
                observation=f"{i_count/n:.0%} of features had a different "
                            f"archetype on a minority of the seeds (modal-"
                            f"archetype agreement < 0.4) while still carrying "
                            f"non-trivial mean importance.",
                why_it_matters="High cross-seed disagreement is the FFCA "
                                "'ensemble in disguise' signature: the loss "
                                "landscape is multi-modal and different seeds "
                                "find different feature-role assignments that "
                                "happen to be roughly equivalent in accuracy. "
                                "This is NOT the same as 'model still training' — "
                                "more epochs will not resolve it.",
                recommendation="Treat the ensemble as load-bearing — do not "
                                "prune by INVESTIGATE-rate alone, and do not "
                                "interpret high INVESTIGATE as evidence of "
                                "incomplete training. If model RMSE is "
                                "satisfactory, accept the ensemble. If RMSE is "
                                "not satisfactory, the right lever is "
                                "architecture / feature engineering, not more "
                                "training epochs.",
                evidence={"investigate_count": i_count, "total": n,
                           "summary": summary,
                           "axis": "seed"},
            ))
        else:
            findings.append(Finding(
                name="trust_instability",
                severity="warn",
                headline=f"{i_count}/{n} features are unstable across checkpoints",
                observation=f"More than half ({i_count/n:.0%}) of features changed "
                            f"archetype between checkpoints (similarity-weighted "
                            f"stability < 0.5).",
                why_it_matters="High INVESTIGATE rate suggests the model has not "
                                "settled into stable feature roles — either it is "
                                "still training, or the data is noisy enough that "
                                "different epochs use the features differently.",
                recommendation="Train for more epochs, or use a learning-rate "
                                "schedule that converges sooner. If accuracy is "
                                "good but features are unstable, the model is "
                                "an ensemble in disguise.",
                evidence={"investigate_count": i_count, "total": n,
                           "summary": summary,
                           "axis": "epoch"},
            ))
    if p_count > 0:
        findings.append(Finding(
            name="trust_prune_recommended",
            severity="info",
            headline=f"{p_count} features are confidently Noise across all checkpoints",
            observation=f"These features were in the Noise archetype at every "
                        f"checkpoint with high stability.",
            why_it_matters="Stable Noise Candidates are the safest pruning "
                            "targets — they are not just unimportant at the "
                            "end, they were never important.",
            recommendation="Prune; expect no accuracy loss.",
            evidence={"prune_count": p_count, "summary": summary},
        ))
    if k_count > 0:
        findings.append(Finding(
            name="trust_keep_recommended",
            severity="info",
            headline=f"{k_count} features are stably important across all checkpoints",
            observation=f"These features retained the same useful archetype "
                        f"(Workhorse / Catalyst / Stable Contributor) at every "
                        f"checkpoint.",
            why_it_matters="High-stability + high-importance features are the "
                            "backbone of the model. Removing or changing them "
                            "will be felt in accuracy.",
            recommendation="Treat these as load-bearing. Protect with extra "
                            "logging in production; do not prune.",
            evidence={"keep_count": k_count, "summary": summary},
        ))
    return findings


# ---------------------------------------------------------------- runner
def run_all(
    signatures: Sequence[FFCASignature],
    checkpoint_labels: Sequence[str],
    trust=None,
    cosens=None,
    mode: str = "trajectory",
) -> list[Finding]:
    """Run every detector and return a flat list of findings, sorted by
    severity (critical → warn → info).

    `mode='ensemble'` skips epoch-axis detectors (drift, overfitting curve)
    that are meaningless on aggregated seed-ensemble signatures, and routes
    trust diagnostics through the seed-axis interpretation.
    """
    findings: list[Finding] = []
    last = signatures[-1]
    findings += detect_capacity(last, mode=mode)
    findings += detect_data_leakage(last)
    findings += detect_tabular_shortcut(last)
    findings += detect_shortcut_learning(last)
    findings += detect_saturation(last)
    if mode == "trajectory" and len(signatures) >= 3:
        # Overfitting detection reads volatility-curve over time; meaningless
        # for aggregated seed signatures.
        findings += detect_overfitting(signatures, checkpoint_labels)
    findings += detect_trust_verdict(trust, mode=mode)
    findings += detect_co_sensitivity_verdict(cosens)
    rank = {"critical": 0, "warn": 1, "info": 2}
    findings.sort(key=lambda f: (rank.get(f.severity, 9), f.name))
    return findings


# ---------------------------------------------------------------- helpers
def _zscore(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    if sd <= 0:
        return np.zeros_like(x)
    return (x - x.mean()) / sd
