"""Adapter from FFCA report.json → SignalContext that the rulebook expects.

The FFCA Python package emits a `report.json` with these top-level fields:
  - signatures: list of per-checkpoint dicts (impact, volatility, nonlinearity,
                interaction, archetypes [int indices])
  - trust:      feature_name → {decision, ...}
  - trust_summary: decision_name → count
  - findings:   pre-computed findings (we ignore — we recompute via the rulebook)
  - timing, n_features, etc.

Signals not present in the standard report (training.val_score_curve,
vision.fbr_curve, etc.) need an *extended* report — see `ExtendedReport` below.
Rules referencing absent signals are skipped with `missing_signal` notes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .archetypes import PACKAGE_INDEX_TO_PAPER, PAPER_NAMES, PAPER_TO_SNAKE


class MissingSignal(KeyError):
    """Raised when a rule references a signal not present in the report."""


@dataclass
class TrustBucket:
    count: int = 0
    features: list[str] = field(default_factory=list)

    @property
    def fraction(self) -> float:
        return float("nan")  # injected by ReportContext after init

    @property
    def fraction_pct(self) -> float:
        return float("nan")

    @property
    def feature_list(self) -> str:
        return ", ".join(self.features[:10]) + (
            f" (+{len(self.features) - 10} more)" if len(self.features) > 10 else ""
        )


@dataclass
class ReportContext:
    """Model-wide signal namespace for one FFCA report."""

    raw: dict
    feature_names: list[str]
    n_features: int

    # final-checkpoint signatures
    impact: np.ndarray
    volatility: np.ndarray
    nonlinearity: np.ndarray
    interaction: np.ndarray
    archetypes: np.ndarray  # paper names per feature

    # per-feature curves across checkpoints (n_checkpoints × n_features)
    impact_curve: np.ndarray
    volatility_curve: np.ndarray
    interaction_curve: np.ndarray

    # trust score buckets
    trust_buckets: dict[str, TrustBucket]
    feature_trust: dict[str, str]  # feature name → decision

    # co-sensitivity (optional — may be None if not in report)
    cosens: dict | None = None

    # extended fields (provided externally via ExtendedReport.attach)
    training: dict[str, Any] = field(default_factory=dict)
    vision: dict[str, Any] = field(default_factory=dict)

    # ── factory ─────────────────────────────────────────────────────────────
    @classmethod
    def from_json(cls, path: str | Path) -> "ReportContext":
        raw = json.loads(Path(path).read_text())
        sigs = raw.get("signatures", [])
        if not sigs:
            raise ValueError(f"report has no signatures: {path}")
        last = sigs[-1]
        n_features = raw.get("n_features", len(last["impact"]))

        feature_names = raw.get("feature_names")
        if not feature_names:
            feature_names = list(raw.get("trust", {}).keys()) or [
                f"f{i}" for i in range(n_features)
            ]

        impact = np.asarray(last["impact"], dtype=float)
        vol = np.asarray(last["volatility"], dtype=float)
        nl = np.asarray(last["nonlinearity"], dtype=float)
        inter = np.asarray(last["interaction"], dtype=float)
        arch_idx = np.asarray(last.get("archetypes", []), dtype=int)
        archetypes = np.array(
            [PACKAGE_INDEX_TO_PAPER[int(a)] for a in arch_idx], dtype=object
        )

        # checkpoint-wise curves
        impact_curve = np.stack([np.asarray(s["impact"], dtype=float) for s in sigs])
        vol_curve = np.stack([np.asarray(s["volatility"], dtype=float) for s in sigs])
        inter_curve = np.stack([np.asarray(s["interaction"], dtype=float) for s in sigs])

        # trust score buckets
        trust = raw.get("trust", {})
        feature_trust: dict[str, str] = {f: t.get("decision", "") for f, t in trust.items()}
        bucket_keys = {
            "CONFIDENTLY KEEP": "confident_keep",
            "KEEP (stable)": "keep_stable",
            "MONITOR (borderline)": "monitor",
            "INVESTIGATE (unstable)": "investigate",
            "CONFIDENTLY PRUNE": "confident_prune",
        }
        buckets: dict[str, TrustBucket] = {v: TrustBucket() for v in bucket_keys.values()}
        for fname, dec in feature_trust.items():
            for paper_dec, snake in bucket_keys.items():
                if dec.startswith(paper_dec.split(" ")[0]) and paper_dec in dec:
                    buckets[snake].features.append(fname)
                    buckets[snake].count += 1
                    break

        # cosens (look for the most-recent cosens result if present)
        cosens = None
        cosens_raw = raw.get("cosens") or raw.get("co_sensitivity")
        if cosens_raw:
            # `summary` (preferred) and `diagnostics` both carry the scalars.
            src = cosens_raw.get("summary") or cosens_raw.get("diagnostics") or cosens_raw
            best_nc = src.get("best_nc_fraction")
            if best_nc is None:
                groups = cosens_raw.get("groups") or {}
                if isinstance(groups, dict):
                    iterable = groups.values()
                elif isinstance(groups, list):
                    iterable = groups
                else:
                    iterable = []
                best_nc = max(
                    (g.get("nc_fraction", 0.0) for g in iterable if isinstance(g, dict)),
                    default=0.0,
                )
            cosens = {
                "best_nc_fraction": float(best_nc) if best_nc is not None else 0.0,
                "permutation_p": float(src.get("permutation_p", 1.0)),
                "bootstrap_ari": float(
                    src.get("bootstrap_ari")
                    or src.get("bootstrap_ari_median", 0.0)
                ),
                "silhouette": float(
                    src.get("silhouette")
                    or src.get("silhouette_observed", 0.0)
                ),
                "n_groups": int(src.get("k") or src.get("n_groups", 0)),
            }

        return cls(
            raw=raw,
            feature_names=feature_names,
            n_features=n_features,
            impact=impact,
            volatility=vol,
            nonlinearity=nl,
            interaction=inter,
            archetypes=archetypes,
            impact_curve=impact_curve,
            volatility_curve=vol_curve,
            interaction_curve=inter_curve,
            trust_buckets=buckets,
            feature_trust=feature_trust,
            cosens=cosens,
        )

    # ── signal lookup ───────────────────────────────────────────────────────
    def get(self, path: str, feature_idx: int | None = None) -> Any:
        """Resolve a dotted signal path. feature_idx is required for `feature.*`."""
        head, _, rest = path.partition(".")

        if head == "feature":
            if feature_idx is None:
                raise MissingSignal(f"{path} requires per-feature context")
            return self._feature_signal(rest, feature_idx)
        if head == "model":
            return self._model_signal(rest)
        if head == "trust":
            return self._trust_signal(rest)
        if head == "cosens":
            return self._cosens_signal(rest)
        if head == "training":
            return self._dict_signal("training", rest, self.training)
        if head == "vision":
            return self._dict_signal("vision", rest, self.vision)
        raise MissingSignal(f"unknown signal root: {path}")

    def _feature_signal(self, rest: str, i: int) -> Any:
        match rest:
            case "name":          return self.feature_names[i]
            case "impact":        return float(self.impact[i])
            case "volatility":    return float(self.volatility[i])
            case "nonlinearity":  return float(self.nonlinearity[i])
            case "interaction":   return float(self.interaction[i])
            case "archetype":     return str(self.archetypes[i])
            case "impact_curve":  return self.impact_curve[:, i]
            case "volatility_curve":  return self.volatility_curve[:, i]
            case "interaction_curve": return self.interaction_curve[:, i]
            case "impact_epoch0": return float(self.impact_curve[0, i])
            case "trust_decision":
                return self.feature_trust.get(self.feature_names[i], "")
            case "nonlinearity_ratio":
                base = float(self.impact[i]) or 1e-12
                return float(self.nonlinearity[i]) / base
            case "impact_ratio_epoch0":
                rest_curve = self.impact_curve[1:, i]
                if rest_curve.size == 0:
                    return float("nan")
                rest_median = float(np.median(rest_curve)) or 1e-12
                return float(self.impact_curve[0, i]) / rest_median
            case "volatility_rank":
                # 1 = highest, n = lowest
                order = np.argsort(-self.volatility)
                return int(np.where(order == i)[0][0]) + 1
            case "impact_dominance":
                # feature.impact / model.impact_mean. >1 = feature carries above-average load.
                mean = float(self.impact.mean()) or 1e-12
                return float(self.impact[i]) / mean
            case "impact_saturation":
                # impact_curve[0] / impact at the final checkpoint. Closer to 1 = the
                # feature reached its final importance immediately (suggests no learning
                # needed, characteristic of leakage).
                final = float(self.impact_curve[-1, i]) or 1e-12
                return float(self.impact_curve[0, i]) / final
        raise MissingSignal(f"feature.{rest}")

    def _model_signal(self, rest: str) -> Any:
        match rest:
            case "n_features":      return self.n_features
            case "impact_mean":     return float(self.impact.mean())
            case "impact_max":      return float(self.impact.max())
            case "impact_min":      return float(self.impact.min())
            case "impact_p95":      return float(np.quantile(self.impact, 0.95))
            case "impact_cov":
                m = float(self.impact.mean()) or 1e-12
                return float(self.impact.std() / m)
            case "impact_concentration_top20_pct":
                return self._impact_concentration(0.20)
            case "impact_concentration_top5_pct":
                return self._impact_concentration(0.05)
            case "interaction_mean": return float(self.interaction.mean())
            case "nonlinearity_mean": return float(self.nonlinearity.mean())
            case "volatility_mean":  return float(self.volatility.mean())
            case "top_nonlinear_drivers":
                return self._top_features_in_archetype("Non-linear Driver", "nonlinearity", k=5)
            case "top_hidden_interactors":
                return self._top_features_in_archetype("Hidden Interactor", "interaction", k=5)
            case "checkpoint_drift_l2_pct":
                return self._checkpoint_drift_pct()
            case "n_checkpoints":
                return int(self.impact_curve.shape[0])
            case "interaction_to_impact_growth_ratio":
                # Captures hierarchical learning without requiring a "spike then plateau"
                # shape: just compares how much top-k Interaction grew vs how much top-k
                # Impact grew across the run. >2 = interactions developed twice as fast,
                # i.e. the model learned linear pieces first then composed them.
                return self._interaction_to_impact_growth_ratio()
        if rest.startswith("archetype_dist."):
            tail = rest.removeprefix("archetype_dist.")
            if tail.endswith("_pct"):
                snake = tail.removesuffix("_pct")
                return self._archetype_pct(snake)
            if tail.endswith("_count"):
                snake = tail.removesuffix("_count")
                return self._archetype_count(snake)
            if tail.endswith("_features"):
                snake = tail.removesuffix("_features")
                return self._archetype_feature_list(snake)
        raise MissingSignal(f"model.{rest}")

    # ── helpers backing the new model signals ──────────────────────────────
    def _impact_concentration(self, top_fraction: float) -> float:
        """% of total |Impact| carried by the top `top_fraction` of features."""
        if self.n_features == 0:
            return 0.0
        k = max(1, int(round(self.n_features * top_fraction)))
        sorted_desc = np.sort(self.impact)[::-1]
        total = float(self.impact.sum()) or 1e-12
        return float(sorted_desc[:k].sum() / total * 100)

    def _top_features_in_archetype(self, paper_name: str, by: str, k: int = 5) -> str:
        mask = (self.archetypes == paper_name)
        if not mask.any():
            return "none"
        signal = {
            "impact": self.impact, "volatility": self.volatility,
            "nonlinearity": self.nonlinearity, "interaction": self.interaction,
        }[by]
        idxs = np.where(mask)[0]
        ranked = idxs[np.argsort(-signal[idxs])][:k]
        return ", ".join(self.feature_names[j] for j in ranked)

    def _archetype_feature_list(self, snake: str) -> str:
        paper_name = next((p for p, s in PAPER_TO_SNAKE.items() if s == snake), None)
        if paper_name is None:
            raise MissingSignal(f"unknown archetype snake: {snake}")
        names = [self.feature_names[i] for i, a in enumerate(self.archetypes) if a == paper_name]
        if not names:
            return "none"
        # rank by impact descending so the most prominent appear first
        names_sorted = sorted(
            names,
            key=lambda n: -float(self.impact[self.feature_names.index(n)]),
        )
        head = names_sorted[:10]
        tail = f" (+{len(names_sorted) - 10} more)" if len(names_sorted) > 10 else ""
        return ", ".join(head) + tail

    def _interaction_to_impact_growth_ratio(self, top_k: int = 5) -> float:
        """Ratio of relative growth of top-k Interaction vs top-k Impact.
        Both growths measured as final/initial of the top-k feature mean across checkpoints."""
        if self.impact_curve.shape[0] < 2:
            return 1.0
        # pick top-k by final Impact
        order = np.argsort(-self.impact)
        top = order[: min(top_k, self.n_features)]
        impact_top = self.impact_curve[:, top].mean(axis=1)
        inter_top = self.interaction_curve[:, top].mean(axis=1)
        impact_start = float(impact_top[0]) or 1e-12
        inter_start = float(inter_top[0]) or 1e-12
        impact_growth = float(impact_top[-1]) / impact_start
        inter_growth = float(inter_top[-1]) / inter_start
        return inter_growth / max(impact_growth, 1e-12)

    def _checkpoint_drift_pct(self) -> float:
        """L2 distance between last 2 checkpoints' joint signature, normalized
        by overall signature scale, returned as a percentage."""
        if self.impact_curve.shape[0] < 2:
            return 0.0
        last = np.concatenate([
            self.impact_curve[-1], self.volatility_curve[-1], self.interaction_curve[-1],
        ])
        prev = np.concatenate([
            self.impact_curve[-2], self.volatility_curve[-2], self.interaction_curve[-2],
        ])
        scale = float(np.linalg.norm(last)) or 1e-12
        return float(np.linalg.norm(last - prev) / scale * 100)

    def _archetype_pct(self, snake: str) -> float:
        paper_name = next((p for p, s in PAPER_TO_SNAKE.items() if s == snake), None)
        if paper_name is None:
            raise MissingSignal(f"unknown archetype snake: {snake}")
        if self.n_features == 0:
            return 0.0
        return float((self.archetypes == paper_name).sum() / self.n_features * 100)

    def _archetype_count(self, snake: str) -> int:
        paper_name = next((p for p, s in PAPER_TO_SNAKE.items() if s == snake), None)
        if paper_name is None:
            raise MissingSignal(f"unknown archetype snake: {snake}")
        return int((self.archetypes == paper_name).sum())

    def _trust_signal(self, rest: str) -> Any:
        bucket_name, _, field = rest.partition(".")
        if bucket_name not in self.trust_buckets:
            raise MissingSignal(f"trust.{bucket_name}")
        b = self.trust_buckets[bucket_name]
        match field:
            case "count":         return b.count
            case "fraction":      return b.count / self.n_features if self.n_features else 0.0
            case "fraction_pct":  return (b.count / self.n_features * 100) if self.n_features else 0.0
            case "feature_list":  return b.feature_list
        raise MissingSignal(f"trust.{rest}")

    def _cosens_signal(self, rest: str) -> Any:
        if self.cosens is None:
            raise MissingSignal(f"cosens.* not present in report")
        if rest == "best_nc_fraction_pct":
            return self.cosens["best_nc_fraction"] * 100
        if rest in self.cosens:
            return self.cosens[rest]
        raise MissingSignal(f"cosens.{rest}")

    @staticmethod
    def _dict_signal(root: str, rest: str, d: dict) -> Any:
        if rest in d:
            return d[rest]
        raise MissingSignal(f"{root}.{rest}")

    # ── iteration helpers ───────────────────────────────────────────────────
    def feature_indices(self):
        return range(self.n_features)

    def attach_training(self, **fields):
        self.training.update(fields)

    def attach_training_history(self, history) -> None:
        """Merge a TrainingHistory's non-None fields into the training dict."""
        self.training.update(history.as_signal_dict())

    def attach_vision(self, **fields):
        self.vision.update(fields)

    def attach_vision_metrics(self, metrics) -> None:
        """Merge a VisionMetrics object's non-None curves into the vision dict."""
        self.vision.update(metrics.as_signal_dict())


def archetype_pct_signals(ctx: ReportContext) -> dict[str, float]:
    """Convenience: emit all model.archetype_dist.*_pct as a flat dict."""
    return {f"model.archetype_dist.{snake}_pct": ctx._archetype_pct(snake)
            for snake in PAPER_TO_SNAKE.values()}
