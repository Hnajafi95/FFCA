"""FFCAReport — the adapter-driven orchestrator.

End-to-end:
  1. For each checkpoint (or just the single current model):
     - smoothing context → compute_signature_core(adapter, loader)
     - CauchyHVP for interaction
     - classify archetypes
     - build a FFCASignature
  2. Across checkpoints:
     - TrustScore (weighted entropy)
     - CoSensitivityGroups on the *last* checkpoint's gradients
  3. Save:
     - report.md   (human-readable summary)
     - report.json (machine-readable, full numbers)
     - plots/      (PNG figures from ffca.viz.generate_all_plots)
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from . import diagnostics as _diag
from .checkpoint import CheckpointLoader
from .core.adapter import FFCAModelAdapter
from .core.archetypes import ARCHETYPE_NAMES, classify
from .core.derivatives import compute_signature_core
from .core.signature import FFCASignature
from .improvements_pkg import CauchyHVP, CoSensitivityGroups, TrustScore


class FFCAReport:
    def __init__(
        self,
        adapter: FFCAModelAdapter,
        loader: Iterable,
        *,
        beta: float = 10.0,
        n_first_order_samples: int = 64,
        n_hessian_samples: int = 16,
        n_diag_probes: int = 48,
        n_cauchy_probes: int = 80,
        n_cauchy_samples: int = 16,
        n_cosens_permutations: int = 50,
        n_cosens_bootstrap: int = 20,
        improvements: bool | dict = True,
    ):
        self.adapter = adapter
        self.loader = loader
        self.beta = beta
        self.n_first_order_samples = n_first_order_samples
        self.n_hessian_samples = n_hessian_samples
        self.n_diag_probes = n_diag_probes
        self.n_cauchy_probes = n_cauchy_probes
        self.n_cauchy_samples = n_cauchy_samples
        self.n_cosens_permutations = n_cosens_permutations
        self.n_cosens_bootstrap = n_cosens_bootstrap

        # improvements gate: True = all three audit-v2 algorithms;
        # False = baseline FFCA paper (correlation-proxy interaction, no Trust, no Co-Sens);
        # dict = granular toggle of {cauchy_hvp, trust_score, co_sensitivity}.
        if isinstance(improvements, bool):
            self.use_cauchy = improvements
            self.use_trust = improvements
            self.use_cosens = improvements
        else:
            self.use_cauchy = improvements.get("cauchy_hvp", True)
            self.use_trust = improvements.get("trust_score", True)
            self.use_cosens = improvements.get("co_sensitivity", True)

        # populated by .run()
        self.signatures: list[FFCASignature] = []
        self.checkpoint_labels: list[str] = []
        self.trust: TrustScore | None = None
        self.cosens: CoSensitivityGroups | None = None
        self.last_gradients: np.ndarray | None = None
        self.timing: dict[str, float] = {}
        self.findings: list[_diag.Finding] = []

    # --------------------------------------------------------------- pipeline
    def _signature_one(self, adapter: FFCAModelAdapter) -> tuple[FFCASignature, np.ndarray]:
        """Compute the full 4-D signature for the current model state."""
        impact, volatility, nonlinearity, gradients = compute_signature_core(
            adapter, self.loader,
            n_first_order_samples=self.n_first_order_samples,
            n_hessian_samples=self.n_hessian_samples,
            n_diag_probes=self.n_diag_probes,
            beta=self.beta,
        )

        from .core.smoothing import smooth
        if self.use_cauchy:
            with smooth(adapter.model, beta=self.beta):
                cauchy = CauchyHVP(n_probes=self.n_cauchy_probes)
                interaction, ci = cauchy.estimate(adapter, self.loader,
                                                  n_samples=self.n_cauchy_samples)
            interaction_method = "cauchy_hvp"
            method_meta = {k: (v.tolist() if hasattr(v, 'tolist') else v)
                           for k, v in cauchy.results.items()
                           if k not in ('interactions', 'ci_lower', 'ci_upper')}
        else:
            # Baseline FFCA paper: gradient-correlation row-sum, no Hessian off-diag.
            corr = np.corrcoef(gradients.T) if gradients.shape[0] > 1 else np.zeros((gradients.shape[1],) * 2)
            corr = np.nan_to_num(corr, nan=0.0)
            np.fill_diagonal(corr, 0.0)
            interaction = np.abs(corr).sum(axis=1)
            ci = None
            interaction_method = "correlation_proxy"
            method_meta = {"n_gradient_samples": int(gradients.shape[0])}

        arch = classify(impact, volatility, nonlinearity, interaction)
        return (
            FFCASignature(
                impact=impact, volatility=volatility, nonlinearity=nonlinearity,
                interaction=interaction, feature_names=adapter.feature_names,
                archetypes=arch, interaction_ci=ci,
                metadata={
                    "n_features": adapter.n_features,
                    "feature_shape": list(adapter.feature_shape),
                    "interaction_method": interaction_method,
                    "interaction_method_meta": method_meta,
                },
            ),
            gradients,
        )

    def run(self, checkpoints: CheckpointLoader | None = None) -> "FFCAReport":
        t0 = time.time()
        if checkpoints is None:
            print(f"FFCAReport: single checkpoint (no CheckpointLoader given)")
            sig, grads = self._signature_one(self.adapter)
            self.signatures = [sig]
            self.checkpoint_labels = ["current"]
            self.last_gradients = grads
        else:
            print(f"FFCAReport: {len(checkpoints)} checkpoint(s)")
            for label, model in checkpoints:
                self.adapter.model = model  # swap underlying model
                # Re-probe channel adapters in case shapes shifted (unlikely
                # but safe).
                if hasattr(self.adapter, "_activation_shape"):
                    self.adapter._activation_shape = None
                t_ckpt = time.time()
                sig, grads = self._signature_one(self.adapter)
                self.signatures.append(sig)
                self.checkpoint_labels.append(label)
                self.last_gradients = grads
                print(f"  ckpt '{label}' done in {time.time() - t_ckpt:.1f}s")
        self.timing["signatures_s"] = time.time() - t0

        # Trust Score across checkpoints (only meaningful for ≥ 2)
        if self.use_trust and len(self.signatures) >= 2:
            t1 = time.time()
            self.trust = TrustScore()
            self.trust.compute(self.signatures, self.signatures[0].feature_names)
            self.timing["trust_s"] = time.time() - t1
            print(f"  TrustScore done in {self.timing['trust_s']:.1f}s")

        # Co-Sensitivity on the last checkpoint's gradients
        if self.use_cosens:
            t2 = time.time()
            self.cosens = CoSensitivityGroups(
                n_permutations=self.n_cosens_permutations,
                n_bootstrap=self.n_cosens_bootstrap,
            )
            last = self.signatures[-1]
            try:
                self.cosens.compute(
                    gradients=self.last_gradients,
                    impact=last.impact, volatility=last.volatility,
                    nonlinearity=last.nonlinearity, interaction=last.interaction,
                )
                self.timing["cosens_s"] = time.time() - t2
                print(f"  CoSensitivity done in {self.timing['cosens_s']:.1f}s "
                      f"(abort={self.cosens.diagnostics['abort_recommended']})")
            except Exception as e:
                print(f"  CoSensitivity skipped: {e}")

        # Run all diagnostic detectors on the final state
        t3 = time.time()
        self.findings = _diag.run_all(
            self.signatures, self.checkpoint_labels,
            trust=self.trust, cosens=self.cosens,
        )
        self.timing["diagnostics_s"] = time.time() - t3
        crit = sum(1 for f in self.findings if f.severity == "critical")
        warn = sum(1 for f in self.findings if f.severity == "warn")
        info = sum(1 for f in self.findings if f.severity == "info")
        print(f"  Diagnostics: {crit} critical, {warn} warn, {info} info "
              f"({self.timing['diagnostics_s']:.1f}s)")
        return self

    # --------------------------------------------------------------- output
    def to_dict(self) -> dict:
        return {
            "n_checkpoints": len(self.signatures),
            "checkpoint_labels": self.checkpoint_labels,
            "feature_names": self.signatures[0].feature_names if self.signatures else None,
            "n_features": self.signatures[0].n_features if self.signatures else 0,
            "signatures": [s.to_dict() for s in self.signatures],
            "trust": self.trust.results if self.trust else None,
            "trust_summary": self.trust.summary() if self.trust else None,
            "co_sensitivity": {
                "groups": self.cosens.results,
                "diagnostics": self.cosens.diagnostics,
                "summary": self.cosens.summary(),
            } if self.cosens else None,
            "findings": [f.to_dict() for f in self.findings],
            "timing": self.timing,
        }

    def save(self, out_dir: str | Path, *,
             save_plots: bool = True,
             plot_formats: Sequence[str] = ("png",)) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. JSON dump
        with open(out_dir / "report.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

        # 2. Markdown summary
        (out_dir / "report.md").write_text(self.generate_markdown())

        # 3. Plots
        if save_plots:
            from .viz import generate_all_plots
            generate_all_plots(self, out_dir / "plots", formats=plot_formats)

        return out_dir

    # --------------------------------------------------------------- markdown
    def generate_markdown(self) -> str:
        out = ["# FFCA Report\n\n"]
        out.append(f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')} — "
                    f"{len(self.signatures)} checkpoint(s), "
                    f"{self.signatures[0].n_features if self.signatures else 0} features._\n\n")

        if not self.signatures:
            out.append("_No signatures computed._\n")
            return "".join(out)
        last = self.signatures[-1]

        # ----- Diagnostics first (the headline answers) -----
        out.append("## Diagnostics — model health overview\n\n")
        if not self.findings:
            out.append("_No diagnostic detectors fired (no signatures available)._\n\n")
        else:
            for f in self.findings:
                out.append(f"### {f.icon} `{f.name}` · **{f.severity}** — {f.headline}\n\n")
                out.append(f"**What was observed.** {f.observation}\n\n")
                out.append(f"**Why it matters.** {f.why_it_matters}\n\n")
                out.append(f"**What to do.** {f.recommendation}\n\n")

        # ----- The 4-D signature (with explanation) -----
        out.append("## The FFCA 4-D signature\n\n")
        out.append(
            "FFCA decomposes each feature's influence on the model into four "
            "independent axes. Higher values mean stronger / less linear / "
            "more context-dependent / more entangled.\n\n"
            "| Axis | Definition | Reads as |\n|---|---|---|\n"
            "| **Impact** | E[\\|∂f/∂x_i\\|] | how much the feature moves the output |\n"
            "| **Volatility** | Var[∂f/∂x_i] | how context-dependent that effect is |\n"
            "| **Non-linearity** | E[\\|∂²f/∂x_i²\\|] | how curved the response is |\n"
            "| **Interaction** | Σ E[\\|∂²f/∂x_i∂x_j\\|] | how much the feature acts through others |\n\n"
        )
        out.append("### Summary at the final checkpoint\n\n")
        out.append("| Dim | mean | std | min | max |\n|-----|------|-----|-----|-----|\n")
        for k in ("impact", "volatility", "nonlinearity", "interaction"):
            v = getattr(last, k)
            out.append(f"| {k} | {v.mean():.4f} | {v.std():.4f} | {v.min():.4f} | {v.max():.4f} |\n")
        meta_method = last.metadata.get("interaction_method", "?")
        out.append(f"\n_Interaction column computed via_ **{meta_method}**.\n")

        out.append("\n### Top 10 features by Impact\n\n")
        out.append(
            "Sorted by Impact (mean absolute gradient). The archetype column "
            "is FFCA's high-level label for what role each feature plays in "
            "the model. See `docs/adapters.md` for the full archetype table.\n\n"
        )
        out.append("| Rank | Feature | Impact | Interaction | Archetype |\n|--|--|--|--|--|\n")
        top = last.top_k(10, by="impact")
        for r, idx in enumerate(top, 1):
            arch = ARCHETYPE_NAMES[int(last.archetypes[idx])] if last.archetypes is not None else "—"
            out.append(f"| {r} | {last.feature_names[idx]} | "
                       f"{last.impact[idx]:.4f} | {last.interaction[idx]:.4f} | {arch} |\n")

        # ----- Archetype distribution -----
        if last.archetypes is not None:
            counts = np.bincount(last.archetypes, minlength=8)
            tot = counts.sum()
            out.append("\n## Archetype distribution\n\n")
            out.append(
                "How FFCA categorises every feature in the model. Healthy "
                "models have a spread; a single dominant bucket suggests "
                "either under- or over-fitting (see Diagnostics above).\n\n"
            )
            out.append("| Archetype | Count | % | What it means |\n|--|--|--|--|\n")
            archetype_meaning = {
                "Noise": "low everywhere — candidate for pruning",
                "Hidden Interactor": "weak alone, strong via interactions",
                "Workhorse": "strong, linear, independent",
                "Catalyst": "strong AND interacting — load-bearing",
                "Nonlinear Driver": "strong with curved relationship",
                "Volatile Specialist": "strong but context-dependent",
                "Stable Contributor": "moderate, reliable",
                "Complex Driver": "complex behaviour across all four axes",
            }
            for i, c in enumerate(counts):
                if c == 0: continue
                out.append(f"| {ARCHETYPE_NAMES[i]} | {c} | {c/tot:.1%} | "
                           f"{archetype_meaning[ARCHETYPE_NAMES[i]]} |\n")

        # ----- Trust Score table -----
        if self.trust is not None:
            out.append("\n## Trust Score (similarity-weighted stability "
                       f"across {len(self.signatures)} checkpoints)\n\n")
            out.append(
                "Each feature is tracked across training checkpoints. A "
                "feature that keeps the same archetype every time gets high "
                "stability; one that flips around gets low stability. The "
                "decision combines that stability with the feature's mean "
                "Impact.\n\n"
            )
            summary = self.trust.summary()
            out.append("| Decision | Count | What it means |\n|--|--|--|\n")
            decision_meaning = {
                "CONFIDENTLY KEEP": "stable + important — load-bearing",
                "KEEP (stable)": "stable but moderate importance",
                "CONFIDENTLY PRUNE": "stable + always Noise — safe to remove",
                "INVESTIGATE (unstable)": "archetype flipped — role uncertain",
                "MONITOR (borderline)": "stability between 0.5 and 0.7",
            }
            for dec, n in sorted(summary.items(), key=lambda x: -x[1]):
                out.append(f"| {dec} | {n} | {decision_meaning.get(dec, '—')} |\n")

            prunable = [(k, v) for k, v in self.trust.results.items() if "PRUNE" in v["decision"]]
            invest = [(k, v) for k, v in self.trust.results.items() if "INVESTIGATE" in v["decision"]]
            if prunable:
                out.append(f"\n**Prunable features** ({len(prunable)}): " +
                           ", ".join(f"`{k}`" for k, _ in prunable[:10]) + "\n")
            if invest:
                out.append(f"\n**Investigate** ({len(invest)}): " +
                           ", ".join(f"`{k}`" for k, _ in invest[:10]) + "\n")

        # ----- Co-Sensitivity table with verdict -----
        if self.cosens is not None and self.cosens.results:
            d = self.cosens.diagnostics
            verdict = "❌ ABORT — no group is safe to prune" if d["abort_recommended"] \
                else "✅ Prune candidate group identified"
            out.append(f"\n## Co-Sensitivity functional groups — {verdict}\n\n")
            out.append(
                "Features are clustered by gradient-correlation distance "
                "(1 − |ρ|) using k-medoids. The package only recommends "
                "pruning when (a) a group is dominated by Noise Candidates "
                "(>50%), (b) the clustering is statistically distinguishable "
                "from random shuffling (perm-p < 0.05), and (c) the "
                "clustering is stable across 80%-bootstrap resamples "
                "(ARI ≥ 0.5).\n\n"
            )
            out.append(f"- **k** = {d['k']} groups\n")
            out.append(f"- **silhouette** = {d['silhouette_observed']:.3f}  "
                       f"_(higher = tighter clusters; >0.5 is strong, >0.2 is moderate)_\n")
            if not np.isnan(d.get('permutation_p', float('nan'))):
                out.append(f"- **permutation p-value** = {d['permutation_p']:.3f}  "
                           f"_(<0.05 means clusters aren't random)_\n")
            if not np.isnan(d.get('bootstrap_ari_median', float('nan'))):
                out.append(f"- **bootstrap ARI** = {d['bootstrap_ari_median']:.3f}  "
                           f"_(≥0.5 means clustering is stable)_\n")
            out.append(f"- **best NC fraction** = {d['best_nc_fraction']:.1%}  "
                       f"_(needs >50% to prune)_\n\n")
            out.append("| Group | Size | NC % | Mean Impact | Recommendation |\n|--|--|--|--|--|\n")
            for gid, g in sorted(self.cosens.results.items()):
                out.append(f"| {gid} | {g['size']} | {g['nc_fraction']:.1%} | "
                           f"{g['mean_impact']} | {g['recommendation']} |\n")

        # ----- Plot index with captions -----
        plots_dir = "plots/"
        out.append("\n## Plots — what's in each one\n\n")
        caption = {
            "01_signature_radar": ("Radar of the four FFCA axes for the top "
                "features. Lines that hug the outer ring on all four axes are "
                "Complex Drivers; ones that bulge only on the Interaction axis "
                "are Hidden Interactors."),
            "02_archetype_distribution": ("How many features fall into each of "
                "the eight archetypes. The shape of this distribution is the "
                "model's high-level health signature."),
            "03_impact_ranking": ("Top features by mean absolute gradient, "
                "colour-coded by archetype. The colour bar at the top is the "
                "same key as the archetype-distribution plot."),
            "04_interaction_ci": ("Per-feature interaction score with its 95% "
                "Cauchy-HVP confidence interval. Features whose error bars do "
                "NOT cross zero are reliably interacting."),
            "05_channel_archetype_grid": ("One coloured square per channel, "
                "numbered left-to-right, top-to-bottom; colour encodes the "
                "channel's archetype. Useful for spotting clusters of "
                "redundant or Noise channels."),
            "05_pixel_interaction_map": ("Pixel-level interaction reshaped "
                "back into the image grid. Bright regions are where the model's "
                "decision is driven by interactions between pixels."),
            "06_fbr_diagnostic": ("Foreground/Background ratio: mean "
                "interaction in the centre half of the image vs the "
                "surrounding ring. FBR < 0.5 is the Waterbirds-style "
                "background-shortcut signature."),
            "10_impact_evolution": ("Top features' Impact across all "
                "checkpoints. Curves that diverge late in training are "
                "becoming more important; ones that collapse to zero have "
                "been forgotten."),
            "11_ranking_evolution": ("Bump chart of feature rank (by Impact) "
                "over time. Lines that cross frequently indicate unstable "
                "feature importance — see the Trust Score above."),
            "12_archetype_evolution": ("Heatmap of each feature's archetype "
                "(colour) at each checkpoint (column). Vertical stripes = "
                "stable archetype; rainbow rows = unstable."),
            "13_trust_scatter": ("Stability vs Importance scatter. Top-right "
                "quadrant = confidently keep; bottom-left + stable = prune; "
                "anything on the left (stability < 0.5) = investigate."),
            "20_co_sensitivity_groups": ("Cluster sizes and Noise-Candidate "
                "fractions. Bars are coloured red if a group is a prune "
                "candidate (NC > 50%), orange if it's borderline (>30%), "
                "green otherwise."),
        }
        for stem, blurb in caption.items():
            out.append(f"### `{plots_dir}{stem}.png`\n\n{blurb}\n\n")

        # Timing
        if self.timing:
            out.append("\n## Timing\n\n")
            for k, v in self.timing.items():
                out.append(f"- {k}: {v:.2f}s\n")

        return "".join(out)
