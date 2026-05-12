"""ffca.viz — comprehensive plotting suite for FFCAReport.

Top-level entry: ``generate_all_plots(report, out_dir, formats)``.
"""
from __future__ import annotations

from pathlib import Path

from . import diagnostics, dynamic, spatial, static


def generate_all_plots(report, out_dir, formats=("png",)):
    """Produce every relevant plot for a finished FFCAReport.

    Auto-selects which plots make sense based on:
      - is it dynamic (multiple checkpoints) or static (one)?
      - does the adapter expose a spatial feature_shape (pixels)?
      - is there a Trust / Co-Sens / interaction CI to plot?

    Plots are saved to ``out_dir/<name>.<fmt>`` for each format requested.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    last_sig = report.signatures[-1]

    # ----- static plots (always) ------------------------------------------
    static.signature_radar(last_sig, top_k=min(6, last_sig.n_features),
                           out=out_dir / "01_signature_radar", formats=formats)
    written.append(out_dir / "01_signature_radar")

    static.archetype_distribution(last_sig, out=out_dir / "02_archetype_distribution",
                                  formats=formats)
    written.append(out_dir / "02_archetype_distribution")

    static.impact_ranking(last_sig, top_k=min(20, last_sig.n_features),
                          out=out_dir / "03_impact_ranking", formats=formats)
    written.append(out_dir / "03_impact_ranking")

    if last_sig.interaction_ci is not None:
        static.interaction_ci_plot(last_sig, top_k=min(30, last_sig.n_features),
                                   out=out_dir / "04_interaction_ci", formats=formats)
        written.append(out_dir / "04_interaction_ci")

    # ----- spatial plots if adapter exposes (C,H,W) -----------------------
    feature_shape = tuple(last_sig.metadata.get("feature_shape", ())) if last_sig.metadata else ()
    if len(feature_shape) == 3:
        spatial.pixel_interaction_map(last_sig, feature_shape,
                                      out=out_dir / "05_pixel_interaction_map",
                                      formats=formats)
        written.append(out_dir / "05_pixel_interaction_map")
        spatial.fbr_diagnostic(last_sig, feature_shape,
                               out=out_dir / "06_fbr_diagnostic",
                               formats=formats)
        written.append(out_dir / "06_fbr_diagnostic")
    elif len(feature_shape) == 1 and last_sig.n_features >= 8:
        # 1-D feature axis — archetype grid (label by convention)
        unit = spatial._infer_unit_label(last_sig.feature_names)
        spatial.channel_archetype_grid(last_sig,
                                       out=out_dir / f"05_{unit}_archetype_grid",
                                       formats=formats, unit_label=unit)
        written.append(out_dir / f"05_{unit}_archetype_grid")

    # ----- dynamic plots --------------------------------------------------
    if len(report.signatures) >= 2:
        dynamic.impact_evolution_curves(report.signatures, report.checkpoint_labels,
                                        top_k=min(8, last_sig.n_features),
                                        out=out_dir / "10_impact_evolution",
                                        formats=formats)
        dynamic.ranking_evolution(report.signatures, report.checkpoint_labels,
                                  top_k=min(8, last_sig.n_features),
                                  out=out_dir / "11_ranking_evolution",
                                  formats=formats)
        dynamic.archetype_evolution_heatmap(report.signatures, report.checkpoint_labels,
                                            top_k=min(30, last_sig.n_features),
                                            out=out_dir / "12_archetype_evolution",
                                            formats=formats)
        if report.trust:
            dynamic.trust_score_scatter(report.trust.results,
                                        out=out_dir / "13_trust_scatter",
                                        formats=formats)

    # ----- diagnostics ----------------------------------------------------
    if report.cosens and report.cosens.results:
        diagnostics.co_sensitivity_groups(report.cosens,
                                          out=out_dir / "20_co_sensitivity_groups",
                                          formats=formats)

    return written


__all__ = ["generate_all_plots", "static", "spatial", "dynamic", "diagnostics"]
