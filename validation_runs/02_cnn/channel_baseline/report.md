# FFCA Report

_Generated 2026-05-11 12:57:51 — 4 checkpoint(s), 128 features._

## Diagnostics — model health overview

### ℹ `capacity` · **info** — Healthy archetype distribution

**What was observed.** Noise 6%, Catalyst 20%, Complex 31%.

**Why it matters.** No archetype dominates — the model has a diverse mix of feature roles.

**What to do.** None.

### ℹ `overfitting` · **info** — No volatility spike detected across 4 checkpoints

**What was observed.** Final-checkpoint mean Volatility = 0.07754 ; median of earlier checkpoints = 0.07987 (ratio = 0.97×).

**Why it matters.** Volatility = Var(∂f/∂x_i) measures how context-dependent each feature's effect is. A late-training jump usually means the model is memorising sample-specific quirks rather than learning a stable rule.

**What to do.** Within healthy training; no action required.

## The FFCA 4-D signature

FFCA decomposes each feature's influence on the model into four independent axes. Higher values mean stronger / less linear / more context-dependent / more entangled.

| Axis | Definition | Reads as |
|---|---|---|
| **Impact** | E[\|∂f/∂x_i\|] | how much the feature moves the output |
| **Volatility** | Var[∂f/∂x_i] | how context-dependent that effect is |
| **Non-linearity** | E[\|∂²f/∂x_i²\|] | how curved the response is |
| **Interaction** | Σ E[\|∂²f/∂x_i∂x_j\|] | how much the feature acts through others |

### Summary at the final checkpoint

| Dim | mean | std | min | max |
|-----|------|-----|-----|-----|
| impact | 0.4971 | 0.3776 | 0.0102 | 1.8766 |
| volatility | 0.0775 | 0.1013 | 0.0002 | 0.8823 |
| nonlinearity | 0.7459 | 0.4529 | 0.0709 | 2.5785 |
| interaction | 50.9423 | 12.9015 | 23.8764 | 71.7311 |

_Interaction column computed via_ **correlation_proxy**.

### Top 10 features by Impact

Sorted by Impact (mean absolute gradient). The archetype column is FFCA's high-level label for what role each feature plays in the model. See `docs/adapters.md` for the full archetype table.

| Rank | Feature | Impact | Interaction | Archetype |
|--|--|--|--|--|
| 1 | ch_36 | 1.8766 | 71.6815 | Catalyst |
| 2 | ch_72 | 1.7826 | 71.7311 | Catalyst |
| 3 | ch_100 | 1.4125 | 67.7427 | Catalyst |
| 4 | ch_103 | 1.3040 | 68.8190 | Catalyst |
| 5 | ch_77 | 1.2100 | 62.5493 | Stable Contributor |
| 6 | ch_5 | 1.1894 | 68.1026 | Catalyst |
| 7 | ch_112 | 1.1845 | 66.5168 | Catalyst |
| 8 | ch_92 | 1.1781 | 67.7590 | Catalyst |
| 9 | ch_120 | 1.1739 | 66.7536 | Catalyst |
| 10 | ch_61 | 1.1690 | 65.6220 | Catalyst |

## Archetype distribution

How FFCA categorises every feature in the model. Healthy models have a spread; a single dominant bucket suggests either under- or over-fitting (see Diagnostics above).

| Archetype | Count | % | What it means |
|--|--|--|--|
| Noise | 8 | 6.2% | low everywhere — candidate for pruning |
| Hidden Interactor | 7 | 5.5% | weak alone, strong via interactions |
| Workhorse | 1 | 0.8% | strong, linear, independent |
| Catalyst | 25 | 19.5% | strong AND interacting — load-bearing |
| Nonlinear Driver | 21 | 16.4% | strong with curved relationship |
| Volatile Specialist | 8 | 6.2% | strong but context-dependent |
| Stable Contributor | 18 | 14.1% | moderate, reliable |
| Complex Driver | 40 | 31.2% | complex behaviour across all four axes |

## Plots — what's in each one

### `plots/01_signature_radar.png`

Radar of the four FFCA axes for the top features. Lines that hug the outer ring on all four axes are Complex Drivers; ones that bulge only on the Interaction axis are Hidden Interactors.

### `plots/02_archetype_distribution.png`

How many features fall into each of the eight archetypes. The shape of this distribution is the model's high-level health signature.

### `plots/03_impact_ranking.png`

Top features by mean absolute gradient, colour-coded by archetype. The colour bar at the top is the same key as the archetype-distribution plot.

### `plots/04_interaction_ci.png`

Per-feature interaction score with its 95% Cauchy-HVP confidence interval. Features whose error bars do NOT cross zero are reliably interacting.

### `plots/05_channel_archetype_grid.png`

One coloured square per channel, numbered left-to-right, top-to-bottom; colour encodes the channel's archetype. Useful for spotting clusters of redundant or Noise channels.

### `plots/05_pixel_interaction_map.png`

Pixel-level interaction reshaped back into the image grid. Bright regions are where the model's decision is driven by interactions between pixels.

### `plots/06_fbr_diagnostic.png`

Foreground/Background ratio: mean interaction in the centre half of the image vs the surrounding ring. FBR < 0.5 is the Waterbirds-style background-shortcut signature.

### `plots/10_impact_evolution.png`

Top features' Impact across all checkpoints. Curves that diverge late in training are becoming more important; ones that collapse to zero have been forgotten.

### `plots/11_ranking_evolution.png`

Bump chart of feature rank (by Impact) over time. Lines that cross frequently indicate unstable feature importance — see the Trust Score above.

### `plots/12_archetype_evolution.png`

Heatmap of each feature's archetype (colour) at each checkpoint (column). Vertical stripes = stable archetype; rainbow rows = unstable.

### `plots/13_trust_scatter.png`

Stability vs Importance scatter. Top-right quadrant = confidently keep; bottom-left + stable = prune; anything on the left (stability < 0.5) = investigate.

### `plots/20_co_sensitivity_groups.png`

Cluster sizes and Noise-Candidate fractions. Bars are coloured red if a group is a prune candidate (NC > 50%), orange if it's borderline (>30%), green otherwise.


## Timing

- signatures_s: 0.19s
- diagnostics_s: 0.00s
