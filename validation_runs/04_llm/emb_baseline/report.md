# FFCA Report

_Generated 2026-05-11 13:01:27 — 1 checkpoint(s), 12288 features._

## Diagnostics — model health overview

### ℹ `capacity` · **info** — Healthy archetype distribution

**What was observed.** Noise 3%, Catalyst 5%, Complex 17%.

**Why it matters.** No archetype dominates — the model has a diverse mix of feature roles.

**What to do.** None.

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
| impact | 0.0927 | 0.0957 | 0.0000 | 1.3149 |
| volatility | 0.0297 | 0.0717 | 0.0000 | 2.0411 |
| nonlinearity | 0.3163 | 0.3081 | 0.0000 | 3.4660 |
| interaction | 6332.9649 | 1984.4106 | 1704.1855 | 8335.4744 |

_Interaction column computed via_ **correlation_proxy**.

### Top 10 features by Impact

Sorted by Impact (mean absolute gradient). The archetype column is FFCA's high-level label for what role each feature plays in the model. See `docs/adapters.md` for the full archetype table.

| Rank | Feature | Impact | Interaction | Archetype |
|--|--|--|--|--|
| 1 | t15_h138 | 1.3149 | 8256.2597 | Nonlinear Driver |
| 2 | t15_h58 | 1.1221 | 4049.9670 | Nonlinear Driver |
| 3 | t15_h632 | 1.0967 | 6967.4790 | Nonlinear Driver |
| 4 | t15_h349 | 1.0072 | 7593.0456 | Nonlinear Driver |
| 5 | t15_h246 | 0.9894 | 4816.4813 | Nonlinear Driver |
| 6 | t15_h607 | 0.9726 | 6179.9274 | Nonlinear Driver |
| 7 | t15_h741 | 0.9569 | 2180.0953 | Nonlinear Driver |
| 8 | t15_h242 | 0.9199 | 5697.7070 | Nonlinear Driver |
| 9 | t15_h322 | 0.9031 | 3805.9035 | Nonlinear Driver |
| 10 | t15_h355 | 0.8807 | 7084.1700 | Nonlinear Driver |

## Archetype distribution

How FFCA categorises every feature in the model. Healthy models have a spread; a single dominant bucket suggests either under- or over-fitting (see Diagnostics above).

| Archetype | Count | % | What it means |
|--|--|--|--|
| Noise | 324 | 2.6% | low everywhere — candidate for pruning |
| Hidden Interactor | 3244 | 26.4% | weak alone, strong via interactions |
| Workhorse | 21 | 0.2% | strong, linear, independent |
| Catalyst | 603 | 4.9% | strong AND interacting — load-bearing |
| Nonlinear Driver | 2979 | 24.2% | strong with curved relationship |
| Volatile Specialist | 1096 | 8.9% | strong but context-dependent |
| Stable Contributor | 1900 | 15.5% | moderate, reliable |
| Complex Driver | 2121 | 17.3% | complex behaviour across all four axes |

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

- signatures_s: 5.23s
- diagnostics_s: 0.00s
