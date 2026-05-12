# FFCA Report

_Generated 2026-05-11 12:57:49 — 4 checkpoint(s), 3072 features._

## Diagnostics — model health overview

### ⚠ `overfitting` · **warn** — Volatility grew 3.3× by the final checkpoint — possible overfitting

**What was observed.** Mean Volatility climbed from 4.694e-05 (median of epochs ['ep1', 'ep3', 'ep6']) to 0.0001541 at 'ep8'.

**Why it matters.** A late-training jump in Volatility means feature effects are now strongly sample-dependent — the model is memorising individual examples rather than generalising.

**What to do.** Consider early stopping at an earlier checkpoint, adding regularisation, or evaluating held-out generalisation.

### ℹ `capacity` · **info** — Healthy archetype distribution

**What was observed.** Noise 4%, Catalyst 14%, Complex 26%.

**Why it matters.** No archetype dominates — the model has a diverse mix of feature roles.

**What to do.** None.

### ℹ `shortcut_learning` · **info** — No background-shortcut signal — Foreground/Background interaction ratio = 0.50

**What was observed.** Mean interaction in centre 50% of the image: 711.6780 ; in the surrounding ring: 706.9201 ; FBR=0.502.

**Why it matters.** A model that relies on background pixels (FBR < 0.5) is using context cues that won't generalise — the Waterbirds failure mode. The model should put more interaction on the central subject.

**What to do.** No remediation needed.

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
| impact | 0.0088 | 0.0043 | 0.0016 | 0.0334 |
| volatility | 0.0002 | 0.0002 | 0.0000 | 0.0018 |
| nonlinearity | 0.0152 | 0.0088 | 0.0008 | 0.0556 |
| interaction | 708.1096 | 67.0247 | 511.4437 | 923.2390 |

_Interaction column computed via_ **correlation_proxy**.

### Top 10 features by Impact

Sorted by Impact (mean absolute gradient). The archetype column is FFCA's high-level label for what role each feature plays in the model. See `docs/adapters.md` for the full archetype table.

| Rank | Feature | Impact | Interaction | Archetype |
|--|--|--|--|--|
| 1 | px_1_13_15 | 0.0334 | 818.7385 | Catalyst |
| 2 | px_1_16_15 | 0.0305 | 777.8053 | Catalyst |
| 3 | px_1_14_15 | 0.0298 | 839.4891 | Catalyst |
| 4 | px_1_13_13 | 0.0290 | 845.2393 | Catalyst |
| 5 | px_1_17_15 | 0.0288 | 789.8538 | Catalyst |
| 6 | px_1_15_12 | 0.0286 | 798.0012 | Catalyst |
| 7 | px_1_15_15 | 0.0284 | 793.4664 | Catalyst |
| 8 | px_1_15_13 | 0.0277 | 848.0470 | Catalyst |
| 9 | px_1_14_13 | 0.0276 | 861.5699 | Catalyst |
| 10 | px_1_17_16 | 0.0276 | 779.2789 | Catalyst |

## Archetype distribution

How FFCA categorises every feature in the model. Healthy models have a spread; a single dominant bucket suggests either under- or over-fitting (see Diagnostics above).

| Archetype | Count | % | What it means |
|--|--|--|--|
| Noise | 112 | 3.6% | low everywhere — candidate for pruning |
| Hidden Interactor | 353 | 11.5% | weak alone, strong via interactions |
| Catalyst | 415 | 13.5% | strong AND interacting — load-bearing |
| Nonlinear Driver | 694 | 22.6% | strong with curved relationship |
| Volatile Specialist | 336 | 10.9% | strong but context-dependent |
| Stable Contributor | 350 | 11.4% | moderate, reliable |
| Complex Driver | 812 | 26.4% | complex behaviour across all four axes |

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

- signatures_s: 1.09s
- diagnostics_s: 0.00s
