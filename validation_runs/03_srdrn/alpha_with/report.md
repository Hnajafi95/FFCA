# FFCA Report

_Generated 2026-05-11 13:04:30 — 1 checkpoint(s), 6 features._

## Diagnostics — model health overview

### ℹ `co_sensitivity` · **info** — Co-Sensitivity refused to recommend any prune (3 groups found, but no prune-safe group)

**What was observed.** k=3 functional groups, silhouette=0.26, permutation p=0.133, bootstrap ARI=0.12, best group NC fraction=0%.  Abort triggered because: no group has >50% Noise Candidates (best=0%) ; clusters indistinguishable from random shuffle (p=0.133) ; bootstrap stability low (ARI=0.12<0.5).

**Why it matters.** Co-Sensitivity clusters features by gradient similarity. It only recommends pruning when a whole group is noise-dominated (>50%) AND the clustering itself is statistically supported. Refusing to prune is the SAFE outcome — the alternative is removing useful features.

**What to do.** Pruning is not warranted from this run. If you need to compress the model, use magnitude-based or movement pruning instead.

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
| impact | 0.1359 | 0.0584 | 0.0541 | 0.2340 |
| volatility | 0.0302 | 0.0249 | 0.0041 | 0.0809 |
| nonlinearity | 0.2133 | 0.1215 | 0.0308 | 0.3671 |
| interaction | 0.4975 | 0.2113 | 0.1381 | 0.8067 |

_Interaction column computed via_ **cauchy_hvp**.

### Top 10 features by Impact

Sorted by Impact (mean absolute gradient). The archetype column is FFCA's high-level label for what role each feature plays in the model. See `docs/adapters.md` for the full archetype table.

| Rank | Feature | Impact | Interaction | Archetype |
|--|--|--|--|--|
| 1 | tasmax | 0.2340 | 0.4590 | Nonlinear Driver |
| 2 | huss | 0.1725 | 0.4618 | Volatile Specialist |
| 3 | tas | 0.1376 | 0.6872 | Catalyst |
| 4 | tasmin | 0.1346 | 0.1381 | Complex Driver |
| 5 | sfcWind | 0.0827 | 0.8067 | Hidden Interactor |
| 6 | pr | 0.0541 | 0.4321 | Complex Driver |

## Archetype distribution

How FFCA categorises every feature in the model. Healthy models have a spread; a single dominant bucket suggests either under- or over-fitting (see Diagnostics above).

| Archetype | Count | % | What it means |
|--|--|--|--|
| Hidden Interactor | 1 | 16.7% | weak alone, strong via interactions |
| Catalyst | 1 | 16.7% | strong AND interacting — load-bearing |
| Nonlinear Driver | 1 | 16.7% | strong with curved relationship |
| Volatile Specialist | 1 | 16.7% | strong but context-dependent |
| Complex Driver | 2 | 33.3% | complex behaviour across all four axes |

## Co-Sensitivity functional groups — ❌ ABORT — no group is safe to prune

Features are clustered by gradient-correlation distance (1 − |ρ|) using k-medoids. The package only recommends pruning when (a) a group is dominated by Noise Candidates (>50%), (b) the clustering is statistically distinguishable from random shuffling (perm-p < 0.05), and (c) the clustering is stable across 80%-bootstrap resamples (ARI ≥ 0.5).

- **k** = 3 groups
- **silhouette** = 0.261  _(higher = tighter clusters; >0.5 is strong, >0.2 is moderate)_
- **permutation p-value** = 0.133  _(<0.05 means clusters aren't random)_
- **bootstrap ARI** = 0.118  _(≥0.5 means clustering is stable)_
- **best NC fraction** = 0.0%  _(needs >50% to prune)_

| Group | Size | NC % | Mean Impact | Recommendation |
|--|--|--|--|--|
| 0 | 3 | 0.0% | 0.1482 | KEEP — mostly useful |
| 1 | 2 | 0.0% | 0.1584 | KEEP — mostly useful |
| 2 | 1 | 0.0% | 0.0541 | KEEP — mostly useful |

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

- signatures_s: 458.73s
- cosens_s: 0.21s
