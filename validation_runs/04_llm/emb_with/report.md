# FFCA Report

_Generated 2026-05-11 13:01:19 — 1 checkpoint(s), 12288 features._

## Diagnostics — model health overview

### ℹ `capacity` · **info** — Healthy archetype distribution

**What was observed.** Noise 10%, Catalyst 21%, Complex 28%.

**Why it matters.** No archetype dominates — the model has a diverse mix of feature roles.

**What to do.** None.

### ℹ `co_sensitivity` · **info** — Co-Sensitivity refused to recommend any prune (7 groups found, but no prune-safe group)

**What was observed.** k=7 functional groups, silhouette=0.48, permutation p=0.000, bootstrap ARI=0.52, best group NC fraction=12%.  Abort triggered because: no group has >50% Noise Candidates (best=12%).

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
| impact | 0.0927 | 0.0957 | 0.0000 | 1.3149 |
| volatility | 0.0297 | 0.0717 | 0.0000 | 2.0411 |
| nonlinearity | 0.3163 | 0.3081 | 0.0000 | 3.4660 |
| interaction | 63.7236 | 45.7464 | 2.6727 | 472.5882 |

_Interaction column computed via_ **cauchy_hvp**.

### Top 10 features by Impact

Sorted by Impact (mean absolute gradient). The archetype column is FFCA's high-level label for what role each feature plays in the model. See `docs/adapters.md` for the full archetype table.

| Rank | Feature | Impact | Interaction | Archetype |
|--|--|--|--|--|
| 1 | t15_h138 | 1.3149 | 255.5944 | Catalyst |
| 2 | t15_h58 | 1.1221 | 290.3001 | Catalyst |
| 3 | t15_h632 | 1.0967 | 180.1278 | Catalyst |
| 4 | t15_h349 | 1.0072 | 224.8061 | Catalyst |
| 5 | t15_h246 | 0.9894 | 271.2095 | Catalyst |
| 6 | t15_h607 | 0.9726 | 290.8379 | Catalyst |
| 7 | t15_h741 | 0.9569 | 262.4370 | Catalyst |
| 8 | t15_h242 | 0.9199 | 385.5701 | Catalyst |
| 9 | t15_h322 | 0.9031 | 302.2015 | Catalyst |
| 10 | t15_h355 | 0.8807 | 210.5761 | Catalyst |

## Archetype distribution

How FFCA categorises every feature in the model. Healthy models have a spread; a single dominant bucket suggests either under- or over-fitting (see Diagnostics above).

| Archetype | Count | % | What it means |
|--|--|--|--|
| Noise | 1224 | 10.0% | low everywhere — candidate for pruning |
| Hidden Interactor | 470 | 3.8% | weak alone, strong via interactions |
| Workhorse | 2 | 0.0% | strong, linear, independent |
| Catalyst | 2602 | 21.2% | strong AND interacting — load-bearing |
| Nonlinear Driver | 1738 | 14.1% | strong with curved relationship |
| Volatile Specialist | 1207 | 9.8% | strong but context-dependent |
| Stable Contributor | 1587 | 12.9% | moderate, reliable |
| Complex Driver | 3458 | 28.1% | complex behaviour across all four axes |

## Co-Sensitivity functional groups — ❌ ABORT — no group is safe to prune

Features are clustered by gradient-correlation distance (1 − |ρ|) using k-medoids. The package only recommends pruning when (a) a group is dominated by Noise Candidates (>50%), (b) the clustering is statistically distinguishable from random shuffling (perm-p < 0.05), and (c) the clustering is stable across 80%-bootstrap resamples (ARI ≥ 0.5).

- **k** = 7 groups
- **silhouette** = 0.482  _(higher = tighter clusters; >0.5 is strong, >0.2 is moderate)_
- **permutation p-value** = 0.000  _(<0.05 means clusters aren't random)_
- **bootstrap ARI** = 0.523  _(≥0.5 means clustering is stable)_
- **best NC fraction** = 12.3%  _(needs >50% to prune)_

| Group | Size | NC % | Mean Impact | Recommendation |
|--|--|--|--|--|
| 0 | 1727 | 5.7% | 0.1493 | KEEP — mostly useful |
| 1 | 7165 | 12.3% | 0.0733 | KEEP — mostly useful |
| 2 | 800 | 5.4% | 0.1124 | KEEP — mostly useful |
| 3 | 681 | 5.1% | 0.1128 | KEEP — mostly useful |
| 4 | 837 | 6.6% | 0.1261 | KEEP — mostly useful |
| 5 | 350 | 10.3% | 0.0813 | KEEP — mostly useful |
| 6 | 728 | 10.2% | 0.0756 | KEEP — mostly useful |

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

- signatures_s: 4.34s
- cosens_s: 154.20s
- diagnostics_s: 0.07s
