# FFCA Report

_Generated 2026-05-11 13:01:29 — 1 checkpoint(s), 768 features._

## Diagnostics — model health overview

### ⚠ `data_leakage` · **warn** — 1 feature(s) carry unusually high Impact with low Non-linearity + Interaction — possible leakage

**What was observed.** Suspect features: ['h7_d48'].  These have Impact z-score > 3.0 but Non-linearity and Interaction z-scores both ≤ −0.5 — meaning they dominate the output through a purely linear path.

**Why it matters.** Leaked features (label-derived, near-target proxies) typically have very high Impact AND very low curvature (the model just memorises the linear shortcut). Genuine drivers usually accumulate some non-linearity or interactions during training.

**What to do.** Audit these features for post-hoc derivation from the target. Run an ablation: drop the feature, re-train, and verify the model still meets its metric goals.

### ℹ `capacity` · **info** — Healthy archetype distribution

**What was observed.** Noise 4%, Catalyst 15%, Complex 25%.

**Why it matters.** No archetype dominates — the model has a diverse mix of feature roles.

**What to do.** None.

### ℹ `co_sensitivity` · **info** — Co-Sensitivity refused to recommend any prune (2 groups found, but no prune-safe group)

**What was observed.** k=2 functional groups, silhouette=0.55, permutation p=0.000, bootstrap ARI=0.59, best group NC fraction=5%.  Abort triggered because: no group has >50% Noise Candidates (best=5%).

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
| impact | 0.0628 | 0.0503 | 0.0054 | 0.6051 |
| volatility | 0.0010 | 0.0019 | 0.0000 | 0.0395 |
| nonlinearity | 0.0028 | 0.0017 | 0.0003 | 0.0205 |
| interaction | 0.2414 | 0.0879 | 0.0440 | 1.2648 |

_Interaction column computed via_ **cauchy_hvp**.

### Top 10 features by Impact

Sorted by Impact (mean absolute gradient). The archetype column is FFCA's high-level label for what role each feature plays in the model. See `docs/adapters.md` for the full archetype table.

| Rank | Feature | Impact | Interaction | Archetype |
|--|--|--|--|--|
| 1 | h6_d9 | 0.6051 | 1.2648 | Catalyst |
| 2 | h5_d6 | 0.3971 | 0.6611 | Catalyst |
| 3 | h10_d0 | 0.3423 | 0.9342 | Catalyst |
| 4 | h4_d32 | 0.3401 | 0.8054 | Catalyst |
| 5 | h7_d48 | 0.2568 | 0.1612 | Stable Contributor |
| 6 | h0_d36 | 0.2506 | 0.2070 | Stable Contributor |
| 7 | h6_d46 | 0.2463 | 0.3001 | Catalyst |
| 8 | h8_d14 | 0.2253 | 0.5422 | Catalyst |
| 9 | h8_d43 | 0.2204 | 0.3698 | Catalyst |
| 10 | h5_d54 | 0.2158 | 0.1502 | Nonlinear Driver |

## Archetype distribution

How FFCA categorises every feature in the model. Healthy models have a spread; a single dominant bucket suggests either under- or over-fitting (see Diagnostics above).

| Archetype | Count | % | What it means |
|--|--|--|--|
| Noise | 28 | 3.6% | low everywhere — candidate for pruning |
| Hidden Interactor | 73 | 9.5% | weak alone, strong via interactions |
| Workhorse | 14 | 1.8% | strong, linear, independent |
| Catalyst | 118 | 15.4% | strong AND interacting — load-bearing |
| Nonlinear Driver | 130 | 16.9% | strong with curved relationship |
| Volatile Specialist | 115 | 15.0% | strong but context-dependent |
| Stable Contributor | 95 | 12.4% | moderate, reliable |
| Complex Driver | 195 | 25.4% | complex behaviour across all four axes |

## Co-Sensitivity functional groups — ❌ ABORT — no group is safe to prune

Features are clustered by gradient-correlation distance (1 − |ρ|) using k-medoids. The package only recommends pruning when (a) a group is dominated by Noise Candidates (>50%), (b) the clustering is statistically distinguishable from random shuffling (perm-p < 0.05), and (c) the clustering is stable across 80%-bootstrap resamples (ARI ≥ 0.5).

- **k** = 2 groups
- **silhouette** = 0.552  _(higher = tighter clusters; >0.5 is strong, >0.2 is moderate)_
- **permutation p-value** = 0.000  _(<0.05 means clusters aren't random)_
- **bootstrap ARI** = 0.592  _(≥0.5 means clustering is stable)_
- **best NC fraction** = 4.9%  _(needs >50% to prune)_

| Group | Size | NC % | Mean Impact | Recommendation |
|--|--|--|--|--|
| 0 | 584 | 3.3% | 0.0629 | KEEP — mostly useful |
| 1 | 184 | 4.9% | 0.0624 | KEEP — mostly useful |

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

- signatures_s: 1.00s
- cosens_s: 0.41s
- diagnostics_s: 0.00s
