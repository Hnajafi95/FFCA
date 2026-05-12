# FFCA Report

_Generated 2026-05-11 12:56:33 — 4 checkpoint(s), 30 features._

## Diagnostics — model health overview

### ✗ `overfitting` · **critical** — Volatility grew 12.3× by the final checkpoint — possible overfitting

**What was observed.** Mean Volatility climbed from 0.01943 (median of epochs ['ep1', 'ep5', 'ep15']) to 0.2383 at 'ep30'.

**Why it matters.** A late-training jump in Volatility means feature effects are now strongly sample-dependent — the model is memorising individual examples rather than generalising.

**What to do.** Consider early stopping at an earlier checkpoint, adding regularisation, or evaluating held-out generalisation.

### ⚠ `trust_instability` · **warn** — 17/30 features are unstable across checkpoints

**What was observed.** More than half (57%) of features changed archetype between checkpoints (similarity-weighted stability < 0.5).

**Why it matters.** High INVESTIGATE rate suggests the model has not settled into stable feature roles — either it is still training, or the data is noisy enough that different epochs use the features differently.

**What to do.** Train for more epochs, or use a learning-rate schedule that converges sooner. If accuracy is good but features are unstable, the model is an ensemble in disguise.

### ℹ `capacity` · **info** — Healthy archetype distribution

**What was observed.** Noise 13%, Catalyst 23%, Complex 27%.

**Why it matters.** No archetype dominates — the model has a diverse mix of feature roles.

**What to do.** None.

### ℹ `co_sensitivity` · **info** — Co-Sensitivity refused to recommend any prune (2 groups found, but no prune-safe group)

**What was observed.** k=2 functional groups, silhouette=0.81, permutation p=0.000, bootstrap ARI=1.00, best group NC fraction=14%.  Abort triggered because: no group has >50% Noise Candidates (best=14%).

**Why it matters.** Co-Sensitivity clusters features by gradient similarity. It only recommends pruning when a whole group is noise-dominated (>50%) AND the clustering itself is statistically supported. Refusing to prune is the SAFE outcome — the alternative is removing useful features.

**What to do.** Pruning is not warranted from this run. If you need to compress the model, use magnitude-based or movement pruning instead.

### ℹ `trust_keep_recommended` · **info** — 6 features are stably important across all checkpoints

**What was observed.** These features retained the same useful archetype (Workhorse / Catalyst / Stable Contributor) at every checkpoint.

**Why it matters.** High-stability + high-importance features are the backbone of the model. Removing or changing them will be felt in accuracy.

**What to do.** Treat these as load-bearing. Protect with extra logging in production; do not prune.

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
| impact | 0.4076 | 0.2016 | 0.0584 | 0.8579 |
| volatility | 0.2383 | 0.2140 | 0.0056 | 0.9347 |
| nonlinearity | 0.0364 | 0.0185 | 0.0122 | 0.0995 |
| interaction | 0.5517 | 0.1536 | 0.3209 | 0.9017 |

_Interaction column computed via_ **cauchy_hvp**.

### Top 10 features by Impact

Sorted by Impact (mean absolute gradient). The archetype column is FFCA's high-level label for what role each feature plays in the model. See `docs/adapters.md` for the full archetype table.

| Rank | Feature | Impact | Interaction | Archetype |
|--|--|--|--|--|
| 1 | worst smoothness | 0.8579 | 0.9017 | Catalyst |
| 2 | worst perimeter | 0.7429 | 0.7835 | Catalyst |
| 3 | worst texture | 0.6845 | 0.8566 | Catalyst |
| 4 | worst area | 0.6813 | 0.6983 | Catalyst |
| 5 | area error | 0.6183 | 0.4932 | Nonlinear Driver |
| 6 | worst radius | 0.6150 | 0.6275 | Catalyst |
| 7 | radius error | 0.5663 | 0.4560 | Nonlinear Driver |
| 8 | worst concave points | 0.5396 | 0.5425 | Volatile Specialist |
| 9 | worst symmetry | 0.5205 | 0.6412 | Catalyst |
| 10 | perimeter error | 0.5039 | 0.4817 | Stable Contributor |

## Archetype distribution

How FFCA categorises every feature in the model. Healthy models have a spread; a single dominant bucket suggests either under- or over-fitting (see Diagnostics above).

| Archetype | Count | % | What it means |
|--|--|--|--|
| Noise | 4 | 13.3% | low everywhere — candidate for pruning |
| Hidden Interactor | 1 | 3.3% | weak alone, strong via interactions |
| Catalyst | 7 | 23.3% | strong AND interacting — load-bearing |
| Nonlinear Driver | 4 | 13.3% | strong with curved relationship |
| Volatile Specialist | 1 | 3.3% | strong but context-dependent |
| Stable Contributor | 5 | 16.7% | moderate, reliable |
| Complex Driver | 8 | 26.7% | complex behaviour across all four axes |

## Trust Score (similarity-weighted stability across 4 checkpoints)

Each feature is tracked across training checkpoints. A feature that keeps the same archetype every time gets high stability; one that flips around gets low stability. The decision combines that stability with the feature's mean Impact.

| Decision | Count | What it means |
|--|--|--|
| INVESTIGATE (unstable) | 17 | archetype flipped — role uncertain |
| MONITOR (borderline) | 7 | stability between 0.5 and 0.7 |
| CONFIDENTLY KEEP | 4 | stable + important — load-bearing |
| KEEP (stable) | 2 | stable but moderate importance |

**Investigate** (17): `mean radius`, `mean perimeter`, `mean area`, `mean smoothness`, `mean symmetry`, `mean fractal dimension`, `radius error`, `texture error`, `smoothness error`, `compactness error`

## Co-Sensitivity functional groups — ❌ ABORT — no group is safe to prune

Features are clustered by gradient-correlation distance (1 − |ρ|) using k-medoids. The package only recommends pruning when (a) a group is dominated by Noise Candidates (>50%), (b) the clustering is statistically distinguishable from random shuffling (perm-p < 0.05), and (c) the clustering is stable across 80%-bootstrap resamples (ARI ≥ 0.5).

- **k** = 2 groups
- **silhouette** = 0.809  _(higher = tighter clusters; >0.5 is strong, >0.2 is moderate)_
- **permutation p-value** = 0.000  _(<0.05 means clusters aren't random)_
- **bootstrap ARI** = 1.000  _(≥0.5 means clustering is stable)_
- **best NC fraction** = 13.8%  _(needs >50% to prune)_

| Group | Size | NC % | Mean Impact | Recommendation |
|--|--|--|--|--|
| 0 | 29 | 13.8% | 0.4196 | KEEP — mostly useful |
| 1 | 1 | 0.0% | 0.0584 | KEEP — mostly useful |

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

- signatures_s: 0.06s
- trust_s: 0.00s
- cosens_s: 0.02s
- diagnostics_s: 0.00s
