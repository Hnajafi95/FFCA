# FFCA Report

_Generated 2026-05-11 12:57:50 — 4 checkpoint(s), 128 features._

## Diagnostics — model health overview

### ⚠ `trust_instability` · **warn** — 66/128 features are unstable across checkpoints

**What was observed.** More than half (52%) of features changed archetype between checkpoints (similarity-weighted stability < 0.5).

**Why it matters.** High INVESTIGATE rate suggests the model has not settled into stable feature roles — either it is still training, or the data is noisy enough that different epochs use the features differently.

**What to do.** Train for more epochs, or use a learning-rate schedule that converges sooner. If accuracy is good but features are unstable, the model is an ensemble in disguise.

### ℹ `capacity` · **info** — Healthy archetype distribution

**What was observed.** Noise 15%, Catalyst 20%, Complex 26%.

**Why it matters.** No archetype dominates — the model has a diverse mix of feature roles.

**What to do.** None.

### ℹ `co_sensitivity` · **info** — Co-Sensitivity refused to recommend any prune (2 groups found, but no prune-safe group)

**What was observed.** k=2 functional groups, silhouette=0.23, permutation p=0.000, bootstrap ARI=0.27, best group NC fraction=15%.  Abort triggered because: no group has >50% Noise Candidates (best=15%) ; bootstrap stability low (ARI=0.27<0.5).

**Why it matters.** Co-Sensitivity clusters features by gradient similarity. It only recommends pruning when a whole group is noise-dominated (>50%) AND the clustering itself is statistically supported. Refusing to prune is the SAFE outcome — the alternative is removing useful features.

**What to do.** Pruning is not warranted from this run. If you need to compress the model, use magnitude-based or movement pruning instead.

### ℹ `overfitting` · **info** — No volatility spike detected across 4 checkpoints

**What was observed.** Final-checkpoint mean Volatility = 0.07754 ; median of earlier checkpoints = 0.07987 (ratio = 0.97×).

**Why it matters.** Volatility = Var(∂f/∂x_i) measures how context-dependent each feature's effect is. A late-training jump usually means the model is memorising sample-specific quirks rather than learning a stable rule.

**What to do.** Within healthy training; no action required.

### ℹ `trust_keep_recommended` · **info** — 21 features are stably important across all checkpoints

**What was observed.** These features retained the same useful archetype (Workhorse / Catalyst / Stable Contributor) at every checkpoint.

**Why it matters.** High-stability + high-importance features are the backbone of the model. Removing or changing them will be felt in accuracy.

**What to do.** Treat these as load-bearing. Protect with extra logging in production; do not prune.

### ℹ `trust_prune_recommended` · **info** — 12 features are confidently Noise across all checkpoints

**What was observed.** These features were in the Noise archetype at every checkpoint with high stability.

**Why it matters.** Stable Noise Candidates are the safest pruning targets — they are not just unimportant at the end, they were never important.

**What to do.** Prune; expect no accuracy loss.

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
| interaction | 31.2497 | 12.5009 | 2.7272 | 63.9127 |

_Interaction column computed via_ **cauchy_hvp**.

### Top 10 features by Impact

Sorted by Impact (mean absolute gradient). The archetype column is FFCA's high-level label for what role each feature plays in the model. See `docs/adapters.md` for the full archetype table.

| Rank | Feature | Impact | Interaction | Archetype |
|--|--|--|--|--|
| 1 | ch_36 | 1.8766 | 63.9127 | Catalyst |
| 2 | ch_72 | 1.7826 | 60.1881 | Catalyst |
| 3 | ch_100 | 1.4125 | 53.0542 | Catalyst |
| 4 | ch_103 | 1.3040 | 47.6332 | Catalyst |
| 5 | ch_77 | 1.2100 | 34.7131 | Stable Contributor |
| 6 | ch_5 | 1.1894 | 39.5869 | Catalyst |
| 7 | ch_112 | 1.1845 | 52.8689 | Catalyst |
| 8 | ch_92 | 1.1781 | 39.6898 | Catalyst |
| 9 | ch_120 | 1.1739 | 43.2011 | Catalyst |
| 10 | ch_61 | 1.1690 | 39.0538 | Nonlinear Driver |

## Archetype distribution

How FFCA categorises every feature in the model. Healthy models have a spread; a single dominant bucket suggests either under- or over-fitting (see Diagnostics above).

| Archetype | Count | % | What it means |
|--|--|--|--|
| Noise | 19 | 14.8% | low everywhere — candidate for pruning |
| Hidden Interactor | 6 | 4.7% | weak alone, strong via interactions |
| Catalyst | 25 | 19.5% | strong AND interacting — load-bearing |
| Nonlinear Driver | 15 | 11.7% | strong with curved relationship |
| Volatile Specialist | 9 | 7.0% | strong but context-dependent |
| Stable Contributor | 21 | 16.4% | moderate, reliable |
| Complex Driver | 33 | 25.8% | complex behaviour across all four axes |

## Trust Score (similarity-weighted stability across 4 checkpoints)

Each feature is tracked across training checkpoints. A feature that keeps the same archetype every time gets high stability; one that flips around gets low stability. The decision combines that stability with the feature's mean Impact.

| Decision | Count | What it means |
|--|--|--|
| INVESTIGATE (unstable) | 66 | archetype flipped — role uncertain |
| MONITOR (borderline) | 29 | stability between 0.5 and 0.7 |
| CONFIDENTLY KEEP | 14 | stable + important — load-bearing |
| CONFIDENTLY PRUNE | 12 | stable + always Noise — safe to remove |
| KEEP (stable) | 7 | stable but moderate importance |

**Prunable features** (12): `ch_8`, `ch_19`, `ch_32`, `ch_48`, `ch_51`, `ch_70`, `ch_86`, `ch_90`, `ch_96`, `ch_99`

**Investigate** (66): `ch_4`, `ch_5`, `ch_6`, `ch_7`, `ch_9`, `ch_12`, `ch_13`, `ch_14`, `ch_15`, `ch_17`

## Co-Sensitivity functional groups — ❌ ABORT — no group is safe to prune

Features are clustered by gradient-correlation distance (1 − |ρ|) using k-medoids. The package only recommends pruning when (a) a group is dominated by Noise Candidates (>50%), (b) the clustering is statistically distinguishable from random shuffling (perm-p < 0.05), and (c) the clustering is stable across 80%-bootstrap resamples (ARI ≥ 0.5).

- **k** = 2 groups
- **silhouette** = 0.232  _(higher = tighter clusters; >0.5 is strong, >0.2 is moderate)_
- **permutation p-value** = 0.000  _(<0.05 means clusters aren't random)_
- **bootstrap ARI** = 0.270  _(≥0.5 means clustering is stable)_
- **best NC fraction** = 15.1%  _(needs >50% to prune)_

| Group | Size | NC % | Mean Impact | Recommendation |
|--|--|--|--|--|
| 0 | 42 | 14.3% | 0.3735 | KEEP — mostly useful |
| 1 | 86 | 15.1% | 0.5575 | KEEP — mostly useful |

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

- signatures_s: 0.59s
- trust_s: 0.01s
- cosens_s: 0.02s
- diagnostics_s: 0.00s
