# FFCA Report

_Generated 2026-05-11 12:57:47 — 4 checkpoint(s), 3072 features._

## Diagnostics — model health overview

### ⚠ `overfitting` · **warn** — Volatility grew 3.3× by the final checkpoint — possible overfitting

**What was observed.** Mean Volatility climbed from 4.694e-05 (median of epochs ['ep1', 'ep3', 'ep6']) to 0.0001541 at 'ep8'.

**Why it matters.** A late-training jump in Volatility means feature effects are now strongly sample-dependent — the model is memorising individual examples rather than generalising.

**What to do.** Consider early stopping at an earlier checkpoint, adding regularisation, or evaluating held-out generalisation.

### ⚠ `trust_instability` · **warn** — 1611/3072 features are unstable across checkpoints

**What was observed.** More than half (52%) of features changed archetype between checkpoints (similarity-weighted stability < 0.5).

**Why it matters.** High INVESTIGATE rate suggests the model has not settled into stable feature roles — either it is still training, or the data is noisy enough that different epochs use the features differently.

**What to do.** Train for more epochs, or use a learning-rate schedule that converges sooner. If accuracy is good but features are unstable, the model is an ensemble in disguise.

### ℹ `capacity` · **info** — Healthy archetype distribution

**What was observed.** Noise 12%, Catalyst 22%, Complex 27%.

**Why it matters.** No archetype dominates — the model has a diverse mix of feature roles.

**What to do.** None.

### ℹ `co_sensitivity` · **info** — Co-Sensitivity refused to recommend any prune (7 groups found, but no prune-safe group)

**What was observed.** k=7 functional groups, silhouette=0.05, permutation p=0.000, bootstrap ARI=0.03, best group NC fraction=15%.  Abort triggered because: no group has >50% Noise Candidates (best=15%) ; bootstrap stability low (ARI=0.03<0.5).

**Why it matters.** Co-Sensitivity clusters features by gradient similarity. It only recommends pruning when a whole group is noise-dominated (>50%) AND the clustering itself is statistically supported. Refusing to prune is the SAFE outcome — the alternative is removing useful features.

**What to do.** Pruning is not warranted from this run. If you need to compress the model, use magnitude-based or movement pruning instead.

### ℹ `shortcut_learning` · **info** — No background-shortcut signal — Foreground/Background interaction ratio = 0.62

**What was observed.** Mean interaction in centre 50% of the image: 0.6460 ; in the surrounding ring: 0.3918 ; FBR=0.622.

**Why it matters.** A model that relies on background pixels (FBR < 0.5) is using context cues that won't generalise — the Waterbirds failure mode. The model should put more interaction on the central subject.

**What to do.** No remediation needed.

### ℹ `trust_keep_recommended` · **info** — 586 features are stably important across all checkpoints

**What was observed.** These features retained the same useful archetype (Workhorse / Catalyst / Stable Contributor) at every checkpoint.

**Why it matters.** High-stability + high-importance features are the backbone of the model. Removing or changing them will be felt in accuracy.

**What to do.** Treat these as load-bearing. Protect with extra logging in production; do not prune.

### ℹ `trust_prune_recommended` · **info** — 194 features are confidently Noise across all checkpoints

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
| impact | 0.0088 | 0.0043 | 0.0016 | 0.0334 |
| volatility | 0.0002 | 0.0002 | 0.0000 | 0.0018 |
| nonlinearity | 0.0152 | 0.0088 | 0.0008 | 0.0556 |
| interaction | 0.4553 | 0.1976 | 0.0801 | 1.2009 |

_Interaction column computed via_ **cauchy_hvp**.

### Top 10 features by Impact

Sorted by Impact (mean absolute gradient). The archetype column is FFCA's high-level label for what role each feature plays in the model. See `docs/adapters.md` for the full archetype table.

| Rank | Feature | Impact | Interaction | Archetype |
|--|--|--|--|--|
| 1 | px_1_13_15 | 0.0334 | 0.6604 | Catalyst |
| 2 | px_1_16_15 | 0.0305 | 0.7747 | Catalyst |
| 3 | px_1_14_15 | 0.0298 | 0.6203 | Catalyst |
| 4 | px_1_13_13 | 0.0290 | 0.6634 | Catalyst |
| 5 | px_1_17_15 | 0.0288 | 1.1503 | Catalyst |
| 6 | px_1_15_12 | 0.0286 | 0.6854 | Catalyst |
| 7 | px_1_15_15 | 0.0284 | 0.8355 | Catalyst |
| 8 | px_1_15_13 | 0.0277 | 0.7799 | Catalyst |
| 9 | px_1_14_13 | 0.0276 | 0.6696 | Catalyst |
| 10 | px_1_17_16 | 0.0276 | 0.8585 | Catalyst |

## Archetype distribution

How FFCA categorises every feature in the model. Healthy models have a spread; a single dominant bucket suggests either under- or over-fitting (see Diagnostics above).

| Archetype | Count | % | What it means |
|--|--|--|--|
| Noise | 369 | 12.0% | low everywhere — candidate for pruning |
| Hidden Interactor | 86 | 2.8% | weak alone, strong via interactions |
| Catalyst | 681 | 22.2% | strong AND interacting — load-bearing |
| Nonlinear Driver | 497 | 16.2% | strong with curved relationship |
| Volatile Specialist | 258 | 8.4% | strong but context-dependent |
| Stable Contributor | 365 | 11.9% | moderate, reliable |
| Complex Driver | 816 | 26.6% | complex behaviour across all four axes |

## Trust Score (similarity-weighted stability across 4 checkpoints)

Each feature is tracked across training checkpoints. A feature that keeps the same archetype every time gets high stability; one that flips around gets low stability. The decision combines that stability with the feature's mean Impact.

| Decision | Count | What it means |
|--|--|--|
| INVESTIGATE (unstable) | 1611 | archetype flipped — role uncertain |
| MONITOR (borderline) | 681 | stability between 0.5 and 0.7 |
| CONFIDENTLY KEEP | 308 | stable + important — load-bearing |
| KEEP (stable) | 278 | stable but moderate importance |
| CONFIDENTLY PRUNE | 194 | stable + always Noise — safe to remove |

**Prunable features** (194): `px_0_0_0`, `px_0_0_1`, `px_0_0_2`, `px_0_0_3`, `px_0_0_4`, `px_0_0_9`, `px_0_0_11`, `px_0_0_16`, `px_0_0_19`, `px_0_0_21`

**Investigate** (1611): `px_0_0_5`, `px_0_0_6`, `px_0_0_7`, `px_0_0_8`, `px_0_0_10`, `px_0_0_12`, `px_0_0_13`, `px_0_0_14`, `px_0_0_15`, `px_0_0_17`

## Co-Sensitivity functional groups — ❌ ABORT — no group is safe to prune

Features are clustered by gradient-correlation distance (1 − |ρ|) using k-medoids. The package only recommends pruning when (a) a group is dominated by Noise Candidates (>50%), (b) the clustering is statistically distinguishable from random shuffling (perm-p < 0.05), and (c) the clustering is stable across 80%-bootstrap resamples (ARI ≥ 0.5).

- **k** = 7 groups
- **silhouette** = 0.051  _(higher = tighter clusters; >0.5 is strong, >0.2 is moderate)_
- **permutation p-value** = 0.000  _(<0.05 means clusters aren't random)_
- **bootstrap ARI** = 0.034  _(≥0.5 means clustering is stable)_
- **best NC fraction** = 14.8%  _(needs >50% to prune)_

| Group | Size | NC % | Mean Impact | Recommendation |
|--|--|--|--|--|
| 0 | 469 | 10.7% | 0.0083 | KEEP — mostly useful |
| 1 | 884 | 11.5% | 0.01 | KEEP — mostly useful |
| 2 | 300 | 14.7% | 0.008 | KEEP — mostly useful |
| 3 | 445 | 14.8% | 0.0077 | KEEP — mostly useful |
| 4 | 313 | 10.9% | 0.009 | KEEP — mostly useful |
| 5 | 234 | 10.7% | 0.0083 | KEEP — mostly useful |
| 6 | 427 | 11.2% | 0.0086 | KEEP — mostly useful |

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

- signatures_s: 1.49s
- trust_s: 0.24s
- cosens_s: 2.70s
- diagnostics_s: 0.00s
