# FFCA Analysis Report
**Generated**: 2026-05-10 19:40:37
**Features**: 30
**Checkpoints**: 7 (epochs: [1, 5, 10, 15, 20, 25, 30])

## 1. Training Summary
| Metric | Value |
|--------|-------|
| Final Training Accuracy | 0.9956 |
| Final Validation Accuracy | 0.9561 |
| Best Validation Accuracy | 0.9561 (epoch 6) |
| Training Epochs | 30 |

## 2. FFCA 4D Signature (Final Checkpoint)
| Dimension | Mean | Std | Min | Max |
|-----------|------|-----|-----|-----|
| Impact | 0.4044 | 0.2079 | 0.0753 | 0.7624 |
| Volatility | 0.2430 | 0.2242 | 0.0076 | 0.7029 |
| Nonlinearity | 0.0395 | 0.0166 | 0.0178 | 0.0706 |
| Interaction | 23.1730 | 4.5293 | 6.4106 | 25.8500 |

## 3. Improvement #1: Cauchy-HVP Interaction Estimation
- **Method**: 100 Cauchy HVP probes
- **Speedup vs full Hessian**: 0x
- **Mean interaction score**: 0.70
- **Features with significant interaction (CI > 0)**: 30/30
- **Top 5 interacting features**:
  1. mean texture: I=1.15 [0.77, 1.52]
  2. worst smoothness: I=1.09 [0.74, 1.44]
  3. worst concavity: I=0.99 [0.67, 1.31]
  4. worst area: I=0.99 [0.66, 1.31]
  5. worst texture: I=0.95 [0.64, 1.26]

## 4. Improvement #5: Temporal Stability Trust Score
| Decision | Count | Percentage |
|----------|-------|------------|
| INVESTIGATE (unstable) | 10 | 33.3% |
| MONITOR (borderline) | 8 | 26.7% |
| CONFIDENTLY KEEP | 6 | 20.0% |
| KEEP (stable) | 5 | 16.7% |
| CONFIDENTLY PRUNE | 1 | 3.3% |

**Prunable features** (1):
- smoothness error: stability=0.842, archetype=Noise

**Features needing investigation** (10):
- mean perimeter: stability=0.259, n_archetypes=3
- mean area: stability=0.345, n_archetypes=3
- mean smoothness: stability=0.216, n_archetypes=3

**High-confidence features** (6):
- perimeter error: stability=0.842, importance=0.2609
- area error: stability=0.729, importance=0.3997
- worst radius: stability=0.729, importance=0.3867

## 5. Improvement #6: Co-Sensitivity Functional Groups
| Group | Size | NC Fraction | Mean Impact | Mean Interaction | Recommendation |
|-------|------|------------|-------------|-----------------|----------------|
| 0 | 27 | 7.4% | 0.4306 | 0.7 | KEEP — mostly useful |
| 1 | 1 | 0.0% | 0.3038 | 0.9 | KEEP — mostly useful |
| 2 | 2 | 100.0% | 0.1022 | 0.4 | PRUNE — noise-dominated group ← |

**Pruning potential**: 2/30 features (6.7%) across 3 groups

## 6. Consolidated Recommendations
- **Prune** 1 features with high confidence (stable Noise Candidates)
- **Investigate** 10 features with unstable archetypes — they may be conditionally useful
- **Co-Sensitivity**: 2 features in noise-dominated functional groups can be pruned together
- **Cauchy-HVP**: Interaction computation is 0x faster than full Hessian (30 features)
