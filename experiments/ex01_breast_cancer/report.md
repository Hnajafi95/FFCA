# FFCA Report
_Generated 2026-05-11 00:10:26 — 4 checkpoint(s), 30 features._

## 4-D signature (last checkpoint)

| Dim | mean | std | min | max |
|-----|------|-----|-----|-----|
| impact | 0.3864 | 0.1884 | 0.0978 | 0.8804 |
| volatility | 0.1993 | 0.1697 | 0.0076 | 0.6955 |
| nonlinearity | 0.0443 | 0.0323 | 0.0110 | 0.1482 |
| interaction | 0.6284 | 0.1939 | 0.2776 | 1.1064 |

## Top 10 features by impact

| Rank | Feature | Impact | Interaction | Archetype |
|--|--|--|--|--|
| 1 | area error | 0.8804 | 1.1064 | Catalyst |
| 2 | worst texture | 0.6782 | 0.9891 | Catalyst |
| 3 | worst perimeter | 0.6653 | 0.7641 | Catalyst |
| 4 | worst smoothness | 0.6234 | 0.9798 | Catalyst |
| 5 | radius error | 0.5635 | 0.7675 | Catalyst |
| 6 | worst radius | 0.5279 | 0.7284 | Nonlinear Driver |
| 7 | worst concave points | 0.5040 | 0.5619 | Nonlinear Driver |
| 8 | mean texture | 0.5006 | 0.8990 | Catalyst |
| 9 | worst concavity | 0.4990 | 0.6543 | Volatile Specialist |
| 10 | compactness error | 0.4618 | 0.8264 | Catalyst |

## Archetype distribution (last checkpoint)

| Archetype | Count | % |
|--|--|--|
| Noise | 3 | 10.0% |
| Hidden Interactor | 1 | 3.3% |
| Catalyst | 7 | 23.3% |
| Nonlinear Driver | 3 | 10.0% |
| Volatile Specialist | 2 | 6.7% |
| Stable Contributor | 4 | 13.3% |
| Complex Driver | 10 | 33.3% |

## Trust Score (across 4 checkpoints)

| Decision | Count |
|--|--|
| INVESTIGATE (unstable) | 23 |
| CONFIDENTLY KEEP | 3 |
| MONITOR (borderline) | 3 |
| KEEP (stable) | 1 |

**Investigate** (23): `mean radius`, `mean texture`, `mean perimeter`, `mean smoothness`, `mean compactness`, `mean concavity`, `mean concave points`, `mean symmetry`, `mean fractal dimension`, `radius error`

## Co-Sensitivity groups

k = 3, silhouette = 0.795, perm-p = 0.000, bootstrap-ARI = 1.000, **abort = True**

| Group | Size | NC % | Mean Impact | Rec |
|--|--|--|--|--|
| 0 | 28 | 10.7% | 0.4032 | KEEP — mostly useful |
| 1 | 1 | 0.0% | 0.1053 | KEEP — mostly useful |
| 2 | 1 | 0.0% | 0.1958 | KEEP — mostly useful |

## Timing

- signatures_s: 0.06s
- trust_s: 0.00s
- cosens_s: 0.02s
