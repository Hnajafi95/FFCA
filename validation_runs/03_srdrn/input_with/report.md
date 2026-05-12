# FFCA Report
_Generated 2026-05-11 12:24:12 — 1 checkpoint(s), 858 features._

## 4-D signature (last checkpoint)

| Dim | mean | std | min | max |
|-----|------|-----|-----|-----|
| impact | 393.6118 | 143.9466 | 64.6961 | 843.4035 |
| volatility | 232583.0095 | 163994.5691 | 6742.7314 | 1070409.8918 |
| nonlinearity | 451.2106 | 239.6152 | 35.8060 | 1799.1921 |
| interaction | 32164.7751 | 11881.5919 | 9418.4929 | 67109.6195 |

## Top 10 features by impact

| Rank | Feature | Impact | Interaction | Archetype |
|--|--|--|--|--|
| 1 | px_5_8_7 | 843.4035 | 57467.4675 | Catalyst |
| 2 | px_2_4_6 | 827.1520 | 37160.2935 | Nonlinear Driver |
| 3 | px_0_5_9 | 824.7684 | 52120.1845 | Catalyst |
| 4 | px_4_7_8 | 819.3486 | 54849.8163 | Catalyst |
| 5 | px_0_6_5 | 793.0687 | 43754.5635 | Catalyst |
| 6 | px_3_4_8 | 792.5756 | 51478.9819 | Catalyst |
| 7 | px_2_7_6 | 782.4452 | 59061.2391 | Catalyst |
| 8 | px_5_8_4 | 752.8408 | 48160.1736 | Catalyst |
| 9 | px_5_9_6 | 751.5929 | 40267.3617 | Volatile Specialist |
| 10 | px_1_5_6 | 748.6721 | 51737.4916 | Catalyst |

## Archetype distribution (last checkpoint)

| Archetype | Count | % |
|--|--|--|
| Noise | 118 | 13.8% |
| Hidden Interactor | 36 | 4.2% |
| Catalyst | 179 | 20.9% |
| Nonlinear Driver | 134 | 15.6% |
| Volatile Specialist | 80 | 9.3% |
| Stable Contributor | 84 | 9.8% |
| Complex Driver | 227 | 26.5% |

## Co-Sensitivity groups

k = 7, silhouette = 0.068, perm-p = 0.267, bootstrap-ARI = 0.053, **abort = True**

| Group | Size | NC % | Mean Impact | Rec |
|--|--|--|--|--|
| 0 | 145 | 10.3% | 399.1506 | KEEP — mostly useful |
| 1 | 127 | 14.2% | 397.1789 | KEEP — mostly useful |
| 2 | 124 | 14.5% | 388.7874 | KEEP — mostly useful |
| 3 | 90 | 25.6% | 358.0314 | KEEP — mostly useful |
| 4 | 137 | 8.0% | 397.5394 | KEEP — mostly useful |
| 5 | 108 | 14.8% | 392.6424 | KEEP — mostly useful |
| 6 | 127 | 13.4% | 410.2332 | KEEP — mostly useful |

## Timing

- signatures_s: 581.85s
- cosens_s: 0.43s
