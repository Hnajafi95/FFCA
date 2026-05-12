# Validation summary: `03_srdrn`

_2 baseline / with-improvements pair(s) found._

## Pair: `alpha`

- **Baseline:** `alpha_baseline/`
- **With improvements:** `alpha_with/`
- **Checkpoints:** ['current']
- **Features:** 6

### Archetype distribution per checkpoint

| Checkpoint | Baseline | With improvements |
|---|---|---|
| current | `Catalyst`=1 (17%), `Noise`=1 (17%), `Volatile Specialist`=1 (17%), `Hidden Interactor`=1 (17%), `Nonlinear Driver`=1 (17%), `Complex Driver`=1 (17%) | `Complex Driver`=2 (33%), `Catalyst`=1 (17%), `Volatile Specialist`=1 (17%), `Hidden Interactor`=1 (17%), `Nonlinear Driver`=1 (17%) |

### Diagnostic findings

- **Baseline only:** _(none)_
- **With-improvements only:** _(none)_
- **Shared:** _(none)_

### Final-checkpoint interaction range

| Statistic | Baseline | With improvements |
|---|---|---|
| min | 1.124 | 0.1381 |
| mean | 1.509 | 0.4975 |
| max | 2.119 | 0.8067 |

### Top-5 feature overlap

- **Jaccard:** 0.67 (4 / 6)
- **Baseline top-5:** `tas`, `sfcWind`, `tasmin`, `tasmax`, `huss`
- **With-improvements top-5:** `sfcWind`, `tas`, `huss`, `tasmax`, `pr`
- **Only in baseline:** `tasmin`
- **Only in with-improvements:** `pr`

## Pair: `input`

- **Baseline:** `input_baseline/`
- **With improvements:** `input_with/`
- **Checkpoints:** ['current']
- **Features:** 858

### Archetype distribution per checkpoint

| Checkpoint | Baseline | With improvements |
|---|---|---|
| current | `Complex Driver`=228 (27%), `Nonlinear Driver`=187 (22%), `Catalyst`=113 (13%), `Hidden Interactor`=102 (12%), `Volatile Specialist`=90 (10%), `Stable Contributor`=84 (10%), `Noise`=54 (6%) | `Complex Driver`=227 (26%), `Catalyst`=179 (21%), `Nonlinear Driver`=134 (16%), `Noise`=118 (14%), `Stable Contributor`=84 (10%), `Volatile Specialist`=80 (9%), `Hidden Interactor`=36 (4%) |

### Diagnostic findings

- **Baseline only:** _(none)_
- **With-improvements only:** _(none)_
- **Shared:** _(none)_

### Final-checkpoint interaction range

| Statistic | Baseline | With improvements |
|---|---|---|
| min | 192.6 | 9418 |
| mean | 213 | 3.216e+04 |
| max | 234 | 6.711e+04 |

### Top-5 feature overlap

- **Jaccard:** 0.00 (0 / 10)
- **Baseline top-5:** `px_2_9_5`, `px_0_0_7`, `px_3_4_9`, `px_3_3_9`, `px_0_3_7`
- **With-improvements top-5:** `px_1_5_8`, `px_2_7_4`, `px_5_7_5`, `px_2_5_6`, `px_0_6_7`
- **Only in baseline:** `px_0_0_7`, `px_0_3_7`, `px_2_9_5`, `px_3_3_9`, `px_3_4_9`
- **Only in with-improvements:** `px_0_6_7`, `px_1_5_8`, `px_2_5_6`, `px_2_7_4`, `px_5_7_5`
