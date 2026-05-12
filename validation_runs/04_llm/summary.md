# Validation summary: `04_llm`

_2 baseline / with-improvements pair(s) found._

## Pair: `emb`

- **Baseline:** `emb_baseline/`
- **With improvements:** `emb_with/`
- **Checkpoints:** ['current']
- **Features:** 12288

### Archetype distribution per checkpoint

| Checkpoint | Baseline | With improvements |
|---|---|---|
| current | `Hidden Interactor`=3244 (26%), `Nonlinear Driver`=2979 (24%), `Complex Driver`=2121 (17%), `Stable Contributor`=1900 (15%), `Volatile Specialist`=1096 (9%), `Catalyst`=603 (5%), `Noise`=324 (3%), `Workhorse`=21 (0%) | `Complex Driver`=3458 (28%), `Catalyst`=2602 (21%), `Nonlinear Driver`=1738 (14%), `Stable Contributor`=1587 (13%), `Noise`=1224 (10%), `Volatile Specialist`=1207 (10%), `Hidden Interactor`=470 (4%), `Workhorse`=2 (0%) |

### Diagnostic findings

- **Baseline only:** _(none)_
- **With-improvements only:** `co_sensitivity`
- **Shared:** `capacity`

Unique with-improvements findings (headline):
- `co_sensitivity` (info): Co-Sensitivity refused to recommend any prune (7 groups found, but no prune-safe group)

### Final-checkpoint interaction range

| Statistic | Baseline | With improvements |
|---|---|---|
| min | 1704 | 2.673 |
| mean | 6333 | 63.72 |
| max | 8335 | 472.6 |

### Top-5 feature overlap

- **Jaccard:** 0.00 (0 / 10)
- **Baseline top-5:** `t7_h754`, `t7_h419`, `t7_h761`, `t7_h340`, `t9_h295`
- **With-improvements top-5:** `t15_h732`, `t15_h762`, `t15_h550`, `t15_h81`, `t15_h203`
- **Only in baseline:** `t7_h340`, `t7_h419`, `t7_h754`, `t7_h761`, `t9_h295`
- **Only in with-improvements:** `t15_h203`, `t15_h550`, `t15_h732`, `t15_h762`, `t15_h81`

## Pair: `head`

- **Baseline:** `head_baseline/`
- **With improvements:** `head_with/`
- **Checkpoints:** ['current']
- **Features:** 768

### Archetype distribution per checkpoint

| Checkpoint | Baseline | With improvements |
|---|---|---|
| current | `Complex Driver`=199 (26%), `Nonlinear Driver`=157 (20%), `Stable Contributor`=114 (15%), `Catalyst`=110 (14%), `Hidden Interactor`=82 (11%), `Volatile Specialist`=49 (6%), `Workhorse`=33 (4%), `Noise`=24 (3%) | `Complex Driver`=195 (25%), `Nonlinear Driver`=130 (17%), `Catalyst`=118 (15%), `Volatile Specialist`=115 (15%), `Stable Contributor`=95 (12%), `Hidden Interactor`=73 (10%), `Noise`=28 (4%), `Workhorse`=14 (2%) |

### Diagnostic findings

- **Baseline only:** _(none)_
- **With-improvements only:** `co_sensitivity`
- **Shared:** `capacity`, `data_leakage`

Unique with-improvements findings (headline):
- `co_sensitivity` (info): Co-Sensitivity refused to recommend any prune (2 groups found, but no prune-safe group)

### Final-checkpoint interaction range

| Statistic | Baseline | With improvements |
|---|---|---|
| min | 139.6 | 0.04402 |
| mean | 496.6 | 0.2414 |
| max | 596.2 | 1.265 |

### Top-5 feature overlap

- **Jaccard:** 0.00 (0 / 10)
- **Baseline top-5:** `h0_d54`, `h1_d16`, `h9_d38`, `h10_d16`, `h8_d1`
- **With-improvements top-5:** `h6_d9`, `h10_d0`, `h4_d32`, `h5_d6`, `h9_d32`
- **Only in baseline:** `h0_d54`, `h10_d16`, `h1_d16`, `h8_d1`, `h9_d38`
- **Only in with-improvements:** `h10_d0`, `h4_d32`, `h5_d6`, `h6_d9`, `h9_d32`
