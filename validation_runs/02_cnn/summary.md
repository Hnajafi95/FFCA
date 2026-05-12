# Validation summary: `02_cnn`

_2 baseline / with-improvements pair(s) found._

## Pair: `channel`

- **Baseline:** `channel_baseline/`
- **With improvements:** `channel_with/`
- **Checkpoints:** ['ep1', 'ep3', 'ep6', 'ep8']
- **Features:** 128

### Archetype distribution per checkpoint

| Checkpoint | Baseline | With improvements |
|---|---|---|
| ep1 | `Complex Driver`=33 (26%), `Nonlinear Driver`=30 (23%), `Catalyst`=22 (17%), `Stable Contributor`=12 (9%), `Volatile Specialist`=11 (9%), `Noise`=11 (9%), `Hidden Interactor`=9 (7%) | `Complex Driver`=30 (23%), `Catalyst`=27 (21%), `Noise`=21 (16%), `Nonlinear Driver`=20 (16%), `Volatile Specialist`=13 (10%), `Stable Contributor`=12 (9%), `Hidden Interactor`=5 (4%) |
| ep3 | `Complex Driver`=36 (28%), `Nonlinear Driver`=29 (23%), `Catalyst`=25 (20%), `Noise`=12 (9%), `Stable Contributor`=11 (9%), `Volatile Specialist`=8 (6%), `Hidden Interactor`=7 (5%) | `Complex Driver`=33 (26%), `Catalyst`=28 (22%), `Noise`=21 (16%), `Stable Contributor`=15 (12%), `Nonlinear Driver`=14 (11%), `Volatile Specialist`=13 (10%), `Hidden Interactor`=4 (3%) |
| ep6 | `Complex Driver`=36 (28%), `Nonlinear Driver`=27 (21%), `Catalyst`=23 (18%), `Volatile Specialist`=13 (10%), `Stable Contributor`=12 (9%), `Hidden Interactor`=9 (7%), `Noise`=8 (6%) | `Complex Driver`=31 (24%), `Catalyst`=28 (22%), `Noise`=21 (16%), `Volatile Specialist`=16 (12%), `Stable Contributor`=14 (11%), `Nonlinear Driver`=14 (11%), `Hidden Interactor`=4 (3%) |
| ep8 | `Complex Driver`=40 (31%), `Catalyst`=25 (20%), `Nonlinear Driver`=21 (16%), `Stable Contributor`=18 (14%), `Volatile Specialist`=8 (6%), `Noise`=8 (6%), `Hidden Interactor`=7 (5%), `Workhorse`=1 (1%) | `Complex Driver`=33 (26%), `Catalyst`=25 (20%), `Stable Contributor`=21 (16%), `Noise`=19 (15%), `Nonlinear Driver`=15 (12%), `Volatile Specialist`=9 (7%), `Hidden Interactor`=6 (5%) |

### Diagnostic findings

- **Baseline only:** _(none)_
- **With-improvements only:** `co_sensitivity`, `trust_instability`, `trust_keep_recommended`, `trust_prune_recommended`
- **Shared:** `capacity`, `overfitting`

Unique with-improvements findings (headline):
- `co_sensitivity` (info): Co-Sensitivity refused to recommend any prune (2 groups found, but no prune-safe group)
- `trust_instability` (warn): 66/128 features are unstable across checkpoints
- `trust_keep_recommended` (info): 21 features are stably important across all checkpoints
- `trust_prune_recommended` (info): 12 features are confidently Noise across all checkpoints

### Final-checkpoint interaction range

| Statistic | Baseline | With improvements |
|---|---|---|
| min | 23.88 | 2.727 |
| mean | 50.94 | 31.25 |
| max | 71.73 | 63.91 |

### Top-5 feature overlap

- **Jaccard:** 0.25 (2 / 8)
- **Baseline top-5:** `ch_72`, `ch_36`, `ch_44`, `ch_70`, `ch_13`
- **With-improvements top-5:** `ch_36`, `ch_116`, `ch_72`, `ch_71`, `ch_43`
- **Only in baseline:** `ch_13`, `ch_44`, `ch_70`
- **Only in with-improvements:** `ch_116`, `ch_43`, `ch_71`

## Pair: `pixel`

- **Baseline:** `pixel_baseline/`
- **With improvements:** `pixel_with/`
- **Checkpoints:** ['ep1', 'ep3', 'ep6', 'ep8']
- **Features:** 3072

### Archetype distribution per checkpoint

| Checkpoint | Baseline | With improvements |
|---|---|---|
| ep1 | `Complex Driver`=927 (30%), `Nonlinear Driver`=664 (22%), `Catalyst`=664 (22%), `Noise`=251 (8%), `Stable Contributor`=251 (8%), `Volatile Specialist`=212 (7%), `Hidden Interactor`=103 (3%) | `Complex Driver`=766 (25%), `Catalyst`=677 (22%), `Nonlinear Driver`=518 (17%), `Noise`=470 (15%), `Stable Contributor`=293 (10%), `Volatile Specialist`=258 (8%), `Hidden Interactor`=90 (3%) |
| ep3 | `Complex Driver`=850 (28%), `Nonlinear Driver`=743 (24%), `Catalyst`=427 (14%), `Hidden Interactor`=340 (11%), `Stable Contributor`=321 (10%), `Volatile Specialist`=270 (9%), `Noise`=121 (4%) | `Complex Driver`=851 (28%), `Catalyst`=678 (22%), `Nonlinear Driver`=411 (13%), `Noise`=400 (13%), `Stable Contributor`=388 (13%), `Volatile Specialist`=254 (8%), `Hidden Interactor`=90 (3%) |
| ep6 | `Complex Driver`=898 (29%), `Nonlinear Driver`=683 (22%), `Catalyst`=442 (14%), `Stable Contributor`=342 (11%), `Hidden Interactor`=326 (11%), `Volatile Specialist`=266 (9%), `Noise`=115 (4%) | `Complex Driver`=868 (28%), `Catalyst`=710 (23%), `Nonlinear Driver`=449 (15%), `Noise`=394 (13%), `Stable Contributor`=374 (12%), `Volatile Specialist`=219 (7%), `Hidden Interactor`=58 (2%) |
| ep8 | `Complex Driver`=812 (26%), `Nonlinear Driver`=694 (23%), `Catalyst`=415 (14%), `Hidden Interactor`=353 (11%), `Stable Contributor`=350 (11%), `Volatile Specialist`=336 (11%), `Noise`=112 (4%) | `Complex Driver`=816 (27%), `Catalyst`=681 (22%), `Nonlinear Driver`=497 (16%), `Noise`=369 (12%), `Stable Contributor`=365 (12%), `Volatile Specialist`=258 (8%), `Hidden Interactor`=86 (3%) |

### Diagnostic findings

- **Baseline only:** _(none)_
- **With-improvements only:** `co_sensitivity`, `trust_instability`, `trust_keep_recommended`, `trust_prune_recommended`
- **Shared:** `capacity`, `overfitting`, `shortcut_learning`

Unique with-improvements findings (headline):
- `co_sensitivity` (info): Co-Sensitivity refused to recommend any prune (7 groups found, but no prune-safe group)
- `trust_instability` (warn): 1611/3072 features are unstable across checkpoints
- `trust_keep_recommended` (info): 586 features are stably important across all checkpoints
- `trust_prune_recommended` (info): 194 features are confidently Noise across all checkpoints

### Final-checkpoint interaction range

| Statistic | Baseline | With improvements |
|---|---|---|
| min | 511.4 | 0.08006 |
| mean | 708.1 | 0.4553 |
| max | 923.2 | 1.201 |

### Top-5 feature overlap

- **Jaccard:** 0.00 (0 / 10)
- **Baseline top-5:** `px_2_11_14`, `px_2_30_3`, `px_1_0_21`, `px_2_30_2`, `px_2_12_13`
- **With-improvements top-5:** `px_1_25_23`, `px_1_25_20`, `px_0_22_20`, `px_1_17_15`, `px_1_24_22`
- **Only in baseline:** `px_1_0_21`, `px_2_11_14`, `px_2_12_13`, `px_2_30_2`, `px_2_30_3`
- **Only in with-improvements:** `px_0_22_20`, `px_1_17_15`, `px_1_24_22`, `px_1_25_20`, `px_1_25_23`
