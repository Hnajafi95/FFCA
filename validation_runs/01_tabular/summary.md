# Validation summary: `01_tabular`

_1 baseline / with-improvements pair(s) found._

## Pair: `01_tabular`

- **Baseline:** `baseline/`
- **With improvements:** `with_improvements/`
- **Checkpoints:** ['ep1', 'ep5', 'ep15', 'ep30']
- **Features:** 30

### Archetype distribution per checkpoint

| Checkpoint | Baseline | With improvements |
|---|---|---|
| ep1 | `Complex Driver`=11 (37%), `Catalyst`=8 (27%), `Stable Contributor`=5 (17%), `Nonlinear Driver`=4 (13%), `Noise`=1 (3%), `Volatile Specialist`=1 (3%) | `Complex Driver`=10 (33%), `Catalyst`=7 (23%), `Nonlinear Driver`=7 (23%), `Volatile Specialist`=2 (7%), `Stable Contributor`=2 (7%), `Noise`=1 (3%), `Hidden Interactor`=1 (3%) |
| ep5 | `Complex Driver`=11 (37%), `Catalyst`=8 (27%), `Nonlinear Driver`=5 (17%), `Volatile Specialist`=2 (7%), `Stable Contributor`=2 (7%), `Noise`=2 (7%) | `Complex Driver`=11 (37%), `Catalyst`=7 (23%), `Nonlinear Driver`=6 (20%), `Volatile Specialist`=2 (7%), `Stable Contributor`=2 (7%), `Noise`=2 (7%) |
| ep15 | `Complex Driver`=9 (30%), `Nonlinear Driver`=7 (23%), `Catalyst`=7 (23%), `Noise`=3 (10%), `Stable Contributor`=2 (7%), `Volatile Specialist`=1 (3%), `Hidden Interactor`=1 (3%) | `Complex Driver`=10 (33%), `Catalyst`=8 (27%), `Nonlinear Driver`=5 (17%), `Stable Contributor`=4 (13%), `Noise`=3 (10%) |
| ep30 | `Complex Driver`=10 (33%), `Catalyst`=8 (27%), `Nonlinear Driver`=6 (20%), `Stable Contributor`=3 (10%), `Noise`=3 (10%) | `Complex Driver`=8 (27%), `Catalyst`=7 (23%), `Stable Contributor`=5 (17%), `Nonlinear Driver`=4 (13%), `Noise`=4 (13%), `Hidden Interactor`=1 (3%), `Volatile Specialist`=1 (3%) |

### Diagnostic findings

- **Baseline only:** _(none)_
- **With-improvements only:** `co_sensitivity`, `trust_instability`, `trust_keep_recommended`
- **Shared:** `capacity`, `overfitting`

Unique with-improvements findings (headline):
- `co_sensitivity` (info): Co-Sensitivity refused to recommend any prune (2 groups found, but no prune-safe group)
- `trust_instability` (warn): 17/30 features are unstable across checkpoints
- `trust_keep_recommended` (info): 6 features are stably important across all checkpoints

### Final-checkpoint interaction range

| Statistic | Baseline | With improvements |
|---|---|---|
| min | 2.913 | 0.3209 |
| mean | 23.34 | 0.5517 |
| max | 25.86 | 0.9017 |

### Top-5 feature overlap

- **Jaccard:** 0.25 (2 / 8)
- **Baseline top-5:** `worst smoothness`, `worst radius`, `worst area`, `worst perimeter`, `mean concave points`
- **With-improvements top-5:** `worst smoothness`, `mean texture`, `worst texture`, `worst perimeter`, `texture error`
- **Only in baseline:** `mean concave points`, `worst area`, `worst radius`
- **Only in with-improvements:** `mean texture`, `texture error`, `worst texture`
