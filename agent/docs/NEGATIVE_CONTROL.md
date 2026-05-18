# Negative control — bike_sharing healthy

**Question:** when we train a model with no engineered pathology, does the
v0.5 rulebook stay quiet?

**Setup (healthy, deliberately):**
- UCI Bike Sharing regression (12 features after categorical encoding).
- MLP `(64, 32)` with ReLU — moderate capacity (not the v0.5 case-study's
  inflated 512×256).
- 80 epochs, Adam lr=1e-3, no engineered weight decay or regularization
  schedule — i.e., a sensible default training run.
- 10 evenly-spaced checkpoints.

**Result:**

| Severity | Count | Rules                                 |
|----------|------:|---------------------------------------|
| critical | **0** | —                                     |
| warn     | 1     | `monitor_bucket_dominant`             |
| info     | 1     | `cosens_weak_clustering_significant`  |
| descriptor | 7   | archetype + concentration + convergence labels |

**Key numbers (well clear of all critical thresholds):**

| Signal                                  | Value | Critical thresholds |
|-----------------------------------------|------:|---------------------|
| val_train_gap                           | 0.020 | overfitting needs > 0.5 |
| nonlinearity_mean                       | 567.3 | insufficient_capacity needs < 1e-3 |
| feature.impact_dominance (max)          | ~3.0  | data_leakage needs > 5 (paired with saturation + small gap) |
| volatility spike ratio (epoch 9)        | ~1.2× | overfitting needs ≥ 1.4× |
| checkpoint_drift_l2_pct (last)          | < 5%  | late_drift needs > 20% |

**Verdict: PASS.** No critical-severity rule fires on a healthy training
run with no engineered pathology. The single warn-level finding
(`monitor_bucket_dominant`, "half the features are in the MONITOR trust
bucket") is honest — it's a descriptor about feature stability, not a
diagnostic. The agent's narration would correctly characterize this as a
healthy model.

## What this validates

1. **Rulebook is not "always firing."** The pathology detectors don't go
   off on a clean run — the v0.5 critical rules (`data_leakage_immediate_dominance`,
   `spurious_correlation_train_val_gap`, `overfitting_volatility_spike`,
   `insufficient_capacity`, `shortcut_learning_*`) all stayed silent.
2. **Per-rule null behavior matches per-rule positive behavior.** Together
   with the 5/6 v0.5 case-study passes, we now have evidence both ways:
   the rules fire when the phenomenon is present, and stay quiet when it
   isn't.
3. **Trust-bucket warnings on healthy models are honest.** The model has
   a real "MONITOR" cluster — features that aren't quite Confident Keep
   or Confident Prune — and the rule labels that condition correctly.
   This is informative, not a false positive.

## Caveats

- **This is n=1 healthy model on the same problem domain we engineered
  pathologies in.** The rigorous version is the next phase: 20 models
  from a public zoo, none engineered, see how often `critical` rules
  fire. Plan to do this after Phase 4 (paper).
- The negative control re-uses the same bike_sharing data we used for the
  v0.5 positive case. A second negative control on a different domain
  (e.g., a healthy MNIST classifier) would strengthen the result.

## Artifacts

- `bike_healthy/report.json` — full FFCA report
- `bike_healthy/history.json` — Keras-style training history
- `bike_healthy/findings_v05.json` — every rule that fired + signature stats
- `bike_healthy/plots/` — FFCA per-feature plots
- `bike_healthy/checkpoints/` — 10 PyTorch state-dicts
