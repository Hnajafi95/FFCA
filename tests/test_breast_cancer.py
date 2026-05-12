"""
Test FFCA Report package on Breast Cancer Wisconsin dataset.
Isolates the effect of each of the 3 improvements through ablation.
"""
import torch, torch.nn as nn, numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import sys, json, time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ffca.analyzer import FFCAReport
from ffca.improvements import CauchyHVP, TrustScore, CoSensitivityGroups

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "breast_cancer"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# =============================================================================
# DATA & MODEL
# =============================================================================
print("Loading Breast Cancer Wisconsin dataset...")
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = list(data.feature_names)

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

class BreastCancerMLP(nn.Module):
    def __init__(self, n_features=30, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 2)
        )
    def forward(self, x): return self.net(x)

# =============================================================================
# BASELINE: FFCA without improvements
# =============================================================================
print("\n" + "="*70)
print("BASELINE: Standard FFCA (no improvements)")
print("="*70)
model_base = BreastCancerMLP(n_features=30)
report_base = FFCAReport(model_base, train_loader, val_loader,
                         feature_names=feature_names, device='cpu')
t0 = time.time()
report_base.run(n_epochs=30, checkpoint_every=5, n_ffca_samples=50,
                enable_improvements=False)
baseline_time = time.time() - t0

# Manually apply improvements post-hoc for comparison
last_sig = report_base.checkpoint_signatures[-1]
all_sigs = report_base.checkpoint_signatures

# =============================================================================
# IMPROVEMENT #1: Cauchy-HVP
# =============================================================================
print("\n" + "="*70)
print("IMPROVEMENT #1: Cauchy-HVP Interaction Estimation (REAL HVP)")
print("="*70)
cauchy = CauchyHVP(n_probes=100)
# Smooth activations on the *trained* model
report_base._enable_smooth()
sample_x = next(iter(val_loader))[0][:16]
t_cauchy = time.time()
cauchy_interactions, cauchy_cis = cauchy.estimate_from_model(model_base, sample_x)
cauchy_time = time.time() - t_cauchy
report_base._restore_relu()

n_sig = np.sum(cauchy_cis[:, 0] > 0)
print(f"  Method: {cauchy.results['method']}, B={cauchy.results['n_probes']}, "
      f"samples={cauchy.results['n_samples']}, time={cauchy_time:.2f}s")
print(f"  Features with significant interaction (CI lower > 0): {n_sig}/{len(cauchy_interactions)}")
print(f"  Operations ratio (full Hessian / Cauchy-HVP): {cauchy.speedup_factor(30):.2f}x")
print(f"  Mean ||H_i:||_1 = {cauchy.results['row_l1'].mean():.4f},  "
      f"mean |H_ii| = {cauchy.results['diag_abs'].mean():.4f}")
top5 = np.argsort(cauchy_interactions)[-5:][::-1]
for rank, idx in enumerate(top5, 1):
    print(f"  #{rank}: {feature_names[idx]} — I={cauchy_interactions[idx]:.4f} "
          f"[{cauchy_cis[idx,0]:.4f}, {cauchy_cis[idx,1]:.4f}]")

# =============================================================================
# IMPROVEMENT #5: Trust Score
# =============================================================================
print("\n" + "="*70)
print("IMPROVEMENT #5: Temporal Stability Trust Score")
print("="*70)
trust = TrustScore()
trust_scores = trust.compute(all_sigs, feature_names)
trust_summary = trust.summary()
total = sum(trust_summary.values())
for dec, count in sorted(trust_summary.items(), key=lambda x: -x[1]):
    print(f"  {dec}: {count} features ({count/total:.1%})")

# Show interesting features
prunable = [(k, v) for k, v in trust_scores.items() if 'PRUNE' in v['decision']]
investigate = [(k, v) for k, v in trust_scores.items() if 'INVESTIGATE' in v['decision']]
keep = [(k, v) for k, v in trust_scores.items() if 'CONFIDENTLY KEEP' in v['decision']]

print(f"\n  Prunable ({len(prunable)}): {[p[0] for p in prunable[:5]]}")
print(f"  Investigate ({len(investigate)}): {[i[0] for i in investigate[:3]]}")
print(f"  High-confidence keep ({len(keep)}): {[k[0] for k in keep[:5]]}")

if prunable:
    worst = prunable[0]
    print(f"\n  Example prunable: {worst[0]} — always {worst[1]['dominant_archetype']}, "
          f"stability={worst[1]['stability']:.3f}")

# =============================================================================
# IMPROVEMENT #6: Co-Sensitivity Groups
# =============================================================================
print("\n" + "="*70)
print("IMPROVEMENT #6: Co-Sensitivity Functional Groups")
print("="*70)
cosens = CoSensitivityGroups(n_permutations=100, n_bootstrap=30)
# Get raw gradients on the trained model
report_base._enable_smooth()
grads = report_base._gather_gradients(sample_x)
report_base._restore_relu()
groups = cosens.compute(
    gradients=grads,
    impact=last_sig['impact'], volatility=last_sig['volatility'],
    nonlinearity=last_sig['nonlinearity'], interaction=cauchy_interactions,
)
cosens_summary = cosens.summary()
print(f"  Diagnostics: k={cosens.diagnostics['k']}, "
      f"silhouette={cosens.diagnostics['silhouette_observed']:.3f}, "
      f"perm-p={cosens.diagnostics['permutation_p']:.3f}, "
      f"bootstrap-ARI={cosens.diagnostics['bootstrap_ari_median']:.3f}, "
      f"abort={cosens.diagnostics['abort_recommended']}")

for gid in sorted(groups.keys()):
    g = groups[gid]
    flag = " ← PRUNE" if "PRUNE" in g['recommendation'] else ""
    print(f"  Group {gid}: {g['size']:2d} features | NC={g['nc_fraction']:.1%} | "
          f"I={g['mean_impact']:.4f} | X={g['mean_interaction']:.1f} | "
          f"{g['recommendation']}{flag}")

print(f"\n  Pruning potential: {cosens_summary['prunable']}/{cosens_summary['total_features']} "
      f"({cosens_summary['prunable']/max(cosens_summary['total_features'],1):.1%})")

# =============================================================================
# FULL REPORT (all improvements enabled)
# =============================================================================
print("\n" + "="*70)
print("FULL PIPELINE: All improvements enabled")
print("="*70)
model_full = BreastCancerMLP(n_features=30)
report_full = FFCAReport(model_full, train_loader, val_loader,
                         feature_names=feature_names, device='cpu')
t0 = time.time()
report_full.run(n_epochs=30, checkpoint_every=5, n_ffca_samples=50,
                enable_improvements=True)
full_time = time.time() - t0

report_path = report_full.save(str(OUTPUT_DIR / "breast_cancer_ffca_report.md"))

# =============================================================================
# ABLATION: Isolate effect of each improvement
# =============================================================================
print("\n" + "="*70)
print("ABLATION STUDY: Isolating Each Improvement's Effect")
print("="*70)

# Re-use the same trained model for the ablation rather than re-training,
# since correctness of each improvement is what's under test here.
abl1_sigs = all_sigs

# Ablation #1: real Cauchy-HVP only — already computed above
c1_sig_count = n_sig
c1_int = cauchy_interactions
print(f"\n--- Ablation: Cauchy-HVP Only ---")
print(f"  Significant interactions detected: {c1_sig_count}/30")

# Ablation #5: real Trust Score only
print("\n--- Ablation: Trust Score Only ---")
t5 = TrustScore()
t5_scores = t5.compute(abl1_sigs, feature_names)
t5_prunable = sum(1 for v in t5_scores.values() if 'PRUNE' in v['decision'])
t5_investigate = sum(1 for v in t5_scores.values() if 'INVESTIGATE' in v['decision'])
print(f"  Prunable: {t5_prunable}, Investigate: {t5_investigate}")

# Ablation #6: real Co-Sensitivity only — uses gradients, with guardrails
print("\n--- Ablation: Co-Sensitivity Only ---")
cs6 = CoSensitivityGroups(n_permutations=100, n_bootstrap=30)
cs6_groups = cs6.compute(
    gradients=grads,
    impact=abl1_sigs[-1]['impact'], volatility=abl1_sigs[-1]['volatility'],
    nonlinearity=abl1_sigs[-1]['nonlinearity'], interaction=c1_int,
)
cs6_prunable = sum(1 for g in cs6_groups.values() if 'PRUNE' in g['recommendation'])
print(f"  Groups: {len(cs6_groups)}, Prunable groups: {cs6_prunable}, "
      f"perm-p={cs6.diagnostics['permutation_p']:.3f}, "
      f"ARI={cs6.diagnostics['bootstrap_ari_median']:.3f}")

# =============================================================================
# SAVE ABLATION RESULTS
# =============================================================================
ablation = {
    'dataset': 'breast_cancer_wisconsin',
    'n_features': 30,
    'n_samples': len(X_train) + len(X_val),
    'feature_names': feature_names,
    'baseline': {
        'final_val_acc': float(report_base.metrics_history['val_acc'][-1]),
        'time_seconds': baseline_time,
    },
    'full_pipeline': {
        'final_val_acc': float(report_full.metrics_history['val_acc'][-1]),
        'time_seconds': full_time,
        'trust_summary': report_full.trust_summary,
        'co_sens_summary': report_full.co_sens_summary,
    },
    'improvement_1_cauchy_hvp': {
        'n_significant_interactions': c1_sig_count,
        'speedup_vs_full_hessian': cauchy.speedup_factor(30),
        'top_interactions': [(feature_names[i], float(cauchy_interactions[i]))
                            for i in np.argsort(cauchy_interactions)[-5:][::-1]],
    },
    'improvement_5_trust_score': {
        'prunable_count': t5_prunable,
        'investigate_count': t5_investigate,
        'prunable_features': [k for k, v in t5_scores.items() if 'PRUNE' in v['decision']],
    },
    'improvement_6_co_sensitivity': {
        'n_groups': len(cs6_groups),
        'prunable_groups': cs6_prunable,
        'group_summary': {str(k): v for k, v in cs6_groups.items()},
    },
}

with open(OUTPUT_DIR / 'ablation_results.json', 'w') as f:
    json.dump(ablation, f, indent=2, default=str)

print(f"\n{'='*70}")
print("RESULTS SAVED")
print(f"  Report: {report_path}")
print(f"  Ablation: {OUTPUT_DIR / 'ablation_results.json'}")
print(f"  Baseline time: {baseline_time:.1f}s")
print(f"  Full pipeline time: {full_time:.1f}s")
