"""
Re-run the three improvements on the saved SRDRN Phase 2 artifacts.

Phase 2.2: per-layer JSON (4D signatures) + npy (gradient correlation matrices).
Phase 2.4: dynamic JSON with 10 checkpoints × 6 climate features.

Cauchy-HVP here uses the *correlation proxy* path because we don't have the
trained SRDRN model checkpoint loaded — only its saved gradient-correlation
matrices. Trust Score and Co-Sensitivity use their full algorithms.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ffca.improvements import CauchyHVP, TrustScore, CoSensitivityGroups

PHASE2_ROOT = Path(
    "/Users/hnaja002/Documents/projects/FFCA/FFCA_dump/FFCA_archetype_dynamic/"
    "claude_playground/FFCA_PHASE2"
)
P22_DIR = PHASE2_ROOT / "phase_2.2" / "results"
P24_PATH = PHASE2_ROOT / "phase_2.4" / "results" / "dynamic_ffca_results_20260128_154442.json"

LAYERS = ['conv2d', 'conv2d_33', 'conv2d_34', 'conv2d_35']
TS_TAG = '20260128_090737'


def load_layer(layer: str):
    j = json.loads((P22_DIR / f"channel_ffca_{layer}_{TS_TAG}.json").read_text())
    corr = np.load(P22_DIR / f"channel_interactions_{layer}_{TS_TAG}.npy")
    return j, corr


def synthesize_gradients_from_corr(corr: np.ndarray, n_samples: int = 80, seed: int = 0):
    """Build a gradient matrix whose empirical correlation matches `corr`.

    For Co-Sensitivity's guardrails (permutation test, bootstrap ARI) we need
    actual gradient vectors. The exact ones are unavailable, but we can sample
    a Gaussian matrix with the prescribed correlation via Cholesky / eigendecomp.
    """
    d = corr.shape[0]
    # Symmetrise & PSD-clip
    corr = (corr + corr.T) / 2
    w, V = np.linalg.eigh(corr)
    w = np.clip(w, 1e-6, None)
    L = V @ np.diag(np.sqrt(w))
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_samples, d))
    return z @ L.T


def run_phase22():
    print("=" * 70)
    print("Phase 2.2 — channel-level FFCA on 4 SRDRN layers")
    print("=" * 70)
    summary = {}
    for layer in LAYERS:
        j, corr = load_layer(layer)
        impact = np.asarray(j['impact'])
        volatility = np.asarray(j['volatility'])
        nonlinearity = np.asarray(j['nonlinearity'])
        d = impact.size
        print(f"\n{layer} ({d} channels)")

        # #1 Cauchy-HVP via correlation proxy (we lack the live model)
        cauchy = CauchyHVP(n_probes=100)
        inter, ci = cauchy.estimate_from_corr(corr)
        n_sig = int((ci[:, 0] > 0).sum())
        print(f"  #1 correlation-proxy interaction: mean={inter.mean():.2f}, "
              f"sig CI>0: {n_sig}/{d}, theoretical-speedup-if-HVP: "
              f"{cauchy.speedup_factor(d):.1f}x")

        # #6 Co-Sensitivity — guardrails enabled, synthetic gradients with same corr
        grads_syn = synthesize_gradients_from_corr(corr, n_samples=80)
        cs = CoSensitivityGroups(n_permutations=50, n_bootstrap=20)
        groups = cs.compute(
            gradients=grads_syn,
            impact=impact, volatility=volatility,
            nonlinearity=nonlinearity, interaction=inter,
        )
        prunable = sum(g['size'] for g in groups.values()
                       if g['recommendation'].startswith('PRUNE'))
        print(f"  #6 k={cs.diagnostics['k']} groups, silhouette="
              f"{cs.diagnostics['silhouette_observed']:.3f}, "
              f"perm-p={cs.diagnostics['permutation_p']:.3f}, "
              f"ARI={cs.diagnostics['bootstrap_ari_median']:.3f}, "
              f"prunable={prunable}/{d}, abort={cs.diagnostics['abort_recommended']}")
        for gid, g in groups.items():
            print(f"     g{gid}: size={g['size']:>3} NC={g['nc_fraction']:>5.1%} "
                  f"medoid=ch_{g['medoid']:<3} {g['recommendation']}")

        summary[layer] = {
            'd': int(d),
            'cauchy_significant': n_sig,
            'cosens_groups': len(groups),
            'cosens_prunable': prunable,
            'cosens_diagnostics': cs.diagnostics,
        }
    return summary


def run_phase24():
    print("\n" + "=" * 70)
    print("Phase 2.4 — dynamic FFCA across 10 checkpoints × 6 climate features")
    print("=" * 70)
    data = json.loads(P24_PATH.read_text())
    feature_names = data['feature_names']
    # signatures is a dict keyed by epoch; produce list ordered by epoch
    sigs_dict = data['signatures']
    epochs = sorted(sigs_dict.keys(), key=int)
    sigs = [sigs_dict[e] for e in epochs]

    ts = TrustScore()
    trust = ts.compute(sigs, feature_names)
    print(f"\nFeature       Stability  Importance  Dominant            Decision")
    print("-" * 80)
    for name in feature_names:
        t = trust[name]
        print(f"{name:<12}  {t['stability']:>8.3f}  {t['importance']:>10.4f}  "
              f"{t['dominant_archetype']:<18}  {t['decision']}")

    # Sanity checks against the paper / handoff doc
    print("\nValidation vs. Phase 2.4 ground truth:")
    pr = trust['pr']
    tasmax = trust['tasmax']
    tasmin = trust['tasmin']
    print(f"  pr     (expect: stable Noise)  → stability={pr['stability']:.3f}, "
          f"dominant={pr['dominant_archetype']}  "
          f"{'PASS' if pr['stability']>=0.7 and pr['dominant_archetype']=='Noise' else 'CHECK'}")
    print(f"  tasmax (expect: stable Keep)   → stability={tasmax['stability']:.3f}, "
          f"dominant={tasmax['dominant_archetype']}  "
          f"{'PASS' if tasmax['stability']>=0.7 else 'CHECK'}")
    print(f"  tasmin (expect: unstable)      → stability={tasmin['stability']:.3f}, "
          f"n_unique={tasmin['n_unique_archetypes']}  "
          f"{'PASS' if tasmin['n_unique_archetypes']>=2 else 'CHECK'}")
    return trust


def main():
    p22 = run_phase22()
    p24 = run_phase24()
    out_dir = Path(__file__).resolve().parents[1] / "experiments" / "climate_phase2"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "srdrn_phase2_audit_v2.json").write_text(json.dumps({
        'phase_2_2': p22,
        'phase_2_4': {k: {kk: vv for kk, vv in v.items() if kk != 'archetype_sequence'}
                      for k, v in p24.items()},
    }, indent=2, default=str))
    print(f"\nSaved → {out_dir/'srdrn_phase2_audit_v2.json'}")


if __name__ == "__main__":
    main()
