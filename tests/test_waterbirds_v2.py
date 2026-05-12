"""
Waterbirds stress test — applies audit-v2 improvements to a small CNN trained
on the Waterbirds shortcut-learning benchmark. Uses local data at
.../FFCA_dump/.../CNN_test/data/waterbirds_v1.0/.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ffca.improvements import CauchyHVP, TrustScore, CoSensitivityGroups

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")
DATA = Path("/Users/hnaja002/Documents/projects/FFCA/FFCA_dump/"
            "FFCA_archetype_dynamic/claude_playground/CNN_test/data/waterbirds_v1.0")
OUT = Path(__file__).resolve().parents[1] / "experiments" / "waterbirds"
OUT.mkdir(parents=True, exist_ok=True)
print(f"Device: {DEVICE}")

IMG = 64
BATCH = 32
EPOCHS = 5
LR = 1e-3
FFCA_N = 16


class WB(Dataset):
    def __init__(self, root, split, tfm):
        df = pd.read_csv(root / "metadata.csv")
        df = df[df['split'] == {'train': 0, 'val': 1, 'test': 2}[split]].reset_index(drop=True)
        self.root, self.df, self.tfm = root, df, tfm

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(self.root / r['img_filename']).convert('RGB')
        return self.tfm(img), int(r['y']), int(r['y'] * 2 + r['place'])


class WBCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.ReLU()
        self.f = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), self.act, nn.AvgPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), self.act, nn.AvgPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), self.act, nn.AvgPool2d(2),
        )
        self.c = nn.Sequential(
            nn.Linear(64 * (IMG // 8) ** 2, 128), self.act, nn.Linear(128, 2)
        )
        self._smooth = False

    def smooth(self, on):
        new = nn.Softplus(beta=10.0) if on else nn.ReLU()
        # Rebuild with new activation
        def rep(m):
            for n, c in m.named_children():
                if isinstance(c, (nn.ReLU, nn.Softplus)):
                    setattr(m, n, new)
                else:
                    rep(c)
        rep(self)
        self._smooth = on

    def forward(self, x):
        x = self.f(x)
        return self.c(x.view(x.size(0), -1))


def main():
    tfm = T.Compose([T.Resize((IMG, IMG)), T.ToTensor(),
                     T.Normalize((0.5,) * 3, (0.5,) * 3)])
    train_loader = DataLoader(WB(DATA, 'train', tfm), batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader = DataLoader(WB(DATA, 'val', tfm), batch_size=BATCH, num_workers=0)

    model = WBCNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    sigs = {}
    history = []
    rng = np.random.default_rng(0)

    for ep in range(EPOCHS):
        model.train()
        for img, y, _ in train_loader:
            img, y = img.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(img), y).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            tot, corr, by_grp = 0, 0, {0: [0, 0], 1: [0, 0], 2: [0, 0], 3: [0, 0]}
            for img, y, g in val_loader:
                img, y, g = img.to(DEVICE), y.to(DEVICE), g.to(DEVICE)
                pred = model(img).argmax(1)
                corr += (pred == y).sum().item(); tot += y.size(0)
                for gv in by_grp:
                    m = g == gv
                    by_grp[gv][0] += ((pred == y) & m).sum().item()
                    by_grp[gv][1] += m.sum().item()
            acc = corr / max(tot, 1)
            grp_acc = {gv: by_grp[gv][0] / max(by_grp[gv][1], 1) for gv in by_grp}
        history.append({'epoch': ep + 1, 'acc': acc, 'group_acc': grp_acc})
        print(f"  Epoch {ep+1}: acc={acc:.3f}  groups={ {k: f'{v:.2f}' for k,v in grp_acc.items()} }")

        # FFCA snapshot at every epoch (small enough)
        model.smooth(True)
        img_batch, _, _ = next(iter(val_loader))
        x = img_batch[:FFCA_N].to(DEVICE).clone().requires_grad_(True)
        out = model(x)
        s = out.gather(1, out.argmax(1, keepdim=True)).sum()
        g = torch.autograd.grad(s, x)[0].detach().cpu().numpy()
        g_flat = g.reshape(g.shape[0], -1)
        impact = np.abs(g_flat).mean(axis=0)
        volatility = g_flat.var(axis=0)
        # Cheap diag: |g|^2/n as rough nonlinearity proxy
        nonlinearity = (g_flat ** 2).mean(axis=0)
        # Interaction: pixel-pixel correlation row-sum
        if g_flat.shape[1] <= 5000:
            corr_mat = np.corrcoef(g_flat.T)
            corr_mat = np.nan_to_num(corr_mat, nan=0.0)
            interaction = np.abs(corr_mat).sum(axis=1) - 1.0
        else:
            interaction = np.zeros_like(impact)
        sigs[str(ep + 1)] = {'impact': impact, 'volatility': volatility,
                             'nonlinearity': nonlinearity, 'interaction': interaction,
                             'gradients': g_flat}
        model.smooth(False)

    # ---- #1 Real Cauchy-HVP on pixel inputs at final checkpoint
    print("\n#1 Cauchy-HVP — REAL HVP on pixels (d = 3*64*64 = 12288)")
    cauchy = CauchyHVP(n_probes=80)
    model.smooth(True)
    img_batch, _, _ = next(iter(val_loader))
    x_real = img_batch[:8].to(DEVICE)  # 8 samples, B=80 probes — ~80 backwards per sample
    t0 = time.time()
    inter, ci = cauchy.estimate_from_model(model, x_real, batch_chunk=2)
    t_hvp = time.time() - t0
    model.smooth(False)

    d = inter.size
    full_cost_seconds = t_hvp * (d / cauchy.n_probes)  # back-of-envelope
    print(f"  Real HVP: {t_hvp:.2f}s on d={d}  →  full-Hessian extrapolation ≈ {full_cost_seconds:.0f}s")
    print(f"  Mean ||H_i:||_1 = {cauchy.results['row_l1'].mean():.4f}, "
          f"mean |H_ii| = {cauchy.results['diag_abs'].mean():.4f}")
    print(f"  Significant CIs: {(ci[:, 0] > 0).sum()}/{d}")

    # Foreground / background ratio at native 64x64
    fg_mask = np.zeros((3, IMG, IMG), dtype=bool)
    pad = IMG // 4
    fg_mask[:, pad:-pad, pad:-pad] = True
    fg_mask_flat = fg_mask.reshape(-1)
    fg = inter[fg_mask_flat].mean()
    bg = inter[~fg_mask_flat].mean()
    fbr = fg / (fg + bg) if (fg + bg) > 0 else float('nan')
    print(f"  Foreground-vs-background interaction ratio: FBR={fbr:.3f}  "
          f"(<0.5 → background-shortcut risk)")

    # ---- #5 Trust Score on per-pixel signatures across epochs
    print("\n#5 Trust Score — weighted entropy across 5 epochs")
    ts = TrustScore()
    feat_names = [f"px_{i}" for i in range(d)]
    sigs_list = [{k: v for k, v in s.items() if k != 'gradients'} for s in sigs.values()]
    trust = ts.compute(sigs_list, feat_names)
    summary = ts.summary()
    total = sum(summary.values())
    for dec, n in sorted(summary.items(), key=lambda x: -x[1]):
        print(f"  {dec}: {n} ({n/total:.1%})")

    # ---- #6 Co-Sensitivity (pixel gradients are huge — subsample)
    print("\n#6 Co-Sensitivity — gradient k-medoids on top-2048 pixels by impact")
    top_k = 2048
    top_idx = np.argsort(sigs[str(EPOCHS)]['impact'])[-top_k:]
    grads = sigs[str(EPOCHS)]['gradients'][:, top_idx]
    cs = CoSensitivityGroups(n_permutations=20, n_bootstrap=10)
    groups = cs.compute(
        gradients=grads,
        impact=sigs[str(EPOCHS)]['impact'][top_idx],
        volatility=sigs[str(EPOCHS)]['volatility'][top_idx],
        nonlinearity=sigs[str(EPOCHS)]['nonlinearity'][top_idx],
        interaction=inter[top_idx],
    )
    print(f"  Top-{top_k} pixels  k={cs.diagnostics['k']}  "
          f"silhouette={cs.diagnostics['silhouette_observed']:.3f}  "
          f"perm-p={cs.diagnostics['permutation_p']:.3f}  "
          f"ARI={cs.diagnostics['bootstrap_ari_median']:.3f}  "
          f"abort={cs.diagnostics['abort_recommended']}")
    for gid in sorted(groups):
        g = groups[gid]
        flag = " ← PRUNE" if g['recommendation'].startswith('PRUNE') else ""
        print(f"   g{gid}: size={g['size']:>4} NC={g['nc_fraction']:>5.1%} "
              f"{g['recommendation']}{flag}")

    payload = {
        'history': history,
        'cauchy_hvp': {
            'method': cauchy.results['method'],
            'n_probes': cauchy.results['n_probes'],
            'n_samples': cauchy.results['n_samples'],
            'd_pixels': int(d),
            'real_hvp_time_seconds': t_hvp,
            'mean_row_l1': float(cauchy.results['row_l1'].mean()),
            'mean_diag_abs': float(cauchy.results['diag_abs'].mean()),
            'fbr': float(fbr),
        },
        'trust_summary': dict(summary),
        'co_sensitivity': {
            'diagnostics': {k: (None if isinstance(v, float) and np.isnan(v) else v)
                            for k, v in cs.diagnostics.items()},
            'groups': {int(k): {kk: vv for kk, vv in v.items() if kk != 'channels'}
                       for k, v in groups.items()},
        },
    }
    (OUT / "waterbirds_v2.json").write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nSaved → {OUT/'waterbirds_v2.json'}")


if __name__ == "__main__":
    main()
