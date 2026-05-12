# Quick start

Three end-to-end demos in increasing complexity.

## 1. Tabular MLP (≈30 s on CPU)

```python
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

from ffca import FFCAReport, TabularAdapter, CheckpointLoader

data = load_breast_cancer()
X = StandardScaler().fit_transform(data.data)
loader = DataLoader(
    TensorDataset(torch.FloatTensor(X), torch.LongTensor(data.target)),
    batch_size=32,
)

def make_model():
    return nn.Sequential(nn.Linear(30, 64), nn.ReLU(),
                         nn.Linear(64, 32), nn.ReLU(),
                         nn.Linear(32, 2))

model = make_model()
# … train and save 4 checkpoints to ep01.pt, ep05.pt, ep15.pt, ep30.pt …

ck = CheckpointLoader(make_model, ["ep01.pt", "ep05.pt", "ep15.pt", "ep30.pt"])
adapter = TabularAdapter(make_model(), feature_names=list(data.feature_names))
report = FFCAReport(adapter, loader).run(checkpoints=ck)
report.save("out/")
```

Outputs land in `out/`:
- `report.md` — top-10 features, archetype distribution, Trust Score
  decisions, Co-Sensitivity groups
- `report.json` — all numbers
- `plots/` — 10 PNGs

## 2. Image CNN at the pixel level

```python
from ffca import FFCAReport, PixelAdapter

adapter = PixelAdapter(cnn_model, input_shape=(3, 32, 32))
report = FFCAReport(adapter, val_loader).run()
report.save("out/")

# Foreground / Background ratio diagnostic
fbr = adapter.fbr(report.signatures[-1].interaction)
print(f"FBR = {fbr:.3f}  ({'shortcut risk' if fbr < 0.5 else 'OK'})")
```

Plots include `06_fbr_diagnostic.png` (foreground vs background bar) and
`05_pixel_interaction_map.png` (heatmap reshaped to H×W).

## 3. CNN at an intermediate layer (the splice trick)

```python
from ffca import FFCAReport, ChannelAdapter

# Inspect a specific conv layer's channels
adapter = ChannelAdapter(cnn_model, layer_name="layer4.2.conv2")
report = FFCAReport(adapter, val_loader).run()
report.save("out/")
```

`ChannelAdapter` registers a forward hook to (a) capture the activation
at the chosen layer and (b) re-inject a leaf tensor so subsequent layers
can be differentiated. The hook is re-resolved on every forward, so
it survives the package's automatic ReLU→Softplus smoothing.

## Adjusting compute budget

Defaults are conservative; bump them up for production reports:

```python
report = FFCAReport(
    adapter, loader,
    n_first_order_samples=128,   # samples for Impact / Volatility
    n_hessian_samples=32,         # samples for Non-linearity (diag H)
    n_diag_probes=80,             # Hutchinson probes per sample
    n_cauchy_probes=200,          # Cauchy probes for Interaction
    n_cauchy_samples=32,
    n_cosens_permutations=200,    # for Co-Sensitivity guardrails
    n_cosens_bootstrap=50,
)
```

## CLI

The `ffca-report` command wraps all of the above:

```bash
ffca-report --help

# Tabular
ffca-report --model-class my_pkg.models:MyMLP --weights ckpt/final.pt \
    --adapter tabular --data data.csv --out reports/

# Image (pixel level)
ffca-report --model-class my_pkg.models:CNN --weights ckpt/final.pt \
    --adapter pixel --data data/val/ --image-size 32 --out reports/

# Channel level + checkpoints
ffca-report --model-class my_pkg.models:CNN \
    --checkpoints ckpt/e1.pt ckpt/e10.pt ckpt/final.pt \
    --adapter channel --layer features.5 \
    --data data/val/ --out reports/
```
