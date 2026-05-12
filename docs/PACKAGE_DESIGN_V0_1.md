# FFCA Package — v0.1.0 design note

**Goal**: a pip-installable `ffca-report` package any PyTorch researcher can
point at their model + checkpoints and get the full FFCA 4-D signature,
8-archetype taxonomy, three audit-v2 improvements (Cauchy-HVP, Trust Score,
Co-Sensitivity), and a directory of plots/reports.

## Public surface area

```python
# Python API
from ffca import FFCAReport
from ffca.adapters import TabularAdapter, PixelAdapter, ChannelAdapter
from ffca.scalars import predicted_class, target_class, regression
from ffca.checkpoint import CheckpointLoader

adapter = ChannelAdapter(model, layer_name="layer4.2.conv2")
report = FFCAReport(adapter, val_loader)
report.run(checkpoints=CheckpointLoader(model_factory, ["e1.pt", "e5.pt"]))
report.save("out/")   # writes report.md, report.json, plots/*.png
```

```bash
# CLI surface (CLI-first per user spec)
ffca-report \
  --model-class my_module:MyResNet \
  --weights resnet50_final.pt \
  --adapter channel --layer layer4.2.conv2 \
  --data data/imagenet_val/ \
  --checkpoints checkpoints/e{1,10,50,160}.pt \
  --scalar predicted_class \
  --out reports/run-2026-05-10/
```

## Module layout (final, for v0.1.0)

```
ffca/
├── __init__.py             # re-exports FFCAReport + main adapters/scalars
├── core/
│   ├── __init__.py
│   ├── adapter.py          # FFCAModelAdapter ABC + Splice utility
│   ├── scalars.py          # predicted_class, target_class, regression, custom
│   ├── smoothing.py        # smooth() context manager (ReLU/LReLU/GELU/PReLU/ELU → Softplus)
│   ├── derivatives.py      # impact/volatility/diag-Hessian via autograd
│   ├── archetypes.py       # 8-archetype classifier + similarity matrix
│   └── signature.py        # FFCASignature dataclass
├── adapters/
│   ├── __init__.py
│   ├── tabular.py          # TabularAdapter
│   ├── pixel.py            # PixelAdapter (input-pixel FFCA, spatial shape preserved)
│   └── channel.py          # ChannelAdapter (the splice trick)
├── improvements/
│   ├── __init__.py
│   ├── cauchy_hvp.py       # ← from audit-v2 (already validated)
│   ├── trust_score.py      # ← from audit-v2
│   └── co_sensitivity.py   # ← from audit-v2
├── viz/
│   ├── __init__.py         # generate_all_plots()
│   ├── static.py           # radar, archetype dist, ranking, CI
│   ├── spatial.py          # pixel maps, channel grids, FBR
│   ├── dynamic.py          # evolution curves, ranking, trust scatter
│   └── diagnostics.py      # co-sens groups, group accuracy
├── checkpoint.py           # CheckpointLoader (plain state_dict for v0.1)
├── report.py               # FFCAReport orchestrator + markdown/JSON
└── cli.py                  # ffca-report entry point
```

## Out of scope for v0.1.0

- Lightning / HF / SafeTensors checkpoint formats → use plain `state_dict`
- Transformer / LLM / GNN adapters → v0.3 / v1.0
- Sphinx docs site → use markdown for v0.1, add Sphinx later
- Streaming median for d > 1M → not needed at this scale
- Auto-detection of model architecture → users must specify adapter + layer

## Universal-model recipe (the bit that scales)

A researcher with a model not in the built-in adapter list writes ~20 lines:

```python
from ffca.core import FFCAModelAdapter
from ffca import FFCAReport

class MyAdapter(FFCAModelAdapter):
    n_features = 768
    feature_names = None

    def feature_input(self, batch):
        return batch["embeddings"].clone().requires_grad_(True)

    def scalar_output(self, x):
        out = self.model(inputs_embeds=x)
        return out.logits[:, -1].max(dim=1).values.sum()

FFCAReport(MyAdapter(model), loader).run().save("out/")
```

That's the universality guarantee: anything model-specific is in the adapter;
everything else is shared.

## Acceptance criteria for v0.1.0a1 ship

- `pip install -e .` works in a fresh venv
- `ffca-report --help` prints sensible help
- `examples/01_tabular_breast_cancer.py` runs end-to-end (< 1 min)
- `examples/03_image_cifar10_channel.py` runs end-to-end (< 10 min)
- Both produce `report.md`, `report.json`, and a `plots/` directory with
  at least 8 PNG files
- `pytest` passes
- Cauchy-HVP validation harness still passes (Spearman ≥ 0.90)
