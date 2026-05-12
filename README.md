# FFCA

[![PyPI - Version](https://img.shields.io/pypi/v/ffca.svg)](https://pypi.org/project/ffca/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-PyTorch%20only-orange.svg)](#framework-support-and-limitations)

> ⚠️ **PyTorch only.** FFCA computes derivatives through `torch.autograd`. It
> does **not** work with TensorFlow / Keras (`.h5`, SavedModel) or JAX models.
> If your model is in TensorFlow, see
> [Framework support and limitations](#framework-support-and-limitations) below.

**Feature-Function Curvature Analysis (FFCA)** — explainability across
*architectures* (MLPs, CNNs, Transformers, …) for any differentiable PyTorch
model. From a trained model + a DataLoader, FFCA produces a 4-D *signature*
per feature (Impact, Volatility, Non-linearity, Interaction), classifies each
feature into one of 8 archetypes, and emits a report that flags overfitting,
data leakage, shortcut learning, unstable feature roles, and prune-safe /
load-bearing features.

Works on tabular MLPs, CNNs (pixel- *or* channel-level), Transformer
embeddings, and attention heads — all through the same primitives. The
adapter is the same; the math is the same; only the wrapper is per-arch.

---

## Install

```bash
pip install ffca                            # core: tabular FFCA
pip install "ffca[image]"                   # +torchvision for CNNs
pip install "ffca[netcdf]"                  # +xarray for scientific data (.nc files)
```

> The PyPI distribution is **`ffca`**, the import name is `ffca`, and the CLI
> binary is **`ffca-report`**.

## Framework support and limitations

FFCA is **PyTorch-only**. Concretely:

| Framework | Supported? | Why |
|---|---|---|
| **PyTorch** (`torch.nn.Module`) | ✅ Yes | All adapters / HVP / autograd code targets `torch.autograd`. |
| **TensorFlow / Keras** (`.h5`, SavedModel, `tf.keras`) | ❌ No | Would need a parallel backend on `tf.GradientTape`. The wizard now detects `.py` files that `import tensorflow` / `keras` and gives a clean error instead of crashing. |
| **JAX / Flax** | ❌ No (planned) | JAX's `jax.grad` / `jax.hessian` map cleanly to FFCA's math but the backend hasn't been written. |
| **ONNX** | ❌ No | ONNX runtimes don't generally expose 2nd-order gradients. |

**If you have a TF/Keras model and want to run FFCA on it:**

1. **Port the architecture to PyTorch.** Most "standard" layers translate
   directly:
   - `tf.keras.layers.Conv2D` → `torch.nn.Conv2d`
   - `BatchNormalization` → `nn.BatchNorm2d`
   - `PReLU(shared_axes=[1,2])` → `nn.PReLU(num_parameters=channels)`
   - `UpSampling2D(size=k)` → `nn.Upsample(scale_factor=k)`
   - `SpatialDropout2D` → `nn.Dropout2d`
   - `Add()` → element-wise `+`
2. **Port the weights.** Keras Conv kernels are `(H, W, in, out)`; PyTorch is
   `(out, in, H, W)`. BatchNorm uses different parameter names. There's no
   built-in `keras2pt` converter in this package — you write a one-shot
   script that reads the `.h5` with `h5py` and assembles a PyTorch
   `state_dict`.
3. **Then run FFCA** as normal on the PyTorch version.

The repo includes one worked example of this at
[`validation_runs/03_srdrn/srdrn_pytorch.py`](validation_runs/03_srdrn/srdrn_pytorch.py)
— a hand-port of the SRDRN super-resolution architecture from Keras.

**Roadmap.** TF/Keras and JAX backends are real engineering efforts (a few
weeks each), not wrappers. Open a GitHub issue if you'd use one — that's the
single most useful signal for prioritising backend work.

## Easiest way to run it — the interactive wizard

If you've never used FFCA before, this is the path:

```bash
ffca-report --interactive
```

The wizard walks you through every question:

1. **Where is your model defined?** — give the importable Python class
   (e.g. `mypkg.models:MyCNN`).
2. **Path to the trained weights** — `.pt` / `.pth` file.
3. **What kind of model is this?** — MLP, CNN, or Transformer.
4. **(CNN only) Pixel-level or channel-level analysis?** — and for
   channel-level, it scans your model, prints every Conv/Linear layer with
   an index, and asks which ones to investigate:
   ```
   Found 12 candidate layer(s) in your model:

     [  0] Conv2d          conv1
     [  1] Conv2d          layer1.0.conv1
     [  2] Conv2d          layer1.0.conv2
     [  3] Conv2d          layer2.0.conv1
     ...
     [ 11] Linear          fc

   Pick one or more layer indices to investigate.
   Examples: '0' (just the first); '0,3,5' (three layers); 'all'.
     Layer indices: 0,3,11
   ```
   FFCA then produces one report per layer (`out/ch_conv1/`,
   `out/ch_layer2_0_conv1/`, `out/ch_fc/`).
5. **Where is the data?** — the wizard explicitly reminds you this must be
   the **same data the model was trained on** (or held-out data from the same
   distribution). Pass arbitrary data and the FFCA signature is meaningless,
   because the derivatives are taken against your specific trained weights.
6. **Where should the report go?** — output directory.

## One-liner (for scripts)

When you already know what you want, skip the wizard and pass flags:

```bash
# Simplest: --model-type does the right thing for MLPs and CNNs
ffca-report \
  --model-class my_pkg.models:MyMLP \
  --weights ckpt/final.pt \
  --model-type mlp \
  --data data.csv \
  --out out/
```

```bash
# CNN, channel-level on one specific layer
ffca-report \
  --model-class torchvision.models:resnet50 \
  --weights ckpt/final.pt \
  --model-type cnn --layer layer4.2.conv2 \
  --data data/imagenet_val/ --image-size 224 \
  --out out/
```

```bash
# Scientific data (NetCDF), e.g. climate / SR models
ffca-report \
  --model-class my_pkg.models:SRNet --weights net.pt \
  --model-type cnn --layer encoder.block4 \
  --data climate.nc --data-channels precip,temp,humidity \
  --target-column rainfall \
  --out out/
```

`--data-format` accepts `auto | csv | imagefolder | netcdf | npy` — `auto`
(default) infers from the path.

## What you get

Each run writes three things into `--out`:

- **`report.md`** — a human-readable report. Diagnostic findings have a
  `headline`, an `observation`, a `why_it_matters`, and a `recommendation`.
- **`report.json`** — the full numerical results (signatures, archetypes,
  Trust Scores, Co-Sensitivity groups, findings, timings). Use this for
  downstream analysis.
- **`plots/`** — six to ten PNGs (depending on adapter and whether you
  supplied checkpoints).

## What the plots show

Each PNG answers a specific question. Read the file name as a hint, and
use this table to know what to look for.

| File                                | What it shows                                                                | What to look for                                                                                                                                                |
| ----------------------------------- | ---------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `01_signature_radar.png`            | The 4-D signature (I, V, N, X) per checkpoint, as a radar.                  | A growing **Volatility** axis late in training ⇒ overfitting. A flat **Interaction** axis on a deep model ⇒ the model isn't actually using cross-feature info. |
| `02_archetype_distribution.png`     | Stacked bar of which of the 8 archetypes each feature belongs to, per checkpoint. | A swelling **Noise** column ⇒ features dying out. A single archetype dominating ⇒ low-capacity / brittle model.                                                |
| `03_impact_ranking.png`             | Top features by Impact (final checkpoint).                                  | Are the model's top features the ones you *expected*? If not — investigate.                                                                                    |
| `04_interaction_ci.png`             | Interaction strength per feature with 95 % Cauchy-HVP confidence intervals. | Wide CIs ⇒ probe count too low (bump `--n-probes`). Tight CIs ⇒ trust the ordering.                                                                            |
| `05_*_archetype_grid.png`           | One small panel per feature/channel showing its archetype across checkpoints. | Look for features whose colour band changes mid-training — those are unstable.                                                                                  |
| `05_pixel_interaction_map.png`      | (Pixel adapter only) The Interaction map projected back to image space.    | A bright centre ⇒ the model is using the subject. A bright border / ring ⇒ background reliance (shortcut).                                                     |
| `06_fbr_diagnostic.png`             | (Pixel adapter) Foreground/Background Ratio histogram.                      | FBR < 0.5 ⇒ background-shortcut learning.                                                                                                                       |
| `10_impact_evolution.png`           | Each feature's Impact over training checkpoints.                            | Features that suddenly spike at the last checkpoint are typical overfitting signatures.                                                                          |
| `11_ranking_evolution.png`          | Bump chart of feature rank across checkpoints.                              | Stable lines ⇒ stable model. Lines that cross repeatedly ⇒ the model hasn't settled.                                                                            |
| `12_archetype_evolution.png`        | Heatmap of (feature × checkpoint) → archetype.                              | Rows that are one solid colour ⇒ stable feature roles. Rows that change colour ⇒ unstable.                                                                       |
| `13_trust_scatter.png`              | Trust Score: Stability vs. Importance per feature.                          | Top-right = "Confidently Keep". Bottom-left = "Confidently Prune". The crowded middle is where you actually have decisions to make.                            |
| `20_co_sensitivity_groups.png`      | Functional groups of features (gradient-correlation k-medoids).             | Groups with high Noise fraction *and* statistical support are prune candidates. The package will refuse to recommend a prune when the evidence is weak.        |

## Diagnostic findings — what they mean

Every run produces a `findings` list. These are the types you can encounter,
along with what triggers each one and the action FFCA recommends:

| Finding                     | Severity         | When it fires                                                                                          | Typical recommendation                                                                  |
| --------------------------- | ---------------- | ------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------- |
| `overfitting`               | critical/warn/info | Mean Volatility at the final checkpoint is N× higher than the median of earlier checkpoints.        | Early stop earlier, add regularisation, or re-check held-out generalisation.            |
| `data_leakage`              | warn             | One or more features have very high Impact AND very low Non-linearity + Interaction (z-scores > 3, < −0.5). | Audit those features for post-hoc derivation from the target; ablate and re-train. |
| `shortcut_learning`         | info/warn        | Foreground/Background interaction ratio is below 0.5 (i.e. the model relies on the image border).    | Re-balance dataset or use augmentation that breaks the spurious correlation.            |
| `trust_instability`         | warn             | More than half of features change archetype between checkpoints.                                       | Train longer, change LR schedule, or examine whether your data is too noisy.            |
| `trust_keep_recommended`    | info             | Features that retain a useful archetype across all checkpoints — the load-bearing set.                 | Protect with extra production logging; do not prune.                                    |
| `trust_prune_recommended`   | info             | Features that are stably Noise across all checkpoints.                                                 | Prune — expected accuracy loss ≈ 0.                                                     |
| `co_sensitivity`            | info             | Co-Sensitivity ran but refused to recommend a prune because statistical support was insufficient.      | Don't prune from this signal; use magnitude/movement pruning instead.                  |
| `capacity`                  | info             | Healthy archetype distribution detected.                                                               | None — informational.                                                                   |
| `saturation`                | warn             | A large fraction of features have near-zero Impact (under-utilised feature space).                     | Reduce model size, or look for dying-ReLU / under-fitting patterns.                     |
| `tabular_shortcut`          | warn             | One feature carries a disproportionate share of total Impact.                                          | Audit it for leakage; consider feature-shuffle test.                                    |

Each finding in `report.json` carries `headline`, `observation`,
`why_it_matters`, and `recommendation` — they read like a code-review
comment, not a numerical dump.

## What FFCA actually computes

| Axis | Symbol | Definition | What it tells you |
| --- | --- | --- | --- |
| Impact | I | E[\|∂f/∂xᵢ\|] | How much does feature *i* move the output? |
| Volatility | V | Var[∂f/∂xᵢ] | Is the effect consistent or context-dependent? |
| Non-linearity | N | E[\|∂²f/∂xᵢ²\|] | Does the relationship curve? |
| Interaction | X | Σⱼ E[\|∂²f/∂xᵢ∂xⱼ\|] | Does feature *i* act through others? |

Classified into 8 archetypes:

| Archetype | High in | Practitioner read |
| --- | --- | --- |
| Noise | nothing | safe to prune (verify with Trust Score) |
| Hidden Interactor | X only | weak alone, strong through interactions |
| Workhorse | I (clean) | linear, reliable, independent |
| Catalyst | I + X | strong + couples with others |
| Nonlinear Driver | I + N | curved, important |
| Volatile Specialist | I + V | strong in some contexts |
| Stable Contributor | I (moderate) | mild but reliable |
| Complex Driver | high everywhere | inspect with the full toolkit |

## Built-in adapters

| Adapter | Use for | Feature axis |
| --- | --- | --- |
| `TabularAdapter` | MLPs, tabular transformers | input columns |
| `PixelAdapter` | image classifiers | C × H × W pixels |
| `ChannelAdapter` | any CNN — intermediate layer | C channels (mean-pooled) |
| `TransformerEmbeddingAdapter` | HF Transformers — input embeddings | hidden-dim |
| `TransformerHeadAdapter` | HF Transformers — attention heads | n_layers × n_heads |

Any model not covered by these gets a ~20-line custom adapter:

```python
from ffca.core import FFCAModelAdapter

class MyAdapter(FFCAModelAdapter):
    n_features = 768
    feature_shape = (768,)
    feature_names = None  # optional

    def feature_input(self, batch):
        return batch["embeddings"].clone().requires_grad_(True)

    def scalar_output(self, x, batch):
        out = self.model(inputs_embeds=x)
        return out.logits[:, -1].max(dim=1).values.sum()
```

See [`docs/adapters.md`](docs/adapters.md) for the full guide.

## The three audit-v2 improvements

Pass `--no-improvements` to fall back to baseline FFCA. Otherwise three
extras are computed automatically and stored under `improvements` in the
report:

1. **Cauchy-HVP** — interaction estimation via Pearlmutter HVP with
   Cauchy(0,1) probes. Median Spearman 0.97 vs the exact Hessian at
   `d=16`; ~150× wall-clock speedup at `d=12,288`.
2. **Trust Score** — similarity-weighted entropy across checkpoints,
   producing a two-axis (Stability × Importance) view per feature.
   Drives the `trust_*` findings.
3. **Co-Sensitivity** — gradient-correlation k-medoids with permutation
   + bootstrap-ARI guardrails. Will refuse to recommend a prune unless
   the statistical evidence backs it up.

## Datasets in `validation_runs/`

The [`validation_runs/`](validation_runs/) directory ships real
baseline-vs-with-improvements report pairs on four model families. Each
has a `summary.md` produced by
[`summarize_validation.py`](validation_runs/summarize_validation.py). Brief
context for what's there:

- [`01_tabular/`](validation_runs/01_tabular/) — **Breast Cancer Wisconsin**
  (sklearn). 30 hand-crafted cell-nucleus features (radius, texture,
  perimeter, …); 569 samples, binary malignant/benign. A simple MLP across
  4 training checkpoints — sanity check that FFCA's classical XAI behaviour
  is right.
- [`02_cnn/`](validation_runs/02_cnn/) — **CIFAR-10** (60 000 32×32 images
  across 10 object classes). A small CNN trained for 8 epochs, run both
  **pixel-level** (which input pixels matter?) and **channel-level**
  (which feature channels matter inside the network?). Demonstrates how
  FFCA differentiates between input-space and feature-space attribution.
- [`03_srdrn/`](validation_runs/03_srdrn/) — **SRDRN super-resolution on
  real GCM climate data**. The network upsamples coarse precipitation
  fields. Shows FFCA on a regression task with image-shaped scientific data.
- [`04_llm/`](validation_runs/04_llm/) — **distilgpt2** on a small text
  corpus. Two adapters: the input **embedding** (12 288 hidden dims) and
  the **attention head** (per (layer, head) pair). Demonstrates that the
  signature framework scales past 10 000 features.

The "Waterbirds" benchmark referenced in the `shortcut_learning` finding
description is a separate dataset specifically designed to expose
background-shortcut learning — water-bird species photos placed on
land-bird backgrounds and vice versa. It's not shipped here, but the FBR
diagnostic in `06_fbr_diagnostic.png` is the metric used to detect that
exact failure mode.

To regenerate the per-directory summaries:

```bash
python validation_runs/summarize_validation.py
```

## Citation

If you use FFCA, please cite:

```bibtex
@article{najafi2025ffca,
  title   = {Feature-Function Curvature Analysis: A Geometric Framework
             for Explaining Differentiable Models},
  author  = {Najafi, Hamed and Luo, Dingding and Liu, Jason},
  journal = {arXiv preprint arXiv:2510.27207},
  year    = {2025}
}
```

## License

MIT — see [LICENSE](LICENSE).
