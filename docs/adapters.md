# Writing your own adapter

If your model isn't covered by the three built-in adapters
(`TabularAdapter`, `PixelAdapter`, `ChannelAdapter`), subclass
`FFCAModelAdapter` and implement two methods. The full interface is
tiny:

```python
class FFCAModelAdapter(abc.ABC):
    n_features: int                   # flattened size per sample
    feature_shape: tuple[int, ...]    # original (unflattened) shape
    feature_names: list[str] | None

    def feature_input(self, batch) -> torch.Tensor: ...
    def scalar_output(self, x, batch) -> torch.Tensor: ...
```

`feature_input` decides **what FFCA is computing gradients with respect
to**. `scalar_output` decides **what scalar FFCA is differentiating**.

## Example 1 — sklearn-style wrapper

You have a model that needs explicit `.train()`/`.eval()` toggles and a
non-standard forward signature. ~10 lines:

```python
from ffca.core import FFCAModelAdapter

class WrappedAdapter(FFCAModelAdapter):
    n_features = 50
    feature_shape = (50,)

    def __init__(self, wrapper):
        super().__init__(wrapper.module)
        self.wrapper = wrapper
        self.feature_names = wrapper.feature_names

    def feature_input(self, batch):
        x = batch.features if hasattr(batch, "features") else batch[0]
        return x.clone().detach().requires_grad_(True)

    def scalar_output(self, x, batch):
        out = self.wrapper.predict_logits(x)
        return out.gather(1, out.argmax(1, keepdim=True)).sum()
```

## Example 2 — transformer token-level FFCA

Differentiate w.r.t. **input embeddings** so the feature axis is per
token × hidden dim:

```python
class TokenAdapter(FFCAModelAdapter):
    def __init__(self, hf_model, seq_len, hidden):
        super().__init__(hf_model)
        self.n_features = seq_len * hidden
        self.feature_shape = (seq_len, hidden)
        self.feature_names = [f"tok{t}_h{h}"
                              for t in range(seq_len) for h in range(hidden)]

    def feature_input(self, batch):
        with torch.no_grad():
            embs = self.model.get_input_embeddings()(batch["input_ids"])
        return embs.clone().requires_grad_(True)

    def scalar_output(self, embs, batch):
        out = self.model(inputs_embeds=embs)
        # next-token log-prob at the last position
        last = out.logits[:, -1]
        return last.gather(1, last.argmax(1, keepdim=True)).sum()
```

## Example 3 — VAE latent space

Make `feature_input` the encoder output, `scalar_output` the decoder's
reconstruction-loss scalar:

```python
class LatentAdapter(FFCAModelAdapter):
    def __init__(self, encoder, decoder, latent_dim):
        super().__init__(decoder)  # the model FFCA differentiates is the decoder
        self.encoder = encoder
        self.n_features = latent_dim
        self.feature_shape = (latent_dim,)
        self.feature_names = [f"z{i}" for i in range(latent_dim)]

    def feature_input(self, batch):
        with torch.no_grad():
            mu, _ = self.encoder(batch[0])
        return mu.clone().requires_grad_(True)

    def scalar_output(self, z, batch):
        recon = self.model(z)
        # negative reconstruction quality as scalar (smaller = better fit)
        return torch.nn.functional.mse_loss(recon, batch[0], reduction='sum')
```

## Tips & gotchas

- **Always set `requires_grad=True`** on the tensor you return from
  `feature_input`. Otherwise autograd has no leaf to differentiate w.r.t.
- **Always return a SCALAR** from `scalar_output`. If your model emits a
  vector, pick one element (`gather`, `[:, idx]`) or aggregate
  (`.sum()`, `.mean()`).
- **Smoothing**: the package wraps your model in
  `ffca.core.smoothing.smooth()` during analysis, replacing module-style
  ReLU/LeakyReLU/PReLU/GELU/ELU/SiLU with `nn.Softplus(beta=10)`. If
  your model uses functional activations inside `.forward()`
  (`torch.nn.functional.relu(...)`), the swap can't reach them; FFCA
  will warn and you'll get zero Non-linearity / Interaction. Fix:
  promote those activations to `nn.Module` attributes.
- **Forward-hook splice**: if your custom adapter needs to splice into
  the middle of a model (like `ChannelAdapter` does), study its
  `_capture_activation` / `_forward_with_replacement` pattern. Always
  resolve the target layer by name on every call — smoothing will swap
  identities under you.
- **Channel-mean splice and linear tails**: if every layer after your
  splice is linear (e.g., a single `Linear` head), the Hessian w.r.t.
  the channel mean is mathematically zero and Non-linearity /
  Interaction will come back as zeros. That's correct, not a bug — your
  signal lives in Impact alone in that case.
