"""Transformer adapters for FFCA — embedding-level and attention-head.

Both adapters work on any HuggingFace `AutoModel` / `AutoModelForCausalLM`
without modification. They auto-detect the input-embedding attribute and
the per-layer attention block by walking common name conventions:

  GPT-2 / DistilGPT2  : model.transformer.wte                          (embeddings)
                         model.transformer.h[L].attn                   (attention block)
  BERT / DistilBERT   : model.bert.embeddings.word_embeddings
                         model.bert.encoder.layer[L].attention.self
  LLaMA / Mistral     : model.model.embed_tokens
                         model.model.layers[L].self_attn
  Generic fallback    : the first nn.Embedding found in the model

If auto-detection fails, pass `embedding_module=` / `attention_layer=` by
hand. Both adapters use the splice trick (forward_hook re-resolved by name
each call) so the package's automatic ReLU→Softplus / MaxPool→AvgPool /
flash-SDP→math-SDPA smoothing does not break them.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..core.adapter import FFCAModelAdapter
from ..core.scalars import ScalarFn, predicted_class


# ---------------------------------------------------------------- locator
def _find_embedding(model: nn.Module) -> nn.Embedding:
    """Return the input-token embedding module of an HF transformer."""
    for path in (
        "transformer.wte",            # GPT-2 family
        "model.embed_tokens",         # LLaMA, Mistral
        "model.decoder.embed_tokens", # OPT, BART
        "bert.embeddings.word_embeddings",
        "embeddings.word_embeddings",
        "shared",                     # T5
    ):
        obj = model
        ok = True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False; break
            obj = getattr(obj, part)
        if ok and isinstance(obj, nn.Embedding):
            return obj
    # Fallback: first nn.Embedding in the module tree
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            return m
    raise AttributeError("No nn.Embedding found in the model")


def _find_attention_layer(model: nn.Module, layer_idx: int = -1) -> nn.Module:
    """Return the attention module at the requested encoder layer index."""
    for path in (
        ("transformer.h", "attn"),                         # GPT-2
        ("model.layers", "self_attn"),                     # LLaMA
        ("bert.encoder.layer", "attention.self"),
        ("encoder.layer", "attention.self"),
        ("model.encoder.layers", "self_attn"),
        ("model.decoder.layers", "self_attn"),
    ):
        block_path, attr_path = path
        obj = model
        ok = True
        for part in block_path.split("."):
            if not hasattr(obj, part):
                ok = False; break
            obj = getattr(obj, part)
        if not ok or not hasattr(obj, "__getitem__"):
            continue
        try:
            block = obj[layer_idx]
        except (IndexError, TypeError):
            continue
        attr_obj = block
        for p in attr_path.split("."):
            if not hasattr(attr_obj, p):
                attr_obj = None; break
            attr_obj = getattr(attr_obj, p)
        if attr_obj is not None and isinstance(attr_obj, nn.Module):
            return attr_obj
    raise AttributeError(
        f"Could not auto-locate attention block at layer index {layer_idx}; "
        f"pass attention_layer= explicitly"
    )


def _last_token_logit_scalar(out: torch.Tensor, batch=None) -> torch.Tensor:
    """Default LLM scalar: max logit at the last sequence position."""
    if hasattr(out, "logits"):
        last = out.logits[:, -1, :]
    elif hasattr(out, "last_hidden_state"):
        last = out.last_hidden_state[:, -1, :]
    else:
        last = out
    return last.max(dim=-1).values.sum()


# ---------------------------------------------------------------- embedding adapter
class TransformerEmbeddingAdapter(FFCAModelAdapter):
    """Treat token × hidden-dim input embeddings as the feature axis.

    For a sequence of length T and hidden size H, this yields T·H features.
    Suitable for understanding which token position / which hidden dim drives
    the model's prediction.

    Args:
        model           : HF model (AutoModel / AutoModelForCausalLM)
        seq_len, hidden : sequence length and hidden size of the input
        scalar          : optional ScalarFn; defaults to last-token max-logit
        embedding_module: optional override for the nn.Embedding to differentiate
    """
    def __init__(
        self,
        model: nn.Module,
        seq_len: int,
        hidden: int,
        *,
        scalar: ScalarFn | None = None,
        embedding_module: nn.Embedding | None = None,
    ):
        super().__init__(model, scalar=scalar or _last_token_logit_scalar)
        self.seq_len = seq_len
        self.hidden = hidden
        self.n_features = seq_len * hidden
        self.feature_shape = (seq_len, hidden)
        self.feature_names = [f"t{t}_h{h}" for t in range(seq_len)
                              for h in range(hidden)]
        self._emb = embedding_module or _find_embedding(model)

    def feature_input(self, batch) -> torch.Tensor:
        ids = batch["input_ids"] if isinstance(batch, dict) else batch[0]
        ids = ids.to(self.device())
        with torch.no_grad():
            embs = self._emb(ids)
        return embs.clone().detach().requires_grad_(True)

    def scalar_output(self, embs: torch.Tensor, batch) -> torch.Tensor:
        kw = {}
        if isinstance(batch, dict):
            if "attention_mask" in batch:
                kw["attention_mask"] = batch["attention_mask"].to(self.device())
        out = self.model(inputs_embeds=embs, **kw)
        return self._scalar(out, batch)


# ---------------------------------------------------------------- head adapter
class TransformerHeadAdapter(FFCAModelAdapter):
    """Per-attention-head pooled activations from a chosen encoder layer.

    For an attention block with H heads of dim D, the feature axis is H·D.
    The mean-pool over tokens turns the (B, T, H·D) attention output into a
    (B, H, D) feature tensor that FFCA can differentiate.

    Args:
        model           : HF model
        n_heads, head_dim: from the model's config
        layer_idx        : which layer (negative indexes from the end; -1 = last)
        attention_layer  : optional override for the attention module
        scalar           : optional ScalarFn
    """
    def __init__(
        self,
        model: nn.Module,
        n_heads: int,
        head_dim: int,
        *,
        layer_idx: int = -1,
        attention_layer: nn.Module | None = None,
        scalar: ScalarFn | None = None,
    ):
        super().__init__(model, scalar=scalar or _last_token_logit_scalar)
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.n_features = n_heads * head_dim
        self.feature_shape = (n_heads, head_dim)
        self.feature_names = [f"h{h}_d{d}" for h in range(n_heads)
                              for d in range(head_dim)]
        self._attn = attention_layer or _find_attention_layer(model, layer_idx)
        self._batch_ids: torch.Tensor | None = None
        self._batch_attn: torch.Tensor | None = None

    def feature_input(self, batch) -> torch.Tensor:
        ids = batch["input_ids"] if isinstance(batch, dict) else batch[0]
        attn_mask = (batch.get("attention_mask")
                     if isinstance(batch, dict) else None)
        ids = ids.to(self.device())
        if attn_mask is not None:
            attn_mask = attn_mask.to(self.device())

        captured: list[torch.Tensor] = []
        def hook(module, inputs, output):
            t = output[0] if isinstance(output, tuple) else output
            captured.append(t.detach())
        h = self._attn.register_forward_hook(hook)
        try:
            with torch.no_grad():
                kw = {"input_ids": ids}
                if attn_mask is not None:
                    kw["attention_mask"] = attn_mask
                self.model(**kw)
        finally:
            h.remove()
        if not captured:
            raise RuntimeError("attention hook never fired; "
                                "check `attention_layer` is on the forward path")
        c = captured[0]
        # Pool over tokens, shape → (B, n_heads, head_dim)
        flat = c.mean(dim=1).reshape(c.size(0), self.n_heads, self.head_dim)
        leaf = flat.clone().detach().requires_grad_(True)
        self._batch_ids = ids
        self._batch_attn = attn_mask
        return leaf

    def scalar_output(self, leaf: torch.Tensor, batch) -> torch.Tensor:
        T = self._batch_ids.size(1)
        injected = leaf.reshape(leaf.size(0), 1, self.n_heads * self.head_dim) \
                       .expand(-1, T, -1).contiguous()
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                return (injected,) + output[1:]
            return injected
        handle = self._attn.register_forward_hook(hook)
        try:
            kw = {"input_ids": self._batch_ids}
            if self._batch_attn is not None:
                kw["attention_mask"] = self._batch_attn
            out = self.model(**kw)
        finally:
            handle.remove()
        return self._scalar(out, batch)
