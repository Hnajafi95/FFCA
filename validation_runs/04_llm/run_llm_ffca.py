"""Validation 04 — FFCA on a small HuggingFace transformer (LLM).

Uses `prajjwal1/bert-tiny` (4.4M params, 2 layers, 128 hidden) — small
enough to run in seconds on CPU/MPS while still being a real transformer
trained on real data.

Two adapters demonstrated:
  EmbeddingTokenAdapter — feature axis = (seq_len × hidden) input embeddings
  HeadActivationAdapter — feature axis = per-attention-head pooled outputs

Both run WITH and WITHOUT the three improvements.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ffca import FFCAReport
from ffca.core.adapter import FFCAModelAdapter
from ffca.core.scalars import predicted_class

OUT = Path(__file__).resolve().parent
DEVICE = torch.device("cpu")  # bert-tiny is fine on CPU; avoids float32/64 mix on MPS

MODEL_ID = "distilgpt2"
SEQ_LEN = 16          # short — controls memory/speed
N_SAMPLES = 8         # FFCA budget (distilgpt2 is heavier than bert-tiny)
BATCH = 2


# --------------------------------------------------------------- dataset
PROMPTS = [
    "The capital of France is Paris.",
    "Machine learning models can fail in surprising ways.",
    "Climate change is altering global precipitation patterns.",
    "Pythagoras showed that the sum of squared sides equals the hypotenuse squared.",
    "Neural networks are universal function approximators.",
    "Photosynthesis converts light energy into chemical energy.",
    "The mitochondria is the powerhouse of the cell.",
    "Shakespeare wrote 37 plays in his lifetime.",
    "Water expands when it freezes into ice.",
    "Atoms are the smallest unit of ordinary matter.",
    "The Pacific Ocean is the largest body of water on Earth.",
    "Einstein proposed the theory of relativity in 1905.",
    "The human brain contains roughly 86 billion neurons.",
    "DNA stores the genetic instructions for life.",
    "Plate tectonics explains continental drift.",
    "The speed of light is approximately 300,000 km per second.",
]


class TextDataset(Dataset):
    def __init__(self, tokenizer, prompts: list[str], seq_len: int):
        enc = tokenizer(prompts, return_tensors="pt", padding="max_length",
                        truncation=True, max_length=seq_len)
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, i):
        return {"input_ids": self.input_ids[i],
                "attention_mask": self.attention_mask[i]}


def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# --------------------------------------------------------------- adapters
class EmbeddingTokenAdapter(FFCAModelAdapter):
    """Treat input embeddings (seq_len × hidden) as the feature axis.

    Feature axis flattens token position × hidden dim, so for bert-tiny
    with seq=16, hidden=128 we get 2048 features per sample.
    """
    def __init__(self, model: nn.Module, seq_len: int, hidden: int,
                 mode: str = "predicted_class"):
        super().__init__(model)
        self.seq_len = seq_len
        self.hidden = hidden
        self.n_features = seq_len * hidden
        self.feature_shape = (seq_len, hidden)
        self.mode = mode
        self.feature_names = [f"t{t}_h{h}" for t in range(seq_len)
                              for h in range(hidden)]

    def feature_input(self, batch):
        ids = batch["input_ids"].to(self.device())
        # tiny-gpt2: model.transformer.wte is the input embedding
        embed = self.model.transformer.wte if hasattr(self.model, "transformer") \
            else self.model.bert.embeddings.word_embeddings
        with torch.no_grad():
            embs = embed(ids)
        return embs.clone().detach().requires_grad_(True)

    def scalar_output(self, embs, batch):
        attn = batch["attention_mask"].to(self.device())
        out = self.model(inputs_embeds=embs, attention_mask=attn)
        # Last-token logit at the predicted class — sum over batch
        logits = out.logits[:, -1, :] if hasattr(out, "logits") \
            else out.last_hidden_state[:, -1, :]
        return logits.max(dim=-1).values.sum()


class HeadActivationAdapter(FFCAModelAdapter):
    """Per-attention-head pooled outputs from the final encoder layer.

    bert-tiny has 2 layers × 2 heads × 64 head-dim. After the final layer,
    we mean-pool over tokens, giving (B, n_heads, head_dim) → flatten to
    n_heads*head_dim features = 128 features.
    """
    def __init__(self, model: nn.Module, n_heads: int, head_dim: int):
        super().__init__(model)
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_features = n_heads * head_dim
        self.feature_shape = (n_heads, head_dim)
        self.feature_names = [f"h{h}_d{d}" for h in range(n_heads)
                              for d in range(head_dim)]
        self._captured = None
        self._hook = None

    def _target_layer(self):
        # tiny-gpt2: model.transformer.h[-1].attn
        if hasattr(self.model, "transformer"):
            return self.model.transformer.h[-1].attn
        return self.model.bert.encoder.layer[-1].attention.self

    def feature_input(self, batch):
        ids = batch["input_ids"].to(self.device())
        attn = batch["attention_mask"].to(self.device())
        captured = []
        def hook(module, inputs, output):
            # tiny-gpt2 returns tuple (attn_out, ...) where attn_out is (B,T,H)
            t = output[0] if isinstance(output, tuple) else output
            captured.append(t.detach())
        h = self._target_layer().register_forward_hook(hook)
        try:
            with torch.no_grad():
                self.model(input_ids=ids, attention_mask=attn)
        finally:
            h.remove()
        # captured[0]: (B, T, n_heads*head_dim)
        c = captured[0]
        flat = c.mean(dim=1).reshape(c.size(0), self.n_heads, self.head_dim)
        leaf = flat.clone().detach().requires_grad_(True)
        self._batch_ids = ids; self._batch_attn = attn
        return leaf

    def scalar_output(self, leaf, batch):
        T = self._batch_ids.size(1)
        injected = leaf.reshape(leaf.size(0), 1, self.n_heads * self.head_dim).expand(-1, T, -1).contiguous()
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                return (injected,) + output[1:]
            return injected
        handle = self._target_layer().register_forward_hook(hook)
        try:
            out = self.model(input_ids=self._batch_ids,
                              attention_mask=self._batch_attn)
        finally:
            handle.remove()
        logits = out.logits[:, -1, :] if hasattr(out, "logits") \
            else out.last_hidden_state[:, -1, :]
        return logits.max(dim=-1).values.sum()


# --------------------------------------------------------------- run helpers
def run_one(adapter, loader, *, improvements: bool, tag: str):
    rep = FFCAReport(
        adapter, loader,
        n_first_order_samples=N_SAMPLES, n_hessian_samples=4,
        n_diag_probes=12, n_cauchy_probes=16, n_cauchy_samples=4,
        n_cosens_permutations=15, n_cosens_bootstrap=8,
        improvements=improvements,
    )
    t0 = time.time()
    rep.run()
    elapsed = time.time() - t0
    out = OUT / tag
    if out.exists():
        shutil.rmtree(out)
    rep.save(out)
    return rep, out, elapsed


def summarize(rep, out, elapsed, label):
    last = rep.signatures[-1]
    arch = np.bincount(last.archetypes, minlength=8).tolist()
    print(f"\n--- {label} ---")
    print(f"  elapsed: {elapsed:.1f}s")
    print(f"  features: {last.n_features}, interaction method: {last.metadata['interaction_method']}")
    print(f"  archetypes [N,HI,W,Cat,NL,V,St,Cx]: {arch}")
    print(f"  impact range: [{last.impact.min():.4e}, {last.impact.max():.4e}]")
    print(f"  interaction range: [{last.interaction.min():.4e}, {last.interaction.max():.4e}]")
    if rep.cosens and rep.cosens.diagnostics.get("k") is not None:
        d = rep.cosens.diagnostics
        print(f"  cosens k={d['k']} silh={d['silhouette_observed']:.3f} abort={d['abort_recommended']}")
    n_plots = len(list((out / "plots").glob("*.png")))
    print(f"  → {n_plots} plots in {out.name}/plots/")


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {MODEL_ID} …")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE).eval()
    cfg = model.config
    n_params = sum(p.numel() for p in model.parameters())
    n_layers = getattr(cfg, "n_layer", getattr(cfg, "num_hidden_layers", 0))
    n_heads = getattr(cfg, "n_head", getattr(cfg, "num_attention_heads", 0))
    n_embed = getattr(cfg, "n_embd", getattr(cfg, "hidden_size", 0))
    print(f"  {n_params:,} parameters, {n_layers} layers × {n_heads} heads, hidden={n_embed}")

    ds = TextDataset(tok, PROMPTS, SEQ_LEN)
    loader = DataLoader(ds, batch_size=BATCH, collate_fn=collate)
    # Stash config attrs we use later
    cfg.num_hidden_layers = n_layers
    cfg.num_attention_heads = n_heads
    cfg.hidden_size = n_embed

    summary = {"model_id": MODEL_ID, "n_params": n_params,
                "seq_len": SEQ_LEN, "hidden": cfg.hidden_size}

    # ---- Test A: input-embedding FFCA ----
    print(f"\n=== bert-tiny — input-embedding FFCA ({SEQ_LEN}×{cfg.hidden_size}"
          f"={SEQ_LEN*cfg.hidden_size} features) ===")
    for impr, tag in ((True, "emb_with"), (False, "emb_baseline")):
        ad = EmbeddingTokenAdapter(model, SEQ_LEN, cfg.hidden_size)
        rep, out, t = run_one(ad, loader, improvements=impr, tag=tag)
        summarize(rep, out, t, f"EMBEDDING — improvements={impr}")
        summary.setdefault("embedding", {})[tag] = {
            "elapsed_s": t,
            "interaction_method": rep.signatures[-1].metadata["interaction_method"],
            "archetypes": np.bincount(rep.signatures[-1].archetypes, minlength=8).tolist(),
        }

    # ---- Test B: per-attention-head FFCA on final encoder layer ----
    print(f"\n=== bert-tiny — attention-head FFCA "
          f"({cfg.num_attention_heads}×{cfg.hidden_size//cfg.num_attention_heads}"
          f"={cfg.hidden_size} features) ===")
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    for impr, tag in ((True, "head_with"), (False, "head_baseline")):
        ad = HeadActivationAdapter(model, cfg.num_attention_heads, head_dim)
        rep, out, t = run_one(ad, loader, improvements=impr, tag=tag)
        summarize(rep, out, t, f"HEAD — improvements={impr}")
        summary.setdefault("head", {})[tag] = {
            "elapsed_s": t,
            "interaction_method": rep.signatures[-1].metadata["interaction_method"],
            "archetypes": np.bincount(rep.signatures[-1].archetypes, minlength=8).tolist(),
        }

    (OUT / "compare.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSaved {OUT/'compare.json'}")


if __name__ == "__main__":
    main()
