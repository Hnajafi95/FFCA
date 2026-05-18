#!/usr/bin/env python3
"""End-to-end retraining + FFCA + agent-ready reports for the FFCA paper case studies.

Run on HPC, scp results back. Everything (training history, checkpoints,
FFCA reports, plots, vision metrics, optional agent narration) goes under
one output directory — one subdirectory per case.

Cases:
  - credit_loan          : UCI German Credit → hierarchical_learning_confirmed
  - california_housing_leak     : Cal Housing + leaked feature  → data_leakage_immediate_dominance
  - california_housing_spurious : Cal Housing + spurious feature → spurious_correlation_volatile_specialist
  - bike_sharing         : UCI Bike Sharing, long training → overfitting_volatility_spike
  - wine_quality         : UCI Wine, deliberately under-capacity → insufficient_capacity
  - waterbirds           : WILDS Waterbirds, ResNet-18 ERM → shortcut_learning_drift_epoch

Usage:
    python case_studies/run_all.py --output-dir results/
    python case_studies/run_all.py --output-dir results/ --cases bike_sharing,wine_quality
    python case_studies/run_all.py --output-dir results/ --narrate  # requires ANTHROPIC_API_KEY
    python case_studies/run_all.py --output-dir results/ --skip-waterbirds  # tabular only
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Resolve repo root so we can import ffca_agent and (optionally) the local FFCA package.
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))

# Try a few candidate locations for the FFCA PyTorch package. Mac dev layout
# has it as a sibling (`projects/FFCA/FFCA_package/`); HPC layouts often have
# it inside the same parent as this repo (`~/FFCA/FFCA_package/`).
_FFCA_PKG_CANDIDATES = [
    REPO_ROOT.parent / "FFCA" / "FFCA_package",   # Mac sibling layout
    REPO_ROOT.parent / "FFCA_package",            # HPC: FFCA_agent and FFCA_package side-by-side
    Path.home() / "FFCA" / "FFCA_package",        # HPC: ~/FFCA/FFCA_package
    Path("/opt/FFCA/FFCA_package"),               # cluster-wide install
]
for _cand in _FFCA_PKG_CANDIDATES:
    if _cand.exists():
        sys.path.insert(0, str(_cand))
        break

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ── FFCA package (PyTorch) ─────────────────────────────────────────────────
from ffca import FFCAReport, CheckpointLoader, TabularAdapter, ChannelAdapter

# ── ffca_agent local ───────────────────────────────────────────────────────
from ffca_agent.evaluator import evaluate_rulebook, load_rulebook
from ffca_agent.report import ReportContext
from ffca_agent.training import TrainingHistory
from ffca_agent.vision import VisionMetrics, compute_fbr, compute_com_distance, compute_minority_acc


RULEBOOK_PATH = REPO_ROOT / "rulebook" / "ffca_rules.yaml"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def _section(name: str):
    print(f"\n=== {name} ===", flush=True)
    t0 = time.time()
    yield
    print(f"  ({time.time() - t0:.1f}s)", flush=True)


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def _summarize_findings(findings) -> str:
    lines = [f"_{len(findings)} findings_"]
    for sev in ("critical", "warn", "info", None):
        rules = [f.rule_id for f in findings if f.severity == sev]
        if rules:
            label = sev or "descriptor"
            lines.append(f"- **{label}**: {', '.join(sorted(set(rules)))}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Tabular training harness
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TabularCase:
    name: str
    description: str
    expected_rules: list[str]
    make_data: Callable[[], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], str]]
    make_model: Callable[[int, int], nn.Module]
    task: str  # 'regression' or 'classification'
    n_epochs: int = 60
    batch_size: int = 64
    lr: float = 1e-3
    # how many checkpoints to keep across training (sampled approx evenly)
    n_checkpoints: int = 12
    notes: str = ""


def _make_mlp(in_features: int, out_features: int, hidden: tuple[int, ...] = (64, 32)) -> nn.Module:
    """Default 2-hidden-layer MLP with ReLU. Adjusts width by `hidden`."""
    layers: list[nn.Module] = []
    prev = in_features
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, out_features))
    return nn.Sequential(*layers)


def _make_loaders(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    task: str, batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    y_dtype = torch.long if task == "classification" else torch.float32
    Xtr = torch.as_tensor(X_tr, dtype=torch.float32)
    Xva = torch.as_tensor(X_va, dtype=torch.float32)
    if task == "classification":
        ytr = torch.as_tensor(y_tr, dtype=torch.long)
        yva = torch.as_tensor(y_va, dtype=torch.long)
    else:
        ytr = torch.as_tensor(y_tr, dtype=torch.float32).reshape(-1, 1)
        yva = torch.as_tensor(y_va, dtype=torch.float32).reshape(-1, 1)
    tr = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    va = DataLoader(TensorDataset(Xva, yva), batch_size=max(batch_size, 256))
    return tr, va


def _train_tabular(
    case: TabularCase,
    out_dir: Path,
    device: torch.device,
    epoch_override: int | None = None,
) -> dict:
    """Train + save per-epoch checkpoints + history. Returns a summary dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    X_tr, X_va, y_tr, y_va, feature_names, task = case.make_data()
    n_in = X_tr.shape[1]
    n_out = (len(np.unique(y_tr)) if task == "classification" else 1)
    model = case.make_model(n_in, n_out).to(device)
    train_loader, val_loader = _make_loaders(X_tr, y_tr, X_va, y_va, task, case.batch_size)

    opt = torch.optim.Adam(model.parameters(), lr=case.lr)
    if task == "classification":
        loss_fn: nn.Module = nn.CrossEntropyLoss()
        metric_name = "val_accuracy"
        higher_is_better = True
    else:
        loss_fn = nn.MSELoss()
        metric_name = "val_loss"
        higher_is_better = False

    n_epochs = epoch_override if epoch_override else case.n_epochs
    # Pick which epochs to snapshot (always include 1 and last; even-spaced otherwise)
    snapshot_epochs = sorted(set(
        np.linspace(1, n_epochs, num=case.n_checkpoints, dtype=int).tolist()
    ))

    history: dict[str, list[float]] = {"loss": [], "val_loss": []}
    if task == "classification":
        history["accuracy"] = []
        history["val_accuracy"] = []

    ckpt_records: list[tuple[str, str]] = []

    for ep in range(1, n_epochs + 1):
        model.train()
        ep_loss = 0.0
        ep_correct = 0
        ep_total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y if task != "regression" else y)
            loss.backward()
            opt.step()
            ep_loss += float(loss.item()) * x.size(0)
            if task == "classification":
                ep_correct += int((out.argmax(1) == y).sum())
                ep_total += int(x.size(0))

        train_loss = ep_loss / max(ep_total or len(train_loader.dataset), 1)
        history["loss"].append(train_loss)
        if task == "classification":
            history["accuracy"].append(ep_correct / max(ep_total, 1))

        # validation
        model.eval()
        v_loss = 0.0
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                v_loss += float(loss_fn(out, y).item()) * x.size(0)
                if task == "classification":
                    v_correct += int((out.argmax(1) == y).sum())
                v_total += int(x.size(0))
        history["val_loss"].append(v_loss / max(v_total, 1))
        if task == "classification":
            history["val_accuracy"].append(v_correct / max(v_total, 1))

        if ep in snapshot_epochs:
            p = ckpt_dir / f"ep_{ep:03d}.pt"
            torch.save(model.state_dict(), p)
            ckpt_records.append((f"ep_{ep:03d}", str(p)))

        if ep == 1 or ep % max(1, n_epochs // 10) == 0 or ep == n_epochs:
            acc_str = (f", val_acc={history['val_accuracy'][-1]:.3f}"
                       if task == "classification" else "")
            print(f"  epoch {ep:3d}/{n_epochs}  loss={train_loss:.4f}  "
                  f"val_loss={history['val_loss'][-1]:.4f}{acc_str}", flush=True)

    _save_json(out_dir / "history.json", history)
    _save_json(out_dir / "case_meta.json", {
        "name": case.name, "description": case.description,
        "expected_rules": case.expected_rules, "notes": case.notes,
        "task": task, "n_features": n_in, "feature_names": feature_names,
        "metric": metric_name, "higher_is_better": higher_is_better,
        "n_epochs": n_epochs, "snapshot_epochs": [int(x) for x in snapshot_epochs],
    })

    return {
        "ckpt_records": ckpt_records,
        "feature_names": feature_names,
        "task": task,
        "n_in": n_in,
        "n_out": n_out,
        "val_loader": val_loader,
    }


def _run_ffca_tabular(
    case: TabularCase,
    trained: dict,
    out_dir: Path,
    device: torch.device,
) -> None:
    """Run FFCA on the per-epoch checkpoints and save report.json + plots."""
    factory = lambda: case.make_model(trained["n_in"], trained["n_out"]).to(device)
    adapter = TabularAdapter(factory(), feature_names=trained["feature_names"])
    ck_loader = CheckpointLoader(factory, trained["ckpt_records"], device=str(device))

    # Modest sample counts so it finishes quickly on tabular. H200 makes this trivial.
    report = FFCAReport(
        adapter,
        trained["val_loader"],
        n_first_order_samples=128,
        n_hessian_samples=16,
        n_diag_probes=48,
        n_cauchy_probes=96,
        n_cauchy_samples=16,
        n_cosens_permutations=80,
        n_cosens_bootstrap=40,
    ).run(checkpoints=ck_loader)

    report.save(out_dir)
    print(f"  FFCA report written to {out_dir / 'report.json'}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Per-case data + model specs
# ─────────────────────────────────────────────────────────────────────────────

def _credit_loan_data():
    """UCI German Credit. 1000 rows, 20 features (after dummies), binary classification."""
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    ds = fetch_openml(name="credit-g", version=1, as_frame=True, parser="liac-arff")
    X = ds.frame.drop(columns=["class"])
    y = (ds.frame["class"] == "good").astype(int).to_numpy()
    X = X.copy()
    for col in X.columns:
        if X[col].dtype.name in ("object", "category"):
            X[col] = X[col].astype("category").cat.codes
    feature_names = list(X.columns)
    X = StandardScaler().fit_transform(X.to_numpy().astype(np.float32))
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    return X_tr, X_va, y_tr, y_va, feature_names, "classification"


def _california_housing_data():
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    ds = fetch_california_housing()
    X = ds.data.copy()
    y = ds.target.copy()
    feature_names = list(ds.feature_names)
    X = StandardScaler().fit_transform(X)
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_tr, X_va, y_tr, y_va, feature_names


def _california_housing_leaked_data():
    """Cal Housing with a leaked feature (a noisy copy of the target appended).
    Replicates the data-leakage demo from App C.6."""
    X_tr, X_va, y_tr, y_va, feature_names = _california_housing_data()
    rng = np.random.default_rng(42)
    leak_tr = y_tr + rng.normal(0, 0.05, size=y_tr.shape)
    leak_va = y_va + rng.normal(0, 0.05, size=y_va.shape)
    X_tr = np.concatenate([X_tr, leak_tr.reshape(-1, 1)], axis=1)
    X_va = np.concatenate([X_va, leak_va.reshape(-1, 1)], axis=1)
    feature_names = feature_names + ["leaked_target"]
    return X_tr, X_va, y_tr, y_va, feature_names, "regression"


def _california_housing_spurious_data():
    """Cal Housing with a spurious feature (correlated with target on train, noise on val).
    Replicates the spurious-correlation demo from App C.6.

    v0.4: increase train-time correlation from 0.9 → 0.99 and shrink noise from
    σ=0.3 → 0.05 so the spurious feature truly dominates the model's reliance.
    Goal: dominance > 5 (vs the 1.91 we got in v0.3, which fell below the
    spurious_correlation_train_val_gap rule's threshold of 3.0).
    """
    X_tr, X_va, y_tr, y_va, feature_names = _california_housing_data()
    rng = np.random.default_rng(7)
    spur_tr = 0.99 * y_tr + rng.normal(0, 0.05, size=y_tr.shape)
    spur_va = rng.normal(0, 1.0, size=y_va.shape)  # NO correlation in val
    X_tr = np.concatenate([X_tr, spur_tr.reshape(-1, 1)], axis=1)
    X_va = np.concatenate([X_va, spur_va.reshape(-1, 1)], axis=1)
    feature_names = feature_names + ["spurious_feature"]
    return X_tr, X_va, y_tr, y_va, feature_names, "regression"


def _bike_sharing_data():
    """UCI Bike Sharing — hourly. Regression on total ride count."""
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    ds = fetch_openml(name="Bike_Sharing_Demand", version=2, as_frame=True, parser="liac-arff")
    df = ds.frame.copy()
    y_col = "count" if "count" in df.columns else df.columns[-1]
    y = df[y_col].astype(float).to_numpy()
    X = df.drop(columns=[y_col])
    for col in X.columns:
        if X[col].dtype.name in ("object", "category"):
            X[col] = X[col].astype("category").cat.codes
    feature_names = list(X.columns)
    X = StandardScaler().fit_transform(X.to_numpy().astype(np.float32))
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_tr, X_va, y_tr, y_va, feature_names, "regression"


def _wine_quality_data():
    """UCI Wine Quality (red). 1599 rows, 11 features, regression on quality score."""
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    ds = fetch_openml(name="wine-quality-red", version=1, as_frame=True, parser="liac-arff")
    df = ds.frame.copy()
    # The target column is `class` on OpenML (string-encoded integer quality 3-8).
    target_col = "quality" if "quality" in df.columns else "class"
    y = df[target_col].astype(float).to_numpy()
    X = df.drop(columns=[target_col])
    feature_names = list(X.columns)
    X = StandardScaler().fit_transform(X.to_numpy().astype(np.float32))
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_tr, X_va, y_tr, y_va, feature_names, "regression"


TABULAR_CASES: dict[str, TabularCase] = {
    "credit_loan": TabularCase(
        name="credit_loan",
        description="UCI German Credit, MLP, classification. Demonstrates hierarchical learning: linear effects develop early, interactions late.",
        expected_rules=["hierarchical_learning_confirmed", "healthy_archetype_distribution"],
        make_data=_credit_loan_data,
        make_model=lambda n_in, n_out: _make_mlp(n_in, n_out, hidden=(128, 64)),
        task="classification",
        # v0.4: shortened from 80 epochs (which overfit hard) to 25 — stops near
        # val_loss minimum, before the model memorizes and growth ratios collapse.
        n_epochs=25,
        n_checkpoints=12,
        notes="High-capacity model for the hierarchical pattern. App C.2 / §5.2. v0.4: shortened training to keep model in the healthy-learning regime.",
    ),
    "california_housing_leak": TabularCase(
        name="california_housing_leak",
        description="Cal Housing + leaked_target (target + small noise) as 9th feature. Should trigger data_leakage_immediate_dominance.",
        expected_rules=["data_leakage_immediate_dominance", "feature_concentration_extreme"],
        make_data=_california_housing_leaked_data,
        make_model=lambda n_in, n_out: _make_mlp(n_in, n_out, hidden=(64, 32)),
        task="regression",
        n_epochs=60,
        n_checkpoints=12,
        notes="App C.6 Fig 18-19 replication.",
    ),
    "california_housing_spurious": TabularCase(
        name="california_housing_spurious",
        description="Cal Housing + spurious_feature (correlated in train, noise in val). Should trigger spurious_correlation_volatile_specialist.",
        expected_rules=["spurious_correlation_volatile_specialist", "archetype_volatile_specialist"],
        make_data=_california_housing_spurious_data,
        make_model=lambda n_in, n_out: _make_mlp(n_in, n_out, hidden=(64, 32)),
        task="regression",
        n_epochs=60,
        n_checkpoints=12,
        notes="App C.6 Fig 16-17 replication.",
    ),
    "bike_sharing": TabularCase(
        name="bike_sharing",
        description="UCI Bike Sharing, regression, long training to capture overfitting volatility spike.",
        expected_rules=["overfitting_volatility_spike", "late_checkpoint_drift"],
        make_data=_bike_sharing_data,
        # v0.4: 4× wider hidden layers to give the model enough parameters to overfit.
        # (Previous (128, 64) converged cleanly without overfitting — val_loss kept dropping.)
        make_model=lambda n_in, n_out: _make_mlp(n_in, n_out, hidden=(512, 256)),
        task="regression",
        n_epochs=500,
        n_checkpoints=25,
        notes="App C.3 Fig 9 replication. v0.4: wider hidden layers (512,256) + 500 epochs to force overfitting.",
    ),
    "wine_quality": TabularCase(
        name="wine_quality",
        description="UCI Wine Quality, pure linear regression (no hidden layer) to demonstrate insufficient_capacity.",
        expected_rules=["insufficient_capacity"],
        make_data=_wine_quality_data,
        # v0.4: a 4-unit hidden layer was still rich enough to develop 27% Complex
        # Drivers — not a true capacity bottleneck. Drop to pure linear regression
        # (no hidden layers at all). Complex Drivers and Catalysts should fail to
        # develop, val_loss should plateau quickly above zero.
        make_model=lambda n_in, n_out: _make_mlp(n_in, n_out, hidden=()),
        task="regression",
        n_epochs=60,
        n_checkpoints=10,
        notes="Linear-only baseline. v0.4 setup: hidden=() so the model can only express linear effects — Complex Drivers and Catalysts must fail to develop.",
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# 4. Waterbirds case (vision)
# ─────────────────────────────────────────────────────────────────────────────

def _run_waterbirds(out_dir: Path, device: torch.device, n_epochs: int = 20) -> dict:
    """ERM training of ResNet-18 on WILDS Waterbirds with per-epoch FBR/COM/minority-acc.

    Approximation: WILDS doesn't ship per-image segmentation masks, so we
    approximate the foreground as the centered 50% box of each image. This is
    standard for Waterbirds shortcut-learning experiments. The point is the
    relative collapse of the FBR over epochs, not the absolute value.
    """
    try:
        from wilds import get_dataset
        from wilds.common.data_loaders import get_eval_loader, get_train_loader
        import torchvision
        from torchvision import transforms
    except ImportError as exc:
        raise RuntimeError(
            f"Waterbirds requires `wilds` and `torchvision`. Install with "
            f"`pip install wilds torchvision`. Original error: {exc}"
        ) from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    data_root = out_dir / "_data"
    data_root.mkdir(exist_ok=True)

    print("  loading Waterbirds (downloads on first run; ~10 GB)…", flush=True)
    dataset = get_dataset(dataset="waterbirds", download=True, root_dir=str(data_root))

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_transform = train_transform

    train_data = dataset.get_subset("train", transform=train_transform)
    val_data = dataset.get_subset("val", transform=eval_transform)
    train_loader = get_train_loader("standard", train_data, batch_size=64, num_workers=4)
    val_loader = get_eval_loader("standard", val_data, batch_size=64, num_workers=4)

    # v0.4: ImageNet-pretrained ResNet-18 (standard for Waterbirds shortcut-learning).
    # The from-scratch setup never learned a stable representation in 20 epochs.
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    # Replace the final layer for binary classification.
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # v0.4: LR warmup (5 epochs linear ramp) + cosine schedule over the rest.
    warmup_epochs = min(5, max(1, n_epochs // 8))
    def _lr_at(epoch_idx: int) -> float:
        if epoch_idx < warmup_epochs:
            return (epoch_idx + 1) / warmup_epochs
        progress = (epoch_idx - warmup_epochs) / max(1, n_epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))  # cosine to 0
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_at)

    history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
    snapshot_epochs = sorted(set(np.linspace(1, n_epochs, num=min(n_epochs, 10), dtype=int).tolist()))
    ckpt_records: list[tuple[str, str]] = []

    fbr_curve, com_curve, minority_curve, overall_curve = [], [], [], []
    epoch_labels: list[str] = []

    # Foreground mask = centered 50% bounding box (Waterbirds birds are reliably central).
    H, W = 224, 224
    fg_mask = np.zeros((H, W), dtype=bool)
    fg_mask[H // 4 : 3 * H // 4, W // 4 : 3 * W // 4] = True

    for ep in range(1, n_epochs + 1):
        model.train()
        ep_loss, ep_correct, ep_total = 0.0, 0, 0
        for batch in train_loader:
            x, y, _ = batch
            x, y = x.to(device), y.to(device).long()
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            ep_loss += float(loss.item()) * x.size(0)
            ep_correct += int((out.argmax(1) == y).sum())
            ep_total += int(x.size(0))
        history["loss"].append(ep_loss / max(ep_total, 1))
        history["accuracy"].append(ep_correct / max(ep_total, 1))
        scheduler.step()

        # Evaluation + vision metrics
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        preds_list, labels_list, groups_list = [], [], []
        # Compute Grad-CAM-style attribution on a fixed subset for FBR/COM (eval-set)
        grad_target_layer = model.layer4[-1]
        fbr_samples, com_samples = [], []

        for i, batch in enumerate(val_loader):
            x, y, meta = batch
            x = x.to(device); y = y.to(device).long()
            x.requires_grad_(False)
            with torch.no_grad():
                out = model(x)
            preds = out.argmax(1)
            preds_list.append(preds.cpu().numpy())
            labels_list.append(y.cpu().numpy())
            # WILDS Waterbirds: meta cols are (background, y, ...). Group id = 2*y + background.
            try:
                bg = meta[:, 0].numpy()
                yy = y.cpu().numpy()
                groups_list.append((2 * yy + bg).astype(int))
            except Exception:
                groups_list.append(np.zeros(len(y), dtype=int))
            v_loss += float(loss_fn(out, y).item()) * x.size(0)
            v_correct += int((preds == y).sum())
            v_total += int(x.size(0))

            # Grad-CAM on a small subset (first batch only) to keep runtime bounded.
            if i == 0:
                cam_imgs = _gradcam_on_batch(model, grad_target_layer, x[:16], y[:16])
                for cam in cam_imgs:
                    fbr_samples.append(compute_fbr(cam, fg_mask))
                    com_samples.append(compute_com_distance(cam, fg_mask))

        history["val_loss"].append(v_loss / max(v_total, 1))
        history["val_accuracy"].append(v_correct / max(v_total, 1))
        preds_all = np.concatenate(preds_list)
        labels_all = np.concatenate(labels_list)
        groups_all = np.concatenate(groups_list)

        minority_acc = compute_minority_acc(preds_all, labels_all, groups_all,
                                            minority_groups=(1, 2))
        minority_curve.append(minority_acc)
        overall_curve.append(history["val_accuracy"][-1])
        fbr_curve.append(float(np.mean(fbr_samples)) if fbr_samples else 0.0)
        com_curve.append(float(np.mean(com_samples)) if com_samples else 0.0)
        epoch_labels.append(f"ep_{ep:03d}")

        if ep in snapshot_epochs:
            p = ckpt_dir / f"ep_{ep:03d}.pt"
            torch.save(model.state_dict(), p)
            ckpt_records.append((f"ep_{ep:03d}", str(p)))

        print(f"  waterbirds ep {ep:2d}/{n_epochs}  "
              f"val_acc={history['val_accuracy'][-1]:.3f}  "
              f"minor_acc={minority_acc:.3f}  "
              f"fbr={fbr_curve[-1]:.2f}  com={com_curve[-1]:.3f}", flush=True)

    # Save artifacts
    _save_json(out_dir / "history.json", history)
    metrics = VisionMetrics(
        fbr_curve=np.array(fbr_curve),
        com_distance_curve=np.array(com_curve),
        minority_acc_curve=np.array(minority_curve),
        overall_acc_curve=np.array(overall_curve),
        epoch_labels=epoch_labels,
        notes=["Foreground approximated as centered 50% box (WILDS does not ship masks)"],
    )
    metrics.save(out_dir / "vision_metrics.json")
    _save_json(out_dir / "case_meta.json", {
        "name": "waterbirds",
        "description": "WILDS Waterbirds ERM, ResNet-18. Demonstrates shortcut learning + drift.",
        "expected_rules": ["shortcut_learning_drift_epoch"],
        "task": "classification (vision)",
        "n_epochs": n_epochs,
        "snapshot_epochs": [int(x) for x in snapshot_epochs],
    })

    # Run FFCA via ChannelAdapter on layer4 features
    try:
        with _section("Waterbirds — FFCA via ChannelAdapter"):
            factory = lambda: torchvision.models.resnet18(weights=None, num_classes=2).to(device)
            # ChannelAdapter takes layer_name (string) for the module to hook.
            # For ResNet-18 the last conv block group is "layer4".
            adapter = ChannelAdapter(factory(), layer_name="layer4")
            ck_loader = CheckpointLoader(factory, ckpt_records, device=str(device))
            report = FFCAReport(
                adapter,
                val_loader,
                n_first_order_samples=64,
                n_hessian_samples=8,
                n_diag_probes=24,
                n_cauchy_probes=48,
                n_cauchy_samples=12,
            ).run(checkpoints=ck_loader)
            report.save(out_dir)
    except Exception as exc:
        print(f"  ChannelAdapter FFCA failed ({exc}); vision_metrics.json still usable.", flush=True)
        (out_dir / "ffca_error.txt").write_text(traceback.format_exc())

    return {"ckpt_records": ckpt_records}


def _gradcam_on_batch(model, target_layer, images, labels):
    """Minimal Grad-CAM. Returns a list of (H, W) numpy heatmaps, one per image."""
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(_module, _inp, out):
        activations.append(out.detach())

    def backward_hook(_module, _grad_inp, grad_out):
        gradients.append(grad_out[0].detach())

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)
    try:
        model.zero_grad()
        out = model(images)
        # backprop against the predicted-label logit (paper-standard for Grad-CAM).
        target = out.gather(1, labels.view(-1, 1)).sum()
        target.backward()
        acts = activations[0]      # (B, C, h, w)
        grads = gradients[0]       # (B, C, h, w)
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1)
        cam = F.relu(cam)
        cam = F.interpolate(cam.unsqueeze(1), size=(images.shape[-2], images.shape[-1]),
                            mode="bilinear", align_corners=False).squeeze(1)
        cam_min = cam.amin(dim=(1, 2), keepdim=True)
        cam_max = cam.amax(dim=(1, 2), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return [c.cpu().numpy() for c in cam]
    finally:
        h1.remove()
        h2.remove()


# ─────────────────────────────────────────────────────────────────────────────
# 5. Optional agent narration
# ─────────────────────────────────────────────────────────────────────────────

def _narrate_case(out_dir: Path, model_id: str | None = None) -> None:
    """Run the agent CLI in --narrate mode and save diagnosis.md."""
    from ffca_agent.llm import DEFAULT_MODEL, Narrator, NarratorError

    report_path = out_dir / "report.json"
    if not report_path.exists():
        print(f"  no report.json in {out_dir}, skipping narration", flush=True)
        return
    history_path = out_dir / "history.json"
    vision_path = out_dir / "vision_metrics.json"

    ctx = ReportContext.from_json(report_path)
    if history_path.exists():
        try:
            hist = TrainingHistory.from_keras_history(history_path)
            ctx.attach_training_history(hist)
        except Exception as exc:
            print(f"  history attach failed ({exc}); proceeding without", flush=True)
    if vision_path.exists():
        try:
            metrics = VisionMetrics.from_json(vision_path)
            ctx.attach_vision_metrics(metrics)
        except Exception as exc:
            print(f"  vision attach failed ({exc}); proceeding without", flush=True)

    rulebook = load_rulebook(RULEBOOK_PATH)
    findings = evaluate_rulebook(rulebook, ctx)

    try:
        narrator = Narrator(model=model_id or DEFAULT_MODEL)
        report = narrator.narrate(findings, ctx, training=ctx.training or None)
        diag = _render_diagnosis(report, findings)
        (out_dir / "diagnosis.md").write_text(diag)
        print(f"  diagnosis written to {out_dir / 'diagnosis.md'} "
              f"(in={report.usage.get('input_tokens', '?')}, "
              f"out={report.usage.get('output_tokens', '?')}, "
              f"cache_read={report.usage.get('cache_read_input_tokens', 0)})",
              flush=True)
    except NarratorError as exc:
        print(f"  narration unavailable ({exc}); writing findings summary only", flush=True)
        (out_dir / "findings_summary.md").write_text(_summarize_findings(findings))


def _render_diagnosis(report, findings) -> str:
    lines = ["# Diagnosis", "", "## Executive summary", "", report.executive_summary, ""]
    if report.actions:
        lines.append("## Ranked actions")
        lines.append("")
        for a in sorted(report.actions, key=lambda x: x.priority):
            ids = ", ".join(f"`{r}`" for r in a.rule_ids) if a.rule_ids else ""
            lines.append(f"{a.priority}. **{a.title}.** {a.rationale}  _(from: {ids})_")
        lines.append("")
    if report.caveats:
        lines.append("## Caveats")
        lines.append("")
        for c in report.caveats:
            lines.append(f"- {c}")
        lines.append("")
    lines.append("## All findings")
    lines.append("")
    lines.append(_summarize_findings(findings))
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def _run_tabular_case(case: TabularCase, out_dir: Path, device, epoch_override=None) -> bool:
    """Returns True on success."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # Clear any stale error.txt from a previous failed run
    err_path = out_dir / "error.txt"
    if err_path.exists():
        err_path.unlink()
    try:
        with _section(f"{case.name} — train"):
            trained = _train_tabular(case, out_dir, device, epoch_override)
        with _section(f"{case.name} — FFCA"):
            _run_ffca_tabular(case, trained, out_dir, device)
        return True
    except Exception as exc:
        print(f"  ERROR in {case.name}: {exc}", flush=True)
        err_path.write_text(traceback.format_exc())
        return False


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Where to write all case-study artifacts.")
    p.add_argument("--cases", default=None,
                   help="Comma-separated case names; default = all. "
                        f"Available: {', '.join(list(TABULAR_CASES.keys()) + ['waterbirds'])}.")
    p.add_argument("--skip-waterbirds", action="store_true",
                   help="Run only the tabular cases (skips the 30+ min vision training).")
    p.add_argument("--narrate", action="store_true",
                   help="After each case, call the agent and write diagnosis.md. "
                        "Requires ANTHROPIC_API_KEY.")
    p.add_argument("--model", default=None,
                   help="Claude model id for narration (default: claude-opus-4-7).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epoch-override", type=int, default=None,
                   help="Force every case to use this many epochs (debug / smoke).")
    p.add_argument("--waterbirds-epochs", type=int, default=40,
                   help="v0.4 default raised from 20 → 40 to give the pretrained ResNet "
                        "time to drift into shortcut learning. Set lower for smoke tests.")
    args = p.parse_args(argv)

    _set_seed(args.seed)
    device = _pick_device()
    print(f"device: {device}", flush=True)
    print(f"torch: {torch.__version__}", flush=True)
    out_root: Path = args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)

    # Snapshot the rulebook so the artifacts directory is self-describing.
    shutil.copy(RULEBOOK_PATH, out_root / "ffca_rules.yaml")

    requested = set(args.cases.split(",")) if args.cases else set(TABULAR_CASES.keys()) | {"waterbirds"}
    if args.skip_waterbirds:
        requested.discard("waterbirds")

    summary: dict[str, dict] = {}

    for case_name, case in TABULAR_CASES.items():
        if case_name not in requested:
            continue
        case_out = out_root / case_name
        ok = _run_tabular_case(case, case_out, device, args.epoch_override)
        summary[case_name] = {"ok": ok}
        if ok and args.narrate:
            with _section(f"{case_name} — narrate"):
                _narrate_case(case_out, model_id=args.model)

    if "waterbirds" in requested:
        case_out = out_root / "waterbirds"
        try:
            with _section("waterbirds — train + FFCA"):
                _run_waterbirds(case_out, device, n_epochs=args.waterbirds_epochs)
            summary["waterbirds"] = {"ok": True}
            if args.narrate:
                with _section("waterbirds — narrate"):
                    _narrate_case(case_out, model_id=args.model)
        except Exception as exc:
            print(f"  ERROR in waterbirds: {exc}", flush=True)
            case_out.mkdir(parents=True, exist_ok=True)
            (case_out / "error.txt").write_text(traceback.format_exc())
            summary["waterbirds"] = {"ok": False}

    _save_json(out_root / "summary.json", summary)

    # Human-readable summary
    md_lines = ["# Case-study run summary", ""]
    for name, s in summary.items():
        status = "✓ ok" if s.get("ok") else "✗ FAILED (see error.txt)"
        md_lines.append(f"- **{name}** — {status}")
    md_lines.append("")
    md_lines.append("Inspect per-case directories for `report.json`, `history.json`, "
                    "`vision_metrics.json` (waterbirds only), `report.md`, and `plots/`. "
                    "If `--narrate` was used, `diagnosis.md` is the LLM-narrated layered output.")
    (out_root / "SUMMARY.md").write_text("\n".join(md_lines))
    print("\nAll done. Top-level summary at", out_root / "SUMMARY.md", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
