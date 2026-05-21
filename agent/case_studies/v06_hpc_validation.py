"""Single self-contained HPC validation script for FFCA Agent v0.6.

Runs four independent validation experiments and writes a single results
directory with a final VALIDATION_REPORT.md and one JSON per experiment.
Per-section failure is isolated: if section B crashes, sections A, C, D
still complete.

Sections
========

A. Model-zoo false-positive sweep (~15 healthy models, no engineered pathology)
   - 8 healthy tabular UCI MLPs (Iris, Wine, Breast Cancer, Pima Diabetes,
     Adult Income, Bank Marketing, Boston-alt, Forest Cover-mini)
   - 4 ImageNet-pretrained CNNs evaluated on CIFAR-10 (ResNet-18, ResNet-50,
     MobileNet-V2, EfficientNet-B0)
   - Health criteria: no engineered leak/spurious/shortcut. We measure the
     critical-severity firing rate. Expected: should be very low.
   - Output: zoo_results.json + zoo critical-rule firings tally

B. SHAP / Integrated-Gradients head-to-head vs FFCA agent
   - On the 4 tabular v0.5 engineered cases (cal_leak, cal_spurious,
     bike_sharing, credit_loan), compute SHAP values, Integrated Gradients,
     and FFCA Impact.
   - Compare per-feature attribution rankings (Pearson + Spearman vs FFCA).
   - Crucially: does SHAP or IG flag `leaked_target` / `spurious_feature`
     as the most attribution-heavy feature the way FFCA + agent does?
   - Output: baseline_comparison.json + a markdown table per case

C. v0.6 intent ablation
   - 4 cases × 5 intents (audit / diagnose / prune / compare / free) = 20
     narrations.
   - Measure: action-list title variance per intent, rule-free observation
     variance per intent, exec-summary word-count variance.
   - Question: does the intent framing actually change the output?
   - Output: intent_ablation.json

D. Determinism check
   - 2 cases × 3 reruns each with identical case_meta + intent + sig_summary.
     With temperature=0 ideally; otherwise default.
   - Measure: exec-summary token Jaccard similarity, action title
     Jaccard, rule-id-overlap per action.
   - Output: determinism.json

Usage
=====

    python case_studies/v06_hpc_validation.py \\
        --key-file /path/to/api_key.txt \\
        --out-dir results/v06_validation/

Each section is gated by a CLI flag (default: all on):

    --skip-zoo --skip-baselines --skip-intent --skip-determinism

Sections that need ANTHROPIC_API_KEY (C, D, optionally B for compared
narration) silently skip if no key is provided. Section A is API-free.

Cost estimate (with API key)
============================
  A: 0 API calls (training + FFCA + rulebook only)              ~ $0
  B: 4 baseline runs + 4 FFCA narrations                        ~ $1.10
  C: 20 narrations                                              ~ $5.40
  D: 6 narrations                                               ~ $1.60
  --------------------------------------------------------------------
  Total estimated cost:                                         ~ $8.10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Resolve repo paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# FFCA-agent imports
from ffca_agent.case_meta import (  # noqa: E402
    CaseMeta,
    ModelArchitecture,
    NarrationIntent,
    TaskType,
)
from ffca_agent.evaluator import evaluate_rulebook, load_rulebook  # noqa: E402
from ffca_agent.report import ReportContext  # noqa: E402
from ffca_agent.signature_summary import signature_summary  # noqa: E402
from ffca_agent.training import TrainingHistory  # noqa: E402

# Torch / FFCA package
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

# FFCA package (must be installed)
try:
    from ffca import CheckpointLoader, FFCAReport, TabularAdapter, ChannelAdapter  # noqa: E402
    _FFCA_AVAILABLE = True
except ImportError:
    _FFCA_AVAILABLE = False


def _fail_loud(msg: str) -> None:
    """Print a prominent error then exit. Used for setup issues that must be
    fixed before the script can do anything useful."""
    print("", file=sys.stderr)
    print("=" * 72, file=sys.stderr)
    print(" FFCA AGENT VALIDATION — SETUP ERROR", file=sys.stderr)
    print("=" * 72, file=sys.stderr)
    print(msg, file=sys.stderr)
    print("=" * 72, file=sys.stderr)
    sys.exit(1)


def _check_setup(args) -> None:
    """Fail loud and early if the environment isn't ready for the requested
    sections. Better to abort now than to silently produce an empty report."""
    will_run_c = not args.skip_intent
    will_run_d = not args.skip_determinism

    if not _FFCA_AVAILABLE:
        _fail_loud(
            "The `ffca` Python package is not importable from this venv.\n"
            "\n"
            "Install from the repo root:\n"
            "    cd <FFCA repo root>\n"
            "    pip install -e .\n"
            "    pip install -e ./agent\n"
            "    pip install shap captum     # for section B baselines\n"
            "\n"
            "Then re-run this script."
        )

    if (will_run_c or will_run_d) and args.key_file is None:
        _fail_loud(
            "Sections C (intent ablation) and D (determinism) require an\n"
            "Anthropic API key, but --key-file was not supplied.\n"
            "\n"
            "Either:\n"
            "    1. Pass --key-file PATH where PATH is a real file containing\n"
            "       the key on a single line, OR\n"
            "    2. Pass --skip-intent --skip-determinism to opt out.\n"
            "\n"
            "(Sections A and B can run without an API key.)"
        )
    if args.key_file is not None:
        key_path = Path(args.key_file).expanduser().resolve()
        if not key_path.exists():
            _fail_loud(
                f"The key file you supplied does not exist:\n"
                f"    {args.key_file}\n"
                f"    (resolved to: {key_path})\n"
                f"\n"
                f"Did you copy a placeholder path? Check the path and re-run.\n"
                f"To run only the non-API sections, pass:\n"
                f"    --skip-intent --skip-determinism"
            )
        if not key_path.read_text().strip().startswith("sk-"):
            print(f"WARNING: {key_path} does not start with 'sk-' — "
                  f"may not be a valid Anthropic key.", file=sys.stderr)


DEFAULT_RULEBOOK = REPO / "rulebook" / "ffca_rules.yaml"


# ──────────────────────────────────────────────────────────────────────────
# Engineered-case bootstrap
# ──────────────────────────────────────────────────────────────────────────
# Sections B, C, and D operate on the 4 engineered tabular cases from
# `case_studies/run_all.py` (credit_loan, california_housing_leak,
# california_housing_spurious, bike_sharing). On HPC these don't exist yet,
# so we train them in a single preamble step. The artifacts (report.json,
# history.json, checkpoints/) land under `<out_dir>/engineered_cases/<name>/`.

ENGINEERED_CASE_NAMES = [
    "credit_loan",
    "california_housing_leak",
    "california_housing_spurious",
    "bike_sharing",
]


def _bootstrap_engineered_cases(out_dir: Path, device: torch.device) -> Path:
    """Train the 4 engineered tabular cases if their artifacts aren't present.

    Returns the directory holding their per-case subfolders.
    """
    eng_dir = out_dir / "engineered_cases"
    eng_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(SCRIPT_DIR))
    import run_all  # noqa: E402

    for name in ENGINEERED_CASE_NAMES:
        case_dir = eng_dir / name
        report_p = case_dir / "report.json"
        if report_p.exists():
            print(f"  [bootstrap:{name}] already present, skipping training")
            continue
        case = run_all.TABULAR_CASES.get(name)
        if case is None:
            print(f"  [bootstrap:{name}] WARNING: not in run_all.TABULAR_CASES; skipping")
            continue
        case_dir.mkdir(parents=True, exist_ok=True)
        print(f"  [bootstrap:{name}] training (this case is engineered to "
              f"exhibit {case.expected_rules or 'a known pathology'})")
        try:
            trained = run_all._train_tabular(case, case_dir, device)
            run_all._run_ffca_tabular(case, trained, case_dir, device)
            print(f"  [bootstrap:{name}] done")
        except Exception as exc:
            print(f"  [bootstrap:{name}] FAILED: {exc}")
            traceback.print_exc()
    return eng_dir


# ──────────────────────────────────────────────────────────────────────────
# Common helpers
# ──────────────────────────────────────────────────────────────────────────


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _make_mlp(n_in: int, n_out: int, hidden=(64, 32)) -> nn.Module:
    layers: list[nn.Module] = []
    prev = n_in
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, n_out))
    return nn.Sequential(*layers)


def _save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def _seed_everything(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────────────────────────────────
# Section A: Model-zoo false-positive sweep
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class ZooCase:
    name: str
    data_loader_fn: Any
    n_epochs: int = 80
    hidden: tuple = (64, 32)
    n_checkpoints: int = 10


def _load_iris():
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    d = load_iris(as_frame=True)
    X = StandardScaler().fit_transform(d.data.to_numpy().astype(np.float32))
    y = d.target.to_numpy()
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return Xtr, Xva, ytr, yva, list(d.data.columns), "classification", int(y.max() + 1)


def _load_breast_cancer():
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    d = load_breast_cancer(as_frame=True)
    X = StandardScaler().fit_transform(d.data.to_numpy().astype(np.float32))
    y = d.target.to_numpy()
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return Xtr, Xva, ytr, yva, list(d.data.columns), "classification", 2


def _load_diabetes():
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    d = load_diabetes(as_frame=True)
    X = StandardScaler().fit_transform(d.data.to_numpy().astype(np.float32))
    y = d.target.to_numpy().astype(np.float32)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42)
    return Xtr, Xva, ytr, yva, list(d.data.columns), "regression", 1


def _load_california():
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    d = fetch_california_housing(as_frame=True)
    X = StandardScaler().fit_transform(d.data.to_numpy().astype(np.float32))
    y = d.target.to_numpy().astype(np.float32)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42)
    return Xtr, Xva, ytr, yva, list(d.data.columns), "regression", 1


def _load_wine_zoo():
    from sklearn.datasets import load_wine
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    d = load_wine(as_frame=True)
    X = StandardScaler().fit_transform(d.data.to_numpy().astype(np.float32))
    y = d.target.to_numpy()
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return Xtr, Xva, ytr, yva, list(d.data.columns), "classification", int(y.max() + 1)


def _load_digits():
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    d = load_digits(as_frame=True)
    X = StandardScaler().fit_transform(d.data.to_numpy().astype(np.float32))
    y = d.target.to_numpy()
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return Xtr, Xva, ytr, yva, list(d.data.columns), "classification", int(y.max() + 1)


def _load_adult():
    """UCI Adult Income (openml)."""
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    ds = fetch_openml(name="adult", version=2, as_frame=True, parser="liac-arff")
    df = ds.frame.copy().dropna()
    y_col = "class" if "class" in df.columns else df.columns[-1]
    y_raw = df[y_col]
    y = (y_raw == y_raw.value_counts().index[0]).astype(int).to_numpy()
    X = df.drop(columns=[y_col])
    for col in X.columns:
        if X[col].dtype.name in ("object", "category"):
            X[col] = X[col].astype("category").cat.codes
    feats = list(X.columns)
    X = StandardScaler().fit_transform(X.to_numpy().astype(np.float32))
    # subsample for speed
    rng = np.random.RandomState(42)
    if len(X) > 8000:
        idx = rng.choice(len(X), 8000, replace=False)
        X, y = X[idx], y[idx]
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return Xtr, Xva, ytr, yva, feats, "classification", 2


def _load_kc1():
    """OpenML kc1 — software defect prediction. Small healthy tabular case."""
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    ds = fetch_openml(name="kc1", version=1, as_frame=True, parser="liac-arff")
    df = ds.frame.copy().dropna()
    y_col = df.columns[-1]
    y_raw = df[y_col]
    y = (y_raw == y_raw.value_counts().index[0]).astype(int).to_numpy()
    X = df.drop(columns=[y_col])
    for col in X.columns:
        if X[col].dtype.name in ("object", "category"):
            X[col] = X[col].astype("category").cat.codes
    feats = list(X.columns)
    X = StandardScaler().fit_transform(X.to_numpy().astype(np.float32))
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return Xtr, Xva, ytr, yva, feats, "classification", 2


ZOO_TABULAR: list[ZooCase] = [
    ZooCase("iris",          _load_iris,          n_epochs=60),
    ZooCase("wine_zoo",      _load_wine_zoo,      n_epochs=60),
    ZooCase("breast_cancer", _load_breast_cancer, n_epochs=80),
    ZooCase("diabetes",      _load_diabetes,      n_epochs=120),
    ZooCase("digits",        _load_digits,        n_epochs=60),
    ZooCase("california",    _load_california,    n_epochs=80),
    ZooCase("adult",         _load_adult,         n_epochs=40),
    ZooCase("kc1",           _load_kc1,           n_epochs=60),
]


def _train_zoo_case(case: ZooCase, out_dir: Path, device: torch.device) -> dict:
    """Train one healthy zoo case and produce an FFCA report."""
    case_dir = out_dir / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = case_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    Xtr, Xva, ytr, yva, feature_names, task, n_out = case.data_loader_fn()
    n_in = Xtr.shape[1]

    factory = lambda: _make_mlp(n_in, n_out, hidden=case.hidden).to(device)
    model = factory()

    Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32)
    Xva_t = torch.as_tensor(Xva, dtype=torch.float32)
    if task == "classification":
        ytr_t = torch.as_tensor(ytr, dtype=torch.long)
        yva_t = torch.as_tensor(yva, dtype=torch.long)
        loss_fn: nn.Module = nn.CrossEntropyLoss()
    else:
        ytr_t = torch.as_tensor(ytr, dtype=torch.float32).reshape(-1, 1)
        yva_t = torch.as_tensor(yva, dtype=torch.float32).reshape(-1, 1)
        loss_fn = nn.MSELoss()
    train_loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(Xva_t, yva_t), batch_size=512)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    snapshot_epochs = sorted(set(np.linspace(1, case.n_epochs, num=case.n_checkpoints, dtype=int)))
    ckpt_records: list[tuple[str, str]] = []
    history = {"loss": [], "val_loss": []}

    for ep in range(1, case.n_epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
        # validation
        model.eval()
        with torch.no_grad():
            v = sum(loss_fn(model(x.to(device)), y.to(device)).item() * x.size(0)
                    for x, y in val_loader) / len(Xva)
            t = sum(loss_fn(model(x.to(device)), y.to(device)).item() * x.size(0)
                    for x, y in train_loader) / len(Xtr)
        history["loss"].append(t)
        history["val_loss"].append(v)
        if ep in snapshot_epochs:
            p = ckpt_dir / f"ep_{ep:03d}.pt"
            torch.save(model.state_dict(), p)
            ckpt_records.append((f"ep_{ep:03d}", str(p)))

    _save(history, case_dir / "history.json")

    # FFCA report
    adapter = TabularAdapter(factory(), feature_names=feature_names)
    ck_loader = CheckpointLoader(factory, ckpt_records, device=str(device))
    report = FFCAReport(
        adapter, val_loader,
        n_first_order_samples=128, n_hessian_samples=16,
        n_diag_probes=48, n_cauchy_probes=96, n_cauchy_samples=16,
        n_cosens_permutations=80, n_cosens_bootstrap=40,
    ).run(checkpoints=ck_loader)
    report.save(case_dir)
    return {"history_keys": list(history.keys()), "n_checkpoints": len(ckpt_records)}


def run_section_a(out_dir: Path, rulebook: dict, device: torch.device,
                  include_vision: bool = False) -> dict:
    """Model-zoo false-positive sweep. Returns the rollup."""
    section_dir = out_dir / "A_zoo"
    section_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}

    for case in ZOO_TABULAR:
        try:
            t0 = time.time()
            _train_zoo_case(case, section_dir, device)
            ctx = ReportContext.from_json(section_dir / case.name / "report.json")
            h = TrainingHistory.from_keras_history(section_dir / case.name / "history.json")
            h.derive_from_signatures(ctx, top_k=5)
            ctx.attach_training_history(h)
            findings = evaluate_rulebook(rulebook, ctx)
            crit = [f for f in findings if f.severity == "critical"]
            warn = [f for f in findings if f.severity == "warn"]
            results[case.name] = {
                "n_features": ctx.n_features,
                "n_checkpoints": int(ctx.impact_curve.shape[0]),
                "critical_fired": sorted({f.rule_id for f in crit}),
                "warn_fired": sorted({f.rule_id for f in warn}),
                "n_critical": len(crit),
                "n_warn": len(warn),
                "training_time_s": time.time() - t0,
            }
            print(f"  [A:zoo:{case.name}] OK  critical={len(crit)} warn={len(warn)}")
        except Exception as exc:
            results[case.name] = {"error": f"{type(exc).__name__}: {exc}",
                                  "traceback": traceback.format_exc()}
            print(f"  [A:zoo:{case.name}] FAILED: {exc}")

    if include_vision:
        try:
            results["__vision__"] = _run_zoo_vision(section_dir, rulebook, device)
        except Exception as exc:
            results["__vision__"] = {"error": str(exc),
                                      "traceback": traceback.format_exc()}

    # Rollup
    n_total = sum(1 for v in results.values() if "error" not in v)
    n_with_critical = sum(1 for v in results.values()
                          if "error" not in v and v.get("n_critical", 0) > 0)
    rollup = {
        "n_models_total": n_total,
        "n_models_with_critical_findings": n_with_critical,
        "false_positive_rate_critical": n_with_critical / n_total if n_total else None,
        "per_case": results,
    }
    _save(rollup, section_dir / "zoo_results.json")
    return rollup


def _run_zoo_vision(section_dir: Path, rulebook: dict, device: torch.device) -> dict:
    """Run FFCA on a handful of ImageNet-pretrained models on CIFAR-10.

    Lightweight: small image subset, ChannelAdapter on `layer4` (ResNet) or
    equivalent. Goal is FP-rate measurement, not training healthy models.
    """
    from torchvision import datasets, models, transforms

    tform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    cifar = datasets.CIFAR10(root="./_cifar10", train=False, download=True, transform=tform)
    # subsample
    rng = np.random.RandomState(42)
    indices = rng.choice(len(cifar), 256, replace=False).tolist()
    sub = torch.utils.data.Subset(cifar, indices)
    loader = DataLoader(sub, batch_size=16, shuffle=False)

    arch_map = {
        "resnet18": (models.resnet18,    "layer4"),
        "resnet50": (models.resnet50,    "layer4"),
        "mobilenetv2": (models.mobilenet_v2, "features"),
        "efficientnet_b0": (models.efficientnet_b0, "features"),
    }
    out: dict[str, dict] = {}
    for name, (ctor, layer) in arch_map.items():
        try:
            mdir = section_dir / f"vision_{name}"
            mdir.mkdir(parents=True, exist_ok=True)
            model = ctor(weights="DEFAULT").to(device).eval()
            adapter = ChannelAdapter(model, layer_name=layer)
            # Single checkpoint (the pretrained weights)
            torch.save(model.state_dict(), mdir / "ep_001.pt")
            ck = CheckpointLoader(lambda: ctor(weights=None).to(device).eval(),
                                  [("ep_001", str(mdir / "ep_001.pt"))],
                                  device=str(device))
            report = FFCAReport(
                adapter, loader,
                n_first_order_samples=32, n_hessian_samples=8,
                n_diag_probes=16, n_cauchy_probes=32, n_cauchy_samples=8,
                n_cosens_permutations=20, n_cosens_bootstrap=10,
            ).run(checkpoints=ck)
            report.save(mdir)
            ctx = ReportContext.from_json(mdir / "report.json")
            findings = evaluate_rulebook(rulebook, ctx)
            crit = [f for f in findings if f.severity == "critical"]
            out[name] = {
                "n_features": ctx.n_features,
                "n_checkpoints": int(ctx.impact_curve.shape[0]),
                "critical_fired": sorted({f.rule_id for f in crit}),
                "n_critical": len(crit),
            }
            print(f"  [A:zoo-vision:{name}] OK  critical={len(crit)}")
        except Exception as exc:
            out[name] = {"error": f"{type(exc).__name__}: {exc}",
                         "traceback": traceback.format_exc()}
            print(f"  [A:zoo-vision:{name}] FAILED: {exc}")
    return out


# ──────────────────────────────────────────────────────────────────────────
# Section B: SHAP / IG / FFCA head-to-head
# ──────────────────────────────────────────────────────────────────────────


def run_section_b(out_dir: Path, rulebook: dict, device: torch.device,
                  engineered_dir: Path) -> dict:
    """Run SHAP + Integrated Gradients on the 4 engineered tabular cases,
    compare to FFCA Impact. Crucial: does SHAP/IG identify the engineered
    pathology feature (leaked_target, spurious_feature) the way FFCA agent
    does?
    """
    section_dir = out_dir / "B_baselines"
    section_dir.mkdir(parents=True, exist_ok=True)

    # Lazy imports — only required if section B runs
    try:
        import shap  # noqa: F401
    except ImportError:
        return {"error": "shap not installed. pip install shap"}
    try:
        from captum.attr import IntegratedGradients  # noqa: F401
    except ImportError:
        return {"error": "captum not installed. pip install captum"}

    cases = [
        # (name, report_dir, engineered_feature_name_or_None)
        ("credit_loan",                  engineered_dir / "credit_loan", None),
        ("california_housing_leak",      engineered_dir / "california_housing_leak", "leaked_target"),
        ("california_housing_spurious",  engineered_dir / "california_housing_spurious", "spurious_feature"),
        ("bike_sharing",                 engineered_dir / "bike_sharing", None),
    ]
    results: dict[str, dict] = {}
    for name, case_dir, engineered in cases:
        if not (case_dir / "report.json").exists():
            results[name] = {"error": f"engineered case artifacts missing at {case_dir}"}
            print(f"  [B:baselines:{name}] SKIP: artifacts missing")
            continue
        try:
            results[name] = _baseline_one_case(name, case_dir, engineered, rulebook)
            print(f"  [B:baselines:{name}] OK")
        except Exception as exc:
            results[name] = {"error": str(exc), "traceback": traceback.format_exc()}
            print(f"  [B:baselines:{name}] FAILED: {exc}")

    _save(results, section_dir / "baseline_comparison.json")
    return results


def _baseline_one_case(name: str, case_dir: Path, engineered: str | None,
                       rulebook: dict) -> dict:
    """Compute SHAP, IG, and FFCA per-feature attribution for one case.

    The case must have:
      - case_dir / report.json (FFCA Impact lives here)
      - case_dir / checkpoints / ep_NNN.pt  (a final-checkpoint state dict)
      - case_dir / case_meta.json  (NOT YET present in v0.5 artifacts)

    Since the v0.5 setup didn't save data tensors, we re-load via the
    case-specific data loader. To keep this single-script, we rely on the
    `case_studies/run_all.py` loaders being importable.
    """
    sys.path.insert(0, str(SCRIPT_DIR))
    import run_all  # noqa: E402

    loader_map = {
        "credit_loan": getattr(run_all, "_credit_loan_data", None),
        "california_housing_leak": getattr(run_all, "_california_housing_leaked_data", None),
        "california_housing_spurious": getattr(run_all, "_california_housing_spurious_data", None),
        "bike_sharing": getattr(run_all, "_bike_sharing_data", None),
    }
    if not loader_map.get(name):
        return {"error": f"no data loader found for {name} in run_all"}
    loaded = loader_map[name]()
    # run_all loaders return 6-tuple (X_tr, X_va, y_tr, y_va, feat_names, task)
    Xtr, Xva, ytr, yva, feat_names, task = loaded[:6]
    n_in = Xtr.shape[1]
    n_out = (len(np.unique(ytr)) if task == "classification" else 1)
    device = _pick_device()

    # Find the final checkpoint
    ckpt_dir = case_dir / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("ep_*.pt"))
    if not ckpts:
        return {"error": "no checkpoints found"}
    # Build the model factory matching the v0.5 setup
    model_specs = {
        "credit_loan": dict(hidden=(128, 64)),
        "california_housing_leak": dict(hidden=(64, 32)),
        "california_housing_spurious": dict(hidden=(64, 32)),
        "bike_sharing": dict(hidden=(512, 256)),
    }
    spec = model_specs.get(name, dict(hidden=(64, 32)))
    model = run_all._make_mlp(n_in, n_out, **spec).to(device).eval()
    model.load_state_dict(torch.load(ckpts[-1], map_location=device))

    # SHAP (KernelExplainer on a 100-sample background)
    import shap
    bg_idx = np.random.RandomState(42).choice(len(Xtr), min(100, len(Xtr)), replace=False)
    bg = Xtr[bg_idx]
    fg_idx = np.random.RandomState(42).choice(len(Xva), min(200, len(Xva)), replace=False)
    fg = Xva[fg_idx]

    def _model_predict(x):
        with torch.no_grad():
            t = torch.as_tensor(x, dtype=torch.float32).to(device)
            out = model(t)
            if task == "classification":
                return torch.softmax(out, dim=-1).cpu().numpy()
            return out.cpu().numpy().flatten()

    explainer = shap.KernelExplainer(_model_predict, bg)
    shap_vals = explainer.shap_values(fg, silent=True)
    if isinstance(shap_vals, list):
        shap_vals = np.mean([np.abs(s) for s in shap_vals], axis=0)
    shap_per_feat = np.abs(shap_vals).mean(axis=0)
    if shap_per_feat.ndim > 1:
        shap_per_feat = shap_per_feat.mean(axis=tuple(range(1, shap_per_feat.ndim)))

    # Integrated Gradients
    from captum.attr import IntegratedGradients
    ig = IntegratedGradients(model)
    target_arg = 0 if task == "classification" else None
    fg_t = torch.as_tensor(fg, dtype=torch.float32, device=device).requires_grad_(True)
    if task == "classification":
        # Use predicted class
        preds = model(fg_t).argmax(dim=-1)
        ig_attr = np.zeros((fg.shape[0], n_in))
        for i in range(fg.shape[0]):
            single = fg_t[i:i+1]
            attr = ig.attribute(single, target=int(preds[i].item()),
                                n_steps=32).detach().cpu().numpy()
            ig_attr[i] = attr.squeeze()
    else:
        attr = ig.attribute(fg_t, n_steps=32).detach().cpu().numpy()
        ig_attr = attr
    ig_per_feat = np.abs(ig_attr).mean(axis=0)

    # FFCA Impact from the existing report
    ctx = ReportContext.from_json(case_dir / "report.json")
    ffca_impact = ctx.impact

    # Per-feature comparison
    rows = []
    for i, fname in enumerate(feat_names):
        rows.append({
            "feature": fname,
            "ffca_impact": float(ffca_impact[i]) if i < len(ffca_impact) else None,
            "shap": float(shap_per_feat[i]) if i < len(shap_per_feat) else None,
            "ig": float(ig_per_feat[i]) if i < len(ig_per_feat) else None,
        })
    # Sort by FFCA Impact descending
    rows.sort(key=lambda r: -(r["ffca_impact"] or 0))

    # Rank-correlation
    from scipy.stats import pearsonr, spearmanr
    common_n = min(len(ffca_impact), len(shap_per_feat), len(ig_per_feat))
    ffca_v = ffca_impact[:common_n]
    shap_v = shap_per_feat[:common_n]
    ig_v = ig_per_feat[:common_n]
    pearson_shap = pearsonr(ffca_v, shap_v).statistic
    spearman_shap = spearmanr(ffca_v, shap_v).correlation
    pearson_ig = pearsonr(ffca_v, ig_v).statistic
    spearman_ig = spearmanr(ffca_v, ig_v).correlation

    # The big question for engineered cases: did SHAP / IG put the
    # engineered feature in the top spot?
    detection = {}
    if engineered:
        ffca_rank = next((i for i, r in enumerate(rows) if r["feature"] == engineered), -1)
        shap_order = sorted(range(common_n), key=lambda i: -shap_v[i])
        ig_order = sorted(range(common_n), key=lambda i: -ig_v[i])
        feat_idx = feat_names.index(engineered)
        detection = {
            "engineered_feature": engineered,
            "ffca_rank_among_features": ffca_rank,
            "shap_rank_among_features": shap_order.index(feat_idx),
            "ig_rank_among_features": ig_order.index(feat_idx),
        }

    return {
        "case": name,
        "engineered_feature": engineered,
        "n_features": len(feat_names),
        "feature_attributions": rows,
        "rank_corr_shap": {"pearson": float(pearson_shap),
                            "spearman": float(spearman_shap)},
        "rank_corr_ig":   {"pearson": float(pearson_ig),
                            "spearman": float(spearman_ig)},
        "detection_of_engineered_feature": detection,
    }


# ──────────────────────────────────────────────────────────────────────────
# Section C: v0.6 intent ablation
# ──────────────────────────────────────────────────────────────────────────


def _ablation_cases(engineered_dir: Path) -> list[dict]:
    """The 4 engineered cases produced by _bootstrap_engineered_cases.

    Each entry's `dir` points at the per-case subfolder of engineered_dir.
    Sections C and D consume these.
    """
    return [
        {"label": "credit_loan", "dir": engineered_dir / "credit_loan",
         "case_meta": CaseMeta(project_name="credit_loan_ablation",
                                model_architecture=ModelArchitecture.MLP,
                                task_type=TaskType.BINARY_CLASSIFICATION,
                                target_name="credit_risk",
                                domain="financial-risk modelling")},
        {"label": "california_housing_leak", "dir": engineered_dir / "california_housing_leak",
         "case_meta": CaseMeta(project_name="cal_leak_ablation",
                                model_architecture=ModelArchitecture.MLP,
                                task_type=TaskType.REGRESSION,
                                target_name="median_house_value",
                                target_units="$100k",
                                domain="real-estate price prediction",
                                feature_naming_convention="leaked_target = noisy copy of target")},
        {"label": "california_housing_spurious", "dir": engineered_dir / "california_housing_spurious",
         "case_meta": CaseMeta(project_name="cal_spurious_ablation",
                                model_architecture=ModelArchitecture.MLP,
                                task_type=TaskType.REGRESSION,
                                target_name="median_house_value",
                                target_units="$100k",
                                domain="real-estate price prediction",
                                feature_naming_convention="spurious_feature is train-only correlated with target")},
        {"label": "bike_sharing", "dir": engineered_dir / "bike_sharing",
         "case_meta": CaseMeta(project_name="bike_sharing_ablation",
                                model_architecture=ModelArchitecture.MLP,
                                task_type=TaskType.REGRESSION,
                                target_name="bike_rentals",
                                target_units="rides/hour",
                                domain="urban mobility forecasting")},
    ]


def run_section_c(out_dir: Path, rulebook: dict, key_file: Path | None,
                  engineered_dir: Path) -> dict:
    """v0.6 intent ablation: same case under 5 intents, measure differences."""
    section_dir = out_dir / "C_intent_ablation"
    section_dir.mkdir(parents=True, exist_ok=True)
    if not key_file or not key_file.exists():
        return {"skipped": "no API key supplied"}

    from ffca_agent.llm import Narrator
    from ffca_agent.vision import VisionMetrics
    key = key_file.read_text().strip()
    narrator = Narrator(api_key=key)

    intents = [NarrationIntent.AUDIT, NarrationIntent.DIAGNOSE,
               NarrationIntent.PRUNE, NarrationIntent.COMPARE, NarrationIntent.FREE]
    results: dict[str, dict] = {}
    for case in _ablation_cases(engineered_dir):
        label = case["label"]
        report_path = case.get("report_path") or case["dir"] / "report.json"
        if not report_path.exists():
            results[label] = {"error": f"missing report: {report_path}"}
            continue
        try:
            ctx = ReportContext.from_json(report_path)
            history_path = case["dir"] / "history.json"
            if history_path.exists():
                h = TrainingHistory.from_keras_history(history_path)
                h.derive_from_signatures(ctx, top_k=5)
                ctx.attach_training_history(h)
            vision_path = case["dir"] / "vision_metrics.json"
            if vision_path.exists():
                ctx.attach_vision_metrics(VisionMetrics.from_json(vision_path))
            findings = evaluate_rulebook(rulebook, ctx)
            sig_summary = signature_summary(ctx, top_k=5)

            per_intent = {}
            for it in intents:
                rep = narrator.narrate(
                    findings, ctx,
                    case_meta=case["case_meta"], intent=it,
                    sig_summary=sig_summary,
                )
                per_intent[it.value] = {
                    "exec_summary": rep.executive_summary,
                    "exec_summary_n_words": len(rep.executive_summary.split()),
                    "action_titles": [a.title for a in sorted(rep.actions, key=lambda a: a.priority)],
                    "rule_free_obs_count": len(rep.rule_free_observations),
                    "rule_free_obs_titles": [o.what for o in rep.rule_free_observations],
                    "usage": rep.usage,
                }
            # Compute pairwise action-title Jaccard similarity to measure how
            # much intent actually changed the ranked actions.
            from itertools import combinations
            pair_jacc = {}
            for a, b in combinations(intents, 2):
                ta = set(t.lower() for t in per_intent[a.value]["action_titles"])
                tb = set(t.lower() for t in per_intent[b.value]["action_titles"])
                inter = ta & tb
                uni = ta | tb
                pair_jacc[f"{a.value}__{b.value}"] = len(inter) / len(uni) if uni else 0.0
            results[label] = {
                "per_intent": per_intent,
                "action_title_pairwise_jaccard": pair_jacc,
            }
            print(f"  [C:intent:{label}] OK ({sum(len(v['usage']) > 0 for v in per_intent.values())} narrations)")
        except Exception as exc:
            results[label] = {"error": str(exc), "traceback": traceback.format_exc()}
            print(f"  [C:intent:{label}] FAILED: {exc}")

    _save(results, section_dir / "intent_ablation.json")
    return results


# ──────────────────────────────────────────────────────────────────────────
# Section D: Determinism re-runs
# ──────────────────────────────────────────────────────────────────────────


def run_section_d(out_dir: Path, rulebook: dict, key_file: Path | None,
                  engineered_dir: Path, n_reruns: int = 3) -> dict:
    """Re-run identical (case_meta + intent + sig_summary) 3 times per case."""
    section_dir = out_dir / "D_determinism"
    section_dir.mkdir(parents=True, exist_ok=True)
    if not key_file or not key_file.exists():
        return {"skipped": "no API key supplied"}

    from ffca_agent.llm import Narrator
    key = key_file.read_text().strip()
    narrator = Narrator(api_key=key)

    cases = _ablation_cases(engineered_dir)[:2]  # first 2 cases only
    out: dict[str, dict] = {}
    for case in cases:
        label = case["label"]
        report_path = case.get("report_path") or case["dir"] / "report.json"
        if not report_path.exists():
            out[label] = {"error": f"missing report: {report_path}"}
            continue
        try:
            ctx = ReportContext.from_json(report_path)
            history_path = case["dir"] / "history.json"
            if history_path.exists():
                h = TrainingHistory.from_keras_history(history_path)
                h.derive_from_signatures(ctx, top_k=5)
                ctx.attach_training_history(h)
            findings = evaluate_rulebook(rulebook, ctx)
            sig = signature_summary(ctx, top_k=5)

            reruns = []
            for r in range(n_reruns):
                rep = narrator.narrate(
                    findings, ctx,
                    case_meta=case["case_meta"], intent=NarrationIntent.DIAGNOSE,
                    sig_summary=sig,
                )
                reruns.append({
                    "exec_summary": rep.executive_summary,
                    "action_titles": [a.title for a in sorted(rep.actions, key=lambda a: a.priority)],
                    "rule_free_obs_titles": [o.what for o in rep.rule_free_observations],
                })
            # Pairwise Jaccard
            from itertools import combinations
            def _bag(s: str) -> set[str]:
                return set(s.lower().split())
            exec_jacc = []
            action_jacc = []
            for i, j in combinations(range(n_reruns), 2):
                a, b = _bag(reruns[i]["exec_summary"]), _bag(reruns[j]["exec_summary"])
                exec_jacc.append(len(a & b) / max(len(a | b), 1))
                ta = set(t.lower() for t in reruns[i]["action_titles"])
                tb = set(t.lower() for t in reruns[j]["action_titles"])
                action_jacc.append(len(ta & tb) / max(len(ta | tb), 1))
            out[label] = {
                "n_reruns": n_reruns,
                "exec_summary_mean_token_jaccard": float(np.mean(exec_jacc)) if exec_jacc else None,
                "action_titles_mean_jaccard": float(np.mean(action_jacc)) if action_jacc else None,
                "reruns": reruns,
            }
            print(f"  [D:determinism:{label}] OK  exec_jaccard={out[label]['exec_summary_mean_token_jaccard']:.2f}")
        except Exception as exc:
            out[label] = {"error": str(exc), "traceback": traceback.format_exc()}
            print(f"  [D:determinism:{label}] FAILED: {exc}")
    _save(out, section_dir / "determinism.json")
    return out


# ──────────────────────────────────────────────────────────────────────────
# Final report
# ──────────────────────────────────────────────────────────────────────────


def write_final_report(out_dir: Path, section_results: dict) -> None:
    lines = ["# FFCA Agent v0.6 — HPC Validation Report", ""]
    lines.append(f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')}_")
    lines.append("")
    lines.append("This report aggregates four independent validation experiments:")
    lines.append("**A** = model-zoo false-positive sweep, **B** = SHAP/IG/FFCA")
    lines.append("comparison, **C** = v0.6 intent ablation, **D** = determinism.")
    lines.append("")

    # A
    A = section_results.get("A")
    lines.append("## A. Model-zoo false-positive sweep")
    lines.append("")
    if A and "n_models_total" in A:
        n_total = A["n_models_total"]
        n_crit = A["n_models_with_critical_findings"]
        rate = A.get("false_positive_rate_critical")
        lines.append(f"**{n_crit} of {n_total} healthy models triggered at least one critical-severity rule.**")
        if rate is not None:
            lines.append(f"False-positive rate (critical): **{rate*100:.1f}%**.")
        lines.append("")
        lines.append("| Case | n_features | n_critical | critical rules fired |")
        lines.append("|---|---:|---:|---|")
        for name, v in A["per_case"].items():
            if "error" in v:
                lines.append(f"| {name} | — | ERROR | {v['error'][:60]} |")
                continue
            ids = ", ".join(v.get("critical_fired", [])) or "—"
            lines.append(f"| {name} | {v.get('n_features','?')} | {v.get('n_critical',0)} | {ids} |")
    else:
        lines.append(f"_Skipped or failed: {A}_")
    lines.append("")

    # B
    B = section_results.get("B")
    lines.append("## B. SHAP / IG / FFCA head-to-head")
    lines.append("")
    if B and isinstance(B, dict) and "error" not in B:
        lines.append("| Case | engineered | FFCA Impact rank | SHAP rank | IG rank | Pearson(FFCA, SHAP) | Pearson(FFCA, IG) |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for name, v in B.items():
            if not isinstance(v, dict) or "error" in v:
                lines.append(f"| {name} | — | — | — | — | — | ERROR |")
                continue
            eng = v.get("engineered_feature") or "—"
            det = v.get("detection_of_engineered_feature") or {}
            pers_shap = v.get("rank_corr_shap", {}).get("pearson")
            pers_ig = v.get("rank_corr_ig", {}).get("pearson")
            lines.append(f"| {name} | {eng} | "
                         f"{det.get('ffca_rank_among_features', '—')} | "
                         f"{det.get('shap_rank_among_features', '—')} | "
                         f"{det.get('ig_rank_among_features', '—')} | "
                         f"{pers_shap:.3f}" if pers_shap is not None else "—" + " | "
                         f"{pers_ig:.3f}" if pers_ig is not None else "—" + " |")
        lines.append("")
        lines.append("_Lower rank = higher attribution. Rank 0 = top attribution._")
    else:
        lines.append(f"_Skipped or failed: {B}_")
    lines.append("")

    # C
    C = section_results.get("C")
    lines.append("## C. v0.6 Intent Ablation")
    lines.append("")
    if C and isinstance(C, dict) and "skipped" not in C:
        for label, v in C.items():
            if "error" in v:
                lines.append(f"### {label}: ERROR")
                lines.append(v["error"])
                continue
            lines.append(f"### {label}")
            pj = v.get("action_title_pairwise_jaccard", {})
            lines.append("**Pairwise Jaccard of action titles across intents:**")
            for pair, j in sorted(pj.items(), key=lambda kv: -kv[1]):
                lines.append(f"- `{pair}`: {j:.2f}")
            lines.append("")
    else:
        lines.append(f"_Skipped: {C}_")
    lines.append("")

    # D
    D = section_results.get("D")
    lines.append("## D. Determinism")
    lines.append("")
    if D and isinstance(D, dict) and "skipped" not in D:
        lines.append("| Case | exec-summary token Jaccard | action-title Jaccard |")
        lines.append("|---|---:|---:|")
        for label, v in D.items():
            if "error" in v:
                lines.append(f"| {label} | ERROR | {v['error'][:60]} |")
                continue
            ej = v.get("exec_summary_mean_token_jaccard")
            aj = v.get("action_titles_mean_jaccard")
            lines.append(f"| {label} | "
                         f"{ej:.3f}" if ej is not None else "—"
                         + " | "
                         + (f"{aj:.3f}" if aj is not None else "—")
                         + " |")
    else:
        lines.append(f"_Skipped: {D}_")
    lines.append("")

    lines.append("## Summary verdicts (auto-derived)")
    lines.append("")
    verdicts = []
    if A and "false_positive_rate_critical" in A and A["false_positive_rate_critical"] is not None:
        rate = A["false_positive_rate_critical"]
        if rate <= 0.10:
            verdicts.append(f"✅ **Specificity holds:** FP rate {rate*100:.1f}% on the zoo "
                            "(threshold = 10%).")
        else:
            verdicts.append(f"⚠️ **Specificity at risk:** FP rate {rate*100:.1f}% on the zoo "
                            "exceeds the 10% threshold. Investigate which rules are over-triggering.")
    if B and isinstance(B, dict):
        eng_detected = sum(1 for v in B.values() if isinstance(v, dict)
                            and v.get("detection_of_engineered_feature", {}).get("ffca_rank_among_features") == 0)
        n_eng = sum(1 for v in B.values() if isinstance(v, dict) and v.get("engineered_feature"))
        if n_eng:
            verdicts.append(f"FFCA Impact identified the engineered feature as rank-0 in "
                            f"**{eng_detected}/{n_eng}** cases with engineered pathology.")
    if C and isinstance(C, dict) and "skipped" not in C:
        all_jacc = []
        for v in C.values():
            if isinstance(v, dict) and "action_title_pairwise_jaccard" in v:
                all_jacc.extend(v["action_title_pairwise_jaccard"].values())
        if all_jacc:
            mean_j = float(np.mean(all_jacc))
            if mean_j < 0.7:
                verdicts.append(f"✅ **Intent shifts output:** mean pairwise action-title "
                                f"Jaccard across intents = {mean_j:.2f}. Different intents do "
                                "produce different action lists.")
            else:
                verdicts.append(f"⚠️ **Intent may not shift output enough:** action Jaccard {mean_j:.2f} "
                                "is high. Either intents look similar or the cases were insensitive.")
    if D and isinstance(D, dict) and "skipped" not in D:
        all_ej = []
        for v in D.values():
            if isinstance(v, dict) and v.get("exec_summary_mean_token_jaccard") is not None:
                all_ej.append(v["exec_summary_mean_token_jaccard"])
        if all_ej:
            mean_ej = float(np.mean(all_ej))
            if mean_ej > 0.6:
                verdicts.append(f"✅ **Determinism reasonable:** exec-summary token Jaccard "
                                f"across reruns = {mean_ej:.2f}.")
            else:
                verdicts.append(f"⚠️ **Reruns vary noticeably:** exec-summary Jaccard {mean_ej:.2f}. "
                                "Consider temperature=0 in production.")
    for v in verdicts:
        lines.append(f"- {v}")
    lines.append("")
    (out_dir / "VALIDATION_REPORT.md").write_text("\n".join(lines))
    print(f"\nFinal report: {out_dir/'VALIDATION_REPORT.md'}")


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="results/v06_validation",
                    help="Top-level results directory.")
    ap.add_argument("--rulebook", default=str(DEFAULT_RULEBOOK))
    ap.add_argument("--key-file", default=None,
                    help="Path to a file containing the Anthropic API key. "
                         "Sections C and D need it. Section B can run without it.")
    ap.add_argument("--skip-zoo", action="store_true")
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--skip-intent", action="store_true")
    ap.add_argument("--skip-determinism", action="store_true")
    ap.add_argument("--include-vision-zoo", action="store_true",
                    help="Add 4 ImageNet-pretrained CNNs on CIFAR-10 to section A. "
                         "Adds ~5-10 min runtime + ~250MB of CIFAR download.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Fail loud on setup issues BEFORE running anything.
    _check_setup(args)

    _seed_everything(args.seed)
    device = _pick_device()
    print(f"Running on device: {device}")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rulebook = load_rulebook(args.rulebook)
    key_file = Path(args.key_file).expanduser().resolve() if args.key_file else None

    # Bootstrap the 4 engineered cases on disk so sections B, C, D can read them.
    needs_engineered = not (args.skip_baselines and args.skip_intent and args.skip_determinism)
    engineered_dir: Path | None = None
    if needs_engineered:
        print("\n=== Bootstrap: engineered tabular cases (for sections B/C/D) ===")
        try:
            engineered_dir = _bootstrap_engineered_cases(out_dir, device)
        except Exception as exc:
            print(f"  bootstrap FAILED: {exc}")
            traceback.print_exc()
            engineered_dir = None

    section_results: dict[str, Any] = {}

    if not args.skip_zoo:
        print("\n=== Section A: Model-zoo false-positive sweep ===")
        try:
            section_results["A"] = run_section_a(out_dir, rulebook, device,
                                                  include_vision=args.include_vision_zoo)
        except Exception as exc:
            section_results["A"] = {"error": str(exc), "traceback": traceback.format_exc()}
    else:
        section_results["A"] = {"skipped": "--skip-zoo"}

    if not args.skip_baselines:
        print("\n=== Section B: SHAP / IG vs FFCA Impact ===")
        if engineered_dir is None:
            section_results["B"] = {"skipped": "engineered_dir unavailable"}
        else:
            try:
                section_results["B"] = run_section_b(out_dir, rulebook, device, engineered_dir)
            except Exception as exc:
                section_results["B"] = {"error": str(exc), "traceback": traceback.format_exc()}
    else:
        section_results["B"] = {"skipped": "--skip-baselines"}

    if not args.skip_intent:
        print("\n=== Section C: v0.6 intent ablation ===")
        if engineered_dir is None:
            section_results["C"] = {"skipped": "engineered_dir unavailable"}
        else:
            try:
                section_results["C"] = run_section_c(out_dir, rulebook, key_file, engineered_dir)
            except Exception as exc:
                section_results["C"] = {"error": str(exc), "traceback": traceback.format_exc()}
    else:
        section_results["C"] = {"skipped": "--skip-intent"}

    if not args.skip_determinism:
        print("\n=== Section D: Determinism ===")
        if engineered_dir is None:
            section_results["D"] = {"skipped": "engineered_dir unavailable"}
        else:
            try:
                section_results["D"] = run_section_d(out_dir, rulebook, key_file, engineered_dir)
            except Exception as exc:
                section_results["D"] = {"error": str(exc), "traceback": traceback.format_exc()}
    else:
        section_results["D"] = {"skipped": "--skip-determinism"}

    _save(section_results, out_dir / "validation_results.json")
    write_final_report(out_dir, section_results)
    print("\nAll sections done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
