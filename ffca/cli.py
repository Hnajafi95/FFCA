"""`ffca-report` command-line entry point.

Two ways to use it:

  1. Interactive wizard (easiest — pick this if you're new):
       ffca-report --interactive

  2. Flag-driven (good for scripts):
       ffca-report --model-class my_pkg.models:MyMLP \\
                   --weights ckpt/final.pt \\
                   --model-type mlp \\
                   --data data.csv \\
                   --out out/

The interactive wizard scans your model, lists its candidate layers,
prompts you to pick which to analyze, and emits one FFCA report per
layer.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Model class import
# --------------------------------------------------------------------------- #
def _import_model_class(spec: str):
    """Parse `pkg.mod:Class` (or `pkg.mod:factory_fn`) and import it."""
    if ":" not in spec:
        raise ValueError(
            f"--model-class must be 'package.module:ClassOrFactory', got {spec!r}"
        )
    mod_name, attr = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    if not hasattr(mod, attr):
        raise AttributeError(f"{mod_name} has no attribute {attr!r}")
    return getattr(mod, attr)


# --------------------------------------------------------------------------- #
# Data loaders
# --------------------------------------------------------------------------- #
def _detect_format(path: Path) -> str:
    """Resolve --data-format=auto using path inspection."""
    if path.is_dir():
        return "imagefolder"
    suf = path.suffix.lower()
    if suf == ".csv":
        return "csv"
    if suf in (".nc", ".cdf", ".netcdf", ".nc4"):
        return "netcdf"
    if suf in (".npy", ".npz"):
        return "npy"
    raise SystemExit(
        f"--data-format=auto could not infer format for {path}; "
        "pass --data-format explicitly (csv|imagefolder|netcdf|npy)"
    )


def _y_from_array(y_raw):
    import numpy as np
    if np.issubdtype(y_raw.dtype, np.integer) or (y_raw.round() == y_raw).all():
        return torch.tensor(y_raw, dtype=torch.long)
    return torch.tensor(y_raw, dtype=torch.float32)


def _load_csv_tabular(path: Path, target_col: str | None, batch_size: int):
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import DataLoader, TensorDataset
    df = pd.read_csv(path)
    if target_col is None:
        target_col = df.columns[-1]
    feature_names = [c for c in df.columns if c != target_col]
    X = StandardScaler().fit_transform(df[feature_names].to_numpy(dtype=np.float32))
    y = _y_from_array(df[target_col].to_numpy())
    ds = TensorDataset(torch.tensor(X), y)
    return DataLoader(ds, batch_size=batch_size), feature_names


def _load_imagefolder(path: Path, image_size: int, batch_size: int):
    import torchvision
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    tfm = T.Compose([T.Resize((image_size, image_size)), T.ToTensor(),
                     T.Normalize((0.5,) * 3, (0.5,) * 3)])
    ds = torchvision.datasets.ImageFolder(str(path), transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=False), (3, image_size, image_size)


def _load_netcdf(path: Path, channels: list[str] | None, adapter_kind: str,
                 batch_size: int, target_col: str | None):
    import numpy as np
    try:
        import xarray as xr
    except ImportError as exc:
        raise SystemExit(
            "NetCDF support requires xarray. Install with: pip install xarray netCDF4"
        ) from exc
    from torch.utils.data import DataLoader, TensorDataset

    ds = xr.open_dataset(path)
    if not channels:
        channels = [v for v in ds.data_vars if v != target_col]
        if not channels:
            raise SystemExit(
                f"NetCDF file {path} has no usable data_vars (after excluding target)"
            )
    missing = [c for c in channels if c not in ds.data_vars]
    if missing:
        raise SystemExit(f"NetCDF data_vars not found: {missing}")

    arrays = [np.asarray(ds[c].values) for c in channels]
    n = arrays[0].shape[0]
    for a, name in zip(arrays, channels):
        if a.shape[0] != n:
            raise SystemExit(
                f"NetCDF channel {name!r} has {a.shape[0]} samples, expected {n}"
            )

    if adapter_kind == "tabular":
        flat = []
        for a in arrays:
            if a.ndim != 1:
                raise SystemExit(
                    "tabular adapter requires 1-D NetCDF variables, "
                    f"got shape {a.shape}"
                )
            flat.append(a.astype(np.float32))
        X = np.stack(flat, axis=1)
        feature_names = list(channels)
        if target_col and target_col in ds.data_vars:
            y = _y_from_array(np.asarray(ds[target_col].values))
        else:
            y = torch.zeros(n, dtype=torch.long)
        td = TensorDataset(torch.tensor(X), y)
        return DataLoader(td, batch_size=batch_size), feature_names

    X = np.stack(arrays, axis=1).astype(np.float32)
    if X.ndim < 3:
        raise SystemExit(
            f"pixel/channel adapter expects at least 2 spatial dims per sample, "
            f"got NetCDF tensor shape {X.shape}"
        )
    input_shape = X.shape[1:]
    if target_col and target_col in ds.data_vars:
        y = _y_from_array(np.asarray(ds[target_col].values))
    else:
        y = torch.zeros(n, dtype=torch.long)
    td = TensorDataset(torch.tensor(X), y)
    return DataLoader(td, batch_size=batch_size), input_shape


def _load_npy(path: Path, adapter_kind: str, batch_size: int,
              target_col: str | None):
    import numpy as np
    from torch.utils.data import DataLoader, TensorDataset
    if path.suffix.lower() == ".npz":
        archive = np.load(path)
        if "X" not in archive:
            raise SystemExit(f"{path} has no key 'X'; available: {archive.files}")
        X = archive["X"]
        y_key = target_col or "y"
        y_raw = archive[y_key] if y_key in archive.files else None
    else:
        X = np.load(path)
        y_raw = None
    X = X.astype(np.float32, copy=False)
    if adapter_kind == "tabular":
        if X.ndim != 2:
            raise SystemExit(
                f"tabular adapter expects 2-D .npy (N, F); got {X.shape}"
            )
        feature_names = [f"col_{i}" for i in range(X.shape[1])]
        y = _y_from_array(y_raw) if y_raw is not None else torch.zeros(
            X.shape[0], dtype=torch.long)
        td = TensorDataset(torch.tensor(X), y)
        return DataLoader(td, batch_size=batch_size), feature_names
    if X.ndim < 3:
        raise SystemExit(
            f"pixel/channel adapter expects (N, C, H, W); got {X.shape}"
        )
    input_shape = X.shape[1:]
    y = _y_from_array(y_raw) if y_raw is not None else torch.zeros(
        X.shape[0], dtype=torch.long)
    td = TensorDataset(torch.tensor(X), y)
    return DataLoader(td, batch_size=batch_size), input_shape


# --------------------------------------------------------------------------- #
# Layer enumeration (for interactive wizard)
# --------------------------------------------------------------------------- #
_INTERESTING_LAYER_TYPES = {
    "Conv1d", "Conv2d", "Conv3d",
    "Linear",
    "MultiheadAttention",
    "TransformerEncoderLayer", "TransformerDecoderLayer",
    "LSTM", "GRU", "RNN",
}


def _enumerate_layers(model: nn.Module) -> list[tuple[int, str, str]]:
    """List (index, dotted_name, type_name) for analyzable layers."""
    rows = []
    for name, mod in model.named_modules():
        if not name:
            continue
        t = type(mod).__name__
        if t in _INTERESTING_LAYER_TYPES:
            rows.append((len(rows), name, t))
    return rows


# --------------------------------------------------------------------------- #
# Interactive wizard
# --------------------------------------------------------------------------- #
def _ask(prompt: str, default: Optional[str] = None,
         choices: Optional[list[str]] = None) -> str:
    suffix = ""
    if choices:
        suffix = f" [{'/'.join(choices)}]"
    if default is not None:
        suffix += f" (default: {default})"
    while True:
        raw = input(f"{prompt}{suffix}: ").strip()
        if not raw and default is not None:
            return default
        if choices and raw not in choices:
            print(f"  Please pick one of: {', '.join(choices)}")
            continue
        if raw:
            return raw


def _wizard() -> dict:
    """Walk the user through everything; return a dict of resolved choices."""
    print()
    print("=" * 60)
    print("FFCA Interactive Wizard")
    print("=" * 60)
    print()
    print("This will walk you through producing an FFCA report for your")
    print("trained PyTorch model. You'll need:")
    print("  1) An importable Python class that defines your model")
    print("  2) A .pt / .pth file with the trained weights")
    print("  3) The data your model was trained on (or held-out data from")
    print("     the same distribution)")
    print()

    # ---- Step 1: model class --------------------------------------------- #
    print("Step 1 — Where is your model defined?")
    print("  Provide it as 'package.module:ClassName'")
    print("  Example: 'mypkg.models:MyCNN'")
    print("  (The package must be importable in this Python environment.)")
    model_class = _ask("  Model class")

    # ---- Step 2: weights ------------------------------------------------- #
    print()
    print("Step 2 — Path to the trained weights (.pt or .pth)")
    weights = _ask("  Weights path")

    # ---- load it for inspection ------------------------------------------ #
    print()
    print("Loading model ...")
    try:
        cls = _import_model_class(model_class)
        model = cls() if isinstance(cls, type) else cls()
        state = torch.load(weights, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  ✓ Loaded {type(model).__name__} "
              f"({n_params:,} parameters)")
    except Exception as e:
        sys.exit(f"  ✗ Could not load model: {e}")

    # ---- Step 3: model type --------------------------------------------- #
    print()
    print("Step 3 — What kind of model is this?")
    print("  1) MLP / tabular — analyze input features (rows of a CSV)")
    print("  2) CNN / image    — analyze pixels or internal feature channels")
    print("  3) Transformer / LLM — analyze input embeddings (use Python API for now)")
    mt = _ask("  Pick", choices=["1", "2", "3"])
    model_type = {"1": "mlp", "2": "cnn", "3": "transformer"}[mt]
    if model_type == "transformer":
        sys.exit(
            "\n  Transformer adapter via CLI is not yet supported in the "
            "interactive wizard.\n  Use the Python API:\n"
            "      from ffca import TransformerEmbeddingAdapter, FFCAReport\n"
            "      adapter = TransformerEmbeddingAdapter(model)\n"
            "      FFCAReport(adapter, loader).run().save('out/')"
        )

    # ---- Step 4: adapter mode ------------------------------------------- #
    jobs: list[tuple[str, str, Optional[str]]] = []  # (label, adapter_kind, layer_name)
    if model_type == "mlp":
        jobs.append(("mlp", "tabular", None))
    else:  # cnn
        print()
        print("Step 4 — CNN analysis mode")
        print("  1) Pixel-level   — which INPUT pixels does the model use?")
        print("  2) Channel-level — which feature channels matter inside the network?")
        cm = _ask("  Pick", choices=["1", "2"])
        if cm == "1":
            jobs.append(("pixel", "pixel", None))
        else:
            layers = _enumerate_layers(model)
            if not layers:
                sys.exit("  ✗ No analyzable layers (Conv/Linear/MHA) found in this model.")
            print()
            print(f"  Found {len(layers)} candidate layer(s) in your model:")
            print()
            for idx, name, t in layers:
                print(f"    [{idx:3d}] {t:15s} {name}")
            print()
            print("  Pick one or more layer indices to investigate.")
            print("  Examples: '0' (just the first); '0,3,5' (three layers);")
            print("            'all' (every candidate layer — slower).")
            picks = _ask("  Layer indices")
            if picks.lower() == "all":
                chosen = [i for i, _, _ in layers]
            else:
                try:
                    chosen = [int(x.strip()) for x in picks.split(",") if x.strip()]
                except ValueError:
                    sys.exit("  ✗ Could not parse layer indices.")
            for ci in chosen:
                if ci < 0 or ci >= len(layers):
                    sys.exit(f"  ✗ Layer index {ci} out of range (have {len(layers)}).")
                _, name, _ = layers[ci]
                # safe label for output directory naming
                label = "ch_" + name.replace(".", "_")
                jobs.append((label, "channel", name))

    # ---- Step 5: data --------------------------------------------------- #
    print()
    print("Step 5 — Path to the data")
    print()
    print("  IMPORTANT: This MUST be the same data your model was trained on,")
    print("             or held-out data from the same distribution. FFCA")
    print("             computes derivatives against your trained model — if")
    print("             you pass unrelated data, the signature is meaningless.")
    print()
    print("  Supported formats:")
    print("    • CSV file        (MLP / tabular)")
    print("    • ImageFolder dir (CNN — torchvision layout)")
    print("    • .npy / .npz     (any adapter)")
    print("    • NetCDF .nc      (scientific data, needs `pip install ffca[netcdf]`)")
    data = _ask("  Data path")

    # ---- Step 6: output -------------------------------------------------- #
    print()
    print("Step 6 — Where to write reports")
    out = _ask("  Output directory", default="ffca_out")

    # ---- Step 7: optional CSV target column ----------------------------- #
    target_column = None
    if Path(data).suffix.lower() == ".csv":
        print()
        target_column = _ask(
            "  (Optional) target column in your CSV "
            "(press Enter to use the last column)", default=""
        ) or None

    print()
    print("=" * 60)
    print(f"Ready to run {len(jobs)} FFCA report(s):")
    for label, kind, layer in jobs:
        if layer:
            print(f"  • {label}: {kind} adapter on layer '{layer}'")
        else:
            print(f"  • {label}: {kind} adapter")
    print("=" * 60)
    print()

    return {
        "model_class": model_class,
        "weights": weights,
        "data": data,
        "out": out,
        "target_column": target_column,
        "jobs": jobs,
    }


# --------------------------------------------------------------------------- #
# CLI parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ffca-report",
        description="Run FFCA on any PyTorch model and produce a report + plots.")
    p.add_argument("--interactive", action="store_true",
                   help="Run the guided wizard instead of using flags. "
                        "Lists your model's layers, asks which to investigate.")
    p.add_argument("--model-class",
                   help="Import path of a class or factory: 'pkg.mod:ClassOrFn'")
    p.add_argument("--model-kwargs", default="{}",
                   help="JSON dict of kwargs to pass to the model factory")
    p.add_argument("--weights",
                   help="Path to a state_dict to load into the freshly built model "
                        "(omit if model factory loads its own weights)")
    p.add_argument("--checkpoints", nargs="*", default=[],
                   help="Optional list of state_dict paths for multi-checkpoint FFCA")
    p.add_argument("--model-type", choices=["mlp", "cnn", "transformer"],
                   help="High-level model type. Sets a sensible --adapter for you: "
                        "mlp→tabular, cnn→pixel (or channel if --layer given).")
    p.add_argument("--adapter", choices=["tabular", "pixel", "channel"],
                   help="Adapter to use. If omitted, derived from --model-type.")
    p.add_argument("--layer", help="Layer dotted name (required for --adapter channel)")
    p.add_argument("--data",
                   help="CSV file, ImageFolder root, NetCDF (.nc) file, or .npy/.npz array")
    p.add_argument("--data-format", default="auto",
                   choices=["auto", "csv", "imagefolder", "netcdf", "npy"],
                   help="Input format. 'auto' infers from --data extension/dirness.")
    p.add_argument("--data-channels", default=None,
                   help="(netcdf) comma-separated list of variables to use as channels/features")
    p.add_argument("--target-column",
                   help="(csv/netcdf) target column or variable name. For .npz, the key for targets (default 'y').")
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--scalar", default="predicted_class",
                   help="predicted_class | true_label | target_class:N | "
                        "regression[:dim] | loss[:ce|mse|bce]")
    p.add_argument("--device", default=None,
                   help="cpu | cuda | mps (auto if omitted)")
    p.add_argument("--out", help="Output directory")
    p.add_argument("--n-samples", type=int, default=32,
                   help="Number of samples used for FFCA estimation (default 32)")
    p.add_argument("--n-probes", type=int, default=64,
                   help="Cauchy-HVP probe count (default 64)")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip PNG plot generation")
    p.add_argument("--no-improvements", action="store_true",
                   help="Skip the 3 audit-v2 improvements: baseline FFCA only")
    return p


# --------------------------------------------------------------------------- #
# Single-run driver (shared by wizard + flag flow)
# --------------------------------------------------------------------------- #
def _run_one(*, model_factory, adapter_kind: str, layer_name: Optional[str],
             data_path: Path, data_fmt: str, data_channels: Optional[list[str]],
             target_column: Optional[str], image_size: int, batch_size: int,
             scalar_name: str, device: torch.device, out_dir: Path,
             n_samples: int, n_probes: int, save_plots: bool,
             improvements: bool, checkpoints: list[str]) -> None:
    initial_model = model_factory()

    feature_names = None
    input_shape = None
    if data_fmt == "csv":
        if adapter_kind != "tabular":
            sys.exit("ERROR: csv data is only valid with the tabular adapter")
        loader, feature_names = _load_csv_tabular(
            data_path, target_column, batch_size)
    elif data_fmt == "imagefolder":
        if adapter_kind == "tabular":
            sys.exit("ERROR: imagefolder data requires the pixel or channel adapter")
        loader, input_shape = _load_imagefolder(
            data_path, image_size, batch_size)
    elif data_fmt == "netcdf":
        loader, payload = _load_netcdf(
            data_path, data_channels, adapter_kind, batch_size, target_column)
        if adapter_kind == "tabular":
            feature_names = payload
        else:
            input_shape = payload
    elif data_fmt == "npy":
        loader, payload = _load_npy(
            data_path, adapter_kind, batch_size, target_column)
        if adapter_kind == "tabular":
            feature_names = payload
        else:
            input_shape = payload
    else:
        sys.exit(f"unknown data-format {data_fmt}")

    from .adapters import ChannelAdapter, PixelAdapter, TabularAdapter
    from .core.scalars import from_name as scalar_from_name
    scalar = scalar_from_name(scalar_name)

    if adapter_kind == "tabular":
        adapter = TabularAdapter(initial_model, feature_names=feature_names, scalar=scalar)
    elif adapter_kind == "pixel":
        adapter = PixelAdapter(initial_model, input_shape=input_shape, scalar=scalar)
    elif adapter_kind == "channel":
        if not layer_name:
            sys.exit("ERROR: --layer is required for --adapter channel")
        adapter = ChannelAdapter(initial_model, layer_name=layer_name, scalar=scalar)
    else:
        sys.exit(f"unknown adapter {adapter_kind}")

    from .checkpoint import CheckpointLoader
    from .report import FFCAReport
    ck_loader = None
    if checkpoints:
        ck_loader = CheckpointLoader(
            model_factory,
            [(Path(p).stem, p) for p in checkpoints],
            device=device,
        )

    report = FFCAReport(
        adapter, loader,
        n_first_order_samples=n_samples,
        n_hessian_samples=max(8, n_samples // 4),
        n_diag_probes=max(24, n_probes // 2),
        n_cauchy_probes=n_probes,
        n_cauchy_samples=max(8, n_samples // 2),
        improvements=improvements,
    )
    report.run(checkpoints=ck_loader)
    out = report.save(out_dir, save_plots=save_plots)
    print(f"[ffca-report] wrote {out}/report.md, report.json, plots/")
    if report.findings:
        print("[ffca-report] findings:")
        for f in report.findings:
            print(f"    {f.severity:8s} {f.name:25s} {f.headline}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def _adapter_from_model_type(model_type: str, layer: Optional[str]) -> str:
    if model_type == "mlp":
        return "tabular"
    if model_type == "cnn":
        return "channel" if layer else "pixel"
    if model_type == "transformer":
        sys.exit(
            "ERROR: --model-type transformer has no CLI flow yet. "
            "Use the Python API: `from ffca import TransformerEmbeddingAdapter`"
        )
    sys.exit(f"unknown --model-type {model_type}")


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)

    # ----- device autodetect ------------------------------------------------
    if args.device is None:
        if torch.backends.mps.is_available():
            args.device = "mps"
        elif torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"
    device = torch.device(args.device)
    print(f"[ffca-report] device={device}")

    if args.interactive:
        wiz = _wizard()
        cls = _import_model_class(wiz["model_class"])
        weights = wiz["weights"]

        def factory():
            m = cls() if isinstance(cls, type) else cls()
            state = torch.load(weights, map_location=device, weights_only=False)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            m.load_state_dict(state, strict=False)
            return m.to(device).eval()

        data_path = Path(wiz["data"])
        data_fmt = _detect_format(data_path) if args.data_format == "auto" else args.data_format
        out_root = Path(wiz["out"])
        out_root.mkdir(parents=True, exist_ok=True)

        for label, adapter_kind, layer_name in wiz["jobs"]:
            sub = out_root if len(wiz["jobs"]) == 1 else (out_root / label)
            print()
            print(f"[ffca-report] === running '{label}' → {sub} ===")
            _run_one(
                model_factory=factory,
                adapter_kind=adapter_kind,
                layer_name=layer_name,
                data_path=data_path,
                data_fmt=data_fmt,
                data_channels=None,
                target_column=wiz["target_column"],
                image_size=args.image_size,
                batch_size=args.batch_size,
                scalar_name=args.scalar,
                device=device,
                out_dir=sub,
                n_samples=args.n_samples,
                n_probes=args.n_probes,
                save_plots=not args.no_plots,
                improvements=not args.no_improvements,
                checkpoints=args.checkpoints,
            )
        return

    # ----- flag-driven flow --------------------------------------------------
    if not args.model_class:
        sys.exit("ERROR: --model-class is required (or use --interactive)")
    if not args.data:
        sys.exit("ERROR: --data is required (or use --interactive)")
    if not args.out:
        sys.exit("ERROR: --out is required (or use --interactive)")

    if args.adapter is None:
        if args.model_type is None:
            sys.exit("ERROR: pass either --adapter or --model-type "
                     "(or use --interactive)")
        args.adapter = _adapter_from_model_type(args.model_type, args.layer)

    cls = _import_model_class(args.model_class)
    kwargs = json.loads(args.model_kwargs)

    def factory():
        m = cls(**kwargs) if isinstance(cls, type) else cls(**kwargs)
        if args.weights and not args.checkpoints:
            state = torch.load(args.weights, map_location=device, weights_only=False)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            m.load_state_dict(state, strict=False)
        return m.to(device).eval()

    data_path = Path(args.data)
    data_fmt = args.data_format
    if data_fmt == "auto":
        data_fmt = _detect_format(data_path)
    data_channels = (
        [c.strip() for c in args.data_channels.split(",") if c.strip()]
        if args.data_channels else None
    )

    _run_one(
        model_factory=factory,
        adapter_kind=args.adapter,
        layer_name=args.layer,
        data_path=data_path,
        data_fmt=data_fmt,
        data_channels=data_channels,
        target_column=args.target_column,
        image_size=args.image_size,
        batch_size=args.batch_size,
        scalar_name=args.scalar,
        device=device,
        out_dir=Path(args.out),
        n_samples=args.n_samples,
        n_probes=args.n_probes,
        save_plots=not args.no_plots,
        improvements=not args.no_improvements,
        checkpoints=args.checkpoints,
    )


if __name__ == "__main__":
    main()
