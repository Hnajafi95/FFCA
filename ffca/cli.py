"""`ffca-report` command-line entry point.

Usage examples:

  Tabular MLP, CSV data:
    ffca-report \\
      --model-class my_pkg.models:MyMLP \\
      --weights ckpt/final.pt \\
      --adapter tabular \\
      --data data.csv \\
      --target-column label \\
      --out out/

  CNN intermediate-layer FFCA across checkpoints:
    ffca-report \\
      --model-class torchvision.models:resnet50 \\
      --weights ckpt/final.pt \\
      --checkpoints ckpt/e10.pt ckpt/e50.pt ckpt/final.pt \\
      --adapter channel --layer layer4.2.conv2 \\
      --data data/imagenet_val/ \\
      --scalar predicted_class \\
      --out out/

The CLI is a thin wrapper around the Python API; the same options map 1-to-1
to `FFCAReport` arguments.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn


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
    """Load a NetCDF file into a DataLoader.

    For pixel/channel adapters: each requested channel becomes one input channel.
    The first axis of the variable is treated as the sample axis; the remaining
    axes are spatial.

    For tabular adapters: each requested variable is a feature; the first axis
    is the sample axis and each variable must be 1-D over it.
    """
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
        X = np.stack(flat, axis=1)  # (n, n_channels)
        feature_names = list(channels)
        if target_col and target_col in ds.data_vars:
            y = _y_from_array(np.asarray(ds[target_col].values))
        else:
            y = torch.zeros(n, dtype=torch.long)
        td = TensorDataset(torch.tensor(X), y)
        return DataLoader(td, batch_size=batch_size), feature_names

    # pixel / channel: stack channels into (N, C, ...)
    X = np.stack(arrays, axis=1).astype(np.float32)  # (n, C, H, W?, ...)
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
    """Load .npy / .npz into a DataLoader.

    .npy: single array. tabular -> (N, F); pixel/channel -> (N, C, H, W).
    .npz: expects key 'X' for inputs and optionally 'y' for targets;
          --target-column overrides the default 'y' key.
    """
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
    # pixel / channel
    if X.ndim < 3:
        raise SystemExit(
            f"pixel/channel adapter expects (N, C, H, W); got {X.shape}"
        )
    input_shape = X.shape[1:]
    y = _y_from_array(y_raw) if y_raw is not None else torch.zeros(
        X.shape[0], dtype=torch.long)
    td = TensorDataset(torch.tensor(X), y)
    return DataLoader(td, batch_size=batch_size), input_shape


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ffca-report",
        description="Run FFCA on any PyTorch model and produce a report + plots.")
    p.add_argument("--model-class", required=True,
                   help="Import path of a class or factory: 'pkg.mod:ClassOrFn'")
    p.add_argument("--model-kwargs", default="{}",
                   help="JSON dict of kwargs to pass to the model factory")
    p.add_argument("--weights",
                   help="Path to a state_dict to load into the freshly built model "
                        "(omit if model factory loads its own weights)")
    p.add_argument("--checkpoints", nargs="*", default=[],
                   help="Optional list of state_dict paths for multi-checkpoint FFCA")
    p.add_argument("--adapter", required=True, choices=["tabular", "pixel", "channel"])
    p.add_argument("--layer", help="Layer dotted name (required for --adapter channel)")
    p.add_argument("--data", required=True,
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
    p.add_argument("--out", required=True, help="Output directory")
    # Sampling budgets
    p.add_argument("--n-samples", type=int, default=32,
                   help="Number of samples used for FFCA estimation (default 32)")
    p.add_argument("--n-probes", type=int, default=64,
                   help="Cauchy-HVP probe count (default 64)")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip PNG plot generation")
    p.add_argument("--no-improvements", action="store_true",
                   help="Skip the 3 audit-v2 improvements: baseline FFCA only "
                        "(correlation-proxy interaction, no Trust Score, no Co-Sensitivity)")
    return p


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

    # ----- model factory ----------------------------------------------------
    cls = _import_model_class(args.model_class)
    kwargs = json.loads(args.model_kwargs)

    def model_factory():
        m = cls(**kwargs) if isinstance(cls, type) else cls(**kwargs)
        if args.weights and not args.checkpoints:
            state = torch.load(args.weights, map_location=device, weights_only=False)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            m.load_state_dict(state, strict=False)
        return m.to(device).eval()

    initial_model = model_factory()

    # ----- data loader + feature info --------------------------------------
    data_path = Path(args.data)
    data_fmt = args.data_format
    if data_fmt == "auto":
        data_fmt = _detect_format(data_path)
    channels = (
        [c.strip() for c in args.data_channels.split(",") if c.strip()]
        if args.data_channels else None
    )
    feature_names = None
    input_shape = None
    if data_fmt == "csv":
        if args.adapter != "tabular":
            sys.exit("ERROR: --data-format csv is only valid with --adapter tabular")
        loader, feature_names = _load_csv_tabular(
            data_path, args.target_column, args.batch_size)
    elif data_fmt == "imagefolder":
        if args.adapter == "tabular":
            sys.exit("ERROR: --data-format imagefolder requires --adapter pixel or channel")
        loader, input_shape = _load_imagefolder(
            data_path, args.image_size, args.batch_size)
    elif data_fmt == "netcdf":
        loader, payload = _load_netcdf(
            data_path, channels, args.adapter, args.batch_size, args.target_column)
        if args.adapter == "tabular":
            feature_names = payload
        else:
            input_shape = payload
    elif data_fmt == "npy":
        loader, payload = _load_npy(
            data_path, args.adapter, args.batch_size, args.target_column)
        if args.adapter == "tabular":
            feature_names = payload
        else:
            input_shape = payload
    else:
        sys.exit(f"unknown data-format {data_fmt}")

    # ----- adapter ----------------------------------------------------------
    from .adapters import ChannelAdapter, PixelAdapter, TabularAdapter
    from .core.scalars import from_name as scalar_from_name
    scalar = scalar_from_name(args.scalar)

    if args.adapter == "tabular":
        adapter = TabularAdapter(initial_model, feature_names=feature_names, scalar=scalar)
    elif args.adapter == "pixel":
        adapter = PixelAdapter(initial_model, input_shape=input_shape, scalar=scalar)
    elif args.adapter == "channel":
        if not args.layer:
            sys.exit("ERROR: --layer is required for --adapter channel")
        adapter = ChannelAdapter(initial_model, layer_name=args.layer, scalar=scalar)
    else:
        sys.exit(f"unknown adapter {args.adapter}")

    # ----- checkpoint loader (optional) ------------------------------------
    from .checkpoint import CheckpointLoader
    from .report import FFCAReport
    ck_loader = None
    if args.checkpoints:
        ck_loader = CheckpointLoader(model_factory,
                                     [(Path(p).stem, p) for p in args.checkpoints],
                                     device=device)

    # ----- run + save ------------------------------------------------------
    report = FFCAReport(
        adapter, loader,
        n_first_order_samples=args.n_samples,
        n_hessian_samples=max(8, args.n_samples // 4),
        n_diag_probes=max(24, args.n_probes // 2),
        n_cauchy_probes=args.n_probes,
        n_cauchy_samples=max(8, args.n_samples // 2),
        improvements=not args.no_improvements,
    )
    report.run(checkpoints=ck_loader)
    out = report.save(args.out, save_plots=not args.no_plots)
    print(f"[ffca-report] wrote {out}/report.md, report.json, plots/")


if __name__ == "__main__":
    main()
