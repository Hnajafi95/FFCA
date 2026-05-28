#!/usr/bin/env python3
"""
compound_flooding_extra_validation.py
======================================

Single-file extra-validation suite for the compound-flooding case study,
designed to close the remaining reviewer gaps in the FFCA paper. Runs end-
to-end on one or more H200 GPUs; output is one tarball-friendly directory.

Seven sections, gated by --sections. A/B are free (no GPU). C/D/E/F/G
retrain 30-seed Keras MLP ensembles under the same protocol as the
original deployment (counterfactual_retraining.py), so results are
directly comparable.

  A. Tail-event re-evaluation        — operational relevance: extreme floods
  B. Permutation importance          — values matter, not just columns
  C. Attribution-baseline backbones  — SHAP / IG / raw-Impact vs FFCA CK
  D. Archetype ablation              — validates the 4D→8-archetype taxonomy
  E. Pair-synergy isolation          — validates the Interaction dimension
  F. Nonlinearity gap                — validates the Nonlinearity dimension
  G. Volatility–uncertainty          — validates the Volatility dimension

Usage on Proxima3 (7× H200), simplest:

    # 1) Single-process run, all sections on one GPU (~30 wall-clock hours):
    CUDA_VISIBLE_DEVICES=0 python -u compound_flooding_extra_validation.py \
        --output-dir extra_validation_runs \
        --sections A,B,C,D,E,F,G

    # 2) Sharded across 7 GPUs (~4-6 hours wall):
    for k in 0 1 2 3 4 5 6; do
        CUDA_VISIBLE_DEVICES=$k python -u compound_flooding_extra_validation.py \
            --output-dir extra_validation_runs \
            --sections A,B,C,D,E,F,G \
            --shard ${k}/7 \
            --gpu-id ${k} > shard_${k}.log 2>&1 &
    done
    wait

    # 3) After all shards finish, generate the final report:
    python compound_flooding_extra_validation.py \
        --output-dir extra_validation_runs \
        --finalize-only

Outputs (in --output-dir):
    section_{A,B,C,D,E,F,G}.csv     — per-test result tables
    MANIFEST.json                    — what ran, with which seeds, on which GPU
    FINAL_REPORT.md                  — auto-derived verdicts per section
    runs/<section>/<exp>/<variant>/  — per-run .h5 ensembles (resumable)

Sanity-check before launch (Phase G regression):
    python compound_flooding_extra_validation.py --sanity-check
    -> expects 24h-gate ensemble RMSE = 3.37 cm ± 0.05 (paper 3.40)

Resume policy: every section writes intermediate JSON; existing .h5 files
are reloaded; existing per-job results are skipped. Safe to relaunch.

References:
    - Training protocol: counterfactual_retraining.py (variant A/B)
    - Feature ordering: Phase G fix, input_specifications order
    - FFCA reports: FFCA_resutls_before_prunning_ensemble/<cat>/<exp>/report.json
    - Backbone definitions: report['trust'] decision strings
"""

# ──────────────────────────────────────────────────────────────────────────
# Module setup
# ──────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Optional, Sequence

warnings.filterwarnings('ignore')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import pandas as pd

# Deferred TF/Keras imports — done after CUDA_VISIBLE_DEVICES is honored
# and after the --gpu-id flag is parsed.
tf = None
keras = None


# ──────────────────────────────────────────────────────────────────────────
# Paths — auto-detect Mac dev layout vs HPC bundle layout
# ──────────────────────────────────────────────────────────────────────────
# Several path roots are tracked separately because the HPC has the files
# split across three locations:
#   ORIGINALS_ROOT        — original 30-seed Keras ensembles (un-pruned)
#                           Mac: compound_flooding/mlmiamicompoundfloodpredictions/
#                           HPC: ~/FFCA/compound_flooding_originals/
#   BUNDLE_ROOT           — variant_D ensembles + CSV
#                           Mac: FFCA_agent/case_studies/feature_perturbation_runs/
#                                (sibling, with CSV pulled from compound_flooding)
#                           HPC: ~/FFCA/feature_perturbation_bundle_post_audit/
#   REPORTS_ROOT          — corrected ensemble FFCA reports (post-Phase G)
#                           Mac: compound_flooding/FFCA_resutls_before_prunning_ensemble/
#                           HPC: ~/FFCA/FFCA_resutls_before_prunning_ensemble/
def _detect_paths():
    candidates_originals = [
        Path('/Users/hnaja002/Documents/projects/compound_flooding/mlmiamicompoundfloodpredictions'),
        Path.home() / 'FFCA' / 'compound_flooding_originals',
        Path.home() / 'compound_flooding_originals',
    ]
    originals_root = next((p for p in candidates_originals if p.exists()),
                           candidates_originals[0])

    candidates_bundle = [
        Path.home() / 'FFCA' / 'feature_perturbation_bundle_post_audit',
        Path('/Users/hnaja002/Documents/projects/FFCA_agent/case_studies'),
    ]
    bundle_root = next((p for p in candidates_bundle if p.exists()),
                        candidates_bundle[0])

    candidates_reports = [
        Path('/Users/hnaja002/Documents/projects/compound_flooding/FFCA_resutls_before_prunning_ensemble'),
        Path.home() / 'FFCA' / 'FFCA_resutls_before_prunning_ensemble',
    ]
    reports_root = next((p for p in candidates_reports if p.exists()),
                         candidates_reports[0])

    # Path to the data CSV — prefer bundle, fall back to Mac layout
    candidates_csv = [
        bundle_root / 'data' / 'Miami_GWL_WL_RAIN_GATE_2017_2024.csv',
        Path('/Users/hnaja002/Documents/projects/compound_flooding/mlmiamicompoundfloodpredictions/Miami_GWL_WL_RAIN_GATE_2017_2024.csv'),
        Path('/Users/hnaja002/Documents/projects/compound_flooding/mlmiamicompoundfloodpredictions/data/Merged/Miami_GWL_WL_RAIN_GATE_2017_2024.csv'),
    ]
    csv_path = next((p for p in candidates_csv if p.exists()), candidates_csv[0])

    return originals_root, bundle_root, reports_root, csv_path


ORIGINALS_ROOT, BUNDLE_ROOT, REPORTS_ROOT, CSV_PATH = _detect_paths()
# Legacy alias retained for backwards-compat with helpers that referenced CF_ROOT
CF_ROOT = ORIGINALS_ROOT.parent
FFCA_PKG = (Path.home() / 'FFCA') if (Path.home() / 'FFCA').exists() else Path('/Users/hnaja002/Documents/projects/FFCA/FFCA_package')
FFCA_AGENT = (Path.home() / 'FFCA_agent') if (Path.home() / 'FFCA_agent').exists() else Path('/Users/hnaja002/Documents/projects/FFCA_agent')


# ──────────────────────────────────────────────────────────────────────────
# Constants matched to the original training protocol
# ──────────────────────────────────────────────────────────────────────────
N_ENSEMBLE = 30
MAX_EPOCHS = 10_000
BATCH_SIZE = 64
Y_BUFFER = 0.20
VAL_YEAR = 2023
TEST_YEAR = 2024
DEFAULT_PATIENCE = 20   # the original deployment used 20; variant_A used 100
TARGET_COL = 'gwl'

# Tail thresholds
TAIL_TOP_PCT = [0.05, 0.01]   # top 5% and top 1% test events

# Experiment registry — covers all 20 deployment runs
EXPERIMENTS: dict[str, tuple[str, int]] = {
    # Measurements Only
    '3hr_measured_sigmoid':                  ('Measurements Only', 3),
    '6hr_measured_sigmoid':                  ('Measurements Only', 6),
    '12hr_measured_sigmoid':                 ('Measurements Only', 12),
    '24hr_measured_sigmoid':                 ('Measurements Only', 24),
    # Predicted Ocean Water Levels (WLS)
    '3hr_perfect_prog_wls_sigmoid':          ('Predicted Ocean Water Levels', 3),
    '6hr_perfect_prog_wls_sigmoid':          ('Predicted Ocean Water Levels', 6),
    '12hr_perfect_prog_wls_sigmoid':         ('Predicted Ocean Water Levels', 12),
    '24hr_perfect_prog_wls_sigmoid':         ('Predicted Ocean Water Levels', 24),
    # Predicted Rainfall
    '3hr_perfect_prog_rain_sigmoid':         ('Predicted Rainfall', 3),
    '6hr_perfect_prog_rain_sigmoid':         ('Predicted Rainfall', 6),
    '12hr_perfect_prog_rain_sigmoid':        ('Predicted Rainfall', 12),
    '24hr_perfect_prog_rain_sigmoid':        ('Predicted Rainfall', 24),
    # Predicted Gate Opening
    '3hr_perfect_prog_gate_sigmoid':         ('Predicted Gate Opening', 3),
    '6hr_perfect_prog_gate_sigmoid':         ('Predicted Gate Opening', 6),
    '12hr_perfect_prog_gate_sigmoid':        ('Predicted Gate Opening', 12),
    '24hr_perfect_prog_gate_sigmoid':        ('Predicted Gate Opening', 24),
    # Predictions All Inputs
    '3hr_perfect_prog_all_inputs_sigmoid':   ('Predictions All Inputs', 3),
    '6hr_perfect_prog_all_inputs_sigmoid':   ('Predictions All Inputs', 6),
    '12hr_perfect_prog_all_inputs_sigmoid':  ('Predictions All Inputs', 12),
    '24hr_perfect_prog_all_inputs_sigmoid':  ('Predictions All Inputs', 24),
}

DEGRADED_EXPS = [
    '12hr_perfect_prog_gate_sigmoid',
    '24hr_perfect_prog_gate_sigmoid',
    '24hr_perfect_prog_all_inputs_sigmoid',
]

# Representative subset for compute-heavy sections (C/D/G)
REPRESENTATIVE_EXPS = [
    '24hr_perfect_prog_gate_sigmoid',           # degraded, long-lead
    '24hr_perfect_prog_all_inputs_sigmoid',     # degraded, biggest feature space
    '12hr_perfect_prog_rain_sigmoid',           # healthy, rain-driven
]

# Single demo experiment for E/F (synergy / nonlinearity gap)
DEMO_EXP = '24hr_perfect_prog_gate_sigmoid'


# ──────────────────────────────────────────────────────────────────────────
# Lazy TF/Keras import — must happen AFTER CUDA_VISIBLE_DEVICES is set
# ──────────────────────────────────────────────────────────────────────────
def _import_tf():
    global tf, keras
    if tf is not None:
        return
    import tensorflow as _tf
    import keras as _keras
    tf, keras = _tf, _keras

    # Memory growth so multiple shards on one node don't OOM each other
    try:
        gpus = tf.config.list_physical_devices('GPU')
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

    try:
        keras.config.enable_unsafe_deserialization()
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────
# Serializable denormalize layer — matches counterfactual_retraining.py
# ──────────────────────────────────────────────────────────────────────────
def _register_denorm():
    from keras.layers import Layer
    from keras.saving import register_keras_serializable

    @register_keras_serializable(package='counterfactual')
    class DenormalizeSigmoid(Layer):
        def __init__(self, y_min, y_max, **kwargs):
            super().__init__(**kwargs)
            self.y_min = float(y_min)
            self.y_max = float(y_max)

        def call(self, x):
            return x * (self.y_max - self.y_min) + self.y_min

        def get_config(self):
            cfg = super().get_config()
            cfg.update(dict(y_min=self.y_min, y_max=self.y_max))
            return cfg

    return DenormalizeSigmoid


# ──────────────────────────────────────────────────────────────────────────
# Experiment "spec" — replaced by features_from_experiment() helper to
# avoid depending on JSON spec files (matches the HPC bundle convention).
# ──────────────────────────────────────────────────────────────────────────
BASE_CHANNELS = ['gwl', 'wl', 'rain', 'stgH', 'stgT', 'gate1', 'gate2']


def features_from_experiment(name: str, lead: int) -> list[str]:
    """Canonical Phase-G feature ordering (input_specifications order).

    Channels appear in BASE_CHANNELS order; within each channel, lag columns
    are sorted by numeric lag with lag-0 interleaved between -1 and +1.

    Mirrors ~/FFCA/feature_perturbation_bundle_post_audit/feature_perturbation_retraining.py.
    """
    if 'perfect_prog_all_inputs' in name:
        pp = {'wl', 'rain', 'gate1', 'gate2'}
    elif 'perfect_prog_gate' in name:
        pp = {'gate1', 'gate2'}
    elif 'perfect_prog_wls' in name:
        pp = {'wl'}
    elif 'perfect_prog_rain' in name:
        pp = {'rain'}
    elif 'measured' in name:
        pp = set()
    else:
        raise ValueError(f'Unknown experiment name pattern: {name}')

    cols: list[str] = []
    for ch in BASE_CHANNELS:
        hi = lead if ch in pp else 0
        for lag in range(-24, hi + 1):
            if lag == 0:
                cols.append(ch)
            elif lag < 0:
                cols.append(f'{ch}_t{lag}')
            else:
                cols.append(f'{ch}_t+{lag}')
    return cols


def find_experiment_spec(exp_name: str) -> dict:
    """Compatibility shim — returns a synthetic spec dict produced from
    features_from_experiment(). Some helpers still want a dict-like spec.
    """
    cat, lead = EXPERIMENTS[exp_name]
    return dict(
        experiment_name=exp_name,
        category=cat,
        lead_time=lead,
        target_column=TARGET_COL,
        feat_cols=features_from_experiment(exp_name, lead),
    )


def recover_hyperparams(exp_name: str) -> dict:
    """Read layer count / neurons / lr from the first saved .h5 of the original ensemble.

    On HPC the originals live under ORIGINALS_ROOT/<category>/<exp>/MLP/models/.
    Falls back to canonical defaults if no original .h5 is available.
    """
    cat, _ = EXPERIMENTS[exp_name]
    candidates = [
        ORIGINALS_ROOT / cat / exp_name / 'MLP' / 'models' / 'hypermodel1.h5',
        ORIGINALS_ROOT / cat / exp_name / 'models' / 'hypermodel1.h5',
    ]
    model_path = next((p for p in candidates if p.exists()), None)
    if model_path is None:
        # Fallback defaults match the bundle's feature_perturbation_retraining.py
        return dict(num_layers=1, neurons=100, lr=0.001, activation='relu')

    try:
        import h5py
        with h5py.File(model_path, 'r') as f:
            cfg_raw = f.attrs['model_config']
            if isinstance(cfg_raw, bytes):
                cfg_raw = cfg_raw.decode()
            cfg = json.loads(cfg_raw)
            tc_raw = f.attrs['training_config']
            if isinstance(tc_raw, bytes):
                tc_raw = tc_raw.decode()
            tc = json.loads(tc_raw)
        dense_units = [
            l['config']['units']
            for l in cfg['config']['layers']
            if l['class_name'] == 'Dense' and l['config']['units'] != 1
        ]
        return dict(
            num_layers=len(dense_units),
            neurons=int(dense_units[0]) if dense_units else 100,
            lr=float(tc['optimizer_config']['config']['learning_rate']),
            activation='relu',
        )
    except Exception:
        return dict(num_layers=1, neurons=100, lr=0.001, activation='relu')


def original_models_dir(exp_name: str) -> Path:
    """Path to the original 30-seed Keras ensemble for exp_name. Raises if missing."""
    cat, _ = EXPERIMENTS[exp_name]
    for sub in (ORIGINALS_ROOT / cat / exp_name / 'MLP' / 'models',
                ORIGINALS_ROOT / cat / exp_name / 'models'):
        if sub.exists() and any(sub.glob('hypermodel*.h5')):
            return sub
    raise FileNotFoundError(f'No original 30-seed ensemble dir found for {exp_name} under {ORIGINALS_ROOT}')


def variant_d_models_dir(exp_name: str) -> Optional[Path]:
    """Path to the variant_D backbone-removal 30-seed ensemble, if present."""
    candidates = [
        BUNDLE_ROOT / 'feature_perturbation_runs' / exp_name / 'variant_D_backbone_removal' / 'MLP' / 'models',
        FFCA_AGENT / 'case_studies' / 'feature_perturbation_runs' / exp_name / 'variant_D_backbone_removal' / 'MLP' / 'models',
    ]
    for p in candidates:
        if p.exists() and any(p.glob('hypermodel*.h5')):
            return p
    return None


# ──────────────────────────────────────────────────────────────────────────
# Feature-matrix construction — Phase G correct (input_specifications order)
# ──────────────────────────────────────────────────────────────────────────
def build_full_feature_matrix(spec: dict) -> tuple[pd.DataFrame, list[str], str, float, float]:
    """Build the full feature matrix for an experiment.

    Uses the canonical Phase-G feature ordering via features_from_experiment().
    Reads the CSV from CSV_PATH (auto-detected) and computes lagged columns
    on the fly. Returns (df, feat_cols, target, y_min, y_max_buffered).
    """
    df_raw = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
    target_col_name = spec['target_column']
    lead_time = spec['lead_time']
    feat_cols = spec.get('feat_cols') or features_from_experiment(spec['experiment_name'], lead_time)

    df = df_raw.copy()
    for c in feat_cols:
        if c in df.columns:
            continue
        # Parse "<channel>_t-<lag>" or "<channel>_t+<lag>"
        if '_t' in c:
            ch, lag_str = c.split('_t', 1)
            lag = int(lag_str.replace('+', ''))   # e.g. "+1" or "-3"
        else:
            ch, lag = c, 0
        if ch not in df.columns:
            continue
        df[c] = df[ch].shift(-lag) if lag != 0 else df[ch]

    target = f'{target_col_name}_t+{lead_time}'
    if target not in df.columns:
        df[target] = df[target_col_name].shift(-lead_time)
    df = df[feat_cols + [target]].dropna()

    y_min = float(df_raw[target_col_name].min())
    y_max_buffered = float(df_raw[target_col_name].max() * (1 + Y_BUFFER))

    return df, feat_cols, target, y_min, y_max_buffered


def split_by_year(df: pd.DataFrame, target: str, feat_cols: list[str], year: int):
    mask = df.index.year == year
    X = df.loc[mask, feat_cols].to_numpy(dtype=np.float32)
    y = df.loc[mask, target].to_numpy(dtype=np.float32)
    return X, y


def split_train_val_test(df: pd.DataFrame, target: str, feat_cols: list[str],
                          val_year: int = VAL_YEAR, test_year: int = TEST_YEAR):
    test_mask = df.index.year == test_year
    val_mask = df.index.year == val_year
    train_mask = ~test_mask & ~val_mask
    cols = feat_cols
    return (
        df.loc[train_mask, cols].to_numpy(dtype=np.float32),
        df.loc[train_mask, target].to_numpy(dtype=np.float32),
        df.loc[val_mask, cols].to_numpy(dtype=np.float32),
        df.loc[val_mask, target].to_numpy(dtype=np.float32),
        df.loc[test_mask, cols].to_numpy(dtype=np.float32),
        df.loc[test_mask, target].to_numpy(dtype=np.float32),
        df.index[test_mask],
    )


# ──────────────────────────────────────────────────────────────────────────
# Model construction + training (matches counterfactual_retraining.py)
# ──────────────────────────────────────────────────────────────────────────
def build_model(hp: dict, y_min: float, y_max_buffered: float):
    from keras.layers import Dense, Dropout
    from keras.models import Sequential
    from keras.optimizers import Adam
    DenormalizeSigmoid = _register_denorm()

    model = Sequential()
    for _ in range(hp['num_layers']):
        model.add(Dense(hp['neurons'], kernel_initializer='he_normal',
                        activation=hp['activation']))
        model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    model.add(DenormalizeSigmoid(y_min=y_min, y_max=y_max_buffered))
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=hp['lr']))
    return model


def train_ensemble(X_train, y_train, X_val, y_val,
                    hp: dict, y_min: float, y_max_buffered: float,
                    models_dir: Path, n: int = N_ENSEMBLE,
                    patience: int = DEFAULT_PATIENCE,
                    epochs: int = MAX_EPOCHS, batch: int = BATCH_SIZE,
                    verbose: int = 0):
    """Train n ensemble members. Resumable — existing hypermodel<i>.h5 is reloaded."""
    from keras.callbacks import EarlyStopping
    _register_denorm()

    models_dir.mkdir(parents=True, exist_ok=True)
    models = []
    for i in range(n):
        path = models_dir / f'hypermodel{i + 1}.h5'
        if path.exists():
            models.append(keras.models.load_model(str(path), safe_mode=False))
            continue

        tf.random.set_seed(i)
        np.random.seed(i)
        model = build_model(hp, y_min, y_max_buffered)
        cb = [EarlyStopping(monitor='val_loss', patience=patience, mode='min',
                            restore_best_weights=True, verbose=0)]
        model.fit(X_train, y_train,
                  validation_data=(X_val, y_val),
                  epochs=epochs, batch_size=batch, validation_batch_size=batch,
                  callbacks=cb, verbose=verbose)
        model.save(str(path))
        models.append(model)
    return models


def _read_legacy_h5_architecture(h5_path: Path) -> tuple[dict, int]:
    """Read `model_config` from an old .h5 saved with a Lambda denorm layer.

    Returns (hp_dict, n_features). Architecture is enough to rebuild a
    DenormalizeSigmoid-based model from scratch; we never deserialize the
    Lambda layer's bytecode.
    """
    import h5py
    with h5py.File(h5_path, 'r') as f:
        cfg_raw = f.attrs['model_config']
        if isinstance(cfg_raw, bytes):
            cfg_raw = cfg_raw.decode()
        cfg = json.loads(cfg_raw)
        tc_raw = f.attrs['training_config']
        if isinstance(tc_raw, bytes):
            tc_raw = tc_raw.decode()
        tc = json.loads(tc_raw)
    dense_units = [
        l['config']['units']
        for l in cfg['config']['layers']
        if l['class_name'] == 'Dense' and l['config']['units'] != 1
    ]
    # First layer's batch_input_shape tells us n_features
    n_features = None
    for l in cfg['config']['layers']:
        cf = l.get('config', {})
        b = cf.get('batch_input_shape') or cf.get('batch_shape')
        if b and len(b) >= 2 and b[-1] is not None:
            n_features = int(b[-1])
            break
    hp = dict(
        num_layers=len(dense_units),
        neurons=int(dense_units[0]) if dense_units else 100,
        lr=float(tc['optimizer_config']['config']['learning_rate']),
        activation='relu',
    )
    return hp, n_features


def _extract_dense_weights(h5_path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    """Read every Dense layer's (kernel, bias) from a legacy Keras .h5."""
    import h5py
    with h5py.File(h5_path, 'r') as f:
        mw = f['model_weights']

        def _suffix_num(key: str) -> int:
            m = re.search(r'_(\d+)$', key)
            return int(m.group(1)) if m else -1

        dense_keys = sorted([k for k in mw.keys() if k.startswith('dense')],
                             key=_suffix_num)
        out = []
        for key in dense_keys:
            grp = mw[key]
            inner = list(grp.values())[0]
            kernel = np.array(inner[f'{key}/kernel'])
            bias = np.array(inner[f'{key}/bias'])
            out.append((kernel, bias))
    return out


def _load_legacy_keras_model(h5_path: Path, y_min: float, y_max_buffered: float):
    """Rebuild a Keras model from a legacy .h5 (with Lambda) via weight extraction.

    The returned model uses DenormalizeSigmoid (serializable) instead of Lambda,
    has the same architecture as the saved one, and identical weights.
    """
    hp, n_features = _read_legacy_h5_architecture(h5_path)
    dense_weights = _extract_dense_weights(h5_path)

    new_model = build_model(hp, y_min, y_max_buffered)
    # Force-build by calling once on a dummy input
    if n_features is None:
        n_features = dense_weights[0][0].shape[0]
    dummy = np.zeros((1, n_features), dtype=np.float32)
    _ = new_model(dummy, training=False)

    # Flatten dense_weights into [kernel0, bias0, kernel1, bias1, ...]
    weights_flat = []
    for kernel, bias in dense_weights:
        weights_flat.extend([kernel, bias])
    new_model.set_weights(weights_flat)
    return new_model


# Per-experiment cached y_min / y_max so the legacy loader can rebuild models.
# Filled lazily on first call to load_ensemble().
_Y_RANGE_CACHE: dict[str, tuple[float, float]] = {}


def _y_range_for_path(models_dir: Path) -> tuple[float, float]:
    """Derive (y_min, y_max_buffered) from the dataset; same constants for every experiment."""
    df_raw = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
    y_min = float(df_raw[TARGET_COL].min())
    y_max_buffered = float(df_raw[TARGET_COL].max() * (1 + Y_BUFFER))
    return y_min, y_max_buffered


def load_ensemble(models_dir: Path, n: int = N_ENSEMBLE):
    """Load up to n saved Keras ensemble members from models_dir.

    Handles three .h5 formats:
      1. New format saved by this script — DenormalizeSigmoid registered as
         `counterfactual>DenormalizeSigmoid`.
      2. Bundle format (feature_perturbation_retraining.py) — same class but
         registered as `Custom>DenormalizeSigmoid`.
      3. Legacy format (Lambda layer with embedded Python bytecode) — produced
         by the original deployment scripts; the marshalled Python bytecode
         is incompatible with newer Python, so we extract weights via h5py
         and rebuild the model from scratch.

    The first two are handled by passing `custom_objects={'DenormalizeSigmoid': …}`
    to load_model so the class is found regardless of which package prefix
    the saved config carries. The third triggers the weight-extraction
    fallback when load_model raises with a marshal-related error.
    """
    DenormSig = _register_denorm()
    custom_objects = {
        'DenormalizeSigmoid': DenormSig,
        'Custom>DenormalizeSigmoid': DenormSig,
        'counterfactual>DenormalizeSigmoid': DenormSig,
    }
    y_min, y_max = _y_range_for_path(models_dir)
    models = []
    for i in range(n):
        path = models_dir / f'hypermodel{i + 1}.h5'
        if not path.exists():
            raise FileNotFoundError(f'Missing ensemble member: {path}')
        try:
            models.append(keras.models.load_model(
                str(path), safe_mode=False, custom_objects=custom_objects))
        except (ValueError, TypeError) as e:
            msg = str(e).lower()
            if 'marshal' in msg or 'lambda' in msg or 'bad marshal' in msg:
                models.append(_load_legacy_keras_model(path, y_min, y_max))
            else:
                raise
    return models


def ensemble_predict(models, X: np.ndarray) -> np.ndarray:
    """Return shape (n_models, n_samples) array of predictions."""
    out = np.zeros((len(models), X.shape[0]), dtype=np.float64)
    for i, m in enumerate(models):
        out[i] = m.predict(X, batch_size=4096, verbose=0).ravel()
    return out


def ensemble_median(models, X: np.ndarray) -> np.ndarray:
    return np.median(ensemble_predict(models, X), axis=0)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ──────────────────────────────────────────────────────────────────────────
# FFCA report loading + backbone helpers
# ──────────────────────────────────────────────────────────────────────────
def load_corrected_report(exp_name: str) -> dict:
    cat, _ = EXPERIMENTS[exp_name]
    p = REPORTS_ROOT / cat / exp_name / 'report.json'
    if not p.exists():
        raise FileNotFoundError(f'Corrected ensemble report not found at {p}')
    with open(p) as f:
        return json.load(f)


# Archetype integer→paper-name mapping (mirrors FFCA_agent/ffca_agent/archetypes.py)
PACKAGE_INDEX_TO_PAPER = {
    0: "Noise Candidate",
    1: "Hidden Interactor",
    2: "Simple Workhorse",
    3: "Interactive Catalyst",
    4: "Non-linear Driver",
    5: "Volatile Specialist",
    6: "Stable Contributor",
    7: "Complex Driver",
}


def _trust_decision(entry) -> str:
    """Handle both nested-dict (corrected ensemble report) and flat-string forms."""
    if isinstance(entry, dict):
        return str(entry.get('decision', ''))
    return str(entry)


def _archetype_to_paper(a) -> str:
    """Normalize per-feature archetype to paper-form name."""
    if isinstance(a, int):
        return PACKAGE_INDEX_TO_PAPER.get(a, f'Unknown_{a}')
    return str(a)


def report_arrays(rep: dict) -> dict:
    """Extract canonical per-feature arrays from a corrected ensemble report."""
    sig = rep['signatures'][0]
    archetypes_paper = [_archetype_to_paper(a) for a in sig['archetypes']]
    return dict(
        feature_names=list(sig['feature_names']),
        impact=np.asarray(sig['impact'], dtype=np.float64),
        volatility=np.asarray(sig['volatility'], dtype=np.float64),
        nonlinearity=np.asarray(sig['nonlinearity'], dtype=np.float64),
        interaction=np.asarray(sig['interaction'], dtype=np.float64),
        archetypes=archetypes_paper,
        archetypes_raw=list(sig['archetypes']),
        trust=dict(rep.get('trust', {})),
    )


def ffca_ck_features(rep: dict) -> list[str]:
    """Names of features tagged CONFIDENTLY KEEP in the corrected ensemble report.

    Handles both the nested-dict trust format used by ensemble-mode reports
    (`trust[feat] = {'decision': 'CONFIDENTLY KEEP', ...}`) and the flat-string
    legacy format.
    """
    trust = rep.get('trust', {})
    out = []
    for name, entry in trust.items():
        if _trust_decision(entry).startswith('CONFIDENTLY KEEP'):
            out.append(name)
    return out


def top_k_by_metric(arrays: dict, metric: str, k: int,
                     restrict_to: Optional[Iterable[str]] = None) -> list[str]:
    """Return the top-k feature names by signature metric (impact / vol / nl / int)."""
    names = arrays['feature_names']
    vals = arrays[metric].copy()
    if restrict_to is not None:
        keep = set(restrict_to)
        for i, n in enumerate(names):
            if n not in keep:
                vals[i] = -np.inf
    order = np.argsort(-vals)
    out = []
    for idx in order:
        if len(out) >= k:
            break
        if vals[idx] == -np.inf:
            continue
        out.append(names[idx])
    return out


def archetype_groups(arrays: dict) -> dict[str, list[str]]:
    """Group feature names by archetype label."""
    grp: dict[str, list[str]] = {}
    for n, a in zip(arrays['feature_names'], arrays['archetypes']):
        grp.setdefault(a, []).append(n)
    return grp


def features_to_drop_to_indices(feat_cols: list[str], to_drop: Sequence[str]) -> tuple[list[str], np.ndarray]:
    """Return (kept_feature_names, keep_index_array_into_feat_cols)."""
    drop_set = set(to_drop)
    keep_idx = np.array([i for i, n in enumerate(feat_cols) if n not in drop_set], dtype=int)
    kept = [feat_cols[i] for i in keep_idx]
    return kept, keep_idx


# ──────────────────────────────────────────────────────────────────────────
# Attribution: SHAP (optional) + IntegratedGradients (TF-native, no extra deps)
# ──────────────────────────────────────────────────────────────────────────
def _try_import_shap():
    try:
        import shap  # type: ignore
        return shap
    except Exception:
        return None


def integrated_gradients(model, X: np.ndarray, baseline: Optional[np.ndarray] = None,
                          steps: int = 32, batch: int = 256) -> np.ndarray:
    """Path-integrated gradient attribution for a Keras model on TF.

    Returns an (n_samples, n_features) array of attributions.
    """
    if baseline is None:
        baseline = np.zeros((1, X.shape[1]), dtype=np.float32)
    if baseline.shape[0] == 1:
        baseline = np.broadcast_to(baseline, X.shape).astype(np.float32)

    alphas = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    attributions = np.zeros_like(X, dtype=np.float64)

    for start in range(0, X.shape[0], batch):
        Xb = X[start:start + batch]
        Bb = baseline[start:start + batch]
        diff = Xb - Bb                       # (b, d)
        grad_accum = np.zeros_like(Xb, dtype=np.float64)

        for a in alphas:
            xinp = Bb + a * diff             # (b, d)
            xtf = tf.convert_to_tensor(xinp)
            with tf.GradientTape() as tape:
                tape.watch(xtf)
                pred = model(xtf, training=False)
            g = tape.gradient(pred, xtf).numpy()   # (b, d)
            grad_accum += g

        avg_grad = grad_accum / float(steps)
        attributions[start:start + batch] = diff * avg_grad

    return attributions


def shap_attribution(model, X: np.ndarray, n_background: int = 200,
                      n_samples: Optional[int] = None) -> np.ndarray:
    """Per-feature mean |SHAP|. Uses shap.DeepExplainer when available."""
    shap = _try_import_shap()
    rng = np.random.default_rng(0)
    if n_samples is not None and X.shape[0] > n_samples:
        idx = rng.choice(X.shape[0], n_samples, replace=False)
        Xs = X[idx]
    else:
        Xs = X

    def _to_flat_per_feature(arr: np.ndarray) -> np.ndarray:
        """Reduce SHAP / gradient outputs to a flat per-feature 1D array.

        Handles shapes: (n_samples, n_features), (n_samples, n_features, 1),
        and (n_features, 1) all → (n_features,)."""
        a = np.asarray(arr)
        # Squeeze any trailing length-1 output dim
        while a.ndim > 2 and a.shape[-1] == 1:
            a = a.squeeze(-1)
        if a.ndim == 2:
            # (n_samples, n_features) → mean over samples
            return np.abs(a).mean(axis=0).ravel()
        return np.abs(a).ravel()

    if shap is None:
        # Fallback: gradient × input (a cheap, deterministic surrogate)
        xtf = tf.convert_to_tensor(Xs.astype(np.float32))
        with tf.GradientTape() as tape:
            tape.watch(xtf)
            pred = model(xtf, training=False)
        g = tape.gradient(pred, xtf).numpy()
        return _to_flat_per_feature(g * Xs)

    bg_idx = rng.choice(X.shape[0], min(n_background, X.shape[0]), replace=False)
    background = X[bg_idx].astype(np.float32)
    try:
        explainer = shap.DeepExplainer(model, background)
        sv = explainer.shap_values(Xs.astype(np.float32), check_additivity=False)
        if isinstance(sv, list):
            sv = sv[0]
        return _to_flat_per_feature(sv)
    except Exception as e:
        print(f'  shap DeepExplainer failed ({e}); falling back to gradient×input')
        xtf = tf.convert_to_tensor(Xs.astype(np.float32))
        with tf.GradientTape() as tape:
            tape.watch(xtf)
            pred = model(xtf, training=False)
        g = tape.gradient(pred, xtf).numpy()
        return _to_flat_per_feature(g * Xs)


def per_feature_grad_variance(model, X: np.ndarray, batch: int = 256) -> np.ndarray:
    """Var_x[|dy/dx_i|] per feature — proxy for FFCA Volatility from a single model."""
    n_samples = X.shape[0]
    grads = np.zeros_like(X, dtype=np.float64)
    for start in range(0, n_samples, batch):
        Xb = X[start:start + batch].astype(np.float32)
        xtf = tf.convert_to_tensor(Xb)
        with tf.GradientTape() as tape:
            tape.watch(xtf)
            pred = model(xtf, training=False)
        g = tape.gradient(pred, xtf).numpy()
        grads[start:start + batch] = np.abs(g)
    return grads.var(axis=0)


# ──────────────────────────────────────────────────────────────────────────
# Job dataclass + queue
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class TrainJob:
    section: str            # 'C' / 'D' / 'E' / 'F' / 'G'
    exp_name: str
    variant_name: str       # e.g. 'baseline_shap_drop' / 'archetype_noise_drop' / 'pair_AB_drop_gwl+gwl_t-1'
    drop_features: list[str]    # features to remove
    extra: dict = field(default_factory=dict)

    @property
    def jobkey(self) -> str:
        return f'{self.section}/{self.exp_name}/{self.variant_name}'


def shard_filter(jobs: list[TrainJob], shard: str) -> list[TrainJob]:
    """Filter jobs by --shard k/N (k in 0..N-1)."""
    if not shard:
        return jobs
    try:
        k, n = shard.split('/')
        k = int(k); n = int(n)
        assert 0 <= k < n
    except Exception:
        raise SystemExit(f'Invalid --shard {shard!r}; expected k/N')
    return [j for i, j in enumerate(jobs) if (i % n) == k]


# ──────────────────────────────────────────────────────────────────────────
# Train + evaluate one job
# ──────────────────────────────────────────────────────────────────────────
def run_train_job(job: TrainJob, output_dir: Path, smoke: bool = False) -> dict:
    """Train a 30-seed ensemble on the experiment with `drop_features` removed,
    evaluate on test year, return a metrics dict. Resumable.
    """
    spec = find_experiment_spec(job.exp_name)
    df, feat_cols, target, y_min, y_max_buf = build_full_feature_matrix(spec)
    kept, keep_idx = features_to_drop_to_indices(feat_cols, job.drop_features)
    if not kept:
        return dict(status='skip-no-features', n_features=0)

    hp = recover_hyperparams(job.exp_name)
    out_dir = output_dir / 'runs' / f'section_{job.section}' / job.exp_name / job.variant_name
    models_dir = out_dir / 'models'
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / 'results.json'
    if results_path.exists():
        with open(results_path) as f:
            cached = json.load(f)
        cached['status'] = 'cached'
        return cached

    df_kept = df[kept + [target]]
    Xtr, ytr, Xv, yv, Xte, yte, idx_te = split_train_val_test(df_kept, target, kept)

    n_seeds = 3 if smoke else N_ENSEMBLE
    t0 = time.time()
    models = train_ensemble(Xtr, ytr, Xv, yv, hp, y_min, y_max_buf, models_dir,
                            n=n_seeds, patience=DEFAULT_PATIENCE,
                            epochs=200 if smoke else MAX_EPOCHS)
    train_time = time.time() - t0

    preds = ensemble_predict(models, Xte)
    med = np.median(preds, axis=0)

    metrics = dict(
        section=job.section,
        exp_name=job.exp_name,
        variant_name=job.variant_name,
        drop_features=list(job.drop_features),
        n_dropped=len(job.drop_features),
        n_kept=len(kept),
        n_seeds=n_seeds,
        train_time_sec=round(train_time, 2),
        test_rmse=rmse(yte, med),
        test_rmse_cm=rmse(yte, med) * 100.0,
        ensemble_std=float(preds.std(axis=0).mean()),
        extra=job.extra,
        status='trained',
    )
    # Per-event metrics for tail-event analysis on TRAINED variants too
    for q in TAIL_TOP_PCT:
        thresh = np.quantile(yte, 1 - q)
        mask = yte >= thresh
        if mask.sum() >= 5:
            metrics[f'tail_top{int(q*100)}pct_rmse_cm'] = rmse(yte[mask], med[mask]) * 100.0
            metrics[f'tail_top{int(q*100)}pct_n'] = int(mask.sum())

    # Save median prediction for downstream analysis (small files)
    np.save(out_dir / 'pred_median.npy', med)
    np.save(out_dir / 'pred_std.npy', preds.std(axis=0))
    np.save(out_dir / 'y_test.npy', yte)
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Free GPU memory between jobs
    del models, preds
    keras.backend.clear_session()
    gc.collect()

    return metrics


# ──────────────────────────────────────────────────────────────────────────
# Section A — Tail-event re-evaluation (no training)
# ──────────────────────────────────────────────────────────────────────────
def run_section_a(output_dir: Path, smoke: bool = False) -> Path:
    """For each of 20 experiments, evaluate original + variant_D ensembles
    on test year 2024 and compute RMSE on top-5% and top-1% extreme events.
    """
    print('\n========== SECTION A: Tail-event re-evaluation ==========')
    rows = []

    for exp_name, (cat, lead) in EXPERIMENTS.items():
        print(f'  [A] {exp_name}')
        try:
            spec = find_experiment_spec(exp_name)
            df, feat_cols, target, y_min, y_max_buf = build_full_feature_matrix(spec)
            Xte, yte = split_by_year(df, target, feat_cols, year=TEST_YEAR)

            orig_dir = original_models_dir(exp_name)
            orig_models = load_ensemble(orig_dir, n=3 if smoke else N_ENSEMBLE)
            orig_med = ensemble_median(orig_models, Xte)
            row = dict(experiment=exp_name, category=cat, lead=lead,
                       n_test=len(yte),
                       orig_mean_rmse_cm=rmse(yte, orig_med) * 100.0)
            for q in TAIL_TOP_PCT:
                thresh = np.quantile(yte, 1 - q)
                mask = yte >= thresh
                row[f'orig_top{int(q*100)}pct_rmse_cm'] = rmse(yte[mask], orig_med[mask]) * 100.0
                row[f'top{int(q*100)}pct_n'] = int(mask.sum())
            del orig_models
            keras.backend.clear_session()
            gc.collect()

            # Look for variant_D ensemble (uses helper that knows both Mac & HPC layouts)
            variant_d_dir = variant_d_models_dir(exp_name)
            if variant_d_dir is not None:
                # variant_D uses a reduced feature set — need to load that spec
                rep = load_corrected_report(exp_name)
                ck = ffca_ck_features(rep)
                kept_d, _ = features_to_drop_to_indices(feat_cols, ck)
                df_d = df[kept_d + [target]]
                Xte_d, _ = split_by_year(df_d, target, kept_d, year=TEST_YEAR)
                d_models = load_ensemble(variant_d_dir, n=3 if smoke else N_ENSEMBLE)
                d_med = ensemble_median(d_models, Xte_d)
                row['variantD_mean_rmse_cm'] = rmse(yte, d_med) * 100.0
                for q in TAIL_TOP_PCT:
                    thresh = np.quantile(yte, 1 - q)
                    mask = yte >= thresh
                    row[f'variantD_top{int(q*100)}pct_rmse_cm'] = rmse(yte[mask], d_med[mask]) * 100.0
                row['delta_mean_cm'] = row['variantD_mean_rmse_cm'] - row['orig_mean_rmse_cm']
                for q in TAIL_TOP_PCT:
                    row[f'delta_top{int(q*100)}pct_cm'] = (
                        row[f'variantD_top{int(q*100)}pct_rmse_cm'] - row[f'orig_top{int(q*100)}pct_rmse_cm']
                    )
                # Headline: amplification factor = tail Δ / mean Δ
                if row['delta_mean_cm'] > 0:
                    row['tail5_amplification'] = row['delta_top5pct_cm'] / row['delta_mean_cm']
                del d_models
                keras.backend.clear_session()
                gc.collect()
            else:
                row['variantD_mean_rmse_cm'] = None

            rows.append(row)
        except Exception as e:
            print(f'    SKIPPED: {e}')
            rows.append(dict(experiment=exp_name, error=str(e)))
            traceback.print_exc()

    csv_path = output_dir / 'section_A.csv'
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f'  Wrote {csv_path}')
    return csv_path


# ──────────────────────────────────────────────────────────────────────────
# Section B — Permutation importance (no retraining)
# ──────────────────────────────────────────────────────────────────────────
def run_section_b(output_dir: Path, smoke: bool = False) -> Path:
    """For each experiment, permute three subsets independently and measure
    ensemble RMSE shift: (a) FFCA CK columns, (b) bottom-K Impact columns,
    (c) random K columns (3 random seeds averaged). K matched to |CK|.

    Hypothesis: permuting CK should produce the largest RMSE increase.
    """
    print('\n========== SECTION B: Permutation importance ==========')
    rows = []
    rng_master = np.random.default_rng(42)

    for exp_name in EXPERIMENTS:
        print(f'  [B] {exp_name}')
        try:
            spec = find_experiment_spec(exp_name)
            df, feat_cols, target, y_min, y_max_buf = build_full_feature_matrix(spec)
            Xte, yte = split_by_year(df, target, feat_cols, year=TEST_YEAR)
            n, d = Xte.shape

            rep = load_corrected_report(exp_name)
            arrays = report_arrays(rep)
            ck = ffca_ck_features(rep)
            K = len(ck)
            if K == 0:
                rows.append(dict(experiment=exp_name, K=0, note='no CK features'))
                continue

            ck_idx = np.array([feat_cols.index(n) for n in ck if n in feat_cols], dtype=int)
            # Bottom-K Impact among features in this experiment (signature feature_names may equal feat_cols)
            sig_names = arrays['feature_names']
            sig_impact = arrays['impact']
            in_exp = np.array([1 if n in feat_cols else 0 for n in sig_names], dtype=bool)
            order = np.argsort(sig_impact)
            bottom_names = []
            for i in order:
                if not in_exp[i]:
                    continue
                if sig_names[i] in ck:
                    continue
                bottom_names.append(sig_names[i])
                if len(bottom_names) >= K:
                    break
            bottom_idx = np.array([feat_cols.index(n) for n in bottom_names], dtype=int)

            orig_dir = original_models_dir(exp_name)
            n_seeds = 3 if smoke else N_ENSEMBLE
            models = load_ensemble(orig_dir, n=n_seeds)

            base_med = ensemble_median(models, Xte)
            base_rmse = rmse(yte, base_med) * 100.0

            def permute_and_score(idxs: np.ndarray, seed: int):
                rng = np.random.default_rng(seed)
                Xp = Xte.copy()
                for j in idxs:
                    Xp[:, j] = rng.permutation(Xp[:, j])
                med_p = ensemble_median(models, Xp)
                return rmse(yte, med_p) * 100.0

            ck_rmse = permute_and_score(ck_idx, seed=0)
            bottom_rmse = permute_and_score(bottom_idx, seed=0)
            random_rmses = []
            for s in range(3):
                rand_idx = rng_master.choice(d, size=K, replace=False)
                random_rmses.append(permute_and_score(rand_idx, seed=100 + s))
            rand_rmse_mean = float(np.mean(random_rmses))
            rand_rmse_std = float(np.std(random_rmses))

            rows.append(dict(
                experiment=exp_name, K=K,
                base_rmse_cm=base_rmse,
                perm_ck_rmse_cm=ck_rmse,
                perm_bottom_rmse_cm=bottom_rmse,
                perm_random_rmse_cm=rand_rmse_mean,
                perm_random_std_cm=rand_rmse_std,
                delta_ck_cm=ck_rmse - base_rmse,
                delta_bottom_cm=bottom_rmse - base_rmse,
                delta_random_cm=rand_rmse_mean - base_rmse,
                ratio_ck_to_random=(ck_rmse - base_rmse) / max(1e-9, rand_rmse_mean - base_rmse),
            ))
            del models
            keras.backend.clear_session()
            gc.collect()

        except Exception as e:
            print(f'    SKIPPED: {e}')
            rows.append(dict(experiment=exp_name, error=str(e)))
            traceback.print_exc()

    csv_path = output_dir / 'section_B.csv'
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f'  Wrote {csv_path}')
    return csv_path


# ──────────────────────────────────────────────────────────────────────────
# Section C — Attribution-baseline backbones (retrain)
# ──────────────────────────────────────────────────────────────────────────
def build_section_c_jobs(experiments: list[str]) -> list[TrainJob]:
    jobs = []
    for exp in experiments:
        rep = load_corrected_report(exp)
        arrays = report_arrays(rep)
        ck = ffca_ck_features(rep)
        K = len(ck)
        if K == 0:
            continue

        spec = find_experiment_spec(exp)
        df, feat_cols, _, _, _ = build_full_feature_matrix(spec)

        # SHAP and IG top-K computed using the loaded original ensemble (done at run time)
        # Raw Impact top-K (no trust filter) computed from report immediately
        sig_names = arrays['feature_names']
        in_exp = [n for n in sig_names if n in feat_cols]
        rawimpact_topk = top_k_by_metric(arrays, 'impact', K, restrict_to=in_exp)
        jobs.append(TrainJob(section='C', exp_name=exp,
                             variant_name='drop_rawimpact_topK',
                             drop_features=rawimpact_topk,
                             extra=dict(K=K, criterion='rawimpact_topK')))
        # SHAP/IG placeholders — actual feature lists computed at run time
        jobs.append(TrainJob(section='C', exp_name=exp,
                             variant_name='drop_shap_topK',
                             drop_features=[],   # filled by runtime helper
                             extra=dict(K=K, criterion='shap_topK', deferred=True)))
        jobs.append(TrainJob(section='C', exp_name=exp,
                             variant_name='drop_ig_topK',
                             drop_features=[],
                             extra=dict(K=K, criterion='ig_topK', deferred=True)))
        # Reference: FFCA CK retrain (variant_D-equivalent, fresh ensemble for clean compare)
        jobs.append(TrainJob(section='C', exp_name=exp,
                             variant_name='drop_ffca_ck',
                             drop_features=list(ck),
                             extra=dict(K=K, criterion='ffca_ck')))
    return jobs


def resolve_deferred_section_c(jobs: list[TrainJob], output_dir: Path,
                                smoke: bool = False):
    """For deferred SHAP/IG jobs, compute the top-K features and fill drop_features.

    Caches the computed ranking to output_dir/runs/section_C/<exp>/_attributions.json
    so it's idempotent.
    """
    by_exp: dict[str, list[TrainJob]] = {}
    for j in jobs:
        if j.section == 'C' and j.extra.get('deferred'):
            by_exp.setdefault(j.exp_name, []).append(j)

    for exp, deferred_jobs in by_exp.items():
        cache_path = output_dir / 'runs' / 'section_C' / exp / '_attributions.json'
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            with open(cache_path) as f:
                cache = json.load(f)
        else:
            print(f'  [C-attr] {exp} computing SHAP/IG…')
            spec = find_experiment_spec(exp)
            df, feat_cols, target, y_min, y_max_buf = build_full_feature_matrix(spec)
            Xtr, ytr, Xv, yv, Xte, yte, _ = split_train_val_test(df, target, feat_cols)
            orig_dir = original_models_dir(exp)
            n_seeds = 3 if smoke else N_ENSEMBLE
            models = load_ensemble(orig_dir, n=n_seeds)
            # Reference subsample for attribution
            rng = np.random.default_rng(0)
            ref_idx = rng.choice(Xte.shape[0], min(500, Xte.shape[0]), replace=False)
            Xref = Xte[ref_idx]
            # Aggregate per-feature SHAP and IG by averaging across ensemble members
            shap_per_seed = np.zeros((n_seeds, Xref.shape[1]))
            ig_per_seed = np.zeros((n_seeds, Xref.shape[1]))
            baseline = Xtr.mean(axis=0, keepdims=True).astype(np.float32)
            for i, m in enumerate(models):
                shap_per_seed[i] = shap_attribution(m, Xref, n_background=200, n_samples=500)
                ig_per_seed[i] = np.abs(integrated_gradients(m, Xref, baseline=baseline, steps=24)).mean(axis=0)
            del models
            keras.backend.clear_session()
            gc.collect()

            shap_mean = shap_per_seed.mean(axis=0).tolist()
            ig_mean = ig_per_seed.mean(axis=0).tolist()
            cache = dict(feat_cols=feat_cols, shap=shap_mean, ig=ig_mean)
            with open(cache_path, 'w') as f:
                json.dump(cache, f)

        feat_cols = cache['feat_cols']
        shap_vals = np.asarray(cache['shap'])
        ig_vals = np.asarray(cache['ig'])

        for j in deferred_jobs:
            K = j.extra['K']
            if j.extra['criterion'] == 'shap_topK':
                idx = np.argsort(-shap_vals)[:K]
            elif j.extra['criterion'] == 'ig_topK':
                idx = np.argsort(-ig_vals)[:K]
            else:
                continue
            j.drop_features = [feat_cols[i] for i in idx]
            j.extra['deferred'] = False


# ──────────────────────────────────────────────────────────────────────────
# Section D — Archetype ablation (retrain)
# ──────────────────────────────────────────────────────────────────────────
ARCHETYPES_TO_TEST = (
    'Interactive Catalyst',
    'Complex Driver',
    'Stable Contributor',
    'Noise Candidate',
    'Volatile Specialist',
    'Hidden Interactor',
    'Non-linear Driver',   # matches PACKAGE_INDEX_TO_PAPER (hyphenated)
    'Simple Workhorse',
)


def build_section_d_jobs(experiments: list[str]) -> list[TrainJob]:
    jobs = []
    for exp in experiments:
        rep = load_corrected_report(exp)
        arrays = report_arrays(rep)
        spec = find_experiment_spec(exp)
        df, feat_cols, _, _, _ = build_full_feature_matrix(spec)
        groups = archetype_groups(arrays)
        for arch in ARCHETYPES_TO_TEST:
            members = [n for n in groups.get(arch, []) if n in feat_cols]
            if not members:
                continue
            slug = arch.lower().replace(' ', '_')
            jobs.append(TrainJob(
                section='D', exp_name=exp,
                variant_name=f'drop_archetype_{slug}',
                drop_features=members,
                extra=dict(archetype=arch, n_in_archetype=len(members)),
            ))
    return jobs


# ──────────────────────────────────────────────────────────────────────────
# Section E — Interaction-pair synergy (retrain)
# ──────────────────────────────────────────────────────────────────────────
def pick_interaction_pairs(rep: dict, feat_cols: list[str],
                            n_high: int = 3, n_low: int = 3) -> dict[str, list[tuple[str, str]]]:
    """Pick demo pairs:
      - high_int: top features by FFCA Interaction scalar, paired by adjacency in the ranking
      - low_int: matched on Impact range, lowest Interaction scalar
    """
    arrays = report_arrays(rep)
    names = arrays['feature_names']
    interaction = arrays['interaction']
    impact = arrays['impact']

    in_exp_mask = np.array([n in feat_cols for n in names])
    valid_idx = np.where(in_exp_mask)[0]

    # Top by interaction, restricted to in-experiment features
    inter_sorted = sorted(valid_idx, key=lambda i: -interaction[i])
    # Pair adjacent picks until we have n_high pairs that don't share features
    high_pairs: list[tuple[str, str]] = []
    used = set()
    i = 0
    while i + 1 < len(inter_sorted) and len(high_pairs) < n_high:
        a, b = inter_sorted[i], inter_sorted[i + 1]
        if names[a] not in used and names[b] not in used:
            high_pairs.append((names[a], names[b]))
            used.add(names[a]); used.add(names[b])
            i += 2
        else:
            i += 1

    # Determine impact range used by the high-int pairs
    hi_impact_min = min(impact[names.index(a)] for a, _ in high_pairs)
    hi_impact_max = max(impact[names.index(b)] for _, b in high_pairs)
    span = max(hi_impact_max - hi_impact_min, 1e-9)
    # Use a generous tolerance — match WITHIN ±1× the span. Falls back to wider.
    for tol_mult in (1.0, 2.0, 4.0, 10.0):
        tol = tol_mult * span
        impact_lo = hi_impact_min - tol
        impact_hi = hi_impact_max + tol
        cands = [i for i in valid_idx
                 if impact_lo <= impact[i] <= impact_hi and names[i] not in used]
        if len(cands) >= 2 * n_low:
            break

    # If still too few, drop the impact-match constraint entirely (note in extra)
    if len(cands) < 2 * n_low:
        cands = [i for i in valid_idx if names[i] not in used]

    cands.sort(key=lambda i: interaction[i])    # lowest interaction first
    low_pairs: list[tuple[str, str]] = []
    j = 0
    while j + 1 < len(cands) and len(low_pairs) < n_low:
        a, b = cands[j], cands[j + 1]
        if names[a] not in used and names[b] not in used:
            low_pairs.append((names[a], names[b]))
            used.add(names[a]); used.add(names[b])
            j += 2
        else:
            j += 1

    return dict(high_int=high_pairs, low_int=low_pairs)


def build_section_e_jobs(demo_exp: str) -> list[TrainJob]:
    rep = load_corrected_report(demo_exp)
    spec = find_experiment_spec(demo_exp)
    _, feat_cols, _, _, _ = build_full_feature_matrix(spec)
    pairs = pick_interaction_pairs(rep, feat_cols, n_high=3, n_low=3)
    jobs: list[TrainJob] = []
    for kind, pair_list in pairs.items():
        for (a, b) in pair_list:
            slug = f'{kind}_{a}_{b}'.replace(' ', '_').replace('/', '-')
            for cond_name, drop in [
                ('drop_A_only', [a]),
                ('drop_B_only', [b]),
                ('drop_AB',      [a, b]),
            ]:
                jobs.append(TrainJob(
                    section='E', exp_name=demo_exp,
                    variant_name=f'{slug}__{cond_name}',
                    drop_features=drop,
                    extra=dict(pair_kind=kind, pair_A=a, pair_B=b,
                               condition=cond_name),
                ))
    return jobs


def synergy_score(rows_for_pair: list[dict], baseline_rmse_cm: float) -> dict:
    """Given the 3 rows for a pair (A only, B only, AB), compute super-additivity:
       Δ_AB − (Δ_A + Δ_B). Positive = synergistic damage = real interaction."""
    by_cond = {r['extra']['condition']: r for r in rows_for_pair}
    if not all(c in by_cond for c in ('drop_A_only', 'drop_B_only', 'drop_AB')):
        return {}
    d_a = by_cond['drop_A_only']['test_rmse_cm'] - baseline_rmse_cm
    d_b = by_cond['drop_B_only']['test_rmse_cm'] - baseline_rmse_cm
    d_ab = by_cond['drop_AB']['test_rmse_cm'] - baseline_rmse_cm
    return dict(delta_A_cm=d_a, delta_B_cm=d_b, delta_AB_cm=d_ab,
                super_additivity_cm=d_ab - (d_a + d_b))


# ──────────────────────────────────────────────────────────────────────────
# Section F — Nonlinearity gap (retrain MLP only — fresh ensembles; LR fit at eval)
# ──────────────────────────────────────────────────────────────────────────
def build_section_f_jobs(demo_exp: str) -> list[TrainJob]:
    """For one experiment, train two NEW MLP ensembles:
       (a) using top-K HIGH-Nonlinearity features only
       (b) using top-K LOW-Nonlinearity features only (matched Impact)
    Then at evaluation, also fit Linear Regression baselines on the same
    subsets and compute the (LR − MLP) RMSE gap. Hypothesis: HIGH-N
    features produce a much larger LR→MLP gap than LOW-N features.
    """
    rep = load_corrected_report(demo_exp)
    arrays = report_arrays(rep)
    spec = find_experiment_spec(demo_exp)
    _, feat_cols, _, _, _ = build_full_feature_matrix(spec)
    names = arrays['feature_names']
    impact = arrays['impact']
    nonlin = arrays['nonlinearity']

    in_exp = np.array([n in feat_cols for n in names])
    valid = np.where(in_exp)[0]

    # K = top 20 features by Impact within the experiment (so subsets are useful)
    K = min(20, len(valid))
    # High-N subset: top-K by nonlinearity among in-experiment features
    hi_n_idx = sorted(valid, key=lambda i: -nonlin[i])[:K]
    high_n_names = [names[i] for i in hi_n_idx]
    # Low-N subset: among features with comparable Impact, lowest Nonlinearity
    hi_impact_lo = min(impact[i] for i in hi_n_idx)
    hi_impact_hi = max(impact[i] for i in hi_n_idx)
    tol = 0.3 * (hi_impact_hi - hi_impact_lo + 1e-9)
    impact_lo = hi_impact_lo - tol
    impact_hi = hi_impact_hi + tol
    used = set(high_n_names)
    cands = [i for i in valid if impact_lo <= impact[i] <= impact_hi and names[i] not in used]
    cands.sort(key=lambda i: nonlin[i])
    low_n_names = [names[i] for i in cands[:K]]
    if len(low_n_names) < K // 2:
        # Fall back to bottom-K nonlinearity across all in-experiment features
        all_sorted = sorted(valid, key=lambda i: nonlin[i])
        low_n_names = [names[i] for i in all_sorted if names[i] not in used][:K]

    # Each MLP variant DROPS everything except the chosen subset
    full_set = set(feat_cols)
    drop_for_high = list(full_set - set(high_n_names))
    drop_for_low = list(full_set - set(low_n_names))

    jobs = [
        TrainJob(section='F', exp_name=demo_exp,
                 variant_name='keep_only_high_nonlinearity',
                 drop_features=drop_for_high,
                 extra=dict(subset_kind='high_nonlinearity',
                            kept_features=high_n_names, K=K)),
        TrainJob(section='F', exp_name=demo_exp,
                 variant_name='keep_only_low_nonlinearity',
                 drop_features=drop_for_low,
                 extra=dict(subset_kind='low_nonlinearity',
                            kept_features=low_n_names, K=K)),
    ]
    return jobs


def fit_linear_baseline_rmse(spec: dict, kept_features: list[str]) -> float:
    """Fit ordinary least squares regression on (train+val) → predict test."""
    from sklearn.linear_model import LinearRegression
    df, feat_cols, target, _, _ = build_full_feature_matrix(spec)
    df_kept = df[[c for c in kept_features if c in df.columns] + [target]]
    Xtr, ytr, Xv, yv, Xte, yte, _ = split_train_val_test(
        df_kept, target, [c for c in kept_features if c in df.columns],
        val_year=VAL_YEAR, test_year=TEST_YEAR,
    )
    # Use train+val together as the LR training set
    X_fit = np.concatenate([Xtr, Xv], axis=0)
    y_fit = np.concatenate([ytr, yv], axis=0)
    lr = LinearRegression()
    lr.fit(X_fit, y_fit)
    pred = lr.predict(Xte)
    return float(rmse(yte, pred) * 100.0)


# ──────────────────────────────────────────────────────────────────────────
# Section G — Volatility–uncertainty link (retrain)
# ──────────────────────────────────────────────────────────────────────────
def build_section_g_jobs(experiments: list[str]) -> list[TrainJob]:
    """For each experiment, retrain ensembles with high-V and low-V (matched Impact)
    subsets removed. At eval, measure both RMSE shift AND change in ensemble std.
    Hypothesis: high-V removal reduces ensemble disagreement more than low-V.
    """
    jobs = []
    for exp in experiments:
        rep = load_corrected_report(exp)
        arrays = report_arrays(rep)
        spec = find_experiment_spec(exp)
        _, feat_cols, _, _, _ = build_full_feature_matrix(spec)
        names = arrays['feature_names']
        impact = arrays['impact']
        vol = arrays['volatility']
        in_exp_mask = np.array([n in feat_cols for n in names])
        valid = np.where(in_exp_mask)[0]
        K = min(20, len(valid) // 4)
        # Top-K high-V
        hi_v_idx = sorted(valid, key=lambda i: -vol[i])[:K]
        hi_v = [names[i] for i in hi_v_idx]
        # Low-V, Impact-matched
        hi_impact_lo = min(impact[i] for i in hi_v_idx)
        hi_impact_hi = max(impact[i] for i in hi_v_idx)
        tol = 0.3 * (hi_impact_hi - hi_impact_lo + 1e-9)
        used = set(hi_v)
        cands = [i for i in valid
                 if (hi_impact_lo - tol) <= impact[i] <= (hi_impact_hi + tol)
                 and names[i] not in used]
        cands.sort(key=lambda i: vol[i])
        lo_v = [names[i] for i in cands[:K]]
        if len(lo_v) < K // 2:
            all_sorted = sorted(valid, key=lambda i: vol[i])
            lo_v = [names[i] for i in all_sorted if names[i] not in used][:K]
        jobs.append(TrainJob(section='G', exp_name=exp,
                             variant_name='drop_high_volatility',
                             drop_features=hi_v,
                             extra=dict(kind='high_volatility', K=K, names=hi_v)))
        jobs.append(TrainJob(section='G', exp_name=exp,
                             variant_name='drop_low_volatility',
                             drop_features=lo_v,
                             extra=dict(kind='low_volatility', K=K, names=lo_v)))
    return jobs


# ──────────────────────────────────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────────────────────────────────
def _shard_csv_suffix(shard: str) -> str:
    """`'0/7'` → `'_shard_0_of_7'`; `''` → `''`. Used to keep per-shard CSV
    paths distinct so concurrent shards do not overwrite each other."""
    if not shard:
        return ''
    k, n = shard.split('/')
    return f'_shard_{int(k)}_of_{int(n)}'


def run_train_section(section: str, jobs: list[TrainJob],
                       output_dir: Path, smoke: bool = False,
                       shard: str = '') -> Path:
    """Execute training jobs (already shard-filtered), write a per-shard CSV.

    When `shard` is set (e.g. `'0/7'`), the CSV path becomes
    `section_<X>_shard_<k>_of_<N>.csv` — this prevents the
    concurrent-overwrite bug where all 7 shards used to write to the same
    `section_<X>.csv` and only the last shard's slice survived.

    The canonical, authoritative output is still the per-job
    `runs/section_<X>/<exp>/<variant>/results.json` files. Call
    `rebuild_section_csv_from_jsons(section, output_dir)` after all shards
    finish to produce the merged canonical `section_<X>.csv`.
    """
    print(f'\n========== SECTION {section}: {len(jobs)} training jobs '
          f'(shard {shard or "n/a"}) ==========')
    rows = []
    for i, job in enumerate(jobs):
        print(f'  [{section} {i+1}/{len(jobs)}] {job.jobkey}  '
              f'(drop {len(job.drop_features)}, kept ~{job.extra.get("kept_count", "?")})')
        try:
            res = run_train_job(job, output_dir, smoke=smoke)
            rows.append(res)
        except Exception as e:
            print(f'    FAILED: {e}')
            rows.append(dict(jobkey=job.jobkey, error=str(e),
                             section=section, exp_name=job.exp_name,
                             variant_name=job.variant_name))
            traceback.print_exc()

    suffix = _shard_csv_suffix(shard)
    out_path = output_dir / f'section_{section}{suffix}.csv'
    # Flatten extra/dict columns
    flat = []
    for r in rows:
        rr = {k: v for k, v in r.items() if k != 'extra'}
        ex = r.get('extra', {})
        if isinstance(ex, dict):
            for k, v in ex.items():
                rr[f'extra_{k}'] = (v if not isinstance(v, list) else ';'.join(map(str, v)))
        flat.append(rr)
    pd.DataFrame(flat).to_csv(out_path, index=False)
    print(f'  Wrote {out_path}  ({len(flat)} rows)')
    return out_path


def rebuild_section_csv_from_jsons(section: str, output_dir: Path) -> Path:
    """Rebuild `section_<X>.csv` from the per-job `results.json` files.

    Per-job JSONs are the authoritative source — they are written atomically
    per job, survive crashes, and are not subject to the concurrent-write
    overwrite that struck the sharded CSV writes during the §A-G run on
    2026-05-25 and again on §H 2026-05-26. Always run this once all shards
    have finished training and before reading `section_<X>.csv` downstream.
    """
    runs_root = output_dir / 'runs' / f'section_{section}'
    rows = []
    if runs_root.exists():
        for rp in sorted(runs_root.rglob('results.json')):
            d = json.load(open(rp))
            rr = {k: v for k, v in d.items() if k != 'extra'}
            ex = d.get('extra', {}) or {}
            for k, v in ex.items():
                rr[f'extra_{k}'] = (v if not isinstance(v, list)
                                    else ';'.join(map(str, v)))
            rows.append(rr)
    final = output_dir / f'section_{section}.csv'
    if rows:
        pd.DataFrame(rows).to_csv(final, index=False)
        print(f'  Rebuilt {final} from {len(rows)} per-job JSON files')
    else:
        # Fall back to merging per-shard CSVs if the per-job JSONs are missing
        parts = sorted(output_dir.glob(f'section_{section}_shard_*.csv'))
        if not parts:
            print(f'  WARNING: nothing to rebuild for §{section} '
                  '(no per-job JSONs, no per-shard CSVs)')
            return final
        merged = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
        keys = [k for k in ('section', 'exp_name', 'variant_name') if k in merged.columns]
        if keys:
            merged = merged.drop_duplicates(subset=keys, keep='last')
        merged.to_csv(final, index=False)
        print(f'  Merged {len(parts)} shard CSVs → {final} ({len(merged)} rows)')
    return final


# ──────────────────────────────────────────────────────────────────────────
# Final report
# ──────────────────────────────────────────────────────────────────────────
def _df_to_markdown(df: pd.DataFrame, round_to: int = 2) -> str:
    """Hand-rolled markdown pipe table — avoids the tabulate dependency."""
    if df is None or len(df) == 0:
        return '_(empty)_'
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].round(round_to)
    headers = list(df.columns)
    lines = ['| ' + ' | '.join(str(h) for h in headers) + ' |',
             '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for _, row in df.iterrows():
        cells = []
        for c in headers:
            v = row[c]
            if pd.isna(v):
                cells.append('')
            elif isinstance(v, float):
                cells.append(f'{v:.{round_to}f}')
            else:
                cells.append(str(v))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def write_final_report(output_dir: Path):
    print('\n========== Generating FINAL_REPORT.md ==========')
    md = ['# Compound-Flooding Extra Validation — Auto-Generated Report\n',
          f'_Generated at {time.strftime("%Y-%m-%d %H:%M:%S")}_\n']

    def load_csv(p):
        try:
            return pd.read_csv(p)
        except Exception:
            return None

    df_a = load_csv(output_dir / 'section_A.csv')
    if df_a is not None and len(df_a):
        md.append('## Section A — Tail-event re-evaluation\n')
        cols = [c for c in ['experiment', 'orig_mean_rmse_cm', 'variantD_mean_rmse_cm',
                            'delta_mean_cm', 'delta_top5pct_cm', 'delta_top1pct_cm',
                            'tail5_amplification'] if c in df_a.columns]
        md.append(_df_to_markdown(df_a[cols], round_to=2))
        if 'tail5_amplification' in df_a.columns:
            n_amp = int((df_a['tail5_amplification'] > 1.5).sum())
            md.append(f'\n**Verdict:** Backbone removal amplifies extreme-event damage '
                      f'in {n_amp}/{df_a["tail5_amplification"].notna().sum()} experiments '
                      f'(tail-5% Δ > 1.5× mean Δ).\n')

    df_b = load_csv(output_dir / 'section_B.csv')
    if df_b is not None and len(df_b):
        md.append('## Section B — Permutation importance\n')
        cols = [c for c in ['experiment', 'K', 'base_rmse_cm', 'delta_ck_cm',
                            'delta_bottom_cm', 'delta_random_cm',
                            'ratio_ck_to_random'] if c in df_b.columns]
        md.append(_df_to_markdown(df_b[cols], round_to=2))
        if 'ratio_ck_to_random' in df_b.columns:
            med = df_b['ratio_ck_to_random'].median()
            md.append(f'\n**Verdict:** median CK/random damage ratio = {med:.2f}× '
                      f'(higher = CK column values carry more model-relevant signal '
                      f'than random columns of equal count).\n')

    df_c = load_csv(output_dir / 'section_C.csv')
    if df_c is not None and len(df_c):
        md.append('## Section C — Attribution-baseline backbones\n')
        cols = [c for c in ['exp_name', 'variant_name', 'n_kept', 'test_rmse_cm',
                            'extra_criterion'] if c in df_c.columns]
        md.append(_df_to_markdown(df_c[cols], round_to=2))
        md.append('\nCompare delta_RMSE for `drop_ffca_ck` vs `drop_shap_topK`, '
                  '`drop_ig_topK`, `drop_rawimpact_topK` per experiment. '
                  'If FFCA CK and SHAP/IG select similar feature sets, expect similar '
                  'RMSE deltas — the paper claim then rests on the *categorical '
                  'diagnosis*, not on a different ranking.\n')

    df_d = load_csv(output_dir / 'section_D.csv')
    if df_d is not None and len(df_d):
        md.append('## Section D — Archetype ablation\n')
        cols = [c for c in ['exp_name', 'variant_name', 'extra_archetype',
                            'extra_n_in_archetype', 'test_rmse_cm']
                if c in df_d.columns]
        md.append(_df_to_markdown(df_d[cols], round_to=2))
        md.append('\nExpected pattern: dropping `Noise Candidate` → near-zero ΔRMSE; '
                  'dropping `Interactive Catalyst` / `Complex Driver` → large ΔRMSE; '
                  '`Stable Contributor` intermediate.\n')

    df_e = load_csv(output_dir / 'section_E.csv')
    if df_e is not None and len(df_e):
        md.append('## Section E — Interaction-pair synergy\n')
        cols = [c for c in ['exp_name', 'extra_pair_kind', 'extra_pair_A',
                            'extra_pair_B', 'extra_condition', 'test_rmse_cm']
                if c in df_e.columns]
        md.append(_df_to_markdown(df_e[cols], round_to=2))
        md.append('\nFor each pair, super-additivity = Δ_AB − (Δ_A + Δ_B). '
                  'High-Interaction pairs should show *positive* super-additivity; '
                  'matched-Impact low-Interaction pairs should show ≈ 0 (additive). '
                  'A positive gap between the two groups defends the Interaction '
                  'dimension as a measurement SHAP/IG cannot produce.\n')

    df_f = load_csv(output_dir / 'section_F.csv')
    if df_f is not None and len(df_f):
        md.append('## Section F — Nonlinearity gap (Linear vs MLP)\n')
        cols = [c for c in ['exp_name', 'extra_subset_kind', 'test_rmse_cm',
                            'extra_lr_rmse_cm', 'extra_lr_minus_mlp_cm']
                if c in df_f.columns]
        md.append(_df_to_markdown(df_f[cols], round_to=2))
        md.append('\nHigh-Nonlinearity subset → expect *large* (LR − MLP) gap '
                  '(linear cannot exploit these features). '
                  'Low-Nonlinearity subset → expect *small* gap (linear is enough). '
                  'A positive separation defends Nonlinearity as a measurable '
                  'function-class property.\n')

    df_g = load_csv(output_dir / 'section_G.csv')
    if df_g is not None and len(df_g):
        md.append('## Section G — Volatility–uncertainty link\n')
        cols = [c for c in ['exp_name', 'variant_name', 'extra_kind',
                            'test_rmse_cm', 'ensemble_std']
                if c in df_g.columns]
        md.append(_df_to_markdown(df_g[cols], round_to=4))
        md.append('\nFor matched-Impact subsets, removing **high-Volatility** features '
                  'should reduce `ensemble_std` (cross-seed disagreement) more than '
                  'removing low-Volatility features. That would link FFCA Volatility '
                  'to per-feature uncertainty contribution — a measurement absent in '
                  'SHAP/IG.\n')

    out = output_dir / 'FINAL_REPORT.md'
    out.write_text('\n'.join(md))
    print(f'  Wrote {out}')


# ──────────────────────────────────────────────────────────────────────────
# Sanity check (Phase G regression)
# ──────────────────────────────────────────────────────────────────────────
def run_sanity_check():
    print('\n========== SANITY: 24h-gate ensemble RMSE should be ≈ 3.37 cm ==========')
    _import_tf()
    exp = '24hr_perfect_prog_gate_sigmoid'
    spec = find_experiment_spec(exp)
    df, feat_cols, target, _, _ = build_full_feature_matrix(spec)
    Xte, yte = split_by_year(df, target, feat_cols, year=TEST_YEAR)
    models = load_ensemble(original_models_dir(exp), n=N_ENSEMBLE)
    med = ensemble_median(models, Xte)
    val = rmse(yte, med) * 100.0
    print(f'  -> {val:.2f} cm  (expect ~3.37 cm; paper reports 3.40)')
    if abs(val - 3.37) > 0.20:
        print('  WARNING: feature ordering may be off — see project_feature_ordering_bug')
    return val


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-dir', type=Path, default=Path('extra_validation_runs'))
    parser.add_argument('--sections', default='A,B,C,D,E,F,G',
                        help='Comma-separated subset of {A,B,C,D,E,F,G}')
    parser.add_argument('--shard', default='', help='k/N for sharding training jobs')
    parser.add_argument('--gpu-id', type=int, default=None,
                        help='If set, pins this process to CUDA_VISIBLE_DEVICES=<gpu_id>')
    parser.add_argument('--originals-root', type=Path, default=None,
                        help='Override original 30-seed ensembles root (defaults: '
                             'Mac=compound_flooding/mlmiamicompoundfloodpredictions, '
                             'HPC=~/FFCA/compound_flooding_originals)')
    parser.add_argument('--bundle-root', type=Path, default=None,
                        help='Override the perturbation bundle root (where the CSV and '
                             'variant_D ensembles live)')
    parser.add_argument('--reports-root', type=Path, default=None,
                        help='Override corrected-ensemble FFCA reports root')
    parser.add_argument('--csv-path', type=Path, default=None,
                        help='Override the data CSV path')
    parser.add_argument('--list-paths', action='store_true',
                        help='Print detected paths + per-experiment artifact status and exit')
    parser.add_argument('--representative', default=','.join(REPRESENTATIVE_EXPS),
                        help='Comma-separated experiments for sections C/D/G')
    parser.add_argument('--demo-exp', default=DEMO_EXP,
                        help='Demo experiment for sections E/F')
    parser.add_argument('--smoke', action='store_true',
                        help='Use 3 seeds + 200 epochs for a fast smoke run')
    parser.add_argument('--sanity-check', action='store_true',
                        help='Verify Phase G feature ordering (24h-gate RMSE ≈ 3.37 cm)')
    parser.add_argument('--finalize-only', action='store_true',
                        help='Skip all sections and just regenerate FINAL_REPORT.md')
    args = parser.parse_args()

    # Path overrides
    global ORIGINALS_ROOT, BUNDLE_ROOT, REPORTS_ROOT, CSV_PATH
    if args.originals_root:
        ORIGINALS_ROOT = args.originals_root
    if args.bundle_root:
        BUNDLE_ROOT = args.bundle_root
    if args.reports_root:
        REPORTS_ROOT = args.reports_root
    if args.csv_path:
        CSV_PATH = args.csv_path

    # GPU pinning
    if args.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        print(f'Pinned to CUDA_VISIBLE_DEVICES={args.gpu_id}')

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.list_paths:
        print(f'ORIGINALS_ROOT = {ORIGINALS_ROOT}   exists={ORIGINALS_ROOT.exists()}')
        print(f'BUNDLE_ROOT    = {BUNDLE_ROOT}   exists={BUNDLE_ROOT.exists()}')
        print(f'REPORTS_ROOT   = {REPORTS_ROOT}   exists={REPORTS_ROOT.exists()}')
        print(f'CSV_PATH       = {CSV_PATH}   exists={CSV_PATH.exists()}')
        print()
        print(f'{"experiment":40s} {"orig":4s} {"vD":3s} {"rpt":3s}')
        for exp_name, (cat, lead) in EXPERIMENTS.items():
            try:
                orig = original_models_dir(exp_name)
                orig_n = len(list(orig.glob('hypermodel*.h5')))
                orig_tag = f'{orig_n:>3d}'
            except FileNotFoundError:
                orig_tag = ' --'
            vd = variant_d_models_dir(exp_name)
            vd_tag = f'{len(list(vd.glob("hypermodel*.h5"))):>2d}' if vd else '--'
            rpt = (REPORTS_ROOT / cat / exp_name / 'report.json')
            rpt_tag = 'OK' if rpt.exists() else '--'
            print(f'{exp_name:40s} {orig_tag:>4s} {vd_tag:>3s} {rpt_tag:>3s}')
        return

    _import_tf()

    if args.sanity_check:
        run_sanity_check()
        return

    if args.finalize_only:
        # Rebuild canonical per-section CSVs from authoritative per-job JSONs
        # *before* writing the final report. This corrects any per-section CSV
        # that may have been clobbered by concurrent-shard writes during
        # training (the bug that bit §A-G on 2026-05-25 and §H on 2026-05-26).
        sections_to_finalize = [s.strip().upper()
                                for s in (args.sections or 'A,B,C,D,E,F,G').split(',')
                                if s.strip()]
        for section in sections_to_finalize:
            try:
                rebuild_section_csv_from_jsons(section, output_dir)
            except Exception as e:
                print(f'  WARNING: rebuild of section_{section}.csv failed: {e}')
        write_final_report(output_dir)
        return

    sections_requested = [s.strip().upper() for s in args.sections.split(',') if s.strip()]
    representative = [s.strip() for s in args.representative.split(',') if s.strip()]
    demo_exp = args.demo_exp

    print(f'ORIGINALS_ROOT = {ORIGINALS_ROOT}')
    print(f'BUNDLE_ROOT    = {BUNDLE_ROOT}')
    print(f'REPORTS_ROOT   = {REPORTS_ROOT}')
    print(f'CSV_PATH       = {CSV_PATH}')
    print(f'OUTPUT_DIR     = {output_dir}')
    print(f'SECTIONS       = {sections_requested}')
    print(f'SHARD          = {args.shard or "none"}')
    print(f'SMOKE          = {args.smoke}')

    manifest = dict(
        started=time.strftime('%Y-%m-%d %H:%M:%S'),
        gpu_id=args.gpu_id,
        shard=args.shard or None,
        sections=sections_requested,
        smoke=args.smoke,
        representative=representative,
        demo_exp=demo_exp,
    )

    # ── Free sections first ─────────────────────────────────────────────
    if 'A' in sections_requested:
        run_section_a(output_dir, smoke=args.smoke)
    if 'B' in sections_requested:
        run_section_b(output_dir, smoke=args.smoke)

    # ── Build job queues for retraining sections ────────────────────────
    all_jobs: list[TrainJob] = []
    if 'C' in sections_requested:
        all_jobs += build_section_c_jobs(representative)
    if 'D' in sections_requested:
        all_jobs += build_section_d_jobs(representative)
    if 'E' in sections_requested:
        all_jobs += build_section_e_jobs(demo_exp)
    if 'F' in sections_requested:
        all_jobs += build_section_f_jobs(demo_exp)
    if 'G' in sections_requested:
        all_jobs += build_section_g_jobs(representative)

    # Deferred attribution resolution must happen BEFORE sharding so each shard
    # sees the same drop_features lists. Compute on shard==0 only, others wait.
    if 'C' in sections_requested:
        is_first_shard = (not args.shard) or args.shard.startswith('0/')
        attr_cache_done = (output_dir / 'runs' / 'section_C' / '_attr_done.marker').exists()
        if is_first_shard and not attr_cache_done:
            resolve_deferred_section_c(all_jobs, output_dir, smoke=args.smoke)
            (output_dir / 'runs' / 'section_C').mkdir(parents=True, exist_ok=True)
            (output_dir / 'runs' / 'section_C' / '_attr_done.marker').write_text('done')
        else:
            # Wait briefly for the first shard to write attribution cache (max 30 min)
            marker = output_dir / 'runs' / 'section_C' / '_attr_done.marker'
            for _ in range(180):
                if marker.exists():
                    break
                time.sleep(10)
            if not marker.exists():
                print('WARNING: attribution cache marker not found — proceeding anyway')
            resolve_deferred_section_c(all_jobs, output_dir, smoke=args.smoke)

    # Apply sharding to the training queue
    shard_jobs = shard_filter(all_jobs, args.shard)
    print(f'\nTotal training jobs: {len(all_jobs)}; this shard: {len(shard_jobs)}')

    # ── Execute by section so per-section CSVs build incrementally ──────
    for section in ['C', 'D', 'E', 'F', 'G']:
        if section not in sections_requested:
            continue
        section_jobs = [j for j in shard_jobs if j.section == section]
        if not section_jobs:
            continue
        run_train_section(section, section_jobs, output_dir,
                          smoke=args.smoke, shard=args.shard)

    # ── Section F post-processing: fit linear baselines on the kept subsets ──
    if 'F' in sections_requested and (output_dir / 'section_F.csv').exists():
        try:
            spec_demo = find_experiment_spec(demo_exp)
            df_f = pd.read_csv(output_dir / 'section_F.csv')
            new_rows = []
            for _, row in df_f.iterrows():
                kept_str = str(row.get('extra_kept_features', ''))
                if not kept_str or kept_str == 'nan':
                    new_rows.append(row.to_dict())
                    continue
                kept = kept_str.split(';')
                lr_rmse = fit_linear_baseline_rmse(spec_demo, kept)
                gap = lr_rmse - float(row['test_rmse_cm'])
                d = row.to_dict()
                d['extra_lr_rmse_cm'] = round(lr_rmse, 3)
                d['extra_lr_minus_mlp_cm'] = round(gap, 3)
                new_rows.append(d)
            pd.DataFrame(new_rows).to_csv(output_dir / 'section_F.csv', index=False)
        except Exception as e:
            print(f'WARNING: section F LR baseline fit failed: {e}')
            traceback.print_exc()

    # Manifest + report
    manifest['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    manifest_path = output_dir / 'MANIFEST.json'
    manifests = []
    if manifest_path.exists():
        try:
            manifests = json.loads(manifest_path.read_text())
            if not isinstance(manifests, list):
                manifests = [manifests]
        except Exception:
            manifests = []
    manifests.append(manifest)
    manifest_path.write_text(json.dumps(manifests, indent=2))

    write_final_report(output_dir)
    print('\nDone.')


if __name__ == '__main__':
    main()
