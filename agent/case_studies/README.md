# Paper case-study revalidation

End-to-end retraining + FFCA + agent diagnosis of the 5 case studies referenced
in the FFCA paper (Najafi/Luo/Liu 2025). Designed to run on an HPC node with a
GPU and `scp` results back.

## What it produces

For each case the orchestrator writes one subdirectory under your chosen
`--output-dir`. Every subdirectory contains:

- `history.json` — per-epoch loss / accuracy in Keras-history format
- `case_meta.json` — case description, expected rules, snapshot epochs
- `checkpoints/ep_NNN.pt` — PyTorch `state_dict()` snapshots for FFCA replay
- `report.json` — full FFCA report (4-D signatures across all checkpoints,
  trust scores, co-sensitivity groups)
- `report.md` — human-readable FFCA report
- `plots/*.png` — radar / archetype distribution / impact ranking / interaction
  CI / archetype evolution / trust scatter / co-sensitivity plots
- `vision_metrics.json` — *Waterbirds only*: per-epoch FBR / COM-distance /
  minority-accuracy curves for the vision rules
- `diagnosis.md` — *with `--narrate`*: layered LLM diagnosis (executive
  summary + ranked actions + caveats + findings)

The root of `--output-dir` also holds `SUMMARY.md`, `summary.json`, and a copy
of the `ffca_rules.yaml` used so the artifacts are self-describing.

## Cases (v0.4 setup)

| Case                            | Dataset                  | Architecture        | Expected rules                                                  | Epochs |
|---------------------------------|--------------------------|---------------------|-----------------------------------------------------------------|-------:|
| `credit_loan`                   | UCI German Credit        | MLP (128, 64)       | `hierarchical_learning_confirmed`                               |     25 |
| `california_housing_leak`       | sklearn + leaked target  | MLP (64, 32)        | `data_leakage_immediate_dominance`                              |     60 |
| `california_housing_spurious`   | sklearn + spurious feat. | MLP (64, 32)        | `spurious_correlation_train_val_gap` (v0.3 renamed)             |     60 |
| `bike_sharing`                  | UCI Bike Sharing         | MLP (512, 256)      | `overfitting_volatility_spike`, `late_checkpoint_drift`         |    500 |
| `wine_quality`                  | UCI Wine Quality (red)   | Linear(11, 1)       | `insufficient_capacity`                                         |     60 |
| `waterbirds`                    | WILDS Waterbirds         | ResNet-18 IN-pretr. | `shortcut_learning_drift_epoch`                                 |     40 |

**v0.4 setup changes vs v0.3** (driven by the validation gate findings; see
`FFCA_runs_results/VALIDATION_GATE_v0.3.md`):

- `credit_loan` shortened from 80 → 25 epochs. Prior setup overfit hard
  (val_loss climbed 0.6 → 1.55), masking the hierarchical-learning growth
  signature. 25 epochs stops near the val_loss minimum.
- `california_housing_spurious` correlation strengthened from `0.9*y + N(0,0.3)`
  → `0.99*y + N(0,0.05)`. Prior setup gave the spurious feature only 1.9× mean
  Impact, below the new rule's 3× threshold.
- `bike_sharing` hidden layers widened from (128, 64) → (512, 256), epochs
  raised from 200 → 500. Prior setup converged cleanly without overfitting.
- `wine_quality` switched from MLP(4,) → pure linear (`Linear(11, 1)`). The
  4-unit hidden layer still developed 27% Complex Drivers; only true linear
  collapse forces the insufficient_capacity signature.
- `waterbirds` uses ImageNet-pretrained ResNet-18 (not from scratch) with LR
  warmup + cosine schedule, 40 epochs. From-scratch ResNet-18 never learned
  a stable representation in 20 epochs — FBR oscillated chaotically.

Each case has a deliberately-tuned setup (capacity, training length, injected
features) to make the target phenomenon emerge. See `run_all.py` for the
exact specs.

## Setup on HPC

```bash
# (1) Get the code
git clone /path/to/FFCA_agent       # or scp the repo from Mac
cd FFCA_agent
pip install -e .                    # installs the rulebook + agent CLI

# (2) Install case-study deps
pip install -r case_studies/requirements.txt

# (3) Make the FFCA package importable. Two options:
#     a) Install it as a wheel/source dist:
pip install -e /path/to/FFCA/FFCA_package
#     b) OR let the orchestrator auto-resolve via sys.path:
#        if your FFCA package sits at `../FFCA/FFCA_package` relative to
#        this repo, it's picked up automatically.

# (4) Set the API key if you want narration to run on HPC. Better practice:
#     skip narration on HPC, scp results back, narrate locally.
export ANTHROPIC_API_KEY="sk-ant-..."   # optional
```

## Running

```bash
# Everything (5 cases). On H200, expect ~15-30 min for tabular + 30-60 min for
# Waterbirds (depending on data download speed).
python case_studies/run_all.py --output-dir results/

# Skip Waterbirds (just the 4 tabular cases; ~15-25 min total on H200)
python case_studies/run_all.py --output-dir results/ --skip-waterbirds

# A single case
python case_studies/run_all.py --output-dir results/ --cases bike_sharing

# Smoke test (2 epochs each)
python case_studies/run_all.py --output-dir _smoke/ --cases credit_loan,wine_quality \
    --epoch-override 2 --skip-waterbirds

# With agent narration (writes diagnosis.md for each case)
python case_studies/run_all.py --output-dir results/ --narrate
```

Run with `python -u` if you want unbuffered stdout for `tail -f` monitoring.

## Slurm template

```bash
#!/bin/bash
#SBATCH --job-name=ffca_cases
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=ffca_cases.%j.log

module load python cuda          # or whatever your cluster wants
source /path/to/venv/bin/activate
cd /path/to/FFCA_agent

python -u case_studies/run_all.py --output-dir results/
```

## Bringing results back

The whole `results/` directory is self-contained. Pull it back with:

```bash
scp -r hpc:/path/to/FFCA_agent/results ./results/
```

Then narrate locally (cheaper for iteration, keeps the API key off HPC):

```bash
python -m ffca_agent.cli results/bike_sharing/report.json \
    --training-history results/bike_sharing/history.json --narrate \
    > results/bike_sharing/diagnosis.md
```

For Waterbirds, also pass `--vision-metrics results/waterbirds/vision_metrics.json`
(coming in a follow-up CLI flag — for now, attach manually via a short Python
script using `ctx.attach_vision_metrics(VisionMetrics.from_json(...))`).

## Validation gate (what success looks like)

The point of this run is to validate the agent's rulebook against
ground-truth phenomena from the paper. Per case, success means:

1. The rule listed in "Expected rules" actually fires in `report.json`'s
   evaluation (visible in the agent's findings list).
2. The `diagnosis.md` executive summary names the phenomenon in plain
   language without inventing facts.
3. Caveats correctly call out anything the rulebook misses.

A failed case means: rule didn't fire (the data didn't exhibit the
phenomenon strongly enough — either need more epochs, stronger signal
injection, or the rule's threshold needs tightening).

## Notes / known limitations

- **Waterbirds foreground**: WILDS doesn't ship segmentation masks. The
  Waterbirds case approximates foreground as the centered 50% box of the
  image; this is reasonable because Waterbirds construction puts birds in
  the center, but it isn't perfect. Documented in `vision_metrics.json`'s
  `notes` field.
- **FFCA package install path**: the orchestrator probes
  `../FFCA/FFCA_package` relative to the repo. If your HPC layout differs,
  either install the package properly with pip or edit the `FFCA_PKG_CANDIDATE`
  line near the top of `run_all.py`.
- **`bike_sharing` is the slowest tabular case** (200 epochs × ~17K rows).
  On H200 it's still under 5 minutes.
- The first Waterbirds run downloads ~10 GB. Set `--output-dir` somewhere
  with enough disk and check your HPC's egress policy.
