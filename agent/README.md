# FFCA Diagnostic Agent

A reproducible diagnosis layer for any FFCA report.

The [FFCA package](../) computes a four-dimensional geometric signature
per feature (Impact, Volatility, Non-linearity, Interaction), an
eight-archetype classification, and a per-feature trust score. That
output is rich --- and hard to read at scale. This agent turns it into a
short, layered, citation-anchored diagnosis you can act on.

---

## What you get

For any FFCA `report.json` the agent produces:

- An **executive summary** in plain language (one paragraph, with concrete
  numbers, not vague qualifiers).
- A **ranked action list** (two to five items) where each item cites the
  rule(s) it derives from.
- A **findings appendix** showing every rule that fired with its full
  diagnosis, recommendation, and evidence.
- Optional **rule-free observations** flagged explicitly as
  lower-confidence (patterns the LLM noticed in a structured signature
  summary that no rule fires on).

Every quantitative claim in the summary traces back to a deterministic
rule firing.

---

## Quick start (three commands)

```bash
# 1. Install
pip install -e .                  # the FFCA package, from the repo root
pip install -e ./agent             # this agent
pip install -e "./agent[narrate]" # add Anthropic SDK for the LLM layer

# 2. Run the deterministic rulebook on an existing FFCA report
python -m ffca_agent.cli /path/to/report.json --format md

# 3. Layer LLM narration on top
export ANTHROPIC_API_KEY=sk-ant-...
python -m ffca_agent.cli /path/to/report.json --narrate
```

That's the full happy path. The first two commands work offline; the
third needs an Anthropic API key.

---

## What if I don't have an FFCA report yet?

Use the FFCA package directly to produce one. The shortest path is:

```python
from ffca import FFCAReport, TabularAdapter, CheckpointLoader
# Load your trained PyTorch model + a validation DataLoader
adapter = TabularAdapter(model, feature_names=feature_names)
ckpts = CheckpointLoader(factory, [("ep_10", "checkpoint.pt")])
FFCAReport(adapter, val_loader).run(checkpoints=ckpts).save("out/")
# out/report.json is now ready for the agent
```

A complete end-to-end example for the four engineered case studies
(California Housing leak, California Housing spurious correlation, Bike
Sharing overfit, German Credit) lives at
`case_studies/run_all.py`.

---

## Run the validation harness

A single self-contained script reproduces every validation experiment
in the paper: model-zoo specificity sweep, SHAP/IG/FFCA head-to-head,
intent ablation, re-run determinism.

```bash
# Full suite (needs an API key for the intent + determinism sections)
python agent/case_studies/v06_hpc_validation.py \
    --key-file /path/to/key.txt \
    --out-dir results/v06_validation/

# Or run only the API-free sections
python agent/case_studies/v06_hpc_validation.py \
    --out-dir results/v06_validation/ \
    --skip-intent --skip-determinism
```

The script auto-trains the four engineered tabular cases on first
launch (around five minutes on a GPU) and reuses the cached artifacts
on subsequent runs. Output: per-section JSON + a top-level
`VALIDATION_REPORT.md`.

---

## Per-project context (recommended)

You can give the agent a small project description --- model
architecture, task type, target name, domain, feature naming
convention --- that gets templated into the narration prompt. This
sharpens per-feature callouts (the agent will, for example, recognise
that `gwl_t-2` is the third lag of a "current groundwater level"
channel rather than a generic identifier).

```bash
# Build a case_meta.json interactively (asks 10 questions)
python -m ffca_agent.cli --questionnaire my_project.json

# Use it
python -m ffca_agent.cli report.json --narrate \
    --case-meta my_project.json \
    --intent diagnose \
    --with-signature-summary
```

Per-narration intent (`audit`, `diagnose`, `prune`, `compare`, `free`)
biases the ranked action list toward what you are deciding from this
report.

---

## Trust hierarchy in the output

The rendered diagnosis separates two kinds of output:

1. **Rule-backed findings** (the executive summary, the ranked actions,
   the appendix). Every numeric claim cites a rule ID and that rule's
   evidence value. The narration layer never makes up numbers.
2. **Rule-free observations** (an optional separate section, rendered
   with a `:warning:` heading and an explicit disclaimer). The LLM is
   told to flag any pattern it sees in the signature summary that no
   rule fires on. These cite a specific summary value, but the
   *inference* from that value is the LLM's --- treat as hypotheses to
   check, not as conclusions.

This separation is enforced by the renderer; the LLM is instructed to
respect it; spot-checks across nearly a hundred narrations in
validation found no quantitative claim that escaped a rule citation.

---

## Common pitfalls

| Symptom                                       | Cause                                                             | Fix                                                              |
|-----------------------------------------------|-------------------------------------------------------------------|------------------------------------------------------------------|
| `ModuleNotFoundError: No module named 'ffca'` | The FFCA package isn't installed in the active venv.              | `pip install -e .` from the repo root.                           |
| `--narrate` silently falls back               | `ANTHROPIC_API_KEY` not set, or the network can't reach the API.  | Export the key, or pass `--key-file PATH` on the wrapper scripts.|
| Dynamic rules silently skip                   | No training history attached to the report.                       | Pass `--training-history PATH`, or use `--epochs-aligned` if the FFCA checkpoints correspond to training epochs. |
| Vision rules don't fire on a CNN              | No `vision_metrics.json` attached.                                | Pass `--vision-metrics PATH`. The validation script under `case_studies/` shows how to compute FBR / COM / minority-acc curves. |
| One "fragile" descriptor fires on a healthy model | The descriptor is a label, not a pathology call.                | Read the severity tag. Only `critical` (and to a lesser extent `warn`) require action.                          |

---

## Layout

```
agent/
├── ffca_agent/               # Python module (CLI + library)
│   ├── case_meta.py          # per-project context + intent enum
│   ├── evaluator.py          # rulebook → Findings
│   ├── llm.py                # narration layer (cached system prompt)
│   ├── report.py             # FFCA report.json → typed signal namespace
│   ├── signature_summary.py  # rule-free observation channel
│   ├── timeseries.py         # spike / plateau / collapse operators
│   ├── training.py           # training-history attach
│   └── vision.py             # vision-metrics attach (FBR / COM / gap)
├── rulebook/
│   ├── ffca_rules.yaml       # 35 rules
│   ├── schema.json           # JSON Schema (Draft 2020-12)
│   └── validate.py           # schema validator + coverage check
├── tests/                    # pytest suite covering evaluator, rules,
│                             # parsers, case-meta + intent, summary
├── case_studies/             # runnable scripts:
│   ├── run_all.py            #   trains the four engineered cases
│   ├── negative_control.py   #   healthy bike-sharing model
│   ├── sensitivity_sweep.py  #   threshold perturbations
│   ├── v06_hpc_validation.py #   full validation suite (single script)
│   ├── llm_judge_determinism.py  # LLM judge for rerun semantics
│   ├── narrate_flooding_all.py   # case-study narration driver
│   └── flooding_figures*.py  #   case-study figure generators
├── docs/                     # validation report, sensitivity table,
│                             # negative-control note, case-study writeup,
│                             # narration-layer notes
├── pyproject.toml
└── README.md                 # this file
```

---

## Citing

If you use this agent in a publication, please cite the FFCA paper (see
the repo root) and credit the agent as a separate layer over it.

## License

MIT --- same as the FFCA package.
