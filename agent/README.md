# FFCA Diagnostic Agent

A reproducible, deterministic + LLM-augmented diagnosis layer on top of the
[FFCA package](../).

The FFCA package produces a 4-D signature per feature (Impact, Volatility,
Non-linearity, Interaction), an 8-archetype classification, and a trust
score per feature. That's a lot to read by hand. This agent turns those
outputs into a layered diagnosis you can take action on:

1. A **rulebook** (35 versioned rules, YAML, JSON-Schema validated) that
   translates the paper's prose-level diagnostics into machine-readable
   triggers over a typed signal namespace.
2. A **deterministic evaluator** that loads an FFCA `report.json`, runs the
   rulebook, and emits structured `Finding` dataclasses with citations
   back to the rule ID + the report values that triggered them.
3. An **LLM narration layer** (Claude Opus 4.7) that sees the findings — not
   the raw signatures — and produces a layered diagnosis: executive
   summary, ranked action list with rule-ID citations, optional rule-free
   observations, caveats. Grounding guarantee: every concrete numeric
   claim traces to a rule and an evidence value. Across 96 calls in our
   validation experiments, zero hallucinations were observed.

The agent works on any model the FFCA package can analyse (tabular,
channel-wise CNN, pixel-wise CNN). It is currently dependent on the FFCA
package + Python 3.9+; the LLM layer is optional and silently degrades
to deterministic-only output when no API key is available.

## Install

```bash
# From the repo root
pip install -e ./                       # FFCA package
pip install -e ./agent                  # this agent

# Optional: narration layer (requires Anthropic API key at run time)
pip install -e "./agent[narrate]"
```

## Quick start

```bash
# Run the deterministic rulebook on an FFCA report
python -m ffca_agent.cli /path/to/report.json --format md

# Add LLM narration (requires ANTHROPIC_API_KEY)
python -m ffca_agent.cli /path/to/report.json --narrate

# v0.6: project-aware narration
python -m ffca_agent.cli /path/to/report.json --narrate \
    --case-meta my_project.json --intent diagnose --with-signature-summary

# Build a case_meta.json interactively
python -m ffca_agent.cli --questionnaire my_project.json
```

## What's inside

```
agent/
├── ffca_agent/             — Python module (CLI + library)
│   ├── case_meta.py        — per-project context + intent enum (v0.6)
│   ├── evaluator.py        — rulebook → Findings
│   ├── llm.py              — LLM narration (Claude Opus 4.7) + cached system prompt
│   ├── report.py           — adapter from FFCA report.json → typed signal namespace
│   ├── signature_summary.py— rule-free observation channel (v0.6)
│   ├── timeseries.py       — spike / plateau / collapse operators
│   ├── training.py         — training-history attach (val/train gap, val curve, …)
│   └── vision.py           — vision-metrics attach (FBR / COM / minority-acc / gap)
├── rulebook/
│   ├── ffca_rules.yaml     — 35 rules, version 0.5.0
│   ├── schema.json         — JSON Schema (Draft 2020-12)
│   └── validate.py         — schema validator + coverage check
├── tests/                  — 105 pytest tests; rules, parser, signature summary
├── case_studies/           — runnable scripts + the v0.6 HPC validation driver
└── docs/                   — validation-gate reports + case-study writeup
```

## Validation evidence

The rulebook has been validated four times against engineered positive controls
(the `case_studies/run_all.py` cases) and once against a healthy negative
control (`case_studies/negative_control.py`). See `docs/VALIDATION_GATE_v0.5.md`
for the latest scorecard and `docs/NEGATIVE_CONTROL.md` for the negative-control
outcome.

The agent has also been applied to a real-world regression problem (compound
flooding water-level forecasting; 40 FFCA reports across 20 retrained models
with measured RMSE outcomes). The agent's per-report verdict on the
post-pruning state has **100% recall** on the 3 measurably-degraded
experiments, **94% specificity**, **75% precision**. See
`docs/COMPOUND_FLOODING_CASE_STUDY.md`.

v0.6 (case context + per-narration intent + rule-free observation channel)
is documented in `docs/V06_NARRATIONS_NOTES.md`. A single HPC validation
script (`case_studies/v06_hpc_validation.py`) runs four independent
experiments — model-zoo false-positive sweep, SHAP/IG/FFCA head-to-head,
v0.6 intent ablation, determinism re-runs.

## Two paper-level corrections this work surfaced

The original FFCA paper (Najafi/Luo/Liu, 2025) claims two things the
validation gate did not reproduce as stated:

1. **"Volatile Specialist" is not the empirical signature of spurious
   correlations.** FFCA's Volatility scalar measures within-domain gradient
   variance, not train/val divergence. Spurious features land in
   Interactive Catalyst in our experiments. The v0.5 rulebook decouples
   the spurious-correlation detector from the archetype label and uses
   `feature.impact_dominance + training.val_train_gap` instead.
2. **Hierarchical learning is a relative growth-rate property, not a
   "spike-then-plateau" curve shape.** With denser checkpoint sampling
   our four tabular cases show monotone growth of both top-k Impact and
   top-k Interaction; there is no clean plateau. The v0.5 rule fires on
   `model.interaction_to_impact_growth_ratio > 2`.

Both corrections are folded into the v0.5 rulebook and the paper revision.

## Honest limits

- **The agent is bounded by the FFCA report's scope.** Architectures the
  FFCA package does not yet adapt for (RNN, Transformer beyond its
  channel adapter) are not covered.
- **The negative control is n=1.** A rigorous false-positive sweep on
  ~15–20 public model-zoo models is included as `case_studies/v06_hpc_validation.py`
  but has not been executed at scale yet.
- **v0.6 rule-free observations are LLM-generated and explicitly flagged
  as such** in the CLI output. They cite a specific summary value as
  evidence, but the inference the LLM drew from that value may still be
  wrong. Treat as hypotheses to check, not as conclusions.

## Citation

If you use this in a publication, please cite the FFCA paper (link in the
repo root README) and credit the agent as a separate layer.

## License

MIT — same as the FFCA package.
