# v0.6 narration notes — what the new layers added

Three v0.6 changes were rolled together:

1. **CaseMeta + questionnaire.** A per-project `case_meta.json` is now
   spliced into the system prompt: project name, model architecture
   (MLP/CNN/RNN/Transformer/other), task type, target name + units,
   domain, pretrained flag, feature naming convention, free-text notes.
   The narrator now knows whether it is reading a flooding regression
   model or a Waterbirds vision classifier without us telling it
   case-by-case in the prompt.

2. **Per-narration intent.** One of `audit / diagnose / prune /
   compare / free`. Templated into the system prompt as a short framing
   block. Each intent biases the ranked action list toward what the user
   actually wants to decide from this analysis.

3. **Rule-free observation channel.** A bounded, structured signature
   summary (top-K features per dimension, curve-shape descriptors with
   monotonicity / spike / late-drift flags, cross-checkpoint CoV
   percentile, archetype-churn count, top interaction pairs when
   present) is added to the user prompt. The LLM is told to populate
   a NEW `rule_free_observations` field with patterns from this
   summary that no rule already fires on, each citing a specific
   numeric value as evidence.

## What the v0.6 round actually produced

| | v0.5 | v0.6 |
|---|---:|---:|
| Engineered cases re-narrated | 6 | 6 |
| Flooding gate cases re-narrated | 8 | 8 |
| Cost                            | $1.04 (engineered) + $1.40 (gate) = $2.44 on the 14 cases  | **$3.82** |
| Rule-free observations          | 0 (channel did not exist) | **48** (3-4 per case) |
| Diagnostic rule IDs that fired  | identical                 | **identical** |
| Executive summary length        | 77-92 words               | 77-116 words (similar) |
| Hallucinations observed         | 0 across all 96 v0.5 calls | 0 across all 14 v0.6 calls |

**Key result: rule-firings are unchanged between v0.5 and v0.6**, so we
didn't degrade coverage of the named pathologies. The new value-add is
entirely in the rule-free observations.

## A taste of what the rule-free channel produced

These are observations the v0.5 narrator could not have made — it
literally did not see the underlying signature summary.

### Flooding `gate/before_3hr` (compare intent)
> "Interaction Top-K curve spiked to 1.258 at checkpoint 17 versus
> initial 0.562 and final 0.526 — a >2× transient that has since
> collapsed. Suggests training passed through a phase where features
> were more entangled than they are now."
> _evidence: interaction_topk_curve_shape: initial=0.562, peak=1.258 @ ckpt 17, final=0.526, has_spike=true._

No rule in the v0.5 rulebook detects "Interaction had a transient
mid-training that has since collapsed."

### Flooding `gate/after_6hr` (compare intent)
> "Pervasive archetype churn beyond the trust-instability headline —
> almost every feature changed role at least three times during
> training, suggesting the instability is system-wide, not a tail
> effect of a few unstable features."
> _evidence: n_features_with_archetype_churn_ge_3 = 53 of 62 (85%)._

The v0.5 rule `trust_instability_high` fires on the *trust* bucket
fractions; the new channel adds the *archetype churn count* which is
a different signal.

### Flooding `gate/after_24hr` (compare intent)
> "`gwl_t-2` stands out: third-highest Impact (0.147) but flagged
> Volatile Specialist with INVESTIGATE trust, while its neighbors
> gwl_t-1 and gwl_t-3 are stable Catalyst/Contributor. Worth a sliced
> look to see if it is genuinely context-dependent or just unsettled."

The agent identified a *physically interpretable* mismatch — `gwl_t-2`
is the odd one out among the gwl lag series — using the feature naming
convention we provided through case_meta. v0.5 could not have done this
because it had no domain context.

### Waterbirds (audit intent)
> "Top-K Impact more than doubled across training while top-K
> Volatility fell to ~27% of its initial value — the model sharpened
> its reliance on a few channels rather than diversifying."
> _evidence: impact_topk fold_change_final_over_initial=2.23 (monotonic increasing); volatility_topk fold_change=0.269._

This is a precise quantitative claim about *training dynamics* that no
rule fires on; it would have to be a hand-crafted detector if we wanted
it in the rulebook.

### Credit Loan v0.5 (diagnose intent)
> "Top-K Volatility curve grew 138× from initial to final checkpoint
> and shows late drift, dwarfing the Impact (12×) and Interaction
> (12×) growth — Volatility is rising faster than useful signal,
> consistent with the overfitting alarm."

This frames the existing `overfitting_volatility_spike` rule firing as
part of a larger ratio story (138× vs 12×), which is information no
single rule's evidence string carried.

## Determinism & cost

- **Cache utilisation.** Each unique case_meta creates a new cache
  entry. The 6 engineered cases all have different case_meta, so each
  paid `cache_creation_input_tokens` on the first call (~3300 tokens
  written into cache). The 8 flooding gate cases share one case_meta,
  so the cache hit from call 8 onward (`cache_read_input_tokens=3341`).
  This is correct behaviour — different system prompt means a different
  cache key.

- **Cost-per-call.** Average $0.27 per call in v0.6 (vs ~$0.17 in v0.5).
  The increase is mostly the larger system prompt + the signature
  summary block in the user prompt. With cohort-shared case_meta the
  per-call cost would drop further.

- **Determinism.** Same case_meta + same intent + same rulebook → same
  system prompt → reproducible narrations on the same report.json (modulo
  LLM sampling temperature). The questionnaire is the deterministic
  on-ramp: walking the same answers produces the same case_meta.

## Operational implications

1. **The rule-free channel is a hedge, not a replacement.** Most
   v0.6 observations were complementary to existing rules. None of
   them invalidated a rule; some pointed at patterns we might want
   to formalise as rules in v0.7 (e.g., a "training-passes-through-
   transient" rule from the interaction-spike examples).

2. **Domain context produces sharper feature-level callouts.** The
   `gwl_t-2 vs gwl_t-1` observation is the clearest example: the
   agent used the feature naming convention from case_meta to spot a
   physically-meaningful anomaly. Without the convention, it would
   have called the feature by its identifier and missed the cross-lag
   comparison.

3. **Intent biases the action list but not the findings.** A
   `compare` intent produces before/after-flavoured action items;
   `audit` puts critical findings up top; `diagnose` resolves
   cross-rule tensions. None of these intents changed *which* rules
   fired — they only changed how the actions were ranked.

## Artifacts

```
ffca_agent/case_meta.py            — CaseMeta + NarrationIntent + intent_prompt_block
ffca_agent/signature_summary.py    — signature_summary(ctx) + CurveShape + TopKEntry
ffca_agent/llm.py                  — _build_system_prompt + sig_summary integration
ffca_agent/cli.py                  — --case-meta, --intent, --with-signature-summary, --questionnaire flags
tests/test_case_meta.py            — 11 new tests
tests/test_signature_summary.py    — 11 new tests
case_studies/narrate_v06.py        — 14-case re-narration driver
case_studies/diff_v05_v06.py       — v0.5 vs v0.6 diff
FFCA_runs_results_v04_real/
  <case>/diagnosis_v6.md           — full v0.6 narration per case
  <case>/findings_v06.json         — structured v0.6 findings (incl. rule-free obs)
  summary_v06.json
  narration_v06_usage.json
  diff_v05_v06.json                — per-case before/after comparison
```

74 tests at v0.5 → 105 tests at v0.6, all passing.
