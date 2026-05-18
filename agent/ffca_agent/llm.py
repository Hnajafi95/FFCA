"""LLM narration layer: turn deterministic Findings into a layered diagnosis.

The deterministic evaluator produces a flat list of Finding objects. End users
benefit from:

  1. A 1-paragraph executive summary describing the model's health and top issues.
  2. A ranked action list — what to do first, second, third — with each action
     traced back to the underlying rule_ids.
  3. The full per-rule appendix (already rendered by the CLI).

This module wraps Anthropic's Claude with a cached system prompt covering FFCA's
4 dimensions and 8 archetypes, plus a strict output spec. The structured part
of the response is fenced JSON between explicit markers; everything else is
freeform prose. Missing ANTHROPIC_API_KEY raises a clear error — the caller
should fall back to the deterministic-only output.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .case_meta import CaseMeta, NarrationIntent
    from .evaluator import Finding
    from .report import ReportContext
    from .signature_summary import SignatureSummary

DEFAULT_MODEL = "claude-opus-4-7"

_RESPONSE_START = "<DIAGNOSIS>"
_RESPONSE_END = "</DIAGNOSIS>"

_SYSTEM_PROMPT_BASE = """You are the FFCA Diagnostic Agent.

FFCA (Feature-Function Curvature Analysis, Najafi/Luo/Liu 2025, arXiv:2510.27207)
is a post-hoc explainability method for any trained model. For each input
feature it derives four scalars from the model's Hessian on a validation
set:

  - Impact         : how much the feature moves the output. Computed from a
                     Jacobian-norm proxy. High = removing the feature would
                     hurt accuracy a lot.
  - Volatility     : how unstable that influence is across the input space.
                     Low = the feature's effect is roughly the same everywhere;
                     high = the effect depends sharply on where you are in
                     input space (often a sign of spurious correlation or
                     overfitting).
  - Nonlinearity   : how much the model's response to the feature curves
                     versus scales linearly. Low = a linear model would
                     suffice for this feature; high = needs PDPs / ICE plots
                     to interpret.
  - Interaction   : how much the feature's effect depends on other features.
                     Low = effects are additive; high = the feature needs
                     joint analysis with its partners.

Features are clustered into 8 archetypes based on these four scalars:

  - Noise Candidate      : near-zero on all four dimensions. The model has
                           effectively ignored this feature. Candidate for
                           pruning, but check trust score first.
  - Simple Workhorse     : high Impact, low everywhere else. Linear-ish driver.
                           Easy to explain — a linear coefficient tells the
                           whole story. Trust these.
  - Stable Contributor   : mid Impact, low Volatility, low Interaction.
                           Reliable secondary driver. Not flashy; trust them.
  - Non-linear Driver    : high Impact AND high Nonlinearity. Strong but
                           curved relationship (diminishing returns, U-shape,
                           threshold). PDPs/ICE essential — never trust a
                           linear interpretation of these.
  - Hidden Interactor    : low Impact, high Interaction. Looks unimportant in
                           isolation; matters via other features. Removing it
                           in isolation may seem safe but breaks joint effects.
                           Use 2D PDPs or SHAP interaction.
  - Interactive Catalyst : high Impact AND high Interaction. A hub feature
                           that amplifies others. Direct effect is real AND
                           it modifies other features' contributions.
  - Volatile Specialist  : high Impact AND high Volatility. Effect changes
                           with context. Often a spurious-correlation
                           fingerprint (paper App C.6); treat with suspicion
                           and run sliced analysis.
  - Complex Driver       : high on all four dimensions. The hardest features
                           to explain — non-linear, interactive, and context-
                           dependent at once. Need every tool in the box.

A trust score is derived from cross-checkpoint stability: how often the
feature stayed in the same useful archetype across training checkpoints.
The buckets are CONFIDENTLY KEEP, KEEP (stable), MONITOR (borderline),
INVESTIGATE (unstable), CONFIDENTLY PRUNE. High INVESTIGATE counts mean the
model has not converged on stable feature roles — pruning decisions taken
from a single checkpoint won't reproduce on the next one.

A Co-Sensitivity step further clusters features into functional groups that
move together under perturbation. When a cluster is mostly Noise Candidates
(>50%), the group is a safe-to-prune block; when clusters are mixed-utility,
co-sensitivity advises against pruning by group.

You are given the output of a deterministic rule evaluator: a list of Findings.
Each Finding has:

  - rule_id     : stable id (e.g., `trust_instability_high`)
  - kind        : "diagnostic" (something to act on) or "descriptor"
                  (a labeled state of the model)
  - severity    : "critical" | "warn" | "info" | null (descriptors)
  - feature     : present for per-feature findings, null for model-wide
  - diagnosis   : what the rule concluded, already formatted with the
                  relevant numbers
  - recommendation : what to do about it
  - evidence    : the numeric evidence that triggered the rule
  - paper_ref   : where the rule comes from in the paper (or "heuristic" for
                  agent-side conventions, which you should NOT cite as paper
                  claims)

Findings are facts. Do not invent new ones, contradict them, or generalize
beyond what's in evidence. If you notice tension between two findings (e.g.,
`healthy_archetype_distribution` and `trust_instability_high` firing together
— a balanced snapshot but unstable features), surface it as a caveat, do not
hide it.

YOUR TASK — produce a layered diagnosis with four parts:

1. Executive summary (1 paragraph, 60-110 words). State the model's overall
   health, the single most important issue, and the headline recommendation.
   Plain language; a model owner who hasn't read the FFCA paper should
   understand. Quote concrete numbers (e.g., "drift 41%", "30 load-bearing
   features"), not vague qualifiers.

2. Ranked action list. Order by urgency: critical severity first, then warn,
   then info/descriptor highlights worth surfacing. Each action is one to two
   sentences. Combine related findings into a single action (e.g., several
   trust findings → one "investigate trust scores" action). Always cite the
   rule_ids you're combining. Aim for 2-5 actions total; do not list every
   descriptor.

3. Rule-free observations (OPTIONAL — may be empty). Sometimes the structured
   signature summary in the user prompt reveals a pattern that no rule fires
   on: e.g., a single feature with vastly higher Volatility than the others,
   or a top-K Impact curve that decreases late, or a feature whose archetype
   churns across checkpoints without crossing the trust-instability threshold.
   When that happens, list it here — but ONLY if you can cite a specific
   numeric value from the signature_summary block in the user prompt. Each
   observation must include an `evidence` string with that number. Do NOT
   duplicate a rule that already fired. If nothing in the summary is worth
   flagging, return an empty list.

4. Honest caveats. List concerns that should temper the diagnosis:
   - Rules that skipped because of missing signals (especially: "no training
     history → dynamic rules skipped" when that's the case).
   - Findings in apparent tension with each other.
   - Thresholds that look unreliable for this particular model (e.g., a
     Pareto check on a model with very few features).
   - Numbers that are clearly rounding artifacts in the context summary.

OUTPUT FORMAT. Emit your response between the markers below, with valid JSON:

""" + _RESPONSE_START + """
{
  "executive_summary": "...",
  "actions": [
    {
      "priority": 1,
      "title": "Short action title",
      "rationale": "1-2 sentences. Why this is the top action.",
      "rule_ids": ["rule_id_1", "rule_id_2"]
    }
  ],
  "rule_free_observations": [
    {
      "what": "Short description of the pattern.",
      "evidence": "The specific summary-stat value(s) you saw."
    }
  ],
  "caveats": [
    "First caveat sentence.",
    "Second caveat sentence."
  ]
}
""" + _RESPONSE_END + """

Concrete example of good output style:

  executive_summary: "Model is information-rich but has not converged. 175
  features split into a healthy archetype mix (32% Complex Drivers, 24%
  Catalysts), and 30 load-bearing features form a stable backbone. However,
  the signature drifts 41% between the last two checkpoints and 71% of
  features change archetype across training. Do not act on pruning or
  architectural conclusions yet — train longer and re-run FFCA before
  making structural decisions."

  action: priority=1, title="Train longer before acting on this report",
  rationale="The signature is still moving 41% between the last checkpoints
  and 71% of features land in INVESTIGATE because their archetype keeps
  changing. Pruning or architecture decisions made now will not reproduce
  next epoch.", rule_ids=["late_checkpoint_drift", "trust_instability_high"]

Style notes: prefer specific feature names over abstract phrasing when the
finding evidence names them; never claim the paper says something that
isn't in the relevant rule's paper_ref; if no findings are critical or warn,
your executive summary should explicitly say the model is healthy.

After the closing marker you may add a short prose note (1-3 sentences) if
you want to highlight something the structured fields can't carry, but the
structured block is the authoritative output. Do not put anything important
outside it.
"""


@dataclass
class NarrationAction:
    priority: int
    title: str
    rationale: str
    rule_ids: list[str]


@dataclass
class RuleFreeObservation:
    """A pattern the LLM noticed in the signature_summary block that no
    rule fires on. Must cite a specific summary-stat value as evidence.
    Separate from rule-backed findings so reviewers can audit it specifically."""
    what: str
    evidence: str


@dataclass
class NarratedReport:
    executive_summary: str
    actions: list[NarrationAction]
    caveats: list[str]
    appendix_findings: list["Finding"]
    rule_free_observations: list[RuleFreeObservation] = field(default_factory=list)
    raw_response: str = ""
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)


def _build_system_prompt(
    case_meta: "CaseMeta | None" = None,
    intent: "NarrationIntent | None" = None,
) -> str:
    """Assemble the templated system prompt.

    Backward-compatible: case_meta=None and intent=None gives the v0.5 prompt.
    Adding a case_meta splices in its `as_prompt_block()`; adding an intent
    splices in the intent-specific framing.
    """
    parts = [_SYSTEM_PROMPT_BASE]
    if case_meta is not None:
        parts.append(case_meta.as_prompt_block())
    if intent is not None:
        from .case_meta import NarrationIntent as _NI, intent_prompt_block
        block = intent_prompt_block(intent)
        if block:
            parts.append(block)
    return "\n\n".join(parts)


class NarratorError(RuntimeError):
    """Raised when narration cannot proceed (missing API key, invalid response, …)."""


class Narrator:
    """Wraps an Anthropic Claude client and produces a NarratedReport."""

    def __init__(self, model: str = DEFAULT_MODEL, client: Any | None = None, api_key: str | None = None):
        self.model = model
        if client is not None:
            self.client = client
            return
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise NarratorError(
                "anthropic SDK not installed. `pip install 'ffca-agent[narrate]'` "
                "or `pip install anthropic>=0.40`."
            ) from exc
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise NarratorError(
                "ANTHROPIC_API_KEY not set. Either export it or pass api_key= "
                "to the Narrator. Without it, the CLI falls back to deterministic "
                "output only."
            )
        self.client = Anthropic(api_key=key)

    def narrate(
        self,
        findings: list["Finding"],
        ctx: "ReportContext",
        training: dict | None = None,
        max_tokens: int = 4000,
        case_meta: "CaseMeta | None" = None,
        intent: "NarrationIntent | None" = None,
        sig_summary: "SignatureSummary | None" = None,
    ) -> NarratedReport:
        """Narrate findings into a layered diagnosis.

        Backward-compatible: case_meta, intent, and sig_summary are optional.
        With all three set the prompt is templated to be deterministic per
        project + intent, and the LLM can produce rule-free observations
        from the signature summary.
        """
        system_prompt = _build_system_prompt(case_meta, intent)
        user_prompt = self._build_user_prompt(findings, ctx, training, sig_summary)
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=[{
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = _extract_text(resp)
        parsed = _parse_structured_response(text)
        usage = _extract_usage(resp)
        return NarratedReport(
            executive_summary=parsed["executive_summary"],
            actions=[
                NarrationAction(
                    priority=int(a.get("priority", i + 1)),
                    title=str(a.get("title", "")),
                    rationale=str(a.get("rationale", "")),
                    rule_ids=list(a.get("rule_ids", [])),
                )
                for i, a in enumerate(parsed.get("actions", []))
            ],
            caveats=list(parsed.get("caveats", [])),
            rule_free_observations=[
                RuleFreeObservation(
                    what=str(o.get("what", "")),
                    evidence=str(o.get("evidence", "")),
                )
                for o in parsed.get("rule_free_observations", [])
                if o.get("what")
            ],
            appendix_findings=list(findings),
            raw_response=text,
            model=self.model,
            usage=usage,
        )

    # ── prompt construction ────────────────────────────────────────────────

    @staticmethod
    def _build_user_prompt(
        findings: list["Finding"],
        ctx: "ReportContext",
        training: dict | None,
        sig_summary: "SignatureSummary | None" = None,
    ) -> str:
        ctx_block = _summarize_context(ctx, training)
        findings_block = json.dumps(
            [_finding_to_payload(f) for f in findings], indent=2
        )
        sections = [
            "## Model under analysis\n\n" + ctx_block,
            "## Findings from the deterministic rule evaluator\n\n"
            + "```json\n" + findings_block + "\n```",
        ]
        if sig_summary is not None:
            sections.append(
                "## Signature summary (rule-free observation channel)\n\n"
                "The block below is a STRUCTURED, BOUNDED summary of the "
                "raw 4D signatures. Use it ONLY to populate the "
                "`rule_free_observations` field, and ONLY when the pattern "
                "is not already captured by a finding above. Cite specific "
                "numeric values from this block as `evidence`.\n\n"
                "```json\n" + json.dumps(sig_summary.to_dict(), indent=2,
                                          default=_json_default) + "\n```"
            )
        sections.append(
            "Produce the layered diagnosis as specified in your system instructions."
        )
        return "\n\n".join(sections)


def _json_default(o: Any) -> Any:
    """Serialise dataclasses + numpy scalars to JSON-friendly types."""
    if hasattr(o, "to_dict"):
        return o.to_dict()
    if hasattr(o, "__dataclass_fields__"):
        from dataclasses import asdict
        return asdict(o)
    if hasattr(o, "item"):  # numpy scalars
        try:
            return o.item()
        except Exception:
            pass
    return str(o)


# ── helpers ────────────────────────────────────────────────────────────────


def _summarize_context(ctx: "ReportContext", training: dict | None) -> str:
    from .archetypes import PAPER_TO_SNAKE
    lines = [
        f"- n_features: {ctx.n_features}",
        f"- n_checkpoints: {ctx.impact_curve.shape[0]}",
        f"- mean Impact: {float(ctx.impact.mean()):.4g}",
        f"- max Impact: {float(ctx.impact.max()):.4g}",
        f"- mean Volatility: {float(ctx.volatility.mean()):.4g}",
        f"- mean Nonlinearity: {float(ctx.nonlinearity.mean()):.4g}",
        f"- mean Interaction: {float(ctx.interaction.mean()):.4g}",
    ]
    # archetype distribution
    arch_counts: dict[str, int] = {}
    for a in ctx.archetypes:
        arch_counts[str(a)] = arch_counts.get(str(a), 0) + 1
    if arch_counts:
        dist = ", ".join(
            f"{name}={count}" for name, count in sorted(arch_counts.items(), key=lambda kv: -kv[1])
        )
        lines.append(f"- archetype distribution: {dist}")
    # trust distribution
    trust_counts = {k: b.count for k, b in ctx.trust_buckets.items() if b.count > 0}
    if trust_counts:
        lines.append("- trust distribution: " + ", ".join(f"{k}={v}" for k, v in trust_counts.items()))
    if ctx.cosens is not None:
        lines.append(
            f"- co-sensitivity: best_nc_fraction={ctx.cosens.get('best_nc_fraction', 0):.2f}, "
            f"perm-p={ctx.cosens.get('permutation_p', 1):.3g}, "
            f"silhouette={ctx.cosens.get('silhouette', 0):.2f}"
        )
    if training:
        lines.append(f"- training history attached: keys={sorted(training.keys())}")
    else:
        lines.append("- training history attached: none (dynamic rules will skip)")
    return "\n".join(lines)


def _finding_to_payload(f: "Finding") -> dict:
    return {
        "rule_id": f.rule_id,
        "rule_name": f.rule_name,
        "kind": f.kind,
        "category": f.category,
        "severity": f.severity,
        "feature": f.feature,
        "diagnosis": f.diagnosis,
        "recommendation": f.recommendation,
        "evidence": f.evidence,
        "paper_ref": f.paper_ref,
    }


def _extract_text(resp: Any) -> str:
    """Pull the text content out of an Anthropic Messages response."""
    if hasattr(resp, "content"):
        chunks = []
        for block in resp.content:
            text = getattr(block, "text", None)
            if text:
                chunks.append(text)
        return "".join(chunks)
    return str(resp)


def _extract_usage(resp: Any) -> dict[str, int]:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return {}
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "cache_creation_input_tokens": int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
        "cache_read_input_tokens": int(getattr(usage, "cache_read_input_tokens", 0) or 0),
    }


_FENCE_PATTERN = re.compile(
    rf"{re.escape(_RESPONSE_START)}\s*(.*?)\s*{re.escape(_RESPONSE_END)}",
    re.DOTALL,
)


def _parse_structured_response(text: str) -> dict:
    m = _FENCE_PATTERN.search(text)
    if not m:
        raise NarratorError(
            f"narrator response missing {_RESPONSE_START}…{_RESPONSE_END} fence. "
            f"Got: {text[:400]!r}"
        )
    payload = m.group(1).strip()
    # tolerate leading ```json fences inside the markers
    payload = re.sub(r"^```(?:json)?\s*", "", payload)
    payload = re.sub(r"\s*```$", "", payload)
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise NarratorError(
            f"narrator response not valid JSON: {exc}\nPayload: {payload[:400]!r}"
        ) from exc
