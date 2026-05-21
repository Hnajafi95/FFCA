"""Per-project case metadata that templates the LLM narrator's system prompt.

The FFCA report alone is dimension-agnostic — it doesn't know whether a
feature is from a flooding model or a credit-risk model, whether the task
is regression or classification, or what the user wants out of the
analysis (audit? prune? diagnose?). This module captures that context in
a small, persistent JSON file and exposes it as a structured prompt block
the narrator splices into the system prompt.

Design goals:
  - Deterministic. Same answers → same system prompt → comparable narrations.
  - Optional. case_meta is always optional; if absent, the narrator falls
    back to the v0.5 generic primer (backward compatible).
  - Cheap. Read once, cached with the system prompt.

Usage:
    from ffca_agent.case_meta import CaseMeta, NarrationIntent

    meta = CaseMeta.from_json("case_meta.json")
    # ... or interactively:
    meta = CaseMeta.from_questionnaire()
    meta.save("case_meta.json")

    narrator.narrate(findings, ctx, case_meta=meta,
                     intent=NarrationIntent.DIAGNOSE)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path


# ── Enums ──────────────────────────────────────────────────────────────────


class ModelArchitecture(str, Enum):
    MLP = "mlp"
    CNN = "cnn"
    RNN = "rnn"
    TRANSFORMER = "transformer"
    OTHER = "other"


class TaskType(str, Enum):
    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    VISION_CLASSIFICATION = "vision_classification"


class NarrationIntent(str, Enum):
    """What the user wants out of this narration.

    Picks where the agent focuses its ranked action list:
      - AUDIT:    is this model safe to ship?  → highlight critical findings
      - DIAGNOSE: why is the model behaving this way?  → root-cause framing
      - PRUNE:    which features to drop?      → trust-bucket + cosens first
      - COMPARE:  before/after, A/B, etc.      → differential framing
      - FREE:     no specific intent           → v0.5 generic narration
    """
    AUDIT = "audit"
    DIAGNOSE = "diagnose"
    PRUNE = "prune"
    COMPARE = "compare"
    FREE = "free"


class CheckpointKind(str, Enum):
    """What the FFCA 'checkpoints' actually are.

    This selects how rules that look across the checkpoint axis are
    interpreted, and shapes the narrator's language.

      - EPOCH: a time-ordered series of checkpoints from one training run
               (e.g., save every 10 epochs). Drift, archetype-trajectory,
               and trust-instability-via-archetype-flips are meaningful.
               High INVESTIGATE rate suggests the model has not converged.

      - SEED:  independently-trained models with different random seeds,
               but otherwise identical training procedure (a deep
               ensemble). There is no time ordering between them; drift
               between consecutive "checkpoints" is meaningless. High
               cross-seed disagreement is the "ensemble in disguise"
               signature — different seeds find different feature roles
               that produce similar accuracy. More training will NOT
               resolve it.

      - MIXED: a hybrid (e.g., multiple seeds × multiple epochs each).
               Currently treated as EPOCH for backward compatibility but
               flagged as ambiguous in narrations.
    """
    EPOCH = "epoch"
    SEED = "seed"
    MIXED = "mixed"


# ── CaseMeta ───────────────────────────────────────────────────────────────


@dataclass
class CaseMeta:
    """Per-project context for the narrator."""

    # Required-ish
    project_name: str = "untitled"
    model_architecture: ModelArchitecture = ModelArchitecture.MLP
    task_type: TaskType = TaskType.REGRESSION
    target_name: str = "target"
    target_units: str = ""

    # Optional but useful
    domain: str = ""
    pretrained: bool = False
    n_seeds: int = 1
    # checkpoint_kind disambiguates the FFCA checkpoint axis. None = legacy
    # (treated as EPOCH for backward compat), but the questionnaire now
    # requires the user to choose explicitly. See CheckpointKind docstring.
    checkpoint_kind: CheckpointKind | None = None
    feature_naming_convention: str = ""  # free-text gloss
    notes: str = ""

    # Used by the narrator UI but not the prompt
    schema_version: str = "0.7.0"

    # ── round-trip ──────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, d: dict) -> "CaseMeta":
        # Convert string enums back to enum instances
        d = dict(d)
        if "model_architecture" in d and isinstance(d["model_architecture"], str):
            d["model_architecture"] = ModelArchitecture(d["model_architecture"])
        if "task_type" in d and isinstance(d["task_type"], str):
            d["task_type"] = TaskType(d["task_type"])
        if "checkpoint_kind" in d and isinstance(d["checkpoint_kind"], str):
            d["checkpoint_kind"] = CheckpointKind(d["checkpoint_kind"])
        # Drop unknown keys to allow forward-compat reads
        valid_keys = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})

    @classmethod
    def from_json(cls, path: str | Path) -> "CaseMeta":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["model_architecture"] = self.model_architecture.value
        d["task_type"] = self.task_type.value
        if self.checkpoint_kind is not None:
            d["checkpoint_kind"] = self.checkpoint_kind.value
        return d

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    # ── interactive builder ─────────────────────────────────────────────

    @classmethod
    def from_questionnaire(
        cls,
        prompt_fn=None,
        existing: "CaseMeta | None" = None,
    ) -> "CaseMeta":
        """Walk the user through the case-meta questionnaire.

        Pass a custom prompt_fn (defaults to input()) for testing. If
        `existing` is provided, current values become the defaults shown.
        """
        ask = prompt_fn or _default_prompt
        base = existing or cls()

        def _choice(label: str, options: list[str], default: str) -> str:
            opts = " | ".join(f"[{i+1}]{o}" for i, o in enumerate(options))
            raw = ask(f"{label} ({opts}) [default {default}]: ").strip()
            if not raw:
                return default
            if raw.isdigit() and 1 <= int(raw) <= len(options):
                return options[int(raw) - 1]
            if raw in options:
                return raw
            print(f"  (didn't recognise '{raw}', using default '{default}')")
            return default

        project_name = ask(f"Project name [default {base.project_name}]: ").strip() or base.project_name

        # ── checkpoint_kind: the critical disambiguation question ──────────
        # This used to default silently and the rule book was applied as if
        # checkpoints were epochs even when they were seeds, which is the
        # root cause of the 2026-05 seed-vs-epoch bug. Ask first, no default.
        print()
        print("FFCA reads a SEQUENCE of model states (the 'checkpoints').")
        print("This drastically changes how stability/drift signals are read:")
        print("  [1] epoch  — multiple snapshots from a SINGLE training run,")
        print("               time-ordered (drift is meaningful, train-longer")
        print("               recommendations apply).")
        print("  [2] seed   — N independently-trained models with different")
        print("               random seeds (a deep ensemble). NO time order;")
        print("               drift between consecutive members is meaningless.")
        print("  [3] mixed  — a hybrid (multiple seeds × multiple epochs).")
        ck_default = base.checkpoint_kind.value if base.checkpoint_kind else None
        ck_str = _choice(
            "What are your checkpoints?",
            [k.value for k in CheckpointKind],
            ck_default or "epoch",
        )
        checkpoint_kind = CheckpointKind(ck_str)

        arch_str = _choice(
            "Model architecture",
            [a.value for a in ModelArchitecture],
            base.model_architecture.value,
        )
        task_str = _choice(
            "Task type",
            [t.value for t in TaskType],
            base.task_type.value,
        )
        target_name = ask(f"Target variable name [default {base.target_name}]: ").strip() or base.target_name
        target_units = ask(f"Target units (e.g. cm, %, USD) [default '{base.target_units}']: ").strip() or base.target_units
        domain = ask(f"Domain (free text) [default '{base.domain}']: ").strip() or base.domain
        pre = ask(f"Pretrained model? (y/n) [default {'y' if base.pretrained else 'n'}]: ").strip().lower()
        pretrained = (pre == "y") if pre else base.pretrained
        if checkpoint_kind in (CheckpointKind.SEED, CheckpointKind.MIXED):
            seeds_prompt = (
                f"How many random seeds (= ensemble size)? [default {base.n_seeds}]: "
            )
        else:
            seeds_prompt = f"How many random seeds? [default {base.n_seeds}]: "
        seeds_raw = ask(seeds_prompt).strip()
        n_seeds = int(seeds_raw) if seeds_raw.isdigit() else base.n_seeds
        conv = ask(
            f"Feature naming convention, e.g. '_t-k = lag, _t+k = forecast' (optional) [default '{base.feature_naming_convention}']: "
        ).strip() or base.feature_naming_convention
        notes = ask(f"Additional notes (optional) [default '{base.notes}']: ").strip() or base.notes

        return cls(
            project_name=project_name,
            model_architecture=ModelArchitecture(arch_str),
            task_type=TaskType(task_str),
            target_name=target_name,
            target_units=target_units,
            domain=domain,
            pretrained=pretrained,
            n_seeds=n_seeds,
            checkpoint_kind=checkpoint_kind,
            feature_naming_convention=conv,
            notes=notes,
        )

    # ── prompt block ────────────────────────────────────────────────────

    def as_prompt_block(self) -> str:
        """Render the case context as a system-prompt-friendly block."""
        lines = [
            "## About this project",
            "",
            f"- Project: {self.project_name or '(unspecified)'}",
            f"- Model architecture: {self.model_architecture.value.upper()}",
            f"- Task: {self.task_type.value.replace('_', ' ')}",
            f"- Target variable: {self.target_name}"
            + (f" ({self.target_units})" if self.target_units else ""),
        ]
        if self.domain:
            lines.append(f"- Domain: {self.domain}")
        if self.pretrained:
            lines.append("- Model uses pre-trained weights (not from scratch).")
        # checkpoint_kind belongs near the top — it changes how across-
        # checkpoint signals (drift, archetype-flip stability, INVESTIGATE
        # rate) should be interpreted. Render explicitly when set.
        if self.checkpoint_kind == CheckpointKind.SEED:
            n_s = self.n_seeds if self.n_seeds and self.n_seeds > 1 else "?"
            lines.append(
                f"- **CHECKPOINT AXIS = SEED.** The {n_s} 'checkpoints' in "
                f"the FFCA report are independently-trained ensemble members "
                f"with different random seeds, NOT time-ordered snapshots of "
                f"a single training run. **Do NOT interpret high INVESTIGATE "
                f"rate or archetype-flip count as 'model not yet converged' "
                f"or recommend 'train longer.'** High cross-seed disagreement "
                f"is the FFCA 'ensemble in disguise' signature: a multi-modal "
                f"loss landscape where different seeds find different feature-"
                f"role assignments of roughly equivalent accuracy. More "
                f"training will NOT resolve it; pruning by INVESTIGATE alone "
                f"is also unsafe. If RMSE is satisfactory, accept the ensemble."
            )
        elif self.checkpoint_kind == CheckpointKind.EPOCH:
            lines.append(
                "- CHECKPOINT AXIS = EPOCH. Checkpoints are time-ordered "
                "snapshots of one training run. Drift, archetype-flip "
                "stability, and 'train longer' recommendations are valid."
            )
        elif self.checkpoint_kind == CheckpointKind.MIXED:
            lines.append(
                "- CHECKPOINT AXIS = MIXED. Multiple seeds × multiple "
                "epochs. Treat dynamic-stability signals with caution; the "
                "INVESTIGATE bucket may conflate within-training drift with "
                "across-seed multi-modality."
            )
        elif self.n_seeds and self.n_seeds > 1:
            # n_seeds > 1 without checkpoint_kind set — likely a seed ensemble
            # but explicitly not declared. Hedge.
            lines.append(
                f"- {self.n_seeds} random seeds were run. Checkpoint axis "
                f"was NOT explicitly declared — verify whether FFCA "
                f"checkpoints are seeds or training epochs before "
                f"interpreting stability signals."
            )
        if self.feature_naming_convention:
            lines.append(
                f"- Feature naming convention: {self.feature_naming_convention}"
            )
        if self.notes:
            lines.append(f"- Notes: {self.notes}")
        lines.append("")
        lines.append(
            "Use this context to interpret feature names, calibrate which "
            "thresholds matter, and pick examples that make sense for the "
            "task type. Do not invent domain facts beyond what is stated here."
        )
        return "\n".join(lines)


# ── helpers ────────────────────────────────────────────────────────────────


def _default_prompt(msg: str) -> str:
    try:
        return input(msg)
    except EOFError:
        return ""


def intent_prompt_block(intent: NarrationIntent) -> str:
    """Per-intent framing slotted into the system prompt."""
    framing = {
        NarrationIntent.AUDIT: (
            "## Your intent: AUDIT\n\n"
            "The user is deciding whether this model is safe to ship. "
            "Lead with critical-severity findings. Be explicit about what is "
            "unknown or unverifiable from the FFCA report alone. If there are "
            "no critical findings, say so plainly and recommend one concrete "
            "verification step the user can run before shipping."
        ),
        NarrationIntent.DIAGNOSE: (
            "## Your intent: DIAGNOSE\n\n"
            "The user wants a root-cause story. Resolve cross-rule tensions "
            "honestly: e.g., 'healthy archetype mix + high INVESTIGATE' is "
            "*unconverged* on the epoch axis but *ensemble-in-disguise* on "
            "the seed axis — the case-context block tells you which. Pick "
            "the right interpretation and recommend the corresponding action "
            "(more training for epoch-axis; accept-the-ensemble for seed-axis). "
            "Never recommend 'train longer' on a seed-axis case."
        ),
        NarrationIntent.PRUNE: (
            "## Your intent: PRUNE\n\n"
            "The user is deciding which features to drop. Lead with the "
            "trust-bucket findings (CONFIDENTLY PRUNE is safe; INVESTIGATE is "
            "NOT safe to prune) and the cosens group analysis. On EPOCH-axis "
            "cases: 'INVESTIGATE = not yet converged' → do not prune until "
            "training stabilises. On SEED-axis cases: 'INVESTIGATE = different "
            "seeds use this feature differently' → pruning by INVESTIGATE is "
            "also unsafe (some seeds rely on it heavily), but the remedy is "
            "different: try targeted sliced retraining or accept the ensemble, "
            "NOT 'train longer.' Always state the axis you're reasoning on."
        ),
        NarrationIntent.COMPARE: (
            "## Your intent: COMPARE\n\n"
            "The user is comparing this report against a previous run "
            "(e.g., before vs after pruning, seed A vs seed B). Frame "
            "findings differentially when possible; note which signals "
            "moved in the right direction and which did not."
        ),
        NarrationIntent.FREE: "",
    }
    return framing.get(intent, "")
