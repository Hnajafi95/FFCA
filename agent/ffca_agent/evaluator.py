"""Rule evaluator: applies the rulebook YAML to a ReportContext, emits Findings.

Decision tree per rule:
  - kind=descriptor + scope=per_feature → evaluated for every feature; one
    finding per matching feature
  - kind=descriptor + scope=model_wide → evaluated once; one finding
  - kind=descriptor + scope=dynamic → evaluated once against curves
  - kind=diagnostic → same scope rules; finding carries severity + recommendation

Triggers are combined per the rule's trigger_logic (all/any). Signals that
raise MissingSignal cause the trigger to count as "unknown" — by default an
unknown trigger fires no rule (conservative). Use --strict to fail loudly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .report import MissingSignal, ReportContext
from .timeseries import collapse_detected, plateau_detected, spike_detected


@dataclass
class Finding:
    rule_id: str
    rule_name: str
    kind: str
    category: str
    scope: str
    severity: str | None
    feature: str | None
    diagnosis: str
    recommendation: str
    evidence: str
    paper_ref: str
    confidence_factors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if v not in (None, [], "")}
        return d


# ── trigger evaluation ──────────────────────────────────────────────────────

_COMPARISON_OPS = {">", ">=", "<", "<=", "==", "!=", "in", "not_in", "approx_eq"}
_TIMESERIES_OPS = {"spike_detected", "plateau_detected", "collapse_detected"}


def _eval_comparison(actual, op: str, expected) -> bool:
    if op == ">":  return actual > expected
    if op == ">=": return actual >= expected
    if op == "<":  return actual < expected
    if op == "<=": return actual <= expected
    if op == "==": return actual == expected
    if op == "!=": return actual != expected
    if op == "in":  return actual in expected
    if op == "not_in": return actual not in expected
    if op == "approx_eq":
        # treat expected as the target value; tolerance = max(0.01 * |expected|, 0.01)
        try:
            tol = max(abs(float(expected)) * 0.01, 0.01)
        except (TypeError, ValueError):
            return actual == expected
        return abs(float(actual) - float(expected)) <= tol
    raise ValueError(f"unsupported comparison op: {op}")


def _eval_timeseries(curve, op: str, value, feature_idx: int | None = None):
    """Returns (fired, info_dict). The curve may be a 1D series or a 2D
    (n_checkpoints × n_features) matrix; in the latter case `feature_idx`
    extracts the relevant column.
    """
    arr = np.asarray(curve)
    if arr.ndim == 2:
        if feature_idx is None:
            return False, {"reason": "matrix curve requires feature_idx"}
        arr = arr[:, feature_idx]

    if op == "spike_detected":
        # Three accepted forms for value:
        #   "epoch_0" / "early_epochs" / "late_epochs" / "any_epoch"  → use default threshold_ratio
        #   numeric (1.3, 2.0, ...)                                   → use as threshold_ratio, when=any_epoch
        #   {when: "epoch_0", threshold_ratio: 5.0}                   → both
        if isinstance(value, dict):
            kwargs = {}
            if "when" in value:
                kwargs["when"] = value["when"]
            if "threshold_ratio" in value:
                kwargs["threshold_ratio"] = float(value["threshold_ratio"])
            if "baseline_fraction" in value:
                kwargs["baseline_fraction"] = float(value["baseline_fraction"])
            if "window_fraction" in value:
                kwargs["window_fraction"] = float(value["window_fraction"])
            result = spike_detected(arr, **kwargs)
        elif isinstance(value, str):
            result = spike_detected(arr, when=value)
        else:
            result = spike_detected(arr, threshold_ratio=float(value))
    elif op == "plateau_detected":
        if isinstance(value, dict):
            kwargs = {}
            if "relative_improvement" in value:
                kwargs["relative_improvement"] = float(value["relative_improvement"])
            if "tail_fraction" in value:
                kwargs["tail_fraction"] = float(value["tail_fraction"])
            result = plateau_detected(arr, **kwargs)
        else:
            result = plateau_detected(arr)
    elif op == "collapse_detected":
        if isinstance(value, dict):
            result = collapse_detected(
                arr,
                final_threshold=float(value["final_threshold"]),
                relative_to=value.get("relative_to", "absolute"),
            )
        else:
            result = collapse_detected(arr, final_threshold=float(value))
    else:
        raise ValueError(f"unsupported time-series op: {op}")

    return result.fired, {
        "epoch": result.epoch,
        "ratio": result.ratio,
        "note": result.note,
    }


def _eval_trigger(trigger: dict, ctx: ReportContext, feature_idx: int | None) -> tuple[bool, dict, str | None]:
    """Returns (fired, extras_for_template, skip_reason).

    skip_reason != None means a signal was missing — caller decides whether to
    treat this as "did not fire" (default) or to abort the rule.
    """
    signal_path = trigger["signal"]
    op = trigger["op"]
    expected = trigger["value"]

    try:
        actual = ctx.get(signal_path, feature_idx=feature_idx)
    except MissingSignal as exc:
        return False, {}, str(exc)

    if op in _TIMESERIES_OPS:
        fired, info = _eval_timeseries(actual, op, expected, feature_idx)
        return fired, info, None

    if op in _COMPARISON_OPS:
        # expected might be a percentage encoded as a number; comparisons
        # are direct.
        return _eval_comparison(actual, op, expected), {"actual": actual}, None

    raise ValueError(f"unknown op: {op}")


# ── rule evaluation ─────────────────────────────────────────────────────────

def _safe_format(template: str, ctx: ReportContext, feature_idx: int | None, extras: dict) -> str:
    """Render an evidence/diagnosis template, leaving unresolved keys verbatim."""
    if not template:
        return ""

    class _SafeDict(dict):
        def __missing__(self, key):
            return "{" + key + "}"

    values = _SafeDict()
    # populate from extras first
    for k, v in extras.items():
        values[k] = v
    # populate ctx.get for things like {feature.impact:.4f}
    # we can't enumerate every signal — instead we let str.format_map handle it
    # by looking up dotted keys via a custom resolver.
    rendered = _format_with_resolver(template, ctx, feature_idx, values)
    return rendered


def _format_with_resolver(template: str, ctx: ReportContext, feature_idx: int | None, extras: dict) -> str:
    """A tiny formatter that handles {a.b.c:fmt} by calling ctx.get('a.b.c')."""
    import re

    pattern = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_.]*)(?::([^}]*))?\}")

    def replace(m: re.Match) -> str:
        key, spec = m.group(1), m.group(2) or ""
        if key in extras:
            value = extras[key]
        else:
            try:
                value = ctx.get(key, feature_idx=feature_idx)
            except MissingSignal:
                return m.group(0)
        try:
            return format(value, spec) if spec else str(value)
        except (TypeError, ValueError):
            return str(value)

    return pattern.sub(replace, template)


_APPLIES_WHEN_GRAMMAR = (
    "Supported forms:\n"
    "  case.<attr> == '<value>'\n"
    "  case.<attr> != '<value>'\n"
    "  case.<attr> in ('a', 'b', ...)\n"
    "  case.<attr> not_in ('a', 'b', ...)\n"
    "Use single quotes around literal values."
)


def _check_applies_when(applies_when: str, ctx: ReportContext) -> bool:
    """Expression check used to gate rules on CaseMeta attributes.

    Returning False causes the evaluator to skip the rule entirely. If
    case_meta is None but an applies_when clause is present, returns False
    — the rule was explicitly gated on case context that wasn't provided.

    Raises ValueError on un-parseable expressions. Previously these
    silently returned True (rule fired unconditionally), which let YAML
    typos like `case.X = 'epoch'` (single equals) sneak through and
    mis-gate the seed-vs-epoch rules.
    """
    import re

    cm = getattr(ctx, "case_meta", None)
    expr = applies_when.strip()

    # case.X == 'v' / case.X != 'v'
    m = re.match(
        r"^case\.(\w+)\s*(==|!=)\s*['\"]([^'\"]+)['\"]$", expr
    )
    if m:
        attr, op, value = m.group(1), m.group(2), m.group(3)
        if cm is None:
            return False
        cm_val = _case_attr_string(cm, attr)
        return (cm_val == value) if op == "==" else (cm_val != value)

    # case.X in ('a', 'b') / case.X not_in ('a', 'b')
    m = re.match(
        r"^case\.(\w+)\s+(in|not_in)\s*\(\s*(.+?)\s*\)$", expr
    )
    if m:
        attr, op, list_body = m.group(1), m.group(2), m.group(3)
        items = [
            s.strip().strip("'\"")
            for s in list_body.split(",")
            if s.strip()
        ]
        if cm is None:
            return False
        cm_val = _case_attr_string(cm, attr)
        in_set = cm_val in items
        return in_set if op == "in" else not in_set

    raise ValueError(
        f"applies_when expression {applies_when!r} is not parseable.\n"
        f"{_APPLIES_WHEN_GRAMMAR}"
    )


def _case_attr_string(cm, attr: str) -> str | None:
    val = getattr(cm, attr, None)
    if val is None:
        return None
    return str(getattr(val, "value", val))


def _eval_rule_at(rule: dict, ctx: ReportContext, feature_idx: int | None) -> Finding | None:
    # v0.7: respect applies_when gating on case context (e.g. checkpoint_kind)
    aw = rule.get("applies_when")
    if aw and not _check_applies_when(aw, ctx):
        return None

    triggers = rule["triggers"]
    logic = rule.get("trigger_logic", "all")

    results: list[bool] = []
    extras: dict = {}
    for trig in triggers:
        fired, info, skip = _eval_trigger(trig, ctx, feature_idx)
        if skip:
            return None  # missing signal — conservative skip
        results.append(fired)
        extras.update(info)

    fires = all(results) if logic == "all" else any(results)
    if not fires:
        return None

    return Finding(
        rule_id=rule["id"],
        rule_name=rule["name"],
        kind=rule["kind"],
        category=rule["category"],
        scope=rule["scope"],
        severity=rule.get("severity"),
        feature=ctx.feature_names[feature_idx] if feature_idx is not None else None,
        diagnosis=_safe_format(rule["diagnosis"], ctx, feature_idx, extras),
        recommendation=_safe_format(rule.get("recommendation", ""), ctx, feature_idx, extras),
        evidence=_safe_format(rule.get("evidence_template", ""), ctx, feature_idx, extras),
        paper_ref=rule.get("paper_ref", ""),
        confidence_factors=list(rule.get("confidence_factors", [])),
    )


def evaluate_rulebook(rulebook: dict, ctx: ReportContext) -> list[Finding]:
    findings: list[Finding] = []
    for rule in rulebook["rules"]:
        scope = rule["scope"]
        if scope == "per_feature":
            for idx in ctx.feature_indices():
                f = _eval_rule_at(rule, ctx, idx)
                if f:
                    findings.append(f)
        else:
            f = _eval_rule_at(rule, ctx, None)
            if f:
                findings.append(f)
    return findings


def load_rulebook(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


# ── summary helpers ─────────────────────────────────────────────────────────

def summarize(findings: list[Finding]) -> dict:
    diag = [f for f in findings if f.kind == "diagnostic"]
    desc = [f for f in findings if f.kind == "descriptor"]
    by_sev: dict[str, int] = {}
    for f in diag:
        by_sev[f.severity or "—"] = by_sev.get(f.severity or "—", 0) + 1
    return {
        "n_findings": len(findings),
        "n_diagnostic": len(diag),
        "n_descriptor": len(desc),
        "diagnostic_by_severity": by_sev,
        "rules_fired": sorted({f.rule_id for f in findings}),
    }
