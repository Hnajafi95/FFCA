"""Tests for the LLM narration layer.

These never hit the live API. The Anthropic client is mocked to return a fixed
structured response; we verify (a) the system prompt is sent with cache_control,
(b) the user message embeds the findings JSON and a context summary, and
(c) the response parser turns the fenced JSON into a NarratedReport correctly.
A separate test asserts that without ANTHROPIC_API_KEY the constructor raises
NarratorError cleanly (so the CLI can fall back).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from ffca_agent.evaluator import Finding, evaluate_rulebook, load_rulebook
from ffca_agent.llm import (
    DEFAULT_MODEL,
    NarratedReport,
    Narrator,
    NarratorError,
    _parse_structured_response,
)
from ffca_agent.report import ReportContext

REPO = Path(__file__).resolve().parents[1]
RULEBOOK_PATH = REPO / "rulebook" / "ffca_rules.yaml"


# ── lightweight fake Anthropic client ──────────────────────────────────────


@dataclass
class _Block:
    text: str
    type: str = "text"


@dataclass
class _Usage:
    input_tokens: int = 1234
    output_tokens: int = 567
    cache_creation_input_tokens: int = 1000
    cache_read_input_tokens: int = 234


@dataclass
class _Response:
    content: list[_Block]
    usage: _Usage


class _FakeMessages:
    def __init__(self, response_text: str, captured: dict):
        self._text = response_text
        self._captured = captured

    def create(self, **kwargs):
        self._captured.update(kwargs)
        return _Response(content=[_Block(text=self._text)], usage=_Usage())


class _FakeClient:
    def __init__(self, response_text: str):
        self.captured: dict = {}
        self.messages = _FakeMessages(response_text, self.captured)


# ── helpers ────────────────────────────────────────────────────────────────


def _synthetic_ctx(tmp_path: Path) -> ReportContext:
    raw = {
        "n_features": 4,
        "feature_names": ["a", "b", "c", "d"],
        "signatures": [{
            "impact": [0.5, 0.3, 0.1, 0.01],
            "volatility": [0.001] * 4,
            "nonlinearity": [0.01] * 4,
            "interaction": [0.05] * 4,
            "archetypes": [2, 3, 6, 0],
        }] * 3,
        "trust": {f: {"decision": "CONFIDENTLY KEEP"} for f in ["a", "b", "c", "d"]},
        "trust_summary": {},
        "co_sensitivity": None,
        "findings": [],
    }
    p = tmp_path / "r.json"
    p.write_text(json.dumps(raw))
    return ReportContext.from_json(p)


def _fake_response_text() -> str:
    payload = {
        "executive_summary": "Model is healthy overall; one warning to investigate.",
        "actions": [
            {"priority": 1, "title": "Investigate trust",
             "rationale": "Trust instability detected.",
             "rule_ids": ["trust_instability_high"]},
            {"priority": 2, "title": "Confirm archetype mix",
             "rationale": "Distribution is balanced.",
             "rule_ids": ["healthy_archetype_distribution"]},
        ],
        "caveats": [
            "Dynamic rules skipped — no training history provided.",
        ],
    }
    return (
        "Some optional prose lead-in.\n"
        "<DIAGNOSIS>\n"
        + json.dumps(payload, indent=2)
        + "\n</DIAGNOSIS>\n"
        "Trailing prose, harmless."
    )


# ── tests ──────────────────────────────────────────────────────────────────


def test_parse_structured_response_strips_fence():
    text = _fake_response_text()
    parsed = _parse_structured_response(text)
    assert parsed["executive_summary"].startswith("Model is healthy")
    assert len(parsed["actions"]) == 2
    assert parsed["actions"][0]["rule_ids"] == ["trust_instability_high"]


def test_parse_response_raises_on_missing_fence():
    with pytest.raises(NarratorError):
        _parse_structured_response("no fence here")


def test_parse_response_raises_on_invalid_json():
    with pytest.raises(NarratorError):
        _parse_structured_response("<DIAGNOSIS>not json</DIAGNOSIS>")


def test_narrator_constructor_requires_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(NarratorError, match="ANTHROPIC_API_KEY"):
        Narrator()


def test_narrate_returns_structured_report_with_cached_system(tmp_path):
    ctx = _synthetic_ctx(tmp_path)
    rulebook = load_rulebook(RULEBOOK_PATH)
    findings = evaluate_rulebook(rulebook, ctx)

    client = _FakeClient(_fake_response_text())
    narrator = Narrator(model="claude-opus-4-7", client=client)
    report = narrator.narrate(findings, ctx)

    assert isinstance(report, NarratedReport)
    assert "Model is healthy" in report.executive_summary
    assert len(report.actions) == 2
    assert report.actions[0].rule_ids == ["trust_instability_high"]
    assert report.caveats and "no training history" in report.caveats[0]
    assert report.model == "claude-opus-4-7"
    assert report.usage["input_tokens"] == 1234
    assert report.usage["cache_read_input_tokens"] == 234

    # confirm what we sent the API
    captured = client.captured
    assert captured["model"] == "claude-opus-4-7"
    assert isinstance(captured["system"], list)
    assert captured["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "FFCA Diagnostic Agent" in captured["system"][0]["text"]

    user_text = captured["messages"][0]["content"]
    assert "n_features: 4" in user_text
    # findings JSON must be embedded
    assert "trust_instability_high" not in user_text or "rule_id" in user_text
    # rule ids actually present in the findings should appear in the user prompt
    for f in findings:
        assert f.rule_id in user_text


def test_narrator_uses_default_model_when_unspecified(tmp_path):
    ctx = _synthetic_ctx(tmp_path)
    rulebook = load_rulebook(RULEBOOK_PATH)
    findings = evaluate_rulebook(rulebook, ctx)
    client = _FakeClient(_fake_response_text())
    narrator = Narrator(client=client)
    assert narrator.model == DEFAULT_MODEL
    report = narrator.narrate(findings, ctx)
    assert client.captured["model"] == DEFAULT_MODEL
    assert report.model == DEFAULT_MODEL


# ── CLI fallback when API key is absent ────────────────────────────────────


def test_cli_falls_back_to_deterministic_without_api_key(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ctx = _synthetic_ctx(tmp_path)
    # write a real report.json on disk so the CLI can read it
    report_path = tmp_path / "r.json"

    from ffca_agent.cli import main
    rc = main([str(report_path), "--narrate"])
    captured = capsys.readouterr()
    assert rc == 0
    # warning should appear on stderr
    assert "narration unavailable" in captured.err
    # but deterministic markdown should still print to stdout
    assert "FFCA Agent Diagnosis" in captured.out


# ── v0.6: templated system prompt + intent + rule-free observations ───────


def _fake_response_with_observations() -> str:
    payload = {
        "executive_summary": "Model is healthy overall.",
        "actions": [
            {"priority": 1, "title": "Trust check",
             "rationale": "All good.", "rule_ids": ["healthy_archetype_distribution"]}
        ],
        "rule_free_observations": [
            {"what": "Top-1 feature dominates by 8x",
             "evidence": "top_by_impact[0].value=4.8 vs top_by_impact[1].value=0.6"},
        ],
        "caveats": ["No training history provided."],
    }
    return (
        "<DIAGNOSIS>\n"
        + json.dumps(payload, indent=2)
        + "\n</DIAGNOSIS>"
    )


def test_build_system_prompt_default_matches_v05_behaviour():
    from ffca_agent.llm import _build_system_prompt
    prompt = _build_system_prompt(case_meta=None, intent=None)
    assert "FFCA Diagnostic Agent" in prompt
    assert "About this project" not in prompt
    assert "Your intent" not in prompt


def test_build_system_prompt_splices_case_meta():
    from ffca_agent.case_meta import CaseMeta, ModelArchitecture, TaskType
    from ffca_agent.llm import _build_system_prompt

    meta = CaseMeta(project_name="flooding", domain="hydrology",
                    target_name="wl", target_units="cm",
                    task_type=TaskType.REGRESSION,
                    model_architecture=ModelArchitecture.MLP)
    prompt = _build_system_prompt(case_meta=meta, intent=None)
    assert "About this project" in prompt
    assert "flooding" in prompt
    assert "hydrology" in prompt


def test_build_system_prompt_splices_intent():
    from ffca_agent.case_meta import NarrationIntent
    from ffca_agent.llm import _build_system_prompt

    prompt = _build_system_prompt(case_meta=None, intent=NarrationIntent.AUDIT)
    assert "Your intent" in prompt
    assert "AUDIT" in prompt


def test_build_system_prompt_intent_free_adds_no_extra_block():
    from ffca_agent.case_meta import NarrationIntent
    from ffca_agent.llm import _build_system_prompt

    free = _build_system_prompt(case_meta=None, intent=NarrationIntent.FREE)
    default = _build_system_prompt(case_meta=None, intent=None)
    assert free.strip() == default.strip()


def test_narrate_with_case_meta_and_intent_uses_templated_prompt(tmp_path):
    from ffca_agent.case_meta import CaseMeta, NarrationIntent, TaskType, ModelArchitecture

    ctx = _synthetic_ctx(tmp_path)
    rulebook = load_rulebook(RULEBOOK_PATH)
    findings = evaluate_rulebook(rulebook, ctx)
    client = _FakeClient(_fake_response_text())
    narrator = Narrator(client=client)
    meta = CaseMeta(project_name="testproj", target_name="y",
                    domain="example",
                    task_type=TaskType.BINARY_CLASSIFICATION,
                    model_architecture=ModelArchitecture.CNN)
    narrator.narrate(findings, ctx, case_meta=meta, intent=NarrationIntent.AUDIT)

    sys_text = client.captured["system"][0]["text"]
    # case-meta block must appear
    assert "testproj" in sys_text
    assert "CNN" in sys_text
    # intent framing must appear
    assert "AUDIT" in sys_text
    # findings JSON should still be in the user prompt
    user_text = client.captured["messages"][0]["content"]
    for f in findings:
        assert f.rule_id in user_text


def test_narrate_parses_rule_free_observations(tmp_path):
    ctx = _synthetic_ctx(tmp_path)
    rulebook = load_rulebook(RULEBOOK_PATH)
    findings = evaluate_rulebook(rulebook, ctx)
    client = _FakeClient(_fake_response_with_observations())
    narrator = Narrator(client=client)
    report = narrator.narrate(findings, ctx)

    assert len(report.rule_free_observations) == 1
    obs = report.rule_free_observations[0]
    assert "dominates by 8x" in obs.what
    assert "top_by_impact" in obs.evidence


def test_narrate_handles_missing_observations_field(tmp_path):
    """v0.5 responses (no rule_free_observations key) must still parse."""
    ctx = _synthetic_ctx(tmp_path)
    rulebook = load_rulebook(RULEBOOK_PATH)
    findings = evaluate_rulebook(rulebook, ctx)
    client = _FakeClient(_fake_response_text())  # no observations field
    narrator = Narrator(client=client)
    report = narrator.narrate(findings, ctx)
    assert report.rule_free_observations == []


def test_narrate_with_signature_summary_in_user_prompt(tmp_path):
    from ffca_agent.signature_summary import signature_summary

    ctx = _synthetic_ctx(tmp_path)
    rulebook = load_rulebook(RULEBOOK_PATH)
    findings = evaluate_rulebook(rulebook, ctx)
    client = _FakeClient(_fake_response_text())
    narrator = Narrator(client=client)
    summary = signature_summary(ctx)
    narrator.narrate(findings, ctx, sig_summary=summary)

    user_text = client.captured["messages"][0]["content"]
    assert "Signature summary" in user_text
    assert "top_by_impact" in user_text
