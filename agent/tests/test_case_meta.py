"""Tests for case_meta: dataclass round-trip, questionnaire, prompt block."""

from __future__ import annotations

import json

import pytest

from ffca_agent.case_meta import (
    CaseMeta,
    ModelArchitecture,
    NarrationIntent,
    TaskType,
    intent_prompt_block,
)


def test_case_meta_default_values_are_sane():
    m = CaseMeta()
    assert m.model_architecture == ModelArchitecture.MLP
    assert m.task_type == TaskType.REGRESSION
    assert m.n_seeds == 1
    assert m.schema_version == "0.6.0"


def test_round_trip_json(tmp_path):
    m1 = CaseMeta(
        project_name="compound_flooding",
        model_architecture=ModelArchitecture.MLP,
        task_type=TaskType.REGRESSION,
        target_name="water_level",
        target_units="cm",
        domain="coastal hydrology",
        pretrained=False,
        n_seeds=3,
        feature_naming_convention="_t-k = lag k, _t+k = forecast k",
    )
    p = tmp_path / "meta.json"
    m1.save(p)
    m2 = CaseMeta.from_json(p)
    assert m2.project_name == m1.project_name
    assert m2.model_architecture == m1.model_architecture
    assert m2.task_type == m1.task_type
    assert m2.target_units == m1.target_units
    assert m2.feature_naming_convention == m1.feature_naming_convention
    assert m2.n_seeds == 3


def test_from_dict_drops_unknown_keys():
    # forward-compat: an extra key in the JSON shouldn't break the loader
    d = {"project_name": "X", "future_field": "ignored"}
    m = CaseMeta.from_dict(d)
    assert m.project_name == "X"


def test_questionnaire_uses_defaults_when_user_hits_enter():
    answers = iter([""] * 20)  # every prompt → "" → use defaults
    m = CaseMeta.from_questionnaire(prompt_fn=lambda _: next(answers))
    assert m.project_name == "untitled"
    assert m.model_architecture == ModelArchitecture.MLP
    assert m.task_type == TaskType.REGRESSION


def test_questionnaire_accepts_numbered_choice_for_enum():
    """Selecting '2' on the model-arch question picks the 2nd option."""
    arch_options = [a.value for a in ModelArchitecture]
    task_options = [t.value for t in TaskType]
    expected_arch = ModelArchitecture(arch_options[1])  # cnn
    expected_task = TaskType(task_options[2])           # multiclass

    answers = iter([
        "MyProject",   # project_name
        "2",           # model_architecture → cnn
        "3",           # task_type → multiclass_classification
        "label",       # target_name
        "",            # target_units (empty)
        "credit risk", # domain
        "y",           # pretrained
        "5",           # n_seeds
        "",            # naming convention (empty)
        "test run",    # notes
    ])
    m = CaseMeta.from_questionnaire(prompt_fn=lambda _: next(answers))
    assert m.project_name == "MyProject"
    assert m.model_architecture == expected_arch
    assert m.task_type == expected_task
    assert m.target_name == "label"
    assert m.target_units == ""
    assert m.domain == "credit risk"
    assert m.pretrained is True
    assert m.n_seeds == 5
    assert m.notes == "test run"


def test_questionnaire_existing_provides_defaults():
    existing = CaseMeta(project_name="prior", domain="aviation",
                        model_architecture=ModelArchitecture.TRANSFORMER)
    answers = iter([""] * 20)
    m = CaseMeta.from_questionnaire(
        prompt_fn=lambda _: next(answers),
        existing=existing,
    )
    assert m.project_name == "prior"
    assert m.domain == "aviation"
    assert m.model_architecture == ModelArchitecture.TRANSFORMER


def test_prompt_block_contains_required_fields():
    m = CaseMeta(
        project_name="flooding",
        target_name="wl",
        target_units="cm",
        domain="hydrology",
        pretrained=True,
        feature_naming_convention="_t-k=lag",
    )
    block = m.as_prompt_block()
    assert "flooding" in block
    assert "MLP" in block
    assert "regression" in block.lower()
    assert "wl (cm)" in block
    assert "hydrology" in block
    assert "pre-trained" in block.lower()
    assert "_t-k=lag" in block
    assert "Do not invent" in block


def test_prompt_block_omits_empty_optional_fields():
    m = CaseMeta(project_name="p", target_name="t")
    block = m.as_prompt_block()
    assert "Domain:" not in block       # domain="" → suppressed
    assert "pre-trained" not in block.lower()  # pretrained=False → suppressed
    assert "Notes:" not in block        # notes="" → suppressed


def test_intent_prompt_blocks_exist_for_each_real_intent():
    # Each intent (except FREE) yields a non-empty block
    for it in NarrationIntent:
        block = intent_prompt_block(it)
        if it == NarrationIntent.FREE:
            assert block == ""
        else:
            assert len(block) > 50
            assert "intent" in block.lower()


def test_intent_audit_prompts_for_ship_decision():
    block = intent_prompt_block(NarrationIntent.AUDIT)
    assert "ship" in block.lower() or "audit" in block.lower()


def test_intent_prune_warns_against_unstable_pruning():
    block = intent_prompt_block(NarrationIntent.PRUNE)
    assert "trust_instability" in block or "INVESTIGATE" in block
