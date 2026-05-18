"""Smoke tests for the FFCA rulebook v0."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

REPO = Path(__file__).resolve().parents[1]
RULEBOOK = REPO / "rulebook" / "ffca_rules.yaml"
SCHEMA = REPO / "rulebook" / "schema.json"

EXPECTED_ARCHETYPES = {
    "Simple Workhorse", "Stable Contributor", "Noise Candidate",
    "Non-linear Driver", "Volatile Specialist", "Hidden Interactor",
    "Interactive Catalyst", "Complex Driver",
}


@pytest.fixture(scope="module")
def rulebook() -> dict:
    return yaml.safe_load(RULEBOOK.read_text())


@pytest.fixture(scope="module")
def schema() -> dict:
    return json.loads(SCHEMA.read_text())


def test_rulebook_parses(rulebook):
    assert "version" in rulebook
    assert "rules" in rulebook
    assert len(rulebook["rules"]) >= 20


def test_rulebook_matches_schema(rulebook, schema):
    validator = Draft202012Validator(schema)
    errors = list(validator.iter_errors(rulebook))
    assert not errors, "\n".join(
        f"at {'.'.join(str(p) for p in e.path)}: {e.message}" for e in errors
    )


def test_rule_ids_unique(rulebook):
    ids = [r["id"] for r in rulebook["rules"]]
    dupes = [i for i, c in Counter(ids).items() if c > 1]
    assert not dupes, f"duplicate ids: {dupes}"


def test_every_archetype_has_descriptor_rule(rulebook):
    archetype_rules = {
        r.get("archetype_hint")
        for r in rulebook["rules"]
        if r["kind"] == "descriptor" and r["category"] == "feature_role"
    }
    missing = EXPECTED_ARCHETYPES - archetype_rules
    assert not missing, f"archetypes without a descriptor rule: {missing}"


def test_diagnostic_rules_have_severity_and_recommendation(rulebook):
    bad = [
        r["id"] for r in rulebook["rules"]
        if r["kind"] == "diagnostic"
        and ("severity" not in r or "recommendation" not in r)
    ]
    assert not bad, f"diagnostic rules missing severity/recommendation: {bad}"


def test_every_rule_has_paper_ref(rulebook):
    missing = [r["id"] for r in rulebook["rules"] if not r.get("paper_ref")]
    assert not missing, f"rules missing paper_ref: {missing}"


def test_coverage_of_paper_pathologies(rulebook):
    """The three big training-pathology categories from App C.6 must each have a rule."""
    ids = {r["id"] for r in rulebook["rules"]}
    must_have = {
        "overfitting_volatility_spike",
        "spurious_correlation_train_val_gap",  # renamed in v0.3
        "data_leakage_immediate_dominance",
    }
    missing = must_have - ids
    assert not missing, f"missing App C.6 pathology rules: {missing}"


def test_coverage_of_capacity_and_dynamics(rulebook):
    ids = {r["id"] for r in rulebook["rules"]}
    must_have = {
        "insufficient_capacity",
        "hierarchical_learning_confirmed",
        "healthy_archetype_distribution",
    }
    missing = must_have - ids
    assert not missing, f"missing Table 6 capacity/dynamics rules: {missing}"


def test_vision_rule_present(rulebook):
    """Waterbirds shortcut-learning rule must exist (App C.4)."""
    ids = {r["id"] for r in rulebook["rules"]}
    assert "shortcut_learning_drift_epoch" in ids


def test_prescriptive_rule_present(rulebook):
    """§5.3 cross-model feature-engineering recipe must exist."""
    ids = {r["id"] for r in rulebook["rules"]}
    assert "linear_baseline_will_fail" in ids
