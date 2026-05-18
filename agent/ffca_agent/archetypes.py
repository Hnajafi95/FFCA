"""Archetype name normalization.

The FFCA Python package uses short internal names (Workhorse, Catalyst, …).
The paper (Najafi, Luo, Liu 2025, Table 5) uses fuller names (Simple Workhorse,
Interactive Catalyst, …). The rulebook follows the paper. This module bridges
them — the report adapter normalizes incoming names to paper form.
"""

from __future__ import annotations

PACKAGE_INDEX_TO_PAPER = {
    0: "Noise Candidate",
    1: "Hidden Interactor",
    2: "Simple Workhorse",
    3: "Interactive Catalyst",
    4: "Non-linear Driver",
    5: "Volatile Specialist",
    6: "Stable Contributor",
    7: "Complex Driver",
}

PACKAGE_NAME_TO_PAPER = {
    "Noise": "Noise Candidate",
    "Hidden Interactor": "Hidden Interactor",
    "Workhorse": "Simple Workhorse",
    "Catalyst": "Interactive Catalyst",
    "Nonlinear Driver": "Non-linear Driver",
    "Volatile Specialist": "Volatile Specialist",
    "Stable Contributor": "Stable Contributor",
    "Complex Driver": "Complex Driver",
}

PAPER_TO_SNAKE = {
    "Noise Candidate": "noise",
    "Hidden Interactor": "hidden_interactor",
    "Simple Workhorse": "simple_workhorse",
    "Interactive Catalyst": "interactive_catalyst",
    "Non-linear Driver": "nonlinear_driver",
    "Volatile Specialist": "volatile_specialist",
    "Stable Contributor": "stable_contributor",
    "Complex Driver": "complex_driver",
}

PAPER_NAMES = tuple(PAPER_TO_SNAKE.keys())


def to_paper(name_or_index) -> str:
    """Normalize any archetype reference to its paper name."""
    if isinstance(name_or_index, int):
        return PACKAGE_INDEX_TO_PAPER[name_or_index]
    if name_or_index in PACKAGE_NAME_TO_PAPER:
        return PACKAGE_NAME_TO_PAPER[name_or_index]
    if name_or_index in PAPER_TO_SNAKE:
        return name_or_index
    raise ValueError(f"Unknown archetype: {name_or_index!r}")
