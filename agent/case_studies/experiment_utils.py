"""Utility assertions for experimental selection code.

The helpers in this module guard against the two recurring case-study bugs
we hit on the compound-flooding extra-validation runs (see
agent/docs/PITFALLS.md for the full incident log):

  1. Subset-selection filters whose tolerance is so loose they constrain
     nothing in practice (e.g. the §F "matched Impact" filter that
     produced subsets with 36× mean-Impact ratio).
  2. Control-pair selection that silently returns an empty list because
     the matching constraint conflicts with the exclusion constraint
     (e.g. the original §H control selector).

Use both helpers in any experimental selection code that names a property
it intends to match or balance. They are cheap, deterministic, and fail
loudly the moment the selection breaks its own contract.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence


class SelectionContractError(AssertionError):
    """Raised when an experimental selection violates the property it claims
    to match (e.g. "matched Impact" but the resulting subsets differ by 36×)."""


def assert_subset_distributions_match(
    subset_a_values: Sequence[float],
    subset_b_values: Sequence[float],
    signal_name: str,
    max_mean_ratio: float = 2.0,
    max_median_ratio: float | None = None,
) -> None:
    """Raise SelectionContractError if the two subsets differ by more than
    `max_mean_ratio`× in mean (or `max_median_ratio`× in median, if given)
    on the named signal.

    Call this right after selecting any two subsets that the rest of the
    code (or the paper text) describes as "matched on X". If the matching
    contract is broken, you want to know now, not after a 5-hour HPC run
    on confounded subsets.

    Example:
        high_n_features = pick_top_k_by('nonlinearity', k=20)
        low_n_features = pick_matched_impact_low_nonlinearity(k=20)
        assert_subset_distributions_match(
            [f.impact for f in high_n_features],
            [f.impact for f in low_n_features],
            signal_name='impact',
            max_mean_ratio=2.0,
        )
    """
    if not subset_a_values or not subset_b_values:
        raise SelectionContractError(
            f'subset distribution match check on {signal_name!r}: '
            f'one or both subsets are empty '
            f'(n_a={len(subset_a_values)}, n_b={len(subset_b_values)})'
        )

    def _mean(xs: Sequence[float]) -> float:
        return sum(xs) / len(xs)

    def _median(xs: Sequence[float]) -> float:
        s = sorted(xs)
        n = len(s)
        return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])

    mean_a, mean_b = _mean(subset_a_values), _mean(subset_b_values)
    if mean_a == 0 or mean_b == 0:
        # Symmetric: if either side has zero mean and the other doesn't, fail.
        if mean_a != mean_b:
            raise SelectionContractError(
                f'mean({signal_name}) is zero on exactly one subset: '
                f'mean_a={mean_a}, mean_b={mean_b}'
            )
    else:
        ratio = max(mean_a, mean_b) / min(mean_a, mean_b)
        if ratio > max_mean_ratio:
            raise SelectionContractError(
                f'subsets are NOT matched on {signal_name!r}: '
                f'mean_a={mean_a:.4g}, mean_b={mean_b:.4g} '
                f'(ratio {ratio:.2f}× exceeds the {max_mean_ratio:.2f}× limit). '
                f'The selection filter is too loose — tighten the matching '
                f'constraint before relying on this contrast.'
            )

    if max_median_ratio is not None:
        med_a, med_b = _median(subset_a_values), _median(subset_b_values)
        if med_a == 0 or med_b == 0:
            if med_a != med_b:
                raise SelectionContractError(
                    f'median({signal_name}) is zero on exactly one subset.'
                )
        else:
            ratio_med = max(med_a, med_b) / min(med_a, med_b)
            if ratio_med > max_median_ratio:
                raise SelectionContractError(
                    f'subsets are NOT median-matched on {signal_name!r}: '
                    f'median_a={med_a:.4g}, median_b={med_b:.4g} '
                    f'(ratio {ratio_med:.2f}× exceeds the {max_median_ratio:.2f}× limit).'
                )


def assert_nonempty(
    items: Iterable,
    name: str,
    context: str = '',
) -> None:
    """Raise SelectionContractError if `items` is empty.

    Use this on control sets, control-pair lists, candidate pools — any
    list whose emptiness would invalidate the experimental conclusion.
    The §H control selector returned [] silently and the experiment ran to
    "completion" with no nulls; this assertion would have caught it at
    pair-selection time.
    """
    n = len(list(items)) if not hasattr(items, '__len__') else len(items)  # type: ignore[arg-type]
    if n == 0:
        msg = f'{name!r} is empty'
        if context:
            msg += f' — {context}'
        msg += (
            '. The experiment cannot reach a defensible conclusion without '
            'these. Re-examine the selection logic before running '
            'downstream training.'
        )
        raise SelectionContractError(msg)
