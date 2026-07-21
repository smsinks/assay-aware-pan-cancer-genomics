"""Small, testable statistical contracts shared by the analysis stages."""
from __future__ import annotations

import numpy as np
from statsmodels.stats.multitest import multipletests


def benjamini_hochberg(p_values) -> np.ndarray:
    """Return BH-adjusted values while preserving missing positions.

    The function rejects finite values outside [0, 1] and applies the correction only
    to finite tests. This is the multiple-testing contract used for each stated family.
    """
    values = np.asarray(p_values, dtype=float)
    finite = np.isfinite(values)
    if np.any((values[finite] < 0) | (values[finite] > 1)):
        raise ValueError("Finite P values must lie within [0, 1]")
    adjusted = np.full(values.shape, np.nan, dtype=float)
    if finite.any():
        adjusted[finite] = multipletests(values[finite], method="fdr_bh")[1]
    return adjusted


def genotype_group(mut_a, mut_b) -> np.ndarray:
    """Encode two binary mutation indicators as the four survival genotype groups."""
    a = np.asarray(mut_a)
    b = np.asarray(mut_b)
    if a.shape != b.shape:
        raise ValueError("Mutation indicator arrays must have the same shape")
    if not np.isin(a, [0, 1]).all() or not np.isin(b, [0, 1]).all():
        raise ValueError("Mutation indicators must be binary")
    return np.select(
        [
            (a == 0) & (b == 0),
            (a == 1) & (b == 0),
            (a == 0) & (b == 1),
            (a == 1) & (b == 1),
        ],
        ["A−/B−", "A only", "B only", "A+B"],
        default="unknown",
    )


def replace_zero_follow_up(months, epsilon_months: float) -> tuple[np.ndarray, np.ndarray]:
    """Replace exactly zero follow-up for sensitivity analysis and return its mask."""
    if not np.isfinite(epsilon_months) or epsilon_months <= 0:
        raise ValueError("The zero-time replacement must be finite and positive")
    values = np.asarray(months, dtype=float).copy()
    zero = np.isfinite(values) & (values == 0)
    values[zero] = epsilon_months
    return values, zero
