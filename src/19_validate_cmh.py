"""Validate the three Mantel--Haenszel specifications produced by Stage 17.

Stage 17 independently reconstructs every prespecified reference context with
``statsmodels.stats.contingency_tables.StratifiedTable`` and writes one record for
each primary no-burden, leave-two-out background-burden and original total-burden
estimate in the full, WES/WGS and targeted-panel populations.  This contract stage
checks that the independent comparison is complete, that all estimates passed the
declared numerical tolerances and that downstream compatibility aliases point to the
primary no-burden specification while the total-burden estimate remains diagnostic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import TABLES


EXPECTED_SPECIFICATION_COUNTS = {
    "noBurden": 38_609,
    "leaveTwoOut": 15_149,
    "totalBurden": 15_034,
}

EXPECTED_CROSS_ASSAY_COUNTS = {
    "noBurden": 13_253,
    "leaveTwoOut": 5_301,
    "totalBurden": 5_450,
}


def main() -> None:
    adjusted = pd.read_csv(
        TABLES / "cooccurrence_curated_adjusted_sensitivity.csv", low_memory=False
    )
    validation = pd.read_csv(TABLES / "cmh_three_specification_validation.csv")

    if len(adjusted) != 74_582 or adjusted.cancer.nunique() != 50:
        raise AssertionError("The complete pairwise analysis contract is not 74,582 rows across 50 cancers")
    expected_grid = {
        (specification, assay)
        for specification in EXPECTED_SPECIFICATION_COUNTS
        for assay in ("full", "wes", "panel")
    }
    observed_grid = set(zip(validation.specification, validation.assay))
    if observed_grid != expected_grid:
        raise AssertionError("The independent CMH validation does not cover all three specifications and assay classes")
    if len(validation) != 135 or not validation.status.eq("PASS").all():
        raise AssertionError("The independent CMH validation is incomplete or contains a failed comparison")
    if validation.absOrDifference.max() > 1e-10:
        raise AssertionError("The CMH odds-ratio difference exceeds the numerical tolerance")
    if validation.maxAbsCiDifference.max() > 1e-4:
        raise AssertionError("The CMH confidence-interval difference exceeds the numerical tolerance")
    if validation.absPDifference.max() > 1e-9:
        raise AssertionError("The CMH P-value difference exceeds the numerical tolerance")

    for specification, expected_count in EXPECTED_SPECIFICATION_COUNTS.items():
        observed = int(adjusted[f"{specification}_full_fdr"].lt(0.05).sum())
        if observed != expected_count:
            raise AssertionError(
                f"{specification} FDR-significant count is {observed:,}; expected {expected_count:,}"
            )
        observed_cross_assay = int(
            adjusted[f"{specification}CrossAssayConcordant"]
            .fillna(False)
            .astype(bool)
            .sum()
        )
        if observed_cross_assay != EXPECTED_CROSS_ASSAY_COUNTS[specification]:
            raise AssertionError(
                f"{specification} cross-assay count is {observed_cross_assay:,}; "
                f"expected {EXPECTED_CROSS_ASSAY_COUNTS[specification]:,}"
            )

    alias_pairs = (
        ("cmh_full_or", "noBurden_full_or"),
        ("cmh_full_ciLow", "noBurden_full_ciLow"),
        ("cmh_full_ciHigh", "noBurden_full_ciHigh"),
        ("cmh_full_p", "noBurden_full_p"),
        ("cmh_full_fdr", "noBurden_full_fdr"),
        ("cmh_wes_or", "noBurden_wes_or"),
        ("cmh_panel_or", "noBurden_panel_or"),
    )
    for alias, primary in alias_pairs:
        left = adjusted[alias].to_numpy(float)
        right = adjusted[primary].to_numpy(float)
        if not np.allclose(left, right, equal_nan=True, rtol=0, atol=0):
            raise AssertionError(f"Compatibility alias {alias} does not exactly equal {primary}")

    print(
        f"Validated {len(validation):,} reference-context/specification/assay estimates: "
        f"{int(validation.status.eq('PASS').sum()):,} PASS"
    )
    print(f"Maximum absolute OR difference: {validation.absOrDifference.max():.3g}")
    print(f"Maximum absolute CI difference: {validation.maxAbsCiDifference.max():.3g}")
    print(f"Maximum absolute P difference: {validation.absPDifference.max():.3g}")
    print("Verified primary, leave-two-out and total-burden diagnostic contracts")


if __name__ == "__main__":
    main()
