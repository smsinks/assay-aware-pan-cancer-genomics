"""Shared mutation-callability contract for targeted-panel analyses.

A targeted-sample mutation is retained only when the sample's documented panel contains
the gene. Rare positive calls outside the recorded membership are not silently counted
against a denominator that excludes their samples: they are separated into an explicit
metadata-conflict audit. Only samples with an explicit WES/WGS assignment remain callable
for every gene; absence of molecular-profile membership metadata is never interpreted as
whole-exome coverage.
"""
from __future__ import annotations

import pandas as pd

from config import PROCESSED


def partition_callable_mutations(
    mutations: pd.DataFrame,
    assay: pd.DataFrame | None = None,
    membership: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return documented-callable mutation rows and targeted-panel conflicts.

    The returned callable frame has exactly the columns and column order supplied in
    ``mutations``. The conflict frame adds assay metadata and a stable reason string for
    audit tables. Every mutation row must resolve to exactly one sample assay assignment.
    """
    required = {"sampleId", "studyId", "entrezGeneId"}
    missing = required - set(mutations.columns)
    if missing:
        raise KeyError(f"Mutation table lacks callability keys: {sorted(missing)}")

    if assay is None:
        assay = pd.read_parquet(PROCESSED / "sample_assay.parquet")
    if membership is None:
        membership = pd.read_parquet(PROCESSED / "panel_gene_membership.parquet")

    assay_columns = [
        "sampleId", "studyId", "genePanelId", "assayType", "panelMetadataAvailable"
    ]
    for optional in ("mutationProfileMembershipStatus", "callabilityEligible"):
        if optional in assay.columns:
            assay_columns.append(optional)
    assay = assay[assay_columns].drop_duplicates(["sampleId", "studyId"])
    if assay.duplicated(["sampleId", "studyId"]).any():
        raise ValueError("Sample assay assignments are not unique")

    member = (
        membership[["genePanelId", "entrezGeneId"]]
        .drop_duplicates()
        .assign(documentedOnPanel=True)
    )
    original_columns = mutations.columns.tolist()
    annotated = mutations.merge(
        assay,
        on=["sampleId", "studyId"],
        how="left",
        validate="many_to_one",
    )
    unresolved = annotated.assayType.isna()
    if unresolved.any():
        examples = annotated.loc[unresolved, ["sampleId", "studyId"]].head().to_dict("records")
        raise ValueError(f"{int(unresolved.sum()):,} mutation rows lack an assay assignment: {examples}")

    annotated = annotated.merge(
        member,
        on=["genePanelId", "entrezGeneId"],
        how="left",
        validate="many_to_one",
    )
    is_wes = annotated.assayType.str.startswith("WES/WGS")
    callable_mask = is_wes | annotated.documentedOnPanel.eq(True)

    kept = annotated.loc[callable_mask, original_columns].copy()
    conflicts = annotated.loc[~callable_mask].copy()
    conflicts["callabilityConflictReason"] = (
        "Observed mutation lacks verified profile membership or assay scope"
    )
    targeted_conflict = conflicts.assayType.eq("Targeted panel")
    conflicts.loc[targeted_conflict, "callabilityConflictReason"] = (
        "Observed targeted-panel mutation absent from documented panel gene list"
    )
    return kept, conflicts
