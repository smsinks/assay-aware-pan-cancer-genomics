"""Stage 20 — build complete curated mutation-callability tables.

The analysis expands the biospecimen-curated, one-sample-per-patient cohort over the
complete 1,341-gene panel
without replacing an unassayed targeted-panel gene by wild type.  WES/WGS samples are
callable for every panel gene; targeted samples are callable only for genes documented
in ``panel_gene_membership.parquet``.

Every study-gene and cancer-group-gene combination is retained, including:

* ``Mutation observed`` -- at least one callable eligible sample is mutated;
* ``Callable, zero mutation events`` -- callable samples exist but no event was seen;
* ``Unassayed`` -- no eligible sample in the group profiled the gene.

For partially profiled groups, callable and unassayed sample counts are both reported.
Mutation numerators are unique sample-gene pairs; mutation-record counts retain multiple
variants in the same sample and gene as a separate provenance field.

Inputs
------
data/processed/analysis_samples_curated.parquet
data/processed/sample_assay.parquet
data/processed/panel_gene_membership.parquet
data/processed/mutations_curated.parquet
data/processed/gene_panel.csv
results/tables/cohort_study_manifest.csv

Outputs
-------
results/complete_tables/curated_study_completeness.csv
results/complete_tables/curated_cancer_gene_prevalence_complete.csv
results/complete_tables/curated_study_gene_callability_complete.csv
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from config import PROCESSED, RESULTS, TABLES


EXPECTED_CURATED_STUDIES = 367
EXPECTED_PANEL_GENES = 1_341
OUTPUT_DIR = RESULTS / "complete_tables"

STUDY_OUTPUT = OUTPUT_DIR / "curated_study_completeness.csv"
CANCER_GENE_OUTPUT = OUTPUT_DIR / "curated_cancer_gene_prevalence_complete.csv"
STUDY_GENE_OUTPUT = OUTPUT_DIR / "curated_study_gene_callability_complete.csv"

ASSAY_WES_DOCUMENTED = "WES/WGS documented"
ASSAY_WES_ASSUMED = "WES/WGS assumed (no panel metadata)"
ASSAY_TARGETED = "Targeted panel"
ASSAY_UNVERIFIED = "Mutation-profile membership or assay scope unverified"


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise KeyError(f"{label} lacks required columns: {sorted(missing)}")


def _write_csv_atomic(frame: pd.DataFrame, destination: Path) -> None:
    """Write a large CSV via /tmp, then copy the completed file into the workspace."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f"{destination.stem}_", suffix=".csv", dir=tempfile.gettempdir(), delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(temporary, index=False, na_rep="", float_format="%.8g")
        shutil.copyfile(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _wilson_interval(successes: pd.Series, totals: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised two-sided 95% Wilson interval, returned as percentages."""
    k = successes.to_numpy(dtype=float)
    n = totals.to_numpy(dtype=float)
    low = np.full(len(k), np.nan, dtype=float)
    high = np.full(len(k), np.nan, dtype=float)
    valid = n > 0
    if not valid.any():
        return low, high
    z = 1.959963984540054
    p = k[valid] / n[valid]
    denominator = 1 + z * z / n[valid]
    centre = (p + z * z / (2 * n[valid])) / denominator
    half = (
        z
        * np.sqrt(p * (1 - p) / n[valid] + z * z / (4 * n[valid] ** 2))
        / denominator
    )
    low[valid] = 100 * np.clip(centre - half, 0, 1)
    high[valid] = 100 * np.clip(centre + half, 0, 1)
    return low, high


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and validate exact eligible sample, assay, panel and mutation artifacts."""
    samples = pd.read_parquet(PROCESSED / "analysis_samples_curated.parquet")
    _require_columns(
        samples,
        {
            "studyId",
            "sampleId",
            "patientId",
            "patientKey",
            "broadCancerCode",
            "broadCancerType",
            "analysisEligible",
        },
        "analysis_samples_curated.parquet",
    )
    samples = samples.loc[samples.analysisEligible].copy()
    samples["sampleKey"] = samples.studyId.astype(str) + "::" + samples.sampleId.astype(str)
    samples["studyPatientKey"] = np.where(
        samples.patientId.notna() & samples.patientId.astype(str).str.strip().ne(""),
        samples.studyId.astype(str) + "::" + samples.patientId.astype(str),
        samples.studyId.astype(str) + "::SAMPLE::" + samples.sampleId.astype(str),
    )
    samples["cancerGroup"] = samples.broadCancerCode.astype(str).str.strip()

    if samples.duplicated(["studyId", "sampleId"]).any():
        raise ValueError("Eligible sample rows are not unique on studyId + sampleId")
    if samples.sampleKey.duplicated().any():
        raise ValueError("Constructed studyId + sampleId sampleKey is not unique")
    if samples.patientKey.isna().any() or samples.patientKey.duplicated().any():
        raise ValueError("Primary cohort is not one unique row per curated patientKey")
    if samples.cancerGroup.eq("").any() or samples.cancerGroup.str.upper().eq("NAN").any():
        raise ValueError("An eligible sample lacks a broad cancer-group assignment")
    if samples.studyId.nunique() != EXPECTED_CURATED_STUDIES:
        raise ValueError(
            f"Expected {EXPECTED_CURATED_STUDIES} contributing studies, found "
            f"{samples.studyId.nunique()}"
        )

    assay = pd.read_parquet(PROCESSED / "sample_assay.parquet")
    _require_columns(
        assay,
        {"studyId", "sampleId", "genePanelId", "assayType", "panelMetadataAvailable"},
        "sample_assay.parquet",
    )
    if assay.duplicated(["studyId", "sampleId"]).any():
        raise ValueError("Sample assay assignments are not unique on studyId + sampleId")
    samples = samples.merge(
        assay,
        on=["studyId", "sampleId"],
        how="left",
        validate="one_to_one",
    )
    if samples.assayType.isna().any():
        examples = samples.loc[samples.assayType.isna(), ["studyId", "sampleId"]].head()
        raise ValueError(f"Eligible samples without assay assignment: {examples.to_dict('records')}")

    allowed_assays = {
        "WES/WGS",
        "WES/WGS (assumed; no panel metadata)",
        "Targeted panel",
        "Unverified mutation-profile membership",
    }
    unexpected = set(samples.assayType) - allowed_assays
    if unexpected:
        raise ValueError(f"Unexpected assay types: {sorted(unexpected)}")
    samples["assayStratumDetailed"] = np.select(
        [
            samples.assayType.eq("WES/WGS"),
            samples.assayType.eq("WES/WGS (assumed; no panel metadata)"),
            samples.assayType.eq("Targeted panel"),
            samples.assayType.eq("Unverified mutation-profile membership"),
        ],
        [ASSAY_WES_DOCUMENTED, ASSAY_WES_ASSUMED, ASSAY_TARGETED, ASSAY_UNVERIFIED],
        default="ERROR",
    )
    samples["assayGroup"] = np.select(
        [
            samples.assayStratumDetailed.eq(ASSAY_TARGETED),
            samples.assayStratumDetailed.isin([ASSAY_WES_DOCUMENTED, ASSAY_WES_ASSUMED]),
        ],
        ["Targeted panel", "WES/WGS"],
        default="Unverified",
    )
    bad_targeted = samples.assayGroup.eq("Targeted panel") & samples.genePanelId.isna()
    if bad_targeted.any():
        raise ValueError(f"{int(bad_targeted.sum()):,} targeted samples lack a gene-panel ID")

    panel = pd.read_csv(PROCESSED / "gene_panel.csv")
    _require_columns(
        panel,
        {"entrezGeneId", "hugoSymbol", "roleInCancer", "cosmicTier", "highConfidence"},
        "gene_panel.csv",
    )
    panel = panel.copy()
    panel["entrezGeneId"] = panel.entrezGeneId.astype(int)
    panel = panel.drop_duplicates("entrezGeneId", keep="first").reset_index(drop=True)
    panel["panelOrder"] = np.arange(len(panel), dtype=int)
    if len(panel) != EXPECTED_PANEL_GENES:
        raise ValueError(f"Expected {EXPECTED_PANEL_GENES} panel genes, found {len(panel)}")

    membership = pd.read_parquet(PROCESSED / "panel_gene_membership.parquet")
    _require_columns(membership, {"genePanelId", "entrezGeneId"}, "panel_gene_membership.parquet")
    membership = membership[["genePanelId", "entrezGeneId"]].drop_duplicates().copy()
    membership["entrezGeneId"] = membership.entrezGeneId.astype(int)
    membership = membership[membership.entrezGeneId.isin(panel.entrezGeneId)]
    used_panels = set(samples.loc[samples.assayGroup.eq("Targeted panel"), "genePanelId"])
    missing_panels = used_panels - set(membership.genePanelId)
    if missing_panels:
        raise ValueError(f"Targeted panels without documented gene membership: {sorted(missing_panels)}")

    mutations_raw = pd.read_parquet(PROCESSED / "mutations_curated.parquet")
    _require_columns(
        mutations_raw,
        {"studyId", "sampleId", "entrezGeneId"},
        "mutations_curated.parquet",
    )
    mutation_columns = ["studyId", "sampleId", "entrezGeneId"]
    mutations = mutations_raw[mutation_columns].merge(
        samples[
            [
                "studyId",
                "sampleId",
                "sampleKey",
                "patientKey",
                "studyPatientKey",
                "cancerGroup",
                "assayGroup",
                "assayStratumDetailed",
                "genePanelId",
            ]
        ],
        on=["studyId", "sampleId"],
        how="inner",
        validate="many_to_one",
    )
    if len(mutations) != len(mutations_raw):
        print(
            "Warning: excluded "
            f"{len(mutations_raw) - len(mutations):,} mutation records outside exact eligible "
            "studyId + sampleId keys"
        )
    mutations["entrezGeneId"] = mutations.entrezGeneId.astype(int)
    off_panel_genes = set(mutations.entrezGeneId) - set(panel.entrezGeneId)
    if off_panel_genes:
        raise ValueError(f"Curated mutations contain genes outside the 1,341-gene panel: {off_panel_genes}")

    # Re-assert the persisted callability contract instead of trusting a positive call to
    # imply profiling.  This protects all later numerators from an off-panel conflict.
    targeted_mutations = mutations.loc[
        mutations.assayGroup.eq("Targeted panel"), ["genePanelId", "entrezGeneId"]
    ].drop_duplicates()
    targeted_audit = targeted_mutations.merge(
        membership.assign(documentedOnPanel=True),
        on=["genePanelId", "entrezGeneId"],
        how="left",
        validate="one_to_one",
    )
    if targeted_audit.documentedOnPanel.ne(True).any():
        bad = targeted_audit.loc[
            targeted_audit.documentedOnPanel.ne(True), ["genePanelId", "entrezGeneId"]
        ].head()
        raise ValueError(f"Curated targeted mutations violate panel membership: {bad.to_dict('records')}")

    return samples, panel, membership, mutations


def build_group_gene_table(
    samples: pd.DataFrame,
    panel: pd.DataFrame,
    membership: pd.DataFrame,
    mutations: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    """Return a complete group x 1,341-gene callability and mutation table."""
    work = samples.copy()
    work["_wesDocumented"] = work.assayStratumDetailed.eq(ASSAY_WES_DOCUMENTED).astype(int)
    work["_wesAssumed"] = work.assayStratumDetailed.eq(ASSAY_WES_ASSUMED).astype(int)
    work["_targeted"] = work.assayStratumDetailed.eq(ASSAY_TARGETED).astype(int)
    work["_unverified"] = work.assayStratumDetailed.eq(ASSAY_UNVERIFIED).astype(int)
    group_summary = (
        work.groupby(group_column, dropna=False)
        .agg(
            nEligibleSamples=("sampleKey", "nunique"),
            nEligiblePatientKeys=("patientKey", "nunique"),
            nWesWgsDocumentedSamples=("_wesDocumented", "sum"),
            nWesWgsAssumedSamples=("_wesAssumed", "sum"),
            nTargetedSamples=("_targeted", "sum"),
            nUnverifiedAssaySamples=("_unverified", "sum"),
        )
        .reset_index()
    )
    group_summary["nWesWgsSamples"] = (
        group_summary.nWesWgsDocumentedSamples + group_summary.nWesWgsAssumedSamples
    )
    if not (
        group_summary.nWesWgsSamples
        + group_summary.nTargetedSamples
        + group_summary.nUnverifiedAssaySamples
        == group_summary.nEligibleSamples
    ).all():
        raise AssertionError(f"Assay counts do not sum to eligible samples for {group_column}")

    gene_columns = [
        "entrezGeneId",
        "hugoSymbol",
        "roleInCancer",
        "cosmicTier",
        "highConfidence",
        "panelOrder",
    ]
    grid = group_summary.merge(panel[gene_columns], how="cross")

    panel_counts = (
        work.loc[work.assayGroup.eq("Targeted panel")]
        .groupby([group_column, "genePanelId"], dropna=False)["sampleKey"]
        .nunique()
        .rename("nPanelSamples")
        .reset_index()
    )
    targeted_callable = (
        panel_counts.merge(membership, on="genePanelId", how="inner", validate="many_to_many")
        .groupby([group_column, "entrezGeneId"], dropna=False)["nPanelSamples"]
        .sum()
        .rename("nCallableTargeted")
        .reset_index()
    )
    grid = grid.merge(
        targeted_callable,
        on=[group_column, "entrezGeneId"],
        how="left",
        validate="one_to_one",
    )
    grid["nCallableTargeted"] = grid.nCallableTargeted.fillna(0).astype(int)
    grid["nCallableWesWgsDocumented"] = grid.nWesWgsDocumentedSamples.astype(int)
    grid["nCallableWesWgsAssumed"] = grid.nWesWgsAssumedSamples.astype(int)
    grid["nCallableWesWgs"] = (
        grid.nCallableWesWgsDocumented + grid.nCallableWesWgsAssumed
    )
    grid["nCallable"] = grid.nCallableWesWgs + grid.nCallableTargeted
    grid["nTargetedUnassayed"] = grid.nTargetedSamples - grid.nCallableTargeted
    grid["nUnassayed"] = grid.nEligibleSamples - grid.nCallable

    unique_mutations = mutations.drop_duplicates(["sampleKey", "entrezGeneId"]).copy()
    unique_mutations["_mutWesDocumented"] = unique_mutations.assayStratumDetailed.eq(
        ASSAY_WES_DOCUMENTED
    ).astype(int)
    unique_mutations["_mutWesAssumed"] = unique_mutations.assayStratumDetailed.eq(
        ASSAY_WES_ASSUMED
    ).astype(int)
    unique_mutations["_mutTargeted"] = unique_mutations.assayStratumDetailed.eq(
        ASSAY_TARGETED
    ).astype(int)
    mutation_counts = (
        unique_mutations.groupby([group_column, "entrezGeneId"], dropna=False)
        .agg(
            nMutated=("sampleKey", "nunique"),
            nMutatedWesWgsDocumented=("_mutWesDocumented", "sum"),
            nMutatedWesWgsAssumed=("_mutWesAssumed", "sum"),
            nMutatedTargeted=("_mutTargeted", "sum"),
        )
        .reset_index()
    )
    mutation_records = (
        mutations.groupby([group_column, "entrezGeneId"], dropna=False)
        .size()
        .rename("nMutationRecords")
        .reset_index()
    )
    grid = grid.merge(
        mutation_counts,
        on=[group_column, "entrezGeneId"],
        how="left",
        validate="one_to_one",
    ).merge(
        mutation_records,
        on=[group_column, "entrezGeneId"],
        how="left",
        validate="one_to_one",
    )
    mutation_integer_columns = [
        "nMutated",
        "nMutatedWesWgsDocumented",
        "nMutatedWesWgsAssumed",
        "nMutatedTargeted",
        "nMutationRecords",
    ]
    grid[mutation_integer_columns] = grid[mutation_integer_columns].fillna(0).astype(int)
    grid["nMutatedWesWgs"] = (
        grid.nMutatedWesWgsDocumented + grid.nMutatedWesWgsAssumed
    )

    numerator_checks = [
        ("nMutated", "nCallable"),
        ("nMutatedWesWgs", "nCallableWesWgs"),
        ("nMutatedWesWgsDocumented", "nCallableWesWgsDocumented"),
        ("nMutatedWesWgsAssumed", "nCallableWesWgsAssumed"),
        ("nMutatedTargeted", "nCallableTargeted"),
    ]
    for numerator, denominator in numerator_checks:
        bad = grid[numerator] > grid[denominator]
        if bad.any():
            examples = grid.loc[
                bad, [group_column, "entrezGeneId", numerator, denominator]
            ].head()
            raise AssertionError(
                f"{numerator} exceeds {denominator}: {examples.to_dict('records')}"
            )
    if (grid.nCallable > grid.nEligibleSamples).any() or (grid.nUnassayed < 0).any():
        raise AssertionError(f"Invalid callable denominator for {group_column}")

    grid["nCallableZeroMutation"] = grid.nCallable - grid.nMutated
    grid["prevalencePct"] = np.where(
        grid.nCallable.gt(0), 100 * grid.nMutated / grid.nCallable, np.nan
    )
    ci_low, ci_high = _wilson_interval(grid.nMutated, grid.nCallable)
    grid["prevalenceCiLowPct"] = ci_low
    grid["prevalenceCiHighPct"] = ci_high
    grid["callabilityState"] = np.select(
        [grid.nCallable.eq(0), grid.nCallable.eq(grid.nEligibleSamples)],
        ["Unassayed in all eligible samples", "Callable in all eligible samples"],
        default="Callable in a subset of eligible samples",
    )
    grid["mutationState"] = np.select(
        [grid.nCallable.eq(0), grid.nMutated.gt(0)],
        ["Unassayed", "Mutation observed"],
        default="Callable, zero mutation events",
    )
    grid["hasUnassayedEligibleSamples"] = grid.nUnassayed.gt(0)
    grid = grid.rename(columns={"hugoSymbol": "gene"})
    return grid.sort_values([group_column, "panelOrder"]).reset_index(drop=True)


def build_study_completeness(
    samples: pd.DataFrame,
    mutations: pd.DataFrame,
    study_gene: pd.DataFrame,
) -> pd.DataFrame:
    """Collapse the complete study-gene matrix to one explicit row per curated study."""
    study_samples = samples.copy()
    study_samples["_wesDocumented"] = study_samples.assayStratumDetailed.eq(
        ASSAY_WES_DOCUMENTED
    ).astype(int)
    study_samples["_wesAssumed"] = study_samples.assayStratumDetailed.eq(
        ASSAY_WES_ASSUMED
    ).astype(int)
    study_samples["_targeted"] = study_samples.assayStratumDetailed.eq(ASSAY_TARGETED).astype(int)
    study_samples["_unverified"] = study_samples.assayStratumDetailed.eq(ASSAY_UNVERIFIED).astype(int)
    sample_summary = (
        study_samples.groupby("studyId")
        .agg(
            nEligibleSamples=("sampleKey", "nunique"),
            nEligiblePatientKeys=("patientKey", "nunique"),
            nStudyPatientKeys=("studyPatientKey", "nunique"),
            nWesWgsDocumentedSamples=("_wesDocumented", "sum"),
            nWesWgsAssumedSamples=("_wesAssumed", "sum"),
            nTargetedSamples=("_targeted", "sum"),
            nUnverifiedAssaySamples=("_unverified", "sum"),
        )
        .reset_index()
    )
    sample_summary["nWesWgsSamples"] = (
        sample_summary.nWesWgsDocumentedSamples + sample_summary.nWesWgsAssumedSamples
    )
    sample_summary["assayStratum"] = np.select(
        [
            sample_summary.nTargetedSamples.eq(0),
            sample_summary.nWesWgsSamples.eq(0),
        ],
        ["WES/WGS only", "Targeted panel only"],
        default="Mixed WES/WGS and targeted panel",
    )
    sample_summary["assayMetadataState"] = np.select(
        [
            sample_summary.nUnverifiedAssaySamples.gt(0),
            sample_summary.nWesWgsAssumedSamples.gt(0),
        ],
        [
            "Includes samples with unverified mutation-profile membership or assay scope",
            "Includes assumed WES/WGS samples without panel metadata",
        ],
        default="All assay assignments documented",
    )

    cancer_counts = (
        study_samples.groupby(["studyId", "cancerGroup"])["sampleKey"]
        .nunique()
        .rename("n")
        .reset_index()
    )
    cancer_counts = cancer_counts.sort_values(
        ["studyId", "n", "cancerGroup"], ascending=[True, False, True]
    )
    primary_cancer = cancer_counts.drop_duplicates("studyId").rename(
        columns={"cancerGroup": "primaryCancerGroup", "n": "nPrimaryCancerGroupSamples"}
    )
    cancer_lists = (
        cancer_counts.groupby("studyId")
        .agg(
            nCancerGroups=("cancerGroup", "nunique"),
            cancerGroups=("cancerGroup", lambda values: ";".join(sorted(set(values)))),
        )
        .reset_index()
    )
    sample_summary = sample_summary.merge(
        primary_cancer[["studyId", "primaryCancerGroup", "nPrimaryCancerGroupSamples"]],
        on="studyId",
        how="left",
        validate="one_to_one",
    ).merge(cancer_lists, on="studyId", how="left", validate="one_to_one")
    sample_summary["cancerGroupState"] = np.where(
        sample_summary.nCancerGroups.eq(1), "Single cancer group", "Multiple cancer groups"
    )
    panel_ids = (
        study_samples.loc[study_samples.assayGroup.eq("Targeted panel")]
        .groupby("studyId")["genePanelId"]
        .agg(lambda values: ";".join(sorted({str(value) for value in values.dropna()})))
        .rename("targetedPanelIds")
        .reset_index()
    )
    sample_summary = sample_summary.merge(panel_ids, on="studyId", how="left")
    sample_summary["targetedPanelIds"] = sample_summary.targetedPanelIds.fillna("")

    mutated_samples = (
        mutations.groupby("studyId")["sampleKey"]
        .nunique()
        .rename("nSamplesWithMutation")
        .reset_index()
    )
    sample_summary = sample_summary.merge(mutated_samples, on="studyId", how="left")
    sample_summary["nSamplesWithMutation"] = sample_summary.nSamplesWithMutation.fillna(0).astype(int)
    sample_summary["nSamplesZeroMutationEvents"] = (
        sample_summary.nEligibleSamples - sample_summary.nSamplesWithMutation
    )
    sample_summary["pctSamplesWithMutation"] = (
        100 * sample_summary.nSamplesWithMutation / sample_summary.nEligibleSamples
    )

    state_frame = study_gene.assign(
        _geneObserved=study_gene.nMutated.gt(0),
        _geneCallableZero=study_gene.nCallable.gt(0) & study_gene.nMutated.eq(0),
        _geneUnassayed=study_gene.nCallable.eq(0),
        _genePartiallyCallable=(study_gene.nCallable.gt(0))
        & (study_gene.nCallable.lt(study_gene.nEligibleSamples)),
        _geneCallableAll=study_gene.nCallable.eq(study_gene.nEligibleSamples),
    )
    coverage_summary = (
        state_frame.groupby("studyId")
        .agg(
            nPanelGenes=("entrezGeneId", "size"),
            nGenesMutationObserved=("_geneObserved", "sum"),
            nGenesCallableZeroMutation=("_geneCallableZero", "sum"),
            nGenesUnassayedAllSamples=("_geneUnassayed", "sum"),
            nGenesPartiallyCallable=("_genePartiallyCallable", "sum"),
            nGenesCallableAllSamples=("_geneCallableAll", "sum"),
            nPotentialSampleGenePairs=("nEligibleSamples", "sum"),
            nCallableSampleGenePairs=("nCallable", "sum"),
            nUnassayedSampleGenePairs=("nUnassayed", "sum"),
            nMutatedSampleGenePairs=("nMutated", "sum"),
            nCallableZeroMutationSampleGenePairs=("nCallableZeroMutation", "sum"),
            nMutationRecords=("nMutationRecords", "sum"),
        )
        .reset_index()
    )
    out = sample_summary.merge(coverage_summary, on="studyId", how="left", validate="one_to_one")
    out["mutationContribution"] = out.nMutationRecords.gt(0)
    out["studyMutationState"] = np.where(
        out.mutationContribution,
        "Mutation events observed",
        "Callable cohort with zero observed mutation events",
    )
    out["studyCallabilityState"] = np.select(
        [
            out.nUnassayedSampleGenePairs.eq(0),
            out.nCallableSampleGenePairs.eq(0),
        ],
        [
            "All panel genes callable in every eligible sample",
            "No documented callable panel genes",
        ],
        default="Partial gene callability; unassayed states retained",
    )

    manifest = pd.read_csv(TABLES / "cohort_study_manifest.csv")
    manifest_columns = [
        column
        for column in (
            "studyId",
            "name",
            "cancerTypeId",
            "refGenome",
            "citation",
            "pmid",
            "importDate",
            "hasCNA",
            "hasSV",
        )
        if column in manifest.columns
    ]
    manifest = manifest[manifest_columns].drop_duplicates("studyId")
    out = out.merge(manifest, on="studyId", how="left", validate="one_to_one").rename(
        columns={"name": "studyName", "cancerTypeId": "portalCancerTypeId"}
    )

    if len(out) != EXPECTED_CURATED_STUDIES or out.studyId.nunique() != len(out):
        raise AssertionError("Study completeness table is not exactly one row per curated study")
    if not (out.nPanelGenes == EXPECTED_PANEL_GENES).all():
        raise AssertionError("A study is missing gene rows")
    if not (
        out.nGenesMutationObserved
        + out.nGenesCallableZeroMutation
        + out.nGenesUnassayedAllSamples
        == EXPECTED_PANEL_GENES
    ).all():
        raise AssertionError("Study gene-state counts do not partition all panel genes")
    if not (
        out.nCallableSampleGenePairs + out.nUnassayedSampleGenePairs
        == out.nPotentialSampleGenePairs
    ).all():
        raise AssertionError("Study callable/unassayed pairs do not partition all possible pairs")
    if not (
        out.nMutatedSampleGenePairs + out.nCallableZeroMutationSampleGenePairs
        == out.nCallableSampleGenePairs
    ).all():
        raise AssertionError("Study mutated/zero-event pairs do not partition callable pairs")

    preferred = [
        "studyId",
        "studyName",
        "citation",
        "pmid",
        "importDate",
        "portalCancerTypeId",
        "primaryCancerGroup",
        "nPrimaryCancerGroupSamples",
        "nCancerGroups",
        "cancerGroups",
        "cancerGroupState",
        "refGenome",
        "hasCNA",
        "hasSV",
        "nEligibleSamples",
        "nEligiblePatientKeys",
        "nStudyPatientKeys",
        "assayStratum",
        "assayMetadataState",
        "nWesWgsSamples",
        "nWesWgsDocumentedSamples",
        "nWesWgsAssumedSamples",
        "nTargetedSamples",
        "nUnverifiedAssaySamples",
        "targetedPanelIds",
        "nSamplesWithMutation",
        "nSamplesZeroMutationEvents",
        "pctSamplesWithMutation",
        "mutationContribution",
        "studyMutationState",
        "studyCallabilityState",
        "nPanelGenes",
        "nGenesMutationObserved",
        "nGenesCallableZeroMutation",
        "nGenesUnassayedAllSamples",
        "nGenesPartiallyCallable",
        "nGenesCallableAllSamples",
        "nPotentialSampleGenePairs",
        "nCallableSampleGenePairs",
        "nUnassayedSampleGenePairs",
        "nMutatedSampleGenePairs",
        "nCallableZeroMutationSampleGenePairs",
        "nMutationRecords",
    ]
    return out[preferred].sort_values("studyId").reset_index(drop=True)


def cancer_group_labels(samples: pd.DataFrame) -> pd.DataFrame:
    counts = (
        samples.groupby(["cancerGroup", "broadCancerType"])["sampleKey"]
        .nunique()
        .rename("n")
        .reset_index()
        .sort_values(["cancerGroup", "n", "broadCancerType"], ascending=[True, False, True])
    )
    primary = counts.drop_duplicates("cancerGroup").rename(
        columns={"broadCancerType": "cancerGroupLabel"}
    )
    labels = (
        counts.groupby("cancerGroup")["broadCancerType"]
        .agg(lambda values: ";".join(sorted(set(values))))
        .rename("cancerGroupSourceLabels")
        .reset_index()
    )
    return primary[["cancerGroup", "cancerGroupLabel"]].merge(
        labels, on="cancerGroup", how="left", validate="one_to_one"
    )


def main() -> None:
    samples, panel, membership, mutations = load_inputs()

    study_gene = build_group_gene_table(
        samples, panel, membership, mutations, group_column="studyId"
    )
    expected_study_gene_rows = EXPECTED_CURATED_STUDIES * EXPECTED_PANEL_GENES
    if len(study_gene) != expected_study_gene_rows:
        raise AssertionError(
            f"Expected {expected_study_gene_rows:,} study-gene rows, found {len(study_gene):,}"
        )
    if study_gene.duplicated(["studyId", "entrezGeneId"]).any():
        raise AssertionError("Study-gene output key is not unique")

    cancer_gene = build_group_gene_table(
        samples, panel, membership, mutations, group_column="cancerGroup"
    )
    n_cancer_groups = samples.cancerGroup.nunique()
    expected_cancer_gene_rows = n_cancer_groups * EXPECTED_PANEL_GENES
    if len(cancer_gene) != expected_cancer_gene_rows:
        raise AssertionError(
            f"Expected {expected_cancer_gene_rows:,} cancer-gene rows, found {len(cancer_gene):,}"
        )
    if cancer_gene.duplicated(["cancerGroup", "entrezGeneId"]).any():
        raise AssertionError("Cancer-group-gene output key is not unique")
    cancer_gene = cancer_gene.merge(
        cancer_group_labels(samples), on="cancerGroup", how="left", validate="many_to_one"
    )

    study_completeness = build_study_completeness(samples, mutations, study_gene)

    study_gene_columns = [
        "studyId",
        "entrezGeneId",
        "gene",
        "roleInCancer",
        "cosmicTier",
        "highConfidence",
        "nEligibleSamples",
        "nEligiblePatientKeys",
        "nWesWgsSamples",
        "nWesWgsDocumentedSamples",
        "nWesWgsAssumedSamples",
        "nTargetedSamples",
        "nUnverifiedAssaySamples",
        "nCallable",
        "nCallableWesWgs",
        "nCallableWesWgsDocumented",
        "nCallableWesWgsAssumed",
        "nCallableTargeted",
        "nUnassayed",
        "nTargetedUnassayed",
        "nMutated",
        "nMutatedWesWgs",
        "nMutatedWesWgsDocumented",
        "nMutatedWesWgsAssumed",
        "nMutatedTargeted",
        "nMutationRecords",
        "nCallableZeroMutation",
        "prevalencePct",
        "prevalenceCiLowPct",
        "prevalenceCiHighPct",
        "callabilityState",
        "mutationState",
        "hasUnassayedEligibleSamples",
    ]
    cancer_gene_columns = [
        "cancerGroup",
        "cancerGroupLabel",
        "cancerGroupSourceLabels",
    ] + [column for column in study_gene_columns if column != "studyId"]

    _write_csv_atomic(study_completeness, STUDY_OUTPUT)
    _write_csv_atomic(cancer_gene[cancer_gene_columns], CANCER_GENE_OUTPUT)
    _write_csv_atomic(study_gene[study_gene_columns], STUDY_GENE_OUTPUT)

    total_possible = int(study_completeness.nPotentialSampleGenePairs.sum())
    total_callable = int(study_completeness.nCallableSampleGenePairs.sum())
    total_unassayed = int(study_completeness.nUnassayedSampleGenePairs.sum())
    total_mutated = int(study_completeness.nMutatedSampleGenePairs.sum())
    total_zero = int(study_completeness.nCallableZeroMutationSampleGenePairs.sum())
    zero_studies = int((~study_completeness.mutationContribution).sum())
    assumed_wes = int(samples.assayStratumDetailed.eq(ASSAY_WES_ASSUMED).sum())

    print("\n=== COMPLETE CURATED COVERAGE TABLES ===")
    print(f"Eligible one-per-patient samples : {len(samples):,}")
    print(f"Contributing studies             : {samples.studyId.nunique():,}")
    print(f"Eligible broad cancer groups     : {n_cancer_groups:,}")
    print(f"Panel genes                      : {len(panel):,}")
    print(f"Study x gene rows                : {len(study_gene):,}")
    print(f"Cancer-group x gene rows         : {len(cancer_gene):,}")
    print(f"Possible sample-gene pairs       : {total_possible:,}")
    print(f"  callable under curated contract: {total_callable:,}")
    print(f"  explicitly unassayed by contract: {total_unassayed:,}")
    print(f"Callable sample-gene pairs       : {total_callable:,}")
    print(f"  mutation observed              : {total_mutated:,}")
    print(f"  zero observed mutation events  : {total_zero:,}")
    print(f"Studies with zero mutation events: {zero_studies:,}")
    print(f"Assumed WES/WGS samples          : {assumed_wes:,}")
    print("\nWrote:")
    print(f"  {STUDY_OUTPUT}")
    print(f"  {CANCER_GENE_OUTPUT}")
    print(f"  {STUDY_GENE_OUTPUT}")


if __name__ == "__main__":
    main()
