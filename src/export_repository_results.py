"""Export compact, notebook-ready tables from the completed scientific analysis.

The analytical stages retain complete matrices and figure source data. This stage
selects the columns required to audit the principal and sensitivity analyses and
writes them to ``results/tables`` with stable names and column order. The export
includes every screened gene-pair row, selects the stable public schema and excludes
large patient-level or model-level inputs that cannot be redistributed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent


def read_csv(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required analysis output is missing: {path}")
    frame = pd.read_csv(path)
    if columns is None:
        return frame
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{path} lacks required columns: {missing}")
    return frame.loc[:, columns].copy()


def read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required analysis output is missing: {path}")
    return pd.read_parquet(path, columns=columns)


def cohort_summary(analysis_root: Path, complete_tables: Path) -> pd.DataFrame:
    processed = analysis_root / "data" / "processed"
    analytical_tables = analysis_root / "results" / "tables"

    cohort = read_csv(processed / "cohort_studies.csv")
    excluded = read_csv(analytical_tables / "studies_excluded.csv")
    canonical = read_parquet(
        processed / "canonical_samples.parquet", ["studyId", "sampleId"]
    )
    samples = read_parquet(
        processed / "analysis_samples_curated.parquet",
        [
            "studyId",
            "sampleId",
            "tissueEligible",
            "analysisEligible",
            "broadCancerCode",
            "analysisCancerCode",
        ],
    )
    panel = read_csv(processed / "gene_panel.csv")
    assay = read_parquet(
        processed / "sample_assay.parquet", ["studyId", "sampleId", "assayType"]
    )
    study_coverage = read_csv(complete_tables / "curated_study_completeness.csv")

    tissue = samples.loc[samples.tissueEligible.fillna(False).astype(bool)].copy()
    selected = samples.loc[samples.analysisEligible.fillna(False).astype(bool)].copy()
    selected_assay = selected[["studyId", "sampleId"]].merge(
        assay,
        on=["studyId", "sampleId"],
        how="left",
        validate="one_to_one",
    )
    if selected_assay.assayType.isna().any():
        raise AssertionError("Selected tumours without an assay assignment")

    coverage_columns = [
        "nPotentialSampleGenePairs",
        "nCallableSampleGenePairs",
        "nUnassayedSampleGenePairs",
        "nMutatedSampleGenePairs",
        "nCallableZeroMutationSampleGenePairs",
    ]
    missing_coverage = [
        column for column in coverage_columns if column not in study_coverage.columns
    ]
    if missing_coverage:
        raise KeyError(
            f"curated_study_completeness.csv lacks required columns: {missing_coverage}"
        )
    coverage = study_coverage[coverage_columns].sum().astype(int)

    records = [
        ("portal_studies_examined", len(cohort) + len(excluded), "studies"),
        ("studies_retained_after_screening", len(cohort), "studies"),
        ("input_study_sample_records", int(cohort.nSamples.sum()), "records"),
        (
            "studies_after_cross_study_specimen_reconciliation",
            canonical.studyId.nunique(),
            "studies",
        ),
        ("canonical_specimens", len(canonical), "specimens"),
        (
            "studies_with_eligible_tissue_specimens",
            tissue.studyId.nunique(),
            "studies",
        ),
        ("eligible_tissue_specimens", len(tissue), "specimens"),
        ("contributing_studies", selected.studyId.nunique(), "studies"),
        ("selected_tissue_tumours", len(selected), "tumours"),
        ("reviewed_cancer_families", selected.broadCancerCode.nunique(), "families"),
        ("detailed_oncotree_codes", selected.analysisCancerCode.nunique(), "codes"),
        ("genes_in_compendium", len(panel), "genes"),
        (
            "documented_wes_wgs_tumours",
            int(selected_assay.assayType.eq("WES/WGS").sum()),
            "tumours",
        ),
        (
            "targeted_panel_tumours",
            int(selected_assay.assayType.eq("Targeted panel").sum()),
            "tumours",
        ),
        (
            "unverified_assay_tumours",
            int(selected_assay.assayType.str.startswith("Unverified", na=False).sum()),
            "tumours",
        ),
        (
            "potential_tumour_gene_observations",
            coverage.nPotentialSampleGenePairs,
            "observations",
        ),
        (
            "callable_tumour_gene_observations",
            coverage.nCallableSampleGenePairs,
            "observations",
        ),
        (
            "unassayed_tumour_gene_observations",
            coverage.nUnassayedSampleGenePairs,
            "observations",
        ),
        (
            "mutation_positive_tumour_gene_observations",
            coverage.nMutatedSampleGenePairs,
            "observations",
        ),
        (
            "callable_mutation_negative_observations",
            coverage.nCallableZeroMutationSampleGenePairs,
            "observations",
        ),
    ]
    summary = pd.DataFrame(records, columns=["metric", "value", "unit"])
    summary["value"] = summary.value.astype(int)

    values = summary.set_index("metric").value
    if (
        values.callable_tumour_gene_observations
        + values.unassayed_tumour_gene_observations
        != values.potential_tumour_gene_observations
    ):
        raise AssertionError("Callable and unassayed observations do not partition the total")
    if (
        values.mutation_positive_tumour_gene_observations
        + values.callable_mutation_negative_observations
        != values.callable_tumour_gene_observations
    ):
        raise AssertionError("Mutation-positive and mutation-negative observations do not partition callability")
    return summary


def export_tables(
    analysis_root: Path, complete_tables: Path, output_dir: Path
) -> dict[str, int]:
    source = analysis_root / "results" / "source_data"
    analytical_tables = analysis_root / "results" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    specifications: dict[str, tuple[Path, list[str] | None]] = {
        "cohort_by_cancer.csv": (
            source / "figureS1_panel_a_cancer_coverage.csv",
            [
                "rank",
                "cancerGroup",
                "cancerGroupLabel",
                "nEligibleSamples",
                "nWesWgsSamples",
                "nTargetedSamples",
                "nCallableSampleGenePairs",
                "nUnassayedSampleGenePairs",
                "nMutatedSampleGenePairs",
                "callableFractionPct",
                "targetedFractionPct",
                "assayStratum",
            ],
        ),
        "gene_prevalence.csv": (
            source / "figure2_panel_a_prevalence.csv",
            [
                "gene",
                "entrezGeneId",
                "roleInCancer",
                "nProfiled",
                "nMutated",
                "freqPct",
                "freqCiLowPct",
                "freqCiHighPct",
                "pctCuratedSamplesProfilingGene",
            ],
        ),
        "mutation_evidence.csv": (
            analytical_tables / "cmc_evidence_curated.csv",
            [
                "gene",
                "nMutationRecords",
                "nTierMatched",
                "pctTierMatched",
                "tierMatchedCiLowPct",
                "tierMatchedCiHighPct",
                "nCMCtier1",
                "nCMCtier2",
                "nCMCtier3",
                "nCMCother",
                "nNotrepresented",
            ],
        ),
        "hotspot_by_cancer.csv": (
            source / "figure2_panel_d_hotspot_context.csv",
            ["gene", "aa", "broadCancerCode", "nSamples", "hotspot"],
        ),
        "landscape_heatmap.csv": (
            source / "figure3_integrated_heatmap.csv",
            [
                "cancerGroup",
                "cancerGroupLabel",
                "gene",
                "roleInCancer",
                "cosmicTier",
                "nEligibleSamples",
                "nCallable",
                "nUnassayed",
                "nMutated",
                "prevalencePct",
                "prevalenceCiLowPct",
                "prevalenceCiHighPct",
                "callabilityState",
            ],
        ),
        "gene_pair_contexts.csv": (
            source / "figure5_panel_a_reference_specifications.csv",
            [
                "displayOrder",
                "cancer",
                "pair",
                "geneA",
                "geneB",
                "displayLabel",
                "noBurden_full_or",
                "noBurden_full_ciLow",
                "noBurden_full_ciHigh",
                "noBurden_full_p",
                "noBurden_full_fdr",
                "leaveTwoOut_full_or",
                "leaveTwoOut_full_ciLow",
                "leaveTwoOut_full_ciHigh",
                "leaveTwoOut_full_p",
                "leaveTwoOut_full_fdr",
                "totalBurden_full_or",
                "totalBurden_full_ciLow",
                "totalBurden_full_ciHigh",
                "totalBurden_full_p",
                "totalBurden_full_fdr",
                "signStableNoBurdenLeaveTwoOut",
                "effectStableNoBurdenLeaveTwoOut",
                "noBurden_full_nInformativeStrata",
                "leaveTwoOut_full_nInformativeStrata",
                "totalBurden_full_nInformativeStrata",
                "effectStabilityDefinition",
            ],
        ),
        "pathway_by_cancer.csv": (
            source / "figure6_pathway_heatmap.csv",
            [
                "cancer",
                "pathway",
                "nAnalysisSamples",
                "nStudies",
                "nDocumentedWesWgs",
                "nTargetedPanel",
                "nWesWithPathwayMutation",
                "wesMutationPct",
                "meanTargetedTemplateCoveragePct",
            ],
        ),
        "functional_crispr.csv": (
            source / "figure7_panel_a_functional_evidence_matrix.csv",
            [
                "gene",
                "freqPct",
                "pctTierMatched",
                "nModelsAdjusted",
                "nMutAdjusted",
                "nHotspotNegativeAdjusted",
                "nInformativeLineages",
                "adjustedEffect",
                "adjustedCiLow",
                "adjustedCiHigh",
                "adjustedFdr",
                "nSensitisingCompounds",
                "bestPrismSensitisation",
                "bestPrismFdr",
                "bestPrismCompound",
            ],
        ),
        "functional_prism_selected.csv": (
            source / "figure7_panel_b_prism_forest.csv",
            [
                "gene",
                "compound",
                "selectionClass",
                "nModelsAdjusted",
                "nMutAdjusted",
                "nHotspotNegativeAdjusted",
                "nInformativeLineages",
                "adjustedSensitisation",
                "adjustedSensitisationCiLow",
                "adjustedSensitisationCiHigh",
                "adjustedFdr",
            ],
        ),
        "functional_cross_layer.csv": (
            source / "figure7_panel_e_cross_layer_synthesis.csv",
            [
                "gene",
                "adjustedEffect",
                "adjustedFdr",
                "nSensitisingCompounds",
                "bestPrismCompound",
                "bestPrismSensitisation",
                "bestPrismFdr",
                "crisprFdrSupported",
                "prismFdrSupported",
            ],
        ),
        "network_contexts.csv": (
            source / "figure8_main_edge_contexts.csv",
            [
                "cancer",
                "pair",
                "geneA",
                "geneB",
                "full_n",
                "full_nBoth",
                "noBurden_full_or",
                "noBurden_full_ciLow",
                "noBurden_full_ciHigh",
                "noBurden_full_fdr",
                "leaveTwoOut_full_or",
                "leaveTwoOut_full_ciLow",
                "leaveTwoOut_full_ciHigh",
                "leaveTwoOut_full_fdr",
                "totalBurden_full_or",
                "totalBurden_full_ciLow",
                "totalBurden_full_ciHigh",
                "totalBurden_full_fdr",
                "primaryDirection",
                "signStableNoBurdenLeaveTwoOut",
                "effectStableNoBurdenLeaveTwoOut",
                "noBurdenCrossAssayConcordant",
                "leaveTwoOutCrossAssayConcordant",
                "edgeModule",
                "displaySelection",
            ],
        ),
        "network_composition.csv": (
            source / "figure8_panel_c_cancer_context_composition.csv",
            ["cancer", "direction", "nContexts"],
        ),
        "survival_medians.csv": (
            analytical_tables / "survival_curated_pair_groups.csv",
            [
                "context",
                "scope",
                "cancer",
                "geneA",
                "geneB",
                "group",
                "nPatients",
                "nEvents",
                "medianOsMonths",
                "medianOsStatus",
            ],
        ),
        "survival_sensitivity.csv": (
            analytical_tables / "survival_model_sensitivity_summary.csv",
            [
                "context",
                "scope",
                "cancer",
                "geneA",
                "geneB",
                "model",
                "varianceEstimator",
                "adjustmentTerms",
                "nPatients",
                "nEvents",
                "nStudies",
                "nStrata",
                "coefficient",
                "standardError",
                "hazardRatio",
                "ciLow",
                "ciHigh",
                "p",
                "fdr",
                "fitStatus",
            ],
        ),
        "hallmark_by_cancer.csv": (
            analytical_tables / "hallmark_mutation_by_cancer_complete.csv",
            [
                "cancerGroup",
                "nEligibleSamples",
                "nDocumentedWesWgs",
                "nTargetedPanel",
                "nStudies",
                "hallmark",
                "nHallmarkGenes",
                "nMutatedDocumentedWesWgs",
                "mutationPctDocumentedWesWgs",
                "mutationCiLowPct",
                "mutationCiHighPct",
                "estimateStatus",
            ],
        ),
        "gene_pair_three_specifications.csv": (
            analytical_tables / "cooccurrence_curated_adjusted_sensitivity.csv",
            [
                "cancer",
                "geneA",
                "geneB",
                "pair",
                "full_n",
                "full_nA",
                "full_nB",
                "full_nBoth",
                "noBurden_full_or",
                "noBurden_full_ciLow",
                "noBurden_full_ciHigh",
                "noBurden_full_p",
                "noBurden_full_fdr",
                "noBurden_full_nStrata",
                "noBurden_full_nInformativeStrata",
                "noBurden_wes_or",
                "noBurden_wes_p",
                "noBurden_wes_fdr",
                "noBurden_panel_or",
                "noBurden_panel_p",
                "noBurden_panel_fdr",
                "noBurdenSameAssayDirection",
                "noBurdenCrossAssayConcordant",
                "leaveTwoOut_full_or",
                "leaveTwoOut_full_ciLow",
                "leaveTwoOut_full_ciHigh",
                "leaveTwoOut_full_p",
                "leaveTwoOut_full_fdr",
                "leaveTwoOut_full_nStrata",
                "leaveTwoOut_full_nInformativeStrata",
                "leaveTwoOut_wes_or",
                "leaveTwoOut_wes_p",
                "leaveTwoOut_wes_fdr",
                "leaveTwoOut_panel_or",
                "leaveTwoOut_panel_p",
                "leaveTwoOut_panel_fdr",
                "leaveTwoOutSameAssayDirection",
                "leaveTwoOutCrossAssayConcordant",
                "totalBurden_full_or",
                "totalBurden_full_ciLow",
                "totalBurden_full_ciHigh",
                "totalBurden_full_p",
                "totalBurden_full_fdr",
                "totalBurden_full_nStrata",
                "totalBurden_full_nInformativeStrata",
                "totalBurden_wes_or",
                "totalBurden_wes_p",
                "totalBurden_wes_fdr",
                "totalBurden_panel_or",
                "totalBurden_panel_p",
                "totalBurden_panel_fdr",
                "totalBurdenSameAssayDirection",
                "totalBurdenCrossAssayConcordant",
                "primaryDirection",
                "leaveTwoOutDirection",
                "totalBurdenDirection",
                "signStableNoBurdenLeaveTwoOut",
                "signStableAcrossAllSpecifications",
                "effectStableNoBurdenLeaveTwoOut",
                "effectStableAcrossAllSpecifications",
                "leaveTwoOutMinusNoBurdenLog2Or",
                "totalBurdenMinusNoBurdenLog2Or",
                "effectStabilityDefinition",
                "primarySpecification",
                "sensitivitySpecification",
                "diagnosticSpecification",
            ],
        ),
        "gene_pair_screen_summary.csv": (
            source / "figure5_panel_c_robustness_funnel.csv",
            None,
        ),
        "gene_pair_study_heterogeneity.csv": (
            analytical_tables / "pairwise_study_heterogeneity_primary.csv",
            None,
        ),
        "gene_pair_leave_one_study_out.csv": (
            analytical_tables / "pairwise_leave_one_study_out_primary.csv",
            None,
        ),
        "gene_pair_off_panel_sensitivity.csv": (
            analytical_tables / "pairwise_assay_discordant_specimen_sensitivity.csv",
            None,
        ),
        "assay_discordance_by_study.csv": (
            analytical_tables / "assay_discordance_study_sensitivity.csv",
            None,
        ),
        "interactome_degree_matched_null.csv": (
            analytical_tables / "interactome_degree_matched_null.csv",
            None,
        ),
        "survival_joint_state_interaction.csv": (
            analytical_tables / "survival_joint_state_and_interaction_summary.csv",
            None,
        ),
        "survival_ph_diagnostics.csv": (
            analytical_tables / "survival_ph_diagnostics.csv",
            None,
        ),
        "survival_piecewise_hazard_ratios.csv": (
            analytical_tables / "survival_piecewise_hazard_ratios.csv",
            None,
        ),
        "survival_rmst_differences.csv": (
            analytical_tables / "survival_rmst_differences.csv",
            None,
        ),
        "survival_time_varying_hazard_ratios.csv": (
            analytical_tables / "survival_time_varying_hazard_ratios.csv",
            None,
        ),
        "survival_primary_tumour_sensitivity.csv": (
            analytical_tables / "survival_primary_tumour_sensitivity.csv",
            None,
        ),
        "survival_off_panel_sensitivity.csv": (
            analytical_tables / "survival_assay_discordance_exclusion_sensitivity.csv",
            None,
        ),
        "survival_off_panel_specimen_audit.csv": (
            analytical_tables / "survival_assay_discordance_specimen_audit.csv",
            None,
        ),
        "survival_study_specific_hazard_ratios.csv": (
            analytical_tables / "survival_study_specific_hazard_ratios.csv",
            None,
        ),
        "survival_study_meta_analysis.csv": (
            analytical_tables / "survival_study_meta_analysis.csv",
            None,
        ),
        "survival_leave_one_study_out.csv": (
            analytical_tables / "survival_leave_one_study_out.csv",
            None,
        ),
        "survival_leave_one_study_out_summary.csv": (
            analytical_tables / "survival_leave_one_study_out_summary.csv",
            None,
        ),
        "pathway_gene_representation.csv": (
            analytical_tables / "pathway_gene_representation.csv",
            None,
        ),
        "functional_three_group_comparator.csv": (
            analytical_tables / "functional_three_group_comparator_sensitivity.csv",
            None,
        ),
        "functional_leave_one_lineage_out.csv": (
            analytical_tables / "functional_leave_one_lineage_out.csv",
            None,
        ),
        "functional_leave_one_lineage_out_summary.csv": (
            analytical_tables / "functional_leave_one_lineage_out_summary.csv",
            None,
        ),
    }

    exported: dict[str, int] = {}
    summary = cohort_summary(analysis_root, complete_tables)
    summary.to_csv(output_dir / "cohort_summary.csv", index=False)
    exported["cohort_summary.csv"] = len(summary)

    for filename, (input_path, columns) in specifications.items():
        frame = read_csv(input_path, columns)
        if filename == "cohort_by_cancer.csv":
            frame["assayStratum"] = frame.assayStratum.replace(
                {"Mixed WES/WGS + targeted panel": "Mixed WES/WGS and targeted panel"}
            )
        elif filename == "mutation_evidence.csv":
            prevalence_order = read_csv(
                analytical_tables / "gene_frequencies_curated.csv", ["gene"]
            )
            order = {
                gene: position
                for position, gene in enumerate(prevalence_order.gene.astype(str))
            }
            frame["_prevalenceOrder"] = frame.gene.astype(str).map(order)
            if frame._prevalenceOrder.isna().any():
                raise AssertionError("Mutation-evidence gene absent from the prevalence table")
            frame = (
                frame.sort_values("_prevalenceOrder", kind="stable")
                .drop(columns="_prevalenceOrder")
                .reset_index(drop=True)
            )
        frame.to_csv(output_dir / filename, index=False, float_format="%.10g")
        exported[filename] = len(frame)

    survival = read_csv(
        analytical_tables / "survival_joint_state_and_interaction_summary.csv"
    )
    for scope, filename in (
        ("pan-cancer", "survival_pan_cancer.csv"),
        ("cancer-specific", "survival_cancer_specific.csv"),
    ):
        frame = survival.loc[survival.scope.eq(scope)].reset_index(drop=True)
        if frame.empty:
            raise AssertionError(f"No {scope} survival models were available for export")
        frame.to_csv(output_dir / filename, index=False, float_format="%.10g")
        exported[filename] = len(frame)

    if len(exported) != 41:
        raise AssertionError(f"Expected 41 repository tables; exported {len(exported)}")
    return exported


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=ROOT,
        help="repository root containing completed analytical outputs",
    )
    parser.add_argument(
        "--complete-table-dir",
        type=Path,
        default=None,
        help="directory containing the complete study and callability tables",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="destination for compact result tables",
    )
    args = parser.parse_args()
    analysis_root = args.analysis_root.resolve()
    complete_tables = (
        args.complete_table_dir.resolve()
        if args.complete_table_dir is not None
        else analysis_root / "results" / "submission_tables"
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else analysis_root / "results" / "tables"
    )
    exported = export_tables(analysis_root, complete_tables, output_dir)
    print("Exported compact repository results:")
    for filename, rows in exported.items():
        print(f"  {filename}: {rows:,} rows")


if __name__ == "__main__":
    main()
