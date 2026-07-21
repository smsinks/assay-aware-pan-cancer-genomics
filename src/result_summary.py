"""Load packaged analysis tables and derive the headline numerical results."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"

MAIN_FIGURES = [
    "figure1_cohort_callability.png",
    "figure2_driver_evidence.png",
    "figure3_integrated_heatmap.png",
    "figure4_four_oncoprints.png",
    "figure5_conditioned_interactions.png",
    "figure6_pathway_landscape.png",
    "figure7_functional_evidence.png",
    "figure8_curated_network.png",
    "figure9_survival.png",
]
SUPPLEMENTARY_FIGURES = [
    "figureS1_complete_coverage.png",
    "figureS2_pathway_all_cancers.png",
    "figureS3_expanded_network.png",
    "figureS4_survival_diagnostics.png",
    "figureS5_hallmark_landscape.png",
    "figureS6_functional_diagnostics.png",
    "figureS7_additional_oncoprints.png",
]
REQUIRED_TABLES = [
    "cohort_summary.csv",
    "cohort_by_cancer.csv",
    "gene_prevalence.csv",
    "mutation_evidence.csv",
    "hotspot_by_cancer.csv",
    "landscape_heatmap.csv",
    "gene_pair_contexts.csv",
    "pathway_by_cancer.csv",
    "functional_crispr.csv",
    "functional_prism_selected.csv",
    "functional_cross_layer.csv",
    "network_contexts.csv",
    "network_composition.csv",
    "survival_medians.csv",
    "survival_pan_cancer.csv",
    "survival_cancer_specific.csv",
    "survival_sensitivity.csv",
    "hallmark_by_cancer.csv",
    "gene_pair_three_specifications.csv",
    "gene_pair_screen_summary.csv",
    "gene_pair_study_heterogeneity.csv",
    "gene_pair_leave_one_study_out.csv",
    "gene_pair_off_panel_sensitivity.csv",
    "assay_discordance_by_study.csv",
    "interactome_degree_matched_null.csv",
    "survival_joint_state_interaction.csv",
    "survival_ph_diagnostics.csv",
    "survival_piecewise_hazard_ratios.csv",
    "survival_rmst_differences.csv",
    "survival_time_varying_hazard_ratios.csv",
    "survival_primary_tumour_sensitivity.csv",
    "survival_off_panel_sensitivity.csv",
    "survival_off_panel_specimen_audit.csv",
    "survival_study_specific_hazard_ratios.csv",
    "survival_study_meta_analysis.csv",
    "survival_leave_one_study_out.csv",
    "survival_leave_one_study_out_summary.csv",
    "pathway_gene_representation.csv",
    "functional_three_group_comparator.csv",
    "functional_leave_one_lineage_out.csv",
    "functional_leave_one_lineage_out_summary.csv",
]


def preflight() -> dict[str, object]:
    """Check the packaged tables and all main and supplementary figure images."""
    missing_tables = [name for name in REQUIRED_TABLES if not (TABLES / name).is_file()]
    missing_main = [name for name in MAIN_FIGURES if not (FIGURES / name).is_file()]
    supplementary = FIGURES / "supplementary"
    missing_supplementary = [
        name for name in SUPPLEMENTARY_FIGURES if not (supplementary / name).is_file()
    ]
    return {
        "ready": not (missing_tables or missing_main or missing_supplementary),
        "tables": len(REQUIRED_TABLES) - len(missing_tables),
        "main_figures": len(MAIN_FIGURES) - len(missing_main),
        "supplementary_figures": len(SUPPLEMENTARY_FIGURES) - len(missing_supplementary),
        "missing": missing_tables + missing_main + missing_supplementary,
    }


def _metric_map() -> dict[str, int]:
    frame = pd.read_csv(TABLES / "cohort_summary.csv")
    return dict(zip(frame.metric, frame.value.astype(int)))


def _gene_row(gene: str) -> pd.Series:
    frame = pd.read_csv(TABLES / "gene_prevalence.csv")
    return frame.loc[frame.gene.eq(gene)].iloc[0]


def _pair_row(cancer: str, pair: str) -> pd.Series:
    frame = pd.read_csv(TABLES / "gene_pair_contexts.csv")
    return frame.loc[frame.cancer.eq(cancer) & frame.pair.eq(pair)].iloc[0]


def headline_results() -> dict[str, object]:
    """Return study-scale, association, functional, network and survival summaries."""
    metrics = _metric_map()
    tp53 = _gene_row("TP53")
    kras = _gene_row("KRAS")
    pik3ca = _gene_row("PIK3CA")
    egfr_kras = _pair_row("LUAD", "EGFR-KRAS")
    keap1_stk11 = _pair_row("LUAD", "KEAP1-STK11")

    functional = pd.read_csv(TABLES / "functional_crispr.csv")
    prism = pd.read_csv(TABLES / "functional_prism_selected.csv")
    functional_groups = pd.read_csv(TABLES / "functional_three_group_comparator.csv")
    functional_loo = pd.read_csv(
        TABLES / "functional_leave_one_lineage_out_summary.csv"
    )
    network = pd.read_csv(TABLES / "network_contexts.csv")
    network_null = pd.read_csv(TABLES / "interactome_degree_matched_null.csv")
    pair_screen = pd.read_csv(TABLES / "gene_pair_screen_summary.csv")
    off_panel = pd.read_csv(TABLES / "assay_discordance_by_study.csv")
    pathway_representation = pd.read_csv(TABLES / "pathway_gene_representation.csv")
    survival_cancer = pd.read_csv(TABLES / "survival_cancer_specific.csv")
    survival_pan = pd.read_csv(TABLES / "survival_pan_cancer.csv")
    survival_joint = pd.read_csv(TABLES / "survival_joint_state_interaction.csv")
    survival_piecewise = pd.read_csv(TABLES / "survival_piecewise_hazard_ratios.csv")
    survival_rmst = pd.read_csv(TABLES / "survival_rmst_differences.csv")

    luad_survival = survival_cancer.loc[
        survival_cancer.cancer.eq("LUAD")
        & survival_cancer.geneA.eq("KEAP1")
        & survival_cancer.geneB.eq("STK11")
    ].iloc[0]
    paad_survival = survival_cancer.loc[
        survival_cancer.cancer.eq("PAAD")
        & survival_cancer.geneA.eq("KRAS")
        & survival_cancer.geneB.eq("TP53")
    ].iloc[0]

    return {
        "cohort": {
            "selected_tissue_tumours": metrics["selected_tissue_tumours"],
            "contributing_studies": metrics["contributing_studies"],
            "reviewed_cancer_families": metrics["reviewed_cancer_families"],
            "detailed_oncotree_codes": metrics["detailed_oncotree_codes"],
            "genes": metrics["genes_in_compendium"],
            "unassayed_fraction_pct": round(
                100
                * metrics["unassayed_tumour_gene_observations"]
                / metrics["potential_tumour_gene_observations"],
                2,
            ),
        },
        "gene_prevalence_pct": {
            "TP53": round(float(tp53.freqPct), 2),
            "KRAS": round(float(kras.freqPct), 2),
            "PIK3CA": round(float(pik3ca.freqPct), 2),
        },
        "luad_conditioned_odds_ratios": {
            "primary_no_burden": {
                "EGFR-KRAS": float(egfr_kras.noBurden_full_or),
                "KEAP1-STK11": float(keap1_stk11.noBurden_full_or),
            },
            "leave_two_out_burden_sensitivity": {
                "EGFR-KRAS": float(egfr_kras.leaveTwoOut_full_or),
                "KEAP1-STK11": float(keap1_stk11.leaveTwoOut_full_or),
            },
            "historical_total_burden_diagnostic": {
                "EGFR-KRAS": float(egfr_kras.totalBurden_full_or),
                "KEAP1-STK11": float(keap1_stk11.totalBurden_full_or),
            },
        },
        "pairwise_screen": {
            row.criterion: int(row.nPairs)
            for row in pair_screen.itertuples(index=False)
        },
        "functional": {
            "displayed_crispr_genotypes": int(len(functional)),
            "crispr_fdr_supported": int(functional.adjustedFdr.lt(0.05).sum()),
            "selected_prism_contexts": int(len(prism)),
            "selected_prism_fdr_supported": int(prism.adjustedFdr.lt(0.05).sum()),
            "three_group_comparisons": int(len(functional_groups)),
            "leave_one_lineage_out_contexts": int(len(functional_loo)),
            "leave_one_lineage_out_direction_stable": int(
                functional_loo.allEffectsSameDirection.fillna(False).astype(bool).sum()
            ),
        },
        "network": {
            "cancer_gene_pair_contexts": int(len(network)),
            "unique_gene_pairs": int(network.pair.nunique()),
            "cancer_families": int(network.cancer.nunique()),
            "main_direct_connections": int(
                network_null.loc[
                    network_null.network.eq("main")
                    & network_null.metric.eq("direct connection"),
                    "observedConnectedEdges",
                ].iloc[0]
            ),
            "main_direct_connection_null_p": float(
                network_null.loc[
                    network_null.network.eq("main")
                    & network_null.metric.eq("direct connection"),
                    "empiricalEnrichmentP",
                ].iloc[0]
            ),
        },
        "survival": {
            "LUAD_KEAP1_STK11_joint_state_hazard_ratio": float(
                luad_survival.jointStateHazardRatio
            ),
            "PAAD_KRAS_TP53_joint_state_hazard_ratio": float(
                paad_survival.jointStateHazardRatio
            ),
            "pan_cancer_pairs": int(len(survival_pan)),
            "primary_joint_state_models": int(len(survival_joint)),
            "primary_joint_state_ph_p_below_0_05": int(
                survival_joint.jointStatePhTestP.lt(0.05).sum()
            ),
            "piecewise_estimates": int(len(survival_piecewise)),
            "rmst_estimates": int(len(survival_rmst)),
        },
        "assay_scope_audit": {
            "off_panel_mutation_records": int(
                off_panel.loc[off_panel.studyId.eq("ALL"), "nMutationRecords"].iloc[0]
            ),
            "affected_specimens": int(
                off_panel.loc[off_panel.studyId.eq("ALL"), "nAffectedSpecimens"].iloc[0]
            ),
        },
        "pathway_representation": {
            "pathways": int(len(pathway_representation)),
            "eligible_published_genes": int(
                pathway_representation.nEligiblePublishedGenes.sum()
            ),
            "represented_genes_summed_within_pathway": int(
                pathway_representation.nGenesRepresented.sum()
            ),
        },
    }


def main() -> None:
    checks = preflight()
    if not checks["ready"]:
        raise FileNotFoundError(f"Packaged result files are incomplete: {checks['missing']}")
    print(json.dumps({"preflight": checks, "results": headline_results()}, indent=2))


if __name__ == "__main__":
    main()
