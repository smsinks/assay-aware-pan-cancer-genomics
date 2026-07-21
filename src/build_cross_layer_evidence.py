"""Build a cross-layer evidence table for selected gene pairs.

The network, functional and survival analyses address complementary questions and use
different analytical populations.  This table does not pool those estimates.  Instead,
it places the evidence available for each displayed cancer--gene-pair context on one
row, preserving the original effect estimates and explicit missing states.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import RESULTS, TABLES


SOURCE = RESULTS / "source_data"
OUTPUT = TABLES / "cross_layer_evidence.csv"


def canonical_pair(gene_a: object, gene_b: object) -> str:
    genes = sorted((str(gene_a).strip().upper(), str(gene_b).strip().upper()))
    return "-".join(genes)


def load_display_contexts() -> pd.DataFrame:
    main = pd.read_csv(SOURCE / "figure8_main_edge_contexts.csv")
    expanded = pd.read_csv(SOURCE / "figureS3_expanded_display_contexts.csv")
    for frame in (main, expanded):
        frame["canonicalPair"] = [
            canonical_pair(a, b) for a, b in zip(frame.geneA, frame.geneB)
        ]
    main_keys = set(zip(main.cancer.astype(str), main.canonicalPair))
    expanded["inMainNetwork"] = [
        (str(cancer), pair) in main_keys
        for cancer, pair in zip(expanded.cancer, expanded.canonicalPair)
    ]
    # A main context must never disappear merely because an expanded-display threshold
    # changed.  Add any such context explicitly, then retain one row per cancer and pair.
    missing = main.loc[
        ~pd.MultiIndex.from_frame(main[["cancer", "canonicalPair"]]).isin(
            pd.MultiIndex.from_frame(expanded[["cancer", "canonicalPair"]])
        )
    ].copy()
    if not missing.empty:
        missing["inMainNetwork"] = True
        expanded = pd.concat([expanded, missing], ignore_index=True, sort=False)
    return expanded.drop_duplicates(["cancer", "canonicalPair"], keep="first")


def functional_summary() -> pd.DataFrame:
    crispr = pd.read_csv(TABLES / "depmap_addiction_lineage_adjusted.csv")
    crispr = crispr[
        [
            "gene",
            "adjustedEffect",
            "adjustedCiLow",
            "adjustedCiHigh",
            "adjustedFdr",
            "nModelsAdjusted",
            "nMutAdjusted",
            "nInformativeLineages",
        ]
    ].copy()
    crispr["crisprMutantSelectiveAssociation"] = (
        crispr.adjustedEffect.lt(0) & crispr.adjustedFdr.lt(0.05)
    )

    drug = pd.read_csv(TABLES / "depmap_drug_lineage_adjusted.csv")
    drug["significantSensitisation"] = (
        drug.adjustedFdr.lt(0.05) & drug.adjustedSensitisation.gt(0)
    )
    records: list[dict[str, object]] = []
    for gene, group in drug.groupby("gene", sort=False):
        significant = group.loc[group.significantSensitisation].copy()
        if significant.empty:
            best = None
        else:
            best = significant.sort_values(
                ["adjustedSensitisation", "adjustedFdr"], ascending=[False, True]
            ).iloc[0]
        records.append(
            {
                "gene": gene,
                "nSignificantSensitisingCompounds": int(len(significant)),
                "strongestSensitisingCompound": None if best is None else best.compound,
                "strongestAdjustedSensitisation": (
                    np.nan if best is None else best.adjustedSensitisation
                ),
                "strongestSensitisationFdr": np.nan if best is None else best.adjustedFdr,
            }
        )
    drug_summary = pd.DataFrame(records)
    return crispr.merge(drug_summary, on="gene", how="outer", validate="one_to_one")


def survival_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    models = pd.read_csv(TABLES / "survival_curated_pair_models.csv")
    primary = models.loc[
        models.model.eq("four-group primary")
        & models.term.eq("Both")
        & models["analysisPopulation"].eq("strictly positive OS time")
        & models.varianceEstimator.eq("model-based")
        & models.fitStatus.eq("estimated")
    ].copy()
    primary["canonicalPair"] = [
        canonical_pair(a, b) for a, b in zip(primary.geneA, primary.geneB)
    ]
    columns = [
        "canonicalPair",
        "cancer",
        "nPatients",
        "nEvents",
        "nStudies",
        "hazardRatio",
        "ciLow",
        "ciHigh",
        "p",
        "fdr",
        "phTestP",
    ]
    cancer_specific = primary.loc[primary.scope.eq("cancer-specific"), columns].copy()
    pan_cancer = primary.loc[primary.scope.eq("pan-cancer"), columns].copy()
    cancer_specific = cancer_specific.drop_duplicates(["cancer", "canonicalPair"])
    pan_cancer = pan_cancer.drop_duplicates(["canonicalPair"])
    return cancer_specific, pan_cancer


def attach_gene_layer(
    contexts: pd.DataFrame, functional: pd.DataFrame, gene_column: str, suffix: str
) -> pd.DataFrame:
    renamed = functional.rename(
        columns={
            column: f"{column}{suffix}"
            for column in functional.columns
            if column != "gene"
        }
    ).rename(columns={"gene": gene_column})
    return contexts.merge(renamed, on=gene_column, how="left", validate="many_to_one")


def main() -> None:
    contexts = load_display_contexts()
    contexts["canonicalPair"] = [
        canonical_pair(a, b) for a, b in zip(contexts.geneA, contexts.geneB)
    ]

    interactome = pd.read_csv(SOURCE / "figure8_interactome_edge_crosswalk.csv")
    interactome["canonicalPair"] = [
        canonical_pair(a, b) for a, b in zip(interactome.geneA, interactome.geneB)
    ]
    interactome_columns = [
        "canonicalPair",
        "interactomeDirect",
        "nInteractomeDirectRecords",
        "interactomeInteractionTypes",
        "interactomeDirectedRecords",
        "interactomePathLengthAtMost2",
        "nSharedInteractomeNeighbours",
        "sharedInteractomeNeighbours",
        "sharedNeighbourListTruncated",
    ]
    interactome = interactome[interactome_columns].drop_duplicates("canonicalPair")
    contexts = contexts.merge(interactome, on="canonicalPair", how="left", validate="many_to_one")

    functional = functional_summary()
    contexts = attach_gene_layer(contexts, functional, "geneA", "GeneA")
    contexts = attach_gene_layer(contexts, functional, "geneB", "GeneB")

    cancer_survival, pan_survival = survival_summary()
    cancer_survival = cancer_survival.rename(
        columns={
            column: f"survivalCancerSpecific{column[0].upper()}{column[1:]}"
            for column in cancer_survival.columns
            if column not in {"cancer", "canonicalPair"}
        }
    )
    contexts = contexts.merge(
        cancer_survival,
        on=["cancer", "canonicalPair"],
        how="left",
        validate="many_to_one",
    )
    pan_survival = pan_survival.drop(columns="cancer").rename(
        columns={
            column: f"survivalPanCancer{column[0].upper()}{column[1:]}"
            for column in pan_survival.columns
            if column != "canonicalPair"
        }
    )
    contexts = contexts.merge(
        pan_survival, on="canonicalPair", how="left", validate="many_to_one"
    )

    contexts["interactomeAnnotation"] = (
        contexts.interactomeDirect.fillna(False)
        | contexts.interactomePathLengthAtMost2.notna()
    )
    contexts["functionalAssociation"] = (
        contexts.crisprMutantSelectiveAssociationGeneA.eq(True)
        | contexts.crisprMutantSelectiveAssociationGeneB.eq(True)
        | contexts.nSignificantSensitisingCompoundsGeneA.fillna(0).gt(0)
        | contexts.nSignificantSensitisingCompoundsGeneB.fillna(0).gt(0)
    )
    contexts["survivalAnalysisAvailable"] = (
        contexts.survivalCancerSpecificHazardRatio.notna()
        | contexts.survivalPanCancerHazardRatio.notna()
    )
    contexts["nAnalyticalLayersWithData"] = (
        1
        + contexts.interactomeAnnotation.astype(int)
        + contexts.functionalAssociation.astype(int)
        + contexts.survivalAnalysisAvailable.astype(int)
    )
    contexts["analyticalLayersWithData"] = contexts.apply(
        lambda row: "; ".join(
            ["conditioned tumour association"]
            + (["interactome annotation"] if row.interactomeAnnotation else [])
            + (["cell-line functional association"] if row.functionalAssociation else [])
            + (["overall-survival analysis"] if row.survivalAnalysisAvailable else [])
        ),
        axis=1,
    )

    null = pd.read_csv(TABLES / "interactome_degree_matched_null.csv")
    null_p = null.pivot(index="network", columns="metric", values="empiricalEnrichmentP")
    contexts["directInteractomeEnrichmentP"] = np.where(
        contexts.inMainNetwork,
        float(null_p.loc["main", "direct connection"]),
        float(null_p.loc["expanded", "direct connection"]),
    )
    contexts["twoStepInteractomeEnrichmentP"] = np.where(
        contexts.inMainNetwork,
        float(null_p.loc["main", "connection within two steps"]),
        float(null_p.loc["expanded", "connection within two steps"]),
    )

    leading = [
        "cancer",
        "geneA",
        "geneB",
        "canonicalPair",
        "inMainNetwork",
        "direction",
        "noBurden_full_or",
        "noBurden_full_ciLow",
        "noBurden_full_ciHigh",
        "noBurden_full_p",
        "noBurden_full_fdr",
        "noBurdenCrossAssayConcordant",
        "leaveTwoOut_full_or",
        "leaveTwoOut_full_ciLow",
        "leaveTwoOut_full_ciHigh",
        "leaveTwoOut_full_p",
        "leaveTwoOut_full_fdr",
        "leaveTwoOutCrossAssayConcordant",
        "signStableNoBurdenLeaveTwoOut",
        "effectStableNoBurdenLeaveTwoOut",
        "interactomeAnnotation",
        "interactomeDirect",
        "interactomePathLengthAtMost2",
        "directInteractomeEnrichmentP",
        "twoStepInteractomeEnrichmentP",
        "functionalAssociation",
        "survivalAnalysisAvailable",
        "nAnalyticalLayersWithData",
        "analyticalLayersWithData",
    ]
    remaining = [column for column in contexts.columns if column not in leading]
    contexts = contexts[leading + remaining].sort_values(
        ["inMainNetwork", "cancer", "canonicalPair"],
        ascending=[False, True, True],
    )
    contexts.to_csv(OUTPUT, index=False)

    if contexts.duplicated(["cancer", "canonicalPair"]).any():
        raise AssertionError("Cross-layer table contains duplicate cancer--pair contexts")
    if not contexts.inMainNetwork.any():
        raise AssertionError("Cross-layer table contains no main-network contexts")
    print(
        f"Wrote {len(contexts):,} displayed cancer--pair contexts to {OUTPUT}; "
        f"{int(contexts.inMainNetwork.sum()):,} are in the main network."
    )


if __name__ == "__main__":
    main()
