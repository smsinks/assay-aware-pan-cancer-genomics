"""Metadata sensitivities for assay discordance and legacy GBM classification.

This stage does not promote observed off-panel mutations into assay-covered positive
calls.  It preserves the strict documented-panel analysis and evaluates whether removing
the affected study or every specimen with an assay-metadata conflict changes gene-level
prevalence.  It also documents IDH1/ATRX alterations in portal-defined or legacy GBM so
that those cases are not described as a molecularly verified IDH-wildtype cohort.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from callability import partition_callable_mutations
from config import PROCESSED, TABLES


AFFECTED_STUDY = "sarcoma_msk_2022"


def _wilson(successes: np.ndarray, totals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    successes = successes.astype(float)
    totals = totals.astype(float)
    low = np.full(len(successes), np.nan)
    high = np.full(len(successes), np.nan)
    valid = totals > 0
    z = 1.959963984540054
    p = successes[valid] / totals[valid]
    denominator = 1 + z * z / totals[valid]
    centre = (p + z * z / (2 * totals[valid])) / denominator
    half = z * np.sqrt(
        p * (1 - p) / totals[valid] + z * z / (4 * totals[valid] ** 2)
    ) / denominator
    low[valid] = 100 * np.clip(centre - half, 0, 1)
    high[valid] = 100 * np.clip(centre + half, 0, 1)
    return low, high


def load_selected() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    samples = pd.read_parquet(PROCESSED / "analysis_samples_curated.parquet")
    samples = samples.loc[samples.analysisEligible].copy()
    assay = pd.read_parquet(PROCESSED / "sample_assay.parquet")
    samples = samples.merge(
        assay[["sampleId", "studyId", "assayType", "genePanelId"]],
        on=["sampleId", "studyId"],
        how="left",
        validate="one_to_one",
    )
    panel = pd.read_csv(PROCESSED / "gene_panel.csv")
    panel["entrezGeneId"] = panel.entrezGeneId.astype(int)
    membership = pd.read_parquet(PROCESSED / "panel_gene_membership.parquet")
    membership = membership[["genePanelId", "entrezGeneId"]].drop_duplicates()
    membership["entrezGeneId"] = membership.entrezGeneId.astype(int)
    mutations = pd.read_parquet(PROCESSED / "mutations_curated.parquet")
    mutations["entrezGeneId"] = mutations.entrezGeneId.astype(int)

    raw = pd.read_parquet(PROCESSED / "mutations_dedup.parquet")
    _, conflicts = partition_callable_mutations(raw)
    conflicts = conflicts[conflicts.sampleId.isin(set(samples.sampleId))].copy()
    return samples, panel, membership, mutations, conflicts


def assay_discordance_summary(conflicts: pd.DataFrame) -> pd.DataFrame:
    conflicts = conflicts.copy()
    conflicts["_sampleGeneKey"] = (
        conflicts.sampleId.astype(str) + "::" + conflicts.entrezGeneId.astype(str)
    )
    detail = (
        conflicts.groupby(["studyId", "genePanelId"], dropna=False)
        .agg(
            nMutationRecords=("sampleId", "size"),
            nAffectedSpecimens=("sampleId", "nunique"),
            nAffectedSampleGenePairs=("_sampleGeneKey", "nunique"),
            nAffectedGenes=("entrezGeneId", "nunique"),
        )
        .reset_index()
        .sort_values("nMutationRecords", ascending=False)
    )
    total_records = int(len(conflicts))
    detail["recordSharePct"] = 100 * detail.nMutationRecords / total_records
    detail["classification"] = (
        "assay-uncertain positive; excluded from mutation numerators and prevalence denominators"
    )
    overall = pd.DataFrame(
        [
            {
                "studyId": "ALL",
                "genePanelId": "ALL TARGETED PANELS",
                "nMutationRecords": total_records,
                "nAffectedSpecimens": conflicts.sampleId.nunique(),
                "nAffectedSampleGenePairs": len(
                    conflicts.drop_duplicates(["sampleId", "entrezGeneId"])
                ),
                "nAffectedGenes": conflicts.entrezGeneId.nunique(),
                "recordSharePct": 100.0,
                "classification": (
                    "assay-uncertain positive; excluded from mutation numerators and prevalence denominators"
                ),
            }
        ]
    )
    output = pd.concat([overall, detail], ignore_index=True)
    output.to_csv(TABLES / "assay_discordance_study_sensitivity.csv", index=False)
    return output


def prevalence_sensitivity(
    samples: pd.DataFrame,
    panel: pd.DataFrame,
    membership: pd.DataFrame,
    mutations: pd.DataFrame,
    conflicts: pd.DataFrame,
) -> pd.DataFrame:
    affected_specimens = set(conflicts.sampleId)
    conditions = {
        "strict documented assay scope": np.ones(len(samples), dtype=bool),
        f"exclude affected study ({AFFECTED_STUDY})": ~samples.studyId.eq(AFFECTED_STUDY),
        "exclude every assay-discordant specimen": ~samples.sampleId.isin(affected_specimens),
    }
    rows: list[pd.DataFrame] = []
    gene_columns = [
        "entrezGeneId", "hugoSymbol", "roleInCancer", "cosmicTier", "highConfidence"
    ]
    for condition, mask in conditions.items():
        cohort = samples.loc[mask].copy()
        n_wes = int(cohort.assayType.eq("WES/WGS").sum())
        panel_counts = (
            cohort.loc[cohort.assayType.eq("Targeted panel"), "genePanelId"]
            .value_counts()
            .rename("nPanelSamples")
            .reset_index()
        )
        targeted = (
            panel_counts.merge(membership, on="genePanelId", how="inner")
            .groupby("entrezGeneId").nPanelSamples.sum()
            .rename("nCallableTargeted")
        )
        unique_mutations = mutations[
            mutations.sampleId.isin(set(cohort.sampleId))
        ].drop_duplicates(["sampleId", "entrezGeneId"])
        mutated = unique_mutations.groupby("entrezGeneId").sampleId.nunique().rename("nMutated")

        result = panel[gene_columns].drop_duplicates("entrezGeneId").copy()
        result = result.merge(targeted, on="entrezGeneId", how="left")
        result = result.merge(mutated, on="entrezGeneId", how="left")
        result[["nCallableTargeted", "nMutated"]] = result[
            ["nCallableTargeted", "nMutated"]
        ].fillna(0).astype(int)
        result["nCallableWesWgs"] = n_wes
        result["nCallable"] = result.nCallableWesWgs + result.nCallableTargeted
        if (result.nMutated > result.nCallable).any():
            raise AssertionError(f"Mutation numerator exceeds denominator in {condition}")
        result["prevalencePct"] = 100 * result.nMutated / result.nCallable.replace(0, np.nan)
        low, high = _wilson(result.nMutated.to_numpy(), result.nCallable.to_numpy())
        result["prevalenceCiLowPct"] = low
        result["prevalenceCiHighPct"] = high
        result["sensitivityCondition"] = condition
        result["nCohortSamples"] = len(cohort)
        result["nCohortStudies"] = cohort.studyId.nunique()
        rows.append(result)

    output = pd.concat(rows, ignore_index=True)
    strict = (
        output[output.sensitivityCondition.eq("strict documented assay scope")]
        .set_index("entrezGeneId")
    )
    output["deltaPrevalencePctVsStrict"] = output.prevalencePct - output.entrezGeneId.map(
        strict.prevalencePct
    )
    output["relativePrevalenceRatioVsStrict"] = output.prevalencePct / output.entrezGeneId.map(
        strict.prevalencePct.replace(0, np.nan)
    )
    output.to_csv(TABLES / "offpanel_prevalence_sensitivity.csv", index=False)
    return output


def gbm_legacy_audit(
    samples: pd.DataFrame,
    panel: pd.DataFrame,
    mutations: pd.DataFrame,
) -> pd.DataFrame:
    gbm = samples[samples.cancerFamilyCode.eq("GBM")].copy()
    gene_ids = panel.set_index("hugoSymbol").entrezGeneId.to_dict()
    mutation_subset = mutations[mutations.sampleId.isin(set(gbm.sampleId))].copy()
    flags: dict[str, set[str]] = {}
    for gene in ("IDH1", "IDH2", "ATRX", "TP53"):
        flags[gene] = set(
            mutation_subset.loc[
                mutation_subset.entrezGeneId.eq(int(gene_ids[gene])), "sampleId"
            ]
        )
        gbm[gene] = gbm.sampleId.isin(flags[gene])
    idh1 = mutation_subset[mutation_subset.entrezGeneId.eq(int(gene_ids["IDH1"]))]
    hotspot_ids = set(
        idh1.loc[idh1.proteinChange.astype(str).str.match(r"R132[ACDEGHKLMPQSTVY]$"), "sampleId"]
    )
    gbm["IDH1R132"] = gbm.sampleId.isin(hotspot_ids)
    gbm["IDH1Other"] = gbm.IDH1 & ~gbm.IDH1R132
    gbm["IDH1AndTP53"] = gbm.IDH1 & gbm.TP53
    gbm["ATRXAndIDH1"] = gbm.ATRX & gbm.IDH1

    audit = (
        gbm.groupby("studyId")
        .agg(
            nPortalDefinedOrLegacyGbm=("sampleId", "size"),
            nIdh1Mutant=("IDH1", "sum"),
            nIdh1R132=("IDH1R132", "sum"),
            nOtherIdh1=("IDH1Other", "sum"),
            nIdh2Mutant=("IDH2", "sum"),
            nAtrxMutant=("ATRX", "sum"),
            nTp53Mutant=("TP53", "sum"),
            nIdh1Tp53Joint=("IDH1AndTP53", "sum"),
            nAtrxIdh1Joint=("ATRXAndIDH1", "sum"),
        )
        .reset_index()
        .sort_values("nPortalDefinedOrLegacyGbm", ascending=False)
    )
    overall = pd.DataFrame(
        [
            {
                "studyId": "ALL",
                "nPortalDefinedOrLegacyGbm": len(gbm),
                "nIdh1Mutant": int(gbm.IDH1.sum()),
                "nIdh1R132": int(gbm.IDH1R132.sum()),
                "nOtherIdh1": int(gbm.IDH1Other.sum()),
                "nIdh2Mutant": int(gbm.IDH2.sum()),
                "nAtrxMutant": int(gbm.ATRX.sum()),
                "nTp53Mutant": int(gbm.TP53.sum()),
                "nIdh1Tp53Joint": int(gbm.IDH1AndTP53.sum()),
                "nAtrxIdh1Joint": int(gbm.ATRXAndIDH1.sum()),
            }
        ]
    )
    output = pd.concat([overall, audit], ignore_index=True)
    output["displayClassification"] = "Portal-defined or legacy glioblastoma"
    output["interpretation"] = (
        "Historical portal GBM releases include IDH1-mutant tumours; IDH-associated results are classification-sensitive"
    )
    output.to_csv(TABLES / "gbm_legacy_classification_audit.csv", index=False)
    return output


def main() -> None:
    samples, panel, membership, mutations, conflicts = load_selected()
    discordance = assay_discordance_summary(conflicts)
    sensitivity = prevalence_sensitivity(samples, panel, membership, mutations, conflicts)
    gbm = gbm_legacy_audit(samples, panel, mutations)
    sarcoma_records = int(
        discordance.loc[discordance.studyId.eq(AFFECTED_STUDY), "nMutationRecords"].sum()
    )
    print(
        f"Assay-discordant positives: {len(conflicts):,} records in "
        f"{conflicts.sampleId.nunique():,} specimens; {sarcoma_records:,} records in {AFFECTED_STUDY}"
    )
    print(
        f"Prevalence sensitivity: {len(sensitivity):,} gene-condition rows; "
        f"legacy GBM audit: {int(gbm.iloc[0].nPortalDefinedOrLegacyGbm):,} tumours"
    )


if __name__ == "__main__":
    main()
