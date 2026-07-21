"""Stage 1b — build an evidence-aware canonical specimen set across the cohort.

cBioPortal exposes the same physical tumour many times: every TCGA cancer appears in an
earlier Firehose release (``brca_tcga``), a GDC re-call (``brca_tcga_gdc``), one or more
publication freezes (``brca_tcga_pub*``) and the harmonised PanCancer Atlas
(``brca_tcga_pan_can_atlas_2018``) — all sharing the same TCGA barcodes. Large aggregator
cohorts (GENIE, MSK-CHORD/MetTropism) likewise re-include samples from primary panel
studies. Counting those barcodes once per study inflates every frequency and corrupts the
co-occurrence denominator.

Bare sample identifiers are not globally unique.  This stage therefore treats identifiers
as study-qualified by default.  Cross-study de-duplication is allowed only when an identifier
has a recognised global namespace (TCGA, MSK ``P-`` or GENIE) or when the study pair appears
in the frozen, evidence-reviewed overlap whitelist.  Unreviewed generic collisions remain
separate specimens.  The kept study within each supported identity component determines
which mutation calls are used for that physical specimen.

Priority (higher kept):
  TCGA PanCancer Atlas (harmonised)          100
  TCGA GDC re-call                            70
  TCGA publication freeze                      60
  Earlier TCGA Firehose release (bare _tcga)   50
  ordinary primary study                       30
  pan-cohort aggregator (GENIE / MSK meta)     10   (re-includes other studies' samples)
Ties broken by shortest studyId then alphabetical (favours the specific cohort).

Outputs:
  data/processed/canonical_samples.parquet   analysis ID -> source ID and kept study
  data/processed/mutations_dedup.parquet      mutations_long restricted to kept (sample,study)
  results/tables/sample_identity_reconciliation.csv  row-level identity mapping
  results/tables/specimen_identity_summary.csv       rule-level audit counts
  results/tables/dedup_impact.csv             before/after per-gene frequency
"""
from __future__ import annotations

import re
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

import cbioportal_client as cb
from config import EXTERNAL, PROCESSED, TABLES, study_priority


GLOBAL_SAMPLE_ID = re.compile(r"^(TCGA-|P-\d+|GENIE-)", flags=re.IGNORECASE)
OVERLAP_WHITELIST = (
    EXTERNAL / "references" / "cross_study_sample_overlap_whitelist_2026-07-18.csv"
)


class UnionFind:
    """Small deterministic union-find used within one colliding bare identifier."""

    def __init__(self, values: list[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        # Lexical rooting makes the identity key reproducible across Python versions.
        lo, hi = sorted((a, b))
        self.parent[hi] = lo


def load_overlap_whitelist() -> set[frozenset[str]]:
    if not OVERLAP_WHITELIST.exists():
        raise FileNotFoundError(
            "The reviewed generic-ID overlap whitelist is required: "
            f"{OVERLAP_WHITELIST}"
        )
    frame = pd.read_csv(OVERLAP_WHITELIST)
    required = {"studyIdA", "studyIdB", "reviewStatus"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Malformed overlap whitelist; required columns are {sorted(required)}")
    frame = frame.loc[frame.reviewStatus.astype(str).str.lower().eq("approved")]
    return {
        frozenset((str(row.studyIdA), str(row.studyIdB)))
        for row in frame.itertuples(index=False)
    }


def assign_specimen_identities(
    all_samples: pd.DataFrame, whitelist: set[frozenset[str]]
) -> pd.DataFrame:
    """Assign an evidence-supported physical-specimen identity to every source row."""
    blocks: list[pd.DataFrame] = []
    for source_id, group in all_samples.groupby("sourceSampleId", sort=False):
        group = group.copy()
        studies = sorted(group.studyId.astype(str).unique())
        if len(studies) == 1:
            group["specimenIdentityKey"] = f"STUDY::{studies[0]}::{source_id}"
            group["specimenIdentityRule"] = "unique study-qualified identifier"
        elif GLOBAL_SAMPLE_ID.match(str(source_id)):
            group["specimenIdentityKey"] = f"GLOBAL::{str(source_id).upper()}"
            group["specimenIdentityRule"] = "recognised global identifier"
        else:
            union = UnionFind(studies)
            for i, left in enumerate(studies):
                for right in studies[i + 1 :]:
                    if frozenset((left, right)) in whitelist:
                        union.union(left, right)
            components: dict[str, list[str]] = defaultdict(list)
            for study_id in studies:
                components[union.find(study_id)].append(study_id)
            component_by_study = {
                study_id: "|".join(sorted(component))
                for component in components.values()
                for study_id in component
            }
            group["_component"] = group.studyId.map(component_by_study)
            group["specimenIdentityKey"] = [
                f"REVIEWED::{component}::{source_id}"
                if "|" in component
                else f"STUDY::{component}::{source_id}"
                for component in group._component
            ]
            group["specimenIdentityRule"] = [
                "reviewed cross-study overlap"
                if "|" in component
                else "unresolved generic collision preserved study-qualified"
                for component in group._component
            ]
            group = group.drop(columns="_component")
        blocks.append(group)
    return pd.concat(blocks, ignore_index=True)


def analysis_ids_for_winners(winners: pd.DataFrame) -> pd.Series:
    """Return unique analysis identifiers while preserving familiar global barcodes."""
    duplicated_source = winners.sourceSampleId.duplicated(keep=False)
    ids = winners.sourceSampleId.astype(str).copy()
    ids.loc[duplicated_source] = (
        winners.loc[duplicated_source, "studyId"].astype(str)
        + "::"
        + winners.loc[duplicated_source, "sourceSampleId"].astype(str)
    )
    if ids.duplicated().any():
        raise AssertionError("Analysis sample identifiers are not unique after qualification")
    return ids


def main() -> None:
    cohort = pd.read_csv(PROCESSED / "cohort_studies.csv")
    ct_map = cohort.set_index("studyId")["cancerTypeId"].to_dict()

    rows: list[tuple[str, str]] = []
    for sid in tqdm(cohort["studyId"], desc="sample lists"):
        try:
            for s in cb.get_sample_ids(sid):
                rows.append((s, sid))
        except RuntimeError:
            continue
    allsamp = pd.DataFrame(rows, columns=["sourceSampleId", "studyId"])
    allsamp["priority"] = allsamp["studyId"].map(study_priority)
    allsamp["cancerTypeId"] = allsamp["studyId"].map(ct_map)
    allsamp["_tie"] = allsamp["studyId"].str.len()

    raw_n = len(allsamp)
    allsamp = assign_specimen_identities(allsamp, load_overlap_whitelist())
    allsamp = allsamp.sort_values(
        ["specimenIdentityKey", "priority", "_tie", "studyId"],
        ascending=[True, False, True, True],
    )
    allsamp["kept"] = ~allsamp.duplicated("specimenIdentityKey", keep="first")
    component_sizes = allsamp.groupby("specimenIdentityKey").size()
    allsamp["nStudyRowsInIdentity"] = allsamp.specimenIdentityKey.map(component_sizes)
    winners = allsamp.loc[allsamp.kept].copy()
    winners["sampleId"] = analysis_ids_for_winners(winners)
    winner_lookup = winners.set_index("specimenIdentityKey")
    allsamp["keptStudyId"] = allsamp.specimenIdentityKey.map(winner_lookup.studyId)
    allsamp["analysisSampleId"] = allsamp.specimenIdentityKey.map(winner_lookup.sampleId)
    canon = winners[
        [
            "sampleId",
            "sourceSampleId",
            "studyId",
            "priority",
            "cancerTypeId",
            "specimenIdentityKey",
            "specimenIdentityRule",
            "nStudyRowsInIdentity",
        ]
    ].copy()
    canon.to_parquet(PROCESSED / "canonical_samples.parquet", index=False)

    allsamp.drop(columns="_tie").to_csv(
        TABLES / "sample_identity_reconciliation.csv", index=False
    )
    identity_summary = (
        allsamp.groupby("specimenIdentityRule", dropna=False)
        .agg(
            nSourceStudyRows=("sourceSampleId", "size"),
            nPhysicalSpecimens=("specimenIdentityKey", "nunique"),
            nRowsKept=("kept", "sum"),
            nRowsRemoved=("kept", lambda values: int((~values).sum())),
        )
        .reset_index()
    )
    identity_summary.to_csv(TABLES / "specimen_identity_summary.csv", index=False)

    print(f"Total (sample,study) rows across cohort : {raw_n:,}")
    print(f"Distinct physical samples (canonical)   : {len(canon):,}")
    print(f"Redundant rows removed                  : {raw_n - len(canon):,} "
          f"({100*(raw_n-len(canon))/raw_n:.1f}%)")

    # Restrict mutations to the kept source (sample, study) pairs and replace the raw
    # identifier with the unique analysis identifier.  Preserve the raw value explicitly.
    ml = pd.read_parquet(PROCESSED / "mutations_long.parquet")
    if "sourceSampleId" in ml.columns:
        ml = ml.drop(columns="sourceSampleId")
    mapping = canon[["studyId", "sourceSampleId", "sampleId"]].rename(
        columns={"sampleId": "analysisSampleId"}
    )
    dedup = (
        ml.rename(columns={"sampleId": "sourceSampleId"})
        .merge(mapping, on=["studyId", "sourceSampleId"], how="inner", validate="many_to_one")
        .rename(columns={"analysisSampleId": "sampleId"})
    )
    dedup.to_parquet(PROCESSED / "mutations_dedup.parquet", index=False)
    print(f"Mutation rows: {len(ml):,} -> {len(dedup):,} after dedup")

    # Before/after pan-cancer frequency for the top genes.
    panel = pd.read_csv(PROCESSED / "gene_panel.csv").set_index("entrezGeneId")["hugoSymbol"]
    n_before = raw_n                       # denominator before specimen reconciliation
    n_after = len(canon)
    def freq(df, n):
        if "studyId" in df.columns and df is ml:
            unique = df.drop_duplicates(["studyId", "sampleId", "entrezGeneId"])
            counts = unique.groupby("entrezGeneId").size()
        else:
            counts = (
                df.drop_duplicates(["sampleId", "entrezGeneId"])
                .groupby("entrezGeneId")["sampleId"]
                .nunique()
            )
        return counts / n * 100
    fb = freq(ml, n_before).rename("freq_before_pct")
    fa = freq(dedup, n_after).rename("freq_after_pct")
    comp = pd.concat([fb, fa], axis=1).dropna()
    comp["gene"] = comp.index.map(panel)
    comp["delta_pct"] = comp["freq_after_pct"] - comp["freq_before_pct"]
    comp = comp.sort_values("freq_after_pct", ascending=False)
    comp.to_csv(TABLES / "dedup_impact.csv")

    print(f"\nDenominator: {n_before:,} (raw) -> {n_after:,} (deduplicated)")
    print("\nTop 12 genes — frequency before vs after dedup:")
    print(comp.head(12)[["gene", "freq_before_pct", "freq_after_pct", "delta_pct"]]
          .to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
