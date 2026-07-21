"""Stage 4 — define assay-aware gene callability and mutation prevalence.

For each gene, the denominator is the number of deduplicated samples in which that gene
was profiled: whole-exome/genome samples plus targeted-panel samples whose documented
panel includes the gene. Genes outside a targeted panel remain unassayed.

Per-sample panel assignment and panel gene lists come from the cBioPortal API (cached).

Outputs:
  data/processed/gene_profiled_counts.parquet   per-gene profiled-sample denominator
  data/processed/sample_assay.parquet           per-sample assay/panel assignment
  data/processed/panel_gene_membership.parquet  targeted-panel gene membership
  results/tables/gene_frequencies_panel_aware.csv
  results/tables/assay_summary.csv
"""
from __future__ import annotations

import pandas as pd
from tqdm import tqdm

import cbioportal_client as cb
from callability import partition_callable_mutations
from config import PROCESSED, TABLES


def mutation_profile_id(study_id: str) -> str | None:
    """The MUTATION_EXTENDED molecular profile id for a study, if any."""
    for p in cb.get_molecular_profiles(study_id):
        if p.get("molecularAlterationType") == "MUTATION_EXTENDED":
            return p["molecularProfileId"]
    return None


def main() -> None:
    canon = pd.read_parquet(PROCESSED / "canonical_samples.parquet")
    panel = pd.read_csv(PROCESSED / "gene_panel.csv")
    our_entrez = set(panel["entrezGeneId"])

    # 1. Per canonical sample, which gene panel was it profiled with?  API responses use
    # source identifiers, whereas the analysis identifier may be study-qualified after
    # the specimen-identity audit.
    source_to_analysis = {
        (row.studyId, row.sourceSampleId): row.sampleId
        for row in canon.itertuples(index=False)
    }
    canonical_analysis_keys = set(zip(canon["sampleId"], canon["studyId"]))
    sample_panel = {}            # (analysisSampleId, studyId) -> genePanelId or None
    panels_seen: set[str] = set()
    for sid in tqdm(sorted(canon["studyId"].unique()), desc="panel data"):
        prof = mutation_profile_id(sid)
        if not prof:
            continue
        try:
            rows = cb.get_gene_panel_data(prof, f"{sid}_all")
        except RuntimeError:
            continue
        for r in rows:
            analysis_id = source_to_analysis.get((sid, r["sampleId"]))
            if analysis_id is None:
                continue
            key = (analysis_id, sid)
            pid = r.get("genePanelId")  # absent/None => WES/WGS
            sample_panel[key] = pid
            if pid and pid != "WES" and pid != "WGS":
                panels_seen.add(pid)

    # 2. Gene set covered by each targeted panel (restricted to our compendium).
    panel_genes: dict[str, set[int]] = {}
    for pid in tqdm(sorted(panels_seen), desc="panel gene lists"):
        try:
            data = cb.get_gene_panel(pid)
        except RuntimeError:
            continue
        panel_genes[pid] = {g["entrezGeneId"] for g in data.get("genes", [])} & our_entrez

    # Persist the assay assignment rather than leaving it as an in-memory intermediate.
    # This is required for assay-stratified sensitivity analyses and makes the important
    # "unassigned means presumed WES/WGS" assumption visible and countable.
    assay_rows = []
    mutation_bearing = set(
        pd.read_parquet(
            PROCESSED / "mutations_dedup.parquet", columns=["studyId", "sampleId"]
        ).itertuples(index=False, name=None)
    )
    source_by_analysis = canon.set_index(["studyId", "sampleId"])["sourceSampleId"].to_dict()
    for r in canon[["sampleId", "studyId"]].itertuples(index=False):
        key = (r.sampleId, r.studyId)
        if key not in sample_panel:
            pid = None
            assay = "Unverified mutation-profile membership"
            metadata_available = False
            membership_status = (
                "mutation record confirms profile membership; assay scope unavailable"
                if (r.studyId, r.sampleId) in mutation_bearing
                else "profile membership unverified"
            )
            callability_eligible = False
        else:
            pid = sample_panel[key]
            if pid and pid not in ("WES", "WGS"):
                assay = "Targeted panel"
            else:
                assay = "WES/WGS"
            metadata_available = True
            membership_status = "confirmed by molecular-profile gene-panel record"
            callability_eligible = True
        assay_rows.append(
            (
                r.sampleId,
                source_by_analysis.get((r.studyId, r.sampleId)),
                r.studyId,
                pid,
                assay,
                metadata_available,
                membership_status,
                callability_eligible,
            )
        )
    assay_df = pd.DataFrame(
        assay_rows,
        columns=[
            "sampleId",
            "sourceSampleId",
            "studyId",
            "genePanelId",
            "assayType",
            "panelMetadataAvailable",
            "mutationProfileMembershipStatus",
            "callabilityEligible",
        ],
    )
    assay_df.to_parquet(PROCESSED / "sample_assay.parquet", index=False)

    membership_rows = [
        (pid, ent)
        for pid, genes in panel_genes.items()
        for ent in sorted(genes)
    ]
    pd.DataFrame(membership_rows, columns=["genePanelId", "entrezGeneId"]).to_parquet(
        PROCESSED / "panel_gene_membership.parquet", index=False
    )

    assay_summary = (
        assay_df.groupby(["assayType", "panelMetadataAvailable"], dropna=False)
        .size()
        .rename("nSamples")
        .reset_index()
    )
    assay_summary["pctSamples"] = 100 * assay_summary["nSamples"] / len(assay_df)
    assay_summary.to_csv(TABLES / "assay_summary.csv", index=False)

    # 3. Per-gene profiled-sample denominator.
    #    WES/None samples profile every gene; panel samples profile only their panel genes.
    assigned = pd.Series(list(sample_panel.values()))
    is_wes = assigned.isna() | assigned.isin(["WES", "WGS"])
    n_wes = int(is_wes.sum())
    n_unassigned = len(canonical_analysis_keys) - len(sample_panel)
    # Samples without a molecular-profile gene-panel record are not silently treated as
    # whole-exome.  They remain in the specimen audit but contribute to no callable
    # denominator until profile membership and assay scope are confirmed.
    wes_base = n_wes

    panel_counts = pd.Series([p for p in sample_panel.values()
                              if p and p not in ("WES", "WGS")]).value_counts()
    rows = []
    for ent in our_entrez:
        on_panels = sum(int(cnt) for pid, cnt in panel_counts.items()
                        if ent in panel_genes.get(pid, set()))
        rows.append((ent, wes_base + on_panels))
    prof_df = pd.DataFrame(rows, columns=["entrezGeneId", "nProfiled"])
    prof_df.to_parquet(PROCESSED / "gene_profiled_counts.parquet", index=False)

    # 4. Numerator: de-duplicated mutated-sample counts.
    dedup = pd.read_parquet(
        PROCESSED / "mutations_dedup.parquet",
        columns=["studyId", "sampleId", "entrezGeneId"],
    )
    dedup, conflicts = partition_callable_mutations(dedup, assay_df)
    num = (dedup.drop_duplicates(["sampleId", "entrezGeneId"])
                .groupby("entrezGeneId")["sampleId"].nunique().rename("nMutated"))
    n_total = len(canon)

    out = prof_df.merge(num, on="entrezGeneId", how="left").fillna({"nMutated": 0})
    out["freq_cohort_wide_pct"] = 100 * out["nMutated"] / n_total
    out["freq_panel_aware_pct"] = 100 * out["nMutated"] / out["nProfiled"].clip(lower=1)
    out["gene"] = out["entrezGeneId"].map(panel.set_index("entrezGeneId")["hugoSymbol"])
    out["pct_cohort_profiling_gene"] = 100 * out["nProfiled"] / n_total
    out = out.sort_values("freq_panel_aware_pct", ascending=False)
    if (out.nMutated > out.nProfiled).any():
        raise AssertionError("A mutation numerator exceeds its documented callable denominator")
    out.to_csv(TABLES / "gene_frequencies_panel_aware.csv", index=False)

    n_targeted = int((assay_df.assayType == "Targeted panel").sum())
    print(f"Canonical samples: {n_total:,} | whole-exome/genome (all-gene): {wes_base:,} "
          f"| targeted-panel: {n_targeted:,} | unverified profile membership: {n_unassigned:,}")
    print(f"Distinct panels used: {len(panel_genes)}")
    print(
        "Targeted mutation rows outside documented panel membership: "
        f"{len(conflicts):,} (excluded from frequency numerators)"
    )
    print("\nAssay assignment:")
    print(assay_summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\nTop 15 genes — cohort-wide vs assay-aware frequency:")
    show = out.head(15)[
        ["gene", "nMutated", "nProfiled", "freq_cohort_wide_pct", "freq_panel_aware_pct"]
    ]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\nWrote {TABLES / 'gene_frequencies_panel_aware.csv'}")


if __name__ == "__main__":
    main()
