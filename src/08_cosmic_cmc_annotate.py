"""Stage 8 — map observed mutations to COSMIC Cancer Mutation Census evidence.

The gene-level Cancer Gene Census says *which genes* are drivers; the CMC classifies
*individual mutations*. Joining it to our observed mutations lets us separate driver from
passenger events at amino-acid resolution and report, per gene, what fraction of the
mutations we see are COSMIC-significant.

Key CMC fields used:
  MUTATION_SIGNIFICANCE_TIER  COSMIC tier 1/2/3 ("Other" = not significant)
  DNDS_DISEASE_QVAL_SIG       dN/dS positive-selection significance
  CLINVAR_CLNSIG              ClinVar clinical significance
  GNOMAD_GENOMES_AF           population AF (to flag likely-germline common variants)

Step 1 streams the (large, gzipped) CMC TSV once, keeps only our panel genes and the
columns above, and writes a compact lookup parquet. Step 2 joins it to mutations_dedup.

Inputs (user-supplied COSMIC v104 download):
  data/external/cosmic/CancerMutationCensus_AllData_Tsv_v104_GRCh37.tar
Outputs:
  data/external/cosmic_cmc_lookup.parquet
  results/tables/mutation_tier_by_gene.csv
"""
from __future__ import annotations

import gzip
import subprocess
import tarfile

import pandas as pd

from config import EXTERNAL, PROCESSED, TABLES

CMC_TAR = EXTERNAL / "cosmic" / "CancerMutationCensus_AllData_Tsv_v104_GRCh37.tar"
LOOKUP = EXTERNAL / "cosmic_cmc_lookup.parquet"
KEEP = ["GENE_NAME", "Mutation AA", "MUTATION_SIGNIFICANCE_TIER",
        "DNDS_DISEASE_QVAL_SIG", "CLINVAR_CLNSIG", "GNOMAD_GENOMES_AF"]


def norm_aa(aa: str) -> str:
    """'p.R175H' / 'R175H' -> 'R175H' for matching against cBioPortal proteinChange."""
    if not isinstance(aa, str):
        return ""
    return aa[2:] if aa.startswith("p.") else aa


def build_lookup(our_symbols: set[str]) -> pd.DataFrame:
    """Stream the gzipped CMC TSV, keep our genes + key columns, cache to parquet."""
    if LOOKUP.exists():
        return pd.read_parquet(LOOKUP)
    member = "CancerMutationCensus_AllData_v104_GRCh37.tsv.gz"
    with tarfile.open(CMC_TAR) as tar:
        fh = tar.extractfile(member)
        chunks = []
        with gzip.open(fh, "rt") as gz:
            for chunk in pd.read_csv(gz, sep="\t", usecols=lambda c: c in KEEP,
                                     chunksize=200_000, low_memory=False, dtype=str):
                chunk["gene"] = chunk["GENE_NAME"].str.split("_").str[0]
                chunk = chunk[chunk["gene"].isin(our_symbols)]
                if len(chunk):
                    chunks.append(chunk)
    lut = pd.concat(chunks, ignore_index=True)
    lut["aa"] = lut["Mutation AA"].map(norm_aa)
    lut["tier"] = lut["MUTATION_SIGNIFICANCE_TIER"].fillna("Other")
    # One row per (gene, aa): keep the most significant tier seen.
    lut["tier_rank"] = lut["tier"].map({"1": 3, "2": 2, "3": 1}).fillna(0)
    lut = (lut.sort_values("tier_rank", ascending=False)
              .drop_duplicates(["gene", "aa"], keep="first"))
    lut = lut[["gene", "aa", "tier", "DNDS_DISEASE_QVAL_SIG", "CLINVAR_CLNSIG", "GNOMAD_GENOMES_AF"]]
    lut.to_parquet(LOOKUP, index=False)
    return lut


def main() -> None:
    if not CMC_TAR.exists():
        raise SystemExit(f"COSMIC CMC tar not found at {CMC_TAR}")
    panel = pd.read_csv(PROCESSED / "gene_panel.csv")
    our_symbols = set(panel["hugoSymbol"])

    lut = build_lookup(our_symbols)
    print(f"CMC lookup: {len(lut):,} (gene, AA) records across {lut.gene.nunique()} of our genes")

    muts = pd.read_parquet(PROCESSED / "mutations_dedup.parquet",
                           columns=["sampleId", "entrezGeneId", "proteinChange"])
    muts["gene"] = muts["entrezGeneId"].map(panel.set_index("entrezGeneId")["hugoSymbol"])
    muts["aa"] = muts["proteinChange"].map(norm_aa)
    ann = muts.merge(lut[["gene", "aa", "tier"]], on=["gene", "aa"], how="left")
    ann["tier"] = ann["tier"].fillna("unmatched")
    ann["is_significant"] = ann["tier"].isin(["1", "2", "3"])

    total = len(ann)
    print(f"\nObserved mutations: {total:,}")
    print("By COSMIC significance tier:")
    print((ann["tier"].value_counts(normalize=True) * 100).round(1).to_string())
    print(f"\nCOSMIC-significant (tier 1-3): {100*ann['is_significant'].mean():.1f}% of observed mutations")

    # Per-gene: share of observed mutations that are COSMIC-significant (drivers among hits).
    by_gene = (ann.groupby("gene")
                  .agg(nMut=("aa", "size"), pctSignificant=("is_significant", lambda s: 100*s.mean()))
                  .sort_values("nMut", ascending=False))
    by_gene.to_csv(TABLES / "mutation_tier_by_gene.csv")
    print("\nTop 12 mutated genes — % of their mutations that are COSMIC-significant drivers:")
    print(by_gene.head(12).to_string(float_format=lambda x: f"{x:.1f}"))
    print(f"\nWrote {TABLES / 'mutation_tier_by_gene.csv'} and {LOOKUP}")


if __name__ == "__main__":
    main()
