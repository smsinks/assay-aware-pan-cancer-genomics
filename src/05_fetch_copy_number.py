"""Stage 5 — fetch high-level copy-number alterations for the cohort × gene panel.

Mutations alone under-count pathway involvement: Cell-cycle (CDKN2A deep deletion, CCNE1/
CCND1 amplification) and RTK-RAS (EGFR/ERBB2 amplification) are largely copy-number driven.
This stage pulls discrete (GISTIC) CNA for every study that has it and keeps only high-level
events — amplification (+2) and deep deletion (-2). Results are reconciled to the
canonical specimen set using the same study-qualified identifiers as the mutation data.

Resumable: one parquet per study under data/raw/cna/.

Outputs:
  data/processed/cna_long.parquet    studyId, sampleId, patientId, entrezGeneId, alteration(+/-2)
  data/processed/cna_dedup.parquet   restricted to canonical (sample, study) pairs
"""
from __future__ import annotations

import pandas as pd
from tqdm import tqdm

import cbioportal_client as cb
from config import PROCESSED, RAW

CNA_DIR = RAW / "cna"
CNA_DIR.mkdir(exist_ok=True)
HIGH_LEVEL = {2, -2}        # amplification / deep deletion only
COLS = ["studyId", "sampleId", "patientId", "entrezGeneId", "alteration"]


def discrete_cna_profile(study_id: str) -> str | None:
    for p in cb.get_molecular_profiles(study_id):
        if (p.get("molecularAlterationType") == "COPY_NUMBER_ALTERATION"
                and p.get("datatype") == "DISCRETE"):
            return p["molecularProfileId"]
    return None


def fetch_study_cna(study_id: str, entrez_ids: list[int]) -> pd.DataFrame | None:
    prof = discrete_cna_profile(study_id)
    if not prof:
        return None
    try:
        data = cb.get_cna(prof, f"{study_id}_all", entrez_ids)
    except RuntimeError:
        return None
    if not data:
        return pd.DataFrame(columns=COLS)
    df = pd.json_normalize(data)
    df = df[df["alteration"].isin(HIGH_LEVEL)]
    df["studyId"] = study_id
    return df[[c for c in COLS if c in df.columns]].reset_index(drop=True)


def main() -> None:
    cohort = pd.read_csv(PROCESSED / "cohort_studies.csv")
    panel = pd.read_csv(PROCESSED / "gene_panel.csv")
    entrez_ids = panel["entrezGeneId"].astype(int).tolist()
    studies = cohort.loc[cohort.get("hasCNA", False), "studyId"] if "hasCNA" in cohort else cohort.studyId
    print(f"Studies with CNA: {len(studies)} | panel: {len(entrez_ids)} genes")

    for sid in tqdm(studies, desc="cna"):
        out = CNA_DIR / f"{sid}.parquet"
        if out.exists():
            continue
        df = fetch_study_cna(sid, entrez_ids)
        if df is not None:
            df.to_parquet(out, index=False)

    frames = [pd.read_parquet(p) for p in sorted(CNA_DIR.glob("*.parquet"))]
    cna = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLS)
    cna.to_parquet(PROCESSED / "cna_long.parquet", index=False)

    # Resolve raw API identifiers to the evidence-aware canonical specimen map.
    canon = pd.read_parquet(PROCESSED / "canonical_samples.parquet")
    mapping = canon[["studyId", "sourceSampleId", "sampleId"]].rename(
        columns={"sampleId": "analysisSampleId"}
    )
    if "sourceSampleId" in cna.columns:
        cna = cna.drop(columns="sourceSampleId")
    dedup = (
        cna.rename(columns={"sampleId": "sourceSampleId"})
        .merge(mapping, on=["studyId", "sourceSampleId"], how="inner", validate="many_to_one")
        .rename(columns={"analysisSampleId": "sampleId"})
    )
    dedup.to_parquet(PROCESSED / "cna_dedup.parquet", index=False)

    print(f"\nStudies with CNA data : {len(frames)}")
    print(f"High-level CNA rows    : {len(cna):,}  ({(cna.alteration==2).sum():,} amp, "
          f"{(cna.alteration==-2).sum():,} deep-del)")
    print(f"After dedup            : {len(dedup):,}")
    print(f"Distinct CNA samples   : {dedup.sampleId.nunique():,}")
    top = (dedup.merge(panel[['entrezGeneId','hugoSymbol']], on='entrezGeneId')
                .groupby(['hugoSymbol','alteration']).size().unstack(fill_value=0))
    if len(top):
        top["tot"] = top.sum(axis=1)
        print("\nTop copy-number-altered genes:")
        print(top.sort_values("tot", ascending=False).head(10).to_string())


if __name__ == "__main__":
    main()
