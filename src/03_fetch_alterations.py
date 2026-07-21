"""Stage 3 — acquire molecular alterations for the cohort and gene compendium.

For every study in cohort_studies.csv, fetch non-synonymous mutations restricted to the
gene panel's Entrez IDs.
Each study's mutations are written as a parquet under data/raw/mutations/ so the step is
resumable: a re-run skips studies already on disk. A combined long-format table is then
assembled at data/processed/mutations_long.parquet.

Run:  python 03_fetch_alterations.py
"""
from __future__ import annotations

import pandas as pd
from tqdm import tqdm

import cbioportal_client as cb
from config import PROCESSED, RAW

MUT_DIR = RAW / "mutations"
CNA_DIR = RAW / "cna"
MUT_DIR.mkdir(exist_ok=True)
CNA_DIR.mkdir(exist_ok=True)

# Keep only protein-changing consequences (mirrors the 2023 "non-synonymous" definition).
NONSYNON = {
    "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins",
    "In_Frame_Del", "In_Frame_Ins", "Splice_Site", "Translation_Start_Site",
    "Nonstop_Mutation", "Splice_Region",
}
MUT_COLS = ["studyId", "sampleId", "patientId", "hugoGeneSymbol", "entrezGeneId",
            "mutationType", "proteinChange", "variantType"]


def fetch_study_mutations(study_id: str, entrez_ids: list[int]) -> pd.DataFrame | None:
    """Pull + tidy mutations for one study; None if the study has no usable profile."""
    profile_id = f"{study_id}_mutations"
    sample_list_id = f"{study_id}_all"
    try:
        muts = cb.get_mutations(profile_id, sample_list_id, entrez_ids)
    except RuntimeError:
        return None
    if not muts:
        return pd.DataFrame(columns=MUT_COLS)
    df = pd.json_normalize(muts)
    df["studyId"] = study_id
    keep = [c for c in MUT_COLS if c in df.columns]
    df = df[keep]
    if "mutationType" in df.columns:
        df = df[df["mutationType"].isin(NONSYNON)]
    return df.reset_index(drop=True)


def main() -> None:
    cohort = pd.read_csv(PROCESSED / "cohort_studies.csv")
    panel = pd.read_csv(PROCESSED / "gene_panel.csv")
    entrez_ids = panel["entrezGeneId"].astype(int).tolist()
    print(f"Cohort: {len(cohort)} studies | panel: {len(entrez_ids)} genes")

    failed = []
    for sid in tqdm(cohort["studyId"], desc="mutations"):
        out = MUT_DIR / f"{sid}.parquet"
        if out.exists():
            continue
        df = fetch_study_mutations(sid, entrez_ids)
        if df is None:
            failed.append(sid)
            continue
        df.to_parquet(out, index=False)

    # Assemble combined long table from whatever is on disk.
    frames = [pd.read_parquet(p) for p in sorted(MUT_DIR.glob("*.parquet"))]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=MUT_COLS)
    combined.to_parquet(PROCESSED / "mutations_long.parquet", index=False)

    print("\n=== ALTERATION FETCH SUMMARY ===")
    print(f"Studies with mutation data : {len(frames)}")
    print(f"Total mutation rows        : {len(combined):,}")
    print(f"Distinct mutated samples   : {combined['sampleId'].nunique():,}")
    print(f"Distinct genes mutated     : {combined['entrezGeneId'].nunique()}")
    if failed:
        print(f"Studies with no usable mutation profile: {len(failed)} -> {failed[:5]}...")
    print(f"\nWrote {PROCESSED / 'mutations_long.parquet'}")

if __name__ == "__main__":
    main()
