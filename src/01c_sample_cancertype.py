"""Stage 1c — assign sample-level OncoTree cancer types.

For each cohort study, pull sample-level clinical data and extract, per sample, the most
specific cancer-type label available (CANCER_TYPE preferred, then ONCOTREE_CODE, then the
study-level cancerTypeId as a fallback). Restricted to canonical (de-duplicated) samples.

Output: data/processed/sample_cancertype.parquet  (analysis and source sample IDs,
studyId, cancerType, detailed cancer type and OncoTree code)
"""
from __future__ import annotations

import pandas as pd
from tqdm import tqdm

import cbioportal_client as cb
from config import PROCESSED

WANT = {"CANCER_TYPE", "CANCER_TYPE_DETAILED", "ONCOTREE_CODE"}


def main() -> None:
    canon = pd.read_parquet(PROCESSED / "canonical_samples.parquet")
    cohort = pd.read_csv(PROCESSED / "cohort_studies.csv").set_index("studyId")["cancerTypeId"].to_dict()
    source_to_analysis = {
        (row.studyId, row.sourceSampleId): row.sampleId
        for row in canon.itertuples(index=False)
    }

    rows = []
    for sid in tqdm(sorted(canon["studyId"].unique()), desc="sample clinical"):
        try:
            data = cb.get_sample_clinical(sid)
        except RuntimeError:
            continue
        df = pd.DataFrame(data)
        if df.empty or "clinicalAttributeId" not in df:
            continue
        df = df[df["clinicalAttributeId"].isin(WANT)]
        if df.empty:
            continue
        wide = df.pivot_table(index="sampleId", columns="clinicalAttributeId",
                              values="value", aggfunc="first")
        for source_sample_id, r in wide.iterrows():
            analysis_sample_id = source_to_analysis.get((sid, source_sample_id))
            if analysis_sample_id is None:
                continue
            ctype = r.get("CANCER_TYPE") or cohort.get(sid)
            rows.append(
                (
                    analysis_sample_id,
                    source_sample_id,
                    sid,
                    ctype,
                    r.get("CANCER_TYPE_DETAILED"),
                    r.get("ONCOTREE_CODE"),
                )
            )

    out = pd.DataFrame(
        rows,
        columns=[
            "sampleId", "sourceSampleId", "studyId", "cancerType",
            "cancerTypeDetailed", "oncotree",
        ],
    )
    # Fall back to study-level type for canonical samples with no sample-level annotation.
    have = set(zip(out["sampleId"], out["studyId"]))
    extra = [
        (row.sampleId, row.sourceSampleId, row.studyId, cohort.get(row.studyId), None, None)
        for row in canon.itertuples(index=False)
        if (row.sampleId, row.studyId) not in have
    ]
    out = pd.concat([out, pd.DataFrame(extra, columns=out.columns)], ignore_index=True)
    out["cancerType"] = out["cancerType"].fillna("unknown").replace("", "unknown")
    out.to_parquet(PROCESSED / "sample_cancertype.parquet", index=False)

    n_typed = (out.cancerType != "mixed").sum() - (out.cancerType == "unknown").sum()
    print(f"Canonical samples typed: {len(out):,}")
    print(f"Distinct cancer types  : {out.cancerType.nunique()}")
    print("Top 15 by sample count:")
    print(out.cancerType.value_counts().head(15).to_string())


if __name__ == "__main__":
    main()
