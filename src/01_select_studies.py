"""Stage 1 — identify eligible cBioPortal studies for the analysis cohort.

Selection logic (data-driven, thresholds live in config.py):
  * keep only studies that have a MUTATION_EXTENDED profile (the core readout);
  * record whether each also has copy-number and structural-variant profiles;
  * exclude cell-line / PDX / organoid cohorts and pediatric-only cohorts by name;
  * require at least MIN_SAMPLES_PER_STUDY real samples.

Writes ``data/processed/cohort_studies.csv`` and prints a reproducible cohort summary.
API responses are cached to support resumable acquisition.
"""
from __future__ import annotations

import sys

import pandas as pd
from tqdm import tqdm

import cbioportal_client as cb
from config import (EXCLUDE_NAME_PATTERNS, EXCLUDE_PEDIATRIC_PATTERNS,
                    MIN_SAMPLES_PER_STUDY, PROCESSED, TABLES)


def _alteration_type_index(profiles: list[dict]) -> dict[str, set[str]]:
    """Map molecularAlterationType -> set of studyIds that offer it (portal-wide)."""
    idx: dict[str, set[str]] = {}
    for p in profiles:
        idx.setdefault(p["molecularAlterationType"], set()).add(p["studyId"])
    return idx


def _is_excluded(name: str, study_id: str) -> str | None:
    """Return a reason string if the study should be dropped, else None."""
    hay = f"{name} {study_id}".lower()
    for pat in EXCLUDE_NAME_PATTERNS:
        if pat in hay:
            return f"cell-line/model ({pat})"
    for pat in EXCLUDE_PEDIATRIC_PATTERNS:
        if pat in hay:
            return f"pediatric ({pat})"
    return None


def main() -> None:
    studies = cb.get_all_studies()
    profiles = cb.get_molecular_profiles()  # portal-wide, single call
    idx = _alteration_type_index(profiles)
    mut, cna, sv = idx.get("MUTATION_EXTENDED", set()), idx.get("COPY_NUMBER_ALTERATION", set()), idx.get("STRUCTURAL_VARIANT", set())
    print(f"Portal: {len(studies)} studies | mutation={len(mut)} cna={len(cna)} sv={len(sv)}")

    rows, dropped = [], []
    # Restrict the (slow) per-study sample-count pull to studies that pass the cheap
    # filters first, so we make far fewer API calls.
    candidates = []
    for s in studies:
        sid, name = s["studyId"], s.get("name", "")
        if sid not in mut:
            dropped.append((sid, "no mutation profile")); continue
        reason = _is_excluded(name, sid)
        if reason:
            dropped.append((sid, reason)); continue
        candidates.append(s)

    for s in tqdm(candidates, desc="sample counts"):
        sid = s["studyId"]
        try:
            n = len(cb.get_sample_ids(sid))
        except RuntimeError as exc:
            dropped.append((sid, f"sample fetch failed: {exc}")); continue
        if n < MIN_SAMPLES_PER_STUDY:
            dropped.append((sid, f"too small ({n} < {MIN_SAMPLES_PER_STUDY})")); continue
        rows.append({
            "studyId": sid, "name": s.get("name", ""),
            "cancerTypeId": s.get("cancerTypeId"), "refGenome": s.get("referenceGenome"),
            "nSamples": n, "hasMutation": True,
            "hasCNA": sid in cna, "hasSV": sid in sv,
        })

    cohort = pd.DataFrame(rows).sort_values("nSamples", ascending=False).reset_index(drop=True)
    out = PROCESSED / "cohort_studies.csv"
    cohort.to_csv(out, index=False)
    pd.DataFrame(dropped, columns=["studyId", "reason"]).to_csv(TABLES / "studies_excluded.csv", index=False)

    print("\n=== COHORT SUMMARY ===")
    print(f"Studies retained : {len(cohort)}")
    print(f"Total samples    : {cohort['nSamples'].sum():,}")
    print(f"Cancer-type ids  : {cohort['cancerTypeId'].nunique()}")
    print(f"With copy-number : {cohort['hasCNA'].sum()} studies")
    print(f"With struct. var : {cohort['hasSV'].sum()} studies")
    print(f"Dropped          : {len(dropped)} studies")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    sys.exit(main())
