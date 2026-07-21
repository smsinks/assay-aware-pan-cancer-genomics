"""Portable configuration for the pan-cancer mutation analysis pipeline.

Holds filesystem paths, the cBioPortal API base URL, study-selection thresholds,
and the global random seed. Importing this module guarantees the data directories
exist so downstream scripts never have to mkdir.
"""
from __future__ import annotations

from pathlib import Path

# --- Reproducibility -------------------------------------------------------
SEED = 42  # surfaced so every stochastic step can be re-seeded identically

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
CACHE = RAW / "cache"          # on-disk API response cache
PROCESSED = DATA / "processed"
EXTERNAL = DATA / "external"   # downloaded gene-compendium sources
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
LOGS = ROOT / "logs"

for _p in (RAW, CACHE, PROCESSED, EXTERNAL, FIGURES, TABLES, LOGS):
    _p.mkdir(parents=True, exist_ok=True)

# --- cBioPortal API --------------------------------------------------------
API_BASE = "https://www.cbioportal.org/api"
# The public portal paginates; this is the max page size it will honour.
API_PAGE_SIZE = 10_000_000
REQUEST_TIMEOUT = 180  # seconds; mutation pulls for large studies are slow
MAX_RETRIES = 4

# --- Study selection thresholds --------------------------------------------
# Mirrors the 2023 paper's intent (primary tumours, exclude cell lines / pediatric /
# pure targeted panels) but is data-driven and tunable here rather than hard-coded
# in the analysis scripts.
MIN_SAMPLES_PER_STUDY = 30          # drop tiny case-report cohorts
# Substrings in studyId / name that flag cohorts to exclude. Reviewed against the
# live study list, not guessed — see 01_select_studies.py for how these are applied.
EXCLUDE_NAME_PATTERNS = (
    "cell line", "cellline", "ccle", "pdx", "xenograft", "organoid",
)
EXCLUDE_PEDIATRIC_PATTERNS = (
    "pediatric", "paediatric", "pedatric", "target_", "wilms", "rhabdoid",
    "neuroblastoma", "ewing",
)
# Reference genome handling — both are kept; harmonisation happens at the gene level
# (Hugo symbols), so hg19/hg38 mixing is acceptable for gene-level frequency analysis.


# --- Deduplicated analysis data --------------------------------------------
def study_priority(sid: str) -> int:
    """Priority for resolving a barcode shared across studies (higher = kept).

    Used by Stage 1b for sample dedup and by Stage 6 to keep one clinical record per
    patient. TCGA PanCancer Atlas (harmonised) wins; earlier Firehose releases and the
    GENIE/MSK aggregators that re-include other cohorts lose.
    """
    s = sid.lower()
    if "pan_can_atlas" in s:
        return 100
    if "_tcga_gdc" in s:
        return 70
    if "_tcga_pub" in s:
        return 60
    if "_tcga" in s:
        return 50
    if "genie" in s or any(k in s for k in
            ("msk_chord", "msk_met", "metropism", "msk_pan", "msk_solid", "_msk_2")):
        return 10
    return 30


def load_analysis_data():
    """Return (mutations_df, n_total_samples, deduped: bool) for the analysis stages.

    cBioPortal lists the same physical tumour under many studies (TCGA Firehose/GDC/
    publication/PanCancer-Atlas versions; GENIE/MSK aggregators). Stage 1b
    (``01b_dedupe_samples.py``) resolves each barcode to one canonical study; when its
    outputs exist they are used so the numerator and denominator are both de-duplicated.
    Falls back to the raw, duplicated data with a clear flag if dedup has not been run.
    """
    import pandas as pd

    dedup = PROCESSED / "mutations_dedup.parquet"
    canon = PROCESSED / "canonical_samples.parquet"
    if dedup.exists() and canon.exists():
        muts = pd.read_parquet(dedup)
        n_total = len(pd.read_parquet(canon, columns=["sampleId"]))
        return muts, n_total, True
    # Fallback: raw mutations + summed study sample counts (DUPLICATED — see Stage 1b).
    muts = pd.read_parquet(PROCESSED / "mutations_long.parquet")
    cohort = pd.read_csv(PROCESSED / "cohort_studies.csv")
    return muts, int(cohort["nSamples"].sum()), False
