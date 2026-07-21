"""Stage 1d — construct the biospecimen- and diagnosis-aware tissue cohort.

Mutation-profiled canonical specimens are classified from study and sample metadata.
Liquid biopsies, normal material, clonal-haematopoiesis cohorts and experimental models
are excluded. One representative tissue tumour is selected per patient key using the
ordered specimen classes primary, recurrent, metastatic and unspecified tissue tumour.
The inclusive tissue set is retained for sampling-unit sensitivity analysis.

Outputs
-------
data/processed/sample_metadata.parquet
data/processed/analysis_samples_curated.parquet
data/processed/mutations_curated.parquet
data/processed/cna_curated.parquet
results/tables/cohort_exclusion_summary.csv
results/tables/cohort_study_manifest.csv
results/tables/cohort_study_disposition.csv
results/tables/mutation_profile_membership_audit.csv
results/tables/cohort_sensitivity.csv
results/tables/gene_frequencies_curated.csv
results/figures/diagnostics/cohort_selection.{pdf,svg,png}
"""
from __future__ import annotations

import re
import warnings
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.proportion import proportion_confint
from tqdm import tqdm

import cbioportal_client as cb
from config import FIGURES, PROCESSED, TABLES
from callability import partition_callable_mutations
from plot_style import COLORS, apply as apply_style, figsize, panel_label, save_figure
from oncotree_taxonomy import annotate_taxonomy

warnings.filterwarnings("ignore")

CLINICAL_FIELDS = {
    "CANCER_TYPE",
    "CANCER_TYPE_DETAILED",
    "ONCOTREE_CODE",
    "SAMPLE_TYPE",
    "SAMPLE_CLASS",
    "SPECIMEN_TYPE",
    "SPECIMEN_PRESERVATION_TYPE",
    "TUMOR_TYPE",
    "SOMATIC_STATUS",
    "PRIMARY_SITE",
    "METASTATIC_SITE",
    "TMB_NONSYNONYMOUS",
    "TMB_SCORE",
    "MUTATION_COUNT",
    "GENE_PANEL",
}

# These are exclusively non-tissue molecular cohorts.  Mixed tissue/liquid cohorts are
# intentionally absent and are resolved using SAMPLE_CLASS below.
EXCLUSIVE_LIQUID_STUDIES = {
    "msk_ctdna_vte_2024",
    "csf_msk_2024",
    "pancreas_ctdna_msk_2025",
    "breast_msk_cfdna_2026",
    "ucec_ccr_cfdna_msk_2022",
}


def clean_text(value):
    if pd.isna(value):
        return None
    s = str(value).strip().strip('"').strip()
    return s or None


def study_exclusion_reason(study_id: str, name: str) -> str | None:
    hay = f"{study_id} {name}".lower()
    if "clonal hematopoiesis" in hay or re.search(r"(^|_)ch(_|$)", study_id.lower()):
        return "clonal haematopoiesis"
    if any(x in hay for x in ("pdmr", "patient-derived model", "patient derived model")):
        return "patient-derived model"
    if "normal_skin_" in study_id.lower() or name.lower().startswith("normal "):
        return "normal-cell cohort"
    if study_id in EXCLUSIVE_LIQUID_STUDIES:
        return "liquid-biopsy cohort"
    return None


def sample_exclusion_reason(row: pd.Series) -> str | None:
    fields = " ".join(
        str(row.get(c) or "")
        for c in ("sampleClass", "sampleType", "specimenPreservationType")
    ).lower()
    if any(x in fields for x in ("cfdna", "ctdna", "cell-free", "cell free")):
        return "liquid-biopsy sample"
    if re.search(r"(^|\s)(normal|control)(\s|$)", fields):
        return "normal/control sample"
    if any(x in fields for x in ("cell line", "cellline", "organoid", "xenograft", "pdx")):
        return "experimental model sample"
    if str(row.get("sampleType") or "").strip().upper() == "CSF":
        return "liquid-biopsy sample"
    return None


def sample_type_group(row: pd.Series) -> str:
    st = str(row.get("sampleType") or "").lower()
    sc = str(row.get("sampleClass") or "").lower()
    if any(x in f"{st} {sc}" for x in ("cfdna", "ctdna", "csf")):
        return "Liquid biopsy"
    if "metasta" in st:
        return "Metastasis"
    if "recurr" in st:
        return "Recurrence"
    if "primary" in st:
        return "Primary"
    if any(x in st for x in ("tumor", "tumour", "biopsy", "resection")) or sc in {
        "tumor",
        "tumour",
        "resection",
        "biopsy",
    }:
        return "Tissue tumour (unspecified)"
    return "Unknown tissue status"


def representative_priority(group: str) -> int:
    return {
        "Primary": 0,
        "Recurrence": 1,
        "Metastasis": 2,
        "Tissue tumour (unspecified)": 3,
        "Unknown tissue status": 4,
        "Liquid biopsy": 9,
    }.get(group, 8)


def add_disease_identity(meta: pd.DataFrame) -> pd.DataFrame:
    """Attach disease labels while preserving a strict one-person sampling key."""
    meta = meta.copy()
    meta["diseaseIdentityCode"] = meta.cancerFamilyCode.astype(str)
    meta["diseaseIdentityResolution"] = "reviewed cancer family; not used as sampling key"
    meta["patientKey"] = meta.personKey.astype(str)
    return meta


def patient_namespace(study_id: str, patient_id: str | None, sample_id: str) -> str:
    """Use globally recognizable identifiers globally; namespace generic IDs by study."""
    p = clean_text(patient_id)
    if not p:
        return f"{study_id}::SAMPLE::{sample_id}"
    if re.match(r"^(TCGA-|P-\d+|GENIE-)", p, flags=re.IGNORECASE):
        return p.upper()
    return f"{study_id}::{p}"


def extract_sample_metadata() -> pd.DataFrame:
    cache = PROCESSED / "sample_metadata.parquet"
    canon = pd.read_parquet(PROCESSED / "canonical_samples.parquet")
    cohort = pd.read_csv(PROCESSED / "cohort_studies.csv")
    study_name = cohort.set_index("studyId")["name"].to_dict()
    study_ct = cohort.set_index("studyId")["cancerTypeId"].to_dict()
    by_study = {sid: set(x.sourceSampleId) for sid, x in canon.groupby("studyId")}
    source_to_analysis = {
        (row.studyId, row.sourceSampleId): row.sampleId
        for row in canon.itertuples(index=False)
    }

    # A fallback patient map is available for mutation-bearing samples even when a study
    # has no sample-level clinical rows.
    mutation_columns = pd.read_parquet(PROCESSED / "mutations_dedup.parquet").columns
    mutation_sample_column = (
        "sourceSampleId" if "sourceSampleId" in mutation_columns else "sampleId"
    )
    muts = pd.read_parquet(
        PROCESSED / "mutations_dedup.parquet",
        columns=["studyId", mutation_sample_column, "patientId"],
    ).rename(columns={mutation_sample_column: "sourceSampleId"})
    mut_patient = (
        muts.dropna(subset=["patientId"])
        .drop_duplicates(["studyId", "sourceSampleId"])
        .set_index(["studyId", "sourceSampleId"])["patientId"]
        .to_dict()
    )

    records: list[dict] = []
    seen: set[tuple[str, str]] = set()
    rename = {
        "CANCER_TYPE": "cancerType",
        "CANCER_TYPE_DETAILED": "cancerTypeDetailed",
        "ONCOTREE_CODE": "oncotreeCode",
        "SAMPLE_TYPE": "sampleType",
        "SAMPLE_CLASS": "sampleClass",
        "SPECIMEN_TYPE": "specimenType",
        "SPECIMEN_PRESERVATION_TYPE": "specimenPreservationType",
        "TUMOR_TYPE": "tumorType",
        "SOMATIC_STATUS": "somaticStatus",
        "PRIMARY_SITE": "primarySite",
        "METASTATIC_SITE": "metastaticSite",
        "TMB_NONSYNONYMOUS": "tmbNonsynonymous",
        "TMB_SCORE": "tmbScore",
        "MUTATION_COUNT": "mutationCountClinical",
        "GENE_PANEL": "genePanelClinical",
    }

    for sid in tqdm(sorted(by_study), desc="sample metadata"):
        try:
            raw = pd.DataFrame(cb.get_sample_clinical(sid))
        except RuntimeError:
            raw = pd.DataFrame()
        if raw.empty or "clinicalAttributeId" not in raw:
            continue
        raw = raw[
            raw["sampleId"].isin(by_study[sid])
            & raw["clinicalAttributeId"].isin(CLINICAL_FIELDS)
        ]
        if raw.empty:
            continue
        wide = raw.pivot_table(
            index="sampleId", columns="clinicalAttributeId", values="value", aggfunc="first"
        )
        patient_map = (
            raw.dropna(subset=["patientId"])
            .drop_duplicates("sampleId")
            .set_index("sampleId")["patientId"]
            .to_dict()
            if "patientId" in raw
            else {}
        )
        for source_sample_id, values in wide.iterrows():
            analysis_sample_id = source_to_analysis.get((sid, source_sample_id))
            if analysis_sample_id is None:
                continue
            rec = {
                "sampleId": analysis_sample_id,
                "sourceSampleId": source_sample_id,
                "studyId": sid,
                "patientId": patient_map.get(source_sample_id)
                or mut_patient.get((sid, source_sample_id)),
                "studyName": study_name.get(sid),
                "studyCancerType": study_ct.get(sid),
            }
            for source, dest in rename.items():
                rec[dest] = clean_text(values.get(source))
            records.append(rec)
            seen.add((analysis_sample_id, sid))

    # Preserve every canonical sample, including studies without sample clinical data.
    for row in canon.itertuples(index=False):
        key = (row.sampleId, row.studyId)
        if key in seen:
            continue
        records.append(
            {
                "sampleId": row.sampleId,
                "sourceSampleId": row.sourceSampleId,
                "studyId": row.studyId,
                "patientId": mut_patient.get((row.studyId, row.sourceSampleId)),
                "studyName": study_name.get(row.studyId),
                "studyCancerType": study_ct.get(row.studyId),
            }
        )

    out = pd.DataFrame(records)
    for col in rename.values():
        if col not in out:
            out[col] = None
    out = annotate_taxonomy(out)
    out["personKey"] = [
        patient_namespace(st, p, s)
        for st, p, s in zip(out.studyId, out.patientId, out.sampleId)
    ]
    out = add_disease_identity(out)
    out.to_parquet(cache, index=False)
    return out


def build_curated_samples(meta: pd.DataFrame) -> pd.DataFrame:
    cohort = pd.read_csv(PROCESSED / "cohort_studies.csv")
    names = cohort.set_index("studyId")["name"].to_dict()
    meta = add_disease_identity(meta)
    meta["studyExclusionReason"] = [
        study_exclusion_reason(sid, names.get(sid, "")) for sid in meta.studyId
    ]
    meta["sampleExclusionReason"] = meta.apply(sample_exclusion_reason, axis=1)
    meta["exclusionReason"] = meta["studyExclusionReason"].fillna(meta["sampleExclusionReason"])
    meta["tissueEligible"] = meta["exclusionReason"].isna()
    meta["sampleTypeGroup"] = meta.apply(sample_type_group, axis=1)
    meta["representativePriority"] = meta["sampleTypeGroup"].map(representative_priority)

    meta["selectedOnePerPatient"] = False
    eligible = meta[meta.tissueEligible].copy()
    eligible = eligible.sort_values(
        ["patientKey", "representativePriority", "studyId", "sampleId"],
        ascending=[True, True, True, True],
    )
    selected_idx = eligible.drop_duplicates("patientKey", keep="first").index
    meta.loc[selected_idx, "selectedOnePerPatient"] = True
    family_counts = eligible.groupby("personKey").cancerFamilyCode.nunique()
    meta["nCancerFamiliesForPerson"] = meta.personKey.map(family_counts).fillna(0).astype(int)
    meta["personHasMultipleCancerFamilies"] = meta.nCancerFamiliesForPerson.gt(1)
    meta["analysisEligible"] = meta["tissueEligible"] & meta["selectedOnePerPatient"]
    meta.to_parquet(PROCESSED / "analysis_samples_curated.parquet", index=False)
    return meta


def assay_denominators(samples: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    assay = pd.read_parquet(PROCESSED / "sample_assay.parquet")
    membership = pd.read_parquet(PROCESSED / "panel_gene_membership.parquet")
    assay = assay.merge(
        samples[["sampleId", "studyId"]], on=["sampleId", "studyId"], how="inner"
    )
    n_wes = int(assay.assayType.str.startswith("WES/WGS").sum())
    panel_counts = assay.loc[assay.assayType == "Targeted panel", "genePanelId"].value_counts()
    member_panels = membership.groupby("entrezGeneId")["genePanelId"].apply(set).to_dict()
    rows = []
    for ent in panel.entrezGeneId.astype(int):
        rows.append(
            (
                ent,
                n_wes
                + sum(int(panel_counts.get(pid, 0)) for pid in member_panels.get(ent, set())),
            )
        )
    return pd.DataFrame(rows, columns=["entrezGeneId", "nProfiled"])


def frequency_table(samples: pd.DataFrame, mutations: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    den = assay_denominators(samples, panel)
    num = (
        mutations.drop_duplicates(["sampleId", "entrezGeneId"])
        .groupby("entrezGeneId")["sampleId"]
        .nunique()
        .rename("nMutated")
    )
    out = den.merge(num, on="entrezGeneId", how="left").fillna({"nMutated": 0})
    out["nMutated"] = out["nMutated"].astype(int)
    if (out.nMutated > out.nProfiled).any():
        offending = out.loc[out.nMutated > out.nProfiled, ["entrezGeneId", "nMutated", "nProfiled"]]
        raise AssertionError(
            "Mutation numerators exceed documented callable denominators: "
            + offending.head().to_dict("records").__repr__()
        )
    out["freqPct"] = 100 * out.nMutated / out.nProfiled.clip(lower=1)
    ci = np.array(
        [
            proportion_confint(int(k), int(n), method="wilson") if n else (np.nan, np.nan)
            for k, n in zip(out.nMutated, out.nProfiled)
        ]
    )
    out["freqCiLowPct"] = 100 * ci[:, 0]
    out["freqCiHighPct"] = 100 * ci[:, 1]
    out = out.merge(
        panel[["entrezGeneId", "hugoSymbol", "roleInCancer", "cosmicTier", "highConfidence"]],
        on="entrezGeneId",
        how="left",
    ).rename(columns={"hugoSymbol": "gene"})
    out["pctCuratedSamplesProfilingGene"] = 100 * out.nProfiled / len(samples)
    return out.sort_values(["freqPct", "nMutated"], ascending=False).reset_index(drop=True)


def write_sensitivity_tables(meta: pd.DataFrame, panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mutations_all = pd.read_parquet(PROCESSED / "mutations_dedup.parquet")
    mutations, conflicts = partition_callable_mutations(mutations_all)
    cna = pd.read_parquet(PROCESSED / "cna_dedup.parquet")
    masks = {
        "All canonical samples": meta.index,
        "Biospecimen-eligible tissue": meta.index[meta.tissueEligible],
        "One tissue sample per patient": meta.index[meta.analysisEligible],
    }

    long_rows = []
    final_freq = None
    for label, idx in masks.items():
        samples = meta.loc[idx, ["sampleId", "studyId"]]
        keep = set(samples.sampleId)
        m = mutations[mutations.sampleId.isin(keep)]
        ft = frequency_table(samples, m, panel)
        ft["cohortDefinition"] = label
        ft["nCohortSamples"] = len(samples)
        long_rows.append(ft)
        if label == "One tissue sample per patient":
            final_freq = ft.copy()

    sensitivity = pd.concat(long_rows, ignore_index=True)
    sensitivity.to_csv(TABLES / "cohort_sensitivity.csv", index=False)
    assert final_freq is not None
    final_freq.drop(columns=["cohortDefinition", "nCohortSamples"]).to_csv(
        TABLES / "gene_frequencies_curated.csv", index=False
    )

    final_ids = set(meta.loc[meta.analysisEligible, "sampleId"])
    mutations[mutations.sampleId.isin(final_ids)].to_parquet(
        PROCESSED / "mutations_curated.parquet", index=False
    )
    conflict_curated = conflicts[conflicts.sampleId.isin(final_ids)].copy()
    symbols = panel.set_index("entrezGeneId")["hugoSymbol"]
    conflict_curated["gene"] = conflict_curated.entrezGeneId.map(symbols)
    conflict_detail = (
        conflict_curated.groupby(
            ["studyId", "genePanelId", "entrezGeneId", "gene", "callabilityConflictReason"],
            dropna=False,
        )
        .agg(
            nSamples=("sampleId", "nunique"),
            nMutationRecords=("sampleId", "size"),
        )
        .reset_index()
        .sort_values(["nSamples", "nMutationRecords"], ascending=False)
    )
    conflict_detail.insert(0, "recordLevel", "Study-panel-gene")
    conflict_detail["nSampleGeneConflicts"] = conflict_detail.nSamples
    overall = pd.DataFrame(
        [
            {
                "recordLevel": "Overall curated cohort",
                "studyId": "ALL",
                "genePanelId": "ALL TARGETED PANELS",
                "entrezGeneId": pd.NA,
                "gene": "ALL",
                "callabilityConflictReason": (
                    "Observed targeted-panel mutation absent from documented panel gene list"
                ),
                "nSamples": conflict_curated.sampleId.nunique(),
                "nMutationRecords": len(conflict_curated),
                "nSampleGeneConflicts": len(
                    conflict_curated.drop_duplicates(["sampleId", "entrezGeneId"])
                ),
            }
        ]
    )
    conflict_audit = pd.concat([overall, conflict_detail], ignore_index=True)
    conflict_audit.to_csv(TABLES / "assay_callability_conflicts.csv", index=False)
    cna[cna.sampleId.isin(final_ids)].to_parquet(PROCESSED / "cna_curated.parquet", index=False)
    return sensitivity, final_freq


def draw_flow(ax, values: dict[str, int]) -> None:
    ax.axis("off")
    boxes = [
        (0.005, 0.22, 0.215, 0.58, f"Portal-selected\n{values['studies']:,} studies\n{values['raw']:,} study-sample rows"),
        (0.265, 0.22, 0.215, 0.58, f"Specimen de-duplication\n{values['canonical']:,} canonical samples\n{values['duplicates']:,} duplicate rows removed"),
        (0.525, 0.22, 0.215, 0.58, f"Biospecimen QC\n{values['tissue']:,} tissue samples\n{values['excluded']:,} non-tissue samples removed"),
        (
            0.785,
            0.22,
            0.21,
            0.58,
            f"Primary analysis\n{values['final']:,} representative cases\none per constructed patient key",
        ),
    ]
    for x, y, w, h, text in boxes:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.015,rounding_size=0.025",
            transform=ax.transAxes,
            facecolor=COLORS["very_light_grey"],
            edgecolor=COLORS["grey"],
            linewidth=0.7,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.5)
    for x1, x2 in ((0.22, 0.265), (0.48, 0.525), (0.74, 0.785)):
        ax.add_patch(
            FancyArrowPatch(
                (x1 + 0.01, 0.51),
                (x2 - 0.01, 0.51),
                transform=ax.transAxes,
                arrowstyle="-|>",
                mutation_scale=8,
                color=COLORS["blue"],
                linewidth=0.9,
            )
        )


def make_figure(meta: pd.DataFrame, sensitivity: pd.DataFrame, curated: pd.DataFrame) -> None:
    apply_style()
    fig = plt.figure(figsize=figsize(180, 178))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[0.72, 1.0, 1.05], hspace=0.48, wspace=0.38)

    ax_a = fig.add_subplot(gs[0, :])
    cohort = pd.read_csv(PROCESSED / "cohort_studies.csv")
    flow_values = {
        "studies": len(cohort),
        "raw": int(cohort.nSamples.sum()),
        "canonical": len(meta),
        "duplicates": int(cohort.nSamples.sum()) - len(meta),
        "tissue": int(meta.tissueEligible.sum()),
        "excluded": int((~meta.tissueEligible).sum()),
        "final": int(meta.analysisEligible.sum()),
    }
    draw_flow(ax_a, flow_values)
    panel_label(ax_a, "a", x=-0.025, y=1.02)

    ax_b = fig.add_subplot(gs[1, 0])
    reasons = meta.loc[~meta.tissueEligible, "exclusionReason"].value_counts().sort_values()
    ax_b.barh(reasons.index, reasons.values, color=COLORS["orange"])
    for y, value in enumerate(reasons.values):
        ax_b.text(value, y, f"  {value:,}", va="center", fontsize=5.5)
    ax_b.set_xlabel("Samples excluded")
    ax_b.set_title("Explicit non-tissue molecular cohorts are material", loc="left")
    panel_label(ax_b, "b")

    ax_c = fig.add_subplot(gs[1, 1])
    assay = pd.read_parquet(PROCESSED / "sample_assay.parquet")
    assay = assay.merge(
        meta.loc[meta.analysisEligible, ["sampleId", "studyId"]],
        on=["sampleId", "studyId"],
        how="inner",
    )
    assay_counts = assay.assayType.replace(
        {"WES/WGS (assumed; no panel metadata)": "WES/WGS (assumed)"}
    ).value_counts()
    assay_counts = assay_counts.reindex(
        ["WES/WGS", "WES/WGS (assumed)", "Targeted panel"], fill_value=0
    )
    colours = [COLORS["blue"], COLORS["sky"], COLORS["purple"]]
    bars = ax_c.bar(range(len(assay_counts)), assay_counts.values, color=colours)
    ax_c.set_xticks(range(len(assay_counts)))
    ax_c.set_xticklabels(assay_counts.index, rotation=25, ha="right")
    ax_c.set_ylabel("Curated samples")
    for bar, value in zip(bars, assay_counts.values):
        ax_c.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}", ha="center", va="bottom", fontsize=5.5)
    ax_c.set_title("Assay composition remains heterogeneous", loc="left")
    panel_label(ax_c, "c")

    # Rank/frequency shifts for genes that move most after biospecimen curation.
    all_f = sensitivity[sensitivity.cohortDefinition == "All canonical samples"].copy()
    cur_f = sensitivity[sensitivity.cohortDefinition == "One tissue sample per patient"].copy()
    merged = all_f[["gene", "freqPct"]].merge(
        cur_f[["gene", "freqPct", "nMutated"]], on="gene", suffixes=("All", "Curated")
    )
    merged["delta"] = merged.freqPctCurated - merged.freqPctAll
    merged["absDelta"] = merged.delta.abs()
    forced = {"TP53", "KRAS", "PIK3CA", "DNMT3A", "PPM1D", "PLEC", "MUC16", "LRP1B"}
    chosen = set(merged.nlargest(7, "absDelta").gene) | forced
    slope = merged[merged.gene.isin(chosen)].sort_values("freqPctCurated")

    ax_d = fig.add_subplot(gs[2, 0])
    for i, row in enumerate(slope.itertuples(index=False)):
        colour = COLORS["blue"] if row.delta >= 0 else COLORS["vermillion"]
        ax_d.plot([0, 1], [row.freqPctAll, row.freqPctCurated], color=colour, alpha=0.65, lw=0.8)
        ax_d.scatter([0, 1], [row.freqPctAll, row.freqPctCurated], color=colour, s=8, zorder=3)
    # Spread right-hand labels vertically while retaining a fine connector to the point.
    labels = slope[["gene", "freqPctCurated"]].sort_values("freqPctCurated").copy()
    positions = labels.freqPctCurated.to_numpy(float).copy()
    # At final 180-mm width, a one-unit gap avoids label collisions in the dense
    # 8–13% prevalence region while preserving the quantitative endpoints.
    min_gap = 1.05
    for i in range(1, len(positions)):
        positions[i] = max(positions[i], positions[i - 1] + min_gap)
    if len(positions) and positions[-1] > slope.freqPctCurated.max() + 2.5:
        positions -= positions[-1] - (slope.freqPctCurated.max() + 2.5)
        for i in range(len(positions) - 2, -1, -1):
            positions[i] = min(positions[i], positions[i + 1] - min_gap)
    for (_, row), y_text in zip(labels.iterrows(), positions):
        ax_d.plot([1.0, 1.035], [row.freqPctCurated, y_text], color=COLORS["grey"], lw=0.35)
        ax_d.text(1.045, y_text, row.gene, va="center", fontsize=4.6)
    ax_d.set_xlim(-0.15, 1.42)
    ax_d.set_xticks([0, 1]); ax_d.set_xticklabels(["All canonical", "Curated tissue\n(one/patient)"])
    ax_d.set_ylabel("Assay-aware prevalence (%)")
    ax_d.set_title("Cohort curation reorders the landscape", loc="left")
    panel_label(ax_d, "d")

    ax_e = fig.add_subplot(gs[2, 1])
    plot = merged[(merged.nMutated >= 40) & (merged.freqPctAll > 0) & (merged.freqPctCurated > 0)]
    ax_e.scatter(
        plot.freqPctAll,
        plot.freqPctCurated,
        s=np.clip(np.sqrt(plot.nMutated), 3, 18),
        color=COLORS["blue"],
        alpha=0.45,
        edgecolors="none",
    )
    lim = max(plot.freqPctAll.max(), plot.freqPctCurated.max()) * 1.04
    ax_e.plot([0, lim], [0, lim], color=COLORS["grey"], ls="--", lw=0.7)
    label_genes = set(merged.nlargest(6, "absDelta").gene) | {"TP53", "KRAS", "PIK3CA", "DNMT3A", "PPM1D"}
    offsets = {
        "TP53": (4, 3), "KRAS": (4, 8), "MUC16": (4, 2), "APC": (4, -2),
        "PIK3CA": (4, -10), "DNMT3A": (4, -8), "PPM1D": (4, -8),
        "LRP1B": (4, 5), "PLEC": (4, 3),
    }
    for row in merged[merged.gene.isin(label_genes)].itertuples(index=False):
        ax_e.annotate(
            row.gene,
            (row.freqPctAll, row.freqPctCurated),
            fontsize=4.6,
            xytext=offsets.get(row.gene, (2, 2)),
            textcoords="offset points",
        )
    rho, p = spearmanr(plot.freqPctAll, plot.freqPctCurated)
    ax_e.text(0.04, 0.94, f"Spearman ρ={rho:.2f}", transform=ax_e.transAxes, va="top")
    ax_e.set_xlabel("All-canonical prevalence (%)")
    ax_e.set_ylabel("Curated prevalence (%)")
    ax_e.set_title("Most common drivers remain rank-correlated", loc="left")
    panel_label(ax_e, "e")

    # Store cohort-selection diagnostics separately from the principal figure inputs.
    source = TABLES.parent / "diagnostics" / "cohort_selection"
    source.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            ("Portal-selected studies", flow_values["studies"], "studies"),
            ("Portal-selected study-sample rows", flow_values["raw"], "study-sample rows"),
            ("Duplicate study-sample rows removed", flow_values["duplicates"], "study-sample rows"),
            ("Canonical sample identifiers", flow_values["canonical"], "samples"),
            ("Non-tissue/non-tumour samples removed", flow_values["excluded"], "samples"),
            ("Biospecimen-eligible tissue samples", flow_values["tissue"], "samples"),
            (
                "Additional eligible specimens omitted by one-per-key rule",
                flow_values["tissue"] - flow_values["final"],
                "samples",
            ),
            ("Representative primary-analysis cases", flow_values["final"], "cases"),
        ],
        columns=["stageOrQuantity", "value", "unit"],
    ).to_csv(source / "cohort_flow.csv", index=False)
    meta.loc[~meta.tissueEligible, ["studyId", "exclusionReason"]].value_counts().rename("nSamples").reset_index().to_csv(
        source / "biospecimen_exclusions.csv", index=False
    )
    assay_counts.rename_axis("assayType").rename("nSamples").reset_index().to_csv(
        source / "assay_composition.csv", index=False
    )
    merged.to_csv(source / "frequency_sensitivity.csv", index=False)

    # Keep the long exclusion labels inside the fixed 180-mm canvas.  Explicit
    # margins are fixed so vector and raster exports use identical geometry.
    # bounding box (which would otherwise change the physical figure width).
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.08, top=0.96)
    diagnostic_figure = FIGURES / "diagnostics" / "cohort_selection"
    diagnostic_figure.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, diagnostic_figure)
    plt.close(fig)


def write_study_manifest(meta: pd.DataFrame) -> pd.DataFrame:
    """Write study-level provenance and the exact contribution to the curated cohort."""
    cohort = pd.read_csv(PROCESSED / "cohort_studies.csv")
    portal = pd.DataFrame(cb.get_all_studies())
    provenance_columns = [
        column
        for column in ("studyId", "citation", "pmid", "importDate", "description")
        if column in portal
    ]
    provenance = portal[provenance_columns].drop_duplicates("studyId")

    counts = (
        meta.groupby("studyId")
        .agg(
            nCanonicalSamples=("sampleId", "size"),
            nTissueEligible=("tissueEligible", "sum"),
            nAnalysisSamples=("analysisEligible", "sum"),
            nConstructedPatientKeys=("patientKey", "nunique"),
            nConstructedPeople=("personKey", "nunique"),
            nCancerFamilies=("cancerFamilyCode", "nunique"),
        )
        .reset_index()
    )
    excluded = (
        meta.loc[~meta.tissueEligible]
        .groupby("studyId")
        .size()
        .rename("nExcludedNonTissue")
        .reset_index()
    )
    assay = pd.read_parquet(PROCESSED / "sample_assay.parquet").merge(
        meta.loc[meta.analysisEligible, ["sampleId", "studyId"]],
        on=["sampleId", "studyId"],
        how="inner",
    )
    assay["assayGroup"] = np.where(
        assay.assayType.str.startswith("WES/WGS"), "WES_WGS", "TargetedPanel"
    )
    assay_counts = (
        assay.pivot_table(
            index="studyId", columns="assayGroup", values="sampleId", aggfunc="nunique", fill_value=0
        )
        .rename_axis(columns=None)
        .reset_index()
        .rename(columns={"WES_WGS": "nAnalysisWesWgs", "TargetedPanel": "nAnalysisTargetedPanel"})
    )
    panel_ids = (
        assay.loc[assay.assayGroup.eq("TargetedPanel")]
        .groupby("studyId")["genePanelId"]
        .agg(lambda values: ";".join(sorted({str(value) for value in values.dropna()})))
        .rename("targetedPanelIds")
        .reset_index()
    )

    manifest = (
        cohort.merge(provenance, on="studyId", how="left")
        .merge(counts, on="studyId", how="left")
        .merge(excluded, on="studyId", how="left")
        .merge(assay_counts, on="studyId", how="left")
        .merge(panel_ids, on="studyId", how="left")
    )
    numeric = [
        "nCanonicalSamples", "nTissueEligible", "nAnalysisSamples",
        "nConstructedPatientKeys", "nConstructedPeople", "nCancerFamilies",
        "nExcludedNonTissue", "nAnalysisWesWgs",
        "nAnalysisTargetedPanel",
    ]
    manifest[numeric] = manifest[numeric].fillna(0).astype(int)
    manifest["contributesToPrimaryAnalysis"] = manifest.nAnalysisSamples.gt(0)
    manifest.to_csv(TABLES / "cohort_study_manifest.csv", index=False)

    # Preserve a complete portal-wide disposition, including the 107 studies excluded
    # before cohort construction and the retained studies that contribute no final case.
    excluded_path = TABLES / "studies_excluded.csv"
    excluded = (
        pd.read_csv(excluded_path)
        if excluded_path.exists()
        else pd.DataFrame(columns=["studyId", "reason"])
    )
    portal_columns = [
        column
        for column in (
            "studyId", "name", "description", "cancerTypeId", "referenceGenome",
            "citation", "pmid", "importDate", "sequencedSampleCount", "allSampleCount",
        )
        if column in portal
    ]
    disposition = (
        portal[portal_columns]
        .drop_duplicates("studyId")
        .merge(excluded[["studyId", "reason"]], on="studyId", how="left")
        .merge(
            manifest,
            on="studyId",
            how="left",
            suffixes=("_portal", "_retained"),
        )
    )
    disposition["studySelectionStage"] = np.where(
        disposition.reason.notna(), "excluded during initial study screening", "retained input cohort"
    )
    disposition["finalDisposition"] = np.select(
        [
            disposition.reason.notna(),
            disposition.nAnalysisSamples.fillna(0).gt(0),
            disposition.nCanonicalSamples.fillna(0).eq(0),
            disposition.nTissueEligible.fillna(0).eq(0),
        ],
        [
            "excluded upstream: " + disposition.reason.fillna(""),
            "contributes selected tissue tumours",
            "retained study; all specimen identities represented by a higher-priority overlapping release",
            "retained study; no eligible tissue tumour",
        ],
        default="retained study; eligible tissue specimen removed by within-disease representative selection",
    )
    disposition.to_csv(TABLES / "cohort_study_disposition.csv", index=False)
    return manifest


def write_mutation_profile_membership_audit(meta: pd.DataFrame) -> pd.DataFrame:
    """Distinguish confirmed zero-event samples from unverified profile membership."""
    selected = meta.loc[meta.analysisEligible, ["studyId", "sampleId"]].copy()
    assay = pd.read_parquet(PROCESSED / "sample_assay.parquet")
    assay_columns = [
        column
        for column in (
            "studyId", "sampleId", "assayType", "panelMetadataAvailable",
            "mutationProfileMembershipStatus", "callabilityEligible",
        )
        if column in assay
    ]
    selected = selected.merge(
        assay[assay_columns], on=["studyId", "sampleId"], how="left", validate="one_to_one"
    )
    mutation_pairs = (
        pd.read_parquet(PROCESSED / "mutations_dedup.parquet", columns=["studyId", "sampleId"])
        .drop_duplicates()
        .assign(hasObservedMutationRow=True)
    )
    selected = selected.merge(
        mutation_pairs, on=["studyId", "sampleId"], how="left", validate="one_to_one"
    )
    selected["hasObservedMutationRow"] = selected.hasObservedMutationRow.fillna(False)
    if "callabilityEligible" not in selected:
        selected["callabilityEligible"] = selected.panelMetadataAvailable.fillna(False)
    selected["callabilityEligible"] = selected.callabilityEligible.fillna(False)

    audit = (
        selected.groupby("studyId")
        .agg(
            nAnalysisSamples=("sampleId", "size"),
            nSamplesWithMutationRows=("hasObservedMutationRow", "sum"),
            nSamplesWithVerifiedAssayScope=("callabilityEligible", "sum"),
        )
        .reset_index()
    )
    audit["nSamplesWithoutMutationRows"] = (
        audit.nAnalysisSamples - audit.nSamplesWithMutationRows
    )
    audit["nSamplesWithUnverifiedAssayScope"] = (
        audit.nAnalysisSamples - audit.nSamplesWithVerifiedAssayScope
    )
    audit["studyHasZeroObservedMutationRows"] = audit.nSamplesWithMutationRows.eq(0)
    audit["zeroMutationInterpretation"] = np.select(
        [
            ~audit.studyHasZeroObservedMutationRows,
            audit.nSamplesWithUnverifiedAssayScope.eq(0),
        ],
        [
            "not a zero-event study",
            "zero observed events with verified mutation-profile membership and assay scope",
        ],
        default="zero observed events; one or more selected samples lack verified assay scope",
    )
    audit.to_csv(TABLES / "mutation_profile_membership_audit.csv", index=False)
    return audit


def write_diagnosis_discordance_audit(meta: pd.DataFrame) -> pd.DataFrame:
    """Tabulate strict one-person selections that span multiple recorded cancer families."""
    eligible = meta.loc[meta.tissueEligible].copy()
    family_counts = eligible.groupby("personKey").cancerFamilyCode.nunique()
    discordant_people = set(family_counts[family_counts.gt(1)].index)
    rows = eligible.loc[eligible.personKey.isin(discordant_people)].copy()
    keep = [
        column
        for column in (
            "personKey", "patientKey", "patientId", "studyId", "sampleId", "sourceSampleId",
            "sampleType", "sampleTypeGroup", "representativePriority", "cancerType",
            "cancerTypeDetailed", "oncotreeCode", "resolvedOncoTreeCode",
            "analysisCancerCode", "cancerFamilyCode", "cancerFamilyName",
            "taxonomyResolutionMethod", "selectedOnePerPatient", "analysisEligible",
        )
        if column in rows
    ]
    rows[keep].sort_values(
        ["personKey", "selectedOnePerPatient", "representativePriority", "studyId", "sampleId"],
        ascending=[True, False, True, True, True],
    ).to_csv(TABLES / "one_per_patient_diagnosis_discordance_audit.csv", index=False)

    summary = (
        rows.groupby("personKey")
        .agg(
            nEligibleSpecimens=("sampleId", "size"),
            nCancerFamilies=("cancerFamilyCode", "nunique"),
            cancerFamilies=("cancerFamilyCode", lambda values: ";".join(sorted(set(values)))),
            nStudies=("studyId", "nunique"),
        )
        .reset_index()
    )
    summary.to_csv(TABLES / "one_per_patient_diagnosis_discordance_summary.csv", index=False)
    return summary


def write_taxonomy_audits(meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write sample-level and aggregated source-to-frozen-taxonomy crosswalks."""
    row_columns = [
        column
        for column in (
            "studyId", "sampleId", "sourceSampleId", "patientId", "personKey",
            "cancerType", "cancerTypeDetailed", "oncotreeCode", "studyCancerType",
            "resolvedOncoTreeCode", "resolvedCancerTypeDetailed", "analysisCancerCode",
            "oncotreeMainType", "oncotreeTissue", "oncotreeRootCode", "oncotreePath",
            "cancerFamilyCode", "cancerFamilyName", "taxonomyResolutionMethod",
            "suppliedOncoTreeCodeCanonical", "detailedDiagnosisCandidateCode",
            "taxonomyCodeRelationship", "taxonomyOverrideApplied",
            "cancerFamilyResolutionMethod", "taxonomyVersion", "tissueEligible",
            "selectedOnePerPatient", "analysisEligible",
        )
        if column in meta
    ]
    meta[row_columns].to_csv(TABLES / "cancer_taxonomy_sample_audit.csv", index=False)

    descendant = meta.loc[
        meta.taxonomyCodeRelationship.eq(
            "detailed diagnosis is descendant of supplied code"
        )
    ]
    descendant[row_columns].to_csv(
        TABLES / "taxonomy_detailed_descendant_resolution_audit.csv", index=False
    )
    discordant = meta.loc[
        meta.taxonomyCodeRelationship.str.contains(
            "discordant|adjudication", case=False, na=False
        )
    ]
    discordant[row_columns].to_csv(
        TABLES / "taxonomy_discordant_source_fields_audit.csv", index=False
    )

    group_columns = [
        "cancerType", "cancerTypeDetailed", "oncotreeCode", "studyCancerType",
        "resolvedOncoTreeCode", "analysisCancerCode", "cancerFamilyCode",
        "cancerFamilyName", "taxonomyResolutionMethod", "cancerFamilyResolutionMethod",
        "taxonomyCodeRelationship", "taxonomyOverrideApplied", "taxonomyVersion",
    ]
    summary = (
        meta.groupby(group_columns, dropna=False)
        .agg(
            nCanonicalSpecimens=("sampleId", "size"),
            nTissueEligible=("tissueEligible", "sum"),
            nSelectedTumours=("analysisEligible", "sum"),
            nStudies=("studyId", "nunique"),
            studyIds=("studyId", lambda values: ";".join(sorted(set(values)))),
        )
        .reset_index()
        .sort_values(["nSelectedTumours", "nCanonicalSpecimens"], ascending=False)
    )
    summary.to_csv(TABLES / "cancer_taxonomy_resolution_summary.csv", index=False)

    selected = meta.loc[meta.analysisEligible].copy()
    lung_codes = ["LUAD", "LUSC", "NSCLC", "SCLC"]
    lung = selected.loc[selected.cancerFamilyCode.isin(lung_codes)]
    lung_summary = (
        lung.groupby(["cancerFamilyCode", "cancerFamilyName"], dropna=False)
        .agg(
            nSelectedTumours=("sampleId", "size"),
            nStudies=("studyId", "nunique"),
            studyIds=("studyId", lambda values: ";".join(sorted(set(values)))),
            nDetailedOncoTreeCodes=("analysisCancerCode", "nunique"),
            detailedOncoTreeCodes=(
                "analysisCancerCode", lambda values: ";".join(sorted(set(values)))
            ),
        )
        .reindex(lung_codes, level="cancerFamilyCode")
        .reset_index()
    )
    lung_summary.to_csv(TABLES / "lung_histology_taxonomy_summary.csv", index=False)
    (
        lung.groupby(["cancerFamilyCode", "cancerFamilyName", "studyId"], dropna=False)
        .agg(
            nSelectedTumours=("sampleId", "size"),
            detailedOncoTreeCodes=(
                "analysisCancerCode", lambda values: ";".join(sorted(set(values)))
            ),
            taxonomyResolutionMethods=(
                "taxonomyResolutionMethod", lambda values: ";".join(sorted(set(values)))
            ),
        )
        .reset_index()
        .sort_values(["cancerFamilyCode", "nSelectedTumours"], ascending=[True, False])
        .to_csv(TABLES / "lung_histology_study_counts.csv", index=False)
    )
    return summary, lung_summary


def main() -> None:
    panel = pd.read_csv(PROCESSED / "gene_panel.csv")
    meta = extract_sample_metadata()
    meta = build_curated_samples(meta)

    summary = (
        meta.assign(
            exclusionReason=meta.exclusionReason.fillna("Eligible tissue sample")
        )
        .groupby("exclusionReason")
        .size()
        .rename("nSamples")
        .reset_index()
        .sort_values("nSamples", ascending=False)
    )
    summary["pctCanonical"] = 100 * summary.nSamples / len(meta)
    summary.to_csv(TABLES / "cohort_exclusion_summary.csv", index=False)

    sensitivity, curated = write_sensitivity_tables(meta, panel)
    study_manifest = write_study_manifest(meta)
    profile_audit = write_mutation_profile_membership_audit(meta)
    diagnosis_audit = write_diagnosis_discordance_audit(meta)
    taxonomy_audit, lung_audit = write_taxonomy_audits(meta)
    make_figure(meta, sensitivity, curated)

    print("\n=== BIOSPECIMEN-AWARE COHORT ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\nCanonical samples                 : {len(meta):,}")
    print(f"Biospecimen-eligible tissue       : {meta.tissueEligible.sum():,}")
    print(f"One tissue sample per patient       : {meta.analysisEligible.sum():,}")
    print(f"Distinct constructed people         : {meta.loc[meta.analysisEligible, 'personKey'].nunique():,}")
    print(
        "People retaining >1 cancer family  : "
        f"{meta.loc[meta.analysisEligible & meta.personHasMultipleCancerFamilies, 'personKey'].nunique():,}"
    )
    print(f"Studies contributing primary cases: {study_manifest.contributesToPrimaryAnalysis.sum():,}")
    print(
        "Selected samples with unverified mutation-profile assay scope: "
        f"{int(profile_audit.nSamplesWithUnverifiedAssayScope.sum()):,}"
    )
    print(f"People with discordant recorded cancer families: {len(diagnosis_audit):,}")
    print("\nResolved lung-family counts:")
    print(lung_audit.to_string(index=False))
    print("\nTop curated assay-aware frequencies:")
    print(
        curated.head(15)[["gene", "nMutated", "nProfiled", "freqPct", "freqCiLowPct", "freqCiHighPct"]]
        .to_string(index=False, float_format=lambda x: f"{x:.2f}")
    )
    print("\nWrote curated cohort artefacts and cohort-selection diagnostics")


if __name__ == "__main__":
    main()
