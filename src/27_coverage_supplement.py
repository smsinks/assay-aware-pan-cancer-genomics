"""Supplementary Figure S1 -- cohort coverage and curation sensitivity.

Every eligible cancer group and every contributing study is represented by a point.
The ranked displays remain readable at publication width, while a matched prevalence
comparison documents the effect of representative-tissue selection on sentinel genes.
The accompanying source tables preserve exact rank-to-name mappings, assay/callability
quantities and prevalence estimates. This figure is deliberately descriptive; it
documents inclusion, curation and denominator structure rather than selecting cohorts
on the basis of observed mutation events.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from config import FIGURES, PROCESSED, TABLES
from plot_style import COLORS, apply as apply_style, figsize, panel_label, save_figure


COMPLETE = TABLES.parent / "complete_tables"
SOURCE = TABLES.parent / "source_data"
SUPP = FIGURES / "supplementary"

ASSAY_COLORS = {
    "WES/WGS only": COLORS["blue"],
    "Targeted panel only": COLORS["purple"],
    "Mixed WES/WGS + targeted panel": COLORS["green"],
}

LEGEND_BOX = {
    "frameon": True,
    "fancybox": True,
    "framealpha": 0.96,
    "facecolor": "#FAFAFA",
    "edgecolor": COLORS["light_grey"],
    "borderpad": 0.45,
}

# Selective labels preserve the complete ranked distributions while identifying
# representative high-volume cohorts, diagnostically useful tail categories and
# notable study-level callability outliers. Offsets are in display points and are
# intentionally staggered to avoid collisions at the final 180-mm figure width.
PANEL_A_LABELS = {
    "BRCA": ("BRCA", (5, 12), "left"),
    "COADREAD": ("COADREAD", (6, 5), "left"),
    "LUAD": ("LUAD", (7, -2), "left"),
    "UCEC": ("UCEC", (6, -10), "left"),
    "LUSC": ("LUSC", (6, 6), "left"),
    "SCLC": ("SCLC", (6, 6), "left"),
    "ODG": ("ODG", (6, 6), "left"),
    "GINET": ("GINET", (6, 7), "left"),
    "VULVA": ("VULVA", (-6, 7), "right"),
    "PERITONEUM": ("PERITONEUM", (-6, 6), "right"),
    "MAST": ("MAST", (-6, 5), "right"),
    "NHL": ("NHL", (-6, 3), "right"),
    "LUNG": ("LUNG", (-6, -3), "right"),
}

PANEL_C_LABELS = {
    "msk_impact_50k_2026": ("MSK-IMPACT 50K", (8, 8), "left"),
    "pancan_pcawg_2020": ("PCAWG", (8, 7), "left"),
    "rectal_msk_2022": ("Rectal MSK", (8, 7), "left"),
    "acyc_mda_2015": ("ACC MDA", (-8, 8), "right"),
    "nfib_ctf_biobank_2025": ("NFib biobank", (8, 7), "left"),
    "ucec_tcga_pub": ("UCEC TCGA", (-8, 7), "right"),
}

PANEL_D_LABELS = {
    "pancan_pcawg_2020": ("PCAWG", (7, -8), "left"),
    "mds_iwg_2022": ("MDS IWG", (7, 7), "left"),
    "rectal_msk_2022": ("Rectal MSK", (7, 7), "left"),
    "lusc_cptac_2021": ("LUSC CPTAC", (-7, -8), "right"),
    "ucec_cptac_2020": ("UCEC CPTAC", (7, 7), "left"),
    "blca_mskcc_solit_2012": ("MSK bladder", (-7, 7), "right"),
    "nsclc_unito_2016": ("NSCLC Turin", (7, 7), "left"),
    "nfib_ctf_biobank_2025": ("NFib", (-7, 7), "right"),
}


def cancer_coverage() -> pd.DataFrame:
    complete = pd.read_csv(COMPLETE / "curated_cancer_gene_prevalence_complete.csv")
    samples = pd.read_parquet(PROCESSED / "analysis_samples_curated.parquet")
    samples = samples[samples.analysisEligible]

    fixed = complete.groupby("cancerGroup", as_index=False).agg(
        cancerGroupLabel=("cancerGroupLabel", "first"),
        nEligibleSamples=("nEligibleSamples", "max"),
        nWesWgsSamples=("nWesWgsSamples", "max"),
        nWesWgsDocumentedSamples=("nWesWgsDocumentedSamples", "max"),
        nWesWgsAssumedSamples=("nWesWgsAssumedSamples", "max"),
        nTargetedSamples=("nTargetedSamples", "max"),
        nCallableSampleGenePairs=("nCallable", "sum"),
        nUnassayedSampleGenePairs=("nUnassayed", "sum"),
        nMutatedSampleGenePairs=("nMutated", "sum"),
        nGenesMutationObserved=("nMutated", lambda values: int((values > 0).sum())),
    )
    studies = (
        samples.groupby("broadCancerCode").studyId.nunique().rename("nStudies").reset_index()
        .rename(columns={"broadCancerCode": "cancerGroup"})
    )
    fixed = fixed.merge(studies, on="cancerGroup", how="left", validate="one_to_one")
    fixed["nPotentialSampleGenePairs"] = fixed.nEligibleSamples * complete.gene.nunique()
    fixed["callableFractionPct"] = (
        100 * fixed.nCallableSampleGenePairs / fixed.nPotentialSampleGenePairs.clip(lower=1)
    )
    fixed["targetedFractionPct"] = (
        100 * fixed.nTargetedSamples / fixed.nEligibleSamples.clip(lower=1)
    )
    fixed["assayStratum"] = np.select(
        [fixed.nTargetedSamples.eq(0), fixed.nWesWgsSamples.eq(0)],
        ["WES/WGS only", "Targeted panel only"],
        default="Mixed WES/WGS + targeted panel",
    )
    fixed = fixed.sort_values(["nEligibleSamples", "cancerGroup"], ascending=[False, True]).reset_index(drop=True)
    fixed.insert(0, "rank", np.arange(1, len(fixed) + 1))
    return fixed


def study_coverage() -> pd.DataFrame:
    study = pd.read_csv(COMPLETE / "curated_study_completeness.csv")
    study["callableFractionPct"] = (
        100 * study.nCallableSampleGenePairs / study.nPotentialSampleGenePairs.clip(lower=1)
    )
    study["zeroEventStudy"] = study.nMutationRecords.eq(0)
    study = study.sort_values(["nEligibleSamples", "studyId"], ascending=[False, True]).reset_index(drop=True)
    study.insert(0, "rank", np.arange(1, len(study) + 1))
    return study


def prevalence_sensitivity() -> pd.DataFrame:
    """Return sentinel-gene prevalence before and after representative-tissue selection."""
    sensitivity = pd.read_csv(TABLES / "cohort_sensitivity.csv")
    all_frequency = sensitivity.loc[
        sensitivity.cohortDefinition.eq("All canonical samples"), ["gene", "freqPct"]
    ]
    curated_frequency = sensitivity.loc[
        sensitivity.cohortDefinition.eq("One tissue sample per patient"),
        ["gene", "freqPct", "nMutated"],
    ]
    shift = all_frequency.merge(
        curated_frequency, on="gene", suffixes=("Canonical", "Curated")
    )
    shift["delta"] = shift.freqPctCurated - shift.freqPctCanonical
    sentinels = [
        "TP53", "KRAS", "PIK3CA", "APC", "DNMT3A", "PPM1D", "MUC16", "LRP1B", "PLEC"
    ]
    result = shift.loc[shift.gene.isin(sentinels)].sort_values("freqPctCurated")
    if set(result.gene) != set(sentinels):
        missing = sorted(set(sentinels) - set(result.gene))
        raise AssertionError(f"Supplementary Figure 1b sentinel genes are absent: {missing}")
    return result


def scatter_by_assay(ax: plt.Axes, frame: pd.DataFrame, y: str, *, study: bool) -> None:
    for assay, group in frame.groupby("assayStratum", sort=False):
        color = ASSAY_COLORS.get(assay, COLORS["grey"])
        sizes = np.clip(np.sqrt(group.nEligibleSamples) * (1.15 if study else 1.45), 5, 36)
        if study:
            observed = ~group.zeroEventStudy
            ax.scatter(
                group.loc[observed, "rank"], group.loc[observed, y],
                s=sizes[observed], color=color, alpha=0.62, edgecolors="none",
            )
            ax.scatter(
                group.loc[~observed, "rank"], group.loc[~observed, y],
                s=np.maximum(sizes[~observed], 13), facecolors="white", edgecolors=color,
                linewidths=0.65,
            )
        else:
            ax.scatter(group["rank"], group[y], s=sizes, color=color, alpha=0.68, edgecolors="none")


def annotate_selected(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    key_column: str,
    y_column: str,
    labels: dict[str, tuple[str, tuple[float, float], str]],
    fontsize: float = 3.75,
) -> None:
    """Add compact, leader-linked labels for prespecified ranked points."""
    indexed = frame.set_index(key_column, drop=False)
    missing = sorted(set(labels) - set(indexed.index.astype(str)))
    if missing:
        raise AssertionError(f"Requested coverage labels are absent: {missing}")
    for key, (display_label, offset, horizontal_alignment) in labels.items():
        row = indexed.loc[key]
        if isinstance(row, pd.DataFrame):
            raise AssertionError(f"Coverage label key is not unique: {key}")
        ax.annotate(
            display_label,
            (float(row["rank"]), float(row[y_column])),
            xytext=offset,
            textcoords="offset points",
            ha=horizontal_alignment,
            va="center",
            fontsize=fontsize,
            color=COLORS["black"],
            arrowprops={
                "arrowstyle": "-",
                "color": COLORS["grey"],
                "lw": 0.35,
                "shrinkA": 1.5,
                "shrinkB": 1.5,
            },
            annotation_clip=False,
            zorder=6,
        )


def main() -> None:
    apply_style()
    SOURCE.mkdir(parents=True, exist_ok=True)
    SUPP.mkdir(parents=True, exist_ok=True)
    cancers = cancer_coverage()
    studies = study_coverage()
    prevalence_shift = prevalence_sensitivity()
    sample_audit = pd.read_parquet(PROCESSED / "analysis_samples_curated.parquet")
    selected_samples = sample_audit.loc[sample_audit.analysisEligible]
    expected_cancers = selected_samples.broadCancerCode.nunique()
    expected_studies = selected_samples.studyId.nunique()

    if len(cancers) != expected_cancers:
        raise AssertionError(f"Expected {expected_cancers} cancer families, found {len(cancers)}")
    if len(studies) != expected_studies:
        raise AssertionError(f"Expected {expected_studies} studies, found {len(studies)}")
    if cancers.cancerGroup.duplicated().any() or studies.studyId.duplicated().any():
        raise AssertionError("Coverage ranks are not unique")

    fig, axes = plt.subplots(2, 2, figsize=figsize(180, 150), gridspec_kw={"hspace": 0.42, "wspace": 0.36})
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    scatter_by_assay(ax_a, cancers, "nEligibleSamples", study=False)
    ax_a.set_yscale("log")
    ax_a.set_xlabel("Cancer-group rank by eligible cases")
    ax_a.set_ylabel("Eligible representative cases (log scale)")
    annotate_selected(
        ax_a,
        cancers,
        key_column="cancerGroup",
        y_column="nEligibleSamples",
        labels=PANEL_A_LABELS,
        fontsize=3.8,
    )
    ax_a.text(0.98, 0.96, f"all {len(cancers)} cancer families", transform=ax_a.transAxes, ha="right", va="top", fontsize=5.1)
    panel_label(ax_a, "a")

    ordered = prevalence_shift.sort_values("freqPctCurated").reset_index(drop=True)
    label_y = ordered.freqPctCurated.to_numpy(float).copy()
    for index in range(1, len(label_y)):
        label_y[index] = max(label_y[index], label_y[index - 1] + 1.15)
    for row, y_text in zip(ordered.itertuples(index=False), label_y):
        color = COLORS["blue"] if row.delta >= 0 else COLORS["vermillion"]
        ax_b.plot(
            [0, 1], [row.freqPctCanonical, row.freqPctCurated],
            color=color, alpha=0.75, lw=0.9,
        )
        ax_b.scatter(
            [0, 1], [row.freqPctCanonical, row.freqPctCurated],
            color=color, s=13, zorder=3,
        )
        ax_b.plot(
            [1.01, 1.075], [row.freqPctCurated, y_text],
            color=COLORS["grey"], lw=0.45,
        )
        ax_b.text(1.085, y_text, row.gene, va="center", fontsize=4.7)
    ax_b.set_xlim(-0.12, 1.35)
    ax_b.set_xticks([0, 1], ["Canonical", "Curated tissue\n(one/patient)"])
    ax_b.set_ylabel("Assay-aware prevalence (%)")
    shift_legend = ax_b.legend(
        handles=[
            Line2D(
                [0], [0], color=COLORS["blue"], lw=1.2, marker="o", markersize=3.2,
                label="Increased after curation",
            ),
            Line2D(
                [0], [0], color=COLORS["vermillion"], lw=1.2, marker="o", markersize=3.2,
                label="Decreased after curation",
            ),
        ],
        frameon=True,
        fancybox=True,
        framealpha=0.96,
        facecolor="#FAFAFA",
        edgecolor=COLORS["light_grey"],
        fontsize=4.2,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.50, 1.015),
        borderaxespad=0,
        borderpad=0.4,
        handlelength=1.5,
        columnspacing=0.9,
    )
    panel_label(ax_b, "b")

    scatter_by_assay(ax_c, studies, "nEligibleSamples", study=True)
    ax_c.set_yscale("log")
    ax_c.set_xlabel("Study rank by eligible cases")
    ax_c.set_ylabel("Eligible representative cases (log scale)")
    annotate_selected(
        ax_c,
        studies,
        key_column="studyId",
        y_column="nEligibleSamples",
        labels=PANEL_C_LABELS,
        fontsize=3.65,
    )
    ax_c.text(0.98, 0.96, f"all {len(studies)} studies", transform=ax_c.transAxes, ha="right", va="top", fontsize=5.1)
    panel_label(ax_c, "c")

    scatter_by_assay(ax_d, studies, "callableFractionPct", study=True)
    ax_d.set_xlabel("Study rank by eligible cases")
    ax_d.set_ylabel("Callable sample–gene pairs (%)")
    ax_d.set_ylim(-2, 103)
    annotate_selected(
        ax_d,
        studies,
        key_column="studyId",
        y_column="callableFractionPct",
        labels=PANEL_D_LABELS,
        fontsize=3.65,
    )
    panel_label(ax_d, "d")

    handles = [
        Line2D([0], [0], marker="o", ls="", markerfacecolor=color, markeredgecolor="none", label=label, markersize=4.2)
        for label, color in ASSAY_COLORS.items()
    ]
    handles.append(Line2D([0], [0], marker="o", ls="", markerfacecolor="white", markeredgecolor=COLORS["grey"], label="zero-event study", markersize=4.2))
    fig.legend(
        handles=handles,
        **LEGEND_BOX,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=4,
        fontsize=4.8,
        columnspacing=1.15,
        handletextpad=0.45,
    )
    fig.subplots_adjust(left=0.12, right=0.985, top=0.94, bottom=0.12)

    cancers["panelALabel"] = cancers.cancerGroup.map(
        {key: value[0] for key, value in PANEL_A_LABELS.items()}
    )
    studies["panelCLabel"] = studies.studyId.map(
        {key: value[0] for key, value in PANEL_C_LABELS.items()}
    )
    studies["panelDLabel"] = studies.studyId.map(
        {key: value[0] for key, value in PANEL_D_LABELS.items()}
    )
    cancers.to_csv(SOURCE / "figureS1_panel_a_cancer_coverage.csv", index=False)
    prevalence_shift.to_csv(SOURCE / "figureS1_panel_b_prevalence_shift.csv", index=False)
    studies.to_csv(SOURCE / "figureS1_panels_c_d_study_coverage.csv", index=False)
    save_figure(fig, SUPP / "figureS1_complete_coverage")
    plt.close(fig)
    print(f"Wrote Supplementary Figure S1 with {len(cancers)} cancer groups and {len(studies)} studies")


if __name__ == "__main__":
    main()
