"""Stage 23 -- integrated Nature-style main Figures 1--5.

This stage consolidates the useful content of the legacy Figures 1--6 and the later
curation Figures 11--13 into five coherent, submission-scaled figures.  Every display
subset is backed by regenerated complete cancer-group-by-gene and study-by-gene tables.
Main-panel filtering is therefore a legibility choice, never an implicit deletion of
zero-event or unassayed groups.

Figures
-------
1. Cancer-family assay composition, family-level callability and study completeness.
2. Driver prevalence, mutation-level evidence and hotspot context.
3. Integrated gene-by-cancer landscape with aligned annotation tracks and cell values.
4. Exactly four mutation-only, callability-aware oncoprints.
5. Cancer-specific, jointly callable and context-conditioned mutation associations.
"""
from __future__ import annotations

import os
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import FIGURES, PROCESSED, TABLES
from nature_style import (
    COLORS,
    aligned_panel_labels,
    apply as apply_style,
    figsize,
    panel_label,
    save_figure,
)

warnings.filterwarnings("ignore")

COMPLETE = TABLES.parent / "submission_tables"
SOURCE = TABLES.parent / "source_data"
SUPP = FIGURES / "supplementary"

ROLE_COLORS = {
    "Oncogenes": COLORS["vermillion"],
    "TSGs": COLORS["blue"],
    "Oncogene/TSG": COLORS["purple"],
    "Other": COLORS["grey"],
}

PATHWAY_COLORS = {
    "RTK–RAS": "#0072B2",
    "PI3K–AKT": "#D55E00",
    "TP53": "#CC79A7",
    "Cell cycle": "#009E73",
    "WNT/β-catenin": "#E69F00",
    "NOTCH": "#56B4E9",
    "TGF-β": "#8C6D31",
    "Hippo": "#6A51A3",
    "Not assigned": "#D9D9D9",
}

LEGEND_EDGE = "#C9CED3"
NOTE_BBOX = {
    "boxstyle": "round,pad=0.34,rounding_size=0.8",
    "facecolor": "white",
    "edgecolor": LEGEND_EDGE,
    "linewidth": 0.55,
    "alpha": 0.96,
}


def _finish_legend(legend, *, pad: float = 0.32) -> None:
    """Apply the same restrained, rounded key treatment across main figures."""
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor(LEGEND_EDGE)
    frame.set_linewidth(0.55)
    frame.set_alpha(0.96)
    frame.set_boxstyle(f"round,pad={pad},rounding_size=0.8")


def setup_dirs() -> None:
    SOURCE.mkdir(parents=True, exist_ok=True)
    SUPP.mkdir(parents=True, exist_ok=True)


def _cancer_coverage_summary() -> pd.DataFrame:
    """Return one frozen callability row for every reviewed cancer family."""
    complete = pd.read_csv(COMPLETE / "curated_cancer_gene_prevalence_complete.csv")
    fixed = complete.groupby("cancerGroup", as_index=False).agg(
        cancerGroupLabel=("cancerGroupLabel", "first"),
        nEligibleSamples=("nEligibleSamples", "max"),
        nWesWgsSamples=("nWesWgsSamples", "max"),
        nTargetedSamples=("nTargetedSamples", "max"),
        nCallableSampleGenePairs=("nCallable", "sum"),
        nUnassayedSampleGenePairs=("nUnassayed", "sum"),
        nMutatedSampleGenePairs=("nMutated", "sum"),
    )
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
        default="Mixed WES/WGS and targeted panel",
    )
    fixed = fixed.sort_values(
        ["nEligibleSamples", "cancerGroup"], ascending=[False, True]
    ).reset_index(drop=True)
    fixed.insert(0, "rank", np.arange(1, len(fixed) + 1))
    return fixed


def _annotate_cancer_callability(ax: plt.Axes, cancers: pd.DataFrame) -> None:
    """Label clinically informative landmarks using fixed non-overlapping anchors."""
    label_positions = {
        "BRCA": (8, 51),
        "COADREAD": (8, 66),
        "LUAD": (13, 57),
        "SARC": (12, 38),
        "UCEC": (17, 46),
        "MDS": (16, 9),
        "AML": (21, 62),
        "HCC": (22, 98),
        "CLL": (25, 89),
        "LUSC": (29, 72),
        "NSCLC": (36, 51),
        "USARC": (39, 27),
        "GCT": (49, 34),
        "MDS_MPN": (49, 18),
        "PCPG": (60, 95),
        "ADRENAL_GLAND": (75, 40),
        "NHL": (79, 28),
    }
    indexed = cancers.set_index("cancerGroup")
    missing = sorted(set(label_positions) - set(indexed.index))
    if missing:
        raise AssertionError(f"Figure 1b label families are absent: {missing}")
    for cancer, (text_x, text_y) in label_positions.items():
        row = indexed.loc[cancer]
        ax.annotate(
            cancer,
            (row["rank"], row["callableFractionPct"]),
            xytext=(text_x, text_y),
            textcoords="data",
            ha="left" if text_x >= row["rank"] else "right",
            va="center",
            fontsize=4.05,
            color=COLORS["black"],
            arrowprops={"arrowstyle": "-", "color": COLORS["grey"], "lw": 0.35},
        )


FIGURE1C_LABEL_SPECS = {
    "lusc_cptac_2021": ("LUSC", (-26, -4)),
    "luad_mskcc_2023_met_organotropism": ("LUAD", (8, 7)),
    "ucec_ccr_msk_2022": ("UCEC", (-27, 8)),
    "mel_mskimpact_2020": ("SKCM", (5, -11)),
    "crc_eo_2020": ("COADREAD", (7, 7)),
    "sarc_mskcc": ("SARC", (-24, 8)),
    "mds_iwg_2022": ("MDS", (7, 7)),
}


def _annotate_study_callability(ax: plt.Axes, studies: pd.DataFrame) -> None:
    """Label selected single-cancer study landmarks without obscuring the point cloud."""
    # Panel c is study-level.  Restricting callouts to studies containing one reviewed
    # cancer family prevents a heterogeneous portal cohort from being presented as if
    # it represented a single cancer.  Fixed offsets were reviewed at the final 180-mm
    # canvas size and distribute labels across low-, intermediate- and high-callability
    # regions.
    indexed = studies.set_index("studyId")
    missing = sorted(set(FIGURE1C_LABEL_SPECS) - set(indexed.index))
    if missing:
        raise AssertionError(f"Figure 1c labelled studies are absent: {missing}")
    for study_id, (cancer, offset) in FIGURE1C_LABEL_SPECS.items():
        row = indexed.loc[study_id]
        if int(row["nCancerGroups"]) != 1 or row["primaryCancerGroup"] != cancer:
            raise AssertionError(
                f"Figure 1c label {cancer} does not map to a single-family {study_id} row"
            )
        ax.annotate(
            cancer,
            (row["nEligibleSamples"], 100 * row["callableFraction"]),
            xytext=offset,
            textcoords="offset points",
            ha="left" if offset[0] >= 0 else "right",
            va="center",
            fontsize=4.05,
            color=COLORS["black"],
            arrowprops={"arrowstyle": "-", "color": COLORS["grey"], "lw": 0.32},
            zorder=4,
        )


def figure1() -> None:
    apply_style()
    samples = pd.read_parquet(PROCESSED / "analysis_samples_curated.parquet")
    curated = samples[samples.analysisEligible].copy()
    studies = pd.read_csv(COMPLETE / "curated_study_completeness.csv")
    assay = pd.read_parquet(PROCESSED / "sample_assay.parquet").merge(
        curated[["sampleId", "studyId", "broadCancerCode"]],
        on=["sampleId", "studyId"],
        how="inner",
        validate="one_to_one",
    )

    # The assay-by-family overview benefits from the full height of the left column;
    # the family- and study-level callability summaries share the right column.  Biospecimen-exclusion
    # counts remain available in Supplementary Data 1 and the Methods, as requested.
    fig = plt.figure(figsize=figsize(180, 148))
    gs = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[0.92, 1.08],
        hspace=0.52,
        wspace=0.40,
    )
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    assay["assayLabel"] = assay.assayType.replace(
        {
            "WES/WGS (assumed; no panel metadata)": "Assumed WES/WGS",
            "Unverified mutation-profile membership": "Unverified profile",
        }
    )
    # Show a broader cross-section of the final reviewed taxonomy while retaining a
    # legible main-panel summary; Supplementary Data 1 retains all 89 families.
    top_cancers = curated.broadCancerCode.value_counts().head(24).index.tolist()
    comp = (
        assay[assay.broadCancerCode.isin(top_cancers)]
        .groupby(["broadCancerCode", "assayLabel"])
        .size()
        .rename("nSamples")
        .reset_index()
    )
    wide = comp.pivot(index="broadCancerCode", columns="assayLabel", values="nSamples").fillna(0)
    wide = wide.reindex(top_cancers[::-1])
    left = np.zeros(len(wide))
    assay_colors = {
        "WES/WGS": COLORS["blue"],
        "Assumed WES/WGS": COLORS["sky"],
        "Unverified profile": COLORS["grey"],
        "Targeted panel": COLORS["purple"],
    }
    # Do not advertise a zero-count assay class in the legend.  The final cohort has
    # unverified-profile specimens but no longer treats them as assumed WES/WGS.
    assay_order = [label for label in assay_colors if label in wide.columns]
    for label in assay_order:
        color = assay_colors[label]
        values_array = wide.get(label, pd.Series(0, index=wide.index)).to_numpy()
        ax_a.barh(np.arange(len(wide)), values_array, left=left, color=color, label=label)
        left += values_array
    ax_a.set_yticks(np.arange(len(wide)), wide.index, fontsize=5.3)
    ax_a.set_xlabel("Representative cases")
    assay_legend = ax_a.legend(
        title="Sequencing assay",
        frameon=True,
        fancybox=True,
        fontsize=4.5,
        title_fontsize=4.8,
        ncol=2,
        loc="lower right",
        borderpad=0.45,
        handlelength=1.6,
        columnspacing=0.9,
    )
    _finish_legend(assay_legend)
    # Match the absolute vertical position of A to the upper-right B label even
    # though panel A spans both GridSpec rows.
    panel_label(ax_a, "a", x=-0.17, y=1.015)
    comp.to_csv(SOURCE / "figure1_panel_a_assay_by_cancer.csv", index=False)

    cancers = _cancer_coverage_summary()
    if len(cancers) != curated.broadCancerCode.nunique():
        raise AssertionError(
            f"Figure 1b expected {curated.broadCancerCode.nunique()} cancer families, "
            f"found {len(cancers)}"
        )
    assay_categories = {
        "WES/WGS only": COLORS["blue"],
        "Targeted panel only": COLORS["purple"],
        "Mixed WES/WGS and targeted panel": COLORS["green"],
    }
    for label, group in cancers.groupby("assayStratum", sort=False):
        ax_b.scatter(
            group["rank"],
            group["callableFractionPct"],
            s=np.clip(np.sqrt(group.nEligibleSamples) * 1.45, 6, 38),
            color=assay_categories.get(label, COLORS["grey"]),
            alpha=0.70,
            edgecolors="white",
            linewidths=0.22,
            zorder=2,
        )
    _annotate_cancer_callability(ax_b, cancers)
    ax_b.set_xlim(-2, len(cancers) + 4)
    ax_b.set_ylim(-2, 103)
    ax_b.set_xlabel("Cancer-family rank by eligible cases")
    ax_b.set_ylabel("Assay-covered sample–gene pairs (%)")
    ax_b.text(
        0.98,
        0.035,
        f"all {len(cancers)} cancer families\npoint area indicates eligible cases",
        transform=ax_b.transAxes,
        ha="right",
        va="bottom",
        fontsize=4.3,
        bbox=NOTE_BBOX,
    )
    panel_label(ax_b, "b", x=-0.17, y=1.04)
    cancers.to_csv(SOURCE / "figure1_panel_b_cancer_callability.csv", index=False)

    study_plot = studies.copy()
    study_plot["callableFraction"] = study_plot.nCallableSampleGenePairs / study_plot.nPotentialSampleGenePairs.clip(lower=1)
    for label, group in study_plot.groupby("assayStratum"):
        color = assay_categories.get(label, COLORS["grey"])
        zero = group.nMutationRecords.eq(0)
        ax_c.scatter(
            group.loc[~zero, "nEligibleSamples"],
            100 * group.loc[~zero, "callableFraction"],
            s=np.clip(np.sqrt(group.loc[~zero, "nEligibleSamples"]) * 1.4, 8, 42),
            color=color,
            alpha=0.62,
            edgecolors="none",
            label=label,
        )
        ax_c.scatter(
            group.loc[zero, "nEligibleSamples"],
            100 * group.loc[zero, "callableFraction"],
            s=20,
            facecolor="white",
            edgecolor=color,
            linewidth=0.7,
        )
    _annotate_study_callability(ax_c, study_plot)
    ax_c.set_xscale("log")
    ax_c.set_xlabel("Eligible cases per contributing study")
    ax_c.set_ylabel("Assay-covered sample–gene pairs (%)")
    assay_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=4, color=color, label=label)
        for label, color in assay_categories.items()
    ]
    assay_handles.append(
        Line2D(
            [0], [0], marker="o", linestyle="", markersize=4, markerfacecolor="white",
            markeredgecolor=COLORS["grey"], label=f"No mutation events (n={int(study_plot.nMutationRecords.eq(0).sum())})",
        )
    )
    study_legend = ax_c.legend(
        handles=assay_handles,
        title=f"Contributing studies (n={len(study_plot)})",
        frameon=True,
        fancybox=True,
        fontsize=4.3,
        title_fontsize=4.7,
        ncol=2,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.015),
        borderaxespad=0,
        borderpad=0.48,
        handletextpad=0.6,
        columnspacing=0.8,
    )
    _finish_legend(study_legend)
    panel_label(ax_c, "c", x=-0.17, y=1.04)
    study_plot["displayCancerLabel"] = study_plot.studyId.map(
        {study_id: spec[0] for study_id, spec in FIGURE1C_LABEL_SPECS.items()}
    ).fillna("")
    study_plot.to_csv(SOURCE / "figure1_panel_c_study_completeness.csv", index=False)

    fig.subplots_adjust(left=0.14, right=0.985, top=0.955, bottom=0.105)
    save_figure(fig, FIGURES / "figure1_cohort_callability")
    plt.close(fig)


def _norm_aa(value) -> str:
    if not isinstance(value, str):
        return ""
    value = value.strip()
    return value[2:] if value.startswith("p.") else value


def _hotspot_context() -> tuple[pd.DataFrame, pd.DataFrame]:
    hotspots = pd.read_csv(TABLES / "mutation_hotspots_curated.csv")
    samples = pd.read_parquet(PROCESSED / "analysis_samples_curated.parquet")
    samples = samples[samples.analysisEligible][["sampleId", "broadCancerCode"]]
    panel = pd.read_csv(PROCESSED / "gene_panel.csv")
    ent2sym = panel.set_index("entrezGeneId").hugoSymbol.to_dict()
    mutations = pd.read_parquet(PROCESSED / "mutations_curated.parquet")
    mutations["gene"] = mutations.entrezGeneId.map(ent2sym)
    mutations["aa"] = mutations.proteinChange.map(_norm_aa)
    wanted = set(zip(hotspots.gene, hotspots.aa))
    mutation_pairs = list(zip(mutations.gene, mutations.aa))
    keep = np.fromiter((pair in wanted for pair in mutation_pairs), dtype=bool, count=len(mutation_pairs))
    context = (
        mutations.loc[keep, ["sampleId", "gene", "aa"]]
        .drop_duplicates()
        .merge(samples, on="sampleId", how="inner")
    )
    counts = (
        context.groupby(["gene", "aa", "broadCancerCode"])["sampleId"]
        .nunique()
        .rename("nSamples")
        .reset_index()
    )
    totals = counts.groupby("broadCancerCode").nSamples.sum().nlargest(11).index.tolist()
    row_order = [f"{row.gene} {row.aa}" for row in hotspots.itertuples(index=False)]
    counts["hotspot"] = counts.gene + " " + counts.aa
    matrix = (
        counts[counts.broadCancerCode.isin(totals)]
        .pivot(index="hotspot", columns="broadCancerCode", values="nSamples")
        .fillna(0)
        .reindex(index=row_order, columns=totals)
    )
    return counts, matrix


def figure2() -> None:
    apply_style()
    prevalence = pd.read_csv(TABLES / "gene_frequencies_curated.csv")
    evidence = pd.read_csv(TABLES / "cmc_evidence_curated.csv")
    standard = pd.read_csv(TABLES / "assay_frequency_standardized.csv")
    hotspot_counts, hotspot_matrix = _hotspot_context()

    fig = plt.figure(figsize=figsize(180, 176))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[0.92, 1.08], hspace=0.43, wspace=0.39)
    ax_a = fig.add_subplot(gs[0, 0]); ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0]); ax_d = fig.add_subplot(gs[1, 1])

    top = prevalence.head(24).sort_values("freqPct")
    y = np.arange(len(top))
    xerr = np.vstack([top.freqPct - top.freqCiLowPct, top.freqCiHighPct - top.freqPct])
    colors = [ROLE_COLORS.get(role, COLORS["grey"]) for role in top.roleInCancer]
    ax_a.errorbar(top.freqPct, y, xerr=xerr, fmt="none", ecolor=COLORS["grey"], elinewidth=0.65, capsize=1.3)
    ax_a.scatter(top.freqPct, y, c=colors, s=18, edgecolor="white", lw=0.3, zorder=3)
    ax_a.set_yticks(y, top.gene)
    ax_a.set_xlabel("Assay-aware prevalence (%, Wilson 95% CI)")
    for yy, callable_pct in zip(y, top.pctCuratedSamplesProfilingGene):
        ax_a.text(top.freqCiHighPct.max() * 1.04, yy, f"{callable_pct:.0f}% assay-covered", va="center", fontsize=4.0, color=COLORS["grey"])
    ax_a.set_xlim(0, top.freqCiHighPct.max() * 1.34)
    role_labels = {
        "Oncogenes": "Oncogene",
        "TSGs": "Tumour suppressor",
        "Oncogene/TSG": "Dual role",
        "Other": "Other/unknown",
    }
    handles = [
        Line2D([0], [0], marker="o", ls="", color=color, label=role_labels[role], markersize=4)
        for role, color in ROLE_COLORS.items()
    ]
    role_legend = ax_a.legend(
        handles=handles,
        title="Gene role",
        frameon=True,
        fancybox=True,
        fontsize=4.0,
        title_fontsize=4.5,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.15),
        ncol=4,
        borderaxespad=0,
        borderpad=0.4,
        handletextpad=0.45,
        columnspacing=0.65,
    )
    _finish_legend(role_legend)
    panel_label(ax_a, "a", x=-0.17, y=1.04)
    top.to_csv(SOURCE / "figure2_panel_a_prevalence.csv", index=False)

    merged = prevalence.merge(evidence, on="gene", how="inner")
    plot = merged[(merged.nMutationRecords >= 40) & merged.freqPct.gt(0)]
    sc = ax_b.scatter(
        plot.freqPct,
        plot.pctTierMatched,
        s=np.clip(np.sqrt(plot.nMutationRecords) * 1.1, 5, 34),
        c=plot.pctCuratedSamplesProfilingGene,
        cmap="viridis",
        vmin=30,
        vmax=100,
        alpha=0.72,
        edgecolor="white",
        linewidth=0.25,
    )
    label_genes = {"TP53", "KRAS", "PIK3CA", "TRRAP", "MUC16", "CSMD3"}
    offsets = {
        "TP53": (-23, -9), "KRAS": (3, 3), "PIK3CA": (-23, -8),
        "TRRAP": (3, -10), "MUC16": (3, 3), "CSMD3": (-24, 3),
    }
    for row in plot[plot.gene.isin(label_genes)].itertuples(index=False):
        ax_b.annotate(row.gene, (row.freqPct, row.pctTierMatched), xytext=offsets.get(row.gene, (3, 3)), textcoords="offset points", fontsize=4.6)
    cb = fig.colorbar(sc, ax=ax_b, fraction=0.036, pad=0.02)
    cb.set_label("Assay-covered cases (%)", fontsize=5.1)
    size_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="", markerfacecolor="#B8BDC2",
            markeredgecolor="white", markeredgewidth=0.3,
            markersize=np.sqrt(np.clip(np.sqrt(records) * 1.1, 5, 34)),
            label=label,
        )
        for records, label in ((100, "100"), (1000, "≥1,000"))
    ]
    size_legend = ax_b.legend(
        handles=size_handles,
        title="Mutation records",
        frameon=True,
        fancybox=True,
        fontsize=4.2,
        title_fontsize=4.6,
        loc="lower right",
        bbox_to_anchor=(0.975, 0.045),
        borderpad=0.42,
        handletextpad=0.55,
    )
    _finish_legend(size_legend)
    ax_b.set_xlabel("Assay-aware prevalence (%)")
    ax_b.set_ylabel("Mutation records matching CMC tier 1–3 (%)")
    panel_label(ax_b, "b", x=-0.12, y=1.04)
    merged.to_csv(SOURCE / "figure2_panel_b_cmc.csv", index=False)

    compare = standard[standard.nCommonCancers >= 2].merge(prevalence[["gene", "nMutated"]], on="gene", how="left")
    sc_c = ax_c.scatter(
        compare.freqWesStandardizedPct,
        compare.freqPanelStandardizedPct,
        s=np.clip(np.sqrt(compare.nMutated), 3, 19),
        c=compare.nCommonCancers,
        cmap="cividis",
        alpha=0.55,
        edgecolors="none",
    )
    lim = max(compare.freqWesStandardizedPct.max(), compare.freqPanelStandardizedPct.max()) * 1.06
    ax_c.plot([0, lim], [0, lim], color=COLORS["grey"], ls=(0, (2, 2)), lw=0.7)
    labels = {"CSMD3", "TP53", "MUC16", "TRRAP", "KRAS", "PIK3CA"}
    assay_offsets = {
        "CSMD3": (4, 2), "TP53": (4, -10), "MUC16": (4, -8),
        "TRRAP": (-24, -10), "KRAS": (4, 4), "PIK3CA": (-26, -9),
    }
    for row in compare[compare.gene.isin(labels)].itertuples(index=False):
        ax_c.annotate(
            row.gene,
            (row.freqWesStandardizedPct, row.freqPanelStandardizedPct),
            xytext=assay_offsets[row.gene],
            textcoords="offset points",
            fontsize=4.5,
        )
    rho = spearmanr(compare.freqWesStandardizedPct, compare.freqPanelStandardizedPct).statistic
    ax_c.text(
        0.04,
        0.95,
        f"n={len(compare)} genes\nSpearman ρ={rho:.2f}",
        transform=ax_c.transAxes,
        va="top",
        fontsize=5.2,
        bbox=NOTE_BBOX,
    )
    common_cb = fig.colorbar(sc_c, ax=ax_c, fraction=0.036, pad=0.02)
    common_cb.set_label("Common cancer groups (n)", fontsize=5.0)
    ax_c.set_xlabel("WES/WGS prevalence, cancer-standardised (%)")
    ax_c.set_ylabel("Targeted-panel prevalence, same weights (%)")
    panel_label(ax_c, "c", x=-0.17, y=1.04)
    compare.to_csv(SOURCE / "figure2_panel_c_assay.csv", index=False)

    # Row-normalized fill exposes disease context while printed integers retain counts.
    row_total = hotspot_matrix.sum(axis=1).replace(0, np.nan)
    row_pct = hotspot_matrix.div(row_total, axis=0) * 100
    image = ax_d.imshow(row_pct, aspect="auto", cmap="Blues", vmin=0, vmax=max(50, np.nanpercentile(row_pct, 98)))
    ax_d.set_xticks(np.arange(hotspot_matrix.shape[1]), hotspot_matrix.columns, rotation=50, ha="right")
    ax_d.set_yticks(np.arange(hotspot_matrix.shape[0]), hotspot_matrix.index, fontsize=4.6)
    ax_d.tick_params(length=0)
    for row in range(hotspot_matrix.shape[0]):
        for col in range(hotspot_matrix.shape[1]):
            count = int(hotspot_matrix.iat[row, col])
            if count:
                color = "white" if row_pct.iat[row, col] > 28 else COLORS["black"]
                ax_d.text(col, row, str(count), ha="center", va="center", fontsize=3.6, color=color)
    cbar = fig.colorbar(image, ax=ax_d, fraction=0.035, pad=0.02)
    cbar.ax.set_title("Cancer\nshare (%)", fontsize=5.0, pad=3)
    panel_label(ax_d, "d", x=-0.12, y=1.04)
    hotspot_counts.to_csv(SOURCE / "figure2_panel_d_hotspot_context.csv", index=False)

    fig.subplots_adjust(left=0.15, right=0.965, top=0.955, bottom=0.12)
    save_figure(fig, FIGURES / "figure2_driver_evidence")
    plt.close(fig)


def _gene_selection(prevalence: pd.DataFrame, evidence: pd.DataFrame, n: int = 34) -> list[str]:
    merged = prevalence.merge(evidence[["gene", "pctTierMatched"]], on="gene", how="left")
    merged["driverScore"] = merged.freqPct * (0.25 + 0.75 * merged.pctTierMatched.fillna(0) / 100)
    forced = [
        "TP53", "KRAS", "PIK3CA", "APC", "BRAF", "PTEN", "EGFR", "STK11", "KEAP1",
        "IDH1", "ATRX", "CTNNB1", "ARID1A", "RB1", "NF1", "CDKN2A", "SMAD4",
    ]
    ordered = list(dict.fromkeys(merged.nlargest(n, "driverScore").gene.tolist() + forced))
    score = merged.set_index("gene").driverScore.to_dict()
    return sorted(ordered, key=lambda gene: score.get(gene, 0), reverse=True)[:n]


def _pathway_map() -> dict[str, str]:
    path = PROCESSED / "sanchez_vega_pathway_membership.csv"
    if not path.exists():
        return {}
    membership = pd.read_csv(path)
    membership = membership[membership.usedInMutationOnlyAnalysis]
    return membership.drop_duplicates("gene").set_index("gene").pathway.to_dict()


FIGURE3_N_CANCER_GROUPS = int(os.environ.get("FIGURE3_N_CANCER_GROUPS", "72"))


def _figure3_sample_group_column(samples: pd.DataFrame, cancer_groups: set[str]) -> str:
    """Resolve the sample column represented by the complete cancer-group table.

    The cohort revision may add a reviewed family/histology code while retaining the
    historical broad code for traceability.  Matching values to ``cancerGroup`` keeps
    Figure 3 compatible with either schema and prevents the plot from silently ranking
    a superseded grouping column after the updated tables are regenerated.
    """
    preferred = [
        "reviewedCancerCode",
        "reviewedCancerGroup",
        "cancerFamilyCode",
        "analysisCancerGroup",
        "broadCancerCode",
    ]
    inferred = [
        column
        for column in samples.columns
        if "cancer" in column.lower() and ("code" in column.lower() or "group" in column.lower())
    ]
    candidates = list(dict.fromkeys(preferred + inferred))
    candidates = [column for column in candidates if column in samples.columns]
    if not candidates:
        raise RuntimeError("No cancer-group column is available for Figure 3 study counts")

    def score(column: str) -> tuple[int, int]:
        values = samples[column].dropna().astype(str)
        return int(values.isin(cancer_groups).sum()), int(values.nunique())

    selected = max(candidates, key=score)
    matched_rows, _ = score(selected)
    if matched_rows == 0:
        raise RuntimeError(
            "No sample-level cancer code matches the regenerated Figure 3 cancer groups"
        )
    return selected


def figure3() -> None:
    apply_style()
    complete = pd.read_csv(COMPLETE / "curated_cancer_gene_prevalence_complete.csv")
    prevalence = pd.read_csv(TABLES / "gene_frequencies_curated.csv")
    evidence = pd.read_csv(TABLES / "cmc_evidence_curated.csv")
    samples = pd.read_parquet(PROCESSED / "analysis_samples_curated.parquet")
    samples = samples[samples.analysisEligible]
    genes = _gene_selection(prevalence, evidence, 34)
    cancer_sizes = (
        complete.groupby("cancerGroup").nEligibleSamples.max().sort_values(ascending=False)
    )
    cancers = cancer_sizes.head(FIGURE3_N_CANCER_GROUPS).index.tolist()
    subset = complete[complete.gene.isin(genes) & complete.cancerGroup.isin(cancers)].copy()
    matrix = subset.pivot(index="gene", columns="cancerGroup", values="prevalencePct").reindex(index=genes, columns=cancers)
    n_callable = subset.pivot(index="gene", columns="cancerGroup", values="nCallable").reindex(index=genes, columns=cancers)

    # A dedicated label band and a shallow legend rail prevent the rotated cancer
    # codes from entering the legend area.  Seventy-two reviewed cancer families make
    # the main landscape appropriately wide; Supplementary Data 1 retains all 89.
    fig = plt.figure(figsize=figsize(180, 124))
    gs = GridSpec(
        5,
        3,
        figure=fig,
        width_ratios=[0.145, 1.0, 0.135],
        height_ratios=[0.18, 0.072, 1.0, 0.185, 0.13],
        hspace=0.025,
        wspace=0.060,
    )
    ax_top = fig.add_subplot(gs[0, 1]); ax_col = fig.add_subplot(gs[1, 1])
    ax_col_key = fig.add_subplot(gs[1, 2])
    ax_tracks = fig.add_subplot(gs[2, 0]); ax_heat = fig.add_subplot(gs[2, 1])
    ax_side = fig.add_subplot(gs[2, 2])
    ax_label_band = fig.add_subplot(gs[3, :])
    ax_label_band.axis("off")
    ax_legend = fig.add_subplot(gs[4, :])
    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)
    ax_legend.axis("off")
    ax_legend.add_patch(
        FancyBboxPatch(
            (0.006, 0.06),
            0.988,
            0.86,
            boxstyle="round,pad=0.01,rounding_size=0.035",
            transform=ax_legend.transAxes,
            facecolor="white",
            edgecolor=LEGEND_EDGE,
            linewidth=0.55,
            clip_on=False,
        )
    )

    x = np.arange(len(cancers))
    targeted = (
        subset.groupby("cancerGroup").agg(nTargeted=("nTargetedSamples", "max"), nTotal=("nEligibleSamples", "max"))
        .reindex(cancers)
    )
    ax_top.bar(x, [cancer_sizes[c] for c in cancers], color=COLORS["light_grey"], width=0.82)
    ax_top.set_xlim(-0.5, len(cancers) - 0.5); ax_top.set_xticks([]); ax_top.set_ylabel("Cases")

    sample_group_column = _figure3_sample_group_column(samples, set(complete.cancerGroup.astype(str)))
    study_counts = samples.groupby(sample_group_column).studyId.nunique().reindex(cancers).fillna(0).astype(int)
    annotation = np.vstack([
        100 * targeted.nTargeted / targeted.nTotal.clip(lower=1),
        100 * study_counts / study_counts.max(),
    ])
    annotation_cmap = LinearSegmentedColormap.from_list("ann", ["white", COLORS["purple"]])
    ax_col.imshow(annotation, aspect="auto", cmap=annotation_cmap, vmin=0, vmax=100)
    ax_col.set_yticks([0, 1], ["Targeted\ncases (%)", "Studies\n(relative)"], fontsize=4.3)
    ax_col.set_xticks([]); ax_col.tick_params(length=0)

    # Keep the column-track key beside the tracks; it no longer consumes a fourth,
    # vertically stacked legend box below the heatmap.
    ax_col_key.set_xlim(0, 1)
    ax_col_key.set_ylim(0, 1)
    ax_col_key.axis("off")
    ax_col_key.text(
        0.50,
        0.86,
        "Track intensity (%)",
        ha="center",
        va="center",
        fontsize=4.05,
        color=COLORS["black"],
    )
    for step in range(40):
        ax_col_key.add_patch(
            Rectangle(
                (0.12 + 0.72 * step / 40, 0.36),
                0.72 / 40 + 0.002,
                0.23,
                transform=ax_col_key.transAxes,
                facecolor=annotation_cmap(step / 39),
                edgecolor="none",
                clip_on=False,
            )
        )
    ax_col_key.add_patch(
        Rectangle(
            (0.12, 0.36),
            0.72,
            0.23,
            transform=ax_col_key.transAxes,
            facecolor="none",
            edgecolor=LEGEND_EDGE,
            linewidth=0.45,
            clip_on=False,
        )
    )
    ax_col_key.text(0.12, 0.19, "0", ha="center", va="center", fontsize=3.8)
    ax_col_key.text(0.84, 0.19, "100", ha="center", va="center", fontsize=3.8)

    panel = pd.read_csv(PROCESSED / "gene_panel.csv").drop_duplicates("hugoSymbol").set_index("hugoSymbol")
    path_map = _pathway_map()
    pathway_colors = PATHWAY_COLORS
    role_values = [ROLE_COLORS.get(panel.roleInCancer.get(gene, "Other"), COLORS["grey"]) for gene in genes]
    tier_values = [
        {1.0: COLORS["vermillion"], 2.0: COLORS["orange"]}.get(panel.cosmicTier.get(gene), COLORS["light_grey"])
        for gene in genes
    ]
    path_values = [pathway_colors.get(path_map.get(gene, "Not assigned"), COLORS["light_grey"]) for gene in genes]
    rgba = np.array(
        [
            [
                matplotlib.colors.to_rgba(role_values[i]),
                matplotlib.colors.to_rgba(tier_values[i]),
                matplotlib.colors.to_rgba(path_values[i]),
            ]
            for i in range(len(genes))
        ]
    )
    ax_tracks.imshow(rgba, aspect="auto")
    ax_tracks.set_xticks([0, 1, 2], ["Role", "CGC tier", "Pathway"], rotation=60, ha="right", fontsize=4.2)
    ax_tracks.set_yticks(np.arange(len(genes)), genes, fontsize=4.7)
    ax_tracks.tick_params(length=0)

    vmax = max(35.0, float(np.nanpercentile(matrix.to_numpy(float), 98)))
    cmap = LinearSegmentedColormap.from_list("prevalence", ["#F7FBFF", "#BDD7E7", COLORS["blue"], "#08306B"])
    masked = np.ma.masked_invalid(matrix.to_numpy(float))
    image = ax_heat.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
    ax_heat.set_xticks(
        np.arange(len(cancers)),
        cancers,
        rotation=90,
        ha="center",
        va="top",
        fontsize=3.65,
    )
    ax_heat.set_yticks([]); ax_heat.tick_params(length=0)
    ax_heat.set_xticks(np.arange(-0.5, len(cancers), 1), minor=True)
    ax_heat.set_yticks(np.arange(-0.5, len(genes), 1), minor=True)
    ax_heat.grid(which="minor", color="white", linewidth=0.35)
    for row in range(len(genes)):
        for col in range(len(cancers)):
            value = matrix.iat[row, col]
            callable_n = n_callable.iat[row, col]
            if not np.isfinite(value) or callable_n == 0:
                text_value, color = "—", COLORS["grey"]
            elif callable_n < 50:
                text_value, color = "·", COLORS["grey"]
            elif value < 0.5:
                text_value, color = "<1", COLORS["black"]
            else:
                text_value = f"{value:.0f}"
                color = "white" if value > 0.57 * vmax else COLORS["black"]
            ax_heat.text(col, row, text_value, ha="center", va="center", fontsize=2.42, color=color)

    global_prev = prevalence.set_index("gene").reindex(genes)
    if len(global_prev) != 34 or global_prev.nProfiled.isna().any():
        raise AssertionError("Figure 3c requires complete pan-cancer prevalence for all 34 displayed genes")
    y = np.arange(len(genes))
    ax_side.barh(y, global_prev.freqPct, color=[ROLE_COLORS.get(role, COLORS["grey"]) for role in global_prev.roleInCancer], height=0.7)
    ax_side.set_ylim(len(genes) - 0.5, -0.5); ax_side.set_yticks([])
    ax_side.set_xlabel("Pan-cancer\nprevalence (%)", labelpad=-2)
    ax_side.grid(axis="x", color=COLORS["very_light_grey"], lw=0.5)

    def draw_title(x_pos: float, title: str) -> None:
        ax_legend.text(
            x_pos,
            0.76,
            title,
            transform=ax_legend.transAxes,
            ha="left",
            va="center",
            fontsize=4.75,
            fontweight="bold",
            color=COLORS["black"],
        )

    def draw_swatch(
        x_pos: float,
        y_pos: float,
        color: str,
        label: str,
        *,
        fontsize: float = 3.95,
    ) -> None:
        ax_legend.add_patch(
            Rectangle(
                (x_pos, y_pos - 0.065),
                0.012,
                0.13,
                transform=ax_legend.transAxes,
                facecolor=color,
                edgecolor="none",
                clip_on=False,
            )
        )
        ax_legend.text(
            x_pos + 0.016,
            y_pos,
            label,
            transform=ax_legend.transAxes,
            ha="left",
            va="center",
            fontsize=fontsize,
            color=COLORS["black"],
        )

    # The integrated display uses one structured legend rail.  The quantitative
    # mutation-prevalence scale now occupies its own section in this rail rather than
    # appearing as a detached bar beside the heatmap.
    draw_title(0.020, "Gene role")
    for x_pos, y_pos, color, label in (
        (0.020, 0.45, ROLE_COLORS["Oncogenes"], "Oncogene"),
        (0.142, 0.45, ROLE_COLORS["Oncogene/TSG"], "Dual role"),
        (0.020, 0.18, ROLE_COLORS["TSGs"], "Tumour suppressor"),
        (0.142, 0.18, ROLE_COLORS["Other"], "Other/unknown"),
    ):
        draw_swatch(x_pos, y_pos, color, label)

    ax_legend.plot([0.280, 0.280], [0.12, 0.82], color=LEGEND_EDGE, lw=0.45, clip_on=False)
    draw_title(0.296, "Cancer Gene Census tier")
    for x_pos, y_pos, color, label in (
        (0.296, 0.39, COLORS["vermillion"], "Tier 1"),
        (0.358, 0.39, COLORS["orange"], "Tier 2"),
        (0.414, 0.39, COLORS["light_grey"], "Other/none"),
    ):
        draw_swatch(x_pos, y_pos, color, label, fontsize=3.8)

    ax_legend.plot([0.478, 0.478], [0.12, 0.82], color=LEGEND_EDGE, lw=0.45, clip_on=False)
    draw_title(0.494, "Canonical pathway")
    pathway_items = [
        ("RTK–RAS", 0.494, 0.49),
        ("PI3K–AKT", 0.594, 0.49),
        ("TP53", 0.694, 0.49),
        ("Cell cycle", 0.494, 0.29),
        ("WNT/β-catenin", 0.594, 0.29),
        ("NOTCH", 0.694, 0.29),
        ("TGF-β", 0.494, 0.09),
        ("Hippo", 0.594, 0.09),
        ("Not assigned", 0.694, 0.09),
    ]
    for name, x_pos, y_pos in pathway_items:
        draw_swatch(x_pos, y_pos, pathway_colors[name], name, fontsize=3.55)

    ax_legend.plot([0.802, 0.802], [0.12, 0.82], color=LEGEND_EDGE, lw=0.45, clip_on=False)
    draw_title(0.818, "Mutation prevalence (%)")
    legend_cbar_ax = ax_legend.inset_axes([0.818, 0.26, 0.162, 0.18])
    legend_cbar = fig.colorbar(image, cax=legend_cbar_ax, orientation="horizontal")
    legend_cbar.set_ticks([0, vmax / 2, vmax])
    legend_cbar.set_ticklabels([f"{value:.0f}" for value in (0, vmax / 2, vmax)])
    legend_cbar.ax.tick_params(axis="x", labelsize=3.7, pad=1.0, length=1.5, width=0.45)
    legend_cbar.outline.set_edgecolor(LEGEND_EDGE)
    legend_cbar.outline.set_linewidth(0.45)

    subset.to_csv(SOURCE / "figure3_integrated_heatmap.csv", index=False)
    pd.DataFrame({"cancerGroup": cancers, "nCases": [cancer_sizes[c] for c in cancers], "nStudies": study_counts.values, "targetedFractionPct": 100 * targeted.nTargeted.values / targeted.nTotal.clip(lower=1).values}).to_csv(
        SOURCE / "figure3_column_annotations.csv", index=False
    )
    pd.DataFrame({"gene": genes, "role": [panel.roleInCancer.get(g) for g in genes], "cosmicTier": [panel.cosmicTier.get(g) for g in genes], "pathway": [path_map.get(g) for g in genes]}).to_csv(
        SOURCE / "figure3_row_annotations.csv", index=False
    )
    (
        global_prev.reset_index()[
            [
                "gene",
                "entrezGeneId",
                "nMutated",
                "nProfiled",
                "freqPct",
                "freqCiLowPct",
                "freqCiHighPct",
                "pctCuratedSamplesProfilingGene",
                "roleInCancer",
            ]
        ]
        .rename(
            columns={
                "nProfiled": "nCallable",
                "freqPct": "prevalencePct",
                "freqCiLowPct": "prevalenceCiLowPct",
                "freqCiHighPct": "prevalenceCiHighPct",
            }
        )
        .to_csv(SOURCE / "figure3_panel_c_pan_cancer_prevalence.csv", index=False)
    )
    fig.subplots_adjust(left=0.178, right=0.992, top=0.96, bottom=0.025)
    # Figure 3 is one integrated gene-by-cancer landscape with aligned marginal
    # tracks, rather than a set of independently interpreted panels; panel letters
    # are therefore deliberately omitted.
    save_figure(fig, FIGURES / "figure3_integrated_heatmap")
    plt.close(fig)

ONCOPRINT_GENES = {
    "LUAD": ["TP53", "KRAS", "EGFR", "STK11", "KEAP1", "PIK3CA", "NF1", "BRAF", "CDKN2A", "RB1"],
    "LUSC": ["TP53", "CDKN2A", "PIK3CA", "NFE2L2", "KEAP1", "PTEN", "RB1", "FAT1", "KMT2D", "NOTCH1"],
    "COADREAD": ["APC", "TP53", "KRAS", "PIK3CA", "BRAF", "SMAD4", "FBXW7", "TCF7L2", "ARID1A", "PTEN"],
    "UCEC": ["PTEN", "PIK3CA", "ARID1A", "TP53", "PIK3R1", "KRAS", "CTNNB1", "KMT2D", "FBXW7", "PPP2R1A"],
}

# Four additional, biologically complementary exemplar cohorts for Supplementary
# Figure 8.  These panels deliberately use the same mutation-only, assay-aware visual
# grammar as Figure 4.  They contain every eligible representative case in each broad
# cancer group; the gene lists are display sets, not comprehensive definitions of the
# cancers or inferred molecular subtypes.
SUPPLEMENTARY_ONCOPRINT_GENES = {
    "BRCA": ["TP53", "PIK3CA", "GATA3", "CDH1", "KMT2C", "MAP3K1", "ESR1", "PTEN", "ARID1A", "AKT1"],
    "PAAD": ["KRAS", "TP53", "SMAD4", "CDKN2A", "ARID1A", "RNF43", "KMT2D", "MEN1", "ATM", "GNAS"],
    "SKCM": ["BRAF", "NRAS", "NF1", "TP53", "PTPRT", "GRIN2A", "CDKN2A", "ARID2", "PTEN", "RAC1"],
    "GBM": ["TP53", "PTEN", "EGFR", "IDH1", "ATRX", "NF1", "PIK3CA", "RB1", "PIK3R1", "CDKN2A"],
}

ONCOPRINT_DISPLAY_LABELS = {
    # Historical portal releases grouped under GBM contain IDH1-mutant tumours
    # and therefore cannot be described as a molecularly verified
    # IDH-wildtype cohort.
    "GBM": "Portal-defined or legacy GBM",
}

STATE_LABELS = {
    -1: "Unassayed",
    0: "Assay-covered, no mutation",
    1: "Missense",
    2: "Truncating/splice",
    3: "In-frame",
    4: "Multiple classes",
    5: "Other protein-altering",
}
STATE_COLORS = [
    "#FFFFFF",
    "#E8ECEF",
    COLORS["green"],
    "#2C3E50",
    COLORS["orange"],
    "#88419D",
    COLORS["vermillion"],
]


def _mutation_class(value: str) -> int:
    text = str(value)
    if any(token in text for token in ("Frame_Shift", "Nonsense", "Splice", "Translation_Start", "Nonstop")):
        return 2
    if "In_Frame" in text:
        return 3
    if "Missense" in text:
        return 1
    return 5


def _oncoprint_data(cancer: str, genes: list[str]) -> tuple[pd.DataFrame, np.ndarray, pd.Series, pd.Series]:
    samples = pd.read_parquet(PROCESSED / "analysis_samples_curated.parquet")
    samples = samples[samples.analysisEligible & samples.broadCancerCode.eq(cancer)].copy()
    assay = pd.read_parquet(PROCESSED / "sample_assay.parquet")
    info = samples[["sampleId", "studyId", "sampleTypeGroup"]].merge(assay, on=["sampleId", "studyId"], how="left", validate="one_to_one")
    panel = pd.read_csv(PROCESSED / "gene_panel.csv")
    sym2ent = panel.set_index("hugoSymbol").entrezGeneId.to_dict(); ent2sym = {int(v): k for k, v in sym2ent.items()}
    memberships = pd.read_parquet(PROCESSED / "panel_gene_membership.parquet")
    member = set(zip(memberships.genePanelId.astype(str), memberships.entrezGeneId.astype(int)))
    n_panel = memberships.groupby("genePanelId").entrezGeneId.nunique().to_dict()
    mutations = pd.read_parquet(PROCESSED / "mutations_curated.parquet")
    mutations = mutations[mutations.sampleId.isin(info.sampleId)]
    mutation_count = mutations.drop_duplicates(["sampleId", "entrezGeneId"]).groupby("sampleId").entrezGeneId.nunique()
    info["callableGeneCount"] = np.where(info.assayType.str.startswith("WES/WGS"), len(panel), info.genePanelId.map(n_panel).fillna(0))
    info["mutatedGeneCount"] = info.sampleId.map(mutation_count).fillna(0)
    info["burdenPct"] = 100 * info.mutatedGeneCount / info.callableGeneCount.clip(lower=1)
    info["assayCode"] = info.assayType.map({"WES/WGS": 0, "WES/WGS (assumed; no panel metadata)": 1, "Targeted panel": 2}).fillna(3).astype(int)

    states = pd.DataFrame(0, index=info.sampleId.astype(str), columns=genes, dtype=int)
    for gene in genes:
        entrez = int(sym2ent[gene])
        callable_mask = info.assayType.str.startswith("WES/WGS") | [
            (str(panel_id), entrez) in member for panel_id in info.genePanelId
        ]
        states.loc[~np.asarray(callable_mask), gene] = -1
    chosen_ents = {int(sym2ent[gene]) for gene in genes}
    event = mutations[mutations.entrezGeneId.isin(chosen_ents)].copy()
    event["gene"] = event.entrezGeneId.astype(int).map(ent2sym)
    event["class"] = event.mutationType.map(_mutation_class)
    for (sample_id, gene), group in event.groupby(["sampleId", "gene"]):
        classes = set(group["class"])
        states.loc[str(sample_id), gene] = 4 if len(classes) > 1 else next(iter(classes))
    order_frame = (states > 0).astype(int)
    order_frame["burdenPct"] = info.set_index("sampleId").loc[order_frame.index, "burdenPct"]
    order_frame["sampleIdOrder"] = order_frame.index
    order = order_frame.sort_values([*genes, "burdenPct", "sampleIdOrder"], ascending=[False] * len(genes) + [False, True]).index
    states = states.loc[order]
    info = info.set_index("sampleId").loc[order].reset_index()
    prevalence = (states.gt(0).sum(axis=0) / states.ge(0).sum(axis=0).clip(lower=1) * 100).reindex(genes)
    return info, states.to_numpy().T, prevalence, states.ge(0).sum(axis=0).reindex(genes)


def _landscape_oncoprint_figure(
    contexts: dict[str, list[str]],
    *,
    stem: Path,
    source_prefix: str,
) -> None:
    """Draw four assay-aware oncoprints in the shared 2 × 2 landscape design."""
    if len(contexts) != 4:
        raise AssertionError(f"Expected four oncoprint contexts, found {len(contexts)}")

    apply_style(); setup_dirs()
    fig = plt.figure(figsize=figsize(180, 125))
    outer = GridSpec(2, 2, figure=fig, hspace=0.16, wspace=0.13)
    state_cmap = ListedColormap(STATE_COLORS)
    state_norm = BoundaryNorm(np.arange(-1.5, 6.5, 1), state_cmap.N)
    panel_axes: list[tuple[str, plt.Axes]] = []

    for index, (cancer, genes) in enumerate(contexts.items()):
        row, col = divmod(index, 2)
        panel = chr(ord("a") + index)
        sub = GridSpecFromSubplotSpec(
            2,
            2,
            subplot_spec=outer[row, col],
            height_ratios=[0.19, 1.0],
            width_ratios=[1.0, 0.14],
            hspace=0.015,
            wspace=0.035,
        )
        ax_top = fig.add_subplot(sub[0, 0])
        panel_axes.append((panel, ax_top))
        ax_top_right = fig.add_subplot(sub[0, 1]); ax_top_right.axis("off")
        ax_matrix = fig.add_subplot(sub[1, 0])
        ax_side = fig.add_subplot(sub[1, 1])
        info, matrix, prevalence, n_callable = _oncoprint_data(cancer, genes)
        n_samples = len(info)
        sample_alteration_count = np.sum(matrix > 0, axis=0)
        n_altered = int(np.sum(sample_alteration_count > 0))

        ax_top.bar(
            np.arange(n_samples),
            sample_alteration_count,
            width=1.0,
            color="#7F8C8D",
            edgecolor="none",
            rasterized=True,
        )
        ax_top.set_xlim(-0.5, n_samples - 0.5)
        top_max = max(1, int(sample_alteration_count.max()))
        ax_top.set_ylim(0, top_max * 1.08)
        ax_top.set_yticks([0, top_max])
        ax_top.tick_params(axis="y", labelsize=3.55, length=1.5, pad=1.0)
        ax_top.set_xticks([])
        ax_top.set_ylabel("Altered\ngenes", fontsize=3.75, labelpad=1.2)
        ax_top.spines["bottom"].set_visible(False)
        display_cancer = ONCOPRINT_DISPLAY_LABELS.get(cancer, cancer)
        ax_top.set_title(
            f"{display_cancer}  (n={n_samples:,}; {n_altered:,} altered)",
            fontsize=5.9,
            pad=2.0,
        )

        ax_matrix.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            cmap=state_cmap,
            norm=state_norm,
            rasterized=True,
        )
        ax_matrix.set_facecolor("white")
        ax_matrix.set_yticks(np.arange(len(genes)), genes, fontsize=4.25)
        ax_matrix.set_xticks([])
        ax_matrix.tick_params(length=0, pad=1.8)

        ax_side.barh(
            np.arange(len(genes)),
            prevalence.values,
            color="#7F8C8D",
            height=0.68,
        )
        ax_side.set_ylim(len(genes) - 0.5, -0.5)
        ax_side.set_yticks([])
        ax_side.set_xlabel("Mutated (%)", fontsize=3.75, labelpad=1.2)
        ax_side.tick_params(axis="x", labelsize=3.45, length=1.7, pad=1.0)
        ax_side.spines["left"].set_visible(False)

        # The per-case displayed-gene count is stored explicitly because it drives
        # the aligned upper marginal; the remaining columns retain the complete
        # assay-aware mutation-state matrix used by the oncoprint itself.
        source = info[
            ["sampleId", "studyId", "assayType", "sampleTypeGroup", "burdenPct"]
        ].copy()
        source["displayedAlteredGeneCount"] = sample_alteration_count
        state_frame = pd.DataFrame(matrix.T, columns=genes).replace(STATE_LABELS)
        pd.concat([source.reset_index(drop=True), state_frame], axis=1).to_csv(
            SOURCE
            / f"{source_prefix}_panel_{panel}_{cancer.lower()}_oncoprint.csv",
            index=False,
        )
        pd.DataFrame(
            {
                "gene": genes,
                "prevalencePct": prevalence.values,
                "nCallable": n_callable.values,
            }
        ).to_csv(
            SOURCE
            / f"{source_prefix}_panel_{panel}_{cancer.lower()}_prevalence.csv",
            index=False,
        )

    legend_order = (1, 2, 3, 4, 5, 0, -1)
    legend = [
        Patch(
            facecolor=STATE_COLORS[value + 1],
            edgecolor=COLORS["grey"] if value in (-1, 0) else "none",
            linewidth=0.5,
            label=STATE_LABELS[value],
        )
        for value in legend_order
    ]
    mutation_legend = fig.legend(
        handles=legend,
        title="Mutation class",
        frameon=True,
        fancybox=True,
        ncol=7,
        loc="lower center",
        fontsize=4.05,
        title_fontsize=4.45,
        bbox_to_anchor=(0.5, 0.014),
        borderpad=0.42,
        handlelength=1.2,
        handleheight=0.85,
        columnspacing=0.82,
        labelspacing=0.28,
    )
    _finish_legend(mutation_legend, pad=0.34)
    fig.subplots_adjust(left=0.105, right=0.99, top=0.955, bottom=0.108)
    aligned_panel_labels(
        fig,
        [tuple(panel_axes[:2]), tuple(panel_axes[2:])],
    )
    save_figure(fig, stem)
    plt.close(fig)


def figure4() -> None:
    """Draw the four main cancer exemplars in the active landscape layout."""
    _landscape_oncoprint_figure(
        ONCOPRINT_GENES,
        stem=FIGURES / "figure4_four_oncoprints",
        source_prefix="figure4",
    )


def supplementary_figure8() -> None:
    """Draw four additional cancer exemplars in the matching landscape layout."""
    _landscape_oncoprint_figure(
        SUPPLEMENTARY_ONCOPRINT_GENES,
        stem=SUPP / "figureS7_additional_oncoprints",
        source_prefix="figureS7",
    )


REFERENCE_CONTEXTS = [
    ("EGFR-KRAS", "LUAD"), ("KEAP1-STK11", "LUAD"), ("KRAS-TP53", "LUAD"),
    ("STK11-TP53", "LUAD"), ("BRAF-KRAS", "COADREAD"), ("BRAF-NRAS", "SKCM"),
    ("PIK3CA-PIK3R1", "UCEC"), ("PTEN-TP53", "UCEC"), ("CTNNB1-TP53", "UCEC"),
    ("PIK3CA-TP53", "BRCA"), ("KRAS-TP53", "PAAD"), ("CALR-JAK2", "MPN"),
    ("NPM1-RUNX1", "AML"), ("SF3B1-SRSF2", "MDS"),
]

HEADLINE_HETEROGENEITY_CONTEXTS = [
    ("KEAP1-STK11", "LUAD"), ("KRAS-TP53", "PAAD"),
    ("PIK3CA-TP53", "BRCA"), ("KRAS-TP53", "LUAD"),
    ("STK11-TP53", "LUAD"), ("BRAF-KRAS", "COADREAD"),
    ("EGFR-KRAS", "LUAD"), ("CALR-JAK2", "MPN"),
]


def _figure5_legacy() -> None:
    apply_style()
    data = pd.read_csv(TABLES / "cooccurrence_curated_adjusted_sensitivity.csv")
    data["log2CmhOr"] = np.log2(data.cmh_full_or)
    ref = pd.concat([data[(data.pair.eq(pair)) & (data.cancer.eq(cancer))] for pair, cancer in REFERENCE_CONTEXTS], ignore_index=True)
    ref = ref.drop_duplicates(["pair", "cancer"])

    fig = plt.figure(figsize=figsize(180, 175))
    gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.54)
    ax_a = fig.add_subplot(gs[0, 0]); ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0]); ax_d = fig.add_subplot(gs[1, 1])

    pairs = list(dict.fromkeys(pair for pair, _ in REFERENCE_CONTEXTS))
    cancers = list(dict.fromkeys(cancer for _, cancer in REFERENCE_CONTEXTS))
    matrix = ref.pivot(index="pair", columns="cancer", values="log2CmhOr").reindex(index=pairs, columns=cancers)
    limit = max(4.0, float(np.nanpercentile(np.abs(matrix), 96)))
    image = ax_a.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax_a.set_xticks(np.arange(len(cancers)), cancers, rotation=50, ha="right", fontsize=4.7)
    ax_a.set_yticks(np.arange(len(pairs)), pairs, fontsize=4.7); ax_a.tick_params(length=0)
    lookup = ref.set_index(["pair", "cancer"])
    for row, pair in enumerate(pairs):
        for col, cancer in enumerate(cancers):
            value = matrix.iat[row, col]
            if not np.isfinite(value):
                ax_a.text(col, row, "—", ha="center", va="center", fontsize=3.8, color=COLORS["grey"])
                continue
            record = lookup.loc[(pair, cancer)]
            color = "white" if abs(value) > 0.55 * limit else COLORS["black"]
            weight = "bold" if record.cmh_full_fdr < 0.05 else "normal"
            ax_a.text(col, row, f"{value:.1f}", ha="center", va="center", fontsize=3.7, color=color, fontweight=weight)
            if bool(record.cmhReplicated):
                ax_a.scatter(col + 0.34, row - 0.32, s=5, facecolor="none", edgecolor=color, linewidth=0.45)
    cbar = fig.colorbar(image, ax=ax_a, fraction=0.035, pad=0.018)
    cbar.ax.set_title("log2 OR", fontsize=5.0, pad=3)
    matrix_legend = [
        Line2D([0], [0], linestyle="", marker="", label="Bold values: FDR q < 0.05"),
        Line2D([0], [0], linestyle="", marker="o", markerfacecolor="none", markeredgecolor=COLORS["black"], markersize=3.5, label="Cross-assay concordant"),
    ]
    matrix_legend_obj = ax_a.legend(
        handles=matrix_legend,
        title="Cell notation",
        frameon=True,
        fancybox=True,
        fontsize=4.2,
        title_fontsize=4.7,
        loc="upper right",
        borderpad=0.43,
        handletextpad=0.4,
    )
    _finish_legend(matrix_legend_obj)
    panel_label(ax_a, "a", x=-0.18, y=1.04)
    ref.to_csv(SOURCE / "figure5_panel_a_reference_matrix.csv", index=False)

    forest = ref[ref.cmh_full_or.notna() & ref.cmh_full_ciLow.notna() & ref.cmh_full_ciHigh.notna()].copy()
    forest["label"] = forest.pair + " (" + forest.cancer + ")"
    forest = forest.sort_values("cmh_full_or", ascending=False).reset_index(drop=True)
    y = np.arange(len(forest))
    colors = np.where(forest.cmh_full_or > 1, COLORS["vermillion"], COLORS["blue"])
    ax_b.hlines(y, forest.cmh_full_ciLow, forest.cmh_full_ciHigh, color=colors, lw=1.0)
    ax_b.scatter(forest.cmh_full_or, y, c=colors, s=19, edgecolor="white", lw=0.3, zorder=3)
    ax_b.axvline(1, color=COLORS["black"], ls=(0, (2, 2)), lw=0.7); ax_b.set_xscale("log")
    ax_b.set_yticks(y, forest.label, fontsize=4.5); ax_b.set_xlabel("Conditioned common odds ratio (95% CI)")
    direction_legend = ax_b.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="", markersize=3.7,
                   color=COLORS["blue"], label="Depletion (OR < 1)"),
            Line2D([0], [0], marker="o", linestyle="", markersize=3.7,
                   color=COLORS["vermillion"], label="Co-occurrence (OR > 1)"),
        ],
        frameon=True,
        fancybox=True,
        ncol=2,
        fontsize=4.1,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.015),
        borderaxespad=0,
        borderpad=0.4,
        handletextpad=0.45,
        columnspacing=0.8,
    )
    _finish_legend(direction_legend)
    panel_label(ax_b, "b", x=-0.12, y=1.04)
    forest.to_csv(SOURCE / "figure5_panel_b_reference_forest.csv", index=False)

    both = data[data.cmh_wes_or.gt(0) & data.cmh_panel_or.gt(0)].copy()
    x = np.log2(both.cmh_wes_or); yv = np.log2(both.cmh_panel_or)
    ax_c.scatter(x, yv, s=7, color=COLORS["light_grey"], alpha=0.28, edgecolors="none")
    highlight_keys = set(REFERENCE_CONTEXTS)
    highlight = both[["pair", "cancer"]].apply(tuple, axis=1).isin(highlight_keys)
    ax_c.scatter(x[highlight], yv[highlight], s=24, color=COLORS["orange"], edgecolor="white", lw=0.4, zorder=3)
    extent = float(np.nanquantile(np.abs(np.r_[x, yv]), 0.995))
    ax_c.set_xlim(-extent, extent)
    ax_c.set_ylim(-extent, extent)
    ax_c.set_aspect("equal", adjustable="box")
    ax_c.plot([-extent, extent], [-extent, extent], color=COLORS["grey"], ls=(0, (2, 2)), lw=0.7)
    rho = spearmanr(x, yv).statistic
    ax_c.text(
        0.04,
        0.95,
        f"n={len(both):,}\nSpearman ρ={rho:.2f}",
        transform=ax_c.transAxes,
        va="top",
        fontsize=5.2,
        bbox=NOTE_BBOX,
    )
    reference_legend = ax_c.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="", markersize=4.0,
                   markerfacecolor=COLORS["orange"], markeredgecolor="white",
                   markeredgewidth=0.35, label="Reference contexts")
        ],
        frameon=True,
        fancybox=True,
        fontsize=4.3,
        loc="upper right",
        borderpad=0.42,
        handletextpad=0.45,
    )
    _finish_legend(reference_legend)
    ax_c.set_xlabel("WES/WGS conditioned log2 OR"); ax_c.set_ylabel("Targeted-panel conditioned log2 OR")
    panel_label(ax_c, "c", x=-0.18, y=1.04)
    # Source Data should contain the exact plotted coordinates and highlight flag,
    # rather than duplicating the 54-column complete interaction scan (which is
    # supplied independently in Supplementary Data 2).
    panel_c_source = both[["cancer", "pair", "geneA", "geneB"]].copy()
    panel_c_source["wesLog2Or"] = x.to_numpy()
    panel_c_source["panelLog2Or"] = yv.to_numpy()
    panel_c_source["isReferenceContext"] = highlight.to_numpy()
    panel_c_source.to_csv(SOURCE / "figure5_panel_c_cross_assay.csv", index=False)

    adjusted = data[data.full_oddsRatio.gt(0) & data.cmh_full_or.gt(0)].copy()
    raw = np.log2(adjusted.full_oddsRatio); conditioned = np.log2(adjusted.cmh_full_or)
    cap = float(np.nanquantile(np.abs(np.r_[raw, conditioned]), 0.995))
    hb = ax_d.hexbin(raw, conditioned, gridsize=48, extent=(-cap, cap, -cap, cap), mincnt=1, bins="log", cmap="Greens", linewidths=0)
    # A true identity reference requires identical limits and equal physical scaling on
    # both axes; otherwise the diagonal is visually displaced from the centre of the
    # point cloud even when its coordinates are mathematically correct.
    ax_d.set_xlim(-cap, cap)
    ax_d.set_ylim(-cap, cap)
    ax_d.set_aspect("equal", adjustable="box")
    ax_d.plot([-cap, cap], [-cap, cap], color=COLORS["grey"], ls=(0, (2, 2)), lw=0.7)
    retain = float((np.sign(raw) == np.sign(conditioned)).mean() * 100); rho2 = spearmanr(raw, conditioned).statistic
    ax_d.text(
        0.04,
        0.95,
        f"n={len(adjusted):,}\nSpearman ρ={rho2:.2f}\ndirection retained={retain:.1f}%",
        transform=ax_d.transAxes,
        va="top",
        fontsize=5.0,
        bbox=NOTE_BBOX,
    )
    density_cb = fig.colorbar(hb, ax=ax_d, fraction=0.036, pad=0.02)
    max_bin_count = float(np.nanmax(hb.get_array()))
    density_ticks = [value for value in (1, 10, 100, 1000, 10000) if value <= max_bin_count]
    density_cb.set_ticks(density_ticks)
    density_cb.set_ticklabels([f"{value:,}" for value in density_ticks])
    density_cb.ax.set_title("Tests per\nhexagon", fontsize=5.0, pad=3)
    density_cb.ax.yaxis.set_ticks_position("left")
    ax_d.set_xlabel("Joint-callability log2 OR"); ax_d.set_ylabel("Context-conditioned log2 OR")
    panel_label(ax_d, "d", x=-0.12, y=1.04)
    panel_d_source = adjusted[["cancer", "pair", "geneA", "geneB"]].copy()
    panel_d_source["jointCallabilityLog2Or"] = raw.to_numpy()
    panel_d_source["conditionedLog2Or"] = conditioned.to_numpy()
    panel_d_source["directionRetained"] = (
        np.sign(raw.to_numpy()) == np.sign(conditioned.to_numpy())
    )
    panel_d_source.to_csv(SOURCE / "figure5_panel_d_conditioning.csv", index=False)

    fig.subplots_adjust(left=0.17, right=0.985, top=0.955, bottom=0.10)
    save_figure(fig, FIGURES / "figure5_conditioned_interactions")
    plt.close(fig)


def figure5() -> None:
    """Compare pairwise specifications and study-level robustness."""
    apply_style()
    data = pd.read_csv(
        TABLES / "cooccurrence_curated_adjusted_sensitivity.csv", low_memory=False
    )
    heterogeneity = pd.read_csv(TABLES / "pairwise_study_heterogeneity_primary.csv")
    reference_order = {
        (pair, cancer): order for order, (pair, cancer) in enumerate(REFERENCE_CONTEXTS)
    }
    ref = data[
        data[["pair", "cancer"]].apply(tuple, axis=1).isin(reference_order)
    ].copy()
    if len(ref) != len(REFERENCE_CONTEXTS):
        raise RuntimeError("A Figure 5 reference context is absent from the three-specification table")
    if ((ref.cancer == "GBM") & ref.pair.str.contains("IDH1", regex=False)).any():
        raise AssertionError("Portal-defined/legacy GBM IDH1 contexts cannot enter Figure 5")
    ref["displayOrder"] = ref[["pair", "cancer"]].apply(tuple, axis=1).map(reference_order)
    ref = ref.sort_values("displayOrder").reset_index(drop=True)
    ref["displayLabel"] = (
        ref.pair.str.replace("-", "–", regex=False) + " (" + ref.cancer + ")"
    )

    fig = plt.figure(figsize=figsize(180, 125))
    outer = GridSpec(
        2, 1, figure=fig, height_ratios=[1.0, 1.0], hspace=0.53,
    )
    # Balance the two analytical comparisons in the upper lane.  The former
    # three-column layout made panel a unnecessarily wide and compressed panel
    # b into a small square.  Independent subgrids retain the useful narrow-wide
    # relationship between panels c and d without imposing it on the top row.
    top = outer[0].subgridspec(1, 2, width_ratios=[1.04, 1.0], wspace=0.39)
    bottom = outer[1].subgridspec(1, 2, width_ratios=[0.78, 1.52], wspace=0.58)
    ax_a = fig.add_subplot(top[0, 0])
    ax_b = fig.add_subplot(top[0, 1])
    ax_c = fig.add_subplot(bottom[0, 0])
    ax_d = fig.add_subplot(bottom[0, 1])

    # a, primary reference estimates with leave-two-out and historical sensitivities.
    y = np.arange(len(ref))
    primary = np.log2(ref.noBurden_full_or.to_numpy(float))
    primary_lo = np.log2(ref.noBurden_full_ciLow.to_numpy(float))
    primary_hi = np.log2(ref.noBurden_full_ciHigh.to_numpy(float))
    leave_two_out = np.log2(ref.leaveTwoOut_full_or.to_numpy(float))
    total_burden = np.log2(ref.totalBurden_full_or.to_numpy(float))
    primary_colours = np.where(primary < 0, COLORS["blue"], COLORS["vermillion"])
    for row, (value, lo, hi, colour) in enumerate(
        zip(primary, primary_lo, primary_hi, primary_colours)
    ):
        ax_a.hlines(row, lo, hi, color=colour, lw=0.85, zorder=2)
        ax_a.scatter(value, row, s=17, color=colour, edgecolor="white", lw=0.3, zorder=4)
    ax_a.scatter(
        leave_two_out, y + 0.17, s=12, marker="D", color=COLORS["orange"],
        edgecolor="white", lw=0.25, zorder=3,
    )
    ax_a.scatter(
        total_burden, y - 0.17, s=14, marker="o", facecolor="white",
        edgecolor=COLORS["grey"], lw=0.65, zorder=3,
    )
    ax_a.axvline(0, color=COLORS["black"], ls=(0, (2, 2)), lw=0.65)
    ax_a.set_yticks(y, ref.displayLabel, fontsize=4.25)
    ax_a.invert_yaxis()
    x_min = float(np.floor(np.nanmin(primary_lo) - 0.25))
    x_max = float(np.ceil(np.nanmax(primary_hi) + 0.25))
    ax_a.set_xlim(x_min, x_max)
    ax_a.set_xlabel("Common log2 odds ratio")
    ax_a.set_title(
        "Reference associations", loc="left", fontsize=7.0,
        fontweight="normal", pad=4,
    )
    specification_legend = ax_a.legend(
        handles=[
            Line2D([0], [0], marker="o", ls="-", lw=0.8, markersize=3.7,
                   color=COLORS["blue"], label="Primary: no burden (95% CI)"),
            Line2D([0], [0], marker="D", ls="", markersize=3.4,
                   color=COLORS["orange"], label="Leave-two-out burden"),
            Line2D([0], [0], marker="o", ls="", markersize=3.5,
                   markerfacecolor="white", markeredgecolor=COLORS["grey"],
                   label="Total-burden diagnostic"),
        ],
        frameon=True, fancybox=True, ncol=3, fontsize=3.9,
        loc="lower right", bbox_to_anchor=(1.0, 1.13), borderaxespad=0,
        borderpad=0.35, handletextpad=0.35, columnspacing=0.65,
    )
    _finish_legend(specification_legend)
    ref_source_columns = [
        "displayOrder", "cancer", "pair", "geneA", "geneB", "displayLabel",
        "noBurden_full_or", "noBurden_full_ciLow", "noBurden_full_ciHigh",
        "noBurden_full_p", "noBurden_full_fdr", "leaveTwoOut_full_or",
        "leaveTwoOut_full_ciLow", "leaveTwoOut_full_ciHigh", "leaveTwoOut_full_p",
        "leaveTwoOut_full_fdr", "totalBurden_full_or", "totalBurden_full_ciLow",
        "totalBurden_full_ciHigh", "totalBurden_full_p", "totalBurden_full_fdr",
        "signStableNoBurdenLeaveTwoOut", "effectStableNoBurdenLeaveTwoOut",
        "noBurden_full_nInformativeStrata", "leaveTwoOut_full_nInformativeStrata",
        "totalBurden_full_nInformativeStrata", "effectStabilityDefinition",
    ]
    ref[ref_source_columns].to_csv(
        SOURCE / "figure5_panel_a_reference_specifications.csv", index=False
    )

    # b, pair-wide primary versus leave-two-out comparison.
    comparison = data[
        data.noBurden_full_or.gt(0) & data.leaveTwoOut_full_or.gt(0)
    ].copy()
    comparison["primaryLog2Or"] = np.log2(comparison.noBurden_full_or)
    comparison["leaveTwoOutLog2Or"] = np.log2(comparison.leaveTwoOut_full_or)
    px = comparison.primaryLog2Or.to_numpy(float)
    py = comparison.leaveTwoOutLog2Or.to_numpy(float)
    extent = max(2.0, float(np.nanquantile(np.abs(np.r_[px, py]), 0.995)))
    plot_x = np.clip(px, -extent, extent)
    plot_y = np.clip(py, -extent, extent)
    density_cmap = LinearSegmentedColormap.from_list(
        "nature_density",
        ["#F5F8FA", "#D6EAF2", "#92CEE0", "#3A97BE", "#17365D"],
    )
    hb = ax_b.hexbin(
        plot_x, plot_y, gridsize=(46, 32),
        extent=(-extent, extent, -extent, extent), mincnt=1, bins="log",
        cmap=density_cmap, linewidths=0,
    )
    ax_b.plot(
        [-extent, extent], [-extent, extent], color=COLORS["vermillion"],
        ls=(0, (2, 2)), lw=0.75,
    )
    ax_b.set_xlim(-extent, extent); ax_b.set_ylim(-extent, extent)
    # Fill the same row height as panel a while preserving matched numerical
    # limits on both axes and the explicit identity reference line.
    ax_b.set_aspect("auto")
    direction_retained = 100 * comparison.signStableNoBurdenLeaveTwoOut.mean()
    effect_retained = 100 * comparison.effectStableNoBurdenLeaveTwoOut.mean()
    rho = spearmanr(px, py).statistic
    ax_b.text(
        0.04, 0.96,
        f"n={len(comparison):,}\nρ={rho:.2f}\ndirection stable={direction_retained:.1f}%\neffect stable={effect_retained:.1f}%",
        transform=ax_b.transAxes, va="top", fontsize=4.55, bbox=NOTE_BBOX,
    )
    density_ax = ax_b.inset_axes([0.64, 0.08, 0.30, 0.025])
    density_cb = fig.colorbar(hb, cax=density_ax, orientation="horizontal")
    density_cb.ax.tick_params(labelsize=3.5, length=1.3, pad=1)
    density_cb.ax.set_title("Tests per hexagon", fontsize=3.8, pad=2)
    ax_b.set_xlabel("Primary no-burden log2 OR")
    ax_b.set_ylabel("Leave-two-out log2 OR")
    ax_b.set_title(
        "Burden sensitivity", loc="left", fontsize=7.0,
        fontweight="normal", pad=4,
    )
    comparison_source = comparison[
        [
            "cancer", "pair", "geneA", "geneB", "noBurden_full_or",
            "leaveTwoOut_full_or", "primaryLog2Or", "leaveTwoOutLog2Or",
            "noBurden_full_fdr", "leaveTwoOut_full_fdr",
            "signStableNoBurdenLeaveTwoOut", "effectStableNoBurdenLeaveTwoOut",
            "leaveTwoOutMinusNoBurdenLog2Or", "effectStabilityDefinition",
        ]
    ].copy()
    comparison_source["primaryCoordinateClippedForDisplay"] = np.abs(px) > extent
    comparison_source["leaveTwoOutCoordinateClippedForDisplay"] = np.abs(py) > extent
    comparison_source.to_csv(
        SOURCE / "figure5_panel_b_primary_leave_two_out.csv", index=False
    )

    # c, nested robustness criteria used to select display candidates.
    jointly_estimable = data.noBurden_full_or.gt(0) & data.leaveTwoOut_full_or.gt(0)
    sign_stable = jointly_estimable & data.signStableNoBurdenLeaveTwoOut.fillna(False)
    effect_stable = sign_stable & data.effectStableNoBurdenLeaveTwoOut.fillna(False)
    both_significant = (
        effect_stable & data.noBurden_full_fdr.lt(0.05)
        & data.leaveTwoOut_full_fdr.lt(0.05)
    )
    primary_assay = both_significant & data.noBurdenCrossAssayConcordant.fillna(False)
    both_assays = primary_assay & data.leaveTwoOutCrossAssayConcordant.fillna(False)
    funnel = pd.DataFrame(
        {
            "criterion": [
                "Jointly estimable", "Direction stable", "Effect stable",
                "Both q < 0.05", "+ primary assay concordance",
                "+ sensitivity assay concordance",
            ],
            "definition": [
                "Finite positive OR under primary and leave-two-out models",
                "Same OR direction under primary and leave-two-out models",
                "Same direction and within two-fold OR, or concordantly strong",
                "Effect stable and within-cancer FDR < 0.05 under both models",
                "Also cross-assay concordant under the primary model",
                "Also cross-assay concordant under the leave-two-out model",
            ],
            "nPairs": [
                int(jointly_estimable.sum()), int(sign_stable.sum()),
                int(effect_stable.sum()), int(both_significant.sum()),
                int(primary_assay.sum()), int(both_assays.sum()),
            ],
        }
    )
    funnel["pctJointlyEstimable"] = 100 * funnel.nPairs / int(jointly_estimable.sum())
    fy = np.arange(len(funnel))
    funnel_colours = [
        "#AEB6BF", COLORS["sky"], COLORS["blue"], COLORS["green"],
        COLORS["orange"], COLORS["vermillion"],
    ]
    ax_c.barh(fy, funnel.pctJointlyEstimable, color=funnel_colours, height=0.64)
    for row in funnel.itertuples():
        inside = row.pctJointlyEstimable >= 26
        x_text = row.pctJointlyEstimable - 2 if inside else row.pctJointlyEstimable + 2
        ax_c.text(
            x_text, row.Index, f"{row.nPairs:,}\n({row.pctJointlyEstimable:.1f}%)",
            va="center", ha="right" if inside else "left", fontsize=4.15,
            color="white" if inside else COLORS["black"],
        )
    ax_c.set_yticks(fy, funnel.criterion, fontsize=4.15)
    ax_c.invert_yaxis(); ax_c.set_xlim(0, 108)
    ax_c.set_xlabel("Percentage of jointly estimable pairs")
    ax_c.set_title(
        "Robustness filters", loc="left", fontsize=7.0,
        fontweight="normal", pad=4,
    )
    funnel.insert(0, "displayOrder", np.arange(len(funnel)))
    funnel.to_csv(SOURCE / "figure5_panel_c_robustness_funnel.csv", index=False)

    # d, study-specific robustness for headline contexts.
    headline_order = {
        (pair, cancer): order
        for order, (pair, cancer) in enumerate(HEADLINE_HETEROGENEITY_CONTEXTS)
    }
    headline = data[
        data[["pair", "cancer"]].apply(tuple, axis=1).isin(headline_order)
    ].merge(heterogeneity, on=["cancer", "pair"], how="left", validate="one_to_one")
    headline["displayOrder"] = (
        headline[["pair", "cancer"]].apply(tuple, axis=1).map(headline_order)
    )
    headline = headline.sort_values("displayOrder").reset_index(drop=True)
    headline["displayLabel"] = (
        headline.pair.str.replace("-", "–", regex=False) + " (" + headline.cancer + ")"
    )
    hy = np.arange(len(headline))
    h_primary = np.log2(headline.noBurden_full_or.to_numpy(float))
    h_primary_lo = np.log2(headline.noBurden_full_ciLow.to_numpy(float))
    h_primary_hi = np.log2(headline.noBurden_full_ciHigh.to_numpy(float))
    h_loo_lo = np.log2(headline.leaveOneStudyOutOrMin.to_numpy(float))
    h_loo_hi = np.log2(headline.leaveOneStudyOutOrMax.to_numpy(float))
    h_random = np.log2(headline.randomEffectsOr.to_numpy(float))
    h_colours = np.where(h_primary < 0, COLORS["blue"], COLORS["vermillion"])
    for row, colour in enumerate(h_colours):
        if np.isfinite(h_loo_lo[row]) and np.isfinite(h_loo_hi[row]):
            ax_d.hlines(
                row, h_loo_lo[row], h_loo_hi[row],
                color=COLORS["light_grey"], lw=3.0, zorder=1,
            )
        ax_d.hlines(row, h_primary_lo[row], h_primary_hi[row], color=colour, lw=0.9, zorder=2)
        ax_d.scatter(
            h_primary[row], row, s=18, color=colour,
            edgecolor="white", lw=0.3, zorder=4,
        )
        if headline.nFiniteStudyEstimates.iloc[row] >= 2 and np.isfinite(h_random[row]):
            ax_d.scatter(
                h_random[row], row + 0.18, s=12, marker="D", color=COLORS["orange"],
                edgecolor="white", lw=0.25, zorder=3,
            )
    ax_d.axvline(0, color=COLORS["black"], ls=(0, (2, 2)), lw=0.65)
    ax_d.set_yticks(hy, headline.displayLabel, fontsize=4.35)
    ax_d.invert_yaxis()
    study_x_min = float(np.floor(np.nanmin(np.r_[h_primary_lo, h_loo_lo]) - 0.25))
    study_x_max = max(
        5.5, float(np.ceil(np.nanmax(np.r_[h_primary_hi, h_loo_hi]) + 2.0))
    )
    ax_d.set_xlim(study_x_min, study_x_max)
    for row, record in headline.iterrows():
        i2_text = (
            f"I² {record.heterogeneityI2Pct:.0f}%"
            if record.nFiniteStudyEstimates >= 2 and np.isfinite(record.heterogeneityI2Pct)
            else "I² n.e."
        )
        ax_d.text(
            study_x_max - 0.20, row, i2_text, va="center", ha="right", fontsize=4.0,
            color=COLORS["grey"],
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5},
        )
    ax_d.set_xlabel("Primary no-burden common log2 odds ratio")
    ax_d.set_title(
        "Study-level robustness", loc="left", fontsize=7.0,
        fontweight="normal", pad=4,
    )
    heterogeneity_legend = ax_d.legend(
        handles=[
            Line2D([0, 1], [0, 0], lw=3.0, color=COLORS["light_grey"],
                   label="Leave-one-study-out"),
            Line2D([0], [0], marker="o", ls="-", lw=0.8, markersize=3.7,
                   color=COLORS["blue"], label="Primary (95% CI)"),
            Line2D([0], [0], marker="D", ls="", markersize=3.4,
                   color=COLORS["orange"], label="Random-effects"),
        ],
        frameon=True, fancybox=True, ncol=3, fontsize=3.8,
        loc="lower right", bbox_to_anchor=(1.0, 1.015), borderaxespad=0,
        borderpad=0.4, handletextpad=0.4, columnspacing=0.75,
    )
    _finish_legend(heterogeneity_legend)
    headline_source_columns = [
        "displayOrder", "cancer", "pair", "geneA", "geneB", "displayLabel",
        "noBurden_full_or", "noBurden_full_ciLow", "noBurden_full_ciHigh",
        "noBurden_full_p", "noBurden_full_fdr", "leaveTwoOut_full_or",
        "leaveTwoOut_full_fdr", "signStableNoBurdenLeaveTwoOut",
        "effectStableNoBurdenLeaveTwoOut", "nStudiesWithJointCoverage",
        "nStudiesWithInformativeStrata", "nFiniteStudyEstimates",
        "leaveOneStudyOutOrMin", "leaveOneStudyOutOrMax",
        "allLeaveOneStudyOutEstimable", "leaveOneStudyOutDirectionStableAmongEstimable",
        "heterogeneityQ", "heterogeneityDf", "heterogeneityP",
        "heterogeneityI2Pct", "tau2DerSimonianLaird", "randomEffectsOr",
        "randomEffectsCiLow", "randomEffectsCiHigh",
    ]
    headline[headline_source_columns].to_csv(
        SOURCE / "figure5_panel_d_study_heterogeneity.csv", index=False
    )

    fig.subplots_adjust(left=0.155, right=0.985, top=0.925, bottom=0.12)
    aligned_panel_labels(
        fig,
        [(("a", ax_a), ("b", ax_b)), (("c", ax_c), ("d", ax_d))],
    )
    save_figure(fig, FIGURES / "figure5_conditioned_interactions")
    plt.close(fig)


def main() -> None:
    setup_dirs()
    figure1()
    print("Wrote Figure 1")
    figure2()
    print("Wrote Figure 2")
    figure3()
    print("Wrote Figure 3; the complete 89-family matrix remains in Supplementary Data 1")
    figure4()
    print("Wrote Figure 4 (landscape 2 x 2 oncoprints)")
    supplementary_figure8()
    print("Wrote Supplementary Figure 7 (matching landscape 2 x 2 oncoprints)")
    figure5()
    print("Wrote Figure 5")


if __name__ == "__main__":
    main()
