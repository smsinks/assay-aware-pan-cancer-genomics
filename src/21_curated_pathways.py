"""Stage 21 -- exact-reference, callability-aware pathway mutation landscape.

This stage replaces the hand-reconstructed pathway lists used by Stage 14 with the
curated templates distributed as Supplemental Table S3 by Sanchez-Vega et al.
(Cell 2018; PMID 29625050).  The workbook is treated as a frozen external reference.

Important scope
---------------
The TCGA paper integrated selected mutations, copy-number alterations, fusions,
expression and methylation.  The present portal compendium does not have a complete
sample-level mask for all of those modalities.  Consequently, this stage reports only
protein-altering mutations in template genes that (i) have a non-empty published
``OQL for GAM`` entry and (ii) occur in this project's prespecified 1,341-gene
compendium.  It must not be interpreted as a reproduction of the TCGA multi-omic
pathway-alteration frequency.

For primary pathway prevalence, the denominator is restricted to selected tissue
samples with documented WES/WGS assay metadata.  A targeted panel rarely contains an
entire pathway template, so targeted samples are summarized by the fraction of the
represented template covered and are never silently counted as pathway wild type.

Outputs
-------
data/processed/sanchez_vega_pathway_membership.csv
results/tables/pathway_mutation_by_cancer_complete.csv
results/tables/pathway_mutation_by_study_complete.csv
results/tables/pathway_gene_representation.csv
results/source_data/figure6_pathway_heatmap.csv
results/source_data/figure6_pathway_annotations.csv
results/figures/figure6_pathway_landscape.{pdf,svg,png}
results/figures/supplementary/figureS2_pathway_all_cancers.{pdf,svg,png}
"""
from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

from config import EXTERNAL, FIGURES, PROCESSED, TABLES
from nature_style import (
    COLORS,
    aligned_panel_labels,
    apply as apply_style,
    figsize,
    save_figure,
)

warnings.filterwarnings("ignore", message="Unknown extension is not supported")

REFERENCE = EXTERNAL / "references" / "sanchez_vega_2018_table_s3.xlsx"
REFERENCE_SHA256 = "35fe8cbb97ea55fd654c9da5ada32602808c1924bd3f524957e89c9ac8acc81b"

PATHWAY_SHEETS = [
    "Cell Cycle",
    "HIPPO",
    "MYC",
    "NOTCH",
    "NRF2",
    "PI3K",
    "TGF-Beta",
    "RTK RAS",
    "TP53",
    "WNT",
]

# Compact display labels retain an exact mapping to the workbook sheet names.
PATHWAY_LABELS = {
    "Cell Cycle": "Cell cycle",
    "HIPPO": "Hippo",
    "MYC": "MYC",
    "NOTCH": "NOTCH",
    "NRF2": "NRF2",
    "PI3K": "PI3K–AKT",
    "TGF-Beta": "TGF-β",
    "RTK RAS": "RTK–RAS",
    "TP53": "TP53",
    "WNT": "WNT/β-catenin",
}

MIN_MAIN_WES = 100
# The main landscape remains readable at journal width with 36 reviewed cancer
# families.  This extends the previous 28-family display while Supplementary Fig. S2
# continues to show the complete 89-family universe.
N_MAIN_CANCERS = 36

LEGEND_BOX = {
    "frameon": True,
    "fancybox": True,
    "framealpha": 0.96,
    "facecolor": "#FAFAFA",
    "edgecolor": COLORS["light_grey"],
    "borderpad": 0.45,
}


def _nan_euclidean_distances(values: np.ndarray) -> np.ndarray:
    """Return pairwise Euclidean distances using each pair's observed features.

    The squared distance is rescaled by the fraction of shared features, matching
    the usual nan-Euclidean definition.  Pairs without any shared measurement are
    left as NaN for explicit handling by the clustering wrapper.
    """
    values = np.asarray(values, dtype=float)
    n_rows, n_features = values.shape
    finite = np.isfinite(values)
    distances = np.zeros((n_rows, n_rows), dtype=float)
    for left in range(n_rows):
        for right in range(left + 1, n_rows):
            shared = finite[left] & finite[right]
            if not shared.any():
                distance = np.nan
            else:
                difference = values[left, shared] - values[right, shared]
                distance = np.sqrt(
                    (n_features / int(shared.sum()))
                    * float(np.dot(difference, difference))
                )
            distances[left, right] = distances[right, left] = distance
    return distances


def _cluster_cancer_order(matrix: pd.DataFrame) -> list[str]:
    """Cluster estimable cancer profiles while retaining unassayed families.

    Pathway-wise standardisation prevents high-prevalence pathways from dominating
    the solution.  Average linkage is applied to missing-aware nan-Euclidean
    distances, with optimal leaf ordering used to place similar adjacent profiles
    together.  Completely unassayed families have no estimable molecular distance
    and therefore form a deterministic alphabetical block after the clustered rows.
    """
    values = matrix.to_numpy(dtype=float)
    observed_rows = np.isfinite(values).any(axis=1)
    observed_names = matrix.index[observed_rows].astype(str).tolist()
    unassayed_names = sorted(matrix.index[~observed_rows].astype(str).tolist())

    if len(observed_names) < 2:
        return observed_names + unassayed_names

    observed_values = values[observed_rows]
    means = np.nanmean(observed_values, axis=0)
    scales = np.nanstd(observed_values, axis=0)
    scales = np.where(np.isfinite(scales) & (scales > 0), scales, 1.0)
    standardised = (observed_values - means) / scales

    distances = _nan_euclidean_distances(standardised)
    upper = distances[np.triu_indices_from(distances, k=1)]
    finite_upper = upper[np.isfinite(upper)]
    fallback = (
        float(finite_upper.max()) * 1.05
        if finite_upper.size and float(finite_upper.max()) > 0
        else 1.0
    )
    distances[~np.isfinite(distances)] = fallback
    np.fill_diagonal(distances, 0.0)

    condensed = squareform(distances, checks=False)
    hierarchy = linkage(condensed, method="average")
    hierarchy = optimal_leaf_ordering(hierarchy, condensed)
    order = leaves_list(hierarchy)
    clustered_names = [observed_names[position] for position in order]
    return clustered_names + unassayed_names


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_templates(panel: pd.DataFrame) -> pd.DataFrame:
    """Extract every role-annotated reference member and persist an audit table."""
    if not REFERENCE.exists():
        raise FileNotFoundError(
            f"Missing official Supplemental Table S3 workbook: {REFERENCE}"
        )
    observed_hash = _sha256(REFERENCE)
    if observed_hash != REFERENCE_SHA256:
        raise ValueError(
            "The Sanchez-Vega reference workbook does not match the frozen SHA-256: "
            f"{observed_hash}"
        )

    panel_lookup = (
        panel[["hugoSymbol", "entrezGeneId"]]
        .drop_duplicates("hugoSymbol")
        .set_index("hugoSymbol")["entrezGeneId"]
        .to_dict()
    )
    rows: list[dict[str, object]] = []
    for sheet in PATHWAY_SHEETS:
        frame = pd.read_excel(
            REFERENCE,
            sheet_name=sheet,
            skiprows=2 if sheet == "Cell Cycle" else 0,
        )
        frame.columns = [str(column).strip() for column in frame.columns]
        frame = frame.loc[frame["Gene"].notna() & frame["OG/TSG"].notna()].copy()
        for _, record in frame.iterrows():
            # Series indexing preserves the workbook's punctuation-bearing headers;
            # ``itertuples`` does not reliably preserve ``OG/TSG`` across pandas
            # versions and can silently erase the role annotation.
            gene = str(record["Gene"]).strip()
            oql = record.get("OQL for GAM")
            in_gam = pd.notna(oql) and str(oql).strip() != ""
            entrez = panel_lookup.get(gene)
            rows.append(
                {
                    "pathwaySheet": sheet,
                    "pathway": PATHWAY_LABELS[sheet],
                    "gene": gene,
                    "role": str(record.get("OG/TSG", "")).strip(),
                    "publishedOqlForGam": None if pd.isna(oql) else str(oql),
                    "hasPublishedOqlForGam": bool(in_gam),
                    "representedInProjectPanel": entrez is not None,
                    "entrezGeneId": entrez,
                    "usedInMutationOnlyAnalysis": bool(in_gam and entrez is not None),
                    "reference": "Sanchez-Vega et al., Cell 2018",
                    "pmid": 29625050,
                    "referenceWorkbookSha256": observed_hash,
                }
            )

    membership = pd.DataFrame(rows)
    if membership.duplicated(["pathwaySheet", "gene"]).any():
        raise AssertionError("Duplicate gene within a Sanchez-Vega pathway sheet")
    if len(membership) != 243:
        raise AssertionError(f"Expected 243 role-annotated template members, found {len(membership)}")
    used = membership[membership.usedInMutationOnlyAnalysis]
    if len(used) != 130:
        raise AssertionError(f"Expected 130 represented GAM genes, found {len(used)}")
    membership.to_csv(PROCESSED / "sanchez_vega_pathway_membership.csv", index=False)

    # Different pathway templates contribute different numbers of genes to the
    # mutation-only analysis.  Preserve that completeness explicitly so a lower
    # pathway prevalence cannot be mistaken for equivalent template coverage.
    representation = (
        membership.groupby("pathway", sort=False)
        .agg(
            nRoleAnnotatedPublishedGenes=("gene", "nunique"),
            nEligiblePublishedGenes=("hasPublishedOqlForGam", "sum"),
            nGenesRepresented=("usedInMutationOnlyAnalysis", "sum"),
        )
        .reset_index()
    )
    representation["representationPct"] = (
        100
        * representation.nGenesRepresented
        / representation.nEligiblePublishedGenes.replace(0, np.nan)
    )
    representation["analysisScope"] = (
        "Protein-altering mutations in represented genes with a published genomic-alteration role"
    )
    representation["reference"] = "Sanchez-Vega et al., Cell 2018"
    representation["pmid"] = 29625050
    representation.to_csv(TABLES / "pathway_gene_representation.csv", index=False)
    return membership


def build_sample_pathway_table(
    samples: pd.DataFrame,
    mutations: pd.DataFrame,
    membership: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one selected sample row and one sample-by-pathway audit row."""
    assay = pd.read_parquet(PROCESSED / "sample_assay.parquet")
    sample_columns = [
        "sampleId",
        "studyId",
        "patientKey",
        "broadCancerCode",
        "analysisCancerCode",
        "sampleTypeGroup",
    ]
    info = samples[sample_columns].merge(
        assay,
        on=["sampleId", "studyId"],
        how="left",
        validate="one_to_one",
    )
    if info.assayType.isna().any():
        raise AssertionError("Selected samples without an assay assignment")
    info["assayGroup"] = np.select(
        [
            info.assayType.eq("WES/WGS"),
            info.assayType.eq("WES/WGS (assumed; no panel metadata)"),
            info.assayType.eq("Unverified mutation-profile membership"),
            info.assayType.eq("Targeted panel"),
        ],
        ["Documented WES/WGS", "Assumed WES/WGS", "Unverified mutation profile", "Targeted panel"],
        default="Other",
    )

    used = membership[membership.usedInMutationOnlyAnalysis].copy()
    gene_path = used[["entrezGeneId", "pathway"]].copy()
    gene_path["entrezGeneId"] = gene_path.entrezGeneId.astype(int)
    path_sizes = gene_path.groupby("pathway")["entrezGeneId"].nunique().to_dict()

    mutation_events = (
        mutations[["sampleId", "entrezGeneId"]]
        .drop_duplicates()
        .merge(gene_path, on="entrezGeneId", how="inner", validate="many_to_one")
        .drop_duplicates(["sampleId", "pathway"])
        .assign(pathwayMutated=True)
    )

    panels = pd.read_parquet(PROCESSED / "panel_gene_membership.parquet")
    panel_path_counts = (
        panels.merge(gene_path, on="entrezGeneId", how="inner")
        .groupby(["genePanelId", "pathway"])["entrezGeneId"]
        .nunique()
        .rename("nTemplateGenesOnPanel")
        .reset_index()
    )

    paths = [PATHWAY_LABELS[sheet] for sheet in PATHWAY_SHEETS]
    expanded = info.assign(_join=1).merge(
        pd.DataFrame({"pathway": paths, "_join": 1}), on="_join", how="inner"
    ).drop(columns="_join")
    expanded["nTemplateGenesInAnalysis"] = expanded.pathway.map(path_sizes).astype(int)
    expanded = expanded.merge(
        panel_path_counts,
        on=["genePanelId", "pathway"],
        how="left",
        validate="many_to_one",
    )
    expanded["nTemplateGenesOnPanel"] = np.where(
        expanded.assayGroup.isin(["Documented WES/WGS", "Assumed WES/WGS"]),
        expanded.nTemplateGenesInAnalysis,
        expanded.nTemplateGenesOnPanel.fillna(0),
    ).astype(int)
    expanded["templateCoveragePct"] = (
        100 * expanded.nTemplateGenesOnPanel / expanded.nTemplateGenesInAnalysis
    )
    expanded = expanded.merge(
        mutation_events[["sampleId", "pathway", "pathwayMutated"]],
        on=["sampleId", "pathway"],
        how="left",
        validate="one_to_one",
    )
    expanded["pathwayMutated"] = expanded.pathwayMutated.fillna(False).astype(bool)
    return info, expanded


def aggregate_complete(
    expanded: pd.DataFrame,
    group_column: str,
    universe: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate all universe groups, retaining zero-event and unassayed states."""
    paths = [PATHWAY_LABELS[sheet] for sheet in PATHWAY_SHEETS]
    grid = universe[[group_column]].drop_duplicates().assign(_join=1).merge(
        pd.DataFrame({"pathway": paths, "_join": 1}), on="_join", how="inner"
    ).drop(columns="_join")

    def summarize(group: pd.DataFrame) -> pd.Series:
        documented = group.assayGroup.eq("Documented WES/WGS")
        assumed = group.assayGroup.eq("Assumed WES/WGS")
        unverified = group.assayGroup.eq("Unverified mutation profile")
        targeted = group.assayGroup.eq("Targeted panel")
        return pd.Series(
            {
                "nAnalysisSamples": group.sampleId.nunique(),
                # pandas removes grouping columns when ``include_groups=False``.
                # A study-level row is necessarily one study; cancer-level groups
                # retain ``studyId`` and can be counted directly.
                "nStudies": (
                    1 if group_column == "studyId" else group.studyId.nunique()
                ),
                "nDocumentedWesWgs": int(documented.sum()),
                "nAssumedWesWgs": int(assumed.sum()),
                "nUnverifiedMutationProfile": int(unverified.sum()),
                "nTargetedPanel": int(targeted.sum()),
                "nWesWithPathwayMutation": int((documented & group.pathwayMutated).sum()),
                "nTargetedWithAnyRepresentedMutation": int(
                    (targeted & group.pathwayMutated).sum()
                ),
                "meanTargetedTemplateCoveragePct": (
                    float(group.loc[targeted, "templateCoveragePct"].mean())
                    if targeted.any()
                    else np.nan
                ),
                "medianTargetedTemplateCoveragePct": (
                    float(group.loc[targeted, "templateCoveragePct"].median())
                    if targeted.any()
                    else np.nan
                ),
                "minTargetedTemplateCoveragePct": (
                    float(group.loc[targeted, "templateCoveragePct"].min())
                    if targeted.any()
                    else np.nan
                ),
                "maxTargetedTemplateCoveragePct": (
                    float(group.loc[targeted, "templateCoveragePct"].max())
                    if targeted.any()
                    else np.nan
                ),
            }
        )

    summary = (
        expanded.groupby([group_column, "pathway"], dropna=False, observed=True)
        .apply(summarize, include_groups=False)
        .reset_index()
    )
    out = grid.merge(summary, on=[group_column, "pathway"], how="left", validate="one_to_one")
    count_columns = [column for column in out.columns if column.startswith("n")]
    out[count_columns] = out[count_columns].fillna(0).astype(int)
    out["wesMutationPct"] = np.where(
        out.nDocumentedWesWgs.gt(0),
        100 * out.nWesWithPathwayMutation / out.nDocumentedWesWgs,
        np.nan,
    )
    out["targetedPartialCoverageMutationPct"] = np.where(
        out.nTargetedPanel.gt(0),
        100 * out.nTargetedWithAnyRepresentedMutation / out.nTargetedPanel,
        np.nan,
    )
    out["primaryEstimateStatus"] = np.where(
        out.nDocumentedWesWgs.gt(0), "estimated", "not assayed by documented WES/WGS"
    )
    out["targetedEstimateStatus"] = np.where(
        out.nTargetedPanel.gt(0),
        "partial template coverage; descriptive only",
        "no targeted-panel samples",
    )
    out["analysisDefinition"] = (
        "At least one retained protein-altering mutation among represented genes with a "
        "published Sanchez-Vega OQL-for-GAM entry; mutation-only, not TCGA multi-omic frequency"
    )
    return out


def _heatmap(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    *,
    annotate: bool,
    vmax: float,
    label_size: float = 5.4,
    annotation_size: float = 4.2,
) -> None:
    cmap = LinearSegmentedColormap.from_list(
        "pathway_mutation", ["#F7FBFF", "#BFD7EA", COLORS["blue"], "#08306B"]
    )
    # A neutral masked state prevents an unassayed cancer group from being read as a
    # true zero.  The supplementary display carries the matching legend swatch.
    cmap.set_bad("#E6E6E6")
    masked = np.ma.masked_invalid(matrix.to_numpy(float))
    image = ax.imshow(masked, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=vmax)
    ax.set_xticks(np.arange(matrix.shape[1]), matrix.columns, rotation=55, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]), matrix.index)
    ax.tick_params(length=0, labelsize=label_size)
    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.35)
    if annotate:
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix.iat[row, column]
                if not np.isfinite(value):
                    label = "—"
                    color = COLORS["grey"]
                else:
                    label = f"{value:.0f}"
                    color = "white" if value > 0.58 * vmax else COLORS["black"]
                ax.text(
                    column,
                    row,
                    label,
                    ha="center",
                    va="center",
                    fontsize=annotation_size,
                    color=color,
                )
    ax._pathway_image = image  # type: ignore[attr-defined]


def make_main_figure(cancer: pd.DataFrame, membership: pd.DataFrame) -> None:
    apply_style()
    source = TABLES.parent / "source_data"
    source.mkdir(parents=True, exist_ok=True)

    cancer_meta = (
        cancer.groupby("cancer", as_index=False)
        .agg(
            nAnalysisSamples=("nAnalysisSamples", "max"),
            nStudies=("nStudies", "max"),
            nDocumentedWesWgs=("nDocumentedWesWgs", "max"),
            nTargetedPanel=("nTargetedPanel", "max"),
        )
    )
    shown = (
        cancer_meta[cancer_meta.nDocumentedWesWgs >= MIN_MAIN_WES]
        .sort_values(["nAnalysisSamples", "nDocumentedWesWgs"], ascending=False)
        .head(N_MAIN_CANCERS)
        .cancer.tolist()
    )
    subset = cancer[cancer.cancer.isin(shown)].copy()
    pathway_order = [PATHWAY_LABELS[sheet] for sheet in PATHWAY_SHEETS]
    cancer_order = (
        cancer_meta.set_index("cancer").loc[shown].sort_values("nAnalysisSamples", ascending=False).index.tolist()
    )
    matrix = (
        subset.pivot(index="pathway", columns="cancer", values="wesMutationPct")
        .reindex(index=pathway_order, columns=cancer_order)
    )
    vmax = max(45.0, float(np.nanpercentile(matrix.to_numpy(float), 98)))

    # A deliberately landscape canvas allows the additional cancer families to be
    # displayed without compressing the pathway labels or the pan-cancer marginal.
    # The narrow WES/WGS track now sits between panels a and b, so no separate
    # bottom band is required.
    fig = plt.figure(figsize=figsize(180, 100))
    gs = GridSpec(
        3,
        4,
        figure=fig,
        width_ratios=[0.021, 0.036, 1.0, 0.18],
        height_ratios=[0.29, 0.12, 1.0],
        hspace=0.055,
        wspace=0.09,
    )
    ax_top = fig.add_subplot(gs[0, 2])
    ax_wes = fig.add_subplot(gs[1, 2])
    ax_heat = fig.add_subplot(gs[2, 2])
    # Constrain the prevalence colour scale to the central 62% of its GridSpec
    # cell.  Its position at the far left and the intervening spacer distinguish it
    # unambiguously from both the pathway matrix and panel c.
    cbar_grid = gs[2, 0].subgridspec(3, 1, height_ratios=[0.19, 0.62, 0.19])
    ax_cbar = fig.add_subplot(cbar_grid[1, 0])
    ax_side = fig.add_subplot(gs[2, 3])

    meta = cancer_meta.set_index("cancer").loc[cancer_order]
    x = np.arange(len(cancer_order))
    ax_top.bar(x, meta.nDocumentedWesWgs, color=COLORS["blue"], width=0.78, label="Documented WES/WGS")
    ax_top.bar(
        x,
        meta.nTargetedPanel,
        bottom=meta.nDocumentedWesWgs,
        color=COLORS["purple"],
        width=0.78,
        alpha=0.78,
        label="Targeted panel (excluded from prevalence)",
    )
    ax_top.set_xlim(-0.5, len(x) - 0.5)
    ax_top.set_ylabel("Cases", labelpad=2)
    ax_top.set_xticks([])
    ax_top.spines[["top", "right", "bottom"]].set_visible(False)
    ax_top.legend(
        **LEGEND_BOX,
        ncol=2,
        loc="upper right",
        fontsize=5.1,
        title="Assay",
        title_fontsize=5.2,
        handlelength=1.3,
        columnspacing=1.2,
    )

    # Sequential assay-size annotation track.  A square-root normalisation retains
    # useful contrast across the 100--4,500-case range while the printed count
    # preserves the exact denominator for every displayed cancer family.
    wes_values = meta.nDocumentedWesWgs.to_numpy(dtype=float)[None, :]
    wes_cmap = LinearSegmentedColormap.from_list(
        "wes_count", ["#F7FBFF", "#9ECAE1", COLORS["blue"], "#08306B"]
    )
    wes_norm = PowerNorm(gamma=0.55, vmin=0, vmax=float(np.nanmax(wes_values)))
    ax_wes.imshow(
        wes_values,
        aspect="auto",
        interpolation="nearest",
        cmap=wes_cmap,
        norm=wes_norm,
    )
    ax_wes.set_yticks([0], ["n WES/WGS"], fontsize=4.5)
    ax_wes.set_xticks([])
    ax_wes.tick_params(length=0, pad=1.5)
    ax_wes.set_xticks(np.arange(-0.5, len(cancer_order), 1), minor=True)
    ax_wes.grid(which="minor", axis="x", color="white", linewidth=0.35)
    ax_wes.spines[:].set_visible(False)
    for column, count in enumerate(wes_values.ravel()):
        text_colour = "white" if wes_norm(count) > 0.57 else COLORS["black"]
        ax_wes.text(
            column,
            0,
            f"{int(count):,}",
            rotation=90,
            ha="center",
            va="center",
            fontsize=3.35,
            color=text_colour,
        )

    _heatmap(ax_heat, matrix, annotate=True, vmax=vmax)
    ax_heat.set_xlabel("")
    ax_heat.set_ylabel("")
    ax_heat.set_xticklabels(matrix.columns, rotation=58, ha="right", fontsize=4.55)
    colorbar = fig.colorbar(ax_heat._pathway_image, cax=ax_cbar)  # type: ignore[attr-defined]
    colorbar.set_ticks(np.linspace(0, vmax, 4))
    colorbar.set_ticklabels([f"{value:.0f}" for value in np.linspace(0, vmax, 4)])
    colorbar.ax.yaxis.set_ticks_position("left")
    colorbar.ax.yaxis.set_label_position("left")
    colorbar.set_label("WES/WGS with ≥1 pathway mutation (%)", fontsize=4.6, labelpad=2.0)
    colorbar.ax.tick_params(labelsize=4.3, pad=1.2)

    pan = (
        cancer.groupby("pathway", as_index=False)
        .agg(
            nWesWithPathwayMutation=("nWesWithPathwayMutation", "sum"),
            nDocumentedWesWgs=("nDocumentedWesWgs", "sum"),
        )
        .set_index("pathway")
        .reindex(pathway_order)
    )
    pan["pct"] = 100 * pan.nWesWithPathwayMutation / pan.nDocumentedWesWgs.clip(lower=1)
    if pan.nDocumentedWesWgs.nunique() != 1 or int(pan.nDocumentedWesWgs.iloc[0]) != 54_249:
        raise AssertionError("Figure 6c requires the frozen 54,249-case documented-WES/WGS denominator")
    y = np.arange(len(pathway_order))
    ax_side.barh(y, pan.pct, color=COLORS["orange"], height=0.68)
    ax_side.set_ylim(len(pathway_order) - 0.5, -0.5)
    ax_side.set_yticks([])
    ax_side.set_xlabel("Pan-cancer\nmutation (%)")
    ax_side.grid(axis="x", color=COLORS["very_light_grey"], lw=0.5)
    for yy, value in zip(y, pan.pct):
        ax_side.text(value, yy, f" {value:.0f}", va="center", fontsize=4.5)

    template_counts = (
        membership.groupby("pathway")
        .agg(
            nTemplateMembers=("gene", "nunique"),
            nPublishedGamGenes=("hasPublishedOqlForGam", "sum"),
            nRepresentedGamGenes=("usedInMutationOnlyAnalysis", "sum"),
        )
        .reindex(pathway_order)
        .reset_index()
    )
    template_counts.to_csv(source / "figure6_pathway_annotations.csv", index=False)
    (
        meta.reset_index()[
            [
                "cancer",
                "nAnalysisSamples",
                "nStudies",
                "nDocumentedWesWgs",
                "nTargetedPanel",
            ]
        ]
        .to_csv(source / "figure6_panel_a_cohort_counts.csv", index=False)
    )
    subset.to_csv(source / "figure6_pathway_heatmap.csv", index=False)
    (
        pan.reset_index()
        .rename(
            columns={
                "nWesWithPathwayMutation": "nMutatedDocumentedWesWgs",
                "pct": "panCancerMutationPct",
            }
        )[
            [
                "pathway",
                "nMutatedDocumentedWesWgs",
                "nDocumentedWesWgs",
                "panCancerMutationPct",
            ]
        ]
        .to_csv(source / "figure6_panel_c_pan_cancer_pathway_marginals.csv", index=False)
    )

    fig.subplots_adjust(left=0.145, right=0.985, top=0.95, bottom=0.19)
    # Figure 6 is a single integrated pathway landscape with aligned cohort and
    # pan-cancer marginal tracks; panel letters are intentionally omitted.
    save_figure(fig, FIGURES / "figure6_pathway_landscape")
    plt.close(fig)


def make_supplementary_figure(cancer: pd.DataFrame) -> None:
    """Display every cancer family in one integrated pathway landscape."""
    apply_style()
    supplementary = FIGURES / "supplementary"
    supplementary.mkdir(parents=True, exist_ok=True)
    pathway_order = [PATHWAY_LABELS[sheet] for sheet in PATHWAY_SHEETS]

    meta = (
        cancer.groupby("cancer", as_index=False)
        .agg(
            nAnalysisSamples=("nAnalysisSamples", "max"),
            nStudies=("nStudies", "max"),
            nDocumentedWesWgs=("nDocumentedWesWgs", "max"),
            nTargetedPanel=("nTargetedPanel", "max"),
        )
    )
    matrix = cancer.pivot(
        index="cancer", columns="pathway", values="wesMutationPct"
    ).reindex(columns=pathway_order)
    cancer_order = _cluster_cancer_order(matrix)
    if len(cancer_order) != 89:
        raise AssertionError(f"Expected 89 cancer families, found {len(cancer_order)}")
    matrix = matrix.reindex(index=cancer_order)
    vmax = max(45.0, float(np.nanpercentile(cancer.wesMutationPct.to_numpy(float), 98)))

    pan = (
        cancer.groupby("pathway", as_index=False)
        .agg(
            nMutatedDocumentedWesWgs=("nWesWithPathwayMutation", "sum"),
            nDocumentedWesWgs=("nDocumentedWesWgs", "sum"),
        )
        .set_index("pathway")
        .reindex(pathway_order)
    )
    pan["panCancerMutationPct"] = (
        100 * pan.nMutatedDocumentedWesWgs / pan.nDocumentedWesWgs.clip(lower=1)
    )
    if pan.nDocumentedWesWgs.nunique() != 1 or int(pan.nDocumentedWesWgs.iloc[0]) != 54_249:
        raise AssertionError(
            "Supplementary Figure S2 requires the frozen 54,249-case WES/WGS denominator"
        )

    standard = pd.read_csv(TABLES / "assay_frequency_standardized.csv")
    prevalence = pd.read_csv(TABLES / "gene_frequencies_curated.csv")
    assay_comparison = standard.loc[standard.nCommonCancers.ge(2)].merge(
        prevalence[["gene", "nMutated"]], on="gene", how="left", validate="one_to_one"
    )
    if len(assay_comparison) != 738:
        raise AssertionError(
            "Supplementary Figure S2b requires the frozen 738-gene cross-assay "
            f"comparison; found {len(assay_comparison)}"
        )

    # Panel a remains one integrated all-cancer pathway landscape. Panel b restores
    # the cancer-standardised WES/WGS-versus-targeted-panel prevalence comparison
    # formerly shown in the main recurrence figure.
    fig = plt.figure(figsize=figsize(220, 245))
    grid = GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=[0.82, 0.56, 0.36],
        height_ratios=[1.0, 0.23],
        hspace=0.055,
        wspace=0.12,
    )
    ax_heat = fig.add_subplot(grid[0, 0:2])
    ax_cohort = fig.add_subplot(grid[0, 2], sharey=ax_heat)
    ax_pan = fig.add_subplot(grid[1, 0])
    ax_assay = fig.add_subplot(grid[1, 1])
    ax_key = fig.add_subplot(grid[1, 2])

    _heatmap(
        ax_heat,
        matrix,
        annotate=True,
        vmax=vmax,
        label_size=3.45,
        annotation_size=2.65,
    )
    ax_heat.set_xlabel("")
    ax_heat.set_ylabel("")
    ax_heat.tick_params(
        axis="x",
        top=True,
        labeltop=True,
        bottom=False,
        labelbottom=False,
        pad=1.2,
    )
    ax_heat.set_xticklabels(
        matrix.columns,
        rotation=42,
        ha="left",
        rotation_mode="anchor",
        fontsize=4.4,
    )
    ax_heat.set_yticklabels(matrix.index, fontsize=3.45)

    ordered_meta = meta.set_index("cancer").loc[cancer_order]
    y = np.arange(len(cancer_order))
    ax_cohort.barh(
        y - 0.18,
        ordered_meta.nDocumentedWesWgs,
        height=0.32,
        color=COLORS["blue"],
        label="Documented WES/WGS",
    )
    ax_cohort.barh(
        y + 0.18,
        ordered_meta.nTargetedPanel,
        height=0.32,
        color=COLORS["purple"],
        alpha=0.82,
        label="Targeted panel",
    )
    ax_cohort.set_xscale("log")
    ax_cohort.set_xlim(0.8, max(10_000, float(ordered_meta[["nDocumentedWesWgs", "nTargetedPanel"]].max().max()) * 1.18))
    ax_cohort.set_ylim(len(cancer_order) - 0.5, -0.5)
    ax_cohort.tick_params(axis="y", left=False, labelleft=False)
    ax_cohort.tick_params(axis="x", labelsize=4.0, pad=1.5)
    ax_cohort.set_xlabel("Cases (log scale)", fontsize=5.0, labelpad=2.0)
    ax_cohort.grid(axis="x", color=COLORS["very_light_grey"], linewidth=0.45)
    cohort_legend = ax_cohort.legend(
        **LEGEND_BOX,
        loc="upper right",
        fontsize=4.15,
        ncol=1,
        handlelength=1.2,
        labelspacing=0.28,
    )
    cohort_legend.get_frame().set_linewidth(0.45)

    x = np.arange(len(pathway_order))
    ax_pan.bar(x, pan.panCancerMutationPct, color=COLORS["orange"], width=0.70)
    ax_pan.set_xlim(-0.5, len(pathway_order) - 0.5)
    ax_pan.set_xticks([])
    ax_pan.set_ylabel("Pan-cancer\nmutation (%)", fontsize=5.0, labelpad=2.0)
    ax_pan.tick_params(axis="y", labelsize=4.1)
    ax_pan.grid(axis="y", color=COLORS["very_light_grey"], linewidth=0.45)
    for column, value in enumerate(pan.panCancerMutationPct):
        ax_pan.text(
            column,
            value + float(pan.panCancerMutationPct.max()) * 0.018,
            f"{value:.0f}",
            ha="center",
            va="top",
            fontsize=3.7,
            color=COLORS["black"],
        )
    ax_pan.set_ylim(float(pan.panCancerMutationPct.max()) * 1.20, 0)

    ax_key.axis("off")
    cax = ax_key.inset_axes([0.08, 0.58, 0.84, 0.12])
    colorbar = fig.colorbar(ax_heat._pathway_image, cax=cax, orientation="horizontal")  # type: ignore[attr-defined]
    colorbar.set_label("WES/WGS mutation (%)", fontsize=4.8, labelpad=1.5)
    colorbar.ax.tick_params(labelsize=4.0, pad=1.0)
    ax_key.legend(
        handles=[
            Patch(
                facecolor="#E6E6E6",
                edgecolor="white",
                label="No documented WES/WGS",
            )
        ],
        **LEGEND_BOX,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        fontsize=4.2,
        handlelength=1.2,
    )

    assay_scatter = ax_assay.scatter(
        assay_comparison.freqWesStandardizedPct,
        assay_comparison.freqPanelStandardizedPct,
        s=np.clip(np.sqrt(assay_comparison.nMutated), 3, 18),
        c=assay_comparison.nCommonCancers,
        cmap="cividis",
        alpha=0.60,
        edgecolors="none",
    )
    assay_limit = 1.06 * max(
        float(assay_comparison.freqWesStandardizedPct.max()),
        float(assay_comparison.freqPanelStandardizedPct.max()),
    )
    ax_assay.plot(
        [0, assay_limit], [0, assay_limit], color=COLORS["grey"],
        lw=0.65, ls=(0, (2, 2)),
    )
    label_offsets = {
        "CSMD3": (3, 2), "TP53": (3, -8), "MUC16": (3, -7),
        "TRRAP": (-22, -8), "KRAS": (3, 3), "PIK3CA": (-23, -8),
    }
    for row in assay_comparison.loc[
        assay_comparison.gene.isin(label_offsets)
    ].itertuples(index=False):
        ax_assay.annotate(
            row.gene,
            (row.freqWesStandardizedPct, row.freqPanelStandardizedPct),
            xytext=label_offsets[row.gene],
            textcoords="offset points",
            fontsize=3.7,
        )
    rho = float(
        spearmanr(
            assay_comparison.freqWesStandardizedPct,
            assay_comparison.freqPanelStandardizedPct,
        ).statistic
    )
    ax_assay.text(
        0.04, 0.96, f"n=738 genes\nSpearman ρ={rho:.3f}",
        transform=ax_assay.transAxes, va="top", fontsize=4.0,
        bbox={
            "boxstyle": "round,pad=0.28", "facecolor": "white",
            "edgecolor": COLORS["light_grey"], "linewidth": 0.45, "alpha": 0.95,
        },
    )
    ax_assay.set_xlim(0, assay_limit)
    ax_assay.set_ylim(0, assay_limit)
    ax_assay.set_aspect("equal", adjustable="box")
    ax_assay.set_xlabel("WES/WGS prevalence, cancer-standardised (%)", fontsize=4.3)
    ax_assay.set_ylabel("Targeted-panel prevalence, same weights (%)", fontsize=4.3)
    assay_cax = ax_key.inset_axes([0.08, 0.28, 0.84, 0.08])
    assay_bar = fig.colorbar(assay_scatter, cax=assay_cax, orientation="horizontal")
    assay_bar.set_label("Shared cancer groups (n)", fontsize=4.0, labelpad=1.2)
    assay_bar.ax.tick_params(labelsize=3.6, pad=0.8, length=1.2)

    assay_comparison.to_csv(
        TABLES.parent / "source_data" / "figureS2_panel_b_assay_concordance.csv",
        index=False,
    )

    fig.subplots_adjust(left=0.095, right=0.986, top=0.936, bottom=0.064)
    aligned_panel_labels(fig, [(('a', ax_heat),), (('b', ax_assay),)])
    save_figure(fig, supplementary / "figureS2_pathway_all_cancers")
    plt.close(fig)


def main() -> None:
    panel = pd.read_csv(PROCESSED / "gene_panel.csv")
    samples = pd.read_parquet(PROCESSED / "analysis_samples_curated.parquet")
    samples = samples.loc[samples.analysisEligible].copy()
    mutations = pd.read_parquet(PROCESSED / "mutations_curated.parquet")

    membership = load_templates(panel)
    info, expanded = build_sample_pathway_table(samples, mutations, membership)

    cancer_universe = info[["broadCancerCode"]].drop_duplicates().rename(
        columns={"broadCancerCode": "cancer"}
    )
    cancer_expanded = expanded.rename(columns={"broadCancerCode": "cancer"})
    cancer = aggregate_complete(cancer_expanded, "cancer", cancer_universe)
    cancer = cancer.sort_values(["nAnalysisSamples", "cancer", "pathway"], ascending=[False, True, True])
    cancer.to_csv(TABLES / "pathway_mutation_by_cancer_complete.csv", index=False)

    study_universe = info[["studyId"]].drop_duplicates()
    study = aggregate_complete(expanded, "studyId", study_universe)
    study = study.sort_values(["nAnalysisSamples", "studyId", "pathway"], ascending=[False, True, True])
    study.to_csv(TABLES / "pathway_mutation_by_study_complete.csv", index=False)

    if cancer.cancer.nunique() != samples.broadCancerCode.nunique():
        raise AssertionError("Not every eligible broad cancer group was retained")
    if study.studyId.nunique() != samples.studyId.nunique():
        raise AssertionError("Not every contributing curated study was retained")
    if len(study) != samples.studyId.nunique() * len(PATHWAY_SHEETS):
        raise AssertionError("Study-by-pathway grid is incomplete")

    make_main_figure(cancer, membership)
    make_supplementary_figure(cancer)

    print(
        f"Reference templates: {len(membership):,} role-annotated members; "
        f"{membership.hasPublishedOqlForGam.sum():,} with OQL-for-GAM; "
        f"{membership.usedInMutationOnlyAnalysis.sum():,} represented in the project panel."
    )
    print(
        f"Complete outputs: {cancer.cancer.nunique():,} cancer groups x 10 pathways; "
        f"{study.studyId.nunique():,} studies x 10 pathways."
    )
    print("Wrote Figure 6, Supplementary Figure S2, and pathway audit tables.")


if __name__ == "__main__":
    main()
