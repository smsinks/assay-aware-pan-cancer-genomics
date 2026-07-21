"""Stage 25 -- COSMIC-hallmark mutation landscape.

The event unit is a *unique sample--hallmark pair*: a specimen is counted
once for a hallmark when it has at least one retained protein-altering mutation in any
panel gene annotated to that hallmark, irrespective of how many annotated genes or
mutation records occur in the specimen.  Primary prevalence denominators contain only
selected tissue specimens with a documented WES/WGS assignment.  The 153 presumed
WES/WGS specimens and all targeted-panel specimens are reported as cohort context but
are not silently treated as fully assayed for every hallmark.

Hallmarks are not mutually exclusive biological categories.  A gene, and therefore a
specimen, may validly contribute to more than one hallmark; percentages must not be
summed across hallmarks.

Outputs
-------
data/processed/cosmic_hallmark_membership_curated.csv
results/tables/hallmark_mutation_by_cancer_complete.csv
results/tables/hallmark_mutation_by_study_complete.csv
results/source_data/figureS5_hallmark_heatmap.csv
results/source_data/figureS5_hallmark_annotations.csv
results/source_data/figureS5_hallmark_denominators.csv
results/figures/supplementary/figureS5_hallmark_landscape.{pdf,svg,png}
"""
from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
from scipy.spatial.distance import squareform

from config import FIGURES, PROCESSED, TABLES
from plot_style import COLORS, apply as apply_style, figsize, save_figure


EXPECTED_CANCER_GROUPS = 89
EXPECTED_STUDIES = 367
EXPECTED_HALLMARKS = 13
EXPECTED_HALLMARK_GENES = 323
EXPECTED_GENE_HALLMARK_PAIRS = 1_294
EXPECTED_ELIGIBLE_SAMPLES = 132_181
EXPECTED_UNVERIFIED_PROFILE = 153
EXPECTED_TARGETED = 77_779
EXPECTED_DOCUMENTED_WES_WGS = 54_249

N_DISPLAY_CANCERS = 60

LEGEND_BOX = {
    "frameon": True,
    "fancybox": True,
    "framealpha": 0.96,
    "facecolor": "#FAFAFA",
    "edgecolor": COLORS["light_grey"],
    "borderpad": 0.45,
}

BREWER_PURPLE = "#756BB1"
BREWER_GREEN = "#238B45"
BREWER_ORANGE = "#D95F0E"


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise KeyError(f"{label} lacks required columns: {sorted(missing)}")


def _nan_euclidean_distances(values: np.ndarray) -> np.ndarray:
    """Calculate pairwise distances from the features shared by each pair."""
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


def _cluster_profile_order(matrix: pd.DataFrame) -> list[str]:
    """Order measured profiles by missing-aware average-linkage clustering."""
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
    clustered = [observed_names[position] for position in leaves_list(hierarchy)]
    return clustered + unassayed_names


def _wilson_interval(successes: pd.Series, totals: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Return two-sided 95% Wilson intervals as percentages."""
    k = successes.to_numpy(float)
    n = totals.to_numpy(float)
    low = np.full(len(k), np.nan)
    high = np.full(len(k), np.nan)
    valid = n > 0
    if valid.any():
        z = 1.959963984540054
        p = k[valid] / n[valid]
        denominator = 1 + z * z / n[valid]
        centre = (p + z * z / (2 * n[valid])) / denominator
        half = (
            z
            * np.sqrt(p * (1 - p) / n[valid] + z * z / (4 * n[valid] ** 2))
            / denominator
        )
        low[valid] = 100 * np.clip(centre - half, 0, 1)
        high[valid] = 100 * np.clip(centre + half, 0, 1)
    return low, high


def load_membership() -> pd.DataFrame:
    """Explode the frozen gene-panel COSMIC hallmark field into an audit table."""
    panel = pd.read_csv(PROCESSED / "gene_panel.csv")
    _require_columns(
        panel,
        {"entrezGeneId", "hugoSymbol", "roleInCancer", "cosmicTier", "hallmarks"},
        "gene_panel.csv",
    )
    membership = panel.dropna(subset=["hallmarks"])[
        ["entrezGeneId", "hugoSymbol", "roleInCancer", "cosmicTier", "hallmarks"]
    ].copy()
    membership["hallmark"] = membership.hallmarks.astype(str).str.split(";")
    membership = membership.explode("hallmark", ignore_index=True)
    membership["hallmark"] = membership.hallmark.astype(str).str.strip()
    membership = membership.loc[membership.hallmark.ne("")].drop(columns="hallmarks")
    membership["entrezGeneId"] = membership.entrezGeneId.astype(int)
    membership = membership.drop_duplicates(["entrezGeneId", "hallmark"])
    membership = membership.rename(columns={"hugoSymbol": "gene"})
    membership["source"] = "COSMIC Cancer Gene Census v104 gene-level Hallmark field"
    membership["countingRule"] = (
        "A documented-WES/WGS specimen is counted at most once within this hallmark"
    )
    membership = membership.sort_values(["hallmark", "gene"]).reset_index(drop=True)

    if membership.hallmark.nunique() != EXPECTED_HALLMARKS:
        raise AssertionError(
            f"Expected {EXPECTED_HALLMARKS} hallmarks, found {membership.hallmark.nunique()}"
        )
    if membership.entrezGeneId.nunique() != EXPECTED_HALLMARK_GENES:
        raise AssertionError(
            f"Expected {EXPECTED_HALLMARK_GENES} hallmark genes, found "
            f"{membership.entrezGeneId.nunique()}"
        )
    if len(membership) != EXPECTED_GENE_HALLMARK_PAIRS:
        raise AssertionError(
            f"Expected {EXPECTED_GENE_HALLMARK_PAIRS} unique gene--hallmark pairs, "
            f"found {len(membership)}"
        )

    membership.to_csv(
        PROCESSED / "cosmic_hallmark_membership_curated.csv", index=False
    )
    return membership


def load_sample_events(
    membership: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return exact eligible samples and deduplicated sample--hallmark mutation events."""
    samples = pd.read_parquet(PROCESSED / "analysis_samples_curated.parquet")
    _require_columns(
        samples,
        {
            "studyId",
            "sampleId",
            "patientKey",
            "broadCancerCode",
            "analysisEligible",
        },
        "analysis_samples_curated.parquet",
    )
    samples = samples.loc[samples.analysisEligible].copy()
    samples = samples[
        ["studyId", "sampleId", "patientKey", "broadCancerCode"]
    ].rename(columns={"broadCancerCode": "cancerGroup"})
    samples["sampleKey"] = samples.studyId.astype(str) + "::" + samples.sampleId.astype(str)
    if samples.duplicated(["studyId", "sampleId"]).any() or samples.sampleKey.duplicated().any():
        raise AssertionError("Eligible samples are not unique on studyId + sampleId")
    if samples.patientKey.duplicated().any():
        raise AssertionError("Primary hallmark cohort is not one specimen per patient key")

    assay = pd.read_parquet(PROCESSED / "sample_assay.parquet")
    _require_columns(
        assay,
        {"studyId", "sampleId", "assayType", "genePanelId"},
        "sample_assay.parquet",
    )
    if assay.duplicated(["studyId", "sampleId"]).any():
        raise AssertionError("Assay assignments are not unique on studyId + sampleId")
    samples = samples.merge(
        assay[["studyId", "sampleId", "assayType", "genePanelId"]],
        on=["studyId", "sampleId"],
        how="left",
        validate="one_to_one",
    )
    if samples.assayType.isna().any():
        raise AssertionError("An eligible sample lacks an assay assignment")
    samples["assayStratum"] = np.select(
        [
            samples.assayType.eq("WES/WGS"),
            samples.assayType.eq("Unverified mutation-profile membership"),
            samples.assayType.eq("Targeted panel"),
        ],
        ["Documented WES/WGS", "Unverified mutation profile", "Targeted panel"],
        default="Unexpected",
    )
    if samples.assayStratum.eq("Unexpected").any():
        unexpected = sorted(samples.loc[samples.assayStratum.eq("Unexpected"), "assayType"].unique())
        raise AssertionError(f"Unexpected assay assignments: {unexpected}")

    assay_counts = samples.assayStratum.value_counts()
    observed = {
        "Documented WES/WGS": int(assay_counts.get("Documented WES/WGS", 0)),
        "Unverified mutation profile": int(assay_counts.get("Unverified mutation profile", 0)),
        "Targeted panel": int(assay_counts.get("Targeted panel", 0)),
    }
    expected = {
        "Documented WES/WGS": EXPECTED_DOCUMENTED_WES_WGS,
        "Unverified mutation profile": EXPECTED_UNVERIFIED_PROFILE,
        "Targeted panel": EXPECTED_TARGETED,
    }
    if len(samples) != EXPECTED_ELIGIBLE_SAMPLES or observed != expected:
        raise AssertionError(
            f"Cohort/assay contract changed: n={len(samples):,}; strata={observed}"
        )
    if samples.cancerGroup.nunique() != EXPECTED_CANCER_GROUPS:
        raise AssertionError("Eligible cancer-family universe does not contain 89 families")
    if samples.studyId.nunique() != EXPECTED_STUDIES:
        raise AssertionError("Eligible study universe does not contain 367 studies")

    mutations = pd.read_parquet(PROCESSED / "mutations_curated.parquet")
    _require_columns(
        mutations,
        {"studyId", "sampleId", "entrezGeneId"},
        "mutations_curated.parquet",
    )
    # Multiple variants in one gene are collapsed before the gene can be expanded to
    # multiple biological hallmarks.  A second collapse then enforces one event per
    # sample and hallmark, which is the numerator contract used throughout this stage.
    mutation_genes = mutations[["studyId", "sampleId", "entrezGeneId"]].drop_duplicates()
    mutation_genes["entrezGeneId"] = mutation_genes.entrezGeneId.astype(int)
    events = mutation_genes.merge(
        membership[["entrezGeneId", "hallmark"]],
        on="entrezGeneId",
        how="inner",
        validate="many_to_many",
    )
    events = events[["studyId", "sampleId", "hallmark"]].drop_duplicates()
    events = events.merge(
        samples[["studyId", "sampleId", "sampleKey"]],
        on=["studyId", "sampleId"],
        how="inner",
        validate="many_to_one",
    )
    if events.duplicated(["sampleKey", "hallmark"]).any():
        raise AssertionError("Sample--hallmark mutation events were not fully deduplicated")
    return samples, events


def aggregate_complete(
    samples: pd.DataFrame,
    events: pd.DataFrame,
    membership: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    """Build a complete group-by-hallmark table, retaining zero and unassayed states."""
    work = samples.copy()
    work["_documented"] = work.assayStratum.eq("Documented WES/WGS").astype(int)
    work["_unverified"] = work.assayStratum.eq("Unverified mutation profile").astype(int)
    work["_targeted"] = work.assayStratum.eq("Targeted panel").astype(int)
    aggregation: dict[str, tuple[str, str]] = {
        "nEligibleSamples": ("sampleKey", "nunique"),
        "nEligiblePatientKeys": ("patientKey", "nunique"),
        "nDocumentedWesWgs": ("_documented", "sum"),
        "nUnverifiedMutationProfile": ("_unverified", "sum"),
        "nTargetedPanel": ("_targeted", "sum"),
    }
    if group_column == "cancerGroup":
        aggregation["nStudies"] = ("studyId", "nunique")
    meta = work.groupby(group_column, as_index=False).agg(**aggregation)
    if group_column == "studyId":
        meta["nStudies"] = 1

    hallmarks = pd.DataFrame({"hallmark": sorted(membership.hallmark.unique())})
    grid = meta.merge(hallmarks, how="cross")
    n_genes = membership.groupby("hallmark").entrezGeneId.nunique().rename("nHallmarkGenes")
    grid = grid.merge(n_genes, on="hallmark", how="left", validate="many_to_one")

    documented_columns = list(
        dict.fromkeys(["studyId", "sampleId", "sampleKey", group_column])
    )
    documented = work.loc[
        work.assayStratum.eq("Documented WES/WGS"), documented_columns
    ]
    documented_events = events.merge(
        documented,
        on=["studyId", "sampleId", "sampleKey"],
        how="inner",
        validate="many_to_one",
    )
    mutated = (
        documented_events.groupby([group_column, "hallmark"])["sampleKey"]
        .nunique()
        .rename("nMutatedDocumentedWesWgs")
        .reset_index()
    )
    grid = grid.merge(
        mutated,
        on=[group_column, "hallmark"],
        how="left",
        validate="one_to_one",
    )
    grid["nMutatedDocumentedWesWgs"] = (
        grid.nMutatedDocumentedWesWgs.fillna(0).astype(int)
    )
    if (grid.nMutatedDocumentedWesWgs > grid.nDocumentedWesWgs).any():
        raise AssertionError("A hallmark numerator exceeds its documented-WES/WGS denominator")
    grid["nDocumentedWesWgsWithoutRetainedHallmarkMutation"] = (
        grid.nDocumentedWesWgs - grid.nMutatedDocumentedWesWgs
    )
    grid["mutationPctDocumentedWesWgs"] = np.where(
        grid.nDocumentedWesWgs.gt(0),
        100 * grid.nMutatedDocumentedWesWgs / grid.nDocumentedWesWgs,
        np.nan,
    )
    low, high = _wilson_interval(
        grid.nMutatedDocumentedWesWgs, grid.nDocumentedWesWgs
    )
    grid["mutationCiLowPct"] = low
    grid["mutationCiHighPct"] = high
    grid["estimateStatus"] = np.where(
        grid.nDocumentedWesWgs.gt(0),
        "Estimated from documented WES/WGS",
        "Not assayed by documented WES/WGS",
    )
    grid["eventDefinition"] = (
        "At least one retained protein-altering mutation in a panel gene annotated to "
        "the hallmark; unique sample-hallmark union"
    )
    grid["denominatorDefinition"] = (
        "Selected one-per-patient tissue specimens with documented WES/WGS assignment"
    )
    grid["nonAdditivityWarning"] = (
        "Hallmarks overlap; do not sum specimens or percentages across hallmarks"
    )
    return grid


def _draw_heatmap(
    ax: plt.Axes, matrix: pd.DataFrame, *, vmax: float
) -> matplotlib.image.AxesImage:
    # YlOrRd is the non-blue sequential ColourBrewer scale.  Grey remains reserved
    # for an unassayed state and is not part of the quantitative colour progression.
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#E6E6E6")
    masked = np.ma.masked_invalid(matrix.to_numpy(float))
    image = ax.imshow(
        masked,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0,
        vmax=vmax,
    )
    ax.set_xticks(np.arange(matrix.shape[1]), matrix.columns, rotation=55, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]), matrix.index)
    ax.tick_params(length=0, labelsize=3.8)
    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.35)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iat[row, column]
            if np.isfinite(value):
                label = f"{value:.0f}"
                red, green, blue, _ = cmap(np.clip(value / vmax, 0, 1))
                luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                color = "white" if luminance < 0.48 else COLORS["black"]
            else:
                label = "—"
                color = COLORS["grey"]
            ax.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                fontsize=2.60,
                color=color,
            )
    return image


def make_figure(cancer: pd.DataFrame, membership: pd.DataFrame) -> None:
    """Create a readable numeric heatmap while keeping the full matrix in the CSV."""
    apply_style()
    supplementary = FIGURES / "supplementary"
    source_data = TABLES.parent / "source_data"
    supplementary.mkdir(parents=True, exist_ok=True)
    source_data.mkdir(parents=True, exist_ok=True)

    meta = (
        cancer.groupby("cancerGroup", as_index=False)
        .agg(
            nEligibleSamples=("nEligibleSamples", "max"),
            nDocumentedWesWgs=("nDocumentedWesWgs", "max"),
            nUnverifiedMutationProfile=("nUnverifiedMutationProfile", "max"),
            nTargetedPanel=("nTargetedPanel", "max"),
            nStudies=("nStudies", "max"),
        )
    )
    shown = (
        meta.loc[meta.nDocumentedWesWgs.gt(0)]
        .sort_values(["nDocumentedWesWgs", "nEligibleSamples"], ascending=False)
        .head(N_DISPLAY_CANCERS)
        .cancerGroup.tolist()
    )
    if len(shown) != N_DISPLAY_CANCERS:
        raise AssertionError(f"Only {len(shown)} cancer groups satisfy the display rule")

    pan = (
        cancer.groupby("hallmark", as_index=False)
        .agg(
            nMutatedDocumentedWesWgs=("nMutatedDocumentedWesWgs", "sum"),
            nDocumentedWesWgs=("nDocumentedWesWgs", "sum"),
            nHallmarkGenes=("nHallmarkGenes", "max"),
        )
    )
    pan["panCancerMutationPct"] = (
        100 * pan.nMutatedDocumentedWesWgs / pan.nDocumentedWesWgs
    )
    subset = cancer.loc[cancer.cancerGroup.isin(shown)].copy()
    unclustered = (
        subset.pivot(
            index="hallmark",
            columns="cancerGroup",
            values="mutationPctDocumentedWesWgs",
        )
        .reindex(columns=shown)
    )
    cancer_order = _cluster_profile_order(unclustered.T)
    hallmark_order = _cluster_profile_order(unclustered)
    matrix = unclustered.reindex(index=hallmark_order, columns=cancer_order)
    vmax = max(55.0, float(np.nanpercentile(matrix.to_numpy(float), 98)))

    fig = plt.figure(figsize=figsize(180, 105))
    grid = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[1.0, 0.17],
        height_ratios=[0.23, 1.0],
        hspace=0.028,
        wspace=0.020,
    )
    ax_top = fig.add_subplot(grid[0, 0])
    ax_heat = fig.add_subplot(grid[1, 0])
    ax_side = fig.add_subplot(grid[1, 1])

    ordered_meta = meta.set_index("cancerGroup").loc[cancer_order]
    x = np.arange(len(cancer_order))
    bars = ax_top.bar(
        x,
        ordered_meta.nDocumentedWesWgs,
        color=BREWER_PURPLE,
        width=0.76,
        label="Documented WES/WGS",
    )
    ax_top.set_xlim(-0.5, len(x) - 0.5)
    ax_top.set_xticks([])
    ax_top.set_ylabel("WES/WGS\ncases", labelpad=2)
    ax_top.spines[["top", "right", "bottom"]].set_visible(False)
    ax_studies = ax_top.twinx()
    study_points = ax_studies.scatter(
        x,
        ordered_meta.nStudies,
        s=7,
        facecolor=BREWER_GREEN,
        edgecolor="white",
        linewidth=0.25,
        zorder=4,
        label="Contributing studies",
    )
    ax_studies.set_ylabel("Studies", color=BREWER_GREEN, labelpad=2)
    ax_studies.tick_params(axis="y", colors=BREWER_GREEN, labelsize=4.2)
    ax_studies.spines["top"].set_visible(False)
    ax_studies.spines["right"].set_color(BREWER_GREEN)
    ax_top.legend(
        [bars, study_points],
        ["Documented WES/WGS cases", "Contributing studies"],
        **LEGEND_BOX,
        loc="upper right",
        fontsize=4.4,
        ncol=2,
        handlelength=1.1,
        columnspacing=0.8,
    )

    image = _draw_heatmap(ax_heat, matrix, vmax=vmax)
    ax_heat.set_xlabel("")
    ax_heat.set_ylabel("")
    # The compact scale occupies the far-left margin, separate from both the hallmark
    # labels and quantitative side bars.
    ax_cbar = fig.add_axes([0.074, 0.430, 0.011, 0.235])
    colorbar = fig.colorbar(image, cax=ax_cbar, orientation="vertical")
    colorbar.ax.set_title(
        "Hallmark-mutant\ntumours (%)",
        fontsize=4.8,
        pad=2.2,
    )
    colorbar.set_ticks([0, vmax / 2, vmax])
    colorbar.set_ticklabels(["0", f"{vmax / 2:.0f}", f"{vmax:.0f}"])
    colorbar.ax.yaxis.set_ticks_position("left")
    colorbar.ax.tick_params(labelsize=4.2, pad=1.2)

    side = pan.set_index("hallmark").reindex(hallmark_order)
    y = np.arange(len(hallmark_order))
    ax_side.barh(y, side.panCancerMutationPct, color=BREWER_ORANGE, height=0.68)
    ax_side.set_ylim(len(hallmark_order) - 0.5, -0.5)
    ax_side.set_yticks([])
    ax_side.set_xlabel("Pan-cancer\nmutation (%)")
    ax_side.grid(axis="x", color=COLORS["very_light_grey"], linewidth=0.5)
    right_limit = max(5.0, float(side.panCancerMutationPct.max()) * 1.23)
    ax_side.set_xlim(0, right_limit)
    for yy, value in zip(y, side.panCancerMutationPct):
        ax_side.text(value, yy, f" {value:.0f}", va="center", fontsize=4.3)

    cancer_positions = {label: position for position, label in enumerate(cancer_order)}
    hallmark_positions = {label: position for position, label in enumerate(hallmark_order)}
    subset["_hallmarkOrder"] = subset.hallmark.map(hallmark_positions)
    subset["_cancerOrder"] = subset.cancerGroup.map(cancer_positions)
    subset = subset.sort_values(["_hallmarkOrder", "_cancerOrder"]).drop(
        columns=["_hallmarkOrder", "_cancerOrder"]
    )
    subset.to_csv(source_data / "figureS5_hallmark_heatmap.csv", index=False)
    annotations = (
        membership.groupby("hallmark", as_index=False)
        .agg(
            nHallmarkGenes=("entrezGeneId", "nunique"),
            nGeneHallmarkPairs=("entrezGeneId", "size"),
        )
        .merge(
            pan[
                [
                    "hallmark",
                    "nMutatedDocumentedWesWgs",
                    "nDocumentedWesWgs",
                    "panCancerMutationPct",
                ]
            ],
            on="hallmark",
            validate="one_to_one",
        )
    )
    annotations["_displayOrder"] = annotations.hallmark.map(hallmark_positions)
    annotations = annotations.sort_values("_displayOrder").drop(columns="_displayOrder")
    annotations["eventCountingUnit"] = "unique documented-WES/WGS sample--hallmark pair"
    annotations.to_csv(source_data / "figureS5_hallmark_annotations.csv", index=False)
    ordered_meta.reset_index().to_csv(
        source_data / "figureS5_hallmark_denominators.csv", index=False
    )

    fig.subplots_adjust(left=0.245, right=0.985, top=0.955, bottom=0.270)
    save_figure(fig, supplementary / "figureS5_hallmark_landscape")
    plt.close(fig)


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    membership = load_membership()
    samples, events = load_sample_events(membership)

    cancer = aggregate_complete(
        samples, events, membership, group_column="cancerGroup"
    ).sort_values(["nEligibleSamples", "cancerGroup", "hallmark"], ascending=[False, True, True])
    study = aggregate_complete(
        samples, events, membership, group_column="studyId"
    ).sort_values(["nEligibleSamples", "studyId", "hallmark"], ascending=[False, True, True])

    if len(cancer) != EXPECTED_CANCER_GROUPS * EXPECTED_HALLMARKS:
        raise AssertionError("Cancer-by-hallmark matrix is incomplete")
    if cancer.cancerGroup.nunique() != EXPECTED_CANCER_GROUPS:
        raise AssertionError("Not all 89 reviewed cancer families were retained")
    if len(study) != EXPECTED_STUDIES * EXPECTED_HALLMARKS:
        raise AssertionError("Study-by-hallmark matrix is incomplete")
    if study.studyId.nunique() != EXPECTED_STUDIES:
        raise AssertionError("Not all 367 contributing studies were retained")

    TABLES.mkdir(parents=True, exist_ok=True)
    cancer.to_csv(TABLES / "hallmark_mutation_by_cancer_complete.csv", index=False)
    study.to_csv(TABLES / "hallmark_mutation_by_study_complete.csv", index=False)
    make_figure(cancer, membership)

    n_sample_hallmark_events = int(
        cancer.nMutatedDocumentedWesWgs.sum()
    )
    print(
        f"Hallmark membership: {membership.entrezGeneId.nunique():,} genes, "
        f"{len(membership):,} gene--hallmark pairs, "
        f"{membership.hallmark.nunique():,} hallmarks."
    )
    print(
        f"Complete outputs: {cancer.cancerGroup.nunique():,} cancer groups x "
        f"{membership.hallmark.nunique():,} hallmarks and "
        f"{study.studyId.nunique():,} studies x {membership.hallmark.nunique():,} hallmarks."
    )
    print(
        f"Primary denominator: {EXPECTED_DOCUMENTED_WES_WGS:,} documented WES/WGS "
        f"specimens; {n_sample_hallmark_events:,} unique sample--hallmark events "
        "after within-hallmark union."
    )
    print("Wrote Supplementary Figure S5 and complete audit tables.")


if __name__ == "__main__":
    main()
