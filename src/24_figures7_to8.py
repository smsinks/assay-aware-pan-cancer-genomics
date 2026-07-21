"""Stage 24 -- integrated functional evidence and descriptive association networks.

This stage builds submission-scaled, assay-aware functional and association-network
figures.

Figure 7
    Five complementary views of recurrent hotspot genotypes: a number-annotated
    cross-layer evidence matrix; an adjusted PRISM forest containing the evaluable
    pre-specified controls and separately identified additional contexts; model-level
    CRISPR gene-effect distributions within the informative lineages used for
    adjustment; a volcano view of the complete lineage-adjusted PRISM scan; and a
    compact CRISPR--PRISM synthesis. Cell fill in the matrix is column-scaled for
    readability; printed values are absolute.

Figure 8
    A compact network restricted to literature-guided, cancer-specific contexts that
    pass the conditioned association and cross-assay concordance criteria, paired with
    an annotation-group-by-cancer matrix and the composition of the displayed
    network. Direct links in the supplied interactome are outlined, with raw directed
    relationship codes retained in a source-data crosswalk rather than interpreted as
    uniformly physical binding. Duplicate gene pairs are collapsed only in the
    drawing; every cancer-specific estimate is retained in source data.

Supplementary Figure S3
    A broader, reproducibly thresholded network, the complete selection cascade and
    WES/WGS-versus-targeted-panel stability.

Supplementary Figure S6
    Diagnostic comparisons of unadjusted and lineage-adjusted CRISPR and PRISM
    estimates.

Coverage contract
-----------------
The plotted networks are deliberate legibility subsets.  Source data include all
screened cancer--gene-pair rows, the complete panel-gene universe and every eligible
cancer group and contributing study in the regenerated cohort.  Thus a cancer group or
study absent from a network drawing is never silently treated as a zero-event cohort.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import math
import re
import warnings
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyBboxPatch, Patch, Rectangle
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import FIGURES, PROCESSED, RESULTS, ROOT, TABLES
from nature_style import (
    COLORS,
    apply as apply_style,
    figure_panel_label,
    figsize,
    panel_label,
    save_figure,
)


warnings.filterwarnings("ignore")

SOURCE = RESULTS / "source_data"
SUPP = FIGURES / "supplementary"
COMPLETE = RESULTS / "submission_tables"
INTERACTOME = ROOT / "Human_Interactome.xlsx"
DM = ROOT / "data" / "external" / "depmap"
_INTERACTOME_GRAPH: nx.Graph | None = None

ROLE_COLORS = {
    "Oncogenes": COLORS["vermillion"],
    "TSGs": COLORS["blue"],
    "Oncogene/TSG": COLORS["purple"],
    "Other": COLORS["grey"],
}

MODULE_ORDER = [
    "RTK–RAS/lung",
    "PI3K",
    "WNT/GI",
    "IDH/chromatin",
    "Myeloid",
    "TP53/lineage",
]
MODULE_COLORS = {
    "RTK–RAS/lung": COLORS["vermillion"],
    "PI3K": COLORS["orange"],
    "WNT/GI": COLORS["green"],
    "IDH/chromatin": COLORS["purple"],
    "Myeloid": COLORS["sky"],
    "TP53/lineage": COLORS["blue"],
}

LEGEND_BOX = {
    "frameon": True,
    "fancybox": True,
    "framealpha": 0.96,
    "facecolor": "#FAFAFA",
    "edgecolor": COLORS["light_grey"],
    "borderpad": 0.45,
}

MODULE_GENES = {
    "RTK–RAS/lung": {
        "EGFR", "KRAS", "BRAF", "NRAS", "HRAS", "FGFR3", "KEAP1", "STK11",
        "NF1", "ERBB2", "ERBB3", "MET", "ALK", "ROS1", "RET",
    },
    "PI3K": {"PIK3CA", "PIK3R1", "PIK3R2", "PTEN", "AKT1", "AKT2", "MTOR"},
    "WNT/GI": {"APC", "RNF43", "CTNNB1", "AXIN1", "AXIN2", "AMER1"},
    "IDH/chromatin": {
        "IDH1", "IDH2", "ATRX", "BAP1", "PBRM1", "ARID1A", "ARID1B", "SMARCA4",
    },
    "Myeloid": {
        "NPM1", "RUNX1", "DNMT3A", "FLT3", "ASXL1", "SF3B1", "SRSF2", "JAK2",
        "CALR", "TET2",
    },
    "TP53/lineage": {
        "TP53", "RB1", "CDKN2A", "ATM", "ESR1", "SPOP", "FOXA1", "SMAD4",
    },
}


def _pair(a: str, b: str) -> str:
    return "-".join(sorted((a, b)))


_XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _xlsx_cell_value(cell: ET.Element, shared_strings: list[str]) -> str | None:
    """Decode the small subset of OOXML cell types used by the interactome file."""
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.iter(f"{_XLSX_NS}t"))
    value = cell.find(f"{_XLSX_NS}v")
    if value is None or value.text is None:
        return None
    if cell_type == "s":
        return shared_strings[int(value.text)]
    return value.text


def _interactome_graph() -> nx.Graph:
    """Load the complete undirected projection used for distance/null analyses.

    The source workbook retains directed relationship codes, but graph distance is
    evaluated on an undirected projection.  Relationship types and orientations are
    still recovered separately for the displayed direct-edge crosswalk.
    """
    global _INTERACTOME_GRAPH
    if _INTERACTOME_GRAPH is not None:
        return _INTERACTOME_GRAPH
    if not INTERACTOME.exists():
        raise FileNotFoundError(f"Supplied interactome workbook not found: {INTERACTOME}")

    graph = nx.Graph()
    with ZipFile(INTERACTOME) as archive:
        shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        shared_strings = [
            "".join(node.text or "" for node in item.iter(f"{_XLSX_NS}t"))
            for item in shared_root.findall(f"{_XLSX_NS}si")
        ]
        with archive.open("xl/worksheets/sheet1.xml") as worksheet:
            for _, row in ET.iterparse(worksheet, events=("end",)):
                if row.tag != f"{_XLSX_NS}row":
                    continue
                values: dict[str, str | None] = {}
                for cell in row.findall(f"{_XLSX_NS}c"):
                    column_match = re.match(r"([A-Z]+)", cell.attrib.get("r", ""))
                    if column_match and column_match.group(1) in {"A", "C"}:
                        values[column_match.group(1)] = _xlsx_cell_value(cell, shared_strings)
                protein1 = values.get("A")
                protein2 = values.get("C")
                if protein1 and protein2 and protein1 != "Protein1" and protein1 != protein2:
                    graph.add_edge(protein1, protein2)
                row.clear()
    _INTERACTOME_GRAPH = graph
    return graph


def _interactome_support(edges: pd.DataFrame) -> pd.DataFrame:
    """Cross-reference displayed statistical edges against the supplied interactome.

    The workbook includes directed regulatory and post-translational relationship
    codes, not a pure physical-binding catalogue.  We therefore report a neutral
    *interactome connection*, retain every raw code and orientation for direct links,
    and keep two-step shared-neighbour support separate from direct evidence.
    """
    requested_pairs = set(edges.pair)
    direct_records: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    graph = _interactome_graph()

    if not INTERACTOME.exists():
        raise FileNotFoundError(f"Supplied interactome workbook not found: {INTERACTOME}")

    with ZipFile(INTERACTOME) as archive:
        shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        shared_strings = [
            "".join(node.text or "" for node in item.iter(f"{_XLSX_NS}t"))
            for item in shared_root.findall(f"{_XLSX_NS}si")
        ]
        with archive.open("xl/worksheets/sheet1.xml") as worksheet:
            for _, row in ET.iterparse(worksheet, events=("end",)):
                if row.tag != f"{_XLSX_NS}row":
                    continue
                values: dict[str, str | None] = {}
                for cell in row.findall(f"{_XLSX_NS}c"):
                    column_match = re.match(r"([A-Z]+)", cell.attrib.get("r", ""))
                    if column_match:
                        values[column_match.group(1)] = _xlsx_cell_value(cell, shared_strings)
                protein1 = values.get("A")
                relationship = values.get("B")
                protein2 = values.get("C")
                if protein1 and protein2 and relationship and protein1 != "Protein1":
                    pair = _pair(protein1, protein2)
                    if pair in requested_pairs:
                        direct_records[pair].add((protein1, relationship, protein2))
                row.clear()

    support_rows: list[dict[str, object]] = []
    for record in edges.itertuples(index=False):
        pair = record.pair
        direct = sorted(direct_records.get(pair, set()))
        if record.geneA in graph and record.geneB in graph:
            shared = sorted(nx.common_neighbors(graph, record.geneA, record.geneB))
        else:
            shared = []
        path_length = 1 if direct else (2 if shared else np.nan)
        support_rows.append(
            {
                "pair": pair,
                "interactomeDirect": bool(direct),
                "nInteractomeDirectRecords": len(direct),
                "interactomeInteractionTypes": ";".join(sorted({item[1] for item in direct})),
                "interactomeDirectedRecords": ";".join(
                    f"{protein1} {relationship} {protein2}"
                    for protein1, relationship, protein2 in direct
                ),
                "interactomePathLengthAtMost2": path_length,
                "nSharedInteractomeNeighbours": len(shared),
                "sharedInteractomeNeighbours": ";".join(shared[:20]),
                "sharedNeighbourListTruncated": len(shared) > 20,
            }
        )
    return pd.DataFrame(support_rows)


def _degree_matched_interactome_null(
    edge_sets: dict[str, pd.DataFrame],
    panel: pd.DataFrame,
    *,
    n_permutations: int = 5_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare displayed connectivity with a degree- and CGC-matched null.

    Each endpoint is sampled independently from genes in the analysis compendium with
    the same COSMIC Cancer Gene Census membership state and log2-degree bin.  Sparse
    bins are expanded to the nearest degree-matched genes within the same CGC state.
    This makes direct and two-step connectivity an evaluated network annotation rather
    than assuming that dense-interactome proximity is independent validation.
    """
    graph = _interactome_graph()
    status = (
        panel[["hugoSymbol", "cosmicTier"]]
        .drop_duplicates("hugoSymbol")
        .assign(cgcMember=lambda frame: frame.cosmicTier.notna())
        .set_index("hugoSymbol")["cgcMember"]
    )
    universe = sorted(set(graph.nodes).intersection(status.index))
    degree = {gene: int(graph.degree(gene)) for gene in universe}
    degree_bin = {gene: int(np.floor(np.log2(degree[gene] + 1))) for gene in universe}
    by_status_bin: dict[tuple[bool, int], list[str]] = defaultdict(list)
    by_status: dict[bool, list[str]] = defaultdict(list)
    for gene in universe:
        state = bool(status.loc[gene])
        by_status_bin[(state, degree_bin[gene])].append(gene)
        by_status[state].append(gene)

    def candidates(gene: str) -> list[str]:
        if gene not in degree or gene not in status.index:
            return []
        state = bool(status.loc[gene])
        exact = by_status_bin.get((state, degree_bin[gene]), [])
        if len(exact) >= 8:
            return exact
        target = np.log2(degree[gene] + 1)
        ranked = sorted(
            by_status[state],
            key=lambda item: (abs(np.log2(degree[item] + 1) - target), item),
        )
        return ranked[: max(8, min(80, len(ranked)))]

    rng = np.random.default_rng(42)
    summary_rows: list[dict[str, object]] = []
    distribution_rows: list[dict[str, object]] = []
    for network_name, edges in edge_sets.items():
        unique = edges[["pair", "geneA", "geneB"]].drop_duplicates("pair").copy()
        unique = unique[
            unique.geneA.isin(universe) & unique.geneB.isin(universe)
        ].reset_index(drop=True)
        if unique.empty:
            raise AssertionError(f"No {network_name} edges are represented in the interactome null universe")
        endpoint_candidates = [
            (candidates(record.geneA), candidates(record.geneB))
            for record in unique.itertuples(index=False)
        ]
        if any(not left or not right for left, right in endpoint_candidates):
            raise AssertionError(f"Unable to degree-match every {network_name} edge")

        observed_direct = int(
            sum(graph.has_edge(record.geneA, record.geneB) for record in unique.itertuples(index=False))
        )
        observed_within_two = int(
            sum(
                graph.has_edge(record.geneA, record.geneB)
                or bool(set(graph[record.geneA]).intersection(graph[record.geneB]))
                for record in unique.itertuples(index=False)
            )
        )

        null_direct = np.zeros(n_permutations, dtype=int)
        null_within_two = np.zeros(n_permutations, dtype=int)
        for permutation in range(n_permutations):
            for left_candidates, right_candidates in endpoint_candidates:
                left = str(rng.choice(left_candidates))
                right = str(rng.choice(right_candidates))
                attempts = 0
                while right == left and attempts < 20:
                    right = str(rng.choice(right_candidates))
                    attempts += 1
                if right == left:
                    continue
                direct = graph.has_edge(left, right)
                within_two = direct or bool(set(graph[left]).intersection(graph[right]))
                null_direct[permutation] += int(direct)
                null_within_two[permutation] += int(within_two)
            distribution_rows.append(
                {
                    "network": network_name,
                    "permutation": permutation + 1,
                    "nDirect": int(null_direct[permutation]),
                    "nWithinTwoSteps": int(null_within_two[permutation]),
                }
            )

        for metric, observed, null_values in (
            ("direct connection", observed_direct, null_direct),
            ("connection within two steps", observed_within_two, null_within_two),
        ):
            summary_rows.append(
                {
                    "network": network_name,
                    "metric": metric,
                    "nObservedEdges": len(unique),
                    "observedConnectedEdges": observed,
                    "observedProportion": observed / len(unique),
                    "nullMeanConnectedEdges": float(null_values.mean()),
                    "nullMeanProportion": float(null_values.mean() / len(unique)),
                    "nullCiLowConnectedEdges": float(np.quantile(null_values, 0.025)),
                    "nullCiHighConnectedEdges": float(np.quantile(null_values, 0.975)),
                    "empiricalEnrichmentP": float(
                        (1 + np.count_nonzero(null_values >= observed))
                        / (n_permutations + 1)
                    ),
                    "nPermutations": n_permutations,
                    "matchingVariables": "log2 interactome degree bin and CGC membership",
                    "randomSeed": 42,
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(distribution_rows)


# These are literature-anchored or lineage-defining contexts already evaluated in the
# Stage 17 screen.  The main figure includes a row only when the primary result has
# FDR < 0.05, is cross-assay concordant and preserves both direction and broadly
# comparable magnitude under pair-specific leave-two-out background-burden
# conditioning.  The constant is therefore a display specification, not a list of
# assumed-positive results.
MAIN_CONTEXTS = [
    # Lung-adenocarcinoma genotype structure. Residual, histologically unspecified
    # NSCLC is analysed separately from LUAD.
    ("LUAD", _pair("EGFR", "KRAS")),
    ("LUAD", _pair("KEAP1", "STK11")),
    ("LUAD", _pair("KRAS", "TP53")),
    ("LUAD", _pair("EGFR", "STK11")),
    ("LUAD", _pair("STK11", "TP53")),
    ("LUAD", _pair("EGFR", "KEAP1")),
    ("LUAD", _pair("ATM", "TP53")),
    # Colorectal alternative driver routes.
    ("COADREAD", _pair("BRAF", "KRAS")),
    ("COADREAD", _pair("APC", "RNF43")),
    ("COADREAD", _pair("APC", "BRAF")),
    ("COADREAD", _pair("KRAS", "TP53")),
    ("COADREAD", _pair("PIK3CA", "TP53")),
    ("COADREAD", _pair("BRAF", "RNF43")),
    ("COADREAD", _pair("KRAS", "NRAS")),
    ("COADREAD", _pair("APC", "KRAS")),
    # Endometrial PI3K/WNT/TP53 structure.
    ("UCEC", _pair("PIK3CA", "PIK3R1")),
    ("UCEC", _pair("AKT1", "PTEN")),
    ("UCEC", _pair("PTEN", "TP53")),
    ("UCEC", _pair("CTNNB1", "TP53")),
    # Melanoma and breast.
    ("SKCM", _pair("BRAF", "NRAS")),
    ("BRCA", _pair("AKT1", "PIK3CA")),
    ("BRCA", _pair("PIK3CA", "TP53")),
    ("BRCA", _pair("ESR1", "TP53")),
    # Bladder.
    ("BLCA", _pair("FGFR3", "TP53")),
    ("BLCA", _pair("FGFR3", "RB1")),
    ("BLCA", _pair("RB1", "TP53")),
    # Portal-defined/legacy GBM associations involving IDH1 or ATRX remain in the
    # complete screen but are excluded from the literature-guided display because
    # historical GBM releases contain IDH-mutant tumours that would now be classified
    # separately.  They are reported in a dedicated classification-sensitivity audit.
    # Pancreatic and myeloid lineages.
    ("PAAD", _pair("KRAS", "TP53")),
    ("PAAD", _pair("CDKN2A", "TP53")),
    ("AML", _pair("NPM1", "RUNX1")),
    ("AML", _pair("DNMT3A", "NPM1")),
    ("AML", _pair("FLT3", "NPM1")),
    ("AML", _pair("ASXL1", "NPM1")),
    ("AML", _pair("DNMT3A", "SRSF2")),
    # Additional lineage-defining contexts.
    ("PRAD", _pair("SPOP", "TP53")),
    ("PRAD", _pair("APC", "SPOP")),
    ("HCC", _pair("CTNNB1", "TP53")),
    ("CHOL", _pair("BAP1", "TP53")),
    ("CCRCC", _pair("BAP1", "PBRM1")),
    ("ESCA", _pair("ARID1A", "TP53")),
    ("HNSC", _pair("CDKN2A", "TP53")),
]


PRISM_POSITIVE_CONTROLS = [
    ("BRAF", "VEMURAFENIB"),
    ("BRAF", "ENCORAFENIB"),
    ("BRAF", "DABRAFENIB"),
    ("PIK3CA", "ALPELISIB"),
    ("PIK3CA", "TASELISIB"),
    ("PIK3CA", "COPANLISIB"),
    ("KRAS", "TRAMETINIB"),
    ("KRAS", "SELUMETINIB"),
    ("KRAS", "BINIMETINIB"),
    ("NRAS", "TRAMETINIB"),
    ("NRAS", "SELUMETINIB"),
    ("NRAS", "BINIMETINIB"),
]

# Canonical same-gene dependencies selected for model-level distribution display.
# All six pass the lineage-adjusted CRISPR FDR threshold and span RTK-RAS, PI3K and
# WNT signalling. The underlying model rows are exactly those entering Stage 18.
CRISPR_DISTRIBUTION_GENES = ["KRAS", "NRAS", "BRAF", "PIK3CA", "CTNNB1", "PTEN"]

# These five contexts supplement, but are never relabelled as, the pre-specified
# pathway controls. They were chosen because each has a positive, globally FDR-
# significant PRISM association and adds a genotype not represented in the control
# set. Their selection class is retained explicitly in panel-level source data.
PRISM_ADDITIONAL_CONTEXTS = [
    ("PIK3R1", "MLN0128"),
    ("STK11", "EVEROLIMUS"),
    ("RAC1", "RKI-1447"),
    ("FBXW7", "BERZOSERTIB"),
    ("TP53", "(+)-CAMPTOTHECIN"),
]

# A deliberately sparse set of volcano labels. All selected points remain outlined;
# the label subset avoids turning the full-screen overview into a text cloud.
VOLCANO_LABEL_CONTEXTS = {
    ("BRAF", "DABRAFENIB"),
    ("PIK3CA", "COPANLISIB"),
    ("KRAS", "TRAMETINIB"),
    ("NRAS", "TRAMETINIB"),
    ("PIK3R1", "MLN0128"),
    ("STK11", "EVEROLIMUS"),
    ("RAC1", "RKI-1447"),
    ("FBXW7", "BERZOSERTIB"),
    ("TP53", "(+)-CAMPTOTHECIN"),
}


def setup_dirs() -> None:
    SOURCE.mkdir(parents=True, exist_ok=True)
    SUPP.mkdir(parents=True, exist_ok=True)


def _safe_log10_q(values: pd.Series) -> pd.Series:
    return -np.log10(pd.to_numeric(values, errors="coerce").clip(lower=1e-300))


def _rescale(values: pd.Series, low: float | None = None, high: float | None = None) -> np.ndarray:
    array = pd.to_numeric(values, errors="coerce").to_numpy(float)
    finite = array[np.isfinite(array)]
    if not len(finite):
        return np.zeros(len(array))
    lower = float(np.nanmin(finite) if low is None else low)
    upper = float(np.nanmax(finite) if high is None else high)
    if upper <= lower:
        return np.full(len(array), 0.5)
    return np.clip((array - lower) / (upper - lower), 0, 1)


def _blend(color: str, amount: float, minimum: float = 0.08) -> tuple[float, float, float]:
    """Blend a colour into white; amount is in [0, 1]."""
    rgb = np.asarray(to_rgb(color))
    weight = minimum + (1 - minimum) * float(np.clip(amount, 0, 1))
    return tuple(1 - weight * (1 - rgb))


def _control_table(drug: pd.DataFrame) -> pd.DataFrame:
    order = {key: index for index, key in enumerate(PRISM_POSITIVE_CONTROLS)}
    result = drug.copy()
    result["controlKey"] = list(zip(result.gene, result.compound.astype(str).str.upper()))
    result = result[result.controlKey.isin(order)].copy()
    result["controlOrder"] = result.controlKey.map(order)
    result = (
        result.sort_values(["controlOrder", "adjustedStandardError"])
        .drop_duplicates("controlKey")
        .sort_values("controlOrder")
    )
    result["pairLabel"] = result.gene + " — " + result.compound.str.title()
    return result.drop(columns="controlKey")


def _functional_evidence_table() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prevalence = pd.read_csv(TABLES / "gene_frequencies_curated.csv")
    evidence = pd.read_csv(TABLES / "cmc_evidence_curated.csv")
    addiction = pd.read_csv(TABLES / "depmap_addiction_lineage_adjusted.csv")
    drug = pd.read_csv(TABLES / "depmap_drug_lineage_adjusted.csv")

    positive = drug[(drug.adjustedFdr < 0.05) & (drug.adjustedSensitisation > 0)].copy()
    drug_summary = (
        positive.groupby("gene")
        .agg(
            nSensitisingCompounds=("compoundId", "nunique"),
            bestPrismSensitisation=("adjustedSensitisation", "max"),
            bestPrismFdr=("adjustedFdr", "min"),
        )
        .reset_index()
    )
    if len(positive):
        best_index = positive.groupby("gene").adjustedSensitisation.idxmax()
        best_names = positive.loc[best_index, ["gene", "compound"]].rename(
            columns={"compound": "bestPrismCompound"}
        )
        drug_summary = drug_summary.merge(best_names, on="gene", how="left")

    matrix = (
        prevalence[
            [
                "gene", "entrezGeneId", "freqPct", "pctCuratedSamplesProfilingGene",
                "roleInCancer", "cosmicTier", "highConfidence",
            ]
        ]
        .merge(evidence[["gene", "pctTierMatched", "nTierMatched", "nMutationRecords"]], on="gene", how="left")
        .merge(addiction, on=["gene", "entrezGeneId", "roleInCancer"], how="inner", suffixes=("", "Crispr"))
        .merge(drug_summary, on="gene", how="left")
    )
    matrix["nSensitisingCompounds"] = matrix.nSensitisingCompounds.fillna(0).astype(int)
    matrix["bestPrismSensitisation"] = matrix.bestPrismSensitisation.fillna(0)
    matrix["bestPrismFdr"] = matrix.bestPrismFdr.fillna(1)
    matrix["bestPrismCompound"] = matrix.bestPrismCompound.fillna("—")
    matrix["crisprNegLog10Fdr"] = _safe_log10_q(matrix.adjustedFdr)
    matrix["mutantSelectiveDependency"] = -matrix.adjustedEffect

    # All adjusted CRISPR discoveries plus two additional clinically familiar genes
    # with drug-screen support provide a compact, deterministic 18-row matrix.
    supported = matrix[matrix.adjustedFdr < 0.05].sort_values(["adjustedFdr", "adjustedEffect"])
    selected_genes = supported.gene.tolist()
    for gene in ("EGFR", "STK11"):
        if gene in set(matrix.gene) and gene not in selected_genes:
            selected_genes.append(gene)
    selected_genes = selected_genes[:18]
    selected = matrix.set_index("gene").loc[selected_genes].reset_index()
    selected["displayOrder"] = np.arange(1, len(selected) + 1)
    selected["selectionReason"] = np.where(
        selected.adjustedFdr < 0.05,
        "lineage-adjusted CRISPR FDR < 0.05",
        "additional clinically familiar gene with PRISM support",
    )
    controls = _control_table(drug)
    return selected, controls, addiction, drug


def _depmap_gene_id(column: str) -> int | None:
    """Extract the terminal Entrez identifier from a DepMap matrix column."""
    match = re.search(r"\((\d+)\)$", str(column))
    return int(match.group(1)) if match else None


def _crispr_distribution_table(addiction: pd.DataFrame) -> pd.DataFrame:
    """Recover the exact informative-lineage model rows for selected CRISPR contrasts.

    Stage 18 estimates the adjusted coefficient only in lineages containing both a
    hotspot-mutant and hotspot-negative model. This helper reconstructs that analysis
    population for the six displayed genes and validates its counts against the saved
    model summaries before exposing any individual model-level distribution.
    """
    selected = addiction.loc[addiction.gene.isin(CRISPR_DISTRIBUTION_GENES)].copy()
    if set(selected.gene) != set(CRISPR_DISTRIBUTION_GENES):
        missing = sorted(set(CRISPR_DISTRIBUTION_GENES) - set(selected.gene))
        raise RuntimeError(f"CRISPR distribution genes absent from adjusted results: {missing}")

    crispr_path = DM / "CRISPRGeneEffect.csv"
    hotspot_path = DM / "OmicsSomaticMutationsMatrixHotspot.csv"
    crispr_header = pd.read_csv(crispr_path, nrows=0).columns
    hotspot_header = pd.read_csv(hotspot_path, nrows=0).columns
    crispr_columns = {
        entrez: column
        for column in crispr_header
        if (entrez := _depmap_gene_id(column)) is not None
    }
    hotspot_columns = {
        entrez: column
        for column in hotspot_header
        if (entrez := _depmap_gene_id(column)) is not None
    }
    selected_entrez = selected.set_index("gene").entrezGeneId.astype(int).to_dict()
    required_entrez = set(selected_entrez.values())
    if not required_entrez.issubset(crispr_columns) or not required_entrez.issubset(hotspot_columns):
        raise RuntimeError("A selected CRISPR distribution gene is absent from a DepMap matrix")

    crispr_id_column = "ModelID" if "ModelID" in crispr_header else str(crispr_header[0])
    crispr_usecols = [crispr_id_column] + [crispr_columns[value] for value in required_entrez]
    crispr = pd.read_csv(crispr_path, usecols=crispr_usecols).set_index(crispr_id_column)
    crispr.index.name = "ModelID"
    hotspot_usecols = ["ModelID", "IsDefaultEntryForModel"] + [
        hotspot_columns[value] for value in required_entrez
    ]
    hotspot = pd.read_csv(hotspot_path, usecols=hotspot_usecols)
    hotspot = hotspot.loc[hotspot.IsDefaultEntryForModel.eq("Yes")].drop(
        columns="IsDefaultEntryForModel"
    )
    hotspot = hotspot.groupby("ModelID", sort=False).max(numeric_only=True)
    model = pd.read_csv(DM / "Model.csv", usecols=["ModelID", "OncotreeLineage"]).set_index("ModelID")

    roles = selected.set_index("gene").roleInCancer.to_dict()
    summaries = selected.set_index("gene")
    rows: list[dict[str, object]] = []
    for order, gene in enumerate(CRISPR_DISTRIBUTION_GENES, start=1):
        entrez = selected_entrez[gene]
        common = crispr.index.intersection(hotspot.index).intersection(model.index)
        outcome = pd.to_numeric(crispr.loc[common, crispr_columns[entrez]], errors="coerce").to_numpy(float)
        hotspot_value = pd.to_numeric(
            hotspot.loc[common, hotspot_columns[entrez]], errors="coerce"
        ).to_numpy(float)
        mutation = np.where(np.isfinite(hotspot_value), hotspot_value > 0, np.nan)
        lineage = model.loc[common, "OncotreeLineage"].to_numpy(object)

        valid = np.isfinite(outcome) & np.isfinite(mutation) & pd.notna(lineage)
        outcome = outcome[valid]
        mutation = mutation[valid].astype(bool)
        lineage = lineage[valid].astype(str)
        lineage_frame = pd.DataFrame({"lineage": lineage, "mutant": mutation})
        counts = lineage_frame.groupby("lineage").mutant.agg(["sum", "count"])
        informative_lineages = set(counts.index[(counts["sum"] > 0) & (counts["sum"] < counts["count"])])
        informative = np.fromiter((value in informative_lineages for value in lineage), dtype=bool)
        outcome = outcome[informative]
        mutation = mutation[informative]
        lineage = lineage[informative]
        model_ids = common.to_numpy()[valid][informative]

        expected = summaries.loc[gene]
        assert int(mutation.sum()) == int(expected.nMutAdjusted)
        assert int((~mutation).sum()) == int(expected.nHotspotNegativeAdjusted)
        assert len(informative_lineages) == int(expected.nInformativeLineages)
        for model_id, group, lineage_value, gene_effect in zip(
            model_ids, mutation, lineage, outcome, strict=True
        ):
            rows.append(
                {
                    "gene": gene,
                    "entrezGeneId": entrez,
                    "roleInCancer": roles.get(gene, "Other"),
                    "displayOrder": order,
                    "modelId": model_id,
                    "oncotreeLineage": lineage_value,
                    "genotypeGroup": "Hotspot-mutant" if group else "Hotspot-negative",
                    "hotspotMutant": bool(group),
                    "crisprGeneEffect": float(gene_effect),
                    "adjustedEffect": float(expected.adjustedEffect),
                    "adjustedFdr": float(expected.adjustedFdr),
                    "comparatorDefinition": expected.comparatorDefinition,
                }
            )
    return pd.DataFrame(rows)


def _highlighted_prism_associations(
    drug: pd.DataFrame, controls: pd.DataFrame
) -> pd.DataFrame:
    """Combine pre-specified controls with five separately labelled contexts."""
    control = controls.copy()
    control = control.sort_values("controlOrder").reset_index(drop=True)
    control["selectionClass"] = "pre-specified pathway control"
    control["selectionBasis"] = "fixed before the PRISM scan"
    # Re-index the evaluable controls consecutively.  ``controlOrder`` retains
    # the original pre-specified position, which can contain gaps when a
    # planned control is not evaluable in the harmonised PRISM screen.
    control["selectionOrder"] = np.arange(1, len(control) + 1)

    additional_rows: list[pd.Series] = []
    for offset, (gene, compound) in enumerate(PRISM_ADDITIONAL_CONTEXTS, start=1):
        candidates = drug.loc[
            drug.gene.eq(gene) & drug.compound.astype(str).str.upper().eq(compound)
        ].sort_values("adjustedStandardError")
        if candidates.empty:
            raise RuntimeError(f"Additional PRISM context not evaluable: {gene}–{compound}")
        record = candidates.iloc[0].copy()
        if not (record.adjustedFdr < 0.05 and record.adjustedSensitisation > 0):
            raise RuntimeError(f"Additional PRISM context lacks sensitising FDR support: {gene}–{compound}")
        record["controlOrder"] = np.nan
        record["pairLabel"] = f"{gene} — {str(record.compound).title()}"
        record["selectionClass"] = "additional FDR-supported context"
        record["selectionBasis"] = (
            "positive adjusted sensitisation; global PRISM FDR < 0.05; "
            "genotype absent from pre-specified control set"
        )
        record["selectionOrder"] = len(control) + offset
        additional_rows.append(record)
    additional = pd.DataFrame(additional_rows)
    highlighted = pd.concat([control, additional], ignore_index=True, sort=False)
    highlighted["plotKey"] = (
        highlighted.gene.astype(str) + "||" + highlighted.compoundId.astype(str)
    )
    highlighted["labelOnVolcano"] = [
        (gene, str(compound).upper()) in VOLCANO_LABEL_CONTEXTS
        for gene, compound in zip(highlighted.gene, highlighted.compound, strict=True)
    ]
    highlighted["volcanoLabel"] = (
        highlighted.gene.astype(str) + "–" + highlighted.compound.astype(str).str.title()
    )
    assert highlighted.plotKey.is_unique
    assert int((highlighted.selectionClass == "additional FDR-supported context").sum()) == 5
    return highlighted.sort_values("selectionOrder").reset_index(drop=True)


def _compact_q(value: float) -> str:
    if value < 1e-3:
        exponent = int(math.floor(math.log10(value)))
        coefficient = value / (10**exponent)
        return f"{coefficient:.1f}e{exponent}"
    return f"{value:.3f}"


def figure7() -> None:
    apply_style()
    selected, controls, addiction, drug = _functional_evidence_table()
    distributions = _crispr_distribution_table(addiction)
    highlighted = _highlighted_prism_associations(drug, controls)

    volcano = drug.copy()
    volcano["volcanoNegLog10Fdr"] = _safe_log10_q(volcano.adjustedFdr)
    volcano["volcanoDisplayNegLog10Fdr"] = volcano.volcanoNegLog10Fdr.clip(upper=16.0)
    volcano["volcanoFdrDisplayCapped"] = volcano.volcanoNegLog10Fdr > 16.0
    volcano["volcanoCategory"] = np.select(
        [
            (volcano.adjustedFdr < 0.05) & (volcano.adjustedSensitisation > 0),
            (volcano.adjustedFdr < 0.05) & (volcano.adjustedSensitisation < 0),
        ],
        ["sensitising; FDR < 0.05", "lower response; FDR < 0.05"],
        default="FDR ≥ 0.05",
    )
    volcano["plotKey"] = volcano.gene.astype(str) + "||" + volcano.compoundId.astype(str)
    highlight_columns = [
        "plotKey", "selectionClass", "selectionBasis", "selectionOrder",
        "labelOnVolcano", "volcanoLabel",
    ]
    volcano = volcano.merge(highlighted[highlight_columns], on="plotKey", how="left")
    volcano["isHighlighted"] = volcano.selectionClass.notna()
    volcano["labelOnVolcano"] = volcano.labelOnVolcano.fillna(False).astype(bool)
    volcano["selectionClass"] = volcano.selectionClass.fillna("not highlighted")
    volcano["selectionBasis"] = volcano.selectionBasis.fillna("complete PRISM screen")

    synthesis = selected.copy()
    synthesis["crisprFdrSupported"] = synthesis.adjustedFdr < 0.05
    synthesis["prismFdrSupported"] = synthesis.nSensitisingCompounds > 0
    synthesis["pointArea"] = 13.0 + 2.8 * synthesis.nSensitisingCompounds
    synthesis_labels = {
        "KRAS", "NRAS", "BRAF", "PIK3CA", "CTNNB1", "PIK3R1", "CREBBP", "STK11"
    }
    synthesis["labelOnPlot"] = synthesis.gene.isin(synthesis_labels)

    compound_labels = {
        "VEMURAFENIB": "Vemurafenib",
        "ENCORAFENIB": "Encorafenib",
        "DABRAFENIB": "Dabrafenib",
        "ALPELISIB": "Alpelisib",
        "TASELISIB": "Taselisib",
        "COPANLISIB": "Copanlisib",
        "TRAMETINIB": "Trametinib",
        "BINIMETINIB": "Binimetinib",
        "MLN0128": "MLN0128",
        "EVEROLIMUS": "Everolimus",
        "RKI-1447": "RKI-1447",
        "BERZOSERTIB": "Berzosertib",
        "(+)-CAMPTOTHECIN": "(+)-camptothecin",
    }
    highlighted["displayLabel"] = [
        f"{gene} — {compound_labels.get(str(compound).upper(), str(compound).title())}"
        for gene, compound in zip(highlighted.gene, highlighted.compound, strict=True)
    ]

    columns = [
        ("freqPct", "Prev.\n(%)", "{:.1f}"),
        ("pctTierMatched", "CMC\nmatch (%)", "{:.0f}"),
        ("adjustedEffect", "CRISPR\nΔ effect", "{:+.2f}"),
        ("crisprNegLog10Fdr", "CRISPR\n−log10 q", "{:.1f}"),
        ("nMutAdjusted", "Mutant\nmodels", "{:.0f}"),
        ("nInformativeLineages", "Inform.\nlineages", "{:.0f}"),
        ("nSensitisingCompounds", "PRISM\nhits (n)", "{:.0f}"),
        ("bestPrismSensitisation", "Best\nΔAUC", "{:.2f}"),
    ]

    # Per-column scores control fill intensity only.  Absolute estimates remain printed.
    scores = {
        "freqPct": _rescale(selected.freqPct, 0, 20),
        "pctTierMatched": _rescale(selected.pctTierMatched.fillna(0), 0, 100),
        "adjustedEffect": _rescale(-selected.adjustedEffect, 0, 1.4),
        "crisprNegLog10Fdr": _rescale(selected.crisprNegLog10Fdr, 0, 20),
        "nMutAdjusted": _rescale(selected.nMutAdjusted, 0, 180),
        "nInformativeLineages": _rescale(selected.nInformativeLineages, 0, 25),
        "nSensitisingCompounds": _rescale(selected.nSensitisingCompounds, 0, 20),
        "bestPrismSensitisation": _rescale(selected.bestPrismSensitisation, 0, 0.18),
    }

    # Landscape composition: the evidence matrix and pharmacological forest share the
    # upper row; the three orthogonal functional summaries form a balanced lower row.
    fig = plt.figure(figsize=figsize(180, 138))
    gs = GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=[1.08, 1.08, 0.92],
        height_ratios=[1.24, 0.86],
        wspace=0.54,
        hspace=0.34,
    )
    ax_a = fig.add_subplot(gs[0, :2])
    ax_b = fig.add_subplot(gs[0, 2])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[1, 2])

    # a — tumour recurrence, mutation evidence, adjusted dependency and PRISM breadth
    # are aligned gene by gene. Fill intensity is column-specific; printed estimates
    # remain absolute and are not combined into a synthetic score.
    nrow, ncol = len(selected), len(columns)
    ax_a.set_xlim(-0.46, ncol)
    ax_a.set_ylim(nrow, 0)
    for row, record in selected.iterrows():
        role_color = ROLE_COLORS.get(record.roleInCancer, COLORS["grey"])
        ax_a.add_patch(
            Rectangle(
                (-0.24, row + 0.07), 0.14, 0.86,
                facecolor=role_color, edgecolor="white", lw=0.35,
            )
        )
        for col, (field, _, formatter) in enumerate(columns):
            value = record[field]
            intensity = scores[field][row]
            if field == "adjustedEffect" and np.isfinite(value) and value > 0:
                cell_color = _blend(COLORS["vermillion"], min(abs(value) / 0.4, 1))
                contrast = min(abs(value) / 0.4, 1)
            else:
                group_color = COLORS["blue"] if col < 2 else COLORS["green"]
                cell_color = _blend(group_color, intensity)
                contrast = intensity
            ax_a.add_patch(
                Rectangle(
                    (col + 0.02, row + 0.06), 0.96, 0.88,
                    facecolor=cell_color, edgecolor="white", lw=0.45,
                )
            )
            label = "—" if pd.isna(value) else formatter.format(value)
            ax_a.text(
                col + 0.50, row + 0.51, label,
                ha="center", va="center", fontsize=4.05,
                color="white" if contrast > 0.66 else COLORS["black"],
            )
    ax_a.set_xticks(np.arange(ncol) + 0.5, [label for _, label, _ in columns])
    ax_a.xaxis.tick_top()
    ax_a.tick_params(axis="x", length=0, pad=2, labelsize=4.15)
    ax_a.set_yticks(np.arange(nrow) + 0.5, selected.gene, fontsize=4.45)
    ax_a.tick_params(axis="y", length=0, pad=1.5)
    for spine in ax_a.spines.values():
        spine.set_visible(False)
    ax_a.text(-0.17, -0.43, "role", ha="center", va="bottom", fontsize=4.0, color=COLORS["grey"])
    ax_a.text(0.98, -1.52, "tumour evidence", ha="center", va="center", fontsize=4.65, fontweight="bold")
    ax_a.text(4.98, -1.52, "lineage-adjusted functional associations", ha="center", va="center", fontsize=4.65, fontweight="bold")
    ax_a.plot([0.03, 1.97], [-1.16, -1.16], color=COLORS["blue"], lw=1.0, clip_on=False)
    ax_a.plot([2.03, 7.97], [-1.16, -1.16], color=COLORS["green"], lw=1.0, clip_on=False)
    panel_label(ax_a, "a", x=-0.085, y=1.065)
    role_legend = [
        Patch(facecolor=color, edgecolor="none", label=label)
        for label, color in ROLE_COLORS.items()
    ]
    fill_legend = [
        Patch(facecolor=_blend(COLORS["blue"], 0.75), edgecolor="none", label="tumour/driver evidence"),
        Patch(facecolor=_blend(COLORS["green"], 0.75), edgecolor="none", label="functional association"),
        Patch(facecolor=_blend(COLORS["vermillion"], 0.75), edgecolor="none", label="mutant less dependent"),
    ]
    ax_a.legend(
        handles=role_legend + fill_legend,
        **LEGEND_BOX,
        ncol=7,
        fontsize=3.45,
        title="Encoding",
        title_fontsize=3.65,
        loc="upper center",
        bbox_to_anchor=(0.50, -0.035),
        handlelength=0.72,
        columnspacing=0.55,
        borderaxespad=0.2,
    )

    # b — adjusted PRISM forest. The ten evaluable, pre-specified controls are
    # retained and five separately selected FDR-supported contexts are appended.
    forest = highlighted.sort_values("selectionOrder").reset_index(drop=True)
    y = np.arange(len(forest))
    for index, record in forest.iterrows():
        is_additional = record.selectionClass == "additional FDR-supported context"
        color = COLORS["orange"] if is_additional else COLORS["blue"]
        marker = "D" if is_additional else "o"
        significant = bool(record.adjustedFdr < 0.05)
        ax_b.errorbar(
            record.adjustedSensitisation,
            index,
            xerr=np.array([[
                record.adjustedSensitisation - record.adjustedSensitisationCiLow
            ], [
                record.adjustedSensitisationCiHigh - record.adjustedSensitisation
            ]]),
            fmt=marker,
            ms=3.5,
            mfc=color if significant else "white",
            mec=color,
            mew=0.65,
            ecolor=color,
            elinewidth=0.75,
            capsize=1.5,
            zorder=3,
        )
        ax_b.text(
            0.284,
            index,
            _compact_q(float(record.adjustedFdr)),
            ha="left",
            va="center",
            fontsize=3.35,
            fontweight="bold" if significant else "normal",
            color=COLORS["black"] if significant else COLORS["grey"],
        )
    ax_b.axvline(0, color=COLORS["black"], lw=0.65, ls=(0, (2, 2)), zorder=0)
    ax_b.axhline(len(controls) - 0.5, color=COLORS["light_grey"], lw=0.55, zorder=0)
    ax_b.set_xlim(-0.015, 0.345)
    ax_b.set_ylim(len(forest) - 0.45, -0.85)
    ax_b.set_yticks(y, forest.displayLabel, fontsize=3.45)
    ax_b.tick_params(axis="y", length=0, pad=1.5)
    ax_b.set_xticks([0, 0.1, 0.2])
    ax_b.set_xlabel("Adjusted sensitisation, ΔAUC\n(95% CI)", labelpad=2.5)
    ax_b.grid(axis="x", color=COLORS["very_light_grey"], lw=0.45)
    ax_b.text(0.284, -0.61, "global q", ha="left", va="bottom", fontsize=3.45, fontweight="bold")
    ax_b.legend(
        handles=[
            Line2D([0], [0], marker="o", ls="", mfc=COLORS["blue"], mec=COLORS["blue"], label="Pre-specified control"),
            Line2D([0], [0], marker="D", ls="", mfc=COLORS["orange"], mec=COLORS["orange"], label="Additional FDR-supported"),
            Line2D([0], [0], marker="o", ls="", mfc="white", mec=COLORS["grey"], label="q ≥ 0.05"),
        ],
        **LEGEND_BOX,
        loc="lower center",
        bbox_to_anchor=(0.50, 1.025),
        ncol=3,
        fontsize=3.05,
        handletextpad=0.3,
        columnspacing=0.45,
    )
    panel_label(ax_b, "b", x=-0.42, y=1.055)

    # c — raw same-gene CRISPR distributions in the exact informative-lineage
    # populations used for the adjusted estimates.
    negative_color = COLORS["light_grey"]
    mutant_color = COLORS["orange"]
    rng = np.random.default_rng(7012026)
    centres = np.arange(len(CRISPR_DISTRIBUTION_GENES), dtype=float)
    offsets = {"Hotspot-negative": -0.17, "Hotspot-mutant": 0.17}
    group_colors = {"Hotspot-negative": negative_color, "Hotspot-mutant": mutant_color}
    summary = addiction.set_index("gene")
    tick_labels: list[str] = []
    for index, gene in enumerate(CRISPR_DISTRIBUTION_GENES):
        gene_frame = distributions.loc[distributions.gene.eq(gene)]
        n_mutant = int(gene_frame.hotspotMutant.sum())
        n_negative = int((~gene_frame.hotspotMutant).sum())
        tick_labels.append(f"{gene}\n{n_mutant}/{n_negative}")
        for group in ("Hotspot-negative", "Hotspot-mutant"):
            values = gene_frame.loc[
                gene_frame.genotypeGroup.eq(group), "crisprGeneEffect"
            ].to_numpy(float)
            position = centres[index] + offsets[group]
            box = ax_c.boxplot(
                [values],
                positions=[position],
                widths=0.27,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False,
                boxprops={"facecolor": group_colors[group], "edgecolor": COLORS["grey"], "lw": 0.55},
                whiskerprops={"color": COLORS["grey"], "lw": 0.55},
                capprops={"color": COLORS["grey"], "lw": 0.55},
                medianprops={"color": COLORS["black"], "lw": 0.8},
            )
            for patch in box["boxes"]:
                patch.set_alpha(0.82)
            jitter = rng.uniform(-0.105, 0.105, len(values))
            ax_c.scatter(
                np.full(len(values), position) + jitter,
                values,
                s=4.0,
                facecolor=group_colors[group],
                edgecolor="none",
                alpha=0.27 if group == "Hotspot-negative" else 0.42,
                rasterized=True,
                zorder=1,
            )
        estimate = summary.loc[gene]
        ax_c.text(
            centres[index],
            0.985,
            f"Δ={estimate.adjustedEffect:+.2f}\nq={_compact_q(float(estimate.adjustedFdr))}",
            transform=ax_c.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=3.45,
            linespacing=0.95,
            bbox={"boxstyle": "round,pad=0.13", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        )
    values_all = distributions.crisprGeneEffect.to_numpy(float)
    data_min, data_max = float(np.nanmin(values_all)), float(np.nanmax(values_all))
    data_span = max(data_max - data_min, 1.0)
    ax_c.set_ylim(data_min - 0.04 * data_span, data_max + 0.17 * data_span)
    ax_c.set_xlim(-0.55, len(CRISPR_DISTRIBUTION_GENES) - 0.45)
    ax_c.axhline(0, color=COLORS["very_light_grey"], lw=0.6, zorder=0)
    ax_c.set_xticks(centres, tick_labels, fontsize=3.35)
    ax_c.tick_params(axis="x", length=0, pad=2)
    ax_c.set_ylabel("Chronos gene effect\n(lower = greater dependency)")
    ax_c.grid(axis="y", color=COLORS["very_light_grey"], lw=0.45)
    ax_c.text(
        0.99,
        0.02,
        "n = hotspot-mutant / hotspot-negative",
        transform=ax_c.transAxes,
        ha="right",
        va="bottom",
        fontsize=3.45,
        color=COLORS["grey"],
    )
    ax_c.legend(
        handles=[
            Patch(facecolor=negative_color, edgecolor=COLORS["grey"], label="Hotspot-negative"),
            Patch(facecolor=mutant_color, edgecolor=COLORS["grey"], label="Hotspot-mutant"),
        ],
        **LEGEND_BOX,
        ncol=2,
        loc="lower left",
        fontsize=3.25,
        handlelength=0.9,
        columnspacing=0.75,
    )
    panel_label(ax_c, "c", x=-0.22, y=1.055)

    # d — the complete 37,719-test adjusted PRISM screen. The selected outline set
    # contains all evaluable pre-specified controls plus five separately classified
    # FDR-supported contexts; only eight labels are drawn to prevent text collisions.
    category_style = {
        "FDR ≥ 0.05": (COLORS["grey"], 3.0, 0.18),
        "lower response; FDR < 0.05": (COLORS["blue"], 6.0, 0.48),
        "sensitising; FDR < 0.05": (COLORS["orange"], 7.0, 0.58),
    }
    for category in ("FDR ≥ 0.05", "lower response; FDR < 0.05", "sensitising; FDR < 0.05"):
        color, size, alpha = category_style[category]
        subset = volcano.loc[volcano.volcanoCategory.eq(category)]
        ax_d.scatter(
            subset.adjustedSensitisation,
            subset.volcanoDisplayNegLog10Fdr,
            s=size,
            facecolor=color,
            edgecolor="none",
            alpha=alpha,
            rasterized=True,
            zorder=1 if category == "FDR ≥ 0.05" else 2,
        )
    selected_points = volcano.loc[volcano.isHighlighted]
    ax_d.scatter(
        selected_points.adjustedSensitisation,
        selected_points.volcanoDisplayNegLog10Fdr,
        s=19,
        facecolor="none",
        edgecolor=COLORS["black"],
        linewidth=0.55,
        zorder=4,
    )
    label_positions = {
        ("FBXW7", "BERZOSERTIB"): (-0.158, 0.75, "left"),
        ("TP53", "(+)-CAMPTOTHECIN"): (-0.158, 1.80, "left"),
        ("PIK3R1", "MLN0128"): (-0.158, 3.85, "left"),
        ("RAC1", "RKI-1447"): (-0.158, 6.10, "left"),
        ("NRAS", "TRAMETINIB"): (0.222, 1.35, "right"),
        ("STK11", "EVEROLIMUS"): (0.222, 2.40, "right"),
        ("BRAF", "DABRAFENIB"): (0.222, 3.45, "right"),
        ("KRAS", "TRAMETINIB"): (0.222, 4.55, "right"),
        ("PIK3CA", "COPANLISIB"): (0.222, 11.30, "right"),
    }
    compound_display = {
        "DABRAFENIB": "Dabrafenib",
        "COPANLISIB": "Copanlisib",
        "TRAMETINIB": "Trametinib",
        "MLN0128": "MLN0128",
        "EVEROLIMUS": "Everolimus",
        "RKI-1447": "RKI-1447",
        "BERZOSERTIB": "Berzosertib",
        "(+)-CAMPTOTHECIN": "Camptothecin",
    }
    for record in volcano.loc[volcano.labelOnVolcano].itertuples(index=False):
        key = (record.gene, str(record.compound).upper())
        text_x, text_y, alignment = label_positions[key]
        ax_d.annotate(
            f"{record.gene}–{compound_display.get(key[1], str(record.compound).title())}",
            (record.adjustedSensitisation, record.volcanoDisplayNegLog10Fdr),
            xytext=(text_x, text_y),
            textcoords="data",
            fontsize=3.25,
            ha=alignment,
            va="center",
            bbox={"boxstyle": "round,pad=0.13", "facecolor": "white", "edgecolor": "none", "alpha": 0.9},
            arrowprops={"arrowstyle": "-", "color": COLORS["light_grey"], "lw": 0.4},
            zorder=5,
        )
    ax_d.axvline(0, color=COLORS["black"], lw=0.6, ls=(0, (2, 2)), zorder=0)
    ax_d.axhline(-math.log10(0.05), color=COLORS["grey"], lw=0.6, ls=(0, (2, 2)), zorder=0)
    x_min = float(volcano.adjustedSensitisation.min())
    x_max = float(volcano.adjustedSensitisation.max())
    x_pad = 0.06 * (x_max - x_min)
    ax_d.set_xlim(x_min - x_pad, x_max + x_pad)
    ax_d.set_ylim(0, 16.35)
    ax_d.set_xlabel("Adjusted PRISM sensitisation (ΔAUC)")
    ax_d.set_ylabel("−log10 global FDR q")
    ax_d.grid(color=COLORS["very_light_grey"], lw=0.4)
    n_sensitising = int((volcano.volcanoCategory == "sensitising; FDR < 0.05").sum())
    n_lower = int((volcano.volcanoCategory == "lower response; FDR < 0.05").sum())
    ax_d.text(
        0.98,
        0.97,
        f"{len(volcano):,} tests\n{n_sensitising} sensitising; {n_lower} lower response\n"
        f"{int(volcano.volcanoFdrDisplayCapped.sum())} q values >16 shown at boundary",
        transform=ax_d.transAxes,
        ha="right",
        va="top",
        fontsize=3.55,
    )
    ax_d.legend(
        handles=[
            Line2D([0], [0], marker="o", ls="", mfc=COLORS["orange"], mec="none", label="Sensitising; q < 0.05"),
            Line2D([0], [0], marker="o", ls="", mfc=COLORS["blue"], mec="none", label="Lower response; q < 0.05"),
            Line2D([0], [0], marker="o", ls="", mfc=COLORS["grey"], mec="none", label="q ≥ 0.05"),
            Line2D([0], [0], marker="o", ls="", mfc="none", mec=COLORS["black"], label="Selected context"),
        ],
        **LEGEND_BOX,
        loc="upper left",
        fontsize=3.15,
        handletextpad=0.35,
    )
    panel_label(ax_d, "d", x=-0.22, y=1.055)

    # e — cross-layer synthesis for the same 18 genes shown in the matrix. Point area
    # reports the breadth of FDR-significant sensitising compounds; the axes retain
    # the adjusted CRISPR and strongest positive adjusted PRISM effects separately.
    for role, role_frame in synthesis.groupby("roleInCancer", sort=False):
        ax_e.scatter(
            role_frame.adjustedEffect,
            role_frame.bestPrismSensitisation,
            s=role_frame.pointArea,
            facecolor=ROLE_COLORS.get(role, COLORS["grey"]),
            edgecolor=[
                COLORS["black"] if supported else COLORS["light_grey"]
                for supported in role_frame.crisprFdrSupported
            ],
            linewidth=0.65,
            alpha=0.86,
            zorder=3,
        )
    synthesis_offsets = {
        "KRAS": (4, -7),
        "NRAS": (4, 5),
        "BRAF": (4, 5),
        "PIK3CA": (4, -7),
        "CTNNB1": (-4, 6),
        "PIK3R1": (-4, -7),
        "CREBBP": (-10, -6),
        "STK11": (-4, 5),
    }
    for record in synthesis.loc[synthesis.labelOnPlot].itertuples(index=False):
        dx, dy = synthesis_offsets[record.gene]
        ax_e.annotate(
            record.gene,
            (record.adjustedEffect, record.bestPrismSensitisation),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left" if dx > 0 else "right",
            va="bottom" if dy > 0 else "top",
            fontsize=3.3,
            fontweight="bold" if record.gene in {"KRAS", "BRAF", "PIK3CA", "STK11"} else "normal",
            bbox={"boxstyle": "round,pad=0.10", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
            zorder=5,
        )
    ax_e.axvline(0, color=COLORS["grey"], lw=0.6, ls=(0, (2, 2)), zorder=0)
    ax_e.axhline(0, color=COLORS["grey"], lw=0.6, ls=(0, (2, 2)), zorder=0)
    x_synth = synthesis.adjustedEffect.to_numpy(float)
    y_synth = synthesis.bestPrismSensitisation.to_numpy(float)
    ax_e.set_xlim(float(x_synth.min()) - 0.13, max(0.20, float(x_synth.max()) + 0.08))
    ax_e.set_ylim(-0.008, max(0.225, float(y_synth.max()) + 0.018))
    ax_e.set_xlabel("Lineage-adjusted CRISPR Δ effect")
    ax_e.set_ylabel("Best FDR-significant PRISM ΔAUC")
    ax_e.grid(color=COLORS["very_light_grey"], lw=0.4)
    ax_e.text(
        0.02, 0.02, "more negative = stronger mutant dependency",
        transform=ax_e.transAxes, ha="left", va="bottom", fontsize=3.7, color=COLORS["grey"],
    )
    size_handles = [
        Line2D(
            [0], [0], marker="o", ls="", mfc="white", mec=COLORS["grey"],
            markersize=math.sqrt(13.0 + 2.8 * count), label=str(count),
        )
        for count in (1, 10, 20)
    ]
    ax_e.legend(
        handles=size_handles,
        **LEGEND_BOX,
        title="Sensitising compounds (n)",
        title_fontsize=3.35,
        fontsize=3.15,
        ncol=3,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.025),
        handletextpad=0.25,
        columnspacing=0.45,
    )
    panel_label(ax_e, "e", x=-0.24, y=1.055)

    fig.subplots_adjust(left=0.085, right=0.992, top=0.945, bottom=0.075)
    save_figure(fig, FIGURES / "figure7_functional_evidence")
    plt.close(fig)

    selected.to_csv(SOURCE / "figure7_panel_a_functional_evidence_matrix.csv", index=False)
    highlighted.to_csv(SOURCE / "figure7_panel_b_prism_forest.csv", index=False)
    distributions.to_csv(SOURCE / "figure7_panel_c_crispr_distributions.csv", index=False)
    volcano.to_csv(SOURCE / "figure7_panel_d_prism_volcano.csv", index=False)
    synthesis.to_csv(SOURCE / "figure7_panel_e_cross_layer_synthesis.csv", index=False)
    addiction.to_csv(SOURCE / "figure7_crispr_complete.csv", index=False)
    drug.to_csv(SOURCE / "figure7_prism_complete.csv", index=False)

    assert len(volcano) == len(drug)
    assert int((volcano.volcanoCategory == "sensitising; FDR < 0.05").sum()) == 130
    assert int((volcano.volcanoCategory == "lower response; FDR < 0.05").sum()) == 393
    assert int(volcano.isHighlighted.sum()) == len(highlighted)
    print(
        "Figure 7: "
        f"{len(distributions):,} model-level CRISPR observations across "
        f"{distributions.gene.nunique()} genotypes; {len(volcano):,} PRISM tests; "
        f"{len(controls)} pre-specified controls plus "
        f"{int((highlighted.selectionClass == 'additional FDR-supported context').sum())} "
        "additional contexts; 18-gene integrated matrix and cross-layer synthesis"
    )

    functional_diagnostics(addiction, drug)


def functional_diagnostics(addiction: pd.DataFrame, drug: pd.DataFrame) -> None:
    """Plot unadjusted and lineage-adjusted CRISPR and PRISM estimates."""
    apply_style()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=figsize(180, 90))

    # CRISPR: every eligible gene is shown, with discoveries highlighted.
    crispr = addiction.copy()
    crispr["significantAtFdr05"] = crispr.adjustedFdr < 0.05
    crispr.to_csv(SOURCE / "figureS6_panel_a_crispr_lineage_adjustment.csv", index=False)
    x = crispr.unadjustedEffect.to_numpy(float)
    y = crispr.adjustedEffect.to_numpy(float)
    extent = max(0.25, float(np.nanmax(np.abs(np.r_[x, y]))) * 1.08)
    ax_a.plot(
        [-extent, extent],
        [-extent, extent],
        color=COLORS["light_grey"],
        lw=0.8,
        ls=(0, (2, 2)),
        zorder=0,
    )
    significant = crispr.significantAtFdr05.to_numpy(bool)
    ax_a.scatter(
        x[~significant],
        y[~significant],
        s=17,
        facecolor=COLORS["grey"],
        edgecolor="white",
        linewidth=0.3,
        alpha=0.65,
    )
    ax_a.scatter(
        x[significant],
        y[significant],
        s=24,
        facecolor=COLORS["orange"],
        edgecolor="white",
        linewidth=0.4,
        alpha=0.95,
        zorder=3,
    )
    label_offsets = {
        "NRAS": (4, -8),
        "KRAS": (4, 3),
        "HRAS": (4, -8),
        "BRAF": (4, 3),
        "CTNNB1": (4, 3),
        "PIK3CA": (4, -8),
        "NFE2L2": (4, 3),
    }
    for row in crispr.loc[crispr.gene.isin(label_offsets)].itertuples(index=False):
        ax_a.annotate(
            row.gene,
            (row.unadjustedEffect, row.adjustedEffect),
            xytext=label_offsets[row.gene],
            textcoords="offset points",
            fontsize=4.7,
        )
    rho = spearmanr(x, y).statistic
    ax_a.text(
        0.03,
        0.97,
        f"n={len(crispr)} genes\nSpearman ρ={rho:.2f}",
        transform=ax_a.transAxes,
        va="top",
        fontsize=5.0,
    )
    ax_a.set_xlim(-extent, extent)
    ax_a.set_ylim(-extent, extent)
    ax_a.set_xlabel("Unadjusted CRISPR effect")
    ax_a.set_ylabel("Lineage-adjusted CRISPR effect")
    ax_a.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                ls="",
                mfc=COLORS["orange"],
                mec="white",
                label="FDR q < 0.05",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                ls="",
                mfc=COLORS["grey"],
                mec="white",
                label="FDR q ≥ 0.05",
            ),
            Line2D(
                [0, 1],
                [0, 1],
                color=COLORS["light_grey"],
                ls=(0, (2, 2)),
                label="Identity",
            ),
        ],
        **LEGEND_BOX,
        loc="lower right",
        fontsize=4.5,
        title="Adjusted result",
        title_fontsize=4.8,
    )
    panel_label(ax_a, "a", x=-0.13, y=1.00)

    # PRISM: density conveys the full screen; significant sensitising associations
    # are overlaid without obscuring the adjustment relationship.
    prism = drug[
        [
            "gene",
            "compoundId",
            "compound",
            "unadjustedSensitisation",
            "adjustedSensitisation",
            "adjustedFdr",
            "nModelsAdjusted",
            "nMutAdjusted",
            "nHotspotNegativeAdjusted",
            "nInformativeLineages",
            "comparatorDefinition",
        ]
    ].copy()
    prism["positiveAtFdr05"] = (
        (prism.adjustedFdr < 0.05) & (prism.adjustedSensitisation > 0)
    )
    prism.to_csv(SOURCE / "figureS6_panel_b_prism_lineage_adjustment.csv", index=False)
    px = prism.unadjustedSensitisation.to_numpy(float)
    py = prism.adjustedSensitisation.to_numpy(float)
    limit = max(0.05, float(np.nanquantile(np.abs(np.r_[px, py]), 0.995)))
    density = ax_b.hexbin(
        px,
        py,
        gridsize=46,
        extent=(-limit, limit, -limit, limit),
        mincnt=1,
        bins="log",
        cmap="Greys",
        linewidths=0,
    )
    positive = prism.positiveAtFdr05.to_numpy(bool)
    ax_b.scatter(
        px[positive],
        py[positive],
        s=8,
        facecolor=COLORS["vermillion"],
        edgecolor="white",
        linewidth=0.2,
        alpha=0.58,
        zorder=3,
    )
    ax_b.plot(
        [-limit, limit],
        [-limit, limit],
        color=COLORS["sky"],
        lw=0.8,
        ls=(0, (2, 2)),
        zorder=2,
    )
    prism_rho = spearmanr(px, py).statistic
    ax_b.text(
        0.03,
        0.97,
        f"n={len(prism):,} tests\nSpearman ρ={prism_rho:.2f}",
        transform=ax_b.transAxes,
        va="top",
        fontsize=5.0,
    )
    ax_b.set_xlim(-limit, limit)
    ax_b.set_ylim(-limit, limit)
    ax_b.set_xlabel("Unadjusted PRISM sensitisation (ΔAUC)")
    ax_b.set_ylabel("Lineage-adjusted PRISM sensitisation (ΔAUC)")
    colorbar = fig.colorbar(density, ax=ax_b, fraction=0.042, pad=0.025)
    colorbar.set_label("Tests per hexagon", fontsize=4.8)
    colorbar.ax.tick_params(labelsize=4.4)
    ax_b.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                ls="",
                mfc=COLORS["vermillion"],
                mec="white",
                label="Sensitising; FDR q < 0.05",
            ),
            Line2D(
                [0, 1],
                [0, 1],
                color=COLORS["sky"],
                ls=(0, (2, 2)),
                label="Identity",
            ),
        ],
        **LEGEND_BOX,
        loc="lower right",
        fontsize=4.5,
    )
    panel_label(ax_b, "b", x=-0.13, y=1.00)

    fig.subplots_adjust(left=0.105, right=0.925, top=0.94, bottom=0.16, wspace=0.33)
    save_figure(fig, SUPP / "figureS6_functional_diagnostics")
    plt.close(fig)


def _node_module(gene: str) -> str:
    for module in MODULE_ORDER:
        if gene in MODULE_GENES[module]:
            return module
    return "TP53/lineage"


def _edge_module(record: pd.Series) -> str:
    genes = {record.geneA, record.geneB}
    cancer = record.cancer
    if cancer in {"AML", "MDS", "MPN", "MBN", "CLL", "DLBCL"} or genes & MODULE_GENES["Myeloid"] and genes <= (MODULE_GENES["Myeloid"] | {"TP53"}):
        return "Myeloid"
    if cancer in {"GBM", "ASTR", "ODG", "BRAIN"} or genes & {"IDH1", "IDH2", "ATRX"}:
        return "IDH/chromatin"
    if (cancer in {"COADREAD", "HCC", "CHOL", "PRAD"} and genes & MODULE_GENES["WNT/GI"]) or genes & {"APC", "RNF43", "CTNNB1"}:
        return "WNT/GI"
    if cancer in {"UCEC", "BRCA"} and genes & MODULE_GENES["PI3K"]:
        return "PI3K"
    if genes & MODULE_GENES["RTK–RAS/lung"]:
        return "RTK–RAS/lung"
    if genes & MODULE_GENES["IDH/chromatin"]:
        return "IDH/chromatin"
    if genes & MODULE_GENES["PI3K"]:
        return "PI3K"
    return "TP53/lineage"


def _select_main_contexts(interactions: pd.DataFrame) -> pd.DataFrame:
    wanted = pd.DataFrame(MAIN_CONTEXTS, columns=["cancer", "pair"])
    selected = wanted.merge(interactions, on=["cancer", "pair"], how="left", indicator=True, validate="one_to_one")
    missing = selected[selected._merge.ne("both")][["cancer", "pair"]]
    if len(missing):
        raise RuntimeError(f"Missing curated Figure 8 contexts:\n{missing.to_string(index=False)}")
    selected = selected.drop(columns="_merge")
    stable = (
        selected.signStableNoBurdenLeaveTwoOut.fillna(False)
        & selected.effectStableNoBurdenLeaveTwoOut.fillna(False)
        & selected.leaveTwoOutCrossAssayConcordant.fillna(False)
    )
    selected = selected[
        selected.cmhReplicated.fillna(False)
        & (selected.cmh_full_fdr < 0.05)
        & (selected.leaveTwoOut_full_fdr < 0.05)
        & stable
    ].copy()
    selected["log2CmhOr"] = np.log2(selected.cmh_full_or)
    selected["edgeModule"] = selected.apply(_edge_module, axis=1)
    selected["displaySelection"] = (
        "literature-guided context; primary and leave-two-out FDR < 0.05; "
        "direction/effect stable; cross-assay concordant under both specifications"
    )
    return selected


def _expanded_contexts(interactions: pd.DataFrame, main: pd.DataFrame, evidence: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    canonical = set().union(*MODULE_GENES.values())
    high_evidence = set(evidence.loc[evidence.pctTierMatched >= 40, "gene"]) | canonical
    candidate = interactions[
        interactions.cmhReplicated.fillna(False)
        & (interactions.cmh_full_fdr < 1e-6)
        & (interactions.leaveTwoOut_full_fdr < 0.05)
        & interactions.leaveTwoOutCrossAssayConcordant.fillna(False)
        & interactions.signStableNoBurdenLeaveTwoOut.fillna(False)
        & interactions.effectStableNoBurdenLeaveTwoOut.fillna(False)
        & interactions.geneA.isin(high_evidence)
        & interactions.geneB.isin(high_evidence)
        & (interactions.full_nA >= 10)
        & (interactions.full_nB >= 10)
        & (interactions.cmh_full_or > 0)
    ].copy()
    candidate["log2CmhOr"] = np.log2(candidate.cmh_full_or)
    candidate["selectionScore"] = _safe_log10_q(candidate.cmh_full_fdr) * candidate.log2CmhOr.abs()
    candidate["largeEffect"] = (candidate.cmh_full_or <= 0.5) | (
        (candidate.cmh_full_or >= 2) & (candidate.full_nBoth >= 10)
    )

    # The expanded drawing uses the top five strong exclusivity results per cancer and
    # retains all co-occurring main contexts.  This avoids turning residual mutation-
    # burden structure in hypermutated cohorts into a visually dominant hairball.
    exclusive = candidate[candidate.cmh_full_or <= 0.5]
    top_exclusive = (
        exclusive.sort_values(["cancer", "selectionScore"], ascending=[True, False])
        .groupby("cancer", group_keys=False)
        .head(5)
    )
    expanded = pd.concat([top_exclusive, main], ignore_index=True, sort=False)
    expanded = expanded.sort_values("cmh_full_fdr").drop_duplicates(["cancer", "pair"])
    expanded["log2CmhOr"] = np.log2(expanded.cmh_full_or)
    expanded["edgeModule"] = expanded.apply(_edge_module, axis=1)
    expanded["displaySelection"] = np.where(
        expanded[["cancer", "pair"]].apply(tuple, axis=1).isin(set(MAIN_CONTEXTS)),
        "literature-guided main context",
        "top-five strong exclusive edge within cancer after evidence, concordance and burden-stability filters",
    )
    return candidate, expanded


def _aggregate_edges(contexts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for pair, group in contexts.groupby("pair", sort=False):
        signs = np.sign(group.log2CmhOr)
        if (signs < 0).all():
            category = "mutually exclusive"
        elif (signs > 0).all():
            category = "co-occurring"
        else:
            category = "context-dependent"
        first = group.iloc[0]
        modules = group.edgeModule.value_counts()
        rows.append(
            {
                "pair": pair,
                "geneA": first.geneA,
                "geneB": first.geneB,
                "nCancerContexts": int(group.cancer.nunique()),
                "cancerContexts": ";".join(group.sort_values("cancer").cancer.unique()),
                "directionCategory": category,
                "bestFdr": float(group.cmh_full_fdr.min()),
                "maxAbsLog2CmhOr": float(group.log2CmhOr.abs().max()),
                "medianLog2CmhOr": float(group.log2CmhOr.median()),
                "edgeModule": modules.index[0],
                "nExclusiveContexts": int((group.log2CmhOr < 0).sum()),
                "nCooccurringContexts": int((group.log2CmhOr > 0).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["bestFdr", "pair"]).reset_index(drop=True)


def _network_nodes(edges: pd.DataFrame, prevalence: pd.DataFrame, evidence: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    genes = sorted(set(edges.geneA) | set(edges.geneB))
    nodes = (
        pd.DataFrame({"gene": genes})
        .merge(
            prevalence[
                [
                    "gene", "entrezGeneId", "freqPct", "freqCiLowPct", "freqCiHighPct",
                    "pctCuratedSamplesProfilingGene", "roleInCancer", "cosmicTier", "highConfidence",
                ]
            ],
            on="gene",
            how="left",
        )
        .merge(evidence[["gene", "pctTierMatched", "nTierMatched", "nMutationRecords"]], on="gene", how="left")
        .merge(
            panel[["hugoSymbol", "inCosmicCGC", "inOncoKB", "inPrior2023", "sources"]].rename(columns={"hugoSymbol": "gene"}),
            on="gene",
            how="left",
        )
    )
    nodes["networkModule"] = nodes.gene.map(_node_module)
    degree = defaultdict(int)
    weighted = defaultdict(int)
    for record in edges.itertuples(index=False):
        degree[record.geneA] += 1
        degree[record.geneB] += 1
        weighted[record.geneA] += record.nCancerContexts
        weighted[record.geneB] += record.nCancerContexts
    nodes["degree"] = nodes.gene.map(degree).fillna(0).astype(int)
    nodes["contextWeightedDegree"] = nodes.gene.map(weighted).fillna(0).astype(int)
    return nodes.sort_values(["networkModule", "degree", "gene"], ascending=[True, False, True]).reset_index(drop=True)


MODULE_ANCHORS = {
    "RTK–RAS/lung": (-0.62, 0.46),
    "PI3K": (0.00, 0.67),
    "WNT/GI": (-0.59, -0.38),
    "IDH/chromatin": (-0.03, -0.38),
    "Myeloid": (0.56, -0.49),
    "TP53/lineage": (0.59, 0.34),
}


def _layout_network(edges: pd.DataFrame, nodes: pd.DataFrame, seed: int) -> tuple[nx.Graph, dict[str, np.ndarray]]:
    graph = nx.Graph()
    for record in nodes.itertuples(index=False):
        graph.add_node(record.gene, module=record.networkModule)
    for record in edges.itertuples(index=False):
        graph.add_edge(
            record.geneA,
            record.geneB,
            weight=1.0 + math.log1p(record.nCancerContexts),
        )
    # A hub-and-ring layout inside fixed module anchors is more legible than a global
    # spring layout for this graph because several biologically meaningful modules are
    # disconnected.  Distance between modules is a visual grouping device, not a
    # quantitative measure.  The seed only controls ring rotation.
    rng = np.random.default_rng(seed)
    positions: dict[str, np.ndarray] = {}
    node_degree = nodes.set_index("gene").degree.to_dict()
    for module in MODULE_ORDER:
        module_genes = nodes.loc[nodes.networkModule.eq(module), "gene"].tolist()
        if not module_genes:
            continue
        module_genes = sorted(module_genes, key=lambda gene: (-node_degree.get(gene, 0), gene))
        anchor = np.asarray(MODULE_ANCHORS[module], dtype=float)
        positions[module_genes[0]] = anchor
        ring = module_genes[1:]
        if not ring:
            continue
        radius_x = min(0.22, 0.115 + 0.010 * len(ring))
        radius_y = min(0.18, 0.095 + 0.008 * len(ring))
        rotation = rng.uniform(0, 2 * np.pi)
        for index, gene in enumerate(ring):
            angle = rotation + 2 * np.pi * index / len(ring)
            positions[gene] = anchor + np.array([radius_x * np.cos(angle), radius_y * np.sin(angle)])
    return graph, positions


def _repel_label_positions(
    positions: dict[str, np.ndarray], nodes: pd.DataFrame, iterations: int = 260
) -> dict[str, np.ndarray]:
    genes = list(positions)
    anchors = np.vstack([positions[gene] for gene in genes])
    labels = anchors.copy()
    module_lookup = nodes.set_index("gene").networkModule.to_dict()
    radial = np.vstack(
        [
            positions[gene] - np.asarray(MODULE_ANCHORS[module_lookup[gene]])
            for gene in genes
        ]
    )
    zero = np.linalg.norm(radial, axis=1) < 1e-6
    radial[zero] = anchors[zero]
    still_zero = np.linalg.norm(radial, axis=1) < 1e-6
    radial[still_zero] = np.array([0.0, 1.0])
    radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-6)
    labels += 0.035 * radial
    widths = np.array([0.0155 * max(len(gene), 3) for gene in genes])
    heights = np.full(len(genes), 0.040)
    for _ in range(iterations):
        delta = labels[:, None, :] - labels[None, :, :]
        overlap_x = widths[:, None] + widths[None, :] - np.abs(delta[:, :, 0])
        overlap_y = heights[:, None] + heights[None, :] - np.abs(delta[:, :, 1])
        overlap = (overlap_x > 0) & (overlap_y > 0)
        np.fill_diagonal(overlap, False)
        distance = np.linalg.norm(delta, axis=2) + 1e-6
        force = np.zeros_like(delta)
        force[overlap] = delta[overlap] / distance[overlap, None]
        displacement = 0.0025 * force.sum(axis=1) + 0.015 * (anchors - labels)
        labels += displacement
        labels = np.clip(labels, -1.03, 1.03)
    return {gene: labels[index] for index, gene in enumerate(genes)}


def _draw_prevalence_key(ax: plt.Axes, max_prevalence: float) -> None:
    """Draw a compact node-area key with percentage labels inside the circles."""
    key = ax.inset_axes([0.605, 0.878, 0.375, 0.112], transform=ax.transAxes, zorder=10)
    key.set_xlim(0, 1)
    key.set_ylim(0, 1)
    key.axis("off")
    key.add_patch(
        FancyBboxPatch(
            (0.015, 0.025),
            0.97,
            0.95,
            boxstyle="round,pad=0.015,rounding_size=0.075",
            transform=key.transAxes,
            facecolor="#FAFAFA",
            edgecolor=COLORS["light_grey"],
            linewidth=0.5,
            clip_on=False,
        )
    )
    key.text(
        0.50,
        0.77,
        "Pan-cancer prevalence (node area)",
        ha="center",
        va="center",
        fontsize=3.85,
        color=COLORS["black"],
    )
    for x, level in zip((0.18, 0.48, 0.80), (2, 10, 30)):
        size = 26 + 155 * math.sqrt(level / max(max_prevalence, 1e-6))
        key.scatter(
            x,
            0.31,
            s=size,
            facecolor="white",
            edgecolor=COLORS["black"],
            linewidth=0.5,
            zorder=2,
        )
        key.text(
            x,
            0.31,
            f"{level}%",
            ha="center",
            va="center",
            fontsize=2.75 if level == 2 else 2.95,
            color=COLORS["black"],
            zorder=3,
        )


def _draw_network(
    ax: plt.Axes,
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    *,
    seed: int,
    label_size: float,
    show_module_legend: bool,
) -> None:
    _, positions = _layout_network(edges, nodes, seed)

    # Very light module envelopes clarify the network grammar without implying that a
    # spring-layout distance is a biological distance.
    for module in MODULE_ORDER:
        genes = nodes.loc[nodes.networkModule.eq(module), "gene"]
        points = np.vstack([positions[gene] for gene in genes if gene in positions]) if len(genes) else np.empty((0, 2))
        if not len(points):
            continue
        centre = points.mean(axis=0)
        width = max(np.ptp(points[:, 0]) + 0.24, 0.33)
        height = max(np.ptp(points[:, 1]) + 0.20, 0.28)
        ax.add_patch(
            Ellipse(
                centre,
                width,
                height,
                facecolor=_blend(MODULE_COLORS[module], 0.11, minimum=0.0),
                edgecolor=_blend(MODULE_COLORS[module], 0.42, minimum=0.0),
                lw=0.45,
                zorder=0,
            )
        )

    direction_colors = {
        "mutually exclusive": COLORS["blue"],
        "co-occurring": COLORS["vermillion"],
        "context-dependent": COLORS["purple"],
    }
    strengths = _safe_log10_q(edges.bestFdr).clip(upper=180)
    widths = 0.35 + 1.45 * _rescale(strengths, 5, 120)
    for (_, record), width in zip(edges.iterrows(), widths):
        a, b = positions[record.geneA], positions[record.geneB]
        linestyle = (0, (2, 1.4)) if record.directionCategory == "context-dependent" else "solid"
        if bool(record.get("interactomeDirect", False)):
            ax.plot(
                [a[0], b[0]],
                [a[1], b[1]],
                color=COLORS["black"],
                lw=width + 0.95,
                ls=linestyle,
                alpha=0.68,
                solid_capstyle="round",
                zorder=0.9,
            )
        ax.plot(
            [a[0], b[0]],
            [a[1], b[1]],
            color=direction_colors[record.directionCategory],
            lw=width,
            ls=linestyle,
            alpha=0.72 if bool(record.get("interactomeDirect", False)) else 0.53,
            solid_capstyle="round",
            zorder=1,
        )

    node_lookup = nodes.set_index("gene")
    prevalence = nodes.freqPct.fillna(0).clip(lower=0)
    sizes = 26 + 155 * np.sqrt(prevalence / max(float(prevalence.max()), 1e-6))
    for (gene, record), size in zip(node_lookup.iterrows(), sizes):
        x, y = positions[gene]
        ax.scatter(
            x,
            y,
            s=size,
            facecolor=MODULE_COLORS[record.networkModule],
            edgecolor="white",
            linewidth=0.65,
            zorder=3,
        )

    label_positions = _repel_label_positions(positions, nodes)
    for gene in node_lookup.index:
        x, y = positions[gene]
        lx, ly = label_positions[gene]
        if np.hypot(lx - x, ly - y) > 0.035:
            ax.plot([x, lx], [y, ly], color=COLORS["grey"], lw=0.3, alpha=0.65, zorder=2)
        ax.text(
            lx,
            ly,
            gene,
            ha="center",
            va="center",
            fontsize=label_size,
            zorder=4,
            bbox={"boxstyle": "round,pad=0.08", "facecolor": "white", "edgecolor": "none", "alpha": 0.68},
        )

    # Tight, asymmetric limits use the network panel efficiently while retaining the
    # complete module envelopes, repelled labels and prevalence key.
    ax.set_xlim(-0.95, 0.95)
    ax.set_ylim(-0.85, 1.02)
    ax.set_aspect("equal")
    ax.set_anchor("N")
    ax.axis("off")

    edge_legend = [
        Line2D([0], [0], color=COLORS["blue"], lw=1.5, label="mutual exclusivity"),
        Line2D([0], [0], color=COLORS["vermillion"], lw=1.5, label="co-occurrence"),
        Line2D([0], [0], color=COLORS["purple"], lw=1.5, ls=(0, (2, 1.4)), label="context-dependent sign"),
        Line2D([0], [0], color=COLORS["grey"], lw=0.45, label=r"$q=10^{-5}$"),
        Line2D([0], [0], color=COLORS["grey"], lw=1.8, label=r"$q\leq10^{-120}$"),
        Line2D([0], [0], color=COLORS["black"], lw=2.7, label="direct interactome link (outline)"),
    ]
    module_legend = [
        Line2D([0], [0], marker="o", ls="", mfc=MODULE_COLORS[module], mec="white", label=module)
        for module in MODULE_ORDER
        if module in set(nodes.networkModule)
    ]
    if show_module_legend:
        max_prevalence = max(float(prevalence.max()), 1e-6)
        _draw_prevalence_key(ax, max_prevalence)
        module_key = ax.legend(
            handles=module_legend,
            **LEGEND_BOX,
            fontsize=4.15,
            ncol=3,
            loc="lower center",
            bbox_to_anchor=(0.50, -0.050),
            columnspacing=0.8,
            handlelength=1.1,
            title="Descriptive group",
            title_fontsize=4.4,
        )
        ax.add_artist(module_key)
        ax.legend(
            handles=edge_legend,
            **LEGEND_BOX,
            fontsize=4.15,
            ncol=3,
            loc="lower center",
            bbox_to_anchor=(0.50, -0.155),
            columnspacing=0.8,
            handlelength=1.3,
            title="Association (colour); q value (width); interactome (outline)",
            title_fontsize=4.4,
        )
    else:
        ax.legend(
            handles=edge_legend,
            **LEGEND_BOX,
            fontsize=4.15,
            ncol=3,
            loc="lower center",
            bbox_to_anchor=(0.50, -0.07),
            columnspacing=0.8,
            handlelength=1.3,
            title="Association (colour); q value (width); interactome (outline)",
            title_fontsize=4.4,
        )


def _module_by_cancer(main: pd.DataFrame, cancer_frame: pd.DataFrame, screened_cancers: set[str]) -> pd.DataFrame:
    summaries = (
        main.groupby(["edgeModule", "cancer"])
        .agg(
            meanLog2CmhOr=("log2CmhOr", "mean"),
            medianLog2CmhOr=("log2CmhOr", "median"),
            nCuratedContexts=("pair", "size"),
            nExclusiveContexts=("log2CmhOr", lambda values: int((values < 0).sum())),
            nCooccurringContexts=("log2CmhOr", lambda values: int((values > 0).sum())),
            minimumFdr=("cmh_full_fdr", "min"),
        )
        .reset_index()
    )
    grid = pd.MultiIndex.from_product(
        [MODULE_ORDER, cancer_frame.cancerGroup.tolist()], names=["edgeModule", "cancer"]
    ).to_frame(index=False)
    grid = grid.merge(summaries, on=["edgeModule", "cancer"], how="left")
    grid = grid.merge(cancer_frame, left_on="cancer", right_on="cancerGroup", how="left").drop(columns="cancerGroup")
    grid["coverageState"] = np.select(
        [grid.nCuratedContexts.notna(), grid.cancer.isin(screened_cancers)],
        ["literature-guided display estimate", "association-screened; no selected main edge"],
        default="not interaction-screened at Stage 17 eligibility thresholds",
    )
    grid["nCuratedContexts"] = grid.nCuratedContexts.fillna(0).astype(int)
    grid["nExclusiveContexts"] = grid.nExclusiveContexts.fillna(0).astype(int)
    grid["nCooccurringContexts"] = grid.nCooccurringContexts.fillna(0).astype(int)
    return grid


def _draw_module_matrix(ax: plt.Axes, main: pd.DataFrame, source_complete: pd.DataFrame) -> None:
    cancers = (
        main.groupby("cancer").size().sort_values(ascending=False).index.tolist()
    )
    matrix = (
        source_complete.pivot(index="edgeModule", columns="cancer", values="meanLog2CmhOr")
        .reindex(index=MODULE_ORDER, columns=cancers)
    )
    counts = (
        source_complete.pivot(index="edgeModule", columns="cancer", values="nCuratedContexts")
        .reindex(index=MODULE_ORDER, columns=cancers)
    )
    values = matrix.to_numpy(float)
    finite = np.abs(values[np.isfinite(values)])
    limit = max(2.5, float(np.nanpercentile(finite, 92)) if len(finite) else 2.5)
    limit = min(limit, 5.0)
    image = ax.imshow(values, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
    ax.set_xticks(np.arange(len(cancers)), cancers, rotation=58, ha="right", fontsize=4.35)
    ax.set_yticks(np.arange(len(MODULE_ORDER)), MODULE_ORDER, fontsize=4.75)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, len(cancers), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(MODULE_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.65)
    ax.tick_params(which="minor", length=0)
    for row in range(len(MODULE_ORDER)):
        for column in range(len(cancers)):
            value = values[row, column]
            count = int(counts.iat[row, column]) if pd.notna(counts.iat[row, column]) else 0
            if not np.isfinite(value) or count == 0:
                ax.text(column, row, "·", ha="center", va="center", fontsize=4.2, color=COLORS["light_grey"])
                continue
            color = "white" if abs(value) > 0.57 * limit else COLORS["black"]
            ax.text(
                column,
                row,
                f"{value:+.1f}\n({count})",
                ha="center",
                va="center",
                fontsize=3.55,
                color=color,
                linespacing=0.85,
            )
    cbar = plt.colorbar(image, ax=ax, orientation="horizontal", fraction=0.045, pad=0.12, aspect=35)
    cbar.set_label(
        "Descriptive mean log2 odds ratio; parentheses show selected contexts",
        fontsize=4.8,
    )
    cbar.ax.tick_params(labelsize=4.4)


def _context_composition(contexts: pd.DataFrame) -> pd.DataFrame:
    return (
        contexts.assign(
            direction=np.where(contexts.log2CmhOr < 0, "mutually exclusive", "co-occurring")
        )
        .groupby(["cancer", "direction"])
        .size()
        .rename("nContexts")
        .reset_index()
    )


def _draw_context_composition(
    ax: plt.Axes,
    context_counts: pd.DataFrame,
    *,
    xlabel: str,
) -> None:
    wide = context_counts.pivot(index="cancer", columns="direction", values="nContexts").fillna(0)
    wide = wide.loc[wide.sum(axis=1).sort_values().index]
    y = np.arange(len(wide))
    exclusive_values = wide.get("mutually exclusive", pd.Series(0, index=wide.index)).to_numpy()
    co_values = wide.get("co-occurring", pd.Series(0, index=wide.index)).to_numpy()
    ax.barh(y, exclusive_values, color=COLORS["blue"], label="mutually exclusive")
    ax.barh(y, co_values, left=exclusive_values, color=COLORS["vermillion"], label="co-occurring")
    ax.set_yticks(y, wide.index, fontsize=4.25)
    ax.set_xlabel(xlabel)
    ax.legend(
        **LEGEND_BOX,
        fontsize=4.0,
        ncol=2,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.005),
        borderaxespad=0,
        columnspacing=0.85,
        handlelength=1.2,
    )


def _complete_coverage_tables(
    interactions: pd.DataFrame,
    main: pd.DataFrame,
    expanded: pd.DataFrame,
    main_nodes: pd.DataFrame,
    expanded_nodes: pd.DataFrame,
    panel: pd.DataFrame,
    prevalence: pd.DataFrame,
    evidence: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cancer_complete = pd.read_csv(COMPLETE / "curated_cancer_gene_prevalence_complete.csv")
    study_complete = pd.read_csv(COMPLETE / "curated_study_completeness.csv")
    cancer_frame = (
        cancer_complete[
            [
                "cancerGroup", "cancerGroupLabel", "cancerGroupSourceLabels", "nEligibleSamples",
                "nEligiblePatientKeys", "nWesWgsSamples", "nWesWgsDocumentedSamples",
                "nWesWgsAssumedSamples", "nTargetedSamples",
            ]
        ]
        .drop_duplicates("cancerGroup")
        .sort_values("cancerGroup")
        .reset_index(drop=True)
    )
    screened = interactions.groupby("cancer").size().rename("nScreenedGenePairTests")
    main_counts = main.groupby("cancer").size().rename("nMainNetworkContexts")
    expanded_counts = expanded.groupby("cancer").size().rename("nExpandedNetworkContexts")
    cancer_coverage = (
        cancer_frame.set_index("cancerGroup")
        .join(screened)
        .join(main_counts)
        .join(expanded_counts)
        .fillna(
            {
                "nScreenedGenePairTests": 0,
                "nMainNetworkContexts": 0,
                "nExpandedNetworkContexts": 0,
            }
        )
        .reset_index()
    )
    for column in ("nScreenedGenePairTests", "nMainNetworkContexts", "nExpandedNetworkContexts"):
        cancer_coverage[column] = cancer_coverage[column].astype(int)
    cancer_coverage["interactionCoverageState"] = np.where(
        cancer_coverage.nScreenedGenePairTests > 0,
        "interaction-screened",
        "not interaction-screened at Stage 17 eligibility thresholds",
    )

    main_gene_set = set(main_nodes.gene)
    expanded_gene_set = set(expanded_nodes.gene)
    nodes_complete = (
        panel.rename(columns={"hugoSymbol": "gene"})
        .merge(
            prevalence[
                [
                    "gene", "nProfiled", "nMutated", "freqPct", "freqCiLowPct", "freqCiHighPct",
                    "pctCuratedSamplesProfilingGene",
                ]
            ],
            on="gene",
            how="left",
        )
        .merge(evidence, on="gene", how="left")
    )
    nodes_complete["networkModule"] = nodes_complete.gene.map(_node_module)
    nodes_complete["inMainNetwork"] = nodes_complete.gene.isin(main_gene_set)
    nodes_complete["inExpandedNetwork"] = nodes_complete.gene.isin(expanded_gene_set)

    interactions.to_csv(SOURCE / "figure8_interaction_screen_complete.csv", index=False)
    nodes_complete.to_csv(SOURCE / "figure8_nodes_complete.csv", index=False)
    cancer_coverage.to_csv(SOURCE / "figure8_cancer_coverage_complete.csv", index=False)
    study_complete.to_csv(SOURCE / "figure8_study_coverage_complete.csv", index=False)
    return cancer_frame, cancer_coverage


def figure8() -> None:
    apply_style()
    interactions = pd.read_csv(TABLES / "cooccurrence_curated_adjusted_sensitivity.csv")
    prevalence = pd.read_csv(TABLES / "gene_frequencies_curated.csv")
    evidence = pd.read_csv(TABLES / "cmc_evidence_curated.csv")
    panel = pd.read_csv(PROCESSED / "gene_panel.csv")

    main = _select_main_contexts(interactions)
    candidate, expanded = _expanded_contexts(interactions, main, evidence)
    main_edges = _aggregate_edges(main)
    expanded_edges = _aggregate_edges(expanded)
    edge_universe = (
        pd.concat(
            [
                main_edges[["pair", "geneA", "geneB"]],
                expanded_edges[["pair", "geneA", "geneB"]],
            ],
            ignore_index=True,
        )
        .drop_duplicates("pair")
        .reset_index(drop=True)
    )
    interactome_support = _interactome_support(edge_universe)
    main_edges = main_edges.merge(interactome_support, on="pair", how="left", validate="one_to_one")
    expanded_edges = expanded_edges.merge(interactome_support, on="pair", how="left", validate="one_to_one")
    main_nodes = _network_nodes(main_edges, prevalence, evidence, panel)
    expanded_nodes = _network_nodes(expanded_edges, prevalence, evidence, panel)
    interactome_null, interactome_null_distribution = _degree_matched_interactome_null(
        {"main": main_edges, "expanded": expanded_edges}, panel
    )

    cancer_frame, cancer_coverage = _complete_coverage_tables(
        interactions, main, expanded, main_nodes, expanded_nodes, panel, prevalence, evidence
    )
    module_complete = _module_by_cancer(main, cancer_frame, set(interactions.cancer))

    main.to_csv(SOURCE / "figure8_main_edge_contexts.csv", index=False)
    main_edges.to_csv(SOURCE / "figure8_main_edge_aggregates.csv", index=False)
    main_nodes.to_csv(SOURCE / "figure8_main_nodes.csv", index=False)
    interactome_crosswalk = edge_universe.merge(
        interactome_support, on="pair", how="left", validate="one_to_one"
    )
    interactome_crosswalk["inMainNetwork"] = interactome_crosswalk.pair.isin(set(main_edges.pair))
    interactome_crosswalk["inExpandedNetwork"] = interactome_crosswalk.pair.isin(set(expanded_edges.pair))
    interactome_crosswalk.to_csv(SOURCE / "figure8_interactome_edge_crosswalk.csv", index=False)
    interactome_null.to_csv(TABLES / "interactome_degree_matched_null.csv", index=False)
    interactome_null.to_csv(SOURCE / "figure8_interactome_degree_matched_null.csv", index=False)
    interactome_null_distribution.to_csv(
        SOURCE / "figureS3_interactome_null_distribution.csv", index=False
    )
    candidate.to_csv(SOURCE / "figureS3_expanded_candidate_contexts.csv", index=False)
    expanded.to_csv(SOURCE / "figureS3_expanded_display_contexts.csv", index=False)
    expanded_edges.to_csv(SOURCE / "figureS3_expanded_edge_aggregates.csv", index=False)
    expanded_nodes.to_csv(SOURCE / "figureS3_expanded_nodes.csv", index=False)
    module_complete.to_csv(SOURCE / "figure8_module_by_cancer_complete.csv", index=False)

    # Main Figure 8.  The network occupies one coherent left-hand field while the two
    # supporting summaries share the right-hand column.  Anchoring the equal-aspect
    # network at its top aligns the a and b panel letters despite their different axes.
    context_counts = _context_composition(main)
    context_counts.to_csv(SOURCE / "figure8_panel_c_cancer_context_composition.csv", index=False)

    fig = plt.figure(figsize=figsize(180, 136))
    gs = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[1.48, 1.0],
        height_ratios=[0.92, 1.08],
        hspace=0.22,
        wspace=0.34,
    )
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])
    _draw_network(ax_a, main_edges, main_nodes, seed=31, label_size=4.35, show_module_legend=True)
    _draw_module_matrix(ax_b, main, module_complete)
    _draw_context_composition(
        ax_c,
        context_counts,
        xlabel="Literature-guided display contexts",
    )
    panel_label(ax_c, "c", x=-0.23, y=1.02)
    fig.subplots_adjust(left=0.04, right=0.985, top=0.955, bottom=0.075)
    for x, label in ((0.018, "a"), (0.565, "b")):
        figure_panel_label(fig, label, x=x, y=0.978)
    save_figure(fig, FIGURES / "figure8_curated_network")
    plt.close(fig)

    # Supplementary Figure S3 uses the same one-network/two-summary grammar.
    fig = plt.figure(figsize=figsize(180, 137))
    gs = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[1.48, 1.0],
        height_ratios=[0.88, 1.12],
        hspace=0.24,
        wspace=0.34,
    )
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    _draw_network(ax_a, expanded_edges, expanded_nodes, seed=47, label_size=3.85, show_module_legend=True)

    high_evidence_genes = set(evidence.loc[evidence.pctTierMatched >= 40, "gene"]) | set().union(*MODULE_GENES.values())
    stable_under_background_burden = (
        interactions.signStableNoBurdenLeaveTwoOut.fillna(False)
        & interactions.effectStableNoBurdenLeaveTwoOut.fillna(False)
        & (interactions.leaveTwoOut_full_fdr < 0.05)
    )
    concordant_under_both = (
        interactions.cmhReplicated.fillna(False)
        & interactions.leaveTwoOutCrossAssayConcordant.fillna(False)
    )
    cascade = [
        ("cancer–pair tests", len(interactions)),
        ("primary FDR < 0.05", int((interactions.cmh_full_fdr < 0.05).sum())),
        ("stable with leave-two-out", int(((interactions.cmh_full_fdr < 0.05) & stable_under_background_burden).sum())),
        ("cross-assay concordant in both", int(((interactions.cmh_full_fdr < 0.05) & stable_under_background_burden & concordant_under_both).sum())),
        (
            "high-evidence; primary q < 1e-6",
            int(
                (
                    concordant_under_both
                    & stable_under_background_burden
                    & (interactions.cmh_full_fdr < 1e-6)
                    & interactions.geneA.isin(high_evidence_genes)
                    & interactions.geneB.isin(high_evidence_genes)
                    & (interactions.full_nA >= 10)
                    & (interactions.full_nB >= 10)
                ).sum()
            ),
        ),
        ("expanded display contexts", len(expanded)),
    ]
    cascade_frame = pd.DataFrame(cascade, columns=["selectionStage", "nCancerPairRows"])
    cascade_frame["displayOrder"] = np.arange(1, len(cascade_frame) + 1)
    cascade_frame.to_csv(SOURCE / "figureS3_panel_b_selection_cascade.csv", index=False)
    cascade_plot = cascade_frame.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(cascade_plot))
    ax_b.barh(y, cascade_plot.nCancerPairRows, color=COLORS["blue"], alpha=0.82)
    ax_b.set_xscale("log")
    ax_b.set_yticks(y, cascade_plot.selectionStage, fontsize=4.25)
    ax_b.set_xlabel("Cancer-specific gene-pair tests (log scale)")
    for yy, value in zip(y, cascade_plot.nCancerPairRows):
        ax_b.text(value * 1.10, yy, f"{int(value):,}", va="center", fontsize=4.25)
    ax_b.set_xlim(25, max(cascade_frame.nCancerPairRows) * 2.6)

    stability = interactions[
        (interactions.noBurden_full_or > 0)
        & (interactions.leaveTwoOut_full_or > 0)
    ].copy()
    stability["primaryLog2Or"] = np.log2(stability.noBurden_full_or)
    stability["leaveTwoOutLog2Or"] = np.log2(stability.leaveTwoOut_full_or)
    stability["inExpandedDisplay"] = stability[["cancer", "pair"]].apply(tuple, axis=1).isin(
        set(expanded[["cancer", "pair"]].apply(tuple, axis=1))
    )
    # Retain the coordinates and stability flags used by this panel.  The
    # complete three-specification scan remains available without duplication in
    # Supplementary Data 2.
    stability_source = stability[
        [
            "cancer", "pair", "geneA", "geneB", "primaryLog2Or",
            "leaveTwoOutLog2Or", "signStableNoBurdenLeaveTwoOut",
            "effectStableNoBurdenLeaveTwoOut", "inExpandedDisplay",
        ]
    ].copy()
    stability_source.to_csv(SOURCE / "figureS3_panel_c_burden_stability.csv", index=False)
    stale_cross_assay = SOURCE / "figureS3_panel_c_cross_assay_stability.csv"
    stale_cross_assay.unlink(missing_ok=True)
    all_values = np.r_[
        stability.primaryLog2Or.to_numpy(), stability.leaveTwoOutLog2Or.to_numpy()
    ]
    extent = max(2.5, float(np.nanquantile(np.abs(all_values), 0.995)))
    ax_c.hexbin(
        stability.primaryLog2Or,
        stability.leaveTwoOutLog2Or,
        gridsize=42,
        extent=(-extent, extent, -extent, extent),
        mincnt=1,
        bins="log",
        cmap="Greys",
        linewidths=0,
    )
    highlight = stability.inExpandedDisplay
    ax_c.scatter(
        stability.loc[highlight, "primaryLog2Or"],
        stability.loc[highlight, "leaveTwoOutLog2Or"],
        s=14,
        facecolor=COLORS["orange"],
        edgecolor="white",
        linewidth=0.35,
        alpha=0.9,
        zorder=3,
    )
    ax_c.plot([-extent, extent], [-extent, extent], color=COLORS["sky"], lw=0.7, ls=(0, (2, 2)))
    rho = spearmanr(
        stability.primaryLog2Or, stability.leaveTwoOutLog2Or
    ).statistic
    ax_c.text(
        0.03,
        0.96,
        f"n={len(stability):,} estimable rows\nSpearman ρ={rho:.2f}",
        transform=ax_c.transAxes,
        va="top",
        fontsize=4.35,
    )
    ax_c.set_xlim(-extent, extent)
    ax_c.set_ylim(-extent, extent)
    ax_c.set_xlabel("Primary no-burden log2 OR")
    ax_c.set_ylabel("Leave-two-out background-burden log2 OR")
    ax_c.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                ls="",
                mfc=COLORS["orange"],
                mec="white",
                label="Expanded display",
            ),
            Line2D(
                [0, 1],
                [0, 1],
                color=COLORS["sky"],
                ls=(0, (2, 2)),
                label="Identity",
            ),
        ],
        **LEGEND_BOX,
        loc="lower right",
        fontsize=4.2,
    )
    panel_label(ax_c, "c", x=-0.23, y=1.02)

    fig.subplots_adjust(left=0.045, right=0.985, top=0.955, bottom=0.075)
    for x, label in ((0.018, "a"), (0.565, "b")):
        figure_panel_label(fig, label, x=x, y=0.978)
    save_figure(fig, SUPP / "figureS3_expanded_network")
    plt.close(fig)

    # Hard validation protects the coverage contract and display/data correspondence.
    cancer_coverage_rows = len(pd.read_csv(SOURCE / "figure8_cancer_coverage_complete.csv"))
    study_coverage_rows = len(pd.read_csv(SOURCE / "figure8_study_coverage_complete.csv"))
    node_coverage_rows = len(pd.read_csv(SOURCE / "figure8_nodes_complete.csv"))
    assert cancer_coverage_rows == len(cancer_frame)
    assert study_coverage_rows == len(pd.read_csv(COMPLETE / "curated_study_completeness.csv"))
    assert node_coverage_rows == len(panel.drop_duplicates("hugoSymbol"))
    assert len(pd.read_csv(SOURCE / "figure8_interaction_screen_complete.csv")) == len(interactions)
    assert len(module_complete) == len(cancer_frame) * len(MODULE_ORDER)
    assert set(main.pair) == set(main_edges.pair)
    assert set(expanded.pair) == set(expanded_edges.pair)
    assert len(interactome_crosswalk) == len(set(main_edges.pair) | set(expanded_edges.pair))
    assert interactome_crosswalk.interactomeDirect.notna().all()

    print(
        f"Figure 8: {len(main_nodes)} nodes, {len(main_edges)} unique edges, "
        f"{len(main)} cancer-context rows across {main.cancer.nunique()} cancers"
    )
    print(
        f"Supplementary Figure S3: {len(expanded_nodes)} nodes, {len(expanded_edges)} unique edges, "
        f"{len(expanded)} cancer-context rows across {expanded.cancer.nunique()} cancers"
    )
    print(
        "Supplied interactome: "
        f"{int(main_edges.interactomeDirect.sum())}/{len(main_edges)} direct main-network links; "
        f"{int(expanded_edges.interactomeDirect.sum())}/{len(expanded_edges)} direct expanded-network links; "
        f"{int(interactome_crosswalk.interactomePathLengthAtMost2.notna().sum())}/"
        f"{len(interactome_crosswalk)} links connected within two steps"
    )
    print(
        f"Complete sources: {len(interactions):,} screened contexts; "
        f"{len(cancer_coverage)} cancers; {study_coverage_rows:,} studies; "
        f"{node_coverage_rows:,} genes"
    )


def main() -> None:
    setup_dirs()
    figure7()
    print(
        "Wrote Figure 7 and Supplementary Figure S6 "
        "(PDF/SVG/PNG plus complete CRISPR/PRISM source data)"
    )
    figure8()
    print("Wrote Figure 8 and Supplementary Figure S3 (PDF/SVG/PNG plus complete source data)")


if __name__ == "__main__":
    main()
