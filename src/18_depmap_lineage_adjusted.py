"""Stage 18 — lineage-adjusted functional support from DepMap and PRISM.

This stage revisits the project's cell-line analyses without interpreting a lineage-
restricted genotype as a genotype-specific phenotype.  For each hotspot-mutated gene,
it fits a within-Oncotree-lineage fixed-effect model for the matched CRISPR gene effect.
It then applies the same within-lineage model to every evaluable PRISM compound.

The coefficient is estimated only from lineages containing both hotspot-mutant and
hotspot-negative models.  A hotspot-negative model has no hotspot call for the tested
gene and is not asserted to be genomically wild-type. Inference uses HC3
heteroskedasticity-robust standard errors. These
observational cell-line associations are described as functional support, never as
causal or clinical validation.

Outputs
-------
results/tables/depmap_addiction_lineage_adjusted.csv
results/tables/depmap_drug_lineage_adjusted.csv
results/figures/figure14_functional_support.{pdf,svg,png}
results/source_data/figure14_panel_{a,b,c,d}_*.csv
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import matplotlib

# A non-interactive backend makes the pipeline deterministic on headless runners and
# prevents macOS from trying to initialise a GUI while exporting vector figures.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import spearmanr, t
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from config import EXTERNAL, FIGURES, PROCESSED, RESULTS, TABLES
import nature_style
from nature_style import COLORS


DM = EXTERNAL / "depmap"
SOURCE = RESULTS / "source_data"
MIN_HOTSPOT_MODELS = 10
MIN_MUT_CRISPR = 5
MIN_WT_CRISPR = 10
MIN_MUT_PRISM = 5
MIN_WT_PRISM = 20

META_COLUMNS = {
    "SequencingID",
    "ModelID",
    "ModelConditionID",
    "IsDefaultEntryForModel",
    "IsDefaultEntryForMC",
}

# Positive controls are specified before looking at the adjusted screen.  They are
# included to calibrate the analysis, not to claim that this dataset validates a drug.
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

PRISM_ADDITIONAL_SENSITIVITIES = [
    ("PIK3R1", "MLN0128"),
    ("STK11", "EVEROLIMUS"),
    ("RAC1", "RKI-1447"),
    ("FBXW7", "BERZOSERTIB"),
    ("TP53", "(+)-CAMPTOTHECIN"),
]


def gene_id(column: str) -> int | None:
    """Return the terminal Entrez identifier from a DepMap matrix column."""
    match = re.search(r"\((\d+)\)$", str(column))
    return int(match.group(1)) if match else None


def load_gene_matrix(path: Path) -> tuple[pd.DataFrame, dict[int, str]]:
    """Load a model-by-gene matrix and map Entrez identifiers to columns."""
    frame = pd.read_csv(path)
    id_column = "ModelID" if "ModelID" in frame.columns else frame.columns[0]
    frame = frame.set_index(id_column)
    frame = frame.drop(columns=[c for c in META_COLUMNS if c in frame.columns], errors="ignore")
    mapping = {entrez: col for col in frame.columns if (entrez := gene_id(col)) is not None}
    return frame, mapping


def load_default_hotspot() -> tuple[pd.DataFrame, dict[int, str]]:
    """Load hotspot calls from each model's default molecular-profile entry only."""
    frame = pd.read_csv(DM / "OmicsSomaticMutationsMatrixHotspot.csv")
    if "IsDefaultEntryForModel" in frame:
        frame = frame.loc[frame["IsDefaultEntryForModel"].eq("Yes")].copy()
    gene_columns = [c for c in frame.columns if gene_id(c) is not None]
    # Max aggregation is a safeguard for future releases with duplicated default rows.
    frame = frame.groupby("ModelID", sort=False)[gene_columns].max(numeric_only=True)
    mapping = {gene_id(col): col for col in gene_columns}
    return frame, mapping


def load_default_damaging() -> tuple[pd.DataFrame, dict[int, str]]:
    """Load broad damaging-alteration calls for three-group comparator sensitivities."""
    frame = pd.read_csv(DM / "OmicsSomaticMutationsMatrixDamaging.csv")
    if "IsDefaultEntryForModel" in frame:
        frame = frame.loc[frame["IsDefaultEntryForModel"].eq("Yes")].copy()
    gene_columns = [column for column in frame.columns if gene_id(column) is not None]
    frame = frame.groupby("ModelID", sort=False)[gene_columns].max(numeric_only=True)
    mapping = {gene_id(column): column for column in gene_columns}
    return frame, mapping


def compare_three_genotype_groups(
    y: np.ndarray,
    hotspot: np.ndarray,
    damaging: np.ndarray,
    lineage: np.ndarray,
    *,
    min_hotspot: int,
    min_other: int,
    min_unaltered: int,
) -> dict[str, float] | None:
    """Fit hotspot, other-alteration and no-retained-alteration contrasts.

    The model uses lineage fixed effects and HC3 inference.  The no-retained-alteration
    group is the reference; the broad damaging matrix separates other retained
    alterations from the no-retained-alteration comparator.
    """
    valid = (
        np.isfinite(y)
        & np.isfinite(hotspot)
        & np.isfinite(damaging)
        & pd.notna(lineage)
    )
    if not valid.any():
        return None
    outcome = y[valid].astype(float)
    hotspot_value = hotspot[valid].astype(float) > 0
    damaging_value = damaging[valid].astype(float) > 0
    lineage_value = lineage[valid].astype(str)
    group = np.where(hotspot_value, "canonical hotspot", np.where(damaging_value, "other alteration", "no retained alteration"))

    frame = pd.DataFrame({"outcome": outcome, "group": group, "lineage": lineage_value})
    counts = frame.group.value_counts()
    if (
        int(counts.get("canonical hotspot", 0)) < min_hotspot
        or int(counts.get("other alteration", 0)) < min_other
        or int(counts.get("no retained alteration", 0)) < min_unaltered
    ):
        return None

    # Restrict to lineages that identify the principal hotspot-versus-unaltered
    # contrast; other-alteration models in those same lineages remain in the fit.
    lineage_counts = frame.groupby(["lineage", "group"]).size().unstack(fill_value=0)
    informative_lineages = lineage_counts.index[
        (lineage_counts.get("canonical hotspot", 0) > 0)
        & (lineage_counts.get("no retained alteration", 0) > 0)
    ]
    frame = frame[frame.lineage.isin(informative_lineages)].copy()
    counts = frame.group.value_counts()
    if (
        int(counts.get("canonical hotspot", 0)) < min_hotspot
        or int(counts.get("other alteration", 0)) < min_other
        or int(counts.get("no retained alteration", 0)) < min_unaltered
    ):
        return None

    design = pd.DataFrame(
        {
            "canonicalHotspot": frame.group.eq("canonical hotspot").astype(float),
            "otherAlteration": frame.group.eq("other alteration").astype(float),
        },
        index=frame.index,
    )
    lineage_dummies = pd.get_dummies(frame.lineage, prefix="lineage", drop_first=True, dtype=float)
    design = pd.concat([design, lineage_dummies], axis=1)
    design = sm.add_constant(design, has_constant="add")
    fit = sm.OLS(frame.outcome.to_numpy(float), design.to_numpy(float)).fit(cov_type="HC3")
    names = list(design.columns)

    def coefficient(name: str) -> tuple[float, float, float, float, float]:
        index = names.index(name)
        ci = fit.conf_int(alpha=0.05)[index]
        return (
            float(fit.params[index]),
            float(fit.bse[index]),
            float(ci[0]),
            float(ci[1]),
            float(fit.pvalues[index]),
        )

    hotspot_effect, hotspot_se, hotspot_low, hotspot_high, hotspot_p = coefficient("canonicalHotspot")
    other_effect, other_se, other_low, other_high, other_p = coefficient("otherAlteration")
    return {
        "nModels": int(len(frame)),
        "nCanonicalHotspot": int(counts.get("canonical hotspot", 0)),
        "nOtherAlteration": int(counts.get("other alteration", 0)),
        "nNoRetainedAlteration": int(counts.get("no retained alteration", 0)),
        "nInformativeLineages": int(len(informative_lineages)),
        "hotspotVsUnalteredEffect": hotspot_effect,
        "hotspotVsUnalteredStandardError": hotspot_se,
        "hotspotVsUnalteredCiLow": hotspot_low,
        "hotspotVsUnalteredCiHigh": hotspot_high,
        "hotspotVsUnalteredP": hotspot_p,
        "otherVsUnalteredEffect": other_effect,
        "otherVsUnalteredStandardError": other_se,
        "otherVsUnalteredCiLow": other_low,
        "otherVsUnalteredCiHigh": other_high,
        "otherVsUnalteredP": other_p,
        "referenceGroup": "no retained hotspot or damaging alteration in the tested gene",
    }


def _ols_hc3(y: np.ndarray, x: np.ndarray, groups: np.ndarray) -> dict[str, float] | None:
    """Fixed-effect OLS for a binary exposure with an exact HC3 robust variance.

    Group-demeaning is the Frisch-Waugh-Lovell transformation for lineage fixed
    effects.  The full-model leverage is 1/n_group + x_within^2/sum(x_within^2),
    which lets us calculate the HC3 sandwich variance without repeatedly constructing
    a large dummy-variable design matrix.
    """
    codes, _ = pd.factorize(groups, sort=False)
    n_groups = int(codes.max()) + 1
    counts = np.bincount(codes).astype(float)
    x_means = np.bincount(codes, weights=x) / counts
    y_means = np.bincount(codes, weights=y) / counts
    x_within = x - x_means[codes]
    y_within = y - y_means[codes]
    sxx = float(x_within @ x_within)
    df_resid = int(len(y) - n_groups - 1)
    if not np.isfinite(sxx) or sxx <= 1e-12 or df_resid <= 1:
        return None

    beta = float((x_within @ y_within) / sxx)
    residual = y_within - beta * x_within
    leverage = 1.0 / counts[codes] + x_within**2 / sxx
    usable = np.abs(x_within) > 1e-12
    denom = np.maximum(1.0 - leverage[usable], 1e-8)
    score = x_within[usable] * residual[usable] / denom
    variance = float(score @ score) / (sxx**2)
    if not np.isfinite(variance) or variance <= 0:
        return None
    standard_error = float(np.sqrt(variance))
    statistic = beta / standard_error
    p_value = float(2 * t.sf(abs(statistic), df_resid))
    critical = float(t.ppf(0.975, df_resid))
    return {
        "effect": beta,
        "standardError": standard_error,
        "ciLow": beta - critical * standard_error,
        "ciHigh": beta + critical * standard_error,
        "statistic": statistic,
        "p": p_value,
        "dfResidual": df_resid,
    }


def compare_within_lineage(
    y: np.ndarray,
    x: np.ndarray,
    lineage: np.ndarray,
    *,
    min_mutant: int,
    min_wild_type: int,
) -> dict[str, float] | None:
    """Return unadjusted and lineage-fixed hotspot-mutant contrasts on the same models."""
    valid = np.isfinite(y) & np.isfinite(x) & pd.notna(lineage)
    if valid.sum() < min_mutant + min_wild_type:
        return None
    y_valid = y[valid].astype(float)
    x_valid = x[valid].astype(float)
    lineage_valid = lineage[valid].astype(str)

    available_mutant = int((x_valid > 0).sum())
    available_wild_type = int((x_valid == 0).sum())
    if available_mutant < min_mutant or available_wild_type < min_wild_type:
        return None

    codes, labels = pd.factorize(lineage_valid, sort=False)
    group_n = np.bincount(codes)
    group_mutant = np.bincount(codes, weights=x_valid)
    informative_group = (group_mutant > 0) & (group_mutant < group_n)
    informative = informative_group[codes]
    if not informative_group.any():
        return None

    y_analysis = y_valid[informative]
    x_analysis = x_valid[informative]
    lineage_analysis = lineage_valid[informative]
    n_mutant = int((x_analysis > 0).sum())
    n_wild_type = int((x_analysis == 0).sum())
    if n_mutant < min_mutant or n_wild_type < min_wild_type:
        return None

    unadjusted = _ols_hc3(y_analysis, x_analysis, np.repeat("all", len(y_analysis)))
    adjusted = _ols_hc3(y_analysis, x_analysis, lineage_analysis)
    if unadjusted is None or adjusted is None:
        return None

    result: dict[str, float] = {
        "nModelsAvailable": int(len(y_valid)),
        "nMutAvailable": available_mutant,
        "nHotspotNegativeAvailable": available_wild_type,
        "nModelsAdjusted": int(len(y_analysis)),
        "nMutAdjusted": n_mutant,
        "nHotspotNegativeAdjusted": n_wild_type,
        "nInformativeLineages": int(informative_group.sum()),
        "nLineagesAvailable": int(len(labels)),
    }
    result.update({f"unadjusted{key[0].upper()}{key[1:]}": value for key, value in unadjusted.items()})
    result.update({f"adjusted{key[0].upper()}{key[1:]}": value for key, value in adjusted.items()})
    return result


def addiction_analysis(
    panel: pd.DataFrame,
    model: pd.DataFrame,
    crispr: pd.DataFrame,
    crispr_columns: dict[int, str],
    hotspot: pd.DataFrame,
    hotspot_columns: dict[int, str],
) -> tuple[pd.DataFrame, list[int]]:
    """Fit same-gene CRISPR associations for recurrent hotspot genotypes."""
    symbol = panel.set_index("entrezGeneId")["hugoSymbol"].to_dict()
    role = panel.set_index("entrezGeneId")["roleInCancer"].to_dict()
    common_models = crispr.index.intersection(hotspot.index).intersection(model.index)
    lineage = model.loc[common_models, "OncotreeLineage"].to_numpy(object)

    testable: list[int] = []
    rows: list[dict[str, object]] = []
    for entrez in sorted(set(panel["entrezGeneId"]) & set(crispr_columns) & set(hotspot_columns)):
        hotspot_value = hotspot.loc[common_models, hotspot_columns[entrez]].to_numpy(float)
        mutation = np.where(np.isfinite(hotspot_value), hotspot_value > 0, np.nan).astype(float)
        if int(np.nansum(mutation)) < MIN_HOTSPOT_MODELS:
            continue
        testable.append(entrez)
        outcome = crispr.loc[common_models, crispr_columns[entrez]].to_numpy(float)
        fit = compare_within_lineage(
            outcome,
            mutation,
            lineage,
            min_mutant=MIN_MUT_CRISPR,
            min_wild_type=MIN_WT_CRISPR,
        )
        if fit is None:
            continue
        rows.append(
            {
                "entrezGeneId": entrez,
                "gene": symbol.get(entrez, str(entrez)),
                "roleInCancer": role.get(entrez, "Other"),
                **fit,
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("No evaluable lineage-adjusted CRISPR associations")
    result["adjustedFdr"] = multipletests(result["adjustedP"], method="fdr_bh")[1]
    result["unadjustedFdr"] = multipletests(result["unadjustedP"], method="fdr_bh")[1]
    result["direction"] = np.where(
        result["adjustedEffect"] < 0,
        "mutant models more dependent",
        "mutant models less dependent",
    )
    result["comparatorDefinition"] = "hotspot-negative (no hotspot call for tested gene)"
    result = result.sort_values(["adjustedP", "adjustedEffect"]).reset_index(drop=True)
    result.to_csv(TABLES / "depmap_addiction_lineage_adjusted.csv", index=False)
    return result, testable


def drug_analysis(
    testable: list[int],
    panel: pd.DataFrame,
    model: pd.DataFrame,
    hotspot: pd.DataFrame,
    hotspot_columns: dict[int, str],
) -> pd.DataFrame:
    """Screen PRISM compounds with the same within-lineage fixed-effect model."""
    auc = pd.read_csv(DM / "REPURPOSINGAUCMatrix.csv", index_col=0)
    compound_map = (
        pd.read_csv(
            DM / "REPURPOSINGResponseCurves.csv",
            usecols=["CompoundID", "CompoundName"],
        )
        .drop_duplicates("CompoundID")
        .set_index("CompoundID")["CompoundName"]
        .to_dict()
    )
    symbol = panel.set_index("entrezGeneId")["hugoSymbol"].to_dict()
    role = panel.set_index("entrezGeneId")["roleInCancer"].to_dict()
    common_models = auc.index.intersection(hotspot.index).intersection(model.index)
    auc = auc.loc[common_models]
    lineage = model.loc[common_models, "OncotreeLineage"].to_numpy(object)
    outcome_matrix = auc.to_numpy(float)

    rows: list[dict[str, object]] = []
    for number, entrez in enumerate(testable, start=1):
        if entrez not in hotspot_columns:
            continue
        hotspot_value = hotspot.loc[common_models, hotspot_columns[entrez]].to_numpy(float)
        mutation = np.where(np.isfinite(hotspot_value), hotspot_value > 0, np.nan).astype(float)
        gene = symbol.get(entrez, str(entrez))
        before = len(rows)
        for index, compound_id in enumerate(auc.columns):
            fit = compare_within_lineage(
                outcome_matrix[:, index],
                mutation,
                lineage,
                min_mutant=MIN_MUT_PRISM,
                min_wild_type=MIN_WT_PRISM,
            )
            if fit is None:
                continue
            rows.append(
                {
                    "entrezGeneId": entrez,
                    "gene": gene,
                    "roleInCancer": role.get(entrez, "Other"),
                    "compoundId": compound_id,
                    "compound": compound_map.get(compound_id, compound_id),
                    **fit,
                }
            )
        print(
            f"  PRISM {number:>2}/{len(testable)} {gene:<10} "
            f"{len(rows) - before:>4} evaluable compounds"
        )

    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("No evaluable lineage-adjusted PRISM associations")
    result["adjustedFdr"] = multipletests(result["adjustedP"], method="fdr_bh")[1]
    result["unadjustedFdr"] = multipletests(result["unadjustedP"], method="fdr_bh")[1]

    # AUC is lower for more sensitive models.  Retain the model coefficient and add a
    # reader-facing sensitisation scale where positive means mutant models had lower AUC.
    for prefix in ("adjusted", "unadjusted"):
        result[f"{prefix}Sensitisation"] = -result[f"{prefix}Effect"]
        result[f"{prefix}SensitisationCiLow"] = -result[f"{prefix}CiHigh"]
        result[f"{prefix}SensitisationCiHigh"] = -result[f"{prefix}CiLow"]
    result["direction"] = np.where(
        result["adjustedSensitisation"] > 0,
        "hotspot-mutant models more sensitive",
        "hotspot-mutant models less sensitive",
    )
    result["comparatorDefinition"] = "hotspot-negative (no hotspot call for tested gene)"
    result = result.sort_values(["adjustedP", "adjustedSensitisation"], ascending=[True, False])
    result.to_csv(TABLES / "depmap_drug_lineage_adjusted.csv", index=False)
    return result


def three_group_sensitivities(
    panel: pd.DataFrame,
    model: pd.DataFrame,
    crispr: pd.DataFrame,
    crispr_columns: dict[int, str],
    hotspot: pd.DataFrame,
    hotspot_columns: dict[int, str],
    damaging: pd.DataFrame,
    damaging_columns: dict[int, str],
    addiction: pd.DataFrame,
    drug: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Separate canonical hotspot, other-alteration and unaltered comparators.

    The complete CRISPR set is refitted.  PRISM sensitivity focuses on prespecified
    positive controls and the five separately labelled candidate contexts displayed in
    Figure 7, avoiding an additional post hoc genome-wide screen.
    """
    symbol_to_entrez = panel.set_index("hugoSymbol").entrezGeneId.to_dict()
    common_crispr = (
        crispr.index.intersection(hotspot.index).intersection(damaging.index).intersection(model.index)
    )
    lineage_crispr = model.loc[common_crispr, "OncotreeLineage"].to_numpy(object)
    crispr_rows: list[dict[str, object]] = []
    for record in addiction.itertuples(index=False):
        entrez = int(record.entrezGeneId)
        if entrez not in crispr_columns or entrez not in hotspot_columns or entrez not in damaging_columns:
            continue
        fit = compare_three_genotype_groups(
            crispr.loc[common_crispr, crispr_columns[entrez]].to_numpy(float),
            hotspot.loc[common_crispr, hotspot_columns[entrez]].to_numpy(float),
            damaging.loc[common_crispr, damaging_columns[entrez]].to_numpy(float),
            lineage_crispr,
            min_hotspot=MIN_MUT_CRISPR,
            min_other=5,
            min_unaltered=MIN_WT_CRISPR,
        )
        if fit is not None:
            crispr_rows.append(
                {
                    "analysisLayer": "CRISPR same-gene dependency",
                    "gene": record.gene,
                    "compoundId": "",
                    "compound": "",
                    **fit,
                }
            )
    crispr_result = pd.DataFrame(crispr_rows)
    if not crispr_result.empty:
        crispr_result["hotspotVsUnalteredFdr"] = multipletests(
            crispr_result.hotspotVsUnalteredP, method="fdr_bh"
        )[1]

    selected_keys = {
        (gene, compound.upper())
        for gene, compound in PRISM_POSITIVE_CONTROLS + PRISM_ADDITIONAL_SENSITIVITIES
    }
    selected_drug = drug.copy()
    selected_drug["_key"] = list(zip(selected_drug.gene, selected_drug.compound.str.upper()))
    selected_drug = (
        selected_drug[selected_drug._key.isin(selected_keys)]
        .sort_values(["_key", "adjustedStandardError"])
        .drop_duplicates("_key")
    )
    auc = pd.read_csv(DM / "REPURPOSINGAUCMatrix.csv", index_col=0)
    common_prism = (
        auc.index.intersection(hotspot.index).intersection(damaging.index).intersection(model.index)
    )
    lineage_prism = model.loc[common_prism, "OncotreeLineage"].to_numpy(object)
    prism_rows: list[dict[str, object]] = []
    for record in selected_drug.itertuples(index=False):
        entrez = int(symbol_to_entrez[record.gene])
        if (
            entrez not in hotspot_columns
            or entrez not in damaging_columns
            or record.compoundId not in auc.columns
        ):
            continue
        fit = compare_three_genotype_groups(
            auc.loc[common_prism, record.compoundId].to_numpy(float),
            hotspot.loc[common_prism, hotspot_columns[entrez]].to_numpy(float),
            damaging.loc[common_prism, damaging_columns[entrez]].to_numpy(float),
            lineage_prism,
            min_hotspot=MIN_MUT_PRISM,
            min_other=5,
            min_unaltered=MIN_WT_PRISM,
        )
        if fit is not None:
            fit["hotspotVsUnalteredSensitisation"] = -fit["hotspotVsUnalteredEffect"]
            fit["hotspotVsUnalteredSensitisationCiLow"] = -fit["hotspotVsUnalteredCiHigh"]
            fit["hotspotVsUnalteredSensitisationCiHigh"] = -fit["hotspotVsUnalteredCiLow"]
            prism_rows.append(
                {
                    "analysisLayer": "PRISM drug response",
                    "gene": record.gene,
                    "compoundId": record.compoundId,
                    "compound": record.compound,
                    **fit,
                }
            )
    prism_result = pd.DataFrame(prism_rows)
    if not prism_result.empty:
        prism_result["hotspotVsUnalteredFdr"] = multipletests(
            prism_result.hotspotVsUnalteredP, method="fdr_bh"
        )[1]

    combined = pd.concat([crispr_result, prism_result], ignore_index=True, sort=False)
    combined.to_csv(TABLES / "functional_three_group_comparator_sensitivity.csv", index=False)

    # Leave-one-lineage-out influence for the same focused set.  The source retains
    # every omitted lineage; the summary exposes the effect range and sign stability.
    loo_rows: list[dict[str, object]] = []
    for record in addiction.itertuples(index=False):
        entrez = int(record.entrezGeneId)
        if entrez not in crispr_columns or entrez not in hotspot_columns:
            continue
        outcome = crispr.loc[common_crispr, crispr_columns[entrez]].to_numpy(float)
        exposure_raw = hotspot.loc[common_crispr, hotspot_columns[entrez]].to_numpy(float)
        exposure = np.where(np.isfinite(exposure_raw), exposure_raw > 0, np.nan).astype(float)
        for omitted in sorted(pd.unique(lineage_crispr[pd.notna(lineage_crispr)])):
            keep = lineage_crispr.astype(str) != str(omitted)
            fit = compare_within_lineage(
                outcome[keep], exposure[keep], lineage_crispr[keep],
                min_mutant=MIN_MUT_CRISPR, min_wild_type=MIN_WT_CRISPR,
            )
            if fit is not None:
                loo_rows.append(
                    {
                        "analysisLayer": "CRISPR same-gene dependency",
                        "gene": record.gene,
                        "compound": "",
                        "omittedLineage": omitted,
                        "effect": fit["adjustedEffect"],
                        "ciLow": fit["adjustedCiLow"],
                        "ciHigh": fit["adjustedCiHigh"],
                        "p": fit["adjustedP"],
                        "nMutant": fit["nMutAdjusted"],
                        "nComparator": fit["nHotspotNegativeAdjusted"],
                    }
                )
    for record in selected_drug.itertuples(index=False):
        entrez = int(symbol_to_entrez[record.gene])
        if entrez not in hotspot_columns or record.compoundId not in auc.columns:
            continue
        outcome = auc.loc[common_prism, record.compoundId].to_numpy(float)
        exposure_raw = hotspot.loc[common_prism, hotspot_columns[entrez]].to_numpy(float)
        exposure = np.where(np.isfinite(exposure_raw), exposure_raw > 0, np.nan).astype(float)
        for omitted in sorted(pd.unique(lineage_prism[pd.notna(lineage_prism)])):
            keep = lineage_prism.astype(str) != str(omitted)
            fit = compare_within_lineage(
                outcome[keep], exposure[keep], lineage_prism[keep],
                min_mutant=MIN_MUT_PRISM, min_wild_type=MIN_WT_PRISM,
            )
            if fit is not None:
                loo_rows.append(
                    {
                        "analysisLayer": "PRISM drug response",
                        "gene": record.gene,
                        "compound": record.compound,
                        "omittedLineage": omitted,
                        "effect": -fit["adjustedEffect"],
                        "ciLow": -fit["adjustedCiHigh"],
                        "ciHigh": -fit["adjustedCiLow"],
                        "p": fit["adjustedP"],
                        "nMutant": fit["nMutAdjusted"],
                        "nComparator": fit["nHotspotNegativeAdjusted"],
                    }
                )
    loo = pd.DataFrame(loo_rows)
    if not loo.empty:
        loo["context"] = np.where(
            loo.compound.astype(str).eq(""), loo.gene, loo.gene + " — " + loo.compound
        )
        loo.to_csv(TABLES / "functional_leave_one_lineage_out.csv", index=False)
        loo_summary = (
            loo.groupby(["analysisLayer", "context"], sort=False)
            .agg(
                nOmittedLineages=("omittedLineage", "nunique"),
                minimumEffect=("effect", "min"),
                maximumEffect=("effect", "max"),
                medianEffect=("effect", "median"),
                allEffectsSameDirection=("effect", lambda values: bool((values > 0).all() or (values < 0).all())),
            )
            .reset_index()
        )
    else:
        loo_summary = pd.DataFrame()
    loo_summary.to_csv(TABLES / "functional_leave_one_lineage_out_summary.csv", index=False)
    return combined, loo, loo_summary


def control_table(drug: pd.DataFrame) -> pd.DataFrame:
    """Select the pre-specified positive controls present in the PRISM release."""
    order = {pair: index for index, pair in enumerate(PRISM_POSITIVE_CONTROLS)}
    control = drug.copy()
    control["controlKey"] = list(zip(control["gene"], control["compound"].str.upper()))
    control = control.loc[control["controlKey"].isin(order)].copy()
    # If a named compound occurs under multiple IDs, retain the most precise estimate.
    control["controlOrder"] = control["controlKey"].map(order)
    control = (
        control.sort_values(["controlOrder", "adjustedStandardError"])
        .drop_duplicates("controlKey")
        .sort_values("controlOrder")
    )
    control["pair"] = control["gene"] + " — " + control["compound"].str.title()
    return control.drop(columns="controlKey")


def make_figure(addiction: pd.DataFrame, drug: pd.DataFrame) -> None:
    """Render a 180-mm, four-panel functional-support figure."""
    nature_style.apply()
    SOURCE.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=nature_style.figsize(180, 154))
    ax_a, ax_b, ax_c, ax_d = axes.flat

    # a — adjusted CRISPR forest plot.
    supported = addiction.loc[addiction["adjustedEffect"] < 0].copy()
    significant = supported.loc[supported["adjustedFdr"] < 0.05]
    forest = (significant if len(significant) >= 8 else supported.nsmallest(12, "adjustedP"))
    forest = forest.nsmallest(12, "adjustedEffect").sort_values("adjustedEffect", ascending=False)
    forest.to_csv(SOURCE / "figure14_panel_a_crispr_forest.csv", index=False)
    ypos = np.arange(len(forest))
    is_significant = forest["adjustedFdr"].to_numpy() < 0.05
    colors = np.where(is_significant, COLORS["blue"], COLORS["grey"])
    ax_a.hlines(ypos, forest["adjustedCiLow"], forest["adjustedCiHigh"], color=colors, lw=1.1)
    ax_a.scatter(forest["adjustedEffect"], ypos, c=colors, s=22, zorder=3, edgecolor="white", lw=0.4)
    ax_a.axvline(0, color=COLORS["black"], lw=0.7, ls=(0, (2, 2)))
    ax_a.set_yticks(ypos, forest["gene"])
    ax_a.set_xlabel("Adjusted CRISPR effect (hotspot-mutant − hotspot-negative)")
    ax_a.set_title("Within-lineage mutant-selective dependency", loc="left", pad=5)
    ax_a.grid(axis="x", color=COLORS["very_light_grey"], lw=0.6)
    ax_a.text(
        0.01,
        0.02,
        "more dependent",
        transform=ax_a.transAxes,
        ha="left",
        va="bottom",
        color=COLORS["blue"],
        fontsize=6,
    )
    nature_style.panel_label(ax_a, "a")

    # b — show how lineage adjustment changes the full addiction scan.
    addiction.to_csv(SOURCE / "figure14_panel_b_crispr_adjustment.csv", index=False)
    x = addiction["unadjustedEffect"].to_numpy()
    y = addiction["adjustedEffect"].to_numpy()
    extent = float(np.nanmax(np.abs(np.r_[x, y]))) * 1.08
    ax_b.plot([-extent, extent], [-extent, extent], color=COLORS["light_grey"], lw=0.8, zorder=0)
    sig = addiction["adjustedFdr"] < 0.05
    ax_b.scatter(x[~sig], y[~sig], s=16, color=COLORS["grey"], alpha=0.65, edgecolor="none")
    ax_b.scatter(x[sig], y[sig], s=24, color=COLORS["orange"], alpha=0.9, edgecolor="white", lw=0.35)
    label_genes = set(addiction.loc[sig].nsmallest(7, "adjustedEffect")["gene"])
    label_genes.update({"KRAS", "BRAF", "PIK3CA", "NRAS"})
    for _, row in addiction.loc[addiction["gene"].isin(label_genes)].iterrows():
        ax_b.annotate(
            row["gene"],
            (row["unadjustedEffect"], row["adjustedEffect"]),
            xytext=(3, 2),
            textcoords="offset points",
            fontsize=5.5,
        )
    rho = spearmanr(x, y).statistic
    ax_b.text(0.03, 0.97, f"n = {len(addiction)} genes\nSpearman ρ = {rho:.2f}", transform=ax_b.transAxes, va="top")
    ax_b.set_xlim(-extent, extent)
    ax_b.set_ylim(-extent, extent)
    ax_b.set_xlabel("Unadjusted CRISPR effect")
    ax_b.set_ylabel("Lineage-adjusted CRISPR effect")
    ax_b.set_title("Lineage adjustment of dependency estimates", loc="left", pad=5)
    ax_b.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["orange"], markeredgecolor="none", label="adjusted FDR < 0.05"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["grey"], markeredgecolor="none", label="FDR ≥ 0.05"),
        ],
        loc="lower right",
        frameon=False,
    )
    nature_style.panel_label(ax_b, "b")

    # c — pre-specified drug/genotype controls, not post-hoc top hits.
    controls = control_table(drug)
    controls.to_csv(SOURCE / "figure14_panel_c_prism_controls.csv", index=False)
    controls_plot = controls.sort_values("adjustedSensitisation", ascending=True)
    ypos = np.arange(len(controls_plot))
    gene_colors = {
        "BRAF": COLORS["blue"],
        "PIK3CA": COLORS["orange"],
        "KRAS": COLORS["green"],
        "NRAS": COLORS["purple"],
    }
    point_colors = [gene_colors.get(g, COLORS["grey"]) for g in controls_plot["gene"]]
    ax_c.hlines(
        ypos,
        controls_plot["adjustedSensitisationCiLow"],
        controls_plot["adjustedSensitisationCiHigh"],
        color=point_colors,
        lw=1.0,
    )
    for index, (_, row) in enumerate(controls_plot.iterrows()):
        ax_c.scatter(
            row["adjustedSensitisation"],
            index,
            s=22,
            facecolor=gene_colors.get(row["gene"], COLORS["grey"]) if row["adjustedFdr"] < 0.05 else "white",
            edgecolor=gene_colors.get(row["gene"], COLORS["grey"]),
            lw=0.8,
            zorder=3,
        )
    ax_c.axvline(0, color=COLORS["black"], lw=0.7, ls=(0, (2, 2)))
    ax_c.set_yticks(ypos, controls_plot["pair"])
    ax_c.set_xlabel("Adjusted sensitisation (hotspot-negative AUC − mutant AUC)")
    ax_c.set_title("Pre-specified drug–genotype positive controls", loc="left", pad=5)
    ax_c.grid(axis="x", color=COLORS["very_light_grey"], lw=0.6)
    ax_c.text(
        0.99,
        0.02,
        "filled: FDR < 0.05",
        transform=ax_c.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.5,
        color=COLORS["grey"],
    )
    nature_style.panel_label(ax_c, "c")

    # d — all evaluable drug/genotype tests, with controls overlaid.
    panel_d = drug[
        ["gene", "compoundId", "compound", "unadjustedSensitisation",
         "adjustedSensitisation", "adjustedFdr", "comparatorDefinition"]
    ].copy()
    panel_d["positiveAtFdr05"] = (
        (panel_d["adjustedFdr"] < 0.05) & (panel_d["adjustedSensitisation"] > 0)
    )
    panel_d.to_csv(SOURCE / "figure14_panel_d_prism_adjustment.csv", index=False)
    dx = drug["unadjustedSensitisation"].to_numpy()
    dy = drug["adjustedSensitisation"].to_numpy()
    limit = float(np.nanquantile(np.abs(np.r_[dx, dy]), 0.995))
    limit = max(limit, 0.05)
    ax_d.hexbin(
        dx,
        dy,
        gridsize=48,
        extent=(-limit, limit, -limit, limit),
        mincnt=1,
        bins="log",
        cmap="Greys",
        linewidths=0,
    )
    ax_d.plot([-limit, limit], [-limit, limit], color=COLORS["sky"], lw=0.8, zorder=1)
    drug_rho = spearmanr(dx, dy).statistic
    supported_pairs = int(((drug["adjustedFdr"] < 0.05) & (drug["adjustedSensitisation"] > 0)).sum())
    ax_d.text(
        0.03,
        0.97,
        f"n = {len(drug):,} tests\nSpearman ρ = {drug_rho:.2f}\n{supported_pairs:,} positive associations at FDR < 0.05",
        transform=ax_d.transAxes,
        va="top",
    )
    ax_d.set_xlim(-limit, limit)
    ax_d.set_ylim(-limit, limit)
    ax_d.set_xlabel("Unadjusted PRISM sensitisation")
    ax_d.set_ylabel("Lineage-adjusted PRISM sensitisation")
    ax_d.set_title("Lineage adjustment across the PRISM screen", loc="left", pad=5)
    nature_style.panel_label(ax_d, "d")

    # The compound labels in panel c are long.  Reserve enough left margin within
    # the fixed 180-mm canvas so the first characters are not clipped at export.
    fig.subplots_adjust(left=0.16, right=0.99, bottom=0.08, top=0.96, wspace=0.40, hspace=0.42)
    nature_style.save_figure(fig, FIGURES / "figure14_functional_support")
    plt.close(fig)


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    SOURCE.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(PROCESSED / "gene_panel.csv")
    panel["entrezGeneId"] = pd.to_numeric(panel["entrezGeneId"], errors="coerce").astype("Int64")
    panel = panel.dropna(subset=["entrezGeneId"]).copy()
    panel["entrezGeneId"] = panel["entrezGeneId"].astype(int)
    model = (
        pd.read_csv(DM / "Model.csv", usecols=["ModelID", "OncotreeLineage"])
        .drop_duplicates("ModelID")
        .dropna(subset=["OncotreeLineage"])
        .set_index("ModelID")
    )
    crispr, crispr_columns = load_gene_matrix(DM / "CRISPRGeneEffect.csv")
    hotspot, hotspot_columns = load_default_hotspot()
    damaging, damaging_columns = load_default_damaging()

    print("Fitting lineage-adjusted CRISPR associations ...")
    addiction, testable = addiction_analysis(
        panel,
        model,
        crispr,
        crispr_columns,
        hotspot,
        hotspot_columns,
    )
    print(f"  {len(addiction)} evaluable genes from {len(testable)} recurrent hotspot genotypes")
    print("Fitting lineage-adjusted PRISM associations ...")
    drug = drug_analysis(testable, panel, model, hotspot, hotspot_columns)
    print("Fitting three-group comparator and leave-one-lineage-out sensitivities ...")
    comparator_sensitivity, lineage_loo, lineage_loo_summary = three_group_sensitivities(
        panel,
        model,
        crispr,
        crispr_columns,
        hotspot,
        hotspot_columns,
        damaging,
        damaging_columns,
        addiction,
        drug,
    )
    make_figure(addiction, drug)

    addiction_supported = addiction.loc[
        (addiction["adjustedFdr"] < 0.05) & (addiction["adjustedEffect"] < 0)
    ]
    drug_supported = drug.loc[
        (drug["adjustedFdr"] < 0.05) & (drug["adjustedSensitisation"] > 0)
    ]
    controls = control_table(drug)
    print("\nLineage-adjusted CRISPR functional-support associations (FDR < 0.05):")
    print(
        addiction_supported[
            [
                "gene",
                "nMutAdjusted",
                "nHotspotNegativeAdjusted",
                "nInformativeLineages",
                "adjustedEffect",
                "adjustedCiLow",
                "adjustedCiHigh",
                "adjustedFdr",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4g}")
    )
    print(
        f"\nPRISM: {len(drug):,} evaluable gene–compound tests; "
        f"{len(drug_supported):,} positive associations at FDR < 0.05."
    )
    print("Pre-specified PRISM positive controls:")
    print(
        controls[
            [
                "gene",
                "compound",
                "nMutAdjusted",
                "nHotspotNegativeAdjusted",
                "nInformativeLineages",
                "adjustedSensitisation",
                "adjustedFdr",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4g}")
    )
    print(
        f"\nComparator sensitivity: {len(comparator_sensitivity):,} three-group models; "
        f"{len(lineage_loo):,} leave-one-lineage-out estimates across "
        f"{len(lineage_loo_summary):,} contexts."
    )
    print("\nWrote lineage-adjusted tables, source data, and Figure 14 (PDF/SVG/PNG).")


if __name__ == "__main__":
    main()
