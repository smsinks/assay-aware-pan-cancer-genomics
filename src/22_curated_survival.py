"""Stage 22 -- cancer-specific and pan-cancer overall-survival analysis.

The historical survival stages pooled cancers, de-duplicated generic patient identifiers
globally and encoded cancer type as an arbitrary continuous integer.  This replacement
uses the biospecimen-curated one-sample-per-patient-key cohort, namespaces every clinical
record by study, requires both genes to be callable in the selected tumour sample, and
fits models within cancer group with study-specific baseline hazards.

Two model families are reported for prespecified gene-pair/cancer contexts:

* four-group model (A−/B− reference, A only, B only, A+B);
* mutation main effects plus A-by-B interaction.

The primary analysis requires a finite, strictly positive overall-survival time and uses
Cox models with study-specific baseline hazards and Efron's method for tied event
times. Conventional model-based
uncertainty is primary because several cancer-specific contexts contain fewer than ten
contributing studies. A study-clustered sandwich sensitivity is added when at least ten
studies contribute, using R ``survival::coxph`` so tied event times are handled by the
same Efron partial likelihood. Additional sensitivity models reintroduce zero-time
records at a fixed half-day value and adjust complete cases for standardised age and
recorded sex. Pan-cancer models use study-by-cancer baseline strata and are explicitly
interpreted as composition-weighted common associations rather than universal effects.

Outputs
-------
data/processed/clinical_survival_curated.parquet
results/tables/survival_curated_pair_models.csv
results/tables/survival_curated_pair_groups.csv
results/tables/survival_curated_missingness.csv
results/tables/survival_curated_cohort_audit.csv
results/tables/survival_curated_stratum_audit.csv
results/tables/survival_endpoint_eligibility.csv
results/tables/survival_model_sensitivity_summary.csv
results/tables/survival_joint_state_and_interaction_summary.csv
results/tables/survival_ph_diagnostics.csv
results/tables/survival_scaled_schoenfeld_residuals.csv
results/tables/survival_time_varying_hazard_ratios.csv
results/tables/survival_piecewise_hazard_ratios.csv
results/tables/survival_rmst_differences.csv
results/tables/survival_primary_tumour_sensitivity.csv
results/tables/survival_assay_discordance_exclusion_sensitivity.csv
results/tables/survival_assay_discordance_specimen_audit.csv
results/tables/survival_study_specific_hazard_ratios.csv
results/tables/survival_study_meta_analysis.csv
results/tables/survival_leave_one_study_out.csv
results/tables/survival_leave_one_study_out_summary.csv
results/tables/survival_extended_diagnostic_audit.csv
results/tables/survival_software_environment.csv
results/source_data/figure9_*.csv
results/figures/figure9_survival.{pdf,svg,png}
results/figures/supplementary/figureS4_survival_diagnostics.{pdf,svg,png}
"""
from __future__ import annotations

import argparse
import importlib
import subprocess
import tempfile
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

import cbioportal_client as cb
from callability import partition_callable_mutations
from config import FIGURES, PROCESSED, TABLES
from nature_style import (
    COLORS,
    aligned_panel_labels,
    apply as apply_style,
    figsize,
    figure_panel_label,
    panel_label,
    save_figure,
)

warnings.filterwarnings("ignore")

CONTEXTS = [
    ("LUAD", "KEAP1", "STK11"),
    ("LUSC", "KEAP1", "STK11"),
    ("PAAD", "KRAS", "TP53"),
    ("COADREAD", "KRAS", "TP53"),
    ("UCEC", "PTEN", "PIK3CA"),
    # The heterogeneous broad BRAIN category includes medulloblastoma,
    # meningioma and other non-diffuse CNS tumours.  Keep glioblastoma
    # separate rather than recreating the former pooled glioma endpoint.
    ("GBM", "IDH1", "TP53"),
    ("BRCA", "PIK3CA", "TP53"),
    # Additional cross-layer contexts are drawn from the prespecified
    # conditioned-interaction reference set. Each has adequate positive-time
    # follow-up and representation of the joint-mutant genotype for the
    # four-group survival model.
    ("LUAD", "EGFR", "KRAS"),
    ("LUAD", "KRAS", "TP53"),
    ("LUAD", "STK11", "TP53"),
    ("COADREAD", "BRAF", "KRAS"),
    ("SKCM", "BRAF", "NRAS"),
    ("UCEC", "PIK3CA", "PIK3R1"),
    ("UCEC", "PTEN", "TP53"),
    ("UCEC", "CTNNB1", "TP53"),
    ("MDS", "SF3B1", "SRSF2"),
]

# Unique pairs represented by the cancer-specific contexts. These models include
# every eligible cancer group and use study-by-cancer baseline hazards.
PANCAN_CONTEXTS = list(dict.fromkeys((gene_a, gene_b) for _, gene_a, gene_b in CONTEXTS))

# Updated Figure 9 contexts are selected from the complete, outcome-independent
# mutation-frequency-gated survival screen.  They combine the strongest newly
# exposed cancer-specific associations with the two leading informative pan-cancer
# results used in the time-resolved display.
FIGURE9_LATEST_CONTEXTS = [
    ("cancer-specific", "LUAD", "KEAP1", "KRAS", "LUAD · KEAP1–KRAS"),
    ("cancer-specific", "MDS", "ASXL1", "RUNX1", "MDS · ASXL1–RUNX1"),
    ("cancer-specific", "PAAD", "CDKN2A", "KRAS", "PAAD · CDKN2A–KRAS"),
    ("cancer-specific", "UCEC", "ARID1A", "PTEN", "UCEC · ARID1A–PTEN"),
    ("cancer-specific", "PAAD", "KRAS", "SMAD4", "PAAD · KRAS–SMAD4"),
    ("cancer-specific", "MDS", "ASXL1", "SRSF2", "MDS · ASXL1–SRSF2"),
    ("pan-cancer", "PAN-CANCER", "KRAS", "TP53", "Pan-cancer · KRAS–TP53"),
    ("pan-cancer", "PAN-CANCER", "KMT2C", "TP53", "Pan-cancer · KMT2C–TP53"),
]

# A complementary established lung-cancer context is used for the second
# Kaplan–Meier display without altering the eight-context time-resolved set.
FIGURE9_SECONDARY_KM_CONTEXT = (
    "cancer-specific", "LUAD", "KEAP1", "STK11", "LUAD · KEAP1–STK11"
)

GROUP_ORDER = ["A−/B−", "A only", "B only", "A+B"]
GROUP_COLORS = {
    "A−/B−": COLORS["grey"],
    "A only": COLORS["blue"],
    "B only": COLORS["orange"],
    "A+B": COLORS["vermillion"],
}

MIN_CONTEXT_N = 80
MIN_CONTEXT_EVENTS = 20
MIN_GROUP_N = 8
MIN_STUDY_N = 10
MIN_STUDY_EVENTS = 2
MIN_CLUSTER_STUDIES = 10
ZERO_TIME_EPSILON_MONTHS = 0.5 / 30.4375  # half a day, expressed in portal OS months
DISPLAY_HORIZON_MONTHS = 120.0

# Outcome-independent discovery thresholds.  Cancer-specific models require both
# genes to reach 10% assay-aware prevalence in the corresponding cancer family;
# pan-cancer models require 5% prevalence across the full cohort.  Survival data are
# used only for model-adequacy checks after this mutation-frequency gate has been
# applied.  Requiring 20 patients in every four-state genotype group limits sparse
# separation without selecting contexts on the observed survival association.
SURVIVAL_SCREEN_CANCER_MIN_PREVALENCE = 0.10
SURVIVAL_SCREEN_PANCANCER_MIN_PREVALENCE = 0.05
SURVIVAL_SCREEN_MIN_GROUP_N = 20

EXTENDED_SURVIVAL_TABLES = (
    "survival_ph_diagnostics.csv",
    "survival_scaled_schoenfeld_residuals.csv",
    "survival_time_varying_hazard_ratios.csv",
    "survival_piecewise_hazard_ratios.csv",
    "survival_rmst_differences.csv",
    "survival_primary_tumour_sensitivity.csv",
    "survival_assay_discordance_exclusion_sensitivity.csv",
    "survival_study_specific_hazard_ratios.csv",
    "survival_study_meta_analysis.csv",
    "survival_leave_one_study_out.csv",
    "survival_extended_diagnostic_audit.csv",
    "survival_software_environment.csv",
)

LEGEND_BOX = {
    "frameon": True,
    "fancybox": True,
    "framealpha": 0.96,
    "facecolor": "#FAFAFA",
    "edgecolor": COLORS["light_grey"],
    "borderpad": 0.45,
}


def parse_event(value) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().upper()
    if text.startswith("1") or any(word in text for word in ("DECEASED", "DEAD")):
        return 1.0
    if text.startswith("0") or any(word in text for word in ("LIVING", "ALIVE")):
        return 0.0
    return np.nan


def parse_sex(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    if text in {"F", "FEMALE", "WOMAN"}:
        return "Female"
    if text in {"M", "MALE", "MAN"}:
        return "Male"
    return None


def build_curated_clinical(samples: pd.DataFrame) -> pd.DataFrame:
    """Pivot cached patient clinical rows without global patient-ID de-duplication."""
    cache = PROCESSED / "clinical_survival_curated.parquet"
    if cache.exists():
        out = pd.read_parquet(cache)
        # Cancer taxonomy can be revised without changing the cached portal
        # clinical values. Always refresh the selected-sample annotations from
        # the current authoritative cohort.
        current_annotations = samples[
            [
                "studyId",
                "sampleId",
                "broadCancerCode",
                "analysisCancerCode",
                "sampleTypeGroup",
            ]
        ].copy()
        out = out.drop(
            columns=["broadCancerCode", "analysisCancerCode", "sampleTypeGroup"],
            errors="ignore",
        ).merge(
            current_annotations,
            on=["studyId", "sampleId"],
            how="inner",
            validate="one_to_one",
        )
    else:
        wanted = {"OS_STATUS", "OS_MONTHS", "AGE", "SEX"}
        study_ids = sorted(samples.studyId.unique())
        frames: list[pd.DataFrame] = []
        for study_id in tqdm(study_ids, desc="curated patient clinical"):
            try:
                raw = pd.DataFrame(cb.get_clinical_data(study_id))
            except RuntimeError:
                continue
            if raw.empty or "clinicalAttributeId" not in raw or "patientId" not in raw:
                continue
            raw = raw.loc[raw.clinicalAttributeId.isin(wanted)].copy()
            if raw.empty:
                continue
            wide = raw.pivot_table(
                index="patientId",
                columns="clinicalAttributeId",
                values="value",
                aggfunc="first",
            ).reset_index()
            wide["studyId"] = study_id
            frames.append(wide)

        if not frames:
            raise RuntimeError("No cached patient-level clinical data were available")
        clinical = pd.concat(frames, ignore_index=True)
        if clinical.duplicated(["studyId", "patientId"]).any():
            raise AssertionError("Patient clinical rows are not unique within study")

        selected = samples[
            [
                "studyId",
                "sampleId",
                "patientId",
                "patientKey",
                "broadCancerCode",
                "analysisCancerCode",
                "sampleTypeGroup",
            ]
        ].copy()
        selected = selected.dropna(subset=["patientId"])
        selected["patientId"] = selected.patientId.astype(str)
        clinical["patientId"] = clinical.patientId.astype(str)
        out = selected.merge(
            clinical,
            on=["studyId", "patientId"],
            how="left",
            validate="one_to_one",
        )
        out["months"] = pd.to_numeric(out.get("OS_MONTHS"), errors="coerce")
        out["event"] = out.get("OS_STATUS", pd.Series(index=out.index, dtype=object)).map(parse_event)
        out["age"] = pd.to_numeric(out.get("AGE"), errors="coerce")
        out["sex"] = out.get("SEX", pd.Series(index=out.index, dtype=object)).map(parse_sex)

    finite_endpoint = out.months.notna() & out.event.notna() & np.isfinite(out.months)
    out["validOsNonnegative"] = finite_endpoint & out.months.ge(0)
    out["validPositiveOs"] = finite_endpoint & out.months.gt(0)
    out["zeroOsTime"] = finite_endpoint & out.months.eq(0)
    out["negativeOsTime"] = finite_endpoint & out.months.lt(0)
    # Keep the historical column name, but make its revised meaning explicit:
    # primary survival inference now requires strictly positive follow-up.
    out["validOs"] = out.validPositiveOs
    out.to_parquet(cache, index=False)
    return out


def callability_and_mutation_flags(
    samples: pd.DataFrame,
    genes: list[str],
) -> pd.DataFrame:
    panel = pd.read_csv(PROCESSED / "gene_panel.csv")
    sym2ent = panel.drop_duplicates("hugoSymbol").set_index("hugoSymbol").entrezGeneId.to_dict()
    missing = sorted(set(genes) - set(sym2ent))
    if missing:
        raise KeyError(f"Prespecified survival genes absent from panel: {missing}")

    assay = pd.read_parquet(PROCESSED / "sample_assay.parquet")
    info = samples[["studyId", "sampleId", "patientKey"]].merge(
        assay,
        on=["studyId", "sampleId"],
        how="left",
        validate="one_to_one",
    )
    membership = pd.read_parquet(PROCESSED / "panel_gene_membership.parquet")
    membership = membership[membership.entrezGeneId.isin([sym2ent[g] for g in genes])]
    member = set(zip(membership.genePanelId.astype(str), membership.entrezGeneId.astype(int)))
    is_wes = info.assayType.str.startswith("WES/WGS")

    mutations = pd.read_parquet(PROCESSED / "mutations_curated.parquet")
    mutations = mutations[mutations.entrezGeneId.isin([sym2ent[g] for g in genes])]
    mutated = set(
        zip(mutations.sampleId.astype(str), mutations.entrezGeneId.astype(int))
    )
    for gene in genes:
        entrez = int(sym2ent[gene])
        info[f"callable_{gene}"] = [
            bool(wes or (str(panel_id), entrez) in member)
            for wes, panel_id in zip(is_wes, info.genePanelId)
        ]
        info[f"mut_{gene}"] = [
            int((str(sample_id), entrez) in mutated)
            for sample_id in info.sampleId
        ]
        if (info[f"mut_{gene}"].eq(1) & ~info[f"callable_{gene}"]).any():
            raise AssertionError(f"A retained {gene} mutation is outside documented callability")
    return info


def survival_screen_specifications() -> tuple[pd.DataFrame, list[str]]:
    """Define the complete survival screen without consulting survival outcomes.

    The candidate-gene universe is identical to the 123-gene mutation-association
    screen: the 120 highest recurrence-by-CMC-evidence scores plus the three
    prespecified controls that were not already in the top 120.  Cancer-specific
    pairs are inherited from the complete jointly assay-covered mutation screen and
    filtered only on each gene's assay-aware prevalence in that cancer.  Pan-cancer
    pairs are all combinations of candidate genes that independently pass the
    pan-cancer prevalence threshold.
    """
    interaction = importlib.import_module("17_curated_interactions")
    panel = pd.read_csv(PROCESSED / "gene_panel.csv")
    candidate_entrez, entrez_to_symbol = interaction.candidate_genes(panel)
    candidate_genes = sorted(entrez_to_symbol[int(gene)] for gene in candidate_entrez)
    if len(candidate_genes) != 123:
        raise AssertionError(
            f"The survival discovery universe should contain 123 genes; found {len(candidate_genes)}"
        )

    pairwise = pd.read_csv(
        TABLES / "cooccurrence_curated_jointly_profiled.csv",
        usecols=["cancer", "geneA", "geneB", "full_n", "full_nA", "full_nB"],
    ).drop_duplicates(["cancer", "geneA", "geneB"])
    pairwise["geneAFrequency"] = pairwise.full_nA / pairwise.full_n
    pairwise["geneBFrequency"] = pairwise.full_nB / pairwise.full_n
    cancer_specific = pairwise.loc[
        pairwise.geneAFrequency.ge(SURVIVAL_SCREEN_CANCER_MIN_PREVALENCE)
        & pairwise.geneBFrequency.ge(SURVIVAL_SCREEN_CANCER_MIN_PREVALENCE)
    ].copy()
    cancer_specific.insert(0, "scope", "cancer-specific")

    prevalence = pd.read_csv(TABLES / "gene_frequencies_curated.csv")
    prevalence = prevalence.loc[prevalence.gene.isin(candidate_genes)].copy()
    prevalence["frequency"] = prevalence.freqPct / 100
    pan_genes = sorted(
        prevalence.loc[
            prevalence.frequency.ge(SURVIVAL_SCREEN_PANCANCER_MIN_PREVALENCE), "gene"
        ].astype(str)
    )
    pan_frequency = prevalence.set_index("gene").frequency.to_dict()
    pan_rows = []
    for index, gene_a in enumerate(pan_genes):
        for gene_b in pan_genes[index + 1 :]:
            pan_rows.append(
                {
                    "scope": "pan-cancer",
                    "cancer": "PAN-CANCER",
                    "geneA": gene_a,
                    "geneB": gene_b,
                    "full_n": len(panel),
                    "full_nA": np.nan,
                    "full_nB": np.nan,
                    "geneAFrequency": float(pan_frequency[gene_a]),
                    "geneBFrequency": float(pan_frequency[gene_b]),
                }
            )
    specifications = pd.concat(
        [
            cancer_specific[
                [
                    "scope", "cancer", "geneA", "geneB", "full_n", "full_nA",
                    "full_nB", "geneAFrequency", "geneBFrequency",
                ]
            ],
            pd.DataFrame(pan_rows),
        ],
        ignore_index=True,
    )
    specifications["frequencyThreshold"] = np.where(
        specifications.scope.eq("cancer-specific"),
        SURVIVAL_SCREEN_CANCER_MIN_PREVALENCE,
        SURVIVAL_SCREEN_PANCANCER_MIN_PREVALENCE,
    )
    specifications["selectionBasis"] = (
        "outcome-independent assay-aware mutation-frequency threshold"
    )
    return specifications, candidate_genes


def _screen_context_eligibility(
    analysis: pd.DataFrame,
    specification: pd.Series,
) -> tuple[dict[str, object], bool]:
    """Apply only endpoint-availability and model-adequacy filters to one context."""
    scope = str(specification.scope)
    cancer = str(specification.cancer)
    gene_a = str(specification.geneA)
    gene_b = str(specification.geneB)
    if scope == "cancer-specific":
        data = analysis.loc[analysis.broadCancerCode.eq(cancer)].copy()
        data["stratumId"] = data.studyId.astype(str)
    else:
        data = analysis.copy()
        data["stratumId"] = data.studyId.astype(str) + "::" + data.broadCancerCode.astype(str)
    data = data.loc[
        data[f"callable_{gene_a}"].eq(True)
        & data[f"callable_{gene_b}"].eq(True)
    ].copy()
    data["genotypeCode"] = (
        2 * data[f"mut_{gene_a}"].astype(int) + data[f"mut_{gene_b}"].astype(int)
    )
    strata = (
        data.groupby("stratumId", observed=True)
        .agg(
            nPatients=("event", "size"),
            nEvents=("event", "sum"),
            nGenotypeGroups=("genotypeCode", "nunique"),
        )
    )
    retained = set(
        strata.index[
            strata.nPatients.ge(MIN_STUDY_N)
            & strata.nEvents.ge(MIN_STUDY_EVENTS)
            & strata.nGenotypeGroups.ge(2)
        ].astype(str)
    )
    data = data.loc[data.stratumId.astype(str).isin(retained)]
    group_counts = data.genotypeCode.value_counts().reindex(range(4), fill_value=0)
    eligible = bool(
        len(data) >= MIN_CONTEXT_N
        and data.event.sum() >= MIN_CONTEXT_EVENTS
        and group_counts.min() >= SURVIVAL_SCREEN_MIN_GROUP_N
    )
    audit = {
        "scope": scope,
        "cancer": cancer,
        "geneA": gene_a,
        "geneB": gene_b,
        "geneAFrequency": float(specification.geneAFrequency),
        "geneBFrequency": float(specification.geneBFrequency),
        "frequencyThreshold": float(specification.frequencyThreshold),
        "nPatients": int(len(data)),
        "nEvents": int(data.event.sum()),
        "nDoubleNegative": int(group_counts.loc[0]),
        "nBOnly": int(group_counts.loc[1]),
        "nAOnly": int(group_counts.loc[2]),
        "nDoubleMutant": int(group_counts.loc[3]),
        "nStudies": int(data.studyId.nunique()),
        "nStrata": int(data.stratumId.nunique()),
        "eligible": eligible,
        "eligibilityRule": (
            f"both genes meet mutation-frequency threshold; n>={MIN_CONTEXT_N}; "
            f"events>={MIN_CONTEXT_EVENTS}; every genotype group>={SURVIVAL_SCREEN_MIN_GROUP_N}; "
            f"retained stratum n>={MIN_STUDY_N}, events>={MIN_STUDY_EVENTS} and >=2 genotype groups"
        ),
    }
    return audit, eligible


def run_unbiased_survival_screen(
    clinical: pd.DataFrame,
    flags: pd.DataFrame,
    specifications: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit every frequency-eligible cancer-specific and pan-cancer pair."""
    analysis = clinical.loc[clinical.validPositiveOs].merge(
        flags,
        on=["studyId", "sampleId"],
        how="inner",
        validate="one_to_one",
    )
    audit_rows: list[dict[str, object]] = []
    eligible_rows: list[dict[str, object]] = []
    for specification in tqdm(
        list(specifications.itertuples(index=False)),
        desc="outcome-independent survival eligibility",
    ):
        record = pd.Series(specification._asdict())
        audit, eligible = _screen_context_eligibility(analysis, record)
        audit_rows.append(audit)
        if eligible:
            eligible_rows.append(audit)
    audit = pd.DataFrame(audit_rows)
    eligible = pd.DataFrame(eligible_rows)
    if eligible.empty:
        raise RuntimeError("No frequency-defined survival context met model-adequacy criteria")

    keep_columns = [
        "studyId", "sampleId", "broadCancerCode", "months", "event",
    ] + [column for column in flags.columns if column.startswith(("callable_", "mut_"))]
    screen_data = analysis[keep_columns].copy()
    r_script = Path(__file__).with_name("survival_unbiased_screen.R")
    with tempfile.TemporaryDirectory(prefix="cancer_survival_screen_") as tmp_dir:
        tmp = Path(tmp_dir)
        data_path = tmp / "screen_data.csv"
        spec_path = tmp / "screen_specs.csv"
        out_path = tmp / "screen_results.csv"
        screen_data.to_csv(data_path, index=False)
        eligible.to_csv(spec_path, index=False)
        completed = subprocess.run(
            ["Rscript", str(r_script), str(data_path), str(spec_path), str(out_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Unbiased R survival screen failed:\n"
                + completed.stdout
                + "\n"
                + completed.stderr
            )
        results = pd.read_csv(out_path)

    results["globalFdr"] = np.nan
    results["withinCancerFdr"] = np.nan
    for (scope, contrast), group in results.groupby(["scope", "contrast"], observed=True):
        indexes = group.index[group.p.notna() & np.isfinite(group.p)]
        if len(indexes):
            results.loc[indexes, "globalFdr"] = multipletests(
                results.loc[indexes, "p"], method="fdr_bh"
            )[1]
    cancer_mask = results.scope.eq("cancer-specific")
    for (cancer, contrast), group in results.loc[cancer_mask].groupby(
        ["cancer", "contrast"], observed=True
    ):
        indexes = group.index[group.p.notna() & np.isfinite(group.p)]
        if len(indexes):
            results.loc[indexes, "withinCancerFdr"] = multipletests(
                results.loc[indexes, "p"], method="fdr_bh"
            )[1]
    results["fdrFamily"] = np.where(
        results.scope.eq("cancer-specific"),
        results.contrast + "; all cancer-specific frequency-eligible contexts",
        results.contrast + "; all pan-cancer frequency-eligible contexts",
    )
    results = results.merge(
        eligible[
            [
                "scope", "cancer", "geneA", "geneB", "geneAFrequency",
                "geneBFrequency", "frequencyThreshold", "eligibilityRule",
            ]
        ],
        on=["scope", "cancer", "geneA", "geneB"],
        how="left",
        validate="many_to_one",
    )
    audit.to_csv(TABLES / "survival_unbiased_screen_audit.csv", index=False)
    results.to_csv(TABLES / "survival_unbiased_screen.csv", index=False)
    return results, audit


def assay_discordance_specimen_audit(
    samples: pd.DataFrame,
) -> tuple[set[tuple[str, str]], pd.DataFrame]:
    """Identify selected specimens carrying any off-panel metadata conflict.

    The sensitivity excludes affected specimens in their entirety. It never
    promotes positive-only, off-panel mutation records into an assay-covered
    denominator and therefore leaves the strict primary definition unchanged.
    """
    mutations = pd.read_parquet(PROCESSED / "mutations_dedup.parquet")
    _, conflicts = partition_callable_mutations(mutations)
    selected_keys = set(
        zip(samples.studyId.astype(str), samples.sampleId.astype(str))
    )
    conflict_pairs = conflicts[["studyId", "sampleId"]].astype(str)
    in_selected = [
        (study_id, sample_id) in selected_keys
        for study_id, sample_id in conflict_pairs.itertuples(index=False, name=None)
    ]
    selected_conflicts = conflicts.loc[in_selected].copy()
    selected_conflicts["studyId"] = selected_conflicts.studyId.astype(str)
    selected_conflicts["sampleId"] = selected_conflicts.sampleId.astype(str)
    conflict_keys = set(
        selected_conflicts[["studyId", "sampleId"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if len(conflict_keys) != 1_832:
        raise AssertionError(
            "Frozen selected cohort should contain 1,832 specimens with at least "
            f"one assay-scope conflict; observed {len(conflict_keys):,}"
        )
    study_counts = (
        selected_conflicts.groupby("studyId", observed=True)
        .agg(
            nConflictSpecimens=("sampleId", "nunique"),
            nConflictMutationRecords=("sampleId", "size"),
            nConflictGenes=("entrezGeneId", "nunique"),
        )
        .reset_index()
        .sort_values(["nConflictSpecimens", "studyId"], ascending=[False, True])
    )
    sarcoma_count = int(
        study_counts.loc[
            study_counts.studyId.eq("sarcoma_msk_2022"), "nConflictSpecimens"
        ].iloc[0]
    )
    if sarcoma_count != 1_635:
        raise AssertionError(
            "Frozen sarcoma_msk_2022 conflict-specimen count should be 1,635; "
            f"observed {sarcoma_count:,}"
        )
    overall = pd.DataFrame(
        [
            {
                "recordLevel": "All selected specimens",
                "studyId": "ALL",
                "nConflictSpecimens": len(conflict_keys),
                "nConflictMutationRecords": len(selected_conflicts),
                "nConflictGenes": selected_conflicts.entrezGeneId.nunique(),
                "pctConflictSpecimens": 100 * len(conflict_keys) / len(samples),
                "sensitivityDefinition": (
                    "exclude every selected specimen with at least one observed mutation "
                    "outside its documented assay scope"
                ),
            }
        ]
    )
    study_counts.insert(0, "recordLevel", "Study")
    study_counts["pctConflictSpecimens"] = np.nan
    study_counts["sensitivityDefinition"] = overall.sensitivityDefinition.iloc[0]
    audit = pd.concat([overall, study_counts], ignore_index=True)
    return conflict_keys, audit


def context_data(
    clinical: pd.DataFrame,
    flags: pd.DataFrame,
    cancer: str,
    gene_a: str,
    gene_b: str,
    scope: str = "cancer-specific",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], pd.DataFrame]:
    if scope == "cancer-specific":
        base = clinical.loc[clinical.broadCancerCode.eq(cancer)].copy()
        context = f"{gene_a}–{gene_b} ({cancer})"
        baseline_hazards = "separate baseline hazard for each contributing study"
    elif scope == "pan-cancer":
        base = clinical.copy()
        cancer = "PAN-CANCER"
        context = f"{gene_a}–{gene_b} (pan-cancer)"
        baseline_hazards = "separate baseline hazard for each study-by-cancer stratum"
    else:
        raise ValueError(f"Unknown survival scope: {scope}")

    data = base.merge(
        flags[
            [
                "studyId",
                "sampleId",
                f"callable_{gene_a}",
                f"callable_{gene_b}",
                f"mut_{gene_a}",
                f"mut_{gene_b}",
            ]
        ],
        on=["studyId", "sampleId"],
        how="left",
        validate="one_to_one",
    )
    audit = {
        "context": context,
        "scope": scope,
        "cancer": cancer,
        "geneA": gene_a,
        "geneB": gene_b,
        "baselineHazards": baseline_hazards,
        "nSelectedScopeCases": int(len(data)),
        "nWithPatientId": int(data.patientId.notna().sum()),
        "nWithFiniteOsStatusAndTime": int(
            (data.months.notna() & data.event.notna() & np.isfinite(data.months)).sum()
        ),
        "nWithNonnegativeOs": int(data.validOsNonnegative.sum()),
        "nWithPositiveOs": int(data.validPositiveOs.sum()),
        "nZeroOsExcludedPrimary": int(data.zeroOsTime.sum()),
        "nNegativeOsExcluded": int(data.negativeOsTime.sum()),
        "nJointlyCallable": int(
            (data[f"callable_{gene_a}"].eq(True) & data[f"callable_{gene_b}"].eq(True)).sum()
        ),
        "nPositiveOsJointlyCallable": int(
            (
                data.validPositiveOs
                & data[f"callable_{gene_a}"].eq(True)
                & data[f"callable_{gene_b}"].eq(True)
            ).sum()
        ),
        "nZeroOsJointlyCallable": int(
            (
                data.zeroOsTime
                & data[f"callable_{gene_a}"].eq(True)
                & data[f"callable_{gene_b}"].eq(True)
            ).sum()
        ),
    }

    data = data.loc[
        data.validOsNonnegative
        & data[f"callable_{gene_a}"].eq(True)
        & data[f"callable_{gene_b}"].eq(True)
    ].copy()
    data["mutA"] = data[f"mut_{gene_a}"].astype(int)
    data["mutB"] = data[f"mut_{gene_b}"].astype(int)
    data["interaction"] = data.mutA * data.mutB
    data["group"] = np.select(
        [
            data.mutA.eq(0) & data.mutB.eq(0),
            data.mutA.eq(1) & data.mutB.eq(0),
            data.mutA.eq(0) & data.mutB.eq(1),
            data.mutA.eq(1) & data.mutB.eq(1),
        ],
        GROUP_ORDER,
        default="unknown",
    )
    if scope == "cancer-specific":
        data["stratumId"] = data.studyId.astype(str)
    else:
        data["stratumId"] = (
            data.studyId.astype(str) + "::" + data.broadCancerCode.astype(str)
        )

    positive = data.loc[data.validPositiveOs].copy()
    stratum_summary = (
        data.groupby("stratumId", observed=True)
        .agg(
            studyId=("studyId", "first"),
            cancerGroup=("broadCancerCode", "first"),
            nNonnegativeTime=("sampleId", "size"),
            nPositiveTime=("validPositiveOs", "sum"),
            nZeroTime=("zeroOsTime", "sum"),
            nZeroTimeEvents=("event", lambda values: int(((data.loc[values.index, "zeroOsTime"]) & values.eq(1)).sum())),
        )
        .reset_index()
    )
    positive_summary = (
        positive.groupby("stratumId", observed=True)
        .agg(
            nPositiveEvents=("event", "sum"),
            nGenotypeGroupsPositive=("group", "nunique"),
        )
        .reset_index()
    )
    stratum_summary = stratum_summary.merge(
        positive_summary, on="stratumId", how="left", validate="one_to_one"
    )
    stratum_summary[["nPositiveEvents", "nGenotypeGroupsPositive"]] = stratum_summary[
        ["nPositiveEvents", "nGenotypeGroupsPositive"]
    ].fillna(0)
    stratum_summary["retainedPrimary"] = (
        stratum_summary.nPositiveTime.ge(MIN_STUDY_N)
        & stratum_summary.nPositiveEvents.ge(MIN_STUDY_EVENTS)
        & stratum_summary.nGenotypeGroupsPositive.ge(2)
    )
    stratum_summary.insert(0, "context", context)
    stratum_summary.insert(1, "scope", scope)
    stratum_summary.insert(2, "cancer", cancer)
    stratum_summary.insert(3, "geneA", gene_a)
    stratum_summary.insert(4, "geneB", gene_b)

    retained_strata = set(
        stratum_summary.loc[stratum_summary.retainedPrimary, "stratumId"].astype(str)
    )
    audit["nStudiesBeforeInformativeFilter"] = int(data.studyId.nunique())
    audit["nStrataBeforeInformativeFilter"] = int(data.stratumId.nunique())
    primary = positive.loc[positive.stratumId.astype(str).isin(retained_strata)].copy()
    zero_sensitivity = data.loc[data.stratumId.astype(str).isin(retained_strata)].copy()
    zero_sensitivity["zeroTimeOriginal"] = zero_sensitivity.zeroOsTime.astype(bool)
    zero_sensitivity.loc[zero_sensitivity.zeroTimeOriginal, "months"] = ZERO_TIME_EPSILON_MONTHS

    audit["nStudiesRetained"] = int(primary.studyId.nunique())
    audit["nStrataRetained"] = int(primary.stratumId.nunique())
    audit["nCancerGroupsRetained"] = int(primary.broadCancerCode.nunique())
    audit["nModelPatients"] = int(len(primary))
    audit["nModelEvents"] = int(primary.event.sum())
    audit["medianOsMonths"] = _median_os(primary)
    audit["medianOsStatus"] = (
        "estimated" if np.isfinite(audit["medianOsMonths"]) else "not reached"
    )
    audit["nZeroTimesReincludedSensitivity"] = int(zero_sensitivity.zeroTimeOriginal.sum())
    audit["nZeroTimeEventsReincludedSensitivity"] = int(
        (zero_sensitivity.zeroTimeOriginal & zero_sensitivity.event.eq(1)).sum()
    )
    audit["zeroTimeSensitivityValueMonths"] = ZERO_TIME_EPSILON_MONTHS

    event_time_counts = primary.loc[primary.event.eq(1)].groupby("months").size()
    audit["nDistinctEventTimes"] = int(len(event_time_counts))
    audit["nEventsAtTiedTimes"] = int(event_time_counts[event_time_counts.gt(1)].sum())
    audit["pctEventsAtTiedTimes"] = (
        100 * audit["nEventsAtTiedTimes"] / max(audit["nModelEvents"], 1)
    )
    audit["maxEventsAtSingleTime"] = int(event_time_counts.max()) if len(event_time_counts) else 0
    audit["clusterSensitivityEligible"] = bool(
        audit["nStudiesRetained"] >= MIN_CLUSTER_STUDIES
    )
    return primary, zero_sensitivity, audit, stratum_summary


def _median_os(data: pd.DataFrame) -> float:
    if data.empty:
        return np.nan
    km = KaplanMeierFitter().fit(data.months, data.event)
    median = float(km.median_survival_time_)
    return median if np.isfinite(median) else np.nan


def summarize_context(
    data: pd.DataFrame,
    scope: str,
    cancer: str,
    gene_a: str,
    gene_b: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    context = f"{gene_a}–{gene_b} ({cancer})" if scope == "cancer-specific" else f"{gene_a}–{gene_b} (pan-cancer)"
    group_rows: list[dict[str, object]] = []
    for group_name in GROUP_ORDER:
        group = data.loc[data.group.eq(group_name)]
        median = _median_os(group)
        group_rows.append(
            {
                "context": context,
                "scope": scope,
                "cancer": cancer,
                "geneA": gene_a,
                "geneB": gene_b,
                "group": group_name,
                "nPatients": int(len(group)),
                "nEvents": int(group.event.sum()),
                "medianOsMonths": median,
                "medianOsStatus": "estimated" if np.isfinite(median) else "not reached",
            }
        )
    group_counts = pd.DataFrame(group_rows)

    missingness = {
        "context": context,
        "scope": scope,
        "cancer": cancer,
        "geneA": gene_a,
        "geneB": gene_b,
        "nPrimaryModel": int(len(data)),
        "nEventsPrimary": int(data.event.sum()),
        "nAgeAvailable": int(data.age.notna().sum()),
        "nSexAvailable": int(data.sex.notna().sum()),
        "nAgeSexComplete": int((data.age.notna() & data.sex.notna()).sum()),
        "ageStandardization": "z score within the age/sex complete-case analysis population",
        "sexCoding": "Female=1, Male=0; Male reference; unrecognised or missing values excluded",
    }
    missingness["pctAgeMissing"] = 100 * (1 - missingness["nAgeAvailable"] / max(len(data), 1))
    missingness["pctSexMissing"] = 100 * (1 - missingness["nSexAvailable"] / max(len(data), 1))
    return group_counts, missingness


def _prepare_dataset(data: pd.DataFrame, dataset_id: str, age_sex: bool = False) -> pd.DataFrame:
    work = data.copy()
    work["A_only"] = ((work.mutA == 1) & (work.mutB == 0)).astype(int)
    work["B_only"] = ((work.mutA == 0) & (work.mutB == 1)).astype(int)
    work["Both"] = ((work.mutA == 1) & (work.mutB == 1)).astype(int)
    work["female"] = work.sex.map({"Male": 0.0, "Female": 1.0})
    if age_sex:
        work = work.dropna(subset=["age", "female"]).copy()
        age_mean = work.age.mean()
        age_sd = work.age.std()
        work["age_z"] = (
            (work.age - age_mean) / age_sd
            if np.isfinite(age_sd) and age_sd > 0
            else np.nan
        )
    else:
        work["age_z"] = np.nan
    work["datasetId"] = dataset_id
    keep = [
        "datasetId", "sampleId", "patientKey", "studyId", "stratumId",
        "broadCancerCode", "sampleTypeGroup", "hasAssayScopeConflict", "months", "event", "mutA", "mutB", "interaction",
        "A_only", "B_only", "Both", "age_z", "female",
    ]
    return work[keep]


def build_model_inputs(
    primary: pd.DataFrame,
    zero_sensitivity: pd.DataFrame,
    scope: str,
    cancer: str,
    gene_a: str,
    gene_b: str,
) -> tuple[list[pd.DataFrame], list[dict[str, object]]]:
    context = f"{gene_a}–{gene_b} ({cancer})" if scope == "cancer-specific" else f"{gene_a}–{gene_b} (pan-cancer)"
    token = f"{scope}_{cancer}_{gene_a}_{gene_b}".replace(" ", "_").replace("/", "_")
    baseline = "study" if scope == "cancer-specific" else "study × cancer group"
    datasets = {
        "positive": _prepare_dataset(primary, f"{token}_positive"),
        "age_sex": _prepare_dataset(primary, f"{token}_age_sex", age_sex=True),
        "zero": _prepare_dataset(zero_sensitivity, f"{token}_zero"),
    }
    rows: list[pd.DataFrame] = list(datasets.values())
    specs: list[dict[str, object]] = []

    def add_spec(
        dataset_key: str,
        model: str,
        parameterization: str,
        variance: str,
        population: str,
    ) -> None:
        specs.append(
            {
                "datasetId": datasets[dataset_key].datasetId.iloc[0],
                "context": context,
                "scope": scope,
                "cancer": cancer,
                "geneA": gene_a,
                "geneB": gene_b,
                "analysisPopulation": population,
                "model": model,
                "parameterization": parameterization,
                "varianceEstimator": variance,
                "clusterThreshold": MIN_CLUSTER_STUDIES,
                "baselineHazards": baseline,
            }
        )

    positive_population = "strictly positive OS time"
    add_spec("positive", "four-group primary", "four-group", "model-based", positive_population)
    add_spec("positive", "interaction primary", "interaction", "model-based", positive_population)
    add_spec(
        "positive", "four-group study-clustered sensitivity", "four-group",
        "study-clustered sandwich", positive_population,
    )
    add_spec(
        "positive", "interaction study-clustered sensitivity", "interaction",
        "study-clustered sandwich", positive_population,
    )

    complete_population = "strictly positive OS time; age/sex complete-case"
    add_spec(
        "age_sex", "four-group age/sex complete-case", "four-group",
        "model-based", complete_population,
    )
    add_spec(
        "age_sex", "interaction age/sex complete-case", "interaction",
        "model-based", complete_population,
    )
    add_spec(
        "age_sex", "four-group age/sex study-clustered sensitivity", "four-group",
        "study-clustered sandwich", complete_population,
    )

    zero_population = "nonnegative OS time; reported zero replaced by half a day"
    add_spec(
        "zero", "four-group zero-time sensitivity", "four-group",
        "model-based", zero_population,
    )
    add_spec(
        "zero", "four-group zero-time study-clustered sensitivity", "four-group",
        "study-clustered sandwich", zero_population,
    )
    return rows, specs


def run_cox_models(
    datasets: list[pd.DataFrame],
    specs: list[dict[str, object]],
) -> pd.DataFrame:
    data = pd.concat(datasets, ignore_index=True)
    spec_frame = pd.DataFrame(specs)
    r_script = Path(__file__).with_name("survival_cox_models.R")
    with tempfile.TemporaryDirectory(prefix="cancer_survival_") as tmp_dir:
        tmp = Path(tmp_dir)
        data_path = tmp / "analysis_data.csv"
        spec_path = tmp / "model_specs.csv"
        out_path = tmp / "cox_results.csv"
        data.to_csv(data_path, index=False)
        spec_frame.to_csv(spec_path, index=False)
        completed = subprocess.run(
            ["Rscript", str(r_script), str(data_path), str(spec_path), str(out_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "R survival model fitting failed:\n"
                + completed.stdout
                + "\n"
                + completed.stderr
            )
        models = pd.read_csv(out_path)

    models["endpoint"] = (
        "Overall survival; duration and vital status as reported by contributing studies"
    )
    models["timeUnit"] = "months"
    models["timeOriginMetadata"] = (
        "Time origin, stage, grade, treatment and metastatic status follow contributing-study metadata and definitions"
    )
    models["fdr"] = np.nan
    models["fdrFamily"] = ""
    for (scope, model), group in models.groupby(["scope", "model"], observed=True):
        target = "Both" if group.parameterization.eq("four-group").all() else "interaction"
        mask = group.term.eq(target) & group.p.notna() & group.fitStatus.str.startswith("estimated")
        if not mask.any():
            continue
        indexes = group.index[mask]
        models.loc[indexes, "fdr"] = multipletests(models.loc[indexes, "p"], method="fdr_bh")[1]
        models.loc[indexes, "fdrFamily"] = f"{scope}; {model}; {target}"
    return models


def _add_bh_fdr(
    frame: pd.DataFrame,
    family_columns: list[str],
    *,
    p_column: str = "p",
) -> pd.DataFrame:
    """Add Benjamini-Hochberg FDRs within explicitly labelled result families."""
    out = frame.copy()
    out["fdr"] = np.nan
    out["fdrFamily"] = ""
    if p_column not in out:
        return out
    for family, group in out.groupby(family_columns, observed=True, dropna=False):
        indexes = group.index[group[p_column].notna() & np.isfinite(group[p_column])]
        if not len(indexes):
            continue
        out.loc[indexes, "fdr"] = multipletests(
            out.loc[indexes, p_column], method="fdr_bh"
        )[1]
        family_values = family if isinstance(family, tuple) else (family,)
        out.loc[indexes, "fdrFamily"] = "; ".join(
            f"{column}={value}"
            for column, value in zip(family_columns, family_values)
        )
    return out


def joint_state_and_interaction_summary(models: pd.DataFrame) -> pd.DataFrame:
    """Keep the joint-state contrast distinct from formal multiplicative interaction."""
    identity = ["context", "scope", "cancer", "geneA", "geneB"]
    joint_columns = identity + [
        "nPatients", "nEvents", "nStudies", "nStrata",
        "coefficient", "standardError", "hazardRatio", "ciLow", "ciHigh",
        "p", "fdr", "phTestP", "fitStatus",
    ]
    interaction_columns = identity + [
        "coefficient", "standardError", "hazardRatio", "ciLow", "ciHigh",
        "p", "fdr", "phTestP", "fitStatus",
    ]
    joint = models.loc[
        models.model.eq("four-group primary") & models.term.eq("Both"),
        joint_columns,
    ].copy()
    interaction = models.loc[
        models.model.eq("interaction primary") & models.term.eq("interaction"),
        interaction_columns,
    ].copy()
    joint = joint.rename(
        columns={
            column: f"jointState{column[0].upper()}{column[1:]}"
            for column in joint_columns
            if column not in identity + ["nPatients", "nEvents", "nStudies", "nStrata"]
        }
    )
    interaction = interaction.rename(
        columns={
            column: f"multiplicativeInteraction{column[0].upper()}{column[1:]}"
            for column in interaction_columns
            if column not in identity
        }
    )
    summary = joint.merge(
        interaction,
        on=identity,
        how="outer",
        validate="one_to_one",
    )
    summary.insert(
        len(identity),
        "jointStateContrast",
        "A+B versus A−/B− in a saturated four-genotype-state Cox model",
    )
    summary.insert(
        len(identity) + 1,
        "multiplicativeInteractionContrast",
        "A×B product term conditional on mutation A and mutation B main effects",
    )
    return summary.sort_values(["scope", "context"]).reset_index(drop=True)


def run_extended_survival_diagnostics(
    datasets: list[pd.DataFrame],
    specs: list[dict[str, object]],
    models: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Run PH, time-varying, piecewise, RMST and study-heterogeneity analyses."""
    spec_frame = pd.DataFrame(specs)
    primary_specs = spec_frame.loc[
        spec_frame.model.eq("four-group primary")
    ].copy()
    if len(primary_specs) != 29 or primary_specs.datasetId.nunique() != 29:
        raise AssertionError(
            "Extended survival diagnostics require exactly 29 unique primary models"
        )
    primary_ids = set(primary_specs.datasetId)
    primary_datasets = [
        frame for frame in datasets if str(frame.datasetId.iloc[0]) in primary_ids
    ]
    if len(primary_datasets) != 29:
        raise AssertionError(
            f"Expected 29 primary model datasets, found {len(primary_datasets)}"
        )
    analysis_data = pd.concat(primary_datasets, ignore_index=True)
    if analysis_data.duplicated(["datasetId", "sampleId"]).any():
        raise AssertionError("Primary survival diagnostic datasets contain duplicate samples")

    r_script = Path(__file__).with_name("survival_extended_diagnostics.R")
    with tempfile.TemporaryDirectory(prefix="cancer_survival_extended_") as tmp_dir:
        tmp = Path(tmp_dir)
        data_path = tmp / "primary_data.csv"
        spec_path = tmp / "primary_specs.csv"
        output_dir = tmp / "outputs"
        analysis_data.to_csv(data_path, index=False)
        primary_specs.to_csv(spec_path, index=False)
        completed = subprocess.run(
            ["Rscript", str(r_script), str(data_path), str(spec_path), str(output_dir)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Extended R survival diagnostics failed:\n"
                + completed.stdout
                + "\n"
                + completed.stderr
            )
        outputs = {
            filename: pd.read_csv(output_dir / filename)
            for filename in EXTENDED_SURVIVAL_TABLES
        }

    expected_rows = {
        "survival_ph_diagnostics.csv": 116,  # 3 coefficients + GLOBAL × 29
        "survival_time_varying_hazard_ratios.csv": 2_900,  # 100-point smooth × 29
        "survival_piecewise_hazard_ratios.csv": 87,  # 3 intervals × 29
        "survival_rmst_differences.csv": 58,  # 2 horizons × 29
        "survival_primary_tumour_sensitivity.csv": 29,
        "survival_assay_discordance_exclusion_sensitivity.csv": 29,
        "survival_extended_diagnostic_audit.csv": 29,
    }
    for filename, expected in expected_rows.items():
        observed = len(outputs[filename])
        if observed != expected:
            raise AssertionError(
                f"{filename} contains {observed:,} rows; expected {expected:,}"
            )
    for filename in (
        "survival_scaled_schoenfeld_residuals.csv",
        "survival_study_specific_hazard_ratios.csv",
    ):
        contexts = outputs[filename].context.nunique()
        if contexts != 29:
            raise AssertionError(
                f"{filename} covers {contexts} primary contexts rather than 29"
            )
    if len(outputs["survival_study_meta_analysis.csv"]) != 28:
        raise AssertionError(
            "Expected 28 study-level meta-analyses; the three-study MDS context "
            "has insufficient direct-contrast study estimates"
        )

    # Repeated primary fits must exactly reproduce the authoritative 29 A+B
    # estimates and rank-transformed cox.zph P values.
    audit = outputs["survival_extended_diagnostic_audit.csv"]
    ph = outputs["survival_ph_diagnostics.csv"]
    if len(audit) != 29 or audit.context.nunique() != 29:
        raise AssertionError("Extended diagnostic audit does not contain all 29 contexts")
    ph_both = ph.loc[ph.term.eq("Both")].copy()
    ph_global = ph.loc[ph.term.eq("GLOBAL")].copy()
    if len(ph_both) != 29 or len(ph_global) != 29:
        raise AssertionError("cox.zph diagnostics are incomplete for the 29 primary models")
    authoritative = models.loc[
        models.model.eq("four-group primary") & models.term.eq("Both"),
        ["context", "hazardRatio", "phTestP"],
    ].copy()
    verification = authoritative.merge(
        audit[["context", "primaryModelHazardRatio"]],
        on="context",
        validate="one_to_one",
    ).merge(
        ph_both[["context", "phTestP"]].rename(columns={"phTestP": "extendedPhTestP"}),
        on="context",
        validate="one_to_one",
    )
    hr_difference = (
        verification.hazardRatio - verification.primaryModelHazardRatio
    ).abs().max()
    ph_difference = (
        verification.phTestP - verification.extendedPhTestP
    ).abs().max()
    if hr_difference > 1e-10 or ph_difference > 1e-10:
        raise AssertionError(
            "Repeated primary survival diagnostics disagree with the frozen estimates: "
            f"max HR difference={hr_difference}, max PH-P difference={ph_difference}"
        )

    # The PH tests are diagnostics rather than confirmatory endpoints, but a
    # global multiplicity column makes their interpretation transparent.
    ph["phFdr"] = np.nan
    both_indexes = ph.index[ph.term.eq("Both") & ph.phTestP.notna()]
    ph.loc[both_indexes, "phFdr"] = multipletests(
        ph.loc[both_indexes, "phTestP"], method="fdr_bh"
    )[1]
    ph["phFdrFamily"] = np.where(
        ph.term.eq("Both"),
        "all 29 primary A+B coefficient-level PH tests",
        "",
    )
    outputs["survival_ph_diagnostics.csv"] = ph
    outputs["survival_piecewise_hazard_ratios.csv"] = _add_bh_fdr(
        outputs["survival_piecewise_hazard_ratios.csv"], ["scope", "interval"]
    )
    outputs["survival_rmst_differences.csv"] = _add_bh_fdr(
        outputs["survival_rmst_differences.csv"], ["scope", "horizonMonths"]
    )
    outputs["survival_primary_tumour_sensitivity.csv"] = _add_bh_fdr(
        outputs["survival_primary_tumour_sensitivity.csv"], ["scope"]
    )
    outputs["survival_assay_discordance_exclusion_sensitivity.csv"] = _add_bh_fdr(
        outputs["survival_assay_discordance_exclusion_sensitivity.csv"], ["scope"]
    )
    outputs["survival_study_meta_analysis.csv"] = _add_bh_fdr(
        outputs["survival_study_meta_analysis.csv"],
        ["scope"],
        p_column="randomP",
    )

    for filename, frame in outputs.items():
        frame.to_csv(TABLES / filename, index=False)

    loo = outputs["survival_leave_one_study_out.csv"].copy()
    loo_summary_rows: list[dict[str, object]] = []
    for (context, method), group in loo.groupby(
        ["context", "leaveOneOutMethod"], observed=True
    ):
        finite = group.loc[group.hazardRatio.notna() & np.isfinite(group.hazardRatio)]
        if finite.empty:
            continue
        full_hr = float(
            authoritative.loc[authoritative.context.eq(context), "hazardRatio"].iloc[0]
        )
        loo_summary_rows.append(
            {
                "context": context,
                "scope": group.scope.iloc[0],
                "cancer": group.cancer.iloc[0],
                "geneA": group.geneA.iloc[0],
                "geneB": group.geneB.iloc[0],
                "leaveOneOutMethod": method,
                "nLeaveOneOutEstimates": int(len(finite)),
                "fullPrimaryHazardRatio": full_hr,
                "minimumLeaveOneOutHazardRatio": float(finite.hazardRatio.min()),
                "maximumLeaveOneOutHazardRatio": float(finite.hazardRatio.max()),
                "directionStableRelativeToOne": bool(
                    ((finite.hazardRatio > 1) == (full_hr > 1)).all()
                ),
                "minimumLeaveOneOutP": float(finite.p.min()),
                "maximumLeaveOneOutP": float(finite.p.max()),
            }
        )
    loo_summary = pd.DataFrame(loo_summary_rows).sort_values(
        ["scope", "context", "leaveOneOutMethod"]
    )
    exact_loo = loo_summary.loc[
        loo_summary.leaveOneOutMethod.eq(
            "exact pooled stratified four-group Cox refit"
        )
    ]
    if len(exact_loo) != 5 or not exact_loo.directionStableRelativeToOne.all():
        raise AssertionError(
            "Exact pooled leave-one-study-out results must cover five principal "
            "cancer-specific contexts with stable effect direction"
        )
    loo_summary.to_csv(
        TABLES / "survival_leave_one_study_out_summary.csv", index=False
    )
    outputs["survival_leave_one_study_out_summary.csv"] = loo_summary

    joint_interaction = joint_state_and_interaction_summary(models)
    if len(joint_interaction) != 29:
        raise AssertionError("Joint-state/interaction summary does not contain 29 contexts")
    joint_interaction.to_csv(
        TABLES / "survival_joint_state_and_interaction_summary.csv", index=False
    )
    outputs["survival_joint_state_and_interaction_summary.csv"] = joint_interaction
    return outputs


def sensitivity_summary(models: pd.DataFrame) -> pd.DataFrame:
    target = models.loc[
        models.term.eq("Both")
        & models.model.isin(
            [
                "four-group primary",
                "four-group study-clustered sensitivity",
                "four-group zero-time sensitivity",
                "four-group zero-time study-clustered sensitivity",
                "four-group age/sex complete-case",
            ]
        )
    ].copy()
    columns = [
        "context", "scope", "cancer", "geneA", "geneB", "model",
        "varianceEstimator", "adjustmentTerms", "nPatients", "nEvents", "nStudies", "nStrata",
        "coefficient", "standardError", "hazardRatio", "ciLow", "ciHigh", "p",
        "fdr", "fitStatus",
    ]
    return target[columns].sort_values(["scope", "context", "model"]).reset_index(drop=True)


def km_curve_source(
    data: pd.DataFrame,
    context: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    curves: list[pd.DataFrame] = []
    risk_times = np.array([0, 12, 24, 36, 60, 84, 120], dtype=float)
    risk_rows: list[dict[str, object]] = []
    median_rows: list[dict[str, object]] = []
    for group_name in GROUP_ORDER:
        group = data[data.group.eq(group_name)]
        if len(group) == 0:
            continue
        km = KaplanMeierFitter(label=group_name).fit(group.months, group.event)
        ci = km.confidence_interval_.copy()
        curve = km.survival_function_.join(ci).reset_index()
        curve.columns = ["months", "survival", "ciLow", "ciHigh"]
        curve.insert(0, "group", group_name)
        curve.insert(0, "context", context)
        curve["displayHorizonMonths"] = DISPLAY_HORIZON_MONTHS
        curves.append(curve)
        median = float(km.median_survival_time_)
        median_rows.append(
            {
                "context": context,
                "group": group_name,
                "nPatients": int(len(group)),
                "nEvents": int(group.event.sum()),
                "medianOsMonths": median if np.isfinite(median) else np.nan,
                "medianOsStatus": "estimated" if np.isfinite(median) else "not reached",
                "displayHorizonMonths": DISPLAY_HORIZON_MONTHS,
            }
        )
        for time in risk_times:
            risk_rows.append(
                {
                    "context": context,
                    "group": group_name,
                    "months": time,
                    "nAtRisk": int((group.months >= time).sum()),
                    "displayHorizonMonths": DISPLAY_HORIZON_MONTHS,
                }
            )
    return (
        pd.concat(curves, ignore_index=True),
        pd.DataFrame(risk_rows),
        pd.DataFrame(median_rows),
    )


def draw_km(
    ax: plt.Axes,
    risk_ax: plt.Axes,
    data: pd.DataFrame,
    gene_a: str,
    gene_b: str,
    letter: str,
    display_cancer: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cancer = display_cancer or str(data.broadCancerCode.iloc[0])
    context = f"{gene_a}–{gene_b} ({cancer})"
    display_label = {
        "A−/B−": "A−/B−",
        "A only": gene_a,
        "B only": gene_b,
        "A+B": "A+B",
    }
    ticks = np.array([0, 12, 24, 36, 60, 84, 120], dtype=float)
    for group_name in GROUP_ORDER:
        group = data[data.group.eq(group_name)]
        if len(group) < MIN_GROUP_N:
            continue
        km = KaplanMeierFitter(label=group_name).fit(group.months, group.event)
        median = float(km.median_survival_time_)
        median_text = f"{median:.1f} mo" if np.isfinite(median) else "NR"
        km.plot_survival_function(
            ax=ax,
            ci_show=True,
            color=GROUP_COLORS[group_name],
            lw=1.15,
            censor_styles={"marker": "+", "ms": 3.0, "mew": 0.6},
            label=f"{display_label[group_name]} · {median_text}",
        )
    valid_groups = data.group.value_counts()
    if (valid_groups >= MIN_GROUP_N).sum() >= 2:
        logrank = multivariate_logrank_test(data.months, data.group, data.event)
        p_text = f"four-group log-rank P={logrank.p_value:.2g}"
        logrank_summary = pd.DataFrame(
            [
                {
                    "context": context,
                    "testLabel": "four-group log-rank test",
                    "analysisN": len(data),
                    "analysisEvents": int(data.event.sum()),
                    "nGroups": int((valid_groups >= MIN_GROUP_N).sum()),
                    "testStatistic": float(logrank.test_statistic),
                    "logRankP": float(logrank.p_value),
                    "displayPText": p_text,
                    "testStatus": "estimated",
                }
            ]
        )
    else:
        p_text = "four-group log-rank not estimable"
        logrank_summary = pd.DataFrame(
            [
                {
                    "context": context,
                    "testLabel": "four-group log-rank test",
                    "analysisN": len(data),
                    "analysisEvents": int(data.event.sum()),
                    "nGroups": int((valid_groups >= MIN_GROUP_N).sum()),
                    "testStatistic": np.nan,
                    "logRankP": np.nan,
                    "displayPText": p_text,
                    "testStatus": "not estimable",
                }
            ]
        )
    ax.text(0.02, 0.03, p_text, transform=ax.transAxes, fontsize=4.25)
    ax.set_xlim(0, DISPLAY_HORIZON_MONTHS)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(ticks)
    ax.set_xlabel("Overall survival (months)", fontsize=4.8)
    ax.set_ylabel("Survival\nprobability", fontsize=4.8)
    ax.tick_params(axis="both", labelsize=4.1)
    km_legend_box = dict(LEGEND_BOX)
    km_legend_box["borderpad"] = 0.28
    ax.legend(
        **km_legend_box,
        fontsize=4.05,
        loc="upper right",
        handlelength=1.35,
        labelspacing=0.25,
    )
    ax.set_title(f"{cancer}: {gene_a}–{gene_b}", loc="left", fontsize=5.8, pad=3.0)
    risk_ax.axis("off")
    risk_ax.set_xlim(ax.get_xlim())
    risk_ax.text(-0.12, 0.92, "No. at risk", transform=risk_ax.transAxes, ha="right", va="top", fontsize=4.2)
    shown_groups = [g for g in GROUP_ORDER if len(data[data.group.eq(g)]) >= MIN_GROUP_N]
    for row_index, group_name in enumerate(shown_groups):
        group = data[data.group.eq(group_name)]
        # Keep the first risk row clear of the KM x-axis tick labels above it.
        y = 0.68 - row_index * 0.19
        risk_ax.text(
            -0.12,
            y,
            display_label[group_name],
            transform=risk_ax.transAxes,
            ha="right",
            va="center",
            fontsize=4.0,
            color=GROUP_COLORS[group_name],
        )
        for time in ticks:
            risk_ax.text(
                time,
                y,
                f"{int((group.months >= time).sum())}",
                ha="center",
                va="center",
                fontsize=4.0,
                color=GROUP_COLORS[group_name],
            )
    risk_ax.text(
        0.5,
        -0.08,
        "Overall survival (months)",
        transform=risk_ax.transAxes,
        ha="center",
        va="top",
        fontsize=4.5,
    )
    risk_ax.set_ylim(0, 1)
    # A shared risk-table axis suppresses the upper axis labels by default;
    # restore the KM time labels explicitly.
    ax.tick_params(axis="x", labelbottom=True)
    curves, risks, medians = km_curve_source(data, context)
    return curves, risks, logrank_summary, medians


def run_figure9_latest_diagnostics(
    clinical: pd.DataFrame,
    flags: pd.DataFrame,
) -> tuple[
    dict[tuple[str, str, str], pd.DataFrame],
    dict[str, pd.DataFrame],
]:
    """Fit the time-resolved Figure 9 diagnostics for the latest screen hits.

    This targeted figure stage does not replace the complete screen or the 29-context
    supplementary diagnostic archive. It refits the eight outcome-independent
    time-resolved contexts selected from that screen, and also prepares the established
    LUAD KEAP1–STK11 context used for the second Kaplan–Meier display.
    """
    contexts: dict[tuple[str, str, str], pd.DataFrame] = {}
    primary_datasets: list[pd.DataFrame] = []
    primary_specs: list[dict[str, object]] = []
    for scope, cancer, gene_a, gene_b, _ in FIGURE9_LATEST_CONTEXTS:
        data, zero_data, _, _ = context_data(
            clinical,
            flags,
            cancer,
            gene_a,
            gene_b,
            scope=scope,
        )
        if len(data) < MIN_CONTEXT_N or data.event.sum() < MIN_CONTEXT_EVENTS:
            raise AssertionError(
                f"Latest Figure 9 context is no longer estimable: {cancer} {gene_a}–{gene_b}"
            )
        contexts[(cancer, gene_a, gene_b)] = data
        datasets, specs = build_model_inputs(
            data, zero_data, scope, cancer, gene_a, gene_b
        )
        primary_spec = next(spec for spec in specs if spec["model"] == "four-group primary")
        primary_dataset = next(
            frame
            for frame in datasets
            if str(frame.datasetId.iloc[0]) == str(primary_spec["datasetId"])
        )
        primary_specs.append(primary_spec)
        primary_datasets.append(primary_dataset)

    km_scope, km_cancer, km_gene_a, km_gene_b, _ = FIGURE9_SECONDARY_KM_CONTEXT
    km_data, _, _, _ = context_data(
        clinical,
        flags,
        km_cancer,
        km_gene_a,
        km_gene_b,
        scope=km_scope,
    )
    if len(km_data) < MIN_CONTEXT_N or km_data.event.sum() < MIN_CONTEXT_EVENTS:
        raise AssertionError(
            "Figure 9f LUAD KEAP1–STK11 context is no longer estimable"
        )
    contexts[(km_cancer, km_gene_a, km_gene_b)] = km_data

    analysis_data = pd.concat(primary_datasets, ignore_index=True)
    spec_frame = pd.DataFrame(primary_specs)
    if len(spec_frame) != 8 or analysis_data.datasetId.nunique() != 8:
        raise AssertionError("Latest Figure 9 diagnostics require eight unique contexts")

    r_script = Path(__file__).with_name("survival_extended_diagnostics.R")
    with tempfile.TemporaryDirectory(prefix="figure9_latest_survival_") as tmp_dir:
        tmp = Path(tmp_dir)
        data_path = tmp / "figure9_data.csv"
        spec_path = tmp / "figure9_specs.csv"
        output_dir = tmp / "outputs"
        analysis_data.to_csv(data_path, index=False)
        spec_frame.to_csv(spec_path, index=False)
        completed = subprocess.run(
            ["Rscript", str(r_script), str(data_path), str(spec_path), str(output_dir)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Latest Figure 9 survival diagnostics failed:\n"
                + completed.stdout
                + "\n"
                + completed.stderr
            )
        filenames = (
            "survival_ph_diagnostics.csv",
            "survival_time_varying_hazard_ratios.csv",
            "survival_piecewise_hazard_ratios.csv",
            "survival_rmst_differences.csv",
            "survival_extended_diagnostic_audit.csv",
        )
        outputs = {
            filename: pd.read_csv(output_dir / filename) for filename in filenames
        }

    expected = {
        "survival_piecewise_hazard_ratios.csv": 24,
        "survival_rmst_differences.csv": 16,
        "survival_time_varying_hazard_ratios.csv": 800,
        "survival_extended_diagnostic_audit.csv": 8,
    }
    for filename, expected_rows in expected.items():
        observed = len(outputs[filename])
        if observed != expected_rows:
            raise AssertionError(
                f"Latest Figure 9 {filename} contains {observed:,} rows; expected {expected_rows:,}"
            )
    outputs["survival_piecewise_hazard_ratios.csv"] = _add_bh_fdr(
        outputs["survival_piecewise_hazard_ratios.csv"], ["interval"]
    )
    outputs["survival_rmst_differences.csv"] = _add_bh_fdr(
        outputs["survival_rmst_differences.csv"], ["horizonMonths"]
    )
    for filename, frame in outputs.items():
        frame.to_csv(
            TABLES / f"figure9_latest_{filename.removeprefix('survival_')}",
            index=False,
        )
    return contexts, outputs


def make_figure(
    contexts: dict[tuple[str, str, str], pd.DataFrame],
    models: pd.DataFrame,
    groups: pd.DataFrame,
    audit: pd.DataFrame,
    extended: dict[str, pd.DataFrame],
    screen_results: pd.DataFrame,
) -> None:
    apply_style()
    source = TABLES.parent / "source_data"
    source.mkdir(parents=True, exist_ok=True)
    # Figure 9 source tables are regenerated as one panel-mapped set.
    for stale_path in source.glob("figure9_panel_*.csv"):
        stale_path.unlink()

    # The contexts are prespecified here only for display.  All 29 primary
    # models and their diagnostics remain available in Supplementary Figure 4
    # and Supplementary Data.  Legacy portal-defined GBM is deliberately not a
    # headline survival context because contemporary molecular classification
    # cannot be reconstructed uniformly across the historical cohorts.
    display_specs = FIGURE9_LATEST_CONTEXTS

    def selected_rows(frame: pd.DataFrame) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for plot_row, (scope_name, cancer, gene_a, gene_b, label) in enumerate(
            display_specs
        ):
            selected = frame.loc[
                frame.scope.eq(scope_name)
                & frame.cancer.eq(cancer)
                & frame.geneA.eq(gene_a)
                & frame.geneB.eq(gene_b)
            ].copy()
            if selected.empty:
                raise AssertionError(f"Figure 9 requires {label}")
            selected["displayContext"] = label
            selected["plotRow"] = plot_row
            frames.append(selected)
        return pd.concat(frames, ignore_index=True)

    piecewise = selected_rows(extended["survival_piecewise_hazard_ratios.csv"])
    rmst = selected_rows(extended["survival_rmst_differences.csv"])
    time_varying = extended["survival_time_varying_hazard_ratios.csv"].copy()

    fig = plt.figure(figsize=figsize(180, 135))
    outer = GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[0.96, 1.04],
        hspace=0.31,
    )
    top = GridSpecFromSubplotSpec(
        1,
        4,
        subplot_spec=outer[0],
        width_ratios=[1.52, 0.91, 0.72, 0.89],
        wspace=0.68,
    )
    bottom = GridSpecFromSubplotSpec(
        1,
        3,
        subplot_spec=outer[1],
        width_ratios=[1.02, 1.02, 1.08],
        wspace=0.37,
    )
    ax_a = fig.add_subplot(top[0, 0])
    ax_b = fig.add_subplot(top[0, 1])
    ax_c = fig.add_subplot(top[0, 2])
    ax_d = fig.add_subplot(top[0, 3])
    left = GridSpecFromSubplotSpec(
        2, 1, subplot_spec=bottom[0, 0], height_ratios=[0.72, 0.28], hspace=0.05
    )
    middle = GridSpecFromSubplotSpec(
        2, 1, subplot_spec=bottom[0, 1], height_ratios=[0.72, 0.28], hspace=0.05
    )
    ax_e = fig.add_subplot(left[0]); risk_e = fig.add_subplot(left[1], sharex=ax_e)
    ax_f = fig.add_subplot(middle[0]); risk_f = fig.add_subplot(middle[1], sharex=ax_f)
    ax_g = fig.add_subplot(bottom[0, 2])

    # a -- retain the complete 2,612-context volcano from the revised figure,
    # compacted for the original three-panel top lane.
    screen_display = screen_results.loc[
        screen_results.contrast.eq("A+B versus A−/B−")
        & screen_results.globalFdr.notna()
        & screen_results.fitStatus.str.startswith("estimated")
        & screen_results.hazardRatio.gt(0)
    ].copy()
    if len(screen_display) != 2_612:
        raise AssertionError(
            f"Figure 9a requires all 2,612 eligible joint-state contexts; found {len(screen_display):,}"
        )
    screen_display["log2HazardRatio"] = np.log2(screen_display.hazardRatio)
    screen_display["plotLog2HazardRatio"] = screen_display.log2HazardRatio.clip(-4.0, 4.0)
    screen_display["hazardRatioPlacedAtPlotBoundary"] = screen_display.log2HazardRatio.ne(
        screen_display.plotLog2HazardRatio
    )
    screen_display["negativeLog10GlobalFdr"] = -np.log10(
        screen_display.globalFdr.clip(lower=np.finfo(float).tiny)
    )
    screen_display["globalFdrSignificant"] = screen_display.globalFdr.lt(0.05)
    screen_display["phStatus"] = np.where(
        screen_display.phTestP.lt(0.05), "PH test P<0.05", "PH test P≥0.05"
    )
    screen_display["displayContext"] = np.where(
        screen_display.scope.eq("pan-cancer"),
        "Pan-cancer · " + screen_display.geneA + "–" + screen_display.geneB,
        screen_display.cancer + " · " + screen_display.geneA + "–" + screen_display.geneB,
    )
    screen_display = screen_display.sort_values(["globalFdr", "p", "cancer", "geneA", "geneB"])
    screen_display.to_csv(
        source / "figure9_panel_a_complete_survival_screen.csv", index=False
    )

    nonsignificant = screen_display.loc[~screen_display.globalFdrSignificant]
    ax_a.scatter(
        nonsignificant.plotLog2HazardRatio,
        nonsignificant.negativeLog10GlobalFdr,
        s=4,
        color=COLORS["light_grey"],
        alpha=0.40,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )
    highlighted_colours = {
        "PAN-CANCER": COLORS["black"],
        "UCEC": COLORS["sky"],
        "SKCM": COLORS["green"],
        "BLCA": COLORS["purple"],
        "MDS": COLORS["blue"],
        "COADREAD": COLORS["orange"],
        "LUAD": COLORS["vermillion"],
        "PAAD": "#A6761D",
    }
    significant = screen_display.loc[screen_display.globalFdrSignificant].copy()
    significant["displayCancer"] = np.where(
        significant.cancer.isin(highlighted_colours), significant.cancer, "Other"
    )
    highlighted_colours["Other"] = COLORS["grey"]
    for cancer, colour in highlighted_colours.items():
        selected = significant.loc[significant.displayCancer.eq(cancer)]
        compatible = selected.loc[selected.phTestP.ge(0.05)]
        flagged = selected.loc[selected.phTestP.lt(0.05)]
        ax_a.scatter(
            compatible.plotLog2HazardRatio,
            compatible.negativeLog10GlobalFdr,
            s=7,
            marker="o",
            color=colour,
            alpha=0.80,
            edgecolor="white",
            linewidth=0.18,
            rasterized=True,
            zorder=2,
        )
        ax_a.scatter(
            flagged.plotLog2HazardRatio,
            flagged.negativeLog10GlobalFdr,
            s=9,
            marker="^",
            facecolor="white",
            edgecolor=colour,
            linewidth=0.55,
            rasterized=True,
            zorder=3,
        )
    ax_a.axvline(0, color=COLORS["black"], lw=0.6, ls=(0, (2, 2)))
    ax_a.axhline(-np.log10(0.05), color=COLORS["grey"], lw=0.6, ls=(0, (2, 2)))
    ax_a.set_xlim(-4.0, 4.0)
    ax_a.set_xlabel("Joint-state log₂ HR", fontsize=4.4)
    ax_a.set_ylabel("−log₁₀(global BH FDR)", fontsize=4.4)
    ax_a.set_title(
        "Complete survival screen",
        loc="left",
        fontsize=5.6,
        pad=3,
        fontweight="normal",
    )
    label_specs = [
        ("pan-cancer", "PAN-CANCER", "KRAS", "TP53"),
        ("pan-cancer", "PAN-CANCER", "PIK3CA", "TP53"),
        ("cancer-specific", "LUAD", "KEAP1", "STK11"),
        ("cancer-specific", "UCEC", "ARID1A", "PTEN"),
    ]
    for row in pd.concat(
        [
            screen_display.loc[
                screen_display.scope.eq(scope_name)
                & screen_display.cancer.eq(cancer)
                & screen_display.geneA.eq(gene_a)
                & screen_display.geneB.eq(gene_b)
            ]
            for scope_name, cancer, gene_a, gene_b in label_specs
        ],
        ignore_index=True,
    ).itertuples(index=False):
        is_top_pancancer = row.displayContext == "Pan-cancer · KRAS–TP53"
        ax_a.annotate(
            row.displayContext,
            (row.plotLog2HazardRatio, row.negativeLog10GlobalFdr),
            xytext=(
                3 if row.plotLog2HazardRatio >= 0 else -3,
                -7 if is_top_pancancer else 2,
            ),
            textcoords="offset points",
            ha="left" if row.plotLog2HazardRatio >= 0 else "right",
            va="top" if is_top_pancancer else "bottom",
            fontsize=2.8,
            arrowprops={"arrowstyle": "-", "lw": 0.25, "color": COLORS["grey"]},
        )
    cancer_legend = ax_a.legend(
        handles=[
            Line2D([0], [0], marker="o", ls="", markersize=2.7, mfc=colour, mec="none", label=("Pan" if cancer == "PAN-CANCER" else cancer))
            for cancer, colour in highlighted_colours.items()
        ],
        loc="upper left",
        ncol=3,
        fontsize=2.45,
        columnspacing=0.38,
        handletextpad=0.15,
        labelspacing=0.15,
        borderpad=0.22,
        frameon=True,
        fancybox=True,
        framealpha=0.92,
        edgecolor=COLORS["light_grey"],
    )
    ax_a.add_artist(cancer_legend)
    ax_a.legend(
        handles=[
            Line2D([0], [0], marker="o", ls="", markersize=2.8, mfc=COLORS["grey"], mec="none", label="PH P≥0.05"),
            Line2D([0], [0], marker="^", ls="", markersize=3.0, mfc="white", mec=COLORS["grey"], label="PH P<0.05"),
        ],
        **LEGEND_BOX,
        loc="upper left",
        # Stack the PH-symbol key immediately below the cancer-family key so
        # neither legend obscures the leading pan-cancer KRAS–TP53 context.
        bbox_to_anchor=(0.0, 0.885),
        fontsize=2.65,
        handletextpad=0.2,
        labelspacing=0.12,
    )

    # b -- combine the leading cancer-specific and pan-cancer joint-state
    # associations in one compact forest plot.
    cancer_leaders = (
        screen_display.loc[screen_display.scope.eq("cancer-specific")]
        .nsmallest(4, ["globalFdr", "p"])
        .copy()
    )
    pancancer_leaders = (
        screen_display.loc[screen_display.scope.eq("pan-cancer")]
        .nsmallest(4, ["globalFdr", "p"])
        .copy()
    )
    cancer_leaders["scopeLabel"] = "Cancer-specific"
    pancancer_leaders["scopeLabel"] = "Pan-cancer"
    leading_forest = pd.concat(
        [cancer_leaders, pancancer_leaders], ignore_index=True
    )
    leading_forest["plotY"] = np.arange(len(leading_forest), dtype=float)
    leading_forest.to_csv(
        source / "figure9_panel_b_leading_associations.csv", index=False
    )
    forest_colours = np.where(
        leading_forest.hazardRatio.ge(1), COLORS["vermillion"], COLORS["blue"]
    )
    ph_compatible = leading_forest.phTestP.ge(0.05)
    group_boundary = len(cancer_leaders) - 0.5
    ax_b.axhspan(
        group_boundary,
        len(leading_forest) - 0.5,
        color=COLORS["very_light_grey"],
        zorder=0,
    )
    ax_b.hlines(
        leading_forest.plotY,
        leading_forest.ciLow,
        leading_forest.ciHigh,
        color=forest_colours,
        lw=0.75,
        alpha=0.9,
        zorder=2,
    )
    ax_b.scatter(
        leading_forest.loc[ph_compatible, "hazardRatio"],
        leading_forest.loc[ph_compatible, "plotY"],
        c=forest_colours[ph_compatible],
        s=13,
        marker="o",
        edgecolor="white",
        linewidth=0.25,
        zorder=3,
    )
    ax_b.scatter(
        leading_forest.loc[~ph_compatible, "hazardRatio"],
        leading_forest.loc[~ph_compatible, "plotY"],
        facecolor="white",
        edgecolor=forest_colours[~ph_compatible],
        s=16,
        marker="^",
        linewidth=0.65,
        zorder=3,
    )
    ax_b.axvline(1, color=COLORS["black"], lw=0.6, ls=(0, (2, 2)))
    ax_b.axhline(group_boundary, color=COLORS["light_grey"], lw=0.55)
    ax_b.set_xscale("log")
    ax_b.set_xlim(0.19, 4.2)
    ax_b.set_xticks([0.25, 0.5, 1, 2, 4], ["0.25", "0.5", "1", "2", "4"])
    ax_b.tick_params(axis="x", which="minor", labelbottom=False)
    ax_b.set_yticks(
        leading_forest.plotY,
        [label.replace(" · ", "\n") for label in leading_forest.displayContext],
        fontsize=3.45,
    )
    ax_b.set_ylim(len(leading_forest) - 0.48, -0.52)
    ax_b.set_xlabel("Joint-state HR (95% CI)", fontsize=4.1)
    ax_b.set_title(
        "Leading survival associations",
        loc="left",
        fontsize=5.4,
        pad=3,
        fontweight="normal",
    )
    ax_b.legend(
        handles=[
            Line2D([0], [0], marker="o", ls="", markersize=2.9, mfc=COLORS["grey"], mec="none", label="PH P≥0.05"),
            Line2D([0], [0], marker="^", ls="", markersize=3.1, mfc="white", mec=COLORS["grey"], label="PH P<0.05"),
        ],
        **LEGEND_BOX,
        loc="lower right",
        fontsize=2.55,
        handletextpad=0.18,
        labelspacing=0.10,
    )

    # c -- interval-specific effects replace the constant-HR forest for the
    # contexts in which the proportional-hazards assumption is untenable.
    interval_order = ["0–12 months", "12–36 months", ">36 months"]
    piecewise["interval"] = pd.Categorical(
        piecewise.interval, categories=interval_order, ordered=True
    )
    piecewise["plotColumn"] = piecewise.interval.cat.codes
    piecewise = piecewise.sort_values(["plotRow", "plotColumn"]).reset_index(drop=True)
    piecewise["log2HazardRatio"] = np.log2(piecewise.hazardRatio)
    piecewise.to_csv(
        source / "figure9_panel_c_piecewise_hazard_ratios.csv", index=False
    )
    hr_matrix = (
        piecewise.pivot(index="displayContext", columns="interval", values="hazardRatio")
        .reindex(
            index=[spec[-1] for spec in display_specs], columns=interval_order
        )
        .to_numpy(float)
    )
    q_matrix = (
        piecewise.pivot(index="displayContext", columns="interval", values="fdr")
        .reindex(
            index=[spec[-1] for spec in display_specs], columns=interval_order
        )
        .to_numpy(float)
    )
    image = ax_c.imshow(
        np.log2(hr_matrix), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto",
        interpolation="nearest",
    )
    ax_c.set_xticks(np.arange(3), ["0–12", "12–36", ">36"], fontsize=4.0)
    ax_c.set_yticks(
        np.arange(len(display_specs)),
        [spec[-1].replace(" · ", "\n") for spec in display_specs],
        fontsize=3.45,
    )
    ax_c.set_xlabel("Months after reported survival origin", fontsize=3.85)
    ax_c.set_title("Time-specific survival associations", loc="left", fontsize=5.2, pad=3)
    ax_c.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax_c.set_yticks(np.arange(-0.5, len(display_specs), 1), minor=True)
    ax_c.grid(which="minor", color="white", lw=0.65)
    ax_c.tick_params(which="minor", bottom=False, left=False)
    for row in range(hr_matrix.shape[0]):
        for column in range(hr_matrix.shape[1]):
            value = hr_matrix[row, column]
            label = "NE" if not np.isfinite(value) else f"{value:.2f}"
            if np.isfinite(q_matrix[row, column]) and q_matrix[row, column] < 0.05:
                label += "*"
            ax_c.text(
                column, row, label, ha="center", va="center", fontsize=4.0,
                color="white" if np.isfinite(value) and abs(np.log2(value)) > 1.15 else COLORS["black"],
            )
    color_ax = ax_c.inset_axes([1.045, 0.55, 0.045, 0.42])
    colour_bar = fig.colorbar(image, cax=color_ax, orientation="vertical")
    colour_bar.set_ticks([-2, 0, 2])
    colour_bar.set_ticklabels(["0.25", "1", "4"])
    colour_bar.set_label("Interval-specific HR", fontsize=3.35, labelpad=1.2)
    colour_bar.ax.tick_params(labelsize=3.5, length=1.5, pad=1)
    ax_c.text(
        1.04, 0.50, "* BH q<0.05", transform=ax_c.transAxes,
        ha="left", va="top", fontsize=3.15,
    )

    # d -- absolute survival-time differences provide a directly interpretable
    # complement to the relative piecewise effects.
    rmst["plotOffset"] = rmst.horizonMonths.map({36: -0.11, 60: 0.11})
    rmst["plotY"] = rmst.plotRow + rmst.plotOffset
    rmst = rmst.sort_values(["plotRow", "horizonMonths"]).reset_index(drop=True)
    rmst.to_csv(source / "figure9_panel_d_rmst_differences.csv", index=False)
    for horizon, colour, marker, label in (
        (36, COLORS["blue"], "o", "36 months"),
        (60, COLORS["purple"], "s", "60 months"),
    ):
        selected = rmst.loc[rmst.horizonMonths.eq(horizon)]
        ax_d.hlines(selected.plotY, selected.ciLow, selected.ciHigh, color=colour, lw=0.9)
        ax_d.scatter(
            selected.rmstDifferenceMonths, selected.plotY, s=19, marker=marker,
            color=colour, edgecolor="white", lw=0.3, zorder=3, label=label,
        )
    ax_d.axvline(0, color=COLORS["black"], lw=0.65, ls=(0, (2, 2)))
    ax_d.set_yticks(
        np.arange(len(display_specs)),
        [spec[-1].replace(" · ", "\n") for spec in display_specs],
        fontsize=3.55,
    )
    ax_d.set_ylim(len(display_specs) - 0.48, -0.52)
    x_min = min(float(rmst.ciLow.min()) - 2.0, -2.0)
    x_max = max(float(rmst.ciHigh.max()) + 2.0, 2.0)
    ax_d.set_xlim(x_min, x_max)
    ax_d.set_xlabel("RMST difference, A+B minus A−/B− (months)", fontsize=3.8)
    ax_d.set_title("Restricted mean survival time", loc="left", fontsize=5.2, pad=3)
    ax_d.legend(
        **LEGEND_BOX, loc="upper right", fontsize=3.8, ncol=1,
        handlelength=1.0, columnspacing=0.8, handletextpad=0.3,
        labelspacing=0.15,
    )

    e_key = ("PAN-CANCER", "KRAS", "TP53")
    f_key = ("LUAD", "KEAP1", "STK11")
    curves_e, risks_e, logrank_e, medians_e = draw_km(
        ax_e,
        risk_e,
        contexts[e_key],
        "KRAS",
        "TP53",
        "e",
        display_cancer="Pan-cancer",
    )
    curves_f, risks_f, logrank_f, medians_f = draw_km(
        ax_f, risk_f, contexts[f_key], "KEAP1", "STK11", "f"
    )
    # ``axis('off')`` on the shared risk-table axes can suppress the KM tick
    # labels in some Matplotlib backends; make both time axes explicit.
    km_ticks = np.array([0, 12, 24, 36, 60, 84, 120], dtype=float)
    for km_ax in (ax_e, ax_f):
        km_ax.set_xticks(km_ticks)
        km_ax.set_xticklabels(["0", "12", "24", "36", "60", "84", "120"])
        km_ax.tick_params(axis="x", labelbottom=True, pad=1.5)
        plt.setp(km_ax.get_xticklabels(), visible=True)
    curves_e.to_csv(source / "figure9_panel_e_km.csv", index=False)
    risks_e.to_csv(source / "figure9_panel_e_risk.csv", index=False)
    logrank_e.to_csv(source / "figure9_panel_e_logrank.csv", index=False)
    medians_e.to_csv(source / "figure9_panel_e_medians.csv", index=False)
    curves_f.to_csv(source / "figure9_panel_f_km.csv", index=False)
    risks_f.to_csv(source / "figure9_panel_f_risk.csv", index=False)
    logrank_f.to_csv(source / "figure9_panel_f_logrank.csv", index=False)
    medians_f.to_csv(source / "figure9_panel_f_medians.csv", index=False)

    # g -- smooth beta(t) estimates derived from scaled Schoenfeld residuals
    # expose how the A+B contrast changes during follow-up.
    time_contexts = [
        ("KEAP1–KRAS (LUAD)", "LUAD · KEAP1–KRAS", COLORS["vermillion"]),
        ("ASXL1–RUNX1 (MDS)", "MDS · ASXL1–RUNX1", COLORS["orange"]),
        ("CDKN2A–KRAS (PAAD)", "PAAD · CDKN2A–KRAS", COLORS["blue"]),
        ("ARID1A–PTEN (UCEC)", "UCEC · ARID1A–PTEN", COLORS["purple"]),
        ("KRAS–TP53 (pan-cancer)", "Pan-cancer · KRAS–TP53", COLORS["green"]),
    ]
    time_frames: list[pd.DataFrame] = []
    for context_name, display_label, colour in time_contexts:
        selected = time_varying.loc[
            time_varying.context.eq(context_name)
            & time_varying.eventTimeMonths.le(60)
        ].copy()
        if selected.empty:
            raise AssertionError(f"Figure 9g requires time-varying estimates for {context_name}")
        selected["displayContext"] = display_label
        time_frames.append(selected)
        ax_g.fill_between(
            selected.eventTimeMonths.to_numpy(float), selected.ciLow.to_numpy(float),
            selected.ciHigh.to_numpy(float), color=colour, alpha=0.09, lw=0,
        )
        ax_g.plot(
            selected.eventTimeMonths, selected.hazardRatio, color=colour, lw=1.05,
            label=display_label,
        )
    time_plot = pd.concat(time_frames, ignore_index=True)
    time_plot.to_csv(source / "figure9_panel_g_time_varying_hazard_ratios.csv", index=False)
    ax_g.axhline(1, color=COLORS["black"], lw=0.65, ls=(0, (2, 2)))
    ax_g.set_xscale("log")
    # Only the hazard-ratio axis is logarithmic; follow-up remains on a linear
    # month scale.
    ax_g.set_xscale("linear")
    ax_g.set_yscale("log")
    ax_g.set_xlim(0, 60)
    ax_g.set_ylim(0.125, 8.5)
    ax_g.set_yticks(
        [0.125, 0.25, 0.5, 1, 2, 4, 8],
        ["0.125", "0.25", "0.5", "1", "2", "4", "8"],
    )
    ax_g.tick_params(axis="y", which="minor", labelleft=False)
    ax_g.set_xlabel("Months after reported survival origin", fontsize=4.5)
    ax_g.set_ylabel("Time-varying A+B vs A−/B− HR", fontsize=4.45)
    ax_g.set_title("Hazard ratios change during follow-up", loc="left", fontsize=5.8, pad=4.0)
    ax_g.legend(
        **LEGEND_BOX, fontsize=3.35, ncol=1, loc="upper right",
        handletextpad=0.35, labelspacing=0.2,
    )
    fig.subplots_adjust(left=0.070, right=0.99, top=0.952, bottom=0.075)

    aligned_panel_labels(
        fig,
        [
            (("a", ax_a), ("b", ax_b), ("c", ax_c), ("d", ax_d)),
            (("e", ax_e), ("f", ax_f), ("g", ax_g)),
        ],
    )

    save_figure(fig, FIGURES / "figure9_survival")
    plt.close(fig)


def make_survival_screen_figure(
    contexts: dict[tuple[str, str, str], pd.DataFrame],
    screen_results: pd.DataFrame,
) -> None:
    """Render the unbiased survival screen as the active main Figure 9.

    The complete screen contributes one joint-state point for every eligible
    cancer/gene-pair context.  Detailed time-dependent diagnostics remain in
    Supplementary Figure 4 rather than competing with the screen-level result in
    the main figure.
    """
    apply_style()
    source = TABLES.parent / "source_data"
    source.mkdir(parents=True, exist_ok=True)
    for stale_path in source.glob("figure9_panel_*.csv"):
        stale_path.unlink()

    joint = screen_results.loc[
        screen_results.contrast.eq("A+B versus A−/B−")
        & screen_results.fitStatus.str.startswith("estimated")
        & screen_results.hazardRatio.gt(0)
        & screen_results.globalFdr.notna()
    ].copy()
    if len(joint) != 2_612:
        raise AssertionError(
            f"Figure 9a requires all 2,612 eligible joint-state contexts; found {len(joint):,}"
        )
    joint["log2HazardRatio"] = np.log2(joint.hazardRatio)
    joint["plotLog2HazardRatio"] = joint.log2HazardRatio.clip(-4.5, 4.5)
    joint["hazardRatioPlacedAtPlotBoundary"] = joint.log2HazardRatio.ne(
        joint.plotLog2HazardRatio
    )
    joint["negativeLog10GlobalFdr"] = -np.log10(
        joint.globalFdr.clip(lower=np.finfo(float).tiny)
    )
    joint["globalFdrSignificant"] = joint.globalFdr.lt(0.05)
    joint["phStatus"] = np.where(
        joint.phTestP.lt(0.05), "PH test P<0.05", "PH test P≥0.05"
    )
    joint["displayContext"] = np.where(
        joint.scope.eq("pan-cancer"),
        "Pan-cancer · " + joint.geneA + "–" + joint.geneB,
        joint.cancer + " · " + joint.geneA + "–" + joint.geneB,
    )
    joint = joint.sort_values(["globalFdr", "p", "cancer", "geneA", "geneB"])
    joint.to_csv(source / "figure9_panel_a_complete_survival_screen.csv", index=False)

    cancer_leaders = (
        joint.loc[joint.scope.eq("cancer-specific")]
        .nsmallest(10, ["globalFdr", "p"])
        .copy()
    )
    pancancer_leaders = (
        joint.loc[joint.scope.eq("pan-cancer")]
        .nsmallest(10, ["globalFdr", "p"])
        .copy()
    )
    cancer_leaders.to_csv(
        source / "figure9_panel_c_cancer_specific_leaders.csv", index=False
    )
    pancancer_leaders.to_csv(
        source / "figure9_panel_d_pancancer_leaders.csv", index=False
    )

    fig = plt.figure(figsize=figsize(180, 122))
    outer = GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[1.08, 0.92],
        hspace=0.46,
    )
    top = GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=outer[0],
        width_ratios=[1.62, 1.0],
        wspace=0.30,
    )
    bottom = GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=outer[1],
        width_ratios=[1.03, 0.97],
        wspace=0.38,
    )
    ax_a = fig.add_subplot(top[0, 0])
    km_grid = GridSpecFromSubplotSpec(
        2,
        1,
        subplot_spec=top[0, 1],
        height_ratios=[0.77, 0.23],
        hspace=0.04,
    )
    ax_b = fig.add_subplot(km_grid[0])
    risk_b = fig.add_subplot(km_grid[1], sharex=ax_b)
    ax_c = fig.add_subplot(bottom[0, 0])
    ax_d = fig.add_subplot(bottom[0, 1])

    # a -- every outcome-independent, mutation-frequency-qualified context.
    significant_cancers = sorted(
        joint.loc[joint.globalFdrSignificant, "cancer"].unique(),
        key=lambda value: (value == "PAN-CANCER", value),
    )
    tab20 = list(plt.colormaps["tab20"].colors)
    cancer_colours = {
        cancer: (
            COLORS["black"]
            if cancer == "PAN-CANCER"
            else tab20[index % len(tab20)]
        )
        for index, cancer in enumerate(significant_cancers)
    }
    nonsignificant = joint.loc[~joint.globalFdrSignificant]
    ax_a.scatter(
        nonsignificant.plotLog2HazardRatio,
        nonsignificant.negativeLog10GlobalFdr,
        s=5,
        color=COLORS["light_grey"],
        alpha=0.42,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )
    for cancer in significant_cancers:
        selected = joint.loc[joint.globalFdrSignificant & joint.cancer.eq(cancer)]
        compatible = selected.loc[selected.phTestP.ge(0.05)]
        flagged = selected.loc[selected.phTestP.lt(0.05)]
        colour = cancer_colours[cancer]
        if len(compatible):
            ax_a.scatter(
                compatible.plotLog2HazardRatio,
                compatible.negativeLog10GlobalFdr,
                s=10,
                marker="o",
                color=colour,
                alpha=0.82,
                edgecolor="white",
                linewidth=0.2,
                rasterized=True,
                zorder=2,
            )
        if len(flagged):
            ax_a.scatter(
                flagged.plotLog2HazardRatio,
                flagged.negativeLog10GlobalFdr,
                s=13,
                marker="^",
                facecolor="white",
                edgecolor=colour,
                linewidth=0.65,
                rasterized=True,
                zorder=3,
            )
    ax_a.axvline(0, color=COLORS["black"], lw=0.65, ls=(0, (2, 2)))
    ax_a.axhline(
        -np.log10(0.05), color=COLORS["grey"], lw=0.65, ls=(0, (2, 2))
    )
    ax_a.set_xlabel("Joint-state log₂ hazard ratio, A+B versus A−/B−")
    ax_a.set_ylabel("−log₁₀(global BH FDR)")
    ax_a.set_xlim(-4.8, 4.8)
    ax_a.set_title(
        "Complete outcome-independent survival screen",
        loc="left",
        fontsize=5.6,
        pad=3.0,
        fontweight="normal",
    )
    label_specs = [
        ("pan-cancer", "PAN-CANCER", "KRAS", "TP53"),
        ("pan-cancer", "PAN-CANCER", "PIK3CA", "TP53"),
        ("cancer-specific", "LUAD", "KEAP1", "STK11"),
        ("cancer-specific", "MDS", "ASXL1", "RUNX1"),
        ("cancer-specific", "UCEC", "ARID1A", "PTEN"),
        ("cancer-specific", "PAAD", "KRAS", "SMAD4"),
        ("pan-cancer", "PAN-CANCER", "APC", "ATM"),
    ]
    label_rows = pd.concat(
        [
            joint.loc[
                joint.scope.eq(scope_name)
                & joint.cancer.eq(cancer)
                & joint.geneA.eq(gene_a)
                & joint.geneB.eq(gene_b)
            ]
            for scope_name, cancer, gene_a, gene_b in label_specs
        ],
        ignore_index=True,
    )
    for row_index, row in enumerate(label_rows.itertuples(index=False)):
        horizontal = 5 if row.log2HazardRatio >= 0 else -5
        vertical = 3 if row_index % 2 == 0 else -3
        ax_a.annotate(
            row.displayContext,
            (row.plotLog2HazardRatio, row.negativeLog10GlobalFdr),
            xytext=(horizontal, vertical),
            textcoords="offset points",
            ha="left" if horizontal > 0 else "right",
            va="bottom" if vertical > 0 else "top",
            fontsize=3.35,
            color=COLORS["black"],
            arrowprops={"arrowstyle": "-", "lw": 0.3, "color": COLORS["grey"]},
            zorder=5,
        )
    cancer_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            ls="",
            markersize=3.0,
            mfc=cancer_colours[cancer],
            mec="none",
            label="Pan-cancer" if cancer == "PAN-CANCER" else cancer,
        )
        for cancer in significant_cancers
    ]
    cancer_legend = ax_a.legend(
        handles=cancer_handles,
        title="Cancer family among global-FDR-significant contexts",
        loc="upper left",
        bbox_to_anchor=(0.005, 0.985),
        ncol=7,
        fontsize=2.75,
        title_fontsize=3.2,
        columnspacing=0.50,
        handletextpad=0.18,
        labelspacing=0.22,
        borderpad=0.28,
        frameon=True,
        fancybox=True,
        framealpha=0.94,
        facecolor="#FFFFFF",
        edgecolor=COLORS["light_grey"],
    )
    ax_a.add_artist(cancer_legend)
    ax_a.legend(
        handles=[
            Line2D(
                [0], [0], marker="o", ls="", markersize=3.5,
                mfc=COLORS["grey"], mec="none", label="PH test P≥0.05",
            ),
            Line2D(
                [0], [0], marker="^", ls="", markersize=3.8,
                mfc="white", mec=COLORS["grey"], label="PH test P<0.05",
            ),
        ],
        **LEGEND_BOX,
        loc="lower left",
        bbox_to_anchor=(0.006, 0.035),
        fontsize=3.2,
        handletextpad=0.30,
        labelspacing=0.18,
    )
    if joint.hazardRatioPlacedAtPlotBoundary.any():
        ax_a.text(
            0.995,
            0.018,
            "HRs outside the displayed range are placed at the boundary",
            transform=ax_a.transAxes,
            ha="right",
            va="bottom",
            fontsize=3.0,
            color=COLORS["grey"],
        )

    # b -- the leading pan-cancer result, shown as four observed genotype states.
    pan_key = ("PAN-CANCER", "KRAS", "TP53")
    if pan_key not in contexts:
        raise AssertionError("Figure 9b requires the pan-cancer KRAS–TP53 context")
    curves_b, risks_b, logrank_b, medians_b = draw_km(
        ax_b,
        risk_b,
        contexts[pan_key],
        "KRAS",
        "TP53",
        "b",
        display_cancer="Pan-cancer",
    )
    leading_pan = joint.loc[
        joint.scope.eq("pan-cancer")
        & joint.geneA.eq("KRAS")
        & joint.geneB.eq("TP53")
    ].iloc[0]
    ax_b.text(
        0.02,
        0.13,
        f"stratified Cox HR={leading_pan.hazardRatio:.2f} "
        f"({leading_pan.ciLow:.2f}–{leading_pan.ciHigh:.2f}); "
        f"global q={leading_pan.globalFdr:.1e}",
        transform=ax_b.transAxes,
        fontsize=3.65,
    )
    curves_b.to_csv(source / "figure9_panel_b_pancancer_km.csv", index=False)
    risks_b.to_csv(source / "figure9_panel_b_pancancer_risk.csv", index=False)
    logrank_b.to_csv(source / "figure9_panel_b_pancancer_logrank.csv", index=False)
    medians_b.to_csv(source / "figure9_panel_b_pancancer_medians.csv", index=False)

    def draw_leader_forest(
        ax: plt.Axes,
        frame: pd.DataFrame,
        title_text: str,
    ) -> None:
        plot = frame.sort_values(["globalFdr", "p"], ascending=True).reset_index(drop=True)
        plot["plotY"] = np.arange(len(plot))
        compatible = plot.phTestP.ge(0.05)
        colours = np.where(
            plot.hazardRatio.ge(1), COLORS["vermillion"], COLORS["blue"]
        )
        ax.hlines(plot.plotY, plot.ciLow, plot.ciHigh, color=colours, lw=0.9, alpha=0.9)
        ax.scatter(
            plot.loc[compatible, "hazardRatio"],
            plot.loc[compatible, "plotY"],
            c=colours[compatible],
            s=19,
            marker="o",
            edgecolor="white",
            linewidth=0.3,
            zorder=3,
        )
        ax.scatter(
            plot.loc[~compatible, "hazardRatio"],
            plot.loc[~compatible, "plotY"],
            facecolor="white",
            edgecolor=colours[~compatible],
            s=22,
            marker="^",
            linewidth=0.7,
            zorder=3,
        )
        ax.axvline(1, color=COLORS["black"], lw=0.65, ls=(0, (2, 2)))
        ax.set_xscale("log")
        left = max(0.12, float(plot.ciLow.min()) * 0.76)
        right = max(4.2, float(plot.ciHigh.max()) * 1.12)
        ax.set_xlim(left, right)
        ticks = [tick for tick in [0.25, 0.5, 1, 2, 4, 8, 16] if left <= tick <= right]
        ax.set_xticks(ticks, [f"{tick:g}" for tick in ticks])
        ax.tick_params(axis="x", which="minor", labelbottom=False)
        ax.set_yticks(
            plot.plotY,
            [label.replace(" · ", "\n") for label in plot.displayContext],
            fontsize=4.1,
        )
        ax.set_ylim(len(plot) - 0.48, -0.52)
        ax.set_xlabel("Joint-state hazard ratio (95% CI)")
        ax.set_title(
            title_text, loc="left", fontsize=5.6, pad=3.0, fontweight="normal"
        )
        ax.legend(
            handles=[
                Line2D(
                    [0], [0], marker="o", ls="", markersize=3.5,
                    mfc=COLORS["grey"], mec="none", label="PH test P≥0.05",
                ),
                Line2D(
                    [0], [0], marker="^", ls="", markersize=3.8,
                    mfc="white", mec=COLORS["grey"], label="PH test P<0.05",
                ),
            ],
            **LEGEND_BOX,
            loc="lower right",
            fontsize=3.4,
            handletextpad=0.25,
            labelspacing=0.16,
        )

    draw_leader_forest(ax_c, cancer_leaders, "Leading cancer-specific associations")
    draw_leader_forest(ax_d, pancancer_leaders, "Leading pan-cancer associations")

    fig.subplots_adjust(left=0.105, right=0.988, top=0.955, bottom=0.078)
    aligned_panel_labels(
        fig,
        [
            (("a", ax_a), ("b", ax_b)),
            (("c", ax_c), ("d", ax_d)),
        ],
    )
    save_figure(fig, FIGURES / "figure9_survival")
    plt.close(fig)


def make_diagnostics(
    models: pd.DataFrame,
    missing: pd.DataFrame,
    endpoint_audit: pd.DataFrame,
    extended: dict[str, pd.DataFrame],
    screen_results: pd.DataFrame,
) -> None:
    apply_style()
    supplementary = FIGURES / "supplementary"
    supplementary.mkdir(parents=True, exist_ok=True)
    source = TABLES.parent / "source_data"
    source.mkdir(parents=True, exist_ok=True)
    for stale_path in source.glob("figureS4_panel_*.csv"):
        stale_path.unlink()
    fig, axes = plt.subplots(2, 3, figsize=figsize(180, 145))
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes.ravel()

    # a -- mutually exclusive endpoint eligibility states.
    endpoint_plot = endpoint_audit.loc[endpoint_audit.nPatients.gt(0)].copy()
    endpoint_plot["plotLabel"] = endpoint_plot.eligibilityState.replace(
        {
            "selected tumour without linked study-specific patient record": "no linked clinical record",
            "missing or non-finite OS time": "OS time missing/non-finite",
            "missing or unrecognised OS status": "OS status missing/unrecognised",
        }
    )
    endpoint_plot = endpoint_plot.iloc[::-1].reset_index(drop=True)
    colors_a = [
        COLORS["blue"] if state == "positive OS time"
        else COLORS["orange"] if state == "zero OS time"
        else COLORS["grey"]
        for state in endpoint_plot.eligibilityState
    ]
    ax_a.barh(np.arange(len(endpoint_plot)), endpoint_plot.nPatients, color=colors_a)
    ax_a.set_yticks(
        np.arange(len(endpoint_plot)), endpoint_plot.plotLabel, fontsize=4.5
    )
    ax_a.set_xscale("log")
    ax_a.set_xlabel("Selected patients (log scale)")
    ax_a.set_title(
        "Overall-survival endpoint eligibility",
        loc="left",
        fontsize=5.5,
        pad=3,
        fontweight="normal",
    )
    for yy, count in enumerate(endpoint_plot.nPatients):
        ax_a.text(count, yy, f"  {int(count):,}", va="center", fontsize=4.5)
    panel_label(ax_a, "a", x=-0.12, y=1.045)
    endpoint_plot.to_csv(source / "figureS4_panel_a_endpoint_eligibility.csv", index=False)

    # b -- retain the author-approved replacement panel: absolute 60-month
    # survival-time differences from the expanded diagnostic subset.
    rmst_plot = extended["survival_rmst_differences.csv"].loc[
        lambda frame: frame.horizonMonths.eq(60)
    ].copy()
    rmst_plot = rmst_plot.sort_values("rmstDifferenceMonths").reset_index(drop=True)
    rmst_plot["plotY"] = np.arange(len(rmst_plot))
    ax_b.barh(
        rmst_plot.plotY,
        rmst_plot.rmstDifferenceMonths,
        color=np.where(
            rmst_plot.rmstDifferenceMonths.lt(0),
            COLORS["vermillion"],
            COLORS["blue"],
        ),
    )
    ax_b.axvline(0, color=COLORS["black"], lw=0.65, ls=(0, (2, 2)))
    ax_b.set_yticks(
        rmst_plot.plotY,
        rmst_plot.context.str.replace(" (GBM)", " (legacy GBM)", regex=False),
        fontsize=3.0,
    )
    ax_b.set_xlabel("60-month RMST difference, A+B minus A−/B− (months)")
    ax_b.set_title(
        "Restricted mean survival time",
        loc="left",
        fontsize=5.5,
        pad=3,
        fontweight="normal",
    )
    panel_label(ax_b, "b", x=-0.12, y=1.045)
    rmst_plot.to_csv(source / "figureS4_panel_b_60_month_rmst.csv", index=False)

    # c -- clustered-to-model-based standard-error ratio for the A+B term.
    conventional = models.loc[
        models.model.eq("four-group primary") & models.term.eq("Both"),
        ["context", "scope", "standardError"],
    ].rename(columns={"standardError": "modelBasedSe"})
    clustered = models.loc[
        models.model.eq("four-group study-clustered sensitivity")
        & models.term.eq("Both")
        & models.fitStatus.str.startswith("estimated"),
        ["context", "standardError", "nStudies"],
    ].rename(columns={"standardError": "studyClusteredSe"})
    variance_plot = conventional.merge(clustered, on="context", validate="one_to_one")
    variance_plot["seRatioClusteredToModelBased"] = (
        variance_plot.studyClusteredSe / variance_plot.modelBasedSe
    )
    variance_plot = variance_plot.sort_values("seRatioClusteredToModelBased").reset_index(drop=True)
    colors_c = np.where(
        variance_plot.scope.eq("pan-cancer"), COLORS["vermillion"], COLORS["blue"]
    )
    ax_c.scatter(
        variance_plot.seRatioClusteredToModelBased,
        np.arange(len(variance_plot)),
        c=colors_c,
        s=20,
    )
    ax_c.axvline(1, color=COLORS["black"], ls=(0, (2, 2)), lw=0.7)
    ax_c.set_yticks(np.arange(len(variance_plot)), variance_plot.context, fontsize=4.4)
    ax_c.set_xlabel("Clustered/model-based SE ratio")
    ax_c.set_title(
        "Variance sensitivity",
        loc="left",
        fontsize=5.5,
        pad=3,
        fontweight="normal",
    )
    ax_c.legend(
        handles=[
            Line2D([0], [0], marker="o", ls="", mfc=COLORS["blue"], mec="none", label="Cancer-specific"),
            Line2D([0], [0], marker="o", ls="", mfc=COLORS["vermillion"], mec="none", label="Pan-cancer"),
        ],
        **LEGEND_BOX,
        loc="best",
        fontsize=4.4,
    )
    panel_label(ax_c, "c", x=-0.12, y=1.045)
    variance_plot.to_csv(source / "figureS4_panel_c_variance_sensitivity.csv", index=False)

    # d -- coefficient-level proportional-hazards diagnostics for the complete
    # joint-state screen, rather than only the former 29-context subset.
    ph = screen_results.loc[
        screen_results.contrast.eq("A+B versus A−/B−")
        & screen_results.fitStatus.str.startswith("estimated")
        & screen_results.phTestP.notna()
    ].copy()
    if len(ph) != 2_612:
        raise AssertionError(
            f"Supplementary Figure 4d requires 2,612 joint-state models; found {len(ph):,}"
        )
    ph["phFdr"] = multipletests(ph.phTestP, method="fdr_bh")[1]
    ph["label"] = np.where(
        ph.scope.eq("pan-cancer"),
        "Pan-cancer · " + ph.geneA + "–" + ph.geneB,
        ph.cancer + " · " + ph.geneA + "–" + ph.geneB,
    )
    ph["negativeLog10P"] = -np.log10(ph.phTestP.clip(lower=np.finfo(float).tiny))
    ph = ph.sort_values("phTestP", ascending=True).reset_index(drop=True)
    ph["plotRank"] = np.arange(1, len(ph) + 1)
    colors = np.where(
        ph.phFdr.lt(0.05), COLORS["vermillion"],
        np.where(ph.phTestP.lt(0.05), COLORS["orange"], COLORS["grey"]),
    )
    ax_d.scatter(
        ph.plotRank,
        ph.negativeLog10P,
        c=colors,
        s=6,
        alpha=0.72,
        linewidths=0,
        rasterized=True,
    )
    ax_d.axhline(-np.log10(0.05), color=COLORS["black"], ls=(0, (2, 2)), lw=0.7)
    ax_d.set_xlabel("Contexts ranked by PH-test P")
    ax_d.set_ylabel("−log₁₀(P), coefficient-level PH test")
    ax_d.set_title(
        "Proportional-hazards diagnostics across all 2,612 contexts",
        loc="left",
        fontsize=5.5,
        pad=3,
        fontweight="normal",
    )
    ax_d.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                ls="",
                mfc=COLORS["vermillion"],
                mec="none",
                label="BH q < 0.05",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                ls="",
                mfc=COLORS["orange"],
                mec="none",
                label="P < 0.05, q ≥ 0.05",
            ),
        ],
        **LEGEND_BOX,
        loc="lower right",
        fontsize=3.55,
        handletextpad=0.4,
    )
    panel_label(ax_d, "d", x=-0.12, y=1.045)
    ph.to_csv(source / "figureS4_panel_d_ph_diagnostics.csv", index=False)

    # e -- a direct scaled-Schoenfeld diagnostic for the strongest PH failure.
    residuals = extended["survival_scaled_schoenfeld_residuals.csv"]
    smooth = extended["survival_time_varying_hazard_ratios.csv"]
    residual_plot = residuals.loc[
        residuals.context.eq("KEAP1–STK11 (LUAD)")
        & residuals.eventTimeMonths.le(60)
    ].copy()
    smooth_plot = smooth.loc[
        smooth.context.eq("KEAP1–STK11 (LUAD)")
        & smooth.eventTimeMonths.le(60)
    ].copy()
    # Deterministic thinning reduces PDF complexity without changing the source
    # table, which retains every residual used by the diagnostic.
    residual_display = residual_plot.iloc[:: max(len(residual_plot) // 1000, 1)]
    ax_e.scatter(
        residual_display.eventTimeMonths, residual_display.scaledSchoenfeldBeta,
        s=3, color=COLORS["grey"], alpha=0.18, linewidths=0,
    )
    ax_e.plot(
        smooth_plot.eventTimeMonths, smooth_plot.coefficient,
        color=COLORS["vermillion"], lw=1.1, label="Spline-smoothed β(t)",
    )
    ax_e.fill_between(
        smooth_plot.eventTimeMonths.to_numpy(float),
        smooth_plot.ciLowCoefficient.to_numpy(float),
        smooth_plot.ciHighCoefficient.to_numpy(float),
        color=COLORS["vermillion"], alpha=0.12, lw=0,
    )
    ax_e.axhline(0, color=COLORS["black"], ls=(0, (2, 2)), lw=0.65)
    ax_e.set_xlim(0, 60)
    residual_limits = residual_plot.scaledSchoenfeldBeta.quantile([0.01, 0.99])
    ax_e.set_ylim(float(residual_limits.iloc[0]) - 0.4, float(residual_limits.iloc[1]) + 0.4)
    ax_e.set_xlabel("Months after reported survival origin")
    ax_e.set_ylabel("Scaled Schoenfeld β")
    ax_e.set_title("LUAD KEAP1–STK11 residual diagnostic", loc="left", fontsize=5.5, pad=3)
    ax_e.legend(
        **LEGEND_BOX, loc="upper right", bbox_to_anchor=(1.0, 1.005),
        fontsize=3.8,
    )
    panel_label(ax_e, "e", x=-0.12, y=1.045)
    residual_plot.to_csv(source / "figureS4_panel_e_scaled_schoenfeld.csv", index=False)
    smooth_plot.to_csv(source / "figureS4_panel_e_time_varying_smooth.csv", index=False)

    # f -- compare the complete pooled stratified estimates with random-effects
    # synthesis of study-specific contrasts for the principal cancer contexts.
    meta = extended["survival_study_meta_analysis.csv"]
    joint = extended["survival_joint_state_and_interaction_summary.csv"]
    meta_specs = [
        ("LUAD", "KEAP1", "STK11", "LUAD · KEAP1–STK11"),
        ("PAAD", "KRAS", "TP53", "PAAD · KRAS–TP53"),
        ("BRCA", "PIK3CA", "TP53", "BRCA · PIK3CA–TP53"),
        ("LUAD", "STK11", "TP53", "LUAD · STK11–TP53"),
        ("LUAD", "KRAS", "TP53", "LUAD · KRAS–TP53"),
    ]
    meta_frames: list[pd.DataFrame] = []
    for plot_row, (cancer, gene_a, gene_b, display_label) in enumerate(meta_specs):
        selected_meta = meta.loc[
            meta.scope.eq("cancer-specific") & meta.cancer.eq(cancer)
            & meta.geneA.eq(gene_a) & meta.geneB.eq(gene_b)
        ].copy()
        selected_joint = joint.loc[
            joint.scope.eq("cancer-specific") & joint.cancer.eq(cancer)
            & joint.geneA.eq(gene_a) & joint.geneB.eq(gene_b),
            ["jointStateHazardRatio", "jointStateCiLow", "jointStateCiHigh"],
        ]
        if len(selected_meta) != 1 or len(selected_joint) != 1:
            raise AssertionError(f"Supplementary Figure 4f requires {display_label}")
        selected_meta["pooledHazardRatio"] = selected_joint.jointStateHazardRatio.iloc[0]
        selected_meta["pooledCiLow"] = selected_joint.jointStateCiLow.iloc[0]
        selected_meta["pooledCiHigh"] = selected_joint.jointStateCiHigh.iloc[0]
        selected_meta["displayContext"] = display_label
        selected_meta["plotRow"] = plot_row
        meta_frames.append(selected_meta)
    meta_plot = pd.concat(meta_frames, ignore_index=True)
    for estimate, low, high, offset, colour, marker, label in (
        ("pooledHazardRatio", "pooledCiLow", "pooledCiHigh", -0.11, COLORS["blue"], "o", "Pooled stratified Cox"),
        ("randomHazardRatio", "randomCiLow", "randomCiHigh", 0.11, COLORS["purple"], "s", "Study random effects"),
    ):
        y = meta_plot.plotRow + offset
        ax_f.hlines(y, meta_plot[low], meta_plot[high], color=colour, lw=0.9)
        ax_f.scatter(meta_plot[estimate], y, color=colour, marker=marker, s=17, label=label, zorder=3)
    ax_f.axvline(1, color=COLORS["black"], ls=(0, (2, 2)), lw=0.65)
    ax_f.set_xscale("log")
    ax_f.set_xticks([0.5, 1, 2, 4, 8], ["0.5", "1", "2", "4", "8"])
    ax_f.tick_params(axis="x", which="minor", labelbottom=False)
    ax_f.set_yticks(
        meta_plot.plotRow,
        [f"{label}\nI²={i2:.0f}%" for label, i2 in zip(meta_plot.displayContext, meta_plot.iSquaredPercent)],
        fontsize=3.7,
    )
    ax_f.set_ylim(len(meta_plot) - 0.48, -0.52)
    ax_f.set_xlabel("A+B vs A−/B− HR (95% CI)")
    ax_f.set_title("Study-level heterogeneity", loc="left", fontsize=5.5, pad=3)
    ax_f.legend(**LEGEND_BOX, loc="lower right", fontsize=3.55, handletextpad=0.3)
    panel_label(ax_f, "f", x=-0.12, y=1.045)
    meta_plot.to_csv(source / "figureS4_panel_f_study_meta_analysis.csv", index=False)

    fig.subplots_adjust(
        left=0.195, right=0.98, top=0.955, bottom=0.095, wspace=0.64, hspace=0.54
    )
    save_figure(fig, supplementary / "figureS4_survival_diagnostics")
    plt.close(fig)


def render_saved_survival_figures() -> None:
    """Rebuild Figure 9 and Supplementary Figure 4 from frozen result tables."""
    samples = pd.read_parquet(PROCESSED / "analysis_samples_curated.parquet")
    samples = samples.loc[samples.analysisEligible].copy()
    clinical = build_curated_clinical(samples)
    conflict_specimens, _ = assay_discordance_specimen_audit(samples)
    clinical["hasAssayScopeConflict"] = [
        (str(study_id), str(sample_id)) in conflict_specimens
        for study_id, sample_id in zip(clinical.studyId, clinical.sampleId)
    ]
    latest_genes = sorted(
        {
            gene
            for _, _, gene_a, gene_b, _ in FIGURE9_LATEST_CONTEXTS
            for gene in (gene_a, gene_b)
        }
        | {FIGURE9_SECONDARY_KM_CONTEXT[2], FIGURE9_SECONDARY_KM_CONTEXT[3]}
    )
    flags = callability_and_mutation_flags(samples, latest_genes)
    contexts, latest_extended = run_figure9_latest_diagnostics(clinical, flags)
    screen_results = pd.read_csv(TABLES / "survival_unbiased_screen.csv")
    models = pd.read_csv(TABLES / "survival_curated_pair_models.csv")
    groups = pd.read_csv(TABLES / "survival_curated_pair_groups.csv")
    audit = pd.read_csv(TABLES / "survival_curated_cohort_audit.csv")
    missing = pd.read_csv(TABLES / "survival_curated_missingness.csv")
    endpoint_audit = pd.read_csv(TABLES / "survival_endpoint_eligibility.csv")
    required_extended = set(EXTENDED_SURVIVAL_TABLES) | {
        "survival_joint_state_and_interaction_summary.csv",
    }
    extended = {
        filename: pd.read_csv(TABLES / filename)
        for filename in sorted(required_extended)
    }
    make_figure(
        contexts,
        models,
        groups,
        audit.loc[audit.nModelPatients.ge(MIN_CONTEXT_N)],
        latest_extended,
        screen_results,
    )
    make_diagnostics(models, missing, endpoint_audit, extended, screen_results)
    print("Rebuilt Figure 9 and Supplementary Figure 4 from frozen survival results.")


def main(*, render_figures: bool = True) -> None:
    samples = pd.read_parquet(PROCESSED / "analysis_samples_curated.parquet")
    samples = samples.loc[samples.analysisEligible].copy()
    resolved_lung_codes = {"LUAD", "LUSC"}
    if not resolved_lung_codes.issubset(set(samples.broadCancerCode.dropna().astype(str))):
        raise RuntimeError(
            "The authoritative cohort does not yet contain both resolved LUAD and LUSC categories. "
            "Rebuild the lung-cancer taxonomy before freezing survival results."
        )
    clinical = build_curated_clinical(samples)
    conflict_specimens, conflict_specimen_audit = assay_discordance_specimen_audit(
        samples
    )
    clinical["hasAssayScopeConflict"] = [
        (str(study_id), str(sample_id)) in conflict_specimens
        for study_id, sample_id in zip(clinical.studyId, clinical.sampleId)
    ]
    screen_specifications, genes = survival_screen_specifications()
    flags = callability_and_mutation_flags(samples, genes)
    screen_results, screen_audit = run_unbiased_survival_screen(
        clinical,
        flags,
        screen_specifications,
    )
    print(
        "Comprehensive survival screen: "
        f"{int(screen_audit.eligible.sum()):,} eligible contexts from "
        f"{len(screen_audit):,} mutation-frequency-qualified candidates; "
        f"{int(screen_results.globalFdr.lt(0.05).sum()):,} target coefficients at global FDR < 0.05."
    )

    # Mutually exclusive endpoint eligibility audit. The primary survival cohort
    # requires a recognized status and a finite, strictly positive time.
    linked = clinical.copy()
    finite_time = linked.months.notna() & np.isfinite(linked.months)
    endpoint_category = np.select(
        [
            ~finite_time,
            linked.event.isna(),
            finite_time & linked.months.lt(0),
            finite_time & linked.months.eq(0),
            finite_time & linked.months.gt(0) & linked.event.notna(),
        ],
        [
            "missing or non-finite OS time",
            "missing or unrecognised OS status",
            "negative OS time",
            "zero OS time",
            "positive OS time",
        ],
        default="other invalid endpoint",
    )
    endpoint_rows = [
        {
            "eligibilityState": "selected tumour without linked study-specific patient record",
            "nPatients": int(len(samples) - len(linked)),
            "primaryEligible": False,
            "zeroTimeSensitivityEligible": False,
        }
    ]
    for category in (
        "missing or non-finite OS time",
        "missing or unrecognised OS status",
        "negative OS time",
        "zero OS time",
        "positive OS time",
        "other invalid endpoint",
    ):
        endpoint_rows.append(
            {
                "eligibilityState": category,
                "nPatients": int((endpoint_category == category).sum()),
                "primaryEligible": category == "positive OS time",
                "zeroTimeSensitivityEligible": category in {"positive OS time", "zero OS time"},
            }
        )
    endpoint_audit = pd.DataFrame(endpoint_rows)
    endpoint_audit["endpoint"] = (
        "Overall-survival duration and vital status as reported by contributing studies"
    )
    endpoint_audit["timeUnit"] = "months"
    endpoint_audit["zeroTimeSensitivityValueMonths"] = ZERO_TIME_EPSILON_MONTHS

    model_datasets: list[pd.DataFrame] = []
    model_specs: list[dict[str, object]] = []
    group_frames: list[pd.DataFrame] = []
    missing_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    stratum_frames: list[pd.DataFrame] = []
    context_frames: dict[tuple[str, str, str], pd.DataFrame] = {}

    for cancer, gene_a, gene_b in CONTEXTS:
        data, zero_data, audit, stratum_audit = context_data(
            clinical, flags, cancer, gene_a, gene_b, scope="cancer-specific"
        )
        if cancer == "LUAD" and not data.broadCancerCode.eq("LUAD").all():
            raise AssertionError("The LUAD survival context contains non-LUAD cases")
        audit_rows.append(audit)
        stratum_frames.append(stratum_audit)
        if len(data) < MIN_CONTEXT_N or data.event.sum() < MIN_CONTEXT_EVENTS:
            print(f"Skip {gene_a}-{gene_b} {cancer}: n={len(data)}, events={int(data.event.sum())}")
            continue
        groups, missing = summarize_context(
            data, "cancer-specific", cancer, gene_a, gene_b
        )
        group_frames.append(groups)
        missing_rows.append(missing)
        context_frames[(cancer, gene_a, gene_b)] = data
        datasets, specs = build_model_inputs(
            data, zero_data, "cancer-specific", cancer, gene_a, gene_b
        )
        model_datasets.extend(datasets)
        model_specs.extend(specs)

    for gene_a, gene_b in PANCAN_CONTEXTS:
        data, zero_data, audit, stratum_audit = context_data(
            clinical, flags, "PAN-CANCER", gene_a, gene_b, scope="pan-cancer"
        )
        audit_rows.append(audit)
        stratum_frames.append(stratum_audit)
        if len(data) < MIN_CONTEXT_N or data.event.sum() < MIN_CONTEXT_EVENTS:
            print(f"Skip pan-cancer {gene_a}-{gene_b}: n={len(data)}, events={int(data.event.sum())}")
            continue
        groups, missing = summarize_context(
            data, "pan-cancer", "PAN-CANCER", gene_a, gene_b
        )
        group_frames.append(groups)
        missing_rows.append(missing)
        context_frames[("PAN-CANCER", gene_a, gene_b)] = data
        datasets, specs = build_model_inputs(
            data, zero_data, "pan-cancer", "PAN-CANCER", gene_a, gene_b
        )
        model_datasets.extend(datasets)
        model_specs.extend(specs)

    if not model_specs:
        raise RuntimeError("No selected cancer-specific or pan-cancer survival context was estimable")
    models = run_cox_models(model_datasets, model_specs)
    groups = pd.concat(group_frames, ignore_index=True)
    missing = pd.DataFrame(missing_rows)
    audit = pd.DataFrame(audit_rows)
    stratum_audit = pd.concat(stratum_frames, ignore_index=True)
    sensitivity = sensitivity_summary(models)
    extended = run_extended_survival_diagnostics(
        model_datasets,
        model_specs,
        models,
    )

    # Internal numerical invariants: clustering changes uncertainty, not the
    # Efron partial-likelihood coefficient; primary cohorts contain no zero or
    # negative times; and robust rows are not estimated below the declared
    # study-cluster threshold.
    primary_data = pd.concat(
        [frame for frame in model_datasets if frame.datasetId.iloc[0].endswith("_positive")],
        ignore_index=True,
    )
    if not primary_data.months.gt(0).all():
        raise AssertionError("Primary survival-model data contain non-positive times")
    conventional = models.loc[
        models.model.eq("four-group primary") & models.term.ne("MODEL"),
        ["context", "term", "coefficient"],
    ]
    clustered = models.loc[
        models.model.eq("four-group study-clustered sensitivity")
        & models.term.ne("MODEL")
        & models.fitStatus.str.startswith("estimated"),
        ["context", "term", "coefficient"],
    ]
    coefficient_check = conventional.merge(
        clustered, on=["context", "term"], suffixes=("ModelBased", "Clustered")
    )
    if len(coefficient_check):
        max_difference = (
            coefficient_check.coefficientModelBased
            - coefficient_check.coefficientClustered
        ).abs().max()
        if max_difference > 1e-10:
            raise AssertionError(
                f"Model-based and clustered Efron coefficients differ by {max_difference}"
            )
    invalid_cluster = models.loc[
        models.varianceEstimator.eq("study-clustered sandwich")
        & models.nStudies.lt(MIN_CLUSTER_STUDIES)
        & models.fitStatus.str.startswith("estimated")
    ]
    if len(invalid_cluster):
        raise AssertionError("A study-clustered model was estimated below the declared threshold")

    models.to_csv(TABLES / "survival_curated_pair_models.csv", index=False)
    groups.to_csv(TABLES / "survival_curated_pair_groups.csv", index=False)
    missing.to_csv(TABLES / "survival_curated_missingness.csv", index=False)
    audit.to_csv(TABLES / "survival_curated_cohort_audit.csv", index=False)
    stratum_audit.to_csv(TABLES / "survival_curated_stratum_audit.csv", index=False)
    endpoint_audit.to_csv(TABLES / "survival_endpoint_eligibility.csv", index=False)
    sensitivity.to_csv(TABLES / "survival_model_sensitivity_summary.csv", index=False)
    conflict_specimen_audit.to_csv(
        TABLES / "survival_assay_discordance_specimen_audit.csv", index=False
    )

    required = {("LUAD", "KEAP1", "STK11"), ("PAAD", "KRAS", "TP53")}
    if not required.issubset(context_frames):
        absent = sorted(required - set(context_frames))
        raise RuntimeError(f"Prespecified KM contexts were not estimable: {absent}")
    if render_figures:
        latest_context_frames, latest_figure_extended = run_figure9_latest_diagnostics(
            clinical, flags
        )
        make_figure(
            latest_context_frames,
            models,
            groups,
            audit.loc[audit.nModelPatients.ge(MIN_CONTEXT_N)],
            latest_figure_extended,
            screen_results,
        )
        make_diagnostics(models, missing, endpoint_audit, extended, screen_results)

    print(f"Curated selected samples: {len(samples):,}")
    print(
        f"Matched selected samples with a patient identifier: {clinical.patientId.notna().sum():,}; "
        f"positive OS: {clinical.validPositiveOs.sum():,}; "
        f"zero-time OS excluded from primary: {clinical.zeroOsTime.sum():,}."
    )
    print("Primary model-based Efron A+B versus A−/B− estimates:")
    primary_both = (
        models.model.eq("four-group primary")
        & models.term.eq("Both")
        & models.p.notna()
    )
    print(
        models.loc[
            primary_both,
            [
                "context", "scope", "nPatients", "nEvents", "nStudies", "nStrata",
                "hazardRatio", "ciLow", "ciHigh", "p", "fdr", "phTestP",
            ],
        ].to_string(index=False, float_format=lambda value: f"{value:.4g}")
    )
    print(
        f"Verified {len(extended['survival_extended_diagnostic_audit.csv'])} primary "
        "A+B models and wrote PH, time-varying, piecewise, RMST, primary-tumour, "
        "study-specific, meta-analysis and leave-one-study-out diagnostics."
    )
    if render_figures:
        print("Wrote Figure 9 and Supplementary Figure S4.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit curated survival models and extended diagnostics."
    )
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help="Regenerate survival result tables without rewriting figure files.",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Rebuild Figure 9 and Supplementary Figure 4 from frozen result tables.",
    )
    arguments = parser.parse_args()
    if arguments.tables_only and arguments.figures_only:
        parser.error("--tables-only and --figures-only cannot be combined")
    if arguments.figures_only:
        render_saved_survival_figures()
    else:
        main(render_figures=not arguments.tables_only)
