"""Assay-covered, cancer-specific gene-pair association analyses.

Each pair is evaluated on its exact shared assay-covered denominator within cancer
type, with WES/WGS and targeted-panel estimates reported as assay-specific sensitivity
populations.

The first pass uses Fisher's exact test on jointly profiled samples. Every screened
pair is then re-estimated with the same validated Cochran--Mantel--Haenszel (CMH)
arithmetic under three prespecified conditioning schemes:

1. primary: detailed cancer code, study and exact assay/panel, without mutation burden;
2. sensitivity: the primary strata plus leave-two-out background-burden quintile; and
3. diagnostic: the primary strata plus total-burden quintile.

For a tested pair A/B, leave-two-out burden is total mutation burden minus the mutation
indicators for A and B. It therefore cannot condition on a total containing the two
outcomes whose association is being estimated. Full-cohort, WES/WGS and targeted-panel
estimates are reported for every pair. Study-specific and leave-one-study-out estimates
are additionally written for the displayed network and principal reference contexts.
"""
from __future__ import annotations

from itertools import combinations
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from scipy.stats import chi2, fisher_exact, spearmanr
from statsmodels.stats.contingency_tables import StratifiedTable
from statsmodels.stats.multitest import multipletests

from callability import partition_callable_mutations
from config import FIGURES, PROCESSED, TABLES
from nature_style import COLORS, apply as apply_style, figsize, panel_label, save_figure

warnings.filterwarnings("ignore")

MIN_CANCER = 300
MIN_GENE_FREQ = 0.02
MIN_GENE_MUT = 10
MIN_PAIR_DENOM = 200
MIN_ASSAY_DENOM = 80
MIN_ASSAY_MUT = 5
MAX_DRIVER_GENES = 120

KNOWN_GENES = {
    "EGFR", "KRAS", "BRAF", "NRAS", "HRAS", "STK11", "KEAP1", "TP53",
    "PIK3CA", "PTEN", "APC", "CTNNB1", "IDH1", "ATRX", "CALR", "JAK2", "NFE2L2",
    "SF3B1", "SRSF2", "RUNX1", "NPM1", "TET2", "ASXL1",
}
CANONICAL_PAIRS = {
    tuple(sorted(x))
    for x in [
        ("EGFR", "KRAS"), ("BRAF", "KRAS"), ("CALR", "JAK2"),
        ("SF3B1", "SRSF2"), ("RUNX1", "NPM1"), ("IDH1", "PTEN"),
        ("PIK3CA", "PTEN"), ("CTNNB1", "TP53"), ("KRAS", "TP53"),
        ("STK11", "TP53"), ("KEAP1", "STK11"), ("BRAF", "NRAS"),
        ("PIK3CA", "PIK3R1"), ("PTEN", "TP53"), ("KEAP1", "NFE2L2"),
    ]
}

# Literature-anchored cancer contexts for the compact reference-pair panel. Pair-wide
# screening results remain in the source table; this list prevents an association in an
# arbitrary cancer group from being presented as a canonical biological positive control.
REFERENCE_CONTEXTS = {
    ("EGFR-KRAS", "LUAD"),
    ("BRAF-KRAS", "COADREAD"),
    ("BRAF-NRAS", "SKCM"),
    ("CALR-JAK2", "MPN"),
    ("NPM1-RUNX1", "AML"),
    ("SF3B1-SRSF2", "MDS"),
    # GBM is used as the explicit CNS context because this association is significant
    # and cross-assay concordant in the final, histology-specific Stage 17 screen.
    ("IDH1-PTEN", "GBM"),
    ("PTEN-TP53", "UCEC"),
    ("PIK3CA-PIK3R1", "UCEC"),
    ("CTNNB1-TP53", "UCEC"),
    ("KEAP1-NFE2L2", "LUSC"),
    ("KEAP1-STK11", "LUAD"),
    ("KRAS-TP53", "PAAD"),
    ("KRAS-TP53", "LUAD"),
    ("STK11-TP53", "LUAD"),
}

PRINCIPAL_HETEROGENEITY_CONTEXTS = {
    ("EGFR-KRAS", "LUAD"),
    ("KEAP1-STK11", "LUAD"),
    ("BRAF-KRAS", "COADREAD"),
    ("CALR-JAK2", "MPN"),
}

SPECIFICATIONS = ("noBurden", "leaveTwoOut", "totalBurden")
ASSAYS = ("full", "wes", "panel")
EFFECT_STABILITY_LOG2_TOLERANCE = 1.0
STRONG_EFFECT_LOG2_THRESHOLD = 2.0


def odds_ci(n11: int, n10: int, n01: int, n00: int) -> tuple[float, float, float]:
    # Haldane-Anscombe correction keeps estimates and CIs finite for complete exclusivity.
    a, b, c, d = (n11 + 0.5, n10 + 0.5, n01 + 0.5, n00 + 0.5)
    log_or = np.log(a * d / (b * c))
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return float(np.exp(log_or)), float(np.exp(log_or - 1.96 * se)), float(np.exp(log_or + 1.96 * se))


def pair_stats(joint: set[str], a_set: set[str], b_set: set[str], min_denom: int, min_mut: int):
    n = len(joint)
    if n < min_denom:
        return None
    aa, bb = a_set & joint, b_set & joint
    n_a, n_b = len(aa), len(bb)
    if min(n_a, n_b) < min_mut or max(n_a, n_b) > n - min_mut:
        return None
    n11 = len(aa & bb)
    n10, n01 = n_a - n11, n_b - n11
    n00 = n - n11 - n10 - n01
    _, p = fisher_exact([[n11, n10], [n01, n00]], alternative="two-sided")
    odds, lo, hi = odds_ci(n11, n10, n01, n00)
    return {
        "n": n, "nA": n_a, "nB": n_b, "nBoth": n11,
        "oddsRatio": odds, "ciLow": lo, "ciHigh": hi, "p": p,
    }


def candidate_genes(panel: pd.DataFrame) -> tuple[list[int], dict[int, str]]:
    prevalence = pd.read_csv(TABLES / "gene_frequencies_curated.csv")
    evidence = pd.read_csv(TABLES / "cmc_evidence_curated.csv")
    x = prevalence.merge(evidence[["gene", "pctTierMatched"]], on="gene", how="left")
    x["driverScore"] = x.freqPct * (0.25 + 0.75 * x.pctTierMatched.fillna(0) / 100)
    chosen = set(x.nlargest(MAX_DRIVER_GENES, "driverScore").gene) | KNOWN_GENES
    p = panel[panel.hugoSymbol.isin(chosen)].copy()
    return p.entrezGeneId.astype(int).tolist(), p.set_index("entrezGeneId").hugoSymbol.to_dict()


def screen_pairs() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    panel = pd.read_csv(PROCESSED / "gene_panel.csv")
    driver_ents, ent2sym = candidate_genes(panel)
    samples = pd.read_parquet(PROCESSED / "analysis_samples_curated.parquet")
    samples = samples[samples.analysisEligible].copy()
    assay = pd.read_parquet(PROCESSED / "sample_assay.parquet").merge(
        samples[["sampleId", "studyId", "broadCancerCode", "analysisCancerCode"]],
        on=["sampleId", "studyId"], how="inner",
    )
    assay["assayGroup"] = np.where(assay.assayType.str.startswith("WES/WGS"), "WES/WGS", "Targeted panel")
    assay["panelStratum"] = np.where(
        assay.assayGroup.eq("WES/WGS"), "WES/WGS", assay.genePanelId.fillna("Targeted-unknown")
    )
    membership = pd.read_parquet(PROCESSED / "panel_gene_membership.parquet")
    member_panels = membership.groupby("entrezGeneId")["genePanelId"].apply(set).to_dict()
    mutations = pd.read_parquet(PROCESSED / "mutations_curated.parquet", columns=["sampleId", "entrezGeneId"])
    mutations = mutations.drop_duplicates(["sampleId", "entrezGeneId"])
    burden = mutations.groupby("sampleId").size().rename("mutationBurden")
    assay = assay.merge(burden, on="sampleId", how="left").fillna({"mutationBurden": 0})
    assay["burdenPct"] = assay.groupby(["studyId", "panelStratum"])["mutationBurden"].rank(pct=True)

    sizes = assay.broadCancerCode.value_counts()
    cancers = sizes[sizes >= MIN_CANCER].index
    rows = []
    context = {}
    for cancer in cancers:
        info = assay[assay.broadCancerCode == cancer].copy().set_index("sampleId", drop=False)
        all_samples = set(info.index)
        wes_samples = set(info.index[info.assayGroup == "WES/WGS"])
        panel_samples = {
            pid: set(group.index)
            for pid, group in info[info.assayGroup == "Targeted panel"].groupby("genePanelId")
        }
        mut_sub = mutations[mutations.sampleId.isin(all_samples) & mutations.entrezGeneId.isin(driver_ents)]
        mut_sets = mut_sub.groupby("entrezGeneId").sampleId.apply(set).to_dict()
        coverage = {}
        coverage_wes = {}
        coverage_panel = {}
        keep = []
        for ent in driver_ents:
            target = set().union(*(panel_samples.get(pid, set()) for pid in member_panels.get(ent, set()))) if member_panels.get(ent) else set()
            coverage_wes[ent] = wes_samples
            coverage_panel[ent] = target
            coverage[ent] = wes_samples | target
            nprof = len(coverage[ent])
            nmut = len(mut_sets.get(ent, set()) & coverage[ent])
            if nprof >= MIN_PAIR_DENOM and nmut >= MIN_GENE_MUT and nmut / nprof >= MIN_GENE_FREQ:
                keep.append(ent)
        context[cancer] = {"info": info, "coverage": coverage, "mut_sets": mut_sets}
        for a, b in combinations(keep, 2):
            full = pair_stats(coverage[a] & coverage[b], mut_sets.get(a, set()), mut_sets.get(b, set()), MIN_PAIR_DENOM, MIN_GENE_MUT)
            if full is None:
                continue
            wes = pair_stats(coverage_wes[a] & coverage_wes[b], mut_sets.get(a, set()), mut_sets.get(b, set()), MIN_ASSAY_DENOM, MIN_ASSAY_MUT)
            target = pair_stats(coverage_panel[a] & coverage_panel[b], mut_sets.get(a, set()), mut_sets.get(b, set()), MIN_ASSAY_DENOM, MIN_ASSAY_MUT)
            ga, gb = ent2sym.get(a, str(a)), ent2sym.get(b, str(b))
            if ga > gb:
                ga, gb, a, b = gb, ga, b, a
            row = {"cancer": cancer, "entrezA": a, "entrezB": b, "geneA": ga, "geneB": gb}
            for prefix, stat in (("full", full), ("wes", wes), ("panel", target)):
                for key in ("n", "nA", "nB", "nBoth", "oddsRatio", "ciLow", "ciHigh", "p"):
                    row[f"{prefix}_{key}"] = stat[key] if stat is not None else np.nan
            rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No jointly profiled pair tests were produced")
    for prefix in ("full", "wes", "panel"):
        out[f"{prefix}_fdr"] = np.nan
        for cancer, idx in out.groupby("cancer").groups.items():
            valid = out.loc[idx, f"{prefix}_p"].notna()
            use = out.loc[idx].index[valid]
            if len(use):
                out.loc[use, f"{prefix}_fdr"] = multipletests(out.loc[use, f"{prefix}_p"], method="fdr_bh")[1]
    out["direction"] = np.where(out.full_oddsRatio < 1, "mutually exclusive", "co-occurring")
    out["pair"] = out.geneA + "-" + out.geneB
    out["sameAssayDirection"] = (
        np.sign(np.log(out.wes_oddsRatio)) == np.sign(np.log(out.panel_oddsRatio))
    )
    out["replicated"] = (
        out.sameAssayDirection
        & (out.wes_fdr < 0.10)
        & (out.panel_p < 0.05)
    )
    out.to_csv(TABLES / "cooccurrence_curated_jointly_profiled.csv", index=False)
    return out, assay, context


def _new_accumulator(n_genes: int) -> dict[str, np.ndarray]:
    return {
        key: np.zeros((n_genes, n_genes), dtype=float)
        for key in (
            "adns", "bcns", "va1", "vamid", "va3", "diff", "diffvar",
            "nStrata", "nInformativeStrata",
        )
    }


def _accumulate_stratum(acc: dict[str, np.ndarray], x: np.ndarray, covered: np.ndarray) -> None:
    """Accumulate vectorised Mantel-Haenszel/RBG quantities for one stratum."""
    n = len(x)
    if n < 2 or covered.sum() < 2:
        return
    xx = x.astype(np.int64, copy=False)
    a = xx.T @ xx
    ng = xx.sum(axis=0)
    cov = covered.astype(np.int64)
    joint = np.outer(cov, cov).astype(bool)
    nn = n * joint.astype(np.int64)
    ab = np.outer(ng, cov)
    ac = np.outer(cov, ng)
    b = ab - a
    c = ac - a
    d = nn - a - b - c

    # Unshifted CMH score test (conditioning strata are supplied by the caller).
    valid_test = joint & (nn > 1)
    expected = np.zeros_like(a, dtype=float)
    variance = np.zeros_like(a, dtype=float)
    expected[valid_test] = ab[valid_test] * ac[valid_test] / nn[valid_test]
    variance[valid_test] = (
        ab[valid_test]
        * ac[valid_test]
        * (nn[valid_test] - ab[valid_test])
        * (nn[valid_test] - ac[valid_test])
        / (nn[valid_test] ** 2 * (nn[valid_test] - 1))
    )
    acc["diff"][valid_test] += a[valid_test] - expected[valid_test]
    acc["diffvar"][valid_test] += variance[valid_test]
    acc["nStrata"][valid_test] += 1
    acc["nInformativeStrata"][valid_test & (variance > 0)] += 1

    # Unshifted Mantel-Haenszel odds ratio and Robins-Breslow-Greenland variance.
    # Sparse zero cells contribute zero cross-products and do not require a separate
    # continuity correction in every stratum. Applying 0.5 per sparse stratum strongly
    # attenuates effects when hundreds of strata are present and makes the interval
    # estimate incoherent with the unshifted MH score test. If the pooled numerator or
    # denominator is globally zero, _finish_accumulator marks the OR/CI non-estimable.
    af, bf, cf, df, nf = [z.astype(float) for z in (a, b, c, d, nn)]
    valid = joint & (nf > 0)
    ad = af * df
    bc = bf * cf
    apd = af + df
    acc["adns"][valid] += ad[valid] / nf[valid]
    acc["bcns"][valid] += bc[valid] / nf[valid]
    acc["va1"][valid] += apd[valid] * ad[valid] / nf[valid] ** 2
    acc["vamid"][valid] += (
        apd[valid] * bc[valid] / nf[valid] ** 2
        + (1 - apd[valid] / nf[valid]) * ad[valid] / nf[valid]
    )
    acc["va3"][valid] += (1 - apd[valid] / nf[valid]) * bc[valid] / nf[valid]


def _finish_accumulator(acc: dict[str, np.ndarray]) -> tuple[np.ndarray, ...]:
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        odds = acc["adns"] / acc["bcns"]
        variance = 0.5 * (
            acc["va1"] / acc["adns"] ** 2
            + acc["vamid"] / (acc["adns"] * acc["bcns"])
            + acc["va3"] / acc["bcns"] ** 2
        )
        se = np.sqrt(np.clip(variance, 0, None))
        lo = np.exp(np.log(odds) - 1.96 * se)
        hi = np.exp(np.log(odds) + 1.96 * se)
        stat = acc["diff"] ** 2 / acc["diffvar"]
        p = chi2.sf(stat, 1)
    # A pooled zero numerator or denominator implies a zero/infinite MH estimate. It is
    # not made finite with pseudo-counts; report the OR and CI as non-estimable. The
    # unshifted score-test P value can remain available as a separate test result.
    estimable = (acc["adns"] > 0) & (acc["bcns"] > 0)
    for arr in (odds, lo, hi):
        arr[(~estimable) | (~np.isfinite(arr))] = np.nan
    p[~np.isfinite(p)] = np.nan
    return odds, lo, hi, p, acc["nStrata"], acc["nInformativeStrata"]


def _cmh_from_counts(counts: np.ndarray, stratum_mask: np.ndarray | None = None) -> dict[str, float]:
    """Return the validated unshifted CMH/RBG estimate from stratum 2x2 counts.

    ``counts`` columns are ordered n00, n01, n10 and n11. The formulae are identical
    to :func:`_accumulate_stratum` and :func:`_finish_accumulator`; this scalar form is
    used only where pair-specific leave-two-out strata prevent matrix accumulation.
    """
    if counts.size == 0:
        counts = np.empty((0, 4), dtype=float)
    else:
        counts = np.asarray(counts, dtype=float)
    if stratum_mask is not None:
        counts = counts[np.asarray(stratum_mask, dtype=bool)]
    if not len(counts):
        return {
            "or": np.nan, "ciLow": np.nan, "ciHigh": np.nan, "p": np.nan,
            "nStrata": 0, "nInformativeStrata": 0, "logOrSe": np.nan,
        }

    d, c, b, a = counts.T
    n = counts.sum(axis=1)
    keep = n > 1
    a, b, c, d, n = (z[keep] for z in (a, b, c, d, n))
    if not len(n):
        return {
            "or": np.nan, "ciLow": np.nan, "ciHigh": np.nan, "p": np.nan,
            "nStrata": 0, "nInformativeStrata": 0, "logOrSe": np.nan,
        }

    ab = a + b
    ac = a + c
    expected = ab * ac / n
    score_variance = ab * ac * (n - ab) * (n - ac) / (n**2 * (n - 1))
    informative = score_variance > 0
    diff = np.sum(a - expected)
    diffvar = np.sum(score_variance)
    p = float(chi2.sf(diff**2 / diffvar, 1)) if diffvar > 0 else np.nan

    ad = a * d
    bc = b * c
    apd = a + d
    adns = np.sum(ad / n)
    bcns = np.sum(bc / n)
    if adns <= 0 or bcns <= 0:
        odds = lo = hi = log_or_se = np.nan
    else:
        va1 = np.sum(apd * ad / n**2)
        vamid = np.sum(apd * bc / n**2 + (1 - apd / n) * ad / n)
        va3 = np.sum((1 - apd / n) * bc / n)
        log_or_variance = 0.5 * (
            va1 / adns**2 + vamid / (adns * bcns) + va3 / bcns**2
        )
        log_or_se = float(np.sqrt(max(log_or_variance, 0)))
        odds = float(adns / bcns)
        lo = float(np.exp(np.log(odds) - 1.96 * log_or_se))
        hi = float(np.exp(np.log(odds) + 1.96 * log_or_se))
    return {
        "or": odds, "ciLow": lo, "ciHigh": hi, "p": p,
        "nStrata": int(len(n)),
        "nInformativeStrata": int(informative.sum()),
        "logOrSe": log_or_se,
    }


def _pair_counts(
    xa: np.ndarray,
    xb: np.ndarray,
    strata: np.ndarray,
    valid: np.ndarray,
    n_strata: int,
) -> np.ndarray:
    """Construct all pair-specific stratum tables with a single bincount call."""
    valid = np.asarray(valid, dtype=bool)
    if not valid.any():
        return np.zeros((n_strata, 4), dtype=np.int64)
    state = 2 * xa[valid].astype(np.int64) + xb[valid].astype(np.int64)
    index = 4 * strata[valid].astype(np.int64) + state
    return np.bincount(index, minlength=4 * n_strata).reshape(n_strata, 4)


def _pair_estimates_by_assay(
    xa: np.ndarray,
    xb: np.ndarray,
    strata: np.ndarray,
    valid: np.ndarray,
    stratum_is_wes: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Estimate one pair in the full cohort and both assay classes."""
    counts = _pair_counts(xa, xb, strata, valid, len(stratum_is_wes))
    return {
        "full": _cmh_from_counts(counts),
        "wes": _cmh_from_counts(counts, stratum_is_wes),
        "panel": _cmh_from_counts(counts, ~stratum_is_wes),
    }


def _leave_two_out_design(info: pd.DataFrame) -> dict[str, np.ndarray | int]:
    """Precompute indices used for pair-specific background-burden quintiles."""
    base_index = pd.MultiIndex.from_frame(
        info[["analysisCancerCode", "studyId", "panelStratum"]],
        names=["analysisCancerCode", "studyId", "panelStratum"],
    )
    base_code, base_levels = pd.factorize(base_index, sort=False)
    base_is_wes = np.asarray(
        [level[2] == "WES/WGS" for level in base_levels], dtype=bool
    )
    burden_index = pd.MultiIndex.from_frame(
        info[["studyId", "panelStratum"]], names=["studyId", "panelStratum"]
    )
    burden_group, burden_levels = pd.factorize(burden_index, sort=False)
    burden = info.mutationBurden.to_numpy(dtype=np.int64)
    burden_group_size = np.bincount(burden_group, minlength=len(burden_levels))
    return {
        "baseCode": base_code,
        "baseIsWes": base_is_wes,
        "burdenGroup": burden_group,
        "burdenGroupSize": burden_group_size,
        "burden": burden,
        "burdenLevels": int(burden.max()) + 1,
    }


def _leave_two_out_pair_strata(
    xa: np.ndarray,
    xb: np.ndarray,
    design: dict[str, np.ndarray | int],
) -> tuple[np.ndarray, np.ndarray]:
    """Construct study–assay-specific quintiles of B_total-I(A)-I(B).

    Average ranks reproduce ``pandas.rank(pct=True)`` for tied integer burdens. The
    rank calculation is pair-specific, whereas the exact histology/study/panel base
    strata are fixed. This uses five burden bins while removing both tested-gene
    indicators from the conditioning covariate.
    """
    burden = np.asarray(design["burden"], dtype=np.int64)
    burden_group = np.asarray(design["burdenGroup"], dtype=np.int64)
    group_size = np.asarray(design["burdenGroupSize"], dtype=float)
    n_levels = int(design["burdenLevels"])
    background = np.maximum(
        burden - xa.astype(np.int64) - xb.astype(np.int64), 0
    )
    histogram = np.bincount(
        burden_group * n_levels + background,
        minlength=len(group_size) * n_levels,
    ).reshape(len(group_size), n_levels)
    cumulative = np.cumsum(histogram, axis=1)
    cell_count = histogram[burden_group, background]
    average_rank = cumulative[burden_group, background] - (cell_count - 1) / 2
    percentile_rank = average_rank / group_size[burden_group]
    burden_bin = np.ceil(np.clip(percentile_rank, 1e-12, 1) * 5).astype(np.int64) - 1
    base_code = np.asarray(design["baseCode"], dtype=np.int64)
    strata = base_code * 5 + burden_bin
    stratum_is_wes = np.repeat(np.asarray(design["baseIsWes"], dtype=bool), 5)
    return strata, stratum_is_wes


def cmh_adjusted(out: pd.DataFrame, context: dict) -> pd.DataFrame:
    """Estimate every pair under primary, sensitivity and diagnostic strata."""
    membership = pd.read_parquet(PROCESSED / "panel_gene_membership.parquet")
    panel_to_genes = membership.groupby("genePanelId")["entrezGeneId"].apply(set).to_dict()
    rows = []
    for cancer, raw in out.groupby("cancer"):
        print(f"  CMH specifications: {cancer} ({len(raw):,} pairs)", flush=True)
        ctx = context[cancer]
        info = ctx["info"].copy()
        info["burdenBin"] = np.ceil(info.burdenPct.clip(lower=1e-9) * 5).clip(1, 5).astype(int)
        genes = sorted(set(raw.entrezA.astype(int)) | set(raw.entrezB.astype(int)))
        gpos = {g: i for i, g in enumerate(genes)}
        spos = {s: i for i, s in enumerate(info.index)}
        x = np.zeros((len(info), len(genes)), dtype=np.int8)
        for j, gene in enumerate(genes):
            idx = [spos[s] for s in ctx["mut_sets"].get(gene, set()) if s in spos]
            if idx:
                x[idx, j] = 1

        coverage = np.zeros((len(info), len(genes)), dtype=bool)
        for j, gene in enumerate(genes):
            idx = [spos[s] for s in ctx["coverage"].get(gene, set()) if s in spos]
            if idx:
                coverage[idx, j] = True

        # The no-burden and total-burden specifications have shared strata
        # across all pairs and therefore retain the fast, vectorised implementation.
        vector_accs = {
            specification: {assay: _new_accumulator(len(genes)) for assay in ASSAYS}
            for specification in ("noBurden", "totalBurden")
        }
        group_fields = {
            "noBurden": ["analysisCancerCode", "studyId", "panelStratum"],
            "totalBurden": ["analysisCancerCode", "studyId", "panelStratum", "burdenBin"],
        }
        for specification, fields in group_fields.items():
            for key, idx in info.groupby(fields, observed=True).indices.items():
                idx = np.asarray(idx, dtype=int)
                if len(idx) < 2:
                    continue
                key = key if isinstance(key, tuple) else (key,)
                panel_stratum = key[2]
                if panel_stratum == "WES/WGS":
                    covered = np.ones(len(genes), dtype=bool)
                    assay_key = "wes"
                else:
                    panel_genes = panel_to_genes.get(panel_stratum, set())
                    covered = np.array([g in panel_genes for g in genes], dtype=bool)
                    assay_key = "panel"
                _accumulate_stratum(vector_accs[specification]["full"], x[idx], covered)
                _accumulate_stratum(vector_accs[specification][assay_key], x[idx], covered)

        finished = {
            specification: {
                assay: _finish_accumulator(vector_accs[specification][assay])
                for assay in ASSAYS
            }
            for specification in vector_accs
        }

        # Leave-two-out background-burden quintiles are pair-specific. Integer-burden
        # histograms recover within-study/assay average ranks without a pandas groupby
        # for every screened pair.
        leave_design = _leave_two_out_design(info)
        base_codes = np.asarray(leave_design["baseCode"], dtype=np.int64)
        base_is_wes = np.asarray(leave_design["baseIsWes"], dtype=bool)
        exclude_affected_study = info.studyId.ne("sarcoma_msk_2022").to_numpy()
        for pair in raw.itertuples(index=False):
            i, j = gpos[int(pair.entrezA)], gpos[int(pair.entrezB)]
            row = {"cancer": cancer, "entrezA": int(pair.entrezA), "entrezB": int(pair.entrezB)}
            for specification in ("noBurden", "totalBurden"):
                for assay in ASSAYS:
                    odds, lo, hi, p, n_strata, n_informative = finished[specification][assay]
                    row.update({
                        f"{specification}_{assay}_or": odds[i, j],
                        f"{specification}_{assay}_ciLow": lo[i, j],
                        f"{specification}_{assay}_ciHigh": hi[i, j],
                        f"{specification}_{assay}_p": p[i, j],
                        f"{specification}_{assay}_nStrata": int(n_strata[i, j]),
                        f"{specification}_{assay}_nInformativeStrata": int(n_informative[i, j]),
                    })

            pair_leave_codes, leave_is_wes = _leave_two_out_pair_strata(
                x[:, i], x[:, j], leave_design
            )
            valid = coverage[:, i] & coverage[:, j]
            leave_estimates = _pair_estimates_by_assay(
                x[:, i], x[:, j], pair_leave_codes, valid, leave_is_wes
            )
            for assay, estimate in leave_estimates.items():
                for statistic in ("or", "ciLow", "ciHigh", "p", "nStrata", "nInformativeStrata"):
                    row[f"leaveTwoOut_{assay}_{statistic}"] = estimate[statistic]

            if cancer == "SARC":
                exclusion_estimates = _pair_estimates_by_assay(
                    x[:, i], x[:, j], base_codes, valid & exclude_affected_study, base_is_wes
                )
                for assay, estimate in exclusion_estimates.items():
                    for statistic in ("or", "ciLow", "ciHigh", "p", "nStrata", "nInformativeStrata"):
                        row[f"excludeSarcomaMsk2022_{assay}_{statistic}"] = estimate[statistic]
            rows.append(row)
    cmh = pd.DataFrame(rows)
    result = out.merge(cmh, on=["cancer", "entrezA", "entrezB"], how="left")
    for specification in SPECIFICATIONS:
        for assay in ASSAYS:
            result[f"{specification}_{assay}_fdr"] = np.nan
            for cancer, idx in result.groupby("cancer").groups.items():
                valid = result.loc[idx, f"{specification}_{assay}_p"].notna()
                use = result.loc[idx].index[valid]
                if len(use):
                    result.loc[use, f"{specification}_{assay}_fdr"] = multipletests(
                        result.loc[use, f"{specification}_{assay}_p"], method="fdr_bh"
                    )[1]

        result[f"{specification}SameAssayDirection"] = (
            np.sign(np.log(result[f"{specification}_wes_or"]))
            == np.sign(np.log(result[f"{specification}_panel_or"]))
        )
        result[f"{specification}CrossAssayConcordant"] = (
            result[f"{specification}SameAssayDirection"]
            & (result[f"{specification}_wes_fdr"] < 0.10)
            & (result[f"{specification}_panel_p"] < 0.05)
            & (np.abs(np.log2(result[f"{specification}_wes_or"])) >= 0.25)
            & (np.abs(np.log2(result[f"{specification}_panel_or"])) >= 0.25)
        )

    # The strict assay-definition analysis is unchanged. This separate exclusion
    # sensitivity addresses the concentration of off-panel conflicts in one sarcoma
    # study without converting positive-only records into assay-covered observations.
    if "excludeSarcomaMsk2022_full_p" in result:
        result["excludeSarcomaMsk2022_full_fdr"] = np.nan
        sarc = result.cancer.eq("SARC") & result.excludeSarcomaMsk2022_full_p.notna()
        if sarc.any():
            result.loc[sarc, "excludeSarcomaMsk2022_full_fdr"] = multipletests(
                result.loc[sarc, "excludeSarcomaMsk2022_full_p"], method="fdr_bh"
            )[1]

    no_burden_log2 = np.log2(result.noBurden_full_or.where(result.noBurden_full_or > 0))
    leave_log2 = np.log2(result.leaveTwoOut_full_or.where(result.leaveTwoOut_full_or > 0))
    total_log2 = np.log2(result.totalBurden_full_or.where(result.totalBurden_full_or > 0))
    result["primaryDirection"] = np.where(
        no_burden_log2 < 0, "mutually exclusive",
        np.where(no_burden_log2 > 0, "co-occurring", "non-estimable"),
    )
    result["leaveTwoOutDirection"] = np.where(
        leave_log2 < 0, "mutually exclusive",
        np.where(leave_log2 > 0, "co-occurring", "non-estimable"),
    )
    result["totalBurdenDirection"] = np.where(
        total_log2 < 0, "mutually exclusive",
        np.where(total_log2 > 0, "co-occurring", "non-estimable"),
    )
    estimable_no_leave = no_burden_log2.notna() & leave_log2.notna()
    estimable_all = estimable_no_leave & total_log2.notna()
    result["signStableNoBurdenLeaveTwoOut"] = (
        estimable_no_leave & (np.sign(no_burden_log2) == np.sign(leave_log2))
    )
    result["signStableAcrossAllSpecifications"] = (
        estimable_all
        & (np.sign(no_burden_log2) == np.sign(leave_log2))
        & (np.sign(no_burden_log2) == np.sign(total_log2))
    )
    result["leaveTwoOutMinusNoBurdenLog2Or"] = leave_log2 - no_burden_log2
    result["totalBurdenMinusNoBurdenLog2Or"] = total_log2 - no_burden_log2
    result["effectStableNoBurdenLeaveTwoOut"] = (
        result.signStableNoBurdenLeaveTwoOut
        & (
            result.leaveTwoOutMinusNoBurdenLog2Or.abs().le(EFFECT_STABILITY_LOG2_TOLERANCE)
            | (
                no_burden_log2.abs().ge(STRONG_EFFECT_LOG2_THRESHOLD)
                & leave_log2.abs().ge(STRONG_EFFECT_LOG2_THRESHOLD)
            )
        )
    )
    result["effectStableAcrossAllSpecifications"] = (
        result.signStableAcrossAllSpecifications
        & (
            (
                result.leaveTwoOutMinusNoBurdenLog2Or.abs().le(EFFECT_STABILITY_LOG2_TOLERANCE)
                & result.totalBurdenMinusNoBurdenLog2Or.abs().le(EFFECT_STABILITY_LOG2_TOLERANCE)
            )
            | (
                no_burden_log2.abs().ge(STRONG_EFFECT_LOG2_THRESHOLD)
                & leave_log2.abs().ge(STRONG_EFFECT_LOG2_THRESHOLD)
                & total_log2.abs().ge(STRONG_EFFECT_LOG2_THRESHOLD)
            )
        )
    )
    result["effectStabilityDefinition"] = (
        "same direction and within two-fold OR, or all estimates are strong "
        "(|log2 OR| >= 2) in the same direction"
    )
    if "excludeSarcomaMsk2022_full_or" in result:
        exclusion_log2 = np.log2(
            result.excludeSarcomaMsk2022_full_or.where(result.excludeSarcomaMsk2022_full_or > 0)
        )
        result["excludeSarcomaMsk2022MinusPrimaryLog2Or"] = exclusion_log2 - no_burden_log2
        result["excludeSarcomaMsk2022DirectionStable"] = (
            exclusion_log2.notna()
            & no_burden_log2.notna()
            & (np.sign(exclusion_log2) == np.sign(no_burden_log2))
        )
        result["excludeSarcomaMsk2022EffectStable"] = (
            result.excludeSarcomaMsk2022DirectionStable
            & (
                result.excludeSarcomaMsk2022MinusPrimaryLog2Or.abs().le(
                    EFFECT_STABILITY_LOG2_TOLERANCE
                )
                | (
                    exclusion_log2.abs().ge(STRONG_EFFECT_LOG2_THRESHOLD)
                    & no_burden_log2.abs().ge(STRONG_EFFECT_LOG2_THRESHOLD)
                )
            )
        )
    result["primarySpecification"] = "detailed histology + study + exact assay/panel; no burden"
    result["sensitivitySpecification"] = (
        "detailed histology + study + exact assay/panel + leave-two-out background-burden quintile"
    )
    result["diagnosticSpecification"] = (
        "detailed histology + study + exact assay/panel + total-burden quintile"
    )

    # Compatibility field names point to the primary no-burden specification.
    # Explicit totalBurden_* fields store the diagnostic total-burden model.
    for assay in ASSAYS:
        for statistic in ("or", "ciLow", "ciHigh", "p", "fdr", "nStrata", "nInformativeStrata"):
            result[f"cmh_{assay}_{statistic}"] = result[f"noBurden_{assay}_{statistic}"]
    result["cmhSameDirection"] = result.noBurdenSameAssayDirection
    result["cmhReplicated"] = result.noBurdenCrossAssayConcordant
    result.to_csv(TABLES / "cooccurrence_curated_adjusted_sensitivity.csv", index=False)
    return result


def _heterogeneity_contexts(result: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return principal, reference and displayed-network contexts with provenance."""
    selected: dict[tuple[str, str], set[str]] = {}
    for pair, cancer in PRINCIPAL_HETEROGENEITY_CONTEXTS:
        selected.setdefault((pair, cancer), set()).add("prespecified principal context")
    for pair, cancer in REFERENCE_CONTEXTS:
        selected.setdefault((pair, cancer), set()).add("prespecified reference context")

    display_path = TABLES.parent / "source_data" / "figure8_main_edge_contexts.csv"
    if display_path.exists():
        displayed = pd.read_csv(display_path, usecols=["cancer", "pair"]).drop_duplicates()
        for record in displayed.itertuples(index=False):
            selected.setdefault((str(record.pair), str(record.cancer)), set()).add(
                "Figure 8 displayed context"
            )
    if result is not None:
        sarc_candidates = result[
            result.cancer.eq("SARC")
            & result.noBurdenCrossAssayConcordant.fillna(False)
            & result.noBurden_full_fdr.lt(1e-6)
            & (
                result.noBurden_full_or.le(0.5)
                | result.noBurden_full_or.ge(2.0)
            )
        ]
        for record in sarc_candidates.itertuples(index=False):
            selected.setdefault((str(record.pair), str(record.cancer)), set()).add(
                "sarcoma assay-metadata sensitivity context"
            )
    return pd.DataFrame(
        [
            {
                "pair": pair,
                "cancer": cancer,
                "selectionSource": "; ".join(sorted(sources)),
            }
            for (pair, cancer), sources in sorted(selected.items(), key=lambda item: (item[0][1], item[0][0]))
        ]
    )


def _meta_analysis(studies: pd.DataFrame) -> dict[str, float]:
    """Fixed/random study-level summaries and Cochran heterogeneity diagnostics."""
    finite = studies[
        studies.studySpecificOr.gt(0)
        & studies.studySpecificLogOrSe.gt(0)
        & studies.studySpecificLogOrSe.notna()
    ].copy()
    if finite.empty:
        return {
            "nFiniteStudyEstimates": 0,
            "heterogeneityQ": np.nan, "heterogeneityDf": np.nan,
            "heterogeneityP": np.nan, "heterogeneityI2Pct": np.nan,
            "tau2DerSimonianLaird": np.nan,
            "fixedEffectOr": np.nan, "fixedEffectCiLow": np.nan, "fixedEffectCiHigh": np.nan,
            "randomEffectsOr": np.nan, "randomEffectsCiLow": np.nan,
            "randomEffectsCiHigh": np.nan,
        }
    yi = np.log(finite.studySpecificOr.to_numpy(float))
    vi = finite.studySpecificLogOrSe.to_numpy(float) ** 2
    wi = 1 / vi
    fixed_log = float(np.sum(wi * yi) / np.sum(wi))
    fixed_se = float(np.sqrt(1 / np.sum(wi)))
    q = float(np.sum(wi * (yi - fixed_log) ** 2))
    df = len(yi) - 1
    q_p = float(chi2.sf(q, df)) if df > 0 else np.nan
    i2 = float(max(0, (q - df) / q) * 100) if q > 0 and df > 0 else 0.0
    c = float(np.sum(wi) - np.sum(wi**2) / np.sum(wi))
    tau2 = float(max(0, (q - df) / c)) if c > 0 and df > 0 else 0.0
    random_weights = 1 / (vi + tau2)
    random_log = float(np.sum(random_weights * yi) / np.sum(random_weights))
    random_se = float(np.sqrt(1 / np.sum(random_weights)))
    return {
        "nFiniteStudyEstimates": int(len(finite)),
        "heterogeneityQ": q, "heterogeneityDf": int(df),
        "heterogeneityP": q_p, "heterogeneityI2Pct": i2,
        "tau2DerSimonianLaird": tau2,
        "fixedEffectOr": float(np.exp(fixed_log)),
        "fixedEffectCiLow": float(np.exp(fixed_log - 1.96 * fixed_se)),
        "fixedEffectCiHigh": float(np.exp(fixed_log + 1.96 * fixed_se)),
        "randomEffectsOr": float(np.exp(random_log)),
        "randomEffectsCiLow": float(np.exp(random_log - 1.96 * random_se)),
        "randomEffectsCiHigh": float(np.exp(random_log + 1.96 * random_se)),
    }


def study_heterogeneity(result: pd.DataFrame, context: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Write study-specific and leave-one-study-out primary estimates.

    These diagnostics are intentionally restricted to the literature-guided displayed
    network plus reference contexts; the 74,582-pair discovery table retains the three
    complete conditioning specifications and informative-stratum counts.
    """
    requested = _heterogeneity_contexts(result)
    selected = requested.merge(
        result[["cancer", "pair", "entrezA", "entrezB"]],
        on=["cancer", "pair"], how="left", validate="one_to_one",
    )
    required = set(PRINCIPAL_HETEROGENEITY_CONTEXTS)
    absent_required = {
        (record.pair, record.cancer)
        for record in selected[selected.entrezA.isna()].itertuples(index=False)
        if (record.pair, record.cancer) in required
    }
    if absent_required:
        raise RuntimeError(f"Principal heterogeneity contexts absent from pair screen: {sorted(absent_required)}")
    selected = selected.dropna(subset=["entrezA", "entrezB"]).copy()

    study_rows: list[dict[str, object]] = []
    leave_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for record in selected.itertuples(index=False):
        ctx = context[record.cancer]
        info = ctx["info"].copy()
        sample_ids = info.index.to_numpy()
        gene_a, gene_b = int(record.entrezA), int(record.entrezB)
        xa = np.fromiter(
            (sample in ctx["mut_sets"].get(gene_a, set()) for sample in sample_ids),
            dtype=np.int8, count=len(info),
        )
        xb = np.fromiter(
            (sample in ctx["mut_sets"].get(gene_b, set()) for sample in sample_ids),
            dtype=np.int8, count=len(info),
        )
        valid = np.fromiter(
            (
                sample in ctx["coverage"].get(gene_a, set())
                and sample in ctx["coverage"].get(gene_b, set())
                for sample in sample_ids
            ),
            dtype=bool, count=len(info),
        )
        base_index = pd.MultiIndex.from_frame(
            info[["analysisCancerCode", "studyId", "panelStratum"]]
        )
        base_codes, base_levels = pd.factorize(base_index, sort=False)
        n_base = len(base_levels)
        studies = info.loc[valid, "studyId"].drop_duplicates().sort_values().tolist()
        study_values = info.studyId.to_numpy()

        overall_counts = _pair_counts(xa, xb, base_codes, valid, n_base)
        overall = _cmh_from_counts(overall_counts)
        context_study_start = len(study_rows)
        for study_id in studies:
            study_mask = study_values == study_id
            study_valid = valid & study_mask
            estimate = _cmh_from_counts(
                _pair_counts(xa, xb, base_codes, study_valid, n_base)
            )
            ids = study_valid
            row = {
                "cancer": record.cancer, "pair": record.pair,
                "selectionSource": record.selectionSource, "studyId": study_id,
                "nJointlyAssayCovered": int(ids.sum()),
                "nGeneA": int(xa[ids].sum()), "nGeneB": int(xb[ids].sum()),
                "nBoth": int((xa[ids] * xb[ids]).sum()),
                "studySpecificOr": estimate["or"],
                "studySpecificCiLow": estimate["ciLow"],
                "studySpecificCiHigh": estimate["ciHigh"],
                "studySpecificP": estimate["p"],
                "studySpecificLogOrSe": estimate["logOrSe"],
                "nStrata": estimate["nStrata"],
                "nInformativeStrata": estimate["nInformativeStrata"],
                "primarySpecification": "detailed histology + exact assay/panel within study; no burden",
            }
            study_rows.append(row)

            leave_valid = valid & ~study_mask
            leave = _cmh_from_counts(
                _pair_counts(xa, xb, base_codes, leave_valid, n_base)
            )
            leave_estimable = bool(
                np.isfinite(leave["or"]) and leave["or"] > 0
                and np.isfinite(overall["or"]) and overall["or"] > 0
            )
            leave_rows.append({
                "cancer": record.cancer, "pair": record.pair,
                "selectionSource": record.selectionSource, "omittedStudyId": study_id,
                "omittedNJointlyAssayCovered": int(study_valid.sum()),
                "remainingNJointlyAssayCovered": int(leave_valid.sum()),
                "leaveOneStudyOutOr": leave["or"],
                "leaveOneStudyOutCiLow": leave["ciLow"],
                "leaveOneStudyOutCiHigh": leave["ciHigh"],
                "leaveOneStudyOutP": leave["p"],
                "leaveOneStudyOutNStrata": leave["nStrata"],
                "leaveOneStudyOutNInformativeStrata": leave["nInformativeStrata"],
                "leaveOneStudyOutEstimable": leave_estimable,
                "sameDirectionAsAllStudies": (
                    bool(np.sign(np.log(leave["or"])) == np.sign(np.log(overall["or"])))
                    if leave_estimable else np.nan
                ),
                "primarySpecification": "detailed histology + study + exact assay/panel; no burden",
            })

        context_studies = pd.DataFrame(study_rows[context_study_start:])
        meta = _meta_analysis(context_studies)
        context_leave = pd.DataFrame(
            [r for r in leave_rows if r["cancer"] == record.cancer and r["pair"] == record.pair]
        )
        finite_leave = context_leave.leaveOneStudyOutOr.dropna()
        estimable_leave = context_leave[context_leave.leaveOneStudyOutEstimable]
        finite_studies = context_studies.studySpecificOr.dropna()
        summary_rows.append({
            "cancer": record.cancer, "pair": record.pair,
            "selectionSource": record.selectionSource,
            "allStudiesPrimaryOr": overall["or"],
            "allStudiesPrimaryCiLow": overall["ciLow"],
            "allStudiesPrimaryCiHigh": overall["ciHigh"],
            "allStudiesPrimaryP": overall["p"],
            "nStudiesWithJointCoverage": int(len(studies)),
            "nStudiesWithInformativeStrata": int(
                context_studies.nInformativeStrata.gt(0).sum()
            ),
            "nAllStudyStrata": overall["nStrata"],
            "nAllStudyInformativeStrata": overall["nInformativeStrata"],
            "studySpecificOrMin": float(finite_studies.min()) if len(finite_studies) else np.nan,
            "studySpecificOrMax": float(finite_studies.max()) if len(finite_studies) else np.nan,
            "leaveOneStudyOutOrMin": float(finite_leave.min()) if len(finite_leave) else np.nan,
            "leaveOneStudyOutOrMax": float(finite_leave.max()) if len(finite_leave) else np.nan,
            "nLeaveOneStudyOutEstimable": int(len(estimable_leave)),
            "allLeaveOneStudyOutEstimable": bool(
                len(context_leave) and len(estimable_leave) == len(context_leave)
            ),
            "leaveOneStudyOutDirectionStableAmongEstimable": bool(
                len(estimable_leave)
                and estimable_leave.sameDirectionAsAllStudies.astype(bool).all()
            ),
            "leaveOneStudyOutDirectionStable": bool(
                len(context_leave)
                and len(estimable_leave) == len(context_leave)
                and estimable_leave.sameDirectionAsAllStudies.astype(bool).all()
            ),
            **meta,
            "primarySpecification": "detailed histology + study + exact assay/panel; no burden",
        })

    study_specific = pd.DataFrame(study_rows)
    leave_one_out = pd.DataFrame(leave_rows)
    heterogeneity = pd.DataFrame(summary_rows)
    study_specific.to_csv(TABLES / "pairwise_study_specific_primary.csv", index=False)
    leave_one_out.to_csv(TABLES / "pairwise_leave_one_study_out_primary.csv", index=False)
    heterogeneity.to_csv(TABLES / "pairwise_study_heterogeneity_primary.csv", index=False)
    return study_specific, leave_one_out, heterogeneity


def assay_discordant_specimen_sensitivity(result: pd.DataFrame, context: dict) -> pd.DataFrame:
    """Refit displayed/reference contexts after removing any conflict specimen.

    The strict documented-panel definition remains the primary analysis. This sensitivity
    excludes, rather than reclassifies, every selected specimen carrying at least one
    mutation outside its documented assay scope.
    """
    raw_mutations = pd.read_parquet(
        PROCESSED / "mutations_dedup.parquet",
        columns=["sampleId", "studyId", "entrezGeneId"],
    )
    _, conflicts = partition_callable_mutations(raw_mutations)
    eligible = pd.read_parquet(
        PROCESSED / "analysis_samples_curated.parquet",
        columns=["sampleId", "studyId", "analysisEligible"],
    )
    selected_conflicts = conflicts.merge(
        eligible.loc[eligible.analysisEligible, ["sampleId", "studyId"]],
        on=["sampleId", "studyId"], how="inner", validate="many_to_one",
    )
    conflict_keys = set(
        selected_conflicts[["sampleId", "studyId"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if len(selected_conflicts) != 2_508 or len(conflict_keys) != 1_832:
        raise AssertionError(
            "Assay-discordance contract drift: expected 2,508 selected records in "
            f"1,832 specimens; observed {len(selected_conflicts):,} records in "
            f"{len(conflict_keys):,} specimens"
        )

    selected = _heterogeneity_contexts(result).merge(
        result[["cancer", "pair", "entrezA", "entrezB"]],
        on=["cancer", "pair"], how="left", validate="one_to_one",
    ).dropna(subset=["entrezA", "entrezB"])
    rows: list[dict[str, object]] = []
    for record in selected.itertuples(index=False):
        ctx = context[record.cancer]
        info = ctx["info"].copy()
        sample_ids = info.index.to_numpy()
        study_ids = info.studyId.to_numpy()
        gene_a, gene_b = int(record.entrezA), int(record.entrezB)
        xa = np.fromiter(
            (sample in ctx["mut_sets"].get(gene_a, set()) for sample in sample_ids),
            dtype=np.int8, count=len(info),
        )
        xb = np.fromiter(
            (sample in ctx["mut_sets"].get(gene_b, set()) for sample in sample_ids),
            dtype=np.int8, count=len(info),
        )
        valid = np.fromiter(
            (
                sample in ctx["coverage"].get(gene_a, set())
                and sample in ctx["coverage"].get(gene_b, set())
                for sample in sample_ids
            ),
            dtype=bool, count=len(info),
        )
        discordant = np.fromiter(
            ((sample, study) in conflict_keys for sample, study in zip(sample_ids, study_ids)),
            dtype=bool, count=len(info),
        )
        retained = valid & ~discordant

        base_index = pd.MultiIndex.from_frame(
            info[["analysisCancerCode", "studyId", "panelStratum"]]
        )
        base_codes, base_levels = pd.factorize(base_index, sort=False)
        base_is_wes = np.asarray([level[2] == "WES/WGS" for level in base_levels], dtype=bool)
        no_burden = _pair_estimates_by_assay(
            xa, xb, base_codes, retained, base_is_wes
        )

        leave_design = _leave_two_out_design(info)
        pair_leave_codes, leave_is_wes = _leave_two_out_pair_strata(
            xa, xb, leave_design
        )
        leave_two_out = _pair_estimates_by_assay(
            xa, xb, pair_leave_codes, retained, leave_is_wes
        )
        row: dict[str, object] = {
            "cancer": record.cancer, "pair": record.pair,
            "selectionSource": record.selectionSource,
            "nJointlyAssayCoveredBeforeExclusion": int(valid.sum()),
            "nAssayDiscordantSpecimensInCancer": int(discordant.sum()),
            "nJointlyAssayCoveredSpecimensExcluded": int((valid & discordant).sum()),
            "nJointlyAssayCoveredAfterExclusion": int(retained.sum()),
            "selectedConflictRecordsAllCancers": int(len(selected_conflicts)),
            "selectedConflictSpecimensAllCancers": int(len(conflict_keys)),
            "sensitivityDefinition": (
                "exclude every selected specimen with at least one mutation outside documented assay scope"
            ),
        }
        for specification, estimates in (
            ("excludeDiscordantNoBurden", no_burden),
            ("excludeDiscordantLeaveTwoOut", leave_two_out),
        ):
            for assay, estimate in estimates.items():
                for statistic in (
                    "or", "ciLow", "ciHigh", "p", "nStrata", "nInformativeStrata"
                ):
                    row[f"{specification}_{assay}_{statistic}"] = estimate[statistic]
        original = result[(result.cancer == record.cancer) & (result.pair == record.pair)].iloc[0]
        for specification, original_field, sensitivity_field in (
            ("noBurden", "noBurden_full_or", "excludeDiscordantNoBurden_full_or"),
            ("leaveTwoOut", "leaveTwoOut_full_or", "excludeDiscordantLeaveTwoOut_full_or"),
        ):
            before = float(original[original_field])
            after = float(row[sensitivity_field])
            estimable = np.isfinite(before) and before > 0 and np.isfinite(after) and after > 0
            row[f"{specification}DirectionStableAfterDiscordantExclusion"] = bool(
                estimable and np.sign(np.log(before)) == np.sign(np.log(after))
            )
            row[f"{specification}DiscordantExclusionLog2OrDifference"] = (
                float(np.log2(after) - np.log2(before)) if estimable else np.nan
            )
        rows.append(row)
    output = pd.DataFrame(rows)
    output.to_csv(
        TABLES / "pairwise_assay_discordant_specimen_sensitivity.csv", index=False
    )
    return output


def validate_three_specification_arithmetic(result: pd.DataFrame, context: dict) -> pd.DataFrame:
    """Independently validate reference-context estimates against statsmodels."""
    rows: list[dict[str, object]] = []
    for pair, cancer in sorted(REFERENCE_CONTEXTS):
        record = result[(result.cancer == cancer) & (result.pair == pair)]
        if len(record) != 1:
            raise AssertionError(f"Expected one screened result for {pair} in {cancer}")
        record = record.iloc[0]
        ctx = context[cancer]
        info = ctx["info"].copy()
        info["burdenBin"] = np.ceil(info.burdenPct.clip(lower=1e-9) * 5).clip(1, 5).astype(int)
        ids = info.index.to_numpy()
        gene_a, gene_b = int(record.entrezA), int(record.entrezB)
        xa = np.fromiter(
            (sample in ctx["mut_sets"].get(gene_a, set()) for sample in ids),
            dtype=np.int8, count=len(info),
        )
        xb = np.fromiter(
            (sample in ctx["mut_sets"].get(gene_b, set()) for sample in ids),
            dtype=np.int8, count=len(info),
        )
        valid = np.fromiter(
            (
                sample in ctx["coverage"].get(gene_a, set())
                and sample in ctx["coverage"].get(gene_b, set())
                for sample in ids
            ),
            dtype=bool, count=len(info),
        )

        specification_counts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for specification, fields in (
            ("noBurden", ["analysisCancerCode", "studyId", "panelStratum"]),
            (
                "totalBurden",
                ["analysisCancerCode", "studyId", "panelStratum", "burdenBin"],
            ),
        ):
            index = pd.MultiIndex.from_frame(info[fields])
            codes, levels = pd.factorize(index, sort=False)
            is_wes = np.asarray([level[2] == "WES/WGS" for level in levels], dtype=bool)
            specification_counts[specification] = (
                _pair_counts(xa, xb, codes, valid, len(levels)), is_wes
            )
        leave_design = _leave_two_out_design(info)
        leave_codes, leave_is_wes = _leave_two_out_pair_strata(xa, xb, leave_design)
        specification_counts["leaveTwoOut"] = (
            _pair_counts(xa, xb, leave_codes, valid, len(leave_is_wes)),
            leave_is_wes,
        )

        for specification, (all_counts, is_wes) in specification_counts.items():
            for assay, assay_mask in (
                ("full", np.ones(len(is_wes), dtype=bool)),
                ("wes", is_wes),
                ("panel", ~is_wes),
            ):
                counts = all_counts[assay_mask]
                counts = counts[counts.sum(axis=1) > 1]
                if not len(counts):
                    continue
                d, c, b, a = counts.T.astype(float)
                tables = np.stack(
                    [np.array([[aa, bb], [cc, dd]]) for aa, bb, cc, dd in zip(a, b, c, d)],
                    axis=2,
                )
                reference = StratifiedTable(tables, shift_zeros=False)
                expected_or = float(reference.oddsratio_pooled)
                expected_lo, expected_hi = map(float, reference.oddsratio_pooled_confint())
                expected_p = float(reference.test_null_odds().pvalue)
                observed_or = record[f"{specification}_{assay}_or"]
                observed_lo = record[f"{specification}_{assay}_ciLow"]
                observed_hi = record[f"{specification}_{assay}_ciHigh"]
                observed_p = record[f"{specification}_{assay}_p"]
                expected_estimable = np.isfinite(expected_or) and expected_or > 0
                observed_estimable = pd.notna(observed_or)
                if expected_estimable != observed_estimable:
                    raise AssertionError(
                        f"CMH estimability differs: {cancer} {pair} {specification} {assay}"
                    )
                if observed_estimable:
                    np.testing.assert_allclose(observed_or, expected_or, rtol=1e-10, atol=1e-12)
                    np.testing.assert_allclose(
                        [observed_lo, observed_hi], [expected_lo, expected_hi],
                        rtol=1e-4, atol=1e-10,
                    )
                np.testing.assert_allclose(observed_p, expected_p, rtol=1e-9, atol=1e-12)
                rows.append({
                    "cancer": cancer, "pair": pair,
                    "specification": specification, "assay": assay,
                    "nStrata": int(len(counts)), "estimable": observed_estimable,
                    "pipelineOr": observed_or, "statsmodelsOr": expected_or,
                    "absOrDifference": (
                        abs(observed_or - expected_or) if observed_estimable else np.nan
                    ),
                    "maxAbsCiDifference": (
                        max(abs(observed_lo - expected_lo), abs(observed_hi - expected_hi))
                        if observed_estimable else np.nan
                    ),
                    "absPDifference": abs(observed_p - expected_p),
                    "status": "PASS",
                })
    validation = pd.DataFrame(rows)
    validation.to_csv(TABLES / "cmh_three_specification_validation.csv", index=False)
    return validation


def make_figure(res: pd.DataFrame) -> None:
    apply_style()
    fig = plt.figure(figsize=figsize(180, 177))
    gs = GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.40)

    ax_a = fig.add_subplot(gs[0, 0])
    both = res.dropna(subset=["cmh_wes_or", "cmh_panel_or"]).copy()
    both = both[(both.cmh_wes_or > 0) & (both.cmh_panel_or > 0)]
    top_highlight = set(
        both[both.cmhReplicated].sort_values("cmh_full_p").head(45).index
    )
    canonical_highlight = set(
        both.index[
            both.apply(lambda r: tuple(sorted((r.geneA, r.geneB))) in CANONICAL_PAIRS, axis=1)
        ]
    )
    highlight = both.index.isin(top_highlight | canonical_highlight)
    ax_a.scatter(
        np.log2(both.cmh_wes_or.clip(1e-3, 1e3)),
        np.log2(both.cmh_panel_or.clip(1e-3, 1e3)),
        s=np.where(highlight, 16, 4),
        c=np.where(highlight, COLORS["vermillion"], COLORS["grey"]),
        alpha=np.where(highlight, 0.80, 0.13),
        edgecolors="none",
    )
    lo = min(np.log2(both.cmh_wes_or.clip(1e-3, 1e3)).min(), np.log2(both.cmh_panel_or.clip(1e-3, 1e3)).min())
    hi = max(np.log2(both.cmh_wes_or.clip(1e-3, 1e3)).max(), np.log2(both.cmh_panel_or.clip(1e-3, 1e3)).max())
    ax_a.plot([lo, hi], [lo, hi], ls="--", color=COLORS["grey"], lw=0.7)
    ax_a.axhline(0, color=COLORS["light_grey"], lw=0.5); ax_a.axvline(0, color=COLORS["light_grey"], lw=0.5)
    rho, _ = spearmanr(np.log2(both.cmh_wes_or), np.log2(both.cmh_panel_or))
    ax_a.text(0.04, 0.95, f"Spearman ρ={rho:.2f}", transform=ax_a.transAxes, va="top")
    wanted = {
        ("EGFR-KRAS", "LUAD"), ("KEAP1-STK11", "LUAD"),
        ("BRAF-KRAS", "COADREAD"), ("ATRX-TP53", "GBM"),
        ("KRAS-TP53", "PAAD"),
    }
    labels = both[both.apply(lambda r: (r.pair, r.cancer) in wanted, axis=1)]
    a_offsets = {
        ("EGFR-KRAS", "LUAD"): (5, -18),
        ("KEAP1-STK11", "LUAD"): (5, 13),
        ("BRAF-KRAS", "COADREAD"): (5, 5),
        ("ATRX-TP53", "GBM"): (5, 3),
        ("KRAS-TP53", "PAAD"): (-43, -16),
    }
    for row in labels.itertuples(index=False):
        ax_a.annotate(
            f"{row.pair}\n{row.cancer}",
            (np.log2(row.cmh_wes_or), np.log2(row.cmh_panel_or)),
            xytext=a_offsets[(row.pair, row.cancer)], textcoords="offset points", fontsize=4.2,
            arrowprops={"arrowstyle": "-", "lw": 0.35, "color": COLORS["grey"]},
        )
    ax_a.set_xlabel("WES/WGS log2 odds ratio")
    ax_a.set_ylabel("Targeted-panel log2 odds ratio")
    ax_a.set_title("Cross-assay adjusted effect directions", loc="left")
    panel_label(ax_a, "a")

    ax_b = fig.add_subplot(gs[0, 1])
    rep = res[res.cmhReplicated].copy()
    rep["strength"] = -np.log10(rep.cmh_full_p.clip(lower=1e-300))
    rep = rep.sort_values("strength", ascending=False).drop_duplicates(["pair", "cancer"]).head(10).iloc[::-1]
    y = np.arange(len(rep))
    for offset, prefix, colour, label in [(-0.12, "wes", COLORS["blue"], "WES/WGS"), (0.12, "panel", COLORS["orange"], "Targeted panel")]:
        value = rep[f"cmh_{prefix}_or"].to_numpy()
        lo_ci = rep[f"cmh_{prefix}_ciLow"].to_numpy(); hi_ci = rep[f"cmh_{prefix}_ciHigh"].to_numpy()
        ax_b.errorbar(value, y + offset, xerr=[value - lo_ci, hi_ci - value], fmt="o", ms=3, color=colour, capsize=1.3, lw=0.7, label=label)
    ax_b.axvline(1, color=COLORS["grey"], ls="--", lw=0.7)
    ax_b.set_xscale("log")
    ax_b.set_yticks(y); ax_b.set_yticklabels(rep.pair + " (" + rep.cancer + ")", fontsize=5.2)
    ax_b.set_xlabel("Conditioned common odds ratio (95% CI)")
    ax_b.set_title("Replicated cancer-specific associations", loc="left")
    ax_b.legend(frameon=False, ncol=2, loc="upper center")
    panel_label(ax_b, "b")

    ax_c = fig.add_subplot(gs[1, 0])
    adj = res.dropna(subset=["cmh_full_or"]).copy()
    adj = adj[(adj.full_oddsRatio > 0) & (adj.cmh_full_or > 0)]
    hx = np.log2(adj.full_oddsRatio).to_numpy()
    hy = np.log2(adj.cmh_full_or).to_numpy()
    hb = ax_c.hexbin(hx, hy, gridsize=45, mincnt=1, bins="log", cmap="Greens", linewidths=0)
    lo2 = min(np.log2(adj.full_oddsRatio).min(), np.log2(adj.cmh_full_or).min())
    hi2 = max(np.log2(adj.full_oddsRatio).max(), np.log2(adj.cmh_full_or).max())
    ax_c.plot([lo2, hi2], [lo2, hi2], ls="--", color=COLORS["grey"], lw=0.7)
    rho_adj, _ = spearmanr(hx, hy)
    sign_same = 100 * np.mean(np.sign(hx) == np.sign(hy))
    ax_c.text(0.04, 0.95, f"Spearman ρ={rho_adj:.2f}\ndirection retained={sign_same:.1f}%", transform=ax_c.transAxes, va="top")
    ax_c.axhline(0, color=COLORS["light_grey"], lw=0.5); ax_c.axvline(0, color=COLORS["light_grey"], lw=0.5)
    ax_c.set_xlabel("Unadjusted log2 odds ratio")
    ax_c.set_ylabel("Histology/study/panel/burden-adjusted log2 odds ratio")
    ax_c.set_title("Adjustment attenuates pooled co-occurrence", loc="left")
    panel_label(ax_c, "c")

    ax_d = fig.add_subplot(gs[1, 1])
    canon = res[res.apply(lambda r: (r.pair, r.cancer) in REFERENCE_CONTEXTS, axis=1)].copy()
    canon = canon[canon.cmh_full_fdr < 0.10]
    canon["label"] = canon.pair + " (" + canon.cancer + ")"
    canon = canon.assign(abslog=lambda x: np.abs(np.log2(x.cmh_full_or))).sort_values("cmh_full_or", ascending=False).iloc[::-1]
    yy = np.arange(len(canon))
    vals = canon.cmh_full_or.to_numpy(); lo_ci = canon.cmh_full_ciLow.to_numpy(); hi_ci = canon.cmh_full_ciHigh.to_numpy()
    colours = [COLORS["blue"] if v < 1 else COLORS["vermillion"] for v in vals]
    for i, (v, l, h, colour) in enumerate(zip(vals, lo_ci, hi_ci, colours)):
        ax_d.errorbar(v, i, xerr=[[v - l], [h - v]], fmt="o", ms=3.3, color=colour, capsize=1.3, lw=0.7)
    ax_d.axvline(1, color=COLORS["grey"], ls="--", lw=0.7)
    ax_d.set_xscale("log")
    ax_d.set_yticks(yy); ax_d.set_yticklabels(canon.label, fontsize=5.2)
    ax_d.set_xlabel("Conditioned common odds ratio (95% CI)")
    ax_d.set_title("Cancer-context biological reference pairs", loc="left")
    panel_label(ax_d, "d")

    # Write the interaction-screen diagnostic outside the panel-mapped source-data
    # directory so that it remains distinct from the principal figure tables.
    source = TABLES.parent / "source_data" / "archive_legacy" / "figure13_interaction_qc"
    source.mkdir(parents=True, exist_ok=True)
    panel_a = both[
        ["cancer", "pair", "geneA", "geneB", "cmh_wes_or", "cmh_panel_or",
         "cmh_wes_fdr", "cmh_panel_fdr", "cmhReplicated"]
    ].copy()
    panel_a["displayHighlighted"] = highlight
    panel_a["displayLabel"] = panel_a.apply(
        lambda row: (row.pair, row.cancer) in wanted, axis=1
    )
    panel_a.to_csv(source / "figure13_panel_a_cross_assay.csv", index=False)
    rep[
        ["cancer", "pair", "cmh_wes_or", "cmh_wes_ciLow", "cmh_wes_ciHigh",
         "cmh_panel_or", "cmh_panel_ciLow", "cmh_panel_ciHigh", "cmh_full_fdr"]
    ].to_csv(source / "figure13_panel_b_replicated.csv", index=False)
    adj[
        ["cancer", "pair", "full_oddsRatio", "cmh_full_or", "cmh_full_fdr"]
    ].to_csv(source / "figure13_panel_c_adjusted_sensitivity.csv", index=False)
    canon[
        ["cancer", "pair", "cmh_full_or", "cmh_full_ciLow", "cmh_full_ciHigh",
         "cmh_full_fdr"]
    ].to_csv(source / "figure13_panel_d_canonical_pairs.csv", index=False)
    save_figure(fig, FIGURES / "figure13_curated_interactions")
    plt.close(fig)


def main() -> None:
    out, _, context = screen_pairs()
    adjusted = cmh_adjusted(out, context)
    study_specific, leave_one_out, heterogeneity = study_heterogeneity(adjusted, context)
    discordant_sensitivity = assay_discordant_specimen_sensitivity(adjusted, context)
    arithmetic_validation = validate_three_specification_arithmetic(adjusted, context)
    make_figure(adjusted)
    print(f"Jointly profiled pair tests: {len(out):,} across {out.cancer.nunique()} cancers")
    print(f"Full-cohort FDR < 0.05: {(out.full_fdr < 0.05).sum():,}")
    for specification, label in (
        ("noBurden", "Primary no-burden"),
        ("leaveTwoOut", "Leave-two-out burden sensitivity"),
        ("totalBurden", "Original total-burden diagnostic"),
    ):
        print(
            f"{label}: FDR < 0.05 = "
            f"{adjusted[f'{specification}_full_fdr'].lt(0.05).sum():,}; "
            f"cross-assay concordant = "
            f"{adjusted[f'{specification}CrossAssayConcordant'].fillna(False).sum():,}"
        )
    no_log = np.log2(adjusted.noBurden_full_or.where(adjusted.noBurden_full_or > 0))
    leave_log = np.log2(adjusted.leaveTwoOut_full_or.where(adjusted.leaveTwoOut_full_or > 0))
    estimable_stability = no_log.notna() & leave_log.notna()
    stable = adjusted.signStableNoBurdenLeaveTwoOut.fillna(False)
    print(
        "Primary/leave-two-out direction stable: "
        f"{stable.sum():,}/{estimable_stability.sum():,} jointly estimable rows; "
        "effect stable (within two-fold, or concordantly strong): "
        f"{adjusted.effectStableNoBurdenLeaveTwoOut.fillna(False).sum():,}"
    )
    print("Top primary cross-assay-concordant pairs:")
    print(
        adjusted[adjusted.cmhReplicated]
        .sort_values("cmh_full_p")
        .head(15)[["cancer", "pair", "full_n", "full_nBoth", "cmh_wes_or", "cmh_panel_or", "cmh_full_fdr"]]
        .to_string(index=False, float_format=lambda x: f"{x:.3g}")
    )
    print(
        f"Study diagnostics: {len(study_specific):,} study-specific estimates, "
        f"{len(leave_one_out):,} leave-one-study-out estimates and "
        f"{len(heterogeneity):,} context summaries"
    )
    print(
        "Assay-discordant-specimen exclusion sensitivity: "
        f"{len(discordant_sensitivity):,} displayed/reference contexts"
    )
    print(
        f"Independent CMH arithmetic validation: {len(arithmetic_validation):,} "
        f"reference/specification/assay estimates PASS; maximum |OR difference| "
        f"{arithmetic_validation.absOrDifference.max():.3g}"
    )
    print("Wrote three-specification interaction, study-heterogeneity and leave-one-study-out tables")


if __name__ == "__main__":
    main()
