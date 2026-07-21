"""Compute the assay-aware driver landscape and mutation-evidence summaries.

This analysis uses the biospecimen-filtered, one-sample-per-patient cohort created by
``01d_curate_cohort.py``. It generates four linked views:

1. assay-aware prevalence with Wilson 95% confidence intervals;
2. the fraction of observed mutation records that match COSMIC CMC significance tiers,
   explicitly retaining ``CMC other`` and ``not represented`` categories;
3. WES/WGS versus targeted-panel prevalence standardised to the same cancer mixture;
4. recurrent amino-acid changes for canonical positive-control genes.

No unmatched mutation is labelled a passenger and no pooled WES/panel comparison is
interpreted without case-mix standardization.
"""
from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.proportion import proportion_confint

from config import EXTERNAL, FIGURES, PROCESSED, TABLES
from plot_style import COLORS, apply as apply_style, figsize, panel_label, save_figure

warnings.filterwarnings("ignore")

MIN_CANCER_SAMPLES = 200
MIN_ASSAY_PER_CANCER = 50
HOTSPOT_GENES = ["KRAS", "PIK3CA", "BRAF", "TP53", "EGFR", "IDH1"]
ROLE_COLORS = {
    "Oncogenes": COLORS["vermillion"],
    "TSGs": COLORS["blue"],
    "Oncogene/TSG": COLORS["purple"],
    "Other": COLORS["grey"],
}
TIER_COLORS = {
    "CMC tier 1": COLORS["vermillion"],
    "CMC tier 2": COLORS["orange"],
    "CMC tier 3": COLORS["purple"],
    "CMC other": COLORS["sky"],
    "Not represented": COLORS["light_grey"],
}


def norm_aa(value) -> str:
    if not isinstance(value, str):
        return ""
    value = value.strip()
    return value[2:] if value.startswith("p.") else value


def cmc_evidence(mutations: pd.DataFrame, panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    lookup = pd.read_parquet(EXTERNAL / "cosmic_cmc_lookup.parquet")
    lookup = lookup[["gene", "aa", "tier"]].copy()
    lookup["aa"] = lookup.aa.map(norm_aa)
    lookup = lookup.sort_values("tier", key=lambda x: x.map({"1": 0, "2": 1, "3": 2, "Other": 3}).fillna(4))
    lookup = lookup.drop_duplicates(["gene", "aa"], keep="first")

    ent2sym = panel.set_index("entrezGeneId")["hugoSymbol"].to_dict()
    obs = mutations[["sampleId", "entrezGeneId", "proteinChange"]].copy()
    obs["gene"] = obs.entrezGeneId.map(ent2sym)
    obs["aa"] = obs.proteinChange.map(norm_aa)
    ann = obs.merge(lookup, on=["gene", "aa"], how="left")
    ann["evidenceCategory"] = ann.tier.map(
        {
            "1": "CMC tier 1",
            "2": "CMC tier 2",
            "3": "CMC tier 3",
            "Other": "CMC other",
        }
    ).fillna("Not represented")
    ann["tierMatched"] = ann.evidenceCategory.isin(["CMC tier 1", "CMC tier 2", "CMC tier 3"])

    rows = []
    for gene, group in ann.groupby("gene"):
        n = len(group)
        k = int(group.tierMatched.sum())
        lo, hi = proportion_confint(k, n, method="wilson") if n else (np.nan, np.nan)
        cats = group.evidenceCategory.value_counts()
        rows.append(
            {
                "gene": gene,
                "nMutationRecords": n,
                "nTierMatched": k,
                "pctTierMatched": 100 * k / n if n else np.nan,
                "tierMatchedCiLowPct": 100 * lo,
                "tierMatchedCiHighPct": 100 * hi,
                **{f"n{c.replace(' ', '').replace('-', '')}": int(cats.get(c, 0)) for c in TIER_COLORS},
            }
        )
    by_gene = pd.DataFrame(rows).sort_values("nMutationRecords", ascending=False)
    by_gene.to_csv(TABLES / "cmc_evidence_curated.csv", index=False)
    return by_gene, ann


def assay_by_cancer(
    samples: pd.DataFrame, mutations: pd.DataFrame, panel: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assay = pd.read_parquet(PROCESSED / "sample_assay.parquet")
    assay = assay.merge(
        samples[["sampleId", "studyId", "broadCancerCode"]],
        on=["sampleId", "studyId"],
        how="inner",
    )
    assay["assayGroup"] = np.select(
        [
            assay.assayType.str.startswith("WES/WGS", na=False),
            assay.assayType.eq("Targeted panel"),
        ],
        ["WES/WGS", "Targeted panel"],
        default="Unverified assay scope",
    )
    # Samples whose mutation-profile membership could not be verified are not
    # assigned to either assay stratum.  This prevents missing assay metadata from
    # being silently interpreted as targeted sequencing in the standardised
    # comparison.
    assay = assay.loc[assay.assayGroup.ne("Unverified assay scope")].copy()
    membership = pd.read_parquet(PROCESSED / "panel_gene_membership.parquet")
    member_panels = membership.groupby("entrezGeneId")["genePanelId"].apply(set).to_dict()

    cancer_sizes = assay.broadCancerCode.value_counts()
    cancers = cancer_sizes[cancer_sizes >= MIN_CANCER_SAMPLES].index
    muts = mutations.drop_duplicates(["sampleId", "entrezGeneId"]).merge(
        assay[["sampleId", "broadCancerCode", "assayGroup"]], on="sampleId", how="inner"
    )
    mut_counts = (
        muts.groupby(["broadCancerCode", "entrezGeneId", "assayGroup"])["sampleId"]
        .nunique()
        .rename("nMutated")
        .to_dict()
    )

    rows = []
    ents = panel.entrezGeneId.astype(int).tolist()
    for cancer in cancers:
        sub = assay[assay.broadCancerCode == cancer]
        n_wes = int((sub.assayGroup == "WES/WGS").sum())
        targeted = sub[sub.assayGroup == "Targeted panel"]
        panel_counts = targeted.genePanelId.value_counts()
        for ent in ents:
            n_panel = sum(int(panel_counts.get(pid, 0)) for pid in member_panels.get(ent, set()))
            rows.append(
                {
                    "cancer": cancer,
                    "entrezGeneId": ent,
                    "nWesProfiled": n_wes,
                    "nWesMutated": int(mut_counts.get((cancer, ent, "WES/WGS"), 0)),
                    "nPanelProfiled": n_panel,
                    "nPanelMutated": int(mut_counts.get((cancer, ent, "Targeted panel"), 0)),
                }
            )
    out = pd.DataFrame(rows)
    out["freqWesPct"] = 100 * out.nWesMutated / out.nWesProfiled.clip(lower=1)
    out["freqPanelPct"] = 100 * out.nPanelMutated / out.nPanelProfiled.clip(lower=1)
    if (out.nWesMutated > out.nWesProfiled).any() or (out.nPanelMutated > out.nPanelProfiled).any():
        raise AssertionError("An assay-stratified numerator exceeds its callable denominator")
    out = out.merge(panel[["entrezGeneId", "hugoSymbol"]], on="entrezGeneId", how="left").rename(
        columns={"hugoSymbol": "gene"}
    )
    out.to_csv(TABLES / "assay_frequency_by_cancer.csv", index=False)

    # Standardize both assays to identical cancer weights.  The per-cancer weight is the
    # smaller profiled group, preventing a cancer represented almost solely in one assay
    # from dominating either standardised estimate.
    standard_rows = []
    for (ent, gene), group in out.groupby(["entrezGeneId", "gene"], dropna=False):
        common = group[
            (group.nWesProfiled >= MIN_ASSAY_PER_CANCER)
            & (group.nPanelProfiled >= MIN_ASSAY_PER_CANCER)
        ].copy()
        if common.empty:
            continue
        weights = np.minimum(common.nWesProfiled, common.nPanelProfiled).astype(float)
        standard_rows.append(
            {
                "entrezGeneId": ent,
                "gene": gene,
                "nCommonCancers": len(common),
                "standardizationWeight": int(weights.sum()),
                "freqWesStandardizedPct": np.average(common.freqWesPct, weights=weights),
                "freqPanelStandardizedPct": np.average(common.freqPanelPct, weights=weights),
            }
        )
    standard = pd.DataFrame(standard_rows)
    standard.to_csv(TABLES / "assay_frequency_standardized.csv", index=False)
    return out, standard


def hotspot_table(ann: pd.DataFrame) -> pd.DataFrame:
    x = ann[ann.gene.isin(HOTSPOT_GENES) & ann.aa.ne("")].copy()
    x = x.drop_duplicates(["sampleId", "gene", "aa"])
    counts = (
        x.groupby(["gene", "aa", "evidenceCategory"])["sampleId"]
        .nunique()
        .rename("nSamples")
        .reset_index()
    )
    order = {g: i for i, g in enumerate(HOTSPOT_GENES)}
    top = (
        counts.sort_values(["gene", "nSamples"], ascending=[True, False])
        .groupby("gene", group_keys=False)
        .head(4)
        .copy()
    )
    top["geneOrder"] = top.gene.map(order)
    top = top.sort_values(["geneOrder", "nSamples"], ascending=[True, False])
    top.to_csv(TABLES / "mutation_hotspots_curated.csv", index=False)
    return top


def make_figure(
    prevalence: pd.DataFrame,
    evidence: pd.DataFrame,
    standard: pd.DataFrame,
    hotspots: pd.DataFrame,
) -> None:
    apply_style()
    fig = plt.figure(figsize=figsize(180, 176))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[0.92, 1.08], hspace=0.38, wspace=0.40)

    ax_a = fig.add_subplot(gs[0, 0])
    top = prevalence.head(22).sort_values("freqPct")
    y = np.arange(len(top))
    xerr = np.vstack([top.freqPct - top.freqCiLowPct, top.freqCiHighPct - top.freqPct])
    colours = [ROLE_COLORS.get(x, COLORS["grey"]) for x in top.roleInCancer]
    ax_a.errorbar(top.freqPct, y, xerr=xerr, fmt="none", ecolor=COLORS["grey"], elinewidth=0.65, capsize=1.5)
    ax_a.scatter(top.freqPct, y, c=colours, s=15, edgecolors="white", linewidths=0.3, zorder=3)
    ax_a.set_yticks(y); ax_a.set_yticklabels(top.gene)
    ax_a.set_xlabel("Assay-aware prevalence (%, Wilson 95% CI)")
    ax_a.set_title("Curated driver-gene prevalence", loc="left")
    handles = [
        plt.Line2D([0], [0], marker="o", ls="", color=c, label=lab, markersize=4)
        for lab, c in ROLE_COLORS.items()
        if lab != "Other"
    ]
    ax_a.legend(handles=handles, frameon=False, ncol=1, loc="lower right")
    panel_label(ax_a, "a")

    ax_b = fig.add_subplot(gs[0, 1])
    merged = prevalence.merge(evidence, on="gene", how="inner")
    plot = merged[(merged.nMutationRecords >= 40) & merged.freqPct.gt(0)]
    sc = ax_b.scatter(
        plot.freqPct,
        plot.pctTierMatched,
        s=np.clip(np.sqrt(plot.nMutationRecords) * 1.2, 5, 34),
        c=plot.pctCuratedSamplesProfilingGene,
        cmap="viridis",
        vmin=30,
        vmax=100,
        alpha=0.72,
        edgecolors="white",
        linewidths=0.25,
    )
    label_genes = {
        "TP53", "KRAS", "PIK3CA", "APC", "BRAF", "IDH1", "KMT2D",
        "MUC16", "CSMD3", "FAT3",
    }
    offsets = {
        "TP53": (-22, -8), "KRAS": (3, 3), "PIK3CA": (3, -7),
        "APC": (3, 3), "BRAF": (3, 3), "IDH1": (3, 3), "KMT2D": (3, -7),
        "MUC16": (3, 3), "CSMD3": (3, 3), "FAT3": (3, -7),
    }
    for row in plot[plot.gene.isin(label_genes)].itertuples(index=False):
        ax_b.annotate(row.gene, (row.freqPct, row.pctTierMatched), xytext=offsets.get(row.gene, (2, 2)), textcoords="offset points", fontsize=4.5)
    cb = fig.colorbar(sc, ax=ax_b, fraction=0.038, pad=0.02)
    cb.set_label("Curated cohort profiling gene (%)", fontsize=5.5)
    ax_b.set_xlabel("Assay-aware prevalence (%)")
    ax_b.set_ylabel("Mutation records matching CMC tier 1-3 (%)")
    ax_b.set_xlim(left=-1, right=max(plot.freqPct.max() * 1.10, 40))
    ax_b.set_title("Prevalence and mutation-level evidence are distinct", loc="left")
    panel_label(ax_b, "b")

    ax_c = fig.add_subplot(gs[1, 0])
    compare = standard[(standard.nCommonCancers >= 2)].merge(
        prevalence[["gene", "nMutated"]], on="gene", how="left"
    )
    ax_c.scatter(
        compare.freqWesStandardizedPct,
        compare.freqPanelStandardizedPct,
        s=np.clip(np.sqrt(compare.nMutated), 3, 19),
        c=compare.nCommonCancers,
        cmap="cividis",
        alpha=0.55,
        edgecolors="none",
    )
    lim = max(compare.freqWesStandardizedPct.max(), compare.freqPanelStandardizedPct.max()) * 1.05
    ax_c.plot([0, lim], [0, lim], color=COLORS["grey"], ls="--", lw=0.7)
    delta = (compare.freqPanelStandardizedPct - compare.freqWesStandardizedPct).abs()
    labels = set(compare.loc[delta.nlargest(4).index, "gene"]) | {"TP53", "KRAS", "PIK3CA", "CSMD3"}
    c_offsets = {
        "TP53": (3, 3), "KRAS": (3, -7), "PIK3CA": (3, -7), "CSMD3": (3, 3),
        "RNF213": (3, 3), "ROBO2": (3, -7), "TRRAP": (3, 3), "NRAS": (3, -7),
    }
    for row in compare[compare.gene.isin(labels)].itertuples(index=False):
        ax_c.annotate(
            row.gene,
            (row.freqWesStandardizedPct, row.freqPanelStandardizedPct),
            xytext=c_offsets.get(row.gene, (2, 2)),
            textcoords="offset points",
            fontsize=4.5,
        )
    rho, _ = spearmanr(compare.freqWesStandardizedPct, compare.freqPanelStandardizedPct)
    ax_c.text(0.04, 0.95, f"Spearman ρ={rho:.2f}", transform=ax_c.transAxes, va="top")
    ax_c.set_xlabel("WES/WGS prevalence, cancer-standardised (%)")
    ax_c.set_ylabel("Targeted-panel prevalence, same cancer weights (%)")
    ax_c.set_title("Assay strata agree after cancer standardization", loc="left")
    panel_label(ax_c, "c")

    ax_d = fig.add_subplot(gs[1, 1])
    hs = hotspots.copy()
    hs["label"] = hs.gene + " " + hs.aa
    hs = hs.iloc[::-1].reset_index(drop=True)
    yy = np.arange(len(hs))
    ax_d.barh(yy, hs.nSamples, color=[TIER_COLORS.get(x, COLORS["grey"]) for x in hs.evidenceCategory])
    ax_d.set_yticks(yy); ax_d.set_yticklabels(hs.label, fontsize=5.2)
    ax_d.set_xlabel("Curated samples carrying amino-acid change")
    ax_d.set_title("Canonical recurrent amino-acid changes", loc="left")
    # Fine separators preserve the six-gene grouping without adding another legend.
    changes = np.where(hs.gene.to_numpy()[1:] != hs.gene.to_numpy()[:-1])[0] + 0.5
    for boundary in changes:
        ax_d.axhline(boundary, color=COLORS["light_grey"], lw=0.45, zorder=0)
    handles = [
        plt.Line2D([0], [0], marker="s", ls="", color=c, label=lab, markersize=4)
        for lab, c in TIER_COLORS.items()
        if lab in set(hs.evidenceCategory)
    ]
    ax_d.legend(handles=handles, frameon=False, ncol=2, loc="lower right")
    panel_label(ax_d, "d")

    source = TABLES.parent / "source_data"
    source.mkdir(parents=True, exist_ok=True)
    prevalence.to_csv(source / "driver_screen_prevalence.csv", index=False)
    merged.to_csv(source / "driver_screen_cmc_evidence.csv", index=False)
    standard.to_csv(source / "driver_screen_assay_standardisation.csv", index=False)
    hotspots.to_csv(source / "driver_screen_hotspots.csv", index=False)
    save_figure(fig, FIGURES / "diagnostics" / "driver_evidence_screen")
    plt.close(fig)


def main() -> None:
    panel = pd.read_csv(PROCESSED / "gene_panel.csv")
    samples = pd.read_parquet(PROCESSED / "analysis_samples_curated.parquet")
    samples = samples[samples.analysisEligible].copy()
    mutations = pd.read_parquet(PROCESSED / "mutations_curated.parquet")
    prevalence = pd.read_csv(TABLES / "gene_frequencies_curated.csv")

    evidence, annotated = cmc_evidence(mutations, panel)
    _, standard = assay_by_cancer(samples, mutations, panel)
    hotspots = hotspot_table(annotated)
    make_figure(prevalence, evidence, standard, hotspots)

    overall = 100 * annotated.tierMatched.mean()
    print(f"Curated mutation records matching CMC tier 1-3: {overall:.1f}%")
    print(f"Assay-standardised genes: {len(standard):,}")
    print("Top recurrent amino-acid changes:")
    print(hotspots.head(20).to_string(index=False))
    print("Wrote curated driver-evidence tables and diagnostic plots")


if __name__ == "__main__":
    main()
