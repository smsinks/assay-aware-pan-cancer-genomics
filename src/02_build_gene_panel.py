"""Stage 2 — assemble a multi-source cancer-gene compendium with provenance.

The compendium is the tagged union of three sources:

  1. COSMIC Cancer Gene Census v104  -- user-exported (login-gated) CSV at
     ``data/external/cosmic_cgc_v104.csv`` with tier, role and Entrez annotations.
  2. OncoKB cancer-gene list  -- PUBLIC API, refreshed + dated each run. geneType
     (oncogene/TSG/dual) plus Vogelstein / MSK-IMPACT / Foundation provenance flags.
  3. The curated 2023 panel  -- retained for direct comparability and for the
     kinase / phosphatase / transcription-factor / receptor sub-classes the other
     sources do not carry.

Every gene carries a `sources` string and per-source boolean flags, so nothing is
asserted without provenance (per the project's no-fabrication rule). Role precedence:
OncoKB (curated binary) > COSMIC > prior-2023.

Outputs:
  data/processed/gene_panel.csv               (the updated analysis panel)
  data/external/oncokb_cancer_gene_list.json  (raw OncoKB pull + access date)
  results/tables/gene_panel_provenance_summary.csv
"""
from __future__ import annotations

import json
from datetime import date

import pandas as pd

import cbioportal_client as cb
from config import EXTERNAL, PROCESSED, TABLES

# Opt into future pandas behaviour so boolean fillna+astype doesn't emit downcast warnings.
pd.set_option("future.no_silent_downcasting", True)

ONCOKB_URL = "https://www.oncokb.org/api/v1/utils/cancerGeneList"
COSMIC_CANDIDATES = [EXTERNAL / "cosmic_cgc_v104.csv", EXTERNAL / "cosmic_cgc_latest.csv"]

ROLE_MAP = {  # OncoKB geneType -> our vocabulary (kept compatible with 2023)
    "ONCOGENE": "Oncogenes", "TSG": "TSGs", "ONCOGENE_AND_TSG": "Oncogene/TSG",
    "INSUFFICIENT_EVIDENCE": "Other", "NEITHER": "Other", None: "Other",
}


def _cosmic_path():
    return next((p for p in COSMIC_CANDIDATES if p.exists()), None)


def _map_cosmic_role(raw: str) -> str:
    """Map COSMIC 'Role in Cancer' (e.g. 'oncogene, TSG, fusion') to our vocabulary."""
    r = raw.lower() if isinstance(raw, str) else ""
    onc, tsg = "oncogene" in r, "tsg" in r
    if onc and tsg:
        return "Oncogene/TSG"
    if onc:
        return "Oncogenes"
    if tsg:
        return "TSGs"
    return "Other"


def fetch_oncokb() -> pd.DataFrame:
    """Pull the OncoKB cancer-gene list and cache it with an access date."""
    import requests
    data = requests.get(ONCOKB_URL, timeout=60).json()
    (EXTERNAL / "oncokb_cancer_gene_list.json").write_text(
        json.dumps({"accessed": date.today().isoformat(), "url": ONCOKB_URL, "genes": data}))
    df = pd.DataFrame(data)
    df = df[df["entrezGeneId"] > 0].copy()
    df["roleOncoKB"] = df["geneType"].map(ROLE_MAP).fillna("Other")
    keep = {"hugoSymbol": "hugoSymbol", "entrezGeneId": "entrezGeneId",
            "roleOncoKB": "roleOncoKB", "sangerCGC": "inSangerCGC",
            "oncokbAnnotated": "oncokbAnnotated", "vogelstein": "vogelstein",
            "mSKImpact": "mskImpact", "foundation": "foundation"}
    out = df[list(keep)].rename(columns=keep)
    out["inOncoKB"] = True
    return out


def load_cosmic_census() -> pd.DataFrame | None:
    """Return COSMIC CGC tier, role and Entrez annotations when available."""
    path = _cosmic_path()
    if path is None:
        return None
    cgc = pd.read_csv(path, encoding="latin-1").rename(columns={
        "Gene Symbol": "hugoSymbol", "Tier": "cosmicTier",
        "Entrez GeneId": "cosmicEntrez", "Role in Cancer": "cosmicRoleRaw"})
    cgc = cgc.dropna(subset=["hugoSymbol"]).drop_duplicates("hugoSymbol")
    cgc["roleCOSMIC"] = cgc["cosmicRoleRaw"].map(_map_cosmic_role)
    cgc["cosmicTier"] = pd.to_numeric(cgc["cosmicTier"], errors="coerce").astype("Int64")
    cgc["inCosmicCGC"] = True
    print(f"COSMIC CGC ({path.name}): {len(cgc)} genes "
          f"(Tier1={int((cgc.cosmicTier==1).sum())}, Tier2={int((cgc.cosmicTier==2).sum())})")
    return cgc[["hugoSymbol", "cosmicEntrez", "cosmicTier", "roleCOSMIC", "inCosmicCGC"]]


def load_prior_classes() -> pd.DataFrame:
    """2023 curated set with its functional sub-classes (kinase/TF/receptor)."""
    df = pd.read_csv(EXTERNAL / "prior2023_gene_classes.csv").rename(columns={
        "HugoSymbol": "hugoSymbol", "RoleInCancer": "rolePrior",
        "Kinase_PPtase": "kinasePptase", "TF": "transcriptionFactor", "Location": "location"})
    df = df[df["hugoSymbol"].notna()].drop_duplicates("hugoSymbol")
    df["inPrior2023"] = True
    return df


def load_hallmarks() -> pd.DataFrame:
    h = pd.read_csv(EXTERNAL / "cosmic_cgc_hallmarks.tsv", sep="\t", encoding="latin-1")
    h["HALLMARK"] = h["HALLMARK"].str.lower().str.strip()
    valid = {"invasion and metastasis", "escaping programmed cell death",
             "proliferative signalling", "suppression of growth",
             "genome instability and mutations", "cell division control", "angiogenesis",
             "change of cellular energetics", "global regulation of gene expression",
             "tumour promoting inflammation", "cell replicative immortality",
             "escaping immune response to cancer", "differentiation and development"}
    h = h[h["HALLMARK"].isin(valid)]
    return (h.groupby("GENE_NAME")["HALLMARK"].agg(lambda s: ";".join(sorted(set(s))))
              .reset_index().rename(columns={"GENE_NAME": "hugoSymbol", "HALLMARK": "hallmarks"}))


def main() -> None:
    onco = fetch_oncokb()
    prior = load_prior_classes()
    cosmic = load_cosmic_census()
    print(f"OncoKB: {len(onco)} | prior-2023: {len(prior)} | "
          f"COSMIC: {0 if cosmic is None else len(cosmic)}")

    # Outer-union all sources on Hugo symbol.
    panel = onco.merge(prior[["hugoSymbol", "kinasePptase", "transcriptionFactor",
                              "location", "rolePrior", "inPrior2023"]],
                       on="hugoSymbol", how="outer")
    if cosmic is not None:
        panel = panel.merge(cosmic, on="hugoSymbol", how="outer")
    for flag in ["inOncoKB", "inPrior2023", "inCosmicCGC"]:
        if flag not in panel:
            panel[flag] = False
        panel[flag] = panel[flag].fillna(False).astype(bool)

    # Coalesce Entrez: OncoKB id, then COSMIC's, then portal lookup for the remainder.
    if "cosmicEntrez" in panel:
        panel["entrezGeneId"] = panel["entrezGeneId"].fillna(panel["cosmicEntrez"])
    missing = panel[panel["entrezGeneId"].isna()]
    if len(missing):
        mapped = pd.DataFrame(cb.map_symbols_to_entrez(missing["hugoSymbol"].tolist()))
        if len(mapped):
            mapped = mapped.rename(columns={"hugoGeneSymbol": "hugoSymbol"})[["hugoSymbol", "entrezGeneId"]]
            panel = panel.merge(mapped, on="hugoSymbol", how="left", suffixes=("", "_m"))
            panel["entrezGeneId"] = panel["entrezGeneId"].fillna(panel["entrezGeneId_m"])
            panel.drop(columns="entrezGeneId_m", inplace=True)

    # Role precedence: OncoKB > COSMIC > prior.
    panel["roleInCancer"] = (panel.get("roleOncoKB")
                             .fillna(panel.get("roleCOSMIC"))
                             .fillna(panel.get("rolePrior")).fillna("Other"))
    for flag in ["oncokbAnnotated", "vogelstein", "mskImpact", "foundation", "inSangerCGC"]:
        panel[flag] = panel[flag].fillna(False).astype(bool)

    panel = panel.merge(load_hallmarks(), on="hugoSymbol", how="left")
    if "cosmicTier" not in panel:
        panel["cosmicTier"] = pd.NA

    def _sources(r):
        s = []
        if r["inCosmicCGC"]: s.append(f"COSMIC_T{int(r['cosmicTier'])}" if pd.notna(r["cosmicTier"]) else "COSMIC")
        if r["inOncoKB"]: s.append("OncoKB")
        if r["inPrior2023"]: s.append("Prior2023")
        if r["vogelstein"]: s.append("Vogelstein")
        return ";".join(s)
    panel["sources"] = panel.apply(_sources, axis=1)

    # High-confidence: in COSMIC CGC, or OncoKB-classified oncogene/TSG, or in the 2023 set.
    panel["highConfidence"] = (
        panel["inCosmicCGC"] | panel["inPrior2023"] |
        panel["roleInCancer"].isin(["Oncogenes", "TSGs", "Oncogene/TSG"]))

    panel = panel[panel["entrezGeneId"].notna()].copy()
    panel["entrezGeneId"] = panel["entrezGeneId"].astype(int)
    keep = ["hugoSymbol", "entrezGeneId", "roleInCancer", "cosmicTier", "inCosmicCGC",
            "inSangerCGC", "inOncoKB", "inPrior2023", "oncokbAnnotated", "vogelstein",
            "mskImpact", "foundation", "kinasePptase", "transcriptionFactor", "location",
            "hallmarks", "sources", "highConfidence"]
    panel = panel[[c for c in keep if c in panel]].drop_duplicates("entrezGeneId").reset_index(drop=True)

    out = PROCESSED / "gene_panel.csv"
    panel.to_csv(out, index=False)

    summ = pd.DataFrame({
        "metric": ["total genes", "high-confidence", "COSMIC CGC", "COSMIC Tier 1",
                   "COSMIC Tier 2", "in OncoKB", "in prior-2023", "oncogenes", "TSGs",
                   "oncogene&TSG", "with hallmark mapping"],
        "count": [len(panel), int(panel.highConfidence.sum()), int(panel.inCosmicCGC.sum()),
                  int((panel.cosmicTier == 1).sum()), int((panel.cosmicTier == 2).sum()),
                  int(panel.inOncoKB.sum()), int(panel.inPrior2023.sum()),
                  int((panel.roleInCancer == "Oncogenes").sum()),
                  int((panel.roleInCancer == "TSGs").sum()),
                  int((panel.roleInCancer == "Oncogene/TSG").sum()),
                  int(panel.hallmarks.notna().sum())],
    })
    summ.to_csv(TABLES / "gene_panel_provenance_summary.csv", index=False)
    print("\n=== UPDATED GENE PANEL (COSMIC v104 + OncoKB + prior) ===")
    print(summ.to_string(index=False))
    print(f"\nNet new genes vs 2023 panel: {len(panel[~panel.inPrior2023])}")
    print(f"COSMIC-only genes added (not in OncoKB/prior): "
          f"{len(panel[panel.inCosmicCGC & ~panel.inOncoKB & ~panel.inPrior2023])}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
