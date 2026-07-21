"""Versioned cancer taxonomy for the pan-cancer cohort.

The portal supplies a mixture of current OncoTree codes, earlier codes, verbose disease
labels and study-level fallbacks.  This module resolves those fields against the frozen
OncoTree 2025-10-03 snapshot.  It never invents acronyms.  Every sample receives:

* the most specific supported current OncoTree code (``analysisCancerCode``);
* the official OncoTree organ root;
* a reviewed, stable analysis family (``cancerFamilyCode``); and
* a resolution method suitable for row-level audit.

Current supplied OncoTree codes take precedence over prose labels.  This is important for
known records such as CPTAC LUSC, where a contradictory broad prose label says lung
adenocarcinoma but the study identity and supplied OncoTree code both indicate LUSC.
"""
from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

from config import EXTERNAL


ONCOTREE_VERSION = "oncotree_2025_10_03"
ONCOTREE_PATH = EXTERNAL / "references" / f"{ONCOTREE_VERSION}.csv"
ONCOTREE_SHA256 = "c4abff7beb654b1557e5c470de90eda0d1f8763e9ae50b889b93d9d60cbfd48a"
SAMPLE_OVERRIDE_PATH = (
    EXTERNAL / "references" / "reviewed_sample_taxonomy_adjudications_2026-07-18.csv"
)

MISSING_CODE_STRINGS = {"", "N/A", "NA", "NONE", "NULL", "UNK", "UNKNOWN", "NOT"}

# Earlier portal codes mapped against the 2025-10-03 hierarchy. Ambiguous diffuse-glioma
# glioma categories resolve to the current diffuse-glioma parent rather than being forced
# into a modern molecular subtype.
EARLIER_CODE_ALIASES = {
    "GBM": "GB",
    "AASTR": "ASTR3",
    "AODG": "ODG3",
    "OAST": "DIFG",
    "AOAST": "DIFG",
    "DIPG": "DMG",
    "ALL": "LYMPH",
    "BALL": "BLLNOS",
    "TALL": "TLL",
    "TMT": "TT",
    "BCL": "LYMPH",
    "LEUK": "MYELOID",
    "CCPRCC": "CCPRC",
    "ATL": "ATLL",
}


def normalise_label(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().casefold()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# Portal prose variants that are not exact OncoTree names.  Values are current official
# codes.  These are deliberately explicit, so a newly encountered label remains unresolved
# until it is reviewed rather than generating an accidental acronym.
DETAIL_LABEL_ALIASES = {
    normalise_label(name): code
    for name, code in {
        "Liver Hepatocellular Carcinoma": "HCC",
        "Gastric Adenocarcinoma": "STAD",
        "Breast Invasive Carcinoma": "BRCA",
        "Gallbladder Adenocarcinoma": "GBAD",
        "Adenocarcinoma of Gallbladder": "GBAD",
        "Diffuse Large B-Cell Lymphoma": "DLBCLNOS",
        "Small Bowel Adenocarcinoma": "SBC",
        "Undifferentiated Pleomorphic Sarcoma/Maligant Fibrous Histiocytoma/High-Grade Spindle Cell Sarcoma": "MFH",
        "Undifferentiated Pleomorphic Sarcoma Malignant Fibrous Histiocytoma": "MFH",
        "Soft Tissue Sarcoma Other": "SARCNOS",
        "Urothelial Carcinoma Other": "BLADDER",
        "Melanoma Other": "MEL",
        "Bone Sarcoma Other": "BONE",
        "Head and Neck Carcinoma Other": "HEAD_NECK",
        "Endometrial Adenocarcinoma": "UCEC",
        "Liver Hepatocellular Carcinoma plus Intrahepatic Cholangiocarcinoma": "HCCIHCH",
        "MiT family translocation renal cell carcinoma": "RCC",
        "Ovarian Serous Carcinoma": "SOC",
        "Unclassified Kidney Renal Cell Carcinoma": "URCC",
        "Papillary Throid Carcinoma": "THPA",
        "Papillary Kidney Renal Cell Carcinoma": "PRCC",
        "Breast Carcinoma Other": "BRCA",
        "Uterine Serous Carcinoma": "USC",
        "Non Small Cell Lung Cancer Other": "NSCLC",
        "Kidney Renal Cell Carcinoma Other": "RCC",
        "Ovarian Adenocarcinoma": "OVT",
        "Gallbladder Carcinoma Other": "GBC",
        "Esophageal Carcinoma Other": "EGC",
        "Pancreatic Cancer Other": "PANCREAS",
        "Colorectal Carcinoma Other": "COADREAD",
        "Thyroid Carcinoma Other": "THYROID",
        "Glioblastoma Multiforme": "GB",
        "Glioblastoma": "GB",
        "Anaplastic Astrocytoma": "ASTR3",
        "Anaplastic Oligodendroglioma": "ODG3",
        "Oligoastrocytoma": "DIFG",
        "Anaplastic Oligoastrocytoma": "DIFG",
        "Acute Lymphoid Leukemia": "LYMPH",
        "Acute Lymphoblastic Leukemia": "LYMPH",
        "B-Cell Acute Lymphoid Leukemia": "BLLNOS",
        "T-Cell Acute Lymphoid Leukemia": "TLL",
        "Diffuse Intrinsic Pontine Glioma": "DMG",
        "Teratoma with Malignant Transformation": "TT",
        # Reviewed labels in releases that omit a sample-level OncoTree code.  These
        # aliases prevent explicit diagnoses in mixed or sarcoma studies from falling
        # through to the study-level MIXED/SOFT_TISSUE umbrella.
        "Fibroblastic Myofibroblastic Tumor": "IMT",
        "Myxoid Fibrosarcoma": "MFS",
        "Mucinous Liposarcoma": "MRLS",
        "Gastric Carcinoma Other": "STOMACH",
        "Spindle Cell Sarcoma": "SARCNOS",
        "Fibromatosis": "DES",
        "Undifferentiated Sarcoma": "MFH",
        "Gastric Neuroendocrine Tumor": "GINETES",
        "Solitary Fibrous Tumor": "SFT",
        "Unclassified Sarcoma": "SARCNOS",
        "Malignant Small Cell Tumor": "SARCNOS",
        "Colorectal Neuroendocrine Tumor": "GINET",
        "Gallbladder Neuroendocrine Tumor": "GBC",
        "Gallbladder Adenosquamous Carcinoma": "GBASC",
        "Adenosquamous Carcinoma of Gallbladder": "GBASC",
        "Esophageal Adenosquamous Carcinoma": "EGC",
        "Chromophobe Kidney Renal Cell Carcinoma": "CHRCC",
        "Ovarian Carcinoma Other": "OOVC",
        "Medullary Thyroid Carcinoma": "THME",
        "Uterine Corpus Endometrial Carcinoma Other": "UCEC",
        "Ovarian Granulosa Cell Tumor": "GRCT",
        "Small Bowel Neuroendocrine Tumor": "GINET",
        "Esophageal Neuroendocrine Tumor": "GINETES",
        "Extrarenal Rhabdoid Tumor": "SARCNOS",
        "Intraductal Papillary Mucinous Neoplasm with an associated Invasive Carcinoma": "IPMN",
        "Breast Microinvasive Carcinoma": "BRCNOS",
        "Follicular Thyroid Carcinoma": "THFO",
        "Round Cell Sarcoma, Other": "RCSNOS",
        "Extraskeletal Osteosarcoma": "OS",
        "Paraosteal Osteosarcoma": "PAOS",
        "Myxoinflammatory Fibroblastic Sarcoma": "SARCNOS",
        "Infantile Sarcoma": "SARCNOS",
        "Small Cell Neuroendocrine Carcinoma of Gallbladder": "SCGBC",
        "Squamous Cell Carcinoma of Gallbladder": "GBC",
        "Mixed Adenocarcinoma and Neuroendocrine Carcinoma": "GBC",
        "Large Cell Neuroendocrine Carcinoma of Gallbladder": "GBC",
    }.items()
}


# Stable analysis-family code for common OncoTree main types.  Rare unmapped main types use
# the official OncoTree root code, never an automatically constructed abbreviation.
FAMILY_BY_MAIN_TYPE = {
    "Adrenocortical Carcinoma": ("ACC", "Adrenocortical carcinoma"),
    "Ampullary Cancer": ("AMPCA", "Ampullary cancer"),
    "Anal Cancer": ("ANAL", "Anal cancer"),
    "Appendiceal Cancer": ("APPX", "Appendiceal cancer"),
    "Bladder Cancer": ("BLCA", "Bladder cancer"),
    "Bladder/Urinary Tract Cancer": ("BLCA", "Bladder and urinary tract cancer"),
    "Bone Cancer": ("BONE", "Bone cancer"),
    "Breast Cancer": ("BRCA", "Breast cancer"),
    "Cancer of Unknown Primary": ("CUP", "Cancer of unknown primary"),
    "Cervical Cancer": ("CESC", "Cervical cancer"),
    "Colorectal Cancer": ("COADREAD", "Colorectal cancer"),
    "Endometrial Cancer": ("UCEC", "Endometrial cancer"),
    "Gastrointestinal Neuroendocrine Tumor": ("GINET", "Gastrointestinal neuroendocrine tumour"),
    "Germ Cell Tumor": ("GCT", "Germ-cell tumour"),
    "Gastrointestinal Stromal Tumor": ("GIST", "Gastrointestinal stromal tumour"),
    "Head and Neck Cancer": ("HNSC", "Head and neck cancer"),
    "Histiocytosis": ("HIST", "Histiocytosis"),
    "Hodgkin Lymphoma": ("HL", "Hodgkin lymphoma"),
    "Leukemia": ("LEUK", "Leukaemia"),
    "B-Lymphoblastic Leukemia/Lymphoma": ("BLL", "B-lymphoblastic leukaemia/lymphoma"),
    "T-Lymphoblastic Leukemia/Lymphoma": ("TLL", "T-lymphoblastic leukaemia/lymphoma"),
    "Mastocytosis": ("MAST", "Mastocytosis"),
    "Mature B-Cell Neoplasms": ("MBN", "Mature B-cell neoplasm"),
    "Mature T and NK Neoplasms": ("TNKN", "Mature T- and NK-cell neoplasm"),
    "Mesothelioma": ("MESO", "Mesothelioma"),
    "Myelodysplastic Syndromes": ("MDS", "Myelodysplastic syndrome"),
    "Myelodysplastic/Myeloproliferative Neoplasms": ("MDS_MPN", "Myelodysplastic/myeloproliferative neoplasm"),
    "Myeloproliferative Neoplasms": ("MPN", "Myeloproliferative neoplasm"),
    "Non-Hodgkin Lymphoma": ("NHL", "Non-Hodgkin lymphoma"),
    "Non-Small Cell Lung Cancer": ("NSCLC", "Non-small-cell lung cancer"),
    "Ovarian Cancer": ("OV", "Ovarian cancer"),
    "Pancreatic Cancer": ("PAAD", "Pancreatic cancer"),
    "Pheochromocytoma": ("PCPG", "Phaeochromocytoma"),
    "Prostate Cancer": ("PRAD", "Prostate cancer"),
    "Retinoblastoma": ("RBL", "Retinoblastoma"),
    "Salivary Gland Cancer": ("SG", "Salivary gland cancer"),
    "Skin Cancer, Non-Melanoma": ("NMSC", "Non-melanoma skin cancer"),
    "Small Bowel Cancer": ("SBWL", "Small-bowel cancer"),
    "Small Cell Lung Cancer": ("SCLC", "Small-cell lung cancer"),
    "Soft Tissue Sarcoma": ("SARC", "Soft-tissue sarcoma"),
    "Testicular Cancer": ("TGCT", "Testicular cancer"),
    "Thymic Cancer": ("THYM", "Thymic cancer"),
    "Thymic Tumor": ("THYM", "Thymic tumour"),
    "Thyroid Cancer": ("THCA", "Thyroid cancer"),
    "Uterine Sarcoma": ("USARC", "Uterine sarcoma"),
}

# Enforce one display label per family code even when different official main types map to
# the same reviewed family.
FAMILY_DISPLAY_LABELS = {
    code: label for code, label in FAMILY_BY_MAIN_TYPE.values()
}
FAMILY_DISPLAY_LABELS.update(
    {
        "BLCA": "Bladder and urinary tract cancer",
        "THYM": "Thymic tumour",
        "SKCM": "Cutaneous and unspecified melanoma",
        "PCPG": "Phaeochromocytoma/paraganglioma",
        "OV": "Ovarian cancer",
        "UCS": "Uterine carcinosarcoma",
        "TGCT": "Testicular germ-cell tumour",
    }
)


@lru_cache(maxsize=1)
def load_oncotree() -> pd.DataFrame:
    if not ONCOTREE_PATH.exists():
        raise FileNotFoundError(f"Frozen OncoTree snapshot is missing: {ONCOTREE_PATH}")
    digest = hashlib.sha256(ONCOTREE_PATH.read_bytes()).hexdigest()
    if digest != ONCOTREE_SHA256:
        raise ValueError(
            f"Frozen OncoTree digest mismatch: expected {ONCOTREE_SHA256}, observed {digest}"
        )
    tree = pd.read_csv(ONCOTREE_PATH, dtype={"code": str})
    required = {"code", "name", "mainType", "tissue", "rootCode", "depth", "path"}
    if not required.issubset(tree.columns):
        raise ValueError(f"OncoTree snapshot lacks columns: {sorted(required - set(tree.columns))}")
    tree["code"] = tree.code.astype(str).str.strip().str.upper()
    if tree.code.duplicated().any():
        raise ValueError("Frozen OncoTree contains duplicate codes")
    return tree


@lru_cache(maxsize=1)
def unique_name_index() -> dict[str, str]:
    tree = load_oncotree()
    candidates: dict[str, set[str]] = {}
    for row in tree.itertuples(index=False):
        candidates.setdefault(normalise_label(row.name), set()).add(row.code)
    return {label: next(iter(codes)) for label, codes in candidates.items() if len(codes) == 1}


@lru_cache(maxsize=1)
def oncotree_lookup() -> pd.DataFrame:
    """Return the frozen tree indexed by current code."""
    return load_oncotree().set_index("code")


def clean_code(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    code = str(value).strip().upper()
    return None if code in MISSING_CODE_STRINGS else code


@lru_cache(maxsize=1)
def sample_overrides() -> dict[tuple[str, str], dict[str, str]]:
    """Return explicitly adjudicated sample-level code overrides."""
    if not SAMPLE_OVERRIDE_PATH.exists():
        return {}
    frame = pd.read_csv(SAMPLE_OVERRIDE_PATH, dtype=str).fillna("")
    required = {"studyId", "sourceSampleId", "action", "adjudicatedOncoTreeCode"}
    if not required.issubset(frame.columns):
        raise ValueError(
            f"Sample taxonomy adjudications lack columns: {sorted(required - set(frame.columns))}"
        )
    selected = frame.loc[frame.action.eq("override")].copy()
    if selected.duplicated(["studyId", "sourceSampleId"]).any():
        raise ValueError("Duplicate sample-level taxonomy override")
    return {
        (row.studyId, row.sourceSampleId): row._asdict()
        for row in selected.itertuples(index=False)
    }


def label_code(value: object) -> tuple[str | None, str | None]:
    """Return the current code supported by one prose label, if reviewed."""
    label = normalise_label(value)
    if not label:
        return None, None
    name_index = unique_name_index()
    if label in name_index:
        return name_index[label], "exact label match"
    if label in DETAIL_LABEL_ALIASES:
        return DETAIL_LABEL_ALIASES[label], "reviewed label alias"
    return None, None


def is_descendant(child: str, parent: str, lookup: pd.DataFrame) -> bool:
    """Return whether ``child`` is a strict descendant of ``parent`` in OncoTree."""
    if child == parent or child not in lookup.index or parent not in lookup.index:
        return False
    return parent in str(lookup.at[child, "path"]).upper().split("/")


def resolve_code(row: pd.Series, valid_codes: set[str]) -> dict[str, object]:
    """Resolve a detailed code while preserving supported specificity.

    A supplied current code is not automatically allowed to erase a more-specific exact
    diagnosis: when the detailed-label candidate is a descendant of the supplied code,
    the descendant is used.  A more-specific supplied code is retained over a broad
    prose label.  Other-branch contradictions retain the supplied code unless the sample
    appears in the reviewed adjudication table.
    """
    lookup = oncotree_lookup()
    source_sample_id = str(row.get("sourceSampleId") or row.get("sampleId") or "")
    override = sample_overrides().get((str(row.get("studyId") or ""), source_sample_id))

    supplied_raw = clean_code(row.get("oncotreeCode"))
    supplied = supplied_raw if supplied_raw in valid_codes else EARLIER_CODE_ALIASES.get(supplied_raw)
    supplied_method = (
        "current supplied OncoTree code"
        if supplied_raw in valid_codes
        else "mapped earlier-code alias"
        if supplied is not None
        else None
    )
    detailed, detailed_method = label_code(row.get("cancerTypeDetailed"))
    cancer, cancer_method = label_code(row.get("cancerType"))

    if override:
        code = clean_code(override["adjudicatedOncoTreeCode"])
        if code not in valid_codes:
            raise ValueError(f"Reviewed override uses invalid OncoTree code: {code}")
        return {
            "code": code,
            "method": "reviewed sample-level adjudication",
            "supplied": supplied,
            "detailed": detailed,
            "relationship": "explicit adjudication of discordant source fields",
            "overrideApplied": True,
        }

    if supplied is not None:
        if detailed is None:
            relationship = "no supported detailed-label candidate"
        elif detailed == supplied:
            relationship = "supplied and detailed diagnosis agree"
        elif is_descendant(detailed, supplied, lookup):
            return {
                "code": detailed,
                "method": f"more-specific {detailed_method} descendant of supplied code",
                "supplied": supplied,
                "detailed": detailed,
                "relationship": "detailed diagnosis is descendant of supplied code",
                "overrideApplied": False,
            }
        elif is_descendant(supplied, detailed, lookup):
            relationship = "supplied code is descendant of detailed diagnosis"
        else:
            relationship = "discordant OncoTree branches; supplied code retained"
        return {
            "code": supplied,
            "method": supplied_method,
            "supplied": supplied,
            "detailed": detailed,
            "relationship": relationship,
            "overrideApplied": False,
        }

    if detailed is not None:
        return {
            "code": detailed,
            "method": (
                "exact detailed-label match"
                if detailed_method == "exact label match"
                else "reviewed cancerTypeDetailed alias"
            ),
            "supplied": None,
            "detailed": detailed,
            "relationship": "no supported supplied code; detailed diagnosis used",
            "overrideApplied": False,
        }
    if cancer is not None:
        return {
            "code": cancer,
            "method": (
                "exact cancer-label match"
                if cancer_method == "exact label match"
                else "reviewed cancerType alias"
            ),
            "supplied": None,
            "detailed": None,
            "relationship": "no supported supplied or detailed code; cancer label used",
            "overrideApplied": False,
        }

    study_raw = clean_code(row.get("studyCancerType"))
    study_code = study_raw if study_raw in valid_codes else EARLIER_CODE_ALIASES.get(study_raw)
    if study_code is not None:
        return {
            "code": study_code,
            "method": (
                "current study-level OncoTree fallback"
                if study_raw in valid_codes
                else "mapped study-level earlier-code alias"
            ),
            "supplied": None,
            "detailed": None,
            "relationship": "sample diagnosis unavailable; study-level code used",
            "overrideApplied": False,
        }
    return {
        "code": None,
        "method": "unresolved after reviewed fallbacks",
        "supplied": supplied,
        "detailed": detailed,
        "relationship": "no supported code",
        "overrideApplied": False,
    }


def code_has_ancestor(path: str, ancestor: str) -> bool:
    return ancestor in str(path).upper().split("/")


def family_for_official(row: pd.Series) -> tuple[str, str, str]:
    """Return family code, display name and resolution method for one tree row."""
    path = str(row["path"])
    current_code = str(row.name)

    # Direct-code decisions preserve conventional pan-cancer categories without
    # collapsing biologically distinct descendants of the same organ root.  In
    # particular, only the generic OVARY code joins the ovarian-cancer family; sex-cord
    # stromal descendants remain a separate residual group.
    direct = {
        "PGNG": (
            "PCPG",
            "Phaeochromocytoma/paraganglioma",
            "reviewed paraganglioma family",
        ),
        "OVARY": ("OV", "Ovarian cancer", "reviewed generic ovarian-cancer family"),
        "UCS": ("UCS", "Uterine carcinosarcoma", "reviewed uterine-carcinosarcoma family"),
        "SKCN": ("SKIN", "Skin", "reviewed benign skin residual"),
    }
    if current_code in direct:
        return direct[current_code]

    # Testicular germ-cell tumours use the conventional pan-cancer family code TGCT.
    # This is an analysis-family label, not the current detailed OncoTree code TGCT
    # (which denotes tenosynovial giant-cell tumour and remains in soft tissue).
    if str(row["rootCode"]) == "TESTIS" and str(row["mainType"]) == "Germ Cell Tumor":
        return (
            "TGCT",
            "Testicular germ-cell tumour",
            "reviewed testicular-root germ-cell family",
        )

    # Clinically recognizable histologies that must not be hidden inside broader portal
    # labels.  Descendant tests keep future subtypes attached to the correct family.
    overrides = (
        ("LUAD", "LUAD", "Lung adenocarcinoma"),
        ("LUSC", "LUSC", "Lung squamous-cell carcinoma"),
        ("SCLC", "SCLC", "Small-cell lung cancer"),
        ("STAD", "STAD", "Stomach adenocarcinoma"),
        ("ESCA", "ESCA", "Oesophageal adenocarcinoma"),
        ("ESCC", "ESCC", "Oesophageal squamous-cell carcinoma"),
        ("GEJ", "GEJ", "Gastro-oesophageal junction adenocarcinoma"),
        ("HCC", "HCC", "Hepatocellular carcinoma"),
        ("CHOL", "CHOL", "Cholangiocarcinoma"),
        ("GBC", "GBC", "Gallbladder cancer"),
        ("CCRCC", "CCRCC", "Clear-cell renal cell carcinoma"),
        ("PRCC", "PRCC", "Papillary renal cell carcinoma"),
        ("CHRCC", "CHRCC", "Chromophobe renal cell carcinoma"),
        ("GB", "GBM", "Portal-defined or legacy glioblastoma"),
        ("ASTR", "ASTR", "Astrocytoma, IDH-mutant"),
        ("ODG", "ODG", "Oligodendroglioma, IDH-mutant and 1p/19q-codeleted"),
        ("MEL", "SKCM", "Cutaneous and unspecified melanoma"),
        ("UM", "UM", "Uveal melanoma"),
        ("CLLSLL", "CLL", "Chronic lymphocytic leukaemia/small lymphocytic lymphoma"),
        ("DLBCLNOS", "DLBCL", "Diffuse large B-cell lymphoma"),
        ("PCM", "MM", "Plasma-cell myeloma"),
        ("AML", "AML", "Acute myeloid leukaemia"),
        ("ACYC", "ACYC", "Adenoid cystic carcinoma"),
        ("THYM", "THYM", "Thymoma"),
        ("COADREAD", "COADREAD", "Colorectal cancer"),
        ("UCEC", "UCEC", "Endometrial cancer"),
        ("BRCA", "BRCA", "Breast cancer"),
    )
    for ancestor, code, label in overrides:
        if code_has_ancestor(path, ancestor):
            return code, label, f"reviewed OncoTree ancestor {ancestor}"

    main_type = str(row["mainType"])
    if main_type in FAMILY_BY_MAIN_TYPE:
        code, label = FAMILY_BY_MAIN_TYPE[main_type]
        return code, label, "reviewed OncoTree main type"
    root = str(row["rootCode"])
    tissue = str(row["tissue"])
    return root, tissue, "official OncoTree root fallback"


def annotate_taxonomy(frame: pd.DataFrame) -> pd.DataFrame:
    """Return ``frame`` with frozen detailed and family cancer classifications."""
    tree = load_oncotree()
    valid = set(tree.code)
    lookup = oncotree_lookup()
    out = frame.copy()
    resolved = [resolve_code(row, valid) for _, row in out.iterrows()]
    out["resolvedOncoTreeCode"] = [value["code"] for value in resolved]
    out["taxonomyResolutionMethod"] = [value["method"] for value in resolved]
    out["suppliedOncoTreeCodeCanonical"] = [value["supplied"] for value in resolved]
    out["detailedDiagnosisCandidateCode"] = [value["detailed"] for value in resolved]
    out["taxonomyCodeRelationship"] = [value["relationship"] for value in resolved]
    out["taxonomyOverrideApplied"] = [bool(value["overrideApplied"]) for value in resolved]
    out["taxonomyVersion"] = ONCOTREE_VERSION

    official_columns = {
        "name": "resolvedCancerTypeDetailed",
        "mainType": "oncotreeMainType",
        "tissue": "oncotreeTissue",
        "rootCode": "oncotreeRootCode",
        "path": "oncotreePath",
    }
    for source, destination in official_columns.items():
        out[destination] = out.resolvedOncoTreeCode.map(lookup[source])

    family_lookup = {
        code: family_for_official(lookup.loc[code])
        for code in out.resolvedOncoTreeCode.dropna().unique()
        if code in lookup.index
    }
    family_values = [
        family_lookup.get(code, ("UNRESOLVED", "Unresolved cancer type", "unresolved"))
        for code in out.resolvedOncoTreeCode
    ]
    out["cancerFamilyCode"] = [value[0] for value in family_values]
    out["cancerFamilyName"] = [value[1] for value in family_values]
    out["cancerFamilyName"] = out.cancerFamilyCode.map(FAMILY_DISPLAY_LABELS).fillna(
        out.cancerFamilyName
    )
    out["cancerFamilyResolutionMethod"] = [value[2] for value in family_values]

    # Compatibility aliases used throughout the existing curated pipeline.
    out["analysisCancerCode"] = out.resolvedOncoTreeCode.fillna("UNRESOLVED")
    out["broadCancerCode"] = out.cancerFamilyCode
    out["broadCancerType"] = out.cancerFamilyName

    if out.analysisCancerCode.isna().any() or out.broadCancerCode.isna().any():
        raise AssertionError("Frozen taxonomy produced missing analysis codes")
    if out[["cancerFamilyCode", "cancerFamilyName"]].drop_duplicates().cancerFamilyCode.duplicated().any():
        raise AssertionError("A reviewed cancer-family code maps to more than one display label")
    return out
