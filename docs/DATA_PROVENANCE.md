# Data provenance

## Public cancer-genomics cohort

cBioPortal molecular and clinical records were retrieved from 10 to 13 June 2026 using
the public REST API. The study screen examined 535 portal records, retained 428 input
studies and yielded 367 contributing studies after specimen reconciliation,
biospecimen selection and one-tumour-per-patient selection.

Cancer taxonomy was resolved against OncoTree release 2025-10-03. The selected cohort
contains 132,181 tissue tumours, 89 analysis families and 673 detailed OncoTree codes.

Documented WES/WGS specimens are treated as assay-covered for every compendium gene;
targeted-panel specimens are assay-covered only for genes present in the documented
panel definition. Because uniform per-locus read depth and quality metrics are not
available, this is an assay-scope definition rather than verified technical
callability. Positive records outside the documented panel scope are retained in a
separate audit and are not used to reconstruct outcome-dependent negative denominators.

## Mutation resources

- COSMIC Cancer Gene Census version 104 supplied cancer-gene membership, role and tier.
- COSMIC Cancer Mutation Census version 104 supplied mutation-level evidence.
- OncoKB supplied cancer-gene annotations through its public endpoint.
- The gene classifications from the 2023 pan-cancer landscape were included for direct
  analytical continuity.

COSMIC exports require provider access and are not distributed in this repository.

## Pathway and hallmark resources

Oncogenic pathway membership and mutation roles come from Supplementary Table S3 of
Sanchez-Vega et al. (Cell, 2018; PMID: 29625050). COSMIC hallmark memberships come from
the Cancer Gene Census version 104 export. Pathway and hallmark denominators use the
54,249 selected tumours with documented WES/WGS assignment.

## Functional resources

The functional analysis uses DepMap Public 26Q1 `CRISPRGeneEffect.csv`, `Model.csv` and
`OmicsSomaticMutationsMatrixHotspot.csv`. Pharmacological-response analysis uses the
PRISM Repurposing Secondary Screen 25Q2 AUC matrix and compound annotations. These large
source matrices must be obtained from their providers for a full run and are excluded
from this repository.

## Network resource

Displayed gene pairs are annotated against a directed human interactome containing
regulatory, transcriptional, activation, inhibition and post-translational relations.
The interactome workbook is supplied separately at the path documented in
`data/external/README.md`. Direct and two-step connectivity are compared with 5,000
random pair sets matched on interactome degree and Cancer Gene Census membership.

## Packaged frozen outputs

The `results/tables/` files retain cohort counts, denominators, estimates, confidence
intervals and multiple-testing results required by the executable notebook. The full
74,582-row gene-pair screen is included with the primary no-burden, leave-two-out and
total-burden specifications. Compact study-heterogeneity, functional-
sensitivity, interactome-null and extended survival tables accompany it. The
`results/figures/` directory contains the nine main and seven supplementary figure
images generated from the same analysis state. Byte sizes and SHA-256 digests are frozen in
`config/results_manifest.tsv`.

These prepared outputs support an independent numerical and integrity audit without
redistributing patient-level or licensed upstream records. Full cohort reconstruction is
a separate gated workflow described in `docs/REPRODUCIBILITY.md`.
