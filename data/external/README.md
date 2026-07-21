# External inputs for full raw-data reconstruction

The frozen-output audit and notebook do not require these files. Rebuilding the cohort,
statistical models and figures requires the resources below at the stated
repository-relative paths. The machine-readable source, access and redistribution record
is [`config/data_manifest.json`](../../config/data_manifest.json).

| Resource | Expected path |
|---|---|
| COSMIC Cancer Gene Census v104 export | `data/external/cosmic_cgc_v104.csv` |
| COSMIC Cancer Gene Census hallmark export | `data/external/cosmic_cgc_hallmarks.tsv` |
| Gene classifications used in the earlier 2023 analysis | `data/external/prior2023_gene_classes.csv` |
| COSMIC Cancer Mutation Census v104 archive | `data/external/cosmic/CancerMutationCensus_AllData_Tsv_v104_GRCh37.tar` |
| Cross-study specimen-overlap mapping | `data/external/references/cross_study_sample_overlap_whitelist_2026-07-18.csv` |
| OncoTree 2025-10-03 taxonomy snapshot | `data/external/references/oncotree_2025_10_03.csv` |
| Sample-level taxonomy mappings | `data/external/references/reviewed_sample_taxonomy_adjudications_2026-07-18.csv` |
| Sanchez-Vega pathway Table S3 | `data/external/references/sanchez_vega_2018_table_s3.xlsx` |
| DepMap model metadata | `data/external/depmap/Model.csv` |
| DepMap Chronos gene effects | `data/external/depmap/CRISPRGeneEffect.csv` |
| DepMap somatic hotspot matrix | `data/external/depmap/OmicsSomaticMutationsMatrixHotspot.csv` |
| PRISM repurposing AUC matrix | `data/external/depmap/REPURPOSINGAUCMatrix.csv` |
| PRISM compound annotations | `data/external/depmap/REPURPOSINGResponseCurves.csv` |
| Directed human interactome | `data/external/interactome/Human_Interactome.xlsx` |

The cBioPortal study, sample, mutation, copy-number, molecular-profile and clinical
records are retrieved from the public REST API during the full run. OncoKB gene
annotations are retrieved from its public endpoint by `src/02_build_gene_panel.py`.

The full sequence defines sample-level assay scope and targeted-panel gene membership
before cohort curation, and retrieves high-level copy-number calls before the mutation
and copy-number datasets are filtered to the final tumour cohort.

Access and use each source under the provider's terms. Keep credentials outside this
repository. Run `python src/run_all.py --dry-run` from the repository root to report every
missing gated file before starting a long reconstruction.
