# Pan-cancer assay-aware analysis of 132,181 tumours

This repository contains the analysis and reproducibility materials for an assay-aware
pan-cancer analysis of 132,181 selected tissue tumours from 367 cBioPortal studies. It
supports two deliberately separate workflows:

1. **Frozen-output audit (open and fast).** Recomputes headline summaries from the
   deposited analytical tables, validates numerical invariants, verifies SHA-256
   checksums and displays the deposited figures. It requires no network access,
   patient-level data, licensed source files or model refitting.
2. **Full raw-data reconstruction (gated and computationally intensive).** Retrieves
   public cBioPortal records, rebuilds the curated cohort, refits all models and
   regenerates the deposited outputs. This route runs only after the resources in
   [`config/data_manifest.json`](config/data_manifest.json) have been supplied.

The packaged CSV files and figures support independent numerical verification. Complete
cohort reconstruction additionally requires the controlled, licensed or very large
upstream resources listed in the data manifest.

## One-command frozen audit

Create the environment:

```bash
conda env create -f environment.yml
conda activate pancancer-mutational-landscape
```

Then run the complete stand-alone audit from the repository root:

```bash
make audit
```

`make audit` performs the frozen numerical audit, checks all redistributed files against
[`config/results_manifest.tsv`](config/results_manifest.tsv), runs the analytical
contract tests and scans the repository for machine-specific paths and generated
operating-system artefacts.

For a short environment smoke test without loading the complete frozen pairwise table,
run:

```bash
make test
```

The complete frozen audit normally finishes within minutes and the deposited repository
occupies approximately 80 MB. Full reconstruction depends on API throughput and the
authorised source matrices; allow many hours and tens of gigabytes for caches,
intermediate tables and model outputs.

An immutable pip installation is also available:

```bash
python -m pip install -r requirements-lock.txt
make audit
```

## Executable notebook

The notebook uses repository-relative paths and the compact frozen tables:

```bash
make notebook
```

This writes an executed copy to `build/pancancer_mutational_landscape.executed.ipynb`.
The tracked notebook contains no stored outputs or local filesystem paths. To work
interactively, run:

```bash
jupyter lab notebooks/pancancer_mutational_landscape.ipynb
```

## Full raw-data reconstruction

The full route is separate from the frozen audit. First place every file listed in
[`data/external/README.md`](data/external/README.md) at its documented relative path.
The data manifest records source versions, access restrictions and redistribution
status. Then run:

```bash
python src/run_all.py --dry-run
make full
```

The dry run fails early with an explicit list of missing resources. The reconstruction
also requires network access to cBioPortal and OncoKB, R 4.5.2 and the R `survival`
package 3.8-3. It can take many hours and uses substantially more storage than the
frozen audit. With no arguments, `src/run_all.py` begins at source acquisition and runs
the complete declared stage sequence rather than starting from pre-curated tables.

Every cBioPortal request made during a full reconstruction writes machine-readable
provenance containing the HTTP method, endpoint, parameters or request body, retrieval
time, response count, request and response hashes, cache filename and pagination state.
The `build_cbioportal_request_manifest.py` stage verifies the cached response hashes and
counts, rejects responses that reach a configured pagination boundary, and writes the
combined CSV and JSON request manifests under `results/`.

## Scientific outputs represented here

- 132,181 selected tissue tumours from 367 contributing studies.
- 89 reviewed cancer families, 673 detailed OncoTree codes and 1,341 genes.
- Explicit assay-covered (operationally callable), mutation-positive,
  assay-covered mutation-negative and unassayed states across 177,254,721 possible
  tumour–gene observations.
- Assay-aware gene prevalence and mutation-level evidence.
- Cancer-specific gene-pair associations under three explicit specifications: the
  primary detailed-histology, study and exact-assay model without mutation burden; a
  leave-two-out background-burden sensitivity; and a total-burden diagnostic.
- Study-specific, leave-one-study-out and heterogeneity summaries for the principal
  gene-pair contexts, plus strict off-panel-record sensitivities.
- Lineage-adjusted CRISPR dependency and PRISM pharmacological-response associations,
  including three-group comparator and leave-one-lineage-out analyses.
- Directed interaction-network annotations with a degree- and cancer-gene-matched
  permutation null.
- A complete, outcome-independent survival screen requiring both genes to reach 10%
  assay-aware prevalence within cancer-specific analyses or 5% pan-cancer prevalence,
  followed by endpoint and four-state group-size criteria. Separate joint genotype-state
  and formal multiplicative-interaction estimates are accompanied by proportional-hazards
  diagnostics, piecewise hazard ratios, restricted mean survival time differences,
  study-level meta-analyses and leave-one-study-out estimates.

The primary no-burden estimates for LUAD are OR = 0.0277 for EGFR–KRAS and OR = 10.83
for KEAP1–STK11. Across all 74,582 tested cancer–gene-pair contexts, 71,360 were
estimable under both the primary and leave-two-out models, 62,072 retained their effect
direction and 41,611 met the prespecified effect-stability criterion. The full screened
matrix is deposited, while the main figures retain selected contexts for readability.

## Repository layout

```text
.
├── CITATION.cff
├── LICENSE
├── Makefile
├── README.md
├── config/
│   ├── data_manifest.json       # gated upstream inputs
│   ├── pipeline.json            # workflow contract
│   └── results_manifest.tsv     # size and SHA-256 for frozen outputs
├── data/external/README.md
├── docs/
│   ├── DATA_DICTIONARY.md
│   ├── DATA_PROVENANCE.md
│   └── REPRODUCIBILITY.md
├── environment.yml
├── notebooks/pancancer_mutational_landscape.ipynb
├── pyproject.toml
├── renv.lock
├── requirements-lock.txt
├── results/
│   ├── figures/                 # 9 main and 7 supplementary PNG files
│   └── tables/                  # 43 frozen analytical result tables
├── src/                         # acquisition, curation, models and figure stages
└── tests/                       # synthetic analytical contract tests
```

## Reproducibility contracts

- Cross-study specimen identity requires a recognised global identifier or an approved
  study-pair overlap mapping; unresolved generic identifier collisions remain
  study-qualified.
- One representative eligible tissue tumour is selected per patient key, with primary
  tumour preceding recurrence, metastasis and unspecified tissue.
- A targeted-panel gene is callable only when documented panel membership includes it;
  a positive call outside the documented scope is audited rather than silently counted.
- Multiple-testing correction uses Benjamini–Hochberg within each stated testing family.
- Primary pair estimates use the unshifted Mantel–Haenszel common odds ratio and score
  test across detailed histology, study and exact assay or panel strata, without
  conditioning on mutation burden. The leave-two-out sensitivity subtracts both tested
  gene indicators before constructing background-burden quintiles. The original
  total-burden specification is retained only as a diagnostic.
- Survival screening requires positive follow-up, at least 80 patients, 20 deaths and
  20 patients in each four-state genotype group. Reported zero-time records enter only
  the explicit half-day sensitivity analysis.
- cBioPortal acquisition records response counts and SHA-256 hashes per request and
  fails if a paginated response reaches its configured page boundary.

See [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the commands and expected scope,
[DATA_PROVENANCE.md](docs/DATA_PROVENANCE.md) for source versions and
[DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md) for table definitions.

## Licence and citation

The code is released under the [MIT Licence](LICENSE). Upstream data remain subject to
their providers' terms. Citation metadata are provided in [CITATION.cff](CITATION.cff).
