# Reproducibility guide

## Route A: frozen-output audit

This is the default verification route. It is deterministic, uses only redistributed files
and does not contact external services or fit statistical models.

```bash
make audit
```

The command performs four checks:

1. all 41 prepared tables, nine main figures and seven supplementary figures are
   present;
2. headline cohort, assay-coverage, recurrence, three-specification gene-pair,
   functional, interactome-null and extended survival summaries can be reconstructed
   from the prepared tables;
3. every redistributed result file matches its recorded byte size and SHA-256 digest;
4. synthetic tests enforce the specimen, patient-selection, assay-callability,
   Benjamini–Hochberg, Mantel–Haenszel and survival-time contracts.

The notebook performs prepared-table audits and displays the deposited figures. It does
not refit models or reconstruct patient-level inputs. Execute it without modifying the
tracked file:

```bash
make notebook
```

## Route B: full raw-data reconstruction

The complete route rebuilds the cohort and all models. It is intentionally gated in a
fresh clone because several upstream resources cannot be redistributed. Supply the
files listed in `config/data_manifest.json`, populate any checksum fields required for
the authorised copies, then check the environment and file gate:

```bash
python src/run_all.py --dry-run
```

Only after that check passes should the complete route be started:

```bash
make full
```

The full route retrieves cBioPortal molecular and clinical records and OncoKB
annotations. It consumes licensed COSMIC exports, DepMap/PRISM files, reviewed taxonomy
and cross-study identity mappings, a pathway supplement and the directed interactome.
Each stage writes repository-relative intermediate files under `data/processed/` and
final outputs under `results/`. The cBioPortal client records the endpoint, method,
parameters or body, retrieval time, response count, request and response SHA-256 hashes,
cache filename and pagination state for every request. The full workflow verifies those
records and writes combined `cbioportal_request_manifest.csv` and
`cbioportal_request_manifest.json` files; it fails when a response reaches the declared
pagination boundary. A successful frozen-output audit demonstrates the
integrity and internal consistency of the deposited analytical release; it is distinct
from a complete reconstruction from the gated starting inputs.

## Environment records

- `environment.yml` is the cross-language Conda specification.
- `requirements-lock.txt` pins the Python packages used by the notebook and audit.
- `pyproject.toml` records project metadata and direct Python dependencies.
- `renv.lock` records R 4.5.2 and `survival` 3.8-3.
- `R/sessionInfo.txt` records the R session used for the survival implementation audit.

## Integrity manifests

`config/results_manifest.tsv` records the relative path, category, byte size and SHA-256
digest of every redistributed CSV table and PNG figure. Rebuild it only when the frozen
outputs are intentionally replaced:

```bash
make manifest
```

`config/data_manifest.json` is different: it describes upstream resources required for
full reconstruction and records why they are absent from the repository. Null SHA-256
fields should be populated by the user from their authorised downloads before a long
reconstruction when provider checksums are available.

## Inputs required for full reconstruction

A public clone verifies the deposited result state and exercises the core analytical
contracts. Patient-level cohort reconstruction and model refitting additionally require
the gated inputs listed in `config/data_manifest.json`. The repository does not
redistribute cBioPortal patient-level cache files, COSMIC exports, DepMap/PRISM matrices
or the directed interactome.
