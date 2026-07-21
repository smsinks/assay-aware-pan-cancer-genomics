"""Run the complete raw-data reconstruction after gated inputs are supplied."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
DATA_MANIFEST = ROOT / "config" / "data_manifest.json"

FULL_SEQUENCE = [
    "01_select_studies.py",
    "02_build_gene_panel.py",
    "03_fetch_alterations.py",
    "01b_dedupe_samples.py",
    "01c_sample_cancertype.py",
    "04_define_assay_callability.py",
    "05_fetch_copy_number.py",
    "01d_curate_cohort.py",
    "08_cosmic_cmc_annotate.py",
    "16_curated_landscape.py",
    "17_curated_interactions.py",
    "19_validate_cmh.py",
    "18_depmap_lineage_adjusted.py",
    "20_complete_coverage.py",
    "21_curated_pathways.py",
    "22_curated_survival.py",
    "23_figures1_to5.py",
    "24_figures7_to8.py",
    "29_metadata_sensitivities.py",
    "25_hallmark_supplement.py",
    "27_coverage_supplement.py",
    "build_cbioportal_request_manifest.py",
    "build_cross_layer_evidence.py",
    "export_repository_results.py",
    "build_results_manifest.py",
]


def required_file_inputs() -> list[tuple[Path, str | None]]:
    manifest = json.loads(DATA_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported data-manifest schema")
    inputs: list[tuple[Path, str | None]] = []
    for resource in manifest["resources"]:
        if (
            resource.get("kind") == "file"
            and resource.get("required_for_full_reconstruction", False)
        ):
            path = Path(resource["path"])
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"External manifest path must be repository-relative: {path}")
            inputs.append((ROOT / path, resource.get("sha256")))
    return inputs


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preflight() -> list[str]:
    problems: list[str] = []
    for path, expected_digest in required_file_inputs():
        if not path.is_file():
            problems.append(f"missing external input: {path.relative_to(ROOT)}")
        elif expected_digest and sha256(path) != expected_digest:
            problems.append(f"external-input SHA-256 mismatch: {path.relative_to(ROOT)}")
    for script in FULL_SEQUENCE:
        if not (SRC / script).is_file():
            problems.append(f"missing pipeline stage: src/{script}")
    if shutil.which("Rscript") is None:
        problems.append("Rscript is unavailable")
    else:
        check = subprocess.run(
            [
                "Rscript",
                "-e",
                "quit(status=ifelse(requireNamespace('survival', quietly=TRUE), 0, 1))",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if check.returncode != 0:
            problems.append("R package 'survival' is unavailable")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full reconstruction from cBioPortal and documented external resources"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="check gated inputs and print the stage sequence without running it",
    )
    args = parser.parse_args()
    problems = preflight()
    if problems:
        detail = "\n  - ".join(problems)
        raise SystemExit(
            "Full reconstruction preflight failed. The frozen-output audit remains "
            f"available with 'python src/run_pipeline.py'.\n  - {detail}"
        )
    if args.dry_run:
        print("Full reconstruction preflight passed. Planned stages:")
        for script in FULL_SEQUENCE:
            print(f"  - {script}")
        return
    for script in FULL_SEQUENCE:
        print(f"\n[{script}]", flush=True)
        subprocess.run([sys.executable, str(SRC / script)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
