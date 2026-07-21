"""Build the integrity manifest for redistributed frozen tables and figure images."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
MANIFEST = ROOT / "config" / "results_manifest.tsv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_files() -> list[Path]:
    tables = sorted((RESULTS / "tables").glob("*.csv"))
    figures = sorted((RESULTS / "figures").glob("*.png"))
    figures += sorted((RESULTS / "figures" / "supplementary").glob("*.png"))
    return tables + figures


def main() -> None:
    files = manifest_files()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            fieldnames=["path", "category", "bytes", "sha256"],
            lineterminator="\n",
        )
        writer.writeheader()
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            writer.writerow(
                {
                    "path": relative,
                    "category": "table" if path.suffix == ".csv" else "figure",
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    print(f"Wrote {len(files)} entries to {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
