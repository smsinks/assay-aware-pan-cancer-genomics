"""Verify the size and SHA-256 digest of every redistributed frozen output."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "config" / "results_manifest.tsv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest() -> dict[str, object]:
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Missing integrity manifest: {MANIFEST.relative_to(ROOT)}")
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    failures: list[str] = []
    seen: set[str] = set()
    for row in rows:
        relative = row["path"]
        if relative in seen:
            failures.append(f"duplicate manifest path: {relative}")
            continue
        seen.add(relative)
        path = (ROOT / relative).resolve()
        try:
            path.relative_to(ROOT.resolve())
        except ValueError:
            failures.append(f"path escapes repository: {relative}")
            continue
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        if path.stat().st_size != int(row["bytes"]):
            failures.append(f"size mismatch: {relative}")
            continue
        if sha256(path) != row["sha256"]:
            failures.append(f"SHA-256 mismatch: {relative}")
    return {"ok": not failures, "entries": len(rows), "failures": failures}


def main() -> None:
    report = verify_manifest()
    if not report["ok"]:
        raise RuntimeError("Frozen-output integrity check failed:\n  - " + "\n  - ".join(report["failures"]))
    print(f"Verified {report['entries']} frozen outputs against SHA-256 manifest")


if __name__ == "__main__":
    main()
