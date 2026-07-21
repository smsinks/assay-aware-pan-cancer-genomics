"""Audit packaged frozen outputs without downloading data or fitting models."""
from __future__ import annotations

from result_summary import main as print_packaged_summary
from verify_results_manifest import verify_manifest


def main() -> None:
    report = verify_manifest()
    if not report["ok"]:
        raise RuntimeError(
            "Frozen-output integrity check failed:\n  - "
            + "\n  - ".join(report["failures"])
        )
    print_packaged_summary()
    print(f"Verified {report['entries']} frozen outputs against SHA-256 manifest")


if __name__ == "__main__":
    main()
