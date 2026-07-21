"""Aggregate per-request cBioPortal metadata into verifiable provenance manifests."""
from __future__ import annotations

import hashlib
import json

import pandas as pd

from config import RAW, RESULTS


METADATA = RAW / "request_metadata"
OUT_CSV = RESULTS / "cbioportal_request_manifest.csv"
OUT_JSON = RESULTS / "cbioportal_request_manifest.json"


def main() -> None:
    paths = sorted(METADATA.glob("*.json"))
    if not paths:
        raise FileNotFoundError(
            "No request metadata were found. Run the cBioPortal acquisition stages first."
        )
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    identifiers = [record["requestId"] for record in records]
    if len(identifiers) != len(set(identifiers)):
        raise AssertionError("Request identifiers are not unique")
    for record in records:
        cache = RAW / "cache" / record["cacheFilename"]
        if not cache.is_file():
            raise FileNotFoundError(cache)
        response_bytes = cache.read_bytes()
        if hashlib.sha256(response_bytes).hexdigest() != record["responseSha256"]:
            raise AssertionError(f"Response hash mismatch for {record['requestId']}")
        response = json.loads(response_bytes)
        observed_count = len(response) if isinstance(response, list) else int(response is not None)
        if observed_count != int(record["responseRecordCount"]):
            raise AssertionError(f"Response-count mismatch for {record['requestId']}")
        if bool(record.get("paginationBoundaryReached", False)):
            raise AssertionError(f"Potentially truncated response for {record['requestId']}")
    frame = pd.DataFrame(records).sort_values(["endpoint", "method", "requestId"])
    for column in ("parameters", "body"):
        frame[column] = frame[column].map(
            lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"))
        )
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(json.dumps(records, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_CSV} with {len(frame):,} verified requests")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
