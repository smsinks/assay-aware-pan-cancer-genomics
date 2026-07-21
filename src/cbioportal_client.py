"""Cached client for the cBioPortal public REST API.

Why this exists: mutation/CNA pulls for the full study compendium are large and slow,
and the public portal occasionally returns transient 5xx / connection errors. This
wrapper adds (1) retry with backoff, (2) transparent on-disk caching keyed by the
request, and (3) helpers for the few endpoints the pipeline needs. It uses the plain
REST API via ``requests`` rather than the Swagger/bravado client, which is brittle
against the portal's evolving spec.

Public studies need no authentication.
"""
from __future__ import annotations

import hashlib
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from config import API_BASE, API_PAGE_SIZE, CACHE, MAX_RETRIES, REQUEST_TIMEOUT

# LibreSSL on macOS system Python triggers a noisy urllib3 warning; silence just that.
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

REQUEST_METADATA = CACHE.parent / "request_metadata"


def _canonical_json_bytes(value: Any) -> bytes:
    """Serialise a response deterministically for hashing and cached storage."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _write_request_metadata(
    *,
    method: str,
    path: str,
    url: str,
    params: Any,
    body: Any,
    cache_path: Path,
    response_bytes: bytes,
    data: Any,
    source: str,
    response_headers: dict[str, str] | None = None,
) -> None:
    """Write one verifiable metadata record per deterministic API request."""
    REQUEST_METADATA.mkdir(parents=True, exist_ok=True)
    retrieved = datetime.fromtimestamp(cache_path.stat().st_mtime, tz=timezone.utc).isoformat()
    request_payload = [method, url, params, body]
    request_bytes = json.dumps(
        request_payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    if isinstance(data, list):
        row_count, response_type = len(data), "list"
    elif isinstance(data, dict):
        row_count, response_type = 1, "object"
    elif data is None:
        row_count, response_type = 0, "null"
    else:
        row_count, response_type = 1, type(data).__name__
    request_params = params if isinstance(params, dict) else {}
    page_size = request_params.get("pageSize")
    page_number = request_params.get("pageNumber", 0) if page_size is not None else None
    pagination_boundary = bool(
        method.upper() == "GET"
        and isinstance(data, list)
        and page_size is not None
        and len(data) >= int(page_size)
    )
    headers = {str(key).lower(): str(value) for key, value in (response_headers or {}).items()}
    record = {
        "requestId": cache_path.stem,
        "method": method,
        "endpoint": path,
        "url": url,
        "parameters": params,
        "body": body,
        "retrievedAtUtc": retrieved,
        "responseRecordCount": row_count,
        "responseType": response_type,
        "responseBytes": len(response_bytes),
        "requestSha256": hashlib.sha256(request_bytes).hexdigest(),
        "responseSha256": hashlib.sha256(response_bytes).hexdigest(),
        "responseSource": source,
        "cacheFilename": cache_path.name,
        "paginationPageSize": page_size,
        "paginationPageNumber": page_number,
        "paginationBoundaryReached": pagination_boundary,
        "responseTotalCountHeader": headers.get("x-total-count") or headers.get("total-count"),
    }
    (REQUEST_METADATA / f"{cache_path.stem}.json").write_text(
        json.dumps(record, sort_keys=True, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _assert_complete_page(method: str, params: Any, data: Any) -> None:
    """Prevent a page-boundary response from being accepted as complete."""
    if method.upper() != "GET" or not isinstance(data, list) or not isinstance(params, dict):
        return
    page_size = params.get("pageSize")
    if page_size is not None and len(data) >= int(page_size):
        raise RuntimeError(
            f"GET response returned {len(data):,} records, reaching pageSize={int(page_size):,}; "
            "explicit pagination is required"
        )


def _cache_key(method: str, url: str, params: Any, body: Any) -> Path:
    """Deterministic cache path for a request."""
    raw = json.dumps([method, url, params, body], sort_keys=True, default=str)
    digest = hashlib.sha256(raw.encode()).hexdigest()[:24]
    return CACHE / f"{digest}.json"


def _request(method: str, path: str, *, params=None, body=None, use_cache=True) -> Any:
    """Issue one API call with caching + retry. Returns parsed JSON.

    Caching is keyed on (method, url, params, body); a cache hit avoids the network
    entirely, which makes the whole pipeline cheap to re-run.
    """
    url = f"{API_BASE}{path}"
    key = _cache_key(method, url, params, body)
    if use_cache and key.exists():
        response_bytes = key.read_bytes()
        data = json.loads(response_bytes)
        _assert_complete_page(method, params, data)
        _write_request_metadata(
            method=method,
            path=path,
            url=url,
            params=params,
            body=body,
            cache_path=key,
            response_bytes=response_bytes,
            data=data,
            source="cache",
        )
        return data

    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.request(
                method, url, params=params, json=body, timeout=REQUEST_TIMEOUT,
                headers={"Accept": "application/json"},
            )
            if resp.status_code == 200:
                data = resp.json()
                _assert_complete_page(method, params, data)
                if use_cache:
                    response_bytes = _canonical_json_bytes(data)
                    key.write_bytes(response_bytes)
                    _write_request_metadata(
                        method=method,
                        path=path,
                        url=url,
                        params=params,
                        body=body,
                        cache_path=key,
                        response_bytes=response_bytes,
                        data=data,
                        source="network",
                        response_headers=dict(resp.headers),
                    )
                return data
            # 4xx are not retryable except 429 (rate limit)
            if 400 <= resp.status_code < 500 and resp.status_code != 429:
                raise RuntimeError(f"{method} {url} -> HTTP {resp.status_code}: {resp.text[:300]}")
            last_err = RuntimeError(f"HTTP {resp.status_code}")
        except (requests.RequestException, ValueError) as exc:  # network / JSON errors
            last_err = exc
        # exponential backoff before next attempt
        time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"{method} {url} failed after {MAX_RETRIES} attempts: {last_err}")


# --- Endpoint helpers ------------------------------------------------------

def get_all_studies() -> list[dict]:
    """All public studies (DETAILED projection)."""
    return _request("GET", "/studies", params={"projection": "DETAILED", "pageSize": API_PAGE_SIZE})


def get_sample_ids(study_id: str) -> list[str]:
    """Sample IDs for a study (true sample count, since allSampleCount is unreliable)."""
    data = _request("GET", f"/studies/{study_id}/samples",
                    params={"projection": "ID", "pageSize": API_PAGE_SIZE})
    return [s["sampleId"] for s in data]


def get_molecular_profiles(study_id: str | None = None) -> list[dict]:
    """Molecular profiles for one study, or for the whole portal if study_id is None."""
    path = f"/studies/{study_id}/molecular-profiles" if study_id else "/molecular-profiles"
    return _request("GET", path, params={"projection": "SUMMARY", "pageSize": API_PAGE_SIZE})


def get_mutations(molecular_profile_id: str, sample_list_id: str, entrez_ids: list[int]) -> list[dict]:
    """Mutations in the given genes for a molecular profile + sample list.

    Uses the fetch endpoint (POST) so we can restrict to our gene panel in one call.
    ``sample_list_id`` is typically ``f"{study_id}_all"``.
    """
    body = {"sampleListId": sample_list_id, "entrezGeneIds": entrez_ids}
    return _request("POST", f"/molecular-profiles/{molecular_profile_id}/mutations/fetch",
                    params={"projection": "DETAILED"}, body=body)


def get_cna(molecular_profile_id: str, sample_list_id: str, entrez_ids: list[int]) -> list[dict]:
    """Discrete copy-number alterations (GISTIC -2..2) for the gene panel."""
    body = {"sampleListId": sample_list_id, "entrezGeneIds": entrez_ids}
    return _request("POST", f"/molecular-profiles/{molecular_profile_id}/discrete-copy-number/fetch",
                    params={"discreteCopyNumberEventType": "ALL", "projection": "DETAILED"}, body=body)


def map_symbols_to_entrez(hugo_symbols: list[str]) -> list[dict]:
    """Resolve Hugo gene symbols to Entrez IDs via the portal gene service.

    Returns dicts with hugoGeneSymbol + entrezGeneId. Symbols the portal does not
    recognise are simply absent from the result (caller should reconcile).
    """
    return _request("POST", "/genes/fetch",
                    params={"geneIdType": "HUGO_GENE_SYMBOL", "projection": "SUMMARY"},
                    body=hugo_symbols)


def get_sample_clinical(study_id: str) -> list[dict]:
    """Sample-level clinical data — used to recover per-sample cancer type (OncoTree).

    Aggregator cohorts (MSK-IMPACT, GENIE) carry the cancer type on each SAMPLE, not the
    study, so study-level cancerTypeId is 'mixed' for them. CANCER_TYPE / ONCOTREE_CODE /
    CANCER_TYPE_DETAILED live here.
    """
    return _request("GET", f"/studies/{study_id}/clinical-data",
                    params={"clinicalDataType": "SAMPLE", "projection": "SUMMARY",
                            "pageSize": API_PAGE_SIZE})


def get_gene_panel_data(molecular_profile_id: str, sample_list_id: str) -> list[dict]:
    """Per-sample gene-panel assignment for a molecular profile.

    Each row gives the ``genePanelId`` a sample was profiled with; a null/absent panel
    means whole-exome/genome (all genes interrogated). This is what makes NaN-aware,
    assay-aware frequencies possible: a gene off a sample's panel is *not assayed*
    (NaN), not wild-type.
    """
    body = {"sampleListId": sample_list_id}
    return _request("POST", f"/molecular-profiles/{molecular_profile_id}/gene-panel-data/fetch",
                    body=body)


def get_gene_panel(panel_id: str) -> dict:
    """A gene panel's gene list (with hugoGeneSymbol + entrezGeneId)."""
    return _request("GET", f"/gene-panels/{panel_id}")


def get_clinical_data(study_id: str) -> list[dict]:
    """Patient-level clinical data (for survival: OS/DFS status + months, age, sex)."""
    return _request("GET", f"/studies/{study_id}/clinical-data",
                    params={"clinicalDataType": "PATIENT", "projection": "DETAILED",
                            "pageSize": API_PAGE_SIZE})


if __name__ == "__main__":
    # Smoke test: prove the client + cache work end to end.
    studies = get_all_studies()
    print(f"Fetched {len(studies)} studies (cached at {CACHE}).")
    n = len(get_sample_ids("brca_tcga_pan_can_atlas_2018"))
    print(f"brca_tcga_pan_can_atlas_2018 has {n} samples.")
