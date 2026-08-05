# -*- coding: utf-8 -*-
"""
starrydata3_client.py — push a NCMRD record into a Starrydata3 server from the
desktop app, over its REST API with an API key. No browser, no session.

    from starrydata3_client import push_record
    res = push_record("http://gpu-box:8300", "sd3_...", record)
    # res -> {"ok": True, "sid": 60005, "curves_indexed": 29, ...}  or {"ok": False, "error": ...}
"""
from typing import Any, Dict


def push_record(url: str, api_key: str, record: Dict[str, Any],
                timeout: float = 120.0) -> Dict[str, Any]:
    """POST a NCMRD record to Starrydata3 /api/v1/records. Never raises —
    returns a dict with ok/sid/curves_indexed or ok=False + error."""
    url = (url or "").strip().rstrip("/")
    api_key = (api_key or "").strip()
    if not url:
        return {"ok": False, "error": "No Starrydata3 URL set"}
    if not api_key:
        return {"ok": False, "error": "No Starrydata3 API key set"}
    try:
        import httpx
    except ImportError:
        return {"ok": False, "error": "httpx not installed"}
    try:
        r = httpx.post(url + "/api/v1/records", json=record,
                       headers={"X-API-Key": api_key}, timeout=timeout)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    try:
        body = r.json()
    except Exception:
        body = {"text": r.text[:300]}
    if r.status_code >= 400:
        detail = body.get("detail", body) if isinstance(body, dict) else body
        return {"ok": False, "error": f"HTTP {r.status_code}: {detail}"}
    return {"ok": True, "sid": body.get("sid"), "doi": body.get("doi"),
            "curves_indexed": body.get("curves_indexed"),
            "ncmrd_curves": body.get("ncmrd_curves"),
            "unit_normalized_curves": body.get("unit_normalized_curves"),
            "non_ncmrd_properties": body.get("non_ncmrd_properties") or [],
            "n_violations": body.get("n_violations"), "url": url}


def push_provenance(url: str, api_key: str, doi: str,
                    figures: "list[Dict[str, Any]]",
                    timeout: float = 120.0) -> Dict[str, Any]:
    """POST digitization provenance (figure crop + calibration + extractor
    versions per figure) for an already-uploaded record. Never raises.

    figures: [{"label", "page", "png_b64", "calibration", "extractor",
               "curator"}] — see Starrydata3 /api/v1/provenance."""
    url = (url or "").strip().rstrip("/")
    if not url or not api_key:
        return {"ok": False, "error": "No Starrydata3 URL/API key set"}
    try:
        import httpx
        r = httpx.post(url + "/api/v1/provenance",
                       json={"doi": doi, "figures": figures},
                       headers={"X-API-Key": (api_key or "").strip()},
                       timeout=timeout)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    try:
        body = r.json()
    except Exception:
        body = {"text": r.text[:300]}
    if r.status_code >= 400:
        detail = body.get("detail", body) if isinstance(body, dict) else body
        return {"ok": False, "error": f"HTTP {r.status_code}: {detail}"}
    return {"ok": True, "figures_stored": body.get("figures_stored")}
