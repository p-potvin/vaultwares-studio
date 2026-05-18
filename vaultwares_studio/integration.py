from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request

from .pipeline import JobManifest


@dataclass(frozen=True)
class VaultFlowsConnectionSettings:
    api_base: str
    app_url: str
    bearer_token: str = ""
    api_key: str = ""


def _trim(url: str) -> str:
    return url.strip().rstrip("/")


def _headers(settings: VaultFlowsConnectionSettings) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if settings.bearer_token.strip():
        headers["Authorization"] = f"Bearer {settings.bearer_token.strip()}"
    if settings.api_key.strip():
        headers["X-Api-Key"] = settings.api_key.strip()
    return headers


def build_vaultflows_workflow(manifest: JobManifest) -> dict[str, Any]:
    source_name = Path(manifest.source_video).stem or manifest.job_id
    return {
        "id": manifest.job_id,
        "name": f"Digital Twin Studio - {source_name}",
        "category": "Digital Twin",
        "description": "Guided local-first digital twin workflow exported from vaultwares-studio for Vault Flows / Vaultwares Pipelines.",
        "pin": True,
        "favorite": True,
        "lastRun": manifest.updated_at,
        "steps": [
            {
                "id": stage.key,
                "title": stage.title,
                "description": stage.description,
                "state": stage.state,
                "message": stage.message,
                "artifacts": [artifact.to_dict() for artifact in stage.artifacts],
                "metadata": stage.metadata,
            }
            for stage in manifest.stages
        ],
    }


def export_vaultflows_workflow(manifest: JobManifest, output_path: Path | str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_vaultflows_workflow(manifest), indent=2), encoding="utf-8")
    return path


def test_vaultwares_api(settings: VaultFlowsConnectionSettings) -> dict[str, Any]:
    api_base = _trim(settings.api_base)
    openapi_url = f"{api_base}/openapi.json"
    config_url = f"{api_base}/config"

    result: dict[str, Any] = {
        "apiBase": api_base,
        "openapiReachable": False,
        "authStatus": "unknown",
    }

    try:
        with request.urlopen(openapi_url, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
            result["openapiReachable"] = True
            result["openapiTitle"] = payload.get("info", {}).get("title", "")
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"OpenAPI probe failed: {exc}"
        return result

    config_request = request.Request(config_url, headers=_headers(settings))
    try:
        with request.urlopen(config_request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
            result["authStatus"] = "ok"
            result["configKeys"] = sorted(payload.keys()) if isinstance(payload, dict) else []
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        result["authStatus"] = f"http-{exc.code}"
        result["authDetail"] = detail
    except Exception as exc:  # noqa: BLE001
        result["authStatus"] = "error"
        result["authDetail"] = str(exc)

    return result


def push_workflow_to_vaultwares(settings: VaultFlowsConnectionSettings, manifest: JobManifest) -> dict[str, Any]:
    api_base = _trim(settings.api_base)
    payload = build_vaultflows_workflow(manifest)
    body = json.dumps(payload).encode("utf-8")
    headers = _headers(settings)

    def send(method: str, url: str) -> dict[str, Any]:
        req = request.Request(url, data=body, headers=headers, method=method)
        with request.urlopen(req, timeout=10) as response:
            response_body = response.read().decode("utf-8")
            return json.loads(response_body) if response_body else {}

    try:
        data = send("POST", f"{api_base}/workflows")
        return {"mode": "created", "data": data}
    except error.HTTPError as exc:
        if exc.code not in (400, 409, 422):
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Workflow create failed ({exc.code}): {detail}") from exc

    try:
        update_payload = {
            "name": payload["name"],
            "category": payload["category"],
            "description": payload["description"],
            "steps": payload["steps"],
            "pin": payload["pin"],
            "favorite": payload["favorite"],
            "lastRun": payload["lastRun"],
        }
        update_body = json.dumps(update_payload).encode("utf-8")
        req = request.Request(
            f"{api_base}/workflows/{payload['id']}",
            data=update_body,
            headers=headers,
            method="PUT",
        )
        with request.urlopen(req, timeout=10) as response:
            response_body = response.read().decode("utf-8")
            data = json.loads(response_body) if response_body else {}
            return {"mode": "updated", "data": data}
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Workflow update failed ({exc.code}): {detail}") from exc
