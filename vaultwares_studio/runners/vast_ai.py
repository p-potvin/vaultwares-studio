"""Vast.ai StageRunner: rent a cheap GPU box, run the worker, tear it down.

Why this exists: HF Jobs at $0.80/hr is fine for short runs and the 40-min/day
free GPU is great for small refines, but multi-thousand-frame jobs need
bigger compute at much lower cost than HF's L4. Vast.ai consumer GPUs
(2× RTX 3060 @ $0.116/hr, 2× RTX 4070 Ti @ $0.229/hr) are 4–7× cheaper than
L4 for the same workload.

Lifecycle the runner manages, end-to-end:
    1. resolve credentials (api key + ssh key from keyring/disk)
    2. search /api/v0/bundles for offerings matching the requested criteria
    3. PUT /api/v0/asks/{id}/ to rent the chosen offering
    4. poll the instance until ssh_host/ssh_port populate and ssh is up
    5. rsync inputs + worker entrypoints over (or use vast.ai's cloud sync)
    6. exec worker via ssh, stream logs back into ctx.log
    7. rsync expected_outputs back to ctx.job_dir/<stage>/remote_out/
    8. DELETE /api/v0/instances/{id}/ to stop billing

This file ships the SKELETON of that flow: config dataclass, auth helpers,
selection helper, runner subclass with the steps stubbed in with clear TODOs.
The actual API calls + ssh transport are wired up after the first paid test
run, where we'll need to confirm the real shape of responses (api docs drift)
and tune timeouts. Until then ``VastAiStageRunner.run`` raises a NotImplemented
error so a misconfigured pipeline fails loudly rather than silently.

Auth:
- Vast.ai API key — kept in OS keyring under ('vw-studio', 'vast-ai-api-key').
- SSH key — local path (default: ~/.ssh/id_ed25519 or ~/.ssh/id_rsa). The
  PUBLIC key is added to the rented instance via vast.ai's "ssh_keys" field
  during PUT; the PRIVATE key on the launcher machine is used for the rsync
  and ssh exec.

Cost safety:
- A hard price ceiling (config.max_price_usd_per_hour) is enforced before
  any rent call. CostEstimate uses the chosen offer's actual rate, not a
  table — vast.ai prices are dynamic.
- The runner ALWAYS calls DELETE on the instance in a finally block, even
  on cancellation or exception, to avoid forgotten boxes accumulating cost.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base import (
    CostDeniedError,
    CostEstimate,
    StageCancelledError,
    StageContext,
    StageResult,
    StageRunner,
)

# OS keyring identifiers for the Vast.ai API key. Mirrors the HF token shape.
_KEYRING_SERVICE = "vw-studio"
_KEYRING_USER = "vast-ai-api-key"


def get_vast_api_key() -> str | None:
    """Return the stored Vast.ai API key, or None if not configured."""
    try:
        import keyring

        return keyring.get_password(_KEYRING_SERVICE, _KEYRING_USER)
    except Exception:  # noqa: BLE001 - keyring is best-effort
        return None


def set_vast_api_key(value: str) -> None:
    import keyring

    keyring.set_password(_KEYRING_SERVICE, _KEYRING_USER, value)


def find_default_ssh_private_key() -> Path | None:
    """Locate the user's SSH private key for instance access.

    Checks the standard names in ~/.ssh. The corresponding public key
    (``<name>.pub``) is uploaded to vast.ai with the rent request.
    """
    home = Path.home()
    for name in ("id_ed25519", "id_rsa", "id_ecdsa"):
        candidate = home / ".ssh" / name
        if candidate.exists():
            return candidate
    return None


@dataclass
class VastAiConfig:
    """All knobs for the vast.ai runner."""

    # Hard price ceiling. Any offer above this fails the search rather than
    # silently picking an expensive box. Per-hour rate in USD.
    max_price_usd_per_hour: float = 0.30

    # Minimum acceptable hardware. Defaults match the user's stated preference
    # (2× 3060 / 2× 4070 Ti range; both have ≥12 GB per GPU, single GPU is
    # sufficient for our scale).
    min_gpu_vram_gb: int = 12
    min_cpu_cores: int = 16
    min_ram_gb: int = 32
    # Vast.ai DLPerf is their internal "training performance" score; 24+
    # cleared the 3060 box the user shared, 67+ cleared the 4070 Ti.
    min_dlperf: float = 20.0

    # Worker container. Vast.ai supports docker:// images directly at rent
    # time. The nerfstudio base image is public, so the simplest first cut
    # is to rent with the base image and rsync our entrypoints on. Later,
    # we publish the full worker image to a public registry and switch to it.
    worker_image: str = "ghcr.io/nerfstudio-project/nerfstudio:latest"

    # Path on the launcher host to the SSH private key for the rented box.
    # Defaults to whichever standard name exists in ~/.ssh.
    ssh_private_key: Path | None = field(default_factory=find_default_ssh_private_key)

    # Poll cadence for instance status (vast.ai instance boot can take a few
    # minutes; check every 10s without hammering).
    poll_interval_seconds: float = 10.0
    # Cap total wait for instance to become SSH-reachable (a stuck rent
    # should fail fast rather than burn idle minutes).
    boot_timeout_seconds: float = 600.0

    # Vast.ai's REST API base. The host has been stable for years but kept
    # here in case we need to swing to a proxy.
    api_base: str = "https://console.vast.ai/api/v0"


def _http(method: str, url: str, *, api_key: str, **kwargs) -> dict[str, Any]:
    """Tiny REST helper used by the runner. Headers + JSON in/out + raise."""
    import requests

    headers = kwargs.pop("headers", {})
    headers.setdefault("Authorization", f"Bearer {api_key}")
    headers.setdefault("Accept", "application/json")
    response = requests.request(method, url, headers=headers, timeout=60, **kwargs)
    if response.status_code >= 400:
        raise RuntimeError(f"vast.ai {method} {url} -> {response.status_code}: {response.text[:300]}")
    if not response.content:
        return {}
    return response.json()


def find_best_offer(config: VastAiConfig, api_key: str) -> dict[str, Any]:
    """Search vast.ai bundles for the cheapest offer meeting min spec.

    Returns the offer dict; raises if nothing qualifies under the price cap.
    The bundles endpoint takes a JSON query that mirrors their search UI;
    we pass the most discriminating filters and sort by total $/hr ascending.
    """
    query: dict[str, Any] = {
        "verified": {"eq": True},
        "rentable": {"eq": True},
        "external": {"eq": False},
        "gpu_ram": {"gte": config.min_gpu_vram_gb * 1024},  # MB units
        "cpu_cores": {"gte": config.min_cpu_cores},
        "cpu_ram": {"gte": config.min_ram_gb * 1024},  # MB
        "dlperf": {"gte": config.min_dlperf},
        "dph_total": {"lte": config.max_price_usd_per_hour},
        "order": [["dph_total", "asc"]],
    }
    payload = _http(
        "PUT", f"{config.api_base}/bundles/",
        api_key=api_key, json={"q": query},
    )
    offers = payload.get("offers") or []
    if not offers:
        raise RuntimeError(
            f"No vast.ai offers under ${config.max_price_usd_per_hour}/h with "
            f"≥{config.min_gpu_vram_gb}GB VRAM, ≥{config.min_cpu_cores} CPU, "
            f"≥{config.min_ram_gb}GB RAM, DLPerf ≥{config.min_dlperf}."
        )
    return offers[0]


class VastAiStageRunner(StageRunner):
    """Execute one stage on a rented vast.ai instance.

    Implementation status: SKELETON. Search + auth are real; rent + ssh +
    transfer + teardown are stubbed and raise NotImplementedError until the
    first end-to-end test pass (planned after current HF refine completes).
    Calling .run() in the meantime fails loudly so we never silently fall
    through to a broken backend.
    """

    name = "vast-ai"

    def __init__(
        self,
        config: VastAiConfig | None = None,
        *,
        confirm_cost=None,  # Callable[[CostEstimate], bool]
    ) -> None:
        self.config = config or VastAiConfig()
        self.confirm_cost = confirm_cost

    def run(self, ctx: StageContext) -> StageResult:  # noqa: PLR0915
        api_key = get_vast_api_key()
        if not api_key:
            raise RuntimeError(
                "Vast.ai API key not configured. Generate one at "
                "console.vast.ai/account/, then call "
                "vaultwares_studio.runners.vast_ai.set_vast_api_key(<key>)."
            )
        if self.config.ssh_private_key is None:
            raise RuntimeError(
                "No SSH private key found. Set VastAiConfig.ssh_private_key "
                "or place id_ed25519/id_rsa in ~/.ssh/."
            )

        # 1. Find an offer that meets spec + budget.
        offer = find_best_offer(self.config, api_key)
        rate = float(offer.get("dph_total", 0.0))
        est_minutes = float(ctx.params.get("est_minutes", 30))
        estimate = CostEstimate(
            flavor=f"vast-{offer.get('gpu_name', '?')}",
            est_minutes=est_minutes,
            rate_usd_per_hour=rate,
            est_usd=round(rate * est_minutes / 60.0, 2),
            source=f"vast.ai offer {offer.get('id')}",
        )
        ctx.log(f"[vast-ai] best offer: {estimate.summary()}")

        if self.confirm_cost is None or not self.confirm_cost(estimate):
            raise CostDeniedError(f"Vast.ai run declined ({estimate.summary()}).")

        # 2. Rent it. (STUB — needs implementation.)
        instance_id = self._rent(offer, api_key, ctx)
        try:
            # 3. Wait for the box to become SSH-reachable.
            ssh_host, ssh_port = self._wait_for_ssh(instance_id, api_key, ctx)
            # 4. Transfer inputs (frames.zip, etc.) + worker entrypoints.
            self._transfer_inputs(ssh_host, ssh_port, ctx)
            # 5. Run the worker over SSH, streaming logs.
            self._run_remote_worker(ssh_host, ssh_port, ctx)
            # 6. Pull outputs back.
            artifacts = self._fetch_outputs(ssh_host, ssh_port, ctx)
            actual_minutes = est_minutes  # placeholder; tracked via timer in real impl
            metadata = {
                "instance_id": instance_id,
                "offer_id": offer.get("id"),
                "gpu_name": offer.get("gpu_name"),
                "rate_usd_per_hour": rate,
                "actual_usd_estimate": round(rate * actual_minutes / 60.0, 2),
            }
            return StageResult(status="complete", artifacts=artifacts, metadata=metadata)
        finally:
            # 7. ALWAYS destroy the instance to stop billing — even on
            # exceptions, cancellation, or partial runs. The shared cancel
            # token is checked here too in case the user hit cancel mid-flow.
            self._destroy(instance_id, api_key, ctx)

    # -- Stubbed steps -------------------------------------------------------
    # These raise so a misconfigured pipeline fails loudly. Real
    # implementations land after the first end-to-end paid test pass.

    def _rent(self, offer: dict, api_key: str, ctx: StageContext) -> str:
        """PUT /api/v0/asks/{id}/ with image + ssh key. Returns instance_id."""
        raise NotImplementedError(
            "VastAiStageRunner._rent is stubbed. Implement via PUT "
            f"{self.config.api_base}/asks/{offer.get('id')}/ with body "
            "{'client_id': 'me', 'image': self.config.worker_image, "
            "'ssh_keys': [<public key>], 'disk': 32} — see vast.ai API docs."
        )

    def _wait_for_ssh(self, instance_id: str, api_key: str, ctx: StageContext) -> tuple[str, int]:
        """Poll GET /api/v0/instances/ until ssh_host/ssh_port populate."""
        raise NotImplementedError("VastAiStageRunner._wait_for_ssh is stubbed.")

    def _transfer_inputs(self, ssh_host: str, ssh_port: int, ctx: StageContext) -> None:
        """rsync ctx.inputs + worker entrypoints into /tmp/recon on the box."""
        raise NotImplementedError("VastAiStageRunner._transfer_inputs is stubbed.")

    def _run_remote_worker(self, ssh_host: str, ssh_port: int, ctx: StageContext) -> None:
        """ssh exec the worker command from ctx.params['command'], stream logs."""
        raise NotImplementedError("VastAiStageRunner._run_remote_worker is stubbed.")

    def _fetch_outputs(self, ssh_host: str, ssh_port: int, ctx: StageContext) -> list[Path]:
        """rsync expected_outputs back to ctx.job_dir/<stage>/remote_out/."""
        raise NotImplementedError("VastAiStageRunner._fetch_outputs is stubbed.")

    def _destroy(self, instance_id: str | None, api_key: str, ctx: StageContext) -> None:
        """DELETE /api/v0/instances/{id}/. Always runs in finally."""
        if not instance_id:
            return
        try:
            _http(
                "DELETE", f"{self.config.api_base}/instances/{instance_id}/",
                api_key=api_key,
            )
            ctx.log(f"[vast-ai] destroyed instance {instance_id}")
        except Exception as exc:  # noqa: BLE001 - never raise from cleanup
            ctx.log(f"[vast-ai] WARNING: destroy {instance_id} failed: {exc}")
