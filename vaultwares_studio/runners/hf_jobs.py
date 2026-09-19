"""HfJobsStageRunner: execute a pipeline stage as a Hugging Face Job.

Contract (same StageRunner seam as local execution):
  1. Estimate cost and require explicit user confirmation BEFORE any network
     call — no unattended paid runs, ever.
  2. Upload ctx.inputs to a private HF dataset repo under
     ``jobs/<job-id>/<stage>/in/``.
  3. Launch a Job whose container downloads ``in/``, runs the stage command
     with VW_IN / VW_OUT env vars, and uploads everything it wrote to
     ``out/``.
  4. Poll job status (>= 10 s interval per company rate-limiting protocol),
     honoring cancellation.
  5. Download ``out/`` files into the locations listed in
     ctx.expected_outputs and report actual-cost metadata for the spend
     ledger.

Requires a Hugging Face PRO account (Jobs are PRO-gated). The token is read
from the OS keyring (service ``vaultwares-studio``), falling back to the
HF_TOKEN environment variable.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

# Matches nerfstudio-style "(42.3%)" progress fragments in remote log lines.
_PERCENT_RE = re.compile(r"\((\d{1,3}(?:\.\d+)?)%\)")

from .. import telemetry
from .base import (
    CostDeniedError,
    CostEstimate,
    StageCancelledError,
    StageContext,
    StageResult,
    StageRunner,
    estimate_cost,
)

ROOT = Path(__file__).resolve().parent.parent.parent
REMOTE_CONFIG_PATH = ROOT / "data" / "remote_compute.json"
KEYRING_SERVICE = "vaultwares-studio"
KEYRING_ACCOUNT = "huggingface"
MIN_POLL_INTERVAL_SECONDS = 10.0

TERMINAL_SUCCESS = {"COMPLETED"}
TERMINAL_FAILURE = {"ERROR", "CANCELED", "CANCELLED", "DELETED"}

# Runs inside the job container (via VW_BOOTSTRAP env): pull inputs, run the
# stage command, push outputs. Kept dependency-free beyond huggingface_hub.
BOOTSTRAP_SOURCE = """
import json, os, subprocess, sys
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

cfg = json.loads(os.environ["VW_STAGE_CONFIG"])
work = Path("/tmp/vw_stage")
in_dir = work / "in"
out_dir = work / "out"
in_dir.mkdir(parents=True, exist_ok=True)
out_dir.mkdir(parents=True, exist_ok=True)

patterns = [cfg["prefix"] + "/in/*"] + cfg.get("extra_inputs", [])
snapshot_download(
    repo_id=cfg["repo"],
    repo_type="dataset",
    allow_patterns=patterns,
    local_dir=str(work / "dl"),
)
staged = work / "dl" / cfg["prefix"] / "in"
if staged.exists():
    for item in staged.iterdir():
        item.replace(in_dir / item.name)
for extra in cfg.get("extra_inputs", []):
    source = work / "dl" / extra
    if source.is_file():
        source.replace(in_dir / source.name)

env = dict(os.environ)
env["VW_IN"] = str(in_dir)
env["VW_OUT"] = str(out_dir)
print(f"[vw-stage] running: {cfg['command']}", flush=True)
result = subprocess.run(cfg["command"], env=env)
print(f"[vw-stage] exit code: {result.returncode}", flush=True)

api = HfApi()
if any(out_dir.iterdir()):
    api.upload_folder(
        folder_path=str(out_dir),
        path_in_repo=cfg["prefix"] + "/out",
        repo_id=cfg["repo"],
        repo_type="dataset",
    )
    print("[vw-stage] outputs uploaded", flush=True)
else:
    print("[vw-stage] no outputs produced", flush=True)
sys.exit(result.returncode)
"""

_PY_SHIM = "import os;exec(os.environ['VW_BOOTSTRAP'])"
# For stock images without huggingface_hub (e.g. python:3.12 in the echo
# test). The vw-studio-worker image has it preinstalled and skips the pip.
_SH_SHIM = (
    "pip install -q 'huggingface_hub>=0.26' >/dev/null 2>&1 && "
    'python -c "import os;exec(os.environ[\'VW_BOOTSTRAP\'])"'
)


@dataclass
class HfJobsConfig:
    enabled: bool = False
    namespace: str = ""  # HF username/org; resolved from the token when empty
    artifact_repo: str = ""  # private dataset repo id, e.g. "user/vw-studio-artifacts"
    default_flavor: str = "l4x1"
    worker_image: str = "python:3.12"  # replaced by vw-studio-worker in M1
    poll_interval_seconds: float = 15.0

    @classmethod
    def load(cls, path: Path = REMOTE_CONFIG_PATH) -> "HfJobsConfig":
        if not path.exists():
            return cls()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})

    def save(self, path: Path = REMOTE_CONFIG_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def get_hf_token() -> str | None:
    """Token from the OS keyring, falling back to the HF_TOKEN env var."""
    try:
        import keyring

        token = keyring.get_password(KEYRING_SERVICE, KEYRING_ACCOUNT)
        if token:
            return token
    except Exception:  # noqa: BLE001 - keyring backend unavailable
        pass
    import os

    return os.environ.get("HF_TOKEN") or None


def set_hf_token(token: str) -> None:
    import keyring

    keyring.set_password(KEYRING_SERVICE, KEYRING_ACCOUNT, token)


class HfJobsStageRunner(StageRunner):
    name = "hf-jobs"

    def __init__(
        self,
        config: HfJobsConfig | None = None,
        confirm_cost: Callable[[CostEstimate], bool] | None = None,
    ) -> None:
        self.config = config or HfJobsConfig.load()
        self.confirm_cost = confirm_cost

    # -- helpers -------------------------------------------------------------

    def _hub(self):
        import huggingface_hub

        return huggingface_hub

    def _api(self):
        return self._hub().HfApi(token=get_hf_token())

    def _resolve_repo(self, api) -> str:
        repo = self.config.artifact_repo
        if not repo:
            user = self.config.namespace or api.whoami()["name"]
            repo = f"{user}/vw-studio-artifacts"
        api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
        return repo

    # -- StageRunner ---------------------------------------------------------

    def run(self, ctx: StageContext) -> StageResult:
        """Record one model run around the job, then delegate.

        Wrapping rather than inlining because _run has many exits -- cancel,
        terminal failure, missing outputs, success -- and a context manager
        covers all of them without threading a record through each branch.

        StageCancelledError maps to `cancelled` on its own (the recorder reads
        "cancel" in the class name), so a user abort does not land in the
        failure rate. CostDeniedError is set explicitly: declining a price is a
        decision, not a fault, so it is `rejected` -- the same shape as a
        budget refusal in vault-inference.
        """
        with telemetry.job_run(
            model=ctx.stage_key or "unknown-stage",
            stage_key=ctx.stage_key,
            studio_job_id=ctx.job_id,
        ) as run:
            try:
                result = self._run(ctx)
            except CostDeniedError as exc:
                run.set(status="rejected", error_class="CostDenied", error_message=str(exc))
                raise

            meta = result.metadata or {}
            duration_seconds = meta.get("duration_seconds")
            run.set(
                cost_usd=meta.get("actual_usd_estimate"),
                flavor=meta.get("flavor"),
                hf_job_id=meta.get("job_id"),
                hf_job_url=meta.get("job_url"),
                est_usd=meta.get("est_usd"),
                rate_source=meta.get("rate_source"),
                output_count=len(result.artifacts),
            )
            if duration_seconds:
                # The runner's own clock covers the remote job; the recorder's
                # wall clock would also include upload and download.
                run.set(remote_seconds=duration_seconds)
            if result.status != "complete":
                run.set(status="error", error_class="StageIncomplete")
            return result

    def _run(self, ctx: StageContext) -> StageResult:  # noqa: PLR0915
        # ctx.params["flavor"] may be a single flavor string or a list of
        # candidates to try in order (e.g. ["t4-small", "a10g-small"]) — HF
        # Jobs has no server-side "any of these" request, so we submit each
        # candidate and fall back to the next if it doesn't leave SCHEDULING
        # within flavor_scheduling_timeout_seconds. Spot GPU availability
        # (esp. l4x1) fluctuates enough that this is worth having generally,
        # not just for this one preset.
        flavor_param = ctx.params.get("flavor", self.config.default_flavor)
        flavor_candidates = flavor_param if isinstance(flavor_param, list) else [flavor_param]
        est_minutes = float(ctx.params.get("est_minutes", 30))
        # Quote against the priciest candidate so the confirm dialog never undercounts.
        estimate = max(
            (estimate_cost(f, est_minutes) for f in flavor_candidates),
            key=lambda e: e.est_usd,
        )

        # Consent gate comes before any network traffic.
        if self.confirm_cost is None or not self.confirm_cost(estimate):
            raise CostDeniedError(f"Remote run declined ({estimate.summary()}).")

        token = get_hf_token()
        if not token:
            raise RuntimeError(
                "No Hugging Face token configured. Set it in Settings (stored in the OS keyring)."
            )

        hub = self._hub()
        api = self._api()
        repo = self._resolve_repo(api)
        prefix = f"jobs/{ctx.job_id}/{ctx.stage_key}"

        for path in ctx.inputs:
            if ctx.skip_inputs_upload:
                ctx.log(f"[hf-jobs] skipping upload of {path.name} (--resume-job: already on HF)")
            else:
                ctx.log(f"[hf-jobs] uploading input {path.name}")
                api.upload_file(
                    path_or_fileobj=str(path),
                    path_in_repo=f"{prefix}/in/{path.name}",
                    repo_id=repo,
                    repo_type="dataset",
                )

        stage_config = {
            "repo": repo,
            "prefix": prefix,
            "command": ctx.params["command"],
            # Repo-relative paths pulled into VW_IN directly from the artifact
            # dataset — large artifacts (checkpoints) need not be re-uploaded.
            "extra_inputs": ctx.params.get("extra_repo_inputs", []),
        }
        image = ctx.params.get("image", self.config.worker_image)
        if ctx.params.get("image_has_hub", False):
            container_cmd = ["python", "-c", _PY_SHIM]
        else:
            container_cmd = ["sh", "-lc", _SH_SHIM]

        # An explicit None means no remote limit at all. COLMAP's cost is
        # superlinear in image count, so a cap sized from an estimate is a
        # guess, and when it fires the container dies with nothing uploaded.
        raw_timeout = ctx.params.get("timeout_seconds", max(900, est_minutes * 60 * 2))
        timeout_seconds = None if raw_timeout is None else int(raw_timeout)
        poll = max(MIN_POLL_INTERVAL_SECONDS, float(self.config.poll_interval_seconds))
        scheduling_timeout = float(ctx.params.get("flavor_scheduling_timeout_seconds", 120.0))

        job = None
        flavor = None
        for candidate in flavor_candidates:
            ctx.log(f"[hf-jobs] launching job: image={image} flavor={candidate} "
                    f"timeout={'none' if timeout_seconds is None else str(timeout_seconds) + 's'}")
            candidate_job = hub.run_job(
                image=image,
                command=container_cmd,
                env={
                    "VW_BOOTSTRAP": BOOTSTRAP_SOURCE,
                    "VW_STAGE_CONFIG": json.dumps(stage_config),
                },
                secrets={"HF_TOKEN": token},
                flavor=candidate,
                timeout=timeout_seconds,
                # Opens an SSH endpoint on the running container. Jobs have no
                # "dev mode" the way Spaces do; this is the equivalent, and it
                # is the only way to watch a training that Rich refuses to
                # narrate into a pipe.
                ssh=bool(ctx.params.get("ssh", False)),
                token=token,
            )
            ctx.log(f"[hf-jobs] job started: {candidate_job.url}")
            wait_started = time.monotonic()
            left_scheduling = False
            while True:
                if ctx.cancel.cancelled:
                    ctx.log("[hf-jobs] cancelling remote job…")
                    hub.cancel_job(job_id=candidate_job.id, token=token)
                    raise StageCancelledError(f"Remote job {candidate_job.id} cancelled.")
                info = hub.inspect_job(job_id=candidate_job.id, token=token)
                if info.status.stage != "SCHEDULING":
                    left_scheduling = True
                    break
                if time.monotonic() - wait_started > scheduling_timeout:
                    break
                time.sleep(poll)
            if left_scheduling:
                job = candidate_job
                flavor = candidate
                break
            ctx.log(
                f"[hf-jobs] {candidate} still SCHEDULING after {scheduling_timeout:.0f}s "
                f"({'trying next flavor' if candidate is not flavor_candidates[-1] else 'no more flavors to try'}) "
                f"— cancelling {candidate_job.id}"
            )
            hub.cancel_job(job_id=candidate_job.id, token=token)

        if job is None:
            raise RuntimeError(
                f"None of the requested flavors became available within "
                f"{scheduling_timeout:.0f}s each: {flavor_candidates}"
            )

        started = time.monotonic()

        # Stream remote logs in a side thread: nerfstudio progress percentages
        # in the lines drive the GUI progress bar.
        streamed_any = threading.Event()

        def _stream_logs() -> None:
            try:
                for line in hub.fetch_job_logs(job_id=job.id, token=token):
                    streamed_any.set()
                    ctx.log(f"[remote] {line}")
                    match = _PERCENT_RE.search(line)
                    if match:
                        ctx.progress(float(match.group(1)), line.strip())
            except Exception as exc:  # noqa: BLE001 - logs are best-effort
                ctx.log(f"[hf-jobs] log stream ended: {exc}")

        log_thread = threading.Thread(target=_stream_logs, daemon=True)
        log_thread.start()

        last_stage = ""
        announced_ssh = False
        while True:
            if ctx.cancel.cancelled:
                ctx.log("[hf-jobs] cancelling remote job…")
                hub.cancel_job(job_id=job.id, token=token)
                raise StageCancelledError(f"Remote job {job.id} cancelled.")
            info = hub.inspect_job(job_id=job.id, token=token)
            stage_name = info.status.stage
            ssh_url = getattr(info.status, "ssh_url", None)
            if ssh_url and not announced_ssh:
                ctx.log(f"[hf-jobs] ssh: {ssh_url}")
                announced_ssh = True
            if stage_name != last_stage:
                ctx.log(f"[hf-jobs] status: {stage_name}")
                ctx.progress(-1.0, f"Remote job {stage_name.lower()}")
                last_stage = stage_name
            if stage_name in TERMINAL_SUCCESS or stage_name in TERMINAL_FAILURE:
                break
            time.sleep(poll)

        duration = time.monotonic() - started
        actual = estimate_cost(flavor, duration / 60.0)
        log_thread.join(timeout=10)
        if not streamed_any.is_set():
            # The streaming generator yielded nothing live; fetch once post-run.
            try:
                for line in hub.fetch_job_logs(job_id=job.id, token=token):
                    safe_line = line.encode("ascii", "replace").decode("ascii")
                    ctx.log(f"[remote] {safe_line}")
            except Exception as exc:  # noqa: BLE001 - logs are best-effort
                ctx.log(f"[hf-jobs] could not fetch job logs: {exc}")

        cost_metadata = {
            "job_id": job.id,
            "job_url": job.url,
            "flavor": flavor,
            "est_usd": estimate.est_usd,
            "actual_usd_estimate": actual.est_usd,
            "duration_seconds": round(duration, 1),
            "rate_source": estimate.source,
        }
        if last_stage in TERMINAL_FAILURE:
            # Stage entrypoints upload a structured error.json on failure —
            # surface its detail instead of just the job URL.
            detail = ""
            try:
                self._download_outputs(api, repo, prefix, ctx)
                error_file = ctx.job_dir / ctx.stage_key / "remote_out" / "error.json"
                if error_file.exists():
                    payload = json.loads(error_file.read_text(encoding="utf-8"))
                    detail = f" [{payload.get('code')}] {payload.get('detail')}"
            except Exception:  # noqa: BLE001 - diagnostics are best-effort
                pass
            raise RuntimeError(
                f"Remote job ended in {last_stage}.{detail} (see {job.url}). Cost: {cost_metadata}"
            )

        downloaded = self._download_outputs(api, repo, prefix, ctx)
        missing = [p for p in ctx.expected_outputs if not p.exists()]
        if missing:
            raise RuntimeError(
                "Remote job completed but expected outputs are missing locally: "
                + ", ".join(str(p) for p in missing)
            )
        ctx.log(f"[hf-jobs] done in {duration:.0f}s, ~${actual.est_usd:.2f} ({len(downloaded)} files)")
        return StageResult(status="complete", artifacts=downloaded, metadata=cost_metadata)

    def _download_outputs(self, api, repo: str, prefix: str, ctx: StageContext) -> list[Path]:
        out_prefix = f"{prefix}/out/"
        remote_files = [f for f in api.list_repo_files(repo, repo_type="dataset") if f.startswith(out_prefix)]
        by_name: dict[str, Path] = {p.name: p for p in ctx.expected_outputs}
        fallback_dir = ctx.job_dir / ctx.stage_key / "remote_out"
        downloaded: list[Path] = []
        with tempfile.TemporaryDirectory() as tmp:
            for remote in remote_files:
                local_tmp = api.hf_hub_download(
                    repo_id=repo, filename=remote, repo_type="dataset", local_dir=tmp
                )
                relative = Path(remote[len(out_prefix):])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"Invalid output artifact path: {relative}")
                name = relative.as_posix()
                # Only root outputs satisfy the stage contract. Nested files
                # often share names (depth/0001 vs confidence/0001) and must
                # retain their directories rather than overwrite each other.
                target = by_name.get(name) if len(relative.parts) == 1 else None
                if target is None:
                    fallback_dir.mkdir(parents=True, exist_ok=True)
                    target = fallback_dir / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(local_tmp, target)
                downloaded.append(target)
                ctx.log(f"[hf-jobs] downloaded {name} -> {target}")
        return downloaded


def run_echo_smoke_test(
    config: HfJobsConfig,
    confirm_cost: Callable[[CostEstimate], bool],
    log: Callable[[str], None],
) -> str:
    """End-to-end M0 verification: round-trip a file through a real cpu-basic job.

    Costs well under $0.01. Raises on any failure; returns a summary string.
    """
    runner = HfJobsStageRunner(config=config, confirm_cost=confirm_cost)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        probe = tmp_path / "echo_probe.txt"
        probe.write_text(f"vaultwares-studio echo {time.time()}", encoding="utf-8")
        expected = tmp_path / "result" / "echo_probe.txt"
        ctx = StageContext(
            job_dir=tmp_path,
            job_id=f"echo-{time.strftime('%Y%m%d-%H%M%S')}",
            stage_key="echo_test",
            params={
                "image": "python:3.12",
                "flavor": "cpu-basic",
                "est_minutes": 5,
                "timeout_seconds": 900,
                "command": ["sh", "-c", 'cp "$VW_IN"/* "$VW_OUT"/ && echo VW_ECHO_OK'],
            },
            inputs=[probe],
            expected_outputs=[expected],
            log=log,
        )
        result = runner.run(ctx)
        roundtrip = expected.read_text(encoding="utf-8")
        if roundtrip != probe.read_text(encoding="utf-8"):
            raise RuntimeError("Echo content mismatch after round-trip.")
        return (
            f"Echo test OK — job {result.metadata['job_id']} "
            f"({result.metadata['duration_seconds']}s, ~${result.metadata['actual_usd_estimate']})"
        )
