"""Recover a remote reconstruction whose local launcher poller died.

The HF Job keeps running and uploads outputs to the artifact dataset regardless
of whether the launcher process is alive. This script pulls those outputs into
the expected on-disk layout, runs the recon tail (splat conversion + gravity
alignment + packed .splat write), and marks the reconstruction stage COMPLETE
so the GUI can advance to camera_staging.

Usage:
    python tools/recover_remote_recon.py --job <local-job-id>
    # e.g. local-run-20260613-211202
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vaultwares_studio.pipeline import (  # noqa: E402
    DigitalTwinStudioRunner,
    StageState,
    load_job_manifest,
    save_job_manifest,
)
from vaultwares_studio.splat_io import convert_splat_outputs, is_gaussian_ply  # noqa: E402


def _resolve_repo() -> str:
    """Resolve the artifact dataset id the same way hf_jobs does."""
    from huggingface_hub import HfApi
    from vaultwares_studio.runners.hf_jobs import get_hf_token

    api = HfApi(token=get_hf_token())
    return f"{api.whoami()['name']}/vw-studio-artifacts"


def _download_recon_outputs(job_id: str, recon_dir: Path, log) -> None:
    from huggingface_hub import HfApi
    from vaultwares_studio.runners.hf_jobs import get_hf_token

    token = get_hf_token()
    if not token:
        raise RuntimeError("HF token not configured.")
    api = HfApi(token=token)
    repo = _resolve_repo()
    out_prefix = f"jobs/{job_id}/reconstruction/out/"
    remote_files = [f for f in api.list_repo_files(repo, repo_type="dataset") if f.startswith(out_prefix)]
    if not remote_files:
        raise RuntimeError(f"No files found at {repo}:{out_prefix}")
    log(f"[recover] found {len(remote_files)} artifact(s) at {repo}:{out_prefix}")

    # Mirror _run_remote_reconstruction.expected_outputs target mapping.
    targets = {
        "splat.ply": recon_dir / "gsplat_export" / "splat.ply",
        "summary.json": recon_dir / "summary.json",
        "model.zip": recon_dir / "remote_out" / "model.zip",
        "processed_min.zip": recon_dir / "remote_out" / "processed_min.zip",
    }
    fallback_dir = recon_dir / "remote_out"
    with tempfile.TemporaryDirectory() as tmp:
        for remote in remote_files:
            local_tmp = api.hf_hub_download(
                repo_id=repo, filename=remote, repo_type="dataset", local_dir=tmp
            )
            name = Path(remote).name
            target = targets.get(name, fallback_dir / name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(local_tmp, target)
            log(f"[recover] {name} -> {target.relative_to(recon_dir.parent)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True, help="Local job id, e.g. local-run-20260613-211202")
    args = parser.parse_args()

    job_dir = REPO_ROOT / "data" / "jobs" / args.job
    manifest_path = job_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    manifest = load_job_manifest(manifest_path)
    runner = DigitalTwinStudioRunner(manifest, lambda msg: print(msg))

    runner.recon_dir.mkdir(parents=True, exist_ok=True)
    _download_recon_outputs(args.job, runner.recon_dir, runner.log)

    splat_path = runner.recon_dir / "gsplat_export" / "splat.ply"
    if not splat_path.exists() or not is_gaussian_ply(splat_path):
        print(f"splat.ply missing or not gaussian: {splat_path}", file=sys.stderr)
        return 1

    stage = runner.stage_for("reconstruction")
    info = convert_splat_outputs(
        splat_path,
        runner.recon_ply_path,
        runner.recon_preview_ply_path,
        runner.recon_stage_path,
        runner.log,
    )
    stage.metadata.update(info)
    runner._gravity_align(stage)
    runner._write_packed_splat(stage)

    stage.metadata["degraded"] = False
    stage.metadata.setdefault("preset", manifest.metadata.get("preset", "standard"))
    stage.metadata["recoveredFromHF"] = True
    stage.message = (
        f"Reconstruction recovered from HF Dataset "
        f"({stage.metadata.get('gaussians', '?')} gaussians)."
    )
    stage.state = StageState.COMPLETE.value

    runner._add_artifact(stage, "Reconstruction Stage", "usd", runner.recon_stage_path, "Reconstruction stage.")
    runner._add_artifact(stage, "Reconstruction PLY", "ply", runner.recon_ply_path, "Gaussian splat output.")
    if runner.recon_preview_ply_path.exists():
        runner._add_artifact(
            stage, "Preview Point Cloud", "ply", runner.recon_preview_ply_path,
            "Decimated point cloud for the live viewer.",
        )

    manifest.current_stage_key = "camera_staging"
    manifest.state = StageState.QUEUED.value
    save_job_manifest(manifest)
    print(f"[recover] manifest saved. Next stage: camera_staging")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
