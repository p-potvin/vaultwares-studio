"""Remote reconstruction wiring test: a fake StageRunner stands in for HF Jobs."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from vaultwares_studio.pipeline import (
    DEFAULT_SOURCE_VIDEO,
    DigitalTwinStudioRunner,
    StageState,
    create_job_manifest,
)
from vaultwares_studio.runners import StageContext, StageResult, StageRunner
from vaultwares_studio.splat_io import GaussianSplat, is_gaussian_ply, write_gaussian_ply


def make_splat(count: int = 250) -> GaussianSplat:
    rng = np.random.default_rng(7)
    return GaussianSplat(
        positions=rng.normal(size=(count, 3)).astype(np.float32),
        sh0=rng.normal(size=(count, 3)).astype(np.float32),
        opacity=rng.normal(size=count).astype(np.float32),
        scales=rng.normal(size=(count, 3)).astype(np.float32),
        rotations=rng.normal(size=(count, 4)).astype(np.float32),
    )


class FakeRemoteRunner(StageRunner):
    name = "fake-remote"

    def __init__(self) -> None:
        self.config = SimpleNamespace(worker_image="example/vw-studio-worker:0.1")
        self.received_ctx: StageContext | None = None

    def run(self, ctx: StageContext) -> StageResult:
        self.received_ctx = ctx
        splat_path = next(p for p in ctx.expected_outputs if p.name == "splat.ply")
        summary_path = next(p for p in ctx.expected_outputs if p.name == "summary.json")
        write_gaussian_ply(make_splat(count=250), splat_path)
        summary_path.write_text('{"registered_images": 42}', encoding="utf-8")
        return StageResult(
            status="complete",
            artifacts=list(ctx.expected_outputs),
            metadata={"job_id": "fake123", "flavor": "l4x1", "actual_usd_estimate": 0.31},
        )


def _prepare_manifest_through_frames():
    manifest = create_job_manifest(source_video=DEFAULT_SOURCE_VIDEO)
    frames_dir = Path(manifest.output_dir) / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), "#336699").save(frames_dir / "frame_0001.png")
    for stage in manifest.stages:
        if stage.key in ("video_intake", "frame_extraction"):
            stage.state = StageState.COMPLETE.value
    return manifest


def test_remote_reconstruction_round_trip():
    manifest = _prepare_manifest_through_frames()
    manifest.metadata["preset"] = "draft"
    fake = FakeRemoteRunner()
    runner = DigitalTwinStudioRunner(manifest, lambda _msg: None, remote_runner=fake)

    runner.run_stage("reconstruction")

    recon_dir = Path(manifest.output_dir) / "reconstruction"
    assert is_gaussian_ply(recon_dir / "cloud.ply")
    assert (recon_dir / "cloud_preview.ply").exists()
    assert (recon_dir / "cloud.usda").exists()

    stage = next(s for s in manifest.stages if s.key == "reconstruction")
    assert stage.state == StageState.COMPLETE.value
    assert stage.metadata["degraded"] is False
    assert stage.metadata["gaussians"] == 250
    assert stage.metadata["preset"] == "draft"
    assert stage.runner == "fake-remote"
    assert manifest.spend_ledger and manifest.spend_ledger[0]["job_id"] == "fake123"

    ctx = fake.received_ctx
    assert ctx is not None
    assert ctx.params["flavor"] == "l4x1"  # draft preset flavor
    assert ctx.params["image"] == "example/vw-studio-worker:0.1"
    assert ctx.inputs and ctx.inputs[0].name == "frames.zip"
    assert "--train-args" in ctx.params["command"]


def test_remote_failure_falls_back_to_placeholder(monkeypatch):
    manifest = _prepare_manifest_through_frames()

    class ExplodingRunner(FakeRemoteRunner):
        def run(self, ctx: StageContext) -> StageResult:
            raise RuntimeError("simulated remote failure")

    # Keep the test hermetic: pretend no local reconstruction tools exist so
    # the fallback goes straight to placeholder outputs.
    monkeypatch.setattr("vaultwares_studio.pipeline.resolve_binary", lambda _name: None)

    runner = DigitalTwinStudioRunner(manifest, lambda _msg: None, remote_runner=ExplodingRunner())
    runner.run_stage("reconstruction")

    stage = next(s for s in manifest.stages if s.key == "reconstruction")
    # Local toolchain (COLMAP) may be absent in CI: the stage still completes
    # with placeholder-safe outputs instead of failing the job.
    assert stage.state == StageState.COMPLETE.value
    recon_dir = Path(manifest.output_dir) / "reconstruction"
    assert (recon_dir / "cloud.ply").exists()
    assert (recon_dir / "cloud.usda").exists()
