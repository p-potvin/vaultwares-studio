"""Remote reconstruction wiring test: a fake StageRunner stands in for HF Jobs."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from vaultwares_studio.pipeline import (
    DEFAULT_SOURCE_VIDEO,
    DigitalTwinStudioRunner,
    StageState,
    create_job_manifest,
    load_job_manifest,
    save_job_manifest,
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
    # Normal run must NOT set skip_inputs_upload.
    assert ctx.skip_inputs_upload is False


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


# -- --resume-job / skip_inputs_upload ----------------------------------------


def test_skip_inputs_upload_flag_passes_through_to_runner():
    """resume_job_id in manifest.metadata propagates skip_inputs_upload=True to ctx."""
    manifest = _prepare_manifest_through_frames()
    manifest.metadata["preset"] = "draft"
    manifest.metadata["resume_job_id"] = "local-run-20260615-065732"
    fake = FakeRemoteRunner()
    runner = DigitalTwinStudioRunner(manifest, lambda _msg: None, remote_runner=fake)

    runner.run_stage("reconstruction")

    ctx = fake.received_ctx
    assert ctx is not None
    assert ctx.skip_inputs_upload is True
    # The HF artifact prefix must point at the *resumed* job, not the new one.
    assert ctx.job_id == "local-run-20260615-065732"


def test_resume_job_uses_prior_hf_prefix_not_new_job_id():
    """ctx.job_id is the prior job's id when resuming, so HF resolves
    jobs/<prior_id>/reconstruction/in/frames.zip — not the new job's prefix."""
    manifest = _prepare_manifest_through_frames()
    manifest.metadata["preset"] = "draft"
    manifest.metadata["resume_job_id"] = "local-run-prior-abc"

    captured: list[StageContext] = []

    class CapturingRunner(FakeRemoteRunner):
        def run(self, ctx: StageContext) -> StageResult:
            captured.append(ctx)
            return super().run(ctx)

    runner = DigitalTwinStudioRunner(manifest, lambda _msg: None, remote_runner=CapturingRunner())
    runner.run_stage("reconstruction")

    assert captured, "runner was never invoked"
    ctx = captured[0]
    assert ctx.job_id == "local-run-prior-abc"
    assert ctx.skip_inputs_upload is True
    # The local manifest job_id should differ from the HF prefix.
    assert ctx.job_id != manifest.job_id


def test_resume_job_manifest_roundtrip():
    """Simulates the headless_remote_run --resume-job setup: prior manifest is
    loaded, stages are pre-completed, resume_job_id is stamped, and the new
    manifest saves and reloads cleanly."""
    # Build a fake 'prior' manifest that looks like the failed refine job.
    prior = create_job_manifest(source_video=DEFAULT_SOURCE_VIDEO)
    prior.metadata["preset"] = "refine"
    prior.metadata["extra_videos"] = ["inputs/cloudyday2_june14_348sec.MOV"]
    prior.metadata["refine_from_job_id"] = "local-run-20260614-234541"
    save_job_manifest(prior)

    # Simulate what headless_remote_run does when --resume-job is given.
    new_manifest = create_job_manifest(source_video=prior.source_video)
    new_manifest.metadata["preset"] = "refine"
    new_manifest.metadata["extra_videos"] = list(prior.metadata["extra_videos"])
    new_manifest.metadata["refine_from_job_id"] = prior.metadata["refine_from_job_id"]
    new_manifest.metadata["resume_job_id"] = prior.job_id

    for stage in new_manifest.stages:
        if stage.key in ("video_intake", "frame_extraction"):
            stage.state = StageState.COMPLETE.value
            stage.message = f"Skipped (resuming from {prior.job_id})"
    new_manifest.current_stage_key = "reconstruction"
    save_job_manifest(new_manifest)

    # Reload and verify all fields survive the round-trip.
    reloaded = load_job_manifest(Path(new_manifest.output_dir) / "manifest.json")
    assert reloaded.metadata["resume_job_id"] == prior.job_id
    assert reloaded.metadata["refine_from_job_id"] == "local-run-20260614-234541"
    assert reloaded.current_stage_key == "reconstruction"
    completed_keys = {s.key for s in reloaded.stages if s.state == StageState.COMPLETE.value}
    assert "video_intake" in completed_keys
    assert "frame_extraction" in completed_keys
    assert "reconstruction" not in completed_keys


# -- Split-job (SfM cpu-upgrade + training l4x1) ------------------------------


class TrackingRunner(FakeRemoteRunner):
    """Records every StageContext passed to run()."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[StageContext] = []

    def run(self, ctx: StageContext) -> StageResult:
        self.calls.append(ctx)
        # For the SfM job produce sfm_processed_min.zip; for training, produce splat + summary.
        if "sfm" in ctx.stage_key:
            sfm_out = next((p for p in ctx.expected_outputs if "sfm_processed_min" in p.name), None)
            if sfm_out:
                sfm_out.parent.mkdir(parents=True, exist_ok=True)
                sfm_out.write_bytes(b"fake-processed")
        else:
            # Delegate to the normal FakeRemoteRunner for training outputs.
            return super().run(ctx)
        return StageResult(
            status="complete",
            artifacts=list(ctx.expected_outputs),
            metadata={"job_id": "sfm-fake", "flavor": ctx.params.get("flavor"), "actual_usd_estimate": 0.14},
        )


def test_split_job_routes_sfm_to_cpu_and_training_to_gpu():
    """A preset with split_jobs=True must launch two distinct runner calls:
    Job A on sfm_flavor (cpu-upgrade), Job B on flavor (l4x1)."""
    manifest = _prepare_manifest_through_frames()
    manifest.metadata["preset"] = "refine"  # refine has split_jobs=True
    tracker = TrackingRunner()
    runner = DigitalTwinStudioRunner(manifest, lambda _msg: None, remote_runner=tracker)

    runner.run_stage("reconstruction")

    assert len(tracker.calls) == 2, f"Expected 2 runner calls, got {len(tracker.calls)}"
    sfm_ctx, train_ctx = tracker.calls

    # Job A: SfM on cpu-upgrade
    assert sfm_ctx.params["flavor"] == "cpu-upgrade"
    assert sfm_ctx.stage_key == "reconstruction_sfm"
    assert "--sfm-only" in sfm_ctx.params["command"]
    # --refine-mode is only added when refine_from_job_id is in the manifest;
    # this test uses a plain manifest with no base job, so it should be absent.
    assert "--refine-mode" not in sfm_ctx.params["command"]
    assert sfm_ctx.skip_inputs_upload is False  # fresh run, not a resume

    # Job B: training on l4x1
    assert train_ctx.params["flavor"] == "l4x1"
    assert train_ctx.stage_key == "reconstruction"
    assert "--train-only" in train_ctx.params["command"]
    assert train_ctx.skip_inputs_upload is True  # training job never uploads local files


def test_split_job_preset_cost_is_combined():
    """Refine preset with split_jobs=True: cost() returns cpu-upgrade + l4x1 combined."""
    from vaultwares_studio.presets import get_preset
    preset = get_preset("refine")
    assert preset.split_jobs is True
    combined = preset.cost()
    # cpu-upgrade 90 min @ $0.10/hr = $0.15; l4x1 20 min @ $0.80/hr = $0.27
    assert combined.est_usd == pytest.approx(0.42, abs=0.05)
    # Individual cost helpers
    assert preset.sfm_cost().flavor == "cpu-upgrade"
    assert preset.train_cost().flavor == "l4x1"
