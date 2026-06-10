import json
import sys
import threading
import time
from pathlib import Path

import pytest

from vaultwares_studio.pipeline import (
    DEFAULT_SOURCE_VIDEO,
    MANIFEST_SCHEMA_VERSION,
    JobManifest,
    compute_extraction_fps,
    create_job_manifest,
    record_spend,
)


def test_compute_extraction_fps_targets_100_frames():
    assert compute_extraction_fps(12) == 8     # short clip: dense sampling
    assert compute_extraction_fps(50) == 2     # ~100 frames at 2 fps
    assert compute_extraction_fps(600) == 2    # long video: floor at 2 fps
    assert compute_extraction_fps(5) == 10     # very short: ceiling at 10 fps
    assert compute_extraction_fps(None) == 2   # unknown duration: legacy default
    assert compute_extraction_fps(0) == 2
from vaultwares_studio.runners import (
    CancelToken,
    CostDeniedError,
    HfJobsConfig,
    HfJobsStageRunner,
    LocalStageRunner,
    StageCancelledError,
    StageContext,
    estimate_cost,
)


# -- LocalStageRunner ---------------------------------------------------------


def test_local_runner_streams_output_lines():
    runner = LocalStageRunner()
    lines: list[str] = []
    runner.run_command(
        [sys.executable, "-c", "print('alpha'); print('beta')"],
        error_message="stream test failed",
        timeout_seconds=60,
        log=lines.append,
    )
    assert "alpha" in lines
    assert "beta" in lines


def test_local_runner_raises_on_nonzero_exit():
    runner = LocalStageRunner()
    with pytest.raises(RuntimeError, match="boom .exit code=3."):
        runner.run_command(
            [sys.executable, "-c", "import sys; sys.exit(3)"],
            error_message="boom",
            timeout_seconds=60,
            log=lambda _line: None,
        )


def test_local_runner_cancel_kills_process():
    runner = LocalStageRunner()
    cancel = CancelToken()
    timer = threading.Timer(0.5, cancel.cancel)
    timer.start()
    started = time.monotonic()
    with pytest.raises(StageCancelledError):
        runner.run_command(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            error_message="cancel test",
            timeout_seconds=120,
            log=lambda _line: None,
            cancel=cancel,
        )
    assert time.monotonic() - started < 30


def test_local_runner_timeout():
    runner = LocalStageRunner()
    with pytest.raises(RuntimeError, match="timed out"):
        runner.run_command(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            error_message="timeout test",
            timeout_seconds=1,
            log=lambda _line: None,
        )


def test_local_runner_run_checks_expected_outputs(tmp_path):
    runner = LocalStageRunner()
    missing = tmp_path / "never_written.txt"
    ctx = StageContext(
        job_dir=tmp_path,
        job_id="job",
        stage_key="stage",
        params={"argv": [sys.executable, "-c", "print('ok')"]},
        expected_outputs=[missing],
    )
    with pytest.raises(RuntimeError, match="expected outputs are missing"):
        runner.run(ctx)


# -- Cost estimation / consent gate -------------------------------------------


def test_estimate_cost_math():
    estimate = estimate_cost("l4x1", 30)
    assert estimate.flavor == "l4x1"
    assert estimate.est_usd == pytest.approx(0.40, abs=0.01)
    assert "l4x1" in estimate.summary()


def test_hf_runner_denies_without_confirmation(tmp_path):
    runner = HfJobsStageRunner(config=HfJobsConfig(), confirm_cost=None)
    ctx = StageContext(job_dir=tmp_path, job_id="job", stage_key="stage", params={"command": ["true"]})
    with pytest.raises(CostDeniedError):
        runner.run(ctx)


def test_hf_runner_denies_when_user_declines(tmp_path):
    runner = HfJobsStageRunner(config=HfJobsConfig(), confirm_cost=lambda _est: False)
    ctx = StageContext(job_dir=tmp_path, job_id="job", stage_key="stage", params={"command": ["true"]})
    with pytest.raises(CostDeniedError):
        runner.run(ctx)


def test_hf_config_round_trip(tmp_path):
    path = tmp_path / "remote_compute.json"
    config = HfJobsConfig(enabled=True, artifact_repo="user/repo", default_flavor="a10g-small")
    config.save(path)
    loaded = HfJobsConfig.load(path)
    assert loaded == config


def test_hf_config_load_ignores_unknown_keys(tmp_path):
    path = tmp_path / "remote_compute.json"
    path.write_text(json.dumps({"default_flavor": "t4-small", "future_field": 1}), encoding="utf-8")
    loaded = HfJobsConfig.load(path)
    assert loaded.default_flavor == "t4-small"


# -- Manifest schema v2 --------------------------------------------------------


def test_new_manifest_has_schema_v2_and_placements():
    manifest = create_job_manifest(source_video=DEFAULT_SOURCE_VIDEO)
    assert manifest.schema_version == MANIFEST_SCHEMA_VERSION
    assert manifest.spend_ledger == []
    placements = {stage.key: stage.placement for stage in manifest.stages}
    assert placements["reconstruction"] == "remote"
    assert placements["frame_extraction"] == "local"
    payload = manifest.to_dict()
    assert payload["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert payload["stages"][2]["placement"] == "remote"


def test_v1_manifest_migrates_on_load():
    manifest = create_job_manifest(source_video=DEFAULT_SOURCE_VIDEO)
    payload = manifest.to_dict()
    # Simulate a v1 manifest written before schema versioning existed.
    payload.pop("schema_version")
    payload.pop("spend_ledger")
    for stage in payload["stages"]:
        for key in ("placement", "runner", "params", "cost"):
            stage.pop(key)

    migrated = JobManifest.from_dict(payload)
    assert migrated.schema_version == MANIFEST_SCHEMA_VERSION
    assert migrated.spend_ledger == []
    placements = {stage.key: stage.placement for stage in migrated.stages}
    assert placements["reconstruction"] == "remote"
    assert placements["video_intake"] == "local"


def test_record_spend_appends_ledger_and_stage_cost():
    manifest = create_job_manifest(source_video=DEFAULT_SOURCE_VIDEO)
    record_spend(
        manifest,
        "reconstruction",
        {"job_id": "abc123", "flavor": "l4x1", "actual_usd_estimate": 0.42},
    )
    assert len(manifest.spend_ledger) == 1
    assert manifest.spend_ledger[0]["stage"] == "reconstruction"
    assert manifest.spend_ledger[0]["actual_usd_estimate"] == 0.42
    recon = next(stage for stage in manifest.stages if stage.key == "reconstruction")
    assert recon.cost["job_id"] == "abc123"
    # Ledger survives a save/load round trip.
    reloaded = JobManifest.from_dict(
        json.loads((Path(manifest.output_dir) / "manifest.json").read_text(encoding="utf-8"))
    )
    assert reloaded.spend_ledger[0]["job_id"] == "abc123"
