import json
import os
from pathlib import Path

import pytest

from vaultwares_studio.camera_director import build_camera_bundle
from vaultwares_studio.integration import build_vaultflows_workflow
from vaultwares_studio.pipeline import (
    DEFAULT_SOURCE_VIDEO,
    DigitalTwinStudioRunner,
    StageState,
    create_job_manifest,
    list_job_manifests,
    load_job_manifest,
    load_latest_job_manifest,
    next_incomplete_stage_key,
    stage_dependencies_complete,
)


def test_create_job_manifest_writes_resumable_job():
    manifest = create_job_manifest(source_video=DEFAULT_SOURCE_VIDEO)
    manifest_path = Path(manifest.output_dir) / "manifest.json"

    assert manifest_path.exists()
    loaded = load_job_manifest(manifest_path)
    assert loaded.job_id == manifest.job_id
    assert loaded.state == StageState.QUEUED.value
    assert [stage.key for stage in loaded.stages] == [
        "video_intake",
        "frame_extraction",
        "reconstruction",
        "usd_cameras",
        "cosmos_output",
    ]
    assert next_incomplete_stage_key(loaded) == "video_intake"
    assert stage_dependencies_complete(loaded, "video_intake") is True
    assert stage_dependencies_complete(loaded, "reconstruction") is False


def test_job_manifest_history_loads_latest_job(tmp_path):
    older = create_job_manifest(source_video=DEFAULT_SOURCE_VIDEO)
    newer = create_job_manifest(source_video=DEFAULT_SOURCE_VIDEO)
    older_dir = tmp_path / "older"
    newer_dir = tmp_path / "newer"
    older_dir.mkdir()
    newer_dir.mkdir()
    older_path = older_dir / "manifest.json"
    newer_path = newer_dir / "manifest.json"
    older_path.write_text(json.dumps(older.to_dict()), encoding="utf-8")
    newer_path.write_text(json.dumps(newer.to_dict()), encoding="utf-8")
    os.utime(older_path, (1, 1))
    os.utime(newer_path, (2, 2))

    manifests = list_job_manifests(tmp_path)
    latest = load_latest_job_manifest(tmp_path)

    assert manifests == [newer_path, older_path]
    assert latest is not None
    assert latest.job_id == newer.job_id


def test_runner_rejects_stage_when_previous_steps_are_incomplete():
    manifest = create_job_manifest(source_video=DEFAULT_SOURCE_VIDEO)
    runner = DigitalTwinStudioRunner(manifest, lambda _message: None)

    with pytest.raises(RuntimeError, match="Complete earlier stages"):
        runner.run_stage("reconstruction")


def test_camera_bundle_contains_presets_and_prompt_plan():
    bundle = build_camera_bundle("show me the desk from the doorway, then orbit left and rise")

    assert bundle["presets"]
    assert bundle["promptPlan"]
    assert any(shot["name"] == "Doorway Start" for shot in bundle["promptPlan"])
    assert any(shot["name"] == "Orbit Move" for shot in bundle["promptPlan"])
    assert any(shot["name"] == "Rise Shot" for shot in bundle["promptPlan"])
    assert len(bundle["allShots"]) >= len(bundle["presets"])


def test_vaultflows_workflow_export_shape():
    manifest = create_job_manifest(source_video=DEFAULT_SOURCE_VIDEO)
    workflow = build_vaultflows_workflow(manifest)

    assert workflow["id"] == manifest.job_id
    assert workflow["category"] == "Digital Twin"
    assert workflow["pin"] is True
    assert workflow["favorite"] is True
    assert len(workflow["steps"]) == len(manifest.stages)
    assert workflow["steps"][0]["id"] == "video_intake"
