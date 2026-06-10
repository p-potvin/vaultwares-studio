import json
from pathlib import Path

import numpy as np
import pytest

from vaultwares_studio.camera_paths import (
    CameraEntity,
    CameraKeyframe,
    author_usd_camera,
    build_visit_path,
    camera_to_world,
    load_captured_entities,
    sample_path,
    to_nerfstudio_camera_path,
)


def make_path_entity() -> CameraEntity:
    return CameraEntity(
        name="Test Path",
        keyframes=[
            CameraKeyframe(t=0.0, position=[0, 1, 4], look_at=[0, 0, 0]),
            CameraKeyframe(t=2.0, position=[4, 1, 0], look_at=[0, 0, 0]),
            CameraKeyframe(t=4.0, position=[0, 1, -4], look_at=[0, 0, 0]),
        ],
    )


def test_camera_to_world_basis():
    matrix = camera_to_world([0, 0, 5], [0, 0, 0])
    rotation = matrix[:3, :3]
    # Orthonormal
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
    # -Z column points from camera toward the target
    forward = -rotation[:, 2]
    np.testing.assert_allclose(forward, [0, 0, -1], atol=1e-9)
    np.testing.assert_allclose(matrix[:3, 3], [0, 0, 5])


def test_sample_path_endpoints_and_count():
    entity = make_path_entity()
    frames = sample_path(entity, fps=30)
    assert len(frames) == 4 * 30 + 1
    np.testing.assert_allclose(frames[0][0], [0, 1, 4], atol=1e-6)
    np.testing.assert_allclose(frames[-1][0], [0, 1, -4], atol=1e-6)


def test_static_entity_samples_once():
    entity = CameraEntity(name="Still", keyframes=[CameraKeyframe(0.0, [1, 2, 3], [0, 0, 0])])
    assert not entity.is_path
    assert len(sample_path(entity)) == 1


def test_nerfstudio_camera_path_document():
    doc = to_nerfstudio_camera_path(make_path_entity(), fps=24, width=1280, height=720)
    assert doc["fps"] == 24
    assert doc["render_width"] == 1280
    assert doc["seconds"] == pytest.approx(4.0, abs=0.1)
    assert len(doc["camera_path"]) == 4 * 24 + 1
    first = doc["camera_path"][0]
    assert len(first["camera_to_world"]) == 16
    assert first["aspect"] == pytest.approx(1280 / 720)


def test_captured_round_trip_and_visit_path(tmp_path):
    captured = tmp_path / "captured_cameras.json"
    captured.write_text(
        json.dumps(
            [
                {"name": "Captured 1", "position": [0, 1, 2], "lookAt": [0, 0, 0], "fovDegrees": 55},
                {"name": "Captured 2", "position": [2, 1, 0], "lookAt": [0, 0, 0]},
            ]
        ),
        encoding="utf-8",
    )
    entities = load_captured_entities(captured)
    assert [entity.name for entity in entities] == ["Captured 1", "Captured 2"]
    assert entities[0].fov_degrees == 55

    path = build_visit_path(entities, seconds_per_stop=2.0)
    assert path is not None and path.is_path
    assert path.duration == pytest.approx(2.0)
    assert build_visit_path(entities[:1]) is None


def test_author_usd_camera_time_samples(tmp_path):
    from pxr import Usd

    stage = Usd.Stage.CreateNew(str(tmp_path / "cams.usda"))
    author_usd_camera(stage, "/World/PathCam", make_path_entity(), fps=24)
    author_usd_camera(
        stage,
        "/World/StillCam",
        CameraEntity(name="Still", keyframes=[CameraKeyframe(0.0, [1, 1, 1], [0, 0, 0])]),
    )
    prim = stage.GetPrimAtPath("/World/PathCam")
    op = prim.GetAttribute("xformOp:transform")
    assert op.GetNumTimeSamples() == 3
    still = stage.GetPrimAtPath("/World/StillCam")
    assert still.GetAttribute("xformOp:transform").GetNumTimeSamples() == 0
    assert prim.GetAttribute("vw:cameraName").Get() == "Test Path"


def test_usd_cameras_stage_integrates_captures():
    from vaultwares_studio.pipeline import (
        DEFAULT_SOURCE_VIDEO,
        DigitalTwinStudioRunner,
        StageState,
        create_job_manifest,
    )

    manifest = create_job_manifest(source_video=DEFAULT_SOURCE_VIDEO)
    for record in manifest.stages:
        if record.key in ("video_intake", "frame_extraction", "reconstruction"):
            record.state = StageState.COMPLETE.value
    usd_dir = Path(manifest.output_dir) / "usd"
    usd_dir.mkdir(parents=True, exist_ok=True)
    (usd_dir / "captured_cameras.json").write_text(
        json.dumps(
            [
                {"name": "Captured 1", "position": [0, 1, 2], "lookAt": [0, 0, 0]},
                {"name": "Captured 2", "position": [2, 1, 0], "lookAt": [0, 0, 0]},
            ]
        ),
        encoding="utf-8",
    )

    runner = DigitalTwinStudioRunner(manifest, lambda _m: None)
    runner.run_stage("usd_cameras")

    names = [camera["name"] for camera in manifest.metadata["cameras"]]
    assert "Captured 1" in names and "Captured Walkthrough" in names
    stage_record = next(record for record in manifest.stages if record.key == "usd_cameras")
    assert stage_record.metadata["capturedCount"] == 2
    assert stage_record.metadata["renderPath"] == "Captured Walkthrough"

    render_doc = json.loads((usd_dir / "camera_path.json").read_text(encoding="utf-8"))
    assert len(render_doc["camera_path"]) >= 2
    assert len(render_doc["camera_path"][0]["camera_to_world"]) == 16
