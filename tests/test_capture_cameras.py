import json
import math
import zipfile

import numpy as np
import pytest
from pxr import Usd, UsdGeom

from vaultwares_studio.camera_scene import compose_scene, save_active_camera, scene_frame_transform
from vaultwares_studio.camera_paths import CameraEntity, CameraKeyframe
from vaultwares_studio.capture_cameras import (
    CaptureFrame,
    frames_from_transforms,
    intrinsics_to_usd,
    load_capture_cameras_json,
    load_capture_frames,
    trajectory_stats,
    write_capture_cameras_json,
)


def _c2w(position, yaw_deg=0.0):
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    m = np.eye(4)
    m[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    m[:3, 3] = position
    return m


def _transforms(count=5, w=1920, h=1080, fx=1600.0):
    return {
        "frames": [
            {"file_path": f"images/frame_{i:05d}.jpg", "transform_matrix": _c2w([i, 0.0, 0.0], yaw_deg=10 * i).tolist(),
             "fl_x": fx, "fl_y": fx + 5, "cx": w / 2, "cy": h / 2, "w": w, "h": h}
            for i in range(count)
        ]
    }


def test_frames_from_transforms_spreads_time_over_duration():
    frames = frames_from_transforms(_transforms(5), duration=8.0)
    assert [f.time for f in frames] == [0.0, 2.0, 4.0, 6.0, 8.0]
    assert frames[3].position.tolist() == [3.0, 0.0, 0.0]
    assert frames[0].intrinsic_matrix[0] == [1600.0, 0.0, 960.0]
    assert frames[0].vertical_fov_degrees == pytest.approx(math.degrees(2 * math.atan(540 / 1605)))


def test_frames_from_transforms_falls_back_to_fps():
    frames = frames_from_transforms({**_transforms(3), "fps": 10}, duration=None)
    assert [f.time for f in frames] == [0.0, 0.1, 0.2]


def test_trajectory_stats_reports_closure():
    square = _transforms(5)
    positions = [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 0, 0.05]]
    for frame, position in zip(square["frames"], positions):
        m = np.eye(4); m[:3, 3] = position
        frame["transform_matrix"] = m.tolist()
    stats = trajectory_stats(frames_from_transforms(square, duration=4.0))
    assert stats["frames"] == 5
    assert stats["closure_distance"] == pytest.approx(0.05)
    assert stats["closure_ratio"] < 0.1
    assert stats["path_length"] == pytest.approx(3.95)


def test_intrinsics_to_usd_preserves_horizontal_fov_and_offsets():
    lens = intrinsics_to_usd(fx=960.0, fy=960.0, cx=1000.0, cy=500.0, width=1920, height=1080)
    # fx == w/2 means a 90 degree horizontal FOV: focal == half the filmback.
    assert lens["focal_length"] == pytest.approx(18.0)
    assert lens["horizontal_aperture"] == 36.0
    assert lens["vertical_aperture"] == pytest.approx(36.0 * 1080 / 1920)
    assert lens["horizontal_aperture_offset"] == pytest.approx(40 * 36 / 1920)
    # Image y is down, USD y is up: a principal point above centre is +y.
    assert lens["vertical_aperture_offset"] == pytest.approx(40 * 36 / 1920)
    with pytest.raises(ValueError):
        intrinsics_to_usd(0, 1, 1, 1, 10, 10)


def test_capture_cameras_json_round_trip(tmp_path):
    frames = frames_from_transforms(_transforms(4), duration=3.0)
    path = write_capture_cameras_json(tmp_path, frames, extra={"job_id": "j"})
    payload = json.loads(path.read_text())
    assert payload["schema"] == 1 and payload["job_id"] == "j"
    assert payload["frames"][2]["K"][1][1] == 1605.0
    assert payload["trajectory"]["frames"] == 4
    loaded = load_capture_cameras_json(tmp_path)
    assert [f.index for f in loaded] == [0, 1, 2, 3]
    assert loaded[1].c2w == frames[1].c2w


def test_load_capture_frames_uses_scene_world(tmp_path):
    remote = tmp_path / "reconstruction" / "remote_out"
    remote.mkdir(parents=True)
    with zipfile.ZipFile(remote / "processed_min.zip", "w") as archive:
        archive.writestr("transforms.json", json.dumps(_transforms(3)))
    with zipfile.ZipFile(remote / "model.zip", "w") as archive:
        archive.writestr("train/dataparser_transforms.json", json.dumps({
            "transform": np.eye(4)[:3].tolist(), "scale": 2.0}))
    frames = load_capture_frames(tmp_path, duration=2.0)
    # Trainer scale doubles every position; intrinsics are untouched.
    assert [f.position[0] for f in frames] == [0.0, 2.0, 4.0]
    assert frames[0].fx == 1600.0
    transform = scene_frame_transform(tmp_path)
    np.testing.assert_allclose(transform[:3, :3], 2.0 * np.eye(3))
    np.testing.assert_allclose(transform @ np.array([1, 0, 0, 1.0]), [2, 0, 0, 1])


def test_compose_scene_authors_capture_cameras_and_mesh(tmp_path):
    cloud = tmp_path / "reconstruction" / "cloud.usda"
    cloud.parent.mkdir(parents=True)
    for target in (cloud, tmp_path / "reconstruction" / "mesh.usda"):
        stage = Usd.Stage.CreateNew(str(target))
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        UsdGeom.Cube.Define(stage, "/World/Thing")
        stage.GetRootLayer().Save()
    frames = frames_from_transforms(_transforms(60), duration=4.0)
    scene = tmp_path / "usd" / "digital_twin_scene.usda"
    entity = CameraEntity("Orbit", keyframes=[CameraKeyframe(0, [0, 1, 4], [0, 0, 0]), CameraKeyframe(2, [4, 1, 0], [0, 0, 0])])
    compose_scene(scene, cloud, [entity], capture_frames=frames, mesh=tmp_path / "reconstruction" / "mesh.usda")
    stage = Usd.Stage.Open(str(scene))
    animated = stage.GetPrimAtPath("/World/Capture/CaptureCamera")
    assert animated.IsA(UsdGeom.Camera)
    camera = UsdGeom.Camera(animated)
    # Per-frame intrinsics are time samples; the 90 degree lens is 18 mm.
    assert camera.GetFocalLengthAttr().GetNumTimeSamples() == 60
    assert camera.GetFocalLengthAttr().Get(0) == pytest.approx(1600 * 36 / 1920)
    matrix = np.asarray(animated.GetAttribute("xformOp:transform").Get(4.0 * 30)).T
    np.testing.assert_allclose(matrix[:3, 3], [59, 0, 0], atol=1e-6)
    assert stage.GetPrimAtPath("/World/Capture/Trajectory").IsA(UsdGeom.BasisCurves)
    keyframes = [p for p in stage.GetPrimAtPath("/World/Capture/Keyframes").GetChildren()]
    # Every 25th frame plus the last one.
    assert [p.GetAttribute("vw:frameIndex").Get() for p in keyframes] == [0, 25, 50, 59]
    assert stage.GetPrimAtPath("/World/DigitalTwin/Surface/Thing")
    assert stage.GetPrimAtPath("/World/Capture").GetAttribute("vw:frameCount").Get() == 60
    text = scene.read_text()
    assert "@../reconstruction/mesh.usda@" in text


def test_save_active_camera_keeps_capture_cameras(tmp_path):
    cloud = tmp_path / "reconstruction" / "cloud.usda"
    cloud.parent.mkdir(parents=True)
    stage = Usd.Stage.CreateNew(str(cloud))
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
    stage.GetRootLayer().Save()
    write_capture_cameras_json(tmp_path, frames_from_transforms(_transforms(3), duration=1.0))
    entity = CameraEntity("Custom", keyframes=[CameraKeyframe(0, [0, 1, 4], [0, 0, 0]), CameraKeyframe(2, [4, 1, 0], [0, 0, 0])])
    save_active_camera(tmp_path, entity)
    scene = Usd.Stage.Open(str(tmp_path / "usd" / "digital_twin_scene.usda"))
    assert scene.GetPrimAtPath("/World/Navigation/Camera_1").IsA(UsdGeom.Camera)
    assert scene.GetPrimAtPath("/World/Capture/CaptureCamera").IsA(UsdGeom.Camera)
