import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom

from vaultwares_studio.camera_paths import CameraEntity, CameraKeyframe
from vaultwares_studio.camera_scene import compose_scene, save_active_camera, prepare_retrace_transforms


def path_entity():
    return CameraEntity("Custom", keyframes=[CameraKeyframe(0, [0, 1, 4], [0, 0, 0]),
                                           CameraKeyframe(2, [4, 1, 0], [0, 0, 0])])


def write_cloud(job):
    cloud = job / "reconstruction" / "cloud.usda"
    cloud.parent.mkdir(parents=True)
    stage = Usd.Stage.CreateNew(str(cloud))
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Cube.Define(stage, "/World/Fixture")
    stage.GetRootLayer().Save()
    return cloud


def test_scene_portable_after_copy_and_repeated_save(tmp_path):
    job = tmp_path / "source"
    write_cloud(job)
    entity = path_entity()
    save_active_camera(job, entity)
    save_active_camera(job, entity)
    shutil.copytree(job, tmp_path / "moved")
    shutil.rmtree(job)
    scene = Usd.Stage.Open(str(tmp_path / "moved/usd/digital_twin_scene.usda"))
    assert scene.GetPrimAtPath("/World/DigitalTwin/Fixture")
    assert scene.GetPrimAtPath("/World/Navigation/Camera_1").IsA(UsdGeom.Camera)
    assert scene.GetEndTimeCode() == 60
    text = (tmp_path / "moved/usd/digital_twin_scene.usda").read_text()
    assert "@../reconstruction/cloud.usda@" in text


def test_render_path_undoes_gravity_without_changing_usd(tmp_path):
    write_cloud(tmp_path)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    (tmp_path / "reconstruction/summary.json").write_text(json.dumps({
        "gravity_aligned": True, "alignment": {"rotation": rotation.tolist()}}))
    entity = path_entity()
    save_active_camera(tmp_path, entity)
    doc = json.loads((tmp_path / "usd/camera_path.json").read_text())
    render = np.array(doc["camera_path"][0]["camera_to_world"]).reshape(4, 4)
    np.testing.assert_allclose(render[:3, 3], rotation.T @ entity.keyframes[0].position)
    stage = Usd.Stage.Open(str(tmp_path / "usd/digital_twin_scene.usda"))
    usd = np.asarray(stage.GetPrimAtPath("/World/Navigation/Camera_1").GetAttribute("xformOp:transform").Get(0)).T
    np.testing.assert_allclose(usd[:3, 3], entity.keyframes[0].position)


def test_retrace_reads_archives_and_applies_trainer_normalization(tmp_path):
    remote = tmp_path / "reconstruction/remote_out"
    remote.mkdir(parents=True)
    pose = np.eye(4); pose[:3, 3] = [2, 3, 4]
    transform = np.eye(4); transform[:3, 3] = [1, 0, 0]
    with zipfile.ZipFile(remote / "processed_min.zip", "w") as archive:
        archive.writestr("transforms.json", json.dumps({"frames": [
            {"file_path": "images/a.jpg", "transform_matrix": pose.tolist()}]}))
    with zipfile.ZipFile(remote / "model.zip", "w") as archive:
        archive.writestr("train/dataparser_transforms.json", json.dumps({
            "transform": transform[:3].tolist(), "scale": 2}))
    prepared = prepare_retrace_transforms(tmp_path)
    result = json.loads(prepared.read_text())
    np.testing.assert_allclose(np.array(result["frames"][0]["transform_matrix"])[:3, 3], [6, 6, 8])
    assert not (remote / "transforms.json").exists()


def test_staging_preserves_explicit_path():
    from vaultwares_studio.pipeline import create_job_manifest, DigitalTwinStudioRunner, StageState
    manifest = create_job_manifest(source_video="sample.mov")
    job = Path(manifest.output_dir)
    write_cloud(job)
    save_active_camera(job, path_entity())
    for record in manifest.stages:
        if record.key in ("video_intake", "frame_extraction", "reconstruction"):
            record.state = StageState.COMPLETE.value
    runner = DigitalTwinStudioRunner(manifest, lambda msg: None)
    runner.run_stage("camera_staging")
    runner.run_stage("camera_staging")
    assert runner.stage_for("camera_staging").metadata["renderPath"] == "Custom"
