import json
import zipfile
from pathlib import Path

from tools.import_zerogpu_artifact import import_artifact


def test_import_zerogpu_artifact_builds_job_contract(tmp_path, monkeypatch):
    archive = tmp_path / "artifact.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("input_manifest.json", json.dumps({"source_name": "sample.MOV", "selected_frames": 500, "gpu_input_size": [672, 378], "preset": {"key": "high"}}))
        z.writestr("streaming/camera_poses.txt", ("1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\n") * 500)
        z.writestr("streaming/intrinsic.txt", "1 1 0 0\n" * 500)
        z.writestr("streaming/loop_closures.txt", "# pairs\n")
        z.writestr("streaming/config.json", "{}")
        z.writestr("streaming/pcd/combined_pcd.ply", "ply\nformat ascii 1.0\nelement vertex 0\nend_header\n")
    import tools.import_zerogpu_artifact as module
    monkeypatch.setattr(module, "JOBS_ROOT", tmp_path / "jobs")
    result = import_artifact(archive, "zerogpu-test", require_d=False)
    assert (result / "manifest.json").exists()
    assert (result / "reconstruction/transforms.json").exists()
    assert json.loads((result / "manifest.json").read_text())["metadata"]["imported"] is True


def test_job_listing_deduplicates_external_and_local_manifest():
    from vaultwares_studio.pipeline import list_job_manifests
    matches = [path for path in list_job_manifests() if path.parent.name == "zerogpu-img1274-loop-on-v2"]
    assert len(matches) == 1
