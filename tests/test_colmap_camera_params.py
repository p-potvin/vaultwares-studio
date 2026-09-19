"""COLMAP must be told the lens, or it verifies every pair as uncalibrated.

Measured on two runs of the same scene. The 14 Sep cpu-upgrade job passed no
`ImageReader.camera_params`, so COLMAP invented focal = 1.2 x 1920 = 2304 px
against ~890 measured for this phone, set `prior_focal_length = 0`, and fell
back to the fundamental matrix:

    config 3 UNCALIBRATED  82.3%    config 2 CALIBRATED  0.0%
    median matches/pair       36    verified pairs       46%

The same scene run locally with params (fx 948, prior_focal 1):

    config 2 CALIBRATED    85.1%    median matches/pair   754
                                    verified pairs        83%
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ENTRYPOINT = Path(__file__).resolve().parents[1] / "docker" / "worker" / "recon_entrypoint.py"


def _load():
    """Import the entrypoint without running it; it is a script, not a package."""
    spec = importlib.util.spec_from_file_location("recon_entrypoint_probe", ENTRYPOINT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except SystemExit:  # pragma: no cover - the script guards its own main
        pass
    return module


CALIBRATION = {
    "fl_x": 886.8492999462234, "fl_y": 885.5472971046771,
    "cx": 965.0856228089613, "cy": 541.7436589579706,
    "w": 1920, "h": 1080,
    "k1": 0.014107910239539938, "k2": -0.014671162489701147,
    "p1": -0.0001947506346839917, "p2": -0.00043094833951948854,
}


def _images(tmp_path: Path, size=(1920, 1080)) -> Path:
    from PIL import Image

    d = tmp_path / "images"
    d.mkdir(exist_ok=True)
    Image.new("RGB", size, (30, 60, 90)).save(d / "frame_00000.jpg")
    return d


def _calibration_file(tmp_path: Path, **over) -> Path:
    path = tmp_path / "calib.json"
    path.write_text(json.dumps({**CALIBRATION, **over}), encoding="utf-8")
    return path


def test_the_real_calibration_becomes_eight_opencv_params(tmp_path):
    module = _load()
    out = module.colmap_camera_params(_calibration_file(tmp_path), _images(tmp_path))
    values = [float(v) for v in out.split(",")]
    assert len(values) == 8
    assert values[0] == pytest.approx(886.849, abs=0.01)
    assert values[2] == pytest.approx(965.086, abs=0.01)
    assert values[4] == pytest.approx(0.0141079, abs=1e-6)
    # The number that matters: nothing near COLMAP's 1.2 x width guess.
    assert values[0] / 1920 == pytest.approx(0.462, abs=0.01)


def test_params_scale_to_the_images_colmap_will_read(tmp_path):
    """A calibration is only valid at the resolution it was measured at."""
    module = _load()
    out = module.colmap_camera_params(_calibration_file(tmp_path), _images(tmp_path, (960, 540)))
    values = [float(v) for v in out.split(",")]
    assert values[0] == pytest.approx(886.849 / 2, abs=0.01)
    assert values[2] == pytest.approx(965.086 / 2, abs=0.01)
    # Distortion is in normalised coordinates and must NOT scale.
    assert values[4] == pytest.approx(0.0141079, abs=1e-6)


def test_a_different_aspect_ratio_is_refused_rather_than_guessed(tmp_path):
    module = _load()
    assert module.colmap_camera_params(_calibration_file(tmp_path), _images(tmp_path, (1440, 1080))) == ""


def test_missing_or_broken_calibration_returns_empty_not_an_error(tmp_path):
    module = _load()
    assert module.colmap_camera_params(None, _images(tmp_path)) == ""
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert module.colmap_camera_params(bad, _images(tmp_path)) == ""
    assert module.colmap_camera_params(_calibration_file(tmp_path, fl_x=0.0), _images(tmp_path)) == ""


def test_the_extractor_passes_params_and_caps_features():
    """Guard the call site, not just the helper."""
    source = ENTRYPOINT.read_text(encoding="utf-8")
    assert "--ImageReader.camera_params" in source
    assert "--SiftExtraction.max_num_features" in source
    assert "UNCALIBRATED" in source, "the warning explains what silence costs"
