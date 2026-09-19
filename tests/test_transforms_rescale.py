"""Intrinsics follow the training frames, not the video that was posed.

The console poses whatever video it is handed and records that size. Training
uses local full-resolution frames. When the two differ — as they did on 17 Sep,
where a failed 1 GB upload meant the merged capture was posed at 960x540 and
trained from 1920x1080 — nothing downstream complains: nerfstudio believes the
declared intrinsics, the splat trains, and every gaussian lands wrong.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from prepare_zerogpu_training import rescale_transforms_to_frames  # noqa: E402


def _frame(tmp_path: Path, width: int, height: int) -> Path:
    path = tmp_path / f"frame_{width}x{height}.jpg"
    Image.new("RGB", (width, height), (40, 60, 80)).save(path)
    return path


def _bundle(**over) -> dict:
    base = {"camera_model": "OPENCV", "fl_x": 456.477, "fl_y": 455.954,
            "cx": 480.0, "cy": 270.0, "w": 960, "h": 540,
            "frames": [{"file_path": "images/frame_00000.jpg"}]}
    base.update(over)
    return base


def test_the_merged_capture_case_doubles_the_focal(tmp_path):
    """960x540 bundle, 1920x1080 frames: every intrinsic scales, nothing else."""
    out = rescale_transforms_to_frames(_bundle(), _frame(tmp_path, 1920, 1080), log=lambda *_: None)
    assert (out["w"], out["h"]) == (1920, 1080)
    assert out["fl_x"] == pytest.approx(912.954)
    assert out["fl_y"] == pytest.approx(911.908)
    assert (out["cx"], out["cy"]) == (960.0, 540.0)
    assert out["frames"][0]["file_path"] == "images/frame_00000.jpg"


def test_a_matching_resolution_is_left_alone(tmp_path):
    before = _bundle(fl_x=906.118, fl_y=908.851, cx=960.0, cy=540.0, w=1920, h=1080)
    after = rescale_transforms_to_frames(json.loads(json.dumps(before)),
                                         _frame(tmp_path, 1920, 1080), log=lambda *_: None)
    assert after == before


def test_distortion_is_resolution_independent(tmp_path):
    """k1/k2/p1/p2 are in normalised coordinates. Scaling them with the focal
    is the classic way to turn a correction into a new error."""
    out = rescale_transforms_to_frames(
        _bundle(k1=0.0141079, k2=-0.0146712, p1=-0.00019475, p2=-0.00043095),
        _frame(tmp_path, 1920, 1080), log=lambda *_: None)
    assert out["k1"] == pytest.approx(0.0141079)
    assert out["k2"] == pytest.approx(-0.0146712)
    assert out["p1"] == pytest.approx(-0.00019475)
    assert out["p2"] == pytest.approx(-0.00043095)


def test_per_frame_intrinsics_are_rescaled_too(tmp_path):
    """The pre-17-Sep bundles carry a camera per frame instead of one on top."""
    bundle = {"camera_angle_x": 1.65, "frames": [
        {"file_path": "images/frame_00000.jpg", "fl_x": 440.95, "fl_y": 441.2,
         "cx": 480.0, "cy": 270.0, "w": 960, "h": 540},
        {"file_path": "images/frame_00001.jpg", "fl_x": 441.50, "fl_y": 441.8,
         "cx": 480.0, "cy": 270.0, "w": 960, "h": 540},
    ]}
    out = rescale_transforms_to_frames(bundle, _frame(tmp_path, 1920, 1080), log=lambda *_: None)
    assert [f["fl_x"] for f in out["frames"]] == pytest.approx([881.90, 883.00], abs=0.01)
    assert all(f["w"] == 1920 and f["h"] == 1080 for f in out["frames"])


def test_a_crop_is_refused_rather_than_guessed(tmp_path):
    """A different aspect ratio moves the principal point unrecoverably."""
    with pytest.raises(SystemExit, match="crop"):
        rescale_transforms_to_frames(_bundle(), _frame(tmp_path, 1440, 1080), log=lambda *_: None)


def test_a_bundle_without_a_declared_size_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="no image size"):
        rescale_transforms_to_frames({"frames": [{"file_path": "a.jpg"}]},
                                     _frame(tmp_path, 1920, 1080), log=lambda *_: None)
