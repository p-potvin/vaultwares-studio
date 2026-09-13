"""camera_calibration: one shared camera per capture, lens included.

The regression these guard against is silent. DA3 hands splatfacto a different
focal length for every frame and no distortion at all; both produce a splat that
trains without complaint and is soft in the near field. Nothing throws.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from vaultwares_studio.camera_calibration import (
    CameraCalibration,
    consensus_intrinsics,
)

# Solved by COLMAP on backyard_134s_sunny.mp4, 491 of 500 frames registered
# (local-run-20260614-234541). The only measurement of this lens we have.
IPHONE_BACKYARD = CameraCalibration(
    fl_x=886.8492999462234,
    fl_y=885.5472971046771,
    cx=965.0856228089613,
    cy=541.7436589579706,
    w=1920,
    h=1080,
    k1=0.014107910239539938,
    k2=-0.014671162489701147,
    p1=-0.0001947506346839917,
    p2=-0.00043094833951948854,
    source="COLMAP local-run-20260614-234541",
)


def test_distortion_is_emitted_when_solved():
    fields = IPHONE_BACKYARD.as_transforms_fields()
    assert fields["k1"] == pytest.approx(0.0141079, rel=1e-5)
    assert fields["camera_model"] == "OPENCV"


def test_no_distortion_block_when_there_is_no_lens_measurement():
    """nerfstudio skips undistortion on an all-zero block, so writing one would
    only imply a calibration that does not exist."""
    plain = CameraCalibration(fl_x=890.0, fl_y=890.0, cx=960.0, cy=540.0, w=1920, h=1080)
    assert not plain.has_distortion
    assert set(plain.as_transforms_fields()) == {
        "camera_model", "fl_x", "fl_y", "cx", "cy", "w", "h"
    }


def test_scaling_moves_the_focal_but_never_the_distortion():
    """k1/k2/p1/p2 are in normalised image coordinates. Scaling them with the
    focal is the classic way to turn a correction into a new error."""
    half = IPHONE_BACKYARD.scaled_to(960, 540)
    assert half.fl_x == pytest.approx(IPHONE_BACKYARD.fl_x / 2)
    assert half.cx == pytest.approx(IPHONE_BACKYARD.cx / 2)
    assert half.k1 == IPHONE_BACKYARD.k1
    assert half.p2 == IPHONE_BACKYARD.p2


def test_round_trips_through_disk(tmp_path):
    path = IPHONE_BACKYARD.save(tmp_path / "lens.json")
    assert CameraCalibration.load(path) == IPHONE_BACKYARD


def test_reads_a_shared_camera_out_of_a_colmap_transforms():
    transforms = {
        "w": 1920, "h": 1080, "fl_x": 886.85, "fl_y": 885.55,
        "cx": 965.09, "cy": 541.74, "k1": 0.0141, "k2": -0.0147,
        "p1": -0.0002, "p2": -0.0004, "camera_model": "OPENCV",
        "frames": [],
    }
    lens = CameraCalibration.from_transforms(transforms, source="colmap")
    assert lens.k1 == pytest.approx(0.0141)
    assert lens.source == "colmap"


def test_per_frame_intrinsics_are_not_a_calibration():
    """A DA3 transforms.json has no top-level camera — refuse it loudly rather
    than silently calibrating off frame zero."""
    with pytest.raises(ValueError, match="no shared camera block"):
        CameraCalibration.from_transforms({"frames": [{"fl_x": 890.0}]})


def test_consensus_uses_the_median_not_the_mean():
    """One frame of mostly sky carries almost no focal signal. It should not be
    allowed to move the camera the other frames agree on."""
    intr = np.array([[500.0, 500.0, 252.0, 140.0]] * 9 + [[5000.0, 5000.0, 252.0, 140.0]])
    result = consensus_intrinsics(intr, (504, 280))
    assert result.fl_x == pytest.approx(500.0)
    assert result.frames == 10


def test_consensus_reports_what_the_disagreement_costs():
    """The measured September spread: 875.3..913.8 px over 500 frames."""
    intr = np.column_stack([
        np.linspace(875.26, 913.83, 500),
        np.linspace(879.99, 918.83, 500),
        np.full(500, 960.0),
        np.full(500, 540.0),
    ])
    result = consensus_intrinsics(intr, (1920, 1080))
    assert result.fl_x_spread == pytest.approx(0.0431, abs=1e-3)
    # Half the long edge times the worst relative focal error: ~20 px of
    # disagreement about where an edge pixel lands. That is the blur.
    assert result.max_edge_shift_px == pytest.approx(960 * 19.285 / 894.5, rel=0.02)
    assert "fl_x_spread_pct" in result.as_dict()


def test_consensus_rejects_the_wrong_shape():
    with pytest.raises(ValueError, match=r"\(N, 4\)"):
        consensus_intrinsics(np.zeros((5, 3)), (504, 280))
    with pytest.raises(ValueError, match="no intrinsics"):
        consensus_intrinsics(np.zeros((0, 4)), (504, 280))


def test_the_lens_calibration_ships_as_plain_json(tmp_path):
    """The worker container reads this file without vaultwares_studio on the
    path, so it has to be ordinary JSON with no custom types."""
    path = IPHONE_BACKYARD.save(tmp_path / "lens.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["k1"] and data["source"]
    assert all(isinstance(v, (int, float, str)) for v in data.values())
