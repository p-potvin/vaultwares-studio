import json
from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from vaultwares_studio.gravity_align import (
    align_cloud,
    compute_alignment,
    quaternion_multiply,
    rotate_gaussian_splat,
    rotation_to_quaternion,
)
from vaultwares_studio.splat_io import GaussianSplat, read_gaussian_ply, write_gaussian_ply


# A wide-thin-tall cloud whose vertical axis is +Z, plus a sparse "sky" tail
# in +Z so the skewness sign check picks the right direction.
def _z_up_cloud(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ground = rng.normal(loc=[0, 0, 0], scale=[1.5, 1.5, 0.1], size=(2000, 3))
    sky_tail = rng.normal(loc=[0, 0, 2.0], scale=[0.8, 0.8, 0.4], size=(120, 3))
    return np.concatenate([ground, sky_tail], axis=0)


def test_compute_alignment_picks_z_as_up_and_rotates_to_y():
    points = _z_up_cloud()
    result = compute_alignment(points)
    rotated = points @ result.rotation.T
    # After rotation the smallest spread should be along Y.
    spread = rotated.max(axis=0) - rotated.min(axis=0)
    assert np.argmin(spread) == 1
    # Original up vector was near +Z (or -Z, then flipped to +Z).
    assert abs(result.up_before[2]) > 0.95
    assert result.up_before[2] > 0  # skew-flip should leave it pointing to sky


def test_rotation_is_proper_orthonormal():
    points = _z_up_cloud(seed=42)
    R = compute_alignment(points).rotation
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-8)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-8)


def test_flip_path_when_cloud_is_upside_down():
    points = _z_up_cloud()
    # Mirror Z so the sky tail now points to -Z; PCA's first eigenvector is the
    # same line — but the skew test should flip the sign.
    flipped_points = points.copy()
    flipped_points[:, 2] *= -1
    result = compute_alignment(flipped_points)
    assert result.flipped is True
    rotated = flipped_points @ result.rotation.T
    # The sky tail (originally large -Z) should now end up in +Y.
    sky = rotated[-120:]
    assert sky[:, 1].mean() > 0


def test_quaternion_round_trip_on_identity():
    quat = rotation_to_quaternion(np.eye(3))
    np.testing.assert_allclose(quat, [1, 0, 0, 0], atol=1e-9)


def test_quaternion_multiply_matches_matrix_product():
    points = _z_up_cloud()
    R = compute_alignment(points).rotation
    splat = _make_splat()
    rotated = rotate_gaussian_splat(splat, R)
    # Rotating the canonical +X axis through both the per-splat quaternion and
    # the composed quaternion should equal applying R then the per-splat rot.
    base_quat = splat.rotations[0].astype(np.float64)
    base_quat = base_quat / np.linalg.norm(base_quat)
    scene_quat = rotation_to_quaternion(R)
    expected = quaternion_multiply(scene_quat, base_quat)
    composed = rotated.rotations[0].astype(np.float64)
    composed = composed / np.linalg.norm(composed)
    # Quaternion equality up to global sign.
    assert np.allclose(composed, expected, atol=1e-6) or np.allclose(composed, -expected, atol=1e-6)


def test_align_cloud_rewrites_files_and_updates_summary(tmp_path):
    splat_path = tmp_path / "cloud.ply"
    preview_path = tmp_path / "cloud_preview.ply"
    summary_path = tmp_path / "summary.json"

    splat = _make_splat()
    write_gaussian_ply(splat, splat_path)
    _write_point_ply(preview_path, _z_up_cloud())

    result = align_cloud(splat_path, preview_path, summary_path=summary_path)
    assert result is not None

    # Preview now has Y as smallest-spread axis.
    rotated_preview_points = _read_points(preview_path)
    spread = rotated_preview_points.max(axis=0) - rotated_preview_points.min(axis=0)
    assert np.argmin(spread) == 1

    # Summary records the rotation and flips the gate flag.
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["gravity_aligned"] is True
    assert summary["alignment"]["angle_from_y_degrees"] > 80.0

    # Re-running is idempotent and returns None.
    assert align_cloud(splat_path, preview_path, summary_path=summary_path) is None


def test_align_cloud_preserves_splat_attribute_columns(tmp_path):
    splat_path = tmp_path / "cloud.ply"
    preview_path = tmp_path / "cloud_preview.ply"
    splat = _make_splat()
    write_gaussian_ply(splat, splat_path)
    _write_point_ply(preview_path, _z_up_cloud(seed=7))

    align_cloud(splat_path, preview_path)
    rewritten = read_gaussian_ply(splat_path)
    assert rewritten.positions.shape == splat.positions.shape
    assert rewritten.sh_rest is not None
    assert rewritten.sh_rest.shape == splat.sh_rest.shape
    # SH coefficients are deliberately left untouched (see module docstring).
    np.testing.assert_allclose(rewritten.sh_rest, splat.sh_rest)
    np.testing.assert_allclose(rewritten.sh0, splat.sh0)
    np.testing.assert_allclose(rewritten.opacity, splat.opacity)
    np.testing.assert_allclose(rewritten.scales, splat.scales)


def test_align_cloud_rotates_captured_cameras(tmp_path):
    splat_path = tmp_path / "cloud.ply"
    preview_path = tmp_path / "cloud_preview.ply"
    captured_path = tmp_path / "captured_cameras.json"
    summary_path = tmp_path / "summary.json"

    write_gaussian_ply(_make_splat(), splat_path)
    _write_point_ply(preview_path, _z_up_cloud(seed=11))
    captured_path.write_text(
        json.dumps(
            [
                {"name": "Cap 1", "position": [1.0, 2.0, 3.0], "lookAt": [0.0, 0.0, 0.0], "up": [0, 1, 0]},
                {"name": "Cap 2", "position": [0.5, 0.5, 0.5], "lookAt": [0.0, 0.0, 0.0]},
            ]
        )
    )

    result = align_cloud(
        splat_path,
        preview_path,
        summary_path=summary_path,
        captured_cameras_path=captured_path,
    )
    assert result is not None

    rotated = json.loads(captured_path.read_text(encoding="utf-8"))
    # Position vectors round-trip through R: applying R to (1,2,3) should match
    # what align_cloud wrote, with no mutations to the name field.
    expected = (result.rotation @ np.array([1.0, 2.0, 3.0])).tolist()
    np.testing.assert_allclose(rotated[0]["position"], expected, atol=1e-6)
    assert rotated[0]["name"] == "Cap 1"
    # Up vector also rotated when present.
    np.testing.assert_allclose(
        rotated[0]["up"], (result.rotation @ np.array([0.0, 1.0, 0.0])).tolist(), atol=1e-6
    )
    # Entries without an "up" key stay valid (no crash, no spurious key added).
    assert "up" not in rotated[1]


def _make_splat(count: int = 16) -> GaussianSplat:
    rng = np.random.default_rng(123)
    return GaussianSplat(
        positions=rng.normal(size=(count, 3)).astype(np.float32),
        sh0=rng.normal(size=(count, 3)).astype(np.float32),
        opacity=rng.normal(size=count).astype(np.float32),
        scales=rng.normal(size=(count, 3)).astype(np.float32),
        rotations=rng.normal(size=(count, 4)).astype(np.float32),
        sh_rest=rng.normal(size=(count, 45)).astype(np.float32),
    )


def _write_point_ply(path: Path, points: np.ndarray) -> None:
    rows = np.zeros(points.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    rows["x"] = points[:, 0]
    rows["y"] = points[:, 1]
    rows["z"] = points[:, 2]
    PlyData([PlyElement.describe(rows, "vertex")], text=False).write(str(path))


def _read_points(path: Path) -> np.ndarray:
    vertex = PlyData.read(str(path))["vertex"]
    return np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)
