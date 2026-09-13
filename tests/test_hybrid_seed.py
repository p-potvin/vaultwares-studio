"""hybrid_seed: COLMAP's poses reconciling DA3's depth.

Every test here builds a synthetic scene with a known answer, because the whole
module exists to recover a number (the per-frame depth scale) that has no ground
truth on real footage. If the maths is right it comes back exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from vaultwares_studio.hybrid_seed import (
    MIN_CORRESPONDENCES,
    OPENGL_TO_OPENCV,
    align_depth_to_sparse,
    backproject,
    build_hybrid_seed,
    opengl_c2w_to_opencv_w2c,
    project,
    voxel_downsample,
)

K = np.array([[300.0, 0.0, 252.0], [0.0, 300.0, 140.0], [0.0, 0.0, 1.0]])
SIZE = (504, 280)


def _c2w(position=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Camera at ``position`` looking down world -Z, OpenGL convention."""
    pose = np.eye(4)
    pose[:3, 3] = position
    return pose


def _plane_depth(distance: float, shape=(280, 504)) -> np.ndarray:
    """A fronto-parallel wall. Every pixel reads the same depth, so this scene
    CANNOT separate scale from shift — any (a, b) with ``a*d + b = truth``
    explains it. Used only where a single depth is the point."""
    return np.full(shape, distance, dtype=np.float64)


def _ramp_depth(near: float, far: float, shape=(280, 504)) -> np.ndarray:
    """A slanted surface: depth varies across the frame, so scale and shift are
    independently determined. This is what a real scene looks like."""
    row = np.linspace(near, far, shape[1])
    return np.broadcast_to(row, shape).astype(np.float64).copy()


def _sparse_from_depth(
    depth: np.ndarray, n: int = 400, seed: int = 0, c2w: np.ndarray | None = None
) -> np.ndarray:
    """World points sampled off a TRUE depth map — COLMAP's sparse cloud, in
    effect: a few hundred pixels whose real depth is known."""
    rng = np.random.default_rng(seed)
    height, width = depth.shape[:2]
    rows = rng.integers(0, height, n)
    columns = rng.integers(0, width, n)
    z = depth[rows, columns]
    x = (columns - K[0, 2]) * z / K[0, 0]
    y = (rows - K[1, 2]) * z / K[1, 1]
    cam = np.column_stack([x, y, z])
    pose = (c2w if c2w is not None else _c2w()) @ OPENGL_TO_OPENCV
    return cam @ pose[:3, :3].T + pose[:3, 3]


def test_pose_conversion_round_trips():
    c2w = _c2w((1.0, 2.0, 3.0))
    w2c = opengl_c2w_to_opencv_w2c(c2w)
    assert np.allclose(np.linalg.inv(w2c), c2w @ OPENGL_TO_OPENCV)


def test_pose_conversion_rejects_the_wrong_shape():
    with pytest.raises(ValueError, match="4x4"):
        opengl_c2w_to_opencv_w2c(np.eye(3))


def test_a_point_straight_ahead_lands_on_the_principal_point():
    uv, depth, inside = project(np.array([[0.0, 0.0, -5.0]]), K, _c2w(), SIZE)
    assert inside[0]
    assert uv[0] == pytest.approx([252.0, 140.0])
    assert depth[0] == pytest.approx(5.0)


def test_points_behind_the_camera_are_not_inside():
    """OpenGL looks down -Z, so +Z world is behind. Getting this backwards
    silently halves the correspondences and biases every scale."""
    _, depth, inside = project(np.array([[0.0, 0.0, 5.0]]), K, _c2w(), SIZE)
    assert depth[0] < 0
    assert not inside[0]


def test_recovers_a_known_scale():
    """The core claim: DA3 predicts depth in its own units, COLMAP's points say
    what those units are worth, and the fit finds the ratio."""
    truth = _ramp_depth(4.0, 12.0)
    alignment = align_depth_to_sparse(truth * 0.25, K, _c2w(), _sparse_from_depth(truth))
    assert alignment.ok
    assert alignment.scale == pytest.approx(4.0, rel=1e-6)
    assert alignment.shift == pytest.approx(0.0, abs=1e-6)
    assert alignment.rmse < 1e-6


def test_recovers_a_known_scale_and_shift():
    """A relative predictor: truth = 0.25 * predicted + 2.0. Needs a depth
    gradient to be solvable at all — see _plane_depth."""
    truth = _ramp_depth(4.0, 12.0)
    alignment = align_depth_to_sparse(
        (truth - 2.0) / 0.25, K, _c2w(), _sparse_from_depth(truth)
    )
    assert alignment.scale == pytest.approx(0.25, rel=1e-6)
    assert alignment.shift == pytest.approx(2.0, abs=1e-6)
    assert alignment.shift_matters


def test_occluded_correspondences_do_not_drag_the_scale():
    """A sparse point behind a wall projects into the frame but reads too far.
    The tail is one-sided, which is exactly what a plain least squares cannot
    survive and what the Huber reweighting is for."""
    truth = _ramp_depth(4.0, 12.0)
    sparse = _sparse_from_depth(truth, n=500)
    # A fifth of them sit four times further along their own ray — behind the
    # surface the depth map describes.
    sparse[::5] *= 4.0
    alignment = align_depth_to_sparse(truth / 2, K, _c2w(), sparse)
    assert alignment.scale == pytest.approx(2.0, rel=0.02)
    assert alignment.inliers < alignment.correspondences


def test_too_few_correspondences_is_reported_not_guessed():
    """A frame of sky gets no fit. Fusing it with a guessed scale would put a
    whole surface in the wrong place, and splatfacto would fit it."""
    truth = _ramp_depth(4.0, 12.0)
    sparse = _sparse_from_depth(truth, n=MIN_CORRESPONDENCES - 1)
    alignment = align_depth_to_sparse(truth, K, _c2w(), sparse)
    assert not alignment.ok
    assert alignment.correspondences < MIN_CORRESPONDENCES


def test_no_visible_points_at_all_is_not_a_crash():
    behind = np.array([[0.0, 0.0, 10.0]] * 50, dtype=float)
    alignment = align_depth_to_sparse(_plane_depth(5.0), K, _c2w(), behind)
    assert not alignment.ok and alignment.correspondences == 0


def test_shift_does_not_matter_for_a_metric_predictor():
    truth = _ramp_depth(4.0, 12.0)
    metric = align_depth_to_sparse(truth / 4, K, _c2w(), _sparse_from_depth(truth))
    assert not metric.shift_matters  # a pure scale already explains it


def test_backprojection_inverts_projection():
    """Round trip: depth map to world and back to the plane it came from."""
    distance = 7.0
    world, rgb = backproject(_plane_depth(distance), K, _c2w(), stride=8)
    assert len(world) == len(rgb)
    assert np.allclose(world[:, 2], -distance)


def test_backprojection_honours_the_camera_position():
    world, _ = backproject(_plane_depth(3.0), K, _c2w((10.0, 0.0, 0.0)), stride=16)
    assert np.allclose(world[:, 2], -3.0)
    assert world[:, 0].mean() == pytest.approx(10.0, abs=0.5)


def test_confidence_filter_keeps_the_requested_fraction():
    depth = _plane_depth(5.0, shape=(100, 100))
    confidence = np.linspace(0, 1, 100 * 100).reshape(100, 100)
    world, _ = backproject(
        depth, K, _c2w(), confidence=confidence, keep_quantile=0.25, stride=1
    )
    assert len(world) == pytest.approx(0.25 * 100 * 100, rel=0.02)


def test_voxel_downsample_collapses_duplicates():
    points = np.repeat(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), 50, axis=0)
    colors = np.zeros((100, 3), dtype=np.uint8)
    kept, kept_colors = voxel_downsample(points, colors, voxel=0.1)
    assert len(kept) == 2 and len(kept_colors) == 2


def test_voxel_downsample_is_a_no_op_at_zero():
    points = np.random.default_rng(0).normal(size=(20, 3))
    kept, _ = voxel_downsample(points, np.zeros((20, 3), np.uint8), voxel=0.0)
    assert len(kept) == 20


def test_fusion_drops_the_frames_it_cannot_align():
    truth = _ramp_depth(4.0, 12.0)
    sparse = _sparse_from_depth(truth)
    good = {"depth": truth / 3, "intrinsics": K, "c2w": _c2w()}
    # Sky: the camera points away, so nothing projects into it.
    blind = {"depth": _plane_depth(1.0), "intrinsics": K, "c2w": _c2w((0.0, 500.0, 0.0))}

    points, colors, report, alignments = build_hybrid_seed(
        [good, blind, good], sparse, stride=8
    )
    assert report.frames_total == 3
    assert report.frames_aligned == 2
    assert len(points) == len(colors) > 0
    assert alignments[1].correspondences == 0
    assert report.median_scale == pytest.approx(3.0, rel=1e-3)


def test_fusion_reports_scale_spread():
    """The headline diagnostic. If DA3's depth were consistently metric against
    COLMAP this would be ~0; a wide spread says the per-frame scale drifts, and
    that is the streaming pipeline's suspected global-scale problem showing up
    in a number we can measure without a GPU."""
    truth = _ramp_depth(4.0, 12.0)
    sparse = _sparse_from_depth(truth)
    frames = [
        {"depth": truth / factor, "intrinsics": K, "c2w": _c2w()}
        for factor in (2.0, 3.0, 4.0)
    ]
    _, _, report, _ = build_hybrid_seed(frames, sparse, stride=16)
    assert report.median_scale == pytest.approx(3.0, rel=1e-3)
    assert report.scale_spread == pytest.approx((4.0 - 2.0) / 3.0, rel=1e-2)


def test_fusion_with_nothing_alignable_returns_empty_not_garbage():
    behind = np.array([[0.0, 0.0, 10.0]] * 50, dtype=float)
    frames = [{"depth": _plane_depth(5.0), "intrinsics": K, "c2w": _c2w()}]
    points, colors, report, _ = build_hybrid_seed(frames, behind)
    assert len(points) == 0 and len(colors) == 0
    assert report.frames_aligned == 0
    assert report.as_dict()["frames_aligned"] == 0
