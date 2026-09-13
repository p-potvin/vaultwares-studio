"""DA3-Streaming -> nerfstudio conversion.

The two conventions guarded here were validated against a known-good
da3-standard run on the same footage (2026-08-03, 80 frames):

  * orientation error WITH the OpenCV->OpenGL flip:    9.07 deg median
  * orientation error WITHOUT it:                    178.91 deg median
  * position RMSE treating C2W correctly:              11.9% of extent
  * position RMSE wrongly inverting to W2C:            26.0% of extent

Both failure modes train without raising, which is exactly why they are pinned.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from vaultwares_studio.streaming_convert import (
    OPENCV_TO_OPENGL,
    load_streaming_poses,
    streaming_to_transforms,
    validate_poses,
)


def _write_stream_dir(tmp_path, poses: np.ndarray, intrinsics: np.ndarray):
    np.savetxt(tmp_path / "camera_poses.txt", poses.reshape(len(poses), 16))
    np.savetxt(tmp_path / "intrinsic.txt", intrinsics)
    return tmp_path


def _pose(rot: np.ndarray | None = None, trans=(0.0, 0.0, 0.0)) -> np.ndarray:
    M = np.eye(4)
    if rot is not None:
        M[:3, :3] = rot
    M[:3, 3] = trans
    return M


def _rot_z(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


# -- loading ------------------------------------------------------------------


def test_load_reads_poses_and_intrinsics(tmp_path):
    poses = np.stack([_pose(trans=(1, 2, 3)), _pose(_rot_z(30), (4, 5, 6))])
    intr = np.array([[100.0, 110.0, 250.0, 140.0], [100.0, 110.0, 250.0, 140.0]])
    _write_stream_dir(tmp_path, poses, intr)

    got_poses, got_intr = load_streaming_poses(tmp_path)
    assert got_poses.shape == (2, 4, 4)
    assert got_intr.shape == (2, 4)
    assert got_poses[0] == pytest.approx(poses[0])


def test_load_handles_a_single_frame(tmp_path):
    """np.loadtxt collapses one row to 1-D; that must not break reshaping."""
    _write_stream_dir(tmp_path, _pose(trans=(1, 2, 3))[None], np.array([[1.0, 2, 3, 4]]))
    poses, intr = load_streaming_poses(tmp_path)
    assert poses.shape == (1, 4, 4)
    assert intr.shape == (1, 4)


def test_load_rejects_pose_intrinsic_count_mismatch(tmp_path):
    np.savetxt(tmp_path / "camera_poses.txt", np.eye(4).reshape(1, 16))
    np.savetxt(tmp_path / "intrinsic.txt", np.array([[1.0, 2, 3, 4], [1, 2, 3, 4]]))
    with pytest.raises(ValueError, match="count mismatch"):
        load_streaming_poses(tmp_path)


def test_load_reports_a_missing_file_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="camera_poses.txt"):
        load_streaming_poses(tmp_path)


# -- pose validation ----------------------------------------------------------


def test_validate_rejects_a_reflection():
    """det(R) == -1 means an axis got mirrored — silent geometry corruption."""
    bad = _pose(np.diag([1.0, 1.0, -1.0]))
    with pytest.raises(ValueError, match="not proper"):
        validate_poses(bad[None])


def test_validate_rejects_a_malformed_bottom_row():
    bad = _pose()
    bad[3, :] = [1.0, 0.0, 0.0, 1.0]
    with pytest.raises(ValueError, match="bottom row"):
        validate_poses(bad[None])


def test_validate_rejects_non_finite():
    bad = _pose()
    bad[0, 3] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        validate_poses(bad[None])


def test_validate_accepts_proper_rotations():
    validate_poses(np.stack([_pose(_rot_z(d), (d, 0, 0)) for d in (0, 45, 90, 180)]))


# -- conversion ---------------------------------------------------------------


def test_translation_survives_the_axis_flip_untouched(tmp_path):
    """The flip negates two rotation columns; position must be unaffected.

    This is what makes the position and orientation checks independent.
    """
    poses = np.stack([_pose(_rot_z(35), (1.5, -2.5, 3.5))])
    _write_stream_dir(tmp_path, poses, np.array([[100.0, 100.0, 252.0, 140.0]]))

    t = streaming_to_transforms(
        tmp_path, ["a.jpg"], stream_size=(504, 280), original_size=(504, 280)
    )
    M = np.array(t["frames"][0]["transform_matrix"])
    assert M[:3, 3] == pytest.approx([1.5, -2.5, 3.5])


def test_flip_negates_exactly_the_y_and_z_basis_columns(tmp_path):
    poses = np.stack([_pose(_rot_z(20), (1, 2, 3))])
    _write_stream_dir(tmp_path, poses, np.array([[100.0, 100.0, 252.0, 140.0]]))

    t = streaming_to_transforms(
        tmp_path, ["a.jpg"], stream_size=(504, 280), original_size=(504, 280)
    )
    M = np.array(t["frames"][0]["transform_matrix"])
    assert M[:3, 0] == pytest.approx(poses[0][:3, 0])
    assert M[:3, 1] == pytest.approx(-poses[0][:3, 1])
    assert M[:3, 2] == pytest.approx(-poses[0][:3, 2])


def test_poses_are_not_inverted(tmp_path):
    """Streaming emits C2W. Inverting would be silent and wrong.

    With a pure translation, C2W position is +t and W2C position is -t, so the
    sign alone distinguishes the two.
    """
    poses = np.stack([_pose(trans=(5.0, 0.0, 0.0))])
    _write_stream_dir(tmp_path, poses, np.array([[100.0, 100.0, 252.0, 140.0]]))

    t = streaming_to_transforms(
        tmp_path, ["a.jpg"], stream_size=(504, 280), original_size=(504, 280)
    )
    assert np.array(t["frames"][0]["transform_matrix"])[0, 3] == pytest.approx(5.0)


def test_intrinsics_scale_from_stream_to_original_resolution(tmp_path):
    """Streaming sees 504x280; splatfacto loads the 1920x1080 originals."""
    _write_stream_dir(tmp_path, _pose()[None], np.array([[100.0, 110.0, 252.0, 140.0]]))

    t = streaming_to_transforms(
        tmp_path, ["a.jpg"], stream_size=(504, 280), original_size=(1920, 1080)
    )
    f = t["frames"][0]
    sx, sy = 1920 / 504, 1080 / 280
    assert f["fl_x"] == pytest.approx(100.0 * sx)
    assert f["fl_y"] == pytest.approx(110.0 * sy)
    assert f["cx"] == pytest.approx(252.0 * sx)
    assert f["cy"] == pytest.approx(140.0 * sy)
    # Principal point stays centred, which is the sanity check that matters.
    assert f["cx"] == pytest.approx(960.0)
    assert f["cy"] == pytest.approx(540.0)
    assert (f["w"], f["h"]) == (1920, 1080)


def test_frame_order_and_paths_follow_the_supplied_names(tmp_path):
    poses = np.stack([_pose(trans=(i, 0, 0)) for i in range(3)])
    intr = np.tile([100.0, 100.0, 252.0, 140.0], (3, 1))
    _write_stream_dir(tmp_path, poses, intr)

    names = ["frame_00001.jpg", "frame_00012.jpg", "frame_00023.jpg"]
    t = streaming_to_transforms(
        tmp_path, names, stream_size=(504, 280), original_size=(504, 280)
    )
    assert [f["file_path"] for f in t["frames"]] == [f"images/{n}" for n in names]
    assert [f["transform_matrix"][0][3] for f in t["frames"]] == [0, 1, 2]


def test_name_count_must_match_pose_count(tmp_path):
    poses = np.stack([_pose(), _pose()])
    _write_stream_dir(tmp_path, poses, np.tile([100.0, 100.0, 252.0, 140.0], (2, 1)))
    with pytest.raises(ValueError, match="one-to-one"):
        streaming_to_transforms(
            tmp_path, ["only_one.jpg"], stream_size=(504, 280), original_size=(504, 280)
        )


def test_payload_carries_the_ply_pointer_splatfacto_needs(tmp_path):
    _write_stream_dir(tmp_path, _pose()[None], np.array([[100.0, 100.0, 252.0, 140.0]]))
    t = streaming_to_transforms(
        tmp_path, ["a.jpg"], stream_size=(504, 280), original_size=(504, 280)
    )
    assert t["ply_file_path"] == "sparse_pc.ply"
    assert t["camera_angle_x"] > 0


def test_flip_matrix_is_its_own_inverse():
    """diag(1,-1,-1,1) squared is identity — applying it twice is a no-op."""
    assert OPENCV_TO_OPENGL @ OPENCV_TO_OPENGL == pytest.approx(np.eye(4))


# -- resolution inference -------------------------------------------------------


def test_infer_stream_size_prefers_the_retained_npz_over_the_fed_size(tmp_path):
    """DA3 works at 504x280 whatever it was fed; the npz image shape says so."""
    results = tmp_path / "results_output"
    results.mkdir()
    np.savez(results / "frame_0.npz", image=np.zeros((280, 504, 3), dtype=np.uint8))
    from vaultwares_studio.streaming_convert import infer_stream_size

    assert infer_stream_size(tmp_path, fallback=(672, 378)) == (504, 280)


def test_infer_stream_size_falls_back_to_the_principal_point(tmp_path):
    from vaultwares_studio.streaming_convert import infer_stream_size

    _write_stream_dir(tmp_path, _pose()[None], np.array([[430.0, 428.0, 252.0, 140.0]]))
    assert infer_stream_size(tmp_path, fallback=(672, 378)) == (504, 280)


def test_wrong_stream_size_is_refused(tmp_path):
    """The September bug: 504-frame intrinsics scaled as if they were 672 wide."""
    _write_stream_dir(tmp_path, _pose()[None], np.array([[430.0, 428.0, 252.0, 140.0]]))
    with pytest.raises(ValueError, match="principal point"):
        streaming_to_transforms(tmp_path, ["a.jpg"], stream_size=(672, 378), original_size=(1920, 1080))


def test_write_bundle_infers_size_and_centres_the_principal_point(tmp_path):
    from vaultwares_studio.streaming_convert import write_processed_bundle

    stream = tmp_path / "stream"
    stream.mkdir()
    _write_stream_dir(stream, _pose()[None], np.array([[430.0, 428.0, 252.0, 140.0]]))
    (stream / "pcd").mkdir()
    (stream / "pcd" / "combined_pcd.ply").write_text("ply\nformat ascii 1.0\nelement vertex 0\nend_header\n")
    transforms_path, _ = write_processed_bundle(
        stream, ["a.jpg"], tmp_path / "out", stream_size=(672, 378), original_size=(1920, 1080)
    )
    frame = json.loads(transforms_path.read_text())["frames"][0]
    assert frame["cx"] == pytest.approx(960.0) and frame["cy"] == pytest.approx(540.0)
    assert frame["fl_x"] == pytest.approx(430.0 * 1920 / 504)
