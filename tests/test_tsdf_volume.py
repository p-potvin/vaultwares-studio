"""tsdf_volume: does the fused volume put the surface where the surface is?

Synthetic depth maps of a known plane, so the answer is checkable rather than
plausible. The property that matters is the zero crossing: a TSDF whose values
are scaled or offset still renders, and its isosurface sits somewhere else.
"""

from __future__ import annotations

import numpy as np
import pytest

from vaultwares_studio.tsdf_volume import (
    PACK_BIAS,
    integrate,
    pack_keys,
    unpack_keys,
)

INTRINSICS = np.array([[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]])


def plane_frame(distance: float, c2w: np.ndarray | None = None, shape=(48, 64)) -> dict:
    """A fronto-parallel wall ``distance`` ahead of the camera.

    OpenCV axes, +Z forward — the convention ``integrate`` expects and the one
    DA3-Streaming's poses are in. With an identity pose the wall therefore sits
    at world ``z = +distance``, and nearer the camera means *smaller* z.

    Depth is stored along the ray, not along the optical axis, matching what a
    depth sensor and DA3 both report.
    """
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]]
    x = (cols - INTRINSICS[0, 2]) / INTRINSICS[0, 0]
    y = (rows - INTRINSICS[1, 2]) / INTRINSICS[1, 1]
    depth = np.full(shape, distance, dtype=np.float32)
    return {
        "depth": depth,
        "conf": None,
        "intrinsics": INTRINSICS,
        "c2w": np.eye(4) if c2w is None else c2w,
    }


def test_packing_round_trips_including_negatives():
    ijk = np.array([[0, 0, 0], [-5, 7, -900], [1000, -1, 3]], dtype=np.int32)
    assert np.array_equal(unpack_keys(pack_keys(ijk)), ijk)


def test_packing_refuses_coordinates_it_cannot_represent():
    with pytest.raises(ValueError, match="packing range"):
        pack_keys(np.array([[PACK_BIAS + 1, 0, 0]], dtype=np.int64))


def test_the_zero_crossing_lands_on_the_surface():
    """The whole point of a TSDF. Voxels in front of the wall must be positive,
    behind it negative, and the sign change must happen at the wall."""
    distance, voxel = 1.0, 0.02
    volume = integrate([plane_frame(distance)], voxel_size=voxel,
                       sdf_trunc=voxel * 4, pixel_stride=1)
    ijk, values = volume.to_arrays(min_weight=1.0)

    # Take the column straight ahead of the camera, where the ray is the axis.
    on_axis = (np.abs(ijk[:, 0]) <= 0) & (np.abs(ijk[:, 1]) <= 0)
    assert on_axis.sum() >= 3
    z = ijk[on_axis, 2] * voxel
    v = values[on_axis]
    order = np.argsort(z)
    z, v = z[order], v[order]

    # +Z forward, so the wall is at z = +distance. Nearer the camera is SMALLER
    # z and is outside the surface, which reads positive.
    assert v[np.argmin(z)] > 0
    assert v[np.argmax(z)] < 0
    # np.interp needs its x increasing; values run positive -> negative as z
    # grows, so both are reversed.
    crossing = np.interp(0.0, v[::-1], z[::-1])
    assert crossing == pytest.approx(distance, abs=1.5 * voxel)


def test_values_are_world_units_not_normalised():
    """A level set holds metres from the surface. Storing signed/sdf_trunc also
    renders, with the isosurface in the wrong place by that factor."""
    voxel = 0.02
    volume = integrate([plane_frame(1.0)], voxel_size=voxel, sdf_trunc=voxel * 4,
                       pixel_stride=1)
    _, values = volume.to_arrays(min_weight=1.0)
    assert np.abs(values).max() == pytest.approx(voxel * 4, rel=1e-3)
    assert np.abs(values).max() < 0.5  # normalised would top out at exactly 1.0


def test_the_band_is_as_wide_as_the_truncation():
    """sdf_trunc decides how many voxels deep the shell is; a band of one voxel
    leaves holes wherever sampling is sparser than the grid."""
    voxel = 0.02
    narrow = integrate([plane_frame(1.0)], voxel_size=voxel, sdf_trunc=voxel,
                       pixel_stride=1)
    wide = integrate([plane_frame(1.0)], voxel_size=voxel, sdf_trunc=voxel * 4,
                     pixel_stride=1)
    assert wide.active > narrow.active


def test_repeat_observations_average_rather_than_accumulate():
    """Weights exist so a voxel seen twenty times reads the same distance as one
    seen once, not twenty times further out."""
    voxel = 0.02
    once = integrate([plane_frame(1.0)], voxel_size=voxel, pixel_stride=1)
    many = integrate([plane_frame(1.0) for _ in range(20)], voxel_size=voxel,
                     pixel_stride=1)
    ijk_a, values_a = once.to_arrays(min_weight=1.0)
    ijk_b, values_b = many.to_arrays(min_weight=1.0)
    assert len(ijk_a) == len(ijk_b)
    order_a, order_b = np.lexsort(ijk_a.T), np.lexsort(ijk_b.T)
    assert np.allclose(values_a[order_a], values_b[order_b], atol=1e-6)
    assert many.weights.max() == pytest.approx(20 * once.weights.max())


def test_min_weight_drops_the_thinly_seen():
    volume = integrate([plane_frame(1.0)] * 3, voxel_size=0.02, pixel_stride=1)
    everything, _ = volume.to_arrays(min_weight=1.0)
    confident, _ = volume.to_arrays(min_weight=3.0)
    assert len(confident) <= len(everything)
    assert len(confident) > 0


def test_a_translated_camera_moves_the_surface_with_it():
    """Poses are applied, not ignored. A grid built from a shifted camera that
    lands in the same place means c2w never made it into the maths."""
    voxel = 0.02
    shifted = np.eye(4)
    shifted[:3, 3] = (1.0, 0.0, 0.0)
    a = integrate([plane_frame(1.0)], voxel_size=voxel, pixel_stride=1)
    b = integrate([plane_frame(1.0, c2w=shifted)], voxel_size=voxel, pixel_stride=1)
    mean_a = unpack_keys(a.keys).mean(axis=0) * voxel
    mean_b = unpack_keys(b.keys).mean(axis=0) * voxel
    assert (mean_b - mean_a)[0] == pytest.approx(1.0, abs=2 * voxel)


def test_confidence_floor_discards_samples():
    frame = plane_frame(1.0)
    frame["conf"] = np.zeros_like(frame["depth"])
    volume = integrate([frame], voxel_size=0.02, pixel_stride=1, conf_floor=0.5)
    assert volume.active == 0


def test_depth_truncation_discards_far_samples():
    volume = integrate([plane_frame(5.0)], voxel_size=0.02, pixel_stride=1,
                       depth_trunc=1.0)
    assert volume.active == 0


def test_pixel_stride_barely_changes_the_voxel_set():
    """The benchmark's headline: at a grid coarser than the pixel sampling,
    stride 2 costs a few percent of the voxels and saves most of the time."""
    dense = integrate([plane_frame(1.0)], voxel_size=0.05, pixel_stride=1)
    sparse = integrate([plane_frame(1.0)], voxel_size=0.05, pixel_stride=2)
    assert sparse.active > 0.8 * dense.active
