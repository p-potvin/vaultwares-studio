import numpy as np
import pytest

from vaultwares_studio.splat_io import GaussianSplat, write_gaussian_ply
from vaultwares_studio.splat_packed import pack_splat, ply_to_splat, write_splat_format

_ROW_BYTES = 32


def _make_splat(count: int = 24, seed: int = 0) -> GaussianSplat:
    rng = np.random.default_rng(seed)
    return GaussianSplat(
        positions=rng.normal(scale=2.0, size=(count, 3)).astype(np.float32),
        sh0=rng.normal(size=(count, 3)).astype(np.float32),
        opacity=rng.normal(size=count).astype(np.float32),
        scales=rng.normal(scale=0.5, size=(count, 3)).astype(np.float32),
        rotations=rng.normal(size=(count, 4)).astype(np.float32),
        sh_rest=rng.normal(size=(count, 45)).astype(np.float32),
    )


def test_pack_splat_row_size_and_layout():
    splat = _make_splat(count=10)
    blob = pack_splat(splat)
    assert len(blob) == 10 * _ROW_BYTES

    arr = np.frombuffer(blob, dtype=np.uint8).reshape(10, _ROW_BYTES)
    centers = arr[:, 0:12].view(np.float32).reshape(10, 3)
    np.testing.assert_allclose(centers, splat.positions, atol=1e-6)

    scales = arr[:, 12:24].view(np.float32).reshape(10, 3)
    np.testing.assert_allclose(scales, np.exp(splat.scales), atol=1e-5)


def test_pack_splat_color_encoding_matches_sh0():
    splat = _make_splat(count=8, seed=42)
    blob = pack_splat(splat)
    arr = np.frombuffer(blob, dtype=np.uint8).reshape(8, _ROW_BYTES)
    rgb = arr[:, 24:27]
    sh_c0 = 0.28209479177387814
    expected = np.round(np.clip(0.5 + sh_c0 * splat.sh0, 0.0, 1.0) * 255.0).astype(np.uint8)
    np.testing.assert_array_equal(rgb, expected)


def test_pack_splat_alpha_encoding_uses_sigmoid():
    splat = _make_splat(count=8, seed=7)
    blob = pack_splat(splat)
    arr = np.frombuffer(blob, dtype=np.uint8).reshape(8, _ROW_BYTES)
    alpha = arr[:, 27]
    expected_sigmoid = 1.0 / (1.0 + np.exp(-splat.opacity))
    expected = np.round(np.clip(expected_sigmoid, 0.0, 1.0) * 255.0).astype(np.uint8)
    np.testing.assert_array_equal(alpha, expected)


def test_pack_splat_quat_round_trip_within_quantisation():
    splat = _make_splat(count=64, seed=3)
    blob = pack_splat(splat)
    arr = np.frombuffer(blob, dtype=np.uint8).reshape(64, _ROW_BYTES)
    quat_bytes = arr[:, 28:32].astype(np.int32)
    # Decode in the order the renderer uses: (w, x, y, z) -> indices (0, 1, 2, 3).
    decoded = (quat_bytes - 128) / 128.0
    # Decoded quaternions should be near the normalised originals (up to one ULP per byte = 1/128).
    expected = splat.rotations / np.linalg.norm(splat.rotations, axis=1, keepdims=True)
    diff = np.abs(decoded - expected)
    assert diff.max() < 1.5 / 128.0


def test_ply_to_splat_round_trip(tmp_path):
    splat = _make_splat(count=32)
    ply_path = tmp_path / "cloud.ply"
    splat_path = tmp_path / "cloud.splat"
    write_gaussian_ply(splat, ply_path)
    size = ply_to_splat(ply_path, splat_path)
    assert size == 32 * _ROW_BYTES
    assert splat_path.read_bytes() == pack_splat(splat)


def test_write_splat_format_returns_byte_count(tmp_path):
    splat = _make_splat(count=5)
    out_path = tmp_path / "cloud.splat"
    size = write_splat_format(splat, out_path)
    assert size == 5 * _ROW_BYTES
    assert out_path.stat().st_size == size
