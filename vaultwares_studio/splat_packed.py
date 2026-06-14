"""3DGS PLY -> ``.splat`` binary conversion for fast viewport loads.

The ``.splat`` format consumed by ``gaussian-splats-3d`` packs each gaussian
into a fixed 32-byte row:

- bytes 0..11   : center (3 × float32)
- bytes 12..23  : scale  (3 × float32)         -- linear, NOT log
- bytes 24..27  : RGBA   (4 × uint8)
- bytes 28..31  : quaternion (4 × uint8), ordered (w, x, y, z), encoded
                  byte = round((q + 1) * 128) so the renderer decodes
                  q = (byte - 128) / 128 then normalises.

3DGS PLYs encode quantities differently from what the renderer wants on the
wire, so this module decodes:

- ``scale_*``  : 3DGS stores log(scale); we exponentiate.
- ``opacity``  : 3DGS stores logit(opacity); we sigmoid.
- ``f_dc_*``   : SH degree-0 coefficients; we recombine via the standard
                 ``rgb = 0.5 + C0 * f_dc`` rule then clip to [0, 1].

For a 477k-gaussian backyard splat this writes ~15 MB instead of ~112 MB,
and the viewer skips the PLY ASCII/binary header parse entirely.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .splat_io import GaussianSplat, read_gaussian_ply

# 3DGS SH degree-0 normalising constant (same as splat_io._SH_C0).
_SH_C0 = 0.28209479177387814
_ROW_BYTES = 32


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def pack_splat(splat: GaussianSplat) -> bytes:
    """Encode a :class:`GaussianSplat` as a ``.splat`` byte blob."""
    count = splat.count
    positions = splat.positions.astype(np.float32)
    # Log-encoded -> linear scale.
    scales = np.exp(splat.scales.astype(np.float32))
    # DC SH -> 0..255 RGB.
    rgb = np.clip(0.5 + _SH_C0 * splat.sh0.astype(np.float32), 0.0, 1.0)
    rgb_bytes = np.round(rgb * 255.0).astype(np.uint8)
    # Logit -> 0..255 alpha.
    alpha = _sigmoid(splat.opacity.astype(np.float32))
    alpha_bytes = np.round(np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
    color = np.empty((count, 4), dtype=np.uint8)
    color[:, :3] = rgb_bytes
    color[:, 3] = alpha_bytes
    # Normalise quaternions then quantise to bytes (w, x, y, z).
    quat = splat.rotations.astype(np.float32)
    norms = np.linalg.norm(quat, axis=1, keepdims=True)
    safe_norms = np.where(norms > 1e-12, norms, 1.0)
    normalised = quat / safe_norms
    quat_bytes = np.round(np.clip(normalised, -1.0, 1.0) * 128.0 + 128.0).clip(0, 255).astype(np.uint8)
    # Row assembly: contiguous (12 + 12 + 4 + 4) bytes per splat.
    rows = np.empty((count, _ROW_BYTES), dtype=np.uint8)
    rows[:, 0:12] = positions.view(np.uint8).reshape(count, 12)
    rows[:, 12:24] = scales.view(np.uint8).reshape(count, 12)
    rows[:, 24:28] = color
    rows[:, 28:32] = quat_bytes
    return rows.tobytes()


def write_splat_format(splat: GaussianSplat, out_path: Path) -> int:
    """Encode ``splat`` to ``out_path`` and return the file size in bytes."""
    payload = pack_splat(splat)
    out_path.write_bytes(payload)
    return len(payload)


def ply_to_splat(ply_path: Path, out_path: Path) -> int:
    """Convenience: read a 3DGS PLY then write the packed .splat alongside it."""
    return write_splat_format(read_gaussian_ply(ply_path), out_path)
