"""Gravity-align reconstructed clouds so world +Y is up.

COLMAP/Nerfstudio do not gravity-align their output by default: the splat
ships in whatever frame the SfM solver happens to find. ``viewer.js`` and the
walk-pattern presets all assume +Y up, so a +Z-up cloud reads as if the
camera is rolled 90 degrees with no way to right itself.

This module fixes that at the reconstruction-stage boundary:

1. PCA on the (decimated) preview cloud picks the thinnest principal axis as
   the scene's vertical direction.
2. A skewness check on point projections decides whether that axis points
   toward the sky (positive skew, long tail of sparse high points) or toward
   the ground (negative skew); we flip if needed.
3. A Rodrigues rotation sends the chosen "up" to (0, 1, 0). The same matrix
   is applied to (a) the full gaussian PLY's positions AND per-splat
   quaternions, (b) the preview point cloud's positions, and (c) recorded in
   ``summary.json`` so downstream consumers (USD authoring, walk patterns,
   ns-render) inherit the same world frame.

Known limitation: spherical-harmonic bands 1-3 (``f_rest_*``) carry
view-dependent shading and would technically need an SH rotation matrix to
remain consistent after the scene is rotated. We leave them as-is; the
visual impact is a subtle highlight drift that is acceptable for the
typical use case and avoidable later by adding an Ivanic-Ruedenberg real-SH
rotation pass.
"""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .splat_io import GaussianSplat, is_gaussian_ply, read_gaussian_ply, write_gaussian_ply

WORLD_UP = np.array([0.0, 1.0, 0.0])


@dataclass(frozen=True)
class AlignmentResult:
    """What ``align_cloud`` did, suitable for serialisation into summary.json."""

    rotation: np.ndarray  # (3, 3)
    up_before: np.ndarray  # (3,) — PCA-estimated up in the pre-rotation frame
    eigenvalues: np.ndarray  # (3,) — ascending
    skewness: float  # projection skewness; >=0 means up_before pointed toward sky
    flipped: bool  # whether we negated the PCA candidate
    angle_from_y_degrees: float  # angle the rotation closes (sanity number)

    def to_summary_dict(self) -> dict:
        return {
            "rotation": [[float(v) for v in row] for row in self.rotation],
            "up_before": [float(v) for v in self.up_before],
            "eigenvalues_ascending": [float(v) for v in self.eigenvalues],
            "skewness": float(self.skewness),
            "flipped": bool(self.flipped),
            "angle_from_y_degrees": float(self.angle_from_y_degrees),
        }


def _rotation_between_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotation that maps unit vector ``source`` onto unit vector ``target``."""
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    cosine = float(np.dot(source, target))
    if cosine > 1.0 - 1e-9:
        return np.eye(3)
    if cosine < -1.0 + 1e-9:
        # 180 deg rotation around any axis orthogonal to ``source``.
        helper = np.array([1.0, 0.0, 0.0]) if abs(source[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(source, helper)
        axis /= np.linalg.norm(axis)
        return _axis_angle(axis, np.pi)
    axis = np.cross(source, target)
    axis /= np.linalg.norm(axis)
    return _axis_angle(axis, float(np.arccos(cosine)))


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues formula: 3x3 rotation around a unit ``axis`` by ``angle`` radians."""
    x, y, z = axis
    cosine = np.cos(angle)
    sine = np.sin(angle)
    one_minus = 1.0 - cosine
    return np.array(
        [
            [cosine + x * x * one_minus, x * y * one_minus - z * sine, x * z * one_minus + y * sine],
            [y * x * one_minus + z * sine, cosine + y * y * one_minus, y * z * one_minus - x * sine],
            [z * x * one_minus - y * sine, z * y * one_minus + x * sine, cosine + z * z * one_minus],
        ]
    )


def compute_alignment(points: np.ndarray) -> AlignmentResult:
    """Estimate the world-up direction from a point cloud and build the rotation."""
    if points.shape[0] < 4:
        raise ValueError("Need at least 4 points for a meaningful PCA.")
    centred = points - points.mean(axis=0)
    cov = np.cov(centred.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)  # ascending
    up_candidate = eigenvectors[:, 0]
    projections = centred @ up_candidate
    # Outdoor scenes have a long, sparse tail toward the sky (trees, clouds);
    # the ground side is dense and short. Positive skewness => sky side.
    std = projections.std()
    skewness = float(np.mean((projections / std) ** 3)) if std > 1e-12 else 0.0
    flipped = skewness < 0.0
    if flipped:
        up_candidate = -up_candidate
        skewness = -skewness
    rotation = _rotation_between_vectors(up_candidate, WORLD_UP)
    angle_from_y = float(np.degrees(np.arccos(abs(np.dot(up_candidate, WORLD_UP)))))
    return AlignmentResult(
        rotation=rotation,
        up_before=up_candidate,
        eigenvalues=eigenvalues,
        skewness=skewness,
        flipped=flipped,
        angle_from_y_degrees=angle_from_y,
    )


def rotation_to_quaternion(R: np.ndarray) -> np.ndarray:
    """3x3 rotation -> unit quaternion (w, x, y, z). Shepperd-Markley variant."""
    trace = R.trace()
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float64)


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton product (w, x, y, z) — composes scene rotation with per-splat rotations."""
    lw, lx, ly, lz = left[..., 0], left[..., 1], left[..., 2], left[..., 3]
    rw, rx, ry, rz = right[..., 0], right[..., 1], right[..., 2], right[..., 3]
    return np.stack(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        axis=-1,
    )


def rotate_gaussian_splat(splat: GaussianSplat, rotation: np.ndarray) -> GaussianSplat:
    """Apply scene ``rotation`` to a splat's positions and per-splat quaternions."""
    rotation = rotation.astype(np.float64)
    rotated_positions = (splat.positions.astype(np.float64) @ rotation.T).astype(np.float32)
    scene_quat = rotation_to_quaternion(rotation).astype(np.float64)
    # Splat PLYs store unnormalised quaternions; preserve that so the renderer's
    # own normalisation pipeline keeps producing the same per-splat scale.
    per_splat = splat.rotations.astype(np.float64)
    norms = np.linalg.norm(per_splat, axis=1, keepdims=True)
    safe_norms = np.where(norms > 1e-12, norms, 1.0)
    normalised = per_splat / safe_norms
    composed = quaternion_multiply(np.broadcast_to(scene_quat, normalised.shape), normalised)
    rotated_quats = (composed * safe_norms).astype(np.float32)
    return GaussianSplat(
        positions=rotated_positions,
        sh0=splat.sh0,
        opacity=splat.opacity,
        scales=splat.scales,
        rotations=rotated_quats,
        sh_rest=splat.sh_rest,  # see module docstring on SH limitation
    )


def _rewrite_point_ply(path: Path, rotation: np.ndarray) -> None:
    """In-place rotation of a points-only PLY (the preview cloud)."""
    from plyfile import PlyData, PlyElement

    data = PlyData.read(str(path))
    vertex = data["vertex"]
    # Copy detaches from the underlying mmap so the file handle drops below.
    structured = np.array(vertex.data, copy=True)
    del data, vertex
    gc.collect()
    points = np.stack([structured["x"], structured["y"], structured["z"]], axis=1).astype(np.float64)
    rotated = points @ rotation.T
    structured["x"] = rotated[:, 0].astype(structured["x"].dtype)
    structured["y"] = rotated[:, 1].astype(structured["y"].dtype)
    structured["z"] = rotated[:, 2].astype(structured["z"].dtype)
    PlyData([PlyElement.describe(structured, "vertex")], text=False).write(str(path))


def align_cloud(
    full_ply: Path,
    preview_ply: Path,
    *,
    summary_path: Path | None = None,
    captured_cameras_path: Path | None = None,
) -> AlignmentResult | None:
    """End-to-end: estimate up axis from the preview cloud, rewrite both PLYs.

    Also rotates ``captured_cameras_path`` (the viewport's captured-pose JSON)
    if it exists, so existing captures still point at the same physical part of
    the scene after the rotation.

    Returns the result, or ``None`` if alignment was already applied (idempotent
    re-runs of the reconstruction stage skip the rotation).
    """
    summary = _read_summary(summary_path)
    if summary.get("gravity_aligned"):
        return None

    if not preview_ply.exists():
        raise FileNotFoundError(f"Preview cloud missing at {preview_ply}; cannot estimate up axis.")
    points = _read_points(preview_ply)
    result = compute_alignment(points)

    if full_ply.exists() and is_gaussian_ply(full_ply):
        # plyfile mmaps the input arrays, so we have to copy + drop refs before
        # the same path is opened for write (Windows holds the read lock).
        original = read_gaussian_ply(full_ply)
        splat = _detach_splat(original)
        del original
        gc.collect()
        rotated = rotate_gaussian_splat(splat, result.rotation)
        write_gaussian_ply(rotated, full_ply)
    elif full_ply.exists():
        # Placeholder / non-gaussian PLY: rotate as plain points.
        _rewrite_point_ply(full_ply, result.rotation)

    _rewrite_point_ply(preview_ply, result.rotation)

    if captured_cameras_path is not None and captured_cameras_path.exists():
        _rewrite_captured_cameras(captured_cameras_path, result.rotation)

    if summary_path is not None:
        summary["gravity_aligned"] = True
        summary["alignment"] = result.to_summary_dict()
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return result


def _rewrite_captured_cameras(path: Path, rotation: np.ndarray) -> None:
    """Apply ``rotation`` to position/lookAt/up of every captured camera pose."""
    try:
        poses = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    if not isinstance(poses, list):
        return
    R = rotation.astype(np.float64)
    for pose in poses:
        for key in ("position", "lookAt", "up"):
            vec = pose.get(key)
            if not isinstance(vec, list) or len(vec) != 3:
                continue
            rotated = R @ np.asarray(vec, dtype=np.float64)
            pose[key] = rotated.tolist()
    # Atomic-ish write: tmp + replace, avoids leaving a half-written JSON.
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(poses, indent=2), encoding="utf-8")
    tmp.replace(path)


def _detach_splat(splat: GaussianSplat) -> GaussianSplat:
    """Deep-copy every array so the source PLY's mmap can be released."""
    return GaussianSplat(
        positions=np.array(splat.positions, copy=True),
        sh0=np.array(splat.sh0, copy=True),
        opacity=np.array(splat.opacity, copy=True),
        scales=np.array(splat.scales, copy=True),
        rotations=np.array(splat.rotations, copy=True),
        sh_rest=np.array(splat.sh_rest, copy=True) if splat.sh_rest is not None else None,
    )


def _read_summary(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _read_points(path: Path) -> np.ndarray:
    from plyfile import PlyData

    vertex = PlyData.read(str(path))["vertex"]
    return np.stack(
        [np.asarray(vertex["x"]), np.asarray(vertex["y"]), np.asarray(vertex["z"])], axis=1
    ).astype(np.float64)
