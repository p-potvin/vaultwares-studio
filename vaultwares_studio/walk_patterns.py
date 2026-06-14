"""Parametric walkthrough patterns that emit ``CameraEntity`` paths.

A pattern takes a :class:`SceneBounds` (center + radius derived from the
reconstructed cloud) plus pattern-specific parameters and returns a single
``CameraEntity`` ready for ``sample_path`` / ``to_nerfstudio_camera_path`` /
``author_usd_camera``.

Patterns are registered in :data:`WALK_PATTERNS` so the GUI and pipeline can
list and build them by name. ``retrace_steps`` consumes the per-frame
Nerfstudio ``transforms.json`` to replay the source-video trajectory — the
"first person walking through the scene" path requested over the orbit/spiral
templates.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .camera_paths import CameraEntity, CameraKeyframe


@dataclass(frozen=True)
class SceneBounds:
    """Centroid and characteristic radius of the reconstructed scene."""

    center: tuple[float, float, float]
    radius: float

    @property
    def center_arr(self) -> np.ndarray:
        return np.asarray(self.center, dtype=np.float64)


# Patterns place the camera with +Y as world up. Reconstructions that come out
# tilted (no gravity alignment) will look skewed — that is a scene-orientation
# bug, not a path bug; gravity-align the cloud upstream and these paths sit
# correctly on top of it.
_WORLD_UP = np.array([0.0, 1.0, 0.0])


def orbit(
    bounds: SceneBounds,
    *,
    seconds: float = 12.0,
    altitude: float = 0.45,
    radius_scale: float = 2.5,
    stops: int = 9,
    direction: str = "cw",
    name: str = "Orbit",
) -> CameraEntity:
    """Closed orbit around the scene centroid at fixed altitude.

    Defaults put the camera at ~2.5x the 5-95-percentile radius from the
    centroid, which (a) satisfies the d > r/tan(FOV/2) framing rule for a
    60deg FOV with room to spare, and (b) sits clear of sparse outlier
    splats that extend past the percentile box. radius_scale=0.9 (the
    original default) put the camera INSIDE the subject and rendered
    walkthroughs showed mostly empty space.
    """
    sign = -1.0 if direction == "ccw" else 1.0
    cx, cy, cz = bounds.center
    r = bounds.radius * radius_scale
    keyframes: list[CameraKeyframe] = []
    for index in range(stops):
        angle = sign * 2.0 * math.pi * index / (stops - 1)
        keyframes.append(
            CameraKeyframe(
                t=seconds * index / (stops - 1),
                position=[cx + r * math.cos(angle), cy + bounds.radius * altitude, cz + r * math.sin(angle)],
                look_at=[cx, cy, cz],
            )
        )
    return CameraEntity(name=name, source="preset", keyframes=keyframes)


def dolly_in(
    bounds: SceneBounds,
    *,
    seconds: float = 6.0,
    start_distance: float = 2.0,
    end_distance: float = 0.3,
    altitude: float = 0.4,
    azimuth: float = 0.0,
    name: str = "Dolly In",
) -> CameraEntity:
    """Straight push toward the centroid along a fixed bearing."""
    cx, cy, cz = bounds.center
    direction = np.array([math.cos(azimuth), 0.0, math.sin(azimuth)])
    height = cy + bounds.radius * altitude
    start = bounds.center_arr + direction * bounds.radius * start_distance
    end = bounds.center_arr + direction * bounds.radius * end_distance
    start[1] = end[1] = height
    return CameraEntity(
        name=name,
        source="preset",
        keyframes=[
            CameraKeyframe(t=0.0, position=start.tolist(), look_at=[cx, cy, cz]),
            CameraKeyframe(t=seconds, position=end.tolist(), look_at=[cx, cy, cz]),
        ],
    )


def dolly_out(bounds: SceneBounds, *, seconds: float = 6.0, **kwargs) -> CameraEntity:
    """Inverse of :func:`dolly_in` (defaults flipped)."""
    kwargs.setdefault("start_distance", 0.4)
    kwargs.setdefault("end_distance", 2.2)
    kwargs.setdefault("name", "Dolly Out")
    return dolly_in(bounds, seconds=seconds, **kwargs)


def crane_up(
    bounds: SceneBounds,
    *,
    seconds: float = 8.0,
    start_altitude: float = 0.0,
    end_altitude: float = 1.6,
    distance: float = 1.2,
    azimuth: float = 0.0,
    name: str = "Crane Up",
) -> CameraEntity:
    """Rising shot — start near eye level, end overhead, always aimed at center."""
    cx, cy, cz = bounds.center
    direction = np.array([math.cos(azimuth), 0.0, math.sin(azimuth)])
    base = bounds.center_arr + direction * bounds.radius * distance
    stops = 9
    keyframes: list[CameraKeyframe] = []
    for index in range(stops):
        u = index / (stops - 1)
        altitude = start_altitude + (end_altitude - start_altitude) * u
        # Pull the camera horizontally inward as it rises so the framing tilts
        # rather than just lifting — without this it feels like a flagpole.
        inward = 1.0 - 0.55 * u
        position = bounds.center_arr + direction * bounds.radius * distance * inward
        position[1] = cy + bounds.radius * altitude
        keyframes.append(
            CameraKeyframe(t=seconds * u, position=position.tolist(), look_at=[cx, cy, cz])
        )
    return CameraEntity(name=name, source="preset", keyframes=keyframes)


def figure_8(
    bounds: SceneBounds,
    *,
    seconds: float = 14.0,
    altitude: float = 0.4,
    radius_scale: float = 0.8,
    stops: int = 17,
    name: str = "Figure 8",
) -> CameraEntity:
    """Lemniscate of Bernoulli over the scene; closed loop."""
    cx, cy, cz = bounds.center
    height = cy + bounds.radius * altitude
    a = bounds.radius * radius_scale
    keyframes: list[CameraKeyframe] = []
    for index in range(stops):
        t = 2.0 * math.pi * index / (stops - 1)
        denom = 1.0 + math.sin(t) * math.sin(t)
        x = a * math.cos(t) / denom
        z = a * math.sin(t) * math.cos(t) / denom
        keyframes.append(
            CameraKeyframe(
                t=seconds * index / (stops - 1),
                position=[cx + x, height, cz + z],
                look_at=[cx, cy, cz],
            )
        )
    return CameraEntity(name=name, source="preset", keyframes=keyframes)


def doorway_reveal(
    bounds: SceneBounds,
    *,
    seconds: float = 7.0,
    start_distance: float = 1.8,
    end_distance: float = 0.6,
    altitude: float = 0.5,
    yaw_degrees: float = 12.0,
    azimuth: float = 0.0,
    name: str = "Doorway Reveal",
) -> CameraEntity:
    """Wide low push into the scene with a small yaw to give parallax."""
    cx, cy, cz = bounds.center
    height = cy + bounds.radius * altitude
    yaw = math.radians(yaw_degrees)
    forward = np.array([math.cos(azimuth), 0.0, math.sin(azimuth)])
    side = np.array([math.cos(azimuth + math.pi / 2), 0.0, math.sin(azimuth + math.pi / 2)])
    start = bounds.center_arr + forward * bounds.radius * start_distance - side * bounds.radius * math.sin(yaw) * 0.4
    end = bounds.center_arr + forward * bounds.radius * end_distance + side * bounds.radius * math.sin(yaw) * 0.4
    start[1] = end[1] = height
    # Look-at drifts opposite the camera so the framing yaws across the scene.
    look_start = bounds.center_arr + side * bounds.radius * 0.25
    look_end = bounds.center_arr - side * bounds.radius * 0.25
    return CameraEntity(
        name=name,
        source="preset",
        keyframes=[
            CameraKeyframe(t=0.0, position=start.tolist(), look_at=look_start.tolist()),
            CameraKeyframe(t=seconds * 0.5, position=((start + end) / 2).tolist(), look_at=[cx, cy, cz]),
            CameraKeyframe(t=seconds, position=end.tolist(), look_at=look_end.tolist()),
        ],
    )


def retrace_steps(
    bounds: SceneBounds,
    *,
    transforms_json: Path,
    stride: int = 1,
    seconds: float | None = None,
    look_ahead: float = 0.5,
    name: str = "Retrace Steps",
) -> CameraEntity:
    """Replay the source-video camera trajectory from a Nerfstudio transforms.json.

    Each registered frame contributes a keyframe at its source-video timestamp
    (uniformly spaced if frames lack ``time``). ``look_at`` is sampled along
    the camera's forward axis at ``look_ahead`` * scene radius — gives the
    first-person feeling of walking through the scene rather than tracking the
    centroid.
    """
    transforms_json = Path(transforms_json)
    data = json.loads(transforms_json.read_text(encoding="utf-8"))
    frames = data.get("frames") or []
    if len(frames) < 2:
        raise ValueError(
            f"transforms.json at {transforms_json} only contains {len(frames)} frames"
        )
    frames = sorted(frames, key=lambda frame: frame.get("file_path", ""))[::max(1, stride)]
    fps = float(data.get("fps") or 30.0)
    keyframes: list[CameraKeyframe] = []
    for index, frame in enumerate(frames):
        matrix = np.asarray(frame["transform_matrix"], dtype=np.float64)
        # Nerfstudio transforms are camera-to-world, OpenGL convention: -Z forward.
        position = matrix[:3, 3]
        forward = -matrix[:3, 2]
        look_at = position + forward * bounds.radius * look_ahead
        t = float(frame.get("time", index / fps))
        keyframes.append(
            CameraKeyframe(t=t, position=position.tolist(), look_at=look_at.tolist())
        )
    if seconds is not None and keyframes[-1].t > 0:
        scale = seconds / keyframes[-1].t
        keyframes = [
            CameraKeyframe(t=keyframe.t * scale, position=keyframe.position, look_at=keyframe.look_at)
            for keyframe in keyframes
        ]
    return CameraEntity(name=name, source="captured", keyframes=keyframes)


WALK_PATTERNS: dict[str, Callable[..., CameraEntity]] = {
    "orbit": orbit,
    "dolly_in": dolly_in,
    "dolly_out": dolly_out,
    "crane_up": crane_up,
    "figure_8": figure_8,
    "doorway_reveal": doorway_reveal,
    "retrace_steps": retrace_steps,
}


def available_patterns() -> list[str]:
    return list(WALK_PATTERNS.keys())


def build_pattern(name: str, bounds: SceneBounds, **params) -> CameraEntity:
    try:
        builder = WALK_PATTERNS[name]
    except KeyError as error:
        raise KeyError(f"Unknown walk pattern '{name}'. Available: {available_patterns()}") from error
    return builder(bounds, **params)


def bounds_from_preview_ply(preview_ply: Path) -> SceneBounds:
    """Robust centroid + radius from the decimated preview cloud."""
    from plyfile import PlyData

    vertex = PlyData.read(str(preview_ply))["vertex"]
    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)
    low, high = np.percentile(points, [5, 95], axis=0)
    center = tuple(float(value) for value in (low + high) / 2)
    radius = float(np.linalg.norm(high - low) / 2) or 2.0
    return SceneBounds(center=center, radius=radius)
