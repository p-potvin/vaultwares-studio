"""Camera entities, keyframed paths, and exporters (USD + nerfstudio).

CameraEntity is the staging unit: a named camera with one or more keyframes
(position + look-at over time). Single-keyframe entities are static shots;
multi-keyframe entities are camera paths. Paths interpolate positions and
look-at targets independently with Catmull-Rom splines, which gives smooth
motion without quaternion bookkeeping.

Exports:
- ``to_nerfstudio_camera_path``: the JSON consumed by ``ns-render
  camera-path`` (OpenGL camera-to-world convention, flat 16-float matrices).
- ``author_usd_camera``: a UsdGeomCamera with a (time-sampled) transform that
  actually aims at the look-at target — both USD and nerfstudio cameras look
  down -Z with +Y up, so one basis builder serves both.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

WORLD_UP = np.array([0.0, 1.0, 0.0])


@dataclass
class CameraKeyframe:
    t: float  # seconds
    position: list[float]
    look_at: list[float]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CameraEntity:
    name: str
    fov_degrees: float = 60.0
    keyframes: list[CameraKeyframe] = field(default_factory=list)
    source: str = "user"  # user | preset | prompt | captured

    @property
    def is_path(self) -> bool:
        return len(self.keyframes) >= 2

    @property
    def duration(self) -> float:
        return self.keyframes[-1].t if self.keyframes else 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "fov_degrees": self.fov_degrees,
            "source": self.source,
            "keyframes": [keyframe.to_dict() for keyframe in self.keyframes],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "CameraEntity":
        return cls(
            name=payload["name"],
            fov_degrees=payload.get("fov_degrees", 60.0),
            source=payload.get("source", "user"),
            keyframes=[CameraKeyframe(**keyframe) for keyframe in payload.get("keyframes", [])],
        )


def load_captured_entities(captured_json: Path) -> list[CameraEntity]:
    """Viewport 'Capture Camera' poses -> static CameraEntities."""
    if not captured_json.exists():
        return []
    try:
        poses = json.loads(captured_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    entities = []
    for index, pose in enumerate(poses, start=1):
        try:
            entities.append(
                CameraEntity(
                    name=pose.get("name", f"Captured {index}"),
                    fov_degrees=float(pose.get("fovDegrees", 60.0)),
                    source="captured",
                    keyframes=[
                        CameraKeyframe(
                            t=0.0,
                            position=[float(v) for v in pose["position"]],
                            look_at=[float(v) for v in pose["lookAt"]],
                        )
                    ],
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return entities


def build_visit_path(
    entities: list[CameraEntity],
    name: str = "Captured Walkthrough",
    seconds_per_stop: float = 3.0,
) -> CameraEntity | None:
    """A path that visits each static camera in capture order."""
    stops = [entity.keyframes[0] for entity in entities if entity.keyframes]
    if len(stops) < 2:
        return None
    return CameraEntity(
        name=name,
        source="captured",
        keyframes=[
            CameraKeyframe(t=index * seconds_per_stop, position=list(stop.position), look_at=list(stop.look_at))
            for index, stop in enumerate(stops)
        ],
    )


# -- interpolation -------------------------------------------------------------


def _catmull_rom(points: np.ndarray, samples: int) -> np.ndarray:
    """Centripetal-ish uniform Catmull-Rom through all control points."""
    count = points.shape[0]
    if count == 1:
        return np.repeat(points, samples, axis=0)
    if count == 2:
        ts = np.linspace(0.0, 1.0, samples)[:, None]
        return points[0] * (1 - ts) + points[1] * ts
    padded = np.vstack([points[0], points, points[-1]])
    out = np.empty((samples, points.shape[1]), dtype=np.float64)
    positions = np.linspace(0.0, count - 1 - 1e-9, samples)
    for row, s in enumerate(positions):
        segment = int(s)
        u = s - segment
        p0, p1, p2, p3 = padded[segment], padded[segment + 1], padded[segment + 2], padded[segment + 3]
        out[row] = 0.5 * (
            (2 * p1)
            + (-p0 + p2) * u
            + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u
            + (-p0 + 3 * p1 - 3 * p2 + p3) * u * u * u
        )
    return out


def sample_path(entity: CameraEntity, fps: int = 30) -> list[tuple[np.ndarray, np.ndarray]]:
    """(position, look_at) per output frame along the entity's keyframes."""
    if not entity.keyframes:
        return []
    frames = max(2, int(round(entity.duration * fps)) + 1) if entity.is_path else 1
    positions = np.array([keyframe.position for keyframe in entity.keyframes], dtype=np.float64)
    targets = np.array([keyframe.look_at for keyframe in entity.keyframes], dtype=np.float64)
    sampled_positions = _catmull_rom(positions, frames)
    sampled_targets = _catmull_rom(targets, frames)
    return list(zip(sampled_positions, sampled_targets))


# -- camera basis ---------------------------------------------------------------


def camera_to_world(position, look_at, up=WORLD_UP) -> np.ndarray:
    """4x4 camera-to-world, -Z forward / +Y up (nerfstudio AND USD convention)."""
    position = np.asarray(position, dtype=np.float64)
    forward = np.asarray(look_at, dtype=np.float64) - position
    norm = np.linalg.norm(forward)
    forward = forward / norm if norm > 1e-9 else np.array([0.0, 0.0, -1.0])
    right = np.cross(forward, up)
    norm = np.linalg.norm(right)
    right = right / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0])
    true_up = np.cross(right, forward)
    matrix = np.eye(4)
    matrix[:3, 0] = right
    matrix[:3, 1] = true_up
    matrix[:3, 2] = -forward
    matrix[:3, 3] = position
    return matrix


# -- exporters -------------------------------------------------------------------


def to_nerfstudio_camera_path(
    entity: CameraEntity,
    fps: int = 30,
    width: int = 1920,
    height: int = 1080,
) -> dict:
    """The JSON document ns-render camera-path consumes."""
    frames = sample_path(entity, fps=fps)
    seconds = max(entity.duration, len(frames) / fps if frames else 0.0)
    return {
        "camera_type": "perspective",
        "render_height": height,
        "render_width": width,
        "fps": fps,
        "seconds": round(seconds, 3),
        "camera_path": [
            {
                "camera_to_world": [float(v) for v in camera_to_world(pos, target).flatten()],
                "fov": entity.fov_degrees,
                "aspect": width / height,
            }
            for pos, target in frames
        ],
    }


def author_usd_camera(stage, prim_path: str, entity: CameraEntity, fps: int = 24):
    """UsdGeomCamera with an aimed (and time-sampled, for paths) transform."""
    from pxr import Gf, Sdf, UsdGeom

    camera = UsdGeom.Camera.Define(stage, prim_path)
    camera.GetFocalLengthAttr().Set(24.0)
    camera.GetHorizontalApertureAttr().Set(36.0)
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 10000.0))
    prim = camera.GetPrim()
    prim.CreateAttribute("vw:cameraName", Sdf.ValueTypeNames.String, custom=True).Set(entity.name)
    prim.CreateAttribute("vw:cameraSource", Sdf.ValueTypeNames.String, custom=True).Set(entity.source)

    op = camera.AddTransformOp()
    if entity.is_path:
        for keyframe in entity.keyframes:
            matrix = camera_to_world(keyframe.position, keyframe.look_at)
            # USD row-vector convention: transpose the column-vector basis.
            op.Set(Gf.Matrix4d(*matrix.T.flatten()), Sdf.TimeCode(keyframe.t * fps))
    else:
        keyframe = entity.keyframes[0]
        matrix = camera_to_world(keyframe.position, keyframe.look_at)
        op.Set(Gf.Matrix4d(*matrix.T.flatten()))
    return camera
