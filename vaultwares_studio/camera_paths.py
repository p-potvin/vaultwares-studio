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
    up: list[float] = field(default_factory=lambda: [0.0, 1.0, 0.0])

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
                            up=[float(v) for v in pose.get("up", [0, 1, 0])],
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
        fov_degrees=entities[0].fov_degrees,
        source="captured",
        keyframes=[
            CameraKeyframe(t=index * seconds_per_stop, position=list(stop.position), look_at=list(stop.look_at), up=list(stop.up))
            for index, stop in enumerate(stops)
        ],
    )


# -- interpolation -------------------------------------------------------------


def _catmull_rom(points: np.ndarray, samples: int, parameters: np.ndarray | None = None) -> np.ndarray:
    """Uniform Catmull-Rom, evaluated at fractional control-point indices."""
    count = points.shape[0]
    if count == 1:
        return np.repeat(points, samples, axis=0)
    if count == 2:
        ts = (parameters if parameters is not None else np.linspace(0.0, 1.0, samples))[:, None]
        return points[0] * (1 - ts) + points[1] * ts
    padded = np.vstack([points[0], points, points[-1]])
    out = np.empty((samples, points.shape[1]), dtype=np.float64)
    positions = parameters if parameters is not None else np.linspace(0.0, count - 1, samples)
    for row, s in enumerate(positions):
        segment = min(int(s), count - 2)
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
    parameters = _sample_parameters(entity, fps)
    frames = len(parameters)
    positions = np.array([keyframe.position for keyframe in entity.keyframes], dtype=np.float64)
    targets = np.array([keyframe.look_at for keyframe in entity.keyframes], dtype=np.float64)
    sampled_positions = _catmull_rom(positions, frames, parameters)
    sampled_targets = _catmull_rom(targets, frames, parameters)
    return list(zip(sampled_positions, sampled_targets))


def _sample_parameters(entity: CameraEntity, fps: int) -> np.ndarray:
    if fps <= 0 or not entity.keyframes:
        raise ValueError("Camera paths require keyframes and a positive frame rate.")
    times = np.array([k.t for k in entity.keyframes], dtype=float)
    if not np.isfinite(times).all() or times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError("Camera keyframe times must start at zero and strictly increase.")
    if not 0 < entity.fov_degrees < 180:
        raise ValueError("Camera field of view must be between 0 and 180 degrees.")
    count = max(2, int(round(entity.duration * fps)) + 1) if entity.is_path else 1
    return np.interp(np.linspace(0, entity.duration, count), times, np.arange(len(times)))


def sample_camera_matrices(entity: CameraEntity, fps: int = 30) -> list[np.ndarray]:
    frames = sample_path(entity, fps)
    parameters = _sample_parameters(entity, fps)
    ups = _catmull_rom(np.asarray([k.up for k in entity.keyframes]), len(frames), parameters)
    return [camera_to_world(pos, target, up) for (pos, target), up in zip(frames, ups)]


def to_viewer_frames(entity: CameraEntity, fps: int = 30) -> list[dict]:
    return [{"position": m[:3, 3].tolist(), "lookAt": (m[:3, 3] - m[:3, 2]).tolist(),
             "up": m[:3, 1].tolist(), "fovDegrees": entity.fov_degrees}
            for m in sample_camera_matrices(entity, fps)]


# -- camera basis ---------------------------------------------------------------


def camera_to_world(position, look_at, up=WORLD_UP) -> np.ndarray:
    """4x4 camera-to-world, -Z forward / +Y up (nerfstudio AND USD convention)."""
    position = np.asarray(position, dtype=np.float64)
    forward = np.asarray(look_at, dtype=np.float64) - position
    norm = np.linalg.norm(forward)
    forward = forward / norm if norm > 1e-9 else np.array([0.0, 0.0, -1.0])
    right = np.cross(forward, up)
    norm = np.linalg.norm(right)
    if norm <= 1e-9:
        axis = np.eye(3)[np.argmin(np.abs(forward))]
        right = np.cross(forward, axis)
        norm = np.linalg.norm(right)
    right = right / norm
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
    frames = sample_camera_matrices(entity, fps=fps)
    seconds = max(entity.duration, len(frames) / fps if frames else 0.0)
    return {
        "camera_type": "perspective",
        "render_height": height,
        "render_width": width,
        "fps": fps,
        "seconds": round(seconds, 3),
        "camera_path": [
            {
                "camera_to_world": matrix.flatten().tolist(),
                "fov": entity.fov_degrees,
                "aspect": width / height,
            }
            for matrix in frames
        ],
    }


def author_usd_camera(stage, prim_path: str, entity: CameraEntity, fps: int = 30):
    """UsdGeomCamera with an aimed (and time-sampled, for paths) transform."""
    from pxr import Gf, Sdf, Usd, UsdGeom

    matrices = sample_camera_matrices(entity, fps)
    camera = UsdGeom.Camera.Define(stage, prim_path)
    # Three.js and ns-render express vertical FOV; keep the 16:9 filmback in
    # sync so exporting a captured view does not silently change its lens.
    vertical_aperture = 36.0 * 9 / 16
    camera.GetFocalLengthAttr().Set(vertical_aperture / (2 * np.tan(np.radians(entity.fov_degrees) / 2)))
    camera.GetHorizontalApertureAttr().Set(36.0)
    camera.GetVerticalApertureAttr().Set(vertical_aperture)
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 10000.0))
    prim = camera.GetPrim()
    prim.SetDisplayName(entity.name)
    prim.CreateAttribute("vw:cameraName", Sdf.ValueTypeNames.String, custom=True).Set(entity.name)
    prim.CreateAttribute("vw:cameraSource", Sdf.ValueTypeNames.String, custom=True).Set(entity.source)

    camera.ClearXformOpOrder()
    attr = prim.GetAttribute("xformOp:transform")
    if attr:
        attr.Clear()
    op = camera.AddTransformOp()
    if entity.is_path:
        stage.SetTimeCodesPerSecond(fps)
        stage.SetFramesPerSecond(fps)
        stage.SetStartTimeCode(0)
        stage.SetEndTimeCode(max(stage.GetEndTimeCode(), entity.duration * fps))
        for t, matrix in zip(np.linspace(0, entity.duration, len(matrices)), matrices):
            # USD row-vector convention: transpose the column-vector basis.
            op.Set(Gf.Matrix4d(*matrix.T.flatten()), Usd.TimeCode(float(t * fps)))
    else:
        matrix = matrices[0]
        op.Set(Gf.Matrix4d(*matrix.T.flatten()))
    return camera
