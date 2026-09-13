"""The real cameras: every posed source frame, with its intrinsics, in the scene.

Camera staging used to author only synthetic cameras (room presets, an orbit,
whatever the user captured in the viewport). The reconstruction's own cameras
— one pose and one intrinsic matrix per registered frame, straight out of DA3
— never reached the USD stage or the hand-off to Cosmos. This module carries
them through:

    transforms.json  (+ dataparser normalisation, + gravity rotation)
        -> CaptureFrame per registered image, in the viewer/scene frame
        -> usd/capture_cameras.json          the machine-readable hand-off
        -> /World/Capture in the USD stage   animated camera + keyframes + path

Conventions, stated once: poses are camera-to-world, OpenGL axes (-Z forward,
+Y up), in the same world the splat viewer and the USD scene use. Intrinsics
are pinhole in pixels of the full-resolution source frame. Scale is DA3's
arbitrary unit until something calibrates it, and the JSON says so.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

SCHEMA = 1
FILM_APERTURE_MM = 36.0  # USD/three.js "35mm" horizontal filmback


@dataclass
class CaptureFrame:
    index: int
    file_path: str
    time: float
    c2w: list[list[float]]
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def matrix(self) -> np.ndarray:
        return np.asarray(self.c2w, dtype=np.float64)

    @property
    def position(self) -> np.ndarray:
        return self.matrix[:3, 3]

    @property
    def intrinsic_matrix(self) -> list[list[float]]:
        return [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]

    @property
    def vertical_fov_degrees(self) -> float:
        return math.degrees(2.0 * math.atan(self.height / (2.0 * self.fy)))

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["K"] = self.intrinsic_matrix
        payload["vertical_fov_degrees"] = round(self.vertical_fov_degrees, 3)
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "CaptureFrame":
        return cls(**{key: payload[key] for key in cls.__dataclass_fields__})


def frames_from_transforms(
    transforms: dict, *, duration: float | None = None
) -> list[CaptureFrame]:
    """Nerfstudio transforms (already in the scene frame) -> CaptureFrames.

    Timestamps come from an explicit ``time`` per frame when present. Otherwise
    the frames are spread evenly over ``duration`` (the console and the frame
    extractor both pick one frame per time bucket, so even spacing is the truth
    to within a bucket), or over ``index / fps`` as a last resort.
    """
    raw = sorted(transforms.get("frames") or [], key=lambda frame: frame.get("file_path", ""))
    if not raw:
        return []
    fps = float(transforms.get("fps") or 30.0)
    count = len(raw)
    frames: list[CaptureFrame] = []
    for index, frame in enumerate(raw):
        matrix = np.eye(4)
        matrix[:3, :] = np.asarray(frame["transform_matrix"], dtype=np.float64)[:3, :]
        if "time" in frame:
            t = float(frame["time"])
        elif duration and count > 1:
            t = index * float(duration) / (count - 1)
        else:
            t = index / fps
        width = int(frame.get("w", transforms.get("w", 0)))
        height = int(frame.get("h", transforms.get("h", 0)))
        frames.append(
            CaptureFrame(
                index=index,
                file_path=str(frame.get("file_path", "")),
                time=round(t, 4),
                c2w=matrix.tolist(),
                fx=float(frame.get("fl_x", transforms.get("fl_x", 0.0))),
                fy=float(frame.get("fl_y", transforms.get("fl_y", 0.0))),
                cx=float(frame.get("cx", transforms.get("cx", width / 2))),
                cy=float(frame.get("cy", transforms.get("cy", height / 2))),
                width=width,
                height=height,
            )
        )
    return frames


def load_capture_frames(job_dir: Path, *, duration: float | None = None) -> list[CaptureFrame]:
    """Frames for a job, mapped into the viewer/scene world.

    ``prepare_retrace_transforms`` already knows where the poses live (loose,
    or inside processed_min.zip) and applies the trainer's normalisation and
    the gravity rotation; this is the same world the splat is displayed in.
    """
    from .camera_scene import prepare_retrace_transforms

    prepared = prepare_retrace_transforms(job_dir)
    if prepared is None:
        return []
    return frames_from_transforms(json.loads(prepared.read_text(encoding="utf-8-sig")), duration=duration)


def trajectory_stats(frames: list[CaptureFrame]) -> dict:
    """Numbers a reader can judge a capture by, none of them metric."""
    if len(frames) < 2:
        return {"frames": len(frames)}
    positions = np.stack([frame.position for frame in frames])
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    length = float(steps.sum())
    low, high = np.percentile(positions, [5, 95], axis=0)
    extent = float(np.linalg.norm(high - low))
    closure = float(np.linalg.norm(positions[-1] - positions[0]))
    return {
        "frames": len(frames),
        "path_length": round(length, 4),
        "extent": round(extent, 4),
        "closure_distance": round(closure, 4),
        # < ~0.1 means the walk came back to where it started, which is what
        # loop closure needs to have something to close.
        "closure_ratio": round(closure / extent, 4) if extent > 0 else None,
        "median_step": round(float(np.median(steps)), 5),
        "max_step": round(float(steps.max()), 5),
        "duration": round(frames[-1].time - frames[0].time, 3),
    }


def write_capture_cameras_json(
    job_dir: Path, frames: list[CaptureFrame], *, extra: dict | None = None
) -> Path:
    """usd/capture_cameras.json: what Cosmos Reason gets handed."""
    usd_dir = job_dir / "usd"
    usd_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SCHEMA,
        "conventions": {
            "pose": "camera_to_world, OpenGL axes (-Z forward, +Y up), scene/viewer world",
            "intrinsics": "pinhole, pixels of the full-resolution source frame",
            "units": "reconstruction units; metric scale unknown until calibrated",
            "time": "seconds from the first registered frame",
        },
        "trajectory": trajectory_stats(frames),
        "frames": [frame.to_dict() for frame in frames],
    }
    if extra:
        payload.update(extra)
    path = usd_dir / "capture_cameras.json"
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return path


def load_capture_cameras_json(job_dir: Path) -> list[CaptureFrame]:
    path = job_dir / "usd" / "capture_cameras.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [CaptureFrame.from_dict(frame) for frame in payload.get("frames", [])]


# -- USD ------------------------------------------------------------------------


def intrinsics_to_usd(fx: float, fy: float, cx: float, cy: float, width: int, height: int) -> dict:
    """Pinhole pixels -> UsdGeomCamera lens attributes, on a 36 mm filmback.

    Horizontal FOV is preserved exactly (focal length is fx in filmback
    units); the vertical aperture absorbs a non-square pixel (fx != fy); the
    aperture offsets carry an off-centre principal point. USD's offset is in
    filmback millimetres with +y up, image cy is +y down, hence the sign.
    """
    if width <= 0 or height <= 0 or fx <= 0 or fy <= 0:
        raise ValueError("intrinsics need positive fx, fy, width and height")
    mm_per_px = FILM_APERTURE_MM / width
    focal = fx * mm_per_px
    return {
        "focal_length": focal,
        "horizontal_aperture": FILM_APERTURE_MM,
        "vertical_aperture": height * mm_per_px * (fx / fy),
        "horizontal_aperture_offset": (cx - width / 2) * mm_per_px,
        "vertical_aperture_offset": (height / 2 - cy) * mm_per_px * (fx / fy),
    }


def author_capture_cameras(
    stage, frames: list[CaptureFrame], *, keyframe_stride: int = 25, fps: int = 30
) -> None:
    """/World/Capture: one animated camera, a trajectory curve, sparse keyframes.

    The animated camera replays the capture with per-frame intrinsics as time
    samples. Keyframe cameras every ``keyframe_stride`` frames are static prims
    a DCC or Cosmos can pick individually without scrubbing time.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt

    if not frames:
        return
    stage.SetTimeCodesPerSecond(fps)
    stage.SetFramesPerSecond(fps)
    capture = UsdGeom.Xform.Define(stage, "/World/Capture")
    prim = capture.GetPrim()
    prim.CreateAttribute("vw:frameCount", Sdf.ValueTypeNames.Int, custom=True).Set(len(frames))
    prim.CreateAttribute("vw:cameraFile", Sdf.ValueTypeNames.String, custom=True).Set("capture_cameras.json")
    stats = trajectory_stats(frames)
    for key in ("path_length", "closure_distance", "closure_ratio"):
        value = stats.get(key)
        if value is not None:
            prim.CreateAttribute(f"vw:{key}", Sdf.ValueTypeNames.Double, custom=True).Set(float(value))

    def _apply_lens(camera, frame: CaptureFrame, time_code=None):
        lens = intrinsics_to_usd(frame.fx, frame.fy, frame.cx, frame.cy, frame.width, frame.height)
        when = Usd.TimeCode.Default() if time_code is None else time_code
        camera.GetFocalLengthAttr().Set(lens["focal_length"], when)
        camera.GetHorizontalApertureAttr().Set(lens["horizontal_aperture"], when)
        camera.GetVerticalApertureAttr().Set(lens["vertical_aperture"], when)
        camera.GetHorizontalApertureOffsetAttr().Set(lens["horizontal_aperture_offset"], when)
        camera.GetVerticalApertureOffsetAttr().Set(lens["vertical_aperture_offset"], when)

    def _matrix(frame: CaptureFrame) -> Gf.Matrix4d:
        # USD stores row vectors: transpose the column-vector camera-to-world.
        return Gf.Matrix4d(*frame.matrix.T.flatten())

    # The animated replay.
    animated = UsdGeom.Camera.Define(stage, "/World/Capture/CaptureCamera")
    animated.GetPrim().SetDisplayName("Capture (animated)")
    animated.GetPrim().CreateAttribute("vw:cameraSource", Sdf.ValueTypeNames.String, custom=True).Set("capture")
    animated.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 10000.0))
    animated.ClearXformOpOrder()
    op = animated.AddTransformOp()
    last_code = 0.0
    for frame in frames:
        code = Usd.TimeCode(float(frame.time * fps))
        op.Set(_matrix(frame), code)
        _apply_lens(animated, frame, code)
        last_code = frame.time * fps
    stage.SetStartTimeCode(min(stage.GetStartTimeCode(), 0.0))
    stage.SetEndTimeCode(max(stage.GetEndTimeCode(), last_code))

    # The trajectory, as a polyline anyone can see without scrubbing.
    curve = UsdGeom.BasisCurves.Define(stage, "/World/Capture/Trajectory")
    curve.CreateTypeAttr(UsdGeom.Tokens.linear)
    curve.CreateCurveVertexCountsAttr(Vt.IntArray([len(frames)]))
    curve.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, frame.position)) for frame in frames]))
    curve.CreateWidthsAttr(Vt.FloatArray([0.01]))
    curve.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    curve.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.0, 0.41, 0.58)]))

    # Sparse static keyframes, plus the last frame so the loop end is a prim.
    stride = max(1, int(keyframe_stride))
    picks = list(range(0, len(frames), stride))
    if picks[-1] != len(frames) - 1:
        picks.append(len(frames) - 1)
    UsdGeom.Scope.Define(stage, "/World/Capture/Keyframes")
    for index in picks:
        frame = frames[index]
        camera = UsdGeom.Camera.Define(stage, f"/World/Capture/Keyframes/Frame_{frame.index:05d}")
        camera.GetPrim().SetDisplayName(f"Frame {frame.index} @ {frame.time:.2f}s")
        camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 10000.0))
        camera.ClearXformOpOrder()
        camera.AddTransformOp().Set(_matrix(frame))
        _apply_lens(camera, frame)
        cprim = camera.GetPrim()
        cprim.CreateAttribute("vw:cameraSource", Sdf.ValueTypeNames.String, custom=True).Set("capture")
        cprim.CreateAttribute("vw:frameIndex", Sdf.ValueTypeNames.Int, custom=True).Set(int(frame.index))
        cprim.CreateAttribute("vw:sourceFile", Sdf.ValueTypeNames.String, custom=True).Set(frame.file_path)
        cprim.CreateAttribute("vw:time", Sdf.ValueTypeNames.Double, custom=True).Set(float(frame.time))
