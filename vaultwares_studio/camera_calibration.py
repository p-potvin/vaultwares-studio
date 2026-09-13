"""One camera model for the whole capture, with the lens included.

Two things separate the COLMAP-era bundles from the DA3-era ones, and both cost
sharpness in the near field.

**DA3 reports a different focal length for every frame.** It is a feed-forward
model run per frame (or per chunk), so nothing ties the frames to a single
physical lens. Measured on ``backyard_134s_sunny.mp4``: 881.7..915.1 px for the
July non-streaming run and 875.3..913.8 px for the September streaming run --
a 4.3% spread, std 9.15 px, from a phone whose focal length did not change.
splatfacto believes those per-frame numbers. A 9 px focal error displaces a
point at the image edge by ``960 * 9 / 895`` ~ 10 px, and the frames disagree
with each other about where the same world point lands. Gaussians cannot satisfy
all of them, so they spread until they satisfy none -- blur, worst at the edges
and worst where parallax is largest, which is the near field.

**DA3 reports no lens distortion at all.** COLMAP solved an OPENCV model for the
same video (k1 0.01411, k2 -0.01467, p1 -0.00019, p2 -0.00043). At the image
corner that is ~14 px of radial displacement. nerfstudio *will* correct it --
``full_images_datamanager`` undistorts every image at load time -- but only when
``distortion_params`` is present and non-zero, and the DA3 bundles leave it
empty, so that code path has never run on one of our splats.

The fix for both is the same shape: emit ONE camera at the top level of
transforms.json, the way COLMAP does, instead of N cameras one per frame.
nerfstudio's dataparser reads top-level intrinsics as the shared default and
per-frame keys as overrides, so writing the shared block and omitting the
per-frame keys is all it takes.

Distortion cannot be recovered from DA3 -- it never estimates one. It is a
property of the lens, not the capture, so it is measured once per phone (by
COLMAP, on any video that phone shot) and carried forward here.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# nerfstudio's Brown-Conrady keys. It supports k3/k4 as well; we only ever solve
# the four COLMAP's OPENCV model gives us, and _undistort_image asserts the 4th
# Brown parameter is zero anyway.
DISTORTION_KEYS = ("k1", "k2", "p1", "p2")


@dataclass(frozen=True)
class CameraCalibration:
    """A physical lens: what stays the same across every frame it shot."""

    fl_x: float
    fl_y: float
    cx: float
    cy: float
    w: int
    h: int
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    camera_model: str = "OPENCV"
    # Free-text note about where the numbers came from, so a bundle carries its
    # own provenance. Never consumed by nerfstudio.
    source: str = ""

    @property
    def has_distortion(self) -> bool:
        """nerfstudio skips undistortion entirely when every term is zero."""
        return any(getattr(self, key) != 0.0 for key in DISTORTION_KEYS)

    def scaled_to(self, width: int, height: int) -> "CameraCalibration":
        """Re-express the intrinsics in a different image resolution.

        Distortion coefficients are in *normalised* image coordinates, so they
        are resolution-independent and carry across untouched. Getting that
        backwards -- scaling k1 with the focal -- is the classic way to turn a
        correction into a new error.
        """
        sx, sy = width / self.w, height / self.h
        return CameraCalibration(
            fl_x=self.fl_x * sx,
            fl_y=self.fl_y * sy,
            cx=self.cx * sx,
            cy=self.cy * sy,
            w=int(width),
            h=int(height),
            k1=self.k1,
            k2=self.k2,
            p1=self.p1,
            p2=self.p2,
            camera_model=self.camera_model,
            source=self.source,
        )

    def as_transforms_fields(self) -> dict:
        """The top-level block of a nerfstudio transforms.json."""
        fields = {
            "camera_model": self.camera_model,
            "fl_x": float(self.fl_x),
            "fl_y": float(self.fl_y),
            "cx": float(self.cx),
            "cy": float(self.cy),
            "w": int(self.w),
            "h": int(self.h),
        }
        # Only emit distortion when there is some. An all-zero block is harmless
        # but implies a calibration that does not exist.
        if self.has_distortion:
            fields.update({key: float(getattr(self, key)) for key in DISTORTION_KEYS})
        return fields

    @classmethod
    def from_transforms(cls, transforms: dict, *, source: str = "") -> "CameraCalibration":
        """Read a shared camera out of a transforms.json -- COLMAP's, usually.

        This is how a lens gets measured: run COLMAP once on any video from the
        phone, then reuse the distortion terms for every DA3 bundle after it.
        """
        missing = [k for k in ("fl_x", "fl_y", "cx", "cy", "w", "h") if k not in transforms]
        if missing:
            raise ValueError(
                f"transforms.json has no shared camera block (missing {', '.join(missing)}); "
                "per-frame intrinsics cannot be read as a lens calibration"
            )
        return cls(
            fl_x=float(transforms["fl_x"]),
            fl_y=float(transforms["fl_y"]),
            cx=float(transforms["cx"]),
            cy=float(transforms["cy"]),
            w=int(transforms["w"]),
            h=int(transforms["h"]),
            camera_model=str(transforms.get("camera_model", "OPENCV")),
            source=source,
            **{key: float(transforms.get(key, 0.0)) for key in DISTORTION_KEYS},
        )

    @classmethod
    def load(cls, path: Path) -> "CameraCalibration":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path


@dataclass(frozen=True)
class IntrinsicsConsensus:
    """What collapsing per-frame intrinsics to one camera actually cost."""

    frames: int
    fl_x: float
    fl_y: float
    cx: float
    cy: float
    fl_x_spread: float  # (max - min) / median, as a fraction
    fl_y_spread: float
    max_edge_shift_px: float

    def as_dict(self) -> dict:
        return {
            "frames": self.frames,
            "fl_x": round(self.fl_x, 3),
            "fl_y": round(self.fl_y, 3),
            "cx": round(self.cx, 3),
            "cy": round(self.cy, 3),
            "fl_x_spread_pct": round(self.fl_x_spread * 100, 3),
            "fl_y_spread_pct": round(self.fl_y_spread * 100, 3),
            "max_edge_shift_px": round(self.max_edge_shift_px, 2),
        }


def consensus_intrinsics(intrinsics: np.ndarray, size: tuple[int, int]) -> IntrinsicsConsensus:
    """Collapse per-frame ``(N, 4)`` fx/fy/cx/cy to the one camera that shot them.

    Median, not mean: DA3's per-frame estimates occasionally include a wild one
    (a frame that is mostly sky has almost no focal signal), and a single outlier
    should not move the shared camera.

    ``max_edge_shift_px`` is the reprojection cost of the disagreement being
    removed -- how far the worst frame's focal moves a point at the image edge,
    relative to the consensus. It is the number that says whether this mattered.
    """
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if intrinsics.ndim != 2 or intrinsics.shape[1] != 4:
        raise ValueError(f"expected (N, 4) fx/fy/cx/cy, got {intrinsics.shape}")
    if len(intrinsics) == 0:
        raise ValueError("no intrinsics to reach consensus on")

    fl_x, fl_y, cx, cy = (float(np.median(intrinsics[:, i])) for i in range(4))
    width, height = size

    def spread(column: int, median: float) -> float:
        if median == 0:
            return 0.0
        values = intrinsics[:, column]
        return float(values.max() - values.min()) / median

    # Worst-case displacement at the image edge, in pixels of this frame size.
    half = max(width, height) / 2.0
    worst = max(
        abs(float(intrinsics[:, 0].max()) - fl_x),
        abs(float(intrinsics[:, 0].min()) - fl_x),
    )
    edge_shift = half * worst / fl_x if fl_x else 0.0

    return IntrinsicsConsensus(
        frames=len(intrinsics),
        fl_x=fl_x,
        fl_y=fl_y,
        cx=cx,
        cy=cy,
        fl_x_spread=spread(0, fl_x),
        fl_y_spread=spread(1, fl_y),
        max_edge_shift_px=edge_shift,
    )
