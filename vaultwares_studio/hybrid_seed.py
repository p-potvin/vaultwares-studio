"""COLMAP's geometry, DA3's density: the hybrid seed cloud.

The split that matters, measured on ``backyard_134s_sunny.mp4`` across three
runs of the same footage:

* **COLMAP is right about where the cameras are.** Sub-pixel features at
  1920x1080, bundle-adjusted jointly, with a lens solved (k1 0.01411,
  k2 -0.01467). Its splat put half its gaussians within 0.16 camera-path
  extents of the walk -- density where you actually stood.
* **COLMAP is sparse.** 185,355 triangulated points from 491 registered
  frames, 377 per frame. Only what its matcher could see in three views.
* **DA3 is dense.** A depth value for every pixel of every frame, textureless
  wall and blank sky included, where a feature matcher has nothing.
* **DA3 is wrong about where the cameras are.** A different focal per frame
  (see ``camera_calibration``), no lens, and -- on the streaming path -- a
  per-chunk SIM3 with no globally consistent scale.

So: take poses and intrinsics from COLMAP, take depth from DA3, and reconcile
them. This module is the reconciliation.

**The scale problem.** DA3's depth is only defined up to an unknown transform
per frame -- nominally metric, in practice drifting. COLMAP's sparse points are
in one consistent frame. Where a sparse point lands inside a frame we get a pair
``(predicted depth, true depth)``, and a few hundred of those per frame is
plenty to solve ``true ~= a * predicted + b``. This is the standard alignment
step (DSNeRF, MonoSDF, DN-Splatter all do a version of it); what is specific
here is that we then keep the aligned depth rather than only a loss term.

**Affine, not scale-only.** A pure scale is the physically motivated model for a
metric predictor. We fit both and report both RMSEs, because the comparison is
diagnostic: if the shift is doing real work, DA3's depth is relative rather than
metric on this scene, and that is worth knowing before trusting it anywhere
else.

**Visibility, then robustness -- in that order.** A sparse point that projects
into a frame is not necessarily *visible* in it; it can sit behind a wall, and
an occluded point always reads too far. The first version of this module
projected the whole cloud into every frame and left a Huber loss to reject the
rest. That does not work, and the measurement says why: on the June 14 backyard
bundle the median frame has **45,966** points projecting into it and COLMAP's
own tracks say it saw **577**. The occluded fraction is ~98.7% -- not a tail an
M-estimator can absorb but the entire population, so the fit described the
occluders. It produced per-frame scales spread over 630%, correlations between
predicted and "true" depth ranging +0.60 to -0.10, and half the frames failing
to align at all. None of that was about the depth predictor.

So correspondences now come from ``colmap_model.read_sparse_model`` -- the
points COLMAP's matcher actually verified in each frame. The Huber loss stays,
because even a verified track can be mismatched, but it is now cleaning up a
genuine tail rather than being asked to find the signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Right-multiplying an OpenGL camera-to-world by this gives the OpenCV one.
# Same matrix streaming_convert applies in the other direction.
OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


@dataclass(frozen=True)
class DepthAlignment:
    """The transform that puts one frame's predicted depth into COLMAP's units."""

    scale: float
    shift: float
    correspondences: int
    inliers: int
    rmse: float  # metres, affine fit
    rmse_scale_only: float  # metres, shift forced to zero

    @property
    def ok(self) -> bool:
        """Enough support, and a scale that is not degenerate.

        A frame looking at the sky can have a handful of correspondences all at
        similar depth; the fit is then unconstrained and any scale explains it.
        """
        return self.inliers >= MIN_CORRESPONDENCES and self.scale > 0

    @property
    def shift_matters(self) -> bool:
        """True when the affine fit beats scale-only by more than noise.

        If this is consistently true, DA3's depth is relative on this scene, not
        metric, and anything downstream treating it as metric is wrong.
        """
        return self.rmse_scale_only > 1.05 * self.rmse

    def apply(self, depth: np.ndarray) -> np.ndarray:
        return np.asarray(depth, dtype=np.float64) * self.scale + self.shift

    def as_dict(self) -> dict:
        return {
            "scale": round(self.scale, 6),
            "shift": round(self.shift, 6),
            "correspondences": self.correspondences,
            "inliers": self.inliers,
            "rmse": round(self.rmse, 5),
            "rmse_scale_only": round(self.rmse_scale_only, 5),
        }


# Below this the fit is not constrained enough to trust. Measured against the
# June 14 COLMAP model for backyard_134s_sunny.mp4 (sparse/0: 49,107 points,
# 487 registered frames), the median frame has 577 VERIFIED observations and the
# worst has 36. So a typical frame clears this comfortably while a genuinely
# weak one — sky, a blank wall, a pose that came out wrong — does not, which is
# the whole job of the threshold. It is a guard, not a tuning knob.
#
# (An earlier revision cited 45,966 here. That was the count from projecting the
#  entire cloud, ~98.7% of which is occluded in any given frame, and using it
#  was the bug this threshold could not have caught.)
MIN_CORRESPONDENCES = 24


def opengl_c2w_to_opencv_w2c(c2w_opengl: np.ndarray) -> np.ndarray:
    """nerfstudio's pose convention to the one projection maths wants."""
    c2w_opengl = np.asarray(c2w_opengl, dtype=np.float64)
    if c2w_opengl.shape != (4, 4):
        raise ValueError(f"expected a 4x4 camera-to-world, got {c2w_opengl.shape}")
    return np.linalg.inv(c2w_opengl @ OPENGL_TO_OPENCV)


def project(
    points: np.ndarray,
    intrinsics: np.ndarray,
    c2w_opengl: np.ndarray,
    size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World points to pixel coordinates.

    Returns ``(uv, depth, inside)`` where ``uv`` is ``(N, 2)`` float pixels,
    ``depth`` is ``(N,)`` metres along the optical axis, and ``inside`` marks
    the points that are in front of the camera AND within the frame.
    """
    points = np.asarray(points, dtype=np.float64)
    w2c = opengl_c2w_to_opencv_w2c(c2w_opengl)
    cam = points @ w2c[:3, :3].T + w2c[:3, 3]
    depth = cam[:, 2]

    width, height = size
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = (cam[:, :2] / depth[:, None]) @ np.asarray(intrinsics, dtype=np.float64)[:2, :2].T
    uv += np.asarray(intrinsics, dtype=np.float64)[:2, 2]

    inside = (
        (depth > 0)
        & np.isfinite(uv).all(axis=1)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )
    return uv, depth, inside


def _huber_irls(
    predicted: np.ndarray,
    truth: np.ndarray,
    *,
    fit_shift: bool,
    iterations: int = 8,
) -> tuple[float, float, np.ndarray]:
    """Iteratively reweighted least squares with a Huber loss.

    The Huber threshold is re-derived each iteration from the median absolute
    residual rather than fixed, because scene scale is arbitrary -- a threshold
    in metres means something different on every run.
    """
    design = (
        np.column_stack([predicted, np.ones_like(predicted)])
        if fit_shift
        else predicted[:, None]
    )
    weights = np.ones_like(predicted)
    coefficients = np.zeros(design.shape[1])

    for _ in range(iterations):
        weighted = design * weights[:, None]
        coefficients, *_ = np.linalg.lstsq(weighted, truth * weights, rcond=None)
        residual = np.abs(design @ coefficients - truth)
        # 1.4826 * MAD is the consistent estimator of sigma for a normal.
        sigma = 1.4826 * np.median(residual)
        if sigma <= 0:
            break
        delta = 1.345 * sigma  # 95% efficiency under normal noise
        weights = np.where(residual <= delta, 1.0, delta / np.maximum(residual, 1e-12))

    scale = float(coefficients[0])
    shift = float(coefficients[1]) if fit_shift else 0.0
    return scale, shift, weights


def align_depth_to_sparse(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    c2w_opengl: np.ndarray,
    sparse_points: np.ndarray,
) -> DepthAlignment:
    """Solve the depth transform for one frame against COLMAP's sparse points.

    ``depth`` is the predictor's map at ITS resolution; ``intrinsics`` must be
    in that same resolution. Everything else is in COLMAP's world.
    """
    depth = np.asarray(depth, dtype=np.float64)
    height, width = depth.shape[:2]
    uv, true_depth, inside = project(sparse_points, intrinsics, c2w_opengl, (width, height))

    if not inside.any():
        return DepthAlignment(1.0, 0.0, 0, 0, float("inf"), float("inf"))

    # Nearest-pixel lookup. Bilinear would be defensible but depth is
    # discontinuous at object boundaries, and interpolating across an edge
    # invents a value that is on neither surface.
    columns = np.clip(np.round(uv[inside, 0]).astype(int), 0, width - 1)
    rows = np.clip(np.round(uv[inside, 1]).astype(int), 0, height - 1)
    predicted = depth[rows, columns]
    truth = true_depth[inside]

    usable = np.isfinite(predicted) & np.isfinite(truth) & (predicted > 0) & (truth > 0)
    predicted, truth = predicted[usable], truth[usable]
    count = int(usable.sum())
    if count < MIN_CORRESPONDENCES:
        return DepthAlignment(1.0, 0.0, count, count, float("inf"), float("inf"))

    scale, shift, weights = _huber_irls(predicted, truth, fit_shift=True)
    scale_only, _, _ = _huber_irls(predicted, truth, fit_shift=False)

    def rmse(a: float, b: float) -> float:
        residual = (predicted * a + b) - truth
        # Score on the inliers: an RMSE dominated by occluded points measures
        # the occlusions, not the fit.
        keep = weights >= 1.0
        if not keep.any():
            keep = np.ones_like(residual, dtype=bool)
        return float(np.sqrt(np.mean(residual[keep] ** 2)))

    return DepthAlignment(
        scale=scale,
        shift=shift,
        correspondences=count,
        inliers=int((weights >= 1.0).sum()),
        rmse=rmse(scale, shift),
        rmse_scale_only=rmse(scale_only, 0.0),
    )


def backproject(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    c2w_opengl: np.ndarray,
    *,
    colors: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
    keep_quantile: float = 0.6,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """One aligned depth map to world points.

    ``keep_quantile`` keeps the most-confident fraction of pixels, matching what
    ``da3_to_sparse_pc`` already does. ``stride`` subsamples the pixel grid --
    at 500 frames the full grid is far more points than any seed needs, and
    striding is cheaper than sampling and then discarding.
    """
    depth = np.asarray(depth, dtype=np.float64)[::stride, ::stride]
    height, width = depth.shape[:2]
    intrinsics = np.asarray(intrinsics, dtype=np.float64)

    rows, columns = np.mgrid[0:height, 0:width]
    # The grid was strided; the intrinsics were not.
    x = (columns * stride - intrinsics[0, 2]) * depth / intrinsics[0, 0]
    y = (rows * stride - intrinsics[1, 2]) * depth / intrinsics[1, 1]
    cam = np.stack([x, y, depth], axis=-1).reshape(-1, 3)

    keep = np.isfinite(cam).all(axis=1) & (cam[:, 2] > 0)
    if confidence is not None and 0.0 < keep_quantile < 1.0:
        conf = np.asarray(confidence, dtype=np.float64)[::stride, ::stride].reshape(-1)
        keep &= conf > np.percentile(conf, (1.0 - keep_quantile) * 100.0)

    c2w = np.asarray(c2w_opengl, dtype=np.float64) @ OPENGL_TO_OPENCV
    world = cam[keep] @ c2w[:3, :3].T + c2w[:3, 3]

    if colors is None:
        rgb = np.full((len(world), 3), 200, dtype=np.uint8)
    else:
        rgb = np.asarray(colors)[::stride, ::stride].reshape(-1, 3)[keep].astype(np.uint8)
    return world, rgb


def voxel_downsample(
    points: np.ndarray, colors: np.ndarray, voxel: float
) -> tuple[np.ndarray, np.ndarray]:
    """Keep one point per voxel — 500 overlapping depth maps are mostly duplicates.

    First-wins rather than centroid-averaging: averaging across a voxel that
    straddles a depth discontinuity places a point in empty space between two
    surfaces, and those are exactly the points that become floaters.
    """
    if voxel <= 0 or len(points) == 0:
        return points, colors
    keys = np.floor(np.asarray(points, dtype=np.float64) / voxel).astype(np.int64)
    _, index = np.unique(keys, axis=0, return_index=True)
    index.sort()  # preserve frame order, so earlier frames win ties
    return points[index], colors[index]


@dataclass(frozen=True)
class HybridSeedReport:
    """What the fusion did, for summary.json and for deciding whether to trust it."""

    frames_total: int
    frames_aligned: int
    points_before_dedup: int
    points_after_dedup: int
    median_scale: float
    scale_spread: float
    median_rmse: float
    shift_matters_fraction: float

    def as_dict(self) -> dict:
        return {
            "frames_total": self.frames_total,
            "frames_aligned": self.frames_aligned,
            "points_before_dedup": self.points_before_dedup,
            "points_after_dedup": self.points_after_dedup,
            "median_scale": round(self.median_scale, 6),
            "scale_spread": round(self.scale_spread, 4),
            "median_rmse": round(self.median_rmse, 5),
            "shift_matters_fraction": round(self.shift_matters_fraction, 3),
        }


def build_hybrid_seed(
    frames: list[dict],
    sparse_points: np.ndarray,
    *,
    voxel: float = 0.0,
    stride: int = 2,
    keep_quantile: float = 0.6,
) -> tuple[np.ndarray, np.ndarray, HybridSeedReport, list[DepthAlignment]]:
    """Fuse per-frame predicted depth into one cloud in COLMAP's world.

    Each entry of ``frames`` is ``{"depth", "intrinsics", "c2w", ...}`` with
    optional ``"colors"``, ``"confidence"`` and ``"sparse_points"`` -- the last
    being the points COLMAP verified in THAT frame, which is what makes the
    alignment meaningful. ``sparse_points`` (the argument) is the fallback for
    frames that do not carry their own, and is materially worse. ``intrinsics`` is the 3x3 in the
    DEPTH map's resolution, which is not the training resolution -- DA3 works
    small and splatfacto loads the originals.

    Frames whose alignment does not converge are dropped rather than fused with
    a guessed scale. A frame placed at the wrong depth does not merely add
    nothing; it adds a whole surface in the wrong place, and splatfacto will
    dutifully fit gaussians to it.
    """
    alignments: list[DepthAlignment] = []
    clouds: list[np.ndarray] = []
    colour_chunks: list[np.ndarray] = []

    for frame in frames:
        # Per-frame visibility when the caller has COLMAP's tracks; the whole
        # cloud only as a fallback, which is known to be far worse -- see the
        # module docstring.
        visible = frame.get("sparse_points")
        alignment = align_depth_to_sparse(
            frame["depth"],
            frame["intrinsics"],
            frame["c2w"],
            sparse_points if visible is None else visible,
        )
        alignments.append(alignment)
        if not alignment.ok:
            continue
        world, rgb = backproject(
            alignment.apply(frame["depth"]),
            frame["intrinsics"],
            frame["c2w"],
            colors=frame.get("colors"),
            confidence=frame.get("confidence"),
            keep_quantile=keep_quantile,
            stride=stride,
        )
        clouds.append(world)
        colour_chunks.append(rgb)

    if clouds:
        points = np.concatenate(clouds)
        colors = np.concatenate(colour_chunks)
    else:
        points = np.zeros((0, 3), dtype=np.float64)
        colors = np.zeros((0, 3), dtype=np.uint8)

    before = len(points)
    points, colors = voxel_downsample(points, colors, voxel)

    good = [a for a in alignments if a.ok]
    scales = np.array([a.scale for a in good]) if good else np.array([1.0])
    return (
        points,
        colors,
        HybridSeedReport(
            frames_total=len(frames),
            frames_aligned=len(good),
            points_before_dedup=before,
            points_after_dedup=len(points),
            median_scale=float(np.median(scales)),
            # Spread of the per-frame scale is the headline diagnostic: if DA3's
            # depth were consistently metric against COLMAP this would be ~0.
            scale_spread=float(scales.max() - scales.min()) / float(np.median(scales)),
            median_rmse=float(np.median([a.rmse for a in good])) if good else float("inf"),
            shift_matters_fraction=(
                float(np.mean([a.shift_matters for a in good])) if good else 0.0
            ),
        ),
        alignments,
    )
