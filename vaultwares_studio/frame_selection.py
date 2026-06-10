"""Blur-aware frame selection for reconstruction.

Handheld walkthrough footage carries motion blur; feeding blurred frames to
COLMAP costs registrations. Strategy: extract at ~2x the wanted density,
score sharpness (variance of Laplacian), then keep the sharpest frame per
time bucket — preserving temporal coverage while dropping the smeared ones.
"""

from __future__ import annotations

from pathlib import Path


def laplacian_sharpness(path: Path) -> float:
    """Variance of the Laplacian on a downscaled grayscale image."""
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return 0.0
    height, width = image.shape[:2]
    if width > 480:
        scale = 480 / width
        image = cv2.resize(image, (480, int(height * scale)))
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def select_sharpest_frames(frames: list[Path], keep_count: int) -> list[Path]:
    """Keep the sharpest frame per time bucket; input order defines time."""
    if keep_count <= 0 or len(frames) <= keep_count:
        return list(frames)
    scored = [(laplacian_sharpness(path), path) for path in frames]
    kept: list[Path] = []
    total = len(scored)
    for bucket in range(keep_count):
        start = bucket * total // keep_count
        end = max(start + 1, (bucket + 1) * total // keep_count)
        kept.append(max(scored[start:end], key=lambda item: item[0])[1])
    return kept


def prune_to_sharpest(frames_dir: Path, keep_count: int, patterns: tuple[str, ...] = ("*.jpg", "*.png")) -> tuple[int, int]:
    """Delete all but the sharpest-per-bucket frames. Returns (before, after)."""
    frames = sorted(path for pattern in patterns for path in frames_dir.glob(pattern))
    keep = set(select_sharpest_frames(frames, keep_count))
    for path in frames:
        if path not in keep:
            path.unlink(missing_ok=True)
    return len(frames), len(keep)
