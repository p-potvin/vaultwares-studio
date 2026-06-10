import numpy as np
from PIL import Image, ImageFilter

from vaultwares_studio.frame_selection import (
    laplacian_sharpness,
    prune_to_sharpest,
    select_sharpest_frames,
)


def _checkerboard(size: int = 256, cells: int = 16) -> Image.Image:
    tile = size // cells
    board = np.indices((size, size)).sum(axis=0) // tile % 2 * 255
    return Image.fromarray(board.astype(np.uint8)).convert("RGB")


def _write_frames(tmp_path, count: int, blur_every: int = 2):
    """Alternate sharp and heavily blurred checkerboard frames."""
    paths = []
    for index in range(count):
        image = _checkerboard()
        if index % blur_every == 1:
            image = image.filter(ImageFilter.GaussianBlur(radius=8))
        path = tmp_path / f"frame_{index:05d}.jpg"
        image.save(path, quality=90)
        paths.append(path)
    return paths


def test_sharpness_orders_blur(tmp_path):
    sharp, blurred = _write_frames(tmp_path, 2)
    assert laplacian_sharpness(sharp) > laplacian_sharpness(blurred) * 5


def test_select_sharpest_keeps_coverage_and_sharpness(tmp_path):
    frames = _write_frames(tmp_path, 20)
    kept = select_sharpest_frames(frames, keep_count=10)
    assert len(kept) == 10
    # Every kept frame should be one of the sharp (even-index) ones.
    assert all(int(path.stem.split("_")[1]) % 2 == 0 for path in kept)
    # Order/coverage preserved: kept indices are increasing.
    indices = [int(path.stem.split("_")[1]) for path in kept]
    assert indices == sorted(indices)


def test_select_noop_when_under_target(tmp_path):
    frames = _write_frames(tmp_path, 5)
    assert select_sharpest_frames(frames, keep_count=10) == frames


def test_prune_deletes_rejected_files(tmp_path):
    _write_frames(tmp_path, 20)
    before, after = prune_to_sharpest(tmp_path, keep_count=10)
    assert (before, after) == (20, 10)
    assert len(list(tmp_path.glob("*.jpg"))) == 10
