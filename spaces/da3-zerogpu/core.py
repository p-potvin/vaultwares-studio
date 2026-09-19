"""Pure configuration and validation helpers for the ZeroGPU DA3 console."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Preset:
    key: str
    label_en: str
    label_qc: str
    width: int
    height: int
    chunk_size: int
    overlap: int
    gpu_duration_seconds: int

    @property
    def tokens_per_window(self) -> int:
        return (self.width // 14) * (self.height // 14) * self.chunk_size


PRESETS = {
    "preview": Preset("preview", "Preview", "Aperçu", 504, 280, 60, 30, 600),
    "high": Preset("high", "High quality (experimental)", "Haute qualité (expérimental)", 672, 378, 90, 45, 1800),
}


# Frames kept per request. 500 was the original cap and stays the default so
# earlier runs remain reproducible; the ceiling exists because streaming
# memory is per chunk, not per sequence, so the sequence length only costs
# GPU seconds (~0.2 s per frame measured at 500) and artifact size.
DEFAULT_KEEP_FRAMES = 500
MAX_KEEP_FRAMES = 2000
# SALAD loop-detection similarity. 0.85 found only near-neighbour pairs on a
# walk that visibly closed; exposed so a run can try lower without a rebuild.
DEFAULT_LOOP_SIMILARITY = 0.85


def get_preset(key: str) -> Preset:
    return PRESETS.get(key, PRESETS["high"])


def validate_capture(duration_seconds: float, frame_count: int, preset: Preset, max_frames: int = MAX_KEEP_FRAMES) -> list[str]:
    errors: list[str] = []
    if duration_seconds <= 0:
        errors.append("Video duration must be positive.")
    if frame_count < preset.chunk_size:
        errors.append(f"{preset.label_en} needs at least {preset.chunk_size} selected frames.")
    if frame_count > max_frames:
        errors.append(f"This console accepts at most {max_frames} selected frames per request.")
    if preset.overlap >= preset.chunk_size:
        errors.append("Overlap must be smaller than the chunk size.")
    return errors


def candidate_fps(duration_seconds: float, keep_frames: int = DEFAULT_KEEP_FRAMES) -> int:
    """ffmpeg sampling rate so the selector has at least ~2 candidates per kept
    frame. The original rule, ``round(1000 / duration)``, is unchanged for the
    default 500 and is what every run before 17 Sep used; a longer request on
    a long video would otherwise run out of candidates at 2 fps."""
    if duration_seconds <= 0:
        return 2
    return max(2, min(10, round(max(1000, 2 * keep_frames) / duration_seconds)))


def selected_frame_indices(candidate_count: int, keep_count: int) -> list[int]:
    """One candidate per time bucket, including both sequence endpoints."""
    if candidate_count <= 0 or keep_count <= 0:
        return []
    if candidate_count <= keep_count:
        return list(range(candidate_count))
    chosen = [min(candidate_count - 1, i * candidate_count // keep_count) for i in range(keep_count)]
    chosen[0], chosen[-1] = 0, candidate_count - 1
    return chosen
