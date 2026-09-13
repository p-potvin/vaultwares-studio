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


def get_preset(key: str) -> Preset:
    return PRESETS.get(key, PRESETS["high"])


def validate_capture(duration_seconds: float, frame_count: int, preset: Preset) -> list[str]:
    errors: list[str] = []
    if duration_seconds <= 0:
        errors.append("Video duration must be positive.")
    if frame_count < preset.chunk_size:
        errors.append(f"{preset.label_en} needs at least {preset.chunk_size} selected frames.")
    if frame_count > 500:
        errors.append("This console accepts at most 500 selected frames per request.")
    if preset.overlap >= preset.chunk_size:
        errors.append("Overlap must be smaller than the chunk size.")
    return errors


def selected_frame_indices(candidate_count: int, keep_count: int) -> list[int]:
    """One candidate per time bucket, including both sequence endpoints."""
    if candidate_count <= 0 or keep_count <= 0:
        return []
    if candidate_count <= keep_count:
        return list(range(candidate_count))
    chosen = [min(candidate_count - 1, i * candidate_count // keep_count) for i in range(keep_count)]
    chosen[0], chosen[-1] = 0, candidate_count - 1
    return chosen
