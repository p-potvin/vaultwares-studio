from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class CameraShot:
    name: str
    description: str
    position: tuple[float, float, float]
    target: tuple[float, float, float]
    source: str

    def to_dict(self) -> dict:
        return asdict(self)


def _preset_shots() -> list[CameraShot]:
    return [
        CameraShot(
            name="Entrance View",
            description="Safe establishing shot from the doorway aimed into the room.",
            position=(0.0, 1.6, 4.8),
            target=(0.0, 1.2, 0.0),
            source="preset",
        ),
        CameraShot(
            name="Workbench Orbit",
            description="Three-quarter view that circles the main subject area.",
            position=(2.6, 1.7, 2.8),
            target=(0.0, 1.0, 0.0),
            source="preset",
        ),
        CameraShot(
            name="Overhead Sweep",
            description="High angle shot for spatial understanding and layout review.",
            position=(0.0, 6.5, 0.4),
            target=(0.0, 0.9, 0.0),
            source="preset",
        ),
    ]


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _prompt_shots(prompt: str) -> list[CameraShot]:
    normalized = _normalize(prompt)
    shots: list[CameraShot] = []

    if "door" in normalized or "entrance" in normalized:
        shots.append(
            CameraShot(
                name="Doorway Start",
                description="Prompt-derived doorway framing.",
                position=(-0.4, 1.6, 5.1),
                target=(0.0, 1.0, 0.0),
                source="prompt",
            )
        )

    if "desk" in normalized or "table" in normalized or "workbench" in normalized:
        shots.append(
            CameraShot(
                name="Desk Focus",
                description="Prompt-derived focal view toward the main work surface.",
                position=(1.8, 1.45, 2.0),
                target=(0.2, 1.1, -0.2),
                source="prompt",
            )
        )

    if "orbit" in normalized:
        side_bias = -2.8 if "left" in normalized else 2.8
        shots.append(
            CameraShot(
                name="Orbit Move",
                description="Prompt-derived lateral orbit shot around the subject.",
                position=(side_bias, 1.8, 1.8),
                target=(0.0, 1.1, 0.0),
                source="prompt",
            )
        )

    if "rise" in normalized or "up" in normalized or "overhead" in normalized:
        shots.append(
            CameraShot(
                name="Rise Shot",
                description="Prompt-derived rising finish shot for overview and reveal.",
                position=(0.0, 5.8, 1.2),
                target=(0.0, 1.1, 0.0),
                source="prompt",
            )
        )

    if not shots:
        shots.append(
            CameraShot(
                name="Prompt Hero",
                description="General prompt-driven hero shot using the requested scene focus.",
                position=(2.2, 1.7, 3.0),
                target=(0.0, 1.1, 0.0),
                source="prompt",
            )
        )

    return shots


def _dedupe(shots: Iterable[CameraShot]) -> list[CameraShot]:
    deduped: list[CameraShot] = []
    seen: set[tuple[str, tuple[float, float, float]]] = set()
    for shot in shots:
        marker = (shot.name, shot.position)
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(shot)
    return deduped


def build_camera_bundle(prompt: str) -> dict[str, object]:
    presets = _preset_shots()
    prompt_shots = _prompt_shots(prompt)
    all_shots = _dedupe([*presets, *prompt_shots])
    return {
        "prompt": prompt,
        "presets": [shot.to_dict() for shot in presets],
        "promptPlan": [shot.to_dict() for shot in prompt_shots],
        "allShots": [shot.to_dict() for shot in all_shots],
    }
