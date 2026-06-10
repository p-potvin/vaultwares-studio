"""Robot Lab backend contract.

NavSimBackend abstracts the navigation simulator so the grid-nav backend
(local, CPU, free) and the Habitat backend (remote Linux worker, M3 stretch)
are interchangeable. EpisodeRecord is the replay/eval currency: a pose
trajectory plus PointNav metrics, renderable in the splat viewport.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field


@dataclass
class Episode:
    start: list[float]  # (x, z) world coordinates
    goal: list[float]
    start_heading: float = 0.0  # radians


@dataclass
class EpisodeRecord:
    episode: Episode
    trajectory: list[list[float]] = field(default_factory=list)  # (x, z, heading) per step
    success: bool = False
    spl: float = 0.0  # Success weighted by Path Length
    steps: int = 0

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["episode"] = asdict(self.episode)
        return payload


class NavSimBackend(ABC):
    """Steppable point-robot navigation simulator over a reconstructed scene."""

    name: str = "base"

    @abstractmethod
    def sample_episodes(self, count: int, seed: int | None = None) -> list[Episode]:
        """Random reachable start/goal pairs in free space."""

    @abstractmethod
    def reset(self, episode: Episode) -> dict:
        """Begin an episode; returns the first observation."""

    @abstractmethod
    def step(self, action: int) -> tuple[dict, float, bool, dict]:
        """Advance one action; returns (observation, reward, done, info)."""

    @abstractmethod
    def agent_pose(self) -> tuple[float, float, float]:
        """Current (x, z, heading)."""
