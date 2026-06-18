"""Reconstruction quality presets.

Each preset picks splatfacto training parameters AND the HF Jobs flavor that
fits them, plus a cost estimate for the consent dialog. LOCAL_DEBUG preserves
the historical 250-iteration quick path used when reconstruction runs on the
local machine (heavy local training stays opt-in; the PC hosts the
VaultWares API).

Flag spelling drifts between nerfstudio releases — the worker entrypoint
probes ``ns-train splatfacto --help`` and drops unknown flags rather than
failing the job.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .runners import CostEstimate, estimate_cost


@dataclass(frozen=True)
class QualityPreset:
    key: str
    label: str
    iterations: int
    downscale_factor: int
    gaussian_cap: int
    flavor: str
    est_minutes: float
    extra_train_args: tuple[str, ...] = field(default_factory=tuple)

    def train_args(self) -> list[str]:
        """splatfacto arguments shared by local and remote execution."""
        args = [
            "--max-num-iterations", str(self.iterations),
            "--vis", "none",
            "--viewer.quit-on-train-completion", "True",
            "--pipeline.datamanager.cache-images", "cpu",
        ]
        args.extend(self.extra_train_args)
        return args

    def cost(self) -> CostEstimate:
        return estimate_cost(self.flavor, self.est_minutes)


PRESETS: dict[str, QualityPreset] = {
    "draft": QualityPreset(
        key="draft",
        label="Draft (fast, low cost)",
        iterations=7_000,
        downscale_factor=4,
        gaussian_cap=300_000,
        flavor="l4x1",
        est_minutes=15,
        extra_train_args=("--pipeline.model.stop-split-at", "5000"),
    ),
    "standard": QualityPreset(
        key="standard",
        label="Standard",
        iterations=15_000,
        # No training downscale — splatfacto sees the full-res images so the
        # output gaussians keep the detail SfM already extracted at full res.
        # Yes, this 4xes training pixels vs the old 2x downscale; the runtime
        # bump (35 -> 60 min) is worth a sharper final splat for any demo.
        downscale_factor=1,
        gaussian_cap=500_000,
        flavor="l4x1",
        est_minutes=60,
        extra_train_args=("--pipeline.model.cull-alpha-thresh", "0.05"),
    ),
    # Refine an existing splat: launcher passes --refine-from, worker resumes
    # from the base model.zip checkpoint via ns-train --load-dir, and we only
    # need enough additional iterations for the new viewpoints to settle into
    # the existing Gaussian state. 5k is the sweet spot per nerfstudio
    # community guidance for fine-tune runs; bump to 8k if the refine ends
    # up under-fit on new views.
    #
    # est_minutes was 25 at first — measured ~120 min on the first real run
    # (1500-frame joint set: ~90 min for feature_extractor + vocab_tree cross-
    # matching + image_registrator, ~15 min for 5k splatfacto resume iters,
    # plus IO around the 1 GB base bundle pull). Image-registrator scales
    # with N, vocab matching scales O(N * K=100). For smaller refines (few
    # hundred new frames) it'll come in well under that.
    "refine": QualityPreset(
        key="refine",
        label="Refine (resume from base checkpoint)",
        iterations=5_000,
        downscale_factor=1,
        gaussian_cap=500_000,
        flavor="l4x1",
        est_minutes=120,
        extra_train_args=("--pipeline.model.cull-alpha-thresh", "0.05"),
    ),
    "high": QualityPreset(
        key="high",
        label="High (slow, best quality)",
        iterations=30_000,
        downscale_factor=2,
        gaussian_cap=1_500_000,
        flavor="a10g-large",
        est_minutes=75,
        extra_train_args=("--pipeline.model.rasterize-mode", "antialiased"),
    ),
    # Historical local quick path: a smoke-level run that proves the toolchain
    # without tying up the local GPU. Used when no remote runner is configured.
    "local-debug": QualityPreset(
        key="local-debug",
        label="Local debug (250 iterations)",
        iterations=250,
        downscale_factor=4,
        gaussian_cap=100_000,
        flavor="cpu-basic",  # unused locally; kept for completeness
        est_minutes=0,
    ),
}

DEFAULT_PRESET_KEY = "standard"


def get_preset(key: str | None) -> QualityPreset:
    return PRESETS.get((key or "").lower(), PRESETS[DEFAULT_PRESET_KEY])
